"""Tests for controlled graph, spike, and workload benchmark data."""

from __future__ import annotations

import pytest
import torch

from benchmark.benchmark_data import (
    GraphSpec,
    SpikeSpec,
    generate_graph,
    generate_spike_trace,
    load_operator_workload,
    prepare_operator_workload,
    save_workload,
)
from benchmark.benchmark_persistent_snn import BenchCase
from benchmark.benchmark_rsnn_cudagraph_compare import (
    PROVIDER_CAPABILITIES,
    ProviderCapability,
    calibrate_input_amplitude,
    prepare_physical_case,
)
from benchmark.benchmark_workload_stats import (
    stats_are_finite,
    summarize_spike_workload,
)
from btorch.sparse import CSR


@pytest.mark.parametrize(
    "kind",
    ["uniform", "powerlaw", "hotspot", "clustered"],
)
def test_synthetic_graph_families_are_reproducible(kind):
    """Every graph family should produce a stable source-oriented CSR graph."""

    spec = GraphSpec(
        kind=kind,
        n_neuron=128,
        fanout=8,
        seed=123,
        max_fanout=32,
        hotspot_count=4,
        community_count=4,
    )

    first, first_stats = generate_graph(spec, torch.device("cpu"))
    second, second_stats = generate_graph(spec, torch.device("cpu"))

    assert first.shape == (128, 128)
    torch.testing.assert_close(first.indptr, second.indptr)
    torch.testing.assert_close(first.indices, second.indices)
    torch.testing.assert_close(first.data, second.data)
    assert first_stats == second_stats
    assert first_stats.graph_type == kind
    assert first_stats.nnz == first.indices.numel()
    assert first_stats.max_fanout >= first_stats.mean_fanout


@pytest.mark.parametrize(
    "kind",
    ["iid", "fanout_correlated", "temporal", "cluster_burst"],
)
def test_spike_trace_families_report_measured_work(kind):
    """Spike generation should return measured activity and active-edge cost."""

    matrix, _ = generate_graph(
        GraphSpec(n_neuron=256, fanout=8, seed=10),
        torch.device("cpu"),
    )
    spec = SpikeSpec(
        kind=kind,
        t_steps=32,
        activity=0.05,
        seed=11,
        community_count=4,
    )

    spikes, stats = generate_spike_trace(matrix, spec, torch.device("cpu"))

    assert spikes.shape == (32, 1, 256)
    assert set(torch.unique(spikes).tolist()).issubset({0.0, 1.0})
    assert stats_are_finite(stats)
    assert stats.requested_activity == pytest.approx(0.05)
    assert stats.measured_activity == pytest.approx(float(spikes.mean()))
    assert stats.mean_active_edges >= 0.0
    assert 0.0 <= stats.mean_collision_ratio <= 1.0


def test_active_edge_and_collision_statistics_are_exact_on_small_graph():
    """A hand-written trace should expose exact fanout and post collisions."""

    # Neuron 0 targets posts 0 and 1; neuron 1 targets posts 1 and 2. When
    # both fire there are four active edges but only three unique posts.
    matrix = CSR.from_edges(
        row=torch.tensor([0, 0, 1, 1]),
        col=torch.tensor([0, 1, 1, 2]),
        data=torch.ones(4),
        shape=(3, 3),
    )
    spikes = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0]],
        ]
    )

    stats = summarize_spike_workload(
        matrix,
        spikes,
        requested_activity=0.5,
    )

    assert stats.mean_active_neurons == pytest.approx(1.5)
    assert stats.mean_active_edges == pytest.approx(3.0)
    assert stats.max_active_edges == 4
    assert stats.mean_collision_ratio == pytest.approx(0.125)
    assert stats.collision_method == "exact"


def test_workload_manifest_round_trip_preserves_id_and_tensors(tmp_path):
    """A saved workload should restore the exact fixed trace and graph."""

    workload = prepare_operator_workload(
        GraphSpec(n_neuron=64, fanout=4, seed=20),
        SpikeSpec(t_steps=8, activity=0.1, seed=21),
        torch.device("cpu"),
    )
    path = tmp_path / "workload.pt"

    save_workload(workload, path)
    restored = load_operator_workload(path, torch.device("cpu"))

    assert restored.workload_id == workload.workload_id
    assert restored.manifest() == workload.manifest()
    torch.testing.assert_close(restored.matrix.indptr, workload.matrix.indptr)
    torch.testing.assert_close(restored.matrix.indices, workload.matrix.indices)
    torch.testing.assert_close(restored.matrix.data, workload.matrix.data)
    torch.testing.assert_close(restored.spike_trace, workload.spike_trace)
    assert path.with_suffix(".pt.json").is_file()


def test_provider_prepare_applies_neuron_padding_without_batch_padding(monkeypatch):
    """Physical provider preparation should pad N while retaining logical B=1."""

    provider = "sputnik_cudagraph"
    monkeypatch.setitem(
        PROVIDER_CAPABILITIES,
        provider,
        ProviderCapability(n_alignment=4),
    )
    matrix = CSR.from_edges(
        row=torch.tensor([0, 2]),
        col=torch.tensor([1, 0]),
        data=torch.ones(2),
        shape=(3, 3),
    )
    case = BenchCase(
        n_neuron=3,
        batch_size=1,
        t_steps=2,
        fanout=1,
        event_rate=0.1,
    )
    x_seq = torch.ones(2, 1, 3)

    physical_matrix, physical_x, physical_case, padding = (
        prepare_physical_case(provider, matrix, x_seq, case)
    )

    assert physical_matrix.shape == (4, 4)
    assert physical_x.shape == (2, 1, 4)
    torch.testing.assert_close(physical_x[..., :3], x_seq)
    torch.testing.assert_close(physical_x[..., 3], torch.zeros(2, 1))
    assert physical_case.batch_size == 1
    assert physical_case.n_neuron == 4
    assert padding.padding_ratio == pytest.approx(4 / 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_closed_loop_calibration_reports_best_observed_activity():
    """Calibration should retain bounded amplitude and an auditable status."""

    device = torch.device("cuda")
    case = BenchCase(
        n_neuron=128,
        batch_size=1,
        t_steps=16,
        fanout=4,
        event_rate=0.05,
    )
    matrix, _ = generate_graph(
        GraphSpec(n_neuron=128, fanout=4, seed=30),
        device,
    )

    result = calibrate_input_amplitude(
        case,
        matrix,
        device,
        target_activity=0.01,
        min_amplitude=0.0,
        max_amplitude=50.0,
        max_iterations=5,
    )

    assert 0.0 <= result.input_amplitude <= 50.0
    assert 0.0 <= result.measured_activity <= 1.0
    assert result.activity_error == pytest.approx(
        abs(result.measured_activity - result.target_activity)
    )
    assert result.calibration_status in {
        "converged",
        "best_effort",
        "nonmonotonic_best_effort",
        "unstable",
    }
