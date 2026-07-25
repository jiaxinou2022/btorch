"""Tests for the standard PyTorch providers in the RSNN comparison."""

from __future__ import annotations

import sys

import pytest
import scipy.sparse as sp
import torch

import benchmark.benchmark_rsnn_cudagraph_compare as comparison
import benchmark.benchmark_rsnn_roofline as roofline
from benchmark.benchmark_persistent_snn import (
    BenchCase,
    RSNNResult,
    make_input_sequence,
    make_recurrent_csr,
)
from benchmark.benchmark_rsnn_cudagraph_compare import (
    FLYBRAIN_DEFAULT_PROVIDERS,
    PROVIDERS,
    DirectCuSparseProvider,
    correctness_metrics,
    load_flybrain_csr,
    make_eager_runner,
    make_torch_csr_weight,
    resolve_dataset_defaults,
)
from btorch.sparse import CSR


def test_torch_csr_weight_uses_post_by_pre_orientation():
    """The public PyTorch weight should map source columns to target rows."""

    # btorch stores the edge 0 -> 2 in source-oriented CSR row 0. PyTorch
    # computes W @ z, so the converted weight must contain W[2, 0], not
    # W[0, 2]. This small asymmetric graph catches an accidental transpose.
    matrix = CSR.from_edges(
        row=torch.tensor([0]),
        col=torch.tensor([2]),
        data=torch.tensor([0.5]),
        shape=(3, 3),
    )

    weight = make_torch_csr_weight(matrix)

    assert weight.layout == torch.sparse_csr
    torch.testing.assert_close(
        weight.to_dense(),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
    )


def test_standard_dense_and_csr_rsnn_baselines_are_equivalent():
    """Dense linear and CSR sparse.mm should evaluate identical dynamics."""

    # Use multiple batches and timesteps so this checks both the B x N
    # transpose around sparse.mm and recurrent state propagation over time.
    case = BenchCase(
        n_neuron=32,
        batch_size=3,
        t_steps=8,
        fanout=5,
        event_rate=0.1,
    )
    device = torch.device("cpu")
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    csr_weight = make_torch_csr_weight(matrix)

    dense = make_eager_runner(
        x_seq, csr_weight.to_dense(), case, sparse=False
    )()
    sparse = make_eager_runner(x_seq, csr_weight, case, sparse=True)()

    torch.testing.assert_close(dense.spikes, sparse.spikes, atol=0, rtol=0)
    torch.testing.assert_close(dense.v, sparse.v, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense.psc, sparse.psc, atol=1e-6, rtol=1e-6)


def test_default_providers_are_standard_pytorch_and_persistent_variants():
    """The published default comparison should not include custom scatter."""

    assert PROVIDERS == (
        "torch_dense_eager",
        "torch_dense_cudagraph",
        "torch_csr_eager",
        "torch_csr_cudagraph",
        "cusparse_direct_eager",
        "cusparse_direct_cudagraph",
        "persistent_plain",
        "persistent_binning",
        "persistent_spike_block",
    )


def test_flybrain_defaults_avoid_dense_whole_brain_weights():
    """FlyBrain defaults should remain sparse at whole-connectome scale.

    FlyWire contains enough neurons that a dense adjacency commonly exceeds
    accelerator memory. Keeping both dense providers out of this dataset's
    implicit provider list prevents an unexpected allocation while leaving
    them available through an explicit ``--providers`` argument.
    """

    assert FLYBRAIN_DEFAULT_PROVIDERS
    assert all(
        not provider.startswith("torch_dense_")
        for provider in FLYBRAIN_DEFAULT_PROVIDERS
    )
    assert set(FLYBRAIN_DEFAULT_PROVIDERS).issubset(PROVIDERS)


def test_flybrain_loader_preserves_signed_weights_and_orientation(monkeypatch):
    """FlyBrain loading should scale signed pre-to-post synapse counts.

    The asymmetric edge locations verify that the loader does not transpose
    the source-oriented CSR graph. Positive and negative values verify that
    excitatory and inhibitory FlyWire connections both survive conversion.
    """

    source = sp.csr_matrix(
        (
            [2.0, -3.0],
            ([0, 2], [1, 0]),
        ),
        shape=(3, 3),
        dtype="float32",
    )
    calls = {}

    def fake_load_flywire_783(*, root, use_weights):
        calls.update(root=root, use_weights=use_weights)
        return source.copy()

    monkeypatch.setattr(
        "connectome_dataset.graph_loader.load_flywire_783",
        fake_load_flywire_783,
    )
    matrix = load_flybrain_csr(
        None,
        weight_scale=0.25,
        device=torch.device("cpu"),
    )

    assert calls == {"root": None, "use_weights": True}
    torch.testing.assert_close(
        matrix.to_dense(),
        torch.tensor(
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [-0.75, 0.0, 0.0],
            ]
        ),
    )


def test_flybrain_is_the_default_dataset(monkeypatch):
    """Both command-line benchmarks should select FlyBrain by default."""

    monkeypatch.setattr(sys, "argv", [comparison.__file__])
    comparison_args = comparison.parse_args()
    monkeypatch.setattr(sys, "argv", [roofline.__file__])
    roofline_args = roofline.parse_args()

    assert comparison_args.dataset == "flybrain"
    assert comparison_args.weight_scale == pytest.approx(0.275)
    assert roofline_args.dataset == "flybrain"
    assert roofline_args.weight_scale == pytest.approx(0.275)


def test_dataset_specific_weight_scale_defaults():
    """Dataset defaults should preserve legacy mouse and uniform behavior."""

    assert resolve_dataset_defaults("flybrain", None) == ("flybrain", 0.275)
    assert resolve_dataset_defaults("flywire_783", None) == ("flybrain", 0.275)
    assert resolve_dataset_defaults("mice_column_v1", None) == (
        "mice_column_v1",
        0.15,
    )
    assert resolve_dataset_defaults("uniform", None) == ("uniform", 0.15)
    assert resolve_dataset_defaults("flybrain", 0.5) == ("flybrain", 0.5)


def test_correctness_accepts_small_relative_error_on_large_states():
    """Large FlyWire states should not fail solely due to absolute error."""

    reference = RSNNResult(
        spikes=torch.zeros(2, 1, 2),
        v=torch.tensor([[190_525.0, -80_000.0]]),
        psc=torch.tensor([[66_377.0, -40_000.0]]),
    )
    result = RSNNResult(
        spikes=reference.spikes.clone(),
        v=reference.v + torch.tensor([[0.72, -0.30]]),
        psc=reference.psc + torch.tensor([[0.48, -0.20]]),
    )

    metrics = correctness_metrics(result, reference)

    assert metrics["status"] == "passed"
    assert metrics["v_max_abs_diff"] > 0.2
    assert metrics["psc_max_abs_diff"] > 0.005
    assert metrics["v_max_normalized_error"] < 1.0
    assert metrics["psc_max_normalized_error"] < 1.0


def test_correctness_rejects_material_error_near_zero():
    """Relative tolerance should not hide incorrect low-magnitude state."""

    reference = RSNNResult(
        spikes=torch.zeros(1, 1, 1),
        v=torch.zeros(1, 1),
        psc=torch.zeros(1, 1),
    )
    result = RSNNResult(
        spikes=reference.spikes.clone(),
        v=torch.tensor([[0.21]]),
        psc=torch.tensor([[0.006]]),
    )

    metrics = correctness_metrics(result, reference)

    assert metrics["status"] == "correctness_failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("use_cudagraph", [False, True])
def test_direct_cusparse_matches_dense_baseline(batch_size, use_cudagraph):
    """Direct SpMV/SpMM and its graph capture should match dense PyTorch."""

    # Batch one exercises cuSPARSE SpMV; batch three exercises SpMM. Testing
    # both eager and captured execution also catches descriptors or workspace
    # whose lifetime ends before CUDA graph replay.
    case = BenchCase(
        n_neuron=64,
        batch_size=batch_size,
        t_steps=8,
        fanout=7,
        event_rate=0.1,
    )
    device = torch.device("cuda")
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    csr_weight = make_torch_csr_weight(matrix)
    reference = make_eager_runner(
        x_seq, csr_weight.to_dense(), case, sparse=False
    )()
    run = DirectCuSparseProvider().fixed_runner(
        x_seq,
        csr_weight,
        case,
        use_cudagraph=use_cudagraph,
    )

    result = run()
    torch.cuda.synchronize()

    torch.testing.assert_close(result.spikes, reference.spikes, atol=0, rtol=0)
    torch.testing.assert_close(result.v, reference.v, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.psc, reference.psc, atol=1e-4, rtol=1e-4)
