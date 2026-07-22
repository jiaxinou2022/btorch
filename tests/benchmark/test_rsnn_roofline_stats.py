"""CPU tests for the optional BlockTask statistics report."""

from types import SimpleNamespace

import torch

import benchmark.benchmark_rsnn_roofline as roofline
from benchmark.benchmark_rsnn_roofline import summarize_block_stats


def test_block_stats_summary_derives_requested_metrics():
    indptr = torch.tensor([0, 2, 4, 4, 5] + [5] * 28, dtype=torch.int32)
    indices = torch.tensor([1, 2, 2, 3, 4], dtype=torch.int32)
    raw = torch.zeros((2, 13), dtype=torch.int32)
    raw[0] = torch.tensor(
        [2, 4, 1, 4, 2, 4, 1, 1, 0, 0, 0, 0, 0b11]
    )
    raw[1, 8] = 2

    summary = summarize_block_stats(raw, indptr, indices)

    assert summary["block_task_count"] == 1
    assert summary["active_rows"] == 2
    assert summary["active_edges"] == 4
    assert summary["active_run_count"] == 1
    assert summary["unique_posts"] == 3
    assert summary["hash_insert_count"] == 0
    assert summary["hash_merge_count"] == 0
    assert summary["hash_overflow_count"] == 0
    assert summary["long_segment_tasks"] == 2
    assert summary["average_spikes_per_task"] == 2.0
    assert summary["average_runs_per_task"] == 1.0
    assert summary["adjacent_active_row_ratio"] == 0.5
    assert summary["run_merge_ratio"] == 2.0
    assert summary["span_utilization"] == 1.0
    assert summary["post_duplicate_ratio"] == 0.25
    assert summary["atomic_reduction_ratio"] == 0.0
    assert summary["hash_task_ratio"] == 0.0


def test_block_stats_models_the_selective_hash_path():
    indptr = torch.tensor([0, 32, 64] + [64] * 30, dtype=torch.int32)
    indices = torch.tensor(list(range(32)) * 2, dtype=torch.int32)
    raw = torch.zeros((1, 13), dtype=torch.int32)
    raw[0] = torch.tensor(
        [2, 64, 1, 64, 32, 64, 0, 1, 0, 0, 0, 0, 0b11]
    )

    summary = summarize_block_stats(raw, indptr, indices)

    assert summary["hash_tasks"] == 1
    assert summary["hash_beneficial_tasks"] == 1
    assert summary["hash_insert_count"] == 32
    assert summary["hash_merge_count"] == 32
    assert summary["flushed_entries"] == 32
    assert summary["atomic_reduction_ratio"] == 0.5


def test_empty_block_stats_have_zero_ratios():
    summary = summarize_block_stats(
        torch.zeros((1, 13), dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
    )

    assert summary["block_task_count"] == 0
    assert summary["post_duplicate_ratio"] == 0.0
    assert summary["atomic_reduction_ratio"] == 0.0
    assert summary["hash_task_ratio"] == 0.0


def test_cusparse_cudagraph_delegates_to_shared_direct_provider(monkeypatch):
    """Roofline graph execution should reuse the comparison implementation."""

    calls = {}
    expected = SimpleNamespace(
        spikes=torch.tensor([1.0]),
        v=torch.tensor([2.0]),
        psc=torch.tensor([3.0]),
    )

    class FakeDirectProvider:
        def fixed_runner(self, x_seq, weight, case, *, use_cudagraph):
            calls.update(
                x_seq=x_seq,
                weight=weight,
                case=case,
                use_cudagraph=use_cudagraph,
            )
            return lambda: expected

    matrix = object()
    x_seq = object()
    case = object()
    converted_weight = object()
    workload = SimpleNamespace(matrix=matrix, x_seq=x_seq, case=case)
    monkeypatch.setattr(roofline, "DirectCuSparseProvider", FakeDirectProvider)
    monkeypatch.setattr(
        roofline,
        "make_torch_csr_weight",
        lambda candidate: converted_weight if candidate is matrix else None,
    )

    run = roofline.make_provider_runner(
        workload,
        provider="cusparse_cudagraph",
        fanout_binning=False,
        spike_block=False,
    )

    assert run() == (expected.spikes, expected.v, expected.psc)
    assert calls == {
        "x_seq": x_seq,
        "weight": converted_weight,
        "case": case,
        "use_cudagraph": True,
    }
