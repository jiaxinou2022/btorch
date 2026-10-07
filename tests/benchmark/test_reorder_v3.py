"""Tests and small examples for the v3 fanout preprocessing experiment."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmark.reorder_analysis.analyze_fanout_preprocessing_v3 import (
    V3Config,
    _topk_block_signatures,
    build_sketch_order,
    build_topk_order,
    build_union_order,
    load_balance,
    select_finalists,
    validate_order,
)


def _alternating_reuse_graph() -> sp.csr_matrix:
    """Create rows whose best pairs are deliberately separated in identity order."""

    rows = np.repeat(np.arange(64, dtype=np.int32), 2)
    columns = np.empty(128, dtype=np.int32)
    for neuron in range(64):
        # Even and odd rows reuse disjoint post sets. Identity blocks mix both
        # groups, whereas exact union packing can create homogeneous blocks.
        columns[2 * neuron : 2 * neuron + 2] = (0, 1) if neuron % 2 == 0 else (32, 33)
    graph = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(64, 64),
    ).tocsr()
    graph.sort_indices()
    return graph


def _static_unique(graph: sp.csr_matrix, order: np.ndarray) -> int:
    """Count exact unique destinations across final 32-neuron blocks."""

    total = 0
    for start in range(0, order.size, 32):
        destinations = [
            graph.indices[graph.indptr[row] : graph.indptr[row + 1]]
            for row in order[start : start + 32]
        ]
        total += np.unique(np.concatenate(destinations)).size
    return total


def test_topk_signature_keeps_counts_and_deterministic_ties():
    """Top-k signatures retain richer block/count information than dominant."""

    rows = np.zeros(7, dtype=np.int32)
    columns = np.array([0, 1, 32, 33, 64, 65, 66], dtype=np.int32)
    graph = sp.coo_matrix(
        (np.ones(7, dtype=np.float32), (rows, columns)), shape=(4, 96)
    ).tocsr()
    signature = _topk_block_signatures(graph.indptr, graph.indices, block_size=32, k=2)

    # Block 2 wins with three edges. Blocks 0 and 1 tie at two, so the lower
    # block id must occupy the second slot.
    np.testing.assert_array_equal(signature[0], np.array([2, 3, 0, 2]))


def test_all_order_builders_return_valid_deterministic_permutations():
    """Compressed and exact builders preserve every neuron exactly once."""

    graph = _alternating_reuse_graph()
    top4 = build_topk_order(graph, 4)
    sketch_a = build_sketch_order(graph, 128)
    sketch_b = build_sketch_order(graph, 128)
    greedy = build_union_order(
        graph,
        np.arange(64, dtype=np.int32),
        window_size=64,
    )

    for order in (top4, sketch_a, greedy):
        validate_order(order, 64)
    np.testing.assert_array_equal(sketch_a, sketch_b)


def test_union_greedy_reduces_exact_static_destinations():
    """The raw reuse objective should pack the crafted overlap groups together."""

    graph = _alternating_reuse_graph()
    identity = np.arange(64, dtype=np.int32)
    greedy = build_union_order(graph, identity, window_size=64)

    assert _static_unique(graph, identity) == 8
    assert _static_unique(graph, greedy) == 4


def test_weighted_normalized_and_seed_ablations_remain_permutations():
    """Every v3 greedy score and seed policy obeys the permutation contract."""

    graph = _alternating_reuse_graph()
    identity = np.arange(64, dtype=np.int32)
    orders = [
        build_union_order(graph, identity, window_size=64, seed_mode="first"),
        build_union_order(
            graph,
            identity,
            window_size=64,
            seed_mode="max_contention",
            weight_mode="log",
        ),
        build_union_order(graph, identity, window_size=64, alpha=0.5),
        build_union_order(graph, identity, window_size=64, weight_mode="sqrt"),
    ]
    for order in orders:
        validate_order(order, 64)


def test_load_balance_reports_requested_tail_statistics():
    """Load reporting includes the CV and upper-tail values required by v3."""

    graph = _alternating_reuse_graph()
    metrics = load_balance(np.arange(64, dtype=np.int32), graph)

    assert metrics["load_mean"] == 64.0
    assert metrics["load_std"] == 0.0
    assert metrics["load_cv"] == 0.0
    assert metrics["load_p95"] == 64.0
    assert metrics["load_p99"] == 64.0
    assert metrics["load_max"] == 64.0


def test_finalist_selection_handles_multiple_passing_families():
    """Passing candidates are selected by quality without a set-construction error."""

    rows = []
    for method, family, reduction in (
        ("dominant", "dominant", 0.0),
        ("union", "union", 0.04),
        ("weighted", "weighted_union", 0.03),
    ):
        for rate in (0.01, 0.02):
            rows.append(
                {
                    "dataset": "flybrain",
                    "control_input_rate": rate,
                    "method": method,
                    "family": family,
                    "atomic_reduction": reduction,
                    "atomic_reduction_ci95_low": reduction - 0.005,
                    "load_cv_ratio": 1.02,
                    "preprocess_ms": 10.0,
                    "temporary_memory_mb": 2.0,
                }
            )
    config = V3Config(
        datasets=("flybrain",),
        target_rates_hz=(10.0,),
        calibration_input_rates=(0.01,),
        calibration_steps=1,
        phase1_steps=1,
        kernel_steps=1,
        synthetic_neurons=64,
        synthetic_degree=2,
        recurrent_strength=0.1,
        greedy_window=64,
        sketch_window=64,
        task_sample=1,
        kernel_warmup=1,
        kernel_repeats=1,
        phase1_atomic_threshold=0.01,
        load_cv_tolerance=0.15,
        seed=1,
    )

    selected = select_finalists(pd.DataFrame(rows), config)

    chosen = selected[selected["selected_for_phase2"]]
    assert set(chosen["method"]) == {"union", "weighted"}
    assert set(chosen["selection_reason"]) == {"threshold_pass"}
