"""Screen fanout packers and benchmark finalists on the persistent hash kernel.

This implements ``preprocess_plan_v3.md`` as a numerical two-phase experiment.
It never creates figures. Phase 1 compares structural quality, realized reuse,
load balance, and preprocessing cost. Phase 2 physically reorders FlyWire and
times the real persistent BlockTask shared-hash kernel on an RTX-class GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from benchmark.reorder_analysis.analyze_reorder_refinement_v2 import (
    BLOCK_SIZE,
    build_fanout_order,
    firing_statistics,
)
from benchmark.reorder_analysis.analyze_reorder_similarity import (
    analyze_layout,
    graph_summary,
    load_graph,
    make_random_events,
    scale_simulation_weights,
    simulate_spikes,
)
from btorch.backend.persistent.reorder import (
    NeuronPermutation,
    ReorderConfig,
    reorder_events,
    reorder_graph,
    reorder_state,
)
from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNState,
    WindowedSpikeEvents,
    make_empty_state,
    make_persistent_snn_workspace,
    persistent_snn_forward,
)
from btorch.sparse import CSR


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_RATES_HZ = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
CALIBRATION_RATES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


@dataclass(frozen=True)
class V3Config:
    """Record all choices that affect the v3 numerical experiment."""

    datasets: tuple[str, ...]
    target_rates_hz: tuple[float, ...]
    calibration_input_rates: tuple[float, ...]
    calibration_steps: int
    phase1_steps: int
    kernel_steps: int
    synthetic_neurons: int
    synthetic_degree: int
    recurrent_strength: float
    greedy_window: int
    sketch_window: int
    task_sample: int
    kernel_warmup: int
    kernel_repeats: int
    phase1_atomic_threshold: float
    load_cv_tolerance: float
    seed: int


@dataclass(frozen=True)
class OrderSpec:
    """Describe one deterministic phase-1 ordering candidate."""

    method: str
    family: str
    builder: Callable[[], np.ndarray]
    complexity: str


@dataclass
class KernelWorkload:
    """Hold one physically reordered real-kernel workload."""

    method: str
    order: np.ndarray
    events: WindowedSpikeEvents
    graph: EventCSRGraph
    state: PersistentSNNState
    params: PersistentSNNParams
    workspace: object
    physical_reorder_ms: float


def _rss_bytes() -> int:
    """Read resident bytes without introducing another dependency."""

    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def measure_builder(
    builder: Callable[[], np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    """Measure wall time and sampled/traced temporary host memory."""

    baseline = _rss_bytes()
    peak_rss = baseline
    stop = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop.wait(0.005):
            peak_rss = max(peak_rss, _rss_bytes())

    sampler = threading.Thread(target=sample_rss, daemon=True)
    tracemalloc.start()
    sampler.start()
    start = time.perf_counter()
    try:
        order = builder()
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stop.set()
        sampler.join()
        _current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    peak_rss = max(peak_rss, _rss_bytes())
    temporary_mb = max(peak_rss - baseline, traced_peak) / 2**20
    return order, elapsed_ms, temporary_mb, order.nbytes / 2**20


@numba.njit(cache=True, parallel=True)
def _topk_block_signatures(
    indptr: np.ndarray,
    indices: np.ndarray,
    block_size: int,
    k: int,
) -> np.ndarray:
    """Return exact ``(block, count)`` top-k signatures for sorted CSR rows."""

    neurons = indptr.size - 1
    output = np.empty((neurons, 2 * k), dtype=np.int64)
    output[:, 0::2] = -1
    output[:, 1::2] = 0
    for neuron in numba.prange(neurons):
        best_blocks = np.full(k, -1, dtype=np.int64)
        best_counts = np.zeros(k, dtype=np.int64)
        edge = indptr[neuron]
        end = indptr[neuron + 1]
        while edge < end:
            block = indices[edge] // block_size
            run_end = edge + 1
            while run_end < end and indices[run_end] // block_size == block:
                run_end += 1
            count = run_end - edge
            position = k
            for slot in range(k):
                if count > best_counts[slot] or (
                    count == best_counts[slot]
                    and (best_blocks[slot] < 0 or block < best_blocks[slot])
                ):
                    position = slot
                    break
            if position < k:
                for slot in range(k - 1, position, -1):
                    best_blocks[slot] = best_blocks[slot - 1]
                    best_counts[slot] = best_counts[slot - 1]
                best_blocks[position] = block
                best_counts[position] = count
            edge = run_end
        for slot in range(k):
            output[neuron, 2 * slot] = best_blocks[slot]
            output[neuron, 2 * slot + 1] = best_counts[slot]
    return output


def build_topk_order(topology: sp.csr_matrix, k: int) -> np.ndarray:
    """Build the v3 top-k post-block lexicographic order."""

    signatures = _topk_block_signatures(
        topology.indptr, topology.indices, BLOCK_SIZE, k
    )
    neurons = topology.shape[0]
    ids = np.arange(neurons, dtype=np.int64)
    degree = np.diff(topology.indptr)
    bucket = np.searchsorted(
        np.array([4, 8, 16, 32, 64, 128, 255]), degree, side="left"
    )
    keys = [ids, bucket]
    keys.extend(signatures[:, column] for column in range(2 * k - 1, -1, -1))
    return np.lexsort(tuple(keys)).astype(np.int32)


@numba.njit(cache=True, parallel=True)
def _bit_sketches(indptr: np.ndarray, indices: np.ndarray, bits: int) -> np.ndarray:
    """Build deterministic 64/128-bit Bloom-style fanout sketches."""

    words = bits // 64
    output = np.zeros((indptr.size - 1, words), dtype=np.uint64)
    for neuron in numba.prange(indptr.size - 1):
        for edge in range(indptr[neuron], indptr[neuron + 1]):
            value = np.uint64(indices[edge]) + np.uint64(0x9E3779B97F4A7C15)
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            value ^= value >> np.uint64(31)
            bit = int(value & np.uint64(bits - 1))
            output[neuron, bit // 64] |= np.uint64(1) << np.uint64(bit % 64)
    return output


def build_sketch_order(topology: sp.csr_matrix, bits: int) -> np.ndarray:
    """Sort neurons by a deterministic fixed-length fanout sketch."""

    sketches = _bit_sketches(topology.indptr, topology.indices, bits)
    ids = np.arange(topology.shape[0], dtype=np.int64)
    keys: list[np.ndarray] = [ids]
    keys.extend(sketches[:, word] for word in range(sketches.shape[1]))
    return np.lexsort(tuple(keys)).astype(np.int32)


@numba.njit(cache=True, parallel=True)
def _row_weight_sums(
    indptr: np.ndarray, indices: np.ndarray, post_weight: np.ndarray
) -> np.ndarray:
    output = np.zeros(indptr.size - 1, dtype=np.float64)
    for neuron in numba.prange(indptr.size - 1):
        total = 0.0
        for edge in range(indptr[neuron], indptr[neuron + 1]):
            total += post_weight[indices[edge]]
        output[neuron] = total
    return output


@numba.njit(cache=True)
def _union_greedy_pack(
    base_order: np.ndarray,
    window_size: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    post_weight: np.ndarray,
    alpha: float,
    seed_mode: int,
) -> np.ndarray:
    """Pack exact destination reuse inside bounded candidate windows."""

    output = np.empty_like(base_order)
    marker = np.zeros(indptr.size - 1, dtype=np.int32)
    contention = _row_weight_sums(indptr, indices, post_weight)
    token = 0
    for global_start in range(0, base_order.size, window_size):
        valid = min(window_size, base_order.size - global_start)
        remaining = np.ones(valid, dtype=np.uint8)
        written = 0
        while written < valid:
            token += 1
            seed = -1
            best_seed = -np.inf
            for candidate in range(valid):
                if remaining[candidate] == 0:
                    continue
                neuron = base_order[global_start + candidate]
                degree = indptr[neuron + 1] - indptr[neuron]
                seed_score = -candidate
                if seed_mode == 1:
                    seed_score = degree
                elif seed_mode == 2:
                    seed_score = contention[neuron]
                if seed_score > best_seed:
                    seed = candidate
                    best_seed = seed_score
            neuron = base_order[global_start + seed]
            remaining[seed] = 0
            output[global_start + written] = neuron
            written += 1
            for edge in range(indptr[neuron], indptr[neuron + 1]):
                marker[indices[edge]] = token

            block_length = 1
            while block_length < BLOCK_SIZE and written < valid:
                best = -1
                best_score = -1.0
                best_reuse = -1
                best_degree = -1
                for candidate in range(valid):
                    if remaining[candidate] == 0:
                        continue
                    neuron = base_order[global_start + candidate]
                    degree = indptr[neuron + 1] - indptr[neuron]
                    weighted_reuse = 0.0
                    reuse = 0
                    for edge in range(indptr[neuron], indptr[neuron + 1]):
                        post = indices[edge]
                        if marker[post] == token:
                            reuse += 1
                            weighted_reuse += post_weight[post]
                    denominator = degree**alpha if degree and alpha else 1.0
                    score = weighted_reuse / denominator
                    if (
                        score > best_score
                        or (score == best_score and reuse > best_reuse)
                        or (
                            score == best_score
                            and reuse == best_reuse
                            and degree > best_degree
                        )
                    ):
                        best = candidate
                        best_score = score
                        best_reuse = reuse
                        best_degree = degree
                neuron = base_order[global_start + best]
                remaining[best] = 0
                output[global_start + written] = neuron
                written += 1
                block_length += 1
                for edge in range(indptr[neuron], indptr[neuron + 1]):
                    marker[indices[edge]] = token
    return output


def build_union_order(
    topology: sp.csr_matrix,
    base_order: np.ndarray,
    *,
    window_size: int,
    weight_mode: str = "raw",
    alpha: float = 0.0,
    seed_mode: str = "max_degree",
) -> np.ndarray:
    """Build one bounded raw, normalized, or contention-weighted packer."""

    fanin = np.bincount(topology.indices, minlength=topology.shape[0]).astype(
        np.float64
    )
    if weight_mode == "raw":
        post_weight = np.ones(topology.shape[0], dtype=np.float64)
    elif weight_mode == "log":
        post_weight = np.log1p(fanin)
    elif weight_mode == "sqrt":
        post_weight = np.sqrt(fanin)
    else:
        raise ValueError(f"unknown weight mode: {weight_mode}")
    seed_id = {"first": 0, "max_degree": 1, "max_contention": 2}[seed_mode]
    return _union_greedy_pack(
        base_order,
        window_size,
        topology.indptr,
        topology.indices,
        post_weight,
        alpha,
        seed_id,
    )


def validate_order(order: np.ndarray, neurons: int) -> None:
    """Reject malformed permutations before any physical graph mutation."""

    if order.shape != (neurons,) or order.dtype != np.int32:
        raise ValueError("order must be an int32 vector with one entry per neuron")
    if not np.array_equal(np.sort(order), np.arange(neurons, dtype=np.int32)):
        raise ValueError("order is not a permutation")


def load_balance(order: np.ndarray, topology: sp.csr_matrix) -> dict[str, float]:
    """Compute exact edge workload distribution over 32-neuron blocks."""

    degree = np.diff(topology.indptr)[order].astype(np.float64)
    pad = (-degree.size) % BLOCK_SIZE
    if pad:
        degree = np.pad(degree, (0, pad))
    loads = degree.reshape(-1, BLOCK_SIZE).sum(axis=1)
    mean = float(loads.mean()) if loads.size else 0.0
    return {
        "load_mean": mean,
        "load_std": float(loads.std()),
        "load_cv": float(loads.std() / mean) if mean else 0.0,
        "load_p95": float(np.quantile(loads, 0.95)),
        "load_p99": float(np.quantile(loads, 0.99)),
        "load_max": float(loads.max(initial=0)),
    }


def order_specs(topology: sp.csr_matrix, config: V3Config) -> list[OrderSpec]:
    """Create the v3 refinement ladder and compact greedy ablations."""

    identity = np.arange(topology.shape[0], dtype=np.int32)
    dominant = build_fanout_order(topology)
    return [
        OrderSpec("identity", "identity", lambda: identity.copy(), "O(N)"),
        OrderSpec("dominant", "dominant", lambda: build_fanout_order(topology), "O(E)"),
        OrderSpec("top4", "topk", lambda: build_topk_order(topology, 4), "O(4E)"),
        OrderSpec(
            "sketch128",
            "sketch",
            lambda: build_sketch_order(topology, 128),
            "O(E)",
        ),
        OrderSpec(
            f"sketch_local_union_w{config.sketch_window}",
            "sketch_local_union",
            lambda: build_union_order(
                topology,
                build_sketch_order(topology, 128),
                window_size=config.sketch_window,
            ),
            "O(E + N*W*B*d)",
        ),
        OrderSpec(
            f"union_w{config.greedy_window}_first",
            "union",
            lambda: build_union_order(
                topology,
                dominant,
                window_size=config.greedy_window,
                seed_mode="first",
            ),
            "O(N*W*B*d)",
        ),
        OrderSpec(
            f"union_w{config.greedy_window}",
            "union",
            lambda: build_union_order(
                topology, dominant, window_size=config.greedy_window
            ),
            "O(N*W*B*d)",
        ),
        OrderSpec(
            f"union_w{config.greedy_window}_contention_seed",
            "union",
            lambda: build_union_order(
                topology,
                dominant,
                window_size=config.greedy_window,
                seed_mode="max_contention",
                weight_mode="log",
            ),
            "O(N*W*B*d)",
        ),
        OrderSpec(
            f"normalized_union_a0.5_w{config.greedy_window}",
            "normalized_union",
            lambda: build_union_order(
                topology,
                dominant,
                window_size=config.greedy_window,
                alpha=0.5,
            ),
            "O(N*W*B*d)",
        ),
        OrderSpec(
            f"weighted_union_log_w{config.greedy_window}",
            "weighted_union",
            lambda: build_union_order(
                topology,
                dominant,
                window_size=config.greedy_window,
                weight_mode="log",
            ),
            "O(N*W*B*d)",
        ),
        OrderSpec(
            f"weighted_union_sqrt_w{config.greedy_window}",
            "weighted_union",
            lambda: build_union_order(
                topology,
                dominant,
                window_size=config.greedy_window,
                weight_mode="sqrt",
            ),
            "O(N*W*B*d)",
        ),
    ]


def calibrate(
    dataset: str,
    matrix: sp.csr_matrix,
    config: V3Config,
    dataset_id: int,
    device: torch.device,
) -> tuple[list[float], list[dict[str, float | int | str]]]:
    """Calibrate input controls against posterior population Hz."""

    rows: list[dict[str, float | int | str]] = []
    for condition, control in enumerate(config.calibration_input_rates):
        spikes = simulate_spikes(
            matrix,
            timesteps=config.calibration_steps,
            input_rate=control,
            input_amplitude=1.1,
            seed=config.seed + 10_000 * dataset_id + condition,
            device=device,
        )
        rows.append(
            {
                "dataset": dataset,
                "control_input_rate": control,
                **firing_statistics(spikes),
            }
        )
    selected: list[float] = []
    targets = config.target_rates_hz if dataset == "flybrain" else (10.0,)
    for target in targets:
        row = min(
            rows, key=lambda item: abs(float(item["population_mean_hz"]) - target)
        )
        control = float(row["control_input_rate"])
        if control not in selected:
            selected.append(control)
        previous = str(row.get("selected_targets_hz", ""))
        row["selected_targets_hz"] = f"{previous},{target}".strip(",")
    return selected, rows


def compare_phase1(frame: pd.DataFrame) -> pd.DataFrame:
    """Add baseline-relative quality and load-balance columns."""

    baseline = frame[frame["method"] == "dominant"].set_index(
        ["dataset", "control_input_rate"]
    )
    result = frame.copy()
    keys = pd.MultiIndex.from_frame(result[["dataset", "control_input_rate"]])
    base_atomic = baseline.loc[keys, "estimated_global_atomic_count"].to_numpy()
    candidate = result["estimated_global_atomic_count"].to_numpy()
    base_se = baseline.loc[keys, "estimated_global_atomic_count_se"].to_numpy()
    candidate_se = result["estimated_global_atomic_count_se"].to_numpy()
    ratio = candidate / base_atomic
    ratio_se = ratio * np.sqrt(
        (candidate_se / candidate) ** 2 + (base_se / base_atomic) ** 2
    )
    result["atomic_reduction"] = 1.0 - ratio
    result["atomic_reduction_ci95_low"] = 1.0 - ratio - 1.96 * ratio_se
    result["atomic_reduction_ci95_high"] = 1.0 - ratio + 1.96 * ratio_se
    result["static_aggregation_gain"] = (
        result["static_potential_aggregation_ratio"].to_numpy()
        - baseline.loc[keys, "static_potential_aggregation_ratio"].to_numpy()
    )
    result["realized_aggregation_gain"] = (
        result["weighted_aggregation_ratio"].to_numpy()
        - baseline.loc[keys, "weighted_aggregation_ratio"].to_numpy()
    )
    result["load_cv_ratio"] = (
        result["load_cv"].to_numpy() / baseline.loc[keys, "load_cv"].to_numpy()
    )
    same = result["method"] == "dominant"
    result.loc[
        same,
        [
            "atomic_reduction",
            "atomic_reduction_ci95_low",
            "atomic_reduction_ci95_high",
            "static_aggregation_gain",
            "realized_aggregation_gain",
        ],
    ] = 0.0
    return result


def select_finalists(frame: pd.DataFrame, config: V3Config) -> pd.DataFrame:
    """Apply v3 quality/load gates and retain family-diverse finalists."""

    fly = frame[frame["dataset"] == "flybrain"]
    summary = (
        fly.groupby(["method", "family"], as_index=False)
        .agg(
            median_atomic_reduction=("atomic_reduction", "median"),
            min_atomic_reduction=("atomic_reduction", "min"),
            median_ci95_low=("atomic_reduction_ci95_low", "median"),
            max_load_cv_ratio=("load_cv_ratio", "max"),
            preprocess_ms=("preprocess_ms", "first"),
            temporary_memory_mb=("temporary_memory_mb", "first"),
        )
        .sort_values("median_atomic_reduction", ascending=False)
    )
    summary["passes_atomic"] = (
        summary["median_atomic_reduction"] > config.phase1_atomic_threshold
    )
    summary["passes_load"] = (
        summary["max_load_cv_ratio"] <= 1.0 + config.load_cv_tolerance
    )
    summary["phase1_pass"] = summary["passes_atomic"] & summary["passes_load"]
    dominant_ms = float(
        summary.loc[summary["method"] == "dominant", "preprocess_ms"].iloc[0]
    )
    summary["incremental_preprocess_ms"] = summary["preprocess_ms"]
    depends_on_dominant = summary["family"].isin(
        ["union", "normalized_union", "weighted_union"]
    )
    summary["total_preprocess_ms"] = summary["preprocess_ms"]
    summary.loc[depends_on_dominant, "total_preprocess_ms"] += dominant_ms
    chosen: list[str] = []
    for row in summary[summary["phase1_pass"]].itertuples():
        if row.family in {"identity", "dominant", "topk", "sketch"}:
            continue
        chosen_families = set(
            summary.loc[summary["method"].isin(chosen), "family"].tolist()
        )
        if row.family not in chosen_families:
            chosen.append(row.method)
        if len(chosen) == 3:
            break
    used_fallback = not chosen
    if used_fallback:
        fallback = summary[
            ~summary["family"].isin(["identity", "dominant", "topk", "sketch"])
        ].head(2)
        chosen = fallback["method"].tolist()
    summary["selected_for_phase2"] = summary["method"].isin(chosen)
    summary["selection_reason"] = "not_selected"
    summary.loc[
        summary["selected_for_phase2"] & summary["phase1_pass"],
        "selection_reason",
    ] = "threshold_pass"
    if used_fallback:
        summary.loc[summary["selected_for_phase2"], "selection_reason"] = (
            "diagnostic_fallback"
        )
    return summary


def custom_permutation(order: np.ndarray, device: torch.device) -> NeuronPermutation:
    """Convert an audited NumPy order to the repository permutation contract."""

    new_to_old = torch.from_numpy(order.astype(np.int64)).to(device)
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(order.size, device=device)
    return NeuronPermutation(
        new_to_old=new_to_old,
        old_to_new=old_to_new,
        config=ReorderConfig(mode="global_similarity"),
    )


def scipy_to_graph(matrix: sp.csr_matrix, device: torch.device) -> EventCSRGraph:
    """Move a SciPy presynaptic-row CSR into the persistent graph contract."""

    torch_matrix = CSR.from_scipy(matrix, device=device, dtype=torch.float32)
    return EventCSRGraph(
        indptr=torch_matrix.indptr.to(torch.int32).contiguous(),
        indices=torch_matrix.indices.to(torch.int32).contiguous(),
        weight=torch_matrix.effective_values().to(torch.float32).contiguous(),
        delay=None,
        shape=matrix.shape,
    )


def prepare_kernel_workload(
    method: str,
    order: np.ndarray,
    events: WindowedSpikeEvents,
    original_graph: EventCSRGraph,
    steps: int,
    device: torch.device,
) -> KernelWorkload:
    """Physically reorder graph, input events, and state for real-kernel timing."""

    state = make_empty_state(1, order.size, device=device, refractory=False)
    permutation = custom_permutation(order, device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    graph = reorder_graph(original_graph, permutation)
    reordered_events = reorder_events(events, permutation)
    reordered_state = reorder_state(state, permutation)
    workspace = make_persistent_snn_workspace(graph, 1)
    torch.cuda.synchronize()
    return KernelWorkload(
        method=method,
        order=order,
        events=reordered_events,
        graph=graph,
        state=reordered_state,
        params=PersistentSNNParams(window_size=steps),
        workspace=workspace,
        physical_reorder_ms=(time.perf_counter() - start) * 1000.0,
    )


def run_kernel(workload: KernelWorkload):
    """Run the production-timing BlockTask shared-hash specialization."""

    return persistent_snn_forward(
        workload.events,
        workload.graph,
        workload.state,
        workload.params,
        backend="cuda_persistent",
        return_mode="dense",
        spike_block=True,
        workspace=workload.workspace,
    )


def benchmark_interleaved(
    workloads: list[KernelWorkload], warmup: int, repeats: int, seed: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Time candidates in randomized round-robin order with CUDA Events."""

    samples: dict[str, list[float]] = {workload.method: [] for workload in workloads}
    outputs: dict[str, object] = {}
    for workload in workloads:
        for _ in range(warmup):
            outputs[workload.method] = run_kernel(workload)
    torch.cuda.synchronize()
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        for index in rng.permutation(len(workloads)):
            workload = workloads[int(index)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[workload.method] = run_kernel(workload)
            end.record()
            torch.cuda.synchronize()
            samples[workload.method].append(start.elapsed_time(end))
    rows = []
    baseline = np.asarray(samples["dominant"])
    for workload in workloads:
        values = np.asarray(samples[workload.method])
        paired_speedup = baseline / values - 1.0
        baseline_median = float(np.median(baseline))
        candidate_median = float(np.median(values))
        rows.append(
            {
                "method": workload.method,
                "median_ms": candidate_median,
                "p25_ms": float(np.quantile(values, 0.25)),
                "p75_ms": float(np.quantile(values, 0.75)),
                "time_per_step_us": float(
                    np.median(values) * 1000 / workload.params.window_size
                ),
                "median_speedup": baseline_median / candidate_median - 1.0,
                "speedup_p25": float(np.quantile(paired_speedup, 0.25)),
                "speedup_p75": float(np.quantile(paired_speedup, 0.75)),
                "physical_reorder_ms": workload.physical_reorder_ms,
            }
        )
    return pd.DataFrame(rows), outputs


def phase2_activity(
    workload: KernelWorkload,
    output: object,
    baseline_workload: KernelWorkload,
    baseline_output: object,
) -> dict[str, float | int]:
    """Measure posterior rate, work volume, and semantic divergence."""

    spikes = output.spikes
    baseline_spikes = baseline_output.spikes
    assert spikes is not None and baseline_spikes is not None
    restored = spikes.index_select(
        -1, custom_permutation(workload.order, spikes.device).old_to_new
    )
    restored_baseline = baseline_spikes.index_select(
        -1,
        custom_permutation(baseline_workload.order, baseline_spikes.device).old_to_new,
    )
    mismatch = int((restored != restored_baseline).sum().item())
    spike_count = int(spikes.sum().item())
    steps, batch, neurons = spikes.shape
    degree = (workload.graph.indptr[1:] - workload.graph.indptr[:-1]).to(torch.int64)
    active_edges = int((spikes.to(torch.int64) * degree.view(1, 1, -1)).sum().item())
    padded = torch.nn.functional.pad(spikes, (0, (-neurons) % BLOCK_SIZE))
    blocks = padded.reshape(steps, batch, -1, BLOCK_SIZE).sum(dim=-1)
    active_tasks = int((blocks > 0).sum().item())
    return {
        "posterior_population_hz": spike_count / neurons / (steps / 1000.0),
        "total_spikes": spike_count,
        "active_block_tasks": active_tasks,
        "edges_processed": active_edges,
        "spike_mismatches_vs_dominant": mismatch,
        "spike_mismatch_rate_vs_dominant": mismatch / spikes.numel(),
    }


def write_report(
    output: Path,
    config: V3Config,
    phase1: pd.DataFrame,
    selection: pd.DataFrame,
    kernel: pd.DataFrame,
) -> None:
    """Write a data-only Pareto report."""

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_markdown(index=False, floatfmt=".4g")

    fly_summary = selection.sort_values("median_atomic_reduction", ascending=False)
    lines = [
        "# Fanout preprocessing experiment (v3)",
        "",
        "Numerical results only; no visualization files were generated.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "## Phase 1 FlyWire screening",
        "",
        table(
            fly_summary,
            [
                "method",
                "family",
                "median_atomic_reduction",
                "median_ci95_low",
                "max_load_cv_ratio",
                "incremental_preprocess_ms",
                "total_preprocess_ms",
                "temporary_memory_mb",
                "phase1_pass",
                "selected_for_phase2",
                "selection_reason",
            ],
        ),
        "",
        "The exact-global greedy variant is intentionally excluded: its quadratic "
        "candidate scan is not an engineering candidate for 138k FlyWire neurons. "
        "All reported greedy methods use explicit bounded windows.",
    ]
    if not kernel.empty:
        lines.extend(
            [
                "",
                "## Phase 2 real persistent shared-hash kernel",
                "",
                table(
                    kernel.sort_values(["posterior_population_hz", "method"]),
                    [
                        "posterior_population_hz",
                        "method",
                        "time_per_step_us",
                        "median_speedup",
                        "speedup_p25",
                        "speedup_p75",
                        "active_block_tasks",
                        "active_block_task_reduction",
                        "edges_processed",
                        "spike_mismatch_rate_vs_dominant",
                    ],
                ),
            ]
        )
        candidates = kernel[kernel["method"] != "dominant"]
        stable = candidates.groupby("method").agg(
            median_speedup=("median_speedup", "median"),
            minimum_speedup=("median_speedup", "min"),
            positive_conditions=(
                "median_speedup",
                lambda value: int((value > 0).sum()),
            ),
            conditions=("median_speedup", "size"),
        )
        lines.extend(
            [
                "",
                "## Cross-rate kernel decision",
                "",
                stable.reset_index().to_markdown(index=False, floatfmt=".4g"),
                "",
                "Adoption requires a stable real-kernel improvement, acceptable "
                "preprocessing cost, and no material load-balance regression. "
                "Nsight counters are not collected automatically because profiling "
                "changes the execution mode; the timed rows use the uninstrumented "
                "production hash specialization.",
            ]
        )
        best_phase1 = fly_summary.iloc[0]
        fastest_condition = candidates.loc[candidates["median_speedup"].idxmax()]
        max_mismatch = float(kernel["spike_mismatch_rate_vs_dominant"].max())
        stable_winner = stable["median_speedup"].idxmax()
        stable_winner_gain = float(stable.loc[stable_winner, "median_speedup"])
        mapping = candidates.merge(
            phase1.loc[
                phase1["dataset"] == "flybrain",
                ["control_input_rate", "method", "atomic_reduction"],
            ],
            on=["control_input_rate", "method"],
        )
        atomic_runtime_correlation = float(
            mapping["atomic_reduction"].corr(mapping["median_speedup"])
        )
        task_runtime_correlation = float(
            mapping["active_block_task_reduction"].corr(mapping["median_speedup"])
        )
        lines.extend(
            [
                "",
                "## Decision",
                "",
                f"The strongest structural candidate was `{best_phase1['method']}` "
                f"at {100 * best_phase1['median_atomic_reduction']:.2f}% median "
                "estimated atomic reduction, but its worst Load-CV ratio was "
                f"{best_phase1['max_load_cv_ratio']:.3f}, so it failed the "
                "15% load-balance gate.",
                "",
                f"The best isolated real-kernel result was "
                f"`{fastest_condition['method']}` at "
                f"{fastest_condition['posterior_population_hz']:.3g} Hz with "
                f"{100 * fastest_condition['median_speedup']:.2f}% speedup. "
                f"The best cross-rate median was `{stable_winner}` at "
                f"{100 * stable_winner_gain:.2f}%. Neither reaches the 1% "
                "adoption threshold or remains positive across rates.",
                "",
                "Decision: reject every tested v3 fanout candidate for the "
                "current persistent shared-hash kernel. Structural atomic "
                "headroom did not translate into runtime speedup. Retain the "
                "dominant baseline.",
                "",
                "Across the selected methods and seven conditions, the "
                "descriptive Pearson association between Phase-1 atomic "
                f"reduction and kernel speedup was {atomic_runtime_correlation:.3f}; "
                "the association between active-BlockTask reduction and speedup "
                f"was {task_runtime_correlation:.3f}. These pooled correlations "
                "are diagnostic rather than causal, but they confirm that the "
                "structural proxies do not predict runtime benefit here.",
                "",
                f"The maximum reordered-vs-baseline spike mismatch rate was "
                f"{max_mismatch:.3g}; this is reported because floating-point "
                "atomic accumulation order can perturb threshold crossings.",
            ]
        )
    lines.extend(
        [
            "",
            "Phase-1 atomic counts are ratio-of-means estimates from uniformly "
            "sampled active BlockTasks. Phase-2 speedups use randomized interleaved "
            "CUDA Event samples against the dominant baseline.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("sanity", "phase1", "phase2", "full"), default="full"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    """Run the requested v3 experiment phases."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the v3 experiment")
    sanity = args.phase == "sanity"
    config = V3Config(
        datasets=("uniform", "flybrain")
        if sanity
        else ("uniform", "community", "spatial", "flybrain"),
        target_rates_hz=TARGET_RATES_HZ,
        calibration_input_rates=CALIBRATION_RATES,
        calibration_steps=128 if sanity else 512,
        phase1_steps=256 if sanity else 1024,
        kernel_steps=128 if sanity else 512,
        synthetic_neurons=2048 if sanity else 8192,
        synthetic_degree=32 if sanity else 64,
        recurrent_strength=0.1,
        greedy_window=64 if sanity else 128,
        sketch_window=128 if sanity else 256,
        task_sample=10_000 if sanity else 100_000,
        kernel_warmup=5 if sanity else 10,
        kernel_repeats=10 if sanity else 30,
        phase1_atomic_threshold=0.01,
        load_cv_tolerance=0.15,
        seed=args.seed,
    )
    output = (
        args.output
        or REPO_ROOT
        / "benchmark"
        / "reorder_analysis"
        / "results"
        / f"v3_{args.phase}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    os.environ.setdefault("BTORCH_BLOCK_EDGE_BUDGET", "512")
    os.environ.setdefault("BTORCH_BLOCK_HASH_AGGREGATION", "512")
    os.environ.setdefault("BTORCH_BLOCK_HASH_CAPACITY", "512")
    os.environ.setdefault("BTORCH_BLOCK_HASH_MAX_PROBE", "4")
    os.environ.setdefault("BTORCH_BLOCK_HASH_MIN_EDGES", "256")
    os.environ.setdefault("BTORCH_BLOCK_HASH_USED_SLOTS", "1")
    from btorch.backend.persistent import plain_version

    plain_version.load(enable_block_hash=True)
    device = torch.device("cuda")
    graph_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    preprocess_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    orders_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    matrices: dict[str, sp.csr_matrix] = {}
    selected_controls: dict[str, list[float]] = {}

    if args.phase in {"sanity", "phase1", "full"}:
        # Compile Numba kernels before recording preprocessing cost.
        tiny = sp.eye(64, format="csr", dtype=np.float32)
        build_topk_order(tiny, 2)
        build_sketch_order(tiny, 128)
        build_union_order(tiny, np.arange(64, dtype=np.int32), window_size=64)
        for dataset_id, dataset in enumerate(config.datasets):
            print(f"[{dataset}] load, calibrate, and build orders", flush=True)
            topology, weighted = load_graph(
                dataset,
                n=config.synthetic_neurons,
                degree=config.synthetic_degree,
                seed=config.seed + 1000 * dataset_id,
            )
            matrix = scale_simulation_weights(
                weighted, dataset, config.recurrent_strength
            )
            matrices[dataset] = matrix
            graph_rows.append(graph_summary(dataset, topology))
            controls, calibration_rows_ds = calibrate(
                dataset, matrix, config, dataset_id, device
            )
            calibration_rows.extend(calibration_rows_ds)
            selected_controls[dataset] = controls
            orders: dict[str, np.ndarray] = {}
            specs = order_specs(topology, config)
            for spec in specs:
                order, elapsed, temporary, retained = measure_builder(spec.builder)
                validate_order(order, topology.shape[0])
                orders[spec.method] = order
                preprocess_rows.append(
                    {
                        "dataset": dataset,
                        "method": spec.method,
                        "family": spec.family,
                        "complexity": spec.complexity,
                        "preprocess_ms": elapsed,
                        "temporary_memory_mb": temporary,
                        "retained_order_mb": retained,
                        **load_balance(order, topology),
                        "order_fingerprint": hashlib.sha1(
                            order.tobytes(), usedforsecurity=False
                        ).hexdigest(),
                    }
                )
            orders_by_dataset[dataset] = orders
            preprocess_lookup = {
                row["method"]: row
                for row in preprocess_rows
                if row["dataset"] == dataset
            }
            for condition_id, control in enumerate(controls):
                spikes = simulate_spikes(
                    matrix,
                    timesteps=config.phase1_steps,
                    input_rate=control,
                    input_amplitude=1.1,
                    seed=config.seed + dataset_id * 100_000 + condition_id,
                    device=device,
                )
                stats = firing_statistics(spikes)
                for method, order in orders.items():
                    measured = analyze_layout(
                        dataset,
                        control,
                        spikes,
                        topology,
                        method,
                        order,
                        block_size=BLOCK_SIZE,
                        task_sample=config.task_sample,
                        seed=config.seed + dataset_id * 1000 + condition_id,
                    )
                    prep = preprocess_lookup[method]
                    metric_rows.append(
                        {
                            **measured,
                            "method": method,
                            "family": prep["family"],
                            "control_input_rate": control,
                            "population_mean_hz": stats["population_mean_hz"],
                            "preprocess_ms": prep["preprocess_ms"],
                            "temporary_memory_mb": prep["temporary_memory_mb"],
                            **{
                                key: prep[key]
                                for key in (
                                    "load_mean",
                                    "load_std",
                                    "load_cv",
                                    "load_p95",
                                    "load_p99",
                                    "load_max",
                                )
                            },
                        }
                    )
        phase1 = compare_phase1(pd.DataFrame(metric_rows))
        # Persist the expensive raw screening results before applying the
        # comparatively cheap finalist policy.
        phase1.to_csv(output / "phase1_metrics.csv", index=False)
        selection = select_finalists(phase1, config)
        pd.DataFrame(graph_rows).to_csv(output / "graph_summary.csv", index=False)
        pd.DataFrame(calibration_rows).to_csv(output / "calibration.csv", index=False)
        pd.DataFrame(preprocess_rows).to_csv(output / "preprocessing.csv", index=False)
        selection.to_csv(output / "phase1_selection.csv", index=False)
    else:
        phase1 = pd.read_csv(output / "phase1_metrics.csv")
        selection = pd.read_csv(output / "phase1_selection.csv")

    kernel_rows: list[dict[str, object]] = []
    if args.phase in {"sanity", "phase2", "full"}:
        if "flybrain" not in matrices:
            topology, weighted = load_graph(
                "flybrain",
                n=config.synthetic_neurons,
                degree=config.synthetic_degree,
                seed=config.seed + 3000,
            )
            matrices["flybrain"] = scale_simulation_weights(
                weighted, "flybrain", config.recurrent_strength
            )
            orders_by_dataset["flybrain"] = {
                spec.method: spec.builder() for spec in order_specs(topology, config)
            }
            controls, _rows = calibrate(
                "flybrain", matrices["flybrain"], config, 3, device
            )
            selected_controls["flybrain"] = controls
        matrix = matrices["flybrain"]
        topology = matrix.copy()
        topology.data.fill(1.0)
        orders = orders_by_dataset["flybrain"]
        finalists = selection.loc[selection["selected_for_phase2"], "method"].tolist()
        methods = ["dominant", *finalists]
        original_graph = scipy_to_graph(matrix, device)
        for condition, control in enumerate(selected_controls["flybrain"]):
            print(f"[kernel] control={control:.4%}, methods={methods}", flush=True)
            events = make_random_events(
                config.kernel_steps,
                matrix.shape[0],
                control,
                1.1,
                config.seed + 900_000 + condition,
                device,
            )
            workloads = [
                prepare_kernel_workload(
                    method,
                    orders[method],
                    events,
                    original_graph,
                    config.kernel_steps,
                    device,
                )
                for method in methods
            ]
            timing, outputs = benchmark_interleaved(
                workloads,
                config.kernel_warmup,
                config.kernel_repeats,
                config.seed + condition,
            )
            baseline_output = outputs["dominant"]
            baseline_workload = workloads[0]
            for workload in workloads:
                timing_row = (
                    timing[timing["method"] == workload.method].iloc[0].to_dict()
                )
                kernel_rows.append(
                    {
                        "control_input_rate": control,
                        **timing_row,
                        **phase2_activity(
                            workload,
                            outputs[workload.method],
                            baseline_workload,
                            baseline_output,
                        ),
                        "order_preprocess_ms": float(
                            selection.loc[
                                selection["method"] == workload.method,
                                "total_preprocess_ms",
                            ].iloc[0]
                        ),
                    }
                )
            del workloads, events
            torch.cuda.empty_cache()
    kernel = pd.DataFrame(kernel_rows)
    if not kernel.empty:
        baseline_tasks = kernel[kernel["method"] == "dominant"].set_index(
            "control_input_rate"
        )["active_block_tasks"]
        kernel["active_block_task_reduction"] = (
            1.0
            - kernel["active_block_tasks"].to_numpy()
            / kernel["control_input_rate"].map(baseline_tasks).to_numpy()
        )
    kernel.to_csv(output / "kernel_benchmark.csv", index=False)
    write_report(output, config, phase1, selection, kernel)
    print(f"complete: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
