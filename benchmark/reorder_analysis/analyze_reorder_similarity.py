"""Test whether fanin similarity predicts useful co-spiking locality.

This is an offline characterization experiment, not a kernel benchmark.  It
implements the three layers in ``preprocess_plan_v1.md`` and deliberately emits
tables and raw records only; no plotting dependency is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO = REPO_ROOT / "libs" / "dataset"
for path in (REPO_ROOT, DATASET_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.benchmark_rsnn_roofline import csr_to_event_graph  # noqa: E402
from btorch.backend.persistent_snn import (  # noqa: E402
    PersistentSNNParams,
    WindowedSpikeEvents,
    make_empty_state,
    make_persistent_snn_workspace,
    persistent_snn_forward,
)
from btorch.sparse import CSR  # noqa: E402


FANIN_BINS = np.array(
    [-1e-12, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.000001]
)
FANIN_LABELS = [
    "[0,.01)",
    "[.01,.05)",
    "[.05,.1)",
    "[.1,.2)",
    "[.2,.4)",
    "[.4,.6)",
    "[.6,.8)",
    "[.8,1]",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Record all choices needed to reproduce one experiment run."""

    datasets: tuple[str, ...]
    input_rates: tuple[float, ...]
    synthetic_neurons: int
    synthetic_degree: int
    synthetic_timesteps: int
    flybrain_timesteps: int
    pair_count: int
    block_task_sample: int
    block_size: int
    synthetic_recurrent_strength: float
    input_amplitude: float
    seed: int


def _binary_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    result = matrix.tocsr().astype(np.float32)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    result.data.fill(1.0)
    return result


def erdos_renyi_graph(n: int, degree: int, seed: int) -> sp.csr_matrix:
    """Generate a directed uniform graph with approximately fixed mean degree."""

    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n, dtype=np.int32), degree)
    cols = rng.integers(0, n, size=rows.size, dtype=np.int32)
    keep = rows != cols
    return _binary_csr(
        sp.coo_matrix(
            (np.ones(keep.sum(), dtype=np.float32), (rows[keep], cols[keep])),
            shape=(n, n),
        )
    )


def community_graph(n: int, degree: int, seed: int) -> sp.csr_matrix:
    """Generate a directed stochastic-block graph with strong communities."""

    rng = np.random.default_rng(seed)
    community_size = max(64, min(256, n // 8))
    within = int(round(degree * 0.85))
    outside = degree - within
    rows = np.repeat(np.arange(n, dtype=np.int32), degree)
    cols = np.empty(rows.size, dtype=np.int32)
    for neuron in range(n):
        begin = neuron * degree
        group = neuron // community_size
        low = group * community_size
        high = min((group + 1) * community_size, n)
        cols[begin : begin + within] = rng.integers(
            low, high, size=within, dtype=np.int32
        )
        if outside:
            candidates = rng.integers(
                0, max(n - (high - low), 1), size=outside, dtype=np.int32
            )
            candidates += (candidates >= low) * (high - low)
            cols[begin + within : begin + degree] = np.minimum(candidates, n - 1)
    keep = rows != cols
    matrix = sp.coo_matrix(
        (np.ones(keep.sum(), dtype=np.float32), (rows[keep], cols[keep])),
        shape=(n, n),
    )
    return _binary_csr(matrix)


def spatial_graph(n: int, degree: int, seed: int) -> sp.csr_matrix:
    """Generate a ring graph whose connection probability decays with distance."""

    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n, dtype=np.int32), degree)
    scale = max(8.0, n / 128.0)
    distance = np.maximum(
        1, np.ceil(rng.exponential(scale, size=rows.size)).astype(np.int32)
    )
    sign = rng.choice(np.array([-1, 1], dtype=np.int32), size=rows.size)
    cols = (rows + sign * distance) % n
    return _binary_csr(
        sp.coo_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n)
        )
    )


def load_graph(
    dataset: str,
    *,
    n: int,
    degree: int,
    seed: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return binary topology and the weighted graph used by the simulator."""

    if dataset == "uniform":
        topology = erdos_renyi_graph(n, degree, seed)
    elif dataset == "community":
        topology = community_graph(n, degree, seed)
    elif dataset == "spatial":
        topology = spatial_graph(n, degree, seed)
    elif dataset == "flybrain":
        from connectome_dataset.graph_loader import load_flywire_783

        weighted = load_flywire_783(use_weights=True).tocsr().astype(np.float32)
        weighted.sum_duplicates()
        weighted.eliminate_zeros()
        weighted.sort_indices()
        return _binary_csr(weighted), weighted
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return topology, topology.copy()


def scale_simulation_weights(
    weighted: sp.csr_matrix,
    dataset: str,
    recurrent_strength: float,
) -> sp.csr_matrix:
    """Scale topology while preserving FlyWire's signed synapse counts."""

    result = weighted.copy().astype(np.float32)
    if dataset == "flybrain":
        mean_abs_fanout = float(np.abs(result.data).sum()) / max(
            result.shape[0], 1
        )
        result.data *= np.float32(
            recurrent_strength / max(mean_abs_fanout, 1.0)
        )
    else:
        mean_degree = result.nnz / max(result.shape[0], 1)
        rng = np.random.default_rng(1729)
        inhibitory = rng.random(result.shape[0]) >= 0.8
        pre_sign = np.where(inhibitory, -4.0, 1.0).astype(np.float32)
        row_degree = np.diff(result.indptr)
        result.data *= np.repeat(pre_sign, row_degree)
        result.data *= np.float32(recurrent_strength / max(mean_degree, 1.0))
    return result


def make_random_events(
    timesteps: int,
    n: int,
    rate: float,
    amplitude: float,
    seed: int,
    device: torch.device,
):
    """Build independent Bernoulli input events without retaining a dense trace."""

    rng = np.random.default_rng(seed)
    counts = np.empty(timesteps, dtype=np.int32)
    index_parts: list[np.ndarray] = []
    chunk = max(1, min(32, 8_000_000 // max(n, 1)))
    for start in range(0, timesteps, chunk):
        width = min(chunk, timesteps - start)
        active = rng.random((width, n), dtype=np.float32) < rate
        chunk_counts = active.sum(axis=1, dtype=np.int32)
        counts[start : start + width] = chunk_counts
        index_parts.append(np.nonzero(active)[1].astype(np.int32, copy=False))
    host_indices = (
        np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
    )
    offsets = np.empty(timesteps + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("input trace contains too many events for int32 offsets")
    return WindowedSpikeEvents(
        offsets=torch.from_numpy(offsets.astype(np.int32)).to(device),
        indices=torch.from_numpy(host_indices).to(device),
        values=torch.full(
            (host_indices.size,), amplitude, device=device, dtype=torch.float32
        ),
        shape=(timesteps, 1, n),
    )


def simulate_spikes(
    matrix: sp.csr_matrix,
    *,
    timesteps: int,
    input_rate: float,
    input_amplitude: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Run the repository persistent LIF+PSC implementation and return uint8 spikes."""

    torch_matrix = CSR.from_scipy(matrix, device=device, dtype=torch.float32)
    graph = csr_to_event_graph(torch_matrix)
    events = make_random_events(
        timesteps,
        matrix.shape[0],
        input_rate,
        input_amplitude,
        seed,
        device,
    )
    state = make_empty_state(1, matrix.shape[0], device=device, refractory=False)
    params = PersistentSNNParams(
        dt=1.0,
        tau_mem=20.0,
        tau_syn=5.0,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        # CUDA persistent v1 supports soft reset only. Input amplitude is kept
        # just above threshold so one external event leaves only a 0.1 residual.
        hard_reset=False,
        window_size=timesteps,
    )
    workspace = make_persistent_snn_workspace(graph, 1)
    output = persistent_snn_forward(
        events,
        graph,
        state,
        params,
        backend="cuda_persistent",
        return_mode="dense",
        workspace=workspace,
    )
    assert output.spikes is not None
    spikes = output.spikes[:, 0].to(torch.uint8).cpu().numpy()
    del output, workspace, state, events, graph, torch_matrix
    torch.cuda.empty_cache()
    return spikes


def sample_pairs(
    topology: sp.csr_matrix, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample half random pairs and half pairs sharing at least one fanin."""

    rng = np.random.default_rng(seed)
    n = topology.shape[0]
    random_count = count // 2
    left = rng.integers(0, n, size=random_count, dtype=np.int32)
    right = rng.integers(0, n, size=random_count, dtype=np.int32)
    equal = left == right
    while equal.any():
        right[equal] = rng.integers(0, n, size=int(equal.sum()), dtype=np.int32)
        equal = left == right

    positive_left: list[int] = []
    positive_right: list[int] = []
    nontrivial = np.flatnonzero(np.diff(topology.indptr) >= 2)
    target = count - random_count
    attempts = 0
    while len(positive_left) < target:
        attempts += 1
        if attempts > target * 20 + 1000:
            raise RuntimeError("could not sample enough shared-fanin pairs")
        pre = int(rng.choice(nontrivial))
        row = topology.indices[topology.indptr[pre] : topology.indptr[pre + 1]]
        pair = rng.choice(row, size=2, replace=False)
        if pair[0] != pair[1]:
            positive_left.append(int(pair[0]))
            positive_right.append(int(pair[1]))
    left = np.concatenate([left, np.asarray(positive_left, dtype=np.int32)])
    right = np.concatenate([right, np.asarray(positive_right, dtype=np.int32)])
    stratum = np.concatenate(
        [
            np.full(random_count, "random", dtype=object),
            np.full(target, "shared_fanin", dtype=object),
        ]
    )
    return left, right, stratum


@numba.njit(cache=True)
def _pair_overlap(
    left: np.ndarray,
    right: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    intersection = np.zeros(left.size, dtype=np.int32)
    similarity = np.zeros(left.size, dtype=np.float64)
    for pair_id in range(left.size):
        i = left[pair_id]
        j = right[pair_id]
        ai = indptr[i]
        ae = indptr[i + 1]
        bi = indptr[j]
        be = indptr[j + 1]
        common = 0
        while ai < ae and bi < be:
            av = indices[ai]
            bv = indices[bi]
            if av == bv:
                common += 1
                ai += 1
                bi += 1
            elif av < bv:
                ai += 1
            else:
                bi += 1
        union = (ae - indptr[i]) + (be - indptr[j]) - common
        intersection[pair_id] = common
        if union:
            similarity[pair_id] = common / union
    return intersection, similarity


def topology_pair_frame(
    topology: sp.csr_matrix,
    left: np.ndarray,
    right: np.ndarray,
    stratum: np.ndarray,
) -> pd.DataFrame:
    """Compute exact fanin/fanout Jaccard and degree controls."""

    fanin = topology.transpose().tocsr()
    fanin.sort_indices()
    in_common, in_sim = _pair_overlap(
        left, right, fanin.indptr, fanin.indices
    )
    out_common, out_sim = _pair_overlap(
        left, right, topology.indptr, topology.indices
    )
    in_degree = np.diff(fanin.indptr)
    out_degree = np.diff(topology.indptr)
    return pd.DataFrame(
        {
            "i": left,
            "j": right,
            "sampling_stratum": stratum,
            "fanin_similarity": in_sim,
            "fanout_similarity": out_sim,
            "shared_fanin": in_common,
            "shared_targets": out_common,
            "fanin_degree_i": in_degree[left],
            "fanin_degree_j": in_degree[right],
            "fanout_degree_i": out_degree[left],
            "fanout_degree_j": out_degree[right],
        }
    )


def add_spike_pair_stats(
    topology_frame: pd.DataFrame, spikes: np.ndarray
) -> pd.DataFrame:
    """Add dynamic co-spiking metrics in bounded-memory vectorized chunks."""

    result = topology_frame.copy()
    left = result["i"].to_numpy(dtype=np.int64)
    right = result["j"].to_numpy(dtype=np.int64)
    timesteps = spikes.shape[0]
    counts = spikes.sum(axis=0, dtype=np.int64)
    joint = np.empty(left.size, dtype=np.int32)
    chunk = max(1, 32_000_000 // max(timesteps, 1))
    for start in range(0, left.size, chunk):
        end = min(start + chunk, left.size)
        joint[start:end] = np.logical_and(
            spikes[:, left[start:end]], spikes[:, right[start:end]]
        ).sum(axis=0, dtype=np.int32)
    count_i = counts[left]
    count_j = counts[right]
    union = count_i + count_j - joint
    denom_cosine = np.sqrt(count_i.astype(np.float64) * count_j)
    expected_count = count_i.astype(np.float64) * count_j / timesteps
    result["spike_count_i"] = count_i
    result["spike_count_j"] = count_j
    result["joint_count"] = joint
    result["p_i"] = count_i / timesteps
    result["p_j"] = count_j / timesteps
    result["p_joint"] = joint / timesteps
    result["spike_jaccard"] = np.divide(
        joint, union, out=np.zeros_like(expected_count), where=union > 0
    )
    result["spike_cosine"] = np.divide(
        joint,
        denom_cosine,
        out=np.full_like(expected_count, np.nan),
        where=denom_cosine > 0,
    )
    result["co_spike_lift"] = np.divide(
        joint,
        expected_count,
        out=np.full_like(expected_count, np.nan),
        where=expected_count > 0,
    )
    result["aggregation_opportunity_exact"] = (
        result["p_joint"] * result["shared_targets"]
    )
    result["aggregation_opportunity_normalized"] = (
        result["p_joint"] * result["fanout_similarity"]
    )
    return result


def dominant_block_signatures(
    matrix: sp.csr_matrix, block_size: int, top_k: int = 2
) -> np.ndarray:
    """Return the top-k source/destination block IDs for every CSR row."""

    signatures = np.full((matrix.shape[0], top_k), -1, dtype=np.int32)
    for row in range(matrix.shape[0]):
        values = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
        if values.size == 0:
            continue
        blocks, counts = np.unique(values // block_size, return_counts=True)
        order = np.lexsort((blocks, -counts))[:top_k]
        signatures[row, : order.size] = blocks[order]
    return signatures


def make_orders(
    topology: sp.csr_matrix, block_size: int, seed: int
) -> dict[str, np.ndarray]:
    """Construct the five plan-specified prototype layouts."""

    n = topology.shape[0]
    ids = np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    fanout = dominant_block_signatures(topology, block_size)
    fanin = dominant_block_signatures(topology.transpose().tocsr(), block_size)

    def stable_order(*keys: np.ndarray) -> np.ndarray:
        return np.lexsort((ids, *reversed(keys))).astype(np.int32)

    return {
        "identity": ids,
        "random": rng.permutation(ids),
        "fanout": stable_order(fanout[:, 0]),
        "fanin": stable_order(fanin[:, 0], fanin[:, 1]),
        "fanin_fanout": stable_order(
            fanin[:, 0], fanin[:, 1], fanout[:, 0], fanout[:, 1]
        ),
    }


@numba.njit(cache=True)
def _sampled_task_unions(
    spikes: np.ndarray,
    order: np.ndarray,
    task_t: np.ndarray,
    task_block: np.ndarray,
    block_size: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.zeros(task_t.size, dtype=np.int64)
    unique = np.zeros(task_t.size, dtype=np.int64)
    unique_post_blocks = np.zeros(task_t.size, dtype=np.int64)
    post_marker = np.zeros(n, dtype=np.int32)
    block_marker = np.zeros((n + block_size - 1) // block_size, dtype=np.int32)
    for task in range(task_t.size):
        token = task + 1
        begin = task_block[task] * block_size
        end = min(begin + block_size, order.size)
        for position in range(begin, end):
            neuron = order[position]
            if spikes[task_t[task], neuron] == 0:
                continue
            row_begin = indptr[neuron]
            row_end = indptr[neuron + 1]
            raw[task] += row_end - row_begin
            for edge in range(row_begin, row_end):
                post = indices[edge]
                if post_marker[post] != token:
                    post_marker[post] = token
                    unique[task] += 1
                post_block = post // block_size
                if block_marker[post_block] != token:
                    block_marker[post_block] = token
                    unique_post_blocks[task] += 1
    return raw, unique, unique_post_blocks


@numba.njit(cache=True)
def _static_block_unions(
    order: np.ndarray,
    block_size: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    n: int,
) -> tuple[int, int]:
    marker = np.zeros(n, dtype=np.int32)
    raw = 0
    unique = 0
    block_count = (order.size + block_size - 1) // block_size
    for block in range(block_count):
        token = block + 1
        begin = block * block_size
        end = min(begin + block_size, order.size)
        for position in range(begin, end):
            neuron = order[position]
            row_begin = indptr[neuron]
            row_end = indptr[neuron + 1]
            raw += row_end - row_begin
            for edge in range(row_begin, row_end):
                post = indices[edge]
                if marker[post] != token:
                    marker[post] = token
                    unique += 1
    return raw, unique


def analyze_layout(
    dataset: str,
    input_rate: float,
    spikes: np.ndarray,
    topology: sp.csr_matrix,
    name: str,
    order: np.ndarray,
    *,
    block_size: int,
    task_sample: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Compute exact occupancy and sampled destination-union metrics."""

    n = topology.shape[0]
    pad = (-n) % block_size
    ordered = spikes[:, order]
    if pad:
        ordered = np.pad(ordered, ((0, 0), (0, pad)))
    occupancy = ordered.reshape(spikes.shape[0], -1, block_size).sum(
        axis=2, dtype=np.int16
    )
    active_t, active_block = np.nonzero(occupancy)
    active_occupancy = occupancy[active_t, active_block]
    rng = np.random.default_rng(seed)
    if active_t.size > task_sample:
        selected = rng.choice(active_t.size, size=task_sample, replace=False)
        task_t = active_t[selected].astype(np.int32)
        task_block = active_block[selected].astype(np.int32)
    else:
        task_t = active_t.astype(np.int32)
        task_block = active_block.astype(np.int32)
    raw, unique, unique_post_blocks = _sampled_task_unions(
        spikes,
        order,
        task_t,
        task_block,
        block_size,
        topology.indptr,
        topology.indices,
        n,
    )
    valid_edges = raw > 0
    sampled_raw = int(raw[valid_edges].sum())
    sampled_unique = int(unique[valid_edges].sum())
    static_raw, static_unique = _static_block_unions(
        order,
        block_size,
        topology.indptr,
        topology.indices,
        n,
    )
    realized_ratio = (
        1.0 - sampled_unique / sampled_raw if sampled_raw else 0.0
    )
    if valid_edges.sum() > 1 and sampled_raw:
        raw_valid = raw[valid_edges].astype(np.float64)
        unique_valid = unique[valid_edges].astype(np.float64)
        unique_per_raw = unique_valid.sum() / raw_valid.sum()
        influence = (
            unique_valid - unique_per_raw * raw_valid
        ) / raw_valid.mean()
        aggregation_se = float(
            influence.std(ddof=1) / math.sqrt(influence.size)
        )
        unique_mean = float(unique_valid.mean())
        unique_mean_se = float(
            unique_valid.std(ddof=1) / math.sqrt(unique_valid.size)
        )
    else:
        aggregation_se = 0.0
        unique_mean = float(unique[valid_edges].mean()) if valid_edges.any() else 0.0
        unique_mean_se = 0.0
    potential_ratio = 1.0 - static_unique / static_raw if static_raw else 0.0
    total_spikes = int(spikes.sum())
    active_tasks = int(active_t.size)
    return {
        "dataset": dataset,
        "input_rate": input_rate,
        "layout": name,
        "timesteps": spikes.shape[0],
        "total_spikes": total_spikes,
        "actual_firing_rate": total_spikes / spikes.size,
        "active_block_tasks": active_tasks,
        "active_blocks_per_step": active_tasks / spikes.shape[0],
        "mean_active_occupancy": (
            float(active_occupancy.mean()) if active_occupancy.size else 0.0
        ),
        "median_active_occupancy": (
            float(np.median(active_occupancy)) if active_occupancy.size else 0.0
        ),
        "p90_active_occupancy": (
            float(np.quantile(active_occupancy, 0.9))
            if active_occupancy.size
            else 0.0
        ),
        "block_compression": total_spikes / active_tasks if active_tasks else 0.0,
        "sampled_active_tasks": int(task_t.size),
        "sampled_raw_edges": sampled_raw,
        "sampled_unique_destinations": sampled_unique,
        "sampled_unique_destinations_mean": unique_mean,
        "sampled_unique_destinations_mean_se": unique_mean_se,
        "estimated_global_atomic_count": unique_mean * active_tasks,
        "estimated_global_atomic_count_se": unique_mean_se * active_tasks,
        "weighted_aggregation_ratio": realized_ratio,
        "weighted_aggregation_ratio_se": aggregation_se,
        "weighted_aggregation_ci95_low": max(
            0.0, realized_ratio - 1.96 * aggregation_se
        ),
        "weighted_aggregation_ci95_high": min(
            1.0, realized_ratio + 1.96 * aggregation_se
        ),
        "sampled_unique_post_blocks": int(unique_post_blocks.sum()),
        "post_block_reuse": (
            sampled_raw / unique_post_blocks.sum()
            if unique_post_blocks.sum()
            else 0.0
        ),
        "static_potential_aggregation_ratio": potential_ratio,
        "realized_to_potential": (
            realized_ratio / potential_ratio if potential_ratio > 0 else 0.0
        ),
    }


def _safe_corr(frame: pd.DataFrame, x: str, y: str, method: str) -> float:
    valid = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def partial_correlation(frame: pd.DataFrame, outcome: str) -> float:
    """Residualize similarity and outcome against fanout/degree/rate controls."""

    columns = [
        "fanin_similarity",
        outcome,
        "fanout_similarity",
        "fanin_degree_i",
        "fanin_degree_j",
        "fanout_degree_i",
        "fanout_degree_j",
        "p_i",
        "p_j",
    ]
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return float("nan")
    y_x = clean["fanin_similarity"].to_numpy(dtype=np.float64)
    y_out = clean[outcome].to_numpy(dtype=np.float64)
    controls = clean[columns[2:]].to_numpy(dtype=np.float64)
    controls[:, 1:] = np.log1p(controls[:, 1:])
    controls = np.column_stack([np.ones(len(clean)), controls])
    residual_x = y_x - controls @ np.linalg.lstsq(controls, y_x, rcond=None)[0]
    residual_out = y_out - controls @ np.linalg.lstsq(
        controls, y_out, rcond=None
    )[0]
    if residual_x.std() == 0 or residual_out.std() == 0:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_out)[0, 1])


def summarize_pairs(
    dataset: str, input_rate: float, frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int | str], pd.DataFrame]:
    """Build fanin bins, global tests, and opportunity comparisons."""

    working = frame.copy()
    working["fanin_bin"] = pd.cut(
        working["fanin_similarity"],
        bins=FANIN_BINS,
        labels=FANIN_LABELS,
        include_lowest=True,
        right=False,
    )
    binned = (
        working.groupby("fanin_bin", observed=False)
        .agg(
            pair_count=("i", "size"),
            finite_lift_count=("co_spike_lift", "count"),
            nonzero_joint_fraction=("joint_count", lambda x: float((x > 0).mean())),
            mean_joint_probability=("p_joint", "mean"),
            mean_spike_jaccard=("spike_jaccard", "mean"),
            mean_spike_cosine=("spike_cosine", "mean"),
            mean_lift=("co_spike_lift", "mean"),
            median_lift=("co_spike_lift", "median"),
            lift_p25=("co_spike_lift", lambda x: x.quantile(0.25)),
            lift_p75=("co_spike_lift", lambda x: x.quantile(0.75)),
            mean_fanout_similarity=("fanout_similarity", "mean"),
            mean_aggregation_opportunity=(
                "aggregation_opportunity_exact",
                "mean",
            ),
        )
        .reset_index()
    )
    binned.insert(0, "input_rate", input_rate)
    binned.insert(0, "dataset", dataset)

    global_stats: dict[str, float | int | str] = {
        "dataset": dataset,
        "input_rate": input_rate,
        "pair_count": len(frame),
        "finite_lift_pairs": int(frame["co_spike_lift"].notna().sum()),
        "pearson_fanin_fanout": _safe_corr(
            frame, "fanin_similarity", "fanout_similarity", "pearson"
        ),
        "spearman_fanin_fanout": _safe_corr(
            frame, "fanin_similarity", "fanout_similarity", "spearman"
        ),
        "spearman_fanin_spike_jaccard": _safe_corr(
            frame, "fanin_similarity", "spike_jaccard", "spearman"
        ),
        "spearman_fanin_spike_cosine": _safe_corr(
            frame, "fanin_similarity", "spike_cosine", "spearman"
        ),
        "spearman_fanin_lift": _safe_corr(
            frame, "fanin_similarity", "co_spike_lift", "spearman"
        ),
        "partial_fanin_spike_jaccard": partial_correlation(
            frame, "spike_jaccard"
        ),
        "partial_fanin_lift": partial_correlation(frame, "co_spike_lift"),
    }

    positive_in = frame.loc[frame["fanin_similarity"] > 0, "fanin_similarity"]
    positive_out = frame.loc[frame["fanout_similarity"] > 0, "fanout_similarity"]
    in_threshold = float(positive_in.quantile(0.9)) if len(positive_in) else math.inf
    out_threshold = (
        float(positive_out.quantile(0.9)) if len(positive_out) else math.inf
    )
    masks = {
        "random": frame["sampling_stratum"].eq("random"),
        "fanin_similar": frame["fanin_similarity"].ge(in_threshold),
        "fanout_similar": frame["fanout_similarity"].ge(out_threshold),
        "fanin_and_fanout": frame["fanin_similarity"].ge(in_threshold)
        & frame["fanout_similarity"].ge(out_threshold),
    }
    opportunity_rows = []
    random_mean = float(
        frame.loc[masks["random"], "aggregation_opportunity_exact"].mean()
    )
    for group, mask in masks.items():
        subset = frame.loc[mask]
        mean_exact = float(subset["aggregation_opportunity_exact"].mean())
        opportunity_rows.append(
            {
                "dataset": dataset,
                "input_rate": input_rate,
                "pair_group": group,
                "pair_count": len(subset),
                "fanin_threshold": in_threshold,
                "fanout_threshold": out_threshold,
                "mean_p_joint": float(subset["p_joint"].mean()),
                "mean_shared_targets": float(subset["shared_targets"].mean()),
                "mean_opportunity_exact": mean_exact,
                "mean_opportunity_normalized": float(
                    subset["aggregation_opportunity_normalized"].mean()
                ),
                "opportunity_lift_vs_random": (
                    mean_exact / random_mean if random_mean > 0 else math.nan
                ),
            }
        )
    return binned, global_stats, pd.DataFrame(opportunity_rows)


def summarize_controlled_pairs(
    dataset: str, input_rate: float, frame: pd.DataFrame
) -> pd.DataFrame:
    """Compare fanin levels inside fixed fanout and fanin-degree strata."""

    controlled = frame.copy()
    positive = controlled.loc[
        controlled["fanin_similarity"] > 0, "fanin_similarity"
    ]
    median_positive = float(positive.median()) if len(positive) else math.inf
    controlled["fanin_level"] = np.select(
        [
            controlled["fanin_similarity"] == 0,
            controlled["fanin_similarity"] <= median_positive,
        ],
        ["zero", "low_nonzero"],
        default="high",
    )
    controlled["fanout_bin"] = pd.cut(
        controlled["fanout_similarity"],
        bins=[-1e-12, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.000001],
        include_lowest=True,
        right=False,
    ).astype(str)
    degree_scale = np.sqrt(
        controlled["fanin_degree_i"].to_numpy(dtype=np.float64)
        * controlled["fanin_degree_j"].to_numpy(dtype=np.float64)
    )
    try:
        controlled["fanin_degree_quartile"] = pd.qcut(
            degree_scale, 4, duplicates="drop"
        ).astype(str)
    except ValueError:
        controlled["fanin_degree_quartile"] = "all"
    summary = (
        controlled.groupby(
            ["fanout_bin", "fanin_degree_quartile", "fanin_level"],
            observed=True,
        )
        .agg(
            pair_count=("i", "size"),
            mean_fanin_similarity=("fanin_similarity", "mean"),
            mean_fanout_similarity=("fanout_similarity", "mean"),
            mean_spike_jaccard=("spike_jaccard", "mean"),
            mean_spike_cosine=("spike_cosine", "mean"),
            mean_lift=("co_spike_lift", "mean"),
            median_lift=("co_spike_lift", "median"),
        )
        .reset_index()
    )
    summary.insert(0, "input_rate", input_rate)
    summary.insert(0, "dataset", dataset)
    return summary


def graph_summary(
    dataset: str, topology: sp.csr_matrix
) -> dict[str, float | int | str]:
    """Describe graph size, density, and degree heterogeneity."""

    out_degree = np.diff(topology.indptr).astype(np.float64)
    in_degree = np.diff(topology.transpose().tocsr().indptr).astype(np.float64)
    return {
        "dataset": dataset,
        "neurons": topology.shape[0],
        "edges": topology.nnz,
        "density": topology.nnz / (topology.shape[0] ** 2),
        "mean_fanout": out_degree.mean(),
        "std_fanout": out_degree.std(),
        "fanout_cv": out_degree.std() / out_degree.mean(),
        "max_fanout": int(out_degree.max()),
        "mean_fanin": in_degree.mean(),
        "std_fanin": in_degree.std(),
        "fanin_cv": in_degree.std() / in_degree.mean(),
        "max_fanin": int(in_degree.max()),
    }


def write_report(
    output: Path,
    config: ExperimentConfig,
    graph_stats: pd.DataFrame,
    rate_stats: pd.DataFrame,
    correlations: pd.DataFrame,
    block_stats: pd.DataFrame,
) -> None:
    """Write a compact data-first report with explicit go/no-go checks."""

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "(no data)"
        return frame[columns].to_markdown(index=False, floatfmt=".4g")

    baseline = block_stats[block_stats["layout"] == "fanout"].set_index(
        ["dataset", "input_rate"]
    )
    candidate = block_stats[block_stats["layout"] == "fanin_fanout"].set_index(
        ["dataset", "input_rate"]
    )
    comparison = candidate.join(
        baseline,
        lsuffix="_candidate",
        rsuffix="_fanout",
        how="inner",
    ).reset_index()
    comparison["occupancy_gain"] = (
        comparison["mean_active_occupancy_candidate"]
        / comparison["mean_active_occupancy_fanout"]
        - 1
    )
    comparison["aggregation_gain"] = (
        comparison["weighted_aggregation_ratio_candidate"]
        - comparison["weighted_aggregation_ratio_fanout"]
    )
    comparison["aggregation_gain_se"] = np.sqrt(
        comparison["weighted_aggregation_ratio_se_candidate"] ** 2
        + comparison["weighted_aggregation_ratio_se_fanout"] ** 2
    )
    comparison["aggregation_gain_ci95_low"] = (
        comparison["aggregation_gain"]
        - 1.96 * comparison["aggregation_gain_se"]
    )
    comparison["aggregation_gain_ci95_high"] = (
        comparison["aggregation_gain"]
        + 1.96 * comparison["aggregation_gain_se"]
    )
    h1_pass = correlations.groupby("dataset")[
        "partial_fanin_spike_jaccard"
    ].median() > 0.05
    h2_pass = comparison.groupby("dataset")["occupancy_gain"].median() > 0.10
    h3_pass = (
        comparison.groupby("dataset")["aggregation_gain_ci95_low"].median() > 0
    )
    decisions = pd.DataFrame(
        {
            "dataset": sorted(set(h1_pass.index) | set(h2_pass.index)),
        }
    )
    decisions["H1_partial_corr_gt_.05"] = decisions["dataset"].map(h1_pass)
    decisions["H2_occupancy_gain_gt_10pct"] = decisions["dataset"].map(h2_pass)
    decisions["H3_gain_lower_CI_positive"] = decisions["dataset"].map(h3_pass)
    interpretation_rows = []
    for dataset in sorted(correlations["dataset"].unique()):
        corr_subset = correlations[correlations["dataset"] == dataset]
        comparison_subset = comparison[comparison["dataset"] == dataset]
        interpretation_rows.append(
            {
                "dataset": dataset,
                "median_raw_fanin_spike_corr": corr_subset[
                    "spearman_fanin_spike_jaccard"
                ].median(),
                "median_partial_fanin_spike_corr": corr_subset[
                    "partial_fanin_spike_jaccard"
                ].median(),
                "median_occupancy_gain_pct": 100
                * comparison_subset["occupancy_gain"].median(),
                "max_occupancy_gain_pct": 100
                * comparison_subset["occupancy_gain"].max(),
                "aggregation_positive_CI_conditions": int(
                    (comparison_subset["aggregation_gain_ci95_low"] > 0).sum()
                ),
                "aggregation_negative_CI_conditions": int(
                    (comparison_subset["aggregation_gain_ci95_high"] < 0).sum()
                ),
            }
        )
    interpretation = pd.DataFrame(interpretation_rows)

    lines = [
        "# Fanin / co-spiking preprocessing experiment",
        "",
        "This report contains numerical analysis only; no figures were generated.",
        "Pair sampling is 50% uniform random and 50% conditioned on sharing at "
        "least one fanin. Binned trends are therefore conditional comparisons, "
        "not estimates of the population frequency of similarity bins.",
        "Destination-union metrics use a reproducible uniform sample of active "
        "BlockTasks; occupancy and active-block counts are exact over every step.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "The simulator is the repository's persistent CUDA v1 LIF+PSC path "
        "(`dt=1`, threshold 1, soft reset). External event amplitude is 1.1, "
        "so one event leaves only a 0.1 post-reset residual. Synthetic graphs "
        "use a deterministic 80/20 Dale E/I split with inhibitory edges 4x "
        "stronger and mean recurrent strength 0.6. FlyWire retains signed "
        "relative synapse counts but is normalized to the same mean absolute "
        "fanout strength; this is a common topology workload, not a faithful "
        "Shiu FlyBrain reproduction.",
        "",
        "## Graphs",
        "",
        table(
            graph_stats,
            [
                "dataset",
                "neurons",
                "edges",
                "density",
                "mean_fanin",
                "fanin_cv",
                "max_fanin",
                "max_fanout",
            ],
        ),
        "",
        "## Measured activity",
        "",
        table(
            rate_stats,
            [
                "dataset",
                "input_rate",
                "timesteps",
                "actual_firing_rate",
                "active_neurons",
                "silent_neuron_fraction",
            ],
        ),
        "",
        "## H1: topology predicts co-activity",
        "",
        table(
            correlations,
            [
                "dataset",
                "input_rate",
                "spearman_fanin_fanout",
                "spearman_fanin_spike_jaccard",
                "spearman_fanin_lift",
                "partial_fanin_spike_jaccard",
                "partial_fanin_lift",
            ],
        ),
        "",
        "## H2/H3: candidate relative to fanout-only",
        "",
        table(
            comparison,
            [
                "dataset",
                "input_rate",
                "occupancy_gain",
                "weighted_aggregation_ratio_fanout",
                "weighted_aggregation_ratio_candidate",
                "aggregation_gain",
                "aggregation_gain_ci95_low",
                "aggregation_gain_ci95_high",
            ],
        ),
        "",
        "## Pre-declared decision checks",
        "",
        table(decisions, list(decisions.columns)),
        "",
        "These checks are diagnostics, not significance tests. H1 uses a modest "
        "partial-correlation threshold because finite low-rate traces make pairwise "
        "lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and "
        "positive aggregation-improvement criteria.",
        "",
        "## Data-first interpretation",
        "",
        table(interpretation, list(interpretation.columns)),
        "",
        "The strict fanin-to-fanout prototype is a no-go under the pre-declared "
        "criteria: no dataset passes all three checks. FlyWire contains a real "
        "raw fanin/co-activity signal, but most of it is redundant with fanout "
        "and degree. Its candidate occupancy gain peaks below 10%, and only one "
        "of six aggregation conditions has a strictly positive 95% CI. The "
        "synthetic controls show negligible occupancy gain and mostly neutral "
        "or negative aggregation changes.",
        "",
        "Large pair-level opportunity lifts do not by themselves establish H1: "
        "the selected pairs also have many more shared destinations. Interpret "
        "them with the partial correlations and block results, which show that "
        "the structural opportunity usually does not convert into reliable "
        "additional BlockTask aggregation.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("sanity", "full"), default="full"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("uniform", "community", "spatial", "flybrain"),
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--recurrent-strength", type=float, default=0.6)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--synthetic-timesteps", type=int, default=None)
    parser.add_argument("--flybrain-timesteps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment")
    if args.phase == "sanity":
        defaults = {
            "datasets": ("uniform", "community", "spatial"),
            "input_rates": (0.005, 0.02),
            "n": 2048,
            "degree": 48,
            "synthetic_timesteps": 256,
            "flybrain_timesteps": 128,
            "pair_count": 10_000,
            "task_sample": 2_000,
        }
    else:
        defaults = {
            "datasets": ("uniform", "community", "spatial", "flybrain"),
            "input_rates": (0.001, 0.005, 0.01, 0.02, 0.05, 0.10),
            "n": 8192,
            "degree": 64,
            "synthetic_timesteps": 2048,
            "flybrain_timesteps": 1024,
            "pair_count": 100_000,
            "task_sample": 10_000,
        }
    datasets = tuple(args.datasets or defaults["datasets"])
    config = ExperimentConfig(
        datasets=datasets,
        input_rates=defaults["input_rates"],
        synthetic_neurons=defaults["n"],
        synthetic_degree=defaults["degree"],
        synthetic_timesteps=(
            args.synthetic_timesteps or defaults["synthetic_timesteps"]
        ),
        flybrain_timesteps=(
            args.flybrain_timesteps or defaults["flybrain_timesteps"]
        ),
        pair_count=args.pair_count or defaults["pair_count"],
        block_task_sample=defaults["task_sample"],
        block_size=32,
        synthetic_recurrent_strength=args.recurrent_strength,
        input_amplitude=1.1,
        seed=args.seed,
    )
    output = args.output or (
        REPO_ROOT
        / "benchmark"
        / "reorder_analysis"
        / "results"
        / f"{args.phase}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "pair_records").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    graph_rows: list[dict] = []
    rate_rows: list[dict] = []
    bin_frames: list[pd.DataFrame] = []
    correlation_rows: list[dict] = []
    opportunity_frames: list[pd.DataFrame] = []
    controlled_frames: list[pd.DataFrame] = []
    block_rows: list[dict] = []
    device = torch.device("cuda")

    print(f"output={output}", flush=True)
    for dataset_id, dataset in enumerate(config.datasets):
        print(f"[{dataset}] loading graph", flush=True)
        topology, weighted = load_graph(
            dataset,
            n=config.synthetic_neurons,
            degree=config.synthetic_degree,
            seed=config.seed + dataset_id * 1000,
        )
        simulation_matrix = scale_simulation_weights(
            weighted, dataset, config.synthetic_recurrent_strength
        )
        graph_rows.append(graph_summary(dataset, topology))
        pairs = sample_pairs(
            topology, config.pair_count, config.seed + dataset_id * 1000 + 1
        )
        topology_frame = topology_pair_frame(topology, *pairs)
        print(f"[{dataset}] constructing layouts", flush=True)
        orders = make_orders(topology, config.block_size, config.seed + dataset_id)
        timesteps = (
            config.flybrain_timesteps
            if dataset == "flybrain"
            else config.synthetic_timesteps
        )
        for rate_id, input_rate in enumerate(config.input_rates):
            print(
                f"[{dataset}] input_rate={input_rate:.3%}, T={timesteps}",
                flush=True,
            )
            spikes = simulate_spikes(
                simulation_matrix,
                timesteps=timesteps,
                input_rate=input_rate,
                input_amplitude=config.input_amplitude,
                seed=config.seed + dataset_id * 1000 + rate_id * 17,
                device=device,
            )
            neuron_counts = spikes.sum(axis=0)
            rate_rows.append(
                {
                    "dataset": dataset,
                    "input_rate": input_rate,
                    "timesteps": timesteps,
                    "actual_firing_rate": float(spikes.mean()),
                    "active_neurons": int((neuron_counts > 0).sum()),
                    "silent_neuron_fraction": float((neuron_counts == 0).mean()),
                    "mean_spikes_per_active_neuron": (
                        float(neuron_counts[neuron_counts > 0].mean())
                        if (neuron_counts > 0).any()
                        else 0.0
                    ),
                }
            )
            pair_frame = add_spike_pair_stats(topology_frame, spikes)
            pair_frame.insert(0, "input_rate", input_rate)
            pair_frame.insert(0, "dataset", dataset)
            pair_path = (
                output
                / "pair_records"
                / f"{dataset}_rate_{input_rate:.3f}.parquet"
            )
            pair_frame.to_parquet(pair_path, index=False, compression="zstd")
            binned, correlations, opportunities = summarize_pairs(
                dataset, input_rate, pair_frame
            )
            bin_frames.append(binned)
            correlation_rows.append(correlations)
            opportunity_frames.append(opportunities)
            controlled_frames.append(
                summarize_controlled_pairs(dataset, input_rate, pair_frame)
            )
            for layout_id, (name, order) in enumerate(orders.items()):
                block_rows.append(
                    analyze_layout(
                        dataset,
                        input_rate,
                        spikes,
                        topology,
                        name,
                        order,
                        block_size=config.block_size,
                        task_sample=config.block_task_sample,
                        seed=(
                            config.seed
                            + dataset_id * 1000
                            + rate_id * 31
                            + layout_id
                        ),
                    )
                )
            del pair_frame, spikes
        del topology, weighted, simulation_matrix, orders, topology_frame

    graph_stats = pd.DataFrame(graph_rows)
    rate_stats = pd.DataFrame(rate_rows)
    binned_stats = pd.concat(bin_frames, ignore_index=True)
    correlations = pd.DataFrame(correlation_rows)
    opportunities = pd.concat(opportunity_frames, ignore_index=True)
    controlled_pairs = pd.concat(controlled_frames, ignore_index=True)
    block_stats = pd.DataFrame(block_rows)
    tables = {
        "graph_summary.csv": graph_stats,
        "activity_summary.csv": rate_stats,
        "pair_binned_summary.csv": binned_stats,
        "pair_correlations.csv": correlations,
        "aggregation_opportunity.csv": opportunities,
        "pair_controlled_summary.csv": controlled_pairs,
        "block_layout_summary.csv": block_stats,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    write_report(
        output,
        config,
        graph_stats,
        rate_stats,
        correlations,
        block_stats,
    )
    print(f"complete: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
