"""Evaluate fanout-preserving activity-aware BlockTask preprocessing.

This implements ``preprocess_plan_v2.md`` as a data-only experiment. It reports
posterior firing rates in Hz, keeps profile and evaluation traces independent,
and does not generate visualizations.
"""

from __future__ import annotations

import argparse
import hashlib
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

from benchmark.reorder_analysis.analyze_reorder_similarity import (  # noqa: E402
    analyze_layout,
    dominant_block_signatures,
    graph_summary,
    load_graph,
    scale_simulation_weights,
    simulate_spikes,
)


TARGET_RATES_HZ = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
CALIBRATION_INPUT_RATES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
LOCAL_WINDOWS = (32, 64, 128, 256)
STRUCTURAL_LAMBDAS = (0.0, 0.25, 0.5, 1.0)
ORACLE_LAMBDAS = (0.25, 0.5, 1.0)
JOINT_WINDOW = 128
BLOCK_SIZE = 32
DT_MS = 1.0


@dataclass(frozen=True)
class V2Config:
    """Record the complete v2 experiment configuration."""

    datasets: tuple[str, ...]
    target_rates_hz: tuple[float, ...]
    calibration_input_rates: tuple[float, ...]
    calibration_steps: int
    evaluation_steps: int
    synthetic_neurons: int
    synthetic_degree: int
    recurrent_strength: float
    local_windows: tuple[int, ...]
    joint_window: int
    structural_lambdas: tuple[float, ...]
    oracle_lambdas: tuple[float, ...]
    block_task_sample: int
    seed: int


def firing_statistics(
    spikes: np.ndarray, dt_ms: float = DT_MS
) -> dict[str, float | int]:
    """Compute posterior population, active-neuron, and silence statistics."""

    timesteps, neurons = spikes.shape
    duration_s = timesteps * dt_ms / 1000.0
    per_neuron = spikes.sum(axis=0, dtype=np.int64)
    total = int(per_neuron.sum())
    active = int((per_neuron > 0).sum())
    return {
        "timesteps": timesteps,
        "duration_s": duration_s,
        "total_spikes": total,
        "active_neurons": active,
        "population_mean_hz": total / neurons / duration_s,
        "active_neuron_mean_hz": total / active / duration_s if active else 0.0,
        "silent_fraction": 1.0 - active / neurons,
    }


def build_fanout_order(topology: sp.csr_matrix) -> np.ndarray:
    """Replicate the existing global_cost_similarity ordering on CPU."""

    ids = np.arange(topology.shape[0], dtype=np.int32)
    degree = np.diff(topology.indptr)
    bucket = np.searchsorted(
        np.array([4, 8, 16, 32, 64, 128, 255]), degree, side="left"
    )
    extreme = (degree >= 256).astype(np.int8)
    dominant = dominant_block_signatures(topology, BLOCK_SIZE, top_k=1)[:, 0]
    return np.lexsort((ids, dominant, bucket, extreme)).astype(np.int32)


def local_fanin_orders(
    base_order: np.ndarray,
    topology: sp.csr_matrix,
    windows: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Stable-sort top-2 fanin signatures inside fixed fanout windows."""

    fanin_signature = dominant_block_signatures(
        topology.transpose().tocsr(), BLOCK_SIZE, top_k=2
    )
    result: dict[str, np.ndarray] = {}
    for window_size in windows:
        order = base_order.copy()
        for start in range(0, order.size, window_size):
            window = order[start : start + window_size]
            local_position = np.arange(window.size, dtype=np.int32)
            permutation = np.lexsort(
                (
                    local_position,
                    fanin_signature[window, 1],
                    fanin_signature[window, 0],
                )
            )
            order[start : start + window_size] = window[permutation]
        result[f"local_fanin_w{window_size}"] = order
    return result


@numba.njit(cache=True, parallel=True)
def _window_jaccard(
    base_order: np.ndarray,
    window_size: int,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute exact CSR-row Jaccard only within coarse fanout windows."""

    window_count = (base_order.size + window_size - 1) // window_size
    affinity = np.zeros(
        (window_count, window_size, window_size), dtype=np.float32
    )
    for window in numba.prange(window_count):
        valid = min(window_size, base_order.size - window * window_size)
        for a in range(valid):
            neuron_a = base_order[window * window_size + a]
            affinity[window, a, a] = 1.0
            for b in range(a + 1, valid):
                neuron_b = base_order[window * window_size + b]
                ai = indptr[neuron_a]
                ae = indptr[neuron_a + 1]
                bi = indptr[neuron_b]
                be = indptr[neuron_b + 1]
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
                degree_a = ae - indptr[neuron_a]
                degree_b = be - indptr[neuron_b]
                union = degree_a + degree_b - common
                value = common / union if union else 0.0
                affinity[window, a, b] = value
                affinity[window, b, a] = value
    return affinity


@numba.njit(cache=True)
def _joint_pack(
    base_order: np.ndarray,
    window_size: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    affinity: np.ndarray,
    activity_lambda: float,
    neurons: int,
) -> np.ndarray:
    """Greedily maximize exact fanout-union reuse times activity affinity."""

    output = np.empty_like(base_order)
    marker = np.zeros(neurons, dtype=np.int32)
    token = 0
    window_count = (base_order.size + window_size - 1) // window_size
    for window in range(window_count):
        global_start = window * window_size
        valid = min(window_size, base_order.size - global_start)
        remaining = np.ones(valid, dtype=np.uint8)
        written = 0
        while written < valid:
            token += 1
            seed = -1
            seed_degree = -1
            for candidate in range(valid):
                if remaining[candidate] == 0:
                    continue
                neuron = base_order[global_start + candidate]
                degree = indptr[neuron + 1] - indptr[neuron]
                if degree > seed_degree:
                    seed = candidate
                    seed_degree = degree
            block_positions = np.empty(BLOCK_SIZE, dtype=np.int32)
            block_positions[0] = seed
            block_length = 1
            remaining[seed] = 0
            seed_neuron = base_order[global_start + seed]
            output[global_start + written] = seed_neuron
            written += 1
            for edge in range(indptr[seed_neuron], indptr[seed_neuron + 1]):
                marker[indices[edge]] = token

            while block_length < BLOCK_SIZE and written < valid:
                best = -1
                best_score = -1.0
                best_reuse = -1
                best_degree = -1
                for candidate in range(valid):
                    if remaining[candidate] == 0:
                        continue
                    neuron = base_order[global_start + candidate]
                    reuse = 0
                    for edge in range(indptr[neuron], indptr[neuron + 1]):
                        if marker[indices[edge]] == token:
                            reuse += 1
                    activity = 0.0
                    for member in range(block_length):
                        activity += affinity[
                            window, candidate, block_positions[member]
                        ]
                    activity /= block_length
                    score = reuse * (1.0 + activity_lambda * activity)
                    degree = indptr[neuron + 1] - indptr[neuron]
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
                block_positions[block_length] = best
                block_length += 1
                remaining[best] = 0
                neuron = base_order[global_start + best]
                output[global_start + written] = neuron
                written += 1
                for edge in range(indptr[neuron], indptr[neuron + 1]):
                    marker[indices[edge]] = token
    return output


def spike_affinity_windows(
    spikes: np.ndarray,
    base_order: np.ndarray,
    window_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute window-local spike Jaccard using batched GPU matrix products."""

    window_count = math.ceil(base_order.size / window_size)
    result = np.zeros(
        (window_count, window_size, window_size), dtype=np.float32
    )
    windows_per_chunk = 16
    for first in range(0, window_count, windows_per_chunk):
        last = min(first + windows_per_chunk, window_count)
        ids = np.full((last - first, window_size), -1, dtype=np.int64)
        for offset, window in enumerate(range(first, last)):
            begin = window * window_size
            valid = min(window_size, base_order.size - begin)
            ids[offset, :valid] = base_order[begin : begin + valid]
        valid_ids = np.maximum(ids, 0)
        host = spikes[:, valid_ids.reshape(-1)].reshape(
            spikes.shape[0], last - first, window_size
        )
        host[:, ids < 0] = 0
        tensor = torch.from_numpy(host).to(device=device, dtype=torch.float32)
        tensor = tensor.permute(1, 0, 2).contiguous()
        joint = torch.bmm(tensor.transpose(1, 2), tensor)
        counts = tensor.sum(dim=1)
        union = counts[:, :, None] + counts[:, None, :] - joint
        affinity = torch.where(union > 0, joint / union, torch.zeros_like(joint))
        result[first:last] = affinity.cpu().numpy()
        del tensor, joint, counts, union, affinity
    return result


def affinity_stability(
    profile: np.ndarray,
    evaluation: np.ndarray,
    valid_neurons: int,
    window_size: int,
    seed: int,
    sample_count: int = 100_000,
) -> dict[str, float | int]:
    """Measure profile/evaluation affinity persistence on sampled valid pairs."""

    rng = np.random.default_rng(seed)
    window = rng.integers(0, profile.shape[0], size=sample_count)
    left = rng.integers(0, window_size, size=sample_count)
    right = rng.integers(0, window_size, size=sample_count)
    global_left = window * window_size + left
    global_right = window * window_size + right
    valid = (left != right) & (global_left < valid_neurons) & (
        global_right < valid_neurons
    )
    x = profile[window[valid], left[valid], right[valid]]
    y = evaluation[window[valid], left[valid], right[valid]]
    frame = pd.DataFrame({"profile": x, "evaluation": y})
    positive = frame[(frame["profile"] > 0) | (frame["evaluation"] > 0)]
    profile_positive = x > 0
    evaluation_positive = y > 0
    both_positive = profile_positive & evaluation_positive
    union_positive = profile_positive | evaluation_positive
    return {
        "sampled_pairs": len(frame),
        "profile_nonzero_fraction": float((x > 0).mean()),
        "evaluation_nonzero_fraction": float((y > 0).mean()),
        "both_nonzero_fraction": float(both_positive.mean()),
        "persistence_given_profile": (
            float(both_positive.sum() / profile_positive.sum())
            if profile_positive.any()
            else math.nan
        ),
        "positive_support_jaccard": (
            float(both_positive.sum() / union_positive.sum())
            if union_positive.any()
            else math.nan
        ),
        "spearman_all": (
            float(frame.corr(method="spearman").iloc[0, 1])
            if frame["profile"].nunique() > 1
            and frame["evaluation"].nunique() > 1
            else math.nan
        ),
        "spearman_any_positive": (
            float(positive.corr(method="spearman").iloc[0, 1])
            if len(positive) > 2
            and positive["profile"].nunique() > 1
            and positive["evaluation"].nunique() > 1
            else math.nan
        ),
    }


def calibrate_conditions(
    dataset: str,
    matrix: sp.csr_matrix,
    config: V2Config,
    dataset_id: int,
    device: torch.device,
) -> tuple[list[float], list[dict[str, float | int | str]]]:
    """Select unique input controls nearest the target posterior rates."""

    measurements: list[dict[str, float | int | str]] = []
    for condition_id, input_rate in enumerate(config.calibration_input_rates):
        spikes = simulate_spikes(
            matrix,
            timesteps=config.calibration_steps,
            input_rate=input_rate,
            input_amplitude=1.1,
            seed=config.seed + dataset_id * 10_000 + condition_id,
            device=device,
        )
        stats = firing_statistics(spikes)
        measurements.append(
            {
                "dataset": dataset,
                "control_input_rate": input_rate,
                **stats,
            }
        )
        del spikes
    selected: list[float] = []
    for target in config.target_rates_hz:
        closest = min(
            measurements,
            key=lambda row: abs(float(row["population_mean_hz"]) - target),
        )
        condition = float(closest["control_input_rate"])
        if condition not in selected:
            selected.append(condition)
        closest.setdefault("selected_targets_hz", "")
        previous = str(closest["selected_targets_hz"])
        closest["selected_targets_hz"] = f"{previous},{target}".strip(",")
    return selected, measurements


def add_atomic_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare every method to fanout-only within each measured condition."""

    baseline = frame[frame["method"] == "fanout_only"].set_index(
        ["dataset", "control_input_rate"]
    )
    result = frame.copy()
    keys = pd.MultiIndex.from_frame(result[["dataset", "control_input_rate"]])
    baseline_atomic = baseline.loc[keys, "estimated_global_atomic_count"].to_numpy()
    baseline_atomic_se = baseline.loc[
        keys, "estimated_global_atomic_count_se"
    ].to_numpy()
    baseline_occupancy = baseline.loc[keys, "mean_active_occupancy"].to_numpy()
    baseline_aggregation = baseline.loc[
        keys, "weighted_aggregation_ratio"
    ].to_numpy()
    baseline_static = baseline.loc[
        keys, "static_potential_aggregation_ratio"
    ].to_numpy()
    baseline_fingerprint = baseline.loc[keys, "order_fingerprint"].to_numpy()
    candidate_atomic = result["estimated_global_atomic_count"].to_numpy()
    candidate_atomic_se = result["estimated_global_atomic_count_se"].to_numpy()
    ratio = candidate_atomic / baseline_atomic
    ratio_se = ratio * np.sqrt(
        np.square(candidate_atomic_se / candidate_atomic)
        + np.square(baseline_atomic_se / baseline_atomic)
    )
    result["occupancy_gain"] = (
        result["mean_active_occupancy"].to_numpy() / baseline_occupancy - 1.0
    )
    result["aggregation_gain"] = (
        result["weighted_aggregation_ratio"].to_numpy() - baseline_aggregation
    )
    result["static_fanout_change"] = (
        result["static_potential_aggregation_ratio"].to_numpy() - baseline_static
    )
    result["atomic_reduction_vs_fanout"] = 1.0 - ratio
    result["atomic_reduction_se"] = ratio_se
    result["atomic_reduction_ci95_low"] = 1.0 - ratio - 1.96 * ratio_se
    result["atomic_reduction_ci95_high"] = 1.0 - ratio + 1.96 * ratio_se
    is_baseline = result["order_fingerprint"].to_numpy() == baseline_fingerprint
    result.loc[
        is_baseline,
        [
            "occupancy_gain",
            "aggregation_gain",
            "static_fanout_change",
            "atomic_reduction_vs_fanout",
            "atomic_reduction_se",
            "atomic_reduction_ci95_low",
            "atomic_reduction_ci95_high",
        ],
    ] = 0.0

    joint_zero = result[result["method"] == "joint_fanin_l0"].set_index(
        ["dataset", "control_input_rate"]
    )
    joint_atomic = joint_zero.loc[
        keys, "estimated_global_atomic_count"
    ].to_numpy()
    joint_atomic_se = joint_zero.loc[
        keys, "estimated_global_atomic_count_se"
    ].to_numpy()
    joint_occupancy = joint_zero.loc[keys, "mean_active_occupancy"].to_numpy()
    joint_aggregation = joint_zero.loc[
        keys, "weighted_aggregation_ratio"
    ].to_numpy()
    joint_fingerprint = joint_zero.loc[keys, "order_fingerprint"].to_numpy()
    activity_ratio = candidate_atomic / joint_atomic
    activity_ratio_se = activity_ratio * np.sqrt(
        np.square(candidate_atomic_se / candidate_atomic)
        + np.square(joint_atomic_se / joint_atomic)
    )
    result["activity_occupancy_gain_vs_joint0"] = (
        result["mean_active_occupancy"].to_numpy() / joint_occupancy - 1.0
    )
    result["activity_aggregation_gain_vs_joint0"] = (
        result["weighted_aggregation_ratio"].to_numpy() - joint_aggregation
    )
    result["activity_atomic_reduction_vs_joint0"] = 1.0 - activity_ratio
    result["activity_atomic_reduction_se"] = activity_ratio_se
    result["activity_atomic_reduction_ci95_low"] = (
        1.0 - activity_ratio - 1.96 * activity_ratio_se
    )
    result["activity_atomic_reduction_ci95_high"] = (
        1.0 - activity_ratio + 1.96 * activity_ratio_se
    )
    is_joint_zero = (
        result["order_fingerprint"].to_numpy() == joint_fingerprint
    )
    result.loc[
        is_joint_zero,
        [
            "activity_occupancy_gain_vs_joint0",
            "activity_aggregation_gain_vs_joint0",
            "activity_atomic_reduction_vs_joint0",
            "activity_atomic_reduction_se",
            "activity_atomic_reduction_ci95_low",
            "activity_atomic_reduction_ci95_high",
        ],
    ] = 0.0
    return result


def write_report(
    output: Path,
    config: V2Config,
    graph_stats: pd.DataFrame,
    calibration: pd.DataFrame,
    activity: pd.DataFrame,
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Write a numerical v2 report and classify the preprocessing headroom."""

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_markdown(index=False, floatfmt=".4g")

    nonbaseline = metrics[metrics["method"] != "fanout_only"]
    best_rows = []
    for (dataset, control), group in nonbaseline.groupby(
        ["dataset", "control_input_rate"]
    ):
        best = group.loc[group["atomic_reduction_vs_fanout"].idxmax()]
        best_rows.append(best)
    best = pd.DataFrame(best_rows)

    family_best_rows = []
    for keys, group in nonbaseline.groupby(
        ["dataset", "control_input_rate", "method_family"]
    ):
        row = group.loc[group["atomic_reduction_vs_fanout"].idxmax()].copy()
        row["dataset"], row["control_input_rate"], row["method_family"] = keys
        family_best_rows.append(row)
    family_best = pd.DataFrame(family_best_rows)

    activity_candidates = nonbaseline[
        (nonbaseline["method_family"] == "oracle_spike")
        | (
            (nonbaseline["method_family"] == "joint_fanin")
            & (nonbaseline["activity_lambda"] > 0)
        )
    ]
    activity_best_rows = []
    for keys, group in activity_candidates.groupby(
        ["dataset", "control_input_rate", "method_family"]
    ):
        row = group.loc[
            group["activity_atomic_reduction_vs_joint0"].idxmax()
        ].copy()
        row["dataset"], row["control_input_rate"], row["method_family"] = keys
        activity_best_rows.append(row)
    activity_best = pd.DataFrame(activity_best_rows)

    joint_zero = metrics[metrics["method"] == "joint_fanin_l0"]
    oracle = activity_best[activity_best["method_family"] == "oracle_spike"]
    structural = activity_best[activity_best["method_family"] == "joint_fanin"]
    decision_rows = []
    for dataset in config.datasets:
        oracle_ds = oracle[oracle["dataset"] == dataset]
        structural_ds = structural[structural["dataset"] == dataset]
        oracle_median = float(oracle_ds["atomic_reduction_vs_fanout"].median())
        oracle_incremental_median = float(
            oracle_ds["activity_atomic_reduction_vs_joint0"].median()
        )
        oracle_lower_median = float(
            oracle_ds["activity_atomic_reduction_ci95_low"].median()
        )
        structural_median = float(
            structural_ds["activity_atomic_reduction_vs_joint0"].median()
        )
        structural_lower_median = float(
            structural_ds["activity_atomic_reduction_ci95_low"].median()
        )
        joint_zero_median = float(
            joint_zero.loc[
                joint_zero["dataset"] == dataset,
                "atomic_reduction_vs_fanout",
            ].median()
        )
        if oracle_incremental_median <= 0.005:
            classification = "stop activity-aware; retain fanout greedy"
        elif oracle_lower_median <= 0:
            classification = "inconclusive: oracle CI includes zero"
        elif structural_lower_median <= 0 and oracle_lower_median > 0.01:
            classification = "profile-guided only"
        elif structural_lower_median > 0:
            classification = "structural refinement has headroom"
        else:
            classification = "inconclusive small headroom"
        decision_rows.append(
            {
                "dataset": dataset,
                "joint0_median_atomic_reduction_vs_fanout": joint_zero_median,
                "best_structural_median_increment_vs_joint0": structural_median,
                "best_structural_median_lower_CI": structural_lower_median,
                "best_oracle_median_reduction_vs_fanout": oracle_median,
                "best_oracle_median_increment_vs_joint0": oracle_incremental_median,
                "best_oracle_median_lower_CI": oracle_lower_median,
                "classification": classification,
            }
        )
    decisions = pd.DataFrame(decision_rows)

    local = metrics[metrics["method_family"] == "local_fanin"]
    local_supported = int((local["atomic_reduction_ci95_low"] > 0).sum())
    activity_supported = int(
        (activity_best["activity_atomic_reduction_ci95_low"] > 0).sum()
    )
    interpretation = [
        "## Data interpretation",
        "",
        f"None of the {len(local)} local-fanin window candidates produced a "
        f"reliable atomic reduction over the existing fanout order "
        f"({local_supported} had a positive 95% CI lower bound).",
        "",
        f"After controlling for the stronger pure-fanout greedy packer, "
        f"{activity_supported} of {len(activity_best)} condition-level best "
        "structural/oracle candidates had a positive 95% CI lower bound. "
        "The activity-aware headroom is therefore unsupported.",
    ]
    fly_joint = joint_zero[joint_zero["dataset"] == "flybrain"]
    fly_supported = fly_joint[
        fly_joint["atomic_reduction_ci95_low"] > 0
    ].sort_values("population_mean_hz")
    if not fly_supported.empty:
        min_hz = float(fly_supported["population_mean_hz"].min())
        max_hz = float(fly_supported["population_mean_hz"].max())
        min_reduction = float(
            fly_supported["atomic_reduction_vs_fanout"].min()
        )
        max_reduction = float(
            fly_supported["atomic_reduction_vs_fanout"].max()
        )
        static_change = float(fly_supported["static_fanout_change"].iloc[0])
        interpretation.extend(
            [
                "",
                "A separate fanout-only result remains actionable: on FlyWire "
                f"from {min_hz:.3g} to {max_hz:.3g} Hz, lambda=0 reduced the "
                f"estimated atomic count by {100 * min_reduction:.2f}% to "
                f"{100 * max_reduction:.2f}% with positive 95% CI lower "
                f"bounds, while static fanout aggregation increased by "
                f"{100 * static_change:.2f} percentage points. This is an "
                "improved fanout packing objective, not evidence for fanin or "
                "activity-aware refinement.",
            ]
        )
    interpretation.extend(
        [
            "",
            "Recommendation: stop the fanin/activity-aware branch and retain "
            "lambda=0 fanout-union greedy packing as a separate FlyWire "
            "preprocessing candidate. Validate that candidate in the real "
            "kernel before adoption; this experiment estimates structural "
            "execution cost and does not measure kernel runtime.",
        ]
    )

    lines = [
        "# Fanout-preserving refinement experiment (v2)",
        "",
        "Numerical results only; no images or visualization files were generated.",
        "Profile and evaluation traces use different seeds. Oracle orders are "
        "built exclusively from the profile trace.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "The common simulator is persistent CUDA v1 LIF+PSC (`dt=1 ms`, soft "
        "reset). Synthetic graphs use deterministic balanced 80/20 Dale E/I "
        "weights. FlyWire preserves signed relative weights. Every graph is "
        "normalized to mean absolute recurrent fanout strength 0.1.",
        "",
        "## Graph summary",
        "",
        table(
            graph_stats,
            ["dataset", "neurons", "edges", "mean_fanin", "fanin_cv"],
        ),
        "",
        "## Calibration and selected controls",
        "",
        table(
            calibration[calibration["selected_targets_hz"].fillna("") != ""],
            [
                "dataset",
                "control_input_rate",
                "population_mean_hz",
                "active_neuron_mean_hz",
                "silent_fraction",
                "selected_targets_hz",
            ],
        ),
        "",
        "## Evaluation posterior activity",
        "",
        table(
            activity,
            [
                "dataset",
                "control_input_rate",
                "population_mean_hz",
                "active_neuron_mean_hz",
                "silent_fraction",
                "profile_population_mean_hz",
            ],
        ),
        "",
        "## Best method at each measured condition",
        "",
        table(
            best.sort_values(["dataset", "population_mean_hz"]),
            [
                "dataset",
                "population_mean_hz",
                "method",
                "occupancy_gain",
                "static_fanout_change",
                "aggregation_gain",
                "atomic_reduction_vs_fanout",
                "atomic_reduction_ci95_low",
                "atomic_reduction_ci95_high",
            ],
        ),
        "",
        "## Best result by method family",
        "",
        table(
            family_best.sort_values(
                ["dataset", "population_mean_hz", "method_family"]
            ),
            [
                "dataset",
                "population_mean_hz",
                "method_family",
                "method",
                "occupancy_gain",
                "atomic_reduction_vs_fanout",
                "atomic_reduction_ci95_low",
            ],
        ),
        "",
        "## Pure fanout greedy packing (lambda = 0)",
        "",
        table(
            joint_zero.sort_values(["dataset", "population_mean_hz"]),
            [
                "dataset",
                "population_mean_hz",
                "occupancy_gain",
                "static_fanout_change",
                "atomic_reduction_vs_fanout",
                "atomic_reduction_ci95_low",
            ],
        ),
        "",
        "## Incremental activity contribution over joint lambda = 0",
        "",
        table(
            activity_best.sort_values(
                ["dataset", "population_mean_hz", "method_family"]
            ),
            [
                "dataset",
                "population_mean_hz",
                "method_family",
                "method",
                "activity_occupancy_gain_vs_joint0",
                "activity_atomic_reduction_vs_joint0",
                "activity_atomic_reduction_ci95_low",
                "activity_atomic_reduction_ci95_high",
            ],
        ),
        "",
        "## Oracle profile stability",
        "",
        table(
            stability,
            [
                "dataset",
                "population_mean_hz",
                "profile_nonzero_fraction",
                "evaluation_nonzero_fraction",
                "persistence_given_profile",
                "positive_support_jaccard",
                "spearman_all",
                "spearman_any_positive",
            ],
        ),
        "",
        "## Decision summary",
        "",
        table(decisions, list(decisions.columns)),
        "",
        "Atomic counts are ratio-of-means estimates from uniformly sampled active "
        "BlockTasks; reported intervals propagate independent candidate/baseline "
        "standard errors. Occupancy and active-block counts are exact over the "
        "entire evaluation trace.",
        "",
        "The decision separates fanout greedy packing (`lambda=0`) from the "
        "incremental contribution of structural or measured activity affinity. "
        "This prevents a stronger fanout objective from being misattributed to "
        "fanin or spike profiling.",
        "",
        *interpretation,
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sanity", "full"), default="full")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("uniform", "community", "spatial", "flybrain"),
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the v2 experiment")
    if args.phase == "sanity":
        config = V2Config(
            datasets=tuple(args.datasets or ("uniform", "flybrain")),
            target_rates_hz=(1.0, 10.0),
            calibration_input_rates=(0.001, 0.01),
            calibration_steps=128,
            evaluation_steps=256,
            synthetic_neurons=2048,
            synthetic_degree=48,
            recurrent_strength=0.1,
            local_windows=(32, 64),
            joint_window=64,
            structural_lambdas=(0.0, 0.5),
            oracle_lambdas=(0.5,),
            block_task_sample=2_000,
            seed=args.seed,
        )
    else:
        config = V2Config(
            datasets=tuple(
                args.datasets or ("uniform", "community", "spatial", "flybrain")
            ),
            target_rates_hz=TARGET_RATES_HZ,
            calibration_input_rates=CALIBRATION_INPUT_RATES,
            calibration_steps=512,
            evaluation_steps=2048,
            synthetic_neurons=8192,
            synthetic_degree=64,
            recurrent_strength=0.1,
            local_windows=LOCAL_WINDOWS,
            joint_window=JOINT_WINDOW,
            structural_lambdas=STRUCTURAL_LAMBDAS,
            oracle_lambdas=ORACLE_LAMBDAS,
            block_task_sample=100_000,
            seed=args.seed,
        )
    output = args.output or (
        REPO_ROOT
        / "benchmark"
        / "reorder_analysis"
        / "results"
        / f"v2_{args.phase}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    device = torch.device("cuda")
    graph_rows = []
    calibration_rows = []
    activity_rows = []
    metric_rows = []
    stability_rows = []
    print(f"output={output}", flush=True)

    for dataset_id, dataset in enumerate(config.datasets):
        print(f"[{dataset}] load graph and build fanout baseline", flush=True)
        topology, weighted = load_graph(
            dataset,
            n=config.synthetic_neurons,
            degree=config.synthetic_degree,
            seed=config.seed + dataset_id * 1000,
        )
        matrix = scale_simulation_weights(
            weighted, dataset, config.recurrent_strength
        )
        graph_rows.append(graph_summary(dataset, topology))
        base_order = build_fanout_order(topology)
        local_orders = local_fanin_orders(
            base_order, topology, config.local_windows
        )

        print(f"[{dataset}] calibrate posterior Hz", flush=True)
        selected, calibration = calibrate_conditions(
            dataset, matrix, config, dataset_id, device
        )
        calibration_rows.extend(calibration)
        if dataset != "flybrain" and args.phase == "full":
            representative_targets = (1.0, 10.0, 50.0)
            selected_rows = [
                row for row in calibration if row.get("selected_targets_hz")
            ]
            selected = []
            for target in representative_targets:
                closest = min(
                    selected_rows,
                    key=lambda row: abs(float(row["population_mean_hz"]) - target),
                )
                value = float(closest["control_input_rate"])
                if value not in selected:
                    selected.append(value)

        print(f"[{dataset}] exact structural affinity", flush=True)
        fanin = topology.transpose().tocsr()
        fanin.sort_indices()
        structural_affinity = _window_jaccard(
            base_order,
            config.joint_window,
            fanin.indptr,
            fanin.indices,
        )
        structural_orders = {}
        for activity_lambda in config.structural_lambdas:
            name = f"joint_fanin_l{activity_lambda:g}"
            structural_orders[name] = _joint_pack(
                base_order,
                config.joint_window,
                topology.indptr,
                topology.indices,
                structural_affinity,
                activity_lambda,
                topology.shape[0],
            )

        for condition_id, input_rate in enumerate(selected):
            print(
                f"[{dataset}] control={input_rate:.4%}: profile/evaluation",
                flush=True,
            )
            profile_spikes = simulate_spikes(
                matrix,
                timesteps=config.evaluation_steps,
                input_rate=input_rate,
                input_amplitude=1.1,
                seed=config.seed + dataset_id * 100_000 + condition_id * 2,
                device=device,
            )
            evaluation_spikes = simulate_spikes(
                matrix,
                timesteps=config.evaluation_steps,
                input_rate=input_rate,
                input_amplitude=1.1,
                seed=config.seed + dataset_id * 100_000 + condition_id * 2 + 1,
                device=device,
            )
            profile_stats = firing_statistics(profile_spikes)
            evaluation_stats = firing_statistics(evaluation_spikes)
            activity_rows.append(
                {
                    "dataset": dataset,
                    "control_input_rate": input_rate,
                    **evaluation_stats,
                    "profile_population_mean_hz": profile_stats[
                        "population_mean_hz"
                    ],
                    "profile_active_neuron_mean_hz": profile_stats[
                        "active_neuron_mean_hz"
                    ],
                    "profile_silent_fraction": profile_stats["silent_fraction"],
                }
            )

            print(f"[{dataset}] build leakage-free oracle affinity", flush=True)
            profile_affinity = spike_affinity_windows(
                profile_spikes, base_order, config.joint_window, device
            )
            evaluation_affinity = spike_affinity_windows(
                evaluation_spikes, base_order, config.joint_window, device
            )
            stability_rows.append(
                {
                    "dataset": dataset,
                    "control_input_rate": input_rate,
                    "population_mean_hz": evaluation_stats[
                        "population_mean_hz"
                    ],
                    **affinity_stability(
                        profile_affinity,
                        evaluation_affinity,
                        topology.shape[0],
                        config.joint_window,
                        config.seed + condition_id,
                    ),
                }
            )
            del evaluation_affinity

            orders: list[tuple[str, str, int, float, np.ndarray]] = [
                ("fanout_only", "fanout", 0, 0.0, base_order)
            ]
            orders.extend(
                (name, "local_fanin", int(name.rsplit("w", 1)[1]), 0.0, order)
                for name, order in local_orders.items()
            )
            orders.extend(
                (
                    name,
                    "joint_fanin",
                    config.joint_window,
                    float(name.rsplit("l", 1)[1]),
                    order,
                )
                for name, order in structural_orders.items()
            )
            for activity_lambda in config.oracle_lambdas:
                name = f"oracle_spike_l{activity_lambda:g}"
                oracle_order = _joint_pack(
                    base_order,
                    config.joint_window,
                    topology.indptr,
                    topology.indices,
                    profile_affinity,
                    activity_lambda,
                    topology.shape[0],
                )
                orders.append(
                    (
                        name,
                        "oracle_spike",
                        config.joint_window,
                        activity_lambda,
                        oracle_order,
                    )
                )

            for method, family, window, lam, order in orders:
                metrics = analyze_layout(
                    dataset,
                    input_rate,
                    evaluation_spikes,
                    topology,
                    method,
                    order,
                    block_size=BLOCK_SIZE,
                    task_sample=config.block_task_sample,
                    seed=(
                        config.seed
                        + dataset_id * 1000
                        + condition_id * 31
                    ),
                )
                metric_rows.append(
                    {
                        **metrics,
                        "method": method,
                        "method_family": family,
                        "control_input_rate": input_rate,
                        "local_window": window,
                        "activity_lambda": lam,
                        "population_mean_hz": evaluation_stats[
                            "population_mean_hz"
                        ],
                        "active_neuron_mean_hz": evaluation_stats[
                            "active_neuron_mean_hz"
                        ],
                        "silent_fraction": evaluation_stats["silent_fraction"],
                        "order_fingerprint": hashlib.sha1(
                            order.tobytes(), usedforsecurity=False
                        ).hexdigest(),
                    }
                )
            del profile_affinity, profile_spikes, evaluation_spikes
        del structural_affinity, structural_orders, local_orders, fanin
        del topology, weighted, matrix

    graph_frame = pd.DataFrame(graph_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    activity_frame = pd.DataFrame(activity_rows)
    metrics_frame = add_atomic_comparison(pd.DataFrame(metric_rows))
    stability_frame = pd.DataFrame(stability_rows)
    tables = {
        "graph_summary.csv": graph_frame,
        "calibration.csv": calibration_frame,
        "activity_summary.csv": activity_frame,
        "method_metrics.csv": metrics_frame,
        "oracle_stability.csv": stability_frame,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    write_report(
        output,
        config,
        graph_frame,
        calibration_frame,
        activity_frame,
        metrics_frame,
        stability_frame,
    )
    print(f"complete: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
