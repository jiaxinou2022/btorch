"""Compute graph and spike workload statistics for sparse benchmarks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from btorch.sparse import CSR


@dataclass(frozen=True)
class GraphStats:
    """Summarize source-oriented CSR structure."""

    n: int
    nnz: int
    mean_fanout: float
    max_fanout: int
    fanout_std: float
    fanout_cv: float
    fanout_gini: float
    p50_fanout: float
    p90_fanout: float
    p95_fanout: float
    p99_fanout: float
    unique_posts_ratio: float
    graph_type: str
    seed: int

    def to_dict(self) -> dict[str, int | float | str]:
        """Convert statistics to a CSV/JSON-compatible dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class SpikeStats:
    """Summarize activity and active-edge cost over a fixed spike trace."""

    requested_activity: float
    measured_activity: float
    mean_active_neurons: float
    p95_active_neurons: float
    max_active_neurons: int
    mean_active_edges: float
    p95_active_edges: float
    max_active_edges: int
    mean_collision_ratio: float
    p95_collision_ratio: float
    temporal_cv: float
    burst_factor: float
    collision_method: str

    def to_dict(self) -> dict[str, int | float | str]:
        """Convert statistics to a CSV/JSON-compatible dictionary."""

        return asdict(self)


def _gini(values: np.ndarray) -> float:
    if values.size == 0 or float(values.sum()) == 0.0:
        return 0.0
    ordered = np.sort(values.astype(np.float64, copy=False))
    indices = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(indices * ordered) / np.sum(ordered) - ordered.size - 1)
        / ordered.size
    )


def summarize_graph(
    matrix: CSR,
    *,
    graph_type: str,
    seed: int,
) -> GraphStats:
    """Compute structural statistics without affecting timed execution."""

    fanout = (matrix.indptr[1:] - matrix.indptr[:-1]).detach().cpu().numpy()
    mean = float(fanout.mean()) if fanout.size else 0.0
    std = float(fanout.std()) if fanout.size else 0.0
    nnz = int(matrix.indices.numel())
    unique_posts = int(torch.unique(matrix.indices.detach().cpu()).numel())
    return GraphStats(
        n=matrix.shape[0],
        nnz=nnz,
        mean_fanout=mean,
        max_fanout=int(fanout.max()) if fanout.size else 0,
        fanout_std=std,
        fanout_cv=std / mean if mean else 0.0,
        fanout_gini=_gini(fanout),
        p50_fanout=float(np.percentile(fanout, 50)) if fanout.size else 0.0,
        p90_fanout=float(np.percentile(fanout, 90)) if fanout.size else 0.0,
        p95_fanout=float(np.percentile(fanout, 95)) if fanout.size else 0.0,
        p99_fanout=float(np.percentile(fanout, 99)) if fanout.size else 0.0,
        unique_posts_ratio=unique_posts / max(nnz, 1),
        graph_type=graph_type,
        seed=seed,
    )


def summarize_spike_workload(
    matrix: CSR,
    spikes: torch.Tensor,
    *,
    requested_activity: float,
    exact_collision_edge_limit: int = 10_000_000,
) -> SpikeStats:
    """Compute active-neuron, active-edge, and post-collision statistics."""

    if spikes.ndim == 3:
        if spikes.shape[1] != 1:
            raise ValueError("workload statistics currently require batch size 1")
        spikes_2d = spikes[:, 0]
    elif spikes.ndim == 2:
        spikes_2d = spikes
    else:
        raise ValueError("spikes must have shape (T, N) or (T, 1, N)")
    if spikes_2d.shape[1] != matrix.shape[0]:
        raise ValueError("spike trace neuron count does not match graph")

    active = spikes_2d.detach().to(device="cpu", dtype=torch.bool)
    fanout = (matrix.indptr[1:] - matrix.indptr[:-1]).detach().cpu()
    active_counts = active.sum(dim=1, dtype=torch.int64)
    active_edges = active.to(torch.int64) @ fanout.to(torch.int64)
    total_active_edges = int(active_edges.sum().item())

    collision_ratios = torch.zeros(active.shape[0], dtype=torch.float64)
    exact = (
        total_active_edges <= exact_collision_edge_limit
        and matrix.indices.numel() <= exact_collision_edge_limit
    )
    if exact:
        source = torch.repeat_interleave(
            torch.arange(matrix.shape[0]),
            fanout,
        )
        posts = matrix.indices.detach().cpu()
        for timestep in range(active.shape[0]):
            timestep_posts = posts[active[timestep, source]]
            edge_count = timestep_posts.numel()
            if edge_count:
                unique_count = torch.unique(timestep_posts).numel()
                collision_ratios[timestep] = 1.0 - unique_count / edge_count
        collision_method = "exact"
    else:
        # Deterministically sample timesteps. Active-edge counts remain exact;
        # only the collision distribution is estimated.
        sample_count = min(16, active.shape[0])
        sampled = torch.linspace(
            0,
            active.shape[0] - 1,
            sample_count,
        ).round().to(torch.int64)
        indptr = matrix.indptr.detach().cpu()
        posts = matrix.indices.detach().cpu()
        sampled_ratios = []
        for timestep in sampled.tolist():
            active_sources = torch.nonzero(
                active[timestep],
                as_tuple=False,
            ).flatten()
            if active_sources.numel() > 2048:
                offsets = torch.linspace(
                    0,
                    active_sources.numel() - 1,
                    2048,
                ).round().to(torch.int64)
                active_sources = active_sources[offsets]
            chunks = [
                posts[indptr[source] : indptr[source + 1]]
                for source in active_sources.tolist()
            ]
            timestep_posts = (
                torch.cat(chunks)
                if chunks
                else torch.empty(0, dtype=posts.dtype)
            )
            edge_count = timestep_posts.numel()
            ratio = (
                1.0 - torch.unique(timestep_posts).numel() / edge_count
                if edge_count
                else 0.0
            )
            sampled_ratios.append(float(ratio))
        collision_ratios.fill_(
            float(np.mean(sampled_ratios)) if sampled_ratios else 0.0
        )
        collision_method = (
            f"sampled_{sample_count}_timesteps_max_2048_sources"
        )

    counts_np = active_counts.numpy()
    edges_np = active_edges.numpy()
    collision_np = collision_ratios.numpy()
    mean_count = float(counts_np.mean()) if counts_np.size else 0.0
    count_std = float(counts_np.std()) if counts_np.size else 0.0
    measured = mean_count / max(matrix.shape[0], 1)
    burst_factor = (
        float(counts_np.max()) / mean_count
        if counts_np.size and mean_count
        else 0.0
    )
    return SpikeStats(
        requested_activity=float(requested_activity),
        measured_activity=measured,
        mean_active_neurons=mean_count,
        p95_active_neurons=(
            float(np.percentile(counts_np, 95)) if counts_np.size else 0.0
        ),
        max_active_neurons=int(counts_np.max()) if counts_np.size else 0,
        mean_active_edges=(
            float(edges_np.mean()) if edges_np.size else 0.0
        ),
        p95_active_edges=(
            float(np.percentile(edges_np, 95)) if edges_np.size else 0.0
        ),
        max_active_edges=int(edges_np.max()) if edges_np.size else 0,
        mean_collision_ratio=(
            float(collision_np.mean()) if collision_np.size else 0.0
        ),
        p95_collision_ratio=(
            float(np.percentile(collision_np, 95))
            if collision_np.size
            else 0.0
        ),
        temporal_cv=count_std / mean_count if mean_count else 0.0,
        burst_factor=burst_factor,
        collision_method=collision_method,
    )


def stats_are_finite(stats: SpikeStats) -> bool:
    """Return whether every floating workload statistic is finite."""

    return all(
        math.isfinite(value)
        for value in stats.to_dict().values()
        if isinstance(value, float)
    )
