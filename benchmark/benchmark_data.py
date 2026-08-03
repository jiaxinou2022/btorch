"""Generate, identify, save, and restore benchmark workloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from benchmark.benchmark_workload_stats import (
    GraphStats,
    SpikeStats,
    summarize_graph,
    summarize_spike_workload,
)
from btorch.sparse import CSR


GraphKind = Literal["uniform", "powerlaw", "hotspot", "clustered"]
SpikeKind = Literal[
    "iid",
    "fanout_correlated",
    "temporal",
    "cluster_burst",
]


@dataclass(frozen=True)
class GraphSpec:
    """Configure one reproducible synthetic source-oriented graph."""

    kind: GraphKind = "uniform"
    n_neuron: int = 4096
    fanout: int = 32
    seed: int = 20250308
    weight_scale: float = 0.15
    powerlaw_alpha: float = 2.0
    min_fanout: int = 1
    max_fanout: int = 1024
    hotspot_count: int = 32
    hotspot_probability: float = 0.8
    community_count: int = 16
    intra_cluster_ratio: float = 0.8


@dataclass(frozen=True)
class SpikeSpec:
    """Configure one reproducible batch-one spike trace."""

    kind: SpikeKind = "iid"
    t_steps: int = 128
    activity: float = 0.01
    seed: int = 20250309
    correlation_strength: float = 1.0
    persistence: float = 0.9
    community_count: int = 16
    cluster_high_rate: float = 0.2


@dataclass
class BenchmarkWorkload:
    """Bundle graph, trace, statistics, and a stable workload identifier."""

    matrix: CSR
    spike_trace: torch.Tensor | None
    input_trace: torch.Tensor | None
    graph_stats: GraphStats
    spike_stats: SpikeStats | None
    logical_n: int
    physical_n: int
    batch_size: int
    t_steps: int
    mode: Literal["operator", "closed_loop"]
    workload_id: str
    graph_spec: GraphSpec | None = None
    spike_spec: SpikeSpec | None = None

    def manifest(self) -> dict[str, object]:
        """Return a JSON-serializable workload manifest."""

        return {
            "workload_id": self.workload_id,
            "logical_n": self.logical_n,
            "physical_n": self.physical_n,
            "batch_size": self.batch_size,
            "t_steps": self.t_steps,
            "mode": self.mode,
            "graph_spec": (
                asdict(self.graph_spec) if self.graph_spec is not None else None
            ),
            "spike_spec": (
                asdict(self.spike_spec) if self.spike_spec is not None else None
            ),
            "graph_stats": self.graph_stats.to_dict(),
            "spike_stats": (
                self.spike_stats.to_dict()
                if self.spike_stats is not None
                else None
            ),
        }


def _validate_graph_spec(spec: GraphSpec) -> None:
    if spec.n_neuron <= 0 or spec.fanout < 0:
        raise ValueError("n_neuron must be positive and fanout non-negative")
    if spec.min_fanout < 0 or spec.max_fanout < spec.min_fanout:
        raise ValueError("invalid power-law fanout bounds")
    if not 0.0 <= spec.hotspot_probability <= 1.0:
        raise ValueError("hotspot_probability must be in [0, 1]")
    if not 0.0 <= spec.intra_cluster_ratio <= 1.0:
        raise ValueError("intra_cluster_ratio must be in [0, 1]")
    if spec.community_count <= 0 or spec.hotspot_count <= 0:
        raise ValueError("community_count and hotspot_count must be positive")


def _degrees(spec: GraphSpec, rng: np.random.Generator) -> np.ndarray:
    n = spec.n_neuron
    if spec.kind == "powerlaw":
        if spec.powerlaw_alpha <= 1.0:
            raise ValueError("powerlaw_alpha must be greater than 1")
        samples = rng.pareto(spec.powerlaw_alpha - 1.0, size=n) + 1.0
        # Preserve the requested mean fanout while changing only the degree
        # distribution. Clipping may lower the final mean slightly for very
        # heavy tails, which is recorded in GraphStats.
        scaled = samples * spec.fanout / max(float(samples.mean()), 1.0)
        degrees = np.floor(scaled).astype(np.int64)
        return np.clip(degrees, spec.min_fanout, min(spec.max_fanout, n))
    return np.full(n, min(spec.fanout, n), dtype=np.int64)


def _generate_posts(
    spec: GraphSpec,
    rows: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = spec.n_neuron
    edge_count = rows.size
    if spec.kind in {"uniform", "powerlaw"}:
        return rng.integers(0, n, size=edge_count, dtype=np.int64)
    if spec.kind == "hotspot":
        hotspot_count = min(spec.hotspot_count, n)
        hot_posts = rng.choice(n, size=hotspot_count, replace=False)
        use_hotspot = rng.random(edge_count) < spec.hotspot_probability
        posts = rng.integers(0, n, size=edge_count, dtype=np.int64)
        posts[use_hotspot] = hot_posts[
            rng.integers(0, hotspot_count, size=int(use_hotspot.sum()))
        ]
        return posts
    if spec.kind == "clustered":
        communities = min(spec.community_count, n)
        community_size = max((n + communities - 1) // communities, 1)
        source_community = np.minimum(rows // community_size, communities - 1)
        intra = rng.random(edge_count) < spec.intra_cluster_ratio
        posts = rng.integers(0, n, size=edge_count, dtype=np.int64)
        local_offset = rng.integers(
            0,
            community_size,
            size=edge_count,
            dtype=np.int64,
        )
        local_posts = source_community * community_size + local_offset
        posts[intra] = np.minimum(local_posts[intra], n - 1)
        return posts
    raise ValueError(f"unknown graph kind: {spec.kind}")


def generate_graph(
    spec: GraphSpec,
    device: torch.device,
) -> tuple[CSR, GraphStats]:
    """Generate a synthetic graph and its structural statistics."""

    _validate_graph_spec(spec)
    rng = np.random.default_rng(spec.seed)
    degrees = _degrees(spec, rng)
    rows = np.repeat(np.arange(spec.n_neuron, dtype=np.int64), degrees)
    posts = _generate_posts(spec, rows, rng)
    values = np.full(rows.size, spec.weight_scale, dtype=np.float32)
    matrix = CSR.from_edges(
        row=torch.from_numpy(rows),
        col=torch.from_numpy(posts),
        data=torch.from_numpy(values),
        shape=(spec.n_neuron, spec.n_neuron),
        device=device,
        dtype=torch.float32,
    )
    stats = summarize_graph(
        matrix,
        graph_type=spec.kind,
        seed=spec.seed,
    )
    return matrix, stats


def _scaled_probabilities(
    weights: torch.Tensor,
    target: float,
) -> torch.Tensor:
    if target <= 0.0:
        return torch.zeros_like(weights)
    if target >= 1.0:
        return torch.ones_like(weights)
    normalized = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
    low, high = 0.0, 1.0
    while float(torch.clamp(normalized * high, max=1.0).mean()) < target:
        high *= 2.0
    for _ in range(40):
        middle = (low + high) / 2.0
        measured = float(torch.clamp(normalized * middle, max=1.0).mean())
        if measured < target:
            low = middle
        else:
            high = middle
    return torch.clamp(normalized * ((low + high) / 2.0), max=1.0)


def generate_spike_trace(
    matrix: CSR,
    spec: SpikeSpec,
    device: torch.device,
) -> tuple[torch.Tensor, SpikeStats]:
    """Generate one fixed spike trace and measured sparse workload."""

    if spec.t_steps <= 0 or not 0.0 <= spec.activity <= 1.0:
        raise ValueError("t_steps must be positive and activity in [0, 1]")
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    shape = (spec.t_steps, matrix.shape[0])
    if spec.kind == "iid":
        probabilities = torch.full(
            (matrix.shape[0],),
            spec.activity,
            device=device,
        )
        trace = torch.rand(shape, device=device, generator=generator) < probabilities
    elif spec.kind == "fanout_correlated":
        fanout = (
            matrix.indptr[1:] - matrix.indptr[:-1]
        ).to(device=device, dtype=torch.float32)
        weights = (fanout + 1.0).pow(spec.correlation_strength)
        probabilities = _scaled_probabilities(weights, spec.activity)
        trace = torch.rand(shape, device=device, generator=generator) < probabilities
    elif spec.kind == "temporal":
        if not 0.0 <= spec.persistence < 1.0:
            raise ValueError("persistence must be in [0, 1)")
        trace = torch.empty(shape, device=device, dtype=torch.bool)
        trace[0] = (
            torch.rand(matrix.shape[0], device=device, generator=generator)
            < spec.activity
        )
        new_probability = (
            spec.activity * (1.0 - spec.persistence)
            / max(1.0 - spec.activity, torch.finfo(torch.float32).eps)
        )
        for timestep in range(1, spec.t_steps):
            keep = (
                torch.rand(matrix.shape[0], device=device, generator=generator)
                < spec.persistence
            )
            activate = (
                torch.rand(matrix.shape[0], device=device, generator=generator)
                < new_probability
            )
            trace[timestep] = (trace[timestep - 1] & keep) | (
                ~trace[timestep - 1] & activate
            )
    elif spec.kind == "cluster_burst":
        communities = min(spec.community_count, matrix.shape[0])
        neuron_community = (
            torch.arange(matrix.shape[0], device=device) * communities
            // matrix.shape[0]
        )
        high_rate = max(spec.cluster_high_rate, spec.activity)
        low_rate = max(
            (
                spec.activity * communities - high_rate
            )
            / max(communities - 1, 1),
            0.0,
        )
        trace = torch.empty(shape, device=device, dtype=torch.bool)
        for timestep in range(spec.t_steps):
            active_community = timestep % communities
            probability = torch.where(
                neuron_community == active_community,
                high_rate,
                low_rate,
            )
            trace[timestep] = (
                torch.rand(
                    matrix.shape[0],
                    device=device,
                    generator=generator,
                )
                < probability
            )
    else:
        raise ValueError(f"unknown spike kind: {spec.kind}")

    spikes = trace.to(torch.float32).unsqueeze(1)
    stats = summarize_spike_workload(
        matrix,
        spikes,
        requested_activity=spec.activity,
    )
    return spikes, stats


def _workload_id(
    graph_spec: GraphSpec,
    spike_spec: SpikeSpec | None,
    mode: str,
) -> str:
    payload = json.dumps(
        {
            "graph": asdict(graph_spec),
            "spike": asdict(spike_spec) if spike_spec is not None else None,
            "mode": mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def prepare_operator_workload(
    graph_spec: GraphSpec,
    spike_spec: SpikeSpec,
    device: torch.device,
) -> BenchmarkWorkload:
    """Prepare one reproducible operator-only workload."""

    matrix, graph_stats = generate_graph(graph_spec, device)
    spikes, spike_stats = generate_spike_trace(matrix, spike_spec, device)
    return BenchmarkWorkload(
        matrix=matrix,
        spike_trace=spikes,
        input_trace=None,
        graph_stats=graph_stats,
        spike_stats=spike_stats,
        logical_n=graph_spec.n_neuron,
        physical_n=graph_spec.n_neuron,
        batch_size=1,
        t_steps=spike_spec.t_steps,
        mode="operator",
        workload_id=_workload_id(graph_spec, spike_spec, "operator"),
        graph_spec=graph_spec,
        spike_spec=spike_spec,
    )


def build_closed_loop_workload(
    matrix: CSR,
    input_trace: torch.Tensor,
    spike_trace: torch.Tensor,
    *,
    dataset: str,
    seed: int,
    requested_activity: float | None,
) -> BenchmarkWorkload:
    """Build a unified closed-loop workload from reference RSNN output."""

    graph_stats = summarize_graph(
        matrix,
        graph_type=dataset,
        seed=seed,
    )
    spike_stats = summarize_spike_workload(
        matrix,
        spike_trace,
        requested_activity=(
            float("nan")
            if requested_activity is None
            else requested_activity
        ),
    )
    hasher = hashlib.sha256(
        json.dumps(
            {
                "dataset": dataset,
                "seed": seed,
                "shape": tuple(input_trace.shape),
                "requested_activity": requested_activity,
            },
            sort_keys=True,
        ).encode()
    )
    for tensor in (
        matrix.indptr,
        matrix.indices,
        matrix.effective_values(),
        input_trace,
    ):
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(memoryview(array))
    return BenchmarkWorkload(
        matrix=matrix,
        spike_trace=spike_trace,
        input_trace=input_trace,
        graph_stats=graph_stats,
        spike_stats=spike_stats,
        logical_n=matrix.shape[0],
        physical_n=matrix.shape[0],
        batch_size=input_trace.shape[1],
        t_steps=input_trace.shape[0],
        mode="closed_loop",
        workload_id=hasher.hexdigest()[:16],
    )


def save_workload(workload: BenchmarkWorkload, path: Path) -> None:
    """Save tensors plus a JSON manifest for exact workload reuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": workload.manifest(),
        "indptr": workload.matrix.indptr.detach().cpu(),
        "indices": workload.matrix.indices.detach().cpu(),
        "data": workload.matrix.data.detach().cpu(),
        "shape": workload.matrix.shape,
        "spike_trace": (
            workload.spike_trace.detach().cpu()
            if workload.spike_trace is not None
            else None
        ),
        "input_trace": (
            workload.input_trace.detach().cpu()
            if workload.input_trace is not None
            else None
        ),
    }
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(workload.manifest(), indent=2, sort_keys=True)
    )


def load_operator_workload(
    path: Path,
    device: torch.device,
) -> BenchmarkWorkload:
    """Restore a saved operator workload on a requested device."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    graph_spec = GraphSpec(**manifest["graph_spec"])
    spike_spec = SpikeSpec(**manifest["spike_spec"])
    matrix = CSR(
        payload["indptr"].to(device),
        payload["indices"].to(device),
        payload["data"].to(device),
        tuple(payload["shape"]),
    )
    spikes = payload["spike_trace"].to(device)
    graph_stats = GraphStats(**manifest["graph_stats"])
    spike_stats = SpikeStats(**manifest["spike_stats"])
    return BenchmarkWorkload(
        matrix=matrix,
        spike_trace=spikes,
        input_trace=None,
        graph_stats=graph_stats,
        spike_stats=spike_stats,
        logical_n=int(manifest["logical_n"]),
        physical_n=int(manifest["physical_n"]),
        batch_size=int(manifest["batch_size"]),
        t_steps=int(manifest["t_steps"]),
        mode="operator",
        workload_id=str(manifest["workload_id"]),
        graph_spec=graph_spec,
        spike_spec=spike_spec,
    )
