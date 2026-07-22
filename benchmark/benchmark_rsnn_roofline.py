"""Profile the persistent RSNN kernel with synthetic or connectome graphs.

The input spike trace, recurrent graph, and initial state are fixed for every
run.  One persistent CUDA launch advances exactly ``--t-steps`` timesteps, so
neither timing nor profiling depends on host polling.

Normal timing (CUDA Events, median of at least 20 runs)::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset uniform --csv roofline.csv

Collect opt-in BlockTask diagnostics (instrumented timing is not production
performance)::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset mice_column_v1 \
        --spike-block --block-stats --csv block_stats.csv

Enable the experimental selective shared-memory hash in a fresh process::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset mice_column_v1 \
        --spike-block --block-hash

Switch to the direct cuSPARSE CUDA Graph baseline::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --provider cusparse_cudagraph

Capture one representative persistent launch with Nsight Compute::

    ncu --set roofline --profile-from-start off \
        --kernel-name 'regex:persistent_snn(_binned)?_kernel' --launch-count 1 \
        micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset mice_column_v1 --mode ncu

For CUDA Graph analysis, use one whole-graph replay::

    ncu --set roofline --profile-from-start off --graph-profiling graph \
        --launch-count 1 micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py \
        --provider cusparse_cudagraph --mode ncu

The profiler interval contains one replay. ``--graph-profiling graph`` requires
a version of Nsight Compute that supports whole-graph profiling.

``active_neurons`` is the number of emitted neuron spike events over all
timesteps and batches. ``active_synapses`` is the sum of the corresponding
pre-synaptic CSR row degrees. Thus ``active_synapses_per_second`` measures
useful event-driven fanout work rather than the graph's total stored edges.
The selective hash path uses 128 slots, at most 8 linear probes, and only
64--96 medium-row edges. ``hash_tasks`` counts tasks entering that path;
``hash_overflow_tasks`` is an overlapping warning set.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTOME_REPO = REPO_ROOT / "connectome_dataset"
for path in (REPO_ROOT, CONNECTOME_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.benchmark_rsnn_cudagraph_compare import (  # noqa: E402
    DirectCuSparseProvider,
    make_torch_csr_weight,
)
from btorch.backend.persistent_snn import (  # noqa: E402
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNState,
    PersistentSNNWorkspace,
    WindowedSpikeEvents,
    make_empty_state,
    make_persistent_snn_workspace,
    persistent_snn_forward,
)
from btorch.sparse import CSR  # noqa: E402


Dataset = Literal["uniform", "mice_column_v1"]
Mode = Literal["benchmark", "ncu"]
Provider = Literal["persistent", "cusparse_cudagraph"]
FANOUT_BINNING_THRESHOLD = 256
LANE_ROW_THRESHOLD = 4
BLOCK_STATS_COLUMNS = 13
HASH_CAPACITY = 128
HASH_MAX_PROBES = 8
HASH_MIN_EDGES = 64
HASH_MAX_EDGES = 96


@dataclass(frozen=True)
class RooflineCase:
    """Describe one fixed-window persistent RSNN workload."""

    dataset: Dataset
    n_neuron: int
    batch_size: int
    t_steps: int
    fanout: int
    event_rate: float
    dt: float
    tau_mem: float
    tau_syn: float
    v_threshold: float
    v_reset: float
    c_m: float
    input_amplitude: float
    weight_scale: float


@dataclass(frozen=True)
class PreparedWorkload:
    """Hold all immutable tensors used by every timed launch."""

    case: RooflineCase
    matrix: CSR
    events: WindowedSpikeEvents
    graph: EventCSRGraph
    state: PersistentSNNState
    params: PersistentSNNParams
    empty_int: torch.Tensor
    x_seq: torch.Tensor
    high_fanout: torch.Tensor
    workspace: PersistentSNNWorkspace


def make_fixed_input_sequence(
    case: RooflineCase, device: torch.device
) -> torch.Tensor:
    """Create one deterministic input spike trace shared by all runs."""

    total = case.t_steps * case.batch_size * case.n_neuron
    n_active = max(0, min(total, int(round(total * case.event_rate))))
    flat = torch.zeros(total, device=device, dtype=torch.float32)
    if n_active:
        stride = max(1, total // n_active)
        active = (torch.arange(n_active, device=device) * stride) % total
        flat[active.to(torch.long)] = case.input_amplitude
    return flat.reshape(case.t_steps, case.batch_size, case.n_neuron)


def make_uniform_csr(case: RooflineCase, device: torch.device) -> CSR:
    """Create the deterministic uniform-fanout graph used by existing benches."""

    if case.fanout < 0 or case.fanout >= case.n_neuron:
        raise ValueError("--fanout must be in [0, n_neuron).")
    edge_count = case.n_neuron * case.fanout
    edge = torch.arange(edge_count, device=device, dtype=torch.long)
    row = torch.div(edge, case.fanout, rounding_mode="floor") if edge_count else edge
    slot = edge.remainder(case.fanout) if edge_count else edge
    col = (row + slot + 1).remainder(case.n_neuron)
    data = torch.full(
        (edge_count,),
        case.weight_scale / max(case.fanout, 1),
        device=device,
        dtype=torch.float32,
    )
    return CSR.from_edges(row, col, data, (case.n_neuron, case.n_neuron))


def load_connectome_csr(
    root: Path | None,
    *,
    weight_scale: float,
    device: torch.device,
) -> CSR:
    """Load mice_column_v1 through connectome_dataset and normalize weights."""

    try:
        from connectome_dataset.graph_loader import load_mice_column_v1
    except ImportError as exc:
        raise RuntimeError(
            "connectome_dataset is required for --dataset mice_column_v1"
        ) from exc

    try:
        scipy_matrix = load_mice_column_v1(root=root, use_weights=False).tocsr()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nFetch connectome_dataset/data/external/mice_column_v1 "
            "(for example with DVC), or pass --connectome-root."
        ) from exc

    if scipy_matrix.shape[0] != scipy_matrix.shape[1]:
        raise ValueError(f"mice_column_v1 must be square, got {scipy_matrix.shape}.")
    average_fanout = scipy_matrix.nnz / max(scipy_matrix.shape[0], 1)
    scipy_matrix.data.fill(weight_scale / max(average_fanout, 1.0))
    return CSR.from_scipy(scipy_matrix, device=device, dtype=torch.float32)


def dense_to_windowed_events(x_seq: torch.Tensor) -> WindowedSpikeEvents:
    """Convert a fixed dense trace to fixed time-batch event buckets."""

    t_steps, batch_size, n_neuron = x_seq.shape
    active = x_seq != 0
    counts = active.sum(dim=2, dtype=torch.int32).reshape(-1)
    offsets = torch.zeros(counts.numel() + 1, device=x_seq.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    indices = torch.nonzero(active.reshape(-1, n_neuron), as_tuple=False)[:, 1]
    return WindowedSpikeEvents(
        offsets=offsets.contiguous(),
        indices=indices.to(torch.int32).contiguous(),
        values=x_seq[active].to(torch.float32).contiguous(),
        shape=(t_steps, batch_size, n_neuron),
    )


def csr_to_event_graph(matrix: CSR) -> EventCSRGraph:
    """Convert btorch CSR storage to the persistent kernel graph contract."""

    return EventCSRGraph(
        indptr=matrix.indptr.to(torch.int32).contiguous(),
        indices=matrix.indices.to(torch.int32).contiguous(),
        weight=matrix.effective_values().to(torch.float32).contiguous(),
        delay=None,
        shape=matrix.shape,
    )


def prepare_workload(
    args: argparse.Namespace, device: torch.device
) -> PreparedWorkload:
    """Build all graph, input, state, and parameter tensors before timing."""

    if args.dataset == "uniform":
        n_neuron = args.n_neuron
        provisional = RooflineCase(
            dataset=args.dataset,
            n_neuron=n_neuron,
            batch_size=args.batch_size,
            t_steps=args.t_steps,
            fanout=args.fanout,
            event_rate=args.event_rate,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
            c_m=args.c_m,
            input_amplitude=args.input_amplitude,
            weight_scale=args.weight_scale,
        )
        matrix = make_uniform_csr(provisional, device)
    else:
        matrix = load_connectome_csr(
            args.connectome_root,
            weight_scale=args.weight_scale,
            device=device,
        )
        n_neuron = matrix.shape[0]
        provisional = RooflineCase(
            dataset=args.dataset,
            n_neuron=n_neuron,
            batch_size=args.batch_size,
            t_steps=args.t_steps,
            fanout=int(round(matrix.indices.numel() / max(n_neuron, 1))),
            event_rate=args.event_rate,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
            c_m=args.c_m,
            input_amplitude=args.input_amplitude,
            weight_scale=args.weight_scale,
        )

    x_seq = make_fixed_input_sequence(provisional, device)
    graph = csr_to_event_graph(matrix)
    return PreparedWorkload(
        case=provisional,
        matrix=matrix,
        events=dense_to_windowed_events(x_seq),
        graph=graph,
        state=make_empty_state(
            provisional.batch_size,
            provisional.n_neuron,
            device=device,
            refractory=False,
        ),
        params=PersistentSNNParams(
            dt=provisional.dt,
            tau_mem=provisional.tau_mem,
            tau_syn=provisional.tau_syn,
            v_threshold=provisional.v_threshold,
            v_reset=provisional.v_reset,
            c_m=provisional.c_m,
            hard_reset=False,
            window_size=provisional.t_steps,
        ),
        empty_int=torch.empty(0, device=device, dtype=torch.int32),
        x_seq=x_seq,
        high_fanout=(
            (graph.indptr[1:] - graph.indptr[:-1])
            >= FANOUT_BINNING_THRESHOLD
        )
        .to(torch.int32)
        .contiguous(),
        workspace=make_persistent_snn_workspace(
            graph,
            provisional.batch_size,
        ),
    )


def run_persistent(
    workload: PreparedWorkload, *, fanout_binning: bool, spike_block: bool
):
    """Execute one fixed-window persistent launch from the same zero state."""

    return persistent_snn_forward(
        workload.events,
        workload.graph,
        workload.state,
        workload.params,
        backend="cuda_persistent",
        return_mode="dense",
        fanout_binning=fanout_binning,
        spike_block=spike_block,
        workspace=workload.workspace,
    )


def run_prepared_operator(
    workload: PreparedWorkload, *, fanout_binning: bool, spike_block: bool
) -> tuple[torch.Tensor, ...]:
    """Call the loaded CUDA op without repeated Python-side validation syncs."""

    case = workload.case
    assert workload.events.values is not None
    op_args = (
        workload.events.offsets,
        workload.events.indices,
        workload.events.values,
        True,
        workload.graph.indptr,
        workload.graph.indices,
        workload.graph.weight,
        True,
        workload.workspace.input_current,
        workload.workspace.queue_batch,
        workload.workspace.queue_edge_start,
        workload.workspace.queue_edge_end,
        workload.workspace.spike_count,
        workload.workspace.work_counter,
    )
    op_tail = (
        workload.empty_int,
        False,
        True,
        workload.state.v,
        workload.state.psc,
        float(case.dt),
        float(case.tau_mem),
        float(case.tau_syn),
        float(case.v_threshold),
        float(case.v_reset),
        float(case.c_m),
        False,
        True,
        False,
    )
    if fanout_binning:
        return torch.ops.btorch_cuda.persistent_snn_forward_binned(
            *op_args,
            workload.high_fanout,
            *op_tail,
        )
    if spike_block:
        return torch.ops.btorch_cuda.persistent_snn_forward_spike_block(
            *op_args,
            *op_tail,
        )
    return torch.ops.btorch_cuda.persistent_snn_forward(*op_args, *op_tail)


def torch_reference(workload: PreparedWorkload) -> tuple[torch.Tensor, ...]:
    """Evaluate the same RSNN equations using independent PyTorch operations."""

    case = workload.case
    x_seq = torch.zeros(
        case.t_steps,
        case.batch_size,
        case.n_neuron,
        device=workload.state.v.device,
        dtype=torch.float32,
    )
    offsets = workload.events.offsets.cpu().tolist()
    for bucket in range(case.t_steps * case.batch_size):
        start, end = offsets[bucket : bucket + 2]
        if end > start:
            t, batch = divmod(bucket, case.batch_size)
            index = workload.events.indices[start:end].to(torch.long)
            x_seq[t, batch].scatter_add_(0, index, workload.events.values[start:end])

    row = workload.matrix._row
    col = workload.matrix.indices
    weight = workload.matrix.effective_values()
    decay = math.exp(-case.dt / case.tau_syn)
    reset_delta = case.v_threshold - case.v_reset
    v = torch.zeros_like(workload.state.v)
    psc = torch.zeros_like(v)
    spikes = []
    for t in range(case.t_steps):
        current = psc + x_seq[t]
        v_pre = v + case.dt * (
            -(v - case.v_reset) / case.tau_mem + current / case.c_m
        )
        z = (v_pre >= case.v_threshold).to(v.dtype)
        v = v_pre - reset_delta * z
        recurrent = torch.zeros_like(psc)
        if col.numel():
            contributions = z[:, row] * weight
            recurrent.scatter_add_(1, col.expand(case.batch_size, -1), contributions)
        psc = psc * decay + recurrent
        spikes.append(z)
    return torch.stack(spikes), v, psc


ProviderOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def make_provider_runner(
    workload: PreparedWorkload,
    *,
    provider: Provider,
    fanout_binning: bool,
    spike_block: bool,
) -> Callable[[], ProviderOutput]:
    """Prepare a provider and return its validation-sync-free repeated call."""

    if provider == "cusparse_cudagraph":
        # Use exactly the same direct cuSPARSE preparation and capture path as
        # benchmark_rsnn_cudagraph_compare.py: post-by-pre PyTorch CSR,
        # int32 direct descriptors, SpMV for B=1 / SpMM otherwise, fixed
        # state/output buffers, three side-stream warmups, then full replay.
        weight = make_torch_csr_weight(workload.matrix)
        result_runner = DirectCuSparseProvider().fixed_runner(
            workload.x_seq,
            weight,
            workload.case,
            use_cudagraph=True,
        )

        def run_cusparse_graph() -> ProviderOutput:
            output = result_runner()
            return output.spikes, output.v, output.psc

        return run_cusparse_graph

    # Load/JIT the selected extension and populate launch caches before timing.
    run_persistent(
        workload,
        fanout_binning=fanout_binning,
        spike_block=spike_block,
    )
    torch.cuda.synchronize()

    def run() -> ProviderOutput:
        output = run_prepared_operator(
            workload,
            fanout_binning=fanout_binning,
            spike_block=spike_block,
        )
        return output[0], output[3], output[4]

    return run


def validate_with_torch(
    workload: PreparedWorkload,
    run: Callable[[], ProviderOutput],
) -> dict[str, float | int]:
    """Validate against PyTorch with tolerances for atomic reduction order."""

    expected_spikes, expected_v, expected_psc = torch_reference(workload)
    actual_spikes, actual_v, actual_psc = run()
    spike_mismatches = int((actual_spikes != expected_spikes).sum().item())
    spike_mismatch_rate = spike_mismatches / expected_spikes.numel()
    v_max_abs_diff = float((actual_v - expected_v).abs().max().item())
    psc_max_abs_diff = float((actual_psc - expected_psc).abs().max().item())

    # Recurrent thresholding can amplify tiny differences between the CUDA
    # kernel's atomicAdd order and PyTorch scatter_add_'s reduction order.
    # Keep strict bounds on both the discrete mismatch rate and final state.
    if spike_mismatch_rate > 1e-3:
        raise AssertionError(
            f"spike mismatch rate {spike_mismatch_rate:.6g} exceeds 1e-3"
        )
    torch.testing.assert_close(actual_v, expected_v, atol=2e-1, rtol=2e-3)
    torch.testing.assert_close(actual_psc, expected_psc, atol=5e-4, rtol=2e-3)
    return {
        "spike_mismatches": spike_mismatches,
        "spike_mismatch_rate": spike_mismatch_rate,
        "v_max_abs_diff": v_max_abs_diff,
        "psc_max_abs_diff": psc_max_abs_diff,
    }


def cuda_event_median_ms(
    run: Callable[[], ProviderOutput], warmup: int, repeat: int
) -> float:
    """Measure full fixed-window forward latency with CUDA Events."""

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
    )
    return float(samples.median().item())


def activity_counts(
    workload: PreparedWorkload,
    run: Callable[[], ProviderOutput],
) -> tuple[int, int, int]:
    """Count spike events, unique active neurons, and traversed synapses."""

    output = run()
    active = output[0] != 0
    active_neurons = int(active.sum().item())
    unique_active_neurons = int(active.any(dim=(0, 1)).sum().item())
    degrees = workload.graph.indptr[1:] - workload.graph.indptr[:-1]
    active_synapses = int((active.to(torch.int64) * degrees).sum().item())
    return active_neurons, unique_active_neurons, active_synapses


def _simulate_task_hash(posts: list[int]) -> tuple[int, int, int, int]:
    """Simulate the proposed bounded shared-memory hash for one BlockTask."""

    keys = [-1] * HASH_CAPACITY
    insert_count = 0
    merge_count = 0
    collision_count = 0
    overflow_count = 0
    for post in posts:
        slot = (post * 2654435761) & (HASH_CAPACITY - 1)
        for probe in range(HASH_MAX_PROBES):
            index = (slot + probe) & (HASH_CAPACITY - 1)
            key = keys[index]
            if key == post:
                merge_count += 1
                break
            if key < 0:
                keys[index] = post
                insert_count += 1
                break
            collision_count += 1
        else:
            overflow_count += 1
    return insert_count, merge_count, collision_count, overflow_count


def summarize_block_stats(
    raw_stats: torch.Tensor,
    graph_indptr: torch.Tensor,
    graph_indices: torch.Tensor,
    *,
    block_hash_enabled: bool = True,
) -> dict[str, float | int]:
    """Augment macro-collected task records with exact offline post statistics."""

    if raw_stats.ndim != 2 or raw_stats.shape[1] != BLOCK_STATS_COLUMNS:
        raise ValueError(
            f"block stats must have shape (records, {BLOCK_STATS_COLUMNS})"
        )
    stats = raw_stats.to(device="cpu", dtype=torch.int64)
    indptr = graph_indptr.to(device="cpu", dtype=torch.int64).tolist()
    indices = graph_indices.to(device="cpu", dtype=torch.int64).tolist()

    totals: dict[str, int] = {
        "block_task_count": 0,
        "active_rows": 0,
        "active_edges": 0,
        "active_run_count": 0,
        "span_edges": 0,
        "max_row_degree": 0,
        "unique_posts": 0,
        "hash_insert_count": 0,
        "hash_merge_count": 0,
        "hash_collision_count": 0,
        "hash_overflow_count": 0,
        "global_atomic_count": 0,
        "flushed_entries": 0,
        "direct_lane_tasks": 0,
        "active_run_tasks": 0,
        "hash_tasks": 0,
        "hash_beneficial_tasks": 0,
        "hash_overflow_tasks": 0,
        "packed_tasks": 0,
        "long_segment_tasks": int(stats[:, 8].sum().item()),
    }
    for record in stats:
        active_rows = int(record[0].item())
        if active_rows == 0:
            continue
        totals["block_task_count"] += 1
        totals["active_rows"] += active_rows
        totals["active_edges"] += int(record[1].item())
        totals["active_run_count"] += int(record[2].item())
        totals["span_edges"] += int(record[3].item())
        totals["max_row_degree"] = max(
            totals["max_row_degree"], int(record[4].item())
        )
        block_start = int(record[11].item())
        active_mask = int(record[12].item()) & 0xFFFFFFFF
        posts: list[int] = []
        medium_posts: list[int] = []
        medium_mask = 0
        tiny_edges = 0
        for lane in range(32):
            if active_mask & (1 << lane):
                neuron = block_start + lane
                row_posts = indices[indptr[neuron] : indptr[neuron + 1]]
                posts.extend(row_posts)
                if len(row_posts) <= LANE_ROW_THRESHOLD:
                    tiny_edges += len(row_posts)
                else:
                    medium_mask |= 1 << lane
                    medium_posts.extend(row_posts)
        unique_posts = len(set(posts))
        medium_rows = medium_mask.bit_count()
        medium_runs = (medium_mask & ~(medium_mask << 1)).bit_count()
        if medium_rows == 0:
            totals["direct_lane_tasks"] += 1
        elif medium_runs * 2 <= medium_rows:
            totals["active_run_tasks"] += 1
        else:
            totals["packed_tasks"] += 1

        use_hash = (
            block_hash_enabled
            and medium_rows >= 2
            and HASH_MIN_EDGES <= len(medium_posts) <= HASH_MAX_EDGES
        )
        if use_hash:
            insert, merge, collisions, overflow = _simulate_task_hash(
                medium_posts
            )
            flushed_entries = tiny_edges + insert + overflow
            totals["hash_tasks"] += 1
            if flushed_entries < len(posts):
                totals["hash_beneficial_tasks"] += 1
            if overflow:
                totals["hash_overflow_tasks"] += 1
        else:
            insert = merge = collisions = overflow = 0
            flushed_entries = len(posts)
        totals["unique_posts"] += unique_posts
        totals["hash_insert_count"] += insert
        totals["hash_merge_count"] += merge
        totals["hash_collision_count"] += collisions
        totals["hash_overflow_count"] += overflow
        totals["flushed_entries"] += flushed_entries
        totals["global_atomic_count"] += flushed_entries

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    task_count = totals["block_task_count"]
    dispatch_count = task_count + totals["long_segment_tasks"]
    derived: dict[str, float] = {
        "average_spikes_per_task": ratio(totals["active_rows"], task_count),
        "average_runs_per_task": ratio(totals["active_run_count"], task_count),
        "adjacent_active_row_ratio": ratio(
            totals["active_rows"] - totals["active_run_count"],
            totals["active_rows"],
        ),
        "run_merge_ratio": ratio(
            totals["active_rows"], totals["active_run_count"]
        ),
        "span_utilization": ratio(totals["active_edges"], totals["span_edges"]),
        "post_duplicate_ratio": (
            1.0 - ratio(totals["unique_posts"], totals["active_edges"])
            if totals["active_edges"]
            else 0.0
        ),
        "atomic_reduction_ratio": (
            1.0 - ratio(totals["flushed_entries"], totals["active_edges"])
            if totals["active_edges"]
            else 0.0
        ),
        "direct_lane_task_ratio": ratio(totals["direct_lane_tasks"], task_count),
        "active_run_task_ratio": ratio(totals["active_run_tasks"], task_count),
        "hash_task_ratio": ratio(totals["hash_tasks"], task_count),
        "hash_beneficial_task_ratio": ratio(
            totals["hash_beneficial_tasks"], task_count
        ),
        "hash_overflow_task_ratio": ratio(
            totals["hash_overflow_tasks"], task_count
        ),
        "long_segment_task_ratio": ratio(
            totals["long_segment_tasks"], dispatch_count
        ),
    }
    return {**totals, **derived}


def benchmark_row(
    workload: PreparedWorkload,
    run: Callable[[], ProviderOutput],
    warmup: int,
    repeat: int,
    correctness: dict[str, float | int] | None,
    provider: Provider,
    fanout_binning: bool,
    spike_block: bool,
    block_hash: bool,
    block_stats: dict[str, float | int] | None = None,
) -> dict:
    """Run the benchmark and return one flat, CSV-friendly result record."""

    latency_ms = cuda_event_median_ms(run, warmup, repeat)
    active_neurons, unique_active_neurons, active_synapses = activity_counts(
        workload, run
    )
    case = workload.case
    row = {
        "dataset": case.dataset,
        "provider": provider,
        "direct_cusparse_primitive": (
            "SpMV" if provider == "cusparse_cudagraph" and case.batch_size == 1
            else "SpMM" if provider == "cusparse_cudagraph" else ""
        ),
        "direct_cusparse_algorithm": (
            "CUSPARSE_SPMV_ALG_DEFAULT"
            if provider == "cusparse_cudagraph" and case.batch_size == 1
            else "CUSPARSE_SPMM_CSR_ALG1"
            if provider == "cusparse_cudagraph"
            else ""
        ),
        "fanout_binning": fanout_binning,
        "spike_block": spike_block,
        "block_hash": block_hash,
        "block_stats": block_stats is not None,
        "grid_blocks": os.environ.get("BTORCH_PERSISTENT_GRID_BLOCKS", "auto"),
        "timing_mode": "instrumented_debug" if block_stats is not None else "normal",
        "total_time_ms": latency_ms,
        "timestep_count": case.t_steps,
        "time_per_timestep_us": latency_ms * 1000.0 / case.t_steps,
        "n_neuron": case.n_neuron,
        "batch_size": case.batch_size,
        "graph_synapses": int(workload.graph.indices.numel()),
        "average_fanout": workload.graph.indices.numel() / case.n_neuron,
        "input_event_rate": case.event_rate,
        "lane_row_threshold": LANE_ROW_THRESHOLD if spike_block else "",
        "hash_capacity": HASH_CAPACITY if spike_block else "",
        "hash_max_probes": HASH_MAX_PROBES if spike_block else "",
        "hash_min_edges": HASH_MIN_EDGES if spike_block else "",
        "hash_max_edges": HASH_MAX_EDGES if spike_block else "",
        "active_neurons": active_neurons,
        "unique_active_neurons": unique_active_neurons,
        "active_synapses": active_synapses,
        "active_synapses_per_second": active_synapses / (latency_ms * 1e-3),
        "warmup": warmup,
        "repeat": repeat,
        "correctness": "passed" if correctness is not None else "skipped",
    }
    if correctness is not None:
        row.update(correctness)
    if block_stats is not None:
        row.update(block_stats)
    return row


def profile_one_launch(
    run: Callable[[], ProviderOutput], warmup: int
) -> None:
    """Warm up, then expose one persistent launch or graph replay."""

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("uniform", "mice_column_v1"),
        default="uniform",
    )
    parser.add_argument(
        "--provider",
        choices=("persistent", "cusparse_cudagraph"),
        default="persistent",
    )
    parser.add_argument(
        "--fanout-binning",
        action="store_true",
        help="Use the opt-in binned persistent kernel (default: disabled).",
    )
    parser.add_argument(
        "--spike-block",
        action="store_true",
        help="Use 32-neuron BlockTasks and long-row SegmentTasks.",
    )
    parser.add_argument(
        "--block-stats",
        action="store_true",
        help="Compile ENABLE_BLOCK_STATS and emit per-BlockTask debug statistics.",
    )
    parser.add_argument(
        "--block-hash",
        action="store_true",
        help="Compile the experimental selective shared-memory BlockTask hash.",
    )
    parser.add_argument("--mode", choices=("benchmark", "ncu"), default="benchmark")
    parser.add_argument(
        "--grid-blocks",
        type=int,
        default=None,
        help=(
            "Override the cooperative persistent grid size. The requested value "
            "must not exceed this kernel's occupancy limit."
        ),
    )
    parser.add_argument("--n-neuron", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--event-rate", type=float, default=0.01)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--v-reset", type=float, default=0.0)
    parser.add_argument("--c-m", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    if not 5 <= args.warmup <= 20:
        parser.error("--warmup must be between 5 and 20.")
    if args.repeat < 20:
        parser.error("--repeat must be at least 20.")
    if args.t_steps <= 0 or args.batch_size <= 0 or args.n_neuron <= 0:
        parser.error("--t-steps, --batch-size, and --n-neuron must be positive.")
    if args.grid_blocks is not None and args.grid_blocks <= 0:
        parser.error("--grid-blocks must be positive.")
    if not 0.0 <= args.event_rate <= 1.0:
        parser.error("--event-rate must be in [0, 1].")
    if args.fanout_binning and args.provider != "persistent":
        parser.error("--fanout-binning is only valid with --provider persistent.")
    if args.spike_block and args.provider != "persistent":
        parser.error("--spike-block is only valid with --provider persistent.")
    if args.fanout_binning and args.spike_block:
        parser.error("--fanout-binning and --spike-block are mutually exclusive.")
    if args.block_stats and not args.spike_block:
        parser.error("--block-stats requires --spike-block.")
    if args.block_hash and not args.spike_block:
        parser.error("--block-hash requires --spike-block.")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the RSNN roofline benchmark.")

    if args.grid_blocks is not None:
        os.environ["BTORCH_PERSISTENT_GRID_BLOCKS"] = str(args.grid_blocks)

    workload = prepare_workload(args, torch.device("cuda"))
    if args.block_stats or args.block_hash:
        from btorch.backend.persistent import plain_version

        plain_version.load(
            enable_block_stats=args.block_stats,
            enable_block_hash=args.block_hash,
        )
    run = make_provider_runner(
        workload,
        provider=args.provider,
        fanout_binning=args.fanout_binning,
        spike_block=args.spike_block,
    )
    torch.cuda.synchronize()
    correctness = None
    if not args.skip_correctness:
        correctness = validate_with_torch(workload, run)
        print(f"PyTorch correctness: passed ({correctness})")

    if args.mode == "ncu":
        profile_one_launch(run, args.warmup)
        print(f"Nsight Compute representative {args.provider} launch completed")
        return

    block_stats = None
    if args.block_stats:
        raw_output = run_prepared_operator(
            workload,
            fanout_binning=False,
            spike_block=True,
        )
        raw_stats = raw_output[5]
        if raw_stats.numel() == 0:
            raise RuntimeError(
                "ENABLE_BLOCK_STATS build returned no statistics; start a fresh "
                "process so the stats extension is loaded first."
            )
        block_stats = summarize_block_stats(
            raw_stats,
            workload.graph.indptr,
            workload.graph.indices,
            block_hash_enabled=args.block_hash,
        )

    row = benchmark_row(
        workload,
        run,
        args.warmup,
        args.repeat,
        correctness,
        args.provider,
        args.fanout_binning,
        args.spike_block,
        args.block_hash,
        block_stats,
    )
    for key, value in row.items():
        print(f"{key}: {value}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
