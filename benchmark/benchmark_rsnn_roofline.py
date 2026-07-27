"""Profile the persistent RSNN kernel with synthetic or connectome graphs.

The input spike trace, recurrent graph, and initial state are fixed for every
run. One persistent CUDA launch advances exactly ``--t-steps`` timesteps, so
neither timing nor profiling depends on host polling. The ``flybrain`` dataset
selects the signed FlyWire graph for this common RSNN workload, not the full
Shiu et al. refractory, delay, and hard-reset dynamics.

Normal timing (CUDA Events, median of at least 20 runs)::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset flybrain --csv roofline.csv

Collect opt-in BlockTask diagnostics (instrumented timing is not production
performance)::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset mice_column_v1 \
        --spike-block --block-stats --csv block_stats.csv

Enable the experimental warp-private hash in a fresh process::

    micromamba run -n ml-py312 python \
        benchmark/benchmark_rsnn_roofline.py --dataset flybrain \
        --spike-block --block-edge-budget 256 --long-segment-size 512 \
        --block-hash

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
The V4 hash path scans aggregation scales, capacities, probe limits, and a
minimum-window threshold through explicit CLI options. Instrumented builds
report flush and fallback atomics separately from the production timing.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTOME_REPO = REPO_ROOT / "connectome_dataset"
for path in (REPO_ROOT, CONNECTOME_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.benchmark_rsnn_cudagraph_compare import (  # noqa: E402
    PSC_ATOL,
    SPIKE_MISMATCH_RATE_TOL,
    STATE_RTOL,
    V_ATOL,
    DirectCuSparseProvider,
    load_flybrain_csr,
    make_torch_csr_weight,
    max_normalized_error,
    resolve_dataset_defaults,
)
from btorch.backend.persistent.reorder import (  # noqa: E402
    NeuronPermutation,
    ReorderConfig,
    build_neuron_permutation,
    reorder_events,
    reorder_graph,
    reorder_state,
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


Dataset = Literal["flybrain", "uniform", "mice_column_v1"]
Mode = Literal["benchmark", "ncu"]
Provider = Literal["persistent", "cusparse_cudagraph"]
FANOUT_BINNING_THRESHOLD = 256
LANE_ROW_THRESHOLD = 4
BLOCK_STATS_COLUMNS = 44
HASH_AGGREGATION_SCALES = (32, 64, 128, 256, 512)
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
    permutation: NeuronPermutation
    reorder_preprocess_ms: float
    reorder_stats: dict[str, float | int]


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
    dataset: Dataset,
    root: Path | None,
    *,
    weight_scale: float,
    device: torch.device,
) -> CSR:
    """Load a connectome graph with its dataset-specific weight conversion."""

    if dataset == "flybrain":
        return load_flybrain_csr(root, weight_scale=weight_scale, device=device)
    if dataset != "mice_column_v1":
        raise ValueError(f"Unsupported connectome dataset: {dataset}.")

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


def summarize_reordered_graph(
    original: EventCSRGraph,
    reordered: EventCSRGraph,
    *,
    block_size: int = 32,
) -> dict[str, float | int]:
    """Summarize static BlockTask cost balance and structural post reuse."""

    def metrics(graph: EventCSRGraph) -> tuple[float, int, float]:
        indptr = graph.indptr.cpu().to(torch.int64)
        indices = graph.indices.cpu().to(torch.int64)
        block_sums = []
        block_maxima = []
        edge_total = 0
        unique_total = 0
        for start in range(0, graph.shape[0], block_size):
            end = min(start + block_size, graph.shape[0])
            edge_start = int(indptr[start])
            edge_end = int(indptr[end])
            degrees = indptr[start + 1 : end + 1] - indptr[start:end]
            block_sums.append(edge_end - edge_start)
            block_maxima.append(int(degrees.max()) if degrees.numel() else 0)
            edge_total += edge_end - edge_start
            unique_total += int(torch.unique(indices[edge_start:edge_end]).numel())
        sums = torch.tensor(block_sums, dtype=torch.float64)
        p99 = (
            float(torch.quantile(sums, 0.99).item())
            if sums.numel()
            else 0.0
        )
        maximum = max(block_maxima, default=0)
        reuse = edge_total / unique_total if unique_total else 0.0
        return p99, maximum, reuse

    old_p99, old_max, old_reuse = metrics(original)
    new_p99, new_max, new_reuse = metrics(reordered)
    return {
        "static_block_edges_p99_before": old_p99,
        "static_block_edges_p99_after": new_p99,
        "static_block_max_fanout_before": old_max,
        "static_block_max_fanout_after": new_max,
        "static_edges_per_unique_post_before": old_reuse,
        "static_edges_per_unique_post_after": new_reuse,
    }


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
            args.dataset,
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
    original_graph = csr_to_event_graph(matrix)
    original_events = dense_to_windowed_events(x_seq)
    original_state = make_empty_state(
        provisional.batch_size,
        provisional.n_neuron,
        device=device,
        refractory=False,
    )
    reorder_config = ReorderConfig(
        mode=args.reorder,
        local_window_size=args.reorder_window,
        extreme_fanout_threshold=args.reorder_extreme_threshold,
    )
    torch.cuda.synchronize()
    preprocess_start = time.perf_counter()
    permutation = build_neuron_permutation(original_graph, reorder_config)
    if args.reorder == "identity":
        graph = original_graph
        events = original_events
        state = original_state
    else:
        graph = reorder_graph(original_graph, permutation)
        events = reorder_events(original_events, permutation)
        state = reorder_state(original_state, permutation)
    torch.cuda.synchronize()
    reorder_preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    reorder_stats = summarize_reordered_graph(original_graph, graph)
    if args.reorder != "identity":
        matrix = CSR(
            graph.indptr.to(torch.long),
            graph.indices.to(torch.long),
            graph.weight,
            graph.shape,
        )
        x_seq = x_seq.index_select(-1, permutation.new_to_old)
    return PreparedWorkload(
        case=provisional,
        matrix=matrix,
        events=events,
        graph=graph,
        state=state,
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
        permutation=permutation,
        reorder_preprocess_ms=reorder_preprocess_ms,
        reorder_stats=reorder_stats,
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
    # Relative tolerance is essential for signed FlyWire counts, whose state
    # magnitudes can be orders of magnitude larger than normalized graphs.
    if spike_mismatch_rate > SPIKE_MISMATCH_RATE_TOL:
        raise AssertionError(
            f"spike mismatch rate {spike_mismatch_rate:.6g} exceeds "
            f"{SPIKE_MISMATCH_RATE_TOL}"
        )
    torch.testing.assert_close(
        actual_v,
        expected_v,
        atol=V_ATOL,
        rtol=STATE_RTOL,
    )
    torch.testing.assert_close(
        actual_psc,
        expected_psc,
        atol=PSC_ATOL,
        rtol=STATE_RTOL,
    )
    return {
        "spike_mismatches": spike_mismatches,
        "spike_mismatch_rate": spike_mismatch_rate,
        "v_max_abs_diff": v_max_abs_diff,
        "psc_max_abs_diff": psc_max_abs_diff,
        "v_max_normalized_error": max_normalized_error(
            actual_v,
            expected_v,
            atol=V_ATOL,
            rtol=STATE_RTOL,
        ),
        "psc_max_normalized_error": max_normalized_error(
            actual_psc,
            expected_psc,
            atol=PSC_ATOL,
            rtol=STATE_RTOL,
        ),
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


def cuda_reorder_io_ms(
    workload: PreparedWorkload,
    output: ProviderOutput,
    *,
    repeat: int = 30,
) -> tuple[float, float]:
    """Measure per-window dynamic input permutation and output restoration."""

    if workload.permutation.config.mode == "identity":
        return 0.0, 0.0

    def measure(operation: Callable[[], None]) -> float:
        for _ in range(5):
            operation()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize()
        samples = torch.tensor(
            [
                start.elapsed_time(end)
                for start, end in zip(starts, ends, strict=True)
            ],
            dtype=torch.float64,
        )
        return float(samples.median().item())

    def permute_input() -> None:
        workload.permutation.old_to_new[
            workload.events.indices.to(torch.int64)
        ]
        workload.state.v.index_select(-1, workload.permutation.new_to_old)
        workload.state.psc.index_select(-1, workload.permutation.new_to_old)

    def restore_output_tensors() -> None:
        output[0].index_select(-1, workload.permutation.old_to_new)
        output[1].index_select(-1, workload.permutation.old_to_new)
        output[2].index_select(-1, workload.permutation.old_to_new)

    return measure(permute_input), measure(restore_output_tensors)


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


@dataclass
class _AggregationScaleAccumulator:
    """Accumulate aggregation bounds without retaining task edge streams."""

    scale: int | None
    covered_edges: int = 0
    ideal_atomics: int = 0
    window_edges: list[int] = field(default_factory=list)
    ratios: list[float] = field(default_factory=list)

    def update(self, posts: list[int]) -> None:
        """Add one original spike-block logical edge stream."""

        window_size = len(posts) if self.scale is None else self.scale
        if window_size == 0:
            return
        for begin in range(0, len(posts), window_size):
            window = posts[begin : begin + window_size]
            edges = len(window)
            unique = len(set(window))
            self.covered_edges += edges
            self.ideal_atomics += unique
            self.window_edges.append(edges)
            self.ratios.append(edges / unique)

    def summarize(self) -> dict[str, float | int]:
        """Return ideal atomics, quantiles, and edge-weighted coverage."""

        def quantile(q: float) -> float:
            if not self.ratios:
                return 0.0
            return float(
                torch.quantile(torch.tensor(self.ratios), q).item()
            )

        result: dict[str, float | int] = {
            "covered_edges": self.covered_edges,
            "unique_posts": self.ideal_atomics,
            "ideal_global_atomics": self.ideal_atomics,
            "ideal_atomic_reduction": (
                1.0 - self.ideal_atomics / self.covered_edges
                if self.covered_edges
                else 0.0
            ),
            "edges_per_unique_p50": quantile(0.50),
            "edges_per_unique_p90": quantile(0.90),
            "edges_per_unique_p99": quantile(0.99),
        }
        for threshold in (1.1, 1.25, 1.5, 2.0):
            suffix = str(threshold).replace(".", "_")
            duplicate_edges = sum(
                edges
                for edges, ratio_value in zip(
                    self.window_edges, self.ratios, strict=True
                )
                if ratio_value >= threshold
            )
            result[f"edge_coverage_ratio_ge_{suffix}"] = (
                duplicate_edges / self.covered_edges
                if self.covered_edges
                else 0.0
            )
        return result


def _aggregation_scale_stats(
    task_posts: list[list[int]],
    scale: int | None,
) -> dict[str, float | int]:
    """Measure the ideal atomic reduction for one aggregation scale."""

    accumulator = _AggregationScaleAccumulator(scale)
    for posts in task_posts:
        accumulator.update(posts)
    return accumulator.summarize()


def summarize_block_stats(
    raw_stats: torch.Tensor,
    graph_indptr: torch.Tensor,
    graph_indices: torch.Tensor,
    *,
    block_hash_enabled: bool = True,
    block_edge_budget: int = 0,
    hash_aggregation: int = 256,
) -> dict[str, float | int]:
    """Augment macro-collected task records with exact offline post statistics."""

    supported_columns = (23, BLOCK_STATS_COLUMNS)
    if raw_stats.ndim != 2 or raw_stats.shape[1] not in supported_columns:
        raise ValueError(
            "block stats must have shape "
            f"(records, one of {supported_columns})"
        )
    stats = raw_stats.to(device="cpu", dtype=torch.int64)
    active_task_edges = stats[:, 1]
    active_task_edges = active_task_edges[active_task_edges > 0]
    logical_edge_budget = (
        max(block_edge_budget, hash_aggregation)
        if block_hash_enabled and block_edge_budget > 0
        else block_edge_budget
    )
    if logical_edge_budget > 0 and active_task_edges.numel():
        full_task_counts = active_task_edges // logical_edge_budget
        full_edges = torch.full(
            (int(full_task_counts.sum().item()),),
            logical_edge_budget,
            dtype=torch.int64,
        )
        remainders = active_task_edges % logical_edge_budget
        logical_task_edges = torch.cat(
            [full_edges, remainders[remainders > 0]]
        )
    else:
        logical_task_edges = active_task_edges

    def task_quantile(q: float) -> float:
        if not logical_task_edges.numel():
            return 0.0
        return float(
            torch.quantile(logical_task_edges.to(torch.float64), q).item()
        )

    def original_task_quantile(q: float) -> float:
        if not active_task_edges.numel():
            return 0.0
        return float(
            torch.quantile(active_task_edges.to(torch.float64), q).item()
        )

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
        "v4_input_edges": int(stats[0, 13].item()) if stats.numel() else 0,
        "v4_global_atomics": int(stats[0, 14].item()) if stats.numel() else 0,
        "v4_reduce_tasks": int(stats[0, 15].item()) if stats.numel() else 0,
        "v4_reduce_edges": int(stats[0, 16].item()) if stats.numel() else 0,
        "kernel_hash_tasks": int(stats[0, 17].item()) if stats.numel() else 0,
        "kernel_hash_windows": int(stats[0, 18].item()) if stats.numel() else 0,
        "kernel_hash_input_edges": (
            int(stats[0, 19].item()) if stats.numel() else 0
        ),
        "kernel_hash_flush_atomics": (
            int(stats[0, 20].item()) if stats.numel() else 0
        ),
        "kernel_hash_fallback_atomics": (
            int(stats[0, 21].item()) if stats.numel() else 0
        ),
        "kernel_hash_probe_attempts": (
            int(stats[0, 22].item()) if stats.numel() else 0
        ),
        "b3_ordinary_tasks": (
            int(stats[0, 23].item())
            if stats.numel() and stats.size(1) > 23
            else 0
        ),
        "b3_ordinary_edges": (
            int(stats[0, 24].item())
            if stats.numel() and stats.size(1) > 24
            else 0
        ),
        "b3_eligible_tasks": (
            int(stats[0, 25].item())
            if stats.numel() and stats.size(1) > 25
            else 0
        ),
        "b3_eligible_edges": (
            int(stats[0, 26].item())
            if stats.numel() and stats.size(1) > 26
            else 0
        ),
        "b3_staged_tasks": (
            int(stats[0, 27].item())
            if stats.numel() and stats.size(1) > 27
            else 0
        ),
        "b3_staged_edges": (
            int(stats[0, 28].item())
            if stats.numel() and stats.size(1) > 28
            else 0
        ),
        "b3_fallback_tasks": (
            int(stats[0, 29].item())
            if stats.numel() and stats.size(1) > 29
            else 0
        ),
        "b3_fallback_edges": (
            int(stats[0, 30].item())
            if stats.numel() and stats.size(1) > 30
            else 0
        ),
        "b3_edges_256_383": (
            int(stats[0, 31].item())
            if stats.numel() and stats.size(1) > 31
            else 0
        ),
        "b3_edges_384_511": (
            int(stats[0, 32].item())
            if stats.numel() and stats.size(1) > 32
            else 0
        ),
        "b3_edges_512": (
            int(stats[0, 33].item())
            if stats.numel() and stats.size(1) > 33
            else 0
        ),
        "b3_one_chunk_tasks": (
            int(stats[0, 34].item())
            if stats.numel() and stats.size(1) > 34
            else 0
        ),
        "b3_two_chunk_tasks": (
            int(stats[0, 35].item())
            if stats.numel() and stats.size(1) > 35
            else 0
        ),
        "b3_three_chunk_tasks": (
            int(stats[0, 36].item())
            if stats.numel() and stats.size(1) > 36
            else 0
        ),
        "b3_four_chunk_tasks": (
            int(stats[0, 37].item())
            if stats.numel() and stats.size(1) > 37
            else 0
        ),
        "b3_max_active_consumers": (
            int(stats[0, 38].item())
            if stats.numel() and stats.size(1) > 38
            else 0
        ),
        "b3_active_consumer_samples": (
            int(stats[0, 39].item())
            if stats.numel() and stats.size(1) > 39
            else 0
        ),
        "b3_active_consumer_sum": (
            int(stats[0, 40].item())
            if stats.numel() and stats.size(1) > 40
            else 0
        ),
        "b3_chunks_produced": (
            int(stats[0, 41].item())
            if stats.numel() and stats.size(1) > 41
            else 0
        ),
        "b3_chunks_consumed": (
            int(stats[0, 42].item())
            if stats.numel() and stats.size(1) > 42
            else 0
        ),
        "b3_producer_idle_loops": (
            int(stats[0, 43].item())
            if stats.numel() and stats.size(1) > 43
            else 0
        ),
        "logical_block_task_count": int(logical_task_edges.numel()),
    }
    aggregation_accumulators = {
        scale: _AggregationScaleAccumulator(scale)
        for scale in (*HASH_AGGREGATION_SCALES, None)
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
        for accumulator in aggregation_accumulators.values():
            accumulator.update(posts)
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
            and block_edge_budget == 0
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
        "v4_atomic_ratio": ratio(
            totals["v4_global_atomics"],
            totals["v4_input_edges"],
        ),
        "kernel_hash_atomic_ratio": ratio(
            (
                totals["kernel_hash_flush_atomics"]
                + totals["kernel_hash_fallback_atomics"]
            ),
            totals["kernel_hash_input_edges"],
        ),
        "kernel_hash_fallback_ratio": ratio(
            totals["kernel_hash_fallback_atomics"],
            totals["kernel_hash_input_edges"],
        ),
        "kernel_hash_average_probes": ratio(
            totals["kernel_hash_probe_attempts"],
            totals["kernel_hash_input_edges"],
        ),
        "logical_tasks_per_spike_block": ratio(
            totals["logical_block_task_count"],
            task_count,
        ),
        "b3_eligible_task_coverage": ratio(
            totals["b3_eligible_tasks"],
            totals["b3_ordinary_tasks"],
        ),
        "b3_eligible_edge_coverage": ratio(
            totals["b3_eligible_edges"],
            totals["b3_ordinary_edges"],
        ),
        "b3_staged_edge_coverage": ratio(
            totals["b3_staged_edges"],
            totals["b3_ordinary_edges"],
        ),
        "b3_average_active_consumers": ratio(
            totals["b3_active_consumer_sum"],
            totals["b3_active_consumer_samples"],
        ),
        "logical_task_edges_p50": task_quantile(0.50),
        "logical_task_edges_p90": task_quantile(0.90),
        "logical_task_edges_p99": task_quantile(0.99),
        "logical_task_edges_max": (
            int(logical_task_edges.max().item())
            if logical_task_edges.numel()
            else 0
        ),
        "original_task_edges_p50": original_task_quantile(0.50),
        "original_task_edges_p90": original_task_quantile(0.90),
        "original_task_edges_p99": original_task_quantile(0.99),
        "original_task_edges_max": (
            int(active_task_edges.max().item())
            if active_task_edges.numel()
            else 0
        ),
    }
    aggregation: dict[str, float | int] = {}
    for scale, accumulator in aggregation_accumulators.items():
        label = "full" if scale is None else str(scale)
        scale_stats = accumulator.summarize()
        aggregation.update(
            {
                f"aggregation_{label}_{key}": value
                for key, value in scale_stats.items()
            }
        )
    return {**totals, **derived, **aggregation}


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
    sample_output = run()
    input_permutation_ms, output_restore_ms = cuda_reorder_io_ms(
        workload,
        sample_output,
    )
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
        "block_edge_budget": int(
            os.environ.get("BTORCH_BLOCK_EDGE_BUDGET", "0")
        ),
        "long_segment_size": int(
            os.environ.get("BTORCH_LONG_SEGMENT_SIZE", "1024")
        ),
        "tile_reduce": {
            "0": "off",
            "1": "all",
            "2": "hinted",
        }[os.environ.get("BTORCH_TILE_REDUCE_MODE", "0")],
        "hash_aggregation": int(
            os.environ.get("BTORCH_BLOCK_HASH_AGGREGATION", "512")
        ),
        "hash_capacity": int(
            os.environ.get("BTORCH_BLOCK_HASH_CAPACITY", "512")
        ),
        "hash_max_probes": int(
            os.environ.get("BTORCH_BLOCK_HASH_MAX_PROBE", "4")
        ),
        "hash_min_edges": int(
            os.environ.get("BTORCH_BLOCK_HASH_MIN_EDGES", "256")
        ),
        "hash_used_slots": (
            os.environ.get("BTORCH_BLOCK_HASH_USED_SLOTS", "1") == "1"
        ),
        "reorder": workload.permutation.config.mode,
        "reorder_window": workload.permutation.config.local_window_size,
        "reorder_extreme_threshold": (
            workload.permutation.config.extreme_fanout_threshold
        ),
        "reorder_preprocess_ms": workload.reorder_preprocess_ms,
        "input_permutation_ms": input_permutation_ms,
        "output_restore_ms": output_restore_ms,
        "end_to_end_time_ms": (
            latency_ms + input_permutation_ms + output_restore_ms
        ),
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
    row.update(workload.reorder_stats)
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


def wait_for_idle_gpu(
    consecutive_samples: int,
    *,
    utilization_threshold: int = 2,
    interval_s: float = 0.2,
    timeout_s: float = 300.0,
) -> None:
    """Wait for a sustained idle window immediately before timing."""

    if consecutive_samples <= 0:
        return
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    physical_device = visible_devices.split(",", maxsplit=1)[0].strip()
    deadline = time.monotonic() + timeout_s
    idle_samples = 0
    while idle_samples < consecutive_samples:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_device}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        utilization = int(result.stdout.strip())
        idle_samples = (
            idle_samples + 1
            if utilization <= utilization_threshold
            else 0
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU did not reach a sustained idle window.")
        if idle_samples < consecutive_samples:
            time.sleep(interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("flybrain", "flywire_783", "uniform", "mice_column_v1"),
        default="flybrain",
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
        help="Compile the warp-private hash for edge-budget BlockTasks.",
    )
    parser.add_argument(
        "--hash-aggregation",
        type=int,
        choices=(128, 256, 512),
        default=512,
        help="Maximum logical edges accumulated by one hash window.",
    )
    parser.add_argument(
        "--hash-capacity",
        type=int,
        choices=(128, 256, 512),
        default=512,
        help="Number of shared-memory slots in each warp-private hash.",
    )
    parser.add_argument(
        "--hash-max-probes",
        type=int,
        choices=(4, 8, 16),
        default=4,
        help="Maximum linear probes before falling back to a global atomic.",
    )
    parser.add_argument(
        "--hash-min-edges",
        type=int,
        choices=(0, 64, 128, 192, 256),
        default=256,
        help="Minimum edges in a window before enabling the hash.",
    )
    parser.add_argument(
        "--hash-used-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush and clear only slots claimed by the current hash window.",
    )
    parser.add_argument(
        "--block-edge-budget",
        type=int,
        choices=(0, 128, 256, 512, 1024),
        default=0,
        help="Maximum ordinary logical edges per BlockTask; 0 keeps v3.",
    )
    parser.add_argument(
        "--long-segment-size",
        type=int,
        choices=(128, 256, 512, 1024, 2048),
        default=1024,
        help="Maximum edges in one extreme-row SegmentTask.",
    )
    parser.add_argument(
        "--tile-reduce",
        choices=("off", "all", "hinted"),
        default="off",
        help="Merge duplicate posts within each 32-edge logical tile.",
    )
    parser.add_argument(
        "--reorder",
        choices=(
            "identity",
            "global_cost_similar",
            "global_cost_bucket",
            "global_cost_balanced",
            "local_cost_similar",
            "local_cost_bucket",
            "global_similarity",
            "global_cost_similarity",
            "global_similarity_cost",
            "local_similarity",
            "local_cost_similarity",
            "local_similarity_cost",
        ),
        default="identity",
        help="Physically reorder neurons and CSR before persistent execution.",
    )
    parser.add_argument(
        "--reorder-window",
        type=int,
        default=256,
        help="Neuron window size used by local reorder strategies.",
    )
    parser.add_argument(
        "--reorder-extreme-threshold",
        type=int,
        default=256,
        help="Fanout at which rows sort after ordinary rows.",
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
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=None,
        help=(
            "Global recurrent weight scale. Defaults to 0.275 for FlyBrain "
            "and 0.15 for mice_column_v1 or uniform."
        ),
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--v-reset", type=float, default=0.0)
    parser.add_argument("--c-m", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--wait-idle-samples",
        type=int,
        default=0,
        help=(
            "Require consecutive <=2%% GPU-utilization samples immediately "
            "before timing; samples are 0.2 seconds apart."
        ),
    )
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    args.dataset, args.weight_scale = resolve_dataset_defaults(
        args.dataset,
        args.weight_scale,
    )
    if not 5 <= args.warmup <= 20:
        parser.error("--warmup must be between 5 and 20.")
    if args.repeat < 20:
        parser.error("--repeat must be at least 20.")
    if args.wait_idle_samples < 0:
        parser.error("--wait-idle-samples must be nonnegative.")
    if args.t_steps <= 0 or args.batch_size <= 0 or args.n_neuron <= 0:
        parser.error("--t-steps, --batch-size, and --n-neuron must be positive.")
    if args.grid_blocks is not None and args.grid_blocks <= 0:
        parser.error("--grid-blocks must be positive.")
    if args.reorder_window <= 0:
        parser.error("--reorder-window must be positive.")
    if args.reorder_extreme_threshold <= 0:
        parser.error("--reorder-extreme-threshold must be positive.")
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
    if args.block_edge_budget and not args.spike_block:
        parser.error("--block-edge-budget requires --spike-block.")
    if args.long_segment_size != 1024 and not args.spike_block:
        parser.error("--long-segment-size requires --spike-block.")
    if args.tile_reduce != "off" and not args.spike_block:
        parser.error("--tile-reduce requires --spike-block.")
    if args.tile_reduce != "off" and args.block_edge_budget == 0:
        parser.error("--tile-reduce requires a nonzero --block-edge-budget.")
    if args.block_hash and args.block_edge_budget == 0:
        parser.error("--block-hash requires a nonzero --block-edge-budget.")
    if args.block_hash and args.tile_reduce != "off":
        parser.error("--block-hash and --tile-reduce are mutually exclusive.")
    if args.reorder != "identity" and args.provider != "persistent":
        parser.error("--reorder is only valid with --provider persistent.")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the RSNN roofline benchmark.")

    if args.grid_blocks is not None:
        os.environ["BTORCH_PERSISTENT_GRID_BLOCKS"] = str(args.grid_blocks)
    os.environ["BTORCH_BLOCK_EDGE_BUDGET"] = str(args.block_edge_budget)
    os.environ["BTORCH_LONG_SEGMENT_SIZE"] = str(args.long_segment_size)
    os.environ["BTORCH_TILE_REDUCE_MODE"] = {
        "off": "0",
        "all": "1",
        "hinted": "2",
    }[args.tile_reduce]
    os.environ["BTORCH_BLOCK_HASH_AGGREGATION"] = str(
        args.hash_aggregation
    )
    os.environ["BTORCH_BLOCK_HASH_CAPACITY"] = str(args.hash_capacity)
    os.environ["BTORCH_BLOCK_HASH_MAX_PROBE"] = str(args.hash_max_probes)
    os.environ["BTORCH_BLOCK_HASH_MIN_EDGES"] = str(args.hash_min_edges)
    os.environ["BTORCH_BLOCK_HASH_USED_SLOTS"] = (
        "1" if args.hash_used_slots else "0"
    )

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
            block_edge_budget=args.block_edge_budget,
            hash_aggregation=args.hash_aggregation,
        )

    wait_for_idle_gpu(args.wait_idle_samples)
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
