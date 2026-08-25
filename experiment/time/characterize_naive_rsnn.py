"""Characterize phase-separated PyTorch and cuSPARSE RSNN execution.

The conventional baseline is a timestep loop without CUDA Graph capture or
per-timestep timing events. Separate Kineto/CUPTI passes sum raw Update and
cuSPARSE Propagation kernel-active time for both eager execution and CUDA Graph
replay. Each mode's launch-and-execution overhead is its synchronized wall time
after subtracting those two components. Neuron updates use eager PyTorch CUDA
operations; recurrent propagation uses the same direct cuSPARSE adapter as the
main RSNN benchmark. Controlled firing partitions are reported as rates in Hz.

Example:
    Run a small synthetic smoke benchmark:

    >>> python experiment/time/characterize_naive_rsnn.py \
    ...     --dataset uniform --n-neuron 8192 --fanout 32
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    make_recurrent_csr,
)
from benchmark.benchmark_rsnn_cudagraph_compare import (  # noqa: E402
    load_flybrain_csr,
    load_mice_column_v1_csr,
    make_torch_csr_weight,
    resolve_dataset_defaults,
)
from btorch.sparse import CSR  # noqa: E402


DEFAULT_FIRING_RATES_HZ = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
SCHEMA_VERSION = 4
TIMING_FIELDS = (
    "eager_wall_total_ms",
    "eager_gpu_total_ms",
    "eager_boundary_ms",
    "instrumented_wall_total_ms",
    "instrumented_gpu_total_ms",
    "update_phase_elapsed_ms",
    "current_phase_elapsed_ms",
    "inter_phase_residual_ms",
    "instrumentation_wall_delta_ms",
    "cudagraph_wall_total_ms",
    "cudagraph_gpu_total_ms",
    "graph_removable_wall_ms",
    "graph_removable_gpu_ms",
    "eager_update_kernel_active_ms",
    "eager_propagation_kernel_active_ms",
    "eager_launch_execution_overhead_ms",
    "cudagraph_update_kernel_active_ms",
    "cudagraph_propagation_kernel_active_ms",
    "cudagraph_launch_execution_overhead_ms",
    "eager_wall_us_per_step",
    "eager_gpu_us_per_step",
    "update_phase_elapsed_us_per_step",
    "current_phase_elapsed_us_per_step",
    "inter_phase_residual_us_per_step",
    "cudagraph_wall_us_per_step",
    "cudagraph_gpu_us_per_step",
    "graph_removable_wall_us_per_step",
    "graph_removable_gpu_us_per_step",
    "eager_update_kernel_active_us_per_step",
    "eager_propagation_kernel_active_us_per_step",
    "eager_launch_execution_overhead_us_per_step",
    "cudagraph_update_kernel_active_us_per_step",
    "cudagraph_propagation_kernel_active_us_per_step",
    "cudagraph_launch_execution_overhead_us_per_step",
    "update_fraction_of_instrumented_gpu",
    "current_fraction_of_instrumented_gpu",
    "inter_phase_fraction_of_instrumented_gpu",
    "graph_removable_wall_fraction_of_eager",
)


@dataclass(frozen=True)
class LoopTiming:
    """Store synchronized wall-clock and stream elapsed time."""

    wall_ms: float
    gpu_ms: float


@dataclass(frozen=True)
class KernelProfile:
    """Store CUPTI kernel-active durations for one complete loop."""

    update_ms: float
    current_ms: float
    update_kernel_count: int
    current_kernel_count: int
    ignored_device_event_count: int


@dataclass(frozen=True)
class EventPair:
    """Hold one reusable CUDA event pair."""

    start: torch.cuda.Event
    end: torch.cuda.Event

    @classmethod
    def create(cls) -> "EventPair":
        """Create and initialize timing events outside measurement."""

        pair = cls(
            start=torch.cuda.Event(enable_timing=True),
            end=torch.cuda.Event(enable_timing=True),
        )
        pair.start.record()
        pair.end.record()
        torch.cuda.synchronize()
        return pair


@dataclass
class RSNNState:
    """Hold mutable state and preallocated propagation tensors."""

    v: torch.Tensor
    psc: torch.Tensor
    spikes: torch.Tensor
    recurrent: torch.Tensor
    plan: int

    def reset(self) -> None:
        """Reset all state without reallocating tensors."""

        self.v.zero_()
        self.psc.zero_()
        self.spikes.zero_()
        self.recurrent.zero_()


@dataclass(frozen=True)
class PhaseEvents:
    """Store CUDA event pairs for every measured timestep."""

    update_start: list[torch.cuda.Event]
    update_end: list[torch.cuda.Event]
    current_start: list[torch.cuda.Event]
    current_end: list[torch.cuda.Event]
    total_start: torch.cuda.Event
    total_end: torch.cuda.Event

    @classmethod
    def create(cls, t_steps: int) -> "PhaseEvents":
        """Create timing-enabled CUDA events outside the measured region."""

        def event_list() -> list[torch.cuda.Event]:
            return [torch.cuda.Event(enable_timing=True) for _ in range(t_steps)]

        return cls(
            update_start=event_list(),
            update_end=event_list(),
            current_start=event_list(),
            current_end=event_list(),
            total_start=torch.cuda.Event(enable_timing=True),
            total_end=torch.cuda.Event(enable_timing=True),
        )

    def phase_times_us(self) -> tuple[list[float], list[float]]:
        """Return per-timestep update and propagation durations."""

        update = [
            start.elapsed_time(end) * 1_000.0
            for start, end in zip(self.update_start, self.update_end, strict=True)
        ]
        current = [
            start.elapsed_time(end) * 1_000.0
            for start, end in zip(
                self.current_start,
                self.current_end,
                strict=True,
            )
        ]
        return update, current

    def initialize(self) -> None:
        """Materialize lazy CUDA event handles outside the timed region."""

        for events in (
            self.update_start,
            self.update_end,
            self.current_start,
            self.current_end,
        ):
            for event in events:
                event.record()
        self.total_start.record()
        self.total_end.record()
        torch.cuda.synchronize()


def load_network(args: argparse.Namespace, device: torch.device) -> CSR:
    """Load one dataset or construct the benchmark's uniform network."""

    if args.dataset == "flybrain":
        return load_flybrain_csr(
            args.connectome_root,
            weight_scale=args.weight_scale,
            device=device,
        )
    if args.dataset == "mice_column_v1":
        return load_mice_column_v1_csr(
            args.connectome_root,
            weight_scale=args.weight_scale,
            device=device,
        )
    case = BenchCase(
        n_neuron=args.n_neuron,
        batch_size=1,
        t_steps=args.t_steps,
        fanout=args.fanout,
        event_rate=0.0,
        weight_scale=args.weight_scale,
    )
    return make_recurrent_csr(case, device)


def prepare_state(
    matrix: CSR,
    t_steps: int,
    extension,
    device: torch.device,
) -> tuple[RSNNState, int]:
    """Prepare state, transposed CSR storage, descriptors, and workspace."""

    weight = make_torch_csr_weight(matrix)
    crow = weight.crow_indices().to(torch.int32).contiguous()
    col = weight.col_indices().to(torch.int32).contiguous()
    values = weight.values().to(torch.float32).contiguous()
    n_neuron = matrix.shape[0]
    spikes = torch.empty(
        (t_steps, 1, n_neuron),
        device=device,
        dtype=torch.float32,
    )
    recurrent = torch.empty(
        (1, n_neuron),
        device=device,
        dtype=torch.float32,
    )
    plan = extension.prepare(crow, col, values, spikes, recurrent, 1)
    state = RSNNState(
        v=torch.empty_like(recurrent),
        psc=torch.empty_like(recurrent),
        spikes=spikes,
        recurrent=recurrent,
        plan=plan,
    )
    state.reset()
    return state, int(extension.workspace_bytes(plan))


def _coprime_stride(n_neuron: int) -> int:
    """Choose a deterministic stride that cycles through all neuron IDs."""

    stride = min(104_729, max(n_neuron - 1, 1))
    while stride > 1 and math.gcd(stride, n_neuron) != 1:
        stride -= 1
    return stride


def generate_spike_control(
    n_neuron: int,
    t_steps: int,
    activity: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Generate fixed exact-count active indices for every timestep.

    One random neuron permutation selects the initial active set. A coprime
    cyclic shift changes the set across timesteps without timed random-number
    generation. The returned indices are replayed for every repeat.
    """

    n_active = min(n_neuron, max(0, int(round(n_neuron * activity))))
    if n_active == 0:
        return torch.empty((t_steps, 0), device=device, dtype=torch.long), 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial = torch.randperm(
        n_neuron,
        device=device,
        dtype=torch.long,
        generator=generator,
    )[:n_active]
    offsets = torch.arange(t_steps, device=device, dtype=torch.long)
    offsets.mul_(_coprime_stride(n_neuron))
    indices = initial.unsqueeze(0) + offsets.unsqueeze(1)
    indices.remainder_(n_neuron)
    return indices, n_active


def neuron_update(
    state: RSNNState,
    active_indices: torch.Tensor,
    timestep: int,
    *,
    dt: float,
    tau_mem: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    decay: float,
) -> None:
    """Update neuron state and construct one controlled spike vector."""

    state.psc.mul_(decay).add_(state.recurrent)
    state.v.add_(dt * (-(state.v - v_reset) / tau_mem + state.psc / c_m))
    spike_t = state.spikes[timestep, 0]
    spike_t.zero_()
    if active_indices.numel():
        spike_t.scatter_(0, active_indices, 1.0)
    state.v.sub_((v_threshold - v_reset) * spike_t)


def run_one_step(
    state: RSNNState,
    active_indices: torch.Tensor,
    timestep: int,
    extension,
    args: argparse.Namespace,
    decay: float,
) -> None:
    """Run one eager PyTorch update followed by direct cuSPARSE propagation."""

    neuron_update(
        state,
        active_indices,
        timestep,
        dt=args.dt,
        tau_mem=args.tau_mem,
        v_threshold=args.v_threshold,
        v_reset=args.v_reset,
        c_m=args.c_m,
        decay=decay,
    )
    extension.propagate(state.plan, timestep)


def run_uninstrumented_loop(
    state: RSNNState,
    spike_indices: torch.Tensor,
    extension,
    args: argparse.Namespace,
    decay: float,
) -> None:
    """Run the ordinary timestep loop without per-phase CUDA events."""

    for timestep in range(args.t_steps):
        run_one_step(
            state,
            spike_indices[timestep],
            timestep,
            extension,
            args,
            decay,
        )


def run_instrumented_loop(
    state: RSNNState,
    spike_indices: torch.Tensor,
    extension,
    args: argparse.Namespace,
    decay: float,
    events: PhaseEvents,
) -> None:
    """Run a separate diagnostic loop with phase-envelope events."""

    for timestep in range(args.t_steps):
        events.update_start[timestep].record()
        neuron_update(
            state,
            spike_indices[timestep],
            timestep,
            dt=args.dt,
            tau_mem=args.tau_mem,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
            c_m=args.c_m,
            decay=decay,
        )
        events.update_end[timestep].record()

        events.current_start[timestep].record()
        extension.propagate(state.plan, timestep)
        events.current_end[timestep].record()


def time_callable(
    run: Callable[[], object],
    state: RSNNState,
    events: EventPair,
) -> LoopTiming:
    """Time one callable after an untimed state reset and synchronization."""

    state.reset()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    events.start.record()
    run()
    events.end.record()
    events.end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1_000.0
    return LoopTiming(
        wall_ms=wall_ms,
        gpu_ms=events.start.elapsed_time(events.end),
    )


def profile_kernel_active_time(
    run: Callable[[], object],
    state: RSNNState,
) -> KernelProfile:
    """Measure device kernel-active time with Kineto/CUPTI.

    This is a separate diagnostic pass. Raw CUDA kernel activities are summed,
    rather than CUDA annotation ranges or phase event envelopes. cuSPARSE
    kernels are identified by their demangled ``cusparse::`` names; all other
    workload kernels belong to the PyTorch update. Profiler-internal activity
    buffer events are explicitly excluded.
    """

    from torch.profiler import ProfilerActivity, profile

    state.reset()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        acc_events=True,
    ) as trace:
        run()
        torch.cuda.synchronize()

    update_us = 0.0
    current_us = 0.0
    update_count = 0
    current_count = 0
    ignored_count = 0
    for event in trace.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        duration_us = float(event.self_device_time_total)
        if duration_us <= 0.0 or event.is_user_annotation:
            continue
        name = str(event.key)
        if name.startswith("Activity Buffer Request"):
            ignored_count += 1
        elif "cusparse::" in name:
            current_us += duration_us
            current_count += 1
        else:
            update_us += duration_us
            update_count += 1

    if update_count == 0 or current_count == 0:
        raise RuntimeError(
            "CUPTI did not report both update and cuSPARSE kernels; "
            f"update_count={update_count}, current_count={current_count}"
        )
    return KernelProfile(
        update_ms=update_us / 1_000.0,
        current_ms=current_us / 1_000.0,
        update_kernel_count=update_count,
        current_kernel_count=current_count,
        ignored_device_event_count=ignored_count,
    )


def prepare_cudagraph(
    state: RSNNState,
    spike_indices: torch.Tensor,
    extension,
    args: argparse.Namespace,
    decay: float,
) -> torch.cuda.CUDAGraph:
    """Capture the same complete loop for a dispatch-overhead proxy."""

    current_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(current_stream)
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            run_uninstrumented_loop(
                state,
                spike_indices,
                extension,
                args,
                decay,
            )
    current_stream.wait_stream(side_stream)
    torch.cuda.synchronize()
    state.reset()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_uninstrumented_loop(
            state,
            spike_indices,
            extension,
            args,
            decay,
        )
    torch.cuda.synchronize()
    return graph


def _warmup(
    state: RSNNState,
    spike_indices: torch.Tensor,
    extension,
    args: argparse.Namespace,
) -> None:
    """Warm up PyTorch, pybind, and cuSPARSE without synchronizing each step."""

    state.reset()
    decay = math.exp(-args.dt / args.tau_syn)
    for step in range(args.warmup):
        timestep = step % args.t_steps
        run_one_step(
            state,
            spike_indices[timestep],
            timestep,
            extension,
            args,
            decay,
        )
    torch.cuda.synchronize()


def benchmark_activity(
    state: RSNNState,
    spike_indices: torch.Tensor,
    n_active: int,
    activity: float,
    spike_seed: int,
    extension,
    args: argparse.Namespace,
    common: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Measure eager, instrumented, and CUDA Graph diagnostic executions."""

    _warmup(state, spike_indices, extension, args)
    repeat_rows: list[dict[str, object]] = []
    timestep_rows: list[dict[str, object]] = []
    actual_partition = n_active / state.v.numel()
    requested_rate_hz = activity * 1_000.0 / args.dt
    actual_rate_hz = actual_partition * 1_000.0 / args.dt
    decay = math.exp(-args.dt / args.tau_syn)
    phase_events = PhaseEvents.create(args.t_steps)
    phase_events.initialize()
    eager_events = EventPair.create()
    graph_events = EventPair.create()

    graph = None
    cudagraph_status = "disabled"
    if args.measure_cudagraph_proxy:
        try:
            graph = prepare_cudagraph(
                state,
                spike_indices,
                extension,
                args,
                decay,
            )
            cudagraph_status = "ok"
        except RuntimeError as error:
            cudagraph_status = f"capture_error: {error}"
            print(f"warning: CUDA Graph proxy unavailable: {error}")

    eager_run = lambda: run_uninstrumented_loop(  # noqa: E731
        state,
        spike_indices,
        extension,
        args,
        decay,
    )
    instrumented_run = lambda: run_instrumented_loop(  # noqa: E731
        state,
        spike_indices,
        extension,
        args,
        decay,
        phase_events,
    )
    graph_run = graph.replay if graph is not None else None

    for repeat in range(args.repeat):
        if graph_run is not None and repeat % 2:
            graph_timing = time_callable(graph_run, state, graph_events)
            eager_timing = time_callable(eager_run, state, eager_events)
        else:
            eager_timing = time_callable(eager_run, state, eager_events)
            graph_timing = (
                time_callable(graph_run, state, graph_events)
                if graph_run is not None
                else None
            )

        eager_kernel_profile = None
        graph_kernel_profile = None
        eager_kernel_profile_status = "disabled"
        graph_kernel_profile_status = "disabled"
        if args.profile_kernel_active:
            profile_targets = [("eager", eager_run)]
            if graph_run is not None:
                profile_targets.append(("cudagraph", graph_run))
            if repeat % 2:
                profile_targets.reverse()
            for profile_name, profile_run in profile_targets:
                try:
                    result = profile_kernel_active_time(profile_run, state)
                except RuntimeError as error:
                    status = f"profile_error: {error}"
                    print(f"warning: {profile_name} kernel profile failed: {error}")
                else:
                    status = "ok"
                    if profile_name == "eager":
                        eager_kernel_profile = result
                    else:
                        graph_kernel_profile = result
                if profile_name == "eager":
                    eager_kernel_profile_status = status
                else:
                    graph_kernel_profile_status = status

        instrumented_timing = time_callable(
            instrumented_run,
            state,
            EventPair(phase_events.total_start, phase_events.total_end),
        )
        update_us, current_us = phase_events.phase_times_us()
        update_ms = sum(update_us) / 1_000.0
        current_ms = sum(current_us) / 1_000.0
        raw_inter_phase_ms = instrumented_timing.gpu_ms - update_ms - current_ms
        inter_phase_ms = max(0.0, raw_inter_phase_ms)
        eager_boundary_ms = max(0.0, eager_timing.wall_ms - eager_timing.gpu_ms)
        instrumentation_delta_ms = instrumented_timing.wall_ms - eager_timing.wall_ms

        if graph_timing is None:
            graph_wall_ms = float("nan")
            graph_gpu_ms = float("nan")
            raw_graph_wall_ms = float("nan")
            raw_graph_gpu_ms = float("nan")
            graph_removable_wall_ms = float("nan")
            graph_removable_gpu_ms = float("nan")
        else:
            graph_wall_ms = graph_timing.wall_ms
            graph_gpu_ms = graph_timing.gpu_ms
            raw_graph_wall_ms = eager_timing.wall_ms - graph_wall_ms
            raw_graph_gpu_ms = eager_timing.gpu_ms - graph_gpu_ms
            graph_removable_wall_ms = max(0.0, raw_graph_wall_ms)
            graph_removable_gpu_ms = max(0.0, raw_graph_gpu_ms)

        if eager_kernel_profile is None:
            eager_update_kernel_ms = float("nan")
            eager_propagation_kernel_ms = float("nan")
            eager_overhead_ms = float("nan")
            eager_update_kernel_count = 0
            eager_propagation_kernel_count = 0
            eager_ignored_device_event_count = 0
        else:
            eager_update_kernel_ms = eager_kernel_profile.update_ms
            eager_propagation_kernel_ms = eager_kernel_profile.current_ms
            eager_overhead_ms = max(
                0.0,
                eager_timing.wall_ms
                - eager_update_kernel_ms
                - eager_propagation_kernel_ms,
            )
            eager_update_kernel_count = eager_kernel_profile.update_kernel_count
            eager_propagation_kernel_count = eager_kernel_profile.current_kernel_count
            eager_ignored_device_event_count = (
                eager_kernel_profile.ignored_device_event_count
            )

        if graph_kernel_profile is None or graph_timing is None:
            graph_update_kernel_ms = float("nan")
            graph_propagation_kernel_ms = float("nan")
            graph_overhead_ms = float("nan")
            graph_update_kernel_count = 0
            graph_propagation_kernel_count = 0
            graph_ignored_device_event_count = 0
            component_consistency_status = "unavailable"
        else:
            graph_update_kernel_ms = graph_kernel_profile.update_ms
            graph_propagation_kernel_ms = graph_kernel_profile.current_ms
            graph_overhead_ms = max(
                0.0,
                graph_timing.wall_ms
                - graph_update_kernel_ms
                - graph_propagation_kernel_ms,
            )
            graph_update_kernel_count = graph_kernel_profile.update_kernel_count
            graph_propagation_kernel_count = graph_kernel_profile.current_kernel_count
            graph_ignored_device_event_count = (
                graph_kernel_profile.ignored_device_event_count
            )
            counts_match = (
                eager_update_kernel_count == graph_update_kernel_count
                and eager_propagation_kernel_count == graph_propagation_kernel_count
            )
            times_match = math.isclose(
                eager_update_kernel_ms,
                graph_update_kernel_ms,
                rel_tol=args.component_consistency_rtol,
            ) and math.isclose(
                eager_propagation_kernel_ms,
                graph_propagation_kernel_ms,
                rel_tol=args.component_consistency_rtol,
            )
            component_consistency_status = (
                "ok" if counts_match and times_match else "mismatch"
            )

        divisor = args.t_steps
        row = {
            **common,
            "requested_average_firing_partition": activity,
            "actual_average_firing_partition": actual_partition,
            "requested_firing_rate_hz": requested_rate_hz,
            "actual_firing_rate_hz": actual_rate_hz,
            "n_active_per_step": n_active,
            "spike_seed": spike_seed,
            "repeat": repeat,
            "cudagraph_status": cudagraph_status,
            "eager_kernel_profile_status": eager_kernel_profile_status,
            "cudagraph_kernel_profile_status": graph_kernel_profile_status,
            "component_consistency_status": component_consistency_status,
            "eager_wall_total_ms": eager_timing.wall_ms,
            "eager_gpu_total_ms": eager_timing.gpu_ms,
            "eager_boundary_ms": eager_boundary_ms,
            "instrumented_wall_total_ms": instrumented_timing.wall_ms,
            "instrumented_gpu_total_ms": instrumented_timing.gpu_ms,
            "update_phase_elapsed_ms": update_ms,
            "current_phase_elapsed_ms": current_ms,
            "raw_inter_phase_residual_ms": raw_inter_phase_ms,
            "inter_phase_residual_ms": inter_phase_ms,
            "instrumentation_wall_delta_ms": instrumentation_delta_ms,
            "cudagraph_wall_total_ms": graph_wall_ms,
            "cudagraph_gpu_total_ms": graph_gpu_ms,
            "raw_graph_removable_wall_ms": raw_graph_wall_ms,
            "raw_graph_removable_gpu_ms": raw_graph_gpu_ms,
            "graph_removable_wall_ms": graph_removable_wall_ms,
            "graph_removable_gpu_ms": graph_removable_gpu_ms,
            "eager_update_kernel_active_ms": eager_update_kernel_ms,
            "eager_propagation_kernel_active_ms": eager_propagation_kernel_ms,
            "eager_launch_execution_overhead_ms": eager_overhead_ms,
            "cudagraph_update_kernel_active_ms": graph_update_kernel_ms,
            "cudagraph_propagation_kernel_active_ms": (graph_propagation_kernel_ms),
            "cudagraph_launch_execution_overhead_ms": graph_overhead_ms,
            "eager_update_kernel_count": eager_update_kernel_count,
            "eager_propagation_kernel_count": eager_propagation_kernel_count,
            "eager_ignored_device_event_count": (eager_ignored_device_event_count),
            "cudagraph_update_kernel_count": graph_update_kernel_count,
            "cudagraph_propagation_kernel_count": (graph_propagation_kernel_count),
            "cudagraph_ignored_device_event_count": (graph_ignored_device_event_count),
            "eager_wall_us_per_step": eager_timing.wall_ms * 1_000.0 / divisor,
            "eager_gpu_us_per_step": eager_timing.gpu_ms * 1_000.0 / divisor,
            "update_phase_elapsed_us_per_step": update_ms * 1_000.0 / divisor,
            "current_phase_elapsed_us_per_step": current_ms * 1_000.0 / divisor,
            "inter_phase_residual_us_per_step": (inter_phase_ms * 1_000.0 / divisor),
            "cudagraph_wall_us_per_step": graph_wall_ms * 1_000.0 / divisor,
            "cudagraph_gpu_us_per_step": graph_gpu_ms * 1_000.0 / divisor,
            "graph_removable_wall_us_per_step": (
                graph_removable_wall_ms * 1_000.0 / divisor
            ),
            "graph_removable_gpu_us_per_step": (
                graph_removable_gpu_ms * 1_000.0 / divisor
            ),
            "eager_update_kernel_active_us_per_step": (
                eager_update_kernel_ms * 1_000.0 / divisor
            ),
            "eager_propagation_kernel_active_us_per_step": (
                eager_propagation_kernel_ms * 1_000.0 / divisor
            ),
            "eager_launch_execution_overhead_us_per_step": (
                eager_overhead_ms * 1_000.0 / divisor
            ),
            "cudagraph_update_kernel_active_us_per_step": (
                graph_update_kernel_ms * 1_000.0 / divisor
            ),
            "cudagraph_propagation_kernel_active_us_per_step": (
                graph_propagation_kernel_ms * 1_000.0 / divisor
            ),
            "cudagraph_launch_execution_overhead_us_per_step": (
                graph_overhead_ms * 1_000.0 / divisor
            ),
            "update_fraction_of_instrumented_gpu": (
                update_ms / instrumented_timing.gpu_ms
            ),
            "current_fraction_of_instrumented_gpu": (
                current_ms / instrumented_timing.gpu_ms
            ),
            "inter_phase_fraction_of_instrumented_gpu": (
                inter_phase_ms / instrumented_timing.gpu_ms
            ),
            "graph_removable_wall_fraction_of_eager": (
                graph_removable_wall_ms / eager_timing.wall_ms
            ),
        }
        repeat_rows.append(row)

        if args.save_timestep_data:
            timestep_rows.extend(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": common["run_id"],
                    "dataset": common["dataset"],
                    "requested_average_firing_partition": activity,
                    "actual_average_firing_partition": actual_partition,
                    "requested_firing_rate_hz": requested_rate_hz,
                    "actual_firing_rate_hz": actual_rate_hz,
                    "n_active_per_step": n_active,
                    "spike_seed": spike_seed,
                    "repeat": repeat,
                    "timestep": timestep,
                    "update_phase_elapsed_us": update_time,
                    "current_phase_elapsed_us": current_time,
                }
                for timestep, (update_time, current_time) in enumerate(
                    zip(update_us, current_us, strict=True)
                )
            )
    return repeat_rows, timestep_rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate repeat samples into plot-ready firing-rate rows."""

    activities = sorted(
        {float(row["requested_average_firing_partition"]) for row in rows}
    )
    aggregated: list[dict[str, object]] = []
    for activity in activities:
        group = [
            row
            for row in rows
            if float(row["requested_average_firing_partition"]) == activity
        ]
        first = group[0]
        output: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": first["run_id"],
            "dataset": first["dataset"],
            "requested_average_firing_partition": activity,
            "actual_average_firing_partition": first["actual_average_firing_partition"],
            "requested_firing_rate_hz": first["requested_firing_rate_hz"],
            "actual_firing_rate_hz": first["actual_firing_rate_hz"],
            "n_active_per_step": first["n_active_per_step"],
            "spike_seed": first["spike_seed"],
            "cudagraph_status": first["cudagraph_status"],
            "eager_kernel_profile_status": first["eager_kernel_profile_status"],
            "cudagraph_kernel_profile_status": first["cudagraph_kernel_profile_status"],
            "component_consistency_status": (
                "ok"
                if all(row["component_consistency_status"] == "ok" for row in group)
                else "mismatch"
            ),
            "repeat_count": len(group),
        }
        for field in TIMING_FIELDS:
            values = [float(row[field]) for row in group]
            output[f"{field}_mean"] = statistics.mean(values)
            output[f"{field}_median"] = statistics.median(values)
            output[f"{field}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            output[f"{field}_min"] = min(values)
            output[f"{field}_max"] = max(values)
        aggregated.append(output)
    return aggregated


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write dictionaries as one CSV with a stable header."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("flybrain", "flywire_783", "mice_column_v1", "uniform"),
        default="flybrain",
    )
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument("--n-neuron", type=int, default=8192)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--weight-scale", type=float, default=None)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--firing-rates-hz",
        type=float,
        nargs="+",
        default=list(DEFAULT_FIRING_RATES_HZ),
        help=(
            "Controlled average firing rates in Hz. Rates are converted to "
            "per-timestep firing partitions using partition = rate * dt / 1000."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--v-reset", type=float, default=0.0)
    parser.add_argument("--c-m", type=float, default=1.0)
    parser.add_argument(
        "--save-timestep-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--measure-cudagraph-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare eager execution with a CUDA Graph replay of the same "
            "loop. The difference is a dispatch proxy, not pure launch time."
        ),
    )
    parser.add_argument(
        "--profile-kernel-active",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a separate Kineto/CUPTI pass to sum raw update and cuSPARSE "
            "kernel-active durations, excluding host launch/API time."
        ),
    )
    parser.add_argument(
        "--component-consistency-rtol",
        type=float,
        default=0.15,
        help=(
            "Relative tolerance used to check that eager and CUDA Graph "
            "Update/Propagation kernel-active times agree (default: 0.15)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    args.dataset, args.weight_scale = resolve_dataset_defaults(
        args.dataset,
        args.weight_scale,
    )
    if args.n_neuron <= 0 or args.fanout < 0:
        parser.error("--n-neuron must be positive and --fanout non-negative")
    if args.t_steps <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("invalid --t-steps, --warmup, or --repeat")
    if not args.firing_rates_hz or any(
        not 0.0 <= value <= 50.0 for value in args.firing_rates_hz
    ):
        parser.error("--firing-rates-hz must contain values in [0, 50]")
    if args.dt <= 0.0 or args.tau_mem <= 0.0 or args.tau_syn <= 0.0:
        parser.error("--dt, --tau-mem, and --tau-syn must be positive")
    if args.c_m <= 0.0:
        parser.error("--c-m must be positive")
    if not 0.0 <= args.component_consistency_rtol <= 1.0:
        parser.error("--component-consistency-rtol must be in [0, 1]")
    args.activities = [
        firing_rate_hz * args.dt / 1_000.0 for firing_rate_hz in args.firing_rates_hz
    ]
    return args


def main() -> None:
    """Run the firing-rate sweep and save raw and aggregate timing data."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for direct cuSPARSE characterization.")
    device = torch.device("cuda", torch.cuda.current_device())

    from benchmark.cusparse_rsnn import load

    extension = load()
    matrix = load_network(args, device)
    state, workspace_bytes = prepare_state(
        matrix,
        args.t_steps,
        extension,
        device,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    common: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": args.dataset,
        "n_neuron": matrix.shape[0],
        "edge_count": matrix.indices.numel(),
        "mean_fanout": matrix.indices.numel() / matrix.shape[0],
        "t_steps": args.t_steps,
        "warmup_steps": args.warmup,
        "seed": args.seed,
        "timing_method": (
            "separate_eager_and_cudagraph_wall_cupti_profiles_plus_phase_events"
        ),
        "update_backend": "pytorch_cuda_eager",
        "current_backend": "cusparse_direct_eager",
        "cusparse_primitive": "SpMV",
        "cusparse_algorithm": "CUSPARSE_SPMV_ALG_DEFAULT",
        "baseline_cuda_graph": False,
        "cudagraph_proxy_requested": args.measure_cudagraph_proxy,
        "kernel_active_profile_requested": args.profile_kernel_active,
        "component_consistency_rtol": args.component_consistency_rtol,
    }

    all_repeat_rows: list[dict[str, object]] = []
    all_timestep_rows: list[dict[str, object]] = []
    for activity_index, activity in enumerate(args.activities):
        spike_seed = args.seed + activity_index
        spike_indices, n_active = generate_spike_control(
            matrix.shape[0],
            args.t_steps,
            activity,
            spike_seed,
            device,
        )
        requested_rate_hz = activity * 1_000.0 / args.dt
        actual_partition = n_active / matrix.shape[0]
        actual_rate_hz = actual_partition * 1_000.0 / args.dt
        print(
            f"rate={requested_rate_hz:g} Hz actual={actual_rate_hz:.4f} Hz "
            f"partition={actual_partition:.6f} active={n_active}"
        )
        repeat_rows, timestep_rows = benchmark_activity(
            state,
            spike_indices,
            n_active,
            activity,
            spike_seed,
            extension,
            args,
            common,
        )
        all_repeat_rows.extend(repeat_rows)
        all_timestep_rows.extend(timestep_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    breakdown_path = args.output_dir / "naive_rsnn_breakdown_v4.csv"
    aggregate_path = args.output_dir / "naive_rsnn_aggregate_v4.csv"
    timestep_path = args.output_dir / "naive_rsnn_timesteps_v4.csv"
    metadata_path = args.output_dir / "naive_rsnn_metadata_v4.json"
    save_csv(all_repeat_rows, breakdown_path)
    save_csv(aggregate_rows(all_repeat_rows), aggregate_path)
    if args.save_timestep_data:
        save_csv(all_timestep_rows, timestep_path)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "network": {
            "n_neuron": matrix.shape[0],
            "edge_count": matrix.indices.numel(),
            "mean_fanout": matrix.indices.numel() / matrix.shape[0],
            "weight_scale": args.weight_scale,
            "workspace_bytes": workspace_bytes,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": torch.cuda.get_device_capability(device),
        },
        "semantics": {
            "controlled_activity": True,
            "average_firing_partition": (
                "mean fraction of neurons forced to spike per timestep"
            ),
            "firing_rate_conversion": (
                "rate_hz = average_firing_partition * 1000 / dt_ms"
            ),
            "spike_patterns_replayed_across_repeats": True,
            "spike_construction_in_update_phase": True,
            "random_generation_in_timing": False,
            "synchronize_per_timestep": False,
            "eager_baseline": ("ordinary loop with only one outer CUDA event pair"),
            "phase_measurement": (
                "separate instrumented pass; values are stream elapsed "
                "phase envelopes, not pure kernel-active time"
            ),
            "kernel_active_measurement": (
                "separate eager and CUDA Graph Kineto/CUPTI passes summing raw "
                "CUDA kernel activity; cuSPARSE kernels are classified by "
                "demangled cusparse:: names"
            ),
            "three_way_launch_execution_overhead": (
                "for each execution mode, max(0, wall time - update kernel "
                "active - cuSPARSE propagation kernel active); includes "
                "critical-path launch/API, framework, scheduling, and GPU idle"
            ),
            "component_consistency_check": (
                "eager and CUDA Graph Update/Propagation raw kernel-active "
                f"durations and kernel counts, rtol={args.component_consistency_rtol}"
            ),
            "eager_boundary": (
                "max(0, eager wall - eager GPU); not cumulative host launch time"
            ),
            "inter_phase_residual": (
                "max(0, instrumented GPU - update envelope - current envelope); "
                "not launch overhead"
            ),
            "graph_removable_proxy": (
                "max(0, eager - CUDA Graph replay); includes graph-removable "
                "Python, framework, API submission, and device dispatch effects"
            ),
            "launch_overhead_claim": "not measured exactly",
        },
        "outputs": {
            "repeat_samples": breakdown_path.name,
            "aggregate": aggregate_path.name,
            "timesteps": timestep_path.name if args.save_timestep_data else None,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved repeat samples: {breakdown_path}")
    print(f"saved aggregate data: {aggregate_path}")
    if args.save_timestep_data:
        print(f"saved timestep data: {timestep_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
