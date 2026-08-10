"""Characterize phase-separated PyTorch and cuSPARSE RSNN execution.

This script measures a conventional timestep loop without CUDA Graph capture.
Neuron-state updates and controlled spike construction use eager PyTorch CUDA
operations; recurrent propagation uses the same direct cuSPARSE adapter as the
main RSNN benchmark. CUDA events measure each phase while synchronized wall
time measures the complete loop.

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


DEFAULT_ACTIVITIES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
SCHEMA_VERSION = 1
TIMING_FIELDS = (
    "wall_total_ms",
    "gpu_total_ms",
    "update_gpu_ms",
    "current_gpu_ms",
    "gpu_gap_ms",
    "host_overhead_ms",
    "execution_overhead_ms",
    "wall_us_per_step",
    "gpu_total_us_per_step",
    "update_us_per_step",
    "current_us_per_step",
    "gpu_gap_us_per_step",
    "host_overhead_us_per_step",
    "execution_overhead_us_per_step",
)


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


def run_one_step(
    state: RSNNState,
    active_indices: torch.Tensor,
    timestep: int,
    extension,
    *,
    dt: float,
    tau_mem: float,
    tau_syn: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
) -> None:
    """Run one eager PyTorch update followed by direct cuSPARSE propagation."""

    decay = math.exp(-dt / tau_syn)
    state.psc.mul_(decay).add_(state.recurrent)
    state.v.add_(dt * (-(state.v - v_reset) / tau_mem + state.psc / c_m))
    spike_t = state.spikes[timestep, 0]
    spike_t.zero_()
    if active_indices.numel():
        spike_t.scatter_(0, active_indices, 1.0)
    state.v.sub_((v_threshold - v_reset) * spike_t)
    extension.propagate(state.plan, timestep)


def _warmup(
    state: RSNNState,
    spike_indices: torch.Tensor,
    extension,
    args: argparse.Namespace,
) -> None:
    """Warm up PyTorch, pybind, and cuSPARSE without synchronizing each step."""

    state.reset()
    for step in range(args.warmup):
        timestep = step % args.t_steps
        run_one_step(
            state,
            spike_indices[timestep],
            timestep,
            extension,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
            c_m=args.c_m,
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
    """Measure all repeats for one controlled firing rate."""

    _warmup(state, spike_indices, extension, args)
    repeat_rows: list[dict[str, object]] = []
    timestep_rows: list[dict[str, object]] = []
    actual_activity = n_active / state.v.numel()
    decay = math.exp(-args.dt / args.tau_syn)
    events = PhaseEvents.create(args.t_steps)
    events.initialize()

    for repeat in range(args.repeat):
        state.reset()
        torch.cuda.synchronize()

        wall_start = time.perf_counter()
        events.total_start.record()
        for timestep in range(args.t_steps):
            events.update_start[timestep].record()

            state.psc.mul_(decay).add_(state.recurrent)
            state.v.add_(
                args.dt
                * (-(state.v - args.v_reset) / args.tau_mem + state.psc / args.c_m)
            )
            spike_t = state.spikes[timestep, 0]
            spike_t.zero_()
            active_t = spike_indices[timestep]
            if active_t.numel():
                spike_t.scatter_(0, active_t, 1.0)
            state.v.sub_((args.v_threshold - args.v_reset) * spike_t)

            events.update_end[timestep].record()
            events.current_start[timestep].record()
            extension.propagate(state.plan, timestep)
            events.current_end[timestep].record()

        events.total_end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1_000.0

        update_us, current_us = events.phase_times_us()
        update_ms = sum(update_us) / 1_000.0
        current_ms = sum(current_us) / 1_000.0
        gpu_total_ms = events.total_start.elapsed_time(events.total_end)
        raw_gpu_gap_ms = gpu_total_ms - update_ms - current_ms
        raw_host_overhead_ms = wall_ms - gpu_total_ms
        gpu_gap_ms = max(0.0, raw_gpu_gap_ms)
        host_overhead_ms = max(0.0, raw_host_overhead_ms)
        execution_overhead_ms = max(0.0, wall_ms - update_ms - current_ms)
        divisor = args.t_steps
        row = {
            **common,
            "requested_activity": activity,
            "actual_activity": actual_activity,
            "n_active_per_step": n_active,
            "spike_seed": spike_seed,
            "repeat": repeat,
            "wall_total_ms": wall_ms,
            "gpu_total_ms": gpu_total_ms,
            "update_gpu_ms": update_ms,
            "current_gpu_ms": current_ms,
            "raw_gpu_gap_ms": raw_gpu_gap_ms,
            "gpu_gap_ms": gpu_gap_ms,
            "raw_host_overhead_ms": raw_host_overhead_ms,
            "host_overhead_ms": host_overhead_ms,
            "execution_overhead_ms": execution_overhead_ms,
            "wall_us_per_step": wall_ms * 1_000.0 / divisor,
            "gpu_total_us_per_step": gpu_total_ms * 1_000.0 / divisor,
            "update_us_per_step": update_ms * 1_000.0 / divisor,
            "current_us_per_step": current_ms * 1_000.0 / divisor,
            "gpu_gap_us_per_step": gpu_gap_ms * 1_000.0 / divisor,
            "host_overhead_us_per_step": host_overhead_ms * 1_000.0 / divisor,
            "execution_overhead_us_per_step": (
                execution_overhead_ms * 1_000.0 / divisor
            ),
            "update_fraction_of_wall": update_ms / wall_ms,
            "current_fraction_of_wall": current_ms / wall_ms,
            "execution_overhead_fraction_of_wall": (execution_overhead_ms / wall_ms),
        }
        repeat_rows.append(row)

        if args.save_timestep_data:
            timestep_rows.extend(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": common["run_id"],
                    "dataset": common["dataset"],
                    "requested_activity": activity,
                    "actual_activity": actual_activity,
                    "n_active_per_step": n_active,
                    "spike_seed": spike_seed,
                    "repeat": repeat,
                    "timestep": timestep,
                    "update_gpu_us": update_time,
                    "current_gpu_us": current_time,
                }
                for timestep, (update_time, current_time) in enumerate(
                    zip(update_us, current_us, strict=True)
                )
            )
    return repeat_rows, timestep_rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate repeat samples into plot-ready firing-rate rows."""

    activities = sorted({float(row["requested_activity"]) for row in rows})
    aggregated: list[dict[str, object]] = []
    for activity in activities:
        group = [row for row in rows if float(row["requested_activity"]) == activity]
        first = group[0]
        output: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": first["run_id"],
            "dataset": first["dataset"],
            "requested_activity": activity,
            "actual_activity": first["actual_activity"],
            "n_active_per_step": first["n_active_per_step"],
            "spike_seed": first["spike_seed"],
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
    parser.add_argument("--t-steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--activities",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACTIVITIES),
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
    if not args.activities or any(not 0.0 <= value <= 1.0 for value in args.activities):
        parser.error("--activities must contain values in [0, 1]")
    if args.dt <= 0.0 or args.tau_mem <= 0.0 or args.tau_syn <= 0.0:
        parser.error("--dt, --tau-mem, and --tau-syn must be positive")
    if args.c_m <= 0.0:
        parser.error("--c-m must be positive")
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
        "timing_method": "per_step_cuda_events_and_synchronized_wall_clock",
        "update_backend": "pytorch_cuda_eager",
        "current_backend": "cusparse_direct_eager",
        "cusparse_primitive": "SpMV",
        "cusparse_algorithm": "CUSPARSE_SPMV_ALG_DEFAULT",
        "cuda_graph": False,
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
        print(
            f"activity={activity:.4%} actual={n_active / matrix.shape[0]:.4%} "
            f"active={n_active}"
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
    breakdown_path = args.output_dir / "naive_rsnn_breakdown.csv"
    aggregate_path = args.output_dir / "naive_rsnn_aggregate.csv"
    timestep_path = args.output_dir / "naive_rsnn_timesteps.csv"
    metadata_path = args.output_dir / "naive_rsnn_metadata.json"
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
            "spike_patterns_replayed_across_repeats": True,
            "spike_construction_in_update_phase": True,
            "random_generation_in_timing": False,
            "synchronize_per_timestep": False,
            "execution_overhead": "max(0, wall - update - current)",
            "gpu_gap": "max(0, gpu_total-update-current)",
            "host_overhead": "max(0, wall-gpu_total)",
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
