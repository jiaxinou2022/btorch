"""Compare the naive fanout-binned kernel against the pipeline-binned kernel.

Both persistent CUDA kernels are compiled into the same extension when the
process is started with ``BTORCH_PERSISTENT_PIPELINE=1``, and
:func:`btorch.backend.persistent_snn.persistent_snn_forward` dispatches on the
``fanout_binning`` flag:

* ``fanout_binning=True`` routes to the naive fanout-binned kernel
  (``persistent_snn_binned_kernel.cu``).
* ``fanout_binning=False`` routes the same op to the UPDATE--propagation
  pipeline kernel (``persistent_snn_pipeline_kernel.cu``) with its two-ended
  HIGH/LOW queue.

This script runs the same RSNN workload through both variants and reports the
median of ``--repeat`` queued latency samples per variant. Every sample is one
independent inference: membrane state and scratch tensors are reset between
samples, matching the queued timing used by the main comparison benchmark.

Usage::

    BTORCH_PERSISTENT_PIPELINE=1 python \\
        benchmark/scripts/benchmark_naive_vs_pipeline_binning.py --repeat 30
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    csr_to_persistent_graph,
    dense_to_windowed_events,
    make_input_sequence,
    make_recurrent_csr,
)
from benchmark.benchmark_rsnn_cudagraph_compare import (  # noqa: E402
    load_flybrain_csr,
    resolve_dataset_defaults,
)
from benchmark.provider_common import time_cuda_callable  # noqa: E402
from btorch.backend.persistent_snn import (  # noqa: E402
    PersistentSNNParams,
    make_empty_state,
    make_persistent_snn_workspace,
    persistent_snn_forward,
)


PIPELINE_ENV_DEFAULTS = {
    "BTORCH_PIPELINE_ROLE_RATIO": "7:1",
    "BTORCH_PIPELINE_DEDICATED_WARPS": "1",
    "BTORCH_PIPELINE_HELPER_WARPS": "1",
    "BTORCH_PIPELINE_TICKET_CHUNK": "1",
    "BTORCH_PIPELINE_STATIC_WAVES": "0",
    "BTORCH_PIPELINE_BINNED_THRESHOLD": "512",
    "BTORCH_PIPELINE_LOW_SUBWARP_SIZE": "8",
    "BTORCH_PIPELINE_HIGH_LOW_RATIO": "2",
}


def effective_pipeline_config() -> dict[str, str]:
    """Return the pipeline env configuration effective for this process."""
    return {
        name: os.environ.get(name, default)
        for name, default in PIPELINE_ENV_DEFAULTS.items()
    }


def build_workload(
    args: argparse.Namespace, device: torch.device
) -> tuple[BenchCase, object, torch.Tensor]:
    """Build the benchmark case, graph matrix, and external input sequence."""

    if args.dataset == "flybrain":
        _, weight_scale = resolve_dataset_defaults(args.dataset, args.weight_scale)
        matrix = load_flybrain_csr(
            args.connectome_root,
            weight_scale=weight_scale,
            device=device,
        )
        n_neuron = matrix.shape[0]
        fanout = int(round(matrix.indices.numel() / max(n_neuron, 1)))
    else:
        graph_case = BenchCase(
            n_neuron=args.n_neuron,
            batch_size=1,
            t_steps=args.t_steps,
            fanout=args.fanout,
            event_rate=args.event_rate,
            weight_scale=args.weight_scale or 0.15,
        )
        matrix = make_recurrent_csr(graph_case, device)
        n_neuron = args.n_neuron
        fanout = args.fanout
    case = BenchCase(
        n_neuron=n_neuron,
        batch_size=1,
        t_steps=args.t_steps,
        fanout=fanout,
        event_rate=args.event_rate,
        dt=args.dt,
        tau_mem=args.tau_mem,
        tau_syn=args.tau_syn,
        v_threshold=args.v_threshold,
        c_m=args.c_m,
        input_amplitude=args.input_amplitude,
        weight_scale=args.weight_scale or 0.15,
    )
    x_seq = make_input_sequence(case, device)
    return case, matrix, x_seq


def make_runners(
    events: object,
    graph: object,
    case: BenchCase,
    device: torch.device,
) -> tuple[object, object, object]:
    """Prepare shared state/workspace and per-variant forward closures.

    Returns a ``(reset, run_naive, run_pipeline)`` triple. ``reset`` restores
    membrane state and zeroes every workspace tensor so each latency sample is
    one independent inference.
    """

    state = make_empty_state(
        case.batch_size,
        case.n_neuron,
        device=device,
        refractory=False,
    )
    workspace = make_persistent_snn_workspace(graph, case.batch_size)
    params = PersistentSNNParams(
        dt=case.dt,
        tau_mem=case.tau_mem,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        c_m=case.c_m,
        hard_reset=case.hard_reset,
        window_size=case.t_steps,
    )
    initial_v = state.v.clone()
    initial_psc = state.psc.clone()
    workspace_tensors = (
        workspace.input_current,
        workspace.queue_batch,
        workspace.queue_edge_start,
        workspace.queue_edge_end,
        workspace.spike_count,
        workspace.work_counter,
    )
    current_state = state

    def reset() -> None:
        nonlocal current_state
        current_state.v.copy_(initial_v)
        current_state.psc.copy_(initial_psc)
        for tensor in workspace_tensors:
            tensor.zero_()

    def make_run(fanout_binning: bool):
        def run() -> torch.Tensor:
            nonlocal current_state
            output = persistent_snn_forward(
                events,
                graph,
                current_state,
                params,
                backend="cuda_persistent",
                return_mode="dense",
                workspace=workspace,
                fanout_binning=fanout_binning,
            )
            current_state = output.state
            assert output.spikes is not None
            return output.spikes

        return run

    return reset, make_run(True), make_run(False)


def summarize(samples: list[float]) -> dict[str, float]:
    """Return mean/median/min/max of raw latency samples in milliseconds."""

    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def write_csv(
    path: Path, naive_samples: list[float], pipeline_samples: list[float]
) -> None:
    """Write one raw latency sample per row for both variants."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["variant", "sample", "latency_ms"],
        )
        writer.writeheader()
        for variant, samples in (
            ("naive_binned", naive_samples),
            ("pipeline_binned", pipeline_samples),
        ):
            for index, value in enumerate(samples):
                writer.writerow(
                    {"variant": variant, "sample": index, "latency_ms": value}
                )
    print(f"Saved raw samples to {path}")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("flybrain", "uniform"),
        default="flybrain",
        help="flybrain uses the real FlyWire v783 graph; uniform is synthetic.",
    )
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument("--n-neuron", type=int, default=2**13)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--event-rate", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--c-m", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the raw latency samples per variant to this CSV.",
    )
    args = parser.parse_args()
    if args.n_neuron <= 0 or args.fanout < 0:
        parser.error("--n-neuron must be positive and --fanout non-negative")
    if args.t_steps <= 0 or args.repeat <= 0 or args.warmup < 0:
        parser.error("--t-steps/--repeat must be positive and --warmup non-negative")
    if not 0.0 <= args.event_rate <= 1.0:
        parser.error("--event-rate must be in [0, 1]")
    return args


def main() -> None:
    """Run the naive-vs-pipeline binning comparison."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this comparison.")
    if os.environ.get("BTORCH_PERSISTENT_PIPELINE", "0") != "1":
        raise SystemExit(
            "BTORCH_PERSISTENT_PIPELINE=1 is required so the extension builds "
            "with the pipeline kernel; re-run with it set."
        )
    device = torch.device("cuda")
    case, matrix, x_seq = build_workload(args, device)
    events = dense_to_windowed_events(x_seq)
    graph = csr_to_persistent_graph(matrix)
    reset, run_naive, run_pipeline = make_runners(events, graph, case, device)

    reset()
    naive_spikes = run_naive()
    reset()
    pipeline_spikes = run_pipeline()
    torch.cuda.synchronize()
    identical = bool(torch.equal(naive_spikes, pipeline_spikes))
    print(
        f"workload N={case.n_neuron} T={case.t_steps} "
        f"B={case.batch_size} edges={matrix.indices.numel()}"
    )
    print(f"pipeline config: {effective_pipeline_config()}")
    print(f"spike output identical (naive vs pipeline): {identical}")

    naive_samples = time_cuda_callable(
        run_naive,
        reset_fn=reset,
        reset_policy="before_sample",
        warmup=args.warmup,
        repeat=args.repeat,
        timing_mode="queued",
    )
    pipeline_samples = time_cuda_callable(
        run_pipeline,
        reset_fn=reset,
        reset_policy="before_sample",
        warmup=args.warmup,
        repeat=args.repeat,
        timing_mode="queued",
    )
    naive_stats = summarize(naive_samples)
    pipeline_stats = summarize(pipeline_samples)
    speedup = naive_stats["median_ms"] / pipeline_stats["median_ms"]
    print(
        "naive_binned    median="
        f"{naive_stats['median_ms']:.4f} ms mean={naive_stats['mean_ms']:.4f} "
        f"min={naive_stats['min_ms']:.4f} max={naive_stats['max_ms']:.4f}"
    )
    print(
        "pipeline_binned median="
        f"{pipeline_stats['median_ms']:.4f} ms "
        f"mean={pipeline_stats['mean_ms']:.4f} "
        f"min={pipeline_stats['min_ms']:.4f} max={pipeline_stats['max_ms']:.4f}"
    )
    print(f"pipeline speedup vs naive_binned = {speedup:.3f}x")
    if args.csv is not None:
        write_csv(args.csv, naive_samples, pipeline_samples)


if __name__ == "__main__":
    main()
