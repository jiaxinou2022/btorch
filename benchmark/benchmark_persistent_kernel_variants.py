"""Benchmark persistent SNN CUDA task schedulers on identical workloads."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    csr_to_persistent_graph,
    dense_to_windowed_events,
    make_input_sequence,
    make_recurrent_csr,
)
from btorch.backend.persistent_snn import (  # noqa: E402
    PersistentSNNParams,
    make_empty_state,
    make_persistent_snn_workspace,
    persistent_snn_forward,
)


Variant = Literal["plain", "binning", "spike_block"]
VARIANTS: tuple[Variant, ...] = ("plain", "binning", "spike_block")


@dataclass(frozen=True)
class VariantResult:
    """Store one scheduler's latency distribution."""

    variant: Variant
    median_ms: float
    p20_ms: float
    p80_ms: float


def _variant_kwargs(variant: Variant) -> dict[str, bool]:
    return {
        "fanout_binning": variant == "binning",
        "spike_block": variant == "spike_block",
    }


def benchmark_variant(
    case: BenchCase,
    variant: Variant,
    *,
    warmup: int,
    repeat: int,
) -> tuple[VariantResult, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Measure one variant and return its output for correctness checks."""

    device = torch.device("cuda")
    x_seq = make_input_sequence(case, device)
    graph = csr_to_persistent_graph(make_recurrent_csr(case, device))
    events = dense_to_windowed_events(x_seq)
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

    def run():
        return persistent_snn_forward(
            events,
            graph,
            state,
            params,
            backend="cuda_persistent",
            return_mode="dense",
            workspace=workspace,
            **_variant_kwargs(variant),
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        output = run()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    samples.sort()

    assert output.spikes is not None
    result = VariantResult(
        variant=variant,
        median_ms=statistics.median(samples),
        p20_ms=samples[int(0.2 * (repeat - 1))],
        p80_ms=samples[int(0.8 * (repeat - 1))],
    )
    tensors = (output.spikes, output.state.v, output.state.psc)
    return result, tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neuron", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--fanout", type=int, nargs="+", default=[16, 128, 256])
    parser.add_argument("--event-rate", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive.")

    rows: list[dict[str, int | float | str]] = []
    for fanout in args.fanout:
        case = BenchCase(
            n_neuron=args.n_neuron,
            batch_size=args.batch_size,
            t_steps=args.t_steps,
            fanout=fanout,
            event_rate=args.event_rate,
        )
        results: list[VariantResult] = []
        outputs = {}
        for variant in VARIANTS:
            result, outputs[variant] = benchmark_variant(
                case,
                variant,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            results.append(result)

        reference = outputs["plain"]
        for variant in ("binning", "spike_block"):
            candidate = outputs[variant]
            torch.testing.assert_close(candidate[0], reference[0], atol=0, rtol=0)
            torch.testing.assert_close(candidate[1], reference[1], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(candidate[2], reference[2], atol=1e-4, rtol=1e-4)

        plain_ms = results[0].median_ms
        for result in results:
            row = {
                "device": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "unknown",
                "n_neuron": case.n_neuron,
                "batch_size": case.batch_size,
                "t_steps": case.t_steps,
                "fanout": fanout,
                "event_rate": case.event_rate,
                "variant": result.variant,
                "median_ms": result.median_ms,
                "p20_ms": result.p20_ms,
                "p80_ms": result.p80_ms,
                "speedup_vs_plain": plain_ms / result.median_ms,
            }
            rows.append(row)
            print(
                f"fanout={fanout:4d} {result.variant:11s} "
                f"median={result.median_ms:8.3f} ms "
                f"speedup={plain_ms / result.median_ms:6.3f}x"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
