"""Run one profiler audit case for prepared device benchmark providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark.benchmark_persistent_snn import (
    BenchCase,
    make_input_sequence,
    make_recurrent_csr,
)
from benchmark.benchmark_rsnn_cudagraph_compare import (
    DirectCuSparseProvider,
    PersistentProvider,
    TorchCUDAGraphProvider,
    make_torch_csr_weight,
)
from benchmark.provider_common import AuditResult, BenchmarkRunner


PROVIDERS = (
    "torch_csr_cudagraph",
    "cusparse_direct_cudagraph",
    "persistent_plain",
    "persistent_binning",
    "persistent_spike_block",
)


def audit_runner(provider: str, runner: BenchmarkRunner) -> AuditResult:
    """Profile one run and summarize suspicious allocation or transfer ops."""

    runner.reset()
    runner()
    torch.cuda.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    runner.reset()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as profile:
        runner()
    averaged_events = profile.key_averages()
    keys = sorted({event.key for event in averaged_events})
    lowered = [key.lower() for key in keys]
    synchronize_count = sum(
        event.count
        for event in averaged_events
        if "synchronize" in event.key.lower()
    )
    allocation_names = (
        "aten::empty",
        "aten::empty_like",
        "aten::contiguous",
        "cudamalloc",
        "cudafree",
    )
    unexpected = tuple(
        key
        for key in keys
        if any(name in key.lower() for name in allocation_names)
    )
    return AuditResult(
        provider=provider,
        audited=True,
        has_cuda_malloc=any("cudamalloc" in key for key in lowered),
        has_cuda_free=any("cudafree" in key for key in lowered),
        has_d2h=any(
            "dtoh" in key or "device to host" in key for key in lowered
        ),
        has_h2d=any(
            "htod" in key or "host to device" in key for key in lowered
        ),
        # Profiler shutdown contributes one cudaDeviceSynchronize. Only an
        # additional synchronization is attributable to the provider run.
        has_internal_sync=synchronize_count > 1,
        unexpected_alloc_ops=unexpected,
        observed_ops=tuple(keys),
    )


def parse_args() -> argparse.Namespace:
    """Parse one-time provider audit options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/provider_audit_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the standard audit workload and write a profiler manifest."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for provider audit.")
    device = torch.device("cuda")
    case = BenchCase(
        n_neuron=4096,
        batch_size=1,
        t_steps=32,
        fanout=32,
        event_rate=0.01,
    )
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    weight = make_torch_csr_weight(matrix)
    graph_provider = TorchCUDAGraphProvider()
    direct_provider = DirectCuSparseProvider()
    persistent_provider = PersistentProvider()
    manifest = {}
    for provider in args.providers:
        if provider == "torch_csr_cudagraph":
            runner = graph_provider.fixed_runner(
                x_seq,
                weight,
                case,
                sparse=True,
            )
        elif provider == "cusparse_direct_cudagraph":
            runner = direct_provider.fixed_runner(
                x_seq,
                weight,
                case,
                use_cudagraph=True,
            )
        else:
            runner = persistent_provider.fixed_runner(
                x_seq,
                matrix,
                case,
                variant=provider.removeprefix("persistent_"),
            )
        result = audit_runner(provider, runner)
        manifest[provider] = result.to_dict()
        print(
            f"{provider:<28} malloc={result.has_cuda_malloc} "
            f"d2h={result.has_d2h} h2d={result.has_h2d} "
            f"sync={result.has_internal_sync} "
            f"alloc_ops={list(result.unexpected_alloc_ops)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved audit manifest to {args.output}")


if __name__ == "__main__":
    main()
