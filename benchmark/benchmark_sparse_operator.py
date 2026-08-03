"""Benchmark fixed sparse operator workloads with controlled graph statistics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from benchmark.benchmark_data import (
    GraphSpec,
    SpikeSpec,
    prepare_operator_workload,
    save_workload,
)
from benchmark.benchmark_persistent_snn import BenchCase
from benchmark.benchmark_rsnn_cudagraph_compare import make_torch_csr_weight
from benchmark.provider_common import (
    PreparedMetadata,
    PreparedOperator,
    inspect_tensor,
    tensor_bytes,
    time_cuda_callable,
)
from benchmark.sota_rsnn_cudagraph import (
    prepare_sputnik_operator,
    prepare_vdha_dense_operator,
)


PROVIDERS = ("torch_csr", "sputnik", "vdha_dense")


def reference_trace(
    weight: torch.Tensor,
    spikes_bn: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a fixed operator trace with PyTorch CSR."""

    return torch.stack(
        [
            torch.sparse.mm(weight, spikes.transpose(0, 1)).transpose(0, 1)
            for spikes in spikes_bn
        ]
    )


def prepare_torch_csr_operator(
    weight: torch.Tensor,
    spikes_bn: torch.Tensor,
    case: BenchCase,
) -> PreparedOperator:
    """Prepare distinct native-NB and adapted-BN PyTorch CSR paths."""

    trace_nb = spikes_bn.view(case.t_steps, case.n_neuron, 1)
    native_output = torch.empty_like(trace_nb)
    adapted_output = torch.empty_like(spikes_bn)

    def native_run() -> None:
        for timestep in range(case.t_steps):
            output = torch.sparse.mm(weight, trace_nb[timestep])
            native_output[timestep].copy_(output)

    def adapted_run() -> None:
        for timestep in range(case.t_steps):
            rhs_nb = spikes_bn[timestep].view(case.n_neuron, 1)
            output = torch.sparse.mm(weight, rhs_nb)
            adapted_output[timestep].view(case.n_neuron, 1).copy_(output)

    audit = inspect_tensor(spikes_bn)
    buffers = [native_output, adapted_output]
    metadata = PreparedMetadata(
        logical_shape=tuple(spikes_bn.shape),
        physical_shape=tuple(spikes_bn.shape),
        input_layout="BN",
        operator_layout="NB",
        index_dtype=str(weight.crow_indices().dtype),
        value_dtype=str(weight.values().dtype),
        compute_dtype=str(weight.values().dtype),
        workspace_bytes=tensor_bytes(buffers),
        persistent_bytes=tensor_bytes(
            [
                weight.crow_indices(),
                weight.col_indices(),
                weight.values(),
                spikes_bn,
                *buffers,
            ]
        ),
        padding_ratio=1.0,
        execution_class="device_adapted",
        timing_scope="gpu_execution",
        allocates_during_run=True,
        layout_transform_mode="zero_copy_view",
        layout_transform_in_timing=False,
        input_contiguous=bool(audit["contiguous"]),
        input_address_mod=int(audit["address_alignment"]),
        native_available=True,
        native_status="available",
    )
    return PreparedOperator(
        native_run=native_run,
        adapted_run=adapted_run,
        transform_run=None,
        native_output=native_output,
        adapted_output=adapted_output,
        metadata=metadata,
        release_fn=lambda: None,
    )


def prepare_operator(
    provider: str,
    weight: torch.Tensor,
    spikes_bn: torch.Tensor,
    case: BenchCase,
) -> PreparedOperator:
    """Prepare one provider through its structured operator contract."""

    if provider == "torch_csr":
        return prepare_torch_csr_operator(weight, spikes_bn, case)
    if provider == "sputnik":
        return prepare_sputnik_operator(weight, spikes_bn, case)
    if provider == "vdha_dense":
        return prepare_vdha_dense_operator(weight, spikes_bn, case)
    raise ValueError(f"unknown operator provider: {provider}")


def _base_row(workload, provider: str, prepared: PreparedOperator) -> dict:
    graph = workload.graph_stats.to_dict()
    spike = workload.spike_stats.to_dict()
    metadata = prepared.metadata
    return {
        "workload_id": workload.workload_id,
        "provider": provider,
        "graph_type": graph.pop("graph_type"),
        "graph_seed": graph.pop("seed"),
        **{f"graph_{key}": value for key, value in graph.items()},
        "spike_kind": workload.spike_spec.kind,
        "spike_seed": workload.spike_spec.seed,
        **spike,
        "logical_n": metadata.logical_n,
        "physical_n": metadata.physical_n,
        "logical_batch": metadata.logical_batch,
        "physical_batch": metadata.physical_batch,
        "execution_class": metadata.execution_class,
        "timing_scope": metadata.timing_scope,
        "input_layout": metadata.input_layout,
        "operator_layout": metadata.operator_layout,
        "layout_transform_mode": metadata.layout_transform_mode,
        "layout_transform_in_timing": metadata.layout_transform_in_timing,
        "native_available": metadata.native_available,
        "native_status": metadata.native_status,
        "workspace_bytes": metadata.workspace_bytes,
        "total_prepared_bytes": metadata.total_prepared_bytes,
        "allocates_during_run": metadata.allocates_during_run,
    }


def benchmark_prepared_operator(
    workload,
    provider: str,
    prepared: PreparedOperator,
    reference: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
    timing_modes: tuple[str, ...],
) -> list[dict]:
    """Validate and time genuinely distinct prepared entry points."""

    prepared.adapted_run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared.adapted_output,
        reference,
        atol=5e-3,
        rtol=2e-3,
    )
    if prepared.native_run is not None:
        prepared.native_run()
        torch.cuda.synchronize()
        assert prepared.native_output is not None
        native_bn = prepared.native_output.view_as(reference)
        torch.testing.assert_close(native_bn, reference, atol=5e-3, rtol=2e-3)

    base = _base_row(workload, provider, prepared)
    rows = []
    run_modes = [("adapted", prepared.adapted_run)]
    if prepared.native_run is not None:
        run_modes.insert(0, ("native", prepared.native_run))
    else:
        rows.append(
            {
                **base,
                "mode": "native",
                "timing_mode": "not_run",
                "status": prepared.metadata.native_status,
                "latency_ms": math.nan,
                "latency_samples_ms": "",
            }
        )

    if prepared.transform_run is None:
        rows.append(
            {
                **base,
                "mode": "transform_only",
                "timing_mode": "analytic",
                "status": (
                    "zero_gpu_work"
                    if prepared.metadata.layout_transform_mode
                    == "zero_copy_view"
                    else "not_required"
                ),
                "latency_ms": 0.0,
                "latency_samples_ms": "",
            }
        )
    else:
        run_modes.append(("transform_only", prepared.transform_run))

    for mode, run in run_modes:
        for timing_mode in timing_modes:
            samples = time_cuda_callable(
                run,
                warmup=warmup,
                repeat=repeat,
                timing_mode=timing_mode,
            )
            rows.append(
                {
                    **base,
                    "mode": mode,
                    "timing_mode": timing_mode,
                    "status": "passed",
                    "latency_ms": statistics.median(samples),
                    "latency_samples_ms": ";".join(
                        f"{sample:.9g}" for sample in samples
                    ),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse controlled sparse operator sweep options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neuron", type=int, nargs="+", default=[4096])
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument(
        "--graph-types",
        nargs="+",
        choices=("uniform", "powerlaw", "hotspot", "clustered"),
        default=["uniform"],
    )
    parser.add_argument(
        "--spike-types",
        nargs="+",
        choices=("iid", "fanout_correlated", "temporal", "cluster_burst"),
        default=["iid"],
    )
    parser.add_argument(
        "--activities",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1],
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    parser.add_argument(
        "--timing-modes",
        nargs="+",
        choices=("isolated", "queued"),
        default=["isolated", "queued"],
    )
    parser.add_argument("--seed", type=int, default=20250308)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--save-workload-root", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    if any(n <= 0 for n in args.n_neuron) or args.t_steps <= 0:
        parser.error("N and T must be positive")
    if args.fanout < 0:
        parser.error("fanout must be non-negative")
    if any(not 0.0 <= activity <= 1.0 for activity in args.activities):
        parser.error("activities must be in [0, 1]")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat positive")
    return args


def main() -> None:
    """Run controlled graph and spike workload sweeps."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this comparison.")
    device = torch.device("cuda")
    rows = []
    for n_neuron in args.n_neuron:
        for graph_index, graph_type in enumerate(args.graph_types):
            graph_spec = GraphSpec(
                kind=graph_type,
                n_neuron=n_neuron,
                fanout=args.fanout,
                seed=args.seed + graph_index,
            )
            for spike_index, spike_type in enumerate(args.spike_types):
                for activity in args.activities:
                    spike_spec = SpikeSpec(
                        kind=spike_type,
                        t_steps=args.t_steps,
                        activity=activity,
                        seed=args.seed + 10_000 + spike_index,
                    )
                    workload = prepare_operator_workload(
                        graph_spec,
                        spike_spec,
                        device,
                    )
                    if args.save_workload_root is not None:
                        save_workload(
                            workload,
                            args.save_workload_root
                            / f"{workload.workload_id}.pt",
                        )
                    assert workload.spike_trace is not None
                    case = BenchCase(
                        n_neuron=n_neuron,
                        batch_size=1,
                        t_steps=args.t_steps,
                        fanout=args.fanout,
                        event_rate=0.0,
                    )
                    weight = make_torch_csr_weight(workload.matrix)
                    reference = reference_trace(
                        weight,
                        workload.spike_trace,
                    )
                    for provider in args.providers:
                        prepared = None
                        try:
                            prepared = prepare_operator(
                                provider,
                                weight,
                                workload.spike_trace,
                                case,
                            )
                            rows.extend(
                                benchmark_prepared_operator(
                                    workload,
                                    provider,
                                    prepared,
                                    reference,
                                    warmup=args.warmup,
                                    repeat=args.repeat,
                                    timing_modes=tuple(args.timing_modes),
                                )
                            )
                        except Exception as exc:  # noqa: BLE001
                            rows.append(
                                {
                                    "workload_id": workload.workload_id,
                                    "provider": provider,
                                    "graph_type": graph_type,
                                    "spike_kind": spike_type,
                                    "requested_activity": activity,
                                    "mode": "unavailable",
                                    "timing_mode": "not_run",
                                    "status": (
                                        f"unavailable:{type(exc).__name__}:{exc}"
                                    ),
                                    "latency_ms": math.nan,
                                    "latency_samples_ms": "",
                                }
                            )
                        finally:
                            if prepared is not None:
                                prepared.release()

    for row in rows:
        print(
            f"{row['provider']:<12} graph={row['graph_type']:<10} "
            f"spikes={row.get('spike_kind', 'unknown'):<18} "
            f"activity={float(row['requested_activity']):<6g} "
            f"{row['mode']:<15} {row['timing_mode']:<8} "
            f"{row['status']:<24} latency={row['latency_ms']:.6f} ms"
        )
    if args.csv is not None and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
