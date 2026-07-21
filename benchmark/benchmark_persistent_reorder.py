"""Benchmark physical neuron/CSR reorder strategies for persistent SNN."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTOME_ROOT = REPO_ROOT / "connectome_dataset"
for path in (REPO_ROOT, CONNECTOME_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    csr_to_persistent_graph,
    dense_to_windowed_events,
    make_input_sequence,
    make_recurrent_csr,
)
from btorch.backend.persistent_snn import (  # noqa: E402
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNReorderPlan,
    build_persistent_snn_reorder_plan,
    make_empty_state,
    make_persistent_snn_workspace,
    persistent_snn_forward,
    reorder_persistent_snn_state,
    reorder_windowed_spike_events,
    restore_persistent_snn_output,
)


@dataclass(frozen=True)
class ReorderCase:
    """Describe one physical reorder strategy."""

    name: str
    method: str
    region_size: int = 256
    post_tile_size: int = 128
    sort_row_edges: bool = False


CASES = (
    ReorderCase("R0_identity", "identity"),
    ReorderCase("R0_identity_edgesort", "identity", sort_row_edges=True),
    ReorderCase("R1_global_fanout", "global_fanout"),
    ReorderCase("R2_local_fanout_r128", "local_fanout", region_size=128),
    ReorderCase("R2_local_fanout_r256", "local_fanout", region_size=256),
    ReorderCase("R2_local_fanout_r512", "local_fanout", region_size=512),
    ReorderCase(
        "R2_local_fanout_r128_edgesort",
        "local_fanout",
        region_size=128,
        sort_row_edges=True,
    ),
    ReorderCase(
        "R3_global_primary_t64",
        "global_primary_tile",
        post_tile_size=64,
    ),
    ReorderCase(
        "R3_global_primary_t128",
        "global_primary_tile",
        post_tile_size=128,
    ),
    ReorderCase(
        "R3_global_primary_t256",
        "global_primary_tile",
        post_tile_size=256,
    ),
    ReorderCase(
        "R4_local_primary_r256_t64",
        "local_primary_tile",
        region_size=256,
        post_tile_size=64,
    ),
    ReorderCase(
        "R4_local_primary_r256_t128",
        "local_primary_tile",
        region_size=256,
        post_tile_size=128,
    ),
    ReorderCase(
        "R4_local_primary_r256_t256",
        "local_primary_tile",
        region_size=256,
        post_tile_size=256,
    ),
    ReorderCase(
        "R4_local_primary_r256_t128_edgesort",
        "local_primary_tile",
        region_size=256,
        post_tile_size=128,
        sort_row_edges=True,
    ),
    ReorderCase(
        "R4_local_primary_r256_t256_edgesort",
        "local_primary_tile",
        region_size=256,
        post_tile_size=256,
        sort_row_edges=True,
    ),
    ReorderCase("R0_identity_repeat", "identity"),
)

FINALIST_NAMES = {
    "R0_identity",
    "R0_identity_edgesort",
    "R2_local_fanout_r128",
    "R2_local_fanout_r128_edgesort",
    "R4_local_primary_r256_t128_edgesort",
    "R4_local_primary_r256_t256_edgesort",
}


def load_graph(args: argparse.Namespace, device: torch.device) -> EventCSRGraph:
    """Load the selected graph and normalize recurrent weights."""

    if args.dataset == "uniform":
        case = BenchCase(
            n_neuron=args.n_neuron,
            batch_size=args.batch_size,
            t_steps=args.t_steps,
            fanout=args.fanout,
            event_rate=args.event_rate,
            weight_scale=args.weight_scale,
        )
        return csr_to_persistent_graph(make_recurrent_csr(case, device))

    from connectome_dataset.graph_loader import load_mice_column_v1

    scipy_matrix = load_mice_column_v1(
        root=args.connectome_root,
        use_weights=False,
    ).tocsr()
    average_fanout = scipy_matrix.nnz / scipy_matrix.shape[0]
    scipy_matrix.data.fill(args.weight_scale / average_fanout)
    from btorch.sparse import CSR

    matrix = CSR.from_scipy(scipy_matrix, device=device, dtype=torch.float32)
    return csr_to_persistent_graph(matrix)


def make_case(
    args: argparse.Namespace,
    graph: EventCSRGraph,
) -> BenchCase:
    """Create input and neuron parameters for the selected graph."""

    return BenchCase(
        n_neuron=graph.shape[0],
        batch_size=args.batch_size,
        t_steps=args.t_steps,
        fanout=int(round(graph.indices.numel() / graph.shape[0])),
        event_rate=args.event_rate,
        input_amplitude=args.input_amplitude,
        weight_scale=args.weight_scale,
    )


def build_plan_timed(
    graph: EventCSRGraph,
    reorder_case: ReorderCase,
) -> tuple[PersistentSNNReorderPlan, float]:
    """Build one plan and return synchronized preprocessing latency."""

    torch.cuda.synchronize()
    start = time.perf_counter()
    plan = build_persistent_snn_reorder_plan(
        graph,
        method=reorder_case.method,
        region_size=reorder_case.region_size,
        post_tile_size=reorder_case.post_tile_size,
        sort_row_edges=reorder_case.sort_row_edges,
    )
    torch.cuda.synchronize()
    return plan, (time.perf_counter() - start) * 1000.0


def timed_input_reorder(events, state, plan):
    """Time one-time input and initial-state conversion."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    physical_events = reorder_windowed_spike_events(events, plan)
    physical_state = reorder_persistent_snn_state(state, plan)
    end.record()
    end.synchronize()
    return physical_events, physical_state, start.elapsed_time(end)


def block_degree_variance(graph: EventCSRGraph) -> float:
    """Return mean within-block fanout variance for complete 32-row blocks."""

    degrees = (graph.indptr[1:] - graph.indptr[:-1]).to(torch.float32)
    complete = degrees.numel() // 32 * 32
    if complete == 0:
        return 0.0
    blocks = degrees[:complete].reshape(-1, 32)
    return float(blocks.var(dim=1, unbiased=False).mean().item())


def benchmark_plan(
    args: argparse.Namespace,
    case: BenchCase,
    events,
    state,
    plan: PersistentSNNReorderPlan,
):
    """Benchmark physical kernel time and final old-ID restoration."""

    physical_events, physical_state, input_reorder_ms = timed_input_reorder(
        events,
        state,
        plan,
    )
    workspace = make_persistent_snn_workspace(
        plan.reordered_graph,
        case.batch_size,
    )
    params = PersistentSNNParams(window_size=case.t_steps)

    def run():
        return persistent_snn_forward(
            physical_events,
            plan.reordered_graph,
            physical_state,
            params,
            backend="cuda_persistent",
            return_mode="dense",
            spike_block=True,
            workspace=workspace,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    kernel_samples = []
    restore_samples = []
    total_samples = []
    for _ in range(args.repeat):
        start = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        physical_output = run()
        kernel_end.record()
        restored_output = restore_persistent_snn_output(physical_output, plan)
        end.record()
        end.synchronize()
        kernel_samples.append(start.elapsed_time(kernel_end))
        restore_samples.append(kernel_end.elapsed_time(end))
        total_samples.append(start.elapsed_time(end))

    return {
        "kernel_ms": statistics.median(kernel_samples),
        "restore_ms": statistics.median(restore_samples),
        "total_ms": statistics.median(total_samples),
        "input_reorder_ms": input_reorder_ms,
    }, restored_output


def benchmark_finalists_interleaved(
    args: argparse.Namespace,
    case: BenchCase,
    graph: EventCSRGraph,
    events,
    state,
) -> list[dict[str, int | float | str | bool]]:
    """Benchmark finalists round-robin to control clock and order effects."""

    params = PersistentSNNParams(window_size=case.t_steps)
    prepared = []
    for reorder_case in CASES:
        if reorder_case.name not in FINALIST_NAMES:
            continue
        plan, preprocess_ms = build_plan_timed(graph, reorder_case)
        physical_events, physical_state, input_reorder_ms = timed_input_reorder(
            events,
            state,
            plan,
        )
        prepared.append(
            {
                "case": reorder_case,
                "plan": plan,
                "events": physical_events,
                "state": physical_state,
                "workspace": make_persistent_snn_workspace(
                    plan.reordered_graph,
                    case.batch_size,
                ),
                "preprocess_ms": preprocess_ms,
                "input_reorder_ms": input_reorder_ms,
                "kernel_samples": [],
                "restore_samples": [],
                "total_samples": [],
                "output": None,
            }
        )

    def run(item):
        plan = item["plan"]
        return persistent_snn_forward(
            item["events"],
            plan.reordered_graph,
            item["state"],
            params,
            backend="cuda_persistent",
            return_mode="dense",
            spike_block=True,
            workspace=item["workspace"],
        )

    for _ in range(args.warmup):
        for item in prepared:
            run(item)
    torch.cuda.synchronize()

    for repeat in range(args.repeat):
        for offset in range(len(prepared)):
            item = prepared[(repeat + offset) % len(prepared)]
            start = torch.cuda.Event(enable_timing=True)
            kernel_end = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            physical_output = run(item)
            kernel_end.record()
            item["output"] = restore_persistent_snn_output(
                physical_output,
                item["plan"],
            )
            end.record()
            end.synchronize()
            item["kernel_samples"].append(start.elapsed_time(kernel_end))
            item["restore_samples"].append(kernel_end.elapsed_time(end))
            item["total_samples"].append(start.elapsed_time(end))

    baseline = prepared[0]
    baseline_kernel = statistics.median(baseline["kernel_samples"])
    baseline_total = statistics.median(baseline["total_samples"])
    reference = baseline["output"]
    assert reference is not None
    assert reference.spikes is not None
    rows = []
    for item in prepared:
        reorder_case = item["case"]
        plan = item["plan"]
        output = item["output"]
        assert output is not None
        assert output.spikes is not None
        kernel_ms = statistics.median(item["kernel_samples"])
        restore_ms = statistics.median(item["restore_samples"])
        total_ms = statistics.median(item["total_samples"])
        row = {
            "dataset": args.dataset,
            "name": reorder_case.name,
            "method": reorder_case.method,
            "region_size": reorder_case.region_size,
            "post_tile_size": reorder_case.post_tile_size,
            "sort_row_edges": reorder_case.sort_row_edges,
            "n_neuron": case.n_neuron,
            "edge_count": graph.indices.numel(),
            "batch_size": case.batch_size,
            "t_steps": case.t_steps,
            "event_rate": case.event_rate,
            "preprocess_ms": item["preprocess_ms"],
            "input_reorder_ms": item["input_reorder_ms"],
            "kernel_ms": kernel_ms,
            "restore_ms": restore_ms,
            "total_ms": total_ms,
            "kernel_speedup": baseline_kernel / kernel_ms,
            "total_speedup": baseline_total / total_ms,
            "block_degree_variance": block_degree_variance(
                plan.reordered_graph
            ),
            "spike_match": float(
                (output.spikes == reference.spikes).float().mean().item()
            ),
            "max_v_error": float(
                (output.state.v - reference.state.v).abs().max().item()
            ),
            "max_psc_error": float(
                (output.state.psc - reference.state.psc).abs().max().item()
            ),
        }
        rows.append(row)
        print(
            f"{reorder_case.name:40s} "
            f"kernel={kernel_ms:7.3f} ms total={total_ms:7.3f} ms "
            f"speedup={row['total_speedup']:6.3f}x"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("uniform", "mice_column_v1"),
        default="mice_column_v1",
    )
    parser.add_argument("--connectome-root", type=Path)
    parser.add_argument("--n-neuron", type=int, default=4096)
    parser.add_argument("--fanout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--event-rate", type=float, default=0.02)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--interleaved-finalists", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/persistent_kernel/physical_reorder.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    device = torch.device("cuda")
    graph = load_graph(args, device)
    case = make_case(args, graph)
    x_seq = make_input_sequence(case, device)
    events = dense_to_windowed_events(x_seq)
    state = make_empty_state(
        case.batch_size,
        case.n_neuron,
        device=device,
        refractory=False,
    )

    if args.interleaved_finalists:
        rows = benchmark_finalists_interleaved(
            args,
            case,
            graph,
            events,
            state,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.output}")
        return

    rows = []
    reference = None
    baseline_kernel_ms = None
    baseline_total_ms = None
    for reorder_case in CASES:
        plan, preprocess_ms = build_plan_timed(graph, reorder_case)
        timings, output = benchmark_plan(args, case, events, state, plan)
        if reference is None:
            reference = output
            baseline_kernel_ms = timings["kernel_ms"]
            baseline_total_ms = timings["total_ms"]
        assert reference.spikes is not None
        assert output.spikes is not None
        spike_match = float((output.spikes == reference.spikes).float().mean().item())
        max_v_error = float((output.state.v - reference.state.v).abs().max().item())
        max_psc_error = float(
            (output.state.psc - reference.state.psc).abs().max().item()
        )
        assert baseline_kernel_ms is not None
        assert baseline_total_ms is not None
        row = {
            "dataset": args.dataset,
            "name": reorder_case.name,
            "method": reorder_case.method,
            "region_size": reorder_case.region_size,
            "post_tile_size": reorder_case.post_tile_size,
            "sort_row_edges": reorder_case.sort_row_edges,
            "n_neuron": case.n_neuron,
            "edge_count": graph.indices.numel(),
            "batch_size": case.batch_size,
            "t_steps": case.t_steps,
            "event_rate": case.event_rate,
            "preprocess_ms": preprocess_ms,
            **timings,
            "kernel_speedup": baseline_kernel_ms / timings["kernel_ms"],
            "total_speedup": baseline_total_ms / timings["total_ms"],
            "block_degree_variance": block_degree_variance(
                plan.reordered_graph
            ),
            "spike_match": spike_match,
            "max_v_error": max_v_error,
            "max_psc_error": max_psc_error,
        }
        rows.append(row)
        print(
            f"{reorder_case.name:40s} "
            f"kernel={timings['kernel_ms']:7.3f} ms "
            f"total={timings['total_ms']:7.3f} ms "
            f"speedup={row['total_speedup']:6.3f}x "
            f"match={spike_match:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
