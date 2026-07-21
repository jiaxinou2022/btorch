"""Compare standard PyTorch RSNN baselines with persistent CUDA kernels.

The default comparison contains four public-API PyTorch baselines and the
three persistent CUDA task schedulers:

* ``torch_dense_eager`` uses :func:`torch.nn.functional.linear`.
* ``torch_dense_cudagraph`` captures the same dense forward with
  :class:`torch.cuda.CUDAGraph`.
* ``torch_csr_eager`` uses :func:`torch.sparse.mm` with a PyTorch CSR tensor.
* ``torch_csr_cudagraph`` captures the same CSR forward with a CUDA graph.
* ``cusparse_direct_eager`` calls cuSPARSE SpMV/SpMM directly from a CUDA
  extension with preallocated descriptors and workspace.
* ``cusparse_direct_cudagraph`` captures that direct CUDA execution.
* ``persistent_plain``, ``persistent_binning``, and
  ``persistent_spike_block`` execute one cooperative CUDA kernel per window.

All providers evaluate the same recurrent LIF and ExponentialPSC equations
from the same zero state and fixed input. Dataset loading, weight conversion,
CUDA graph capture, and persistent workspace allocation are outside timing.
The direct provider uses ``CUSPARSE_SPMV_ALG_DEFAULT`` for batch size one and
``CUSPARSE_SPMM_CSR_ALG1`` otherwise.

Usage::

    python benchmark/benchmark_rsnn_cudagraph_compare.py \
        --dataset mice_column_v1 --t-steps 128 --batch-size 1 \
        --csv benchmark/mice_v1_standard_baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTOME_ROOT = REPO_ROOT / "connectome_dataset"
for import_root in (REPO_ROOT, CONNECTOME_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    RSNNResult,
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
from btorch.sparse import CSR, sparse_mm  # noqa: E402


Provider = Literal[
    "torch_dense_eager",
    "torch_dense_cudagraph",
    "torch_csr_eager",
    "torch_csr_cudagraph",
    "cusparse_direct_eager",
    "cusparse_direct_cudagraph",
    "persistent_plain",
    "persistent_binning",
    "persistent_spike_block",
]

PROVIDERS: tuple[Provider, ...] = (
    "torch_dense_eager",
    "torch_dense_cudagraph",
    "torch_csr_eager",
    "torch_csr_cudagraph",
    "cusparse_direct_eager",
    "cusparse_direct_cudagraph",
    "persistent_plain",
    "persistent_binning",
    "persistent_spike_block",
)


def load_mice_column_v1_csr(
    root: Path | None, *, weight_scale: float, device: torch.device
) -> CSR:
    """Load the mice V1 column graph and normalize its recurrent weights."""

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
            "or pass --connectome-root."
        ) from exc
    if scipy_matrix.shape[0] != scipy_matrix.shape[1]:
        raise ValueError(f"mice_column_v1 must be square, got {scipy_matrix.shape}.")
    average_fanout = scipy_matrix.nnz / max(scipy_matrix.shape[0], 1)
    scipy_matrix.data.fill(weight_scale / max(average_fanout, 1.0))
    return CSR.from_scipy(scipy_matrix, device=device, dtype=torch.float32)


def precompute_csr_row(matrix: CSR) -> torch.Tensor:
    """Expand CSR pointers into source rows for legacy benchmark imports."""

    counts = matrix.indptr[1:] - matrix.indptr[:-1]
    return torch.repeat_interleave(
        torch.arange(matrix.shape[0], device=matrix.indptr.device), counts
    )


def csr_mm_with_cached_row(
    matrix: CSR, x: torch.Tensor, row: torch.Tensor
) -> torch.Tensor:
    """Evaluate the legacy source-oriented CSR gather/scatter operation."""

    leading = x.shape[:-1]
    x2d = x.reshape(-1, matrix.shape[0])
    contributions = x2d[:, row] * matrix.effective_values()
    result = torch.zeros(x2d.shape[0], matrix.shape[1], device=x.device, dtype=x.dtype)
    result.scatter_add_(1, matrix.indices.expand(x2d.shape[0], -1), contributions)
    return result.reshape(*leading, matrix.shape[1])


def native_sparse_rsnn_forward(
    x_seq: torch.Tensor,
    matrix: CSR,
    v0: torch.Tensor,
    psc0: torch.Tensor,
    *,
    dt: float,
    tau_mem: float,
    tau_syn: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    t_steps: int,
    row: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the legacy btorch CSR path used by the roofline benchmark."""

    decay = math.exp(-dt / tau_syn)
    reset_delta = v_threshold - v_reset
    v = v0
    psc = psc0
    spikes = []
    for t in range(t_steps):
        current = psc + x_seq[t]
        v_pre = v + dt * (-(v - v_reset) / tau_mem + current / c_m)
        z = (v_pre >= v_threshold).to(v.dtype)
        v = v_pre - reset_delta * z
        recurrent = (
            csr_mm_with_cached_row(matrix, z, row)
            if row is not None
            else sparse_mm(matrix, z)
        )
        psc = psc * decay + recurrent
        spikes.append(z)
    return torch.stack(spikes, dim=0), v, psc


def make_torch_csr_weight(matrix: CSR) -> torch.Tensor:
    """Convert source-oriented btorch CSR into ``(N_post, N_pre)`` CSR.

    PyTorch sparse matrix multiplication computes ``W @ z.T``. The btorch
    graph stores rows by pre-synaptic neuron, so its edges are transposed once
    during untimed preparation. COO coalescing also gives defined behavior if
    a dataset contains duplicate edges.
    """

    source = precompute_csr_row(matrix)
    edge_index = torch.stack((matrix.indices.to(torch.long), source.to(torch.long)))
    coo = torch.sparse_coo_tensor(
        edge_index,
        matrix.effective_values(),
        size=(matrix.shape[1], matrix.shape[0]),
        device=matrix.data.device,
        dtype=matrix.data.dtype,
    ).coalesce()
    return coo.to_sparse_csr()


def torch_dense_rsnn_forward(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    v0: torch.Tensor,
    psc0: torch.Tensor,
    case: BenchCase,
) -> RSNNResult:
    """Run the RSNN with the standard dense PyTorch linear operator."""

    decay = math.exp(-case.dt / case.tau_syn)
    reset_delta = case.v_threshold - case.v_reset
    v = v0
    psc = psc0
    spikes = []
    for t in range(case.t_steps):
        current = psc + x_seq[t]
        v_pre = v + case.dt * (
            -(v - case.v_reset) / case.tau_mem + current / case.c_m
        )
        z = (v_pre >= case.v_threshold).to(v.dtype)
        v = v_pre - reset_delta * z
        recurrent = functional.linear(z, weight)
        psc = psc * decay + recurrent
        spikes.append(z)
    return RSNNResult(spikes=torch.stack(spikes), v=v, psc=psc)


def torch_csr_rsnn_forward(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    v0: torch.Tensor,
    psc0: torch.Tensor,
    case: BenchCase,
) -> RSNNResult:
    """Run the RSNN with the standard PyTorch CSR sparse matrix operator."""

    decay = math.exp(-case.dt / case.tau_syn)
    reset_delta = case.v_threshold - case.v_reset
    v = v0
    psc = psc0
    spikes = []
    for t in range(case.t_steps):
        current = psc + x_seq[t]
        v_pre = v + case.dt * (
            -(v - case.v_reset) / case.tau_mem + current / case.c_m
        )
        z = (v_pre >= case.v_threshold).to(v.dtype)
        v = v_pre - reset_delta * z
        recurrent = torch.sparse.mm(weight, z.transpose(0, 1)).transpose(0, 1)
        psc = psc * decay + recurrent
        spikes.append(z)
    return RSNNResult(spikes=torch.stack(spikes), v=v, psc=psc)


def make_eager_runner(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    case: BenchCase,
    *,
    sparse: bool,
):
    """Prepare zero state and return a standard eager forward runner."""

    v0 = torch.zeros(
        case.batch_size,
        case.n_neuron,
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    psc0 = torch.zeros_like(v0)
    forward = torch_csr_rsnn_forward if sparse else torch_dense_rsnn_forward

    def run() -> RSNNResult:
        return forward(x_seq, weight, v0, psc0, case)

    return run


class TorchCUDAGraphProvider:
    """Capture standard dense or CSR PyTorch forward and replay it."""

    def __init__(self) -> None:
        self._runners: dict[tuple, object] = {}

    def fixed_runner(
        self,
        x_seq: torch.Tensor,
        weight: torch.Tensor,
        case: BenchCase,
        *,
        sparse: bool,
    ):
        key = (id(weight), case.t_steps, case.batch_size, x_seq.shape, sparse)
        cached = self._runners.get(key)
        if cached is not None:
            return cached

        static_x = x_seq.clone()
        static_v0 = torch.zeros(
            case.batch_size,
            case.n_neuron,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
        static_psc0 = torch.zeros_like(static_v0)
        forward = torch_csr_rsnn_forward if sparse else torch_dense_rsnn_forward

        def execute() -> RSNNResult:
            return forward(static_x, weight, static_v0, static_psc0, case)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                execute()
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = execute()

        def run() -> RSNNResult:
            graph.replay()
            return output

        # The closure keeps the graph, static tensors, weight, and outputs
        # alive at their captured addresses.
        self._runners[key] = run
        return run


class PersistentProvider:
    """Cache immutable event input and scratch space for persistent timing."""

    def __init__(self) -> None:
        self._runners: dict[tuple, object] = {}

    def fixed_runner(
        self, x_seq: torch.Tensor, matrix: CSR, case: BenchCase, *, variant: str
    ):
        key = (id(matrix), case.t_steps, case.batch_size, x_seq.shape, variant)
        cached = self._runners.get(key)
        if cached is not None:
            return cached

        events = dense_to_windowed_events(x_seq)
        graph = csr_to_persistent_graph(matrix)
        state = make_empty_state(
            case.batch_size,
            case.n_neuron,
            device=x_seq.device,
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
        options = {
            "fanout_binning": variant == "binning",
            "spike_block": variant == "spike_block",
        }

        def run() -> RSNNResult:
            output = persistent_snn_forward(
                events,
                graph,
                state,
                params,
                backend="cuda_persistent",
                return_mode="dense",
                workspace=workspace,
                **options,
            )
            assert output.spikes is not None
            return RSNNResult(
                spikes=output.spikes,
                v=output.state.v,
                psc=output.state.psc,
            )

        self._runners[key] = run
        return run


class DirectCuSparseProvider:
    """Run preallocated direct cuSPARSE SpMV/SpMM, eagerly or as a graph."""

    def __init__(self) -> None:
        self._runners: dict[tuple, object] = {}

    def fixed_runner(
        self,
        x_seq: torch.Tensor,
        weight: torch.Tensor,
        case: BenchCase,
        *,
        use_cudagraph: bool,
    ):
        key = (
            id(weight),
            case.t_steps,
            case.batch_size,
            x_seq.shape,
            use_cudagraph,
        )
        cached = self._runners.get(key)
        if cached is not None:
            return cached

        from benchmark.cusparse_rsnn import load

        extension = load()
        # Direct cuSPARSE uses 32-bit indices, matching the persistent kernels
        # and avoiding the slower 64-bit index path. Conversion is untimed.
        crow = weight.crow_indices().to(torch.int32).contiguous()
        col = weight.col_indices().to(torch.int32).contiguous()
        values = weight.values().contiguous()
        static_x = x_seq.clone()
        v = torch.empty(
            case.batch_size,
            case.n_neuron,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
        psc = torch.empty_like(v)
        spikes = torch.empty_like(x_seq)
        recurrent = torch.empty_like(v)
        plan = extension.prepare(
            crow,
            col,
            values,
            spikes,
            recurrent,
            case.batch_size,
        )

        def execute() -> RSNNResult:
            extension.run(
                plan,
                static_x,
                v,
                psc,
                case.dt,
                case.tau_mem,
                case.tau_syn,
                case.v_threshold,
                case.v_reset,
                case.c_m,
            )
            return RSNNResult(spikes=spikes, v=v, psc=psc)

        if not use_cudagraph:
            self._runners[key] = execute
            return execute

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                execute()
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = execute()

        def replay() -> RSNNResult:
            graph.replay()
            return output

        self._runners[key] = replay
        return replay


def time_ms(fn, *, warmup: int, repeat: int) -> float:
    """Measure median GPU stream time with CUDA events."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = [
        start.elapsed_time(end)
        for start, end in zip(starts, ends, strict=True)
    ]
    return float(torch.tensor(samples, dtype=torch.float64).median().item())


def correctness_metrics(result: RSNNResult, reference: RSNNResult) -> dict:
    """Return correctness metrics tolerant of floating reduction order."""

    spike_mismatches = int((result.spikes != reference.spikes).sum().item())
    spike_mismatch_rate = spike_mismatches / reference.spikes.numel()
    v_diff = float((result.v - reference.v).abs().max().item())
    psc_diff = float((result.psc - reference.psc).abs().max().item())
    passed = spike_mismatch_rate <= 1e-3 and v_diff <= 2e-1 and psc_diff <= 5e-3
    return {
        "status": "passed" if passed else "correctness_failed",
        "spike_mismatches": spike_mismatches,
        "spike_mismatch_rate": spike_mismatch_rate,
        "v_max_abs_diff": v_diff,
        "psc_max_abs_diff": psc_diff,
    }


def _empty_metrics(status: str = "not_checked") -> dict:
    return {
        "status": status,
        "spike_mismatches": -1,
        "spike_mismatch_rate": float("nan"),
        "v_max_abs_diff": float("nan"),
        "psc_max_abs_diff": float("nan"),
    }


def bench_case(
    case: BenchCase,
    *,
    device: torch.device,
    dataset: str,
    matrix: CSR,
    providers: tuple[Provider, ...],
    graph_provider: TorchCUDAGraphProvider,
    direct_cusparse_provider: DirectCuSparseProvider,
    persistent_provider: PersistentProvider,
    warmup: int,
    repeat: int,
    check_correctness: bool,
) -> list[dict]:
    """Prepare and benchmark all selected providers for one case."""

    x_seq = make_input_sequence(case, device)
    csr_weight = make_torch_csr_weight(matrix)
    needs_dense = check_correctness or any(
        provider.startswith("torch_dense_") for provider in providers
    )
    dense_weight = csr_weight.to_dense() if needs_dense else None

    reference = None
    if check_correctness:
        assert dense_weight is not None
        reference = make_eager_runner(
            x_seq, dense_weight, case, sparse=False
        )()

    rows = []
    for provider in providers:
        try:
            if provider == "torch_dense_eager":
                assert dense_weight is not None
                run = make_eager_runner(x_seq, dense_weight, case, sparse=False)
            elif provider == "torch_dense_cudagraph":
                assert dense_weight is not None
                run = graph_provider.fixed_runner(
                    x_seq, dense_weight, case, sparse=False
                )
            elif provider == "torch_csr_eager":
                run = make_eager_runner(x_seq, csr_weight, case, sparse=True)
            elif provider == "torch_csr_cudagraph":
                run = graph_provider.fixed_runner(
                    x_seq, csr_weight, case, sparse=True
                )
            elif provider == "cusparse_direct_eager":
                run = direct_cusparse_provider.fixed_runner(
                    x_seq, csr_weight, case, use_cudagraph=False
                )
            elif provider == "cusparse_direct_cudagraph":
                run = direct_cusparse_provider.fixed_runner(
                    x_seq, csr_weight, case, use_cudagraph=True
                )
            elif provider.startswith("persistent_"):
                run = persistent_provider.fixed_runner(
                    x_seq,
                    matrix,
                    case,
                    variant=provider.removeprefix("persistent_"),
                )
            else:
                raise ValueError(provider)

            result = run()
            metrics = (
                correctness_metrics(result, reference)
                if reference is not None
                else _empty_metrics()
            )
            latency = time_ms(run, warmup=warmup, repeat=repeat)
        except Exception as exc:  # noqa: BLE001
            metrics = _empty_metrics(f"error:{type(exc).__name__}: {exc}")
            latency = float("nan")
            print(f"[error] provider={provider} N={case.n_neuron}: {exc}")

        rows.append(
            {
                "provider": provider,
                "dataset": dataset,
                "n_neuron": case.n_neuron,
                "t_steps": case.t_steps,
                "batch_size": case.batch_size,
                "graph_synapses": int(matrix.indices.numel()),
                "average_fanout": matrix.indices.numel() / case.n_neuron,
                "direct_cusparse_primitive": (
                    "SpMV" if case.batch_size == 1 else "SpMM"
                ),
                "direct_cusparse_algorithm": (
                    "CUSPARSE_SPMV_ALG_DEFAULT"
                    if case.batch_size == 1
                    else "CUSPARSE_SPMM_CSR_ALG1"
                ),
                **metrics,
                "latency_ms": latency,
                "speedup_vs_dense_eager": float("nan"),
                "speedup_vs_dense_cudagraph": float("nan"),
                "speedup_vs_csr_eager": float("nan"),
                "speedup_vs_csr_cudagraph": float("nan"),
                "speedup_vs_cusparse_direct_eager": float("nan"),
                "speedup_vs_cusparse_direct_cudagraph": float("nan"),
            }
        )

    latency_by_provider = {
        row["provider"]: float(row["latency_ms"])
        for row in rows
        if math.isfinite(float(row["latency_ms"]))
    }
    baselines = (
        ("torch_dense_eager", "speedup_vs_dense_eager"),
        ("torch_dense_cudagraph", "speedup_vs_dense_cudagraph"),
        ("torch_csr_eager", "speedup_vs_csr_eager"),
        ("torch_csr_cudagraph", "speedup_vs_csr_cudagraph"),
        ("cusparse_direct_eager", "speedup_vs_cusparse_direct_eager"),
        (
            "cusparse_direct_cudagraph",
            "speedup_vs_cusparse_direct_cudagraph",
        ),
    )
    for row in rows:
        latency = float(row["latency_ms"])
        if not math.isfinite(latency) or latency <= 0:
            continue
        for baseline, column in baselines:
            if baseline in latency_by_provider:
                row[column] = latency_by_provider[baseline] / latency
    return rows


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("uniform", "mice_column_v1", "mice_v1_column"),
        default="mice_column_v1",
    )
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument("--n-neuron", type=int, default=2**13)
    parser.add_argument("--t-steps", type=int, nargs="+", default=[128])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--event-rate", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--c-m", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS)
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.batch_size <= 0 or any(t_steps <= 0 for t_steps in args.t_steps):
        parser.error("--batch-size and every --t-steps value must be positive")
    if args.n_neuron <= 0 or args.fanout < 0:
        parser.error("--n-neuron must be positive and --fanout non-negative")
    if not 0.0 <= args.event_rate <= 1.0:
        parser.error("--event-rate must be in [0, 1]")
    return args


def main() -> None:
    """Run all requested benchmark cases."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this comparison.")
    device = torch.device("cuda")

    dataset = (
        "mice_column_v1" if args.dataset == "mice_v1_column" else args.dataset
    )
    matrix = None
    if dataset == "mice_column_v1":
        matrix = load_mice_column_v1_csr(
            args.connectome_root,
            weight_scale=args.weight_scale,
            device=device,
        )

    graph_provider = TorchCUDAGraphProvider()
    direct_cusparse_provider = DirectCuSparseProvider()
    persistent_provider = PersistentProvider()
    all_rows: list[dict] = []
    for t_steps in args.t_steps:
        if matrix is None:
            n_neuron = args.n_neuron
            fanout = args.fanout
        else:
            n_neuron = matrix.shape[0]
            fanout = int(round(matrix.indices.numel() / max(n_neuron, 1)))
        case = BenchCase(
            n_neuron=n_neuron,
            batch_size=args.batch_size,
            t_steps=t_steps,
            fanout=fanout,
            event_rate=args.event_rate,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            c_m=args.c_m,
            input_amplitude=args.input_amplitude,
            weight_scale=args.weight_scale,
        )
        case_matrix = (
            matrix if matrix is not None else make_recurrent_csr(case, device)
        )
        print(
            f"=== dataset={dataset} N={case.n_neuron} T={t_steps} "
            f"edges={case_matrix.indices.numel()} ==="
        )
        rows = bench_case(
            case,
            device=device,
            dataset=dataset,
            matrix=case_matrix,
            providers=tuple(args.providers),
            graph_provider=graph_provider,
            direct_cusparse_provider=direct_cusparse_provider,
            persistent_provider=persistent_provider,
            warmup=args.warmup,
            repeat=args.repeat,
            check_correctness=not args.skip_correctness,
        )
        for row in rows:
            print(
                f"  {row['provider']:<28} {row['status']:<20} "
                f"latency={row['latency_ms']:.4f} ms "
                "vs_direct_graph="
                f"{row['speedup_vs_cusparse_direct_cudagraph']:.3f}x"
            )
        all_rows.extend(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
