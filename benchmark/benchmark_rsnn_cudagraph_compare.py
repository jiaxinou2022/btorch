"""Compare RSNN implementations across PyTorch and SNN code generators.

The available comparison contains four public-API PyTorch baselines and the
three persistent CUDA task schedulers:

* ``torch_dense_eager`` uses :func:`torch.nn.functional.linear`.
* ``torch_dense_cudagraph`` captures the same dense forward with
  :class:`torch.cuda.CUDAGraph`.
* ``torch_csr_eager`` uses :func:`torch.sparse.mm` with a PyTorch CSR tensor.
* ``torch_csr_cudagraph`` captures the same CSR forward with a CUDA graph.
* ``cusparse_direct_eager`` calls cuSPARSE SpMV/SpMM directly from a CUDA
  extension with preallocated descriptors and workspace.
* ``cusparse_direct_cudagraph`` captures that direct CUDA execution.
* ``vdha_cudagraph``, ``vdha_pipe_cudagraph``, and ``sputnik_cudagraph`` use
  graph-safe, device-pointer launches from the corresponding
  connectome_dataset SOTA kernels.
* In-tree CUDA SpMSpV kernels expose full-window CUDA Graph variants that
  consume dense device spikes without host conversion. Their split-ABI eager
  variants remain available for end-to-end adapter comparisons. Triton, Finch,
  and the adaptive selectors remain host-controlled recurrent operators.
* ``torch_csr_host_e2e`` runs the CSR reference through the same synchronized
  wall-clock harness as the host-controlled providers.
* ``mh_spgemm_eager``, ``dtc_spmm_eager``, and ``flashsparse_eager`` also use
  correctness-first host-controlled fallbacks because their current public
  wrappers synchronize or allocate dynamically and cannot be captured.
* ``persistent_plain``, ``persistent_binning``, and
  ``persistent_spike_block`` execute one cooperative CUDA kernel per window.
* ``genn`` uses PyGeNN's generated CUDA backend with sparse connectivity.
* ``brian2cuda`` uses Brian2CUDA's single-precision standalone backend.

All providers evaluate the same recurrent LIF and ExponentialPSC equations
from the same zero state and fixed input. Thus ``flybrain`` means this common
RSNN workload running on the signed FlyWire graph; it does not enable the full
Shiu et al. refractory, delay, and hard-reset dynamics. Dataset loading,
weight conversion, code generation/compilation, CUDA graph capture, and
persistent workspace allocation are outside timing. The direct provider uses
``CUSPARSE_SPMV_ALG_DEFAULT`` for batch size one and
``CUSPARSE_SPMM_CSR_ALG1`` otherwise. GeNN and Brian2CUDA run in isolated
processes so each framework owns a clean CUDA context. Their device-synchronised
steady-state samples exclude process startup, compilation, initialisation, and
host/device result transfer. GeNN uses its native CUDA-event kernel timers and
presynaptic sparse parallelism, which avoids pathological empty-lane work on
the heavy-tailed FlyWire out-degree distribution. Framework versions, build
time, timing method, and every raw latency sample are written to the CSV.

FlyBrain (FlyWire v783) is the default dataset. Its signed synapse counts are
scaled by ``--weight-scale``. Dense providers are opt-in for FlyBrain because
materializing its whole-brain adjacency as a dense tensor is usually
impractical. The main comparison is one fixed batch-1 RSNN workload. Providers
whose public API cannot execute that workload are reported as not applicable
instead of changing the workload to suit the provider.

Usage::

    python benchmark/benchmark_rsnn_cudagraph_compare.py \
        --dataset flybrain --t-steps 128 --batch-size 1 \
        --csv benchmark/flybrain_standard_baselines.csv

To retain generated sources for artifact review, pass
``--external-build-root benchmark/generated``. The external providers are
optional: ``genn`` requires PyGeNN 5 and ``brian2cuda`` requires Brian2CUDA;
an unavailable framework is emitted as an error row without stopping the
remaining comparison. Framework optimization flags are exposed as
``--genn-*`` and ``--brian2cuda-*`` options and serialized into each external
provider's ``timing_method`` field for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "libs" / "dataset"
for import_root in (REPO_ROOT, DATASET_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from benchmark.benchmark_data import build_closed_loop_workload  # noqa: E402
from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    RSNNResult,
    csr_to_persistent_graph,
    dense_to_windowed_events,
    make_input_sequence,
    make_recurrent_csr,
)
from benchmark.provider_common import (  # noqa: E402
    BenchmarkRunner,
    PreparedMetadata,
    inspect_tensor,
    tensor_bytes,
    time_cuda_callable,
)
from benchmark.sota_rsnn_cudagraph import (  # noqa: E402
    CUDA_SPMSPV_CUDAGRAPH_PROVIDERS,
    SOTA_CUDAGRAPH_PROVIDERS,
    SOTA_EAGER_PROVIDERS,
    SOTA_PROVIDERS,
    SPMSPV_EAGER_PROVIDERS,
    SotaCUDAGraphProvider,
    SotaEagerProvider,
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
    "vdha_cudagraph",
    "vdha_pipe_cudagraph",
    "sputnik_cudagraph",
    "tilespmspv_cudagraph",
    "sortspmspv_cudagraph",
    "globalatomic_cudagraph",
    "blockatomic_cudagraph",
    "blocksort_cudagraph",
    "naivespmspv_cudagraph",
    "holaspmspv_cudagraph",
    "mh_spgemm_eager",
    "dtc_spmm_eager",
    "flashsparse_eager",
    "torch_csr_host_e2e",
    "vdha_spmspv_eager",
    "vdha_pipe_spmspv_eager",
    "tilespmspv_eager",
    "sortspmspv_eager",
    "globalatomic_eager",
    "blockatomic_eager",
    "blocksort_eager",
    "naivespmspv_eager",
    "holaspmspv_eager",
    "triton_spmspv_eager",
    "finch_spmspv_eager",
    "adaptive_spmspv_eager",
    "adaptive_direct_spmspv_eager",
    "persistent_plain",
    "persistent_binning",
    "persistent_spike_block",
    "genn",
    "brian2cuda",
]

PROVIDERS: tuple[Provider, ...] = (
    "torch_dense_eager",
    "torch_dense_cudagraph",
    "torch_csr_eager",
    "torch_csr_cudagraph",
    "cusparse_direct_eager",
    "cusparse_direct_cudagraph",
    *SOTA_PROVIDERS,
    "persistent_plain",
    "persistent_binning",
    "persistent_spike_block",
    "genn",
    "brian2cuda",
)

FLYBRAIN_DEFAULT_PROVIDERS: tuple[Provider, ...] = tuple(
    provider for provider in PROVIDERS if not provider.startswith("torch_dense_")
)
DEFAULT_WEIGHT_SCALES = {
    "flybrain": 0.275,
    "mice_column_v1": 0.15,
    "uniform": 0.15,
}
SPIKE_MISMATCH_RATE_TOL = 1e-3
STATE_RTOL = 2e-3
V_ATOL = 2e-1
PSC_ATOL = 5e-3


@dataclass(frozen=True)
class ProviderCapability:
    """Describe workload restrictions known before provider preparation."""

    batch_sizes: frozenset[int] | None = None
    batch_multiple: int = 1
    n_alignment: int = 1
    allow_n_padding: bool = True

    def supports(self, case: BenchCase) -> tuple[bool, str]:
        """Return whether the provider can execute a benchmark case."""

        if self.batch_sizes is not None and case.batch_size not in self.batch_sizes:
            supported = ", ".join(str(value) for value in sorted(self.batch_sizes))
            return False, f"batch_size={case.batch_size}; supported={supported}"
        if case.batch_size % self.batch_multiple:
            return False, (
                f"batch_size={case.batch_size}; requires a multiple of "
                f"{self.batch_multiple}"
            )
        if case.n_neuron % self.n_alignment and not self.allow_n_padding:
            return False, (
                f"n_neuron={case.n_neuron}; requires alignment {self.n_alignment}"
            )
        return True, ""


@dataclass(frozen=True)
class PaddingInfo:
    """Record reversible neuron-dimension padding."""

    logical_n: int
    physical_n: int

    @property
    def padding_ratio(self) -> float:
        """Return physical neurons per logical neuron."""

        return self.physical_n / self.logical_n


@dataclass(frozen=True)
class CalibrationResult:
    """Record closed-loop firing-rate calibration."""

    input_amplitude: float
    target_activity: float
    measured_activity: float
    activity_error: float
    calibration_status: str
    stable: bool
    iterations: int

    def to_dict(self) -> dict[str, object]:
        """Return CSV-compatible calibration fields."""

        return {
            "input_amplitude": self.input_amplitude,
            "target_activity": self.target_activity,
            "measured_activity": self.measured_activity,
            "activity_error": self.activity_error,
            "calibration_status": self.calibration_status,
            "stable": self.stable,
            "iterations": self.iterations,
        }


def pad_csr_neurons(
    matrix: CSR,
    n_alignment: int,
) -> tuple[CSR, PaddingInfo]:
    """Pad a square CSR graph with isolated neurons to an alignment."""

    if n_alignment <= 0:
        raise ValueError("n_alignment must be positive")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square graph, got {matrix.shape}")
    logical_n = matrix.shape[0]
    physical_n = (logical_n + n_alignment - 1) // n_alignment * n_alignment
    info = PaddingInfo(logical_n=logical_n, physical_n=physical_n)
    if physical_n == logical_n:
        return matrix, info
    padded_indptr = torch.empty(
        physical_n + 1,
        device=matrix.indptr.device,
        dtype=matrix.indptr.dtype,
    )
    padded_indptr[: logical_n + 1].copy_(matrix.indptr)
    padded_indptr[logical_n + 1 :].fill_(matrix.indices.numel())
    padded = CSR(
        padded_indptr,
        matrix.indices,
        matrix.data,
        shape=(physical_n, physical_n),
    )
    return padded, info


def prepare_physical_case(
    provider: Provider,
    matrix: CSR,
    x_seq: torch.Tensor,
    case: BenchCase,
) -> tuple[CSR, torch.Tensor, BenchCase, PaddingInfo]:
    """Apply provider neuron alignment without padding the batch dimension."""

    capability = PROVIDER_CAPABILITIES.get(provider, ProviderCapability())
    physical_matrix, padding = pad_csr_neurons(
        matrix,
        capability.n_alignment,
    )
    if padding.physical_n == padding.logical_n:
        return matrix, x_seq, case, padding
    physical_x = torch.zeros(
        case.t_steps,
        case.batch_size,
        padding.physical_n,
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    physical_x[..., : padding.logical_n].copy_(x_seq)
    physical_case = dataclass_replace(case, n_neuron=padding.physical_n)
    return physical_matrix, physical_x, physical_case, padding


def crop_result(result: RSNNResult, logical_n: int) -> RSNNResult:
    """Crop a padded provider result back to logical neurons."""

    return RSNNResult(
        spikes=result.spikes[..., :logical_n],
        v=result.v[..., :logical_n],
        psc=result.psc[..., :logical_n],
    )


PROVIDER_CAPABILITIES: dict[Provider, ProviderCapability] = {
    **{
        provider: ProviderCapability(batch_sizes=frozenset({1}))
        for provider in CUDA_SPMSPV_CUDAGRAPH_PROVIDERS
    },
    **{
        provider: ProviderCapability(batch_sizes=frozenset({1}))
        for provider in SPMSPV_EAGER_PROVIDERS
    },
    "mh_spgemm_eager": ProviderCapability(batch_sizes=frozenset({32, 64, 128})),
    "dtc_spmm_eager": ProviderCapability(batch_multiple=16),
    "flashsparse_eager": ProviderCapability(batch_multiple=8),
}


def provider_supports(provider: Provider, case: BenchCase) -> tuple[bool, str]:
    """Check a provider's declared capability without invoking its backend."""

    capability = PROVIDER_CAPABILITIES.get(provider, ProviderCapability())
    return capability.supports(case)


class ProviderNotApplicable(RuntimeError):
    """Signal a declared workload incompatibility before provider setup."""


def make_prepared_metadata(
    x_seq: torch.Tensor,
    case: BenchCase,
    *,
    operator_layout: str,
    index_dtype: str,
    workspace_bytes: int = 0,
    persistent_bytes: int = 0,
    execution_class: Literal[
        "device_native", "device_adapted", "host_wrapper"
    ] = "device_native",
    timing_scope: str = "gpu_execution",
    fusion_level: str = "none",
    uses_host_control: bool = False,
    uses_internal_sync: bool = False,
    uses_d2h: bool = False,
    uses_h2d: bool = False,
    allocates_during_run: bool = False,
    graph_capturable: bool = False,
    layout_transform_mode: str = "none",
    layout_transform_in_timing: bool = False,
    spike_representation: str = "dense",
) -> PreparedMetadata:
    """Build metadata from buffers actually created during preparation."""

    audit = inspect_tensor(x_seq)
    logical_shape = tuple(x_seq.shape)
    expected_shape = (case.t_steps, case.batch_size, case.n_neuron)
    if logical_shape != expected_shape:
        raise ValueError(
            f"prepared input shape {logical_shape} does not match {expected_shape}"
        )
    return PreparedMetadata(
        logical_shape=logical_shape,
        physical_shape=logical_shape,
        input_layout="BN",
        operator_layout=operator_layout,
        index_dtype=index_dtype,
        value_dtype=str(x_seq.dtype),
        compute_dtype=str(x_seq.dtype),
        workspace_bytes=workspace_bytes,
        persistent_bytes=persistent_bytes,
        padding_ratio=1.0,
        execution_class=execution_class,
        timing_scope=timing_scope,
        fusion_level=fusion_level,
        uses_host_control=uses_host_control,
        uses_internal_sync=uses_internal_sync,
        uses_d2h=uses_d2h,
        uses_h2d=uses_h2d,
        allocates_during_run=allocates_during_run,
        graph_capturable=graph_capturable,
        layout_transform_mode=layout_transform_mode,
        layout_transform_in_timing=layout_transform_in_timing,
        spike_representation=spike_representation,
        input_contiguous=bool(audit["contiguous"]),
        input_address_mod=int(audit["address_alignment"]),
    )


def unprepared_metadata(provider: Provider, case: BenchCase) -> PreparedMetadata:
    """Return declared metadata for a provider that was not prepared."""

    host_wrapper = provider in SOTA_EAGER_PROVIDERS or provider in {
        "genn",
        "brian2cuda",
    }
    device_adapted = provider in SOTA_CUDAGRAPH_PROVIDERS
    return PreparedMetadata(
        logical_shape=(case.t_steps, case.batch_size, case.n_neuron),
        physical_shape=(case.t_steps, case.batch_size, case.n_neuron),
        input_layout="BN",
        operator_layout="not_prepared",
        index_dtype="unknown",
        value_dtype="unknown",
        compute_dtype="unknown",
        workspace_bytes=0,
        persistent_bytes=0,
        padding_ratio=1.0,
        execution_class=(
            "host_wrapper"
            if host_wrapper
            else "device_adapted"
            if device_adapted
            else "device_native"
        ),
        timing_scope=("public_wrapper_e2e" if host_wrapper else "gpu_execution"),
        uses_host_control=host_wrapper,
        graph_capturable=provider.endswith("_cudagraph"),
        layout_transform_mode="not_prepared",
        spike_representation="unknown",
    )


def resolve_dataset_defaults(
    dataset: str,
    weight_scale: float | None,
) -> tuple[str, float]:
    """Canonicalize a dataset name and select its default weight scale."""

    aliases = {"mice_v1_column": "mice_column_v1", "flywire_783": "flybrain"}
    canonical = aliases.get(dataset, dataset)
    if canonical not in DEFAULT_WEIGHT_SCALES:
        raise ValueError(f"Unsupported dataset: {dataset}.")
    scale = (
        DEFAULT_WEIGHT_SCALES[canonical]
        if weight_scale is None
        else float(weight_scale)
    )
    return canonical, scale


def load_flybrain_csr(
    root: Path | None, *, weight_scale: float, device: torch.device
) -> CSR:
    """Load the signed FlyWire v783 graph used by the FlyBrain model.

    The source data stores signed synapse counts with pre-synaptic rows. The
    FlyBrain model converts those counts to synaptic strength by multiplying
    them by one scalar ``w_syn``; ``weight_scale`` serves that role here.
    """

    try:
        from connectome_dataset.graph_loader import load_flywire_783
    except ImportError as exc:
        raise RuntimeError(
            "connectome_dataset is required for --dataset flybrain"
        ) from exc

    try:
        scipy_matrix = load_flywire_783(root=root, use_weights=True).tocsr()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nFetch connectome_dataset/data/external/flywire_783 "
            "or pass --connectome-root."
        ) from exc
    if scipy_matrix.shape[0] != scipy_matrix.shape[1]:
        raise ValueError(f"flybrain must be square, got {scipy_matrix.shape}.")
    scipy_matrix.data *= weight_scale
    return CSR.from_scipy(scipy_matrix, device=device, dtype=torch.float32)


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
        v_pre = v + case.dt * (-(v - case.v_reset) / case.tau_mem + current / case.c_m)
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
        v_pre = v + case.dt * (-(v - case.v_reset) / case.tau_mem + current / case.c_m)
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

    metadata = make_prepared_metadata(
        x_seq,
        case,
        operator_layout="post_by_pre_dense" if not sparse else "csr_post_by_pre",
        index_dtype=("none" if not sparse else str(weight.crow_indices().dtype)),
        persistent_bytes=(
            tensor_bytes(
                [
                    weight.crow_indices(),
                    weight.col_indices(),
                    weight.values(),
                ]
            )
            if sparse
            else weight.numel() * weight.element_size()
        ),
        allocates_during_run=True,
    )
    return BenchmarkRunner(
        run_fn=run,
        reset_fn=lambda: None,
        metadata=metadata,
    )


def calibrate_input_amplitude(
    case: BenchCase,
    matrix: CSR,
    device: torch.device,
    target_activity: float,
    *,
    min_amplitude: float = 0.0,
    max_amplitude: float = 100.0,
    max_iterations: int = 12,
    relative_tolerance: float = 0.1,
) -> CalibrationResult:
    """Calibrate deterministic external input against closed-loop activity."""

    if not 0.0 <= target_activity <= 1.0:
        raise ValueError("target_activity must be in [0, 1]")
    if min_amplitude < 0.0 or max_amplitude <= min_amplitude:
        raise ValueError("invalid calibration amplitude bounds")
    weight = make_torch_csr_weight(matrix)
    low, high = min_amplitude, max_amplitude
    observations: list[tuple[float, float, bool]] = []
    best: tuple[float, float, bool] | None = None
    for _ in range(max_iterations):
        amplitude = (low + high) / 2.0
        candidate = dataclass_replace(case, input_amplitude=amplitude)
        x_seq = make_input_sequence(candidate, device)
        result = make_eager_runner(
            x_seq,
            weight,
            candidate,
            sparse=True,
        )()
        measured = float(result.spikes.mean().item())
        timestep_rates = result.spikes.float().mean(dim=(1, 2))
        mean_rate = float(timestep_rates.mean().item())
        burst_factor = (
            float(timestep_rates.max().item()) / mean_rate if mean_rate else 0.0
        )
        stable = bool(
            torch.isfinite(result.v).all()
            and torch.isfinite(result.psc).all()
            and burst_factor <= 10.0
        )
        observations.append((amplitude, measured, stable))
        if best is None or abs(measured - target_activity) < abs(
            best[1] - target_activity
        ):
            best = (amplitude, measured, stable)
        if measured < target_activity:
            low = amplitude
        else:
            high = amplitude
    assert best is not None
    ordered = sorted(observations)
    monotonic = all(
        right[1] + 1e-6 >= left[1]
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    error = abs(best[1] - target_activity)
    tolerance = max(
        target_activity * relative_tolerance,
        1.0 / max(case.t_steps * case.n_neuron, 1),
    )
    if not best[2]:
        status = "unstable"
    elif not monotonic:
        status = "nonmonotonic_best_effort"
    elif error <= tolerance:
        status = "converged"
    else:
        status = "best_effort"
    return CalibrationResult(
        input_amplitude=best[0],
        target_activity=target_activity,
        measured_activity=best[1],
        activity_error=error,
        calibration_status=status,
        stable=best[2],
        iterations=len(observations),
    )


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
        metadata = make_prepared_metadata(
            static_x,
            case,
            operator_layout=("csr_post_by_pre" if sparse else "post_by_pre_dense"),
            index_dtype=(str(weight.crow_indices().dtype) if sparse else "none"),
            persistent_bytes=(
                (
                    tensor_bytes(
                        [
                            weight.crow_indices(),
                            weight.col_indices(),
                            weight.values(),
                        ]
                    )
                    if sparse
                    else weight.numel() * weight.element_size()
                )
                + tensor_bytes([static_x, static_v0, static_psc0])
            ),
            graph_capturable=True,
        )
        runner = BenchmarkRunner(
            run_fn=run,
            reset_fn=lambda: None,
            metadata=metadata,
        )
        self._runners[key] = runner
        return runner


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
        initial_v = state.v.clone()
        initial_psc = state.psc.clone()
        current_state = state
        workspace_tensors = (
            workspace.input_current,
            workspace.queue_batch,
            workspace.queue_edge_start,
            workspace.queue_edge_end,
            workspace.spike_count,
            workspace.work_counter,
        )

        def reset() -> None:
            current_state.v.copy_(initial_v)
            current_state.psc.copy_(initial_psc)
            for tensor in workspace_tensors:
                tensor.zero_()

        def run() -> RSNNResult:
            nonlocal current_state
            output = persistent_snn_forward(
                events,
                graph,
                current_state,
                params,
                backend="cuda_persistent",
                return_mode="dense",
                workspace=workspace,
                **options,
            )
            current_state = output.state
            assert output.spikes is not None
            return RSNNResult(
                spikes=output.spikes,
                v=output.state.v,
                psc=output.state.psc,
            )

        runner = BenchmarkRunner(
            run_fn=run,
            reset_fn=reset,
            metadata=make_prepared_metadata(
                x_seq,
                case,
                operator_layout="event_csr",
                index_dtype=str(graph.indptr.dtype),
                workspace_bytes=tensor_bytes(list(workspace_tensors)),
                persistent_bytes=tensor_bytes(
                    [
                        events.offsets,
                        events.indices,
                        graph.indptr,
                        graph.indices,
                        graph.weight,
                        state.v,
                        state.psc,
                    ]
                ),
                fusion_level="full_fusion",
                layout_transform_mode="dense_to_events_precomputed",
                layout_transform_in_timing=False,
                spike_representation="windowed_events",
            ),
            reset_policy="before_sample",
        )
        runner.reset()
        self._runners[key] = runner
        return runner


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
        workspace_bytes = int(extension.workspace_bytes(plan))

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
            runner = BenchmarkRunner(
                run_fn=execute,
                reset_fn=lambda: None,
                metadata=make_prepared_metadata(
                    static_x,
                    case,
                    operator_layout="csr_post_by_pre",
                    index_dtype=str(crow.dtype),
                    workspace_bytes=workspace_bytes,
                    persistent_bytes=tensor_bytes(
                        [
                            crow,
                            col,
                            values,
                            static_x,
                            v,
                            psc,
                            spikes,
                            recurrent,
                        ]
                    ),
                    fusion_level="sparse_only",
                    allocates_during_run=False,
                    graph_capturable=True,
                ),
            )
            self._runners[key] = runner
            return runner

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

        runner = BenchmarkRunner(
            run_fn=replay,
            reset_fn=lambda: None,
            metadata=make_prepared_metadata(
                static_x,
                case,
                operator_layout="csr_post_by_pre",
                index_dtype=str(crow.dtype),
                workspace_bytes=workspace_bytes,
                persistent_bytes=tensor_bytes(
                    [
                        crow,
                        col,
                        values,
                        static_x,
                        v,
                        psc,
                        spikes,
                        recurrent,
                    ]
                ),
                fusion_level="sparse_only",
                graph_capturable=True,
            ),
        )
        self._runners[key] = runner
        return runner


def _reset_runner(fn) -> None:
    reset = getattr(fn, "reset", None)
    if reset is not None:
        reset()


def _reset_before_sample(fn) -> None:
    if getattr(fn, "reset_policy", "stateless") == "before_sample":
        _reset_runner(fn)


def audit_runner_memory(runner: BenchmarkRunner, *, repeat: int = 3) -> None:
    """Record persistent allocator deltas across untimed prepared runs."""

    metadata = runner.metadata
    if not torch.cuda.is_available() or metadata.execution_class == "host_wrapper":
        return
    runner.reset()
    runner()
    torch.cuda.synchronize()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    for _ in range(repeat):
        _reset_before_sample(runner)
        result = runner()
        del result
    torch.cuda.synchronize()
    metadata.run_allocated_bytes = torch.cuda.memory_allocated() - before_allocated
    metadata.run_reserved_delta = torch.cuda.memory_reserved() - before_reserved
    metadata.allocates_during_run = metadata.allocates_during_run or (
        metadata.run_allocated_bytes != 0 or metadata.run_reserved_delta != 0
    )


def time_samples_ms(fn, *, warmup: int, repeat: int) -> list[float]:
    """Measure GPU execution after isolated warmup and timing resets."""

    return time_cuda_callable(
        fn,
        reset_fn=getattr(fn, "reset", lambda: None),
        reset_policy=getattr(fn, "reset_policy", "stateless"),
        warmup=warmup,
        repeat=repeat,
        timing_mode="queued",
    )


def time_host_samples_ms(fn, *, warmup: int, repeat: int) -> list[float]:
    """Measure synchronized wall time for host-controlled GPU providers."""

    _reset_runner(fn)
    for _ in range(warmup):
        _reset_before_sample(fn)
        fn()
    torch.cuda.synchronize()
    _reset_runner(fn)
    samples = []
    for _ in range(repeat):
        _reset_before_sample(fn)
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def time_host_samples_with_result(
    fn, *, warmup: int, repeat: int
) -> tuple[list[float], RSNNResult]:
    """Measure host-controlled execution and retain the last timed result."""

    _reset_runner(fn)
    for _ in range(warmup):
        _reset_before_sample(fn)
        fn()
    torch.cuda.synchronize()
    _reset_runner(fn)
    samples = []
    result = None
    for _ in range(repeat):
        _reset_before_sample(fn)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    assert result is not None
    return samples, result


def median_ms(samples: list[float]) -> float:
    """Return the conventional median of non-empty latency samples."""

    return float(statistics.median(samples))


def latency_summary(samples: list[float]) -> dict[str, float | int]:
    """Summarize raw latency samples with auditable standard statistics."""

    if not samples:
        return {
            "sample_count": 0,
            "latency_mean_ms": float("nan"),
            "latency_std_ms": float("nan"),
            "latency_min_ms": float("nan"),
            "latency_max_ms": float("nan"),
        }
    return {
        "sample_count": len(samples),
        "latency_mean_ms": float(statistics.mean(samples)),
        "latency_std_ms": (
            float(statistics.stdev(samples)) if len(samples) > 1 else float("nan")
        ),
        "latency_min_ms": float(min(samples)),
        "latency_max_ms": float(max(samples)),
    }


def time_ms(fn, *, warmup: int, repeat: int) -> float:
    """Measure median GPU stream time with CUDA events."""

    return median_ms(time_samples_ms(fn, warmup=warmup, repeat=repeat))


def max_normalized_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> float:
    """Return the largest error relative to an ``atol + rtol * abs(x)`` bound."""

    bound = atol + rtol * expected.abs()
    return float(((actual - expected).abs() / bound).max().item())


def correctness_metrics(
    result: RSNNResult,
    reference: RSNNResult,
    *,
    strict_state: bool = True,
) -> dict:
    """Return exact-state or spike-equivalence correctness metrics.

    Reduced-precision tensor-core recurrent operators can follow a nearby
    threshold trajectory while their final analog state diverges. For those
    providers, spike equivalence is the acceptance criterion and state errors
    remain diagnostic. Exact providers must satisfy both criteria.
    """

    spike_mismatches = int((result.spikes != reference.spikes).sum().item())
    spike_mismatch_rate = spike_mismatches / reference.spikes.numel()
    v_diff = float((result.v - reference.v).abs().max().item())
    psc_diff = float((result.psc - reference.psc).abs().max().item())
    v_normalized = max_normalized_error(
        result.v,
        reference.v,
        atol=V_ATOL,
        rtol=STATE_RTOL,
    )
    psc_normalized = max_normalized_error(
        result.psc,
        reference.psc,
        atol=PSC_ATOL,
        rtol=STATE_RTOL,
    )
    spikes_passed = spike_mismatch_rate <= SPIKE_MISMATCH_RATE_TOL
    states_finite = bool(
        torch.isfinite(result.v).all() and torch.isfinite(result.psc).all()
    )
    state_passed = v_normalized <= 1.0 and psc_normalized <= 1.0
    passed = spikes_passed and states_finite and (state_passed or not strict_state)
    if passed and not strict_state and not state_passed:
        status = "passed_spike_equivalent"
    else:
        status = "passed" if passed else "correctness_failed"
    return {
        "status": status,
        "spike_mismatches": spike_mismatches,
        "spike_mismatch_rate": spike_mismatch_rate,
        "v_max_abs_diff": v_diff,
        "psc_max_abs_diff": psc_diff,
        "v_max_normalized_error": v_normalized,
        "psc_max_normalized_error": psc_normalized,
    }


def _empty_metrics(status: str = "not_checked") -> dict:
    return {
        "status": status,
        "spike_mismatches": -1,
        "spike_mismatch_rate": float("nan"),
        "v_max_abs_diff": float("nan"),
        "psc_max_abs_diff": float("nan"),
        "v_max_normalized_error": float("nan"),
        "psc_max_normalized_error": float("nan"),
    }


def comparable_rows(row: dict, baseline: dict) -> bool:
    """Return whether two result rows belong to one comparison group."""

    fields = (
        "dataset",
        "logical_n",
        "logical_batch",
        "t_steps",
        "timing_scope",
        "execution_class",
        "benchmark_mode",
        "state_semantics",
    )
    return all(row[field] == baseline[field] for field in fields)


def bench_case(
    case: BenchCase,
    *,
    device: torch.device,
    dataset: str,
    matrix: CSR,
    providers: tuple[Provider, ...],
    graph_provider: TorchCUDAGraphProvider,
    direct_cusparse_provider: DirectCuSparseProvider,
    sota_cudagraph_provider: SotaCUDAGraphProvider,
    sota_eager_provider: SotaEagerProvider | None = None,
    persistent_provider: PersistentProvider,
    external_build_root: Path | None = None,
    external_timeout: float = 1800.0,
    external_provider_options: dict[str, dict[str, object]] | None = None,
    warmup: int,
    repeat: int,
    eager_warmup: int = 0,
    eager_repeat: int = 1,
    check_correctness: bool,
    continuous_state: bool = False,
    requested_activity: float | None = None,
    calibration: dict[str, object] | None = None,
    audit_manifest: dict[str, dict[str, object]] | None = None,
) -> list[dict]:
    """Prepare and benchmark all selected providers for one case."""

    x_seq = make_input_sequence(case, device)
    csr_weight = make_torch_csr_weight(matrix)
    reference = make_eager_runner(x_seq, csr_weight, case, sparse=True)()
    workload = build_closed_loop_workload(
        matrix,
        x_seq,
        reference.spikes,
        dataset=dataset,
        seed=20250308,
        requested_activity=requested_activity,
    )
    graph_stats = workload.graph_stats
    assert workload.spike_stats is not None
    spike_stats = workload.spike_stats
    workload_id = workload.workload_id
    physical_cache: dict[
        int, tuple[CSR, torch.Tensor, BenchCase, PaddingInfo, torch.Tensor]
    ] = {}

    rows = []
    for provider in providers:
        provider_audit = (audit_manifest or {}).get(provider, {})
        prepared_metadata = unprepared_metadata(provider, case)
        framework_version = ""
        build_time_s = float("nan")
        timing_method = "torch_cuda_event"
        samples: list[float] = []
        status_detail = ""
        try:
            supported, unsupported_reason = provider_supports(provider, case)
            if not supported:
                raise ProviderNotApplicable(unsupported_reason)
            capability = PROVIDER_CAPABILITIES.get(
                provider,
                ProviderCapability(),
            )
            cached_physical = physical_cache.get(capability.n_alignment)
            if cached_physical is None:
                (
                    provider_matrix,
                    provider_x,
                    provider_case,
                    padding,
                ) = prepare_physical_case(provider, matrix, x_seq, case)
                provider_csr_weight = make_torch_csr_weight(provider_matrix)
                cached_physical = (
                    provider_matrix,
                    provider_x,
                    provider_case,
                    padding,
                    provider_csr_weight,
                )
                physical_cache[capability.n_alignment] = cached_physical
            (
                provider_matrix,
                provider_x,
                provider_case,
                padding,
                provider_csr_weight,
            ) = cached_physical
            if provider in ("genn", "brian2cuda"):
                from benchmark.external_rsnn_simulators import run_external_provider

                external = run_external_provider(
                    provider,
                    x_seq,
                    matrix,
                    case,
                    warmup=warmup,
                    repeat=repeat,
                    check_correctness=check_correctness,
                    build_root=external_build_root,
                    timeout_s=external_timeout,
                    provider_options=((external_provider_options or {}).get(provider)),
                )
                result = external.result
                samples = external.latency_samples_ms
                framework_version = external.framework_version
                build_time_s = external.build_time_s
                timing_method = external.timing_method
                prepared_metadata = unprepared_metadata(provider, case)
                prepared_metadata.timing_scope = "framework_native_timing"
                metrics = (
                    correctness_metrics(result, reference)
                    if check_correctness and result is not None
                    else _empty_metrics()
                )
                latency = median_ms(samples)
                run = None
            elif provider == "torch_dense_eager":
                dense_weight = provider_csr_weight.to_dense()
                run = make_eager_runner(
                    provider_x,
                    dense_weight,
                    provider_case,
                    sparse=False,
                )
            elif provider == "torch_dense_cudagraph":
                dense_weight = provider_csr_weight.to_dense()
                run = graph_provider.fixed_runner(
                    provider_x,
                    dense_weight,
                    provider_case,
                    sparse=False,
                )
            elif provider == "torch_csr_eager":
                run = make_eager_runner(
                    provider_x,
                    provider_csr_weight,
                    provider_case,
                    sparse=True,
                )
            elif provider == "torch_csr_cudagraph":
                run = graph_provider.fixed_runner(
                    provider_x,
                    provider_csr_weight,
                    provider_case,
                    sparse=True,
                )
            elif provider == "cusparse_direct_eager":
                run = direct_cusparse_provider.fixed_runner(
                    provider_x,
                    provider_csr_weight,
                    provider_case,
                    use_cudagraph=False,
                )
            elif provider == "cusparse_direct_cudagraph":
                run = direct_cusparse_provider.fixed_runner(
                    provider_x,
                    provider_csr_weight,
                    provider_case,
                    use_cudagraph=True,
                )
            elif provider in SOTA_CUDAGRAPH_PROVIDERS:
                run = sota_cudagraph_provider.fixed_runner(
                    provider,
                    provider_x,
                    provider_csr_weight,
                    provider_case,
                )
                timing_method = "torch_cuda_event_cudagraph_replay"
            elif provider in SOTA_EAGER_PROVIDERS:
                if sota_eager_provider is None:
                    sota_eager_provider = SotaEagerProvider()
                print(
                    f"[prepare] provider={provider} N={case.n_neuron} "
                    f"B={case.batch_size} T={case.t_steps}",
                    flush=True,
                )
                torch.cuda.synchronize()
                prepare_start = time.perf_counter()
                run = sota_eager_provider.fixed_runner(
                    provider,
                    provider_x,
                    provider_csr_weight,
                    provider_case,
                )
                torch.cuda.synchronize()
                build_time_s = time.perf_counter() - prepare_start
                print(
                    f"[run] provider={provider} "
                    f"prepare={build_time_s:.3f}s "
                    f"samples={eager_repeat}",
                    flush=True,
                )
                timing_method = "synchronized_wall_clock_host_controlled_eager"
            elif provider.startswith("persistent_"):
                run = persistent_provider.fixed_runner(
                    provider_x,
                    provider_matrix,
                    provider_case,
                    variant=provider.removeprefix("persistent_"),
                )
            else:
                raise ValueError(provider)

            if run is not None:
                if not isinstance(run, BenchmarkRunner):
                    raise TypeError(f"{provider} prepare must return BenchmarkRunner")
                run.prepare()
                prepared_metadata = run.metadata
                prepared_metadata.logical_shape = tuple(x_seq.shape)
                prepared_metadata.physical_shape = tuple(provider_x.shape)
                prepared_metadata.padding_ratio = padding.padding_ratio
                if continuous_state and run.reset_policy == "before_sample":
                    run.reset_policy = "before_phase"
                    prepared_metadata.state_semantics = "continuous_samples"
                audit_runner_memory(run)
                if provider in SOTA_EAGER_PROVIDERS:
                    samples, result = time_host_samples_with_result(
                        run,
                        warmup=eager_warmup,
                        repeat=eager_repeat,
                    )
                else:
                    run.reset()
                    result = run()
                result = crop_result(result, case.n_neuron)
                metrics = (
                    correctness_metrics(
                        result,
                        reference,
                        strict_state=provider
                        not in {"dtc_spmm_eager", "flashsparse_eager"},
                    )
                    if check_correctness
                    else _empty_metrics()
                )
                if provider not in SOTA_EAGER_PROVIDERS:
                    samples = time_samples_ms(run, warmup=warmup, repeat=repeat)
                latency = median_ms(samples)
        except ProviderNotApplicable as exc:
            metrics = _empty_metrics("not_applicable")
            status_detail = str(exc)
            timing_method = "not_run"
            latency = float("nan")
            print(
                f"[not_applicable] provider={provider} "
                f"N={case.n_neuron} B={case.batch_size}: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            optional_unavailable = provider in SOTA_PROVIDERS and isinstance(
                exc, (ImportError, NotImplementedError, OSError)
            )
            outcome = "unavailable" if optional_unavailable else "error"
            metrics = _empty_metrics(f"{outcome}:{type(exc).__name__}: {exc}")
            status_detail = str(exc)
            latency = float("nan")
            print(f"[{outcome}] provider={provider} N={case.n_neuron}: {exc}")

        rows.append(
            {
                "provider": provider,
                "dataset": dataset,
                "workload_id": workload_id,
                "n_neuron": case.n_neuron,
                "t_steps": case.t_steps,
                "batch_size": case.batch_size,
                "graph_synapses": int(matrix.indices.numel()),
                "average_fanout": matrix.indices.numel() / case.n_neuron,
                "graph_type": graph_stats.graph_type,
                "graph_seed": graph_stats.seed,
                "graph_max_fanout": graph_stats.max_fanout,
                "graph_fanout_std": graph_stats.fanout_std,
                "graph_fanout_cv": graph_stats.fanout_cv,
                "graph_fanout_gini": graph_stats.fanout_gini,
                "graph_p95_fanout": graph_stats.p95_fanout,
                "graph_p99_fanout": graph_stats.p99_fanout,
                "requested_activity": spike_stats.requested_activity,
                "measured_activity": spike_stats.measured_activity,
                "mean_active_neurons": spike_stats.mean_active_neurons,
                "p95_active_neurons": spike_stats.p95_active_neurons,
                "max_active_neurons": spike_stats.max_active_neurons,
                "mean_active_edges": spike_stats.mean_active_edges,
                "p95_active_edges": spike_stats.p95_active_edges,
                "max_active_edges": spike_stats.max_active_edges,
                "mean_collision_ratio": spike_stats.mean_collision_ratio,
                "p95_collision_ratio": spike_stats.p95_collision_ratio,
                "temporal_cv": spike_stats.temporal_cv,
                "burst_factor": spike_stats.burst_factor,
                "collision_method": spike_stats.collision_method,
                "calibration_status": (
                    (calibration or {}).get("calibration_status", "not_requested")
                ),
                "calibration_input_amplitude": (
                    (calibration or {}).get(
                        "input_amplitude",
                        case.input_amplitude,
                    )
                ),
                "calibration_target_activity": (
                    (calibration or {}).get(
                        "target_activity",
                        float("nan"),
                    )
                ),
                "calibration_measured_activity": (
                    (calibration or {}).get(
                        "measured_activity",
                        float("nan"),
                    )
                ),
                "calibration_activity_error": (
                    (calibration or {}).get(
                        "activity_error",
                        float("nan"),
                    )
                ),
                "calibration_stable": ((calibration or {}).get("stable", True)),
                "calibration_iterations": ((calibration or {}).get("iterations", 0)),
                "direct_cusparse_primitive": (
                    "SpMV" if case.batch_size == 1 else "SpMM"
                ),
                "direct_cusparse_algorithm": (
                    "CUSPARSE_SPMV_ALG_DEFAULT"
                    if case.batch_size == 1
                    else "CUSPARSE_SPMM_CSR_ALG1"
                ),
                "framework_version": framework_version,
                "build_time_s": build_time_s,
                "timing_method": timing_method,
                "timing_scope": prepared_metadata.timing_scope,
                "benchmark_mode": "full_rsnn",
                "execution_class": prepared_metadata.execution_class,
                "state_semantics": prepared_metadata.state_semantics,
                "fusion_level": prepared_metadata.fusion_level,
                "logical_shape": str(prepared_metadata.logical_shape),
                "physical_shape": str(prepared_metadata.physical_shape),
                "logical_n": prepared_metadata.logical_n,
                "physical_n": prepared_metadata.physical_n,
                "logical_batch": prepared_metadata.logical_batch,
                "physical_batch": prepared_metadata.physical_batch,
                "input_layout": prepared_metadata.input_layout,
                "operator_layout": prepared_metadata.operator_layout,
                "required_layout": prepared_metadata.operator_layout,
                "layout_transform_mode": (prepared_metadata.layout_transform_mode),
                "layout_transform_in_timing": (
                    prepared_metadata.layout_transform_in_timing
                ),
                "need_layout_transform": (
                    prepared_metadata.layout_transform_mode != "none"
                ),
                "input_contiguous": prepared_metadata.input_contiguous,
                "input_address_mod": prepared_metadata.input_address_mod,
                "index_dtype": prepared_metadata.index_dtype,
                "value_dtype": prepared_metadata.value_dtype,
                "compute_dtype": prepared_metadata.compute_dtype,
                "spike_representation": (prepared_metadata.spike_representation),
                "selected_kernels": prepared_metadata.selected_kernels,
                "selector_model": prepared_metadata.selector_model,
                "uses_host_control": prepared_metadata.uses_host_control,
                "uses_internal_sync": prepared_metadata.uses_internal_sync,
                "uses_d2h": prepared_metadata.uses_d2h,
                "uses_h2d": prepared_metadata.uses_h2d,
                "allocates_during_run": (
                    prepared_metadata.allocates_during_run
                    or bool(provider_audit.get("unexpected_alloc_ops", []))
                ),
                "graph_capturable": prepared_metadata.graph_capturable,
                "padding_ratio": prepared_metadata.padding_ratio,
                "workspace_bytes": prepared_metadata.workspace_bytes,
                "total_prepared_bytes": (prepared_metadata.total_prepared_bytes),
                "persistent_bytes": prepared_metadata.persistent_bytes,
                "run_allocated_bytes": (prepared_metadata.run_allocated_bytes),
                "run_reserved_delta": prepared_metadata.run_reserved_delta,
                "profiler_audited": provider_audit.get("audited", False),
                "profiler_cuda_malloc": provider_audit.get(
                    "cuda_malloc_in_run",
                    "unknown",
                ),
                "profiler_cuda_free": provider_audit.get(
                    "cuda_free_in_run",
                    "unknown",
                ),
                "profiler_d2h": provider_audit.get(
                    "d2h_in_run",
                    "unknown",
                ),
                "profiler_h2d": provider_audit.get(
                    "h2d_in_run",
                    "unknown",
                ),
                "profiler_internal_sync": provider_audit.get(
                    "internal_sync_in_run",
                    "unknown",
                ),
                "profiler_unexpected_alloc_ops": ";".join(
                    str(value)
                    for value in provider_audit.get(
                        "unexpected_alloc_ops",
                        [],
                    )
                ),
                "latency_samples_ms": ";".join(f"{x:.9g}" for x in samples),
                **latency_summary(samples),
                **metrics,
                "correctness_status": metrics["status"],
                "status_detail": status_detail,
                "latency_ms": latency,
                "speedup_vs_dense_eager": float("nan"),
                "speedup_vs_dense_cudagraph": float("nan"),
                "speedup_vs_csr_eager": float("nan"),
                "speedup_vs_csr_host_e2e": float("nan"),
                "speedup_vs_csr_cudagraph": float("nan"),
                "speedup_vs_cusparse_direct_eager": float("nan"),
                "speedup_vs_cusparse_direct_cudagraph": float("nan"),
                "speedup_vs_genn": float("nan"),
                "speedup_vs_brian2cuda": float("nan"),
                "speedup_vs_same_eager": float("nan"),
            }
        )

    row_by_provider = {row["provider"]: row for row in rows}
    baselines = (
        ("torch_dense_eager", "speedup_vs_dense_eager"),
        ("torch_dense_cudagraph", "speedup_vs_dense_cudagraph"),
        ("torch_csr_eager", "speedup_vs_csr_eager"),
        ("torch_csr_host_e2e", "speedup_vs_csr_host_e2e"),
        ("torch_csr_cudagraph", "speedup_vs_csr_cudagraph"),
        ("cusparse_direct_eager", "speedup_vs_cusparse_direct_eager"),
        (
            "cusparse_direct_cudagraph",
            "speedup_vs_cusparse_direct_cudagraph",
        ),
        ("genn", "speedup_vs_genn"),
        ("brian2cuda", "speedup_vs_brian2cuda"),
    )
    for row in rows:
        latency = float(row["latency_ms"])
        if not math.isfinite(latency) or latency <= 0:
            continue
        for baseline, column in baselines:
            baseline_row = row_by_provider.get(baseline)
            if baseline_row is None:
                continue
            baseline_latency = float(baseline_row["latency_ms"])
            if comparable_rows(row, baseline_row) and math.isfinite(baseline_latency):
                row[column] = baseline_latency / latency

        eager_provider = {
            graph_provider: f"{kernel_name}_eager"
            for graph_provider, kernel_name in CUDA_SPMSPV_CUDAGRAPH_PROVIDERS.items()
        }.get(str(row["provider"]))
        eager_provider = {
            "vdha_eager": "vdha_spmspv_eager",
            "vdha_pipe_eager": "vdha_pipe_spmspv_eager",
        }.get(eager_provider, eager_provider)
        eager_row = row_by_provider.get(eager_provider) if eager_provider else None
        if eager_row is not None:
            eager_latency = float(eager_row["latency_ms"])
            same_workload = all(
                row[field] == eager_row[field]
                for field in (
                    "dataset",
                    "logical_n",
                    "logical_batch",
                    "t_steps",
                    "benchmark_mode",
                    "state_semantics",
                )
            )
            if same_workload and math.isfinite(eager_latency):
                row["speedup_vs_same_eager"] = eager_latency / latency
    return rows


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=(
            "flybrain",
            "flywire_783",
            "uniform",
            "mice_column_v1",
            "mice_v1_column",
        ),
        default="flybrain",
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
    parser.add_argument(
        "--target-activity",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Calibrate input amplitude to each requested closed-loop firing "
            "rate before benchmarking."
        ),
    )
    parser.add_argument("--calibration-min-amplitude", type=float, default=0.0)
    parser.add_argument("--calibration-max-amplitude", type=float, default=100.0)
    parser.add_argument("--calibration-iterations", type=int, default=12)
    parser.add_argument("--calibration-relative-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=None,
        help=(
            "Global recurrent weight scale. Defaults to 0.275 for FlyBrain "
            "and 0.15 for mice_column_v1 or uniform."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--eager-warmup",
        type=int,
        default=0,
        help="Warmups for host-controlled SOTA fallbacks (default: 0).",
    )
    parser.add_argument(
        "--eager-repeat",
        type=int,
        default=1,
        help="Timed repeats for host-controlled SOTA fallbacks (default: 1).",
    )
    parser.add_argument(
        "--spmspv-selector-model",
        type=Path,
        default=None,
        help=("RTX-specific cascade selector model for adaptive_spmspv_eager."),
    )
    parser.add_argument(
        "--spmspv-direct-selector-model",
        type=Path,
        default=None,
        help=("RTX-specific direct selector model for adaptive_direct_spmspv_eager."),
    )
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=None)
    parser.add_argument(
        "--external-build-root",
        type=Path,
        default=None,
        help=(
            "Keep GeNN/Brian2CUDA generated projects below this directory. "
            "By default a temporary directory is removed after each case."
        ),
    )
    parser.add_argument(
        "--external-timeout",
        type=float,
        default=1800.0,
        help=(
            "Maximum seconds allowed for one GeNN or Brian2CUDA worker, "
            "including code generation and compilation (default: 1800)."
        ),
    )
    parser.add_argument(
        "--genn-optimize-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GeNN release compiler optimization (default: enabled).",
    )
    parser.add_argument(
        "--brian2cuda-sm-multiplier",
        type=int,
        default=1,
        help="Brian2CUDA blocks per streaming multiprocessor (default: 1).",
    )
    parser.add_argument(
        "--brian2cuda-parallel-blocks",
        type=int,
        default=1,
        help=(
            "Brian2CUDA synaptic parallel blocks; 0 uses the SM count times "
            "--brian2cuda-sm-multiplier (default: 1)."
        ),
    )
    parser.add_argument(
        "--brian2cuda-extra-threshold-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--brian2cuda-calc-occupancy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--brian2cuda-launch-bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--brian2cuda-syn-launch-bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--continuous-state",
        action="store_true",
        help=(
            "Carry mutable provider state between samples. The default treats "
            "every latency sample as one independent inference."
        ),
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=None,
        help="Attach one-time provider profiler audit results to CSV rows.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.eager_warmup < 0 or args.eager_repeat <= 0:
        parser.error("--eager-warmup must be non-negative and --eager-repeat positive")
    if args.batch_size != 1:
        parser.error("the main RSNN comparison requires --batch-size 1")
    if any(t_steps <= 0 for t_steps in args.t_steps):
        parser.error("every --t-steps value must be positive")
    if args.n_neuron <= 0 or args.fanout < 0:
        parser.error("--n-neuron must be positive and --fanout non-negative")
    if args.external_timeout <= 0:
        parser.error("--external-timeout must be positive")
    if args.brian2cuda_sm_multiplier <= 0:
        parser.error("--brian2cuda-sm-multiplier must be positive")
    if args.brian2cuda_parallel_blocks < 0:
        parser.error("--brian2cuda-parallel-blocks must be non-negative")
    if not 0.0 <= args.event_rate <= 1.0:
        parser.error("--event-rate must be in [0, 1]")
    if args.target_activity is not None and any(
        not 0.0 <= target <= 1.0 for target in args.target_activity
    ):
        parser.error("--target-activity values must be in [0, 1]")
    if (
        args.calibration_min_amplitude < 0.0
        or args.calibration_max_amplitude <= args.calibration_min_amplitude
    ):
        parser.error("invalid calibration amplitude bounds")
    if args.calibration_iterations <= 0:
        parser.error("--calibration-iterations must be positive")
    if args.calibration_relative_tolerance < 0.0:
        parser.error("--calibration-relative-tolerance must be non-negative")
    args.dataset, args.weight_scale = resolve_dataset_defaults(
        args.dataset,
        args.weight_scale,
    )
    return args


def main() -> None:
    """Run all requested benchmark cases."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this comparison.")
    device = torch.device("cuda")

    dataset = args.dataset
    providers = tuple(
        args.providers
        if args.providers is not None
        else (FLYBRAIN_DEFAULT_PROVIDERS if dataset == "flybrain" else PROVIDERS)
    )
    matrix = None
    if dataset == "flybrain":
        matrix = load_flybrain_csr(
            args.connectome_root,
            weight_scale=args.weight_scale,
            device=device,
        )
    elif dataset == "mice_column_v1":
        matrix = load_mice_column_v1_csr(
            args.connectome_root,
            weight_scale=args.weight_scale,
            device=device,
        )

    graph_provider = TorchCUDAGraphProvider()
    direct_cusparse_provider = DirectCuSparseProvider()
    sota_cudagraph_provider = SotaCUDAGraphProvider()
    sota_eager_provider = SotaEagerProvider(
        selector_model=args.spmspv_selector_model,
        direct_selector_model=args.spmspv_direct_selector_model,
    )
    persistent_provider = PersistentProvider()
    external_provider_options = {
        "genn": {
            "optimize_code": args.genn_optimize_code,
        },
        "brian2cuda": {
            "sm_multiplier": args.brian2cuda_sm_multiplier,
            "parallel_blocks": args.brian2cuda_parallel_blocks,
            "calc_occupancy": args.brian2cuda_calc_occupancy,
            "extra_threshold_kernel": (args.brian2cuda_extra_threshold_kernel),
            "launch_bounds": args.brian2cuda_launch_bounds,
            "syn_launch_bounds": args.brian2cuda_syn_launch_bounds,
        },
    }
    all_rows: list[dict] = []
    audit_manifest = (
        json.loads(args.audit_manifest.read_text())
        if args.audit_manifest is not None
        else None
    )
    for t_steps in args.t_steps:
        if matrix is None:
            n_neuron = args.n_neuron
            fanout = args.fanout
        else:
            n_neuron = matrix.shape[0]
            fanout = int(round(matrix.indices.numel() / max(n_neuron, 1)))
        case = BenchCase(
            n_neuron=n_neuron,
            batch_size=1,
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
        case_matrix = matrix if matrix is not None else make_recurrent_csr(case, device)
        targets = args.target_activity or [None]
        for target_activity in targets:
            calibration_result = None
            calibrated_case = case
            if target_activity is not None:
                calibration_result = calibrate_input_amplitude(
                    case,
                    case_matrix,
                    device,
                    target_activity,
                    min_amplitude=args.calibration_min_amplitude,
                    max_amplitude=args.calibration_max_amplitude,
                    max_iterations=args.calibration_iterations,
                    relative_tolerance=(args.calibration_relative_tolerance),
                )
                calibrated_case = dataclass_replace(
                    case,
                    input_amplitude=calibration_result.input_amplitude,
                )
            print(
                f"=== dataset={dataset} N={case.n_neuron} T={t_steps} "
                f"B=1 edges={case_matrix.indices.numel()} "
                f"target_activity={target_activity} ==="
            )
            rows = bench_case(
                calibrated_case,
                device=device,
                dataset=dataset,
                matrix=case_matrix,
                providers=providers,
                graph_provider=graph_provider,
                direct_cusparse_provider=direct_cusparse_provider,
                sota_cudagraph_provider=sota_cudagraph_provider,
                sota_eager_provider=sota_eager_provider,
                persistent_provider=persistent_provider,
                external_build_root=args.external_build_root,
                external_timeout=args.external_timeout,
                external_provider_options=external_provider_options,
                warmup=args.warmup,
                repeat=args.repeat,
                eager_warmup=args.eager_warmup,
                eager_repeat=args.eager_repeat,
                check_correctness=not args.skip_correctness,
                continuous_state=args.continuous_state,
                requested_activity=target_activity,
                calibration=(
                    calibration_result.to_dict()
                    if calibration_result is not None
                    else None
                ),
                audit_manifest=audit_manifest,
            )
            for row in rows:
                print(
                    f"  {row['provider']:<28} {row['status']:<20} "
                    f"activity={row['measured_activity']:.4%} "
                    f"latency={row['latency_ms']:.4f} ms "
                    "vs_direct_graph="
                    f"{row['speedup_vs_cusparse_direct_cudagraph']:.3f}x "
                    f"vs_same_eager={row['speedup_vs_same_eager']:.3f}x"
                )
            all_rows.extend(rows)
            sota_eager_provider.close()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved CSV to {args.csv}")
    sota_cudagraph_provider.close()


if __name__ == "__main__":
    main()
