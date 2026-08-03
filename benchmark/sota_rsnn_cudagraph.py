"""CUDA Graph adapters for connectome_dataset sparse kernels.

The microbenchmark providers in ``connectome_dataset`` accept host operands and
often synchronize to return their own timing. RSNN graph capture instead needs
an allocation-free device-pointer launch on the current CUDA stream. This
module implements that contract for kernels whose C ABI exposes it and reports
an explicit incompatibility for providers that still use a host-controlled
pipeline.
"""

from __future__ import annotations

import ctypes
import gc
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from benchmark.benchmark_persistent_snn import BenchCase, RSNNResult
from benchmark.provider_common import (
    BenchmarkRunner,
    PreparedMetadata,
    PreparedOperator,
    inspect_tensor,
    tensor_bytes,
)


SOTA_CUDAGRAPH_PROVIDERS = (
    "vdha_cudagraph",
    "sputnik_cudagraph",
)
SOTA_EAGER_PROVIDERS = (
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
)
SOTA_PROVIDERS = SOTA_CUDAGRAPH_PROVIDERS + SOTA_EAGER_PROVIDERS

CUDA_SPMSPV_PROVIDERS = {
    "vdha_spmspv_eager": "vdha",
    "vdha_pipe_spmspv_eager": "vdha_pipe",
    "tilespmspv_eager": "tilespmspv",
    "sortspmspv_eager": "sortspmspv",
    "globalatomic_eager": "globalatomic",
    "blockatomic_eager": "blockatomic",
    "blocksort_eager": "blocksort",
    "naivespmspv_eager": "naivespmspv",
    "holaspmspv_eager": "holaspmspv",
}
SELECTOR_SPMSPV_PROVIDERS = {
    "adaptive_spmspv_eager": "cascade",
    "adaptive_direct_spmspv_eager": "direct",
}
SPMSPV_EAGER_PROVIDERS = tuple(
    (
        *CUDA_SPMSPV_PROVIDERS,
        "triton_spmspv_eager",
        "finch_spmspv_eager",
        *SELECTOR_SPMSPV_PROVIDERS,
    )
)


def _pointer(array: np.ndarray) -> ctypes.c_void_p:
    return array.ctypes.data_as(ctypes.c_void_p)


def _torch_pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _weight_arrays(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        weight.crow_indices().to(device="cpu", dtype=torch.int32).numpy(),
        weight.col_indices().to(device="cpu", dtype=torch.int32).numpy(),
        weight.values().to(device="cpu", dtype=torch.float32).numpy(),
    )


def _sputnik_matmul(
    weight: torch.Tensor, case: BenchCase
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[], None]]:
    try:
        import connectome_bench_sputnik as sputnik
    except ImportError as exc:
        raise OSError(
            "Sputnik provider is not installed; install "
            "examples/connectome_bench_sputnik and build its shared library"
        ) from exc

    lib = sputnik._load_lib()
    row, col, values = _weight_arrays(weight)
    lengths = np.diff(row)
    swizzle = np.argsort(-lengths).astype(np.int32)
    handle = lib.cbn_sputnik_prepare(
        weight.shape[0],
        weight.shape[1],
        case.batch_size,
        values.size,
        _pointer(row),
        _pointer(col),
        _pointer(values),
        _pointer(swizzle),
    )
    if not handle:
        raise RuntimeError("cbn_sputnik_prepare failed")

    output_nb = torch.empty(
        weight.shape[0],
        case.batch_size,
        device=weight.device,
        dtype=weight.dtype,
    )
    rhs_nb = None if case.batch_size == 1 else torch.empty_like(output_nb)

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        if case.batch_size == 1:
            rhs = spikes.view(weight.shape[1], 1)
        else:
            assert rhs_nb is not None
            rhs_nb.copy_(spikes.transpose(0, 1))
            rhs = rhs_nb
        error = lib.cbn_sputnik_compute_device(
            handle,
            _torch_pointer(rhs),
            _torch_pointer(output_nb),
            _stream_pointer(),
        )
        if error:
            raise RuntimeError(f"Sputnik launch failed (CUDA error {error})")
        return output_nb.transpose(0, 1)

    matmul._btorch_buffers = tuple(  # type: ignore[attr-defined]
        tensor for tensor in (output_nb, rhs_nb) if tensor is not None
    )
    matmul._btorch_transform_mode = (  # type: ignore[attr-defined]
        "zero_copy_view" if case.batch_size == 1 else "gpu_pack_copy"
    )

    return matmul, lambda: lib.cbn_sputnik_free(handle)


def _vdha_matmul(
    weight: torch.Tensor, case: BenchCase
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[], None]]:
    """Prepare VDHA's dynamic dense-input device API.

    The available sparse ``cbn_vdha_prepare`` ABI accepts one host spike list
    while creating the handle, so it cannot consume the changing spike set of
    a recurrent window. Until a device ``indices + count`` entry point exists,
    the dynamic dense-input API is the only semantically valid RSNN adapter.
    """

    if case.batch_size != 1:
        raise NotImplementedError("VDHA CUDA Graph adapter requires batch size 1")
    try:
        import connectome_bench_vdha as vdha
    except ImportError as exc:
        raise OSError(
            "VDHA provider is not installed; install "
            "examples/connectome_bench_vdha and build its shared library"
        ) from exc

    lib = vdha._load_lib("fp32")
    row, col, values = _weight_arrays(weight)
    # The post-by-pre CSR weight is also the CSC representation needed by
    # VDHA after transposing its interpretation: each CSR row would be a post
    # row, so form CSC explicitly on the host during untimed preparation.
    import scipy.sparse as sp

    matrix = sp.csr_matrix((values, col, row), shape=tuple(weight.shape)).tocsc()
    col_ptr = matrix.indptr.astype(np.int32)
    csc_row = matrix.indices.astype(np.int32)
    csc_values = matrix.data.astype(np.float32)
    handle = lib.cbn_vdha_prepare_dense(
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        _pointer(col_ptr),
        _pointer(csc_row),
        _pointer(csc_values),
    )
    if not handle:
        raise RuntimeError("cbn_vdha_prepare_dense failed")

    output = torch.empty(
        matrix.shape[0],
        device=weight.device,
        dtype=weight.dtype,
    )

    def launch(spikes: torch.Tensor, target: torch.Tensor) -> None:
        error = lib.cbn_vdha_compute_dense_device(
            handle,
            _torch_pointer(spikes.reshape(-1)),
            _torch_pointer(target),
            _stream_pointer(),
        )
        if error:
            raise RuntimeError(f"VDHA launch failed (CUDA error {error})")

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        launch(spikes, output)
        return output.unsqueeze(0)

    matmul._btorch_buffers = (output,)  # type: ignore[attr-defined]
    matmul._btorch_transform_mode = "none"  # type: ignore[attr-defined]
    matmul._btorch_launch = launch  # type: ignore[attr-defined]

    return matmul, lambda: lib.cbn_vdha_free(handle)


def prepare_matmul(
    provider: str,
    weight: torch.Tensor,
    case: BenchCase,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[], None]]:
    """Prepare a dynamic device-resident recurrent multiply."""

    if provider == "sputnik_cudagraph":
        return _sputnik_matmul(weight, case)
    if provider == "vdha_cudagraph":
        return _vdha_matmul(weight, case)
    raise ValueError(f"Unknown SOTA CUDA Graph provider: {provider}")


def prepare_sputnik_operator(
    weight: torch.Tensor,
    spike_trace_bn: torch.Tensor,
    case: BenchCase,
) -> PreparedOperator:
    """Prepare distinct native-NB and adapted-BN Sputnik traces."""

    if case.batch_size != 1:
        raise NotImplementedError("Sputnik operator audit currently requires B=1")
    try:
        import connectome_bench_sputnik as sputnik
    except ImportError as exc:
        raise OSError("Sputnik provider is not installed") from exc
    lib = sputnik._load_lib()
    row, col, values = _weight_arrays(weight)
    swizzle = np.argsort(-np.diff(row)).astype(np.int32)
    handle = lib.cbn_sputnik_prepare(
        weight.shape[0],
        weight.shape[1],
        1,
        values.size,
        _pointer(row),
        _pointer(col),
        _pointer(values),
        _pointer(swizzle),
    )
    if not handle:
        raise RuntimeError("cbn_sputnik_prepare failed")

    trace_nb = spike_trace_bn.view(case.t_steps, case.n_neuron, 1)
    native_output = torch.empty_like(trace_nb)
    adapted_output = torch.empty_like(spike_trace_bn)

    def launch(rhs: torch.Tensor, output: torch.Tensor) -> None:
        error = lib.cbn_sputnik_compute_device(
            handle,
            _torch_pointer(rhs),
            _torch_pointer(output),
            _stream_pointer(),
        )
        if error:
            raise RuntimeError(f"Sputnik launch failed (CUDA error {error})")

    def native_run() -> None:
        for timestep in range(case.t_steps):
            launch(trace_nb[timestep], native_output[timestep])

    def adapted_run() -> None:
        for timestep in range(case.t_steps):
            rhs_nb = spike_trace_bn[timestep].view(case.n_neuron, 1)
            output_nb = adapted_output[timestep].view(case.n_neuron, 1)
            launch(rhs_nb, output_nb)

    audit = inspect_tensor(spike_trace_bn)
    buffers = [native_output, adapted_output]
    metadata = PreparedMetadata(
        logical_shape=tuple(spike_trace_bn.shape),
        physical_shape=tuple(spike_trace_bn.shape),
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
                spike_trace_bn,
                *buffers,
            ]
        ),
        padding_ratio=1.0,
        execution_class="device_adapted",
        timing_scope="gpu_execution",
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
        release_fn=lambda: lib.cbn_sputnik_free(handle),
    )


def prepare_vdha_dense_operator(
    weight: torch.Tensor,
    spike_trace_bn: torch.Tensor,
    case: BenchCase,
) -> PreparedOperator:
    """Prepare VDHA dense input without inventing a duplicate native path."""

    matmul, release = _vdha_matmul(weight, case)
    adapted_output = torch.empty_like(spike_trace_bn)

    def adapted_run() -> None:
        launch = getattr(matmul, "_btorch_launch")
        for timestep in range(case.t_steps):
            launch(
                spike_trace_bn[timestep],
                adapted_output[timestep].view(case.n_neuron),
            )

    audit = inspect_tensor(spike_trace_bn)
    adapter_buffers = list(getattr(matmul, "_btorch_buffers", ()))
    metadata = PreparedMetadata(
        logical_shape=tuple(spike_trace_bn.shape),
        physical_shape=tuple(spike_trace_bn.shape),
        input_layout="BN",
        operator_layout="dense_vector",
        index_dtype=str(weight.crow_indices().dtype),
        value_dtype=str(weight.values().dtype),
        compute_dtype=str(weight.values().dtype),
        workspace_bytes=tensor_bytes(adapter_buffers),
        persistent_bytes=tensor_bytes(
            [
                weight.crow_indices(),
                weight.col_indices(),
                weight.values(),
                spike_trace_bn,
                adapted_output,
                *adapter_buffers,
            ]
        ),
        padding_ratio=1.0,
        execution_class="device_adapted",
        timing_scope="gpu_execution",
        layout_transform_mode="none",
        layout_transform_in_timing=False,
        spike_representation="dense",
        input_contiguous=bool(audit["contiguous"]),
        input_address_mod=int(audit["address_alignment"]),
        native_available=False,
        native_status="not_distinct_from_adapter",
    )
    return PreparedOperator(
        native_run=None,
        adapted_run=adapted_run,
        transform_run=None,
        native_output=None,
        adapted_output=adapted_output,
        metadata=metadata,
        release_fn=release,
    )


def rsnn_forward(
    x_seq: torch.Tensor,
    matmul: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    psc0: torch.Tensor,
    case: BenchCase,
) -> RSNNResult:
    """Run common RSNN dynamics with a supplied recurrent operator."""

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
        psc = psc * decay + matmul(z)
        spikes.append(z)
    return RSNNResult(spikes=torch.stack(spikes), v=v, psc=psc)


class SotaCUDAGraphProvider:
    """Capture full RSNN windows using connectome_dataset SOTA kernels."""

    def __init__(self) -> None:
        self._runners: dict[tuple, object] = {}
        self._release: list[Callable[[], None]] = []

    def fixed_runner(
        self,
        provider: str,
        x_seq: torch.Tensor,
        weight: torch.Tensor,
        case: BenchCase,
    ):
        key = (provider, id(weight), case.t_steps, case.batch_size, x_seq.shape)
        cached = self._runners.get(key)
        if cached is not None:
            return cached

        matmul, release = prepare_matmul(provider, weight, case)
        self._release.append(release)
        static_x = x_seq.clone()
        static_v0 = torch.zeros(
            case.batch_size,
            case.n_neuron,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
        static_psc0 = torch.zeros_like(static_v0)

        def execute() -> RSNNResult:
            return rsnn_forward(static_x, matmul, static_v0, static_psc0, case)

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

        adapter_buffers = tuple(getattr(matmul, "_btorch_buffers", ()))
        input_audit = inspect_tensor(static_x)
        transform_mode = str(getattr(matmul, "_btorch_transform_mode", "unknown"))
        metadata = PreparedMetadata(
            logical_shape=tuple(static_x.shape),
            physical_shape=tuple(static_x.shape),
            input_layout="BN",
            operator_layout=(
                "NB" if provider == "sputnik_cudagraph" else "dense_vector"
            ),
            index_dtype=str(weight.crow_indices().dtype),
            value_dtype=str(weight.values().dtype),
            compute_dtype=str(weight.values().dtype),
            workspace_bytes=tensor_bytes(list(adapter_buffers)),
            persistent_bytes=tensor_bytes(
                [
                    weight.crow_indices(),
                    weight.col_indices(),
                    weight.values(),
                    static_x,
                    static_v0,
                    static_psc0,
                    *adapter_buffers,
                ]
            ),
            padding_ratio=1.0,
            execution_class="device_adapted",
            timing_scope="gpu_execution",
            fusion_level="sparse_only",
            graph_capturable=True,
            layout_transform_mode=transform_mode,
            layout_transform_in_timing=(
                provider == "sputnik_cudagraph" and case.batch_size != 1
            ),
            spike_representation=(
                "dense" if provider == "vdha_cudagraph" else "dense_matrix"
            ),
            input_contiguous=bool(input_audit["contiguous"]),
            input_address_mod=int(input_audit["address_alignment"]),
        )
        runner = BenchmarkRunner(
            run_fn=replay,
            reset_fn=lambda: None,
            metadata=metadata,
        )
        self._runners[key] = runner
        return runner

    def close(self) -> None:
        """Release native handles after captured graphs are discarded."""

        self._runners.clear()
        while self._release:
            self._release.pop()()


def _scipy_weight(weight: torch.Tensor):
    import scipy.sparse as sp

    row, col, values = _weight_arrays(weight)
    return sp.csr_matrix((values, col, row), shape=tuple(weight.shape))


def _device_result(result, device: torch.device) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result.to(device=device, dtype=torch.float32)
    if hasattr(result, "get"):
        result = result.get()
    return torch.as_tensor(np.asarray(result), device=device, dtype=torch.float32)


def _require_batch_one(provider: str, case: BenchCase) -> None:
    """Reject matrix-vector providers for batched recurrent operands."""

    if case.batch_size != 1:
        raise NotImplementedError(f"{provider} requires batch size 1")


def _host_sparse_operand(spikes: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert one dense device spike vector to host COO arrays."""

    values = spikes.detach().to(device="cpu", dtype=torch.float32).numpy()
    values = values.reshape(-1)
    indices = np.flatnonzero(values).astype(np.int32, copy=False)
    return indices, np.ascontiguousarray(values[indices], dtype=np.float32)


def _set_matmul_attributes(
    matmul: Callable[[torch.Tensor], torch.Tensor],
    *,
    release: Callable[[], None] = lambda: None,
    operator_layout: str = "public_wrapper",
    spike_representation: str = "dense",
    uses_d2h: bool = True,
    uses_h2d: bool = True,
    uses_internal_sync: bool = True,
    buffers: tuple[torch.Tensor, ...] = (),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Attach metadata consumed by :class:`SotaEagerProvider`."""

    matmul._btorch_release = release  # type: ignore[attr-defined]
    matmul._btorch_operator_layout = operator_layout  # type: ignore[attr-defined]
    matmul._btorch_spike_representation = spike_representation  # type: ignore[attr-defined]
    matmul._btorch_uses_d2h = uses_d2h  # type: ignore[attr-defined]
    matmul._btorch_uses_h2d = uses_h2d  # type: ignore[attr-defined]
    matmul._btorch_uses_internal_sync = uses_internal_sync  # type: ignore[attr-defined]
    matmul._btorch_buffers = buffers  # type: ignore[attr-defined]
    return matmul


def _prepare_cuda_spmspv_matmul(
    provider: str,
    matrix,
    case: BenchCase,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare one split-ABI CUDA SpMSpV recurrent operator."""

    _require_batch_one(provider, case)
    from connectome_dataset.benchmarks.cuda.kernels import (
        KERNELS_BY_NAME,
        PreparedKernel,
        supports_preprocess,
    )

    kernel_name = CUDA_SPMSPV_PROVIDERS[provider]
    spec = KERNELS_BY_NAME[kernel_name]
    if not supports_preprocess(spec, "fp32"):
        raise OSError(
            f"{kernel_name} split preprocess ABI is unavailable; rebuild "
            "its shared library"
        )
    prepared = PreparedKernel(spec, matrix, precision="fp32")

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        indices, values = _host_sparse_operand(spikes)
        result = prepared.step(indices, values)
        return _device_result(np.asarray(result), spikes.device).unsqueeze(0)

    return _set_matmul_attributes(
        matmul,
        release=prepared.free,
        operator_layout=f"spmspv_{spec.matrix_format}",
        spike_representation="host_sparse_indices",
    )


def _prepare_triton_spmspv_matmul(
    matrix,
    case: BenchCase,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare Triton push SpMSpV with a dynamic device spike operand."""

    provider = "triton_spmspv_eager"
    _require_batch_one(provider, case)
    from connectome_dataset.benchmarks.triton.spmspv import (
        available,
        build_kernel,
    )

    if not available():
        raise OSError("Triton SpMSpV dependencies are unavailable")
    kernel = build_kernel()
    csc = matrix.tocsc().astype(np.float32)
    device = torch.device("cuda")
    col_ptr = torch.as_tensor(csc.indptr.astype(np.int32), device=device)
    row_idx = torch.as_tensor(csc.indices.astype(np.int32), device=device)
    values = torch.as_tensor(csc.data, device=device)
    output = torch.empty(csc.shape[0], device=device, dtype=torch.float32)

    # Trigger Triton JIT while provider construction is timed as preprocessing.
    dummy_index = torch.zeros(1, device=device, dtype=torch.int32)
    dummy_value = torch.zeros(1, device=device, dtype=torch.float32)
    output.zero_()
    kernel[(1,)](
        col_ptr,
        row_idx,
        values,
        dummy_index,
        dummy_value,
        output,
        1,
        BLOCK=128,
    )

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        flat = spikes.reshape(-1)
        active = torch.nonzero(flat, as_tuple=False).reshape(-1).to(torch.int32)
        active_values = flat[active.to(torch.long)]
        output.zero_()
        n_active = active.numel()
        if n_active:
            kernel[(n_active,)](
                col_ptr,
                row_idx,
                values,
                active,
                active_values,
                output,
                n_active,
                BLOCK=128,
            )
        return output.unsqueeze(0)

    return _set_matmul_attributes(
        matmul,
        operator_layout="spmspv_csc_triton_push",
        spike_representation="dynamic_device_indices",
        uses_d2h=False,
        uses_h2d=False,
        uses_internal_sync=False,
        buffers=(col_ptr, row_idx, values, output, dummy_index, dummy_value),
    )


def _prepare_finch_spmspv_matmul(
    matrix,
    case: BenchCase,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare a reusable Finch matrix and rebuild only sparse x each step."""

    provider = "finch_spmspv_eager"
    _require_batch_one(provider, case)
    julia_exe = Path(
        os.environ.get("PYTHON_JULIAPKG_EXE", "/snap/julia/current/bin/julia")
    )
    if not julia_exe.is_file():
        raise OSError(
            "Finch requires a Julia executable; set PYTHON_JULIAPKG_EXE "
            f"to a valid path (not {julia_exe})"
        )
    from connectome_dataset.benchmarks.julia import finch

    jl = finch._julia()
    csc = matrix.tocsc().astype(np.float32)
    indptr = np.ascontiguousarray(csc.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csc.indices, dtype=np.int64)
    values = np.ascontiguousarray(csc.data, dtype=np.float32)
    a_tensor = jl.finch_build_matrix(
        csc.shape[0], csc.shape[1], indptr, indices, values, jl.Float32
    )
    y_tensor = jl.finch_alloc_y(csc.shape[0], jl.Float32)

    # Compile the Finch loop once during preprocessing, including the empty-x case.
    empty_i = np.empty(0, dtype=np.int64)
    empty_v = np.empty(0, dtype=np.float32)
    empty_x = jl.finch_build_vector(csc.shape[1], empty_i, empty_v, jl.Float32)
    jl.finch_spmspv_b(y_tensor, a_tensor, empty_x)

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        x_indices, x_values = _host_sparse_operand(spikes)
        x_tensor = jl.finch_build_vector(
            csc.shape[1],
            x_indices.astype(np.int64, copy=False),
            x_values,
            jl.Float32,
        )
        result = jl.finch_spmspv_b(y_tensor, a_tensor, x_tensor)
        output = np.asarray(jl.Array(result), dtype=np.float32)
        return _device_result(output, spikes.device).unsqueeze(0)

    return _set_matmul_attributes(
        matmul,
        operator_layout="spmspv_csc_finch",
        spike_representation="host_sparse_indices",
    )


def _prepare_adaptive_spmspv_matmul(
    provider: str,
    matrix,
    case: BenchCase,
    model_path: Path | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare a per-timestep adaptive selector with lazy CUDA handles."""

    _require_batch_one(provider, case)
    if model_path is None:
        raise OSError(f"{provider} requires an explicit selector model path")
    if not model_path.exists():
        raise OSError(f"selector model not found: {model_path}")

    from connectome_dataset.benchmarks.cuda import adaptive
    from connectome_dataset.benchmarks.cuda.kernels import (
        KERNELS_BY_NAME,
        PreparedKernel,
        supports_preprocess,
    )

    kind = SELECTOR_SPMSPV_PROVIDERS[provider]
    selector = (
        adaptive.AdaptiveSelector.load(model_path)
        if kind == "cascade"
        else adaptive.DirectKernelSelector.load(model_path)
    )
    csc = matrix.tocsc().astype(np.float32)
    matrix_features = adaptive.matrix_features(csc)
    prepared: dict[str, object] = {}
    selected: set[str] = set()
    frozen = False

    def get_kernel(kernel_name: str):
        nonlocal prepared
        if kernel_name in prepared:
            return prepared[kernel_name]
        if frozen:
            raise RuntimeError(
                f"selector chose unprimed kernel {kernel_name!r} during timed execution"
            )
        spec = KERNELS_BY_NAME[kernel_name]
        if not supports_preprocess(spec, "fp32"):
            raise OSError(f"selector chose unavailable kernel {kernel_name!r}")
        prepared[kernel_name] = PreparedKernel(spec, csc, precision="fp32")
        return prepared[kernel_name]

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        x_indices, x_values = _host_sparse_operand(spikes)
        features = {
            **matrix_features,
            **adaptive.vector_features(csc, x_indices),
        }
        kernel_name = selector.predict(features)
        selected.add(kernel_name)
        kernel = get_kernel(kernel_name)
        result = kernel.step(x_indices, x_values)
        return _device_result(np.asarray(result), spikes.device).unsqueeze(0)

    def freeze() -> None:
        nonlocal frozen
        frozen = True

    def release() -> None:
        for kernel in prepared.values():
            kernel.free()
        prepared.clear()

    matmul._btorch_freeze_after_prime = freeze  # type: ignore[attr-defined]
    matmul._btorch_selected_kernels = selected  # type: ignore[attr-defined]
    matmul._btorch_selector_model = str(model_path)  # type: ignore[attr-defined]
    return _set_matmul_attributes(
        matmul,
        release=release,
        operator_layout=f"adaptive_spmspv_{kind}",
        spike_representation="host_sparse_indices",
    )


def prepare_eager_matmul(
    provider: str,
    weight: torch.Tensor,
    case: BenchCase,
    *,
    selector_model_paths: dict[str, Path | None] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare a correctness-first host-controlled fallback.

    Provider preprocessing happens once while this function creates the
    callable. Host synchronization and operand conversion remain part of every
    timed invocation. These paths are used only when a provider does not expose
    an allocation-free raw CUDA launch.
    """

    if provider == "torch_csr_host_e2e":

        def torch_csr_matmul(spikes: torch.Tensor) -> torch.Tensor:
            return torch.sparse.mm(weight, spikes.transpose(0, 1)).transpose(0, 1)

        return _set_matmul_attributes(
            torch_csr_matmul,
            operator_layout="csr_post_by_pre",
            spike_representation="dense",
            uses_d2h=False,
            uses_h2d=False,
            uses_internal_sync=False,
        )

    matrix = _scipy_weight(weight)
    if provider in CUDA_SPMSPV_PROVIDERS:
        return _prepare_cuda_spmspv_matmul(provider, matrix, case)
    if provider == "triton_spmspv_eager":
        return _prepare_triton_spmspv_matmul(matrix, case)
    if provider == "finch_spmspv_eager":
        return _prepare_finch_spmspv_matmul(matrix, case)
    if provider in SELECTOR_SPMSPV_PROVIDERS:
        paths = selector_model_paths or {}
        return _prepare_adaptive_spmspv_matmul(
            provider, matrix, case, paths.get(provider)
        )
    if provider == "mh_spgemm_eager":
        if case.batch_size < 32:
            raise NotImplementedError(
                "MH-SpGEMM eager fallback requires batch size >= 32"
            )
        try:
            from connectome_dataset.benchmarks.torch.btorch import delivery
        except ImportError as exc:
            raise OSError("MH-SpGEMM RSNN delivery adapter is unavailable") from exc

        original_weight = matrix.transpose().tocsr()

        def mh_matmul(spikes: torch.Tensor) -> torch.Tensor:
            spike_array = spikes.detach().cpu().numpy()
            _, output, _ = delivery.mh_deliver(
                original_weight, spike_array, timesteps=1
            )
            return _device_result(output, spikes.device)

        return mh_matmul

    if provider == "dtc_spmm_eager":
        if case.batch_size % 16:
            raise NotImplementedError(
                "DTC-SpMM eager fallback requires batch size divisible by 16"
            )
        from connectome_dataset.benchmarks.torch.dtc import spmm

        run = spmm.make_dynamic_fn(
            matrix,
            batch_size=case.batch_size,
        )

        def dtc_matmul(spikes: torch.Tensor) -> torch.Tensor:
            rhs = spikes.transpose(0, 1).contiguous()
            output, _ = run(rhs)
            return _device_result(output, spikes.device).transpose(0, 1)

        return dtc_matmul

    if provider == "flashsparse_eager":
        if case.batch_size % 8:
            raise NotImplementedError(
                "FlashSparse eager fallback requires batch size divisible by 8"
            )
        from connectome_dataset.benchmarks.torch.flashsparse import spmm

        run = spmm.make_dynamic_fn(
            matrix,
            batch_size=case.batch_size,
            epochs=1,
        )

        def flashsparse_matmul(spikes: torch.Tensor) -> torch.Tensor:
            rhs = spikes.transpose(0, 1).contiguous()
            output, _ = run(rhs)
            return _device_result(output, spikes.device).transpose(0, 1)

        return flashsparse_matmul

    raise ValueError(f"Unknown SOTA eager provider: {provider}")


class SotaEagerProvider:
    """Run host-controlled fallbacks for non-capturable SOTA providers."""

    def __init__(
        self,
        *,
        selector_model: Path | None = None,
        direct_selector_model: Path | None = None,
    ) -> None:
        self._runners: dict[tuple, object] = {}
        self._release: list[Callable[[], None]] = []
        self._selector_model_paths = {
            "adaptive_spmspv_eager": selector_model,
            "adaptive_direct_spmspv_eager": direct_selector_model,
        }

    def fixed_runner(
        self,
        provider: str,
        x_seq: torch.Tensor,
        weight: torch.Tensor,
        case: BenchCase,
    ):
        key = (provider, id(weight), case.t_steps, case.batch_size, x_seq.shape)
        cached = self._runners.get(key)
        if cached is not None:
            return cached

        # One full-window eager runner is consumed before the next provider is
        # prepared. Retaining prior native handles only duplicates the graph and
        # worst-case workspace, which is prohibitive for whole-connectome cases.
        self._drop_cached()

        matmul = prepare_eager_matmul(
            provider,
            weight,
            case,
            selector_model_paths=self._selector_model_paths,
        )
        release = getattr(matmul, "_btorch_release", lambda: None)
        self._release.append(release)
        v0 = torch.zeros(
            case.batch_size,
            case.n_neuron,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
        psc0 = torch.zeros_like(v0)

        def run() -> RSNNResult:
            return rsnn_forward(x_seq, matmul, v0, psc0, case)

        freeze_after_prime = getattr(matmul, "_btorch_freeze_after_prime", None)
        if freeze_after_prime is not None:
            run()
            torch.cuda.synchronize()
            freeze_after_prime()

        input_audit = inspect_tensor(x_seq)
        metadata = PreparedMetadata(
            logical_shape=tuple(x_seq.shape),
            physical_shape=tuple(x_seq.shape),
            input_layout="BN",
            operator_layout=str(
                getattr(matmul, "_btorch_operator_layout", "public_wrapper")
            ),
            index_dtype=str(weight.crow_indices().dtype),
            value_dtype=str(weight.values().dtype),
            compute_dtype=str(weight.values().dtype),
            workspace_bytes=tensor_bytes(list(getattr(matmul, "_btorch_buffers", ()))),
            persistent_bytes=tensor_bytes(
                [
                    weight.crow_indices(),
                    weight.col_indices(),
                    weight.values(),
                    x_seq,
                    v0,
                    psc0,
                ]
            ),
            padding_ratio=1.0,
            execution_class="host_wrapper",
            timing_scope="public_wrapper_e2e",
            fusion_level="sparse_only",
            uses_host_control=True,
            uses_internal_sync=bool(
                getattr(matmul, "_btorch_uses_internal_sync", True)
            ),
            uses_d2h=bool(getattr(matmul, "_btorch_uses_d2h", True)),
            uses_h2d=bool(getattr(matmul, "_btorch_uses_h2d", True)),
            allocates_during_run=True,
            graph_capturable=False,
            layout_transform_mode="host_wrapper_conversion",
            layout_transform_in_timing=True,
            spike_representation=str(
                getattr(matmul, "_btorch_spike_representation", "dense")
            ),
            input_contiguous=bool(input_audit["contiguous"]),
            input_address_mod=int(input_audit["address_alignment"]),
            selected_kernels=";".join(
                sorted(getattr(matmul, "_btorch_selected_kernels", ()))
            ),
            selector_model=str(getattr(matmul, "_btorch_selector_model", "")),
        )
        runner = BenchmarkRunner(
            run_fn=run,
            reset_fn=lambda: None,
            metadata=metadata,
        )
        self._runners[key] = runner
        return runner

    def close(self) -> None:
        """Drop plans and release cached CUDA allocations between workloads."""

        self._drop_cached()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _drop_cached(self) -> None:
        """Release the most recently cached eager runner and native handles."""

        self._runners.clear()
        while self._release:
            self._release.pop()()
