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
import math
from collections.abc import Callable

import numpy as np
import torch

from benchmark.benchmark_persistent_snn import BenchCase, RSNNResult


SOTA_CUDAGRAPH_PROVIDERS = (
    "vdha_cudagraph",
    "mh_spgemm_cudagraph",
    "sputnik_cudagraph",
    "dtc_spmm_cudagraph",
    "flashsparse_cudagraph",
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

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        rhs = spikes.transpose(0, 1).contiguous()
        output = torch.empty(
            weight.shape[0],
            case.batch_size,
            device=spikes.device,
            dtype=spikes.dtype,
        )
        error = lib.cbn_sputnik_compute_device(
            handle,
            _torch_pointer(rhs),
            _torch_pointer(output),
            _stream_pointer(),
        )
        if error:
            raise RuntimeError(f"Sputnik launch failed (CUDA error {error})")
        return output.transpose(0, 1)

    return matmul, lambda: lib.cbn_sputnik_free(handle)


def _vdha_matmul(
    weight: torch.Tensor, case: BenchCase
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[], None]]:
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

    matrix = sp.csr_matrix(
        (values, col, row), shape=tuple(weight.shape)
    ).tocsc()
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

    def matmul(spikes: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            matrix.shape[0], device=spikes.device, dtype=spikes.dtype
        )
        error = lib.cbn_vdha_compute_dense_device(
            handle,
            _torch_pointer(spikes.reshape(-1)),
            _torch_pointer(output),
            _stream_pointer(),
        )
        if error:
            raise RuntimeError(f"VDHA launch failed (CUDA error {error})")
        return output.unsqueeze(0)

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
    limitations = {
        "mh_spgemm_cudagraph": (
            "MH-SpGEMM has data-dependent sparse output allocation and host D2H "
            "conversion in cbn_mh_compute_timed"
        ),
        "dtc_spmm_cudagraph": (
            "DTC-SpMM's patched entry point creates/times its output internally "
            "and synchronizes while returning the timing tensor"
        ),
        "flashsparse_cudagraph": (
            "FlashSparse's public entry point copies a CPU RHS and synchronizes "
            "to return internally averaged CUDA-event timing"
        ),
    }
    if provider in limitations:
        raise NotImplementedError(
            f"{limitations[provider]}; its current connectome_dataset wrapper is "
            "not CUDA Graph capturable"
        )
    raise ValueError(f"Unknown SOTA CUDA Graph provider: {provider}")


def rsnn_forward(
    x_seq: torch.Tensor,
    matmul: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    psc0: torch.Tensor,
    case: BenchCase,
) -> RSNNResult:
    """Run common RSNN dynamics with a graph-capturable recurrent operator."""

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
            return rsnn_forward(
                static_x, matmul, static_v0, static_psc0, case
            )

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

    def close(self) -> None:
        """Release native handles after captured graphs are discarded."""

        self._runners.clear()
        while self._release:
            self._release.pop()()

