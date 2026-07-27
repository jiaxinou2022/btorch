"""Out-of-tree VDHA SpMSpV provider (Li et al., PPoPP '26), reached via the registry.

VDHA computes ``y = A·x`` with a sparse operand vector using vector-driven hash
aggregation — the event-driven formulation that also underlies SNN spike delivery. This
package ctypes-loads ``libconnectome_vdha.so`` (built by ``scripts/build_vdha.sh``; point
``CONNECTOME_VDHA_LIB`` at it, else the repo's ``cpp/build-vdha`` is tried). The reordered
segment work-list is built once in prepare() (it depends only on x); the benchmarked
callable runs the device aggregation and returns VDHA's own CUDA-event kernel time.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from connectome_dataset.benchmarks.providers import SpMSpVProvider

_LIBS: dict[str, ctypes.CDLL] = {}
# Internal averaging iterations for the self-timed kernel (matches the FlashSparse/DTC
# pattern: the callable reports a stable mean CUDA-event time, so the harness runs it once).
_ITERS = 50

# One shared lib per matrix-value type; the numpy dtype of csc_val must match the lib's.
_DTYPES = {"fp32": np.float32, "fp16": np.float16}


def _lib_path_for(precision: str) -> str | None:
    sfx = precision  # fp32 / fp16
    candidates = []
    env = os.environ.get(f"CONNECTOME_VDHA_LIB_{sfx.upper()}") or (
        os.environ.get("CONNECTOME_VDHA_LIB") if precision == "fp32" else None
    )
    if env:
        candidates.append(env)
    roots = list(Path(__file__).resolve().parents) + list(Path.cwd().resolve().parents) + [Path.cwd()]
    for base in roots:
        candidates.append(base / "cpp" / "build-vdha" / f"libconnectome_vdha_{sfx}.so")
    return next((str(c) for c in candidates if Path(c).exists()), None)


def _load_lib(precision: str):
    if precision in _LIBS:
        return _LIBS[precision]
    lib_path = _lib_path_for(precision)
    if lib_path is None:
        raise OSError(f"libconnectome_vdha_{precision}.so not found; build with scripts/build_vdha.sh")
    lib = ctypes.CDLL(lib_path)
    lib.cbn_vdha_prepare.restype = ctypes.c_void_p
    lib.cbn_vdha_prepare.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p] * 2
    lib.cbn_vdha_compute_timed.restype = ctypes.c_int
    lib.cbn_vdha_compute_timed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.cbn_vdha_prepare_dense.restype = ctypes.c_void_p
    lib.cbn_vdha_prepare_dense.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cbn_vdha_compute_dense_device.restype = ctypes.c_int
    lib.cbn_vdha_compute_dense_device.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cbn_vdha_copy_out.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cbn_vdha_free.argtypes = [ctypes.c_void_p]
    _LIBS[precision] = lib
    return lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class _Result:
    """Lazily copies the device result vector to host — only when the oracle asks."""

    def __init__(self, lib, handle, n: int):
        self._lib, self._h, self._n = lib, handle, n

    def __array__(self, dtype=None):
        y = np.empty(self._n, dtype=np.float32)
        self._lib.cbn_vdha_copy_out(self._h, _ptr(y))
        return y.astype(dtype) if dtype is not None else y


def _make_fn(matrix, *, variant, device, precision="fp32", x_indices=None, x_values=None):
    lib = _load_lib(precision)
    # scipy.sparse has no float16 dtype: keep the matrix float32, cast only the raw value
    # array to the loaded lib's value type (numpy supports float16 arrays).
    a = matrix.tocsc().astype(np.float32)  # y = A·x needs A's columns
    n, num_cols = a.shape
    nnz = a.nnz
    col_ptr = a.indptr.astype(np.int32)
    csc_row = a.indices.astype(np.int32)
    csc_val = a.data.astype(_DTYPES[precision])  # value type matches the loaded lib
    x_idx = np.ascontiguousarray(x_indices, dtype=np.int32)
    x_val = np.ascontiguousarray(x_values, dtype=np.float32)

    handle = lib.cbn_vdha_prepare(
        n, num_cols, nnz, _ptr(col_ptr), _ptr(csc_row), _ptr(csc_val),
        x_idx.size, _ptr(x_idx), _ptr(x_val),
    )
    if not handle:
        raise RuntimeError("cbn_vdha_prepare failed")

    def run():
        ms = ctypes.c_double(0.0)
        rc = lib.cbn_vdha_compute_timed(handle, _ITERS, ctypes.byref(ms))
        if rc != 0:
            raise RuntimeError(f"vdha compute failed (cuda err {rc})")
        return _Result(lib, handle, n), ms.value

    return run


def get_providers() -> list[SpMSpVProvider]:
    return [
        SpMSpVProvider(
            framework="cuda",  # standalone CUDA kernel reached via ctypes
            provider="vdha",
            make_fn=_make_fn,
            variants=["csc"],
            self_timed=True,  # VDHA returns its own CUDA-event kernel time
            supports=lambda variant, precision: precision in _DTYPES and _lib_path_for(precision) is not None,
        )
    ]
