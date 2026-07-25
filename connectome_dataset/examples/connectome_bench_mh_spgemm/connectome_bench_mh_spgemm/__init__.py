"""Out-of-tree MH-SpGEMM SpGEMM provider (Yang et al.), reached via the registry.

MH-SpGEMM computes ``C = A·Aᵀ`` in fp64 using masking + hashing cooperative optimization.
This package ctypes-loads ``libconnectome_mh_spgemm.so`` (built by
``scripts/build_mh_spgemm.sh``; point ``CONNECTOME_MH_SPGEMM_LIB`` at it, else the repo's
``cpp/build-mh_spgemm`` is tried). A and B = Aᵀ are uploaded once in prepare(); the
benchmarked callable runs the pipeline and returns MH-SpGEMM's own phase-summed runtime.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from connectome_dataset.benchmarks.providers import SpgemmProvider

_LIBS: dict[str, ctypes.CDLL] = {}
_ITERS = 10  # internal averaging iterations for the self-timed pipeline

# MH-SpGEMM builds one shared lib per value type; the numpy dtype must match the lib's.
_DTYPES = {"fp32": np.float32, "fp16": np.float16, "fp64": np.float64}
_SUFFIX = {"fp32": "fp32", "fp16": "fp16", "fp64": "fp64"}


def _lib_path_for(precision: str) -> str | None:
    sfx = _SUFFIX[precision]
    candidates = []
    env = os.environ.get(f"CONNECTOME_MH_SPGEMM_LIB_{sfx.upper()}") or (
        os.environ.get("CONNECTOME_MH_SPGEMM_LIB") if precision == "fp32" else None
    )
    if env:
        candidates.append(env)
    roots = list(Path(__file__).resolve().parents) + list(Path.cwd().resolve().parents) + [Path.cwd()]
    for base in roots:
        candidates.append(base / "cpp" / "build-mh_spgemm" / f"libconnectome_mh_spgemm_{sfx}.so")
    return next((str(c) for c in candidates if Path(c).exists()), None)


def _load_lib(precision: str):
    if precision in _LIBS:
        return _LIBS[precision]
    lib_path = _lib_path_for(precision)
    if lib_path is None:
        raise OSError(f"libconnectome_mh_spgemm_{_SUFFIX[precision]}.so not found; build with scripts/build_mh_spgemm.sh")
    lib = ctypes.CDLL(lib_path)
    lib.cbn_mh_prepare.restype = ctypes.c_void_p
    lib.cbn_mh_prepare.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3
    lib.cbn_mh_compute_timed.restype = ctypes.c_int
    lib.cbn_mh_compute_timed.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    ]
    lib.cbn_mh_result_nnz.restype = ctypes.c_int
    lib.cbn_mh_result_nnz.argtypes = [ctypes.c_void_p]
    lib.cbn_mh_copy_out.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
    lib.cbn_mh_free.argtypes = [ctypes.c_void_p]
    _LIBS[precision] = lib
    return lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class _Result:
    """Lazily copies the device result C to a host SciPy CSR — only when the oracle asks."""

    def __init__(self, lib, handle, m: int, dtype):
        self._lib, self._h, self._m, self._dtype = lib, handle, m, dtype

    def get_csr(self) -> sp.csr_matrix:
        nnz = self._lib.cbn_mh_result_nnz(self._h)
        indptr = np.empty(self._m + 1, dtype=np.int32)
        indices = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=self._dtype)  # value type matches the loaded lib
        self._lib.cbn_mh_copy_out(self._h, _ptr(indptr), _ptr(indices), _ptr(data))
        return sp.csr_matrix((data.astype(np.float64), indices, indptr), shape=(self._m, self._m))


def _make_fn(matrix, *, variant, device, precision="fp32"):
    lib = _load_lib(precision)
    dtype = _DTYPES[precision]
    # scipy.sparse has no float16 dtype, so keep the matrix in float32 and cast only the
    # raw value array to the C-ABI's value type (numpy supports float16 arrays fine).
    a = matrix.tocsr().astype(np.float32)
    a.sort_indices()  # MH-SpGEMM's mask kernels require ascending column indices
    m, k = a.shape
    a_ptr = a.indptr.astype(np.int32)
    a_col = a.indices.astype(np.int32)
    a_val = a.data.astype(dtype)
    handle = lib.cbn_mh_prepare(m, k, a.nnz, _ptr(a_ptr), _ptr(a_col), _ptr(a_val))
    if not handle:
        raise RuntimeError("cbn_mh_prepare failed")

    def run():
        ms = ctypes.c_double(0.0)
        nnz = ctypes.c_int(0)
        rc = lib.cbn_mh_compute_timed(handle, _ITERS, ctypes.byref(ms), ctypes.byref(nnz))
        if rc != 0:
            raise RuntimeError(f"mh-spgemm compute failed (rc {rc})")
        return _Result(lib, handle, m, dtype), ms.value

    return run


# The upstream MH-SpGEMM kernel is fp64-centric: its shared-memory hash tables and atomics
# assume a wide VALUE_TYPE, and the __half build faults (aborts the process) on graphs that
# hit the global-memory-pool path. fp16 is therefore not advertised — fp32 is solid, and the
# fp16 lib can still be built (MH_DTYPES=half) for experimentation. VDHA covers fp16 SpMSpV.
_SAFE_PRECISIONS = {"fp32", "fp64"}


def _supports(variant: str, precision: str) -> bool:
    return precision in _SAFE_PRECISIONS and _lib_path_for(precision) is not None


def get_providers() -> list[SpgemmProvider]:
    return [
        SpgemmProvider(
            framework="cuda",  # standalone CUDA kernel reached via ctypes
            provider="mh_spgemm",
            make_fn=_make_fn,
            variants=["csr"],
            self_timed=True,  # returns MH-SpGEMM's own phase-summed runtime
            supports=_supports,
        )
    ]
