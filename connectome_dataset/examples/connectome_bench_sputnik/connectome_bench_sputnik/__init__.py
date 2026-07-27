"""Out-of-tree Sputnik SpMM provider — the same kernel as the in-tree C++ baseline,
reached instead through the entry-point registry to validate that path with a real
SOTA kernel.

It ctypes-loads ``libconnectome_sputnik.so`` (built by
``cmake -DCONNECTOME_BENCH_BUILD_SPUTNIK=ON``; point ``CONNECTOME_SPUTNIK_LIB`` at it,
else the repo's ``cpp/build-sputnik`` is tried). Device buffers persist across calls, so
the benchmarked callable times only the kernel; the host copy happens off the timed path
via the result's ``get()`` (which ``providers.to_numpy`` uses for oracle validation).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from connectome_dataset.benchmarks.providers import SpmmProvider

_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    candidates = []
    if os.environ.get("CONNECTOME_SPUTNIK_LIB"):
        candidates.append(os.environ["CONNECTOME_SPUTNIK_LIB"])
    # Search from the install location and from the working directory (the package is
    # pip-installed into site-packages, so its parents won't include the repo build dir).
    roots = list(Path(__file__).resolve().parents) + list(Path.cwd().resolve().parents) + [Path.cwd()]
    for base in roots:
        candidates.append(base / "cpp" / "build-sputnik" / "libconnectome_sputnik.so")
    lib_path = next((str(c) for c in candidates if Path(c).exists()), None)
    if lib_path is None:
        raise OSError("libconnectome_sputnik.so not found; set CONNECTOME_SPUTNIK_LIB")
    lib = ctypes.CDLL(lib_path)
    lib.cbn_sputnik_prepare.restype = ctypes.c_void_p
    lib.cbn_sputnik_prepare.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4
    lib.cbn_sputnik_compute.restype = ctypes.c_int
    lib.cbn_sputnik_compute.argtypes = [ctypes.c_void_p]
    lib.cbn_sputnik_compute_device.restype = ctypes.c_int
    lib.cbn_sputnik_compute_device.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cbn_sputnik_copy_out.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cbn_sputnik_free.argtypes = [ctypes.c_void_p]
    _LIB = lib
    return lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class _Result:
    """Lazily copies the device result to host — only when oracle validation asks."""

    def __init__(self, lib, handle, m: int, n: int):
        self._lib, self._h, self._m, self._n = lib, handle, m, n

    def get(self) -> np.ndarray:
        y = np.empty((self._m, self._n), dtype=np.float32)
        self._lib.cbn_sputnik_copy_out(self._h, _ptr(y))
        return y


def _make_fn(matrix, *, variant, device, batch_size, precision="fp32", x_np=None):
    lib = _load_lib()
    a = matrix.tocsr().astype(np.float32)
    m, k = a.shape
    n, nnz = batch_size, a.nnz
    row_off = a.indptr.astype(np.int32)
    col_idx = a.indices.astype(np.int32)
    vals = a.data.astype(np.float32)
    # longest-row-first swizzle (Sputnik's load balancing), computed host-side
    lengths = np.diff(row_off)
    swizzle = np.argsort(-lengths).astype(np.int32)
    handle = lib.cbn_sputnik_prepare(m, k, n, nnz, _ptr(row_off), _ptr(col_idx), _ptr(vals), _ptr(swizzle))
    if not handle:
        raise RuntimeError("cbn_sputnik_prepare failed")

    def run():
        rc = lib.cbn_sputnik_compute(handle)
        if rc != 0:
            raise RuntimeError(f"sputnik compute failed (cuda err {rc})")
        return _Result(lib, handle, m, n)

    return run


def get_providers() -> list[SpmmProvider]:
    return [
        SpmmProvider(
            # Same native CUDA kernel as the in-tree C++ baseline, just reached via ctypes —
            # framework = the kernel's runtime (native C++), so it unifies with cpp.sputnik.
            framework="cpp",
            provider="sputnik",
            make_fn=_make_fn,
            variants=["csr"],
            supports=lambda variant, precision: precision == "fp32",
        )
    ]
