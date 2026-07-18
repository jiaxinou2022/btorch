"""cuSPARSE SpMM/SpMV baseline via CuPy.

``cupyx.scipy.sparse`` dispatches sparse @ dense to cuSPARSE (``cusparseSpMM`` /
``cusparseSpMV``), so this is the canonical cuSPARSE baseline every accelerated
kernel is ranked against (``leaderboard.baseline_provider == "cusparse"``). SpMV
is the ``N=1`` case of SpMM — one provider covers both.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # no cupy, no driver, or no visible device
    cp = None
    csp = None
    CUPY_AVAILABLE = False

VARIANTS = ["csr", "coo"]

# cuSPARSE SpMM compute types exposed through CuPy. Reduced precision is wired for
# CSR only (half COO SpMM is not reliably supported).
_DTYPES = {"fp32": np.float32, "fp16": np.float16}


def supports(variant: str, precision: str) -> bool:
    return precision in _DTYPES and (precision == "fp32" or variant == "csr")


def sync() -> None:
    cp.cuda.runtime.deviceSynchronize()


def make_fn(
    mat: sp.csr_matrix, *, variant: str, batch_size: int,
    precision: str = "fp32", x_np: np.ndarray | None = None,
):
    """Return a cuSPARSE sparse @ dense callable (result stays on device)."""
    dtype = _DTYPES[precision]
    mat = mat.astype(dtype)
    if variant == "csr":
        a = csp.csr_matrix(mat)
    elif variant == "coo":
        a = csp.coo_matrix(mat)
    else:
        raise ValueError(f"Unknown cusparse variant '{variant}'")
    if x_np is None:
        x_np = np.random.default_rng(0).standard_normal((mat.shape[0], batch_size), dtype=np.float32)
    x = cp.asarray(x_np.astype(dtype))
    return lambda: a @ x
