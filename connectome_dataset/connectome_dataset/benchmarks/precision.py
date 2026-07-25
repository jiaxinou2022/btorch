"""Precision knob + reference-oracle validation for the sparse targets.

SpMM oracles live here; SpGEMM (``C = A·Aᵀ``) and SpMSpV (``y = A·x``) oracles are at the
bottom of the module. All follow the same contract: a SciPy reference computed in the
baseline precision, and a ``validate_*`` that returns False (→ ``status="incorrect"``)
for a fast-but-wrong kernel so the leaderboard excludes it.

Original SpMM notes follow.

Precision knob + reference-oracle validation for SpMM.

``precision`` is an optimization knob (recorded in ``knobs``), not a variant: it
trades numerical accuracy for tensor-core throughput, the flagship modern-CUDA
axis. Values:

- ``fp32`` — IEEE single, the baseline / oracle precision.
- ``tf32`` — fp32 storage, TF32 tensor-core math mode (Ampere+); operands unchanged.
- ``fp16`` / ``bf16`` — operands cast to half / bfloat16.

Because the leaderboard folds ``precision`` into the workload-cell signature, a
kernel is only ever ranked against others *at the same precision*. Every reduced-
precision result is checked against an fp32 SciPy oracle within a
precision-appropriate tolerance; a kernel that is fast but wrong is recorded with
``status="incorrect"`` and excluded from ranking.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Relative tolerance for validating a reduced-precision SpMM against the fp32 oracle.
# Set by the unit round-off of the format, with headroom for accumulation over a row.
_RTOL = {"fp64": 1e-6, "fp32": 1e-4, "tf32": 4e-2, "fp16": 6e-2, "bf16": 1e-1}


def rtol(precision: str) -> float:
    return _RTOL.get(precision, 1e-2)


def oracle_spmm(matrix: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """The fp32 SciPy reference every accelerated SpMM is validated against."""
    return matrix.astype(np.float32) @ x.astype(np.float32)


def validate_spmm(result: np.ndarray, matrix: sp.csr_matrix, x: np.ndarray, precision: str) -> bool:
    ref = oracle_spmm(matrix, x)
    got = np.asarray(result, dtype=np.float32).reshape(ref.shape)
    scale = float(np.abs(ref).max()) or 1.0
    return bool(np.allclose(got, ref, rtol=rtol(precision), atol=rtol(precision) * scale))


# --- SpGEMM: C = A @ A.T -----------------------------------------------------------

def oracle_spgemm(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """The fp64 SciPy reference for C = A·Aᵀ (matches MH-SpGEMM's double accumulation)."""
    a = matrix.astype(np.float64)
    c = (a @ a.T).tocsr()
    c.sum_duplicates()
    c.eliminate_zeros()
    return c


def validate_spgemm(result: sp.csr_matrix, matrix: sp.csr_matrix, precision: str = "fp64") -> bool:
    """True iff the kernel's C matches the oracle in shape, nnz pattern, and values.

    Kernels differ on whether numerically-zero products are kept, so we compare the
    *sparse difference* under an absolute+relative tolerance rather than nnz equality.
    """
    ref = oracle_spgemm(matrix)
    got = result.tocsr().astype(np.float64)
    got.sum_duplicates()
    got.eliminate_zeros()
    if got.shape != ref.shape:
        return False
    tol = rtol(precision if precision in _RTOL else "fp32")
    scale = float(abs(ref).max()) if ref.nnz else 1.0
    diff = (got - ref).tocoo()
    max_abs = float(np.abs(diff.data).max()) if diff.nnz else 0.0
    return max_abs <= tol * scale + 1e-9


# --- SpMSpV: y = A @ x, x sparse ---------------------------------------------------

def make_sparse_vector(n: int, sparsity: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A random sparse operand for SpMSpV: ``ceil(sparsity·n)`` active (index, value) pairs.

    Returns sorted int32 indices and fp32 values — the format VDHA and the oracle share.
    """
    rng = np.random.default_rng(seed)
    k = max(1, int(np.ceil(sparsity * n)))
    idx = np.sort(rng.choice(n, size=min(k, n), replace=False)).astype(np.int32)
    val = rng.standard_normal(idx.size).astype(np.float32)
    return idx, val


def oracle_spmspv(matrix: sp.csr_matrix, x_indices: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """The fp32 SciPy reference for y = A·x with x given as (indices, values)."""
    x = np.zeros(matrix.shape[1], dtype=np.float32)
    x[np.asarray(x_indices, dtype=np.int64)] = np.asarray(x_values, dtype=np.float32)
    return matrix.astype(np.float32) @ x


def validate_spmspv(
    result: np.ndarray, matrix: sp.csr_matrix, x_indices: np.ndarray, x_values: np.ndarray,
    precision: str = "fp32",
) -> bool:
    ref = oracle_spmspv(matrix, x_indices, x_values)
    got = np.asarray(result, dtype=np.float32).reshape(ref.shape)
    scale = float(np.abs(ref).max()) or 1.0
    return bool(np.allclose(got, ref, rtol=rtol(precision), atol=rtol(precision) * scale))
