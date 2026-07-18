"""Precision knob + reference-oracle validation for SpMM.

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
_RTOL = {"fp32": 1e-4, "tf32": 4e-2, "fp16": 6e-2, "bf16": 1e-1}


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
