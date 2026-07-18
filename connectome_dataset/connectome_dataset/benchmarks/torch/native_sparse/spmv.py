"""SpMV setup helpers — PyTorch native CSR / COO.

The cuSPARSE baseline lives in its own provider (``cupy/cusparse``), not here.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

VARIANTS = ["csr", "coo"]

# precision -> torch dtype for the operands. tf32 is a dense tensor-core math mode
# (no effect on cuSPARSE SpMM), so it is not offered here — it belongs to the dense
# RSNN path. Reduced precision is only wired for CSR; COO half-SpMM is unsupported
# on most builds.
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def supports(variant: str, precision: str) -> bool:
    return precision == "fp32" or variant == "csr"


def _to_torch_csr(mat: sp.csr_matrix, device: str, dtype: torch.dtype) -> torch.Tensor:
    mat = mat.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.tensor(mat.indptr, dtype=torch.int32, device=device),
        torch.tensor(mat.indices, dtype=torch.int32, device=device),
        torch.tensor(mat.data, device=device).to(dtype),
        size=mat.shape,
        device=device,
    )


def _to_torch_coo(mat: sp.spmatrix, device: str, dtype: torch.dtype) -> torch.Tensor:
    coo = mat.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64, device=device)
    val = torch.tensor(coo.data, device=device).to(dtype)
    return torch.sparse_coo_tensor(idx, val, size=mat.shape).coalesce()


def make_fn(
    mat: sp.csr_matrix, *, variant: str, device: str, batch_size: int,
    precision: str = "fp32", x_np: np.ndarray | None = None,
):
    """Return a PyTorch SpMV callable for the given native-sparse variant/precision."""
    if x_np is None:
        x_np = np.random.rand(mat.shape[0], batch_size).astype(np.float32)
    dtype = _DTYPES[precision]
    x = torch.tensor(x_np, device=device).to(dtype)

    if variant == "csr":
        a = _to_torch_csr(mat, device, dtype)
        return (lambda: torch.mm(a, x)) if batch_size > 1 else (lambda: torch.mv(a, x[:, 0]))

    if variant == "coo":
        a = _to_torch_coo(mat, device, dtype)
        return lambda: torch.sparse.mm(a, x)

    raise ValueError(f"Unknown native sparse variant '{variant}'")
