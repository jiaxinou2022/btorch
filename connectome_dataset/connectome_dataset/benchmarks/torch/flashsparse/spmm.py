"""FlashSparse (tensor-core weighted SpMM) wrapper — the single source of truth used by
both the in-tree benchmark leaf and the out-of-tree provider package.

FlashSparse is a general weighted SpMM: ``FS_Block.blockProcess_fp16`` reorders the real
per-nonzero values into its blocked TCF layout and ``FS_SpMM.forward_fp16`` multiplies by
them. It manages its own device transfers and returns a kernel-only time from internal
CUDA events, so callers time it via that returned value (``self_timed``), not wall-clock.

Constraints: fp16 only; the dense width ``N`` must be a multiple of 8 (tensor-core tile),
so ``N=1`` SpMV is unsupported. Build the extensions with ``scripts/build_flashsparse.sh``;
point ``FLASHSPARSE_ROOT`` at ``external/FlashSparse/FlashSparse`` (else the repo default).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_MODS = None


def supports(variant: str, precision: str) -> bool:
    return precision == "fp16"


def load_mods():
    """Import FS_SpMM / FS_Block, adding the built extensions to sys.path. Raises if absent."""
    global _MODS
    if _MODS is not None:
        return _MODS
    roots = []
    if os.environ.get("FLASHSPARSE_ROOT"):
        roots.append(Path(os.environ["FLASHSPARSE_ROOT"]))
    for base in list(Path.cwd().resolve().parents) + [Path.cwd()]:
        roots.append(base / "external" / "FlashSparse" / "FlashSparse")
    root = next((r for r in roots if (r / "SpMM").exists() and (r / "Block").exists()), None)
    if root is None:
        raise OSError("FlashSparse not found; set FLASHSPARSE_ROOT or run scripts/build_flashsparse.sh")
    for sub in ("SpMM", "Block"):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import FS_Block
    import FS_SpMM

    _MODS = (FS_SpMM, FS_Block)
    return _MODS


def make_fn(matrix, *, variant="csr", device="cuda", batch_size, precision="fp16", x_np=None, epochs=50):
    """Return a self-timed callable → ``(output_fp16, kernel_ms)`` for the weighted SpMM."""
    import torch

    if batch_size % 8 != 0:  # tensor-core tile: N must be a multiple of 8
        raise NotImplementedError(f"FlashSparse needs N%8==0, got N={batch_size}")
    fs_spmm, _fs_block = load_mods()
    fs_block = _fs_block

    a = matrix.tocsr().astype(np.float32)
    n = a.shape[0]
    padded = n + (8 - n % 8) % 8  # pad rows to a multiple of the 8-row window
    indptr = np.concatenate([a.indptr, np.full(padded - n, a.indptr[-1], dtype=a.indptr.dtype)])
    row = torch.from_numpy(indptr.astype(np.int32))
    col = torch.from_numpy(a.indices.astype(np.int32))
    vals = torch.from_numpy(a.data.astype(np.float16))
    row_p, col_i, val_b = fs_block.blockProcess_fp16(row, col, vals, 8, 8)  # -> blocked TCF layout

    if x_np is None:
        x_np = np.random.default_rng(0).random((n, batch_size), dtype=np.float32)
    rhs = torch.from_numpy(x_np.astype(np.float16))  # forward_fp16 copies to device itself

    def run():
        out, ms = fs_spmm.forward_fp16(row_p, col_i, val_b, rhs, padded, batch_size, n, epochs)
        return out, float(ms.item())

    return run
