"""DTC-SpMM (tensor-core weighted SpMM, Fan et al. ASPLOS'24) wrapper — the single
source of truth for the in-tree leaf and the out-of-tree provider.

DTC-SpMM *is* a weighted SpMM: its kernel does ``sparse_A[TCblocktile_id[e]] = valuesA[e]``
and multiplies by that per-nonzero value. The upstream benchmark wrapper hardcodes
``valuesA = ones`` (GNN aggregation), so we apply a small patch
(scripts/patches/dtc_weighted.patch) that (a) adds a ``run_DTCSpMM_weighted`` entry taking a
real ``valuesA`` tensor and (b) returns the kernel's internally-averaged time. We compute
``valuesA`` in DTC's blocked (TCblocktile_id) order from the preprocess outputs, so it
computes the same weighted A@X as cuSPARSE/Sputnik/FlashSparse and ranks in the same cells.

Constraints: fp32; the dense width N must be a multiple of BLK_H=16 (so N=1/8 skip). It
manages its own device timing and returns kernel-only ms -> the provider is ``self_timed``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_MOD = None
_BLK_H, _BLK_W = 16, 8


def supports(variant: str, precision: str) -> bool:
    return precision == "fp32"


def load_mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    roots = []
    if os.environ.get("DTC_ROOT"):
        roots.append(Path(os.environ["DTC_ROOT"]))
    for base in list(Path.cwd().resolve().parents) + [Path.cwd()]:
        roots.append(base / "external" / "DTC-SpMM" / "DTC-SpMM")
    root = next((r for r in roots if list(r.glob("DTCSpMM*.so"))), None)
    if root is None:
        raise OSError("DTCSpMM extension not found; run scripts/build_dtc.sh")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import DTCSpMM

    _MOD = DTCSpMM
    return _MOD


def _blocked_values(a, TCblocktileId, TCblockoffset, TCblockRowid, SparseAToXindex):
    """Real per-nonzero values reordered into DTC's blocked layout.

    Each blocked entry e belongs to global block ``b`` (via TCblockoffset) at tile position
    TCblocktileId[e]; its matrix coordinate is
    ``(TCblockRowid[b]*BLK_H + tid//BLK_W, SparseAToXindex[b*BLK_W + tid%BLK_W])``. We look
    that coordinate up in A.
    """
    tid = TCblocktileId.cpu().numpy().astype(np.int64)
    offs = TCblockoffset.cpu().numpy().astype(np.int64)
    rowid = TCblockRowid.cpu().numpy().astype(np.int64)
    atox = SparseAToXindex.cpu().numpy().astype(np.int64)
    n = a.shape[0]
    e = np.arange(tid.shape[0], dtype=np.int64)
    b = np.searchsorted(offs, e, side="right") - 1
    grow = rowid[b] * _BLK_H + tid // _BLK_W
    gcol = atox[b * _BLK_W + tid % _BLK_W]
    coo = a.tocoo()
    key = coo.row.astype(np.int64) * n + coo.col.astype(np.int64)
    order = np.argsort(key)
    key_s, data_s = key[order], coo.data[order]
    q = grow * n + gcol
    pos = np.clip(np.searchsorted(key_s, q), 0, key_s.shape[0] - 1)
    vals = np.where(key_s[pos] == q, data_s[pos], 0.0).astype(np.float32)
    return vals


def make_fn(matrix, *, variant="csr", device="cuda", batch_size, precision="fp32", x_np=None):
    import torch

    if batch_size % _BLK_H != 0:  # tensor-core row window
        raise NotImplementedError(f"DTC-SpMM needs N%{_BLK_H}==0, got N={batch_size}")
    dtc = load_mod()
    a = matrix.tocsr().astype(np.float32)
    n, nnz = a.shape[0], a.nnz
    col = torch.tensor(a.indices, dtype=torch.int32, device="cuda")
    rowp = torch.tensor(a.indptr, dtype=torch.int32, device="cuda")
    nrw = (n + _BLK_H - 1) // _BLK_H
    edge_to_col = torch.zeros(nnz, dtype=torch.int32, device="cuda")
    edge_to_row = torch.zeros(nnz, dtype=torch.int32, device="cuda")
    block_part = torch.zeros(nrw, dtype=torch.int32, device="cuda")
    rwo, tc_rowid, tc_tileid, tc_offset, atox, _bc = dtc.preprocess_gpu(
        col, rowp, n, _BLK_H, _BLK_W, block_part, edge_to_col, edge_to_row
    )
    values = torch.tensor(_blocked_values(a, tc_tileid, tc_offset, tc_rowid, atox),
                          dtype=torch.float32, device="cuda")
    if x_np is None:
        x_np = np.random.default_rng(0).random((n, batch_size), dtype=np.float32)
    x = torch.tensor(x_np.astype(np.float32), device="cuda")

    def run():
        out, ms = dtc.run_DTCSpMM_weighted(x, rwo, tc_tileid, tc_offset, atox, values, n, nnz, "float_nonsplit")
        return out, float(ms.item())

    return run
