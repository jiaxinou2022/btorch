"""Spike-delivery backends for the btorch connectome RSNN.

btorch's ``SparseConn`` propagates spikes as ``Y = spikes @ W`` (presynaptic spikes through
the recurrent connectivity). This module provides that same delivery three ways — the axis
the btorch RSNN records as ``variant`` — so VDHA and MH-SpGEMM are benchmarked against
btorch's own native SpMV as the recurrent operator:

    variant     op                              spikes operand
    ---------   -----------------------------   ------------------------
    native      torch sparse ``spikes @ W``     dense (btorch's native backend)
    vdha        SpMSpV ``y = Wᵀ·s``             one sparse spike vector
    mh_spgemm   SpGEMM ``Y = Wᵀ·Sᵀ``            batch of sparse spike vectors

``spikes @ W == Wᵀ · spikesᵀ``, so the event-driven kernels prepare with ``Wᵀ`` and deliver
the spikes as the sparse operand. Each backend returns ``(mean_deliver_ms_per_step, Y)`` over
``timesteps`` steps at the operating firing rate (the beNNch deliver phase), all CUDA-event
timed; ``Y`` (dense, all postsynaptic neurons) is validated against a SciPy reference.
"""
from __future__ import annotations

import ctypes

import numpy as np
import scipy.sparse as sp


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


def weighted_connectivity(matrix: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    """Random synaptic weights on the connectome's sparsity pattern (as btorch's SparseRSNN)."""
    coo = matrix.tocoo().astype(np.float32)
    coo.data[:] = (np.random.default_rng(seed).standard_normal(coo.nnz) * 0.1).astype(np.float32)
    w = sp.csr_matrix(coo)
    w.sum_duplicates()
    w.sort_indices()
    return w


def reference(W: sp.csr_matrix, spikes: np.ndarray) -> np.ndarray:
    """SciPy ground truth ``Y = spikes @ W`` (spikes: (n,) or (batch, n))."""
    return spikes.astype(np.float32) @ W


def native_deliver(W: sp.csr_matrix, spikes: np.ndarray, timesteps: int, device: str):
    """torch native sparse SpMV — the backend btorch's ``SparseConn(sparse_backend="native")``
    uses internally: ``spikes @ W == (Wᵀ · spikesᵀ)ᵀ`` via ``torch.sparse.mm``, CUDA-event timed."""
    import torch

    wt = W.T.tocoo()
    idx = torch.tensor(np.vstack([wt.row, wt.col]), dtype=torch.long, device=device)
    wt_sp = torch.sparse_coo_tensor(idx, torch.tensor(wt.data, dtype=torch.float32, device=device),
                                    size=wt.shape).coalesce()
    xt = torch.as_tensor(np.atleast_2d(spikes).astype(np.float32).T, device=device)  # (n, batch)
    sync = torch.cuda.synchronize
    for _ in range(3):
        torch.sparse.mm(wt_sp, xt)
    sync()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timesteps):
        out = torch.sparse.mm(wt_sp, xt)  # (n_dst, batch) = (spikes @ W)ᵀ
    stop.record()
    sync()
    return start.elapsed_time(stop) / timesteps, out.T.detach().cpu().numpy()


def vdha_deliver(W: sp.csr_matrix, spikes: np.ndarray, timesteps: int):
    """SpMSpV delivery ``y = Wᵀ·s`` for one sparse spike vector, over ``timesteps``."""
    import connectome_bench_vdha as pkg

    lib = pkg._load_lib("fp32")
    a = W.T.tocsc().astype(np.float32)  # A = Wᵀ, y = A·s = spikes @ W
    n, num_cols = a.shape
    idx = np.nonzero(spikes.reshape(-1))[0].astype(np.int32)
    val = spikes.reshape(-1)[idx].astype(np.float32)
    col_ptr, csc_row, csc_val = a.indptr.astype(np.int32), a.indices.astype(np.int32), a.data.astype(np.float32)
    handle = lib.cbn_vdha_prepare(n, num_cols, a.nnz, _ptr(col_ptr), _ptr(csc_row), _ptr(csc_val),
                                  idx.size, _ptr(idx), _ptr(val))
    if not handle:
        raise RuntimeError("cbn_vdha_prepare failed")
    ms = ctypes.c_double(0.0)
    rc = lib.cbn_vdha_compute_timed(handle, timesteps, ctypes.byref(ms))
    y = np.empty(n, dtype=np.float32)
    lib.cbn_vdha_copy_out(handle, _ptr(y))
    lib.cbn_vdha_free(handle)
    if rc != 0:
        raise RuntimeError(f"vdha deliver failed (cuda err {rc})")
    return ms.value, y


def mh_deliver(W: sp.csr_matrix, spikes: np.ndarray, timesteps: int):
    """SpGEMM delivery ``Y = spikes @ W`` for a batch of sparse spike vectors, over ``timesteps``.

    Computed as ``Wᵀ · Sᵀ`` compressed to the spiking neurons (``Wᵀ[:, active]·Sᵀ[active,:]``):
    the event-driven form has no empty operand rows, which MH-SpGEMM's dense-ish B-tiling needs
    (it also needs ≥ BLOCK_SIZE columns, i.e. batch ≥ 32).
    """
    import connectome_bench_mh_spgemm as pkg

    lib = pkg._load_lib("fp32")
    lib.cbn_mh_prepare_ab.restype = ctypes.c_void_p
    lib.cbn_mh_prepare_ab.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p] * 3
    lib.cbn_mh_result_cols.restype = ctypes.c_int
    lib.cbn_mh_result_cols.argtypes = [ctypes.c_void_p]

    spikes = np.atleast_2d(spikes)      # (batch, n)
    batch, n = spikes.shape
    St = sp.csr_matrix(spikes.T.astype(np.float32))     # Sᵀ: (n, batch), spikes as columns
    active = np.unique(St.tocsc().indices).astype(np.int64)  # spiking presynaptic neurons
    if active.size == 0:
        return 0.0, np.zeros((batch, n), dtype=np.float32)
    a = W.T.tocsr()[:, active].tocsr().astype(np.float32)  # Wᵀ[:, active]: (n_dst, |active|)
    s = St[active, :].astype(np.float32)                   # Sᵀ[active, :]: (|active|, batch), no empty rows
    for mat in (a, s):
        mat.sum_duplicates()
        mat.eliminate_zeros()
        mat.sort_indices()
    m, k = a.shape
    ap, ac, av = a.indptr.astype(np.int32), a.indices.astype(np.int32), a.data.astype(np.float32)
    bp, bc, bv = s.indptr.astype(np.int32), s.indices.astype(np.int32), s.data.astype(np.float32)
    handle = lib.cbn_mh_prepare_ab(m, k, batch, a.nnz, _ptr(ap), _ptr(ac), _ptr(av),
                                   s.nnz, _ptr(bp), _ptr(bc), _ptr(bv))
    if not handle:
        raise RuntimeError("cbn_mh_prepare_ab failed")
    ms = ctypes.c_double(0.0)
    nnz = ctypes.c_int(0)
    rc = lib.cbn_mh_compute_timed(handle, timesteps, ctypes.byref(ms), ctypes.byref(nnz))
    if rc != 0:
        lib.cbn_mh_free(handle)
        raise RuntimeError(f"mh deliver failed (rc {rc})")
    ncols = lib.cbn_mh_result_cols(handle)
    indptr = np.empty(m + 1, dtype=np.int32)
    indices = np.empty(nnz.value, dtype=np.int32)
    data = np.empty(nnz.value, dtype=np.float32)
    lib.cbn_mh_copy_out(handle, _ptr(indptr), _ptr(indices), _ptr(data))
    lib.cbn_mh_free(handle)
    C = sp.csr_matrix((data, indices, indptr), shape=(m, ncols))  # Wᵀ·Sᵀ = (spikes @ W)ᵀ
    return ms.value, np.asarray(C.todense(), dtype=np.float32).T   # -> (batch, n_dst)
