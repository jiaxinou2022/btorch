"""Diagnose why cupy.cusparse looks slow vs cpp.cusparse (same cuSPARSE library).

Times cupy's ``a @ x`` three ways on real SNAP graphs:
  (a) per-call sync  — exactly what the benchmark harness measures (Python dispatch +
      fresh output allocation + full deviceSynchronize every call);
  (b) amortized      — many ops queued back-to-back under one pair of CUDA events, so
      Python dispatch overlaps GPU work: steady-state kernel throughput;
  (c) preallocated   — cusparseSpMM via cupy's low-level binding into a reused output
      buffer (no per-call allocation), amortized under CUDA events.
If (b)/(c) ≈ cpp.cusparse but (a) is much slower, the gap is harness overhead, not cuSPARSE.
"""
from __future__ import annotations

import time

import numpy as np

import cupy as cp
import cupyx.scipy.sparse as csp
from connectome_dataset.benchmarks.cases import load_spmv_case

GRAPHS = ["suitesparse_snap_ca_astroph", "suitesparse_snap_soc_epinions1", "suitesparse_snap_web_stanford"]
BATCHES = [1, 32, 128]
R = 200


def _amortized_ms(op, reps=R):
    for _ in range(5):
        op()
    cp.cuda.runtime.deviceSynchronize()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(reps):
        op()
    e1.record()
    e1.synchronize()
    return cp.cuda.get_elapsed_time(e0, e1) / reps  # ms


def _percall_sync_ms(op, reps=R):
    for _ in range(5):
        op()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        op()
        cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def main():
    print(f"cupy {cp.__version__}  |  device {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"{'graph':28s} {'N':>4} {'nnz':>9}  {'per-call-sync':>16} {'amortized':>16} {'prealloc':>16}  {'overhead':>8}")
    for g in GRAPHS:
        A = load_spmv_case(g).matrix.astype(np.float32)
        a = csp.csr_matrix(A)
        nnz = A.nnz
        for N in BATCHES:
            x = cp.asarray(np.random.default_rng(0).random((A.shape[0], N), dtype=np.float32))
            flops = 2.0 * nnz * N

            ta = _percall_sync_ms(lambda: a @ x)                       # harness style
            tb = _amortized_ms(lambda: a @ x)                          # steady-state kernel
            # preallocated: reuse output buffer via cusparse spmm binding
            y = cp.empty((A.shape[0], N), dtype=np.float32)
            try:
                from cupyx.cusparse import spmm as _spmm  # low-level, out= reuse
                tc = _amortized_ms(lambda: _spmm(a, x, y=y, alpha=1, beta=0))
                tcs = f"{tc:8.4f}ms {flops/tc/1e9:6.1f}"
            except Exception as e:  # noqa: BLE001
                tcs = f"n/a ({type(e).__name__})"

            def tf(ms):
                return flops / (ms * 1e-3) / 1e12

            print(f"{g:28s} {N:>4} {nnz:>9}  {ta:8.4f}ms {tf(ta):5.3f}TF  {tb:8.4f}ms {tf(tb):5.3f}TF  {tcs:>16}  {ta/tb:6.1f}x")


if __name__ == "__main__":
    main()
