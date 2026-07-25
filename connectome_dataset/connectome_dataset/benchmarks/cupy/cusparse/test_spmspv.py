"""cuSPARSE SpMV baseline for the SpMSpV target (CuPy).

cuSPARSE has no sparse-vector SpMSpV; the honest baseline — and the one the VDHA paper
compares against — is a dense-vector ``cusparseSpMV`` over the densified operand, which
is sparsity-insensitive (it touches all of A regardless of how few x entries are set).
This quantifies exactly the work a vector-driven kernel like VDHA saves.

Run:  pytest connectome_dataset/benchmarks/cupy/cusparse/test_spmspv.py -v -s
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.config import SPMSPV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmspv

from .spmv import CUPY_AVAILABLE, sync

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy / CUDA device unavailable")


def _products_touched(matrix, x_indices) -> int:
    col_nnz = np.diff(matrix.tocsc().indptr)
    return int(col_nnz[np.asarray(x_indices, dtype=np.int64)].sum())


_DTYPES = {"fp32": np.float32, "fp16": np.float16}


@pytest.mark.parametrize("vector_sparsity", SPMSPV_DEFAULTS.vector_sparsities)
@pytest.mark.parametrize("variant", ["csr"])
def test_spmspv(benchmark, spmspv_case, bench_cfg, variant, vector_sparsity, precision):
    if precision not in _DTYPES:
        pytest.skip(f"cupy cuSPARSE SpMV baseline supports {list(_DTYPES)}, not {precision}")
    import cupy as cp
    import cupyx.scipy.sparse as csp

    dtype = _DTYPES[precision]
    x_idx, x_val = prec.make_sparse_vector(spmspv_case.n, vector_sparsity, seed=SPMSPV_DEFAULTS.seed)
    x_dense = np.zeros(spmspv_case.n, dtype=dtype)
    x_dense[x_idx] = x_val.astype(dtype)
    x = cp.asarray(x_dense)

    # scipy.sparse has no float16: build float32 on host, cast on device (skip if the
    # fp16 SpMV path is unsupported).
    try:
        a = csp.csr_matrix(spmspv_case.matrix.astype(np.float32))
        if precision == "fp16":
            a = a.astype(cp.float16)

        def fn():
            return a @ x

        compile_ms = measure_compile_ms(fn, sync)
        out = fn()
    except (NotImplementedError, ValueError, TypeError) as e:
        pytest.skip(f"cusparse spmv {precision} unsupported: {e}")
    sync()
    got = np.asarray(out.get(), dtype=np.float32)
    status = "ok" if prec.validate_spmspv(got, spmspv_case.matrix, x_idx, x_val, precision) else "incorrect"
    if bench_cfg.strict_correctness:
        assert status == "ok", f"cusparse spmspv/{precision} failed oracle"

    # Self-time the kernel with CUDA events (steady-state, amortized), so cuSPARSE is
    # compared to VDHA on kernel-only time — not wall-clock inflated by per-call dispatch,
    # which dominates at these microsecond scales and would otherwise flatter VDHA.
    self_ms = _cuda_event_ms(fn, sync)

    def run():
        result = fn()
        sync()
        return result

    run_pedantic(
        benchmark, run, warmup=bench_cfg.warmup_spmspv, rounds=bench_cfg.rounds_spmspv,
        extra_info=base_extra_info(
            framework="cupy", provider="cusparse", target="spmspv", variant=variant,
            case=spmspv_case, device="cuda",
            problem={"vector_sparsity": float(vector_sparsity), "vector_nnz": int(x_idx.size)},
            knobs={"precision": precision},
            total_flops=flops_spmspv(_products_touched(spmspv_case.matrix, x_idx)),
            timings={"compile_ms": compile_ms, "self_timed_ms": self_ms}, status=status,
        ),
    )


def _cuda_event_ms(fn, sync, iters: int = 50) -> float:
    """Mean kernel time (ms) over `iters` back-to-back launches under one CUDA-event pair."""
    import cupy as cp

    for _ in range(5):
        fn()
    sync()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop)) / iters
