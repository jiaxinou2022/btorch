"""cuSPARSE SpGEMM baseline (CuPy): C = A·Aᵀ via ``cusparseSpGEMM``.

``cupyx.scipy.sparse``'s sparse @ sparse dispatches to cuSPARSE's SpGEMM, so this is the
designated leaderboard baseline that MH-SpGEMM (and any other SpGEMM kernel) is ranked
against. cuSPARSE here accumulates in fp32; the oracle is fp64 SciPy.

Run:  pytest connectome_dataset/benchmarks/cupy/cusparse/test_spgemm.py -v -s
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spgemm

from .spmv import CUPY_AVAILABLE, sync

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy / CUDA device unavailable")


_DTYPES = {"fp32": np.float32, "fp16": np.float16}


@pytest.mark.parametrize("variant", ["csr"])
def test_spgemm(benchmark, spgemm_case, bench_cfg, variant, precision):
    # Shares each leaderboard cell with MH-SpGEMM at the same precision (fp32 / fp16).
    if precision not in _DTYPES:
        pytest.skip(f"cupy cuSPARSE SpGEMM baseline supports {list(_DTYPES)}, not {precision}")
    import cupy as cp
    import cupyx.scipy.sparse as csp

    # scipy.sparse has no float16; build float32 on host, then cast on device (cuSPARSE
    # fp16 SpGEMM may be unsupported — skip cleanly if so).
    try:
        a = csp.csr_matrix(spgemm_case.matrix.astype(np.float32))
        if precision == "fp16":
            a = a.astype(cp.float16)
        at = a.T.tocsr()

        def fn():
            return a @ at

        compile_ms = measure_compile_ms(fn, sync)
        out = fn()
    except (NotImplementedError, ValueError, TypeError) as e:
        pytest.skip(f"cusparse spgemm {precision} unsupported: {e}")
    except cp.cuda.memory.OutOfMemoryError as e:
        # cuSPARSE SpGEMM needs a large workspace on the densest products; a kernel that
        # OOMs where MH-SpGEMM fits is itself a result, but we can't time it here.
        pytest.skip(f"cusparse spgemm OOM at {precision}: {e}")
    sync()
    got = out.astype(cp.float32).get().tocsr()  # -> host scipy CSR (scipy has no float16)
    status = "ok" if prec.validate_spgemm(got, spgemm_case.matrix, precision) else "incorrect"
    if bench_cfg.strict_correctness:
        assert status == "ok", f"cusparse spgemm/{precision} failed oracle"

    def run():
        result = fn()
        sync()
        return result

    run_pedantic(
        benchmark, run, warmup=bench_cfg.warmup_spgemm, rounds=bench_cfg.rounds_spgemm,
        extra_info=base_extra_info(
            framework="cupy", provider="cusparse", target="spgemm", variant=variant,
            case=spgemm_case, device="cuda", problem={}, knobs={"precision": precision},
            total_flops=flops_spgemm(spgemm_case.intermediate_products),
            timings={"compile_ms": compile_ms}, status=status,
        ),
    )
