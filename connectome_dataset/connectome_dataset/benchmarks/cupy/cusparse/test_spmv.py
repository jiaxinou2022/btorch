"""cuSPARSE SpMM/SpMV baseline benchmarks (CuPy), via pytest-benchmark.

Run:
    pytest connectome_dataset/benchmarks/cupy/ -v -s
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmm

from .spmv import CUPY_AVAILABLE, VARIANTS, make_fn, supports, sync

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy / CUDA device unavailable")


@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
@pytest.mark.parametrize("variant", VARIANTS)
def test_spmv(benchmark, spmv_case, bench_cfg, variant, batch_size, precision):
    if not supports(variant, precision):
        pytest.skip(f"cusparse {variant} does not support {precision}")

    x_np = np.random.default_rng(0).random((spmv_case.n, batch_size), dtype=np.float32)
    fn = make_fn(
        spmv_case.matrix, variant=variant, batch_size=batch_size,
        precision=precision, x_np=x_np,
    )
    try:
        compile_ms = measure_compile_ms(fn, sync)
        out = fn()
    except (NotImplementedError, ValueError, TypeError) as e:
        pytest.skip(f"cusparse {variant}/{precision} unsupported: {e}")
    sync()
    got = np.asarray(out.get(), dtype=np.float32)  # device -> host
    status = "ok" if prec.validate_spmm(got, spmv_case.matrix, x_np, precision) else "incorrect"

    def run():
        result = fn()
        sync()
        return result

    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_spmv,
        rounds=bench_cfg.rounds_spmv,
        extra_info=base_extra_info(
            framework="cupy",
            provider="cusparse",  # the designated leaderboard baseline
            target="spmm",
            variant=variant,
            case=spmv_case,
            device="cuda",
            problem={"N": batch_size},
            knobs={"precision": precision},
            total_flops=flops_spmm(spmv_case.nnz, batch_size),
            timings={"compile_ms": compile_ms},
            status=status,
        ),
    )
