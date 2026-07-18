"""SpMV benchmarks — JAX / brainevent, via pytest-benchmark.

Run:
    pytest connectome_dataset/benchmarks/jax/ -v -s
    pytest connectome_dataset/benchmarks/jax/test_spmv.py -k bcoo --benchmark-autosave
"""
from __future__ import annotations

import jax
import pytest

from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmm

from .spmv import BRAINEVENT_AVAILABLE, make_fn

_skip_brainevent = pytest.mark.skipif(not BRAINEVENT_AVAILABLE, reason="brainevent not installed")

VARIANTS = [
    ("jax", "bcoo"),
    pytest.param(("brainevent", "csr"), marks=_skip_brainevent),
    pytest.param(("brainevent", "event"), marks=_skip_brainevent),
]

# The spmv.py `alg` maps to a provider in the (framework, provider, target, variant) identity.
_PROVIDER = {"jax": "jax_native", "brainevent": "brainevent"}


@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
@pytest.mark.parametrize("alg_variant", VARIANTS)
def test_spmv(benchmark, spmv_case, bench_cfg, alg_variant, batch_size):
    alg, variant = alg_variant
    fn = make_fn(spmv_case.matrix, alg=alg, variant=variant, batch_size=batch_size)

    def run():
        return jax.block_until_ready(fn())

    compile_ms = measure_compile_ms(run)

    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_spmv,
        rounds=bench_cfg.rounds_spmv,
        extra_info=base_extra_info(
            framework="jax",
            provider=_PROVIDER.get(alg, alg),
            target="spmm",
            variant=variant,
            case=spmv_case,
            device=jax.default_backend(),
            problem={"N": batch_size},
            total_flops=flops_spmm(spmv_case.nnz, batch_size),
            timings={"compile_ms": compile_ms},
        ),
    )
