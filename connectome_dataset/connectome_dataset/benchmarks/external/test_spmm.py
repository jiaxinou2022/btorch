"""Generic benchmark for out-of-tree SpMM providers discovered via the registry.

Any provider registered under the ``connectome_bench.spmm`` entry-point group (or via
``providers.register_spmm``) is run here through the *same* cases, precision knob,
oracle validation, and record schema as the in-tree baselines — so a third-party
kernel lands on the same leaderboard without editing this repo. When no external
provider is installed, the parametrization collapses to a single skip.
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import providers
from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmm

_DISCOVERED = providers.discover_spmm()
_CASES = [(p, v) for p in _DISCOVERED for v in p.variants]
_PARAMS = _CASES or [pytest.param((None, None), marks=pytest.mark.skip(reason="no external SpMM providers registered"))]


def _ids(pv):
    p, v = pv
    return "none" if p is None else f"{p.framework}.{p.provider}.{v}"


@pytest.mark.parametrize("provider_variant", _PARAMS, ids=_ids)
@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
def test_external_spmm(benchmark, spmv_case, bench_cfg, provider_variant, batch_size, precision):
    provider, variant = provider_variant
    if provider.supports is not None and not provider.supports(variant, precision):
        pytest.skip(f"{provider.provider}/{variant} does not support {precision}")

    device = bench_cfg.device or "cpu"
    x_np = np.random.default_rng(0).random((spmv_case.n, batch_size), dtype=np.float32)
    try:
        fn = provider.make_fn(
            spmv_case.matrix, variant=variant, device=device,
            batch_size=batch_size, precision=precision, x_np=x_np,
        )
        compile_ms = measure_compile_ms(lambda: fn()[0] if provider.self_timed else fn(), provider.sync)
        out = fn()
    except NotImplementedError as e:
        pytest.skip(f"{provider.provider}/{variant}/{precision} unsupported on {device}: {e}")
    if provider.sync is not None:
        provider.sync()

    result = out[0] if provider.self_timed else out
    got = providers.to_numpy(result)
    status = "ok" if prec.validate_spmm(got, spmv_case.matrix, x_np, precision) else "incorrect"

    timings = {"compile_ms": compile_ms}
    warmup, rounds = bench_cfg.warmup_spmv, bench_cfg.rounds_spmv
    if provider.self_timed:
        # Kernel reports its own (transfer-free) time — the authoritative number. Ingest
        # prefers self_timed_ms over the wall-clock that pytest-benchmark records below.
        # These kernels average internally over many iterations, so a single wall round
        # suffices (avoids re-running a 1000-iteration internal loop per pedantic round).
        timings["self_timed_ms"] = float(out[1])
        warmup, rounds = 0, 1

    def run():
        r = fn()
        if provider.sync is not None:
            provider.sync()
        return r[0] if provider.self_timed else r

    run_pedantic(
        benchmark, run, warmup=warmup, rounds=rounds,
        extra_info=base_extra_info(
            framework=provider.framework, provider=provider.provider, target="spmm", variant=variant,
            case=spmv_case, device=device, problem={"N": batch_size}, knobs={"precision": precision},
            total_flops=flops_spmm(spmv_case.nnz, batch_size), timings=timings, status=status,
        ),
    )
