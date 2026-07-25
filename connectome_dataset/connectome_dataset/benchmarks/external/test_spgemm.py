"""Generic benchmark for out-of-tree SpGEMM providers discovered via the registry.

Any provider registered under ``connectome_bench.spgemm`` (or via
``providers.register_spgemm``) computes ``C = A·Aᵀ`` over the shared cases and is checked
against the fp64 SciPy oracle, landing on the same leaderboard as the in-tree baselines.
When no external provider is installed, the parametrization collapses to a single skip.
"""
from __future__ import annotations

import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks import providers
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spgemm

_DISCOVERED = providers.discover_spgemm()
_CASES = [(p, v) for p in _DISCOVERED for v in p.variants]
_PARAMS = _CASES or [pytest.param((None, None), marks=pytest.mark.skip(reason="no external SpGEMM providers registered"))]


def _ids(pv):
    p, v = pv
    return "none" if p is None else f"{p.framework}.{p.provider}.{v}"


@pytest.mark.parametrize("provider_variant", _PARAMS, ids=_ids)
def test_external_spgemm(benchmark, spgemm_case, bench_cfg, provider_variant, precision):
    provider, variant = provider_variant
    if provider.supports is not None and not provider.supports(variant, precision):
        pytest.skip(f"{provider.provider}/{variant} does not support {precision}")

    device = bench_cfg.device or "cpu"
    try:
        fn = provider.make_fn(spgemm_case.matrix, variant=variant, device=device, precision=precision)
        compile_ms = measure_compile_ms(lambda: fn()[0] if provider.self_timed else fn(), provider.sync)
        out = fn()
    except NotImplementedError as e:
        pytest.skip(f"{provider.provider}/{variant}/{precision} unsupported on {device}: {e}")
    if provider.sync is not None:
        provider.sync()

    result = out[0] if provider.self_timed else out
    got = providers.to_csr(result)
    status = "ok" if prec.validate_spgemm(got, spgemm_case.matrix, precision) else "incorrect"
    if bench_cfg.strict_correctness:
        assert status == "ok", f"{provider.provider}/{variant}/{precision} failed SpGEMM oracle"

    timings = {"compile_ms": compile_ms}
    warmup, rounds = bench_cfg.warmup_spgemm, bench_cfg.rounds_spgemm
    if provider.self_timed:
        # Kernel reports its own transfer-free time; one wall round suffices.
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
            framework=provider.framework, provider=provider.provider, target="spgemm", variant=variant,
            case=spgemm_case, device=device, problem={}, knobs={"precision": precision},
            total_flops=flops_spgemm(spgemm_case.intermediate_products), timings=timings, status=status,
        ),
    )
