"""Generic benchmark for out-of-tree SpMSpV providers discovered via the registry.

Any provider registered under ``connectome_bench.spmspv`` (or via
``providers.register_spmspv``) computes ``y = A·x`` with a sparse operand ``x`` over the
shared cases, swept across vector sparsities, and is checked against the SciPy oracle.
This is the target VDHA targets and the one that maps onto event-driven spike delivery.
When no external provider is installed, the parametrization collapses to a single skip.
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks import providers
from connectome_dataset.benchmarks.config import SPMSPV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmspv

_DISCOVERED = providers.discover_spmspv()
_CASES = [(p, v) for p in _DISCOVERED for v in p.variants]
_PARAMS = _CASES or [pytest.param((None, None), marks=pytest.mark.skip(reason="no external SpMSpV providers registered"))]


def _ids(pv):
    p, v = pv
    return "none" if p is None else f"{p.framework}.{p.provider}.{v}"


def _products_touched(matrix, x_indices) -> int:
    """Scalar multiply-adds actually formed: sum of A's column nnz over active x indices."""
    col_nnz = np.diff(matrix.tocsc().indptr)
    return int(col_nnz[np.asarray(x_indices, dtype=np.int64)].sum())


@pytest.mark.parametrize("provider_variant", _PARAMS, ids=_ids)
@pytest.mark.parametrize("vector_sparsity", SPMSPV_DEFAULTS.vector_sparsities)
def test_external_spmspv(benchmark, spmspv_case, bench_cfg, provider_variant, vector_sparsity, precision):
    provider, variant = provider_variant
    if provider.supports is not None and not provider.supports(variant, precision):
        pytest.skip(f"{provider.provider}/{variant} does not support {precision}")

    device = bench_cfg.device or "cpu"
    x_idx, x_val = prec.make_sparse_vector(spmspv_case.n, vector_sparsity, seed=SPMSPV_DEFAULTS.seed)
    try:
        fn = provider.make_fn(
            spmspv_case.matrix, variant=variant, device=device,
            precision=precision, x_indices=x_idx, x_values=x_val,
        )
        compile_ms = measure_compile_ms(lambda: fn()[0] if provider.self_timed else fn(), provider.sync)
        out = fn()
    except NotImplementedError as e:
        pytest.skip(f"{provider.provider}/{variant}/{precision} unsupported on {device}: {e}")
    if provider.sync is not None:
        provider.sync()

    result = out[0] if provider.self_timed else out
    got = providers.to_numpy(result)
    status = "ok" if prec.validate_spmspv(got, spmspv_case.matrix, x_idx, x_val, precision) else "incorrect"
    if bench_cfg.strict_correctness:
        assert status == "ok", f"{provider.provider}/{variant}/{precision} failed SpMSpV oracle"

    timings = {"compile_ms": compile_ms}
    warmup, rounds = bench_cfg.warmup_spmspv, bench_cfg.rounds_spmspv
    if provider.self_timed:
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
            framework=provider.framework, provider=provider.provider, target="spmspv", variant=variant,
            case=spmspv_case, device=device,
            problem={"vector_sparsity": float(vector_sparsity), "vector_nnz": int(x_idx.size)},
            knobs={"precision": precision},
            total_flops=flops_spmspv(_products_touched(spmspv_case.matrix, x_idx)),
            timings=timings, status=status,
        ),
    )
