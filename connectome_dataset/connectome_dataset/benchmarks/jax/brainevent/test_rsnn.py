"""RSNN benchmarks — JAX / brainevent, via pytest-benchmark.

Run:
    pytest connectome_dataset/benchmarks/jax/brainevent/ -v -s
    pytest connectome_dataset/benchmarks/jax/brainevent/test_rsnn.py -k jax_bcoo
"""
from __future__ import annotations

import jax
import pytest

from connectome_dataset.benchmarks.cases import SpmvCase
from connectome_dataset.benchmarks.config import RSNN_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_rsnn_step

from .rsnn import BRAINEVENT_AVAILABLE, dense_matrix, make_fn

_skip_brainevent = pytest.mark.skipif(not BRAINEVENT_AVAILABLE, reason="brainevent not installed")

BACKEND_PARAMS = [
    pytest.param("brainevent", marks=_skip_brainevent),
    "jax_bcoo",
]

# backend -> (provider, variant) in the identity tuple; keeps provider naming consistent
# with the SpMV suite (jax-native library vs the brainevent library).
_BACKEND = {"brainevent": ("brainevent", "csr_event"), "jax_bcoo": ("jax_native", "bcoo")}
_DT_MS = 0.5  # LIF integration step used by the jax RSNN kernels (0.5 ms)


def _synthetic_case(mat, n: int) -> SpmvCase:
    # Framework-namespaced id: this is a 4%-sparse random matrix, a different
    # workload from btorch's fully-dense n*n case — they must not share a cell.
    return SpmvCase(
        case_id=f"syn_jax_rsnn_{n}", graph_id=f"syn_jax_{n}", replicate=1,
        matrix=mat, n=n, nnz=mat.nnz, density=mat.nnz / (n * n),
    )


def _run_rsnn(benchmark, bench_cfg, *, case, backend, timesteps, batch_size, kind):
    fn = make_fn(case.matrix, backend=backend, timesteps=timesteps, batch_size=batch_size)

    def run():
        return jax.block_until_ready(fn())

    compile_ms = measure_compile_ms(run)
    provider, variant = _BACKEND.get(backend, (backend, "default"))
    total_flops = flops_rsnn_step(case.nnz, case.n) * timesteps * batch_size
    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_rsnn,
        rounds=bench_cfg.rounds_rsnn,
        extra_info=base_extra_info(
            framework="jax",
            provider=provider,
            target="rsnn",
            variant=variant,
            case=case,
            device=jax.default_backend(),
            problem={"timesteps": timesteps, "batch_size": batch_size, "pass": "fwd"},
            knobs={"case": kind},
            total_flops=total_flops,
            timings={"compile_ms": compile_ms},
            extra={"dt_ms": _DT_MS},
        ),
    )


@pytest.mark.parametrize("n", RSNN_DEFAULTS.dense_sizes)
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_rsnn_dense(benchmark, bench_cfg, backend, n):
    if bench_cfg.mode not in ("dense", "all"):
        pytest.skip("--mode excludes dense")
    timesteps = bench_cfg.timesteps or RSNN_DEFAULTS.timesteps
    batch_size = bench_cfg.batch_size or RSNN_DEFAULTS.batch_size
    _run_rsnn(benchmark, bench_cfg, case=_synthetic_case(dense_matrix(n), n),
              backend=backend, timesteps=timesteps, batch_size=batch_size, kind="dense")


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_rsnn_sparse(benchmark, rsnn_case, bench_cfg, backend):
    if bench_cfg.mode not in ("sparse", "all"):
        pytest.skip("--mode excludes sparse")
    _run_rsnn(benchmark, bench_cfg, case=rsnn_case, backend=backend,
              timesteps=rsnn_case.timesteps, batch_size=rsnn_case.batch_size, kind="sparse")
