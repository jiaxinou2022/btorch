"""Native-NEST macaque multi-area model — reference RSNN benchmark.

Records on the RSNN track as ``nest.native.rsnn.{microcircuit,multiarea}`` — the reference
simulation the btorch port is checked against. Sizes come from ``MULTIAREA_DEFAULTS`` and are
small by design; full scale (4.13M neurons) is opt-in via the config, never built here.

Run:  pytest connectome_dataset/benchmarks/nest/native/test_microcircuit.py -v -s
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from connectome_dataset.benchmarks.config import MULTIAREA_DEFAULTS as D
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.cortical_network import (
    MULTIAREA_MESOSCALE_ROOT, microcircuit_spec, multiarea_spec,
)

_TEST_TIMESTEPS = 300


def _spec(variant: str):
    if variant == "microcircuit":
        return microcircuit_spec(area=D.microcircuit_area, n_scaling=D.microcircuit_scaling,
                                 k_scaling=D.k_scaling)
    return multiarea_spec(n_scaling=D.multiarea_scaling, k_scaling=D.k_scaling)


@pytest.mark.parametrize("variant", ["microcircuit", "multiarea"])
def test_nest_ground_state(benchmark, bench_cfg, variant):
    pytest.importorskip("nest")
    if not (MULTIAREA_MESOSCALE_ROOT / "multiarea_mesoscale.npz").exists():
        pytest.skip("multiarea mesoscale not staged (see datasets/multiarea/README.md)")

    from .microcircuit import build_and_run

    spec = _spec(variant)

    def run():
        return build_and_run(spec, timesteps=_TEST_TIMESTEPS, dt_ms=D.dt_ms,
                             warmup_ms=D.warmup_ms, seed=D.seed)

    pop_rates = run()
    mean_hz = float(pop_rates.mean())
    ok = 0.0 < mean_hz < 50.0 and bool((pop_rates > 0).any())
    if bench_cfg.strict_correctness:
        assert ok, f"NEST {variant} ground state implausible: mean {mean_hz:.1f} Hz"

    n = spec.n_neurons
    nnz = int(np.round(spec.K_int.sum(1) @ spec.N))
    case = SimpleNamespace(
        case_id=f"multiarea_{variant}", graph_id=f"multiarea_{variant}",
        n=n, nnz=nnz, density=nnz / (n * n) if n else 0.0, matrix=None, metadata={},
    )
    run_pedantic(
        benchmark, run, warmup=0, rounds=1,
        extra_info=base_extra_info(
            framework="nest", provider="native", target="rsnn", variant=variant,
            case=case, device="cpu",
            problem={"timesteps": _TEST_TIMESTEPS, "batch_size": 1, "pass": "fwd"},
            knobs={"n_scaling": spec.n_scaling, "k_scaling": spec.k_scaling},
            status="ok" if ok else "incorrect",
            extra={
                "model": "multiarea_iaf_psc_exp", "n_populations": spec.n_pop,
                "mean_rate_hz": mean_hz, "dt_ms": D.dt_ms, "n_scaling": spec.n_scaling,
            },
        ),
    )
