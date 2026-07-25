"""btorch macaque multi-area model — RSNN benchmark (microcircuit + multiarea variants).

Records on the RSNN track as ``torch.btorch.rsnn.{microcircuit,multiarea}``. Both variants
are the same iaf_psc_exp network at different scale; sizes come from ``MULTIAREA_DEFAULTS``
and are deliberately small (full scale is 4.13M neurons / ~24e9 synapses — never built here).
The btorch port is cross-checked against the native-NEST reference (per-population rate
correlation r ~ 0.94); see connectome_dataset/benchmarks/nest/native/microcircuit.py.

Run:  pytest connectome_dataset/benchmarks/torch/btorch/test_microcircuit.py -v -s
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from connectome_dataset.benchmarks.config import MULTIAREA_DEFAULTS as D
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.cortical_network import instantiate_connectivity, microcircuit_spec, multiarea_spec
from connectome_dataset.metrics import flops_spmspv

_TEST_TIMESTEPS = 300  # short measured window keeps the benchmark quick


def _spec(variant: str):
    if variant == "microcircuit":
        return microcircuit_spec(area=D.microcircuit_area, n_scaling=D.microcircuit_scaling,
                                 k_scaling=D.k_scaling)
    return multiarea_spec(n_scaling=D.multiarea_scaling, k_scaling=D.k_scaling)


@pytest.mark.parametrize("variant", ["microcircuit", "multiarea"])
def test_multiarea_ground_state(benchmark, bench_cfg, variant):
    pytest.importorskip("btorch")
    if not _mesoscale_present():
        pytest.skip("multiarea mesoscale not staged (see datasets/multiarea/README.md)")

    import torch

    from .microcircuit import CorticalMicrocircuitModel

    device = bench_cfg.device or "cpu"
    spec = _spec(variant)
    W = instantiate_connectivity(spec, seed=D.seed)
    model = CorticalMicrocircuitModel(spec, W, device=device)

    def run():
        rates = model.simulate(timesteps=_TEST_TIMESTEPS, dt_ms=D.dt_ms,
                               warmup_ms=D.warmup_ms, seed=D.seed)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return rates

    rates = run().cpu().numpy()
    active_ratio = float((rates > 0).mean())
    mean_hz = float(rates.mean())
    # sanity: cortical ground state is asynchronous-irregular at a few Hz, not silent/saturated.
    ok = 0.0 < mean_hz < 50.0 and active_ratio > 0.0
    if bench_cfg.strict_correctness:
        assert ok, f"{variant} ground state implausible: mean {mean_hz:.1f} Hz, active {active_ratio:.2f}"

    n = W.shape[0]
    case = SimpleNamespace(
        case_id=f"multiarea_{variant}", graph_id=f"multiarea_{variant}",
        n=n, nnz=int(W.nnz), density=W.nnz / (n * n), matrix=None, metadata={},
    )
    run_pedantic(
        benchmark, run, warmup=0, rounds=1,
        extra_info=base_extra_info(
            framework="torch", provider="btorch", target="rsnn", variant=variant,
            case=case, device=device,
            problem={"timesteps": _TEST_TIMESTEPS, "batch_size": 1, "pass": "fwd"},
            knobs={"precision": "fp32", "n_scaling": spec.n_scaling, "k_scaling": spec.k_scaling},
            total_flops=flops_spmspv(int(W.nnz)) * _TEST_TIMESTEPS,
            status="ok" if ok else "incorrect",
            extra={
                "model": "multiarea_iaf_psc_exp", "n_populations": spec.n_pop,
                "mean_rate_hz": mean_hz, "active_ratio": active_ratio,
                "dt_ms": D.dt_ms, "n_scaling": spec.n_scaling,
            },
        ),
    )


def _mesoscale_present() -> bool:
    from connectome_dataset.cortical_network import MULTIAREA_MESOSCALE_ROOT
    return (MULTIAREA_MESOSCALE_ROOT / "multiarea_mesoscale.npz").exists()
