"""beNNch RSNN benchmarks — balanced E-I networks on brainstate/brainpy, via pytest-benchmark.

The workloads are the Vogels-Abbott / Brette-2007 ``coba``/``cuba`` balanced networks
and a ``connectome`` variant (connectome adjacency as the recurrent operator). Each is
measured with the beNNch phase-resolved methodology (rsnn/phases.py): the pytest-benchmark
hot loop times the propagation loop (→ real-time factor + event rates), while build/compile
and the per-phase (update / deliver) microbenchmarks are recorded alongside.

Run:
    pytest connectome_dataset/benchmarks/jax/brainstate/ -v -s
    pytest connectome_dataset/benchmarks/jax/brainstate/ --rsnn-model cuba --rsnn-scale 1,2,4
    pytest connectome_dataset/benchmarks/jax/brainstate/ --rsnn-model connectome --graph mice_column_v1
"""
from __future__ import annotations

import jax
import pytest

from connectome_dataset.benchmarks.config import BENNCH_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.metrics import flops_rsnn_step

# Optional deps: the beNNch models need the full brain* simulator stack. Skip cleanly
# (rather than error) where it is not installed.
pytest.importorskip("brainstate")
pytest.importorskip("brainpy")
pytest.importorskip("braintools")

from connectome_dataset.benchmarks.rsnn import models, phases  # noqa: E402


@pytest.mark.parametrize("model", ["cuba", "coba"])
def test_bennch_metrics_sane(model):
    """Fast CPU validation: the network runs and the beNNch metrics are well-formed."""
    sim = phases.build_sim(model, scale=0.05, timesteps=100, seed=0)
    ref = phases.reference_run(sim)
    phase_ms = phases.phase_timings(sim, ref, reps=1)
    assert sim.spec.num == 200 and sim.spec.nnz == 200 * models.CONN_PER_PRE
    assert ref["n_spikes"] > 0  # a balanced network fires
    assert ref["n_syn_events"] == ref["n_spikes"] * sim.spec.conn_per_pre
    assert 0.0 < ref["firing_rate_hz"] < 1000.0
    assert phase_ms["phase_full_ms"] > 0.0
    assert phase_ms["phase_deliver_ms"] > 0.0 and phase_ms["phase_update_ms"] > 0.0
    assert phase_ms["phase_communicate_ms"] == 0.0


def test_rsnn_bennch(benchmark, request, rsnn_bennch, rsnn_timesteps, bench_cfg):
    model, scale = rsnn_bennch
    # Only the connectome variant needs a catalog matrix; coba/cuda are self-contained.
    matrix = request.getfixturevalue("rsnn_case").matrix if model == "connectome" else None
    sim = phases.build_sim(
        model, scale=scale, timesteps=rsnn_timesteps, seed=BENNCH_DEFAULTS.seed, matrix=matrix,
    )
    ref = phases.reference_run(sim)
    phase_ms = phases.phase_timings(sim, ref, reps=BENNCH_DEFAULTS.phase_reps)

    spec = sim.spec
    case = models.case_from_spec(spec)
    total_flops = flops_rsnn_step(spec.nnz, spec.num) * rsnn_timesteps

    def run():
        return jax.block_until_ready(sim.run())

    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_rsnn if bench_cfg.warmup_rsnn else BENNCH_DEFAULTS.warmup,
        rounds=bench_cfg.rounds_rsnn if bench_cfg.rounds_rsnn else BENNCH_DEFAULTS.rep,
        extra_info=base_extra_info(
            # variant = the balanced-network model (within-provider workload kernel).
            framework="jax", provider="brainstate", target="rsnn", variant=model,
            case=case, device=jax.default_backend(),
            problem={"timesteps": rsnn_timesteps, "batch_size": 1, "pass": "fwd"},
            knobs={"model": model, "scale": scale},
            total_flops=total_flops,
            timings={"compile_ms": sim.compile_ms, "build_ms": sim.build_ms, **phase_ms},
            extra={
                "dt_ms": sim.dt_ms,
                "n_spikes": ref["n_spikes"],
                "n_syn_events": ref["n_syn_events"],
                "firing_rate_hz": ref["firing_rate_hz"],
                "scale": scale,
                "seed": BENNCH_DEFAULTS.seed,
            },
        ),
    )
