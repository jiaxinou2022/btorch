"""beNNch-style phase-resolved measurement of the balanced E-I networks.

Adopts the beNNch methodology (Albers et al. 2022, *A modular workflow for
performance benchmarking of neuronal network simulations*, Front. Neuroinform.):

- separate **network construction/build** from **state propagation** (the hot loop);
- decompose one propagation step into the beNNch phases. For a clock-driven,
  single-device XLA simulation the cleanly separable phases are ``update`` (LIF
  neuron dynamics) and ``deliver`` (event-driven synaptic delivery: the sparse
  SpMV + synapse filtering). ``gather`` (spike readout) is trivial and folded into
  ``update``; ``communicate`` is zero on a single device (no MPI). The split is a
  microbenchmark of each phase at the network's operating firing rate — XLA fuses
  the full step, so the phases are timed as isolated sub-loops rather than carved
  out of the fused kernel;
- metrics: real-time factor ``T_wall / T_model``, spikes/s, synaptic-events/s;
- time only the hot loop, repeat over several seeds, report full provenance.

The primary hot-loop timing is done by pytest-benchmark in the leaf; this module
provides the one-time build/compile timing, the reference run (spike + event
counts for the rate metrics + correctness), and the per-phase microbenchmarks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import brainstate
import brainunit as u
import jax
import numpy as np

from .models import DT, INPUT_CURRENT, NetworkSpec, build_network

_DT_MS = float(DT.to_decimal(u.ms))


@dataclass(slots=True)
class BuiltSim:
    net: Any
    spec: NetworkSpec
    run: Callable[[], Any]  # full T-step propagation loop (the pytest-benchmark body)
    timesteps: int
    dt_ms: float
    build_ms: float
    compile_ms: float
    _kwargs: dict[str, Any]  # construction args, for rebuilding isolated phase nets


def _for_loop(net, step: Callable, timesteps: int):
    @brainstate.transform.jit
    def loop():
        with brainstate.environ.context(dt=DT):
            times = u.math.arange(0.0 * u.ms, timesteps * DT, brainstate.environ.get_dt())
            brainstate.transform.for_loop(step, times)
        return net.n_spike_exc.value, net.n_spike_inh.value

    return loop


def build_sim(
    model: str, *, scale: float = 1.0, timesteps: int = 1000, seed: int = 0, matrix=None
) -> BuiltSim:
    """Construct + compile a network; time the build and the first (compiling) call."""
    kwargs = dict(model=model, scale=scale, seed=seed, matrix=matrix)
    t0 = time.perf_counter()
    net, spec = build_network(model, scale=scale, seed=seed, matrix=matrix)
    brainstate.nn.init_all_states(net)
    build_ms = (time.perf_counter() - t0) * 1000.0

    run = _for_loop(net, lambda t: net.update(t, INPUT_CURRENT), timesteps)
    t0 = time.perf_counter()
    jax.block_until_ready(run())
    compile_ms = (time.perf_counter() - t0) * 1000.0
    return BuiltSim(net, spec, run, timesteps, _DT_MS, build_ms, compile_ms, kwargs)


def reference_run(sim: BuiltSim) -> dict[str, Any]:
    """Reset to a fresh state, run once, and count spikes + synaptic events.

    These drive the spikes/s and synaptic-events/s rates (ingest divides the counts
    by the measured hot-loop time) and give a mean firing rate to validate against.
    """
    brainstate.nn.reset_all_states(sim.net)
    n_exc, n_inh = jax.block_until_ready(sim.run())
    n_exc, n_inh = int(n_exc), int(n_inh)
    n_spikes = n_exc + n_inh
    spec = sim.spec
    if spec.model == "connectome":
        avg_out = spec.nnz / spec.num if spec.num else 0.0
        n_syn_events = int(round(n_spikes * avg_out))
    else:
        n_syn_events = (n_exc + n_inh) * spec.conn_per_pre
    model_time_s = sim.timesteps * sim.dt_ms * 1e-3
    firing_rate_hz = n_spikes / spec.num / model_time_s if spec.num and model_time_s else 0.0
    return {
        "n_spikes": n_spikes,
        "n_spikes_exc": n_exc,
        "n_spikes_inh": n_inh,
        "n_syn_events": n_syn_events,
        "model_time_s": model_time_s,
        "firing_rate_hz": firing_rate_hz,
    }


def _time_loop(loop: Callable, *, reps: int) -> float:
    """Min wall time (ms) over ``reps`` calls, after one warmup."""
    jax.block_until_ready(loop())
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(loop())
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def phase_timings(sim: BuiltSim, ref: dict[str, Any], *, reps: int = 3) -> dict[str, float]:
    """Per-phase hot-loop microbenchmarks (beNNch update / deliver decomposition).

    ``deliver`` and ``update`` are timed as isolated ``timesteps``-long sub-loops.
    The spike vector driving ``deliver`` is a fixed random pattern at the measured
    firing probability, so the event-driven cost is exercised at the operating point.
    """
    spec = sim.spec
    p = ref["firing_rate_hz"] * sim.dt_ms * 1e-3  # per-step spike probability
    rng = np.random.default_rng(spec.seed)
    spk = jax.numpy.asarray(rng.random(spec.num) < max(p, 1e-4))

    # Isolated phases run on *fresh* network instances: re-jitting one net's methods
    # into several interleaved compiled loops leaks tracers through brainstate's
    # stateful modules, so each phase gets its own net.
    def _phase_loop(step_factory):
        net, _ = build_network(**sim._kwargs)
        brainstate.nn.init_all_states(net)
        return _for_loop(net, step_factory(net), sim.timesteps)

    def deliver_factory(net):
        def step(t):
            with brainstate.environ.context(t=t):
                net._deliver(spk)
        return step

    def neuron_factory(net):
        def step(t):
            with brainstate.environ.context(t=t):
                net._neuron(INPUT_CURRENT)
        return step

    full = _time_loop(sim.run, reps=reps)
    deliver_ms = _time_loop(_phase_loop(deliver_factory), reps=reps)
    update_ms = _time_loop(_phase_loop(neuron_factory), reps=reps)
    return {
        "phase_full_ms": full,
        "phase_deliver_ms": deliver_ms,
        "phase_update_ms": update_ms,
        "phase_gather_ms": 0.0,       # spike readout, fused into update
        "phase_communicate_ms": 0.0,  # single device, no inter-rank exchange
    }
