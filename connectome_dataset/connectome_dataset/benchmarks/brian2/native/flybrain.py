"""Native-brian2 whole-brain Drosophila LIF model (Shiu et al. 2024) — the reference sim.

A direct port of the upstream ``model.py`` onto the ``flywire_783`` connectome: the exact
brian2 network the btorch ``FlyBrainModel`` is validated against (per-neuron rate correlation
r ~ 0.999). brian2 is a benchmark *framework* (peer of jax/torch/nest); this is its native
Shiu-model provider.

Dynamics (verbatim from upstream), with ``v_0 == v_rst == -52 mV``::

    dv/dt = (v_0 - v + g) / t_mbr    (unless refractory)
    dg/dt = -g / tau                 (unless refractory)
    spike when v > v_th  ->  v = v_rst, g = 0    (refractory t_rfc)
    synapse:  on presynaptic spike (delay t_dly),  g += w = (Excitatory x Connectivity) * w_syn

Activation drives target neurons with a Poisson input to ``v`` (weight ``w_syn * f_poi``,
refractory 0) exactly as upstream; silencing zeros a neuron's incoming and outgoing synapses.
"""
from __future__ import annotations

from textwrap import dedent

import numpy as np
import scipy.sparse as sp

# Shiu et al. (2024) constants (mV, ms, Hz) — the same values as the btorch FlyBrainParams.
_P = dict(v_0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2,
          t_dly=1.8, w_syn=0.275, f_poi=250.0)

_EQS = dedent("""
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau              : volt (unless refractory)
    rfc                           : second
""")


def _silenced(W: sp.csr_matrix, silence_idx: np.ndarray) -> sp.csr_matrix:
    """Zero every synapse to and from the silenced neurons (adjacency row + column)."""
    if silence_idx is None or len(silence_idx) == 0:
        return W
    keep = np.ones(W.shape[0], dtype=np.float32)
    keep[silence_idx] = 0.0
    d = sp.diags(keep)
    return (d @ W @ d).tocsr()


def build_and_run(W: sp.csr_matrix, activate_idx: np.ndarray, silence_idx: np.ndarray | None = None,
                  *, t_run_ms: float, n_run: int, r_poi_hz: float = 150.0, dt_ms: float = 0.1,
                  bg_rate_hz: float = 3500.0, bg_weight: float = 0.30, seed: int = 0) -> np.ndarray:
    """Simulate the Shiu model on connectome ``W`` in brian2; return per-neuron rate (Hz).

    ``W`` is the signed connectome (rows=presynaptic, cols=postsynaptic; ``Excitatory x
    Connectivity``). ``bg_rate_hz`` / ``bg_weight`` add the same tonic background as the btorch
    model (an independent Poisson drive per neuron into the synaptic variable ``g``, so it is
    smoothly filtered and cannot overshoot; ``bg_rate_hz=0`` disables). brian2's ``(unless
    refractory)`` equations freeze ``v`` during the refractory period natively — which is the
    behaviour the btorch port emulates by masking input to refractory neurons. Rates are
    averaged over ``n_run`` Poisson trials.
    """
    from brian2 import (Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses,
                        defaultclock, mV, ms, seed as bseed, start_scope)

    coo = _silenced(W, silence_idx).tocoo()
    n = W.shape[0]
    ns = {k: _P[k] * (mV if k.startswith("v") else ms) for k in ("v_0", "v_rst", "v_th", "t_mbr", "tau")}
    counts = np.zeros(n)
    for trial in range(n_run):
        start_scope()
        defaultclock.dt = dt_ms * ms
        bseed(seed + trial)
        neu = NeuronGroup(n, _EQS, method="exact", threshold="v > v_th",
                          reset="v = v_rst; g = 0*mV", refractory="rfc", namespace=ns)
        neu.v = _P["v_0"] * mV
        neu.rfc = _P["t_rfc"] * ms
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=_P["t_dly"] * ms, namespace=ns)
        syn.connect(i=coo.row, j=coo.col)
        syn.w = (coo.data * _P["w_syn"]) * mV
        pois = []
        if bg_rate_hz > 0.0:  # tonic background into g (smooth, membrane-filtered)
            pois.append(PoissonInput(neu, "g", 1, bg_rate_hz * Hz, weight=bg_weight * mV))
        for i in activate_idx:
            pois.append(PoissonInput(neu[int(i):int(i) + 1], "v", 1, r_poi_hz * Hz,
                                     weight=_P["w_syn"] * _P["f_poi"] * mV))
            neu.rfc[int(i)] = 0 * ms  # Poisson targets fire without refractory (as upstream)
        mon = SpikeMonitor(neu)
        Network(neu, syn, mon, *pois).run(t_run_ms * ms)
        counts += np.asarray(mon.count)
    return counts / (n_run * t_run_ms * 1e-3)
