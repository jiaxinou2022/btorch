"""Native-NEST macaque multi-area model / cortical microcircuit (the reference sim).

Builds the network the way the published model does — ``iaf_psc_exp`` neurons connected
population-pair-by-pair with NEST's native ``fixed_indegree`` rule from the mesoscale
in-degrees ``K``, driven by a per-population Poisson background — so NEST is an
independent reference for the btorch port (torch/btorch/microcircuit.py), not a replay
of the same random draws. Scale is set by the :class:`CorticalNetworkSpec`, so the
reference never builds more than the requested (small) network.
"""
from __future__ import annotations

import numpy as np

from connectome_dataset.cortical_network import CorticalNetworkSpec


def build_and_run(spec: CorticalNetworkSpec, *, timesteps: int, dt_ms: float = 0.1,
                  warmup_ms: float = 100.0, seed: int = 0, n_threads: int = 4) -> np.ndarray:
    """Simulate the spec in NEST; return per-population firing rate (Hz).

    The measured window excludes a ``warmup_ms`` transient so the returned rates are the
    stationary background activity — directly comparable to the btorch model's readout.
    """
    import nest

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt_ms, "local_num_threads": int(n_threads), "rng_seed": seed + 1})

    p = spec.neuron
    neuron_params = dict(
        C_m=p["C_m"], tau_m=p["tau_m"], E_L=p["E_L"], V_th=p["V_th"], V_reset=p["V_reset"],
        t_ref=p["t_ref"], tau_syn_ex=p["tau_syn_ex"], tau_syn_in=p["tau_syn_in"], V_m=p["E_L"],
    )
    pops = [nest.Create("iaf_psc_exp", int(n), params=neuron_params) for n in spec.N]

    delay_e, delay_i = spec.delay["delay_e"], spec.delay["delay_i"]
    for t in range(spec.n_pop):
        for s in range(spec.n_pop):
            indeg = int(round(spec.K_int[t, s]))
            if indeg == 0 or spec.N[s] == 0:
                continue
            w = float(spec.W_int[t, s])
            nest.Connect(pops[s], pops[t],
                         {"rule": "fixed_indegree", "indegree": indeg},
                         {"weight": w, "delay": delay_e if w >= 0 else delay_i})

    # Per-population Poisson background: K_ext external synapses at bg_rate each.
    for t in range(spec.n_pop):
        if spec.N[t] == 0 or spec.K_ext[t] == 0:
            continue
        pg = nest.Create("poisson_generator", params={"rate": float(spec.K_ext[t] * spec.bg_rate)})
        nest.Connect(pg, pops[t], "all_to_all", {"weight": float(spec.W_ext[t]), "delay": delay_e})

    recorders = [nest.Create("spike_recorder") for _ in range(spec.n_pop)]
    for rec, pop in zip(recorders, pops):
        nest.Connect(pop, rec)

    nest.Simulate(warmup_ms)  # transient, discarded
    for rec in recorders:
        rec.n_events = 0
    nest.Simulate(timesteps * dt_ms)

    window_s = timesteps * dt_ms * 1e-3
    return np.array([rec.n_events / (int(n) * window_s) if n else 0.0
                     for rec, n in zip(recorders, spec.N)])
