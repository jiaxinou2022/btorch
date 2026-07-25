"""btorch implementation of the macaque multi-area model (Schmidt et al. 2018).

The multi-area model and its single-area cortical microcircuit share one dynamics:
``iaf_psc_exp`` leaky integrate-and-fire neurons with exponential (current-based)
synapses, driven by a Poisson background. This module runs that network in btorch on a
connectivity instantiated at any scale by :mod:`connectome_dataset.cortical_network` —
so ``microcircuit`` (one area, 8 populations) and ``multiarea`` (all 254 populations)
are the same code at different scale, never the untestable 24-billion-synapse full size.

``iaf_psc_exp`` maps exactly onto btorch ``LIF``: since ``E_L == V_reset == -65 mV`` the
membrane leaks toward its reset, so ``LIF(v_reset=-65, tau=tau_m, c_m=C_m)`` driven by
the synaptic current ``I`` (pA) reproduces ``dV/dt = -(V-E_L)/tau_m + I/C_m``. Recurrent
current is btorch ``SparseConn`` (signed weights in pA); the background is a per-neuron
Poisson charge each step.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from btorch.models import environ
from btorch.models.linear import SparseConn
from btorch.models.neurons.lif import LIF

from connectome_dataset.cortical_network import CorticalNetworkSpec, instantiate_connectivity


class CorticalMicrocircuitModel(nn.Module):
    """LIF (``iaf_psc_exp``) cortical network with Poisson background, in btorch.

    Args:
        spec: population-level description (already scaled) from
            :func:`connectome_dataset.cortical_network.microcircuit_spec` or
            ``multiarea_spec``.
        connectivity: optional pre-instantiated neuron-level adjacency (rows=pre,
            cols=post, signed pA). Built from ``spec`` if omitted.
    """

    def __init__(
        self,
        spec: CorticalNetworkSpec,
        connectivity: sp.csr_matrix | None = None,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.spec = spec
        self.device = device
        self.dtype = dtype
        W = connectivity if connectivity is not None else instantiate_connectivity(spec, seed=seed)
        self.n = W.shape[0]
        self.pop_of_neuron = torch.as_tensor(spec.population_of_neuron(), device=device)

        p = spec.neuron
        self.neuron = LIF(
            self.n,
            v_threshold=p["V_th"], v_reset=p["V_reset"],
            c_m=p["C_m"], tau=p["tau_m"], tau_ref=p["t_ref"],
            device=device, dtype=dtype,
        )
        self.conn = SparseConn(
            sp.coo_array(W.tocoo()), enforce_dale=False,
            sparse_backend="native", device=device, dtype=dtype,
        )
        # Per-neuron background: expected external charge rate (pA/ms) and its Poisson mean.
        pop = spec.population_of_neuron()
        self._w_ext = torch.as_tensor(spec.W_ext[pop], device=device, dtype=dtype)
        self._k_ext = torch.as_tensor(spec.K_ext[pop], device=device, dtype=dtype)

    @torch.no_grad()
    def simulate(self, timesteps: int, *, dt_ms: float = 0.1, warmup_ms: float = 100.0,
                 seed: int = 0) -> torch.Tensor:
        """Run the network; return per-neuron firing rate (Hz) over the measured window.

        A ``warmup_ms`` transient is simulated but not counted, so the reported rate is
        the network's stationary background activity.
        """
        p = self.spec
        decay = float(np.exp(-dt_ms / p.neuron["tau_syn_ex"]))
        delay_steps = max(1, int(round(p.delay["delay_e"] / dt_ms)))
        bg_lambda = self._k_ext * p.bg_rate * dt_ms * 1e-3  # mean external spikes / neuron / step
        warmup_steps = int(round(warmup_ms / dt_ms))

        environ.set(dt=dt_ms)
        self.neuron.init_state(batch_size=1, device=self.device, dtype=self.dtype)
        self.neuron.v.fill_(p.neuron["E_L"])
        gen = torch.Generator(device=self.device).manual_seed(seed)
        g = torch.zeros(1, self.n, device=self.device, dtype=self.dtype)
        buffer: deque[torch.Tensor] = deque(
            (torch.zeros(1, self.n, device=self.device, dtype=self.dtype)
             for _ in range(delay_steps)), maxlen=delay_steps)
        counts = torch.zeros(1, self.n, device=self.device, dtype=self.dtype)

        for step in range(warmup_steps + timesteps):
            ext = torch.poisson(bg_lambda.expand(1, self.n), generator=gen) * self._w_ext
            g = g * decay + self.conn(buffer[0]) + ext
            spk = self.neuron(g)
            buffer.append(spk)
            if step >= warmup_steps:
                counts += spk
        return (counts / (timesteps * dt_ms * 1e-3)).squeeze(0)

    def population_rates(self, neuron_rates: torch.Tensor) -> dict[str, float]:
        """Mean firing rate (Hz) per population label."""
        pop = self.pop_of_neuron
        out = {}
        for i, label in enumerate(self.spec.labels):
            sel = pop == i
            out[label] = float(neuron_rates[sel].mean()) if sel.any() else 0.0
        return out
