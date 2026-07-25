"""btorch reimplementation of the Shiu et al. whole-brain *Drosophila* LIF model.

A faithful, brian2-free port of ``philshiu/Drosophila_brain_model`` (Shiu et al.,
Nature 2024) onto btorch primitives and the ``flywire_783`` connectome. The neuron
and synapse constants are lifted verbatim from the upstream ``model.py``:

    dv/dt = (v_0 - v + g) / t_mbr      leaky integrate-and-fire membrane
    dg/dt = -g / tau_syn               exponential synaptic conductance
    spike when v > v_th  ->  v = v_rst, g = 0    (refractory t_rfc)
    synapse:  on presynaptic spike (after delay t_dly),  g += w
              w = (Excitatory x Connectivity) * w_syn      (signed: <0 inhibitory)

Because ``v_0 == v_rst == -52 mV`` here, the membrane leaks toward its own reset,
which is exactly btorch ``LIF`` with ``v_reset = -52, tau = c_m = t_mbr`` driven by
current ``I = g``. The recurrent ``g += (spikes @ W)`` is btorch ``SparseConn``.

**Activation** models optogenetics: upstream drives target neurons with a Poisson
input whose per-event kick (``w_syn * f_poi = 68.75 mV``) dwarfs the 7 mV
spike gap while their refractory period is set to zero — i.e. every Poisson event
forces a spike. We reproduce that mechanism directly and dt-robustly by emitting a
Poisson spike train at ``r_poi`` from the activated neurons, which then propagates
through the recurrent connectivity. **Silencing** zeros every synapse to and from a
neuron (its adjacency row and column), as upstream does.

**Tonic background** (``bg_rate`` / ``bg_weight``, on by default) is an extension beyond
the stimulus-only upstream model: an independent per-neuron Poisson drive injected through
the synaptic conductance ``g`` (never straight into ``v``), tuned to the diffusion regime so
it is a smooth near-DC depolarisation that holds the brain at a low asynchronous-irregular
baseline (~8 Hz) instead of silent. Two safeguards keep ``v`` bounded without any clamp on it:
``hard_reset`` (the upstream ``v = v_rst``), and — since btorch's LIF integrates ``v`` every
step whereas the Shiu equations are ``(unless refractory)`` — masking synaptic input to
refractory neurons, so a strong recurrent hit can no longer accumulate on a frozen neuron and
overshoot. Strongly-inhibited neurons still hyperpolarise deeply; that is faithful model
behaviour (the brian2 reference shows it too), not numerical blow-up.

Example::

    from connectome_dataset.graph_loader import load_flywire_783
    from connectome_dataset.benchmarks.torch.btorch.flybrain import FlyBrainModel

    W, flyids = load_flywire_783(return_ids=True)
    model = FlyBrainModel(W, flyids, device="cuda")
    rates = model.run_experiment(activate=[720575940624319294], t_run=200.0, n_run=8)
    # rates: mean firing rate (Hz) per neuron, averaged over trials
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from btorch.models import environ
from btorch.models.linear import SparseConn
from btorch.models.neurons.lif import LIF


@dataclass
class FlyBrainParams:
    """Shiu et al. (2024) model constants (units: mV, ms, Hz)."""

    v_rest: float = -52.0        # resting potential v_0 (== reset here)
    v_reset: float = -52.0       # reset potential v_rst after a spike
    v_threshold: float = -45.0   # spike threshold v_th
    t_mbr: float = 20.0          # membrane time constant
    tau_syn: float = 5.0         # synaptic time constant tau
    t_refractory: float = 2.2    # refractory period t_rfc
    t_delay: float = 1.8         # synaptic delay t_dly
    w_syn: float = 0.275         # weight per unit of signed connectivity
    r_poi: float = 150.0         # Poisson activation rate
    f_poi: float = 250.0         # upstream Poisson scaling (provenance; see module docstring)
    # Tonic background (an extension beyond the stimulus-only upstream model): an independent
    # Poisson drive per neuron injected through the synaptic conductance g (NOT straight into v).
    # Tuned to the diffusion regime — many small events per step — so g is a smooth near-DC
    # depolarisation and v never over/undershoots; it holds the brain at a low asynchronous-
    # irregular baseline instead of silent. bg_rate=0 disables.
    bg_rate: float = 3500.0      # background input rate per neuron (Hz); 0 disables
    bg_weight: float = 0.30      # mV added to g per background event (small -> smooth)
    dt: float = 0.1              # integration step (brian2 default)
    t_run: float = 1000.0        # trial duration
    n_run: int = 30              # trials averaged for a rate estimate


class FlyBrainModel(nn.Module):
    """Whole-brain *Drosophila* LIF network on a connectome, in btorch.

    Args:
        connectome: N x N signed adjacency, rows = presynaptic, cols = postsynaptic
            (e.g. from :func:`connectome_dataset.graph_loader.load_flywire_783`). Its
            values are the signed ``Excitatory x Connectivity`` counts; this class
            multiplies them by ``w_syn`` to get synaptic strength.
        flyids: optional FlyWire ID for each index, so experiments can address
            neurons by FlyWire ID instead of row index.
        params: model constants (defaults to the published values).
    """

    def __init__(
        self,
        connectome: sp.spmatrix,
        flyids: Sequence[int] | np.ndarray | None = None,
        params: FlyBrainParams = FlyBrainParams(),
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.p = params
        self.device = device
        self.dtype = dtype
        self.n = connectome.shape[0]
        self.base_weights = connectome.tocsr().astype(np.float32)
        self.flyids = None if flyids is None else np.asarray(flyids)
        self._id2idx = (
            {int(f): i for i, f in enumerate(self.flyids)} if flyids is not None else None
        )

        self.neuron = LIF(
            self.n,
            v_threshold=params.v_threshold,
            v_reset=params.v_reset,
            c_m=params.t_mbr,        # I / c_m with c_m == tau == t_mbr => g / t_mbr
            tau=params.t_mbr,
            tau_ref=params.t_refractory,
            hard_reset=True,         # Shiu reset is v = v_rst (absolute); also prevents overshoot
            device=device,
            dtype=dtype,
        )
        self.conn: SparseConn | None = None
        self._build_conn(silence_idx=None)

    def _build_conn(self, silence_idx: np.ndarray | None) -> None:
        """(Re)build the recurrent synapse, zeroing silenced rows and columns."""
        w = self.base_weights
        if silence_idx is not None and len(silence_idx):
            keep = np.ones(self.n, dtype=np.float32)
            keep[silence_idx] = 0.0
            d = sp.diags(keep)
            w = (d @ w @ d).tocsr()  # zero all synapses to and from silenced neurons
        scaled = (w * self.p.w_syn).tocoo()
        self.conn = SparseConn(
            sp.coo_array((scaled.data, (scaled.row, scaled.col)), shape=scaled.shape),
            enforce_dale=False,
            sparse_backend="native",
            device=self.device,
            dtype=self.dtype,
        )

    def _to_idx(self, neurons: Iterable[int]) -> np.ndarray:
        """Map FlyWire IDs (large) or raw row indices (< N) to row indices."""
        out = []
        for x in neurons:
            x = int(x)
            if self._id2idx is not None and x in self._id2idx:
                out.append(self._id2idx[x])
            elif 0 <= x < self.n:
                out.append(x)
            else:
                raise KeyError(f"neuron {x} is neither a known FlyWire ID nor an index < {self.n}")
        return np.asarray(out, dtype=np.int64)

    @torch.no_grad()
    def simulate(
        self,
        activate_idx: np.ndarray,
        silence_idx: np.ndarray | None = None,
        *,
        n_run: int | None = None,
        t_run: float | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run the network and return per-(trial, neuron) firing rates in Hz.

        Args:
            activate_idx: row indices driven with a Poisson spike train at ``r_poi``.
            silence_idx: row indices whose synapses (in and out) are zeroed.
            n_run: number of parallel trials (batch); defaults to ``params.n_run``.
            t_run: trial duration in ms; defaults to ``params.t_run``.
            seed: RNG seed for the Poisson activation.

        Returns:
            Tensor of shape ``(n_run, N)`` with firing rate (Hz) per trial and neuron.
        """
        p = self.p
        n_run = n_run or p.n_run
        steps = int(round((t_run or p.t_run) / p.dt))
        delay_steps = int(round(p.t_delay / p.dt))
        decay = float(np.exp(-p.dt / p.tau_syn))
        p_spike = p.r_poi * p.dt * 1e-3  # Poisson spike probability per dt
        bg_lambda = p.bg_rate * p.dt * 1e-3  # mean background events / neuron / step

        environ.set(dt=p.dt)
        self._build_conn(silence_idx)
        self.neuron.init_state(batch_size=n_run, device=self.device, dtype=self.dtype)
        self.neuron.v.fill_(p.v_rest)

        act = torch.as_tensor(activate_idx, device=self.device, dtype=torch.long)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        g = torch.zeros(n_run, self.n, device=self.device, dtype=self.dtype)
        counts = torch.zeros(n_run, self.n, device=self.device, dtype=self.dtype)
        buffer: deque[torch.Tensor] = deque(
            (torch.zeros(n_run, self.n, device=self.device, dtype=self.dtype)
             for _ in range(delay_steps)),
            maxlen=max(delay_steps, 1),
        )
        prev = torch.zeros(n_run, self.n, device=self.device, dtype=self.dtype)

        for _ in range(steps):
            delayed = buffer[0] if delay_steps > 0 else prev
            g = g * decay + self.conn(delayed)          # exponential synaptic conductance
            if bg_lambda > 0.0:                           # tonic background: smooth, through g
                g = g + torch.poisson(torch.full_like(g, bg_lambda), generator=gen) * p.bg_weight
            # Shiu dynamics are "(unless refractory)": a refractory neuron does not integrate
            # input. btorch's LIF integrates v every step, which lets a strong recurrent hit
            # accumulate on a refractory neuron and overshoot; masking the input to refractory
            # neurons restores the intended behaviour and bounds v (no clamp on v itself).
            inp = g * (self.neuron.refractory == 0).to(g.dtype) if self.neuron._use_refractory else g
            spk = self.neuron(inp)                        # LIF integrate + fire + hard reset
            if act.numel():                               # optogenetic Poisson drive
                forced = (torch.rand(n_run, act.numel(), generator=gen,
                                     device=self.device, dtype=self.dtype) < p_spike)
                spk[:, act] = forced.to(self.dtype)
            g = g * (1.0 - spk)                            # Shiu reset: g = 0 on spike
            counts += spk
            if delay_steps > 0:
                buffer.append(spk)
            else:
                prev = spk

        return counts / (steps * p.dt * 1e-3)

    def run_experiment(
        self,
        activate: Iterable[int],
        silence: Iterable[int] = (),
        **kwargs,
    ) -> np.ndarray:
        """Activation/silencing experiment addressed by FlyWire ID or index.

        Returns the mean firing rate (Hz) per neuron, averaged over trials — the
        readout the upstream model reports.
        """
        rates = self.simulate(self._to_idx(activate), self._to_idx(silence), **kwargs)
        return rates.mean(0).cpu().numpy()
