"""btorch Billeh GLIF mouse-V1 model (Billeh et al. 2020) with per-cell-type parameters.

``mice_v1_guozhang`` is the point-GLIF V1 model: every neuron belongs to one of ~111 cell
types, each with its own GLIF3 parameter fit (Allen Cell Types). This model builds a
**heterogeneous GLIF3 population** — per-neuron threshold, reset, leak, capacitance, membrane
time constant, refractory period, and two after-spike currents — on the connectome, driven by
a background Poisson current. It is the mouse analogue of the fly ``flybrain`` model.

Network size is configurable: :func:`connectome_dataset.graph_loader.load_mice_v1_glif`
sub-samples the r<400um core to any ``n_neurons`` (as the upstream ``load_sparse.py``), so a
test runs a few thousand neurons while the full core is 51,978.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from btorch.models import environ
from btorch.models.linear import SparseConn
from btorch.models.neurons.glif import GLIF3


@dataclass
class MouseV1Params:
    """Simulation knobs for the Billeh GLIF mouse-V1 model (neuron params come per cell type)."""

    tau_syn: float = 5.0     # synaptic time constant (ms); the 4 receptor channels are collapsed
    dt: float = 1.0          # integration step (ms), as the upstream V1 model
    bg_rate: float = 250.0   # per-neuron background input rate (Hz); 0 disables
    bg_weight: float = 100.0 # current (pA) per background event — tuned to a ~10 Hz ground state
    seed: int = 0


class MouseV1GLIF(nn.Module):
    """Heterogeneous GLIF3 V1 population on the mouse connectome, in btorch.

    Args:
        connectivity: signed N x N adjacency (pA), rows=pre, cols=post.
        neurons: per-neuron GLIF3 parameters from ``load_mice_v1_glif`` (each ``(n,)`` or
            ``(n, 2)`` for ``k`` / ``asc_amps``), plus ``node_type_id``.
    """

    def __init__(
        self,
        connectivity: sp.csr_matrix,
        neurons: dict[str, np.ndarray],
        params: MouseV1Params = MouseV1Params(),
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.p = params
        self.device = device
        self.dtype = dtype
        self.n = connectivity.shape[0]
        self.node_type_id = np.asarray(neurons["node_type_id"])

        def tens(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        self.neuron = GLIF3(
            self.n,
            v_threshold=tens(neurons["v_threshold"]), v_reset=tens(neurons["v_reset"]),
            v_rest=tens(neurons["v_rest"]), c_m=tens(neurons["c_m"]), tau=tens(neurons["tau"]),
            tau_ref=tens(neurons["tau_ref"]), k=tens(neurons["k"]), asc_amps=tens(neurons["asc_amps"]),
            device=device, dtype=dtype,
        )
        self.conn = SparseConn(
            sp.coo_array(connectivity.tocoo()), enforce_dale=False,
            sparse_backend="native", device=device, dtype=dtype,
        )

    @torch.no_grad()
    def simulate(self, timesteps: int, *, warmup: int = 100, seed: int | None = None) -> torch.Tensor:
        """Run the network; return per-neuron firing rate (Hz) over the measured window."""
        p = self.p
        seed = p.seed if seed is None else seed
        decay = float(np.exp(-p.dt / p.tau_syn))
        bg_lambda = p.bg_rate * p.dt * 1e-3

        environ.set(dt=p.dt)
        self.neuron.init_state(batch_size=1, device=self.device, dtype=self.dtype)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        g = torch.zeros(1, self.n, device=self.device, dtype=self.dtype)
        counts = torch.zeros(1, self.n, device=self.device, dtype=self.dtype)
        prev = torch.zeros(1, self.n, device=self.device, dtype=self.dtype)

        for step in range(warmup + timesteps):
            g = g * decay + self.conn(prev)
            if bg_lambda > 0.0:
                g = g + torch.poisson(torch.full_like(g, bg_lambda), generator=gen) * p.bg_weight
            spk = self.neuron(g)
            prev = spk
            if step >= warmup:
                counts += spk
        return (counts / (timesteps * p.dt * 1e-3)).squeeze(0)

    def population_rates(self, neuron_rates: torch.Tensor) -> dict[int, float]:
        """Mean firing rate (Hz) per GLIF cell type."""
        r = neuron_rates.detach().cpu().numpy()
        return {int(t): float(r[self.node_type_id == t].mean())
                for t in np.unique(self.node_type_id)}
