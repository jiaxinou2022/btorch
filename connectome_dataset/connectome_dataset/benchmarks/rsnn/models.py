"""beNNch benchmark network models — balanced E-I spiking networks.

The workloads are the Vogels-Abbott / Brette-2007 balanced excitatory-inhibitory
networks (``coba`` and ``cuba``) plus a ``connectome`` variant whose recurrent
operator is a real connectome adjacency matrix — the repo's raison d'être.

The neuron / synapse / projection primitives all come from ``brainpy.state`` and
``brainstate.nn`` (LIF-with-refractory, exponential synapses, event-driven
fixed-probability or CSR connectivity); this module only specifies the network
wiring. The two ``EINet`` parameter sets are taken directly from brainevent's
``examples/{COBA,CUBA}_2005.py``:

- Brette et al. (2007) *Simulation of networks of spiking neurons*, J. Comput. Neurosci.
- Vogels & Abbott (2005) *Signal propagation and logic gating*, J. Neurosci.

The ``update`` step is decomposed into ``_deliver`` (synaptic delivery: the
event-driven SpMV + synapse dynamics) and ``_neuron`` (LIF integration) so the
beNNch phase-resolved timer (:mod:`.phases`) can microbenchmark each phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import brainpy
import brainstate
import braintools
import brainunit as u
import scipy.sparse as sp

# Per-model parameters, lifted from brainevent's COBA_2005 / CUBA_2005 EINet.
_MODEL_PARAMS = {
    "cuba": dict(
        v_rest=-49.0, w_exc=1.62, w_inh=-9.0,
        out_exc=lambda: brainpy.state.CUBA.desc(scale=u.volt),
        out_inh=lambda: brainpy.state.CUBA.desc(scale=u.volt),
    ),
    "coba": dict(
        v_rest=-60.0, w_exc=0.6, w_inh=6.7,
        out_exc=lambda: brainpy.state.COBA.desc(E=0.0 * u.mV),
        out_inh=lambda: brainpy.state.COBA.desc(E=-80.0 * u.mV),
    ),
}

CONN_PER_PRE = 80  # fixed out-degree of the balanced network (both E and I)
DT = 0.1 * u.ms  # clock-driven integration step (0.1 ms), as in the upstream models
INPUT_CURRENT = 20.0 * u.mA


@dataclass(slots=True)
class NetworkSpec:
    """Structural description of a built network (drives the record + case)."""
    model: str
    num: int
    n_exc: int
    n_inh: int
    nnz: int
    conn_per_pre: int
    scale: float
    seed: int
    matrix: sp.csr_matrix | None = None  # populated only for the connectome variant

    @property
    def density(self) -> float:
        return self.nnz / (self.num * self.num) if self.num else 0.0


class EINet(brainstate.nn.Module):
    """Balanced E-I network with event-driven fixed-probability connectivity."""

    def __init__(self, model: str, scale: float, *, seed: int = 0):
        super().__init__()
        p = _MODEL_PARAMS[model]
        self.model = model
        self.n_exc = int(3200 * scale)
        self.n_inh = int(800 * scale)
        self.num = self.n_exc + self.n_inh
        self.N = brainpy.state.LIFRef(
            self.num,
            V_rest=p["v_rest"] * u.mV, V_th=-50.0 * u.mV, V_reset=-60.0 * u.mV,
            tau=20.0 * u.ms, tau_ref=5.0 * u.ms,
            V_initializer=braintools.init.Normal(-55.0, 2.0, unit=u.mV),
        )
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                self.n_exc, self.num, conn_num=CONN_PER_PRE / self.num,
                conn_weight=p["w_exc"] * u.mS, seed=seed,
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=5.0 * u.ms),
            out=p["out_exc"](), post=self.N,
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                self.n_inh, self.num, conn_num=CONN_PER_PRE / self.num,
                conn_weight=p["w_inh"] * u.mS, seed=seed + 1,
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=10.0 * u.ms),
            out=p["out_inh"](), post=self.N,
        )

    def init_state(self, *args, **kwargs):
        self.n_spike_exc = brainstate.ShortTermState(u.math.zeros((), dtype=int))
        self.n_spike_inh = brainstate.ShortTermState(u.math.zeros((), dtype=int))

    def _deliver(self, spk):
        self.E(spk[: self.n_exc])
        self.I(spk[self.n_exc :])

    def _neuron(self, inp):
        self.N(inp)

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike()
            self._deliver(spk)
            self._neuron(inp)
            self.n_spike_exc.value += spk[: self.n_exc].sum().astype(int)
            self.n_spike_inh.value += spk[self.n_exc :].sum().astype(int)
            return spk


class ConnectomeNet(brainstate.nn.Module):
    """LIF population whose recurrent operator is a connectome CSR matrix.

    The fixed sparse adjacency ``matrix`` is the recurrent connectivity; spikes are
    delivered through :class:`brainstate.nn.SparseLinear` (event-driven CSR SpMV).
    """

    def __init__(self, matrix: sp.csr_matrix, *, seed: int = 0):
        super().__init__()
        self.model = "connectome"
        n = matrix.shape[0]
        self.num = self.n_exc = n
        self.n_inh = 0
        self.N = brainpy.state.LIFRef(
            n, V_rest=-49.0 * u.mV, V_th=-50.0 * u.mV, V_reset=-60.0 * u.mV,
            tau=20.0 * u.ms, tau_ref=5.0 * u.ms,
            V_initializer=braintools.init.Normal(-55.0, 2.0, unit=u.mV),
        )
        csr = u.sparse.CSR((matrix.data * u.mS, matrix.indices, matrix.indptr), shape=matrix.shape)
        self.rec = brainpy.state.AlignPostProj(
            comm=brainstate.nn.SparseLinear(csr, in_size=n),
            syn=brainpy.state.Expon.desc(n, tau=5.0 * u.ms),
            out=brainpy.state.CUBA.desc(scale=u.volt), post=self.N,
        )

    def init_state(self, *args, **kwargs):
        self.n_spike_exc = brainstate.ShortTermState(u.math.zeros((), dtype=int))
        self.n_spike_inh = brainstate.ShortTermState(u.math.zeros((), dtype=int))

    def _deliver(self, spk):
        self.rec(spk)

    def _neuron(self, inp):
        self.N(inp)

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike()
            self._deliver(spk)
            self._neuron(inp)
            self.n_spike_exc.value += spk.sum().astype(int)
            return spk


def build_network(
    model: str, *, scale: float = 1.0, seed: int = 0, matrix: sp.csr_matrix | None = None
) -> tuple[brainstate.nn.Module, NetworkSpec]:
    """Construct a benchmark network and its structural spec.

    ``coba``/``cuba`` scale by neuron count (``scale``); ``connectome`` uses the
    provided catalog adjacency as the recurrent operator.
    """
    if model == "connectome":
        if matrix is None:
            raise ValueError("connectome model requires a `matrix`")
        matrix = matrix.tocsr()
        net = ConnectomeNet(matrix, seed=seed)
        avg_deg = matrix.nnz / matrix.shape[0] if matrix.shape[0] else 0
        spec = NetworkSpec(
            model="connectome", num=matrix.shape[0], n_exc=matrix.shape[0], n_inh=0,
            nnz=int(matrix.nnz), conn_per_pre=int(round(avg_deg)), scale=scale,
            seed=seed, matrix=matrix,
        )
    elif model in _MODEL_PARAMS:
        net = EINet(model, scale, seed=seed)
        spec = NetworkSpec(
            model=model, num=net.num, n_exc=net.n_exc, n_inh=net.n_inh,
            nnz=net.num * CONN_PER_PRE, conn_per_pre=CONN_PER_PRE, scale=scale, seed=seed,
        )
    else:
        raise ValueError(f"Unknown rsnn model '{model}' (coba|cuba|connectome)")
    return net, spec


def case_from_spec(spec: NetworkSpec) -> Any:
    """A minimal case object for ``base_extra_info`` / ``stratification``."""
    return SimpleNamespace(
        case_id=f"rsnn_{spec.model}_s{spec.scale:g}_n{spec.num}",
        graph_id=f"rsnn_{spec.model}",
        n=spec.num, nnz=spec.nnz, density=spec.density,
        matrix=spec.matrix, metadata={},
    )
