"""RSNN model classes and setup helpers — btorch."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from btorch.models import glif, rnn, synapse
from btorch.models.linear import DenseConn, SparseConn


def _make_neuron(n: int, device: str, dtype: torch.dtype) -> glif.GLIF3:
    return glif.GLIF3(
        n_neuron=n,
        v_threshold=-45.0,
        v_reset=-60.0,
        c_m=2.0,
        tau=20.0,
        tau_ref=2.0,
        k=[0.1, 0.2],
        asc_amps=[1.0, -2.0],
        step_mode="s",
        backend="torch",
        device=device,
        dtype=dtype,
    )


def _make_psc(n: int, conn) -> synapse.AlphaPSC:
    return synapse.AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")


class DenseRSNN(nn.Module):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, *, device: str, dtype: torch.dtype):
        super().__init__()
        self.fc_in = nn.Linear(num_input, num_hidden, bias=False, device=device, dtype=dtype)
        conn = DenseConn(num_hidden, num_hidden, bias=None, device=device, dtype=dtype)
        self.brain = rnn.RecurrentNN(
            neuron=_make_neuron(num_hidden, device, dtype),
            synapse=_make_psc(num_hidden, conn),
            step_mode="m",
        )
        self.fc_out = nn.Linear(num_hidden, num_output, bias=False, device=device, dtype=dtype)
        self.num_hidden = num_hidden

    def forward(self, x):
        x = self.fc_in(x)
        spike, _ = self.brain(x)
        return self.fc_out(spike.mean(0))


class SparseRSNN(nn.Module):
    def __init__(self, conn_mat: sp.spmatrix, num_input: int, num_output: int, *, device: str, dtype: torch.dtype):
        super().__init__()
        n = conn_mat.shape[0]
        coo = conn_mat.tocoo().astype(np.float32)
        # Random synaptic weights on the real connectome's sparsity pattern; seeded
        # so timing runs are reproducible. (Values don't affect SpMV cost, but an
        # unseeded RNG makes runs non-deterministic.)
        rng = np.random.default_rng(0)
        coo.data[:] = (rng.standard_normal(coo.nnz) * 0.1).astype(np.float32)
        self.fc_in = nn.Linear(num_input, n, bias=False, device=device, dtype=dtype)
        conn = SparseConn(sp.csr_array(coo), sparse_backend="native", device=device, dtype=dtype)
        self.brain = rnn.RecurrentNN(
            neuron=_make_neuron(n, device, dtype),
            synapse=_make_psc(n, conn),
            step_mode="m",
        )
        self.fc_out = nn.Linear(n, num_output, bias=False, device=device, dtype=dtype)
        self.num_hidden = n

    def forward(self, x):
        x = self.fc_in(x)
        spike, _ = self.brain(x)
        return self.fc_out(spike.mean(0))
