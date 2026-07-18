"""RSNN setup helpers — JAX / brainevent."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

try:
    import brainevent as _brainevent
    BRAINEVENT_AVAILABLE = True
except Exception:
    _brainevent = None
    BRAINEVENT_AVAILABLE = False

V_TH = 1.0
V_RESET = 0.0
DECAY_SYN = float(jnp.exp(-0.5e-3 / 5e-3))
DECAY_MEM = float(jnp.exp(-0.5e-3 / 20e-3))


def dense_matrix(n: int) -> sp.csr_matrix:
    rng = np.random.default_rng(0)
    nnz = max(1, int(n * n * 0.04))
    row = rng.integers(0, n, nnz).astype(np.int32)
    col = rng.integers(0, n, nnz).astype(np.int32)
    data = rng.standard_normal(nnz).astype(np.float32)
    return sp.csr_matrix((data, (row, col)), shape=(n, n))


def _make_step_fn(conn_mat: sp.csr_matrix, backend: str):
    n = conn_mat.shape[0]

    if backend == "brainevent":
        if not BRAINEVENT_AVAILABLE:
            raise RuntimeError("brainevent not installed")
        graph = _brainevent.CSR(
            data=jnp.array(conn_mat.data, dtype=jnp.float32),
            indices=jnp.array(conn_mat.indices, dtype=jnp.int32),
            indptr=jnp.array(conn_mat.indptr, dtype=jnp.int32),
            shape=(n, n),
        )

        def single_spmv(spikes):
            return graph @ _brainevent.BinaryArray(spikes > 0.5)

    elif backend == "jax_bcoo":
        import jax.experimental.sparse as jsparse

        coo = conn_mat.tocoo().astype(np.float32)
        graph = jsparse.BCOO(
            (
                jnp.array(coo.data, dtype=jnp.float32),
                jnp.stack(
                    [jnp.array(coo.row, dtype=jnp.int32), jnp.array(coo.col, dtype=jnp.int32)],
                    axis=1,
                ),
            ),
            shape=(n, n),
        )

        def single_spmv(spikes):
            return graph @ spikes.astype(jnp.float32)

    else:
        raise ValueError(f"Unknown backend '{backend}'")

    def step(carry, _):
        v, g = carry
        s_prev = (v > V_TH).astype(jnp.float32)
        syn_in = jax.vmap(single_spmv)(s_prev)
        g_new = DECAY_SYN * g + syn_in
        v_new = DECAY_MEM * v + (1.0 - DECAY_MEM) * g_new
        spike = (v_new >= V_TH).astype(jnp.float32)
        v_new = v_new * (1.0 - spike) + V_RESET * spike
        return (v_new, g_new), spike

    return step


def make_fn(conn_mat: sp.csr_matrix, *, backend: str, timesteps: int, batch_size: int):
    """Return a JIT-compiled RSNN forward-pass callable."""
    n = conn_mat.shape[0]
    step_fn = _make_step_fn(conn_mat, backend)
    v0 = jnp.zeros((batch_size, n))
    g0 = jnp.zeros((batch_size, n))
    return jax.jit(lambda: jax.lax.scan(step_fn, (v0, g0), None, length=timesteps)[1])
