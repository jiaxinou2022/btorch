"""SpMV setup helpers — JAX / brainevent variants."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

try:
    import brainevent as _brainevent
    BRAINEVENT_AVAILABLE = True
except Exception:
    _brainevent = None
    BRAINEVENT_AVAILABLE = False


def _to_jax_bcoo(mat: sp.csr_matrix) -> jsparse.BCOO:
    coo = mat.tocoo().astype(np.float32)
    return jsparse.BCOO(
        (jnp.array(coo.data), jnp.stack([jnp.array(coo.row), jnp.array(coo.col)], axis=1)),
        shape=mat.shape,
    )


def _to_brainevent_csr(mat: sp.csr_matrix):
    mat = mat.tocsr().astype(np.float32)
    return _brainevent.CSR(
        data=jnp.array(mat.data),
        indices=jnp.array(mat.indices),
        indptr=jnp.array(mat.indptr),
        shape=mat.shape,
    )


def make_fn(mat: sp.csr_matrix, *, alg: str, variant: str, batch_size: int):
    """Return a JIT-compiled SpMV callable for the given JAX variant."""
    n = mat.shape[0]
    x = jnp.array(np.random.rand(n, batch_size).astype(np.float32))

    if (alg, variant) == ("jax", "bcoo"):
        a = _to_jax_bcoo(mat)
        return jax.jit(lambda: a @ x)

    if (alg, variant) == ("brainevent", "csr"):
        a = _to_brainevent_csr(mat)
        return jax.jit(lambda: a @ x)

    if (alg, variant) == ("brainevent", "event"):
        a = _to_brainevent_csr(mat)
        spikes = jnp.array((np.random.rand(n, batch_size) > 0.9).astype(bool))
        if batch_size == 1:
            return jax.jit(lambda: a @ _brainevent.BinaryArray(spikes[:, 0]))
        return jax.jit(lambda: jax.vmap(lambda col: a @ _brainevent.BinaryArray(col))(spikes.T).T)

    raise ValueError(f"Unknown JAX SpMV variant '{alg}/{variant}'")
