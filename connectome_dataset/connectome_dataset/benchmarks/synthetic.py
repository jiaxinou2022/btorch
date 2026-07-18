"""Synthetic controlled-sweep matrix generators for benchmark corpora.

Real connectome adjacency is scattered, low-density, and heavy-tailed in degree.
A single real graph fixes all of those axes at once, so it can't tell you *which*
structural property flips an algorithm's ranking. These generators produce scipy
CSR adjacency with **independently controllable** structure so we can sweep the
axes that flip which algorithm / CUDA feature wins — degree coefficient-of-variation,
density, and bandwidth — one at a time.

Families (see docs/benchmark_design/literature.md):

- ``erdos_renyi`` — uniform random edges; low degree-CoV baseline.
- ``power_law``   — heavy-tailed degree via Chung–Lu sampling; connectome-like
  CoV / Gini. ``gamma`` tunes the tail (smaller → heavier).
- ``banded``      — nonzeros confined to a diagonal band; low CoV, structured
  locality (SMaT-style).
- ``rmat``        — recursive/self-similar community structure (Graph500 R-MAT).

All generators are deterministic in ``seed``, return a square ``scipy.sparse.csr_matrix``
of float32, target a given ``avg_degree`` (expected nonzeros per row), and place an
explicit unit-ish diagonal off by default. Duplicate edges from sampling are merged,
so the realized nnz is at or below ``n * avg_degree``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from connectome_dataset.benchmarks.cases import (
    SpgemmCase,
    SpmvCase,
    spgemm_case_from_matrix,
    spmv_case_from_matrix,
)


def _coo(n: int, rows: np.ndarray, cols: np.ndarray, rng: np.random.Generator) -> sp.csr_matrix:
    """Assemble a CSR matrix from edge endpoints, dropping self-loops and merging duplicates."""
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    data = (rng.standard_normal(rows.size) * 0.1).astype(np.float32)
    m = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    m.sum_duplicates()  # merge parallel edges so nnz reflects distinct connections
    return m.tocsr()


def erdos_renyi(n: int, avg_degree: float = 8.0, *, seed: int = 0) -> sp.csr_matrix:
    """Uniform random directed graph — the low-CoV reference point."""
    rng = np.random.default_rng(seed)
    m_edges = int(round(n * avg_degree))
    rows = rng.integers(0, n, size=m_edges)
    cols = rng.integers(0, n, size=m_edges)
    return _coo(n, rows, cols, rng)


def power_law(
    n: int, avg_degree: float = 8.0, *, gamma: float = 2.5, seed: int = 0
) -> sp.csr_matrix:
    """Heavy-tailed degree via Chung–Lu sampling (edge endpoint ∝ node weight).

    Node weights follow ``w_i ∝ i^{-1/(gamma-1)}`` (a power-law degree sequence);
    edges are drawn by sampling both endpoints ∝ ``w``. This reproduces the
    high degree-CoV / Gini of connectome hubs without fixing any other axis.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-1.0 / (gamma - 1.0))
    weights /= weights.sum()
    m_edges = int(round(n * avg_degree))
    perm = rng.permutation(n)  # decouple weight ordering from node index
    rows = perm[rng.choice(n, size=m_edges, p=weights)]
    cols = perm[rng.choice(n, size=m_edges, p=weights)]
    return _coo(n, rows, cols, rng)


def banded(n: int, avg_degree: float = 8.0, *, seed: int = 0) -> sp.csr_matrix:
    """Nonzeros confined to a diagonal band of half-width ~avg_degree/2.

    Structured locality with near-constant row degree (low CoV) — the opposite
    structural regime from ``power_law``.
    """
    rng = np.random.default_rng(seed)
    half = max(1, int(round(avg_degree / 2.0)))
    rows_list, cols_list = [], []
    for offset in range(-half, half + 1):
        if offset == 0:
            continue
        idx = np.arange(n)
        cols = idx + offset
        valid = (cols >= 0) & (cols < n)
        rows_list.append(idx[valid])
        cols_list.append(cols[valid])
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    return _coo(n, rows, cols, rng)


def rmat(
    n: int,
    avg_degree: float = 8.0,
    *,
    a: float = 0.57,
    b: float = 0.19,
    c: float = 0.19,
    seed: int = 0,
) -> sp.csr_matrix:
    """Graph500 R-MAT: recursive subdivision producing self-similar communities.

    ``(a, b, c, d=1-a-b-c)`` are the quadrant probabilities. Endpoints are drawn
    in an ``n_pad = 2**scale`` space then truncated to ``n``.
    """
    rng = np.random.default_rng(seed)
    scale = int(np.ceil(np.log2(max(n, 2))))  # endpoints drawn in a 2**scale space, then truncated to n
    m_edges = int(round(n * avg_degree))
    d = 1.0 - a - b - c
    rows = np.zeros(m_edges, dtype=np.int64)
    cols = np.zeros(m_edges, dtype=np.int64)
    for bit in range(scale):
        step = 1 << bit
        u = rng.random(m_edges)
        # top-half (a|b) vs bottom-half (c|d) sets the row bit; left vs right the col bit
        row_bit = u >= (a + b)
        p_right_top = b / (a + b) if (a + b) > 0 else 0.0
        p_right_bot = d / (c + d) if (c + d) > 0 else 0.0
        v = rng.random(m_edges)
        col_bit = np.where(row_bit, v < p_right_bot, v < p_right_top)
        rows += row_bit.astype(np.int64) * step
        cols += col_bit.astype(np.int64) * step
    # truncate the padded space down to n
    keep = (rows < n) & (cols < n)
    return _coo(n, rows[keep], cols[keep], rng)


_GENERATORS: dict[str, Callable[..., sp.csr_matrix]] = {
    "erdos_renyi": erdos_renyi,
    "power_law": power_law,
    "banded": banded,
    "rmat": rmat,
}


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """A named point in the synthetic sweep."""

    family: str
    n: int
    avg_degree: float = 8.0
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        tag = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        base = f"syn_{self.family}_n{self.n}_d{self.avg_degree:g}"
        return f"{base}_{tag}" if tag else base

    def generate(self) -> sp.csr_matrix:
        gen = _GENERATORS.get(self.family)
        if gen is None:
            raise ValueError(f"unknown synthetic family: {self.family!r}")
        return gen(self.n, self.avg_degree, seed=self.seed, **self.params)


def spmv_case(spec: SyntheticSpec, **kwargs: Any) -> SpmvCase:
    return spmv_case_from_matrix(
        spec.generate(),
        case_id=f"{spec.name}_spmv",
        graph_id=spec.name,
        metadata={"synthetic": True, "family": spec.family, "seed": spec.seed},
        **kwargs,
    )


def spgemm_case(spec: SyntheticSpec, **kwargs: Any) -> SpgemmCase:
    return spgemm_case_from_matrix(
        spec.generate(),
        case_id=f"{spec.name}_spgemm",
        graph_id=spec.name,
        metadata={"synthetic": True, "family": spec.family, "seed": spec.seed},
        **kwargs,
    )


def default_sweep(n: int = 4096, avg_degree: float = 16.0) -> list[SyntheticSpec]:
    """A small deterministic sweep spanning the CoV / locality regimes.

    Same n and avg_degree across families so only *structure* varies — the whole
    point of a controlled sweep. ``power_law`` is sampled at two tail exponents.
    """
    return [
        SyntheticSpec("banded", n, avg_degree),
        SyntheticSpec("erdos_renyi", n, avg_degree),
        SyntheticSpec("power_law", n, avg_degree, params={"gamma": 2.5}),
        SyntheticSpec("power_law", n, avg_degree, params={"gamma": 2.1}),
        SyntheticSpec("rmat", n, avg_degree),
    ]
