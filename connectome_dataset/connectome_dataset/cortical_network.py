"""Configurable cortical spiking networks from the macaque multi-area model.

The INM-6 multi-area model (Schmidt et al. 2018) is specified at the *mesoscale*:
population sizes ``N`` and mean in-degrees ``K`` between ``P`` populations. From that
one description this module instantiates neuron-level connectivity at **any scale** —
which is essential, because full scale is 4.13M neurons and ~24 billion synapses.
Two views share all the machinery:

  * ``microcircuit_spec(area=...)`` — one cortical area, 8 populations (the
    Potjans-Diesmann-type layered microcircuit that the multi-area model is built on).
  * ``multiarea_spec(areas=...)`` — the full 32-area / 254-population network.

``n_scaling`` scales population sizes and ``k_scaling`` scales in-degrees; both default
small so nothing large is ever built by accident. :func:`instantiate_connectivity`
refuses to allocate more than ``max_nnz`` synapses (raises), so an over-ambitious scale
fails fast instead of exhausting memory.

Stage the mesoscale once with ``scripts/build_multiarea_mesoscale.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("CONNECTOME_DATA_ROOT", _REPO / "data"))
MULTIAREA_MESOSCALE_ROOT = DATA_ROOT / "external" / "multiarea_mesoscale"

# Instantiating more synapses than this refuses (guards against the 24-billion full
# scale). Override per call; the default sits well above routine benchmark graphs.
DEFAULT_MAX_NNZ = 400_000_000


@dataclass
class CorticalNetworkSpec:
    """Population-level description of a (possibly scaled) cortical network."""

    N: np.ndarray            # (P,) neurons per population (already scaled)
    K_int: np.ndarray        # (P, P) mean in-degree, target <- source (scaled)
    K_ext: np.ndarray        # (P,) external (background) in-degree per population
    W_int: np.ndarray        # (P, P) mean synaptic weight (pA; <0 inhibitory)
    W_ext: np.ndarray        # (P,) external synaptic weight (pA)
    labels: list[str]        # (P,) "<area>_<pop>" per population
    neuron: dict             # iaf_psc_exp parameters (C_m, tau_m, E_L, V_th, ...)
    delay: dict              # delay parameters (delay_e, delay_i, ...)
    bg_rate: float           # background Poisson rate per external synapse (Hz)
    n_scaling: float = 1.0
    k_scaling: float = 1.0

    @property
    def n_neurons(self) -> int:
        return int(self.N.sum())

    @property
    def n_pop(self) -> int:
        return len(self.labels)

    def population_of_neuron(self) -> np.ndarray:
        """(n_neurons,) population index for each neuron, in block order."""
        return np.repeat(np.arange(self.n_pop), self.N.astype(np.int64))


def load_multiarea_mesoscale(root: str | Path | None = None) -> dict:
    """Load the staged full-scale mesoscale arrays and neuron/delay params."""
    root = Path(root) if root is not None else MULTIAREA_MESOSCALE_ROOT
    npz_path = root / "multiarea_mesoscale.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"multiarea mesoscale not staged at {root}; run "
            "scripts/build_multiarea_mesoscale.py (see datasets/multiarea/README.md)"
        )
    z = np.load(npz_path, allow_pickle=True)
    params = json.loads((root / "params.json").read_text())
    return dict(
        N=z["N"], K_int=z["K_int"], K_ext=z["K_ext"], W_int=z["W_int"], W_ext=z["W_ext"],
        labels=[str(x) for x in z["labels"]], neuron=params["neuron"],
        delay=params["delay"], bg_rate=params["bg_rate"],
    )


def _scaled_spec(meso: dict, sel: np.ndarray, n_scaling: float, k_scaling: float) -> CorticalNetworkSpec:
    """Slice the mesoscale to the selected populations and apply scaling."""
    N = np.maximum(1, np.round(meso["N"][sel] * n_scaling)).astype(np.int64)
    return CorticalNetworkSpec(
        N=N,
        K_int=meso["K_int"][np.ix_(sel, sel)] * k_scaling,
        K_ext=meso["K_ext"][sel] * k_scaling,
        W_int=meso["W_int"][np.ix_(sel, sel)],
        W_ext=meso["W_ext"][sel],
        labels=[meso["labels"][i] for i in sel],
        neuron=meso["neuron"], delay=meso["delay"], bg_rate=meso["bg_rate"],
        n_scaling=n_scaling, k_scaling=k_scaling,
    )


def multiarea_spec(root: str | Path | None = None, *, n_scaling: float = 0.01,
                   k_scaling: float = 1.0, areas: list[str] | None = None) -> CorticalNetworkSpec:
    """The multi-area network (all areas, or a subset), scaled by ``n_scaling``.

    Full scale (``n_scaling=1``) is 4.13M neurons / ~24e9 synapses and cannot be
    instantiated at neuron level — keep ``n_scaling`` small (the default 0.01 gives
    ~41k neurons).
    """
    meso = load_multiarea_mesoscale(root)
    if areas is None:
        sel = np.arange(len(meso["labels"]))
    else:
        want = {a for a in areas}
        sel = np.array([i for i, lab in enumerate(meso["labels"])
                        if lab.split("_")[0] in want])
    return _scaled_spec(meso, sel, n_scaling, k_scaling)


def microcircuit_spec(root: str | Path | None = None, *, area: str = "V1",
                      n_scaling: float = 1.0, k_scaling: float = 1.0) -> CorticalNetworkSpec:
    """One cortical area's 8-population microcircuit (Potjans-Diesmann-type).

    Full scale is ~a few x10^4 neurons per area, so ``n_scaling=1`` is fine for a
    single microcircuit; scale down for quick tests.
    """
    meso = load_multiarea_mesoscale(root)
    sel = np.array([i for i, lab in enumerate(meso["labels"]) if lab.split("_")[0] == area])
    if sel.size == 0:
        raise ValueError(f"unknown area '{area}'")
    return _scaled_spec(meso, sel, n_scaling, k_scaling)


def instantiate_connectivity(spec: CorticalNetworkSpec, *, seed: int = 0,
                             max_nnz: int = DEFAULT_MAX_NNZ) -> sp.csr_matrix:
    """Draw a neuron-level signed adjacency (rows=presynaptic, cols=postsynaptic).

    For each ordered population pair (target ``t`` <- source ``s``) the model's mean
    in-degree fixes the synapse count ``round(K_int[t, s] * N[t])``; those synapses
    connect random source neurons to random target neurons (multapses allowed, as in
    the model). The signed weight ``W_int[t, s]`` (pA) is assigned per block.

    Raises ``MemoryError`` if the total synapse count exceeds ``max_nnz`` — the guard
    that keeps a too-large scale from trying to allocate billions of edges.
    """
    rng = np.random.default_rng(seed)
    N = spec.N.astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(N)])
    n = int(offsets[-1])

    syn_per_block = np.round(spec.K_int * N[:, None]).astype(np.int64)  # (t, s)
    total = int(syn_per_block.sum())
    if total > max_nnz:
        raise MemoryError(
            f"instantiation would create {total:,} synapses (> max_nnz={max_nnz:,}); "
            f"lower n_scaling/k_scaling (n_scaling={spec.n_scaling}, neurons={n:,})"
        )

    rows, cols, vals = [], [], []
    for t in range(spec.n_pop):
        for s in range(spec.n_pop):
            m = int(syn_per_block[t, s])
            if m == 0 or N[s] == 0 or N[t] == 0:
                continue
            pre = rng.integers(offsets[s], offsets[s + 1], m)
            post = rng.integers(offsets[t], offsets[t + 1], m)
            rows.append(pre)
            cols.append(post)
            vals.append(np.full(m, spec.W_int[t, s], dtype=np.float32))
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    mat = sp.csr_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64))),
        shape=(n, n), dtype=np.float32,
    )
    mat.sum_duplicates()  # merge multapses into summed weights
    return mat
