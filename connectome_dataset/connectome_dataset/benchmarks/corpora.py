"""Named matrix sets — curated corpora the benchmark sweeps as a unit.

A single graph fixes every structural axis at once, so a run over one graph can't
tell you how an algorithm's ranking moves with structure. These named sets pair
with the synthetic sweep (:mod:`connectome_dataset.benchmarks.synthetic`): the
synthetic sweep isolates one axis at a time under controlled conditions, while
``connectome`` samples the *real* corpus across the axes that matter.

Selection for ``connectome`` deliberately spans:
- size:    n from ~200 (macro) to ~21.7k / 3.5M nnz (the large-scale stress case)
- density: from 0.15% (ultra-sparse EM) to 48% (dense macro tract-tracing)
- degree skew: Gini from ~0.24 (regular) to ~0.87 (heavy-tailed hubs)

These are the axes that flip which algorithm and which CUDA feature wins, so a
sweep over this set exercises every regime the connectome workloads actually hit.
"""
from __future__ import annotations

from connectome_dataset.catalog import find_graph, load_catalog

# Curated real-connectome corpus, ordered sparse/scattered -> dense/regular.
CONNECTOME_REPRESENTATIVE = [
    "kasthuri_v4",           # n=1029  0.15%  Gini .87 — ultra-sparse, heavy-tailed (hardest scatter)
    "drosophila_medulla_1",  # n=1781  0.31%           — sparse insect neuropil
    "mouse_retina_1",        # n=1123  7.2%   Gini .66 — mid-density, heavy-tailed
    "fly_larva",             # n=2952  1.27%           — mid-size sparse
    "mice_column_v1",        # n=4166  4.19%           — the primary V1-column target
    "rat_brain_1",           # n=503   11%             — denser macro-connectome
    "mouse_brain_1",         # n=213   48%    Gini .24 — near-dense macro (tract-tracing)
    "fly_hemibrain",         # n=21739 0.75%  3.5M nnz — large-scale stress case
]

# Standard SuiteSparse graph collections (Newman / DIMACS10 / Arenas), the literature
# baseline corpus. Curated to span the structural axes the connectome set also spans, but
# on *non-connectome* graphs so a kernel's ranking can be checked off-distribution:
# heavy-tailed social/collaboration (high Gini) vs near-uniform planar meshes (Gini ~0).
SUITESPARSE_LITERATURE = [
    "suitesparse_newman_karate",          # n=34    tiny classic
    "suitesparse_newman_polblogs",        # n=1490  political blogs, Gini .70 (heaviest tail)
    "suitesparse_newman_as_22july06",     # n=22963 internet AS graph, Gini .63
    "suitesparse_arenas_pgpgiantcompo",   # n=10680 PGP trust web, Gini .59
    "suitesparse_newman_cond_mat",        # n=16726 cond-mat collaboration, Gini .48
    "suitesparse_arenas_email",           # n=1133  email network
    "suitesparse_newman_netscience",      # n=1589  sparse collaboration
    "suitesparse_newman_power",           # n=4941  US power grid, very low density
    "suitesparse_dimacs10_delaunay_n12",  # n=4096  Delaunay mesh, regular degree
    "suitesparse_dimacs10_cs4",           # n=22499 mesh, near-uniform (Gini .02)
    "suitesparse_dimacs10_fe_sphere",     # n=16386 FE sphere mesh (Gini ~0, the uniform extreme)
    "suitesparse_dimacs10_cti",           # n=16840 mesh
]

# SNAP power-law graphs (social / web / citation / collaboration / communication) from the
# SuiteSparse SNAP group — larger, off-distribution but connectome-like heavy-tailed graphs
# (Gini mostly 0.5–0.85), 29k–2.3M nnz (up to ~23× the small `suitesparse` set). These are
# the kind of graphs the tensor-core SpMM papers (FlashSparse, DTC-SpMM) benchmark. Fetch with
# ``scripts/download_suitesparse.py --groups SNAP --matrices <names…>`` then ``connectome-catalog``.
SNAP_POWER_LAW = [
    "suitesparse_snap_ca_grqc",          # n=5242    collaboration (GR-QC)
    "suitesparse_snap_ca_hepth",         # n=9877    collaboration (Hep-Th)
    "suitesparse_snap_ca_condmat",       # n=23133   collaboration (cond-mat)
    "suitesparse_snap_ca_hepph",         # n=12008   collaboration, Gini .73
    "suitesparse_snap_ca_astroph",       # n=18772   collaboration (astro)
    "suitesparse_snap_cit_hepth",        # n=27770   citation (Hep-Th)
    "suitesparse_snap_cit_hepph",        # n=34546   citation (Hep-Ph)
    "suitesparse_snap_email_enron",      # n=36692   email, Gini .73
    "suitesparse_snap_email_euall",      # n=265214  email (EU, large, sparse)
    "suitesparse_snap_wiki_vote",        # n=8297    who-votes-whom, Gini .84
    "suitesparse_snap_soc_epinions1",    # n=75888   trust network, Gini .81
    "suitesparse_snap_soc_slashdot0811", # n=77360   social (Slashdot)
    "suitesparse_snap_p2p_gnutella31",   # n=62586   peer-to-peer overlay
    "suitesparse_snap_amazon0302",       # n=262111  co-purchase, Gini .06 (near-regular contrast)
    "suitesparse_snap_web_notredame",    # n=325729  web graph, Gini .85 (heaviest tail)
    "suitesparse_snap_web_stanford",     # n=281903  web graph, 2.3M nnz (largest)
]

MATRIX_SETS: dict[str, list[str]] = {
    "connectome": CONNECTOME_REPRESENTATIVE,
    "suitesparse": SUITESPARSE_LITERATURE,
    "snap": SNAP_POWER_LAW,
}


def _all_graph_names() -> list[str]:
    catalog = load_catalog()
    items = catalog if isinstance(catalog, list) else catalog.get("graphs", [])
    return [e.get("name") for e in items if e.get("name")]


def resolve_matrix_set(name: str) -> list[str]:
    """Resolve a corpus selector to graph IDs.

    Accepts a named set (``connectome``/``suitesparse``), a single graph ID, or a
    ``prefix*`` glob that expands to every catalog graph with that prefix (e.g.
    ``suitesparse_newman_*`` or ``suitesparse_*``).
    """
    if name in MATRIX_SETS:
        return list(MATRIX_SETS[name])
    if name.endswith("*"):
        prefix = name[:-1]
        matched = sorted(g for g in _all_graph_names() if g.startswith(prefix))
        if not matched:
            raise ValueError(f"no catalog graphs match prefix {prefix!r}")
        return matched
    if find_graph(name) is not None:
        return [name]
    raise ValueError(
        f"unknown matrix set / graph {name!r}; known sets: {sorted(MATRIX_SETS)}"
    )
