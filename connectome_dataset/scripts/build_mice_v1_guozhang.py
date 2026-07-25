"""Build the ``mice_v1_guozhang`` dataset from Guozhang's V1 GLIF network.

Reads the raw Billeh/Guozhang V1 model (``network_dat.pkl`` + ``network/v1_nodes.h5``)
and collapses it into a single neuron-to-neuron weighted adjacency for the sparse
benchmark corpus. The raw payload is large (a 1.7 GB pickle plus stimuli we do not
need); this keeps only the connectivity we benchmark on:

  * select the model *core* (radial distance ``r < CORE_RADIUS`` um, the canonical
    51,978-neuron V1 column used by the Billeh model),
  * sum the four GLIF receptor-type channels into one synaptic weight per
    (presynaptic, postsynaptic) pair, giving an N x N signed CSR,
  * orient it rows=presynaptic, cols=postsynaptic — the same convention as
    ``mice_column_v1`` (``pre_simple_id`` rows, ``post_simple_id`` cols).

Output (DVC-managed under ``data/external/mice_v1_guozhang``):
  * ``mice_v1_guozhang.npz``    — scipy CSR adjacency (float32, signed weights)
  * ``generation_params.yaml``  — provenance for the catalog / manifest

Usage:
    micromamba run -n ml-py312 python scripts/build_mice_v1_guozhang.py \
        --src /data/fanqixuan/dataset/GLIF_V1_network --core-radius 400
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path("/data/fanqixuan/dataset/GLIF_V1_network")
DEFAULT_OUT = _REPO / "data" / "external" / "mice_v1_guozhang"
CORE_RADIUS_UM = 400.0


def core_selection(h5_path: Path, core_radius: float) -> np.ndarray:
    """Boolean mask over all BMTK nodes selecting the radial core (r < core_radius)."""
    with h5py.File(h5_path, "r") as h5:
        nodes = h5["nodes"]["v1"]
        assert np.diff(nodes["node_id"]).var() < 1e-12, "node_id must be contiguous"
        x = np.asarray(nodes["0"]["x"], dtype=np.float64)
        z = np.asarray(nodes["0"]["z"], dtype=np.float64)
    return np.sqrt(x**2 + z**2) < core_radius


def build_adjacency(pkl_path: Path, h5_path: Path, core_radius: float):
    """Collapse the GLIF edge groups into an N x N signed weighted CSR (rows=pre, cols=post)."""
    import pickle

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    sel = core_selection(h5_path, core_radius)
    n = int(sel.sum())
    bmtk_to_local = np.full(sel.size, -1, dtype=np.int64)
    bmtk_to_local[sel] = np.arange(n)

    pre, post, weight = [], [], []
    for edge in d["edges"]:
        src = bmtk_to_local[np.asarray(edge["source"])]
        dst = bmtk_to_local[np.asarray(edge["target"])]
        keep = (src >= 0) & (dst >= 0)
        if not keep.any():
            continue
        w = np.asarray(edge["params"]["weight"], dtype=np.float32)[keep]
        pre.append(src[keep])
        post.append(dst[keep])
        weight.append(w)

    rows = np.concatenate(pre)
    cols = np.concatenate(post)
    data = np.concatenate(weight)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    mat.sum_duplicates()  # collapse the four receptor channels into one weight
    mat.sort_indices()
    return mat


def build_neuron_params(pkl_path: Path, h5_path: Path, core_radius: float) -> dict:
    """Extract the per-cell-type GLIF3 parameters and per-neuron type id for the core.

    Each of the ~111 GLIF cell types carries a full point-GLIF parameter set (Allen Cell
    Types / Billeh et al. 2020). We stage the compact per-type table plus the per-neuron
    ``node_type_id`` that indexes it, and the ``x, z`` coordinates so the loader can select a
    configurable radial sub-network (as in the upstream ``load_sparse.py``).
    """
    import pickle

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    with h5py.File(h5_path, "r") as h5:
        x = np.asarray(h5["nodes"]["v1"]["0"]["x"], dtype=np.float32)
        z = np.asarray(h5["nodes"]["v1"]["0"]["z"], dtype=np.float32)
    sel = np.sqrt(x**2 + z**2) < core_radius
    n = int(sel.sum())
    bmtk_to_local = np.full(sel.size, -1, dtype=np.int64)
    bmtk_to_local[sel] = np.arange(n)

    scalar_keys = ["V_th", "E_L", "V_reset", "C_m", "g", "t_ref"]
    table = {k: [] for k in scalar_keys}
    k_rates, asc_amps = [], []
    node_type_id = np.full(n, -1, dtype=np.int64)
    for tid, nt in enumerate(d["nodes"]):
        loc = bmtk_to_local[np.asarray(nt["ids"])]
        loc = loc[loc >= 0]
        node_type_id[loc] = tid
        p = nt["params"]
        for k in scalar_keys:
            table[k].append(float(p[k]))
        k_rates.append(np.asarray(p["k"], dtype=np.float32))
        asc_amps.append(np.asarray(p["asc_amps"], dtype=np.float32))
    assert (node_type_id >= 0).all(), "every core neuron must map to a GLIF cell type"

    out = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
    out["k"] = np.stack(k_rates)                 # (n_types, 2) ASC decay rates
    out["asc_amps"] = np.stack(asc_amps)         # (n_types, 2) ASC amplitudes (pA)
    out["node_type_id"] = node_type_id           # (n,) type index per core neuron
    out["x"] = x[sel]
    out["z"] = z[sel]
    return out


def write_provenance(out_dir: Path, src: Path, mat, core_radius: float) -> None:
    n, nnz = mat.shape[0], mat.nnz
    text = (
        "# Provenance for the mice_v1_guozhang dataset (see scripts/build_mice_v1_guozhang.py).\n"
        f"source_dir: {src}\n"
        "source_files:\n"
        "  - network_dat.pkl\n"
        "  - network/v1_nodes.h5\n"
        "source_family: guozhang_mouse_v1\n"
        "selection: core\n"
        f"core_radius_um: {core_radius}\n"
        "receptor_types_collapsed: 4\n"
        "orientation: rows=presynaptic, cols=postsynaptic\n"
        "weighted: true\n"
        f"n_neurons: {n}\n"
        f"n_edges: {nnz}\n"
        f"density: {nnz / (n * n):.8f}\n"
    )
    (out_dir / "generation_params.yaml").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="GLIF_V1_network source dir")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dataset dir")
    ap.add_argument("--core-radius", type=float, default=CORE_RADIUS_UM)
    args = ap.parse_args()

    pkl_path = args.src / "network_dat.pkl"
    h5_path = args.src / "network" / "v1_nodes.h5"
    for p in (pkl_path, h5_path):
        if not p.exists():
            raise FileNotFoundError(p)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mat = build_adjacency(pkl_path, h5_path, args.core_radius)
    npz_path = args.out / "mice_v1_guozhang.npz"
    if npz_path.exists():  # DVC-managed symlink is read-only — keep the existing connectivity
        print(f"[keep] {npz_path} already exists (DVC-managed); not overwriting")
    else:
        sp.save_npz(npz_path, mat)
        write_provenance(args.out, args.src, mat, args.core_radius)

    neurons = build_neuron_params(pkl_path, h5_path, args.core_radius)
    neurons_path = args.out / "mice_v1_guozhang_neurons.npz"
    np.savez_compressed(neurons_path, **neurons)

    n, nnz = mat.shape[0], mat.nnz
    print(f"mice_v1_guozhang: n={n:,}  nnz={nnz:,}  density={nnz / (n * n):.3e}  "
          f"weight[min={mat.data.min():.3g}, max={mat.data.max():.3g}]  ({time.time() - t0:.1f}s)")
    print(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {neurons_path}: {neurons['k'].shape[0]} GLIF cell types, "
          f"{neurons['node_type_id'].size:,} neurons ({neurons_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
