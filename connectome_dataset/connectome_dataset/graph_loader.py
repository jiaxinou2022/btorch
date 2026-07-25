"""Load connectome graphs from various formats into scipy sparse CSR matrices.

Supported formats:
  - .graphml         (NetworkX)
  - .csv.zip         (networks.skewed.de edge list: source,target[,weight])
  - mice_column_v1   (processed parquet files: neurons + connections)
  - conn2res_csv     (conn2res toolbox: dense conn.csv adjacency matrix)
  - conn2res_npy     (conn2res toolbox: dense .npy adjacency matrix)
  - suitesparse_mm   (SuiteSparse Matrix Market .tar.gz archive)
  - synthetic tiling (tile an existing sparse matrix N times via block_diag)
"""

from __future__ import annotations

import zipfile
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


# ── default data root (repo root / data/) ──────────────────────────────────
# Works for editable installs; override with CONNECTOME_DATA_ROOT env var.
import os as _os
_REPO = Path(__file__).parent.parent
DATA_ROOT = Path(_os.environ.get("CONNECTOME_DATA_ROOT", _REPO / "data"))

MICE_COLUMN_V1_ROOT = DATA_ROOT / "external" / "mice_column_v1"
MICE_V1_GUOZHANG_ROOT = DATA_ROOT / "external" / "mice_v1_guozhang"
FLYWIRE_783_ROOT = DATA_ROOT / "external" / "flywire_783"
CONN2RES_DATA = DATA_ROOT / "conn2res"
SUITESPARSE_DATA = DATA_ROOT / "external" / "suitesparse"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_graphml(path: str | Path) -> sp.csr_matrix:
    import networkx as nx
    G = nx.read_graphml(str(path))
    return sp.csr_matrix(nx.to_scipy_sparse_array(G, dtype=np.float32))


def load_csv_zip(path: str | Path) -> sp.csr_matrix:
    """Load an edges.csv inside a .csv.zip (networks.skewed.de format).

    The CSV has a comment header line starting with '#', followed by
    columns: source, target[, weight, ...].  Only source/target/weight
    are used.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        edge_file = next(n for n in names if "edges" in n.lower())
        with zf.open(edge_file) as f:
            rows: list[int] = []
            cols: list[int] = []
            weights: list[float] = []
            node_map: dict[str, int] = {}

            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                src_str, tgt_str = parts[0].strip(), parts[1].strip()
                # Try each extra column for a valid float weight; default to 1.0
                w = 1.0
                for wstr in parts[2:]:
                    try:
                        w = float(wstr.strip())
                        break
                    except (ValueError, IndexError):
                        continue
                if src_str not in node_map:
                    node_map[src_str] = len(node_map)
                if tgt_str not in node_map:
                    node_map[tgt_str] = len(node_map)
                rows.append(node_map[src_str])
                cols.append(node_map[tgt_str])
                weights.append(w)

    n = len(node_map)
    mat = sp.csr_matrix(
        (np.array(weights, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    mat.sum_duplicates()
    return mat


def load_mice_column_v1(
    root: str | Path | None = None,
    use_weights: bool = False,
) -> sp.csr_matrix:
    """Load the mice_column_v1 Guozhang V1 network from processed parquet files.

    Args:
        root: Directory containing mice_connections_processed.parquet and
            mice_neurons_processed.parquet. Defaults to data/external/mice_column_v1.
        use_weights: If True, use the stored signed synaptic weights; otherwise
            return a binary adjacency matrix.
    """
    import pandas as pd

    root = Path(root) if root is not None else MICE_COLUMN_V1_ROOT
    neurons_path = root / "mice_neurons_processed.parquet"
    connections_path = root / "mice_connections_processed.parquet"

    if not neurons_path.exists() or not connections_path.exists():
        raise FileNotFoundError(
            "mice_column_v1 requires mice_neurons_processed.parquet and "
            f"mice_connections_processed.parquet under {root}"
        )

    neurons = pd.read_parquet(neurons_path, columns=["simple_id"])
    connections = pd.read_parquet(
        connections_path,
        columns=["pre_simple_id", "post_simple_id", "weight"],
    )
    n = int(neurons["simple_id"].max()) + 1
    data = (
        connections["weight"].to_numpy(dtype=np.float32)
        if use_weights
        else np.ones(len(connections), dtype=np.float32)
    )
    mat = sp.csr_matrix(
        (
            data,
            (
                connections["pre_simple_id"].to_numpy(dtype=np.int64),
                connections["post_simple_id"].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(n, n),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    return mat


def load_flywire_783(
    root: str | Path | None = None,
    use_weights: bool = True,
    return_ids: bool = False,
) -> sp.csr_matrix | tuple[sp.csr_matrix, np.ndarray]:
    """Load the whole-brain adult *Drosophila* connectome (FlyWire v783).

    Reads the Shiu et al. (2024) leaky-integrate-and-fire model's own preprocessed
    tables in their original format — ``Completeness_783.csv`` (the neuron list,
    indexed by FlyWire ID) and ``Connectivity_783.parquet`` (the synapse table) —
    and assembles the N x N adjacency with rows = presynaptic, cols = postsynaptic.

    The signed weight is the connectivity's ``Excitatory x Connectivity`` column
    (synapse count times the presynaptic sign, negative for inhibitory), which is
    exactly what the model multiplies by ``w_syn`` to get synaptic strength.

    Args:
        root: Directory holding Completeness_783.csv and Connectivity_783.parquet.
            Defaults to data/external/flywire_783.
        use_weights: If True, use the signed ``Excitatory x Connectivity`` weight;
            otherwise return a binary adjacency.
        return_ids: If True, also return the FlyWire ID for each row/col index
            (index i corresponds to ``flyids[i]``), which the btorch FlyBrain model
            needs to address neurons by FlyWire ID.

    Returns:
        The CSR adjacency, or ``(adjacency, flyids)`` when ``return_ids`` is set.
    """
    import pandas as pd

    root = Path(root) if root is not None else FLYWIRE_783_ROOT
    comp_path = root / "Completeness_783.csv"
    con_path = root / "Connectivity_783.parquet"
    if not comp_path.exists() or not con_path.exists():
        raise FileNotFoundError(
            "flywire_783 requires Completeness_783.csv and Connectivity_783.parquet "
            f"under {root} (see datasets/flywire_783/README.md)"
        )

    flyids = pd.read_csv(comp_path, index_col=0).index.to_numpy()
    n = len(flyids)
    con = pd.read_parquet(
        con_path,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    data = (
        con["Excitatory x Connectivity"].to_numpy(dtype=np.float32)
        if use_weights
        else np.ones(len(con), dtype=np.float32)
    )
    mat = sp.csr_matrix(
        (
            data,
            (
                con["Presynaptic_Index"].to_numpy(dtype=np.int64),
                con["Postsynaptic_Index"].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(n, n),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    return (mat, flyids) if return_ids else mat


def load_flywire_drive_types(root: str | Path | None = None) -> dict[str, list[int]]:
    """Sensory input ("drive") neuron groups for the FlyWire brain model.

    Returns ``{drive_type: [flywire_id, ...]}`` from ``datasets/flywire_783/drive_types.json``
    — the gustatory (sugar / bitter / Ir94e / water GRNs) and auditory/mechanosensory
    (Johnston's Organ) neuron sets the Shiu et al. experiments activate. Enable one, several,
    or (via ``all_flywire_drive_ids``) all of them at once as the model's optogenetic drive.
    """
    import json

    root = Path(root) if root is not None else FLYWIRE_783_ROOT
    # The neuron lists are repo-local metadata, not part of the (DVC-managed) data payload.
    meta = _REPO / "datasets" / "flywire_783" / "drive_types.json"
    if not meta.exists():
        raise FileNotFoundError(f"flywire drive types not found at {meta}")
    return json.loads(meta.read_text())["drive_types"]


def all_flywire_drive_ids(root: str | Path | None = None) -> list[int]:
    """Union of every FlyWire drive-type neuron id — all sensory drives enabled at once."""
    groups = load_flywire_drive_types(root)
    return sorted({int(i) for ids in groups.values() for i in ids})


def load_mice_v1_glif(
    root: str | Path | None = None,
    n_neurons: int | None = None,
    *,
    seed: int = 3000,
    connected_selection: bool = False,
) -> tuple[sp.csr_matrix, dict[str, np.ndarray]]:
    """Load the Billeh GLIF mouse-V1 core with per-cell-type neuron parameters, at any size.

    ``mice_v1_guozhang`` is the Billeh et al. (2020) point-GLIF V1 model: every neuron has a
    cell type, and each of the ~111 types carries a full GLIF3 parameter set. This returns the
    signed connectivity (pA) together with the **per-neuron** GLIF3 parameters expanded from the
    type table, so the btorch mouse-V1 model can build a heterogeneous population.

    ``n_neurons`` selects a configurable sub-network of the r<400um core (the whole core is
    51,978 neurons), mirroring the upstream ``load_sparse.py``: a random subsample by default,
    or the ``n_neurons`` closest to the column centre when ``connected_selection`` is set.

    Returns ``(W, neurons)`` where ``neurons`` maps ``v_threshold, v_reset, v_rest, c_m, tau,
    tau_ref`` (each ``(n,)``), ``k, asc_amps`` (each ``(n, 2)``) and ``node_type_id`` (``(n,)``).
    """
    root = Path(root) if root is not None else MICE_V1_GUOZHANG_ROOT
    conn_path = root / "mice_v1_guozhang.npz"
    neurons_path = root / "mice_v1_guozhang_neurons.npz"
    if not conn_path.exists() or not neurons_path.exists():
        raise FileNotFoundError(
            "mice_v1_guozhang requires mice_v1_guozhang.npz and mice_v1_guozhang_neurons.npz "
            f"under {root} (rebuild with scripts/build_mice_v1_guozhang.py)"
        )
    W = sp.load_npz(conn_path).tocsr()
    z = np.load(neurons_path)
    n_full = W.shape[0]
    tid = z["node_type_id"]

    if n_neurons is not None and 0 < n_neurons < n_full:
        if connected_selection:
            r = np.sqrt(z["x"].astype(np.float64) ** 2 + z["z"].astype(np.float64) ** 2)
            sel = np.sort(np.argsort(r)[:n_neurons])
        else:
            sel = np.sort(np.random.default_rng(seed).choice(n_full, size=n_neurons, replace=False))
        W = W[sel][:, sel].tocsr()
        tid = tid[sel]

    tau = (z["C_m"] / z["g"])[tid]
    neurons = _glif3_neurons(z["V_th"][tid], z["V_reset"][tid], z["E_L"][tid],
                             z["C_m"][tid], tau, z["t_ref"][tid], z["k"][tid], z["asc_amps"][tid], tid)
    return W, neurons


def _glif3_neurons(v_th, v_reset, v_rest, c_m, tau, t_ref, k, asc_amps, tid) -> dict[str, np.ndarray]:
    """Assemble the per-neuron GLIF3 parameter dict the btorch MouseV1GLIF model consumes."""
    return {
        "v_threshold": np.asarray(v_th, np.float32), "v_reset": np.asarray(v_reset, np.float32),
        "v_rest": np.asarray(v_rest, np.float32), "c_m": np.asarray(c_m, np.float32),
        "tau": np.asarray(tau, np.float32), "tau_ref": np.asarray(t_ref, np.float32),
        "k": np.asarray(k, np.float32), "asc_amps": np.asarray(asc_amps, np.float32),
        "node_type_id": np.asarray(tid, np.int64),
    }


def load_mice_column_v1_glif(
    root: str | Path | None = None, replicate: int = 1,
) -> tuple[sp.csr_matrix, dict[str, np.ndarray]]:
    """Load the ``mice_column_v1`` GLIF model — a *different* mouse-V1 model from the Billeh core.

    Guozhang's generated VISp column (from ``~/src/mice_unnamed_torch_dev``): signed connectome
    weights with per-cell-type Allen VISp GLIF fits (staged by ``scripts/build_mice_column_v1_glif.py``).
    Size scales by ``replicate`` — block-diagonal tiling of the 4,166-neuron column (the column's
    own scaling, not radial subsampling) — keeping its generated connectivity.

    Returns ``(W, neurons)`` with the same per-neuron GLIF3 dict as :func:`load_mice_v1_glif`.
    """
    root = Path(root) if root is not None else MICE_COLUMN_V1_ROOT
    neurons_path = root / "mice_column_v1_neurons.npz"
    if not neurons_path.exists():
        raise FileNotFoundError(
            f"mice_column_v1 GLIF params not staged at {neurons_path}; "
            "run scripts/build_mice_column_v1_glif.py"
        )
    W = load_mice_column_v1(root, use_weights=True)
    z = np.load(neurons_path)
    tid = z["node_type_id"]
    if replicate > 1:  # tile the column block-diagonally, as the --replicate size axis
        W = sp.block_diag([W] * replicate, format="csr")
        tid = np.tile(tid, replicate)
    idx = tid % z["V_th"].size
    neurons = _glif3_neurons(z["V_th"][idx], z["V_reset"][idx], z["E_L"][idx], z["C_m"][idx],
                             z["tau"][idx], z["t_ref"][idx], z["k"][idx], z["asc_amps"][idx], tid)
    return W.tocsr(), neurons


def load_mice_pkl(path: str | Path | None = None, use_weights: bool = False) -> sp.csr_matrix:
    """Backward-compatible alias for the repo-local mice_column_v1 parquet data.

    The old default loaded an external network_bundle.pkl. This repo must not
    depend on that checkout, so callers without an explicit path now load
    data/external/mice_column_v1 instead.
    """
    if path is None:
        return load_mice_column_v1(use_weights=use_weights)

    path = Path(path)
    if path.is_dir():
        return load_mice_column_v1(path, use_weights=use_weights)

    import pickle

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    net = bundle["net"]
    mat = sp.csr_matrix(net.weight_matrix if use_weights else net.adj_matrix)
    return mat.astype(np.float32)


def load_conn2res_csv(conn_dir: str | Path) -> sp.csr_matrix:
    """Load conn2res-format dense adjacency matrix (conn.csv) → sparse CSR.

    The file is a headerless CSV with one row per neuron/region.
    """
    path = Path(conn_dir) / "conn.csv"
    mat = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    return sp.csr_matrix(mat)


def load_conn2res_npy(path: str | Path) -> sp.csr_matrix:
    """Load a dense .npy adjacency matrix → sparse CSR."""
    mat = np.load(str(path)).astype(np.float32)
    return sp.csr_matrix(mat)


def load_npz(path: str | Path) -> sp.csr_matrix:
    """Load a scipy-saved sparse matrix (``.npz``) as CSR float32.

    Used by preprocessed dense-from-raw datasets (e.g. ``mice_v1_guozhang``, built once
    from the large GLIF V1 model by ``scripts/build_mice_v1_guozhang.py``).
    """
    return sp.load_npz(str(path)).tocsr().astype(np.float32)


def load_suitesparse_mm(path: str | Path) -> sp.csr_matrix:
    """Load a SuiteSparse Matrix Market archive or .mtx file as CSR."""
    from scipy.io import mmread

    path = Path(path)
    if path.suffix == ".mtx":
        mat = mmread(str(path))
        return sp.csr_matrix(mat, dtype=np.float32)

    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith(".mtx")), None)
            if member is None:
                raise FileNotFoundError(f"No .mtx file found in {path}")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Could not extract {member.name} from {path}")
            mat = mmread(extracted)
            return sp.csr_matrix(mat, dtype=np.float32)

    raise ValueError(f"Unsupported SuiteSparse Matrix Market path: {path}")


def load_graph_entry(entry: dict[str, Any], catalog: list[dict[str, Any]]) -> sp.csr_matrix:
    """Load a catalog entry into a CSR matrix.

    Synthetic entries are interpreted as block-diagonal replications of their
    base graph. This loader is shared by benchmarks so runtime graph loading
    stays consistent across PyTorch/JAX paths.
    """
    fmt, path = entry["source_format"], entry["source_path"]
    if fmt == "graphml":
        return load_graphml(path)
    if fmt == "csv.zip":
        return load_csv_zip(path)
    if fmt == "mice_column_v1":
        return load_mice_column_v1(path)
    if fmt == "pkl":
        return load_mice_pkl(path)
    if fmt == "conn2res_csv":
        return load_conn2res_csv(path)
    if fmt == "conn2res_npy":
        return load_conn2res_npy(path)
    if fmt == "npz":
        return load_npz(path)
    if fmt == "suitesparse_mm":
        return load_suitesparse_mm(path)
    if fmt == "synthetic":
        base_name = entry["name"].rsplit("_x", 1)[0]
        times = int(entry["name"].rsplit("_x", 1)[1])
        base = next(g for g in catalog if g["name"] == base_name)
        return tile_matrix(load_graph_entry(base, catalog), times)
    raise ValueError(f"Unknown format: {fmt}")


# Default density of the random inter-block edges added when replicating a graph:
# 1% of the inter-block region. Keeps scaled-up networks connected (a single network,
# not `times` disconnected islands) rather than pure block-diagonal. Configurable.
DEFAULT_REPLICATE_INTER_DENSITY = 0.01


def tile_matrix(
    mat: sp.spmatrix, times: int, *, inter_density: float = 0.0, seed: int = 0
) -> sp.csr_matrix:
    """Replicate `mat` `times` times block-diagonally, plus random inter-block edges.

    The block diagonal gives a network `times` × larger with identical local
    structure. With ``inter_density > 0`` we also fill the off-diagonal (inter-block)
    region at that density — ``round(inter_density · (N² − times·n²))`` random directed
    edges whose endpoints fall in *different* blocks — so the copies form one connected
    network instead of isolated islands. Inter-block weights are drawn ``N(0, 0.1)``
    (matching the synthetic ER generator) and merged with the tiled edges. Deterministic
    in ``seed``.
    """
    tiled = sp.block_diag([mat] * times, format="coo", dtype=np.float32)
    n = mat.shape[0]
    total = n * times
    inter_pairs = total * total - times * n * n  # off-block-diagonal ordered pairs
    n_inter = int(round(inter_density * inter_pairs)) if times > 1 else 0
    if n_inter <= 0:
        return tiled.tocsr()

    rng = np.random.default_rng(seed)
    rows, cols, got = [], [], 0
    while got < n_inter:  # reject intra-block / self pairs, oversampling to fill
        draw = 2 * (n_inter - got)
        r = rng.integers(0, total, size=draw)
        c = rng.integers(0, total, size=draw)
        keep = (r // n) != (c // n)
        rows.append(r[keep])
        cols.append(c[keep])
        got += int(keep.sum())
    r = np.concatenate(rows)[:n_inter]
    c = np.concatenate(cols)[:n_inter]
    w = (rng.standard_normal(n_inter) * 0.1).astype(np.float32)
    inter = sp.coo_matrix((w, (r, c)), shape=(total, total), dtype=np.float32)
    out = (tiled + inter).tocsr()
    out.sum_duplicates()
    return out


def replicate_graph_entry(
    entry: dict[str, Any],
    catalog: list[dict[str, Any]],
    times: int,
    *,
    inter_density: float = DEFAULT_REPLICATE_INTER_DENSITY,
    seed: int = 0,
) -> tuple[dict[str, Any], sp.csr_matrix]:
    """Load and replicate any square catalog graph (block-diagonal + random inter-block edges).

    ``inter_density`` is the density of the random cross-block edges (0 → pure
    block-diagonal). See :func:`tile_matrix`.
    """
    if times < 1:
        raise ValueError(f"replicate times must be >= 1, got {times}")

    mat = load_graph_entry(entry, catalog)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Cannot replicate non-square graph {entry['name']}: {mat.shape}")

    if times == 1:
        return entry, mat

    replicated = tile_matrix(mat, times, inter_density=inter_density, seed=seed)
    n = replicated.shape[0]
    kind = (
        "block-diagonal" if inter_density <= 0
        else f"block-diagonal + {inter_density:g}-density random inter-block"
    )
    metadata = dict(entry)
    metadata.update(
        {
            "name": f"{entry['name']}_x{times}",
            "description": f"Runtime {times}x {kind} replication of {entry['name']}.",
            "source_format": "runtime_synthetic",
            "n_rows": int(n),
            "n_cols": int(n),
            "nnz": int(replicated.nnz),
            "density": float(replicated.nnz / (n * n)) if n else 0.0,
            "is_square": True,
            "is_symmetric": bool((replicated - replicated.T).nnz == 0),
            "replicate_inter_density": float(inter_density),
            "notes": f"Runtime {kind} replication of {entry['name']} x {times}",
        }
    )
    return metadata, replicated


def graph_properties(mat: sp.spmatrix) -> dict[str, Any]:
    """Compute SuiteSparse-style graph properties from a sparse matrix."""
    mat = mat.tocsr().astype(np.float32)
    n_rows, n_cols = mat.shape
    nnz = mat.nnz

    is_square = n_rows == n_cols
    density = nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0.0

    is_symmetric = False
    if is_square:
        diff = mat - mat.T
        is_symmetric = diff.nnz == 0

    data = mat.data
    row_nnz = np.diff(mat.indptr)
    col_nnz = np.diff(mat.tocsc().indptr) if n_cols else np.array([], dtype=np.int64)

    def _summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
        if values.size == 0:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_p50": 0.0,
                f"{prefix}_p90": 0.0,
                f"{prefix}_p99": 0.0,
                f"{prefix}_gini": 0.0,
            }
        arr = values.astype(np.float64)
        return {
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_max": float(arr.max()),
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()),
            f"{prefix}_p50": float(np.percentile(arr, 50)),
            f"{prefix}_p90": float(np.percentile(arr, 90)),
            f"{prefix}_p99": float(np.percentile(arr, 99)),
            f"{prefix}_gini": float(_gini(arr)),
        }

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nnz": int(nnz),
        "density": float(density),
        "is_square": bool(is_square),
        "is_symmetric": bool(is_symmetric),
        "val_min": float(data.min()) if nnz > 0 else 0.0,
        "val_max": float(data.max()) if nnz > 0 else 0.0,
        "val_mean": float(data.mean()) if nnz > 0 else 0.0,
        "dtype": str(mat.dtype),
        **_summary(row_nnz, "row_nnz"),
        **_summary(row_nnz, "out_degree"),
        **_summary(col_nnz, "in_degree"),
    }


def _gini(values: np.ndarray) -> float:
    arr = np.sort(values.astype(np.float64))
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    index = np.arange(1, arr.size + 1)
    return float((2 * np.sum(index * arr) / (arr.size * np.sum(arr))) - (arr.size + 1) / arr.size)
