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
    if fmt == "suitesparse_mm":
        return load_suitesparse_mm(path)
    if fmt == "synthetic":
        base_name = entry["name"].rsplit("_x", 1)[0]
        times = int(entry["name"].rsplit("_x", 1)[1])
        base = next(g for g in catalog if g["name"] == base_name)
        return tile_matrix(load_graph_entry(base, catalog), times)
    raise ValueError(f"Unknown format: {fmt}")


def tile_matrix(mat: sp.spmatrix, times: int) -> sp.csr_matrix:
    """Create a block-diagonal matrix with `times` copies of `mat`.

    Each copy is independent (no inter-block edges), giving a synthetic
    network that is `times` × larger with identical local structure.
    """
    blocks = [mat] * times
    tiled = sp.block_diag(blocks, format="csr", dtype=np.float32)
    return tiled


def replicate_graph_entry(
    entry: dict[str, Any],
    catalog: list[dict[str, Any]],
    times: int,
) -> tuple[dict[str, Any], sp.csr_matrix]:
    """Load and block-diagonal replicate any square catalog graph."""
    if times < 1:
        raise ValueError(f"replicate times must be >= 1, got {times}")

    mat = load_graph_entry(entry, catalog)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Cannot replicate non-square graph {entry['name']}: {mat.shape}")

    if times == 1:
        return entry, mat

    replicated = tile_matrix(mat, times)
    n = replicated.shape[0]
    metadata = dict(entry)
    metadata.update(
        {
            "name": f"{entry['name']}_x{times}",
            "description": (
                f"Runtime {times}x block-diagonal replication of {entry['name']}."
            ),
            "source_format": "runtime_synthetic",
            "n_rows": int(n),
            "n_cols": int(n),
            "nnz": int(replicated.nnz),
            "density": float(replicated.nnz / (n * n)) if n else 0.0,
            "is_square": True,
            "is_symmetric": bool((replicated - replicated.T).nnz == 0),
            "notes": f"Runtime block-diagonal replication of {entry['name']} x {times}",
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
