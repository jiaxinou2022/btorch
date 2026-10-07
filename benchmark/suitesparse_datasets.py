"""Load prepared SuiteSparse matrices for the RSNN benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse
import torch

from btorch.sparse import CSR


@dataclass(frozen=True)
class SuiteSparseDataset:
    """Describe one reproducible SuiteSparse input matrix."""

    group: str
    matrix: str
    rows: int
    nnz: int
    kind: str

    @property
    def url(self) -> str:
        """Return the canonical Matrix Market archive URL."""

        return (
            f"https://sparse.tamu.edu/MM/{self.group}/{self.matrix}.tar.gz"
        )


SUITESPARSE_DATASETS = {
    "orkut": SuiteSparseDataset(
        "SNAP", "com-Orkut", 3_072_441, 234_370_166, "binary"
    ),
    "hollywood_2009": SuiteSparseDataset(
        "LAW", "hollywood-2009", 1_139_905, 113_891_327, "binary"
    ),
    "vas_stokes_4m": SuiteSparseDataset(
        "VLSI", "vas_stokes_4M", 4_382_246, 131_577_616, "real"
    ),
    "uk_2002": SuiteSparseDataset(
        "LAW", "uk-2002", 18_520_486, 298_113_762, "binary"
    ),
    "queen_4147": SuiteSparseDataset(
        "Janna", "Queen_4147", 4_147_110, 316_548_962, "real"
    ),
}
DEFAULT_PROCESSED_ROOT = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "dataset"
    / "data"
    / "external"
    / "suitesparse"
    / "processed"
)


def resolve_prepared_path(root: Path | None, dataset: str) -> Path:
    """Resolve a prepared SciPy CSR archive from a file or directory."""

    if dataset not in SUITESPARSE_DATASETS:
        raise ValueError(f"Unsupported SuiteSparse dataset: {dataset}.")
    if root is None:
        root = DEFAULT_PROCESSED_ROOT
    root = Path(root)
    candidates = (
        root,
        root / f"{dataset}.npz",
        root / f"{SUITESPARSE_DATASETS[dataset].matrix}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No prepared CSR for {dataset} under {root}. Run "
        "benchmark/prepare_suitesparse_dataset.py first."
    )


def _normalized_values(
    matrix: scipy.sparse.csr_matrix,
    spec: SuiteSparseDataset,
    weight_scale: float,
) -> np.ndarray:
    """Return FP32 values with controlled mean absolute weighted fanout."""

    matrix.data = matrix.data.astype(np.float32, copy=False)
    if spec.kind == "binary":
        values = np.ones(matrix.nnz, dtype=np.float32)
        mean_abs_fanout = matrix.nnz / spec.rows
    else:
        values = matrix.data
        if not np.isfinite(values).all():
            raise ValueError("prepared matrix contains non-finite values")
        mean_abs_fanout = float(np.abs(values).sum(dtype=np.float64)) / spec.rows
    values *= weight_scale / max(mean_abs_fanout, 1.0)
    return values


def _torch_csr_from_scipy(
    matrix: scipy.sparse.csr_matrix,
    values: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Transfer a canonical SciPy CSR without a COO intermediate."""

    indptr = torch.as_tensor(matrix.indptr).to(device=device, dtype=torch.long)
    indices = torch.as_tensor(matrix.indices).to(device=device, dtype=torch.long)
    data = torch.as_tensor(values).to(device=device, dtype=torch.float32)
    return torch.sparse_csr_tensor(
        indptr,
        indices,
        data,
        size=matrix.shape,
        device=device,
        dtype=torch.float32,
    )


def load_suitesparse_csr(
    dataset: str,
    root: Path | None,
    *,
    weight_scale: float,
    device: torch.device,
) -> CSR:
    """Load a canonical SciPy CSR and transfer it directly to a device.

    Binary graph matrices receive one positive weight per stored edge. Real
    matrices retain their signs and relative magnitudes. In both cases the
    mean absolute weighted fanout is normalized to ``weight_scale``.
    """

    spec = SUITESPARSE_DATASETS[dataset]
    path = resolve_prepared_path(root, dataset)
    matrix = scipy.sparse.load_npz(path).tocsr(copy=False)
    if matrix.shape != (spec.rows, spec.rows):
        raise ValueError(
            f"{dataset} shape mismatch: expected {(spec.rows, spec.rows)}, "
            f"got {matrix.shape}."
        )
    if matrix.nnz != spec.nnz:
        raise ValueError(
            f"{dataset} nnz mismatch: expected {spec.nnz}, got {matrix.nnz}."
        )
    if not matrix.has_canonical_format:
        raise ValueError(f"{path} is not canonical CSR.")

    values = _normalized_values(matrix, spec, weight_scale)

    indptr = torch.as_tensor(matrix.indptr).to(device=device, dtype=torch.long)
    indices = torch.as_tensor(matrix.indices).to(device=device, dtype=torch.long)
    data = torch.as_tensor(values).to(device=device, dtype=torch.float32)
    result = CSR(indptr, indices, data, matrix.shape)

    transpose_path = path.with_name(f"{path.stem}_transpose.npz")
    if not transpose_path.is_file():
        raise FileNotFoundError(
            f"No prepared transposed CSR at {transpose_path}. Re-run the "
            "SuiteSparse preparation job."
        )
    transpose = scipy.sparse.load_npz(transpose_path).tocsr(copy=False)
    if transpose.shape != matrix.shape or transpose.nnz != matrix.nnz:
        raise ValueError(f"invalid transposed CSR: {transpose_path}")
    transpose_values = _normalized_values(transpose, spec, weight_scale)
    result._cache["torch_csr_weight"] = _torch_csr_from_scipy(
        transpose,
        transpose_values,
        device,
    )
    return result
