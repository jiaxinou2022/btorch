"""Convert a SuiteSparse Matrix Market archive into canonical SciPy CSR."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import scipy.io
import scipy.sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.suitesparse_datasets import SUITESPARSE_DATASETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUITESPARSE_DATASETS)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_matrix_member(
    archive: tarfile.TarFile,
    matrix_name: str,
) -> tarfile.TarInfo:
    """Return the named primary Matrix Market payload in an archive."""

    members = [
        member
        for member in archive.getmembers()
        if Path(member.name).name == f"{matrix_name}.mtx"
    ]
    if len(members) != 1:
        raise RuntimeError(
            f"expected one {matrix_name}.mtx member, found {len(members)}"
        )
    member = members[0]
    if not member.isfile():
        raise RuntimeError(f"Matrix Market member is not a file: {member.name}")
    return member


def main() -> None:
    """Prepare one matrix and write a validation manifest."""

    args = parse_args()
    spec = SUITESPARSE_DATASETS[args.dataset]
    archive_path = args.raw_root / spec.group / f"{spec.matrix}.tar.gz"
    output_path = args.output_root / f"{args.dataset}.npz"
    transpose_path = args.output_root / f"{args.dataset}_transpose.npz"
    manifest_path = args.output_root / f"{args.dataset}.json"
    if output_path.exists() and transpose_path.exists() and not args.force:
        print(f"[resume] {output_path}")
        return
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tarfile.open(archive_path, "r:gz") as archive:
        member = find_matrix_member(archive, spec.matrix)
        with tempfile.TemporaryDirectory(
            prefix=f"{args.dataset}-", dir=args.output_root
        ) as temporary:
            temporary_path = Path(temporary) / Path(member.name).name
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read {member.name}")
            with source, temporary_path.open("wb") as destination:
                while chunk := source.read(16 * 1024 * 1024):
                    destination.write(chunk)
            matrix = scipy.io.mmread(temporary_path, spmatrix=True).tocsr()

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.shape != (spec.rows, spec.rows):
        raise ValueError(f"shape mismatch: {matrix.shape}")
    if matrix.nnz != spec.nnz:
        raise ValueError(f"nnz mismatch: expected {spec.nnz}, got {matrix.nnz}")
    scipy.sparse.save_npz(output_path, matrix, compressed=False)
    transpose = matrix.transpose().tocsr()
    transpose.sum_duplicates()
    transpose.sort_indices()
    scipy.sparse.save_npz(transpose_path, transpose, compressed=False)
    manifest = {
        "dataset": args.dataset,
        "group": spec.group,
        "matrix": spec.matrix,
        "source_url": spec.url,
        "source_archive": str(archive_path),
        "output": str(output_path),
        "transpose_output": str(transpose_path),
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "dtype": str(matrix.dtype),
        "canonical_csr": bool(matrix.has_canonical_format),
        "elapsed_s": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
