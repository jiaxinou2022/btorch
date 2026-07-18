"""Export a benchmark matrix set to Matrix Market files for the C++ runner.

The C++ harness loads Matrix Market (.mtx); the connectome catalog stores graphml/etc.
This bridges them so the C++ cuSPARSE baseline runs on the same corpus as the Python
providers (same case_id, so records land in the same leaderboard cell).

    python scripts/export_corpus_mtx.py --matrix-set connectome --out .cache/mtx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io

from connectome_dataset.benchmarks import corpora
from connectome_dataset.benchmarks.cases import load_spmv_case


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-set", default="connectome", dest="matrix_set")
    p.add_argument("--out", default=".cache/mtx")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for graph in corpora.resolve_matrix_set(args.matrix_set):
        case = load_spmv_case(graph)
        # Name by graph_id: the C++ loader derives case_id = "<stem>_spmv", matching the
        # Python case_id so both land in the same leaderboard cell.
        dest = out / f"{case.graph_id}.mtx"
        scipy.io.mmwrite(str(dest), case.matrix, field="real", symmetry="general")
        print(f"{graph:28s} n={case.n:<8} nnz={case.nnz:<9} -> {dest}")


if __name__ == "__main__":
    main()
