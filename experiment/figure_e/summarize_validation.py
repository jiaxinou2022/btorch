"""Summarize CSR, GPU-memory, and benchmark smoke validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASET_ORDER = (
    "hollywood_2009",
    "vas_stokes_4m",
    "orkut",
    "uk_2002",
    "queen_4147",
)
LABELS = {
    "hollywood_2009": "Hollywood-2009",
    "vas_stokes_4m": "vas_stokes_4M",
    "orkut": "Orkut",
    "uk_2002": "uk-2002",
    "queen_4147": "Queen_4147",
}


def parse_args() -> argparse.Namespace:
    """Parse validation summary arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/figure_e"),
    )
    return parser.parse_args()


def main() -> None:
    """Build a compact reader-facing validation table."""

    args = parse_args()
    rows = []
    for dataset in DATASET_ORDER:
        memory = json.loads(
            (args.results_root / f"{dataset}_gpu_memory.json").read_text()
        )
        smoke = pd.read_csv(args.results_root / f"{dataset}_smoke.csv")
        statuses = smoke["correctness_status"].astype(str)
        smoke_passed = bool(statuses.str.startswith("passed").all())
        trace_gib = memory["t256_input_plus_reference_mib"] / 1024
        gpu_gib = memory["gpu_total_memory_mib"] / 1024
        rows.append(
            {
                "Dataset": LABELS[dataset],
                "Neurons": f"{memory['shape'][0]:,}",
                "Nonzeros": f"{memory['nnz']:,}",
                "Source CSR (GiB)": (
                    f"{memory['source_csr_storage_mib'] / 1024:.2f}"
                ),
                "Transpose CSR (GiB)": (
                    f"{memory['transposed_torch_csr_storage_mib'] / 1024:.2f}"
                ),
                "GPU peak allocated (GiB)": (
                    f"{memory['peak_allocated_mib'] / 1024:.2f}"
                ),
                "T=256 input + reference (GiB)": f"{trace_gib:.2f}",
                "CSR + SpMV": "pass" if memory["spmv_result_finite"] else "fail",
                "T=8 benchmark smoke": "pass" if smoke_passed else "fail",
                "Full T=256 sweep": (
                    "excluded: traces exceed VRAM"
                    if trace_gib >= gpu_gib
                    else "eligible"
                ),
            }
        )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "dataset_validation_table.csv", index=False)
    (args.output_dir / "dataset_validation_table.md").write_text(
        table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
