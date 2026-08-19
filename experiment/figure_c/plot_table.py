"""Render the publication-style Figure C ablation table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_DIR = EXPERIMENT_DIR / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read normalized Figure C metrics."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise ValueError(f"expected four ablation variants, found {len(rows)}")
    return rows


def fmt_norm(row: dict[str, str], field: str) -> str:
    """Format one normalized metric with two decimal places."""

    return f"{float(row[field]):.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "figure_c_normalized.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Draw Table C as vector PDF/SVG and 600-DPI PNG."""

    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.05), constrained_layout=False)
    ax.set_axis_off()

    headers = (
        "Variant",
        "Block",
        "Sort",
        "Hash",
        "PROP\nTime ↓",
        "E2E\nStep ↓",
        "Global\nInst. ↓",
        "Atomic\nTx. ↓",
        "Long\nScorebd. ↓",
        "Issue\nBusy ↑",
    )
    body = []
    for row in rows:
        body.append(
            (
                row["label"],
                "✓" if row["block"] == "1" else "",
                "✓" if row["similarity"] == "1" else "",
                "✓" if row["hash"] == "1" else "",
                fmt_norm(row, "prop_time_ms_norm"),
                fmt_norm(row, "step_time_ms_norm"),
                fmt_norm(row, "global_mem_inst_norm"),
                fmt_norm(row, "atomic_tx_norm"),
                fmt_norm(row, "long_scoreboard_norm"),
                fmt_norm(row, "issue_busy_pct_norm"),
            )
        )

    column_widths = [
        0.125,
        0.060,
        0.065,
        0.060,
        0.090,
        0.090,
        0.110,
        0.100,
        0.140,
        0.100,
    ]
    table = ax.table(
        cellText=body,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths,
        bbox=[0.015, 0.25, 0.97, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.30)

    n_rows = len(body) + 1
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.45)
        cell.visible_edges = ""
        if row_index == 0:
            cell.set_text_props(weight="semibold")
            cell.visible_edges = "B"
            cell.set_linewidth(0.65)
        elif row_index == n_rows - 1:
            cell.visible_edges = "B"
            cell.set_linewidth(0.65)
        if column_index in (0, 3, 5, 6, 7, 8):
            cell.visible_edges += "R"
        if column_index == 0 and row_index > 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.10

    # Matplotlib's selected serif fonts do not consistently contain a check
    # mark. Use DejaVu Sans only for the three boolean optimization columns.
    for row_index in range(1, n_rows):
        for column_index in (1, 2, 3):
            table[row_index, column_index].get_text().set_fontfamily("DejaVu Sans")
            table[row_index, column_index].get_text().set_fontsize(9.0)

    # Add the reference image's clean top rule without copying its content.
    ax.add_line(
        Line2D(
            [0.015, 0.985],
            [0.93, 0.93],
            transform=ax.transAxes,
            color="black",
            linewidth=0.75,
        )
    )
    ax.text(
        0.015,
        0.13,
        (
            "Table C. Block and hash micro-ablation on FlyBrain, normalized "
            "to V0 Binning. Lower is better except Issue Busy."
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
    )
    ax.text(
        0.015,
        0.045,
        (
            "RTX 5090; batch=1; 128 steps; event rate=0.01. Atomic Tx denotes "
            "global reduction sectors; timing is the CUDA Event median."
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.7,
        color="#333333",
    )

    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            args.output_dir / f"table_c_block_hash_ablation.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            **kwargs,
        )
    plt.close(fig)
    print(f"Wrote Table C to {args.output_dir}")


if __name__ == "__main__":
    main()
