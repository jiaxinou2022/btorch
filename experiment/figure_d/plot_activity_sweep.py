"""Aggregate all activity sweeps and render the publication figure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROVIDER_STYLE = {
    "globalatomic_cudagraph": ("GlobalAtomic", "#0072B2", "o", "-"),
    "tilespmspv_cudagraph": ("TileSpMSpV", "#E69F00", "s", "--"),
    "vdha_cudagraph": ("VDHA", "#009E73", "^", "-."),
    "persistent_spike_block": ("PaceRSNN", "#CC79A7", "D", "-"),
}
DATASET_ORDER = ("flybrain", "microns_mm3", "multiarea_mam", "uniform")
DATASET_TITLES = {
    "flybrain": "FlyBrain · 138,639 neurons · 15.1M synapses",
    "microns_mm3": "MICRONS mm³ · 60,048 neurons · 7.0M synapses",
    "multiarea_mam": "Macaque multi-area · 20,649 neurons · 22.0M synapses",
    "uniform": "Uniform · 131,072 neurons · 16.8M synapses",
}
DATASET_LABELS = {
    "flybrain": "FlyBrain",
    "microns_mm3": "MICRONS mm³",
    "multiarea_mam": "Macaque multi-area",
    "uniform": "Uniform",
}
BASELINE = "cusparse_direct_cudagraph"
FIGURE_SIZE = (7.2, 5.4)


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/figure_d"))
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def configure_style() -> tuple[object | None, object | None]:
    """Use SciPilot's journal style and QA helpers when available."""
    root = os.environ.get("SCIPILOT_FIGURE_SKILL_ROOT")
    if root:
        scripts = Path(root) / "scripts"
        sys.path.insert(0, str(scripts))
        from export_figure import export_figure
        from layout_tools import add_panel_labels, finalize_figure
        from setup_style import setup_style
        from visual_qa import audit_layout, print_report, render_preview

        setup_style(journal="nature", lang="en")
        return (
            (
                finalize_figure,
                add_panel_labels,
                render_preview,
                audit_layout,
                print_report,
            ),
            export_figure,
        )
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return None, None


def load_and_summarize(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw rows and aggregate independent workload seeds."""
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = {BASELINE, *PROVIDER_STYLE}
    if set(raw["dataset"]) != set(DATASET_ORDER):
        raise ValueError("inputs must contain exactly the four planned datasets")
    if set(raw["provider"]) != expected:
        raise ValueError("raw CSVs must contain exactly the five figure providers")
    keys = ["dataset", "input_seed", "requested_activity", "provider"]
    if raw.duplicated(keys).any():
        raise ValueError("duplicate provider rows within a workload")
    failed = raw[~raw["correctness_status"].str.startswith("passed")]
    if not failed.empty:
        raise ValueError(f"correctness or availability failures:\n{failed[keys]}")

    workload = ["dataset", "input_seed", "requested_activity"]
    baseline = raw[raw["provider"] == BASELINE][workload + ["latency_ms"]].rename(
        columns={"latency_ms": "baseline_latency_ms"}
    )
    data = raw.merge(baseline, on=workload)
    data["speedup"] = data["baseline_latency_ms"] / data["latency_ms"]
    methods = data[data["provider"].isin(PROVIDER_STYLE)].copy()
    summary = (
        methods.groupby(["dataset", "requested_activity", "provider"], as_index=False)
        .agg(
            measured_activity=("measured_activity", "median"),
            measured_min=("measured_activity", "min"),
            measured_max=("measured_activity", "max"),
            speedup=("speedup", "median"),
            speedup_min=("speedup", "min"),
            speedup_max=("speedup", "max"),
            latency_ms=("latency_ms", "median"),
            seeds=("input_seed", "nunique"),
        )
        .sort_values(["dataset", "provider", "measured_activity"])
    )
    if not (summary["seeds"] == 3).all():
        raise ValueError("every plotted point must contain three workload seeds")
    return raw, summary


def write_tables(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """Write combined raw data plus tidy and reader-facing summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = Path("benchmark/results/activity_sweep/all_datasets_rtx5090_raw.csv")
    combined.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(combined, index=False)
    summary.to_csv(output_dir / "activity_sweep_summary.csv", index=False)
    rows = []
    for (dataset, target), group in summary.groupby(
        ["dataset", "requested_activity"], sort=False
    ):
        row = {
            "Dataset": DATASET_LABELS[dataset],
            "Target activity": f"{100 * target:g}%",
            "Measured activity": (
                f"{100 * group['measured_activity'].median():.2f}% "
                f"[{100 * group['measured_min'].min():.2f}, "
                f"{100 * group['measured_max'].max():.2f}]"
            ),
        }
        for provider, (label, _, _, _) in PROVIDER_STYLE.items():
            value = group[group["provider"] == provider].iloc[0]
            row[label] = (
                f"{value['speedup']:.2f}× "
                f"[{value['speedup_min']:.2f}, {value['speedup_max']:.2f}]"
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "activity_sweep_table.csv", index=False)
    (output_dir / "activity_sweep_table.md").write_text(
        table.to_markdown(index=False) + "\n", encoding="utf-8"
    )


def draw(summary: pd.DataFrame) -> plt.Figure:
    """Draw small multiples against measured neuronal activity."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, DATASET_ORDER, strict=True):
        panel = summary[summary["dataset"] == dataset]
        for provider, (_, color, marker, linestyle) in PROVIDER_STYLE.items():
            group = panel[panel["provider"] == provider].sort_values(
                "measured_activity"
            )
            x = 100 * group["measured_activity"].to_numpy()
            y = group["speedup"].to_numpy()
            lower = y - group["speedup_min"].to_numpy()
            upper = group["speedup_max"].to_numpy() - y
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([lower, upper]),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.25,
                markersize=3.8,
                capsize=2,
                elinewidth=0.7,
            )
        ax.axhline(1.0, color="#D55E00", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.set_xlim(0.34, 23.0)
        ax.set_ylim(0.5, 11.0)
        ax.set_title(DATASET_TITLES[dataset], loc="left", y=0.98, pad=0, fontsize=7.2)
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.grid(axis="x", visible=False)
    fig.supxlabel("Measured average firing rate (%)", y=0.035)
    fig.supylabel("Speedup over cuSPARSE + CUDA Graph", x=0.018)
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=4,
            label=label,
        )
        for label, color, marker, linestyle in PROVIDER_STYLE.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#D55E00",
            linestyle=":",
            linewidth=1.0,
            label="cuSPARSE baseline (1×)",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )
    return fig


def main() -> None:
    """Create tables, preview, and optionally final vector outputs."""
    args = parse_args()
    qa_helpers, exporter = configure_style()
    raw, summary = load_and_summarize(args.inputs)
    write_tables(raw, summary, args.output_dir)
    fig = draw(summary)
    preview = args.output_dir / "activity_sweep_preview.png"
    if qa_helpers is not None:
        (
            finalize_figure,
            add_panel_labels,
            render_preview,
            audit_layout,
            print_report,
        ) = qa_helpers
        finalize_figure(fig)
        positions = (
            (0.095, 0.52, 0.415, 0.34),
            (0.57, 0.52, 0.415, 0.34),
            (0.095, 0.12, 0.415, 0.34),
            (0.57, 0.12, 0.415, 0.34),
        )
        for axis, position in zip(fig.axes, positions, strict=True):
            axis.set_position(position)
        add_panel_labels(fig, labels=["a", "b", "c", "d"], style="nature")
        render_preview(fig, preview, dpi=150)
        print_report(audit_layout(fig))
    else:
        fig.savefig(preview, dpi=150)
    if args.final:
        basename = str(args.output_dir / "figure_d_activity_sweep")
        if exporter is not None:
            exporter(
                fig,
                basename=basename,
                formats=["pdf", "svg", "png"],
                size_inches=FIGURE_SIZE,
                dpi=600,
                grayscale_preview=False,
                tight=False,
            )
            from PIL import Image

            Image.open(f"{basename}.png").convert("L").save(
                f"{basename}_grayscale.png", dpi=(600, 600)
            )
        else:
            fig.savefig(f"{basename}.pdf")
            fig.savefig(f"{basename}.svg")
            fig.savefig(f"{basename}.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
