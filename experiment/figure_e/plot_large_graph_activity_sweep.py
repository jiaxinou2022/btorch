"""Render Figure E from provider-isolated large-graph activity sweeps."""

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
DATASET_ORDER = (
    "hollywood_2009",
    "vas_stokes_4m",
    "orkut",
    "queen_4147",
)
DATASET_TITLES = {
    "hollywood_2009": "Hollywood-2009 · 1.14M neurons · 113.9M nonzeros",
    "vas_stokes_4m": "vas_stokes_4M · 4.38M neurons · 131.6M nonzeros",
    "orkut": "Orkut · 3.07M neurons · 234.4M nonzeros",
    "queen_4147": "Queen_4147 · 4.15M neurons · 316.5M nonzeros",
}
DATASET_LABELS = {
    "hollywood_2009": "Hollywood-2009",
    "vas_stokes_4m": "vas_stokes_4M",
    "orkut": "Orkut",
    "queen_4147": "Queen_4147",
}
BASELINE = "cusparse_direct_cudagraph"
EXPECTED_RATES_HZ = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
FIGURE_SIZE = (7.2, 5.4)


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/figure_e"),
    )
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def configure_style() -> tuple[object | None, object | None]:
    """Use SciPilot's Nature style and QA helpers when available."""

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
    """Validate raw rows and aggregate the three workload seeds."""

    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected_providers = {BASELINE, *PROVIDER_STYLE}
    if set(raw["dataset"]) != set(DATASET_ORDER):
        raise ValueError("inputs must contain exactly the four Figure E datasets")
    if set(raw["provider"]) != expected_providers:
        raise ValueError("inputs must contain exactly the five figure providers")
    rates = sorted(raw["requested_firing_rate_hz"].unique())
    if not np.allclose(rates, EXPECTED_RATES_HZ):
        raise ValueError("inputs do not contain the planned 0--50 Hz sweep")
    keys = ["dataset", "input_seed", "requested_firing_rate_hz", "provider"]
    if raw.duplicated(keys).any():
        raise ValueError("duplicate provider rows within a workload")
    failed = raw[~raw["correctness_status"].str.startswith("passed")]
    if not failed.empty:
        raise ValueError(f"correctness or availability failures:\n{failed[keys]}")

    workload = ["dataset", "input_seed", "requested_firing_rate_hz"]
    baseline = raw[raw["provider"] == BASELINE][
        workload + ["latency_ms"]
    ].rename(columns={"latency_ms": "baseline_latency_ms"})
    methods = raw.merge(baseline, on=workload)
    methods["speedup"] = methods["baseline_latency_ms"] / methods["latency_ms"]
    methods = methods[methods["provider"].isin(PROVIDER_STYLE)].copy()
    summary = (
        methods.groupby(
            ["dataset", "requested_firing_rate_hz", "provider"],
            as_index=False,
        )
        .agg(
            measured_firing_rate_hz=("measured_firing_rate_hz", "median"),
            measured_min_hz=("measured_firing_rate_hz", "min"),
            measured_max_hz=("measured_firing_rate_hz", "max"),
            speedup=("speedup", "median"),
            speedup_min=("speedup", "min"),
            speedup_max=("speedup", "max"),
            latency_ms=("latency_ms", "median"),
            seeds=("input_seed", "nunique"),
        )
        .sort_values(["dataset", "provider", "measured_firing_rate_hz"])
    )
    if not (summary["seeds"] == 3).all():
        raise ValueError("every plotted point must contain three workload seeds")
    return raw, summary


def write_tables(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """Write combined raw data plus tidy and reader-facing tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "large_graph_activity_raw.csv", index=False)
    summary.to_csv(output_dir / "large_graph_activity_summary.csv", index=False)
    rows = []
    for (dataset, target), group in summary.groupby(
        ["dataset", "requested_firing_rate_hz"], sort=False
    ):
        row = {
            "Dataset": DATASET_LABELS[dataset],
            "Target firing rate": f"{target:g} Hz",
            "Measured firing rate": (
                f"{group['measured_firing_rate_hz'].median():.3g} Hz "
                f"[{group['measured_min_hz'].min():.3g}, "
                f"{group['measured_max_hz'].max():.3g}]"
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
    table.to_csv(output_dir / "large_graph_activity_table.csv", index=False)
    (output_dir / "large_graph_activity_table.md").write_text(
        table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def draw(summary: pd.DataFrame) -> plt.Figure:
    """Draw comparable small multiples against measured firing rate."""

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True, sharey=True)
    y_upper = max(2.0, float(np.ceil(summary["speedup_max"].max() * 1.08)))
    for axis, dataset in zip(axes.flat, DATASET_ORDER, strict=True):
        panel = summary[summary["dataset"] == dataset]
        for provider, (_, color, marker, linestyle) in PROVIDER_STYLE.items():
            group = panel[panel["provider"] == provider].sort_values(
                "measured_firing_rate_hz"
            )
            x = group["measured_firing_rate_hz"].to_numpy()
            y = group["speedup"].to_numpy()
            axis.errorbar(
                x,
                y,
                yerr=np.vstack(
                    [
                        y - group["speedup_min"].to_numpy(),
                        group["speedup_max"].to_numpy() - y,
                    ]
                ),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.25,
                markersize=3.8,
                capsize=2,
                elinewidth=0.7,
            )
        axis.axhline(1.0, color="#D55E00", linestyle=":", linewidth=1.0)
        axis.set_xscale("symlog", linthresh=0.1, linscale=0.7, base=10)
        axis.set_xticks([0, 0.1, 0.5, 1, 5, 10, 50])
        axis.set_xticklabels(["0", "0.1", "0.5", "1", "5", "10", "50"])
        axis.set_xlim(0.0, 55.0)
        axis.set_ylim(0.0, y_upper)
        axis.set_title(DATASET_TITLES[dataset], loc="left", y=0.98, pad=0, fontsize=7.2)
        axis.grid(axis="y", color="0.88", linewidth=0.5)
        axis.grid(axis="x", visible=False)
    fig.supxlabel("Average firing rate (Hz)", y=0.035)
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
    """Create tables, preview, and final publication outputs."""

    args = parse_args()
    qa_helpers, exporter = configure_style()
    raw, summary = load_and_summarize(args.inputs)
    write_tables(raw, summary, args.output_dir)
    figure = draw(summary)
    preview = args.output_dir / "large_graph_activity_preview.png"
    if qa_helpers is not None:
        (
            finalize_figure,
            add_panel_labels,
            render_preview,
            audit_layout,
            print_report,
        ) = qa_helpers
        finalize_figure(figure)
        positions = (
            (0.095, 0.52, 0.415, 0.34),
            (0.57, 0.52, 0.415, 0.34),
            (0.095, 0.12, 0.415, 0.34),
            (0.57, 0.12, 0.415, 0.34),
        )
        for axis, position in zip(figure.axes, positions, strict=True):
            axis.set_position(position)
        add_panel_labels(figure, labels=["a", "b", "c", "d"], style="nature")
        render_preview(figure, preview, dpi=150)
        print_report(audit_layout(figure))
    else:
        figure.savefig(preview, dpi=150)
    if args.final:
        basename = str(args.output_dir / "figure_e_large_graph_activity_sweep")
        if exporter is not None:
            exporter(
                figure,
                basename=basename,
                formats=["pdf", "svg", "png"],
                size_inches=FIGURE_SIZE,
                dpi=600,
                grayscale_preview=True,
                tight=False,
            )
        else:
            figure.savefig(f"{basename}.pdf")
            figure.savefig(f"{basename}.svg")
            figure.savefig(f"{basename}.png", dpi=600)
    plt.close(figure)


if __name__ == "__main__":
    main()
