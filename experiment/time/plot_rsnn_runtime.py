"""Plot paired eager/CUDA Graph RSNN runtime breakdowns.

The script consumes schema-v4 repeat-level measurements for a 128-timestep
simulation. Each firing-rate group contains paired stacked bars for eager and
CUDA Graph execution. From bottom to top, each bar shows launch-and-execution
overhead, Update kernel-active time, and cuSPARSE Propagation kernel-active
time. The residual includes launch costs but is not a pure launch API metric.

Example:
    Generate the publication figure from the default result directory:

    >>> python experiment/time/plot_rsnn_runtime.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image


OKABE_ITO = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
}
FIGURE_SIZE = (7.2, 3.4)
EAGER_COMPONENTS = (
    "eager_launch_execution_overhead_ms",
    "eager_update_kernel_active_ms",
    "eager_propagation_kernel_active_ms",
)
GRAPH_COMPONENTS = (
    "cudagraph_launch_execution_overhead_ms",
    "cudagraph_update_kernel_active_ms",
    "cudagraph_propagation_kernel_active_ms",
)
COMPONENT_LABELS = (
    "Launch & execution overhead",
    "Update",
    "Propagation",
)
COMPONENT_COLORS = (
    OKABE_ITO["vermillion"],
    OKABE_ITO["orange"],
    OKABE_ITO["blue"],
)


def setup_style() -> None:
    """Configure an English Nature-style plotting theme."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": True,
        }
    )


def _relative_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return a symmetric relative difference with a stable zero case."""

    scale = pd.concat((left.abs(), right.abs()), axis=1).max(axis=1)
    difference = (left - right).abs()
    return difference.where(scale > 0.0, 0.0).div(scale.where(scale > 0.0, 1.0))


def component_consistency_table(
    frame: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """Compare mean eager and CUDA Graph kernel-active components."""

    grouped = frame.groupby("requested_firing_rate_hz", sort=True)
    table = grouped[
        [
            "eager_update_kernel_active_ms",
            "cudagraph_update_kernel_active_ms",
            "eager_propagation_kernel_active_ms",
            "cudagraph_propagation_kernel_active_ms",
        ]
    ].mean()
    table["update_relative_difference"] = _relative_difference(
        table["eager_update_kernel_active_ms"],
        table["cudagraph_update_kernel_active_ms"],
    )
    table["propagation_relative_difference"] = _relative_difference(
        table["eager_propagation_kernel_active_ms"],
        table["cudagraph_propagation_kernel_active_ms"],
    )
    table["consistency_status"] = np.where(
        table[
            [
                "update_relative_difference",
                "propagation_relative_difference",
            ]
        ].max(axis=1)
        <= tolerance,
        "ok",
        "mismatch",
    )
    return table.reset_index()


def load_results(path: Path, consistency_rtol: float) -> pd.DataFrame:
    """Load and validate repeat-level schema-v4 paired timing data."""

    frame = pd.read_csv(path)
    required = {
        "schema_version",
        "t_steps",
        "repeat",
        "requested_average_firing_partition",
        "actual_average_firing_partition",
        "requested_firing_rate_hz",
        "actual_firing_rate_hz",
        "eager_wall_total_ms",
        "cudagraph_wall_total_ms",
        "eager_kernel_profile_status",
        "cudagraph_kernel_profile_status",
        "eager_update_kernel_count",
        "eager_propagation_kernel_count",
        "cudagraph_update_kernel_count",
        "cudagraph_propagation_kernel_count",
        *EAGER_COMPONENTS,
        *GRAPH_COMPONENTS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing schema-v4 columns: {sorted(missing)}")
    if not (frame["schema_version"] == 4).all():
        raise ValueError("Expected only schema_version=4 rows")
    if not (frame["t_steps"] == 128).all():
        values = sorted(frame["t_steps"].unique())
        raise ValueError(f"Expected T=128 for every row, found {values}")
    if not frame["requested_firing_rate_hz"].between(0.0, 50.0).all():
        raise ValueError("Requested firing rates must lie in [0, 50] Hz")
    for column in (
        "eager_kernel_profile_status",
        "cudagraph_kernel_profile_status",
    ):
        if not (frame[column] == "ok").all():
            statuses = sorted(frame[column].astype(str).unique())
            raise ValueError(f"Incomplete {column}: {statuses}")

    components = frame[[*EAGER_COMPONENTS, *GRAPH_COMPONENTS]]
    if not np.isfinite(components.to_numpy()).all() or (components < 0).any().any():
        raise ValueError("Paired timing components must be finite and nonnegative")
    for mode, columns, total in (
        ("eager", EAGER_COMPONENTS, "eager_wall_total_ms"),
        ("CUDA Graph", GRAPH_COMPONENTS, "cudagraph_wall_total_ms"),
    ):
        reconstructed = frame[list(columns)].sum(axis=1)
        if not np.allclose(reconstructed, frame[total], rtol=1e-6, atol=1e-6):
            raise ValueError(f"{mode} components do not sum to end-to-end runtime")

    counts_match = (
        frame["eager_update_kernel_count"] == frame["cudagraph_update_kernel_count"]
    ) & (
        frame["eager_propagation_kernel_count"]
        == frame["cudagraph_propagation_kernel_count"]
    )
    if not counts_match.all():
        raise ValueError("Eager and CUDA Graph kernel counts differ")
    group_sizes = frame.groupby("requested_firing_rate_hz").size()
    if group_sizes.nunique() != 1 or group_sizes.iloc[0] < 2:
        raise ValueError("Every firing rate must have equal repeats and n >= 2")

    consistency = component_consistency_table(frame, consistency_rtol)
    mismatches = consistency[consistency["consistency_status"] != "ok"]
    if not mismatches.empty:
        details = mismatches[
            [
                "requested_firing_rate_hz",
                "update_relative_difference",
                "propagation_relative_difference",
            ]
        ].to_dict("records")
        raise ValueError(
            "Eager/CUDA Graph Update or Propagation differs beyond "
            f"rtol={consistency_rtol}: {details}"
        )
    return frame.sort_values(["requested_firing_rate_hz", "repeat"]).reset_index(
        drop=True
    )


def _export(fig: plt.Figure, basename: Path) -> None:
    """Export exact-size vector figures, PNG, and a grayscale preview."""

    basename.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*FIGURE_SIZE)
    fig.canvas.draw()
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            basename.with_suffix(f".{suffix}"),
            facecolor="white",
            transparent=False,
            **kwargs,
        )
    png_path = basename.with_suffix(".png")
    gray_path = basename.parent / f"{basename.name}_grayscale.png"
    with Image.open(png_path) as image:
        image.convert("L").save(gray_path, dpi=(600, 600))


def _draw_stacked_mode(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x: np.ndarray,
    columns: tuple[str, str, str],
    *,
    width: float,
    hatch: str | None,
) -> None:
    """Draw one mode's three-component stacked bars."""

    bottom = np.zeros(len(summary), dtype=float)
    for column, color in zip(columns, COMPONENT_COLORS, strict=True):
        values = summary[(column, "mean")].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            zorder=2,
        )
        bottom += values


def plot_paired_breakdown(
    frame: pd.DataFrame,
    output_dir: Path,
    consistency_rtol: float,
) -> dict[str, float]:
    """Plot paired stacked end-to-end runtime bars over firing rate."""

    grouped = frame.groupby("requested_firing_rate_hz", sort=True)
    columns = [
        "eager_wall_total_ms",
        "cudagraph_wall_total_ms",
        *EAGER_COMPONENTS,
        *GRAPH_COMPONENTS,
    ]
    summary = grouped[columns].agg(["mean", "std"])
    rates = summary.index.to_numpy(dtype=float)
    positions = np.arange(len(rates), dtype=float)
    width = 0.34
    eager_x = positions - width / 2.0
    graph_x = positions + width / 2.0

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _draw_stacked_mode(
        ax,
        summary,
        eager_x,
        EAGER_COMPONENTS,
        width=width,
        hatch=None,
    )
    _draw_stacked_mode(
        ax,
        summary,
        graph_x,
        GRAPH_COMPONENTS,
        width=width,
        hatch="////",
    )

    repeats = sorted(frame["repeat"].unique())
    jitter = dict(zip(repeats, np.linspace(-0.055, 0.055, len(repeats)), strict=True))
    rate_to_position = dict(zip(rates, positions, strict=True))
    for mode, total_column, offset, marker in (
        ("Eager", "eager_wall_total_ms", -width / 2.0, "o"),
        ("CUDA Graph", "cudagraph_wall_total_ms", width / 2.0, "D"),
    ):
        x_raw = frame.apply(
            lambda row: (
                rate_to_position[row["requested_firing_rate_hz"]]
                + offset
                + jitter[row["repeat"]]
            ),
            axis=1,
        )
        ax.scatter(
            x_raw,
            frame[total_column],
            s=7,
            marker=marker,
            facecolor="black",
            edgecolor="none",
            alpha=0.25,
            zorder=4,
        )
        mean = summary[(total_column, "mean")].to_numpy(dtype=float)
        sd = summary[(total_column, "std")].to_numpy(dtype=float)
        x_mode = eager_x if mode == "Eager" else graph_x
        ax.errorbar(
            x_mode,
            mean,
            yerr=sd,
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=5,
        )

    ax.set_xticks(positions, [f"{rate:g}" for rate in rates])
    ax.set_xlabel("Average firing rate (Hz)")
    ax.set_ylabel("End-to-end runtime for T=128 (ms)")
    eager_top = (
        summary[("eager_wall_total_ms", "mean")]
        + summary[("eager_wall_total_ms", "std")]
    ).max()
    graph_top = (
        summary[("cudagraph_wall_total_ms", "mean")]
        + summary[("cudagraph_wall_total_ms", "std")]
    ).max()
    ax.set_ylim(0, max(eager_top, graph_top) * 1.24)
    ax.set_title("Runtime breakdown with and without CUDA Graph")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)

    phase_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for color, label in zip(COMPONENT_COLORS, COMPONENT_LABELS, strict=True)
    ]
    mode_handles = [
        Patch(facecolor="white", edgecolor="black", label="Eager"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="////",
            label="CUDA Graph",
        ),
    ]
    ax.legend(
        handles=[*phase_handles, *mode_handles],
        frameon=False,
        loc="upper center",
        ncol=5,
    )

    basename = output_dir / "rsnn_runtime_paired_breakdown_t128"
    _export(fig, basename)
    plt.close(fig)

    consistency = component_consistency_table(frame, consistency_rtol)
    consistency.to_csv(
        output_dir / "rsnn_runtime_component_consistency.csv",
        index=False,
    )
    return {
        "n": float(grouped.size().iloc[0]),
        "max_update_relative_difference": float(
            consistency["update_relative_difference"].max()
        ),
        "max_propagation_relative_difference": float(
            consistency["propagation_relative_difference"].max()
        ),
    }


def write_figure_notes(
    frame: pd.DataFrame,
    audit: dict[str, float],
    output_dir: Path,
) -> None:
    """Write captions that state statistics and measurement semantics."""

    n = int(audit["n"])
    notes = f"""RSNN paired runtime-breakdown figure notes

Each firing-rate group shows paired stacked bars for conventional eager and
CUDA Graph execution of the same T=128 simulation. From bottom to top, the
components are Launch & execution overhead, Update, and Propagation. Bars are
component means across n={n} repeats. Black points show individual end-to-end
runtime measurements; error bars show mean ± SD.

The controlled average firing partition is converted to Hz as
rate_hz = partition * 1000 / dt_ms. The plotted range is 0–50 Hz. Update and
Propagation are raw CUDA kernel-active durations from separate CUPTI profiles
of eager execution and CUDA Graph replay. Their maximum mean relative
differences across firing rates are
{audit["max_update_relative_difference"] * 100:.2f}% (Update) and
{audit["max_propagation_relative_difference"] * 100:.2f}% (Propagation).

Launch & execution overhead is the additive residual between synchronized
wall time and the two kernel-active components. It includes critical-path
kernel launch/API cost, but also framework, graph-replay submission,
scheduling, synchronization, and GPU idle time. It is not a pure cumulative
launch API measurement.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rsnn_runtime_paired_figure_notes.txt").write_text(
        notes,
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "results" / "naive_rsnn_breakdown_v4.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "figures")
    parser.add_argument(
        "--component-consistency-rtol",
        type=float,
        default=0.15,
        help="Maximum allowed eager/Graph mean component difference.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.component_consistency_rtol <= 1.0:
        parser.error("--component-consistency-rtol must be in [0, 1]")
    return args


def main() -> None:
    """Generate the paired runtime figure, audit table, and caption notes."""

    args = parse_args()
    setup_style()
    frame = load_results(args.input, args.component_consistency_rtol)
    audit = plot_paired_breakdown(
        frame,
        args.output_dir,
        args.component_consistency_rtol,
    )
    write_figure_notes(frame, audit, args.output_dir)
    print(f"saved paired publication figure to {args.output_dir}")


if __name__ == "__main__":
    main()
