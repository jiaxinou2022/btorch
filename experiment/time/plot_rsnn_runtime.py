"""Plot publication-ready RSNN runtime characterization figures.

The script consumes schema-v3 repeat-level measurements. Update and synaptic
current computation are raw CUDA kernel-active durations collected by CUPTI.
Execution overhead is the remaining uninstrumented eager wall time and
therefore includes kernel launch/API costs as well as framework, scheduling,
allocation, synchronization, and GPU idle time.

Example:
    Generate figures from the default result directory:

    >>> python experiment/time/plot_rsnn_runtime.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#8A8A8A",
    "light_gray": "#D5D5D5",
}
COMPOSITION_SIZE = (3.5, 2.4)
SWEEP_SIZE = (7.2, 2.8)


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
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
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


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate repeat-level schema-v3 timing data."""

    frame = pd.read_csv(path)
    required = {
        "schema_version",
        "requested_activity",
        "repeat",
        "eager_wall_us_per_step",
        "update_kernel_active_us_per_step",
        "current_kernel_active_us_per_step",
        "execution_overhead_us_per_step",
        "kernel_profile_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing schema-v3 columns: {sorted(missing)}")
    if not (frame["schema_version"] == 3).all():
        raise ValueError("Expected only schema_version=3 rows")
    if not (frame["kernel_profile_status"] == "ok").all():
        statuses = sorted(frame["kernel_profile_status"].astype(str).unique())
        raise ValueError(f"CUPTI kernel profile is incomplete: {statuses}")
    components = frame[
        [
            "update_kernel_active_us_per_step",
            "current_kernel_active_us_per_step",
            "execution_overhead_us_per_step",
        ]
    ]
    if components.isna().any().any() or (components < 0).any().any():
        raise ValueError("Three-way timing components must be finite and nonnegative")
    reconstructed = components.sum(axis=1)
    if not np.allclose(
        reconstructed,
        frame["eager_wall_us_per_step"],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Three-way timing components do not sum to eager wall time")
    group_sizes = frame.groupby("requested_activity").size()
    if group_sizes.nunique() != 1 or group_sizes.iloc[0] < 2:
        raise ValueError(
            "Every firing rate must have the same number of repeats and n >= 2"
        )
    return frame.sort_values(["requested_activity", "repeat"]).reset_index(drop=True)


def _export(fig: plt.Figure, basename: Path, size: tuple[float, float]) -> None:
    """Export exact-size vector figures, PNG, and a grayscale preview."""

    basename.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*size)
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


def plot_composition(
    frame: pd.DataFrame,
    output_dir: Path,
    typical_activity: float,
) -> dict[str, float]:
    """Plot the normalized three-way eager runtime decomposition."""

    available = frame["requested_activity"].unique()
    activity = float(available[np.argmin(np.abs(available - typical_activity))])
    if not np.isclose(activity, typical_activity, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Requested typical activity {typical_activity} is unavailable; "
            f"choices are {available.tolist()}"
        )
    selected = frame[frame["requested_activity"] == activity].copy()

    component_columns = (
        "update_kernel_active_us_per_step",
        "current_kernel_active_us_per_step",
        "execution_overhead_us_per_step",
    )
    fractions = selected.loc[:, component_columns].div(
        selected["eager_wall_us_per_step"],
        axis=0,
    )
    components = fractions.mean().to_numpy(dtype=float, copy=True)
    components /= components.sum()

    fig, ax = plt.subplots(figsize=COMPOSITION_SIZE)
    colors = (
        OKABE_ITO["orange"],
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
    )
    hatches = ("///", "\\\\\\", "...")
    percentages = components * 100.0
    labels = (
        f"Update ({percentages[0]:.1f}%)",
        f"Synaptic current computation ({percentages[1]:.1f}%)",
        f"Execution overhead, incl. launch ({percentages[2]:.1f}%)",
    )
    left = 0.0
    for value, color, hatch, label in zip(
        components,
        colors,
        hatches,
        labels,
        strict=True,
    ):
        ax.barh(
            0,
            value * 100.0,
            left=left * 100.0,
            height=0.42,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            hatch=hatch,
            label=label,
        )
        left += value

    overhead_center = percentages[:2].sum() + percentages[2] / 2.0
    ax.text(
        overhead_center,
        0,
        f"{percentages[2]:.1f}%",
        ha="center",
        va="center",
        fontsize=6.3,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.43, 0.43)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of eager timestep time (%)")
    ax.set_title(
        f"Runtime composition at {activity * 100:g}% firing rate "
        f"(mean, n={len(selected)})"
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=1,
    )

    _export(
        fig,
        output_dir / "rsnn_runtime_three_way_composition",
        COMPOSITION_SIZE,
    )
    plt.close(fig)
    return {
        "activity": activity,
        "n": float(len(selected)),
        "update_pct": percentages[0],
        "current_pct": percentages[1],
        "overhead_pct": percentages[2],
    }


def _draw_repeat_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Draw repeat points, mean line, and an SD band."""

    grouped = frame.groupby("requested_activity", sort=True)[column]
    mean = grouped.mean()
    sd = grouped.std(ddof=1)
    x = mean.index.to_numpy(dtype=float)
    y = mean.to_numpy(dtype=float)
    error = sd.to_numpy(dtype=float)
    ax.fill_between(
        x,
        y - error,
        y + error,
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.7,
        label=label,
        zorder=3,
    )

    repeats = sorted(frame["repeat"].unique())
    offsets = dict(zip(repeats, np.linspace(-0.032, 0.032, len(repeats)), strict=True))
    jittered_x = frame.apply(
        lambda row: row["requested_activity"] * 2.0 ** offsets[row["repeat"]],
        axis=1,
    )
    ax.scatter(
        jittered_x,
        frame[column],
        s=7,
        facecolor=color,
        edgecolor="none",
        alpha=0.28,
        zorder=2,
    )


def plot_activity_sweep(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot the three absolute runtime components against firing rate."""

    fig, axes = plt.subplots(1, 2, figsize=SWEEP_SIZE, sharex=True)
    left, right = axes

    _draw_repeat_series(
        left,
        frame,
        "update_kernel_active_us_per_step",
        "Update",
        OKABE_ITO["orange"],
        "^",
        "-",
    )
    _draw_repeat_series(
        left,
        frame,
        "current_kernel_active_us_per_step",
        "Synaptic current computation",
        OKABE_ITO["blue"],
        "s",
        "--",
    )
    left.set_ylabel("Time per timestep (μs)")
    left.set_title("Kernel-active computation")
    left.legend(frameon=False, loc="best")

    _draw_repeat_series(
        right,
        frame,
        "execution_overhead_us_per_step",
        "Execution overhead (incl. launch)",
        OKABE_ITO["sky"],
        "o",
        "-",
    )
    right.set_ylabel("Time per timestep (μs)")
    right.set_title("Execution overhead")
    right.legend(frameon=False, loc="best")

    activities = np.sort(frame["requested_activity"].unique())
    labels = [f"{value * 100:g}" for value in activities]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlim(activities[0] / 1.18, activities[-1] * 1.18)
        ax.set_xticks(activities, labels)
        ax.set_xlabel("Firing rate (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":")
        ax.set_axisbelow(True)
    for ax, label in zip(axes, ("a", "b"), strict=True):
        ax.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-20, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            annotation_clip=False,
        )

    _export(fig, output_dir / "rsnn_runtime_three_way_activity_sweep", SWEEP_SIZE)
    plt.close(fig)


def write_figure_notes(
    composition: dict[str, float],
    output_dir: Path,
) -> None:
    """Write captions that state statistics and timing semantics."""

    n = int(composition["n"])
    activity = composition["activity"] * 100.0
    notes = f"""RSNN runtime characterization figure notes

Runtime composition
-------------------
Normalized runtime composition at {activity:g}% controlled firing rate.
The stacked bar reports the mean per-repeat fraction of uninstrumented eager
wall time across n={n} repeats: Update={composition["update_pct"]:.2f}%,
Synaptic current computation={composition["current_pct"]:.2f}%, and Execution
overhead={composition["overhead_pct"]:.2f}%. Update and synaptic current are
CUPTI raw CUDA kernel-active durations. Execution overhead is eager wall time
minus both kernel-active components. It includes repeated kernel launch/API
costs, but also framework, allocation, scheduling, synchronization, and GPU
idle time; it must not be interpreted as pure launch time.

Activity sweep
--------------
Points are individual benchmark repeats; lines are means and shaded bands are
mean ± SD, n={n} repeats per firing rate. Update and synaptic current are raw
kernel-active durations from a separate CUPTI profiling pass. Execution
overhead is the additive residual against the uninstrumented eager wall pass.
The cuSPARSE SpMV traverses the fixed CSR nonzeros at every firing rate, so its
runtime is not expected to scale with the number of nonzero spike values.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rsnn_runtime_figure_notes.txt").write_text(
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
        default=base / "results" / "naive_rsnn_breakdown_v3.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "figures")
    parser.add_argument("--typical-activity", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    """Generate both runtime figures and their caption notes."""

    args = parse_args()
    setup_style()
    frame = load_results(args.input)
    composition = plot_composition(
        frame,
        args.output_dir,
        args.typical_activity,
    )
    plot_activity_sweep(frame, args.output_dir)
    write_figure_notes(composition, args.output_dir)
    print(f"saved publication figures to {args.output_dir}")


if __name__ == "__main__":
    main()
