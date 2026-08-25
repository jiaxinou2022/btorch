"""Plot paired eager/CUDA Graph RSNN runtime breakdowns.

The script consumes schema-v5 repeat-level measurements for a 128-timestep
simulation. Each firing-rate group contains paired stacked bars for eager and
CUDA Graph execution. From bottom to top, each bar shows the non-kernel
critical-path residual, Update kernel-active time, and cuSPARSE Propagation
kernel-active time. A second panel shows host submission time and CPU launch API
count without adding overlapped host time to the end-to-end stack.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


OKABE_ITO = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
}
FIGURE_SIZE = (7.2, 3.6)
EAGER_COMPONENTS = (
    "eager_non_kernel_critical_path_residual_ms",
    "eager_update_kernel_active_ms",
    "eager_propagation_kernel_active_ms",
)
GRAPH_COMPONENTS = (
    "cudagraph_non_kernel_critical_path_residual_ms",
    "cudagraph_update_kernel_active_ms",
    "cudagraph_propagation_kernel_active_ms",
)
COMPONENT_LABELS = (
    "Non-kernel critical-path residual",
    "Update",
    "Propagation",
)
COMPONENT_COLORS = (
    OKABE_ITO["vermillion"],
    OKABE_ITO["orange"],
    OKABE_ITO["blue"],
)
HOST_COLUMNS = (
    "eager_host_submission_ms",
    "cudagraph_host_submission_ms",
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


def _profile_quality_reason(frame: pd.DataFrame) -> pd.Series:
    """Describe why a paired component profile cannot support a stack."""

    eager_residual = (
        frame["eager_wall_total_ms"]
        - frame["eager_update_kernel_active_ms"]
        - frame["eager_propagation_kernel_active_ms"]
    )
    graph_residual = (
        frame["cudagraph_wall_total_ms"]
        - frame["cudagraph_update_kernel_active_ms"]
        - frame["cudagraph_propagation_kernel_active_ms"]
    )
    reasons: list[str] = []
    for index in frame.index:
        row_reasons = []
        if frame.at[index, "eager_kernel_profile_status"] != "ok":
            row_reasons.append("eager_profile_failed")
        if frame.at[index, "cudagraph_kernel_profile_status"] != "ok":
            row_reasons.append("cudagraph_profile_failed")
        if frame.at[index, "component_consistency_status"] != "ok":
            row_reasons.append("paired_component_mismatch")
        if not np.isfinite(eager_residual.at[index]) or eager_residual.at[index] < 0:
            row_reasons.append("negative_eager_residual")
        if not np.isfinite(graph_residual.at[index]) or graph_residual.at[index] < 0:
            row_reasons.append("negative_cudagraph_residual")
        reasons.append(";".join(row_reasons))
    return pd.Series(reasons, index=frame.index, dtype="string")


def load_results(
    path: Path,
    consistency_rtol: float,
    min_valid_repeats: int,
) -> pd.DataFrame:
    """Load and validate repeat-level schema-v5 paired timing data."""

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
        *HOST_COLUMNS,
        "eager_kernel_profile_status",
        "cudagraph_kernel_profile_status",
        "component_consistency_status",
        "eager_update_kernel_count",
        "eager_propagation_kernel_count",
        "cudagraph_update_kernel_count",
        "cudagraph_propagation_kernel_count",
        "eager_cpu_launch_api_count",
        "cudagraph_cpu_launch_api_count",
        *EAGER_COMPONENTS,
        *GRAPH_COMPONENTS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing schema-v5 columns: {sorted(missing)}")
    if not (frame["schema_version"] == 5).all():
        raise ValueError("Expected only schema_version=5 rows")
    if not (frame["t_steps"] == 128).all():
        values = sorted(frame["t_steps"].unique())
        raise ValueError(f"Expected T=128 for every row, found {values}")
    if not frame["requested_firing_rate_hz"].between(0.0, 50.0).all():
        raise ValueError("Requested firing rates must lie in [0, 50] Hz")
    host_submission = frame[list(HOST_COLUMNS)]
    if (
        not np.isfinite(host_submission.to_numpy()).all()
        or (host_submission <= 0).any().any()
    ):
        raise ValueError("Host submission times must be finite and positive")
    components = frame[[*EAGER_COMPONENTS, *GRAPH_COMPONENTS]]
    finite_nonnegative = np.isfinite(components.to_numpy()).all(axis=1) & (
        components >= 0
    ).all(axis=1)

    counts_match = (
        frame["eager_update_kernel_count"] == frame["cudagraph_update_kernel_count"]
    ) & (
        frame["eager_propagation_kernel_count"]
        == frame["cudagraph_propagation_kernel_count"]
    )
    frame["component_profile_exclusion_reason"] = _profile_quality_reason(frame)

    def append_reason(mask: pd.Series | np.ndarray, reason: str) -> None:
        """Append one semicolon-delimited exclusion reason in place."""

        current = frame.loc[mask, "component_profile_exclusion_reason"]
        frame.loc[mask, "component_profile_exclusion_reason"] = current.map(
            lambda value: f"{value};{reason}" if value else reason
        )

    append_reason(~finite_nonnegative, "nonfinite_or_negative_component")
    append_reason(~counts_match, "kernel_count_mismatch")
    frame["component_profile_valid"] = (
        finite_nonnegative
        & counts_match
        & frame["component_profile_exclusion_reason"].eq("")
    )
    group_sizes = frame.groupby("requested_firing_rate_hz").size()
    if group_sizes.nunique() != 1 or group_sizes.iloc[0] < 2:
        raise ValueError("Every firing rate must have equal repeats and n >= 2")
    eager_launch_counts = frame["eager_cpu_launch_api_count"]
    graph_launch_counts = sorted(frame["cudagraph_cpu_launch_api_count"].unique())
    if (eager_launch_counts <= 1).any():
        raise ValueError("Eager launch count must exceed one")
    if graph_launch_counts != [1]:
        raise ValueError(f"Unexpected CUDA Graph launch counts: {graph_launch_counts}")

    valid = frame[frame["component_profile_valid"]]
    valid_group_sizes = valid.groupby("requested_firing_rate_hz").size()
    if (
        len(valid_group_sizes) != len(group_sizes)
        or (valid_group_sizes < min_valid_repeats).any()
    ):
        raise ValueError(
            "Insufficient uncontaminated component profiles; valid repeats by "
            f"rate: {valid_group_sizes.to_dict()}"
        )
    for mode, columns, total in (
        ("eager", EAGER_COMPONENTS, "eager_wall_total_ms"),
        ("CUDA Graph", GRAPH_COMPONENTS, "cudagraph_wall_total_ms"),
    ):
        reconstructed = valid[list(columns)].sum(axis=1)
        if not np.allclose(reconstructed, valid[total], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Valid {mode} components are not additive")

    consistency = component_consistency_table(valid, consistency_rtol)
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


def _draw_host_submission(ax: plt.Axes, frame: pd.DataFrame) -> dict[str, float]:
    """Draw host submission distributions without implying additivity."""

    values = [frame[column].to_numpy(dtype=float) for column in HOST_COLUMNS]
    boxes = ax.boxplot(
        values,
        positions=(0.0, 1.0),
        widths=0.44,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.7},
        capprops={"color": "black", "linewidth": 0.7},
        boxprops={"edgecolor": "black", "linewidth": 0.7},
    )
    for box, color, hatch in zip(
        boxes["boxes"],
        ("white", "white"),
        (None, "////"),
        strict=True,
    ):
        box.set_facecolor(color)
        box.set_hatch(hatch)

    for position, samples, marker in zip(
        (0.0, 1.0),
        values,
        ("o", "D"),
        strict=True,
    ):
        jitter = np.linspace(-0.13, 0.13, len(samples))
        ax.scatter(
            position + jitter,
            samples,
            s=7,
            marker=marker,
            facecolor=OKABE_ITO["blue"],
            edgecolor="none",
            alpha=0.28,
            zorder=3,
        )

    eager_launch_min = int(frame["eager_cpu_launch_api_count"].min())
    eager_launch_max = int(frame["eager_cpu_launch_api_count"].max())
    graph_launches = int(frame["cudagraph_cpu_launch_api_count"].iloc[0])
    eager_launch_label = (
        str(eager_launch_min)
        if eager_launch_min == eager_launch_max
        else f"{eager_launch_min}-{eager_launch_max}"
    )
    ax.set_xticks(
        (0.0, 1.0),
        (
            f"Eager\n{eager_launch_label} launches",
            f"CUDA Graph\n{graph_launches} launch",
        ),
    )
    ax.set_yscale("log")
    ax.set_ylabel("Host submission time (ms, log scale)")
    ax.set_title("b  Host submission", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    eager_median = float(np.median(values[0]))
    graph_median = float(np.median(values[1]))
    return {
        "eager_launch_count_min": float(eager_launch_min),
        "eager_launch_count_max": float(eager_launch_max),
        "graph_launch_count": float(graph_launches),
        "eager_submission_median_ms": eager_median,
        "graph_submission_median_ms": graph_median,
        "submission_reduction": eager_median / graph_median,
    }


def plot_paired_breakdown(
    frame: pd.DataFrame,
    output_dir: Path,
    consistency_rtol: float,
) -> dict[str, float]:
    """Plot paired stacked end-to-end runtime bars over firing rate."""

    valid = frame[frame["component_profile_valid"]].copy()
    excluded = frame[~frame["component_profile_valid"]].copy()
    grouped = valid.groupby("requested_firing_rate_hz", sort=True)
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

    fig, (ax, host_ax) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        gridspec_kw={"width_ratios": (2.35, 1.0)},
    )
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
        x_raw = valid.apply(
            lambda row: (
                rate_to_position[row["requested_firing_rate_hz"]]
                + offset
                + jitter[row["repeat"]]
            ),
            axis=1,
        )
        ax.scatter(
            x_raw,
            valid[total_column],
            s=7,
            marker=marker,
            facecolor="black",
            edgecolor="none",
            alpha=0.25,
            zorder=4,
        )
        if not excluded.empty:
            x_excluded = excluded.apply(
                lambda row: (
                    rate_to_position[row["requested_firing_rate_hz"]]
                    + offset
                    + jitter[row["repeat"]]
                ),
                axis=1,
            )
            ax.scatter(
                x_excluded,
                excluded[total_column],
                s=11,
                marker="x",
                color="#777777",
                linewidth=0.6,
                alpha=0.65,
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
    ax.set_title("a  End-to-end runtime", loc="left", fontweight="bold")
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
        Line2D(
            [],
            [],
            color="#777777",
            marker="x",
            linestyle="none",
            markersize=4,
            label="Excluded profile pair",
        ),
    ]
    ax.legend(
        handles=[*phase_handles, *mode_handles],
        frameon=False,
        loc="upper center",
        ncol=3,
    )
    host_audit = _draw_host_submission(host_ax, frame)

    basename = output_dir / "rsnn_runtime_launch_exposure_t128"
    _export(fig, basename)
    plt.close(fig)

    consistency = component_consistency_table(valid, consistency_rtol)
    consistency.to_csv(
        output_dir / "rsnn_runtime_component_consistency_v5.csv",
        index=False,
    )
    excluded.to_csv(
        output_dir / "rsnn_runtime_excluded_component_profiles_v5.csv",
        index=False,
    )
    valid_counts = grouped.size()
    return {
        "min_n": float(valid_counts.min()),
        "max_n": float(valid_counts.max()),
        "excluded_n": float(len(excluded)),
        "max_update_relative_difference": float(
            consistency["update_relative_difference"].max()
        ),
        "max_propagation_relative_difference": float(
            consistency["propagation_relative_difference"].max()
        ),
        **host_audit,
    }


def write_figure_notes(
    frame: pd.DataFrame,
    audit: dict[str, float],
    output_dir: Path,
) -> None:
    """Write captions that state statistics and measurement semantics."""

    min_n = int(audit["min_n"])
    max_n = int(audit["max_n"])
    excluded_n = int(audit["excluded_n"])
    eager_launch_min = int(audit["eager_launch_count_min"])
    eager_launch_max = int(audit["eager_launch_count_max"])
    eager_launch_range = (
        str(eager_launch_min)
        if eager_launch_min == eager_launch_max
        else f"{eager_launch_min}-{eager_launch_max}"
    )
    notes = f"""RSNN paired runtime-breakdown figure notes

Panel a shows paired stacked bars for conventional eager and CUDA Graph
execution of the same T=128 simulation. From bottom to top, the components are
Non-kernel critical-path residual, Update, and Propagation. Bars are
component means across n={min_n}–{max_n} valid paired profiles per firing rate.
Black points show the corresponding end-to-end runtime measurements; error
bars show mean ± SD. Gray crosses show {excluded_n} paired profiles excluded
from component summaries because of a failed profile, component mismatch,
kernel-count mismatch, or a negative residual caused by cross-pass timing
contamination. Excluded rows remain unchanged in the raw CSV and are listed in
rsnn_runtime_excluded_component_profiles_v5.csv.

The controlled average firing partition is converted to Hz as
rate_hz = partition * 1000 / dt_ms. The plotted range is 0–50 Hz. Update and
Propagation are raw CUDA kernel-active durations from separate CUPTI profiles
of eager execution and CUDA Graph replay. Their maximum mean relative
differences across firing rates are
{audit["max_update_relative_difference"] * 100:.2f}% (Update) and
{audit["max_propagation_relative_difference"] * 100:.2f}% (Propagation).

The non-kernel critical-path residual is the additive difference between
synchronized wall time and the two kernel-active components. It is not
cumulative launch API time because CPU submission overlaps GPU execution.

Panel b separately shows host submission time on a logarithmic axis. Boxes
show median and IQR; whiskers extend to 1.5 x IQR and all measurements are
overlaid. Eager submission issued {eager_launch_range} launch API calls,
whereas CUDA Graph replay issued {int(audit["graph_launch_count"])}. Median
host submission decreased from {audit["eager_submission_median_ms"]:.4f} ms
to {audit["graph_submission_median_ms"]:.4f} ms, a
{audit["submission_reduction"]:.1f}x reduction. Host submission is shown in a
separate panel and is not added to panel a because it overlaps device work.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rsnn_runtime_launch_exposure_notes.txt").write_text(
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
        default=base / "results" / "naive_rsnn_breakdown_v5.csv",
    )
    parser.add_argument(
        "--min-valid-repeats",
        type=int,
        default=5,
        help="Minimum valid paired component profiles required per rate.",
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
    if args.min_valid_repeats < 2:
        parser.error("--min-valid-repeats must be at least 2")
    return args


def main() -> None:
    """Generate the paired runtime figure, audit table, and caption notes."""

    args = parse_args()
    setup_style()
    frame = load_results(
        args.input,
        args.component_consistency_rtol,
        args.min_valid_repeats,
    )
    audit = plot_paired_breakdown(
        frame,
        args.output_dir,
        args.component_consistency_rtol,
    )
    write_figure_notes(frame, audit, args.output_dir)
    print(f"saved paired publication figure to {args.output_dir}")


if __name__ == "__main__":
    main()
