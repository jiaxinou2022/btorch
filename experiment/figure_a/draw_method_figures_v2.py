"""Redraw the image-generated RSNN method concepts as vector figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
SKILL_SCRIPTS = Path(
    "/home/lenovo/.codex/skills/scipilot-figure-skill/scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


INK = "#202124"
MUTED = "#5F6368"
GRAY = "#B9BDC4"
LIGHT_GRAY = "#F4F5F6"
MEMORY = "#E7E8EA"
BLUE = "#179AD6"
BLUE_LIGHT = "#DCEFFA"
ORANGE = "#E69F00"
ORANGE_LIGHT = "#FBE8BE"
GREEN = "#17823B"
GREEN_LIGHT = "#DFF1E1"
PURPLE = "#7440A5"
PURPLE_LIGHT = "#E8DFF2"
RED = "#D55E00"
RED_LIGHT = "#F6DDD2"


def setup_style() -> None:
    """Configure publication-safe vector output."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fc: str = "white",
    ec: str = INK,
    lw: float = 1.0,
    ls: str = "-",
    radius: float = 0.010,
    zorder: int = 1,
) -> FancyBboxPatch:
    """Draw a rounded rectangle in normalized figure coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 7.0,
    weight: str = "normal",
    color: str = INK,
    ha: str = "center",
    va: str = "center",
    zorder: int = 8,
    **kwargs,
):
    """Draw consistently styled text."""
    return ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
        **kwargs,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.2,
    ls: str = "-",
    scale: float = 9,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    zorder: int = 6,
) -> FancyArrowPatch:
    """Draw a directional data path."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def cells(
    ax,
    x: float,
    y: float,
    values: list[str],
    *,
    cell_w: float,
    cell_h: float,
    colors: list[str] | None = None,
    edgecolor: str = INK,
    linewidth: float = 0.55,
    fontsize: float = 5.5,
    zorder: int = 4,
) -> None:
    """Draw an array as individually addressable cells."""
    for index, value in enumerate(values):
        color = colors[index] if colors else "white"
        ax.add_patch(
            Rectangle(
                (x + index * cell_w, y),
                cell_w,
                cell_h,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
        )
        if value:
            text(
                ax,
                x + (index + 0.5) * cell_w,
                y + cell_h / 2,
                value,
                size=fontsize,
                zorder=zorder + 1,
            )


def neuron(ax, x: float, y: float, active: bool, name: str) -> None:
    """Draw a neuron glyph and its identifier."""
    ax.add_patch(
        Circle(
            (x, y),
            0.009,
            transform=ax.transAxes,
            facecolor=BLUE if active else "white",
            edgecolor=INK,
            linewidth=0.8,
            zorder=6,
        )
    )
    text(ax, x - 0.017, y, name, size=5.8, ha="right")


def build_figure_a_v2():
    """Build the nested neuron-task-warp-block propagation figure."""
    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    text(
        ax,
        0.5,
        0.972,
        "Nested Allocation of Block-Oriented Propagation Work",
        size=10.8,
        weight="bold",
    )
    text(
        ax,
        0.5,
        0.943,
        "The same toy CSR work is partitioned at neuron, warp, and block levels",
        size=6.5,
        color=MUTED,
    )

    # Outermost block/CTA allocation encloses the whole toy workload.
    box(
        ax,
        0.035,
        0.105,
        0.930,
        0.805,
        fc="#FBFDFB",
        ec=GREEN,
        lw=1.7,
        radius=0.018,
    )
    box(ax, 0.050, 0.862, 0.200, 0.040, fc=GREEN_LIGHT, ec=GREEN)
    text(
        ax,
        0.150,
        0.882,
        "BLOCK / CTA ALLOCATION",
        size=6.7,
        weight="bold",
        color=GREEN,
    )

    # Warp boundary contains a shared 32-neuron work view and its task stream.
    box(
        ax,
        0.060,
        0.240,
        0.645,
        0.600,
        fc="white",
        ec=BLUE,
        lw=1.6,
        ls="--",
        radius=0.016,
    )
    box(ax, 0.075, 0.808, 0.160, 0.040, fc=BLUE_LIGHT, ec=BLUE, ls="--")
    text(
        ax,
        0.155,
        0.828,
        "WARP: 32 LANES",
        size=6.8,
        weight="bold",
        color="#166B8E",
    )

    text(ax, 0.083, 0.783, "Neuron ID", size=6.1, weight="bold")
    text(ax, 0.205, 0.783, "CSR row span", size=6.1, weight="bold")
    text(ax, 0.423, 0.783, "Active-edge prefix", size=6.1, weight="bold")
    text(ax, 0.600, 0.783, "Bounded logical tasks", size=6.1, weight="bold")

    row_y = [0.730, 0.664, 0.598, 0.532, 0.466, 0.400]
    names = ["N0", "N1", "N2", "N3", "N4", "N31"]
    active = [True, False, True, False, True, True]
    row_values = [
        ["3", "7", "7", "9", ""],
        ["", "", "", "", ""],
        ["1", "7", "2", "", ""],
        ["", "", "", "", ""],
        ["7", "3", "7", "9", "7"],
        ["3", "8", "", "", ""],
    ]
    prefixes = [
        ["3", "7", "7", "9"],
        ["1", "7", "2"],
        ["7", "3", "7", "9", "7"],
        ["3", "8"],
    ]
    active_index = 0
    for y, name, is_active, values in zip(row_y, names, active, row_values):
        neuron(ax, 0.094, y, is_active, name)
        arrow(ax, (0.106, y), (0.126, y), lw=0.7, scale=6)
        cells(
            ax,
            0.130,
            y - 0.018,
            values,
            cell_w=0.039,
            cell_h=0.036,
            colors=[BLUE_LIGHT] * 5 if is_active else ["white"] * 5,
        )
        if is_active:
            box(
                ax,
                0.121,
                y - 0.027,
                0.214,
                0.054,
                fc="none",
                ec=ORANGE,
                lw=1.2,
                radius=0.008,
                zorder=5,
            )
            prefix = prefixes[active_index]
            py = y - 0.018
            cells(
                ax,
                0.365,
                py,
                prefix,
                cell_w=0.028,
                cell_h=0.036,
                colors=[LIGHT_GRAY] * len(prefix),
            )
            arrow(ax, (0.337, y), (0.369, y), color=ORANGE, lw=0.9)
            task_w = max(0.078, len(prefix) * 0.030 + 0.018)
            box(
                ax,
                0.538,
                y - 0.026,
                task_w,
                0.052,
                fc=ORANGE_LIGHT,
                ec=ORANGE,
                lw=1.0,
                ls="--",
                radius=0.007,
            )
            cells(
                ax,
                0.547,
                py,
                prefix,
                cell_w=0.030,
                cell_h=0.036,
                colors=[ORANGE_LIGHT] * len(prefix),
                fontsize=5.0,
            )
            arrow(
                ax,
                (0.365 + len(prefix) * 0.028, y),
                (0.534, y),
                color=BLUE,
                lw=1.0,
            )
            active_index += 1

    # High-fanout branch remains within the CTA but outside normal warp work.
    text(
        ax,
        0.079,
        0.302,
        "Exceptional high-fanout row (degree >= 256)",
        size=5.7,
        weight="bold",
        ha="left",
    )
    neuron(ax, 0.094, 0.270, True, "Nk")
    long_values = ["4", "7", "3", "1", "9", "7", "2", "3", "7", "6", "3", "7"]
    cells(
        ax,
        0.130,
        0.252,
        long_values,
        cell_w=0.025,
        cell_h=0.036,
        colors=[RED_LIGHT] * len(long_values),
        fontsize=4.8,
    )
    for index in range(3):
        box(
            ax,
            0.450 + index * 0.080,
            0.245,
            0.075,
            0.050,
            fc=RED_LIGHT,
            ec=RED,
            ls="--",
            radius=0.006,
        )
        text(
            ax,
            0.4875 + index * 0.080,
            0.270,
            f"Fragment {index}",
            size=4.6,
        )
    arrow(ax, (0.515, 0.232), (0.515, 0.184), color=ORANGE, lw=1.4)

    # Dynamic scheduler joins normal logical tasks and long-row fragments.
    box(
        ax,
        0.250,
        0.125,
        0.410,
        0.055,
        fc=PURPLE_LIGHT,
        ec=PURPLE,
        lw=1.2,
        radius=0.009,
    )
    text(ax, 0.455, 0.1525, "Dynamic PROP pool", size=6.6, weight="bold")
    arrow(
        ax,
        (0.632, 0.400),
        (0.570, 0.183),
        color=BLUE,
        lw=1.0,
        ls="--",
        connectionstyle="arc3,rad=0.15",
    )

    # Block-local shared hash consumes the same toy task destinations.
    box(
        ax,
        0.725,
        0.295,
        0.215,
        0.500,
        fc=GREEN_LIGHT,
        ec=GREEN,
        lw=1.4,
        radius=0.014,
    )
    text(
        ax,
        0.8325,
        0.768,
        "Block-local shared hash",
        size=6.8,
        weight="bold",
        color=GREEN,
    )
    text(ax, 0.755, 0.725, "Post", size=5.6, weight="bold")
    text(ax, 0.855, 0.725, "Accumulated value", size=5.6, weight="bold")
    hash_posts = ["3", "7", "9", "1", "2", "8"]
    for index, post in enumerate(hash_posts):
        y = 0.671 - index * 0.056
        cells(
            ax,
            0.747,
            y,
            [post],
            cell_w=0.050,
            cell_h=0.043,
            colors=["white"],
        )
        cells(
            ax,
            0.797,
            y,
            [f"sum(w -> p{post})"],
            cell_w=0.118,
            cell_h=0.043,
            colors=["white"],
            fontsize=4.6,
        )
    for source_y, target_y in zip(
        [0.730, 0.664, 0.598, 0.466, 0.400],
        [0.692, 0.636, 0.580, 0.692, 0.636],
    ):
        arrow(
            ax,
            (0.704, source_y),
            (0.743, target_y),
            color=INK,
            lw=1.0,
            connectionstyle="arc3,rad=0.12",
        )

    text(ax, 0.832, 0.265, "Unique-post atomic flush", size=5.6, color=PURPLE)
    cells(
        ax,
        0.746,
        0.195,
        ["PSC[3]", "PSC[7]", "PSC[9]"],
        cell_w=0.057,
        cell_h=0.046,
        colors=[PURPLE_LIGHT] * 3,
        fontsize=4.7,
    )
    for index in range(3):
        arrow(
            ax,
            (0.772 + index * 0.057, 0.295),
            (0.774 + index * 0.057, 0.244),
            color=PURPLE,
            lw=1.4,
        )
    text(ax, 0.832, 0.167, "Global PSC", size=6.2, weight="bold")

    # Compact legend retains meaning in grayscale.
    legend_y = 0.057
    neuron(ax, 0.090, legend_y, True, "")
    text(ax, 0.105, legend_y, "active", size=5.2, ha="left")
    neuron(ax, 0.166, legend_y, False, "")
    text(ax, 0.181, legend_y, "inactive", size=5.2, ha="left")
    box(ax, 0.245, 0.043, 0.050, 0.028, fc="none", ec=ORANGE, radius=0.004)
    text(ax, 0.305, legend_y, "neuron task", size=5.2, ha="left")
    box(ax, 0.402, 0.043, 0.050, 0.028, fc="none", ec=BLUE, ls="--")
    text(ax, 0.462, legend_y, "warp", size=5.2, ha="left")
    box(ax, 0.530, 0.043, 0.050, 0.028, fc="none", ec=GREEN)
    text(ax, 0.590, legend_y, "block", size=5.2, ha="left")
    arrow(ax, (0.665, legend_y), (0.715, legend_y), color=PURPLE, lw=1.5)
    text(ax, 0.725, legend_y, "global atomic", size=5.2, ha="left")
    return fig


def ownership_strip(ax, x: float, y: float) -> None:
    """Draw contiguous CTA ownership across a neuron array."""
    colors = [GREEN_LIGHT] * 6 + [BLUE_LIGHT] * 5 + [ORANGE_LIGHT] * 5
    cells(
        ax,
        x,
        y,
        [""] * 16,
        cell_w=0.016,
        cell_h=0.025,
        colors=colors,
        linewidth=0.45,
    )
    text(ax, x + 0.048, y - 0.017, "CTA 0", size=4.8, color=GREEN)
    text(ax, x + 0.136, y - 0.017, "CTA 1", size=4.8, color="#166B8E")
    text(ax, x + 0.216, y - 0.017, "CTA 2", size=4.8, color=RED)


def warp_lanes(ax, x: float, y: float, token: bool = False) -> None:
    """Draw eight visible warp-lane strips."""
    box(ax, x, y, 0.150, 0.190, fc="white", ec=GRAY, radius=0.010)
    text(ax, x + 0.075, y + 0.172, "8 warp lanes", size=5.7, weight="bold")
    for lane in range(8):
        yy = y + 0.138 - lane * 0.017
        text(ax, x + 0.012, yy + 0.006, str(lane), size=4.2)
        lane_colors = ["white"] * 7
        if token and lane < 4:
            lane_colors[lane + 1] = ORANGE_LIGHT
        elif not token:
            lane_colors[(lane * 3) % 7] = GREEN_LIGHT
        cells(
            ax,
            x + 0.024,
            yy,
            [""] * 7,
            cell_w=0.015,
            cell_h=0.012,
            colors=lane_colors,
            linewidth=0.35,
        )


def stage_circle(ax, x: float, y: float, title: str) -> None:
    """Draw a timestep update stage."""
    ax.add_patch(
        Circle(
            (x, y),
            0.032,
            transform=ax.transAxes,
            facecolor=GREEN_LIGHT,
            edgecolor=GREEN,
            linewidth=1.0,
            zorder=5,
        )
    )
    text(ax, x, y + 0.008, "UPDATE", size=4.8, weight="bold")
    text(ax, x, y - 0.012, title, size=4.7)


def build_figure_b_v2():
    """Build the pictorial heterogeneous data-movement figure."""
    fig = plt.figure(figsize=(7.6, 4.8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    text(
        ax,
        0.5,
        0.975,
        "Heterogeneous Data Movement in Persistent RSNN Execution",
        size=10.5,
        weight="bold",
    )

    # Concrete global-memory arrays.
    box(ax, 0.025, 0.760, 0.950, 0.165, fc=LIGHT_GRAY, ec=GRAY, radius=0.012)
    text(ax, 0.043, 0.903, "GLOBAL MEMORY", size=6.4, weight="bold", ha="left")
    text(ax, 0.160, 0.883, "Static contiguous ownership", size=5.5, weight="bold")
    ownership_strip(ax, 0.055, 0.835)
    text(ax, 0.415, 0.883, "V", size=5.7, weight="bold")
    cells(
        ax,
        0.340,
        0.835,
        [""] * 10,
        cell_w=0.015,
        cell_h=0.025,
        colors=[GREEN_LIGHT] * 10,
    )
    text(ax, 0.560, 0.883, "PSC / input_current", size=5.7, weight="bold")
    cells(
        ax,
        0.500,
        0.835,
        [""] * 11,
        cell_w=0.015,
        cell_h=0.025,
        colors=[BLUE_LIGHT] * 11,
    )
    text(ax, 0.818, 0.883, "Global connectivity (CSR)", size=5.7, weight="bold")
    for index, name in enumerate(["indptr", "indices", "weights"]):
        yy = 0.858 - index * 0.031
        text(ax, 0.730, yy + 0.012, name, size=4.8, weight="bold", ha="right")
        cells(
            ax,
            0.740,
            yy,
            [""] * 12,
            cell_w=0.016,
            cell_h=0.023,
            colors=["white"] * 12,
            linewidth=0.45,
        )

    # Persistent CTA region, with UPDATE and PROP as distinct visual policies.
    box(ax, 0.025, 0.215, 0.950, 0.520, fc="#FBFBFC", ec=GRAY, radius=0.012)
    text(
        ax,
        0.043,
        0.727,
        "ON-CHIP PERSISTENT CTA",
        size=6.4,
        weight="bold",
        ha="left",
    )
    ax.plot(
        [0.495, 0.495],
        [0.235, 0.705],
        transform=ax.transAxes,
        color=GRAY,
        linewidth=0.8,
        linestyle="--",
    )
    box(ax, 0.060, 0.674, 0.390, 0.042, fc=GREEN_LIGHT, ec=GREEN)
    text(ax, 0.255, 0.695, "UPDATE — locality-oriented", size=6.8, weight="bold")
    box(ax, 0.535, 0.674, 0.390, 0.042, fc=PURPLE_LIGHT, ec=PURPLE)
    text(ax, 0.730, 0.695, "PROP — balance-oriented", size=6.8, weight="bold")

    # UPDATE: warp lanes, conditional V bank, and explicit temporal loop.
    warp_lanes(ax, 0.050, 0.390, token=False)
    box(ax, 0.050, 0.590, 0.150, 0.055, fc=GREEN_LIGHT, ec=GREEN)
    text(ax, 0.125, 0.627, "Static ownership", size=5.5, weight="bold", color=GREEN)
    text(ax, 0.125, 0.607, "B=1, neurons/CTA <= 256", size=4.3, weight="bold")
    box(ax, 0.222, 0.402, 0.240, 0.230, fc="#F6FBF6", ec=GREEN, radius=0.012)
    text(ax, 0.342, 0.610, "Conditional shared V", size=6.1, weight="bold", color=GREEN)
    text(ax, 0.342, 0.586, "load once before timestep loop", size=5.0, color=MUTED)
    cells(
        ax,
        0.248,
        0.535,
        ["n", "n+1", "n+2", "...", "n+k"],
        cell_w=0.037,
        cell_h=0.034,
        colors=[GREEN_LIGHT] * 5,
        fontsize=4.5,
    )
    stage_circle(ax, 0.273, 0.468, "t")
    stage_circle(ax, 0.342, 0.468, "t+1")
    stage_circle(ax, 0.411, 0.468, "t+2")
    arrow(ax, (0.305, 0.468), (0.310, 0.468), color=GREEN, lw=1.0)
    arrow(ax, (0.374, 0.468), (0.379, 0.468), color=GREEN, lw=1.0)
    arrow(
        ax,
        (0.430, 0.442),
        (0.258, 0.442),
        color=GREEN,
        lw=1.0,
        connectionstyle="arc3,rad=-0.35",
    )
    text(ax, 0.342, 0.417, "reuse the same shared V", size=4.9, color=GREEN)
    arrow(ax, (0.415, 0.834), (0.342, 0.635), color=GREEN, lw=2.0)

    # PSC/current remains a per-timestep global path.
    cells(
        ax,
        0.235,
        0.310,
        ["PSC"] + [""] * 4,
        cell_w=0.039,
        cell_h=0.034,
        colors=[BLUE_LIGHT] * 5,
        fontsize=4.3,
    )
    cells(
        ax,
        0.235,
        0.260,
        ["Iin"] + [""] * 4,
        cell_w=0.039,
        cell_h=0.034,
        colors=[BLUE_LIGHT] * 5,
        fontsize=4.3,
    )
    for index, x in enumerate((0.435, 0.451, 0.467)):
        arrow(
            ax,
            (0.515 + index * 0.016, 0.833),
            (x, 0.346),
            color=BLUE,
            lw=0.65,
            ls="--",
            connectionstyle="arc3,rad=0.05",
        )
    text(
        ax,
        0.342,
        0.236,
        "global load/store every timestep",
        size=4.9,
        color="#166B8E",
    )
    arrow(ax, (0.200, 0.485), (0.220, 0.485), color=GREEN, lw=1.4)

    # PROP: dynamic tokens and visible warp allocation.
    text(ax, 0.585, 0.626, "Dynamic task pool", size=5.8, weight="bold", color=ORANGE)
    cells(
        ax,
        0.525,
        0.585,
        ["T0", "T1", "T2", "T3", "T4"],
        cell_w=0.026,
        cell_h=0.031,
        colors=[ORANGE_LIGHT] * 5,
        fontsize=4.4,
    )
    warp_lanes(ax, 0.520, 0.365, token=True)
    text(ax, 0.595, 0.557, "Dynamic acquire", size=5.4, weight="bold")
    for index in range(4):
        arrow(
            ax,
            (0.538 + index * 0.026, 0.583),
            (0.555 + index * 0.020, 0.518 - index * 0.017),
            color=ORANGE,
            lw=0.8,
            connectionstyle="arc3,rad=0.12",
        )

    # Connectivity requests converge into thick grouped bundles.
    for index in range(3):
        arrow(
            ax,
            (0.800 + index * 0.034, 0.805),
            (0.735, 0.555),
            color=BLUE,
            lw=0.75,
            ls="--",
            connectionstyle="arc3,rad=0.10",
        )
    box(ax, 0.700, 0.462, 0.145, 0.093, fc=BLUE_LIGHT, ec=BLUE, radius=0.008)
    text(ax, 0.7725, 0.532, "Grouped edge access", size=5.6, weight="bold")
    text(ax, 0.7725, 0.506, "warp processor", size=5.0)
    text(ax, 0.7725, 0.481, "prefix -> logical edges", size=4.7, color=MUTED)
    for offset in (-0.018, 0.0, 0.018):
        arrow(
            ax,
            (0.670, 0.457 + offset),
            (0.696, 0.495 + offset / 2),
            color=BLUE,
            lw=2.2,
            scale=8,
        )

    # Repeated post IDs become visible contribution tokens and hash slots.
    text(ax, 0.888, 0.430, "Edge contributions (post IDs)", size=5.3, weight="bold")
    contribution_x = 0.695
    contributions = ["7", "3", "7", "9", "7", "3"]
    cells(
        ax,
        contribution_x,
        0.390,
        contributions,
        cell_w=0.042,
        cell_h=0.034,
        colors=[PURPLE_LIGHT] * 6,
        edgecolor=PURPLE,
    )
    arrow(ax, (0.846, 0.460), (0.846, 0.426), color=PURPLE, lw=1.2)
    box(ax, 0.695, 0.270, 0.255, 0.095, fc=GREEN_LIGHT, ec=GREEN, radius=0.010)
    text(
        ax,
        0.8225,
        0.345,
        "Per-warp shared hash",
        size=5.8,
        weight="bold",
        color=GREEN,
    )
    hash_slots = [("post 3", 1), ("post 7", 3), ("post 9", 1)]
    for index, (post, width) in enumerate(hash_slots):
        x = 0.710 + index * 0.080
        text(ax, x + 0.031, 0.321, post, size=4.6, weight="bold")
        cells(
            ax,
            x,
            0.286,
            [""] * 3,
            cell_w=0.021,
            cell_h=0.025,
            colors=[PURPLE_LIGHT] * width + ["white"] * (3 - width),
            edgecolor=GREEN,
            linewidth=0.4,
        )
    for index in range(6):
        arrow(
            ax,
            (contribution_x + (index + 0.5) * 0.042, 0.389),
            (0.735 + (index % 3) * 0.080, 0.365),
            color=PURPLE,
            lw=0.65,
            scale=6,
        )

    # Unique-post flush to a concrete global PSC array, plus probe fallback.
    text(ax, 0.822, 0.244, "Unique-post flush", size=5.0, color=PURPLE)
    cells(
        ax,
        0.665,
        0.165,
        ["", "", "", "3", "", "", "", "7", "", "9", "", ""],
        cell_w=0.023,
        cell_h=0.035,
        colors=["white"] * 3
        + [PURPLE_LIGHT]
        + ["white"] * 3
        + [PURPLE_LIGHT]
        + ["white"]
        + [PURPLE_LIGHT]
        + ["white"] * 2,
        edgecolor=INK,
        fontsize=4.7,
    )
    text(ax, 0.635, 0.1825, "Global PSC", size=5.5, weight="bold", ha="right")
    for x in (0.735, 0.815, 0.895):
        arrow(ax, (x, 0.268), (x, 0.203), color=PURPLE, lw=1.4)
    box(ax, 0.560, 0.255, 0.095, 0.055, fc="white", ec=RED, ls="--")
    text(ax, 0.6075, 0.2825, "probe fallback", size=4.7, color=RED)
    arrow(
        ax,
        (0.695, 0.300),
        (0.650, 0.280),
        color=RED,
        lw=0.8,
        ls="--",
    )
    arrow(
        ax,
        (0.607, 0.254),
        (0.688, 0.202),
        color=RED,
        lw=0.8,
        ls="--",
        connectionstyle="arc3,rad=0.12",
    )

    # Bottom policy statements reinforce the two different dimensions.
    text(
        ax,
        0.255,
        0.105,
        "Temporal reuse across timesteps",
        size=6.1,
        weight="bold",
        color=GREEN,
    )
    text(
        ax,
        0.750,
        0.105,
        "Request aggregation within a timestep",
        size=6.1,
        weight="bold",
        color=PURPLE,
    )
    return fig


def audit_and_export(fig, basename: str, size: tuple[float, float]) -> None:
    """Render, audit, then export vector and raster deliverables."""
    render_preview(fig, str(HERE / f"{basename}_preview.png"), dpi=180)
    print(f"\nLayout audit: {basename}")
    print_report(audit_layout(fig))
    export_figure(
        fig,
        str(HERE / basename),
        formats=("svg", "pdf", "png"),
        dpi=300,
        size_inches=size,
        grayscale_preview=True,
        tight=False,
    )


def main() -> None:
    """Generate both version-two method figures."""
    setup_style()
    figure_a = build_figure_a_v2()
    audit_and_export(figure_a, "figure_a_nested_allocation_v2", (7.6, 4.6))
    plt.close(figure_a)

    figure_b = build_figure_b_v2()
    audit_and_export(figure_b, "figure_b_pictorial_data_movement_v2", (7.6, 4.8))
    plt.close(figure_b)


if __name__ == "__main__":
    main()
