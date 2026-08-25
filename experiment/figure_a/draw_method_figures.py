"""Draw code-grounded method figures for persistent RSNN execution."""

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


INK = "#222222"
MUTED = "#5F6368"
PANEL = "#F3F4F6"
BLUE = "#56B4E9"
BLUE_LIGHT = "#DCEFFA"
ORANGE = "#E69F00"
ORANGE_LIGHT = "#FBE8BE"
GREEN = "#009E73"
GREEN_LIGHT = "#D9F0E8"
PURPLE = "#8B6BB8"
PURPLE_LIGHT = "#E8DFF2"
RED = "#D55E00"
RED_LIGHT = "#F6DDD2"
GRAY = "#B9BDC4"
GRAY_LIGHT = "#E7E8EA"


def configure_style() -> None:
    """Configure a compact publication style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = INK,
    linewidth: float = 1.0,
    radius: float = 0.012,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    """Add a rounded rectangle in axis coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.2,
    style: str = "-|>",
    linestyle: str = "-",
    mutation_scale: float = 9,
    connectionstyle: str = "arc3",
    zorder: int = 4,
) -> FancyArrowPatch:
    """Add an arrow in axis coordinates."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7,
    weight: str = "normal",
    color: str = INK,
    ha: str = "center",
    va: str = "center",
    zorder: int = 6,
    **kwargs,
):
    """Add consistently styled text in axis coordinates."""
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
        **kwargs,
    )


def draw_neuron(ax, x: float, y: float, active: bool, name: str) -> None:
    """Draw one active or inactive neuron."""
    circle = Circle(
        (x, y),
        0.010,
        transform=ax.transAxes,
        facecolor=BLUE if active else "white",
        edgecolor=INK,
        linewidth=0.9,
        zorder=5,
    )
    ax.add_patch(circle)
    label(ax, x - 0.019, y, name, size=5.9, ha="right")


def draw_edge_cells(
    ax,
    x: float,
    y: float,
    count: int,
    *,
    color: str,
    cell_width: float = 0.014,
    height: float = 0.021,
) -> None:
    """Draw a compact fanout span."""
    for index in range(count):
        ax.add_patch(
            Rectangle(
                (x + index * cell_width, y - height / 2),
                cell_width,
                height,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=INK,
                linewidth=0.55,
                zorder=4,
            )
        )


def draw_task(ax, x: float, y: float, width: float, text: str) -> None:
    """Draw a propagation task."""
    rounded_box(
        ax,
        x,
        y,
        width,
        0.041,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        radius=0.007,
    )
    label(ax, x + width / 2, y + 0.0205, text, size=6.1, weight="bold")


def build_figure_a():
    """Build Figure A: hierarchical propagation work organization."""
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    label(
        ax,
        0.5,
        0.975,
        "Hierarchical Block-Oriented Propagation",
        size=11,
        weight="bold",
    )
    label(
        ax,
        0.5,
        0.949,
        "Two contractions: task issuing, then global update issuing",
        size=6.7,
        color=MUTED,
    )

    # Level 1: one firing neuron creates one fine-grained task.
    rounded_box(ax, 0.035, 0.715, 0.93, 0.205, facecolor=PANEL, edgecolor=GRAY)
    label(ax, 0.053, 0.895, "a", size=9, weight="bold", ha="left")
    label(
        ax,
        0.078,
        0.895,
        "Neuron-level tasking",
        size=8.2,
        weight="bold",
        ha="left",
    )
    label(
        ax,
        0.947,
        0.895,
        "One active neuron -> one task",
        size=6.2,
        color=MUTED,
        ha="right",
    )
    neuron_y = [0.850, 0.811, 0.772, 0.733]
    neuron_names = ["N0", "N2", "N3", "N5"]
    spans = [5, 3, 6, 4]
    for row, (y, name, span) in enumerate(zip(neuron_y, neuron_names, spans)):
        draw_neuron(ax, 0.105, y, True, name)
        arrow(ax, (0.118, y), (0.148, y), linewidth=0.8, mutation_scale=6)
        draw_edge_cells(ax, 0.152, y, span, color=BLUE_LIGHT)
        task_x = 0.265 + row * 0.145
        arrow(
            ax,
            (0.152 + span * 0.014, y),
            (task_x, 0.838),
            color=RED,
            linewidth=0.9,
            linestyle="--",
            mutation_scale=7,
        )
        draw_task(ax, task_x, 0.818, 0.095, f"Neuron task {row}")
    label(
        ax,
        0.865,
        0.790,
        "many independent\nsmall work units",
        size=7.0,
        weight="bold",
        color=RED,
    )
    arrow(ax, (0.50, 0.710), (0.50, 0.683), color=RED, linewidth=2.0)
    label(
        ax,
        0.525,
        0.696,
        "task-side contraction",
        size=6.0,
        color=RED,
        ha="left",
    )

    # Level 2: warp-local 32-neuron group and long-row branch.
    rounded_box(ax, 0.035, 0.360, 0.93, 0.315, facecolor=PANEL, edgecolor=GRAY)
    label(ax, 0.053, 0.650, "b", size=9, weight="bold", ha="left")
    label(
        ax,
        0.078,
        0.650,
        "Block-level task formation",
        size=8.2,
        weight="bold",
        ha="left",
    )
    rounded_box(
        ax,
        0.065,
        0.525,
        0.555,
        0.095,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.4,
        linestyle="--",
    )
    label(
        ax,
        0.080,
        0.603,
        "Warp-local neuron group (32 lanes)",
        size=6.5,
        weight="bold",
        ha="left",
        color="#176B87",
    )
    active_pattern = [True, False, True, True, False, True, False, True]
    for index, active in enumerate(active_pattern):
        x = 0.112 + index * 0.061
        draw_neuron(ax, x, 0.563, active, f"N{index}" if index < 7 else "N31")
    label(
        ax,
        0.604,
        0.538,
        "logical only; CSR rows remain unchanged",
        size=5.6,
        color=MUTED,
        ha="right",
    )
    arrow(ax, (0.338, 0.520), (0.338, 0.488), color=RED)
    label(
        ax,
        0.085,
        0.476,
        "Active-row prefix / logical edge offsets",
        size=6.3,
        weight="bold",
        ha="left",
    )
    stream_x = 0.085
    stream_y = 0.435
    row_widths = [0.110, 0.066, 0.132, 0.088]
    row_labels = ["N0", "N2", "N3", "N5"]
    row_colors = [BLUE_LIGHT, GREEN_LIGHT, ORANGE_LIGHT, PURPLE_LIGHT]
    running = stream_x
    prefix = 0
    label(ax, stream_x, 0.421, "0", size=5.3, ha="center", color=MUTED)
    for width, text, color in zip(row_widths, row_labels, row_colors):
        ax.add_patch(
            Rectangle(
                (running, stream_y),
                width,
                0.032,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=INK,
                linewidth=0.8,
            )
        )
        label(ax, running + width / 2, stream_y + 0.016, text, size=6.1)
        running += width
        prefix += round(width * 100)
        label(ax, running, 0.421, str(prefix), size=5.3, color=MUTED)
    arrow(ax, (0.338, 0.414), (0.338, 0.392), color=RED)
    draw_task(ax, 0.094, 0.368, 0.132, "Logical task 0")
    draw_task(ax, 0.242, 0.368, 0.132, "Logical task 1")
    draw_task(ax, 0.390, 0.368, 0.132, "Logical task 2")
    label(
        ax,
        0.535,
        0.389,
        "bounded by kBlockTaskSpan",
        size=5.7,
        color=MUTED,
        ha="left",
    )

    rounded_box(
        ax,
        0.665,
        0.485,
        0.265,
        0.135,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.2,
    )
    label(
        ax,
        0.685,
        0.601,
        "High-fanout row (degree >= 256)",
        size=6.3,
        weight="bold",
        ha="left",
    )
    draw_neuron(ax, 0.700, 0.562, True, "Nlong")
    draw_edge_cells(
        ax,
        0.727,
        0.562,
        12,
        color=ORANGE_LIGHT,
        cell_width=0.014,
    )
    arrow(ax, (0.798, 0.545), (0.798, 0.522), color=RED)
    for index in range(3):
        rounded_box(
            ax,
            0.686 + index * 0.079,
            0.496,
            0.066,
            0.027,
            facecolor=ORANGE_LIGHT,
            edgecolor=ORANGE,
            radius=0.004,
        )
        label(ax, 0.719 + index * 0.079, 0.5095, f"Frag {index}", size=5.4)
    label(
        ax,
        0.798,
        0.472,
        "fixed-size segments (default 1024 edges)",
        size=5.6,
        color=MUTED,
    )
    arrow(ax, (0.470, 0.367), (0.615, 0.385), color=RED, linestyle="--")
    arrow(ax, (0.798, 0.484), (0.700, 0.414), color=RED, linestyle="--")
    rounded_box(
        ax,
        0.585,
        0.371,
        0.260,
        0.051,
        facecolor=RED_LIGHT,
        edgecolor=RED,
        linewidth=1.2,
    )
    label(
        ax,
        0.715,
        0.3965,
        "Dynamic PROP task pool",
        size=7.0,
        weight="bold",
    )
    label(
        ax,
        0.942,
        0.386,
        "ordinary tasks +\nlong-row fragments",
        size=5.8,
        color=MUTED,
        ha="right",
    )
    arrow(ax, (0.50, 0.354), (0.50, 0.327), color=RED, linewidth=2.0)
    label(
        ax,
        0.525,
        0.340,
        "update-side contraction",
        size=6.0,
        color=RED,
        ha="left",
    )

    # Level 3: repeated destinations merge in a shared hash.
    rounded_box(ax, 0.035, 0.050, 0.93, 0.270, facecolor=PANEL, edgecolor=GRAY)
    label(ax, 0.053, 0.295, "c", size=9, weight="bold", ha="left")
    label(
        ax,
        0.078,
        0.295,
        "Block-local accumulation",
        size=8.2,
        weight="bold",
        ha="left",
    )
    posts = [7, 3, 7, 9, 7, 3]
    for index, post in enumerate(posts):
        x = 0.075 + index * 0.073
        rounded_box(
            ax,
            x,
            0.215,
            0.058,
            0.043,
            facecolor=BLUE_LIGHT if post == 7 else ORANGE_LIGHT,
            edgecolor=INK,
            radius=0.005,
        )
        label(ax, x + 0.029, 0.2365, f"e{index}->p{post}", size=5.6)
        arrow(ax, (x + 0.029, 0.211), (0.335, 0.170), color=RED, linewidth=0.8)
    rounded_box(
        ax,
        0.247,
        0.090,
        0.255,
        0.091,
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        linewidth=1.3,
    )
    label(ax, 0.375, 0.161, "Per-warp shared hash", size=6.8, weight="bold")
    label(ax, 0.375, 0.138, "p3: w1 + w5     p7: w0 + w2 + w4", size=5.7)
    label(ax, 0.375, 0.113, "p9: w3", size=5.7)
    unique_posts = [(3, 0.610), (7, 0.710), (9, 0.810)]
    for index, (post, x) in enumerate(unique_posts):
        arrow(
            ax,
            (0.502, 0.116 + index * 0.019),
            (x, 0.135),
            color=PURPLE,
            linewidth=1.0,
            connectionstyle=f"arc3,rad={0.12 - index * 0.08}",
        )
        rounded_box(
            ax,
            x,
            0.109,
            0.075,
            0.052,
            facecolor=PURPLE_LIGHT,
            edgecolor=PURPLE,
            radius=0.005,
        )
        label(ax, x + 0.0375, 0.135, f"atomic p{post}", size=5.6, weight="bold")
    rounded_box(
        ax,
        0.585,
        0.205,
        0.305,
        0.052,
        facecolor=GRAY_LIGHT,
        edgecolor=INK,
    )
    label(ax, 0.738, 0.231, "Global PSC", size=7.0, weight="bold")
    for _, x in unique_posts:
        arrow(ax, (x + 0.0375, 0.163), (0.738, 0.202), color=PURPLE)
    label(
        ax,
        0.947,
        0.082,
        "conditional hash;\nprobe failures use\nglobal atomic",
        size=5.2,
        color=MUTED,
        ha="right",
    )
    label(
        ax,
        0.085,
        0.075,
        "many edge contributions",
        size=6.1,
        color=RED,
        ha="left",
    )
    label(
        ax,
        0.605,
        0.075,
        "fewer global atomic updates",
        size=6.1,
        color=PURPLE,
        ha="left",
    )
    return fig


def memory_bar(ax, x: float, y: float, width: float, text: str) -> None:
    """Draw a global-memory object."""
    rounded_box(
        ax,
        x,
        y,
        width,
        0.050,
        facecolor=GRAY_LIGHT,
        edgecolor=INK,
        radius=0.005,
    )
    label(ax, x + width / 2, y + 0.025, text, size=6.0, weight="bold")


def build_figure_b():
    """Build Figure B: heterogeneous data movement."""
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    label(
        ax,
        0.5,
        0.975,
        "Heterogeneous Data Movement in Persistent RSNN Execution",
        size=10.6,
        weight="bold",
    )

    # Horizontal memory/execution regions.
    ax.add_patch(
        Rectangle(
            (0.025, 0.710),
            0.950,
            0.185,
            transform=ax.transAxes,
            facecolor="#FAFAFA",
            edgecolor=GRAY,
            linewidth=0.8,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.025, 0.225),
            0.950,
            0.450,
            transform=ax.transAxes,
            facecolor=PANEL,
            edgecolor=GRAY,
            linewidth=0.8,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.025, 0.065),
            0.950,
            0.125,
            transform=ax.transAxes,
            facecolor="#FAFAFA",
            edgecolor=GRAY,
            linewidth=0.8,
        )
    )
    label(ax, 0.044, 0.875, "GLOBAL MEMORY", size=6.2, weight="bold", ha="left")
    label(
        ax,
        0.044,
        0.655,
        "ON-CHIP / PERSISTENT CTA",
        size=6.2,
        weight="bold",
        ha="left",
    )
    label(ax, 0.044, 0.170, "EXECUTION", size=6.2, weight="bold", ha="left")
    ax.plot(
        [0.5, 0.5],
        [0.078, 0.886],
        color=GRAY,
        linewidth=1.0,
        linestyle="--",
        transform=ax.transAxes,
        zorder=2,
    )

    # Column titles.
    rounded_box(
        ax,
        0.065,
        0.905,
        0.390,
        0.038,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        radius=0.006,
    )
    label(
        ax,
        0.260,
        0.924,
        "UPDATE — static ownership for temporal locality",
        size=7.1,
        weight="bold",
    )
    rounded_box(
        ax,
        0.545,
        0.905,
        0.390,
        0.038,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        radius=0.006,
    )
    label(
        ax,
        0.740,
        0.924,
        "PROP — dynamic scheduling for load balance",
        size=7.1,
        weight="bold",
    )

    # UPDATE global arrays and conditional voltage residency.
    memory_bar(ax, 0.068, 0.788, 0.090, "V")
    memory_bar(ax, 0.170, 0.788, 0.105, "PSC")
    memory_bar(ax, 0.287, 0.788, 0.130, "input current")
    label(
        ax,
        0.242,
        0.755,
        "per-neuron arrays",
        size=5.5,
        color=MUTED,
    )
    rounded_box(
        ax,
        0.075,
        0.540,
        0.365,
        0.090,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=1.2,
    )
    label(
        ax,
        0.258,
        0.608,
        "Fixed contiguous neuron ownership",
        size=6.8,
        weight="bold",
    )
    label(
        ax,
        0.258,
        0.579,
        "CTA 0: [0, N0)   CTA 1: [N0, N1)   CTA 2: [N1, N2)",
        size=5.7,
    )
    label(
        ax,
        0.258,
        0.554,
        "ownership is stable across all timesteps",
        size=5.6,
        color=MUTED,
    )
    arrow(ax, (0.214, 0.786), (0.214, 0.633), color=BLUE, linewidth=1.1)
    arrow(ax, (0.352, 0.786), (0.352, 0.633), color=BLUE, linewidth=1.1)

    rounded_box(
        ax,
        0.076,
        0.353,
        0.225,
        0.135,
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        linewidth=1.3,
    )
    label(ax, 0.188, 0.465, "Conditional shared V", size=6.8, weight="bold")
    label(ax, 0.188, 0.435, "load before timestep loop", size=5.7)
    label(ax, 0.188, 0.408, "reuse V at t, t+1, t+2, ...", size=5.7)
    label(ax, 0.188, 0.378, "write back after loop", size=5.7)
    arrow(ax, (0.113, 0.786), (0.113, 0.490), color=GREEN, linewidth=2.0)
    arrow(
        ax,
        (0.109, 0.354),
        (0.107, 0.786),
        color=GREEN,
        linewidth=1.2,
        linestyle="--",
        connectionstyle="arc3,rad=-0.18",
    )
    arrow(
        ax,
        (0.277, 0.402),
        (0.277, 0.455),
        color=GREEN,
        linewidth=1.2,
        connectionstyle="arc3,rad=0.55",
    )
    rounded_box(
        ax,
        0.315,
        0.353,
        0.126,
        0.135,
        facecolor="white",
        edgecolor=GRAY,
        linestyle="--",
    )
    label(ax, 0.378, 0.463, "Residency guard", size=6.1, weight="bold")
    label(ax, 0.378, 0.428, "batch_size == 1", size=5.4)
    label(ax, 0.378, 0.401, "neurons / CTA <= 256", size=5.4)
    label(ax, 0.378, 0.371, "else V remains global", size=5.4, color=RED)
    arrow(ax, (0.158, 0.788), (0.188, 0.489), color=GREEN, linewidth=1.2)

    rounded_box(
        ax,
        0.088,
        0.245,
        0.335,
        0.065,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
    )
    label(ax, 0.255, 0.289, "Neuron UPDATE", size=6.8, weight="bold")
    label(
        ax,
        0.255,
        0.263,
        "PSC/current still use global load/store each timestep",
        size=5.3,
    )
    arrow(ax, (0.258, 0.538), (0.258, 0.312), color=BLUE, linewidth=1.4)
    arrow(ax, (0.188, 0.350), (0.188, 0.312), color=GREEN, linewidth=1.4)

    # UPDATE to PROP interface.
    rounded_box(
        ax,
        0.326,
        0.105,
        0.348,
        0.052,
        facecolor=RED_LIGHT,
        edgecolor=RED,
        linewidth=1.2,
    )
    label(
        ax,
        0.500,
        0.131,
        "spike mask + block descriptor / long-row edge range",
        size=5.9,
        weight="bold",
    )
    arrow(ax, (0.255, 0.244), (0.392, 0.159), color=RED, linewidth=1.6)

    # PROP global inputs.
    memory_bar(ax, 0.535, 0.788, 0.118, "task queues")
    memory_bar(ax, 0.664, 0.788, 0.094, "indptr")
    memory_bar(ax, 0.769, 0.788, 0.092, "indices")
    memory_bar(ax, 0.872, 0.788, 0.082, "weights")
    label(
        ax,
        0.811,
        0.755,
        "connectivity remains in global memory",
        size=5.5,
        color=MUTED,
    )
    rounded_box(
        ax,
        0.552,
        0.557,
        0.180,
        0.070,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=1.2,
    )
    label(ax, 0.642, 0.605, "Dynamic acquire", size=6.5, weight="bold")
    label(ax, 0.642, 0.578, "warp pulls next task", size=5.5)
    arrow(ax, (0.594, 0.786), (0.617, 0.630), color=ORANGE, linewidth=1.5)
    arrow(
        ax,
        (0.500, 0.158),
        (0.520, 0.540),
        color=RED,
        linewidth=1.0,
        style="-",
        linestyle="--",
        connectionstyle="angle3,angleA=90,angleB=0",
    )
    arrow(
        ax,
        (0.520, 0.540),
        (0.552, 0.592),
        color=RED,
        linewidth=1.2,
        linestyle="--",
    )

    rounded_box(
        ax,
        0.747,
        0.548,
        0.183,
        0.090,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=1.2,
    )
    label(ax, 0.838, 0.618, "Grouped edge access", size=6.5, weight="bold")
    label(ax, 0.838, 0.591, "active-row prefix", size=5.5)
    label(ax, 0.838, 0.566, "bounded logical segment", size=5.5)
    arrow(ax, (0.733, 0.592), (0.744, 0.592), color=ORANGE, linewidth=2.6)
    for x in (0.711, 0.815, 0.918):
        arrow(
            ax,
            (x, 0.786),
            (0.838, 0.641),
            color=BLUE,
            linewidth=0.9,
            linestyle="--",
        )
    label(
        ax,
        0.840,
        0.522,
        "fewer fine-grained request issuances",
        size=5.5,
        color="#176B87",
    )

    rounded_box(
        ax,
        0.610,
        0.415,
        0.280,
        0.074,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        linewidth=1.2,
    )
    label(ax, 0.750, 0.468, "Warp edge processing", size=6.6, weight="bold")
    label(
        ax,
        0.750,
        0.440,
        "map logical edges -> global indices / weights",
        size=5.4,
    )
    arrow(ax, (0.838, 0.546), (0.786, 0.491), color=BLUE, linewidth=2.0)
    for x in (0.660, 0.705, 0.750, 0.795, 0.840):
        arrow(ax, (x, 0.414), (x, 0.363), color=RED, linewidth=0.8)

    rounded_box(
        ax,
        0.620,
        0.270,
        0.260,
        0.090,
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        linewidth=1.3,
    )
    label(ax, 0.750, 0.340, "Per-warp shared hash", size=6.8, weight="bold")
    label(ax, 0.750, 0.311, "atomicCAS keys + local atomicAdd", size=5.5)
    label(
        ax,
        0.750,
        0.285,
        "used only for eligible windows; fallback is global",
        size=5.2,
        color=MUTED,
    )
    for x in (0.683, 0.750, 0.817):
        arrow(ax, (x, 0.268), (x, 0.205), color=PURPLE, linewidth=1.2)
    rounded_box(
        ax,
        0.632,
        0.195,
        0.236,
        0.037,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        radius=0.005,
    )
    label(ax, 0.750, 0.2135, "unique posts -> global atomic flush", size=5.7)
    memory_bar(ax, 0.690, 0.073, 0.120, "Global PSC")
    arrow(ax, (0.750, 0.194), (0.750, 0.125), color=PURPLE, linewidth=2.0)

    # Execution policy summaries.
    label(
        ax,
        0.255,
        0.085,
        "vertical reuse across timesteps",
        size=6.0,
        weight="bold",
        color=GREEN,
    )
    label(
        ax,
        0.895,
        0.145,
        "within-timestep\ncontraction",
        size=6.0,
        weight="bold",
        color=RED,
    )
    return fig


def audit_and_export(fig, basename: str, size: tuple[float, float]) -> None:
    """Run the SciPilot visual QA layer and export publication files."""
    preview = HERE / f"{basename}_preview.png"
    render_preview(fig, str(preview), dpi=180)
    print(f"\nLayout audit: {basename}")
    print_report(audit_layout(fig))
    export_figure(
        fig,
        str(HERE / basename),
        formats=("pdf", "svg", "png"),
        dpi=300,
        size_inches=size,
        grayscale_preview=True,
        tight=False,
    )


def main() -> None:
    """Generate both method figures."""
    configure_style()
    figure_a = build_figure_a()
    audit_and_export(
        figure_a,
        "hierarchical_block_oriented_propagation",
        (7.2, 5.4),
    )
    plt.close(figure_a)

    figure_b = build_figure_b()
    audit_and_export(
        figure_b,
        "heterogeneous_persistent_data_movement",
        (7.2, 4.8),
    )
    plt.close(figure_b)


if __name__ == "__main__":
    main()
