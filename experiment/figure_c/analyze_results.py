"""Aggregate the Figure C timing, software-counter, and NCU results."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


@dataclass(frozen=True)
class Variant:
    """Describe one fixed Figure C ablation variant."""

    key: str
    label: str
    block: bool
    similarity: bool
    hash: bool

    @property
    def timing_path(self) -> Path:
        return RAW_DIR / f"{self.key}_timing.csv"

    @property
    def stats_path(self) -> Path:
        return RAW_DIR / f"{self.key}_stats.csv"

    @property
    def ncu_path(self) -> Path:
        return RAW_DIR / f"{self.key}.ncu-rep"


VARIANTS = (
    Variant("v0_binning", "Binning", False, False, False),
    Variant("v1_block", "+ Block", True, False, False),
    Variant("v2_block_sort", "+ Sort", True, True, False),
    Variant("v3_block_sort_hash", "+ Hash", True, True, True),
)

NCU_METRICS = {
    "global_load_inst": "smsp__sass_inst_executed_op_global_ld.sum",
    "global_store_inst": "smsp__sass_inst_executed_op_global_st.sum",
    "global_reduction_inst": "smsp__inst_executed_op_global_red.sum",
    "atomic_tx": "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l2_sectors": "lts__t_sectors_srcunit_tex.sum",
    "dram_read_sectors": "dram__sectors_read.sum",
    "dram_write_sectors": "dram__sectors_write.sum",
    "long_scoreboard": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "issue_busy": "smsp__issue_active.avg.pct_of_peak_sustained_active",
}

WORKLOAD_FIELDS = (
    "dataset",
    "timestep_count",
    "n_neuron",
    "batch_size",
    "graph_synapses",
    "input_event_rate",
    "active_neurons",
    "active_synapses",
)


def read_single_row(path: Path) -> dict[str, str]:
    """Read a CSV that contains exactly one experimental result row."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


def read_ncu_report(path: Path) -> dict[str, str]:
    """Export one Nsight Compute report and return its raw metric row."""

    result = subprocess.run(
        ["ncu", "--import", str(path), "--csv", "--page", "raw"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) < 3:
        raise ValueError(f"Nsight Compute returned no raw result for {path}")
    if len(rows) > 3:
        raise ValueError(
            f"expected one profiled kernel in {path}, found {len(rows) - 2}"
        )
    return dict(zip(rows[0], rows[2], strict=True))


def value(row: dict[str, str], field: str) -> float:
    """Return a required numeric CSV field."""

    raw = row.get(field, "")
    if raw == "":
        raise ValueError(f"missing required numeric field: {field}")
    return float(raw)


def assert_common_workload(timing_rows: list[dict[str, str]]) -> None:
    """Reject accidental normalization across different workloads."""

    baseline = timing_rows[0]
    for row in timing_rows[1:]:
        mismatches = [
            field for field in WORKLOAD_FIELDS if row.get(field) != baseline.get(field)
        ]
        if mismatches:
            raise ValueError(
                "variant workload differs from V0 in: " + ", ".join(mismatches)
            )


def make_hardware_row(
    variant: Variant,
    timing: dict[str, str],
    ncu: dict[str, str],
) -> dict[str, str | float | int]:
    """Combine timing and hardware counters for one variant."""

    metrics = {name: value(ncu, key) for name, key in NCU_METRICS.items()}
    global_mem_inst = (
        metrics["global_load_inst"]
        + metrics["global_store_inst"]
        + metrics["global_reduction_inst"]
    )
    return {
        "variant": variant.key,
        "label": variant.label,
        "block": int(variant.block),
        "similarity": int(variant.similarity),
        "hash": int(variant.hash),
        "prop_time_ms": value(timing, "total_time_ms"),
        "step_time_ms": value(timing, "end_to_end_time_ms"),
        "global_mem_inst": global_mem_inst,
        "atomic_tx": metrics["atomic_tx"],
        "l2_sectors": metrics["l2_sectors"],
        "dram_sectors": (metrics["dram_read_sectors"] + metrics["dram_write_sectors"]),
        "long_scoreboard": metrics["long_scoreboard"],
        "issue_busy_pct": metrics["issue_busy"],
        "gpu": ncu["Device"],
        "compute_capability": ncu["CC"],
        "kernel": ncu["Kernel Name"],
    }


def normalize_rows(
    rows: list[dict[str, str | float | int]],
) -> list[dict[str, str | float | int]]:
    """Normalize quantitative metrics to V0 Binning."""

    metric_names = (
        "prop_time_ms",
        "step_time_ms",
        "global_mem_inst",
        "atomic_tx",
        "l2_sectors",
        "dram_sectors",
        "long_scoreboard",
        "issue_busy_pct",
    )
    baseline = rows[0]
    normalized = []
    for row in rows:
        output = dict(row)
        for metric in metric_names:
            output[f"{metric}_norm"] = float(row[metric]) / float(baseline[metric])
        normalized.append(output)
    return normalized


def make_software_row(
    variant: Variant,
    reference_edges: int,
) -> dict[str, str | float | int]:
    """Build the requested software-counter record."""

    if not variant.block:
        return {
            "variant": variant.key,
            "label": variant.label,
            "edges_processed": reference_edges,
            "tasks_generated": "",
            "fragments_generated": "",
            "global_updates_emitted": reference_edges,
            "hash_entries_flushed": 0,
            "hash_fallback_updates": 0,
            "atomic_per_edge": 1.0,
            "merge_ratio": 0.0,
        }

    stats = read_single_row(variant.stats_path)
    edges = int(value(stats, "active_synapses"))
    if edges != reference_edges:
        raise ValueError(
            f"software-counter workload differs for {variant.key}: "
            f"{edges} != {reference_edges} edges"
        )
    ordinary_edges = int(value(stats, "v4_input_edges"))
    long_edges = edges - ordinary_edges
    ordinary_updates = int(value(stats, "v4_global_atomics"))
    global_updates = long_edges + ordinary_updates
    atomic_per_edge = global_updates / edges
    return {
        "variant": variant.key,
        "label": variant.label,
        "edges_processed": edges,
        "tasks_generated": int(value(stats, "block_task_count"))
        + int(value(stats, "long_segment_tasks")),
        "fragments_generated": int(value(stats, "logical_block_task_count"))
        + int(value(stats, "long_segment_tasks")),
        "global_updates_emitted": global_updates,
        "hash_entries_flushed": int(value(stats, "kernel_hash_flush_atomics")),
        "hash_fallback_updates": int(value(stats, "kernel_hash_fallback_atomics")),
        "atomic_per_edge": atomic_per_edge,
        "merge_ratio": 1.0 - atomic_per_edge,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write records with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "results")
    return parser.parse_args()


def main() -> None:
    """Create the absolute, normalized, and software-counter CSV files."""

    args = parse_args()
    timing_rows = [read_single_row(variant.timing_path) for variant in VARIANTS]
    assert_common_workload(timing_rows)
    hardware_rows = [
        make_hardware_row(variant, timing, read_ncu_report(variant.ncu_path))
        for variant, timing in zip(VARIANTS, timing_rows, strict=True)
    ]
    normalized_rows = normalize_rows(hardware_rows)
    reference_edges = int(
        value(read_single_row(VARIANTS[1].stats_path), "active_synapses")
    )
    software_rows = [
        make_software_row(variant, reference_edges) for variant in VARIANTS
    ]
    write_csv(args.output_dir / "figure_c_metrics.csv", hardware_rows)
    write_csv(args.output_dir / "figure_c_normalized.csv", normalized_rows)
    write_csv(args.output_dir / "figure_c_software_counters.csv", software_rows)

    print(f"Wrote Figure C data to {args.output_dir}")


if __name__ == "__main__":
    main()
