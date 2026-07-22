"""Profile the three persistent kernels across cooperative grid sizes.

Run this script inside the project environment on the target GPU::

    micromamba run -n ml-py312 python \
        benchmark/profile_persistent_grid_sweep.py

The script records CUDA-event median latency separately from Nsight Compute
metrics. ``tail_duration_ms`` is the spread between the maximum and minimum
active SM cycle counts, converted with the measured SM cycle rate. It measures
the final inter-SM work tail rather than the kernel's full duration.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOFLINE_SCRIPT = REPO_ROOT / "benchmark" / "benchmark_rsnn_roofline.py"
DEFAULT_BLOCKS = (170, 340, 510, 680, 850)
VARIANTS = {
    "plain": (),
    "binning": ("--fanout-binning",),
    "spike_block": ("--spike-block",),
}
NCU_METRICS = (
    "gpu__time_duration.sum",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "sm__cycles_active.max",
    "sm__cycles_active.min",
    "sm__cycles_elapsed.avg.per_second",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=list(DEFAULT_BLOCKS),
    )
    parser.add_argument(
        "--dataset",
        choices=("uniform", "mice_column_v1"),
        default="mice_column_v1",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/grid_sweep"))
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the PyTorch correctness check for every sweep point.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Nsight Compute reports.",
    )
    args = parser.parse_args()
    if any(blocks <= 0 for blocks in args.blocks):
        parser.error("--blocks values must be positive.")
    if args.repeat < 20:
        parser.error("--repeat must be at least 20.")
    return args


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command from the repository root and retain its output."""

    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
        )
    return result


def read_benchmark_latency(path: Path) -> float:
    """Read the CUDA-event median latency from one roofline CSV."""

    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    return float(row["total_time_ms"])


def parse_ncu_raw(text: str) -> tuple[dict[str, float], dict[str, str]]:
    """Extract the persistent-kernel row and metric units from NCU CSV."""

    lines = text.splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith('"ID"')
    )
    rows = list(csv.DictReader(lines[header_index:]))
    if len(rows) < 2:
        raise RuntimeError("Nsight Compute raw output did not contain metric rows.")
    units = rows[0]
    data = next(
        row
        for row in rows[1:]
        if "persistent_snn" in row.get("Kernel Name", "")
    )
    missing = [metric for metric in NCU_METRICS if not data.get(metric)]
    if missing:
        raise RuntimeError(f"Nsight Compute did not return metrics: {missing}")
    values = {metric: float(data[metric]) for metric in NCU_METRICS}
    return values, units


def cycles_to_ms(cycles: float, rate: float, unit: str) -> float:
    """Convert cycles using an Nsight Compute cycle-rate metric."""

    scale = {
        "Ghz": 1e-6,
        "cycle/nsecond": 1e-6,
        "cycle/usecond": 1e-3,
        "cycle/msecond": 1.0,
        "cycle/second": 1e3,
    }
    try:
        return cycles / rate * scale[unit]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported SM cycle-rate unit: {unit!r}") from exc


def duration_to_ms(duration: float, unit: str) -> float:
    """Convert an Nsight Compute duration metric to milliseconds."""

    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    try:
        return duration * scale[unit]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported duration unit: {unit!r}") from exc


def common_roofline_args(args: argparse.Namespace, blocks: int) -> list[str]:
    """Build arguments shared by benchmark and profiling runs."""

    result = [
        str(ROOFLINE_SCRIPT),
        "--dataset",
        args.dataset,
        "--grid-blocks",
        str(blocks),
        "--warmup",
        str(args.warmup),
    ]
    if args.connectome_root is not None:
        result.extend(("--connectome-root", str(args.connectome_root)))
    return result


def profile_point(
    args: argparse.Namespace,
    *,
    variant: str,
    variant_args: tuple[str, ...],
    blocks: int,
    sm_count: int,
) -> dict[str, float | int | str]:
    """Benchmark and profile one variant/grid-size pair."""

    stem = f"{variant}_{blocks}_blocks"
    latency_csv = args.output_dir / f"{stem}_latency.csv"
    report_stem = args.output_dir / stem
    report_path = report_stem.with_suffix(".ncu-rep")
    benchmark_command = [
        sys.executable,
        *common_roofline_args(args, blocks),
        *variant_args,
        "--repeat",
        str(args.repeat),
        "--csv",
        str(latency_csv),
    ]
    if not args.validate:
        benchmark_command.append("--skip-correctness")
    run_checked(benchmark_command)
    latency_ms = read_benchmark_latency(latency_csv)

    if args.force or not report_path.exists():
        profile_command = [
            args.ncu,
            "--metrics",
            ",".join(NCU_METRICS),
            "--profile-from-start",
            "off",
            "--kernel-name",
            "regex:persistent_snn.*kernel",
            "--launch-count",
            "1",
            "--force-overwrite",
            "-o",
            str(report_stem),
            sys.executable,
            *common_roofline_args(args, blocks),
            *variant_args,
            "--mode",
            "ncu",
            "--skip-correctness",
        ]
        run_checked(profile_command)

    raw = run_checked(
        [
            args.ncu,
            "--import",
            str(report_path),
            "--page",
            "raw",
            "--csv",
            "--metrics",
            ",".join(NCU_METRICS),
        ]
    )
    metrics, units = parse_ncu_raw(raw.stdout)
    barrier = metrics[
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ]
    warp_latency = metrics["smsp__average_warp_latency_per_inst_issued.ratio"]
    cycle_spread = (
        metrics["sm__cycles_active.max"] - metrics["sm__cycles_active.min"]
    )
    rate_name = "sm__cycles_elapsed.avg.per_second"
    tail_duration_ms = cycles_to_ms(
        cycle_spread,
        metrics[rate_name],
        units[rate_name],
    )
    return {
        "variant": variant,
        "grid_blocks": blocks,
        "sm_count": sm_count,
        "blocks_per_sm": blocks / sm_count,
        "latency_ms": latency_ms,
        "ncu_duration_ms": duration_to_ms(
            metrics["gpu__time_duration.sum"],
            units["gpu__time_duration.sum"],
        ),
        "eligible_warps_per_scheduler": metrics[
            "smsp__warps_eligible.avg.per_cycle_active"
        ],
        "active_warps_per_scheduler": metrics[
            "smsp__warps_active.avg.per_cycle_active"
        ],
        "barrier_stall_cycles": barrier,
        "barrier_stall_pct": 100.0 * barrier / warp_latency,
        "tail_duration_ms": tail_duration_ms,
        "report": str(report_path),
    }


def write_results(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write accumulated sweep rows so interrupted runs retain results."""

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the complete persistent grid sweep."""

    args = parse_args()
    if shutil.which(args.ncu) is None:
        raise SystemExit(f"Nsight Compute executable not found: {args.ncu}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the persistent grid sweep.")
    properties = torch.cuda.get_device_properties(0)
    sm_count = properties.multi_processor_count
    print(f"GPU: {properties.name}; SMs: {sm_count}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "persistent_grid_sweep.csv"
    rows: list[dict[str, float | int | str]] = []
    for variant, variant_args in VARIANTS.items():
        for blocks in args.blocks:
            rows.append(
                profile_point(
                    args,
                    variant=variant,
                    variant_args=variant_args,
                    blocks=blocks,
                    sm_count=sm_count,
                )
            )
            write_results(summary_path, rows)
            print(f"Updated {summary_path}", flush=True)


if __name__ == "__main__":
    main()
