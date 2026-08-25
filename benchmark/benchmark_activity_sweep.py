"""Run the paper activity sweep as isolated, resumable benchmark cases.

Each target-activity and workload-seed pair runs in a fresh Python process.
This prevents CUDA Graph and persistent-runner state from leaking between
workloads while reusing :mod:`benchmark_rsnn_cudagraph_compare` unchanged for
provider preparation, correctness checks, and timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("flybrain", "microns_mm3", "multiarea_mam", "uniform")
DEFAULT_TARGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
DEFAULT_PROVIDERS = (
    "cusparse_direct_cudagraph",
    "tilespmspv_cudagraph",
    "globalatomic_cudagraph",
    "vdha_cudagraph",
    "persistent_spike_block",
)
# FlyWire's mean absolute signed fanout is 393.056. This multiplier retains
# every relative edge weight while setting aggregate recurrent strength to
# approximately 0.15 for the common threshold-1 RSNN dynamics.
DEFAULT_FLYBRAIN_WEIGHT_SCALE = 0.15 / 393.0562438964844


def parse_args() -> argparse.Namespace:
    """Parse activity-sweep command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="flybrain")
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--t-steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--weight-scale", type=float, default=None)
    parser.add_argument("--n-neuron", type=int, default=131_072)
    parser.add_argument("--fanout", type=int, default=128)
    parser.add_argument("--multiarea-n-scaling", type=float, default=0.005)
    parser.add_argument("--multiarea-k-scaling", type=float, default=1.0)
    parser.add_argument("--calibration-max-amplitude", type=float, default=100.0)
    parser.add_argument("--calibration-iterations", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any(not 0.0 < value < 1.0 for value in args.targets):
        parser.error("all targets must be in (0, 1)")
    if args.t_steps <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("invalid timing configuration")
    if args.n_neuron <= 0 or args.fanout <= 0:
        parser.error("uniform dimensions must be positive")
    if args.calibration_max_amplitude <= 0.0:
        parser.error("--calibration-max-amplitude must be positive")
    if args.calibration_iterations <= 0:
        parser.error("--calibration-iterations must be positive")
    if args.weight_scale is None:
        args.weight_scale = (
            DEFAULT_FLYBRAIN_WEIGHT_SCALE
            if args.dataset == "flybrain"
            else 0.15
        )
    if args.output is None:
        args.output = Path(
            "benchmark/results/activity_sweep/"
            f"{args.dataset}_rtx5090_raw.csv"
        )
    return args


def valid_part(path: Path) -> bool:
    """Return whether a part CSV contains every requested provider."""

    if not path.exists():
        return False
    with path.open(newline="") as file:
        providers = {row["provider"] for row in csv.DictReader(file)}
    return providers == set(DEFAULT_PROVIDERS)


def run_case(args: argparse.Namespace, target: float, seed: int, part: Path) -> None:
    """Run one isolated target and seed combination."""

    command = [
        sys.executable,
        "benchmark/benchmark_rsnn_cudagraph_compare.py",
        "--dataset",
        args.dataset,
        "--t-steps",
        str(args.t_steps),
        "--target-activity",
        str(target),
        "--input-seed",
        str(seed),
        "--weight-scale",
        repr(args.weight_scale),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--calibration-max-amplitude",
        str(args.calibration_max_amplitude),
        "--calibration-iterations",
        str(args.calibration_iterations),
        "--providers",
        *DEFAULT_PROVIDERS,
        "--csv",
        str(part),
    ]
    if args.dataset == "uniform":
        command.extend(
            [
                "--n-neuron",
                str(args.n_neuron),
                "--fanout",
                str(args.fanout),
            ]
        )
    elif args.dataset == "multiarea_mam":
        command.extend(
            [
                "--multiarea-n-scaling",
                str(args.multiarea_n_scaling),
                "--multiarea-k-scaling",
                str(args.multiarea_k_scaling),
            ]
        )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def merge_parts(parts: list[Path], output: Path) -> None:
    """Merge homogeneous case CSVs into the final raw-data table."""

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for part in parts:
        with part.open(newline="") as file:
            reader = csv.DictReader(file)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or ())
            elif list(reader.fieldnames or ()) != fieldnames:
                raise RuntimeError(f"CSV schema mismatch in {part}")
            rows.extend(reader)
    if fieldnames is None:
        raise RuntimeError("activity sweep produced no rows")
    baseline_by_workload = {
        (row["input_seed"], row["requested_activity"]): float(row["latency_ms"])
        for row in rows
        if row["provider"] == "cusparse_direct_cudagraph"
    }
    derived_field = "activity_sweep_speedup_vs_cusparse"
    fieldnames.append(derived_field)
    for row in rows:
        baseline = baseline_by_workload[
            (row["input_seed"], row["requested_activity"])
        ]
        latency = float(row["latency_ms"])
        row[derived_field] = baseline / latency
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run missing cases and merge their raw measurements."""

    args = parse_args()
    output = (REPO_ROOT / args.output).resolve()
    part_dir = output.parent / f"{output.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for seed in args.seeds:
        for target in args.targets:
            target_label = f"{target:.6f}".rstrip("0").rstrip(".")
            part = part_dir / f"target_{target_label}_seed_{seed}.csv"
            parts.append(part)
            if args.force or not valid_part(part):
                run_case(args, target, seed, part)
            else:
                print(f"[resume] {part}")
    merge_parts(parts, output)
    manifest = {
        "dataset": args.dataset,
        "targets": list(args.targets),
        "seeds": list(args.seeds),
        "providers": list(DEFAULT_PROVIDERS),
        "t_steps": args.t_steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "calibration_max_amplitude": args.calibration_max_amplitude,
        "calibration_iterations": args.calibration_iterations,
        "weight_scale": args.weight_scale,
        "weight_normalization": "mean_abs_weighted_fanout~=0.15",
        "n_neuron": args.n_neuron if args.dataset == "uniform" else None,
        "fanout": args.fanout if args.dataset == "uniform" else None,
        "multiarea_n_scaling": (
            args.multiarea_n_scaling
            if args.dataset == "multiarea_mam"
            else None
        ),
        "multiarea_k_scaling": (
            args.multiarea_k_scaling
            if args.dataset == "multiarea_mam"
            else None
        ),
        "output": str(output.relative_to(REPO_ROOT)),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(parts) * len(DEFAULT_PROVIDERS)} rows to {output}")


if __name__ == "__main__":
    main()
