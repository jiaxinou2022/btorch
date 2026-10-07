"""Run the paper firing-rate sweep as isolated, resumable benchmark cases.

Each target firing rate and workload-seed pair runs in a fresh Python process.
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
DATASETS = (
    "flybrain",
    "microns_mm3",
    "multiarea_mam",
    "uniform",
    "orkut",
    "hollywood_2009",
    "vas_stokes_4m",
    "uk_2002",
    "queen_4147",
)
DEFAULT_TARGET_RATES_HZ = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
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


def firing_rate_hz_to_activity(rate_hz: float, dt_ms: float) -> float:
    """Convert average firing rate in Hz to spikes per neuron per timestep."""

    if rate_hz < 0.0:
        raise ValueError("firing rate must be non-negative")
    if dt_ms <= 0.0:
        raise ValueError("timestep must be positive")
    return rate_hz * dt_ms / 1000.0


def external_event_rate_for_target(
    target_activity: float,
    max_event_rate: float,
) -> float:
    """Choose a sparse external trace that resolves low target rates."""

    if not 0.0 <= target_activity <= 1.0:
        raise ValueError("target activity must be in [0, 1]")
    if not 0.0 < max_event_rate <= 1.0:
        raise ValueError("maximum external event rate must be in (0, 1]")
    return min(max_event_rate, target_activity)


def parse_args() -> argparse.Namespace:
    """Parse activity-sweep command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="flybrain")
    parser.add_argument(
        "--target-rates-hz",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_RATES_HZ,
        help="Average neuronal firing rates in Hz (default: 0--50 Hz).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--t-steps", type=int, default=256)
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=1.0,
        help="Simulation timestep in milliseconds (default: 1.0).",
    )
    parser.add_argument(
        "--max-external-event-rate",
        type=float,
        default=0.01,
        help=(
            "Maximum external event probability per timestep; lower target "
            "rates use a proportionally sparser input trace."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--connectome-root", type=Path, default=None)
    parser.add_argument(
        "--isolate-providers",
        action="store_true",
        help="Run every provider in a fresh process to cap peak GPU memory.",
    )
    parser.add_argument("--weight-scale", type=float, default=None)
    parser.add_argument("--n-neuron", type=int, default=131_072)
    parser.add_argument("--fanout", type=int, default=128)
    parser.add_argument("--multiarea-n-scaling", type=float, default=0.005)
    parser.add_argument("--multiarea-k-scaling", type=float, default=1.0)
    parser.add_argument("--calibration-max-amplitude", type=float, default=100.0)
    parser.add_argument("--calibration-iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any(not 0.0 <= value <= 50.0 for value in args.target_rates_hz):
        parser.error("all target firing rates must be in [0, 50] Hz")
    if args.dt_ms <= 0.0:
        parser.error("--dt-ms must be positive")
    if not 0.0 < args.max_external_event_rate <= 1.0:
        parser.error("--max-external-event-rate must be in (0, 1]")
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
            f"{args.dataset}_rtx5090_hz_raw.csv"
        )
    return args


def valid_part(path: Path) -> bool:
    """Return whether a part CSV contains every passing provider."""

    if not path.exists():
        return False
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    providers = {row["provider"] for row in rows}
    return providers == set(DEFAULT_PROVIDERS) and all(
        row["correctness_status"].startswith("passed") for row in rows
    )


def valid_provider_part(path: Path, provider: str) -> bool:
    """Return whether an isolated part contains its one requested provider."""

    if not path.exists():
        return False
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return (
        len(rows) == 1
        and rows[0]["provider"] == provider
        and rows[0]["correctness_status"].startswith("passed")
    )


def run_case(
    args: argparse.Namespace,
    target_rate_hz: float,
    seed: int,
    part: Path,
) -> None:
    """Run one isolated target and seed combination."""

    target_activity = firing_rate_hz_to_activity(target_rate_hz, args.dt_ms)
    external_event_rate = external_event_rate_for_target(
        target_activity,
        args.max_external_event_rate,
    )

    command = [
        sys.executable,
        "benchmark/benchmark_rsnn_cudagraph_compare.py",
        "--dataset",
        args.dataset,
        "--t-steps",
        str(args.t_steps),
        "--dt",
        str(args.dt_ms),
        "--target-activity",
        str(target_activity),
        "--event-rate",
        str(external_event_rate),
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
    elif args.connectome_root is not None:
        command.extend(["--connectome-root", str(args.connectome_root)])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.isolate_providers:
        subprocess.run(
            [*command, "--providers", *DEFAULT_PROVIDERS, "--csv", str(part)],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        return

    provider_parts = []
    calibrated_amplitude = None
    for provider in DEFAULT_PROVIDERS:
        provider_part = part.with_name(f"{part.stem}_{provider}.csv")
        provider_parts.append(provider_part)
        if not args.force and valid_provider_part(provider_part, provider):
            print(f"[resume] {provider_part}")
            if calibrated_amplitude is None:
                with provider_part.open(newline="") as file:
                    calibrated_amplitude = float(
                        next(csv.DictReader(file))["calibration_input_amplitude"]
                    )
            continue
        provider_command = [*command]
        if calibrated_amplitude is not None:
            provider_command.extend(
                ["--calibrated-input-amplitude", repr(calibrated_amplitude)]
            )
        provider_command.extend(
            ["--providers", provider, "--csv", str(provider_part)]
        )
        subprocess.run(
            provider_command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        if not valid_provider_part(provider_part, provider):
            raise RuntimeError(
                f"provider did not produce one passing row: {provider_part}"
            )
        if calibrated_amplitude is None:
            with provider_part.open(newline="") as file:
                calibrated_amplitude = float(
                    next(csv.DictReader(file))["calibration_input_amplitude"]
                )
    merge_provider_parts(provider_parts, part)


def merge_provider_parts(parts: list[Path], output: Path) -> None:
    """Merge provider-isolated rows for one deterministic workload."""

    rows = []
    fieldnames = None
    for path in parts:
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            current_fields = list(reader.fieldnames or ())
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise RuntimeError(f"CSV schema mismatch in {path}")
            rows.extend(reader)
    if fieldnames is None:
        raise RuntimeError("provider-isolated run produced no rows")
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        for target_rate_hz in args.target_rates_hz:
            target_label = f"{target_rate_hz:.6f}".rstrip("0").rstrip(".")
            target_label = target_label or "0"
            part = part_dir / f"target_{target_label}hz_seed_{seed}.csv"
            parts.append(part)
            if args.force or not valid_part(part):
                run_case(args, target_rate_hz, seed, part)
            else:
                print(f"[resume] {part}")
    merge_parts(parts, output)
    manifest = {
        "dataset": args.dataset,
        "target_rates_hz": list(args.target_rates_hz),
        "dt_ms": args.dt_ms,
        "external_event_rate_policy": "min(max_rate, target_activity)",
        "max_external_event_rate": args.max_external_event_rate,
        "seeds": list(args.seeds),
        "providers": list(DEFAULT_PROVIDERS),
        "t_steps": args.t_steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "calibration_max_amplitude": args.calibration_max_amplitude,
        "calibration_iterations": args.calibration_iterations,
        "weight_scale": args.weight_scale,
        "connectome_root": (
            str(args.connectome_root) if args.connectome_root is not None else None
        ),
        "isolate_providers": args.isolate_providers,
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
