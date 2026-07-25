"""Run all benchmarks via pytest — JAX and torch in separate processes."""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys

_BENCH = "connectome_dataset/benchmarks"
_AUTOSAVE_GLOB = ".benchmarks/**/*.json"


def _pytest(path: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", "pytest", f"{_BENCH}/{path}", "-v", "-s", "--benchmark-autosave"] + extra
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, check=False)
    if r.returncode not in (0, 5):  # 5 = no tests collected
        print(f"[WARN] pytest {path} exited with code {r.returncode}")


def _ingest_and_report(new_files: list[str]) -> None:
    from connectome_dataset.benchmarks.history import ingest, leaderboard, provenance, report, store

    if not new_files:
        return
    prov = provenance.collect()
    records = []
    for f in new_files:
        records += ingest.ingest_pytest_benchmark(f, prov)
    store.append(records)
    lb = leaderboard.build(store.load())
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)
    Path("results/report.html").write_text(report.to_html(lb, prov))
    Path("results/leaderboard.md").write_text(report.to_markdown(lb))
    print(f"Ingested {len(records)} records; wrote results/report.html + results/leaderboard.md")


def main() -> None:
    p = argparse.ArgumentParser(description="Run all benchmarks (JAX + torch, separate processes)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--graph", default="mice_column_v1")
    p.add_argument("--mode", default="all", choices=["dense", "sparse", "all"])
    p.add_argument("--quick", action="store_true", help="Fewer rounds for a smoke test")
    p.add_argument("--no-report", action="store_true", help="Skip history ingest + leaderboard report")
    p.add_argument("--synthetic", action="store_true", help="Run the synthetic controlled sweep instead of --graph")
    p.add_argument("--synthetic-size", type=int, default=None, dest="synthetic_size")
    p.add_argument("--synthetic-degree", type=float, default=None, dest="synthetic_degree")
    p.add_argument("--matrix-set", default=None, dest="matrix_set",
                   help="Sweep a named real-connectome corpus (e.g. 'connectome') instead of --graph")
    p.add_argument("--precisions", default=None,
                   help="Comma-separated precision knobs to sweep (e.g. fp32,fp16,bf16)")
    args = p.parse_args()

    common = ["--device", args.device, "--graph", args.graph, "--mode", args.mode]
    if args.quick:
        common += ["--warmup", "2", "--rounds", "5"]
    if args.precisions:
        common += ["--precisions", args.precisions]
    if args.synthetic:
        common += ["--synthetic"]
        if args.synthetic_size is not None:
            common += ["--synthetic-size", str(args.synthetic_size)]
        if args.synthetic_degree is not None:
            common += ["--synthetic-degree", str(args.synthetic_degree)]
    elif args.matrix_set:
        common += ["--matrix-set", args.matrix_set]

    subprocess.run([sys.executable, "-m", "connectome_dataset.catalog"], check=False)

    before = set(glob.glob(_AUTOSAVE_GLOB, recursive=True))
    _pytest("jax/", common)
    _pytest("torch/", common)
    _pytest("cupy/", common)  # cuSPARSE baseline (GPU only; skips cleanly on CPU)
    _pytest("nest/", common)  # NEST reference simulator (skips cleanly if nest not installed)
    _pytest("brian2/", common)  # brian2 reference simulator (skips cleanly if brian2 not installed)
    _pytest("external/", common)  # out-of-tree providers via entry points (skips if none installed)
    new_files = sorted(set(glob.glob(_AUTOSAVE_GLOB, recursive=True)) - before)

    if not args.no_report:
        _ingest_and_report(new_files)

    print("\nAll benchmarks complete.")
    print("Leaderboard:        results/leaderboard.md  ·  results/report.html")
    print("Compare runs with:  pytest-benchmark compare")


if __name__ == "__main__":
    main()
