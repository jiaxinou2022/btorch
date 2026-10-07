#!/bin/bash
#SBATCH --job-name=ss_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/zhanghan/suitesparse/slurm_logs/smoke-%A_%a.out
#SBATCH --error=/data/zhanghan/suitesparse/slurm_logs/smoke-%A_%a.err

set -euo pipefail

datasets=(hollywood_2009 vas_stokes_4m orkut uk_2002 queen_4147)
task_id="${SLURM_ARRAY_TASK_ID:?submit this script as --array=0-4}"
dataset="${datasets[${task_id}]}"
cd /home/zhanghan/btorch
/home/zhanghan/bin/micromamba run -n ml-py312 \
  python benchmark/benchmark_rsnn_cudagraph_compare.py \
  --dataset "${dataset}" \
  --connectome-root /data/zhanghan/suitesparse/processed \
  --t-steps 8 --dt 1 --event-rate 0.001 --input-seed 0 \
  --weight-scale 0.15 --warmup 1 --repeat 2 \
  --providers cusparse_direct_cudagraph persistent_spike_block \
  --csv "/data/zhanghan/suitesparse/results/${dataset}_smoke.csv"

/home/zhanghan/bin/micromamba run -n ml-py312 python -c '
import csv
import sys

with open(sys.argv[1], newline="") as file:
    rows = list(csv.DictReader(file))
expected = {"cusparse_direct_cudagraph", "persistent_spike_block"}
providers = {row["provider"] for row in rows}
passed = all(row["correctness_status"].startswith("passed") for row in rows)
if len(rows) != 2 or providers != expected or not passed:
    raise SystemExit(f"smoke validation failed: {rows}")
' "/data/zhanghan/suitesparse/results/${dataset}_smoke.csv"
