#!/bin/bash
# Precision-knob sweep on a GPU node: run the connectome corpus at fp32 / fp16 / bf16
# so the leaderboard compares tensor-core precisions within each workload cell.
#   sbatch scripts/bench_precision.sbatch
#SBATCH --job-name=bench-precision
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/precision-%j.out
#SBATCH --error=results/slurm/precision-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

"$MM" run -n "$ENV" connectome-bench --device cuda --matrix-set connectome \
    --precisions fp32,fp16,bf16 "$@"

echo "=== done at $(date -Is) — leaderboard: results/leaderboard.md ==="
