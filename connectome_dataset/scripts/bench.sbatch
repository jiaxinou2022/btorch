#!/bin/bash
# Run the connectome benchmark suite on a GPU node.
#   sbatch scripts/bench.sbatch                 # catalog graph + synthetic sweep, cuda
#   sbatch scripts/bench.sbatch --quick         # smoke test (fewer rounds)
# Extra args are forwarded to `connectome-bench`.
#SBATCH --job-name=connectome-bench
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/bench-%j.out
#SBATCH --error=results/slurm/bench-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo

# 1) real catalog graph (mice_column_v1 by default)
echo "=== catalog suite ==="
"$MM" run -n "$ENV" connectome-bench --device cuda "$@"

# 2) synthetic controlled sweep (banded / ER / power-law / RMAT)
echo "=== synthetic sweep ==="
"$MM" run -n "$ENV" connectome-bench --device cuda --synthetic "$@"

# 3) curated real-connectome corpus (spans size / density / degree-skew)
echo "=== connectome corpus ==="
"$MM" run -n "$ENV" connectome-bench --device cuda --matrix-set connectome "$@"

echo "=== done at $(date -Is) — leaderboard: results/leaderboard.md, results/report.html ==="
