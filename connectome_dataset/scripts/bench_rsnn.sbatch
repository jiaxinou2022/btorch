#!/bin/bash
# beNNch RSNN scaling sweep on a GPU node (balanced E-I coba/cuba + connectome).
#   sbatch scripts/bench_rsnn.sbatch                    # full strong-scaling sweep
#   sbatch scripts/bench_rsnn.sbatch --rsnn-scale 1,2   # smaller sweep
# Extra args are forwarded to pytest (e.g. --rsnn-model, --rsnn-scale, --rsnn-timesteps).
#SBATCH --job-name=bench-rsnn
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/bench-rsnn-%j.out
#SBATCH --error=results/slurm/bench-rsnn-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312
LEAF=connectome_dataset/benchmarks/jax/brainstate/

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true
echo

# Strong scaling: fixed model, growing network. 3200*scale exc + 800*scale inh neurons.
SCALES="${RSNN_SCALES:-1,2,4,8,16}"

echo "=== beNNch balanced E-I (coba/cuba) strong scaling: scales=$SCALES ==="
"$MM" run -n "$ENV" python -m pytest "$LEAF" -v -s --benchmark-autosave \
    --device cuda --rsnn-model cuba,coba --rsnn-scale "$SCALES" "$@"

# Connectome-connectivity variant: the catalog adjacency is the recurrent operator.
echo "=== beNNch connectome-connectivity RSNN ==="
"$MM" run -n "$ENV" python -m pytest "$LEAF" -v -s --benchmark-autosave \
    --device cuda --rsnn-model connectome --graph "${RSNN_GRAPH:-mice_column_v1}" "$@"

# Fold the new runs into the shared history + leaderboard.
echo "=== ingest + report ==="
"$MM" run -n "$ENV" connectome-bench-report ingest --pytest ".benchmarks/**/*.json"
"$MM" run -n "$ENV" connectome-bench-report report

echo "=== done at $(date -Is) — leaderboard: results/leaderboard.md ==="
