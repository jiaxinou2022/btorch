#!/bin/bash
# Build DTC-SpMM (weighted, patched) on the GPU node and run it over the connectome corpus
# at fp32 (in-tree leaf + out-of-tree provider), validating against the oracle.
#   sbatch scripts/bench_dtc.sbatch
#SBATCH --job-name=bench-dtc
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/dtc-%j.out
#SBATCH --error=results/slurm/dtc-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

echo "=== fetch + build DTC-SpMM (sm_120, weighted patch) ==="
bash scripts/fetch_external.sh
MICROMAMBA="$MM" ENV="$ENV" TORCH_CUDA_ARCH_LIST=12.0 bash scripts/build_dtc.sh
export DTC_ROOT="$PWD/external/DTC-SpMM/DTC-SpMM"

echo "=== install provider + confirm discovery ==="
"$MM" run -n "$ENV" pip install -e examples/connectome_bench_dtc --quiet
"$MM" run -n "$ENV" python -c "from connectome_dataset.benchmarks import providers; [print(p.framework,p.provider,'self_timed=',p.self_timed) for p in providers.discover_spmm()]"

echo "=== in-tree leaf over corpus at fp32 (N multiple of 16) ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/torch/dtc/ \
    --device cuda --matrix-set connectome --precisions fp32 -v -s --benchmark-autosave

echo "=== out-of-tree provider (registry discovery) over corpus at fp32 ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/external/ \
    --device cuda --matrix-set connectome --precisions fp32 -v -s --benchmark-autosave -k dtc

echo "=== ingest + leaderboard ==="
for j in $(ls -dt .benchmarks/*/*.json 2>/dev/null | head -2); do
  "$MM" run -n "$ENV" connectome-bench-report ingest --pytest "$j"
done
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
echo "=== done at $(date -Is) ==="
