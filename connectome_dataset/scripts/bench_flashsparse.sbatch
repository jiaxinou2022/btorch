#!/bin/bash
# Build FlashSparse's tensor-core SpMM extensions on the GPU node and run the provider
# (out-of-tree, via the registry) over the connectome corpus at fp16. Validates a real
# weighted tensor-core kernel against the oracle and lands it on the leaderboard.
#   sbatch scripts/bench_flashsparse.sbatch
#SBATCH --job-name=bench-flashsparse
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/flashsparse-%j.out
#SBATCH --error=results/slurm/flashsparse-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

echo "=== fetch + build FlashSparse extensions (sm_120) ==="
bash scripts/fetch_external.sh
MICROMAMBA="$MM" ENV="$ENV" TORCH_CUDA_ARCH_LIST=12.0 bash scripts/build_flashsparse.sh
export FLASHSPARSE_ROOT="$PWD/external/FlashSparse/FlashSparse"

echo "=== install provider package + confirm discovery ==="
"$MM" run -n "$ENV" pip install -e examples/connectome_bench_flashsparse --quiet
"$MM" run -n "$ENV" python -c "from connectome_dataset.benchmarks import providers; [print(p.framework,p.provider,'self_timed=',p.self_timed) for p in providers.discover_spmm()]"

echo "=== in-tree leaf over corpus at fp16 (N multiple of 8) ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/torch/flashsparse/ \
    --device cuda --matrix-set connectome --precisions fp16 -v -s --benchmark-autosave

echo "=== out-of-tree provider (registry discovery) over corpus at fp16 ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/external/ \
    --device cuda --matrix-set connectome --precisions fp16 -v -s --benchmark-autosave

echo "=== ingest + leaderboard ==="
NEW=$(ls -dt .benchmarks/*/*.json 2>/dev/null | head -1 || true)
[ -n "$NEW" ] && "$MM" run -n "$ENV" connectome-bench-report ingest --pytest "$NEW"
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
echo "=== done at $(date -Is) ==="
