#!/bin/bash
# Validate the out-of-tree provider path end-to-end on GPU: build the Sputnik shared
# lib, install the example provider packages, and run the discovery-driven benchmark
# (external/test_spmm.py) over the corpus. Proves a real SOTA kernel reaches the
# leaderboard through the entry-point registry with no edit to the main tree.
#   sbatch scripts/bench_external.sbatch
#SBATCH --job-name=bench-external
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/external-%j.out
#SBATCH --error=results/slurm/external-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

echo "=== build sputnik shared lib + install example providers ==="
bash scripts/fetch_external.sh
"$MM" run -n "$ENV" cmake -B cpp/build-sputnik -S cpp -DCONNECTOME_BENCH_BUILD_SPUTNIK=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
"$MM" run -n "$ENV" cmake --build cpp/build-sputnik --target connectome_sputnik -j8
export CONNECTOME_SPUTNIK_LIB="$PWD/cpp/build-sputnik/libconnectome_sputnik.so"
"$MM" run -n "$ENV" pip install -e examples/connectome_bench_scipy examples/connectome_bench_sputnik --quiet

echo "=== discovered providers ==="
"$MM" run -n "$ENV" python -c "from connectome_dataset.benchmarks import providers; [print(p.framework, p.provider, p.variants) for p in providers.discover_spmm()]"

echo "=== run discovery-driven benchmark over the corpus (fp32) ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/external/ \
    --device cuda --matrix-set connectome -v -s --benchmark-autosave

echo "=== ingest + leaderboard ==="
NEW=$(ls -dt .benchmarks/*/*.json 2>/dev/null | head -1 || true)
[ -n "$NEW" ] && "$MM" run -n "$ENV" connectome-bench-report ingest --pytest "$NEW"
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
echo "=== done at $(date -Is) ==="
