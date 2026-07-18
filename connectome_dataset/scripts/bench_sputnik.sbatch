#!/bin/bash
# Build the in-tree Sputnik + cuSPARSE baselines on the GPU node and run both over the
# connectome corpus (fp32), so the leaderboard compares a real SOTA CUDA-core kernel
# (Sputnik) against cuSPARSE and the Python providers in the same cells.
#   sbatch scripts/bench_sputnik.sbatch
#SBATCH --job-name=bench-sputnik
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/sputnik-%j.out
#SBATCH --error=results/slurm/sputnik-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm results/cpp .cache/mtx

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

echo "=== fetch external (sputnik) + build ==="
bash scripts/fetch_external.sh
"$MM" run -n "$ENV" cmake -B cpp/build-sputnik -S cpp -DCONNECTOME_BENCH_BUILD_SPUTNIK=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
"$MM" run -n "$ENV" cmake --build cpp/build-sputnik --target connectome_bench -j8
BIN=cpp/build-sputnik/connectome_bench

echo "=== export corpus + run sputnik vs cusparse (fp32) ==="
"$MM" run -n "$ENV" python scripts/export_corpus_mtx.py --matrix-set connectome --out .cache/mtx
for mtx in .cache/mtx/*.mtx; do
  graph=$(basename "$mtx" .mtx)
  for alg in cusparse.csr sputnik.csr; do
    for batch in 1 8 32 128; do
      out="results/cpp/${graph}_${alg}_b${batch}"
      "$BIN" --matrix "$mtx" --target spmv --alg "$alg" --dtype float32 \
             --batch "$batch" --warmup 10 --rep 50 --out "$out" \
        || echo "[warn] failed: $graph $alg b$batch"
    done
  done
done

echo "=== ingest + leaderboard ==="
"$MM" run -n "$ENV" connectome-bench-report ingest --cpp "results/cpp/**/records.jsonl"
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md

echo "=== done at $(date -Is) ==="
