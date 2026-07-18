#!/bin/bash
# Build the C++ cuSPARSE baseline on the GPU node and run it over the connectome
# corpus at fp32 / fp16 / bf16 — the real cuSPARSE half-precision path CuPy can't
# express. Records land in the same leaderboard as the Python providers.
#   sbatch scripts/bench_cpp_cusparse.sbatch
#SBATCH --job-name=cpp-cusparse
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/cpp-cusparse-%j.out
#SBATCH --error=results/slurm/cpp-cusparse-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm results/cpp .cache/mtx

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

echo "=== build CUDA baseline ==="
"$MM" run -n "$ENV" cmake -B cpp/build-cuda -S cpp -DCONNECTOME_BENCH_BUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
"$MM" run -n "$ENV" cmake --build cpp/build-cuda --target connectome_bench -j8
BIN=cpp/build-cuda/connectome_bench

echo "=== export connectome corpus to .mtx ==="
"$MM" run -n "$ENV" python scripts/export_corpus_mtx.py --matrix-set connectome --out .cache/mtx

echo "=== run cuSPARSE over corpus x {fp32,fp16,bf16} x batch ==="
for mtx in .cache/mtx/*.mtx; do
  graph=$(basename "$mtx" .mtx)
  for dtype in float32 float16 bfloat16; do
    for batch in 1 8 32 128; do
      out="results/cpp/${graph}_${dtype}_b${batch}"
      # one bad case (e.g. an unsupported combo) must not abort the sweep
      "$BIN" --matrix "$mtx" --target spmv --alg cusparse.csr --dtype "$dtype" \
             --batch "$batch" --warmup 10 --rep 50 --out "$out" \
        || echo "[warn] failed: $graph $dtype b$batch"
    done
  done
done

echo "=== ingest + leaderboard ==="
"$MM" run -n "$ENV" connectome-bench-report ingest --cpp "results/cpp/**/records.jsonl"
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md

echo "=== done at $(date -Is) — leaderboard: results/leaderboard.md ==="
