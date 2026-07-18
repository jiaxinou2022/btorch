#!/bin/bash
# Full SpMM comparison over the SNAP power-law corpus: build the CUDA baselines + SOTA
# kernels and run them all (weighted). cuSPARSE/Sputnik (C++, fp32) via the runner;
# torch/cupy/jax + DTC-SpMM (fp32) and FlashSparse (fp16) via the Python harness.
#   sbatch scripts/bench_snap.sbatch
#SBATCH --job-name=bench-snap
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/snap-%j.out
#SBATCH --error=results/slurm/snap-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm results/cpp .cache/mtx

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312
SET=snap

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true
"$MM" run -n "$ENV" python -m connectome_dataset.catalog >/dev/null 2>&1 || true

echo "=== build C++ cuSPARSE + Sputnik, and FlashSparse + DTC extensions ==="
bash scripts/fetch_external.sh
"$MM" run -n "$ENV" cmake -B cpp/build-sputnik -S cpp -DCONNECTOME_BENCH_BUILD_SPUTNIK=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
"$MM" run -n "$ENV" cmake --build cpp/build-sputnik --target connectome_bench -j8
MICROMAMBA="$MM" ENV="$ENV" TORCH_CUDA_ARCH_LIST=12.0 bash scripts/build_flashsparse.sh
MICROMAMBA="$MM" ENV="$ENV" TORCH_CUDA_ARCH_LIST=12.0 bash scripts/build_dtc.sh
export FLASHSPARSE_ROOT="$PWD/external/FlashSparse/FlashSparse"
export DTC_ROOT="$PWD/external/DTC-SpMM/DTC-SpMM"
"$MM" run -n "$ENV" pip install -e examples/connectome_bench_dtc examples/connectome_bench_flashsparse --quiet

echo "=== C++ cuSPARSE + Sputnik over SNAP (fp32) ==="
"$MM" run -n "$ENV" python scripts/export_corpus_mtx.py --matrix-set "$SET" --out .cache/mtx-snap
for mtx in .cache/mtx-snap/*.mtx; do
  g=$(basename "$mtx" .mtx)
  for alg in cusparse.csr sputnik.csr; do
    for b in 1 8 32 128; do
      cpp/build-sputnik/connectome_bench --matrix "$mtx" --target spmv --alg "$alg" --dtype float32 \
        --batch "$b" --warmup 5 --rep 20 --out "results/cpp/snap_${g}_${alg}_b${b}" \
        || echo "[warn] failed $g $alg b$b"
    done
  done
done

echo "=== Python providers over SNAP (fp32: torch/cupy/jax + in-tree DTC) ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/torch connectome_dataset/benchmarks/cupy connectome_dataset/benchmarks/jax \
    --device cuda --matrix-set "$SET" --precisions fp32 -q --benchmark-autosave || true
echo "=== FlashSparse over SNAP (fp16) ==="
"$MM" run -n "$ENV" python -m pytest connectome_dataset/benchmarks/torch/flashsparse \
    --device cuda --matrix-set "$SET" --precisions fp16 -q --benchmark-autosave || true

echo "=== ingest + leaderboard ==="
"$MM" run -n "$ENV" connectome-bench-report ingest --cpp "results/cpp/snap_*/**/records.jsonl" \
    --pytest ".benchmarks/**/*.json"
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
echo "=== done at $(date -Is) ==="
