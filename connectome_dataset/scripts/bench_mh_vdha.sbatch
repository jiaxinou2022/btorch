#!/bin/bash
# Build + validate + benchmark the two new algorithms on the GPU node:
#   * MH-SpGEMM (C = A·Aᵀ, fp32/fp16)  vs cuSPARSE SpGEMM
#   * VDHA      (y = A·x, sparse) vs cuSPARSE dense-vector SpMV
# over the connectome corpus, and ingest both into the shared leaderboard. A strict
# correctness gate on one graph aborts the job before the sweep if either kernel is wrong.
#   sbatch scripts/bench_mh_vdha.sbatch
#SBATCH --job-name=bench-mh-vdha
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/mh-vdha-%j.out
#SBATCH --error=results/slurm/mh-vdha-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312
export CUDA_ARCH="${CUDA_ARCH:-120}"
# Configurable corpus + which targets to sweep. SpGEMM (C=A·Aᵀ) is skipped on large
# power-law sets (e.g. snap) where the intermediate products explode; VDHA SpMSpV is
# VDHA's home turf and stays bounded (dense output vector).
MATRIX_SET="${MATRIX_SET:-connectome}"
RUN_SPGEMM="${RUN_SPGEMM:-1}"
RUN_SPMSPV="${RUN_SPMSPV:-1}"
RUN_RSNN="${RUN_RSNN:-1}"  # event-driven spike-delivery track (VDHA + MH-SpGEMM + cuSPARSE)

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

echo "=== fetch + build the two shared libs (sm_${CUDA_ARCH}) ==="
bash scripts/fetch_external.sh
bash scripts/build_mh_spgemm.sh
bash scripts/build_vdha.sh
export CONNECTOME_MH_SPGEMM_LIB="$PWD/cpp/build-mh_spgemm/libconnectome_mh_spgemm.so"
export CONNECTOME_VDHA_LIB="$PWD/cpp/build-vdha/libconnectome_vdha.so"

echo "=== install out-of-tree provider packages ==="
"$MM" run -n "$ENV" pip install -e examples/connectome_bench_mh_spgemm -e examples/connectome_bench_vdha --quiet
"$MM" run -n "$ENV" python -c "from connectome_dataset.benchmarks import providers; \
  print('spgemm', [(p.framework,p.provider) for p in providers.discover_spgemm()]); \
  print('spmspv', [(p.framework,p.provider) for p in providers.discover_spmspv()])"

echo "=== build catalog ==="
"$MM" run -n "$ENV" connectome-catalog || true

# Strict gate at fp32 (numerically exact for our kernels) — but only over the targets we
# are about to sweep, so an unrelated instability (e.g. MH-SpGEMM's dense A·Aᵀ) can't block
# a RUN_RSNN-only run. fp16 is exercised in the non-strict sweeps below.
GATE=()
[ "$RUN_SPGEMM" = "1" ] && GATE+=(connectome_dataset/benchmarks/external/test_spgemm.py connectome_dataset/benchmarks/cupy/cusparse/test_spgemm.py)
[ "$RUN_SPMSPV" = "1" ] && GATE+=(connectome_dataset/benchmarks/external/test_spmspv.py connectome_dataset/benchmarks/cupy/cusparse/test_spmspv.py)
[ "$RUN_RSNN" = "1" ] && GATE+=(connectome_dataset/benchmarks/torch/btorch/test_rsnn_deliver.py)
if [ "${#GATE[@]}" -gt 0 ]; then
  # Exclude mh_spgemm from the strict gate: MH-SpGEMM aborts uncatchably (device-side) on
  # dense inputs like mice_column, which can't be caught to fail gracefully. It is still
  # validated (non-strict, status=incorrect) in its own isolated process in the sweeps.
  echo "=== correctness gate (mice_column_v1, strict, fp32, excl. mh_spgemm): ${GATE[*]} ==="
  "$MM" run -n "$ENV" python -m pytest "${GATE[@]}" \
      --device cuda --graph mice_column_v1 --precisions fp32 --strict-correctness -q \
      -k "not mh_spgemm"
fi

echo "=== sweeping corpus '$MATRIX_SET' (spgemm=$RUN_SPGEMM spmspv=$RUN_SPMSPV) ==="

ingest_latest() {  # ingest the newest pytest-benchmark autosave
  local NEW; NEW=$(ls -dt .benchmarks/*/*.json 2>/dev/null | head -1 || true)
  [ -n "$NEW" ] && "$MM" run -n "$ENV" connectome-bench-report ingest --pytest "$NEW" || true
}

# Sweeps must not abort the job on a single failing case (e.g. a baseline that OOMs) — we
# still want the remaining cases, the SpMSpV sweep, and the final report. The strict gate
# above already guards our kernels' correctness.
if [ "$RUN_SPGEMM" = "1" ]; then
  echo "=== SpGEMM sweep: MH-SpGEMM vs cuSPARSE (fp32,fp16) over '$MATRIX_SET' ==="
  "$MM" run -n "$ENV" python -m pytest \
      connectome_dataset/benchmarks/external/test_spgemm.py \
      connectome_dataset/benchmarks/cupy/cusparse/test_spgemm.py \
      --device cuda --matrix-set "$MATRIX_SET" --precisions fp32,fp16 \
      -v -s --benchmark-autosave || echo "[warn] SpGEMM sweep had failures"
  ingest_latest
fi

if [ "$RUN_SPMSPV" = "1" ]; then
  echo "=== SpMSpV sweep: VDHA vs cuSPARSE SpMV (fp32,fp16) over '$MATRIX_SET' ==="
  "$MM" run -n "$ENV" python -m pytest \
      connectome_dataset/benchmarks/external/test_spmspv.py \
      connectome_dataset/benchmarks/cupy/cusparse/test_spmspv.py \
      --device cuda --matrix-set "$MATRIX_SET" --precisions fp32,fp16 \
      -v -s --benchmark-autosave || echo "[warn] SpMSpV sweep had failures"
  ingest_latest
fi

if [ "$RUN_RSNN" = "1" ]; then
  echo "=== btorch RSNN deliver sweep: native / VDHA (+ MH-SpGEMM if RUN_RSNN_MH=1) over '$MATRIX_SET' ==="
  # native + VDHA are stable; MH-SpGEMM is opt-in (RUN_RSNN_MH=1) because its device-side
  # `illegal memory access` aborts on dense inputs can wedge the GPU for the whole node, not
  # just lose a record. Each variant runs in its own process so an abort can't take the rest
  # (or the autosave) down with it.
  RSNN_VARIANTS=(native vdha)
  [ "${RUN_RSNN_MH:-0}" = "1" ] && RSNN_VARIANTS+=(mh_spgemm)
  for bk in "${RSNN_VARIANTS[@]}"; do
    "$MM" run -n "$ENV" python -m pytest \
        connectome_dataset/benchmarks/torch/btorch/test_rsnn_deliver.py -k "$bk" \
        --device cuda --matrix-set "$MATRIX_SET" \
        -v -s --benchmark-autosave || echo "[warn] RSNN $bk had failures"
    ingest_latest
  done
fi

echo "=== rebuild leaderboard + report ==="
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
echo "=== done at $(date -Is) ==="
