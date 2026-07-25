#!/bin/bash
# Demo GPU benchmark over the Guozhang mouse-V1 dataset family:
#   * mice_column_v1     n=4,166    — the 200um V1 column
#   * mice_column_v1 x4 replicate    — scale-up: 4 tiled copies + 1% random inter-block edges
#   * mice_v1_guozhang   n=51,978   — the full GLIF V1 model core (~12x larger)
# Sweeps the standard sparse targets (SpMM / SpGEMM / SpMSpV / RSNN) across every
# installed framework/provider, then a replicate-scaling demo, and ingests both into
# the shared leaderboard. Frameworks run in separate processes (jax/torch/cupy/external).
#   sbatch scripts/bench_v1.sbatch
#SBATCH --job-name=bench-v1
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/bench-v1-%j.out
#SBATCH --error=results/slurm/bench-v1-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/fanqixuan/micromamba}"
MM="${MICROMAMBA:-/home/fanqixuan/bin/micromamba}"
ENV=ml-py312

# Which parts to run + how far to scale the replicate demo.
RUN_SET="${RUN_SET:-1}"            # the v1 matrix-set sweep (mice_column_v1 + mice_v1_guozhang)
RUN_REPLICATE="${RUN_REPLICATE:-1}"  # the mice_column_v1 replicate-scaling demo
REPLICATE="${REPLICATE:-4}"        # replicate factor for the scale-up demo
INTER_DENSITY="${INTER_DENSITY:-0.01}"  # density of random inter-block edges (0 = pure block-diagonal)

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is) ==="
nvidia-smi -L || true

# Ensure the DVC-managed payload is present, then (re)build the catalog so
# mice_v1_guozhang is registered. build_mice_v1_guozhang.py can regenerate it from the raw model.
echo "=== ensure mice_v1_guozhang payload + catalog ==="
if [ ! -f data/external/mice_v1_guozhang/mice_v1_guozhang.npz ]; then
  "$MM" run -n "$ENV" dvc pull data/external/mice_v1_guozhang.dvc \
    || "$MM" run -n "$ENV" python scripts/build_mice_v1_guozhang.py
fi
"$MM" run -n "$ENV" connectome-catalog | grep -E "mice_v1_guozhang|mice_column_v1" || true

if [ "$RUN_SET" = "1" ]; then
  echo "=== v1 family sweep (mice_column_v1 + mice_v1_guozhang), all targets/frameworks ==="
  "$MM" run -n "$ENV" connectome-bench --device cuda --matrix-set v1
fi

if [ "$RUN_REPLICATE" = "1" ]; then
  echo "=== scale-up demo: mice_column_v1 x{1,$REPLICATE}, inter-density $INTER_DENSITY (torch) ==="
  # run_all does not forward --replicate, so drive pytest directly (its conftest does).
  before=$(ls -dt .benchmarks/*/*.json 2>/dev/null | head -1 || true)
  for r in 1 "$REPLICATE"; do
    "$MM" run -n "$ENV" python -m pytest \
        connectome_dataset/benchmarks/torch \
        --device cuda --graph mice_column_v1 --replicate "$r" \
        --replicate-inter-density "$INTER_DENSITY" \
        -v -s --benchmark-autosave || echo "[warn] replicate x$r had failures"
  done
  NEW=$(ls -dt .benchmarks/*/*.json 2>/dev/null | head -1 || true)
  [ -n "$NEW" ] && [ "$NEW" != "$before" ] && \
    "$MM" run -n "$ENV" connectome-bench-report ingest --pytest "$NEW" || true
fi

echo "=== rebuild leaderboard + report ==="
"$MM" run -n "$ENV" connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
echo "=== done at $(date -Is) — results/leaderboard.md, results/report.html ==="
