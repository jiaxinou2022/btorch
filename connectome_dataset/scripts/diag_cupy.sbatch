#!/bin/bash
#SBATCH --job-name=diag-cupy
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/diag-cupy-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm
/home/fanqixuan/bin/micromamba run -n ml-py312 python -m connectome_dataset.catalog >/dev/null 2>&1 || true
/home/fanqixuan/bin/micromamba run -n ml-py312 python scripts/diag_cupy_cusparse.py
