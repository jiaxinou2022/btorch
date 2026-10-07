#!/bin/bash
#SBATCH --job-name=ss_gpu_check
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/zhanghan/suitesparse/slurm_logs/gpu-check-%A_%a.out
#SBATCH --error=/data/zhanghan/suitesparse/slurm_logs/gpu-check-%A_%a.err

set -euo pipefail

datasets=(orkut hollywood_2009 vas_stokes_4m uk_2002 queen_4147)
task_id="${SLURM_ARRAY_TASK_ID:?submit this script as --array=0-4}"
dataset="${datasets[${task_id}]}"
cd /home/zhanghan/btorch
/home/zhanghan/bin/micromamba run -n ml-py312 \
  python benchmark/validate_suitesparse_gpu.py "${dataset}" \
  --root /data/zhanghan/suitesparse/processed \
  --output "/data/zhanghan/suitesparse/results/${dataset}_gpu_memory.json"
