#!/bin/bash
#SBATCH --job-name=ss_activity
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/zhanghan/suitesparse/slurm_logs/activity-%A_%a.out
#SBATCH --error=/data/zhanghan/suitesparse/slurm_logs/activity-%A_%a.err

set -euo pipefail

datasets=(hollywood_2009 vas_stokes_4m orkut queen_4147)
rates=(0 0.1 0.2 0.5 1 2 5 10 20 50)
task_id="${SLURM_ARRAY_TASK_ID:?submit this script as --array=0-119%4}"
dataset_index=$((task_id / 30))
within_dataset=$((task_id % 30))
seed=$((within_dataset / 10))
rate_index=$((within_dataset % 10))
dataset="${datasets[${dataset_index}]}"
rate="${rates[${rate_index}]}"
output_root=/data/zhanghan/suitesparse/results/activity_parts

mkdir -p "${output_root}"
cd /home/zhanghan/btorch
/home/zhanghan/bin/micromamba run -n ml-py312 \
  python benchmark/benchmark_activity_sweep.py \
  --dataset "${dataset}" \
  --connectome-root /data/zhanghan/suitesparse/processed \
  --target-rates-hz "${rate}" --seeds "${seed}" \
  --t-steps 256 --dt-ms 1 --warmup 10 --repeat 20 \
  --calibration-iterations 20 --isolate-providers \
  --output "${output_root}/${dataset}_${rate}hz_seed${seed}.csv"
