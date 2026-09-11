#!/bin/bash
#SBATCH --job-name=M4_attention_swe
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/slurm_logs/%x_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON=/global/common/software/nersc9/pytorch/2.11.0/bin/python
RUN_ROOT=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1
LOG_DIR=${RUN_ROOT}/slurm_logs
OUTPUT_ROOT=${RUN_ROOT}/experiments
RUN_LOG=${LOG_DIR}/M4_attention_swe_${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOB_NAME=${SLURM_JOB_NAME}"
  echo "HOSTNAME=$(hostname)"
  echo "NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "PYTHON=${PYTHON}"
} | tee "${RUN_LOG}"

"${PYTHON}" - <<'PY' | tee -a "${RUN_LOG}"
import sys
import torch
print(sys.executable)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
print("cuda_runtime=" + str(torch.version.cuda))
print("device_count=" + str(torch.cuda.device_count()))
if torch.cuda.is_available():
    print("gpu_name=" + torch.cuda.get_device_name(0))
PY

srun -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK}" \
  bash -lc "cd '${PROJECT_ROOT}' && \
    PYTHONUNBUFFERED=1 PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' \
    '${PYTHON}' -u scripts/run_cmip6_cnn_experiment.py \
      --architecture M4_attention_swe \
      --predictor-cache-dir '${RUN_ROOT}/data_cache' \
      --output-root '${OUTPUT_ROOT}' \
      --seed 20260813 \
      --batch-size 2 \
      --max-epochs 100 \
      --early-stopping-patience 15 \
      --learning-rate 1e-3 \
      --weight-decay 1e-4 \
      --num-workers 0" \
  |& tee -a "${RUN_LOG}"
