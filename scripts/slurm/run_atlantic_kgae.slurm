#!/bin/bash
#SBATCH --job-name=atlantic_kgae
#SBATCH --time=12:00:00
#SBATCH --account=m2637
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/atlantic_kgae/logs/atlantic_kgae_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/atlantic_kgae/logs/atlantic_kgae_%j.err

set -euo pipefail

PROJECT_ROOT=/global/homes/h/hyvchen/Snow-Predication-at-Sierra
HOME_OUT=${PROJECT_ROOT}/artifacts/atlantic_kgae
HEAVY_OUT=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/atlantic_kgae
LOG_DIR=${HEAVY_OUT}/logs
PYTHON=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
STATUS_JSON=${HOME_OUT}/slurm_submission.json

mkdir -p "${HOME_OUT}" "${HEAVY_OUT}" "${LOG_DIR}" \
  "${HEAVY_OUT}/checkpoints" "${HEAVY_OUT}/latents" "${HEAVY_OUT}/reconstructions" "${HEAVY_OUT}/metrics"

export MPLCONFIGDIR=/tmp/mpl_atlantic_kgae_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_atlantic_kgae_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

{
  echo "{"
  echo "  \"job_id\": \"${SLURM_JOB_ID}\","
  echo "  \"job_name\": \"${SLURM_JOB_NAME}\","
  echo "  \"node_list\": \"${SLURM_JOB_NODELIST}\","
  echo "  \"python\": \"${PYTHON}\","
  echo "  \"status\": \"submitted\""
  echo "}"
} > "${STATUS_JSON}"

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "PYTHON=${PYTHON}"
"${PYTHON}" -c "import sys, torch, xarray as xr, numpy as np, pandas as pd; print(sys.executable); print('torch', torch.__version__); print('xarray', xr.__version__); print('numpy', np.__version__); print('pandas', pd.__version__)"

cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/run_atlantic_kgae.py \
  --mode full \
  --home-output "${HOME_OUT}" \
  --pscratch-output "${HEAVY_OUT}" \
  --device cpu \
  --epochs 120

{
  echo "{"
  echo "  \"job_id\": \"${SLURM_JOB_ID}\","
  echo "  \"job_name\": \"${SLURM_JOB_NAME}\","
  echo "  \"node_list\": \"${SLURM_JOB_NODELIST}\","
  echo "  \"python\": \"${PYTHON}\","
  echo "  \"status\": \"completed\""
  echo "}"
} > "${STATUS_JSON}"
