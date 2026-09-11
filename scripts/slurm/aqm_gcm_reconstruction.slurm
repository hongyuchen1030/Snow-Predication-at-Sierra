#!/bin/bash
#SBATCH --job-name=aqm_gcm_reconstruction
#SBATCH --time=12:00:00
#SBATCH --account=m2637
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/aqm_gcm_reconstruction/logs/aqm_gcm_reconstruction_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/aqm_gcm_reconstruction/logs/aqm_gcm_reconstruction_%j.err

set -euo pipefail

PROJECT_ROOT=/global/homes/h/hyvchen/Snow-Predication-at-Sierra
LOG_DIR=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/aqm_gcm_reconstruction/logs
mkdir -p "${LOG_DIR}"

export MPLCONFIGDIR=/tmp/mpl_aqm_gcm_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_aqm_gcm_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

module load python 2>/dev/null || true
module load conda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "PYTHON=$(which python)"
python -c "import sys, xarray as xr, numpy as np, pandas as pd, scipy, pyproj; print(sys.executable); print('xarray', xr.__version__); print('scipy', scipy.__version__); print('pyproj', pyproj.__version__)"

cd "${PROJECT_ROOT}"

python scripts/aqm_gcm_reconstruction.py
