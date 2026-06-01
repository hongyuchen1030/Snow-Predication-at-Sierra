#!/bin/bash
#SBATCH --job-name=s2s_rf_top20_mon
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/slurm_logs/%x_%j.out
#SBATCH --error=/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/slurm_logs/%x_%j.err

set -euo pipefail

REPO_DIR="/global/homes/h/hyvchen/Snow-Predication-at-Sierra"
OUTPUT_DIR="$REPO_DIR/artifacts/s2s_pc6_t2m_top20_land_monthly_loyo_random_forest"
RUN_LOG="$OUTPUT_DIR/run_log.txt"

mkdir -p "$REPO_DIR/artifacts/slurm_logs"
mkdir -p "$OUTPUT_DIR"

source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true
module purge
module load python
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build

cd "$REPO_DIR"

printf "hostname=%s\nSLURM_JOB_ID=%s\nPWD=%s\n" "$(hostname)" "${SLURM_JOB_ID:-}" "$PWD" | tee "$RUN_LOG"
which python | tee -a "$RUN_LOG"
python -c "import sys, sklearn, xarray, pandas; print(sys.executable); print('sklearn=' + sklearn.__version__); print('xarray=' + xarray.__version__); print('pandas=' + pandas.__version__)" | tee -a "$RUN_LOG"

srun -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK}" \
  bash -lc 'PYTHONPATH=src:. python -u scripts/run_s2s_pc6_t2m_top20_land_monthly_loyo_random_forest.py --output-dir "'"$OUTPUT_DIR"'"' \
  |& tee -a "$RUN_LOG"
