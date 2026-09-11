#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A m2637
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -J mpi_ssp370_gapfill
#SBATCH -o /pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/logs/mpi_ssp370_gapfill_%j.out
#SBATCH -e /pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/logs/mpi_ssp370_gapfill_%j.err
set -euo pipefail

module load climate-utils/2025.01
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
cd "$REPO_DIR"

echo "=== [1/4] regrid+standardize (canonical output-root, incremental) ==="
"$PYTHON_BIN" scripts/process_cmip6_selected_regrid_standardize.py

echo "=== [2/4] rebuild CPM/AQM aux labels (canonical) ==="
"$PYTHON_BIN" scripts/build_cmip6_aux_labels.py

echo "=== [3/4] rebuild WUS-D3 SWE labels (canonical) ==="
"$PYTHON_BIN" scripts/build_wusd3_swe_labels.py

echo "=== [4/4] rebuild final 19-var training manifest into VALIDATION dir (not canonical) ==="
"$PYTHON_BIN" scripts/prepare_cmip6_cnn_experiment.py \
  --output-dir /pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1_VALIDATION_mpi_ssp370_fix/data_cache \
  --predictor-set all19

echo "=== DONE ==="
