#!/bin/bash
set -euo pipefail

REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python

module load climate-utils/2025.01

exec salloc -N 1 -C cpu -q interactive -A m2637 -t 04:00:00 \
  --cpus-per-task=8 --mem=128G \
  srun --overlap --ntasks=1 --cpus-per-task=8 /bin/bash -lc "
    set -euo pipefail
    module load climate-utils/2025.01
    cd '${REPO_DIR}'
    echo '=== [1/4] regrid+standardize (canonical output-root, incremental) ==='
    '${PYTHON_BIN}' scripts/process_cmip6_selected_regrid_standardize.py
    echo '=== [1/4] OK ==='

    echo '=== [2/4] rebuild CPM/AQM aux labels (canonical) ==='
    '${PYTHON_BIN}' scripts/build_cmip6_aux_labels.py
    echo '=== [2/4] OK ==='

    echo '=== [3/4] rebuild WUS-D3 SWE labels (canonical) ==='
    '${PYTHON_BIN}' scripts/build_wusd3_swe_labels.py
    echo '=== [3/4] OK ==='

    echo '=== [4/4] rebuild final 19-var training manifest into VALIDATION dir (not canonical) ==='
    '${PYTHON_BIN}' scripts/prepare_cmip6_cnn_experiment.py \\
      --output-dir /pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1_VALIDATION_mpi_ssp370_fix/data_cache \\
      --predictor-set all19
    echo '=== [4/4] OK ==='

    echo '=== ALL STAGES DONE ==='
  "
