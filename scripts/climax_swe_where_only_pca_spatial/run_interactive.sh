#!/bin/bash
set -euo pipefail
REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
SCRIPT_DIR="${REPO_DIR}/scripts/climax_swe_where_only_pca_spatial"
OUT_DIR=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/climax_swe_where_only_pca_spatial

exec salloc -N 1 -C cpu -q interactive -A m2637 -t 04:00:00 \
  --cpus-per-task=32 --mem=64G \
  srun --overlap --ntasks=1 --cpus-per-task=32 /bin/bash -lc "
    set -euo pipefail
    module load climate-utils/2025.01
    cd '${REPO_DIR}'
    '${PYTHON_BIN}' '${SCRIPT_DIR}/run_experiment_a.py' 2>&1 | tee '${OUT_DIR}/logs/run_experiment_a.log'
  "
