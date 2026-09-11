#!/bin/bash
set -euo pipefail
REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
SCRIPT_DIR="${REPO_DIR}/scripts/climax_patchwise_swe_ridge_audit"

exec salloc -N 1 -C cpu -q interactive -A m2637 -t 00:30:00 \
  --cpus-per-task=8 --mem=32G \
  srun --overlap --ntasks=1 --cpus-per-task=8 /bin/bash -lc "
    set -euo pipefail
    module load climate-utils/2025.01
    cd '${REPO_DIR}'
    '${PYTHON_BIN}' '${SCRIPT_DIR}/debug_fast_ridge.py'
  "
