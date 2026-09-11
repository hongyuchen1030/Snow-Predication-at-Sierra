#!/bin/bash
set -euo pipefail

REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
SCRIPT_DIR="${REPO_DIR}/scripts/climax_patchwise_swe_ridge_audit"
CMD="${1:-validate}"

exec salloc -N 1 -C cpu -q interactive -A m2637 -t 04:00:00 \
  --cpus-per-task=32 --mem=128G \
  srun --overlap --ntasks=1 --cpus-per-task=32 /bin/bash -lc "
    set -euo pipefail
    module load climate-utils/2025.01
    cd '${REPO_DIR}'
    if [ '${CMD}' = 'validate' ]; then
      '${PYTHON_BIN}' '${SCRIPT_DIR}/validate_fast_ridge_loyo.py'
    else
      '${PYTHON_BIN}' '${SCRIPT_DIR}/run_patchwise_audit.py'
    fi
  "
