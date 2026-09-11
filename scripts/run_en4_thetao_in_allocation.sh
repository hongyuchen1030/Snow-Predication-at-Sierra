#!/bin/bash
# Run the EN4 build as an explicit step inside an existing interactive allocation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 SLURM_JOB_ID" >&2
  exit 2
fi

REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
STAGE_DIR=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/en4_thetao_stage/full_1984_2021
OUTPUT_DIR=${STAGE_DIR}/processed_cnn_wy1985_2021
PYTHON_BIN=/global/u1/h/hyvchen/.conda/envs/uxarray_build/bin/python

exec srun --jobid="$1" --overlap --ntasks=1 --cpus-per-task=8 /bin/bash -lc "
  cd '${REPO_DIR}'
  '${PYTHON_BIN}' scripts/stage_en4_thetao_obs_audit.py build-en4-thetao \\
    --stage-dir '${STAGE_DIR}' \\
    --output-dir '${OUTPUT_DIR}' \\
    --start-month 1984-09 \\
    --end-month 2021-03 \\
    --build-cnn-rows \\
    --start-row-year 1984 \\
    --end-row-year 2020
"
