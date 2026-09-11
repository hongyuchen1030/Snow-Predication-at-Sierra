#!/bin/bash
set -euo pipefail

REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
SCRIPT="${REPO_DIR}/scripts/climax_era5_march25_obs_latents/extract_climax_march25_obs_latents.py"

exec salloc -N 1 -C gpu -q interactive -A m2637_g -t 04:00:00 \
  --gpus=1 --cpus-per-task=32 --mem=128G \
  srun --overlap --ntasks=1 --cpus-per-task=32 --gpus=1 /bin/bash -lc "
    set -euo pipefail
    module load climate-utils/2025.01
    cd '${REPO_DIR}'
    '${PYTHON_BIN}' '${SCRIPT}'
  "
