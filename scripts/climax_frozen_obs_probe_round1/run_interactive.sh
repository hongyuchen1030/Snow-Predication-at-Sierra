#!/bin/bash
set -euo pipefail

REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
PYTHON_BIN=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
PROBE_SCRIPT="${REPO_DIR}/scripts/climax_frozen_obs_probe_round1/run_probe.py"
PLOT_SCRIPT="${REPO_DIR}/scripts/climax_frozen_obs_probe_round1/make_plots.py"

exec salloc -N 1 -C cpu -q interactive -A m2637 -t 04:00:00 \
  --cpus-per-task=16 --mem=64G \
  srun --overlap --ntasks=1 --cpus-per-task=16 /bin/bash -lc "
    set -euo pipefail
    module load climate-utils/2025.01
    export PROBE_PYTHON_BIN='${PYTHON_BIN}'
    export PROBE_SKIP_CPM_EXTENSION=1
    cd '${REPO_DIR}'
    echo '=== [1/2] run_probe.py ==='
    '${PYTHON_BIN}' '${PROBE_SCRIPT}'
    echo '=== [1/2] OK ==='
    echo '=== [2/2] make_plots.py ==='
    '${PYTHON_BIN}' '${PLOT_SCRIPT}'
    echo '=== [2/2] OK ==='
    echo '=== ALL DONE ==='
  "
