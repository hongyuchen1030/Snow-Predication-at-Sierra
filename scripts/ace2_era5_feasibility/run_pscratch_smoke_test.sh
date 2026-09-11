#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/global/u1/h/hyvchen/Snow-Predication-at-Sierra}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${PSCRATCH:-/pscratch/sd/h/hyvchen}}"
ACE2_ROOT="${ACE2_ROOT:-${SCRATCH_ROOT}/Snow-Predication-at-Sierra_ace2}"
ACE2_RUNTIME="${ACE2_RUNTIME:-${ACE2_ROOT}/runtime}"
ACE2_LOGS="${ACE2_LOGS:-${ACE2_ROOT}/logs}"

cd "${REPO}"
source "${ACE2_RUNTIME}/venv_full/bin/activate"
unset PYTHONPATH
export ACE2_PYTHON="${ACE2_RUNTIME}/venv_full/bin/python"
export PIP_CACHE_DIR="${ACE2_RUNTIME}/pip_cache"
export TMPDIR="${ACE2_RUNTIME}/tmp"
export MPLCONFIGDIR="${ACE2_RUNTIME}/mpl_cache"
export XDG_CACHE_HOME="${ACE2_RUNTIME}/xdg_cache"

./scripts/ace2_era5_feasibility/run_stock_inference_smoke_test.sh \
  artifacts/ace2_era5_feasibility/configs/smoke_test_inference_config.yaml \
  >> "${ACE2_LOGS}/smoke_test.log" 2>&1
