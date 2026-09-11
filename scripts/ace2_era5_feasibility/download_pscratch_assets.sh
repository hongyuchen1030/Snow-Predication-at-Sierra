#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/global/u1/h/hyvchen/Snow-Predication-at-Sierra}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${PSCRATCH:-/pscratch/sd/h/hyvchen}}"
ACE2_ROOT="${ACE2_ROOT:-${SCRATCH_ROOT}/Snow-Predication-at-Sierra_ace2}"
ACE2_ASSETS="${ACE2_ASSETS:-${ACE2_ROOT}/assets}"
ACE2_RUNTIME="${ACE2_RUNTIME:-${ACE2_ROOT}/runtime}"
ACE2_LOGS="${ACE2_LOGS:-${ACE2_ROOT}/logs}"
SNAPSHOT_DIR="${ACE2_ROOT}/hf_snapshot"

mkdir -p "${ACE2_ASSETS}/checkpoints" "${ACE2_ASSETS}/forcing" "${ACE2_ASSETS}/initial_conditions" "${ACE2_LOGS}"

source "${ACE2_RUNTIME}/venv_full/bin/activate"
unset PYTHONPATH
export PIP_CACHE_DIR="${ACE2_RUNTIME}/pip_cache"
export TMPDIR="${ACE2_RUNTIME}/tmp"
export MPLCONFIGDIR="${ACE2_RUNTIME}/mpl_cache"
export XDG_CACHE_HOME="${ACE2_RUNTIME}/xdg_cache"

python "${REPO}/scripts/ace2_era5_feasibility/download_ace2_assets.py" \
  --output-dir "${SNAPSHOT_DIR}" \
  --ic-year 2020 \
  --forcing-years 2020 \
  >> "${ACE2_LOGS}/asset_download.log" 2>&1

cp -f "${SNAPSHOT_DIR}/ace2_era5_ckpt.tar" "${ACE2_ASSETS}/checkpoints/ace2_era5_ckpt.tar"
cp -f "${SNAPSHOT_DIR}/forcing_data/forcing_2020.nc" "${ACE2_ASSETS}/forcing/forcing_2020.nc"
cp -f "${SNAPSHOT_DIR}/initial_conditions/ic_2020.nc" "${ACE2_ASSETS}/initial_conditions/ic_2020.nc"
if [[ -f "${SNAPSHOT_DIR}/inference_config.yaml" ]]; then
  cp -f "${SNAPSHOT_DIR}/inference_config.yaml" "${ACE2_ROOT}/inference_config_reference.yaml"
fi
if [[ -f "${SNAPSHOT_DIR}/README.md" ]]; then
  cp -f "${SNAPSHOT_DIR}/README.md" "${ACE2_ROOT}/README_reference.md"
fi

ls -lh "${ACE2_ASSETS}/checkpoints/ace2_era5_ckpt.tar" \
       "${ACE2_ASSETS}/forcing/forcing_2020.nc" \
       "${ACE2_ASSETS}/initial_conditions/ic_2020.nc"
