#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/global/u1/h/hyvchen/Snow-Predication-at-Sierra}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${PSCRATCH:-/pscratch/sd/h/hyvchen}}"
ACE2_ROOT="${ACE2_ROOT:-${SCRATCH_ROOT}/Snow-Predication-at-Sierra_ace2}"
ACE2_RUNTIME="${ACE2_RUNTIME:-${ACE2_ROOT}/runtime}"
ACE2_LOGS="${ACE2_LOGS:-${ACE2_ROOT}/logs}"

mkdir -p "${ACE2_RUNTIME}/tmp" "${ACE2_RUNTIME}/pip_cache" "${ACE2_RUNTIME}/mpl_cache" "${ACE2_RUNTIME}/xdg_cache" "${ACE2_LOGS}"

source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true
module load python
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build

python -m venv "${ACE2_RUNTIME}/venv_full"
source "${ACE2_RUNTIME}/venv_full/bin/activate"
unset PYTHONPATH

export PIP_CACHE_DIR="${ACE2_RUNTIME}/pip_cache"
export TMPDIR="${ACE2_RUNTIME}/tmp"
export MPLCONFIGDIR="${ACE2_RUNTIME}/mpl_cache"
export XDG_CACHE_HOME="${ACE2_RUNTIME}/xdg_cache"

python -m pip install --upgrade pip setuptools wheel >> "${ACE2_LOGS}/venv_setup.log" 2>&1
python -m pip install torch huggingface_hub fme==2026.5.1 >> "${ACE2_LOGS}/venv_setup.log" 2>&1

python -c "import sys, torch, fme, huggingface_hub; print(sys.executable); print(torch.__version__); print(fme.__file__); print(huggingface_hub.__version__)"
