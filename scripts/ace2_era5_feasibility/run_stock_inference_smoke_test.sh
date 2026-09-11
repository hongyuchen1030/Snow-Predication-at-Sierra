#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <config-yaml>" >&2
  exit 2
fi

CONFIG_PATH="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ACE2_PYTHON="${ACE2_PYTHON:-}"
ACE2_PYTHONPATH="${ACE2_PYTHONPATH:-}"
export REPO="${REPO:-${ROOT_DIR}}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-${PSCRATCH:-/pscratch/sd/h/hyvchen}}"
export ACE2_ROOT="${ACE2_ROOT:-${SCRATCH_ROOT}/Snow-Predication-at-Sierra_ace2}"
export ACE2_ASSETS="${ACE2_ASSETS:-${ACE2_ROOT}/assets}"
export ACE2_RUNTIME="${ACE2_RUNTIME:-${ACE2_ROOT}/runtime}"
export ACE2_OUTPUTS="${ACE2_OUTPUTS:-${ACE2_ROOT}/outputs}"
export ACE2_LOGS="${ACE2_LOGS:-${ACE2_ROOT}/logs}"
mkdir -p "${ACE2_ASSETS}" "${ACE2_RUNTIME}" "${ACE2_OUTPUTS}" "${ACE2_LOGS}"

if [[ -n "${ACE2_PYTHONPATH}" ]]; then
  export PYTHONPATH="${ACE2_PYTHONPATH}:${PYTHONPATH:-}"
fi

if [[ -n "${ACE2_PYTHON}" ]]; then
  "${ACE2_PYTHON}" -m fme.ace.validate_config "${CONFIG_PATH}" --config_type inference
  "${ACE2_PYTHON}" -m fme.ace.inference "${CONFIG_PATH}"
  exit 0
fi

if [[ -f ".venv_ace2/bin/activate" ]]; then
  # Fall back to a repo-local environment if one is available.
  source .venv_ace2/bin/activate
  python -m fme.ace.validate_config "${CONFIG_PATH}" --config_type inference
  python -m fme.ace.inference "${CONFIG_PATH}"
  exit 0
fi

echo "missing ACE2 runtime: set ACE2_PYTHON or create .venv_ace2" >&2
exit 1
