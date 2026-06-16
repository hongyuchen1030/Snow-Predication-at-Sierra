#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cobe2_lod_loyo_mpl}"
export PYTHONPATH="${PYTHONPATH:-src:.}"

exec /global/homes/h/hyvchen/.conda/envs/uxarray_build/bin/python -u scripts/run_cobe2_sierra_swe_lod_loyo.py
