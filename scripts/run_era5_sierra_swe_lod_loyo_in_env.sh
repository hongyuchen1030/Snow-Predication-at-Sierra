#!/usr/bin/env bash
set -euo pipefail

source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true
module load python
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build
python scripts/run_era5_sierra_swe_lod_loyo.py
python scripts/plot_era5_lod_loyo_mode1_mode2_latepooled_selection_frequency.py
