#!/usr/bin/env bash
set -euo pipefail

PROJECT=/global/homes/h/hyvchen/Snow-Predication-at-Sierra
OUT=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1
LOG="$OUT/logs/build.log"
mkdir -p "$(dirname "$LOG")"

source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh || true
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build
module load climate-utils
python "$PROJECT/scripts/build_taiesm1_daily_12var_1day_predictors.py" --output-root "$OUT" 2>&1 | tee -a "$LOG"
