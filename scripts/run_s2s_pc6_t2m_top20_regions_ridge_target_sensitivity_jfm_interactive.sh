#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$REPO_ROOT/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
LABEL_DIR="$ARTIFACT_DIR/top20_region_labels"
OUTPUT_DIR="$REPO_ROOT/artifacts/s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm"
DEBUG_OUTPUT_DIR="$OUTPUT_DIR/trainall_debug"

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-missing}"

source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true
module purge
module load python
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build

cd "$REPO_ROOT"

python -c "import numpy, pandas, xarray, sklearn, matplotlib, scipy; print('packages OK')"

mkdir -p "$OUTPUT_DIR" "$DEBUG_OUTPUT_DIR"

PYTHONPATH=src:. python scripts/run_s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --label-dir "$LABEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --input-months Jun Jul Aug Sep Oct Nov \
  --target-months Jan Feb Mar 2>&1 | tee "$OUTPUT_DIR/run_log.txt"

PYTHONPATH=src:. python scripts/run_s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --label-dir "$LABEL_DIR" \
  --output-dir "$DEBUG_OUTPUT_DIR" \
  --input-months Jun Jul Aug Sep Oct Nov \
  --target-months Jan Feb Mar \
  --train-all-debug 2>&1 | tee "$DEBUG_OUTPUT_DIR/run_log.txt"
