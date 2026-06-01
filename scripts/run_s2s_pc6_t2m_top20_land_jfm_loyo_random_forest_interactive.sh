#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/results/s2s_pc6_t2m_top20_land_jfm_loyo_random_forest"

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-missing}"

source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true
module purge
module load python
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uxarray_build

cd "$REPO_ROOT"

python -c "import numpy, pandas, xarray, sklearn, matplotlib; print('packages OK')"

mkdir -p "$OUTPUT_DIR"

PYTHONPATH=src:. python -u scripts/run_s2s_pc6_t2m_top20_land_jfm_loyo_random_forest.py \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/run_log.txt"
