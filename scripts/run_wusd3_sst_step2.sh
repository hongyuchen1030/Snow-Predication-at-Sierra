#!/bin/bash

###############################################################################
# WUS-D3 SST Processing Step 2 Script
# Process WUS-D3 models for SST anomaly computation
###############################################################################

set -e

# Change to project root
cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra

echo "=========================================="
echo "WUS-D3 SST Processing Step 2"
echo "=========================================="
echo "Purpose: Process tskin from WUS-D3 models for SST anomaly computation"
echo "Ocean masking: Uses COBE2 SST mean field (isfinite = ocean)"
echo "=========================================="

# Verify we're on a compute node
echo ""
echo "Compute Environment:"
hostname
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: Not running on a compute node (no SLURM_JOB_ID)"
    echo "Please submit this script to SLURM:"
    echo "sbatch scripts/run_wusd3_sst_step2.sh"
    exit 1
fi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
squeue -u $USER
echo ""

# Load environment
echo "Loading environment modules..."
module purge
module load python
module load conda
module load climate-utils

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Verify tools
echo ""
echo "Verifying tools..."
which ncdump
python -c "import xarray, numpy, scipy, cartopy, pyproj; print('✓ Dependencies OK: xarray, numpy, scipy, cartopy, pyproj')" || exit 1

# Create output directory
mkdir -p artifacts/sst_pca/model_sst_anomalies

# Run the WUS-D3 processing script
echo ""
echo "=========================================="
echo "Starting WUS-D3 SST processing..."
echo "=========================================="
PYTHONPATH=src:. python -u tools/process_wusd3_sst_step2.py

echo ""
echo "=========================================="
echo "WUS-D3 SST processing complete!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh artifacts/sst_pca/model_sst_anomalies/
echo ""
echo "Summary report:"
cat artifacts/sst_pca/model_sst_anomalies/model_sst_step2_summary.md