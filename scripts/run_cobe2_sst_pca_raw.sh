#!/bin/bash

###############################################################################
# COBE2 SST PCA Raw Analysis Script (No Mean Removal)
# Following Paul's suggestion: Run PCA directly on raw SST field
###############################################################################

set -e

# Change to project root
cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra

echo "=========================================="
echo "COBE2 Raw SST PCA Workflow"
echo "=========================================="
echo "Purpose: Run PCA directly on raw COBE2 SST (no anomalies, no mean removal)"
echo "Expected: First EOF should represent mean temperature pattern"
echo "=========================================="

# Print compute environment
echo ""
echo "Compute Environment:"
hostname
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
python -c "import xarray, numpy, sklearn; print('✓ Dependencies OK: xarray, numpy, sklearn')" || exit 1

# Create output directory
mkdir -p artifacts/sst_pca/cobe2_raw_no_mean

# Run the raw PCA script with verification
echo ""
echo "=========================================="
echo "Starting COBE2 Raw SST PCA analysis..."
echo "=========================================="
PYTHONPATH=src:. python -u tools/compute_cobe2_sst_pca_raw.py \
    --cobe2-sst-file /global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc \
    --lat-min 32.5 \
    --lat-max 43.0 \
    --lon-min -134.5 \
    --lon-max -114.0 \
    --n-components 5 \
    --output-dir artifacts/sst_pca/cobe2_raw_no_mean

echo ""
echo "=========================================="
echo "PCA analysis complete!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh artifacts/sst_pca/cobe2_raw_no_mean/
echo ""
echo "Report:"
cat artifacts/sst_pca/cobe2_raw_no_mean/cobe2_raw_sst_pca_report.md
