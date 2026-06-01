#!/bin/bash
set -e

cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra

echo "=========================================="
echo "COBE2 SST PCA Workflow"
echo "=========================================="

# Load environment
module purge
module load python
module load conda
module load climate-utils
conda activate uxarray_build

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Verify ncdump is available
echo "ncdump: $(which ncdump)"

# Create output directory
mkdir -p artifacts/sst_pca/cobe2

# Run the PCA script
echo "Starting COBE2 SST PCA analysis..."
PYTHONPATH=src:. python -u tools/compute_cobe2_sst_pca.py \
    --cobe2-sst-file /global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc \
    --lat-min 32.5 \
    --lat-max 43.0 \
    --lon-min -134.5 \
    --lon-max -114.0 \
    --n-components 5 \
    --output-dir artifacts/sst_pca/cobe2

echo "=========================================="
echo "PCA analysis complete!"
echo "=========================================="
