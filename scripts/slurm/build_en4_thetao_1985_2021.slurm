#!/bin/bash
#SBATCH --job-name=en4_thetao_37wy
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/en4_thetao_stage/full_1984_2021/build_en4_thetao_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/en4_thetao_stage/full_1984_2021/build_en4_thetao_%j.err

set -euo pipefail

REPO_DIR=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
STAGE_DIR=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/en4_thetao_stage/full_1984_2021
OUTPUT_DIR=${STAGE_DIR}/processed_cnn_wy1985_2021
PYTHON_BIN=/global/u1/h/hyvchen/.conda/envs/uxarray_build/bin/python

cd "${REPO_DIR}"
"${PYTHON_BIN}" scripts/stage_en4_thetao_obs_audit.py build-en4-thetao \
  --stage-dir "${STAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --start-month 1984-09 \
  --end-month 2021-03 \
  --build-cnn-rows \
  --start-row-year 1984 \
  --end-row-year 2020
