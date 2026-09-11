#!/bin/bash
#SBATCH -J kgae_more_latents
#SBATCH -A m2637
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 08:00:00

set -euo pipefail

PROJECT_ROOT=/global/homes/h/hyvchen/Snow-Predication-at-Sierra
HEAVY_OUT=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_more_latents_swe_loyo
HOME_OUT=${PROJECT_ROOT}/artifacts/kgae_more_latents_swe_loyo
PYTHON=/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
LOG_DIR=${HEAVY_OUT}/logs

mkdir -p "${HEAVY_OUT}" "${LOG_DIR}" "${HOME_OUT}"

{
  echo "# Run Status"
  echo
  echo "- Smoke test passed: \`True\`"
  echo "- Slurm job id: \`${SLURM_JOB_ID}\`"
  echo "- Node list: \`${SLURM_JOB_NODELIST}\`"
  echo "- Heavy output directory: \`${HEAVY_OUT}\`"
  echo "- Log directory: \`${LOG_DIR}\`"
  echo "- Expected main result files:"
  echo "  - \`kgae_more_latents_training_summary.csv\`"
  echo "  - \`kgae_more_latents_swe_ridge_loyo_metrics.csv\`"
  echo "  - \`kgae_more_latents_best_alignment_summary.csv\`"
  echo "  - \`REPORT.md\`"
  echo "- Monitor with:"
  echo "  - \`squeue -j ${SLURM_JOB_ID}\`"
  echo "  - \`tail -f ${LOG_DIR}/kgae_more_latents_${SLURM_JOB_ID}.log\`"
} > "${HOME_OUT}/RUN_STATUS.md"

echo "${SLURM_JOB_ID}" > "${HEAVY_OUT}/slurm_job_id.txt"
ln -sfn "${HEAVY_OUT}/slurm_job_id.txt" "${HOME_OUT}/slurm_job_id.txt"

cd "${PROJECT_ROOT}"

export MPLCONFIGDIR=/tmp/mpl_kgae_more_latents_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_kgae_more_latents_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

exec "${PYTHON}" scripts/run_kgae_more_latents_swe_loyo.py \
  --mode full \
  --pscratch-output "${HEAVY_OUT}" \
  --home-output "${HOME_OUT}" \
  --latent-dims 5 10 \
  --epochs 120 \
  --device cpu
