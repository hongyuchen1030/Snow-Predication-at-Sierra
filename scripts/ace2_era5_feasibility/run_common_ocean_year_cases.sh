#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <year>" >&2
  exit 2
fi

YEAR="$1"
REPO="${REPO:-/global/homes/h/hyvchen/Snow-Predication-at-Sierra}"
REPORT_DIR="${REPORT_DIR:-${REPO}/artifacts/ace2_era5_feasibility}"

for STATE in actual neutral reversed; do
  CONFIG_PATH="${REPORT_DIR}/configs/${YEAR}_pacific_common_ocean_${STATE}_season.yaml"
  OUTPUT_DIR="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/outputs/pacific_anomaly_reversal_${YEAR}_common_ocean/${STATE}_season"
  SUMMARY_JSON="${REPORT_DIR}/${YEAR}_pacific_common_ocean_${STATE}_season_runtime_summary.json"
  SUMMARY_TXT="${REPORT_DIR}/${YEAR}_pacific_common_ocean_${STATE}_season_runtime_summary.txt"
  "${REPO}/scripts/ace2_era5_feasibility/run_ace2_inference_with_summary.sh" "${CONFIG_PATH}" "${OUTPUT_DIR}" "${SUMMARY_JSON}" "${SUMMARY_TXT}"
done
