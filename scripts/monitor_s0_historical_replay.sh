#!/usr/bin/env bash
set -euo pipefail

job_id=57568228
output_dir=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1
log_path="$output_dir/status_20min.log"
experiment_dir="$output_dir/experiments/S0_static_cnn_swe_only"

while true; do
    date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$log_path"
    squeue -j "$job_id" -o '%.18i %.2t %.10M %.6D %R' >> "$log_path" 2>&1 || true
    if test -f "$experiment_dir/metrics_summary.json"; then
        rg '"best_epoch"|"best_val_total_loss"|"swe_r2"' "$experiment_dir/metrics_summary.json" >> "$log_path"
        exit 0
    fi
    tail -n 2 "$output_dir/logs/S0_static_cnn_swe_only.log" >> "$log_path" 2>&1 || true
    sleep 1200
done
