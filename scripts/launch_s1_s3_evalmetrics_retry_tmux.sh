#!/usr/bin/env bash
set -euo pipefail

repo_root=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
output_root=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_evalmetrics_retry_v1

mkdir -p "$output_root"
exec salloc -N 1 -C gpu -q interactive -A m2637 -t 04:00:00 \
    /usr/bin/bash "$repo_root/scripts/run_s1_s3_evalmetrics_retry.sh" \
    2>&1 | tee "$output_root/allocation.log"
