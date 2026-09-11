#!/usr/bin/env bash
set -euo pipefail

repo_root=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
python_bin=/global/common/software/nersc9/pytorch/2.11.0/bin/python
cache_dir=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache
output_root=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1
audit_path="$output_root/historical_batch_order_audit.json"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_root/logs"
"$python_bin" "$repo_root/scripts/run_cmip6_cnn_experiment.py" \
    --architecture S0_static_cnn_swe_only \
    --predictor-cache-dir "$cache_dir" \
    --output-dir "$output_root/experiments/S0_static_cnn_swe_only" \
    --seed 20260813 \
    --batch-size 2 \
    --max-epochs 100 \
    --early-stopping-patience 15 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --num-workers 0 \
    --historical-batch-order-audit "$audit_path" \
    2>&1 | tee "$output_root/logs/S0_static_cnn_swe_only.log"
