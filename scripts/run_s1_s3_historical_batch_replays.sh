#!/usr/bin/env bash
set -euo pipefail

repo_root=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
python_bin=/global/common/software/nersc9/pytorch/2.11.0/bin/python
cache_dir=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache
output_root=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1
audit_path="$output_root/historical_batch_order_audit.json"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_root/logs"
for architecture in S1_static_latent_self_attention S2_static_swe_token S3_static_residual_gated_attention; do
    "$python_bin" "$repo_root/scripts/run_cmip6_cnn_experiment.py" \
        --architecture "$architecture" \
        --predictor-cache-dir "$cache_dir" \
        --output-dir "$output_root/experiments/$architecture" \
        --seed 20260813 \
        --batch-size 2 \
        --max-epochs 100 \
        --early-stopping-patience 15 \
        --learning-rate 1e-3 \
        --weight-decay 1e-4 \
        --num-workers 0 \
        --historical-batch-order-audit "$audit_path" \
        2>&1 | tee "$output_root/logs/$architecture.log"
done
