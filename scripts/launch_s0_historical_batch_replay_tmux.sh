#!/usr/bin/env bash
set -euo pipefail

repo_root=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
output_root=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1
python_bin=/global/common/software/nersc9/pytorch/2.11.0/bin/python
split_json=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only/split.json

mkdir -p "$output_root"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" \
    "$repo_root/scripts/build_s0_historical_batch_order_audit.py" \
    --split-json "$split_json" \
    --output "$output_root/historical_batch_order_audit.json"
exec salloc -N 1 -C gpu -q interactive -A m2637 -t 04:00:00 \
    /usr/bin/bash "$repo_root/scripts/run_s0_historical_batch_replay.sh" \
    2>&1 | tee "$output_root/allocation.log"
