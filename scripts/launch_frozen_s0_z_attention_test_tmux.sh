#!/usr/bin/env bash
set -euo pipefail

repo_root=/global/u1/h/hyvchen/Snow-Predication-at-Sierra
python_bin=/global/common/software/nersc9/pytorch/2.11.0/bin/python
cache_dir=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache
split_json=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only/split.json
s0_checkpoint=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt
output_root=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1
plot_dir=/global/homes/h/hyvchen/frozen_s0_z_attention_test_v1

mkdir -p "$output_root/logs" "$plot_dir"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$output_root/matplotlib_cache"
exec salloc -N 1 -C gpu -q interactive -A m2637 -t 04:00:00 \
    "$python_bin" "$repo_root/scripts/run_frozen_s0_z_attention_test.py" \
    --cache-dir "$cache_dir" \
    --split-json "$split_json" \
    --s0-checkpoint "$s0_checkpoint" \
    --output-root "$output_root" \
    --plot-dir "$plot_dir" \
    2>&1 | tee "$output_root/logs/frozen_s0_z_attention_test.log"
