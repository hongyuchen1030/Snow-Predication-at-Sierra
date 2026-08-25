# Codex Task State

Last updated: 2026-08-25 UTC

## Active Task

Reproduce the historical S0 no-auxiliary learning trajectory while reporting
training metrics from an end-of-epoch eval-mode pass. Do not overwrite the
existing "First Trial without CPM & AQM head" table in
`docs/Current_Status3_Simulations_Trained_ml.md`. Append a separately labeled
corrected-results table only after all requested results are verified.

## Active S0 Replay

- tmux session: `s0_historical_batch_replay`
- Slurm allocation: `57568228`
- output root: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1`
- model output: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only`
- live log: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/logs/S0_static_cnn_swe_only.log`

The run uses seed 20260813, batch size 2, max epochs 100, patience 15,
AdamW learning rate 1e-3, weight decay 1e-4, AMP enabled, the original
cache/split, and the original S0 architecture. At the time of this note it is
running. Check it with:

```bash
tmux capture-pane -pt s0_historical_batch_replay:0 -S -80
squeue -j 57568228 -o '%.18i %.2t %.10M %.6D %R'
```

## Batch Replay Contract

- persistent audit: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/historical_batch_order_audit.json`
- per-run verification: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/historical_s0_batch_order_verification.json`
- audit source initially existed only at `/tmp/cmip6_train_loader_order_audit.json` and was lost after reconnect; never use `/tmp` for handoff artifacts.

The recovered audit has exact historical S0 orders for epochs 1-3. Its original
SHA-256 digests are verified before training. Epochs 4+ are generated from the
same original global CPU DataLoader RNG path captured immediately after model
initialization. This distinction must be retained in any final report.

## Related Completed Artifacts

Corrected-metric S1-S3 retry summaries are complete at:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_evalmetrics_retry_v1/experiments/S1_static_latent_self_attention/metrics_summary.json`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_evalmetrics_retry_v1/experiments/S2_static_swe_token/metrics_summary.json`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_evalmetrics_retry_v1/experiments/S3_static_residual_gated_attention/metrics_summary.json`

The first corrected S0 v2 rerun is complete but did not preserve the historical
batch order and must not be used as the historical-trajectory reproduction.

## Required Completion Steps

1. Wait for the active S0 replay to write `history.csv` and `metrics_summary.json`.
2. Compare every epoch's S0 validation loss and R2 to the original historical
   S0 history at `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only/history.csv`.
3. Report the maximum absolute validation-loss and validation-R2 differences,
   selected best epoch, corrected eval-mode train metrics at the reproduced
   historical best epoch, and validation metrics at that epoch.
4. Append, do not overwrite, a labeled corrected-results table after the
   existing first-trial table.

## Files Added Or Changed For This Task

- `src/snow_ml/cmip6_cnn_experiment.py`
- `scripts/run_cmip6_cnn_experiment.py`
- `scripts/build_s0_historical_batch_order_audit.py`
- `scripts/run_s0_historical_batch_replay.sh`
- `scripts/launch_s0_historical_batch_replay_tmux.sh`
- `scripts/run_s0_s3_evalmetrics_v2.sh`
- `scripts/run_s1_s3_evalmetrics_retry.sh`
- `scripts/launch_s1_s3_evalmetrics_retry_tmux.sh`
