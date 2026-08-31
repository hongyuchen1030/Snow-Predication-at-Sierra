# Codex Task State

Last updated: 2026-08-25 UTC

## Completed Work

### Corrected S0-S3 Historical Replays

The S0-S3 no-auxiliary experiments were replayed with their historical
optimization batch trajectories while official training metrics were computed
from a separate post-epoch eval-mode training pass. The historical validation
loss and validation R2 trajectories were reproduced exactly for all four
architectures.

| Model | Historical/replay best epoch | Corrected train R2 | Replay validation R2 |
|---|---:|---:|---:|
| S0 | 27 | 0.550295 | 0.375452 |
| S1 | 6 | 0.230593 | 0.239554 |
| S2 | 4 | -0.007913 | -0.041406 |
| S3 | 7 | 0.198131 | 0.212341 |

Primary comparison report:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_historical_replay_verification_v1/historical_replay_summary.md`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_historical_replay_verification_v1/epoch_by_epoch_validation_comparison.csv`

Replay outputs:

- S0: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/`
- S1-S3: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/`

### S1 Attention Diagnostic and Ablations

The controlled S0/S1 diagnostic experiment is complete. It used model seed
`20260813`, a dedicated `torch.Generator` training-sampler seed `20260825`,
batch size 2, AdamW learning rate `1e-3`, weight decay `1e-4`, dropout 0.1 for
the S1 attention block, maximum 100 epochs, and patience 15. Training batch
order was identical across all variants for every shared epoch.

Completed variants:

| Variant | Best epoch | Best validation R2 |
|---|---:|---:|
| E: S0 | 23 | 0.308141 |
| A: attention only | 5 | 0.190230 |
| B: FFN only | 19 | 0.281486 |
| C: gated attention | 9 | 0.285705 |
| D: full S1 | 15 | 0.249003 |

Evidence summary:

- S1 adds exactly 2,224 trainable parameters over S0, a 0.201 percent
  increase; this is exactly the latent attention block.
- Full-S1 attention remains diffuse across every epoch. No row has attention
  maximum greater than 0.75 or 0.9; there is no attention-concentration or
  effective-rank-collapse evidence.
- The gated-attention model finishes with gate `0.004921` and substantially
  outperforms the un-gated attention-only path, supporting a mostly S0-like
  direct path under this single-seed controlled experiment.

Final report and plots:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s1_attention_diagnostics_v1/s1_attention_diagnostic_report.md`
- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s1_attention_diagnostics_v1/`

Raw diagnostic artifacts and checkpoints:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/s1_attention_diagnostics_v1/parameter_counts.csv`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/s1_attention_diagnostics_v1/ablation_history.csv`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/s1_attention_diagnostics_v1/s1_latent_diagnostics.csv`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/s1_attention_diagnostics_v1/s1_attention_diagnostics.csv`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/s1_attention_diagnostics_v1/checkpoints/`

## Chat-Reconnect Safeguard

Do not rely on a visible Codex chat transcript as the sole task record. Keep
durable task state, commands, results, and artifact paths in this file and in
committed scripts. Never use `/tmp` for handoff-only artifacts; temporary files
were lost after a reconnect during the original replay work.

There is no active Slurm allocation or tmux session for the completed replay or
diagnostic experiments.

## Storage Contract

All durable task outputs must remain inside the repository. Do not create
project output folders directly under `/global/homes/h/hyvchen/`.

- Documentation and summaries: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/docs/`
- Source and runnable scripts: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/scripts/`
- Plots, reports, CSVs, and other durable experiment artifacts:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/<experiment>/`
- Large live training outputs and checkpoints: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/<experiment>/`

The misplaced home-directory result folders were moved on 2026-08-25 into
`artifacts/cmip6_historical_replay_trajectory_plots_v1/`,
`artifacts/s1_attention_diagnostics_v1/`, and
`artifacts/frozen_s0_z_attention_test_v1/`.

## Relevant Commits

- `a8f794e` Generalize historical batch replay for S1 to S3
- `e9bca01` Support historical validation metric columns in replay report
- `4eeedd6` Add corrected replay trajectory analysis
- `0440aae` Add S1 attention diagnostic ablations
- `cf3d981` Add partial S1 ablation report generator

## Frozen S0-Z Attention Mechanism Test

Scientific question: determine whether S1 attention itself degrades an
already-good frozen S0 latent representation, or whether the end-to-end S1
failure primarily arises from harmful gradients into the CNN encoder.

- completed allocation: `57612981` on `nid001109`
- exact frozen S0 checkpoint:
  `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt`
- checkpoint provenance: S0 historical-replay best epoch 27; validation loss
  `0.40094377089132505`
- raw output root:
  `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1/`
- final plot/report root:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/frozen_s0_z_attention_test_v1/`
- live log:
  `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1/logs/frozen_s0_z_attention_test.log`

The experiment has two branches:

- `F0_frozen_S0_Z_MLP`: exact frozen S0 `backbone + project`, then a freshly
  initialized ordinary S0-shaped MLP head.
- `F1_frozen_S0_Z_attention`: the exact same frozen S0 `backbone + project`,
  then freshly initialized S1 `LatentSelfAttentionBlock + MLP` head.

Both use model seed `20260813`, dedicated training-sampler seed `20260826`,
batch size 2, AdamW learning rate `1e-3`, weight decay `1e-4`, maximum 100
epochs, and patience 15. The frozen encoder has 1,097,856 parameters
(`backbone + project`), no dropout or BatchNorm, `requires_grad=False` for
every parameter, and is forced to eval mode during both downstream training
and evaluation. Each branch records encoder hashes and a fixed-batch latent
before/after comparison; expected maximum changes are zero.

Final best-validation metrics:

| Variant | Best epoch | Eval-mode train R2 | Validation R2 | Train-minus-validation R2 |
|---|---:|---:|---:|---:|
| F0 frozen S0 Z + MLP | 20 | 0.600153 | 0.397116 | 0.203037 |
| F1 frozen S0 Z + attention + MLP | 4 | 0.596605 | 0.374623 | 0.221982 |

The frozen-encoder checks passed for both branches: maximum encoder parameter
change and maximum fixed-batch latent-Z change are both zero. F1's best
validation R2 is 0.022493 below F0's. The tmux session exited and the
allocation was relinquished after the outputs were written.

Expected final artifacts:

- `frozen_s0_z_metrics.csv`
- `f1_attention_diagnostics.csv`
- `f1_latent_diagnostics.csv`
- `comparison_summary.csv`
- `run_config.json` and `run_metadata.json`
- `checkpoints/F0_frozen_S0_Z_MLP/best_trainable_downstream.pt`
- `checkpoints/F1_frozen_S0_Z_attention/best_trainable_downstream.pt`
- R2, loss, and F1 attention plots plus `README.md` in the artifact directory

## Active Work

### TaiESM1 Daily 12-Variable One-Day Predictor Dataset

- tmux session: `taiesm1_daily_12var_1day_v2`
- interactive Slurm allocation: `57797382` on `nid001093` (project `m2637`, one GPU, four-hour interactive QoS)
- tmux status: active; CDO has generated weights and is processing historical `rlut` source years 1980-1989, with 1990-1999 next
- launcher: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/scripts/launch_taiesm1_daily_12var_1day_predictors_tmux.sh`
- builder: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/scripts/build_taiesm1_daily_12var_1day_predictors.py`
- environment: `climate-utils/2025.01` for CDO, plus the explicit `uxarray_build` Python executable
- output root: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1/`
- durable plots/reports/CSVs, if generated: repository `artifacts/<experiment>/`, never directly under `/global/homes/h/hyvchen/`

The build is data preparation only: TaiESM1 `r1i1p1f1`, historical and ssp370,
12 specified daily channels, exact pressure-level selection, CDO bilinear
regridding to the 1.5-degree global grid, physical values without normalization,
and resumable per-channel/year outputs. No model training or random split is
part of this task.
