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

- `/global/homes/h/hyvchen/s1_attention_diagnostics_v1/s1_attention_diagnostic_report.md`
- `/global/homes/h/hyvchen/s1_attention_diagnostics_v1/`

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

## Relevant Commits

- `a8f794e` Generalize historical batch replay for S1 to S3
- `e9bca01` Support historical validation metric columns in replay report
- `4eeedd6` Add corrected replay trajectory analysis
- `0440aae` Add S1 attention diagnostic ablations
- `cf3d981` Add partial S1 ablation report generator
