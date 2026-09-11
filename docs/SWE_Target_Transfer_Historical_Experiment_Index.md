# SWE Target Transfer Historical Experiment Index

This index was reconstructed from saved artifacts, reports, metric tables, and the original helper scripts under this repository. No retraining was performed. No existing artifact was modified.

Two memory corrections from the saved evidence:

- The raw model-vs-observation pattern diagnostic was a **CMIP6/WUS-D3** comparison, not CMIP5.
- The completed normalized-target transfer artifact is a **parent-model standardized anomaly** experiment. I did **not** find a completed saved artifact using a percent-of-maximum target or an explicitly Xu Han-labeled normalization.

# 1. Absolute SWE transfer

## Experiment purpose

Apply the already-trained simulation models directly to the observational predictor tensor and evaluate April 1 Sierra SWE in **physical mm**, with no observational refit or observational normalization fitting.

## Exact target definition

- Observational target source: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc`
- Field actually used for metrics: `sierra_swe_apr1_mean_mm`
- Units: `mm`
- Region: `DEFAULT_SIERRA_REGION` (`lat 35-42`, `lon -122.5..-118.0`)
- Aggregation: area-weighted Sierra mean of UCLA `SWE_Post`, Stats index `0`
- Observational evaluation period: `WY1985-WY2021` (`37` water years)

## Artifact and report paths

- Main artifact directory: `artifacts/s0_attention_observational_transfer_v1/`
- Main report: `artifacts/s0_attention_observational_transfer_v1/README.md`
- Machine-readable report: `artifacts/s0_attention_observational_transfer_v1/full_report.json`

## Checkpoints used

- `S0`: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt` (`epoch 27`)
- `frozen_s0_attention`: encoder `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt` (`epoch 27`) plus head `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1/checkpoints/F1_frozen_S0_Z_attention/best_trainable_downstream.pt` (`epoch 4`)
- `end_to_end_attention`: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/S1_static_latent_self_attention/best_checkpoint.pt` (`epoch 6`)

## Plot paths

- `artifacts/s0_attention_observational_transfer_v1/observational_timeseries.png`
- `artifacts/s0_attention_observational_transfer_v1/observational_scatter.png`
- `artifacts/s0_attention_observational_transfer_v1/simulation_vs_observational_r2.png`

## Headline simulation metrics

| Model | Simulation validation R² |
|---|---:|
| S0 | 0.375452 |
| frozen S0 + attention | 0.374623 |
| end-to-end attention | 0.239554 |

## Headline observational metrics

| Model | Obs R² | Obs Pearson r | Obs RMSE (mm) | Obs MAE (mm) |
|---|---:|---:|---:|---:|
| S0 | -0.638994 | -0.196358 | 40.597768 | 35.133821 |
| frozen S0 + attention | -1.394847 | -0.198648 | 49.074108 | 41.756416 |
| end-to-end attention | -0.152455 | 0.046634 | 34.042822 | 29.296902 |

## One-paragraph conclusion

The saved report is explicit that all three simulation-trained models failed to transfer to real observations on the absolute-mm target: every observational R² was negative, and the simulation ranking did not carry over. The least-bad observational R² came from end-to-end attention, but the report attributes that to near-constant predictions clustered near the simulation climatology rather than meaningful recovered signal. The artifact also records direct evidence of predictor-space distribution shift after simulation-derived normalization and carries forward a pre-existing WUS-D3-versus-UCLA footprint mismatch caveat.

# 2. Normalized/anomaly SWE transfer

## Experiment purpose

Replace the raw April 1 SWE target with a standardized anomaly target so the model predicts relative departures from each domain's own climatology rather than absolute magnitude in mm.

## Exact target definition

Recovered from `scripts/build_parent_model_standardized_swe_target.py` and `artifacts/s0_attention_swe_std_anomaly_v1/target_normalization.csv`:

- Simulation target formula: `y[m,t] = (SWE[m,t] - mu_m) / sigma_m`
- `mu_m` and `sigma_m` were computed from the **training-split samples only** of each CMIP parent model `m`
- Parent-model normalization was used separately for:
  - `EC-Earth3:r102i1p1f1`
  - `MIROC6:r1i1p1f1`
  - `MPI-ESM1-2-HR:r3i1p1f1`
  - `TaiESM1:r1i1p1f1`
- Observational target formula in transfer evaluation: `y_obs = (SWE_obs - mu_obs) / sigma_obs`
- `mu_obs` and `sigma_obs` came from all `37` UCLA water years (`WY1985-WY2021`)
- This was **not** a percent-of-maximum target and I did not find a completed saved Xu Han-branded variant

Per-model saved normalization statistics:

| Domain/model | mu SWE (mm) | sigma SWE (mm) |
|---|---:|---:|
| EC-Earth3:r102i1p1f1 | 46.638863 | 35.364595 |
| MIROC6:r1i1p1f1 | 49.601309 | 37.469936 |
| MPI-ESM1-2-HR:r3i1p1f1 | 95.729990 | 46.662862 |
| TaiESM1:r1i1p1f1 | 53.110272 | 44.850861 |
| UCLA observational | 55.031064 | 31.711260 |

## Artifact and report paths

- Main artifact directory: `artifacts/s0_attention_swe_std_anomaly_v1/`
- Main machine-readable report: `artifacts/s0_attention_swe_std_anomaly_v1/full_report.json`
- Saved metric table: `artifacts/s0_attention_swe_std_anomaly_v1/final_metrics.csv`
- No saved README was found in this artifact directory during the targeted search

## Models and checkpoints

- `S0`: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_std_anomaly_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt` (`epoch 9`)
- `attention`: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_std_anomaly_v1/experiments/S1_static_latent_self_attention/best_checkpoint.pt` (`epoch 7`)

## Plot paths

- `artifacts/s0_attention_swe_std_anomaly_v1/r2_vs_epoch_S0.png`
- `artifacts/s0_attention_swe_std_anomaly_v1/r2_vs_epoch_attention.png`
- `artifacts/s0_attention_swe_std_anomaly_v1/observed_vs_predicted_anomaly_timeseries.png`
- `artifacts/s0_attention_swe_std_anomaly_v1/observed_vs_predicted_anomaly_scatter.png`
- `artifacts/s0_attention_swe_std_anomaly_v1/simulation_vs_observational_r2.png`

## Headline simulation metrics

| Model | Simulation validation R² |
|---|---:|
| S0 | 0.226095 |
| attention | 0.122217 |

## Headline observational metrics

| Model | Obs R² | Obs Pearson r | Obs RMSE (standardized anomaly) | Obs MAE (standardized anomaly) |
|---|---:|---:|---:|---:|
| S0 | 0.021850 | 0.148774 | 0.989015 | 0.809411 |
| attention | 0.007455 | 0.098831 | 0.996266 | 0.811866 |

## Multiple-normalization note

Within the targeted search scope, this is the only completed saved CNN transfer artifact I found for normalized/anomaly SWE. I did not find a separate completed saved transfer artifact using percent-of-maximum scaling or a different named anomaly formula.

## One-paragraph conclusion

The completed normalized-target transfer experiment used parent-model standardization on the simulation side and UCLA's own 37-year standardization on the observational side. That change was enough to lift observational R² slightly above zero for both saved models, with S0 best, but the resulting transfer skill remained weak: Pearson r stayed below `0.15`, and the attention model trailed S0 on both simulation and observational R². The saved evidence supports “standardized anomaly by parent model,” not a percent-max or Xu Han-labeled normalization.

# 3. SWE percentile transfer

## Experiment purpose

Replace absolute SWE magnitude with a percentile target so the evaluation asks whether the model recovers the **relative wet/dry ordering** of real Sierra water years.

## Exact target definition

Recovered from `artifacts/s0_attention_swe_percentile_v1/percentile_target_definition.md`:

- For each CMIP parent model `m`, fit an empirical CDF to that model's April 1 Sierra SWE using **training-split samples only**
- Plotting-position convention: `p_i = (rank_i - 0.5) / N`
- Both train and validation samples of model `m` are mapped to percentiles by `numpy.interp` against that training-fitted CDF
- Percentiles are clipped to `[0.5/N, (N-0.5)/N]`, so no value is exactly `0` or `1`
- UCLA observational percentiles use the same plotting-position convention over all `37` observed years as one fixed reference distribution
- Yes, the empirical CDF was built **separately by parent model**

Parent-model percentile metadata:

| Domain/model | n train | n val | raw SWE train min (mm) | raw SWE train max (mm) |
|---|---:|---:|---:|---:|
| EC-Earth3:r102i1p1f1 | 96 | 24 | 0.052657 | 194.705215 |
| MIROC6:r1i1p1f1 | 96 | 24 | 0.655892 | 179.783875 |
| MPI-ESM1-2-HR:r3i1p1f1 | 31 | 3 | 19.827530 | 190.556168 |
| TaiESM1:r1i1p1f1 | 95 | 24 | 0.575544 | 232.367523 |
| UCLA observational | 0 | 37 | 3.664957 | 136.373520 |

## Artifact and report paths

- Main artifact directory: `artifacts/s0_attention_swe_percentile_v1/`
- Main report: `artifacts/s0_attention_swe_percentile_v1/README.md`
- Machine-readable report: `artifacts/s0_attention_swe_percentile_v1/full_report.json`
- Target-definition note: `artifacts/s0_attention_swe_percentile_v1/percentile_target_definition.md`

## Models and checkpoints

- `S0`: `cmip6_cnn_swe_percentile_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt` (`epoch 15`)
- `Frozen-S0 + attention`: `cmip6_cnn_swe_percentile_v1/experiments/frozen_S0_attention/best_checkpoint.pt` (`epoch 39`)
- `End-to-end attention`: `cmip6_cnn_swe_percentile_v1/experiments/S1_static_latent_self_attention/best_checkpoint.pt` (`epoch 21`)

## Plot paths

- `artifacts/s0_attention_swe_percentile_v1/main_observational_percentile_timeseries.png`
- `artifacts/s0_attention_swe_percentile_v1/shape_only_observational_comparison.png`
- `artifacts/s0_attention_swe_percentile_v1/observational_scatter.png`
- `artifacts/s0_attention_swe_percentile_v1/observational_skill_metrics.png`
- `artifacts/s0_attention_swe_percentile_v1/simulation_vs_observational_correlation.png`

## Headline simulation metrics

| Model | Sim val Pearson r | Sim val Spearman rho | Sim val wet/dry acc |
|---|---:|---:|---:|
| S0 | 0.499374 | 0.534243 | 0.640000 |
| Frozen-S0 + attention | 0.525263 | 0.535220 | 0.680000 |
| End-to-end attention | 0.593321 | 0.567881 | 0.760000 |

## Headline observational metrics

| Model | Obs Pearson r | Obs Spearman rho | Obs wet/dry acc | Obs tercile acc | Obs R² |
|---|---:|---:|---:|---:|---:|
| S0 | 0.152697 | 0.175913 | 0.486486 | 0.351351 | -0.001659 |
| Frozen-S0 + attention | 0.159124 | 0.193457 | 0.621622 | 0.351351 | 0.019171 |
| End-to-end attention | 0.010814 | -0.004979 | 0.594595 | 0.405405 | -0.174110 |

## One-paragraph conclusion

This saved experiment is the clearest completed transfer success among the three target choices. The historical artifact concludes that frozen-S0 plus attention transfers best on the percentile target, with the best observational Pearson r, Spearman rho, wet/dry accuracy, and the only positive observational R², while end-to-end attention shows the strongest simulation metrics but collapses on the observational transfer task. The “shape-only” visualization exists and is explicitly marked diagnostic-only, not used for metrics.

# 4. Raw CMIP-vs-observation SWE pattern diagnostic

## Experiment purpose

Check whether raw, non-scaled CMIP/WUS-D3 April 1 Sierra SWE itself tracks the same year-to-year pattern as UCLA observations before any percentile or anomaly transformation.

## Historical-record correction

The saved artifact evidence points to **CMIP6 parent-model SWE extracted from WUS-D3**, not CMIP5. The key raw overlay plot was produced by `scripts/plot_ucla_vs_cmip_swe_timeseries.py`, whose docstring says: `UCLA-observed vs. CMIP-simulated (WUS-D3) April-1 Sierra SWE, WY1985-2021`.

## Exact target/comparison definition

- UCLA series: raw April 1 Sierra SWE mean in `mm`, `WY1985-WY2021`
- CMIP series: raw April 1 Sierra SWE mean in `mm`, extracted on the same UCLA Sierra region/mask logic but on the WUS-D3 `d02` native grid
- Models plotted: `EC-Earth3`, `MIROC6`, `MPI-ESM1-2-HR`, `TaiESM1`
- Years plotted: `WY1985-WY2021`
- Comparison CSV columns: UCLA plus one raw-SWE trajectory for each of the four CMIP6 parent models

## Artifact and report paths

- Key plot/CSV artifact directory: `artifacts/s0_attention_observational_transfer_v1/`
- Key CSV: `artifacts/s0_attention_observational_transfer_v1/ucla_vs_cmip_swe_apr1_timeseries.csv`
- Key plot: `artifacts/s0_attention_observational_transfer_v1/ucla_vs_cmip_swe_apr1_timeseries.png`
- Supporting scatter/metric files:
  - `artifacts/s0_attention_observational_transfer_v1/ucla_vs_cmip_scatter.png`
  - `artifacts/s0_attention_observational_transfer_v1/ucla_vs_cmip_r2_metrics.csv`
  - `artifacts/s0_attention_observational_transfer_v1/lineplot_observed_vs_all_six_models.png`
  - `artifacts/s0_attention_observational_transfer_v1/lineplot_observed_vs_S0_S1_S2_S3.png`
- Related follow-on write-up focused on raw-distribution comparability: `artifacts/swe_sim_vs_ucla_distribution_v1/README.md`

## Headline metrics from the raw UCLA-vs-CMIP comparison

| Model | R² | Pearson r | RMSE (mm) | MAE (mm) |
|---|---:|---:|---:|---:|
| EC-Earth3 | -1.106657 | 0.150444 | 46.026762 | 36.523104 |
| MIROC6 | -2.283042 | -0.235334 | 57.458162 | 46.619850 |
| MPI-ESM1-2-HR | -3.869851 | -0.190639 | 69.979580 | 57.948457 |
| TaiESM1 | -2.295631 | -0.021199 | 57.568223 | 41.875717 |

## Scientific conclusion written at the time

The saved raw-distribution report (`artifacts/swe_sim_vs_ucla_distribution_v1/README.md`) says the simulation and UCLA SWE distributions are broadly similar in central tendency and overall range, but the simulation-training distribution is more dispersed. The raw year-to-year pattern comparison itself is weak: all four individual CMIP6 parent-model trajectories have negative R² against UCLA, and only EC-Earth3 reaches even a modest positive Pearson correlation (`0.150444`). So the historical evidence does not support “raw CMIP SWE already reproduces the observed annual pattern”; at best it supports broad distributional comparability with poor year-order tracking.

# Compact comparison table

| Experiment | Target | Best obs Pearson r | Best obs R² | Best wet/dry | Main artifact dir | Main plot |
|---|---|---:|---:|---:|---|---|
| Absolute SWE transfer | April 1 Sierra SWE mean in `mm` | 0.046634 | -0.152455 | N/A | `artifacts/s0_attention_observational_transfer_v1/` | `artifacts/s0_attention_observational_transfer_v1/observational_timeseries.png` |
| Normalized/anomaly SWE transfer | Parent-model standardized anomaly on simulation side; UCLA standardized anomaly on obs side | 0.148774 | 0.021850 | N/A | `artifacts/s0_attention_swe_std_anomaly_v1/` | `artifacts/s0_attention_swe_std_anomaly_v1/observed_vs_predicted_anomaly_timeseries.png` |
| SWE percentile transfer | Parent-model empirical-CDF percentile in `[0,1]` | 0.159124 | 0.019171 | 0.621622 | `artifacts/s0_attention_swe_percentile_v1/` | `artifacts/s0_attention_swe_percentile_v1/main_observational_percentile_timeseries.png` |
| Raw CMIP-vs-observation pattern diagnostic | Raw April 1 Sierra SWE mean in `mm`, UCLA vs 4 CMIP6/WUS-D3 parent-model series | 0.150444 | -1.106657 | N/A | `artifacts/s0_attention_observational_transfer_v1/` | `artifacts/s0_attention_observational_transfer_v1/ucla_vs_cmip_swe_apr1_timeseries.png` |
