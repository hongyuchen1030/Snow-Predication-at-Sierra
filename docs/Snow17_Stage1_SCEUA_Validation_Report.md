# Stage 1 Scientific Validation Report: Snow-17 as an ERA5-Land P/T → Sierra SWE Translator

**Scientific question**: Given high-quality ERA5-Land P/T forcing, can a Snow-17 model
calibrated only on historical years (WY1985-2004) translate P/T into Sierra SWE that
generalizes to unseen observational years (WY2005-2021)?

All artifacts: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_stage1_scua_validation/`
(mirrored, small files only, at the same relative path under the repo's `artifacts/`).
Leakage audit: [leakage_audit.md](../artifacts/snow17_stage1_scua_validation/leakage_audit.md) — **passed**, reviewed before any held-out number below was computed.

---

## Headline result (held-out WY2005-2020, before adding WY2021)

Whole-Sierra, daily trajectory, held-out WY2005-2020 (never touched by calibration):

| Metric | Value |
|---|---|
| NSE | **0.934** |
| RMSE | 65.4 mm |
| MAE | 37.0 mm |
| Pearson r | 0.982 |
| Mean bias | +19.0 mm (wet) |

Whole-Sierra April-1 SWE, held-out WY2005-2020 (16 years):

| Metric | Value |
|---|---|
| NSE | 0.931 |
| R² | 0.982 |
| Pearson r | 0.991 |
| RMSE | 79.6 mm |
| MAE | 68.7 mm |
| Mean bias | +67.8 mm (wet) |
| Sign/anomaly agreement across years | 93.75% (15/16 years) |

Snow-17, calibrated **only** on WY1985-2004 and never re-touched, reproduces the
interannual dry/wet pattern of Sierra SWE almost perfectly (r ≈ 0.99) and captures
~93% of daily variance out of sample, but with a **systematic wet bias concentrated in
low-snow years** (detailed in Q8 below).

---

## Answers to the 9 required questions

### 1. Final frozen θ (all 7 calibrated + 5 fixed values, per region)

Fixed (all regions): `rvs=1.0, mbase=1.0, tipm=0.1, nmf=0.15, plwhc=0.04`

| Param | Bounds | North | Central | South |
|---|---|---|---|---|
| scf | [0.9, 1.2] | 1.091 | 1.128 | **1.200 (bound)** |
| mfmax | [0.5, 1.3] | 0.757 | **1.296 (bound)** | 1.026 |
| mfmin | [0.1, 0.6] | 0.249 | **0.600 (bound)** | **0.600 (bound)** |
| uadj | [0.05, 0.2] | **0.0500 (bound)** | **0.0503 (bound)** | **0.0500 (bound)** |
| pxtemp | [0.0, 2.0] | 0.255 | 1.098 | 1.176 |
| pxtemp1 | [-2.0, 0.0] | **-1.999 (bound)** | **-1.997 (bound)** | -0.404 |
| pxtemp2 | [0.0, 4.0] | 3.453 | **3.994 (bound)** | **3.982 (bound)** |

Full JSON with optimizer settings, seed, evaluation counts, git commit, and
convergence diagnostics: `frozen_parameters/theta_{N,C,S}.json`.

**Notable caveat**: UADJ saturates at its lower bound (0.05) in **all three regions**,
and 5 of the other 18 calibrated values also sit at a bound edge. This is a strong,
consistent signal that the pre-registered [0.05, 0.2] UADJ range (and several other
bounds) may not bracket the true optimum for this domain — a legitimate limitation,
not a bug (see Q8).

### 2. Does SCE-UA improve on the midpoint baseline (WY1985-2004)?

Yes, in every region and for whole-Sierra:

| Region | NSE (midpoint) | NSE (SCE-UA) | Δ NSE |
|---|---|---|---|
| North | 0.913 | 0.945 | +0.032 |
| Central | 0.941 | 0.957 | +0.016 |
| South | 0.895 | 0.945 | +0.050 |
| Whole-Sierra | 0.932 | 0.958 | +0.026 |

South improves the most (midpoint was worst there); Central improves the least
(midpoint was already strong there). Full comparison: `calibration/calibration_metrics.json`.

### 3. How well does frozen θ match observed SWE in WY2005-2020?

Good overall skill with a systematic bias (see headline table + Q8). Region breakdown,
full daily trajectory:

| Region | NSE | RMSE (mm) | Pearson r | Mean bias (mm) |
|---|---|---|---|---|
| North | 0.949 | 46.4 | 0.976 | -8.0 |
| Central | 0.914 | 80.4 | 0.980 | +29.3 |
| South | 0.899 | 86.5 | 0.976 | +27.7 |
| Whole-Sierra | 0.934 | 65.4 | 0.982 | +19.0 |

### 4. Yearly April-1 observed vs. predicted (WY2005-2020)

Full table: `heldout_2005_2020/heldout_yearly_apr1_2005_2020.csv`. Plot A ([artifacts/snow17_stage1_scua_validation/plots/plotA_yearly_april1_lines.png](../artifacts/snow17_stage1_scua_validation/plots/plotA_yearly_april1_lines.png)) shows the model tracks every wet/dry swing correctly; Plot B ([plotB_april1_scatter.png](../artifacts/snow17_stage1_scua_validation/plots/plotB_april1_scatter.png)) shows Central/South points sit consistently above the 1:1 line. Selected years:

| WY | Obs Whole-Sierra Apr-1 | Pred | Signed error |
|---|---|---|---|
| 2011 (wet) | 991.6 mm | 1042.7 mm | +51.1 mm (+5.2%) |
| 2015 (extreme drought) | 45.5 mm | 51.4 mm | +5.9 mm (+13.1%) |
| 2016 (drought recovery) | 466.1 mm | 621.5 mm | +155.4 mm (+33.3%) |
| 2017 (wettest on record) | 1016.3 mm | 1132.7 mm | +116.4 mm (+11.5%) |
| 2020 (dry) | 326.4 mm | 407.4 mm | +81.0 mm (+24.8%) |

### 5. Regional daily-trajectory held-out metrics (WY2005-2020)

See table under Q3. North is the strongest performer and least biased; Central and
South have similar NSE (~0.90-0.91) but a larger, consistently positive bias.

### 6. Whole-Sierra April-1 held-out metrics

Pearson r = 0.991, R² = 0.982, RMSE = 79.6 mm, MAE = 68.7 mm, mean bias = +67.8 mm,
sign/anomaly agreement = 93.75% across 16 years (only 1 year had predicted anomaly
sign disagree with observed anomaly sign — a near-normal year where both were close
to the multi-year mean). Secondary standardized-anomaly diagnostic (climatology fit
**only** on WY1985-2004, applied unchanged to WY2005+, never touching held-out data
to define normalization): whole-Sierra daily NSE in z-score space = 0.934 (identical
to the physical-mm NSE, as expected since NSE is invariant to affine rescaling) —
included per spec as a secondary, leakage-safe cross-check, not a replacement for the
physical-mm comparison above. Full numbers: `heldout_2005_2020/heldout_2005_2020_metrics.json`.

### 7. Does performance hold when WY2021 is added?

Yes, essentially unchanged:

| Metric (Whole-Sierra, daily) | WY2005-2020 | WY2005-2021 |
|---|---|---|
| NSE | 0.934 | 0.934 |
| RMSE | 65.4 mm | 64.0 mm |
| Pearson r | 0.982 | 0.982 |
| Mean bias | +19.0 mm | +18.6 mm |

| Metric (Whole-Sierra, April-1) | WY2005-2020 | WY2005-2021 |
|---|---|---|
| NSE | 0.931 | 0.930 |
| Pearson r | 0.991 | 0.991 |
| Sign/anomaly agreement | 93.75% (16 yr) | 94.1% (17 yr) |

WY2021 (a dry year) is absorbed without degrading skill. Full: `heldout_2005_2021/heldout_2005_2021_metrics.json`.

### 8. What systematic errors remain?

**(a) Amplitude — low-snow-year wet bias, Central & South only.** Splitting WY2005-2020
by observed whole-Sierra April-1 SWE into low/mid/high terciles:

| Region | Mean % error, low-snow years | Mean % error, high-snow years |
|---|---|---|
| North | +1.5% | -0.1% |
| Central | **+38.1%** | +9.4% |
| South | **+42.7%** | +7.8% |
| Whole-Sierra | +30.7% | +6.8% |

Central and South overpredict April-1 SWE by ~40% on average in the driest years
(2007, 2012, 2013, 2014, 2015) but are much closer to unbiased in wet years. North is
nearly unbiased across the full range, though it is the one region that *underpredicts*
in the single most extreme drought year (WY2015: obs 45 mm vs. pred 11 mm, -76%) while
Central/South overpredict that same year (Plot C, top panel).

**(b) Timing — melt persists too long in low-snow years.** Plot C (WY2015 panel, [plots/plotC_representative_years.png](../artifacts/snow17_stage1_scua_validation/plots/plotC_representative_years.png)) shows Central/South predicted SWE
tracking a secondary mid-winter/spring bump the observations do not show, i.e. the
model is slower to fully ablate a shallow snowpack than reality. Mean |peak-date error|
across all held-out years is 17-22 days depending on region — not negligible, but small
relative to the ~150-200 day snow season.

**(c) High-snow years** show smaller, still-positive biases in Central/South (Plot C,
WY2017 panel) and a mild underprediction in North's peak.

**(d) Parameter-bound saturation.** UADJ pinned at its lower bound in all 3 regions,
plus SCF/MFMIN/PXTEMP1/PXTEMP2 saturating in South and Central specifically (see Q1).
This is the most likely structural driver of (a): a wind-function coefficient forced
to its floor, combined with SCF pinned at its ceiling in South, is consistent with the
optimizer wanting *more* precipitation captured and *less* wind-driven melt than the
prescribed bounds allow — which would show up exactly as the observed wet, slow-melt
bias in low-snow years, where the seasonal snowpack is thin enough that these
parameter effects dominate the mass/energy balance.

### 9. Did the leakage audit pass?

**Yes.** `leakage_audit.md` (written before any held-out metric was computed) checked:
SCE-UA objective/evaluation restricted to WY1985-2004 only; optimizer stopping rule is
a function of in-sample objective trajectory only; objective/hyperparameter choices
were based on calibration-period convergence behavior only; regional mask and
occupancy weights are static geography, not fit to any year's SWE; elevation
correction uses a fixed literature lapse-rate constant on static terrain products;
standardized-anomaly climatology uses only WY1985-2004; frozen θ files are written
before any held-out script runs and never modified afterward; the full-period
simulation is one continuous run per region (no per-water-year re-zeroing). No
mechanism was found that lets WY2005+ SWE influence the frozen parameters, the
optimizer configuration, or any preprocessing/normalization step.

---

## Reproducibility check

Independently re-ran `Snow17RegionSetup.simulation()`/`objectivefunction()` for each
region using the exact saved `calibrated_parameters` vector (no optimizer involved):
reproduced 1-NSE matched the recorded value to the reported precision for all three
regions (North 0.055148, Central 0.043014, South 0.055383 — exact matches).

## Compute / provenance summary

SCE-UA (SPOTPY, `parallel="mpc"`, `dbformat="ram"`), seed=42, `ngs=7`, `kstop=5`,
`pcento=peps=0.01`, requested 2000 repetitions/region (South converged in exactly
2000, Central in 2000, North in 1995 — all within budget). Regions optimized
independently. A first production attempt (3000 reps, `kstop=10`) was lost to a 4-hour
Perlmutter walltime expiration mid-run (no checkpointing exists in this SPOTPY
configuration); the budget was reduced based on that run's observed calibration-period
plateau behavior, not on any held-out signal. Full detail: `provenance.md`.

---

## Verdict

```
B. Snow-17 has partial held-out skill but important limitations must be documented before ACE2 coupling.
```

**Rationale**: Held-out daily/April-1 correlation (r ≈ 0.98-0.99) and variance
explained (NSE ≈ 0.90-0.96, held up unchanged with WY2021 added) are genuinely strong,
and the interannual wet/dry signal that ACE2 coupling would need to reproduce is
captured with 93-94% year-to-year sign agreement — this rules out verdict C. However,
the systematic ~30-40% wet bias in low-snow years for Central/South, the uniform UADJ
bound saturation across all three regions, and the slow-melt timing artifact in dry
years are not incidental noise — they are a consistent, physically-interpretable bias
structure that would need to be either corrected (e.g. widened UADJ/SCF bounds and
recalibration) or explicitly accounted for before using Snow-17 as a trusted P/T→SWE
translator for ACE2-driven projections, particularly for drought-year assessments.
This rules out a clean verdict A.

Per the task scope, this concludes Stage 1. No recalibration on WY1985-2021, no ACE2,
CMIP6, or NeuralGCM forcing has been used or is proposed as a next step here.
