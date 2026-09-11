# Short-Timescale (7-day) Pretraining Dataset — Feasibility Audit

**Update:** §4 below checked daily availability by listing table-directory names
only (`ls`). That check has since been redone by opening the actual NetCDF files and
reading their metadata directly — see
[CMIP6_Daily_Variable_Availability_Verified.md](CMIP6_Daily_Variable_Availability_Verified.md).
The ground-truth result is *more* restrictive than §4 states below: EC-Earth3's daily
`ta`/`tas`/`ua`/`va`/`zg` directories exist but are empty (no files), so EC-Earth3
actually has real daily data for only 1 of 19 fields (`psl`), not the 6 implied here.
The overall conclusion (no consistent daily 19-variable/4-model dataset exists) is
unchanged and now confirmed at the file-metadata level.

**Status: BLOCKED at the data-inspection stage per the task's explicit stop condition.**
No dataset-generation code was written and no ML model was touched. This document
records what was inspected, what was reused, and exactly why construction stopped.

## 1. What "the existing pipeline" is

The ~394-simulated-water-year seasonal CNN pipeline referenced in the task is:

- Predictor construction: [scripts/process_cmip6_selected_regrid_standardize.py](../scripts/process_cmip6_selected_regrid_standardize.py)
  regrids 19 CMIP6 fields from 4 parent GCMs to a common 1.5°×1.5° global grid
  (120 lat × 240 lon) and assembles them into 12-month (Sep–Aug) water-year rows.
- Predictor stacking: [scripts/prepare_cmip6_cnn_experiment.py](../scripts/prepare_cmip6_cnn_experiment.py)
  (`PREDICTOR_SPECS`, 19 entries) merges those fields with SWE/CPM/AQM labels into
  `inputs_physical.npy` / `inputs_valid_mask.npy` / `targets.npy` + `manifest.csv`.
- SWE target construction: [scripts/build_wusd3_swe_labels.py](../scripts/build_wusd3_swe_labels.py),
  using `snow_ml.data.DEFAULT_SIERRA_REGION` / `build_sierra_mask` and
  `snow_ml.data_wusd3` to compute an April-1 area-weighted Sierra SWE scalar from
  WUS-D3 daily "snow" output. This is the exact 394-row canonical table
  (`docs`/`artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels.csv`,
  `canonical_table_note`: "restricted to the exact 394-row overlap with the CMIP6
  predictor/auxiliary manifest").
- CNN model code that consumes this tensor: [src/snow_ml/cmip6_cnn_experiment.py](../src/snow_ml/cmip6_cnn_experiment.py)
  (`in_channels_per_month = 19 predictors × 2 [value + valid-mask] = 38`, confirmed at
  `scripts/build_s0_historical_batch_order_audit.py:44`).

These are the functions this task intended to reuse (regridding, Sierra masking, area
weighting, unit handling, manifest joins) rather than reimplement.

## 2. The 19 atmospheric input variables, exactly as currently used

From `PREDICTOR_SPECS` in `prepare_cmip6_cnn_experiment.py`, matched to `FIELD_SPECS`
in `process_cmip6_selected_regrid_standardize.py`:

| # | Field name | CMIP6 variable_id | Level / depth | CMIP6 table_id | Native temporal resolution |
|---|---|---|---|---|---|
| 1 | `tos` | `tos` | surface | **Omon** | monthly |
| 2 | `siconc` | `siconc` | surface | **SImon** | monthly |
| 3 | `rlut` | `rlut` | TOA | **Amon** | monthly |
| 4 | `zg_500` | `zg` | 500 hPa | **Amon** | monthly |
| 5 | `ta_850` | `ta` | 850 hPa | **Amon** | monthly |
| 6 | `ua_850` | `ua` | 850 hPa | **Amon** | monthly |
| 7 | `va_850` | `va` | 850 hPa | **Amon** | monthly |
| 8 | `ua_200` | `ua` | 200 hPa | **Amon** | monthly |
| 9 | `va_200` | `va` | 200 hPa | **Amon** | monthly |
| 10 | `hus_850` | `hus` | 850 hPa | **Amon** | monthly |
| 11 | `psl` | `psl` | surface | **Amon** | monthly |
| 12 | `tas` | `tas` | surface | **Amon** | monthly |
| 13 | `zg_50` | `zg` | 50 hPa | **Amon** | monthly |
| 14 | `ta_50` | `ta` | 50 hPa | **Amon** | monthly |
| 15 | `ua_50` | `ua` | 50 hPa | **Amon** | monthly |
| 16 | `va_50` | `va` | 50 hPa | **Amon** | monthly |
| 17 | `mrso` | `mrso` | column soil moisture | **Lmon** | monthly |
| 18 | `thetao_50m` | `thetao` | 50 m depth | **Omon** | monthly |
| 19 | `thetao_100m` | `thetao` | 100 m depth | **Omon** | monthly |

Every one of the 19 fields is sourced exclusively from a **monthly** CMIP6 table
(`Amon`/`Omon`/`SImon`/`Lmon`) — see `FIELD_SPECS` (lines 105–125 of
`process_cmip6_selected_regrid_standardize.py`). Source archive:
`/global/cfs/projectdirs/m3522/cmip6/CMIP6`. Grid: regridded to a common global
1.5°×1.5° grid, 120×240, `lat ∈ [-89.25, 90)` step 1.5, `lon ∈ [0.75, 360)` step 1.5.
Units are kept as the native CMIP6 attribute units for each field (K for
`ta`/`tas`/`thetao`, Pa for `psl`, m for `zg`, m/s for `ua`/`va`, kg kg⁻¹ for `hus`,
W m⁻² for `rlut`, kg m⁻² for `mrso`, fraction for `siconc`); the pipeline additionally
applies a pooled train+val per-(month, lat, lon) z-score standardization
(`compute_feature_stats` in `src/snow_ml/cmip6_cnn_experiment.py`) — this is
train/val-pooled normalization already in the existing pipeline, not something newly
introduced here, and is flagged only as documentation per the task's request.

Parent models / members (4 GCMs × historical+ssp370, `PARENT_RUNS` in
`process_cmip6_selected_regrid_standardize.py`): `EC-Earth3:r102i1p1f1`,
`MIROC6:r1i1p1f1`, `MPI-ESM1-2-HR:r3i1p1f1`, `TaiESM1:r1i1p1f1`.

## 3. SWE target side — this part is fine

The SWE source is WUS-D3 dynamically-downscaled `snow` output (mm SWE), **available
at true daily resolution** (`day` coordinate per file-year; confirmed in
`build_wusd3_swe_labels.py`: `"temporal_resolution": "daily"`). The April-1 seasonal
label is just one daily snapshot pulled from this daily series via
`load_wusd3_snapshot`. Sierra definition: `DEFAULT_SIERRA_REGION` in
`src/snow_ml/data.py` (`lat 35.0–42.0`, `lon -122.5 to -118.0`), applied via
`build_sierra_mask`, area-weighted by native WUS-D3 d02 cell area
(`area_from_bounds_2d`). **Daily SWE(t) and SWE(t+7) and their difference could be
constructed today with no blocking issue** — the SWE/target side of the task is
achievable using existing functions.

## 4. The blocking finding: the atmospheric side is not available at daily resolution

Per the task's explicit instruction — *"If any of the 19 inputs are not available at
the daily/sub-daily resolution needed to construct 7-day windows, stop and report the
issue instead of silently interpolating or substituting variables"* — this audit
stopped here rather than fabricating a workaround.

The existing pipeline's 19 atmospheric channels are built exclusively from monthly
CMIP6 tables. To check whether a *daily* CMIP6 source could substitute without
changing the physical variable identity, I inspected the raw archive
(`/global/cfs/projectdirs/m3522/cmip6/CMIP6/.../{historical,ssp370}/<member>/`) for
daily (`day`/`Oday`/`SIday`) tables for the same 4 parent runs:

| Field | EC-Earth3 (r102i1p1f1) | MIROC6 (r1i1p1f1) | MPI-ESM1-2-HR (r3i1p1f1) | TaiESM1 (r1i1p1f1) |
|---|---|---|---|---|
| `zg`, `ta`, `ua`, `va`, `psl`, `tas` (all levels used) | daily ✅ (`day`) | daily ✅ | daily ✅ | daily ✅ |
| `hus` (850 hPa) | **monthly only** ❌ (`day` table has no `hus`, only `huss`) | daily ✅ | daily ✅ | daily ✅ |
| `rlut` | **monthly only** ❌ (absent from `day` table) | daily ✅ | daily ✅ | daily ✅ |
| `mrso` | **monthly only** ❌ (absent from `day` table, no daily land table) | daily ✅ (`day`) | daily ✅ (`day`) | **monthly only** ❌ |
| `tos` | **monthly only** ❌ (no `Oday` directory at all) | daily ✅ (`Oday`) | daily ✅ (`Oday`) | **monthly only** ❌ (no `Oday`) |
| `siconc` | **monthly only** ❌ (no `SIday`) | daily ✅ (`SIday`) | daily ✅ (`SIday`) | **monthly only** ❌ (no `SIday`) |
| `thetao` (both 50m, 100m) | **monthly only** ❌ | **monthly only** ❌ (no `thetao` in `Oday`) | **monthly only** ❌ (no `thetao` in `Oday`) | **monthly only** ❌ |

Findings:

- **`thetao_50m` and `thetao_100m` have no daily CMIP6 product for any of the 4
  parent models.** These 2 of the 19 variables cannot be sourced at daily resolution
  at all, from any model, under the current variable definition.
- **`tos`, `siconc`, `rlut`, `hus_850`, `mrso` are daily for 2 of the 4 parent models
  (MIROC6, MPI-ESM1-2-HR) but not for EC-Earth3 or TaiESM1** (EC-Earth3 and TaiESM1
  are each missing a different subset of these 5 fields at daily resolution — not
  even consistent with each other).
- Only 6 of the 19 fields (`zg`, `ta`, `ua`, `va` at the used levels, `psl`, `tas`)
  are daily-available across all 4 models.

There is no way to assemble a **daily-resolution, all-19-variable, all-4-model**
input tensor that reuses the existing preprocessing without either (a) dropping
variables the task said must stay fixed, (b) dropping parent models/samples the task
did not ask to drop, or (c) substituting a different physical source for the missing
fields — all three of which the task explicitly told me not to do silently.

## 5. What is NOT blocked

- The SWE target pipeline (daily, mm, Sierra-masked, area-weighted) is ready to use
  as-is for `SWE(t)`, `SWE(t+7)`, and `ΔSWE_7d`.
- The Nov 1–Mar 31 windowing, water-year/member boundary logic, and integrity-check
  design from the task spec are all straightforward to implement once an
  atmospheric-input decision is made — none of that logic was blocked, only the
  19-variable/daily-resolution atmospheric side.

## 6. Decision needed before any dataset code is written

This is a scientific-scope decision, not something to default silently. Options
(not mutually exclusive with future ablations):

**A.** Restrict the short-timescale experiment to the 2 parent models with daily
coverage for the ocean/land/OLR fields (MIROC6, MPI-ESM1-2-HR) and drop `thetao_50m`
/`thetao_100m` (no daily source anywhere) → an 17-variable, 2-model subset of the 394
water years, still using genuinely-daily physical fields.

**B.** Keep exactly 19 variables and all 394 water-years/4 models, but accept that at
7-day/daily granularity the monthly-only fields (`thetao`×2, and for EC-Earth3/TaiESM1
also `tos`, `siconc`, `rlut`, `hus_850`, `mrso`) must be held constant across each
month (i.e., the same monthly value repeated for every day in that month) — this is
exactly the "silently interpolate/substitute" pattern the task said not to do without
flagging, so I am flagging it rather than doing it.

**C.** Use a different daily-native input source entirely for the short-timescale
experiment (e.g., the WUS-D3 d02 dynamically-downscaled daily meteorological forcing
that already drives the daily SWE simulation) — this means defining a new
short-timescale variable set rather than reusing the CMIP6-19, which the task also
said not to do without checking first.

I'm stopping for a decision on A/B/C (or another direction) before writing any
dataset-generation code.
