# ERA5 Sierra SWE LOD Preprocessing Plan

Status date: `2026-06-09`

This report reflects the corrected monthly Sierra SWE preprocessing run only. No LOD was run.

- Invalid Apr 1 SWE provenance files remain archived at:
  `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_preprocessing/swe_targets/invalid_apr1_target/`
- The corrected monthly Sierra SWE product was successfully created from the raw UCLA SWE files using a daily-first area-weighted reduction.
- The full run used water years `1985..2021`, produced `13,505` daily samples and `444` monthly samples, and wrote both the NetCDF and JSON outputs under `/pscratch`.

## Table 1. ERA5 predictor preprocessing inventory

| Predictor | Raw data location | Existing/reused processed data location | New processed data location | Year/domain used | Processing actually performed | Final processed data: variable name, units, dimensions, time coverage, monthly/anomaly status |
| --- | --- | --- | --- | --- | --- | --- |
| `ERA5 predictors` | not updated in this turn | not updated in this turn | not updated in this turn | not updated in this turn | this turn only fixed and completed the SWE monthly target preprocessing; ERA5 predictor preprocessing was not rerun here | no new ERA5 predictor output was created in this turn |

## Table 2. SWE target preprocessing inventory

| Target | Raw SWE source location | Existing/reused processed SWE location | New processed SWE location | Region/time target | Processing actually performed | Final target data: variable name, units, dimensions, water-year/month coverage, raw/monthly/anomaly status |
| --- | --- | --- | --- | --- | --- | --- |
| `Monthly Sierra-area SWE` | `/global/cfs/projectdirs/m3522/datalake/UCLA_WUS_SNOWv1/WUS_UCLA_SR_v01_ALL_0_agg_16_WY{water_year}_SD_SWE_SCA_POST.nc`; variable `SWE_Post`; `Stats=0`; raw units `m`; raw dims `(time, Stats, Latitude, Longitude)` | invalid Apr 1 target archived only for provenance: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_preprocessing/swe_targets/invalid_apr1_target/` | NetCDF: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_preprocessing/swe_targets/sierra_swe_monthly_area_average_anomaly_wy1985_2021.nc`; summary: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_preprocessing/swe_targets/sierra_swe_monthly_area_average_anomaly_wy1985_2021_summary.json` | Sierra region `35..42N`, `-122.5..-118W`; daily coverage `1984-10-01..2021-09-30`; monthly coverage `1984-10..2021-09`; water years `1985..2021` | completed. For each water-year file: subset once to the Sierra box, select `SWE_Post` at `Stats=0`, compute fixed weights `W_i = f_i A_i`, reduce the native-grid daily field to a scalar series by `SWE_Sierra_daily(t) = sum_i[f_i A_i valid_i(t) SWE_i(t)] / sum_i[f_i A_i valid_i(t)]`, then compute `SWE_monthly(y,m) = mean_daily_values_in_month[SWE_Sierra_daily(t)]`, `clim_SWE(m) = mean_y[SWE_monthly(y,m)]`, and `SWE_anom(y,m) = SWE_monthly(y,m) - clim_SWE(m)`. Grid-cell area used `A_i = R^2 * |sin(lat_north)-sin(lat_south)| * |lon_east-lon_west|` with coordinate bounds inferred from adjacent cell centers. Repo mask utility type is `fractional`; on this native Sierra subset the processed cells had no partial-weight values, so `mask_has_fractional_values_on_processed_subset = false`. | variables: `sierra_swe_daily_mean_m` `(daily_time)` `m`; `sierra_swe_monthly_mean_m` `(time)` `m`; `sierra_swe_monthly_mean_mm` `(time)` `mm`; `sierra_swe_monthly_anom_m` `(time)` `m`; `sierra_swe_monthly_anom_mm` `(time)` `mm`; `sierra_swe_monthly_climatology_m` `(month)` `m`; `sierra_swe_monthly_climatology_mm` `(month)` `mm`. Coverage: `13,505` daily samples and `444` monthly samples. Runtime `5948.36 s`; peak memory `19367.92 MB`. Product is daily scalar + monthly mean + monthly anomaly. |

## Notes

- Smoke test first:
  - ran `WY1985` only with no final write
  - verified `365` daily samples, `12` monthly samples
  - verified scalar SWE range before launching the full run
- Full run log:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/swe_monthly_target_full.log`
- Summary JSON fields report:
  - `raw_files_used`
  - `water_years_used`
  - `number_of_daily_samples`
  - `number_of_monthly_samples`
  - `mask_type_attr`
  - `mask_has_fractional_values_on_processed_subset`
  - `number_of_nonzero_weight_cells`
  - `total_effective_sierra_area_m2`
  - `grid_cell_area_formula`
  - `runtime_seconds`
  - `peak_memory_mb`
  - `output_path`
