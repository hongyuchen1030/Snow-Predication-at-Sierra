#!/usr/bin/env python3
"""Recover fixed CPM/AQM reference patterns and build CMIP6 auxiliary labels."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cmip6_aux_labels"

CPM_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cpm_era5_reproduction"
CPM_FIXED_PROJ_DIR = PROJECT_ROOT / "artifacts" / "cpm_swe_attribution_37yr_fixed_cpm"
AQM_ATTRIBUTION_DIR = PROJECT_ROOT / "artifacts" / "atlantic_pc245_attribution"
AQM_LOYO_DIR = PROJECT_ROOT / "artifacts" / "aqm_index_loyo"

CMIP6_RAW_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_regridded_1p5deg/raw")
CMIP6_ZG_PATH = CMIP6_RAW_DIR / "zg_500hPa_1p5deg_model_years.nc"
CMIP6_TOS_PATH = CMIP6_RAW_DIR / "tos_1p5deg_model_years.nc"


def ensure_output_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def sqrt_cos_weights(lat: xr.DataArray) -> xr.DataArray:
    values = np.sqrt(np.cos(np.deg2rad(lat.values.astype(float))))
    return xr.DataArray(values, coords={lat.dims[0]: lat.values}, dims=lat.dims)


def cos_weights(lat: xr.DataArray) -> xr.DataArray:
    values = np.cos(np.deg2rad(lat.values.astype(float)))
    return xr.DataArray(values, coords={lat.dims[0]: lat.values}, dims=lat.dims)


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def rmse_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) == 0:
        return float("nan")
    diff = x[mask] - y[mask]
    return float(np.sqrt(np.mean(diff**2)))


def linear_fit_and_standardize(
    values: np.ndarray, time_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(time_index, dtype=float)
    y = np.asarray(values, dtype=float)
    x_center = float(np.mean(x))
    x_var = float(np.sum((x - x_center) ** 2))
    if x_var <= 0.0:
        raise ValueError("Time axis variance must be positive for detrending.")
    y_mean = np.mean(y, axis=0)
    slopes = np.sum((x[:, None] - x_center) * (y - y_mean[None, :]), axis=0) / x_var
    intercepts = y_mean - slopes * x_center
    detrended = y - (slopes[None, :] * x[:, None] + intercepts[None, :])
    mu = np.mean(detrended, axis=0)
    sigma = np.std(detrended, axis=0, ddof=1)
    keep = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(mu)
    if not np.any(keep):
        raise ValueError("No valid columns remain after detrending and standardization.")
    return detrended[:, keep], mu[keep], sigma[keep], slopes[keep], intercepts[keep], keep


def infer_experiment_from_row_year(row_year: int) -> str:
    return "historical" if int(row_year) <= 2013 else "ssp370"


def monthly_anomalies_by_model(da: xr.DataArray) -> xr.DataArray:
    month_vals = da["month_in_model_year"].values.astype(int)
    outputs = []
    for model_value in pd.unique(da["model"].values.astype(str)):
        sub = da.sel(sample=(da["model"] == model_value))
        sub_out = []
        for month in month_vals:
            block = sub.sel(month_in_model_year=month)
            clim = block.mean("sample", skipna=True)
            sub_out.append((block - clim).expand_dims(month_in_model_year=[month]))
        model_da = xr.concat(sub_out, dim="month_in_model_year").transpose("sample", "month_in_model_year", "lat", "lon")
        outputs.append(model_da)
    return xr.concat(outputs, dim="sample").sortby("sample")


def seasonal_average_from_model_years(
    da: xr.DataArray,
    months: tuple[int, ...],
) -> xr.DataArray:
    month_coord = da["month_in_model_year"].values.astype(int)
    missing = [month for month in months if month not in month_coord]
    if missing:
        raise ValueError(f"Missing months from model-year coordinate: {missing}")
    return da.sel(month_in_model_year=list(months)).mean("month_in_model_year", skipna=True)


def validate_cpm_roundtrip() -> dict:
    eof_ds = xr.open_dataset(CPM_ARTIFACT_DIR / "era5_z500_eof1to6.nc")
    anom_ds = xr.open_dataset(CPM_ARTIFACT_DIR / "era5_z500_daily_anomalies.nc")
    pc_df = pd.read_csv(CPM_ARTIFACT_DIR / "era5_z500_pc1to6_daily.csv", parse_dates=["date"])

    eof3 = eof_ds["eof_unweighted_sigma_scaled_m"].sel(component=3)
    pc3_std = float(eof_ds["pc_standard_deviation"].sel(component=3).item())
    weights = sqrt_cos_weights(eof3["latitude"])
    projected = (anom_ds["z500_m_anomaly"] * weights).transpose("date", "latitude", "longitude").values
    eof_weighted = (eof3 * weights / pc3_std).values.astype(float)
    proj_series = np.tensordot(projected.astype(float), eof_weighted.astype(float), axes=([1, 2], [0, 1]))
    saved = pc_df["PC3"].to_numpy(dtype=float)

    overlap = pd.DataFrame(
        {
            "date": pc_df["date"],
            "saved_pc3": saved,
            "projected_pc3_from_saved_anomalies": proj_series,
        }
    )
    overlap["difference"] = overlap["projected_pc3_from_saved_anomalies"] - overlap["saved_pc3"]
    overlap.to_csv(ARTIFACT_DIR / "cpm_reference_roundtrip.csv", index=False)

    meta = json.loads((CPM_ARTIFACT_DIR / "era5_z500_processing_metadata.json").read_text())
    return {
        "pattern_path": str(CPM_ARTIFACT_DIR / "era5_z500_eof1to6.nc"),
        "pattern_variable": "eof_unweighted_sigma_scaled_m(component=3)",
        "pc_series_path": str(CPM_ARTIFACT_DIR / "era5_z500_pc1to6_daily.csv"),
        "anomaly_path": str(CPM_ARTIFACT_DIR / "era5_z500_daily_anomalies.nc"),
        "domain_lat_range": [float(eof3["latitude"].min()), float(eof3["latitude"].max())],
        "domain_lon_range_360": [float(eof3["longitude"].min()), float(eof3["longitude"].max())],
        "season": "daily Nov-Apr wet season; final annual scalar = mean Nov-Apr daily PC3 by water year",
        "anomaly_definition": "ERA5 daily Z500 anomalies relative to saved calendar-day climatology",
        "weighting": "sqrt(cos(lat)) in EOF basis; projection equivalent to cos(lat)-weighted dot product divided by PC3 std",
        "normalization": "saved EOF3 divided by saved pc_standard_deviation",
        "sign_convention": int(meta.get("best_match_sign", 1)),
        "units": "m",
        "roundtrip_correlation": corrcoef_safe(saved, proj_series),
        "roundtrip_rmse": rmse_safe(saved, proj_series),
        "roundtrip_n": int(len(saved)),
    }


@dataclass
class AQMFixedLoading:
    loading_vector: np.ndarray
    loading_field: xr.DataArray
    keep_mask_field: xr.DataArray
    train_year_zero: int
    x_mu: np.ndarray
    x_sigma: np.ndarray
    x_slopes: np.ndarray
    x_intercepts: np.ndarray
    valid_flat_mask: np.ndarray


def reconstruct_aqm_fixed_loading() -> tuple[AQMFixedLoading, dict]:
    from scripts import run_atlantic_pc245_attribution as attribution

    hadisst = attribution.load_hadisst_sst()
    gpcc = attribution.load_gpcc_precip()

    train_sst = attribution.latitude_slice(
        hadisst.sel(time=slice("1890-11-01", "2019-02-28")),
        attribution.AQM_SST_LAT_MIN,
        attribution.AQM_SST_LAT_MAX,
    )
    train_precip = attribution.latitude_slice(
        gpcc.sel(time=slice("1890-12-01", "2019-03-31"), lon=slice(attribution.AQM_PRECIP_LON_MIN, attribution.AQM_PRECIP_LON_MAX)),
        attribution.AQM_PRECIP_LAT_MIN,
        attribution.AQM_PRECIP_LAT_MAX,
    )

    train_sst_season = attribution.seasonal_average_label_by_year(train_sst, attribution.SEASON_MONTHS_SST, 3)
    train_precip_season = attribution.seasonal_average_label_by_year(train_precip, attribution.SEASON_MONTHS_PRECIP, 3)
    common_train_years = np.intersect1d(
        train_sst_season["time"].dt.year.values.astype(int),
        train_precip_season["time"].dt.year.values.astype(int),
    )
    train_sst_season = train_sst_season.sel(time=train_sst_season["time"].dt.year.isin(common_train_years))
    train_precip_season = train_precip_season.sel(time=train_precip_season["time"].dt.year.isin(common_train_years))

    x_train_raw = train_sst_season.values.reshape(train_sst_season.sizes["time"], -1)
    y_train_raw = train_precip_season.values.reshape(train_precip_season.sizes["time"], -1)
    xmask = np.isfinite(x_train_raw).all(axis=0)
    ymask = np.isfinite(y_train_raw).all(axis=0)
    x_train = x_train_raw[:, xmask]
    y_train = y_train_raw[:, ymask]

    train_years = train_sst_season["time"].dt.year.values.astype(int)
    t_train = train_years - int(train_years.min())

    x_train_dt, x_mu, x_sigma, x_slopes, x_intercepts, x_keep = linear_fit_and_standardize(x_train, t_train)
    y_train_dt, y_mu, y_sigma, _, _, _ = linear_fit_and_standardize(y_train, t_train)
    x_train_std = (x_train_dt - x_mu[None, :]) / x_sigma[None, :]
    y_train_std = (y_train_dt - y_mu[None, :]) / y_sigma[None, :]

    cxy = (x_train_std.T @ y_train_std) / float(x_train_std.shape[0] - 1)
    u, _, _ = np.linalg.svd(cxy, full_matrices=False)
    loading = u[:, 1]

    full_valid_mask = np.zeros(xmask.shape, dtype=bool)
    full_valid_mask[np.where(xmask)[0][x_keep]] = True
    loading_full = np.full(xmask.shape, np.nan, dtype=float)
    loading_full[full_valid_mask] = loading

    lat = train_sst_season["lat"].values
    lon = train_sst_season["lon"].values
    loading_field = xr.DataArray(
        loading_full.reshape(len(lat), len(lon)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="aqm_sst_mode2_loading",
        attrs={
            "long_name": "Reconstructed Stone-style AQM SST-side mode-2 loading",
            "description": "Deterministically reconstructed from HadISST+GPCC lagged MCA workflow.",
        },
    )
    keep_mask_field = xr.DataArray(
        full_valid_mask.reshape(len(lat), len(lon)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="valid_mask",
    )

    ds_out = xr.Dataset({"aqm_sst_mode2_loading": loading_field, "valid_mask": keep_mask_field})
    ds_out.to_netcdf(ARTIFACT_DIR / "aqm_reconstructed_mode2_loading.nc")

    saved_idx = pd.read_csv(AQM_ATTRIBUTION_DIR / "reference_indices_monthly_and_seasonal.csv", parse_dates=["date"])
    saved_idx = saved_idx.loc[saved_idx["AQM_S2_NDJF"].notna(), ["date", "AQM_S2_NDJF"]].copy()

    full_sst = attribution.latitude_slice(
        hadisst.sel(time=slice("1890-11-01", "2018-02-28")),
        attribution.AQM_SST_LAT_MIN,
        attribution.AQM_SST_LAT_MAX,
    )
    full_sst_season = attribution.seasonal_average_label_by_year(full_sst, attribution.SEASON_MONTHS_SST, 3)
    full_years = full_sst_season["time"].dt.year.values.astype(int)
    x_full_raw = full_sst_season.values.reshape(full_sst_season.sizes["time"], -1)
    x_full = x_full_raw[:, xmask][:, x_keep]
    t_full = full_years - int(train_years.min())
    x_full_dt = x_full - (x_slopes[None, :] * t_full[:, None] + x_intercepts[None, :])
    x_full_std = (x_full_dt - x_mu[None, :]) / x_sigma[None, :]
    reconstructed = x_full_std @ loading
    reconstructed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(full_sst_season["time"].values),
            "aqm_reconstructed_from_fixed_loading": reconstructed.astype(float),
        }
    )
    overlap = saved_idx.merge(reconstructed_df, on="date", how="inner")

    corr = corrcoef_safe(
        overlap["AQM_S2_NDJF"].to_numpy(dtype=float),
        overlap["aqm_reconstructed_from_fixed_loading"].to_numpy(dtype=float),
    )
    sign = 1.0
    if np.isfinite(corr) and corr < 0.0:
        sign = -1.0
        loading = loading * sign
        loading_field = loading_field * sign
        reconstructed = reconstructed * sign
        reconstructed_df["aqm_reconstructed_from_fixed_loading"] = reconstructed.astype(float)
        overlap = saved_idx.merge(reconstructed_df, on="date", how="inner")
        corr = corrcoef_safe(
            overlap["AQM_S2_NDJF"].to_numpy(dtype=float),
            overlap["aqm_reconstructed_from_fixed_loading"].to_numpy(dtype=float),
        )
        xr.Dataset({"aqm_sst_mode2_loading": loading_field, "valid_mask": keep_mask_field}).to_netcdf(
            ARTIFACT_DIR / "aqm_reconstructed_mode2_loading.nc"
        )

    overlap["difference"] = overlap["aqm_reconstructed_from_fixed_loading"] - overlap["AQM_S2_NDJF"]
    overlap.to_csv(ARTIFACT_DIR / "aqm_reference_roundtrip.csv", index=False)

    provenance = json.loads((AQM_LOYO_DIR / "aqm_source_and_method.json").read_text())
    summary = {
        "saved_pattern_exists": True,
        "saved_pattern_path": str(AQM_ATTRIBUTION_DIR / "reference_patterns_raw.nc"),
        "saved_pattern_variable": "AQM",
        "saved_pattern_note": "Saved artifact is the homogeneous SST correlation map, not the MCA singular vector used for scalar projection.",
        "reconstructed_loading_path": str(ARTIFACT_DIR / "aqm_reconstructed_mode2_loading.nc"),
        "training_period_start": str(pd.Timestamp(train_sst_season["time"].values[0]).date()),
        "training_period_end": str(pd.Timestamp(train_sst_season["time"].values[-1]).date()),
        "domain_lat_range": [float(loading_field["lat"].min()), float(loading_field["lat"].max())],
        "domain_lon_range_360": [float(loading_field["lon"].min()), float(loading_field["lon"].max())],
        "season": "NDJF SST seasonal means labeled on March 1 of year Y",
        "weighting": "No extra latitude weighting in the Stone-style SST-side singular-vector projection after gridpoint standardization",
        "normalization": "Linear detrending and gridpoint standardization before MCA; projection onto reconstructed mode-2 SST loading",
        "sign_adjustment_applied": bool(sign < 0.0),
        "roundtrip_correlation": corr,
        "roundtrip_rmse": rmse_safe(
            overlap["AQM_S2_NDJF"].to_numpy(dtype=float),
            overlap["aqm_reconstructed_from_fixed_loading"].to_numpy(dtype=float),
        ),
        "roundtrip_n": int(len(overlap)),
        "reference_index_path": str(AQM_ATTRIBUTION_DIR / "reference_indices_monthly_and_seasonal.csv"),
        "source_method_json": str(AQM_LOYO_DIR / "aqm_source_and_method.json"),
        "source_validation_overlap_correlation": provenance.get("tail_extension", {}).get("validation_against_saved_overlap", {}).get("overlap_correlation"),
    }

    return (
        AQMFixedLoading(
            loading_vector=loading,
            loading_field=loading_field,
            keep_mask_field=keep_mask_field,
            train_year_zero=int(train_years.min()),
            x_mu=x_mu,
            x_sigma=x_sigma,
            x_slopes=x_slopes,
            x_intercepts=x_intercepts,
            valid_flat_mask=full_valid_mask,
        ),
        summary,
    )


def build_cpm_labels_from_cmip6() -> tuple[pd.DataFrame, dict]:
    ds = xr.open_dataset(CMIP6_ZG_PATH)
    var_name = "zg_500hPa"
    da = ds[var_name]
    anoms = monthly_anomalies_by_model(da)

    eof_ds = xr.open_dataset(CPM_ARTIFACT_DIR / "era5_z500_eof1to6.nc")
    eof3 = eof_ds["eof_unweighted_sigma_scaled_m"].sel(component=3)
    pc3_std = float(eof_ds["pc_standard_deviation"].sel(component=3).item())
    weights = sqrt_cos_weights(eof3["latitude"])
    eof_weighted = (eof3 * weights / pc3_std).values.astype(float)

    cpm_months = anoms.sel(month_in_model_year=[11, 12, 1, 2, 3])
    cpm_months = cpm_months.interp(lat=eof3["latitude"], lon=eof3["longitude"], method="linear")
    proj = np.tensordot((cpm_months.values.astype(float) * weights.values[None, None, :, None]), eof_weighted, axes=([2, 3], [0, 1]))
    cpm_monthly = xr.DataArray(
        proj,
        coords={"sample": cpm_months["sample"], "month_in_model_year": cpm_months["month_in_model_year"]},
        dims=("sample", "month_in_model_year"),
        name="cpm_monthly_projection",
    )
    cpm_annual = cpm_monthly.mean("month_in_model_year", skipna=True)

    out = pd.DataFrame(
        {
            "sample": ds["sample"].values.astype(int),
            "model_member": ds["model"].values.astype(str),
            "row_year": ds["row_year"].values.astype(int),
            "water_year": ds["row_year"].values.astype(int) + 1,
            "experiment": [infer_experiment_from_row_year(v) for v in ds["row_year"].values.astype(int)],
            "CPM_label": cpm_annual.values.astype(float),
        }
    )
    model_parts = out["model_member"].str.split(":", n=1, expand=True)
    out["source_id"] = model_parts[0]
    out["member_id"] = model_parts[1]
    return out, {
        "cmip6_source_path": str(CMIP6_ZG_PATH),
        "cmip6_variable": var_name,
        "cmip6_preprocessing_note": "Monthly CMIP6 Z500 anomalies computed per model and calendar month on the regridded 1.5-degree products, then projected onto fixed ERA5 EOF3 and averaged across Nov-Mar months.",
        "temporal_compatibility_note": "Reference CPM uses daily ERA5 anomalies; CMIP6 labels use native monthly means because the processed predictor archive is monthly. The canonical CMIP6 CPM_label is now the pre-April Nov-Mar mean projection.",
        "previous_canonical_definition": "Nov-Apr mean projection on the same fixed ERA5 EOF3 pattern.",
        "label_count": int(len(out)),
    }


def build_aqm_labels_from_cmip6(fixed: AQMFixedLoading) -> tuple[pd.DataFrame, dict]:
    ds = xr.open_dataset(CMIP6_TOS_PATH)
    var_name = "tos"
    da = ds[var_name]

    ndjf = seasonal_average_from_model_years(da, months=(11, 12, 1, 2))
    ndjf = ndjf.interp(lat=fixed.loading_field["lat"], lon=fixed.loading_field["lon"], method="linear")

    label_rows = []
    for model_value in pd.unique(ds["model"].values.astype(str)):
        sub = ndjf.sel(sample=(ds["model"] == model_value))
        years = ds["row_year"].where(ds["model"] == model_value, drop=True).values.astype(int)
        arr = sub.values.reshape(sub.sizes["sample"], -1)
        arr = arr[:, fixed.valid_flat_mask]
        time_index = years - int(years.min())
        detrended, mu, sigma, _, _, keep = linear_fit_and_standardize(arr, time_index)
        standardized = (detrended - mu[None, :]) / sigma[None, :]
        loading_use = fixed.loading_vector[keep]
        if standardized.shape[1] != loading_use.shape[0]:
            raise ValueError(f"AQM reconstructed loading length mismatch for model {model_value}.")
        projected = standardized @ loading_use
        label_rows.append(
            pd.DataFrame(
                {
                    "model_member": model_value,
                    "row_year": years.astype(int),
                    "water_year": years.astype(int) + 1,
                    "AQM_label": projected.astype(float),
                }
            )
        )

    out = pd.concat(label_rows, ignore_index=True)
    out["experiment"] = out["row_year"].map(infer_experiment_from_row_year)
    model_parts = out["model_member"].str.split(":", n=1, expand=True)
    out["source_id"] = model_parts[0]
    out["member_id"] = model_parts[1]
    sample_map = pd.DataFrame(
        {
            "sample": ds["sample"].values.astype(int),
            "model_member": ds["model"].values.astype(str),
            "row_year": ds["row_year"].values.astype(int),
        }
    )
    out = sample_map.merge(out, on=["model_member", "row_year"], how="inner")
    return out, {
        "cmip6_source_path": str(CMIP6_TOS_PATH),
        "cmip6_variable": var_name,
        "cmip6_preprocessing_note": "NDJF seasonal means from CMIP6 tos rows; per-model gridpoint detrending and standardization; projection onto fixed reconstructed observational AQM mode-2 SST loading.",
        "label_count": int(len(out)),
    }


def write_summary(cpm_ref: dict, aqm_ref: dict, cpm_cmip6: dict, aqm_cmip6: dict, label_df: pd.DataFrame) -> None:
    summary = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "cpm_reference": cpm_ref,
        "aqm_reference": aqm_ref,
        "cpm_cmip6": cpm_cmip6,
        "aqm_cmip6": aqm_cmip6,
        "label_table_path": str(ARTIFACT_DIR / "cmip6_auxiliary_labels.csv"),
        "label_count": int(len(label_df)),
        "by_model_counts": label_df.groupby("model_member").size().to_dict(),
        "missing_mpi_ssp370_note": "MPI-ESM1-2-HR:r3i1p1f1 contributes historical rows only because the local ssp370 branch remains unavailable.",
    }
    (ARTIFACT_DIR / "cmip6_auxiliary_labels_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> int:
    ensure_output_dir()

    cpm_reference = validate_cpm_roundtrip()
    aqm_fixed, aqm_reference = reconstruct_aqm_fixed_loading()
    cpm_labels, cpm_meta = build_cpm_labels_from_cmip6()
    aqm_labels, aqm_meta = build_aqm_labels_from_cmip6(aqm_fixed)

    label_df = cpm_labels.merge(
        aqm_labels[["sample", "model_member", "row_year", "water_year", "experiment", "AQM_label"]],
        on=["sample", "model_member", "row_year", "water_year", "experiment"],
        how="inner",
    )
    label_df = label_df[
        ["sample", "source_id", "member_id", "model_member", "experiment", "row_year", "water_year", "CPM_label", "AQM_label"]
    ].sort_values(["source_id", "row_year"]).reset_index(drop=True)
    label_df.to_csv(ARTIFACT_DIR / "cmip6_auxiliary_labels.csv", index=False)

    write_summary(cpm_reference, aqm_reference, cpm_meta, aqm_meta, label_df)
    print(f"Wrote {len(label_df)} rows to {ARTIFACT_DIR / 'cmip6_auxiliary_labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
