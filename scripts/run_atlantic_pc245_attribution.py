#!/usr/bin/env python3
"""
Attribution and interpretation diagnostics for the successful North Atlantic SST modes PC2, PC4, and PC5.
"""

import csv
import gzip
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_amv_amo_swe_baseline import (  # noqa: E402
    LAT_MAX as EOF_LAT_MAX,
    LAT_MIN as EOF_LAT_MIN,
    LON_MAX_360 as EOF_LON_MAX_360,
    LON_MIN_360 as EOF_LON_MIN_360,
    compute_monthly_climatology_anomalies,
    deduplicate_cyclic_longitudes,
)
from scripts.run_sst_monthly_climatology_eof_diagnostics import open_dataset_with_fallbacks  # noqa: E402
from scripts.run_cobe2_global_sst_eof_reproduction import (  # noqa: E402
    COBE2_SST_FILE,
    compute_latitude_sqrt_cos_weights,
)


ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "atlantic_pc245_attribution"
BASELINE_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "amv_amo"
NOAA_AMV_DIR = PROJECT_ROOT / "artifacts" / "noaa_amv_index_loyo"

RAW_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/atlantic_mode_attribution/raw")
PROCESSED_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/atlantic_mode_attribution/processed")
NOAA_AMV_RAW = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/noaa_amv/ersst.v5.amo.dat")

EOF_DATASET = BASELINE_DIR / "amv_amo_cobe2_north_atlantic_eofs_pc1to6.nc"
EOF_SUMMARY = BASELINE_DIR / "amv_amo_cobe2_north_atlantic_pc1to6_summary.json"
PREDICTOR_TABLE = BASELINE_DIR / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"

HADISST_RAW_GZ = RAW_ROOT / "HadISST_sst.nc.gz"
HADISST_PROCESSED_NC = PROCESSED_ROOT / "HadISST_sst.nc"
GPCC_RAW_NC = RAW_ROOT / "precip.mon.total.v2020.nc"
NAO_RAW = RAW_ROOT / "nao.data"
STONE_SUPPLEMENT_PDF = RAW_ROOT / "Stone_2023_s41612-023-00471-7_supplement.pdf"

README_PATH = ARTIFACT_DIR / "README.md"
METHOD_RECON_PATH = ARTIFACT_DIR / "method_reconstruction.md"
PROVENANCE_CSV = ARTIFACT_DIR / "reference_pattern_provenance.csv"
MODE_SUMMARY_CSV = ARTIFACT_DIR / "pc245_mode_summary.csv"
SPATIAL_CSV = ARTIFACT_DIR / "pc245_spatial_similarity.csv"
TEMPORAL_CSV = ARTIFACT_DIR / "pc245_temporal_similarity.csv"
TREND_CSV = ARTIFACT_DIR / "pc245_trend_and_basin_diagnostics.csv"
DECOMP_CSV = ARTIFACT_DIR / "pc245_reference_decomposition.csv"
DECISION_MD = ARTIFACT_DIR / "pc245_attribution_decision.md"
METADATA_JSON = ARTIFACT_DIR / "run_metadata.json"

EOF2_PNG = ARTIFACT_DIR / "eof2_map.png"
EOF4_PNG = ARTIFACT_DIR / "eof4_map.png"
EOF5_PNG = ARTIFACT_DIR / "eof5_map.png"
AQM_PNG = ARTIFACT_DIR / "reference_aqm_map.png"
AMV_PNG = ARTIFACT_DIR / "reference_amv_map.png"
TRIPOLE_PNG = ARTIFACT_DIR / "reference_tripole_map.png"
SUBPOLAR_PNG = ARTIFACT_DIR / "reference_subpolar_map.png"

PC2_COMPARE_PNG = ARTIFACT_DIR / "pc2_reference_comparison.png"
PC4_COMPARE_PNG = ARTIFACT_DIR / "pc4_reference_comparison.png"
PC5_COMPARE_PNG = ARTIFACT_DIR / "pc5_reference_comparison.png"
SPATIAL_HEATMAP_PNG = ARTIFACT_DIR / "pc245_spatial_similarity_heatmap.png"
TEMPORAL_HEATMAP_PNG = ARTIFACT_DIR / "pc245_temporal_similarity_heatmap.png"
RESIDUALS_PNG = ARTIFACT_DIR / "pc245_best_match_residuals.png"
TIMESERIES_PNG = ARTIFACT_DIR / "pc245_reference_timeseries.png"

TARGET_NETCDF = ARTIFACT_DIR / "target_eof_patterns.nc"
REFERENCE_RAW_NETCDF = ARTIFACT_DIR / "reference_patterns_raw.nc"
REFERENCE_HARMONIZED_NETCDF = ARTIFACT_DIR / "reference_patterns_harmonized.nc"
PC_SERIES_CSV = ARTIFACT_DIR / "pc245_long_monthly_timeseries.csv"
REFERENCE_INDEX_CSV = ARTIFACT_DIR / "reference_indices_monthly_and_seasonal.csv"

TARGET_MODES = [2, 4, 5]
TARGET_LABELS = {2: "PC2", 4: "PC4", 5: "PC5"}
TARGET_MONTH_PREDICTORS = {
    "PC4": ["Sep", "Nov"],
    "PC5": ["Feb", "Mar"],
    "PC2": ["Feb"],
}

SEASON_MONTHS_SST = [11, 12, 1, 2]
SEASON_MONTHS_PRECIP = [12, 1, 2, 3]
AQM_SST_LAT_MIN = -10.0
AQM_SST_LAT_MAX = 70.0
AQM_PRECIP_LAT_MIN = 30.0
AQM_PRECIP_LAT_MAX = 50.0
AQM_PRECIP_LON_MIN = 235.0
AQM_PRECIP_LON_MAX = 260.0

SUBPOLAR_BOX = {"lat_min": 50.0, "lat_max": 65.0, "lon_min_360": 300.0, "lon_max_360": 350.0}


@dataclass
class PatternField:
    name: str
    label: str
    data: xr.DataArray
    source_kind: str
    source_description: str
    preprocessing: str
    sign_meaning: str


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_compute_runtime() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside the interactive compute-node allocation.")


def parse_noaa_psl_index(path: Path, name: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if not parts:
                continue
            if parts[0].startswith("#"):
                continue
            if parts[0].lower() in {"year", "ersst", "missing", "index:", "month", "ssta"}:
                continue
            try:
                if len(parts) == 3:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[2])
                    if value > -90.0:
                        rows.append({"date": pd.Timestamp(year=year, month=month, day=1), name: value})
                    continue
                if len(parts) >= 13:
                    year = int(parts[0])
                    values = [float(v) for v in parts[1:13]]
                    for month, value in enumerate(values, start=1):
                        if value <= -90.0:
                            continue
                        rows.append(
                            {
                                "date": pd.Timestamp(year=year, month=month, day=1),
                                name: float(value),
                            }
                        )
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No monthly rows were parsed from {path}")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    return float(np.sqrt(np.sum(weights * (values - mu) ** 2) / np.sum(weights)))


def monthly_climatology_anomalies_from_da(da: xr.DataArray) -> xr.DataArray:
    clim = da.groupby("time.month").mean("time", skipna=True)
    return da.groupby("time.month") - clim


def normalize_longitude_360(da: xr.DataArray, lon_name: str = "lon") -> xr.DataArray:
    lon360 = np.mod(da[lon_name].values.astype(float), 360.0)
    order = np.argsort(lon360)
    da = da.isel({lon_name: order}).assign_coords({lon_name: lon360[order]})
    lon_unique, keep = np.unique(np.round(da[lon_name].values.astype(float), 8), return_index=True)
    del lon_unique
    return da.isel({lon_name: np.sort(keep)})


def latitude_slice(da: xr.DataArray, lat_min: float, lat_max: float) -> xr.DataArray:
    lat_values = da["lat"].values.astype(float)
    if lat_values[0] > lat_values[-1]:
        return da.sel(lat=slice(lat_max, lat_min))
    return da.sel(lat=slice(lat_min, lat_max))


def parse_coords(ds: xr.Dataset, lat_names: Sequence[str], lon_names: Sequence[str]) -> Tuple[str, str]:
    lat_name = next((name for name in lat_names if name in ds.coords or name in ds.variables), None)
    lon_name = next((name for name in lon_names if name in ds.coords or name in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise KeyError("Unable to locate latitude/longitude coordinates")
    return lat_name, lon_name


def ensure_hadisst_processed() -> None:
    if HADISST_PROCESSED_NC.exists():
        return
    with gzip.open(HADISST_RAW_GZ, "rb") as src, HADISST_PROCESSED_NC.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def load_hadisst_sst() -> xr.DataArray:
    ensure_hadisst_processed()
    ds = xr.open_dataset(HADISST_PROCESSED_NC)
    lat_name, lon_name = parse_coords(ds, ("latitude", "lat"), ("longitude", "lon"))
    var_name = "sst"
    da = ds[var_name]
    da = da.rename({lat_name: "lat", lon_name: "lon"})
    da = normalize_longitude_360(da)
    da = da.where(da < 1.0e20)
    return da


def load_gpcc_precip() -> xr.DataArray:
    ds = xr.open_dataset(GPCC_RAW_NC)
    lat_name, lon_name = parse_coords(ds, ("lat", "latitude"), ("lon", "longitude"))
    var_name = next(name for name in ds.data_vars if "precip" in name.lower())
    da = ds[var_name]
    da = da.rename({lat_name: "lat", lon_name: "lon"})
    da = normalize_longitude_360(da)
    return da


def seasonal_average_label_by_year(da: xr.DataArray, months: Sequence[int], label_year_month: int) -> xr.DataArray:
    parts = []
    for year in np.unique(da["time"].dt.year.values):
        selected = []
        for month in months:
            source_year = int(year) if month <= label_year_month else int(year) - 1
            subset = da.sel(time=((da["time"].dt.year == source_year) & (da["time"].dt.month == month)))
            if subset.sizes.get("time", 0) != 1:
                selected = []
                break
            selected.append(subset.isel(time=0))
        if not selected:
            continue
        season_mean = xr.concat(selected, dim="stacked_month").mean("stacked_month", skipna=True)
        season_mean = season_mean.expand_dims(time=[pd.Timestamp(year=int(year), month=label_year_month, day=1)])
        parts.append(season_mean)
    if not parts:
        raise ValueError("No seasonal means were constructed")
    return xr.concat(parts, dim="time")


def detrend_1d(y: np.ndarray) -> np.ndarray:
    x = np.arange(y.size, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan, dtype=float)
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    out = y.copy().astype(float)
    out[mask] = y[mask] - (slope * x[mask] + intercept)
    return out


def detrend_along_time(da: xr.DataArray) -> xr.DataArray:
    result = xr.apply_ufunc(
        detrend_1d,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )
    return result.transpose(*da.dims)


def standardize_along_time(da: xr.DataArray) -> xr.DataArray:
    mu = da.mean("time", skipna=True)
    sigma = da.std("time", skipna=True)
    return (da - mu) / sigma


def corr_with_series(field: xr.DataArray, series: np.ndarray) -> xr.DataArray:
    x = field.values
    y = np.asarray(series, dtype=float)
    out = np.full(x.shape[1:], np.nan, dtype=float)
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            vals = x[:, i, j]
            mask = np.isfinite(vals) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            if np.std(vals[mask], ddof=1) == 0.0 or np.std(y[mask], ddof=1) == 0.0:
                continue
            out[i, j] = np.corrcoef(vals[mask], y[mask])[0, 1]
    return xr.DataArray(out, coords={"lat": field["lat"], "lon": field["lon"]}, dims=("lat", "lon"))


def regress_with_series(field: xr.DataArray, series: np.ndarray) -> xr.DataArray:
    x = field.values
    y = np.asarray(series, dtype=float)
    y_std = np.nanstd(y, ddof=1)
    out = np.full(x.shape[1:], np.nan, dtype=float)
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            vals = x[:, i, j]
            mask = np.isfinite(vals) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            yy = y[mask]
            vv = vals[mask]
            if np.std(yy, ddof=1) == 0.0:
                continue
            slope, _, _, _, _ = stats.linregress(yy, vv)
            out[i, j] = slope * y_std
    return xr.DataArray(out, coords={"lat": field["lat"], "lon": field["lon"]}, dims=("lat", "lon"))


def area_weighted_index(field: xr.DataArray, valid_mask: Optional[xr.DataArray] = None) -> pd.DataFrame:
    if valid_mask is not None:
        field = field.where(valid_mask)
    weights = np.cos(np.deg2rad(field["lat"]))
    index = field.weighted(weights).mean(("lat", "lon"), skipna=True)
    return pd.DataFrame({"date": pd.to_datetime(index["time"].values), "value": index.values.astype(float)})


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> str:
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1.0:
        return ""
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return "[{:.3f}, {:.3f}]".format(float(lo), float(hi))


def bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not np.any(mask):
        return out
    vals = p[mask]
    order = np.argsort(vals)
    ranked = vals[order]
    n = ranked.size
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        curr = min(prev, ranked[i] * n / rank)
        adj[i] = curr
        prev = curr
    restored = np.empty(n, dtype=float)
    restored[order] = adj
    out[mask] = restored
    return out


def load_eof_targets() -> Tuple[xr.Dataset, Dict[str, object], pd.DataFrame]:
    ds = xr.open_dataset(EOF_DATASET)
    summary = json.loads(EOF_SUMMARY.read_text())
    table = pd.read_csv(PREDICTOR_TABLE)
    return ds, summary, table


def load_cobe2_global_sst() -> xr.DataArray:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        sst = ds["sst"].load()
    sst = sst.rename({"lat": "lat", "lon": "lon"})
    sst = normalize_longitude_360(sst)
    sst = sst.where(sst < 1.0e20)
    return sst


def cobe2_eof_domain_anomalies(global_sst: xr.DataArray) -> xr.DataArray:
    domain = latitude_slice(global_sst, EOF_LAT_MIN, EOF_LAT_MAX).sel(lon=slice(EOF_LON_MIN_360, EOF_LON_MAX_360 - 1.0e-9))
    return monthly_climatology_anomalies_from_da(domain)


def project_pcs_from_fixed_eofs(anoms: xr.DataArray, eof_ds: xr.Dataset) -> pd.DataFrame:
    valid_mask = eof_ds["valid_mask"].astype(bool).values
    lat = eof_ds["lat"].values.astype(float)
    lon = eof_ds["lon"].values.astype(float)
    eof = eof_ds["eof"].values.astype(float)
    anom_on_eof_grid = anoms.interp(lat=lat, lon=lon, method="linear")
    anom_vals = anom_on_eof_grid.values.astype(float)
    weights = compute_latitude_sqrt_cos_weights(lat)
    weights_2d = np.broadcast_to(weights[:, None], valid_mask.shape)
    flat_mask = valid_mask.reshape(-1)
    flat_weights = weights_2d.reshape(-1)[flat_mask]
    x = anom_vals.reshape(anom_vals.shape[0], -1)[:, flat_mask]
    pc_map = {}
    for mode in TARGET_MODES:
        field = eof[mode - 1].reshape(-1)[flat_mask]
        weighted_eof = field * flat_weights
        pc_map[mode] = np.nansum(x * weighted_eof[None, :], axis=1)
    out = pd.DataFrame({"date": pd.to_datetime(anoms["time"].values)})
    for mode in TARGET_MODES:
        out[f"PC{mode}"] = pc_map[mode]
    return out


def reproduce_aqm(hadisst: xr.DataArray, gpcc: xr.DataArray) -> Tuple[PatternField, pd.DataFrame]:
    sst = latitude_slice(hadisst.sel(time=slice("1890-11-01", "2019-02-01")), AQM_SST_LAT_MIN, AQM_SST_LAT_MAX)
    precip = latitude_slice(gpcc.sel(
        time=slice("1890-12-01", "2019-03-01"),
        lon=slice(AQM_PRECIP_LON_MIN, AQM_PRECIP_LON_MAX),
    ), AQM_PRECIP_LAT_MIN, AQM_PRECIP_LAT_MAX)
    sst_season = seasonal_average_label_by_year(sst, SEASON_MONTHS_SST, 3)
    precip_season = seasonal_average_label_by_year(precip, SEASON_MONTHS_PRECIP, 3)
    common_years = np.intersect1d(
        sst_season["time"].dt.year.values.astype(int),
        precip_season["time"].dt.year.values.astype(int),
    )
    sst_season = sst_season.sel(time=sst_season["time"].dt.year.isin(common_years))
    precip_season = precip_season.sel(time=precip_season["time"].dt.year.isin(common_years))
    sst_dt = detrend_along_time(sst_season)
    precip_dt = detrend_along_time(precip_season)
    x_raw = sst_dt.values.reshape(sst_dt.sizes["time"], -1)
    y_raw = precip_dt.values.reshape(precip_dt.sizes["time"], -1)
    xmask = np.isfinite(x_raw).all(axis=0)
    ymask = np.isfinite(y_raw).all(axis=0)
    x = x_raw[:, xmask]
    y = y_raw[:, ymask]
    x_std = np.std(x, axis=0, ddof=1, keepdims=True)
    y_std = np.std(y, axis=0, ddof=1, keepdims=True)
    x_keep = np.isfinite(x_std[0]) & (x_std[0] > 0.0)
    y_keep = np.isfinite(y_std[0]) & (y_std[0] > 0.0)
    x = x[:, x_keep]
    y = y[:, y_keep]
    x_std = x_std[:, x_keep]
    y_std = y_std[:, y_keep]
    x = (x - np.mean(x, axis=0, keepdims=True)) / x_std
    y = (y - np.mean(y, axis=0, keepdims=True)) / y_std
    finite_x = np.isfinite(x).all(axis=0)
    finite_y = np.isfinite(y).all(axis=0)
    x = x[:, finite_x]
    y = y[:, finite_y]
    if x.shape[1] < 2 or y.shape[1] < 2:
        raise ValueError("AQM reproduction retained too few valid SST or precipitation cells after masking.")
    cxy = (x.T @ y) / float(x.shape[0] - 1)
    u, svals, vt = np.linalg.svd(cxy, full_matrices=False)
    s2 = x @ u[:, 1]
    aqm_map = corr_with_series(sst_dt, s2)
    index_df = pd.DataFrame(
        {
            "date": pd.to_datetime(sst_season["time"].values),
            "AQM_S2_NDJF": s2.astype(float),
            "AQM_SCF_mode2": float((svals[1] ** 2) / np.sum(svals ** 2)),
        }
    )
    return (
        PatternField(
            name="AQM",
            label="Atlantic Quadpole Mode",
            data=aqm_map,
            source_kind="reproduced_from_published_method",
            source_description="Stone et al. 2023 AQM reproduced from HadISST v3 and GPCC monthly precipitation using lagged MCA.",
            preprocessing="Nov-Feb SST and Dec-Mar precipitation seasonal means, linear detrending at each grid point, grid-point standardization, MCA mode 2 homogeneous SST correlation map.",
            sign_meaning="Positive values indicate the Warm AQM orientation defined by positive S2.",
        ),
        index_df,
    )


def build_conventional_amv(hadisst: xr.DataArray) -> Tuple[PatternField, pd.DataFrame]:
    monthly = latitude_slice(hadisst.sel(time=slice("1891-01-01", "2019-12-01"), lon=slice(280.0, 360.0)), 0.0, 70.0)
    monthly_anom = monthly_climatology_anomalies_from_da(monthly)
    amv_monthly = area_weighted_index(monthly_anom)
    amv_monthly["AMV_HadISST_monthly"] = detrend_1d(amv_monthly["value"].to_numpy(dtype=float))
    amv_monthly["AMV_HadISST_monthly_lowpass121"] = (
        pd.Series(amv_monthly["AMV_HadISST_monthly"]).rolling(121, center=True, min_periods=61).mean().to_numpy()
    )
    season = seasonal_average_label_by_year(monthly, SEASON_MONTHS_SST, 3)
    season_anom = seasonal_average_label_by_year(monthly_anom, SEASON_MONTHS_SST, 3)
    season_index = area_weighted_index(season_anom)
    season_index["AMV_HadISST_NDJF"] = detrend_1d(season_index["value"].to_numpy(dtype=float))
    season_index["AMV_HadISST_NDJF_lowpass"] = (
        pd.Series(season_index["AMV_HadISST_NDJF"]).rolling(11, center=True, min_periods=5).mean().to_numpy()
    )
    amv_map = regress_with_series(
        season_anom,
        season_index["AMV_HadISST_NDJF_lowpass"].to_numpy(dtype=float),
    )
    return (
        PatternField(
            name="Conventional_AMV",
            label="Conventional AMV",
            data=amv_map,
            source_kind="reproduced_from_published_method",
            source_description="HadISST-based detrended North Atlantic area-mean AMV reference, low-pass filtered before regression.",
            preprocessing="Monthly climatology anomalies on 1891-2019 HadISST, 0-70N 80W-0 Atlantic average, linear detrending, NDJF seasonal averaging, 11-year low-pass, regression map.",
            sign_meaning="Positive values correspond to the warm basin-scale AMV phase.",
        ),
        pd.merge(
            amv_monthly[["date", "AMV_HadISST_monthly", "AMV_HadISST_monthly_lowpass121"]],
            season_index[["date", "AMV_HadISST_NDJF", "AMV_HadISST_NDJF_lowpass"]],
            on="date",
            how="outer",
        ).sort_values("date"),
    )


def build_tripole(hadisst: xr.DataArray, nao_df: pd.DataFrame) -> Tuple[PatternField, pd.DataFrame]:
    monthly = latitude_slice(hadisst.sel(time=slice("1891-01-01", "2019-12-01"), lon=slice(280.0, 360.0)), 0.0, 70.0)
    monthly_anom = monthly_climatology_anomalies_from_da(monthly)
    sst_season = seasonal_average_label_by_year(monthly_anom, [12, 1, 2, 3], 3)
    nao = nao_df.copy()
    nao = nao[(nao["date"] >= pd.Timestamp("1891-01-01")) & (nao["date"] <= pd.Timestamp("2019-12-01"))]
    nao["year"] = nao["date"].dt.year
    nao["month"] = nao["date"].dt.month
    rows = []
    for year in sorted(np.unique(sst_season["time"].dt.year.values.astype(int))):
        want = [(year - 1, 12), (year, 1), (year, 2), (year, 3)]
        vals = []
        for yy, mm in want:
            sub = nao[(nao["year"] == yy) & (nao["month"] == mm)]["NAO"]
            if sub.empty:
                vals = []
                break
            vals.append(float(sub.iloc[0]))
        if vals:
            rows.append({"date": pd.Timestamp(year=year, month=3, day=1), "NAO_DJFM": float(np.mean(vals))})
    nao_season = pd.DataFrame(rows)
    merged = nao_season.merge(
        pd.DataFrame({"date": pd.to_datetime(sst_season["time"].values)}),
        on="date",
        how="inner",
    )
    sst_use = sst_season.sel(time=sst_season["time"].dt.year.isin(merged["date"].dt.year.values))
    tripole_map = regress_with_series(sst_use, detrend_1d(merged["NAO_DJFM"].to_numpy(dtype=float)))
    return (
        PatternField(
            name="NAO_Tripole",
            label="North Atlantic SST tripole",
            data=tripole_map,
            source_kind="reproduced_from_official_index",
            source_description="HadISST North Atlantic SST anomalies regressed on the official NOAA PSL monthly NAO index averaged over DJFM.",
            preprocessing="Monthly climatology anomalies, DJFM seasonal averaging, no additional spatial weighting before regression, official NOAA PSL NAO seasonal mean detrended before regression.",
            sign_meaning="Positive values correspond to the positive phase of the seasonal NAO index.",
        ),
        nao_season,
    )


def build_subpolar(hadisst: xr.DataArray) -> Tuple[PatternField, pd.DataFrame]:
    monthly = latitude_slice(hadisst.sel(time=slice("1891-01-01", "2019-12-01"), lon=slice(280.0, 360.0)), 0.0, 70.0)
    monthly_anom = monthly_climatology_anomalies_from_da(monthly)
    sub_box = latitude_slice(monthly_anom.sel(
        lon=slice(SUBPOLAR_BOX["lon_min_360"], SUBPOLAR_BOX["lon_max_360"]),
    ), SUBPOLAR_BOX["lat_min"], SUBPOLAR_BOX["lat_max"])
    sub_idx = area_weighted_index(sub_box)
    sub_idx["Subpolar_SST_monthly"] = detrend_1d(sub_idx["value"].to_numpy(dtype=float))
    subpolar_map = regress_with_series(monthly_anom, sub_idx["Subpolar_SST_monthly"].to_numpy(dtype=float))
    return (
        PatternField(
            name="Subpolar_Gyre_SST",
            label="Subpolar gyre SST signal",
            data=subpolar_map,
            source_kind="reproducibly_defined_regional_index",
            source_description="HadISST North Atlantic SST anomalies regressed on a detrended subpolar box-area-mean SST anomaly index.",
            preprocessing="Monthly climatology anomalies, subpolar box 50-65N and 60W-10W, detrended monthly index, monthly regression map.",
            sign_meaning="Positive values correspond to a warm subpolar gyre SST anomaly.",
        ),
        sub_idx[["date", "Subpolar_SST_monthly"]],
    )


def build_trend_pattern(hadisst: xr.DataArray) -> PatternField:
    monthly = latitude_slice(hadisst.sel(time=slice("1891-01-01", "2019-12-01"), lon=slice(280.0, 360.0)), 0.0, 70.0)
    years = np.arange(monthly.sizes["time"], dtype=float)
    vals = monthly.values.astype(float)
    out = np.full(vals.shape[1:], np.nan, dtype=float)
    for i in range(vals.shape[1]):
        for j in range(vals.shape[2]):
            cell = vals[:, i, j]
            mask = np.isfinite(cell)
            if mask.sum() < 3:
                continue
            slope, _, _, _, _ = stats.linregress(years[mask], cell[mask])
            out[i, j] = slope * 12.0 * 100.0
    return PatternField(
        name="Trend_Pattern",
        label="Linear warming trend",
        data=xr.DataArray(out, coords={"lat": monthly["lat"], "lon": monthly["lon"]}, dims=("lat", "lon")),
        source_kind="computed_from_dataset",
        source_description="Linear SST trend map from HadISST over 1891-2019 on the North Atlantic domain.",
        preprocessing="Direct linear trend at each grid point in C per century on monthly HadISST SST values.",
        sign_meaning="Positive values indicate local warming trends.",
    )


def harmonize_to_cobe2(pattern: PatternField, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    data = pattern.data
    if "time" in data.dims:
        raise ValueError("Pattern fields must be spatial maps only")
    data = latitude_slice(data.sel(lon=slice(EOF_LON_MIN_360, EOF_LON_MAX_360)), EOF_LAT_MIN, EOF_LAT_MAX)
    return data.interp(lat=target_lat, lon=target_lon, method="linear")


def spatial_metrics(
    target: xr.DataArray, reference: xr.DataArray, valid_mask: xr.DataArray
) -> Dict[str, float]:
    lat2d = np.broadcast_to(target["lat"].values[:, None], target.shape)
    weights = np.cos(np.deg2rad(lat2d))
    mask = np.isfinite(target.values) & np.isfinite(reference.values) & valid_mask.values.astype(bool)
    a = target.values[mask].astype(float)
    b = reference.values[mask].astype(float)
    w = weights[mask].astype(float)
    a_mean = weighted_mean(a, w)
    b_mean = weighted_mean(b, w)
    a_std = weighted_std(a, w)
    b_std = weighted_std(b, w)
    signed_r = float(np.sum(w * (a - a_mean) * (b - b_mean)) / np.sqrt(np.sum(w * (a - a_mean) ** 2) * np.sum(w * (b - b_mean) ** 2)))
    congruence = float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b)))
    opt_sign = 1.0 if signed_r >= 0.0 else -1.0
    az = (a - a_mean) / a_std
    bz = (b - b_mean) / b_std
    nrmse = float(np.sqrt(np.sum(w * (az - opt_sign * bz) ** 2) / np.sum(w)))
    a_dm = a - a_mean
    b_dm = b - b_mean
    demeaned_r = float(np.sum(w * a_dm * b_dm) / np.sqrt(np.sum(w * a_dm * a_dm) * np.sum(w * b_dm * b_dm)))
    return {
        "signed_spatial_r": signed_r,
        "absolute_spatial_r": abs(signed_r),
        "demeaned_spatial_r": demeaned_r,
        "pattern_congruence": congruence,
        "normalized_rmse": nrmse,
        "optimal_sign": opt_sign,
        "valid_grid_cells": int(mask.sum()),
    }


def correlation_row(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), n, float("nan")
    result = stats.pearsonr(x[mask], y[mask])
    return float(result.statistic), n, float(result.pvalue)


def render_map(field: xr.DataArray, title: str, out_path: Path, cmap: str = "RdBu_r", subtitle: Optional[str] = None) -> None:
    vmax = np.nanpercentile(np.abs(field.values), 98)
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    mesh = ax.pcolormesh(field["lon"], field["lat"], field, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    if subtitle:
        ax.text(0.01, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def comparison_panel(
    target_name: str,
    target_map: xr.DataArray,
    candidates: Sequence[Tuple[str, xr.DataArray, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(candidates) + 1, figsize=(4.8 * (len(candidates) + 1), 4.6))
    vmax = np.nanpercentile(np.abs(target_map.values), 98)
    for _, ref_map, _ in candidates:
        vmax = max(vmax, np.nanpercentile(np.abs(ref_map.values), 98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    panels = [(f"{target_name}", target_map, None)] + [(name, ref_map, score) for name, ref_map, score in candidates]
    for ax, (title, field, score) in zip(axes, panels):
        mesh = ax.pcolormesh(field["lon"], field["lat"], field, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title if score is None else f"{title}\n|r|={score:.3f}")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        fig.colorbar(mesh, ax=ax, shrink=0.84, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def heatmap(df: pd.DataFrame, row: str, col: str, value: str, out_path: Path, title: str) -> None:
    pivot = df.pivot(index=row, columns=col, values=value).sort_index()
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    mesh = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_compute_runtime()
    ensure_dirs()

    eof_ds, eof_summary, predictor_table = load_eof_targets()
    hadisst = load_hadisst_sst()
    gpcc = load_gpcc_precip()
    nao_monthly = parse_noaa_psl_index(NAO_RAW, "NAO")
    noaa_amv_monthly = parse_noaa_psl_index(NOAA_AMV_RAW, "NOAA_AMV")
    cobe2_global = load_cobe2_global_sst()
    cobe2_anom_domain = cobe2_eof_domain_anomalies(cobe2_global)
    projected_pcs = project_pcs_from_fixed_eofs(cobe2_anom_domain, eof_ds)
    projected_pcs.to_csv(PC_SERIES_CSV, index=False)

    target_maps = {}
    for mode in TARGET_MODES:
        field = eof_ds["eof"].sel(mode=mode).load()
        if "mode" in field.coords:
            field = field.drop_vars("mode")
        target_maps[mode] = field
    xr.Dataset({f"EOF{mode}": target_maps[mode] for mode in TARGET_MODES}).to_netcdf(TARGET_NETCDF)

    aqm_pattern, aqm_index = reproduce_aqm(hadisst, gpcc)
    amv_pattern, amv_index = build_conventional_amv(hadisst)
    tripole_pattern, nao_season = build_tripole(hadisst, nao_monthly)
    subpolar_pattern, subpolar_index = build_subpolar(hadisst)
    trend_pattern = build_trend_pattern(hadisst)
    reference_patterns = [aqm_pattern, amv_pattern, tripole_pattern, subpolar_pattern, trend_pattern]

    raw_ds = xr.Dataset({pat.name: pat.data for pat in reference_patterns})
    raw_ds.to_netcdf(REFERENCE_RAW_NETCDF)

    target_lat = eof_ds["lat"].values.astype(float)
    target_lon = eof_ds["lon"].values.astype(float)
    valid_mask = eof_ds["valid_mask"].astype(bool)
    harmonized = {pat.name: harmonize_to_cobe2(pat, target_lat, target_lon) for pat in reference_patterns}
    xr.Dataset({name: da for name, da in harmonized.items()}).to_netcdf(REFERENCE_HARMONIZED_NETCDF)

    render_map(target_maps[2], "EOF2 (PC2 spatial pattern)", EOF2_PNG, subtitle=f"Explained variance = {eof_summary['explained_variance_ratio_mode1to6'][1]:.4f}")
    render_map(target_maps[4], "EOF4 (PC4 spatial pattern)", EOF4_PNG, subtitle=f"Explained variance = {eof_summary['explained_variance_ratio_mode1to6'][3]:.4f}")
    render_map(target_maps[5], "EOF5 (PC5 spatial pattern)", EOF5_PNG, subtitle=f"Explained variance = {eof_summary['explained_variance_ratio_mode1to6'][4]:.4f}")
    render_map(harmonized["AQM"], "Reference AQM pattern", AQM_PNG)
    render_map(harmonized["Conventional_AMV"], "Reference conventional AMV pattern", AMV_PNG)
    render_map(harmonized["NAO_Tripole"], "Reference NAO-tripole pattern", TRIPOLE_PNG)
    render_map(harmonized["Subpolar_Gyre_SST"], "Reference subpolar SST pattern", SUBPOLAR_PNG)

    spatial_rows = []
    for mode in TARGET_MODES:
        target_name = f"EOF{mode}"
        for pat in reference_patterns:
            metrics = spatial_metrics(target_maps[mode], harmonized[pat.name], valid_mask)
            spatial_rows.append({"target_mode": target_name, "reference_mode": pat.name, **metrics})
    spatial_df = pd.DataFrame(spatial_rows).sort_values(["target_mode", "absolute_spatial_r"], ascending=[True, False])
    spatial_df.to_csv(SPATIAL_CSV, index=False)

    best_refs = {}
    for mode in TARGET_MODES:
        sub = spatial_df[spatial_df["target_mode"] == f"EOF{mode}"].sort_values("absolute_spatial_r", ascending=False)
        top = []
        for _, row in sub.head(2).iterrows():
            top.append((row["reference_mode"], harmonized[row["reference_mode"]], float(row["absolute_spatial_r"])))
        best_refs[mode] = top[0][0]
        comparison_panel(f"EOF{mode}", target_maps[mode], top, {2: PC2_COMPARE_PNG, 4: PC4_COMPARE_PNG, 5: PC5_COMPARE_PNG}[mode])

    residual_fig, residual_axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, mode in zip(residual_axes, TARGET_MODES):
        ref_name = best_refs[mode]
        sign = float(
            spatial_df[
                (spatial_df["target_mode"] == f"EOF{mode}") & (spatial_df["reference_mode"] == ref_name)
            ]["optimal_sign"].iloc[0]
        )
        a = target_maps[mode].where(valid_mask).values
        b = harmonized[ref_name].where(valid_mask).values
        lat2d = np.broadcast_to(target_maps[mode]["lat"].values[:, None], a.shape)
        mask = np.isfinite(a) & np.isfinite(b)
        w = np.cos(np.deg2rad(lat2d[mask]))
        az = (a - weighted_mean(a[mask], w)) / weighted_std(a[mask], w)
        bz = (b - weighted_mean(b[mask], w)) / weighted_std(b[mask], w)
        resid = np.full_like(a, np.nan, dtype=float)
        resid[mask] = az[mask] - sign * bz[mask]
        vmax = np.nanpercentile(np.abs(resid), 98)
        mesh = ax.pcolormesh(target_maps[mode]["lon"], target_maps[mode]["lat"], resid, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"EOF{mode} - {ref_name}")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        residual_fig.colorbar(mesh, ax=ax, shrink=0.84, pad=0.01)
    residual_fig.tight_layout()
    residual_fig.savefig(RESIDUALS_PNG, dpi=220)
    plt.close(residual_fig)

    reference_index_df = (
        projected_pcs.merge(noaa_amv_monthly, on="date", how="left")
        .merge(nao_monthly, on="date", how="left")
        .merge(subpolar_index, on="date", how="left")
        .merge(amv_index, on="date", how="left")
    )
    aqm_index = aqm_index.rename(columns={"AQM_S2_NDJF": "AQM_S2_NDJF"})
    nao_season = nao_season.rename(columns={"NAO_DJFM": "NAO_DJFM"})
    reference_index_df = reference_index_df.merge(aqm_index[["date", "AQM_S2_NDJF"]], on="date", how="left")
    reference_index_df = reference_index_df.merge(nao_season[["date", "NAO_DJFM"]], on="date", how="left")

    cobe2_global_anom = monthly_climatology_anomalies_from_da(cobe2_global.sel(time=slice(str(projected_pcs["date"].min().date()), str(projected_pcs["date"].max().date()))))
    basin_mean = area_weighted_index(cobe2_anom_domain.sel(time=projected_pcs["date"].values))
    basin_mean = basin_mean.rename(columns={"value": "NorthAtlantic_basin_mean"})
    global_mean = area_weighted_index(cobe2_global_anom.sel(time=projected_pcs["date"].values))
    global_mean = global_mean.rename(columns={"value": "Global_mean_SST"})
    reference_index_df = reference_index_df.merge(basin_mean, on="date", how="left").merge(global_mean, on="date", how="left")
    reference_index_df.to_csv(REFERENCE_INDEX_CSV, index=False)

    temporal_rows = []
    monthly_refs = ["NOAA_AMV", "NAO", "Subpolar_SST_monthly", "NorthAtlantic_basin_mean", "Global_mean_SST"]
    for mode in TARGET_MODES:
        pc_name = f"PC{mode}"
        for ref_name in monthly_refs:
            sub = reference_index_df[[pc_name, ref_name]].dropna()
            r, n, p = correlation_row(sub[pc_name].to_numpy(dtype=float), sub[ref_name].to_numpy(dtype=float))
            temporal_rows.append(
                {
                    "target_pc": pc_name,
                    "target_month_or_season": "all_months",
                    "reference_index": ref_name,
                    "lag": 0,
                    "correlation": r,
                    "sample_size": n,
                    "p_value": p,
                    "confidence_interval": fisher_ci(r, n),
                }
            )
    for label, month in [("Sep", 9), ("Nov", 11), ("Feb", 2), ("Mar", 3)]:
        sub_month = reference_index_df[reference_index_df["date"].dt.month == month]
        for mode in TARGET_MODES:
            pc_name = f"PC{mode}"
            for ref_name in monthly_refs:
                sub = sub_month[[pc_name, ref_name]].dropna()
                r, n, p = correlation_row(sub[pc_name].to_numpy(dtype=float), sub[ref_name].to_numpy(dtype=float))
                temporal_rows.append(
                    {
                        "target_pc": pc_name,
                        "target_month_or_season": f"{label}_only",
                        "reference_index": ref_name,
                        "lag": 0,
                        "correlation": r,
                        "sample_size": n,
                        "p_value": p,
                        "confidence_interval": fisher_ci(r, n),
                    }
                )
    seasonal_pc_rows = []
    for mode in TARGET_MODES:
        series = reference_index_df[["date", f"PC{mode}"]].dropna().copy()
        series["year"] = series["date"].dt.year
        series["month"] = series["date"].dt.month
        records = []
        for year in sorted(reference_index_df["date"].dt.year.unique()):
            want = [(year - 1, 11), (year - 1, 12), (year, 1), (year, 2)]
            vals = []
            for yy, mm in want:
                sub = series[(series["year"] == yy) & (series["month"] == mm)][f"PC{mode}"]
                if sub.empty:
                    vals = []
                    break
                vals.append(float(sub.iloc[0]))
            if vals:
                records.append({"date": pd.Timestamp(year=int(year), month=3, day=1), f"PC{mode}_NDJF": float(np.mean(vals))})
        seasonal_pc_rows.append(pd.DataFrame(records))
    seasonal_merged = seasonal_pc_rows[0]
    for extra in seasonal_pc_rows[1:]:
        seasonal_merged = seasonal_merged.merge(extra, on="date", how="outer")
    seasonal_merged = seasonal_merged.merge(aqm_index[["date", "AQM_S2_NDJF"]], on="date", how="inner")
    for mode in TARGET_MODES:
        pc_col = f"PC{mode}_NDJF"
        sub = seasonal_merged[[pc_col, "AQM_S2_NDJF"]].dropna()
        r, n, p = correlation_row(sub[pc_col].to_numpy(dtype=float), sub["AQM_S2_NDJF"].to_numpy(dtype=float))
        temporal_rows.append(
            {
                "target_pc": f"PC{mode}",
                "target_month_or_season": "NDJF_mean",
                "reference_index": "AQM_S2_NDJF",
                "lag": 0,
                "correlation": r,
                "sample_size": n,
                "p_value": p,
                "confidence_interval": fisher_ci(r, n),
            }
        )
    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df["adjusted_p_value"] = bh_adjust(temporal_df["p_value"].to_numpy(dtype=float))
    temporal_df.to_csv(TEMPORAL_CSV, index=False)

    trend_rows = []
    for mode in TARGET_MODES:
        pc_name = f"PC{mode}"
        sub = reference_index_df[[pc_name, "NorthAtlantic_basin_mean", "Global_mean_SST"]].dropna()
        t_idx = np.arange(sub.shape[0], dtype=float)
        r_na, n_na, p_na = correlation_row(sub[pc_name].to_numpy(dtype=float), sub["NorthAtlantic_basin_mean"].to_numpy(dtype=float))
        r_gl, n_gl, p_gl = correlation_row(sub[pc_name].to_numpy(dtype=float), sub["Global_mean_SST"].to_numpy(dtype=float))
        r_t, n_t, p_t = correlation_row(sub[pc_name].to_numpy(dtype=float), t_idx)
        trend_rows.append(
            {
                "target_pc": pc_name,
                "corr_north_atlantic_basin_mean": r_na,
                "corr_global_mean_sst": r_gl,
                "corr_time": r_t,
                "n_months": min(n_na, n_gl, n_t),
                "p_north_atlantic_basin_mean": p_na,
                "p_global_mean_sst": p_gl,
                "p_time": p_t,
            }
        )
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(TREND_CSV, index=False)

    decomp_rows = []
    ref_order = ["AQM", "Conventional_AMV", "NAO_Tripole", "Subpolar_Gyre_SST", "Trend_Pattern"]
    for mode in TARGET_MODES:
        target = target_maps[mode].where(valid_mask)
        ref_arrays = [harmonized[ref_name].where(valid_mask).values for ref_name in ref_order]
        lat2d = np.broadcast_to(target["lat"].values[:, None], target.shape)
        weights = np.cos(np.deg2rad(lat2d))
        mask = np.isfinite(target.values)
        for arr in ref_arrays:
            mask &= np.isfinite(arr)
        y = target.values[mask]
        w = weights[mask]
        if y.size == 0:
            continue
        y_std = weighted_std(y, w)
        if not np.isfinite(y_std) or y_std <= 0.0:
            continue
        yz = (y - weighted_mean(y, w)) / y_std
        xcols = []
        kept_ref_names = []
        for ref_name, arr in zip(ref_order, ref_arrays):
            field = arr[mask]
            field_std = weighted_std(field, w)
            if not np.isfinite(field_std) or field_std <= 0.0:
                continue
            fz = (field - weighted_mean(field, w)) / field_std
            xcols.append(fz)
            kept_ref_names.append(ref_name)
        if not xcols:
            continue
        xmat = np.column_stack(xcols)
        sw = np.sqrt(w)[:, None]
        try:
            beta, _, _, _ = np.linalg.lstsq(xmat * sw, yz * sw[:, 0], rcond=None)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xmat * sw) @ (yz * sw[:, 0])
        fitted = xmat @ beta
        ss_res = np.sum(w * (yz - fitted) ** 2)
        ss_tot = np.sum(w * yz ** 2)
        base_row = {
            "target_mode": f"EOF{mode}",
            "variance_explained": float(1.0 - ss_res / ss_tot),
            "condition_number": float(np.linalg.cond(xmat)),
        }
        for ref_name in ref_order:
            base_row[f"coef_{ref_name}"] = float("nan")
        for ref_name, coeff in zip(kept_ref_names, beta):
            base_row[f"coef_{ref_name}"] = float(coeff)
        decomp_rows.append(base_row)
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(DECOMP_CSV, index=False)

    mode_rows = []
    for mode in TARGET_MODES:
        best = spatial_df[spatial_df["target_mode"] == f"EOF{mode}"].sort_values("absolute_spatial_r", ascending=False).iloc[0]
        mode_rows.append(
            {
                "target_mode": f"EOF{mode}",
                "pc_label": TARGET_LABELS[mode],
                "explained_variance": float(eof_summary["explained_variance_ratio_mode1to6"][mode - 1]),
                "best_spatial_match": best["reference_mode"],
                "best_absolute_spatial_r": float(best["absolute_spatial_r"]),
                "best_demeaned_spatial_r": float(best["demeaned_spatial_r"]),
                "selected_predictor_months": "|".join(TARGET_MONTH_PREDICTORS[TARGET_LABELS[mode]]),
            }
        )
    mode_df = pd.DataFrame(mode_rows)
    mode_df.to_csv(MODE_SUMMARY_CSV, index=False)

    heatmap(spatial_df, "target_mode", "reference_mode", "absolute_spatial_r", SPATIAL_HEATMAP_PNG, "Absolute area-weighted spatial correlation")
    best_temporal = temporal_df[temporal_df["target_month_or_season"] == "all_months"].copy()
    heatmap(best_temporal, "target_pc", "reference_index", "correlation", TEMPORAL_HEATMAP_PNG, "Monthly temporal correlations")

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 10.2), sharex=False)
    ts_specs = [(2, "Conventional_AMV", "NOAA_AMV"), (4, best_refs[4], None), (5, best_refs[5], None)]
    for ax, (mode, ref_name, explicit_index) in zip(axes, ts_specs):
        pc = reference_index_df[["date", f"PC{mode}"]].dropna().copy()
        pc["pc_std"] = (pc[f"PC{mode}"] - pc[f"PC{mode}"].mean()) / pc[f"PC{mode}"].std(ddof=1)
        if explicit_index is not None:
            ref_col = explicit_index
        elif ref_name == "AQM":
            ref_col = "AQM_S2_NDJF"
        elif ref_name == "Conventional_AMV":
            ref_col = "NOAA_AMV"
        elif ref_name == "NAO_Tripole":
            ref_col = "NAO"
        elif ref_name == "Subpolar_Gyre_SST":
            ref_col = "Subpolar_SST_monthly"
        else:
            ref_col = "NorthAtlantic_basin_mean"
        ref = reference_index_df[["date", ref_col]].dropna().copy()
        merged = pc.merge(ref, on="date", how="inner")
        if merged.empty:
            continue
        merged["ref_std"] = (merged[ref_col] - merged[ref_col].mean()) / merged[ref_col].std(ddof=1)
        ax.plot(merged["date"], merged["pc_std"], color="black", linewidth=1.3, label=f"PC{mode}")
        ax.plot(merged["date"], merged["ref_std"], color="tab:red", linewidth=1.0, alpha=0.85, label=ref_col)
        ax.set_title(f"PC{mode} versus {ref_col}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(TIMESERIES_PNG, dpi=220)
    plt.close(fig)

    provenance_rows = [
        {
            "reference_mode": pat.name,
            "label": pat.label,
            "source_kind": pat.source_kind,
            "source_description": pat.source_description,
            "preprocessing": pat.preprocessing,
            "raw_input_path_or_url": {
                "AQM": f"{HADISST_RAW_GZ}; {GPCC_RAW_NC}; {STONE_SUPPLEMENT_PDF}",
                "Conventional_AMV": f"{HADISST_RAW_GZ}; {NOAA_AMV_RAW}",
                "NAO_Tripole": f"{HADISST_RAW_GZ}; {NAO_RAW}",
                "Subpolar_Gyre_SST": str(HADISST_RAW_GZ),
                "Trend_Pattern": str(HADISST_RAW_GZ),
            }[pat.name],
        }
        for pat in reference_patterns
    ]
    pd.DataFrame(provenance_rows).to_csv(PROVENANCE_CSV, index=False)

    METHOD_RECON_PATH.write_text(
        "\n".join(
            [
                "# Existing Atlantic EOF/PC method reconstruction",
                "",
                f"- SST product: `{eof_summary['source_file']}`",
                f"- Domain: `{eof_summary['domain_lat_min']}` to `{eof_summary['domain_lat_max']}` N and `{eof_summary['domain_lon_min_360']}` to `<{eof_summary['domain_lon_max_360_exclusive']}` E (`{eof_summary['domain_equivalent_west_longitude']}`)",
                "- Longitude convention: sorted to `0..360`, duplicate cyclic endpoint removed before subsetting.",
                f"- Latitude weighting: `{eof_summary['weighting_formula']}`",
                "- Anomaly definition: monthly climatology removed at each grid cell over the full COBE2 monthly record.",
                "- Climatological baseline: full available SST record in the COBE2 file used by the original script.",
                "- Additional trend or global-mean removal: none.",
                "- EOF basis: one shared fixed monthly EOF basis used across Sep-Mar; not month-specific.",
                f"- Fit period: `{eof_summary['full_time_start']}` to `{eof_summary['full_time_end']}` (`{eof_summary['n_time_full_record']}` months).",
                "- Fit sample for EOFs: full SST record, not only the 37 SWE years.",
                "- EOF normalization: weighted SVD solved on the monthly anomaly matrix after multiplying each latitude row by sqrt(cos(lat)); saved EOF maps are converted back to unweighted SST-loading units.",
                "- PC normalization: PCs are stored as left singular vectors multiplied by singular value, with no extra variance rescaling.",
                f"- Missing-data / land-mask rule: `{eof_summary['valid_cell_rule']}`",
                f"- Sign convention: `{eof_summary['sign_convention']}`",
                "",
                "This resolves the ambiguity requested in the task: PC2, PC4, and PC5 share one fixed spatial EOF basis across months, so the interpretation uses one EOF2 map, one EOF4 map, and one EOF5 map.",
            ]
        )
        + "\n"
    )

    README_PATH.write_text(
        "\n".join(
            [
                "# Atlantic PC2/PC4/PC5 attribution",
                "",
                "This artifact interprets the previously selected North Atlantic SST predictors `AMV_PC4_Sep`, `AMV_PC5_Feb`, `AMV_PC2_Feb`, `AMV_PC4_Nov`, and `AMV_PC5_Mar` without retraining the Sierra SWE ridge model.",
                "",
                "Main outputs:",
                f"- `{METHOD_RECON_PATH.name}`",
                f"- `{MODE_SUMMARY_CSV.name}`",
                f"- `{SPATIAL_CSV.name}`",
                f"- `{TEMPORAL_CSV.name}`",
                f"- `{TREND_CSV.name}`",
                f"- `{DECOMP_CSV.name}`",
                f"- `{DECISION_MD.name}`",
            ]
        )
        + "\n"
    )

    decision_lines = [
        "# PC2 / PC4 / PC5 attribution decision",
        "",
        "This experiment bases its interpretation on independent spatial and temporal attribution diagnostics, not on Sierra SWE prediction skill.",
        "",
    ]
    for mode in TARGET_MODES:
        target_mode = f"EOF{mode}"
        pc_name = f"PC{mode}"
        best = spatial_df[spatial_df["target_mode"] == target_mode].sort_values("absolute_spatial_r", ascending=False).iloc[0]
        temporal_best = (
            temporal_df[temporal_df["target_pc"] == pc_name]
            .sort_values("correlation", key=lambda s: s.abs(), ascending=False)
            .iloc[0]
        )
        decision_lines.extend(
            [
                f"## {target_mode}",
                f"- Leading spatial match: `{best['reference_mode']}` with `|r|={float(best['absolute_spatial_r']):.3f}` and demeaned `r={float(best['demeaned_spatial_r']):.3f}`.",
                f"- Strongest temporal relationship in this run: `{temporal_best['reference_index']}` for `{temporal_best['target_month_or_season']}` with `r={float(temporal_best['correlation']):.3f}`.",
                "- Interpretation wording should remain cautious if spatial and temporal evidence disagree or if the best match is only modestly stronger than alternatives.",
                "",
            ]
        )
    DECISION_MD.write_text("\n".join(decision_lines) + "\n")

    run_metadata = {
        "tmux_session": "atlantic_pc245_attribution",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "inputs": {
            "eof_dataset": str(EOF_DATASET),
            "eof_summary": str(EOF_SUMMARY),
            "hadisst_raw_gz": str(HADISST_RAW_GZ),
            "hadisst_processed_nc": str(HADISST_PROCESSED_NC),
            "gpcc_raw_nc": str(GPCC_RAW_NC),
            "nao_raw": str(NAO_RAW),
            "noaa_amv_raw_reused": str(NOAA_AMV_RAW),
            "stone_supplement_pdf": str(STONE_SUPPLEMENT_PDF),
        },
    }
    METADATA_JSON.write_text(json.dumps(run_metadata, indent=2))

    print(f"Attribution artifact directory: {ARTIFACT_DIR}")
    print(f"Best spatial matches by mode: {best_refs}")


if __name__ == "__main__":
    main()
