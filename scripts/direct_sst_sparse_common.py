#!/usr/bin/env python3
"""
Shared helpers for direct full-Pacific SST sparse encoder experiments.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import ElasticNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)
SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=int)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
PERIOD_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]
BASELINE_METRICS = {"R2": 0.495860, "RMSE": 0.022516, "sign_accuracy": 0.783784}
ALPHA_GRID = np.asarray([1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0, 10.0], dtype=float)
L1_RATIO_GRID = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99], dtype=float)
NETCDF_ENGINES = [None, "netcdf4", "h5netcdf", "scipy"]
ELASTIC_NET_TOL = 1.0e-4
ELASTIC_NET_MAX_ITER = 20000
COEF_EPS = 1.0e-10


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    last_error: Optional[Exception] = None
    for engine in NETCDF_ENGINES:
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Unable to open dataset {path}: {last_error}")


def normalize_lon_360(lon: np.ndarray) -> np.ndarray:
    lon_360 = np.mod(np.asarray(lon, dtype=float), 360.0)
    lon_360[lon_360 < 0.0] += 360.0
    return lon_360


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    yy = y_true[mask]
    pp = y_pred[mask]
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "mean_error": float(np.mean(error)),
        "median_abs_error": float(np.median(np.abs(error))),
    }


def standardize_train_only(
    x_train_raw: np.ndarray, x_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("Predictor train-fold standard deviation must be positive.")
    return (
        (x_train_raw - x_mean[None, :]) / x_std[None, :],
        (x_test_raw - x_mean) / x_std,
        x_mean,
        x_std,
    )


def standardize_target_train_only(y_train_raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Target train-fold standard deviation must be positive.")
    return (y_train_raw - y_mean) / y_std, y_mean, y_std


def fit_elastic_net_standardized(
    x_train_std: np.ndarray,
    y_train_std: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> ElasticNet:
    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=ELASTIC_NET_MAX_ITER,
        tol=ELASTIC_NET_TOL,
        selection="cyclic",
        random_state=0,
    )
    model.fit(x_train_std, y_train_std)
    return model


def load_target_table() -> pd.DataFrame:
    table = pd.read_csv(TARGET_TABLE_CSV)
    required = ["water_year", "obs_swe"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required target columns in {TARGET_TABLE_CSV}: {missing}")
    table = table[required].copy()
    table["water_year"] = table["water_year"].astype(int)
    table = table[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    table = table.sort_values("water_year").reset_index(drop=True)
    if table["water_year"].tolist() != WATER_YEARS.tolist():
        raise ValueError("Target table does not match WY1985--WY2021.")
    return table


def build_raw_sst_cube() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if not SST_FILE.exists():
        raise FileNotFoundError(f"Missing raw COBE2 SST file: {SST_FILE}")
    selected_times: List[np.datetime64] = []
    for water_year in WATER_YEARS:
        for month_name in MONTHS:
            calendar_year = water_year - 1 if month_name in {"Sep", "Oct", "Nov", "Dec"} else water_year
            selected_times.append(np.datetime64("{:04d}-{:02d}-01".format(calendar_year, MONTH_TO_NUMBER[month_name])))

    with open_dataset_with_fallbacks(SST_FILE) as ds:
        if "sst" not in ds:
            raise KeyError(f"Expected variable 'sst' in {SST_FILE}")
        lat_values = np.asarray(ds["lat"].values, dtype=float)
        lat_slice = slice(60.0, -10.0) if lat_values[0] > lat_values[-1] else slice(-10.0, 60.0)
        sst = ds["sst"].sel(time=selected_times, lat=lat_slice, lon=slice(120.0, 280.0)).load()
        cube = np.asarray(sst.values, dtype=float).reshape(len(WATER_YEARS), len(MONTHS), sst.sizes["lat"], sst.sizes["lon"])
        lat = np.asarray(sst["lat"].values, dtype=float)
        lon = normalize_lon_360(np.asarray(sst["lon"].values, dtype=float))
        missing_value = sst.attrs.get("missing_value")
        fill_value = sst.attrs.get("_FillValue")
        if missing_value is not None:
            cube = np.where(cube == float(missing_value), np.nan, cube)
        if fill_value is not None:
            cube = np.where(cube == float(fill_value), np.nan, cube)
        aligned_times = pd.to_datetime(np.asarray(sst["time"].values))

    valid_feature_mask = np.all(np.isfinite(cube), axis=0)
    valid_feature_mask_flat = valid_feature_mask.reshape(-1)
    month_grid = np.repeat(np.arange(len(MONTHS), dtype=int), lat.size * lon.size)
    lat_grid = np.tile(np.repeat(lat, lon.size), len(MONTHS))
    lon_grid = np.tile(np.tile(lon, lat.size), len(MONTHS))
    flat_index = np.arange(valid_feature_mask_flat.size, dtype=int)
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))

    monthly_clim_all = np.mean(cube[:, valid_feature_mask], axis=0)
    anomalies_all = cube[:, valid_feature_mask] - monthly_clim_all[None, :]
    weighted_all = anomalies_all * np.tile(np.repeat(lat_weights, lon.size), len(MONTHS))[valid_feature_mask_flat][None, :]
    if not np.all(np.isfinite(weighted_all)):
        raise ValueError("All-years cleaned SST matrix still contains non-finite values.")

    metadata = {
        "source_variable": "sst",
        "aligned_timestamp_matrix": np.asarray(aligned_times).reshape(len(WATER_YEARS), len(MONTHS)).astype("datetime64[D]").astype(str).tolist(),
        "domain": {
            "lat_min": float(np.min(lat)),
            "lat_max": float(np.max(lat)),
            "lon_min_360": float(np.min(lon)),
            "lon_max_360": float(np.max(lon)),
        },
        "months": MONTHS,
        "feature_shape_before_mask": [int(len(MONTHS)), int(lat.size), int(lon.size)],
        "original_feature_count": int(valid_feature_mask_flat.size),
        "num_valid_features": int(np.sum(valid_feature_mask_flat)),
        "removed_feature_count": int(valid_feature_mask_flat.size - np.sum(valid_feature_mask_flat)),
        "final_matrix_shape": [int(len(WATER_YEARS)), int(np.sum(valid_feature_mask_flat))],
        "valid_feature_mask_shape": [int(len(MONTHS)), int(lat.size), int(lon.size)],
        "area_weighting": {
            "formula": "weighted_sst = sst_anomaly * sqrt(cos(lat))",
            "latitude_sqrt_cos": lat_weights.astype(float).tolist(),
        },
        "anomaly_definition": "Within each inner/outer training fold, subtract the train-fold monthly climatology separately for Sep, Oct, Nov, Dec, Jan, Feb, and Mar.",
        "feature_records": {
            "month": [MONTHS[idx] for idx in month_grid.tolist()],
            "lat": lat_grid.astype(float).tolist(),
            "lon": lon_grid.astype(float).tolist(),
            "flat_feature_index": flat_index.tolist(),
            "is_valid": valid_feature_mask_flat.astype(int).tolist(),
        },
        "valid_feature_mask": valid_feature_mask.astype(int).tolist(),
    }
    return cube, lat, lon, metadata


def raw_valid_flat_indices(valid_feature_mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(valid_feature_mask.reshape(-1))


def valid_feature_lat_weights(lat: np.ndarray, lon: np.ndarray, valid_feature_mask: np.ndarray) -> np.ndarray:
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    return np.tile(np.repeat(lat_weights, lon.size), len(MONTHS))[valid_feature_mask.reshape(-1)]


def prepare_weighted_features(
    train_cube_raw: np.ndarray,
    test_cube_raw: np.ndarray,
    valid_feature_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_flat = valid_feature_mask.reshape(-1)
    train_valid = train_cube_raw.reshape(train_cube_raw.shape[0], -1)[:, valid_flat]
    test_valid = test_cube_raw.reshape(-1)[valid_flat]
    monthly_clim = np.mean(train_valid, axis=0)
    train_anom = train_valid - monthly_clim[None, :]
    test_anom = test_valid - monthly_clim
    lat_w = valid_feature_lat_weights(lat, lon, valid_feature_mask)
    x_train = train_anom * lat_w[None, :]
    x_test = test_anom * lat_w
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
        raise ValueError("Non-finite values entered the cleaned weighted SST feature matrix.")
    return x_train, x_test


def feature_records_dataframe(metadata: Dict[str, object]) -> pd.DataFrame:
    records = metadata["feature_records"]
    return pd.DataFrame(
        {
            "month": records["month"],
            "lat": records["lat"],
            "lon": records["lon"],
            "flat_feature_index": records["flat_feature_index"],
            "is_valid": records["is_valid"],
        }
    )


def compute_period_metrics(predictions: pd.DataFrame, model_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name in predictions[model_col].drop_duplicates().tolist():
        subset = predictions[predictions[model_col] == model_name].copy()
        years = subset["heldout_wy"].to_numpy(dtype=int)
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        for period_name, selector in PERIOD_SPECS:
            mask = selector(years)
            if int(np.sum(mask)) < 2:
                continue
            rows.append(
                {
                    model_col: model_name,
                    "period": period_name,
                    "n_years": int(np.sum(mask)),
                    **compute_metric_bundle(obs[mask], pred[mask]),
                }
            )
    return pd.DataFrame(rows)


def choose_best_hyperparams(rows: List[Dict[str, float]]) -> Tuple[float, float]:
    best = min(rows, key=lambda row: (row["mse"], row["alpha"], row["l1_ratio"]))
    return float(best["alpha"]), float(best["l1_ratio"])


def elastic_net_inner_loyo(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
) -> Tuple[float, float, List[Dict[str, float]]]:
    n_train = x_train_raw.shape[0]
    score_rows: List[Dict[str, float]] = []
    for alpha in ALPHA_GRID.tolist():
        for l1_ratio in L1_RATIO_GRID.tolist():
            squared_errors: List[float] = []
            for inner_idx in range(n_train):
                mask = np.ones(n_train, dtype=bool)
                mask[inner_idx] = False
                x_inner_train = x_train_raw[mask, :]
                x_inner_valid = x_train_raw[~mask, :][0]
                y_inner_train = y_train_raw[mask]
                y_inner_valid = float(y_train_raw[~mask][0])
                x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(x_inner_train, x_inner_valid)
                y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train)
                model = fit_elastic_net_standardized(x_inner_train_std, y_inner_train_std, alpha, l1_ratio)
                pred_std = float(model.predict(x_inner_valid_std[None, :])[0])
                y_inner_valid_std = float((y_inner_valid - y_mean) / y_std)
                squared_errors.append((pred_std - y_inner_valid_std) ** 2)
            score_rows.append({"alpha": float(alpha), "l1_ratio": float(l1_ratio), "mse": float(np.mean(squared_errors))})
    best_alpha, best_l1_ratio = choose_best_hyperparams(score_rows)
    return best_alpha, best_l1_ratio, score_rows


def patch_grid_for_size(lat: np.ndarray, lon: np.ndarray, patch_size_deg: int) -> Dict[str, np.ndarray]:
    lat_start = -10.0
    lon_start = 120.0
    lat_ids = np.floor((np.asarray(lat, dtype=float) - lat_start) / float(patch_size_deg)).astype(int)
    lon_ids = np.floor((np.asarray(lon, dtype=float) - lon_start) / float(patch_size_deg)).astype(int)
    lat_count = int(lat_ids.max()) + 1
    lon_count = int(lon_ids.max()) + 1
    lat_centers = lat_start + patch_size_deg * (np.arange(lat_count, dtype=float) + 0.5)
    lon_centers = lon_start + patch_size_deg * (np.arange(lon_count, dtype=float) + 0.5)
    return {
        "lat_ids": lat_ids,
        "lon_ids": lon_ids,
        "lat_count": np.asarray(lat_count),
        "lon_count": np.asarray(lon_count),
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
    }


def prepare_patch_features(
    train_cube_raw: np.ndarray,
    test_cube_raw: np.ndarray,
    valid_feature_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    patch_size_deg: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    patch_grid = patch_grid_for_size(lat, lon, patch_size_deg)
    lat_ids = patch_grid["lat_ids"]
    lon_ids = patch_grid["lon_ids"]
    lat_count = int(patch_grid["lat_count"])
    lon_count = int(patch_grid["lon_count"])
    cos_weights = np.cos(np.deg2rad(lat))
    monthly_clim = np.mean(train_cube_raw, axis=0)
    train_anom = train_cube_raw - monthly_clim[None, :, :, :]
    test_anom = test_cube_raw - monthly_clim
    feature_rows: List[Dict[str, object]] = []
    x_train_cols: List[np.ndarray] = []
    x_test_cols: List[float] = []

    for month_idx, month_name in enumerate(MONTHS):
        valid_month = valid_feature_mask[month_idx, :, :]
        for lat_patch in range(lat_count):
            lat_mask = lat_ids == lat_patch
            for lon_patch in range(lon_count):
                cell_mask = valid_month & lat_mask[:, None] & (lon_ids == lon_patch)[None, :]
                if not np.any(cell_mask):
                    continue
                weights_2d = np.broadcast_to(cos_weights[:, None], cell_mask.shape)
                group_weights = weights_2d[cell_mask]
                weight_sum = float(np.sum(group_weights))
                train_vals = train_anom[:, month_idx, :, :][:, cell_mask]
                test_vals = test_anom[month_idx, :, :][cell_mask]
                train_mean = np.sum(train_vals * group_weights[None, :], axis=1) / weight_sum
                test_mean = float(np.sum(test_vals * group_weights) / weight_sum)
                x_train_cols.append(train_mean.astype(float))
                x_test_cols.append(test_mean)
                feature_rows.append(
                    {
                        "patch_size_deg": int(patch_size_deg),
                        "month": month_name,
                        "month_index": int(month_idx),
                        "lat_patch": int(lat_patch),
                        "lon_patch": int(lon_patch),
                        "lat_center": float(patch_grid["lat_centers"][lat_patch]),
                        "lon_center": float(patch_grid["lon_centers"][lon_patch]),
                        "num_cells": int(np.sum(cell_mask)),
                    }
                )

    x_train = np.column_stack(x_train_cols).astype(float)
    x_test = np.asarray(x_test_cols, dtype=float)
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
        raise ValueError(f"Non-finite patch features produced for patch size {patch_size_deg}.")
    return x_train, x_test, pd.DataFrame(feature_rows)


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
