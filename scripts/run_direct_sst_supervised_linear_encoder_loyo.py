#!/usr/bin/env python3
"""
Strict LOYO direct-SST supervised linear encoder baseline for Sierra SWE.
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cross_decomposition import PLSRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "direct_sst_supervised_linear_encoder_loyo"
)
TARGET_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)
SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")

METADATA_JSON = OUTPUT_DIR / "direct_sst_encoder_predictor_metadata.json"
PREDICTIONS_CSV = OUTPUT_DIR / "direct_sst_encoder_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "direct_sst_encoder_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "direct_sst_encoder_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "direct_sst_encoder_selected_alpha_by_fold.csv"
SCORES_CSV = OUTPUT_DIR / "direct_sst_encoder_scores_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "direct_sst_encoder_beta_by_fold.csv"
MODE_SUMMARY_CSV = OUTPUT_DIR / "direct_sst_encoder_mode_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "direct_sst_encoder_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "direct_sst_encoder_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "direct_sst_encoder_scatter_k1_to_k5.png"
ERROR_PNG = OUTPUT_DIR / "direct_sst_encoder_error_by_year.png"
COMPONENT_MAPS_PNG = OUTPUT_DIR / "direct_sst_encoder_component_maps_k1_to_k5.png"
COMPONENT_STABILITY_PNG = OUTPUT_DIR / "direct_sst_encoder_component_stability.png"
MEAN_STABILITY_NC = OUTPUT_DIR / "encoder_modes_mean_stability.nc"
RUN_LOG = OUTPUT_DIR / "run.log"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=int)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
MODEL_NAMES = [f"DIRECT_SST_PLS_K{k}" for k in range(1, 6)]
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
PERIOD_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]
BASELINE_METRICS = {"R2": 0.495860, "RMSE": 0.022516, "sign_accuracy": 0.783784}
NETCDF_ENGINES = [None, "netcdf4", "h5netcdf", "scipy"]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    last_error: Optional[Exception] = None
    for engine in NETCDF_ENGINES:
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on local engines
            last_error = exc
    raise RuntimeError(f"Unable to open dataset {path}: {last_error}")


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


def fit_ridge_with_intercept(z_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    z_mean = np.mean(z_train, axis=0)
    y_mean = float(np.mean(y_train))
    z_centered = z_train - z_mean[None, :]
    y_centered = y_train - y_mean
    gram = z_centered.T @ z_centered
    rhs = z_centered.T @ y_centered
    beta = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    intercept = float(y_mean - z_mean @ beta)
    return beta, intercept


def predict_ridge(z: np.ndarray, beta: np.ndarray, intercept: float) -> np.ndarray:
    return z @ beta + intercept


def normalize_lon_360(lon: np.ndarray) -> np.ndarray:
    lon_360 = np.mod(np.asarray(lon, dtype=float), 360.0)
    lon_360[lon_360 < 0.0] += 360.0
    return lon_360


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


def build_sst_feature_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
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
    if not np.any(valid_feature_mask):
        raise ValueError("No all-year-finite SST features remained after applying the required WY1985--WY2021 mask.")

    valid_feature_mask_flat = valid_feature_mask.reshape(-1)
    month_grid = np.repeat(np.arange(len(MONTHS), dtype=int), lat.size * lon.size)
    lat_grid = np.tile(np.repeat(lat, lon.size), len(MONTHS))
    lon_grid = np.tile(np.tile(lon, lat.size), len(MONTHS))
    valid_indices = np.flatnonzero(valid_feature_mask_flat)
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))

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
        "num_valid_features": int(valid_indices.size),
        "valid_feature_mask_shape": [int(len(MONTHS)), int(lat.size), int(lon.size)],
        "feature_mapping": {
            "flat_feature_index": valid_indices.astype(int).tolist(),
            "month": [MONTHS[idx] for idx in month_grid[valid_indices].tolist()],
            "lat": lat_grid[valid_indices].astype(float).tolist(),
            "lon": lon_grid[valid_indices].astype(float).tolist(),
        },
        "area_weighting": {
            "formula": "weighted_sst = sst_anomaly * sqrt(cos(lat))",
            "latitude_sqrt_cos": lat_weights.astype(float).tolist(),
        },
        "anomaly_definition": "Within each inner/outer training fold, subtract the train-fold monthly climatology separately for Sep, Oct, Nov, Dec, Jan, Feb, and Mar.",
    }
    return cube, lat, lon, {"valid_feature_mask": valid_feature_mask, **metadata}


def prepare_weighted_features(
    train_cube_raw: np.ndarray,
    test_cube_raw: np.ndarray,
    valid_feature_mask: np.ndarray,
    lat_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    monthly_clim = np.nanmean(train_cube_raw, axis=0, dtype=np.float64)
    train_anom = train_cube_raw - monthly_clim[None, :, :, :]
    test_anom = test_cube_raw - monthly_clim
    train_weighted = train_anom * lat_weights[None, None, :, None]
    test_weighted = test_anom * lat_weights[None, :, None]
    valid_flat = valid_feature_mask.reshape(-1)
    x_train = train_weighted.reshape(train_weighted.shape[0], -1)[:, valid_flat]
    x_test = test_weighted.reshape(-1)[valid_flat]
    return x_train, x_test


def inner_loyo_best_alpha(
    train_cube_raw: np.ndarray,
    y_train_raw: np.ndarray,
    k: int,
    valid_feature_mask: np.ndarray,
    lat_weights: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    n_train = train_cube_raw.shape[0]
    error_sums = {float(alpha): 0.0 for alpha in ALPHA_GRID.tolist()}
    for inner_idx in range(n_train):
        mask = np.ones(n_train, dtype=bool)
        mask[inner_idx] = False
        x_inner_train, x_inner_valid = prepare_weighted_features(
            train_cube_raw[mask, :, :, :],
            train_cube_raw[~mask, :, :, :][0],
            valid_feature_mask,
            lat_weights,
        )
        y_inner_train = y_train_raw[mask]
        y_inner_valid = float(y_train_raw[~mask][0])

        x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(x_inner_train, x_inner_valid)
        y_inner_train_std, y_inner_mean, y_inner_std = standardize_target_train_only(y_inner_train)

        pls = PLSRegression(n_components=int(k), scale=False)
        pls.fit(x_inner_train_std, y_inner_train_std[:, None])
        z_inner_train = np.asarray(pls.transform(x_inner_train_std), dtype=float)
        z_inner_valid = np.asarray(pls.transform(x_inner_valid_std[None, :]), dtype=float)[0]

        for alpha in ALPHA_GRID.tolist():
            beta_std, intercept_std = fit_ridge_with_intercept(z_inner_train, y_inner_train_std, float(alpha))
            pred_std = float(predict_ridge(z_inner_valid[None, :], beta_std, intercept_std)[0])
            y_inner_valid_std = float((y_inner_valid - y_inner_mean) / y_inner_std)
            error_sums[float(alpha)] += float((pred_std - y_inner_valid_std) ** 2)

    alpha_to_mse = {float(alpha): error_sums[float(alpha)] / float(n_train) for alpha in ALPHA_GRID.tolist()}
    best_alpha = min(alpha_to_mse.items(), key=lambda item: (item[1], item[0]))[0]
    return float(best_alpha), alpha_to_mse


def aligned_component_maps(
    x_mean: np.ndarray,
    x_std: np.ndarray,
    pls: PLSRegression,
    beta_score_raw: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted_space = np.asarray(pls.x_weights_, dtype=float).copy()
    weighted_space *= np.where(beta_score_raw[None, :] < 0.0, -1.0, 1.0)
    beta_aligned = np.abs(beta_score_raw).copy()

    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    feature_lat_weights = np.tile(np.repeat(lat_weights, lon.size), len(MONTHS))[valid_feature_mask.reshape(-1)]
    physical_space = weighted_space * feature_lat_weights[:, None]

    n_components = weighted_space.shape[1]
    weighted_grid = np.full((n_components, len(MONTHS), lat.size, lon.size), np.nan, dtype=float)
    physical_grid = np.full_like(weighted_grid, np.nan)
    weighted_grid.reshape(n_components, -1)[:, valid_feature_mask.reshape(-1)] = weighted_space.T
    physical_grid.reshape(n_components, -1)[:, valid_feature_mask.reshape(-1)] = physical_space.T
    return weighted_grid, physical_grid, beta_aligned


def vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa, ddof=1) == 0.0 or np.std(bb, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def choose_mode_descriptor(
    mean_map: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> str:
    magnitude = np.abs(mean_map)
    if not np.any(np.isfinite(magnitude)):
        return "indeterminate"
    month_max = int(np.nanargmax(np.nanmax(magnitude, axis=(1, 2))))
    month_slice = magnitude[month_max]
    peak_index = int(np.nanargmax(month_slice))
    lat_idx, lon_idx = np.unravel_index(peak_index, month_slice.shape)
    peak_lat = float(lat[lat_idx])
    peak_lon = float(lon[lon_idx])
    zonal_fraction = float(np.nanmean(month_slice > 0.5 * np.nanmax(month_slice)))
    if zonal_fraction < 0.08:
        return f"localized peak near {MONTHS[month_max]} {peak_lat:.1f}N {peak_lon:.1f}E"
    if peak_lon >= 300.0 or peak_lon <= 20.0:
        return "broad Atlantic-like structure"
    if 180.0 <= peak_lon <= 260.0:
        return "broad central/eastern Pacific-like structure"
    return "broad basin-scale mixed Pacific structure"


def run_model(
    k: int,
    raw_cube: np.ndarray,
    y_raw: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    years = WATER_YEARS.copy()
    model_name = f"DIRECT_SST_PLS_K{k}"

    prediction_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []

    preds = np.full(years.size, np.nan, dtype=float)
    selected_alphas = np.full(years.size, np.nan, dtype=float)
    fold_weighted_maps = np.full((years.size, k, len(MONTHS), lat.size, lon.size), np.nan, dtype=float)
    fold_physical_maps = np.full_like(fold_weighted_maps, np.nan)
    fold_betas = np.full((years.size, k), np.nan, dtype=float)
    fold_sign_flips = np.ones((years.size, k), dtype=int)
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))

    for outer_idx, water_year in enumerate(years):
        mask = np.ones(years.size, dtype=bool)
        mask[outer_idx] = False
        x_train, x_test = prepare_weighted_features(
            raw_cube[mask, :, :, :],
            raw_cube[~mask, :, :, :][0],
            valid_feature_mask,
            lat_weights,
        )
        y_train = y_raw[mask]
        y_test = float(y_raw[~mask][0])
        train_years = years[mask]

        selected_alpha, inner_alpha_to_mse = inner_loyo_best_alpha(
            raw_cube[mask, :, :, :],
            y_train,
            k,
            valid_feature_mask,
            lat_weights,
        )
        x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train, x_test)
        y_train_std, y_mean, y_std = standardize_target_train_only(y_train)

        pls = PLSRegression(n_components=int(k), scale=False)
        pls.fit(x_train_std, y_train_std[:, None])
        z_train = np.asarray(pls.transform(x_train_std), dtype=float)
        z_test = np.asarray(pls.transform(x_test_std[None, :]), dtype=float)[0]
        beta_std, intercept_std = fit_ridge_with_intercept(z_train, y_train_std, selected_alpha)
        pred_std = float(predict_ridge(z_test[None, :], beta_std, intercept_std)[0])
        pred_raw = float(y_mean + y_std * pred_std)
        beta_score_raw = y_std * beta_std
        intercept_raw = float(y_mean + y_std * intercept_std)

        weighted_maps, physical_maps, beta_score_aligned = aligned_component_maps(
            x_mean=x_mean,
            x_std=x_std,
            pls=pls,
            beta_score_raw=beta_score_raw,
            lat=lat,
            lon=lon,
            valid_feature_mask=valid_feature_mask,
        )
        aligned_train_scores = z_train.copy()
        aligned_test_score = z_test.copy()
        sign_flip = np.where(beta_score_raw < 0.0, -1, 1)
        aligned_train_scores *= sign_flip[None, :]
        aligned_test_score *= sign_flip
        fold_sign_flips[outer_idx, :] = sign_flip.astype(int)

        preds[outer_idx] = pred_raw
        selected_alphas[outer_idx] = selected_alpha
        fold_weighted_maps[outer_idx, :, :, :, :] = weighted_maps
        fold_physical_maps[outer_idx, :, :, :, :] = physical_maps
        fold_betas[outer_idx, :] = beta_score_aligned

        error = pred_raw - y_test
        prediction_rows.append(
            {
                "model_name": model_name,
                "k": int(k),
                "heldout_wy": int(water_year),
                "obs_swe": y_test,
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error,
                "residual_obs_minus_pred": -error,
                "abs_error": abs(error),
                "sign_correct": float((np.sign(pred_raw) == np.sign(y_test)) and (pred_raw != 0.0) and (y_test != 0.0)),
                "selected_alpha": float(selected_alpha),
                "num_latent_components": int(k),
            }
        )
        for alpha, mse in inner_alpha_to_mse.items():
            alpha_rows.append(
                {
                    "model_name": model_name,
                    "k": int(k),
                    "heldout_wy": int(water_year),
                    "alpha": float(alpha),
                    "inner_cv_mse": float(mse),
                    "selected_alpha": float(selected_alpha),
                    "is_selected": int(np.isclose(alpha, selected_alpha)),
                }
            )
        for component_idx in range(k):
            beta_rows.append(
                {
                    "model_name": model_name,
                    "k": int(k),
                    "heldout_wy": int(water_year),
                    "intercept": intercept_raw,
                    "component": int(component_idx + 1),
                    "ridge_beta": float(beta_score_aligned[component_idx]),
                    "selected_alpha": float(selected_alpha),
                    "sign_flip_applied": int(sign_flip[component_idx]),
                }
            )
        for row_year, row_scores in zip(train_years, aligned_train_scores):
            for component_idx in range(k):
                score_rows.append(
                    {
                        "model_name": model_name,
                        "k": int(k),
                        "heldout_wy": int(water_year),
                        "row_water_year": int(row_year),
                        "is_train_or_test": "train",
                        "component": int(component_idx + 1),
                        "score": float(row_scores[component_idx]),
                    }
                )
        for component_idx in range(k):
            score_rows.append(
                {
                    "model_name": model_name,
                    "k": int(k),
                    "heldout_wy": int(water_year),
                    "row_water_year": int(water_year),
                    "is_train_or_test": "test",
                    "component": int(component_idx + 1),
                    "score": float(aligned_test_score[component_idx]),
                }
            )
        print(
            f"LOYO heldout_WY={int(water_year)} model={model_name} selected_alpha={selected_alpha:g} "
            f"obs={y_test:.6f} pred={pred_raw:.6f}",
            flush=True,
        )

    pair_rows: List[Dict[str, object]] = []
    for component_idx in range(k):
        vectors = fold_physical_maps[:, component_idx, :, :, :].reshape(years.size, -1)
        pair_corrs = [
            vector_corr(vectors[i], vectors[j])
            for i, j in combinations(range(years.size), 2)
        ]
        pair_corrs_arr = np.asarray(pair_corrs, dtype=float)
        pair_rows.append(
            {
                "k": int(k),
                "component": int(component_idx + 1),
                "mean_pairwise_map_correlation": float(np.nanmean(pair_corrs_arr)),
                "median_pairwise_map_correlation": float(np.nanmedian(pair_corrs_arr)),
                "min_pairwise_map_correlation": float(np.nanmin(pair_corrs_arr)),
                "max_pairwise_map_correlation": float(np.nanmax(pair_corrs_arr)),
                "mean_ridge_beta": float(np.nanmean(fold_betas[:, component_idx])),
                "std_ridge_beta": float(np.nanstd(fold_betas[:, component_idx], ddof=1)),
                "positive_sign_fraction_before_alignment": float(np.mean(fold_sign_flips[:, component_idx] > 0)),
            }
        )

    metrics_bundle = compute_metric_bundle(y_raw, preds)
    alpha_selected = selected_alphas[np.isfinite(selected_alphas)]
    if alpha_selected.size == 0:
        alpha_mode = float("nan")
    else:
        values, counts = np.unique(alpha_selected, return_counts=True)
        alpha_mode = float(values[np.argmax(counts)])
    metrics_bundle.update(
        {
            "model_name": model_name,
            "k": int(k),
            "num_latent_components": int(k),
            "selected_alpha_mode": alpha_mode,
            "selected_alpha_median": float(np.nanmedian(selected_alphas)),
        }
    )

    mode_summary = pd.DataFrame(pair_rows)
    mean_physical_maps = np.nanmean(fold_physical_maps, axis=0)
    std_physical_maps = np.nanstd(fold_physical_maps, axis=0, ddof=1)
    mode_descriptions = [
        choose_mode_descriptor(mean_physical_maps[idx], lat, lon)
        for idx in range(k)
    ]
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(alpha_rows),
        pd.DataFrame(beta_rows),
        pd.DataFrame(score_rows),
        {
            "metrics": metrics_bundle,
            "mode_summary": mode_summary,
            "fold_weighted_maps": fold_weighted_maps,
            "fold_physical_maps": fold_physical_maps,
            "mean_physical_maps": mean_physical_maps,
            "std_physical_maps": std_physical_maps,
            "valid_feature_mask": valid_feature_mask,
            "lat": lat,
            "lon": lon,
            "mode_descriptions": mode_descriptions,
        },
    )


def compute_period_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for k in range(1, 6):
        model_name = f"DIRECT_SST_PLS_K{k}"
        subset = predictions[predictions["model_name"] == model_name].copy()
        years = subset["heldout_wy"].to_numpy(dtype=int)
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        for period_name, selector in PERIOD_SPECS:
            mask = selector(years)
            if int(np.sum(mask)) < 2:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "k": int(k),
                    "period": period_name,
                    "n_years": int(np.sum(mask)),
                    **compute_metric_bundle(obs[mask], pred[mask]),
                }
            )
    return pd.DataFrame(rows)


def write_fold_map_netcdf(
    k: int,
    payload: Dict[str, object],
    lat: np.ndarray,
    lon: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> None:
    path = OUTPUT_DIR / f"encoder_modes_by_fold_k{k}.nc"
    years = WATER_YEARS.astype(np.int32)
    components = np.arange(1, k + 1, dtype=np.int32)
    ds = xr.Dataset(
        data_vars={
            "pls_weight_weighted_space": (
                ("heldout_wy", "component", "month", "lat", "lon"),
                np.asarray(payload["fold_weighted_maps"], dtype=np.float32),
            ),
            "activation_mode_weighted_space": (
                ("heldout_wy", "component", "month", "lat", "lon"),
                np.asarray(payload["fold_weighted_maps"], dtype=np.float32),
            ),
            "activation_mode_physical_space": (
                ("heldout_wy", "component", "month", "lat", "lon"),
                np.asarray(payload["fold_physical_maps"], dtype=np.float32),
            ),
            "valid_feature_mask": (
                ("month", "lat", "lon"),
                np.asarray(valid_feature_mask, dtype=np.int8),
            ),
        },
        coords={
            "heldout_wy": years,
            "component": components,
            "month": np.asarray(MONTHS, dtype=object),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
        },
        attrs={
            "description": f"Foldwise direct-SST supervised encoder maps for k={k}",
            "weighting_formula": "weighted_sst = physical_sst * sqrt(cos(lat))",
            "sign_convention": "Each component sign was flipped when needed so the corresponding ridge beta is non-negative.",
        },
    )
    ds.to_netcdf(path)


def write_mean_stability_netcdf(
    payloads_by_k: Dict[int, Dict[str, object]],
    mode_summary_df: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    max_k = 5
    mean_maps = np.full((max_k, max_k, len(MONTHS), lat.size, lon.size), np.nan, dtype=np.float32)
    std_maps = np.full_like(mean_maps, np.nan)
    scores = np.full((max_k, max_k), np.nan, dtype=np.float32)
    for k in range(1, 6):
        payload = payloads_by_k[k]
        mean_maps[k - 1, :k, :, :, :] = np.asarray(payload["mean_physical_maps"], dtype=np.float32)
        std_maps[k - 1, :k, :, :, :] = np.asarray(payload["std_physical_maps"], dtype=np.float32)
        sub = mode_summary_df[mode_summary_df["k"] == k].sort_values("component")
        scores[k - 1, :k] = sub["mean_pairwise_map_correlation"].to_numpy(dtype=np.float32)
    ds = xr.Dataset(
        data_vars={
            "mean_activation_mode_physical_space": (
                ("k", "component", "month", "lat", "lon"),
                mean_maps,
            ),
            "std_activation_mode_physical_space": (
                ("k", "component", "month", "lat", "lon"),
                std_maps,
            ),
            "selection_or_stability_score": (
                ("k", "component"),
                scores,
            ),
        },
        coords={
            "k": np.arange(1, 6, dtype=np.int32),
            "component": np.arange(1, 6, dtype=np.int32),
            "month": np.asarray(MONTHS, dtype=object),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
        },
        attrs={"description": "Mean and fold-to-fold variability of direct-SST encoder activation maps."},
    )
    ds.to_netcdf(MEAN_STABILITY_NC)


def plot_observed_vs_predicted(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    base = predictions[predictions["model_name"] == MODEL_NAMES[0]].sort_values("heldout_wy")
    ax.plot(base["heldout_wy"], base["obs_swe"], color="black", linewidth=2.6, label="Observed SWE anomaly")
    colors = ["#0b3c5d", "#328cc1", "#d9b310", "#1d2731", "#b85c38"]
    for model_name, color in zip(MODEL_NAMES, colors):
        subset = predictions[predictions["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(subset["heldout_wy"], subset["pred_swe"], linewidth=1.5, label=model_name, color=color)
    ax.set_title("Strict LOYO predictions using direct supervised SST encoder")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(predictions: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True, sharex=True, sharey=True)
    colors = ["#0b3c5d", "#328cc1", "#d9b310", "#1d2731", "#b85c38"]
    all_obs = predictions["obs_swe"].to_numpy(dtype=float)
    all_pred = predictions["pred_swe"].to_numpy(dtype=float)
    min_val = float(min(np.min(all_obs), np.min(all_pred)))
    max_val = float(max(np.max(all_obs), np.max(all_pred)))
    for ax, model_name, color in zip(axes, MODEL_NAMES, colors):
        subset = predictions[predictions["model_name"] == model_name].copy()
        metric_row = metrics_df.loc[metrics_df["model_name"] == model_name].iloc[0]
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        ax.scatter(obs, pred, s=34, color=color, alpha=0.85)
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"K={int(metric_row['k'])}")
        ax.set_xlabel("Observed SWE anomaly (m)")
        ax.text(
            0.03,
            0.97,
            f"RMSE={metric_row['RMSE']:.3f}\nR2={metric_row['R2']:.3f}\nr={metric_row['r']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted SWE anomaly (m)")
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_year(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    colors = ["#0b3c5d", "#328cc1", "#d9b310", "#1d2731", "#b85c38"]
    for model_name, color in zip(MODEL_NAMES, colors):
        subset = predictions[predictions["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            subset["heldout_wy"],
            subset["error_pred_minus_obs"],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            color=color,
            label=model_name,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Prediction error by year: direct supervised SST encoder")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Prediction error (pred - obs) (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.savefig(ERROR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_component_maps(
    best_k: int,
    payload: Dict[str, object],
    mode_summary_df: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    mean_maps = np.asarray(payload["mean_physical_maps"], dtype=float)
    descriptions = list(payload["mode_descriptions"])
    n_components = best_k
    fig, axes = plt.subplots(n_components, 2, figsize=(12, 3.2 * n_components), constrained_layout=True)
    if n_components == 1:
        axes = np.asarray([axes])
    for component_idx in range(n_components):
        month_abs = np.nanmax(np.abs(mean_maps[component_idx]), axis=(1, 2))
        max_month_idx = int(np.nanargmax(month_abs))
        monthly_map = mean_maps[component_idx, max_month_idx, :, :]
        integrated_map = np.nanmean(mean_maps[component_idx], axis=0)
        stability_row = mode_summary_df[
            (mode_summary_df["k"] == best_k) & (mode_summary_df["component"] == component_idx + 1)
        ].iloc[0]
        for ax, arr, title_suffix in (
            (axes[component_idx, 0], monthly_map, f"Peak month: {MONTHS[max_month_idx]}"),
            (axes[component_idx, 1], integrated_map, "Sep-Mar mean"),
        ):
            vmax = float(np.nanmax(np.abs(arr)))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            mesh = ax.pcolormesh(lon, lat, arr, cmap="coolwarm", shading="auto", vmin=-vmax, vmax=vmax)
            ax.set_xlabel("Longitude (E)")
            ax.set_ylabel("Latitude")
            ax.set_title(
                f"K={best_k} C{component_idx + 1} {title_suffix}\n"
                f"stab={stability_row['mean_pairwise_map_correlation']:.2f} "
                f"beta={stability_row['mean_ridge_beta']:.3f}\n"
                f"{descriptions[component_idx]}"
            )
            fig.colorbar(mesh, ax=ax, shrink=0.86)
    fig.savefig(COMPONENT_MAPS_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_component_stability(mode_summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    width = 0.14
    component_positions = np.arange(1, 6)
    colors = ["#0b3c5d", "#328cc1", "#d9b310", "#1d2731", "#b85c38"]
    for idx, k in enumerate(range(1, 6)):
        sub = mode_summary_df[mode_summary_df["k"] == k].sort_values("component")
        x = component_positions[: len(sub)] + (idx - 2) * width
        ax.bar(x, sub["mean_pairwise_map_correlation"].to_numpy(dtype=float), width=width, color=colors[idx], label=f"K={k}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean pairwise fold-to-fold map correlation")
    ax.set_title("Direct SST encoder component stability")
    ax.set_xticks(component_positions)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", ncol=3)
    fig.savefig(COMPONENT_STABILITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_alpha_choices(alpha_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for k in range(1, 6):
        sub = alpha_df[(alpha_df["k"] == k) & (alpha_df["is_selected"] == 1)].copy()
        values = sub["selected_alpha"].to_numpy(dtype=float)
        uniques, counts = np.unique(values, return_counts=True)
        summary[f"DIRECT_SST_PLS_K{k}"] = {
            "selected_alpha_mode": float(uniques[np.argmax(counts)]),
            "selected_alpha_median": float(np.median(values)),
            "alpha_counts": {str(float(alpha)): int(count) for alpha, count in zip(uniques, counts)},
        }
    return summary


def build_short_answer(
    metrics_df: pd.DataFrame,
    mode_summary_df: pd.DataFrame,
    payloads_by_k: Dict[int, Dict[str, object]],
) -> str:
    best_r2 = metrics_df.sort_values("R2", ascending=False).iloc[0]
    best_rmse = metrics_df.sort_values("RMSE", ascending=True).iloc[0]
    baseline_r2 = BASELINE_METRICS["R2"]
    baseline_rmse = BASELINE_METRICS["RMSE"]
    best_k = int(best_rmse["k"])
    best_desc = payloads_by_k[best_k]["mode_descriptions"][0]
    stability = mode_summary_df.loc[mode_summary_df["k"] == best_k, "mean_pairwise_map_correlation"].mean()
    usefulness = "does" if float(best_r2["R2"]) > 0.0 else "does not"
    comparison = "does not beat"
    if float(best_r2["R2"]) >= baseline_r2 or float(best_rmse["RMSE"]) <= baseline_rmse:
        comparison = "beats or matches"
    elif float(best_r2["R2"]) >= baseline_r2 - 0.10:
        comparison = "approaches"
    stability_phrase = "moderately stable" if stability >= 0.4 else "not very stable"
    return (
        f"The direct supervised SST encoder {usefulness} provide useful strict LOYO predictability, "
        f"with K={best_k} giving the best RMSE and {best_r2['model_name']} giving the best R2. "
        f"Relative to the Z1_Z2 + AMV/AMO K5 baseline, the best direct encoder {comparison} that reference. "
        f"The learned activation modes are {stability_phrase} across folds and look most like {best_desc}."
    )


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()

    target_df = load_target_table()
    raw_cube, lat, lon, feature_metadata = build_sst_feature_matrix()
    y_raw = target_df["obs_swe"].to_numpy(dtype=float)
    valid_feature_mask = np.asarray(feature_metadata["valid_feature_mask"], dtype=bool)

    metadata_payload = {
        "input_sst_file": str(SST_FILE),
        "target_file": str(TARGET_TABLE_CSV),
        "domain": feature_metadata["domain"],
        "months": MONTHS,
        "water_years": WATER_YEARS.astype(int).tolist(),
        "feature_shape": feature_metadata["feature_shape_before_mask"],
        "num_valid_features": int(feature_metadata["num_valid_features"]),
        "area_weighting": feature_metadata["area_weighting"],
        "feature_mapping": feature_metadata["feature_mapping"],
        "source_variable": feature_metadata["source_variable"],
    }
    METADATA_JSON.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    prediction_frames: List[pd.DataFrame] = []
    alpha_frames: List[pd.DataFrame] = []
    beta_frames: List[pd.DataFrame] = []
    score_frames: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []
    mode_frames: List[pd.DataFrame] = []
    payloads_by_k: Dict[int, Dict[str, object]] = {}

    for k in range(1, 6):
        pred_df, alpha_df, beta_df, score_df, payload = run_model(
            k=k,
            raw_cube=raw_cube,
            y_raw=y_raw,
            lat=lat,
            lon=lon,
            valid_feature_mask=valid_feature_mask,
        )
        prediction_frames.append(pred_df)
        alpha_frames.append(alpha_df)
        beta_frames.append(beta_df)
        score_frames.append(score_df)
        metric_rows.append(payload["metrics"])
        mode_frames.append(payload["mode_summary"])
        payloads_by_k[k] = payload
        write_fold_map_netcdf(k, payload, lat, lon, valid_feature_mask)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    alpha_df = pd.concat(alpha_frames, ignore_index=True)
    beta_df = pd.concat(beta_frames, ignore_index=True)
    score_df = pd.concat(score_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows).sort_values("k").reset_index(drop=True)
    mode_summary_df = pd.concat(mode_frames, ignore_index=True).sort_values(["k", "component"]).reset_index(drop=True)
    period_metrics_df = compute_period_metrics(predictions_df)

    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)
    score_df.to_csv(SCORES_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_metrics_df.to_csv(PERIOD_METRICS_CSV, index=False)
    mode_summary_df.to_csv(MODE_SUMMARY_CSV, index=False)
    write_mean_stability_netcdf(payloads_by_k, mode_summary_df, lat, lon)

    plot_observed_vs_predicted(predictions_df)
    plot_scatter(predictions_df, metrics_df)
    plot_error_by_year(predictions_df)
    best_rmse_row = metrics_df.sort_values("RMSE", ascending=True).iloc[0]
    best_k = int(best_rmse_row["k"])
    plot_component_maps(best_k, payloads_by_k[best_k], mode_summary_df, lat, lon)
    plot_component_stability(mode_summary_df)

    best_r2_row = metrics_df.sort_values("R2", ascending=False).iloc[0]
    best_sign_row = metrics_df.sort_values("sign_accuracy", ascending=False).iloc[0]
    alpha_summary = summarize_alpha_choices(alpha_df)
    short_answer = build_short_answer(metrics_df, mode_summary_df, payloads_by_k)

    summary_payload = {
        "input_sst_file": str(SST_FILE),
        "target_file": str(TARGET_TABLE_CSV),
        "domain": feature_metadata["domain"],
        "months": MONTHS,
        "water_years": [int(WATER_YEAR_START), int(WATER_YEAR_END)],
        "feature_shape": feature_metadata["feature_shape_before_mask"],
        "num_valid_features": int(feature_metadata["num_valid_features"]),
        "area_weighting": feature_metadata["area_weighting"],
        "models": MODEL_NAMES,
        "alpha_grid": ALPHA_GRID.astype(float).tolist(),
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_metrics_df.to_dict(orient="records"),
        "selected_alpha_summary": alpha_summary,
        "best_model_by_R2": best_r2_row.to_dict(),
        "best_model_by_RMSE": best_rmse_row.to_dict(),
        "best_model_by_sign_accuracy": best_sign_row.to_dict(),
        "mode_stability_summary": mode_summary_df.to_dict(orient="records"),
        "comparison_to_current_baseline": {
            "baseline_name": "Z1_Z2_AMV_AMO_K5",
            **BASELINE_METRICS,
        },
        "short_answer": short_answer,
        "mode_description_by_k": {
            f"K{k}": payloads_by_k[k]["mode_descriptions"]
            for k in range(1, 6)
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print("Metrics:", flush=True)
    for _, row in metrics_df.iterrows():
        print(
            f"{row['model_name']}: R2={row['R2']:.6f} RMSE={row['RMSE']:.6f} "
            f"sign_accuracy={row['sign_accuracy']:.6f} alpha_mode={row['selected_alpha_mode']}",
            flush=True,
        )
    print(f"Best model by R2: {best_r2_row['model_name']}", flush=True)
    print(f"Best model by RMSE: {best_rmse_row['model_name']}", flush=True)
    print(f"Best model by sign accuracy: {best_sign_row['model_name']}", flush=True)
    print("Mode stability summary:", flush=True)
    for k in range(1, 6):
        sub = mode_summary_df[mode_summary_df["k"] == k]
        print(
            f"K{k}: mean_pairwise_corr={sub['mean_pairwise_map_correlation'].mean():.4f} "
            f"min_pairwise_corr={sub['min_pairwise_map_correlation'].min():.4f}",
            flush=True,
        )
    print(
        "Comparison to Z1_Z2 + AMV/AMO K5 baseline: "
        f"baseline_R2={BASELINE_METRICS['R2']:.6f} baseline_RMSE={BASELINE_METRICS['RMSE']:.6f} "
        f"baseline_sign_accuracy={BASELINE_METRICS['sign_accuracy']:.6f}",
        flush=True,
    )
    print(f"Short answer: {short_answer}", flush=True)


if __name__ == "__main__":
    main()
