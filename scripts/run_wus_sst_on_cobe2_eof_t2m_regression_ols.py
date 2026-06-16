#!/usr/bin/env python3
"""
Lightweight OLS counterpart to the WUS projected-PC -> WUS-D3 T2m regression.

Reuses saved projected PCs and saved monthly T2 anomalies only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
DOMAIN = "d03"
N_MODES = 6
PROJECTED_PC_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "wus_sst_projected_onto_cobe2_eofs"
    / DOMAIN
    / DATASET_ID
    / "projected_pc_timeseries_and_mask.nc"
)
T2_ANOM_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_t2_monthly_anomalies"
    f"/{DOMAIN}/{DATASET_ID}/{DATASET_ID}_{DOMAIN}_t2_monthly_anomaly.nc"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "wus_sst_on_cobe2_eof_t2m_regression_ols" / DATASET_ID


def to_month_start(values: Sequence[np.datetime64]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("datetime64[M]").astype("datetime64[ns]")


def intersect_months(*time_axes: Sequence[np.datetime64]) -> np.ndarray:
    common = to_month_start(time_axes[0])
    for axis in time_axes[1:]:
        common = np.intersect1d(common, to_month_start(axis), assume_unique=False)
    return np.asarray(common, dtype="datetime64[ns]")


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    month_values = to_month_start(time_values)
    index_by_month = {month: idx for idx, month in enumerate(month_values.tolist())}
    return np.asarray(data)[[index_by_month[month] for month in target_months.tolist()]]


def september_august_years_from_time(time_values: np.ndarray) -> np.ndarray:
    months = np.array([int(np.datetime_as_string(value, unit="D")[5:7]) for value in time_values], dtype=np.int32)
    years = np.array([int(np.datetime_as_string(value, unit="D")[:4]) for value in time_values], dtype=np.int32)
    return np.where(months >= 9, years, years - 1)


def load_projected_pcs() -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(PROJECTED_PC_FILE) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        x = np.asarray(ds["projected_pc"].values, dtype=np.float64)
    return time, x


def load_t2_anomaly() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(T2_ANOM_FILE) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        values = np.asarray(ds["t2_anomaly"].values, dtype=np.float64)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return time, values, lat, lon


def flatten_land_targets(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_land = np.isfinite(values).all(axis=0)
    if int(np.count_nonzero(valid_land)) == 0:
        raise ValueError("No all-time-finite land cells")
    return np.asarray(values[:, valid_land], dtype=np.float64), valid_land


def fit_predict_ols(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0, ddof=1)
    x_std = np.where(x_std > 0.0, x_std, 1.0)
    x_train_std = (x_train - x_mean) / x_std
    x_test_std = (x_test - x_mean) / x_std
    a_train = np.concatenate([np.ones((x_train_std.shape[0], 1)), x_train_std], axis=1)
    a_test = np.concatenate([np.ones((x_test_std.shape[0], 1)), x_test_std], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(a_train, y_train, rcond=None)
    return a_test @ coeffs


def run_loyo_ols(x: np.ndarray, y: np.ndarray, block_years: np.ndarray) -> np.ndarray:
    unique_years = np.unique(block_years)
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    for holdout in unique_years:
        train_mask = block_years != holdout
        test_mask = block_years == holdout
        predictions[test_mask] = fit_predict_ols(x[train_mask], y[train_mask], x[test_mask])
    return predictions


def r2_per_cell(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean[np.newaxis, :]) ** 2, axis=0)
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = ss_tot > 0.0
    out[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    return out


def compute_grid_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r2 = r2_per_cell(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    corr = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for idx in range(y_true.shape[1]):
        if np.std(y_true[:, idx]) == 0.0 or np.std(y_pred[:, idx]) == 0.0:
            continue
        corr[idx] = np.corrcoef(y_true[:, idx], y_pred[:, idx])[0, 1]
    return r2, corr, rmse, mae


def expand_to_grid(values: np.ndarray, valid_land: np.ndarray) -> np.ndarray:
    grid = np.full(valid_land.shape, np.nan, dtype=np.float64)
    grid[valid_land] = values
    return grid


def fit_full_ols_coefficients_standardized(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    x_std = np.where(x_std > 0.0, x_std, 1.0)
    x_stdzd = (x - x_mean) / x_std
    a = np.concatenate([np.ones((x_stdzd.shape[0], 1)), x_stdzd], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    intercept = coeffs[0]
    beta_std = coeffs[1:]
    return intercept, beta_std, x_std


def plot_single_map(field: np.ndarray, title: str, output_name: str, cmap: str, symmetric: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    values = np.asarray(field, dtype=np.float64)
    if symmetric:
        vmax = float(np.nanmax(np.abs(values)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    mesh = ax.pcolormesh(values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("lon2d index")
    ax.set_ylabel("lat2d index")
    fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.savefig(OUTPUT_DIR / output_name, dpi=220)
    plt.close(fig)


def plot_coefficient_maps(coefficient_maps: np.ndarray, suffix: str, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes_flat = axes.ravel()
    vmax = float(np.nanmax(np.abs(coefficient_maps)))
    for mode_index in range(N_MODES):
        ax = axes_flat[mode_index]
        mesh = ax.pcolormesh(coefficient_maps[mode_index], cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Mode {mode_index + 1} coefficient")
        ax.set_xlabel("lon2d index")
        ax.set_ylabel("lat2d index")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(OUTPUT_DIR / suffix, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pc_time, x = load_projected_pcs()
    t2_time, t2_values, latitude, longitude = load_t2_anomaly()
    overlap_months = intersect_months(pc_time, t2_time)
    x = select_by_months(pc_time, x, overlap_months)
    t2 = select_by_months(t2_time, t2_values, overlap_months)
    block_years = september_august_years_from_time(overlap_months)
    y_flat, valid_land = flatten_land_targets(t2)

    y_pred = run_loyo_ols(x, y_flat, block_years)
    r2, corr, rmse, mae = compute_grid_metrics(y_flat, y_pred)
    intercept, beta_std, x_std = fit_full_ols_coefficients_standardized(x, y_flat)
    beta_raw = beta_std / x_std[:, np.newaxis]

    r2_map = expand_to_grid(r2, valid_land)
    corr_map = expand_to_grid(corr, valid_land)
    rmse_map = expand_to_grid(rmse, valid_land)
    mae_map = expand_to_grid(mae, valid_land)
    coef_std_maps = np.full((N_MODES,) + valid_land.shape, np.nan, dtype=np.float64)
    coef_raw_maps = np.full((N_MODES,) + valid_land.shape, np.nan, dtype=np.float64)
    for mode_index in range(N_MODES):
        coef_std_maps[mode_index][valid_land] = beta_std[mode_index]
        coef_raw_maps[mode_index][valid_land] = beta_raw[mode_index]

    ds = xr.Dataset(
        data_vars={
            "r2": (("lat2d", "lon2d"), r2_map.astype(np.float32)),
            "correlation": (("lat2d", "lon2d"), corr_map.astype(np.float32)),
            "rmse": (("lat2d", "lon2d"), rmse_map.astype(np.float32)),
            "mae": (("lat2d", "lon2d"), mae_map.astype(np.float32)),
            "coefficient_standardized_pc": (("mode", "lat2d", "lon2d"), coef_std_maps.astype(np.float32)),
            "coefficient_raw_pc": (("mode", "lat2d", "lon2d"), coef_raw_maps.astype(np.float32)),
            "projected_pc_std_raw": (("mode",), x.std(axis=0, ddof=1).astype(np.float32)),
        },
        coords={
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude.astype(np.float32)),
        },
        attrs={
            "description": "OLS counterpart for WUS projected-PC to WUS-D3 T2m regression",
            "formula_prediction": "Y_hat = A_hat B_hat with B_hat = (A^T A)^(-1) A^T Y on standardized predictors plus intercept",
            "formula_beta_standardized": "coefficient_standardized_pc is K per 1 sigma projected PC",
            "formula_beta_raw": "coefficient_raw_pc is K per 1 raw projected-PC unit",
            "pc_standardization": "Predictors standardized using sample mean/std over all overlap months (ddof=1) for final OLS coefficient fit",
            "dataset_id": DATASET_ID,
            "domain": DOMAIN,
        },
    )
    ds.to_netcdf(OUTPUT_DIR / "gridcell_metrics_and_coefficients_ols.nc")

    np.save(OUTPUT_DIR / "X_projected_pcs.npy", x.astype(np.float32))
    np.save(OUTPUT_DIR / "Y_t2_anomaly.npy", y_flat.astype(np.float32))
    np.save(OUTPUT_DIR / "Y_pred_t2_anomaly.npy", y_pred.astype(np.float32))
    np.save(OUTPUT_DIR / "sep_aug_years.npy", block_years.astype(np.int32))

    summary = {
        "dataset_id": DATASET_ID,
        "domain": DOMAIN,
        "projected_pc_file": str(PROJECTED_PC_FILE),
        "t2_anomaly_file": str(T2_ANOM_FILE),
        "n_common_months": int(overlap_months.size),
        "date_range_start": str(overlap_months[0].astype("datetime64[D]")),
        "date_range_end": str(overlap_months[-1].astype("datetime64[D]")),
        "projection_matrix_shape": [int(x.shape[0]), int(x.shape[1])],
        "target_matrix_shape": [int(y_flat.shape[0]), int(y_flat.shape[1])],
        "n_land_gridcells": int(np.count_nonzero(valid_land)),
        "regression_type": "ordinary_least_squares",
        "pc_standardized_for_final_fit": True,
        "pc_standardization_ddof": 1,
        "intercept_included": True,
        "mean_r2": float(np.nanmean(r2)),
        "mean_correlation": float(np.nanmean(corr)),
        "mean_rmse": float(np.nanmean(rmse)),
        "mean_mae": float(np.nanmean(mae)),
        "projected_pc_std_raw": [float(v) for v in x.std(axis=0, ddof=1).tolist()],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (OUTPUT_DIR / "method_notes.txt").open("w", encoding="utf-8") as handle:
        handle.write("Final fit uses OLS on standardized projected PCs with intercept.\n")
        handle.write("Saved coefficient_standardized_pc maps are directly interpretable as K per 1 sigma projected PC.\n")
        handle.write("Saved coefficient_raw_pc maps convert those coefficients back to raw projected-PC units.\n")

    plot_single_map(r2_map, "OLS R2 map", "r2_map.png", "viridis")
    plot_single_map(corr_map, "OLS correlation map", "correlation_map.png", "coolwarm", symmetric=True)
    plot_single_map(rmse_map, "OLS RMSE map (K)", "rmse_map.png", "magma")
    plot_single_map(mae_map, "OLS MAE map (K)", "mae_map.png", "magma")
    plot_coefficient_maps(
        coef_std_maps,
        "coefficient_maps_standardized_pc_modes1to6.png",
        "WUS projected-PC -> WUS-D3 T2m OLS coefficients (standardized PCs)",
    )
    plot_coefficient_maps(
        coef_raw_maps,
        "coefficient_maps_raw_pc_modes1to6.png",
        "WUS projected-PC -> WUS-D3 T2m OLS coefficients (raw PCs)",
    )

    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
