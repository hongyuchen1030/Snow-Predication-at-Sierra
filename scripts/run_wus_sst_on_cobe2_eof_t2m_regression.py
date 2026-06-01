#!/usr/bin/env python3
"""
Project monthly WUS-D3 SST anomalies onto COBE2 global EOFs and predict
monthly WUS-D3 overland T2m anomalies with leave-one-water-year-out ridge regression.
"""

import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    ensure_runtime_on_compute_node,
    get_runtime,
    open_dataset_with_fallbacks,
)


DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
N_MODES = 6
ALPHA_GRID = np.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)

COBE2_EOF_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "sst_pca"
    / "cobe2_global_monthly_climatology_anomaly"
    / "cobe2_global_monthly_clim_sst_eofs.nc"
)
WUS_SST_MONTHLY_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_sst_on_cobe2_grid_monthly"
    f"/{DATASET_ID}/{DATASET_ID}_tskin_on_cobe2_grid_monthly_mean.nc"
)
WUS_T2_MONTHLY_ANOM_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_t2_monthly_anomalies"
    f"/{DATASET_ID}/{DATASET_ID}_t2_monthly_anomaly.nc"
)

PROJECTED_DIR = PROJECT_ROOT / "artifacts" / "wus_sst_projected_onto_cobe2_eofs" / DATASET_ID
REGRESSION_DIR = PROJECT_ROOT / "artifacts" / "wus_sst_on_cobe2_eof_t2m_regression" / DATASET_ID


@dataclass(frozen=True)
class Cobe2Reference:
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    eof: np.ndarray
    pc: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class ProjectionResult:
    time: np.ndarray
    projected_pc: np.ndarray
    projected_pc_correlations: np.ndarray
    projected_pc_std: np.ndarray
    mask: np.ndarray
    weighting_formula: str


def ensure_output_dirs() -> None:
    PROJECTED_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)


def format_date(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def to_month_start(times: Sequence[np.datetime64]) -> np.ndarray:
    values = np.asarray(times, dtype="datetime64[ns]")
    return values.astype("datetime64[M]").astype("datetime64[ns]")


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


def load_cobe2_reference() -> Cobe2Reference:
    with open_dataset_with_fallbacks(COBE2_EOF_FILE) as ds:
        return Cobe2Reference(
            time=np.asarray(ds["time"].values, dtype="datetime64[ns]"),
            latitude=np.asarray(ds["lat"].values, dtype=np.float64),
            longitude=np.asarray(ds["lon"].values, dtype=np.float64),
            eof=np.asarray(ds["eof"].values[:N_MODES], dtype=np.float64),
            pc=np.asarray(ds["pc"].values[:, :N_MODES], dtype=np.float64),
            valid_mask=np.asarray(ds["valid_mask"].values, dtype=bool),
        )


def load_wus_sst_monthly() -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    with open_dataset_with_fallbacks(WUS_SST_MONTHLY_FILE) as ds:
        values = np.asarray(ds["tskin"].values, dtype=np.float64)
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        attrs = dict(ds.attrs)
    return time, values, attrs


def load_wus_t2_monthly_anomaly() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(WUS_T2_MONTHLY_ANOM_FILE) as ds:
        values = np.asarray(ds["t2_anomaly"].values, dtype=np.float64)
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
        if "landmask" in ds.coords:
            landmask = np.asarray(ds["landmask"].values, dtype=np.int8)
        else:
            landmask = np.isfinite(values[0]).astype(np.int8)
    return time, values, latitude, longitude, landmask


def compute_projection(
    cobe2: Cobe2Reference,
    wus_sst_time: np.ndarray,
    wus_sst_values: np.ndarray,
    overlap_months: np.ndarray,
) -> ProjectionResult:
    wus_sst_overlap = select_by_months(wus_sst_time, wus_sst_values, overlap_months)
    cobe2_pc_overlap = select_by_months(cobe2.time, cobe2.pc, overlap_months)

    shared_mask = cobe2.valid_mask & np.isfinite(wus_sst_overlap).all(axis=0)
    if int(np.count_nonzero(shared_mask)) == 0:
        raise ValueError("No shared valid cells remain for WUS SST projection")

    lat_weights_1d = np.sqrt(np.clip(np.cos(np.deg2rad(cobe2.latitude)), 0.0, None))
    weights_2d = np.broadcast_to(lat_weights_1d[:, np.newaxis], shared_mask.shape)

    weighted_anom = np.asarray(wus_sst_overlap[:, shared_mask], dtype=np.float64) * weights_2d[shared_mask][np.newaxis, :]
    weighted_eof = np.asarray(cobe2.eof[:, shared_mask], dtype=np.float64) * weights_2d[shared_mask][np.newaxis, :]
    projected_pc = weighted_anom @ weighted_eof.T

    correlations = np.full(N_MODES, np.nan, dtype=np.float64)
    for mode_index in range(N_MODES):
        left = projected_pc[:, mode_index]
        right = cobe2_pc_overlap[:, mode_index]
        finite = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(finite) >= 3:
            correlations[mode_index] = np.corrcoef(left[finite], right[finite])[0, 1]

    return ProjectionResult(
        time=overlap_months,
        projected_pc=projected_pc.astype(np.float32),
        projected_pc_correlations=correlations,
        projected_pc_std=np.std(projected_pc, axis=0, ddof=1),
        mask=shared_mask,
        weighting_formula="sqrt(cos(lat)) applied to both anomalies and EOF loadings during projection",
    )


def flatten_land_targets(values: np.ndarray, landmask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_land = (landmask == 1) & np.isfinite(values).all(axis=0)
    if int(np.count_nonzero(valid_land)) == 0:
        raise ValueError("No all-time-finite land cells in the WUS T2 anomaly target")
    flattened = np.asarray(values[:, valid_land], dtype=np.float64)
    return flattened, valid_land


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0, ddof=0)
    x_std = np.where(x_std > 0.0, x_std, 1.0)
    x_train_scaled = (x_train - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(x_train_scaled, y_train)
    return model.predict(x_test_scaled)


def r2_per_cell(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean[np.newaxis, :]) ** 2, axis=0)
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = ss_tot > 0.0
    out[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    return out


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(~np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


def choose_alpha_nested(x_train: np.ndarray, y_train: np.ndarray, block_year_train: np.ndarray) -> float:
    unique_years = np.unique(block_year_train)
    if unique_years.size <= 1:
        return float(ALPHA_GRID[0])

    alpha_scores: List[Tuple[float, float]] = []
    for alpha in ALPHA_GRID:
        fold_scores = []
        for holdout in unique_years:
            inner_train = block_year_train != holdout
            inner_val = block_year_train == holdout
            if np.count_nonzero(inner_train) < 12 or np.count_nonzero(inner_val) == 0:
                continue
            pred = fit_predict_ridge(x_train[inner_train], y_train[inner_train], x_train[inner_val], float(alpha))
            fold_scores.append(safe_nanmean(r2_per_cell(y_train[inner_val], pred)))
        score = safe_nanmean(np.asarray(fold_scores, dtype=np.float64)) if fold_scores else -np.inf
        alpha_scores.append((float(alpha), float(score)))
    alpha_scores.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return float(alpha_scores[0][0])


def run_loyo_regression(x: np.ndarray, y: np.ndarray, block_years: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_years = np.unique(block_years)
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    selected_alpha = np.full(unique_years.size, np.nan, dtype=np.float64)

    for fold_index, holdout in enumerate(unique_years):
        train_mask = block_years != holdout
        test_mask = block_years == holdout
        alpha = choose_alpha_nested(x[train_mask], y[train_mask], block_years[train_mask])
        selected_alpha[fold_index] = alpha
        predictions[test_mask] = fit_predict_ridge(x[train_mask], y[train_mask], x[test_mask], alpha)
        print(
            "Outer fold %d/%d sep_aug_year=%d alpha=%.6g test_months=%d"
            % (fold_index + 1, unique_years.size, int(holdout), alpha, int(np.count_nonzero(test_mask))),
            flush=True,
        )
    return predictions, selected_alpha


def compute_grid_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r2 = r2_per_cell(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    corr = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for idx in range(y_true.shape[1]):
        left = y_true[:, idx]
        right = y_pred[:, idx]
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            continue
        corr[idx] = np.corrcoef(left, right)[0, 1]
    return r2, corr, rmse, mae


def expand_to_grid(values_flat: np.ndarray, valid_land: np.ndarray) -> np.ndarray:
    grid = np.full(valid_land.shape, np.nan, dtype=np.float64)
    grid[valid_land] = values_flat
    return grid


def save_projected_outputs(result: ProjectionResult) -> None:
    np.save(PROJECTED_DIR / "projected_pc_timeseries.npy", result.projected_pc)
    with (PROJECTED_DIR / "projected_pc_timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time"] + ["PC%d" % mode for mode in range(1, N_MODES + 1)])
        for time_value, row in zip(result.time, result.projected_pc):
            writer.writerow([format_date(time_value)] + ["%.12g" % float(value) for value in row])

    metadata = {
        "dataset_id": DATASET_ID,
        "time_start": format_date(result.time[0]),
        "time_end": format_date(result.time[-1]),
        "n_time": int(result.time.size),
        "n_modes": int(result.projected_pc.shape[1]),
        "projection_shared_cell_count": int(np.count_nonzero(result.mask)),
        "weighting": result.weighting_formula,
        "cobe2_overlap_correlations": [float(value) for value in result.projected_pc_correlations.tolist()],
        "projected_pc_std": [float(value) for value in result.projected_pc_std.tolist()],
        "wus_sst_monthly_file": str(WUS_SST_MONTHLY_FILE),
        "cobe2_eof_file": str(COBE2_EOF_FILE),
    }
    (PROJECTED_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(N_MODES, 1, figsize=(12, 2.0 * N_MODES), sharex=True, constrained_layout=True)
    if N_MODES == 1:
        axes = [axes]
    time_plot = result.time.astype("datetime64[ns]")
    for mode_index, ax in enumerate(axes):
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.plot(time_plot, result.projected_pc[:, mode_index], color="black", linewidth=1.0)
        corr_value = result.projected_pc_correlations[mode_index]
        ax.set_title("Projected PC%d (std=%.3f, corr with COBE2 PC%d=%.3f)" % (
            mode_index + 1,
            float(result.projected_pc_std[mode_index]),
            mode_index + 1,
            float(corr_value),
        ))
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("WUS-on-COBE2 projected monthly SST pseudo-PCs", fontsize=14)
    fig.savefig(PROJECTED_DIR / "projected_pc_timeseries_modes1to6.png", dpi=220)
    plt.close(fig)


def save_metric_dataset(
    latitude: np.ndarray,
    longitude: np.ndarray,
    r2_map: np.ndarray,
    corr_map: np.ndarray,
    rmse_map: np.ndarray,
    mae_map: np.ndarray,
    coefficient_maps: np.ndarray,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "r2": (("lat2d", "lon2d"), r2_map.astype(np.float32)),
            "correlation": (("lat2d", "lon2d"), corr_map.astype(np.float32)),
            "rmse": (("lat2d", "lon2d"), rmse_map.astype(np.float32)),
            "mae": (("lat2d", "lon2d"), mae_map.astype(np.float32)),
            "coefficient": (("mode", "lat2d", "lon2d"), coefficient_maps.astype(np.float32)),
        },
        coords={
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude.astype(np.float32)),
        },
        attrs={
            "description": "WUS-D3 SST projected onto COBE2 EOFs used to predict WUS-D3 overland T2m",
        },
    )
    ds.to_netcdf(REGRESSION_DIR / "gridcell_metrics_and_coefficients.nc")


def plot_single_map(data: np.ndarray, title: str, output_name: str, cmap: str, symmetric: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("No finite values available for plot %s" % output_name)
    vmax = float(np.nanmax(np.abs(data))) if symmetric else float(np.nanmax(data))
    vmin = -vmax if symmetric else float(np.nanmin(data))
    mesh = ax.pcolormesh(data, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("lon2d index")
    ax.set_ylabel("lat2d index")
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.savefig(REGRESSION_DIR / output_name, dpi=220)
    plt.close(fig)


def plot_coefficient_maps(coefficient_maps: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes_flat = axes.ravel()
    vmax = float(np.nanmax(np.abs(coefficient_maps)))
    for mode_index in range(N_MODES):
        ax = axes_flat[mode_index]
        mesh = ax.pcolormesh(coefficient_maps[mode_index], cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title("Mode %d coefficient" % (mode_index + 1))
        ax.set_xlabel("lon2d index")
        ax.set_ylabel("lat2d index")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.suptitle("WUS-D3 SST projected onto COBE2 EOFs used to predict WUS-D3 overland T2m", fontsize=14)
    fig.savefig(REGRESSION_DIR / "coefficient_maps_modes1to6.png", dpi=220)
    plt.close(fig)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dirs()

    if not WUS_SST_MONTHLY_FILE.exists():
        raise FileNotFoundError("Missing WUS monthly SST file: %s" % WUS_SST_MONTHLY_FILE)
    if not WUS_T2_MONTHLY_ANOM_FILE.exists():
        raise FileNotFoundError("Missing WUS monthly T2 anomaly file: %s" % WUS_T2_MONTHLY_ANOM_FILE)

    cobe2 = load_cobe2_reference()
    wus_sst_time, wus_sst_values, wus_sst_attrs = load_wus_sst_monthly()
    wus_t2_time, wus_t2_values, latitude, longitude, landmask = load_wus_t2_monthly_anomaly()

    overlap_months = intersect_months(cobe2.time, wus_sst_time, wus_t2_time)
    if overlap_months.size == 0:
        raise ValueError("No common monthly overlap between COBE2, WUS SST, and WUS T2")

    projection = compute_projection(cobe2, wus_sst_time, wus_sst_values, overlap_months)
    save_projected_outputs(projection)

    y_overlap = select_by_months(wus_t2_time, wus_t2_values, overlap_months)
    y_flat, valid_land = flatten_land_targets(y_overlap, landmask)
    x = np.asarray(projection.projected_pc, dtype=np.float64)
    block_years = september_august_years_from_time(overlap_months)
    unique_years, counts = np.unique(block_years, return_counts=True)
    full_years = unique_years[counts == 12]
    full_mask = np.isin(block_years, full_years)
    overlap_months = overlap_months[full_mask]
    x = x[full_mask]
    y_flat = y_flat[full_mask]
    block_years = block_years[full_mask]

    print(
        "Aligned monthly data: months=%d start=%s end=%s predictor_shape=%s target_shape=%s sep_aug_years=%d"
        % (
            overlap_months.size,
            format_date(overlap_months[0]),
            format_date(overlap_months[-1]),
            tuple(int(v) for v in x.shape),
            tuple(int(v) for v in y_flat.shape),
            int(np.unique(block_years).size),
        ),
        flush=True,
    )

    y_pred, selected_alpha = run_loyo_regression(x, y_flat, block_years)
    r2, corr, rmse, mae = compute_grid_metrics(y_flat, y_pred)

    alpha_mode = Counter([float(value) for value in selected_alpha.tolist()]).most_common(1)[0][0]
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    x_std = np.where(x_std > 0.0, x_std, 1.0)
    final_model = Ridge(alpha=float(alpha_mode), fit_intercept=True)
    final_model.fit((x - x_mean) / x_std, y_flat)
    coefficients_flat = (final_model.coef_ / x_std[np.newaxis, :]).T

    r2_map = expand_to_grid(r2, valid_land)
    corr_map = expand_to_grid(corr, valid_land)
    rmse_map = expand_to_grid(rmse, valid_land)
    mae_map = expand_to_grid(mae, valid_land)
    coefficient_maps = np.full((N_MODES,) + valid_land.shape, np.nan, dtype=np.float64)
    for mode_index in range(N_MODES):
        coefficient_maps[mode_index][valid_land] = coefficients_flat[mode_index]

    save_metric_dataset(latitude, longitude, r2_map, corr_map, rmse_map, mae_map, coefficient_maps)
    np.save(REGRESSION_DIR / "X_projected_pcs.npy", x.astype(np.float32))
    np.save(REGRESSION_DIR / "Y_t2_anomaly.npy", y_flat.astype(np.float32))
    np.save(REGRESSION_DIR / "Y_pred_t2_anomaly.npy", y_pred.astype(np.float32))
    np.save(REGRESSION_DIR / "sep_aug_years.npy", block_years.astype(np.int32))

    with (REGRESSION_DIR / "selected_alpha_by_outer_fold.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sep_aug_year", "alpha"])
        for block_year, alpha in zip(np.unique(block_years), selected_alpha):
            writer.writerow([int(block_year), "%.12g" % float(alpha)])

    summary = {
        "dataset_id": DATASET_ID,
        "runtime_hostname": runtime.hostname,
        "runtime_slurm_job_id": runtime.slurm_job_id,
        "wus_sst_monthly_file": str(WUS_SST_MONTHLY_FILE),
        "wus_t2_monthly_anomaly_file": str(WUS_T2_MONTHLY_ANOM_FILE),
        "cobe2_eof_file": str(COBE2_EOF_FILE),
        "date_range_start": format_date(overlap_months[0]),
        "date_range_end": format_date(overlap_months[-1]),
        "n_common_months": int(overlap_months.size),
        "projection_matrix_shape": [int(x.shape[0]), int(x.shape[1])],
        "target_matrix_shape": [int(y_flat.shape[0]), int(y_flat.shape[1])],
        "n_land_gridcells": int(np.count_nonzero(valid_land)),
        "selected_alpha_mode": float(alpha_mode),
        "selected_alphas": [float(value) for value in selected_alpha.tolist()],
        "sep_aug_years_start": int(np.min(block_years)),
        "sep_aug_years_end": int(np.max(block_years)),
        "projected_pc_std": [float(value) for value in projection.projected_pc_std.tolist()],
        "projected_pc_correlation_with_cobe2": [float(value) for value in projection.projected_pc_correlations.tolist()],
        "mean_r2": float(np.nanmean(r2)),
        "mean_correlation": float(np.nanmean(corr)),
        "mean_rmse": float(np.nanmean(rmse)),
        "mean_mae": float(np.nanmean(mae)),
        "weighting": projection.weighting_formula,
        "wus_sst_source_attrs": wus_sst_attrs,
    }
    (REGRESSION_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_single_map(r2_map, "R2 map: WUS-D3 SST projected onto COBE2 EOFs -> WUS overland T2m", "r2_map.png", "viridis")
    plot_single_map(corr_map, "Correlation map: WUS-D3 SST projected onto COBE2 EOFs -> WUS overland T2m", "correlation_map.png", "coolwarm", symmetric=True)
    plot_single_map(rmse_map, "RMSE map (K): WUS-D3 SST projected onto COBE2 EOFs -> WUS overland T2m", "rmse_map.png", "magma")
    plot_single_map(mae_map, "MAE map (K): WUS-D3 SST projected onto COBE2 EOFs -> WUS overland T2m", "mae_map.png", "magma")
    plot_coefficient_maps(coefficient_maps)

    print("Projected output directory: %s" % PROJECTED_DIR, flush=True)
    print("Regression output directory: %s" % REGRESSION_DIR, flush=True)
    print("Mean metrics: R2=%.6f corr=%.6f RMSE=%.6f MAE=%.6f" % (
        float(np.nanmean(r2)),
        float(np.nanmean(corr)),
        float(np.nanmean(rmse)),
        float(np.nanmean(mae)),
    ), flush=True)


if __name__ == "__main__":
    main()
