#!/usr/bin/env python3
"""
Run Route 3: Gaussian-smoothed Sierra ERA5-Land T2m predictability using
standardized COBE2 Pacific SST PCs 1-6 as joint predictors.
"""

import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
except ImportError:  # pragma: no cover
    scipy_gaussian_filter = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_cobe2_pacific_sierra_t2m_level1_diagnostic import (
    COBE2_SST_FILE,
    ERA5_MONTHLY_CLIM_FILE,
    ERA5_MONTHLY_MEAN_FILE,
    LAT_CHUNK,
    LON_CHUNK,
    PACIFIC_SST_REGION_360,
    SIERRA_T2M_REGION_360,
    TIME_CHUNK,
    area_weighted_mean,
    build_time_index,
    ensure_runtime_on_compute_node,
    format_month,
    get_runtime,
    load_pacific_cobe2_pca,
    month_number,
    standardize_pc_matrix,
    subset_era5_region_360,
    to_month_start,
)


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_gaussian_smoothing"
MODEL_NAME = "PC1_to_PC6"
N_PREDICTORS = 6
SIGMA_VALUES = [0, 1, 2, 4, 8, 16]
SMOOTH_MASK_MIN_FRACTION = 1.0e-6

PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL2_PC1TO6_ROUTE3_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6_route3_gaussian_smoothing",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_gaussian_smoothing"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_gaussian_smoothing.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_gaussian_smoothing_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_gaussian_smoothing_summary.json"
MEAN_R2_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_mean_r2_vs_sigma.png"
MULTIPANEL_MAP_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_r2_maps.png"
MAP_FIGURE_TEMPLATE = "cobe2_pacific_sierra_t2m_level2_pc1to6_route3_r2_sigma{sigma}.png"

ERA5_VARIABLE = "t2m"


@dataclass(frozen=True)
class SummaryRow:
    sigma: int
    approximate_radius_grid_cells: float
    approximate_radius_km: float
    number_of_valid_grid_points: int
    mean_r2: float
    median_r2: float
    max_r2: float
    min_r2: float


def ensure_output_dir() -> None:
    PSCRATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HOME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def output_dir_size_text() -> str:
    total_bytes = 0
    for path in PSCRATCH_OUTPUT_DIR.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
    units = ["B", "K", "M", "G", "T", "P"]
    size = float(total_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)}{units[unit_index]}"
    return f"{size:.1f}{units[unit_index]}"


def gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(truncate * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float64)


def convolve_along_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    if radius == 0:
        return np.asarray(values, dtype=np.float64).copy()
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(np.asarray(values, dtype=np.float64), pad_width, mode="constant", constant_values=0.0)
    convolved = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)
    return np.asarray(convolved, dtype=np.float64)


def gaussian_filter_2d(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(values, dtype=np.float64).copy()
    if scipy_gaussian_filter is not None:
        return np.asarray(scipy_gaussian_filter(values, sigma=sigma, mode="constant", cval=0.0), dtype=np.float64)
    kernel = gaussian_kernel_1d(sigma)
    smoothed = convolve_along_axis(values, kernel, axis=0)
    smoothed = convolve_along_axis(smoothed, kernel, axis=1)
    return smoothed


def fit_multivariate_regression(
    predictors: np.ndarray,
    anomalies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != N_PREDICTORS:
        raise ValueError(f"Predictor matrix must have shape (n_time, {N_PREDICTORS})")
    if y.ndim != 3 or y.shape[0] != a.shape[0]:
        raise ValueError("Anomaly cube must have shape (n_time, lat, lon) aligned with predictors")

    y_flat = y.reshape(y.shape[0], -1)
    valid_columns = np.isfinite(y_flat).all(axis=0)
    if not np.any(valid_columns):
        raise ValueError("No all-time-finite Sierra grid points available for regression")

    y_valid = y_flat[:, valid_columns]
    b_valid, _, _, _ = np.linalg.lstsq(a, y_valid, rcond=None)
    yhat_valid = a @ b_valid

    residual_sum = np.sum((y_valid - yhat_valid) ** 2, axis=0)
    total_sum = np.sum(y_valid ** 2, axis=0)
    r2_valid = np.full(total_sum.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(total_sum) & (total_sum > 0.0)
    r2_valid[positive] = 1.0 - (residual_sum[positive] / total_sum[positive])

    coefficient_maps = np.full((N_PREDICTORS, y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    coefficient_maps.reshape(N_PREDICTORS, -1)[:, valid_columns] = b_valid

    r2_map = np.full((y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    r2_map.reshape(-1)[valid_columns] = r2_valid
    return coefficient_maps, r2_map


def approximate_sigma_km(sigma: float, latitude: np.ndarray, longitude: np.ndarray) -> float:
    if sigma <= 0.0:
        return 0.0
    lat_spacing_deg = float(np.mean(np.abs(np.diff(latitude))))
    lon_spacing_deg = float(np.mean(np.abs(np.diff(longitude))))
    mean_lat = float(np.mean(latitude))
    lat_km = 111.32 * lat_spacing_deg
    lon_km = 111.32 * math.cos(math.radians(mean_lat)) * lon_spacing_deg
    cell_scale_km = math.sqrt(max(lat_km, 0.0) * max(lon_km, 0.0))
    return float(sigma * cell_scale_km)


def smooth_anomalies_gaussian(
    anomalies: np.ndarray,
    latitude: np.ndarray,
    sigma: float,
) -> np.ndarray:
    values = np.asarray(anomalies, dtype=np.float64)
    if sigma <= 0.0:
        return values.copy()

    land_mask = np.isfinite(values).all(axis=0)
    if not np.any(land_mask):
        raise ValueError("Gaussian smoothing requires at least one valid Sierra land grid point")

    area_weights = np.broadcast_to(np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float64)))[:, np.newaxis], land_mask.shape)
    static_weight_field = np.where(land_mask, area_weights, 0.0)
    smooth_weight_field = gaussian_filter_2d(static_weight_field, sigma=sigma)
    min_weight = SMOOTH_MASK_MIN_FRACTION * float(np.nanmax(smooth_weight_field))
    smoothed = np.full_like(values, np.nan, dtype=np.float64)

    for time_index in range(values.shape[0]):
        weighted_data = np.where(land_mask, values[time_index], 0.0) * static_weight_field
        smooth_weighted_data = gaussian_filter_2d(weighted_data, sigma=sigma)
        smoothed_slice = np.full(land_mask.shape, np.nan, dtype=np.float64)
        valid_denom = smooth_weight_field > min_weight
        smoothed_slice[valid_denom] = smooth_weighted_data[valid_denom] / smooth_weight_field[valid_denom]
        smoothed_slice[~land_mask] = np.nan
        smoothed[time_index] = smoothed_slice

    return smoothed


def summarize_sigma_r2(
    sigma: int,
    r2_map: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> SummaryRow:
    r2_valid = np.asarray(r2_map, dtype=np.float64)
    finite = np.isfinite(r2_valid)
    return SummaryRow(
        sigma=sigma,
        approximate_radius_grid_cells=float(sigma),
        approximate_radius_km=approximate_sigma_km(float(sigma), latitude, longitude),
        number_of_valid_grid_points=int(np.count_nonzero(finite)),
        mean_r2=area_weighted_mean(r2_valid, latitude),
        median_r2=float(np.nanmedian(r2_valid[finite])) if np.any(finite) else float("nan"),
        max_r2=float(np.nanmax(r2_valid)) if np.any(finite) else float("nan"),
        min_r2=float(np.nanmin(r2_valid)) if np.any(finite) else float("nan"),
    )


def save_netcdf(
    overlap_months: np.ndarray,
    predictor_matrix_raw_mean: np.ndarray,
    predictor_matrix_raw_std: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    r2_maps: np.ndarray,
    coefficient_maps: np.ndarray,
    summaries: List[SummaryRow],
    runtime,
) -> None:
    sigma_array = np.asarray([row.sigma for row in summaries], dtype=np.int32)
    ds = xr.Dataset(
        data_vars={
            "pacific_cobe2_pc_mean_raw": (("mode",), predictor_matrix_raw_mean.astype(np.float32)),
            "pacific_cobe2_pc_std_raw": (("mode",), predictor_matrix_raw_std.astype(np.float32)),
            "sigma": (("sigma_case",), sigma_array),
            "approximate_radius_grid_cells": (
                ("sigma_case",),
                np.asarray([row.approximate_radius_grid_cells for row in summaries], dtype=np.float32),
            ),
            "approximate_radius_km": (
                ("sigma_case",),
                np.asarray([row.approximate_radius_km for row in summaries], dtype=np.float32),
            ),
            "valid_grid_point_count": (
                ("sigma_case",),
                np.asarray([row.number_of_valid_grid_points for row in summaries], dtype=np.int32),
            ),
            "mean_r2": (("sigma_case",), np.asarray([row.mean_r2 for row in summaries], dtype=np.float32)),
            "median_r2": (("sigma_case",), np.asarray([row.median_r2 for row in summaries], dtype=np.float32)),
            "max_r2": (("sigma_case",), np.asarray([row.max_r2 for row in summaries], dtype=np.float32)),
            "min_r2": (("sigma_case",), np.asarray([row.min_r2 for row in summaries], dtype=np.float32)),
            "sierra_era5_t2m_multi_pc_r2": (
                ("sigma_case", "sierra_latitude", "sierra_longitude"),
                r2_maps.astype(np.float32),
            ),
            "sierra_era5_t2m_multi_pc_beta": (
                ("sigma_case", "mode", "sierra_latitude", "sierra_longitude"),
                coefficient_maps.astype(np.float32),
            ),
        },
        coords={
            "time": overlap_months.astype("datetime64[ns]"),
            "sigma_case": np.arange(len(summaries), dtype=np.int32),
            "mode": np.arange(1, N_PREDICTORS + 1, dtype=np.int32),
            "sierra_latitude": latitude.astype(np.float32),
            "sierra_longitude": longitude.astype(np.float32),
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "sigma_values_grid_cells": json.dumps(SIGMA_VALUES),
            "pacific_sst_region_360": json.dumps(PACIFIC_SST_REGION_360.as_dict()),
            "sierra_t2m_region_360": json.dumps(SIERRA_T2M_REGION_360.as_dict()),
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "gaussian_smoothing_method": "normalized convolution over Sierra land mask using cosine-latitude area weights",
            "gaussian_kernel": "K_sigma(d) = exp(-d^2 / (2 sigma^2))",
            "formula_bhat": "B_hat_sigma = (A^T A)^(-1) A^T Y_sigma",
            "formula_yhat": "Y_hat_sigma = A B_hat_sigma",
            "formula_r2": "R2_sigma(r) = 1 - sum_t[(Y_sigma(t,r)-Y_hat_sigma(t,r))^2] / sum_t[Y_sigma(t,r)^2]",
            "smooth_mask_min_fraction": SMOOTH_MASK_MIN_FRACTION,
            "time_overlap_start": format_month(overlap_months[0]),
            "time_overlap_end": format_month(overlap_months[-1]),
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(
        NETCDF_FILE,
        engine="netcdf4",
        encoding={
            "sierra_era5_t2m_multi_pc_r2": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
            "sierra_era5_t2m_multi_pc_beta": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
        },
    )


def save_summary(
    summaries: List[SummaryRow],
    pacific: Dict[str, np.ndarray],
    sierra_shape: List[int],
    runtime,
) -> None:
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sigma",
                "approximate_radius_grid_cells",
                "approximate_radius_km",
                "number_of_valid_grid_points",
                "mean_R2",
                "median_R2",
                "max_R2",
                "min_R2",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.sigma,
                    "{:.12g}".format(row.approximate_radius_grid_cells),
                    "{:.12g}".format(row.approximate_radius_km),
                    row.number_of_valid_grid_points,
                    "{:.12g}".format(row.mean_r2),
                    "{:.12g}".format(row.median_r2),
                    "{:.12g}".format(row.max_r2),
                    "{:.12g}".format(row.min_r2),
                ]
            )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "sigma_values_grid_cells": SIGMA_VALUES,
        "pacific_sst_region": PACIFIC_SST_REGION_360.as_dict(),
        "sierra_t2m_region": SIERRA_T2M_REGION_360.as_dict(),
        "input_cobe2_sst_path": str(COBE2_SST_FILE),
        "input_era5_monthly_mean_path": str(ERA5_MONTHLY_MEAN_FILE),
        "input_era5_monthly_climatology_path": str(ERA5_MONTHLY_CLIM_FILE),
        "output_netcdf_path": str(NETCDF_FILE),
        "mean_r2_figure_path": str(MEAN_R2_FIGURE_FILE),
        "map_figure_path": str(MULTIPANEL_MAP_FIGURE_FILE),
        "pc_standardized": True,
        "sierra_t2m_shape": sierra_shape,
        "pacific_sst_shape": [int(pacific["latitude"].size), int(pacific["longitude"].size)],
        "smooth_mask_min_fraction": SMOOTH_MASK_MIN_FRACTION,
        "summary_rows": [asdict(row) for row in summaries],
        "slurm_job_id": runtime.slurm_job_id,
        "compute_node": runtime.hostname,
        "output_directory_size": output_dir_size_text(),
    }
    SUMMARY_JSON_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_r2_map(
    latitude: np.ndarray,
    longitude: np.ndarray,
    r2_map: np.ndarray,
    summary: SummaryRow,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.5), constrained_layout=True)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        r2_map,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=max(0.05, float(np.nanmax(r2_map))),
    )
    ax.set_title(rf"$\sigma={summary.sigma}$ | mean $R^2={summary.mean_r2:.3f}$")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
    ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, shrink=0.88).set_label(r"$R^2_\sigma(r)$")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mean_r2_vs_sigma(summaries: List[SummaryRow]) -> None:
    sigma = np.asarray([row.sigma for row in summaries], dtype=np.float64)
    mean_r2 = np.asarray([row.mean_r2 for row in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(sigma, mean_r2, marker="o", linewidth=2)
    ax.set_xlabel(r"Gaussian smoothing $\sigma$ [grid cells]")
    ax.set_ylabel(r"Area-weighted mean $R^2$")
    ax.set_title(r"Route 3: Sierra T2m predictability vs. Gaussian neighborhood smoothing")
    ax.set_xticks(sigma)
    ax.grid(True, alpha=0.3)
    fig.savefig(MEAN_R2_FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_multipanel_maps(
    latitude: np.ndarray,
    longitude: np.ndarray,
    r2_maps_by_sigma: Dict[int, np.ndarray],
    summaries_by_sigma: Dict[int, SummaryRow],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.28, wspace=0.22)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    for ax, sigma in zip(axes.flat, SIGMA_VALUES):
        r2_map = r2_maps_by_sigma[sigma]
        summary = summaries_by_sigma[sigma]
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            r2_map,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max(0.05, float(np.nanmax(r2_map))),
        )
        ax.set_title(rf"$\sigma={sigma}$ | mean $R^2={summary.mean_r2:.3f}$")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
        ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.80).set_label(r"$R^2_\sigma(r)$")
    fig.savefig(MULTIPANEL_MAP_FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()
    for path in [NETCDF_FILE, SUMMARY_CSV_FILE, SUMMARY_JSON_FILE, MEAN_R2_FIGURE_FILE, MULTIPANEL_MAP_FIGURE_FILE]:
        remove_if_exists(path)
    for sigma in SIGMA_VALUES:
        remove_if_exists(HOME_OUTPUT_DIR / MAP_FIGURE_TEMPLATE.format(sigma=sigma))

    pacific = load_pacific_cobe2_pca(PACIFIC_SST_REGION_360)
    monthly_mean_ds = xr.open_dataset(
        ERA5_MONTHLY_MEAN_FILE,
        engine="netcdf4",
        chunks={"time": TIME_CHUNK, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )
    monthly_clim_ds = xr.open_dataset(
        ERA5_MONTHLY_CLIM_FILE,
        engine="netcdf4",
        chunks={"month": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )

    try:
        monthly_mean = subset_era5_region_360(monthly_mean_ds[ERA5_VARIABLE], SIERRA_T2M_REGION_360)
        monthly_clim = subset_era5_region_360(monthly_clim_ds[ERA5_VARIABLE], SIERRA_T2M_REGION_360)
        era5_time = to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        overlap_months = np.intersect1d(pacific["time"], era5_time, assume_unique=False)
        if overlap_months.size == 0:
            raise ValueError("No overlapping months between Pacific COBE2 PC time and Sierra ERA5 time")

        pacific_index = build_time_index(pacific["time"])
        era5_index = build_time_index(era5_time)
        predictor_matrix_raw = np.stack(
            [pacific["pc"][pacific_index[month], :N_PREDICTORS] for month in overlap_months.tolist()],
            axis=0,
        )
        predictor_matrix, predictor_matrix_raw_mean, predictor_matrix_raw_std = standardize_pc_matrix(predictor_matrix_raw)

        sierra_latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float64)
        sierra_longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float64)
        anomaly_slices: List[np.ndarray] = []

        print(
            "Computing Route 3 Gaussian-smoothed Sierra T2m regression for overlap "
            f"{format_month(overlap_months[0])} to {format_month(overlap_months[-1])} "
            f"({int(overlap_months.size)} months)",
            flush=True,
        )
        for step_index, month_value in enumerate(overlap_months.tolist(), start=1):
            era5_time_index = era5_index[month_value]
            monthly_mean_slice = monthly_mean.isel(time=era5_time_index)
            monthly_clim_slice = monthly_clim.sel(month=month_number(month_value))
            anomaly_slice = (monthly_mean_slice - monthly_clim_slice).astype(np.float64).load().values
            anomaly_slices.append(anomaly_slice)
            if step_index == 1 or step_index % 120 == 0 or step_index == overlap_months.size:
                print(
                    f"  processed overlap month {step_index}/{int(overlap_months.size)}: {format_month(month_value)}",
                    flush=True,
                )

        anomalies = np.stack(anomaly_slices, axis=0)

        r2_map_list: List[np.ndarray] = []
        coefficient_map_list: List[np.ndarray] = []
        summaries: List[SummaryRow] = []
        r2_maps_by_sigma: Dict[int, np.ndarray] = {}
        summaries_by_sigma: Dict[int, SummaryRow] = {}

        for sigma in SIGMA_VALUES:
            print(f"  smoothing sigma {sigma}", flush=True)
            smoothed_anomalies = smooth_anomalies_gaussian(anomalies, sierra_latitude, float(sigma))
            coefficient_maps, r2_map = fit_multivariate_regression(predictor_matrix, smoothed_anomalies)
            summary = summarize_sigma_r2(sigma, r2_map, sierra_latitude, sierra_longitude)
            r2_map_list.append(r2_map)
            coefficient_map_list.append(coefficient_maps)
            summaries.append(summary)
            r2_maps_by_sigma[sigma] = r2_map
            summaries_by_sigma[sigma] = summary

        save_netcdf(
            overlap_months=overlap_months,
            predictor_matrix_raw_mean=predictor_matrix_raw_mean,
            predictor_matrix_raw_std=predictor_matrix_raw_std,
            latitude=sierra_latitude,
            longitude=sierra_longitude,
            r2_maps=np.stack(r2_map_list, axis=0),
            coefficient_maps=np.stack(coefficient_map_list, axis=0),
            summaries=summaries,
            runtime=runtime,
        )
        save_summary(
            summaries=summaries,
            pacific=pacific,
            sierra_shape=[int(sierra_latitude.size), int(sierra_longitude.size)],
            runtime=runtime,
        )
        for summary in summaries:
            plot_r2_map(
                latitude=sierra_latitude,
                longitude=sierra_longitude,
                r2_map=r2_maps_by_sigma[summary.sigma],
                summary=summary,
                output_path=HOME_OUTPUT_DIR / MAP_FIGURE_TEMPLATE.format(sigma=summary.sigma),
            )
        plot_mean_r2_vs_sigma(summaries)
        plot_multipanel_maps(sierra_latitude, sierra_longitude, r2_maps_by_sigma, summaries_by_sigma)
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Mean R2 figure: {MEAN_R2_FIGURE_FILE}", flush=True)
    print(f"Multi-panel map figure: {MULTIPANEL_MAP_FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
