#!/usr/bin/env python3
"""
Run Route 2: high-predictability regional aggregation for matched-region ERA5-Land T2m
using standardized COBE2 Pacific SST PCs 1-6.
"""

import csv
import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_cobe2_pacific_sierra_t2m_level1_diagnostic import (
    ERA5_MONTHLY_CLIM_FILE,
    ERA5_MONTHLY_MEAN_FILE,
    LAT_CHUNK,
    LON_CHUNK,
    PACIFIC_SST_REGION_360,
    SIERRA_T2M_REGION_360,
    TIME_CHUNK,
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


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability"
MODEL_NAME = "PC1_to_PC6"
N_PREDICTORS = 6
TOP_PERCENTILES = [10, 20, 30, 40, 50]

PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL2_PC1TO6_ROUTE2_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_summary.json"
R2_MAP_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_fine_r2_map.png"
MASK_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_mask_maps.png"
REGIONAL_R2_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_regional_r2_vs_percent.png"

ERA5_VARIABLE = "t2m"


@dataclass(frozen=True)
class SummaryRow:
    group_name: str
    top_percent: int
    r2_threshold_used: float
    number_of_grid_points: int
    area_weight_sum: float
    regional_r2: float
    mean_local_r2_inside_group: float
    median_local_r2_inside_group: float
    max_local_r2_inside_group: float
    min_local_r2_inside_group: float


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


def area_weights(latitude: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    return np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], shape)


def fit_multivariate_regression_r2(predictors: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    total_sum = float(np.sum(y ** 2))
    if not np.isfinite(total_sum) or total_sum <= 0.0:
        return float("nan")
    coefficients, _, _, _ = np.linalg.lstsq(predictors, y[:, np.newaxis], rcond=None)
    yhat = predictors @ coefficients
    residual_sum = float(np.sum((y[:, np.newaxis] - yhat) ** 2))
    return 1.0 - (residual_sum / total_sum)


def fit_fine_grid_r2_map(predictors: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    values = np.asarray(anomalies, dtype=np.float64)
    y_flat = values.reshape(values.shape[0], -1)
    valid_columns = np.isfinite(y_flat).all(axis=0)
    if not np.any(valid_columns):
        raise ValueError("No all-time-finite matched-region grid points available for Route 2")
    y_valid = y_flat[:, valid_columns]
    coefficients, _, _, _ = np.linalg.lstsq(predictors, y_valid, rcond=None)
    yhat = predictors @ coefficients
    residual_sum = np.sum((y_valid - yhat) ** 2, axis=0)
    total_sum = np.sum(y_valid ** 2, axis=0)
    r2_valid = np.full(total_sum.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(total_sum) & (total_sum > 0.0)
    r2_valid[positive] = 1.0 - (residual_sum[positive] / total_sum[positive])
    r2_map = np.full(values.shape[1] * values.shape[2], np.nan, dtype=np.float64)
    r2_map[valid_columns] = r2_valid
    return r2_map.reshape(values.shape[1], values.shape[2])


def weighted_group_series(anomalies: np.ndarray, weights_2d: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    valid_weights = np.where(mask, weights_2d, 0.0)
    weight_sum = float(np.sum(valid_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Selected group has zero area weight")
    weighted_values = np.where(mask[np.newaxis, :, :], anomalies, 0.0) * valid_weights[np.newaxis, :, :]
    series = np.sum(weighted_values, axis=(1, 2)) / weight_sum
    return series.astype(np.float64), weight_sum


def select_top_percent_mask(r2_map: np.ndarray, valid_mask: np.ndarray, top_percent: int) -> Tuple[np.ndarray, float]:
    values = np.asarray(r2_map, dtype=np.float64)
    valid_values = values[valid_mask]
    if valid_values.size == 0:
        raise ValueError("No valid R2 values available for percentile mask selection")
    threshold = float(np.nanpercentile(valid_values, 100 - top_percent))
    mask = valid_mask & np.isfinite(values) & (values >= threshold)
    return mask, threshold


def build_summary_row(
    group_name: str,
    top_percent: int,
    threshold: float,
    mask: np.ndarray,
    weight_sum: float,
    regional_r2: float,
    r2_map: np.ndarray,
) -> SummaryRow:
    local_values = np.asarray(r2_map, dtype=np.float64)[mask]
    return SummaryRow(
        group_name=group_name,
        top_percent=top_percent,
        r2_threshold_used=threshold,
        number_of_grid_points=int(np.count_nonzero(mask)),
        area_weight_sum=float(weight_sum),
        regional_r2=float(regional_r2),
        mean_local_r2_inside_group=float(np.nanmean(local_values)),
        median_local_r2_inside_group=float(np.nanmedian(local_values)),
        max_local_r2_inside_group=float(np.nanmax(local_values)),
        min_local_r2_inside_group=float(np.nanmin(local_values)),
    )


def save_netcdf(
    latitude: np.ndarray,
    longitude: np.ndarray,
    fine_r2_map: np.ndarray,
    masks: Dict[int, np.ndarray],
    summaries: List[SummaryRow],
    overlap_months: np.ndarray,
    predictor_matrix_raw_mean: np.ndarray,
    predictor_matrix_raw_std: np.ndarray,
    runtime,
) -> None:
    percent_dim = np.asarray([row.top_percent for row in summaries], dtype=np.int32)
    mask_stack = np.stack([masks[row.top_percent] for row in summaries], axis=0)
    ds = xr.Dataset(
        data_vars={
            "fine_local_r2": (("latitude", "longitude"), fine_r2_map.astype(np.float32)),
            "group_mask": (("top_percent", "latitude", "longitude"), mask_stack.astype(np.int8)),
            "regional_r2": (("top_percent",), np.asarray([row.regional_r2 for row in summaries], dtype=np.float32)),
            "r2_threshold_used": (("top_percent",), np.asarray([row.r2_threshold_used for row in summaries], dtype=np.float32)),
            "number_of_grid_points": (("top_percent",), np.asarray([row.number_of_grid_points for row in summaries], dtype=np.int32)),
            "area_weight_sum": (("top_percent",), np.asarray([row.area_weight_sum for row in summaries], dtype=np.float32)),
            "mean_local_r2_inside_group": (("top_percent",), np.asarray([row.mean_local_r2_inside_group for row in summaries], dtype=np.float32)),
            "median_local_r2_inside_group": (("top_percent",), np.asarray([row.median_local_r2_inside_group for row in summaries], dtype=np.float32)),
            "max_local_r2_inside_group": (("top_percent",), np.asarray([row.max_local_r2_inside_group for row in summaries], dtype=np.float32)),
            "min_local_r2_inside_group": (("top_percent",), np.asarray([row.min_local_r2_inside_group for row in summaries], dtype=np.float32)),
            "pacific_cobe2_pc_mean_raw": (("mode",), predictor_matrix_raw_mean.astype(np.float32)),
            "pacific_cobe2_pc_std_raw": (("mode",), predictor_matrix_raw_std.astype(np.float32)),
        },
        coords={
            "latitude": latitude.astype(np.float32),
            "longitude": longitude.astype(np.float32),
            "top_percent": percent_dim,
            "mode": np.arange(1, N_PREDICTORS + 1, dtype=np.int32),
            "time": overlap_months.astype("datetime64[ns]"),
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "top_percentiles": json.dumps(TOP_PERCENTILES),
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "selection_note": "Exploratory in-sample aggregation using groups selected from the same fine-grid R2 map",
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(NETCDF_FILE, engine="netcdf4")


def save_summary(summaries: List[SummaryRow], fine_grid_mean_r2: float, whole_sierra_r2: float, runtime) -> None:
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_name",
                "top_percent",
                "R2_threshold_used",
                "number_of_grid_points",
                "area_weight_sum",
                "regional_R2",
                "mean_local_R2_inside_group",
                "median_local_R2_inside_group",
                "max_local_R2_inside_group",
                "min_local_R2_inside_group",
            ]
        )
        writer.writerow(["fine_grid_mean_baseline", 100, "", "", "", "{:.12g}".format(fine_grid_mean_r2), "", "", "", ""])
        writer.writerow(["whole_sierra_baseline", 100, "", "", "", "{:.12g}".format(whole_sierra_r2), "", "", "", ""])
        for row in summaries:
            writer.writerow(
                [
                    row.group_name,
                    row.top_percent,
                    "{:.12g}".format(row.r2_threshold_used),
                    row.number_of_grid_points,
                    "{:.12g}".format(row.area_weight_sum),
                    "{:.12g}".format(row.regional_r2),
                    "{:.12g}".format(row.mean_local_r2_inside_group),
                    "{:.12g}".format(row.median_local_r2_inside_group),
                    "{:.12g}".format(row.max_local_r2_inside_group),
                    "{:.12g}".format(row.min_local_r2_inside_group),
                ]
            )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "top_percentiles": TOP_PERCENTILES,
        "pc_standardized": True,
        "selection_note": "Exploratory in-sample aggregation using groups selected from the same fine-grid R2 map",
        "fine_grid_area_weighted_mean_r2": fine_grid_mean_r2,
        "whole_sierra_regional_r2": whole_sierra_r2,
        "summary_rows": [asdict(row) for row in summaries],
        "slurm_job_id": runtime.slurm_job_id,
        "compute_node": runtime.hostname,
        "output_directory_size": output_dir_size_text(),
    }
    SUMMARY_JSON_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_fine_r2_map(latitude: np.ndarray, longitude: np.ndarray, r2_map: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.5), constrained_layout=True)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    mesh = ax.pcolormesh(lon2d, lat2d, r2_map, cmap="viridis", shading="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(r2_map))))
    ax.set_title(r"Fine-grid matched-region $R^2(r)$ from standardized PC1-PC6 regression")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
    ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, shrink=0.88).set_label(r"$R^2(r)$")
    fig.savefig(R2_MAP_FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_mask_maps(latitude: np.ndarray, longitude: np.ndarray, masks: Dict[int, np.ndarray], summaries: List[SummaryRow]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.30, wspace=0.22)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    for ax, row in zip(axes.flat, summaries):
        mask = masks[row.top_percent]
        mesh = ax.pcolormesh(lon2d, lat2d, mask.astype(float), cmap="Greens", shading="auto", vmin=0.0, vmax=1.0)
        ax.set_title(rf"Top {row.top_percent}% by local $R^2$ | regional $R^2={row.regional_r2:.3f}$")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
        ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.80).set_label("selected")
    fig.savefig(MASK_FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_regional_r2_vs_percent(summaries: List[SummaryRow], fine_grid_mean_r2: float, whole_sierra_r2: float) -> None:
    percents = np.asarray([row.top_percent for row in summaries], dtype=np.float64)
    regional_r2 = np.asarray([row.regional_r2 for row in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.plot(percents, regional_r2, marker="o", linewidth=2, label="top-predictability group")
    ax.axhline(fine_grid_mean_r2, color="gray", linestyle="--", label="fine-grid mean R2")
    ax.axhline(whole_sierra_r2, color="tab:orange", linestyle=":", label="whole-region regional R2")
    ax.set_xlabel("Top percent selected")
    ax.set_ylabel("Regional $R^2$")
    ax.set_title(r"Route 2: regional $R^2$ vs. high-predictability matched-region selection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(REGIONAL_R2_FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()
    for path in [NETCDF_FILE, SUMMARY_CSV_FILE, SUMMARY_JSON_FILE, R2_MAP_FIGURE_FILE, MASK_FIGURE_FILE, REGIONAL_R2_FIGURE_FILE]:
        remove_if_exists(path)

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
            raise ValueError("No overlapping months between Pacific COBE2 PC time and matched-region ERA5 time")

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
            "Computing Route 2 high-predictability aggregation for overlap "
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
        fine_r2_map = fit_fine_grid_r2_map(predictor_matrix, anomalies)
        valid_mask = np.isfinite(fine_r2_map)
        weights_2d = area_weights(sierra_latitude, fine_r2_map.shape)

        whole_sierra_series, _ = weighted_group_series(anomalies, np.where(valid_mask, weights_2d, 0.0), valid_mask)
        whole_sierra_r2 = fit_multivariate_regression_r2(predictor_matrix, whole_sierra_series)
        fine_grid_mean_r2 = float(np.sum(fine_r2_map[valid_mask] * weights_2d[valid_mask]) / np.sum(weights_2d[valid_mask]))

        masks: Dict[int, np.ndarray] = {}
        summaries: List[SummaryRow] = []
        for top_percent in TOP_PERCENTILES:
            mask, threshold = select_top_percent_mask(fine_r2_map, valid_mask, top_percent)
            masks[top_percent] = mask
            group_series, weight_sum = weighted_group_series(anomalies, weights_2d, mask)
            regional_r2 = fit_multivariate_regression_r2(predictor_matrix, group_series)
            summaries.append(
                build_summary_row(
                    group_name=f"top_{top_percent}_percent",
                    top_percent=top_percent,
                    threshold=threshold,
                    mask=mask,
                    weight_sum=weight_sum,
                    regional_r2=regional_r2,
                    r2_map=fine_r2_map,
                )
            )
            print(f"  selected top {top_percent}% mask | regional R2={regional_r2:.4f}", flush=True)

        save_netcdf(
            latitude=sierra_latitude,
            longitude=sierra_longitude,
            fine_r2_map=fine_r2_map,
            masks=masks,
            summaries=summaries,
            overlap_months=overlap_months,
            predictor_matrix_raw_mean=predictor_matrix_raw_mean,
            predictor_matrix_raw_std=predictor_matrix_raw_std,
            runtime=runtime,
        )
        save_summary(summaries=summaries, fine_grid_mean_r2=fine_grid_mean_r2, whole_sierra_r2=whole_sierra_r2, runtime=runtime)
        plot_fine_r2_map(sierra_latitude, sierra_longitude, fine_r2_map)
        plot_mask_maps(sierra_latitude, sierra_longitude, masks, summaries)
        plot_regional_r2_vs_percent(summaries, fine_grid_mean_r2, whole_sierra_r2)
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Fine R2 figure: {R2_MAP_FIGURE_FILE}", flush=True)
    print(f"Mask figure: {MASK_FIGURE_FILE}", flush=True)
    print(f"Regional R2 figure: {REGIONAL_R2_FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
