#!/usr/bin/env python3
"""
Run Route 2 Sierra-only high-predictability regional aggregation using the
expanded-domain PC1-PC6 local R2 field as the selection source.
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
from snow_ml.data import RegionBounds


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
MODEL_NAME = "PC1_to_PC6"
N_PREDICTORS = 6
TOP_PERCENTILES = [10, 20, 30, 40, 50]

EXPANDED_LEVEL2_NETCDF_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6/cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
)
PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL2_PC1TO6_ROUTE2_SIERRA_ONLY_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only_summary.json"
R2_MAP_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_sierra_only_fine_r2_map.png"
MASK_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_sierra_only_mask_maps.png"
REGIONAL_R2_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_sierra_only_regional_r2_vs_percent.png"

ERA5_VARIABLE = "t2m"
SIERRA_SELECTION_REGION_360 = RegionBounds(lat_min=35.0, lat_max=42.0, lon_min=236.0, lon_max=243.0)


@dataclass(frozen=True)
class SummaryRow:
    group_name: str
    top_percent: int
    r2_threshold_used_within_sierra: float
    number_of_sierra_grid_points_selected: int
    total_number_of_valid_sierra_grid_points: int
    area_weight_sum: float
    regional_r2: float
    mean_local_r2_inside_group: float
    median_local_r2_inside_group: float
    max_local_r2_inside_group: float
    min_local_r2_inside_group: float
    sierra_fine_grid_mean_r2: float
    whole_sierra_regional_r2: float


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


def weighted_group_series(anomalies: np.ndarray, weights_2d: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    valid_weights = np.where(mask, weights_2d, 0.0)
    weight_sum = float(np.sum(valid_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Selected group has zero area weight")
    weighted_values = np.where(mask[np.newaxis, :, :], anomalies, 0.0) * valid_weights[np.newaxis, :, :]
    series = np.sum(weighted_values, axis=(1, 2)) / weight_sum
    return series.astype(np.float64), weight_sum


def select_top_percent_mask(r2_map: np.ndarray, sierra_valid_mask: np.ndarray, top_percent: int) -> Tuple[np.ndarray, float]:
    sierra_values = np.asarray(r2_map, dtype=np.float64)[sierra_valid_mask]
    if sierra_values.size == 0:
        raise ValueError("No valid Sierra R2 values available for percentile mask selection")
    threshold = float(np.nanpercentile(sierra_values, 100 - top_percent))
    mask = sierra_valid_mask & np.isfinite(r2_map) & (r2_map >= threshold)
    return mask, threshold


def build_sierra_box_mask(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(np.asarray(latitude, dtype=np.float64), np.asarray(longitude, dtype=np.float64), indexing="ij")
    return (
        (lat2d >= SIERRA_SELECTION_REGION_360.lat_min)
        & (lat2d <= SIERRA_SELECTION_REGION_360.lat_max)
        & (lon2d >= SIERRA_SELECTION_REGION_360.lon_min)
        & (lon2d <= SIERRA_SELECTION_REGION_360.lon_max)
    )


def load_expanded_domain_r2_source() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(EXPANDED_LEVEL2_NETCDF_FILE, engine="netcdf4", decode_times=True) as ds:
        latitude = np.asarray(ds["sierra_latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["sierra_longitude"].values, dtype=np.float64)
        overlap_months = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        fine_r2_map = np.asarray(ds["sierra_era5_t2m_multi_pc_r2"].values, dtype=np.float64)
    return latitude, longitude, overlap_months, fine_r2_map


def load_aligned_predictor_matrix(overlap_months: np.ndarray) -> np.ndarray:
    pacific = load_pacific_cobe2_pca(PACIFIC_SST_REGION_360)
    pacific_index = build_time_index(pacific["time"])
    predictor_matrix_raw = np.stack(
        [pacific["pc"][pacific_index[month], :N_PREDICTORS] for month in overlap_months.tolist()],
        axis=0,
    )
    predictor_matrix, _, _ = standardize_pc_matrix(predictor_matrix_raw)
    return predictor_matrix


def load_aligned_anomalies(overlap_months: np.ndarray) -> np.ndarray:
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
        monthly_mean = subset_era5_region_360(monthly_mean_ds[ERA5_VARIABLE], PACIFIC_SST_REGION_360)
        monthly_clim = subset_era5_region_360(monthly_clim_ds[ERA5_VARIABLE], PACIFIC_SST_REGION_360)
        era5_time = to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        era5_index = build_time_index(era5_time)

        anomaly_slices: List[np.ndarray] = []
        print(
            "Computing Sierra-only Route 2 aggregation from expanded-domain anomalies for overlap "
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
        return np.stack(anomaly_slices, axis=0)
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()


def build_summary_row(
    group_name: str,
    top_percent: int,
    threshold: float,
    mask: np.ndarray,
    weight_sum: float,
    regional_r2: float,
    r2_map: np.ndarray,
    total_valid_sierra_points: int,
    sierra_fine_grid_mean_r2: float,
    whole_sierra_r2: float,
) -> SummaryRow:
    local_values = np.asarray(r2_map, dtype=np.float64)[mask]
    return SummaryRow(
        group_name=group_name,
        top_percent=top_percent,
        r2_threshold_used_within_sierra=threshold,
        number_of_sierra_grid_points_selected=int(np.count_nonzero(mask)),
        total_number_of_valid_sierra_grid_points=total_valid_sierra_points,
        area_weight_sum=float(weight_sum),
        regional_r2=float(regional_r2),
        mean_local_r2_inside_group=float(np.nanmean(local_values)),
        median_local_r2_inside_group=float(np.nanmedian(local_values)),
        max_local_r2_inside_group=float(np.nanmax(local_values)),
        min_local_r2_inside_group=float(np.nanmin(local_values)),
        sierra_fine_grid_mean_r2=float(sierra_fine_grid_mean_r2),
        whole_sierra_regional_r2=float(whole_sierra_r2),
    )


def save_netcdf(
    latitude: np.ndarray,
    longitude: np.ndarray,
    fine_r2_map: np.ndarray,
    sierra_valid_mask: np.ndarray,
    masks: Dict[int, np.ndarray],
    summaries: List[SummaryRow],
    overlap_months: np.ndarray,
    runtime,
) -> None:
    percent_dim = np.asarray([row.top_percent for row in summaries], dtype=np.int32)
    mask_stack = np.stack([masks[row.top_percent] for row in summaries], axis=0)
    ds = xr.Dataset(
        data_vars={
            "fine_local_r2": (("latitude", "longitude"), fine_r2_map.astype(np.float32)),
            "valid_sierra_mask": (("latitude", "longitude"), sierra_valid_mask.astype(np.int8)),
            "group_mask": (("top_percent", "latitude", "longitude"), mask_stack.astype(np.int8)),
            "regional_r2": (("top_percent",), np.asarray([row.regional_r2 for row in summaries], dtype=np.float32)),
            "r2_threshold_used_within_sierra": (
                ("top_percent",),
                np.asarray([row.r2_threshold_used_within_sierra for row in summaries], dtype=np.float32),
            ),
            "number_of_sierra_grid_points_selected": (
                ("top_percent",),
                np.asarray([row.number_of_sierra_grid_points_selected for row in summaries], dtype=np.int32),
            ),
            "total_number_of_valid_sierra_grid_points": (
                ("top_percent",),
                np.asarray([row.total_number_of_valid_sierra_grid_points for row in summaries], dtype=np.int32),
            ),
            "area_weight_sum": (("top_percent",), np.asarray([row.area_weight_sum for row in summaries], dtype=np.float32)),
            "mean_local_r2_inside_group": (
                ("top_percent",),
                np.asarray([row.mean_local_r2_inside_group for row in summaries], dtype=np.float32),
            ),
            "median_local_r2_inside_group": (
                ("top_percent",),
                np.asarray([row.median_local_r2_inside_group for row in summaries], dtype=np.float32),
            ),
            "max_local_r2_inside_group": (
                ("top_percent",),
                np.asarray([row.max_local_r2_inside_group for row in summaries], dtype=np.float32),
            ),
            "min_local_r2_inside_group": (
                ("top_percent",),
                np.asarray([row.min_local_r2_inside_group for row in summaries], dtype=np.float32),
            ),
        },
        coords={
            "latitude": latitude.astype(np.float32),
            "longitude": longitude.astype(np.float32),
            "top_percent": percent_dim,
            "time": overlap_months.astype("datetime64[ns]"),
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "selection_source": str(EXPANDED_LEVEL2_NETCDF_FILE),
            "selection_note": (
                "Local R2 source is the expanded-domain PC1-PC6 regression result; "
                "top-percent ranking and regional aggregation are restricted to valid Sierra grid points only."
            ),
            "sierra_selection_region_360": json.dumps(SIERRA_SELECTION_REGION_360.as_dict()),
            "top_percentiles": json.dumps(TOP_PERCENTILES),
            "pc_standardized": "true",
            "pc_standardization": "recomputed over the same overlap months as the expanded-domain regression",
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(NETCDF_FILE, engine="netcdf4")


def save_summary(summaries: List[SummaryRow], sierra_fine_grid_mean_r2: float, whole_sierra_r2: float, runtime) -> None:
    fieldnames = [
        "group_name",
        "top_percent",
        "R2_threshold_used_within_Sierra",
        "number_of_Sierra_grid_points_selected",
        "total_number_of_valid_Sierra_grid_points",
        "area_weight_sum",
        "regional_R2",
        "mean_local_R2_inside_group",
        "median_local_R2_inside_group",
        "max_local_R2_inside_group",
        "min_local_R2_inside_group",
        "Sierra_fine_grid_mean_R2",
        "whole_Sierra_regional_R2",
    ]
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerow(
            [
                "sierra_fine_grid_mean_baseline",
                "",
                "",
                "",
                summaries[0].total_number_of_valid_sierra_grid_points if summaries else "",
                "",
                "{:.12g}".format(sierra_fine_grid_mean_r2),
                "",
                "",
                "",
                "",
                "{:.12g}".format(sierra_fine_grid_mean_r2),
                "{:.12g}".format(whole_sierra_r2),
            ]
        )
        writer.writerow(
            [
                "whole_sierra_regional_baseline",
                "",
                "",
                summaries[0].total_number_of_valid_sierra_grid_points if summaries else "",
                summaries[0].total_number_of_valid_sierra_grid_points if summaries else "",
                "",
                "{:.12g}".format(whole_sierra_r2),
                "",
                "",
                "",
                "",
                "{:.12g}".format(sierra_fine_grid_mean_r2),
                "{:.12g}".format(whole_sierra_r2),
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.group_name,
                    row.top_percent,
                    "{:.12g}".format(row.r2_threshold_used_within_sierra),
                    row.number_of_sierra_grid_points_selected,
                    row.total_number_of_valid_sierra_grid_points,
                    "{:.12g}".format(row.area_weight_sum),
                    "{:.12g}".format(row.regional_r2),
                    "{:.12g}".format(row.mean_local_r2_inside_group),
                    "{:.12g}".format(row.median_local_r2_inside_group),
                    "{:.12g}".format(row.max_local_r2_inside_group),
                    "{:.12g}".format(row.min_local_r2_inside_group),
                    "{:.12g}".format(row.sierra_fine_grid_mean_r2),
                    "{:.12g}".format(row.whole_sierra_regional_r2),
                ]
            )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "selection_source": str(EXPANDED_LEVEL2_NETCDF_FILE),
        "sierra_selection_region_360": SIERRA_SELECTION_REGION_360.as_dict(),
        "top_percentiles": TOP_PERCENTILES,
        "pc_standardized": True,
        "selection_note": (
            "Expanded-domain local R2 is used as the source field, but candidate ranking and aggregation "
            "are restricted to Sierra-box valid ERA5-Land grid points."
        ),
        "sierra_fine_grid_mean_r2": sierra_fine_grid_mean_r2,
        "whole_sierra_regional_r2": whole_sierra_r2,
        "summary_rows": [asdict(row) for row in summaries],
        "slurm_job_id": runtime.slurm_job_id,
        "compute_node": runtime.hostname,
        "output_directory_size": output_dir_size_text(),
    }
    SUMMARY_JSON_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_fine_r2_map(latitude: np.ndarray, longitude: np.ndarray, r2_map: np.ndarray, sierra_valid_mask: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.5), constrained_layout=True)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    masked_r2 = np.where(sierra_valid_mask, r2_map, np.nan)
    vmax = float(np.nanmax(masked_r2))
    mesh = ax.pcolormesh(lon2d, lat2d, masked_r2, cmap="viridis", shading="auto", vmin=0.0, vmax=max(0.05, vmax))
    ax.set_title(r"Sierra-only local $R^2(r)$ from expanded-domain standardized PC1-PC6 regression")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(SIERRA_SELECTION_REGION_360.lon_min, SIERRA_SELECTION_REGION_360.lon_max)
    ax.set_ylim(SIERRA_SELECTION_REGION_360.lat_min, SIERRA_SELECTION_REGION_360.lat_max)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, shrink=0.88).set_label(r"$R^2(r)$")
    fig.savefig(R2_MAP_FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_mask_maps(latitude: np.ndarray, longitude: np.ndarray, masks: Dict[int, np.ndarray], summaries: List[SummaryRow]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.30, wspace=0.22)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    axes_flat = axes.flatten()
    for ax, row in zip(axes_flat, summaries):
        mask = masks[row.top_percent]
        mesh = ax.pcolormesh(lon2d, lat2d, mask.astype(float), cmap="Greens", shading="auto", vmin=0.0, vmax=1.0)
        ax.set_title(rf"Top {row.top_percent}% within Sierra | regional $R^2={row.regional_r2:.3f}$")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(SIERRA_SELECTION_REGION_360.lon_min, SIERRA_SELECTION_REGION_360.lon_max)
        ax.set_ylim(SIERRA_SELECTION_REGION_360.lat_min, SIERRA_SELECTION_REGION_360.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.80).set_label("selected")
    unused_axes = axes_flat[len(summaries):]
    for ax in unused_axes:
        ax.axis("off")
    fig.savefig(MASK_FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_regional_r2_vs_percent(summaries: List[SummaryRow], sierra_fine_grid_mean_r2: float, whole_sierra_r2: float) -> None:
    percents = np.asarray([row.top_percent for row in summaries], dtype=np.float64)
    regional_r2 = np.asarray([row.regional_r2 for row in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.plot(percents, regional_r2, marker="o", linewidth=2, label="Sierra-only high-predictability group")
    ax.axhline(sierra_fine_grid_mean_r2, color="gray", linestyle="--", label="Sierra fine-grid mean R2")
    ax.axhline(whole_sierra_r2, color="tab:orange", linestyle=":", label="whole-Sierra regional R2")
    ax.set_xlabel("Top percent selected within Sierra")
    ax.set_ylabel("Regional $R^2$")
    ax.set_title(r"Sierra-only Route 2: regional $R^2$ vs. top local-$R^2$ selection")
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

    latitude, longitude, overlap_months, fine_r2_map = load_expanded_domain_r2_source()
    predictor_matrix = load_aligned_predictor_matrix(overlap_months)
    anomalies = load_aligned_anomalies(overlap_months)

    if anomalies.shape[1:] != fine_r2_map.shape:
        raise ValueError(
            f"Expanded-domain anomaly grid {anomalies.shape[1:]} does not match source R2 grid {fine_r2_map.shape}"
        )

    weights_2d = area_weights(latitude, fine_r2_map.shape)
    valid_mask = np.isfinite(fine_r2_map) & np.isfinite(anomalies).all(axis=0)
    sierra_box_mask = build_sierra_box_mask(latitude, longitude)
    sierra_valid_mask = valid_mask & sierra_box_mask
    if not np.any(sierra_valid_mask):
        raise ValueError("No valid Sierra ERA5-Land grid points found inside the requested Sierra box")

    total_valid_sierra_points = int(np.count_nonzero(sierra_valid_mask))
    whole_sierra_series, _ = weighted_group_series(anomalies, weights_2d, sierra_valid_mask)
    whole_sierra_r2 = fit_multivariate_regression_r2(predictor_matrix, whole_sierra_series)
    sierra_fine_grid_mean_r2 = float(
        np.sum(fine_r2_map[sierra_valid_mask] * weights_2d[sierra_valid_mask]) / np.sum(weights_2d[sierra_valid_mask])
    )

    masks: Dict[int, np.ndarray] = {}
    summaries: List[SummaryRow] = []
    for top_percent in TOP_PERCENTILES:
        mask, threshold = select_top_percent_mask(fine_r2_map, sierra_valid_mask, top_percent)
        masks[top_percent] = mask
        group_series, weight_sum = weighted_group_series(anomalies, weights_2d, mask)
        regional_r2 = fit_multivariate_regression_r2(predictor_matrix, group_series)
        summaries.append(
            build_summary_row(
                group_name=f"sierra_top_{top_percent}_percent",
                top_percent=top_percent,
                threshold=threshold,
                mask=mask,
                weight_sum=weight_sum,
                regional_r2=regional_r2,
                r2_map=fine_r2_map,
                total_valid_sierra_points=total_valid_sierra_points,
                sierra_fine_grid_mean_r2=sierra_fine_grid_mean_r2,
                whole_sierra_r2=whole_sierra_r2,
            )
        )
        print(
            f"  selected Sierra-only top {top_percent}% mask "
            f"({int(np.count_nonzero(mask))} points) | regional R2={regional_r2:.4f}",
            flush=True,
        )

    save_netcdf(
        latitude=latitude,
        longitude=longitude,
        fine_r2_map=fine_r2_map,
        sierra_valid_mask=sierra_valid_mask,
        masks=masks,
        summaries=summaries,
        overlap_months=overlap_months,
        runtime=runtime,
    )
    save_summary(
        summaries=summaries,
        sierra_fine_grid_mean_r2=sierra_fine_grid_mean_r2,
        whole_sierra_r2=whole_sierra_r2,
        runtime=runtime,
    )
    plot_fine_r2_map(latitude, longitude, fine_r2_map, sierra_valid_mask)
    plot_mask_maps(latitude, longitude, masks, summaries)
    plot_regional_r2_vs_percent(summaries, sierra_fine_grid_mean_r2, whole_sierra_r2)

    print(f"Expanded-domain R2 source: {EXPANDED_LEVEL2_NETCDF_FILE}", flush=True)
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
