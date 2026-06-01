#!/usr/bin/env python3
"""
Run a Sierra ERA5-Land T2m spatial coarsening experiment using standardized
COBE2 Pacific SST PCs 1-6 as joint predictors.
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


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening"
MODEL_NAME = "PC1_to_PC6"
N_PREDICTORS = 6
BLOCK_FACTORS = [1, 2, 4, 8, 16, 32]
MIN_VALID_FRACTION = 0.25

PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL2_PC1TO6_COARSENING_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening_summary.json"
SCALE_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening_scale_sensitivity.png"
COMBINED_MAP_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening_r2_maps.png"
MAP_FIGURE_TEMPLATE = "cobe2_pacific_sierra_t2m_level2_pc1to6_coarsening_r2_blockfactor{factor}.png"

ERA5_VARIABLE = "t2m"


@dataclass(frozen=True)
class SummaryRow:
    block_factor: int
    approximate_block_size_km: float
    number_of_valid_blocks: int
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


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.size < 2:
        step = 0.5
        return np.array([centers[0] - step, centers[0] + step], dtype=np.float64)
    diffs = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * diffs[0]
    edges[-1] = centers[-1] + 0.5 * diffs[-1]
    return edges


def fit_block_regression(predictors: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return float("nan")
    total_sum = float(np.sum(y ** 2))
    if not np.isfinite(total_sum) or total_sum <= 0.0:
        return float("nan")
    coefficients, _, _, _ = np.linalg.lstsq(predictors, y[:, np.newaxis], rcond=None)
    yhat = predictors @ coefficients
    residual_sum = float(np.sum((y[:, np.newaxis] - yhat) ** 2))
    return 1.0 - (residual_sum / total_sum)


def approximate_block_size_km(block_factor: int, latitude: np.ndarray, longitude: np.ndarray) -> float:
    lat_spacing_deg = float(np.mean(np.abs(np.diff(latitude))))
    lon_spacing_deg = float(np.mean(np.abs(np.diff(longitude))))
    mean_lat = float(np.mean(latitude))
    lat_km = 111.32 * lat_spacing_deg * block_factor
    lon_km = 111.32 * math.cos(math.radians(mean_lat)) * lon_spacing_deg * block_factor
    return float(math.sqrt(max(lat_km, 0.0) * max(lon_km, 0.0)))


def coarsen_and_regress(
    anomalies: np.ndarray,
    predictors: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    block_factor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SummaryRow]:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    values = np.asarray(anomalies, dtype=np.float64)
    valid_mask = np.isfinite(values).all(axis=0)
    weights_2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], valid_mask.shape)
    weights_2d = np.where(valid_mask, weights_2d, 0.0)

    lat_edges_full = coordinate_edges(lat)
    lon_edges_full = coordinate_edges(lon)
    lat_starts = list(range(0, lat.size, block_factor))
    lon_starts = list(range(0, lon.size, block_factor))

    r2_map = np.full((len(lat_starts), len(lon_starts)), np.nan, dtype=np.float64)
    block_weight_map = np.full_like(r2_map, np.nan)
    min_valid_cells = max(1, int(math.ceil(MIN_VALID_FRACTION * block_factor * block_factor)))

    for lat_block_index, lat_start in enumerate(lat_starts):
        lat_end = min(lat_start + block_factor, lat.size)
        for lon_block_index, lon_start in enumerate(lon_starts):
            lon_end = min(lon_start + block_factor, lon.size)
            block_valid = valid_mask[lat_start:lat_end, lon_start:lon_end]
            valid_count = int(np.count_nonzero(block_valid))
            if valid_count < min_valid_cells:
                continue
            block_weights = weights_2d[lat_start:lat_end, lon_start:lon_end]
            total_weight = float(np.sum(block_weights))
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                continue
            block_series = (
                np.sum(values[:, lat_start:lat_end, lon_start:lon_end] * block_weights[np.newaxis, :, :], axis=(1, 2))
                / total_weight
            )
            r2_value = fit_block_regression(predictors, block_series)
            if not np.isfinite(r2_value):
                continue
            r2_map[lat_block_index, lon_block_index] = r2_value
            block_weight_map[lat_block_index, lon_block_index] = total_weight

    lat_edges = np.array([lat_edges_full[start] for start in lat_starts] + [lat_edges_full[min(lat_starts[-1] + block_factor, lat.size)]], dtype=np.float64)
    lon_edges = np.array([lon_edges_full[start] for start in lon_starts] + [lon_edges_full[min(lon_starts[-1] + block_factor, lon.size)]], dtype=np.float64)

    valid_blocks = np.isfinite(r2_map)
    weighted_sum = np.nansum(np.where(valid_blocks, r2_map * block_weight_map, 0.0))
    weight_sum = np.nansum(np.where(valid_blocks, block_weight_map, 0.0))
    mean_r2 = float(weighted_sum / weight_sum) if weight_sum > 0.0 else float("nan")
    summary = SummaryRow(
        block_factor=block_factor,
        approximate_block_size_km=approximate_block_size_km(block_factor, lat, lon),
        number_of_valid_blocks=int(np.count_nonzero(valid_blocks)),
        mean_r2=mean_r2,
        median_r2=float(np.nanmedian(r2_map[valid_blocks])) if np.any(valid_blocks) else float("nan"),
        max_r2=float(np.nanmax(r2_map)) if np.any(valid_blocks) else float("nan"),
        min_r2=float(np.nanmin(r2_map)) if np.any(valid_blocks) else float("nan"),
    )
    return r2_map, block_weight_map, lat_edges, lon_edges, summary


def save_netcdf(
    results: Dict[int, Dict[str, np.ndarray]],
    summaries: List[SummaryRow],
    overlap_months: np.ndarray,
    predictor_matrix_raw_mean: np.ndarray,
    predictor_matrix_raw_std: np.ndarray,
    runtime,
) -> None:
    data_vars: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {
        "pacific_cobe2_pc_mean_raw": (("mode",), predictor_matrix_raw_mean.astype(np.float32)),
        "pacific_cobe2_pc_std_raw": (("mode",), predictor_matrix_raw_std.astype(np.float32)),
        "coarsening_block_factor": (("coarsening_case",), np.asarray([row.block_factor for row in summaries], dtype=np.int32)),
        "coarsening_mean_r2": (("coarsening_case",), np.asarray([row.mean_r2 for row in summaries], dtype=np.float32)),
        "coarsening_median_r2": (("coarsening_case",), np.asarray([row.median_r2 for row in summaries], dtype=np.float32)),
        "coarsening_max_r2": (("coarsening_case",), np.asarray([row.max_r2 for row in summaries], dtype=np.float32)),
        "coarsening_min_r2": (("coarsening_case",), np.asarray([row.min_r2 for row in summaries], dtype=np.float32)),
        "coarsening_valid_block_count": (("coarsening_case",), np.asarray([row.number_of_valid_blocks for row in summaries], dtype=np.int32)),
        "coarsening_block_size_km": (("coarsening_case",), np.asarray([row.approximate_block_size_km for row in summaries], dtype=np.float32)),
    }
    coords: Dict[str, np.ndarray] = {
        "mode": np.arange(1, N_PREDICTORS + 1, dtype=np.int32),
        "coarsening_case": np.arange(len(summaries), dtype=np.int32),
        "time": overlap_months.astype("datetime64[ns]"),
    }
    for factor, result in results.items():
        lat_dim = f"lat_block_factor_{factor}"
        lon_dim = f"lon_block_factor_{factor}"
        lat_edge_dim = f"lat_edge_block_factor_{factor}"
        lon_edge_dim = f"lon_edge_block_factor_{factor}"
        data_vars[f"r2_block_factor_{factor}"] = ((lat_dim, lon_dim), result["r2_map"].astype(np.float32))
        data_vars[f"block_weight_factor_{factor}"] = ((lat_dim, lon_dim), result["block_weight_map"].astype(np.float32))
        coords[lat_edge_dim] = result["lat_edges"].astype(np.float32)
        coords[lon_edge_dim] = result["lon_edges"].astype(np.float32)
        coords[lat_dim] = (result["lat_edges"][:-1] + result["lat_edges"][1:]).astype(np.float32) * 0.5
        coords[lon_dim] = (result["lon_edges"][:-1] + result["lon_edges"][1:]).astype(np.float32) * 0.5

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "block_factors": json.dumps(BLOCK_FACTORS),
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "coarsening_method": "area-weighted block mean using cosine latitude weights over valid Sierra land cells",
            "minimum_valid_fraction_per_block": MIN_VALID_FRACTION,
            "formula_bhat": "b_hat_G = (A^T A)^(-1) A^T Y_G",
            "formula_r2": "R2_G = 1 - sum_t[(Y_G(t)-Y_hat_G(t))^2] / sum_t[Y_G(t)^2]",
            "time_overlap_start": format_month(overlap_months[0]),
            "time_overlap_end": format_month(overlap_months[-1]),
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(NETCDF_FILE, engine="netcdf4")


def save_summary(summaries: List[SummaryRow], runtime) -> None:
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "block_factor",
                "approximate_block_size_km",
                "number_of_valid_blocks",
                "mean_R2",
                "median_R2",
                "max_R2",
                "min_R2",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.block_factor,
                    "{:.12g}".format(row.approximate_block_size_km),
                    row.number_of_valid_blocks,
                    "{:.12g}".format(row.mean_r2),
                    "{:.12g}".format(row.median_r2),
                    "{:.12g}".format(row.max_r2),
                    "{:.12g}".format(row.min_r2),
                ]
            )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "block_factors": BLOCK_FACTORS,
        "pc_standardized": True,
        "minimum_valid_fraction_per_block": MIN_VALID_FRACTION,
        "summary_rows": [asdict(row) for row in summaries],
        "slurm_job_id": runtime.slurm_job_id,
        "compute_node": runtime.hostname,
        "output_directory_size": output_dir_size_text(),
    }
    SUMMARY_JSON_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_r2_map(result: Dict[str, np.ndarray], summary: SummaryRow, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(
        result["lon_edges"],
        result["lat_edges"],
        result["r2_map"],
        cmap="viridis",
        shading="flat",
        vmin=0.0,
        vmax=max(0.05, float(np.nanmax(result["r2_map"]))),
    )
    ax.set_title(
        rf"Block factor {summary.block_factor} | mean $R^2={summary.mean_r2:.3f}$ | "
        rf"valid blocks={summary.number_of_valid_blocks}"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
    ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, shrink=0.88).set_label(r"$R^2_G$")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_maps(results: Dict[int, Dict[str, np.ndarray]], summaries: List[SummaryRow]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.08, hspace=0.30, wspace=0.22)
    for ax, row in zip(axes.flat, summaries):
        result = results[row.block_factor]
        mesh = ax.pcolormesh(
            result["lon_edges"],
            result["lat_edges"],
            result["r2_map"],
            cmap="viridis",
            shading="flat",
            vmin=0.0,
            vmax=max(0.05, float(np.nanmax(result["r2_map"]))),
        )
        ax.set_title(rf"factor {row.block_factor} | mean $R^2={row.mean_r2:.3f}$")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
        ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.80).set_label(r"$R^2_G$")
    fig.savefig(COMBINED_MAP_FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_scale_sensitivity(summaries: List[SummaryRow]) -> None:
    factors = np.asarray([row.block_factor for row in summaries], dtype=np.float64)
    mean_r2 = np.asarray([row.mean_r2 for row in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(factors, mean_r2, marker="o", linewidth=2)
    ax.set_xlabel("Block factor")
    ax.set_ylabel(r"Area-weighted mean $R^2$")
    ax.set_title(r"PC1-PC6 Sierra T2m predictability vs. spatial coarsening")
    ax.set_xscale("log", base=2)
    ax.set_xticks(factors)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    fig.savefig(SCALE_FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()
    remove_if_exists(NETCDF_FILE)
    remove_if_exists(SUMMARY_CSV_FILE)
    remove_if_exists(SUMMARY_JSON_FILE)
    remove_if_exists(SCALE_FIGURE_FILE)
    remove_if_exists(COMBINED_MAP_FIGURE_FILE)
    for factor in BLOCK_FACTORS:
        remove_if_exists(HOME_OUTPUT_DIR / MAP_FIGURE_TEMPLATE.format(factor=factor))

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
            "Computing Sierra ERA5-Land T2m PC1-PC6 coarsening experiment for overlap "
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

        results: Dict[int, Dict[str, np.ndarray]] = {}
        summaries: List[SummaryRow] = []
        for factor in BLOCK_FACTORS:
            print(f"  coarsening block factor {factor}", flush=True)
            r2_map, block_weight_map, lat_edges, lon_edges, summary = coarsen_and_regress(
                anomalies=anomalies,
                predictors=predictor_matrix,
                latitude=sierra_latitude,
                longitude=sierra_longitude,
                block_factor=factor,
            )
            results[factor] = {
                "r2_map": r2_map,
                "block_weight_map": block_weight_map,
                "lat_edges": lat_edges,
                "lon_edges": lon_edges,
            }
            summaries.append(summary)

        save_netcdf(
            results=results,
            summaries=summaries,
            overlap_months=overlap_months,
            predictor_matrix_raw_mean=predictor_matrix_raw_mean,
            predictor_matrix_raw_std=predictor_matrix_raw_std,
            runtime=runtime,
        )
        save_summary(summaries=summaries, runtime=runtime)
        for row in summaries:
            plot_r2_map(
                result=results[row.block_factor],
                summary=row,
                output_path=HOME_OUTPUT_DIR / MAP_FIGURE_TEMPLATE.format(factor=row.block_factor),
            )
        plot_combined_maps(results, summaries)
        plot_scale_sensitivity(summaries)
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Combined map figure: {COMBINED_MAP_FIGURE_FILE}", flush=True)
    print(f"Scale sensitivity figure: {SCALE_FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
