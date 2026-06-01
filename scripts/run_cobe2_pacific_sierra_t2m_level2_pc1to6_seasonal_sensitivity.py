#!/usr/bin/env python3
"""
Run a Sierra-only seasonal and monthly PC1-PC6 T2m predictability sensitivity
diagnostic using the corrected expanded-domain predictor alignment.
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
    month_number,
    subset_era5_region_360,
    to_month_start,
)
from snow_ml.data import RegionBounds


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_sensitivity"
MODEL_NAME = "PC1_to_PC6"
N_PREDICTORS = 6

EXPANDED_LEVEL2_NETCDF_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6/cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
)
PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL2_PC1TO6_SEASONAL_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_sensitivity",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_sensitivity"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_sensitivity.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_sensitivity_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_sensitivity_summary.json"
SEASONAL_MAP_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_seasonal_r2_maps.png"
MONTHLY_MAP_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_monthly_r2_maps.png"
MONTHLY_LINE_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_monthly_mean_r2.png"

ERA5_VARIABLE = "t2m"
SIERRA_REGION_360 = RegionBounds(lat_min=35.0, lat_max=42.0, lon_min=236.0, lon_max=243.0)

SEASONAL_SUBSETS = [
    ("Dec", (12,)),
    ("Jan", (1,)),
    ("Feb", (2,)),
    ("DJF", (12, 1, 2)),
]
MONTHLY_SUBSETS = [
    ("Jan", (1,)),
    ("Feb", (2,)),
    ("Mar", (3,)),
    ("Apr", (4,)),
    ("May", (5,)),
    ("Jun", (6,)),
    ("Jul", (7,)),
    ("Aug", (8,)),
    ("Sep", (9,)),
    ("Oct", (10,)),
    ("Nov", (11,)),
    ("Dec", (12,)),
]


@dataclass(frozen=True)
class SummaryRow:
    subset_name: str
    subset_group: str
    n_time_samples: int
    sierra_lat_min: float
    sierra_lat_max: float
    sierra_lon_min: float
    sierra_lon_max: float
    number_of_valid_sierra_grid_points: int
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


def build_sierra_box_mask(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(np.asarray(latitude, dtype=np.float64), np.asarray(longitude, dtype=np.float64), indexing="ij")
    return (
        (lat2d >= SIERRA_REGION_360.lat_min)
        & (lat2d <= SIERRA_REGION_360.lat_max)
        & (lon2d >= SIERRA_REGION_360.lon_min)
        & (lon2d <= SIERRA_REGION_360.lon_max)
    )


def area_weights(latitude: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    return np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], shape)


def masked_area_weighted_mean(r2_map: np.ndarray, weights_2d: np.ndarray, mask: np.ndarray) -> float:
    valid_weights = np.where(mask & np.isfinite(r2_map), weights_2d, 0.0)
    weight_sum = float(np.sum(valid_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(np.sum(np.where(mask, r2_map, 0.0) * valid_weights) / weight_sum)


def fit_subset_r2_map(predictors: np.ndarray, anomalies: np.ndarray, base_sierra_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(anomalies, dtype=np.float64)
    a = np.asarray(predictors, dtype=np.float64)
    y_flat = y.reshape(y.shape[0], -1)
    base_mask_flat = base_sierra_mask.reshape(-1)
    valid_columns = base_mask_flat & np.isfinite(y_flat).all(axis=0)
    if not np.any(valid_columns):
        raise ValueError("No valid Sierra grid points available for the requested subset")
    y_valid = y_flat[:, valid_columns]
    b_valid, _, _, _ = np.linalg.lstsq(a, y_valid, rcond=None)
    yhat_valid = a @ b_valid
    residual_sum = np.sum((y_valid - yhat_valid) ** 2, axis=0)
    total_sum = np.sum(y_valid ** 2, axis=0)
    r2_valid = np.full(total_sum.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(total_sum) & (total_sum > 0.0)
    r2_valid[positive] = 1.0 - (residual_sum[positive] / total_sum[positive])
    r2_map = np.full(y.shape[1] * y.shape[2], np.nan, dtype=np.float64)
    r2_map[valid_columns] = r2_valid
    valid_mask = np.full(y.shape[1] * y.shape[2], False, dtype=bool)
    valid_mask[valid_columns] = True
    return r2_map.reshape(y.shape[1], y.shape[2]), valid_mask.reshape(y.shape[1], y.shape[2])


def subset_index(overlap_months: np.ndarray, month_values: Tuple[int, ...]) -> np.ndarray:
    months = np.asarray([month_number(value) for value in overlap_months.tolist()], dtype=np.int32)
    return np.isin(months, np.asarray(month_values, dtype=np.int32))


def summarize_subset(
    subset_name: str,
    subset_group: str,
    r2_map: np.ndarray,
    valid_mask: np.ndarray,
    weights_2d: np.ndarray,
) -> SummaryRow:
    values = np.asarray(r2_map, dtype=np.float64)[valid_mask]
    return SummaryRow(
        subset_name=subset_name,
        subset_group=subset_group,
        n_time_samples=0,
        sierra_lat_min=SIERRA_REGION_360.lat_min,
        sierra_lat_max=SIERRA_REGION_360.lat_max,
        sierra_lon_min=SIERRA_REGION_360.lon_min,
        sierra_lon_max=SIERRA_REGION_360.lon_max,
        number_of_valid_sierra_grid_points=int(np.count_nonzero(valid_mask)),
        mean_r2=masked_area_weighted_mean(r2_map, weights_2d, valid_mask),
        median_r2=float(np.nanmedian(values)),
        max_r2=float(np.nanmax(values)),
        min_r2=float(np.nanmin(values)),
    )


def replace_sample_count(row: SummaryRow, n_time_samples: int) -> SummaryRow:
    return SummaryRow(
        subset_name=row.subset_name,
        subset_group=row.subset_group,
        n_time_samples=n_time_samples,
        sierra_lat_min=row.sierra_lat_min,
        sierra_lat_max=row.sierra_lat_max,
        sierra_lon_min=row.sierra_lon_min,
        sierra_lon_max=row.sierra_lon_max,
        number_of_valid_sierra_grid_points=row.number_of_valid_sierra_grid_points,
        mean_r2=row.mean_r2,
        median_r2=row.median_r2,
        max_r2=row.max_r2,
        min_r2=row.min_r2,
    )


def load_expanded_alignment() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(EXPANDED_LEVEL2_NETCDF_FILE, engine="netcdf4", decode_times=True) as ds:
        overlap_months = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        predictor_matrix = np.asarray(ds["pacific_cobe2_pc"].values, dtype=np.float64)
        latitude = np.asarray(ds["sierra_latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["sierra_longitude"].values, dtype=np.float64)
        expanded_r2_map = np.asarray(ds["sierra_era5_t2m_multi_pc_r2"].values, dtype=np.float64)
    return overlap_months, predictor_matrix, latitude, longitude, expanded_r2_map


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
            "Computing seasonal sensitivity anomalies over the expanded T2m domain for overlap "
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


def plot_subset_maps(
    latitude: np.ndarray,
    longitude: np.ndarray,
    subset_order: List[str],
    subset_rows: Dict[str, SummaryRow],
    subset_maps: Dict[str, np.ndarray],
    figure_file: Path,
    nrows: int,
    ncols: int,
    title: str,
) -> None:
    vmax = 0.0
    for subset_name in subset_order:
        subset_vmax = float(np.nanmax(subset_maps[subset_name]))
        if np.isfinite(subset_vmax):
            vmax = max(vmax, subset_vmax)
    vmax = max(0.05, vmax)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.08, hspace=0.30, wspace=0.20)
    axes_flat = np.atleast_1d(axes).flatten()
    lon2d, lat2d = np.meshgrid(longitude, latitude)

    for ax, subset_name in zip(axes_flat, subset_order):
        row = subset_rows[subset_name]
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            subset_maps[subset_name],
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{subset_name} | n={row.n_time_samples} | mean R2={row.mean_r2:.3f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(SIERRA_REGION_360.lon_min, SIERRA_REGION_360.lon_max)
        ax.set_ylim(SIERRA_REGION_360.lat_min, SIERRA_REGION_360.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.84).set_label(r"$R^2(r)$")

    for ax in axes_flat[len(subset_order):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)


def plot_monthly_mean_r2(monthly_rows: List[SummaryRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(monthly_rows), dtype=np.int32)
    y = np.asarray([row.mean_r2 for row in monthly_rows], dtype=np.float64)
    labels = [row.subset_name for row in monthly_rows]
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sierra mean $R^2$")
    ax.set_title("Sierra T2m PC1-PC6 monthly sensitivity")
    ax.grid(True, alpha=0.3)
    fig.savefig(MONTHLY_LINE_FIGURE_FILE, dpi=200)
    plt.close(fig)


def save_netcdf(
    latitude: np.ndarray,
    longitude: np.ndarray,
    overlap_months: np.ndarray,
    base_sierra_mask: np.ndarray,
    seasonal_rows: List[SummaryRow],
    seasonal_maps: Dict[str, np.ndarray],
    monthly_rows: List[SummaryRow],
    monthly_maps: Dict[str, np.ndarray],
    runtime,
) -> None:
    seasonal_names = np.asarray([row.subset_name for row in seasonal_rows], dtype="U8")
    monthly_names = np.asarray([row.subset_name for row in monthly_rows], dtype="U8")
    seasonal_stack = np.stack([seasonal_maps[name] for name in seasonal_names.tolist()], axis=0)
    monthly_stack = np.stack([monthly_maps[name] for name in monthly_names.tolist()], axis=0)
    ds = xr.Dataset(
        data_vars={
            "valid_sierra_mask": (("latitude", "longitude"), base_sierra_mask.astype(np.int8)),
            "seasonal_r2": (("seasonal_subset", "latitude", "longitude"), seasonal_stack.astype(np.float32)),
            "monthly_r2": (("monthly_subset", "latitude", "longitude"), monthly_stack.astype(np.float32)),
            "seasonal_mean_r2": (("seasonal_subset",), np.asarray([row.mean_r2 for row in seasonal_rows], dtype=np.float32)),
            "monthly_mean_r2": (("monthly_subset",), np.asarray([row.mean_r2 for row in monthly_rows], dtype=np.float32)),
            "seasonal_n_time_samples": (
                ("seasonal_subset",),
                np.asarray([row.n_time_samples for row in seasonal_rows], dtype=np.int32),
            ),
            "monthly_n_time_samples": (
                ("monthly_subset",),
                np.asarray([row.n_time_samples for row in monthly_rows], dtype=np.int32),
            ),
        },
        coords={
            "latitude": latitude.astype(np.float32),
            "longitude": longitude.astype(np.float32),
            "time": overlap_months.astype("datetime64[ns]"),
            "seasonal_subset": seasonal_names,
            "monthly_subset": monthly_names,
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "selection_source": str(EXPANDED_LEVEL2_NETCDF_FILE),
            "expanded_t2m_domain_360": json.dumps(PACIFIC_SST_REGION_360.as_dict()),
            "sierra_evaluation_region_360": json.dumps(SIERRA_REGION_360.as_dict()),
            "selection_note": "Predictors and expanded-domain alignment come from the corrected expanded-domain Level 2 run; final R2 maps and summaries are Sierra-only.",
            "pc_standardized": "true",
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(NETCDF_FILE, engine="netcdf4")


def save_summary(seasonal_rows: List[SummaryRow], monthly_rows: List[SummaryRow], runtime) -> None:
    fieldnames = [
        "subset_name",
        "subset_group",
        "n_time_samples",
        "Sierra_lat_min",
        "Sierra_lat_max",
        "Sierra_lon_min",
        "Sierra_lon_max",
        "number_of_valid_Sierra_grid_points",
        "mean_R2",
        "median_R2",
        "max_R2",
        "min_R2",
    ]
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in seasonal_rows + monthly_rows:
            writer.writerow(
                [
                    row.subset_name,
                    row.subset_group,
                    row.n_time_samples,
                    "{:.12g}".format(row.sierra_lat_min),
                    "{:.12g}".format(row.sierra_lat_max),
                    "{:.12g}".format(row.sierra_lon_min),
                    "{:.12g}".format(row.sierra_lon_max),
                    row.number_of_valid_sierra_grid_points,
                    "{:.12g}".format(row.mean_r2),
                    "{:.12g}".format(row.median_r2),
                    "{:.12g}".format(row.max_r2),
                    "{:.12g}".format(row.min_r2),
                ]
            )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "selection_source": str(EXPANDED_LEVEL2_NETCDF_FILE),
        "expanded_t2m_domain_360": PACIFIC_SST_REGION_360.as_dict(),
        "sierra_evaluation_region_360": SIERRA_REGION_360.as_dict(),
        "seasonal_rows": [asdict(row) for row in seasonal_rows],
        "monthly_rows": [asdict(row) for row in monthly_rows],
        "slurm_job_id": runtime.slurm_job_id,
        "compute_node": runtime.hostname,
        "output_directory_size": output_dir_size_text(),
    }
    SUMMARY_JSON_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_subset_collection(
    subset_group: str,
    subset_defs: List[Tuple[str, Tuple[int, ...]]],
    overlap_months: np.ndarray,
    predictor_matrix: np.ndarray,
    anomalies: np.ndarray,
    base_sierra_mask: np.ndarray,
    weights_2d: np.ndarray,
) -> Tuple[List[SummaryRow], Dict[str, np.ndarray]]:
    rows: List[SummaryRow] = []
    maps: Dict[str, np.ndarray] = {}
    for subset_name, month_values in subset_defs:
        use_index = subset_index(overlap_months, month_values)
        n_time_samples = int(np.count_nonzero(use_index))
        if n_time_samples <= N_PREDICTORS:
            raise ValueError(f"Subset {subset_name} has too few samples ({n_time_samples}) for {N_PREDICTORS}-PC regression")
        subset_predictors = predictor_matrix[use_index]
        subset_anomalies = anomalies[use_index]
        subset_r2_map, subset_valid_mask = fit_subset_r2_map(subset_predictors, subset_anomalies, base_sierra_mask)
        row = summarize_subset(
            subset_name=subset_name,
            subset_group=subset_group,
            r2_map=subset_r2_map,
            valid_mask=subset_valid_mask,
            weights_2d=weights_2d,
        )
        row = replace_sample_count(row, n_time_samples)
        rows.append(row)
        maps[subset_name] = subset_r2_map
        print(
            f"  {subset_group} subset {subset_name}: n={n_time_samples}, "
            f"valid Sierra cells={row.number_of_valid_sierra_grid_points}, mean R2={row.mean_r2:.4f}",
            flush=True,
        )
    return rows, maps


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()
    for path in [
        NETCDF_FILE,
        SUMMARY_CSV_FILE,
        SUMMARY_JSON_FILE,
        SEASONAL_MAP_FIGURE_FILE,
        MONTHLY_MAP_FIGURE_FILE,
        MONTHLY_LINE_FIGURE_FILE,
    ]:
        remove_if_exists(path)

    overlap_months, predictor_matrix, latitude, longitude, expanded_r2_map = load_expanded_alignment()
    anomalies = load_aligned_anomalies(overlap_months)
    if anomalies.shape[1:] != expanded_r2_map.shape:
        raise ValueError(
            f"Expanded anomaly grid {anomalies.shape[1:]} does not match expanded regression grid {expanded_r2_map.shape}"
        )

    sierra_box_mask = build_sierra_box_mask(latitude, longitude)
    base_sierra_mask = sierra_box_mask & np.isfinite(expanded_r2_map) & np.isfinite(anomalies).all(axis=0)
    if not np.any(base_sierra_mask):
        raise ValueError("No valid Sierra ERA5-Land grid points found for seasonal sensitivity evaluation")
    weights_2d = area_weights(latitude, expanded_r2_map.shape)

    dec_count = int(np.count_nonzero(subset_index(overlap_months, (12,))))
    jan_count = int(np.count_nonzero(subset_index(overlap_months, (1,))))
    feb_count = int(np.count_nonzero(subset_index(overlap_months, (2,))))
    djf_count = int(np.count_nonzero(subset_index(overlap_months, (12, 1, 2))))

    print(f"Expanded T2m prediction domain lat/lon range: {latitude.min():.3f}..{latitude.max():.3f}, {longitude.min():.3f}..{longitude.max():.3f}", flush=True)
    print(
        "Sierra evaluation domain lat/lon range: "
        f"{SIERRA_REGION_360.lat_min:.3f}..{SIERRA_REGION_360.lat_max:.3f}, "
        f"{SIERRA_REGION_360.lon_min:.3f}..{SIERRA_REGION_360.lon_max:.3f}",
        flush=True,
    )
    print(f"Number of valid Sierra grid points: {int(np.count_nonzero(base_sierra_mask))}", flush=True)
    print(f"Time range used: {format_month(overlap_months[0])} to {format_month(overlap_months[-1])}", flush=True)
    print(f"Number of samples: Dec={dec_count}, Jan={jan_count}, Feb={feb_count}, DJF={djf_count}", flush=True)
    print("Confirmed: final R2 maps and summary statistics are Sierra-only.", flush=True)

    seasonal_rows, seasonal_maps = run_subset_collection(
        subset_group="seasonal",
        subset_defs=SEASONAL_SUBSETS,
        overlap_months=overlap_months,
        predictor_matrix=predictor_matrix,
        anomalies=anomalies,
        base_sierra_mask=base_sierra_mask,
        weights_2d=weights_2d,
    )
    monthly_rows, monthly_maps = run_subset_collection(
        subset_group="monthly",
        subset_defs=MONTHLY_SUBSETS,
        overlap_months=overlap_months,
        predictor_matrix=predictor_matrix,
        anomalies=anomalies,
        base_sierra_mask=base_sierra_mask,
        weights_2d=weights_2d,
    )

    seasonal_row_map = {row.subset_name: row for row in seasonal_rows}
    monthly_row_map = {row.subset_name: row for row in monthly_rows}
    plot_subset_maps(
        latitude=latitude,
        longitude=longitude,
        subset_order=[name for name, _ in SEASONAL_SUBSETS],
        subset_rows=seasonal_row_map,
        subset_maps=seasonal_maps,
        figure_file=SEASONAL_MAP_FIGURE_FILE,
        nrows=2,
        ncols=2,
        title="Sierra T2m seasonal PC1-PC6 sensitivity",
    )
    plot_subset_maps(
        latitude=latitude,
        longitude=longitude,
        subset_order=[name for name, _ in MONTHLY_SUBSETS],
        subset_rows=monthly_row_map,
        subset_maps=monthly_maps,
        figure_file=MONTHLY_MAP_FIGURE_FILE,
        nrows=3,
        ncols=4,
        title="Sierra T2m monthly PC1-PC6 sensitivity",
    )
    plot_monthly_mean_r2(monthly_rows)
    save_netcdf(
        latitude=latitude,
        longitude=longitude,
        overlap_months=overlap_months,
        base_sierra_mask=base_sierra_mask,
        seasonal_rows=seasonal_rows,
        seasonal_maps=seasonal_maps,
        monthly_rows=monthly_rows,
        monthly_maps=monthly_maps,
        runtime=runtime,
    )
    save_summary(seasonal_rows, monthly_rows, runtime)

    print(f"Expanded-domain source: {EXPANDED_LEVEL2_NETCDF_FILE}", flush=True)
    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Seasonal map figure: {SEASONAL_MAP_FIGURE_FILE}", flush=True)
    print(f"Monthly map figure: {MONTHLY_MAP_FIGURE_FILE}", flush=True)
    print(f"Monthly line figure: {MONTHLY_LINE_FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
