#!/usr/bin/env python3
"""
Regenerate monthly Level 2 six-PC R2 maps for the shared WUS-domain SST setup.

Two outputs are produced for a selected WUS dataset/domain:
1. Masked/shared-domain observation reference:
   masked COBE2 projected scores -> ERA5-Land Sierra T2m anomalies
2. Masked/shared-domain WUS result:
   saved WUS projected pseudo-PCs -> WUSD-03 Sierra T2m anomalies

Both follow the exact month-by-month Level 2 OLS logic from the existing
observational monthly sensitivity script:
  B_hat = (A^T A)^(-1) A^T Y, solved numerically with np.linalg.lstsq
  R2(r) = 1 - sum_t[(Y - Y_hat)^2] / sum_t[Y^2]

No expensive WUS daily preprocessing is redone.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
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
from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    COBE2_SST_FILE,
    compute_monthly_climatology_anomalies,
    open_dataset_with_fallbacks,
)
from snow_ml.data import DEFAULT_SIERRA_REGION, RegionBounds


DEFAULT_DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
DEFAULT_DOMAIN = "d03"
N_PREDICTORS = 6
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
SIERRA_REGION_360 = RegionBounds(lat_min=35.0, lat_max=42.0, lon_min=236.0, lon_max=243.0)

PROJECTED_PC_ROOT = PROJECT_ROOT / "artifacts" / "wus_sst_projected_onto_cobe2_eofs"
WUS_T2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_t2_monthly_anomalies")
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "sharedmask_sierra_t2m_level2_pc1to6_monthly_r2"


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


@dataclass(frozen=True)
class SharedMaskReference:
    dataset_id: str
    domain: str
    time: np.ndarray
    shared_mask: np.ndarray
    cobe2_eof: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    projection_weighting: str
    projected_pc_wus_raw: np.ndarray


@dataclass(frozen=True)
class TargetField:
    time: np.ndarray
    anomalies: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    base_sierra_mask: np.ndarray
    weights_2d: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared-mask monthly Level 2 OLS Sierra R2 diagnostics.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="WUS dataset id to use.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="WUS domain, default d03.")
    return parser.parse_args()


def output_dir(dataset_id: str, domain: str) -> Path:
    return OUTPUT_ROOT / domain / dataset_id


def projected_pc_file(dataset_id: str, domain: str) -> Path:
    return PROJECTED_PC_ROOT / domain / dataset_id / "projected_pc_timeseries_and_mask.nc"


def wus_t2_file(dataset_id: str, domain: str) -> Path:
    return WUS_T2_ROOT / domain / dataset_id / f"{dataset_id}_{domain}_t2_monthly_anomaly.nc"


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def month_subset_index(overlap_months: np.ndarray, month_values: Tuple[int, ...]) -> np.ndarray:
    months = np.asarray([month_number(value) for value in overlap_months.tolist()], dtype=np.int32)
    return np.isin(months, np.asarray(month_values, dtype=np.int32))


def standardize_pc_matrix(pc_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(pc_matrix, dtype=np.float64)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
        raise ValueError("Cannot standardize projected scores with non-finite or zero std")
    standardized = (values - means[np.newaxis, :]) / stds[np.newaxis, :]
    return standardized, means, stds


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


def area_weights_from_latitude(latitude: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    return np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], shape)


def masked_area_weighted_mean(r2_map: np.ndarray, weights_2d: np.ndarray, mask: np.ndarray) -> float:
    valid_weights = np.where(mask & np.isfinite(r2_map), weights_2d, 0.0)
    weight_sum = float(np.sum(valid_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(np.sum(np.where(mask, r2_map, 0.0) * valid_weights) / weight_sum)


def summarize_subset(
    subset_name: str,
    subset_group: str,
    r2_map: np.ndarray,
    valid_mask: np.ndarray,
    weights_2d: np.ndarray,
    sierra_region: RegionBounds,
) -> SummaryRow:
    values = np.asarray(r2_map, dtype=np.float64)[valid_mask]
    return SummaryRow(
        subset_name=subset_name,
        subset_group=subset_group,
        n_time_samples=0,
        sierra_lat_min=sierra_region.lat_min,
        sierra_lat_max=sierra_region.lat_max,
        sierra_lon_min=sierra_region.lon_min,
        sierra_lon_max=sierra_region.lon_max,
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


def load_shared_mask_reference(dataset_id: str, domain: str) -> SharedMaskReference:
    path = projected_pc_file(dataset_id, domain)
    if not path.exists():
        raise FileNotFoundError(f"Missing projected PC file: {path}")
    with xr.open_dataset(path) as ds:
        return SharedMaskReference(
            dataset_id=dataset_id,
            domain=domain,
            time=np.asarray(ds["time"].values, dtype="datetime64[ns]"),
            shared_mask=np.asarray(ds["projection_shared_mask"].values, dtype=bool),
            cobe2_eof=np.asarray(ds["cobe2_eof"].values, dtype=np.float64),
            latitude=np.asarray(ds["lat"].values, dtype=np.float64),
            longitude=np.asarray(ds["lon"].values, dtype=np.float64),
            projection_weighting=str(ds.attrs.get("projection_weighting", "")),
            projected_pc_wus_raw=np.asarray(ds["projected_pc"].values, dtype=np.float64),
        )


def normalize_longitude_to_minus180_180(lon: np.ndarray) -> np.ndarray:
    lon_values = np.asarray(lon, dtype=np.float64).copy()
    return np.where(lon_values > 180.0, lon_values - 360.0, lon_values)


def load_cobe2_monthly_anomalies_on_eof_grid() -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading COBE2 SST from {COBE2_SST_FILE}", flush=True)
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        sst = np.asarray(ds["sst"].values, dtype=np.float64)
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        lon = normalize_longitude_to_minus180_180(np.asarray(ds["lon"].values, dtype=np.float64))
        lon_sort_idx = np.argsort(lon)
        sst = sst[:, :, lon_sort_idx]
        lon = lon[lon_sort_idx]
        missing_value = float(ds["sst"].attrs.get("missing_value", 1.0e20))
        sst = np.where(sst >= missing_value, np.nan, sst)
    print(f"Computing COBE2 monthly climatology anomalies for {int(time.size)} months", flush=True)
    climatology, anomalies = compute_monthly_climatology_anomalies(sst, time)
    _ = climatology
    print(f"Finished COBE2 anomalies with shape {anomalies.shape}", flush=True)
    return to_month_start(time), anomalies


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    month_values = to_month_start(time_values)
    index_by_month = {month: idx for idx, month in enumerate(month_values.tolist())}
    return np.asarray(data)[[index_by_month[month] for month in target_months.tolist()]]


def compute_masked_cobe2_projected_scores(
    reference: SharedMaskReference,
    cobe2_time: np.ndarray,
    cobe2_anomalies: np.ndarray,
) -> np.ndarray:
    overlap_anom = select_by_months(cobe2_time, cobe2_anomalies, reference.time)
    shared_mask = reference.shared_mask
    lat_weights_1d = np.sqrt(np.clip(np.cos(np.deg2rad(reference.latitude)), 0.0, None))
    weights_2d = np.broadcast_to(lat_weights_1d[:, np.newaxis], shared_mask.shape)
    weights_flat = weights_2d[shared_mask]
    weighted_anom = np.asarray(overlap_anom[:, shared_mask], dtype=np.float64) * weights_flat[np.newaxis, :]
    weighted_eof = np.asarray(reference.cobe2_eof[:, shared_mask], dtype=np.float64) * weights_flat[np.newaxis, :]
    return weighted_anom @ weighted_eof.T


def load_era5_target(overlap_months: np.ndarray) -> TargetField:
    print(f"Loading ERA5 monthly mean from {ERA5_MONTHLY_MEAN_FILE}", flush=True)
    monthly_mean_ds = xr.open_dataset(
        ERA5_MONTHLY_MEAN_FILE,
        engine="netcdf4",
        chunks={"time": TIME_CHUNK, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )
    print(f"Loading ERA5 monthly climatology from {ERA5_MONTHLY_CLIM_FILE}", flush=True)
    monthly_clim_ds = xr.open_dataset(
        ERA5_MONTHLY_CLIM_FILE,
        engine="netcdf4",
        chunks={"month": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )
    try:
        monthly_mean = subset_era5_region_360(monthly_mean_ds["t2m"], SIERRA_REGION_360)
        monthly_clim = subset_era5_region_360(monthly_clim_ds["t2m"], SIERRA_REGION_360)
        latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float64)
        longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float64)
        era5_time = to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        era5_index = build_time_index(era5_time)
        overlap_indices = [era5_index[month_value] for month_value in overlap_months.tolist()]
        monthly_mean_overlap = monthly_mean.isel(time=overlap_indices).astype(np.float64).load()
        monthly_clim_stack = xr.concat(
            [monthly_clim.sel(month=month_number(month_value)) for month_value in overlap_months.tolist()],
            dim="time",
        )
        anomalies = (monthly_mean_overlap - monthly_clim_stack).values
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()
    print(f"Finished ERA5 overlap anomaly load with shape {anomalies.shape}", flush=True)

    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    base_sierra_mask = (
        (lat2d >= SIERRA_REGION_360.lat_min)
        & (lat2d <= SIERRA_REGION_360.lat_max)
        & (lon2d >= SIERRA_REGION_360.lon_min)
        & (lon2d <= SIERRA_REGION_360.lon_max)
        & np.isfinite(anomalies).all(axis=0)
    )
    if not np.any(base_sierra_mask):
        raise ValueError("No valid Sierra ERA5-Land grid points found in overlap period")
    return TargetField(
        time=overlap_months,
        anomalies=anomalies,
        latitude=latitude,
        longitude=longitude,
        base_sierra_mask=base_sierra_mask,
        weights_2d=area_weights_from_latitude(latitude, anomalies.shape[1:]),
    )


def load_wus_target(dataset_id: str, domain: str, overlap_months: np.ndarray) -> TargetField:
    path = wus_t2_file(dataset_id, domain)
    if not path.exists():
        raise FileNotFoundError(f"Missing WUS monthly T2 anomaly file: {path}")
    print(f"Loading WUS monthly T2 anomalies from {path}", flush=True)
    with xr.open_dataset(path) as ds:
        time = to_month_start(np.asarray(ds["time"].values, dtype="datetime64[ns]"))
        anomalies = select_by_months(time, np.asarray(ds["t2_anomaly"].values, dtype=np.float64), overlap_months)
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.mod(np.asarray(ds["longitude"].values, dtype=np.float64), 360.0)
        landmask = np.asarray(ds["landmask"].values, dtype=bool)
    base_sierra_mask = (
        landmask
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= SIERRA_REGION_360.lat_min)
        & (latitude <= SIERRA_REGION_360.lat_max)
        & (longitude >= SIERRA_REGION_360.lon_min)
        & (longitude <= SIERRA_REGION_360.lon_max)
        & np.isfinite(anomalies).all(axis=0)
    )
    if not np.any(base_sierra_mask):
        raise ValueError("No valid Sierra WUS land grid points found in overlap period")
    print(f"Finished WUS overlap anomaly load with shape {anomalies.shape}", flush=True)
    return TargetField(
        time=overlap_months,
        anomalies=anomalies,
        latitude=latitude,
        longitude=longitude,
        base_sierra_mask=base_sierra_mask,
        weights_2d=np.cos(np.deg2rad(latitude)),
    )


def run_monthly_collection(
    overlap_months: np.ndarray,
    predictor_matrix: np.ndarray,
    target: TargetField,
    subset_group: str,
) -> Tuple[List[SummaryRow], Dict[str, np.ndarray]]:
    rows: List[SummaryRow] = []
    maps: Dict[str, np.ndarray] = {}
    for subset_name, month_values in MONTHLY_SUBSETS:
        use_index = month_subset_index(overlap_months, month_values)
        n_time_samples = int(np.count_nonzero(use_index))
        if n_time_samples <= N_PREDICTORS:
            raise ValueError(f"Subset {subset_name} has too few samples ({n_time_samples}) for six-PC regression")
        subset_predictors = predictor_matrix[use_index]
        subset_anomalies = target.anomalies[use_index]
        subset_r2_map, subset_valid_mask = fit_subset_r2_map(subset_predictors, subset_anomalies, target.base_sierra_mask)
        row = summarize_subset(
            subset_name=subset_name,
            subset_group=subset_group,
            r2_map=subset_r2_map,
            valid_mask=subset_valid_mask,
            weights_2d=target.weights_2d,
            sierra_region=SIERRA_REGION_360,
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


def plot_subset_maps(
    latitude: np.ndarray,
    longitude: np.ndarray,
    subset_order: List[str],
    subset_rows: Dict[str, SummaryRow],
    subset_maps: Dict[str, np.ndarray],
    figure_file: Path,
    title: str,
) -> None:
    vmax = 0.0
    for subset_name in subset_order:
        subset_vmax = float(np.nanmax(subset_maps[subset_name]))
        if np.isfinite(subset_vmax):
            vmax = max(vmax, subset_vmax)
    vmax = max(0.05, vmax)

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 11.8), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.08, hspace=0.32, wspace=0.20)
    axes_flat = np.atleast_1d(axes).flatten()
    lon2d, lat2d = np.meshgrid(longitude, latitude) if longitude.ndim == 1 else (longitude, latitude)

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

    fig.suptitle(title, fontsize=14)
    fig.savefig(figure_file, dpi=220)
    plt.close(fig)


def plot_monthly_mean_r2(rows: List[SummaryRow], figure_file: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(rows), dtype=np.int32)
    y = np.asarray([row.mean_r2 for row in rows], dtype=np.float64)
    labels = [row.subset_name for row in rows]
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sierra mean $R^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(figure_file, dpi=220)
    plt.close(fig)


def save_summary_csv(path: Path, rows: List[SummaryRow]) -> None:
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
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in rows:
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


def save_monthly_netcdf(
    path: Path,
    target: TargetField,
    overlap_months: np.ndarray,
    rows: List[SummaryRow],
    maps: Dict[str, np.ndarray],
    attrs: Dict[str, object],
) -> None:
    subset_names = np.asarray([row.subset_name for row in rows], dtype="U8")
    stack = np.stack([maps[name] for name in subset_names.tolist()], axis=0)
    coords: Dict[str, object] = {
        "time": overlap_months.astype("datetime64[ns]"),
        "monthly_subset": subset_names,
    }
    if target.longitude.ndim == 1:
        coords["latitude"] = target.latitude.astype(np.float32)
        coords["longitude"] = target.longitude.astype(np.float32)
        dims = ("monthly_subset", "latitude", "longitude")
        mask_dims = ("latitude", "longitude")
    else:
        coords["lat2d"] = np.arange(target.latitude.shape[0], dtype=np.int32)
        coords["lon2d"] = np.arange(target.latitude.shape[1], dtype=np.int32)
        coords["latitude"] = (("lat2d", "lon2d"), target.latitude.astype(np.float32))
        coords["longitude"] = (("lat2d", "lon2d"), target.longitude.astype(np.float32))
        dims = ("monthly_subset", "lat2d", "lon2d")
        mask_dims = ("lat2d", "lon2d")
    sanitized_attrs: Dict[str, object] = {}
    for key, value in attrs.items():
        if isinstance(value, (dict, list, tuple)):
            sanitized_attrs[key] = json.dumps(value)
        elif isinstance(value, (bool, np.bool_)):
            sanitized_attrs[key] = "true" if bool(value) else "false"
        elif value is None:
            sanitized_attrs[key] = ""
        else:
            sanitized_attrs[key] = value
    ds = xr.Dataset(
        data_vars={
            "monthly_r2": (dims, stack.astype(np.float32)),
            "monthly_mean_r2": (("monthly_subset",), np.asarray([row.mean_r2 for row in rows], dtype=np.float32)),
            "monthly_n_time_samples": (("monthly_subset",), np.asarray([row.n_time_samples for row in rows], dtype=np.int32)),
            "valid_sierra_mask": (mask_dims, target.base_sierra_mask.astype(np.int8)),
        },
        coords=coords,
        attrs=sanitized_attrs,
    )
    ds.to_netcdf(path, engine="netcdf4")


def run_masked_observation_case(reference: SharedMaskReference, out_dir: Path) -> Dict[str, object]:
    cobe2_time, cobe2_anomalies = load_cobe2_monthly_anomalies_on_eof_grid()
    print("Projecting masked COBE2 scores onto the shared COBE2 EOF templates", flush=True)
    projected_raw = compute_masked_cobe2_projected_scores(reference, cobe2_time, cobe2_anomalies)
    projected_std, projected_mean, projected_std_raw = standardize_pc_matrix(projected_raw)
    target = load_era5_target(reference.time)
    rows, maps = run_monthly_collection(reference.time, projected_std, target, "monthly")

    prefix = "cobe2_sharedmask_sierra_t2m_level2_pc1to6"
    plot_subset_maps(
        latitude=target.latitude,
        longitude=target.longitude,
        subset_order=[name for name, _ in MONTHLY_SUBSETS],
        subset_rows={row.subset_name: row for row in rows},
        subset_maps=maps,
        figure_file=out_dir / f"{prefix}_monthly_r2_maps.png",
        title="Masked/shared-domain COBE2 PC1-PC6 -> ERA5-Land Sierra monthly T2m",
    )
    plot_monthly_mean_r2(
        rows,
        out_dir / f"{prefix}_monthly_mean_r2.png",
        "Masked/shared-domain COBE2 -> ERA5 Sierra monthly mean $R^2$",
    )
    save_summary_csv(out_dir / f"{prefix}_summary.csv", rows)
    summary = {
        "case": "masked_observation_reference",
        "dataset_id": reference.dataset_id,
        "domain": reference.domain,
        "shared_mask_projected_pc_file": str(projected_pc_file(reference.dataset_id, reference.domain)),
        "cobe2_sst_file": str(COBE2_SST_FILE),
        "era5_monthly_mean_file": str(ERA5_MONTHLY_MEAN_FILE),
        "era5_monthly_climatology_file": str(ERA5_MONTHLY_CLIM_FILE),
        "time_start": format_month(reference.time[0]),
        "time_end": format_month(reference.time[-1]),
        "n_overlap_months": int(reference.time.size),
        "pc_standardized": True,
        "pc_standardization": "sample mean and sample std over overlap months (ddof=1)",
        "projected_score_mean_raw": [float(v) for v in projected_mean.tolist()],
        "projected_score_std_raw": [float(v) for v in projected_std_raw.tolist()],
        "projection_weighting": reference.projection_weighting,
        "shared_mask_cell_count": int(np.count_nonzero(reference.shared_mask)),
        "monthly_rows": [asdict(row) for row in rows],
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_monthly_netcdf(
        out_dir / f"{prefix}.nc",
        target=target,
        overlap_months=reference.time,
        rows=rows,
        maps=maps,
        attrs=summary,
    )
    return summary


def run_masked_wus_case(reference: SharedMaskReference, out_dir: Path) -> Dict[str, object]:
    print("Standardizing saved WUS projected pseudo-PCs", flush=True)
    projected_std, projected_mean, projected_std_raw = standardize_pc_matrix(reference.projected_pc_wus_raw)
    target = load_wus_target(reference.dataset_id, reference.domain, reference.time)
    rows, maps = run_monthly_collection(reference.time, projected_std, target, "monthly")

    prefix = "wusd3_sharedmask_sierra_t2m_level2_pc1to6"
    plot_subset_maps(
        latitude=target.latitude,
        longitude=target.longitude,
        subset_order=[name for name, _ in MONTHLY_SUBSETS],
        subset_rows={row.subset_name: row for row in rows},
        subset_maps=maps,
        figure_file=out_dir / f"{prefix}_monthly_r2_maps.png",
        title=f"{reference.dataset_id} masked/shared-domain WUS PC1-PC6 -> WUSD-03 Sierra monthly T2m",
    )
    plot_monthly_mean_r2(
        rows,
        out_dir / f"{prefix}_monthly_mean_r2.png",
        "Masked/shared-domain WUSD-03 -> WUSD-03 Sierra monthly mean $R^2$",
    )
    save_summary_csv(out_dir / f"{prefix}_summary.csv", rows)
    summary = {
        "case": "masked_wusd3_result",
        "dataset_id": reference.dataset_id,
        "domain": reference.domain,
        "shared_mask_projected_pc_file": str(projected_pc_file(reference.dataset_id, reference.domain)),
        "wusd3_t2_anomaly_file": str(wus_t2_file(reference.dataset_id, reference.domain)),
        "time_start": format_month(reference.time[0]),
        "time_end": format_month(reference.time[-1]),
        "n_overlap_months": int(reference.time.size),
        "pc_standardized": True,
        "pc_standardization": "sample mean and sample std over overlap months (ddof=1)",
        "projected_score_mean_raw": [float(v) for v in projected_mean.tolist()],
        "projected_score_std_raw": [float(v) for v in projected_std_raw.tolist()],
        "projection_weighting": reference.projection_weighting,
        "shared_mask_cell_count": int(np.count_nonzero(reference.shared_mask)),
        "monthly_rows": [asdict(row) for row in rows],
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_monthly_netcdf(
        out_dir / f"{prefix}.nc",
        target=target,
        overlap_months=reference.time,
        rows=rows,
        maps=maps,
        attrs=summary,
    )
    return summary


def main() -> None:
    args = parse_args()
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)

    out_dir = output_dir(args.dataset_id, args.domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("cobe2_sharedmask_sierra_t2m_level2_pc1to6*"):
        remove_if_exists(path)
    for path in out_dir.glob("wusd3_sharedmask_sierra_t2m_level2_pc1to6*"):
        remove_if_exists(path)

    reference = load_shared_mask_reference(args.dataset_id, args.domain)
    print(
        f"Using shared-mask reference from {projected_pc_file(args.dataset_id, args.domain)} "
        f"time={format_month(reference.time[0])}..{format_month(reference.time[-1])} "
        f"n_time={int(reference.time.size)} shared_cells={int(np.count_nonzero(reference.shared_mask))}",
        flush=True,
    )
    obs_summary = run_masked_observation_case(reference, out_dir)
    wus_summary = run_masked_wus_case(reference, out_dir)

    comparison = {
        "dataset_id": args.dataset_id,
        "domain": args.domain,
        "time_start": format_month(reference.time[0]),
        "time_end": format_month(reference.time[-1]),
        "n_overlap_months": int(reference.time.size),
        "obs_summary_json": str(out_dir / "cobe2_sharedmask_sierra_t2m_level2_pc1to6_summary.json"),
        "wus_summary_json": str(out_dir / "wusd3_sharedmask_sierra_t2m_level2_pc1to6_summary.json"),
        "obs_monthly_mean_r2": {row["subset_name"]: row["mean_r2"] for row in obs_summary["monthly_rows"]},
        "wus_monthly_mean_r2": {row["subset_name"]: row["mean_r2"] for row in wus_summary["monthly_rows"]},
    }
    (out_dir / "sharedmask_monthly_level2_comparison_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote shared-mask monthly outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
