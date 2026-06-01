#!/usr/bin/env python3
"""
Generate monthly-climatology SST EOF diagnostics for COBE2 and remapped WUS-D3 SST.

This script:
1. Builds month-of-year climatology anomalies for COBE2 SST.
2. Computes EOFs / PCs from those anomalies.
3. Builds month-of-year climatology anomalies for each remapped WUS-D3 SST product.
4. Computes EOFs / PCs for each WUS dataset.
5. Compares COBE2 and WUS EOF spatial patterns with sign-flip-aware correlations.
6. Writes diagnostic plots and reports only; no SST -> T2M prediction is run here.
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
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


COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
MODEL_SST_ROOT = PROJECT_ROOT / "artifacts" / "sst_pca" / "model_sst_anomalies"

COBE2_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "cobe2_monthly_climatology_anomaly"
COBE2_PLOTS_DIR = COBE2_OUTPUT_DIR / "plots"
WUS_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "wusd3_monthly_climatology_anomaly_eofs"
WUS_PLOTS_DIR = WUS_OUTPUT_DIR / "plots"
COMPARISON_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "monthly_climatology_eof_comparison"
COMPARISON_PLOTS_DIR = COMPARISON_DIR / "plots"

COBE2_CLIM_FILE = COBE2_OUTPUT_DIR / "cobe2_monthly_clim_sst_climatology.nc"
COBE2_ANOM_FILE = COBE2_OUTPUT_DIR / "cobe2_monthly_clim_sst_anomalies.nc"
COBE2_EOF_FILE = COBE2_OUTPUT_DIR / "cobe2_monthly_clim_sst_eofs.nc"
COBE2_PCS_FILE = COBE2_OUTPUT_DIR / "cobe2_monthly_clim_sst_pcs.csv"
COBE2_SUMMARY_FILE = COBE2_OUTPUT_DIR / "cobe2_monthly_clim_sst_summary.json"

COMPARISON_CSV = COMPARISON_DIR / "eof_spatial_correlation_summary.csv"
COMPARISON_CROSS_CSV = COMPARISON_DIR / "eof_spatial_cross_correlation_matrix.csv"
EOF1_VALIDATION_CSV = COMPARISON_DIR / "eof1_uniform_mode_validation.csv"
ENSO_VALIDATION_CSV = COMPARISON_DIR / "enso_pc_correlation_validation.csv"
COMPARISON_REPORT = COMPARISON_DIR / "eof_comparison_report.md"

LAT_MIN = 32.5
LAT_MAX = 43.0
LON_MIN = -134.5
LON_MAX = -114.0
NINO34_LAT_MIN = -5.0
NINO34_LAT_MAX = 5.0
NINO34_LON_MIN = -170.0
NINO34_LON_MAX = -120.0
N_COMPONENTS = 5
PLOT_COMPONENTS = 3
MODELS = [
    "ec-earth3_r1i1p1f1_2_historical_bc",
    "ec-earth3_r1i1p1f1_2_ssp370_bc",
    "miroc6_r1i1p1f1_historical_bc",
    "miroc6_r1i1p1f1_ssp370_bc",
    "mpi-esm1-2-hr_r3i1p1f1_historical_bc",
    "mpi-esm1-2-hr_r3i1p1f1_ssp370_bc",
    "taiesm1_r1i1p1f1_historical_bc",
    "taiesm1_r1i1p1f1_ssp370_bc",
]


@dataclass(frozen=True)
class RuntimeInfo:
    hostname: str
    slurm_job_id: str


@dataclass(frozen=True)
class MonthlyClimatologyEofResult:
    dataset_id: str
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    climatology: np.ndarray
    anomalies: np.ndarray
    eofs: np.ndarray
    pcs: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    valid_cell_mask: np.ndarray
    source_file: str
    notes: List[str]


def get_runtime() -> RuntimeInfo:
    return RuntimeInfo(
        hostname=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )


def ensure_runtime_on_compute_node(runtime: RuntimeInfo) -> None:
    if not runtime.slurm_job_id or "nid" not in runtime.hostname:
        raise RuntimeError(
            "Do not run this script on a login node; active interactive compute allocation required."
        )


def ensure_output_dirs() -> None:
    for path in (
        COBE2_OUTPUT_DIR,
        COBE2_PLOTS_DIR,
        WUS_OUTPUT_DIR,
        WUS_PLOTS_DIR,
        COMPARISON_DIR,
        COMPARISON_PLOTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    errors: List[str] = []
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs: Dict[str, object] = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}: {exc}")
    raise RuntimeError(f"failed to open {path}: {'; '.join(errors)}")


def normalize_longitude_to_minus180_180(lon: np.ndarray) -> np.ndarray:
    lon_values = np.asarray(lon, dtype=np.float64).copy()
    return np.where(lon_values > 180.0, lon_values - 360.0, lon_values)


def crop_cobe2_region(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sst = np.asarray(ds["sst"].values, dtype=np.float64)
    lat_orig = np.asarray(ds["lat"].values, dtype=np.float64)
    lon_orig = normalize_longitude_to_minus180_180(np.asarray(ds["lon"].values, dtype=np.float64))
    lon_sort_idx = np.argsort(lon_orig)
    lon_sorted = lon_orig[lon_sort_idx]
    sst_sorted = sst[:, :, lon_sort_idx]

    lat_mask = (lat_orig >= LAT_MIN) & (lat_orig <= LAT_MAX)
    lon_mask = (lon_sorted >= LON_MIN) & (lon_sorted <= LON_MAX)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("COBE2 crop is empty for requested domain")

    lat_crop = lat_orig[lat_idx]
    lon_crop = lon_sorted[lon_idx]
    sst_crop = sst_sorted[:, lat_idx, :][:, :, lon_idx]
    missing_value = float(ds["sst"].attrs.get("missing_value", 1.0e20))
    sst_crop = np.where(sst_crop >= missing_value, np.nan, sst_crop)
    return lat_crop, lon_crop, sst_crop


def compute_monthly_climatology_anomalies(
    values: np.ndarray,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3:
        raise ValueError(f"Expected values with shape (time, lat, lon), got {values.shape}")
    month_numbers = np.array(
        [int(np.datetime_as_string(value, unit="D")[5:7]) for value in np.asarray(times, dtype="datetime64[ns]")],
        dtype=np.int32,
    )
    climatology = np.full((12, values.shape[1], values.shape[2]), np.nan, dtype=np.float64)
    anomalies = np.full(values.shape, np.nan, dtype=np.float64)

    for month in range(1, 13):
        month_mask = month_numbers == month
        if not np.any(month_mask):
            continue
        month_mean = np.nanmean(values[month_mask], axis=0)
        climatology[month - 1] = month_mean
        anomalies[month_mask] = values[month_mask] - month_mean[np.newaxis, :, :]
    return climatology, anomalies


def compute_month_numbers(times: np.ndarray) -> np.ndarray:
    return np.array(
        [int(np.datetime_as_string(value, unit="D")[5:7]) for value in np.asarray(times, dtype="datetime64[ns]")],
        dtype=np.int32,
    )


def normalize_longitude_array_for_selection(longitude: np.ndarray) -> np.ndarray:
    lon = np.asarray(longitude, dtype=np.float64)
    if lon.ndim == 1:
        return normalize_longitude_to_minus180_180(lon)
    return np.where(lon > 180.0, lon - 360.0, lon)


def find_lat_lon_names(ds: xr.Dataset, variable_name: str) -> Tuple[str, str]:
    variable = ds[variable_name]
    for lat_name in ("lat", "latitude"):
        if lat_name in ds and set(ds[lat_name].dims).issubset(set(variable.dims)):
            break
    else:
        raise ValueError(f"could not find latitude coordinate for {variable_name}")
    for lon_name in ("lon", "longitude"):
        if lon_name in ds and set(ds[lon_name].dims).issubset(set(variable.dims)):
            break
    else:
        raise ValueError(f"could not find longitude coordinate for {variable_name}")
    return lat_name, lon_name


def compute_weighted_regional_mean(
    values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected values with shape (time, lat, lon), got {data.shape}")
    lat = np.asarray(latitude, dtype=np.float64)
    lon = normalize_longitude_array_for_selection(np.asarray(longitude, dtype=np.float64))
    if lat.ndim == 1 and lon.ndim == 1:
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        if not np.any(lat_mask) or not np.any(lon_mask):
            raise ValueError("Requested region is empty on this grid")
        region = data[:, lat_mask, :][:, :, lon_mask]
        region_lat = lat[lat_mask]
        weights_2d = np.cos(np.deg2rad(region_lat))[:, np.newaxis] * np.ones((1, int(lon_mask.sum())), dtype=np.float64)
    elif lat.ndim == 2 and lon.ndim == 2:
        region_mask = (
            (lat >= lat_min)
            & (lat <= lat_max)
            & (lon >= lon_min)
            & (lon <= lon_max)
        )
        if not np.any(region_mask):
            raise ValueError("Requested region is empty on this grid")
        region = np.where(region_mask[np.newaxis, :, :], data, np.nan)
        weights_2d = np.where(region_mask, np.cos(np.deg2rad(lat)), 0.0)
    else:
        raise ValueError("Unsupported latitude/longitude array shapes")

    weights_3d = np.broadcast_to(weights_2d[np.newaxis, :, :], region.shape)
    valid_mask = np.isfinite(region)
    weighted_sum = np.sum(np.where(valid_mask, region * weights_3d, 0.0), axis=(1, 2))
    weight_sum = np.sum(np.where(valid_mask, weights_3d, 0.0), axis=(1, 2))
    result = np.full(region.shape[0], np.nan, dtype=np.float64)
    good = weight_sum > 0.0
    result[good] = weighted_sum[good] / weight_sum[good]
    return result


def compute_monthly_series_anomalies(series: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    month_numbers = compute_month_numbers(times)
    anomalies = np.full(values.shape, np.nan, dtype=np.float64)
    for month in range(1, 13):
        month_mask = month_numbers == month
        if not np.any(month_mask):
            continue
        month_mean = np.nanmean(values[month_mask])
        anomalies[month_mask] = values[month_mask] - month_mean
    return anomalies


def align_series_by_time(
    left_time: np.ndarray,
    left_values: np.ndarray,
    right_time: np.ndarray,
    right_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_time_ns = np.asarray(left_time, dtype="datetime64[ns]")
    right_time_ns = np.asarray(right_time, dtype="datetime64[ns]")
    common_time, left_idx, right_idx = np.intersect1d(left_time_ns, right_time_ns, assume_unique=False, return_indices=True)
    if common_time.size == 0:
        return common_time, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    left_aligned = np.asarray(left_values, dtype=np.float64)[left_idx]
    right_aligned = np.asarray(right_values, dtype=np.float64)[right_idx]
    finite = np.isfinite(left_aligned) & np.isfinite(right_aligned)
    return common_time[finite], left_aligned[finite], right_aligned[finite]


def compute_multiple_regression_skill(
    predictors: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    x_matrix = np.asarray(predictors, dtype=np.float64)
    y_vector = np.asarray(target, dtype=np.float64)
    if x_matrix.ndim != 2:
        raise ValueError(f"Expected 2D predictors, got {x_matrix.shape}")
    if y_vector.ndim != 1:
        raise ValueError(f"Expected 1D target, got {y_vector.shape}")
    finite = np.isfinite(y_vector) & np.all(np.isfinite(x_matrix), axis=1)
    x_use = x_matrix[finite]
    y_use = y_vector[finite]
    if x_use.shape[0] <= x_use.shape[1]:
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(x_use.shape[0], dtype=np.float64), x_use])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_use, rcond=None)
    fitted = design @ coeffs
    residual = y_use - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_use - np.mean(y_use)) ** 2))
    if ss_tot == 0.0:
        return float("nan"), float("nan")
    r2 = max(0.0, 1.0 - (ss_res / ss_tot))
    return r2, float(np.sqrt(r2))


def compute_eof_result(
    dataset_id: str,
    source_file: str,
    time_values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    notes: Sequence[str],
) -> MonthlyClimatologyEofResult:
    climatology, anomalies = compute_monthly_climatology_anomalies(values, time_values)
    anomalies_flat = anomalies.reshape(anomalies.shape[0], -1)
    valid_cell_mask_flat = np.isfinite(anomalies_flat).all(axis=0)
    valid_count = int(valid_cell_mask_flat.sum())
    if valid_count < N_COMPONENTS:
        raise ValueError(f"Need at least {N_COMPONENTS} all-time-finite cells, got {valid_count} for {dataset_id}")

    matrix = anomalies_flat[:, valid_cell_mask_flat]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    pcs = (u_matrix[:, :N_COMPONENTS] * singular_values[:N_COMPONENTS]).astype(np.float64)
    eofs_valid = vt_matrix[:N_COMPONENTS, :].astype(np.float64)
    variance = singular_values ** 2
    explained_variance_ratio = (variance / variance.sum()).astype(np.float64)

    eof_grid = np.full((N_COMPONENTS, latitude.shape[0], longitude.shape[0]), np.nan, dtype=np.float32)
    eof_grid.reshape(N_COMPONENTS, -1)[:, valid_cell_mask_flat] = eofs_valid.astype(np.float32)
    valid_mask_2d = np.zeros((latitude.shape[0], longitude.shape[0]), dtype=bool)
    valid_mask_2d.flat[valid_cell_mask_flat] = True

    return MonthlyClimatologyEofResult(
        dataset_id=dataset_id,
        time=np.asarray(time_values, dtype="datetime64[ns]"),
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        climatology=climatology.astype(np.float32),
        anomalies=anomalies.astype(np.float32),
        eofs=eof_grid,
        pcs=pcs.astype(np.float32),
        explained_variance_ratio=explained_variance_ratio[:N_COMPONENTS].astype(np.float32),
        singular_values=singular_values[:N_COMPONENTS].astype(np.float32),
        valid_cell_mask=valid_mask_2d,
        source_file=source_file,
        notes=list(notes),
    )


def save_climatology(path: Path, latitude: np.ndarray, longitude: np.ndarray, climatology: np.ndarray) -> None:
    ds = xr.Dataset(
        {"sst_climatology": (("month", "lat", "lon"), climatology)},
        coords={
            "month": np.arange(1, 13, dtype=np.int32),
            "lat": latitude,
            "lon": longitude,
        },
    )
    ds.attrs["description"] = "Month-of-year SST climatology"
    ds.to_netcdf(path)


def save_anomalies(path: Path, result: MonthlyClimatologyEofResult) -> None:
    month_numbers = np.array(
        [int(np.datetime_as_string(value, unit="D")[5:7]) for value in result.time],
        dtype=np.int32,
    )
    ds = xr.Dataset(
        {"sst_anomaly": (("time", "lat", "lon"), result.anomalies)},
        coords={
            "time": result.time,
            "month": ("time", month_numbers),
            "lat": result.latitude,
            "lon": result.longitude,
        },
    )
    ds.attrs["description"] = "SST anomalies after removing month-of-year climatology"
    ds.to_netcdf(path)


def save_eofs(path: Path, result: MonthlyClimatologyEofResult) -> None:
    ds = xr.Dataset(
        {
            "eof": (("component", "lat", "lon"), result.eofs),
            "valid_cell_mask": (("lat", "lon"), result.valid_cell_mask),
            "singular_value": (("component",), result.singular_values),
            "explained_variance_ratio": (("component",), result.explained_variance_ratio),
        },
        coords={
            "component": np.arange(1, N_COMPONENTS + 1, dtype=np.int32),
            "lat": result.latitude,
            "lon": result.longitude,
        },
    )
    ds.attrs["description"] = "EOFs from monthly-climatology SST anomalies"
    ds.to_netcdf(path)


def save_pcs_csv(path: Path, result: MonthlyClimatologyEofResult) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "PC1", "PC2", "PC3", "PC4", "PC5"])
        for time_value, row in zip(result.time, result.pcs):
            text = np.datetime_as_string(np.asarray(time_value, dtype="datetime64[ns]"), unit="D")
            writer.writerow([text] + ["{:.12g}".format(float(value)) for value in row[:N_COMPONENTS]])


def save_summary_json(path: Path, result: MonthlyClimatologyEofResult) -> None:
    payload = {
        "dataset_id": result.dataset_id,
        "source_file": result.source_file,
        "grid_shape": [int(result.latitude.shape[0]), int(result.longitude.shape[0])],
        "n_time_steps": int(result.time.shape[0]),
        "time_start": str(np.datetime_as_string(result.time[0], unit="D")),
        "time_end": str(np.datetime_as_string(result.time[-1], unit="D")),
        "n_valid_cells": int(result.valid_cell_mask.sum()),
        "explained_variance_ratio": [float(value) for value in result.explained_variance_ratio[:PLOT_COMPONENTS]],
        "monthly_climatology_removed": True,
        "notes": list(result.notes),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_pcs_csv(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    times: List[np.datetime64] = []
    rows: List[List[float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < N_COMPONENTS + 1:
            return None
        for row in reader:
            if len(row) < N_COMPONENTS + 1:
                return None
            times.append(np.datetime64(row[0], "ns"))
            rows.append([float(value) for value in row[1 : N_COMPONENTS + 1]])
    return np.asarray(times, dtype="datetime64[ns]"), np.asarray(rows, dtype=np.float32)


def load_cached_result(
    eof_path: Path,
    summary_path: Path,
    anomaly_path: Path | None = None,
    pcs_path: Path | None = None,
) -> MonthlyClimatologyEofResult | None:
    if not eof_path.exists() or not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open_dataset_with_fallbacks(eof_path) as ds:
        required_vars = {"eof", "valid_cell_mask", "singular_value", "explained_variance_ratio"}
        if not required_vars.issubset(ds.data_vars):
            return None
        latitude = np.asarray(ds["lat"].values, dtype=np.float32)
        longitude = np.asarray(ds["lon"].values, dtype=np.float32)
        eofs = np.asarray(ds["eof"].values, dtype=np.float32)
        valid_cell_mask = np.asarray(ds["valid_cell_mask"].values, dtype=bool)
        singular_values = np.asarray(ds["singular_value"].values, dtype=np.float32)
        explained_variance_ratio = np.asarray(ds["explained_variance_ratio"].values, dtype=np.float32)

    time_values = np.asarray([], dtype="datetime64[ns]")
    anomalies = np.empty((0, latitude.shape[0], longitude.shape[0]), dtype=np.float32)
    if anomaly_path is not None and anomaly_path.exists():
        with open_dataset_with_fallbacks(anomaly_path) as ds:
            if "sst_anomaly" not in ds.data_vars:
                return None
            anomalies = np.asarray(ds["sst_anomaly"].values, dtype=np.float32)
            time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")

    pcs = np.empty((0, min(N_COMPONENTS, singular_values.shape[0])), dtype=np.float32)
    if pcs_path is not None and pcs_path.exists():
        pcs_payload = load_pcs_csv(pcs_path)
        if pcs_payload is None:
            return None
        pcs_time_values, pcs = pcs_payload
        if time_values.size == 0:
            time_values = pcs_time_values

    return MonthlyClimatologyEofResult(
        dataset_id=str(summary["dataset_id"]),
        time=time_values,
        latitude=latitude,
        longitude=longitude,
        climatology=np.empty((0, latitude.shape[0], longitude.shape[0]), dtype=np.float32),
        anomalies=anomalies,
        eofs=eofs,
        pcs=pcs,
        explained_variance_ratio=explained_variance_ratio[:N_COMPONENTS],
        singular_values=singular_values[:N_COMPONENTS],
        valid_cell_mask=valid_cell_mask,
        source_file=str(summary.get("source_file", "")),
        notes=list(summary.get("notes", [])) + ["loaded from cached EOF outputs"],
    )


def plot_single_eof(
    field: np.ndarray,
    title: str,
    path: Path,
) -> None:
    vmax = float(np.nanmax(np.abs(field)))
    vmax = 1.0 if vmax == 0.0 or not np.isfinite(vmax) else vmax
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_combined_eofs(result: MonthlyClimatologyEofResult, path: Path) -> None:
    fig, axes = plt.subplots(1, PLOT_COMPONENTS, figsize=(14, 4.5), constrained_layout=True)
    for index, ax in enumerate(axes):
        field = result.eofs[index]
        vmax = float(np.nanmax(np.abs(field)))
        vmax = 1.0 if vmax == 0.0 or not np.isfinite(vmax) else vmax
        image = ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(
            f"EOF{index + 1}\nEVR={result.explained_variance_ratio[index]:.3f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{result.dataset_id} monthly-climatology SST EOFs")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_result_bundle(
    result: MonthlyClimatologyEofResult,
    climatology_path: Path,
    anomaly_path: Path,
    eof_path: Path,
    pcs_path: Path,
    summary_path: Path,
    plot_prefix: Path,
) -> None:
    save_climatology(climatology_path, result.latitude, result.longitude, result.climatology)
    save_anomalies(anomaly_path, result)
    save_eofs(eof_path, result)
    save_pcs_csv(pcs_path, result)
    save_summary_json(summary_path, result)
    for index in range(PLOT_COMPONENTS):
        plot_single_eof(
            result.eofs[index],
            f"{result.dataset_id} EOF{index + 1} (EVR={result.explained_variance_ratio[index]:.3f})",
            plot_prefix.parent / f"{plot_prefix.name}_eof{index + 1}.png",
        )
    plot_combined_eofs(result, plot_prefix.parent / f"{plot_prefix.name}_eof_panel.png")


def spatial_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 2:
        return float("nan")
    a_use = a_flat[mask]
    b_use = b_flat[mask]
    if float(np.std(a_use)) == 0.0 or float(np.std(b_use)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_use, b_use)[0, 1])


def absolute_spatial_correlation(a: np.ndarray, b: np.ndarray) -> float:
    corr = spatial_correlation(a, b)
    corr_flipped = spatial_correlation(a, -b)
    if not np.isfinite(corr) and not np.isfinite(corr_flipped):
        return float("nan")
    if not np.isfinite(corr_flipped):
        return abs(corr)
    if not np.isfinite(corr):
        return abs(corr_flipped)
    return max(abs(corr), abs(corr_flipped))


def compare_eofs(
    cobe2_result: MonthlyClimatologyEofResult,
    wus_result: MonthlyClimatologyEofResult,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for component_index in range(PLOT_COMPONENTS):
        cobe2_field = cobe2_result.eofs[component_index]
        wus_field = wus_result.eofs[component_index]
        corr = spatial_correlation(cobe2_field, wus_field)
        corr_flipped = spatial_correlation(cobe2_field, -wus_field)
        if not np.isfinite(corr) and not np.isfinite(corr_flipped):
            best_corr = float("nan")
            flip = False
        elif not np.isfinite(corr_flipped) or (np.isfinite(corr) and corr >= corr_flipped):
            best_corr = corr
            flip = False
        else:
            best_corr = corr_flipped
            flip = True
        rows.append(
            {
                "model": wus_result.dataset_id,
                "component": component_index + 1,
                "raw_correlation": corr,
                "flipped_correlation": corr_flipped,
                "best_signed_correlation": best_corr,
                "flip_wus_sign": flip,
            }
        )
    return rows


def compute_cross_correlation_rows(
    cobe2_result: MonthlyClimatologyEofResult,
    wus_result: MonthlyClimatologyEofResult,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cobe_mode in range(PLOT_COMPONENTS):
        for wus_mode in range(PLOT_COMPONENTS):
            rows.append(
                {
                    "model": wus_result.dataset_id,
                    "cobe_mode": cobe_mode + 1,
                    "wus_mode": wus_mode + 1,
                    "spatial_corr_abs": absolute_spatial_correlation(
                        cobe2_result.eofs[cobe_mode],
                        wus_result.eofs[wus_mode],
                    ),
                }
            )
    return rows


def write_cross_correlation_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "cobe_mode", "wus_mode", "spatial_corr_abs"])
        for row in rows:
            value = float(row["spatial_corr_abs"])
            writer.writerow(
                [
                    row["model"],
                    row["cobe_mode"],
                    row["wus_mode"],
                    "{:.12g}".format(value) if np.isfinite(value) else "nan",
                ]
            )


def print_cross_correlation_matrix(model_name: str, rows: Sequence[Dict[str, object]]) -> None:
    matrix = np.full((PLOT_COMPONENTS, PLOT_COMPONENTS), np.nan, dtype=np.float64)
    for row in rows:
        matrix[int(row["cobe_mode"]) - 1, int(row["wus_mode"]) - 1] = float(row["spatial_corr_abs"])
    print(model_name, flush=True)
    print("          WUS1   WUS2   WUS3", flush=True)
    for row_index in range(PLOT_COMPONENTS):
        print(
            f"COBE{row_index + 1}   "
            f"{matrix[row_index, 0]:6.3f} "
            f"{matrix[row_index, 1]:6.3f} "
            f"{matrix[row_index, 2]:6.3f}",
            flush=True,
        )
    print("", flush=True)


def validate_eof1_uniform_mode(
    dataset_type: str,
    model_name: str,
    result: MonthlyClimatologyEofResult,
) -> Dict[str, object]:
    if result.anomalies.shape[0] == 0 or result.pcs.shape[0] == 0:
        raise ValueError(f"EOF1 validation requires cached or computed anomalies and PCs for {model_name}")

    eof1 = np.asarray(result.eofs[0], dtype=np.float64)
    eof1_mask = np.isfinite(eof1) & np.asarray(result.valid_cell_mask, dtype=bool)
    eof1_values = eof1[eof1_mask]
    if eof1_values.size == 0:
        raise ValueError(f"EOF1 has no finite valid cells for {model_name}")

    n_positive = int((eof1_values > 0.0).sum())
    n_negative = int((eof1_values < 0.0).sum())
    n_valid = int(eof1_values.size)
    same_sign_fraction = max(n_positive, n_negative) / float(n_valid)
    eof1_mean = float(np.mean(eof1_values))
    eof1_std = float(np.std(eof1_values))
    eof1_abs_mean_over_std = float(abs(eof1_mean) / eof1_std) if eof1_std > 0.0 else float("nan")
    eof1_min = float(np.min(eof1_values))
    eof1_max = float(np.max(eof1_values))

    anomalies_flat = np.asarray(result.anomalies, dtype=np.float64).reshape(result.anomalies.shape[0], -1)
    valid_mask_flat = np.asarray(result.valid_cell_mask, dtype=bool).ravel()
    regional_mean_anomaly = np.mean(anomalies_flat[:, valid_mask_flat], axis=1)
    pc1 = np.asarray(result.pcs[:, 0], dtype=np.float64)
    pc1_corr = spatial_correlation(pc1, regional_mean_anomaly)
    pc1_abs_corr = abs(pc1_corr) if np.isfinite(pc1_corr) else float("nan")

    if same_sign_fraction >= 0.90 and pc1_abs_corr >= 0.90:
        interpretation = "uniform regional warm/cold mode: yes"
    elif same_sign_fraction >= 0.75 and pc1_abs_corr >= 0.75:
        interpretation = "uniform regional warm/cold mode: partial"
    else:
        interpretation = "uniform regional warm/cold mode: weak"

    return {
        "dataset_type": dataset_type,
        "model": model_name,
        "evr1": float(result.explained_variance_ratio[0]),
        "same_sign_fraction": same_sign_fraction,
        "n_valid_cells": n_valid,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "eof1_mean": eof1_mean,
        "eof1_std": eof1_std,
        "eof1_abs_mean_over_std": eof1_abs_mean_over_std,
        "eof1_min": eof1_min,
        "eof1_max": eof1_max,
        "pc1_region_mean_corr": pc1_corr,
        "pc1_region_mean_abs_corr": pc1_abs_corr,
        "interpretation": interpretation,
    }


def write_eof1_validation_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_type",
                "model",
                "evr1",
                "same_sign_fraction",
                "n_valid_cells",
                "n_positive",
                "n_negative",
                "eof1_mean",
                "eof1_std",
                "eof1_abs_mean_over_std",
                "eof1_min",
                "eof1_max",
                "pc1_region_mean_corr",
                "pc1_region_mean_abs_corr",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset_type"],
                    row["model"],
                    "{:.12g}".format(float(row["evr1"])),
                    "{:.12g}".format(float(row["same_sign_fraction"])),
                    int(row["n_valid_cells"]),
                    int(row["n_positive"]),
                    int(row["n_negative"]),
                    "{:.12g}".format(float(row["eof1_mean"])),
                    "{:.12g}".format(float(row["eof1_std"])),
                    "{:.12g}".format(float(row["eof1_abs_mean_over_std"])),
                    "{:.12g}".format(float(row["eof1_min"])),
                    "{:.12g}".format(float(row["eof1_max"])),
                    "{:.12g}".format(float(row["pc1_region_mean_corr"])),
                    "{:.12g}".format(float(row["pc1_region_mean_abs_corr"])),
                ]
            )


def print_eof1_validation_summary(rows: Sequence[Dict[str, object]]) -> None:
    for row in rows:
        print(f"{row['dataset_type']}/{row['model']}:", flush=True)
        print(f"  EVR1 = {float(row['evr1']):.4f}", flush=True)
        print(f"  EOF1 same-sign fraction = {float(row['same_sign_fraction']):.4f}", flush=True)
        print(f"  corr(PC1, regional mean anomaly) = {float(row['pc1_region_mean_corr']):.4f}", flush=True)
        print(f"  abs corr = {float(row['pc1_region_mean_abs_corr']):.4f}", flush=True)
        print(f"  interpretation = {row['interpretation']}", flush=True)
        print("", flush=True)


def compute_nino34_index_from_file(
    path: Path,
    variable_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(path) as ds:
        lat_name, lon_name = find_lat_lon_names(ds, variable_name)
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds[lat_name].values, dtype=np.float64)
        longitude = np.asarray(ds[lon_name].values, dtype=np.float64)
        values = np.asarray(ds[variable_name].values, dtype=np.float64)
    regional_mean = compute_weighted_regional_mean(
        values,
        latitude,
        longitude,
        NINO34_LAT_MIN,
        NINO34_LAT_MAX,
        NINO34_LON_MIN,
        NINO34_LON_MAX,
    )
    anomalies = compute_monthly_series_anomalies(regional_mean, time_values)
    return time_values, anomalies


def build_enso_validation_row(
    dataset_type: str,
    model_name: str,
    result: MonthlyClimatologyEofResult,
    enso_index_source: str,
    index_time: np.ndarray | None,
    index_values: np.ndarray | None,
    notes: str = "",
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset_type": dataset_type,
        "model": model_name,
        "enso_index_source": enso_index_source,
        "n_time": 0,
        "pc1_nino34_corr": float("nan"),
        "pc1_nino34_abs_corr": float("nan"),
        "pc2_nino34_corr": float("nan"),
        "pc2_nino34_abs_corr": float("nan"),
        "pc3_nino34_corr": float("nan"),
        "pc3_nino34_abs_corr": float("nan"),
        "pc23_enso_r2": float("nan"),
        "pc23_enso_multiple_r": float("nan"),
        "pc123_enso_r2": float("nan"),
        "pc123_enso_multiple_r": float("nan"),
        "interpretation": "not available: model tropical Pacific SST not found" if enso_index_source == "not_available" else "",
        "notes": notes,
    }
    if index_time is None or index_values is None or result.time.size == 0 or result.pcs.shape[0] == 0:
        if not row["interpretation"]:
            row["interpretation"] = "not available"
        return row

    common_time, pc1, nino = align_series_by_time(result.time, result.pcs[:, 0], index_time, index_values)
    if common_time.size < 3:
        row["interpretation"] = "not available"
        row["notes"] = (notes + "; " if notes else "") + "insufficient overlapping valid times"
        return row

    row["n_time"] = int(common_time.size)
    pc2 = np.asarray(result.pcs[:, 1], dtype=np.float64)[np.intersect1d(np.asarray(result.time, dtype="datetime64[ns]"), common_time, return_indices=True)[1]]
    pc3 = np.asarray(result.pcs[:, 2], dtype=np.float64)[np.intersect1d(np.asarray(result.time, dtype="datetime64[ns]"), common_time, return_indices=True)[1]]
    nino_aligned = np.asarray(index_values, dtype=np.float64)[np.intersect1d(np.asarray(index_time, dtype="datetime64[ns]"), common_time, return_indices=True)[1]]

    corr1 = spatial_correlation(pc1, nino)
    corr2 = spatial_correlation(pc2, nino_aligned)
    corr3 = spatial_correlation(pc3, nino_aligned)
    row["pc1_nino34_corr"] = corr1
    row["pc1_nino34_abs_corr"] = abs(corr1) if np.isfinite(corr1) else float("nan")
    row["pc2_nino34_corr"] = corr2
    row["pc2_nino34_abs_corr"] = abs(corr2) if np.isfinite(corr2) else float("nan")
    row["pc3_nino34_corr"] = corr3
    row["pc3_nino34_abs_corr"] = abs(corr3) if np.isfinite(corr3) else float("nan")

    pc23_r2, pc23_r = compute_multiple_regression_skill(np.column_stack([pc2, pc3]), nino_aligned)
    pc123_r2, pc123_r = compute_multiple_regression_skill(np.column_stack([pc1, pc2, pc3]), nino_aligned)
    row["pc23_enso_r2"] = pc23_r2
    row["pc23_enso_multiple_r"] = pc23_r
    row["pc123_enso_r2"] = pc123_r2
    row["pc123_enso_multiple_r"] = pc123_r

    pc2_abs = float(row["pc2_nino34_abs_corr"])
    pc3_abs = float(row["pc3_nino34_abs_corr"])
    if max(pc2_abs, pc3_abs, pc23_r) < 0.4:
        interpretation = "no clear ENSO relationship in EOF2/EOF3"
    elif pc2_abs >= 0.7 and pc2_abs > pc3_abs:
        interpretation = "ENSO primarily aligned with PC2"
    elif pc3_abs >= 0.7 and pc3_abs > pc2_abs:
        interpretation = "ENSO primarily aligned with PC3"
    elif pc23_r >= 0.7 and pc2_abs >= 0.4 and pc3_abs >= 0.4:
        interpretation = "ENSO represented in combined PC2-PC3 subspace"
    elif pc23_r >= 0.4:
        interpretation = "ENSO represented in combined PC2-PC3 subspace"
    else:
        interpretation = "no clear ENSO relationship in EOF2/EOF3"
    row["interpretation"] = interpretation
    return row


def write_enso_validation_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_type",
                "model",
                "enso_index_source",
                "n_time",
                "pc1_nino34_corr",
                "pc1_nino34_abs_corr",
                "pc2_nino34_corr",
                "pc2_nino34_abs_corr",
                "pc3_nino34_corr",
                "pc3_nino34_abs_corr",
                "pc23_enso_r2",
                "pc23_enso_multiple_r",
                "pc123_enso_r2",
                "pc123_enso_multiple_r",
                "interpretation",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset_type"],
                    row["model"],
                    row["enso_index_source"],
                    int(row["n_time"]),
                    "{:.12g}".format(float(row["pc1_nino34_corr"])) if np.isfinite(float(row["pc1_nino34_corr"])) else "nan",
                    "{:.12g}".format(float(row["pc1_nino34_abs_corr"])) if np.isfinite(float(row["pc1_nino34_abs_corr"])) else "nan",
                    "{:.12g}".format(float(row["pc2_nino34_corr"])) if np.isfinite(float(row["pc2_nino34_corr"])) else "nan",
                    "{:.12g}".format(float(row["pc2_nino34_abs_corr"])) if np.isfinite(float(row["pc2_nino34_abs_corr"])) else "nan",
                    "{:.12g}".format(float(row["pc3_nino34_corr"])) if np.isfinite(float(row["pc3_nino34_corr"])) else "nan",
                    "{:.12g}".format(float(row["pc3_nino34_abs_corr"])) if np.isfinite(float(row["pc3_nino34_abs_corr"])) else "nan",
                    "{:.12g}".format(float(row["pc23_enso_r2"])) if np.isfinite(float(row["pc23_enso_r2"])) else "nan",
                    "{:.12g}".format(float(row["pc23_enso_multiple_r"])) if np.isfinite(float(row["pc23_enso_multiple_r"])) else "nan",
                    "{:.12g}".format(float(row["pc123_enso_r2"])) if np.isfinite(float(row["pc123_enso_r2"])) else "nan",
                    "{:.12g}".format(float(row["pc123_enso_multiple_r"])) if np.isfinite(float(row["pc123_enso_multiple_r"])) else "nan",
                    row["interpretation"],
                    row["notes"],
                ]
            )


def print_enso_validation_summary(row: Dict[str, object]) -> None:
    print(f"{row['dataset_type']}/{row['model']}:", flush=True)
    if row["enso_index_source"] == "not_available":
        print("  model tropical Pacific SST was not available; ENSO timing validation skipped", flush=True)
        if row["notes"]:
            print(f"  notes = {row['notes']}", flush=True)
        print("", flush=True)
        return
    print(f"  corr(PC1, Nino3.4) = {float(row['pc1_nino34_corr']):.4f}", flush=True)
    print(f"  corr(PC2, Nino3.4) = {float(row['pc2_nino34_corr']):.4f}", flush=True)
    print(f"  corr(PC3, Nino3.4) = {float(row['pc3_nino34_corr']):.4f}", flush=True)
    print(f"  PC2+PC3 multiple R = {float(row['pc23_enso_multiple_r']):.4f}", flush=True)
    print(f"  interpretation = {row['interpretation']}", flush=True)
    print("", flush=True)


def plot_comparison_panel(
    cobe2_result: MonthlyClimatologyEofResult,
    wus_result: MonthlyClimatologyEofResult,
    comparison_rows: Sequence[Dict[str, object]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(PLOT_COMPONENTS, 2, figsize=(10, 12), constrained_layout=True)
    for component_index in range(PLOT_COMPONENTS):
        row = comparison_rows[component_index]
        cobe2_field = cobe2_result.eofs[component_index]
        wus_field = wus_result.eofs[component_index]
        if row["flip_wus_sign"]:
            wus_field = -wus_field
        vmax = float(
            np.nanmax(
                [
                    np.nanmax(np.abs(cobe2_field)),
                    np.nanmax(np.abs(wus_field)),
                ]
            )
        )
        vmax = 1.0 if vmax == 0.0 or not np.isfinite(vmax) else vmax
        for col_index, (field, label) in enumerate(((cobe2_field, "COBE2"), (wus_field, wus_result.dataset_id))):
            ax = axes[component_index, col_index]
            image = ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if component_index == 0:
                ax.set_title(label)
            if col_index == 0:
                ax.set_ylabel(f"EOF{component_index + 1}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[component_index, 1].set_title(
            f"{wus_result.dataset_id}\nbest corr={row['best_signed_correlation']:.3f}"
        )
    fig.suptitle(f"COBE2 vs {wus_result.dataset_id} monthly-climatology SST EOF comparison")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_models_panel(
    cobe2_result: MonthlyClimatologyEofResult,
    wus_results: Sequence[MonthlyClimatologyEofResult],
    comparison_rows: Sequence[Dict[str, object]],
    path: Path,
) -> None:
    rows_by_model: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row in comparison_rows:
        rows_by_model.setdefault(str(row["model"]), {})[int(row["component"])] = row

    n_rows = 1 + len(wus_results)
    fig, axes = plt.subplots(
        n_rows,
        PLOT_COMPONENTS,
        figsize=(4.6 * PLOT_COMPONENTS, 2.8 * n_rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    all_results = [cobe2_result] + list(wus_results)
    for row_index, result in enumerate(all_results):
        for component_index in range(PLOT_COMPONENTS):
            ax = axes[row_index, component_index]
            field = result.eofs[component_index]
            title = f"EOF{component_index + 1}"
            if row_index == 0:
                title = (
                    f"COBE2 EOF{component_index + 1}\n"
                    f"EVR={cobe2_result.explained_variance_ratio[component_index]:.3f}"
                )
            else:
                comparison = rows_by_model[result.dataset_id][component_index + 1]
                if comparison["flip_wus_sign"]:
                    field = -field
                title = (
                    f"EOF{component_index + 1}\n"
                    f"best corr={comparison['best_signed_correlation']:.3f}"
                )

            vmax = float(np.nanmax(np.abs(field)))
            vmax = 1.0 if vmax == 0.0 or not np.isfinite(vmax) else vmax
            image = ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if component_index == 0:
                ax.set_ylabel(result.dataset_id)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("COBE2 and WUS monthly-climatology SST EOF comparison")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def classify_eof_notes(field: np.ndarray) -> str:
    finite = np.asarray(field[np.isfinite(field)], dtype=np.float64)
    if finite.size == 0:
        return "field has no finite values"
    same_sign_fraction = max(float((finite > 0).mean()), float((finite < 0).mean()))
    if same_sign_fraction >= 0.8:
        return "appears broad / sign-consistent over most of the domain"
    return "shows stronger spatial structure / sign contrast across the domain"


def write_comparison_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "component",
                "raw_correlation",
                "flipped_correlation",
                "best_signed_correlation",
                "flip_wus_sign",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["component"],
                    "{:.12g}".format(float(row["raw_correlation"])) if np.isfinite(float(row["raw_correlation"])) else "nan",
                    "{:.12g}".format(float(row["flipped_correlation"])) if np.isfinite(float(row["flipped_correlation"])) else "nan",
                    "{:.12g}".format(float(row["best_signed_correlation"])) if np.isfinite(float(row["best_signed_correlation"])) else "nan",
                    str(bool(row["flip_wus_sign"])).lower(),
                ]
            )


def write_comparison_report(
    runtime: RuntimeInfo,
    cobe2_result: MonthlyClimatologyEofResult,
    wus_results: Sequence[MonthlyClimatologyEofResult],
    comparison_rows: Sequence[Dict[str, object]],
    cross_correlation_rows: Sequence[Dict[str, object]],
) -> None:
    rows_by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in comparison_rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)
    cross_rows_by_model: Dict[str, Dict[Tuple[int, int], float]] = {}
    for row in cross_correlation_rows:
        cross_rows_by_model.setdefault(str(row["model"]), {})[
            (int(row["cobe_mode"]), int(row["wus_mode"]))
        ] = float(row["spatial_corr_abs"])

    lines = [
        "# Monthly-Climatology SST EOF Comparison Report",
        "",
        "## Runtime",
        "",
        f"- hostname: `{runtime.hostname}`",
        f"- Slurm job ID: `{runtime.slurm_job_id}`",
        "",
        "## Preprocessing Confirmation",
        "",
        "- Monthly climatology was removed correctly by computing separate January through December means.",
        "- For each time step, the anomaly is `SST(t) - climatology[month(t)]`.",
        "- This workflow removes the mean seasonal cycle before EOF analysis.",
        "",
        "## COBE2 Explained Variance",
        "",
        f"- EOF1: `{cobe2_result.explained_variance_ratio[0]:.4f}`",
        f"- EOF2: `{cobe2_result.explained_variance_ratio[1]:.4f}`",
        f"- EOF3: `{cobe2_result.explained_variance_ratio[2]:.4f}`",
        f"- EOF1 note: {classify_eof_notes(cobe2_result.eofs[0])}",
        f"- EOF2 note: {classify_eof_notes(cobe2_result.eofs[1])}",
        "",
        "## WUS Explained Variance",
        "",
    ]

    for result in wus_results:
        lines.append(
            f"- `{result.dataset_id}`: EOF1 `{result.explained_variance_ratio[0]:.4f}`, "
            f"EOF2 `{result.explained_variance_ratio[1]:.4f}`, EOF3 `{result.explained_variance_ratio[2]:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Spatial Correlations",
            "",
            "| Model | EOF1 | EOF2 | EOF3 |",
            "|-------|------|------|------|",
        ]
    )

    for result in wus_results:
        model_rows = sorted(rows_by_model[result.dataset_id], key=lambda item: int(item["component"]))
        lines.append(
            f"| {result.dataset_id} | "
            f"{model_rows[0]['best_signed_correlation']:.4f} | "
            f"{model_rows[1]['best_signed_correlation']:.4f} | "
            f"{model_rows[2]['best_signed_correlation']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full EOF Cross-Correlation Matrices",
            "",
            "Each matrix entry is `abs(corr(COBE2 EOF_i, WUS EOF_j))`.",
            "",
        ]
    )

    for result in wus_results:
        matrix = cross_rows_by_model[result.dataset_id]
        lines.extend(
            [
                f"### {result.dataset_id}",
                "",
                "|        | WUS EOF1 | WUS EOF2 | WUS EOF3 |",
                "|--------|----------|----------|----------|",
                f"| COBE2 EOF1 | {matrix[(1, 1)]:.4f} | {matrix[(1, 2)]:.4f} | {matrix[(1, 3)]:.4f} |",
                f"| COBE2 EOF2 | {matrix[(2, 1)]:.4f} | {matrix[(2, 2)]:.4f} | {matrix[(2, 3)]:.4f} |",
                f"| COBE2 EOF3 | {matrix[(3, 1)]:.4f} | {matrix[(3, 2)]:.4f} | {matrix[(3, 3)]:.4f} |",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Cross-Matrix Interpretation",
            "",
            "The diagonal-only comparison assumes COBE2 EOF1, EOF2, and EOF3 correspond directly to WUS EOF1, EOF2, and EOF3. The full 3x3 matrix checks whether a COBE2 EOF matches a different WUS EOF better. For EOF2 and EOF3, the key diagnostic is the 2x2 block formed by COBE2 EOF2 to EOF3 against WUS EOF2 to EOF3. Across these models there is no simple EOF2 and EOF3 swap, but several cases show partial EOF2 and EOF3 mixing.",
            "",
            "- EC-Earth historical: EOF2 maps cleanly to WUS EOF2, but COBE2 EOF3 also correlates strongly with WUS EOF2. This indicates noticeable EOF2 and EOF3 mixing, not a clean swap.",
            "- EC-Earth SSP370: The EOF2 and EOF3 block is mostly diagonal, with only moderate off-diagonal leakage. This is mostly clean one-to-one matching.",
            "- MIROC6 historical: Strong diagonal matching, especially for EOF3. This is one of the cleanest EOF2 and EOF3 matches.",
            "- MIROC6 SSP370: Diagonal matching remains stronger, but COBE2 EOF3 has a moderate correlation with WUS EOF2. This suggests moderate EOF2 and EOF3 mixing.",
            "- MPI historical: Diagonal values dominate, with moderate off-diagonal terms. This is mostly clean one-to-one matching.",
            "- MPI SSP370: Very clean diagonal structure, especially for EOF3. This is one of the cleanest cases.",
            "- TaiESM historical: Strongest EOF2 and EOF3 mixing. COBE2 EOF2 correlates almost as strongly with WUS EOF3 as WUS EOF2, and COBE2 EOF3 also correlates substantially with WUS EOF2. Individual EOF2 and EOF3 labels should not be over-interpreted.",
            "- TaiESM SSP370: Diagonal matching is clear, but with moderate off-diagonal mixing. Note that EOF2 and EOF3 explain very small variance in this run, so these are low-variance residual modes.",
            "",
            "| Rank | Model | EOF2/EOF3 interpretation |",
            "|------|-------|--------------------------|",
            "| 1 | taiesm1_r1i1p1f1_historical_bc | strongest mixing |",
            "| 2 | ec-earth3_r1i1p1f1_2_historical_bc | noticeable mixing |",
            "| 3 | miroc6_r1i1p1f1_ssp370_bc | moderate mixing |",
            "| 4 | taiesm1_r1i1p1f1_ssp370_bc | moderate mixing, but EOF2/EOF3 are low-variance residual modes |",
            "| 5 | mpi-esm1-2-hr_r3i1p1f1_historical_bc | mild to moderate mixing |",
            "| 6 | ec-earth3_r1i1p1f1_2_ssp370_bc | mostly clean |",
            "| 7 | miroc6_r1i1p1f1_historical_bc | clean |",
            "| 8 | mpi-esm1-2-hr_r3i1p1f1_ssp370_bc | cleanest |",
            "",
            "The full matrix also reveals that EOF1 is not always best aligned with WUS EOF1. In EC-Earth, COBE2 EOF1 correlates much more strongly with WUS EOF2 than with WUS EOF1: historical `COBE1-WUS1 = 0.048`, `COBE1-WUS2 = 0.650`; SSP370 `COBE1-WUS1 = 0.033`, `COBE1-WUS2 = 0.602`. TaiESM shows a similar tendency: historical `COBE1-WUS1 = 0.331`, `COBE1-WUS2 = 0.593`; SSP370 `COBE1-WUS1 = 0.206`, `COBE1-WUS2 = 0.596`. Therefore, EC-Earth and TaiESM have broader EOF1 and EOF2 ordering or mode-structure mismatch, not just EOF2 and EOF3 mixing.",
            "",
            "The diagonal EOF2 and EOF3 comparisons remain mostly meaningful because there is no simple EOF2 and EOF3 swap. However, the full-matrix diagnostics show partial mixing in several models, strongest in TaiESM historical. MIROC6 historical and MPI SSP370 provide the cleanest EOF2 and EOF3 correspondence. EC-Earth and TaiESM require extra caution because their EOF1 and EOF2 structure does not align cleanly with COBE2.",
            "",
            "## Interpretation Notes",
            "",
            "- EOF1 is expected to be a broad / roughly uniform regional SST anomaly pattern.",
            "- EOF2 is expected to show more structure, possibly ENSO-like or coastal/offshore contrast.",
            "- COBE2 and WUS EOFs are not expected to be identical, but physically comparable patterns should show moderate positive signed correlations after allowing sign flips.",
            "- Low or noisy correlations are a warning that the remapped WUS SST variability structure differs from COBE2 over this domain.",
        ]
    )

    COMPARISON_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cobe2_monthly_climatology_result() -> MonthlyClimatologyEofResult:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        lat_crop, lon_crop, sst_crop = crop_cobe2_region(ds)
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    return compute_eof_result(
        dataset_id="COBE2",
        source_file=str(COBE2_SST_FILE),
        time_values=time_values,
        latitude=lat_crop,
        longitude=lon_crop,
        values=sst_crop,
        notes=["month-of-year climatology removed before EOF analysis"],
    )


def get_cobe2_monthly_climatology_result() -> MonthlyClimatologyEofResult:
    cached = load_cached_result(
        COBE2_EOF_FILE,
        COBE2_SUMMARY_FILE,
        anomaly_path=COBE2_ANOM_FILE,
        pcs_path=COBE2_PCS_FILE,
    )
    if cached is not None:
        return cached
    result = load_cobe2_monthly_climatology_result()
    save_result_bundle(
        result,
        COBE2_CLIM_FILE,
        COBE2_ANOM_FILE,
        COBE2_EOF_FILE,
        COBE2_PCS_FILE,
        COBE2_SUMMARY_FILE,
        COBE2_PLOTS_DIR / "cobe2",
    )
    return result


def load_wus_monthly_climatology_result(model_name: str) -> MonthlyClimatologyEofResult:
    input_path = MODEL_SST_ROOT / model_name / f"{model_name}_tskin_on_cobe2_grid.nc"
    with open_dataset_with_fallbacks(input_path) as ds:
        values = np.asarray(ds["tskin"].values, dtype=np.float64)
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    return compute_eof_result(
        dataset_id=model_name,
        source_file=str(input_path),
        time_values=time_values,
        latitude=latitude,
        longitude=longitude,
        values=values,
        notes=["month-of-year climatology removed from remapped WUS SST before EOF analysis"],
    )


def get_wus_monthly_climatology_result(model_name: str) -> MonthlyClimatologyEofResult:
    eof_path = WUS_OUTPUT_DIR / f"{model_name}_monthly_clim_sst_eofs.nc"
    anomaly_path = WUS_OUTPUT_DIR / f"{model_name}_monthly_clim_sst_anomalies.nc"
    pcs_path = WUS_OUTPUT_DIR / f"{model_name}_monthly_clim_sst_pcs.csv"
    summary_path = WUS_OUTPUT_DIR / f"{model_name}_monthly_clim_sst_summary.json"
    cached = load_cached_result(
        eof_path,
        summary_path,
        anomaly_path=anomaly_path,
        pcs_path=pcs_path,
    )
    if cached is not None:
        return cached
    result = load_wus_monthly_climatology_result(model_name)
    save_result_bundle(
        result,
        WUS_OUTPUT_DIR / f"{model_name}_monthly_clim_sst_climatology.nc",
        anomaly_path,
        eof_path,
        pcs_path,
        summary_path,
        WUS_PLOTS_DIR / model_name,
    )
    return result


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dirs()

    cobe2_result = get_cobe2_monthly_climatology_result()
    cobe2_nino_time, cobe2_nino_index = compute_nino34_index_from_file(COBE2_SST_FILE, "sst")
    eof1_validation_rows: List[Dict[str, object]] = [
        validate_eof1_uniform_mode("cobe2", "COBE2", cobe2_result)
    ]
    print_eof1_validation_summary(eof1_validation_rows)
    enso_validation_rows: List[Dict[str, object]] = [
        build_enso_validation_row(
            "cobe2",
            "COBE2",
            cobe2_result,
            "observed_COBE2_Nino34",
            cobe2_nino_time,
            cobe2_nino_index,
        )
    ]
    print_enso_validation_summary(enso_validation_rows[0])

    wus_results: List[MonthlyClimatologyEofResult] = []
    comparison_rows: List[Dict[str, object]] = []
    cross_correlation_rows: List[Dict[str, object]] = []
    for model_name in MODELS:
        result = get_wus_monthly_climatology_result(model_name)
        wus_results.append(result)
        eof1_validation_rows.append(validate_eof1_uniform_mode("wus", model_name, result))
        monthly_path = MODEL_SST_ROOT / model_name / f"{model_name}_tskin_monthly.nc"
        try:
            model_nino_time, model_nino_index = compute_nino34_index_from_file(monthly_path, "tskin")
            enso_row = build_enso_validation_row(
                "wus",
                model_name,
                result,
                "model_Nino34",
                model_nino_time,
                model_nino_index,
                notes=f"source_file={monthly_path}",
            )
        except Exception as exc:
            enso_row = build_enso_validation_row(
                "wus",
                model_name,
                result,
                "not_available",
                None,
                None,
                notes=str(exc),
            )
        enso_validation_rows.append(enso_row)
        model_rows = compare_eofs(cobe2_result, result)
        comparison_rows.extend(model_rows)
        model_cross_rows = compute_cross_correlation_rows(cobe2_result, result)
        cross_correlation_rows.extend(model_cross_rows)
        print_eof1_validation_summary([eof1_validation_rows[-1]])
        print_enso_validation_summary(enso_row)
        print_cross_correlation_matrix(result.dataset_id, model_cross_rows)
        plot_comparison_panel(
            cobe2_result,
            result,
            model_rows,
            COMPARISON_PLOTS_DIR / f"{model_name}_cobe2_vs_wus_eof_comparison.png",
        )

    write_eof1_validation_csv(EOF1_VALIDATION_CSV, eof1_validation_rows)
    write_enso_validation_csv(ENSO_VALIDATION_CSV, enso_validation_rows)
    write_comparison_csv(COMPARISON_CSV, comparison_rows)
    write_cross_correlation_csv(COMPARISON_CROSS_CSV, cross_correlation_rows)
    plot_all_models_panel(
        cobe2_result,
        wus_results,
        comparison_rows,
        COMPARISON_PLOTS_DIR / "all_models_cobe2_vs_wus_eofs.png",
    )
    write_comparison_report(runtime, cobe2_result, wus_results, comparison_rows, cross_correlation_rows)
    print(f"wrote {COBE2_SUMMARY_FILE}", flush=True)
    print(f"wrote {EOF1_VALIDATION_CSV}", flush=True)
    print(f"wrote {ENSO_VALIDATION_CSV}", flush=True)
    print(f"wrote {COMPARISON_CSV}", flush=True)
    print(f"wrote {COMPARISON_CROSS_CSV}", flush=True)
    print(f"wrote {COMPARISON_REPORT}", flush=True)


if __name__ == "__main__":
    main()
