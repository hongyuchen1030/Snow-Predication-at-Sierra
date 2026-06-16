#!/usr/bin/env python3
"""
Build a shared-region COBE2 EOF basis tied to one WUS dataset/domain and run
matched observational and WUS SST-to-T2m diagnostics from that basis.

The workflow is:
1. Define the shared SST mask M on the COBE2 grid from
   COBE2 valid ocean cells intersect WUS SST cells that are finite at all
   overlap months.
2. Recompute COBE2 EOFs/PCs directly from COBE2 monthly SST anomalies
   restricted to M and the WUS overlap period.
3. Run shared-region observational Level 1, Level 2, and monthly Level 2
   diagnostics:
      shared-region COBE2 PCs -> ERA5 Sierra T2m anomalies
4. Project WUS monthly SST anomalies onto the new shared-region COBE2 EOFs.
5. Run matched WUS Level 1, Level 2, and monthly Level 2 diagnostics:
      WUS projected scores -> WUSD-03 Sierra T2m anomalies

No new EOFs are computed from WUS SST.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
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
N_MODES = 6
SIERRA_REGION_360 = RegionBounds(lat_min=35.0, lat_max=42.0, lon_min=236.0, lon_max=243.0)
COBE2_GLOBAL_EOF_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "sst_pca"
    / "cobe2_global_monthly_climatology_anomaly"
    / "cobe2_global_monthly_clim_sst_eofs.nc"
)
WUS_SST_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_sst_on_cobe2_grid_monthly")
WUS_T2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_t2_monthly_anomalies")

SHARED_COBE2_EOF_ROOT = PROJECT_ROOT / "artifacts" / "cobe2_sharedregion_sst_pca"
OBS_LEVEL1_ROOT = PROJECT_ROOT / "artifacts" / "cobe2_sharedregion_sierra_t2m_level1_diagnostic"
OBS_LEVEL2_ROOT = PROJECT_ROOT / "artifacts" / "cobe2_sharedregion_sierra_t2m_level2_pc1to6"
OBS_MONTHLY_ROOT = PROJECT_ROOT / "artifacts" / "cobe2_sharedregion_sierra_t2m_level2_pc1to6_seasonal_sensitivity"
WUS_PROJ_ROOT = PROJECT_ROOT / "artifacts" / "wus_sst_projected_onto_cobe2_sharedregion_eofs"
WUS_LEVEL1_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_sharedregion_pc_t2m_level1_ols"
WUS_LEVEL2_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_sharedregion_pc_t2m_level2_ols"
WUS_MONTHLY_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_sharedregion_pc_t2m_level2_pc1to6_seasonal_sensitivity"

MONTHLY_SUBSETS = [
    ("Jan", 1),
    ("Feb", 2),
    ("Mar", 3),
    ("Apr", 4),
    ("May", 5),
    ("Jun", 6),
    ("Jul", 7),
    ("Aug", 8),
    ("Sep", 9),
    ("Oct", 10),
    ("Nov", 11),
    ("Dec", 12),
]


@dataclass(frozen=True)
class SharedRegionReference:
    dataset_id: str
    domain: str
    overlap_months: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    shared_mask: np.ndarray
    weighting_note: str
    cobe2_eof_unweighted: np.ndarray
    cobe2_pc_raw: np.ndarray
    cobe2_singular_values: np.ndarray
    cobe2_explained_variance_ratio: np.ndarray
    weighted_gram: np.ndarray
    weighted_mask_cell_count: int
    cobe2_score_mean_raw: np.ndarray
    cobe2_score_std_raw: np.ndarray


@dataclass(frozen=True)
class Level1SummaryRow:
    mode: int
    explained_variance_ratio: float
    pc_mean_raw: float
    pc_std_raw: float
    area_weighted_mean_r2: float
    area_weighted_mean_corr: float
    max_local_r2: float


@dataclass(frozen=True)
class Level2Summary:
    mean_r2_joint: float
    max_r2_joint: float
    min_r2_joint: float
    median_r2_joint: float


@dataclass(frozen=True)
class MonthlySummaryRow:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared-region COBE2/WUS Sierra T2m diagnostics.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="WUS dataset id to use.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="WUS domain, default d03.")
    return parser.parse_args()


def dataset_output_dir(root: Path, domain: str, dataset_id: str) -> Path:
    return root / domain / dataset_id


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_file_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def normalize_lon_to_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(lon, dtype=np.float64), 360.0)


def normalize_lon_to_minus180_180(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=np.float64)
    return np.where(values > 180.0, values - 360.0, values)


def standardize_pc_matrix(pc_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(pc_matrix, dtype=np.float64)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
        raise ValueError("Cannot standardize PCs with non-finite or zero std")
    standardized = (values - means[np.newaxis, :]) / stds[np.newaxis, :]
    return standardized, means, stds


def intersect_months(*time_axes: Sequence[np.datetime64]) -> np.ndarray:
    common = to_month_start(time_axes[0])
    for axis in time_axes[1:]:
        common = np.intersect1d(common, to_month_start(axis), assume_unique=False)
    return np.asarray(common, dtype="datetime64[ns]")


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    month_values = to_month_start(time_values)
    index_by_month = {month: idx for idx, month in enumerate(month_values.tolist())}
    return np.asarray(data)[[index_by_month[month] for month in target_months.tolist()]]


def month_subset_index(overlap_months: np.ndarray, month_number_value: int) -> np.ndarray:
    months = np.asarray([month_number(value) for value in overlap_months.tolist()], dtype=np.int32)
    return months == int(month_number_value)


def area_weighted_mean(field: np.ndarray, latitude_2d: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    weights = np.cos(np.deg2rad(np.asarray(latitude_2d, dtype=np.float64)))
    valid = np.isfinite(values)
    weighted_sum = np.nansum(np.where(valid, values * weights, 0.0))
    weight_sum = np.nansum(np.where(valid, weights, 0.0))
    if not np.isfinite(weight_sum) or weight_sum == 0.0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def masked_area_weighted_mean(r2_map: np.ndarray, weights_2d: np.ndarray, mask: np.ndarray) -> float:
    valid_weights = np.where(mask & np.isfinite(r2_map), weights_2d, 0.0)
    weight_sum = float(np.sum(valid_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(np.sum(np.where(mask, r2_map, 0.0) * valid_weights) / weight_sum)


def area_weights_from_latitude(latitude: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    if lat.ndim == 1:
        return np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], shape)
    return np.cos(np.deg2rad(lat))


def load_cobe2_valid_mask() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with open_dataset_with_fallbacks(COBE2_GLOBAL_EOF_FILE) as ds:
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        valid_mask = np.asarray(ds["valid_mask"].values, dtype=bool)
        weighting_note = str(ds.attrs.get("latitude_weighting", "sqrt(cos(lat)) weighting"))
    return latitude, longitude, valid_mask, weighting_note


def load_wus_sst_monthly_anomaly(dataset_id: str, domain: str) -> Tuple[np.ndarray, np.ndarray]:
    path = WUS_SST_ROOT / domain / dataset_id / f"{dataset_id}_{domain}_tskin_on_cobe2_grid_monthly_anomaly.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing WUS SST monthly anomaly file: {path}")
    with open_dataset_with_fallbacks(path) as ds:
        if "tskin_anomaly" in ds.data_vars:
            var_name = "tskin_anomaly"
        elif "tskin" in ds.data_vars:
            var_name = "tskin"
        else:
            raise ValueError(f"Expected SST anomaly variable in {path}, found {list(ds.data_vars)}")
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        values = np.asarray(ds[var_name].values, dtype=np.float64)
    return to_month_start(time), values


def build_shared_mask(dataset_id: str, domain: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    latitude, longitude, cobe2_valid_mask, weighting_note = load_cobe2_valid_mask()
    wus_time, wus_anom = load_wus_sst_monthly_anomaly(dataset_id, domain)
    shared_mask = cobe2_valid_mask & np.isfinite(wus_anom).all(axis=0)
    if int(np.count_nonzero(shared_mask)) < N_MODES:
        raise ValueError("Shared mask has too few valid ocean cells for a 6-mode EOF solve")
    return latitude, longitude, wus_time, shared_mask, weighting_note


def load_cobe2_overlap_anomalies(
    overlap_months: np.ndarray,
    latitude_target: np.ndarray,
    longitude_target: np.ndarray,
) -> np.ndarray:
    print(f"Loading COBE2 SST from {COBE2_SST_FILE}", flush=True)
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        sst = np.asarray(ds["sst"].values, dtype=np.float64)
        time = to_month_start(np.asarray(ds["time"].values, dtype="datetime64[ns]"))
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = normalize_lon_to_minus180_180(np.asarray(ds["lon"].values, dtype=np.float64))
        sort_idx = np.argsort(lon)
        lon = lon[sort_idx]
        sst = sst[:, :, sort_idx]
        missing_value = float(ds["sst"].attrs.get("missing_value", 1.0e20))
        sst = np.where(sst >= missing_value, np.nan, sst)
    if not np.allclose(lat, latitude_target):
        raise ValueError("COBE2 latitude grid does not match expected shared-mask latitude grid")
    if not np.allclose(lon, longitude_target):
        raise ValueError("COBE2 longitude grid does not match expected shared-mask longitude grid")
    use_index = np.isin(time, overlap_months)
    overlap_values = sst[use_index]
    overlap_time = time[use_index]
    if overlap_time.size != overlap_months.size or not np.array_equal(overlap_time, overlap_months):
        raise ValueError("COBE2 overlap months do not align with requested shared overlap period")
    print(f"Computing COBE2 month-of-year climatology anomalies over {int(overlap_time.size)} overlap months", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        _, anomalies = compute_monthly_climatology_anomalies(overlap_values, overlap_time)
    return anomalies


def solve_shared_region_cobe2_eofs(
    dataset_id: str,
    domain: str,
    overlap_months: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    shared_mask: np.ndarray,
    weighting_note: str,
) -> SharedRegionReference:
    anomalies = load_cobe2_overlap_anomalies(overlap_months, latitude, longitude)
    anomalies_flat = anomalies[:, shared_mask]
    if anomalies_flat.shape[1] < N_MODES:
        raise ValueError("Shared-mask anomaly matrix has too few spatial cells for 6 EOF modes")
    lat_weights_1d = np.sqrt(np.clip(np.cos(np.deg2rad(latitude)), 0.0, None))
    weights_2d = np.broadcast_to(lat_weights_1d[:, np.newaxis], shared_mask.shape)
    weights_flat = weights_2d[shared_mask]
    weighted_matrix = anomalies_flat * weights_flat[np.newaxis, :]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(weighted_matrix, full_matrices=False)
    variance = singular_values ** 2
    explained_variance_ratio = variance / variance.sum()
    pcs = (u_matrix[:, :N_MODES] * singular_values[:N_MODES]).astype(np.float64)
    weighted_eofs_valid = vt_matrix[:N_MODES].astype(np.float64)
    unweighted_eofs_valid = weighted_eofs_valid / weights_flat[np.newaxis, :]
    eof_grid = np.full((N_MODES, latitude.size, longitude.size), np.nan, dtype=np.float64)
    eof_grid.reshape(N_MODES, -1)[:, shared_mask.reshape(-1)] = unweighted_eofs_valid
    weighted_gram = weighted_eofs_valid @ weighted_eofs_valid.T
    _, raw_mean, raw_std = standardize_pc_matrix(pcs)
    return SharedRegionReference(
        dataset_id=dataset_id,
        domain=domain,
        overlap_months=overlap_months,
        latitude=latitude,
        longitude=longitude,
        shared_mask=shared_mask,
        weighting_note=weighting_note,
        cobe2_eof_unweighted=eof_grid,
        cobe2_pc_raw=pcs,
        cobe2_singular_values=singular_values[:N_MODES].astype(np.float64),
        cobe2_explained_variance_ratio=explained_variance_ratio[:N_MODES].astype(np.float64),
        weighted_gram=weighted_gram.astype(np.float64),
        weighted_mask_cell_count=int(np.count_nonzero(shared_mask)),
        cobe2_score_mean_raw=raw_mean.astype(np.float64),
        cobe2_score_std_raw=raw_std.astype(np.float64),
    )


def save_shared_region_eof_outputs(reference: SharedRegionReference) -> None:
    out_dir = dataset_output_dir(SHARED_COBE2_EOF_ROOT, reference.domain, reference.dataset_id)
    ensure_dir(out_dir)
    ds = xr.Dataset(
        data_vars={
            "eof": (("mode", "lat", "lon"), reference.cobe2_eof_unweighted.astype(np.float32)),
            "pc": (("time", "mode"), reference.cobe2_pc_raw.astype(np.float32)),
            "singular_value": (("mode",), reference.cobe2_singular_values.astype(np.float32)),
            "explained_variance_ratio": (("mode",), reference.cobe2_explained_variance_ratio.astype(np.float32)),
            "valid_mask": (("lat", "lon"), reference.shared_mask.astype(np.int8)),
            "weighted_gram_matrix": (("mode", "mode_2"), reference.weighted_gram.astype(np.float32)),
            "pc_mean_raw": (("mode",), reference.cobe2_score_mean_raw.astype(np.float32)),
            "pc_std_raw": (("mode",), reference.cobe2_score_std_raw.astype(np.float32)),
        },
        coords={
            "time": reference.overlap_months.astype("datetime64[ns]"),
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "mode_2": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat": reference.latitude.astype(np.float32),
            "lon": reference.longitude.astype(np.float32),
        },
        attrs={
            "dataset_id": reference.dataset_id,
            "domain": reference.domain,
            "description": "COBE2 shared-region monthly-climatology SST EOF solve restricted to the WUS shared SST mask",
            "monthly_climatology_removed": "true",
            "latitude_weighting": reference.weighting_note,
            "pc_time_period": f"{format_month(reference.overlap_months[0])} to {format_month(reference.overlap_months[-1])}",
            "shared_mask_cell_count": int(np.count_nonzero(reference.shared_mask)),
        },
    )
    ds.to_netcdf(out_dir / "cobe2_sharedregion_monthly_clim_sst_eofs.nc")
    np.save(out_dir / "cobe2_sharedregion_pc_timeseries.npy", reference.cobe2_pc_raw.astype(np.float32))
    with (out_dir / "cobe2_sharedregion_pc_timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time"] + [f"PC{idx}" for idx in range(1, N_MODES + 1)])
        for time_value, row in zip(reference.overlap_months, reference.cobe2_pc_raw):
            writer.writerow([format_month(time_value)] + [f"{float(value):.12g}" for value in row])

    lon_plot = normalize_lon_to_360(reference.longitude)
    lon_plot = np.where(lon_plot > 180.0, lon_plot - 360.0, lon_plot)
    lon2d, lat2d = np.meshgrid(lon_plot, reference.latitude)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for mode_index, ax in enumerate(axes.ravel()):
        field = np.where(reference.shared_mask, reference.cobe2_eof_unweighted[mode_index], np.nan)
        vmax = float(np.nanmax(np.abs(field)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = ax.pcolormesh(lon2d, lat2d, field, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(
            f"EOF{mode_index + 1} | EVR={float(reference.cobe2_explained_variance_ratio[mode_index]):.3f}"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    fig.suptitle(f"{reference.dataset_id} {reference.domain} shared-region COBE2 EOFs 1-6", fontsize=14)
    fig.savefig(out_dir / "cobe2_sharedregion_eofs_modes1to6.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mesh = ax.imshow(reference.weighted_gram, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(N_MODES))
    ax.set_yticks(np.arange(N_MODES))
    ax.set_xticklabels(np.arange(1, N_MODES + 1))
    ax.set_yticklabels(np.arange(1, N_MODES + 1))
    ax.set_xlabel("Mode j")
    ax.set_ylabel("Mode i")
    ax.set_title("Weighted EOF Gram Matrix over shared mask")
    fig.colorbar(mesh, ax=ax, shrink=0.86)
    fig.savefig(out_dir / "cobe2_sharedregion_weighted_gram_matrix.png", dpi=220)
    plt.close(fig)

    summary = {
        "dataset_id": reference.dataset_id,
        "domain": reference.domain,
        "time_start": format_month(reference.overlap_months[0]),
        "time_end": format_month(reference.overlap_months[-1]),
        "n_time": int(reference.overlap_months.size),
        "shared_mask_cell_count": int(np.count_nonzero(reference.shared_mask)),
        "weighting": reference.weighting_note,
        "explained_variance_ratio": [float(v) for v in reference.cobe2_explained_variance_ratio.tolist()],
        "pc_mean_raw": [float(v) for v in reference.cobe2_score_mean_raw.tolist()],
        "pc_std_raw": [float(v) for v in reference.cobe2_score_std_raw.tolist()],
        "weighted_gram_matrix": [[float(value) for value in row] for row in reference.weighted_gram.tolist()],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def load_era5_sierra_anomalies(overlap_months: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return anomalies, latitude, longitude


def load_wus_t2_sierra_anomalies(dataset_id: str, domain: str, overlap_months: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = WUS_T2_ROOT / domain / dataset_id / f"{dataset_id}_{domain}_t2_monthly_anomaly.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing WUS T2 anomaly file: {path}")
    with xr.open_dataset(path) as ds:
        time = to_month_start(np.asarray(ds["time"].values, dtype="datetime64[ns]"))
        anomalies = select_by_months(time, np.asarray(ds["t2_anomaly"].values, dtype=np.float64), overlap_months)
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = normalize_lon_to_360(np.asarray(ds["longitude"].values, dtype=np.float64))
        landmask = np.asarray(ds["landmask"].values, dtype=bool)
    return anomalies, latitude, longitude, landmask


def compute_beta_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    a = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    denominator = float(np.sum(a ** 2))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("Standardized PC denominator is zero")
    return np.nansum(a[:, np.newaxis, np.newaxis] * y, axis=0) / denominator


def compute_corr_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    a = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    a_centered = a - np.nanmean(a)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        y_centered = y - np.nanmean(y, axis=0, keepdims=True)
    numerator = np.nansum(a_centered[:, np.newaxis, np.newaxis] * y_centered, axis=0)
    denominator = np.sqrt(np.nansum(a_centered ** 2) * np.nansum(y_centered ** 2, axis=0))
    out = np.full(y.shape[1:], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def fit_multivariate_regression(predictors: np.ndarray, anomalies: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    y_flat = y.reshape(y.shape[0], -1)
    valid_columns = valid_mask.reshape(-1) & np.isfinite(y_flat).all(axis=0)
    if not np.any(valid_columns):
        raise ValueError("No valid target cells remain for multivariate regression")
    y_valid = y_flat[:, valid_columns]
    coeff_valid, _, _, _ = np.linalg.lstsq(a, y_valid, rcond=None)
    yhat_valid = a @ coeff_valid
    residual_sum = np.sum((y_valid - yhat_valid) ** 2, axis=0)
    total_sum = np.sum(y_valid ** 2, axis=0)
    r2_valid = np.full(total_sum.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(total_sum) & (total_sum > 0.0)
    r2_valid[positive] = 1.0 - (residual_sum[positive] / total_sum[positive])
    coeff_maps = np.full((N_MODES, y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    coeff_maps.reshape(N_MODES, -1)[:, valid_columns] = coeff_valid
    r2_map = np.full((y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    r2_map.reshape(-1)[valid_columns] = r2_valid
    return coeff_maps, r2_map


def build_era5_sierra_mask(anomalies: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    return (
        (lat2d >= SIERRA_REGION_360.lat_min)
        & (lat2d <= SIERRA_REGION_360.lat_max)
        & (lon2d >= SIERRA_REGION_360.lon_min)
        & (lon2d <= SIERRA_REGION_360.lon_max)
        & np.isfinite(anomalies).all(axis=0)
    )


def build_wus_sierra_mask(anomalies: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, landmask: np.ndarray) -> np.ndarray:
    return (
        landmask
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= SIERRA_REGION_360.lat_min)
        & (latitude <= SIERRA_REGION_360.lat_max)
        & (longitude >= SIERRA_REGION_360.lon_min)
        & (longitude <= SIERRA_REGION_360.lon_max)
        & np.isfinite(anomalies).all(axis=0)
    )


def subset_sierra_rows_cols(latitude_2d: np.ndarray, longitude_2d: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon_plot = normalize_lon_to_minus180_180(longitude_2d)
    regional = (
        (latitude_2d >= SIERRA_REGION_360.lat_min)
        & (latitude_2d <= SIERRA_REGION_360.lat_max)
        & (lon_plot >= DEFAULT_SIERRA_REGION.lon_min)
        & (lon_plot <= DEFAULT_SIERRA_REGION.lon_max)
    )
    row_idx = np.where(np.any(regional, axis=1))[0]
    col_idx = np.where(np.any(regional, axis=0))[0]
    if row_idx.size == 0 or col_idx.size == 0:
        raise ValueError("Sierra region does not intersect plotting grid")
    return row_idx, col_idx


def plot_level1_maps(
    output_path: Path,
    title: str,
    latitude_2d: np.ndarray,
    longitude_2d: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
    rows: List[Level1SummaryRow],
) -> None:
    row_idx, col_idx = subset_sierra_rows_cols(latitude_2d, longitude_2d, np.isfinite(r2_maps[0]))
    lat = latitude_2d[row_idx[:, None], col_idx[None, :]]
    lon = normalize_lon_to_minus180_180(longitude_2d[row_idx[:, None], col_idx[None, :]])
    beta_crop = beta_maps[:, row_idx[:, None], col_idx[None, :]]
    corr_crop = corr_maps[:, row_idx[:, None], col_idx[None, :]]
    r2_crop = r2_maps[:, row_idx[:, None], col_idx[None, :]]
    fig, axes = plt.subplots(N_MODES, 3, figsize=(15, 24), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.05, hspace=0.52, wspace=0.26)
    for mode_index in range(N_MODES):
        beta_ax, corr_ax, r2_ax = axes[mode_index]
        beta = beta_crop[mode_index]
        beta_vmax = float(np.nanmax(np.abs(beta)))
        if not np.isfinite(beta_vmax) or beta_vmax == 0.0:
            beta_vmax = 1.0
        beta_mesh = beta_ax.pcolormesh(lon, lat, beta, cmap="RdBu_r", shading="auto", vmin=-beta_vmax, vmax=beta_vmax)
        beta_ax.set_title(rf"PC{mode_index + 1} regression $\beta_{{{mode_index + 1}}}(r)$")
        fig.colorbar(beta_mesh, ax=beta_ax, shrink=0.9).set_label(r"$\beta_k(r)$ [K / 1$\sigma$ PC]")

        corr = corr_crop[mode_index]
        corr_mesh = corr_ax.pcolormesh(lon, lat, corr, cmap="RdBu_r", shading="auto", vmin=-1.0, vmax=1.0)
        corr_ax.set_title(
            rf"PC{mode_index + 1} correlation $\rho_{{{mode_index + 1}}}(r)$ | "
            rf"$\overline{{\rho}}={rows[mode_index].area_weighted_mean_corr:.3f}$"
        )
        fig.colorbar(corr_mesh, ax=corr_ax, shrink=0.9).set_label(r"$\rho_k(r)$")

        r2 = r2_crop[mode_index]
        r2_mesh = r2_ax.pcolormesh(lon, lat, r2, cmap="viridis", shading="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(r2))))
        r2_ax.set_title(
            rf"PC{mode_index + 1} explained variance $R_{{{mode_index + 1}}}^2(r)$ | "
            rf"$\overline{{R^2}}={rows[mode_index].area_weighted_mean_r2:.3f}$"
        )
        fig.colorbar(r2_mesh, ax=r2_ax, shrink=0.9).set_label(r"$R_k^2(r)$")
        for ax in (beta_ax, corr_ax, r2_ax):
            ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
            ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
    fig.suptitle(title, fontsize=15, y=0.992)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_level2_coefficients(
    output_path: Path,
    title: str,
    latitude_2d: np.ndarray,
    longitude_2d: np.ndarray,
    coefficient_maps: np.ndarray,
) -> None:
    row_idx, col_idx = subset_sierra_rows_cols(latitude_2d, longitude_2d, np.isfinite(coefficient_maps[0]))
    lat = latitude_2d[row_idx[:, None], col_idx[None, :]]
    lon = normalize_lon_to_minus180_180(longitude_2d[row_idx[:, None], col_idx[None, :]])
    coeff_crop = coefficient_maps[:, row_idx[:, None], col_idx[None, :]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(coeff_crop)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    for mode_index, ax in enumerate(axes.ravel()):
        mesh = ax.pcolormesh(lon, lat, coeff_crop[mode_index], cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(rf"PC{mode_index + 1} coefficient $\hat{{B}}_{{{mode_index + 1}}}(r)$")
        ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
        ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.86).set_label(r"$\hat{B}_k(r)$ [K / 1$\sigma$ PC]")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_level2_r2(output_path: Path, title: str, latitude_2d: np.ndarray, longitude_2d: np.ndarray, r2_map: np.ndarray) -> None:
    row_idx, col_idx = subset_sierra_rows_cols(latitude_2d, longitude_2d, np.isfinite(r2_map))
    lat = latitude_2d[row_idx[:, None], col_idx[None, :]]
    lon = normalize_lon_to_minus180_180(longitude_2d[row_idx[:, None], col_idx[None, :]])
    field = r2_map[row_idx[:, None], col_idx[None, :]]
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    mesh = ax.pcolormesh(lon, lat, field, shading="auto", cmap="viridis", vmin=0.0, vmax=max(0.05, float(np.nanmax(field))))
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
    ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, shrink=0.86).set_label(r"$R^2(r)$")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compute_monthly_level2_maps(
    overlap_months: np.ndarray,
    predictor_matrix: np.ndarray,
    anomalies: np.ndarray,
    base_mask: np.ndarray,
    weights_2d: np.ndarray,
) -> Tuple[List[MonthlySummaryRow], Dict[str, np.ndarray]]:
    rows: List[MonthlySummaryRow] = []
    maps: Dict[str, np.ndarray] = {}
    for subset_name, month_value in MONTHLY_SUBSETS:
        use_index = month_subset_index(overlap_months, month_value)
        n_time_samples = int(np.count_nonzero(use_index))
        if n_time_samples <= N_MODES:
            raise ValueError(f"Month {subset_name} has too few samples ({n_time_samples}) for six-PC regression")
        coeff_maps, r2_map = fit_multivariate_regression(predictor_matrix[use_index], anomalies[use_index], base_mask)
        _ = coeff_maps
        values = np.asarray(r2_map, dtype=np.float64)[base_mask]
        row = MonthlySummaryRow(
            subset_name=subset_name,
            subset_group="monthly",
            n_time_samples=n_time_samples,
            sierra_lat_min=SIERRA_REGION_360.lat_min,
            sierra_lat_max=SIERRA_REGION_360.lat_max,
            sierra_lon_min=SIERRA_REGION_360.lon_min,
            sierra_lon_max=SIERRA_REGION_360.lon_max,
            number_of_valid_sierra_grid_points=int(np.count_nonzero(base_mask)),
            mean_r2=masked_area_weighted_mean(r2_map, weights_2d, base_mask),
            median_r2=float(np.nanmedian(values)),
            max_r2=float(np.nanmax(values)),
            min_r2=float(np.nanmin(values)),
        )
        rows.append(row)
        maps[subset_name] = r2_map
        print(
            f"  monthly subset {subset_name}: n={n_time_samples}, "
            f"valid Sierra cells={row.number_of_valid_sierra_grid_points}, mean R2={row.mean_r2:.4f}",
            flush=True,
        )
    return rows, maps


def plot_monthly_maps(
    output_path: Path,
    title: str,
    latitude_2d: np.ndarray,
    longitude_2d: np.ndarray,
    monthly_rows: List[MonthlySummaryRow],
    monthly_maps: Dict[str, np.ndarray],
) -> None:
    vmax = 0.0
    for subset_name, _ in MONTHLY_SUBSETS:
        subset_vmax = float(np.nanmax(monthly_maps[subset_name]))
        if np.isfinite(subset_vmax):
            vmax = max(vmax, subset_vmax)
    vmax = max(0.05, vmax)
    row_idx, col_idx = subset_sierra_rows_cols(latitude_2d, longitude_2d, np.isfinite(monthly_maps[MONTHLY_SUBSETS[0][0]]))
    lat = latitude_2d[row_idx[:, None], col_idx[None, :]]
    lon = normalize_lon_to_minus180_180(longitude_2d[row_idx[:, None], col_idx[None, :]])

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 11.8), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.08, hspace=0.32, wspace=0.20)
    axes_flat = np.atleast_1d(axes).flatten()
    row_map = {row.subset_name: row for row in monthly_rows}
    for ax, (subset_name, _) in zip(axes_flat, MONTHLY_SUBSETS):
        field = monthly_maps[subset_name][row_idx[:, None], col_idx[None, :]]
        row = row_map[subset_name]
        mesh = ax.pcolormesh(lon, lat, field, cmap="viridis", shading="auto", vmin=0.0, vmax=vmax)
        ax.set_title(f"{subset_name} | n={row.n_time_samples} | mean R2={row.mean_r2:.3f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
        ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.84).set_label(r"$R^2(r)$")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_monthly_mean_r2(output_path: Path, title: str, monthly_rows: List[MonthlySummaryRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(monthly_rows), dtype=np.int32)
    y = np.asarray([row.mean_r2 for row in monthly_rows], dtype=np.float64)
    labels = [row.subset_name for row in monthly_rows]
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sierra mean $R^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_level1_outputs(
    root: Path,
    domain: str,
    dataset_id: str,
    netcdf_name: str,
    summary_name: str,
    figure_name: str,
    description: str,
    overlap_months: np.ndarray,
    latitude_2d: np.ndarray,
    longitude_2d: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
    standardized_pc: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    explained_variance_ratio: np.ndarray,
    rows: List[Level1SummaryRow],
    figure_title: str,
) -> None:
    out_dir = dataset_output_dir(root, domain, dataset_id)
    ensure_dir(out_dir)
    ds = xr.Dataset(
        data_vars={
            "beta": (("mode", "lat2d", "lon2d"), beta_maps.astype(np.float32)),
            "corr": (("mode", "lat2d", "lon2d"), corr_maps.astype(np.float32)),
            "r2": (("mode", "lat2d", "lon2d"), r2_maps.astype(np.float32)),
            "projected_pc_standardized": (("time", "mode"), standardized_pc.astype(np.float32)),
            "projected_pc_raw_mean": (("mode",), raw_mean.astype(np.float32)),
            "projected_pc_raw_std": (("mode",), raw_std.astype(np.float32)),
            "explained_variance_ratio": (("mode",), explained_variance_ratio.astype(np.float32)),
        },
        coords={
            "time": overlap_months,
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude_2d.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude_2d.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude_2d.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude_2d.astype(np.float32)),
        },
        attrs={
            "description": description,
            "formula_beta": "beta_k(r) = sum_t[a_k(t) Y(t,r)] / sum_t[a_k(t)^2]",
            "formula_corr": "rho_k(r) = corr(a_k(t), Y(t,r))",
            "formula_r2": "R_k^2(r) = rho_k(r)^2",
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "dataset_id": dataset_id,
            "domain": domain,
        },
    )
    ds.to_netcdf(out_dir / netcdf_name)
    (out_dir / summary_name).write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "domain": domain,
                "overlap_start": format_month(overlap_months[0]),
                "overlap_end": format_month(overlap_months[-1]),
                "n_overlap_months": int(overlap_months.size),
                "pc_standardized": True,
                "summary_rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_level1_maps(out_dir / figure_name, figure_title, latitude_2d, longitude_2d, beta_maps, corr_maps, r2_maps, rows)


def save_level2_outputs(
    root: Path,
    domain: str,
    dataset_id: str,
    netcdf_name: str,
    summary_name: str,
    coeff_figure_name: str,
    r2_figure_name: str,
    description: str,
    overlap_months: np.ndarray,
    latitude_2d: np.ndarray,
    longitude_2d: np.ndarray,
    coeff_maps: np.ndarray,
    r2_map: np.ndarray,
    standardized_pc: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    summary: Level2Summary,
    coeff_title: str,
    r2_title: str,
) -> None:
    out_dir = dataset_output_dir(root, domain, dataset_id)
    ensure_dir(out_dir)
    ds = xr.Dataset(
        data_vars={
            "multi_pc_beta": (("mode", "lat2d", "lon2d"), coeff_maps.astype(np.float32)),
            "multi_pc_r2": (("lat2d", "lon2d"), r2_map.astype(np.float32)),
            "projected_pc_standardized": (("time", "mode"), standardized_pc.astype(np.float32)),
            "projected_pc_raw_mean": (("mode",), raw_mean.astype(np.float32)),
            "projected_pc_raw_std": (("mode",), raw_std.astype(np.float32)),
        },
        coords={
            "time": overlap_months,
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude_2d.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude_2d.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude_2d.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude_2d.astype(np.float32)),
        },
        attrs={
            "description": description,
            "formula_bhat": "B_hat = (A^T A)^(-1) A^T Y",
            "formula_yhat": "Y_hat = A B_hat",
            "formula_r2": "R2(r) = 1 - sum_t[(Y(t,r)-Y_hat(t,r))^2] / sum_t[Y(t,r)^2]",
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "dataset_id": dataset_id,
            "domain": domain,
        },
    )
    ds.to_netcdf(out_dir / netcdf_name)
    (out_dir / summary_name).write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "domain": domain,
                "overlap_start": format_month(overlap_months[0]),
                "overlap_end": format_month(overlap_months[-1]),
                "n_overlap_months": int(overlap_months.size),
                "pc_standardized": True,
                "mean_r2_joint": summary.mean_r2_joint,
                "max_r2_joint": summary.max_r2_joint,
                "min_r2_joint": summary.min_r2_joint,
                "median_r2_joint": summary.median_r2_joint,
                "projected_pc_mean_raw": [float(v) for v in raw_mean.tolist()],
                "projected_pc_std_raw": [float(v) for v in raw_std.tolist()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_level2_coefficients(out_dir / coeff_figure_name, coeff_title, latitude_2d, longitude_2d, coeff_maps)
    plot_level2_r2(out_dir / r2_figure_name, r2_title, latitude_2d, longitude_2d, r2_map)


def save_monthly_outputs(
    root: Path,
    domain: str,
    dataset_id: str,
    base_name: str,
    title_map: str,
    title_line: str,
    overlap_months: np.ndarray,
    latitude_2d: np.ndarray,
    longitude_2d: np.ndarray,
    monthly_rows: List[MonthlySummaryRow],
    monthly_maps: Dict[str, np.ndarray],
    standardized_pc: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    description: str,
) -> None:
    out_dir = dataset_output_dir(root, domain, dataset_id)
    ensure_dir(out_dir)
    subset_names = np.asarray([row.subset_name for row in monthly_rows], dtype="U8")
    stack = np.stack([monthly_maps[name] for name in subset_names.tolist()], axis=0)
    ds = xr.Dataset(
        data_vars={
            "monthly_r2": (("monthly_subset", "lat2d", "lon2d"), stack.astype(np.float32)),
            "monthly_mean_r2": (("monthly_subset",), np.asarray([row.mean_r2 for row in monthly_rows], dtype=np.float32)),
            "monthly_n_time_samples": (("monthly_subset",), np.asarray([row.n_time_samples for row in monthly_rows], dtype=np.int32)),
            "projected_pc_standardized": (("time", "mode"), standardized_pc.astype(np.float32)),
            "projected_pc_raw_mean": (("mode",), raw_mean.astype(np.float32)),
            "projected_pc_raw_std": (("mode",), raw_std.astype(np.float32)),
            "latitude": (("lat2d", "lon2d"), latitude_2d.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude_2d.astype(np.float32)),
        },
        coords={
            "time": overlap_months.astype("datetime64[ns]"),
            "monthly_subset": subset_names,
            "lat2d": np.arange(latitude_2d.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude_2d.shape[1], dtype=np.int32),
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
        },
        attrs={
            "description": description,
            "formula_bhat_monthly": "B_hat_m = (A_m^T A_m)^(-1) A_m^T Y_m",
            "formula_yhat_monthly": "Y_hat_m = A_m B_hat_m",
            "formula_r2_monthly": "R_m^2(r) = 1 - sum_{t in m}[(Y_m(t,r)-Y_hat_m(t,r))^2] / sum_{t in m}[Y_m(t,r)^2]",
            "pc_standardized": "true",
            "dataset_id": dataset_id,
            "domain": domain,
        },
    )
    ds.to_netcdf(out_dir / f"{base_name}.nc")
    with (out_dir / f"{base_name}_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(monthly_rows[0]).keys()))
        writer.writeheader()
        for row in monthly_rows:
            writer.writerow(asdict(row))
    (out_dir / f"{base_name}_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "domain": domain,
                "overlap_start": format_month(overlap_months[0]),
                "overlap_end": format_month(overlap_months[-1]),
                "n_overlap_months": int(overlap_months.size),
                "pc_standardized": True,
                "projected_pc_mean_raw": [float(v) for v in raw_mean.tolist()],
                "projected_pc_std_raw": [float(v) for v in raw_std.tolist()],
                "monthly_rows": [asdict(row) for row in monthly_rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_monthly_maps(out_dir / f"{base_name}_monthly_r2_maps.png", title_map, latitude_2d, longitude_2d, monthly_rows, monthly_maps)
    plot_monthly_mean_r2(out_dir / f"{base_name}_monthly_mean_r2.png", title_line, monthly_rows)


def run_observational_diagnostics(reference: SharedRegionReference) -> None:
    out_level1 = dataset_output_dir(OBS_LEVEL1_ROOT, reference.domain, reference.dataset_id)
    out_level2 = dataset_output_dir(OBS_LEVEL2_ROOT, reference.domain, reference.dataset_id)
    out_monthly = dataset_output_dir(OBS_MONTHLY_ROOT, reference.domain, reference.dataset_id)
    ensure_dir(out_level1)
    ensure_dir(out_level2)
    ensure_dir(out_monthly)

    anomalies, latitude, longitude = load_era5_sierra_anomalies(reference.overlap_months)
    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    base_mask = build_era5_sierra_mask(anomalies, latitude, longitude)
    weights_2d = area_weights_from_latitude(latitude, anomalies.shape[1:])
    pc_std, pc_mean, pc_std_raw = standardize_pc_matrix(reference.cobe2_pc_raw)

    beta_maps = np.full((N_MODES,) + anomalies.shape[1:], np.nan, dtype=np.float64)
    corr_maps = np.full_like(beta_maps, np.nan)
    r2_maps = np.full_like(beta_maps, np.nan)
    level1_rows: List[Level1SummaryRow] = []
    for mode_index in range(N_MODES):
        beta = compute_beta_map(pc_std[:, mode_index], anomalies)
        corr = compute_corr_map(pc_std[:, mode_index], anomalies)
        r2 = corr ** 2
        beta_maps[mode_index] = beta
        corr_maps[mode_index] = corr
        r2_maps[mode_index] = r2
        level1_rows.append(
            Level1SummaryRow(
                mode=mode_index + 1,
                explained_variance_ratio=float(reference.cobe2_explained_variance_ratio[mode_index]),
                pc_mean_raw=float(pc_mean[mode_index]),
                pc_std_raw=float(pc_std_raw[mode_index]),
                area_weighted_mean_r2=area_weighted_mean(r2, lat2d),
                area_weighted_mean_corr=area_weighted_mean(corr, lat2d),
                max_local_r2=float(np.nanmax(r2)),
            )
        )
    save_level1_outputs(
        root=OBS_LEVEL1_ROOT,
        domain=reference.domain,
        dataset_id=reference.dataset_id,
        netcdf_name="cobe2_sharedregion_sierra_t2m_level1_diagnostic.nc",
        summary_name="summary.json",
        figure_name="cobe2_sharedregion_sierra_t2m_level1_maps_modes1to6.png",
        description="Shared-region COBE2 Level 1 OLS diagnostic using PCs recomputed on the WUS shared SST domain",
        overlap_months=reference.overlap_months,
        latitude_2d=lat2d,
        longitude_2d=lon2d,
        beta_maps=beta_maps,
        corr_maps=corr_maps,
        r2_maps=r2_maps,
        standardized_pc=pc_std,
        raw_mean=pc_mean,
        raw_std=pc_std_raw,
        explained_variance_ratio=reference.cobe2_explained_variance_ratio,
        rows=level1_rows,
        figure_title=f"{reference.dataset_id} shared-region COBE2 PCs vs ERA5 Sierra T2 anomalies",
    )

    coeff_maps, r2_map = fit_multivariate_regression(pc_std, anomalies, base_mask)
    level2_summary = Level2Summary(
        mean_r2_joint=masked_area_weighted_mean(r2_map, weights_2d, base_mask),
        max_r2_joint=float(np.nanmax(r2_map)),
        min_r2_joint=float(np.nanmin(r2_map)),
        median_r2_joint=float(np.nanmedian(r2_map[base_mask])),
    )
    save_level2_outputs(
        root=OBS_LEVEL2_ROOT,
        domain=reference.domain,
        dataset_id=reference.dataset_id,
        netcdf_name="cobe2_sharedregion_sierra_t2m_level2_pc1to6.nc",
        summary_name="summary.json",
        coeff_figure_name="cobe2_sharedregion_sierra_t2m_level2_pc1to6_coefficients_modes1to6.png",
        r2_figure_name="cobe2_sharedregion_sierra_t2m_level2_pc1to6_r2_map.png",
        description="Shared-region COBE2 Level 2 OLS diagnostic using PCs recomputed on the WUS shared SST domain",
        overlap_months=reference.overlap_months,
        latitude_2d=lat2d,
        longitude_2d=lon2d,
        coeff_maps=coeff_maps,
        r2_map=r2_map,
        standardized_pc=pc_std,
        raw_mean=pc_mean,
        raw_std=pc_std_raw,
        summary=level2_summary,
        coeff_title=f"{reference.dataset_id} shared-region COBE2 Level 2 OLS coefficients",
        r2_title="Shared-region COBE2 Level 2 OLS joint R2",
    )

    monthly_rows, monthly_maps = compute_monthly_level2_maps(reference.overlap_months, pc_std, anomalies, base_mask, weights_2d)
    save_monthly_outputs(
        root=OBS_MONTHLY_ROOT,
        domain=reference.domain,
        dataset_id=reference.dataset_id,
        base_name="cobe2_sharedregion_sierra_t2m_level2_pc1to6",
        title_map="Shared-region COBE2 PC1-PC6 -> ERA5 Sierra monthly $R^2$",
        title_line="Shared-region COBE2 -> ERA5 Sierra monthly mean $R^2$",
        overlap_months=reference.overlap_months,
        latitude_2d=lat2d,
        longitude_2d=lon2d,
        monthly_rows=monthly_rows,
        monthly_maps=monthly_maps,
        standardized_pc=pc_std,
        raw_mean=pc_mean,
        raw_std=pc_std_raw,
        description="Shared-region COBE2 monthly Level 2 OLS diagnostic using PCs recomputed on the WUS shared SST domain",
    )


def project_wus_onto_sharedregion_eofs(reference: SharedRegionReference) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wus_time, wus_anom = load_wus_sst_monthly_anomaly(reference.dataset_id, reference.domain)
    overlap_values = select_by_months(wus_time, wus_anom, reference.overlap_months)
    lat_weights_1d = np.sqrt(np.clip(np.cos(np.deg2rad(reference.latitude)), 0.0, None))
    weights_2d = np.broadcast_to(lat_weights_1d[:, np.newaxis], reference.shared_mask.shape)
    weights_flat = weights_2d[reference.shared_mask]
    weighted_anom = overlap_values[:, reference.shared_mask] * weights_flat[np.newaxis, :]
    weighted_eof = reference.cobe2_eof_unweighted[:, reference.shared_mask] * weights_flat[np.newaxis, :]
    projected_pc = weighted_anom @ weighted_eof.T
    pc_std, pc_mean, pc_std_raw = standardize_pc_matrix(projected_pc)
    return projected_pc, pc_mean, pc_std_raw


def save_wus_projection_outputs(reference: SharedRegionReference, projected_pc: np.ndarray) -> None:
    out_dir = dataset_output_dir(WUS_PROJ_ROOT, reference.domain, reference.dataset_id)
    ensure_dir(out_dir)
    np.save(out_dir / "projected_pc_timeseries.npy", projected_pc.astype(np.float32))
    with (out_dir / "projected_pc_timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time"] + [f"PC{idx}" for idx in range(1, N_MODES + 1)])
        for time_value, row in zip(reference.overlap_months, projected_pc):
            writer.writerow([format_month(time_value)] + [f"{float(value):.12g}" for value in row])
    ds = xr.Dataset(
        data_vars={
            "projected_pc": (("time", "mode"), projected_pc.astype(np.float32)),
            "projection_shared_mask": (("lat", "lon"), reference.shared_mask.astype(np.int8)),
            "cobe2_sharedregion_eof": (("mode", "lat", "lon"), reference.cobe2_eof_unweighted.astype(np.float32)),
        },
        coords={
            "time": reference.overlap_months.astype("datetime64[ns]"),
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat": reference.latitude.astype(np.float32),
            "lon": reference.longitude.astype(np.float32),
        },
        attrs={
            "dataset_id": reference.dataset_id,
            "domain": reference.domain,
            "description": "WUS SST anomalies projected onto shared-region COBE2 EOFs",
            "projection_weighting": reference.weighting_note,
        },
    )
    ds.to_netcdf(out_dir / "projected_pc_timeseries_and_mask.nc")
    fig, axes = plt.subplots(N_MODES, 1, figsize=(12, 2.0 * N_MODES), sharex=True, constrained_layout=True)
    time_plot = reference.overlap_months.astype("datetime64[ns]")
    pc_std = np.std(projected_pc, axis=0, ddof=1)
    if N_MODES == 1:
        axes = [axes]
    for mode_index, ax in enumerate(axes):
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax.plot(time_plot, projected_pc[:, mode_index], color="black", linewidth=1.0)
        ax.set_ylabel(f"PC{mode_index + 1}")
        ax.set_title(f"Projected PC{mode_index + 1} std={float(pc_std[mode_index]):.3f}")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"{reference.dataset_id} {reference.domain} WUS projected PCs on shared-region COBE2 EOFs", fontsize=14)
    fig.savefig(out_dir / "projected_pc_timeseries_modes1to6.png", dpi=220)
    plt.close(fig)


def run_wus_diagnostics(reference: SharedRegionReference) -> None:
    projected_pc_raw, pc_mean, pc_std_raw = project_wus_onto_sharedregion_eofs(reference)
    save_wus_projection_outputs(reference, projected_pc_raw)
    pc_std, _, _ = standardize_pc_matrix(projected_pc_raw)

    anomalies, latitude, longitude, landmask = load_wus_t2_sierra_anomalies(reference.dataset_id, reference.domain, reference.overlap_months)
    base_mask = build_wus_sierra_mask(anomalies, latitude, longitude, landmask)
    weights_2d = area_weights_from_latitude(latitude, anomalies.shape[1:])
    if not np.any(base_mask):
        raise ValueError("No valid Sierra WUS land cells found for shared-region diagnostics")

    beta_maps = np.full((N_MODES,) + anomalies.shape[1:], np.nan, dtype=np.float64)
    corr_maps = np.full_like(beta_maps, np.nan)
    r2_maps = np.full_like(beta_maps, np.nan)
    level1_rows: List[Level1SummaryRow] = []
    for mode_index in range(N_MODES):
        beta = compute_beta_map(pc_std[:, mode_index], anomalies)
        corr = compute_corr_map(pc_std[:, mode_index], anomalies)
        r2 = corr ** 2
        beta_maps[mode_index] = beta
        corr_maps[mode_index] = corr
        r2_maps[mode_index] = r2
        level1_rows.append(
            Level1SummaryRow(
                mode=mode_index + 1,
                explained_variance_ratio=float(reference.cobe2_explained_variance_ratio[mode_index]),
                pc_mean_raw=float(pc_mean[mode_index]),
                pc_std_raw=float(pc_std_raw[mode_index]),
                area_weighted_mean_r2=area_weighted_mean(r2, latitude),
                area_weighted_mean_corr=area_weighted_mean(corr, latitude),
                max_local_r2=float(np.nanmax(r2)),
            )
        )
    save_level1_outputs(
        root=WUS_LEVEL1_ROOT,
        domain=reference.domain,
        dataset_id=reference.dataset_id,
        netcdf_name="wusd3_sharedregion_pc_t2m_level1_ols.nc",
        summary_name="summary.json",
        figure_name="wusd3_sharedregion_pc_t2m_level1_ols_maps_modes1to6.png",
        description="WUS Level 1 OLS diagnostic using WUS SST projected onto shared-region COBE2 EOFs",
        overlap_months=reference.overlap_months,
        latitude_2d=latitude,
        longitude_2d=longitude,
        beta_maps=beta_maps,
        corr_maps=corr_maps,
        r2_maps=r2_maps,
        standardized_pc=pc_std,
        raw_mean=pc_mean,
        raw_std=pc_std_raw,
        explained_variance_ratio=reference.cobe2_explained_variance_ratio,
        rows=level1_rows,
        figure_title=f"{reference.dataset_id} WUS shared-region projected PCs vs WUSD-03 Sierra T2 anomalies",
    )

    coeff_maps, r2_map = fit_multivariate_regression(pc_std, anomalies, base_mask)
    level2_summary = Level2Summary(
        mean_r2_joint=masked_area_weighted_mean(r2_map, weights_2d, base_mask),
        max_r2_joint=float(np.nanmax(r2_map)),
        min_r2_joint=float(np.nanmin(r2_map)),
        median_r2_joint=float(np.nanmedian(r2_map[base_mask])),
    )
    save_level2_outputs(
        root=WUS_LEVEL2_ROOT,
        domain=reference.domain,
        dataset_id=reference.dataset_id,
        netcdf_name="wusd3_sharedregion_pc_t2m_level2_ols.nc",
        summary_name="summary.json",
        coeff_figure_name="wusd3_sharedregion_pc_t2m_level2_ols_coefficients_modes1to6.png",
        r2_figure_name="wusd3_sharedregion_pc_t2m_level2_ols_r2_map.png",
        description="WUS Level 2 OLS diagnostic using WUS SST projected onto shared-region COBE2 EOFs",
        overlap_months=reference.overlap_months,
        latitude_2d=latitude,
        longitude_2d=longitude,
        coeff_maps=coeff_maps,
        r2_map=r2_map,
        standardized_pc=pc_std,
        raw_mean=pc_mean,
        raw_std=pc_std_raw,
        summary=level2_summary,
        coeff_title=f"{reference.dataset_id} WUS shared-region Level 2 OLS coefficients",
        r2_title="WUS shared-region Level 2 OLS joint R2",
    )

    monthly_rows, monthly_maps = compute_monthly_level2_maps(reference.overlap_months, pc_std, anomalies, base_mask, weights_2d)
    save_monthly_outputs(
        root=WUS_MONTHLY_ROOT,
        domain=reference.domain,
        dataset_id=reference.dataset_id,
        base_name="wusd3_sharedregion_pc_t2m_level2_pc1to6",
        title_map=f"{reference.dataset_id} WUS shared-region PC1-PC6 -> WUSD-03 Sierra monthly $R^2$",
        title_line="WUS shared-region -> WUSD-03 Sierra monthly mean $R^2$",
        overlap_months=reference.overlap_months,
        latitude_2d=latitude,
        longitude_2d=longitude,
        monthly_rows=monthly_rows,
        monthly_maps=monthly_maps,
        standardized_pc=pc_std,
        raw_mean=pc_mean,
        raw_std=pc_std_raw,
        description="WUS monthly Level 2 OLS diagnostic using WUS SST projected onto shared-region COBE2 EOFs",
    )


def main() -> None:
    args = parse_args()
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    latitude, longitude, overlap_months, shared_mask, weighting_note = build_shared_mask(args.dataset_id, args.domain)
    print(
        f"Shared mask for {args.dataset_id} {args.domain}: "
        f"time={format_month(overlap_months[0])}..{format_month(overlap_months[-1])} "
        f"n_time={int(overlap_months.size)} shared_cells={int(np.count_nonzero(shared_mask))}",
        flush=True,
    )
    reference = solve_shared_region_cobe2_eofs(
        dataset_id=args.dataset_id,
        domain=args.domain,
        overlap_months=overlap_months,
        latitude=latitude,
        longitude=longitude,
        shared_mask=shared_mask,
        weighting_note=weighting_note,
    )
    save_shared_region_eof_outputs(reference)
    run_observational_diagnostics(reference)
    run_wus_diagnostics(reference)
    print(
        f"Finished shared-region COBE2/WUS diagnostics for {args.dataset_id} {args.domain}",
        flush=True,
    )


if __name__ == "__main__":
    main()
