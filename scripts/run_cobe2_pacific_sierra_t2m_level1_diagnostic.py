#!/usr/bin/env python3
"""
Run a Level 1 diagnostic linking Pacific COBE2 SST PCs to matched-region ERA5-Land T2m anomalies.

This script:
1. Subsets COBE2 SST to the Pacific domain (lat -10..60, lon 120..280 in 0..360).
2. Removes month-of-year climatology and computes a 6-mode weighted EOF/PCA decomposition.
3. Subsets ERA5-Land monthly mean and climatology to the matching Pacific response domain.
4. Builds matched-region monthly T2m anomalies over the common overlap period.
5. For each Pacific SST PC k, computes over matched-region grid cells:
   - regression coefficient beta_k(r)
   - temporal correlation rho_k(r)
   - explained variance R_k^2(r)
6. Computes area-weighted matched-region summary metrics per PC.
"""

import csv
import json
import os
import sys
import warnings
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

from scripts.run_cobe2_global_sst_eof_reproduction import COBE2_SST_FILE
from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    compute_monthly_climatology_anomalies,
    ensure_runtime_on_compute_node,
    get_runtime,
    open_dataset_with_fallbacks,
)
from snow_ml.data import RegionBounds


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level1_diagnostic"
PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL1_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level1_diagnostic",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level1_diagnostic"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level1_diagnostic.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level1_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level1_summary.json"
FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level1_maps_modes1to6.png"
EOF_FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_cobe2_eofs_modes1to6.png"

ERA5_MONTHLY_MEAN_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5land_t2m_monthly_anomalies/era5land_t2m_monthly_mean.nc"
)
ERA5_MONTHLY_CLIM_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5land_t2m_monthly_anomalies/era5land_t2m_monthly_climatology.nc"
)

COBE2_VARIABLE = "sst"
ERA5_VARIABLE = "t2m"
N_MODES = 6
MODE_SIGN = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64)
TIME_CHUNK = 12
LAT_CHUNK = 180
LON_CHUNK = 360

PACIFIC_SST_REGION_360 = RegionBounds(
    lat_min=-10.0,
    lat_max=60.0,
    lon_min=120.0,
    lon_max=280.0,
)
SIERRA_T2M_REGION_360 = PACIFIC_SST_REGION_360
T2M_RESPONSE_REGION_LABEL = "matched Pacific ERA5-Land T2m region"


@dataclass(frozen=True)
class SummaryRow:
    mode: int
    explained_variance_ratio: float
    pc_std_raw: float
    pc_mean_raw: float
    area_weighted_mean_r2: float
    area_weighted_mean_corr: float
    max_local_r2: float


@dataclass(frozen=True)
class SummaryPayload:
    experiment_name: str
    pacific_sst_region: Dict[str, float]
    sierra_t2m_region: Dict[str, float]
    input_cobe2_sst_path: str
    input_era5_monthly_mean_path: str
    input_era5_monthly_climatology_path: str
    output_netcdf_path: str
    output_figure_path: str
    overlap_start: str
    overlap_end: str
    n_overlap_months: int
    n_modes: int
    pc_standardized: bool
    mode_signs: List[float]
    pacific_sst_shape: List[int]
    sierra_t2m_shape: List[int]
    slurm_job_id: str
    compute_node: str
    summary_rows: List[Dict[str, float]]


def ensure_output_dir() -> None:
    PSCRATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HOME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def format_month(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def to_month_start(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("datetime64[M]").astype("datetime64[ns]")


def month_number(value: np.datetime64) -> int:
    return int(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D")[5:7])


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


def build_time_index(month_values: np.ndarray) -> Dict[np.datetime64, int]:
    return {month: index for index, month in enumerate(month_values.tolist())}


def normalize_lon_to_360(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=np.float64)
    return np.mod(values, 360.0)


def subset_lonlat_3d_360(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    region: RegionBounds,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = normalize_lon_to_360(np.asarray(longitude, dtype=np.float64))
    lon_sort_idx = np.argsort(lon)
    lon_sorted = lon[lon_sort_idx]
    values_sorted = np.take(values, lon_sort_idx, axis=-1)

    lat_mask = (lat >= region.lat_min) & (lat <= region.lat_max)
    lon_mask = (lon_sorted >= region.lon_min) & (lon_sorted <= region.lon_max)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Requested region does not overlap source grid")
    return lat[lat_idx], lon_sorted[lon_idx], values_sorted[:, lat_idx, :][:, :, lon_idx]


def solve_weighted_pacific_eofs(
    time_values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
) -> Dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        climatology, anomalies = compute_monthly_climatology_anomalies(values, time_values)

    anomalies_flat = anomalies.reshape(anomalies.shape[0], -1)
    valid_mask_flat = np.isfinite(anomalies_flat).all(axis=0)
    n_valid_cells = int(valid_mask_flat.sum())
    if n_valid_cells < N_MODES:
        raise ValueError(f"Need at least {N_MODES} all-time-finite ocean cells, got {n_valid_cells}")

    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(latitude)), 0.0, None))
    weights_2d = np.broadcast_to(lat_weights[:, np.newaxis], (latitude.size, longitude.size))
    weights_flat = weights_2d.reshape(-1)[valid_mask_flat]
    anomaly_matrix = anomalies_flat[:, valid_mask_flat]
    weighted_matrix = anomaly_matrix * weights_flat[np.newaxis, :]

    gram_matrix = weighted_matrix @ weighted_matrix.T
    eigenvalues, u_matrix = np.linalg.eigh(gram_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    u_matrix = u_matrix[:, order]
    singular_values_all = np.sqrt(eigenvalues)
    total_variance = float(np.sum(singular_values_all ** 2))
    if total_variance <= 0.0:
        raise ValueError("Total weighted anomaly variance is zero")
    explained_variance_ratio = (singular_values_all ** 2) / total_variance

    pcs = (u_matrix[:, :N_MODES] * singular_values_all[:N_MODES]).astype(np.float64)

    weighted_eofs_valid = np.zeros((N_MODES, n_valid_cells), dtype=np.float64)
    for mode_index in range(N_MODES):
        singular_value = singular_values_all[mode_index]
        if singular_value <= 0.0:
            continue
        weighted_eofs_valid[mode_index] = (weighted_matrix.T @ u_matrix[:, mode_index]) / singular_value

    unweighted_eofs_valid = np.full_like(weighted_eofs_valid, np.nan)
    positive_weight = weights_flat > 0.0
    unweighted_eofs_valid[:, positive_weight] = (
        weighted_eofs_valid[:, positive_weight] / weights_flat[np.newaxis, positive_weight]
    )

    eof_grid = np.full((N_MODES, latitude.size, longitude.size), np.nan, dtype=np.float64)
    eof_grid.reshape(N_MODES, -1)[:, valid_mask_flat] = unweighted_eofs_valid
    valid_mask_2d = np.zeros((latitude.size, longitude.size), dtype=bool)
    valid_mask_2d.reshape(-1)[valid_mask_flat] = True

    pcs = pcs * MODE_SIGN[np.newaxis, :]
    eof_grid = eof_grid * MODE_SIGN[:, np.newaxis, np.newaxis]

    return {
        "time": np.asarray(time_values, dtype="datetime64[ns]"),
        "latitude": np.asarray(latitude, dtype=np.float64),
        "longitude": np.asarray(longitude, dtype=np.float64),
        "pc": pcs.astype(np.float64),
        "eof": eof_grid.astype(np.float64),
        "valid_mask": valid_mask_2d,
        "explained_variance_ratio": explained_variance_ratio[:N_MODES].astype(np.float64),
        "singular_value": singular_values_all[:N_MODES].astype(np.float64),
        "climatology": climatology.astype(np.float64),
    }


def load_pacific_cobe2_pca(region: RegionBounds) -> Dict[str, np.ndarray]:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        if COBE2_VARIABLE not in ds:
            raise KeyError(f"Expected variable {COBE2_VARIABLE!r} in {COBE2_SST_FILE}")
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        values = np.asarray(ds[COBE2_VARIABLE].values, dtype=np.float64)
        missing_value = float(ds[COBE2_VARIABLE].attrs.get("missing_value", 1.0e20))

    values = np.where(values >= missing_value, np.nan, values)
    lat_crop, lon_crop, values_crop = subset_lonlat_3d_360(latitude, longitude, values, region)
    return solve_weighted_pacific_eofs(time_values, lat_crop, lon_crop, values_crop)


def subset_era5_region_360(field: xr.DataArray, region: RegionBounds) -> xr.DataArray:
    lon_name = "longitude"
    lat_name = "latitude"
    subset = field.assign_coords({lon_name: field[lon_name] % 360.0}).sortby(lat_name).sortby(lon_name)
    return subset.sel(
        {
            lat_name: slice(region.lat_min, region.lat_max),
            lon_name: slice(region.lon_min, region.lon_max),
        }
    )


def compute_corr_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    pc_values = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    pc_centered = pc_values - np.nanmean(pc_values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        y_centered = y - np.nanmean(y, axis=0, keepdims=True)
    numerator = np.nansum(pc_centered[:, np.newaxis, np.newaxis] * y_centered, axis=0)
    denominator = np.sqrt(np.nansum(pc_centered ** 2) * np.nansum(y_centered ** 2, axis=0))
    corr = np.full(y.shape[1:], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    corr[valid] = numerator[valid] / denominator[valid]
    return corr


def compute_beta_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    pc_values = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    denominator = float(np.nansum(pc_values ** 2))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("PC denominator is zero or non-finite")
    return np.nansum(pc_values[:, np.newaxis, np.newaxis] * y, axis=0) / denominator


def area_weighted_mean(field: np.ndarray, latitude: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(values)
    weighted_sum = np.nansum(np.where(valid, values * weights, 0.0))
    weight_sum = np.nansum(np.where(valid, weights, 0.0))
    if not np.isfinite(weight_sum) or weight_sum == 0.0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def standardize_pc_matrix(pc_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(pc_matrix, dtype=np.float64)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
        raise ValueError("Cannot standardize PCs with non-finite or zero standard deviation")
    standardized = (values - means[np.newaxis, :]) / stds[np.newaxis, :]
    return standardized, means, stds


def compute_level1_maps(
    pc_overlap: np.ndarray,
    anomalies: np.ndarray,
    latitude: np.ndarray,
    explained_variance_ratio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SummaryRow]]:
    beta_maps = np.full((N_MODES, anomalies.shape[1], anomalies.shape[2]), np.nan, dtype=np.float64)
    corr_maps = np.full_like(beta_maps, np.nan)
    r2_maps = np.full_like(beta_maps, np.nan)
    rows: List[SummaryRow] = []

    for mode_index in range(N_MODES):
        pc_values = np.asarray(pc_overlap[:, mode_index], dtype=np.float64)
        beta = compute_beta_map(pc_values, anomalies)
        corr = compute_corr_map(pc_values, anomalies)
        r2 = corr ** 2
        beta_maps[mode_index] = beta
        corr_maps[mode_index] = corr
        r2_maps[mode_index] = r2
        rows.append(
            SummaryRow(
                mode=mode_index + 1,
                explained_variance_ratio=float(explained_variance_ratio[mode_index]),
                pc_std_raw=float(np.std(pc_values, ddof=1)),
                pc_mean_raw=float(np.mean(pc_values)),
                area_weighted_mean_r2=area_weighted_mean(r2, latitude),
                area_weighted_mean_corr=area_weighted_mean(corr, latitude),
                max_local_r2=float(np.nanmax(r2)),
            )
        )

    return beta_maps, corr_maps, r2_maps, rows


def save_netcdf(
    pacific: Dict[str, np.ndarray],
    overlap_months: np.ndarray,
    pc_overlap: np.ndarray,
    pc_overlap_raw_mean: np.ndarray,
    pc_overlap_raw_std: np.ndarray,
    sierra_latitude: np.ndarray,
    sierra_longitude: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
    rows: List[SummaryRow],
    runtime,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "pacific_cobe2_eof": (
                ("mode", "pacific_latitude", "pacific_longitude"),
                pacific["eof"].astype(np.float32),
            ),
            "pacific_cobe2_pc": (("time", "mode"), pc_overlap.astype(np.float32)),
            "pacific_cobe2_pc_std_raw": (
                ("mode",),
                pc_overlap_raw_std.astype(np.float32),
            ),
            "pacific_cobe2_pc_mean_raw": (
                ("mode",),
                pc_overlap_raw_mean.astype(np.float32),
            ),
            "sierra_era5_t2m_beta": (
                ("mode", "sierra_latitude", "sierra_longitude"),
                beta_maps.astype(np.float32),
            ),
            "sierra_era5_t2m_corr": (
                ("mode", "sierra_latitude", "sierra_longitude"),
                corr_maps.astype(np.float32),
            ),
            "sierra_era5_t2m_r2": (
                ("mode", "sierra_latitude", "sierra_longitude"),
                r2_maps.astype(np.float32),
            ),
            "sierra_area_weighted_mean_r2": (
                ("mode",),
                np.asarray([row.area_weighted_mean_r2 for row in rows], dtype=np.float32),
            ),
            "sierra_area_weighted_mean_corr": (
                ("mode",),
                np.asarray([row.area_weighted_mean_corr for row in rows], dtype=np.float32),
            ),
            "sierra_max_local_r2": (
                ("mode",),
                np.asarray([row.max_local_r2 for row in rows], dtype=np.float32),
            ),
            "explained_variance_ratio": (("mode",), pacific["explained_variance_ratio"].astype(np.float32)),
            "singular_value": (("mode",), pacific["singular_value"].astype(np.float32)),
            "pacific_valid_mask": (
                ("pacific_latitude", "pacific_longitude"),
                pacific["valid_mask"],
            ),
        },
        coords={
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "time": overlap_months.astype("datetime64[ns]"),
            "pacific_latitude": pacific["latitude"].astype(np.float32),
            "pacific_longitude": pacific["longitude"].astype(np.float32),
            "sierra_latitude": sierra_latitude.astype(np.float32),
            "sierra_longitude": sierra_longitude.astype(np.float32),
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "pacific_sst_region_360": json.dumps(PACIFIC_SST_REGION_360.as_dict()),
            "sierra_t2m_region_360": json.dumps(SIERRA_T2M_REGION_360.as_dict()),
            "formula_corr": "rho_k(r) = corr(a_k(t), Y(t, r))",
            "formula_r2": "R_k^2(r) = rho_k(r)^2",
            "formula_beta": "beta_k(r) = sum_t[a_k(t) * Y(t, r)] / sum_t[a_k(t)^2]",
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "pc_sign_flips_applied": json.dumps([float(value) for value in MODE_SIGN.tolist()]),
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
            "sierra_era5_t2m_beta": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
            "sierra_era5_t2m_corr": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
            "sierra_era5_t2m_r2": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
            "pacific_cobe2_eof": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
        },
    )


def save_summary(rows: List[SummaryRow], pacific: Dict[str, np.ndarray], overlap_months: np.ndarray, sierra_shape: List[int], runtime) -> None:
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "explained_variance_ratio",
                "pc_mean_raw",
                "pc_std_raw",
                "sierra_area_weighted_mean_r2",
                "sierra_area_weighted_mean_corr",
                "sierra_max_local_r2",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.mode,
                    "{:.12g}".format(row.explained_variance_ratio),
                    "{:.12g}".format(row.pc_mean_raw),
                    "{:.12g}".format(row.pc_std_raw),
                    "{:.12g}".format(row.area_weighted_mean_r2),
                    "{:.12g}".format(row.area_weighted_mean_corr),
                    "{:.12g}".format(row.max_local_r2),
                ]
            )

    payload = SummaryPayload(
        experiment_name=EXPERIMENT_NAME,
        pacific_sst_region=PACIFIC_SST_REGION_360.as_dict(),
        sierra_t2m_region=SIERRA_T2M_REGION_360.as_dict(),
        input_cobe2_sst_path=str(COBE2_SST_FILE),
        input_era5_monthly_mean_path=str(ERA5_MONTHLY_MEAN_FILE),
        input_era5_monthly_climatology_path=str(ERA5_MONTHLY_CLIM_FILE),
        output_netcdf_path=str(NETCDF_FILE),
        output_figure_path=str(FIGURE_FILE),
        overlap_start=format_month(overlap_months[0]),
        overlap_end=format_month(overlap_months[-1]),
        n_overlap_months=int(overlap_months.size),
        n_modes=N_MODES,
        pc_standardized=True,
        mode_signs=[float(value) for value in MODE_SIGN.tolist()],
        pacific_sst_shape=[int(pacific["latitude"].size), int(pacific["longitude"].size)],
        sierra_t2m_shape=sierra_shape,
        slurm_job_id=runtime.slurm_job_id,
        compute_node=runtime.hostname,
        summary_rows=[asdict(row) for row in rows],
    )
    summary = asdict(payload)
    summary["output_directory_size"] = output_dir_size_text()
    SUMMARY_JSON_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def diagnostic_text_box() -> str:
    return "\n".join(
        [
            r"$\beta_k(r)$ is the physical T2m response pattern over the matched Pacific ERA5-Land region.",
            r"$\beta_k(r)=\frac{\sum_t a_k(t)Y(t,r)}{\sum_t a_k(t)^2}$, "
            r"where $Y(t,r)$ is the ERA5-Land T2m anomaly at time $t$ and matched-region land grid point $r$, "
            r"and $a_k(t)$ is COBE2 Pacific SST PC $k$ at time $t$.",
            r"PCs are standardized here, so $\beta_k(r)$ is in K per one-standard-deviation increase in PC$k$.",
            r"$k$: PC index, for example $k=1$ means PC1. "
            r"$t$: time index, for example one month or one season. "
            r"$r$: one matched-region ERA5-Land grid point.",
            r"$\rho_k(r)=\operatorname{corr}(a_k(t),Y(t,r))$ is the correlation between COBE2 Pacific SST PC $k$ "
            r"and matched-region T2m at grid point $r$.",
            r"$R_k^2(r)=\rho_k(r)^2$ is the fraction of local matched-region T2m variance explained by perfect knowledge of SST PC $k$.",
        ]
    )


def load_plot_inputs_from_netcdf() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[SummaryRow]]:
    with xr.open_dataset(NETCDF_FILE, engine="netcdf4", decode_times=True) as ds:
        sierra_latitude = np.asarray(ds["sierra_latitude"].values, dtype=np.float64)
        sierra_longitude = np.asarray(ds["sierra_longitude"].values, dtype=np.float64)
        beta_maps = np.asarray(ds["sierra_era5_t2m_beta"].values, dtype=np.float64)
        corr_maps = np.asarray(ds["sierra_era5_t2m_corr"].values, dtype=np.float64)
        r2_maps = np.asarray(ds["sierra_era5_t2m_r2"].values, dtype=np.float64)
        explained_variance_ratio = np.asarray(ds["explained_variance_ratio"].values, dtype=np.float64)
        pc_std_raw = np.asarray(ds["pacific_cobe2_pc_std_raw"].values, dtype=np.float64)
        pc_mean_raw = np.asarray(ds["pacific_cobe2_pc_mean_raw"].values, dtype=np.float64)
        mean_r2 = np.asarray(ds["sierra_area_weighted_mean_r2"].values, dtype=np.float64)
        mean_corr = np.asarray(ds["sierra_area_weighted_mean_corr"].values, dtype=np.float64)
        max_local_r2 = np.asarray(ds["sierra_max_local_r2"].values, dtype=np.float64)

    rows = [
        SummaryRow(
            mode=mode_index + 1,
            explained_variance_ratio=float(explained_variance_ratio[mode_index]),
            pc_std_raw=float(pc_std_raw[mode_index]),
            pc_mean_raw=float(pc_mean_raw[mode_index]),
            area_weighted_mean_r2=float(mean_r2[mode_index]),
            area_weighted_mean_corr=float(mean_corr[mode_index]),
            max_local_r2=float(max_local_r2[mode_index]),
        )
        for mode_index in range(N_MODES)
    ]
    return sierra_latitude, sierra_longitude, beta_maps, corr_maps, r2_maps, rows


def plot_maps(
    sierra_latitude: np.ndarray,
    sierra_longitude: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
    rows: List[SummaryRow],
) -> None:
    fig, axes = plt.subplots(N_MODES, 3, figsize=(15, 24), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.17, hspace=0.52, wspace=0.26)
    lon2d, lat2d = np.meshgrid(sierra_longitude, sierra_latitude)
    for mode_index in range(N_MODES):
        beta_ax = axes[mode_index, 0]
        corr_ax = axes[mode_index, 1]
        r2_ax = axes[mode_index, 2]

        beta = np.asarray(beta_maps[mode_index], dtype=np.float64)
        beta_vmax = float(np.nanmax(np.abs(beta)))
        if not np.isfinite(beta_vmax) or beta_vmax == 0.0:
            beta_vmax = 1.0
        beta_mesh = beta_ax.pcolormesh(
            lon2d,
            lat2d,
            beta,
            cmap="RdBu_r",
            shading="auto",
            vmin=-beta_vmax,
            vmax=beta_vmax,
        )
        beta_ax.set_title(rf"PC{mode_index + 1} regression $\beta_{{{mode_index + 1}}}(r)$")
        beta_ax.set_ylabel("Latitude")
        fig.colorbar(beta_mesh, ax=beta_ax, shrink=0.9).set_label(r"$\beta_k(r)$ [K / 1$\sigma$ PC]")

        corr = np.asarray(corr_maps[mode_index], dtype=np.float64)
        corr_mesh = corr_ax.pcolormesh(
            lon2d,
            lat2d,
            corr,
            cmap="RdBu_r",
            shading="auto",
            vmin=-1.0,
            vmax=1.0,
        )
        corr_ax.set_title(
            rf"PC{mode_index + 1} correlation $\rho_{{{mode_index + 1}}}(r)$ | "
            rf"$\overline{{\rho}}={rows[mode_index].area_weighted_mean_corr:.3f}$"
        )
        fig.colorbar(corr_mesh, ax=corr_ax, shrink=0.9).set_label(r"$\rho_k(r)$")

        r2 = np.asarray(r2_maps[mode_index], dtype=np.float64)
        r2_mesh = r2_ax.pcolormesh(
            lon2d,
            lat2d,
            r2,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max(0.05, float(np.nanmax(r2))),
        )
        r2_ax.set_title(
            rf"PC{mode_index + 1} explained variance $R_{{{mode_index + 1}}}^2(r)$ | "
            rf"$\overline{{R^2}}={rows[mode_index].area_weighted_mean_r2:.3f}$"
        )
        fig.colorbar(r2_mesh, ax=r2_ax, shrink=0.9).set_label(r"$R_k^2(r)$")

        for ax in (beta_ax, corr_ax, r2_ax):
            ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
            ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Longitude")

    fig.text(
        0.5,
        0.055,
        diagnostic_text_box(),
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#666666"},
    )
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def plot_pacific_eofs(pacific: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.95, bottom=0.06, hspace=0.32, wspace=0.22)
    lon2d, lat2d = np.meshgrid(pacific["longitude"], pacific["latitude"])
    for mode_index, ax in enumerate(axes.flat):
        eof = np.asarray(pacific["eof"][mode_index], dtype=np.float64)
        vmax = float(np.nanmax(np.abs(eof)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            eof,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(
            rf"COBE2 Pacific EOF{mode_index + 1} | EVR={pacific['explained_variance_ratio'][mode_index]:.3f}"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(PACIFIC_SST_REGION_360.lon_min, PACIFIC_SST_REGION_360.lon_max)
        ax.set_ylim(PACIFIC_SST_REGION_360.lat_min, PACIFIC_SST_REGION_360.lat_max)
        fig.colorbar(mesh, ax=ax, shrink=0.86).set_label("EOF loading")
    fig.savefig(EOF_FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    plot_only = "--plot-only" in sys.argv[1:]
    ensure_output_dir()
    remove_if_exists(FIGURE_FILE)
    remove_if_exists(EOF_FIGURE_FILE)

    if plot_only:
        if not NETCDF_FILE.exists():
            raise FileNotFoundError(f"Plot-only mode requires existing NetCDF: {NETCDF_FILE}")
        sierra_latitude, sierra_longitude, beta_maps, corr_maps, r2_maps, rows = load_plot_inputs_from_netcdf()
        plot_maps(
            sierra_latitude=sierra_latitude,
            sierra_longitude=sierra_longitude,
            beta_maps=beta_maps,
            corr_maps=corr_maps,
            r2_maps=r2_maps,
            rows=rows,
        )
        with xr.open_dataset(NETCDF_FILE, engine="netcdf4", decode_times=True) as ds:
            pacific = {
                "latitude": np.asarray(ds["pacific_latitude"].values, dtype=np.float64),
                "longitude": np.asarray(ds["pacific_longitude"].values, dtype=np.float64),
                "eof": np.asarray(ds["pacific_cobe2_eof"].values, dtype=np.float64),
                "explained_variance_ratio": np.asarray(ds["explained_variance_ratio"].values, dtype=np.float64),
            }
        plot_pacific_eofs(pacific)
        print(f"Figure: {FIGURE_FILE}", flush=True)
        print(f"EOF Figure: {EOF_FIGURE_FILE}", flush=True)
        return

    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    remove_if_exists(NETCDF_FILE)
    remove_if_exists(SUMMARY_CSV_FILE)
    remove_if_exists(SUMMARY_JSON_FILE)

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
        pc_overlap_raw = np.stack([pacific["pc"][pacific_index[month], :] for month in overlap_months.tolist()], axis=0)
        pc_overlap, pc_overlap_raw_mean, pc_overlap_raw_std = standardize_pc_matrix(pc_overlap_raw)

        sierra_latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float64)
        sierra_longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float64)
        anomaly_slices: List[np.ndarray] = []

        print(
            "Computing matched-region ERA5-Land T2m Level 1 diagnostics for overlap "
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
        beta_maps, corr_maps, r2_maps, rows = compute_level1_maps(
            pc_overlap=pc_overlap,
            anomalies=anomalies,
            latitude=sierra_latitude,
            explained_variance_ratio=pacific["explained_variance_ratio"],
        )
        save_netcdf(
            pacific=pacific,
            overlap_months=overlap_months,
            pc_overlap=pc_overlap,
            sierra_latitude=sierra_latitude,
            sierra_longitude=sierra_longitude,
            beta_maps=beta_maps,
            corr_maps=corr_maps,
            r2_maps=r2_maps,
            rows=rows,
            runtime=runtime,
            pc_overlap_raw_mean=pc_overlap_raw_mean,
            pc_overlap_raw_std=pc_overlap_raw_std,
        )
        save_summary(
            rows=rows,
            pacific=pacific,
            overlap_months=overlap_months,
            sierra_shape=[int(sierra_latitude.size), int(sierra_longitude.size)],
            runtime=runtime,
        )
        plot_maps(
            sierra_latitude=sierra_latitude,
            sierra_longitude=sierra_longitude,
            beta_maps=beta_maps,
            corr_maps=corr_maps,
            r2_maps=r2_maps,
            rows=rows,
        )
        plot_pacific_eofs(pacific)
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Figure: {FIGURE_FILE}", flush=True)
    print(f"EOF Figure: {EOF_FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
