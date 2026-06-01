#!/usr/bin/env python3
"""
Compute regional COBE2 EOF/PC modes and COBE2-PC-weighted ERA5-Land T2M patterns
over the project's Pacific-Sierra domain.

This reruns both steps regionally:
1. PCA / EOF on monthly-climatology COBE2 SST anomalies over the Pacific-Sierra region.
2. Formula-based COBE2-PC-weighted ERA5 T2M pattern over the same region:
   COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))
"""

import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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
    normalize_longitude_to_minus180_180,
    open_dataset_with_fallbacks,
)
from snow_ml.data import DEFAULT_MODEL_REGION, RegionBounds


PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_PC_ERA5_T2M_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_pc_era5_t2m_patterns",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_pc_era5_t2m_patterns"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_pc_era5_t2m_patterns.nc"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_pc_era5_t2m_patterns_summary.json"
FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_pc_era5_t2m_patterns_modes1to6.png"

ERA5_MONTHLY_MEAN_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5land_t2m_monthly_anomalies/era5land_t2m_monthly_mean.nc"
)
ERA5_MONTHLY_CLIM_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5land_t2m_monthly_anomalies/era5land_t2m_monthly_climatology.nc"
)

REGION = DEFAULT_MODEL_REGION
ERA5_VARIABLE = "t2m"
COBE2_VARIABLE = "sst"
N_MODES = 6
TIME_CHUNK = 12
LAT_CHUNK = 180
LON_CHUNK = 360
MODE_SIGN = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class SummaryPayload:
    region: Dict[str, float]
    input_monthly_mean_path: str
    input_monthly_climatology_path: str
    input_cobe2_sst_path: str
    output_netcdf_path: str
    output_figure_path: str
    formula_implemented: str
    overlap_start: str
    overlap_end: str
    n_overlap_months: int
    mode_signs: List[float]
    pc_std: List[float]
    explained_variance_ratio: List[float]
    era5_spatial_shape: List[int]
    cobe2_spatial_shape: List[int]
    units: str
    slurm_job_id: str
    compute_node: str


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


def subset_lat_lon_3d(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    region: RegionBounds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = normalize_longitude_to_minus180_180(np.asarray(longitude, dtype=np.float64))
    lon_sort = np.argsort(lon)
    lon_sorted = lon[lon_sort]
    values_sorted = np.take(values, lon_sort, axis=-1)

    lat_mask = (lat >= region.lat_min) & (lat <= region.lat_max)
    lon_mask = (lon_sorted >= region.lon_min) & (lon_sorted <= region.lon_max)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(f"Region {region.as_dict()} does not overlap source grid")
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    return lat[lat_idx], lon_sorted[lon_idx], values_sorted[:, lat_idx, :][:, :, lon_idx]


def solve_weighted_regional_eofs(
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
    n_valid = int(valid_mask_flat.sum())
    if n_valid < N_MODES:
        raise ValueError(f"Need at least {N_MODES} all-time-finite ocean cells, got {n_valid}")

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
    weighted_eofs_valid = np.zeros((N_MODES, n_valid), dtype=np.float64)
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
        "eof": eof_grid.astype(np.float64),
        "pc": pcs.astype(np.float64),
        "valid_mask": valid_mask_2d,
        "climatology": climatology.astype(np.float64),
        "singular_value": singular_values_all[:N_MODES].astype(np.float64),
        "explained_variance_ratio": explained_variance_ratio[:N_MODES].astype(np.float64),
    }


def load_regional_cobe2_reference(region: RegionBounds) -> Dict[str, np.ndarray]:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        values = np.asarray(ds[COBE2_VARIABLE].values, dtype=np.float64)
        missing_value = float(ds[COBE2_VARIABLE].attrs.get("missing_value", 1.0e20))
    values = np.where(values >= missing_value, np.nan, values)
    lat_crop, lon_crop, values_crop = subset_lat_lon_3d(latitude, longitude, values, region)
    return solve_weighted_regional_eofs(time, lat_crop, lon_crop, values_crop)


def subset_era5_region(field: xr.DataArray, region: RegionBounds) -> xr.DataArray:
    lon_name = "longitude"
    lat_name = "latitude"
    wrapped_lon = xr.where(field[lon_name] > 180.0, field[lon_name] - 360.0, field[lon_name])
    subset = field.assign_coords({lon_name: wrapped_lon}).sortby(lat_name).sortby(lon_name)
    return subset.sel(
        {
            lat_name: slice(region.lat_min, region.lat_max),
            lon_name: slice(region.lon_min, region.lon_max),
        }
    )


def save_outputs(
    overlap_months: np.ndarray,
    cobe2: Dict[str, np.ndarray],
    era5_latitude: np.ndarray,
    era5_longitude: np.ndarray,
    pattern_values: np.ndarray,
    pc_std: np.ndarray,
    units: str,
    runtime,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "cobe2_pc_era5_t2m_pattern": (
                ("mode", "era5_latitude", "era5_longitude"),
                pattern_values.astype(np.float32),
            ),
            "cobe2_eof": (("mode", "cobe2_latitude", "cobe2_longitude"), cobe2["eof"].astype(np.float32)),
            "cobe2_pc": (("time", "mode"), cobe2["pc"].astype(np.float32)),
            "cobe2_pc_std": (("mode",), pc_std.astype(np.float32)),
            "explained_variance_ratio": (("mode",), cobe2["explained_variance_ratio"].astype(np.float32)),
            "singular_value": (("mode",), cobe2["singular_value"].astype(np.float32)),
            "cobe2_valid_mask": (("cobe2_latitude", "cobe2_longitude"), cobe2["valid_mask"]),
            "n_overlap_months": ((), np.int32(overlap_months.size)),
        },
        coords={
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "time": cobe2["time"].astype("datetime64[ns]"),
            "era5_latitude": era5_latitude.astype(np.float32),
            "era5_longitude": era5_longitude.astype(np.float32),
            "cobe2_latitude": cobe2["latitude"].astype(np.float32),
            "cobe2_longitude": cobe2["longitude"].astype(np.float32),
        },
        attrs={
            "description": "Pacific-Sierra regional COBE2 EOF/PC and COBE2-PC-weighted ERA5-Land T2M patterns",
            "formula": "COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))",
            "region": json.dumps(REGION.as_dict()),
            "cobe2_pc_time_start": format_month(overlap_months[0]),
            "cobe2_pc_time_end": format_month(overlap_months[-1]),
            "units": units,
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(
        NETCDF_FILE,
        engine="netcdf4",
        encoding={
            "cobe2_pc_era5_t2m_pattern": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            },
            "cobe2_eof": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            },
        },
    )


def save_summary(
    overlap_months: np.ndarray,
    cobe2: Dict[str, np.ndarray],
    pc_std: np.ndarray,
    era5_shape: List[int],
    units: str,
    runtime,
) -> None:
    payload = SummaryPayload(
        region=REGION.as_dict(),
        input_monthly_mean_path=str(ERA5_MONTHLY_MEAN_FILE),
        input_monthly_climatology_path=str(ERA5_MONTHLY_CLIM_FILE),
        input_cobe2_sst_path=str(COBE2_SST_FILE),
        output_netcdf_path=str(NETCDF_FILE),
        output_figure_path=str(FIGURE_FILE),
        formula_implemented="COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))",
        overlap_start=format_month(overlap_months[0]),
        overlap_end=format_month(overlap_months[-1]),
        n_overlap_months=int(overlap_months.size),
        mode_signs=[float(value) for value in MODE_SIGN.tolist()],
        pc_std=[float(value) for value in pc_std.tolist()],
        explained_variance_ratio=[float(value) for value in cobe2["explained_variance_ratio"].tolist()],
        era5_spatial_shape=era5_shape,
        cobe2_spatial_shape=[int(cobe2["latitude"].size), int(cobe2["longitude"].size)],
        units=units,
        slurm_job_id=runtime.slurm_job_id,
        compute_node=runtime.hostname,
    )
    summary = asdict(payload)
    summary["output_directory_size"] = output_dir_size_text()
    SUMMARY_JSON_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_patterns(
    pattern_values: np.ndarray,
    era5_latitude: np.ndarray,
    era5_longitude: np.ndarray,
    cobe2: Dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(
        N_MODES,
        3,
        figsize=(18, 22),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.4, 2.4, 1.5]},
    )
    overlap_time = cobe2["time"].astype("datetime64[D]").astype(object)
    era5_lon2d, era5_lat2d = np.meshgrid(era5_longitude, era5_latitude)
    cobe2_lon2d, cobe2_lat2d = np.meshgrid(cobe2["longitude"], cobe2["latitude"])
    lon_min = REGION.lon_min
    lon_max = REGION.lon_max
    lat_min = REGION.lat_min
    lat_max = REGION.lat_max

    for mode_index in range(N_MODES):
        pattern_ax = axes[mode_index, 0]
        eof_ax = axes[mode_index, 1]
        ts_ax = axes[mode_index, 2]

        pattern = np.asarray(pattern_values[mode_index], dtype=np.float64)
        vmax = float(np.nanmax(np.abs(pattern)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = pattern_ax.pcolormesh(
            era5_lon2d,
            era5_lat2d,
            pattern,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        pattern_ax.set_title(
            f"Pacific-Sierra COBE2 PC{mode_index + 1} weighted ERA5 T2M | "
            f"EVR={float(cobe2['explained_variance_ratio'][mode_index]):.3f}"
        )
        pattern_ax.set_xlabel("Longitude")
        pattern_ax.set_ylabel("Latitude")
        pattern_ax.set_xlim(lon_min, lon_max)
        pattern_ax.set_ylim(lat_min, lat_max)
        pattern_ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=pattern_ax, shrink=0.9).set_label("K")

        eof = np.asarray(cobe2["eof"][mode_index], dtype=np.float64)
        eof = np.where(cobe2["valid_mask"], eof, np.nan)
        eof_vmax = float(np.nanmax(np.abs(eof)))
        if not np.isfinite(eof_vmax) or eof_vmax == 0.0:
            eof_vmax = 1.0
        eof_mesh = eof_ax.pcolormesh(
            cobe2_lon2d,
            cobe2_lat2d,
            eof,
            cmap="RdBu_r",
            shading="auto",
            vmin=-eof_vmax,
            vmax=eof_vmax,
        )
        eof_ax.set_title(f"Pacific-Sierra COBE2 EOF{mode_index + 1}")
        eof_ax.set_xlabel("Longitude")
        eof_ax.set_ylabel("Latitude")
        eof_ax.set_xlim(lon_min, lon_max)
        eof_ax.set_ylim(lat_min, lat_max)
        eof_ax.set_aspect("equal", adjustable="box")
        fig.colorbar(eof_mesh, ax=eof_ax, shrink=0.9).set_label("EOF loading")

        ts_ax.plot(overlap_time, cobe2["pc"][:, mode_index], color="black", linewidth=1.0)
        ts_ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ts_ax.set_title(f"Pacific-Sierra COBE2 PC{mode_index + 1}")
        ts_ax.set_ylabel("PC value")
        ts_ax.grid(True, alpha=0.25, linewidth=0.5)
        if mode_index == N_MODES - 1:
            ts_ax.set_xlabel("Time")

    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()
    remove_if_exists(NETCDF_FILE)
    remove_if_exists(SUMMARY_JSON_FILE)
    remove_if_exists(FIGURE_FILE)

    cobe2 = load_regional_cobe2_reference(REGION)

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
        monthly_mean = subset_era5_region(monthly_mean_ds[ERA5_VARIABLE], REGION)
        monthly_clim = subset_era5_region(monthly_clim_ds[ERA5_VARIABLE], REGION)
        era5_time = to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        overlap_months = np.intersect1d(cobe2["time"], era5_time, assume_unique=False)
        if overlap_months.size == 0:
            raise ValueError("No overlapping months between regional COBE2 PC time and ERA5 monthly mean time")

        cobe2_index = build_time_index(cobe2["time"])
        era5_index = build_time_index(era5_time)
        pc_overlap = np.stack([cobe2["pc"][cobe2_index[month], :] for month in overlap_months.tolist()], axis=0)
        pc_std = np.std(pc_overlap, axis=0, ddof=1)
        if not np.isfinite(pc_std).all() or np.any(pc_std == 0.0):
            raise ValueError(f"Invalid PC standard deviation values: {pc_std}")

        era5_latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float64)
        era5_longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float64)
        pattern_sum = np.zeros((N_MODES, era5_latitude.size, era5_longitude.size), dtype=np.float64)
        units = str(monthly_mean.attrs.get("units", "K anomaly"))

        print(
            "Computing Pacific-Sierra regional COBE2 PC-weighted ERA5 T2M patterns for overlap "
            f"{format_month(overlap_months[0])} to {format_month(overlap_months[-1])} "
            f"({int(overlap_months.size)} months)",
            flush=True,
        )
        for step_index, month_value in enumerate(overlap_months.tolist(), start=1):
            era5_time_index = era5_index[month_value]
            monthly_mean_slice = monthly_mean.isel(time=era5_time_index)
            monthly_clim_slice = monthly_clim.sel(month=month_number(month_value))
            anomaly_slice = (monthly_mean_slice - monthly_clim_slice).astype(np.float64).load().values
            pattern_sum += pc_overlap[step_index - 1, :, np.newaxis, np.newaxis] * anomaly_slice[np.newaxis, :, :]
            if step_index == 1 or step_index % 120 == 0 or step_index == overlap_months.size:
                print(
                    f"  processed overlap month {step_index}/{int(overlap_months.size)}: {format_month(month_value)}",
                    flush=True,
                )

        pattern_values = pattern_sum / pc_std[:, np.newaxis, np.newaxis]
        save_outputs(
            overlap_months=overlap_months,
            cobe2=cobe2,
            era5_latitude=era5_latitude,
            era5_longitude=era5_longitude,
            pattern_values=pattern_values,
            pc_std=pc_std,
            units=units,
            runtime=runtime,
        )
        save_summary(
            overlap_months=overlap_months,
            cobe2=cobe2,
            pc_std=pc_std,
            era5_shape=[int(era5_latitude.size), int(era5_longitude.size)],
            units=units,
            runtime=runtime,
        )
        plot_patterns(
            pattern_values=pattern_values,
            era5_latitude=era5_latitude,
            era5_longitude=era5_longitude,
            cobe2=cobe2,
        )
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Figure: {FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
