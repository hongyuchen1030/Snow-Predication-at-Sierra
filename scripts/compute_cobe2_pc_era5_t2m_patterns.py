#!/usr/bin/env python3
"""
Compute COBE2 PC-weighted ERA5-Land T2M spatial patterns directly from
monthly mean and monthly climatology files, without writing the full ERA5
monthly anomaly cube.
"""

from __future__ import annotations

import json
import os
import sys
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

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    ensure_runtime_on_compute_node,
    get_runtime,
    open_dataset_with_fallbacks,
)


PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PC_ERA5_T2M_PATTERNS_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pc_era5_t2m_patterns",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pc_era5_t2m_patterns"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pc_era5_t2m_patterns.nc"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pc_era5_t2m_patterns_summary.json"
FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pc_era5_t2m_patterns_modes1to6.png"

ERA5_MONTHLY_MEAN_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5land_t2m_monthly_anomalies/era5land_t2m_monthly_mean.nc"
)
ERA5_MONTHLY_CLIM_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5land_t2m_monthly_anomalies/era5land_t2m_monthly_climatology.nc"
)
COBE2_EOF_FILE = Path(
    "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/sst_pca/cobe2_global_monthly_climatology_anomaly/cobe2_global_monthly_clim_sst_eofs.nc"
)

ERA5_VARIABLE = "t2m"
PC_VARIABLE = "pc"
N_MODES = 6
TIME_CHUNK = 12
LAT_CHUNK = 180
LON_CHUNK = 360
MODE_SIGN = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class SummaryPayload:
    input_monthly_mean_path: str
    input_monthly_climatology_path: str
    input_cobe2_pc_path: str
    output_netcdf_path: str
    output_figure_path: str
    formula_implemented: str
    full_monthly_anomaly_saved: bool
    overlap_start: str
    overlap_end: str
    n_overlap_months: int
    mode_signs: List[float]
    pc_std: List[float]
    era5_spatial_shape: List[int]
    units: str
    slurm_job_id: str
    compute_node: str


def ensure_output_dir() -> None:
    PSCRATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HOME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def load_cobe2_reference() -> Dict[str, np.ndarray]:
    with open_dataset_with_fallbacks(COBE2_EOF_FILE) as ds:
        time = to_month_start(np.asarray(ds["time"].values, dtype="datetime64[ns]"))
        pc = np.asarray(ds[PC_VARIABLE].values[:, :N_MODES], dtype=np.float64) * MODE_SIGN[np.newaxis, :]
        eof = np.asarray(ds["eof"].values[:N_MODES, :, :], dtype=np.float64) * MODE_SIGN[:, np.newaxis, np.newaxis]
        eof_lat = np.asarray(ds["lat"].values, dtype=np.float64)
        eof_lon = np.asarray(ds["lon"].values, dtype=np.float64)
        valid_mask = (
            np.asarray(ds["valid_mask"].values, dtype=bool)
            if "valid_mask" in ds
            else np.isfinite(eof[0])
        )
        explained_variance_ratio = (
            np.asarray(ds["explained_variance_ratio"].values[:N_MODES], dtype=np.float64)
            if "explained_variance_ratio" in ds
            else np.full(N_MODES, np.nan, dtype=np.float64)
        )
        singular_value = (
            np.asarray(ds["singular_value"].values[:N_MODES], dtype=np.float64)
            if "singular_value" in ds
            else np.full(N_MODES, np.nan, dtype=np.float64)
        )
    return {
        "time": time,
        "pc": pc,
        "eof": eof,
        "eof_lat": eof_lat,
        "eof_lon": eof_lon,
        "valid_mask": valid_mask,
        "explained_variance_ratio": explained_variance_ratio,
        "singular_value": singular_value,
    }


def build_time_index(month_values: np.ndarray) -> Dict[np.datetime64, int]:
    return {month: index for index, month in enumerate(month_values.tolist())}


def normalize_longitude_to_360(longitude: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrapped_lon = np.mod(np.asarray(longitude, dtype=np.float64), 360.0)
    sort_index = np.argsort(wrapped_lon)
    return wrapped_lon[sort_index], np.take(field, sort_index, axis=-1)


def add_cyclic_longitude_column(longitude: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cyclic_longitude = np.concatenate([np.asarray(longitude, dtype=np.float64), [float(longitude[0]) + 360.0]])
    cyclic_field = np.concatenate([field, field[..., :1]], axis=-1)
    return cyclic_longitude, cyclic_field


def recenter_field_to_360(
    longitude: np.ndarray,
    field: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    shifted_longitude, shifted_field = normalize_longitude_to_360(longitude, field)
    shifted_mask = None
    if mask is not None:
        _, shifted_mask = normalize_longitude_to_360(longitude, np.asarray(mask, dtype=np.float64))
    shifted_longitude, shifted_field = add_cyclic_longitude_column(shifted_longitude, shifted_field)
    if shifted_mask is not None:
        _, shifted_mask = add_cyclic_longitude_column(shifted_longitude[:-1], shifted_mask)
    return shifted_longitude, shifted_field, shifted_mask


def save_outputs(
    overlap_months: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    pattern_values: np.ndarray,
    pc_std: np.ndarray,
    explained_variance_ratio: np.ndarray,
    singular_value: np.ndarray,
    units: str,
    runtime,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "cobe2_pc_era5_t2m_pattern": (
                ("mode", "latitude", "longitude"),
                pattern_values.astype(np.float32),
            ),
            "cobe2_pc_std": (("mode",), pc_std.astype(np.float32)),
            "explained_variance_ratio": (("mode",), explained_variance_ratio.astype(np.float32)),
            "singular_value": (("mode",), singular_value.astype(np.float32)),
            "n_overlap_months": ((), np.int32(overlap_months.size)),
        },
        coords={
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "latitude": latitude.astype(np.float32),
            "longitude": longitude.astype(np.float32),
        },
        attrs={
            "description": "COBE2 PC-weighted ERA5-Land monthly T2M spatial patterns",
            "formula": "COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))",
            "cobe2_pc_time_start": format_month(overlap_months[0]),
            "cobe2_pc_time_end": format_month(overlap_months[-1]),
            "units": units,
            "full_monthly_anomaly_saved": "false",
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    encoding = {
        "cobe2_pc_era5_t2m_pattern": {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "chunksizes": [N_MODES, LAT_CHUNK, LON_CHUNK],
            "_FillValue": np.float32(np.nan),
        },
        "cobe2_pc_std": {"dtype": "float32"},
        "explained_variance_ratio": {"dtype": "float32"},
        "singular_value": {"dtype": "float32"},
        "n_overlap_months": {"dtype": "int32"},
    }
    ds.to_netcdf(NETCDF_FILE, engine="netcdf4", encoding=encoding)


def save_summary(
    overlap_months: np.ndarray,
    pc_std: np.ndarray,
    spatial_shape: List[int],
    units: str,
    runtime,
) -> None:
    payload = SummaryPayload(
        input_monthly_mean_path=str(ERA5_MONTHLY_MEAN_FILE),
        input_monthly_climatology_path=str(ERA5_MONTHLY_CLIM_FILE),
        input_cobe2_pc_path=str(COBE2_EOF_FILE),
        output_netcdf_path=str(NETCDF_FILE),
        output_figure_path=str(FIGURE_FILE),
        formula_implemented="COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))",
        full_monthly_anomaly_saved=False,
        overlap_start=format_month(overlap_months[0]),
        overlap_end=format_month(overlap_months[-1]),
        n_overlap_months=int(overlap_months.size),
        mode_signs=[float(value) for value in MODE_SIGN.tolist()],
        pc_std=[float(value) for value in pc_std.tolist()],
        era5_spatial_shape=spatial_shape,
        units=units,
        slurm_job_id=runtime.slurm_job_id,
        compute_node=runtime.hostname,
    )
    summary = asdict(payload)
    summary["output_directory_size"] = output_dir_size_text()
    SUMMARY_JSON_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_patterns(
    pattern_values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    overlap_months: np.ndarray,
    pc_overlap: np.ndarray,
    eof_values: np.ndarray,
    eof_latitude: np.ndarray,
    eof_longitude: np.ndarray,
    eof_valid_mask: np.ndarray,
    explained_variance_ratio: np.ndarray,
    units: str,
) -> None:
    fig, axes = plt.subplots(
        N_MODES,
        3,
        figsize=(23, 24),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4.5, 4.2, 1.7]},
    )
    overlap_time = overlap_months.astype("datetime64[D]").astype(object)
    lon_min = 0.0
    lon_max = 360.0
    lat_min = float(min(np.nanmin(np.asarray(latitude, dtype=np.float64)), np.nanmin(np.asarray(eof_latitude, dtype=np.float64))))
    lat_max = float(max(np.nanmax(np.asarray(latitude, dtype=np.float64)), np.nanmax(np.asarray(eof_latitude, dtype=np.float64))))
    for mode_index in range(N_MODES):
        pattern_ax = axes[mode_index, 0]
        eof_ax = axes[mode_index, 1]
        ts_ax = axes[mode_index, 2]
        field = np.asarray(pattern_values[mode_index], dtype=np.float64)
        era5_plot_longitude, field, _ = recenter_field_to_360(longitude, field)
        era5_lon2d, era5_lat2d = np.meshgrid(era5_plot_longitude, latitude)
        vmax = float(np.nanmax(np.abs(field)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = pattern_ax.pcolormesh(
            era5_lon2d,
            era5_lat2d,
            field,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        evr = float(explained_variance_ratio[mode_index])
        title = f"COBE2 PC{mode_index + 1} weighted ERA5 T2M pattern"
        if np.isfinite(evr):
            title += f" | EVR={evr:.3f}"
        pattern_ax.set_title(title)
        pattern_ax.set_xlabel("Longitude")
        pattern_ax.set_ylabel("Latitude")
        pattern_ax.set_xlim(lon_min, lon_max)
        pattern_ax.set_ylim(lat_min, lat_max)
        pattern_ax.set_aspect("equal", adjustable="box")
        colorbar = fig.colorbar(mesh, ax=pattern_ax, shrink=0.9)
        colorbar.set_label(units)

        eof_field = np.asarray(eof_values[mode_index], dtype=np.float64)
        eof_plot_longitude, eof_field, eof_mask = recenter_field_to_360(
            eof_longitude,
            eof_field,
            mask=eof_valid_mask,
        )
        eof_field = np.where(eof_mask >= 0.5, eof_field, np.nan)
        eof_lon2d, eof_lat2d = np.meshgrid(eof_plot_longitude, eof_latitude)
        eof_vmax = float(np.nanmax(np.abs(eof_field)))
        if not np.isfinite(eof_vmax) or eof_vmax == 0.0:
            eof_vmax = 1.0
        eof_mesh = eof_ax.pcolormesh(
            eof_lon2d,
            eof_lat2d,
            eof_field,
            cmap="RdBu_r",
            shading="auto",
            vmin=-eof_vmax,
            vmax=eof_vmax,
        )
        eof_ax.set_title(f"COBE2 EOF{mode_index + 1}")
        eof_ax.set_xlabel("Longitude")
        eof_ax.set_ylabel("Latitude")
        eof_ax.set_xlim(lon_min, lon_max)
        eof_ax.set_ylim(lat_min, lat_max)
        eof_ax.set_aspect("equal", adjustable="box")
        eof_colorbar = fig.colorbar(eof_mesh, ax=eof_ax, shrink=0.9)
        eof_colorbar.set_label("EOF loading")

        pc_series = np.asarray(pc_overlap[:, mode_index], dtype=np.float64)
        ts_ax.plot(overlap_time, pc_series, color="black", linewidth=1.0)
        ts_ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ts_ax.set_title(f"COBE2 PC{mode_index + 1} time series")
        ts_ax.set_ylabel("PC value")
        ts_ax.grid(True, alpha=0.25, linewidth=0.5)
        if mode_index == N_MODES - 1:
            ts_ax.set_xlabel("Time")
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def load_saved_patterns() -> Dict[str, np.ndarray | str]:
    with xr.open_dataset(NETCDF_FILE, engine="netcdf4", decode_times=True) as ds:
        return {
            "pattern_values": np.asarray(ds["cobe2_pc_era5_t2m_pattern"].values, dtype=np.float64)
            * MODE_SIGN[:, np.newaxis, np.newaxis],
            "latitude": np.asarray(ds["latitude"].values, dtype=np.float64),
            "longitude": np.asarray(ds["longitude"].values, dtype=np.float64),
            "units": str(ds.attrs.get("units", "K anomaly")),
        }


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()
    remove_if_exists(FIGURE_FILE)
    plot_only = "--plot-only" in sys.argv[1:]

    cobe2 = load_cobe2_reference()
    pc_time = cobe2["time"]
    pc_values = cobe2["pc"]
    eof_values = cobe2["eof"]
    eof_latitude = cobe2["eof_lat"]
    eof_longitude = cobe2["eof_lon"]
    eof_valid_mask = cobe2["valid_mask"]
    explained_variance_ratio = cobe2["explained_variance_ratio"]
    singular_value = cobe2["singular_value"]

    monthly_mean_ds = xr.open_dataset(
        ERA5_MONTHLY_MEAN_FILE,
        engine="netcdf4",
        chunks={"time": TIME_CHUNK, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )

    monthly_clim_ds = None if plot_only else xr.open_dataset(
        ERA5_MONTHLY_CLIM_FILE,
        engine="netcdf4",
        chunks={"month": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )

    try:
        monthly_mean = monthly_mean_ds[ERA5_VARIABLE]
        era5_time = to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        overlap_months = np.intersect1d(pc_time, era5_time, assume_unique=False)
        if overlap_months.size == 0:
            raise ValueError("No overlapping months between COBE2 PC time and ERA5 monthly mean time")

        pc_index = build_time_index(pc_time)
        era5_index = build_time_index(era5_time)
        pc_overlap = np.stack([pc_values[pc_index[month], :] for month in overlap_months.tolist()], axis=0)
        if plot_only:
            saved = load_saved_patterns()
            pattern_values = saved["pattern_values"]
            latitude = saved["latitude"]
            longitude = saved["longitude"]
            units = saved["units"]
            print(f"Plot-only mode: reusing saved patterns from {NETCDF_FILE}", flush=True)
        else:
            remove_if_exists(NETCDF_FILE)
            remove_if_exists(SUMMARY_JSON_FILE)
            monthly_clim = monthly_clim_ds[ERA5_VARIABLE]
            pc_std = np.std(pc_overlap, axis=0, ddof=1)
            if not np.isfinite(pc_std).all() or np.any(pc_std == 0.0):
                raise ValueError(f"Invalid PC standard deviation values: {pc_std}")

            latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float64)
            longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float64)
            pattern_sum = np.zeros((N_MODES, latitude.size, longitude.size), dtype=np.float64)
            units = str(monthly_mean.attrs.get("units", "K anomaly"))

            print(
                "Computing direct COBE2 PC-weighted ERA5 T2M patterns for overlap "
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
                latitude=latitude,
                longitude=longitude,
                pattern_values=pattern_values,
                pc_std=pc_std,
                explained_variance_ratio=explained_variance_ratio,
                singular_value=singular_value,
                units=units,
                runtime=runtime,
            )
            save_summary(
                overlap_months=overlap_months,
                pc_std=pc_std,
                spatial_shape=[int(latitude.size), int(longitude.size)],
                units=units,
                runtime=runtime,
            )
        plot_patterns(
            pattern_values=pattern_values,
            latitude=latitude,
            longitude=longitude,
            overlap_months=overlap_months,
            pc_overlap=pc_overlap,
            eof_values=eof_values,
            eof_latitude=eof_latitude,
            eof_longitude=eof_longitude,
            eof_valid_mask=eof_valid_mask,
            explained_variance_ratio=explained_variance_ratio,
            units=units,
        )
    finally:
        monthly_mean_ds.close()
        if monthly_clim_ds is not None:
            monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Figure: {FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
