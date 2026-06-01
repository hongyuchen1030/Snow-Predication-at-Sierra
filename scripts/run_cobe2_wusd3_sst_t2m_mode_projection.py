#!/usr/bin/env python3
"""
Run an additive COBE2/WUS-D3 SST-to-T2M mode projection diagnostic.

This script implements the following formulas exactly for modes 1, 2, and 3:

1. COBE2/ERA5 T2M pattern
   COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))

2. WUS-D3 projected PC using COBE2 EOF
   WUS_D3_PC_k(t) = sum_x [COBE2_EOF_k(x) * WUS_D3_SST_anom(t, x)]

3. WUS-D3 T2M pattern
   WUS_D3_T2M_k(x) = sum_t [WUS_D3_PC_k(t) * WUS_D3_T2M_anom(t, x)] / stddev(WUS_D3_PC_k(t))

No regression scaling, extra PC mean subtraction, or division by n_time is applied.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    compute_monthly_climatology_anomalies,
    ensure_runtime_on_compute_node,
    get_runtime,
    normalize_longitude_to_minus180_180,
    open_dataset_with_fallbacks,
)
from snow_ml.data_wusd3 import DEFAULT_WUSD3_DATASET_ID


EXPERIMENT_NAME = "cobe2_wusd3_sst_t2m_mode_projection"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "cobe2_wusd3_sst_t2m_mode_projection"
NETCDF_FILE = OUTPUT_DIR / "cobe2_wusd3_sst_t2m_mode_projection.nc"
SUMMARY_CSV_FILE = OUTPUT_DIR / "cobe2_wusd3_sst_t2m_mode_projection_summary.csv"
METHOD_SUMMARY_FILE = OUTPUT_DIR / "cobe2_wusd3_sst_t2m_mode_projection_method_summary.md"
FIGURE_FILE = OUTPUT_DIR / "cobe2_wusd3_sst_t2m_mode_projection_modes123.png"

COBE2_EOF_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "sst_pca"
    / "cobe2_global_monthly_climatology_anomaly"
    / "cobe2_global_monthly_clim_sst_eofs.nc"
)
ERA5_LAND_T2M_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5-Land/2m_temperature")
WUSD3_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily")
WRFINPUT_D01 = Path("/global/cfs/projectdirs/m3522/cmip6/WUS-D3/wrfinput_d01")

WUSD3_DOMAIN = "d01"
WUSD3_SST_VARIABLE = "tskin"
WUSD3_T2M_VARIABLE = "t2"
ERA5_T2M_VARIABLE = "t2m"
N_MODES = 3
ERA5_MARGIN_DEGREES = 0.5
@dataclass(frozen=True)
class GridData:
    latitude: np.ndarray
    longitude: np.ndarray
    land_mask: np.ndarray
    ocean_mask: np.ndarray


@dataclass(frozen=True)
class Cobe2Reference:
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    eof: np.ndarray
    pc: np.ndarray
    singular_value: np.ndarray
    explained_variance_ratio: np.ndarray
    valid_mask: np.ndarray
    source_file: str


@dataclass(frozen=True)
class MonthlyCube:
    time: np.ndarray
    values: np.ndarray
    units: str
    source_files: List[str]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_date(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def to_month_start(times: Sequence[np.datetime64]) -> np.ndarray:
    values = np.asarray(times, dtype="datetime64[ns]")
    return values.astype("datetime64[M]").astype("datetime64[ns]")


def month_number_array(times: Sequence[np.datetime64]) -> np.ndarray:
    return np.array(
        [int(np.datetime_as_string(value, unit="D")[5:7]) for value in np.asarray(times, dtype="datetime64[ns]")],
        dtype=np.int32,
    )


def intersect_months(*time_axes: Sequence[np.datetime64]) -> np.ndarray:
    common = to_month_start(time_axes[0])
    for axis in time_axes[1:]:
        common = np.intersect1d(common, to_month_start(axis), assume_unique=False)
    return np.asarray(common, dtype="datetime64[ns]")


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    time_values = to_month_start(time_values)
    index_by_month = {month: index for index, month in enumerate(time_values.tolist())}
    indices = [index_by_month[month] for month in target_months.tolist()]
    return np.asarray(data)[indices]


def load_wusd3_grid() -> GridData:
    with open_dataset_with_fallbacks(WRFINPUT_D01) as ds:
        latitude = np.asarray(ds["XLAT"].isel(Time=0).values, dtype=np.float64)
        longitude = np.asarray(ds["XLONG"].isel(Time=0).values, dtype=np.float64)
        land_mask_raw = np.asarray(ds["LANDMASK"].isel(Time=0).values, dtype=np.int8)
    land_mask = land_mask_raw == 1
    ocean_mask = ~land_mask
    return GridData(
        latitude=latitude,
        longitude=longitude,
        land_mask=land_mask,
        ocean_mask=ocean_mask,
    )


def load_cobe2_reference() -> Cobe2Reference:
    with open_dataset_with_fallbacks(COBE2_EOF_FILE) as ds:
        eof = np.asarray(ds["eof"].values[:N_MODES], dtype=np.float64)
        pc = np.asarray(ds["pc"].values[:, :N_MODES], dtype=np.float64)
        singular_value = np.asarray(ds["singular_value"].values[:N_MODES], dtype=np.float64)
        explained_variance_ratio = np.asarray(ds["explained_variance_ratio"].values[:N_MODES], dtype=np.float64)
        valid_mask = np.asarray(ds["valid_mask"].values, dtype=bool)
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        source_file = str(ds.attrs.get("source_file", ""))
    return Cobe2Reference(
        time=time,
        latitude=latitude,
        longitude=longitude,
        eof=eof,
        pc=pc,
        singular_value=singular_value,
        explained_variance_ratio=explained_variance_ratio,
        valid_mask=valid_mask,
        source_file=source_file,
    )


def find_wusd3_files(dataset_id: str, variable_name: str) -> List[Path]:
    base_dir = WUSD3_ROOT / dataset_id / "postprocess" / WUSD3_DOMAIN
    paths = sorted(base_dir.glob(f"{variable_name}.daily.*.nc"))
    if not paths:
        raise FileNotFoundError(f"No {variable_name} files found under {base_dir}")
    return paths


def resample_daily_file_to_monthly(path: Path, variable_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(path) as ds:
        if variable_name not in ds:
            raise KeyError(f"Expected variable {variable_name!r} in {path}")
        data = ds[[variable_name]].rename({"day": "time"}).resample(time="MS").mean()
        time_values = np.asarray(data["time"].values, dtype="datetime64[ns]")
        values = np.asarray(data[variable_name].values, dtype=np.float64)
    return time_values, values


def load_wusd3_monthly_native_cube(
    dataset_id: str,
    variable_name: str,
    spatial_mask: np.ndarray,
) -> MonthlyCube:
    times: List[np.ndarray] = []
    values: List[np.ndarray] = []
    source_files: List[str] = []

    for path in find_wusd3_files(dataset_id, variable_name):
        monthly_time, monthly_values = resample_daily_file_to_monthly(path, variable_name)
        masked = np.where(spatial_mask[np.newaxis, :, :], monthly_values, np.nan)
        times.append(monthly_time)
        values.append(masked.astype(np.float32))
        source_files.append(str(path))

    time_all = np.concatenate(times, axis=0)
    value_all = np.concatenate(values, axis=0)
    sort_index = np.argsort(time_all)
    return MonthlyCube(
        time=np.asarray(time_all[sort_index], dtype="datetime64[ns]"),
        values=np.asarray(value_all[sort_index], dtype=np.float32),
        units="K",
        source_files=source_files,
    )


def remap_wusd3_sst_monthly_to_cobe2_grid(
    monthly_sst: MonthlyCube,
    grid: GridData,
    cobe2: Cobe2Reference,
) -> MonthlyCube:
    source_points = np.column_stack([grid.latitude[grid.ocean_mask], grid.longitude[grid.ocean_mask]])
    if source_points.shape[0] == 0:
        raise ValueError("WUS-D3 ocean mask has zero valid source cells")

    target_lon, target_lat = np.meshgrid(cobe2.longitude, cobe2.latitude)
    target_points = np.column_stack([target_lat.ravel(), target_lon.ravel()])

    source_values = np.asarray(monthly_sst.values[:, grid.ocean_mask], dtype=np.float64).T
    interpolator = LinearNDInterpolator(source_points, source_values, fill_value=np.nan)
    interpolated = interpolator(target_points).T.reshape(
        monthly_sst.values.shape[0],
        cobe2.latitude.size,
        cobe2.longitude.size,
    )
    return MonthlyCube(
        time=monthly_sst.time,
        values=np.asarray(interpolated, dtype=np.float32),
        units=monthly_sst.units,
        source_files=monthly_sst.source_files,
    )


def compute_era5_monthly_on_wusd3_grid(
    target_months: np.ndarray,
    grid: GridData,
) -> MonthlyCube:
    land_lat = grid.latitude[grid.land_mask]
    land_lon = normalize_longitude_to_minus180_180(grid.longitude[grid.land_mask])
    lat_min = float(np.nanmin(land_lat)) - ERA5_MARGIN_DEGREES
    lat_max = float(np.nanmax(land_lat)) + ERA5_MARGIN_DEGREES
    lon_min = float(np.nanmin(land_lon)) - ERA5_MARGIN_DEGREES
    lon_max = float(np.nanmax(land_lon)) + ERA5_MARGIN_DEGREES
    lon_min_360 = lon_min % 360.0
    lon_max_360 = lon_max % 360.0
    target_lon_360 = grid.longitude % 360.0

    year_values = sorted(
        {
            int(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D")[:4])
            for value in np.asarray(target_months, dtype="datetime64[ns]")
        }
    )
    all_times: List[np.ndarray] = []
    all_values: List[np.ndarray] = []
    source_files: List[str] = []

    for year in year_values:
        path = ERA5_LAND_T2M_ROOT / f"ERA5_{year}_2m_temperature.nc"
        if not path.exists():
            raise FileNotFoundError(f"ERA5-Land file not found: {path}")
        print(f"  ERA5-Land year {year}: loading hourly subset from {path.name}", flush=True)

        with open_dataset_with_fallbacks(path) as ds:
            field = ds[ERA5_T2M_VARIABLE]
            if lon_min_360 <= lon_max_360:
                subset = field.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, lon_max_360))
            else:
                west = field.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, 360.0))
                east = field.sel(latitude=slice(lat_max, lat_min), longitude=slice(0.0, lon_max_360))
                subset = xr.concat([west, east], dim="longitude")
            subset_time = np.asarray(subset["time"].values, dtype="datetime64[ns]")
            subset_months = to_month_start(subset_time)
            monthly_fields: List[np.ndarray] = []
            monthly_time_list: List[np.datetime64] = []
            monthly_lat = np.asarray(subset["latitude"].values, dtype=np.float64)
            monthly_lon = np.asarray(subset["longitude"].values, dtype=np.float64)
            month_breaks = np.flatnonzero(subset_months[1:] != subset_months[:-1]) + 1
            month_starts = np.concatenate([np.array([0], dtype=np.int64), month_breaks])
            month_ends = np.concatenate([month_breaks, np.array([subset_months.size], dtype=np.int64)])
            for start_index, end_index in zip(month_starts.tolist(), month_ends.tolist()):
                month_value = np.asarray(subset_months[start_index], dtype="datetime64[ns]")
                month_block = np.asarray(subset.isel(time=slice(start_index, end_index)).values, dtype=np.float64)
                monthly_fields.append(np.nanmean(month_block, axis=0))
                monthly_time_list.append(month_value)
            monthly_time = np.asarray(monthly_time_list, dtype="datetime64[ns]")
            monthly_values = np.stack(monthly_fields, axis=0)

        if monthly_lat[0] > monthly_lat[-1]:
            monthly_lat = monthly_lat[::-1]
            monthly_values = monthly_values[:, ::-1, :]
        if monthly_lon[0] > monthly_lon[-1]:
            monthly_lon = monthly_lon[::-1]
            monthly_values = monthly_values[:, :, ::-1]

        regridded_year = np.full((monthly_values.shape[0],) + grid.latitude.shape, np.nan, dtype=np.float32)
        points = np.column_stack([grid.latitude.ravel(), target_lon_360.ravel()])
        for month_index in range(monthly_values.shape[0]):
            print(
                "    interpolating ERA5-Land month "
                f"{format_date(monthly_time[month_index])}",
                flush=True,
            )
            interpolator = RegularGridInterpolator(
                (monthly_lat, monthly_lon),
                monthly_values[month_index],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            regridded = interpolator(points).reshape(grid.latitude.shape)
            regridded = np.where(grid.land_mask, regridded, np.nan)
            regridded_year[month_index] = regridded.astype(np.float32)

        all_times.append(monthly_time)
        all_values.append(regridded_year)
        source_files.append(str(path))

    time_all = np.concatenate(all_times, axis=0)
    values_all = np.concatenate(all_values, axis=0)
    sort_index = np.argsort(time_all)
    time_sorted = time_all[sort_index]
    values_sorted = values_all[sort_index]

    month_set = set(target_months.tolist())
    keep = np.array([month in month_set for month in time_sorted.tolist()], dtype=bool)
    return MonthlyCube(
        time=np.asarray(time_sorted[keep], dtype="datetime64[ns]"),
        values=np.asarray(values_sorted[keep], dtype=np.float32),
        units="K",
        source_files=source_files,
    )


def build_monthly_anomaly_cube(monthly_cube: MonthlyCube) -> Tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        climatology, anomalies = compute_monthly_climatology_anomalies(
            np.asarray(monthly_cube.values, dtype=np.float64),
            np.asarray(monthly_cube.time, dtype="datetime64[ns]"),
        )
    return climatology.astype(np.float32), anomalies.astype(np.float32)


def compute_pattern_from_pc(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    pc_values = np.asarray(pc, dtype=np.float64)
    if pc_values.ndim != 1:
        raise ValueError(f"Expected 1D PC series, got shape {pc_values.shape}")
    if anomalies.shape[0] != pc_values.size:
        raise ValueError(f"Time dimension mismatch: pc={pc_values.size}, anomalies={anomalies.shape[0]}")
    pc_std = float(np.std(pc_values, ddof=1))
    if not np.isfinite(pc_std) or pc_std == 0.0:
        raise ValueError("PC standard deviation is zero or non-finite")
    return np.sum(pc_values[:, np.newaxis, np.newaxis] * np.asarray(anomalies, dtype=np.float64), axis=0) / pc_std


def compute_wusd3_projected_pcs(
    cobe2_eof: np.ndarray,
    wusd3_sst_anom: np.ndarray,
    shared_mask: np.ndarray,
) -> np.ndarray:
    eof_values = np.asarray(cobe2_eof[:, shared_mask], dtype=np.float64)
    sst_values = np.asarray(wusd3_sst_anom[:, shared_mask], dtype=np.float64)
    if not np.isfinite(sst_values).all():
        bad_count = int(np.size(sst_values) - np.isfinite(sst_values).sum())
        raise ValueError(f"Remapped WUS-D3 SST anomalies contain {bad_count} non-finite shared-mask values")
    return sst_values @ eof_values.T


def determine_projection_note(shared_mask: np.ndarray, cobe2_lat: np.ndarray, cobe2_lon: np.ndarray) -> str:
    lat_idx, lon_idx = np.where(shared_mask)
    if lat_idx.size == 0:
        return "No shared SST projection cells"
    lat_min = float(np.nanmin(cobe2_lat[lat_idx]))
    lat_max = float(np.nanmax(cobe2_lat[lat_idx]))
    lon_min = float(np.nanmin(cobe2_lon[lon_idx]))
    lon_max = float(np.nanmax(cobe2_lon[lon_idx]))
    return (
        "WUS-D3 SST is only available over the regional d01 domain, so formula 2 uses the shared remapped "
        f"COBE2-grid domain bounded approximately by lat {lat_min:.1f} to {lat_max:.1f} and lon {lon_min:.1f} to {lon_max:.1f}."
    )


def save_netcdf(
    time_values: np.ndarray,
    cobe2: Cobe2Reference,
    grid: GridData,
    cobe2_t2m_pattern: np.ndarray,
    wusd3_projected_pc: np.ndarray,
    wusd3_t2m_pattern: np.ndarray,
    cobe2_pc: np.ndarray,
    projection_shared_mask: np.ndarray,
    era5_overlap_start: str,
    era5_overlap_end: str,
    dataset_id: str,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "cobe2_t2m_pattern": (("mode", "south_north", "west_east"), cobe2_t2m_pattern.astype(np.float32)),
            "wusd3_projected_pc": (("time", "mode"), wusd3_projected_pc.astype(np.float32)),
            "wusd3_t2m_pattern": (("mode", "south_north", "west_east"), wusd3_t2m_pattern.astype(np.float32)),
            "cobe2_pc": (("time", "mode"), cobe2_pc.astype(np.float32)),
            "cobe2_eof": (("mode", "lat", "lon"), cobe2.eof.astype(np.float32)),
            "singular_value": (("mode",), cobe2.singular_value.astype(np.float32)),
            "explained_variance_ratio": (("mode",), cobe2.explained_variance_ratio.astype(np.float32)),
            "cobe2_valid_mask": (("lat", "lon"), cobe2.valid_mask),
            "wusd3_land_mask": (("south_north", "west_east"), grid.land_mask),
            "wusd3_ocean_mask": (("south_north", "west_east"), grid.ocean_mask),
            "projection_shared_mask": (("lat", "lon"), projection_shared_mask),
        },
        coords={
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "time": np.asarray(time_values, dtype="datetime64[ns]"),
            "lat": cobe2.latitude.astype(np.float32),
            "lon": cobe2.longitude.astype(np.float32),
            "south_north": np.arange(grid.latitude.shape[0], dtype=np.int32),
            "west_east": np.arange(grid.latitude.shape[1], dtype=np.int32),
            "lat2d": (("south_north", "west_east"), grid.latitude.astype(np.float32)),
            "lon2d": (("south_north", "west_east"), grid.longitude.astype(np.float32)),
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "description": "Additive COBE2/WUS-D3 SST-to-T2M mode projection diagnostic",
            "dataset_id": dataset_id,
            "cobe2_eof_file": str(COBE2_EOF_FILE),
            "time_overlap_start": era5_overlap_start,
            "time_overlap_end": era5_overlap_end,
            "formula_1": "COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))",
            "formula_2": "WUS_D3_PC_k(t) = sum_x [COBE2_EOF_k(x) * WUS_D3_SST_anom(t, x)]",
            "formula_3": "WUS_D3_T2M_k(x) = sum_t [WUS_D3_PC_k(t) * WUS_D3_T2M_anom(t, x)] / stddev(WUS_D3_PC_k(t))",
            "pc_std_ddof": 1,
            "pc_sign_flips_applied": "false",
        },
    )
    ds.to_netcdf(NETCDF_FILE)


def save_summary_csv(
    dataset_id: str,
    time_values: np.ndarray,
    cobe2: Cobe2Reference,
    cobe2_pc: np.ndarray,
    wusd3_projected_pc: np.ndarray,
    notes: str,
) -> None:
    overlap_start = format_date(time_values[0])
    overlap_end = format_date(time_values[-1])
    n_time = int(time_values.size)

    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "cobe2_evr",
                "cobe2_singular_value",
                "cobe2_era5_overlap_start",
                "cobe2_era5_overlap_end",
                "cobe2_era5_n_time",
                "cobe2_pc_std",
                "wusd3_overlap_start",
                "wusd3_overlap_end",
                "wusd3_n_time",
                "wusd3_projected_pc_std",
                "wusd3_projected_pc_min",
                "wusd3_projected_pc_max",
                "wusd3_projected_pc_mean",
                "notes",
            ]
        )
        for mode_index in range(N_MODES):
            projected_pc = np.asarray(wusd3_projected_pc[:, mode_index], dtype=np.float64)
            writer.writerow(
                [
                    mode_index + 1,
                    "{:.12g}".format(float(cobe2.explained_variance_ratio[mode_index])),
                    "{:.12g}".format(float(cobe2.singular_value[mode_index])),
                    overlap_start,
                    overlap_end,
                    n_time,
                    "{:.12g}".format(float(np.std(cobe2_pc[:, mode_index], ddof=1))),
                    overlap_start,
                    overlap_end,
                    n_time,
                    "{:.12g}".format(float(np.std(projected_pc, ddof=1))),
                    "{:.12g}".format(float(np.min(projected_pc))),
                    "{:.12g}".format(float(np.max(projected_pc))),
                    "{:.12g}".format(float(np.mean(projected_pc))),
                    notes if mode_index == 0 else "",
                ]
            )


def save_method_summary(
    dataset_id: str,
    cobe2: Cobe2Reference,
    grid: GridData,
    projection_shared_mask: np.ndarray,
    overlap_months: np.ndarray,
    wus_sst_files: Sequence[str],
    wus_t2m_files: Sequence[str],
    era5_files: Sequence[str],
) -> None:
    note = determine_projection_note(projection_shared_mask, cobe2.latitude, cobe2.longitude)
    land_count = int(np.count_nonzero(grid.land_mask))
    ocean_count = int(np.count_nonzero(grid.ocean_mask))
    shared_count = int(np.count_nonzero(projection_shared_mask))

    lines = [
        "# COBE2/WUS-D3 SST-to-T2M Mode Projection Method Summary",
        "",
        "## Inputs",
        f"- COBE2 EOF/PC file: `{COBE2_EOF_FILE}`",
        f"- COBE2 EOF source SST file: `{cobe2.source_file}`",
        f"- WUS-D3 dataset id: `{dataset_id}`",
        f"- WUS-D3 SST files: `{wus_sst_files[0]}` through `{wus_sst_files[-1]}`",
        f"- WUS-D3 T2M files: `{wus_t2m_files[0]}` through `{wus_t2m_files[-1]}`",
        f"- ERA5-Land T2M files: `{era5_files[0]}` through `{era5_files[-1]}`",
        "",
        "## Variables",
        "- COBE2 EOF file variables read: `eof(mode, lat, lon)`, `pc(time, mode)`, `singular_value(mode)`, `explained_variance_ratio(mode)`, `valid_mask(lat, lon)`, `time`.",
        f"- WUS-D3 SST variable: `{WUSD3_SST_VARIABLE}`.",
        f"- WUS-D3 T2M variable: `{WUSD3_T2M_VARIABLE}`.",
        f"- ERA5-Land T2M variable: `{ERA5_T2M_VARIABLE}`.",
        "",
        "## Anomaly Definition",
        "- Monthly climatology anomalies were applied to every time-varying field used in the diagnostic.",
        "- Exact rule: `field_anom(t, x) = field(t, x) - climatology[month(t), x]`.",
        "- No additional time-mean subtraction was applied after monthly climatology removal.",
        "",
        "## Exact Formulas Implemented",
        "- `COBE2_T2M_k(x) = sum_t [COBE2_PC_k(t) * ERA5_T2M_anom(t, x)] / stddev(COBE2_PC_k(t))`",
        "- `WUS_D3_PC_k(t) = sum_x [COBE2_EOF_k(x) * WUS_D3_SST_anom(t, x)]`",
        "- `WUS_D3_T2M_k(x) = sum_t [WUS_D3_PC_k(t) * WUS_D3_T2M_anom(t, x)] / stddev(WUS_D3_PC_k(t))`",
        "- `stddev` was computed with `ddof=1`.",
        "- No regression scaling was used.",
        "- No extra PC mean subtraction was applied.",
        "- No division by `n_time` was applied.",
        "- EOF/PC signs were left unchanged.",
        "",
        "## Time Overlap",
        f"- Shared overlap months used in all three formulas: {format_date(overlap_months[0])} through {format_date(overlap_months[-1])} ({int(overlap_months.size)} months).",
        "",
        "## Grid Handling",
        f"- WUS-D3 d01 grid shape: {grid.latitude.shape[0]} x {grid.latitude.shape[1]}.",
        f"- WUS-D3 d01 land cells retained for T2M: {land_count}.",
        f"- WUS-D3 d01 ocean cells retained for SST source remapping: {ocean_count}.",
        "- WUS-D3 monthly `tskin` fields were first masked to WRF ocean cells, then bilinearly remapped to the COBE2 global EOF grid using `LinearNDInterpolator` on the fixed d01 latitude/longitude coordinates.",
        f"- Shared SST projection cells on the COBE2 grid: {shared_count}.",
        f"- {note}",
        "- ERA5-Land monthly mean `t2m` fields were subset over the WUS land envelope and linearly interpolated to the native WUS-D3 d01 grid before land masking.",
        "",
        "## Units",
        "- COBE2 EOF maps are in saved EOF loading units from the existing global COBE2 diagnostic.",
        "- ERA5-Land and WUS-D3 T2M fields are in Kelvin.",
        "- WUS-D3 `tskin` monthly means were treated in Kelvin.",
        "",
        "## Limitations",
        f"- The COBE2 EOF basis is global, but WUS-D3 SST is only available on the regional d01 domain for `{dataset_id}`.",
        "- Formula 2 therefore reflects the shared remapped WUS-D3 SST domain rather than a full-global SST projection.",
        "- ERA5-Land is a land product, so the COBE2/ERA5 T2M pattern is evaluated on the WUS-D3 overland d01 grid rather than on an ocean-inclusive global T2M grid.",
    ]
    METHOD_SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    time_values: np.ndarray,
    grid: GridData,
    cobe2_t2m_pattern: np.ndarray,
    wusd3_projected_pc: np.ndarray,
    wusd3_t2m_pattern: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
    lon2d = grid.longitude
    lat2d = grid.latitude

    for mode_index in range(N_MODES):
        left_ax = axes[mode_index, 0]
        center_ax = axes[mode_index, 1]
        right_ax = axes[mode_index, 2]

        left_field = np.asarray(cobe2_t2m_pattern[mode_index], dtype=np.float64)
        left_vmax = float(np.nanmax(np.abs(left_field)))
        left_vmax = 1.0 if not np.isfinite(left_vmax) or left_vmax == 0.0 else left_vmax
        left_mesh = left_ax.pcolormesh(
            lon2d,
            lat2d,
            left_field,
            cmap="RdBu_r",
            shading="auto",
            vmin=-left_vmax,
            vmax=left_vmax,
        )
        left_ax.set_title(f"Mode {mode_index + 1}: COBE2 PC weighted ERA5 T2M")
        left_ax.set_xlabel("Longitude")
        left_ax.set_ylabel("Latitude")
        fig.colorbar(left_mesh, ax=left_ax, shrink=0.8)

        center_ax.plot(time_values.astype("datetime64[ns]"), wusd3_projected_pc[:, mode_index], color="black", linewidth=1.2)
        center_ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        center_ax.set_title(f"Mode {mode_index + 1}: WUS-D3 projected PC")
        center_ax.set_xlabel("Time")
        center_ax.set_ylabel("PC amplitude")
        center_ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        center_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for label in center_ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        right_field = np.asarray(wusd3_t2m_pattern[mode_index], dtype=np.float64)
        right_vmax = float(np.nanmax(np.abs(right_field)))
        right_vmax = 1.0 if not np.isfinite(right_vmax) or right_vmax == 0.0 else right_vmax
        right_mesh = right_ax.pcolormesh(
            lon2d,
            lat2d,
            right_field,
            cmap="RdBu_r",
            shading="auto",
            vmin=-right_vmax,
            vmax=right_vmax,
        )
        right_ax.set_title(f"Mode {mode_index + 1}: WUS-D3 PC weighted T2M")
        right_ax.set_xlabel("Longitude")
        right_ax.set_ylabel("Latitude")
        fig.colorbar(right_mesh, ax=right_ax, shrink=0.8)

    fig.suptitle("COBE2/WUS-D3 SST-to-T2M Mode Projection", fontsize=16)
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()

    dataset_id = DEFAULT_WUSD3_DATASET_ID
    print(f"Running {EXPERIMENT_NAME} for dataset={dataset_id}", flush=True)

    cobe2 = load_cobe2_reference()
    grid = load_wusd3_grid()

    print("Loading WUS-D3 monthly SST (native d01 ocean mask)...", flush=True)
    wus_sst_native = load_wusd3_monthly_native_cube(dataset_id, WUSD3_SST_VARIABLE, grid.ocean_mask)
    print("Remapping WUS-D3 monthly SST to COBE2 EOF grid...", flush=True)
    wus_sst_remapped = remap_wusd3_sst_monthly_to_cobe2_grid(wus_sst_native, grid, cobe2)
    _, wus_sst_anom = build_monthly_anomaly_cube(wus_sst_remapped)

    print("Loading WUS-D3 monthly T2M (native d01 land mask)...", flush=True)
    wus_t2m_monthly = load_wusd3_monthly_native_cube(dataset_id, WUSD3_T2M_VARIABLE, grid.land_mask)
    _, wus_t2m_anom = build_monthly_anomaly_cube(wus_t2m_monthly)

    overlap_months = intersect_months(cobe2.time, wus_sst_remapped.time, wus_t2m_monthly.time)
    if overlap_months.size == 0:
        raise ValueError("No shared monthly overlap among COBE2, WUS-D3 SST, and WUS-D3 T2M")

    print("Loading ERA5-Land monthly T2M on the WUS-D3 grid...", flush=True)
    era5_monthly = compute_era5_monthly_on_wusd3_grid(overlap_months, grid)
    _, era5_t2m_anom = build_monthly_anomaly_cube(era5_monthly)

    overlap_months = intersect_months(overlap_months, era5_monthly.time)
    if overlap_months.size == 0:
        raise ValueError("No shared monthly overlap after ERA5-Land loading")

    cobe2_pc_overlap = select_by_months(cobe2.time, cobe2.pc[:, :N_MODES], overlap_months)
    wus_sst_anom_overlap = select_by_months(wus_sst_remapped.time, wus_sst_anom, overlap_months)
    wus_t2m_anom_overlap = select_by_months(wus_t2m_monthly.time, wus_t2m_anom, overlap_months)
    era5_t2m_anom_overlap = select_by_months(era5_monthly.time, era5_t2m_anom, overlap_months)

    projection_shared_mask = cobe2.valid_mask & np.isfinite(wus_sst_anom_overlap).all(axis=0)
    if int(np.count_nonzero(projection_shared_mask)) == 0:
        raise ValueError("No shared valid cells remain for the WUS-D3 SST projection onto COBE2 EOFs")

    print(
        "Computing additive mode diagnostics with overlap "
        f"{format_date(overlap_months[0])} to {format_date(overlap_months[-1])} "
        f"({int(overlap_months.size)} months)...",
        flush=True,
    )
    wusd3_projected_pc = compute_wusd3_projected_pcs(cobe2.eof[:N_MODES], wus_sst_anom_overlap, projection_shared_mask)

    cobe2_t2m_pattern = np.full((N_MODES,) + grid.latitude.shape, np.nan, dtype=np.float64)
    wusd3_t2m_pattern = np.full((N_MODES,) + grid.latitude.shape, np.nan, dtype=np.float64)
    for mode_index in range(N_MODES):
        cobe2_t2m_pattern[mode_index] = compute_pattern_from_pc(cobe2_pc_overlap[:, mode_index], era5_t2m_anom_overlap)
        wusd3_t2m_pattern[mode_index] = compute_pattern_from_pc(wusd3_projected_pc[:, mode_index], wus_t2m_anom_overlap)

    note = determine_projection_note(projection_shared_mask, cobe2.latitude, cobe2.longitude)
    save_netcdf(
        time_values=overlap_months,
        cobe2=cobe2,
        grid=grid,
        cobe2_t2m_pattern=cobe2_t2m_pattern,
        wusd3_projected_pc=wusd3_projected_pc,
        wusd3_t2m_pattern=wusd3_t2m_pattern,
        cobe2_pc=cobe2_pc_overlap,
        projection_shared_mask=projection_shared_mask,
        era5_overlap_start=format_date(overlap_months[0]),
        era5_overlap_end=format_date(overlap_months[-1]),
        dataset_id=dataset_id,
    )
    save_summary_csv(
        dataset_id=dataset_id,
        time_values=overlap_months,
        cobe2=cobe2,
        cobe2_pc=cobe2_pc_overlap,
        wusd3_projected_pc=wusd3_projected_pc,
        notes=note,
    )
    save_method_summary(
        dataset_id=dataset_id,
        cobe2=cobe2,
        grid=grid,
        projection_shared_mask=projection_shared_mask,
        overlap_months=overlap_months,
        wus_sst_files=wus_sst_native.source_files,
        wus_t2m_files=wus_t2m_monthly.source_files,
        era5_files=era5_monthly.source_files,
    )
    plot_results(
        time_values=overlap_months,
        grid=grid,
        cobe2_t2m_pattern=cobe2_t2m_pattern,
        wusd3_projected_pc=wusd3_projected_pc,
        wusd3_t2m_pattern=wusd3_t2m_pattern,
    )

    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Method summary: {METHOD_SUMMARY_FILE}", flush=True)
    print(f"Figure: {FIGURE_FILE}", flush=True)
    for mode_index in range(N_MODES):
        print(
            f"Mode {mode_index + 1}: EVR={cobe2.explained_variance_ratio[mode_index]:.6f} "
            f"overlap={format_date(overlap_months[0])}..{format_date(overlap_months[-1])} "
            f"n_time={int(overlap_months.size)} "
            f"WUS_D3_PC_std={np.std(wusd3_projected_pc[:, mode_index], ddof=1):.6f}",
            flush=True,
        )
    print(f"Notes: {note}", flush=True)


if __name__ == "__main__":
    main()
