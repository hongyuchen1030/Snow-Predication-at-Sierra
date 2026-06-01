#!/usr/bin/env python3
"""
Aggregate ERA5-Land hourly 2m temperature to monthly means and compute
monthly-climatology anomalies.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import dask
from dask.diagnostics import ProgressBar
from netCDF4 import Dataset, date2num
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "era5land_t2m_monthly_anomalies"
OUTPUT_DIR = Path(os.environ.get("ERA5LAND_T2M_MONTHLY_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).expanduser()
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate_monthly_years"
MONTHLY_MEAN_FILE = OUTPUT_DIR / "era5land_t2m_monthly_mean.nc"
MONTHLY_CLIM_FILE = OUTPUT_DIR / "era5land_t2m_monthly_climatology.nc"
MONTHLY_ANOM_FILE = OUTPUT_DIR / "era5land_t2m_monthly_anomaly.nc"
SUMMARY_JSON_FILE = OUTPUT_DIR / "era5land_t2m_monthly_anomaly_summary.json"
REUSE_EXISTING_MONTHLY_MEAN = os.environ.get("ERA5LAND_T2M_REUSE_EXISTING_MONTHLY_MEAN", "").lower() in {
    "1",
    "true",
    "yes",
}

ERA5_LAND_T2M_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5-Land/2m_temperature")
ERA5_VARIABLE = "t2m"
ERA5_FILE_PATTERN = "ERA5_*_2m_temperature.nc"
ERA5_YEAR_PATTERN = re.compile(r"ERA5_(\d{4})_2m_temperature\.nc$")

TIME_CHUNK = 744
LAT_CHUNK = 180
LON_CHUNK = 360
NETCDF_ENGINE = "netcdf4"
NETCDF_TIME_UNITS = "days since 1900-01-01 00:00:00"
NETCDF_TIME_CALENDAR = "proleptic_gregorian"
READ_BLOCK_HOURS = 24
N_WORKERS = int(os.environ.get("ERA5LAND_T2M_MONTHLY_N_WORKERS", str(min(32, max(1, os.cpu_count() or 1)))))


@dataclass(frozen=True)
class RuntimeInfo:
    hostname: str
    slurm_job_id: str


@dataclass(frozen=True)
class SummaryPayload:
    input_monthly_mean_file: str
    variable_name: str
    output_monthly_climatology: str
    output_monthly_anomaly: str
    output_summary_json: str
    monthly_output_n_time: int
    monthly_mean_start: str
    monthly_mean_end: str
    climatology_months_available: List[int]
    spatial_grid_shape: List[int]
    units: str
    output_directory_size: str
    reused_existing_monthly_mean_file: bool
    ran_on_compute_node: bool
    slurm_job_id: str
    hostname: str
    chunking: Dict[str, int]


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


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)


def discover_era5_files() -> List[Path]:
    paths = sorted(ERA5_LAND_T2M_ROOT.glob(ERA5_FILE_PATTERN))
    valid_paths = [path for path in paths if ERA5_YEAR_PATTERN.fullmatch(path.name)]
    if not valid_paths:
        raise FileNotFoundError(f"No ERA5-Land T2M files found under {ERA5_LAND_T2M_ROOT}")
    return valid_paths


def format_time(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="s"))


def open_hourly_file(path: Path) -> xr.Dataset:
    return xr.open_dataset(
        path,
        engine=NETCDF_ENGINE,
        chunks={"time": TIME_CHUNK, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )


def build_monthly_mean(hourly: xr.DataArray) -> xr.DataArray:
    monthly = hourly.resample(time="MS").mean(keep_attrs=True)
    monthly = monthly.astype(np.float32)
    monthly.name = ERA5_VARIABLE
    monthly.attrs.update(hourly.attrs)
    monthly.attrs["description"] = "ERA5-Land monthly mean 2m temperature aggregated from hourly data"
    monthly.attrs["time_aggregation"] = "calendar-month mean from hourly ERA5-Land T2M"
    monthly.attrs["time_coordinate_convention"] = "month start"
    return monthly


def open_monthly_dataset(paths: List[Path]) -> xr.Dataset:
    datasets = [
        xr.open_dataset(
            path,
            engine=NETCDF_ENGINE,
            chunks={"time": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
            decode_times=True,
        )[[ERA5_VARIABLE]]
        for path in paths
    ]
    return xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")


def build_monthly_climatology(monthly: xr.DataArray) -> xr.DataArray:
    climatology = monthly.groupby("time.month").mean("time", keep_attrs=True)
    climatology = climatology.astype(np.float32)
    climatology.name = ERA5_VARIABLE
    climatology.attrs.update(monthly.attrs)
    climatology.attrs["description"] = "ERA5-Land month-of-year climatology of monthly mean 2m temperature"
    climatology.attrs["climatology_definition"] = "mean of all monthly means for each calendar month"
    return climatology


def build_monthly_anomaly(monthly: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    anomaly = monthly.groupby("time.month") - climatology
    anomaly = anomaly.astype(np.float32)
    anomaly.name = ERA5_VARIABLE
    anomaly.attrs.update(monthly.attrs)
    anomaly.attrs["description"] = "ERA5-Land monthly mean 2m temperature anomaly relative to month-of-year climatology"
    anomaly.attrs["anomaly_definition"] = "monthly_mean(time, x) - climatology[month(time), x]"
    return anomaly


def dataset_for_output(data_array: xr.DataArray, description: str) -> xr.Dataset:
    ds = data_array.to_dataset(name=ERA5_VARIABLE)
    ds.attrs["description"] = description
    ds.attrs["source"] = "ERA5-Land hourly 2m temperature"
    ds.attrs["variable_name"] = ERA5_VARIABLE
    return ds


def build_encoding(ds: xr.Dataset) -> Dict[str, Dict[str, object]]:
    chunksizes_by_dims = {
        ("time", "latitude", "longitude"): [12, LAT_CHUNK, LON_CHUNK],
        ("month", "latitude", "longitude"): [12, LAT_CHUNK, LON_CHUNK],
    }
    encoding: Dict[str, Dict[str, object]] = {}
    for var_name, data_array in ds.data_vars.items():
        dims = tuple(data_array.dims)
        var_encoding: Dict[str, object] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
        }
        if dims in chunksizes_by_dims:
            var_encoding["chunksizes"] = chunksizes_by_dims[dims]
        if "_FillValue" in data_array.encoding:
            var_encoding["_FillValue"] = data_array.encoding["_FillValue"]
        else:
            var_encoding["_FillValue"] = np.float32(np.nan)
        encoding[var_name] = var_encoding
    return encoding


def write_dataset(ds: xr.Dataset, path: Path) -> None:
    encoding = build_encoding(ds)
    delayed = ds.to_netcdf(
        path,
        engine=NETCDF_ENGINE,
        encoding=encoding,
        compute=False,
    )
    with ProgressBar():
        delayed.compute()


def year_from_path(path: Path) -> int:
    match = ERA5_YEAR_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Could not parse ERA5-Land year from {path}")
    return int(match.group(1))


def write_summary(payload: SummaryPayload) -> None:
    SUMMARY_JSON_FILE.write_text(json.dumps(asdict(payload), indent=2) + "\n", encoding="utf-8")


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def output_dir_size_text() -> str:
    total_bytes = 0
    for path in OUTPUT_DIR.rglob("*"):
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


def intermediate_monthly_path(path: Path) -> Path:
    return INTERMEDIATE_DIR / f"era5land_t2m_monthly_mean_{year_from_path(path):04d}.nc"


def month_start_values(time_values: np.ndarray) -> np.ndarray:
    months = time_values.astype("datetime64[M]")
    month_change = np.concatenate(([True], months[1:] != months[:-1]))
    return months[month_change]


def initialize_monthly_mean_file(
    path: Path,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    variable_attrs: Dict[str, object],
) -> None:
    with Dataset(path, mode="w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("latitude", int(latitude.size))
        ds.createDimension("longitude", int(longitude.size))

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = NETCDF_TIME_UNITS
        time_var.calendar = NETCDF_TIME_CALENDAR
        time_var.standard_name = "time"
        time_var.long_name = "time"

        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lat_var[:] = np.asarray(latitude.values, dtype=np.float32)
        for key, value in latitude.attrs.items():
            lat_var.setncattr(key, value)

        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = np.asarray(longitude.values, dtype=np.float32)
        for key, value in longitude.attrs.items():
            lon_var.setncattr(key, value)

        t2m_var = ds.createVariable(
            ERA5_VARIABLE,
            "f4",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=(12, LAT_CHUNK, LON_CHUNK),
            fill_value=np.float32(np.nan),
        )
        for key, value in variable_attrs.items():
            t2m_var.setncattr(key, value)

        ds.description = "ERA5-Land monthly mean 2m temperature aggregated from hourly files"
        ds.source = "ERA5-Land hourly 2m temperature"
        ds.variable_name = ERA5_VARIABLE


def append_monthly_mean_step(path: Path, time_value: np.datetime64, data: xr.DataArray, time_index: int) -> None:
    timestamp = time_value.astype("datetime64[s]").astype(object)
    numeric_time = date2num(timestamp, units=NETCDF_TIME_UNITS, calendar=NETCDF_TIME_CALENDAR)
    with Dataset(path, mode="a") as ds:
        ds.variables["time"][time_index] = numeric_time
        ds.variables[ERA5_VARIABLE][time_index, :, :] = np.asarray(data.values, dtype=np.float32)


def compute_monthly_mean_streaming(hourly_var, time_slice: slice) -> np.ndarray:
    month_sum: np.ndarray | None = None
    month_count: np.ndarray | None = None

    for block_start in range(time_slice.start, time_slice.stop, READ_BLOCK_HOURS):
        block_stop = min(block_start + READ_BLOCK_HOURS, time_slice.stop)
        block = hourly_var[block_start:block_stop, :, :]
        if np.ma.isMaskedArray(block):
            valid_mask = ~np.ma.getmaskarray(block)
            block_values = np.asarray(block.filled(0.0), dtype=np.float64)
        else:
            block_values = np.asarray(block, dtype=np.float64)
            valid_mask = np.isfinite(block_values)
            block_values = np.where(valid_mask, block_values, 0.0)

        block_sum = block_values.sum(axis=0, dtype=np.float64)
        block_count = valid_mask.sum(axis=0, dtype=np.int32)
        if month_sum is None:
            month_sum = block_sum
            month_count = block_count
        else:
            month_sum += block_sum
            month_count += block_count

    if month_sum is None or month_count is None:
        raise RuntimeError(f"Empty month slice {time_slice} encountered while aggregating monthly mean.")

    monthly_mean = np.full(month_sum.shape, np.nan, dtype=np.float32)
    np.divide(month_sum, month_count, out=monthly_mean, where=month_count > 0)
    return monthly_mean


def process_year_file(input_path_text: str) -> Dict[str, object]:
    input_path = Path(input_path_text)
    output_path = intermediate_monthly_path(input_path)

    hourly_start = ""
    hourly_end = ""
    hourly_n_time = 0
    monthly_n_time = 0
    units = ""
    spatial_grid_shape: List[int] = []

    with open_hourly_file(input_path) as hourly_ds, Dataset(input_path) as hourly_nc:
        if ERA5_VARIABLE not in hourly_ds:
            raise KeyError(f"Expected variable {ERA5_VARIABLE!r} in {input_path}")
        hourly = hourly_ds[ERA5_VARIABLE]
        hourly_var = hourly_nc.variables[ERA5_VARIABLE]
        time_values = np.asarray(hourly["time"].values, dtype="datetime64[ns]")
        hourly_start = format_time(np.asarray(time_values[0], dtype="datetime64[ns]"))
        hourly_end = format_time(np.asarray(time_values[-1], dtype="datetime64[ns]"))
        hourly_n_time = int(hourly.sizes["time"])
        units = str(hourly.attrs.get("units", ""))
        spatial_grid_shape = [int(hourly.sizes["latitude"]), int(hourly.sizes["longitude"])]

        monthly_mean_attrs = dict(build_monthly_mean(hourly.isel(time=slice(0, 1))).attrs)
        initialize_monthly_mean_file(output_path, hourly_ds["latitude"], hourly_ds["longitude"], monthly_mean_attrs)

        month_values = month_start_values(time_values)
        month_keys = time_values.astype("datetime64[M]")
        for month_index, month_value in enumerate(month_values):
            month_indices = np.flatnonzero(month_keys == month_value)
            month_slice = slice(int(month_indices[0]), int(month_indices[-1]) + 1)
            monthly_mean_values = compute_monthly_mean_streaming(hourly_var, month_slice)
            monthly_mean_step = hourly.isel(time=0, drop=True).copy(data=monthly_mean_values)
            monthly_mean_step.attrs = dict(monthly_mean_attrs)
            append_monthly_mean_step(
                output_path,
                month_value.astype("datetime64[ns]"),
                monthly_mean_step,
                month_index,
            )
            monthly_n_time += 1

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "year": year_from_path(input_path),
        "hourly_start": hourly_start,
        "hourly_end": hourly_end,
        "hourly_n_time": hourly_n_time,
        "monthly_n_time": monthly_n_time,
        "units": units,
        "spatial_grid_shape": spatial_grid_shape,
    }


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()

    dask.config.set({"array.slicing.split_large_chunks": True})

    if REUSE_EXISTING_MONTHLY_MEAN:
        if not MONTHLY_MEAN_FILE.exists():
            raise FileNotFoundError(f"Monthly mean file does not exist for reuse mode: {MONTHLY_MEAN_FILE}")
        input_files: List[Path] = []
        monthly_n_time = 0
        spatial_grid_shape: List[int] = []
        units = ""
        print(f"Reusing existing monthly mean file {MONTHLY_MEAN_FILE}", flush=True)
    else:
        input_files = discover_era5_files()
        print(f"Discovered {len(input_files)} ERA5-Land yearly T2M files", flush=True)

        hourly_start = ""
        hourly_end = ""
        spatial_grid_shape = []
        units = ""
        monthly_n_time = 0
        intermediate_paths: List[Path] = []
        print(f"Using {N_WORKERS} worker processes for year-level monthly aggregation", flush=True)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_path = {executor.submit(process_year_file, str(path)): path for path in input_files}
            year_results: List[Dict[str, object]] = []
            for completed_index, future in enumerate(as_completed(future_to_path), start=1):
                input_path = future_to_path[future]
                result = future.result()
                year_results.append(result)
                intermediate_paths.append(Path(result["output_path"]))
                print(
                    f"[{completed_index}/{len(input_files)}] Finished monthly aggregation for {input_path.name}",
                    flush=True,
                )

        year_results.sort(key=lambda item: int(item["year"]))
        hourly_start = str(year_results[0]["hourly_start"])
        hourly_end = str(year_results[-1]["hourly_end"])
        monthly_n_time = int(sum(int(item["monthly_n_time"]) for item in year_results))
        spatial_grid_shape = list(year_results[0]["spatial_grid_shape"])
        units = str(year_results[0]["units"])

        print(
            f"Hourly input range: {hourly_start} to {hourly_end}",
            flush=True,
        )
        print(
            f"Spatial grid: latitude={spatial_grid_shape[0]} longitude={spatial_grid_shape[1]}",
            flush=True,
        )

        with open_monthly_dataset(sorted(intermediate_paths)) as monthly_ds_lazy:
            monthly_mean_full = monthly_ds_lazy[ERA5_VARIABLE].astype(np.float32)
            monthly_mean_full.attrs.update(monthly_ds_lazy[ERA5_VARIABLE].attrs)
            monthly_mean_full.attrs["description"] = "ERA5-Land monthly mean 2m temperature aggregated from hourly files"
            monthly_mean_ds = dataset_for_output(
                monthly_mean_full,
                "ERA5-Land monthly mean 2m temperature aggregated from hourly files",
            )
            print(f"Writing monthly means to {MONTHLY_MEAN_FILE}", flush=True)
            write_dataset(monthly_mean_ds, MONTHLY_MEAN_FILE)

    remove_if_exists(MONTHLY_CLIM_FILE)
    remove_if_exists(MONTHLY_ANOM_FILE)
    remove_if_exists(SUMMARY_JSON_FILE)

    monthly_ds = xr.open_dataset(
        MONTHLY_MEAN_FILE,
        engine=NETCDF_ENGINE,
        chunks={"time": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
    )
    try:
        monthly_mean = monthly_ds[ERA5_VARIABLE]
        monthly_n_time = int(monthly_mean.sizes["time"])
        anomaly_start = format_time(np.asarray(monthly_mean["time"].values[0], dtype="datetime64[ns]"))
        anomaly_end = format_time(np.asarray(monthly_mean["time"].values[-1], dtype="datetime64[ns]"))
        if not spatial_grid_shape:
            spatial_grid_shape = [int(monthly_mean.sizes["latitude"]), int(monthly_mean.sizes["longitude"])]
        if not units:
            units = str(monthly_mean.attrs.get("units", ""))

        monthly_climatology = build_monthly_climatology(monthly_mean)
        monthly_clim_ds = dataset_for_output(
            monthly_climatology,
            "ERA5-Land month-of-year climatology from monthly mean 2m temperature",
        )
        print(f"Writing monthly climatology to {MONTHLY_CLIM_FILE}", flush=True)
        write_dataset(monthly_clim_ds, MONTHLY_CLIM_FILE)

        monthly_anomaly = build_monthly_anomaly(monthly_mean, monthly_climatology)
        monthly_anom_ds = dataset_for_output(
            monthly_anomaly,
            "ERA5-Land monthly mean 2m temperature anomalies relative to month-of-year climatology",
        )
        print(f"Writing monthly anomalies to {MONTHLY_ANOM_FILE}", flush=True)
        write_dataset(monthly_anom_ds, MONTHLY_ANOM_FILE)
        climatology_months_available = [int(value) for value in np.asarray(monthly_climatology["month"].values).tolist()]
    finally:
        monthly_ds.close()

    summary = SummaryPayload(
        input_monthly_mean_file=str(MONTHLY_MEAN_FILE),
        variable_name=ERA5_VARIABLE,
        output_monthly_climatology=str(MONTHLY_CLIM_FILE),
        output_monthly_anomaly=str(MONTHLY_ANOM_FILE),
        output_summary_json=str(SUMMARY_JSON_FILE),
        monthly_output_n_time=monthly_n_time,
        monthly_mean_start=anomaly_start,
        monthly_mean_end=anomaly_end,
        climatology_months_available=climatology_months_available,
        spatial_grid_shape=spatial_grid_shape,
        units=units,
        output_directory_size=output_dir_size_text(),
        reused_existing_monthly_mean_file=REUSE_EXISTING_MONTHLY_MEAN,
        ran_on_compute_node=bool(runtime.slurm_job_id and "nid" in runtime.hostname),
        slurm_job_id=runtime.slurm_job_id,
        hostname=runtime.hostname,
        chunking={"time": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
    )
    write_summary(summary)
    print(f"Wrote summary JSON to {SUMMARY_JSON_FILE}", flush=True)


if __name__ == "__main__":
    main()
