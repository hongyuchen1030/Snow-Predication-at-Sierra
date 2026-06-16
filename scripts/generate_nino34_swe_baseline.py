#!/usr/bin/env python3
"""
Generate the Ni\~no 3.4 Sep--Mar water-year predictor table for the Sierra SWE baseline.
"""

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    find_lat_lon_names,
    open_dataset_with_fallbacks,
)


COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "nino34"
CSV_PATH = OUTPUT_DIR / "nino34_monthly_wy1985_2021_sep_mar.csv"
NETCDF_PATH = OUTPUT_DIR / "nino34_monthly_wy1985_2021_sep_mar.nc"
JSON_PATH = OUTPUT_DIR / "nino34_monthly_wy1985_2021_sep_mar_summary.json"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)

NINO34_LAT_MIN = -5.0
NINO34_LAT_MAX = 5.0
NINO34_LON_MIN_360 = 190.0
NINO34_LON_MAX_360 = 240.0

MONTH_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]


@dataclass(frozen=True)
class RunSummary:
    source_file: str
    output_csv: str
    output_netcdf: str
    water_year_start: int
    water_year_end: int
    n_water_years: int
    domain_lat_min: float
    domain_lat_max: float
    domain_lon_min_360: float
    domain_lon_max_360: float
    month_sequence: List[str]
    longitude_convention: str
    anomaly_definition: str
    runtime_hostname: str
    slurm_job_id: Optional[str]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def selected_times() -> Tuple[np.ndarray, np.ndarray]:
    times: List[np.datetime64] = []
    for wy in WATER_YEARS:
        for _, year_offset, month in MONTH_SPECS:
            year = int(wy + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
    return WATER_YEARS.copy(), np.asarray(times, dtype="datetime64[ns]")


def compute_weighted_regional_mean_360(
    values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min_360: float,
    lon_max_360: float,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64) % 360.0
    if data.ndim != 3:
        raise ValueError("Expected SST values with shape (time, lat, lon)")
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Expected 1D latitude and longitude coordinates")

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min_360) & (lon <= lon_max_360)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError("Requested region is empty on this grid")

    region = data[:, lat_mask, :][:, :, lon_mask]
    region_lat = lat[lat_mask]
    weights = np.cos(np.deg2rad(region_lat))[:, np.newaxis]
    weights_3d = np.broadcast_to(weights[np.newaxis, :, :], region.shape)
    valid = np.isfinite(region)
    weighted_sum = np.sum(np.where(valid, region * weights_3d, 0.0), axis=(1, 2))
    weight_sum = np.sum(np.where(valid, weights_3d, 0.0), axis=(1, 2))
    result = np.full(region.shape[0], np.nan, dtype=np.float64)
    good = weight_sum > 0.0
    result[good] = weighted_sum[good] / weight_sum[good]
    return result


def load_monthly_nino34_series() -> Tuple[np.ndarray, np.ndarray]:
    water_years, times = selected_times()
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        lat_name, lon_name = find_lat_lon_names(ds, "sst")
        subset = ds["sst"].sel(time=times).load()
        latitude = np.asarray(subset[lat_name].values, dtype=np.float64)
        longitude = np.asarray(subset[lon_name].values, dtype=np.float64)
        values = np.asarray(subset.values, dtype=np.float64)

    nino34_monthly = compute_weighted_regional_mean_360(
        values,
        latitude,
        longitude,
        NINO34_LAT_MIN,
        NINO34_LAT_MAX,
        NINO34_LON_MIN_360,
        NINO34_LON_MAX_360,
    )
    if nino34_monthly.shape[0] != water_years.size * len(MONTH_SPECS):
        raise ValueError(
            f"Unexpected monthly series length {nino34_monthly.shape[0]} for {water_years.size} water years"
        )
    return water_years, nino34_monthly.reshape(water_years.size, len(MONTH_SPECS))


def build_monthly_anomalies(monthly_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(monthly_values, dtype=np.float64)
    climatology = np.nanmean(values, axis=0)
    anomalies = values - climatology[np.newaxis, :]
    return climatology, anomalies


def write_csv(water_years: np.ndarray, anomalies: np.ndarray) -> None:
    fieldnames = ["water_year"] + [f"Nino34_{name}" for name, _, _ in MONTH_SPECS]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row_index, wy in enumerate(water_years):
            writer.writerow([int(wy)] + [f"{float(value):.12g}" for value in anomalies[row_index]])


def write_netcdf(water_years: np.ndarray, raw_values: np.ndarray, climatology: np.ndarray, anomalies: np.ndarray) -> None:
    month_names = np.asarray([name for name, _, _ in MONTH_SPECS], dtype=object)
    ds = xr.Dataset(
        data_vars={
            "nino34_index_raw": (("water_year", "month"), raw_values.astype(np.float32)),
            "nino34_climatology": (("month",), climatology.astype(np.float32)),
            "nino34_index_anomaly": (("water_year", "month"), anomalies.astype(np.float32)),
        },
        coords={
            "water_year": water_years.astype(np.int32),
            "month": month_names,
        },
        attrs={
            "description": "Niño 3.4 monthly SST anomaly table for the Sierra SWE baseline",
            "latitude_range": f"{NINO34_LAT_MIN} to {NINO34_LAT_MAX}",
            "longitude_range_360": f"{NINO34_LON_MIN_360} to {NINO34_LON_MAX_360}",
            "longitude_convention": "0..360",
            "anomaly_definition": "monthly raw regional mean minus the Sep/Oct/Nov/Dec/Jan/Feb/Mar climatology over WY1985-2021",
            "source_file": str(COBE2_SST_FILE),
        },
    )
    ds.to_netcdf(NETCDF_PATH)


def write_summary() -> None:
    summary = RunSummary(
        source_file=str(COBE2_SST_FILE),
        output_csv=str(CSV_PATH),
        output_netcdf=str(NETCDF_PATH),
        water_year_start=WATER_YEAR_START,
        water_year_end=WATER_YEAR_END,
        n_water_years=int(WATER_YEARS.size),
        domain_lat_min=NINO34_LAT_MIN,
        domain_lat_max=NINO34_LAT_MAX,
        domain_lon_min_360=NINO34_LON_MIN_360,
        domain_lon_max_360=NINO34_LON_MAX_360,
        month_sequence=[name for name, _, _ in MONTH_SPECS],
        longitude_convention="0..360",
        anomaly_definition="monthly raw regional mean minus the month-specific climatology over the aligned WY1985-2021 Sep--Mar samples",
        runtime_hostname=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    JSON_PATH.write_text(json.dumps(asdict(summary), indent=2) + "\n")


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    water_years, raw_values = load_monthly_nino34_series()
    climatology, anomalies = build_monthly_anomalies(raw_values)
    write_csv(water_years, anomalies)
    write_netcdf(water_years, raw_values, climatology, anomalies)
    write_summary()
    print(f"Wrote CSV: {CSV_PATH}", flush=True)
    print(f"Wrote NetCDF: {NETCDF_PATH}", flush=True)
    print(f"Wrote summary JSON: {JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()
