#!/usr/bin/env python3
"""
Month-specific lagged MJO phase composites for overland ERA5-Land daily T2m anomalies.

This script:
1. Downloads or reuses NOAA PSL RMM* daily data.
2. Builds daily ERA5-Land T2m means over a broad California analysis box.
3. Computes day-of-year climatology and daily anomalies at each land grid cell.
4. Aggregates land-only regional daily anomaly time series for:
   - broad California
   - Sierra region
   - northern Sierra
   - central Sierra
   - southern Sierra
5. Composites target-month regional anomalies by active MJO phase at lags 0..N.
6. Writes a summary CSV, month/region heatmaps, and lag-summary diagnostic plots.
7. Optionally estimates significance by permuting phase labels within base-day year-month blocks.
8. Optionally builds anomaly composite maps for user-selected phase-lag cells.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import sys
import urllib.request
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.paths import ERA5_LAND_ROOT
from snow_ml.data import DEFAULT_MODEL_REGION, RegionBounds

from scripts.run_cobe2_pacific_sierra_t2m_level1_diagnostic import subset_era5_region_360
from scripts.run_s2s_pc6_t2m_top20_regions_loyo_models import load_label_mask
from scripts.run_sst_monthly_climatology_eof_diagnostics import ensure_runtime_on_compute_node, get_runtime


RMM_URL = "https://psl.noaa.gov/mjo/mjoindex/rmm_star_data.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "mjo_lagged_era5land_t2m_phase_composites"
DEFAULT_RMM_FILE = DEFAULT_OUTPUT_DIR / "inputs" / "rmm_star_data.txt"
DEFAULT_DAILY_DIR = DEFAULT_OUTPUT_DIR / "intermediate_daily_t2m"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "heatmaps"
DEFAULT_MAPS_DIR = DEFAULT_OUTPUT_DIR / "maps"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "mjo_phase_lag_summary.csv"
DEFAULT_METADATA_JSON = DEFAULT_OUTPUT_DIR / "mjo_phase_lag_metadata.json"
DEFAULT_LABEL_NETCDF = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
    / "top20_region_labels"
    / "cleaned_top20_region_labels.nc"
)
ERA5_VARIABLE = "t2m"
ERA5_FILE_TEMPLATE = "ERA5_{year:04d}_2m_temperature.nc"
ERA5_HOURLY_DIR = Path(ERA5_LAND_ROOT) / "2m_temperature"
MONTH_NAME_TO_NUMBER = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4}
MONTH_NUMBER_TO_NAME = {value: key for key, value in MONTH_NAME_TO_NUMBER.items()}
MAX_LAG_DEFAULT = 42
ACTIVE_AMPLITUDE_THRESHOLD = 1.0
LAG_WINDOW_SIZE = 7
TIME_CHUNK = 744
LAT_CHUNK = 180
LON_CHUNK = 360
NETCDF_ENGINE = "netcdf4"
SIERRA_REGION_360 = RegionBounds(lat_min=35.0, lat_max=42.0, lon_min=236.0, lon_max=243.0)
BROAD_CALIFORNIA_REGION_360 = RegionBounds(
    lat_min=float(DEFAULT_MODEL_REGION.lat_min),
    lat_max=float(DEFAULT_MODEL_REGION.lat_max),
    lon_min=float(DEFAULT_MODEL_REGION.lon_min % 360.0),
    lon_max=float(DEFAULT_MODEL_REGION.lon_max % 360.0),
)
WESTERN_US_MAP_REGION_360 = RegionBounds(lat_min=30.0, lat_max=50.0, lon_min=230.0, lon_max=255.0)


@dataclass(frozen=True)
class RegionSpec:
    key: str
    title: str
    description: str
    mask: np.ndarray


@dataclass(frozen=True)
class SelectedCell:
    target_month: int
    phase: int
    lag: int

    @property
    def month_name(self) -> str:
        return MONTH_NUMBER_TO_NAME[self.target_month]

    @property
    def label(self) -> str:
        return f"{self.month_name}_phase{self.phase}_lag{self.lag:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rmm-file", type=Path, default=DEFAULT_RMM_FILE)
    parser.add_argument("--rmm-url", type=str, default=RMM_URL)
    parser.add_argument("--label-netcdf", type=Path, default=DEFAULT_LABEL_NETCDF)
    parser.add_argument("--target-months", nargs="+", default=["Jan", "Feb", "Mar", "Apr"])
    parser.add_argument("--year-start", type=int, default=1980)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--max-lag", type=int, default=MAX_LAG_DEFAULT)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--selected-map-cell", action="append", default=[])
    parser.add_argument("--download-rmm", action="store_true", default=True)
    parser.add_argument("--no-download-rmm", dest="download_rmm", action="store_false")
    parser.add_argument("--skip-runtime-check", action="store_true")
    parser.add_argument("--reuse-daily-files", action="store_true", default=True)
    parser.add_argument("--no-reuse-daily-files", dest="reuse_daily_files", action="store_false")
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "inputs": output_dir / "inputs",
        "daily": output_dir / "intermediate_daily_t2m",
        "heatmaps": output_dir / "heatmaps",
        "maps": output_dir / "maps",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def region_to_dict_360(region: RegionBounds) -> Dict[str, float]:
    return {
        "lat_min": float(region.lat_min),
        "lat_max": float(region.lat_max),
        "lon_min": float(region.lon_min),
        "lon_max": float(region.lon_max),
    }


def month_day_strings(time_values: np.ndarray) -> np.ndarray:
    dates = np.asarray(time_values, dtype="datetime64[D]")
    return np.array([str(value)[5:] for value in dates], dtype=object)


def target_month_numbers(names: Sequence[str]) -> List[int]:
    month_numbers: List[int] = []
    for name in names:
        if name not in MONTH_NAME_TO_NUMBER:
            raise ValueError(f"Unsupported target month {name!r}; expected subset of {sorted(MONTH_NAME_TO_NUMBER)}")
        month_numbers.append(MONTH_NAME_TO_NUMBER[name])
    if len(set(month_numbers)) != len(month_numbers):
        raise ValueError("Target months must be unique")
    return month_numbers


def parse_selected_cells(raw_values: Sequence[str], max_lag: int) -> List[SelectedCell]:
    selected: List[SelectedCell] = []
    for raw_value in raw_values:
        parts = raw_value.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --selected-map-cell value {raw_value!r}; expected Month:Phase:Lag, for example Jan:3:14"
            )
        month_text, phase_text, lag_text = parts
        month_number = MONTH_NAME_TO_NUMBER.get(month_text)
        if month_number is None:
            raise ValueError(f"Unsupported selected-cell month {month_text!r}")
        phase = int(phase_text)
        lag = int(lag_text)
        if phase < 1 or phase > 8:
            raise ValueError(f"Selected cell phase must be 1..8, got {phase}")
        if lag < 0 or lag > max_lag:
            raise ValueError(f"Selected cell lag must be in 0..{max_lag}, got {lag}")
        selected.append(SelectedCell(target_month=month_number, phase=phase, lag=lag))
    return selected


def download_rmm_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    destination.write_text(text, encoding="utf-8")


def phase_from_rmm_components(rmm1: np.ndarray, rmm2: np.ndarray) -> np.ndarray:
    angle = (np.degrees(np.arctan2(rmm2, rmm1)) + 360.0) % 360.0
    return ((np.floor((angle + 22.5) / 45.0).astype(np.int32) % 8) + 1).astype(np.int32)


def load_rmm_dataframe(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"RMM file not found: {path}")

    rows: List[Tuple[np.datetime64, float, float, int, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
        except ValueError:
            continue
        date_value = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "D")
        amplitude = float(np.hypot(rmm1, rmm2))
        if len(parts) >= 7:
            try:
                phase = int(float(parts[5]))
            except ValueError:
                phase = int(phase_from_rmm_components(np.asarray([rmm1]), np.asarray([rmm2]))[0])
        else:
            phase = int(phase_from_rmm_components(np.asarray([rmm1]), np.asarray([rmm2]))[0])
        rows.append((date_value, rmm1, rmm2, phase, amplitude))

    if not rows:
        raise ValueError(f"No usable RMM rows parsed from {path}")

    dates = np.asarray([row[0] for row in rows], dtype="datetime64[D]")
    rmm1 = np.asarray([row[1] for row in rows], dtype=np.float64)
    rmm2 = np.asarray([row[2] for row in rows], dtype=np.float64)
    phase = np.asarray([row[3] for row in rows], dtype=np.int32)
    amplitude = np.asarray([row[4] for row in rows], dtype=np.float64)
    active = amplitude > ACTIVE_AMPLITUDE_THRESHOLD
    return {
        "date": dates,
        "rmm1": rmm1,
        "rmm2": rmm2,
        "phase": phase,
        "amplitude": amplitude,
        "active": active,
    }


def subset_hourly_t2m_to_region(path: Path, region: RegionBounds, target_months: Sequence[int]) -> xr.DataArray:
    year = int(path.stem.split("_")[1])
    selected_months = sorted(set(int(month) for month in target_months))
    if not selected_months:
        raise ValueError("target_months must not be empty")

    with Dataset(path) as ds:
        if ERA5_VARIABLE not in ds.variables:
            raise KeyError(f"Expected variable {ERA5_VARIABLE!r} in {path}")
        latitude_all = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        longitude_all = np.mod(np.asarray(ds.variables["longitude"][:], dtype=np.float64), 360.0)
        lat_index = np.where((latitude_all >= region.lat_min) & (latitude_all <= region.lat_max))[0]
        lon_index = np.where((longitude_all >= region.lon_min) & (longitude_all <= region.lon_max))[0]
        if lat_index.size == 0 or lon_index.size == 0:
            raise ValueError(f"Requested region {region_to_dict_360(region)} does not overlap {path}")

        var = ds.variables[ERA5_VARIABLE]
        daily_slices: List[np.ndarray] = []
        daily_dates: List[np.datetime64] = []
        for month in selected_months:
            n_days = calendar.monthrange(year, month)[1]
            start_hour = (date(year, month, 1) - date(year, 1, 1)).days * 24
            end_hour = start_hour + n_days * 24
            block = np.asarray(var[start_hour:end_hour, lat_index, lon_index], dtype=np.float32)
            block = block.reshape(n_days, 24, lat_index.size, lon_index.size)
            daily_slices.append(np.nanmean(block, axis=1, dtype=np.float32))
            month_dates = [np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "D") for day in range(1, n_days + 1)]
            daily_dates.extend(month_dates)

    daily_values = np.concatenate(daily_slices, axis=0).astype(np.float32)
    latitude = latitude_all[lat_index]
    longitude = longitude_all[lon_index]
    daily = xr.DataArray(
        daily_values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.asarray(daily_dates, dtype="datetime64[D]"),
            "latitude": latitude,
            "longitude": longitude,
        },
        name=ERA5_VARIABLE,
    )
    daily.attrs["description"] = "ERA5-Land daily mean 2m temperature"
    return daily


def write_daily_subset_if_needed(
    year: int,
    output_dir: Path,
    region: RegionBounds,
    target_months: Sequence[int],
    reuse_existing: bool,
) -> Path:
    destination = output_dir / f"era5land_t2m_daily_mean_{year:04d}.nc"
    if destination.exists() and reuse_existing:
        return destination

    source = Path(ERA5_LAND_ROOT) / "2m_temperature" / ERA5_FILE_TEMPLATE.format(year=year)
    if not source.exists():
        raise FileNotFoundError(f"ERA5-Land file not found for {year}: {source}")
    print(
        f"Building daily ERA5-Land subset for {year} over target months "
        + ",".join(MONTH_NUMBER_TO_NAME[month] for month in target_months),
        flush=True,
    )
    daily = subset_hourly_t2m_to_region(source, region, target_months)
    ds = daily.to_dataset(name=ERA5_VARIABLE)
    ds.attrs["description"] = "Subset daily mean ERA5-Land T2m for MJO lag composite analysis"
    ds.attrs["subset_region_360"] = json.dumps(region_to_dict_360(region), sort_keys=True)
    ds.attrs["target_months"] = ",".join(MONTH_NUMBER_TO_NAME[month] for month in target_months)
    encoding = {
        ERA5_VARIABLE: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "chunksizes": [min(31, daily.sizes["time"]), daily.sizes["latitude"], daily.sizes["longitude"]],
            "_FillValue": np.float32(np.nan),
        }
    }
    ds.to_netcdf(destination, engine=NETCDF_ENGINE, encoding=encoding)
    return destination


def discover_available_era5_hourly_years() -> List[int]:
    years: List[int] = []
    for path in sorted(ERA5_HOURLY_DIR.glob("ERA5_*_2m_temperature.nc")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            years.append(int(parts[1]))
        except ValueError:
            continue
    if not years:
        raise FileNotFoundError(f"No ERA5-Land hourly T2m files found under {ERA5_HOURLY_DIR}")
    return sorted(set(years))


def load_daily_subset(path: Path) -> xr.DataArray:
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        return ds[ERA5_VARIABLE].load()


def build_dayofyear_climatology(daily_files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    monthday_order: List[str] = []
    monthday_index: Dict[str, int] = {}
    sums: List[np.ndarray] = []
    counts: List[np.ndarray] = []

    for file_index, path in enumerate(daily_files):
        daily = load_daily_subset(path)
        if latitude is None:
            latitude = np.asarray(daily["latitude"].values, dtype=np.float64)
            longitude = np.asarray(daily["longitude"].values, dtype=np.float64)
        values = np.asarray(daily.values, dtype=np.float64)
        labels = month_day_strings(np.asarray(daily["time"].values, dtype="datetime64[D]"))
        for time_index, label in enumerate(labels.tolist()):
            if label not in monthday_index:
                monthday_index[label] = len(monthday_order)
                monthday_order.append(label)
                sums.append(np.zeros(values.shape[1:], dtype=np.float64))
                counts.append(np.zeros(values.shape[1:], dtype=np.int32))
            index = monthday_index[label]
            slice_values = values[time_index]
            valid = np.isfinite(slice_values)
            sums[index][valid] += slice_values[valid]
            counts[index][valid] += 1
        if (file_index + 1) % 10 == 0 or file_index + 1 == len(daily_files):
            print(f"Built climatology contributions from {file_index + 1}/{len(daily_files)} yearly daily files", flush=True)

    if latitude is None or longitude is None:
        raise ValueError("No daily files were available for climatology construction")

    climatology = np.full((len(monthday_order), latitude.size, longitude.size), np.nan, dtype=np.float64)
    for index in range(len(monthday_order)):
        valid = counts[index] > 0
        climatology[index, valid] = sums[index][valid] / counts[index][valid]
    return (
        np.asarray(latitude, dtype=np.float64),
        np.asarray(longitude, dtype=np.float64),
        np.asarray(monthday_order, dtype=object),
        climatology,
    )


def infer_sierra_subregion_masks(label_path: Path, latitude: np.ndarray, longitude: np.ndarray) -> Dict[str, np.ndarray]:
    label_latitude, label_longitude, cleaned_labels, regions = load_label_mask(label_path)
    lat_index = np.where((label_latitude >= latitude.min() - 1.0e-6) & (label_latitude <= latitude.max() + 1.0e-6))[0]
    lon_index = np.where((label_longitude >= longitude.min() - 1.0e-6) & (label_longitude <= longitude.max() + 1.0e-6))[0]
    if lat_index.size != latitude.size or lon_index.size != longitude.size:
        raise ValueError("Could not align Sierra label grid to the ERA5 daily subset coordinates")
    label_latitude = label_latitude[lat_index]
    label_longitude = label_longitude[lon_index]
    cleaned_labels = cleaned_labels[np.ix_(lat_index, lon_index)]

    if np.allclose(label_latitude[::-1], latitude, atol=1.0e-6):
        label_latitude = label_latitude[::-1]
        cleaned_labels = cleaned_labels[::-1, :]

    lat_matches = np.allclose(label_latitude, latitude, atol=1.0e-6)
    lon_matches = np.allclose(label_longitude, longitude, atol=1.0e-6)
    if not lat_matches or not lon_matches:
        raise ValueError("Sierra label grid does not match ERA5 daily subset grid")

    rows_by_name: Dict[str, Tuple[float, int]] = {}
    for region in regions:
        if region.semantic_label == 3:
            rows_by_name["northern_sierra"] = (region.centroid_lat, region.source_cleaned_label)
        else:
            rows_by_name[f"remaining_{region.source_cleaned_label}"] = (region.centroid_lat, region.source_cleaned_label)

    remaining = sorted(
        [value for key, value in rows_by_name.items() if key.startswith("remaining_")],
        key=lambda item: item[0],
    )
    if len(remaining) != 2:
        raise ValueError(f"Expected two non-northern Sierra subregions, found {remaining}")

    southern_label = remaining[0][1]
    central_label = remaining[1][1]
    northern_label = rows_by_name["northern_sierra"][1]
    return {
        "northern_sierra": cleaned_labels == northern_label,
        "central_sierra": cleaned_labels == central_label,
        "southern_sierra": cleaned_labels == southern_label,
    }


def build_region_specs(
    latitude: np.ndarray,
    longitude: np.ndarray,
    climatology: np.ndarray,
    label_path: Path,
) -> List[RegionSpec]:
    landmask = np.isfinite(np.nanmean(climatology, axis=0))
    lat2d = np.broadcast_to(latitude[:, np.newaxis], landmask.shape)
    lon2d = np.broadcast_to(longitude[np.newaxis, :], landmask.shape)

    broad_mask = (
        (lat2d >= BROAD_CALIFORNIA_REGION_360.lat_min)
        & (lat2d <= BROAD_CALIFORNIA_REGION_360.lat_max)
        & (lon2d >= BROAD_CALIFORNIA_REGION_360.lon_min)
        & (lon2d <= BROAD_CALIFORNIA_REGION_360.lon_max)
        & landmask
    )
    sierra_mask = (
        (lat2d >= SIERRA_REGION_360.lat_min)
        & (lat2d <= SIERRA_REGION_360.lat_max)
        & (lon2d >= SIERRA_REGION_360.lon_min)
        & (lon2d <= SIERRA_REGION_360.lon_max)
        & landmask
    )
    subregions = infer_sierra_subregion_masks(label_path, latitude, longitude)

    return [
        RegionSpec(
            key="broad_california",
            title="Broad California",
            description="Repo broad California analysis box with land-only ERA5-Land cells",
            mask=broad_mask,
        ),
        RegionSpec(
            key="sierra_region",
            title="Sierra Region",
            description="Geographic Sierra box lat 35..42, lon 236..243 in 0..360 longitude",
            mask=sierra_mask,
        ),
        RegionSpec(
            key="northern_sierra",
            title="Northern Sierra",
            description="Top-20 Sierra subregion with highest centroid latitude on the cleaned label grid",
            mask=subregions["northern_sierra"] & landmask,
        ),
        RegionSpec(
            key="central_sierra",
            title="Central Sierra",
            description="Among the two non-northern cleaned Sierra subregions, the one with higher centroid latitude",
            mask=subregions["central_sierra"] & landmask,
        ),
        RegionSpec(
            key="southern_sierra",
            title="Southern Sierra",
            description="Among the two non-northern cleaned Sierra subregions, the one with lower centroid latitude",
            mask=subregions["southern_sierra"] & landmask,
        ),
    ]


def weighted_mean(field_2d: np.ndarray, mask_2d: np.ndarray, latitude: np.ndarray) -> float:
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, np.newaxis], mask_2d.shape)
    valid = mask_2d & np.isfinite(field_2d)
    if not np.any(valid):
        return float("nan")
    numerator = np.sum(field_2d[valid] * weights[valid])
    denominator = np.sum(weights[valid])
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def build_regional_daily_anomalies(
    daily_files: Sequence[Path],
    latitude: np.ndarray,
    monthday_labels: np.ndarray,
    climatology: np.ndarray,
    region_specs: Sequence[RegionSpec],
) -> Dict[str, np.ndarray]:
    label_to_index = {str(label): index for index, label in enumerate(monthday_labels.tolist())}
    dates: List[np.datetime64] = []
    regional_rows: List[np.ndarray] = []

    for file_index, path in enumerate(daily_files):
        daily = load_daily_subset(path)
        values = np.asarray(daily.values, dtype=np.float64)
        day_labels = month_day_strings(np.asarray(daily["time"].values, dtype="datetime64[D]"))
        for time_index, day_label in enumerate(day_labels.tolist()):
            clim_index = label_to_index[day_label]
            anomaly = values[time_index] - climatology[clim_index]
            row = np.array(
                [weighted_mean(anomaly, region.mask, latitude) for region in region_specs],
                dtype=np.float64,
            )
            dates.append(np.asarray(daily["time"].values[time_index], dtype="datetime64[D]"))
            regional_rows.append(row)
        if (file_index + 1) % 10 == 0 or file_index + 1 == len(daily_files):
            print(f"Built regional anomalies from {file_index + 1}/{len(daily_files)} yearly daily files", flush=True)

    regional_values = np.vstack(regional_rows) if regional_rows else np.zeros((0, len(region_specs)), dtype=np.float64)
    date_values = np.asarray(dates, dtype="datetime64[D]")
    month_values = np.array([int(str(value)[5:7]) for value in date_values], dtype=np.int32)
    yearmonth_codes = np.array([int(str(value)[:4] + str(value)[5:7]) for value in date_values], dtype=np.int32)
    return {
        "date": date_values,
        "month": month_values,
        "yearmonth": yearmonth_codes,
        "values": regional_values,
    }


def compute_monthly_region_standard_deviation(
    regional_daily: Dict[str, np.ndarray],
    target_months: Sequence[int],
) -> np.ndarray:
    values = regional_daily["values"]
    months = regional_daily["month"]
    result = np.full((len(target_months), values.shape[1]), np.nan, dtype=np.float64)
    for month_index, month_number in enumerate(target_months):
        month_mask = months == month_number
        if not np.any(month_mask):
            continue
        result[month_index] = np.nanstd(values[month_mask], axis=0, ddof=1)
    return result


def build_active_sample_table(
    rmm: Dict[str, np.ndarray],
    regional_daily: Dict[str, np.ndarray],
    target_months: Sequence[int],
    max_lag: int,
) -> Dict[str, np.ndarray]:
    target_month_lookup = {month: index for index, month in enumerate(target_months)}
    daily_date_to_index = {
        np.datetime64(date_value, "D"): index for index, date_value in enumerate(np.asarray(regional_daily["date"], dtype="datetime64[D]"))
    }
    rmm_date_to_index = {
        np.datetime64(date_value, "D"): index for index, date_value in enumerate(np.asarray(rmm["date"], dtype="datetime64[D]"))
    }

    overlap_dates = sorted(set(daily_date_to_index).intersection(set(rmm_date_to_index)))
    if not overlap_dates:
        raise ValueError("No overlapping dates between RMM* and ERA5-Land daily anomalies")

    active_rmm_indices: List[int] = []
    for date_value in overlap_dates:
        rmm_index = rmm_date_to_index[date_value]
        if bool(rmm["active"][rmm_index]):
            active_rmm_indices.append(rmm_index)

    active_index_lookup = {rmm_index: active_index for active_index, rmm_index in enumerate(active_rmm_indices)}
    sample_active_index: List[int] = []
    sample_phase: List[int] = []
    sample_target_month_index: List[int] = []
    sample_lag: List[int] = []
    sample_target_daily_index: List[int] = []
    sample_target_date: List[np.datetime64] = []
    sample_base_yearmonth: List[int] = []

    for rmm_index in active_rmm_indices:
        base_date = np.asarray(rmm["date"][rmm_index], dtype="datetime64[D]")
        phase = int(rmm["phase"][rmm_index])
        base_yearmonth = int(str(base_date)[:4] + str(base_date)[5:7])
        for lag in range(max_lag + 1):
            target_date = base_date + np.timedelta64(lag, "D")
            daily_index = daily_date_to_index.get(target_date)
            if daily_index is None:
                continue
            target_month = int(str(target_date)[5:7])
            if target_month not in target_month_lookup:
                continue
            sample_active_index.append(active_index_lookup[rmm_index])
            sample_phase.append(phase)
            sample_target_month_index.append(target_month_lookup[target_month])
            sample_lag.append(lag)
            sample_target_daily_index.append(daily_index)
            sample_target_date.append(target_date)
            sample_base_yearmonth.append(base_yearmonth)

    if not sample_target_daily_index:
        raise ValueError("No active MJO phase-lag samples were available for the requested target months")

    active_dates = np.asarray([rmm["date"][index] for index in active_rmm_indices], dtype="datetime64[D]")
    active_phases = np.asarray([rmm["phase"][index] for index in active_rmm_indices], dtype=np.int32)
    active_base_yearmonth = np.asarray(
        [int(str(date_value)[:4] + str(date_value)[5:7]) for date_value in active_dates],
        dtype=np.int32,
    )
    return {
        "active_dates": active_dates,
        "active_phases": active_phases,
        "active_base_yearmonth": active_base_yearmonth,
        "sample_active_index": np.asarray(sample_active_index, dtype=np.int32),
        "sample_phase": np.asarray(sample_phase, dtype=np.int32),
        "sample_target_month_index": np.asarray(sample_target_month_index, dtype=np.int32),
        "sample_lag": np.asarray(sample_lag, dtype=np.int32),
        "sample_target_daily_index": np.asarray(sample_target_daily_index, dtype=np.int32),
        "sample_target_date": np.asarray(sample_target_date, dtype="datetime64[D]"),
        "sample_base_yearmonth": np.asarray(sample_base_yearmonth, dtype=np.int32),
    }


def summarize_samples(
    sample_table: Dict[str, np.ndarray],
    regional_daily: Dict[str, np.ndarray],
    target_months: Sequence[int],
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    region_values = regional_daily["values"][sample_table["sample_target_daily_index"]]
    n_regions = region_values.shape[1]
    sums = np.zeros((len(target_months), 8, max_lag + 1, n_regions), dtype=np.float64)
    counts = np.zeros((len(target_months), 8, max_lag + 1, n_regions), dtype=np.int32)

    month_index = sample_table["sample_target_month_index"]
    phase_index = sample_table["sample_phase"] - 1
    lag_index = sample_table["sample_lag"]
    for region_index in range(n_regions):
        values = region_values[:, region_index]
        valid = np.isfinite(values)
        np.add.at(sums[..., region_index], (month_index[valid], phase_index[valid], lag_index[valid]), values[valid])
        np.add.at(counts[..., region_index], (month_index[valid], phase_index[valid], lag_index[valid]), 1)
    return sums, counts


def permutation_p_values(
    sample_table: Dict[str, np.ndarray],
    regional_daily: Dict[str, np.ndarray],
    observed_means: np.ndarray,
    observed_counts: np.ndarray,
    target_months: Sequence[int],
    max_lag: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    if n_permutations <= 0:
        raise ValueError("Permutation p-value calculation requires n_permutations > 0")

    rng = np.random.default_rng(seed)
    region_values = regional_daily["values"][sample_table["sample_target_daily_index"]]
    n_regions = region_values.shape[1]
    exceedances = np.zeros_like(observed_means, dtype=np.int32)
    phase_base = sample_table["active_phases"].copy()
    sample_active_index = sample_table["sample_active_index"]
    month_index = sample_table["sample_target_month_index"]
    lag_index = sample_table["sample_lag"]

    block_to_members: Dict[int, np.ndarray] = {}
    unique_blocks = np.unique(sample_table["active_base_yearmonth"])
    for block in unique_blocks.tolist():
        block_to_members[int(block)] = np.where(sample_table["active_base_yearmonth"] == block)[0]

    for permutation_index in range(n_permutations):
        permuted_phase = phase_base.copy()
        for block, members in block_to_members.items():
            if members.size <= 1:
                continue
            permuted_phase[members] = rng.permutation(permuted_phase[members])
        phase_index = permuted_phase[sample_active_index] - 1
        perm_sums = np.zeros((len(target_months), 8, max_lag + 1, n_regions), dtype=np.float64)
        perm_counts = np.zeros((len(target_months), 8, max_lag + 1, n_regions), dtype=np.int32)
        for region_index in range(n_regions):
            values = region_values[:, region_index]
            valid = np.isfinite(values)
            np.add.at(perm_sums[..., region_index], (month_index[valid], phase_index[valid], lag_index[valid]), values[valid])
            np.add.at(perm_counts[..., region_index], (month_index[valid], phase_index[valid], lag_index[valid]), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            perm_means = perm_sums / perm_counts
        valid_observed = observed_counts > 0
        exceedances[valid_observed] += np.abs(perm_means[valid_observed]) >= np.abs(observed_means[valid_observed])
        if (permutation_index + 1) % 25 == 0 or permutation_index + 1 == n_permutations:
            print(f"Completed permutation {permutation_index + 1}/{n_permutations}", flush=True)

    p_values = np.full_like(observed_means, np.nan, dtype=np.float64)
    valid_observed = observed_counts > 0
    p_values[valid_observed] = (exceedances[valid_observed] + 1.0) / (n_permutations + 1.0)
    return p_values


def write_summary_csv(
    path: Path,
    region_specs: Sequence[RegionSpec],
    target_months: Sequence[int],
    max_lag: int,
    means: np.ndarray,
    counts: np.ndarray,
    standardized: np.ndarray,
    p_values: np.ndarray | None,
) -> None:
    fieldnames = [
        "target_month",
        "region",
        "phase",
        "lag",
        "sample_size",
        "mean_T2m_anomaly",
        "standardized_T2m_anomaly",
    ]
    if p_values is not None:
        fieldnames.extend(["p_value", "significant_at_0_05"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for month_index, month_number in enumerate(target_months):
            for region_index, region in enumerate(region_specs):
                for phase in range(1, 9):
                    for lag in range(max_lag + 1):
                        row = {
                            "target_month": MONTH_NUMBER_TO_NAME[month_number],
                            "region": region.key,
                            "phase": phase,
                            "lag": lag,
                            "sample_size": int(counts[month_index, phase - 1, lag, region_index]),
                            "mean_T2m_anomaly": float(means[month_index, phase - 1, lag, region_index]),
                            "standardized_T2m_anomaly": float(standardized[month_index, phase - 1, lag, region_index]),
                        }
                        if p_values is not None:
                            p_value = p_values[month_index, phase - 1, lag, region_index]
                            row["p_value"] = float(p_value) if np.isfinite(p_value) else ""
                            row["significant_at_0_05"] = bool(np.isfinite(p_value) and p_value <= 0.05)
                        writer.writerow(row)


def plot_heatmaps(
    output_dir: Path,
    region_specs: Sequence[RegionSpec],
    target_months: Sequence[int],
    means: np.ndarray,
    counts: np.ndarray,
    p_values: np.ndarray | None,
) -> None:
    vmax = float(np.nanmax(np.abs(means))) if np.any(np.isfinite(means)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    for month_index, month_number in enumerate(target_months):
        month_name = MONTH_NUMBER_TO_NAME[month_number]
        xticks = np.arange(0, means.shape[2], 7)
        if xticks[-1] != means.shape[2] - 1:
            xticks = np.append(xticks, means.shape[2] - 1)
        for region_index, region in enumerate(region_specs):
            if region.key != "sierra_region":
                continue
            data = means[month_index, :, :, region_index]
            fig, ax = plt.subplots(figsize=(10.2, 4.8))
            mesh = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=(-0.5, data.shape[1] - 0.5, 0.5, 8.5),
            )
            ax.set_title(
                f"{month_name} | {region.title}\n"
                f"Target T2m dates: all {month_name} days across all available years. "
                f"MJO date = target T2m date - lag."
            )
            ax.set_xlabel("Lag before target T2m day (days)")
            ax.set_ylabel("RMM* phase")
            ax.set_yticks(np.arange(1, 9))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(lag)) for lag in xticks.tolist()])
            if p_values is not None:
                significant = np.isfinite(p_values[month_index, :, :, region_index]) & (
                    p_values[month_index, :, :, region_index] <= 0.05
                )
                phase_idx, lag_idx = np.where(significant)
                ax.scatter(lag_idx, phase_idx + 1, s=12, facecolors="none", edgecolors="black", linewidths=0.7)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Mean regional T2m anomaly (K)")
            output_path = output_dir / f"{month_name.lower()}_{region.key}_phase_lag_heatmap.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)


def plot_sample_size_heatmaps(
    output_dir: Path,
    region_specs: Sequence[RegionSpec],
    target_months: Sequence[int],
    counts: np.ndarray,
) -> None:
    vmax = int(np.nanmax(counts)) if counts.size else 1
    vmax = max(vmax, 1)
    for month_index, month_number in enumerate(target_months):
        month_name = MONTH_NUMBER_TO_NAME[month_number]
        for region_index, region in enumerate(region_specs):
            if region.key != "sierra_region":
                continue
            data = counts[month_index, :, :, region_index]
            fig, ax = plt.subplots(figsize=(10.2, 4.8))
            mesh = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="YlGnBu",
                vmin=0,
                vmax=vmax,
                extent=(-0.5, data.shape[1] - 0.5, 0.5, 8.5),
            )
            ax.set_title(f"{month_name} | {region.title}\nSample size by phase and lag")
            ax.set_xlabel("Lag (days)")
            ax.set_ylabel("RMM* phase")
            ax.set_yticks(np.arange(1, 9))
            ax.set_xticks(np.arange(0, data.shape[1], 7))
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Number of samples")
            output_path = output_dir / f"{month_name.lower()}_{region.key}_sample_size_heatmap.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)


def lag_window_bounds(max_lag: int) -> List[Tuple[int, int, str]]:
    windows: List[Tuple[int, int, str]] = []
    start = 0
    while start <= max_lag:
        end = min(start + LAG_WINDOW_SIZE, max_lag + 1) - 1
        windows.append((start, end, f"{start}-{end}"))
        start = end + 1
    return windows


def plot_lag_response_lines(
    output_dir: Path,
    region_specs: Sequence[RegionSpec],
    target_months: Sequence[int],
    means: np.ndarray,
    counts: np.ndarray,
) -> None:
    lags = np.arange(means.shape[2], dtype=np.int32)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, 8))
    for month_index, month_number in enumerate(target_months):
        month_name = MONTH_NUMBER_TO_NAME[month_number]
        for region_index, region in enumerate(region_specs):
            if region.key != "sierra_region":
                continue
            fig, ax = plt.subplots(figsize=(10.4, 5.0))
            for phase_index in range(8):
                ax.plot(
                    lags,
                    means[month_index, phase_index, :, region_index],
                    color=colors[phase_index],
                    linewidth=1.8,
                    marker="o",
                    markersize=3.2,
                    label=f"Phase {phase_index + 1}",
                )
            ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
            ax.set_title(
                f"{month_name} | {region.title}\n"
                f"Target T2m dates: all {month_name} days across all available years. "
                f"MJO date = target T2m date - lag."
            )
            ax.set_xlabel("Lag before target T2m day (days)")
            ax.set_ylabel("Mean regional daily-mean T2m anomaly (K)")
            ax.set_xlim(0, means.shape[2] - 1)
            ax.set_xticks(np.arange(0, means.shape[2], 7))
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
            output_path = output_dir / f"{month_name.lower()}_{region.key}_lag_response_lines.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10.4, 5.0))
            for phase_index in range(8):
                ax.plot(
                    lags,
                    counts[month_index, phase_index, :, region_index],
                    color=colors[phase_index],
                    linewidth=1.8,
                    marker="o",
                    markersize=3.2,
                    label=f"Phase {phase_index + 1}",
                )
            ax.set_title(f"{month_name} | {region.title}\nSample size by lag and phase")
            ax.set_xlabel("Lag before target T2m day (days)")
            ax.set_ylabel("Sample size")
            ax.set_xlim(0, counts.shape[2] - 1)
            ax.set_xticks(np.arange(0, counts.shape[2], 7))
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
            output_path = output_dir / f"{month_name.lower()}_{region.key}_lag_response_sample_size.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)


def plot_lag_window_summaries(
    output_dir: Path,
    region_specs: Sequence[RegionSpec],
    target_months: Sequence[int],
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    windows = lag_window_bounds(counts.shape[2] - 1)
    window_labels = [label for _, _, label in windows]
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, 8))
    x = np.arange(len(windows), dtype=np.float64)
    width = 0.095
    offsets = (np.arange(8, dtype=np.float64) - 3.5) * width

    for month_index, month_number in enumerate(target_months):
        month_name = MONTH_NUMBER_TO_NAME[month_number]
        for region_index, region in enumerate(region_specs):
            if region.key != "sierra_region":
                continue
            window_means = np.full((8, len(windows)), np.nan, dtype=np.float64)
            window_counts = np.zeros((8, len(windows)), dtype=np.int32)
            for window_index, (start, end, _) in enumerate(windows):
                lag_slice = slice(start, end + 1)
                summed = np.sum(sums[month_index, :, lag_slice, region_index], axis=1)
                counted = np.sum(counts[month_index, :, lag_slice, region_index], axis=1)
                valid = counted > 0
                window_counts[:, window_index] = counted
                window_means[valid, window_index] = summed[valid] / counted[valid]

            fig, ax = plt.subplots(figsize=(10.6, 5.2))
            for phase_index in range(8):
                ax.bar(
                    x + offsets[phase_index],
                    window_means[phase_index],
                    width=width,
                    color=colors[phase_index],
                    label=f"Phase {phase_index + 1}",
                )
            ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
            ax.set_title(
                f"{month_name} | {region.title}\n"
                f"Lag-window composite means using windows {', '.join(window_labels)} days"
            )
            ax.set_xlabel("Lag window before target T2m day (days)")
            ax.set_ylabel("Mean regional daily-mean T2m anomaly (K)")
            ax.set_xticks(x)
            ax.set_xticklabels(window_labels)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
            output_path = output_dir / f"{month_name.lower()}_{region.key}_lag_window_summary.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10.6, 5.2))
            for phase_index in range(8):
                ax.bar(
                    x + offsets[phase_index],
                    window_counts[phase_index],
                    width=width,
                    color=colors[phase_index],
                    label=f"Phase {phase_index + 1}",
                )
            ax.set_title(f"{month_name} | {region.title}\nSample size aggregated by lag window")
            ax.set_xlabel("Lag window before target T2m day (days)")
            ax.set_ylabel("Sample size")
            ax.set_xticks(x)
            ax.set_xticklabels(window_labels)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
            output_path = output_dir / f"{month_name.lower()}_{region.key}_lag_window_sample_size.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)


def composite_map_for_dates(
    dates_to_average: np.ndarray,
    region: RegionBounds,
    climatology_lookup: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if dates_to_average.size == 0:
        raise ValueError("No dates were supplied for composite map generation")
    normalized_dates = np.asarray(dates_to_average, dtype="datetime64[D]")
    years = sorted(set(int(str(date_value)[:4]) for date_value in normalized_dates))
    dates_set = {np.datetime64(date_value, "D") for date_value in normalized_dates}
    accumulated: np.ndarray | None = None
    counts: np.ndarray | None = None
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    n_samples = 0

    for year in years:
        source = Path(ERA5_LAND_ROOT) / "2m_temperature" / ERA5_FILE_TEMPLATE.format(year=year)
        requested_months = sorted({int(str(date_value)[5:7]) for date_value in normalized_dates})
        daily = subset_hourly_t2m_to_region(source, region, requested_months)
        values = np.asarray(daily.values, dtype=np.float64)
        day_labels = month_day_strings(np.asarray(daily["time"].values, dtype="datetime64[D]"))
        if latitude is None:
            latitude = np.asarray(daily["latitude"].values, dtype=np.float64)
            longitude = np.asarray(daily["longitude"].values, dtype=np.float64)
            accumulated = np.zeros(values.shape[1:], dtype=np.float64)
            counts = np.zeros(values.shape[1:], dtype=np.int32)
        for time_index, date_value in enumerate(np.asarray(daily["time"].values, dtype="datetime64[D]")):
            if date_value not in dates_set:
                continue
            anomaly = values[time_index] - climatology_lookup[str(date_value)[5:]]
            valid = np.isfinite(anomaly)
            accumulated[valid] += anomaly[valid]
            counts[valid] += 1
            n_samples += 1

    if accumulated is None or counts is None or latitude is None or longitude is None:
        raise ValueError("Composite map pass produced no data")
    composite = np.full_like(accumulated, np.nan, dtype=np.float64)
    valid = counts > 0
    composite[valid] = accumulated[valid] / counts[valid]
    return latitude, longitude, composite, n_samples


def plot_composite_map(
    output_path: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    composite: np.ndarray,
    title: str,
) -> None:
    vmax = float(np.nanmax(np.abs(composite))) if np.any(np.isfinite(composite)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    mesh = ax.pcolormesh(longitude, latitude, composite, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Longitude (0..360)")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Composite T2m anomaly (K)")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    region_specs: Sequence[RegionSpec],
    daily_files: Sequence[Path],
    rmm: Dict[str, np.ndarray],
    regional_daily: Dict[str, np.ndarray],
    sample_table: Dict[str, np.ndarray],
    p_values: np.ndarray | None,
) -> None:
    payload = {
        "analysis": "month-specific lagged active-MJO phase composites for overland ERA5-Land T2m anomalies",
        "target_months": list(args.target_months),
        "year_start": int(args.year_start),
        "year_end": int(args.year_end),
        "max_lag_days": int(args.max_lag),
        "active_definition": f"sqrt(RMM1*^2 + RMM2*^2) > {ACTIVE_AMPLITUDE_THRESHOLD}",
        "rmm_file": str(args.rmm_file),
        "rmm_url": str(args.rmm_url),
        "era5_root": str(ERA5_LAND_ROOT),
        "daily_intermediate_files": [str(path_value) for path_value in daily_files],
        "analysis_region_360": region_to_dict_360(BROAD_CALIFORNIA_REGION_360),
        "sierra_region_360": region_to_dict_360(SIERRA_REGION_360),
        "western_us_map_region_360": region_to_dict_360(WESTERN_US_MAP_REGION_360),
        "label_netcdf": str(args.label_netcdf),
        "regions": [
            {
                "key": region.key,
                "title": region.title,
                "description": region.description,
                "n_mask_cells": int(np.count_nonzero(region.mask)),
            }
            for region in region_specs
        ],
        "n_rmm_days_total": int(rmm["date"].size),
        "n_active_rmm_days_overlap": int(sample_table["active_dates"].size),
        "n_phase_lag_samples": int(sample_table["sample_target_date"].size),
        "n_permutations": int(args.n_permutations),
        "p_values_included": bool(p_values is not None),
        "regional_daily_start": str(regional_daily["date"][0]) if regional_daily["date"].size else "",
        "regional_daily_end": str(regional_daily["date"][-1]) if regional_daily["date"].size else "",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_lag < 0:
        raise ValueError("--max-lag must be non-negative")
    if args.year_end < args.year_start:
        raise ValueError("--year-end must be >= --year-start")

    available_years = discover_available_era5_hourly_years()
    available_start = available_years[0]
    available_end = available_years[-1]
    if args.year_start < available_start:
        raise ValueError(
            f"Requested --year-start {args.year_start} is earlier than the first available ERA5-Land hourly year {available_start}"
        )
    if args.year_end > available_end:
        print(
            f"Requested --year-end {args.year_end} exceeds the last available ERA5-Land hourly year {available_end}; "
            f"clamping to {available_end}.",
            flush=True,
        )
        args.year_end = available_end

    target_months = target_month_numbers(args.target_months)
    selected_cells = parse_selected_cells(args.selected_map_cell, args.max_lag)
    invalid_selected_months = [cell.label for cell in selected_cells if cell.target_month not in target_months]
    if invalid_selected_months:
        raise ValueError(
            "Selected map cells must use one of the requested target months; invalid entries: "
            + ", ".join(invalid_selected_months)
        )
    output_paths = ensure_output_dirs(args.output_dir)

    if not args.skip_runtime_check:
        runtime = get_runtime()
        ensure_runtime_on_compute_node(runtime)

    if args.download_rmm and not args.rmm_file.exists():
        print(f"Downloading RMM* data from {args.rmm_url}", flush=True)
        download_rmm_file(args.rmm_url, args.rmm_file)

    print(f"Loading RMM* data from {args.rmm_file}", flush=True)
    rmm = load_rmm_dataframe(args.rmm_file)

    daily_files: List[Path] = []
    for year in range(args.year_start, args.year_end + 1):
        daily_path = write_daily_subset_if_needed(
            year=year,
            output_dir=output_paths["daily"],
            region=BROAD_CALIFORNIA_REGION_360,
            target_months=target_months,
            reuse_existing=args.reuse_daily_files,
        )
        daily_files.append(daily_path)
        print(f"Prepared daily ERA5-Land subset for {year}: {daily_path}", flush=True)

    latitude, longitude, monthday_labels, climatology = build_dayofyear_climatology(daily_files)
    region_specs = build_region_specs(latitude, longitude, climatology, args.label_netcdf)
    regional_daily = build_regional_daily_anomalies(daily_files, latitude, monthday_labels, climatology, region_specs)
    region_month_std = compute_monthly_region_standard_deviation(regional_daily, target_months)

    sample_table = build_active_sample_table(rmm, regional_daily, target_months, args.max_lag)
    sums, counts = summarize_samples(sample_table, regional_daily, target_months, args.max_lag)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts

    standardized = np.full_like(means, np.nan, dtype=np.float64)
    for month_index in range(len(target_months)):
        for region_index in range(len(region_specs)):
            std_value = region_month_std[month_index, region_index]
            if np.isfinite(std_value) and std_value > 0.0:
                standardized[month_index, :, :, region_index] = means[month_index, :, :, region_index] / std_value

    p_values: np.ndarray | None = None
    if args.n_permutations > 0:
        p_values = permutation_p_values(
            sample_table=sample_table,
            regional_daily=regional_daily,
            observed_means=means,
            observed_counts=counts,
            target_months=target_months,
            max_lag=args.max_lag,
            n_permutations=args.n_permutations,
            seed=args.permutation_seed,
        )

    write_summary_csv(
        path=DEFAULT_SUMMARY_CSV if args.output_dir == DEFAULT_OUTPUT_DIR else args.output_dir / DEFAULT_SUMMARY_CSV.name,
        region_specs=region_specs,
        target_months=target_months,
        max_lag=args.max_lag,
        means=means,
        counts=counts,
        standardized=standardized,
        p_values=p_values,
    )
    plot_heatmaps(output_paths["heatmaps"], region_specs, target_months, means, counts, p_values)
    plot_sample_size_heatmaps(
        output_dir=output_paths["heatmaps"],
        region_specs=region_specs,
        target_months=target_months,
        counts=counts,
    )
    plot_lag_response_lines(
        output_dir=output_paths["heatmaps"],
        region_specs=region_specs,
        target_months=target_months,
        means=means,
        counts=counts,
    )
    plot_lag_window_summaries(
        output_dir=output_paths["heatmaps"],
        region_specs=region_specs,
        target_months=target_months,
        sums=sums,
        counts=counts,
    )

    if selected_cells:
        climatology_lookup = {str(label): climatology[index] for index, label in enumerate(monthday_labels.tolist())}
        for cell in selected_cells:
            cell_mask = (
                (sample_table["sample_phase"] == cell.phase)
                & (sample_table["sample_lag"] == cell.lag)
                & (sample_table["sample_target_month_index"] == target_months.index(cell.target_month))
            )
            target_dates = sample_table["sample_target_date"][cell_mask]
            if target_dates.size == 0:
                print(f"Skipping map for {cell.label}: no matching samples", flush=True)
                continue
            lat_map, lon_map, composite, n_samples = composite_map_for_dates(
                dates_to_average=target_dates,
                region=WESTERN_US_MAP_REGION_360,
                climatology_lookup=climatology_lookup,
            )
            output_path = output_paths["maps"] / f"{cell.label}_western_us_composite.png"
            plot_composite_map(
                output_path=output_path,
                latitude=lat_map,
                longitude=lon_map,
                composite=composite,
                title=f"{cell.month_name} phase {cell.phase} lag {cell.lag} | western-U.S. overland T2m anomaly composite (n={n_samples})",
            )

    write_metadata(
        path=DEFAULT_METADATA_JSON if args.output_dir == DEFAULT_OUTPUT_DIR else args.output_dir / DEFAULT_METADATA_JSON.name,
        args=args,
        region_specs=region_specs,
        daily_files=daily_files,
        rmm=rmm,
        regional_daily=regional_daily,
        sample_table=sample_table,
        p_values=p_values,
    )
    print("Completed lagged MJO phase-composite analysis setup and outputs.", flush=True)


if __name__ == "__main__":
    main()
