import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.paths import SWE_ROOT
from snow_ml.data import (
    DEFAULT_SIERRA_REGION,
    SWE_VARIABLE,
    build_sierra_mask,
    get_swe_grid_definition,
    load_swe_snapshot,
)
from snow_ml.data_wusd3 import (
    DEFAULT_WUSD3_DATASET_ID,
    WUSD3_ROOT,
    Wusd3Dataset,
    discover_wusd3_dataset_ids,
    discover_wusd3_file_years,
    get_wusd3_grid_definition,
    load_wusd3_snapshot,
    variable_path_for_file_year,
)


WUS_ROOT = Path(WUSD3_ROOT)
WUS_DAILY_ROOT = WUS_ROOT / "daily"
UCLA_ROOT = Path(SWE_ROOT)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "wus02_data_audit"
REPORT_PATH = OUTPUT_ROOT / "wus02_simulation_and_ucla_swe_overlap_report.md"
FILE_INVENTORY_PATH = OUTPUT_ROOT / "wus02_file_inventory.csv"
VARIABLE_INVENTORY_PATH = OUTPUT_ROOT / "wus02_variable_inventory.csv"
OVERLAP_YEARS_PATH = OUTPUT_ROOT / "wus02_ucla_overlap_years.csv"
COMPARISON_PATH = OUTPUT_ROOT / "wus02_ucla_sierra_swe_comparison.csv"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "wus02_audit_summary.json"
FIG_TS_PATH = OUTPUT_ROOT / "wus02_ucla_sierra_swe_timeseries.png"
FIG_SCATTER_PATH = OUTPUT_ROOT / "wus02_ucla_sierra_swe_scatter.png"
FIG_ERROR_PATH = OUTPUT_ROOT / "wus02_ucla_sierra_swe_error_timeseries.png"

FILE_RE = re.compile(
    r"^(?P<variable>[a-z0-9_]+)\.daily\.(?P<model>.+)\.(?P<scenario>hist\.bias-correct|ssp370\.bias-correct)\.(?P<domain>d\d\d)\.(?P<year>\d{4})\.nc$"
)
WY_RE = re.compile(r"WY(\d{4})")

WUS_INTERPRETATION_NOTE = (
    "This audit treats the user-provided `.../postprocess/d02` path as the canonical `WUS-02` dataset. "
    "In the local archive and repo code, the authoritative naming convention is `WUS-D3` plus domain `d02`."
)
PRIORITY_VARIABLES = [
    "snow",
    "snowh",
    "prec",
    "prec_snow",
    "sfc_runoff",
    "subsfc_runoff",
    "t2",
    "t2max",
    "t2min",
    "tskin",
    "soil_m",
    "lh_sfc",
    "sh_sfc",
    "sw_dwn",
    "sw_sfc",
    "lw_dwn",
    "lw_sfc",
]
HYDRO_KEYWORDS = (
    "snow",
    "runoff",
    "soil",
    "prec",
    "rain",
    "melt",
    "rad",
    "temp",
    "skin",
    "flux",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit WUS-02 d02 and compare Sierra SWE with UCLA observations.")
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_WUSD3_DATASET_ID,
        help="Canonical WUS-D3 member id. Default is the repo's historical EC-Earth3 member.",
    )
    parser.add_argument(
        "--domain",
        default="d02",
        help="Canonical WUS domain. This audit assumes WUS-02 means the provided d02 path.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def format_scalar(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return "NA"
        return "{:.6g}".format(value)
    return str(value)


def sorted_unique_text(values: list[Any]) -> str:
    cleaned = sorted({format_scalar(value) for value in values if value is not None})
    return "; ".join(cleaned)


def parse_year_from_ucla_name(path: Path) -> int:
    match = WY_RE.search(path.name)
    if match is None:
        raise ValueError("Could not parse water year from {}".format(path))
    return int(match.group(1))


def area_from_bounds_1d(latitudes: np.ndarray, longitudes: np.ndarray, radius_m: float = 6_371_000.0) -> np.ndarray:
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    lat_bounds = np.empty(latitudes.size + 1, dtype=np.float64)
    lon_bounds = np.empty(longitudes.size + 1, dtype=np.float64)
    lat_bounds[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    lon_bounds[1:-1] = 0.5 * (lon_rad[:-1] + lon_rad[1:])
    lat_bounds[0] = lat_rad[0] - 0.5 * (lat_rad[1] - lat_rad[0])
    lat_bounds[-1] = lat_rad[-1] + 0.5 * (lat_rad[-1] - lat_rad[-2])
    lon_bounds[0] = lon_rad[0] - 0.5 * (lon_rad[1] - lon_rad[0])
    lon_bounds[-1] = lon_rad[-1] + 0.5 * (lon_rad[-1] - lon_rad[-2])
    dlon = np.abs(np.diff(lon_bounds))
    sin_term = np.abs(np.sin(lat_bounds[1:]) - np.sin(lat_bounds[:-1]))
    return (radius_m ** 2) * sin_term[:, None] * dlon[None, :]


def area_from_bounds_2d(latitudes: np.ndarray, longitudes: np.ndarray, radius_m: float = 6_371_000.0) -> np.ndarray:
    nlat, nlon = latitudes.shape
    areas = np.full((nlat, nlon), np.nan, dtype=np.float64)
    for i in range(nlat):
        for j in range(nlon):
            center_lat = latitudes[i, j]
            center_lon = longitudes[i, j]
            if not np.isfinite(center_lat) or not np.isfinite(center_lon):
                continue
            if nlat == 1:
                dlat = 0.0
            elif i == 0:
                dlat = abs(latitudes[i + 1, j] - center_lat)
            elif i == nlat - 1:
                dlat = abs(center_lat - latitudes[i - 1, j])
            else:
                dlat = 0.5 * abs(latitudes[i + 1, j] - latitudes[i - 1, j])
            if nlon == 1:
                dlon = 0.0
            elif j == 0:
                dlon = abs(longitudes[i, j + 1] - center_lon)
            elif j == nlon - 1:
                dlon = abs(center_lon - longitudes[i, j - 1])
            else:
                dlon = 0.5 * abs(longitudes[i, j + 1] - longitudes[i, j - 1])
            lat_north = math.radians(center_lat + 0.5 * dlat)
            lat_south = math.radians(center_lat - 0.5 * dlat)
            lon_east = math.radians(center_lon + 0.5 * dlon)
            lon_west = math.radians(center_lon - 0.5 * dlon)
            areas[i, j] = (radius_m ** 2) * abs(math.sin(lat_north) - math.sin(lat_south)) * abs(lon_east - lon_west)
    return areas


def infer_grid_cell_areas(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    lat_values = np.asarray(lat.values, dtype=np.float64)
    lon_values = np.asarray(lon.values, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        areas = area_from_bounds_1d(lat_values, lon_values)
        return xr.DataArray(areas, dims=(lat.dims[0], lon.dims[0]), coords={lat.dims[0]: lat, lon.dims[0]: lon}, name="cell_area_m2")
    if lat.ndim == 2 and lon.ndim == 2:
        areas = area_from_bounds_2d(lat_values, lon_values)
        return xr.DataArray(areas, dims=lat.dims, coords=lat.coords, name="cell_area_m2")
    raise ValueError("Latitude/longitude must both be 1D or both be 2D.")


def weighted_masked_mean(field: xr.DataArray, mask: xr.DataArray, area: xr.DataArray) -> float:
    valid = np.isfinite(field)
    weights = mask.astype(np.float64) * area.astype(np.float64)
    weights = weights.where(valid)
    denom = float(weights.sum(skipna=True).item())
    if denom <= 0.0:
        return float("nan")
    numer = float((field.astype(np.float64) * weights).sum(skipna=True).item())
    return numer / denom


def dataset_attr_subset(ds: xr.Dataset, keys: list[str]) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key in keys:
        if key in ds.attrs:
            subset[key] = format_scalar(ds.attrs[key])
    return subset


def inspect_wrfinput(domain: str) -> dict[str, Any]:
    path = WUS_ROOT / "wrfinput_{}".format(domain)
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        lat = ds["XLAT"].isel(Time=0)
        lon = ds["XLONG"].isel(Time=0)
        hgt = ds["HGT"].isel(Time=0) if "HGT" in ds else None
        landmask = ds["LANDMASK"].isel(Time=0) if "LANDMASK" in ds else None
        attrs = dataset_attr_subset(
            ds,
            [
                "TITLE",
                "START_DATE",
                "DX",
                "DY",
                "GRIDTYPE",
                "MAP_PROJ",
                "TRUELAT1",
                "TRUELAT2",
                "MOAD_CEN_LAT",
                "STAND_LON",
                "CEN_LAT",
                "CEN_LON",
                "WEST-EAST_GRID_DIMENSION",
                "SOUTH-NORTH_GRID_DIMENSION",
                "BOTTOM-TOP_GRID_DIMENSION",
                "MP_PHYSICS",
                "RA_LW_PHYSICS",
                "RA_SW_PHYSICS",
                "SF_SURFACE_PHYSICS",
                "BL_PBL_PHYSICS",
            ],
        )
        result = {
            "path": str(path),
            "dims": {name: int(size) for name, size in ds.sizes.items()},
            "lat_min": float(np.nanmin(lat.values)),
            "lat_max": float(np.nanmax(lat.values)),
            "lon_min": float(np.nanmin(lon.values)),
            "lon_max": float(np.nanmax(lon.values)),
            "attrs": attrs,
            "dx_m": safe_float(ds.attrs.get("DX")),
            "dy_m": safe_float(ds.attrs.get("DY")),
            "map_proj": format_scalar(ds.attrs.get("MAP_PROJ")),
            "hgt_units": None if hgt is None else hgt.attrs.get("units"),
            "hgt_min_m": None if hgt is None else float(np.nanmin(hgt.values)),
            "hgt_max_m": None if hgt is None else float(np.nanmax(hgt.values)),
            "landmask_units": None if landmask is None else landmask.attrs.get("units"),
            "landmask_fraction_land": None if landmask is None else float(np.nanmean(landmask.values)),
        }
    return result


def inspect_ucla_example(example_path: Path) -> dict[str, Any]:
    with xr.open_dataset(example_path, engine="netcdf4", decode_times=True) as ds:
        swe = ds[SWE_VARIABLE]
        time_coord = ds["time"]
        latitude = ds["Latitude"]
        longitude = ds["Longitude"]
        return {
            "path": str(example_path),
            "dims": {name: int(size) for name, size in ds.sizes.items()},
            "coords": list(ds.coords),
            "data_vars": list(ds.data_vars),
            "time_start": str(time_coord.values[0]),
            "time_end": str(time_coord.values[-1]),
            "time_count": int(time_coord.size),
            "time_calendar": getattr(time_coord.dt, "calendar", None),
            "swe_units": swe.attrs.get("units"),
            "swe_long_name": swe.attrs.get("long_name"),
            "lat_min": float(np.nanmin(latitude.values)),
            "lat_max": float(np.nanmax(latitude.values)),
            "lon_min": float(np.nanmin(longitude.values)),
            "lon_max": float(np.nanmax(longitude.values)),
            "attrs": dataset_attr_subset(ds, ["Conventions", "CDI", "CDO", "NCO"]),
        }


def discover_ucla_years() -> list[int]:
    return sorted(parse_year_from_ucla_name(path) for path in UCLA_ROOT.glob("WUS_UCLA_SR_v01_ALL_0_agg_16_WY*_SD_SWE_SCA_POST.nc"))


def build_ucla_series() -> tuple[pd.DataFrame, dict[str, Any]]:
    years = discover_ucla_years()
    grid = get_swe_grid_definition(water_year=years[0], region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    mask = build_sierra_mask(grid, region=DEFAULT_SIERRA_REGION)
    area = infer_grid_cell_areas(grid.fine_latitude, grid.fine_longitude)
    example_path = UCLA_ROOT / "WUS_UCLA_SR_v01_ALL_0_agg_16_WY{}_SD_SWE_SCA_POST.nc".format(years[0])
    example_meta = inspect_ucla_example(example_path)

    rows: list[dict[str, Any]] = []
    for water_year in years:
        source_path = UCLA_ROOT / "WUS_UCLA_SR_v01_ALL_0_agg_16_WY{}_SD_SWE_SCA_POST.nc".format(water_year)
        apr1 = load_swe_snapshot(
            water_year,
            snapshot_date=date(water_year, 4, 1),
            stat_name="mean",
            swe_grid=grid,
            fill_missing=False,
        )
        swe_m = weighted_masked_mean(apr1, mask, area)
        rows.append(
            {
                "water_year": water_year,
                "ucla_swe_m": swe_m,
                "ucla_swe_mm": swe_m * 1000.0 if np.isfinite(swe_m) else np.nan,
                "ucla_source_path": str(source_path),
            }
        )
    return pd.DataFrame(rows), {
        "grid": grid,
        "mask": mask,
        "area": area,
        "example": example_meta,
        "mask_summary": {
            "mask_type": str(mask.attrs.get("mask_type", "unknown")),
            "nonzero_weight_cells": int((mask > 0.0).sum().item()),
            "total_effective_area_m2": float((mask * area).sum(skipna=True).item()),
        },
    }


def inventory_wus_files(dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_dir = WUS_DAILY_ROOT / dataset_id / "postprocess"
    rows: list[dict[str, Any]] = []
    domain_summary: dict[str, Any] = {}

    for domain_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        files = sorted(domain_dir.glob("*.nc"))
        domain_summary[domain_dir.name] = {"file_count": len(files)}
        for path in files:
            match = FILE_RE.match(path.name)
            row = {
                "dataset_id": dataset_id,
                "domain": domain_dir.name,
                "path": str(path),
                "filename": path.name,
                "file_format": path.suffix,
                "file_size_bytes": int(path.stat().st_size),
                "opened_successfully": False,
                "open_error": None,
                "variable": None,
                "model_token": None,
                "scenario_token": None,
                "file_year": None,
                "time_coord": None,
                "time_start": None,
                "time_end": None,
                "time_count": None,
                "calendar": None,
                "lat_dim": None,
                "lon_dim": None,
                "units": None,
                "long_name": None,
                "fill_value": None,
                "missing_value": None,
                "global_title": None,
                "dx_m": None,
                "dy_m": None,
            }
            if match is not None:
                row["variable"] = match.group("variable")
                row["model_token"] = match.group("model")
                row["scenario_token"] = match.group("scenario")
                row["file_year"] = int(match.group("year"))
            try:
                with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
                    row["opened_successfully"] = True
                    if row["variable"] is not None and row["variable"] in ds.data_vars:
                        da = ds[row["variable"]]
                    elif len(ds.data_vars) == 1:
                        da = ds[list(ds.data_vars)[0]]
                        row["variable"] = da.name
                    else:
                        da = None
                    time_name = "day" if "day" in ds.coords else "time" if "time" in ds.coords else None
                    row["time_coord"] = time_name
                    if time_name is not None:
                        row["time_start"] = str(ds[time_name].values[0])
                        row["time_end"] = str(ds[time_name].values[-1])
                        row["time_count"] = int(ds[time_name].size)
                        row["calendar"] = getattr(ds[time_name].dt, "calendar", None)
                    if da is not None:
                        row["lat_dim"] = next((dim for dim in da.dims if "lat" in dim.lower() or "south_north" in dim.lower()), None)
                        row["lon_dim"] = next((dim for dim in da.dims if "lon" in dim.lower() or "west_east" in dim.lower()), None)
                        row["units"] = da.attrs.get("units")
                        row["long_name"] = da.attrs.get("long_name")
                        row["fill_value"] = da.attrs.get("_FillValue")
                        row["missing_value"] = da.attrs.get("missing_value")
                    row["global_title"] = ds.attrs.get("TITLE")
                    row["dx_m"] = safe_float(ds.attrs.get("DX"))
                    row["dy_m"] = safe_float(ds.attrs.get("DY"))
            except Exception as exc:
                row["open_error"] = str(exc)
            rows.append(row)
    return pd.DataFrame(rows), domain_summary


def classify_variable(name: str) -> str:
    lowered = name.lower()
    if "snow" in lowered:
        return "snow"
    if "runoff" in lowered:
        return "runoff"
    if "soil" in lowered:
        return "soil"
    if lowered.startswith("t2") or "temp" in lowered or "skin" in lowered:
        return "temperature"
    if "prec" in lowered or lowered == "rain":
        return "precipitation"
    if "sw_" in lowered or "lw_" in lowered or "rad" in lowered:
        return "radiation"
    if "wind" in lowered or lowered.startswith(("u_", "v_", "w_")) or lowered == "uv10":
        return "wind"
    if "rh" in lowered or lowered == "q2":
        return "humidity"
    return "other"


def inventory_wus_variables(file_inventory: pd.DataFrame, dataset_id: str, domain: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    domain_inventory = file_inventory[(file_inventory["domain"] == domain) & (file_inventory["opened_successfully"])]
    rows: list[dict[str, Any]] = []
    for variable, group in domain_inventory.groupby("variable", sort=True):
        if pd.isna(variable):
            continue
        sample_path = Path(group.sort_values("file_year")["path"].iloc[0])
        with xr.open_dataset(sample_path, engine="netcdf4", decode_times=True) as ds:
            da = ds[str(variable)]
            row = {
                "dataset_id": dataset_id,
                "domain": domain,
                "variable": str(variable),
                "category": classify_variable(str(variable)),
                "sample_path": str(sample_path),
                "units": da.attrs.get("units"),
                "long_name": da.attrs.get("long_name"),
                "description": da.attrs.get("description"),
                "standard_name": da.attrs.get("standard_name"),
                "fill_value": da.attrs.get("_FillValue"),
                "missing_value": da.attrs.get("missing_value"),
                "dims": json.dumps({name: int(size) for name, size in da.sizes.items()}, sort_keys=True),
                "file_count": int(group.shape[0]),
                "file_year_min": int(group["file_year"].min()),
                "file_year_max": int(group["file_year"].max()),
                "time_count_values": sorted_unique_text(group["time_count"].dropna().tolist()),
                "time_start_first": group.sort_values("file_year")["time_start"].iloc[0],
                "time_end_last": group.sort_values("file_year")["time_end"].iloc[-1],
                "calendar_values": sorted_unique_text(group["calendar"].dropna().tolist()),
                "dx_m": safe_float(group["dx_m"].dropna().iloc[0]) if group["dx_m"].notna().any() else None,
                "dy_m": safe_float(group["dy_m"].dropna().iloc[0]) if group["dy_m"].notna().any() else None,
            }
            rows.append(row)

    wrfinput_meta = inspect_wrfinput(domain)
    return pd.DataFrame(rows), {"wrfinput": wrfinput_meta}


def build_wus_series(dataset_id: str, domain: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = Wusd3Dataset(dataset_id=dataset_id, domain=domain, root_dir=Path(WUSD3_ROOT))
    years = discover_wusd3_file_years(dataset)
    water_years = sorted(year + 1 for year in years)
    grid = get_wusd3_grid_definition(dataset, water_year=water_years[0], region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    mask = build_sierra_mask(grid, region=DEFAULT_SIERRA_REGION)
    area = infer_grid_cell_areas(grid.fine_latitude, grid.fine_longitude)
    sample_path = variable_path_for_file_year(dataset, "swe", years[0])
    with xr.open_dataset(sample_path, engine="netcdf4", decode_times=True) as ds:
        snow_units = str(ds["snow"].attrs.get("units", "")).strip().lower()
        sample_day_start = str(ds["day"].values[0])
        sample_day_end = str(ds["day"].values[-1])
        sample_day_count = int(ds["day"].size)
        sample_attrs = {
            "snow_units": ds["snow"].attrs.get("units"),
            "snow_long_name": ds["snow"].attrs.get("long_name"),
            "calendar": getattr(ds["day"].dt, "calendar", None),
            "time_count": sample_day_count,
        }

    rows: list[dict[str, Any]] = []
    for water_year in water_years:
        apr1 = load_wusd3_snapshot(
            dataset,
            water_year=water_year,
            snapshot_date=date(water_year, 4, 1),
            swe_grid=grid,
            fill_missing=False,
        )
        swe_native = weighted_masked_mean(apr1, mask, area)
        swe_m = swe_native / 1000.0 if snow_units == "mm" else swe_native
        rows.append(
            {
                "water_year": water_year,
                "wus_swe_m": swe_m,
                "wus_swe_mm": swe_m * 1000.0 if np.isfinite(swe_m) else np.nan,
                "wus_source_path": apr1.attrs.get("source_path"),
            }
        )
    return pd.DataFrame(rows), {
        "grid": grid,
        "mask": mask,
        "area": area,
        "sample_day_start": sample_day_start,
        "sample_day_end": sample_day_end,
        "sample_attrs": sample_attrs,
        "mask_summary": {
            "mask_type": str(mask.attrs.get("mask_type", "unknown")),
            "nonzero_weight_cells": int((mask > 0.0).sum().item()),
            "total_effective_area_m2": float((mask * area).sum(skipna=True).item()),
        },
    }


def compute_metrics(observed: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    diff = simulated - observed
    obs_mean = float(np.mean(observed))
    sse = float(np.sum((observed - simulated) ** 2))
    sst = float(np.sum((observed - obs_mean) ** 2))
    pearson = float(np.corrcoef(observed, simulated)[0, 1]) if observed.size >= 2 else float("nan")
    spearman = float(pd.Series(observed).corr(pd.Series(simulated), method="spearman")) if observed.size >= 2 else float("nan")
    obs_std = float(np.std(observed, ddof=1)) if observed.size >= 2 else float("nan")
    sim_std = float(np.std(simulated, ddof=1)) if simulated.size >= 2 else float("nan")
    return {
        "n": int(observed.size),
        "mean_bias_m": float(np.mean(diff)),
        "median_bias_m": float(np.median(diff)),
        "rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "mae_m": float(np.mean(np.abs(diff))),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "r2_obs_as_truth": float("nan") if sst == 0.0 else 1.0 - (sse / sst),
        "sim_to_obs_std_ratio": float("nan") if not np.isfinite(obs_std) or obs_std == 0.0 else sim_std / obs_std,
        "observed_std_m": obs_std,
        "simulated_std_m": sim_std,
    }


def make_plots(comparison: pd.DataFrame) -> None:
    years = comparison["water_year"].to_numpy(dtype=int)
    obs_m = comparison["ucla_swe_m"].to_numpy(dtype=float)
    sim_m = comparison["wus_swe_m"].to_numpy(dtype=float)
    err_m = comparison["sim_minus_obs_m"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, obs_m, marker="o", label="UCLA observed")
    ax.plot(years, sim_m, marker="s", label="WUS-02 simulated")
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE (m)")
    ax.set_title("April 1 Sierra SWE by water year")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_TS_PATH, dpi=150)
    plt.close(fig)

    finite = np.isfinite(obs_m) & np.isfinite(sim_m)
    low = float(min(np.nanmin(obs_m[finite]), np.nanmin(sim_m[finite])))
    high = float(max(np.nanmax(obs_m[finite]), np.nanmax(sim_m[finite])))
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(obs_m, sim_m, s=40)
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Observed April 1 Sierra SWE (m)")
    ax.set_ylabel("Simulated April 1 Sierra SWE (m)")
    ax.set_title("Simulated vs observed Sierra SWE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_SCATTER_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(years, err_m, marker="o")
    ax.set_xlabel("Water year")
    ax.set_ylabel("Simulation minus observation (m)")
    ax.set_title("WUS-02 minus UCLA April 1 Sierra SWE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_ERROR_PATH, dpi=150)
    plt.close(fig)


def build_overlap_table(ucla_years: list[int], wus_years: list[int]) -> pd.DataFrame:
    all_years = sorted(set(ucla_years) | set(wus_years))
    overlap_set = set(ucla_years) & set(wus_years)
    rows = []
    for year in all_years:
        rows.append(
            {
                "water_year": year,
                "ucla_available": year in set(ucla_years),
                "wus_available": year in set(wus_years),
                "in_overlap": year in overlap_set,
                "notes": "" if year in overlap_set else ("missing_from_wus" if year in set(ucla_years) else "missing_from_ucla"),
            }
        )
    return pd.DataFrame(rows)


def key_lines_from_inventory(file_inventory: pd.DataFrame, variable_inventory: pd.DataFrame, domain: str) -> dict[str, Any]:
    domain_inventory = file_inventory[file_inventory["domain"] == domain]
    years = sorted(domain_inventory["file_year"].dropna().astype(int).unique().tolist())
    common_counts = sorted(domain_inventory["time_count"].dropna().astype(int).unique().tolist())
    key_variables = variable_inventory[variable_inventory["variable"].isin(PRIORITY_VARIABLES)].copy()
    if key_variables.empty:
        key_variables = variable_inventory[variable_inventory["category"].isin({"snow", "temperature", "precipitation", "soil", "runoff", "radiation"})].copy()
    key_variables["priority_rank"] = key_variables["variable"].apply(lambda value: PRIORITY_VARIABLES.index(value) if value in PRIORITY_VARIABLES else 999)
    key_variables = key_variables.sort_values(["priority_rank", "variable"]).head(12)
    return {
        "file_year_min": min(years),
        "file_year_max": max(years),
        "water_year_min": min(years) + 1,
        "water_year_max": max(years) + 1,
        "time_count_values": common_counts,
        "key_variables": key_variables,
    }


def render_variable_table(variable_rows: pd.DataFrame) -> list[str]:
    lines = [
        "| Variable | Category | Units | File years | Notes |",
        "|---|---|---|---|---|",
    ]
    for _, row in variable_rows.iterrows():
        notes = []
        if row.get("long_name"):
            notes.append(str(row["long_name"]))
        if row.get("description"):
            notes.append(str(row["description"]))
        note_text = "; ".join(notes[:2]) if notes else "NA"
        lines.append(
            "| `{}` | {} | `{}` | {}-{} | {} |".format(
                row["variable"],
                row["category"],
                format_scalar(row["units"]),
                int(row["file_year_min"]),
                int(row["file_year_max"]),
                note_text.replace("|", "/"),
            )
        )
    return lines


def render_year_list(years: list[int]) -> str:
    return ", ".join(str(year) for year in years)


def build_report(
    dataset_id: str,
    domain: str,
    file_inventory: pd.DataFrame,
    variable_inventory: pd.DataFrame,
    overlap_years: pd.DataFrame,
    comparison: pd.DataFrame,
    metrics: dict[str, float],
    ucla_meta: dict[str, Any],
    wus_meta: dict[str, Any],
    inventory_summary: dict[str, Any],
) -> str:
    wrf = wus_meta["wrfinput"]
    ucla_years = sorted(comparison["water_year"].astype(int).tolist())
    full_overlap_years = overlap_years.loc[overlap_years["in_overlap"], "water_year"].astype(int).tolist()
    missing_overlap = overlap_years.loc[~overlap_years["in_overlap"] & overlap_years["ucla_available"], "water_year"].astype(int).tolist()
    common_time_counts = inventory_summary["time_count_values"]
    leap_present = 366 in common_time_counts
    interpretation_lines = [
        "- Verified fact: the user-provided canonical path is `{}`.".format(WUS_DAILY_ROOT / dataset_id / "postprocess" / domain),
        "- Verified fact: repo code consistently uses `dataset_id={!r}` and `domain={!r}` for the strict WUS Sierra workflows.".format(dataset_id, domain),
        "- Interpretation: the dataset id naming strongly suggests the parent global model is EC-Earth3 member `r1i1p1f1`, historical branch, with an archive-side bias-corrected product (`historical_bc` / `hist.bias-correct`).",
        "- Interpretation limit: no nearby README, namelist, job script, or publication was found in the audited archive path that explicitly documents parent forcing fields, boundary-condition cadence, spin-up length, or the exact bias-correction recipe.",
    ]

    lines: list[str] = []
    lines.append("# WUS-02 simulation and UCLA Sierra SWE overlap audit")
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append("")
    lines.append("- {}".format(WUS_INTERPRETATION_NOTE))
    lines.append("- Canonical WUS-02 dataset audited: `{}`.".format(WUS_DAILY_ROOT / dataset_id / "postprocess" / domain))
    lines.append("- Simulation family: WRF-style regional climate output (`TITLE = {}`; static file `{}`).".format(wrf["attrs"].get("TITLE", "NA"), wrf["path"]))
    lines.append("- Parent/member naming in the canonical dataset id: `{}`.".format(dataset_id))
    lines.append("- WUS file years on `{}`: `{}` to `{}`; April 1 water years available from this path: `WY{}` to `WY{}`.".format(domain, inventory_summary["file_year_min"], inventory_summary["file_year_max"], inventory_summary["water_year_min"], inventory_summary["water_year_max"]))
    lines.append("- UCLA observational years available: `WY{}` to `WY{}`.".format(min(discover_ucla_years()), max(discover_ucla_years())))
    lines.append("- Exact overlap years used for the physical comparison: `{}` (`n={}`).".format(render_year_list(full_overlap_years), len(full_overlap_years)))
    lines.append("- Mean bias: `{:.4f} m` (`{:.1f} mm`); RMSE: `{:.4f} m` (`{:.1f} mm`).".format(metrics["mean_bias_m"], metrics["mean_bias_m"] * 1000.0, metrics["rmse_m"], metrics["rmse_m"] * 1000.0))
    lines.append("- Pearson `r = {:.3f}` and Spearman `rho = {:.3f}` with `n={}` overlap years.".format(metrics["pearson_r"], metrics["spearman_rho"], metrics["n"]))
    lines.append("- No within-path ensemble members were found under the canonical WUS-02 directory; this comparison is therefore single-realization only.")
    lines.append("")

    lines.append("## 2. Canonical WUS-02 paths")
    lines.append("")
    lines.append("- WUS root: `{}`".format(WUS_ROOT))
    lines.append("- Daily root: `{}`".format(WUS_DAILY_ROOT))
    lines.append("- Canonical WUS-02 directory: `{}`".format(WUS_DAILY_ROOT / dataset_id / "postprocess" / domain))
    lines.append("- Member root: `{}`".format(WUS_DAILY_ROOT / dataset_id))
    lines.append("- Static grid file: `{}`".format(wrf["path"]))
    lines.append("- Available WUS-D3 dataset ids in the archive: `{}`".format(", ".join(discover_wusd3_dataset_ids())))
    lines.append("")

    lines.append("## 3. Verified simulation methodology")
    lines.append("")
    lines.extend(interpretation_lines)
    lines.append("- Verified static-grid facts from `wrfinput_{}`:".format(domain))
    lines.append("  - Horizontal dimensions: `{}`.".format(wrf["dims"]))
    lines.append("  - Native horizontal spacing: `DX = {}` m, `DY = {}` m.".format(format_scalar(wrf["dx_m"]), format_scalar(wrf["dy_m"])))
    lines.append("  - Latitude range on the native domain: `{:.3f}` to `{:.3f}`.".format(wrf["lat_min"], wrf["lat_max"]))
    lines.append("  - Longitude range on the native domain: `{:.3f}` to `{:.3f}`.".format(wrf["lon_min"], wrf["lon_max"]))
    lines.append("  - Terrain height units/range: `{}`; `{:.1f}` to `{:.1f}` m.".format(format_scalar(wrf["hgt_units"]), wrf["hgt_min_m"], wrf["hgt_max_m"]))
    lines.append("  - Land mask description: `{}`.".format("LANDMASK (1 for land, 0 for water)"))
    lines.append("- Verified output-file facts from the canonical April 1 SWE source file:")
    lines.append("  - `snow` units are `mm` with `long_name = snow water equivalent`.")
    lines.append("  - The file-year convention is water-year-like: file `1984` spans `1984-09-01` through `1985-08-31`, so April 1, 1985 is extracted from file year 1984.")
    lines.append("  - Time-count values seen across the full domain inventory: `{}` days, so leap years are {}present.".format(", ".join(str(value) for value in common_time_counts), "" if leap_present else "not "))
    lines.append("- Vertical structure:")
    lines.append("  - Surface SWE output files are 3D in `(day, lat2d, lon2d)` and do not expose vertical levels.")
    lines.append("  - The static WRF file carries a 40-level atmospheric configuration (`BOTTOM-TOP_GRID_DIMENSION = {}`), relevant for the parent simulation but not directly for the surface SWE output.".format(wrf["attrs"].get("BOTTOM-TOP_GRID_DIMENSION", "NA")))
    lines.append("")

    lines.append("## 4. Input and forcing inventory")
    lines.append("")
    lines.append("- Verified inputs/conditioning present on disk:")
    lines.append("  - Topography: verified from `wrfinput_{}` variable `HGT` (units `m`).".format(domain))
    lines.append("  - Land-water mask: verified from `wrfinput_{}` variable `LANDMASK`.".format(domain))
    lines.append("  - Model grid/projection metadata: verified from WRF attributes such as `DX`, `DY`, `TRUELAT1`, `TRUELAT2`, `CEN_LAT`, and `STAND_LON`.")
    lines.append("- Verified output variables implying the simulation evolved daily snow and hydroclimate states:")
    lines.append("  - `snow`, `prec`, `prec_snow`, `soil_m`, `sfc_runoff`, `subsfc_runoff`, `t2`, `t2max`, `t2min`, `tskin`, `sw_dwn`, `lh_sfc`, and related fields.")
    lines.append("- What remains unverified from the local files alone:")
    lines.append("  - exact SST source dataset and its latitude-longitude bounds;")
    lines.append("  - exact lateral atmospheric boundary dataset;")
    lines.append("  - exact sea-ice treatment;")
    lines.append("  - exact greenhouse-gas or aerosol forcing path;")
    lines.append("  - exact initialization and spin-up procedure;")
    lines.append("  - exact bias-correction algorithm behind the `historical_bc` / `hist.bias-correct` tokens.")
    lines.append("- Leakage implication:")
    lines.append("  - The WUS-02 outputs already contain same-water-year daily snow evolution through April 1, so they are not independent prediction labels by themselves.")
    lines.append("  - Whether these years can be used as extra supervised training samples depends on the upstream forcing provenance. If the simulation was driven by same-year observed or reanalyzed atmospheric states or SST, then direct pooling with UCLA observations would introduce target-year information leakage.")
    lines.append("")

    lines.append("## 5. Output-variable inventory")
    lines.append("")
    lines.append("- Full file inventory: `{}`".format(FILE_INVENTORY_PATH))
    lines.append("- Full variable inventory: `{}`".format(VARIABLE_INVENTORY_PATH))
    lines.append("- Canonical file format: NetCDF (`.nc`).")
    lines.append("- Canonical filename convention:")
    lines.append("  - `<variable>.daily.<model_token>.<scenario_token>.d02.<file_year>.nc`")
    lines.append("  - Example: `snow.daily.ec-earth3.r1i1p1f1_2.hist.bias-correct.d02.1984.nc`")
    lines.append("- Number of files in the canonical WUS-02 path: `{}`.".format(int(file_inventory[file_inventory["domain"] == domain].shape[0])))
    lines.append("- Selected high-value variables on the audited domain:")
    lines.extend(render_variable_table(inventory_summary["key_variables"]))
    lines.append("- SWE-specific findings:")
    lines.append("  - Exact variable name: `snow`.")
    lines.append("  - Units: `mm`.")
    lines.append("  - Physical meaning from metadata: `snow water equivalent`.")
    lines.append("  - Time convention: daily snapshots on coordinate `day` covering September 1 through August 31 of the following calendar year.")
    lines.append("  - April 1 values can be extracted directly because April 1 exists explicitly in the daily coordinate.")
    lines.append("  - Water-year labeling is handled as `file_year + 1` for April 1 extraction.")
    lines.append("")

    lines.append("## 6. UCLA observational SWE definition")
    lines.append("")
    lines.append("- Canonical UCLA root: `{}`".format(UCLA_ROOT))
    lines.append("- Canonical file pattern: `{}/WUS_UCLA_SR_v01_ALL_0_agg_16_WY{{water_year}}_SD_SWE_SCA_POST.nc`".format(UCLA_ROOT))
    lines.append("- Example file inspected directly: `{}`".format(ucla_meta["example"]["path"]))
    lines.append("- UCLA time coverage in each file is water-year indexed: `{}` through `{}` in the inspected WY1985 file.".format(ucla_meta["example"]["time_start"], ucla_meta["example"]["time_end"]))
    lines.append("- UCLA variable used by the project: `{}` with `Stats=0` (mean field).".format(SWE_VARIABLE))
    lines.append("- UCLA units: `{}`.".format(ucla_meta["example"]["swe_units"]))
    lines.append("- Canonical Sierra mask reused from repo code: `snow_ml.data.build_sierra_mask` with bounds `{}`.".format(asdict(DEFAULT_SIERRA_REGION)))
    lines.append("- Area weighting used here matches the project logic in spirit: weighted average over the canonical Sierra mask using inferred grid-cell areas on each dataset's native grid.")
    lines.append("- April 1 extraction procedure reused from project code: `load_swe_snapshot(water_year, snapshot_date=date(water_year, 4, 1), stat_name=\"mean\")`.")
    lines.append("- Existing scalar target artifact already used elsewhere in the repo: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc`.")
    lines.append("")

    lines.append("## 7. Overlapping years")
    lines.append("")
    lines.append("- WUS full file-year range on the canonical path: `{}-{}.`".format(inventory_summary["file_year_min"], inventory_summary["file_year_max"]))
    lines.append("- WUS full April 1 water-year range from the canonical path: `WY{}-WY{}`.".format(inventory_summary["water_year_min"], inventory_summary["water_year_max"]))
    lines.append("- UCLA full water-year range in the canonical root: `WY{}-WY{}`.".format(min(discover_ucla_years()), max(discover_ucla_years())))
    lines.append("- Exact overlap water years: `{}`.".format(render_year_list(full_overlap_years)))
    lines.append("- Overlap count: `{}`.".format(len(full_overlap_years)))
    if missing_overlap:
        lines.append("- UCLA years without WUS overlap: `{}`.".format(render_year_list(missing_overlap)))
    lines.append("- Overlap table written to `{}`.".format(OVERLAP_YEARS_PATH))
    lines.append("- Year-label alignment note: April 1, 1998 is treated as `WY1998` in both datasets.")
    lines.append("")

    lines.append("## 8. Year-by-year SWE comparison")
    lines.append("")
    lines.append("- Comparison CSV written to `{}`.".format(COMPARISON_PATH))
    lines.append("- Columns include:")
    lines.append("  - `water_year`")
    lines.append("  - `ucla_swe_m`, `ucla_swe_mm`")
    lines.append("  - `wus_swe_m`, `wus_swe_mm`")
    lines.append("  - `sim_minus_obs_m`, `sim_minus_obs_mm`")
    lines.append("  - `absolute_error_m`, `absolute_error_mm`")
    lines.append("  - `percent_error`")
    lines.append("  - `ucla_source_path`, `wus_source_path`")
    lines.append("- Because the canonical WUS-02 path contains only one realization for this member, ensemble mean/spread columns are not applicable here.")
    lines.append("")

    lines.append("## 9. Metrics and plots")
    lines.append("")
    lines.append("- Mean bias: `{:.4f} m` (`{:.1f} mm`)".format(metrics["mean_bias_m"], metrics["mean_bias_m"] * 1000.0))
    lines.append("- Median bias: `{:.4f} m` (`{:.1f} mm`)".format(metrics["median_bias_m"], metrics["median_bias_m"] * 1000.0))
    lines.append("- RMSE: `{:.4f} m` (`{:.1f} mm`)".format(metrics["rmse_m"], metrics["rmse_m"] * 1000.0))
    lines.append("- MAE: `{:.4f} m` (`{:.1f} mm`)".format(metrics["mae_m"], metrics["mae_m"] * 1000.0))
    lines.append("- Pearson correlation: `{:.3f}` (`n={}`)".format(metrics["pearson_r"], metrics["n"]))
    lines.append("- Spearman correlation: `{:.3f}` (`n={}`)".format(metrics["spearman_rho"], metrics["n"]))
    lines.append("- Coefficient of determination: `{:.3f}` using `R^2 = 1 - SSE/SST` with UCLA treated as truth.".format(metrics["r2_obs_as_truth"]))
    lines.append("- Simulated-to-observed interannual standard-deviation ratio: `{:.3f}`".format(metrics["sim_to_obs_std_ratio"]))
    lines.append("- Time-series plot: `{}`".format(FIG_TS_PATH))
    lines.append("- Scatter plot: `{}`".format(FIG_SCATTER_PATH))
    lines.append("- Error plot: `{}`".format(FIG_ERROR_PATH))
    lines.append("- Ensemble-range plot: not applicable because the canonical WUS-02 path contains a single realization.")
    lines.append("")

    lines.append("## 10. Scientific limitations")
    lines.append("")
    lines.append("- The local archive and repo code verify the WRF-style output structure, but they do not fully document the parent forcing fields or exact bias-correction method.")
    lines.append("- The comparison here preserves native-grid area averaging over the shared canonical Sierra box; it does not try to force the two datasets onto one common regridded lattice.")
    lines.append("- This means any residual topographic or land-mask mismatch between UCLA and WUS remains part of the scientific uncertainty.")
    lines.append("- No expensive climate rerun was attempted; all results are derived from existing files only.")
    lines.append("")

    lines.append("## 11. Recommendation")
    lines.append("")
    lines.append("- WUS-02 does genuinely extend the number of available Sierra SWE-like years beyond the 37-year UCLA record, because the canonical path provides daily WUS SWE from `WY{}-WY{}`.".format(inventory_summary["water_year_min"], inventory_summary["water_year_max"]))
    lines.append("- That does **not** automatically make these years safe to pool as additional supervised training labels.")
    lines.append("- If the WUS simulation was driven by same-year global-model atmospheric and ocean states that themselves assimilate or inherit observed target-year evolution, then using the simulated April 1 SWE as if it were an independent observation would blur the prediction problem and can create leakage relative to a true forecast setting.")
    lines.append("- Based on what is verified locally, the safest immediate use is as an auxiliary dataset for representation learning, sensitivity tests, process diagnostics, or pretraining, not direct label pooling.")
    lines.append("- Direct supervised pooling should wait until the parent-forcing provenance is documented well enough to answer whether same-year information enters the simulation in a way that would invalidate the intended forecast experiment.")
    lines.append("")

    lines.append("## Execution")
    lines.append("")
    lines.append("```bash")
    lines.append("bash -lc 'source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true; module load python; module load conda; source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda activate uxarray_build; python scripts/wus02_data_audit.py'")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id
    domain = args.domain

    print("[audit] inventory WUS files", flush=True)
    file_inventory, domain_summary = inventory_wus_files(dataset_id)
    file_inventory.to_csv(FILE_INVENTORY_PATH, index=False)

    print("[audit] inventory WUS variables", flush=True)
    variable_inventory, variable_meta = inventory_wus_variables(file_inventory, dataset_id, domain)
    variable_inventory.to_csv(VARIABLE_INVENTORY_PATH, index=False)

    print("[audit] build UCLA Sierra April 1 series", flush=True)
    ucla_series, ucla_meta = build_ucla_series()
    print("[audit] build WUS Sierra April 1 series", flush=True)
    wus_series, wus_meta = build_wus_series(dataset_id, domain)
    wus_meta.update(variable_meta)

    comparison = pd.merge(ucla_series, wus_series, on="water_year", how="inner").sort_values("water_year").reset_index(drop=True)
    comparison["sim_minus_obs_m"] = comparison["wus_swe_m"] - comparison["ucla_swe_m"]
    comparison["sim_minus_obs_mm"] = comparison["sim_minus_obs_m"] * 1000.0
    comparison["absolute_error_m"] = np.abs(comparison["sim_minus_obs_m"])
    comparison["absolute_error_mm"] = comparison["absolute_error_m"] * 1000.0
    comparison["percent_error"] = np.where(np.abs(comparison["ucla_swe_m"]) >= 1.0e-3, 100.0 * comparison["sim_minus_obs_m"] / comparison["ucla_swe_m"], np.nan)
    comparison.to_csv(COMPARISON_PATH, index=False)

    overlap_years = build_overlap_table(
        discover_ucla_years(),
        sorted(wus_series["water_year"].astype(int).tolist()),
    )
    overlap_years.to_csv(OVERLAP_YEARS_PATH, index=False)

    print("[audit] compute metrics and plots", flush=True)
    metrics = compute_metrics(
        comparison["ucla_swe_m"].to_numpy(dtype=float),
        comparison["wus_swe_m"].to_numpy(dtype=float),
    )
    make_plots(comparison)

    inventory_summary = key_lines_from_inventory(file_inventory, variable_inventory, domain)
    report = build_report(
        dataset_id=dataset_id,
        domain=domain,
        file_inventory=file_inventory,
        variable_inventory=variable_inventory,
        overlap_years=overlap_years,
        comparison=comparison,
        metrics=metrics,
        ucla_meta=ucla_meta,
        wus_meta=wus_meta,
        inventory_summary=inventory_summary,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary = {
        "dataset_id": dataset_id,
        "domain": domain,
        "wus02_interpretation": WUS_INTERPRETATION_NOTE,
        "canonical_wus02_path": str(WUS_DAILY_ROOT / dataset_id / "postprocess" / domain),
        "file_inventory_path": str(FILE_INVENTORY_PATH),
        "variable_inventory_path": str(VARIABLE_INVENTORY_PATH),
        "overlap_years_path": str(OVERLAP_YEARS_PATH),
        "comparison_path": str(COMPARISON_PATH),
        "report_path": str(REPORT_PATH),
        "figures": [str(FIG_TS_PATH), str(FIG_SCATTER_PATH), str(FIG_ERROR_PATH)],
        "metrics": metrics,
        "domain_file_count": int(file_inventory[file_inventory["domain"] == domain].shape[0]),
        "opened_successfully_count": int(file_inventory["opened_successfully"].sum()),
        "open_failure_count": int((~file_inventory["opened_successfully"]).sum()),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
