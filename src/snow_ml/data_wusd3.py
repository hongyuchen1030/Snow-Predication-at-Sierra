import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from pyproj import Proj

from snow_ml.data import (
    COARSEN_TRIMMING_POLICY,
    DEFAULT_COARSEN_FACTOR,
    DEFAULT_MODEL_REGION,
    RegionBounds,
)


WUSD3_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3")
DEFAULT_WUSD3_DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
DEFAULT_WUSD3_DOMAIN = "d02"
WUSD3_SWE_VARIABLE = "snow"
WUSD3_FILE_PREFIX = {
    "swe": "snow",
}
WUSD3_FILE_MIDDLE = {
    "historical_bc": "hist.bias-correct",
    "ssp370_bc": "ssp370.bias-correct",
}
WUSD3_PROJECTION_RADIUS_METERS = 6370000.0


@dataclass(frozen=True)
class Wusd3Dataset:
    dataset_id: str
    domain: str
    root_dir: Path


@dataclass(frozen=True)
class Wusd3GridDefinition:
    dataset_id: str
    domain: str
    file_path: str
    latitude_name: str
    longitude_name: str
    time_name: str
    grid_shape: Tuple[int, int]
    cropped_shape: Tuple[int, int]
    trimmed_shape: Tuple[int, int]
    coarsen_factor: int
    requested_region: RegionBounds
    effective_region: RegionBounds
    latitude: xr.DataArray
    longitude: xr.DataArray
    fine_latitude: xr.DataArray
    fine_longitude: xr.DataArray
    region_mask: xr.DataArray
    row_slice: slice
    col_slice: slice


def default_wusd3_dataset() -> Wusd3Dataset:
    return Wusd3Dataset(
        dataset_id=DEFAULT_WUSD3_DATASET_ID,
        domain=DEFAULT_WUSD3_DOMAIN,
        root_dir=WUSD3_ROOT,
    )


def wusd3_dataset_dir(dataset: Wusd3Dataset) -> Path:
    path = dataset.root_dir / "daily" / dataset.dataset_id / "postprocess" / dataset.domain
    if not path.exists():
        raise FileNotFoundError("WUS-D3 dataset directory not found: %s" % path)
    return path


def discover_wusd3_dataset_ids(root_dir: Path = WUSD3_ROOT) -> List[str]:
    daily_root = root_dir / "daily"
    if not daily_root.exists():
        return []
    return sorted(path.name for path in daily_root.iterdir() if path.is_dir())


def discover_wusd3_file_years(
    dataset: Wusd3Dataset,
    variable_key: str = "swe",
) -> List[int]:
    variable_prefix = WUSD3_FILE_PREFIX[variable_key]
    pattern = re.compile(r"^%s\.daily\..*\.d\d\d\.(\d{4})\.nc$" % re.escape(variable_prefix))
    years: List[int] = []
    for path in sorted(wusd3_dataset_dir(dataset).glob("%s.daily.*.nc" % variable_prefix)):
        match = pattern.match(path.name)
        if match:
            years.append(int(match.group(1)))
    return years


def discover_wusd3_water_years(dataset: Wusd3Dataset) -> List[int]:
    return [year + 1 for year in discover_wusd3_file_years(dataset, variable_key="swe")]


def inspect_wusd3_file(path: Path, engine: str = "netcdf4") -> Dict[str, object]:
    with xr.open_dataset(path, engine=engine, decode_times=True) as ds:
        return {
            "path": str(path),
            "variables": list(ds.data_vars),
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "coordinates": list(ds.coords),
            "global_attributes": dict(ds.attrs),
        }


def file_year_for_date(snapshot_date: date) -> int:
    if snapshot_date.month >= 9:
        return snapshot_date.year
    return snapshot_date.year - 1


def water_year_for_date(snapshot_date: date) -> int:
    if snapshot_date.month >= 10:
        return snapshot_date.year + 1
    return snapshot_date.year


def variable_path_for_file_year(
    dataset: Wusd3Dataset,
    variable_key: str,
    file_year: int,
) -> Path:
    scenario_token = _scenario_token(dataset.dataset_id)
    model_token = _model_token(dataset.dataset_id, scenario_token)
    variable_prefix = WUSD3_FILE_PREFIX[variable_key]
    middle = WUSD3_FILE_MIDDLE[scenario_token]
    path = (
        wusd3_dataset_dir(dataset)
        / ("%s.daily.%s.%s.%s.%04d.nc" % (variable_prefix, model_token, middle, dataset.domain, file_year))
    )
    if not path.exists():
        raise FileNotFoundError(
            "WUS-D3 file not found for dataset=%s variable=%s file_year=%s: %s"
            % (dataset.dataset_id, variable_key, file_year, path)
        )
    return path


def get_wusd3_grid_definition(
    dataset: Wusd3Dataset,
    *,
    water_year: int,
    region: RegionBounds = DEFAULT_MODEL_REGION,
    coarsen_factor: int = DEFAULT_COARSEN_FACTOR,
) -> Wusd3GridDefinition:
    file_year = water_year - 1
    path = variable_path_for_file_year(dataset, "swe", file_year)
    latitude_2d, longitude_2d = _reconstruct_wusd3_coordinates(path)
    row_slice, col_slice = _crop_wusd3_indices(latitude_2d, longitude_2d, region)
    cropped_latitude = latitude_2d.isel(lat2d=row_slice, lon2d=col_slice)
    cropped_longitude = longitude_2d.isel(lat2d=row_slice, lon2d=col_slice)
    cropped_region_mask = (
        (cropped_latitude >= region.lat_min)
        & (cropped_latitude <= region.lat_max)
        & (cropped_longitude >= region.lon_min)
        & (cropped_longitude <= region.lon_max)
    )
    trimmed_latitude, trimmed_longitude = _trim_wusd3_coordinates(
        cropped_latitude,
        cropped_longitude,
        coarsen_factor,
    )
    trimmed_region_mask = cropped_region_mask.isel(
        lat2d=slice(0, int(trimmed_latitude.shape[0])),
        lon2d=slice(0, int(trimmed_latitude.shape[1])),
    )
    masked_trimmed_latitude = trimmed_latitude.where(trimmed_region_mask)
    masked_trimmed_longitude = trimmed_longitude.where(trimmed_region_mask)
    latitude, longitude = _coarsen_wusd3_coordinates(
        masked_trimmed_latitude,
        masked_trimmed_longitude,
        coarsen_factor,
    )
    effective_region = RegionBounds(
        lat_min=float(np.nanmin(masked_trimmed_latitude.values)),
        lat_max=float(np.nanmax(masked_trimmed_latitude.values)),
        lon_min=float(np.nanmin(masked_trimmed_longitude.values)),
        lon_max=float(np.nanmax(masked_trimmed_longitude.values)),
    )
    return Wusd3GridDefinition(
        dataset_id=dataset.dataset_id,
        domain=dataset.domain,
        file_path=str(path),
        latitude_name="lat2d",
        longitude_name="lon2d",
        time_name="day",
        grid_shape=(int(latitude.shape[0]), int(latitude.shape[1])),
        cropped_shape=(int(cropped_latitude.shape[0]), int(cropped_latitude.shape[1])),
        trimmed_shape=(int(trimmed_latitude.shape[0]), int(trimmed_latitude.shape[1])),
        coarsen_factor=int(coarsen_factor),
        requested_region=region,
        effective_region=effective_region,
        latitude=latitude,
        longitude=longitude,
        fine_latitude=masked_trimmed_latitude,
        fine_longitude=masked_trimmed_longitude,
        region_mask=trimmed_region_mask.astype(bool),
        row_slice=row_slice,
        col_slice=col_slice,
    )


def load_wusd3_snapshot(
    dataset: Wusd3Dataset,
    *,
    water_year: int,
    snapshot_date: date,
    swe_grid: Wusd3GridDefinition,
    fill_missing: bool = False,
) -> xr.DataArray:
    file_year = file_year_for_date(snapshot_date)
    path = variable_path_for_file_year(dataset, "swe", file_year)
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        swe = ds[WUSD3_SWE_VARIABLE].sel(day=np.datetime64(snapshot_date.isoformat()))
        swe = swe.astype("float32")
        swe = swe.where(np.isfinite(swe))
        swe = swe.isel(lat2d=swe_grid.row_slice, lon2d=swe_grid.col_slice)
        swe = swe.isel(
            lat2d=slice(0, swe_grid.trimmed_shape[0]),
            lon2d=slice(0, swe_grid.trimmed_shape[1]),
        )
        swe = swe.where(swe_grid.region_mask)
        swe = _coarsen_wusd3_field(swe, swe_grid, reduction="mean").load()
    swe.name = "wusd3_swe_%s" % snapshot_date.isoformat()
    swe.attrs["source_path"] = str(path)
    swe.attrs["water_year"] = int(water_year)
    if fill_missing:
        return swe.fillna(0.0)
    return swe


def load_wusd3_target_swe_map(
    dataset: Wusd3Dataset,
    *,
    water_year: int,
    swe_grid: Wusd3GridDefinition,
    fill_missing: bool = False,
) -> xr.DataArray:
    return load_wusd3_snapshot(
        dataset,
        water_year=water_year,
        snapshot_date=date(water_year, 4, 1),
        swe_grid=swe_grid,
        fill_missing=fill_missing,
    )


def _reconstruct_wusd3_coordinates(path: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        attrs = dict(ds.attrs)
        sample = ds[WUSD3_SWE_VARIABLE]
        south_north = int(sample.sizes["lat2d"])
        west_east = int(sample.sizes["lon2d"])
    proj = Proj(
        proj="lcc",
        lat_1=float(attrs["TRUELAT1"]),
        lat_2=float(attrs["TRUELAT2"]),
        lat_0=float(attrs["CEN_LAT"]),
        lon_0=float(attrs["STAND_LON"]),
        a=WUSD3_PROJECTION_RADIUS_METERS,
        b=WUSD3_PROJECTION_RADIUS_METERS,
    )
    x_center, y_center = proj(float(attrs["CEN_LON"]), float(attrs["CEN_LAT"]))
    dx = float(attrs["DX"])
    dy = float(attrs["DY"])
    x_values = (np.arange(west_east, dtype=np.float64) - (west_east - 1) / 2.0) * dx + x_center
    y_values = (np.arange(south_north, dtype=np.float64) - (south_north - 1) / 2.0) * dy + y_center
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    longitude, latitude = proj(x_grid, y_grid, inverse=True)
    latitude_array = xr.DataArray(
        latitude.astype(np.float32),
        dims=("lat2d", "lon2d"),
        name="wusd3_latitude",
    )
    longitude_array = xr.DataArray(
        longitude.astype(np.float32),
        dims=("lat2d", "lon2d"),
        name="wusd3_longitude",
    )
    return latitude_array, longitude_array


def _crop_wusd3_indices(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    region: RegionBounds,
) -> Tuple[slice, slice]:
    mask = (
        (latitude.values >= region.lat_min)
        & (latitude.values <= region.lat_max)
        & (longitude.values >= region.lon_min)
        & (longitude.values <= region.lon_max)
    )
    if not np.any(mask):
        raise ValueError(
            "Requested region %s does not overlap reconstructed WUS-D3 coordinates." % region.as_dict()
        )
    row_hits = np.where(mask.any(axis=1))[0]
    col_hits = np.where(mask.any(axis=0))[0]
    return (
        slice(int(row_hits[0]), int(row_hits[-1]) + 1),
        slice(int(col_hits[0]), int(col_hits[-1]) + 1),
    )


def _trim_wusd3_coordinates(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    coarsen_factor: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    trimmed_lat_size = (int(latitude.shape[0]) // coarsen_factor) * coarsen_factor
    trimmed_lon_size = (int(latitude.shape[1]) // coarsen_factor) * coarsen_factor
    if trimmed_lat_size == 0 or trimmed_lon_size == 0:
        raise ValueError(
            "Coarsen factor %s is too large for cropped WUS-D3 grid %s."
            % (coarsen_factor, (int(latitude.shape[0]), int(latitude.shape[1])))
        )
    return (
        latitude.isel(lat2d=slice(0, trimmed_lat_size), lon2d=slice(0, trimmed_lon_size)),
        longitude.isel(lat2d=slice(0, trimmed_lat_size), lon2d=slice(0, trimmed_lon_size)),
    )


def _coarsen_wusd3_coordinates(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    coarsen_factor: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    return (
        latitude.coarsen(lat2d=coarsen_factor, lon2d=coarsen_factor, boundary=COARSEN_TRIMMING_POLICY).mean().astype(np.float32),
        longitude.coarsen(lat2d=coarsen_factor, lon2d=coarsen_factor, boundary=COARSEN_TRIMMING_POLICY).mean().astype(np.float32),
    )


def _coarsen_wusd3_field(
    values: xr.DataArray,
    swe_grid: Wusd3GridDefinition,
    *,
    reduction: str,
) -> xr.DataArray:
    coarsened = values.coarsen(
        lat2d=swe_grid.coarsen_factor,
        lon2d=swe_grid.coarsen_factor,
        boundary=COARSEN_TRIMMING_POLICY,
    )
    if reduction == "mean":
        return coarsened.mean()
    if reduction == "sum":
        return coarsened.sum()
    if reduction == "std":
        return coarsened.std()
    raise ValueError("Unsupported coarsen reduction: %s" % reduction)


def _scenario_token(dataset_id: str) -> str:
    for suffix in WUSD3_FILE_MIDDLE:
        if dataset_id.endswith(suffix):
            return suffix
    raise ValueError(
        "Unsupported WUS-D3 dataset_id %r. Expected one ending with one of %s."
        % (dataset_id, sorted(WUSD3_FILE_MIDDLE))
    )


def _model_token(dataset_id: str, scenario_token: str) -> str:
    suffix = "_%s" % scenario_token
    if not dataset_id.endswith(suffix):
        raise ValueError("dataset_id %r does not end with expected suffix %r" % (dataset_id, suffix))
    stem = dataset_id[: -len(suffix)]
    if "_" not in stem:
        return stem
    model_name, remainder = stem.split("_", 1)
    return "%s.%s" % (model_name, remainder)
