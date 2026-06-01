from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

from config.paths import (
    COBE2_SST_LTM_FILE,
    COBE2_SST_MONTHLY_MEAN_FILE,
    ERA5_LAND_GEOPOTENTIAL_FILE,
    ERA5_LAND_ROOT,
    SST_ROOT,
    SWE_ROOT,
)

SWE_ROOT_PATH = Path(SWE_ROOT)
SST_ROOT_PATH = Path(SST_ROOT)
SST_MONTHLY_MEAN_PATH = Path(COBE2_SST_MONTHLY_MEAN_FILE)
SST_LTM_PATH = Path(COBE2_SST_LTM_FILE)
ERA5_LAND_ROOT_PATH = Path(ERA5_LAND_ROOT)
ERA5_LAND_GEOPOTENTIAL_PATH = Path(ERA5_LAND_GEOPOTENTIAL_FILE)

SWE_VARIABLE = "SWE_Post"
SWE_MISSING_VALUE = -999.0
GRAVITY_ACCELERATION = 9.81
DEFAULT_TARGET_MONTH_DAYS = (
    "01-01",
    "01-15",
    "02-01",
    "02-15",
    "03-01",
    "03-15",
    "04-01",
)
HISTORY_DAYS = 730
DEFAULT_HISTORY_YEARS = 2
ERA5_DAILY_REDUCTIONS = {
    "t2m": "mean",
    "tp": "sum",
}
SWE_STAT_INDEX = {
    "mean": 0,
    "std": 1,
    "median": 2,
    "p25": 3,
    "p75": 4,
}
LONGITUDE_NORMALIZATION_NOTE = "Normalize source longitudes from 0..360 to -180..180 when needed."
ERA5_EARLY_SUBSET_LAT_BUFFER_DEGREES = 1.0
ERA5_EARLY_SUBSET_LON_BUFFER_DEGREES = 1.0


@dataclass(frozen=True)
class RegionBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lat_min": float(self.lat_min),
            "lat_max": float(self.lat_max),
            "lon_min": float(self.lon_min),
            "lon_max": float(self.lon_max),
        }


DEFAULT_MODEL_REGION = RegionBounds(
    lat_min=32.5,
    lat_max=43.0,
    lon_min=-134.5,
    lon_max=-114.0,
)
DEFAULT_SIERRA_REGION = RegionBounds(
    lat_min=35.0,
    lat_max=42.0,
    lon_min=-122.5,
    lon_max=-118.0,
)
DEFAULT_COARSEN_FACTOR = 8
COARSEN_TRIMMING_POLICY = "trim"
MASK_TYPE = "fractional"


def _profile_print(message: str) -> None:
    print(message, flush=True)


def _timed_print(label: str, start: float) -> float:
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s", flush=True)
    return elapsed


@dataclass(frozen=True)
class ForecastConfig:
    water_year: int = 2014
    target_month_day: str = "04-01"
    sst_months: int = 12
    history_years: int = DEFAULT_HISTORY_YEARS
    region: RegionBounds = field(default_factory=lambda: DEFAULT_MODEL_REGION)
    coarsen_factor: int = DEFAULT_COARSEN_FACTOR

    def __post_init__(self) -> None:
        if int(self.history_years) < 1:
            raise ValueError(f"history_years must be >= 1, got {self.history_years}")

    @property
    def atmospheric_days(self) -> int:
        return atmospheric_history_days(self.history_years)


@dataclass(frozen=True)
class SweGridDefinition:
    water_year: int
    file_path: str
    latitude_name: str
    longitude_name: str
    time_name: str
    has_stats_dimension: bool
    latitude_is_2d: bool
    longitude_is_2d: bool
    grid_shape: tuple[int, ...]
    latitude: xr.DataArray
    longitude: xr.DataArray
    cropped_shape: tuple[int, ...]
    trimmed_shape: tuple[int, ...]
    coarsen_factor: int
    requested_region: RegionBounds
    effective_region: RegionBounds
    fine_latitude: xr.DataArray
    fine_longitude: xr.DataArray


def default_forecast_config() -> ForecastConfig:
    return ForecastConfig()


def atmospheric_history_days(history_years: int) -> int:
    if int(history_years) < 1:
        raise ValueError(f"history_years must be >= 1, got {history_years}")
    return int(history_years) * 365


def add_region_args(
    parser: argparse.ArgumentParser,
    *,
    defaults: RegionBounds | None = DEFAULT_MODEL_REGION,
) -> None:
    parser.add_argument("--lat-min", type=float, default=None if defaults is None else defaults.lat_min)
    parser.add_argument("--lat-max", type=float, default=None if defaults is None else defaults.lat_max)
    parser.add_argument("--lon-min", type=float, default=None if defaults is None else defaults.lon_min)
    parser.add_argument("--lon-max", type=float, default=None if defaults is None else defaults.lon_max)
    parser.add_argument("--coarsen-factor", type=int, default=DEFAULT_COARSEN_FACTOR)


def region_from_args(
    args: argparse.Namespace,
    *,
    default: RegionBounds | None = DEFAULT_MODEL_REGION,
) -> RegionBounds | None:
    values = {
        "lat_min": getattr(args, "lat_min", None),
        "lat_max": getattr(args, "lat_max", None),
        "lon_min": getattr(args, "lon_min", None),
        "lon_max": getattr(args, "lon_max", None),
    }
    if all(value is None for value in values.values()):
        return default
    if any(value is None for value in values.values()):
        raise ValueError("Region overrides require all four bounds: --lat-min --lat-max --lon-min --lon-max")
    return RegionBounds(**values)


def region_from_dict(payload: dict[str, object]) -> RegionBounds:
    return RegionBounds(
        lat_min=float(payload["lat_min"]),
        lat_max=float(payload["lat_max"]),
        lon_min=float(payload["lon_min"]),
        lon_max=float(payload["lon_max"]),
    )


def region_to_dict(region: RegionBounds | None) -> dict[str, float] | None:
    if region is None:
        return None
    return region.as_dict()


def regions_match(left: RegionBounds, right: RegionBounds, *, atol: float = 1.0e-6) -> bool:
    return bool(
        abs(left.lat_min - right.lat_min) <= atol
        and abs(left.lat_max - right.lat_max) <= atol
        and abs(left.lon_min - right.lon_min) <= atol
        and abs(left.lon_max - right.lon_max) <= atol
    )


def discover_swe_files(limit: int = 10) -> list[Path]:
    return sorted(SWE_ROOT_PATH.glob("*.nc"))[:limit]


def discover_sst_files(limit: int = 10) -> list[Path]:
    return sorted(SST_ROOT_PATH.glob("*.nc"))[:limit]


def swe_file_for_water_year(water_year: int) -> Path:
    path = SWE_ROOT_PATH / f"WUS_UCLA_SR_v01_ALL_0_agg_16_WY{water_year}_SD_SWE_SCA_POST.nc"
    if not path.exists():
        raise FileNotFoundError(f"SWE file not found for water year {water_year}: {path}")
    return path


def era5_land_yearly_file(variable_name: str, year: int) -> Path:
    if variable_name == "t2m":
        path = ERA5_LAND_ROOT_PATH / "2m_temperature" / f"ERA5_{year}_2m_temperature.nc"
    elif variable_name == "tp":
        path = (
            ERA5_LAND_ROOT_PATH
            / "total_precipitation"
            / f"ERA5_{year}_total_precipitation.nc"
        )
    else:
        raise ValueError(f"Unsupported ERA5-Land variable: {variable_name}")

    if not path.exists():
        raise FileNotFoundError(f"ERA5-Land file not found for {variable_name} {year}: {path}")
    return path


def parse_month_day(month_day: str) -> tuple[int, int]:
    month_text, day_text = month_day.split("-", maxsplit=1)
    return int(month_text), int(day_text)


def target_date(config: ForecastConfig) -> date:
    month, day = parse_month_day(config.target_month_day)
    target_year = config.water_year if month < 10 else config.water_year - 1
    return date(target_year, month, day)


def target_day_of_year(config: ForecastConfig) -> int:
    return int(target_date(config).timetuple().tm_yday)


def swe_initial_condition_date(config: ForecastConfig) -> date:
    return date(config.water_year - 1, 12, 31)


def sst_window_dates(config: ForecastConfig) -> list[date]:
    if config.sst_months != 12:
        raise ValueError(
            f"This baseline expects a 12-month SST window, got {config.sst_months}."
        )
    target_year = target_date(config).year
    return [
        date(target_year - 1, 3, 1),
        date(target_year - 1, 4, 1),
        date(target_year - 1, 5, 1),
        date(target_year - 1, 6, 1),
        date(target_year - 1, 7, 1),
        date(target_year - 1, 8, 1),
        date(target_year - 1, 9, 1),
        date(target_year - 1, 10, 1),
        date(target_year - 1, 11, 1),
        date(target_year - 1, 12, 1),
        date(target_year, 1, 1),
        date(target_year, 2, 1),
    ]


def atmospheric_window_start_date(config: ForecastConfig) -> date:
    return target_date(config) - timedelta(days=config.atmospheric_days - 1)


def inspect_swe_grid(water_year: int = 2014) -> SweGridDefinition:
    grid = get_swe_grid_definition(water_year)
    print(f"SWE file: {grid.file_path}")
    print(f"SWE latitude coordinate: {grid.latitude_name}")
    print(f"SWE longitude coordinate: {grid.longitude_name}")
    print(f"SWE time coordinate: {grid.time_name}")
    print(f"SWE latitude is 2D: {grid.latitude_is_2d}")
    print(f"SWE longitude is 2D: {grid.longitude_is_2d}")
    print(f"cropped grid shape: {grid.cropped_shape}")
    print(f"trimmed grid shape: {grid.trimmed_shape}")
    print(f"coarsened grid shape: {grid.grid_shape}")
    print(f"coarsen factor: {grid.coarsen_factor}")
    print(f"requested region: {grid.requested_region.as_dict()}")
    print(f"effective region: {grid.effective_region.as_dict()}")
    print(f"SWE Stats dimension present: {grid.has_stats_dimension}")
    return grid


def get_swe_grid_definition(
    water_year: int = 2014,
    region: RegionBounds | None = None,
    coarsen_factor: int = DEFAULT_COARSEN_FACTOR,
) -> SweGridDefinition:
    path = swe_file_for_water_year(water_year)
    requested_region = region or DEFAULT_MODEL_REGION
    if coarsen_factor < 1:
        raise ValueError(f"coarsen_factor must be >= 1, got {coarsen_factor}")
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        variable = ds[SWE_VARIABLE]
        latitude_name = _find_coord_name(ds, ("Latitude", "latitude", "lat"))
        longitude_name = _find_coord_name(ds, ("Longitude", "longitude", "lon"))
        time_name = _find_coord_name(ds, ("time",))
        latitude = ds[latitude_name].load()
        longitude = ds[longitude_name].load()
        cropped_latitude, cropped_longitude = _crop_swe_coordinates(latitude, longitude, requested_region)
        trimmed_latitude, trimmed_longitude = _trim_swe_coordinates(
            cropped_latitude,
            cropped_longitude,
            coarsen_factor,
        )
        latitude, longitude = _coarsen_swe_coordinates(
            trimmed_latitude,
            trimmed_longitude,
            coarsen_factor,
        )
        effective_region = RegionBounds(
            lat_min=float(np.min(trimmed_latitude.values)),
            lat_max=float(np.max(trimmed_latitude.values)),
            lon_min=float(np.min(trimmed_longitude.values)),
            lon_max=float(np.max(trimmed_longitude.values)),
        )
        grid_shape = (int(latitude.size), int(longitude.size))
        return SweGridDefinition(
            water_year=water_year,
            file_path=str(path),
            latitude_name=latitude_name,
            longitude_name=longitude_name,
            time_name=time_name,
            has_stats_dimension="Stats" in variable.dims,
            latitude_is_2d=latitude.ndim == 2,
            longitude_is_2d=longitude.ndim == 2,
            grid_shape=grid_shape,
            latitude=latitude,
            longitude=longitude,
            cropped_shape=(int(cropped_latitude.size), int(cropped_longitude.size)),
            trimmed_shape=(int(trimmed_latitude.size), int(trimmed_longitude.size)),
            coarsen_factor=coarsen_factor,
            requested_region=requested_region,
            effective_region=effective_region,
            fine_latitude=trimmed_latitude,
            fine_longitude=trimmed_longitude,
        )


def get_regional_swe_grid_definition(
    water_year: int = 2014,
    region: RegionBounds | None = None,
    coarsen_factor: int = DEFAULT_COARSEN_FACTOR,
) -> SweGridDefinition:
    return get_swe_grid_definition(
        water_year=water_year,
        region=region,
        coarsen_factor=coarsen_factor,
    )


def open_dataset_with_fallbacks(
    path: Path,
    *,
    decode_times: bool = True,
) -> tuple[xr.Dataset, str]:
    failures: list[str] = []
    for engine_name in ("netcdf4", "h5netcdf", None):
        kwargs = {"decode_times": decode_times}
        if engine_name is not None:
            kwargs["engine"] = engine_name
        try:
            dataset = xr.open_dataset(path, **kwargs)
            return dataset, engine_name or "default"
        except Exception as exc:
            engine_label = engine_name or "default"
            failure = f"engine={engine_label}: {exc}"
            failures.append(failure)
            _profile_print(f"open_dataset failed for {path} with {failure}")
    raise RuntimeError(f"Could not open {path}. Failures: {failures}")


def load_swe_snapshot(
    water_year: int,
    snapshot_date: date,
    *,
    stat_name: str = "mean",
    swe_grid: SweGridDefinition | None = None,
    fill_missing: bool = True,
) -> xr.DataArray:
    path = swe_file_for_water_year(water_year)
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        swe = ds[SWE_VARIABLE].sel(time=np.datetime64(snapshot_date.isoformat()))
        swe = swe.where(swe != SWE_MISSING_VALUE)
        if "Stats" in swe.dims:
            swe = swe.isel(Stats=SWE_STAT_INDEX[stat_name], drop=True)
        if swe_grid is not None:
            swe = _subset_field_to_swe_grid(swe, swe_grid, coarsen_method="mean")
        swe = swe.astype("float32").load()
    swe.name = f"swe_{stat_name}_{snapshot_date.isoformat()}"
    swe.attrs["selected_statistic"] = stat_name
    swe.attrs["selected_stat_index"] = SWE_STAT_INDEX[stat_name]
    swe.attrs["selected_stat_reason"] = (
        "Stat index chosen from UCLA documentation: "
        "0=mean, 1=std, 2=median, 3=25th percentile, 4=75th percentile."
    )
    if fill_missing:
        return _replace_nonfinite_dataarray(swe, fill_threshold=1.0e19)
    return swe


def load_target_swe_map(
    config: ForecastConfig,
    *,
    swe_grid: SweGridDefinition | None = None,
    fill_missing: bool = True,
) -> xr.DataArray:
    return load_swe_snapshot(
        config.water_year,
        target_date(config),
        stat_name="mean",
        swe_grid=swe_grid,
        fill_missing=fill_missing,
    )


def build_valid_swe_mask(
    target_swe: xr.DataArray,
    swe_grid: SweGridDefinition | None = None,
) -> xr.DataArray:
    mask = xr.DataArray(
        np.isfinite(target_swe.values).astype(np.float32),
        dims=target_swe.dims,
        coords=target_swe.coords,
        name="valid_swe_mask",
    )
    if swe_grid is None:
        return mask
    mask = _subset_field_to_swe_grid(mask, swe_grid, coarsen_method="mean")
    mask.attrs["mask_type"] = MASK_TYPE
    return mask


def load_sst_history_on_swe_grid(
    config: ForecastConfig,
    swe_grid: SweGridDefinition,
) -> xr.DataArray:
    window_dates = sst_window_dates(config)
    start = np.datetime64(window_dates[0].isoformat())
    end = np.datetime64(window_dates[-1].isoformat())
    _profile_print("start load_sst_history_on_swe_grid")
    with xr.open_dataset(SST_MONTHLY_MEAN_PATH, engine="netcdf4", decode_times=True) as ds:
        step_start = perf_counter()
        sst = ds["sst"].sel(time=slice(start, end)).astype("float32")
        _timed_print("load_sst_raw_and_slice_window", step_start)
        step_start = perf_counter()
        sst = _prepare_field_for_regridding(
            sst,
            profile_label="sst",
        )
        _timed_print("prepare_sst_for_regridding", step_start)
        step_start = perf_counter()
        regridded = regrid_to_swe_grid(
            sst,
            swe_grid,
            method="linear",
            profile_label="sst",
        )
        _timed_print("build_sst_regrid_graph", step_start)
        step_start = perf_counter()
        _profile_print("start sst materialize")
        regridded = regridded.load()
        _timed_print("materialize_sst", step_start)
    step_start = perf_counter()
    regridded = _replace_nonfinite_dataarray(regridded, fill_threshold=1.0e19)
    _timed_print("replace_nonfinite_sst", step_start)
    regridded.name = "sst_history_on_swe_grid"
    return regridded


def load_era5_daily_history_on_swe_grid(
    variable_name: str,
    config: ForecastConfig,
    swe_grid: SweGridDefinition,
) -> xr.DataArray:
    start_date = atmospheric_window_start_date(config)
    end_date = target_date(config)
    parts: list[xr.DataArray] = []
    total_start = perf_counter()
    _profile_print(f"start load_{variable_name}_history")
    for year in range(start_date.year, end_date.year + 1):
        _profile_print(f"start {variable_name} year={year}")
        year_start = perf_counter()
        path = era5_land_yearly_file(variable_name, year)
        _profile_print(f"start {variable_name} year={year} open_dataset")
        step_start = perf_counter()
        with xr.open_dataset(
            path,
            engine="netcdf4",
            decode_times=True,
        ) as ds:
            _timed_print(f"{variable_name} year={year} open_dataset", step_start)
            _profile_print(f"start {variable_name} year={year} select_variable")
            step_start = perf_counter()
            field = ds[variable_name]
            _timed_print(f"{variable_name} year={year} select_variable", step_start)
            if "time" in ds.coords:
                _profile_print(f"start {variable_name} year={year} use_time_coordinate")
                step_start = perf_counter()
                _ = ds["time"]
                _timed_print(f"{variable_name} year={year} use_time_coordinate", step_start)
            selection_start = np.datetime64(start_date.isoformat())
            selection_end = np.datetime64(end_date.isoformat()) + np.timedelta64(23, "h")
            _profile_print(f"start {variable_name} year={year} select_time_window")
            step_start = perf_counter()
            field = field.sel(time=slice(selection_start, selection_end))
            _timed_print(f"{variable_name} year={year} select_time_window", step_start)
            _profile_print(
                f"{variable_name} year={year} selected field "
                f"shape={field.shape} dims={field.dims} dtype={field.dtype}"
            )
            if "time" in field.sizes:
                _profile_print(
                    f"{variable_name} year={year} selected time_length={field.sizes['time']}"
                )
                if int(field.sizes["time"]) > 0:
                    time_values = field["time"].values
                    _profile_print(
                        f"{variable_name} year={year} selected time_range="
                        f"{time_values[0]} -> {time_values[-1]}"
                    )
            if field.sizes.get("time", 0) == 0:
                _profile_print(f"{variable_name} year={year} select_time_window returned 0 slices")
                continue
            _profile_print(f"start {variable_name} year={year} early_spatial_subset")
            step_start = perf_counter()
            field = _subset_era5_field_to_region(
                field,
                swe_grid.effective_region,
                latitude_buffer_degrees=ERA5_EARLY_SUBSET_LAT_BUFFER_DEGREES,
                longitude_buffer_degrees=ERA5_EARLY_SUBSET_LON_BUFFER_DEGREES,
            )
            _timed_print(f"{variable_name} year={year} early_spatial_subset", step_start)
            _profile_print(
                f"{variable_name} year={year} cropped field "
                f"shape={field.shape} dims={field.dims} dtype={field.dtype}"
            )
            if "time" in field.sizes:
                _profile_print(
                    f"{variable_name} year={year} cropped time_length={field.sizes['time']}"
                )
            _profile_print(f"start {variable_name} year={year} build_daily_history")
            daily_history_start = perf_counter()
            use_fixed_blocks, fixed_block_reason = _can_use_exact_24h_daily_reduction(field)
            _profile_print(
                f"{variable_name} year={year} exact_24h_daily_reduction="
                f"{use_fixed_blocks} reason={fixed_block_reason}"
            )
            if use_fixed_blocks:
                _profile_print(f"start {variable_name} year={year} create_24h_coarsen")
                step_start = perf_counter()
                coarsener = field.coarsen(time=24, boundary="exact")
                _timed_print(f"{variable_name} year={year} create_24h_coarsen", step_start)
                if ERA5_DAILY_REDUCTIONS[variable_name] == "mean":
                    _profile_print(f"start {variable_name} year={year} reduce_daily_mean")
                    step_start = perf_counter()
                    daily = coarsener.mean()
                    _timed_print(f"{variable_name} year={year} reduce_daily_mean", step_start)
                else:
                    _profile_print(f"start {variable_name} year={year} reduce_daily_sum")
                    step_start = perf_counter()
                    daily = coarsener.sum()
                    _timed_print(f"{variable_name} year={year} reduce_daily_sum", step_start)
                daily = daily.assign_coords(time=field["time"].values[::24])
            else:
                step_start = perf_counter()
                resampler = field.resample(time="1D")
                if ERA5_DAILY_REDUCTIONS[variable_name] == "mean":
                    _timed_print(f"{variable_name} year={year} create_resampler", step_start)
                    _profile_print(f"start {variable_name} year={year} reduce_daily_mean")
                    step_start = perf_counter()
                    daily = resampler.mean()
                    _timed_print(f"{variable_name} year={year} reduce_daily_mean", step_start)
                else:
                    _timed_print(f"{variable_name} year={year} create_resampler", step_start)
                    _profile_print(f"start {variable_name} year={year} reduce_daily_sum")
                    step_start = perf_counter()
                    daily = resampler.sum()
                    _timed_print(f"{variable_name} year={year} reduce_daily_sum", step_start)
            _timed_print(f"{variable_name} year={year} build_daily_history", daily_history_start)
            _profile_print(
                f"{variable_name} year={year} daily field "
                f"shape={daily.shape} dims={daily.dims} dtype={daily.dtype}"
            )
            if "time" in daily.sizes:
                _profile_print(
                    f"{variable_name} year={year} daily time_length={daily.sizes['time']}"
                )
                if int(daily.sizes["time"]) > 0:
                    daily_time_values = daily["time"].values
                    _profile_print(
                        f"{variable_name} year={year} daily time_range="
                        f"{daily_time_values[0]} -> {daily_time_values[-1]}"
                    )
            _profile_print(f"start {variable_name} year={year} crop_region")
            step_start = perf_counter()
            daily = daily.astype("float32")
            _timed_print(f"{variable_name} year={year} crop_region", step_start)
            _profile_print(f"start {variable_name} year={year} prepare_for_regridding")
            step_start = perf_counter()
            daily = _prepare_field_for_regridding(
                daily,
                profile_label=f"{variable_name} year={year}",
            )
            _timed_print(f"{variable_name} year={year} prepare_for_regridding", step_start)
            _profile_print(f"start {variable_name} year={year} regrid_to_model_grid")
            step_start = perf_counter()
            daily = regrid_to_swe_grid(
                daily,
                swe_grid,
                method="linear",
                profile_label=f"{variable_name} year={year}",
            )
            _timed_print(f"{variable_name} year={year} regrid_to_model_grid", step_start)
            _profile_print(f"start {variable_name} year={year} materialize")
            _profile_print(
                f"{variable_name} year={year} pre-load daily "
                f"shape={daily.shape} dims={daily.dims} dtype={daily.dtype} sizes={dict(daily.sizes)}"
            )
            step_start = perf_counter()
            daily = daily.load()
            _timed_print(f"{variable_name} year={year} materialize", step_start)
            _profile_print(f"start {variable_name} year={year} append_store")
            step_start = perf_counter()
            parts.append(daily)
            _timed_print(f"{variable_name} year={year} append_store", step_start)
        _timed_print(f"{variable_name} year={year} total", year_start)

    if not parts:
        raise RuntimeError(
            f"No ERA5-Land daily history found for {variable_name} "
            f"between {start_date} and {end_date}"
        )

    _profile_print(f"start concat_{variable_name}_years")
    step_start = perf_counter()
    combined = xr.concat(parts, dim="time").sortby("time")
    _timed_print(f"concat_{variable_name}_years", step_start)
    _profile_print(f"start {variable_name} replace_nonfinite")
    step_start = perf_counter()
    combined = _replace_nonfinite_dataarray(combined, fill_threshold=1.0e19)
    _timed_print(f"{variable_name} replace_nonfinite", step_start)
    combined.name = f"{variable_name}_daily_on_swe_grid"
    _timed_print(f"load_{variable_name}_history_total", total_start)
    return combined


def load_terrain_height_on_swe_grid(swe_grid: SweGridDefinition) -> xr.DataArray:
    with xr.open_dataset(
        ERA5_LAND_GEOPOTENTIAL_PATH,
        engine="netcdf4",
        decode_times=False,
    ) as ds:
        geopotential = ds["z"].isel(time=0).astype("float32")
        geopotential = _prepare_field_for_regridding(geopotential)
        geopotential = regrid_to_swe_grid(geopotential, swe_grid, method="linear").load()

    terrain = (geopotential / GRAVITY_ACCELERATION).rename("terrain_height")
    terrain.attrs["source_variable"] = "z"
    terrain.attrs["source_units"] = "m**2 s**-2"
    terrain.attrs["derived_units"] = "m"
    terrain.attrs["derivation"] = "terrain_height = z / 9.81"
    return _replace_nonfinite_dataarray(terrain, fill_threshold=1.0e19)


def build_date_encoding_channels(
    config: ForecastConfig,
    swe_grid: SweGridDefinition,
) -> tuple[xr.DataArray, xr.DataArray]:
    day_index = target_day_of_year(config)
    angle = 2.0 * np.pi * day_index / 365.0
    sin_value = float(np.sin(angle))
    cos_value = float(np.cos(angle))
    sin_channel = _constant_channel_on_swe_grid(
        swe_grid,
        value=sin_value,
        name="query_date_sin",
    )
    cos_channel = _constant_channel_on_swe_grid(
        swe_grid,
        value=cos_value,
        name="query_date_cos",
    )
    return sin_channel, cos_channel


def build_sierra_mask(
    swe_grid: SweGridDefinition,
    *,
    region: RegionBounds = DEFAULT_SIERRA_REGION,
) -> xr.DataArray:
    if swe_grid.fine_latitude.ndim == 1 and swe_grid.fine_longitude.ndim == 1:
        lat2d, lon2d = xr.broadcast(swe_grid.fine_latitude, swe_grid.fine_longitude)
        mask = (
            (lat2d >= region.lat_min)
            & (lat2d <= region.lat_max)
            & (lon2d >= region.lon_min)
            & (lon2d <= region.lon_max)
        )
    elif swe_grid.fine_latitude.ndim == 2 and swe_grid.fine_longitude.ndim == 2:
        mask = (
            (swe_grid.fine_latitude >= region.lat_min)
            & (swe_grid.fine_latitude <= region.lat_max)
            & (swe_grid.fine_longitude >= region.lon_min)
            & (swe_grid.fine_longitude <= region.lon_max)
        )
    else:
        raise ValueError(
            "Unsupported SWE coordinate layout. "
            "Latitude and longitude must both be 1D or both be 2D."
        )

    mask = mask.astype(np.float32).rename("sierra_mask")
    mask = _coarsen_spatial_field(mask, swe_grid, reduction="mean")
    mask.attrs["region"] = region.as_dict()
    mask.attrs["mask_type"] = MASK_TYPE
    return mask


def summarize_mask(mask: xr.DataArray) -> dict[str, object]:
    mask_sum = float(mask.sum().item())
    total_count = int(mask.size)
    return {
        "mask_shape": list(mask.shape),
        "weighted_cell_sum": mask_sum,
        "total_cell_count": total_count,
        "mean_weight": float(mask_sum / total_count) if total_count else 0.0,
        "mask_type": str(mask.attrs.get("mask_type", "binary_or_unknown")),
        "dims": list(mask.dims),
    }


def aggregate_over_mask(field: xr.DataArray, mask: xr.DataArray) -> dict[str, float]:
    masked = field.where(mask)
    spatial_dims = list(mask.dims)
    return {
        "sum": float(masked.sum(dim=spatial_dims, skipna=True).item()),
        "mean": float(masked.mean(dim=spatial_dims, skipna=True).item()),
    }


def print_current_forecast_setup(config: ForecastConfig | None = None) -> None:
    active = config or default_forecast_config()
    _profile_print(f"water year: {active.water_year}")
    _profile_print(f"selected target month-day: {active.target_month_day}")
    _profile_print(f"history years: {active.history_years}")
    _profile_print(f"atmospheric history days: {active.atmospheric_days}")
    _profile_print(f"target date: {target_date(active).isoformat()}")
    _profile_print(f"target day-of-year: {target_day_of_year(active)}")
    _profile_print(f"sst history window: {[value.isoformat() for value in sst_window_dates(active)]}")
    _profile_print(
        "atmospheric history start: "
        f"{atmospheric_window_start_date(active).isoformat()}"
    )
    _profile_print(f"Dec 31 initial-condition date: {swe_initial_condition_date(active).isoformat()}")
    _profile_print(f"requested model region: {active.region.as_dict()}")
    _profile_print(f"coarsen factor: {active.coarsen_factor}")
    _profile_print(f"SWE root: {SWE_ROOT_PATH}")
    _profile_print(f"SST monthly file: {SST_MONTHLY_MEAN_PATH}")
    _profile_print(f"SST climatology file: {SST_LTM_PATH}")
    _profile_print(f"terrain source file: {ERA5_LAND_GEOPOTENTIAL_PATH}")


def regrid_to_swe_grid(
    field: xr.DataArray,
    swe_grid: SweGridDefinition,
    *,
    method: str = "linear",
    profile_label: str | None = None,
) -> xr.DataArray:
    if profile_label is not None:
        _profile_print(f"start {profile_label} find_regrid_coordinates")
    step_start = perf_counter()
    field_lat_name = _find_coord_name(field.to_dataset(name="field"), ("latitude", "lat", "Latitude"))
    field_lon_name = _find_coord_name(field.to_dataset(name="field"), ("longitude", "lon", "Longitude"))
    if profile_label is not None:
        _timed_print(f"{profile_label} find_regrid_coordinates", step_start)
    if swe_grid.latitude.ndim != 1 or swe_grid.longitude.ndim != 1:
        raise ValueError("Current regridding helper expects a SWE grid with 1D latitude/longitude.")
    if profile_label is not None:
        _profile_print(f"start {profile_label} interp")
    step_start = perf_counter()
    regridded = field.interp(
        {
            field_lat_name: swe_grid.latitude.values,
            field_lon_name: swe_grid.longitude.values,
        },
        method=method,
    )
    if profile_label is not None:
        _timed_print(f"{profile_label} interp", step_start)
    if field_lat_name != swe_grid.latitude_name or field_lon_name != swe_grid.longitude_name:
        if profile_label is not None:
            _profile_print(f"start {profile_label} rename_regridded_coordinates")
        step_start = perf_counter()
        regridded = regridded.rename(
            {
                field_lat_name: swe_grid.latitude_name,
                field_lon_name: swe_grid.longitude_name,
            }
        )
        if profile_label is not None:
            _timed_print(f"{profile_label} rename_regridded_coordinates", step_start)
    if profile_label is not None:
        _profile_print(f"start {profile_label} assign_regridded_coordinates")
    step_start = perf_counter()
    regridded = regridded.assign_coords(
        {
            swe_grid.latitude_name: swe_grid.latitude,
            swe_grid.longitude_name: swe_grid.longitude,
        }
    )
    if profile_label is not None:
        _timed_print(f"{profile_label} assign_regridded_coordinates", step_start)
    return regridded


def _prepare_field_for_regridding(
    field: xr.DataArray,
    *,
    profile_label: str | None = None,
) -> xr.DataArray:
    if profile_label is not None:
        _profile_print(f"start {profile_label} find_input_coordinates")
    step_start = perf_counter()
    latitude_name = _find_coord_name(field.to_dataset(name="field"), ("latitude", "lat", "Latitude"))
    longitude_name = _find_coord_name(field.to_dataset(name="field"), ("longitude", "lon", "Longitude"))
    if profile_label is not None:
        _timed_print(f"{profile_label} find_input_coordinates", step_start)
    prepared = field
    if profile_label is not None:
        _profile_print(f"start {profile_label} normalize_longitude")
    step_start = perf_counter()
    prepared = _normalize_longitude_coordinate(prepared, longitude_name)
    if profile_label is not None:
        _timed_print(f"{profile_label} normalize_longitude", step_start)
    if profile_label is not None:
        _profile_print(f"start {profile_label} sort_latitude")
    step_start = perf_counter()
    prepared = prepared.sortby(latitude_name)
    if profile_label is not None:
        _timed_print(f"{profile_label} sort_latitude", step_start)
    if profile_label is not None:
        _profile_print(f"start {profile_label} sort_longitude")
    step_start = perf_counter()
    prepared = prepared.sortby(longitude_name)
    if profile_label is not None:
        _timed_print(f"{profile_label} sort_longitude", step_start)
    return prepared


def _subset_era5_field_to_region(
    field: xr.DataArray,
    region: RegionBounds,
    *,
    latitude_buffer_degrees: float = ERA5_EARLY_SUBSET_LAT_BUFFER_DEGREES,
    longitude_buffer_degrees: float = ERA5_EARLY_SUBSET_LON_BUFFER_DEGREES,
) -> xr.DataArray:
    latitude_name = _find_coord_name(field.to_dataset(name="field"), ("latitude", "lat", "Latitude"))
    longitude_name = _find_coord_name(field.to_dataset(name="field"), ("longitude", "lon", "Longitude"))

    step_start = perf_counter()
    _profile_print("start era5 early subset normalize_longitude")
    subset = _normalize_longitude_coordinate(field, longitude_name)
    _timed_print("era5 early subset normalize_longitude", step_start)

    latitude = subset[latitude_name]
    longitude = subset[longitude_name]
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("ERA5 early spatial subset expects 1D latitude and longitude coordinates.")

    lat_min = region.lat_min - latitude_buffer_degrees
    lat_max = region.lat_max + latitude_buffer_degrees
    lon_min = region.lon_min - longitude_buffer_degrees
    lon_max = region.lon_max + longitude_buffer_degrees

    latitude_values = np.asarray(latitude.values, dtype=np.float64)
    longitude_values = np.asarray(longitude.values, dtype=np.float64)

    if latitude_values.size > 1 and np.any(np.diff(latitude_values) < 0.0):
        step_start = perf_counter()
        _profile_print("start era5 early subset sort_latitude")
        subset = subset.sortby(latitude_name)
        _timed_print("era5 early subset sort_latitude", step_start)
        latitude = subset[latitude_name]
        latitude_values = np.asarray(latitude.values, dtype=np.float64)

    if longitude_values.size > 1 and np.any(np.diff(longitude_values) < 0.0):
        step_start = perf_counter()
        _profile_print("start era5 early subset sort_longitude")
        subset = subset.sortby(longitude_name)
        _timed_print("era5 early subset sort_longitude", step_start)
        longitude = subset[longitude_name]
        longitude_values = np.asarray(longitude.values, dtype=np.float64)

    if latitude_values.size > 1 and latitude_values[0] <= latitude_values[-1]:
        latitude_slice = slice(lat_min, lat_max)
    else:
        latitude_slice = slice(lat_max, lat_min)
    longitude_slice = slice(lon_min, lon_max)

    step_start = perf_counter()
    _profile_print("start era5 early subset slice_subset")
    subset = subset.sel(
        {
            latitude_name: latitude_slice,
            longitude_name: longitude_slice,
        }
    )
    _timed_print("era5 early subset slice_subset", step_start)
    _profile_print(
        f"era5 early subset result shape={subset.shape} dims={subset.dims}"
    )
    if "time" in subset.sizes:
        _profile_print(f"era5 early subset result time_length={subset.sizes['time']}")
    if subset.sizes.get(latitude_name, 0) > 0:
        subset_latitude = subset[latitude_name].values
        _profile_print(
            f"era5 early subset latitude_range={subset_latitude[0]} -> {subset_latitude[-1]}"
        )
    if subset.sizes.get(longitude_name, 0) > 0:
        subset_longitude = subset[longitude_name].values
        _profile_print(
            f"era5 early subset longitude_range={subset_longitude[0]} -> {subset_longitude[-1]}"
        )
    return subset


def _can_use_exact_24h_daily_reduction(field: xr.DataArray) -> tuple[bool, str]:
    if "time" not in field.coords:
        return False, "missing_time_coordinate"

    time_values = np.asarray(field["time"].values)
    if time_values.ndim != 1 or time_values.size == 0:
        return False, "empty_or_non_1d_time"

    if time_values.size % 24 != 0:
        return False, "time_length_not_divisible_by_24"

    first_time = np.datetime64(time_values[0], "h")
    last_time = np.datetime64(time_values[-1], "h")
    if int((first_time - first_time.astype("datetime64[D]")) / np.timedelta64(1, "h")) != 0:
        return False, "first_timestamp_not_midnight"
    if int((last_time - last_time.astype("datetime64[D]")) / np.timedelta64(1, "h")) != 23:
        return False, "last_timestamp_not_23h"

    time_deltas = np.diff(time_values.astype("datetime64[h]"))
    if time_deltas.size and not np.all(time_deltas == np.timedelta64(1, "h")):
        return False, "time_steps_not_strictly_hourly"

    return True, "exact_hourly_blocks"


def _normalize_longitude_coordinate(field: xr.DataArray, longitude_name: str) -> xr.DataArray:
    longitude = field[longitude_name]
    if longitude.ndim != 1:
        return field
    normalized = xr.where(longitude > 180.0, longitude - 360.0, longitude)
    field = field.assign_coords({longitude_name: normalized})
    return field.sortby(longitude_name)


def _crop_swe_coordinates(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    region: RegionBounds,
) -> tuple[xr.DataArray, xr.DataArray]:
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Regional cropping currently expects 1D SWE latitude and longitude coordinates.")
    lat_indexer = _coordinate_indexer(latitude, region.lat_min, region.lat_max)
    lon_indexer = _coordinate_indexer(longitude, region.lon_min, region.lon_max)
    return latitude.isel({latitude.dims[0]: lat_indexer}), longitude.isel({longitude.dims[0]: lon_indexer})


def _trim_swe_coordinates(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    coarsen_factor: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    trimmed_lat_size = (int(latitude.size) // coarsen_factor) * coarsen_factor
    trimmed_lon_size = (int(longitude.size) // coarsen_factor) * coarsen_factor
    if trimmed_lat_size == 0 or trimmed_lon_size == 0:
        raise ValueError(
            f"Coarsen factor {coarsen_factor} is too large for cropped SWE grid "
            f"{(int(latitude.size), int(longitude.size))}."
        )
    return (
        latitude.isel({latitude.dims[0]: slice(0, trimmed_lat_size)}),
        longitude.isel({longitude.dims[0]: slice(0, trimmed_lon_size)}),
    )


def _coarsen_swe_coordinates(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    coarsen_factor: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    coarse_latitude = latitude.coarsen({latitude.dims[0]: coarsen_factor}, boundary=COARSEN_TRIMMING_POLICY).mean()
    coarse_longitude = longitude.coarsen({longitude.dims[0]: coarsen_factor}, boundary=COARSEN_TRIMMING_POLICY).mean()
    return coarse_latitude.astype(np.float32), coarse_longitude.astype(np.float32)


def _coordinate_indexer(values: xr.DataArray, lower: float, upper: float) -> slice:
    coordinate = np.asarray(values.values, dtype=np.float64)
    matches = np.where((coordinate >= lower) & (coordinate <= upper))[0]
    if matches.size == 0:
        raise ValueError(
            f"Requested region [{lower}, {upper}] does not overlap available coordinate range "
            f"[{float(np.min(coordinate))}, {float(np.max(coordinate))}] for {values.name}."
        )
    return slice(int(matches[0]), int(matches[-1]) + 1)


def _subset_field_to_swe_grid(
    values: xr.DataArray,
    swe_grid: SweGridDefinition,
    *,
    coarsen_method: str,
) -> xr.DataArray:
    subset = values.sel(
        {
            swe_grid.latitude_name: swe_grid.fine_latitude,
            swe_grid.longitude_name: swe_grid.fine_longitude,
        }
    )
    return _coarsen_spatial_field(subset, swe_grid, reduction=coarsen_method)


def _coarsen_spatial_field(
    values: xr.DataArray,
    swe_grid: SweGridDefinition,
    *,
    reduction: str,
) -> xr.DataArray:
    coarsened = values.coarsen(
        {
            swe_grid.latitude_name: swe_grid.coarsen_factor,
            swe_grid.longitude_name: swe_grid.coarsen_factor,
        },
        boundary=COARSEN_TRIMMING_POLICY,
    )
    if reduction == "mean":
        reduced = coarsened.mean()
    elif reduction == "sum":
        reduced = coarsened.sum()
    elif reduction == "std":
        reduced = coarsened.std()
    else:
        raise ValueError(f"Unsupported coarsen reduction: {reduction}")
    return reduced.assign_coords(
        {
            swe_grid.latitude_name: swe_grid.latitude,
            swe_grid.longitude_name: swe_grid.longitude,
        }
    )


def _constant_channel_on_swe_grid(
    swe_grid: SweGridDefinition,
    *,
    value: float,
    name: str,
) -> xr.DataArray:
    if swe_grid.latitude.ndim == 1 and swe_grid.longitude.ndim == 1:
        values = np.full(swe_grid.grid_shape, value, dtype=np.float32)
        return xr.DataArray(
            values,
            dims=(swe_grid.latitude_name, swe_grid.longitude_name),
            coords={
                swe_grid.latitude_name: swe_grid.latitude,
                swe_grid.longitude_name: swe_grid.longitude,
            },
            name=name,
        )
    if swe_grid.latitude.ndim == 2 and swe_grid.longitude.ndim == 2:
        values = np.full(swe_grid.grid_shape, value, dtype=np.float32)
        return xr.DataArray(
            values,
            dims=swe_grid.latitude.dims,
            coords={
                swe_grid.latitude_name: swe_grid.latitude,
                swe_grid.longitude_name: swe_grid.longitude,
            },
            name=name,
        )
    raise ValueError("Unsupported SWE coordinate layout for constant channel creation.")


def _replace_nonfinite_dataarray(
    values: xr.DataArray,
    *,
    fill_threshold: float | None = None,
) -> xr.DataArray:
    array = np.asarray(values.values, dtype=np.float32)
    if fill_threshold is not None:
        array = np.where(np.abs(array) >= fill_threshold, np.nan, array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return xr.DataArray(
        array,
        dims=values.dims,
        coords=values.coords,
        attrs=values.attrs,
        name=values.name,
    )


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"Could not find coordinate among candidates: {candidates}")


def _infer_grid_shape(
    variable: xr.DataArray,
    latitude_name: str,
    longitude_name: str,
) -> tuple[int, ...]:
    if latitude_name in variable.dims and longitude_name in variable.dims:
        return (
            int(variable.sizes[latitude_name]),
            int(variable.sizes[longitude_name]),
        )
    return tuple(int(size) for size in variable.shape[-2:])
