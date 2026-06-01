from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import xarray as xr

from snow_ml.data import (
    DEFAULT_TARGET_MONTH_DAYS,
    DEFAULT_MODEL_REGION,
    ForecastConfig,
    atmospheric_window_start_date,
    build_valid_swe_mask,
    build_date_encoding_channels,
    build_sierra_mask,
    default_forecast_config,
    get_regional_swe_grid_definition,
    load_era5_daily_history_on_swe_grid,
    load_sst_history_on_swe_grid,
    load_swe_snapshot,
    load_target_swe_map,
    load_terrain_height_on_swe_grid,
    print_current_forecast_setup,
    summarize_mask,
    RegionBounds,
    region_to_dict,
    swe_initial_condition_date,
    sst_window_dates,
    target_date,
    DEFAULT_SIERRA_REGION,
)

STANDARDIZATION_EPS = 1.0e-6
INPUT_CHANNEL_COUNT = 66
PRECIP_BACKGROUND_CHOICE = "std"


def _profile_print(message: str) -> None:
    print(message, flush=True)


def _timed_print(label: str, start: float) -> float:
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s", flush=True)
    return elapsed


@dataclass(frozen=True)
class ForecastWindowSummary:
    sst_history_shape: tuple[int, ...]
    t2m_daily_shape: tuple[int, ...]
    tp_daily_shape: tuple[int, ...]
    swe_initial_condition_shape: tuple[int, ...]
    terrain_height_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class DeterministicForecastSpec:
    name: str
    model_description: str
    model_input_channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    status: str


def forecast_config_from_env() -> ForecastConfig:
    default = default_forecast_config()
    water_year = int(os.environ.get("SNOW_ML_WATER_YEAR", str(default.water_year)))
    target_month_day = os.environ.get("SNOW_ML_TARGET_MMDD", default.target_month_day)
    history_years = int(os.environ.get("SNOW_ML_HISTORY_YEARS", str(default.history_years)))
    region = RegionBounds(
        lat_min=float(os.environ.get("SNOW_ML_LAT_MIN", str(default.region.lat_min))),
        lat_max=float(os.environ.get("SNOW_ML_LAT_MAX", str(default.region.lat_max))),
        lon_min=float(os.environ.get("SNOW_ML_LON_MIN", str(default.region.lon_min))),
        lon_max=float(os.environ.get("SNOW_ML_LON_MAX", str(default.region.lon_max))),
    )
    coarsen_factor = int(os.environ.get("SNOW_ML_COARSEN_FACTOR", str(default.coarsen_factor)))
    return ForecastConfig(
        water_year=water_year,
        target_month_day=target_month_day,
        sst_months=default.sst_months,
        history_years=history_years,
        region=region,
        coarsen_factor=coarsen_factor,
    )


def describe_current_model_design() -> DeterministicForecastSpec:
    return DeterministicForecastSpec(
        name="date_conditioned_regional_swe_grid_baseline",
        model_description=(
            "One date-conditioned 2D U-Net baseline on a configurable cropped SWE regional grid. "
            "Each sample is indexed by water year and target date, all non-SWE fields are "
            "regridded to the cropped SWE grid, and the model predicts the target-date SWE mean map."
        ),
        model_input_channels=tuple(channel_names()),
        target_channels=("swe_mean_target",),
        status=(
            "Deterministic map-prediction baseline. Training uses a valid-SWE mask and "
            "Sierra regional diagnostics are computed after prediction."
        ),
    )


def channel_names() -> list[str]:
    names: list[str] = []
    for month_index in range(12):
        names.append(f"sst_month_{month_index + 1:02d}")
    names.extend(_multiscale_channel_names("t2m"))
    names.extend(_multiscale_channel_names("tp"))
    names.append("swe_dec31_mean")
    names.append("terrain_height")
    names.append("query_date_sin")
    names.append("query_date_cos")
    if len(names) != INPUT_CHANNEL_COUNT:
        raise RuntimeError(f"Expected {INPUT_CHANNEL_COUNT} input channels, got {len(names)}")
    return names


def build_training_sample(
    config: ForecastConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    total_start = perf_counter()
    timing_summary: list[tuple[str, float]] = []
    active = config or forecast_config_from_env()
    _profile_print(
        f"start sample wy={active.water_year} mmdd={active.target_month_day}"
    )
    print_current_forecast_setup(active)

    start = perf_counter()
    swe_grid = get_regional_swe_grid_definition(
        active.water_year,
        active.region,
        active.coarsen_factor,
    )
    timing_summary.append(("get_regional_swe_grid_definition", _timed_print("get_regional_swe_grid_definition", start)))
    start = perf_counter()
    sierra_mask = build_sierra_mask(swe_grid)
    timing_summary.append(("build_sierra_mask", _timed_print("build_sierra_mask", start)))
    start = perf_counter()
    sst_history = load_sst_history_on_swe_grid(active, swe_grid)
    timing_summary.append(("load_sst_history_on_swe_grid", _timed_print("load_sst_history_on_swe_grid", start)))
    start = perf_counter()
    t2m_daily = load_era5_daily_history_on_swe_grid("t2m", active, swe_grid)
    timing_summary.append(("load_t2m_history", _timed_print("load_t2m_history", start)))
    start = perf_counter()
    tp_daily = load_era5_daily_history_on_swe_grid("tp", active, swe_grid)
    timing_summary.append(("load_tp_history", _timed_print("load_tp_history", start)))
    start = perf_counter()
    swe_initial = load_swe_snapshot(
        active.water_year,
        swe_initial_condition_date(active),
        stat_name="mean",
        swe_grid=swe_grid,
    )
    timing_summary.append(("load_dec31_swe", _timed_print("load_dec31_swe", start)))
    start = perf_counter()
    terrain_height = load_terrain_height_on_swe_grid(swe_grid)
    timing_summary.append(("load_terrain", _timed_print("load_terrain", start)))
    start = perf_counter()
    date_sin, date_cos = build_date_encoding_channels(active, swe_grid)
    timing_summary.append(("build_date_encoding_channels", _timed_print("build_date_encoding_channels", start)))
    start = perf_counter()
    fine_target_with_nan = load_target_swe_map(active, swe_grid=None, fill_missing=False)
    timing_summary.append(("build_target_swe_fine", _timed_print("build_target_swe_fine", start)))
    start = perf_counter()
    target_with_nan = load_target_swe_map(active, swe_grid=swe_grid, fill_missing=False)
    timing_summary.append(("build_target_swe_coarse", _timed_print("build_target_swe_coarse", start)))
    start = perf_counter()
    valid_swe_mask = build_valid_swe_mask(fine_target_with_nan, swe_grid)
    timing_summary.append(("build_valid_swe_mask", _timed_print("build_valid_swe_mask", start)))
    target = target_with_nan.where(valid_swe_mask, 0.0).astype(np.float32)

    start = perf_counter()
    model_fields, model_channel_names = _build_model_input_fields(
        sst_history=sst_history,
        t2m_daily=t2m_daily,
        tp_daily=tp_daily,
        swe_initial=swe_initial,
        terrain_height=terrain_height,
        date_sin=date_sin,
        date_cos=date_cos,
    )
    timing_summary.append(("build_model_input_fields", _timed_print("build_model_input_fields", start)))
    start = perf_counter()
    input_stack = xr.concat(model_fields, dim="channel").assign_coords(
        channel=model_channel_names
    )
    timing_summary.append(("stack_final_input", _timed_print("stack_final_input", start)))
    target_stack = target.expand_dims(channel=["swe_mean_target"])

    start = perf_counter()
    input_array = np.asarray(input_stack.values, dtype=np.float32)
    target_array = np.asarray(target_stack.values, dtype=np.float32)
    valid_mask_array = np.asarray(valid_swe_mask.values, dtype=np.float32)[None, ...]
    sierra_mask_array = np.asarray(sierra_mask.values, dtype=np.float32)[None, ...]
    timing_summary.append(("np.asarray", _timed_print("np.asarray", start)))

    window_summary = ForecastWindowSummary(
        sst_history_shape=tuple(int(size) for size in sst_history.shape),
        t2m_daily_shape=tuple(int(size) for size in t2m_daily.shape),
        tp_daily_shape=tuple(int(size) for size in tp_daily.shape),
        swe_initial_condition_shape=tuple(int(size) for size in swe_initial.shape),
        terrain_height_shape=tuple(int(size) for size in terrain_height.shape),
        target_shape=tuple(int(size) for size in target.shape),
    )

    _profile_print(f"requested region: {swe_grid.requested_region.as_dict()}")
    _profile_print(f"effective grid region: {swe_grid.effective_region.as_dict()}")
    _profile_print(f"cropped regional SWE grid shape: {swe_grid.cropped_shape}")
    _profile_print(f"trimmed regional SWE grid shape: {swe_grid.trimmed_shape}")
    _profile_print(f"coarsened regional SWE grid shape: {swe_grid.grid_shape}")
    _profile_print(f"coarsen factor: {swe_grid.coarsen_factor}")
    _profile_print(f"valid SWE mask summary: {summarize_mask(valid_swe_mask)}")
    _profile_print(f"Sierra mask summary: {summarize_mask(sierra_mask)}")
    _profile_print(f"model input channel count: {len(model_channel_names)}")
    _profile_print(f"final input stack shape (C, H, W): {tuple(input_array.shape)}")
    _profile_print(f"final target shape (1, H, W): {tuple(target_array.shape)}")
    _profile_print(f"valid mask shape (1, H, W): {tuple(valid_mask_array.shape)}")
    _profile_print(f"sierra mask shape (1, H, W): {tuple(sierra_mask_array.shape)}")

    metadata = {
        "experiment_name": "date_conditioned_regional_swe_grid_baseline",
        "baseline_model_description": describe_current_model_design().model_description,
        "inputs_used": list(model_channel_names),
        "target_definition": (
            "Target is the SWE mean field on the target date, using SWE Stats index 0 "
            "from the UCLA dataset."
        ),
        "target_statistic_name": "mean",
        "target_statistic_index": 0,
        "target_month_day": active.target_month_day,
        "target_date_iso": target_date(active).isoformat(),
        "swe_initial_condition_date_iso": swe_initial_condition_date(active).isoformat(),
        "sst_window_dates": [value.isoformat() for value in sst_window_dates(active)],
        "atmospheric_window_start_date_iso": atmospheric_window_start_date(active).isoformat(),
        "water_year": active.water_year,
        "history_years": active.history_years,
        "atmospheric_history_days": active.atmospheric_days,
        "default_target_month_days": list(DEFAULT_TARGET_MONTH_DAYS),
        "default_model_region": region_to_dict(DEFAULT_MODEL_REGION),
        "requested_region": region_to_dict(swe_grid.requested_region),
        "effective_region": region_to_dict(swe_grid.effective_region),
        "coarsen_factor": swe_grid.coarsen_factor,
        "original_cropped_grid_shape": list(swe_grid.cropped_shape),
        "trimmed_grid_shape": list(swe_grid.trimmed_shape),
        "coarsened_grid_shape": list(swe_grid.grid_shape),
        "mask_type": "fractional",
        "trimming_policy": "trim",
        "sierra_region": region_to_dict(DEFAULT_SIERRA_REGION),
        "precipitation_background_choice": PRECIP_BACKGROUND_CHOICE,
        "longitude_normalization": "normalize source longitudes from 0..360 to -180..180 before interpolation",
        "window_summary": {
            "sst_history_shape": list(window_summary.sst_history_shape),
            "t2m_daily_shape": list(window_summary.t2m_daily_shape),
            "tp_daily_shape": list(window_summary.tp_daily_shape),
            "swe_initial_condition_shape": list(window_summary.swe_initial_condition_shape),
            "terrain_height_shape": list(window_summary.terrain_height_shape),
            "target_shape": list(window_summary.target_shape),
        },
        "grid": {
            "shape": list(swe_grid.grid_shape),
            "cropped_shape": list(swe_grid.cropped_shape),
            "trimmed_shape": list(swe_grid.trimmed_shape),
            "latitude_name": swe_grid.latitude_name,
            "longitude_name": swe_grid.longitude_name,
            "latitude_is_2d": swe_grid.latitude_is_2d,
            "longitude_is_2d": swe_grid.longitude_is_2d,
        },
        "valid_swe_mask_summary": summarize_mask(valid_swe_mask),
        "sierra_mask_summary": summarize_mask(sierra_mask),
    }
    total_elapsed = perf_counter() - total_start
    _timed_print("sample_total", total_start)
    _profile_print(
        "sample_timing_summary: "
        + ", ".join(f"{label}={elapsed:.3f}s" for label, elapsed in timing_summary)
        + f", sample_total={total_elapsed:.3f}s"
    )
    return input_array, target_array, valid_mask_array, sierra_mask_array, metadata


def _build_model_input_fields(
    *,
    sst_history: xr.DataArray,
    t2m_daily: xr.DataArray,
    tp_daily: xr.DataArray,
    swe_initial: xr.DataArray,
    terrain_height: xr.DataArray,
    date_sin: xr.DataArray,
    date_cos: xr.DataArray,
) -> tuple[list[xr.DataArray], list[str]]:
    fields: list[xr.DataArray] = []
    names: list[str] = []

    for month_index in range(int(sst_history.sizes["time"])):
        fields.append(_standardize_channel(sst_history.isel(time=month_index)))
        names.append(f"sst_month_{month_index + 1:02d}")

    t2m_fields, t2m_names = _build_multiscale_history_fields("t2m", t2m_daily, weekly_reduction="mean")
    fields.extend(t2m_fields)
    names.extend(t2m_names)

    tp_fields, tp_names = _build_multiscale_history_fields("tp", tp_daily, weekly_reduction="sum")
    fields.extend(tp_fields)
    names.extend(tp_names)

    fields.append(_standardize_channel(swe_initial))
    names.append("swe_dec31_mean")

    fields.append(_standardize_channel(terrain_height))
    names.append("terrain_height")

    fields.append(date_sin.astype(np.float32))
    names.append("query_date_sin")

    fields.append(date_cos.astype(np.float32))
    names.append("query_date_cos")

    _print_channel_stats(names, fields)
    if names != channel_names():
        raise RuntimeError(f"Unexpected channel order: {names}")
    return fields, names


def _build_multiscale_history_fields(
    prefix: str,
    values: xr.DataArray,
    *,
    weekly_reduction: str,
) -> tuple[list[xr.DataArray], list[str]]:
    total_start = perf_counter()
    _profile_print(f"start build_{prefix}_channels")
    step_start = perf_counter()
    history = _last_history_days(values)
    _timed_print(f"slice_{prefix}_history_window", step_start)
    fields: list[xr.DataArray] = []
    names: list[str] = []

    daily_offsets = list(range(13, -1, -1))
    step_start = perf_counter()
    for offset in daily_offsets:
        fields.append(_standardize_channel(history.isel(time=-(offset + 1))))
        names.append(f"{prefix}_dminus{offset:03d}")
    _timed_print(f"build_{prefix}_daily_channels", step_start)

    weekly_bins = (
        (55, 49),
        (48, 42),
        (41, 35),
        (34, 28),
        (27, 21),
        (20, 14),
    )
    step_start = perf_counter()
    for older, newer in weekly_bins:
        weekly_slice = _slice_by_offsets(history, older, newer)
        if weekly_reduction == "mean":
            weekly_field = weekly_slice.mean(dim="time")
        else:
            weekly_field = weekly_slice.sum(dim="time")
        fields.append(_standardize_channel(weekly_field))
        names.append(f"{prefix}_days_{older:03d}_{newer:03d}")
    _timed_print(f"build_{prefix}_weekly_channels", step_start)

    coarse_bins = (
        (119, 56),
        (239, 120),
        (364, 240),
        (729, 365),
    )
    step_start = perf_counter()
    for older, newer in coarse_bins:
        coarse_slice = _slice_by_offsets(history, older, newer)
        if prefix == "tp":
            coarse_field = coarse_slice.sum(dim="time")
        else:
            coarse_field = coarse_slice.mean(dim="time")
        fields.append(_standardize_channel(coarse_field))
        names.append(f"{prefix}_days_{older:03d}_{newer:03d}")

    full_background = history.std(dim="time")
    fields.append(_standardize_channel(full_background))
    names.append(f"{prefix}_full_2y_std")
    _timed_print(f"build_{prefix}_coarse_channels", step_start)
    _timed_print(f"build_{prefix}_channels_total", total_start)

    return fields, names


def _multiscale_channel_names(prefix: str) -> list[str]:
    names = [f"{prefix}_dminus{offset:03d}" for offset in range(13, -1, -1)]
    names.extend(
        [
            f"{prefix}_days_055_049",
            f"{prefix}_days_048_042",
            f"{prefix}_days_041_035",
            f"{prefix}_days_034_028",
            f"{prefix}_days_027_021",
            f"{prefix}_days_020_014",
            f"{prefix}_days_119_056",
            f"{prefix}_days_239_120",
            f"{prefix}_days_364_240",
            f"{prefix}_days_729_365",
            f"{prefix}_full_2y_std",
        ]
    )
    if len(names) != 25:
        raise RuntimeError(f"Expected 25 {prefix} channels, got {len(names)}")
    return names


def _last_history_days(values: xr.DataArray) -> xr.DataArray:
    history_days = int(values.sizes["time"])
    minimum_required_days = 365
    if history_days < minimum_required_days:
        raise ValueError(
            f"Expected at least {minimum_required_days} daily slices, got {values.sizes['time']}"
        )
    history = values.isel(time=slice(-history_days, None)).sortby("time")
    if int(history.sizes["time"]) != history_days:
        raise ValueError(
            f"Expected exactly {history_days} daily slices after trimming, got {history.sizes['time']}"
        )
    return history


def _slice_by_offsets(values: xr.DataArray, older: int, newer: int) -> xr.DataArray:
    history_days = int(values.sizes["time"])
    clipped_older = min(older, history_days - 1)
    clipped_newer = min(newer, history_days - 1)
    start_index = history_days - 1 - clipped_older
    stop_index = history_days - clipped_newer
    return values.isel(time=slice(start_index, stop_index))


def _standardize_channel(values: xr.DataArray) -> xr.DataArray:
    fixed = np.nan_to_num(values.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean = float(np.mean(fixed))
    std = float(np.std(fixed))
    scale = max(std, STANDARDIZATION_EPS)
    scaled = ((fixed - mean) / scale).astype(np.float32)
    spatial_coords = {name: coord for name, coord in values.coords.items() if name in values.dims}
    return xr.DataArray(
        scaled,
        dims=values.dims,
        coords=spatial_coords,
        attrs=values.attrs,
        name=values.name,
    )


def _print_channel_stats(variable_names: list[str], arrays: list[xr.DataArray]) -> None:
    print(f"channel names head: {variable_names[:5]}", flush=True)
    print(f"channel names tail: {variable_names[-5:]}", flush=True)
    for name, values in zip(variable_names[:4] + variable_names[-4:], arrays[:4] + arrays[-4:], strict=True):
        print(f"{name} shape: {tuple(int(size) for size in values.shape)}", flush=True)
        print(f"{name} min: {float(values.min().item())}", flush=True)
        print(f"{name} max: {float(values.max().item())}", flush=True)
        print(f"{name} mean: {float(values.mean().item())}", flush=True)
        print(f"{name} std: {float(values.std().item())}", flush=True)
