#!/usr/bin/env python3
"""
Efficient daily-first Sierra SWE area-average preprocessing.
"""

import argparse
import json
import os
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import (  # noqa: E402
    DEFAULT_SIERRA_REGION,
    SWE_MISSING_VALUE,
    SWE_VARIABLE,
    build_sierra_mask,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)


OUTPUT_ROOT = Path(
    os.environ.get(
        "ERA5_SIERRA_SWE_LOD_OUTPUT_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_preprocessing",
    )
).expanduser()
SWE_ROOT = OUTPUT_ROOT / "swe_targets"
INVALID_ROOT = SWE_ROOT / "invalid_apr1_target"
INVALID_NC = SWE_ROOT / "sierra_swe_apr1_area_average_wy1985_2018.nc"
INVALID_JSON = SWE_ROOT / "sierra_swe_apr1_area_average_wy1985_2018_summary.json"
OUTPUT_NC = SWE_ROOT / "sierra_swe_monthly_area_average_anomaly_wy1985_2021.nc"
OUTPUT_JSON = SWE_ROOT / "sierra_swe_monthly_area_average_anomaly_wy1985_2021_summary.json"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
MEAN_STAT_INDEX = 0
EARTH_RADIUS_M = 6_371_000.0
NETCDF_ENGINE = "netcdf4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-water-year", type=int, default=None)
    parser.add_argument("--water-year-start", type=int, default=WATER_YEAR_START)
    parser.add_argument("--water-year-end", type=int, default=WATER_YEAR_END)
    parser.add_argument("--write-output", action="store_true")
    return parser.parse_args()


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def infer_coordinate_bounds(values: xr.DataArray) -> np.ndarray:
    centers = np.asarray(values.values, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Expected a 1D coordinate with at least two points.")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    bounds = np.empty(centers.size + 1, dtype=np.float64)
    bounds[1:-1] = midpoints
    bounds[0] = centers[0] - (midpoints[0] - centers[0])
    bounds[-1] = centers[-1] + (centers[-1] - midpoints[-1])
    return bounds


def cell_area_from_bounds(latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
    lat_bounds_deg = infer_coordinate_bounds(latitude)
    lon_bounds_deg = infer_coordinate_bounds(longitude)
    lat_bounds_rad = np.deg2rad(lat_bounds_deg)
    lon_bounds_rad = np.deg2rad(lon_bounds_deg)
    lat_band = np.abs(np.sin(lat_bounds_rad[1:]) - np.sin(lat_bounds_rad[:-1]))
    lon_band = np.abs(lon_bounds_rad[1:] - lon_bounds_rad[:-1])
    area = (EARTH_RADIUS_M**2) * lat_band[:, None] * lon_band[None, :]
    return xr.DataArray(
        area.astype(np.float64),
        dims=(latitude.dims[0], longitude.dims[0]),
        coords={latitude.dims[0]: latitude, longitude.dims[0]: longitude},
        name="grid_cell_area_m2",
        attrs={
            "units": "m2",
            "formula": (
                "A_i = R^2 * |sin(lat_north) - sin(lat_south)| * |lon_east - lon_west|; "
                "coordinate bounds inferred from adjacent cell centers."
            ),
            "earth_radius_m": EARTH_RADIUS_M,
        },
    )


def archive_invalid_apr1_outputs() -> dict[str, str]:
    SWE_ROOT.mkdir(parents=True, exist_ok=True)
    INVALID_ROOT.mkdir(parents=True, exist_ok=True)
    moved = {}
    for source in (INVALID_NC, INVALID_JSON):
        if source.exists():
            destination = INVALID_ROOT / source.name
            source.replace(destination)
            moved[str(source)] = str(destination)
    readme_path = INVALID_ROOT / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Invalid Apr 1 SWE target\n\n"
            "These archived files were created from an Apr 1 framing that is invalid for the current monthly LOD workflow.\n"
            "They are retained only for provenance and should not be used as active SWE targets.\n",
            encoding="utf-8",
        )
    moved["README"] = str(readme_path)
    return moved


def build_weights() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, dict[str, object]]:
    swe_grid = get_regional_swe_grid_definition(
        water_year=WATER_YEAR_START,
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )
    mask = build_sierra_mask(swe_grid, region=DEFAULT_SIERRA_REGION).astype(np.float64)
    area = cell_area_from_bounds(swe_grid.latitude, swe_grid.longitude)
    weights = (mask * area).rename("sierra_weight_m2")
    nonzero = weights > 0.0
    positive_mask_values = mask.where(nonzero, drop=True).values
    summary = {
        "sierra_region": asdict(DEFAULT_SIERRA_REGION),
        "mask_type_attr": str(mask.attrs.get("mask_type", "unknown")),
        "mask_has_fractional_values": bool(
            np.any((positive_mask_values > 0.0) & (positive_mask_values < 1.0))
        ),
        "nonzero_weight_cell_count": int(nonzero.sum().item()),
        "total_effective_sierra_area_m2": float(weights.sum().item()),
        "grid_shape": list(weights.shape),
        "latitude_name": swe_grid.latitude_name,
        "longitude_name": swe_grid.longitude_name,
    }
    return mask, area, weights, summary


def reduce_water_year_to_daily_scalar(water_year: int, weights: xr.DataArray) -> xr.DataArray:
    path = swe_file_for_water_year(water_year)
    lat_name, lon_name = weights.dims
    weight_values = np.asarray(weights.values, dtype=np.float64)
    wy_start = perf_counter()
    print(f"processing water year {water_year}: {path}", flush=True)
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        swe = ds[SWE_VARIABLE].isel(Stats=MEAN_STAT_INDEX, drop=True)
        swe = swe.sel({lat_name: weights[lat_name], lon_name: weights[lon_name]})
        swe = swe.where(swe != SWE_MISSING_VALUE).load()
        swe_values = np.asarray(swe.values, dtype=np.float64)
        valid = np.isfinite(swe_values)
        numerator = np.where(valid, swe_values * weight_values[None, :, :], 0.0).sum(axis=(1, 2))
        denominator = np.where(valid, weight_values[None, :, :], 0.0).sum(axis=(1, 2))
        daily_values = (numerator / denominator).astype(np.float32)
        daily = xr.DataArray(
            daily_values,
            dims=("time",),
            coords={"time": swe["time"].values},
            name="sierra_swe_daily_mean_m",
            attrs={
                "units": "m",
                "equation": (
            "SWE_Sierra_daily(t) = sum_i[f_i * A_i * valid_i(t) * SWE_i(t)] / "
            "sum_i[f_i * A_i * valid_i(t)]"
                ),
            },
        )
        print(
            f"completed water year {water_year}: days={daily.sizes['time']} elapsed_seconds={perf_counter() - wy_start:.2f}",
            flush=True,
        )
        return daily


def build_outputs(daily: xr.DataArray) -> xr.Dataset:
    monthly = daily.resample(time="MS").mean().astype(np.float32).rename("sierra_swe_monthly_mean_m")
    monthly.attrs["units"] = "m"
    climatology = monthly.groupby("time.month").mean(dim="time", skipna=True).astype(np.float32)
    climatology = climatology.rename("sierra_swe_monthly_climatology_m")
    climatology.attrs["units"] = "m"
    anomaly = (monthly.groupby("time.month") - climatology).astype(np.float32).rename("sierra_swe_monthly_anom_m")
    anomaly.attrs["units"] = "m"
    monthly_mm = (monthly * 1000.0).astype(np.float32).rename("sierra_swe_monthly_mean_mm")
    monthly_mm.attrs["units"] = "mm"
    climatology_mm = (climatology * 1000.0).astype(np.float32).rename("sierra_swe_monthly_climatology_mm")
    climatology_mm.attrs["units"] = "mm"
    anomaly_mm = (anomaly * 1000.0).astype(np.float32).rename("sierra_swe_monthly_anom_mm")
    anomaly_mm.attrs["units"] = "mm"
    daily_named = daily.rename({"time": "daily_time"}).rename("sierra_swe_daily_mean_m")
    return xr.Dataset(
        data_vars={
            "sierra_swe_daily_mean_m": daily_named,
            "sierra_swe_monthly_mean_m": monthly,
            "sierra_swe_monthly_mean_mm": monthly_mm,
            "sierra_swe_monthly_anom_m": anomaly,
            "sierra_swe_monthly_anom_mm": anomaly_mm,
            "sierra_swe_monthly_climatology_m": climatology,
            "sierra_swe_monthly_climatology_mm": climatology_mm,
        }
    )


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def smoke_stats(daily: xr.DataArray, monthly: xr.DataArray, weight_summary: dict[str, object]) -> dict[str, object]:
    return {
        "nonzero_weight_cells": int(weight_summary["nonzero_weight_cell_count"]),
        "daily_shape": list(daily.shape),
        "monthly_shape": list(monthly.shape),
        "daily_min_m": float(daily.min().item()),
        "daily_max_m": float(daily.max().item()),
        "daily_mean_m": float(daily.mean().item()),
        "monthly_min_m": float(monthly.min().item()),
        "monthly_max_m": float(monthly.max().item()),
        "monthly_mean_m": float(monthly.mean().item()),
    }


def main() -> None:
    args = parse_args()
    ensure_runtime_on_compute_node()
    start = perf_counter()
    archived = archive_invalid_apr1_outputs()
    mask, area, weights, weight_summary = build_weights()
    water_years = (
        [args.smoke_water_year]
        if args.smoke_water_year is not None
        else list(range(args.water_year_start, args.water_year_end + 1))
    )
    daily_series = [reduce_water_year_to_daily_scalar(wy, weights) for wy in water_years]
    daily = xr.concat(daily_series, dim="time").sortby("time").astype(np.float32).rename("sierra_swe_daily_mean_m")
    dataset = build_outputs(daily)
    runtime_seconds = perf_counter() - start
    summary = {
        "raw_files_used": [str(swe_file_for_water_year(wy)) for wy in water_years],
        "water_years_used": water_years,
        "number_of_daily_samples": int(dataset["sierra_swe_daily_mean_m"].sizes["daily_time"]),
        "number_of_monthly_samples": int(dataset["sierra_swe_monthly_mean_m"].sizes["time"]),
        "mask_type_attr": str(weight_summary["mask_type_attr"]),
        "mask_is_fractional": bool(weight_summary["mask_has_fractional_values"]),
        "mask_has_fractional_values_on_processed_subset": bool(weight_summary["mask_has_fractional_values"]),
        "number_of_nonzero_weight_cells": int(weight_summary["nonzero_weight_cell_count"]),
        "total_effective_sierra_area_m2": float(weight_summary["total_effective_sierra_area_m2"]),
        "grid_cell_area_formula": area.attrs["formula"],
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb(),
        "output_path": str(OUTPUT_NC),
        "archived_invalid_apr1_outputs": archived,
        "sierra_region": asdict(DEFAULT_SIERRA_REGION),
    }
    smoke = smoke_stats(
        dataset["sierra_swe_daily_mean_m"],
        dataset["sierra_swe_monthly_mean_m"],
        weight_summary,
    )
    print(json.dumps({"mode": "smoke" if args.smoke_water_year else "full", **smoke}, indent=2), flush=True)
    if args.write_output:
        SWE_ROOT.mkdir(parents=True, exist_ok=True)
        dataset.attrs.update(
            {
                "description": "Daily-first Sierra SWE area-average and monthly anomaly product.",
                "sierra_region": json.dumps(asdict(DEFAULT_SIERRA_REGION)),
                "mask_utility": "snow_ml.data.build_sierra_mask",
                "mask_type_attr": str(mask.attrs.get("mask_type", "unknown")),
                "mask_has_fractional_values": str(weight_summary["mask_has_fractional_values"]).lower(),
                "grid_cell_area_formula": area.attrs["formula"],
                "equation_daily": "SWE_Sierra_daily(t) = sum_i[f_i * A_i * valid_i(t) * SWE_i(t)] / sum_i[f_i * A_i * valid_i(t)]",
                "equation_monthly": "SWE_monthly(y,m) = mean_daily_values_in_month[SWE_Sierra_daily(t)]",
                "equation_anomaly": "SWE_anom(y,m) = SWE_monthly(y,m) - clim_SWE(m)",
            }
        )
        dataset.to_netcdf(OUTPUT_NC, engine=NETCDF_ENGINE)
        OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output_nc": str(OUTPUT_NC), "output_json": str(OUTPUT_JSON)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
