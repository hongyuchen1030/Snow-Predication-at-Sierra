#!/usr/bin/env python3
"""
Build the April 1 Sierra SWE anomaly target for the COBE2 SST--Sierra SWE LOD setup.
"""

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
        "COBE2_SIERRA_SWE_LOD_OUTPUT_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup",
    )
).expanduser()
TARGET_ROOT = OUTPUT_ROOT / "targets"
OUTPUT_NC = TARGET_ROOT / "sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
OUTPUT_JSON = TARGET_ROOT / "sierra_swe_apr1_anomaly_standardized_wy1985_2021_summary.json"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
MEAN_STAT_INDEX = 0
EARTH_RADIUS_M = 6_371_000.0
NETCDF_ENGINE = "netcdf4"


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def infer_coordinate_bounds(values: xr.DataArray) -> np.ndarray:
    centers = np.asarray(values.values, dtype=np.float64)
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
        },
    )


def build_weights() -> tuple[xr.DataArray, xr.DataArray, dict[str, object]]:
    swe_grid = get_regional_swe_grid_definition(
        water_year=WATER_YEAR_START,
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )
    mask = build_sierra_mask(swe_grid, region=DEFAULT_SIERRA_REGION).astype(np.float64)
    area = cell_area_from_bounds(swe_grid.latitude, swe_grid.longitude)
    weights = (mask * area).rename("sierra_weight_m2")
    positive_mask_values = mask.where(weights > 0.0, drop=True).values
    summary = {
        "sierra_region": asdict(DEFAULT_SIERRA_REGION),
        "mask_type_attr": str(mask.attrs.get("mask_type", "unknown")),
        "mask_has_fractional_values_on_processed_subset": bool(
            np.any((positive_mask_values > 0.0) & (positive_mask_values < 1.0))
        ),
        "number_of_nonzero_weight_cells": int((weights > 0.0).sum().item()),
        "total_effective_sierra_area_m2": float(weights.sum().item()),
    }
    return weights, area, summary


def apr1_scalar_for_water_year(water_year: int, weights: xr.DataArray) -> tuple[str, float]:
    path = swe_file_for_water_year(water_year)
    target_date = f"{water_year}-04-01"
    lat_name, lon_name = weights.dims
    weight_values = np.asarray(weights.values, dtype=np.float64)
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        swe = ds[SWE_VARIABLE].isel(Stats=MEAN_STAT_INDEX, drop=True)
        swe = swe.sel(time=np.datetime64(target_date))
        swe = swe.sel({lat_name: weights[lat_name], lon_name: weights[lon_name]})
        swe_values = np.asarray(swe.where(swe != SWE_MISSING_VALUE).values, dtype=np.float64)
    valid = np.isfinite(swe_values)
    numerator = np.where(valid, swe_values * weight_values, 0.0).sum()
    denominator = np.where(valid, weight_values, 0.0).sum()
    return target_date, float(numerator / denominator)


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def main() -> None:
    ensure_runtime_on_compute_node()
    start = perf_counter()
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    weights, area, weight_summary = build_weights()
    rows = []
    for water_year in range(WATER_YEAR_START, WATER_YEAR_END + 1):
        target_date, mean_m = apr1_scalar_for_water_year(water_year, weights)
        rows.append((water_year, target_date, mean_m))
        print(f"processed WY{water_year}: {mean_m:.6f} m", flush=True)

    water_years = np.asarray([row[0] for row in rows], dtype=np.int32)
    target_dates = np.asarray([row[1] for row in rows], dtype="U10")
    mean_m = np.asarray([row[2] for row in rows], dtype=np.float32)
    mean_mm = (mean_m * 1000.0).astype(np.float32)
    climatology_m = float(mean_m.mean())
    climatology_mm = float(mean_mm.mean())
    anom_m = (mean_m - climatology_m).astype(np.float32)
    anom_mm = (mean_mm - climatology_mm).astype(np.float32)
    sigma_m = float(anom_m.std(ddof=1))
    standardized = (anom_m / sigma_m).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "sierra_swe_apr1_mean_m": xr.DataArray(mean_m, dims=("water_year",), attrs={"units": "m"}),
            "sierra_swe_apr1_mean_mm": xr.DataArray(mean_mm, dims=("water_year",), attrs={"units": "mm"}),
            "sierra_swe_apr1_anom_m": xr.DataArray(anom_m, dims=("water_year",), attrs={"units": "m"}),
            "sierra_swe_apr1_anom_mm": xr.DataArray(anom_mm, dims=("water_year",), attrs={"units": "mm"}),
            "sierra_swe_apr1_standardized": xr.DataArray(standardized, dims=("water_year",), attrs={"units": "1"}),
            "target_date": xr.DataArray(target_dates, dims=("water_year",)),
        },
        coords={"water_year": xr.DataArray(water_years, dims=("water_year",))},
        attrs={
            "description": "April 1 Sierra SWE area-average, anomaly, and standardized anomaly for the COBE2 SST--Sierra SWE LOD setup.",
            "sierra_region": json.dumps(asdict(DEFAULT_SIERRA_REGION)),
            "mask_type_attr": str(weight_summary["mask_type_attr"]),
            "mask_has_fractional_values_on_processed_subset": str(weight_summary["mask_has_fractional_values_on_processed_subset"]).lower(),
            "grid_cell_area_formula": area.attrs["formula"],
            "equation_apr1_mean": "SWE_Sierra_Apr1(y) = sum_i[f_i A_i valid_i(y) SWE_i^{Apr1}(y)] / sum_i[f_i A_i valid_i(y)]",
            "equation_anomaly": "SWE_Sierra_Apr1'(y) = SWE_Sierra_Apr1(y) - mean_y[SWE_Sierra_Apr1(y)]",
            "equation_standardized": "SWE_tilde(y) = SWE_Sierra_Apr1'(y) / std_y[SWE_Sierra_Apr1'(y)] with ddof=1",
        },
    )
    ds.to_netcdf(OUTPUT_NC, engine=NETCDF_ENGINE)

    summary = {
        "raw_swe_files_used": [str(swe_file_for_water_year(wy)) for wy in water_years],
        "water_years_used": water_years.tolist(),
        "n_samples": int(water_years.size),
        "target_dates": target_dates.tolist(),
        "sierra_region": asdict(DEFAULT_SIERRA_REGION),
        "mask_type_attr": str(weight_summary["mask_type_attr"]),
        "mask_has_fractional_values_on_processed_subset": bool(weight_summary["mask_has_fractional_values_on_processed_subset"]),
        "number_of_nonzero_weight_cells": int(weight_summary["number_of_nonzero_weight_cells"]),
        "total_effective_sierra_area_m2": float(weight_summary["total_effective_sierra_area_m2"]),
        "grid_cell_area_formula": area.attrs["formula"],
        "apr1_climatology_m": climatology_m,
        "apr1_climatology_mm": climatology_mm,
        "apr1_anomaly_std_m_ddof1": sigma_m,
        "runtime_seconds": perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
        "output_path": str(OUTPUT_NC),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_nc": str(OUTPUT_NC), "output_json": str(OUTPUT_JSON)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
