#!/usr/bin/env python3
"""Compare pairwise NeuralGCM M=3 ensemble members."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
OUTPUT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs/wy2021_actual_m003")
CSV_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_pairwise_comparison.csv"
JSON_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_pairwise_comparison.json"

VARIABLES = [
    "precipitation_cumulative_mean",
    "temperature_850hPa",
    "v_component_of_wind_850hPa",
]

SIERRA_BOX = {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0}


def lon_convention(values: np.ndarray) -> str:
    return "0_to_360" if float(np.nanmin(values)) >= 0.0 else "minus180_to_180"


def convert_bounds(bounds: dict, convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"] - 360.0 if bounds["lon_min_360"] > 180.0 else bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"] - 360.0 if bounds["lon_max_360"] > 180.0 else bounds["lon_max_360"]
    return lon_min, lon_max


def subset_region(da: xr.DataArray, bounds: dict) -> xr.DataArray:
    lat_name, lon_name = "latitude", "longitude"
    lat_vals = np.asarray(da[lat_name].values)
    lon_vals = np.asarray(da[lon_name].values)
    lon_min, lon_max = convert_bounds(bounds, lon_convention(lon_vals))
    lat_slice = slice(bounds["lat_min"], bounds["lat_max"]) if lat_vals[0] <= lat_vals[-1] else slice(bounds["lat_max"], bounds["lat_min"])
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def weighted_mean(da: xr.DataArray) -> xr.DataArray:
    lat_name = "latitude"
    weights = xr.DataArray(np.cos(np.deg2rad(da[lat_name].values)), coords={lat_name: da[lat_name]}, dims=(lat_name,))
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    return da.weighted(weights).mean(dim=spatial_dims, skipna=True)


def sierra_total_precip(ds: xr.Dataset) -> float:
    precip = ds["precipitation_cumulative_mean"]
    if "surface" in precip.dims:
        precip = precip.isel(surface=0)
    series = weighted_mean(subset_region(precip, SIERRA_BOX))
    return float(np.nansum(np.asarray(series.values, dtype=float)))


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    exact_identical = bool(np.array_equal(a, b))
    numerically_identical = bool(np.allclose(a, b, atol=1e-12, rtol=0.0))
    diff = a - b
    return {
        "exact_identical": exact_identical,
        "numerically_identical_tolerance_1e-12": numerically_identical,
        "max_abs_difference": float(np.max(np.abs(diff))),
        "mean_abs_difference": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "correlation": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
    }


def main() -> None:
    member_paths = {member_dir.name: member_dir / "predictions_6h.nc" for member_dir in sorted(OUTPUT_ROOT.glob("member_*"))}
    rows = []
    report = {"pairs": {}}
    for left, right in itertools.combinations(sorted(member_paths), 2):
        with xr.open_dataset(member_paths[left]) as ds_left, xr.open_dataset(member_paths[right]) as ds_right:
            left_sierra = sierra_total_precip(ds_left)
            right_sierra = sierra_total_precip(ds_right)
            pair_report = {
                "left_member": left,
                "right_member": right,
                "sierra_total_precip_left_native_units": left_sierra,
                "sierra_total_precip_right_native_units": right_sierra,
                "sierra_total_precip_difference_native_units": left_sierra - right_sierra,
                "variables": {},
            }
            for variable in VARIABLES:
                a = np.asarray(ds_left[variable].values, dtype=float)
                b = np.asarray(ds_right[variable].values, dtype=float)
                variable_report = compare_arrays(a, b)
                pair_report["variables"][variable] = variable_report
                row = {
                    "left_member": left,
                    "right_member": right,
                    "variable": variable,
                    **variable_report,
                    "sierra_total_precip_left_native_units": left_sierra,
                    "sierra_total_precip_right_native_units": right_sierra,
                    "sierra_total_precip_difference_native_units": left_sierra - right_sierra,
                }
                rows.append(row)
            report["pairs"][f"{left}_vs_{right}"] = pair_report

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
