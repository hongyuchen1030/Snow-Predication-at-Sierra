#!/usr/bin/env python3
"""Extract simple regional smoke-test features from ACE2 monthly mean predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
    "CA_NV_BOX": {"lat_min": 32.0, "lat_max": 43.0, "lon_min_360": 234.0, "lon_max_360": 246.0},
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
    "NORTH_PACIFIC_BOX": {"lat_min": 20.0, "lat_max": 60.0, "lon_min_360": 140.0, "lon_max_360": 240.0},
}

TARGET_VARS = ("TMP2m", "PRATEsfc", "VGRD10m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to monthly_mean_predictions.nc")
    parser.add_argument("--member-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--summary-json", type=Path, required=True, help="Output JSON summary path")
    return parser.parse_args()


def detect_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    return lat_name, lon_name


def longitude_convention(lon_values: np.ndarray) -> str:
    if float(np.nanmin(lon_values)) >= 0.0:
        return "0_to_360"
    return "minus180_to_180"


def convert_region_bounds(bounds: dict[str, float], convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"]
    if lon_min > 180.0:
        lon_min -= 360.0
    if lon_max > 180.0:
        lon_max -= 360.0
    return lon_min, lon_max


def area_weighted_mean(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    lat_rad = np.deg2rad(da[lat_name])
    weights = xr.DataArray(np.cos(lat_rad), coords={lat_name: da[lat_name]}, dims=(lat_name,))
    mean_dims = [d for d in da.dims if d not in ("sample", "time")]
    return da.weighted(weights).mean(dim=mean_dims, skipna=True)


def infer_precip_conversion(ds: xr.Dataset) -> dict[str, Any]:
    result = {
        "units": None,
        "classification": "missing",
        "conversion_reliable": False,
        "warning": None,
        "seconds_per_interval": None,
    }
    if "PRATEsfc" not in ds.data_vars:
        result["warning"] = "PRATEsfc is not present in the smoke-test output."
        return result

    attrs = dict(ds["PRATEsfc"].attrs)
    units = str(attrs.get("units", ""))
    result["units"] = units
    if units.lower() in {"kg/m**2/s", "kg/m^2/s", "kg m-2 s-1", "mm/s", "m/s"}:
        result["classification"] = "precipitation_rate"
    else:
        result["classification"] = "unclear"
        result["warning"] = "PRATEsfc units do not clearly indicate an instantaneous precipitation rate."
        return result

    counts = ds.coords.get("counts")
    time_attrs = dict(ds["time"].attrs) if "time" in ds.coords else {}
    time_units = str(time_attrs.get("units", "")).strip().lower()
    if counts is not None and "month" in time_units:
        result["warning"] = (
            "PRATEsfc units indicate a rate, but monthly_mean_predictions.nc only stores a monthly-style time "
            "coordinate plus counts. The exact averaging interval is not reliable from this file alone."
        )
        return result

    result["warning"] = "Could not infer a trustworthy seconds-per-interval value from the smoke-test file alone."
    return result


def build_time_labels(ds: xr.Dataset) -> tuple[str, str]:
    init_time = "unknown"
    valid_time = "unknown"
    if "init_time" in ds.coords:
        init_values = np.ravel(ds["init_time"].values)
        if init_values.size:
            init_time = str(init_values[0])
    if "valid_time" in ds.coords:
        valid_values = np.ravel(ds["valid_time"].values)
        if valid_values.size:
            valid_time = str(valid_values[-1])
    return init_time, valid_time


def main() -> None:
    args = parse_args()
    with xr.open_dataset(args.input) as ds:
        lat_name, lon_name = detect_coord_names(ds)
        lon_values = np.asarray(ds[lon_name].values)
        convention = longitude_convention(lon_values)
        init_time, valid_time = build_time_labels(ds)
        precip_info = infer_precip_conversion(ds)

        feature_row: dict[str, Any] = {
            "dataset_label": "smoke_test",
            "input_path": str(args.input),
            "time_start": init_time,
            "time_end": valid_time,
            "longitude_convention": convention,
        }
        summary: dict[str, Any] = {
            "input_path": str(args.input),
            "variables": [name for name in TARGET_VARS if name in ds.data_vars],
            "longitude_convention": convention,
            "regions": {},
            "precipitation_interpretation": precip_info,
        }

        for region_name, bounds in REGIONS.items():
            lon_min, lon_max = convert_region_bounds(bounds, convention)
            subset = ds.sel({lat_name: slice(bounds["lat_min"], bounds["lat_max"]), lon_name: slice(lon_min, lon_max)})
            lat_count = int(subset.sizes.get(lat_name, 0))
            lon_count = int(subset.sizes.get(lon_name, 0))
            cell_count = lat_count * lon_count
            if cell_count == 0:
                raise ValueError(f"{region_name} selected zero grid cells.")

            region_summary: dict[str, Any] = {
                "bounds_requested": bounds,
                "bounds_used": {
                    "lat_min": bounds["lat_min"],
                    "lat_max": bounds["lat_max"],
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                },
                "grid_cell_count": cell_count,
                "lat_count": lat_count,
                "lon_count": lon_count,
                "variables": {},
            }

            for var_name in TARGET_VARS:
                if var_name not in subset.data_vars:
                    continue
                regional_mean = area_weighted_mean(subset[var_name], lat_name=lat_name)
                values = np.asarray(regional_mean.values, dtype=float).reshape(-1)
                var_summary = {
                    "units": subset[var_name].attrs.get("units"),
                    "mean_over_available_time": float(np.nanmean(values)),
                    "first_available_time_value": float(values[0]),
                    "last_available_time_value": float(values[-1]),
                    "min_over_time": float(np.nanmin(values)),
                    "max_over_time": float(np.nanmax(values)),
                }
                region_summary["variables"][var_name] = var_summary

                prefix = f"{region_name}__{var_name}"
                feature_row[f"{prefix}__mean"] = var_summary["mean_over_available_time"]
                feature_row[f"{prefix}__first"] = var_summary["first_available_time_value"]
                feature_row[f"{prefix}__last"] = var_summary["last_available_time_value"]
                feature_row[f"{prefix}__min"] = var_summary["min_over_time"]
                feature_row[f"{prefix}__max"] = var_summary["max_over_time"]

                if var_name == "PRATEsfc":
                    feature_row[f"{prefix}__units"] = subset[var_name].attrs.get("units", "")
                    if precip_info["conversion_reliable"] and precip_info["seconds_per_interval"] is not None:
                        total = var_summary["mean_over_available_time"] * float(precip_info["seconds_per_interval"])
                        region_summary["variables"][var_name]["approx_total_over_interval"] = total
                        feature_row[f"{prefix}__approx_total"] = total

            summary["regions"][region_name] = region_summary

    args.member_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([feature_row]).to_csv(args.member_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame([feature_row]).to_csv(index=False).strip())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
