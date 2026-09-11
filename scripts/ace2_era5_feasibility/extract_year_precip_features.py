#!/usr/bin/env python3
"""Extract provisional regional precipitation features from ACE2 output."""

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

TARGET_VARS = ("PRATEsfc", "TMP2m", "VGRD10m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--year-label", required=True, help="Label such as free_year_2020")
    parser.add_argument("--member-id", default="m001", help="Member identifier for reporting")
    return parser.parse_args()


def coord_names(ds: xr.Dataset) -> tuple[str, str]:
    return ("lat" if "lat" in ds.coords else "latitude", "lon" if "lon" in ds.coords else "longitude")


def lon_convention(values: np.ndarray) -> str:
    return "0_to_360" if float(np.nanmin(values)) >= 0.0 else "minus180_to_180"


def convert_bounds(bounds: dict[str, float], convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"] - 360.0 if bounds["lon_min_360"] > 180.0 else bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"] - 360.0 if bounds["lon_max_360"] > 180.0 else bounds["lon_max_360"]
    return lon_min, lon_max


def weighted_region_mean(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da[lat_name].values)),
        coords={lat_name: da[lat_name]},
        dims=(lat_name,),
    )
    spatial_dims = [d for d in da.dims if d not in ("sample", "time")]
    return da.weighted(lat_weights).mean(dim=spatial_dims, skipna=True)


def time_labels(ds: xr.Dataset) -> list[str]:
    if "valid_time" in ds.coords:
        return [str(v)[:7] for v in np.ravel(ds["valid_time"].values)]
    if "time" in ds.coords:
        return [str(v) for v in np.ravel(ds["time"].values)]
    return [f"index_{i:03d}" for i in range(int(ds.sizes.get("time", 0)))]


def precip_conversion_info(ds: xr.Dataset) -> dict[str, Any]:
    info = {
        "units": None,
        "conversion_reliable": False,
        "seconds_per_interval": None,
        "warning": None,
    }
    if "PRATEsfc" not in ds.data_vars:
        info["warning"] = "PRATEsfc not present."
        return info
    units = str(ds["PRATEsfc"].attrs.get("units", ""))
    info["units"] = units
    if units.lower() not in {"kg/m**2/s", "kg/m^2/s", "kg m-2 s-1", "mm/s", "m/s"}:
        info["warning"] = "PRATEsfc units are not clearly a precipitation rate."
        return info
    if "valid_time" not in ds.coords:
        info["warning"] = "No valid_time coordinate available."
        return info
    valid = np.ravel(ds["valid_time"].values)
    if valid.size < 2:
        info["warning"] = "Only one time slice is present, so period totals are not reliable."
        return info
    deltas = np.diff(valid).astype("timedelta64[s]").astype(np.int64)
    if len(deltas) == 0 or np.any(deltas <= 0) or len(set(deltas.tolist())) != 1:
        info["warning"] = "Time intervals are absent or not uniform."
        return info
    info["conversion_reliable"] = True
    info["seconds_per_interval"] = int(deltas[0])
    return info


def main() -> None:
    args = parse_args()
    with xr.open_dataset(args.input) as ds:
        lat_name, lon_name = coord_names(ds)
        labels = time_labels(ds)
        convention = lon_convention(np.asarray(ds[lon_name].values))
        precip_info = precip_conversion_info(ds)

        row: dict[str, Any] = {
            "dataset_label": args.year_label,
            "member_id": args.member_id,
            "input_path": str(args.input),
            "longitude_convention": convention,
            "time_label_start": labels[0] if labels else "unknown",
            "time_label_end": labels[-1] if labels else "unknown",
        }
        summary: dict[str, Any] = {
            "dataset_label": args.year_label,
            "member_id": args.member_id,
            "input_path": str(args.input),
            "longitude_convention": convention,
            "time_labels": labels,
            "precipitation_interpretation": precip_info,
            "regions": {},
        }

        for region_name, bounds in REGIONS.items():
            lon_min, lon_max = convert_bounds(bounds, convention)
            subset = ds.sel({lat_name: slice(bounds["lat_min"], bounds["lat_max"]), lon_name: slice(lon_min, lon_max)})
            lat_count = int(subset.sizes.get(lat_name, 0))
            lon_count = int(subset.sizes.get(lon_name, 0))
            grid_cell_count = lat_count * lon_count
            if grid_cell_count == 0:
                raise ValueError(f"{region_name} selected zero grid cells.")
            region_summary: dict[str, Any] = {
                "grid_cell_count": grid_cell_count,
                "lat_count": lat_count,
                "lon_count": lon_count,
                "bounds_requested": bounds,
                "bounds_used": {
                    "lat_min": bounds["lat_min"],
                    "lat_max": bounds["lat_max"],
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                },
                "variables": {},
            }

            for var_name in TARGET_VARS:
                if var_name not in subset.data_vars:
                    continue
                series = weighted_region_mean(subset[var_name], lat_name=lat_name)
                values = np.asarray(series.values, dtype=float).reshape(-1)
                var_summary: dict[str, Any] = {
                    "units": subset[var_name].attrs.get("units"),
                    "mean_over_available_time": float(np.nanmean(values)),
                    "first_value": float(values[0]),
                    "last_value": float(values[-1]),
                    "min_over_time": float(np.nanmin(values)),
                    "max_over_time": float(np.nanmax(values)),
                    "monthly_values": {label: float(values[idx]) for idx, label in enumerate(labels)},
                }
                if var_name == "PRATEsfc" and precip_info["conversion_reliable"]:
                    seconds = float(precip_info["seconds_per_interval"])
                    var_summary["approx_total_per_interval"] = {
                        label: float(values[idx] * seconds) for idx, label in enumerate(labels)
                    }
                    var_summary["mean_approx_total_per_interval"] = float(np.nanmean(values) * seconds)
                region_summary["variables"][var_name] = var_summary

                prefix = f"{region_name}__{var_name}"
                row[f"{prefix}__mean_over_available_time"] = var_summary["mean_over_available_time"]
                row[f"{prefix}__first"] = var_summary["first_value"]
                row[f"{prefix}__last"] = var_summary["last_value"]
                row[f"{prefix}__min"] = var_summary["min_over_time"]
                row[f"{prefix}__max"] = var_summary["max_over_time"]
                row[f"{prefix}__units"] = subset[var_name].attrs.get("units", "")
                for label, value in var_summary["monthly_values"].items():
                    row[f"{prefix}__{label}"] = value

            summary["regions"][region_name] = region_summary

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame([row]).to_csv(index=False).strip())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
