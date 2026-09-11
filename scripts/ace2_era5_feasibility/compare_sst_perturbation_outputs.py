#!/usr/bin/env python3
"""Compare ACE2 outputs from SST-perturbed forcing states against control."""

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
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
}
TARGET_VARS = ("PRATEsfc", "TMP2m", "VGRD10m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--perturbed", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def coord_names(ds: xr.Dataset) -> tuple[str, str]:
    return ("lat" if "lat" in ds.coords else "latitude", "lon" if "lon" in ds.coords else "longitude")


def convert_bounds(bounds: dict[str, float], lon_values: np.ndarray) -> tuple[float, float]:
    if float(np.nanmin(lon_values)) >= 0.0:
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


def compare_pair(control: xr.DataArray, perturbed: xr.DataArray) -> dict[str, Any]:
    diff = perturbed - control
    control_values = np.asarray(control.values, dtype=float).reshape(-1)
    pert_values = np.asarray(perturbed.values, dtype=float).reshape(-1)
    diff_values = np.asarray(diff.values, dtype=float).reshape(-1)
    finite = np.isfinite(control_values) & np.isfinite(pert_values)
    result = {
        "max_abs_difference": float(np.nanmax(np.abs(diff_values))),
        "mean_abs_difference": float(np.nanmean(np.abs(diff_values))),
        "rmse_difference": float(np.sqrt(np.nanmean(diff_values**2))),
        "mean_signed_difference": float(np.nanmean(diff_values)),
        "exactly_identical": bool(np.array_equal(control.values, perturbed.values)),
    }
    if finite.sum() >= 2:
        result["correlation"] = float(np.corrcoef(control_values[finite], pert_values[finite])[0, 1])
    else:
        result["correlation"] = None
    return result


def regional_differences(control: xr.Dataset, perturbed: xr.Dataset, var_name: str) -> dict[str, Any]:
    lat_name, lon_name = coord_names(control)
    lon_values = np.asarray(control[lon_name].values)
    out: dict[str, Any] = {}
    for region_name, bounds in REGIONS.items():
        lon_min, lon_max = convert_bounds(bounds, lon_values)
        c_sub = control[var_name].sel({lat_name: slice(bounds["lat_min"], bounds["lat_max"]), lon_name: slice(lon_min, lon_max)})
        p_sub = perturbed[var_name].sel({lat_name: slice(bounds["lat_min"], bounds["lat_max"]), lon_name: slice(lon_min, lon_max)})
        c_mean = weighted_region_mean(c_sub, lat_name)
        p_mean = weighted_region_mean(p_sub, lat_name)
        diff = p_mean - c_mean
        out[region_name] = {
            "control_mean_over_time": float(np.nanmean(c_mean.values)),
            "perturbed_mean_over_time": float(np.nanmean(p_mean.values)),
            "mean_signed_difference_over_time": float(np.nanmean(diff.values)),
            "max_abs_difference_over_time": float(np.nanmax(np.abs(diff.values))),
        }
    return out


def main() -> None:
    args = parse_args()
    if len(args.perturbed) != len(args.labels):
        raise ValueError("--perturbed and --labels must have the same length.")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "control_path": str(args.control),
        "comparisons": [],
    }

    with xr.open_dataset(args.control) as control_ds:
        for label, path in zip(args.labels, args.perturbed):
            with xr.open_dataset(path) as perturbed_ds:
                comparison: dict[str, Any] = {"label": label, "path": str(path), "variables": {}}
                for var_name in TARGET_VARS:
                    if var_name not in control_ds.data_vars or var_name not in perturbed_ds.data_vars:
                        continue
                    metrics = compare_pair(control_ds[var_name], perturbed_ds[var_name])
                    regional = regional_differences(control_ds, perturbed_ds, var_name)
                    comparison["variables"][var_name] = {"pairwise": metrics, "regional": regional}
                    row = {
                        "label": label,
                        "variable": var_name,
                        **metrics,
                    }
                    for region_name, region_metrics in regional.items():
                        for key, value in region_metrics.items():
                            row[f"{region_name}__{key}"] = value
                    rows.append(row)
                details["comparisons"].append(comparison)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_csv(index=False).strip())
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
