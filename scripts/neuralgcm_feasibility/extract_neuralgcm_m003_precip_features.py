#!/usr/bin/env python3
"""Extract per-member regional precipitation features for NeuralGCM M=3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
OUTPUT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs/wy2021_actual_m003")
CSV_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_member_features.csv"
JSON_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_distribution_summary.json"

REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
    "CA_NV_BOX": {"lat_min": 32.0, "lat_max": 43.0, "lon_min_360": 234.0, "lon_max_360": 246.0},
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
}


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


def monthly_totals(series: xr.DataArray) -> dict[str, float]:
    out = {}
    for timestamp, value in zip(series.time.values, np.asarray(series.values, dtype=float), strict=True):
        month = str(np.datetime_as_string(timestamp, unit="M"))
        out[month] = float(out.get(month, 0.0) + value)
    return out


def main() -> None:
    rows = []
    summary = {
        "native_precipitation_units_caveat": "precipitation_cumulative_mean lacks explicit units in xarray attrs; totals are reported in native cumulative model units",
        "members": {},
        "ensemble": {},
    }

    for member_dir in sorted(OUTPUT_ROOT.glob("member_*")):
        output_file = member_dir / "predictions_6h.nc"
        with xr.open_dataset(output_file) as ds:
            precip = ds["precipitation_cumulative_mean"]
            if "surface" in precip.dims:
                precip = precip.isel(surface=0)
            member_summary = {}
            for region_name, bounds in REGIONS.items():
                region_precip = subset_region(precip, bounds)
                region_series = weighted_mean(region_precip)
                totals = monthly_totals(region_series)
                row = {
                    "member_id": member_dir.name,
                    "region": region_name,
                    "sep_mar_total_precip_native_units": float(np.nansum(list(totals.values()))),
                    "monthly_mean_native_units": float(np.nanmean(list(totals.values()))),
                    "monthly_std_native_units": float(np.nanstd(list(totals.values()))),
                }
                for month, value in totals.items():
                    row[f"{month}_precip_native_units"] = value
                rows.append(row)
                member_summary[region_name] = {
                    "sep_mar_total_precip_native_units": row["sep_mar_total_precip_native_units"],
                    "monthly_total_precip_native_units": totals,
                    "monthly_mean_native_units": row["monthly_mean_native_units"],
                    "monthly_std_native_units": row["monthly_std_native_units"],
                }
            summary["members"][member_dir.name] = member_summary

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    for region_name in REGIONS:
        region_df = df[df["region"] == region_name].copy()
        totals = region_df["sep_mar_total_precip_native_units"].astype(float)
        summary["ensemble"][region_name] = {
            "member_ids": region_df["member_id"].tolist(),
            "sep_mar_total_precip_native_units": totals.tolist(),
            "ensemble_mean_native_units": float(totals.mean()),
            "ensemble_std_native_units": float(totals.std(ddof=0)),
            "ensemble_min_native_units": float(totals.min()),
            "ensemble_max_native_units": float(totals.max()),
        }

    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
