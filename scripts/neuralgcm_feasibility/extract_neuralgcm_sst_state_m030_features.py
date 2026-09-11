#!/usr/bin/env python3
"""Extract regional precipitation features for actual vs reversed NeuralGCM ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
OUTPUT_BASE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs")
REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
    "CA_NV_BOX": {"lat_min": 32.0, "lat_max": 43.0, "lon_min_360": 234.0, "lon_max_360": 246.0},
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
}
STATES = ("actual", "reversed_pacific")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-year", type=int, required=True)
    return parser.parse_args()


def lon_convention(values: np.ndarray) -> str:
    return "0_to_360" if float(np.nanmin(values)) >= 0.0 else "minus180_to_180"


def convert_bounds(bounds: dict, convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"] - 360.0 if bounds["lon_min_360"] > 180.0 else bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"] - 360.0 if bounds["lon_max_360"] > 180.0 else bounds["lon_max_360"]
    return lon_min, lon_max


def subset_region(da: xr.DataArray, bounds: dict) -> xr.DataArray:
    lat_vals = np.asarray(da["latitude"].values)
    lon_vals = np.asarray(da["longitude"].values)
    lon_min, lon_max = convert_bounds(bounds, lon_convention(lon_vals))
    lat_slice = slice(bounds["lat_min"], bounds["lat_max"]) if lat_vals[0] <= lat_vals[-1] else slice(bounds["lat_max"], bounds["lat_min"])
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
    return da.sel(latitude=lat_slice, longitude=lon_slice)


def weighted_mean(da: xr.DataArray) -> xr.DataArray:
    weights = xr.DataArray(np.cos(np.deg2rad(da["latitude"].values)), coords={"latitude": da["latitude"]}, dims=("latitude",))
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    return da.weighted(weights).mean(dim=spatial_dims, skipna=True)


def cumulative_depth_to_monthly_mm(series: xr.DataArray) -> tuple[dict[str, float], float, float, float]:
    ordered = series.sortby("time")
    values_m = np.asarray(ordered.values, dtype=float)
    times = ordered.time.values
    if values_m.size == 0:
        raise ValueError("Empty time series for cumulative precipitation.")

    month_labels = np.asarray([str(np.datetime_as_string(t, unit="M")) for t in times])
    monthly_totals_mm: dict[str, float] = {}
    previous_cumulative_m = float(values_m[0])
    for month in dict.fromkeys(month_labels):
        month_mask = month_labels == month
        month_end_cumulative_m = float(values_m[np.where(month_mask)[0][-1]])
        monthly_totals_mm[month] = 1000.0 * (month_end_cumulative_m - previous_cumulative_m)
        previous_cumulative_m = month_end_cumulative_m

    initial_m = float(values_m[0])
    final_m = float(values_m[-1])
    sep_mar_total_mm = 1000.0 * (final_m - initial_m)
    return monthly_totals_mm, sep_mar_total_mm, initial_m, final_m


def main() -> None:
    args = parse_args()
    csv_out = REPORT_DIR / f"neuralgcm_wy{args.water_year}_sst_state_m030_member_features.csv"
    json_out = REPORT_DIR / f"neuralgcm_wy{args.water_year}_sst_state_m030_distribution_summary.json"

    rows = []
    summary = {
        "water_year": args.water_year,
        "precipitation_interpretation": "precipitation_cumulative_mean is treated as cumulative precipitation depth in meters; reported totals are converted to mm using cumulative differences",
        "states": {},
    }

    for state in STATES:
        state_root = OUTPUT_BASE / f"wy{args.water_year}_{state}_m030"
        state_summary = {}
        for member_dir in sorted(state_root.glob("member_*")):
            output_file = member_dir / "predictions_6h.nc"
            if not output_file.exists():
                continue
            with xr.open_dataset(output_file) as ds:
                precip = ds["precipitation_cumulative_mean"]
                if "surface" in precip.dims:
                    precip = precip.isel(surface=0)
                member_regions = {}
                for region_name, bounds in REGIONS.items():
                    region_precip = subset_region(precip, bounds)
                    region_series = weighted_mean(region_precip)
                    totals_mm, sep_mar_total_mm, initial_m, final_m = cumulative_depth_to_monthly_mm(region_series)
                    row = {
                        "water_year": args.water_year,
                        "state": state,
                        "member_id": member_dir.name,
                        "rng_key": int(member_dir.name.split("_")[-1]),
                        "region": region_name,
                        "precipitation_units": "mm",
                        "cumulative_initial_precip_m": initial_m,
                        "cumulative_final_precip_m": final_m,
                        "sep_mar_total_precip_mm": sep_mar_total_mm,
                        "monthly_mean_mm": float(np.nanmean(list(totals_mm.values()))),
                        "monthly_std_mm": float(np.nanstd(list(totals_mm.values()), ddof=0)),
                    }
                    for month, value in totals_mm.items():
                        row[f"{month}_precip_mm"] = value
                    rows.append(row)
                    member_regions[region_name] = row
                state_summary[member_dir.name] = member_regions
        summary["states"][state] = state_summary

    df = pd.DataFrame(rows).sort_values(["state", "member_id", "region"]).reset_index(drop=True)
    df.to_csv(csv_out, index=False)

    ensemble_summary = {}
    for state in STATES:
        state_df = df[df["state"] == state]
        ensemble_summary[state] = {}
        for region_name in REGIONS:
            region_df = state_df[state_df["region"] == region_name]
            totals_mm = region_df["sep_mar_total_precip_mm"].astype(float)
            ensemble_summary[state][region_name] = {
                "n_completed": int(len(region_df)),
                "member_ids": region_df["member_id"].tolist(),
                "rng_keys": region_df["rng_key"].astype(int).tolist(),
                "mean_mm": float(totals_mm.mean()) if len(region_df) else None,
                "std_mm": float(totals_mm.std(ddof=0)) if len(region_df) else None,
                "min_mm": float(totals_mm.min()) if len(region_df) else None,
                "max_mm": float(totals_mm.max()) if len(region_df) else None,
            }
    summary["ensemble"] = ensemble_summary
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
