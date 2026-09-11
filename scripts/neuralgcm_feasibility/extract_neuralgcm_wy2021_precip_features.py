#!/usr/bin/env python3
"""Extract regional precipitation and basic climate features from NeuralGCM output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
DEFAULT_OUTPUT_FILE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs/wy2021_actual_m001/predictions_6h.nc")
FEATURE_CSV = REPORT_DIR / "neuralgcm_wy2021_actual_m001_precip_features.csv"
FEATURE_JSON = REPORT_DIR / "neuralgcm_wy2021_actual_m001_precip_summary.json"
PRISM_CSV = REPORT_DIR / "neuralgcm_wy2021_actual_vs_prism.csv"
PRISM_JSON = REPORT_DIR / "neuralgcm_wy2021_actual_vs_prism.json"

REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
    "CA_NV_BOX": {"lat_min": 32.0, "lat_max": 43.0, "lon_min_360": 234.0, "lon_max_360": 246.0},
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
    "PACIFIC_DOMAIN": {"lat_min": -10.0, "lat_max": 60.0, "lon_min_360": 120.0, "lon_max_360": 280.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--prism-2020", type=Path, default=Path("/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_2020.nc"))
    parser.add_argument("--prism-2021", type=Path, default=Path("/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_2021.nc"))
    return parser.parse_args()


def detect_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    return lat_name, lon_name


def lon_convention(values: np.ndarray) -> str:
    return "0_to_360" if float(np.nanmin(values)) >= 0.0 else "minus180_to_180"


def convert_bounds(bounds: dict, convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"] - 360.0 if bounds["lon_min_360"] > 180.0 else bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"] - 360.0 if bounds["lon_max_360"] > 180.0 else bounds["lon_max_360"]
    return lon_min, lon_max


def subset_region(da: xr.DataArray, lat_name: str, lon_name: str, bounds: dict) -> xr.DataArray:
    lat_vals = np.asarray(da[lat_name].values)
    lon_vals = np.asarray(da[lon_name].values)
    lon_min, lon_max = convert_bounds(bounds, lon_convention(lon_vals))
    lat_slice = slice(bounds["lat_min"], bounds["lat_max"]) if lat_vals[0] <= lat_vals[-1] else slice(bounds["lat_max"], bounds["lat_min"])
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def weighted_mean(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    weights = xr.DataArray(np.cos(np.deg2rad(da[lat_name].values)), coords={lat_name: da[lat_name]}, dims=(lat_name,))
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    return da.weighted(weights).mean(dim=spatial_dims, skipna=True)


def pick_precip_variable(ds: xr.Dataset) -> str:
    candidates = [name for name in ds.data_vars if "precip" in name.lower() or "rain" in name.lower()]
    if not candidates:
        raise KeyError("Could not identify a precipitation variable in NeuralGCM output.")
    return candidates[0]


def pick_variable(ds: xr.Dataset, keywords: list[str]) -> str | None:
    for name in ds.data_vars:
        lower = name.lower()
        if any(keyword in lower for keyword in keywords):
            return name
    return None


def precip_kind_and_scale(da: xr.DataArray) -> tuple[str, float | None, str]:
    name = da.name.lower()
    units = str(da.attrs.get("units", "")).lower()
    if "cumulative" in name or "accum" in name:
        if "kg" in units and "m" in units:
            return "interval_total_mm", 1.0, "1 kg m-2 = 1 mm"
        if units in {"mm", "millimeter", "millimeters"}:
            return "interval_total_mm", 1.0, "already in mm"
        if units in {"m", "meter", "meters"}:
            return "interval_total_mm", 1000.0, "meters to mm"
        return "interval_total_native", 1.0, "cumulative precipitation field with missing units; totals are reported in native model units only"
    if "kg" in units and "m" in units and "s" in units:
        return "rate", 6.0 * 3600.0, "rate multiplied by 6-hour interval"
    if units in {"m/s", "m s-1"}:
        return "rate", 1000.0 * 6.0 * 3600.0, "meters per second multiplied by 6-hour interval and converted to mm"
    return "unknown", None, "precipitation units unclear"


def monthly_region_values(series: xr.DataArray) -> dict[str, float]:
    out = {}
    for timestamp, value in zip(series.time.values, np.asarray(series.values, dtype=float), strict=True):
        label = str(np.datetime_as_string(timestamp, unit="M"))
        out[label] = float(out.get(label, 0.0) + value)
    return out


def compare_with_prism(region_name: str, model_monthly: dict[str, float], prism_2020: Path, prism_2021: Path) -> dict | None:
    if not prism_2020.exists() or not prism_2021.exists():
        return None
    datasets = [xr.open_dataset(prism_2020), xr.open_dataset(prism_2021)]
    try:
        monthly = xr.concat([ds["PPT"] for ds in datasets], dim="time").sortby("time")
        monthly = monthly.sel(time=slice("2020-09-01", "2021-03-31")).resample(time="MS").sum()
        subset = subset_region(monthly, "lat", "lon", REGIONS[region_name])
        series = weighted_mean(subset, "lat")
        prism_monthly = {
            str(np.datetime_as_string(ts, unit="M")): float(val)
            for ts, val in zip(series.time.values, np.asarray(series.values, dtype=float), strict=True)
        }
    finally:
        for ds in datasets:
            ds.close()

    months = sorted(set(model_monthly) & set(prism_monthly))
    if not months:
        return None
    model_values = np.asarray([model_monthly[m] for m in months], dtype=float)
    prism_values = np.asarray([prism_monthly[m] for m in months], dtype=float)
    diff = model_values - prism_values
    return {
        "region": region_name,
        "months": months,
        "model_total_mm": float(model_values.sum()),
        "prism_total_mm": float(prism_values.sum()),
        "model_minus_prism_mm": float(diff.sum()),
        "model_over_prism": float(model_values.sum() / prism_values.sum()) if prism_values.sum() != 0 else None,
        "monthly_correlation": float(np.corrcoef(model_values, prism_values)[0, 1]) if len(months) >= 2 else None,
        "monthly_rmse_mm": float(np.sqrt(np.mean(diff ** 2))),
        "monthly_bias_mm": float(np.mean(diff)),
    }


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input) as ds:
        lat_name, lon_name = detect_coord_names(ds)
        precip_var = pick_precip_variable(ds)
        temp_var = pick_variable(ds, ["2m_temperature", "tmp2m", "temperature"])
        wind_var = pick_variable(ds, ["10m_u_component", "10m_v_component", "wind"])

        precip_da = ds[precip_var]
        precip_kind, precip_scale, precip_note = precip_kind_and_scale(precip_da)
        if precip_kind in {"interval_total_mm", "interval_total_native"}:
            precip_mm = precip_da * precip_scale
        elif precip_kind == "rate":
            precip_mm = precip_da * precip_scale
        else:
            precip_mm = None

        rows = []
        summary = {
            "input_file": str(args.input),
            "precipitation_variable": precip_var,
            "precipitation_units": str(precip_da.attrs.get("units", "")),
            "precipitation_interpretation": precip_kind,
            "precipitation_conversion_note": precip_note,
            "regions": {},
        }
        prism_rows = []

        for region_name, bounds in REGIONS.items():
            region_entry = {"grid_cell_count": None}

            precip_subset = subset_region(precip_da, lat_name, lon_name, bounds)
            region_entry["grid_cell_count"] = int(precip_subset.sizes[lat_name] * precip_subset.sizes[lon_name])
            mean_rate = weighted_mean(precip_subset, lat_name)
            monthly_mean_rate = {
                str(np.datetime_as_string(ts, unit="M")): float(val)
                for ts, val in zip(mean_rate.time.values, np.asarray(mean_rate.values, dtype=float), strict=True)
            }
            row = {
                "region": region_name,
                "precipitation_variable": precip_var,
                "precipitation_units": str(precip_da.attrs.get("units", "")),
                "sep_mar_mean_precip_native_units": float(np.nanmean(mean_rate.values)),
            }
            region_entry["monthly_mean_precip_native_units"] = monthly_mean_rate

            if precip_mm is not None:
                precip_mm_subset = subset_region(precip_mm, lat_name, lon_name, bounds)
                precip_mm_series = weighted_mean(precip_mm_subset, lat_name)
                monthly_totals = monthly_region_values(precip_mm_series)
                total_value = float(np.nansum(list(monthly_totals.values())))
                if precip_kind == "interval_total_mm" or precip_kind == "rate":
                    row["sep_mar_total_precip_mm"] = total_value
                    for month, value in monthly_totals.items():
                        row[f"{month}_precip_mm"] = value
                    region_entry["monthly_total_precip_mm"] = monthly_totals

                    prism = compare_with_prism(region_name, monthly_totals, args.prism_2020, args.prism_2021)
                    if prism is not None:
                        prism_rows.append(prism)
                else:
                    row["sep_mar_total_precip_mm"] = None
                    row["sep_mar_total_precip_native_units"] = total_value
                    for month, value in monthly_totals.items():
                        row[f"{month}_precip_native_units"] = value
                    region_entry["monthly_total_precip_native_units"] = monthly_totals
            else:
                row["sep_mar_total_precip_mm"] = None

            if temp_var is not None:
                temp_series = weighted_mean(subset_region(ds[temp_var], lat_name, lon_name, bounds), lat_name)
                row["sep_mar_mean_temperature"] = float(np.nanmean(temp_series.values))
                region_entry["temperature_variable"] = temp_var
            if wind_var is not None:
                wind_series = weighted_mean(subset_region(ds[wind_var], lat_name, lon_name, bounds), lat_name)
                row["sep_mar_mean_wind"] = float(np.nanmean(wind_series.values))
                region_entry["wind_variable"] = wind_var

            summary["regions"][region_name] = region_entry
            rows.append(row)

    pd.DataFrame(rows).to_csv(FEATURE_CSV, index=False)
    FEATURE_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if prism_rows:
        prism_df = pd.DataFrame(prism_rows)
        prism_df.to_csv(PRISM_CSV, index=False)
        PRISM_JSON.write_text(json.dumps({"regions": prism_rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
