#!/usr/bin/env python3
"""Extract member-level and reduced water-year features from ACE2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_REGIONS = {
    "sierra_nevada": {"lat_min": 35.0, "lat_max": 42.5, "lon_min": 235.0, "lon_max": 242.0},
    "california_western_us": {"lat_min": 30.0, "lat_max": 50.0, "lon_min": 230.0, "lon_max": 250.0},
    "north_pacific": {"lat_min": 20.0, "lat_max": 60.0, "lon_min": 150.0, "lon_max": 240.0},
}

VAR_ALIASES = {
    "precip": ["PRATEsfc", "precip", "precipitation", "tp"],
    "temp2m": ["TMP2m", "t2m", "tas"],
    "slp": ["SLP", "PRMSL", "msl", "PRESsfc"],
    "z500": ["Z500", "HGT500", "z500"],
    "u10": ["UGRD10m", "u10"],
    "v10": ["VGRD10m", "v10"],
    "humidity": ["Q2m", "RH2m", "q2m", "r2m", "specific_humidity", "relative_humidity"],
    "u": ["UGRD", "u"],
    "v": ["VGRD", "v"],
}

PRECIP_WET_THRESHOLD = 1.0e-5
TEMP_COLD_THRESHOLD_K = 273.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="ACE2 output files.")
    parser.add_argument("--water-year", type=int, required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--member-csv", type=Path, required=True)
    parser.add_argument("--reduced-csv", type=Path, required=True)
    parser.add_argument("--regions-json", type=Path, help="Optional JSON file of named bounding boxes.")
    return parser.parse_args()


def load_regions(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return DEFAULT_REGIONS
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_var(ds: xr.Dataset, aliases: list[str]) -> str | None:
    lowered = {name.lower(): name for name in ds.data_vars}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    for name in ("lon", "longitude"):
        if name in ds.coords:
            lon = xr.where(ds[name] < 0, ds[name] + 360.0, ds[name])
            return ds.assign_coords({name: lon}).sortby(name)
    return ds


def subset_region(ds: xr.Dataset, bounds: dict[str, float]) -> xr.Dataset:
    ds = normalize_lon(ds)
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    return ds.sel(
        {
            lat_name: slice(bounds["lat_min"], bounds["lat_max"]),
            lon_name: slice(bounds["lon_min"], bounds["lon_max"]),
        }
    )


def spatial_mean(da: xr.DataArray) -> xr.DataArray:
    dims = [d for d in da.dims if d not in ("time", "sample", "member")]
    if not dims:
        return da
    return da.mean(dim=dims, skipna=True)


def to_monthly(da: xr.DataArray) -> xr.DataArray:
    return da.resample(time="MS").mean()


def compute_feature_block(ds: xr.Dataset, region_name: str, region_bounds: dict[str, float]) -> dict[str, float]:
    block: dict[str, float] = {}
    region = subset_region(ds, region_bounds)

    selected = {label: pick_var(region, aliases) for label, aliases in VAR_ALIASES.items()}
    month_index = None

    for label, var_name in selected.items():
        if var_name is None:
            continue
        monthly = to_monthly(spatial_mean(region[var_name]))
        month_index = monthly["time"].dt.strftime("%Y-%m").values
        values = monthly.values.astype(float)
        block[f"{region_name}__{label}__mean"] = float(np.nanmean(values))
        block[f"{region_name}__{label}__std"] = float(np.nanstd(values))
        for idx, month in enumerate(month_index):
            block[f"{region_name}__{label}__{month}"] = float(values[idx])

    precip_name = selected["precip"]
    temp_name = selected["temp2m"]
    if precip_name is not None:
        precip = spatial_mean(region[precip_name])
        wet = (precip > PRECIP_WET_THRESHOLD).mean(dim="time")
        block[f"{region_name}__wet_probability"] = float(wet.values)
        if temp_name is not None:
            temp = spatial_mean(region[temp_name])
            cold_wet = ((precip > PRECIP_WET_THRESHOLD) & (temp < TEMP_COLD_THRESHOLD_K)).mean(dim="time")
            block[f"{region_name}__cold_wet_probability"] = float(cold_wet.values)

    if selected["u"] is not None and selected["v"] is not None:
        u = spatial_mean(region[selected["u"]])
        v = spatial_mean(region[selected["v"]])
        ivt_proxy = np.sqrt(u**2 + v**2)
        block[f"{region_name}__ivt_proxy_mean"] = float(ivt_proxy.mean().values)

    return block


def reduce_ensemble(member_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["water_year"]
    feature_cols = [col for col in member_df.columns if col not in {"water_year", "member_id"}]
    row: dict[str, Any] = {"water_year": int(member_df["water_year"].iloc[0])}
    for col in feature_cols:
        values = member_df[col].astype(float).to_numpy()
        row[f"ensemble_mean__{col}"] = float(np.nanmean(values))
        row[f"ensemble_std__{col}"] = float(np.nanstd(values))
        row[f"ensemble_q10__{col}"] = float(np.nanquantile(values, 0.10))
        row[f"ensemble_q50__{col}"] = float(np.nanquantile(values, 0.50))
        row[f"ensemble_q90__{col}"] = float(np.nanquantile(values, 0.90))
    return pd.DataFrame([row], columns=id_cols + [c for c in row if c not in id_cols])


def append_row(path: Path, row_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    row_df.to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    args = parse_args()
    regions = load_regions(args.regions_json)
    ds = xr.open_mfdataset(sorted(map(str, args.input)), combine="by_coords")
    try:
        member_row: dict[str, Any] = {"water_year": args.water_year, "member_id": args.member_id}
        for region_name, bounds in regions.items():
            member_row.update(compute_feature_block(ds, region_name, bounds))
        member_df = pd.DataFrame([member_row])
        append_row(args.member_csv, member_df)

        existing = pd.read_csv(args.member_csv)
        reduced_df = reduce_ensemble(existing[existing["water_year"] == args.water_year])
        remaining = pd.DataFrame()
        if args.reduced_csv.exists():
            current = pd.read_csv(args.reduced_csv)
            remaining = current[current["water_year"] != args.water_year]
        final_df = pd.concat([remaining, reduced_df], ignore_index=True)
        final_df.to_csv(args.reduced_csv, index=False)
        print(member_df.to_csv(index=False).strip())
        print(reduced_df.to_csv(index=False).strip())
    finally:
        ds.close()


if __name__ == "__main__":
    main()
