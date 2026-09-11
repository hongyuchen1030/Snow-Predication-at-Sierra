#!/usr/bin/env python3
"""Build an ACE2-grid monthly surface_temperature climatology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Monthly climatology source file.")
    parser.add_argument("--template", type=Path, required=True, help="ACE2 forcing file used as grid template.")
    parser.add_argument("--output", type=Path, required=True, help="ACE2-grid climatology NetCDF output path.")
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with xr.open_dataset(args.source, decode_times=False) as source_ds, xr.open_dataset(args.template) as template_ds:
        if "sst" not in source_ds.data_vars:
            raise KeyError("Expected variable 'sst' in climatology source.")
        if "surface_temperature" not in template_ds.data_vars:
            raise KeyError("Expected variable 'surface_temperature' in ACE2 template.")

        source_sst = source_ds["sst"]
        if str(source_sst.attrs.get("units", "")).lower() not in {"degc", "c", "celsius"}:
            raise ValueError(f"Unexpected source SST units: {source_sst.attrs.get('units')}")

        source_sst_k = source_sst.astype("float32") + np.float32(273.15)
        source_sst_k = source_sst_k.rename({"time": "month", "lat": "source_latitude", "lon": "longitude"})
        source_sst_k = source_sst_k.assign_coords(
            month=("month", np.arange(1, int(source_sst_k.sizes["month"]) + 1, dtype=np.int32)),
            source_latitude=("source_latitude", source_ds["lat"].values.astype(np.float64)),
            longitude=("longitude", source_ds["lon"].values.astype(np.float64)),
        )

        target_lat = template_ds["latitude"].values.astype(np.float64)
        target_lon = template_ds["longitude"].values.astype(np.float64)
        regridded = source_sst_k.interp(
            source_latitude=target_lat,
            longitude=target_lon,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        ).rename({"source_latitude": "latitude"})
        regridded = regridded.assign_coords(
            latitude=("latitude", target_lat),
            longitude=("longitude", target_lon),
        )
        regridded = regridded.transpose("month", "latitude", "longitude").astype("float32")
        regridded.name = "surface_temperature_climatology"
        regridded.attrs = {
            "long_name": "Monthly surface temperature climatology on the ACE2 forcing grid",
            "units": "K",
            "source_file": str(args.source),
            "source_variable": "sst",
            "source_units": str(source_sst.attrs.get("units", "")),
            "regridding_method": "xarray linear interpolation in latitude; longitude coordinates already aligned",
            "target_grid_file": str(args.template),
        }

        out_ds = xr.Dataset({"surface_temperature_climatology": regridded})
        out_ds["month"].attrs = {"long_name": "calendar month", "units": "1-12"}
        out_ds["latitude"].attrs = dict(template_ds["latitude"].attrs)
        out_ds["longitude"].attrs = dict(template_ds["longitude"].attrs)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(args.output)

        report: dict[str, Any] = {
            "source_file": str(args.source),
            "template_file": str(args.template),
            "output_file": str(args.output),
            "source_variable": "sst",
            "source_units": str(source_sst.attrs.get("units", "")),
            "output_variable": "surface_temperature_climatology",
            "output_units": "K",
            "source_dims": {k: int(v) for k, v in source_ds.sizes.items()},
            "template_dims": {k: int(v) for k, v in template_ds.sizes.items()},
            "output_dims": {k: int(v) for k, v in out_ds.sizes.items()},
            "source_latitude_range": [float(source_ds["lat"].min()), float(source_ds["lat"].max())],
            "template_latitude_range": [float(template_ds["latitude"].min()), float(template_ds["latitude"].max())],
            "source_longitude_range": [float(source_ds["lon"].min()), float(source_ds["lon"].max())],
            "template_longitude_range": [float(template_ds["longitude"].min()), float(template_ds["longitude"].max())],
            "regridding_needed": True,
            "regridding_method": "xarray linear interpolation",
            "min_k": float(regridded.min()),
            "max_k": float(regridded.max()),
        }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
