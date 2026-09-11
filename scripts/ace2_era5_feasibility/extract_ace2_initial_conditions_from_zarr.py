#!/usr/bin/env python3
"""Extract ACE2-compatible initial-condition files from the shared ERA5 zarr archive."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr


TIME_UNITS = "hours since 1940-01-01 12:00:00"
TIME_START = datetime(1940, 1, 1, 12, 0, 0)
TIME_STEP = timedelta(hours=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", type=Path, required=True)
    parser.add_argument("--template-ic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    parser.add_argument("--validation-output-json", type=Path, default=None)
    parser.add_argument(
        "--validate-against-template-time",
        default=None,
        help="If provided, extract this timestamp and compare against the template IC time slice.",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "")
    if "T" not in normalized:
        normalized = normalized + "T00:00:00"
    return datetime.fromisoformat(normalized)


def datetime_to_time_index(ts: datetime) -> int:
    delta = ts - TIME_START
    hours = delta.total_seconds() / 3600.0
    if hours % 6 != 0:
        raise ValueError(f"{ts.isoformat()} is not on the 6-hour archive grid.")
    index = int(round(hours / 6.0))
    if index < 0:
        raise ValueError(f"{ts.isoformat()} precedes the archive start.")
    return index


def datetime_to_time_value(ts: datetime) -> np.int64:
    delta = ts - TIME_START
    hours = delta.total_seconds() / 3600.0
    return np.int64(round(hours))


def load_template(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as ds:
        return ds.load()


def build_dataset(
    store: zarr.hierarchy.Group,
    template: xr.Dataset,
    ts: datetime,
) -> xr.Dataset:
    time_index = datetime_to_time_index(ts)
    time_value = datetime_to_time_value(ts)
    data_vars: dict[str, xr.DataArray] = {}
    for name in template.data_vars:
        array = store[name]
        values = np.asarray(array[time_index : time_index + 1, :, :], dtype=template[name].dtype)
        data_vars[name] = xr.DataArray(
            values,
            dims=template[name].dims,
            coords={
                "time": np.array([np.datetime64(ts)], dtype="datetime64[ns]"),
                "latitude": template["latitude"].values,
                "longitude": template["longitude"].values,
            },
            attrs=dict(template[name].attrs),
        )

    ds = xr.Dataset(data_vars=data_vars, attrs=dict(template.attrs))
    ds = ds.assign_coords(
        time=xr.DataArray(
            np.array([np.datetime64(ts)], dtype="datetime64[ns]"),
            dims=("time",),
            attrs={},
        ),
        latitude=xr.DataArray(
            np.asarray(template["latitude"].values),
            dims=("latitude",),
            attrs=dict(template["latitude"].attrs),
        ),
        longitude=xr.DataArray(
            np.asarray(template["longitude"].values),
            dims=("longitude",),
            attrs=dict(template["longitude"].attrs),
        ),
    )
    # Keep the raw hours-since origin as encoding so the on-disk time layout matches the template style.
    ds["time"].encoding.update({"units": TIME_UNITS, "calendar": "proleptic_gregorian", "dtype": "i8"})
    for name in ds.data_vars:
        ds[name].encoding["_FillValue"] = np.float32(np.nan)
        ds[name].encoding["dtype"] = "float32"
    return ds


def compare_datasets(reference: xr.Dataset, candidate: xr.Dataset, reference_time: str) -> dict[str, Any]:
    ref_slice = reference.sel(time=np.datetime64(reference_time))
    out: dict[str, Any] = {
        "reference_time": reference_time,
        "candidate_time": str(candidate.time.values[0]),
        "dimension_match": dict(ref_slice.sizes) == dict(candidate.isel(time=0).sizes),
        "latitude_equal": bool(np.array_equal(ref_slice["latitude"].values, candidate["latitude"].values)),
        "longitude_equal": bool(np.array_equal(ref_slice["longitude"].values, candidate["longitude"].values)),
        "variables": {},
    }
    all_ok = True
    cand_slice = candidate.isel(time=0)
    for name in reference.data_vars:
        ref_da = ref_slice[name]
        cand_da = cand_slice[name]
        max_abs = float(np.nanmax(np.abs(np.asarray(ref_da.values) - np.asarray(cand_da.values))))
        attrs_equal = dict(ref_da.attrs) == dict(cand_da.attrs)
        dtype_equal = str(ref_da.dtype) == str(cand_da.dtype)
        equal = bool(np.array_equal(ref_da.values, cand_da.values))
        if not (equal and attrs_equal and dtype_equal):
            all_ok = False
        out["variables"][name] = {
            "exact_values_equal": equal,
            "max_abs_difference": max_abs,
            "attrs_equal": attrs_equal,
            "dtype_equal": dtype_equal,
        }
    out["all_variables_exact"] = all_ok
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = load_template(args.template_ic)
    store = zarr.open_group(str(args.zarr_path), mode="r")

    validation_report: dict[str, Any] | None = None
    if args.validate_against_template_time is not None:
        validation_time = parse_timestamp(args.validate_against_template_time)
        validation_ds = build_dataset(store, template, validation_time)
        validation_report = compare_datasets(
            reference=template,
            candidate=validation_ds,
            reference_time=str(np.datetime64(validation_time)),
        )
        if args.validation_output_json is not None:
            args.validation_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.validation_output_json.write_text(
                json.dumps(validation_report, indent=2) + "\n",
                encoding="utf-8",
            )

    for timestamp in args.timestamps:
        ts = parse_timestamp(timestamp)
        ds = build_dataset(store, template, ts)
        output_path = args.output_dir / f"ic_{ts.strftime('%Y-%m-%d')}.nc"
        ds.to_netcdf(output_path, engine="netcdf4")
        print(output_path)

    if validation_report is not None:
        print(json.dumps(validation_report, indent=2))


if __name__ == "__main__":
    main()
