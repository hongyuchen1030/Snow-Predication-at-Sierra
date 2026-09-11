#!/usr/bin/env python3
"""Inspect the ACE2 smoke-test monthly prediction output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


TARGET_VARS = ("TMP2m", "PRATEsfc", "VGRD10m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to monthly_mean_predictions.nc")
    parser.add_argument("--report-txt", type=Path, required=True, help="Path to text report output")
    parser.add_argument("--report-json", type=Path, required=True, help="Path to JSON report output")
    return parser.parse_args()


def infer_latitude_order(lat_values: np.ndarray) -> str:
    if lat_values.size < 2:
        return "single_value_or_unknown"
    if np.all(np.diff(lat_values) > 0):
        return "ascending_south_to_north"
    if np.all(np.diff(lat_values) < 0):
        return "descending_north_to_south"
    return "non_monotonic"


def infer_longitude_convention(lon_values: np.ndarray) -> str:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    if lon_min >= 0.0 and lon_max > 180.0:
        return "0_to_360"
    if lon_min < 0.0 and lon_max <= 180.0:
        return "minus180_to_180"
    return "mixed_or_unclear"


def classify_precip_metadata(attrs: dict[str, Any]) -> str:
    units = str(attrs.get("units", "")).strip().lower()
    long_name = str(attrs.get("long_name", "")).strip().lower()
    if "kg/m**2/s" in units or "kg m-2 s-1" in units or units in {"kg/m^2/s", "mm/s", "m/s"}:
        return "precipitation_rate"
    if "accum" in long_name or units in {"mm", "kg/m**2", "kg/m^2", "m"}:
        return "accumulated_precipitation"
    return "unclear"


def summarize_variable(da: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(da.values)
    finite = np.isfinite(values)
    missing_count = int(values.size - finite.sum())
    summary: dict[str, Any] = {
        "dims": list(da.dims),
        "shape": [int(v) for v in da.shape],
        "attrs": {key: (value.item() if hasattr(value, "item") else value) for key, value in da.attrs.items()},
        "missing_value_count": missing_count,
    }
    if finite.any():
        summary.update(
            {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None})
    return summary


def build_report(path: Path) -> dict[str, Any]:
    file_size_bytes = path.stat().st_size
    with xr.open_dataset(path) as ds:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        lat_values = np.asarray(ds[lat_name].values)
        lon_values = np.asarray(ds[lon_name].values)

        variables = {name: summarize_variable(ds[name]) for name in TARGET_VARS if name in ds.data_vars}
        time_values = ds["time"].values.tolist() if "time" in ds.coords else []
        valid_time_values = []
        if "valid_time" in ds.coords:
            valid_time_values = [str(v) for v in np.ravel(ds["valid_time"].values)]
        init_time_values = []
        if "init_time" in ds.coords:
            init_time_values = [str(v) for v in np.ravel(ds["init_time"].values)]

        precip_metadata_classification = "missing"
        if "PRATEsfc" in ds.data_vars:
            precip_metadata_classification = classify_precip_metadata(dict(ds["PRATEsfc"].attrs))

        return {
            "input_path": str(path),
            "file_size_bytes": file_size_bytes,
            "global_attrs": {key: value for key, value in ds.attrs.items()},
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "coordinates": list(ds.coords),
            "data_vars": list(ds.data_vars),
            "latitude_name": lat_name,
            "longitude_name": lon_name,
            "latitude_order": infer_latitude_order(lat_values),
            "longitude_convention": infer_longitude_convention(lon_values),
            "latitude_preview": lat_values[:5].astype(float).tolist(),
            "longitude_preview": lon_values[:5].astype(float).tolist(),
            "time_values": [int(v) if np.isscalar(v) else v for v in np.ravel(time_values).tolist()],
            "time_attrs": dict(ds["time"].attrs) if "time" in ds.coords else {},
            "valid_time_values": valid_time_values,
            "init_time_values": init_time_values,
            "variables": variables,
            "prate_classification": precip_metadata_classification,
        }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"input_path: {report['input_path']}",
        f"file_size_bytes: {report['file_size_bytes']}",
        f"data_vars: {report['data_vars']}",
        f"dimensions: {report['dimensions']}",
        f"coordinates: {report['coordinates']}",
        f"latitude_name: {report['latitude_name']}",
        f"longitude_name: {report['longitude_name']}",
        f"latitude_order: {report['latitude_order']}",
        f"longitude_convention: {report['longitude_convention']}",
        f"latitude_preview: {report['latitude_preview']}",
        f"longitude_preview: {report['longitude_preview']}",
        f"time_values: {report['time_values']}",
        f"time_attrs: {report['time_attrs']}",
        f"init_time_values: {report['init_time_values']}",
        f"valid_time_values: {report['valid_time_values']}",
        f"prate_classification: {report['prate_classification']}",
        "",
    ]
    for var_name, summary in report["variables"].items():
        lines.extend(
            [
                f"[{var_name}]",
                f"  dims: {summary['dims']}",
                f"  shape: {summary['shape']}",
                f"  attrs: {summary['attrs']}",
                f"  missing_value_count: {summary['missing_value_count']}",
                f"  min: {summary['min']}",
                f"  max: {summary['max']}",
                f"  mean: {summary['mean']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args.input)
    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.write_text(render_text(report), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render_text(report))


if __name__ == "__main__":
    main()
