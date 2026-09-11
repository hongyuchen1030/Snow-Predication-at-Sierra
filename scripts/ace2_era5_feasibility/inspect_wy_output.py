#!/usr/bin/env python3
"""Inspect a real water-year ACE2 monthly prediction output file."""

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
    parser.add_argument("--report-txt", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def infer_latitude_order(values: np.ndarray) -> str:
    if values.size < 2:
        return "single_value_or_unknown"
    diffs = np.diff(values)
    if np.all(diffs > 0):
        return "ascending_south_to_north"
    if np.all(diffs < 0):
        return "descending_north_to_south"
    return "non_monotonic"


def infer_longitude_convention(values: np.ndarray) -> str:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin >= 0.0 and vmax > 180.0:
        return "0_to_360"
    if vmin < 0.0 and vmax <= 180.0:
        return "minus180_to_180"
    return "mixed_or_unclear"


def summarize_var(da: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(da.values)
    finite = np.isfinite(values)
    out = {
        "dims": list(da.dims),
        "shape": [int(v) for v in da.shape],
        "attrs": {k: v for k, v in da.attrs.items()},
        "missing_value_count": int(values.size - finite.sum()),
    }
    if finite.any():
        out["min"] = float(np.nanmin(values))
        out["max"] = float(np.nanmax(values))
        out["mean"] = float(np.nanmean(values))
    else:
        out["min"] = None
        out["max"] = None
        out["mean"] = None
    return out


def infer_precip_reliability(ds: xr.Dataset) -> dict[str, Any]:
    result = {
        "units": None,
        "classification": "missing",
        "conversion_reliable": False,
        "warning": None,
        "seconds_per_interval": None,
    }
    if "PRATEsfc" not in ds.data_vars:
        result["warning"] = "PRATEsfc not present."
        return result
    units = str(ds["PRATEsfc"].attrs.get("units", ""))
    result["units"] = units
    lower = units.lower()
    if lower in {"kg/m**2/s", "kg/m^2/s", "kg m-2 s-1", "mm/s", "m/s"}:
        result["classification"] = "precipitation_rate"
    else:
        result["classification"] = "unclear"
        result["warning"] = "PRATEsfc units are not clearly an instantaneous precipitation rate."
        return result

    if "valid_time" not in ds.coords:
        result["warning"] = "No valid_time coordinate available for inferring interval length."
        return result
    valid = np.ravel(ds["valid_time"].values)
    if valid.size < 2:
        result["warning"] = "Only one valid_time is present, so period-total conversion is not reliable."
        return result
    deltas = np.diff(valid).astype("timedelta64[s]").astype(np.int64)
    if deltas.size == 0 or np.any(deltas <= 0):
        result["warning"] = "Could not infer a positive time interval from valid_time."
        return result
    if len(set(deltas.tolist())) != 1:
        result["warning"] = "Time intervals are not uniform, so a simple precipitation total is not reliable."
        return result
    result["conversion_reliable"] = True
    result["seconds_per_interval"] = int(deltas[0])
    return result


def build_report(path: Path) -> dict[str, Any]:
    with xr.open_dataset(path) as ds:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        lat = np.asarray(ds[lat_name].values)
        lon = np.asarray(ds[lon_name].values)
        report = {
            "input_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "dimensions": {k: int(v) for k, v in ds.sizes.items()},
            "coordinates": list(ds.coords),
            "data_vars": list(ds.data_vars),
            "latitude_order": infer_latitude_order(lat),
            "longitude_convention": infer_longitude_convention(lon),
            "time_values": [int(v) for v in np.ravel(ds["time"].values).tolist()] if "time" in ds.coords else [],
            "time_attrs": dict(ds["time"].attrs) if "time" in ds.coords else {},
            "init_time_values": [str(v) for v in np.ravel(ds["init_time"].values)] if "init_time" in ds.coords else [],
            "valid_time_values": [str(v) for v in np.ravel(ds["valid_time"].values)] if "valid_time" in ds.coords else [],
            "variables": {},
            "precipitation_interpretation": infer_precip_reliability(ds),
        }
        for var_name in TARGET_VARS:
            if var_name in ds.data_vars:
                report["variables"][var_name] = summarize_var(ds[var_name])
        return report


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"input_path: {report['input_path']}",
        f"file_size_bytes: {report['file_size_bytes']}",
        f"dimensions: {report['dimensions']}",
        f"coordinates: {report['coordinates']}",
        f"data_vars: {report['data_vars']}",
        f"latitude_order: {report['latitude_order']}",
        f"longitude_convention: {report['longitude_convention']}",
        f"time_values: {report['time_values']}",
        f"time_attrs: {report['time_attrs']}",
        f"init_time_values: {report['init_time_values']}",
        f"valid_time_values: {report['valid_time_values']}",
        f"precipitation_interpretation: {report['precipitation_interpretation']}",
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
