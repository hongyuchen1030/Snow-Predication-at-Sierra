#!/usr/bin/env python3
"""Inspect ACE2 forcing files and identify the SST-like forcing variable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--report-txt", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def detect_sst_variable(ds: xr.Dataset) -> str | None:
    candidates: list[tuple[int, str]] = []
    for name, da in ds.data_vars.items():
        score = 0
        lower_name = name.lower()
        attrs = {k: str(v).lower() for k, v in da.attrs.items()}
        haystacks = [lower_name, attrs.get("long_name", ""), attrs.get("standard_name", ""), attrs.get("grib_name", "")]
        if "sst" in lower_name:
            score += 5
        if "surface_temperature" == lower_name:
            score += 4
        if "skin temperature" in haystacks[1] or "skin temperature" in haystacks[3]:
            score += 4
        if attrs.get("units", "").lower() in {"k", "kelvin", "c", "degc", "celsius"}:
            score += 1
        if score:
            candidates.append((score, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


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


def summarize_variable(da: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(da.values)
    finite = np.isfinite(values)
    summary: dict[str, Any] = {
        "dims": list(da.dims),
        "shape": [int(v) for v in da.shape],
        "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in da.attrs.items()},
        "missing_value_count": int(values.size - finite.sum()),
    }
    if finite.any():
        summary["min"] = float(np.nanmin(values))
        summary["max"] = float(np.nanmax(values))
        summary["mean"] = float(np.nanmean(values))
    else:
        summary["min"] = None
        summary["max"] = None
        summary["mean"] = None
    return summary


def inspect_file(path: Path) -> dict[str, Any]:
    with xr.open_dataset(path) as ds:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        sst_var = detect_sst_variable(ds)
        sea_ice_vars = [name for name in ds.data_vars if "ice" in name.lower()]
        mask_vars = [
            name for name in ds.data_vars if any(token in name.lower() for token in ("land_fraction", "ocean_fraction", "mask"))
        ]
        report: dict[str, Any] = {
            "input_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "variables": list(ds.data_vars),
            "dimensions": {k: int(v) for k, v in ds.sizes.items()},
            "coordinates": list(ds.coords),
            "time_coverage": {
                "start": str(ds["time"].values[0]) if "time" in ds.coords else None,
                "end": str(ds["time"].values[-1]) if "time" in ds.coords else None,
                "count": int(ds.sizes.get("time", 0)),
            },
            "latitude_order": infer_latitude_order(np.asarray(ds[lat_name].values)),
            "longitude_convention": infer_longitude_convention(np.asarray(ds[lon_name].values)),
            "latitude_range": [float(ds[lat_name].min()), float(ds[lat_name].max())],
            "longitude_range": [float(ds[lon_name].min()), float(ds[lon_name].max())],
            "sst_variable": sst_var,
            "sea_ice_variables": sea_ice_vars,
            "mask_variables": mask_vars,
            "variables_summary": {},
        }
        if sst_var is not None:
            sst_summary = summarize_variable(ds[sst_var])
            report["variables_summary"][sst_var] = sst_summary
            units = str(ds[sst_var].attrs.get("units", ""))
            report["sst_units"] = units
            if units.lower() in {"k", "kelvin"}:
                report["sst_temperature_scale"] = "kelvin"
            elif units.lower() in {"c", "degc", "celsius"}:
                report["sst_temperature_scale"] = "celsius"
            else:
                report["sst_temperature_scale"] = "unclear"
        for name in sea_ice_vars + mask_vars:
            if name not in report["variables_summary"]:
                report["variables_summary"][name] = summarize_variable(ds[name])
        return report


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for entry in report["files"]:
        lines.extend(
            [
                f"input_path: {entry['input_path']}",
                f"file_size_bytes: {entry['file_size_bytes']}",
                f"variables: {entry['variables']}",
                f"dimensions: {entry['dimensions']}",
                f"coordinates: {entry['coordinates']}",
                f"time_coverage: {entry['time_coverage']}",
                f"latitude_order: {entry['latitude_order']}",
                f"longitude_convention: {entry['longitude_convention']}",
                f"latitude_range: {entry['latitude_range']}",
                f"longitude_range: {entry['longitude_range']}",
                f"sst_variable: {entry['sst_variable']}",
                f"sst_units: {entry.get('sst_units')}",
                f"sst_temperature_scale: {entry.get('sst_temperature_scale')}",
                f"sea_ice_variables: {entry['sea_ice_variables']}",
                f"mask_variables: {entry['mask_variables']}",
            ]
        )
        for name, summary in entry["variables_summary"].items():
            lines.extend(
                [
                    f"[{name}]",
                    f"  dims: {summary['dims']}",
                    f"  shape: {summary['shape']}",
                    f"  attrs: {summary['attrs']}",
                    f"  missing_value_count: {summary['missing_value_count']}",
                    f"  min: {summary['min']}",
                    f"  max: {summary['max']}",
                    f"  mean: {summary['mean']}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = {"files": [inspect_file(path) for path in args.inputs]}
    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.write_text(render_text(report), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render_text(report))


if __name__ == "__main__":
    main()
