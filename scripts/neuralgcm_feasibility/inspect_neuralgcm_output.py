#!/usr/bin/env python3
"""Inspect NeuralGCM output files and summarize variables and basic stats."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs/wy2021_actual_m001")
TXT_OUT = REPORT_DIR / "wy2021_actual_m001_output_inspection.txt"
JSON_OUT = REPORT_DIR / "wy2021_actual_m001_output_inspection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def variable_role(name: str) -> str | None:
    lower = name.lower()
    if "precip" in lower or "rain" in lower:
        return "precipitation"
    if "temp" in lower:
        return "temperature"
    if "wind" in lower or lower.startswith("u_") or lower.startswith("v_"):
        return "wind"
    if "humid" in lower:
        return "humidity"
    return None


def stats_for_array(da: xr.DataArray) -> dict:
    values = np.asarray(da.values)
    finite = np.isfinite(values)
    return {
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "finite_count": int(finite.sum()),
        "min": float(np.nanmin(values)) if finite.any() else None,
        "max": float(np.nanmax(values)) if finite.any() else None,
        "mean": float(np.nanmean(values)) if finite.any() else None,
    }


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    nc_files = sorted(args.output_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {args.output_dir}")

    report = {
        "output_dir": str(args.output_dir),
        "files": [],
        "variables": {},
        "precipitation_variable_candidates": [],
    }

    for path in nc_files:
        with xr.open_dataset(path) as ds:
            file_report = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "dims": {k: int(v) for k, v in ds.sizes.items()},
                "coords": list(ds.coords),
                "data_vars": list(ds.data_vars),
            }
            if "time" in ds.coords:
                file_report["time_start"] = str(np.asarray(ds.time.values[0]).item())
                file_report["time_end"] = str(np.asarray(ds.time.values[-1]).item())
                file_report["time_count"] = int(ds.sizes["time"])
            for var_name in ds.data_vars:
                da = ds[var_name]
                var_info = {
                    "dims": list(da.dims),
                    "shape": [int(v) for v in da.shape],
                    "attrs": {k: str(v) for k, v in da.attrs.items()},
                    "role_hint": variable_role(var_name),
                }
                if var_info["role_hint"] in {"precipitation", "temperature", "wind", "humidity"}:
                    var_info["stats"] = stats_for_array(da)
                report["variables"][var_name] = var_info
                if var_info["role_hint"] == "precipitation":
                    report["precipitation_variable_candidates"].append(var_name)
            report["files"].append(file_report)

    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "NeuralGCM output inspection",
        f"output_dir: {args.output_dir}",
        f"files: {len(report['files'])}",
        f"precipitation_variable_candidates: {report['precipitation_variable_candidates']}",
        "",
    ]
    for file_report in report["files"]:
        lines.extend(
            [
                f"file: {file_report['path']}",
                f"  size_bytes: {file_report['size_bytes']}",
                f"  dims: {file_report['dims']}",
                f"  coords: {file_report['coords']}",
                f"  data_vars: {file_report['data_vars']}",
            ]
        )
        if "time_start" in file_report:
            lines.extend(
                [
                    f"  time_start: {file_report['time_start']}",
                    f"  time_end: {file_report['time_end']}",
                    f"  time_count: {file_report['time_count']}",
                ]
            )
    for var_name, var_info in report["variables"].items():
        lines.append(f"variable: {var_name}")
        lines.append(f"  role_hint: {var_info['role_hint']}")
        lines.append(f"  dims: {var_info['dims']}")
        lines.append(f"  shape: {var_info['shape']}")
        lines.append(f"  attrs: {var_info['attrs']}")
        if "stats" in var_info:
            lines.append(f"  stats: {var_info['stats']}")
    TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
