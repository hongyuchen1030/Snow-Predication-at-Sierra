#!/usr/bin/env python3
"""Create ACE2 forcing states for Pacific SST actual/neutral/reversed experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr


PACIFIC_DOMAIN = {
    "lat_min": -10.0,
    "lat_max": 60.0,
    "lon_min_360": 120.0,
    "lon_max_360": 280.0,
    "ocean_fraction_threshold": 0.5,
}

STATE_NAMES = ("actual", "neutral", "reversed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--climatology", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-txt", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    return parser.parse_args()


def build_mask(ds: xr.Dataset) -> xr.DataArray:
    lat = ds["latitude"]
    lon = ds["longitude"]
    ocean_fraction = ds["ocean_fraction"]
    lat_mask = (lat >= PACIFIC_DOMAIN["lat_min"]) & (lat <= PACIFIC_DOMAIN["lat_max"])
    lon_mask = (lon >= PACIFIC_DOMAIN["lon_min_360"]) & (lon <= PACIFIC_DOMAIN["lon_max_360"])
    lat_grid, lon_grid = xr.broadcast(lat, lon)
    latlon_mask = (
        (lat_grid >= PACIFIC_DOMAIN["lat_min"])
        & (lat_grid <= PACIFIC_DOMAIN["lat_max"])
        & (lon_grid >= PACIFIC_DOMAIN["lon_min_360"])
        & (lon_grid <= PACIFIC_DOMAIN["lon_max_360"])
    ).transpose("latitude", "longitude")
    ocean_mask = ocean_fraction.isel(time=0) > PACIFIC_DOMAIN["ocean_fraction_threshold"]
    mask2d = latlon_mask & ocean_mask
    if int(mask2d.sum().item()) == 0:
        raise ValueError("Pacific ocean mask selected zero grid cells.")
    return mask2d


def monthly_climatology_for_times(clim: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    months = xr.DataArray(times.dt.month.values.astype(np.int32), coords={"time": times}, dims=("time",))
    return clim.sel(month=months).assign_coords(time=times)


def compute_state_surface_temperature(state: str, actual: xr.DataArray, clim_for_time: xr.DataArray, mask3d: xr.DataArray) -> xr.DataArray:
    if state == "actual":
        return actual
    if state == "neutral":
        state_values = xr.where(mask3d, clim_for_time, actual)
        return state_values.astype(actual.dtype)
    if state == "reversed":
        reversed_values = 2.0 * clim_for_time - actual
        state_values = xr.where(mask3d, reversed_values, actual)
        return state_values.astype(actual.dtype)
    raise ValueError(f"Unknown state: {state}")


def suspicious_count(values: xr.DataArray) -> dict[str, Any]:
    arr = np.asarray(values.values, dtype=float)
    finite = np.isfinite(arr)
    return {
        "count_below_180K": int(np.sum(finite & (arr < 180.0))),
        "count_above_330K": int(np.sum(finite & (arr > 330.0))),
        "count_below_271_35K": int(np.sum(finite & (arr < 271.35))),
        "count_above_310K": int(np.sum(finite & (arr > 310.0))),
    }


def write_state_file(
    input_path: Path,
    output_path: Path,
    state: str,
    clim_ds: xr.Dataset,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state == "actual":
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
        os.symlink(input_path, output_path)
        return (
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "state": state,
                "write_mode": "symlink_to_control",
            },
            validate_state(input_path, output_path, state, clim_ds),
        )

    with xr.open_dataset(input_path) as ds:
        clim = clim_ds["surface_temperature_climatology"]
        mask2d = build_mask(ds)
        mask3d = mask2d.broadcast_like(ds["surface_temperature"])
        clim_for_time = monthly_climatology_for_times(clim, ds["time"])
        state_sst = compute_state_surface_temperature(state, ds["surface_temperature"], clim_for_time, mask3d)
        state_sst_values = np.asarray(state_sst.values, dtype=ds["surface_temperature"].dtype)

    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    shutil.copy2(input_path, output_path)
    with netCDF4.Dataset(output_path, mode="r+") as nc_out:
        nc_out.variables["surface_temperature"][:] = state_sst_values

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "state": state,
        "write_mode": "copy_control_then_replace_surface_temperature_in_place",
        "pacific_ocean_grid_cell_count_2d": int(mask2d.sum().item()),
    }

    return metadata, validate_state(input_path, output_path, state, clim_ds)


def validate_state(input_path: Path, output_path: Path, state: str, clim_ds: xr.Dataset) -> dict[str, Any]:
    with xr.open_dataset(input_path) as control, xr.open_dataset(output_path) as candidate:
        clim = clim_ds["surface_temperature_climatology"]
        mask2d = build_mask(control)
        mask3d = mask2d.broadcast_like(control["surface_temperature"])
        control_sst = control["surface_temperature"]
        candidate_sst = candidate["surface_temperature"]
        clim_for_time = monthly_climatology_for_times(clim, control["time"]).astype(control_sst.dtype)

        if state == "actual":
            expected_inside = control_sst
        elif state == "neutral":
            expected_inside = clim_for_time
        elif state == "reversed":
            expected_inside = (2.0 * clim_for_time - control_sst).astype(control_sst.dtype)
        else:
            raise ValueError(state)

        expected_full = xr.where(mask3d, expected_inside, control_sst).astype(control_sst.dtype)
        error = candidate_sst - expected_full
        outside_error = (candidate_sst - control_sst).where(~mask3d)

        unchanged_flags: dict[str, bool] = {}
        all_non_sst_unchanged = True
        for name in control.data_vars:
            if name == "surface_temperature":
                continue
            same = bool(control[name].identical(candidate[name]))
            unchanged_flags[name] = same
            all_non_sst_unchanged = all_non_sst_unchanged and same

        control_inside = control_sst.where(mask3d)
        clim_inside = clim_for_time.where(mask3d)
        candidate_inside = candidate_sst.where(mask3d)
        reversed_inside = (2.0 * clim_for_time - control_sst).where(mask3d)

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "state": state,
            "surface_temperature_only_changed_variable": all_non_sst_unchanged,
            "all_non_sst_variables_exactly_unchanged": all_non_sst_unchanged,
            "non_sst_variable_exact_unchanged": unchanged_flags,
            "outside_pacific_mask_surface_temperature_exactly_unchanged": bool(np.array_equal(
                np.nan_to_num(outside_error.values, nan=0.0),
                np.zeros_like(np.nan_to_num(outside_error.values, nan=0.0)),
            )),
            "pacific_ocean_grid_cell_count_2d": int(mask2d.sum().item()),
            "pacific_ocean_grid_value_count_3d": int(mask3d.sum().item()),
            "control_min_k_inside_pacific": float(np.nanmin(control_inside.values)),
            "control_max_k_inside_pacific": float(np.nanmax(control_inside.values)),
            "climatology_min_k_inside_pacific": float(np.nanmin(clim_inside.values)),
            "climatology_max_k_inside_pacific": float(np.nanmax(clim_inside.values)),
            f"{state}_min_k_inside_pacific": float(np.nanmin(candidate_inside.values)),
            f"{state}_max_k_inside_pacific": float(np.nanmax(candidate_inside.values)),
            "reversed_formula_min_k_inside_pacific": float(np.nanmin(reversed_inside.values)),
            "reversed_formula_max_k_inside_pacific": float(np.nanmax(reversed_inside.values)),
            "max_abs_error_expected_formula": float(np.nanmax(np.abs(error.values))),
            "mean_abs_error_expected_formula": float(np.nanmean(np.abs(error.values))),
            "suspicious_counts_inside_pacific": suspicious_count(candidate_inside),
            "mask_definition": PACIFIC_DOMAIN,
        }


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for entry in report["created_files"]:
        lines.extend(
            [
                f"state: {entry['state']}",
                f"input_path: {entry['input_path']}",
                f"output_path: {entry['output_path']}",
                f"write_mode: {entry['write_mode']}",
            ]
        )
        if "pacific_ocean_grid_cell_count_2d" in entry:
            lines.append(f"pacific_ocean_grid_cell_count_2d: {entry['pacific_ocean_grid_cell_count_2d']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.validation_csv.parent.mkdir(parents=True, exist_ok=True)
    args.validation_json.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.climatology) as clim_ds:
        if "surface_temperature_climatology" not in clim_ds.data_vars:
            raise KeyError("Expected surface_temperature_climatology in climatology file.")

        created_files: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []

        for state in STATE_NAMES:
            state_dir = args.output_root / state
            state_dir.mkdir(parents=True, exist_ok=True)
            for input_path in args.inputs:
                output_path = state_dir / input_path.name
                metadata, validation = write_state_file(input_path, output_path, state, clim_ds)
                created_files.append(metadata)
                validations.append(validation)

    report = {
        "created_files": created_files,
        "climatology_path": str(args.climatology),
        "mask_definition": PACIFIC_DOMAIN,
    }
    args.report_txt.write_text(render_text(report), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(validations).to_csv(args.validation_csv, index=False)
    args.validation_json.write_text(json.dumps({"validations": validations}, indent=2) + "\n", encoding="utf-8")
    print(render_text(report))
    print(pd.DataFrame(validations).to_csv(index=False).strip())


if __name__ == "__main__":
    main()
