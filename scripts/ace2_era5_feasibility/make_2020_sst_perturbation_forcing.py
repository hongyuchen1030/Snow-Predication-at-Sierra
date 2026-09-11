#!/usr/bin/env python3
"""Create and validate Niño3.4 SST perturbation forcing files for ACE2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


PERTURBATIONS = {
    "2020_nino34_plus_1K": 1.0,
    "2020_nino34_minus_1K": -1.0,
}

NINO34 = {
    "lat_min": -5.0,
    "lat_max": 5.0,
    "lon_min_360": 190.0,
    "lon_max_360": 240.0,
}

SST_CLIP_RANGE_K = (180.0, 330.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-txt", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--sst-var", default="surface_temperature")
    return parser.parse_args()


def lon_bounds_for_dataset(ds: xr.Dataset, lon_name: str) -> tuple[float, float]:
    lon_values = np.asarray(ds[lon_name].values)
    if float(np.nanmin(lon_values)) >= 0.0:
        return NINO34["lon_min_360"], NINO34["lon_max_360"]
    return NINO34["lon_min_360"] - 360.0, NINO34["lon_max_360"] - 360.0


def region_mask(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.DataArray:
    lon_min, lon_max = lon_bounds_for_dataset(ds, lon_name)
    lat_mask = (ds[lat_name] >= NINO34["lat_min"]) & (ds[lat_name] <= NINO34["lat_max"])
    lon_mask = (ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max)
    return lat_mask.broadcast_like(ds[lon_name]) & lon_mask.broadcast_like(ds[lat_name])


def build_2d_mask(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.DataArray:
    lat = ds[lat_name]
    lon = ds[lon_name]
    lon_grid, lat_grid = xr.broadcast(lon, lat)
    lon_min, lon_max = lon_bounds_for_dataset(ds, lon_name)
    return (
        (lat_grid >= NINO34["lat_min"])
        & (lat_grid <= NINO34["lat_max"])
        & (lon_grid >= lon_min)
        & (lon_grid <= lon_max)
    ).transpose(lat_name, lon_name)


def create_modified_dataset(ds: xr.Dataset, sst_var: str, delta_k: float) -> tuple[xr.Dataset, dict[str, Any]]:
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    mask2d = build_2d_mask(ds, lat_name, lon_name)
    if int(mask2d.sum().item()) == 0:
        raise ValueError("Niño3.4 mask selected zero grid cells.")
    sst = ds[sst_var]
    mask3d = mask2d.broadcast_like(sst)
    perturbed_only = (sst + delta_k).where(mask3d)
    clipped_perturbed_only = perturbed_only.clip(min=SST_CLIP_RANGE_K[0], max=SST_CLIP_RANGE_K[1])
    clipped = xr.where(mask3d, clipped_perturbed_only, sst)

    clipping_mask = np.asarray(np.not_equal(perturbed_only.values, clipped_perturbed_only.values))
    clip_count = int(clipping_mask.sum())

    out = ds.copy(deep=False)
    out[sst_var] = clipped
    out[sst_var].attrs = dict(sst.attrs)
    out[sst_var].encoding = dict(sst.encoding)

    metadata = {
        "mask_grid_cell_count": int(mask2d.sum().item()),
        "delta_k": delta_k,
        "clipped_value_count": clip_count,
        "clip_range_k": list(SST_CLIP_RANGE_K),
    }
    return out, metadata


def validate_output(
    control_path: Path,
    output_path: Path,
    sst_var: str,
    perturbation_name: str,
    expected_delta_k: float,
) -> dict[str, Any]:
    with xr.open_dataset(control_path) as control, xr.open_dataset(output_path) as perturbed:
        lat_name = "lat" if "lat" in control.coords else "latitude"
        lon_name = "lon" if "lon" in control.coords else "longitude"
        mask2d = build_2d_mask(control, lat_name, lon_name)
        mask3d = mask2d.broadcast_like(control[sst_var])

        sst_diff = perturbed[sst_var] - control[sst_var]
        inside = sst_diff.where(mask3d)
        outside = sst_diff.where(~mask3d)

        non_sst_results: dict[str, bool] = {}
        all_non_sst_unchanged = True
        for name in control.data_vars:
            if name == sst_var:
                continue
            unchanged = bool(control[name].identical(perturbed[name]))
            non_sst_results[name] = unchanged
            all_non_sst_unchanged = all_non_sst_unchanged and unchanged

        max_abs = float(np.nanmax(np.abs(sst_diff.values)))
        mean_inside = float(np.nanmean(inside.values))
        outside_values = outside.values
        finite_outside = np.isfinite(outside_values)
        mean_outside = float(np.nanmean(outside_values)) if finite_outside.any() else 0.0
        modified_grid_cells = int(np.sum(np.abs(np.nan_to_num(inside.values)) > 0.0))

        return {
            "control_path": str(control_path),
            "perturbed_path": str(output_path),
            "perturbation_name": perturbation_name,
            "expected_delta_k": expected_delta_k,
            "max_sst_difference": max_abs,
            "mean_sst_difference_inside_nino34": mean_inside,
            "mean_sst_difference_outside_nino34": mean_outside,
            "modified_grid_cells": modified_grid_cells,
            "all_non_sst_variables_exactly_unchanged": all_non_sst_unchanged,
            "non_sst_variable_exact_unchanged": non_sst_results,
        }


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for entry in report["created_files"]:
        lines.extend(
            [
                f"input_path: {entry['input_path']}",
                f"output_path: {entry['output_path']}",
                f"perturbation_name: {entry['perturbation_name']}",
                f"sst_variable: {entry['sst_variable']}",
                f"delta_k: {entry['delta_k']}",
                f"mask_grid_cell_count: {entry['mask_grid_cell_count']}",
                f"clipped_value_count: {entry['clipped_value_count']}",
                f"clip_range_k: {entry['clip_range_k']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    created_files: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.validation_csv.parent.mkdir(parents=True, exist_ok=True)
    args.validation_json.parent.mkdir(parents=True, exist_ok=True)

    for perturbation_name, delta_k in PERTURBATIONS.items():
        perturbation_dir = args.output_root / perturbation_name
        perturbation_dir.mkdir(parents=True, exist_ok=True)
        for input_path in args.inputs:
            with xr.open_dataset(input_path) as ds:
                if args.sst_var not in ds.data_vars:
                    raise KeyError(f"{args.sst_var} not found in {input_path}")
                modified_ds, metadata = create_modified_dataset(ds, args.sst_var, delta_k)
                output_path = perturbation_dir / input_path.name
                modified_ds.to_netcdf(output_path)

            created_files.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "perturbation_name": perturbation_name,
                    "sst_variable": args.sst_var,
                    **metadata,
                }
            )
            validation_rows.append(
                validate_output(
                    control_path=input_path,
                    output_path=output_path,
                    sst_var=args.sst_var,
                    perturbation_name=perturbation_name,
                    expected_delta_k=delta_k,
                )
            )

    report = {"created_files": created_files}
    args.report_txt.write_text(render_text(report), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(validation_rows).to_csv(args.validation_csv, index=False)
    args.validation_json.write_text(json.dumps({"validations": validation_rows}, indent=2) + "\n", encoding="utf-8")
    print(render_text(report))
    print(pd.DataFrame(validation_rows).to_csv(index=False).strip())


if __name__ == "__main__":
    main()
