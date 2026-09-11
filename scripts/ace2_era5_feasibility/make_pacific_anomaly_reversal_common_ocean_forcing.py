#!/usr/bin/env python3
"""Create Pacific anomaly-reversal forcing with a conservative common-ocean mask."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    return parser.parse_args()


def build_domain_mask(ds: xr.Dataset) -> xr.DataArray:
    lat_grid, lon_grid = xr.broadcast(ds["latitude"], ds["longitude"])
    return (
        (lat_grid >= PACIFIC_DOMAIN["lat_min"])
        & (lat_grid <= PACIFIC_DOMAIN["lat_max"])
        & (lon_grid >= PACIFIC_DOMAIN["lon_min_360"])
        & (lon_grid <= PACIFIC_DOMAIN["lon_max_360"])
    ).transpose("latitude", "longitude")


def build_ace2_ocean_mask(ds: xr.Dataset) -> xr.DataArray:
    return ds["ocean_fraction"].isel(time=0) > PACIFIC_DOMAIN["ocean_fraction_threshold"]


def monthly_climatology_for_times(clim: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    months = xr.DataArray(times.dt.month.values.astype(np.int32), coords={"time": times}, dims=("time",))
    return clim.sel(month=months).assign_coords(time=times)


def suspicious_counts(values: np.ndarray) -> dict[str, int]:
    finite = np.isfinite(values)
    return {
        "count_below_180K": int(np.sum(finite & (values < 180.0))),
        "count_above_330K": int(np.sum(finite & (values > 330.0))),
        "count_below_271_35K": int(np.sum(finite & (values < 271.35))),
        "count_above_310K": int(np.sum(finite & (values > 310.0))),
    }


def state_surface_temperature(state: str, actual: xr.DataArray, clim_for_time: xr.DataArray, mask3d: xr.DataArray) -> xr.DataArray:
    if state == "actual":
        return actual.astype(actual.dtype)
    if state == "neutral":
        return xr.where(mask3d, clim_for_time, actual).astype(actual.dtype)
    if state == "reversed":
        return xr.where(mask3d, 2.0 * clim_for_time - actual, actual).astype(actual.dtype)
    raise ValueError(f"Unknown state {state}")


def arrays_equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def make_state_file(input_path: Path, output_path: Path, state: str, clim_ds: xr.Dataset) -> dict[str, Any]:
    with xr.open_dataset(input_path) as ds:
        clim = clim_ds["surface_temperature_climatology"]
        domain_mask = build_domain_mask(ds)
        ace2_ocean_mask = build_ace2_ocean_mask(ds)
        clim_for_time = monthly_climatology_for_times(clim, ds["time"]).astype(ds["surface_temperature"].dtype)
        finite_clim_mask = xr.apply_ufunc(np.isfinite, clim_for_time)
        common_mask3d = (domain_mask & ace2_ocean_mask).broadcast_like(ds["surface_temperature"]) & finite_clim_mask
        new_sst = state_surface_temperature(state, ds["surface_temperature"], clim_for_time, common_mask3d)
        new_values = np.asarray(new_sst.values, dtype=ds["surface_temperature"].dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    shutil.copy2(input_path, output_path)
    with netCDF4.Dataset(output_path, mode="r+") as nc_out:
        nc_out.variables["surface_temperature"][:] = new_values

    return validate_state(input_path, output_path, state, clim_ds)


def validate_state(input_path: Path, output_path: Path, state: str, clim_ds: xr.Dataset) -> dict[str, Any]:
    with xr.open_dataset(input_path) as control, xr.open_dataset(output_path) as candidate:
        clim = clim_ds["surface_temperature_climatology"]
        domain_mask = build_domain_mask(control)
        ace2_ocean_mask = build_ace2_ocean_mask(control)
        clim_for_time = monthly_climatology_for_times(clim, control["time"]).astype(control["surface_temperature"].dtype)
        finite_clim_mask = xr.apply_ufunc(np.isfinite, clim_for_time)
        base_mask2d = domain_mask & ace2_ocean_mask
        common_mask3d = base_mask2d.broadcast_like(control["surface_temperature"]) & finite_clim_mask
        excluded_nan_mask3d = base_mask2d.broadcast_like(control["surface_temperature"]) & ~finite_clim_mask

        control_sst = control["surface_temperature"]
        candidate_sst = candidate["surface_temperature"]
        expected = state_surface_temperature(state, control_sst, clim_for_time, common_mask3d)
        error = (candidate_sst - expected).where(common_mask3d)

        unchanged_flags: dict[str, bool] = {}
        all_non_sst_unchanged = True
        for name in control.data_vars:
            if name == "surface_temperature":
                continue
            same = arrays_equal_with_nan(np.asarray(control[name].values), np.asarray(candidate[name].values))
            unchanged_flags[name] = same
            all_non_sst_unchanged = all_non_sst_unchanged and same

        candidate_values = np.asarray(candidate_sst.values, dtype=float)
        finite_candidate = np.isfinite(candidate_values)
        common_count_2d_by_month = {}
        excluded_count_2d_by_month = {}
        for month in np.asarray(clim_ds["month"].values):
            month_mask = np.asarray(clim.sel(month=month).notnull().values, dtype=bool)
            common_count_2d_by_month[int(month)] = int(np.sum(np.asarray(base_mask2d.values, dtype=bool) & month_mask))
            excluded_count_2d_by_month[int(month)] = int(np.sum(np.asarray(base_mask2d.values, dtype=bool) & ~month_mask))

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "state": state,
            "surface_temperature_nan_count": int(np.isnan(candidate_values).sum()),
            "surface_temperature_inf_count": int(np.isinf(candidate_values).sum()),
            "surface_temperature_finite_min": float(np.nanmin(candidate_values)) if finite_candidate.any() else None,
            "surface_temperature_finite_max": float(np.nanmax(candidate_values)) if finite_candidate.any() else None,
            "surface_temperature_finite_mean": float(np.nanmean(candidate_values)) if finite_candidate.any() else None,
            "surface_temperature_unchanged_outside_common_ocean_mask": arrays_equal_with_nan(
                np.asarray(candidate_sst.where(~common_mask3d).values),
                np.asarray(control_sst.where(~common_mask3d).values),
            ),
            "all_non_sst_variables_exactly_unchanged": all_non_sst_unchanged,
            "non_sst_variable_exact_unchanged": unchanged_flags,
            "common_ocean_mask_value_count_3d": int(np.sum(np.asarray(common_mask3d.values, dtype=bool))),
            "common_ocean_mask_union_cell_count_2d": int(np.sum(np.any(np.asarray(common_mask3d.values, dtype=bool), axis=0))),
            "common_ocean_mask_cell_count_2d_by_month": common_count_2d_by_month,
            "excluded_ace2_ocean_but_nan_cobe2_value_count_3d": int(np.sum(np.asarray(excluded_nan_mask3d.values, dtype=bool))),
            "excluded_ace2_ocean_but_nan_cobe2_cell_count_2d_union": int(np.sum(np.any(np.asarray(excluded_nan_mask3d.values, dtype=bool), axis=0))),
            "excluded_ace2_ocean_but_nan_cobe2_cell_count_2d_by_month": excluded_count_2d_by_month,
            "formula_max_abs_error_inside_common_ocean_mask": float(np.nanmax(np.abs(error.values))) if np.isfinite(error.values).any() else 0.0,
            "formula_mean_abs_error_inside_common_ocean_mask": float(np.nanmean(np.abs(error.values))) if np.isfinite(error.values).any() else 0.0,
            "physically_impossible_sst": suspicious_counts(candidate_values),
            "no_physically_impossible_sst_values": int(np.sum((candidate_values < 180.0) | (candidate_values > 330.0))) == 0,
            "mask_definition": {
                **PACIFIC_DOMAIN,
                "finite_climatology_required": True,
            },
        }


def main() -> None:
    args = parse_args()
    args.validation_csv.parent.mkdir(parents=True, exist_ok=True)
    args.validation_json.parent.mkdir(parents=True, exist_ok=True)

    validations: list[dict[str, Any]] = []
    with xr.open_dataset(args.climatology) as clim_ds:
        if "surface_temperature_climatology" not in clim_ds.data_vars:
            raise KeyError("Expected surface_temperature_climatology in climatology file.")

        for state in STATE_NAMES:
            state_dir = args.output_root / state
            for input_path in args.inputs:
                validations.append(make_state_file(input_path, state_dir / input_path.name, state, clim_ds))

    pd.DataFrame(validations).to_csv(args.validation_csv, index=False)
    args.validation_json.write_text(json.dumps({"validations": validations}, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(validations).to_csv(index=False).strip())


if __name__ == "__main__":
    main()
