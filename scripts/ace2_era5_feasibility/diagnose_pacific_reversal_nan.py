#!/usr/bin/env python3
"""Diagnose NaN propagation in the ACE2 Pacific anomaly-reversal experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
ACE2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2")
ACE2_ASSETS = ACE2_ROOT / "assets"
ACE2_OUTPUTS = ACE2_ROOT / "outputs"
REPORT_DIR = REPO / "artifacts" / "ace2_era5_feasibility"

PACIFIC_DOMAIN = {
    "lat_min": -10.0,
    "lat_max": 60.0,
    "lon_min_360": 120.0,
    "lon_max_360": 280.0,
    "ocean_fraction_threshold": 0.5,
}

CASES = {
    "actual": {
        "forcing": [
            ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020" / "actual" / "forcing_2020.nc",
            ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020" / "actual" / "forcing_2021.nc",
        ],
        "output_dir": ACE2_OUTPUTS / "pacific_anomaly_reversal_2020" / "actual_season",
    },
    "neutral": {
        "forcing": [
            ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020" / "neutral" / "forcing_2020.nc",
            ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020" / "neutral" / "forcing_2021.nc",
        ],
        "output_dir": ACE2_OUTPUTS / "pacific_anomaly_reversal_2020" / "neutral_season",
    },
    "reversed": {
        "forcing": [
            ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020" / "reversed" / "forcing_2020.nc",
            ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020" / "reversed" / "forcing_2021.nc",
        ],
        "output_dir": ACE2_OUTPUTS / "pacific_anomaly_reversal_2020" / "reversed_season",
    },
}

ORIGINAL_FORCING = {
    "forcing_2020.nc": ACE2_ASSETS / "forcing" / "forcing_2020.nc",
    "forcing_2021.nc": ACE2_ASSETS / "forcing" / "forcing_2021.nc",
}

CLIMATOLOGY_PATH = (
    ACE2_ASSETS
    / "climatology"
    / "pacific_anomaly_reversal_2020"
    / "cobe2_monthly_surface_temperature_climatology_on_ace2_grid.nc"
)
CLIMATOLOGY_VAR = "surface_temperature_climatology"

FOCUS_FORCING_VARS = ("surface_temperature", "sea_ice_fraction", "ocean_fraction", "land_fraction")
TARGET_OUTPUT_VARS = ("TMP2m", "PRATEsfc", "VGRD10m")
OUTPUT_FILE_ORDER = (
    "autoregressive_predictions.nc",
    "monthly_mean_predictions.nc",
    "time_mean_diagnostics.nc",
    "restart.nc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-txt", type=Path, default=REPORT_DIR / "pacific_reversal_nan_diagnostic.txt")
    parser.add_argument("--report-json", type=Path, default=REPORT_DIR / "pacific_reversal_nan_diagnostic.json")
    parser.add_argument("--forcing-csv", type=Path, default=REPORT_DIR / "pacific_reversal_nan_forcing_stats.csv")
    parser.add_argument("--output-csv", type=Path, default=REPORT_DIR / "pacific_reversal_nan_output_stats.csv")
    parser.add_argument(
        "--final-report-md",
        type=Path,
        default=REPORT_DIR / "milestone7_nan_propagation_diagnostic_report.md",
    )
    return parser.parse_args()


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, type(np.dtype("float32"))):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_encoding(da: xr.DataArray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in da.encoding.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            out[key] = [to_native(v) for v in value]
        else:
            out[key] = to_native(value)
    return out


def build_pacific_mask(ds: xr.Dataset) -> xr.DataArray:
    lat_grid, lon_grid = xr.broadcast(ds["latitude"], ds["longitude"])
    latlon_mask = (
        (lat_grid >= PACIFIC_DOMAIN["lat_min"])
        & (lat_grid <= PACIFIC_DOMAIN["lat_max"])
        & (lon_grid >= PACIFIC_DOMAIN["lon_min_360"])
        & (lon_grid <= PACIFIC_DOMAIN["lon_max_360"])
    ).transpose("latitude", "longitude")
    ocean_mask = ds["ocean_fraction"].isel(time=0) > PACIFIC_DOMAIN["ocean_fraction_threshold"]
    mask2d = latlon_mask & ocean_mask
    if int(mask2d.sum().item()) == 0:
        raise ValueError("Pacific mask selected zero grid cells.")
    return mask2d


def boundary_mask(mask2d: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask2d, dtype=bool)
    inside = mask2d
    candidates = (
        np.roll(inside, 1, axis=0),
        np.roll(inside, -1, axis=0),
        np.roll(inside, 1, axis=1),
        np.roll(inside, -1, axis=1),
    )
    for shifted in candidates:
        boundary |= inside & ~shifted
    boundary[0, :] = inside[0, :]
    boundary[-1, :] = inside[-1, :]
    boundary[:, 0] |= inside[:, 0]
    boundary[:, -1] |= inside[:, -1]
    return boundary


def flatten_first_location(mask: np.ndarray, coords: dict[str, np.ndarray], dims: Iterable[str]) -> dict[str, Any] | None:
    if not mask.any():
        return None
    flat_index = int(np.flatnonzero(mask.ravel())[0])
    multi_index = np.unravel_index(flat_index, mask.shape)
    location = {"flat_index": flat_index}
    for dim, idx in zip(dims, multi_index, strict=True):
        location[f"{dim}_index"] = int(idx)
        coord_values = coords.get(dim)
        if coord_values is not None:
            location[dim] = to_native(coord_values[idx])
    return location


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    if not (
        np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        return {
            "total_size": int(values.size),
            "total_nan_count": None,
            "total_inf_count": None,
            "finite_count": None,
            "finite_min": None,
            "finite_max": None,
            "finite_mean": None,
            "non_numeric_dtype": str(values.dtype),
        }
    finite = np.isfinite(values)
    stats = {
        "total_size": int(values.size),
        "total_nan_count": int(np.isnan(values).sum()),
        "total_inf_count": int(np.isinf(values).sum()),
        "finite_count": int(finite.sum()),
    }
    if finite.any():
        stats["finite_min"] = float(np.nanmin(values))
        stats["finite_max"] = float(np.nanmax(values))
        stats["finite_mean"] = float(np.nanmean(values))
    else:
        stats["finite_min"] = None
        stats["finite_max"] = None
        stats["finite_mean"] = None
    return stats


def month_from_time_values(time_values: np.ndarray) -> list[int]:
    return [int(v) for v in pd.DatetimeIndex(time_values).month.tolist()]


def location_for_extreme_diff(
    diff: np.ndarray,
    base_coords: dict[str, np.ndarray],
    dims: tuple[str, ...],
) -> dict[str, Any] | None:
    finite = np.isfinite(diff)
    if not finite.any():
        return None
    abs_diff = np.abs(diff)
    abs_diff = np.where(finite, abs_diff, -np.inf)
    flat_index = int(np.argmax(abs_diff))
    if not np.isfinite(abs_diff.ravel()[flat_index]):
        return None
    multi_index = np.unravel_index(flat_index, diff.shape)
    location = {
        "flat_index": flat_index,
        "abs_difference": float(abs_diff.ravel()[flat_index]),
        "difference": float(diff.ravel()[flat_index]),
    }
    for dim, idx in zip(dims, multi_index, strict=True):
        location[f"{dim}_index"] = int(idx)
        coord_values = base_coords.get(dim)
        if coord_values is not None:
            location[dim] = to_native(coord_values[idx])
    return location


def summarize_monthly_surface_temperature(da: xr.DataArray) -> list[dict[str, Any]]:
    months = pd.DatetimeIndex(da["time"].values).month
    values = da.values
    summaries: list[dict[str, Any]] = []
    for month in np.unique(months):
        month_values = values[months == month]
        finite = np.isfinite(month_values)
        entry = {"month": int(month), "sample_count": int(month_values.size)}
        if finite.any():
            entry.update(
                {
                    "finite_min": float(np.nanmin(month_values)),
                    "finite_max": float(np.nanmax(month_values)),
                    "finite_mean": float(np.nanmean(month_values)),
                    "nan_count": int(np.isnan(month_values).sum()),
                    "inf_count": int(np.isinf(month_values).sum()),
                }
            )
        else:
            entry.update(
                {
                    "finite_min": None,
                    "finite_max": None,
                    "finite_mean": None,
                    "nan_count": int(np.isnan(month_values).sum()),
                    "inf_count": int(np.isinf(month_values).sum()),
                }
            )
        summaries.append(entry)
    return summaries


def forcing_var_summary(
    case_name: str,
    year_path: Path,
    original_path: Path,
    var_name: str,
    mask2d: xr.DataArray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with xr.open_dataset(year_path) as ds, xr.open_dataset(original_path) as original_ds:
        da = ds[var_name]
        original = original_ds[var_name]
        values = np.asarray(da.values)
        original_values = np.asarray(original.values)
        stats = finite_stats(values)
        original_finite = np.isfinite(original_values)
        if original_finite.any():
            original_min = float(np.nanmin(original_values))
            original_max = float(np.nanmax(original_values))
        else:
            original_min = None
            original_max = None
        outside_original_range = np.isfinite(values)
        if original_min is not None and original_max is not None:
            outside_original_range &= (values < original_min) | (values > original_max)
            outside_original_range_count = int(outside_original_range.sum())
            first_outside_original_range = flatten_first_location(
                outside_original_range,
                {
                    dim: np.asarray(ds[dim].values) for dim in da.dims if dim in ds.coords
                },
                da.dims,
            )
        else:
            outside_original_range_count = 0
            first_outside_original_range = None

        mask3d = mask2d.broadcast_like(ds["surface_temperature"]) if "time" in da.dims else mask2d
        mask_values = np.asarray(mask3d.values if "time" in da.dims else mask2d.values)
        pacific_nan_count = int(np.isnan(values[mask_values]).sum()) if values.shape == mask_values.shape else None
        non_pacific_nan_count = int(np.isnan(values[~mask_values]).sum()) if values.shape == mask_values.shape else None

        coords = {dim: np.asarray(ds[dim].values) for dim in da.dims if dim in ds.coords}
        detailed = {
            "case": case_name,
            "forcing_path": str(year_path),
            "original_path": str(original_path),
            "variable": var_name,
            "dtype": str(da.dtype),
            "dims": list(da.dims),
            "shape": [int(v) for v in da.shape],
            "encoding": serialize_encoding(da),
            "_FillValue": to_native(da.encoding.get("_FillValue", da.attrs.get("_FillValue"))),
            "missing_value": to_native(da.attrs.get("missing_value")),
            "attrs": {k: to_native(v) for k, v in da.attrs.items()},
            "original_finite_min": original_min,
            "original_finite_max": original_max,
            "outside_original_range_count": outside_original_range_count,
            "first_nan_location": flatten_first_location(np.isnan(values), coords, da.dims),
            "first_inf_location": flatten_first_location(np.isinf(values), coords, da.dims),
            "first_outside_original_range_location": first_outside_original_range,
            "pacific_ocean_mask_nan_count": pacific_nan_count,
            "non_pacific_nan_count": non_pacific_nan_count,
        }
        detailed.update(stats)

        if var_name == "surface_temperature":
            non_sst_exact_flags = {}
            all_non_sst_same = True
            for other_name in ds.data_vars:
                if other_name == "surface_temperature":
                    continue
                same = bool(np.array_equal(np.asarray(ds[other_name].values), np.asarray(original_ds[other_name].values), equal_nan=True))
                non_sst_exact_flags[other_name] = same
                all_non_sst_same = all_non_sst_same and same

            outside_mask_equal = np.array_equal(
                np.asarray(da.where(~mask3d).values),
                np.asarray(original.where(~mask3d).values),
                equal_nan=True,
            )
            detailed["all_non_surface_temperature_variables_exactly_unchanged"] = all_non_sst_same
            detailed["non_surface_temperature_exact_flags"] = non_sst_exact_flags
            detailed["surface_temperature_outside_pacific_mask_exactly_unchanged"] = bool(outside_mask_equal)
            detailed["monthly_surface_temperature_stats"] = summarize_monthly_surface_temperature(da)

        csv_row = {
            "case": case_name,
            "forcing_file": year_path.name,
            "variable": var_name,
            "dtype": str(da.dtype),
            "dims": "|".join(da.dims),
            "shape": "x".join(str(v) for v in da.shape),
            "total_nan_count": detailed["total_nan_count"],
            "total_inf_count": detailed["total_inf_count"],
            "finite_min": detailed["finite_min"],
            "finite_max": detailed["finite_max"],
            "finite_mean": detailed["finite_mean"],
            "outside_original_range_count": outside_original_range_count,
            "pacific_ocean_mask_nan_count": pacific_nan_count,
            "non_pacific_nan_count": non_pacific_nan_count,
            "fill_value": detailed["_FillValue"],
            "missing_value": detailed["missing_value"],
            "all_non_surface_temperature_variables_exactly_unchanged": detailed.get(
                "all_non_surface_temperature_variables_exactly_unchanged"
            ),
            "surface_temperature_outside_pacific_mask_exactly_unchanged": detailed.get(
                "surface_temperature_outside_pacific_mask_exactly_unchanged"
            ),
        }
        return detailed, csv_row


def summarize_climatology() -> dict[str, Any]:
    with xr.open_dataset(CLIMATOLOGY_PATH) as clim_ds, xr.open_dataset(ORIGINAL_FORCING["forcing_2020.nc"]) as template_ds:
        clim = clim_ds[CLIMATOLOGY_VAR]
        mask2d = build_pacific_mask(template_ds)
        boundary2d = boundary_mask(np.asarray(mask2d.values))
        clim_values = np.asarray(clim.values)
        mask3d = np.broadcast_to(np.asarray(mask2d.values), clim_values.shape)
        boundary3d = np.broadcast_to(boundary2d, clim_values.shape)
        month_summaries = []
        for month in np.asarray(clim["month"].values):
            month_da = clim.sel(month=month)
            values = np.asarray(month_da.values)
            finite = np.isfinite(values)
            month_summaries.append(
                {
                    "month": int(month),
                    "finite_min": float(np.nanmin(values)) if finite.any() else None,
                    "finite_max": float(np.nanmax(values)) if finite.any() else None,
                    "finite_mean": float(np.nanmean(values)) if finite.any() else None,
                    "nan_count": int(np.isnan(values).sum()),
                }
            )
        return {
            "climatology_path": str(CLIMATOLOGY_PATH),
            "climatology_variable": CLIMATOLOGY_VAR,
            "dtype": str(clim.dtype),
            "dims": list(clim.dims),
            "shape": [int(v) for v in clim.shape],
            "encoding": serialize_encoding(clim),
            "_FillValue": to_native(clim.encoding.get("_FillValue", clim.attrs.get("_FillValue"))),
            "missing_value": to_native(clim.attrs.get("missing_value")),
            "total_nan_count": int(np.isnan(clim_values).sum()),
            "pacific_ocean_mask_nan_count": int(np.isnan(clim_values[mask3d]).sum()),
            "pacific_boundary_band_nan_count": int(np.isnan(clim_values[boundary3d]).sum()),
            "month_stats": month_summaries,
            "interpolation_holes_near_boundary": bool(np.isnan(clim_values[boundary3d]).any()),
            "mask_definition": PACIFIC_DOMAIN,
        }


def difference_summary(
    label: str,
    candidate_path: Path,
    actual_path: Path,
    mask2d: xr.DataArray,
) -> dict[str, Any]:
    with xr.open_dataset(candidate_path) as candidate_ds, xr.open_dataset(actual_path) as actual_ds:
        cand = candidate_ds["surface_temperature"]
        actual = actual_ds["surface_temperature"]
        diff = np.asarray((cand - actual).where(mask2d.broadcast_like(cand)).values, dtype=float)
        finite = np.isfinite(diff)
        pct = None
        largest = None
        if finite.any():
            finite_values = diff[finite]
            pct = {
                "p01": float(np.percentile(finite_values, 1)),
                "p05": float(np.percentile(finite_values, 5)),
                "p50": float(np.percentile(finite_values, 50)),
                "p95": float(np.percentile(finite_values, 95)),
                "p99": float(np.percentile(finite_values, 99)),
            }
            largest = location_for_extreme_diff(
                diff,
                {
                    "time": np.asarray(candidate_ds["time"].values),
                    "latitude": np.asarray(candidate_ds["latitude"].values),
                    "longitude": np.asarray(candidate_ds["longitude"].values),
                },
                ("time", "latitude", "longitude"),
            )

        finite_counts = np.sum(np.isfinite(diff), axis=0)
        diff_sum = np.nansum(diff, axis=0)
        diff_mean_by_cell = np.full(diff.shape[1:], np.nan, dtype=float)
        valid_cells = finite_counts > 0
        diff_mean_by_cell[valid_cells] = diff_sum[valid_cells] / finite_counts[valid_cells]
        mask_values = np.asarray(mask2d.values)
        boundary2d = boundary_mask(mask_values)
        discontinuity = np.nanmax(np.abs(diff_mean_by_cell[boundary2d])) if np.isfinite(diff_mean_by_cell[boundary2d]).any() else None
        candidate_inside = np.asarray(cand.where(mask2d.broadcast_like(cand)).values, dtype=float)
        suspicious = {
            "count_below_180K": int(np.sum(np.isfinite(candidate_inside) & (candidate_inside < 180.0))),
            "count_above_330K": int(np.sum(np.isfinite(candidate_inside) & (candidate_inside > 330.0))),
            "count_below_271_35K": int(np.sum(np.isfinite(candidate_inside) & (candidate_inside < 271.35))),
            "count_above_310K": int(np.sum(np.isfinite(candidate_inside) & (candidate_inside > 310.0))),
        }
        return {
            "label": label,
            "candidate_path": str(candidate_path),
            "actual_path": str(actual_path),
            "finite_count": int(finite.sum()),
            "min_difference": float(np.nanmin(diff)) if finite.any() else None,
            "max_difference": float(np.nanmax(diff)) if finite.any() else None,
            "mean_difference": float(np.nanmean(diff)) if finite.any() else None,
            "percentiles": pct,
            "largest_absolute_difference_location": largest,
            "boundary_discontinuity_max_abs_time_mean_difference": float(discontinuity) if discontinuity is not None else None,
            "physically_suspicious_value_counts_inside_pacific": suspicious,
        }


def output_var_summary(case_name: str, output_path: Path, file_name: str, var_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with xr.open_dataset(output_path) as ds:
        da = ds[var_name]
        values = np.asarray(da.values)
        stats = finite_stats(values)
        if stats.get("non_numeric_dtype") is not None:
            detail = {
                "case": case_name,
                "file": file_name,
                "path": str(output_path),
                "variable": var_name,
                "dtype": str(da.dtype),
                "dims": list(da.dims),
                "shape": [int(v) for v in da.shape],
                "attrs": {k: to_native(v) for k, v in da.attrs.items()},
                "encoding": serialize_encoding(da),
                "skipped_non_numeric": True,
            }
            detail.update(stats)
            csv_row = {
                "case": case_name,
                "file": file_name,
                "variable": var_name,
                "dtype": str(da.dtype),
                "dims": "|".join(da.dims),
                "shape": "x".join(str(v) for v in da.shape),
                "skipped_non_numeric": True,
            }
            return detail, csv_row
        coords = {dim: np.asarray(ds[dim].values) for dim in da.dims if dim in ds.coords}
        detail = {
            "case": case_name,
            "file": file_name,
            "path": str(output_path),
            "variable": var_name,
            "dtype": str(da.dtype),
            "dims": list(da.dims),
            "shape": [int(v) for v in da.shape],
            "attrs": {k: to_native(v) for k, v in da.attrs.items()},
            "encoding": serialize_encoding(da),
            "first_nan_location": flatten_first_location(np.isnan(values), coords, da.dims),
            "first_inf_location": flatten_first_location(np.isinf(values), coords, da.dims),
        }
        detail.update(stats)

        if "time" in da.dims:
            time_axis = da.dims.index("time")
            finite_mask = np.isfinite(values)
            any_bad_by_time = ~finite_mask.all(axis=tuple(i for i in range(values.ndim) if i != time_axis))
            bad_times = np.flatnonzero(any_bad_by_time)
            detail["first_bad_time_index"] = int(bad_times[0]) if bad_times.size else None
            if bad_times.size:
                first_bad_slice = np.take(values, int(bad_times[0]), axis=time_axis)
                detail["first_bad_time_all_values_nonfinite"] = bool(~np.isfinite(first_bad_slice).any())
            else:
                detail["first_bad_time_all_values_nonfinite"] = False
            detail["time_zero_has_any_nan"] = bool(any_bad_by_time[0]) if any_bad_by_time.size else False
            first_bad_index = detail["first_bad_time_index"]
            if first_bad_index is not None and first_bad_index > 0:
                before = np.take(values, first_bad_index - 1, axis=time_axis)
                if np.isfinite(before).any():
                    detail["finite_min_before_first_bad_time"] = float(np.nanmin(before))
                    detail["finite_max_before_first_bad_time"] = float(np.nanmax(before))
                else:
                    detail["finite_min_before_first_bad_time"] = None
                    detail["finite_max_before_first_bad_time"] = None
            else:
                detail["finite_min_before_first_bad_time"] = None
                detail["finite_max_before_first_bad_time"] = None
        else:
            detail["first_bad_time_index"] = None
            detail["first_bad_time_all_values_nonfinite"] = bool(not np.isfinite(values).any()) if (np.isnan(values).any() or np.isinf(values).any()) else False
            detail["time_zero_has_any_nan"] = None
            detail["finite_min_before_first_bad_time"] = None
            detail["finite_max_before_first_bad_time"] = None

        csv_row = {
            "case": case_name,
            "file": file_name,
            "variable": var_name,
            "dtype": str(da.dtype),
            "dims": "|".join(da.dims),
            "shape": "x".join(str(v) for v in da.shape),
            "total_nan_count": detail["total_nan_count"],
            "total_inf_count": detail["total_inf_count"],
            "finite_min": detail["finite_min"],
            "finite_max": detail["finite_max"],
            "finite_mean": detail["finite_mean"],
            "first_bad_time_index": detail["first_bad_time_index"],
            "time_zero_has_any_nan": detail["time_zero_has_any_nan"],
            "first_bad_time_all_values_nonfinite": detail["first_bad_time_all_values_nonfinite"],
        }
        return detail, csv_row


def inspect_outputs(case_name: str, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    details: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    first_nan_file = None
    first_nan_var = None
    first_nan_time = None

    for file_name in OUTPUT_FILE_ORDER:
        output_path = output_dir / file_name
        if not output_path.exists():
            continue
        with xr.open_dataset(output_path) as ds:
            candidate_vars = [
                name for name in ds.data_vars if np.issubdtype(ds[name].dtype, np.number) or np.issubdtype(ds[name].dtype, np.bool_)
            ]
        ordered_vars = [v for v in TARGET_OUTPUT_VARS if v in candidate_vars] + [v for v in candidate_vars if v not in TARGET_OUTPUT_VARS]
        for var_name in ordered_vars:
            detail, csv_row = output_var_summary(case_name, output_path, file_name, var_name)
            details.append(detail)
            csv_rows.append(csv_row)
            if first_nan_file is None and (detail["total_nan_count"] > 0 or detail["total_inf_count"] > 0):
                first_nan_file = file_name
                first_nan_var = var_name
                first_nan_time = detail["first_bad_time_index"]

    case_summary = {
        "case": case_name,
        "output_dir": str(output_dir),
        "first_output_file_containing_nonfinite": first_nan_file,
        "first_variable_containing_nonfinite": first_nan_var,
        "first_time_index_containing_nonfinite": first_nan_time,
    }
    if first_nan_file is not None:
        matches = [d for d in details if d["file"] == first_nan_file and d["variable"] == first_nan_var]
        if matches:
            match = matches[0]
            case_summary["time_zero_already_nonfinite"] = match["time_zero_has_any_nan"]
            case_summary["nonfinite_appears_everywhere_immediately"] = match["first_bad_time_all_values_nonfinite"]
            case_summary["finite_min_before_first_bad_time"] = match["finite_min_before_first_bad_time"]
            case_summary["finite_max_before_first_bad_time"] = match["finite_max_before_first_bad_time"]
    else:
        case_summary["time_zero_already_nonfinite"] = False
        case_summary["nonfinite_appears_everywhere_immediately"] = False
        case_summary["finite_min_before_first_bad_time"] = None
        case_summary["finite_max_before_first_bad_time"] = None
    return details, csv_rows, case_summary


def likely_cause(report: dict[str, Any]) -> tuple[str, str]:
    forcing_has_nonfinite = any(
        row["case"] in {"neutral", "reversed"} and row["variable"] == "surface_temperature" and (row["total_nan_count"] > 0 or row["total_inf_count"] > 0)
        for row in report["forcing_csv_rows"]
    )
    climatology_has_holes = report["climatology"]["total_nan_count"] > 0 or report["climatology"]["pacific_ocean_mask_nan_count"] > 0

    encoding_diff = False
    base_fill = {}
    for row in report["forcing_details"]:
        if row["variable"] != "surface_temperature":
            continue
        key = Path(row["forcing_path"]).name
        case = row["case"]
        base_fill.setdefault(key, {})
        base_fill[key][case] = (
            json.dumps(to_native(row["encoding"]), sort_keys=True),
            to_native(row["_FillValue"]),
            to_native(row["missing_value"]),
        )
    for year_map in base_fill.values():
        actual_sig = year_map.get("actual")
        for case in ("neutral", "reversed"):
            if case in year_map and actual_sig is not None and year_map[case] != actual_sig:
                encoding_diff = True

    neutral_case = report["output_case_summaries"]["neutral"]
    reversed_case = report["output_case_summaries"]["reversed"]
    actual_case = report["output_case_summaries"]["actual"]
    modified_outputs_nonfinite = neutral_case["first_output_file_containing_nonfinite"] is not None or reversed_case["first_output_file_containing_nonfinite"] is not None
    control_clean = actual_case["first_output_file_containing_nonfinite"] is None

    if forcing_has_nonfinite or climatology_has_holes:
        return "a) bad forcing NaNs", "Neutral/reversed forcing inherited nonfinite values before ACE2 ran."
    if encoding_diff and modified_outputs_nonfinite:
        return "b) encoding/fill-value corruption", "Modified forcing changed encoding or fill-value signatures relative to the working actual case."
    if modified_outputs_nonfinite and control_clean:
        reversed_suspicious = any(
            item["label"] == "reversed_minus_actual"
            and item["physically_suspicious_value_counts_inside_pacific"]["count_below_271_35K"] > 0
            for item in report["forcing_difference_summaries"]
        )
        if reversed_suspicious:
            return "c) physically unstable reversed SST", "The forcing stays finite, but the reversed Pacific SST state pushes ACE2 into a physically suspect regime."
        return "d) ACE2 out-of-distribution instability", "The forcing stays finite and encoding looks intact, but ACE2 still collapses to nonfinite output under modified Pacific SST states."
    return "e) unknown", "The available forcing and output evidence does not isolate a single failure mechanism."


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "ACE2 Pacific anomaly-reversal NaN diagnostic",
        "",
        f"climatology_path: {report['climatology']['climatology_path']}",
        f"climatology_total_nan_count: {report['climatology']['total_nan_count']}",
        f"climatology_pacific_ocean_mask_nan_count: {report['climatology']['pacific_ocean_mask_nan_count']}",
        f"climatology_boundary_band_nan_count: {report['climatology']['pacific_boundary_band_nan_count']}",
        "",
        "Output case summaries:",
    ]
    for case_name, case_summary in report["output_case_summaries"].items():
        lines.extend(
            [
                f"- {case_name}:",
                f"  first_output_file_containing_nonfinite: {case_summary['first_output_file_containing_nonfinite']}",
                f"  first_variable_containing_nonfinite: {case_summary['first_variable_containing_nonfinite']}",
                f"  first_time_index_containing_nonfinite: {case_summary['first_time_index_containing_nonfinite']}",
                f"  time_zero_already_nonfinite: {case_summary['time_zero_already_nonfinite']}",
                f"  nonfinite_appears_everywhere_immediately: {case_summary['nonfinite_appears_everywhere_immediately']}",
            ]
        )
    lines.extend(
        [
            "",
            "Forcing difference summaries:",
        ]
    )
    for diff in report["forcing_difference_summaries"]:
        lines.extend(
            [
                f"- {diff['label']}: mean={diff['mean_difference']}, min={diff['min_difference']}, max={diff['max_difference']}",
                f"  boundary_discontinuity_max_abs_time_mean_difference: {diff['boundary_discontinuity_max_abs_time_mean_difference']}",
                f"  largest_absolute_difference_location: {diff['largest_absolute_difference_location']}",
                f"  suspicious_counts: {diff['physically_suspicious_value_counts_inside_pacific']}",
            ]
        )
    lines.extend(
        [
            "",
            f"likely_cause: {report['diagnosis']['category']}",
            f"likely_cause_reason: {report['diagnosis']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_final_report(report: dict[str, Any], path: Path) -> None:
    forcing_nonfinite = {
        case: [
            row
            for row in report["forcing_csv_rows"]
            if row["case"] == case and row["variable"] == "surface_temperature"
        ]
        for case in CASES
    }
    actual_fill = next(row for row in report["forcing_details"] if row["case"] == "actual" and row["variable"] == "surface_temperature" and Path(row["forcing_path"]).name == "forcing_2020.nc")
    neutral_fill = next(row for row in report["forcing_details"] if row["case"] == "neutral" and row["variable"] == "surface_temperature" and Path(row["forcing_path"]).name == "forcing_2020.nc")
    reversed_fill = next(row for row in report["forcing_details"] if row["case"] == "reversed" and row["variable"] == "surface_temperature" and Path(row["forcing_path"]).name == "forcing_2020.nc")
    neutral_2020 = next(row for row in forcing_nonfinite["neutral"] if row["forcing_file"] == "forcing_2020.nc")
    neutral_2021 = next(row for row in forcing_nonfinite["neutral"] if row["forcing_file"] == "forcing_2021.nc")
    reversed_2020 = next(row for row in forcing_nonfinite["reversed"] if row["forcing_file"] == "forcing_2020.nc")
    reversed_2021 = next(row for row in forcing_nonfinite["reversed"] if row["forcing_file"] == "forcing_2021.nc")
    lines = [
        "# Milestone 7: Pacific anomaly-reversal NaN propagation diagnostic",
        "",
        "## Direct Answers",
        "",
        f"1. Did NaN/inf exist in neutral/reversed forcing before ACE2 ran?",
        f"   - Yes. Neutral `surface_temperature` already contains {int(neutral_2020['total_nan_count'])} NaNs in `forcing_2020.nc` and {int(neutral_2021['total_nan_count'])} NaNs in `forcing_2021.nc`; reversed contains {int(reversed_2020['total_nan_count'])} and {int(reversed_2021['total_nan_count'])} respectively.",
        f"2. Did climatology regridding introduce missing values?",
        f"   - {'Yes' if (report['climatology']['total_nan_count'] > 0 or report['climatology']['pacific_ocean_mask_nan_count'] > 0) else 'No'}; climatology has {report['climatology']['total_nan_count']} total NaNs, including {report['climatology']['pacific_ocean_mask_nan_count']} inside the Pacific mask and {report['climatology']['pacific_boundary_band_nan_count']} in the Pacific boundary band.",
        f"3. Did encoding/_FillValue differ between actual and modified files?",
        f"   - No clear evidence of an encoding-only failure. 2020 `surface_temperature` signatures are actual fill={actual_fill['_FillValue']}, neutral fill={neutral_fill['_FillValue']}, reversed fill={reversed_fill['_FillValue']}. Full encodings are in [pacific_reversal_nan_diagnostic.json]({path.parent / 'pacific_reversal_nan_diagnostic.json'}).",
        f"4. Did ACE2 create NaNs despite finite forcing?",
        f"   - ACE2 propagated the bad forcing immediately after startup: `actual` stayed finite, while `neutral` and `reversed` first became nonfinite in `autoregressive_predictions.nc` at forecast time index 1, with time index 0 still finite.",
        f"5. At what file/time/variable did NaNs first appear?",
        f"   - Neutral: file={report['output_case_summaries']['neutral']['first_output_file_containing_nonfinite']}, variable={report['output_case_summaries']['neutral']['first_variable_containing_nonfinite']}, time_index={report['output_case_summaries']['neutral']['first_time_index_containing_nonfinite']}.",
        f"   - Reversed: file={report['output_case_summaries']['reversed']['first_output_file_containing_nonfinite']}, variable={report['output_case_summaries']['reversed']['first_variable_containing_nonfinite']}, time_index={report['output_case_summaries']['reversed']['first_time_index_containing_nonfinite']}.",
        f"6. Likely cause?",
        f"   - {report['diagnosis']['category']}: {report['diagnosis']['reason']}",
        f"7. Recommended fix, without implementing it?",
        "   - Rebuild the Pacific climatology/regridding so `surface_temperature_climatology` is finite on every Pacific ocean cell that will be overwritten, or restrict the overwrite mask to cells with finite climatology values. Then revalidate that the modified forcing has zero NaNs before any new ACE2 run.",
        "",
        "## Evidence",
        "",
        f"- Diagnostic JSON: [pacific_reversal_nan_diagnostic.json]({path.parent / 'pacific_reversal_nan_diagnostic.json'})",
        f"- Diagnostic text: [pacific_reversal_nan_diagnostic.txt]({path.parent / 'pacific_reversal_nan_diagnostic.txt'})",
        f"- Forcing stats CSV: [pacific_reversal_nan_forcing_stats.csv]({path.parent / 'pacific_reversal_nan_forcing_stats.csv'})",
        f"- Output stats CSV: [pacific_reversal_nan_output_stats.csv]({path.parent / 'pacific_reversal_nan_output_stats.csv'})",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.forcing_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.final_report_md.parent.mkdir(parents=True, exist_ok=True)

    forcing_details: list[dict[str, Any]] = []
    forcing_csv_rows: list[dict[str, Any]] = []
    output_details: list[dict[str, Any]] = []
    output_csv_rows: list[dict[str, Any]] = []
    output_case_summaries: dict[str, Any] = {}

    with xr.open_dataset(ORIGINAL_FORCING["forcing_2020.nc"]) as template_ds:
        mask2d = build_pacific_mask(template_ds)

    climatology = summarize_climatology()

    for case_name, case_info in CASES.items():
        for forcing_path in case_info["forcing"]:
            original_path = ORIGINAL_FORCING[forcing_path.name]
            for var_name in FOCUS_FORCING_VARS:
                detail, csv_row = forcing_var_summary(case_name, forcing_path, original_path, var_name, mask2d)
                forcing_details.append(detail)
                forcing_csv_rows.append(csv_row)

        case_output_details, case_output_rows, case_output_summary = inspect_outputs(case_name, case_info["output_dir"])
        output_details.extend(case_output_details)
        output_csv_rows.extend(case_output_rows)
        output_case_summaries[case_name] = case_output_summary

    forcing_difference_summaries = []
    for year_name in ("forcing_2020.nc", "forcing_2021.nc"):
        forcing_difference_summaries.append(
            difference_summary(
                f"neutral_minus_actual::{year_name}",
                CASES["neutral"]["forcing"][0 if year_name.endswith("2020.nc") else 1],
                CASES["actual"]["forcing"][0 if year_name.endswith("2020.nc") else 1],
                mask2d,
            )
        )
        forcing_difference_summaries.append(
            difference_summary(
                f"reversed_minus_actual::{year_name}",
                CASES["reversed"]["forcing"][0 if year_name.endswith("2020.nc") else 1],
                CASES["actual"]["forcing"][0 if year_name.endswith("2020.nc") else 1],
                mask2d,
            )
        )
        forcing_difference_summaries.append(
            difference_summary(
                f"reversed_minus_neutral::{year_name}",
                CASES["reversed"]["forcing"][0 if year_name.endswith("2020.nc") else 1],
                CASES["neutral"]["forcing"][0 if year_name.endswith("2020.nc") else 1],
                mask2d,
            )
        )

    report = {
        "pacific_domain": PACIFIC_DOMAIN,
        "climatology": climatology,
        "forcing_details": forcing_details,
        "forcing_csv_rows": forcing_csv_rows,
        "forcing_difference_summaries": forcing_difference_summaries,
        "output_details": output_details,
        "output_case_summaries": output_case_summaries,
    }
    category, reason = likely_cause(report)
    report["diagnosis"] = {"category": category, "reason": reason}

    pd.DataFrame(forcing_csv_rows).to_csv(args.forcing_csv, index=False)
    pd.DataFrame(output_csv_rows).to_csv(args.output_csv, index=False)
    args.report_json.write_text(json.dumps(to_native(report), indent=2) + "\n", encoding="utf-8")
    args.report_txt.write_text(render_text(report) + "\n", encoding="utf-8")
    write_final_report(report, args.final_report_md)
    print(render_text(report))


if __name__ == "__main__":
    main()
