#!/usr/bin/env python3
"""Create and validate reversed-Pacific NeuralGCM forcing for a chosen water year."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
SCRATCH_ROOT = Path("/pscratch/sd/h/hyvchen")
NEURALGCM_ROOT = SCRATCH_ROOT / "Snow-Predication-at-Sierra_neuralgcm"
NEURALGCM_ASSETS = NEURALGCM_ROOT / "assets"
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"

COBE2_CLIM_SOURCE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.ltm.1991-2020.nc")
PACIFIC_DOMAIN = {
    "lat_min": -10.0,
    "lat_max": 60.0,
    "lon_min_360": 120.0,
    "lon_max_360": 280.0,
}
CLIM_VAR = "sea_surface_temperature_climatology"
SST_VAR = "sea_surface_temperature"
SEA_ICE_VAR = "sea_ice_cover"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-year", type=int, required=True)
    parser.add_argument("--actual-label", required=True, help="Prepared-input label for actual forcing, e.g. wy1998_actual_m001")
    parser.add_argument("--reversed-label", required=True, help="Prepared-input label for reversed forcing, e.g. wy1998_reversed_pacific_m030")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def arrays_equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def extend_periodic_longitude(source_da: xr.DataArray) -> xr.DataArray:
    lon = np.asarray(source_da["lon"].values, dtype=float)
    first = source_da.isel(lon=0).assign_coords(lon=lon[0] + 360.0)
    last = source_da.isel(lon=-1).assign_coords(lon=lon[-1] - 360.0)
    return xr.concat([last, source_da, first], dim="lon").sortby("lon")


def build_climatology(clim_output: Path, actual_forcing: Path, force: bool) -> dict:
    clim_output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_file": str(COBE2_CLIM_SOURCE),
        "output_file": str(clim_output),
        "source_variable": "sst",
        "output_variable": CLIM_VAR,
        "source_units": "degC",
        "output_units": "K",
        "period": "1991-2020 monthly climatology",
        "regridding_method": "xarray linear interpolation in latitude and periodic longitude handling",
        "scientific_formula_reused_from_ace2": True,
    }
    if clim_output.exists() and not force:
        with xr.open_dataset(clim_output) as ds:
            report["output_dims"] = {k: int(v) for k, v in ds.sizes.items()}
            report["reused_existing_output"] = True
        return report

    with xr.open_dataset(COBE2_CLIM_SOURCE, decode_times=False) as source_ds, xr.open_dataset(actual_forcing) as template_ds:
        source_sst = source_ds["sst"]
        units = str(source_sst.attrs.get("units", "")).lower()
        if units not in {"degc", "c", "celsius"}:
            raise ValueError(f"Unexpected COBE2 SST units: {source_sst.attrs.get('units')}")

        source_sst_k = (source_sst.astype("float32") + np.float32(273.15)).rename({"time": "month", "lat": "latitude", "lon": "lon"})
        source_sst_k = source_sst_k.assign_coords(
            month=("month", np.arange(1, int(source_sst_k.sizes["month"]) + 1, dtype=np.int32)),
            latitude=("latitude", source_ds["lat"].values.astype(np.float64)),
            lon=("lon", source_ds["lon"].values.astype(np.float64)),
        )

        target_lat = template_ds["latitude"].values.astype(np.float64)
        target_lon = template_ds["longitude"].values.astype(np.float64)
        periodic_source = extend_periodic_longitude(source_sst_k)
        regridded = periodic_source.interp(latitude=target_lat, lon=target_lon, method="linear")
        regridded = regridded.rename({"lon": "longitude"}).assign_coords(
            latitude=("latitude", target_lat),
            longitude=("longitude", target_lon),
        )
        regridded = regridded.transpose("month", "longitude", "latitude").astype("float32")
        regridded.name = CLIM_VAR
        regridded.attrs = {
            "long_name": "Monthly sea surface temperature climatology on the NeuralGCM forcing grid",
            "units": "K",
            "source_file": str(COBE2_CLIM_SOURCE),
            "source_variable": "sst",
            "source_units": str(source_sst.attrs.get("units", "")),
            "period": "1991-2020",
            "regridding_method": "xarray linear interpolation in latitude with periodic longitude extension",
            "target_grid_file": str(actual_forcing),
        }

        out_ds = xr.Dataset({CLIM_VAR: regridded})
        out_ds["month"].attrs = {"long_name": "calendar month", "units": "1-12"}
        out_ds["longitude"].attrs = dict(template_ds["longitude"].attrs)
        out_ds["latitude"].attrs = dict(template_ds["latitude"].attrs)
        out_ds.to_netcdf(clim_output)

        report.update(
            {
                "reused_existing_output": False,
                "source_dims": {k: int(v) for k, v in source_ds.sizes.items()},
                "template_dims": {k: int(v) for k, v in template_ds.sizes.items()},
                "output_dims": {k: int(v) for k, v in out_ds.sizes.items()},
                "target_latitude_range": [float(target_lat.min()), float(target_lat.max())],
                "target_longitude_range": [float(target_lon.min()), float(target_lon.max())],
                "min_k": float(regridded.min()),
                "max_k": float(regridded.max()),
            }
        )
    return report


def build_pacific_mask(ds: xr.Dataset) -> xr.DataArray:
    lat_grid, lon_grid = xr.broadcast(ds["latitude"], ds["longitude"])
    return (
        (lat_grid >= PACIFIC_DOMAIN["lat_min"])
        & (lat_grid <= PACIFIC_DOMAIN["lat_max"])
        & (lon_grid >= PACIFIC_DOMAIN["lon_min_360"])
        & (lon_grid <= PACIFIC_DOMAIN["lon_max_360"])
    ).transpose("longitude", "latitude")


def climatology_for_time(clim: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    months = xr.DataArray(times.dt.month.values.astype(np.int32), coords={"time": times}, dims=("time",))
    return clim.sel(month=months).assign_coords(time=times)


def suspicious_sst_counts(values: np.ndarray) -> dict[str, int]:
    finite = np.isfinite(values)
    return {
        "count_below_180K": int(np.sum(finite & (values < 180.0))),
        "count_above_330K": int(np.sum(finite & (values > 330.0))),
        "count_below_271_35K": int(np.sum(finite & (values < 271.35))),
        "count_above_310K": int(np.sum(finite & (values > 310.0))),
    }


def build_reversed_and_validate(actual_forcing: Path, reversed_forcing: Path, clim_output: Path, climatology_report: dict, force: bool) -> dict:
    reversed_forcing.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(actual_forcing) as actual_ds, xr.open_dataset(clim_output) as clim_ds:
        actual_sst = actual_ds[SST_VAR]
        clim_sst = climatology_for_time(clim_ds[CLIM_VAR], actual_ds["time"]).transpose("time", "longitude", "latitude")
        pacific_mask_2d = build_pacific_mask(actual_ds)
        common_mask = pacific_mask_2d.broadcast_like(actual_sst) & xr.apply_ufunc(np.isfinite, actual_sst) & xr.apply_ufunc(np.isfinite, clim_sst)
        reversed_sst = xr.where(common_mask, 2.0 * clim_sst - actual_sst, actual_sst).astype(actual_sst.dtype)

        reversed_values = np.asarray(reversed_sst.values, dtype=actual_sst.dtype)
        reversed_float = np.asarray(reversed_values, dtype=float)
        suspicious = suspicious_sst_counts(reversed_float)
        if suspicious["count_below_180K"] > 0 or suspicious["count_above_330K"] > 0:
            raise RuntimeError("Reversed NeuralGCM SST contains physically implausible values outside [180K, 330K]; stopping before writing forcing.")

        if reversed_forcing.exists() and force:
            reversed_forcing.unlink()
        if not reversed_forcing.exists():
            shutil.copy2(actual_forcing, reversed_forcing)
            with netCDF4.Dataset(reversed_forcing, mode="r+") as nc_out:
                nc_out.variables[SST_VAR][:] = reversed_values

    with xr.open_dataset(actual_forcing) as actual_ds, xr.open_dataset(reversed_forcing) as reversed_ds, xr.open_dataset(clim_output) as clim_ds:
        actual_sst = actual_ds[SST_VAR]
        reversed_sst = reversed_ds[SST_VAR]
        clim_sst = climatology_for_time(clim_ds[CLIM_VAR], actual_ds["time"]).transpose("time", "longitude", "latitude")
        pacific_mask_2d = build_pacific_mask(actual_ds)
        common_mask = pacific_mask_2d.broadcast_like(actual_sst) & xr.apply_ufunc(np.isfinite, actual_sst) & xr.apply_ufunc(np.isfinite, clim_sst)
        expected_reversed = xr.where(common_mask, 2.0 * clim_sst - actual_sst, actual_sst).astype(actual_sst.dtype)
        diff = (reversed_sst - expected_reversed).where(common_mask)

        unchanged_flags = {}
        for name in actual_ds.data_vars:
            if name == SST_VAR:
                continue
            unchanged_flags[name] = arrays_equal_with_nan(np.asarray(actual_ds[name].values), np.asarray(reversed_ds[name].values))

        month_values = np.asarray(actual_ds["time"].dt.month.values, dtype=int)
        perturbed_by_month = {}
        mask_values = np.asarray(common_mask.values, dtype=bool)
        actual_values = np.asarray(actual_sst.values, dtype=float)
        clim_values = np.asarray(clim_sst.values, dtype=float)
        reversed_values = np.asarray(reversed_sst.values, dtype=float)
        for month in range(1, 13):
            month_mask = month_values == month
            perturbed_by_month[month] = int(mask_values[month_mask].sum())

        overall_mask = mask_values
        validation = {
            "scientific_setup": {
                "state": "reversed_pacific",
                "sst_formula_inside_mask": "2 * SST_clim - SST_actual",
                "pacific_domain": PACIFIC_DOMAIN,
                "common_mask_definition": "Pacific domain AND finite(SST_actual) AND finite(SST_clim)",
                "clipping_performed": False,
            },
            "actual_forcing_path": str(actual_forcing),
            "reversed_forcing_path": str(reversed_forcing),
            "climatology_path": str(clim_output),
            "climatology_report": climatology_report,
            "time_count": int(actual_ds.sizes["time"]),
            "grid_sizes": {k: int(v) for k, v in actual_ds.sizes.items()},
            "sea_surface_temperature_nan_count": int(np.isnan(reversed_values).sum()),
            "sea_surface_temperature_inf_count": int(np.isinf(reversed_values).sum()),
            "sea_ice_cover_nan_count": int(np.isnan(np.asarray(reversed_ds[SEA_ICE_VAR].values)).sum()),
            "sea_ice_cover_inf_count": int(np.isinf(np.asarray(reversed_ds[SEA_ICE_VAR].values)).sum()),
            "sea_ice_cover_exactly_unchanged": unchanged_flags.get(SEA_ICE_VAR, False),
            "all_non_sst_variables_exactly_unchanged": all(unchanged_flags.values()),
            "non_sst_variable_exact_unchanged": unchanged_flags,
            "sst_unchanged_outside_mask": arrays_equal_with_nan(np.asarray(reversed_sst.where(~common_mask).values), np.asarray(actual_sst.where(~common_mask).values)),
            "formula_max_abs_error_inside_mask": float(np.nanmax(np.abs(diff.values))) if np.isfinite(diff.values).any() else 0.0,
            "formula_mean_abs_error_inside_mask": float(np.nanmean(np.abs(diff.values))) if np.isfinite(diff.values).any() else 0.0,
            "perturbed_grid_cell_count_3d": int(overall_mask.sum()),
            "perturbed_grid_cell_count_2d_union": int(np.any(overall_mask, axis=0).sum()),
            "perturbed_grid_cell_count_by_time": [int(v) for v in overall_mask.reshape(overall_mask.shape[0], -1).sum(axis=1)],
            "perturbed_grid_cell_count_by_month": perturbed_by_month,
            "inside_mask_stats_overall": {
                "actual_min": float(actual_values[overall_mask].min()),
                "actual_max": float(actual_values[overall_mask].max()),
                "actual_mean": float(actual_values[overall_mask].mean()),
                "climatology_min": float(clim_values[overall_mask].min()),
                "climatology_max": float(clim_values[overall_mask].max()),
                "climatology_mean": float(clim_values[overall_mask].mean()),
                "reversed_min": float(reversed_values[overall_mask].min()),
                "reversed_max": float(reversed_values[overall_mask].max()),
                "reversed_mean": float(reversed_values[overall_mask].mean()),
            },
            "suspicious_sst_counts": suspicious_sst_counts(reversed_values),
        }
    return validation


def write_text_report(path: Path, title: str, validation: dict) -> None:
    lines = [
        title,
        f"actual_forcing_path: {validation['actual_forcing_path']}",
        f"reversed_forcing_path: {validation['reversed_forcing_path']}",
        f"climatology_path: {validation['climatology_path']}",
        f"common_mask_definition: {validation['scientific_setup']['common_mask_definition']}",
        f"sea_surface_temperature_nan_count: {validation['sea_surface_temperature_nan_count']}",
        f"sea_surface_temperature_inf_count: {validation['sea_surface_temperature_inf_count']}",
        f"sea_ice_cover_nan_count: {validation['sea_ice_cover_nan_count']}",
        f"sea_ice_cover_inf_count: {validation['sea_ice_cover_inf_count']}",
        f"all_non_sst_variables_exactly_unchanged: {validation['all_non_sst_variables_exactly_unchanged']}",
        f"sea_ice_cover_exactly_unchanged: {validation['sea_ice_cover_exactly_unchanged']}",
        f"sst_unchanged_outside_mask: {validation['sst_unchanged_outside_mask']}",
        f"formula_max_abs_error_inside_mask: {validation['formula_max_abs_error_inside_mask']}",
        f"formula_mean_abs_error_inside_mask: {validation['formula_mean_abs_error_inside_mask']}",
        f"perturbed_grid_cell_count_3d: {validation['perturbed_grid_cell_count_3d']}",
        f"perturbed_grid_cell_count_2d_union: {validation['perturbed_grid_cell_count_2d_union']}",
        f"inside_mask_stats_overall: {validation['inside_mask_stats_overall']}",
        f"suspicious_sst_counts: {validation['suspicious_sst_counts']}",
        "clipping_performed: False",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    actual_forcing = NEURALGCM_ASSETS / "prepared_inputs" / args.actual_label / "forcing_regridded_6h.nc"
    reversed_dir = NEURALGCM_ASSETS / "prepared_inputs" / args.reversed_label
    reversed_forcing = reversed_dir / "forcing_regridded_6h.nc"
    clim_dir = NEURALGCM_ASSETS / "climatology" / args.reversed_label
    clim_output = clim_dir / "cobe2_monthly_sea_surface_temperature_climatology_on_neuralgcm_grid.nc"
    validation_json = REPORT_DIR / f"neuralgcm_wy{args.water_year}_reversed_pacific_forcing_validation.json"
    validation_txt = REPORT_DIR / f"neuralgcm_wy{args.water_year}_reversed_pacific_forcing_validation.txt"

    if not actual_forcing.exists():
        raise FileNotFoundError(f"Missing actual forcing: {actual_forcing}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    climatology_report = build_climatology(clim_output=clim_output, actual_forcing=actual_forcing, force=args.force)
    validation = build_reversed_and_validate(actual_forcing=actual_forcing, reversed_forcing=reversed_forcing, clim_output=clim_output, climatology_report=climatology_report, force=args.force)
    save_json(validation_json, validation)
    write_text_report(validation_txt, f"NeuralGCM WY{args.water_year} reversed-Pacific forcing validation", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
