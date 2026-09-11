#!/usr/bin/env python3
"""Regrid and standardize selected CMIP6 monthly inputs on a common 1.5 deg grid.

This script:
1. Uses native monthly CMIP6 fields for the selected parent-model runs.
2. Selects exact pressure levels or interpolates exact ocean depths where requested.
3. Regrids each field to a common 1.5 x 1.5 degree global regular grid.
4. Builds WUS-style model-year rows with 12 months: Sep..Dec of year Y and Jan..Aug of Y+1.
5. Standardizes each (month, lat, lon) column across pooled available model-year rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cftime
import numpy as np
import pandas as pd
import xarray as xr


CMIP6_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/CMIP6")
PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "PSCRATCH",
        "/pscratch/sd/h/hyvchen",
    )
) / "Snow-Predication-at-Sierra" / "cmip6_regridded_1p5deg"

TARGET_LON = np.arange(0.75, 360.0, 1.5, dtype=np.float64)
TARGET_LAT = np.arange(-89.25, 90.0, 1.5, dtype=np.float64)
TARGET_MONTHS = np.array([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int16)
TARGET_MONTH_LABELS = np.array(
    ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    dtype=object,
)
TARGET_YSIZE = 120
TARGET_XSIZE = 240
ZERO_VAR_TOL = 1.0e-12


@dataclass(frozen=True)
class ParentRun:
    model: str
    institution_id: str
    member_id: str
    hist_root: Path
    ssp_root: Path


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    variable_id: str
    level_pa: int | None
    depth_m: float | None
    table_id: str
    method: str
    source_family: str
    rep_variable: str
    output_name: str


PARENT_RUNS: tuple[ParentRun, ...] = (
    ParentRun(
        model="EC-Earth3",
        institution_id="EC-Earth-Consortium",
        member_id="r102i1p1f1",
        hist_root=CMIP6_ROOT / "CMIP" / "EC-Earth-Consortium" / "EC-Earth3" / "historical" / "r102i1p1f1",
        ssp_root=CMIP6_ROOT / "ScenarioMIP" / "EC-Earth-Consortium" / "EC-Earth3" / "ssp370" / "r102i1p1f1",
    ),
    ParentRun(
        model="MIROC6",
        institution_id="MIROC",
        member_id="r1i1p1f1",
        hist_root=CMIP6_ROOT / "CMIP" / "MIROC" / "MIROC6" / "historical" / "r1i1p1f1",
        ssp_root=CMIP6_ROOT / "ScenarioMIP" / "MIROC" / "MIROC6" / "ssp370" / "r1i1p1f1",
    ),
    ParentRun(
        # MPI-ESM1-2-HR ScenarioMIP/ssp370 is published under institution_id
        # DKRZ, not MPI-M (which only holds the CMIP/historical branch) --
        # confirmed against both the local mirror and ESGF.
        model="MPI-ESM1-2-HR",
        institution_id="MPI-M",
        member_id="r3i1p1f1",
        hist_root=CMIP6_ROOT / "CMIP" / "MPI-M" / "MPI-ESM1-2-HR" / "historical" / "r3i1p1f1",
        ssp_root=CMIP6_ROOT / "ScenarioMIP" / "DKRZ" / "MPI-ESM1-2-HR" / "ssp370" / "r3i1p1f1",
    ),
    ParentRun(
        model="TaiESM1",
        institution_id="AS-RCEC",
        member_id="r1i1p1f1",
        hist_root=CMIP6_ROOT / "CMIP" / "AS-RCEC" / "TaiESM1" / "historical" / "r1i1p1f1",
        ssp_root=CMIP6_ROOT / "ScenarioMIP" / "AS-RCEC" / "TaiESM1" / "ssp370" / "r1i1p1f1",
    ),
)


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("tos", "tos", None, None, "Omon", "bilinear", "ocean_seaice", "tos", "tos"),
    FieldSpec("thetao_50m", "thetao", None, 50.0, "Omon", "bilinear", "ocean_seaice", "tos", "thetao_50m"),
    FieldSpec("thetao_100m", "thetao", None, 100.0, "Omon", "bilinear", "ocean_seaice", "tos", "thetao_100m"),
    FieldSpec("siconc", "siconc", None, None, "SImon", "conservative", "ocean_seaice", "tos", "siconc"),
    FieldSpec("rlut", "rlut", None, None, "Amon", "bilinear", "atmos_land", "zg", "rlut"),
    FieldSpec("zg_500", "zg", 50000, None, "Amon", "bilinear", "atmos_land", "zg", "zg_500hPa"),
    FieldSpec("ta_850", "ta", 85000, None, "Amon", "bilinear", "atmos_land", "zg", "ta_850hPa"),
    FieldSpec("ua_850", "ua", 85000, None, "Amon", "bilinear", "atmos_land", "zg", "ua_850hPa"),
    FieldSpec("va_850", "va", 85000, None, "Amon", "bilinear", "atmos_land", "zg", "va_850hPa"),
    FieldSpec("ua_200", "ua", 20000, None, "Amon", "bilinear", "atmos_land", "zg", "ua_200hPa"),
    FieldSpec("va_200", "va", 20000, None, "Amon", "bilinear", "atmos_land", "zg", "va_200hPa"),
    FieldSpec("hus_850", "hus", 85000, None, "Amon", "bilinear", "atmos_land", "zg", "hus_850hPa"),
    FieldSpec("psl", "psl", None, None, "Amon", "bilinear", "atmos_land", "zg", "psl"),
    FieldSpec("tas", "tas", None, None, "Amon", "bilinear", "atmos_land", "zg", "tas"),
    FieldSpec("zg_50", "zg", 5000, None, "Amon", "bilinear", "atmos_land", "zg", "zg_50hPa"),
    FieldSpec("ta_50", "ta", 5000, None, "Amon", "bilinear", "atmos_land", "zg", "ta_50hPa"),
    FieldSpec("ua_50", "ua", 5000, None, "Amon", "bilinear", "atmos_land", "zg", "ua_50hPa"),
    FieldSpec("va_50", "va", 5000, None, "Amon", "bilinear", "atmos_land", "zg", "va_50hPa"),
    FieldSpec("mrso", "mrso", None, None, "Lmon", "conservative", "atmos_land", "mrso", "mrso"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--field", action="append", default=None, help="Optional field_id subset.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def choose_latest_version_dir(base: Path) -> Path:
    dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No version directories under {base}")
    return dirs[-1]


def find_source_files(parent: ParentRun, experiment: str, variable_id: str, table_id: str) -> list[Path]:
    root = parent.hist_root if experiment == "historical" else parent.ssp_root
    if not root.exists():
        return []
    base = root / table_id / variable_id
    if not base.exists():
        return []
    files: list[Path] = []
    for grid_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        version_dir = choose_latest_version_dir(grid_dir)
        files.extend(sorted(version_dir.glob("*.nc")))
        if files:
            break
    return files


def write_target_grid(grid_path: Path) -> None:
    text = "\n".join(
        [
            "gridtype = lonlat",
            f"xsize    = {TARGET_XSIZE}",
            f"ysize    = {TARGET_YSIZE}",
            "xfirst   = 0.75",
            "xinc     = 1.5",
            "yfirst   = -89.25",
            "yinc     = 1.5",
            "",
        ]
    )
    grid_path.write_text(text, encoding="utf-8")


def representative_file(parent: ParentRun, spec: FieldSpec) -> Path:
    rep_var = spec.rep_variable
    rep_table = "Omon" if spec.source_family == "ocean_seaice" else ("Lmon" if rep_var == "mrso" else "Amon")
    files = find_source_files(parent, "historical", rep_var, rep_table)
    if not files:
        raise FileNotFoundError(f"No representative files for {parent.model} {rep_var} {rep_table}")
    return files[0]


def weight_file_path(output_root: Path, parent: ParentRun, family: str, method: str) -> Path:
    return output_root / "weights" / f"{parent.model}_{parent.member_id}_{family}_{method}_to_1p5deg.nc"


def ensure_weight_file(output_root: Path, grid_file: Path, parent: ParentRun, spec: FieldSpec, overwrite: bool) -> Path:
    weight_path = weight_file_path(output_root, parent, spec.source_family, spec.method)
    if weight_path.exists() and not overwrite:
        return weight_path
    ensure_dir(weight_path.parent)
    src_file = representative_file(parent, spec)
    genop = "genbil" if spec.method == "bilinear" else "gencon"
    cmd = ["cdo", "-O", "-L", f"{genop},{grid_file}", str(src_file), str(weight_path)]
    run_cmd(cmd)
    return weight_path


def selected_output_path(output_root: Path, parent: ParentRun, spec: FieldSpec, experiment: str) -> Path:
    return output_root / "raw" / "by_model_monthly" / parent.model / experiment / f"{spec.output_name}_1p5deg_monthly.nc"


def merged_source_cache_path(output_root: Path, parent: ParentRun, spec: FieldSpec, experiment: str) -> Path:
    return (
        output_root
        / "raw"
        / "source_merged_cache"
        / parent.model
        / experiment
        / f"{spec.variable_id}_{spec.table_id}_merged.nc"
    )


def native_selected_stats(ds: xr.Dataset, var_name: str) -> dict[str, float | None]:
    arr = np.asarray(ds[var_name].values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _find_pressure_coordinate_name(da: xr.DataArray) -> str:
    for coord_name in da.coords:
        coord = da.coords[coord_name]
        standard_name = str(coord.attrs.get("standard_name", "")).lower()
        units = str(coord.attrs.get("units", "")).lower()
        axis = str(coord.attrs.get("axis", "")).upper()
        if standard_name == "air_pressure":
            return coord_name
        if axis == "Z" and units == "pa":
            return coord_name
    for dim_name in da.dims:
        if dim_name in da.coords:
            coord = da.coords[dim_name]
            standard_name = str(coord.attrs.get("standard_name", "")).lower()
            units = str(coord.attrs.get("units", "")).lower()
            axis = str(coord.attrs.get("axis", "")).upper()
            if standard_name == "air_pressure":
                return dim_name
            if axis == "Z" and units == "pa":
                return dim_name
    raise ValueError(f"Could not find a pressure coordinate for variable {da.name!r}.")


def _find_depth_coordinate_name(da: xr.DataArray) -> str:
    for coord_name in da.coords:
        coord = da.coords[coord_name]
        standard_name = str(coord.attrs.get("standard_name", "")).lower()
        units = str(coord.attrs.get("units", "")).lower()
        axis = str(coord.attrs.get("axis", "")).upper()
        positive = str(coord.attrs.get("positive", "")).lower()
        if standard_name in {"depth", "ocean_sigma_z"}:
            return coord_name
        if axis == "Z" and (units in {"m", "meter", "meters", "metre", "metres"} or positive == "down"):
            return coord_name
    for dim_name in da.dims:
        if dim_name in da.coords:
            coord = da.coords[dim_name]
            standard_name = str(coord.attrs.get("standard_name", "")).lower()
            units = str(coord.attrs.get("units", "")).lower()
            axis = str(coord.attrs.get("axis", "")).upper()
            positive = str(coord.attrs.get("positive", "")).lower()
            if standard_name in {"depth", "ocean_sigma_z"}:
                return dim_name
            if axis == "Z" and (units in {"m", "meter", "meters", "metre", "metres"} or positive == "down"):
                return dim_name
    raise ValueError(f"Could not find an ocean depth coordinate for variable {da.name!r}.")


def _depth_values_in_meters(coord: xr.DataArray) -> np.ndarray:
    values = np.asarray(coord.values, dtype=np.float64)
    units = str(coord.attrs.get("units", "")).strip().lower()
    if units in {"", "m", "meter", "meters", "metre", "metres"}:
        return values
    if units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return values / 100.0
    if units in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return values / 1000.0
    raise ValueError(f"Unsupported depth units {coord.attrs.get('units')!r} for coordinate {coord.name!r}")


def _select_pressure_level_exact(
    merged_path: Path,
    selected_path: Path,
    variable_id: str,
    level_pa: int,
) -> None:
    with xr.open_dataset(merged_path, decode_times=False) as ds:
        if variable_id not in ds.data_vars:
            raise KeyError(f"Variable {variable_id!r} not found in {merged_path}. Found {list(ds.data_vars)}")
        da = ds[variable_id]
        pressure_coord_name = _find_pressure_coordinate_name(da)
        pressure_coord = ds[pressure_coord_name]
        pressure_values = np.asarray(pressure_coord.values)
        matches = np.where(pressure_values == level_pa)[0]
        if matches.size == 0:
            available = ", ".join(str(int(v)) for v in pressure_values.tolist())
            raise ValueError(
                f"Requested pressure level {level_pa} Pa not found for {variable_id!r} in {merged_path}. "
                f"Available levels: {available}"
            )
        selected = da.sel({pressure_coord_name: level_pa}).squeeze(drop=True)
        out = selected.to_dataset(name=variable_id)
        out.attrs = ds.attrs.copy()
        out[variable_id].attrs = da.attrs.copy()
        out[variable_id].attrs["selected_pressure_level_pa"] = int(level_pa)
        out.coords[pressure_coord_name] = xr.DataArray(np.asarray(level_pa), attrs=pressure_coord.attrs.copy())
        for aux_name in ("time_bnds", "lat_bnds", "lon_bnds", "latitude", "longitude"):
            if aux_name in ds.variables and aux_name not in out.variables:
                out[aux_name] = ds[aux_name]
        encoding = {name: {"zlib": True, "complevel": 2} for name in out.data_vars}
        out.to_netcdf(selected_path, engine="netcdf4", encoding=encoding)


def _select_depth_level_interp(
    merged_path: Path,
    selected_path: Path,
    variable_id: str,
    depth_m: float,
) -> None:
    with xr.open_dataset(merged_path, decode_times=False) as ds:
        if variable_id not in ds.data_vars:
            raise KeyError(f"Variable {variable_id!r} not found in {merged_path}. Found {list(ds.data_vars)}")
        da = ds[variable_id]
        depth_coord_name = _find_depth_coordinate_name(da)
        depth_coord = ds[depth_coord_name]
        depth_values_m = _depth_values_in_meters(depth_coord)
        if depth_values_m.ndim != 1:
            raise ValueError(f"Depth coordinate {depth_coord_name!r} must be 1-D for {variable_id!r}")
        finite_depths = depth_values_m[np.isfinite(depth_values_m)]
        if finite_depths.size == 0:
            raise ValueError(f"No finite depth values available for {variable_id!r} in {merged_path}")
        min_depth = float(np.min(finite_depths))
        max_depth = float(np.max(finite_depths))
        if not (min_depth <= depth_m <= max_depth):
            raise ValueError(
                f"Requested depth {depth_m} m outside native range [{min_depth}, {max_depth}] m "
                f"for {variable_id!r} in {merged_path}"
            )
        work = da.assign_coords({depth_coord_name: depth_values_m})
        if depth_values_m.size > 1 and np.any(np.diff(depth_values_m) < 0):
            work = work.sortby(depth_coord_name)
        selected = work.interp({depth_coord_name: depth_m}).squeeze(drop=True)
        out = selected.to_dataset(name=variable_id)
        out.attrs = ds.attrs.copy()
        out[variable_id].attrs = da.attrs.copy()
        out[variable_id].attrs["selected_depth_m"] = float(depth_m)
        out[variable_id].attrs["source_depth_coordinate"] = depth_coord_name
        out.coords[depth_coord_name] = xr.DataArray(np.asarray(depth_m, dtype=np.float64), attrs=depth_coord.attrs.copy())
        out[depth_coord_name].attrs["units"] = "m"
        for aux_name in ("time_bnds", "lat_bnds", "lon_bnds", "latitude", "longitude"):
            if aux_name in ds.variables and aux_name not in out.variables:
                out[aux_name] = ds[aux_name]
        encoding = {name: {"zlib": True, "complevel": 2} for name in out.data_vars}
        out.to_netcdf(selected_path, engine="netcdf4", encoding=encoding)


def merge_select_and_regrid(
    output_root: Path,
    grid_file: Path,
    parent: ParentRun,
    spec: FieldSpec,
    experiment: str,
    overwrite: bool,
) -> Path | None:
    src_files = find_source_files(parent, experiment, spec.variable_id, spec.table_id)
    if not src_files:
        return None
    out_path = selected_output_path(output_root, parent, spec, experiment)
    if out_path.exists() and not overwrite:
        return out_path
    ensure_dir(out_path.parent)
    weight_path = ensure_weight_file(output_root, grid_file, parent, spec, overwrite)
    merged_cache = merged_source_cache_path(output_root, parent, spec, experiment)
    ensure_dir(merged_cache.parent)
    if overwrite or not merged_cache.exists():
        run_cmd(["cdo", "-O", "-L", "mergetime", *[str(p) for p in src_files], str(merged_cache)])
    with tempfile.TemporaryDirectory(prefix="cmip6-regrid-") as tmp:
        tmpdir = Path(tmp)
        selected = tmpdir / "selected.nc"
        if spec.level_pa is not None:
            _select_pressure_level_exact(merged_cache, selected, spec.variable_id, spec.level_pa)
        elif spec.depth_m is not None:
            _select_depth_level_interp(merged_cache, selected, spec.variable_id, spec.depth_m)
        else:
            run_cmd(["cdo", "-O", "-L", f"selname,{spec.variable_id}", str(merged_cache), str(selected)])
        remap_cmd = ["cdo", "-O", "-L", f"remap,{grid_file},{weight_path}", str(selected), str(out_path)]
        run_cmd(remap_cmd)
    return out_path


def harmonize_calendar(ds: xr.Dataset, calendar: str = "proleptic_gregorian") -> xr.Dataset:
    """Normalize a dataset's time coordinate (and time_bnds) to one calendar.

    Source centers can ship historical and ssp370 branches under different
    cftime calendars (e.g. EC-Earth3: Gregorian for historical, Proleptic
    Gregorian for ssp370). xr.Dataset.convert_calendar only reindexes the
    time coordinate itself, not an auxiliary time_bnds variable, so encoding
    a concatenation of the two to NetCDF fails when it tries to diff across
    the boundary. Rebuild time_bnds under the same target calendar so the
    combined dataset is internally consistent.
    """
    ds = ds.convert_calendar(calendar, use_cftime=True)
    if "time_bnds" in ds.variables:
        target_cls = {"proleptic_gregorian": cftime.DatetimeProlepticGregorian}[calendar]
        convert = np.vectorize(
            lambda d: target_cls(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond)
        )
        ds = ds.assign(time_bnds=(ds["time_bnds"].dims, convert(ds["time_bnds"].values)))
    return ds


def normalize_lon_to_360(values: np.ndarray) -> np.ndarray:
    out = np.mod(values, 360.0)
    out[np.isclose(out, 360.0)] = 0.0
    return out


def select_data_var(ds: xr.Dataset, variable_id: str) -> xr.DataArray:
    if variable_id in ds.data_vars:
        return ds[variable_id]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    raise KeyError(f"Could not identify data variable for {variable_id}; found {list(ds.data_vars)}")


def validate_target_grid(da: xr.DataArray) -> dict[str, object]:
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat = np.asarray(da[lat_name].values, dtype=np.float64)
    lon = normalize_lon_to_360(np.asarray(da[lon_name].values, dtype=np.float64))
    lat_ok = lat.shape == (TARGET_YSIZE,) and np.allclose(lat, TARGET_LAT)
    lon_ok = lon.shape == (TARGET_XSIZE,) and np.allclose(lon, TARGET_LON)
    return {
        "shape_ok": tuple(da.shape[-2:]) == (TARGET_YSIZE, TARGET_XSIZE),
        "lat_ok": bool(lat_ok),
        "lon_ok": bool(lon_ok),
    }


def open_selected_dataset(path: Path, variable_id: str) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=True, use_cftime=True)
    da = select_data_var(ds, variable_id)
    rename_map: dict[str, str] = {}
    if "latitude" in da.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in da.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def build_model_year_rows(
    regridded_paths: dict[str, Path],
    variable_id: str,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    samples: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    for model_key, path in regridded_paths.items():
        if path is None:
            continue
        ds = open_selected_dataset(path, variable_id)
        da = select_data_var(ds, variable_id).astype(np.float32)
        years = da["time"].dt.year.values
        months = da["time"].dt.month.values
        lookup: dict[tuple[int, int], int] = {(int(y), int(m)): i for i, (y, m) in enumerate(zip(years, months, strict=False))}
        for start_year in range(1980, 2100):
            wanted = [(start_year, 9), (start_year, 10), (start_year, 11), (start_year, 12)] + [
                (start_year + 1, m) for m in range(1, 9)
            ]
            if not all(key in lookup for key in wanted):
                gaps.append({"model": model_key, "row_year": start_year, "reason": "missing_months"})
                continue
            idx = [lookup[key] for key in wanted]
            sample = np.asarray(da.isel(time=idx).values, dtype=np.float32)
            samples.append(sample)
            rows.append({"model": model_key, "row_year": start_year})
        ds.close()
    if not samples:
        raise RuntimeError(f"No samples constructed for {variable_id}")
    stacked = np.stack(samples, axis=0)
    return stacked, rows, gaps


def save_field_raw_dataset(
    path: Path,
    field_name: str,
    values: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    ensure_dir(path.parent)
    ds = xr.Dataset(
        {
            field_name: xr.DataArray(
                values,
                dims=("sample", "month_in_model_year", "lat", "lon"),
                coords={
                    "sample": np.arange(values.shape[0], dtype=np.int32),
                    "month_in_model_year": TARGET_MONTHS,
                    "lat": TARGET_LAT,
                    "lon": TARGET_LON,
                },
                attrs={"description": "Regridded physical values on common 1.5 degree grid"},
            )
        },
        coords={
            "month_label": ("month_in_model_year", TARGET_MONTH_LABELS),
            "model": ("sample", np.array([row["model"] for row in rows], dtype=object)),
            "row_year": ("sample", np.array([row["row_year"] for row in rows], dtype=np.int16)),
        },
        attrs={
            "grid_resolution_degrees": "1.5 x 1.5",
            "month_order": "Sep,Oct,Nov,Dec,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug",
        },
    )
    encoding = {
        field_name: {
            "zlib": True,
            "complevel": 2,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, 12, TARGET_YSIZE, TARGET_XSIZE),
        }
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values64 = values.astype(np.float64, copy=False)
    mu = np.nanmean(values64, axis=0)
    sigma = np.nanstd(values64, axis=0, ddof=0)
    valid_count = np.sum(np.isfinite(values64), axis=0).astype(np.int32)
    zero_var = (valid_count > 0) & (sigma <= ZERO_VAR_TOL)
    standardized = np.full(values64.shape, np.nan, dtype=np.float32)
    centered = values64 - mu[None, ...]
    safe_sigma = sigma.copy()
    safe_sigma[zero_var] = 1.0
    standardized = np.divide(centered, safe_sigma[None, ...], out=np.full_like(values64, np.nan), where=np.isfinite(values64))
    standardized[:, zero_var] = np.where(np.isfinite(values64[:, zero_var]), 0.0, np.nan)
    return standardized.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32), valid_count, zero_var


def save_standardized_dataset(
    path: Path,
    field_name: str,
    standardized: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    ensure_dir(path.parent)
    ds = xr.Dataset(
        {
            field_name: xr.DataArray(
                standardized,
                dims=("sample", "month_in_model_year", "lat", "lon"),
                coords={
                    "sample": np.arange(standardized.shape[0], dtype=np.int32),
                    "month_in_model_year": TARGET_MONTHS,
                    "lat": TARGET_LAT,
                    "lon": TARGET_LON,
                },
                attrs={"description": "Per-column standardized values pooled across available model-year rows"},
            )
        },
        coords={
            "month_label": ("month_in_model_year", TARGET_MONTH_LABELS),
            "model": ("sample", np.array([row["model"] for row in rows], dtype=object)),
            "row_year": ("sample", np.array([row["row_year"] for row in rows], dtype=np.int16)),
        },
    )
    encoding = {
        field_name: {
            "zlib": True,
            "complevel": 2,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, 12, TARGET_YSIZE, TARGET_XSIZE),
        }
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def save_stats_dataset(
    path: Path,
    field_name: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    valid_count: np.ndarray,
    zero_var: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    ds = xr.Dataset(
        {
            "mu": xr.DataArray(mu, dims=("month_in_model_year", "lat", "lon")),
            "sigma": xr.DataArray(sigma, dims=("month_in_model_year", "lat", "lon")),
            "valid_count": xr.DataArray(valid_count, dims=("month_in_model_year", "lat", "lon")),
            "zero_variance_mask": xr.DataArray(zero_var.astype(np.int8), dims=("month_in_model_year", "lat", "lon")),
        },
        coords={
            "month_in_model_year": TARGET_MONTHS,
            "month_label": ("month_in_model_year", TARGET_MONTH_LABELS),
            "lat": TARGET_LAT,
            "lon": TARGET_LON,
        },
        attrs={
            "field_name": field_name,
            "standardization": "pooled over available model-year rows; NaN excluded; ddof=0",
            "zero_variance_tolerance": ZERO_VAR_TOL,
        },
    )
    encoding = {name: {"zlib": True, "complevel": 2} for name in ds.data_vars}
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def path_storage_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024.0 ** 3)


def upsert_csv_rows(csv_path: Path, rows: list[dict[str, object]], key_columns: list[str]) -> None:
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        for column in new_df.columns:
            if column not in old_df.columns:
                old_df[column] = np.nan
        for column in old_df.columns:
            if column not in new_df.columns:
                new_df[column] = np.nan
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    fields = [spec for spec in FIELD_SPECS if args.field is None or spec.field_id in set(args.field)]
    ensure_dir(output_root)
    grid_file = output_root / "grid_1p5deg.txt"
    write_target_grid(grid_file)
    manifest_rows: list[dict[str, object]] = []
    weight_manifest: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    regridded_by_field: dict[str, dict[str, Path | None]] = {}

    for spec in fields:
        regridded_by_field[spec.field_id] = {}
        for parent in PARENT_RUNS:
            weight_path = ensure_weight_file(output_root, grid_file, parent, spec, args.overwrite)
            weight_manifest.append(
                {
                    "model": parent.model,
                    "member_id": parent.member_id,
                    "family": spec.source_family,
                    "method": spec.method,
                    "weight_file": str(weight_path),
                }
            )
            hist_path = merge_select_and_regrid(output_root, grid_file, parent, spec, "historical", args.overwrite)
            ssp_path = merge_select_and_regrid(output_root, grid_file, parent, spec, "ssp370", args.overwrite)
            combined_key = f"{parent.model}:{parent.member_id}"
            if hist_path is not None and ssp_path is not None:
                with xr.open_dataset(hist_path, decode_times=True, use_cftime=True) as ds_hist, xr.open_dataset(
                    ssp_path, decode_times=True, use_cftime=True
                ) as ds_ssp:
                    hist_var = select_data_var(ds_hist, spec.variable_id)
                    ssp_var = select_data_var(ds_ssp, spec.variable_id)
                    check_hist = validate_target_grid(hist_var)
                    check_ssp = validate_target_grid(ssp_var)
                    validation_rows.append(
                        {
                            "field": spec.field_id,
                            "model": parent.model,
                            "experiment": "historical",
                            **check_hist,
                            "time_count": int(hist_var.sizes["time"]),
                            "units": hist_var.attrs.get("units", ""),
                        }
                    )
                    validation_rows.append(
                        {
                            "field": spec.field_id,
                            "model": parent.model,
                            "experiment": "ssp370",
                            **check_ssp,
                            "time_count": int(ssp_var.sizes["time"]),
                            "units": ssp_var.attrs.get("units", ""),
                        }
                    )
            elif hist_path is not None:
                with xr.open_dataset(hist_path, decode_times=True, use_cftime=True) as ds_hist:
                    hist_var = select_data_var(ds_hist, spec.variable_id)
                    check_hist = validate_target_grid(hist_var)
                    validation_rows.append(
                        {
                            "field": spec.field_id,
                            "model": parent.model,
                            "experiment": "historical_only",
                            **check_hist,
                            "time_count": int(hist_var.sizes["time"]),
                            "units": hist_var.attrs.get("units", ""),
                        }
                    )
            if hist_path is None and ssp_path is None:
                regridded_by_field[spec.field_id][combined_key] = None
                continue
            if hist_path is not None and ssp_path is None:
                regridded_by_field[spec.field_id][combined_key] = hist_path
            elif hist_path is None and ssp_path is not None:
                regridded_by_field[spec.field_id][combined_key] = ssp_path
            else:
                combined_path = output_root / "raw" / "by_model_monthly" / parent.model / "combined" / f"{spec.output_name}_1p5deg_monthly.nc"
                ensure_dir(combined_path.parent)
                if args.overwrite or not combined_path.exists():
                    with xr.open_dataset(hist_path, decode_times=True, use_cftime=True) as ds_hist, xr.open_dataset(
                        ssp_path, decode_times=True, use_cftime=True
                    ) as ds_ssp:
                        ds_hist = harmonize_calendar(ds_hist)
                        ds_ssp = harmonize_calendar(ds_ssp)
                        combined = xr.concat([ds_hist, ds_ssp], dim="time").sortby("time")
                        encoding = {
                            name: {
                                "zlib": True,
                                "complevel": 2,
                                "dtype": "float32",
                                "_FillValue": np.float32(np.nan),
                            }
                            for name in combined.data_vars
                        }
                        # Write to a temp path and rename into place atomically so a
                        # crash mid-encode (e.g. the calendar mismatch this guarded
                        # against) can never leave a truncated file at combined_path
                        # that a later run's cache check would mistake for valid.
                        tmp_combined_path = combined_path.with_suffix(".nc.tmp")
                        combined.to_netcdf(tmp_combined_path, engine="netcdf4", encoding=encoding)
                        tmp_combined_path.replace(combined_path)
                regridded_by_field[spec.field_id][combined_key] = combined_path

        raw_field_path = output_root / "raw" / f"{spec.output_name}_1p5deg_model_years.nc"
        std_field_path = output_root / "standardized" / f"{spec.output_name}_1p5deg_model_years_standardized.nc"
        stats_field_path = output_root / "normalization_stats" / f"{spec.output_name}_1p5deg_stats.nc"

        field_inputs = regridded_by_field[spec.field_id]
        values, rows, gaps = build_model_year_rows(field_inputs, spec.variable_id)
        save_field_raw_dataset(raw_field_path, spec.output_name, values, rows)
        standardized, mu, sigma, valid_count, zero_var = standardize_values(values)
        save_standardized_dataset(std_field_path, spec.output_name, standardized, rows)
        save_stats_dataset(stats_field_path, spec.output_name, mu, sigma, valid_count, zero_var)

        nan_verified = bool(np.isnan(values).any() == np.isnan(standardized).any())
        zero_var_count = int(np.sum(zero_var))
        manifest_rows.append(
            {
                "field": spec.field_id,
                "level_pa": spec.level_pa,
                "depth_m": spec.depth_m,
                "regridding_method": spec.method,
                "weight_file": str(weight_file_path(output_root, PARENT_RUNS[0], spec.source_family, spec.method)),
                "raw_regridded_output": str(raw_field_path),
                "standardized_output": str(std_field_path),
                "stats_output": str(stats_field_path),
                "valid_rows_years": int(values.shape[0]),
                "nan_mask_behavior_verified": nan_verified,
                "zero_variance_columns": zero_var_count,
                "qa_status": "ok",
                "status": "ok",
                "storage_gb_raw": round(path_storage_gb(raw_field_path), 3),
                "storage_gb_standardized": round(path_storage_gb(std_field_path), 3),
                "gaps": json.dumps(gaps[:20]),
            }
        )

    ensure_dir(PROJECT_ROOT / "artifacts" / "cmip6_regrid_standardize")
    upsert_csv_rows(
        PROJECT_ROOT / "artifacts" / "cmip6_regrid_standardize" / "field_summary.csv",
        manifest_rows,
        ["field"],
    )
    upsert_csv_rows(
        PROJECT_ROOT / "artifacts" / "cmip6_regrid_standardize" / "weight_manifest.csv",
        list(pd.DataFrame(weight_manifest).drop_duplicates().to_dict("records")),
        ["model", "member_id", "family", "method", "weight_file"],
    )
    upsert_csv_rows(
        PROJECT_ROOT / "artifacts" / "cmip6_regrid_standardize" / "validation_checks.csv",
        validation_rows,
        ["field", "model", "experiment"],
    )

    summary = {
        "output_root": str(output_root),
        "grid": {
            "lat_centers": [float(TARGET_LAT[0]), float(TARGET_LAT[-1]), float(TARGET_LAT[1] - TARGET_LAT[0])],
            "lon_centers": [float(TARGET_LON[0]), float(TARGET_LON[-1]), float(TARGET_LON[1] - TARGET_LON[0])],
            "shape": [TARGET_YSIZE, TARGET_XSIZE],
        },
        "fields_processed": [spec.field_id for spec in fields],
        "thetao_status": (
            "thetao_50m and thetao_100m processed; MPI-ESM1-2-HR ssp370 remains unavailable locally"
            if any(spec.variable_id == "thetao" for spec in fields)
            else "pending vertical-coordinate decision before horizontal/tensor harmonization"
        ),
    }
    (PROJECT_ROOT / "artifacts" / "cmip6_regrid_standardize" / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
