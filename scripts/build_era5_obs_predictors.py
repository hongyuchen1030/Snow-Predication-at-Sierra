#!/usr/bin/env python3
"""Build ERA5/ERA5-Land-derived observational analogs of 16 CMIP6 CNN predictors.

Standalone, read-only with respect to production CMIP6 preprocessing. Produces one
`<field>_1p5deg_cnn_rows.nc` per field, on the exact CMIP6-training grid and in the exact
Sep-Mar water-year row layout used by `scripts/stage_en4_thetao_obs_audit.py` for
thetao_50m/thetao_100m, so all 19 fields can later be combined by
`scripts/assemble_era5_cmip6_obs_cache.py`.

Fields covered here (matches OBS_TRANSFER_16_PREDICTORS in prepare_cmip6_cnn_experiment.py):
    tos, siconc, rlut, zg_500, ta_850, ua_850, va_850, ua_200, va_200, hus_850,
    psl, tas, zg_50, ta_50, ua_50, va_50
`mrso` is built separately by preprocess_era5land_full_column_soil_moisture.py; thetao_50m/
thetao_100m are already built from EN4 by stage_en4_thetao_obs_audit.py.

Every unit/sign conversion and source-selection rule below was verified against actual file
metadata (see /global/homes/h/hyvchen/.claude/plans/atomic-launching-prism.md for the
verification record); nothing here is a guessed convention.

Source dispatch:
  - tos (sstk), siconc (ci), psl (msl), tas (2t): e5.oper.an.sfc, hourly, monthly-mean.
  - zg_500, zg_50 (z): e5.oper.an.pl, hourly only (never precomputed monthly) for all 37 years.
  - ta_*, ua_*, va_*, hus_850 (t/u/v/q): e5.oper.an.pl.monthly (NCAR RDA ds633.0 precomputed
    monthly mean) through 2019-05; e5.oper.an.pl hourly aggregation for the WY2020/WY2021 tail
    (Sep2019-Mar2020, Sep2020-Mar2021), which precomputed monthly does not cover.
  - rlut (mtnlwrf): e5.oper.fc.sfc.meanflux, forecast-step (2 inits/day x 12-hour steps)
    reconstruction into an hourly-equivalent series, then monthly mean, then sign-flipped
    (rlut = -mtnlwrf; net-downward-at-TOA -> outgoing).

Requires `module load climate-utils/2025.01` (provides `cdo`) and an xarray-capable env
(`conda activate uxarray_build`). Heavy hourly reads (z for all years, meanflux reconstruction)
should run on a compute node.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from process_cmip6_selected_regrid_standardize import (  # noqa: E402
    TARGET_LAT,
    TARGET_LON,
    TARGET_MONTH_LABELS,
    TARGET_MONTHS,
    TARGET_XSIZE,
    TARGET_YSIZE,
    write_target_grid,
)

ERA5_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5")
AN_SFC_ROOT = ERA5_ROOT / "e5.oper.an.sfc"
AN_PL_ROOT = ERA5_ROOT / "e5.oper.an.pl"
AN_PL_MONTHLY_ROOT = ERA5_ROOT / "e5.oper.an.pl.monthly"
FC_MEANFLUX_ROOT = ERA5_ROOT / "e5.oper.fc.sfc.meanflux"

PSCRATCH_ROOT = Path(os.environ.get("PSCRATCH", "/pscratch/sd/h/hyvchen")).expanduser()
STAGE_DIR = PSCRATCH_ROOT / "Snow-Predication-at-Sierra" / "data" / "era5_obs_predictor_stage"
NATIVE_DIR = STAGE_DIR / "native_monthly"
REGRID_DIR = STAGE_DIR / "regridded_1p5deg"
CNN_ROWS_DIR = STAGE_DIR / "cnn_rows"
WEIGHTS_DIR = STAGE_DIR / "weights"
MANIFEST_DIR = STAGE_DIR / "manifest"
GRID_FILE = STAGE_DIR / "grid_1p5deg.txt"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
MONTHS_OF_INTEREST = {9, 10, 11, 12, 1, 2, 3}
START_MONTH_KEY = "198409"
END_MONTH_KEY = "202103"
# NCAR RDA ds633.0 precomputed pressure-level monthly means (t/u/v/q) end here; verified by
# listing e5.oper.an.pl.monthly, last available month key is 201905.
PL_MONTHLY_LAST_COVERED_KEY = "201905"

NETCDF_ENGINE = "netcdf4"


def identity(x: np.ndarray) -> np.ndarray:
    return x


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    output_name: str
    source_family: str  # "an.sfc" | "an.pl" | "fc.meanflux"
    var_code: str
    file_varname: str
    level_hpa: Optional[float]
    transform: Callable[[np.ndarray], np.ndarray]
    transform_desc: str
    regrid_method: str  # "bilinear" | "conservative"
    target_units: str
    cmip6_standard_name: str
    always_hourly: bool = False


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("tos", "tos", "an.sfc", "128_034_sstk", "SSTK", None,
              lambda x: x - 273.15, "K -> degC: x - 273.15",
              "bilinear", "degC", "sea_surface_temperature"),
    FieldSpec("siconc", "siconc", "an.sfc", "128_031_ci", "CI", None,
              lambda x: x * 100.0, "fraction (0-1) -> percent: x * 100",
              "conservative", "%", "sea_ice_area_fraction"),
    FieldSpec("psl", "psl", "an.sfc", "128_151_msl", "MSL", None,
              identity, "Pa -> Pa: none",
              "bilinear", "Pa", "air_pressure_at_mean_sea_level"),
    FieldSpec("tas", "tas", "an.sfc", "128_167_2t", "VAR_2T", None,
              identity, "K -> K: none",
              "bilinear", "K", "air_temperature"),
    FieldSpec("rlut", "rlut", "fc.meanflux", "235_040_mtnlwrf", "MTNLWRF", None,
              lambda x: -x, "W/m2 -> W/m2: rlut = -mtnlwrf (net-downward-at-TOA -> outgoing)",
              "bilinear", "W m-2", "toa_outgoing_longwave_flux"),
    FieldSpec("zg_500", "zg_500hPa", "an.pl", "128_129_z", "Z", 500.0,
              lambda x: x / 9.80665, "m2/s2 -> m: x / 9.80665 (standard gravity)",
              "bilinear", "m", "geopotential_height", always_hourly=True),
    FieldSpec("zg_50", "zg_50hPa", "an.pl", "128_129_z", "Z", 50.0,
              lambda x: x / 9.80665, "m2/s2 -> m: x / 9.80665 (standard gravity)",
              "bilinear", "m", "geopotential_height", always_hourly=True),
    FieldSpec("ta_850", "ta_850hPa", "an.pl", "128_130_t", "T", 850.0,
              identity, "K -> K: none",
              "bilinear", "K", "air_temperature"),
    FieldSpec("ta_50", "ta_50hPa", "an.pl", "128_130_t", "T", 50.0,
              identity, "K -> K: none",
              "bilinear", "K", "air_temperature"),
    FieldSpec("ua_850", "ua_850hPa", "an.pl", "128_131_u", "U", 850.0,
              identity, "m/s -> m/s: none",
              "bilinear", "m s-1", "eastward_wind"),
    FieldSpec("ua_200", "ua_200hPa", "an.pl", "128_131_u", "U", 200.0,
              identity, "m/s -> m/s: none",
              "bilinear", "m s-1", "eastward_wind"),
    FieldSpec("ua_50", "ua_50hPa", "an.pl", "128_131_u", "U", 50.0,
              identity, "m/s -> m/s: none",
              "bilinear", "m s-1", "eastward_wind"),
    FieldSpec("va_850", "va_850hPa", "an.pl", "128_132_v", "V", 850.0,
              identity, "m/s -> m/s: none",
              "bilinear", "m s-1", "northward_wind"),
    FieldSpec("va_200", "va_200hPa", "an.pl", "128_132_v", "V", 200.0,
              identity, "m/s -> m/s: none",
              "bilinear", "m s-1", "northward_wind"),
    FieldSpec("va_50", "va_50hPa", "an.pl", "128_132_v", "V", 50.0,
              identity, "m/s -> m/s: none",
              "bilinear", "m s-1", "northward_wind"),
    FieldSpec("hus_850", "hus_850hPa", "an.pl", "128_133_q", "Q", 850.0,
              identity, "kg/kg -> 1 (dimensionless): none",
              "bilinear", "1", "specific_humidity"),
)

FIELD_SPECS_BY_ID = {spec.field_id: spec for spec in FIELD_SPECS}


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if "nid" not in hostname or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Refusing to run heavy ERA5 aggregation on a login node; "
            "run inside an interactive/batch compute-node allocation."
        )


def ensure_dirs() -> None:
    for path in (NATIVE_DIR, REGRID_DIR, CNN_ROWS_DIR, WEIGHTS_DIR, MANIFEST_DIR):
        path.mkdir(parents=True, exist_ok=True)


def month_keys_sep_mar(start_key: str, end_key: str) -> list[str]:
    keys: list[str] = []
    year, month = int(start_key[:4]), int(start_key[4:6])
    end_year, end_month = int(end_key[:4]), int(end_key[4:6])
    while (year, month) <= (end_year, end_month):
        if month in MONTHS_OF_INTEREST:
            keys.append(f"{year:04d}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return keys


ALL_MONTH_KEYS = month_keys_sep_mar(START_MONTH_KEY, END_MONTH_KEY)


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Per-source-family monthly-mean loaders. Each returns
# (field[lat,lon] float32, lat[lat] float64, lon[lon] float64, native_units, source_files)
# ---------------------------------------------------------------------------


def load_an_sfc_month(var_code: str, file_varname: str, month_key: str):
    month_dir = AN_SFC_ROOT / month_key
    matches = sorted(month_dir.glob(f"e5.oper.an.sfc.{var_code}.ll025sc.{month_key}0100_*.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one an.sfc file for {var_code} {month_key}, found {len(matches)}")
    path = matches[0]
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=False) as ds:
        da = ds[file_varname]
        monthly_mean = da.mean(dim="time", skipna=True, keep_attrs=True).load()
        field = np.asarray(monthly_mean.values, dtype=np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        units = str(da.attrs.get("units", ""))
    return field, lat, lon, units, [str(path)]


# ERA5 hourly pressure-level files tag the horizontal grid in the filename: wind-vector
# components (u, v) are stored on the "ll025uv" grid, every other pressure-level scalar (z, t,
# q, ...) on "ll025sc". Verified directly against the archive (e5.oper.an.pl/*/); the
# precomputed monthly-mean product (e5.oper.an.pl.monthly) always uses ll025sc regardless of
# variable, so this distinction only matters for the hourly fallback path.
AN_PL_WIND_VAR_CODES = {"128_131_u", "128_132_v"}


def load_an_pl_hourly_month(var_code: str, file_varname: str, level_hpa: float, month_key: str):
    grid_tag = "ll025uv" if var_code in AN_PL_WIND_VAR_CODES else "ll025sc"
    month_dir = AN_PL_ROOT / month_key
    matches = sorted(month_dir.glob(f"e5.oper.an.pl.{var_code}.{grid_tag}.{month_key}*.nc"))
    if not matches:
        raise FileNotFoundError(f"No hourly an.pl files for {var_code} {month_key} (grid_tag={grid_tag})")
    running_sum = None
    running_count = None
    lat = lon = units = None
    for path in matches:
        with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=False) as ds:
            da = ds[file_varname].sel(level=level_hpa)
            actual_level = float(np.asarray(da["level"].values))
            if abs(actual_level - level_hpa) > 1e-6:
                raise ValueError(f"Level mismatch for {var_code} {path}: wanted {level_hpa}, found {actual_level}")
            vals = np.asarray(da.values, dtype=np.float64)
            day_sum = np.nansum(vals, axis=0)
            day_count = np.sum(np.isfinite(vals), axis=0)
            running_sum = day_sum if running_sum is None else running_sum + day_sum
            running_count = day_count if running_count is None else running_count + day_count
            if lat is None:
                lat = np.asarray(ds["latitude"].values, dtype=np.float64)
                lon = np.asarray(ds["longitude"].values, dtype=np.float64)
                units = str(ds[file_varname].attrs.get("units", ""))
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(running_count > 0, running_sum / running_count, np.nan).astype(np.float32)
    return field, lat, lon, units, [str(p) for p in matches]


def load_an_pl_monthly(var_code: str, file_varname: str, level_hpa: float, month_key: str):
    path = AN_PL_MONTHLY_ROOT / f"e5.oper.an.pl.{var_code}.ll025sc.{month_key}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=False) as ds:
        da = ds[file_varname]
        sel = da.sel(level=level_hpa)
        actual_level = float(np.asarray(sel["level"].values))
        if abs(actual_level - level_hpa) > 1e-6:
            raise ValueError(f"Level mismatch for {var_code} {path}: wanted {level_hpa}, found {actual_level}")
        if "time" in sel.dims:
            sel = sel.isel(time=0)
        field = np.asarray(sel.values, dtype=np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        units = str(da.attrs.get("units", ""))
    return field, lat, lon, units, [str(path)]


def load_meanflux_month(var_code: str, file_varname: str, month_key: str):
    month_dir = FC_MEANFLUX_ROOT / month_key
    matches = sorted(month_dir.glob(f"e5.oper.fc.sfc.meanflux.{var_code}.ll025sc.{month_key}*_*.nc"))
    if not matches:
        raise FileNotFoundError(f"No meanflux files for {var_code} {month_key}")
    running_sum = None
    running_count = None
    lat = lon = units = None
    for path in matches:
        with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=False) as ds:
            da = ds[file_varname]
            vals = np.asarray(da.values, dtype=np.float64)
            n_init, n_fh, n_lat, n_lon = vals.shape
            flat = vals.reshape(n_init * n_fh, n_lat, n_lon)
            block_sum = np.nansum(flat, axis=0)
            block_count = np.sum(np.isfinite(flat), axis=0)
            running_sum = block_sum if running_sum is None else running_sum + block_sum
            running_count = block_count if running_count is None else running_count + block_count
            if lat is None:
                lat = np.asarray(ds["latitude"].values, dtype=np.float64)
                lon = np.asarray(ds["longitude"].values, dtype=np.float64)
                units = str(da.attrs.get("units", ""))
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(running_count > 0, running_sum / running_count, np.nan).astype(np.float32)
    return field, lat, lon, units, [str(p) for p in matches]


def load_one_month(spec: FieldSpec, month_key: str):
    if spec.source_family == "an.sfc":
        raw, lat, lon, units, src = load_an_sfc_month(spec.var_code, spec.file_varname, month_key)
        source_kind = "an.sfc_hourly"
    elif spec.source_family == "fc.meanflux":
        raw, lat, lon, units, src = load_meanflux_month(spec.var_code, spec.file_varname, month_key)
        source_kind = "fc.sfc.meanflux_forecast_step_reconstruction"
    elif spec.source_family == "an.pl":
        use_precomputed = (not spec.always_hourly) and (month_key <= PL_MONTHLY_LAST_COVERED_KEY)
        if use_precomputed:
            raw, lat, lon, units, src = load_an_pl_monthly(spec.var_code, spec.file_varname, spec.level_hpa, month_key)
            source_kind = "an.pl.monthly_precomputed_ds633.0"
        else:
            raw, lat, lon, units, src = load_an_pl_hourly_month(spec.var_code, spec.file_varname, spec.level_hpa, month_key)
            source_kind = "an.pl_hourly_aggregated"
    else:
        raise ValueError(f"Unknown source family: {spec.source_family}")
    transformed = spec.transform(raw.astype(np.float64)).astype(np.float32)
    return transformed, lat, lon, units, src, source_kind


def build_field_native_monthly(spec: FieldSpec) -> Path:
    out_path = NATIVE_DIR / f"{spec.field_id}_native_monthly.nc"
    manifest_path = MANIFEST_DIR / f"{spec.field_id}_manifest.json"
    if out_path.exists() and manifest_path.exists():
        print(f"[{spec.field_id}] reusing existing native monthly file: {out_path}", flush=True)
        return out_path

    fields: list[np.ndarray] = []
    provenance: list[dict] = []
    lat = lon = None
    for month_key in ALL_MONTH_KEYS:
        field, lat_i, lon_i, native_units, src, source_kind = load_one_month(spec, month_key)
        if lat is None:
            lat, lon = lat_i, lon_i
        else:
            if not np.array_equal(lat, lat_i) or not np.array_equal(lon, lon_i):
                raise ValueError(f"Grid mismatch for {spec.field_id} at {month_key}")
        fields.append(field)
        provenance.append(
            {
                "month_key": month_key,
                "source_kind": source_kind,
                "source_files": src,
                "native_units": native_units,
            }
        )
        print(f"[{spec.field_id}] {month_key}: {source_kind} native_units={native_units}", flush=True)

    stacked = np.stack(fields, axis=0)
    write_native_monthly_stack(spec, stacked, lat, lon, out_path, manifest_path, provenance)
    return out_path


def write_native_monthly_stack(
    spec: FieldSpec,
    stacked: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    out_path: Path,
    manifest_path: Path,
    provenance: list[dict],
) -> None:
    # Deliberately do NOT attach month_key as an xarray coordinate: CDO's netCDF reader treats
    # any non-spatial auxiliary coordinate as a grid coordinate it must interpret, and aborts
    # ("Unsupported generic coordinates") on a string-typed one. The month_index -> month_key
    # mapping is authoritatively recorded in the JSON manifest instead, and is reconstructed
    # deterministically from ALL_MONTH_KEYS (same list used to build this stack) downstream.
    da = xr.DataArray(
        stacked,
        dims=("month_index", "latitude", "longitude"),
        coords={
            "month_index": np.arange(len(ALL_MONTH_KEYS), dtype=np.int32),
            "latitude": lat,
            "longitude": lon,
        },
        name=spec.output_name,
        attrs={
            "units": spec.target_units,
            "standard_name": spec.cmip6_standard_name,
            "transform": spec.transform_desc,
        },
    )
    ds_out = da.to_dataset()
    # CDO needs proper CF lat/lon attrs to recognize the source grid; without units/axis it
    # falls back to "generic coordinates" and genbil/gencon abort.
    ds_out["latitude"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds_out["longitude"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    ds_out.to_netcdf(
        out_path,
        engine=NETCDF_ENGINE,
        encoding={
            spec.output_name: {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
                "chunksizes": (1, len(lat), len(lon)),
            },
            "latitude": {"_FillValue": None},
            "longitude": {"_FillValue": None},
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "field_id": spec.field_id,
                "output_name": spec.output_name,
                "source_family": spec.source_family,
                "var_code": spec.var_code,
                "level_hpa": spec.level_hpa,
                "transform": spec.transform_desc,
                "target_units": spec.target_units,
                "regrid_method": spec.regrid_method,
                "months": provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{spec.field_id}] wrote native monthly file: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Fast path for zg_500 + zg_50: both come from the same hourly geopotential (z) files, which
# are never precomputed monthly and must be read for all 37 years. Each daily file bundles all
# 37 pressure levels in one compressed chunk, so selecting a single level still forces
# decompressing the whole ~3.6GB/day-file (observed ~10-11 min/month single-threaded - at that
# rate the full 259-month, two-separate-processes build would take ~1-2 days). This combined,
# parallel path (a) extracts both needed levels in one pass per file instead of two independent
# full re-reads, and (b) parallelizes across months with a process pool.
# ---------------------------------------------------------------------------

ZG_LEVELS = (500.0, 50.0)
ZG_VAR_CODE = "128_129_z"
ZG_FILE_VARNAME = "Z"


def _load_z_pair_one_month(month_key: str):
    month_dir = AN_PL_ROOT / month_key
    matches = sorted(month_dir.glob(f"e5.oper.an.pl.{ZG_VAR_CODE}.ll025sc.{month_key}*.nc"))
    if not matches:
        raise FileNotFoundError(f"No hourly an.pl files for {ZG_VAR_CODE} {month_key}")
    running_sum = {lvl: None for lvl in ZG_LEVELS}
    running_count = {lvl: None for lvl in ZG_LEVELS}
    lat = lon = None
    for path in matches:
        with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=False) as ds:
            da = ds[ZG_FILE_VARNAME].sel(level=list(ZG_LEVELS))
            actual_levels = np.asarray(da["level"].values, dtype=np.float64)
            if not np.allclose(sorted(actual_levels), sorted(ZG_LEVELS)):
                raise ValueError(f"Level mismatch for {ZG_VAR_CODE} {path}: wanted {ZG_LEVELS}, found {actual_levels}")
            vals = np.asarray(da.values, dtype=np.float64)  # (time, level, lat, lon), level order per da["level"]
            level_order = list(np.asarray(da["level"].values, dtype=np.float64))
            for lvl in ZG_LEVELS:
                idx = level_order.index(lvl)
                day_sum = np.nansum(vals[:, idx], axis=0)
                day_count = np.sum(np.isfinite(vals[:, idx]), axis=0)
                running_sum[lvl] = day_sum if running_sum[lvl] is None else running_sum[lvl] + day_sum
                running_count[lvl] = day_count if running_count[lvl] is None else running_count[lvl] + day_count
            if lat is None:
                lat = np.asarray(ds["latitude"].values, dtype=np.float64)
                lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    fields = {}
    for lvl in ZG_LEVELS:
        with np.errstate(invalid="ignore", divide="ignore"):
            fields[lvl] = np.where(running_count[lvl] > 0, running_sum[lvl] / running_count[lvl], np.nan).astype(np.float32)
    return month_key, fields[500.0], fields[50.0], lat, lon


def build_zg_pair_parallel(n_workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    spec500 = FIELD_SPECS_BY_ID["zg_500"]
    spec50 = FIELD_SPECS_BY_ID["zg_50"]
    out500 = NATIVE_DIR / "zg_500_native_monthly.nc"
    out50 = NATIVE_DIR / "zg_50_native_monthly.nc"
    man500 = MANIFEST_DIR / "zg_500_manifest.json"
    man50 = MANIFEST_DIR / "zg_50_manifest.json"
    if out500.exists() and man500.exists() and out50.exists() and man50.exists():
        print("[zg_pair] both native monthly files already exist; nothing to do.", flush=True)
        return

    results: dict[str, tuple] = {}
    total = len(ALL_MONTH_KEYS)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_z_pair_one_month, mk): mk for mk in ALL_MONTH_KEYS}
        completed = 0
        for future in as_completed(futures):
            month_key, field500, field50, lat, lon = future.result()
            results[month_key] = (field500, field50, lat, lon)
            completed += 1
            print(f"[zg_pair] {month_key} done ({completed}/{total})", flush=True)

    lat_ref = results[ALL_MONTH_KEYS[0]][2]
    lon_ref = results[ALL_MONTH_KEYS[0]][3]
    stack500 = np.stack([results[mk][0] for mk in ALL_MONTH_KEYS], axis=0) / 9.80665
    stack50 = np.stack([results[mk][1] for mk in ALL_MONTH_KEYS], axis=0) / 9.80665
    stack500 = stack500.astype(np.float32)
    stack50 = stack50.astype(np.float32)

    provenance = [
        {
            "month_key": mk,
            "source_kind": "an.pl_hourly_aggregated_parallel_combined_zg_pair",
            "source_files": ["e5.oper.an.pl (per-day files, both 500hPa and 50hPa extracted in one pass)"],
            "native_units": "m**2 s**-2",
        }
        for mk in ALL_MONTH_KEYS
    ]
    write_native_monthly_stack(spec500, stack500, lat_ref, lon_ref, out500, man500, provenance)
    write_native_monthly_stack(spec50, stack50, lat_ref, lon_ref, out50, man50, provenance)
    print("[zg_pair] done.", flush=True)


def weight_file_path(method: str) -> Path:
    return WEIGHTS_DIR / f"era5_0p25deg_to_1p5deg_{method}.nc"


def ensure_weight_file(method: str, template_native_path: Path, template_varname: str) -> Path:
    weight_path = weight_file_path(method)
    if weight_path.exists():
        return weight_path
    if not GRID_FILE.exists():
        write_target_grid(GRID_FILE)
    genop = "genbil" if method == "bilinear" else "gencon"
    run_cmd(["cdo", "-O", "-L", f"{genop},{GRID_FILE}", str(template_native_path), str(weight_path)])
    return weight_path


def regrid_field(spec: FieldSpec, native_path: Path) -> Path:
    out_path = REGRID_DIR / f"{spec.field_id}_1p5deg_monthly.nc"
    if out_path.exists():
        print(f"[{spec.field_id}] reusing existing regridded file: {out_path}", flush=True)
        return out_path
    weight_path = ensure_weight_file(spec.regrid_method, native_path, spec.output_name)
    run_cmd(["cdo", "-O", "-L", f"remap,{GRID_FILE},{weight_path}", str(native_path), str(out_path)])
    return out_path


def build_cnn_rows(spec: FieldSpec, regridded_path: Path) -> Path:
    out_path = CNN_ROWS_DIR / f"{spec.field_id}_1p5deg_cnn_rows.nc"
    if out_path.exists():
        print(f"[{spec.field_id}] reusing existing cnn_rows file: {out_path}", flush=True)
        return out_path

    with xr.open_dataset(regridded_path, engine=NETCDF_ENGINE) as ds:
        da = ds[spec.output_name]
        if da.sizes["month_index"] != len(ALL_MONTH_KEYS):
            raise ValueError(
                f"{spec.field_id}: regridded month_index size {da.sizes['month_index']} != "
                f"len(ALL_MONTH_KEYS)={len(ALL_MONTH_KEYS)}; cached file was likely built from a "
                "different month range (e.g. a --months test run) - delete and rebuild."
            )
        month_keys = list(ALL_MONTH_KEYS)
        lat = np.asarray(ds["lat"].values if "lat" in ds.coords else ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values if "lon" in ds.coords else ds["longitude"].values, dtype=np.float64)
        if not np.allclose(lat, TARGET_LAT) or not np.allclose(lon, TARGET_LON):
            raise ValueError(f"{spec.field_id}: regridded coordinates do not match TARGET_LAT/TARGET_LON")

        index_by_key = {key: i for i, key in enumerate(month_keys)}
        samples = []
        water_years = []
        for water_year in range(WATER_YEAR_START, WATER_YEAR_END + 1):
            wanted = [f"{water_year - 1}{m:02d}" for m in (9, 10, 11, 12)] + [f"{water_year}{m:02d}" for m in (1, 2, 3)]
            if not all(key in index_by_key for key in wanted):
                raise ValueError(f"{spec.field_id}: missing months for water year {water_year}: {wanted}")
            idx = [index_by_key[key] for key in wanted]
            samples.append(np.asarray(da.isel(month_index=idx).values, dtype=np.float32))
            water_years.append(water_year)

        values = np.stack(samples, axis=0)

    ds_out = xr.Dataset(
        {
            spec.output_name: xr.DataArray(
                values,
                dims=("sample", "month_in_model_year", "lat", "lon"),
                coords={
                    "sample": np.arange(values.shape[0], dtype=np.int32),
                    "month_in_model_year": TARGET_MONTHS[:7],
                    "lat": TARGET_LAT,
                    "lon": TARGET_LON,
                },
                attrs={"units": spec.target_units, "standard_name": spec.cmip6_standard_name},
            )
        },
        coords={
            "month_label": ("month_in_model_year", TARGET_MONTH_LABELS[:7]),
            "water_year": ("sample", np.asarray(water_years, dtype=np.int16)),
        },
        attrs={
            "grid_resolution_degrees": "1.5 x 1.5",
            "month_order": "Sep,Oct,Nov,Dec,Jan,Feb,Mar",
            "source": "ERA5 (NCAR RDA ds633.0), see manifest JSON for per-month provenance",
        },
    )
    ds_out.to_netcdf(
        out_path,
        engine=NETCDF_ENGINE,
        encoding={
            spec.output_name: {
                "zlib": True,
                "complevel": 2,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
                "chunksizes": (1, 7, TARGET_YSIZE, TARGET_XSIZE),
            }
        },
    )
    print(f"[{spec.field_id}] wrote cnn_rows file: {out_path}", flush=True)
    return out_path


def bootstrap_weight_files() -> None:
    """Build both CDO weight files (bilinear, conservative) once, sequentially, from a throwaway
    single-month template file. Must run before any parallel per-field invocation of this script
    to avoid concurrent processes racing to write the same weight file (ensure_weight_file() is
    not otherwise safe to call from multiple processes at once)."""
    ensure_dirs()
    if weight_file_path("bilinear").exists() and weight_file_path("conservative").exists():
        print("Both weight files already exist; nothing to bootstrap.", flush=True)
        return
    template_field, template_key = FIELD_SPECS_BY_ID["tas"], ALL_MONTH_KEYS[0]
    field, lat, lon, _, _, _ = load_one_month(template_field, template_key)
    tmp_path = WEIGHTS_DIR / "_bootstrap_template.nc"
    da = xr.DataArray(
        field[None, :, :],
        dims=("month_index", "latitude", "longitude"),
        coords={"month_index": np.arange(1, dtype=np.int32), "latitude": lat, "longitude": lon},
        name="template",
    )
    ds_tmp = da.to_dataset()
    ds_tmp["latitude"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds_tmp["longitude"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    ds_tmp.to_netcdf(tmp_path, engine=NETCDF_ENGINE, encoding={"latitude": {"_FillValue": None}, "longitude": {"_FillValue": None}})
    for method in ("bilinear", "conservative"):
        ensure_weight_file(method, tmp_path, "template")
    tmp_path.unlink(missing_ok=True)
    print("Weight files bootstrapped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", action="append", default=None, help="Optional field_id subset (repeatable).")
    parser.add_argument("--months", action="append", default=None, help="Optional month_key subset for testing (repeatable, YYYYMM).")
    parser.add_argument("--skip-compute-node-check", action="store_true")
    parser.add_argument("--bootstrap-weights-only", action="store_true", help="Build both CDO weight files and exit (run once before parallel field jobs).")
    parser.add_argument("--build-zg-pair-parallel", type=int, default=None, metavar="N_WORKERS", help="Build zg_500+zg_50 native monthly stacks together via N parallel workers, then exit (much faster than --field zg_500/zg_50).")
    args = parser.parse_args()

    if not args.skip_compute_node_check:
        ensure_runtime_on_compute_node()

    ensure_dirs()

    if args.bootstrap_weights_only:
        bootstrap_weight_files()
        return

    global ALL_MONTH_KEYS
    if args.months:
        ALL_MONTH_KEYS = sorted(args.months)

    if args.build_zg_pair_parallel is not None:
        build_zg_pair_parallel(args.build_zg_pair_parallel)
        for field_id in ("zg_500", "zg_50"):
            spec = FIELD_SPECS_BY_ID[field_id]
            native_path = NATIVE_DIR / f"{field_id}_native_monthly.nc"
            regridded_path = regrid_field(spec, native_path)
            if not args.months:
                build_cnn_rows(spec, regridded_path)
        return

    fields = [FIELD_SPECS_BY_ID[f] for f in args.field] if args.field else list(FIELD_SPECS)

    for spec in fields:
        print(f"=== Processing {spec.field_id} ===", flush=True)
        native_path = build_field_native_monthly(spec)
        regridded_path = regrid_field(spec, native_path)
        if not args.months:
            build_cnn_rows(spec, regridded_path)
        else:
            print(f"[{spec.field_id}] --months test mode: skipping cnn_rows build (needs full Sep-Mar 1985-2021 coverage)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
