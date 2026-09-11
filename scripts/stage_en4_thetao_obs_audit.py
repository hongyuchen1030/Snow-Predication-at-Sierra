#!/usr/bin/env python3
"""Stage EN4 thetao inputs for observational CNN evaluation without touching production code.

This script mirrors the repository's thetao handling as closely as possible:
- exact depths 50 m and 100 m via linear interpolation in depth
- exact target grid centers from the CMIP6 preprocessing pipeline
- model-year month order Sep..Aug, with the CNN consuming Sep..Mar

The script is intentionally standalone and read-only with respect to existing
production datasets. It supports three small, staged actions:

1. Download a minimal set of EN4 yearly archives into scratch staging.
2. Build processed EN4 thetao_50m/thetao_100m fields on the CNN 1.5 degree grid.
3. Compare representative overlap months against the local CMEMS product.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from process_cmip6_selected_regrid_standardize import TARGET_LAT, TARGET_LON, TARGET_MONTH_LABELS, TARGET_MONTHS


EN4_BASE_URL = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/EN.4.2.2"
EN4_ARCHIVE_TEMPLATE = "EN.4.2.2.analyses.g10.{year}.zip"
DEFAULT_STAGE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/en4_thetao_stage")
DEFAULT_CMEMS_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/CMEMS")
DEFAULT_OUTPUT_DIR = DEFAULT_STAGE_DIR / "processed"
DEFAULT_COMPARISON_MONTHS = ("1993-09", "2003-01", "2013-03")
CMEMS_THETAO_VARS = ("thetao_cglo", "thetao_glor", "thetao_oras")


@dataclass(frozen=True)
class MonthRequest:
    year: int
    month: int

    @property
    def year_month(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage-en4")
    stage.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    stage.add_argument("--base-url", default=EN4_BASE_URL)
    stage.add_argument("--year", action="append", type=int, required=True)
    stage.add_argument("--overwrite", action="store_true")

    build = subparsers.add_parser("build-en4-thetao")
    build.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--month", action="append", default=None, help="YYYY-MM")
    build.add_argument("--start-month", help="Inclusive YYYY-MM; use with --end-month")
    build.add_argument("--end-month", help="Inclusive YYYY-MM; use with --start-month")
    build.add_argument("--build-model-years", action="store_true")
    build.add_argument("--build-cnn-rows", action="store_true")
    build.add_argument("--start-row-year", type=int, default=1984)
    build.add_argument("--end-row-year", type=int, default=2020)
    build.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare-cmems")
    compare.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    compare.add_argument("--cmems-root", type=Path, default=DEFAULT_CMEMS_ROOT)
    compare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "cmems_overlap")
    compare.add_argument("--month", action="append", default=list(DEFAULT_COMPARISON_MONTHS), help="YYYY-MM")
    compare.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_month_strings(months: Iterable[str]) -> list[MonthRequest]:
    out: list[MonthRequest] = []
    for value in months:
        year_s, month_s = value.split("-", 1)
        out.append(MonthRequest(year=int(year_s), month=int(month_s)))
    return out


def month_range(start_month: str, end_month: str) -> list[MonthRequest]:
    """Return every calendar month in an inclusive YYYY-MM interval."""
    start = parse_month_strings([start_month])[0]
    end = parse_month_strings([end_month])[0]
    if (end.year, end.month) < (start.year, start.month):
        raise ValueError("--end-month must not be earlier than --start-month")
    months: list[MonthRequest] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(MonthRequest(year=year, month=month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def normalize_lon_to_360(values: np.ndarray) -> np.ndarray:
    out = np.mod(values, 360.0)
    out[np.isclose(out, 360.0)] = 0.0
    return out


def open_en4_month_from_zip(zip_path: Path, request: MonthRequest) -> xr.Dataset:
    member_name = f"EN.4.2.2.f.analysis.g10.{request.year:04d}{request.month:02d}.nc"
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if member_name not in names:
            available = ", ".join(sorted(list(names))[:6])
            raise FileNotFoundError(f"{member_name} not found in {zip_path}; sample members: {available}")
        with tempfile.TemporaryDirectory(prefix="en4-month-") as tmp:
            extracted = Path(zf.extract(member_name, path=tmp))
            ds = xr.open_dataset(extracted, decode_times=True)
            ds.load()
    return ds


def identify_depth_coord(da: xr.DataArray) -> str:
    for name in da.coords:
        lowered = name.lower()
        if lowered == "depth" or lowered.startswith("depth"):
            return name
    raise KeyError(f"Could not find depth coordinate in {list(da.coords)}")


def prepare_rectilinear_grid(da: xr.DataArray) -> xr.DataArray:
    rename_map: dict[str, str] = {}
    if "latitude" in da.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in da.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    lon = normalize_lon_to_360(np.asarray(da["lon"].values, dtype=np.float64))
    lat = np.asarray(da["lat"].values, dtype=np.float64)
    da = da.assign_coords(lon=lon, lat=lat)
    da = da.sortby("lon")
    da = da.sortby("lat")
    if da.sizes["lon"] > 1 and np.isclose(float(da["lon"].values[0]), float(da["lon"].values[-1])):
        da = da.isel(lon=slice(0, -1))
    return da


def extract_thetao_on_target_grid(ds: xr.Dataset, depth_m: float) -> xr.DataArray:
    da = ds["temperature"]
    depth_name = identify_depth_coord(da)
    work = da.assign_coords({depth_name: np.asarray(ds[depth_name].values, dtype=np.float64)})
    if np.any(np.diff(np.asarray(work[depth_name].values, dtype=np.float64)) < 0):
        work = work.sortby(depth_name)
    selected = work.interp({depth_name: depth_m}).squeeze(drop=True)
    units = str(selected.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin"}:
        selected = selected - 273.15
        selected.attrs["units"] = "degrees_C"
    selected = prepare_rectilinear_grid(selected)
    selected = selected.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")
    selected.name = f"thetao_{int(depth_m)}m"
    selected.attrs["source_dataset"] = "EN4.2.2.g10"
    selected.attrs["source_variable"] = "temperature"
    selected.attrs["source_definition"] = "sea_water_potential_temperature"
    selected.attrs["selected_depth_m"] = float(depth_m)
    selected.attrs["target_grid"] = "1.5 degree regular lon-lat"
    return selected.astype(np.float32)


def build_month_stack(month_requests: list[MonthRequest], stage_dir: Path) -> dict[str, xr.DataArray]:
    stacks: dict[str, list[xr.DataArray]] = {"thetao_50m": [], "thetao_100m": []}
    time_values: list[np.datetime64] = []
    for request in month_requests:
        zip_path = stage_dir / EN4_ARCHIVE_TEMPLATE.format(year=request.year)
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing staged EN4 archive: {zip_path}")
        ds = open_en4_month_from_zip(zip_path, request)
        time_values.append(np.datetime64(f"{request.year:04d}-{request.month:02d}-01"))
        stacks["thetao_50m"].append(extract_thetao_on_target_grid(ds, 50.0))
        stacks["thetao_100m"].append(extract_thetao_on_target_grid(ds, 100.0))
        ds.close()
    out: dict[str, xr.DataArray] = {}
    for field_name, items in stacks.items():
        out[field_name] = xr.concat(items, dim=xr.DataArray(np.asarray(time_values), dims=("time",), name="time"))
    return out


def save_processed_monthlies(output_dir: Path, stacks: dict[str, xr.DataArray]) -> None:
    ensure_dir(output_dir)
    for field_name, da in stacks.items():
        ds = da.to_dataset(name=field_name)
        ds.attrs["month_order_note"] = "Monthly time dimension; CNN later consumes Sep-Mar rows from model-year packing."
        path = output_dir / f"{field_name}_1p5deg_monthly.nc"
        encoding = {
            field_name: {
                "zlib": True,
                "complevel": 2,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
                "chunksizes": (1, len(TARGET_LAT), len(TARGET_LON)),
            }
        }
        ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def build_model_year_dataset(da: xr.DataArray, field_name: str, start_row_year: int, end_row_year: int) -> xr.Dataset:
    lookup = {
        (int(ts.astype("datetime64[Y]").astype(int) + 1970), int(str(ts)[5:7])): idx
        for idx, ts in enumerate(np.asarray(da["time"].values))
    }
    samples: list[np.ndarray] = []
    rows: list[int] = []
    for row_year in range(start_row_year, end_row_year + 1):
        wanted = [(row_year, 9), (row_year, 10), (row_year, 11), (row_year, 12)] + [(row_year + 1, m) for m in range(1, 9)]
        if not all(key in lookup for key in wanted):
            continue
        idx = [lookup[key] for key in wanted]
        samples.append(np.asarray(da.isel(time=idx).values, dtype=np.float32))
        rows.append(row_year)
    if not samples:
        raise RuntimeError(f"No complete model-year rows found for {field_name}")
    values = np.stack(samples, axis=0)
    return xr.Dataset(
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
                attrs={"description": "EN4 staged thetao values on the CNN 1.5 degree grid"},
            )
        },
        coords={
            "month_label": ("month_in_model_year", TARGET_MONTH_LABELS),
            "model": ("sample", np.asarray(["EN4.2.2.g10"] * len(rows), dtype=object)),
            "row_year": ("sample", np.asarray(rows, dtype=np.int16)),
        },
        attrs={
            "grid_resolution_degrees": "1.5 x 1.5",
            "month_order": "Sep,Oct,Nov,Dec,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug",
            "cnn_consumed_months": "Sep,Oct,Nov,Dec,Jan,Feb,Mar",
        },
    )


def build_cnn_row_dataset(da: xr.DataArray, field_name: str, start_row_year: int, end_row_year: int) -> xr.Dataset:
    cnn_months = np.asarray(TARGET_MONTHS[:7], dtype=np.int16)
    cnn_labels = np.asarray(TARGET_MONTH_LABELS[:7], dtype=object)
    lookup = {(int(str(ts)[:4]), int(str(ts)[5:7])): idx for idx, ts in enumerate(np.asarray(da["time"].values))}
    samples: list[np.ndarray] = []
    rows: list[int] = []
    for row_year in range(start_row_year, end_row_year + 1):
        wanted = [(row_year, 9), (row_year, 10), (row_year, 11), (row_year, 12)] + [(row_year + 1, m) for m in range(1, 4)]
        if not all(key in lookup for key in wanted):
            continue
        idx = [lookup[key] for key in wanted]
        samples.append(np.asarray(da.isel(time=idx).values, dtype=np.float32))
        rows.append(row_year + 1)
    if not samples:
        raise RuntimeError(f"No complete Sep-Mar CNN rows found for {field_name}")
    values = np.stack(samples, axis=0)
    return xr.Dataset(
        {
            field_name: xr.DataArray(
                values,
                dims=("sample", "month_in_model_year", "lat", "lon"),
                coords={
                    "sample": np.arange(values.shape[0], dtype=np.int32),
                    "month_in_model_year": cnn_months,
                    "lat": TARGET_LAT,
                    "lon": TARGET_LON,
                },
                attrs={"description": "EN4 staged thetao values in the exact Sep-Mar month layout consumed by the CNN"},
            )
        },
        coords={
            "month_label": ("month_in_model_year", cnn_labels),
            "model": ("sample", np.asarray(["EN4.2.2.g10"] * len(rows), dtype=object)),
            "water_year": ("sample", np.asarray(rows, dtype=np.int16)),
        },
        attrs={
            "grid_resolution_degrees": "1.5 x 1.5",
            "month_order": "Sep,Oct,Nov,Dec,Jan,Feb,Mar",
            "cnn_ready_note": "Per-variable physical values on the same month and grid structure consumed by S0/U1/S1/S2 inputs before channel stacking and mask concatenation.",
        },
    )


def verify_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as zf:
        members = sorted(zf.namelist())
    return {
        "path": str(path),
        "member_count": len(members),
        "first_member": members[0] if members else None,
        "last_member": members[-1] if members else None,
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_en4_year(stage_dir: Path, base_url: str, year: int, overwrite: bool) -> Path:
    ensure_dir(stage_dir)
    filename = EN4_ARCHIVE_TEMPLATE.format(year=year)
    dest = stage_dir / filename
    if dest.exists() and not overwrite:
        return dest
    url = f"{base_url}/{filename}"
    with urllib.request.urlopen(url) as response, dest.open("wb") as sink:
        shutil.copyfileobj(response, sink)
    return dest


def run_stage_en4(args: argparse.Namespace) -> None:
    for year in sorted(set(args.year)):
        download_en4_year(args.stage_dir, args.base_url, year, args.overwrite)
    # Rebuild a cumulative manifest so interrupted or deliberately batched
    # staging remains auditable as one acquisition.
    manifest_rows = [verify_zip(path) for path in sorted(args.stage_dir.glob("EN.4.2.2.analyses.g10.*.zip"))]
    manifest = {
        "access_date_utc": np.datetime_as_string(np.datetime64("now"), unit="s"),
        "dataset": "EN4.2.2.g10",
        "archives": manifest_rows,
    }
    write_json(args.stage_dir / "stage_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def list_monthly_cmems_files(cmems_root: Path, request: MonthRequest) -> list[Path]:
    month_dir = cmems_root / f"{request.year:04d}" / f"{request.month:02d}"
    if not month_dir.exists():
        raise FileNotFoundError(f"CMEMS month directory not found: {month_dir}")
    return sorted(month_dir.glob("cmems_mod_glo_phy-all_my_0.25deg_P1D-m-*.nc"))


def open_cmems_month_mean(cmems_root: Path, request: MonthRequest, depth_m: float) -> xr.DataArray:
    files = list_monthly_cmems_files(cmems_root, request)
    if not files:
        raise FileNotFoundError(f"No CMEMS daily files found for {request.year_month}")
    monthly_members: list[xr.DataArray] = []
    for var_name in CMEMS_THETAO_VARS:
        running_sum: xr.DataArray | None = None
        running_count: xr.DataArray | None = None
        template: xr.DataArray | None = None
        for path in files:
            with xr.open_dataset(path, decode_times=False, engine="netcdf4") as ds:
                daily = ds[var_name].isel(time=0).load()
            template = daily if template is None else template
            valid = xr.where(np.isfinite(daily), 1.0, 0.0).astype(np.float32)
            cleaned = daily.fillna(0.0).astype(np.float32)
            running_sum = cleaned if running_sum is None else (running_sum + cleaned)
            running_count = valid if running_count is None else (running_count + valid)
        assert template is not None and running_sum is not None and running_count is not None
        monthly_mean = xr.where(running_count > 0.0, running_sum / running_count, np.nan)
        monthly_mean.attrs = template.attrs.copy()
        monthly_members.append(monthly_mean)
    ensemble_mean = xr.concat(monthly_members, dim="member").mean(dim="member", skipna=True)
    ensemble_mean = prepare_rectilinear_grid(ensemble_mean.squeeze(drop=True))
    depth_name = identify_depth_coord(ensemble_mean)
    work = ensemble_mean.assign_coords({depth_name: np.asarray(ensemble_mean[depth_name].values, dtype=np.float64)})
    if np.any(np.diff(np.asarray(work[depth_name].values, dtype=np.float64)) < 0):
        work = work.sortby(depth_name)
    selected = work.interp({depth_name: depth_m}).squeeze(drop=True)
    selected = selected.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")
    selected.name = f"cmems_thetao_{int(depth_m)}m"
    selected.attrs["units"] = "degrees_C"
    return selected.astype(np.float32)


def compute_pair_metrics(reference: xr.DataArray, candidate: xr.DataArray) -> dict[str, float | int]:
    ref = np.asarray(reference.values, dtype=np.float64)
    cand = np.asarray(candidate.values, dtype=np.float64)
    mask = np.isfinite(ref) & np.isfinite(cand)
    if int(mask.sum()) < 2:
        return {"n": int(mask.sum()), "bias_degC": math.nan, "rmse_degC": math.nan, "spatial_corr": math.nan}
    ref_use = ref[mask]
    cand_use = cand[mask]
    bias = float(np.mean(cand_use - ref_use))
    rmse = float(np.sqrt(np.mean((cand_use - ref_use) ** 2)))
    corr = float(np.corrcoef(ref_use, cand_use)[0, 1])
    return {"n": int(mask.sum()), "bias_degC": bias, "rmse_degC": rmse, "spatial_corr": corr}


def run_build_en4_thetao(args: argparse.Namespace) -> None:
    if bool(args.start_month) != bool(args.end_month):
        raise ValueError("Provide both --start-month and --end-month together")
    if args.month and args.start_month:
        raise ValueError("Use either repeated --month values or --start-month/--end-month, not both")
    month_requests = (
        month_range(args.start_month, args.end_month)
        if args.start_month
        else parse_month_strings(args.month) if args.month else []
    )
    if not month_requests:
        raise ValueError("Provide at least one --month YYYY-MM for build-en4-thetao")
    stacks = build_month_stack(month_requests, args.stage_dir)
    save_processed_monthlies(args.output_dir, stacks)
    summary: dict[str, object] = {
        "access_date_utc": np.datetime_as_string(np.datetime64("now"), unit="s"),
        "dataset": "EN4.2.2.g10",
        "months": [request.year_month for request in month_requests],
        "outputs": {},
    }
    for field_name, da in stacks.items():
        summary["outputs"][field_name] = {
            "path": str(args.output_dir / f"{field_name}_1p5deg_monthly.nc"),
            "shape": list(da.shape),
            "units": str(da.attrs.get("units", "")),
            "lat_size": int(da.sizes["lat"]),
            "lon_size": int(da.sizes["lon"]),
        }
        if args.build_model_years:
            ds_model_year = build_model_year_dataset(da, field_name, args.start_row_year, args.end_row_year)
            out_path = args.output_dir / f"{field_name}_1p5deg_model_years.nc"
            encoding = {
                field_name: {
                    "zlib": True,
                    "complevel": 2,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan),
                    "chunksizes": (1, 12, len(TARGET_LAT), len(TARGET_LON)),
                }
            }
            ds_model_year.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
            ds_model_year.close()
            summary["outputs"][field_name]["model_year_path"] = str(out_path)
        if args.build_cnn_rows:
            ds_cnn = build_cnn_row_dataset(da, field_name, args.start_row_year, args.end_row_year)
            cnn_path = args.output_dir / f"{field_name}_1p5deg_cnn_rows.nc"
            encoding = {
                field_name: {
                    "zlib": True,
                    "complevel": 2,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan),
                    "chunksizes": (1, 7, len(TARGET_LAT), len(TARGET_LON)),
                }
            }
            ds_cnn.to_netcdf(cnn_path, engine="netcdf4", encoding=encoding)
            ds_cnn.close()
            summary["outputs"][field_name]["cnn_row_path"] = str(cnn_path)
    write_json(args.output_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_compare_cmems(args: argparse.Namespace) -> None:
    month_requests = parse_month_strings(args.month)
    ensure_dir(args.output_dir)
    metrics_rows: list[dict[str, object]] = []
    per_month_dir = args.output_dir / "per_month"
    ensure_dir(per_month_dir)
    for request in month_requests:
        stacks = build_month_stack([request], args.stage_dir)
        en4_50 = stacks["thetao_50m"].isel(time=0)
        en4_100 = stacks["thetao_100m"].isel(time=0)
        cmems_50 = open_cmems_month_mean(args.cmems_root, request, 50.0)
        cmems_100 = open_cmems_month_mean(args.cmems_root, request, 100.0)
        month_metrics = {
            "month": request.year_month,
            "thetao_50m": compute_pair_metrics(cmems_50, en4_50),
            "thetao_100m": compute_pair_metrics(cmems_100, en4_100),
        }
        write_json(per_month_dir / f"{request.year_month}.json", month_metrics)
        metrics_rows.append(month_metrics)
    summary = {
        "access_date_utc": np.datetime_as_string(np.datetime64("now"), unit="s"),
        "dataset_candidate": "EN4.2.2.g10",
        "reference_dataset": "CMEMS GLOBAL_MULTIYEAR_PHY_ENS_001_031 daily ensemble mean",
        "months": [request.year_month for request in month_requests],
        "metrics": metrics_rows,
    }
    write_json(args.output_dir / "comparison_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "stage-en4":
        run_stage_en4(args)
    elif args.command == "build-en4-thetao":
        run_build_en4_thetao(args)
    elif args.command == "compare-cmems":
        run_compare_cmems(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
