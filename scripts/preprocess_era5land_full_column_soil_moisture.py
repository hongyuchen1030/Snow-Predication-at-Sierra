#!/usr/bin/env python3
"""Build an ERA5-Land full-column soil-moisture counterpart for mrso."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr
try:
    import dask
    from dask.diagnostics import ProgressBar
    # On a many-core node (e.g. Perlmutter's 256-thread CPU nodes), dask's default threaded
    # scheduler auto-sizes its pool to the detected core count. netCDF4/HDF5 reads/writes are
    # serialized behind xarray's global lock regardless, so hundreds of competing threads only
    # add futex contention without any throughput benefit - observed in practice as the process
    # going fully idle (near-zero forward progress, near-zero CPU time) for over an hour.
    # A small fixed pool avoids this.
    dask.config.set(scheduler="threads", num_workers=8)
except ModuleNotFoundError:
    ProgressBar = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "era5land_full_column_soil_moisture_1p5deg"
OUTPUT_DIR = Path(os.environ.get("ERA5LAND_MRSO_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).expanduser()
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate_yearly_monthly"
MONTHLY_FILE = OUTPUT_DIR / "era5land_full_column_soil_moisture_monthly_1p5deg.nc"
WY_FILE = OUTPUT_DIR / "era5land_full_column_soil_moisture_wy1985_2021_sep_mar_1p5deg.nc"
SUMMARY_JSON = OUTPUT_DIR / "era5land_full_column_soil_moisture_summary.json"

ERA5_LAND_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5-Land")
START_YEAR = 1984
END_YEAR = 2021
WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
MONTH_LABELS = ("Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar")
TARGET_LON = np.arange(0.75, 360.0, 1.5, dtype=np.float64)
TARGET_LAT = np.arange(-89.25, 90.0, 1.5, dtype=np.float64)
WATER_DENSITY_KG_M3 = 1000.0
TIME_CHUNK = 744
LAT_CHUNK = 180
LON_CHUNK = 360

LAYER_SPECS = (
    ("swvl1", "volumetric_soil_water_layer_1", 0.07),
    ("swvl2", "volumetric_soil_water_layer_2", 0.21),
    ("swvl3", "volumetric_soil_water_layer_3", 0.72),
    ("swvl4", "volumetric_soil_water_layer_4", 1.89),
)


@dataclass(frozen=True)
class SummaryPayload:
    monthly_output: str
    water_year_output: str
    summary_json: str
    year_range: list[int]
    water_year_range: list[int]
    monthly_time_count: int
    spatial_grid_shape: list[int]
    layer_thickness_m: dict[str, float]
    layer_source_files: dict[str, str]
    output_units: str
    runtime_hostname: str
    slurm_job_id: str


def runtime_hostname() -> str:
    return os.uname().nodename


def ensure_runtime_on_compute_node() -> None:
    hostname = runtime_hostname()
    if "nid" not in hostname or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Active compute-node allocation required; do not run this on a login node.")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)


def source_path(folder: str, year: int) -> Path:
    return ERA5_LAND_ROOT / folder / f"ERA5_{year}_{folder}.nc"


def open_layer(path: Path, variable_name: str) -> xr.DataArray:
    open_kwargs = {"engine": "netcdf4", "decode_times": True}
    if ProgressBar is not None:
        open_kwargs["chunks"] = {"time": TIME_CHUNK, "latitude": LAT_CHUNK, "longitude": LON_CHUNK}
    ds = xr.open_dataset(path, **open_kwargs)
    return ds[variable_name]


def monthly_full_column_for_year(year: int) -> xr.Dataset:
    combined: xr.DataArray | None = None
    source_files: dict[str, str] = {}
    for variable_name, folder, thickness_m in LAYER_SPECS:
        path = source_path(folder, year)
        if not path.exists():
            raise FileNotFoundError(f"Missing ERA5-Land source file for {variable_name} {year}: {path}")
        source_files[variable_name] = str(path)
        layer = open_layer(path, variable_name)
        monthly = layer.resample(time="MS").mean(keep_attrs=True).astype(np.float32)
        layer_mass = monthly * np.float32(thickness_m * WATER_DENSITY_KG_M3)
        combined = layer_mass if combined is None else (combined + layer_mass)

    assert combined is not None
    combined = combined.rename("mrso_full_column")
    combined.attrs["description"] = "ERA5-Land monthly full-column soil moisture integrated from swvl1-swvl4"
    combined.attrs["units"] = "kg m-2"
    combined.attrs["integration"] = "sum(swvl_i * layer_thickness_m * 1000 kg m-3)"
    combined.attrs["source_layers"] = ", ".join(spec[0] for spec in LAYER_SPECS)
    combined = combined.sortby("latitude")
    regridded = combined.interp(latitude=TARGET_LAT, longitude=TARGET_LON, method="linear")
    regridded = regridded.astype(np.float32)
    valid_mask = xr.where(np.isfinite(regridded), np.uint8(1), np.uint8(0)).rename("valid_mask")
    ds_out = xr.Dataset({"mrso_full_column": regridded, "valid_mask": valid_mask})
    ds_out.attrs["description"] = "ERA5-Land monthly full-column soil moisture on the shared 1.5 degree grid"
    for variable_name, path in source_files.items():
        ds_out.attrs[f"source_{variable_name}"] = path
    return ds_out


def write_dataset(ds: xr.Dataset, path: Path) -> None:
    encoding = {}
    for var_name, data_array in ds.data_vars.items():
        dims = tuple(data_array.dims)
        if dims == ("time", "latitude", "longitude"):
            chunksizes = [12, len(TARGET_LAT), len(TARGET_LON)]
        elif dims == ("water_year", "month_in_model_year", "latitude", "longitude"):
            chunksizes = [1, len(MONTH_LABELS), len(TARGET_LAT), len(TARGET_LON)]
        else:
            raise ValueError(f"Unexpected variable dims for {var_name}: {dims}")
        encoding[var_name] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32" if var_name == "mrso_full_column" else "uint8",
            "chunksizes": chunksizes,
            "_FillValue": np.float32(np.nan) if var_name == "mrso_full_column" else np.uint8(0),
        }
    if ProgressBar is None:
        ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
        return
    delayed = ds.to_netcdf(path, engine="netcdf4", encoding=encoding, compute=False)
    with ProgressBar():
        delayed.compute()


def yearly_path(year: int) -> Path:
    return INTERMEDIATE_DIR / f"era5land_full_column_soil_moisture_{year:04d}_monthly_1p5deg.nc"


def build_monthly_files() -> list[Path]:
    yearly_paths: list[Path] = []
    for year in range(START_YEAR, END_YEAR + 1):
        out_path = yearly_path(year)
        yearly_paths.append(out_path)
        if out_path.exists() and _yearly_file_is_valid(out_path):
            print(f"Using existing yearly monthly file for {year}: {out_path}", flush=True)
            continue
        print(f"Building monthly full-column soil moisture for {year}", flush=True)
        ds_year = monthly_full_column_for_year(year)
        finite_frac = float(np.isfinite(ds_year["mrso_full_column"].values).mean())
        if finite_frac < MIN_VALID_FINITE_FRACTION:
            raise RuntimeError(
                f"Year {year}: computed full-column soil moisture is {finite_frac:.4%} finite "
                f"(expected roughly global land fraction, ~25-40%) - refusing to cache a bad result."
            )
        write_dataset(ds_year, out_path)
        ds_year.close()
    return yearly_paths


def _build_one_year_worker(year: int) -> tuple[int, str]:
    """Picklable per-year worker for ProcessPoolExecutor. Different years read completely
    independent source files (ERA5_<year>_<layer>.nc), so process-level parallelism here has no
    cross-worker HDF5 lock contention - unlike dask's threaded scheduler sharing one process's
    global lock across hundreds of threads, which is what caused the earlier hang."""
    out_path = yearly_path(year)
    if out_path.exists() and _yearly_file_is_valid(out_path):
        return year, "reused"
    ds_year = monthly_full_column_for_year(year)
    finite_frac = float(np.isfinite(ds_year["mrso_full_column"].values).mean())
    if finite_frac < MIN_VALID_FINITE_FRACTION:
        raise RuntimeError(
            f"Year {year}: computed full-column soil moisture is {finite_frac:.4%} finite "
            f"(expected roughly global land fraction, ~25-40%). This has been observed when too "
            f"many years are built concurrently on one node (silent data corruption under heavy "
            f"resource contention, not a clean crash) - refusing to cache a bad result. Reduce "
            f"--year-workers and retry this year."
        )
    write_dataset(ds_year, out_path)
    ds_year.close()
    return year, "built"


MIN_VALID_FINITE_FRACTION = 0.05


def _yearly_file_is_valid(path: Path) -> bool:
    """Guards against trusting a cached file that looks present but was actually corrupted -
    observed in practice: full-year builds run at high (12-16 way) process concurrency silently
    produced all-NaN output instead of raising, and the naive exists()-only cache check accepted
    them. Re-validated every time a cached file is about to be reused, not just on first build."""
    try:
        with xr.open_dataset(path) as ds:
            finite_frac = float(np.isfinite(ds["mrso_full_column"].values).mean())
    except Exception:
        return False
    return finite_frac >= MIN_VALID_FINITE_FRACTION


def build_monthly_files_parallel(n_workers: int) -> list[Path]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    years = list(range(START_YEAR, END_YEAR + 1))
    pending = [y for y in years if not (yearly_path(y).exists() and _yearly_file_is_valid(yearly_path(y)))]
    for y in years:
        if y not in pending:
            print(f"Using existing yearly monthly file for {y}: {yearly_path(y)}", flush=True)
    if pending:
        print(f"Building {len(pending)} missing years in parallel with {n_workers} workers: {pending}", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_build_one_year_worker, y): y for y in pending}
            completed = 0
            for future in as_completed(futures):
                year, status = future.result()
                completed += 1
                print(f"[mrso_year] {year} {status} ({completed}/{len(pending)})", flush=True)
    return [yearly_path(y) for y in years]


def combine_yearly_monthly(paths: list[Path]) -> xr.Dataset:
    open_kwargs = {"engine": "netcdf4", "decode_times": True}
    if ProgressBar is not None:
        open_kwargs["chunks"] = {"time": 12, "latitude": len(TARGET_LAT), "longitude": len(TARGET_LON)}
    datasets = [xr.open_dataset(path, **open_kwargs) for path in paths]
    combined = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override", combine_attrs="override")
    combined = combined.sortby("time")
    combined.attrs["description"] = "ERA5-Land monthly full-column soil moisture on the shared 1.5 degree grid"
    return combined


def build_water_year_pack(monthly_ds: xr.Dataset) -> xr.Dataset:
    rows = []
    masks = []
    for water_year in range(WATER_YEAR_START, WATER_YEAR_END + 1):
        months = [
            np.datetime64(f"{water_year - 1}-09-01"),
            np.datetime64(f"{water_year - 1}-10-01"),
            np.datetime64(f"{water_year - 1}-11-01"),
            np.datetime64(f"{water_year - 1}-12-01"),
            np.datetime64(f"{water_year}-01-01"),
            np.datetime64(f"{water_year}-02-01"),
            np.datetime64(f"{water_year}-03-01"),
        ]
        subset = monthly_ds["mrso_full_column"].sel(time=months)
        subset_mask = monthly_ds["valid_mask"].sel(time=months)
        if subset.sizes["time"] != len(MONTH_LABELS):
            raise ValueError(f"Missing Sep-Mar monthly full-column soil moisture for water year {water_year}")
        rows.append(subset.values.astype(np.float32))
        masks.append(subset_mask.values.astype(np.uint8))

    data = np.stack(rows, axis=0)
    mask = np.stack(masks, axis=0)
    ds_out = xr.Dataset(
        data_vars={
            "mrso_full_column": (
                ("water_year", "month_in_model_year", "latitude", "longitude"),
                data,
            ),
            "valid_mask": (
                ("water_year", "month_in_model_year", "latitude", "longitude"),
                mask,
            ),
        },
        coords={
            "water_year": np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32),
            "month_in_model_year": np.arange(len(MONTH_LABELS), dtype=np.int16),
            "month_label": ("month_in_model_year", np.asarray(MONTH_LABELS, dtype=object)),
            "latitude": TARGET_LAT.astype(np.float32),
            "longitude": TARGET_LON.astype(np.float32),
        },
    )
    ds_out["mrso_full_column"].attrs["units"] = "kg m-2"
    ds_out["mrso_full_column"].attrs["description"] = "ERA5-Land Sep-Mar full-column soil moisture packed by water year"
    ds_out["valid_mask"].attrs["description"] = "Finite-value mask for ERA5-Land Sep-Mar full-column soil moisture"
    ds_out.attrs["description"] = "ERA5-Land full-column soil moisture counterpart for observation-compatible S0-17 inference"
    return ds_out


def write_summary(monthly_ds: xr.Dataset) -> None:
    payload = SummaryPayload(
        monthly_output=str(MONTHLY_FILE),
        water_year_output=str(WY_FILE),
        summary_json=str(SUMMARY_JSON),
        year_range=[START_YEAR, END_YEAR],
        water_year_range=[WATER_YEAR_START, WATER_YEAR_END],
        monthly_time_count=int(monthly_ds.sizes["time"]),
        spatial_grid_shape=[int(monthly_ds.sizes["latitude"]), int(monthly_ds.sizes["longitude"])],
        layer_thickness_m={name: thickness for name, _folder, thickness in LAYER_SPECS},
        layer_source_files={name: str(source_path(folder, START_YEAR)) for name, folder, _ in LAYER_SPECS},
        output_units="kg m-2",
        runtime_hostname=runtime_hostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    SUMMARY_JSON.write_text(json.dumps(asdict(payload), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year-workers", type=int, default=None, help="Parallelize per-year builds across N processes instead of running sequentially.")
    args = parser.parse_args()

    ensure_runtime_on_compute_node()
    ensure_dirs()
    if args.year_workers:
        yearly_paths = build_monthly_files_parallel(args.year_workers)
    else:
        yearly_paths = build_monthly_files()
    monthly_ds = combine_yearly_monthly(yearly_paths)
    write_dataset(monthly_ds, MONTHLY_FILE)
    wy_ds = build_water_year_pack(monthly_ds)
    write_dataset(wy_ds, WY_FILE)
    write_summary(monthly_ds)
    print(f"Wrote monthly full-column soil moisture: {MONTHLY_FILE}", flush=True)
    print(f"Wrote WY1985-WY2021 Sep-Mar pack: {WY_FILE}", flush=True)
    print(f"Wrote summary: {SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
