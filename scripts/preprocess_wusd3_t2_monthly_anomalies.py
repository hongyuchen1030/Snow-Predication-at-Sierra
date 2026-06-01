#!/usr/bin/env python3
"""
Build reusable monthly WUS-D3 t2 artifacts from existing daily files.

This script intentionally does the minimum missing preprocessing needed for
monthly SST->T2m analysis:
1. Aggregate daily WUS-D3 `t2` to monthly means.
2. Apply the native WUS-D3 domain land mask.
3. Save monthly mean, month-of-year climatology, and monthly anomalies.

It processes one dataset at a time and writes isolated outputs so downstream
analysis can reuse them without re-reading daily files.
"""

import argparse
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WUSD3_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily")
WRFINPUT_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/WUS-D3")
DEFAULT_DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
DEFAULT_DOMAIN = "d01"
DEFAULT_VARIABLE = "t2"


@dataclass(frozen=True)
class GridData:
    latitude: np.ndarray
    longitude: np.ndarray
    land_mask: np.ndarray


@dataclass(frozen=True)
class SummaryPayload:
    dataset_id: str
    variable_name: str
    domain: str
    monthly_mean_file: str
    monthly_climatology_file: str
    monthly_anomaly_file: str
    summary_json_file: str
    input_files: List[str]
    time_start: str
    time_end: str
    n_time: int
    grid_shape: List[int]
    n_land_cells: int
    units: str
    anomaly_definition: str
    reused_existing_monthly_mean: bool


def default_output_dir() -> Path:
    pscratch = os.environ.get("PSCRATCH", "")
    if pscratch:
        return Path(pscratch) / "Snow-Predication-at-Sierra" / "artifacts" / "wusd3_t2_monthly_anomalies"
    return PROJECT_ROOT / "artifacts" / "wusd3_t2_monthly_anomalies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable monthly WUS-D3 t2 artifacts.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="WUS-D3 dataset id.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="WUS-D3 domain, default d01.")
    parser.add_argument("--variable-name", default=DEFAULT_VARIABLE, help="Variable name, default t2.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output directory for monthly artifacts.",
    )
    parser.add_argument(
        "--reuse-existing-monthly-mean",
        action="store_true",
        help="Skip daily aggregation and reuse an existing monthly mean file if present.",
    )
    return parser.parse_args()


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    errors: List[str] = []
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs: Dict[str, object] = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}: {exc}")
    raise RuntimeError(f"failed to open {path}: {'; '.join(errors)}")


def format_date(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def find_wusd3_files(dataset_id: str, domain: str, variable_name: str) -> List[Path]:
    base_dir = WUSD3_ROOT / dataset_id / "postprocess" / domain
    paths = sorted(base_dir.glob(f"{variable_name}.daily.*.nc"))
    if not paths:
        raise FileNotFoundError(f"No {variable_name} files found under {base_dir}")
    return paths


def wrfinput_path_for_domain(domain: str) -> Path:
    path = WRFINPUT_ROOT / ("wrfinput_%s" % domain)
    if not path.exists():
        raise FileNotFoundError("Missing WRF input file for domain %s: %s" % (domain, path))
    return path


def load_wusd3_grid(domain: str) -> GridData:
    with open_dataset_with_fallbacks(wrfinput_path_for_domain(domain)) as ds:
        latitude = np.asarray(ds["XLAT"].isel(Time=0).values, dtype=np.float64)
        longitude = np.asarray(ds["XLONG"].isel(Time=0).values, dtype=np.float64)
        land_mask_raw = np.asarray(ds["LANDMASK"].isel(Time=0).values, dtype=np.int8)
    return GridData(
        latitude=latitude,
        longitude=longitude,
        land_mask=land_mask_raw == 1,
    )


def year_from_path(path: Path) -> int:
    match = re.search(r"\.(\d{4})\.nc$", path.name)
    if not match:
        raise ValueError(f"Could not infer year from {path}")
    return int(match.group(1))


def intermediate_year_file(output_dir: Path, year: int) -> Path:
    return output_dir / "intermediate_monthly_years" / f"wusd3_t2_monthly_mean_{year:04d}.nc"


def process_year_file(path: Path, variable_name: str, grid: GridData, output_dir: Path) -> Path:
    year = year_from_path(path)
    target = intermediate_year_file(output_dir, year)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    print(f"Aggregating monthly means for {path.name}", flush=True)
    with open_dataset_with_fallbacks(path) as ds:
        if variable_name not in ds:
            raise KeyError(f"Expected variable {variable_name!r} in {path}")
        monthly = ds[[variable_name]].rename({"day": "time"}).resample(time="MS").mean(keep_attrs=True)
        values = np.asarray(monthly[variable_name].values, dtype=np.float32)
        values = np.where(grid.land_mask[np.newaxis, :, :], values, np.nan)
        monthly_ds = xr.Dataset(
            data_vars={
                variable_name: (("time", "lat2d", "lon2d"), values),
            },
            coords={
                "time": np.asarray(monthly["time"].values, dtype="datetime64[ns]"),
                "lat2d": np.arange(grid.latitude.shape[0], dtype=np.int32),
                "lon2d": np.arange(grid.latitude.shape[1], dtype=np.int32),
                "latitude": (("lat2d", "lon2d"), grid.latitude.astype(np.float32)),
                "longitude": (("lat2d", "lon2d"), grid.longitude.astype(np.float32)),
                "landmask": (("lat2d", "lon2d"), grid.land_mask.astype(np.int8)),
            },
            attrs={
                "source_file": str(path),
                "description": "WUS-D3 monthly mean 2m temperature on native domain land cells",
            },
        )
        monthly_ds[variable_name].attrs.update(monthly[variable_name].attrs)
        monthly_ds[variable_name].attrs["units"] = monthly[variable_name].attrs.get("units", "K")
        monthly_ds.to_netcdf(target)
    return target


def load_combined_monthly_mean(paths: Sequence[Path], variable_name: str) -> xr.Dataset:
    datasets: List[xr.Dataset] = []
    try:
        for path in sorted(paths):
            ds = open_dataset_with_fallbacks(path)
            datasets.append(ds)
        combined = xr.concat(datasets, dim="time").sortby("time")
        return combined.load()
    finally:
        for ds in datasets:
            ds.close()


def build_monthly_climatology(monthly_mean: xr.DataArray) -> xr.DataArray:
    climatology = monthly_mean.groupby("time.month").mean(dim="time", skipna=True)
    climatology.name = monthly_mean.name
    climatology.attrs.update(monthly_mean.attrs)
    climatology.attrs["description"] = "WUS-D3 month-of-year climatology of monthly mean 2m temperature"
    return climatology.astype(np.float32)


def build_monthly_anomaly(monthly_mean: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        anomaly = monthly_mean.groupby("time.month") - climatology
    anomaly.name = f"{monthly_mean.name}_anomaly"
    anomaly.attrs.update(monthly_mean.attrs)
    anomaly.attrs["description"] = "WUS-D3 monthly mean 2m temperature anomaly relative to month-of-year climatology"
    return anomaly.astype(np.float32)


def dataset_for_output(data_array: xr.DataArray, description: str) -> xr.Dataset:
    ds = data_array.to_dataset(name=data_array.name)
    ds.attrs["description"] = description
    return ds


def write_summary(path: Path, summary: SummaryPayload) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.domain / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stem = "%s_%s_%s_monthly" % (args.dataset_id, args.domain, args.variable_name)
    monthly_mean_file = output_dir / ("%s_mean.nc" % file_stem)
    monthly_clim_file = output_dir / ("%s_climatology.nc" % file_stem)
    monthly_anom_file = output_dir / ("%s_anomaly.nc" % file_stem)
    summary_file = output_dir / ("%s_summary.json" % file_stem)

    grid = load_wusd3_grid(args.domain)
    input_files = find_wusd3_files(args.dataset_id, args.domain, args.variable_name)

    if args.reuse_existing_monthly_mean:
        if not monthly_mean_file.exists():
            raise FileNotFoundError(f"Requested reuse, but monthly mean file is missing: {monthly_mean_file}")
        print(f"Reusing existing monthly mean file {monthly_mean_file}", flush=True)
    else:
        intermediate_paths = [
            process_year_file(path, args.variable_name, grid, output_dir)
            for path in input_files
        ]
        combined = load_combined_monthly_mean(intermediate_paths, args.variable_name)
        combined.attrs["source_files"] = ",".join(str(path) for path in input_files)
        combined.attrs["dataset_id"] = args.dataset_id
        combined.attrs["domain"] = args.domain
        print(f"Writing monthly mean file {monthly_mean_file}", flush=True)
        combined.to_netcdf(monthly_mean_file)

    with open_dataset_with_fallbacks(monthly_mean_file) as ds:
        monthly_mean = ds[args.variable_name].load()
        time_values = np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]")
        units = str(monthly_mean.attrs.get("units", "K"))
        climatology = build_monthly_climatology(monthly_mean)
        anomaly = build_monthly_anomaly(monthly_mean, climatology)

    print(f"Writing monthly climatology file {monthly_clim_file}", flush=True)
    dataset_for_output(
        climatology,
        "WUS-D3 month-of-year climatology from monthly mean 2m temperature",
    ).to_netcdf(monthly_clim_file)

    print(f"Writing monthly anomaly file {monthly_anom_file}", flush=True)
    dataset_for_output(
        anomaly,
        "WUS-D3 monthly mean 2m temperature anomalies relative to month-of-year climatology",
    ).to_netcdf(monthly_anom_file)

    summary = SummaryPayload(
        dataset_id=args.dataset_id,
        variable_name=args.variable_name,
        domain=args.domain,
        monthly_mean_file=str(monthly_mean_file),
        monthly_climatology_file=str(monthly_clim_file),
        monthly_anomaly_file=str(monthly_anom_file),
        summary_json_file=str(summary_file),
        input_files=[str(path) for path in input_files],
        time_start=format_date(time_values[0]),
        time_end=format_date(time_values[-1]),
        n_time=int(time_values.size),
        grid_shape=[int(grid.latitude.shape[0]), int(grid.latitude.shape[1])],
        n_land_cells=int(np.count_nonzero(grid.land_mask)),
        units=units,
        anomaly_definition="monthly mean minus month-of-year climatology computed separately for each calendar month",
        reused_existing_monthly_mean=bool(args.reuse_existing_monthly_mean),
    )
    write_summary(summary_file, summary)
    print(f"Summary JSON: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
