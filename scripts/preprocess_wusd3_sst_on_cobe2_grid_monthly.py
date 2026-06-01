#!/usr/bin/env python3
"""
Build reusable monthly WUS-D3 SST artifacts on the COBE2 global EOF grid.

This is the minimum missing preprocessing needed for WUS-on-COBE2 projection:
1. Aggregate daily WUS-D3 `tskin` to monthly means.
2. Apply the native WUS domain ocean mask.
3. Interpolate to the COBE2 global EOF grid.
4. Apply the COBE2 valid ocean mask.
5. Save monthly mean, month-of-year climatology, and monthly anomalies.
"""

import argparse
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WUSD3_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily")
WRFINPUT_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/WUS-D3")
COBE2_EOF_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "sst_pca"
    / "cobe2_global_monthly_climatology_anomaly"
    / "cobe2_global_monthly_clim_sst_eofs.nc"
)
DEFAULT_DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
DEFAULT_DOMAIN = "d01"
DEFAULT_VARIABLE = "tskin"


@dataclass(frozen=True)
class GridData:
    latitude: np.ndarray
    longitude: np.ndarray
    ocean_mask: np.ndarray


@dataclass(frozen=True)
class Cobe2Grid:
    latitude: np.ndarray
    longitude: np.ndarray
    valid_mask: np.ndarray


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
    cobe2_grid_shape: List[int]
    n_valid_ocean_cells: int
    units: str
    anomaly_definition: str
    reused_existing_monthly_mean: bool


def default_output_dir() -> Path:
    pscratch = os.environ.get("PSCRATCH", "")
    if pscratch:
        return Path(pscratch) / "Snow-Predication-at-Sierra" / "artifacts" / "wusd3_sst_on_cobe2_grid_monthly"
    return PROJECT_ROOT / "artifacts" / "wusd3_sst_on_cobe2_grid_monthly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable monthly WUS-D3 SST on the COBE2 EOF grid.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="WUS-D3 dataset id.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="WUS-D3 domain, default d01.")
    parser.add_argument("--variable-name", default=DEFAULT_VARIABLE, help="Variable name, default tskin.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Output directory.")
    parser.add_argument(
        "--reuse-existing-monthly-mean",
        action="store_true",
        help="Reuse an existing remapped monthly mean file if present.",
    )
    return parser.parse_args()


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    errors = []
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append("%s: %s" % (engine or "default", exc))
    raise RuntimeError("failed to open %s: %s" % (path, "; ".join(errors)))


def format_date(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def find_wusd3_files(dataset_id: str, domain: str, variable_name: str) -> List[Path]:
    base_dir = WUSD3_ROOT / dataset_id / "postprocess" / domain
    paths = sorted(base_dir.glob("%s.daily.*.nc" % variable_name))
    if not paths:
        raise FileNotFoundError("No %s files found under %s" % (variable_name, base_dir))
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
        ocean_mask=land_mask_raw == 0,
    )


def load_cobe2_grid() -> Cobe2Grid:
    with open_dataset_with_fallbacks(COBE2_EOF_FILE) as ds:
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        valid_mask = np.asarray(ds["valid_mask"].values, dtype=bool)
    return Cobe2Grid(latitude=latitude, longitude=longitude, valid_mask=valid_mask)


def year_from_path(path: Path) -> int:
    match = re.search(r"\.(\d{4})\.nc$", path.name)
    if not match:
        raise ValueError("Could not infer year from %s" % path)
    return int(match.group(1))


def intermediate_year_file(output_dir: Path, year: int) -> Path:
    return output_dir / "intermediate_monthly_years" / ("wusd3_sst_on_cobe2_monthly_mean_%04d.nc" % year)


def process_year_file(
    path: Path,
    variable_name: str,
    wus_grid: GridData,
    cobe2_grid: Cobe2Grid,
    output_dir: Path,
) -> Path:
    year = year_from_path(path)
    target = intermediate_year_file(output_dir, year)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    print("Aggregating and remapping %s" % path.name, flush=True)
    with open_dataset_with_fallbacks(path) as ds:
        monthly = ds[[variable_name]].rename({"day": "time"}).resample(time="MS").mean(keep_attrs=True)
        values = np.asarray(monthly[variable_name].values, dtype=np.float64)
        values = np.where(wus_grid.ocean_mask[np.newaxis, :, :], values, np.nan)

        source_points = np.column_stack(
            [wus_grid.latitude[wus_grid.ocean_mask], wus_grid.longitude[wus_grid.ocean_mask]]
        )
        target_lon, target_lat = np.meshgrid(cobe2_grid.longitude, cobe2_grid.latitude)
        target_points = np.column_stack([target_lat.ravel(), target_lon.ravel()])

        source_values = np.asarray(values[:, wus_grid.ocean_mask], dtype=np.float64).T
        interpolator = LinearNDInterpolator(source_points, source_values, fill_value=np.nan)
        remapped = interpolator(target_points).T.reshape(
            values.shape[0],
            cobe2_grid.latitude.size,
            cobe2_grid.longitude.size,
        )
        remapped = np.where(cobe2_grid.valid_mask[np.newaxis, :, :], remapped, np.nan).astype(np.float32)

        monthly_ds = xr.Dataset(
            data_vars={variable_name: (("time", "lat", "lon"), remapped)},
            coords={
                "time": np.asarray(monthly["time"].values, dtype="datetime64[ns]"),
                "lat": cobe2_grid.latitude.astype(np.float32),
                "lon": cobe2_grid.longitude.astype(np.float32),
                "valid_mask": (("lat", "lon"), cobe2_grid.valid_mask.astype(np.int8)),
            },
            attrs={
                "source_file": str(path),
                "description": "WUS-D3 monthly mean SST remapped to the COBE2 global EOF grid",
            },
        )
        monthly_ds[variable_name].attrs.update(monthly[variable_name].attrs)
        monthly_ds[variable_name].attrs["units"] = monthly[variable_name].attrs.get("units", "K")
        monthly_ds.to_netcdf(target)
    return target


def load_combined_monthly_mean(paths: Sequence[Path]) -> xr.Dataset:
    datasets = []
    try:
        for path in sorted(paths):
            ds = open_dataset_with_fallbacks(path)
            datasets.append(ds)
        return xr.concat(datasets, dim="time").sortby("time").load()
    finally:
        for ds in datasets:
            ds.close()


def build_monthly_climatology(monthly_mean: xr.DataArray) -> xr.DataArray:
    climatology = monthly_mean.groupby("time.month").mean(dim="time", skipna=True)
    climatology.name = monthly_mean.name
    climatology.attrs.update(monthly_mean.attrs)
    climatology.attrs["description"] = "WUS-D3 month-of-year climatology of monthly mean SST on the COBE2 grid"
    return climatology.astype(np.float32)


def build_monthly_anomaly(monthly_mean: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        anomaly = monthly_mean.groupby("time.month") - climatology
    anomaly.name = "%s_anomaly" % monthly_mean.name
    anomaly.attrs.update(monthly_mean.attrs)
    anomaly.attrs["description"] = "WUS-D3 monthly mean SST anomaly relative to month-of-year climatology on the COBE2 grid"
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

    file_stem = "%s_%s_%s_on_cobe2_grid_monthly" % (args.dataset_id, args.domain, args.variable_name)
    monthly_mean_file = output_dir / (file_stem + "_mean.nc")
    monthly_clim_file = output_dir / (file_stem + "_climatology.nc")
    monthly_anom_file = output_dir / (file_stem + "_anomaly.nc")
    summary_file = output_dir / (file_stem + "_summary.json")

    wus_grid = load_wusd3_grid(args.domain)
    cobe2_grid = load_cobe2_grid()
    input_files = find_wusd3_files(args.dataset_id, args.domain, args.variable_name)

    if args.reuse_existing_monthly_mean:
        if not monthly_mean_file.exists():
            raise FileNotFoundError("Requested reuse, but monthly mean file is missing: %s" % monthly_mean_file)
        print("Reusing existing monthly mean file %s" % monthly_mean_file, flush=True)
    else:
        intermediate_paths = [
            process_year_file(path, args.variable_name, wus_grid, cobe2_grid, output_dir)
            for path in input_files
        ]
        combined = load_combined_monthly_mean(intermediate_paths)
        combined.attrs["source_files"] = ",".join(str(path) for path in input_files)
        combined.attrs["dataset_id"] = args.dataset_id
        combined.attrs["domain"] = args.domain
        print("Writing monthly mean file %s" % monthly_mean_file, flush=True)
        combined.to_netcdf(monthly_mean_file)

    with open_dataset_with_fallbacks(monthly_mean_file) as ds:
        monthly_mean = ds[args.variable_name].load()
        time_values = np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]")
        units = str(monthly_mean.attrs.get("units", "K"))
        climatology = build_monthly_climatology(monthly_mean)
        anomaly = build_monthly_anomaly(monthly_mean, climatology)

    print("Writing monthly climatology file %s" % monthly_clim_file, flush=True)
    dataset_for_output(
        climatology,
        "WUS-D3 month-of-year climatology from monthly mean SST on the COBE2 grid",
    ).to_netcdf(monthly_clim_file)

    print("Writing monthly anomaly file %s" % monthly_anom_file, flush=True)
    dataset_for_output(
        anomaly,
        "WUS-D3 monthly mean SST anomalies relative to month-of-year climatology on the COBE2 grid",
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
        cobe2_grid_shape=[int(cobe2_grid.latitude.size), int(cobe2_grid.longitude.size)],
        n_valid_ocean_cells=int(np.count_nonzero(cobe2_grid.valid_mask)),
        units=units,
        anomaly_definition="monthly mean minus month-of-year climatology computed separately for each calendar month",
        reused_existing_monthly_mean=bool(args.reuse_existing_monthly_mean),
    )
    write_summary(summary_file, summary)
    print("Summary JSON: %s" % summary_file, flush=True)


if __name__ == "__main__":
    main()
