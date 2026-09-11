#!/usr/bin/env python3
"""Regrid ERA5-Land 2m temperature onto the NeuralGCM coarse grid."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
SCRATCH_ROOT = Path(os.environ.get("PSCRATCH", "/pscratch/sd/h/hyvchen"))
NEURALGCM_ROOT = SCRATCH_ROOT / "Snow-Predication-at-Sierra_neuralgcm"
ERA5_LAND_T2M_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5-Land/2m_temperature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-year", type=int, required=True)
    parser.add_argument("--actual-label", required=True, help="Prepared NeuralGCM input label, e.g. wy1998_actual_m001")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=31,
        help="Number of source days to process per chunk before regridding.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="PSCRATCH output root. Defaults to assets/regridded_aux/era5land_t2m_neuralgcm_grid/wyXXXX.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dt64_to_datetime(value: np.datetime64) -> datetime:
    return datetime.utcfromtimestamp((value - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))


def datetime_to_dt64(value: datetime) -> np.datetime64:
    return np.datetime64(value.isoformat())


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=to_serializable) + "\n", encoding="utf-8")


def log(message: str) -> None:
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)
    sys.stdout.flush()


def make_grid(ds: xr.Dataset | xr.DataArray, lat_name: str, lon_name: str) -> spherical_harmonic.Grid:
    return spherical_harmonic.Grid(
        latitude_nodes=ds.sizes[lat_name],
        longitude_nodes=ds.sizes[lon_name],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds[lat_name]),
        longitude_offset=xarray_utils.infer_longitude_offset(ds[lon_name]),
    )


def collect_chunk_bounds(start_dt: datetime, end_dt: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    bounds: list[tuple[datetime, datetime]] = []
    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days) - timedelta(hours=1), end_dt)
        bounds.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(hours=1)
    return bounds


def main() -> None:
    args = parse_args()
    water_year = args.water_year
    start_dt = datetime(water_year - 1, 9, 1, 0, 0, 0)
    end_dt = datetime(water_year, 3, 31, 18, 0, 0)

    prepared_dir = NEURALGCM_ROOT / "assets" / "prepared_inputs" / args.actual_label
    target_forcing_path = prepared_dir / "forcing_regridded_6h.nc"
    if not target_forcing_path.exists():
        raise FileNotFoundError(f"Target NeuralGCM forcing file not found: {target_forcing_path}")

    if args.output_root is None:
        output_dir = NEURALGCM_ROOT / "assets" / "regridded_aux" / "era5land_t2m_neuralgcm_grid" / f"wy{water_year}"
    else:
        output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_6h_path = output_dir / "era5land_t2m_6h_mean_on_neuralgcm_grid.nc"
    out_daily_path = output_dir / "era5land_t2m_daily_mean_on_neuralgcm_grid.nc"
    summary_path = output_dir / "regrid_summary.json"
    chunks_dir = output_dir / "chunks_6h"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and out_6h_path.exists() and out_daily_path.exists() and summary_path.exists():
        print(json.dumps({"status": "reused", "output_dir": str(output_dir)}, indent=2))
        return

    log(f"Starting ERA5-Land -> NeuralGCM T2M regridding for WY{water_year}")
    log(f"Prepared NeuralGCM forcing grid source: {target_forcing_path}")
    log(f"Output directory: {output_dir}")

    with xr.open_dataset(target_forcing_path) as target_ds:
        target_times = np.asarray(target_ds.time.values)
        target_lat = target_ds.latitude
        target_lon = target_ds.longitude
        target_grid = make_grid(target_ds, "latitude", "longitude")
    log(
        "Loaded target grid with "
        f"{target_lat.size} latitudes, {target_lon.size} longitudes, {target_times.size} target times"
    )

    year_paths = {
        water_year - 1: ERA5_LAND_T2M_ROOT / f"ERA5_{water_year - 1}_2m_temperature.nc",
        water_year: ERA5_LAND_T2M_ROOT / f"ERA5_{water_year}_2m_temperature.nc",
    }
    for year, path in year_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing ERA5-Land source file for {year}: {path}")

    source_grid: spherical_harmonic.Grid | None = None
    regridder: horizontal_interpolation.ConservativeRegridder | None = None
    six_hour_chunks: list[xr.DataArray] = []
    chunk_summaries: list[dict[str, object]] = []

    year_windows = [
        (water_year - 1, start_dt, datetime(water_year - 1, 12, 31, 23, 0, 0)),
        (water_year, datetime(water_year, 1, 1, 0, 0, 0), end_dt),
    ]

    total_chunks_expected = sum(len(collect_chunk_bounds(year_start, year_end, args.chunk_days)) for _, year_start, year_end in year_windows)
    chunk_counter = 0

    for year, year_start, year_end in year_windows:
        source_path = year_paths[year]
        for chunk_start, chunk_end in collect_chunk_bounds(year_start, year_end, args.chunk_days):
            chunk_counter += 1
            chunk_file = chunks_dir / f"chunk_{chunk_counter:03d}_{chunk_start:%Y%m%d%H}_{chunk_end:%Y%m%d%H}.nc"
            chunk_meta_file = chunks_dir / f"chunk_{chunk_counter:03d}_{chunk_start:%Y%m%d%H}_{chunk_end:%Y%m%d%H}.json"
            if chunk_file.exists() and not args.force:
                log(
                    f"Reusing checkpoint chunk {chunk_counter}/{total_chunks_expected}: "
                    f"{chunk_file.name}"
                )
                with xr.open_dataset(chunk_file) as chunk_ds:
                    regridded = chunk_ds["t2m"].load()
                six_hour_chunks.append(regridded)
                meta = {
                    "source_path": str(source_path),
                    "chunk_index": chunk_counter,
                    "chunk_start": chunk_start.isoformat(),
                    "chunk_end": chunk_end.isoformat(),
                    "n_source_times": None,
                    "n_output_times": int(regridded.sizes["time"]),
                    "checkpoint_file": str(chunk_file),
                    "reused_checkpoint": True,
                }
                chunk_summaries.append(meta)
                save_json(chunk_meta_file, meta)
                continue

            log(
                f"Processing chunk {chunk_counter}/{total_chunks_expected} "
                f"from {chunk_start.isoformat()} to {chunk_end.isoformat()} "
                f"using source {source_path.name}"
            )
            with xr.open_dataset(source_path) as ds:
                log(f"Opened source dataset for chunk {chunk_counter}: {source_path}")
                source_da = ds["t2m"].sel(
                    time=slice(datetime_to_dt64(chunk_start), datetime_to_dt64(chunk_end))
                )
                if source_da.sizes.get("time", 0) == 0:
                    log(f"Skipping chunk {chunk_counter}: no source time steps found")
                    continue
                if source_grid is None:
                    log("Building source and target regridder objects")
                    source_grid = make_grid(source_da.to_dataset(name="t2m"), "latitude", "longitude")
                    regridder = horizontal_interpolation.ConservativeRegridder(source_grid, target_grid, skipna=True)
                chunk_target_times = target_times[
                    (target_times >= datetime_to_dt64(chunk_start)) & (target_times <= datetime_to_dt64(chunk_end))
                ]
                if chunk_target_times.size == 0:
                    log(f"Skipping chunk {chunk_counter}: no target times overlap this window")
                    continue
                log(
                    f"Chunk {chunk_counter}: resampling {int(source_da.sizes['time'])} hourly source steps "
                    f"to {int(chunk_target_times.size)} target 6-hour steps"
                )
                six_hour = source_da.resample(time="6h").mean().sel(time=chunk_target_times)
                six_hour = six_hour.load()
                log(f"Chunk {chunk_counter}: regridding onto NeuralGCM coarse grid")
                regridded = xarray_utils.regrid(six_hour.to_dataset(name="t2m"), regridder)["t2m"]
                log(f"Chunk {chunk_counter}: filling any remaining NaNs with nearest neighbors")
                regridded = xarray_utils.fill_nan_with_nearest(regridded.to_dataset(name="t2m"))["t2m"]
                regridded = regridded.assign_coords(latitude=target_lat, longitude=target_lon)
                regridded.attrs.update(
                    {
                        "long_name": "ERA5-Land 2m temperature regridded to NeuralGCM coarse grid",
                        "units": source_da.attrs.get("units", "K"),
                        "source_dataset": str(source_path),
                        "regridding_method": "dinosaur.horizontal_interpolation.ConservativeRegridder",
                        "target_grid_source": str(target_forcing_path),
                    }
                )
                tmp_chunk_file = chunk_file.with_suffix(".nc.tmp")
                regridded.to_dataset(name="t2m").to_netcdf(tmp_chunk_file)
                shutil.move(tmp_chunk_file, chunk_file)
                six_hour_chunks.append(regridded)
                meta = {
                    "source_path": str(source_path),
                    "chunk_index": chunk_counter,
                    "chunk_start": chunk_start.isoformat(),
                    "chunk_end": chunk_end.isoformat(),
                    "n_source_times": int(source_da.sizes["time"]),
                    "n_output_times": int(regridded.sizes["time"]),
                    "checkpoint_file": str(chunk_file),
                    "checkpoint_size_bytes": int(chunk_file.stat().st_size),
                    "nan_count": int(regridded.isnull().sum().item()),
                    "min_k": float(regridded.min().item()),
                    "max_k": float(regridded.max().item()),
                    "reused_checkpoint": False,
                }
                chunk_summaries.append(meta)
                save_json(chunk_meta_file, meta)
                log(
                    f"Chunk {chunk_counter} checkpoint written: {chunk_file.name} "
                    f"({meta['checkpoint_size_bytes']} bytes)"
                )

    if not six_hour_chunks:
        raise RuntimeError("No ERA5-Land chunks were processed.")

    log(f"Combining {len(six_hour_chunks)} chunk checkpoints into season-level outputs")
    combined_6h = xr.concat(six_hour_chunks, dim="time").sortby("time")
    _, unique_idx = np.unique(combined_6h.time.values, return_index=True)
    combined_6h = combined_6h.isel(time=np.sort(unique_idx))
    combined_6h = combined_6h.sel(time=target_times)

    log("Computing daily means from 6-hour regridded series")
    daily = combined_6h.resample(time="1D").mean()
    daily.attrs = dict(combined_6h.attrs)
    daily.attrs["time_aggregation"] = "daily_mean_from_6h_means"

    encoding = {"t2m": {"zlib": True, "complevel": 2}}
    tmp_6h_path = out_6h_path.with_suffix(".nc.tmp")
    tmp_daily_path = out_daily_path.with_suffix(".nc.tmp")
    log(f"Writing season-level 6-hour output: {out_6h_path}")
    combined_6h.to_dataset(name="t2m").to_netcdf(tmp_6h_path, encoding=encoding)
    shutil.move(tmp_6h_path, out_6h_path)
    log(f"Writing season-level daily output: {out_daily_path}")
    daily.to_dataset(name="t2m").to_netcdf(tmp_daily_path, encoding=encoding)
    shutil.move(tmp_daily_path, out_daily_path)

    payload = {
        "status": "completed",
        "water_year": water_year,
        "actual_label": args.actual_label,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "target_forcing_path": str(target_forcing_path),
        "output_dir": str(output_dir),
        "output_6h_path": str(out_6h_path),
        "output_daily_path": str(out_daily_path),
        "source_year_paths": {str(year): str(path) for year, path in year_paths.items()},
        "chunk_days": args.chunk_days,
        "chunk_count": len(chunk_summaries),
        "chunks_dir": str(chunks_dir),
        "chunk_summaries": chunk_summaries,
        "six_hour_summary": {
            "time_count": int(combined_6h.sizes["time"]),
            "time_start": str(np.asarray(combined_6h.time.values[0]).item()),
            "time_end": str(np.asarray(combined_6h.time.values[-1]).item()),
            "latitude_count": int(combined_6h.sizes["latitude"]),
            "longitude_count": int(combined_6h.sizes["longitude"]),
            "min_k": float(combined_6h.min().item()),
            "max_k": float(combined_6h.max().item()),
            "mean_k": float(combined_6h.mean().item()),
            "nan_count": int(combined_6h.isnull().sum().item()),
            "size_bytes": int(out_6h_path.stat().st_size),
        },
        "daily_summary": {
            "time_count": int(daily.sizes["time"]),
            "time_start": str(np.asarray(daily.time.values[0]).item()),
            "time_end": str(np.asarray(daily.time.values[-1]).item()),
            "min_k": float(daily.min().item()),
            "max_k": float(daily.max().item()),
            "mean_k": float(daily.mean().item()),
            "nan_count": int(daily.isnull().sum().item()),
            "size_bytes": int(out_daily_path.stat().st_size),
        },
        "grid_strategy": {
            "common_grid": "NeuralGCM native coarse grid",
            "regridding_direction": "ERA5-Land T2M to NeuralGCM grid",
        },
    }
    save_json(summary_path, payload)
    log(f"Summary written: {summary_path}")
    print(json.dumps(payload, indent=2, default=to_serializable))


if __name__ == "__main__":
    main()
