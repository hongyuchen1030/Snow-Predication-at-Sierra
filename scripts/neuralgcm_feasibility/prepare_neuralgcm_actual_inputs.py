#!/usr/bin/env python3
"""Prepare NeuralGCM actual historical inputs for a requested WY season."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import gcsfs
import neuralgcm
import numpy as np
import xarray as xr
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
SCRATCH_ROOT = Path(os.environ.get("PSCRATCH", "/pscratch/sd/h/hyvchen"))
NEURALGCM_ROOT = SCRATCH_ROOT / "Snow-Predication-at-Sierra_neuralgcm"
NEURALGCM_ASSETS = NEURALGCM_ROOT / "assets"
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
CHECKPOINT_DIR = NEURALGCM_ASSETS / "checkpoints"
CHECKPOINT_NAME = "v1_precip/stochastic_precip_2_8_deg.pkl"
CHECKPOINT_GCS = f"gs://neuralgcm/models/{CHECKPOINT_NAME}"
CHECKPOINT_LOCAL = CHECKPOINT_DIR / "stochastic_precip_2_8_deg.pkl"
ARCO_ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
OUTER_STEP_HOURS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Prepared-input directory label, e.g. wy1998_actual_m001")
    parser.add_argument("--start", required=True, help="Start timestamp, e.g. 1997-09-01T00:00:00")
    parser.add_argument("--end", required=True, help="End timestamp, e.g. 1998-03-31T18:00:00")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def download_checkpoint_if_needed() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_LOCAL.exists():
        return CHECKPOINT_LOCAL
    fs = gcsfs.GCSFileSystem(token="anon")
    with fs.open(CHECKPOINT_GCS, "rb") as src, CHECKPOINT_LOCAL.open("wb") as dst:
        dst.write(src.read())
    return CHECKPOINT_LOCAL


def load_model(checkpoint_path: Path) -> neuralgcm.PressureLevelModel:
    with checkpoint_path.open("rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def open_arco_era5() -> xr.Dataset:
    return xr.open_zarr(ARCO_ERA5_PATH, chunks=None, storage_options={"token": "anon"})


def make_source_grid(ds: xr.Dataset) -> spherical_harmonic.Grid:
    return spherical_harmonic.Grid(
        latitude_nodes=ds.sizes["latitude"],
        longitude_nodes=ds.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude),
    )


def summarize_dataset(path: Path, variables: list[str]) -> dict:
    with xr.open_dataset(path) as ds:
        time_count = 0
        summary = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "variables": list(ds.data_vars),
            "sizes": {k: int(v) for k, v in ds.sizes.items()},
        }
        if "time" in ds.coords:
            time_values = np.asarray(ds.time.values)
            if time_values.shape == ():
                time_count = 1
                summary["time_start"] = str(time_values.item())
                summary["time_end"] = str(time_values.item())
            else:
                time_count = int(len(time_values))
                if time_count:
                    summary["time_start"] = str(np.asarray(time_values[0]).item())
                    summary["time_end"] = str(np.asarray(time_values[-1]).item())
        summary["time_count"] = time_count
        if "sea_surface_temperature" in ds:
            arr = ds["sea_surface_temperature"]
            summary["sea_surface_temperature_units"] = arr.attrs.get("units")
            summary["sea_surface_temperature_nan_count"] = int(arr.isnull().sum().item())
        if "sea_ice_cover" in ds:
            arr = ds["sea_ice_cover"]
            summary["sea_ice_cover_units"] = arr.attrs.get("units")
            summary["sea_ice_cover_nan_count"] = int(arr.isnull().sum().item())
        missing = [name for name in variables if name not in ds.data_vars]
        summary["missing_expected_variables"] = missing
    return summary


def main() -> None:
    args = parse_args()
    start = np.datetime64(args.start)
    end = np.datetime64(args.end)
    prep_dir = NEURALGCM_ASSETS / "prepared_inputs" / args.label
    prep_dir.mkdir(parents=True, exist_ok=True)
    init_path = prep_dir / "initial_condition_regridded.nc"
    forcing_path = prep_dir / "forcing_regridded_6h.nc"

    report_json = REPORT_DIR / ("%s_prep_summary.json" % args.label)
    report_txt = REPORT_DIR / ("%s_prep_summary.txt" % args.label)

    checkpoint_path = download_checkpoint_if_needed()
    model = load_model(checkpoint_path)

    reused = init_path.exists() and forcing_path.exists() and not args.force
    prep_info: dict[str, object] = {}

    if not reused:
        full_era5 = open_arco_era5()
        try:
            source_grid = make_source_grid(full_era5)
            regridder = horizontal_interpolation.ConservativeRegridder(
                source_grid, model.data_coords.horizontal, skipna=True
            )

            initial_ds = full_era5[model.input_variables].sel(time=start).compute()
            initial_regridded = xarray_utils.regrid(initial_ds, regridder)
            initial_regridded = xarray_utils.fill_nan_with_nearest(initial_regridded)
            initial_regridded.to_netcdf(init_path)

            forcing_raw = (
                full_era5[model.forcing_variables]
                .pipe(
                    xarray_utils.selective_temporal_shift,
                    variables=model.forcing_variables,
                    time_shift="%d hours" % OUTER_STEP_HOURS,
                )
                .sel(time=slice(start, end))
                .isel(time=slice(None, None, OUTER_STEP_HOURS))
                .compute()
            )
            forcing_regridded = xarray_utils.regrid(forcing_raw, regridder)
            forcing_regridded = xarray_utils.fill_nan_with_nearest(forcing_regridded)
            forcing_regridded.to_netcdf(forcing_path)

            prep_info = {
                "reused_prepared_inputs": False,
                "forcing_raw_time_count": int(forcing_raw.sizes["time"]),
                "forcing_start": str(np.asarray(forcing_regridded.time.values[0]).item()),
                "forcing_end": str(np.asarray(forcing_regridded.time.values[-1]).item()),
            }
        finally:
            full_era5.close()
    else:
        prep_info = {"reused_prepared_inputs": True}

    initial_summary = summarize_dataset(init_path, list(model.input_variables))
    forcing_summary = summarize_dataset(forcing_path, list(model.forcing_variables))

    ready = (
        initial_summary["missing_expected_variables"] == []
        and forcing_summary["missing_expected_variables"] == []
        and forcing_summary.get("time_count", 0) > 0
        and forcing_summary.get("sea_surface_temperature_nan_count", 1) == 0
        and forcing_summary.get("sea_ice_cover_nan_count", 1) == 0
    )

    payload = {
        "label": args.label,
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_local_path": str(checkpoint_path),
        "remote_input_source": ARCO_ERA5_PATH,
        "start": str(start),
        "end": str(end),
        "outer_step_hours": OUTER_STEP_HOURS,
        "forcing_domain_requirement": "global",
        "scientific_sst_domain": {
            "lat_min": -10.0,
            "lat_max": 60.0,
            "lon_min_360": 120.0,
            "lon_max_360": 280.0,
        },
        "model_input_variables": list(model.input_variables),
        "model_forcing_variables": list(model.forcing_variables),
        "prepared_input_dir": str(prep_dir),
        "initial_condition_summary": initial_summary,
        "forcing_summary": forcing_summary,
        "prep_info": prep_info,
        "ready_for_next_actual_reversed_experiment": bool(ready),
        "blocker": None if ready else "Prepared files missing expected variables, time coverage, or finite SST/sea-ice forcing.",
        "sst_source_used": "ARCO ERA5 sea_surface_temperature",
        "sea_ice_source_used": "ARCO ERA5 sea_ice_cover",
        "cobe_sst_used_for_actual_prep": False,
        "notes": [
            "This preparation reuses the same global ARCO ERA5 + NeuralGCM regridding logic that worked for WY2021.",
            "COBE SST was not required for actual historical preparation because ARCO ERA5 already exposes the exact forcing variable names expected by the checkpoint.",
        ],
    }
    save_json(report_json, payload)

    lines = [
        "label: %s" % args.label,
        "ready_for_next_actual_reversed_experiment: %s" % payload["ready_for_next_actual_reversed_experiment"],
        "blocker: %s" % payload["blocker"],
        "checkpoint: %s" % checkpoint_path,
        "remote_input_source: %s" % ARCO_ERA5_PATH,
        "prepared_input_dir: %s" % prep_dir,
        "initial_condition_file: %s" % init_path,
        "forcing_file: %s" % forcing_path,
        "forcing_time_count: %s" % forcing_summary.get("time_count"),
        "forcing_time_start: %s" % forcing_summary.get("time_start"),
        "forcing_time_end: %s" % forcing_summary.get("time_end"),
        "sst_source_used: %s" % payload["sst_source_used"],
        "sea_ice_source_used: %s" % payload["sea_ice_source_used"],
        "cobe_sst_used_for_actual_prep: %s" % payload["cobe_sst_used_for_actual_prep"],
    ]
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=to_serializable))


if __name__ == "__main__":
    main()
