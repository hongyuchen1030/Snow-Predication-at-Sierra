#!/usr/bin/env python3
"""Run one NeuralGCM WY2021 actual historical stochastic trajectory."""

from __future__ import annotations

import json
import os
import pickle
import socket
import time
from pathlib import Path

import gcsfs
import jax
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
NEURALGCM_OUTPUTS = NEURALGCM_ROOT / "outputs"
NEURALGCM_LOGS = NEURALGCM_ROOT / "logs"
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"

CONFIG_PATH = REPORT_DIR / "configs" / "wy2021_actual_m001_config.json"
RUNTIME_SUMMARY_TXT = REPORT_DIR / "wy2021_actual_m001_runtime_summary.txt"
RUNTIME_SUMMARY_JSON = REPORT_DIR / "wy2021_actual_m001_runtime_summary.json"

OUTPUT_DIR = NEURALGCM_OUTPUTS / "wy2021_actual_m001"
PREP_DIR = NEURALGCM_ASSETS / "prepared_inputs" / "wy2021_actual_m001"
CHECKPOINT_DIR = NEURALGCM_ASSETS / "checkpoints"
CHECKPOINT_NAME = "v1_precip/stochastic_precip_2_8_deg.pkl"
CHECKPOINT_GCS = f"gs://neuralgcm/models/{CHECKPOINT_NAME}"
CHECKPOINT_LOCAL = CHECKPOINT_DIR / "stochastic_precip_2_8_deg.pkl"
ARCO_ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

START = np.datetime64("2020-09-01T00:00:00")
END = np.datetime64("2021-03-31T18:00:00")
OUTER_STEP_HOURS = 6
RNG_SEED = 20210901


def ensure_dirs() -> None:
    for path in [REPORT_DIR, REPORT_DIR / "configs", CHECKPOINT_DIR, PREP_DIR, OUTPUT_DIR, NEURALGCM_LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=to_serializable) + "\n", encoding="utf-8")


def download_checkpoint_if_needed() -> Path:
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


def stage_inputs(model: neuralgcm.PressureLevelModel) -> tuple[Path, Path, dict]:
    init_path = PREP_DIR / "initial_condition_regridded.nc"
    forcing_path = PREP_DIR / "forcing_regridded_6h.nc"
    if init_path.exists() and forcing_path.exists():
        with xr.open_dataset(forcing_path) as forcing_ds:
            forcing_times = forcing_ds.time.values
        return init_path, forcing_path, {"reused_prepared_inputs": True, "forcing_time_count": int(len(forcing_times))}

    full_era5 = open_arco_era5()
    try:
        source_grid = make_source_grid(full_era5)
        regridder = horizontal_interpolation.ConservativeRegridder(
            source_grid, model.data_coords.horizontal, skipna=True
        )

        initial_ds = full_era5[model.input_variables].sel(time=START).compute()
        initial_regridded = xarray_utils.regrid(initial_ds, regridder)
        initial_regridded = xarray_utils.fill_nan_with_nearest(initial_regridded)
        initial_regridded.to_netcdf(init_path)

        forcing_raw = (
            full_era5[model.forcing_variables]
            .pipe(
                xarray_utils.selective_temporal_shift,
                variables=model.forcing_variables,
                time_shift=f"{OUTER_STEP_HOURS} hours",
            )
            .sel(time=slice(START, END))
            .isel(time=slice(None, None, OUTER_STEP_HOURS))
            .compute()
        )
        forcing_regridded = xarray_utils.regrid(forcing_raw, regridder)
        forcing_regridded = xarray_utils.fill_nan_with_nearest(forcing_regridded)
        forcing_regridded.to_netcdf(forcing_path)

        return init_path, forcing_path, {
            "reused_prepared_inputs": False,
            "forcing_time_count": int(forcing_regridded.sizes["time"]),
            "forcing_start": str(np.asarray(forcing_regridded.time.values[0]).item()),
            "forcing_end": str(np.asarray(forcing_regridded.time.values[-1]).item()),
        }
    finally:
        full_era5.close()


def main() -> None:
    ensure_dirs()

    step_start = time.time()
    print("Starting NeuralGCM WY2021 actual M=1 run...", flush=True)
    checkpoint_path = download_checkpoint_if_needed()
    print(f"Checkpoint ready: {checkpoint_path}", flush=True)
    model = load_model(checkpoint_path)
    print(
        f"Loaded model {CHECKPOINT_NAME} on grid {model.data_coords.horizontal.longitude_nodes}x{model.data_coords.horizontal.latitude_nodes}",
        flush=True,
    )

    config = {
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_gcs": CHECKPOINT_GCS,
        "checkpoint_local": str(checkpoint_path),
        "remote_input_source": ARCO_ERA5_PATH,
        "start": str(START),
        "end": str(END),
        "outer_step_hours": OUTER_STEP_HOURS,
        "rng_seed": RNG_SEED,
        "output_dir": str(OUTPUT_DIR),
        "prepared_input_dir": str(PREP_DIR),
        "input_variables": list(model.input_variables),
        "forcing_variables": list(model.forcing_variables),
        "model_timestep_seconds": int(model.timestep / np.timedelta64(1, "s")),
        "model_grid": {
            "longitude_nodes": int(model.data_coords.horizontal.longitude_nodes),
            "latitude_nodes": int(model.data_coords.horizontal.latitude_nodes),
            "latitude_spacing": str(model.data_coords.horizontal.latitude_spacing),
        },
        "forcing_domain_requirement": "global",
        "scientific_sst_domain": {
            "lat_min": -10.0,
            "lat_max": 60.0,
            "lon_min_360": 120.0,
            "lon_max_360": 280.0,
        },
    }
    save_json(CONFIG_PATH, config)
    print(f"Wrote config: {CONFIG_PATH}", flush=True)

    init_path, forcing_path, prep_info = stage_inputs(model)
    print(
        f"Prepared inputs ready: init={init_path} forcing={forcing_path} reused={prep_info['reused_prepared_inputs']}",
        flush=True,
    )

    with xr.open_dataset(init_path) as initial_ds, xr.open_dataset(forcing_path) as forcing_ds:
        inputs = model.inputs_from_xarray(initial_ds)
        input_forcings = model.forcings_from_xarray(forcing_ds.isel(time=0))
        temporal_forcings = model.forcings_from_xarray(forcing_ds)
        output_times = np.asarray(forcing_ds.time.values)

    steps = int(len(output_times))
    rng_key = jax.random.key(RNG_SEED)
    print(f"Encode start. steps={steps} outer_step_hours={OUTER_STEP_HOURS}", flush=True)

    encode_start = time.time()
    initial_state = model.encode(inputs, input_forcings, rng_key=rng_key)
    encode_end = time.time()
    print(f"Encode done in {encode_end - encode_start:.3f} s", flush=True)

    unroll_start = time.time()
    print("Unroll start...", flush=True)
    final_state, predictions = model.unroll(
        initial_state,
        temporal_forcings,
        steps=steps,
        timedelta=np.timedelta64(OUTER_STEP_HOURS, "h"),
        start_with_input=True,
    )
    unroll_end = time.time()
    print(f"Unroll done in {unroll_end - unroll_start:.3f} s", flush=True)

    predictions_ds = model.data_to_xarray(predictions, times=output_times)
    diagnostics = xr.Dataset()
    for name in ["precipitation_cumulative_mean", "evaporation", "sim_time"]:
        if name in predictions_ds:
            diagnostics[name] = predictions_ds[name]
    if "temperature" in predictions_ds:
        diagnostics["temperature_850hPa"] = predictions_ds["temperature"].sel(level=850, method="nearest")
        diagnostics["temperature_850hPa"].attrs["selected_level_hpa"] = 850
    if "u_component_of_wind" in predictions_ds:
        diagnostics["u_component_of_wind_850hPa"] = predictions_ds["u_component_of_wind"].sel(level=850, method="nearest")
        diagnostics["u_component_of_wind_850hPa"].attrs["selected_level_hpa"] = 850
    if "v_component_of_wind" in predictions_ds:
        diagnostics["v_component_of_wind_850hPa"] = predictions_ds["v_component_of_wind"].sel(level=850, method="nearest")
        diagnostics["v_component_of_wind_850hPa"].attrs["selected_level_hpa"] = 850
    if "specific_humidity" in predictions_ds:
        diagnostics["specific_humidity_850hPa"] = predictions_ds["specific_humidity"].sel(level=850, method="nearest")
        diagnostics["specific_humidity_850hPa"].attrs["selected_level_hpa"] = 850

    for coord_name in predictions_ds.coords:
        diagnostics = diagnostics.assign_coords({coord_name: predictions_ds[coord_name]})

    output_nc = OUTPUT_DIR / "predictions_6h.nc"
    print(f"Writing output NetCDF: {output_nc}", flush=True)
    diagnostics.to_netcdf(output_nc)

    metadata = {
        "output_variables": list(diagnostics.data_vars),
        "output_sizes": {k: int(v) for k, v in diagnostics.sizes.items()},
        "output_time_start": str(np.asarray(diagnostics.time.values[0]).item()),
        "output_time_end": str(np.asarray(diagnostics.time.values[-1]).item()),
        "output_file": str(output_nc),
        "output_file_size_bytes": int(output_nc.stat().st_size),
        "prepared_input_initial_condition": str(init_path),
        "prepared_input_forcing": str(forcing_path),
        "prepared_input_forcing_size_bytes": int(forcing_path.stat().st_size),
        "prepared_input_initial_condition_size_bytes": int(init_path.stat().st_size),
        "full_model_variables_before_reduction": list(predictions_ds.data_vars),
    }
    save_json(OUTPUT_DIR / "model_output_metadata.json", metadata)

    total_end = time.time()
    runtime_summary = {
        "command": "python scripts/neuralgcm_feasibility/run_neuralgcm_wy2021_actual_m001.py",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "node": socket.gethostname(),
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_local_path": str(checkpoint_path),
        "rng_seed": RNG_SEED,
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "start_time_unix": step_start,
        "end_time_unix": total_end,
        "model_timestep_seconds": int(model.timestep / np.timedelta64(1, "s")),
        "outer_step_hours": OUTER_STEP_HOURS,
        "number_of_output_steps": steps,
        "encode_runtime_seconds": encode_end - encode_start,
        "unroll_runtime_seconds": unroll_end - unroll_start,
        "total_runtime_seconds": total_end - step_start,
        "steps_per_second": steps / max(unroll_end - unroll_start, 1e-6),
        "output_inventory": [
            {
                "path": str(output_nc),
                "size_bytes": int(output_nc.stat().st_size),
            },
            {
                "path": str(OUTPUT_DIR / "model_output_metadata.json"),
                "size_bytes": int((OUTPUT_DIR / "model_output_metadata.json").stat().st_size),
            },
        ],
        "total_output_size_bytes": int(
            sum(item["size_bytes"] for item in metadata.get("output_inventory", []))
        ),
        "prepared_input_info": prep_info,
        "output_metadata": metadata,
        "config_path": str(CONFIG_PATH),
    }
    # Include the main output sizes explicitly because metadata.output_inventory is not nested.
    runtime_summary["total_output_size_bytes"] = int(
        output_nc.stat().st_size + (OUTPUT_DIR / "model_output_metadata.json").stat().st_size
    )
    save_json(RUNTIME_SUMMARY_JSON, runtime_summary)

    txt_lines = [
        "NeuralGCM WY2021 actual M=1 runtime summary",
        f"slurm_job_id: {runtime_summary['slurm_job_id']}",
        f"node: {runtime_summary['node']}",
        f"checkpoint: {CHECKPOINT_NAME}",
        f"rng_seed: {RNG_SEED}",
        f"start: {START}",
        f"end: {END}",
        f"outer_step_hours: {OUTER_STEP_HOURS}",
        f"number_of_output_steps: {steps}",
        f"encode_runtime_seconds: {runtime_summary['encode_runtime_seconds']:.3f}",
        f"unroll_runtime_seconds: {runtime_summary['unroll_runtime_seconds']:.3f}",
        f"total_runtime_seconds: {runtime_summary['total_runtime_seconds']:.3f}",
        f"steps_per_second: {runtime_summary['steps_per_second']:.6f}",
        f"output_file: {output_nc}",
        f"output_file_size_bytes: {output_nc.stat().st_size}",
        f"prepared_initial_condition: {init_path}",
        f"prepared_forcing: {forcing_path}",
        f"config_path: {CONFIG_PATH}",
    ]
    RUNTIME_SUMMARY_TXT.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"Runtime summaries written: {RUNTIME_SUMMARY_JSON} and {RUNTIME_SUMMARY_TXT}", flush=True)


if __name__ == "__main__":
    main()
