#!/usr/bin/env python3
"""Run a same-IC same-forcing NeuralGCM stochastic ensemble for WY2021."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import socket
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import neuralgcm
import numpy as np
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
SCRATCH_ROOT = Path(os.environ.get("PSCRATCH", "/pscratch/sd/h/hyvchen"))
NEURALGCM_ROOT = SCRATCH_ROOT / "Snow-Predication-at-Sierra_neuralgcm"
NEURALGCM_ASSETS = NEURALGCM_ROOT / "assets"
NEURALGCM_OUTPUTS = NEURALGCM_ROOT / "outputs"
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"

CHECKPOINT_NAME = "v1_precip/stochastic_precip_2_8_deg.pkl"
CHECKPOINT_LOCAL = NEURALGCM_ASSETS / "checkpoints" / "stochastic_precip_2_8_deg.pkl"
PREP_DIR = NEURALGCM_ASSETS / "prepared_inputs" / "wy2021_actual_m001"
IC_PATH = PREP_DIR / "initial_condition_regridded.nc"
FORCING_PATH = PREP_DIR / "forcing_regridded_6h.nc"
M003_OUTPUT_ROOT = NEURALGCM_OUTPUTS / "wy2021_actual_m003"

RUNTIME_SUMMARY_JSON = REPORT_DIR / "neuralgcm_wy2021_actual_m003_runtime_summary.json"
RUNTIME_SUMMARY_TXT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_runtime_summary.txt"

MEMBERS = [
    ("member_000", 0),
    ("member_001", 1),
    ("member_002", 2),
]
EXPECTED_VARIABLES = [
    "precipitation_cumulative_mean",
    "evaporation",
    "sim_time",
    "temperature_850hPa",
    "u_component_of_wind_850hPa",
    "v_component_of_wind_850hPa",
    "specific_humidity_850hPa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rerun members even if valid outputs already exist")
    return parser.parse_args()


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    M003_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


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


def load_model() -> neuralgcm.PressureLevelModel:
    with CHECKPOINT_LOCAL.open("rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def output_paths(member_name: str) -> tuple[Path, Path]:
    member_dir = M003_OUTPUT_ROOT / member_name
    return member_dir / "predictions_6h.nc", member_dir / "model_output_metadata.json"


def validate_member_output(path: Path) -> tuple[bool, dict]:
    if not path.exists():
        return False, {"exists": False}
    try:
        with xr.open_dataset(path) as ds:
            missing = [name for name in EXPECTED_VARIABLES if name not in ds.data_vars]
            if missing:
                return False, {"exists": True, "missing_variables": missing}
            finite_report = {}
            for name in EXPECTED_VARIABLES:
                values = np.asarray(ds[name].values)
                finite_report[name] = {
                    "nan_count": int(np.isnan(values).sum()),
                    "inf_count": int(np.isinf(values).sum()),
                }
                if finite_report[name]["nan_count"] > 0 or finite_report[name]["inf_count"] > 0:
                    return False, {"exists": True, "finite_report": finite_report}
            return True, {
                "exists": True,
                "sizes": {k: int(v) for k, v in ds.sizes.items()},
                "variables": list(ds.data_vars),
                "finite_report": finite_report,
            }
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return False, {"exists": True, "error": f"{type(exc).__name__}: {exc}"}


def build_diagnostics(predictions_ds: xr.Dataset) -> xr.Dataset:
    coords = {name: np.asarray(coord.values) for name, coord in predictions_ds.coords.items()}
    data_vars = {}
    if "precipitation_cumulative_mean" in predictions_ds:
        da = predictions_ds["precipitation_cumulative_mean"]
        data_vars["precipitation_cumulative_mean"] = (da.dims, np.asarray(da.values), dict(da.attrs))
    if "evaporation" in predictions_ds:
        da = predictions_ds["evaporation"]
        data_vars["evaporation"] = (da.dims, np.asarray(da.values), dict(da.attrs))
    if "sim_time" in predictions_ds:
        da = predictions_ds["sim_time"]
        data_vars["sim_time"] = (da.dims, np.asarray(da.values), dict(da.attrs))
    if "temperature" in predictions_ds:
        da = predictions_ds["temperature"].sel(level=850, method="nearest")
        attrs = dict(da.attrs)
        attrs["selected_level_hpa"] = 850
        data_vars["temperature_850hPa"] = (da.dims, np.asarray(da.values), attrs)
    if "u_component_of_wind" in predictions_ds:
        da = predictions_ds["u_component_of_wind"].sel(level=850, method="nearest")
        attrs = dict(da.attrs)
        attrs["selected_level_hpa"] = 850
        data_vars["u_component_of_wind_850hPa"] = (da.dims, np.asarray(da.values), attrs)
    if "v_component_of_wind" in predictions_ds:
        da = predictions_ds["v_component_of_wind"].sel(level=850, method="nearest")
        attrs = dict(da.attrs)
        attrs["selected_level_hpa"] = 850
        data_vars["v_component_of_wind_850hPa"] = (da.dims, np.asarray(da.values), attrs)
    if "specific_humidity" in predictions_ds:
        da = predictions_ds["specific_humidity"].sel(level=850, method="nearest")
        attrs = dict(da.attrs)
        attrs["selected_level_hpa"] = 850
        data_vars["specific_humidity_850hPa"] = (da.dims, np.asarray(da.values), attrs)
    diagnostics = xr.Dataset(coords=coords)
    for name, (dims, values, attrs) in data_vars.items():
        diagnostics[name] = xr.DataArray(values, dims=dims, attrs=attrs)
    return diagnostics


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not IC_PATH.exists() or not FORCING_PATH.exists() or not CHECKPOINT_LOCAL.exists():
        raise FileNotFoundError("Missing prepared IC/forcing or checkpoint required for M=3 run.")

    model = load_model()
    print(
        f"Starting NeuralGCM WY2021 actual M=3 run with checkpoint {CHECKPOINT_NAME}",
        flush=True,
    )
    overall_start = time.time()
    summary = {
        "command": "python scripts/neuralgcm_feasibility/run_neuralgcm_wy2021_actual_m003.py",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "node": socket.gethostname(),
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_local_path": str(CHECKPOINT_LOCAL),
        "initial_condition_path": str(IC_PATH),
        "forcing_path": str(FORCING_PATH),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "members": [],
    }

    with xr.open_dataset(IC_PATH) as initial_ds, xr.open_dataset(FORCING_PATH) as forcing_ds:
        inputs = model.inputs_from_xarray(initial_ds)
        input_forcings = model.forcings_from_xarray(forcing_ds.isel(time=0))
        temporal_forcings = model.forcings_from_xarray(forcing_ds)
        output_times = np.asarray(forcing_ds.time.values)
        forcing_sizes = {k: int(v) for k, v in forcing_ds.sizes.items()}
    steps = int(len(output_times))

    for member_name, rng_seed in MEMBERS:
        output_nc, output_meta = output_paths(member_name)
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        valid, validation = validate_member_output(output_nc)
        member_start = time.time()
        member_record = {
            "member_name": member_name,
            "rng_key": rng_seed,
            "output_path": str(output_nc),
            "metadata_path": str(output_meta),
            "used_existing_output": False,
            "skipped_existing_valid_output": False,
            "validation_before_run": validation,
        }

        if valid and not args.force:
            print(f"{member_name}: reusing existing valid output {output_nc}", flush=True)
            member_record["used_existing_output"] = True
            member_record["skipped_existing_valid_output"] = True
            member_record["runtime_seconds"] = 0.0
            member_record["output_file_size_bytes"] = int(output_nc.stat().st_size)
            member_record["metadata_file_size_bytes"] = int(output_meta.stat().st_size) if output_meta.exists() else None
            summary["members"].append(member_record)
            continue

        print(f"{member_name}: rng_key={rng_seed} encode start", flush=True)
        encode_start = time.time()
        initial_state = model.encode(inputs, input_forcings, rng_key=jax.random.key(rng_seed))
        encode_end = time.time()
        print(f"{member_name}: encode done in {encode_end - encode_start:.3f} s", flush=True)

        unroll_start = time.time()
        print(f"{member_name}: unroll start", flush=True)
        _, predictions = model.unroll(
            initial_state,
            temporal_forcings,
            steps=steps,
            timedelta=np.timedelta64(6, "h"),
            start_with_input=True,
        )
        unroll_end = time.time()
        print(f"{member_name}: unroll done in {unroll_end - unroll_start:.3f} s", flush=True)

        predictions_ds = model.data_to_xarray(predictions, times=output_times)
        diagnostics = build_diagnostics(predictions_ds)
        del predictions
        del predictions_ds
        gc.collect()
        jax.clear_caches()
        print(f"{member_name}: writing {output_nc}", flush=True)
        diagnostics.to_netcdf(output_nc)

        metadata = {
            "member_name": member_name,
            "rng_key": rng_seed,
            "checkpoint_name": CHECKPOINT_NAME,
            "initial_condition_path": str(IC_PATH),
            "forcing_path": str(FORCING_PATH),
            "output_variables": list(diagnostics.data_vars),
            "output_sizes": {k: int(v) for k, v in diagnostics.sizes.items()},
            "output_time_start": str(np.asarray(diagnostics.time.values[0]).item()),
            "output_time_end": str(np.asarray(diagnostics.time.values[-1]).item()),
            "output_file": str(output_nc),
            "output_file_size_bytes": int(output_nc.stat().st_size),
            "forcing_sizes": forcing_sizes,
            "full_model_variables_before_reduction": list(predictions_ds.data_vars),
        }
        save_json(output_meta, metadata)

        member_end = time.time()
        member_record.update(
            {
                "encode_runtime_seconds": encode_end - encode_start,
                "unroll_runtime_seconds": unroll_end - unroll_start,
                "runtime_seconds": member_end - member_start,
                "steps": steps,
                "steps_per_second": steps / max(unroll_end - unroll_start, 1e-6),
                "output_file_size_bytes": int(output_nc.stat().st_size),
                "metadata_file_size_bytes": int(output_meta.stat().st_size),
            }
        )
        final_valid, final_validation = validate_member_output(output_nc)
        member_record["validation_after_run"] = final_validation
        member_record["valid_after_run"] = final_valid
        summary["members"].append(member_record)
        print(f"{member_name}: finished valid_after_run={final_valid}", flush=True)

        del initial_state
        del diagnostics
        gc.collect()
        jax.clear_caches()

    summary["number_of_output_steps"] = steps
    summary["outer_step_hours"] = 6
    summary["total_runtime_seconds"] = time.time() - overall_start
    save_json(RUNTIME_SUMMARY_JSON, summary)
    print(f"Wrote runtime summary: {RUNTIME_SUMMARY_JSON}", flush=True)

    lines = [
        "NeuralGCM WY2021 actual M=3 runtime summary",
        f"slurm_job_id: {summary['slurm_job_id']}",
        f"node: {summary['node']}",
        f"checkpoint: {CHECKPOINT_NAME}",
        f"initial_condition_path: {IC_PATH}",
        f"forcing_path: {FORCING_PATH}",
        f"steps: {steps}",
        f"total_runtime_seconds: {summary['total_runtime_seconds']:.3f}",
        "",
    ]
    for member in summary["members"]:
        lines.extend(
            [
                f"{member['member_name']}:",
                f"  rng_key: {member['rng_key']}",
                f"  used_existing_output: {member['used_existing_output']}",
                f"  runtime_seconds: {member['runtime_seconds']:.3f}",
                f"  output_path: {member['output_path']}",
                f"  output_file_size_bytes: {member.get('output_file_size_bytes')}",
            ]
        )
    RUNTIME_SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
