#!/usr/bin/env python3
"""Run a WY2021 NeuralGCM stochastic ensemble for one SST state."""

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
IC_PATH = NEURALGCM_ASSETS / "prepared_inputs" / "wy2021_actual_m001" / "initial_condition_regridded.nc"
ACTUAL_FORCING_PATH = NEURALGCM_ASSETS / "prepared_inputs" / "wy2021_actual_m001" / "forcing_regridded_6h.nc"
REVERSED_FORCING_PATH = NEURALGCM_ASSETS / "prepared_inputs" / "wy2021_reversed_pacific_m030" / "forcing_regridded_6h.nc"
M003_ROOT = NEURALGCM_OUTPUTS / "wy2021_actual_m003"

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
    parser.add_argument("--state", choices=["actual", "reversed_pacific"], required=True)
    parser.add_argument("--start-member", type=int, default=0)
    parser.add_argument("--end-member", type=int, default=29)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def state_output_root(state: str) -> Path:
    return NEURALGCM_OUTPUTS / f"wy2021_{state}_m030"


def state_runtime_paths(state: str) -> tuple[Path, Path]:
    stem = f"neuralgcm_wy2021_{state}_m030_runtime_summary"
    return REPORT_DIR / f"{stem}.json", REPORT_DIR / f"{stem}.txt"


def forcing_path_for_state(state: str) -> Path:
    return ACTUAL_FORCING_PATH if state == "actual" else REVERSED_FORCING_PATH


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


def load_model() -> neuralgcm.PressureLevelModel:
    with CHECKPOINT_LOCAL.open("rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def output_paths(output_root: Path, member_name: str) -> tuple[Path, Path]:
    member_dir = output_root / member_name
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
    except Exception as exc:
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


def reusable_actual_m003_member(member_name: str, rng_seed: int) -> tuple[bool, dict]:
    output_nc = M003_ROOT / member_name / "predictions_6h.nc"
    meta_json = M003_ROOT / member_name / "model_output_metadata.json"
    valid, validation = validate_member_output(output_nc)
    if not valid or not meta_json.exists():
        return False, {"valid_output": valid, "validation": validation, "metadata_exists": meta_json.exists()}
    metadata = json.loads(meta_json.read_text(encoding="utf-8"))
    compatible = (
        metadata.get("rng_key") == rng_seed
        and metadata.get("checkpoint_name") == CHECKPOINT_NAME
        and metadata.get("initial_condition_path") == str(IC_PATH)
        and metadata.get("forcing_path") == str(ACTUAL_FORCING_PATH)
    )
    return compatible, {"validation": validation, "metadata": metadata}


def adopt_existing_actual_member(member_name: str, rng_seed: int, output_root: Path) -> tuple[bool, dict]:
    compatible, details = reusable_actual_m003_member(member_name, rng_seed)
    if not compatible:
        return False, details

    source_nc = M003_ROOT / member_name / "predictions_6h.nc"
    dest_nc, dest_meta = output_paths(output_root, member_name)
    dest_nc.parent.mkdir(parents=True, exist_ok=True)
    if not dest_nc.exists():
        os.link(source_nc, dest_nc)

    metadata = {
        "member_name": member_name,
        "rng_key": rng_seed,
        "state": "actual",
        "checkpoint_name": CHECKPOINT_NAME,
        "initial_condition_path": str(IC_PATH),
        "forcing_path": str(ACTUAL_FORCING_PATH),
        "output_variables": EXPECTED_VARIABLES,
        "output_file": str(dest_nc),
        "output_file_size_bytes": int(dest_nc.stat().st_size),
        "adopted_from_existing_output": str(source_nc),
        "adopted_from_m003": True,
    }
    save_json(dest_meta, metadata)
    return True, details


def build_runtime_text(summary: dict) -> str:
    lines = [
        f"NeuralGCM WY2021 {summary['state']} M=30 runtime summary",
        f"slurm_job_id: {summary['slurm_job_id']}",
        f"node: {summary['node']}",
        f"checkpoint: {summary['checkpoint_name']}",
        f"initial_condition_path: {summary['initial_condition_path']}",
        f"forcing_path: {summary['forcing_path']}",
        f"member_range: {summary['start_member']}..{summary['end_member']}",
        f"steps: {summary['number_of_output_steps']}",
        f"total_runtime_seconds: {summary['total_runtime_seconds']:.3f}",
        "",
    ]
    for member in summary["members"]:
        lines.extend(
            [
                f"{member['member_name']}:",
                f"  rng_key: {member['rng_key']}",
                f"  used_existing_output: {member.get('used_existing_output', False)}",
                f"  runtime_seconds: {member['runtime_seconds']:.3f}",
                f"  output_path: {member['output_path']}",
                f"  output_file_size_bytes: {member.get('output_file_size_bytes')}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_root = state_output_root(args.state)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_json, runtime_txt = state_runtime_paths(args.state)

    forcing_path = forcing_path_for_state(args.state)
    if not CHECKPOINT_LOCAL.exists() or not IC_PATH.exists() or not forcing_path.exists():
        raise FileNotFoundError("Missing checkpoint, IC, or forcing file for requested state.")

    model = load_model()
    overall_start = time.time()
    summary = {
        "command": f"python scripts/neuralgcm_feasibility/run_neuralgcm_wy2021_sst_state_m030.py --state {args.state} --start-member {args.start_member} --end-member {args.end_member}",
        "state": args.state,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "node": socket.gethostname(),
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_local_path": str(CHECKPOINT_LOCAL),
        "initial_condition_path": str(IC_PATH),
        "forcing_path": str(forcing_path),
        "output_root": str(output_root),
        "start_member": args.start_member,
        "end_member": args.end_member,
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "members": [],
    }

    with xr.open_dataset(IC_PATH) as initial_ds, xr.open_dataset(forcing_path) as forcing_ds:
        inputs = model.inputs_from_xarray(initial_ds)
        input_forcings = model.forcings_from_xarray(forcing_ds.isel(time=0))
        temporal_forcings = model.forcings_from_xarray(forcing_ds)
        output_times = np.asarray(forcing_ds.time.values)
        forcing_sizes = {k: int(v) for k, v in forcing_ds.sizes.items()}

    steps = int(len(output_times))

    for member_id in range(args.start_member, args.end_member + 1):
        member_name = f"member_{member_id:03d}"
        rng_seed = member_id
        output_nc, output_meta = output_paths(output_root, member_name)
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        member_start = time.time()
        valid, validation = validate_member_output(output_nc)
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
            member_record["used_existing_output"] = True
            member_record["skipped_existing_valid_output"] = True
            member_record["runtime_seconds"] = 0.0
            member_record["output_file_size_bytes"] = int(output_nc.stat().st_size)
            member_record["metadata_file_size_bytes"] = int(output_meta.stat().st_size) if output_meta.exists() else None
            summary["members"].append(member_record)
            print(f"{args.state} {member_name}: reusing existing valid output", flush=True)
            continue

        if args.state == "actual" and not output_nc.exists() and not args.force:
            adopted, adopt_details = adopt_existing_actual_member(member_name, rng_seed, output_root)
            if adopted:
                final_valid, final_validation = validate_member_output(output_nc)
                member_record.update(
                    {
                        "used_existing_output": True,
                        "adopted_from_existing_m003": True,
                        "validation_before_run": adopt_details.get("validation"),
                        "validation_after_run": final_validation,
                        "valid_after_run": final_valid,
                        "runtime_seconds": 0.0,
                        "output_file_size_bytes": int(output_nc.stat().st_size),
                        "metadata_file_size_bytes": int(output_meta.stat().st_size),
                    }
                )
                summary["members"].append(member_record)
                print(f"{args.state} {member_name}: adopted compatible member from {M003_ROOT}", flush=True)
                continue

        print(f"{args.state} {member_name}: rng_key={rng_seed} encode start", flush=True)
        encode_start = time.time()
        initial_state = model.encode(inputs, input_forcings, rng_key=jax.random.key(rng_seed))
        encode_end = time.time()
        print(f"{args.state} {member_name}: encode done in {encode_end - encode_start:.3f} s", flush=True)

        unroll_start = time.time()
        _, predictions = model.unroll(
            initial_state,
            temporal_forcings,
            steps=steps,
            timedelta=np.timedelta64(6, "h"),
            start_with_input=True,
        )
        unroll_end = time.time()
        print(f"{args.state} {member_name}: unroll done in {unroll_end - unroll_start:.3f} s", flush=True)

        predictions_ds = model.data_to_xarray(predictions, times=output_times)
        full_variable_names = list(predictions_ds.data_vars)
        diagnostics = build_diagnostics(predictions_ds)
        del predictions
        del predictions_ds
        gc.collect()
        jax.clear_caches()
        diagnostics.to_netcdf(output_nc)

        metadata = {
            "member_name": member_name,
            "rng_key": rng_seed,
            "state": args.state,
            "checkpoint_name": CHECKPOINT_NAME,
            "initial_condition_path": str(IC_PATH),
            "forcing_path": str(forcing_path),
            "season_start": "2020-09-01T00:00:00",
            "season_end": "2021-03-31T18:00:00",
            "steps": steps,
            "outer_step_hours": 6,
            "output_variables": list(diagnostics.data_vars),
            "output_sizes": {k: int(v) for k, v in diagnostics.sizes.items()},
            "output_time_start": str(np.asarray(diagnostics.time.values[0]).item()),
            "output_time_end": str(np.asarray(diagnostics.time.values[-1]).item()),
            "output_file": str(output_nc),
            "output_file_size_bytes": int(output_nc.stat().st_size),
            "forcing_sizes": forcing_sizes,
            "full_model_variables_before_reduction": full_variable_names,
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
        print(f"{args.state} {member_name}: finished valid_after_run={final_valid}", flush=True)

        del initial_state
        del diagnostics
        gc.collect()
        jax.clear_caches()

    summary["number_of_output_steps"] = steps
    summary["outer_step_hours"] = 6
    summary["total_runtime_seconds"] = time.time() - overall_start
    save_json(runtime_json, summary)
    runtime_txt.write_text(build_runtime_text(summary), encoding="utf-8")
    print(f"Wrote runtime summary: {runtime_json}", flush=True)


if __name__ == "__main__":
    main()
