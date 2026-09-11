#!/usr/bin/env python3
"""Run a provisional 48-HRU Snow-17 translation for ACE member 20151101.

This is a mechanical ACE-to-Snow-17 proof run.  It uses one documented shared
parameter set and a zero-SWE cold start because no Sierra calibration/restart
package is presently available.  Its output is not a calibrated SWE forecast.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
FORCING_PATH = ACE_ROOT / "snow17_inputs" / "member_20151101_snow17_forcing.nc"
SOURCE_ACE_PATH = ACE_ROOT / "member_20151101" / "autoregressive_predictions.nc"
SNOW17_BINARY = ACE_ROOT / "snow17_build" / "snow17"
RUN_ROOT = ACE_ROOT / "snow17_member_20151101_provisional"
FORCING_DIR = RUN_ROOT / "forcing_csv"
PARAM_DIR = RUN_ROOT / "params"
OUTPUT_DIR = RUN_ROOT / "output"
STATE_DIR = RUN_ROOT / "states"
CONFIG_DIR = RUN_ROOT / "config"
DIAGNOSTIC_DIR = RUN_ROOT / "diagnostics"

PARAMETER_SOURCE = "NOAA-OWP Snow17 official ex1 HHWM8IL parameter values; shared provisionally across all ACE cells."
PARAMETERS = {
    "scf": 2.15177,
    "mfmax": 0.930472,
    "mfmin": 0.137,
    "uadj": 0.003103,
    "si": 1515.0,
    "pxtemp": 0.713424,
    "nmf": 0.150,
    "tipm": 0.200,
    "mbase": 0.0,
    "plwhc": 0.030,
    "daygm": 0.300,
}
ADC = (0.05, 0.09, 0.16, 0.31, 0.54, 0.74, 0.84, 0.89, 0.93, 0.97, 1.00)
STEP_SECONDS = 21600


def timestamp_text(value: np.datetime64) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d%H")


def elevation_from_pressure_m(pressure_pa: np.ndarray) -> np.ndarray:
    """Estimate static elevation from ACE surface pressure using standard atmosphere."""
    if np.any(~np.isfinite(pressure_pa)) or np.any(pressure_pa <= 0.0):
        raise ValueError("Surface pressure is non-finite or non-positive.")
    return 44_330.0 * (1.0 - (pressure_pa / 101_325.0) ** 0.1903)


def write_parameter_file(cell: pd.Series, elevation_m: float, cell_root: Path) -> Path:
    path = cell_root / "params" / "snow17_params.txt"
    values: dict[str, object] = {
        "hru_id": f"c{int(cell.name):03d}",
        # A one-HRU run only needs a positive area.  The project-level area
        # aggregation is deliberately performed afterward with frozen weights.
        "hru_area": 1.0,
        "latitude": float(cell.ace_lat),
        "elev": float(elevation_m),
        **PARAMETERS,
    }
    lines = [f"{name} {value}" for name, value in values.items()]
    lines.extend(f"adc{index + 1} {value}" for index, value in enumerate(ADC))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_forcing_csvs(forcing: xr.Dataset, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ace_valid_time = forcing.time.values.astype("datetime64[s]")
    model_times = ace_valid_time - np.timedelta64(6, "h")
    precipitation_rate = forcing.precipitation_rate_kg_m2_s.values
    temperature_c = forcing.air_temperature_c.values
    if not np.all(np.diff(model_times) == np.timedelta64(6, "h")):
        raise RuntimeError("Expected uniform six-hour model forcing timestamps.")
    for index in range(len(cells)):
        timestamps = pd.DatetimeIndex(model_times)
        table = pd.DataFrame(
            {
                "year": timestamps.year,
                "mo": timestamps.month,
                "dy": timestamps.day,
                "hr": timestamps.hour,
                "prec_mm_s-1": precipitation_rate[:, index],
                "tavg_degc": temperature_c[:, index],
            }
        )
        table.to_csv(FORCING_DIR / f"forcing.c{index:03d}.csv", index=False, float_format="%.9g")
    return model_times, ace_valid_time


def write_namelist(model_times: np.ndarray, cell_index: int, parameter_path: Path, cell_root: Path) -> Path:
    path = cell_root / "config" / "namelist.input"
    start, end = timestamp_text(model_times[0]), timestamp_text(model_times[-1])
    content = f'''&SNOW17_CONTROL
main_id             = "c{cell_index:03d}"
n_hrus              = 1
forcing_root        = "{FORCING_DIR}/forcing."
output_root         = "{cell_root}/output/output.snow17."
snow17_param_file   = "{parameter_path}"
output_hrus         = 0
start_datehr        = {start}
end_datehr          = {end}
model_timestep      = {STEP_SECONDS}
warm_start_run      = 0
write_states        = 1
snow_state_in_root  = "{cell_root}/states/snow17_states."
snow_state_out_root = "{cell_root}/states/snow17_states."
/
'''
    path.write_text(content, encoding="utf-8")
    return path


def parse_snow17_output(cell_roots: list[Path], ace_valid_time: np.ndarray, cells: pd.DataFrame) -> xr.Dataset:
    cell_swe = []
    cell_snowh = []
    for index, cell_root in enumerate(cell_roots):
        table = pd.read_csv(cell_root / "output" / f"output.snow17.c{index:03d}.txt", sep=r"\s+")
        if len(table) != len(ace_valid_time):
            raise RuntimeError(f"Unexpected row count for cell {index}: {len(table)}")
        cell_swe.append(table["sneqv"].to_numpy(dtype=np.float32))
        cell_snowh.append(table["snowh"].to_numpy(dtype=np.float32))
    swe = np.stack(cell_swe, axis=1)
    snowh = np.stack(cell_snowh, axis=1)
    weights = cells.normalized_sierra_weight.to_numpy(dtype=np.float64)
    area_average = (swe * weights[None, :]).sum(axis=1).astype(np.float32)
    result = xr.Dataset(
        data_vars={
            "swe_cell_mm": (("time", "cell"), swe, {"units": "mm", "long_name": "Provisional Snow-17 SWE by ACE cell"}),
            "snow_depth_cell_mm": (("time", "cell"), snowh, {"units": "mm"}),
            "swe_sierra_area_weighted_mean_mm": (("time",), area_average, {"units": "mm"}),
        },
        coords={
            "time": ("time", ace_valid_time),
            "cell": ("cell", np.arange(len(cells), dtype=np.int32)),
            "ace_lat_index": ("cell", cells.ace_lat_index.to_numpy(dtype=np.int32)),
            "ace_lon_index": ("cell", cells.ace_lon_index.to_numpy(dtype=np.int32)),
            "ace_lat": ("cell", cells.ace_lat.to_numpy(dtype=np.float32)),
            "ace_lon": ("cell", cells.ace_lon.to_numpy(dtype=np.float32)),
            "normalized_sierra_weight": ("cell", weights),
        },
        attrs={
            "status": "Provisional cold-start Snow-17 translation; not a calibrated Sierra SWE forecast.",
            "parameter_source": PARAMETER_SOURCE,
            "initial_state": "Cold start: all Snow-17 state variables initialized to zero at 2015-11-01T00:00:00Z.",
            "forcing_source": str(FORCING_PATH),
            "snow17_binary": str(SNOW17_BINARY),
            "output_time_semantics": "SWE state after each 6-hour ACE forcing interval, labeled with ACE valid time.",
            "aggregation": "Manual frozen UCLA-ACE weights applied after independent one-HRU Snow-17 runs.",
        },
    )
    return result


def main() -> None:
    if "--allow-provisional-demo" not in sys.argv:
        raise RuntimeError(
            "Refusing to run: this script uses NOAA example parameters, a cold start, "
            "and a pressure-derived elevation. It is a software-only demonstration, "
            "not a scientific Sierra Snow-17 workflow. Pass --allow-provisional-demo "
            "only for an explicitly requested execution smoke test."
        )
    if not FORCING_PATH.exists() or not SNOW17_BINARY.exists():
        raise FileNotFoundError("Missing staged forcing or standalone Snow-17 executable.")
    for directory in (FORCING_DIR, PARAM_DIR, OUTPUT_DIR, STATE_DIR, CONFIG_DIR, DIAGNOSTIC_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(FORCING_PATH) as input_dataset:
        forcing = input_dataset.load()
    cells = pd.read_csv(ACE_ROOT / "snow17_inputs" / "ace_snow17_sierra_units.csv")
    if len(cells) != 48 or not np.isclose(cells.normalized_sierra_weight.sum(), 1.0, atol=1.0e-10):
        raise RuntimeError("The frozen ACE overlap definition is invalid.")
    with xr.open_dataset(SOURCE_ACE_PATH, decode_times=False) as source:
        lat_index = xr.DataArray(cells.ace_lat_index.to_numpy(dtype=np.int64), dims="cell")
        lon_index = xr.DataArray(cells.ace_lon_index.to_numpy(dtype=np.int64), dims="cell")
        pressure_pa = source.PRESsfc.isel(sample=0, time=0, lat=lat_index, lon=lon_index).load().values
    elevation_m = elevation_from_pressure_m(np.asarray(pressure_pa, dtype=np.float64))

    model_times, ace_valid_time = write_forcing_csvs(forcing, cells)
    cell_roots: list[Path] = []
    snow17_logs: list[str] = []
    for index, cell in cells.iterrows():
        cell_root = RUN_ROOT / "cells" / f"c{index:03d}"
        for directory in (cell_root / "params", cell_root / "output", cell_root / "states", cell_root / "config"):
            directory.mkdir(parents=True, exist_ok=True)
        parameter_path = write_parameter_file(cell, elevation_m[index], cell_root)
        namelist_path = write_namelist(model_times, index, parameter_path, cell_root)
        completed = subprocess.run([str(SNOW17_BINARY), str(namelist_path)], capture_output=True, text=True, check=False)
        (cell_root / "snow17_stdout.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Snow-17 failed for cell {index} with return code {completed.returncode}.")
        cell_roots.append(cell_root)
        snow17_logs.append(str(cell_root / "snow17_stdout.log"))
    prepared_metadata = {
        "run_type": "provisional cold-start workflow proof",
        "known_scientific_limitations": [
            "Snow-17 parameters are shared official-example values, not Sierra calibrated.",
            "No November 1 Snow-17 restart state is available; all cells begin with zero SWE and zero internal state.",
            "Cell elevation is a standard-atmosphere estimate from ACE PRESsfc at the first forecast output.",
        ],
        "forcing_timing": "ACE valid-time rate was assigned to the preceding six-hour Snow-17 interval; output is relabeled to ACE valid time.",
        "parameterization": "48 independent one-HRU runs using the same provisional parameters; frozen Sierra weights applied after simulation.",
        "elevation_estimate_m_min": float(elevation_m.min()),
        "elevation_estimate_m_max": float(elevation_m.max()),
        "weight_sum": float(cells.normalized_sierra_weight.sum()),
    }
    (DIAGNOSTIC_DIR / "prepared_input_metadata.json").write_text(json.dumps(prepared_metadata, indent=2) + "\n", encoding="utf-8")

    result = parse_snow17_output(cell_roots, ace_valid_time, cells)
    output_netcdf = RUN_ROOT / "snow17_member_20151101_provisional_swe.nc"
    result.to_netcdf(output_netcdf, engine="netcdf4")
    summary = {
        **prepared_metadata,
        "snow17_exit_code": 0,
        "cell_log_count": len(snow17_logs),
        "output_netcdf": str(output_netcdf),
        "time_count": int(result.sizes["time"]),
        "cell_count": int(result.sizes["cell"]),
        "output_start": str(result.time.values[0]),
        "output_end": str(result.time.values[-1]),
        "sierra_swe_final_mm": float(result.swe_sierra_area_weighted_mean_mm.isel(time=-1)),
        "sierra_swe_max_mm": float(result.swe_sierra_area_weighted_mean_mm.max()),
    }
    (DIAGNOSTIC_DIR / "snow17_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
