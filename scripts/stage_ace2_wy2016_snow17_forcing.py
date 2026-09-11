#!/usr/bin/env python3
"""Stage ACE2-ERA5 WY2016 lag-member forcing for later Snow-17 execution.

This script only extracts and converts forcing.  It neither initializes nor
runs Snow-17 and it does not aggregate the 48 independent ACE cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
SPATIAL_TABLE = PROJECT_ROOT / "artifacts" / "ace2_era5_snow17_spatial_discretization_test" / "ace_snow17_sierra_units.csv"
OUTPUT_ROOT = ACE_ROOT / "snow17_inputs"
FORECAST_END = np.datetime64("2016-04-01T00:00:00")
MEMBERS = ("20151101", "20151102", "20151103", "20151104", "20151105")
STEP_SECONDS = 6 * 60 * 60


def valid_times(dataset: xr.Dataset) -> np.ndarray:
    values = np.asarray(dataset["valid_time"].isel(sample=0).values)
    return values.astype("datetime64[us]")


def stage_member(member: str, units: pd.DataFrame) -> dict[str, object]:
    source = ACE_ROOT / f"member_{member}" / "autoregressive_predictions.nc"
    output = OUTPUT_ROOT / f"member_{member}_snow17_forcing.nc"
    lat_indices = xr.DataArray(units["ace_lat_index"].to_numpy(dtype=np.int64), dims=("cell",))
    lon_indices = xr.DataArray(units["ace_lon_index"].to_numpy(dtype=np.int64), dims=("cell",))

    with xr.open_dataset(source, decode_times=False) as dataset:
        times = valid_times(dataset)
        keep = times <= FORECAST_END
        times = times[keep]
        if times.size == 0:
            raise RuntimeError(f"{source} has no valid forecasts at or before {FORECAST_END}.")
        if not np.all(np.diff(times) == np.timedelta64(6, "h")):
            raise RuntimeError(f"{source} does not provide a uniform six-hour valid-time cadence.")

        selection = {"sample": 0, "time": np.flatnonzero(keep), "lat": lat_indices, "lon": lon_indices}
        rate = dataset["PRATEsfc"].isel(selection).transpose("time", "cell").load()
        temperature_k = dataset["TMP2m"].isel(selection).transpose("time", "cell").load()
        init_time = np.asarray(dataset["init_time"].values).astype("datetime64[us]")[0]

    rate_values = np.asarray(rate.values, dtype=np.float32)
    temperature_k_values = np.asarray(temperature_k.values, dtype=np.float32)
    if not np.isfinite(rate_values).all() or not np.isfinite(temperature_k_values).all():
        raise RuntimeError(f"{source} has non-finite selected precipitation or temperature values.")
    if np.any(rate_values < 0.0):
        raise RuntimeError(f"{source} has negative selected precipitation rates.")

    coords = {
        "time": ("time", times),
        "cell": ("cell", np.arange(len(units), dtype=np.int32)),
        "ace_lat_index": ("cell", units["ace_lat_index"].to_numpy(dtype=np.int32)),
        "ace_lon_index": ("cell", units["ace_lon_index"].to_numpy(dtype=np.int32)),
        "ace_lat": ("cell", units["ace_lat"].to_numpy(dtype=np.float32)),
        "ace_lon": ("cell", units["ace_lon"].to_numpy(dtype=np.float32)),
        "ace_lon_0_360": ("cell", units["ace_lon_0_360"].to_numpy(dtype=np.float32)),
        "normalized_sierra_weight": ("cell", units["normalized_sierra_weight"].to_numpy(dtype=np.float64)),
        "overlap_fraction": ("cell", units["overlap_fraction"].to_numpy(dtype=np.float64)),
        "intersection_area_m2": ("cell", units["intersection_area_m2"].to_numpy(dtype=np.float64)),
    }
    staged = xr.Dataset(
        data_vars={
            "precipitation_rate_kg_m2_s": (
                ("time", "cell"),
                rate_values,
                {
                    "units": "kg m-2 s-1",
                    "source_variable": "PRATEsfc",
                    "long_name": "ACE2-ERA5 surface precipitation rate",
                },
            ),
            "precipitation_mm_6h": (
                ("time", "cell"),
                (rate_values * STEP_SECONDS).astype(np.float32),
                {
                    "units": "mm",
                    "source_variable": "PRATEsfc",
                    "conversion": "kg m-2 s-1 multiplied by 21600 seconds; 1 kg m-2 liquid water equals 1 mm.",
                    "long_name": "Six-hour precipitation depth for Snow-17 forcing",
                },
            ),
            "air_temperature_k": (
                ("time", "cell"),
                temperature_k_values,
                {"units": "K", "source_variable": "TMP2m", "long_name": "ACE2-ERA5 2 m air temperature"},
            ),
            "air_temperature_c": (
                ("time", "cell"),
                (temperature_k_values - 273.15).astype(np.float32),
                {
                    "units": "degree_Celsius",
                    "source_variable": "TMP2m",
                    "conversion": "K minus 273.15",
                    "long_name": "Two-meter air temperature for Snow-17 forcing",
                },
            ),
        },
        coords=coords,
        attrs={
            "title": "ACE2-ERA5 WY2016 native-cell Snow-17 forcing staging product",
            "source_ace_output": str(source),
            "forecast_initialization_time": np.datetime_as_string(init_time, unit="s") + "Z",
            "forecast_valid_time_end_included": np.datetime_as_string(FORECAST_END, unit="s") + "Z",
            "native_cadence": "6 hours",
            "time_semantics": "ACE valid time; each precipitation depth is rate times one six-hour model step ending at the listed valid timestamp.",
            "spatial_units": "48 native ACE cells intersecting the UCLA Sierra target mask",
            "spatial_table": str(SPATIAL_TABLE),
            "aggregation_rule_for_later_use": "Run Snow-17 independently per cell; only then compute sum(normalized_sierra_weight * SWE_cell).",
            "snow17_status": "Forcing staged only; Snow-17 was not run.",
        },
    )
    staged.to_netcdf(output, engine="netcdf4")
    return {
        "member": member,
        "source": str(source),
        "output": str(output),
        "init_time": staged.attrs["forecast_initialization_time"],
        "first_valid_time": np.datetime_as_string(times[0], unit="s") + "Z",
        "last_valid_time": np.datetime_as_string(times[-1], unit="s") + "Z",
        "time_count": int(times.size),
        "cell_count": int(len(units)),
        "weight_sum": float(units["normalized_sierra_weight"].sum()),
        "precipitation_mm_6h_min": float(rate_values.min() * STEP_SECONDS),
        "precipitation_mm_6h_max": float(rate_values.max() * STEP_SECONDS),
        "temperature_c_min": float(temperature_k_values.min() - 273.15),
        "temperature_c_max": float(temperature_k_values.max() - 273.15),
    }


def main() -> None:
    if not SPATIAL_TABLE.exists():
        raise FileNotFoundError(f"Missing frozen ACE/UCLA overlap table: {SPATIAL_TABLE}")
    units = pd.read_csv(SPATIAL_TABLE)
    required = {"ace_lat_index", "ace_lon_index", "normalized_sierra_weight", "overlap_fraction", "intersection_area_m2"}
    missing = required.difference(units.columns)
    if missing or len(units) != 48:
        raise RuntimeError(f"Expected the frozen 48-cell overlap table; missing={sorted(missing)}, rows={len(units)}")
    if not np.isclose(units["normalized_sierra_weight"].sum(), 1.0, atol=1.0e-12):
        raise RuntimeError("Frozen Sierra weights do not sum to one.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    units.to_csv(OUTPUT_ROOT / "ace_snow17_sierra_units.csv", index=False)
    members = [stage_member(member, units) for member in MEMBERS]
    manifest = {
        "purpose": "Stage ACE2-ERA5 precipitation and temperature forcing for a future Snow-17 run only.",
        "source_ensemble_root": str(ACE_ROOT),
        "spatial_overlap_table_source": str(SPATIAL_TABLE),
        "spatial_overlap_table_copy": str(OUTPUT_ROOT / "ace_snow17_sierra_units.csv"),
        "forecast_end": "2016-04-01T00:00:00Z",
        "members": members,
        "snow17_not_run": True,
    }
    manifest_path = OUTPUT_ROOT / "forcing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
