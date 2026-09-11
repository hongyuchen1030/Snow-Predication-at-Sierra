#!/usr/bin/env python3
"""Diagnose the non-scientific provisional Snow-17 software demonstration.

This script never runs Snow-17.  It retains the original output solely to show
which forcing and setup limitations could explain its very small SWE values.
The 0 C fraction is a thermodynamic diagnostic, not a Snow-17 PXTEMP choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
FORCING = ACE_ROOT / "snow17_inputs/member_20151101_snow17_forcing.nc"
PROVISIONAL = ACE_ROOT / "snow17_member_20151101_provisional/snow17_member_20151101_provisional_swe.nc"
OUTPUT = PROJECT_ROOT / "artifacts/ace2_era5_snow17_scientific_validity_audit"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(FORCING) as ds:
        forcing = ds.load()
    with xr.open_dataset(PROVISIONAL) as ds:
        provisional = ds.load()

    weights = forcing.normalized_sierra_weight.values.astype(np.float64)
    precipitation = forcing.precipitation_mm_6h.values.astype(np.float64)
    temperature_c = forcing.air_temperature_c.values.astype(np.float64)
    swe = provisional.swe_cell_mm.values.astype(np.float64)
    times = pd.DatetimeIndex(forcing.time.values)
    if not np.allclose(weights.sum(), 1.0):
        raise RuntimeError("Frozen UCLA-ACE weights must sum to one.")

    mean_precip = precipitation @ weights
    mean_temperature = temperature_c @ weights
    below_freezing = (temperature_c <= 0.0) @ weights
    sierra_swe = swe @ weights
    cell_table = pd.DataFrame(
        {
            "cell": forcing.cell.values,
            "ace_lat": forcing.ace_lat.values,
            "ace_lon": forcing.ace_lon.values,
            "normalized_sierra_weight": weights,
            "ace_precipitation_total_mm": precipitation.sum(axis=0),
            "ace_temperature_mean_c": temperature_c.mean(axis=0),
            "fraction_six_hour_periods_at_or_below_0c": (temperature_c <= 0.0).mean(axis=0),
            "provisional_swe_max_mm": swe.max(axis=0),
            "provisional_swe_final_mm": swe[-1],
        }
    ).sort_values("normalized_sierra_weight", ascending=False)
    cell_table.to_csv(OUTPUT / "member_20151101_cell_diagnostics.csv", index=False)

    summary = {
        "classification": "diagnostic only; provisional Snow-17 SWE is not a forecast",
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "cadence_hours": 6,
        "sierra_area_weighted_ace_precipitation_total_mm": float(mean_precip.sum()),
        "sierra_area_weighted_ace_temperature_mean_c": float(mean_temperature.mean()),
        "sierra_area_weighted_fraction_of_cell_area_time_at_or_below_0c": float(below_freezing.mean()),
        "zero_celsius_note": "Physical freezing reference only; not a calibrated Snow-17 PXTEMP.",
        "provisional_swe_max_mm": float(sierra_swe.max()),
        "provisional_swe_final_mm": float(sierra_swe[-1]),
        "cell_precipitation_total_mm_range": [float(cell_table.ace_precipitation_total_mm.min()), float(cell_table.ace_precipitation_total_mm.max())],
        "cell_temperature_mean_c_range": [float(cell_table.ace_temperature_mean_c.min()), float(cell_table.ace_temperature_mean_c.max())],
        "cell_below_0c_fraction_range": [float(cell_table.fraction_six_hour_periods_at_or_below_0c.min()), float(cell_table.fraction_six_hour_periods_at_or_below_0c.max())],
        "known_non_scientific_inputs": [
            "NOAA-OWP example Snow-17 parameters shared across all cells",
            "zero cold-start Snow-17 state",
            "pressure-derived terrain estimate",
            "no Sierra-specific temperature elevation correction",
        ],
    }
    (OUTPUT / "member_20151101_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(times, np.cumsum(mean_precip), color="#176b87", linewidth=2, label="ACE precipitation, cumulative")
    axes[0].set_ylabel("mm")
    axes[0].set_title("Member 20151101 forcing and provisional Snow-17 diagnostic")
    axes[0].legend(loc="upper left")
    temperature_axis = axes[0].twinx()
    temperature_axis.plot(times, mean_temperature, color="#c85200", alpha=0.75, label="ACE TMP2m")
    temperature_axis.set_ylabel("deg C")
    temperature_axis.axhline(0.0, color="black", linewidth=0.7, linestyle="--")

    axes[1].plot(times, below_freezing, color="#5b5ea6", linewidth=1.8)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("area fraction")
    axes[1].set_title("ACE cell-area fraction at or below 0 C (diagnostic reference only)")

    axes[2].plot(times, sierra_swe, color="#11823b", linewidth=2)
    axes[2].set_ylabel("mm SWE")
    axes[2].set_title("Provisional output: software demonstration only, not a forecast")
    axes[2].set_xlabel("ACE valid time")
    fig.savefig(OUTPUT / "member_20151101_timeseries_diagnostic.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    scatter = axes[0].scatter(
        cell_table.ace_temperature_mean_c,
        cell_table.ace_precipitation_total_mm,
        s=40 + 1400 * cell_table.normalized_sierra_weight,
        c=cell_table.fraction_six_hour_periods_at_or_below_0c,
        cmap="Blues",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].set_xlabel("mean ACE TMP2m (deg C)")
    axes[0].set_ylabel("ACE precipitation total (mm)")
    axes[0].set_title("48 ACE cells")
    fig.colorbar(scatter, ax=axes[0], label="fraction at or below 0 C")
    axes[1].scatter(
        cell_table.ace_temperature_mean_c,
        cell_table.provisional_swe_max_mm,
        s=40 + 1400 * cell_table.normalized_sierra_weight,
        color="#11823b",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_xlabel("mean ACE TMP2m (deg C)")
    axes[1].set_ylabel("maximum provisional SWE (mm)")
    axes[1].set_title("Demonstration output, not scientific SWE")
    fig.savefig(OUTPUT / "member_20151101_cell_diagnostic.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
