#!/usr/bin/env python3
"""Combine 19 observational predictor fields into the frozen CMIP6 CNN's raw input cache format.

Sources (each already on the exact CMIP6-training 1.5-degree grid, Sep-Mar water-year rows):
  - 16 fields from scripts/build_era5_obs_predictors.py (ERA5-derived)
  - mrso from scripts/preprocess_era5land_full_column_soil_moisture.py (ERA5-Land-derived)
  - thetao_50m, thetao_100m from scripts/stage_en4_thetao_obs_audit.py (EN4-derived, already complete)

Writes inputs_physical.npy / inputs_valid_mask.npy / manifest.csv / summary.json /
predictor_inventory.json to a standalone scratch directory, in exactly the PREDICTOR_SPECS
order from scripts/prepare_cmip6_cnn_experiment.py, so any of the frozen S0/S1/S2/U1
checkpoints can consume it via their own normalization_stats.npz. Never writes into or reads
from cmip6_cnn_architecture_screen_v1/data_cache (the production CMIP6 training cache).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from process_cmip6_selected_regrid_standardize import TARGET_LAT, TARGET_LON  # noqa: E402
from build_era5_obs_predictors import FIELD_SPECS_BY_ID, CNN_ROWS_DIR  # noqa: E402

MONTH_LABELS = ("Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar")

# Matches PREDICTOR_SPECS order in scripts/prepare_cmip6_cnn_experiment.py:14-34 exactly.
PREDICTOR_ORDER = (
    "tos", "siconc", "rlut", "zg_500", "ta_850", "ua_850", "va_850",
    "ua_200", "va_200", "hus_850", "psl", "tas", "zg_50", "ta_50",
    "ua_50", "va_50", "mrso", "thetao_50m", "thetao_100m",
)

ERA5LAND_MRSO_DIR = PROJECT_ROOT / "artifacts" / "era5land_full_column_soil_moisture_1p5deg"
MRSO_WY_FILE = ERA5LAND_MRSO_DIR / "era5land_full_column_soil_moisture_wy1985_2021_sep_mar_1p5deg.nc"

EN4_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/en4_thetao_stage/full_1984_2021/processed_cnn_wy1985_2021"
)
THETAO_50M_FILE = EN4_DIR / "thetao_50m_1p5deg_cnn_rows.nc"
THETAO_100M_FILE = EN4_DIR / "thetao_100m_1p5deg_cnn_rows.nc"

DEFAULT_OUTPUT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational"
)


def load_field_array(field_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (physical[water_year,7,lat,lon] float32, water_years[int]) sorted by water_year."""
    if field_id in FIELD_SPECS_BY_ID:
        spec = FIELD_SPECS_BY_ID[field_id]
        path = CNN_ROWS_DIR / f"{field_id}_1p5deg_cnn_rows.nc"
        var_name = spec.output_name
        lat_name, lon_name = "lat", "lon"
    elif field_id == "mrso":
        path = MRSO_WY_FILE
        var_name = "mrso_full_column"
        lat_name, lon_name = "latitude", "longitude"
    elif field_id == "thetao_50m":
        path = THETAO_50M_FILE
        var_name = "thetao_50m"
        lat_name, lon_name = "lat", "lon"
    elif field_id == "thetao_100m":
        path = THETAO_100M_FILE
        var_name = "thetao_100m"
        lat_name, lon_name = "lat", "lon"
    else:
        raise ValueError(f"Unknown field_id: {field_id}")

    if not path.exists():
        raise FileNotFoundError(f"{field_id}: expected input file not found: {path}")

    with xr.open_dataset(path) as ds:
        if var_name not in ds.data_vars:
            raise ValueError(f"{field_id}: variable {var_name!r} not found in {path}; found {list(ds.data_vars)}")
        da = ds[var_name]
        lat = np.asarray(ds[lat_name].values, dtype=np.float64)
        lon = np.asarray(ds[lon_name].values, dtype=np.float64)
        if lat.shape != TARGET_LAT.shape or not np.allclose(lat, TARGET_LAT):
            raise ValueError(f"{field_id}: latitude grid mismatch against TARGET_LAT in {path}")
        if lon.shape != TARGET_LON.shape or not np.allclose(lon, TARGET_LON):
            raise ValueError(f"{field_id}: longitude grid mismatch against TARGET_LON in {path}")
        if "water_year" not in ds.coords:
            raise ValueError(f"{field_id}: no water_year coordinate found in {path}")
        water_years = np.asarray(ds["water_year"].values, dtype=np.int64)
        order = np.argsort(water_years)
        water_years = water_years[order]
        values = np.asarray(da.values, dtype=np.float32)[order]

    expected_shape = (len(water_years), len(MONTH_LABELS), len(TARGET_LAT), len(TARGET_LON))
    if values.shape != expected_shape:
        raise ValueError(f"{field_id}: unexpected array shape {values.shape} != {expected_shape} from {path}")
    return values, water_years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_water_years: np.ndarray | None = None
    physical_by_field: dict[str, np.ndarray] = {}
    for field_id in PREDICTOR_ORDER:
        values, water_years = load_field_array(field_id)
        if reference_water_years is None:
            reference_water_years = water_years
        elif not np.array_equal(reference_water_years, water_years):
            raise ValueError(f"{field_id}: water_year set {water_years.tolist()} != reference {reference_water_years.tolist()}")
        physical_by_field[field_id] = values
        finite_frac = float(np.isfinite(values).mean())
        print(f"[{field_id}] loaded shape={values.shape} finite_fraction={finite_frac:.4f}", flush=True)

    assert reference_water_years is not None
    n_samples = len(reference_water_years)
    n_months = len(MONTH_LABELS)
    n_fields = len(PREDICTOR_ORDER)
    n_lat = len(TARGET_LAT)
    n_lon = len(TARGET_LON)

    physical = np.zeros((n_samples, n_months, n_fields, n_lat, n_lon), dtype=np.float32)
    for field_index, field_id in enumerate(PREDICTOR_ORDER):
        physical[:, :, field_index, :, :] = physical_by_field[field_id]
    mask = np.isfinite(physical).astype(np.uint8)

    physical_path = output_dir / "inputs_physical.npy"
    mask_path = output_dir / "inputs_valid_mask.npy"
    np.save(physical_path, physical)
    np.save(mask_path, mask)

    manifest_df = pd.DataFrame(
        {
            "sample_index": np.arange(n_samples, dtype=np.int64),
            "water_year": reference_water_years,
            "row_year": reference_water_years - 1,
            "source_id": "observational",
            "member_id": "ERA5_ERA5Land_EN4",
            "model_member": "observational:ERA5_ERA5Land_EN4",
            "experiment": "observational",
        }
    )
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "output_dir": str(output_dir),
        "predictor_set": "observational_all19",
        "sample_count": int(n_samples),
        "month_labels": list(MONTH_LABELS),
        "predictor_fields": list(PREDICTOR_ORDER),
        "physical_shape": [int(n_samples), n_months, n_fields, n_lat, n_lon],
        "valid_mask_shape": [int(n_samples), n_months, n_fields, n_lat, n_lon],
        "water_year_range": [int(reference_water_years.min()), int(reference_water_years.max())],
        "water_years": [int(y) for y in reference_water_years],
        "grid": {
            "lat_first_last_step": [float(TARGET_LAT[0]), float(TARGET_LAT[-1]), float(TARGET_LAT[1] - TARGET_LAT[0])],
            "lon_first_last_step": [float(TARGET_LON[0]), float(TARGET_LON[-1]), float(TARGET_LON[1] - TARGET_LON[0])],
            "shape": [n_lat, n_lon],
        },
        "sources": {
            "era5_derived_16": "ERA5 (NCAR RDA ds633.0), built by scripts/build_era5_obs_predictors.py",
            "mrso": "ERA5-Land, built by scripts/preprocess_era5land_full_column_soil_moisture.py",
            "thetao_50m_100m": "EN4.2.2.g10, built by scripts/stage_en4_thetao_obs_audit.py",
        },
        "finite_fraction_by_field": {
            field_id: float(mask[:, :, i].mean()) for i, field_id in enumerate(PREDICTOR_ORDER)
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    predictor_inventory = {
        "predictor_set": "observational_all19",
        "predictor_fields": list(PREDICTOR_ORDER),
        "predictor_count": n_fields,
        "mask_count": n_fields,
        "stacked_channel_count": n_fields * 2,
        "months": list(MONTH_LABELS),
        "stacked_tensor_shape_per_sample": [n_months, n_fields * 2, n_lat, n_lon],
        "flattened_cnn_input_shape_per_sample": [n_months * n_fields * 2, n_lat, n_lon],
    }
    with (output_dir / "predictor_inventory.json").open("w") as handle:
        json.dump(predictor_inventory, handle, indent=2, sort_keys=True)

    print(f"\nWrote physical inputs: {physical_path} shape={physical.shape}")
    print(f"Wrote valid mask: {mask_path} shape={mask.shape}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")
    print(f"Wrote predictor inventory: {output_dir / 'predictor_inventory.json'}")
    print("\nFinite fraction per field:")
    for field_id in PREDICTOR_ORDER:
        print(f"  {field_id}: {summary['finite_fraction_by_field'][field_id]:.4f}")


if __name__ == "__main__":
    main()
