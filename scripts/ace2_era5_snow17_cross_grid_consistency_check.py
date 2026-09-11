#!/usr/bin/env python3
"""Check whether UCLA, ACE, and WUS-D3 use compatible Sierra SWE averaging definitions."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ace2_era5_snow17_spatial_discretization_test import (  # noqa: E402
    ACE_OUTPUT,
    build_overlap,
    load_ace_grid,
    load_ucla_mask,
    spherical_rectangle_area,
)
from build_wusd3_swe_labels import build_mask_and_area  # noqa: E402
from snow_ml.data import DEFAULT_SIERRA_REGION  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ace2_era5_snow17_spatial_discretization_test" / "cross_grid_consistency"


def ucla_weights(mask_info: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask_info["mask"], dtype=np.float64)
    lat = np.asarray(mask_info["lat"], dtype=np.float64)
    lon = np.asarray(mask_info["lon"], dtype=np.float64)
    lat_edges = np.asarray(mask_info["lat_edges"], dtype=np.float64)
    lon_edges = np.asarray(mask_info["lon_edges"], dtype=np.float64)
    area = np.empty(mask.shape, dtype=np.float64)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            area[row, col] = spherical_rectangle_area(
                lat_edges[row], lat_edges[row + 1], lon_edges[col], lon_edges[col + 1]
            )
    weights = mask * area
    weights /= weights.sum()
    return weights, *np.meshgrid(lat, lon, indexing="ij")


def weighted_tests(weights: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> dict[str, float]:
    latitude = np.where(np.isfinite(latitude), latitude, 0.0)
    longitude = np.where(np.isfinite(longitude), longitude, 0.0)
    return {
        "weight_sum": float(weights.sum()),
        "constant_one_mean": float(np.sum(weights)),
        "latitude_mean_degrees_north": float(np.sum(weights * latitude)),
        "longitude_mean_degrees_east": float(np.sum(weights * longitude)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_info = load_ucla_mask(2016)
    ace_lat, ace_lon, ace_lat_indices, ace_lon_indices, _ = load_ace_grid(ACE_OUTPUT)
    ace = build_overlap(mask_info, ace_lat, ace_lon, ace_lat_indices, ace_lon_indices)

    ucla_weight, ucla_lat, ucla_lon = ucla_weights(mask_info)
    ace_weight = ace.normalized_sierra_weight.to_numpy(dtype=np.float64)
    _, wus_grid, wus_mask, wus_area, wus_summary = build_mask_and_area("ec-earth3_r1i1p1f1_2_historical_bc")
    wus_weight = np.asarray((wus_mask * wus_area).values, dtype=np.float64)
    # The curvilinear WUS-D3 grid carries NaN area outside its support.
    wus_weight = np.where(np.isfinite(wus_weight), wus_weight, 0.0)
    wus_weight /= wus_weight.sum()
    wus_lat = np.asarray(wus_grid.fine_latitude.values, dtype=np.float64)
    wus_lon = np.asarray(wus_grid.fine_longitude.values, dtype=np.float64)

    tests = {
        "ucla_native": weighted_tests(ucla_weight, ucla_lat, ucla_lon),
        "ace_overlap_units": weighted_tests(
            ace_weight,
            ace.ace_lat.to_numpy(dtype=np.float64),
            ace.ace_lon.to_numpy(dtype=np.float64),
        ),
        "wusd3_d02_cnn_labels": weighted_tests(wus_weight, wus_lat, wus_lon),
    }
    summary = {
        "purpose": "Static spatial-definition audit only; no Snow-17 or SWE simulation was run.",
        "canonical_requested_box": asdict(DEFAULT_SIERRA_REGION),
        "ucla_effective_native_cell_bounds": mask_info["bounds"],
        "ace_overlap_reconstructs_ucla_area_m2": float(ace.intersection_area_m2.sum()),
        "ucla_area_m2": float(mask_info["area_m2"]),
        "ucla_ace_relative_area_difference": float((ace.intersection_area_m2.sum() - mask_info["area_m2"]) / mask_info["area_m2"]),
        "wusd3_existing_cnn_label_area_m2": float(wus_summary["total_selected_area_m2"]),
        "wusd3_selected_center_bounds": {
            "lat_min": float(wus_summary["lat_min_selected"]),
            "lat_max": float(wus_summary["lat_max_selected"]),
            "lon_min": float(wus_summary["lon_min_selected"]),
            "lon_max": float(wus_summary["lon_max_selected"]),
        },
        "normalization_and_synthetic_field_tests": tests,
        "interpretation": (
            "All three products use a normalized native-cell-area-weighted mean SWE depth. "
            "UCLA and ACE represent the identical UCLA native-cell footprint by construction. "
            "WUS-D3 applies the same canonical coordinate bounds on its curvilinear d02 grid, "
            "so its selected centers and native area estimate differ slightly from the UCLA/ACE footprint."
        ),
        "compatibility_decision": (
            "The future ACE-to-Snow-17 scalar should be interpreted as the same type of target as the "
            "current CNN SWE_label: a regional area-weighted mean SWE depth, not a grid-cell map or total snow volume."
        ),
    }
    output = OUTPUT_DIR / "cross_grid_consistency_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
