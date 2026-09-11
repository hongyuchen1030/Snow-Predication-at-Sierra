#!/usr/bin/env python3
"""Audit the actual S0-17 CNN scalar-label footprint against UCLA/ACE overlap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_wusd3_swe_labels import build_mask_and_area  # noqa: E402
from snow_ml.data import DEFAULT_SIERRA_REGION, build_sierra_mask, get_regional_swe_grid_definition  # noqa: E402

OUTPUT = PROJECT_ROOT / "artifacts" / "cnn_swe_target_spatial_audit"
WUS_ID = "ec-earth3_r1i1p1f1_2_historical_bc"


def cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    midpoint = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - (midpoint[0] - centers[0])], midpoint, [centers[-1] + (centers[-1] - midpoint[-1])]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, wus_grid, wus_mask, wus_area, wus_summary = build_mask_and_area(WUS_ID)
    ucla_grid = get_regional_swe_grid_definition(
        water_year=2016,
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )
    ucla_mask = build_sierra_mask(ucla_grid, region=DEFAULT_SIERRA_REGION).astype(bool)
    ucla_lat = np.asarray(ucla_mask[ucla_grid.latitude_name].values, dtype=np.float64)
    ucla_lon = np.asarray(ucla_mask[ucla_grid.longitude_name].values, dtype=np.float64)
    ucla_values = np.asarray(ucla_mask.values, dtype=bool)
    if np.any(np.diff(ucla_lat) < 0):
        order = np.argsort(ucla_lat)
        ucla_lat, ucla_values = ucla_lat[order], ucla_values[order, :]
    if np.any(np.diff(ucla_lon) < 0):
        order = np.argsort(ucla_lon)
        ucla_lon, ucla_values = ucla_lon[order], ucla_values[:, order]
    ucla_lat_edges, ucla_lon_edges = cell_edges(ucla_lat), cell_edges(ucla_lon)
    ucla_area_m2 = 0.0
    # This uses the exact UCLA native selected cells, matching the ACE overlap script.
    for row, col in zip(*np.where(ucla_values), strict=True):
        south, north = ucla_lat_edges[row], ucla_lat_edges[row + 1]
        west, east = ucla_lon_edges[col], ucla_lon_edges[col + 1]
        ucla_area_m2 += (6_371_000.0**2) * (np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))) * np.deg2rad(east - west)

    # The scalar label multiplies the geographic selection by inferred native
    # area. Cells with undefined geometry have zero/undefined aggregation
    # weight and are not contributors to the target.
    wus_selected = np.asarray((wus_mask * wus_area).fillna(0.0).values > 0.0, dtype=bool)
    wus_lat = np.asarray(wus_grid.fine_latitude.values, dtype=np.float64)
    wus_lon = np.asarray(wus_grid.fine_longitude.values, dtype=np.float64)
    extent = (-122.7, -117.8, 34.8, 42.2)
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for axis in axes:
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("longitude (degrees east)")
    axes[0].set_ylabel("latitude (degrees north)")
    axes[0].scatter(wus_lon[wus_selected], wus_lat[wus_selected], s=1.0, color="#177e89", rasterized=True)
    axes[0].set_title(f"CNN label: {int(wus_selected.sum())} WUS-D3 d02 cells")
    axes[1].pcolormesh(ucla_lon_edges, ucla_lat_edges, ucla_values, shading="flat", cmap="Greys", vmin=0, vmax=1, rasterized=True)
    axes[1].set_title(f"UCLA/ACE: {int(ucla_values.sum())} UCLA cells")
    axes[2].pcolormesh(ucla_lon_edges, ucla_lat_edges, ucla_values, shading="flat", cmap="Greys", vmin=0, vmax=1, alpha=0.35, rasterized=True)
    axes[2].scatter(wus_lon[wus_selected], wus_lat[wus_selected], s=1.0, color="#177e89", rasterized=True)
    axes[2].set_title("Direct comparison: WUS-D3 points over UCLA cells")
    for axis in axes:
        axis.plot([-122.5, -118.0, -118.0, -122.5, -122.5], [35.0, 35.0, 42.0, 42.0, 35.0], color="#be3b30", linewidth=1.0)
    figure.savefig(OUTPUT / "cnn_wusd3_vs_ucla_ace_target_footprint.png", dpi=200)
    plt.close(figure)

    summary = {
        "decision": "MISMATCH",
        "cnn_target": "WUS-D3 d02 native selected-cell area-weighted mean April 1 snow depth",
        "cnn_selected_cells": int(wus_selected.sum()),
        "cnn_selected_area_m2": float(wus_summary["total_selected_area_m2"]),
        "cnn_selected_area_km2": float(wus_summary["total_selected_area_m2"]) / 1.0e6,
        "ucla_selected_cells": int(ucla_values.sum()),
        "ucla_selected_area_m2": float(ucla_area_m2),
        "ucla_selected_area_km2": float(ucla_area_m2) / 1.0e6,
        "area_difference_ucla_minus_cnn_km2": float((ucla_area_m2 - float(wus_summary["total_selected_area_m2"])) / 1.0e6),
        "cnn_fraction_of_ucla_area": float(wus_summary["total_selected_area_m2"]) / float(ucla_area_m2),
        "canonical_coordinate_box": {"lat_min": 35.0, "lat_max": 42.0, "lon_min": -122.5, "lon_max": -118.0},
        "cnn_selected_center_bounds": {
            "lat_min": float(wus_lat[wus_selected].min()), "lat_max": float(wus_lat[wus_selected].max()),
            "lon_min": float(wus_lon[wus_selected].min()), "lon_max": float(wus_lon[wus_selected].max()),
        },
        "ucla_native_cell_bounds": {
            "lat_south": float(ucla_lat_edges[np.where(ucla_values)[0].min()]), "lat_north": float(ucla_lat_edges[np.where(ucla_values)[0].max() + 1]),
            "lon_west": float(ucla_lon_edges[np.where(ucla_values)[1].min()]), "lon_east": float(ucla_lon_edges[np.where(ucla_values)[1].max() + 1]),
        },
        "note": "Both masks begin with the same coordinate limits but are not the same native physical footprint; no mountain, land, elevation, basin, or shapefile mask is applied to the CNN labels.",
    }
    (OUTPUT / "cnn_wusd3_vs_ucla_ace_target_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
