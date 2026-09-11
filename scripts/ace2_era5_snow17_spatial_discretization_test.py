#!/usr/bin/env python3
"""Map the existing UCLA Sierra SWE mask onto native ACE2-ERA5 grid cells.

This is deliberately an overlap/discretization diagnostic only.  It does not
translate forcing and does not run Snow-17 or calculate SWE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.data import (  # noqa: E402
    DEFAULT_SIERRA_REGION,
    build_sierra_mask,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)

EARTH_RADIUS_M = 6_371_000.0
ACE_OUTPUT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/"
    "lag_ensemble_wy2016/member_20151101/autoregressive_predictions.nc"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ace2_era5_snow17_spatial_discretization_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-output", type=Path, default=ACE_OUTPUT)
    parser.add_argument("--water-year", type=int, default=2016)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def coordinate_bounds(centers: np.ndarray) -> np.ndarray:
    """Infer cell edges exactly as in the existing UCLA area-weight workflow."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Expected a one-dimensional coordinate with at least two values.")
    if not np.all(np.diff(centers) > 0.0):
        raise ValueError("Coordinates must be increasing before inferring bounds.")
    mids = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate(([centers[0] - (mids[0] - centers[0])], mids, [centers[-1] + (centers[-1] - mids[-1])]))


def spherical_rectangle_area(lat_south: float, lat_north: float, lon_west: float, lon_east: float) -> float:
    """Area on a sphere for a lon-lat rectangle; inputs are degrees."""
    if lat_north <= lat_south or lon_east <= lon_west:
        return 0.0
    return float(
        EARTH_RADIUS_M**2
        * (np.sin(np.deg2rad(lat_north)) - np.sin(np.deg2rad(lat_south)))
        * np.deg2rad(lon_east - lon_west)
    )


def overlap_interval(left_low: float, left_high: float, right_low: float, right_high: float) -> tuple[float, float] | None:
    low, high = max(left_low, right_low), min(left_high, right_high)
    return (low, high) if high > low else None


def load_ucla_mask(water_year: int) -> dict[str, object]:
    """Reuse the project's mask utility on its uncoarsened UCLA native grid."""
    swe_grid = get_regional_swe_grid_definition(
        water_year=water_year,
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )
    mask = build_sierra_mask(swe_grid, region=DEFAULT_SIERRA_REGION).astype(np.float64)
    lat = np.asarray(mask[swe_grid.latitude_name].values, dtype=np.float64)
    lon = np.asarray(mask[swe_grid.longitude_name].values, dtype=np.float64)
    mask_values = np.asarray(mask.values, dtype=np.float64)

    # The UCLA source stores latitude north-to-south.  Sort only for geometry
    # and plotting; the boolean selection remains the exact utility output.
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat = lat[lat_order]
    lon = lon[lon_order]
    mask_values = mask_values[np.ix_(lat_order, lon_order)]
    lat_edges, lon_edges = coordinate_bounds(lat), coordinate_bounds(lon)
    selected = mask_values > 0.0
    row_ids, col_ids = np.where(selected)
    if row_ids.size == 0:
        raise RuntimeError("The existing UCLA Sierra mask contains no selected cells.")

    mask_area = 0.0
    for row, col in zip(row_ids, col_ids, strict=True):
        mask_area += mask_values[row, col] * spherical_rectangle_area(
            lat_edges[row], lat_edges[row + 1], lon_edges[col], lon_edges[col + 1]
        )

    return {
        "mask": mask_values,
        "lat": lat,
        "lon": lon,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "selected_count": int(row_ids.size),
        "area_m2": float(mask_area),
        # The union is rectangular because the exact existing utility selects
        # every native UCLA cell whose center lies in DEFAULT_SIERRA_REGION.
        "bounds": {
            "lat_south": float(lat_edges[row_ids.min()]),
            "lat_north": float(lat_edges[row_ids.max() + 1]),
            "lon_west": float(lon_edges[col_ids.min()]),
            "lon_east": float(lon_edges[col_ids.max() + 1]),
        },
        "grid": swe_grid,
    }


def load_ace_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with xr.open_dataset(path, decode_times=False) as dataset:
        if "lat" not in dataset.coords or "lon" not in dataset.coords:
            raise ValueError(f"ACE output lacks expected lat/lon coordinates: {path}")
        lat = np.asarray(dataset.lat.values, dtype=np.float64)
        lon_0360 = np.asarray(dataset.lon.values, dtype=np.float64)
        lat_dims = list(dataset.lat.dims)
        lon_dims = list(dataset.lon.dims)
    # ACE output longitude is 0..360.  The Sierra is wholly away from the seam.
    lon = np.where(lon_0360 > 180.0, lon_0360 - 360.0, lon_0360)
    lat_order, lon_order = np.argsort(lat), np.argsort(lon)
    metadata = {
        "source": str(path),
        "latitude_dimension": lat_dims,
        "longitude_dimension": lon_dims,
        "shape": [int(lat.size), int(lon.size)],
        "source_longitude_convention": "0_to_360_degrees_east",
        "analysis_longitude_convention": "-180_to_180_degrees_east",
        "latitude_is_ascending_in_source": bool(np.all(np.diff(lat) > 0.0)),
        "latitude_cell_centers_source": [float(lat[0]), float(lat[-1])],
        "longitude_cell_centers_source": [float(lon_0360[0]), float(lon_0360[-1])],
        "median_latitude_spacing_degrees": float(np.median(np.diff(np.sort(lat)))),
        "longitude_spacing_degrees": float(np.median(np.diff(np.sort(lon)))),
    }
    return lat[lat_order], lon[lon_order], lat_order, lon_order, metadata


def build_overlap(
    mask_info: dict[str, object],
    ace_lat: np.ndarray,
    ace_lon: np.ndarray,
    ace_lat_source_indices: np.ndarray,
    ace_lon_source_indices: np.ndarray,
) -> pd.DataFrame:
    bounds = mask_info["bounds"]
    assert isinstance(bounds, dict)
    ace_lat_edges, ace_lon_edges = coordinate_bounds(ace_lat), coordinate_bounds(ace_lon)
    rows: list[dict[str, float | int]] = []
    for lat_idx, lat_center in enumerate(ace_lat):
        lat_overlap = overlap_interval(
            ace_lat_edges[lat_idx], ace_lat_edges[lat_idx + 1], bounds["lat_south"], bounds["lat_north"]
        )
        if lat_overlap is None:
            continue
        for lon_idx, lon_center in enumerate(ace_lon):
            lon_overlap = overlap_interval(
                ace_lon_edges[lon_idx], ace_lon_edges[lon_idx + 1], bounds["lon_west"], bounds["lon_east"]
            )
            if lon_overlap is None:
                continue
            ace_area = spherical_rectangle_area(
                ace_lat_edges[lat_idx], ace_lat_edges[lat_idx + 1], ace_lon_edges[lon_idx], ace_lon_edges[lon_idx + 1]
            )
            intersection = spherical_rectangle_area(
                lat_overlap[0], lat_overlap[1], lon_overlap[0], lon_overlap[1]
            )
            rows.append(
                {
                    "ace_lat_index": int(ace_lat_source_indices[lat_idx]),
                    "ace_lon_index": int(ace_lon_source_indices[lon_idx]),
                    "ace_lat": float(lat_center),
                    "ace_lon": float(lon_center),
                    "ace_lon_0_360": float(lon_center % 360.0),
                    "ace_cell_area_m2": ace_area,
                    "intersection_area_m2": intersection,
                    "overlap_fraction": intersection / ace_area,
                    "intersection_lat_south": lat_overlap[0],
                    "intersection_lat_north": lat_overlap[1],
                    "intersection_lon_west": lon_overlap[0],
                    "intersection_lon_east": lon_overlap[1],
                    "ace_lat_south": float(ace_lat_edges[lat_idx]),
                    "ace_lat_north": float(ace_lat_edges[lat_idx + 1]),
                    "ace_lon_west": float(ace_lon_edges[lon_idx]),
                    "ace_lon_east": float(ace_lon_edges[lon_idx + 1]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No native ACE cells intersect the existing UCLA Sierra mask.")
    frame["normalized_sierra_weight"] = frame["intersection_area_m2"] / frame["intersection_area_m2"].sum()
    return frame.sort_values("normalized_sierra_weight", ascending=False, ignore_index=True)


def draw_cell_boundaries(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        axis.add_patch(
            Rectangle(
                (row.ace_lon_west, row.ace_lat_south),
                row.ace_lon_east - row.ace_lon_west,
                row.ace_lat_north - row.ace_lat_south,
                fill=False,
                edgecolor="black",
                linewidth=0.85,
                zorder=5,
            )
        )


def plot_overlap(mask_info: dict[str, object], frame: pd.DataFrame, output: Path) -> None:
    mask = np.ma.masked_where(np.asarray(mask_info["mask"]) <= 0.0, np.asarray(mask_info["mask"]))
    lon_edges = np.asarray(mask_info["lon_edges"])
    lat_edges = np.asarray(mask_info["lat_edges"])
    bounds = mask_info["bounds"]
    assert isinstance(bounds, dict)
    margin = 0.35
    xlim = (bounds["lon_west"] - margin, bounds["lon_east"] + margin)
    ylim = (bounds["lat_south"] - margin, bounds["lat_north"] + margin)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), sharex=True, sharey=True, constrained_layout=True)
    for axis in axes:
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Longitude (degrees east)")
        axis.grid(False)
    axes[0].set_ylabel("Latitude (degrees north)")

    axes[0].pcolormesh(lon_edges, lat_edges, mask, shading="flat", cmap="Blues", vmin=0.0, vmax=1.0, rasterized=True)
    axes[0].set_title("UCLA Sierra SWE Mask")

    axes[1].pcolormesh(lon_edges, lat_edges, mask, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.25, rasterized=True)
    for row in frame.itertuples(index=False):
        axes[1].add_patch(
            Rectangle(
                (row.ace_lon_west, row.ace_lat_south),
                row.ace_lon_east - row.ace_lon_west,
                row.ace_lat_north - row.ace_lat_south,
                facecolor="#4C78A8",
                edgecolor="none",
                alpha=0.35,
                zorder=3,
            )
        )
    draw_cell_boundaries(axes[1], frame)
    axes[1].set_title("ACE Grid / Initial Snow-17 Units")

    axes[2].pcolormesh(lon_edges, lat_edges, mask, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.18, rasterized=True)
    norm = Normalize(vmin=float(frame.normalized_sierra_weight.min()), vmax=float(frame.normalized_sierra_weight.max()))
    cmap = plt.get_cmap("YlOrRd")
    for row in frame.itertuples(index=False):
        axes[2].add_patch(
            Rectangle(
                (row.intersection_lon_west, row.intersection_lat_south),
                row.intersection_lon_east - row.intersection_lon_west,
                row.intersection_lat_north - row.intersection_lat_south,
                facecolor=cmap(norm(row.normalized_sierra_weight)),
                edgecolor="none",
                alpha=0.90,
                zorder=3,
            )
        )
    draw_cell_boundaries(axes[2], frame)
    axes[2].set_title("UCLA-ACE Overlap and Sierra Weights")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[2], shrink=0.82, pad=0.03)
    colorbar.set_label("Normalized Sierra weight")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_info = load_ucla_mask(args.water_year)
    ace_lat, ace_lon, ace_lat_source_indices, ace_lon_source_indices, ace_metadata = load_ace_grid(args.ace_output)
    overlap = build_overlap(mask_info, ace_lat, ace_lon, ace_lat_source_indices, ace_lon_source_indices)

    table_path = args.output_dir / "ace_snow17_sierra_units.csv"
    figure_path = args.output_dir / "ucla_ace_snow17_spatial_overlap.png"
    summary_path = args.output_dir / "spatial_discretization_summary.json"
    report_path = args.output_dir / "README.md"
    overlap.to_csv(table_path, index=False, float_format="%.12g")
    plot_overlap(mask_info, overlap, figure_path)

    mask_area = float(mask_info["area_m2"])
    reconstructed_area = float(overlap.intersection_area_m2.sum())
    partial = int(np.count_nonzero(overlap.overlap_fraction.to_numpy() < (1.0 - 1.0e-12)))
    full = int(len(overlap) - partial)
    grid = mask_info["grid"]
    summary = {
        "purpose": "Spatial discretization only; Snow-17 and SWE calculations were not run.",
        "ucla_source_file": str(swe_file_for_water_year(args.water_year)),
        "ucla_variable": "SWE_Post",
        "ucla_mask_utility": "snow_ml.data.build_sierra_mask",
        "ucla_mask_region_definition": asdict(DEFAULT_SIERRA_REGION),
        "ucla_mask_dimensions": list(np.asarray(mask_info["mask"]).shape),
        "ucla_mask_latitude_spacing_degrees": float(np.median(np.diff(np.asarray(mask_info["lat"])))),
        "ucla_mask_longitude_spacing_degrees": float(np.median(np.diff(np.asarray(mask_info["lon"])))),
        "ucla_valid_sierra_cell_count": int(mask_info["selected_count"]),
        "ucla_mask_effective_bounds_from_native_cells": mask_info["bounds"],
        "ucla_mask_total_physical_area_m2": mask_area,
        "ucla_grid_coordinate_names": {"latitude": grid.latitude_name, "longitude": grid.longitude_name},
        "ace_grid": ace_metadata,
        "native_ace_intersecting_cell_count": int(len(overlap)),
        "fully_overlapping_ace_cell_count": full,
        "partially_overlapping_ace_cell_count": partial,
        "ace_sierra_intersection_area_m2": reconstructed_area,
        "ucla_minus_intersection_area_m2": mask_area - reconstructed_area,
        "relative_area_difference": (mask_area - reconstructed_area) / mask_area,
        "normalized_weight_min": float(overlap.normalized_sierra_weight.min()),
        "normalized_weight_max": float(overlap.normalized_sierra_weight.max()),
        "normalized_weight_sum": float(overlap.normalized_sierra_weight.sum()),
        "area_method": "Spherical lon-lat rectangles using R=6,371,000 m and coordinate bounds inferred from adjacent cell centers.",
        "outputs": {"units_table": str(table_path), "figure": str(figure_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        "# ACE2-ERA5 to Snow-17 spatial discretization test\n\n"
        "This artifact maps the existing UCLA Sierra SWE mask onto intersecting native ACE2-ERA5 cells. "
        "It does not run Snow-17 and does not calculate SWE.\n\n"
        f"- UCLA source: `{summary['ucla_source_file']}` (`SWE_Post`)\n"
        f"- Existing mask utility: `{summary['ucla_mask_utility']}`\n"
        f"- UCLA selected cells: `{summary['ucla_valid_sierra_cell_count']}`\n"
        f"- Native ACE/Snow-17 units: `{summary['native_ace_intersecting_cell_count']}`\n"
        f"- Weight sum: `{summary['normalized_weight_sum']:.15f}`\n"
        f"- Units table: `{table_path.name}`\n"
        f"- Figure: `{figure_path.name}`\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nSelected ACE/Snow-17 units (largest normalized Sierra weight first):")
    print(
        overlap[
            [
                "ace_lat_index",
                "ace_lon_index",
                "ace_lat",
                "ace_lon",
                "intersection_area_m2",
                "overlap_fraction",
                "normalized_sierra_weight",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
