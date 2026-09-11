#!/usr/bin/env python3
"""Visualize COBE2 climatology coverage against the ACE2 Pacific overwrite mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
ACE2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2")
ACE2_ASSETS = ACE2_ROOT / "assets"
REPORT_DIR = REPO / "artifacts" / "ace2_era5_feasibility"

COBE2_SOURCE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.ltm.1991-2020.nc")
ACE2_GRID_CLIM = (
    ACE2_ASSETS
    / "climatology"
    / "pacific_anomaly_reversal_2020"
    / "cobe2_monthly_surface_temperature_climatology_on_ace2_grid.nc"
)
ACE2_FORCING = ACE2_ASSETS / "forcing" / "forcing_2020.nc"

PACIFIC_DOMAIN = {
    "lat_min": -10.0,
    "lat_max": 60.0,
    "lon_min_360": 120.0,
    "lon_max_360": 280.0,
    "ocean_fraction_threshold": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobe2-source", type=Path, default=COBE2_SOURCE)
    parser.add_argument("--ace2-grid-climatology", type=Path, default=ACE2_GRID_CLIM)
    parser.add_argument("--ace2-forcing", type=Path, default=ACE2_FORCING)
    parser.add_argument("--month", type=int, default=1, help="Calendar month to visualize, 1-12.")
    parser.add_argument("--png", type=Path, default=REPORT_DIR / "cobe2_vs_ace2_mask_january.png")
    parser.add_argument("--pdf", type=Path, default=REPORT_DIR / "cobe2_vs_ace2_mask_january.pdf")
    parser.add_argument("--metadata-json", type=Path, default=REPORT_DIR / "cobe2_vs_ace2_mask_january_metadata.json")
    parser.add_argument("--zoom-png", type=Path, default=REPORT_DIR / "cobe2_vs_ace2_mask_january_zoom.png")
    parser.add_argument("--zoom-pdf", type=Path, default=REPORT_DIR / "cobe2_vs_ace2_mask_january_zoom.pdf")
    parser.add_argument("--note-md", type=Path, default=REPORT_DIR / "cobe2_vs_ace2_mask_visual_note.md")
    return parser.parse_args()


def boundary_mask(mask2d: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask2d, dtype=bool)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        shifted = np.roll(mask2d, shift, axis=axis)
        boundary |= mask2d & ~shifted
    boundary[0, :] |= mask2d[0, :]
    boundary[-1, :] |= mask2d[-1, :]
    boundary[:, 0] |= mask2d[:, 0]
    boundary[:, -1] |= mask2d[:, -1]
    return boundary


def subset_2d(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    lon = da[lon_name]
    lat = da[lat_name]
    lon_mask = (lon >= PACIFIC_DOMAIN["lon_min_360"]) & (lon <= PACIFIC_DOMAIN["lon_max_360"])
    lat_mask = (lat >= PACIFIC_DOMAIN["lat_min"]) & (lat <= PACIFIC_DOMAIN["lat_max"])
    return da.sel({lon_name: lon[lon_mask], lat_name: lat[lat_mask]})


def select_source_month(ds: xr.Dataset, month: int) -> tuple[xr.DataArray, str]:
    time_values = ds["time"].values
    month_numbers = [int(getattr(v, "month", i + 1)) for i, v in enumerate(time_values)]
    if month not in month_numbers:
        raise ValueError(f"Month {month} not found in source climatology time coordinate: {month_numbers}")
    month_index = month_numbers.index(month)
    sst = ds["sst"].isel(time=month_index)
    return sst, f"source time index {month_index} -> month {month}"


def build_overwrite_mask(forcing_ds: xr.Dataset) -> xr.DataArray:
    lat_grid, lon_grid = xr.broadcast(forcing_ds["latitude"], forcing_ds["longitude"])
    domain_mask = (
        (lat_grid >= PACIFIC_DOMAIN["lat_min"])
        & (lat_grid <= PACIFIC_DOMAIN["lat_max"])
        & (lon_grid >= PACIFIC_DOMAIN["lon_min_360"])
        & (lon_grid <= PACIFIC_DOMAIN["lon_max_360"])
    ).transpose("latitude", "longitude")
    ocean_mask = forcing_ds["ocean_fraction"].isel(time=0) > PACIFIC_DOMAIN["ocean_fraction_threshold"]
    return domain_mask & ocean_mask


def add_mask_boundary(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, color: str, linewidth: float, label: str | None = None) -> None:
    ax.contour(lon, lat, mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)
    if label:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def plot_nan_points(ax: plt.Axes, lon2d: np.ndarray, lat2d: np.ndarray, nan_mask: np.ndarray, color: str, size: float, label: str) -> None:
    if nan_mask.any():
        ax.scatter(lon2d[nan_mask], lat2d[nan_mask], s=size, c=color, marker="s", linewidths=0, alpha=0.85, label=label)


def extent_from_mask(lon2d: np.ndarray, lat2d: np.ndarray, mask: np.ndarray, pad_lon: float = 8.0, pad_lat: float = 5.0) -> dict[str, float] | None:
    if not mask.any():
        return None
    lon_vals = lon2d[mask]
    lat_vals = lat2d[mask]
    return {
        "lon_min": float(max(PACIFIC_DOMAIN["lon_min_360"], np.nanmin(lon_vals) - pad_lon)),
        "lon_max": float(min(PACIFIC_DOMAIN["lon_max_360"], np.nanmax(lon_vals) + pad_lon)),
        "lat_min": float(max(PACIFIC_DOMAIN["lat_min"], np.nanmin(lat_vals) - pad_lat)),
        "lat_max": float(min(PACIFIC_DOMAIN["lat_max"], np.nanmax(lat_vals) + pad_lat)),
    }


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    best_component = np.zeros_like(mask, dtype=bool)
    best_size = 0
    nlat, nlon = mask.shape
    for start in zip(*np.where(mask), strict=False):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component_points: list[tuple[int, int]] = []
        while stack:
            i, j = stack.pop()
            component_points.append((i, j))
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < nlat and 0 <= nj < nlon and mask[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    stack.append((ni, nj))
        if len(component_points) > best_size:
            best_size = len(component_points)
            best_component = np.zeros_like(mask, dtype=bool)
            for i, j in component_points:
                best_component[i, j] = True
    return best_component


def main() -> None:
    args = parse_args()
    for path in (args.png, args.pdf, args.metadata_json, args.zoom_png, args.zoom_pdf, args.note_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.cobe2_source, decode_times=False) as source_ds, xr.open_dataset(
        args.ace2_grid_climatology
    ) as clim_ds, xr.open_dataset(args.ace2_forcing) as forcing_ds:
        source_sst, source_month_note = select_source_month(source_ds, args.month)
        source_subset = subset_2d(source_sst, "lat", "lon")
        source_lon = np.asarray(source_subset["lon"].values)
        source_lat = np.asarray(source_subset["lat"].values)
        source_lon2d, source_lat2d = np.meshgrid(source_lon, source_lat)
        source_values = np.asarray(source_subset.values, dtype=float)
        source_finite = np.isfinite(source_values)

        clim_subset = subset_2d(clim_ds["surface_temperature_climatology"].sel(month=args.month), "latitude", "longitude")
        clim_lon = np.asarray(clim_subset["longitude"].values)
        clim_lat = np.asarray(clim_subset["latitude"].values)
        clim_lon2d, clim_lat2d = np.meshgrid(clim_lon, clim_lat)
        clim_values = np.asarray(clim_subset.values, dtype=float)
        clim_finite = np.isfinite(clim_values)

        forcing_subset = subset_2d(forcing_ds["surface_temperature"].isel(time=0), "latitude", "longitude")
        overwrite_mask_full = build_overwrite_mask(forcing_ds)
        overwrite_mask_subset = subset_2d(overwrite_mask_full, "latitude", "longitude")
        overwrite_mask = np.asarray(overwrite_mask_subset.values, dtype=bool)

        finite_outside_overwrite = clim_finite & ~overwrite_mask
        overwrite_and_finite = overwrite_mask & clim_finite
        overwrite_and_nan = overwrite_mask & ~clim_finite

        mismatch = np.zeros_like(clim_values, dtype=np.int8)
        mismatch[overwrite_and_finite] = 1
        mismatch[overwrite_and_nan] = 2
        mismatch[finite_outside_overwrite] = 3

        common_finite = np.concatenate([source_values[source_finite], clim_values[clim_finite]])
        vmin = float(np.nanpercentile(common_finite, 2))
        vmax = float(np.nanpercentile(common_finite, 98))

        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8), constrained_layout=True)

        im0 = axes[0].pcolormesh(source_lon, source_lat, source_values, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
        plot_nan_points(axes[0], source_lon2d, source_lat2d, ~source_finite, color="black", size=6, label="COBE2 missing")
        add_mask_boundary(axes[0], source_lon2d, source_lat2d, source_finite, color="black", linewidth=1.0)
        axes[0].set_title("COBE2 January climatology\n(native grid)")
        axes[0].set_xlim(PACIFIC_DOMAIN["lon_min_360"], PACIFIC_DOMAIN["lon_max_360"])
        axes[0].set_ylim(PACIFIC_DOMAIN["lat_min"], PACIFIC_DOMAIN["lat_max"])
        axes[0].set_xlabel("Longitude (degE)")
        axes[0].set_ylabel("Latitude")
        fig.colorbar(im0, ax=axes[0], shrink=0.84, label="SST climatology (degC)")

        im1 = axes[1].pcolormesh(clim_lon, clim_lat, clim_values, cmap="coolwarm", shading="auto", vmin=vmin + 273.15, vmax=vmax + 273.15)
        plot_nan_points(axes[1], clim_lon2d, clim_lat2d, ~clim_finite, color="black", size=10, label="ACE2-grid climatology missing")
        add_mask_boundary(axes[1], clim_lon2d, clim_lat2d, overwrite_mask, color="lime", linewidth=1.2)
        axes[1].set_title("ACE2-grid January climatology\nwith overwrite mask")
        axes[1].set_xlim(PACIFIC_DOMAIN["lon_min_360"], PACIFIC_DOMAIN["lon_max_360"])
        axes[1].set_ylim(PACIFIC_DOMAIN["lat_min"], PACIFIC_DOMAIN["lat_max"])
        axes[1].set_xlabel("Longitude (degE)")
        axes[1].set_ylabel("Latitude")
        fig.colorbar(im1, ax=axes[1], shrink=0.84, label="ACE2-grid climatology (K)")

        mismatch_cmap = ListedColormap(["#f7f7f7", "#2ca25f", "#d73027", "#2b8cbe"])
        mismatch_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], mismatch_cmap.N)
        axes[2].pcolormesh(clim_lon, clim_lat, mismatch, cmap=mismatch_cmap, norm=mismatch_norm, shading="auto")
        add_mask_boundary(axes[2], clim_lon2d, clim_lat2d, overwrite_mask, color="black", linewidth=1.1)
        add_mask_boundary(axes[2], clim_lon2d, clim_lat2d, clim_finite, color="gold", linewidth=1.0)
        axes[2].set_title("Mismatch categories on ACE2 grid")
        axes[2].set_xlim(PACIFIC_DOMAIN["lon_min_360"], PACIFIC_DOMAIN["lon_max_360"])
        axes[2].set_ylim(PACIFIC_DOMAIN["lat_min"], PACIFIC_DOMAIN["lat_max"])
        axes[2].set_xlabel("Longitude (degE)")
        axes[2].set_ylabel("Latitude")
        axes[2].legend(
            handles=[
                Patch(facecolor="#2ca25f", edgecolor="none", label="ACE2 overwrite + finite COBE2"),
                Patch(facecolor="#d73027", edgecolor="none", label="ACE2 overwrite + NaN COBE2"),
                Patch(facecolor="#2b8cbe", edgecolor="none", label="Finite COBE2 outside overwrite"),
                Patch(facecolor="none", edgecolor="black", label="ACE2 overwrite boundary"),
                Patch(facecolor="none", edgecolor="gold", label="Finite COBE2 boundary"),
            ],
            loc="lower right",
            fontsize=8,
            frameon=True,
        )

        for ax in axes:
            ax.grid(True, linewidth=0.3, color="0.7", alpha=0.6)

        fig.savefig(args.png, dpi=220)
        fig.savefig(args.pdf)
        plt.close(fig)

        largest_bad_component = largest_connected_component(overwrite_and_nan) if overwrite_and_nan.any() else overwrite_and_nan
        zoom_extent = extent_from_mask(clim_lon2d, clim_lat2d, largest_bad_component, pad_lon=5.0, pad_lat=4.0)
        zoom_made = False
        if zoom_extent is not None:
            zoom_fig, zoom_axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
            zoom_axes[0].pcolormesh(clim_lon, clim_lat, clim_values, cmap="coolwarm", shading="auto")
            plot_nan_points(zoom_axes[0], clim_lon2d, clim_lat2d, overwrite_and_nan, color="#d73027", size=14, label="Overwrite + NaN")
            add_mask_boundary(zoom_axes[0], clim_lon2d, clim_lat2d, overwrite_mask, color="black", linewidth=1.1)
            zoom_axes[0].set_title("ACE2-grid climatology zoom")
            zoom_axes[1].pcolormesh(clim_lon, clim_lat, mismatch, cmap=mismatch_cmap, norm=mismatch_norm, shading="auto")
            add_mask_boundary(zoom_axes[1], clim_lon2d, clim_lat2d, overwrite_mask, color="black", linewidth=1.1)
            add_mask_boundary(zoom_axes[1], clim_lon2d, clim_lat2d, clim_finite, color="gold", linewidth=1.0)
            zoom_axes[1].set_title("Mismatch zoom")
            for ax in zoom_axes:
                ax.set_xlim(zoom_extent["lon_min"], zoom_extent["lon_max"])
                ax.set_ylim(zoom_extent["lat_min"], zoom_extent["lat_max"])
                ax.set_xlabel("Longitude (degE)")
                ax.set_ylabel("Latitude")
                ax.grid(True, linewidth=0.3, color="0.7", alpha=0.6)
            zoom_fig.savefig(args.zoom_png, dpi=220)
            zoom_fig.savefig(args.zoom_pdf)
            plt.close(zoom_fig)
            zoom_made = True

        bad_lon = clim_lon2d[overwrite_and_nan]
        bad_lat = clim_lat2d[overwrite_and_nan]
        metadata = {
            "cobe2_source_path": str(args.cobe2_source),
            "ace2_grid_climatology_path": str(args.ace2_grid_climatology),
            "ace2_forcing_path": str(args.ace2_forcing),
            "month": args.month,
            "month_name": "January" if args.month == 1 else f"month_{args.month}",
            "source_month_inference": source_month_note,
            "column_2_field": "ACE2-grid climatology",
            "pacific_domain": PACIFIC_DOMAIN,
            "counts": {
                "source_native_finite_count": int(source_finite.sum()),
                "source_native_nan_count": int((~source_finite).sum()),
                "ace2_grid_finite_count": int(clim_finite.sum()),
                "ace2_grid_nan_count": int((~clim_finite).sum()),
                "overwrite_mask_count": int(overwrite_mask.sum()),
                "overwrite_and_finite_count": int(overwrite_and_finite.sum()),
                "overwrite_and_nan_count": int(overwrite_and_nan.sum()),
                "finite_outside_overwrite_count": int(finite_outside_overwrite.sum()),
                "largest_bad_component_count": int(largest_bad_component.sum()),
            },
            "main_bad_region_bbox": (
                {
                    "lon_min": float(np.nanmin(bad_lon)),
                    "lon_max": float(np.nanmax(bad_lon)),
                    "lat_min": float(np.nanmin(bad_lat)),
                    "lat_max": float(np.nanmax(bad_lat)),
                }
                if bad_lon.size
                else None
            ),
            "largest_bad_component_bbox": extent_from_mask(clim_lon2d, clim_lat2d, largest_bad_component, pad_lon=0.0, pad_lat=0.0),
            "zoom_figure_created": zoom_made,
            "zoom_extent": zoom_extent,
        }
        args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        bad_region_note = "No ACE2 overwrite + NaN region found."
        if metadata["main_bad_region_bbox"] is not None:
            bbox = metadata["main_bad_region_bbox"]
            bad_region_note = (
                f"Main bad region spans roughly {bbox['lat_min']:.1f} to {bbox['lat_max']:.1f} deg latitude "
                f"and {bbox['lon_min']:.1f} to {bbox['lon_max']:.1f} deg longitude."
            )

        note_lines = [
            "# COBE2 vs ACE2 Mask Visual Note",
            "",
            f"- COBE2 file used: `{args.cobe2_source}`",
            f"- ACE2 file used: `{args.ace2_forcing}`",
            f"- ACE2-grid climatology used: `{args.ace2_grid_climatology}`",
            f"- Month plotted: `{metadata['month_name']}` ({source_month_note})",
            "- Column 2 used the ACE2-grid climatology, not the raw ACE2 actual surface temperature field.",
            f"- {bad_region_note}",
        ]
        if metadata["main_bad_region_bbox"] is not None:
            largest_bbox = metadata["largest_bad_component_bbox"]
            location_note = "coastlines and mask edges"
            note_lines.append(
                f"- The bad cells are concentrated mainly near {location_note}. The largest connected patch is around "
                f"{largest_bbox['lat_min']:.1f} to {largest_bbox['lat_max']:.1f} deg latitude and "
                f"{largest_bbox['lon_min']:.1f} to {largest_bbox['lon_max']:.1f} deg longitude."
            )
        else:
            note_lines.append("- No bad overwrite cells were found in January.")
        args.note_md.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
