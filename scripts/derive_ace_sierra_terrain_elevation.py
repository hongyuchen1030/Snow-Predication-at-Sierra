#!/usr/bin/env python3
"""Derive documented terrain elevations for the frozen 48 UCLA-ACE overlaps.

``z_sierra_m`` is an area-weighted mean of the existing USGS NED 1/3-arc-
second DEM within each exact rectangular UCLA-mask/ACE-cell intersection.
``z_ace_hgtsfc_m`` is the ACE forcing static HGTsfc value at that native cell.
Neither quantity is inferred from surface pressure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
CELLS = ACE_ROOT / "snow17_inputs/ace_snow17_sierra_units.csv"
FORCING = ACE_ROOT / "forcing_2015_2016/forcing_2015.nc"
DEM_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/swe_target_spatial_diagnostic/external_data/usgs_dem")
OUTPUT = PROJECT_ROOT / "artifacts/ace2_era5_snow17_scientific_validity_audit"
GRAVITY = 9.80665


def dem_mean_in_rectangle(path: Path, west: float, south: float, east: float, north: float) -> tuple[float, int]:
    """Return a cos(latitude)-weighted DEM mean on a 0.004-degree analysis mesh."""
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        clipped_west, clipped_east = max(west, left), min(east, right)
        clipped_south, clipped_north = max(south, bottom), min(north, top)
        if clipped_west >= clipped_east or clipped_south >= clipped_north:
            return np.nan, 0
        window = from_bounds(clipped_west, clipped_south, clipped_east, clipped_north, src.transform)
        # This retained mesh is about 0.4 km at Sierra latitudes: much finer
        # than ACE but avoids materializing the 19 GB DEM mosaic.
        height = max(1, int(np.ceil((clipped_north - clipped_south) / 0.004)))
        width = max(1, int(np.ceil((clipped_east - clipped_west) / 0.004)))
        data = src.read(1, window=window, out_shape=(height, width), resampling=Resampling.average, masked=True)
        transform = src.window_transform(window) * Affine.scale(window.width / width, window.height / height)
        rows = np.arange(height)
        latitudes = rasterio.transform.xy(transform, rows, np.zeros(height), offset="center")[1]
        valid = ~np.ma.getmaskarray(data)
        values = np.asarray(data.filled(np.nan), dtype=np.float64)
        weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))[:, None] * valid
        if not np.any(valid):
            return np.nan, 0
        return float(np.nansum(values * weights) / np.nansum(weights)), int(valid.sum())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(CELLS)
    dem_files = sorted(DEM_DIR.glob("*.tif"))
    if len(cells) != 48 or not dem_files:
        raise RuntimeError("Expected the frozen 48-cell table and cached USGS DEM tiles.")
    with xr.open_dataset(FORCING) as ds:
        hgt = ds.HGTsfc.isel(
            latitude=xr.DataArray(cells.ace_lat_index.to_numpy(dtype=np.int64), dims="cell"),
            longitude=xr.DataArray(cells.ace_lon_index.to_numpy(dtype=np.int64), dims="cell"),
        ).load().values.astype(np.float64)
        hgt_attrs = {key: str(value) for key, value in ds.HGTsfc.attrs.items()}

    results = []
    for row in cells.itertuples(index=False):
        numerator = 0.0
        count = 0
        for dem in dem_files:
            value, n = dem_mean_in_rectangle(
                dem,
                row.intersection_lon_west,
                row.intersection_lat_south,
                row.intersection_lon_east,
                row.intersection_lat_north,
            )
            if n:
                numerator += value * n
                count += n
        if not count:
            raise RuntimeError(f"No DEM data for ACE cell ({row.ace_lat}, {row.ace_lon}).")
        results.append((numerator / count, count))

    terrain = cells.copy()
    terrain["z_sierra_m"] = [value for value, _ in results]
    terrain["dem_sample_count"] = [count for _, count in results]
    terrain["z_ace_hgtsfc_m"] = hgt
    terrain["sierra_minus_ace_elevation_m"] = terrain.z_sierra_m - terrain.z_ace_hgtsfc_m
    terrain.to_csv(OUTPUT / "ace48_sierra_intersection_terrain.csv", index=False)
    summary = {
        "z_sierra_definition": "cos(latitude)-weighted USGS NED 1/3 arc-second DEM mean within the frozen UCLA-mask/ACE-cell intersection",
        "z_sierra_dem_source": "USGS The National Map, National Elevation Dataset (NED) 1/3 arc-second GeoTIFF tiles cached by the existing project spatial diagnostic",
        "dem_tile_count": len(dem_files),
        "effective_analysis_resolution_degrees": 0.004,
        "z_ace_definition": "HGTsfc static forcing value at the native ACE cell; field comes from ERA5 atmospheric invariant topography",
        "z_ace_hgtsfc_attributes": hgt_attrs,
        "z_sierra_m_range": [float(terrain.z_sierra_m.min()), float(terrain.z_sierra_m.max())],
        "z_ace_hgtsfc_m_range": [float(terrain.z_ace_hgtsfc_m.min()), float(terrain.z_ace_hgtsfc_m.max())],
        "sierra_minus_ace_elevation_m_range": [float(terrain.sierra_minus_ace_elevation_m.min()), float(terrain.sierra_minus_ace_elevation_m.max())],
        "sierra_weighted_z_sierra_m": float(np.dot(terrain.z_sierra_m, terrain.normalized_sierra_weight)),
        "sierra_weighted_z_ace_hgtsfc_m": float(np.dot(terrain.z_ace_hgtsfc_m, terrain.normalized_sierra_weight)),
        "warning": "The generic 6.5 K/km lapse rate is not applied or endorsed by this artifact.",
    }
    (OUTPUT / "ace48_sierra_intersection_terrain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
