#!/usr/bin/env python3
"""
Three-panel Sierra SWE spatial diagnostic.

Panels:
1. Raw April 1 SWE anomaly/climatology on the Sierra SWE grid.
2. Raw USGS DEM elevation over the Sierra diagnostic box.
3. SWE field overlaid on the DEM terrain map.

This script uses:
- existing UCLA/SWE accessors from snow_ml.data
- USGS 3DEP / The National Map DEM GeoTIFFs
- no geopandas
- no fiona
- no pyogrio
"""

from __future__ import annotations

import json
import math
import os
import resource
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VENDOR = PROJECT_ROOT / ".python_vendor"
SRC_ROOT = PROJECT_ROOT / "src"

for candidate in (PYTHON_VENDOR, PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from matplotlib.colors import LightSource
from PIL import Image
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT

from snow_ml.data import (
    DEFAULT_SIERRA_REGION,
    SWE_MISSING_VALUE,
    SWE_VARIABLE,
    build_sierra_mask,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_target_spatial_diagnostic"

# Keep large external DEM data on pscratch, not home.
SCRATCH_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "swe_target_spatial_diagnostic"
)
EXTERNAL_DATA_DIR = SCRATCH_ROOT / "external_data"
DEM_DIR = EXTERNAL_DATA_DIR / "usgs_dem"

FIG_PNG = OUTPUT_DIR / "swe_dem_four_panel_diagnostic.png"
FIG_PDF = OUTPUT_DIR / "swe_dem_four_panel_diagnostic.pdf"
META_JSON = OUTPUT_DIR / "swe_dem_four_panel_diagnostic_metadata.json"
RUN_LOG = OUTPUT_DIR / "swe_dem_four_panel_diagnostic.log"

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)

MEAN_STAT_INDEX = 0
NETCDF_ENGINE = "netcdf4"
REQUEST_TIMEOUT = 120
MAX_DEM_PLOT_DIM = 2200

# Correct Sierra SWE diagnostic box:
# 122.5W--118W, 35N--42N
PLOT_EXTENT = (-122.5, -118.0, 35.0, 42.0)
DEM_API_BBOX = "-122.5,35.0,-118.0,42.0"

DEM_API_DATASET_NAMES = [
    "Digital Elevation Model (DEM) 1/3 arc-second",
    "Digital Elevation Model (DEM) 1 arc-second",
    "National Elevation Dataset (NED) 1/3 arc-second",
    "National Elevation Dataset (NED) 1 arc-second",
]

DEM_API_QUERY_TEMPLATE = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    "?datasets={dataset}"
    f"&bbox={DEM_API_BBOX}"
    "&prodFormats=GeoTIFF"
    "&outputFormat=JSON"
    "&max=500"
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, SCRATCH_ROOT, EXTERNAL_DATA_DIR, DEM_DIR):
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(message, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def format_crs(crs: Any) -> str:
    if crs is None:
        return "unknown"
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def sanitize_name(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()


def convert_lon_to_minus180_180(lon_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))

    if lon_max > 180.0:
        converted = ((lon_values + 180.0) % 360.0) - 180.0
        order = np.argsort(converted)
        return converted[order], order, "0_to_360", "-180_to_180"

    order = np.argsort(lon_values)
    return lon_values[order], order, "-180_to_180", "-180_to_180"


def collapse_historical_dem_urls(download_urls: Sequence[str]) -> List[str]:
    latest_by_tile: Dict[str, tuple[str, str]] = {}

    for url in download_urls:
        filename = Path(url.split("?")[0]).name
        stem = Path(filename).stem
        parts = stem.split("_")

        tile_key = stem
        version_key = ""

        if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
            tile_key = "_".join(parts[:-1])
            version_key = parts[-1]

        previous = latest_by_tile.get(tile_key)
        if previous is None or version_key > previous[0]:
            latest_by_tile[tile_key] = (version_key, url)

    return sorted(value[1] for value in latest_by_tile.values())


# ---------------------------------------------------------------------
# SWE loading
# ---------------------------------------------------------------------

def load_spatial_apr1_fields() -> Dict[str, Any]:
    log("Loading raw April 1 SWE fields from snow_ml.data")

    swe_grid = get_regional_swe_grid_definition(
        water_year=WATER_YEAR_START,
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )

    current_mask = build_sierra_mask(swe_grid, region=DEFAULT_SIERRA_REGION).astype(np.float32)

    lat_name = swe_grid.latitude_name
    lon_name = swe_grid.longitude_name
    time_name = swe_grid.time_name

    lat = np.asarray(swe_grid.latitude.values, dtype=np.float64)
    raw_lon = np.asarray(swe_grid.longitude.values, dtype=np.float64)
    lon, lon_order, lon_before, lon_after = convert_lon_to_minus180_180(raw_lon)

    fields: List[np.ndarray] = []
    sample_path = None
    sample_time_start = None
    sample_time_end = None

    for water_year in WATER_YEARS:
        path = swe_file_for_water_year(int(water_year))

        if sample_path is None:
            sample_path = str(path)

        with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
            if sample_time_start is None:
                sample_time_start = str(np.datetime_as_string(ds[time_name].values[0], unit="D"))
                sample_time_end = str(np.datetime_as_string(ds[time_name].values[-1], unit="D"))

            swe = ds[SWE_VARIABLE].isel(Stats=MEAN_STAT_INDEX, drop=True)
            swe = swe.sel(time=np.datetime64(f"{int(water_year)}-04-01"))
            swe = swe.sel({lat_name: swe_grid.latitude, lon_name: swe_grid.longitude}).load()

        values = np.asarray(swe.where(swe != SWE_MISSING_VALUE).values, dtype=np.float32)
        values = values[:, lon_order]
        fields.append(values)

        log(f"Loaded April 1 SWE grid for WY{int(water_year)}")

    cube = np.stack(fields, axis=0)

    # Raw April 1 SWE climatology.
    swe_climatology = np.nanmean(cube, axis=0).astype(np.float32)

    # April 1 SWE anomaly for each year relative to the 1985--2021 April 1 climatology.
    anomaly_cube = cube - swe_climatology[None, :, :]

    # Mean anomaly over all years should be close to zero by construction.
    # For visualization, mean absolute anomaly is more informative than the signed mean anomaly.
    mean_anomaly = np.nanmean(anomaly_cube, axis=0).astype(np.float32)
    mean_abs_anomaly = np.nanmean(np.abs(anomaly_cube), axis=0).astype(np.float32)

    valid_any = np.any(np.isfinite(cube), axis=0).astype(np.float32)
    valid_fraction = np.mean(np.isfinite(cube), axis=0).astype(np.float32)

    current_mask_values = np.asarray(current_mask.values, dtype=np.float32)[:, lon_order]

    return {
        "lat": lat,
        "lon": lon,
        "cube": cube,
        "swe_climatology": swe_climatology,
        "anomaly_cube": anomaly_cube,
        "mean_anomaly": mean_anomaly,
        "mean_abs_anomaly": mean_abs_anomaly,
        "valid_any": valid_any,
        "valid_fraction": valid_fraction,
        "current_mask": current_mask_values,
        "raw_spatial_source_template": str(swe_file_for_water_year(WATER_YEAR_START)).replace(
            str(WATER_YEAR_START), "{water_year}"
        ),
        "raw_spatial_source_example_file": sample_path,
        "raw_variable_name": SWE_VARIABLE,
        "raw_coordinate_names": {
            "time": time_name,
            "stat": "Stats",
            "latitude": lat_name,
            "longitude": lon_name,
        },
        "raw_time_span_for_one_file": {
            "start": sample_time_start,
            "end": sample_time_end,
        },
        "longitude_convention_before": lon_before,
        "longitude_convention_after": lon_after,
        "mask_type": str(current_mask.attrs.get("mask_type", "unknown")),
    }


# ---------------------------------------------------------------------
# DEM API / download / mosaic
# ---------------------------------------------------------------------

def query_dem_downloads() -> Dict[str, Any]:
    log("Starting DEM API search")

    selected_dataset = None
    selected_query_url = None
    selected_items: List[Dict[str, Any]] = []

    for dataset_name in DEM_API_DATASET_NAMES:
        params = {
            "datasets": dataset_name,
            "bbox": DEM_API_BBOX,
            "prodFormats": "GeoTIFF",
            "outputFormat": "JSON",
            "max": "500",
        }

        query_url = DEM_API_QUERY_TEMPLATE.format(
            dataset=requests.utils.quote(dataset_name, safe="")
        )

        log(f"Querying DEM API dataset: {dataset_name}")
        log(f"Query URL: {query_url}")

        try:
            response = requests.get(
                "https://tnmaccess.nationalmap.gov/api/v1/products",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            log(f"DEM API query failed for dataset '{dataset_name}': {exc!r}")
            continue

        items = payload.get("items", []) or []
        log(f"DEM API dataset '{dataset_name}' returned {len(items)} item(s)")

        if items:
            selected_dataset = dataset_name
            selected_query_url = query_url
            selected_items = items
            break

    if not selected_items or selected_dataset is None or selected_query_url is None:
        raise RuntimeError("No DEM products found from The National Map API for the requested Sierra bbox.")

    download_urls = []
    for item in selected_items:
        download_url = item.get("downloadURL")
        if download_url:
            download_urls.append(download_url)

    download_urls = sorted(set(download_urls))
    collapsed_urls = collapse_historical_dem_urls(download_urls)

    if len(collapsed_urls) != len(download_urls):
        log(
            "Collapsed historical DEM versions from "
            f"{len(download_urls)} URLs to {len(collapsed_urls)} latest tile URL(s)"
        )

    download_urls = collapsed_urls

    if not download_urls:
        raise RuntimeError("DEM API returned products but no downloadURL fields were present.")

    log(f"Using {len(download_urls)} DEM download URL(s)")

    return {
        "dataset_name": selected_dataset,
        "query_url": selected_query_url,
        "download_urls": download_urls,
    }


def download_file(url: str, destination: Path) -> Path:
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    log(f"Downloading: {url}")
    log(f"Destination: {destination}")

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    tmp_path.replace(destination)
    return destination


def ensure_download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        log(f"Reusing existing download: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    return download_file(url, destination)


def extract_geotiffs_from_archive(archive_path: Path, extract_dir: Path) -> List[Path]:
    geotiffs: List[Path] = []
    suffix = archive_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        return [archive_path]

    if suffix != ".zip":
        raise RuntimeError(f"Unsupported DEM download format: {archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith((".tif", ".tiff")):
                target = extract_dir / Path(member).name

                if not target.exists() or target.stat().st_size == 0:
                    with zf.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

                geotiffs.append(target)

    if not geotiffs:
        raise RuntimeError(f"No GeoTIFF files found inside DEM archive: {archive_path}")

    return geotiffs


def build_dem_mosaic(dem_info: Dict[str, Any]) -> Dict[str, Any]:
    log(f"Building DEM mosaic from {len(dem_info['download_urls'])} download URL(s)")

    geotiff_paths: List[Path] = []
    archive_paths: List[Path] = []

    for idx, url in enumerate(dem_info["download_urls"], start=1):
        filename = Path(url.split("?")[0]).name
        if not filename:
            filename = f"dem_tile_{idx}.zip"

        destination = DEM_DIR / filename
        archive_path = ensure_download(url, destination)
        archive_paths.append(archive_path)

        extract_dir = DEM_DIR / f"extracted_{sanitize_name(archive_path.stem)}"
        geotiff_paths.extend(extract_geotiffs_from_archive(archive_path, extract_dir))

    geotiff_paths = sorted(set(geotiff_paths))
    log(f"Found {len(geotiff_paths)} GeoTIFF(s)")

    sources = []
    vrts = []

    try:
        for path in geotiff_paths:
            src = rasterio.open(path)
            sources.append(src)
            vrts.append(WarpedVRT(src, crs="EPSG:4326"))

        west, east, south, north = PLOT_EXTENT

        log("Merging DEM tiles in EPSG:4326")
        mosaic, transform = merge(
            vrts,
            bounds=(west, south, east, north),
            nodata=np.nan,
        )

        dem = mosaic[0].astype(np.float32)
        dem[~np.isfinite(dem)] = np.nan

        west_b, south_b, east_b, north_b = array_bounds(
            dem.shape[0],
            dem.shape[1],
            transform,
        )

        finite = np.isfinite(dem)
        if not finite.any():
            raise RuntimeError("DEM mosaic contains no finite values.")

        dx = abs(transform.a)
        dy = abs(transform.e)

        dem_filled = np.nan_to_num(dem, nan=float(np.nanmedian(dem[finite])))

        light = LightSource(azdeg=315, altdeg=45)
        hillshade = light.hillshade(dem_filled, dx=dx, dy=dy)
        hillshade[~finite] = np.nan

        plot_step = max(1, math.ceil(max(dem.shape) / MAX_DEM_PLOT_DIM))

        if plot_step > 1:
            log(f"Downsampling DEM mosaic for plotting with step={plot_step}")
            dem = dem[::plot_step, ::plot_step]
            hillshade = hillshade[::plot_step, ::plot_step]

        return {
            "dem": dem,
            "hillshade": hillshade.astype(np.float32),
            "extent": (west_b, east_b, south_b, north_b),
            "download_urls": dem_info["download_urls"],
            "query_url": dem_info["query_url"],
            "dataset_name": dem_info["dataset_name"],
            "archive_paths": [str(path) for path in archive_paths],
            "geotiff_paths": [str(path) for path in geotiff_paths],
            "source_crs_list": [format_crs(src.crs) for src in sources],
            "plot_crs": "EPSG:4326",
            "plot_step": plot_step,
            "dem_min_m": float(np.nanmin(dem)),
            "dem_max_m": float(np.nanmax(dem)),
            "dem_mean_m": float(np.nanmean(dem)),
        }

    finally:
        for vrt in vrts:
            vrt.close()
        for src in sources:
            src.close()


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def draw_region_box(ax: Any) -> None:
    west = DEFAULT_SIERRA_REGION.lon_min
    east = DEFAULT_SIERRA_REGION.lon_max
    south = DEFAULT_SIERRA_REGION.lat_min
    north = DEFAULT_SIERRA_REGION.lat_max

    xs = [west, east, east, west, west]
    ys = [south, south, north, north, south]

    ax.plot(xs, ys, color="k", linewidth=1.0, linestyle="--", zorder=20)


def plot_four_panel(swe: Dict[str, Any], dem: Dict[str, Any]) -> None:
    log("Rendering four-panel SWE/DEM diagnostic figure")

    lon = swe["lon"]
    lat = swe["lat"]

    # For spatial-footprint checking, raw climatological SWE is more useful than
    # mean anomaly, because the signed mean anomaly over all years is near zero.
    swe_field = swe["swe_climatology"]
    valid_any = swe["valid_any"]

    elevation = dem["dem"]
    hillshade = dem["hillshade"]
    extent = dem["extent"]

    lon2d, lat2d = np.meshgrid(lon, lat)

    # DEM coordinate grid for contours.
    rows, cols = elevation.shape
    dem_lon = np.linspace(extent[0], extent[1], cols)
    dem_lat = np.linspace(extent[3], extent[2], rows)
    dem_lon2d, dem_lat2d = np.meshgrid(dem_lon, dem_lat)

    # SWE-positive mask.
    # 0.05 m is only used to avoid drawing tiny numerical noise.
    swe_positive = np.where(np.isfinite(swe_field) & (swe_field > 0.05), 1.0, np.nan)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(26.0, 7.2),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Panel 1: SWE grid only
    # ------------------------------------------------------------
    ax0 = axes[0]
    mesh0 = ax0.pcolormesh(
        lon,
        lat,
        swe_field,
        shading="auto",
        cmap="Blues",
        rasterized=True,
    )
    ax0.contour(
        lon2d,
        lat2d,
        valid_any,
        levels=[0.5],
        colors="black",
        linewidths=1.2,
    )
    draw_region_box(ax0)

    ax0.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax0.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_title(
        "1. April 1 SWE climatology\n"
        "on raw Sierra SWE grid"
    )
    ax0.grid(True, linewidth=0.35, linestyle=":", alpha=0.55)
    cbar0 = fig.colorbar(mesh0, ax=ax0, shrink=0.82)
    cbar0.set_label("April 1 SWE climatology (m)")

    # ------------------------------------------------------------
    # Panel 2: DEM only, colored terrain
    # ------------------------------------------------------------
    ax1 = axes[1]
    im1 = ax1.imshow(
        elevation,
        extent=extent,
        origin="upper",
        cmap="terrain",
        vmin=np.nanpercentile(elevation, 1),
        vmax=np.nanpercentile(elevation, 99),
    )
    draw_region_box(ax1)

    ax1.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax1.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title(
        "2. Raw USGS DEM elevation\n"
        "Sierra Nevada diagnostic box"
    )
    ax1.grid(True, linewidth=0.35, linestyle=":", alpha=0.55)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.82)
    cbar1.set_label("Elevation (m)")

    # ------------------------------------------------------------
    # Panel 3: readable overlay
    # Grayscale hillshade + high-elevation contours + SWE footprint.
    # ------------------------------------------------------------
    ax2 = axes[2]

    # Gray hillshade only; no terrain colors here.
    ax2.imshow(
        hillshade,
        extent=extent,
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=1.0,
    )

    # DEM elevation contours to make the mountain belt visible.
    elev_levels = [1500, 2000, 2500, 3000]
    cs_dem = ax2.contour(
        dem_lon2d,
        dem_lat2d,
        elevation,
        levels=elev_levels,
        colors="0.35",
        linewidths=0.65,
    )
    ax2.clabel(cs_dem, inline=True, fontsize=6, fmt="%d m")

    # SWE footprint as strong warm-color transparent mask.
    ax2.pcolormesh(
        lon,
        lat,
        swe_positive,
        shading="auto",
        cmap="autumn",
        alpha=0.45,
        rasterized=True,
    )

    # Valid SWE footprint boundary.
    ax2.contour(
        lon2d,
        lat2d,
        valid_any,
        levels=[0.5],
        colors="black",
        linewidths=1.3,
    )

    # SWE magnitude contours.
    swe_levels = [0.5, 1.0, 2.0, 3.0]
    cs_swe = ax2.contour(
        lon2d,
        lat2d,
        swe_field,
        levels=swe_levels,
        colors="red",
        linewidths=0.9,
    )
    ax2.clabel(cs_swe, inline=True, fontsize=7, fmt="%.1f m")

    draw_region_box(ax2)

    ax2.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax2.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title(
        "3. SWE footprint over DEM hillshade\n"
        "black = valid cells, red = SWE contours"
    )
    ax2.grid(True, linewidth=0.35, linestyle=":", alpha=0.55)

    # ------------------------------------------------------------
    # Panel 4: binary diagnostic
    # White background + DEM elevation contours + SWE-valid mask.
    # This is the cleanest footprint-verification panel.
    # ------------------------------------------------------------
    ax3 = axes[3]

    # Draw high-elevation DEM contours first.
    cs_dem2 = ax3.contour(
        dem_lon2d,
        dem_lat2d,
        elevation,
        levels=[1500, 2000, 2500, 3000],
        colors="0.55",
        linewidths=0.75,
    )
    ax3.clabel(cs_dem2, inline=True, fontsize=6, fmt="%d m")

    # Binary SWE-positive cells.
    ax3.pcolormesh(
        lon,
        lat,
        swe_positive,
        shading="auto",
        cmap="autumn",
        alpha=0.75,
        rasterized=True,
    )

    # Valid SWE footprint boundary.
    ax3.contour(
        lon2d,
        lat2d,
        valid_any,
        levels=[0.5],
        colors="black",
        linewidths=1.5,
    )

    # Optional SWE magnitude contours.
    cs_swe2 = ax3.contour(
        lon2d,
        lat2d,
        swe_field,
        levels=[0.5, 1.0, 2.0, 3.0],
        colors="red",
        linewidths=0.9,
    )
    ax3.clabel(cs_swe2, inline=True, fontsize=7, fmt="%.1f m")

    draw_region_box(ax3)

    ax3.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax3.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title(
        "4. Binary SWE footprint check\n"
        "warm color = SWE-positive cells"
    )
    ax3.grid(True, linewidth=0.35, linestyle=":", alpha=0.55)

    fig.suptitle(
        "Sierra SWE target spatial diagnostic: SWE grid, DEM terrain, readable overlay, and binary footprint",
        fontsize=15,
    )

    fig.savefig(FIG_PNG, dpi=220)
    Image.open(FIG_PNG).convert("RGB").save(FIG_PDF, "PDF", resolution=220.0)
    plt.close(fig)

    log(f"Wrote PNG: {FIG_PNG}")
    log(f"Wrote PDF: {FIG_PDF}")


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def write_metadata(swe: Dict[str, Any], dem: Dict[str, Any], runtime_seconds: float) -> None:
    payload = {
        "purpose": "Three-panel diagnostic comparing raw Sierra SWE grid with USGS DEM terrain context.",
        "plot_extent_lon_lat": {
            "lon_min": PLOT_EXTENT[0],
            "lon_max": PLOT_EXTENT[1],
            "lat_min": PLOT_EXTENT[2],
            "lat_max": PLOT_EXTENT[3],
        },
        "swe_source_file_path_template": swe["raw_spatial_source_template"],
        "swe_source_example_file": swe["raw_spatial_source_example_file"],
        "swe_variable_name": swe["raw_variable_name"],
        "swe_coordinate_names": swe["raw_coordinate_names"],
        "swe_time_water_year_range": {
            "water_year_start": int(WATER_YEAR_START),
            "water_year_end": int(WATER_YEAR_END),
            "selected_target_dates": [f"{int(value)}-04-01" for value in WATER_YEARS.tolist()],
        },
        "plotted_swe_field": "April 1 SWE climatology over WY1985--WY2021",
        "note_on_anomaly": (
            "The script also computes April 1 SWE anomalies relative to the WY1985--WY2021 "
            "April 1 climatology. The signed mean anomaly over all years is near zero by "
            "construction, so the figure uses raw SWE climatology for spatial footprint checking."
        ),
        "swe_longitude_convention_before": swe["longitude_convention_before"],
        "swe_longitude_convention_after": swe["longitude_convention_after"],
        "dem_api_bbox": DEM_API_BBOX,
        "dem_api_query_url": dem["query_url"],
        "dem_dataset_name_used": dem["dataset_name"],
        "dem_download_urls_used": dem["download_urls"],
        "dem_downloaded_archives": dem["archive_paths"],
        "dem_geotiff_paths": dem["geotiff_paths"],
        "dem_source_crs_list": dem["source_crs_list"],
        "dem_plot_crs": dem["plot_crs"],
        "dem_plot_step": dem["plot_step"],
        "pre_existing_sierra_mask_found": True,
        "pre_existing_sierra_mask_source": "snow_ml.data.build_sierra_mask",
        "current_sierra_mask_type": swe["mask_type"],
        "current_sierra_region_bounds": asdict(DEFAULT_SIERRA_REGION),
        "equal_width_latitude_bands_used_anywhere_in_this_diagnostic": False,
        "artifacts": {
            "figure_png": str(FIG_PNG),
            "figure_pdf": str(FIG_PDF),
            "metadata_json": str(META_JSON),
            "run_log": str(RUN_LOG),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peak_memory_mb": peak_memory_mb(),
    }

    META_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote metadata JSON: {META_JSON}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    if RUN_LOG.exists():
        RUN_LOG.unlink()

    start = perf_counter()

    log("Starting three-panel Sierra SWE/DEM spatial diagnostic")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"DEM cache directory: {DEM_DIR}")
    log(f"Plot extent lon/lat: {PLOT_EXTENT}")

    swe = load_spatial_apr1_fields()
    dem_info = query_dem_downloads()
    dem = build_dem_mosaic(dem_info)

    plot_four_panel(swe, dem)

    runtime_seconds = perf_counter() - start
    write_metadata(swe, dem, runtime_seconds)

    log("Done")


if __name__ == "__main__":
    main()