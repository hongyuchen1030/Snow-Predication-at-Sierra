#!/usr/bin/env python3
"""
Create a georeferenced SWE spatial diagnostic for the Sierra Nevada target product.
"""

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
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VENDOR = PROJECT_ROOT / ".python_vendor"
SRC_ROOT = PROJECT_ROOT / "src"

for candidate in (PYTHON_VENDOR, PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
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

from snow_ml.data import (  # noqa: E402
    DEFAULT_SIERRA_REGION,
    SWE_MISSING_VALUE,
    SWE_VARIABLE,
    build_sierra_mask,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)


CURRENT_TARGET_PRODUCT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
)
CURRENT_TARGET_SUMMARY = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_apr1_anomaly_standardized_wy1985_2021_summary.json"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_target_spatial_diagnostic"
EXTERNAL_DATA_DIR = OUTPUT_DIR / "external_data"
DEM_DIR = EXTERNAL_DATA_DIR / "usgs_dem"
NE_DIR = EXTERNAL_DATA_DIR / "natural_earth"
FIG_PNG = OUTPUT_DIR / "swe_grid_sierra_region_diagnostic.png"
FIG_PDF = OUTPUT_DIR / "swe_grid_sierra_region_diagnostic.pdf"
META_JSON = OUTPUT_DIR / "swe_target_spatial_metadata.json"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
MEAN_STAT_INDEX = 0
NETCDF_ENGINE = "netcdf4"
REQUEST_TIMEOUT = 120
MAX_DEM_PLOT_DIM = 2200

PLOT_EXTENT = (-123.5, -117.0, 34.5, 42.5)
DEM_API_DATASET_NAMES = [
    "Digital Elevation Model (DEM) 1 arc-second",
    "Digital Elevation Model (DEM) 1/3 arc-second",
    "National Elevation Dataset (NED) 1 arc-second",
    "National Elevation Dataset (NED) 1/3 arc-second",
]
DEM_API_QUERY_TEMPLATE = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    "?datasets={dataset}&bbox=-123.5,34.5,-117.0,42.5&prodFormats=GeoTIFF&outputFormat=JSON"
)
NE_ADMIN1_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
NE_COAST_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, EXTERNAL_DATA_DIR, DEM_DIR, NE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def log(message: str) -> None:
    print(message, flush=True)


def format_crs(crs: Any) -> str:
    if crs is None:
        return "unknown"
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def sanitize_name(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()


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


def convert_lon_to_minus180_180(lon_values: np.ndarray) -> tuple[np.ndarray, str, str]:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    if lon_max > 180.0:
        converted = ((lon_values + 180.0) % 360.0) - 180.0
        order = np.argsort(converted)
        return converted[order], "0_to_360", "-180_to_180"
    return lon_values.copy(), "-180_to_180", "-180_to_180"


def load_current_target_metadata() -> Dict[str, Any]:
    log(f"Loading current target metadata from: {CURRENT_TARGET_PRODUCT}")
    with xr.open_dataset(CURRENT_TARGET_PRODUCT, engine=NETCDF_ENGINE) as ds:
        info = {
            "current_prediction_target_product_path": str(CURRENT_TARGET_PRODUCT),
            "current_prediction_target_variables": list(ds.data_vars),
            "current_prediction_target_coordinates": list(ds.coords),
            "current_prediction_target_attrs": {key: str(value) for key, value in ds.attrs.items()},
            "current_prediction_target_water_years": [int(value) for value in ds["water_year"].values.tolist()],
            "current_prediction_target_kind": (
                "area-weighted Sierra regional April 1 SWE time series with raw mean, anomaly, "
                "and standardized anomaly; not a gridded spatial target"
            ),
        }
    if CURRENT_TARGET_SUMMARY.exists():
        info["current_prediction_target_summary_json"] = str(CURRENT_TARGET_SUMMARY)
    return info


def open_dataset_with_fallback(url: str, params: Dict[str, str]) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def query_dem_downloads() -> Dict[str, Any]:
    log("Starting DEM API search")
    selected_dataset = None
    selected_query_url = None
    selected_items: List[Dict[str, Any]] = []

    for dataset_name in DEM_API_DATASET_NAMES:
        params = {
            "datasets": dataset_name,
            "bbox": "-123.5,34.5,-117.0,42.5",
            "prodFormats": "GeoTIFF",
            "outputFormat": "JSON",
        }
        query_url = DEM_API_QUERY_TEMPLATE.format(dataset=requests.utils.quote(dataset_name, safe=""))
        log(f"Querying DEM API: {query_url}")
        try:
            payload = open_dataset_with_fallback(
                "https://tnmaccess.nationalmap.gov/api/v1/products",
                params=params,
            )
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

    return {
        "dataset_name": selected_dataset,
        "query_url": selected_query_url,
        "download_urls": download_urls,
    }


def download_file(url: str, destination: Path) -> Path:
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    log(f"Downloading: {url}")
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
    sources = []
    vrts = []
    try:
        for path in geotiff_paths:
            src = rasterio.open(path)
            sources.append(src)
            vrts.append(WarpedVRT(src, crs="EPSG:4326"))

        west, east, south, north = PLOT_EXTENT[0], PLOT_EXTENT[1], PLOT_EXTENT[2], PLOT_EXTENT[3]
        mosaic, transform = merge(vrts, bounds=(west, south, east, north), nodata=np.nan)
        dem = mosaic[0].astype(np.float32)
        dem[~np.isfinite(dem)] = np.nan

        south_b, west_b, north_b, east_b = array_bounds(dem.shape[0], dem.shape[1], transform)
        dx = abs(transform.a)
        dy = abs(transform.e)
        light = LightSource(azdeg=315, altdeg=45)
        hillshade = light.hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem[np.isfinite(dem)])), dx=dx, dy=dy)
        hillshade[~np.isfinite(dem)] = np.nan

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
        }
    finally:
        for vrt in vrts:
            vrt.close()
        for src in sources:
            src.close()


def load_natural_earth_layer(url: str, destination_dir: Path, clip_columns: Iterable[str] | None = None) -> Dict[str, Any]:
    log(f"Loading Natural Earth layer: {url}")
    filename = Path(url).name
    zip_path = ensure_download(url, destination_dir / filename)
    extract_dir = destination_dir / sanitize_name(Path(filename).stem)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    shp_candidates = sorted(extract_dir.glob("*.shp"))
    if not shp_candidates:
        raise RuntimeError(f"No shapefile found inside Natural Earth archive: {zip_path}")
    reader = shpreader.Reader(str(shp_candidates[0]))

    west, east, south, north = PLOT_EXTENT[0], PLOT_EXTENT[1], PLOT_EXTENT[2], PLOT_EXTENT[3]
    geometries = []
    allowed_names = {"California", "Nevada", "Oregon"}
    for record in reader.records():
        attrs = record.attributes
        if clip_columns:
            matched_column = None
            for column in clip_columns:
                if column in attrs:
                    matched_column = column
                    break
            if matched_column is not None and attrs[matched_column] not in allowed_names:
                continue

        geom = record.geometry
        minx, miny, maxx, maxy = geom.bounds
        if maxx < west or minx > east or maxy < south or miny > north:
            continue
        geometries.append(geom)

    return {
        "geometries": geometries,
        "download_url": url,
        "zip_path": str(zip_path),
        "source_crs": "EPSG:4326",
        "plot_crs": "EPSG:4326",
    }


def load_spatial_apr1_fields() -> Dict[str, Any]:
    log("Preparing Sierra SWE grid subset and loading raw April 1 SWE fields")
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
    lon, lon_before, lon_after = convert_lon_to_minus180_180(raw_lon)
    order = np.argsort(lon)
    lon = lon[order]

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
        values = values[:, order]
        fields.append(values)
        log(f"loaded raw April 1 SWE grid for WY{int(water_year)}")

    cube = np.stack(fields, axis=0)
    climatology = np.nanmean(cube, axis=0).astype(np.float32)
    valid_any = np.any(np.isfinite(cube), axis=0).astype(np.float32)
    valid_fraction = np.mean(np.isfinite(cube), axis=0).astype(np.float32)
    current_mask_values = np.asarray(current_mask.values, dtype=np.float32)[:, order]

    return {
        "lat": lat,
        "lon": lon,
        "cube": cube,
        "climatology": climatology,
        "valid_any": valid_any,
        "valid_fraction": valid_fraction,
        "current_mask": current_mask_values,
        "raw_spatial_source_template": str(swe_file_for_water_year(WATER_YEAR_START)).replace(str(WATER_YEAR_START), "{water_year}"),
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


def draw_region_box(ax: Any) -> None:
    west = DEFAULT_SIERRA_REGION.lon_min
    east = DEFAULT_SIERRA_REGION.lon_max
    south = DEFAULT_SIERRA_REGION.lat_min
    north = DEFAULT_SIERRA_REGION.lat_max
    xs = [west, east, east, west, west]
    ys = [south, south, north, north, south]
    ax.plot(xs, ys, color="#8b0000", linewidth=1.0, linestyle="--", transform=ccrs.PlateCarree(), zorder=6)


def add_grid(ax: Any) -> None:
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#666666", alpha=0.45, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False


def plot_diagnostic(
    swe: Dict[str, Any],
    dem: Dict[str, Any],
    admin1: Dict[str, Any],
    coast: Dict[str, Any],
) -> None:
    log("Rendering diagnostic figure")
    lon = swe["lon"]
    lat = swe["lat"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    clim = swe["climatology"]
    valid_any = swe["valid_any"]

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    projection = ccrs.PlateCarree()

    ax0 = fig.add_subplot(gs[0, 0], projection=projection)
    mesh0 = ax0.pcolormesh(
        lon,
        lat,
        clim,
        shading="auto",
        cmap="Blues",
        transform=projection,
        rasterized=True,
        zorder=2,
    )
    ax0.contour(
        lon2d,
        lat2d,
        valid_any,
        levels=[0.5],
        colors="k",
        linewidths=1.0,
        transform=projection,
        zorder=4,
    )
    draw_region_box(ax0)
    ax0.set_extent(PLOT_EXTENT, crs=projection)
    add_grid(ax0)
    ax0.set_title("Left: April 1 SWE climatology on the SWE grid\nblack contour = any valid SWE cell, red dashed = current Sierra box")
    cbar0 = fig.colorbar(mesh0, ax=ax0, shrink=0.92, pad=0.02)
    cbar0.set_label("April 1 SWE climatology (m)")

    ax1 = fig.add_subplot(gs[0, 1], projection=projection)
    ax1.imshow(
        dem["hillshade"],
        extent=dem["extent"],
        origin="upper",
        cmap="gray",
        alpha=0.92,
        transform=projection,
        zorder=0,
    )
    ax1.imshow(
        np.ma.masked_invalid(dem["dem"]),
        extent=dem["extent"],
        origin="upper",
        cmap="terrain",
        alpha=0.35,
        transform=projection,
        zorder=1,
    )
    if coast["geometries"]:
        ax1.add_geometries(
            coast["geometries"],
            crs=projection,
            facecolor="none",
            edgecolor="#202020",
            linewidth=0.7,
            zorder=3,
        )
    if admin1["geometries"]:
        ax1.add_geometries(
            admin1["geometries"],
            crs=projection,
            facecolor="none",
            edgecolor="#6e6e6e",
            linewidth=0.6,
            zorder=3,
        )
    mesh1 = ax1.pcolormesh(
        lon,
        lat,
        clim,
        shading="auto",
        cmap="Blues",
        alpha=0.48,
        transform=projection,
        rasterized=True,
        zorder=4,
    )
    ax1.contour(
        lon2d,
        lat2d,
        valid_any,
        levels=[0.5],
        colors="k",
        linewidths=1.0,
        transform=projection,
        zorder=5,
    )
    draw_region_box(ax1)
    ax1.set_extent(PLOT_EXTENT, crs=projection)
    add_grid(ax1)
    ax1.set_title("Right: SWE over georeferenced 3DEP DEM hillshade\nNatural Earth state boundaries and coastline in EPSG:4326")
    cbar1 = fig.colorbar(mesh1, ax=ax1, shrink=0.92, pad=0.02)
    cbar1.set_label("April 1 SWE climatology (m)")

    fig.suptitle(
        "SWE target spatial diagnostic: gridded UCLA April 1 SWE vs Sierra Nevada terrain context",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(FIG_PNG, dpi=220)
    plt.close(fig)

    Image.open(FIG_PNG).convert("RGB").save(FIG_PDF, "PDF", resolution=220.0)


def write_metadata(
    current_target: Dict[str, Any],
    swe: Dict[str, Any],
    dem: Dict[str, Any],
    admin1: Dict[str, Any],
    coast: Dict[str, Any],
    runtime_seconds: float,
) -> None:
    log(f"Writing metadata JSON: {META_JSON}")
    payload = {
        **current_target,
        "swe_source_file_path": swe["raw_spatial_source_template"],
        "swe_source_example_file": swe["raw_spatial_source_example_file"],
        "swe_variable_name": swe["raw_variable_name"],
        "swe_coordinate_names": swe["raw_coordinate_names"],
        "swe_time_water_year_range": {
            "raw_file_time_axis": "daily dates from Oct 1 of previous calendar year through Sep 30 of the water year",
            "selected_target_dates": [f"{int(value)}-04-01" for value in WATER_YEARS.tolist()],
            "water_year_range": [int(WATER_YEAR_START), int(WATER_YEAR_END)],
        },
        "plotted_field_type": "April 1 SWE climatology over WY1985--WY2021 on the actual gridded SWE field",
        "swe_value_type": "raw gridded SWE climatology derived from daily April 1 SWE_Post values",
        "swe_longitude_convention_before": swe["longitude_convention_before"],
        "swe_longitude_convention_after": swe["longitude_convention_after"],
        "dem_api_query_url": dem["query_url"],
        "dem_dataset_name_used": dem["dataset_name"],
        "dem_download_urls_used": dem["download_urls"],
        "dem_downloaded_archives": dem["archive_paths"],
        "dem_geotiff_paths": dem["geotiff_paths"],
        "natural_earth_admin1_download_url": admin1["download_url"],
        "natural_earth_coastline_download_url": coast["download_url"],
        "layer_crs": {
            "swe_grid_plot_crs": "EPSG:4326",
            "dem_source_crs_list": dem["source_crs_list"],
            "dem_plot_crs": dem["plot_crs"],
            "natural_earth_admin1_source_crs": admin1["source_crs"],
            "natural_earth_admin1_plot_crs": admin1["plot_crs"],
            "natural_earth_coastline_source_crs": coast["source_crs"],
            "natural_earth_coastline_plot_crs": coast["plot_crs"],
        },
        "pre_existing_sierra_mask_found": True,
        "pre_existing_sierra_mask_source": "snow_ml.data.build_sierra_mask",
        "pre_existing_north_middle_south_sierra_definition_found": False,
        "pre_existing_north_middle_south_note": (
            "No pre-existing North/Middle/South Sierra SWE target definition was found in the SWE target workflow."
        ),
        "equal_width_latitude_bands_used_anywhere_in_this_diagnostic": False,
        "masking_method": (
            "Loaded the actual gridded SWE field on the Sierra subset, plotted the SWE climatology on the native "
            "lon-lat grid, and overlaid the current repo Sierra box mask only as a diagnostic red dashed boundary."
        ),
        "current_sierra_mask_type": swe["mask_type"],
        "current_sierra_region_bounds": asdict(DEFAULT_SIERRA_REGION),
        "plot_extent_lon_lat": {
            "lon_min": PLOT_EXTENT[0],
            "lon_max": PLOT_EXTENT[1],
            "lat_min": PLOT_EXTENT[2],
            "lat_max": PLOT_EXTENT[3],
        },
        "artifacts": {
            "figure_png": str(FIG_PNG),
            "figure_pdf": str(FIG_PDF),
            "metadata_json": str(META_JSON),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peak_memory_mb": peak_memory_mb(),
    }
    META_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dirs()
    start = perf_counter()
    log(f"Starting georeferenced SWE spatial diagnostic on host {os.uname().nodename}")
    log(f"Using workspace-local PYTHON_VENDOR path: {PYTHON_VENDOR}")

    current_target = load_current_target_metadata()
    swe = load_spatial_apr1_fields()
    dem_info = query_dem_downloads()
    dem = build_dem_mosaic(dem_info)
    admin1 = load_natural_earth_layer(NE_ADMIN1_URL, NE_DIR, clip_columns=("name", "name_en", "adm1name"))
    coast = load_natural_earth_layer(NE_COAST_URL, NE_DIR)
    plot_diagnostic(swe, dem, admin1, coast)
    runtime_seconds = perf_counter() - start
    write_metadata(current_target, swe, dem, admin1, coast, runtime_seconds)

    log(f"Wrote figure PNG: {FIG_PNG}")
    log(f"Wrote figure PDF: {FIG_PDF}")
    log(f"Wrote metadata JSON: {META_JSON}")


if __name__ == "__main__":
    main()
