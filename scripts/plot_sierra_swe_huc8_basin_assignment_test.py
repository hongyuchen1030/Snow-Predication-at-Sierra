#!/usr/bin/env python3
"""
Test-run assignment of Sierra SWE grid cells to North/Central/South Sierra
regions using official USGS WBD HUC8 watershed polygons and the CDEC/DWR
regional snowpack grouping.

This script loads ONE water year of April 1 SWE, queries the USGS WBD HUC8
ArcGIS REST service, assigns each valid Sierra SWE grid cell to a CDEC-style
North/Central/South basin group, and makes a diagnostic visualization.

No geopandas/fiona/pyogrio required.

Outputs:
  artifacts/swe_target_spatial_diagnostic/basin_assignment_test_wy2021.png
  artifacts/swe_target_spatial_diagnostic/basin_assignment_test_wy2021.pdf
  artifacts/swe_target_spatial_diagnostic/basin_assignment_test_wy2021_metadata.json
  artifacts/swe_target_spatial_diagnostic/wbd_huc8_features_used_wy2021.csv
  artifacts/swe_target_spatial_diagnostic/basin_assignment_grid_wy2021.npz
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from PIL import Image

from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import Polygon, MultiPolygon
from snow_ml.data import (
    DEFAULT_SIERRA_REGION,
    SWE_MISSING_VALUE,
    SWE_VARIABLE,
    build_sierra_mask,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_target_spatial_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWE_FOOTPRINT_THRESHOLD_M = 0.05
TEST_WATER_YEAR = 2021
TARGET_DATE = f"{TEST_WATER_YEAR}-04-01"

MEAN_STAT_INDEX = 0
NETCDF_ENGINE = "netcdf4"
REQUEST_TIMEOUT = 120

# Sierra SWE diagnostic box:
# 122.5W--118W, 35N--42N
PLOT_EXTENT = (-122.5, -118.0, 35.0, 42.0)

# Slightly expanded WBD query box so basin polygons crossing the Sierra box are included.
WBD_QUERY_BBOX = (-123.5, 34.5, -117.0, 42.5)

WBD_HUC8_QUERY_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query"
)

FIG_PNG = OUTPUT_DIR / f"basin_assignment_test_wy{TEST_WATER_YEAR}.png"
FIG_PDF = OUTPUT_DIR / f"basin_assignment_test_wy{TEST_WATER_YEAR}.pdf"
META_JSON = OUTPUT_DIR / f"basin_assignment_test_wy{TEST_WATER_YEAR}_metadata.json"
FEATURE_CSV = OUTPUT_DIR / f"wbd_huc8_features_used_wy{TEST_WATER_YEAR}.csv"
GRID_NPZ = OUTPUT_DIR / f"basin_assignment_grid_wy{TEST_WATER_YEAR}.npz"
GRID_NC = OUTPUT_DIR / f"basin_assignment_grid_wy{TEST_WATER_YEAR}.nc"
UNASSIGNED_OVERLAP_CSV = OUTPUT_DIR / f"unassigned_huc8_intersecting_active_swe_wy{TEST_WATER_YEAR}.csv"

# CDEC/DWR regional snowpack grouping, translated into HUC8 name keywords.
# Source concept:
#   NORTH:   Trinity through Feather & Truckee
#   CENTRAL: Yuba & Tahoe through Merced & Walker
#   SOUTH:   San Joaquin & Mono through Kern
#
# This is intentionally keyword based for the first diagnostic. The output CSV
# records every HUC8 name and assigned group so we can refine the list if needed.
NORTH_KEYWORDS = [
    "trinity",
    "sacramento",
    "mccloud",
    "shasta",
    "pit",
    "cow creek",
    "clear creek",
    "battle creek",
    "antelope",
    "mill creek",
    "deer creek",
    "butte",
    "feather",
    "truckee",
]

CENTRAL_KEYWORDS = [
    "yuba",
    "tahoe",
    "american",
    "cosumnes",
    "mokelumne",
    "calaveras",
    "stanislaus",
    "tuolumne",
    "merced",
    "walker",
    "carson",
]

SOUTH_KEYWORDS = [
    "san joaquin",
    "mono",
    "owens",
    "kings",
    "kaweah",
    "tule",
    "kern",
]

# Exact polygon-name overrides are added only after verification from the
# unassigned-HUC8 overlap diagnostic CSV.
SOUTH_HUC8_NAME_OVERRIDES: List[str] = [
    "Upper King",
]

GROUP_CODE = {
    "unassigned": 0,
    "north": 1,
    "central": 2,
    "south": 3,
}

GROUP_LABEL = {
    0: "Unassigned",
    1: "North",
    2: "Central",
    3: "South",
}

GROUP_COLORS = {
    0: "#d9d9d9",
    1: "#1f78b4",
    2: "#33a02c",
    3: "#e31a1c",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def convert_lon_to_minus180_180(lon_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, str]:
    lon_max = float(np.nanmax(lon_values))
    if lon_max > 180.0:
        converted = ((lon_values + 180.0) % 360.0) - 180.0
        order = np.argsort(converted)
        return converted[order], order, "0_to_360", "-180_to_180"

    order = np.argsort(lon_values)
    return lon_values[order], order, "-180_to_180", "-180_to_180"


def classify_huc8_name(name: str) -> str:
    name_lower = (name or "").lower()

    if any(keyword in name_lower for keyword in NORTH_KEYWORDS):
        return "north"

    if any(keyword in name_lower for keyword in CENTRAL_KEYWORDS):
        return "central"

    if any(keyword in name_lower for keyword in SOUTH_KEYWORDS):
        return "south"

    if any(override.lower() == name_lower for override in SOUTH_HUC8_NAME_OVERRIDES):
        return "south"

    return "unassigned"


def iter_polygon_parts(geom: BaseGeometry) -> Iterable[BaseGeometry]:
    if geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield part
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_polygon_parts(part)


def add_shapely_polygons(
    ax: Any,
    geoms: Iterable[BaseGeometry],
    facecolor: str,
    edgecolor: str = "black",
    linewidth: float = 0.6,
    alpha: float = 0.35,
    zorder: int = 2,
) -> None:
    patches: List[MplPolygon] = []

    for geom in geoms:
        for poly in iter_polygon_parts(geom):
            x, y = poly.exterior.xy
            coords = np.column_stack([x, y])
            patches.append(MplPolygon(coords, closed=True))

    if not patches:
        return

    collection = PatchCollection(
        patches,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(collection)


def add_shapely_boundaries(
    ax: Any,
    geoms: Iterable[BaseGeometry],
    color: str = "black",
    linewidth: float = 0.6,
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    for geom in geoms:
        for poly in iter_polygon_parts(geom):
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def draw_sierra_box(ax: Any) -> None:
    west = DEFAULT_SIERRA_REGION.lon_min
    east = DEFAULT_SIERRA_REGION.lon_max
    south = DEFAULT_SIERRA_REGION.lat_min
    north = DEFAULT_SIERRA_REGION.lat_max

    xs = [west, east, east, west, west]
    ys = [south, south, north, north, south]

    ax.plot(xs, ys, color="black", linewidth=1.0, linestyle="--", zorder=20)


def set_sierra_axes(ax: Any) -> None:
    ax.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.35, linestyle=":", alpha=0.55)


# ---------------------------------------------------------------------
# SWE loading
# ---------------------------------------------------------------------

def load_one_year_apr1_swe() -> Dict[str, Any]:
    log(f"Loading raw April 1 SWE grid for WY{TEST_WATER_YEAR}")

    swe_grid = get_regional_swe_grid_definition(
        water_year=TEST_WATER_YEAR,
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

    path = swe_file_for_water_year(TEST_WATER_YEAR)

    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        swe = ds[SWE_VARIABLE].isel(Stats=MEAN_STAT_INDEX, drop=True)
        swe = swe.sel(time=np.datetime64(TARGET_DATE))
        swe = swe.sel({lat_name: swe_grid.latitude, lon_name: swe_grid.longitude}).load()

    values = np.asarray(swe.where(swe != SWE_MISSING_VALUE).values, dtype=np.float32)
    values = values[:, lon_order]

    mask_values = np.asarray(current_mask.values, dtype=np.float32)[:, lon_order]

    finite_swe = np.isfinite(values)
    inside_sierra_mask = mask_values > 0.5

    # This is the full finite Sierra grid, useful for diagnostics only.
    valid_sierra_grid = finite_swe & inside_sierra_mask

    # This is the actual active SWE footprint used for basin assignment.
    # Do not assign all finite zero-SWE cells; otherwise the whole grid becomes colored.
    valid_swe_footprint = (
        finite_swe
        & inside_sierra_mask
        & (values > SWE_FOOTPRINT_THRESHOLD_M)
    )

    return {
        "lat": lat,
        "lon": lon,
        "swe": values,
        "sierra_mask": mask_values,
        "valid_sierra_grid": valid_sierra_grid,
        "valid_swe_footprint": valid_swe_footprint,
        # Keep old key for compatibility, but now it means active SWE footprint.
        "valid_swe_footprint": valid_swe_footprint,
        "source_file": str(path),
        "variable_name": SWE_VARIABLE,
        "target_date": TARGET_DATE,
        "coordinate_names": {
            "time": time_name,
            "stat": "Stats",
            "latitude": lat_name,
            "longitude": lon_name,
        },
        "longitude_convention_before": lon_before,
        "longitude_convention_after": lon_after,
    }


# ---------------------------------------------------------------------
# WBD HUC8 query and grouping
# ---------------------------------------------------------------------

def query_wbd_huc8_json() -> Dict[str, Any]:
    """
    Query USGS WBD HUC8 polygons from the ArcGIS REST service.

    Important:
    Use f=json instead of f=geojson. The GeoJSON request can trigger a
    500 server error for this bbox and geometry size. The JSON response
    returns Esri polygon rings, which we convert to Shapely locally.
    """
    log("Querying USGS WBD HUC8 polygons from ArcGIS REST service using f=json")

    west, south, east, north = WBD_QUERY_BBOX
    common_params = {
        "outFields": "huc8,name,areasqkm,states",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
        "resultRecordCount": "2000",
        "returnTrueCurves": "false",
        "geometryPrecision": "5",
    }

    def fetch_json(params: Dict[str, Any], label: str) -> Dict[str, Any]:
        response = requests.get(WBD_HUC8_QUERY_URL, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            log(f"WBD query mode {label} failed with status {response.status_code}")
            log(f"URL: {response.url}")
            log(f"Response text first 1000 chars:\n{response.text[:1000]}")
            response.raise_for_status()
        payload = response.json()
        return {"response": response, "payload": payload}

    attempts = [
        (
            "bbox_intersects",
            {
                **common_params,
                "where": "1=1",
                "geometry": f"{west},{south},{east},{north}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
            },
        ),
    ]

    last_error: Any = None
    for attempt_name, params in attempts:
        log(f"Trying WBD query mode: {attempt_name}")
        fetched = fetch_json(params, attempt_name)
        response = fetched["response"]
        payload = fetched["payload"]
        if "error" in payload:
            last_error = payload["error"]
            log(f"WBD query mode {attempt_name} returned service error: {payload['error']}")
            continue

        features = payload.get("features", [])
        if not features:
            last_error = "no features returned"
            log(f"WBD query mode {attempt_name} returned no features")
            continue

        log(f"WBD query mode {attempt_name} returned {len(features)} features")
        return {
            "query_url": response.url,
            "query_mode": attempt_name,
            "json": payload,
        }

    # Final fallback: get nationwide ObjectIDs only, then fetch geometry in
    # small chunks and clip/filter locally. This avoids complex server-side
    # spatial/attribute queries that can trigger HTTP 500s on this layer.
    log("Trying WBD query mode: objectid_chunk_fallback")
    ids_params = {
        "where": "1=1",
        "returnIdsOnly": "true",
        "f": "json",
    }
    fetched_ids = fetch_json(ids_params, "objectid_chunk_fallback_ids")
    ids_payload = fetched_ids["payload"]
    if "error" in ids_payload:
        last_error = ids_payload["error"]
        raise RuntimeError(f"WBD service failed for all query modes; last error: {last_error}")

    object_ids = ids_payload.get("objectIds", []) or []
    if not object_ids:
        raise RuntimeError("WBD objectid_chunk_fallback returned no objectIds.")

    log(f"WBD objectid_chunk_fallback got {len(object_ids)} objectIds; fetching geometry in chunks")

    def fetch_objectid_chunk(chunk_ids: List[int], label: str) -> List[Dict[str, Any]]:
        nonlocal last_error
        chunk_params = {
            **common_params,
            "objectIds": ",".join(str(obj) for obj in chunk_ids),
        }
        fetched_chunk = fetch_json(chunk_params, label)
        chunk_payload = fetched_chunk["payload"]
        if "error" not in chunk_payload:
            return chunk_payload.get("features", []) or []

        last_error = chunk_payload["error"]
        if len(chunk_ids) == 1:
            log(
                "Skipping problematic WBD objectId after repeated service error: "
                f"{chunk_ids[0]} error={last_error}"
            )
            return []

        mid = len(chunk_ids) // 2
        left = chunk_ids[:mid]
        right = chunk_ids[mid:]
        log(
            "WBD objectId chunk failed; splitting chunk "
            f"size={len(chunk_ids)} into {len(left)} and {len(right)}"
        )
        return fetch_objectid_chunk(left, f"{label}_left") + fetch_objectid_chunk(right, f"{label}_right")

    chunk_size = 100
    merged_features: List[Dict[str, Any]] = []
    for start in range(0, len(object_ids), chunk_size):
        chunk = object_ids[start : start + chunk_size]
        merged_features.extend(fetch_objectid_chunk(chunk, f"objectid_chunk_{start}"))

    if not merged_features:
        raise RuntimeError("WBD objectid_chunk_fallback returned no features.")

    log(f"WBD objectid_chunk_fallback returned {len(merged_features)} features before local clipping")
    return {
        "query_url": fetched_ids["response"].url,
        "query_mode": "objectid_chunk_fallback",
        "json": {
            "features": merged_features,
        },
    }

def esri_polygon_to_shapely(geometry: Dict[str, Any]) -> BaseGeometry:
    """
    Convert an Esri JSON polygon geometry to a Shapely geometry.

    ArcGIS polygon geometry format:
        {"rings": [[[x, y], [x, y], ...], ...]}

    For this first diagnostic, we convert each ring into a polygon and union
    the polygons. This is sufficient for plotting HUC8 polygons and assigning
    SWE grid-cell centers inside the Sierra diagnostic box.
    """
    rings = geometry.get("rings", [])
    polygons: List[Polygon] = []

    for ring in rings:
        if len(ring) < 4:
            continue

        coords = [(float(x), float(y)) for x, y in ring]

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if not poly.is_empty and poly.area > 0:
            polygons.append(poly)

    if not polygons:
        return Polygon()

    if len(polygons) == 1:
        return polygons[0]

    geom = unary_union(polygons)

    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom

def build_grouped_huc8_features(wbd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build grouped HUC8 geometries from the Esri JSON WBD response.

    The grouping is based on HUC8 basin names and the CDEC/DWR regional
    snowpack descriptions:
      North:   Trinity through Feather & Truckee
      Central: Yuba & Tahoe through Merced & Walker
      South:   San Joaquin & Mono through Kern
    """
    features_out: List[Dict[str, Any]] = []
    feature_geoms: List[Dict[str, Any]] = []
    grouped_geoms: Dict[str, List[BaseGeometry]] = {
        "north": [],
        "central": [],
        "south": [],
        "unassigned": [],
    }

    query_box = box(
        WBD_QUERY_BBOX[0],
        WBD_QUERY_BBOX[1],
        WBD_QUERY_BBOX[2],
        WBD_QUERY_BBOX[3],
    )

    for feature in wbd["json"]["features"]:
        attrs = feature.get("attributes", {}) or {}

        huc8 = str(attrs.get("huc8", "") or attrs.get("HUC8", ""))
        name = str(attrs.get("name", "") or attrs.get("NAME", ""))
        states = str(attrs.get("states", "") or attrs.get("STATES", ""))
        areasqkm = attrs.get("areasqkm", None)

        group = classify_huc8_name(name)

        geom = esri_polygon_to_shapely(feature.get("geometry", {}))

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            log(f"Skipping empty HUC8 geometry: huc8={huc8}, name={name}")
            continue

        if not geom.intersects(query_box):
            continue

        grouped_geoms[group].append(geom)

        feature_record = {
            "huc8": huc8,
            "name": name,
            "states": states,
            "areasqkm": areasqkm,
            "assigned_group": group,
        }
        features_out.append(feature_record)
        feature_geoms.append(
            {
                **feature_record,
                "geometry": geom,
            }
        )

    group_unions: Dict[str, BaseGeometry | None] = {}
    group_prepared: Dict[str, Any] = {}

    for group in ("north", "central", "south"):
        geoms = grouped_geoms[group]

        if geoms:
            union_geom = unary_union(geoms)

            if not union_geom.is_valid:
                union_geom = union_geom.buffer(0)

            group_unions[group] = union_geom
            group_prepared[group] = prep(union_geom)
        else:
            group_unions[group] = None

    log(
        "Grouped HUC8 feature counts: "
        f"north={len(grouped_geoms['north'])}, "
        f"central={len(grouped_geoms['central'])}, "
        f"south={len(grouped_geoms['south'])}, "
        f"unassigned={len(grouped_geoms['unassigned'])}"
    )

    return {
        "features": features_out,
        "feature_geoms": feature_geoms,
        "grouped_geoms": grouped_geoms,
        "group_unions": group_unions,
        "group_prepared": group_prepared,
    }

def write_huc8_feature_csv(grouped: Dict[str, Any]) -> None:
    log(f"Writing HUC8 feature summary CSV: {FEATURE_CSV}")

    rows = grouped["features"]

    with FEATURE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["huc8", "name", "states", "areasqkm", "assigned_group"],
        )
        writer.writeheader()
        writer.writerows(rows)


def diagnose_unassigned_huc8_over_swe(
    swe: Dict[str, Any],
    grouped: Dict[str, Any],
) -> Path:
    out_csv = UNASSIGNED_OVERLAP_CSV

    lon = swe["lon"]
    lat = swe["lat"]
    valid_swe_footprint = swe["valid_swe_footprint"]

    rows = []

    for feature in grouped["feature_geoms"]:
        if feature["assigned_group"] != "unassigned":
            continue

        geom = feature["geometry"]
        prepared = prep(geom)

        count = 0
        min_lon = np.inf
        max_lon = -np.inf
        min_lat = np.inf
        max_lat = -np.inf

        for i, lat_value in enumerate(lat):
            for j, lon_value in enumerate(lon):
                if not valid_swe_footprint[i, j]:
                    continue

                point = Point(float(lon_value), float(lat_value))

                if prepared.intersects(point):
                    count += 1
                    min_lon = min(min_lon, float(lon_value))
                    max_lon = max(max_lon, float(lon_value))
                    min_lat = min(min_lat, float(lat_value))
                    max_lat = max(max_lat, float(lat_value))

        if count > 0:
            rows.append(
                {
                    "huc8": feature["huc8"],
                    "name": feature["name"],
                    "states": feature["states"],
                    "areasqkm": feature["areasqkm"],
                    "active_swe_cell_count": count,
                    "overlap_lon_min": min_lon,
                    "overlap_lon_max": max_lon,
                    "overlap_lat_min": min_lat,
                    "overlap_lat_max": max_lat,
                }
            )

    rows = sorted(rows, key=lambda row: row["active_swe_cell_count"], reverse=True)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "huc8",
                "name",
                "states",
                "areasqkm",
                "active_swe_cell_count",
                "overlap_lon_min",
                "overlap_lon_max",
                "overlap_lat_min",
                "overlap_lat_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote unassigned HUC8/SWE overlap diagnostic: {out_csv}")
    for row in rows[:20]:
        log(
            "Unassigned HUC8 overlapping active SWE: "
            f"huc8={row['huc8']}, name={row['name']}, "
            f"count={row['active_swe_cell_count']}, "
            f"bbox=({row['overlap_lon_min']}, {row['overlap_lat_min']}, "
            f"{row['overlap_lon_max']}, {row['overlap_lat_max']})"
        )

    return out_csv


# ---------------------------------------------------------------------
# Grid assignment
# ---------------------------------------------------------------------

def assign_swe_grid_to_basin_groups(swe: Dict[str, Any], grouped: Dict[str, Any]) -> Dict[str, Any]:
    log("Assigning active SWE-footprint grid cells to basin groups")

    lon = swe["lon"]
    lat = swe["lat"]

    # Use only the active SWE footprint, not all finite zero-SWE cells.
    valid_swe_footprint = swe["valid_swe_footprint"]

    assignment = np.full(valid_swe_footprint.shape, np.nan, dtype=np.float32)

    group_prepared = grouped["group_prepared"]

    n_valid = int(np.sum(valid_swe_footprint))
    n_assigned_by_huc = 0

    for i, lat_value in enumerate(lat):
        for j, lon_value in enumerate(lon):
            if not valid_swe_footprint[i, j]:
                continue

            point = Point(float(lon_value), float(lat_value))

            assigned_group = "unassigned"

            for group in ("north", "central", "south"):
                prepared_geom = group_prepared.get(group)
                if prepared_geom is not None and prepared_geom.intersects(point):
                    assigned_group = group
                    break

            if assigned_group != "unassigned":
                n_assigned_by_huc += 1
                assignment[i, j] = float(GROUP_CODE[assigned_group])

    counts = {
        "North": int(np.sum(assignment == GROUP_CODE["north"])),
        "Central": int(np.sum(assignment == GROUP_CODE["central"])),
        "South": int(np.sum(assignment == GROUP_CODE["south"])),
        "Unassigned active SWE cells": int(np.sum(valid_swe_footprint & ~np.isfinite(assignment))),
        "Non-SWE / masked-out cells": int(np.sum(~valid_swe_footprint)),
    }

    log(f"Active SWE-footprint cells: {n_valid}")
    log(f"Assigned by HUC8 N/C/S polygons: {n_assigned_by_huc}")
    log(f"Assignment counts: {counts}")

    np.savez(
        GRID_NPZ,
        lon=lon,
        lat=lat,
        swe=swe["swe"],
        valid_sierra_grid=swe["valid_sierra_grid"].astype(np.int8),
        valid_swe_footprint=valid_swe_footprint.astype(np.int8),
        assignment=assignment,
        group_code=json.dumps(GROUP_CODE),
        group_label=json.dumps(GROUP_LABEL),
        swe_footprint_threshold_m=SWE_FOOTPRINT_THRESHOLD_M,
        forced_unassigned_active_swe_to_south=False,
    )

    label_da = xr.DataArray(
        assignment,
        dims=("lat", "lon"),
        coords={
            "lat": lat,
            "lon": lon,
        },
        name="sierra_swe_region_label",
        attrs={
            "description": (
                "North/Central/South Sierra SWE region label assigned from "
                "USGS WBD HUC8 basin polygons and CDEC/DWR basin grouping. "
                "Active SWE-footprint grid cells remain unassigned unless they "
                "fall inside a polygon already labeled North, Central, or South."
            ),
            "label_1": "North",
            "label_2": "Central",
            "label_3": "South",
            "swe_footprint_threshold_m": float(SWE_FOOTPRINT_THRESHOLD_M),
            "test_water_year": int(TEST_WATER_YEAR),
            "target_date": TARGET_DATE,
            "forced_unassigned_active_swe_to_south": "false",
        },
    )

    label_da.to_netcdf(GRID_NC)

    return {
        "assignment": assignment,
        "counts": counts,
        "n_valid_swe_footprint_cells": n_valid,
        "n_assigned_by_huc_cells": n_assigned_by_huc,
        "n_assigned_ncs_cells": int(np.sum(np.isfinite(assignment))),
        "n_unassigned_active_swe_cells": int(np.sum(valid_swe_footprint & ~np.isfinite(assignment))),
        "grid_npz": str(GRID_NPZ),
        "grid_nc": str(GRID_NC),
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_assignment_diagnostic(
    swe: Dict[str, Any],
    grouped: Dict[str, Any],
    assigned: Dict[str, Any],
) -> None:
    log("Rendering basin-assignment diagnostic figure")

    lon = swe["lon"]
    lat = swe["lat"]
    swe_values = swe["swe"]
    valid_swe_footprint = swe["valid_swe_footprint"]
    assignment = assigned["assignment"]

    # Discrete basin-group colormap for assigned grid cells.
    # 1 = North, 2 = Central, 3 = South.
    group_cmap = ListedColormap(
        [
            GROUP_COLORS[1],  # North
            GROUP_COLORS[2],  # Central
            GROUP_COLORS[3],  # South
        ]
    )
    group_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], group_cmap.N)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.5, 6.5),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Panel 1: raw SWE footprint for one year
    # ------------------------------------------------------------
    ax0 = axes[0]

    # Only show active SWE footprint, not zero-SWE finite cells.
    swe_positive = np.where(valid_swe_footprint, swe_values, np.nan)

    mesh0 = ax0.pcolormesh(
        lon,
        lat,
        swe_positive,
        shading="auto",
        cmap="Blues",
        rasterized=True,
    )

    draw_sierra_box(ax0)
    set_sierra_axes(ax0)
    ax0.set_title(
        f"1. April 1 SWE footprint\nWY{TEST_WATER_YEAR}"
    )

    cbar0 = fig.colorbar(mesh0, ax=ax0, shrink=0.82)
    cbar0.set_label("April 1 SWE (m)")

    # ------------------------------------------------------------
    # Panel 2: WBD HUC8 basin polygons grouped by CDEC region
    # ------------------------------------------------------------
    ax1 = axes[1]

    add_shapely_polygons(
        ax1,
        grouped["grouped_geoms"]["unassigned"],
        facecolor=GROUP_COLORS[0],
        edgecolor="0.55",
        linewidth=0.4,
        alpha=0.20,
        zorder=1,
    )

    for group in ("north", "central", "south"):
        add_shapely_polygons(
            ax1,
            grouped["grouped_geoms"][group],
            facecolor=GROUP_COLORS[GROUP_CODE[group]],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.30,
            zorder=2,
        )

    all_geoms = []
    for group in ("unassigned", "north", "central", "south"):
        all_geoms.extend(grouped["grouped_geoms"][group])

    add_shapely_boundaries(
        ax1,
        all_geoms,
        color="0.25",
        linewidth=0.4,
        alpha=0.8,
        zorder=5,
    )

    draw_sierra_box(ax1)
    set_sierra_axes(ax1)
    ax1.set_title(
        "2. USGS WBD HUC8 polygons\ncolored by CDEC basin group"
    )

    handles = [
        plt.Line2D([0], [0], color=GROUP_COLORS[1], lw=6, label="North"),
        plt.Line2D([0], [0], color=GROUP_COLORS[2], lw=6, label="Central"),
        plt.Line2D([0], [0], color=GROUP_COLORS[3], lw=6, label="South"),
        plt.Line2D([0], [0], color=GROUP_COLORS[0], lw=6, label="Unassigned"),
    ]
    ax1.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)

    # ------------------------------------------------------------
    # Panel 3: final clean N/C/S SWE grid-cell assignment only
    # ------------------------------------------------------------
    ax2 = axes[2]

    assignment_masked = np.ma.masked_invalid(assignment)

    mesh2 = ax2.pcolormesh(
        lon,
        lat,
        assignment_masked,
        shading="auto",
        cmap=group_cmap,
        norm=group_norm,
        rasterized=True,
    )

    # Light basin outlines only. No black SWE-contour speckles.
    add_shapely_boundaries(
        ax2,
        all_geoms,
        color="0.70",
        linewidth=0.30,
        alpha=0.5,
        zorder=4,
    )

    draw_sierra_box(ax2)
    set_sierra_axes(ax2)
    ax2.set_title(
        "3. Final N/C/S SWE grid assignment\n"
        "blue = North, green = Central, red = South"
    )

    cbar2 = fig.colorbar(mesh2, ax=ax2, shrink=0.82, ticks=[1, 2, 3])
    cbar2.ax.set_yticklabels(["North", "Central", "South"])
    cbar2.set_label("Assigned SWE region")

    fig.suptitle(
        "Test assignment of Sierra April 1 SWE grid cells using USGS WBD HUC8 basin polygons\n"
        "CDEC grouping: North = Trinity through Feather & Truckee; "
        "Central = Yuba/Tahoe through Merced/Walker; "
        "South = San Joaquin/Mono through Kern",
        fontsize=13,
    )

    fig.savefig(FIG_PNG, dpi=220)
    Image.open(FIG_PNG).convert("RGB").save(FIG_PDF, "PDF", resolution=220.0)
    plt.close(fig)

    log(f"Wrote PNG: {FIG_PNG}")
    log(f"Wrote PDF: {FIG_PDF}")


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def write_metadata(
    swe: Dict[str, Any],
    wbd: Dict[str, Any],
    grouped: Dict[str, Any],
    assigned: Dict[str, Any],
    unassigned_overlap_csv: Path,
    runtime_seconds: float,
) -> None:
    group_feature_counts = {}
    for group in ("north", "central", "south", "unassigned"):
        group_feature_counts[group] = len(grouped["grouped_geoms"][group])

    payload = {
        "purpose": (
            "Test assignment of one-year April 1 Sierra SWE grid cells to North/Central/South "
            "regions using official USGS WBD HUC8 basin polygons and CDEC/DWR snowpack basin grouping."
        ),
        "test_water_year": int(TEST_WATER_YEAR),
        "target_date": TARGET_DATE,
        "swe_source_file": swe["source_file"],
        "swe_variable_name": swe["variable_name"],
        "swe_coordinate_names": swe["coordinate_names"],
        "swe_longitude_convention_before": swe["longitude_convention_before"],
        "swe_longitude_convention_after": swe["longitude_convention_after"],
        "sierra_region_bounds_from_repo": asdict(DEFAULT_SIERRA_REGION),
        "plot_extent_lon_lat": {
            "lon_min": PLOT_EXTENT[0],
            "lon_max": PLOT_EXTENT[1],
            "lat_min": PLOT_EXTENT[2],
            "lat_max": PLOT_EXTENT[3],
        },
        "wbd_source": {
            "service": "USGS WBD ArcGIS REST service",
            "layer": "MapServer/4, 8-digit HU (Subbasin)",
            "query_url": wbd["query_url"],
            "query_bbox_lon_lat": {
                "lon_min": WBD_QUERY_BBOX[0],
                "lat_min": WBD_QUERY_BBOX[1],
                "lon_max": WBD_QUERY_BBOX[2],
                "lat_max": WBD_QUERY_BBOX[3],
            },
        },
        "cdec_reference_grouping": {
            "north": "Trinity through Feather & Truckee",
            "central": "Yuba & Tahoe through Merced & Walker",
            "south": "San Joaquin & Mono through Kern",
        },
        "huc8_name_keyword_mapping": {
            "north_keywords": NORTH_KEYWORDS,
            "central_keywords": CENTRAL_KEYWORDS,
            "south_keywords": SOUTH_KEYWORDS,
            "south_huc8_name_overrides": SOUTH_HUC8_NAME_OVERRIDES,
        },
        "huc8_feature_counts_by_group": group_feature_counts,
        "swe_grid_assignment_counts": assigned["counts"],
        "swe_footprint_threshold_m": SWE_FOOTPRINT_THRESHOLD_M,
        "n_valid_swe_footprint_cells": assigned["n_valid_swe_footprint_cells"],
        "n_assigned_by_huc_cells": assigned["n_assigned_by_huc_cells"],
        "n_unassigned_active_swe_cells": assigned["n_unassigned_active_swe_cells"],
        "n_assigned_north_central_south_cells": assigned["n_assigned_ncs_cells"],
        "forced_unassigned_active_swe_to_south": False,
        "south_grouping_exact_huc8_name_added": SOUTH_HUC8_NAME_OVERRIDES,
        "unassigned_huc8_overlap_csv": str(unassigned_overlap_csv),
        "notes": [
            "This is a first diagnostic based on WBD HUC8 polygon names and CDEC basin descriptions.",
            "The CSV output should be inspected to verify whether any HUC8 names should be reclassified.",
            "Equal-width latitude bands are not used.",
            "No pixel-level forced assignment is used.",
            "If some valid SWE cells remain unassigned, inspect the HUC8 feature CSV and the unassigned-overlap CSV, then refine polygon labels.",
        ],
        "outputs": {
            "figure_png": str(FIG_PNG),
            "figure_pdf": str(FIG_PDF),
            "metadata_json": str(META_JSON),
            "huc8_feature_csv": str(FEATURE_CSV),
            "unassigned_huc8_overlap_csv": str(unassigned_overlap_csv),
            "grid_assignment_npz": str(GRID_NPZ),
            "grid_assignment_nc": assigned.get("grid_nc"),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    META_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote metadata JSON: {META_JSON}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    start = perf_counter()

    log("Starting Sierra SWE HUC8 basin-assignment test")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Test water year: WY{TEST_WATER_YEAR}")
    log(f"Target date: {TARGET_DATE}")

    swe = load_one_year_apr1_swe()

    wbd = query_wbd_huc8_json()
    grouped = build_grouped_huc8_features(wbd)
    write_huc8_feature_csv(grouped)
    unassigned_overlap_csv = diagnose_unassigned_huc8_over_swe(swe, grouped)

    assigned = assign_swe_grid_to_basin_groups(swe, grouped)

    plot_assignment_diagnostic(swe, grouped, assigned)

    runtime_seconds = perf_counter() - start
    write_metadata(swe, wbd, grouped, assigned, unassigned_overlap_csv, runtime_seconds)

    log("Done")


if __name__ == "__main__":
    main()
