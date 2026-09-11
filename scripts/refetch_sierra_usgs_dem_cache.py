"""
Re-fetch the USGS NED DEM tiles previously cached at
artifacts/swe_target_spatial_diagnostic/external_data/usgs_dem (home), which
has been purged (no longer present at either the home path or its pscratch
migration snapshot -- consistent with NERSC's scratch retention policy on
~3-month-old unused files).

Reuses the EXACT same query/dataset-name-list/bbox already established and
tested in scripts/run_swe_target_spatial_diagnostic.py (query_dem_downloads),
inlined here to avoid that module's unconditional rasterio/cartopy imports
(and a vendored rasterio build that is ABI-incompatible with this Python).

This time the cache is stored under pscratch (heavy data), not home, per the
project's storage policy.
"""
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import requests

REQUEST_TIMEOUT = 120
DEM_API_DATASET_NAMES = [
    "Digital Elevation Model (DEM) 1 arc-second",
    "Digital Elevation Model (DEM) 1/3 arc-second",
    "National Elevation Dataset (NED) 1 arc-second",
    "National Elevation Dataset (NED) 1/3 arc-second",
]
DEM_QUERY_BBOX = "-123.5,34.5,-117.0,42.5"  # identical to run_swe_target_spatial_diagnostic.py

NEW_DEM_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "swe_target_spatial_diagnostic/external_data/usgs_dem"
)


def log(msg):
    print(msg, flush=True)


def sanitize_name(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()


def collapse_historical_dem_urls(download_urls):
    latest_by_tile: Dict[str, Any] = {}
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


def query_dem_downloads() -> Dict[str, Any]:
    log("Starting DEM API search")
    for dataset_name in DEM_API_DATASET_NAMES:
        all_items: List[Dict[str, Any]] = []
        offset = 0
        page_size = 200
        total_reported = None
        while True:
            params = {
                "datasets": dataset_name,
                "bbox": DEM_QUERY_BBOX,
                "prodFormats": "GeoTIFF",
                "outputFormat": "JSON",
                "max": str(page_size),
                "offset": str(offset),
            }
            log(f"Querying DEM API for dataset: {dataset_name} (offset={offset})")
            try:
                response = requests.get(
                    "https://tnmaccess.nationalmap.gov/api/v1/products",
                    params=params, timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                log(f"  failed: {exc!r}")
                break
            items = payload.get("items", []) or []
            total_reported = payload.get("total", total_reported)
            log(f"  returned {len(items)} item(s) this page (API-reported total={total_reported})")
            all_items.extend(items)
            offset += len(items)
            if len(items) < page_size or (total_reported is not None and offset >= total_reported):
                break
        log(f"  {dataset_name}: {len(all_items)} item(s) across all pages")
        if all_items:
            download_urls = sorted({item["downloadURL"] for item in all_items if item.get("downloadURL")})
            collapsed = collapse_historical_dem_urls(download_urls)
            if len(collapsed) != len(download_urls):
                log(f"  collapsed {len(download_urls)} -> {len(collapsed)} latest tile URL(s)")
            return {"dataset_name": dataset_name, "download_urls": collapsed}
    raise RuntimeError("No DEM products found from The National Map API for the requested Sierra bbox.")


def ensure_download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        log(f"  reusing existing download: {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    log(f"  downloading: {url}")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(destination)
    return destination


def extract_geotiffs_from_archive(archive_path: Path, extract_dir: Path) -> List[Path]:
    suffix = archive_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return [archive_path]
    if suffix != ".zip":
        raise RuntimeError(f"Unsupported DEM download format: {archive_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    geotiffs = []
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


def main():
    NEW_DEM_DIR.mkdir(parents=True, exist_ok=True)
    dem_info = query_dem_downloads()
    log(f"Dataset selected: {dem_info['dataset_name']}")
    log(f"Number of tile URLs: {len(dem_info['download_urls'])}")

    geotiff_paths = []
    for idx, url in enumerate(dem_info["download_urls"], start=1):
        filename = Path(url.split("?")[0]).name
        destination = NEW_DEM_DIR / filename
        log(f"[{idx}/{len(dem_info['download_urls'])}] {filename}")
        archive_path = ensure_download(url, destination)
        extract_dir = NEW_DEM_DIR / f"extracted_{sanitize_name(archive_path.stem)}"
        tiles = extract_geotiffs_from_archive(archive_path, extract_dir)
        geotiff_paths.extend(tiles)
        log(f"  -> {len(tiles)} GeoTIFF(s)")

    log(f"\nTotal GeoTIFF tiles cached: {len(set(geotiff_paths))}")
    log(f"Cache directory: {NEW_DEM_DIR}")


if __name__ == "__main__":
    main()
