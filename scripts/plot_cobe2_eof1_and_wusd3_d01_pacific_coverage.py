#!/usr/bin/env python3
"""
Plot the COBE2 Pacific EOF1 and the WUS-D3 d01 SST coverage on the same region.

This script writes a two-panel figure:
1. COBE2 EOF1 over the requested Pacific box.
2. A representative WUS-D3 d01 SST field remapped onto the same box, with
   missing/unavailable cells left white.

The WUS-D3 panel uses the default historical dataset and the first available
yearly `tskin` file. The daily file is aggregated to monthly means and then
averaged across those months to produce a lightweight representative SST field.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_cobe2_wusd3_sst_t2m_mode_projection import (
    COBE2_EOF_FILE,
    WUSD3_DOMAIN,
    WUSD3_ROOT,
    WUSD3_SST_VARIABLE,
    load_cobe2_reference,
    load_wusd3_grid,
)
from snow_ml.data_wusd3 import DEFAULT_WUSD3_DATASET_ID


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "wusd3_pacific_region_visuals"
OUTPUT_FIGURE = OUTPUT_DIR / "cobe2_eof1_and_wusd3_d01_pacific_coverage.png"
OUTPUT_FIGURE_FIGURES = PROJECT_ROOT / "figures" / "cobe2_eof1_and_wusd3_d01_pacific_coverage.png"
OUTPUT_METADATA = OUTPUT_DIR / "cobe2_eof1_and_wusd3_d01_pacific_coverage.json"
WUS_SST_MONTHLY_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_sst_on_cobe2_grid_monthly"
    f"/{DEFAULT_WUSD3_DATASET_ID}/{DEFAULT_WUSD3_DATASET_ID}_tskin_on_cobe2_grid_monthly_mean.nc"
)

PACIFIC_LAT_MIN = -10.0
PACIFIC_LAT_MAX = 60.0
PACIFIC_LON_MIN = 120.0
PACIFIC_LON_MAX = 280.0


def ensure_output_dir() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Expected existing output directory, found missing path: {OUTPUT_DIR}")


def find_first_wusd3_tskin_file(dataset_id: str) -> Path:
    base_dir = WUSD3_ROOT / dataset_id / "postprocess" / WUSD3_DOMAIN
    paths = sorted(base_dir.glob(f"{WUSD3_SST_VARIABLE}.daily.*.nc"))
    if not paths:
        raise FileNotFoundError(f"No {WUSD3_SST_VARIABLE} files found under {base_dir}")
    return paths[0]


def load_representative_wusd3_sst_native(path: Path, ocean_mask: np.ndarray) -> tuple[np.ndarray, str]:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        monthly = ds[[WUSD3_SST_VARIABLE]].rename({"day": "time"}).resample(time="MS").mean()
        values = np.asarray(monthly[WUSD3_SST_VARIABLE].values, dtype=np.float64)
    representative = np.nanmean(values, axis=0)
    representative = np.where(ocean_mask, representative, np.nan)
    return representative.astype(np.float32), path.name


def load_representative_wusd3_sst_remapped(path: Path) -> tuple[np.ndarray, str]:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        values = np.asarray(ds["tskin"].values, dtype=np.float64)
    representative = np.nanmean(values, axis=0)
    return representative.astype(np.float32), path.name


def remap_native_field_to_cobe2(
    native_field: np.ndarray,
    native_latitude: np.ndarray,
    native_longitude: np.ndarray,
    ocean_mask: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    source_points = np.column_stack([native_latitude[ocean_mask], native_longitude[ocean_mask]])
    source_values = np.asarray(native_field[ocean_mask], dtype=np.float64)
    if source_points.shape[0] == 0:
        raise ValueError("WUS-D3 d01 ocean mask has zero ocean cells")
    interpolator = LinearNDInterpolator(source_points, source_values, fill_value=np.nan)
    target_lon2d, target_lat2d = np.meshgrid(target_longitude, target_latitude)
    target_points = np.column_stack([target_lat2d.ravel(), target_lon2d.ravel()])
    remapped = interpolator(target_points).reshape(target_latitude.size, target_longitude.size)
    return np.asarray(remapped, dtype=np.float32)


def reorder_longitude_to_360(
    longitude: np.ndarray,
    *fields: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    longitude_360 = np.mod(np.asarray(longitude, dtype=np.float64), 360.0)
    sort_index = np.argsort(longitude_360)
    reordered_fields = [np.take(np.asarray(field), sort_index, axis=-1) for field in fields]
    return longitude_360[sort_index], reordered_fields


def pacific_indices(latitude: np.ndarray, longitude_360: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_idx = np.where((latitude >= PACIFIC_LAT_MIN) & (latitude <= PACIFIC_LAT_MAX))[0]
    lon_idx = np.where((longitude_360 >= PACIFIC_LON_MIN) & (longitude_360 <= PACIFIC_LON_MAX))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Pacific crop indices are empty")
    return lat_idx, lon_idx


def crop_native_pacific_region(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    longitude_360 = np.mod(longitude, 360.0)
    region_mask = (
        (latitude >= PACIFIC_LAT_MIN)
        & (latitude <= PACIFIC_LAT_MAX)
        & (longitude_360 >= PACIFIC_LON_MIN)
        & (longitude_360 <= PACIFIC_LON_MAX)
    )
    cropped_values = np.where(region_mask, values, np.nan)
    return latitude, longitude_360, cropped_values, region_mask


def plot_fields(
    native_latitude: np.ndarray,
    native_longitude_360: np.ndarray,
    native_wus_sst_crop: np.ndarray,
    native_support_mask: np.ndarray,
    cobe2_latitude: np.ndarray,
    cobe2_longitude: np.ndarray,
    eof1_crop: np.ndarray,
    remapped_wus_sst_crop: np.ndarray,
    dataset_id: str,
    sample_file: str,
    remapped_file: str,
) -> None:
    lon2d, lat2d = np.meshgrid(cobe2_longitude, cobe2_latitude)
    eof_vmax = float(np.nanmax(np.abs(eof1_crop)))
    if not np.isfinite(eof_vmax) or eof_vmax == 0.0:
        eof_vmax = 1.0

    native_strict = np.where(native_support_mask, native_wus_sst_crop, np.nan)
    remapped_support_mask = np.isfinite(remapped_wus_sst_crop)
    remapped_strict = np.where(remapped_support_mask, remapped_wus_sst_crop, np.nan)

    sst_vmin = float(min(np.nanmin(native_strict), np.nanmin(remapped_strict)))
    sst_vmax = float(max(np.nanmax(native_strict), np.nanmax(remapped_strict)))
    if not np.isfinite(sst_vmin) or not np.isfinite(sst_vmax) or sst_vmin == sst_vmax:
        sst_vmin, sst_vmax = 0.0, 1.0

    eof_cmap = plt.get_cmap("RdBu_r").copy()
    sst_cmap = plt.get_cmap("coolwarm").copy()
    eof_cmap.set_bad(color="white")
    sst_cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)

    eof_mesh = axes[0].pcolormesh(
        lon2d,
        lat2d,
        eof1_crop,
        cmap=eof_cmap,
        shading="auto",
        vmin=-eof_vmax,
        vmax=eof_vmax,
    )
    axes[0].contour(
        lon2d,
        lat2d,
        remapped_support_mask.astype(np.int8),
        levels=[0.5],
        colors="black",
        linewidths=0.9,
    )
    axes[0].set_title("COBE2 Pacific EOF1\nblack contour = WUS d01 SST coverage")
    axes[0].set_xlabel("Longitude (0..360E)")
    axes[0].set_ylabel("Latitude")
    axes[0].set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
    axes[0].set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
    fig.colorbar(eof_mesh, ax=axes[0], shrink=0.9).set_label("EOF loading")

    native_mesh = axes[1].pcolormesh(
        native_longitude_360,
        native_latitude,
        native_strict,
        cmap=sst_cmap,
        shading="auto",
        vmin=sst_vmin,
        vmax=sst_vmax,
    )
    axes[1].set_title("Naive available WUS-D3 d01 SST region\nwhite = no native SST value")
    axes[1].set_xlabel("Longitude (0..360E)")
    axes[1].set_ylabel("Latitude")
    axes[1].set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
    axes[1].set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
    fig.colorbar(native_mesh, ax=axes[1], shrink=0.9).set_label("Representative SST (K)")

    remapped_mesh = axes[2].pcolormesh(
        lon2d,
        lat2d,
        remapped_strict,
        cmap=sst_cmap,
        shading="auto",
        vmin=sst_vmin,
        vmax=sst_vmax,
    )
    axes[2].set_title("Monthly WUS SST remapped to COBE2 grid\nwhite = unsupported after remap")
    axes[2].set_xlabel("Longitude (0..360E)")
    axes[2].set_ylabel("Latitude")
    axes[2].set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
    axes[2].set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
    fig.colorbar(remapped_mesh, ax=axes[2], shrink=0.9).set_label("Representative SST (K)")

    fig.suptitle(
        "Pacific-region view for SST projection setup\n"
        f"WUS dataset: {dataset_id} | native sample: {sample_file} | remapped monthly file: {remapped_file}",
        fontsize=13,
    )
    fig.savefig(OUTPUT_FIGURE, dpi=220)
    plt.close(fig)


def write_metadata(
    dataset_id: str,
    sample_file: str,
    cobe2_latitude: np.ndarray,
    cobe2_longitude: np.ndarray,
    wus_sst_crop: np.ndarray,
) -> None:
    metadata = {
        "cobe2_eof_file": str(COBE2_EOF_FILE),
        "wusd3_dataset_id": dataset_id,
        "wusd3_domain": WUSD3_DOMAIN,
        "wusd3_sample_file": sample_file,
        "wusd3_remapped_monthly_mean_file": str(WUS_SST_MONTHLY_FILE),
        "output_figure": str(OUTPUT_FIGURE),
        "output_figure_figures_copy": str(OUTPUT_FIGURE_FIGURES),
        "pacific_region_360": {
            "lat_min": PACIFIC_LAT_MIN,
            "lat_max": PACIFIC_LAT_MAX,
            "lon_min": PACIFIC_LON_MIN,
            "lon_max": PACIFIC_LON_MAX,
        },
        "cropped_shape": [int(cobe2_latitude.size), int(cobe2_longitude.size)],
        "available_cell_count": int(np.count_nonzero(np.isfinite(wus_sst_crop))),
        "missing_cell_count": int(np.count_nonzero(~np.isfinite(wus_sst_crop))),
        "notes": [
            "Left panel is COBE2 EOF1 over the requested Pacific region.",
            "Middle panel is the naive available native WUS-D3 d01 SST region from the first available yearly file, aggregated to monthly means and averaged across those months.",
            "Right panel is the representative WUS monthly SST from the existing monthly mean artifact already remapped to the COBE2 grid.",
            "The right panel is plotted only on the strict support mask cobe2.valid_mask AND finite(remapped WUS SST).",
            "Panels 2 and 3 show only SST values on available cells; the boundary appears naturally from missing values shown as white.",
            "White cells indicate non-ocean, outside-domain, or unsupported WUS-D3 SST cells.",
        ],
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dir()

    dataset_id = DEFAULT_WUSD3_DATASET_ID
    cobe2 = load_cobe2_reference()
    grid = load_wusd3_grid()
    sample_file = find_first_wusd3_tskin_file(dataset_id)
    native_sst, sample_name = load_representative_wusd3_sst_native(sample_file, grid.ocean_mask)
    if not WUS_SST_MONTHLY_FILE.exists():
        raise FileNotFoundError(f"Missing remapped WUS monthly mean file: {WUS_SST_MONTHLY_FILE}")
    remapped_sst, remapped_file_name = load_representative_wusd3_sst_remapped(WUS_SST_MONTHLY_FILE)
    native_latitude, native_longitude_360, native_sst_crop, native_support_mask = crop_native_pacific_region(
        grid.latitude,
        grid.longitude,
        native_sst,
    )
    cobe2_longitude_360, reordered_fields = reorder_longitude_to_360(
        cobe2.longitude,
        cobe2.eof[0],
        cobe2.valid_mask,
        remapped_sst,
    )
    eof1_360 = reordered_fields[0]
    valid_mask_360 = reordered_fields[1]
    remapped_sst_360 = reordered_fields[2]

    lat_idx, lon_idx = pacific_indices(cobe2.latitude, cobe2_longitude_360)
    cobe2_latitude = cobe2.latitude[lat_idx]
    cobe2_longitude = cobe2_longitude_360[lon_idx]
    valid_mask_crop = np.asarray(valid_mask_360[lat_idx, :][:, lon_idx], dtype=bool)
    eof1_crop = np.asarray(eof1_360[lat_idx, :][:, lon_idx], dtype=np.float64)
    eof1_crop = np.where(valid_mask_crop, eof1_crop, np.nan)
    remapped_sst_crop = np.asarray(remapped_sst_360[lat_idx, :][:, lon_idx], dtype=np.float64)
    wus_sst_crop = np.where(valid_mask_crop, remapped_sst_crop, np.nan)

    plot_fields(
        native_latitude=native_latitude,
        native_longitude_360=native_longitude_360,
        native_wus_sst_crop=native_sst_crop,
        native_support_mask=native_support_mask & np.isfinite(native_sst_crop),
        cobe2_latitude=cobe2_latitude,
        cobe2_longitude=cobe2_longitude,
        eof1_crop=eof1_crop,
        remapped_wus_sst_crop=wus_sst_crop,
        dataset_id=dataset_id,
        sample_file=sample_name,
        remapped_file=remapped_file_name,
    )
    write_metadata(
        dataset_id=dataset_id,
        sample_file=sample_name,
        cobe2_latitude=cobe2_latitude,
        cobe2_longitude=cobe2_longitude,
        wus_sst_crop=wus_sst_crop,
    )
    OUTPUT_FIGURE_FIGURES.write_bytes(OUTPUT_FIGURE.read_bytes())
    print(f"Saved figure: {OUTPUT_FIGURE}", flush=True)
    print(f"Saved figure copy: {OUTPUT_FIGURE_FIGURES}", flush=True)
    print(f"Saved metadata: {OUTPUT_METADATA}", flush=True)


if __name__ == "__main__":
    main()
