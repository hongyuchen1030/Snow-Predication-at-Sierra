"""
ERA5-Land <-> coarsen-8 SWE grid alignment audit.
Read-only: does not modify any existing SWE, ERA5, or mask artifact.
"""
import sys
import os
import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from snow_ml.data import (
    DEFAULT_SIERRA_REGION,
    SWE_VARIABLE,
    SWE_MISSING_VALUE,
    get_swe_grid_definition,
    swe_file_for_water_year,
    era5_land_yearly_file,
)
from config.paths import ERA5_LAND_ROOT, ERA5_LAND_GEOPOTENTIAL_FILE

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_grid_alignment_audit")
for sub in ("plots", "logs"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

KNN_MASK_NPZ = REPO_ROOT / "artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"
REP_YEAR = 2021
GRAVITY = 9.80665
KM_PER_DEG_LAT = 111.132  # WGS84 mean; fine for this regional audit

GROUP_LABEL = {1: "North", 2: "Central", 3: "South"}


def log(msg):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def main():
    summary = {}
    grid_metadata = {}

    # ------------------------------------------------------------------
    section("1. Recover the existing coarsen-8 SWE convention")
    # ------------------------------------------------------------------
    grid1 = get_swe_grid_definition(water_year=REP_YEAR, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    grid8 = get_swe_grid_definition(water_year=REP_YEAR, region=DEFAULT_SIERRA_REGION, coarsen_factor=8)

    native_lat = np.asarray(grid1.latitude.values, dtype=np.float64)
    native_lon = np.asarray(grid1.longitude.values, dtype=np.float64)
    d_lat_native = float(np.median(np.diff(native_lat)))
    d_lon_native = float(np.median(np.diff(native_lon)))

    coarse_lat = np.asarray(grid8.latitude.values, dtype=np.float64)
    coarse_lon = np.asarray(grid8.longitude.values, dtype=np.float64)
    d_lat_coarse = float(np.median(np.diff(coarse_lat)))
    d_lon_coarse = float(np.median(np.diff(coarse_lon)))

    log(f"SWE source file: {grid1.file_path}")
    log(f"Native (coarsen_factor=1) grid shape [lat,lon]: {grid1.grid_shape}")
    log(f"Native lat spacing (median diff): {d_lat_native:.6f} deg")
    log(f"Native lon spacing (median diff): {d_lon_native:.6f} deg")
    log(f"Native lat range: [{native_lat.min():.4f}, {native_lat.max():.4f}]")
    log(f"Native lon range: [{native_lon.min():.4f}, {native_lon.max():.4f}]")
    log(f"Cropped shape (region only, coarsen=1): {grid1.cropped_shape}")
    log(f"Trimmed shape (coarsen=1, i.e. no trim needed): {grid1.trimmed_shape}")
    log(f"Trimmed shape (coarsen=8, truncated to multiple of 8): {grid8.trimmed_shape}")
    log(f"Coarsened (factor=8) grid shape [lat,lon]: {grid8.grid_shape}")
    log(f"Coarsened lat spacing (median diff): {d_lat_coarse:.6f} deg")
    log(f"Coarsened lon spacing (median diff): {d_lon_coarse:.6f} deg")
    log(f"Coarsened lat range: [{coarse_lat.min():.4f}, {coarse_lat.max():.4f}]")
    log(f"Coarsened lon range: [{coarse_lon.min():.4f}, {coarse_lon.max():.4f}]")
    log("Coarsening implementation: xarray .coarsen({lat,lon}: 8, boundary='trim').mean()")
    log("  -> simple block mean over the 8x8 native cells, xarray default skipna=True")
    log("     (NaN/missing native cells excluded from the mean, not zero-filled;")
    log("      this is a valid-cell mean, NOT physically area-weighted by cos(lat)).")
    log("Missing-value handling: SWE_MISSING_VALUE (-999.0) is converted to NaN via")
    log("  `swe.where(swe != SWE_MISSING_VALUE)` BEFORE coarsening (see load_swe_snapshot),")
    log("  so the coarsen-8 mean already skips missing native cells correctly.")
    log("Materialization: generated on demand by get_swe_grid_definition()/coarsen calls;")
    log("  no separately materialized coarsen-8 SWE NetCDF product exists in the repo.")

    grid_metadata["swe_native"] = {
        "file": grid1.file_path,
        "grid_shape_lat_lon": list(grid1.grid_shape),
        "lat_spacing_deg": d_lat_native,
        "lon_spacing_deg": d_lon_native,
        "lat_range": [float(native_lat.min()), float(native_lat.max())],
        "lon_range": [float(native_lon.min()), float(native_lon.max())],
    }
    grid_metadata["swe_coarsen8"] = {
        "grid_shape_lat_lon": list(grid8.grid_shape),
        "lat_spacing_deg": d_lat_coarse,
        "lon_spacing_deg": d_lon_coarse,
        "lat_range": [float(coarse_lat.min()), float(coarse_lat.max())],
        "lon_range": [float(coarse_lon.min()), float(coarse_lon.max())],
        "trimmed_native_shape_lat_lon": list(grid8.trimmed_shape),
        "coarsen_method": "xarray .coarsen(boundary='trim').mean(), skipna=True valid-cell mean",
        "materialized": False,
    }

    # ------------------------------------------------------------------
    section("2. Inspect the native ERA5-Land grid")
    # ------------------------------------------------------------------
    tp_path = era5_land_yearly_file("tp", REP_YEAR)
    t2m_path = era5_land_yearly_file("t2m", REP_YEAR)
    log(f"Precipitation file: {tp_path}")
    log(f"Temperature file: {t2m_path}")

    with xr.open_dataset(tp_path, decode_times=False) as ds_tp:
        log(f"\n--- {tp_path.name} ---")
        log(f"Data vars: {list(ds_tp.data_vars)}")
        log(f"Coords: {list(ds_tp.coords)}")
        tp_varname = [v for v in ds_tp.data_vars if v.lower() not in ("time_bnds",)][0]
        tp_var = ds_tp[tp_varname]
        log(f"Precip variable name: {tp_varname}")
        log(f"Precip variable dims: {tp_var.dims}, shape: {tp_var.shape}")
        log(f"Precip attrs: {json.dumps({k: str(v) for k, v in tp_var.attrs.items()}, indent=2)}")
        era5_lat_name = [c for c in ds_tp.coords if "lat" in c.lower()][0]
        era5_lon_name = [c for c in ds_tp.coords if "lon" in c.lower()][0]
        era5_time_name = [c for c in ds_tp.coords if "time" in c.lower()][0]
        era5_lat = np.asarray(ds_tp[era5_lat_name].values, dtype=np.float64)
        era5_lon_raw = np.asarray(ds_tp[era5_lon_name].values, dtype=np.float64)
        log(f"lat coord name={era5_lat_name}, lon coord name={era5_lon_name}, time coord name={era5_time_name}")
        log(f"lat size={era5_lat.size}, first/last = {era5_lat[0]:.4f}, {era5_lat[-1]:.4f}")
        log(f"lon size={era5_lon_raw.size}, first/last = {era5_lon_raw[0]:.4f}, {era5_lon_raw[-1]:.4f}")
        log(f"lat monotonic decreasing: {bool(np.all(np.diff(era5_lat) < 0))}, "
            f"monotonic increasing: {bool(np.all(np.diff(era5_lat) > 0))}")
        time_vals = np.asarray(ds_tp[era5_time_name].values)
        log(f"time size={time_vals.size}, first 3 values (raw, decode_times=False): {time_vals[:3]}")
        log(f"time attrs: {dict(ds_tp[era5_time_name].attrs)}")
        n_time_check = min(3, time_vals.size - 1)
        if n_time_check >= 1:
            diffs = np.diff(time_vals[: n_time_check + 1])
            log(f"time step diffs (raw units): {diffs}")

    with xr.open_dataset(t2m_path, decode_times=False) as ds_t2m:
        log(f"\n--- {t2m_path.name} ---")
        log(f"Data vars: {list(ds_t2m.data_vars)}")
        t2m_varname = [v for v in ds_t2m.data_vars if v.lower() not in ("time_bnds",)][0]
        t2m_var = ds_t2m[t2m_varname]
        log(f"Temperature variable name: {t2m_varname}")
        log(f"Temperature variable dims: {t2m_var.dims}, shape: {t2m_var.shape}")
        log(f"Temperature attrs: {json.dumps({k: str(v) for k, v in t2m_var.attrs.items()}, indent=2)}")
        t2m_time_name = [c for c in ds_t2m.coords if "time" in c.lower()][0]
        t2m_time_vals = np.asarray(ds_t2m[t2m_time_name].values)
        n_time_check2 = min(3, t2m_time_vals.size - 1)
        if n_time_check2 >= 1:
            diffs2 = np.diff(t2m_time_vals[: n_time_check2 + 1])
            log(f"time step diffs (raw units): {diffs2}")

    # normalize lon convention
    era5_lon = np.where(era5_lon_raw > 180.0, era5_lon_raw - 360.0, era5_lon_raw)
    lon_order = np.argsort(era5_lon)
    era5_lon_sorted = era5_lon[lon_order]

    d_lat_era5 = float(np.median(np.diff(era5_lat))) if era5_lat[0] < era5_lat[-1] else float(-np.median(np.diff(era5_lat)))
    d_lat_era5 = abs(float(np.median(np.abs(np.diff(era5_lat)))))
    d_lon_era5 = abs(float(np.median(np.abs(np.diff(era5_lon_sorted)))))

    log(f"\nERA5-Land lon convention (raw): {'0-360' if era5_lon_raw.max() > 180 else '-180..180'}")
    log(f"ERA5-Land lat spacing: {d_lat_era5:.6f} deg")
    log(f"ERA5-Land lon spacing: {d_lon_era5:.6f} deg")

    grid_metadata["era5_land"] = {
        "precip_file": str(tp_path),
        "temperature_file": str(t2m_path),
        "precip_variable": tp_varname,
        "temperature_variable": t2m_varname,
        "lat_coord_name": era5_lat_name,
        "lon_coord_name": era5_lon_name,
        "lat_size": int(era5_lat.size),
        "lon_size": int(era5_lon_raw.size),
        "lat_spacing_deg": d_lat_era5,
        "lon_spacing_deg": d_lon_era5,
        "lat_first_last": [float(era5_lat[0]), float(era5_lat[-1])],
        "lon_first_last_raw": [float(era5_lon_raw[0]), float(era5_lon_raw[-1])],
        "lon_convention": "0_to_360" if era5_lon_raw.max() > 180 else "-180_to_180",
    }

    # ------------------------------------------------------------------
    section("3. Compare ERA5-Land and coarsen-8 SWE spatial resolution")
    # ------------------------------------------------------------------
    mean_lat_sierra = float(np.mean(coarse_lat))
    km_per_deg_lon_sierra = KM_PER_DEG_LAT * np.cos(np.deg2rad(mean_lat_sierra))

    era5_dlat_km = d_lat_era5 * KM_PER_DEG_LAT
    era5_dlon_km = d_lon_era5 * km_per_deg_lon_sierra
    coarse_dlat_km = d_lat_coarse * KM_PER_DEG_LAT
    coarse_dlon_km = d_lon_coarse * km_per_deg_lon_sierra
    native_dlat_km = d_lat_native * KM_PER_DEG_LAT
    native_dlon_km = d_lon_native * km_per_deg_lon_sierra

    log(f"(km conversion uses mean Sierra latitude = {mean_lat_sierra:.3f} deg for the cos(lat) longitude factor)")
    log(f"\nERA5-Land:  dlat={d_lat_era5:.6f} deg ({era5_dlat_km:.3f} km)  "
        f"dlon={d_lon_era5:.6f} deg ({era5_dlon_km:.3f} km)")
    log(f"Native SWE: dlat={d_lat_native:.6f} deg ({native_dlat_km:.3f} km)  "
        f"dlon={d_lon_native:.6f} deg ({native_dlon_km:.3f} km)")
    log(f"Coarsen-8:  dlat={d_lat_coarse:.6f} deg ({coarse_dlat_km:.3f} km)  "
        f"dlon={d_lon_coarse:.6f} deg ({coarse_dlon_km:.3f} km)")
    log(f"Ratio coarsen-8 / ERA5-Land: dlat x{coarse_dlat_km/era5_dlat_km:.3f}, dlon x{coarse_dlon_km/era5_dlon_km:.3f}")

    grid_metadata["km_spacing_at_mean_sierra_lat"] = {
        "mean_lat_used": mean_lat_sierra,
        "era5_land_km": {"dlat": era5_dlat_km, "dlon": era5_dlon_km},
        "swe_native_km": {"dlat": native_dlat_km, "dlon": native_dlon_km},
        "swe_coarsen8_km": {"dlat": coarse_dlat_km, "dlon": coarse_dlon_km},
    }

    # ------------------------------------------------------------------
    section("6. Coarsen the finalized North/Central/South mask (done before section 4/5 so we have a domain)")
    # ------------------------------------------------------------------
    knn = np.load(KNN_MASK_NPZ, allow_pickle=True)
    mask_lat = knn["lat"]
    mask_lon = knn["lon"]
    mask_assignment = knn["assignment"].astype(np.float32)

    # Confirm the KNN-filled mask's native lat/lon match the coarsen_factor=1 SWE grid exactly.
    lat_match = mask_lat.shape == native_lat.shape and np.allclose(mask_lat, native_lat, atol=1e-5)
    lon_match = mask_lon.shape == native_lon.shape and np.allclose(mask_lon, native_lon, atol=1e-5)
    log(f"KNN-filled mask native lat matches WY{REP_YEAR} native SWE lat exactly: {lat_match} "
        f"(mask shape {mask_lat.shape}, swe shape {native_lat.shape})")
    log(f"KNN-filled mask native lon matches WY{REP_YEAR} native SWE lon exactly: {lon_match} "
        f"(mask shape {mask_lon.shape}, swe shape {native_lon.shape})")
    if not (lat_match and lon_match):
        raise RuntimeError("KNN-filled mask grid does not match the native SWE grid; cannot safely coarsen.")

    trimmed_lat_size, trimmed_lon_size = grid8.trimmed_shape
    log(f"Truncating native mask from {mask_assignment.shape} to a multiple of 8: "
        f"({trimmed_lat_size}, {trimmed_lon_size}) [trim from the end, matching _trim_swe_coordinates]")
    mask_trim = mask_assignment[:trimmed_lat_size, :trimmed_lon_size]

    n_lat_c, n_lon_c = grid8.grid_shape
    mask_blocks = mask_trim.reshape(n_lat_c, 8, n_lon_c, 8).transpose(0, 2, 1, 3).reshape(n_lat_c, n_lon_c, 64)

    coarse_region = np.full((n_lat_c, n_lon_c), np.nan, dtype=np.float32)
    n_mixed = 0
    mixed_examples = []
    for a in range(n_lat_c):
        for b in range(n_lon_c):
            block = mask_blocks[a, b]
            valid = block[np.isfinite(block)]
            if valid.size == 0:
                continue  # stays NaN: outside-Sierra / no valid native Sierra cell in this block
            counts = Counter(valid.tolist())
            if len(counts) > 1:
                n_mixed += 1
                if len(mixed_examples) < 5:
                    mixed_examples.append({
                        "lat_idx": a, "lon_idx": b,
                        "lat": float(coarse_lat[a]), "lon": float(coarse_lon[b]),
                        "counts": {GROUP_LABEL[int(k)]: int(v) for k, v in counts.items()},
                    })
            top_count = max(counts.values())
            winners = sorted([k for k, v in counts.items() if v == top_count])
            coarse_region[a, b] = winners[0]  # deterministic tie-break: lower numeric code (North<Central<South)

    n_north = int(np.sum(coarse_region == 1))
    n_central = int(np.sum(coarse_region == 2))
    n_south = int(np.sum(coarse_region == 3))
    n_outside = int(np.sum(np.isnan(coarse_region)))
    log(f"\nCoarsen-8 region mask ({n_lat_c}x{n_lon_c} = {n_lat_c*n_lon_c} cells):")
    log(f"  North:   {n_north}")
    log(f"  Central: {n_central}")
    log(f"  South:   {n_south}")
    log(f"  Outside-Sierra (NaN): {n_outside}")
    log(f"  Mixed-region blocks (majority-vote resolved): {n_mixed}")
    log(f"  Mixed-block resolution rule: majority class among the (up to 64) valid native cells; "
        f"exact ties broken by lower numeric code (North < Central < South), deterministic.")
    for ex in mixed_examples:
        log(f"    example mixed block: {ex}")

    summary["coarsen8_region_mask"] = {
        "shape_lat_lon": [n_lat_c, n_lon_c],
        "north_cells": n_north,
        "central_cells": n_central,
        "south_cells": n_south,
        "outside_sierra_cells": n_outside,
        "mixed_region_blocks": n_mixed,
        "mixed_block_resolution": "majority vote among up to 64 native cells; ties broken by North<Central<South",
    }

    # save coarse mask artifacts
    np.savez(
        OUT_DIR / "coarsen8_region_mask.npz",
        lat=coarse_lat, lon=coarse_lon, assignment=coarse_region,
    )
    import netCDF4
    with netCDF4.Dataset(OUT_DIR / "coarsen8_region_mask.nc", "w") as ds:
        ds.createDimension("lat", n_lat_c)
        ds.createDimension("lon", n_lon_c)
        la = ds.createVariable("lat", "f8", ("lat",)); la[:] = coarse_lat
        lo = ds.createVariable("lon", "f8", ("lon",)); lo[:] = coarse_lon
        av = ds.createVariable("sierra_swe_region_label", "f4", ("lat", "lon"), fill_value=np.nan)
        av[:, :] = coarse_region
        av.label_1 = "North"; av.label_2 = "Central"; av.label_3 = "South"
        ds.source = "Coarsened (majority vote, 8x8 blocks) from artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"
    log(f"Saved coarsen8_region_mask.npz / .nc to {OUT_DIR}")

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    colors = {1: "#1f78b4", 2: "#33a02c", 3: "#e31a1c"}
    labels = {1: "North", 2: "Central", 3: "South"}
    lat2d_c, lon2d_c = np.meshgrid(coarse_lat, coarse_lon, indexing="ij")
    for code, color in colors.items():
        m = coarse_region == code
        ax.scatter(lon2d_c[m], lat2d_c[m], s=25, color=color, label=labels[code], marker="s")
    mixed_mask = np.zeros_like(coarse_region, dtype=bool)
    for a in range(n_lat_c):
        for b in range(n_lon_c):
            block = mask_blocks[a, b]
            valid = block[np.isfinite(block)]
            if valid.size and len(set(valid.tolist())) > 1:
                mixed_mask[a, b] = True
    ax.scatter(lon2d_c[mixed_mask], lat2d_c[mixed_mask], s=40, facecolors="none",
               edgecolors="black", linewidth=1.0, label=f"Mixed block (n={n_mixed})", marker="o")
    ax.set_xlim(-123.0, -117.5); ax.set_ylim(34.5, 42.2)
    ax.set_title(f"Coarsen-8 North/Central/South region mask (WY{REP_YEAR} lineage)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(markerscale=1.2, loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(OUT_DIR / "plots" / "coarsen8_region_mask.png", dpi=200)
    log(f"Saved plot: {OUT_DIR / 'plots' / 'coarsen8_region_mask.png'}")

    # ------------------------------------------------------------------
    section("4. Quantify grid-center mismatch (coarsen-8 SWE domain vs ERA5-Land)")
    # ------------------------------------------------------------------
    era5_lat2d, era5_lon2d = np.meshgrid(era5_lat, era5_lon_sorted, indexing="ij")
    era5_points = np.column_stack([era5_lat2d.ravel(), era5_lon2d.ravel()])
    era5_tree = cKDTree(era5_points)

    in_domain = np.isfinite(coarse_region)
    dom_lat = lat2d_c[in_domain]
    dom_lon = lon2d_c[in_domain]
    dom_region = coarse_region[in_domain]
    query_points = np.column_stack([dom_lat, dom_lon])

    dist_deg, idx = era5_tree.query(query_points, k=1)
    # convert nearest-neighbor angular offset to km using local cos(lat) scaling per point
    dlat_component = np.abs(era5_points[idx, 0] - dom_lat) * KM_PER_DEG_LAT
    dlon_component = np.abs(era5_points[idx, 1] - dom_lon) * KM_PER_DEG_LAT * np.cos(np.deg2rad(dom_lat))
    dist_km = np.sqrt(dlat_component**2 + dlon_component**2)

    n_domain = len(dom_lat)
    unique_era5_idx, counts_per_era5 = np.unique(idx, return_counts=True)
    n_unique_map = int(np.sum(counts_per_era5 == 1))
    n_shared_map = int(np.sum(counts_per_era5 > 1))
    n_shared_cells = int(np.sum(counts_per_era5[counts_per_era5 > 1]))

    era5_lat_min, era5_lat_max = era5_lat.min(), era5_lat.max()
    era5_lon_min, era5_lon_max = era5_lon_sorted.min(), era5_lon_sorted.max()
    n_outside_coverage = int(np.sum(
        (dom_lat < era5_lat_min) | (dom_lat > era5_lat_max) |
        (dom_lon < era5_lon_min) | (dom_lon > era5_lon_max)
    ))

    dist_stats = {
        "min_km": float(np.min(dist_km)),
        "median_km": float(np.median(dist_km)),
        "mean_km": float(np.mean(dist_km)),
        "p90_km": float(np.percentile(dist_km, 90)),
        "p95_km": float(np.percentile(dist_km, 95)),
        "max_km": float(np.max(dist_km)),
    }
    log(f"Coarsen-8 cells inside finalized Sierra domain: {n_domain}")
    log(f"Nearest-ERA5-Land-center distance (km): {json.dumps(dist_stats, indent=2)}")
    log(f"Coarsen-8 cells mapping to a UNIQUE ERA5-Land nearest cell: {n_unique_map} "
        f"({100*n_unique_map/len(unique_era5_idx) if len(unique_era5_idx) else 0:.1f}% of used ERA5 cells)")
    log(f"ERA5-Land cells claimed by >1 coarsen-8 cell (shared): {n_shared_map} such ERA5 cells, "
        f"covering {n_shared_cells} coarsen-8 cells total")
    log(f"Coarsen-8 cells outside ERA5-Land coverage bounds: {n_outside_coverage}")

    # save per-cell distances CSV
    import csv
    csv_path = OUT_DIR / "era5_to_swe_nearest_distance.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["swe_lat", "swe_lon", "region", "nearest_era5_lat", "nearest_era5_lon", "distance_km"])
        for k in range(n_domain):
            writer.writerow([
                f"{dom_lat[k]:.6f}", f"{dom_lon[k]:.6f}", GROUP_LABEL[int(dom_region[k])],
                f"{era5_points[idx[k],0]:.6f}", f"{era5_points[idx[k],1]:.6f}", f"{dist_km[k]:.4f}",
            ])
    log(f"Saved per-cell distances: {csv_path}")

    summary["nearest_center_mismatch"] = {
        "n_coarsen8_cells_in_domain": n_domain,
        "distance_km_stats": dist_stats,
        "n_unique_era5_cells_used": int(len(unique_era5_idx)),
        "n_era5_cells_shared_by_multiple_swe_cells": n_shared_map,
        "n_swe_cells_sharing_an_era5_cell": n_shared_cells,
        "n_swe_cells_outside_era5_coverage": n_outside_coverage,
    }

    fig2, ax2 = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    sc = ax2.scatter(dom_lon, dom_lat, c=dist_km, s=28, cmap="viridis", marker="s")
    ax2.set_xlim(-123.0, -117.5); ax2.set_ylim(34.5, 42.2)
    ax2.set_title("Nearest ERA5-Land grid-center distance (km) per coarsen-8 SWE cell")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
    fig2.colorbar(sc, ax=ax2, label="distance (km)")
    ax2.grid(True, linestyle=":", alpha=0.4)
    fig2.savefig(OUT_DIR / "plots" / "era5_swe_nearest_distance_map.png", dpi=200)
    log(f"Saved plot: {OUT_DIR / 'plots' / 'era5_swe_nearest_distance_map.png'}")

    # ------------------------------------------------------------------
    section("7. Observational SWE aggregation consistency (native vs coarsen-8), WY%d" % REP_YEAR)
    # ------------------------------------------------------------------
    with xr.open_dataset(swe_file_for_water_year(REP_YEAR), engine="netcdf4", decode_times=True) as ds:
        swe_native_field = ds[SWE_VARIABLE].isel(Stats=0, drop=True).sel(time=np.datetime64(f"{REP_YEAR}-04-01"))
        lat_name = grid1.latitude_name
        lon_name = grid1.longitude_name
        swe_native_field = swe_native_field.sel({lat_name: grid1.latitude, lon_name: grid1.longitude}).load()
    swe_native_vals = np.asarray(swe_native_field.where(swe_native_field != SWE_MISSING_VALUE).values, dtype=np.float64)

    # native regional means using the KNN-filled native mask (unchanged from before)
    native_region = mask_assignment  # 1576x1013-ish, already aligned (confirmed above)
    region_native_mean = {}
    for code, name in GROUP_LABEL.items():
        sel = (native_region == code) & np.isfinite(swe_native_vals)
        region_native_mean[name] = float(np.nanmean(swe_native_vals[sel])) if np.any(sel) else float("nan")

    # coarsen SWE the same way the project already does (block mean over ALL valid
    # native cells in the block, skipna) -- this is what the repo's coarsen-8 pipeline
    # actually produces, with no notion of "region membership" at coarsening time.
    swe_native_trim = swe_native_vals[:trimmed_lat_size, :trimmed_lon_size]
    swe_blocks = swe_native_trim.reshape(n_lat_c, 8, n_lon_c, 8).transpose(0, 2, 1, 3).reshape(n_lat_c, n_lon_c, 64)
    with np.errstate(invalid="ignore"):
        swe_coarse_allvalid = np.nanmean(swe_blocks, axis=2)

    region_coarse_mean_allvalid = {}
    for code, name in GROUP_LABEL.items():
        sel = (coarse_region == code) & np.isfinite(swe_coarse_allvalid)
        region_coarse_mean_allvalid[name] = (
            float(np.nanmean(swe_coarse_allvalid[sel])) if np.any(sel) else float("nan")
        )

    # Fair, apples-to-apples version: restrict the block average to only the native
    # cells that are themselves region-assigned (native_region == code), matching
    # exactly what the native regional mean already does. This isolates the effect
    # of block-averaging itself from the effect of the region mask being sparse
    # (assigned only where WY2021 had >0.05 m active snow).
    region_trim = native_region[:trimmed_lat_size, :trimmed_lon_size]
    region_blocks = region_trim.reshape(n_lat_c, 8, n_lon_c, 8).transpose(0, 2, 1, 3).reshape(n_lat_c, n_lon_c, 64)

    region_coarse_mean_masked = {}
    for code, name in GROUP_LABEL.items():
        member = region_blocks == code
        masked_vals = np.where(member, swe_blocks, np.nan)
        with np.errstate(invalid="ignore"):
            block_mean = np.nanmean(masked_vals, axis=2)
        sel = np.isfinite(block_mean)
        region_coarse_mean_masked[name] = float(np.nanmean(block_mean[sel])) if np.any(sel) else float("nan")

    log(f"Native SWE units (attrs): {dict(swe_native_field.attrs)}")
    log("\nRegional mean April-1 SWE, native grid vs two coarsen-8 variants (units: meters):")
    log("  'coarse (all-valid-block)'  = repo's actual coarsen-8 convention: mean over ALL valid")
    log("     native cells in the 8x8 block, then select blocks by majority region label.")
    log("     This mixes in unassigned (low/zero-snow, not in the active WY2021 footprint)")
    log("     neighbor cells that share the block but were never region-assigned.")
    log("  'coarse (region-masked)'    = mean restricted to only region-assigned native cells")
    log("     within each block -- the fair, apples-to-apples comparison to the native mean.")
    consistency = {}
    for name in ("North", "Central", "South"):
        nat = region_native_mean[name]
        crs_all = region_coarse_mean_allvalid[name]
        crs_masked = region_coarse_mean_masked[name]
        diff_all = crs_all - nat
        diff_masked = crs_masked - nat
        pct_all = 100.0 * diff_all / nat if nat else float("nan")
        pct_masked = 100.0 * diff_masked / nat if nat else float("nan")
        log(f"  {name}: native={nat:.5f}  "
            f"coarse(all-valid-block)={crs_all:.5f} (diff {diff_all:+.5f}, {pct_all:+.2f}%)  "
            f"coarse(region-masked)={crs_masked:.5f} (diff {diff_masked:+.5f}, {pct_masked:+.2f}%)")
        consistency[name] = {
            "native_mean": nat,
            "coarsen8_all_valid_block_mean": crs_all,
            "coarsen8_all_valid_block_diff": diff_all,
            "coarsen8_all_valid_block_pct_diff": pct_all,
            "coarsen8_region_masked_mean": crs_masked,
            "coarsen8_region_masked_diff": diff_masked,
            "coarsen8_region_masked_pct_diff": pct_masked,
        }

    summary["swe_aggregation_consistency_wy%d" % REP_YEAR] = consistency

    # ------------------------------------------------------------------
    section("Elevation/orography available on each grid (reporting only, no lapse-rate decision)")
    # ------------------------------------------------------------------
    if Path(ERA5_LAND_GEOPOTENTIAL_FILE).exists():
        with xr.open_dataset(ERA5_LAND_GEOPOTENTIAL_FILE, decode_times=False) as ds_z:
            zvars = list(ds_z.data_vars)
            log(f"ERA5-Land geopotential file variables: {zvars}")
            grid_metadata["era5_land_orography_file"] = str(ERA5_LAND_GEOPOTENTIAL_FILE)
            grid_metadata["era5_land_orography_variables"] = zvars
    else:
        log(f"ERA5-Land geopotential file not found at {ERA5_LAND_GEOPOTENTIAL_FILE}")
    log("SWE-grid-side elevation: USGS NED DEM already cached at "
        "artifacts/swe_target_spatial_diagnostic/external_data/usgs_dem (from prior work); "
        "not reprocessed here. No lapse-rate correction is chosen in this task.")

    # ------------------------------------------------------------------
    log("\nWriting summary.json and grid_metadata.json")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "grid_metadata.json").write_text(json.dumps(grid_metadata, indent=2) + "\n")
    log("DONE.")


if __name__ == "__main__":
    main()
