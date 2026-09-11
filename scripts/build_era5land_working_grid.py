"""
Build the ERA5-Land-native (0.1 deg) Snow-17 working grid.

Discards coarsen-8 as the production grid (per grid_alignment_audit.py findings:
coarsen-8 is ~2.8x FINER than ERA5-Land, which would require upsampling ERA5-Land,
not coarsening SWE to match it). Instead:
  - ERA5-Land precipitation/temperature are used at their native 0.1 deg resolution,
    completely unmodified -- no interpolation of ERA5-Land is performed anywhere.
  - Fine native UCLA SWE is aggregated DOWNWARD onto the ERA5-Land grid (mean of
    native cells whose center falls inside each ERA5-Land 0.1x0.1 deg cell).
  - The finalized KNN-filled Sierra North/Central/South mask is aggregated the same
    way (majority vote of native labels per ERA5-Land cell).

Read-only with respect to all pre-existing artifacts; only writes to a new
artifact directory.
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

from snow_ml.data import (
    DEFAULT_SIERRA_REGION,
    SWE_VARIABLE,
    SWE_MISSING_VALUE,
    get_swe_grid_definition,
    swe_file_for_water_year,
    era5_land_yearly_file,
)

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_era5land_working_grid")
for sub in ("plots", "logs"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

KNN_MASK_NPZ = REPO_ROOT / "artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"
REP_YEAR = 2021
GROUP_LABEL = {1: "North", 2: "Central", 3: "South"}


def log(msg):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def main():
    summary = {}

    # ------------------------------------------------------------------
    section("1. Load native SWE grid + finalized KNN-filled Sierra mask")
    # ------------------------------------------------------------------
    grid1 = get_swe_grid_definition(water_year=REP_YEAR, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    native_lat = np.asarray(grid1.latitude.values, dtype=np.float64)
    native_lon = np.asarray(grid1.longitude.values, dtype=np.float64)
    log(f"Native SWE grid shape: {grid1.grid_shape}, lat range [{native_lat.min():.4f},{native_lat.max():.4f}], "
        f"lon range [{native_lon.min():.4f},{native_lon.max():.4f}]")

    knn = np.load(KNN_MASK_NPZ, allow_pickle=True)
    mask_lat = knn["lat"]
    mask_lon = knn["lon"]
    native_region = knn["assignment"].astype(np.float32)
    lat_match = mask_lat.shape == native_lat.shape and np.allclose(mask_lat, native_lat, atol=1e-5)
    lon_match = mask_lon.shape == native_lon.shape and np.allclose(mask_lon, native_lon, atol=1e-5)
    log(f"KNN-filled mask grid matches native SWE grid exactly: lat={lat_match}, lon={lon_match}")
    if not (lat_match and lon_match):
        raise RuntimeError("KNN-filled mask grid does not match native SWE grid.")

    with xr.open_dataset(swe_file_for_water_year(REP_YEAR), engine="netcdf4", decode_times=True) as ds:
        swe_field = ds[SWE_VARIABLE].isel(Stats=0, drop=True).sel(time=np.datetime64(f"{REP_YEAR}-04-01"))
        swe_field = swe_field.sel({grid1.latitude_name: grid1.latitude, grid1.longitude_name: grid1.longitude}).load()
    swe_native = np.asarray(swe_field.where(swe_field != SWE_MISSING_VALUE).values, dtype=np.float64)
    log(f"Native April-1 WY{REP_YEAR} SWE loaded, units={swe_field.attrs.get('units')}, shape={swe_native.shape}")

    # ------------------------------------------------------------------
    section("2. Load ERA5-Land native grid (precip file used only for its coordinates)")
    # ------------------------------------------------------------------
    tp_path = era5_land_yearly_file("tp", REP_YEAR)
    with xr.open_dataset(tp_path, decode_times=False) as ds_tp:
        era5_lat_full = np.asarray(ds_tp["latitude"].values, dtype=np.float64)
        era5_lon_full_raw = np.asarray(ds_tp["longitude"].values, dtype=np.float64)
    era5_lon_full = np.where(era5_lon_full_raw > 180.0, era5_lon_full_raw - 360.0, era5_lon_full_raw)

    d_lat_era5 = era5_lat_full[1] - era5_lat_full[0]
    d_lon_era5 = era5_lon_full[1] - era5_lon_full[0] if era5_lon_full[1] > era5_lon_full[0] else 0.1
    log(f"ERA5-Land full grid: lat {era5_lat_full.size} pts step {d_lat_era5:.5f}, "
        f"lon {era5_lon_full.size} pts step {d_lon_era5:.5f}")

    # ------------------------------------------------------------------
    section("3. Assign every native SWE pixel to its containing ERA5-Land cell")
    # ------------------------------------------------------------------
    lat2d_native, lon2d_native = np.meshgrid(native_lat, native_lon, indexing="ij")

    era5_lat_idx = np.round((lat2d_native - era5_lat_full[0]) / d_lat_era5).astype(np.int64)
    era5_lon_idx_raw = np.round((lon2d_native - era5_lon_full[0]) / (era5_lon_full[1] - era5_lon_full[0])).astype(np.int64)
    # lon index computed against unrotated/raw 0-360 array ordering; recompute using raw lon values directly
    lon2d_native_0_360 = np.where(lon2d_native < 0, lon2d_native + 360.0, lon2d_native)
    era5_lon_idx = np.round((lon2d_native_0_360 - era5_lon_full_raw[0]) / (era5_lon_full_raw[1] - era5_lon_full_raw[0])).astype(np.int64)

    lat_ok = (era5_lat_idx >= 0) & (era5_lat_idx < era5_lat_full.size)
    lon_ok = (era5_lon_idx >= 0) & (era5_lon_idx < era5_lon_full_raw.size)
    in_bounds = lat_ok & lon_ok
    log(f"Native pixels mapped in-bounds to an ERA5-Land cell: {int(in_bounds.sum())} / {in_bounds.size}")

    # restrict to a bounding box of used ERA5 indices to keep arrays small
    used_lat_idx = np.unique(era5_lat_idx[in_bounds])
    used_lon_idx = np.unique(era5_lon_idx[in_bounds])
    lat_lo, lat_hi = used_lat_idx.min(), used_lat_idx.max()
    lon_lo, lon_hi = used_lon_idx.min(), used_lon_idx.max()
    n_lat_e = lat_hi - lat_lo + 1
    n_lon_e = lon_hi - lon_lo + 1
    era5_sub_lat = era5_lat_full[lat_lo:lat_hi + 1]
    era5_sub_lon_raw = era5_lon_full_raw[lon_lo:lon_hi + 1]
    era5_sub_lon = era5_lon_full[lon_lo:lon_hi + 1]
    log(f"ERA5-Land working-grid subset: {n_lat_e} x {n_lon_e} cells "
        f"(lat idx [{lat_lo},{lat_hi}], lon idx [{lon_lo},{lon_hi}])")
    log(f"Subset lat range [{era5_sub_lat.min():.3f},{era5_sub_lat.max():.3f}], "
        f"lon range [{era5_sub_lon.min():.3f},{era5_sub_lon.max():.3f}]")

    rel_lat_idx = era5_lat_idx - lat_lo
    rel_lon_idx = era5_lon_idx - lon_lo

    # ------------------------------------------------------------------
    section("4. Occupancy: how many native pixels fall in each ERA5-Land cell")
    # ------------------------------------------------------------------
    flat_cell = np.where(in_bounds, rel_lat_idx * n_lon_e + rel_lon_idx, -1)
    valid_flat = flat_cell[flat_cell >= 0]
    occ_counts = np.bincount(valid_flat, minlength=n_lat_e * n_lon_e)
    used_cells = occ_counts[occ_counts > 0]
    log(f"ERA5-Land cells touched by >=1 native pixel: {int((occ_counts > 0).sum())} / {n_lat_e * n_lon_e}")
    log(f"Native pixels per touched ERA5-Land cell: min={used_cells.min()}, median={np.median(used_cells):.1f}, "
        f"mean={used_cells.mean():.1f}, max={used_cells.max()}")
    theoretical = (d_lat_era5 / np.median(np.diff(native_lat))) * (abs(era5_sub_lon_raw[1]-era5_sub_lon_raw[0]) / np.median(np.diff(native_lon)))
    log(f"Theoretical native-pixels-per-ERA5-cell (area ratio): ~{abs(theoretical):.1f}")

    # ------------------------------------------------------------------
    section("5. Aggregate native SWE onto the ERA5-Land grid (mean, skip missing)")
    # ------------------------------------------------------------------
    sum_swe = np.zeros(n_lat_e * n_lon_e, dtype=np.float64)
    count_swe = np.zeros(n_lat_e * n_lon_e, dtype=np.int64)
    finite_swe = np.isfinite(swe_native)
    valid_pix = in_bounds & finite_swe
    np.add.at(sum_swe, flat_cell[valid_pix], swe_native[valid_pix])
    np.add.at(count_swe, flat_cell[valid_pix], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        era5_swe = np.where(count_swe > 0, sum_swe / np.maximum(count_swe, 1), np.nan).reshape(n_lat_e, n_lon_e)
    log(f"ERA5-Land-grid SWE field built: {int(np.isfinite(era5_swe).sum())} cells with >=1 valid native SWE pixel")

    # ------------------------------------------------------------------
    section("6. Aggregate the finalized Sierra region mask onto the ERA5-Land grid (majority vote)")
    # ------------------------------------------------------------------
    era5_region = np.full(n_lat_e * n_lon_e, np.nan, dtype=np.float32)
    n_mixed = 0
    mixed_examples = []
    region_flat = native_region.ravel()
    cell_flat = flat_cell.ravel()
    valid_region_pix = (cell_flat >= 0) & np.isfinite(region_flat)
    cells_with_region = np.unique(cell_flat[valid_region_pix])
    for cell in cells_with_region:
        members = region_flat[valid_region_pix][cell_flat[valid_region_pix] == cell]
        counts = Counter(members.tolist())
        top = max(counts.values())
        winners = sorted([k for k, v in counts.items() if v == top])
        era5_region[cell] = winners[0]
        if len(counts) > 1:
            n_mixed += 1
            if len(mixed_examples) < 5:
                a, b = divmod(cell, n_lon_e)
                mixed_examples.append({
                    "lat": float(era5_sub_lat[a]), "lon": float(era5_sub_lon[b]),
                    "counts": {GROUP_LABEL[int(k)]: int(v) for k, v in counts.items()},
                })
    era5_region = era5_region.reshape(n_lat_e, n_lon_e)

    n_north = int(np.sum(era5_region == 1))
    n_central = int(np.sum(era5_region == 2))
    n_south = int(np.sum(era5_region == 3))
    n_outside = n_lat_e * n_lon_e - n_north - n_central - n_south
    log(f"ERA5-Land-grid region mask ({n_lat_e}x{n_lon_e}={n_lat_e*n_lon_e} cells): "
        f"North={n_north}, Central={n_central}, South={n_south}, Outside-Sierra(NaN)={n_outside}")
    log(f"Mixed-region ERA5 cells (majority-vote resolved): {n_mixed}")
    for ex in mixed_examples:
        log(f"  example mixed cell: {ex}")

    summary["era5land_region_mask"] = {
        "shape_lat_lon": [int(n_lat_e), int(n_lon_e)],
        "north_cells": n_north, "central_cells": n_central, "south_cells": n_south,
        "outside_sierra_cells": int(n_outside), "mixed_region_cells": n_mixed,
        "mixed_resolution": "majority vote among native cells in the ERA5-Land cell footprint; "
                             "exact ties broken by North<Central<South",
    }
    summary["native_pixels_per_era5_cell"] = {
        "min": int(used_cells.min()), "median": float(np.median(used_cells)),
        "mean": float(used_cells.mean()), "max": int(used_cells.max()),
        "theoretical_area_ratio": float(abs(theoretical)),
    }

    # ------------------------------------------------------------------
    section("7. Consistency check: native flat regional mean vs ERA5-grid regional mean, WY%d" % REP_YEAR)
    # ------------------------------------------------------------------
    region_native_mean = {}
    for code, name in GROUP_LABEL.items():
        sel = (native_region == code) & np.isfinite(swe_native)
        region_native_mean[name] = float(np.nanmean(swe_native[sel])) if np.any(sel) else float("nan")

    region_era5_mean = {}
    for code, name in GROUP_LABEL.items():
        sel = (era5_region == code) & np.isfinite(era5_swe)
        region_era5_mean[name] = float(np.nanmean(era5_swe[sel])) if np.any(sel) else float("nan")

    # Fair, apples-to-apples version: restrict the per-ERA5-cell SWE average to only
    # the native pixels that are themselves region-assigned (native_region == code),
    # exactly matching what the native regional mean does. Isolates the effect of
    # aggregation itself from the effect of averaging in unassigned neighbor pixels.
    region_era5_mean_masked = {}
    for code, name in GROUP_LABEL.items():
        sum_masked = np.zeros(n_lat_e * n_lon_e, dtype=np.float64)
        count_masked = np.zeros(n_lat_e * n_lon_e, dtype=np.int64)
        member_pix = in_bounds & finite_swe & (native_region == code)
        np.add.at(sum_masked, flat_cell[member_pix], swe_native[member_pix])
        np.add.at(count_masked, flat_cell[member_pix], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            block_mean = np.where(count_masked > 0, sum_masked / np.maximum(count_masked, 1), np.nan)
        sel = np.isfinite(block_mean)
        region_era5_mean_masked[name] = float(np.nanmean(block_mean[sel])) if np.any(sel) else float("nan")

    consistency = {}
    log("\nRegional mean April-1 SWE (meters): native flat mean vs two ERA5-Land-grid variants")
    log("  'era5-grid (all-valid-cell)' = mean over ALL valid native pixels in the ERA5 cell,")
    log("     then select cells by majority region label (mixes in unassigned/low-snow neighbors).")
    log("  'era5-grid (region-masked)'  = mean restricted to only region-assigned native pixels")
    log("     within each ERA5 cell -- the fair, apples-to-apples comparison to the native mean.")
    for name in ("North", "Central", "South"):
        nat = region_native_mean[name]
        e5_all = region_era5_mean[name]
        e5_masked = region_era5_mean_masked[name]
        diff_all = e5_all - nat
        diff_masked = e5_masked - nat
        pct_all = 100.0 * diff_all / nat if nat else float("nan")
        pct_masked = 100.0 * diff_masked / nat if nat else float("nan")
        log(f"  {name}: native={nat:.5f}  "
            f"era5-grid(all-valid)={e5_all:.5f} ({pct_all:+.2f}%)  "
            f"era5-grid(region-masked)={e5_masked:.5f} ({pct_masked:+.2f}%)")
        consistency[name] = {
            "native_mean": nat,
            "era5land_grid_all_valid_mean": e5_all, "era5land_grid_all_valid_pct_diff": pct_all,
            "era5land_grid_region_masked_mean": e5_masked, "era5land_grid_region_masked_pct_diff": pct_masked,
        }
    summary["swe_aggregation_consistency_wy%d" % REP_YEAR] = consistency

    # ------------------------------------------------------------------
    section("Save artifacts")
    # ------------------------------------------------------------------
    np.savez(
        OUT_DIR / "era5land_working_grid.npz",
        lat=era5_sub_lat, lon=era5_sub_lon, lon_raw_0_360=era5_sub_lon_raw,
        region=era5_region, swe_wy2021=era5_swe, occupancy=occ_counts.reshape(n_lat_e, n_lon_e),
    )
    import netCDF4
    with netCDF4.Dataset(OUT_DIR / "era5land_working_grid.nc", "w") as ds:
        ds.createDimension("lat", n_lat_e)
        ds.createDimension("lon", n_lon_e)
        la = ds.createVariable("lat", "f8", ("lat",)); la[:] = era5_sub_lat
        lo = ds.createVariable("lon", "f8", ("lon",)); lo[:] = era5_sub_lon
        rv = ds.createVariable("sierra_swe_region_label", "f4", ("lat", "lon"), fill_value=np.nan)
        rv[:, :] = era5_region
        rv.label_1 = "North"; rv.label_2 = "Central"; rv.label_3 = "South"
        sv = ds.createVariable("swe_wy2021_apr1", "f4", ("lat", "lon"), fill_value=np.nan)
        sv[:, :] = era5_swe.astype(np.float32)
        sv.units = "meters"
        ov = ds.createVariable("native_pixel_occupancy", "i4", ("lat", "lon"))
        ov[:, :] = occ_counts.reshape(n_lat_e, n_lon_e).astype(np.int32)
        ds.source = ("ERA5-Land-native (0.1 deg) Snow-17 working grid. Region label aggregated by majority "
                     "vote from artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz; "
                     "SWE aggregated by mean of native UCLA SWE pixels whose center falls in each ERA5-Land cell.")
    log(f"Saved: {OUT_DIR / 'era5land_working_grid.npz'} / .nc")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"Wrote summary: {OUT_DIR / 'summary.json'}")

    # plots
    lat2d_e, lon2d_e = np.meshgrid(era5_sub_lat, era5_sub_lon, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    colors = {1: "#1f78b4", 2: "#33a02c", 3: "#e31a1c"}
    labels = {1: "North", 2: "Central", 3: "South"}
    ax = axes[0]
    for code, color in colors.items():
        m = era5_region == code
        ax.scatter(lon2d_e[m], lat2d_e[m], s=60, color=color, label=labels[code], marker="s")
    mixed_mask = np.zeros_like(era5_region, dtype=bool)
    for cell in cells_with_region:
        members = region_flat[valid_region_pix][cell_flat[valid_region_pix] == cell]
        if len(set(members.tolist())) > 1:
            a, b = divmod(cell, n_lon_e)
            mixed_mask[a, b] = True
    ax.scatter(lon2d_e[mixed_mask], lat2d_e[mixed_mask], s=90, facecolors="none",
               edgecolors="black", linewidth=1.2, label=f"Mixed cell (n={n_mixed})", marker="o")
    ax.set_xlim(-123.0, -117.5); ax.set_ylim(34.5, 42.2)
    ax.set_title("ERA5-Land-native (0.1 deg) North/Central/South region mask")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(markerscale=1.0, loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax2 = axes[1]
    sc = ax2.pcolormesh(era5_sub_lon, era5_sub_lat, occ_counts.reshape(n_lat_e, n_lon_e), shading="auto", cmap="viridis")
    ax2.set_xlim(-123.0, -117.5); ax2.set_ylim(34.5, 42.2)
    ax2.set_title("Native SWE pixels per ERA5-Land cell (occupancy)")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax2, label="native pixel count")
    ax2.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(OUT_DIR / "plots" / "era5land_working_grid.png", dpi=200)
    log(f"Saved plot: {OUT_DIR / 'plots' / 'era5land_working_grid.png'}")

    log("\nDONE.")


if __name__ == "__main__":
    main()
