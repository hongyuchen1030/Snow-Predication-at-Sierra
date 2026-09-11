"""
Elevation mismatch audit: ERA5-Land orography vs. USGS NED DEM Sierra terrain,
on the finalized ERA5-Land-native (0.1 deg) Snow-17 working grid.

Pure diagnostic. Does not run Snow-17, does not run SCE-UA, does not touch
ACE2, does not choose or apply any lapse rate or temperature correction.
"""
import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_elevation_mismatch_audit")
for sub in ("plots", "logs"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

DEM_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "swe_target_spatial_diagnostic/external_data/usgs_dem"
)
GROUP_LABEL = {1: "North", 2: "Central", 3: "South"}
LAPSE_RATES_K_PER_KM = [4.0, 5.0, 6.0, 6.5]
ABS_THRESHOLDS_M = [100, 250, 500, 750, 1000]


def log(msg):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def stat_block(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"count": 0}
    return {
        "count": int(v.size),
        "min": float(np.min(v)), "max": float(np.max(v)),
        "mean": float(np.mean(v)), "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "mean_abs": float(np.mean(np.abs(v))), "median_abs": float(np.median(np.abs(v))),
    }


def threshold_fractions(values: np.ndarray) -> dict:
    v = np.abs(values[np.isfinite(values)])
    if v.size == 0:
        return {}
    out = {}
    for t in ABS_THRESHOLDS_M:
        out[f"lt_{t}m"] = float(np.mean(v < t))
    out["ge_1000m"] = float(np.mean(v >= 1000))
    return out


def main():
    section("0. Load precompute artifact")
    pre = np.load(OUT_DIR / "precompute.npz", allow_pickle=True)
    era5_lat = pre["lat"]; era5_lon = pre["lon"]
    region = pre["region"]
    z_era5_m = pre["z_era5_m"]
    swe_clim = pre["swe_clim_wy1985_2004"]
    native_lat = pre["native_lat"]; native_lon = pre["native_lon"]
    native_region = pre["native_region"]
    flat_cell = pre["flat_cell"]
    n_lat_e = int(pre["n_lat_e"]); n_lon_e = int(pre["n_lon_e"])
    log(f"Working grid: {n_lat_e} x {n_lon_e}. Sierra-assigned ERA5 cells: {int(np.isfinite(region).sum())}")

    section("1. Load DEM tiles and point-sample at every region-assigned native SWE pixel")
    dem_paths = sorted(DEM_DIR.glob("*.tif")) + sorted(DEM_DIR.glob("extracted_*/*.tif"))
    dem_paths = sorted(set(dem_paths))
    log(f"DEM tiles found: {len(dem_paths)}")
    for p in dem_paths:
        log(f"  {p}")

    lat2d_native, lon2d_native = np.meshgrid(native_lat, native_lon, indexing="ij")
    finite_region = np.isfinite(native_region)
    pix_lat = lat2d_native[finite_region]
    pix_lon = lon2d_native[finite_region]
    pix_cell = flat_cell[finite_region]
    pix_region = native_region[finite_region]
    n_pix = pix_lat.size
    log(f"Region-assigned native SWE pixels to sample: {n_pix}")

    dem_values = np.full(n_pix, np.nan, dtype=np.float64)
    remaining = np.ones(n_pix, dtype=bool)
    tile_coverage_count = 0
    for path in dem_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            in_tile = remaining & (pix_lon >= b.left) & (pix_lon <= b.right) & \
                      (pix_lat >= b.bottom) & (pix_lat <= b.top)
            n_in_tile = int(in_tile.sum())
            if n_in_tile == 0:
                continue
            coords = list(zip(pix_lon[in_tile], pix_lat[in_tile]))
            sampled = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
            nodata = src.nodata
            if nodata is not None:
                sampled = np.where(sampled == nodata, np.nan, sampled)
            dem_values[np.where(in_tile)[0] if False else np.flatnonzero(in_tile)] = sampled
            remaining[in_tile] = False
            tile_coverage_count += n_in_tile
            log(f"  {path.name}: bounds=({b.left:.2f},{b.bottom:.2f},{b.right:.2f},{b.top:.2f}) "
                f"sampled {n_in_tile} pixels, {int(remaining.sum())} remaining overall")
        if not remaining.any():
            break

    n_unsampled = int(remaining.sum())
    log(f"Total pixels sampled by a DEM tile: {tile_coverage_count} / {n_pix}")
    log(f"Pixels with no covering DEM tile (left NaN): {n_unsampled}")

    section("2. Aggregate DEM elevation onto the ERA5-Land working grid (region-assigned pixels only)")
    z_sierra_dem_m = np.full(n_lat_e * n_lon_e, np.nan, dtype=np.float64)
    valid_dem = np.isfinite(dem_values)
    sum_z = np.zeros(n_lat_e * n_lon_e, dtype=np.float64)
    cnt_z = np.zeros(n_lat_e * n_lon_e, dtype=np.int64)
    np.add.at(sum_z, pix_cell[valid_dem], dem_values[valid_dem])
    np.add.at(cnt_z, pix_cell[valid_dem], 1)
    with np.errstate(invalid="ignore"):
        z_sierra_dem_m = np.where(cnt_z > 0, sum_z / np.maximum(cnt_z, 1), np.nan).reshape(n_lat_e, n_lon_e)
    log("z_sierra_dem_m method: simple (unweighted) mean of single-nearest-DEM-pixel point samples "
        "taken at each region-assigned native SWE pixel's exact center; NOT an area-weighted window mean. "
        "Justification: the DEM (1 arc-second, ~30 m) already resolves terrain at ~13x finer linear "
        "spacing than the native SWE pixel (~0.0044 deg, ~490 m), so a single point sample per SWE pixel "
        "center already reflects the local terrain far more finely than the SWE/region grid itself; only "
        "unrelated (non-Sierra) terrain is excluded by construction, per the native-pixel region mask.")
    log(f"z_sierra_dem_m finite cells: {int(np.isfinite(z_sierra_dem_m).sum())} (region-assigned cells: "
        f"{int(np.isfinite(region).sum())})")

    section("3. Compute elevation mismatch dz = z_sierra_dem_m - z_era5_m")
    dz = z_sierra_dem_m - z_era5_m
    dz_valid_mask = np.isfinite(region) & np.isfinite(dz)
    log(f"Sierra ERA5 cells with a valid dz: {int(dz_valid_mask.sum())} / {int(np.isfinite(region).sum())}")

    elevation_stats = {}
    domains = {"Full_Sierra": dz_valid_mask}
    for code, name in GROUP_LABEL.items():
        domains[name] = dz_valid_mask & (region == code)
    for name, mask in domains.items():
        vals = dz[mask]
        elevation_stats[name] = stat_block(vals)
        elevation_stats[name]["threshold_fractions"] = threshold_fractions(vals)
        log(f"\n{name} (n={elevation_stats[name].get('count', 0)}):")
        log(json.dumps(elevation_stats[name], indent=2))

    section("4. Lapse-rate scale diagnostic (DIAGNOSTIC ONLY -- no correction applied/chosen)")
    lapse_diag = {}
    full_abs_dz_km = np.abs(dz[dz_valid_mask]) / 1000.0
    for lapse in LAPSE_RATES_K_PER_KM:
        dt = lapse * full_abs_dz_km
        lapse_diag[f"{lapse}_K_per_km"] = {
            "median_abs_dT_K": float(np.median(dt)),
            "mean_abs_dT_K": float(np.mean(dt)),
            "p90_abs_dT_K": float(np.percentile(dt, 90)),
            "p95_abs_dT_K": float(np.percentile(dt, 95)),
            "max_abs_dT_K": float(np.max(dt)),
        }
        log(f"  lapse={lapse} K/km: {lapse_diag[f'{lapse}_K_per_km']}")

    section("5. Snow-importance diagnostic (WY1985-2004 calibration-period climatology only)")
    clim_mask = dz_valid_mask & np.isfinite(swe_clim)
    dz_c = dz[clim_mask]
    swe_c = swe_clim[clim_mask]
    abs_dz_c = np.abs(dz_c)
    corr = float(np.corrcoef(abs_dz_c, swe_c)[0, 1]) if abs_dz_c.size > 2 else float("nan")
    log(f"Cells with both valid dz and calibration-period SWE climatology: {int(clim_mask.sum())}")
    log(f"Pearson correlation(|dz|, WY1985-2004 climatological April-1 SWE): {corr:.4f}")

    snow_importance = {"n_cells": int(clim_mask.sum()), "pearson_corr_abs_dz_vs_clim_swe": corr}
    for frac, label in [(0.25, "top25pct_swe_cells"), (0.10, "top10pct_swe_cells")]:
        n_top = max(1, int(round(frac * swe_c.size)))
        top_idx = np.argsort(swe_c)[-n_top:]
        stats_top = stat_block(dz_c[top_idx])
        snow_importance[label] = stats_top
        log(f"\n{label} (n={n_top}): {json.dumps(stats_top, indent=2)}")

    section("6. ERA5-Land vs DEM smoothing check (no assumed error direction)")
    era5_vals = z_era5_m[dz_valid_mask]
    dem_vals = z_sierra_dem_m[dz_valid_mask]
    smoothing_note = {
        "era5_land_mean_m": float(np.mean(era5_vals)),
        "dem_mean_m": float(np.mean(dem_vals)),
        "era5_land_std_m": float(np.std(era5_vals)),
        "dem_std_m": float(np.std(dem_vals)),
        "era5_land_max_m": float(np.max(era5_vals)),
        "dem_max_m": float(np.max(dem_vals)),
        "interpretation": (
            "If DEM std/max exceed ERA5-Land std/max substantially, ERA5-Land's coarse 0.1 deg "
            "orography is smoothing out real Sierra relief (a known, expected property of any "
            "coarse-grid orography field, not evidence that ERA5-Land is 'wrong' per se)."
        ),
    }
    log(json.dumps(smoothing_note, indent=2))

    section("7. Save artifacts")
    summary = {
        "elevation_mismatch_stats_by_domain": elevation_stats,
        "lapse_rate_scale_diagnostic": lapse_diag,
        "snow_importance_diagnostic": snow_importance,
        "era5_vs_dem_smoothing_check": smoothing_note,
        "n_native_pixels_sampled": int(tile_coverage_count),
        "n_native_pixels_unsampled_no_dem_tile": n_unsampled,
    }
    (OUT_DIR / "elevation_stats.json").write_text(json.dumps(elevation_stats, indent=2) + "\n")
    (OUT_DIR / "lapse_rate_scale_diagnostic.json").write_text(json.dumps(lapse_diag, indent=2) + "\n")
    (OUT_DIR / "snow_importance_diagnostic.json").write_text(json.dumps(snow_importance, indent=2) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    np.savez(
        OUT_DIR / "elevation_comparison.npz",
        lat=era5_lat, lon=era5_lon, region=region,
        z_era5_m=z_era5_m, z_sierra_dem_m=z_sierra_dem_m, dz=dz,
        swe_clim_wy1985_2004=swe_clim,
    )
    import netCDF4
    with netCDF4.Dataset(OUT_DIR / "elevation_comparison.nc", "w") as ds:
        ds.createDimension("lat", n_lat_e); ds.createDimension("lon", n_lon_e)
        la = ds.createVariable("lat", "f8", ("lat",)); la[:] = era5_lat
        lo = ds.createVariable("lon", "f8", ("lon",)); lo[:] = era5_lon
        rv = ds.createVariable("sierra_swe_region_label", "f4", ("lat", "lon"), fill_value=np.nan); rv[:, :] = region
        rv.label_1 = "North"; rv.label_2 = "Central"; rv.label_3 = "South"
        e5v = ds.createVariable("z_era5_m", "f4", ("lat", "lon"), fill_value=np.nan); e5v[:, :] = z_era5_m
        e5v.units = "m"; e5v.source = "ERA5_geopotential.nc / 9.80665"
        demv = ds.createVariable("z_sierra_dem_m", "f4", ("lat", "lon"), fill_value=np.nan); demv[:, :] = z_sierra_dem_m
        demv.units = "m"; demv.source = "USGS NED 1 arc-second, point-sampled at region-assigned native SWE pixel centers"
        dzv = ds.createVariable("dz_m", "f4", ("lat", "lon"), fill_value=np.nan); dzv[:, :] = dz
        dzv.units = "m"; dzv.description = "z_sierra_dem_m - z_era5_m"
        swv = ds.createVariable("swe_clim_wy1985_2004_m", "f4", ("lat", "lon"), fill_value=np.nan); swv[:, :] = swe_clim
    log(f"Saved elevation_comparison.npz / .nc / *.json to {OUT_DIR}")

    section("8. Plots")
    lat2d_e, lon2d_e = np.meshgrid(era5_lat, era5_lon, indexing="ij")

    def add_region_boundaries(ax):
        for code, color in {1: "#1f78b4", 2: "#33a02c", 3: "#e31a1c"}.items():
            mask = (region == code).astype(float)
            if mask.max() > 0:
                ax.contour(lon2d_e, lat2d_e, mask, levels=[0.5], colors=color, linewidths=1.2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    ax = axes[0, 0]
    m0 = ax.pcolormesh(era5_lon, era5_lat, np.where(np.isfinite(region), z_era5_m, np.nan),
                        shading="auto", cmap="terrain")
    fig.colorbar(m0, ax=ax, label="m")
    add_region_boundaries(ax)
    ax.set_title("ERA5-Land elevation (geopotential/g)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    ax = axes[0, 1]
    m1 = ax.pcolormesh(era5_lon, era5_lat, z_sierra_dem_m, shading="auto", cmap="terrain")
    fig.colorbar(m1, ax=ax, label="m")
    add_region_boundaries(ax)
    ax.set_title("DEM-derived Sierra elevation (region-assigned native pixels)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    ax = axes[1, 0]
    vmax = np.nanpercentile(np.abs(dz), 98) if np.isfinite(dz).any() else 1.0
    m2 = ax.pcolormesh(era5_lon, era5_lat, dz, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(m2, ax=ax, label="m (DEM - ERA5-Land)")
    add_region_boundaries(ax)
    ax.set_title("Signed elevation mismatch dz")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    ax = axes[1, 1]
    m3 = ax.pcolormesh(era5_lon, era5_lat, np.abs(dz), shading="auto", cmap="magma_r")
    fig.colorbar(m3, ax=ax, label="m")
    add_region_boundaries(ax)
    ax.set_title("Absolute elevation mismatch |dz|")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    for ax in axes.ravel():
        ax.set_xlim(-123.0, -117.5); ax.set_ylim(34.5, 42.2)
        ax.grid(True, linestyle=":", alpha=0.3)

    fig.suptitle("Elevation mismatch audit: ERA5-Land vs USGS NED DEM over the Sierra working grid", fontsize=13)
    fig.savefig(OUT_DIR / "plots" / "elevation_mismatch_maps.png", dpi=200)
    log(f"Saved: {OUT_DIR / 'plots' / 'elevation_mismatch_maps.png'}")

    # scatter: |dz| vs climatological SWE
    fig2, ax2 = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = {1: "#1f78b4", 2: "#33a02c", 3: "#e31a1c"}
    for code, name in GROUP_LABEL.items():
        sel = clim_mask & (region == code)
        ax2.scatter(swe_clim[sel], np.abs(dz)[sel], s=20, color=colors[code], label=name, alpha=0.7)
    ax2.set_xlabel("WY1985-2004 climatological April-1 SWE (m)")
    ax2.set_ylabel("|dz| (m)")
    ax2.set_title(f"Elevation mismatch vs. calibration-period snow importance\nPearson r = {corr:.3f}")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.4)
    fig2.savefig(OUT_DIR / "plots" / "abs_dz_vs_snow_importance.png", dpi=200)
    log(f"Saved: {OUT_DIR / 'plots' / 'abs_dz_vs_snow_importance.png'}")

    log("\nDONE.")


if __name__ == "__main__":
    main()
