"""
Precompute stage for the elevation mismatch audit (does not need the DEM):
  - reproduce the native-SWE-pixel -> ERA5-Land-cell binning exactly as used to
    build artifacts/snow17_era5land_working_grid/ (verified against it)
  - extract ERA5-Land elevation (from geopotential) on the same working grid
  - build a calibration-period-only (WY1985-2004) climatological April-1 SWE
    snow-importance field on the same working grid
"""
import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import xarray as xr

from snow_ml.data import (
    DEFAULT_SIERRA_REGION,
    SWE_VARIABLE,
    SWE_MISSING_VALUE,
    get_swe_grid_definition,
    swe_file_for_water_year,
    era5_land_yearly_file,
)
from config.paths import ERA5_LAND_GEOPOTENTIAL_FILE

WORKING_GRID_NPZ = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "snow17_era5land_working_grid/era5land_working_grid.npz"
)
KNN_MASK_NPZ = REPO_ROOT / "artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"
REP_YEAR = 2021
CAL_YEARS = list(range(1985, 2005))  # WY1985-2004 inclusive, calibration period only
GRAVITY = 9.80665

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_elevation_mismatch_audit")
for sub in ("plots", "logs"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def main():
    section("1. Reproduce native-pixel -> ERA5-Land-cell binning (verify against working grid)")
    grid1 = get_swe_grid_definition(water_year=REP_YEAR, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    native_lat = np.asarray(grid1.latitude.values, dtype=np.float64)
    native_lon = np.asarray(grid1.longitude.values, dtype=np.float64)

    knn = np.load(KNN_MASK_NPZ, allow_pickle=True)
    native_region = knn["assignment"].astype(np.float32)
    assert np.allclose(knn["lat"], native_lat, atol=1e-5) and np.allclose(knn["lon"], native_lon, atol=1e-5)

    tp_path = era5_land_yearly_file("tp", REP_YEAR)
    with xr.open_dataset(tp_path, decode_times=False) as ds_tp:
        era5_lat_full = np.asarray(ds_tp["latitude"].values, dtype=np.float64)
        era5_lon_full_raw = np.asarray(ds_tp["longitude"].values, dtype=np.float64)
    d_lat_era5 = era5_lat_full[1] - era5_lat_full[0]
    d_lon_era5_raw = era5_lon_full_raw[1] - era5_lon_full_raw[0]

    lat2d_native, lon2d_native = np.meshgrid(native_lat, native_lon, indexing="ij")
    era5_lat_idx = np.round((lat2d_native - era5_lat_full[0]) / d_lat_era5).astype(np.int64)
    lon2d_native_0_360 = np.where(lon2d_native < 0, lon2d_native + 360.0, lon2d_native)
    era5_lon_idx = np.round((lon2d_native_0_360 - era5_lon_full_raw[0]) / d_lon_era5_raw).astype(np.int64)

    in_bounds = (era5_lat_idx >= 0) & (era5_lat_idx < era5_lat_full.size) & \
                (era5_lon_idx >= 0) & (era5_lon_idx < era5_lon_full_raw.size)

    used_lat_idx = np.unique(era5_lat_idx[in_bounds])
    used_lon_idx = np.unique(era5_lon_idx[in_bounds])
    lat_lo, lat_hi = used_lat_idx.min(), used_lat_idx.max()
    lon_lo, lon_hi = used_lon_idx.min(), used_lon_idx.max()
    n_lat_e = lat_hi - lat_lo + 1
    n_lon_e = lon_hi - lon_lo + 1
    era5_sub_lat = era5_lat_full[lat_lo:lat_hi + 1]
    era5_sub_lon_raw = era5_lon_full_raw[lon_lo:lon_hi + 1]
    era5_sub_lon = np.where(era5_sub_lon_raw > 180.0, era5_sub_lon_raw - 360.0, era5_sub_lon_raw)

    rel_lat_idx = era5_lat_idx - lat_lo
    rel_lon_idx = era5_lon_idx - lon_lo
    flat_cell = np.where(in_bounds, rel_lat_idx * n_lon_e + rel_lon_idx, -1)

    # verify against the saved working grid
    wg = np.load(WORKING_GRID_NPZ, allow_pickle=True)
    shape_match = (n_lat_e, n_lon_e) == wg["region"].shape
    lat_match = np.allclose(era5_sub_lat, wg["lat"], atol=1e-6)
    lon_match = np.allclose(era5_sub_lon, wg["lon"], atol=1e-6)
    log(f"Reproduced binning shape {(n_lat_e, n_lon_e)} matches saved working grid: {shape_match}")
    log(f"Reproduced lat/lon match saved working grid: lat={lat_match}, lon={lon_match}")
    if not (shape_match and lat_match and lon_match):
        raise RuntimeError("Reproduced ERA5-Land binning does not match the saved working grid; aborting.")

    # rebuild the region mask the same way, confirm identical to saved
    from collections import Counter
    region_flat = native_region.ravel()
    cell_flat = flat_cell.ravel()
    valid_region_pix = (cell_flat >= 0) & np.isfinite(region_flat)
    era5_region = np.full(n_lat_e * n_lon_e, np.nan, dtype=np.float32)
    cells_with_region = np.unique(cell_flat[valid_region_pix])
    for cell in cells_with_region:
        members = region_flat[valid_region_pix][cell_flat[valid_region_pix] == cell]
        counts = Counter(members.tolist())
        top = max(counts.values())
        winners = sorted([k for k, v in counts.items() if v == top])
        era5_region[cell] = winners[0]
    era5_region = era5_region.reshape(n_lat_e, n_lon_e)
    saved_region = wg["region"]
    both_nan = np.isnan(era5_region) & np.isnan(saved_region)
    region_identical = bool(np.all(both_nan | (era5_region == saved_region)))
    log(f"Reproduced region mask identical to saved working grid: {region_identical}")
    if not region_identical:
        raise RuntimeError("Reproduced region mask does not match saved working grid; aborting.")

    section("2. ERA5-Land elevation from geopotential")
    log(f"Geopotential file: {ERA5_LAND_GEOPOTENTIAL_FILE}")
    with xr.open_dataset(ERA5_LAND_GEOPOTENTIAL_FILE, decode_times=False) as ds_z:
        zvar_name = [v for v in ds_z.data_vars][0]
        zvar = ds_z[zvar_name]
        log(f"Variable name: {zvar_name}, dims: {zvar.dims}, shape: {zvar.shape}")
        log(f"Attrs: {dict(zvar.attrs)}")
        z_lat_name = [c for c in ds_z.coords if "lat" in c.lower()][0]
        z_lon_name = [c for c in ds_z.coords if "lon" in c.lower()][0]
        z_lat_full = np.asarray(ds_z[z_lat_name].values, dtype=np.float64)
        z_lon_full_raw = np.asarray(ds_z[z_lon_name].values, dtype=np.float64)
        assert np.allclose(z_lat_full, era5_lat_full, atol=1e-4), "geopotential lat grid differs from tp/t2m grid"
        assert np.allclose(z_lon_full_raw, era5_lon_full_raw, atol=1e-4), "geopotential lon grid differs from tp/t2m grid"
        z_sel = zvar.isel(
            {zvar.dims[-2]: slice(lat_lo, lat_hi + 1), zvar.dims[-1]: slice(lon_lo, lon_hi + 1)}
        )
        z_vals = np.asarray(z_sel.values, dtype=np.float64)
        while z_vals.ndim > 2:
            z_vals = z_vals[0]

    units = str(zvar.attrs.get("units", "")).lower()
    is_geopotential = "m**2" in units or "m2 s-2" in units or "m2/s2" in units or units in ("m**2 s**-2",)
    log(f"Units field: '{zvar.attrs.get('units')}' -> interpreted as geopotential: {is_geopotential}")
    if is_geopotential or "s**-2" in units or units == "" or "m2" in units.replace(" ", ""):
        z_era5_m = z_vals / GRAVITY
        conversion = f"z_era5_m = geopotential / g, g={GRAVITY} m/s^2 (documented standard gravity)"
    else:
        z_era5_m = z_vals
        conversion = "used directly as elevation (units already meters)"
    log(f"Conversion applied: {conversion}")
    log(f"z_era5_m range on working grid: [{np.nanmin(z_era5_m):.1f}, {np.nanmax(z_era5_m):.1f}] m")

    section("3. Calibration-period-only (WY1985-2004) April-1 SWE climatology on the working grid")
    swe_sum = np.zeros((n_lat_e, n_lon_e), dtype=np.float64)
    swe_count = np.zeros((n_lat_e, n_lon_e), dtype=np.int64)
    years_used = []
    for year in CAL_YEARS:
        try:
            path = swe_file_for_water_year(year)
        except FileNotFoundError:
            log(f"  WY{year}: SWE file not found, skipping")
            continue
        with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
            swe_y = ds[SWE_VARIABLE].isel(Stats=0, drop=True).sel(time=np.datetime64(f"{year}-04-01"))
            swe_y = swe_y.sel({grid1.latitude_name: grid1.latitude, grid1.longitude_name: grid1.longitude}).load()
        vals = np.asarray(swe_y.where(swe_y != SWE_MISSING_VALUE).values, dtype=np.float64)
        finite = np.isfinite(vals) & np.isfinite(native_region)
        cell_idx = flat_cell[finite]
        v = vals[finite]
        np.add.at(swe_sum.ravel(), cell_idx, v)
        np.add.at(swe_count.ravel(), cell_idx, 1)
        years_used.append(year)
        log(f"  WY{year}: accumulated ({len(years_used)}/{len(CAL_YEARS)})")

    with np.errstate(invalid="ignore"):
        swe_clim = np.where(swe_count > 0, swe_sum / np.maximum(swe_count, 1), np.nan)
    log(f"Climatology built from {len(years_used)} calibration years: {years_used}")
    log(f"swe_clim finite cells: {int(np.isfinite(swe_clim).sum())} / {n_lat_e * n_lon_e}")

    section("Save precompute artifact")
    np.savez(
        OUT_DIR / "precompute.npz",
        lat=era5_sub_lat, lon=era5_sub_lon, lon_raw_0_360=era5_sub_lon_raw,
        region=era5_region, z_era5_m=z_era5_m, swe_clim_wy1985_2004=swe_clim,
        native_lat=native_lat, native_lon=native_lon, native_region=native_region,
        flat_cell=flat_cell, n_lat_e=n_lat_e, n_lon_e=n_lon_e,
    )
    (OUT_DIR / "precompute_meta.json").write_text(json.dumps({
        "gravity_constant": GRAVITY,
        "geopotential_conversion": conversion,
        "geopotential_units_attr": str(zvar.attrs.get("units")),
        "calibration_years_used": years_used,
    }, indent=2) + "\n")
    log(f"Saved: {OUT_DIR / 'precompute.npz'}")
    log("\nDONE.")


if __name__ == "__main__":
    main()
