"""
Stage 1 forcing cache: parallelized ERA5-Land tp/t2m extraction + reset-aware
daily aggregation + -4.5 K/km elevation correction, for the full span needed
by calibration (WY1985-2004) + held-out evaluation (WY2005-2021), preceded
by a 1-year spin-up (WY1984, i.e. starting 1983-10-01).

Years needed: 1983-2021 inclusive (39 years).

This reuses the EXACT reset-aware tp logic and elevation-correction equation
already validated in scripts/snow17_smoke_test_wy1985.py, only parallelized
across years (I/O-bound, independent per-year files) so SCE-UA candidate
evaluations never touch the raw ERA5-Land files again.
"""
import sys, os, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import xarray as xr

from snow_ml.data import era5_land_yearly_file

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_stage1_scua_validation")
CACHE_DIR = OUT_DIR / "forcing_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WORKING_GRID_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_era5land_working_grid/era5land_working_grid.npz")
ELEVATION_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_elevation_mismatch_audit/elevation_comparison.npz")

LAPSE_RATE_K_PER_KM = -4.5
YEARS = list(range(1983, 2022))  # 1983-2021 inclusive
N_WORKERS = 8


def log(msg):
    print(msg, flush=True)


def read_one_year(args):
    year, lat_lo, n_lat_e, lon_lo, n_lon_e = args
    tp_path = era5_land_yearly_file("tp", year)
    t2m_path = era5_land_yearly_file("t2m", year)
    with xr.open_dataset(tp_path, decode_times=False) as ds_tp:
        time_name = [c for c in ds_tp.coords if "time" in c.lower()][0]
        tp_box = ds_tp["tp"].isel(latitude=slice(lat_lo, lat_lo + n_lat_e),
                                   longitude=slice(lon_lo, lon_lo + n_lon_e)).load()
        time_hours = pd.to_datetime(ds_tp[time_name].values, unit="h", origin=pd.Timestamp("1900-01-01"))
    with xr.open_dataset(t2m_path, decode_times=False) as ds_t2m:
        t2m_box = ds_t2m["t2m"].isel(latitude=slice(lat_lo, lat_lo + n_lat_e),
                                      longitude=slice(lon_lo, lon_lo + n_lon_e)).load()
    return year, time_hours.values, np.asarray(tp_box.values, dtype=np.float32), np.asarray(t2m_box.values, dtype=np.float32)


def main():
    t0 = time.time()
    wg = np.load(WORKING_GRID_NPZ, allow_pickle=True)
    era5_lat = wg["lat"]; era5_lon = wg["lon"]
    region = wg["region"]
    n_lat_e, n_lon_e = region.shape
    log(f"Working grid: {n_lat_e} x {n_lon_e}, Sierra cells: {int(np.isfinite(region).sum())}")

    ez = np.load(ELEVATION_NPZ, allow_pickle=True)
    assert np.allclose(ez["lat"], era5_lat, atol=1e-6) and np.allclose(ez["lon"], era5_lon, atol=1e-6)
    dz = ez["dz"]

    with xr.open_dataset(era5_land_yearly_file("tp", 2021), decode_times=False) as ds_tp:
        full_era5_lat = np.asarray(ds_tp["latitude"].values, dtype=np.float64)
        full_era5_lon_raw = np.asarray(ds_tp["longitude"].values, dtype=np.float64)
    d_lat_era5 = full_era5_lat[1] - full_era5_lat[0]
    d_lon_era5_raw = full_era5_lon_raw[1] - full_era5_lon_raw[0]
    lat_lo = int(np.round((era5_lat[0] - full_era5_lat[0]) / d_lat_era5))
    era5_lon_raw0 = era5_lon[0] if era5_lon[0] >= 0 else era5_lon[0] + 360.0
    lon_lo = int(np.round((era5_lon_raw0 - full_era5_lon_raw[0]) / d_lon_era5_raw))
    log(f"Window: lat0_idx={lat_lo}, lon0_idx={lon_lo}, shape=({n_lat_e},{n_lon_e})")

    log(f"Reading {len(YEARS)} years with {N_WORKERS} parallel workers...")
    jobs = [(y, lat_lo, n_lat_e, lon_lo, n_lon_e) for y in YEARS]
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(read_one_year, j): j[0] for j in jobs}
        n_done = 0
        for fut in as_completed(futs):
            year, times, tp, t2m = fut.result()
            results[year] = (times, tp, t2m)
            n_done += 1
            log(f"  [{n_done}/{len(YEARS)}] year {year} done ({time.time()-t0:.0f}s elapsed)")

    log(f"All years read in {time.time()-t0:.0f}s. Concatenating...")
    all_times = np.concatenate([results[y][0] for y in YEARS])
    all_tp = np.concatenate([results[y][1] for y in YEARS], axis=0)
    all_t2m = np.concatenate([results[y][2] for y in YEARS], axis=0)
    del results

    all_times = pd.DatetimeIndex(all_times)
    order = np.argsort(all_times.values)
    all_times = all_times[order]; all_tp = all_tp[order]; all_t2m = all_t2m[order]
    _, uniq_idx = np.unique(all_times.values, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    all_times = all_times[uniq_idx]; all_tp = all_tp[uniq_idx]; all_t2m = all_t2m[uniq_idx]
    log(f"Combined hourly array: {all_tp.shape}, {all_times.min()} to {all_times.max()}")

    is_midnight = all_times.hour == 0
    midnight_times = all_times[is_midnight]
    daily_index = midnight_times - pd.Timedelta(days=1)
    daily_precip_mm = all_tp[is_midnight] * 1000.0

    daily_group = all_times.floor("D")
    uniq_days = pd.DatetimeIndex(sorted(set(daily_group)))
    day_pos = {d: i for i, d in enumerate(uniq_days)}
    counts = np.zeros(len(uniq_days))
    sums = np.zeros((len(uniq_days), n_lat_e, n_lon_e))
    day_idx_arr = np.array([day_pos[d] for d in daily_group])
    np.add.at(sums, day_idx_arr, all_t2m)
    np.add.at(counts, day_idx_arr, 1)
    t2m_daily_C = sums / counts[:, None, None] - 273.15

    precip_df_index = pd.DatetimeIndex(daily_index)
    common_index = precip_df_index.intersection(uniq_days).sort_values()
    precip_pos = {d: i for i, d in enumerate(precip_df_index)}
    tair_pos = {d: i for i, d in enumerate(uniq_days)}
    precip_arr = np.stack([daily_precip_mm[precip_pos[d]] for d in common_index])
    tair_arr = np.stack([t2m_daily_C[tair_pos[d]] for d in common_index])
    log(f"Aligned daily series: {len(common_index)} days, {common_index.min()} to {common_index.max()}")

    assert np.all(precip_arr[np.isfinite(precip_arr)] >= -1e-6), "negative precipitation found"
    sierra_ii, sierra_jj = np.where(np.isfinite(region))
    log(f"Precip (mm/day) range over Sierra cells: "
        f"[{precip_arr[:, sierra_ii, sierra_jj].min():.3f}, {precip_arr[:, sierra_ii, sierra_jj].max():.3f}]")
    log(f"Uncorrected T (degC) range over Sierra cells: "
        f"[{tair_arr[:, sierra_ii, sierra_jj].min():.2f}, {tair_arr[:, sierra_ii, sierra_jj].max():.2f}]")

    tair_corrected = tair_arr + (LAPSE_RATE_K_PER_KM / 1000.0) * dz[None, :, :]
    log(f"Corrected T (degC) range over Sierra cells: "
        f"[{tair_corrected[:, sierra_ii, sierra_jj].min():.2f}, {tair_corrected[:, sierra_ii, sierra_jj].max():.2f}]")

    out_path = CACHE_DIR / "daily_forcing_1983_2021.npz"
    np.savez_compressed(
        out_path,
        common_index=common_index.values.astype("datetime64[ns]"),
        precip_mm=precip_arr.astype(np.float32),
        tair_corrected_degC=tair_corrected.astype(np.float32),
        era5_lat=era5_lat, era5_lon=era5_lon, region=region,
        lapse_rate_K_per_km=LAPSE_RATE_K_PER_KM,
    )
    log(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
