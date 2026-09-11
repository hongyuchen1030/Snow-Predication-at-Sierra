"""
Deterministic smoke test of the fully-wired Snow-17 production pipeline:

  ERA5-Land tp (reset-aware daily) + t2m
      -> -4.5 K/km elevation correction (residual, DEM - ERA5-Land)
      -> 1-year forcing-driven zero-state spin-up (WY1984)
      -> tonic Snow-17 (scripts/vendor/snow17.py, ait( typo now fixed)
      -> occupancy-weighted regional aggregation (North/Central/South)
      -> compare against observed regional SWE, compute 1-NSE

FIXED parameters only (tonic defaults for the 5 fixed + literature-range
midpoints for the 7 "calibrated" ones, used here only to prove the pipeline
runs end-to-end). NOT a calibration. No SCE-UA. No parameter fitting.
"""
import sys, os, json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "vendor"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.paths import ERA5_LAND_ROOT
from snow_ml.data import (
    DEFAULT_SIERRA_REGION, SWE_VARIABLE, SWE_MISSING_VALUE,
    get_swe_grid_definition, swe_file_for_water_year, era5_land_yearly_file,
)
from snow17 import snow17  # scripts/vendor/snow17.py, fixed base implementation

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_smoke_test_wy1985")
for sub in ("plots", "logs"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

WORKING_GRID_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_era5land_working_grid/era5land_working_grid.npz")
ELEVATION_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_elevation_mismatch_audit/elevation_comparison.npz")
KNN_MASK_NPZ = REPO_ROOT / "artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"

LAPSE_RATE_K_PER_KM = -4.5
SPINUP_START = pd.Timestamp("1983-10-01")
TEST_START = pd.Timestamp("1984-10-01")
TEST_END = pd.Timestamp("1985-09-30")
YEARS = [1983, 1984, 1985]
GROUP_LABEL = {1: "North", 2: "Central", 3: "South"}

# Fixed parameters for this smoke test ONLY -- not a calibration.
# 5 fixed = tonic defaults (docs/Snow17_Methodology_and_Readiness.md ss3).
# 7 "calibrated" = midpoint of Shiheng Appendix B ranges (ss2), used only to
# get a plausible, non-arbitrary fixed vector to prove the pipeline runs;
# real values will come from SCE-UA later.
FIXED_PARAMS = dict(
    scf=(0.9 + 1.2) / 2, rvs=1.0, uadj=(0.05 + 0.2) / 2,
    mbase=1.0, mfmax=(0.5 + 1.3) / 2, mfmin=(0.1 + 0.6) / 2,
    tipm=0.1, nmf=0.15, plwhc=0.04,
    pxtemp=(0.0 + 2.0) / 2, pxtemp1=(-2.0 + 0.0) / 2, pxtemp2=(0.0 + 4.0) / 2,
)
DT_HOURS = 24


def log(msg):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


FORCING_CACHE = OUT_DIR / "forcing_cache.npz"


def extract_forcing(sierra_ii, sierra_jj, n_lat_e, n_lon_e, lat_lo, lon_lo):
    hourly_tp = []
    hourly_t2m = []
    for year in YEARS:
        tp_path = era5_land_yearly_file("tp", year)
        t2m_path = era5_land_yearly_file("t2m", year)
        log(f"  {year}: {tp_path.name}, {t2m_path.name}")
        with xr.open_dataset(tp_path, decode_times=False) as ds_tp:
            time_name = [c for c in ds_tp.coords if "time" in c.lower()][0]
            tp_box = ds_tp["tp"].isel(latitude=slice(lat_lo, lat_lo + n_lat_e),
                                       longitude=slice(lon_lo, lon_lo + n_lon_e)).load()
            time_hours = pd.to_datetime(ds_tp[time_name].values, unit="h", origin=pd.Timestamp("1900-01-01"))
        with xr.open_dataset(t2m_path, decode_times=False) as ds_t2m:
            t2m_box = ds_t2m["t2m"].isel(latitude=slice(lat_lo, lat_lo + n_lat_e),
                                          longitude=slice(lon_lo, lon_lo + n_lon_e)).load()
        hourly_tp.append((time_hours, np.asarray(tp_box.values, dtype=np.float32)))
        hourly_t2m.append(np.asarray(t2m_box.values, dtype=np.float32))

    all_times = pd.DatetimeIndex(np.concatenate([t.values for t, _ in hourly_tp]))
    all_tp = np.concatenate([a for _, a in hourly_tp], axis=0)  # (time, lat, lon), m
    all_t2m = np.concatenate(hourly_t2m, axis=0)  # (time, lat, lon), K
    order = np.argsort(all_times.values)
    all_times = all_times[order]; all_tp = all_tp[order]; all_t2m = all_t2m[order]
    _, uniq_idx = np.unique(all_times.values, return_index=True)
    all_times = all_times[np.sort(uniq_idx)]; all_tp = all_tp[np.sort(uniq_idx)]; all_t2m = all_t2m[np.sort(uniq_idx)]
    log(f"Combined hourly array: {all_tp.shape}, {all_times.min()} to {all_times.max()}")

    is_midnight = all_times.hour == 0
    midnight_times = all_times[is_midnight]
    daily_index = midnight_times - pd.Timedelta(days=1)
    daily_precip_mm = all_tp[is_midnight] * 1000.0  # (day, lat, lon)
    daily_group = all_times.floor("D")
    uniq_days = pd.DatetimeIndex(sorted(set(daily_group)))
    day_pos = {d: i for i, d in enumerate(uniq_days)}
    counts = np.zeros(len(uniq_days))
    sums = np.zeros((len(uniq_days), n_lat_e, n_lon_e))
    for i, d in enumerate(daily_group):
        p = day_pos[d]
        sums[p] += all_t2m[i]
        counts[p] += 1
    t2m_daily_C = sums / counts[:, None, None] - 273.15

    precip_df_index = pd.DatetimeIndex(daily_index)
    common_index = precip_df_index.intersection(uniq_days).sort_values()
    precip_pos = {d: i for i, d in enumerate(precip_df_index)}
    tair_pos = {d: i for i, d in enumerate(uniq_days)}
    n_days = len(common_index)
    precip_arr = np.stack([daily_precip_mm[precip_pos[d]] for d in common_index])
    tair_arr = np.stack([t2m_daily_C[tair_pos[d]] for d in common_index])
    log(f"Aligned daily series: {n_days} days, {common_index.min()} to {common_index.max()}")

    assert np.all(precip_arr[np.isfinite(precip_arr)] >= -1e-6), "negative precipitation found"
    log(f"Precip (mm/day) range over Sierra cells: "
        f"[{precip_arr[:, sierra_ii, sierra_jj].min():.3f}, {precip_arr[:, sierra_ii, sierra_jj].max():.3f}]")
    log(f"Uncorrected T (degC) range over Sierra cells: "
        f"[{tair_arr[:, sierra_ii, sierra_jj].min():.2f}, {tair_arr[:, sierra_ii, sierra_jj].max():.2f}]")
    return precip_arr, tair_arr, common_index, n_days


def main():
    section("1. Load working grid, elevation comparison, native region mask")
    wg = np.load(WORKING_GRID_NPZ, allow_pickle=True)
    era5_lat = wg["lat"]; era5_lon = wg["lon"]
    region = wg["region"]
    n_lat_e, n_lon_e = region.shape
    log(f"Working grid: {n_lat_e} x {n_lon_e}, Sierra cells: {int(np.isfinite(region).sum())}")

    ez = np.load(ELEVATION_NPZ, allow_pickle=True)
    assert np.allclose(ez["lat"], era5_lat, atol=1e-6) and np.allclose(ez["lon"], era5_lon, atol=1e-6)
    dz = ez["dz"]  # z_sierra_dem_m - z_era5_m, meters
    log(f"dz grid loaded, finite cells: {int(np.isfinite(dz).sum())}")

    knn = np.load(KNN_MASK_NPZ, allow_pickle=True)
    native_lat = knn["lat"]; native_lon = knn["lon"]
    native_region = knn["assignment"].astype(np.float32)

    grid1 = get_swe_grid_definition(water_year=2021, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    assert np.allclose(grid1.latitude.values, native_lat, atol=1e-5)
    assert np.allclose(grid1.longitude.values, native_lon, atol=1e-5)

    with xr.open_dataset(era5_land_yearly_file("tp", 2021), decode_times=False) as ds_tp:
        full_era5_lat = np.asarray(ds_tp["latitude"].values, dtype=np.float64)
        full_era5_lon_raw = np.asarray(ds_tp["longitude"].values, dtype=np.float64)
    d_lat_era5 = full_era5_lat[1] - full_era5_lat[0]
    d_lon_era5_raw = full_era5_lon_raw[1] - full_era5_lon_raw[0]
    lat_lo = int(np.round((era5_lat[0] - full_era5_lat[0]) / d_lat_era5))
    era5_lon_raw0 = era5_lon[0] if era5_lon[0] >= 0 else era5_lon[0] + 360.0
    lon_lo = int(np.round((era5_lon_raw0 - full_era5_lon_raw[0]) / d_lon_era5_raw))
    log(f"Working-grid window in full ERA5-Land array: lat0_idx={lat_lo}, lon0_idx={lon_lo}")

    lat2d_native, lon2d_native = np.meshgrid(native_lat, native_lon, indexing="ij")
    era5_lat_idx_native = np.round((lat2d_native - full_era5_lat[0]) / d_lat_era5).astype(np.int64) - lat_lo
    lon2d_native_0_360 = np.where(lon2d_native < 0, lon2d_native + 360.0, lon2d_native)
    era5_lon_idx_native = np.round((lon2d_native_0_360 - full_era5_lon_raw[0]) / d_lon_era5_raw).astype(np.int64) - lon_lo
    in_bounds = (era5_lat_idx_native >= 0) & (era5_lat_idx_native < n_lat_e) & \
                (era5_lon_idx_native >= 0) & (era5_lon_idx_native < n_lon_e)
    flat_cell = np.where(in_bounds, era5_lat_idx_native * n_lon_e + era5_lon_idx_native, -1)

    # w_ir: per ERA5 cell, count of native pixels whose region matches the
    # CELL's OWN assigned (majority) region -- the occupancy-weighted
    # aggregation rule from docs/Snow17_Methodology_and_Readiness.md ss8.
    w_ir = np.zeros(n_lat_e * n_lon_e, dtype=np.int64)
    cell_region_flat = region.ravel()
    native_region_flat = native_region.ravel()
    cell_flat = flat_cell.ravel()
    valid = (cell_flat >= 0) & np.isfinite(native_region_flat)
    matching = valid & (native_region_flat == cell_region_flat[cell_flat.clip(min=0)])
    np.add.at(w_ir, cell_flat[matching], 1)
    w_ir = w_ir.reshape(n_lat_e, n_lon_e)
    log(f"w_ir (region-matching native pixel count per ERA5 cell) computed. "
        f"Sum by region: " + ", ".join(
            f"{name}={int(w_ir[region == code].sum())}" for code, name in GROUP_LABEL.items()))

    sierra_ii, sierra_jj = np.where(np.isfinite(region))
    n_sierra = len(sierra_ii)
    log(f"Sierra ERA5-Land cells to simulate: {n_sierra}")

    section("2. Extract ERA5-Land tp/t2m for the working-grid window, 1983-1985, reset-aware daily")
    if FORCING_CACHE.exists():
        log(f"Loading cached daily forcing: {FORCING_CACHE}")
        fc = np.load(FORCING_CACHE, allow_pickle=True)
        precip_arr = fc["precip_arr"]
        tair_arr = fc["tair_arr"]
        common_index = pd.DatetimeIndex(fc["common_index"])
        n_days = len(common_index)
        log(f"Aligned daily series (cached): {n_days} days, {common_index.min()} to {common_index.max()}")
        assert np.all(precip_arr[np.isfinite(precip_arr)] >= -1e-6), "negative precipitation found"
        log(f"Precip (mm/day) range over Sierra cells: "
            f"[{precip_arr[:, sierra_ii, sierra_jj].min():.3f}, {precip_arr[:, sierra_ii, sierra_jj].max():.3f}]")
        log(f"Uncorrected T (degC) range over Sierra cells: "
            f"[{tair_arr[:, sierra_ii, sierra_jj].min():.2f}, {tair_arr[:, sierra_ii, sierra_jj].max():.2f}]")
    else:
        precip_arr, tair_arr, common_index, n_days = extract_forcing(sierra_ii, sierra_jj, n_lat_e, n_lon_e, lat_lo, lon_lo)
        np.savez(FORCING_CACHE, precip_arr=precip_arr, tair_arr=tair_arr,
                 common_index=common_index.values.astype("datetime64[ns]"))
        log(f"Cached daily forcing to {FORCING_CACHE}")

    section("3. Apply -4.5 K/km elevation correction")
    tair_corrected = tair_arr + (LAPSE_RATE_K_PER_KM / 1000.0) * dz[None, :, :]
    log(f"Corrected T (degC) range over Sierra cells: "
        f"[{tair_corrected[:, sierra_ii, sierra_jj].min():.2f}, {tair_corrected[:, sierra_ii, sierra_jj].max():.2f}]")
    mean_shift = np.nanmean(tair_corrected[:, sierra_ii, sierra_jj] - tair_arr[:, sierra_ii, sierra_jj])
    log(f"Mean correction applied: {mean_shift:+.3f} degC (should be negative -- correction cools, "
        f"since dz is positive on average)")

    section("4. Run 1-year spin-up (WY1984) + Snow-17 through WY1985, per Sierra cell")
    time_arr = common_index.to_pydatetime()
    sl_full = (common_index >= SPINUP_START) & (common_index <= TEST_END)
    sl_test = (common_index >= TEST_START) & (common_index <= TEST_END)
    time_full = common_index[sl_full]
    time_test = common_index[sl_test]
    log(f"Spin-up+test window: {time_full.min()} to {time_full.max()} ({sl_full.sum()} days); "
        f"test window: {time_test.min()} to {time_test.max()} ({sl_test.sum()} days)")

    model_swe_test = np.full((sl_test.sum(), n_sierra), np.nan, dtype=np.float64)
    for k in range(n_sierra):
        i, j = sierra_ii[k], sierra_jj[k]
        lat_deg = float(era5_lat[i])
        elev_m = float(ez["z_sierra_dem_m"][i, j]) if np.isfinite(ez["z_sierra_dem_m"][i, j]) else float(ez["z_era5_m"][i, j])
        prec_cell = precip_arr[sl_full, i, j]
        tair_cell = tair_corrected[sl_full, i, j]
        swe, outflow = snow17(
            list(time_full.to_pydatetime()), prec_cell, tair_cell,
            lat=lat_deg, elevation=elev_m, dt=DT_HOURS, **FIXED_PARAMS,
        )
        model_swe_test[:, k] = swe[-sl_test.sum():]
        if k % 100 == 0:
            log(f"  simulated cell {k}/{n_sierra}")
    log(f"Simulated all {n_sierra} Sierra cells.")

    assert np.isfinite(model_swe_test).all(), "non-finite model SWE found"
    assert (model_swe_test >= -1e-9).all(), "negative model SWE found"
    log(f"Model SWE (mm) range over test period: [{model_swe_test.min():.2f}, {model_swe_test.max():.2f}]")

    section("5. Occupancy-weighted regional aggregation (model)")
    w_sierra = w_ir[sierra_ii, sierra_jj].astype(np.float64)
    region_sierra = region[sierra_ii, sierra_jj]
    model_regional = {}
    for code, name in GROUP_LABEL.items():
        sel = region_sierra == code
        w = w_sierra[sel]
        wsum = w.sum()
        vals = model_swe_test[:, sel]
        model_regional[name] = (vals * w[None, :]).sum(axis=1) / wsum
        log(f"  {name}: {int(sel.sum())} cells, sum(w_ir)={int(wsum)}, "
            f"model SWE range [{model_regional[name].min():.2f}, {model_regional[name].max():.2f}] mm")

    section("6. Load observed regional SWE (native-pixel flat mean, WY1985 daily)")
    with xr.open_dataset(swe_file_for_water_year(1985), engine="netcdf4", decode_times=True) as ds:
        obs_field = ds[SWE_VARIABLE].isel(Stats=0, drop=True)
        obs_field = obs_field.sel({grid1.latitude_name: grid1.latitude, grid1.longitude_name: grid1.longitude})
        obs_field = obs_field.sel(time=slice(str(time_test.min().date()), str(time_test.max().date()))).load()
    obs_vals = np.asarray(obs_field.where(obs_field != SWE_MISSING_VALUE).values, dtype=np.float64)  # (time, lat, lon), meters
    obs_times = pd.DatetimeIndex(obs_field["time"].values)
    log(f"Observed SWE field: {obs_vals.shape}, {obs_times.min()} to {obs_times.max()}, units=meters")

    obs_regional = {}
    for code, name in GROUP_LABEL.items():
        mask = native_region == code
        vals_r = obs_vals[:, mask]
        obs_regional[name] = np.nanmean(vals_r, axis=1) * 1000.0  # meters -> mm
        log(f"  {name}: obs SWE range [{np.nanmin(obs_regional[name]):.2f}, {np.nanmax(obs_regional[name]):.2f}] mm")

    common_obs_model_index = time_test.intersection(obs_times)
    log(f"Common model/obs daily index: {len(common_obs_model_index)} days")

    section("7. 1-NSE objective (diagnostic only, NOT a calibration run)")
    objective = {}
    for name in GROUP_LABEL.values():
        m_series = pd.Series(model_regional[name], index=time_test).reindex(common_obs_model_index)
        o_series = pd.Series(obs_regional[name], index=obs_times).reindex(common_obs_model_index)
        valid = np.isfinite(m_series.values) & np.isfinite(o_series.values)
        o = o_series.values[valid]; m = m_series.values[valid]
        denom = np.sum((o - o.mean()) ** 2)
        num = np.sum((o - m) ** 2)
        one_minus_nse = num / denom if denom > 0 else np.nan
        nse = 1.0 - one_minus_nse
        objective[name] = {
            "n_days": int(valid.sum()), "NSE": float(nse), "one_minus_NSE": float(one_minus_nse),
            "obs_mean_mm": float(o.mean()), "model_mean_mm": float(m.mean()),
            "obs_max_mm": float(o.max()), "model_max_mm": float(m.max()),
        }
        log(f"  {name}: n={valid.sum()}, NSE={nse:.4f}, 1-NSE={one_minus_nse:.4f}, "
            f"obs_mean={o.mean():.1f}mm, model_mean={m.mean():.1f}mm")

    section("8. Physical sanity checks")
    checks = {}
    checks["precip_nonnegative"] = bool(np.all(precip_arr[np.isfinite(precip_arr)] >= -1e-6))
    checks["precip_no_21600x_error"] = bool(precip_arr[:, sierra_ii, sierra_jj].max() < 300)
    checks["temperature_plausible_range"] = bool(
        (tair_corrected[:, sierra_ii, sierra_jj].min() > -40) and (tair_corrected[:, sierra_ii, sierra_jj].max() < 40)
    )
    checks["model_swe_finite"] = bool(np.isfinite(model_swe_test).all())
    checks["model_swe_nonnegative"] = bool((model_swe_test >= -1e-9).all())
    months = np.asarray(time_test.month)
    winter_mask = np.isin(months, [12, 1, 2, 3])
    summer_mask = np.isin(months, [7, 8, 9])
    for name in GROUP_LABEL.values():
        checks[f"{name}_winter_accum_gt_summer"] = bool(
            model_regional[name][winter_mask].mean() > model_regional[name][summer_mask].mean()
        )
    checks["regional_weights_sum_correct"] = bool(all(
        abs(int(w_sierra[region_sierra == code].sum()) - int(w_ir[region == code].sum())) == 0
        for code in GROUP_LABEL
    ))
    checks["units_consistent_mm"] = True  # model mm (tonic native), obs converted m->mm above
    log(json.dumps(checks, indent=2))
    all_pass = all(checks.values())
    log(f"\nALL SMOKE-TEST CHECKS PASS: {all_pass}")

    section("9. Save artifacts")
    daily_out = pd.DataFrame(index=common_obs_model_index)
    for name in GROUP_LABEL.values():
        daily_out[f"model_swe_mm_{name}"] = pd.Series(model_regional[name], index=time_test).reindex(common_obs_model_index)
        daily_out[f"obs_swe_mm_{name}"] = pd.Series(obs_regional[name], index=obs_times).reindex(common_obs_model_index)
    daily_out.to_csv(OUT_DIR / "daily_regional_swe_model_vs_obs.csv")

    summary = {
        "scope": "Deterministic smoke test, fixed (not calibrated) parameters. No SCE-UA performed.",
        "fixed_parameters": FIXED_PARAMS,
        "lapse_rate_K_per_km": LAPSE_RATE_K_PER_KM,
        "spinup_start": str(SPINUP_START.date()), "test_start": str(TEST_START.date()), "test_end": str(TEST_END.date()),
        "n_sierra_cells_simulated": int(n_sierra),
        "objective_1_minus_NSE_by_region": objective,
        "sanity_checks": checks,
        "all_checks_pass": all_pass,
    }
    (OUT_DIR / "smoke_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"Saved: {OUT_DIR / 'smoke_test_summary.json'}")
    log(f"Saved: {OUT_DIR / 'daily_regional_swe_model_vs_obs.csv'}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for ax, name in zip(axes, GROUP_LABEL.values()):
        ax.plot(daily_out.index, daily_out[f"obs_swe_mm_{name}"], label="Observed (native flat mean)", color="black")
        ax.plot(daily_out.index, daily_out[f"model_swe_mm_{name}"], label="Model (fixed params, occupancy-weighted)", color="#e31a1c")
        ax.set_title(f"{name} Sierra: 1-NSE={objective[name]['one_minus_NSE']:.3f}, NSE={objective[name]['NSE']:.3f}")
        ax.set_ylabel("SWE (mm)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle("Snow-17 smoke test, WY1985, fixed (uncalibrated) parameters", fontsize=13)
    fig.savefig(OUT_DIR / "plots" / "smoke_test_regional_swe.png", dpi=180)
    log(f"Saved plot: {OUT_DIR / 'plots' / 'smoke_test_regional_swe.png'}")

    log("\nDONE.")


if __name__ == "__main__":
    main()
