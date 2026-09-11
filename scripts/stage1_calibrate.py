"""
Stage 1 SCE-UA calibration (WY1985-2004) via SPOTPY, per region.

Usage:
  python3 stage1_calibrate.py benchmark <region>
  python3 stage1_calibrate.py smoke <region>
  python3 stage1_calibrate.py calibrate <region> [n_repetitions] [ngs] [seed]

region in {North, Central, South}
"""
import sys, os, json, time, subprocess
from pathlib import Path
from collections import Counter

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "vendor"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

from snow17 import snow17
import spotpy
import spotpy.parallel.mproc as _mproc
_mproc.process_count = int(os.environ.get("SPOTPY_WORKERS", "80"))

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_stage1_scua_validation")
CACHE_DIR = OUT_DIR / "forcing_cache"
WORKING_GRID_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_era5land_working_grid/era5land_working_grid.npz")
ELEVATION_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_elevation_mismatch_audit/elevation_comparison.npz")
KNN_MASK_NPZ = REPO_ROOT / "artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"

SPINUP_START = pd.Timestamp("1983-10-01")
CAL_START = pd.Timestamp("1984-10-01")   # WY1985 start
CAL_END = pd.Timestamp("2004-09-30")     # WY2004 end
REGION_CODE = {"North": 1, "Central": 2, "South": 3}

FIXED_PARAMS = dict(rvs=1.0, mbase=1.0, tipm=0.1, nmf=0.15, plwhc=0.04)
BOUNDS = {
    "scf": (0.9, 1.2), "mfmax": (0.5, 1.3), "mfmin": (0.1, 0.6), "uadj": (0.05, 0.2),
    "pxtemp": (0.0, 2.0), "pxtemp1": (-2.0, 0.0), "pxtemp2": (0.0, 4.0),
}
DT_HOURS = 24


def log(msg):
    print(msg, flush=True)


def load_region_data(region_name):
    code = REGION_CODE[region_name]

    fc = np.load(CACHE_DIR / "daily_forcing_1983_2021.npz", allow_pickle=True)
    common_index = pd.DatetimeIndex(fc["common_index"])
    precip_all = fc["precip_mm"]
    tair_all = fc["tair_corrected_degC"]
    era5_lat = fc["era5_lat"]; era5_lon = fc["era5_lon"]
    region = fc["region"]
    n_lat_e, n_lon_e = region.shape

    ez = np.load(ELEVATION_NPZ, allow_pickle=True)
    z_dem = ez["z_sierra_dem_m"]; z_era5 = ez["z_era5_m"]
    elev_use = np.where(np.isfinite(z_dem), z_dem, z_era5)

    knn = np.load(KNN_MASK_NPZ, allow_pickle=True)
    native_lat = knn["lat"]; native_lon = knn["lon"]
    native_region = knn["assignment"].astype(np.float32)

    from snow_ml.data import DEFAULT_SIERRA_REGION, get_swe_grid_definition, era5_land_yearly_file
    grid1 = get_swe_grid_definition(water_year=2021, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    assert np.allclose(grid1.latitude.values, native_lat, atol=1e-5)

    import xarray as xr
    with xr.open_dataset(era5_land_yearly_file("tp", 2021), decode_times=False) as ds_tp:
        full_era5_lat = np.asarray(ds_tp["latitude"].values, dtype=np.float64)
        full_era5_lon_raw = np.asarray(ds_tp["longitude"].values, dtype=np.float64)
    d_lat_era5 = full_era5_lat[1] - full_era5_lat[0]
    d_lon_era5_raw = full_era5_lon_raw[1] - full_era5_lon_raw[0]
    lat_lo = int(np.round((era5_lat[0] - full_era5_lat[0]) / d_lat_era5))
    era5_lon_raw0 = era5_lon[0] if era5_lon[0] >= 0 else era5_lon[0] + 360.0
    lon_lo = int(np.round((era5_lon_raw0 - full_era5_lon_raw[0]) / d_lon_era5_raw))

    lat2d_native, lon2d_native = np.meshgrid(native_lat, native_lon, indexing="ij")
    era5_lat_idx_native = np.round((lat2d_native - full_era5_lat[0]) / d_lat_era5).astype(np.int64) - lat_lo
    lon2d_native_0_360 = np.where(lon2d_native < 0, lon2d_native + 360.0, lon2d_native)
    era5_lon_idx_native = np.round((lon2d_native_0_360 - full_era5_lon_raw[0]) / d_lon_era5_raw).astype(np.int64) - lon_lo
    in_bounds = (era5_lat_idx_native >= 0) & (era5_lat_idx_native < n_lat_e) & \
                (era5_lon_idx_native >= 0) & (era5_lon_idx_native < n_lon_e)
    flat_cell = np.where(in_bounds, era5_lat_idx_native * n_lon_e + era5_lon_idx_native, -1)

    w_ir = np.zeros(n_lat_e * n_lon_e, dtype=np.int64)
    cell_region_flat = region.ravel()
    native_region_flat = native_region.ravel()
    cell_flat = flat_cell.ravel()
    valid = (cell_flat >= 0) & np.isfinite(native_region_flat)
    matching = valid & (native_region_flat == cell_region_flat[cell_flat.clip(min=0)])
    np.add.at(w_ir, cell_flat[matching], 1)
    w_ir = w_ir.reshape(n_lat_e, n_lon_e)

    cell_ii, cell_jj = np.where(region == code)
    w_cells = w_ir[cell_ii, cell_jj].astype(np.float64)
    lat_cells = era5_lat[cell_ii]
    elev_cells = elev_use[cell_ii, cell_jj]
    precip_cells = precip_all[:, cell_ii, cell_jj]  # (time, n_cells)
    tair_cells = tair_all[:, cell_ii, cell_jj]

    obs = pd.read_csv(CACHE_DIR / "observed_regional_swe_wy1985_2021.csv", index_col=0, parse_dates=True)

    return {
        "common_index": common_index, "precip_cells": precip_cells, "tair_cells": tair_cells,
        "lat_cells": lat_cells, "elev_cells": elev_cells, "w_cells": w_cells,
        "n_cells": len(cell_ii), "obs": obs,
    }


def run_region_snow17(data, params, sl):
    """Run snow17 for every cell in the region over the boolean time-slice sl
    (aligned to data['common_index']), return occupancy-weighted daily
    regional SWE for that slice."""
    idx = np.where(sl)[0]
    n_t = len(idx)
    n_cells = data["n_cells"]
    swe_out = np.empty((n_t, n_cells), dtype=np.float64)
    time_pyd = list(data["common_index"][idx].to_pydatetime())
    for k in range(n_cells):
        prec_k = data["precip_cells"][idx, k]
        tair_k = data["tair_cells"][idx, k]
        swe, outflow = snow17(
            time_pyd, prec_k, tair_k,
            lat=float(data["lat_cells"][k]), elevation=float(data["elev_cells"][k]),
            dt=DT_HOURS, **params, **FIXED_PARAMS,
        )
        swe_out[:, k] = swe
    w = data["w_cells"]
    regional = (swe_out * w[None, :]).sum(axis=1) / w.sum()
    return regional


class Snow17RegionSetup:
    def __init__(self, region_name):
        self.region_name = region_name
        self.progress_path = OUT_DIR / "logs" / f"progress_{region_name.lower()}.log"
        self.data = load_region_data(region_name)
        self.sl_full = (self.data["common_index"] >= SPINUP_START) & (self.data["common_index"] <= CAL_END)
        self.time_full = self.data["common_index"][self.sl_full]
        self.sl_cal_within_full = (self.time_full >= CAL_START) & (self.time_full <= CAL_END)
        obs_series = self.data["obs"][region_name]
        self.obs_aligned = obs_series.reindex(self.time_full[self.sl_cal_within_full]).values
        self.params = [
            spotpy.parameter.Uniform("scf", *BOUNDS["scf"]),
            spotpy.parameter.Uniform("mfmax", *BOUNDS["mfmax"]),
            spotpy.parameter.Uniform("mfmin", *BOUNDS["mfmin"]),
            spotpy.parameter.Uniform("uadj", *BOUNDS["uadj"]),
            spotpy.parameter.Uniform("pxtemp", *BOUNDS["pxtemp"]),
            spotpy.parameter.Uniform("pxtemp1", *BOUNDS["pxtemp1"]),
            spotpy.parameter.Uniform("pxtemp2", *BOUNDS["pxtemp2"]),
        ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        params = dict(scf=vector[0], mfmax=vector[1], mfmin=vector[2], uadj=vector[3],
                      pxtemp=vector[4], pxtemp1=vector[5], pxtemp2=vector[6])
        full_sim = run_region_snow17(self.data, params, self.sl_full)
        return full_sim[self.sl_cal_within_full.values if hasattr(self.sl_cal_within_full, "values") else self.sl_cal_within_full]

    def evaluation(self):
        return self.obs_aligned

    def objectivefunction(self, simulation, evaluation, params=None):
        sim = np.asarray(simulation, dtype=np.float64)
        obs = np.asarray(evaluation, dtype=np.float64)
        valid = np.isfinite(sim) & np.isfinite(obs)
        if valid.sum() < 10:
            obj = 1e6
        else:
            o = obs[valid]; s = sim[valid]
            denom = np.sum((o - o.mean()) ** 2)
            obj = 1e6 if denom <= 0 else float(np.sum((o - s) ** 2) / denom)
        try:
            with open(self.progress_path, "a") as fh:
                fh.write(f"{time.time():.1f},{os.getpid()},{obj:.6f}\n")
        except Exception:
            pass
        return obj


def cmd_benchmark(region):
    log(f"Benchmarking one full model evaluation for region={region}...")
    setup = Snow17RegionSetup(region)
    midpoint = {k: (lo + hi) / 2 for k, (lo, hi) in BOUNDS.items()}
    log(f"n_cells={setup.data['n_cells']}, spinup+cal days={setup.sl_full.sum()}, cal-only days={setup.sl_cal_within_full.sum()}")
    t0 = time.time()
    sim = setup.simulation(list(midpoint.values()))
    elapsed = time.time() - t0
    obj = setup.objectivefunction(sim, setup.evaluation())
    log(f"One evaluation took {elapsed:.2f} s. 1-NSE at midpoint params = {obj:.4f}")
    (OUT_DIR / "logs" / f"benchmark_{region}.json").write_text(json.dumps({
        "region": region, "seconds_per_evaluation": elapsed, "n_cells": setup.data["n_cells"],
        "one_minus_nse_at_midpoint": obj,
    }, indent=2) + "\n")


def cmd_smoke(region):
    log(f"Smoke-testing optimizer wiring for region={region}...")
    setup = Snow17RegionSetup(region)
    rng = np.random.default_rng(42)
    results = []
    for trial in range(5):
        vec = [rng.uniform(lo, hi) for lo, hi in BOUNDS.values()]
        sim = setup.simulation(vec)
        obj = setup.objectivefunction(sim, setup.evaluation())
        assert np.isfinite(obj), f"non-finite objective for trial {trial}"
        assert np.isfinite(sim).all(), f"non-finite simulation for trial {trial}"
        results.append({"vector": vec, "objective": obj, "sim_mean": float(np.nanmean(sim))})
        log(f"  trial {trial}: vec={[round(v,4) for v in vec]}, 1-NSE={obj:.4f}, sim_mean={np.nanmean(sim):.2f}mm")
    sim_means = [r["sim_mean"] for r in results]
    assert len(set(np.round(sim_means, 3))) > 1, "all candidates produced IDENTICAL output -- parameters not affecting simulation!"
    log("PASS: candidates produce different output, objectives finite.")
    (OUT_DIR / "logs" / f"smoke_optimizer_{region}.json").write_text(json.dumps(results, indent=2, default=float) + "\n")


def cmd_calibrate(region, repetitions, ngs, seed, kstop=5, pcento=0.01, peps=0.01):
    log(f"Calibrating region={region}, repetitions={repetitions}, ngs={ngs}, seed={seed}, "
        f"kstop={kstop}, pcento={pcento}, peps={peps}")
    np.random.seed(seed)
    setup = Snow17RegionSetup(region)
    dbname = str(OUT_DIR / "calibration" / region.lower() / f"sceua_{region.lower()}_seed{seed}")
    # NOTE: dbformat="ram" used deliberately, not "csv" -- dbformat="csv"
    # combined with parallel="mpc" was found (this session) to produce a
    # corrupted/header-less CSV that getdata() cannot parse back correctly
    # (field names come back as garbled null-padded strings). dbformat="ram"
    # returns the same in-memory structured array with correct field names
    # (verified: 'like1','parscf',...,'chain'); we write the CSV ourselves
    # from that array below, sidestepping the bug entirely.
    sampler = spotpy.algorithms.sceua(setup, dbname=dbname, dbformat="ram", parallel="mpc", save_sim=False)
    t0 = time.time()
    sampler.sample(repetitions, ngs=ngs, kstop=kstop, pcento=pcento, peps=peps)
    elapsed = time.time() - t0
    log(f"Calibration done in {elapsed:.0f}s ({elapsed/3600:.2f}h)")

    results = sampler.getdata()
    log(f"Result fields: {results.dtype.names}")
    like_col = "like1"
    best_idx = int(np.argmin(results[like_col]))
    best = results[best_idx]
    best_params = {name: float(best[f"par{name}"]) for name in BOUNDS.keys()}
    best_obj = float(best[like_col])
    nse = 1.0 - best_obj

    theta = {
        "region": region,
        "calibrated_parameters": best_params,
        "fixed_parameters": FIXED_PARAMS,
        "bounds": BOUNDS,
        "objective": "1 - NSE (minimize)",
        "one_minus_nse": best_obj,
        "nse_calibration_wy1985_2004": nse,
        "optimizer": "SPOTPY sceua",
        "optimizer_seed": seed,
        "n_repetitions_requested": repetitions,
        "ngs": ngs,
        "n_evaluations_actual": int(len(results)),
        "elapsed_seconds": elapsed,
        "calibration_period": "WY1985-2004",
        "spinup": "1 year forcing-driven zero-state, start 1983-10-01",
        "forcing_cache": str(CACHE_DIR / "daily_forcing_1983_2021.npz"),
        "snow17_source": "scripts/vendor/snow17.py (ait typo fixed)",
    }
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).decode().strip()
        theta["git_commit"] = commit
    except Exception:
        theta["git_commit"] = None

    out_json = OUT_DIR / "frozen_parameters" / f"theta_{region[0]}.json"
    out_json.write_text(json.dumps(theta, indent=2) + "\n")
    log(f"Saved frozen parameters: {out_json}")
    log(json.dumps(theta, indent=2))

    conv_csv = OUT_DIR / "calibration" / region.lower() / f"convergence_history_seed{seed}.csv"
    results_df = pd.DataFrame(results)
    results_df["region"] = region
    results_df["seed"] = seed
    results_df.to_csv(conv_csv, index=False)
    log(f"Saved convergence history: {conv_csv} ({len(results_df)} rows)")


if __name__ == "__main__":
    mode = sys.argv[1]
    region = sys.argv[2]
    if mode == "benchmark":
        cmd_benchmark(region)
    elif mode == "smoke":
        cmd_smoke(region)
    elif mode == "calibrate":
        repetitions = int(sys.argv[3])
        ngs = int(sys.argv[4])
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
        cmd_calibrate(region, repetitions, ngs, seed)
    else:
        raise ValueError(f"unknown mode {mode}")
