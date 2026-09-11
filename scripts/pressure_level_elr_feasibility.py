"""
Pressure-level approximation feasibility check for Dutra et al. (2020) ELR,
plus an I/O benchmark for the reduced-cost extraction plan.

Dutra's exact criterion (Section 2.2.2, confirmed from the primary text):
  - ELR computed from ERA5 MODEL levels (hybrid sigma-pressure), NOT pressure
    levels.
  - 16 combinations of model levels, centered between model level 124
    (~500 m above the surface) and model level 116 (~1200 m above the
    surface).
  - height measured above the surface (ERA5's own model orography).

This script checks: for our local 37-level PRESSURE archive, how many
standard pressure levels fall in that same 500-1200 m AGL window, for 9
representative Sierra cells (North/Central/South x low/med/high), using
ERA5's OWN invariant surface geopotential as the height-above-surface
reference (the same reference frame Dutra's "above the surface" uses).
"""
import sys, os, json, time
from pathlib import Path

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import xarray as xr

CELLS_JSON = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "snow17_spinup_convergence_test/representative_cells.json"
)
CELLS = json.load(open(CELLS_JSON))

GRAVITY = 9.80665
ERA5_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5")
INVARIANT_Z = ERA5_ROOT / "e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"

SAMPLE_DAY_WINTER = ("198501", "e5.oper.an.pl.128_129_z.ll025sc.1985011500_1985011523.nc",
                     "e5.oper.an.pl.128_130_t.ll025sc.1985011500_1985011523.nc")
SAMPLE_DAY_SPRING = ("198504", "e5.oper.an.pl.128_129_z.ll025sc.1985041500_1985041523.nc",
                     "e5.oper.an.pl.128_130_t.ll025sc.1985041500_1985041523.nc")

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/dutra2020_elr_feasibility_audit")


def log(msg):
    print(msg, flush=True)


def nearest_index(coord, target):
    return int(np.argmin(np.abs(coord - target)))


def main():
    # ------------------------------------------------------------------
    log("=== Loading ERA5 invariant surface geopotential ===")
    with xr.open_dataset(INVARIANT_Z, decode_times=False) as ds:
        lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
        lon_name = [c for c in ds.coords if "lon" in c.lower()][0]
        era5_lat = np.asarray(ds[lat_name].values, dtype=np.float64)
        era5_lon_raw = np.asarray(ds[lon_name].values, dtype=np.float64)
        z_sfc_varname = [v for v in ds.data_vars if v.upper() == "Z"][0]
        z_sfc_full = ds[z_sfc_varname].isel(time=0).values if "time" in ds[z_sfc_varname].dims else ds[z_sfc_varname].values
    era5_lon = np.where(era5_lon_raw > 180.0, era5_lon_raw - 360.0, era5_lon_raw)

    cell_idx = {}
    for name, c in CELLS.items():
        li = nearest_index(era5_lat, c["lat"])
        lj = nearest_index(era5_lon, c["lon"])
        z_sfc_m = float(z_sfc_full[li, lj]) / GRAVITY
        cell_idx[name] = {"lat_idx": li, "lon_idx": lj, "z_surface_m": z_sfc_m,
                           "target_lat": c["lat"], "target_lon": c["lon"],
                           "swe_grid_elev_m": c["elevation_m"]}
        log(f"{name}: era5_lat_idx={li} era5_lon_idx={lj} "
            f"ERA5-own-surface-elev={z_sfc_m:.1f} m  (SWE-grid-DEM elev={c['elevation_m']:.0f} m)")

    # ------------------------------------------------------------------
    log("\n=== Pressure-level height-above-surface check (winter sample: 1985-01-15 00Z) ===")
    z_path = ERA5_ROOT / "e5.oper.an.pl" / SAMPLE_DAY_WINTER[0] / SAMPLE_DAY_WINTER[1]
    with xr.open_dataset(z_path, decode_times=False) as ds:
        levels = np.asarray(ds["level"].values, dtype=np.float64)
        z_var = ds["Z"] if "Z" in ds.data_vars else ds[[v for v in ds.data_vars if v.upper() == "Z"][0]]
        z_at_00z = z_var.isel(time=0)  # (level, lat, lon)

    results = {}
    for name, info in cell_idx.items():
        li, lj = info["lat_idx"], info["lon_idx"]
        z_profile_m = z_at_00z.isel(latitude=li, longitude=lj).values / GRAVITY  # geopotential height per level, meters
        height_agl = z_profile_m - info["z_surface_m"]
        in_window = (height_agl >= 500.0) & (height_agl <= 1200.0)
        below_ground = height_agl < 0.0
        n_in_window = int(np.sum(in_window))
        n_below_ground = int(np.sum(below_ground))
        levels_in_window = levels[in_window].tolist()
        results[name] = {
            "z_surface_m": info["z_surface_m"],
            "n_pressure_levels_in_500_1200m_AGL": n_in_window,
            "pressure_levels_in_window_hPa": levels_in_window,
            "height_agl_at_each_level_m": dict(zip(levels.astype(int).astype(str).tolist(), np.round(height_agl, 1).tolist())),
            "n_levels_below_ground": n_below_ground,
            "levels_below_ground_hPa": levels[below_ground].tolist(),
        }
        log(f"{name} (surf={info['z_surface_m']:.0f}m): "
            f"{n_in_window} pressure levels in [500,1200]m AGL -> {levels_in_window} hPa; "
            f"{n_below_ground} levels below ground -> {levels[below_ground].tolist()}")

    (OUT_DIR / "pressure_level_window_check_winter.json").write_text(json.dumps(results, indent=2) + "\n")

    # spring sample too, for robustness (geopotential/pressure-height relation varies with season/synoptic state)
    log("\n=== Pressure-level height-above-surface check (spring sample: 1985-04-15 00Z) ===")
    z_path2 = ERA5_ROOT / "e5.oper.an.pl" / SAMPLE_DAY_SPRING[0] / SAMPLE_DAY_SPRING[1]
    with xr.open_dataset(z_path2, decode_times=False) as ds:
        z_var2 = ds["Z"] if "Z" in ds.data_vars else ds[[v for v in ds.data_vars if v.upper() == "Z"][0]]
        z_at_00z_spring = z_var2.isel(time=0)

    results_spring = {}
    for name, info in cell_idx.items():
        li, lj = info["lat_idx"], info["lon_idx"]
        z_profile_m = z_at_00z_spring.isel(latitude=li, longitude=lj).values / GRAVITY
        height_agl = z_profile_m - info["z_surface_m"]
        in_window = (height_agl >= 500.0) & (height_agl <= 1200.0)
        n_in_window = int(np.sum(in_window))
        results_spring[name] = {
            "n_pressure_levels_in_500_1200m_AGL": n_in_window,
            "pressure_levels_in_window_hPa": levels[in_window].tolist(),
        }
        log(f"{name}: {n_in_window} levels in window -> {levels[in_window].tolist()} hPa")

    (OUT_DIR / "pressure_level_window_check_spring.json").write_text(json.dumps(results_spring, indent=2) + "\n")

    # ------------------------------------------------------------------
    log("\n=== I/O benchmark: selective chunk read (4 synoptic hours, 2 vars) vs full day read ===")
    t_path = ERA5_ROOT / "e5.oper.an.pl" / SAMPLE_DAY_WINTER[0] / SAMPLE_DAY_WINTER[2]

    # (a) selective: only hours 0,6,12,18, only t and z, only Sierra box
    lat_box = slice(nearest_index(era5_lat, 42.5), nearest_index(era5_lat, 34.5) + 1)
    lon_box_lo, lon_box_hi = nearest_index(era5_lon, -123.5), nearest_index(era5_lon, -117.0)

    start = time.perf_counter()
    with xr.open_dataset(t_path, decode_times=False) as ds:
        t_sel = ds["T"].isel(time=[0, 6, 12, 18], latitude=lat_box, longitude=slice(lon_box_lo, lon_box_hi)).load()
    with xr.open_dataset(z_path, decode_times=False) as ds:
        z_sel = ds["Z"].isel(time=[0, 6, 12, 18], latitude=lat_box, longitude=slice(lon_box_lo, lon_box_hi)).load()
    elapsed_selective = time.perf_counter() - start
    bytes_selective = t_sel.nbytes + z_sel.nbytes
    log(f"Selective read (4 hours x 2 vars x Sierra box x 37 levels): "
        f"{elapsed_selective:.2f} s wall, {bytes_selective/1e6:.1f} MB decompressed in memory")

    # (b) full comparison: read all 24 hours, full global, one variable (t), for direct cost comparison
    start = time.perf_counter()
    with xr.open_dataset(t_path, decode_times=False) as ds:
        t_full = ds["T"].isel(time=slice(0, 4)).load()  # only 4 timesteps, but FULL global grid, all levels
    elapsed_full4 = time.perf_counter() - start
    bytes_full4 = t_full.nbytes
    log(f"Full-global read, same 4 timesteps, all levels, 1 var (t): "
        f"{elapsed_full4:.2f} s wall, {bytes_full4/1e6:.1f} MB decompressed")

    benchmark = {
        "selective_4hr_2var_sierra_box": {"seconds": elapsed_selective, "mb_decompressed": bytes_selective / 1e6},
        "full_global_4hr_1var": {"seconds": elapsed_full4, "mb_decompressed": bytes_full4 / 1e6},
    }
    (OUT_DIR / "io_benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    log(f"\nBenchmark saved: {OUT_DIR / 'io_benchmark.json'}")
    log("\nDONE.")


if __name__ == "__main__":
    main()
