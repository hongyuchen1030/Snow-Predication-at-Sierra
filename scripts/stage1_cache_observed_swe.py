"""
Cache daily observed regional SWE (native-pixel flat mean, North/Central/
South/whole-Sierra) for WY1985-2021, from the KNN-filled finalized region
mask. Used both as the calibration target (WY1985-2004) and the held-out
comparison target (WY2005-2021) -- this script itself performs NO
calibration and makes NO parameter decisions; it only caches observations.
"""
import sys, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from snow_ml.data import (
    DEFAULT_SIERRA_REGION, SWE_VARIABLE, SWE_MISSING_VALUE,
    get_swe_grid_definition, swe_file_for_water_year,
)

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_stage1_scua_validation")
CACHE_DIR = OUT_DIR / "forcing_cache"
KNN_MASK_NPZ = REPO_ROOT / "artifacts/sierra_basin_assignment_knn_filled/basin_assignment_grid_wy2021_knn_filled.npz"
WATER_YEARS = list(range(1985, 2022))  # WY1985-2021 inclusive
GROUP_LABEL = {1: "North", 2: "Central", 3: "South"}
N_WORKERS = 6


def log(msg):
    print(msg, flush=True)


def read_one_wy(year):
    knn = np.load(KNN_MASK_NPZ, allow_pickle=True)
    native_region = knn["assignment"].astype(np.float32)
    grid1 = get_swe_grid_definition(water_year=year, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    with xr.open_dataset(swe_file_for_water_year(year), engine="netcdf4", decode_times=True) as ds:
        field = ds[SWE_VARIABLE].isel(Stats=0, drop=True)
        field = field.sel({grid1.latitude_name: grid1.latitude, grid1.longitude_name: grid1.longitude}).load()
    vals = np.asarray(field.where(field != SWE_MISSING_VALUE).values, dtype=np.float64)  # (time, lat, lon), m
    times = pd.DatetimeIndex(field["time"].values)
    row = {"time": times}
    for code, name in GROUP_LABEL.items():
        mask = native_region == code
        row[name] = np.nanmean(vals[:, mask], axis=1) * 1000.0  # mm
    whole_mask = np.isfinite(native_region)
    row["Whole_Sierra"] = np.nanmean(vals[:, whole_mask], axis=1) * 1000.0
    return year, row


def main():
    t0 = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(read_one_wy, y): y for y in WATER_YEARS}
        n_done = 0
        for fut in as_completed(futs):
            year, row = fut.result()
            df = pd.DataFrame({k: v for k, v in row.items() if k != "time"}, index=row["time"])
            df["water_year"] = year
            frames.append(df)
            n_done += 1
            log(f"  [{n_done}/{len(WATER_YEARS)}] WY{year} done ({time.time()-t0:.0f}s elapsed)")

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    out_path = CACHE_DIR / "observed_regional_swe_wy1985_2021.csv"
    full.to_csv(out_path)
    log(f"Saved: {out_path}, shape={full.shape}, {full.index.min()} to {full.index.max()}")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
