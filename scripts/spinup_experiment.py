"""
Snow-17 spin-up-duration convergence experiment.
No calibration. No ACE2. No observed SWE. Meteorological-forcing-driven only.
"""
import sys, os, json
from pathlib import Path
from datetime import datetime

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
from snow_ml.data import era5_land_yearly_file
from snow17_instrumented import snow17_instrumented

OUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_spinup_convergence_test")
for sub in ("plots", "logs", "forcing"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

CELLS_JSON = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_spinup_convergence_test/representative_cells.json")
CELLS = json.load(open(CELLS_JSON))

# Fixed Snow-17 parameters for THIS SENSITIVITY TEST ONLY (not calibrated, not
# production values). Source: tonic's own documented defaults, each explicitly
# attributed in the upstream docstring to Shamir & Georgakakos (2007),
# American River Basin -- an independently published Sierra-adjacent basin
# parameterization, not an invented/tutorial placeholder. The SAME fixed
# vector is used for every run and every cell in this experiment, as required.
FIXED_PARAMS = dict(
    scf=1.0, rvs=1, uadj=0.04, mbase=1.0, mfmax=1.05, mfmin=0.6,
    tipm=0.1, nmf=0.15, plwhc=0.04, pxtemp=1.0, pxtemp1=-1.0, pxtemp2=3.0,
)
DT_HOURS = 24  # daily timestep, matches project's existing ERA5-Land daily aggregation convention

CANDIDATES = [
    ("6mo", "1984-04-01"),
    ("1yr", "1983-10-01"),
    ("2yr", "1982-10-01"),
    ("3yr", "1981-10-01"),
    ("5yr_ref", "1979-10-01"),
]
TARGET_DATE = pd.Timestamp("1984-10-01")
END_DATE = pd.Timestamp("1985-09-30")  # through end of WY1985
YEARS = list(range(1979, 1986))
STATE_NAMES = ["ait", "w_qx", "w_q", "w_i", "deficit"]


def log(msg):
    print(msg, flush=True)


def nearest_index(coord_array, target, ascending_check=None):
    return int(np.argmin(np.abs(coord_array - target)))


def get_cell_indices(sample_ds, lat_name, lon_name):
    """Compute (lat_idx, lon_idx) per cell against this file's own coordinates."""
    file_lat = np.asarray(sample_ds[lat_name].values, dtype=np.float64)
    file_lon_raw = np.asarray(sample_ds[lon_name].values, dtype=np.float64)
    file_lon = np.where(file_lon_raw > 180.0, file_lon_raw - 360.0, file_lon_raw)
    idx_map = {}
    for name, c in CELLS.items():
        lat_idx = nearest_index(file_lat, c["lat"])
        lon_idx = nearest_index(file_lon, c["lon"])
        idx_map[name] = (lat_idx, lon_idx)
    return idx_map, file_lat, file_lon


def extract_forcing_all_years():
    """Extract daily precip (mm) and temperature (degC) for all 9 cells, 1979-1985.

    IMPORTANT accumulation-convention fix: ERA5-Land `tp` is archived as
    precipitation ACCUMULATED SINCE 00 UTC, resetting every day (confirmed by
    direct inspection of raw hourly values: e.g. 1979-10-23 01:00-24:00 rises
    monotonically from ~0 to a daily total, then 1979-10-24 01:00 drops back
    to a small value and rises again). The value stamped at hour 00:00 of day
    D+1 IS the full accumulated total for calendar day D. Summing the 24
    already-cumulative hourly values within a day (the naive approach)
    over-counts by roughly an order of magnitude -- confirmed by an initial
    run of this script that produced a >1200 mm/day maximum, which is
    physically implausible for the Sierra Nevada. The correct daily total is
    therefore the hour-00:00 sample attributed to the PRECEDING calendar day,
    not resample('D').sum() of the raw hourly series. `t2m` is an
    instantaneous (non-accumulated) state variable, so a plain daily mean of
    the 24 hourly values for that calendar day is correct as before.
    """
    cell_names = list(CELLS.keys())
    hourly_frames = []
    idx_map = None

    for year in YEARS:
        tp_path = era5_land_yearly_file("tp", year)
        t2m_path = era5_land_yearly_file("t2m", year)
        log(f"Extracting forcing for {year}: {tp_path.name}, {t2m_path.name}")

        with xr.open_dataset(tp_path, decode_times=False) as ds_tp:
            lat_name = [c for c in ds_tp.coords if "lat" in c.lower()][0]
            lon_name = [c for c in ds_tp.coords if "lon" in c.lower()][0]
            time_name = [c for c in ds_tp.coords if "time" in c.lower()][0]
            if idx_map is None:
                idx_map, file_lat, file_lon = get_cell_indices(ds_tp, lat_name, lon_name)
                log("Cell -> nearest ERA5-Land grid index (lat_idx, lon_idx):")
                for name, (li, lj) in idx_map.items():
                    log(f"  {name}: idx=({li},{lj}) -> era5 lat={file_lat[li]:.4f} lon={file_lon[lj]:.4f} "
                        f"(target lat={CELLS[name]['lat']:.4f} lon={CELLS[name]['lon']:.4f})")
                # Bounding box covering all 9 cells, so each year's extraction is ONE
                # contiguous hyperslab read (fast) instead of scattered fancy-index
                # point reads (extremely slow on this chunked networked-filesystem file).
                lat_indices = [v[0] for v in idx_map.values()]
                lon_indices = [v[1] for v in idx_map.values()]
                lat_lo, lat_hi = min(lat_indices), max(lat_indices) + 1
                lon_lo, lon_hi = min(lon_indices), max(lon_indices) + 1
                log(f"Bounding-box read window: lat[{lat_lo}:{lat_hi}] lon[{lon_lo}:{lon_hi}] "
                    f"({lat_hi - lat_lo} x {lon_hi - lon_lo} cells)")
            tp_varname = [v for v in ds_tp.data_vars if v.lower() not in ("time_bnds",)][0]
            tp_box = ds_tp[tp_varname].isel(
                {lat_name: slice(lat_lo, lat_hi), lon_name: slice(lon_lo, lon_hi)}
            ).load()
            time_hours = pd.to_datetime(
                ds_tp[time_name].values, unit="h", origin=pd.Timestamp("1900-01-01")
            )
        tp_box_vals = np.asarray(tp_box.values, dtype=np.float64)  # (time, lat_box, lon_box)
        tp_vals = np.column_stack([
            tp_box_vals[:, idx_map[n][0] - lat_lo, idx_map[n][1] - lon_lo] for n in cell_names
        ])  # (time, cell), units m (accumulated per hour)

        with xr.open_dataset(t2m_path, decode_times=False) as ds_t2m:
            t2m_varname = [v for v in ds_t2m.data_vars if v.lower() not in ("time_bnds",)][0]
            t2m_box = ds_t2m[t2m_varname].isel(
                {lat_name: slice(lat_lo, lat_hi), lon_name: slice(lon_lo, lon_hi)}
            ).load()
        t2m_box_vals = np.asarray(t2m_box.values, dtype=np.float64)
        t2m_vals = np.column_stack([
            t2m_box_vals[:, idx_map[n][0] - lat_lo, idx_map[n][1] - lon_lo] for n in cell_names
        ])  # (time, cell), units K

        df = pd.DataFrame(index=time_hours)
        for k, name in enumerate(cell_names):
            df[f"tp_m_{name}"] = tp_vals[:, k]
            df[f"t2m_K_{name}"] = t2m_vals[:, k]
        hourly_frames.append(df)

    hourly = pd.concat(hourly_frames).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    log(f"Combined hourly series: shape={hourly.shape}, range={hourly.index.min()} to {hourly.index.max()}")

    # Correct daily aggregation across the FULL concatenated hourly series
    # (so the year-boundary reset, e.g. Dec 31 -> Jan 1, is handled correctly).
    is_midnight = hourly.index.hour == 0
    midnight_rows = hourly[is_midnight]
    daily_index = (midnight_rows.index - pd.Timedelta(days=1))  # attribute the 00:00 sample to the day it concludes

    daily = pd.DataFrame(index=daily_index)
    for name in cell_names:
        tp_daily_m = midnight_rows[f"tp_m_{name}"].values  # accumulated total for the PRECEDING calendar day
        daily[f"precip_mm_{name}"] = tp_daily_m * 1000.0
        t2m_daily_K = hourly[f"t2m_K_{name}"].resample("D").mean()
        daily[f"tair_degC_{name}"] = (t2m_daily_K - 273.15).reindex(daily.index)
    daily = daily.sort_index()

    # Drop boundary rows with no full underlying hourly day (e.g. 1978-12-31,
    # an artifact of shifting the hour-00:00 sample back by one day at the
    # very start of the series). Our earliest spin-up start is 1979-10-01,
    # well inside the safe window, so this costs nothing.
    daily = daily.dropna(how="any")

    full = daily
    full.to_csv(OUT_DIR / "forcing" / "daily_forcing_all_cells_1979_1985.csv")
    log(f"Saved combined daily forcing: {OUT_DIR / 'forcing' / 'daily_forcing_all_cells_1979_1985.csv'}, "
        f"shape={full.shape}, range={full.index.min()} to {full.index.max()}")

    # sanity checks
    for name in cell_names:
        p = full[f"precip_mm_{name}"]
        t = full[f"tair_degC_{name}"]
        assert (p >= -1e-6).all(), f"Negative precipitation found for {name}"
        assert np.isfinite(p).all() and np.isfinite(t).all(), f"Non-finite forcing for {name}"
    log("Sanity check passed: all precip >= 0, all forcing finite.")
    log(f"Precip (mm/day) range across all cells/days: [{full.filter(like='precip_mm_').values.min():.3f}, "
        f"{full.filter(like='precip_mm_').values.max():.3f}]")
    log(f"Temperature (degC) range across all cells/days: [{full.filter(like='tair_degC_').values.min():.2f}, "
        f"{full.filter(like='tair_degC_').values.max():.2f}]")

    return full


def run_all_spinups(forcing: pd.DataFrame):
    results = {}  # results[cell][run_label] = dict with 'time','swe','ait',...
    for cell_name, cell in CELLS.items():
        lat = cell["lat"]
        elev = cell["elevation_m"]
        prec_full = forcing[f"precip_mm_{cell_name}"]
        tair_full = forcing[f"tair_degC_{cell_name}"]
        results[cell_name] = {}
        for label, start_str in CANDIDATES:
            start = pd.Timestamp(start_str)
            sl = (forcing.index >= start) & (forcing.index <= END_DATE)
            time_arr = forcing.index[sl].to_pydatetime()
            prec_arr = prec_full.values[sl]
            tair_arr = tair_full.values[sl]
            swe, outflow, ait, w_qx, w_q, w_i, deficit = snow17_instrumented(
                time_arr, prec_arr, tair_arr, lat=lat, elevation=elev, dt=DT_HOURS, **FIXED_PARAMS
            )
            results[cell_name][label] = {
                "start": start,
                "time": pd.DatetimeIndex(time_arr),
                "swe": swe, "ait": ait, "w_qx": w_qx, "w_q": w_q, "w_i": w_i, "deficit": deficit,
            }
            n_target = np.sum(pd.DatetimeIndex(time_arr) == TARGET_DATE)
            assert n_target == 1, f"target date not found exactly once for {cell_name}/{label}"
        log(f"Completed spin-up runs for cell {cell_name} (lat={lat:.3f}, elev={elev:.0f}m)")
    return results


def analyze_convergence(results):
    rows = []
    traj_diff_rows = []
    for cell_name, per_run in results.items():
        ref = per_run["5yr_ref"]
        ref_time = ref["time"]
        ref_target_idx = np.where(ref_time == TARGET_DATE)[0][0]
        ref_state_at_target = {s: ref[s][ref_target_idx] for s in STATE_NAMES}
        ref_swe_at_target = ref["swe"][ref_target_idx]

        for label, start_str in CANDIDATES:
            run = per_run[label]
            t_idx = np.where(run["time"] == TARGET_DATE)[0][0]
            row = {"cell": cell_name, "run": label, "start_date": start_str,
                   "spinup_days": int((TARGET_DATE - pd.Timestamp(start_str)).days)}
            for s in STATE_NAMES:
                val = run[s][t_idx]
                ref_val = ref_state_at_target[s]
                row[f"{s}_value"] = val
                row[f"{s}_abs_diff_vs_5yr"] = val - ref_val
                row[f"{s}_rel_diff_vs_5yr_pct"] = (
                    100.0 * (val - ref_val) / ref_val if abs(ref_val) > 1e-9 else
                    (0.0 if abs(val - ref_val) < 1e-9 else float("inf"))
                )
            swe_val = run["swe"][t_idx]
            row["model_swe_mm"] = swe_val
            row["model_swe_abs_diff_vs_5yr_mm"] = swe_val - ref_swe_at_target
            row["model_swe_rel_diff_vs_5yr_pct"] = (
                100.0 * (swe_val - ref_swe_at_target) / ref_swe_at_target if abs(ref_swe_at_target) > 1e-9
                else (0.0 if abs(swe_val - ref_swe_at_target) < 1e-9 else float("inf"))
            )
            rows.append(row)

            # trajectory persistence: |candidate - reference| at each day post-target, through END_DATE
            common_time = run["time"][run["time"] >= TARGET_DATE]
            ref_common_idx = np.searchsorted(ref_time.values, common_time.values)
            run_common_idx = np.searchsorted(run["time"].values, common_time.values)
            swe_diff_traj = run["swe"][run_common_idx] - ref["swe"][ref_common_idx]
            for day_offset in (0, 7, 30, 60, 90, 150, 210, 270, 330):
                if day_offset < len(swe_diff_traj):
                    traj_diff_rows.append({
                        "cell": cell_name, "run": label,
                        "days_after_1984_10_01": day_offset,
                        "swe_diff_mm": float(swe_diff_traj[day_offset]),
                    })

    df = pd.DataFrame(rows)
    traj_df = pd.DataFrame(traj_diff_rows)
    df.to_csv(OUT_DIR / "state_convergence_at_1984-10-01.csv", index=False)
    traj_df.to_csv(OUT_DIR / "swe_diff_trajectory_post_1984-10-01.csv", index=False)
    log(f"Saved: {OUT_DIR / 'state_convergence_at_1984-10-01.csv'}")
    log(f"Saved: {OUT_DIR / 'swe_diff_trajectory_post_1984-10-01.csv'}")
    return df, traj_df


def summarize_and_print(df, traj_df):
    log("\n" + "=" * 78)
    log("STATE CONVERGENCE AT 1984-10-01 (vs 5-year reference spin-up)")
    log("=" * 78)
    agg_rows = []
    for label, start_str in CANDIDATES:
        if label == "5yr_ref":
            continue
        sub = df[df["run"] == label]
        row = {"run": label, "spinup_days": sub["spinup_days"].iloc[0]}
        for s in STATE_NAMES + ["model_swe"]:
            col = f"{s}_abs_diff_vs_5yr" if s != "model_swe" else "model_swe_abs_diff_vs_5yr_mm"
            vals = sub[col].abs()
            row[f"{s}_max_abs_diff"] = vals.max()
            row[f"{s}_p95_abs_diff"] = vals.quantile(0.95)
            row[f"{s}_mean_abs_diff"] = vals.mean()
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT_DIR / "convergence_summary_by_run.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        log(str(agg))
    log(f"\nSaved: {OUT_DIR / 'convergence_summary_by_run.csv'}")

    log("\n" + "=" * 78)
    log("SWE DIFFERENCE PERSISTENCE THROUGH WY1985 (vs 5-year reference)")
    log("=" * 78)
    pivot = traj_df.pivot_table(index="days_after_1984_10_01", columns="run", values="swe_diff_mm", aggfunc="mean")
    with pd.option_context("display.width", 200):
        log(str(pivot))
    pivot.to_csv(OUT_DIR / "swe_diff_trajectory_mean_across_cells.csv")

    return agg, pivot


def make_plots(results, df, traj_df):
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = {"6mo": "#e41a1c", "1yr": "#377eb8", "2yr": "#4daf4a", "3yr": "#984ea3", "5yr_ref": "black"}
    for label, start_str in CANDIDATES:
        sub = df[df["run"] == label].sort_values("cell")
        ax.plot(sub["cell"], sub["model_swe_mm"], "o-", color=colors[label], label=label)
    ax.set_xticklabels(sub["cell"], rotation=45, ha="right")
    ax.set_ylabel("Snow-17 SWE at 1984-10-01 (mm)")
    ax.set_title("Snow-17 modeled SWE at 1984-10-01 by spin-up duration, per cell")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(OUT_DIR / "plots" / "swe_at_target_date_by_spinup.png", dpi=180)

    fig2, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True, sharex=True)
    for ax, (cell_name, cell) in zip(axes.ravel(), CELLS.items()):
        ref = results[cell_name]["5yr_ref"]
        for label, start_str in CANDIDATES:
            run = results[cell_name][label]
            common_time = run["time"][run["time"] >= TARGET_DATE]
            ref_idx = np.searchsorted(ref["time"].values, common_time.values)
            run_idx = np.searchsorted(run["time"].values, common_time.values)
            diff = run["swe"][run_idx] - ref["swe"][ref_idx]
            days = (common_time - TARGET_DATE).days
            ax.plot(days, diff, color=colors[label], label=label, linewidth=1.3)
        ax.set_title(f"{cell_name} (elev={cell['elevation_m']:.0f}m)", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("days after 1984-10-01")
    for ax in axes[:, 0]:
        ax.set_ylabel("SWE diff vs 5yr-ref (mm)")
    fig2.suptitle("SWE difference vs 5-year reference spin-up, through WY1985", fontsize=13)
    fig2.savefig(OUT_DIR / "plots" / "swe_diff_trajectory_grid.png", dpi=170)

    log(f"Saved plots to {OUT_DIR / 'plots'}")


def main():
    log(f"Output directory: {OUT_DIR}")
    log(f"Representative cells: {list(CELLS.keys())}")
    log(f"Fixed parameters (spin-up test only, NOT calibrated): {FIXED_PARAMS}")
    log(f"Candidates: {CANDIDATES}")
    log(f"Target date: {TARGET_DATE.date()}, run end: {END_DATE.date()}")

    forcing = extract_forcing_all_years()
    results = run_all_spinups(forcing)
    df, traj_df = analyze_convergence(results)
    agg, pivot = summarize_and_print(df, traj_df)
    make_plots(results, df, traj_df)

    log("\nDONE.")


if __name__ == "__main__":
    main()
