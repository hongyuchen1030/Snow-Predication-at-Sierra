#!/usr/bin/env python3
"""Extend the verified CPM index to 37 years by projecting ERA5 onto fixed EOF3."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(os.environ.get("PWD", str(Path(__file__).absolute().parents[1])))
DEFAULT_CPM_DIR = PROJECT_ROOT / "artifacts" / "cpm_era5_reproduction"
DEFAULT_SWE_TABLE = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"
DEFAULT_UPSTREAM_SUMMARY = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo" / "full37_patch_predictor_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_swe_attribution_37yr_fixed_cpm"
DEFAULT_ERA5_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl")
FILE_RE = re.compile(r"e5\.oper\.an\.pl\.128_129_z\.ll025sc\.(\d{10})_(\d{10})\.nc$")
GRAVITY = 9.80665
WET_MONTHS = {11, 12, 1, 2, 3, 4}


def require_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import netCDF4 as nc
        import pandas as pd
        import xarray as xr
        from scipy import stats
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing required dependency: {exc}") from exc
    return plt, nc, pd, xr, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpm-artifact-dir", type=Path, default=DEFAULT_CPM_DIR)
    parser.add_argument("--era5-input-path", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--swe-table-path", type=Path, default=DEFAULT_SWE_TABLE)
    parser.add_argument("--upstream-summary-path", type=Path, default=DEFAULT_UPSTREAM_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-water-year", type=int, default=1985)
    parser.add_argument("--end-water-year", type=int, default=2021)
    parser.add_argument("--run-monthly-analysis", action="store_true", default=True)
    parser.add_argument("--skip-monthly-analysis", dest="run_monthly_analysis", action="store_false")
    return parser.parse_args()


def month_starts(start, end, pd):
    current = pd.Timestamp(year=start.year, month=start.month, day=1)
    stop = pd.Timestamp(year=end.year, month=end.month, day=1)
    while current <= stop:
        yield current
        current = current + pd.offsets.MonthBegin(1)


def standardize_lon_360(values: np.ndarray) -> np.ndarray:
    return np.mod(values, 360.0)


def assign_water_year(index) -> np.ndarray:
    years = index.year.to_numpy()
    months = index.month.to_numpy()
    return np.where(months >= 11, years + 1, years)


def is_leap_day(timestamp) -> bool:
    """True for Feb 29. The original CPM calendar-day climatology
    (run_cpm_era5_eof_reproduction.py, --drop-leap-day=True) excludes Feb 29
    entirely rather than folding it into Feb 28 or inventing a climatology
    value for it, so every leap year contributes the same 365 calendar days
    as a non-leap year. Any code building a calendar-day ("MM-DD") key
    against that climatology must apply this same exclusion first."""
    return timestamp.month == 2 and timestamp.day == 29


def parse_day_from_path(path: Path, pd):
    match = FILE_RE.match(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d%H")


def list_daily_files(root: Path, start, end, pd):
    records = []
    for month_start in month_starts(start, end, pd):
        month_dir = root / month_start.strftime("%Y%m")
        if not month_dir.exists():
            continue
        for path in sorted(month_dir.glob("*.nc")):
            if "128_129_z" not in path.name:
                continue
            day = parse_day_from_path(path, pd)
            if day is None:
                continue
            if day < start or day > end:
                continue
            if day.month not in WET_MONTHS:
                continue
            records.append((day.normalize(), path))
    if not records:
        raise FileNotFoundError(f"No ERA5 Z files found under {root} for {start}..{end}.")
    return records


def validate_daily_records(records, start, end, pd):
    seen = {}
    for day, path in records:
        if day in seen:
            raise ValueError(f"Duplicated daily ERA5 file for {day.date()}: {path}")
        seen[day] = path
    expected_days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    wet_expected = [day for day in expected_days if day.month in WET_MONTHS]
    missing = [day.strftime("%Y-%m-%d") for day in wet_expected if day not in seen]
    if missing:
        raise ValueError(f"Missing wet-season days in extension: {missing[:10]}")
    return wet_expected, seen


def detect_indices(sample_path: Path, level_hpa: int, latitudes_target: np.ndarray, longitudes_target: np.ndarray, nc):
    with nc.Dataset(sample_path) as ds:
        levels = np.asarray(ds.variables["level"][:], dtype=np.int32)
        level_matches = np.where(levels == level_hpa)[0]
        if len(level_matches) != 1:
            raise ValueError(f"Could not resolve exact {level_hpa} hPa level.")
        level_index = int(level_matches[0])
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = standardize_lon_360(np.asarray(ds.variables["longitude"][:], dtype=np.float64))
        lat_mask = (lat >= latitudes_target.min()) & (lat <= latitudes_target.max())
        lon_mask = (lon >= longitudes_target.min()) & (lon <= longitudes_target.max())
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        return {
            "level_index": level_index,
            "latitudes": lat[lat_idx],
            "longitudes_360": lon[lon_idx],
            "lat_indices": lat_idx,
            "lon_indices": lon_idx,
            "source_units": getattr(ds.variables["Z"], "units", ""),
        }


def read_one_day(path: Path, level_index: int, lat_slice: slice, lon_slice: slice, nc):
    with nc.Dataset(path) as ds:
        z = np.asarray(ds.variables["Z"][:, level_index, lat_slice, lon_slice], dtype=np.float64)
        if z.shape[0] != 24:
            raise ValueError(f"Expected 24 hourly samples in {path}, found {z.shape[0]}")
        return z.mean(axis=0).astype(np.float32)


def read_one_day_task(task):
    day_i, path_str, level_index, lat_start, lat_stop, lon_start, lon_stop = task
    import netCDF4 as nc

    with nc.Dataset(path_str) as ds:
        z = np.asarray(ds.variables["Z"][:, level_index, lat_start:lat_stop, lon_start:lon_stop], dtype=np.float64)
        if z.shape[0] != 24:
            raise ValueError(f"Expected 24 hourly samples in {path_str}, found {z.shape[0]}")
        return day_i, z.mean(axis=0).astype(np.float32)


def fisher_ci(r: float, n: int, stats) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(0.975)
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x, ddof=1)


def linear_detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return corrcoef_safe(x[:-1], x[1:])


def effective_sample_size(n: int, r1_x: float, r1_y: float) -> float:
    if not np.isfinite(r1_x) or not np.isfinite(r1_y):
        return float(n)
    denom = 1.0 + r1_x * r1_y
    num = 1.0 - r1_x * r1_y
    if denom <= 0.0:
        return float(n)
    neff = n * num / denom
    return float(min(max(neff, 3.0), n))


def bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.asarray(pvals)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * m / rank)
        q[i] = val
        prev = val
    out = np.empty(m, dtype=float)
    out[order] = q
    return out.tolist()


def ols_fit(X: np.ndarray, y: np.ndarray, stats):
    n = len(y)
    p = X.shape[1]
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    sigma2 = rss / (n - p)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=n - p))
    hat = X @ xtx_inv @ X.T
    leverage = np.diag(hat)
    cooks = (resid**2 / (p * sigma2)) * (leverage / (1.0 - leverage) ** 2)
    return {"beta": beta, "r2": r2, "adj_r2": adj_r2, "se": se, "pvals": pvals, "cooks_distance": cooks}


def df_to_markdown(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def plot_scatter(x, y, xlabel, ylabel, title, stats_text, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    ax.scatter(x, y, color="tab:blue", alpha=0.85)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xx, intercept + slope * xx, color="tab:red", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_standardized_timeseries(df, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    years = df["water_year"].to_numpy()
    ax.plot(years, zscore(df["cpm_pc3_nov_apr_mean"].to_numpy()), marker="o", linewidth=1.5, label="CPM", color="tab:green")
    ax.plot(years, zscore(df["obs_swe"].to_numpy()), marker="o", linewidth=1.5, label="Observed SWE", color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Water year")
    ax.set_ylabel("Standardized value")
    ax.set_title("Standardized fixed-CPM annual index and observed SWE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loyo_scatter(obs, pred, stats_text, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(obs, pred, color="tab:purple")
    lo = min(np.min(obs), np.min(pred))
    hi = max(np.max(obs), np.max(pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    coef = np.polyfit(obs, pred, 1)
    xx = np.linspace(lo, hi, 100)
    ax.plot(xx, coef[1] + coef[0] * xx, color="tab:red", linewidth=1.2)
    ax.set_xlabel("Observed SWE")
    ax.set_ylabel("LOYO predicted SWE")
    ax.set_title("LOYO fixed-CPM SWE model")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_correlations(monthly_df, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(monthly_df))
    ax.bar(x, monthly_df["pearson_r"], color="teal")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_df["month"])
    ax.set_ylabel("Pearson r")
    ax.set_title("Monthly fixed-CPM vs observed SWE correlations")
    for xi, row in zip(x, monthly_df.itertuples(index=False)):
        ax.text(xi, row.pearson_r, f"p={row.p_value:.3f}\nq={row.q_value_bh:.3f}", ha="center", va="bottom" if row.pearson_r >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    plt, nc, pd, xr, stats = require_deps()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpm_dir = args.cpm_artifact_dir
    pc_df = pd.read_csv(cpm_dir / "era5_z500_pc1to6_daily.csv", parse_dates=["date"])
    anom_ds = xr.open_dataset(cpm_dir / "era5_z500_daily_anomalies.nc")
    clim_ds = xr.open_dataset(cpm_dir / "era5_z500_calendar_day_climatology.nc")
    eof_ds = xr.open_dataset(cpm_dir / "era5_z500_eof1to6.nc")
    metadata = json.loads((cpm_dir / "era5_z500_processing_metadata.json").read_text())
    upstream_summary = json.loads(args.upstream_summary_path.read_text())
    swe_table = pd.read_csv(args.swe_table_path)

    eof3 = eof_ds["eof_unweighted_sigma_scaled_m"].sel(component=3)
    pc3_std = float(eof_ds["pc_standard_deviation"].sel(component=3).item())
    lat = eof3["latitude"].values.astype(np.float64)
    lon = eof3["longitude"].values.astype(np.float64)
    weights = np.sqrt(np.cos(np.deg2rad(lat))).astype(np.float64)
    eof_weighted = eof3.values.astype(np.float64) * weights[:, None] / pc3_std

    base_anoms = anom_ds["z500_m_anomaly"]
    base_proj = np.tensordot(base_anoms.values.astype(np.float64) * weights[None, :, None], eof_weighted, axes=([1, 2], [0, 1]))
    base_pc3 = pc_df["PC3"].to_numpy(dtype=float)
    overlap_check = pd.DataFrame(
        {
            "date": pd.to_datetime(pc_df["date"]),
            "saved_pc3": base_pc3,
            "projected_pc3_from_saved_anomalies": base_proj,
        }
    )
    overlap_check["difference"] = overlap_check["projected_pc3_from_saved_anomalies"] - overlap_check["saved_pc3"]
    overlap_check.to_csv(args.output_dir / "fixed_cpm_projection_overlap_check.csv", index=False)
    base_projection_r = corrcoef_safe(base_pc3, base_proj)

    last_saved_date = pd.Timestamp(pc_df["date"].max())
    target_end_date = pd.Timestamp(year=args.end_water_year, month=4, day=30)
    extension_rows = []
    if target_end_date > last_saved_date:
        ext_start = last_saved_date + pd.Timedelta(days=1)
        records = list_daily_files(args.era5_input_path, ext_start, target_end_date, pd)
        wet_expected, day_to_path = validate_daily_records(records, ext_start, target_end_date, pd)
        dates_ext = pd.DatetimeIndex(wet_expected)
        print(f"Projecting extension days: {len(dates_ext)} daily fields from {ext_start.date()} to {target_end_date.date()}", flush=True)
        sample_path = records[0][1]
        info = detect_indices(sample_path, int(metadata.get("level_hpa", 500)), lat, lon, nc)
        lat_start = int(info["lat_indices"][0])
        lat_stop = int(info["lat_indices"][-1] + 1)
        lon_start = int(info["lon_indices"][0])
        lon_stop = int(info["lon_indices"][-1] + 1)
        tasks = [
            (i, str(day_to_path[day]), int(info["level_index"]), lat_start, lat_stop, lon_start, lon_stop)
            for i, day in enumerate(dates_ext)
        ]
        daily_native = np.empty((len(dates_ext), len(info["latitudes"]), len(info["longitudes_360"])), dtype=np.float32)
        n_workers = min(8, os.cpu_count() or 1)
        print(f"Reading extension ERA5 daily means with {n_workers} worker(s).", flush=True)
        if n_workers <= 1:
            iterator = map(read_one_day_task, tasks)
        else:
            pool = mp.Pool(processes=n_workers)
            iterator = pool.imap(read_one_day_task, tasks, chunksize=8)
        try:
            for completed, (day_i, day_mean) in enumerate(iterator, start=1):
                daily_native[day_i] = day_mean / GRAVITY
                if completed == 1 or completed % 100 == 0 or completed == len(dates_ext):
                    print(f"Extension read progress: {completed}/{len(dates_ext)} ({dates_ext[day_i].strftime('%Y-%m-%d')})", flush=True)
        finally:
            if n_workers > 1:
                pool.close()
                pool.join()

        # The original CPM climatology (run_cpm_era5_eof_reproduction.py,
        # --drop-leap-day defaults to True) drops Feb 29 entirely before
        # building the calendar-day climatology, so it has exactly 365
        # month_day keys and no "02-29" entry -- leap years contribute the
        # same 365 calendar days as non-leap years, Feb 28 is not duplicated
        # and no Feb-29 climatology value is invented. Apply the identical
        # exclusion here so extension days can always find their climatology
        # key; without this, any leap year in the extension window (e.g.
        # 2020) raises KeyError('02-29') on the climatology lookup below.
        leap_mask = np.array([is_leap_day(ts) for ts in dates_ext])
        n_leap_days_dropped = int(leap_mask.sum())
        if n_leap_days_dropped:
            dropped_dates = [ts.strftime("%Y-%m-%d") for ts in dates_ext[leap_mask]]
            print(f"Dropping {n_leap_days_dropped} leap day(s) from the extension period to match "
                  f"the original climatology's 365-day convention: {dropped_dates}", flush=True)
        keep = ~leap_mask
        dates_ext = dates_ext[keep]
        daily_native = daily_native[keep]

        ext_da = xr.DataArray(
            daily_native,
            dims=("date", "latitude", "longitude"),
            coords={"date": dates_ext, "latitude": info["latitudes"], "longitude": info["longitudes_360"]},
        ).sortby("latitude").sortby("longitude")
        interp = ext_da.interp(latitude=lat, longitude=lon, method="linear", kwargs={"fill_value": "extrapolate"})
        print("Interpolated extension fields onto saved CPM grid.", flush=True)

        clim = clim_ds["z500_climatology_m"]
        month_days = np.array([ts.strftime("%m-%d") for ts in dates_ext], dtype=object)
        ext_anoms = np.empty_like(interp.values, dtype=np.float32)
        for md in sorted(set(month_days)):
            mask = month_days == md
            clim_map = clim.sel(month_day=md).values.astype(np.float32)
            ext_anoms[mask] = interp.values[mask].astype(np.float32) - clim_map
        print("Computed extension daily anomalies using saved calendar-day climatology.", flush=True)

        ext_proj = np.tensordot(ext_anoms.astype(np.float64) * weights[None, :, None], eof_weighted, axes=([1, 2], [0, 1]))
        print("Projected extension anomalies onto fixed EOF3.", flush=True)
        extension_rows = [
            {
                "date": day,
                "water_year": int(assign_water_year(pd.DatetimeIndex([day]))[0]),
                "PC3": float(score),
                "source": "projected_extension",
            }
            for day, score in zip(dates_ext, ext_proj)
        ]

        xr.Dataset(
            {
                "z500_m_anomaly": xr.DataArray(
                    ext_anoms,
                    dims=("date", "latitude", "longitude"),
                    coords={"date": dates_ext, "latitude": lat, "longitude": lon},
                )
            }
        ).to_netcdf(args.output_dir / "era5_z500_daily_anomalies_extension_wy2017_2021.nc", engine="scipy")

    combined_pc = pc_df[["date", "water_year", "PC3"]].copy()
    combined_pc["source"] = "saved_training_period"
    if extension_rows:
        ext_df = pd.DataFrame(extension_rows)
        combined_pc = pd.concat([combined_pc, ext_df], ignore_index=True)
    combined_pc = combined_pc.sort_values("date").reset_index(drop=True)
    combined_pc.to_csv(args.output_dir / "cpm_pc3_daily_extended_fixed_eof3.csv", index=False)

    annual_cpm = (
        combined_pc.groupby("water_year", as_index=False)
        .agg(
            cpm_pc3_nov_apr_mean=("PC3", "mean"),
            number_of_days=("PC3", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .sort_values("water_year")
    )
    annual_cpm = annual_cpm[(annual_cpm["water_year"] >= args.start_water_year) & (annual_cpm["water_year"] <= args.end_water_year)]
    annual_cpm.to_csv(args.output_dir / "cpm_pc3_nov_apr_by_water_year.csv", index=False)

    swe = swe_table[["water_year", "obs_swe"]].copy()
    swe["water_year"] = swe["water_year"].astype(int)
    swe = swe[(swe["water_year"] >= args.start_water_year) & (swe["water_year"] <= args.end_water_year)].sort_values("water_year")
    swe.to_csv(args.output_dir / "observed_swe_by_water_year.csv", index=False)

    overlap = swe.merge(annual_cpm, on="water_year", how="inner").sort_values("water_year").reset_index(drop=True)
    overlap.to_csv(args.output_dir / "cpm_swe_overlap_table.csv", index=False)

    years = overlap["water_year"].to_numpy()
    swe_values = overlap["obs_swe"].to_numpy(dtype=float)
    cpm_values = overlap["cpm_pc3_nov_apr_mean"].to_numpy(dtype=float)
    n = len(overlap)

    pearson_r, pearson_p = stats.pearsonr(cpm_values, swe_values)
    spearman_rho, spearman_p = stats.spearmanr(cpm_values, swe_values)
    slope, intercept, _, _, _ = stats.linregress(cpm_values, swe_values)
    ci_lo, ci_hi = fisher_ci(float(pearson_r), n, stats)

    plot_scatter(
        cpm_values,
        swe_values,
        "Wet-season CPM (fixed EOF3; Nov-Apr mean)",
        "Observed SWE",
        "Fixed-CPM vs observed SWE (37 years)",
        f"r={pearson_r:.3f}\np={pearson_p:.3g}\nn={n}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        args.output_dir / "cpm_vs_swe_scatter.png",
        plt,
    )
    plot_standardized_timeseries(overlap[["water_year", "cpm_pc3_nov_apr_mean", "obs_swe"]], args.output_dir / "cpm_swe_standardized_timeseries.png", plt)

    X = np.column_stack([np.ones(n), cpm_values])
    fit = ols_fit(X, swe_values, stats)
    beta0, beta1 = fit["beta"]
    loyo_preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xi = np.column_stack([np.ones(mask.sum()), cpm_values[mask]])
        yi = swe_values[mask]
        beta_i = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        loyo_preds.append(float(np.array([1.0, cpm_values[i]]) @ beta_i))
    loyo_preds = np.asarray(loyo_preds)
    loyo_r = corrcoef_safe(swe_values, loyo_preds)
    loyo_r2 = 1.0 - np.sum((swe_values - loyo_preds) ** 2) / np.sum((swe_values - np.mean(swe_values)) ** 2)
    loyo_rmse = float(np.sqrt(np.mean((swe_values - loyo_preds) ** 2)))
    loyo_mae = float(np.mean(np.abs(swe_values - loyo_preds)))
    pd.DataFrame({"water_year": years, "obs_swe": swe_values, "pred_swe": loyo_preds}).to_csv(args.output_dir / "cpm_swe_loyo_predictions.csv", index=False)
    plot_loyo_scatter(
        swe_values,
        loyo_preds,
        f"r={loyo_r:.3f}\nR²={loyo_r2:.3f}\nRMSE={loyo_rmse:.4f}\nMAE={loyo_mae:.4f}",
        args.output_dir / "cpm_swe_loyo_scatter.png",
        plt,
    )

    model_summary = pd.DataFrame(
        [
            {
                "model": "SWE ~ fixed_CPM",
                "intercept": beta0,
                "beta_cpm": beta1,
                "se_intercept": fit["se"][0],
                "se_beta_cpm": fit["se"][1],
                "p_intercept": fit["pvals"][0],
                "p_beta_cpm": fit["pvals"][1],
                "in_sample_R2": fit["r2"],
                "adjusted_R2": fit["adj_r2"],
                "LOYO_r": loyo_r,
                "LOYO_R2": loyo_r2,
                "LOYO_RMSE": loyo_rmse,
                "LOYO_MAE": loyo_mae,
            }
        ]
    )
    model_summary.to_csv(args.output_dir / "cpm_swe_model_summary.csv", index=False)

    detrended_r = corrcoef_safe(linear_detrend(cpm_values), linear_detrend(swe_values))
    loo_rs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        loo_rs.append(corrcoef_safe(cpm_values[mask], swe_values[mask]))
    loo_rs = np.asarray(loo_rs)
    influence_idx = int(np.nanargmax(np.abs(loo_rs - pearson_r)))
    r1_cpm = lag1_autocorr(cpm_values)
    r1_swe = lag1_autocorr(swe_values)
    neff = effective_sample_size(n, r1_cpm, r1_swe)
    t_eff = pearson_r * math.sqrt((neff - 2.0) / max(1.0e-12, 1.0 - pearson_r**2))
    p_eff = float(2.0 * (1.0 - stats.t.cdf(abs(t_eff), df=max(1.0, neff - 2.0))))
    robustness_df = pd.DataFrame(
        [
            {
                "raw_r": pearson_r,
                "detrended_r": detrended_r,
                "leave_one_out_min_r": float(np.nanmin(loo_rs)),
                "leave_one_out_max_r": float(np.nanmax(loo_rs)),
                "most_influential_year": int(years[influence_idx]),
                "max_leave_one_out_change_in_r": float(np.nanmax(np.abs(loo_rs - pearson_r))),
                "max_cooks_distance": float(np.nanmax(fit["cooks_distance"])),
                "lag1_autocorr_cpm": r1_cpm,
                "lag1_autocorr_swe": r1_swe,
                "effective_sample_size": neff,
                "effective_sample_size_p_value": p_eff,
            }
        ]
    )
    robustness_df.to_csv(args.output_dir / "cpm_swe_robustness_summary.csv", index=False)

    monthly_df = pd.DataFrame()
    if args.run_monthly_analysis:
        combined_pc["month"] = pd.to_datetime(combined_pc["date"]).dt.month
        month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
        monthly_rows = []
        for month in [11, 12, 1, 2, 3, 4]:
            sub = combined_pc[combined_pc["month"] == month]
            month_mean = sub.groupby("water_year", as_index=False)["PC3"].mean().rename(columns={"PC3": "cpm_month_mean"})
            joined = swe.merge(month_mean, on="water_year", how="inner").sort_values("water_year")
            r, p = stats.pearsonr(joined["cpm_month_mean"], joined["obs_swe"])
            monthly_rows.append({"month": month_names[month], "n": int(len(joined)), "pearson_r": float(r), "p_value": float(p)})
        monthly_df = pd.DataFrame(monthly_rows)
        monthly_df["q_value_bh"] = bh_adjust(monthly_df["p_value"].tolist())
        monthly_df.to_csv(args.output_dir / "cpm_monthly_swe_correlations.csv", index=False)
        plot_monthly_correlations(monthly_df, args.output_dir / "cpm_monthly_swe_correlations.png", plt)

    source_audit = f"""# Fixed-CPM 37-Year SWE Attribution Source Audit

- Base CPM artifact directory: `{args.cpm_artifact_dir}`
- Base daily PC source: `{args.cpm_artifact_dir / 'era5_z500_pc1to6_daily.csv'}`
- Base anomaly source: `{args.cpm_artifact_dir / 'era5_z500_daily_anomalies.nc'}`
- Base climatology source: `{args.cpm_artifact_dir / 'era5_z500_calendar_day_climatology.nc'}`
- Base EOF source: `{args.cpm_artifact_dir / 'era5_z500_eof1to6.nc'}`
- Extension ERA5 source: `{args.era5_input_path}`
- Fixed CPM definition used here: saved ERA5 EOF3 from the verified CPM reproduction
- Extension method: project additional ERA5 daily anomaly maps onto the fixed saved EOF3 basis
- Overlap check on saved period: corr(saved PC3, projection from saved anomalies) = `{base_projection_r:.12f}`
- SWE source table: `{args.swe_table_path}`
- Upstream target source recorded in metadata: `{upstream_summary['target_source']}`
- Shared overlap used here: `WY{int(years[0])}-WY{int(years[-1])}` with `n={n}`
- Important note: this is a fixed-EOF3 extension, not a full 37-year EOF refit.
"""
    (args.output_dir / "cpm_swe_source_audit.md").write_text(source_audit)

    primary_table = pd.DataFrame(
        [
            {
                "Predictor": "Fixed CPM EOF3 index",
                "Years": f"WY{int(years[0])}-WY{int(years[-1])}",
                "n": int(n),
                "Pearson r": float(pearson_r),
                "p-value": float(pearson_p),
                "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                "Spearman rho": float(spearman_rho),
                "Detrended r": float(detrended_r),
            }
        ]
    )
    model_table = pd.DataFrame(
        [
            {
                "Model": "SWE ~ fixed CPM",
                "In-sample R²": float(fit["r2"]),
                "Adjusted R²": float(fit["adj_r2"]),
                "LOYO r": float(loyo_r),
                "LOYO R²": float(loyo_r2),
                "LOYO RMSE": float(loyo_rmse),
                "LOYO MAE": float(loyo_mae),
            }
        ]
    )

    report = f"""# Fixed-CPM 37-Year SWE Attribution Report

## Inputs

- Base CPM artifact directory: `{args.cpm_artifact_dir}`
- SWE source table: `{args.swe_table_path}`
- Upstream metadata file: `{args.upstream_summary_path}`
- ERA5 extension source: `{args.era5_input_path}`

## Method

- The verified ERA5 CPM map was kept fixed as saved `EOF3`.
- Saved daily anomalies from the original CPM reproduction were used to verify the projection formula.
- Additional wet-season ERA5 days through `2021-04-30` were converted to daily Z500 anomalies using the saved calendar-day climatology and then projected onto the fixed EOF3 basis.
- This extends the CPM index to match the full 37-year SWE target period without changing the already verified CPM spatial pattern.
- This is not a full 37-year EOF refit.

## Overlap

- Overlap years: `WY{int(years[0])}-WY{int(years[-1])}`
- Number of paired years: `{n}`
- Overlap table: `{args.output_dir / 'cpm_swe_overlap_table.csv'}`

## Primary result table

{df_to_markdown(primary_table)}

## Model result table

{df_to_markdown(model_table)}

## Single-predictor model details

- projection check correlation on saved-period daily PC3 = {base_projection_r:.12f}
- intercept = {beta0:.6f}
- beta_cpm = {beta1:.6f}
- SE(beta_cpm) = {fit['se'][1]:.6f}
- p(beta_cpm) = {fit['pvals'][1]:.6f}
- Spearman p-value = {spearman_p:.6f}

## Robustness diagnostics

{df_to_markdown(robustness_df)}

## Monthly timing analysis

{df_to_markdown(monthly_df) if not monthly_df.empty else 'Not run.'}

## Artifact paths

- Source audit: `{args.output_dir / 'cpm_swe_source_audit.md'}`
- Extended daily PC3: `{args.output_dir / 'cpm_pc3_daily_extended_fixed_eof3.csv'}`
- Annual CPM table: `{args.output_dir / 'cpm_pc3_nov_apr_by_water_year.csv'}`
- Overlap table: `{args.output_dir / 'cpm_swe_overlap_table.csv'}`
- Scatter plot: `{args.output_dir / 'cpm_vs_swe_scatter.png'}`
- Standardized time series: `{args.output_dir / 'cpm_swe_standardized_timeseries.png'}`
- LOYO predictions: `{args.output_dir / 'cpm_swe_loyo_predictions.csv'}`
- Model summary: `{args.output_dir / 'cpm_swe_model_summary.csv'}`

## Direct conclusion

- Does the fixed CPM index show evidence of association with SWE over 37 years?
  - Answer from the raw annual correlation above.
- Was the sample extended using available ERA5 Z500?
  - Yes, by projecting the added ERA5 years onto the fixed verified CPM EOF3.
"""
    (args.output_dir / "report.md").write_text(report)

    print(f"Base CPM source: {args.cpm_artifact_dir / 'era5_z500_pc1to6_daily.csv'}")
    print(f"Extension ERA5 source: {args.era5_input_path}")
    print(f"Projection check corr(saved PC3, projected overlap): {base_projection_r:.12f}")
    print(f"Overlap years: WY{int(years[0])}-WY{int(years[-1])}")
    print(f"Number of years: {n}")
    print(f"r(fixed CPM, SWE): {pearson_r:.6f}")
    print(f"Spearman rho(fixed CPM, SWE): {spearman_rho:.6f}")
    print(f"LOYO correlation: {loyo_r:.6f}")
    print(f"Main artifact directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
