#!/usr/bin/env python3
"""Run the ERA5 CPM EOF reproduction and ERA-Interim comparison."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(os.environ.get("PWD", str(Path(__file__).absolute().parents[1])))
DEFAULT_ERA5_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl")
DEFAULT_ERAI_DIR = PROJECT_ROOT / "artifacts" / "cpm_era_interim_reproduction"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_era5_reproduction"
FILE_RE = re.compile(r"e5\.oper\.an\.pl\.128_129_z\.ll025sc\.(\d{10})_(\d{10})\.nc$")
GRAVITY = 9.80665
WET_MONTHS = {11, 12, 1, 2, 3, 4}


def require_deps():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        import netCDF4 as nc
        import pandas as pd
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing required dependency: {exc}") from exc
    return ccrs, cfeature, plt, nc, pd, xr


@dataclass(frozen=True)
class Domain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-input-path", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--erai-reference-dir", type=Path, default=DEFAULT_ERAI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="1981-11-01")
    parser.add_argument("--end-date", default="2016-04-30")
    parser.add_argument("--lat-min", type=float, default=20.0)
    parser.add_argument("--lat-max", type=float, default=75.0)
    parser.add_argument("--lon-min", type=float, default=190.0)
    parser.add_argument("--lon-max", type=float, default=270.0)
    parser.add_argument("--plot-lat-min", type=float, default=25.0)
    parser.add_argument("--plot-lat-max", type=float, default=60.0)
    parser.add_argument("--plot-lon-min", type=float, default=210.0)
    parser.add_argument("--plot-lon-max", type=float, default=250.0)
    parser.add_argument("--n-eofs", type=int, default=6)
    parser.add_argument("--level-hpa", type=int, default=500)
    parser.add_argument("--interp-to-1deg", action="store_true", default=True)
    parser.add_argument("--no-interp-to-1deg", dest="interp_to_1deg", action="store_false")
    parser.add_argument("--drop-leap-day", action="store_true", default=True)
    parser.add_argument("--keep-leap-day", dest="drop_leap_day", action="store_false")
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def write_dataset_netcdf(dataset, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        dataset.to_netcdf(tmp_path, engine="scipy", mode="w")
        shutil.move(str(tmp_path), str(target))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def month_starts(start: "pd.Timestamp", end: "pd.Timestamp", pd):
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


def parse_day_from_path(path: Path, pd):
    match = FILE_RE.match(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d%H")


def list_daily_files(root: Path, start, end, pd) -> list[tuple["pd.Timestamp", Path]]:
    records: list[tuple["pd.Timestamp", Path]] = []
    for month_start in month_starts(start, end, pd):
        month_dir = root / month_start.strftime("%Y%m")
        if not month_dir.exists():
            continue
        for path in sorted(month_dir.glob("*.nc")):
            day = parse_day_from_path(path, pd)
            if day is None:
                continue
            if day < start or day > end:
                continue
            if day.month not in WET_MONTHS:
                continue
            records.append((day.normalize(), path))
    if not records:
        raise FileNotFoundError(f"No ERA5 files found under {root} for {start}..{end}.")
    return records


def validate_daily_records(records, start, end, pd):
    seen = {}
    duplicates = []
    for day, path in records:
        if day in seen:
            duplicates.append(str(path))
        seen[day] = path
    if duplicates:
        raise ValueError(f"Duplicated daily ERA5 files: {duplicates[:5]}")

    expected_days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    wet_expected = [day for day in expected_days if day.month in WET_MONTHS]
    missing = [day.strftime("%Y-%m-%d") for day in wet_expected if day not in seen]
    if missing:
        raise ValueError(f"Missing wet-season days: {missing[:10]}")
    return wet_expected, seen


def detect_indices(sample_path: Path, domain: Domain, level_hpa: int, nc):
    with nc.Dataset(sample_path) as ds:
        levels = np.asarray(ds.variables["level"][:], dtype=np.int32)
        level_matches = np.where(levels == level_hpa)[0]
        if len(level_matches) != 1:
            raise ValueError(f"Could not resolve exact {level_hpa} hPa level.")
        level_index = int(level_matches[0])
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = standardize_lon_360(np.asarray(ds.variables["longitude"][:], dtype=np.float64))
        lat_mask = (lat >= domain.lat_min) & (lat <= domain.lat_max)
        lon_mask = (lon >= domain.lon_min) & (lon <= domain.lon_max)
        if not lat_mask.any() or not lon_mask.any():
            raise ValueError("ERA5 domain selection returned no cells.")
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        return {
            "level_index": level_index,
            "level_value_hpa": int(levels[level_index]),
            "latitudes": lat[lat_idx],
            "longitudes_360": lon[lon_idx],
            "lat_indices": lat_idx,
            "lon_indices": lon_idx,
            "lat_descending": bool(lat[lat_idx[0]] > lat[lat_idx[-1]]),
            "source_units": getattr(ds.variables["Z"], "units", ""),
        }


def read_one_day(task):
    day_i, path, level_index, lat_start, lat_stop, lon_start, lon_stop = task
    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        z = np.asarray(ds.variables["Z"][:, level_index, lat_start:lat_stop, lon_start:lon_stop], dtype=np.float64)
        times = np.asarray(ds.variables["time"][:])
        if z.shape[0] != 24:
            raise ValueError(f"Expected 24 hourly samples in {path}, found {z.shape[0]}")
        if len(times) != 24:
            raise ValueError(f"Expected 24 time steps in {path}, found {len(times)}")
        return day_i, z.mean(axis=0).astype(np.float32)


def read_daily_means(dates, day_to_path, info, n_workers):
    lat_start = int(info["lat_indices"][0])
    lat_stop = int(info["lat_indices"][-1] + 1)
    lon_start = int(info["lon_indices"][0])
    lon_stop = int(info["lon_indices"][-1] + 1)
    tasks = [
        (day_i, str(day_to_path[day]), int(info["level_index"]), lat_start, lat_stop, lon_start, lon_stop)
        for day_i, day in enumerate(dates)
    ]
    nlat = len(info["lat_indices"])
    nlon = len(info["lon_indices"])
    daily = np.empty((len(dates), nlat, nlon), dtype=np.float32)
    print(f"Reading ERA5 daily means with {n_workers} worker(s).", flush=True)
    if n_workers <= 1:
        iterator = map(read_one_day, tasks)
    else:
        pool = mp.Pool(processes=n_workers)
        iterator = pool.imap(read_one_day, tasks, chunksize=8)
    try:
        for completed, (day_i, day_mean) in enumerate(iterator, start=1):
            daily[day_i] = day_mean
            if completed == 1 or completed % 100 == 0 or completed == len(dates):
                print(f"Reading daily means: {completed}/{len(dates)} ({dates[day_i].strftime('%Y-%m-%d')})", flush=True)
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()
    return daily


def interpolate_daily_grid(values: np.ndarray, dates, water_year, latitudes, longitudes_360, target_lats, target_lons, xr):
    da = xr.DataArray(
        values,
        dims=("date", "latitude", "longitude"),
        coords={
            "date": dates,
            "water_year": ("date", water_year),
            "latitude": latitudes,
            "longitude": longitudes_360,
        },
    ).sortby("latitude").sortby("longitude")
    interp = da.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return interp.values.astype(np.float32), interp["latitude"].values.astype(np.float64), interp["longitude"].values.astype(np.float64)


def calendar_day_anomalies(daily_values: np.ndarray, dates):
    month_day = np.array([ts.strftime("%m-%d") for ts in dates], dtype=object)
    unique_md = sorted(set(month_day))
    climatology = np.empty((len(unique_md), daily_values.shape[1], daily_values.shape[2]), dtype=np.float32)
    anomalies = np.empty_like(daily_values, dtype=np.float32)
    counts = {}
    for i, md in enumerate(unique_md):
        mask = month_day == md
        clim = daily_values[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        climatology[i] = clim
        anomalies[mask] = daily_values[mask] - clim
        counts[md] = int(mask.sum())
    return anomalies, {"month_day": unique_md, "climatology": climatology, "counts": counts}


def build_daily_dataset(name, values, dates, water_year, latitudes, longitudes, xr):
    return xr.Dataset(
        {
            name: xr.DataArray(
                values,
                dims=("date", "latitude", "longitude"),
                coords={
                    "date": dates,
                    "water_year": ("date", water_year),
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
            )
        }
    )


def create_climatology_dataset(clim_info, latitudes, longitudes, xr):
    return xr.Dataset(
        {
            "z500_climatology_m": xr.DataArray(
                clim_info["climatology"],
                dims=("month_day", "latitude", "longitude"),
                coords={
                    "month_day": clim_info["month_day"],
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
            )
        }
    )


def weighted_spatial_corr(a, b, latitudes):
    w = np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, a.shape[1]))
    mask = np.isfinite(a) & np.isfinite(b)
    x = a[mask]
    y = b[mask]
    ww = w[mask]
    ww = ww / ww.sum()
    xbar = np.sum(ww * x)
    ybar = np.sum(ww * y)
    cov = np.sum(ww * (x - xbar) * (y - ybar))
    varx = np.sum(ww * (x - xbar) ** 2)
    vary = np.sum(ww * (y - ybar) ** 2)
    if varx <= 0.0 or vary <= 0.0:
        return np.nan
    return float(cov / np.sqrt(varx * vary))


def plot_eof_grid(eof_maps, variance_ratio, longitudes_west, latitudes, domain, title, ccrs, cfeature, plt, out_path: Path):
    proj = ccrs.PlateCarree()
    n_modes = eof_maps.shape[0]
    ncols = 2
    nrows = math.ceil(n_modes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes).ravel()
    vmax = max(float(np.nanmax(np.abs(eof_maps))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    lon2d, lat2d = np.meshgrid(longitudes_west, latitudes)
    for idx, ax in enumerate(axes[:n_modes]):
        cf = ax.contourf(lon2d, lat2d, eof_maps[idx], levels=levels, cmap="RdBu_r", extend="both", transform=proj)
        ax.set_extent([domain.lon_min - 360.0, domain.lon_max - 360.0, domain.lat_min, domain.lat_max], crs=proj)
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        try:
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.set_title(f"EOF{idx + 1} ({variance_ratio[idx] * 100:.2f}%)", fontsize=11)
    for ax in axes[n_modes:]:
        ax.remove()
    cbar = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.045, pad=0.04)
    cbar.set_label("EOF loading scaled to one standard deviation of PC (m)")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_window(data, variance_ratio, latitudes, longitudes_west, lat_min, lat_max, lon_min_w, lon_max_w, title, ccrs, cfeature, plt, out_path: Path):
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes_west >= lon_min_w) & (longitudes_west <= lon_max_w)
    sub = data[np.ix_(lat_mask, lon_mask)]
    sub_lat = latitudes[lat_mask]
    sub_lon = longitudes_west[lon_mask]
    lon2d, lat2d = np.meshgrid(sub_lon, sub_lat)
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": proj})
    vmax = max(float(np.nanmax(np.abs(sub))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    cf = ax.contourf(lon2d, lat2d, sub, levels=levels, cmap="RdBu_r", extend="both", transform=proj)
    ax.set_extent([lon_min_w, lon_max_w, lat_min, lat_max], crs=proj)
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    try:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35)
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(f"{title}\n({variance_ratio * 100:.2f}% variance)")
    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("EOF loading scaled to one standard deviation of PC (m)")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pc_grid(pc_df, plt, out_path: Path, title: str):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    dates = pc_df["date"]
    for idx, ax in enumerate(axes, start=1):
        series = pc_df[f"PC{idx}"]
        smooth = series.rolling(window=31, center=True, min_periods=1).mean()
        ax.plot(dates, series, color="0.7", linewidth=0.5)
        ax.plot(dates, smooth, color="tab:blue", linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_title(f"PC{idx}")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reference_comparison(erai_map, era5_map, latitudes, longitudes_west, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    lon2d, lat2d = np.meshgrid(longitudes_west, latitudes)
    vmax = max(float(np.nanmax(np.abs(erai_map))), float(np.nanmax(np.abs(era5_map))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    cf = axes[0].contourf(lon2d, lat2d, erai_map, levels=levels, cmap="RdBu_r", extend="both")
    axes[0].set_title("ERA-Interim EOF3 Reference")
    axes[1].contourf(lon2d, lat2d, era5_map, levels=levels, cmap="RdBu_r", extend="both")
    axes[1].set_title("Best-Match ERA5 EOF")
    for ax in axes:
        ax.set_xlim(longitudes_west.min(), longitudes_west.max())
        ax.set_ylim(latitudes.min(), latitudes.max())
        ax.set_xlabel("Longitude (deg W)")
        ax.set_ylabel("Latitude")
    fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.06, pad=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def df_to_markdown_table(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def main() -> int:
    ccrs, cfeature, plt, nc, pd, xr = require_deps()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    domain = Domain(args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    output_paths = {
        "native_daily": args.output_dir / "era5_z500_daily_wetseason_native_ll025.nc",
        "daily_means": args.output_dir / "era5_z500_daily_wetseason.nc",
        "daily_anoms": args.output_dir / "era5_z500_daily_anomalies.nc",
        "clim": args.output_dir / "era5_z500_calendar_day_climatology.nc",
        "eof_nc": args.output_dir / "era5_z500_eof1to6.nc",
        "pc_csv": args.output_dir / "era5_z500_pc1to6_daily.csv",
        "summary_csv": args.output_dir / "era5_z500_pca_summary.csv",
        "quality_json": args.output_dir / "era5_data_quality_summary.json",
        "metadata_json": args.output_dir / "era5_z500_processing_metadata.json",
        "mode_summary_csv": args.output_dir / "era5_vs_erai_cpm_mode_summary.csv",
        "mode_summary_json": args.output_dir / "era5_vs_erai_cpm_mode_summary.json",
        "eof_png": args.output_dir / "era5_z500_eof1to6.png",
        "pc_png": args.output_dir / "era5_z500_pc1to6_daily.png",
        "best_match_png": args.output_dir / "era5_best_match_vs_erai_eof3.png",
        "best_window_png": args.output_dir / "era5_best_match_figure1a_window.png",
        "report_md": args.output_dir / "report.md",
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    erai_eof_ds = xr.open_dataset(args.erai_reference_dir / "era_interim_z500_eof1to6.nc")
    erai_eof3 = erai_eof_ds["eof_unweighted_sigma_scaled_m"].sel(component=3)
    erai_lat = erai_eof3["latitude"].values.astype(np.float64)
    erai_lon = erai_eof3["longitude"].values.astype(np.float64)
    erai_lon_w = np.where(erai_lon > 180.0, erai_lon - 360.0, erai_lon)
    erai_map = erai_eof3.values.astype(np.float64)
    erai_pc = pd.read_csv(args.erai_reference_dir / "era_interim_z500_pc1to6_daily.csv", parse_dates=["date"])
    erai_pc3 = erai_pc[["date", "PC3"]].rename(columns={"PC3": "ERAI_PC3"})

    records = list_daily_files(args.era5_input_path, start, end, pd)
    wet_expected, day_to_path = validate_daily_records(records, start, end, pd)
    dates = pd.DatetimeIndex(wet_expected)
    print(f"Resolved {len(records)} ERA5 daily files across {len(dates)} wet-season days.", flush=True)
    sample_path = records[0][1]
    info = detect_indices(sample_path, domain, args.level_hpa, nc)
    daily_geopotential = read_daily_means(dates, day_to_path, info, args.n_workers)
    daily_z500_m = (daily_geopotential / GRAVITY).astype(np.float32)
    water_year = assign_water_year(dates)
    leap_days = [ts.strftime("%Y-%m-%d") for ts in dates if ts.month == 2 and ts.day == 29]
    excluded_dates = []
    if args.drop_leap_day and leap_days:
        keep = np.array([not (ts.month == 2 and ts.day == 29) for ts in dates], dtype=bool)
        excluded_dates = leap_days
        dates = dates[keep]
        water_year = water_year[keep]
        daily_z500_m = daily_z500_m[keep]

    native_ds = build_daily_dataset(
        "z500_m_daily_native",
        daily_z500_m,
        dates,
        water_year,
        info["latitudes"],
        info["longitudes_360"],
        xr,
    )
    native_ds["z500_m_daily_native"].attrs.update({"units": "m", "source_units": info["source_units"]})
    write_dataset_netcdf(native_ds, output_paths["native_daily"])

    analysis_values = daily_z500_m
    analysis_lat = info["latitudes"]
    analysis_lon = info["longitudes_360"]
    if args.interp_to_1deg:
        target_lats = np.arange(args.lat_min, args.lat_max + 0.001, 1.0, dtype=np.float64)
        target_lons = np.arange(args.lon_min, args.lon_max + 0.001, 1.0, dtype=np.float64)
        analysis_values, analysis_lat, analysis_lon = interpolate_daily_grid(
            daily_z500_m,
            dates,
            water_year,
            info["latitudes"],
            info["longitudes_360"],
            target_lats,
            target_lons,
            xr,
        )
        print("Interpolated ERA5 daily fields to 1 x 1 degree grid.", flush=True)

    daily_ds = build_daily_dataset("z500_m_daily", analysis_values, dates, water_year, analysis_lat, analysis_lon, xr)
    write_dataset_netcdf(daily_ds, output_paths["daily_means"])

    anomalies, clim_info = calendar_day_anomalies(analysis_values, dates)
    write_dataset_netcdf(build_daily_dataset("z500_m_anomaly", anomalies, dates, water_year, analysis_lat, analysis_lon, xr), output_paths["daily_anoms"])
    write_dataset_netcdf(create_climatology_dataset(clim_info, analysis_lat, analysis_lon, xr), output_paths["clim"])

    weights = np.sqrt(np.cos(np.deg2rad(analysis_lat))).astype(np.float64)
    weighted = anomalies.astype(np.float64) * weights[None, :, None]
    if np.isnan(weighted).any():
        raise ValueError("Unexpected NaNs found in ERA5 weighted anomalies.")
    X = weighted.reshape(weighted.shape[0], -1)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    print("Completed ERA5 deterministic SVD.", flush=True)
    pcs = u[:, : args.n_eofs] * s[: args.n_eofs]
    explained_variance = (s ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    cumulative = np.cumsum(explained_variance_ratio)
    pc_std = pcs.std(axis=0, ddof=1)
    eof_weighted = vt[: args.n_eofs].reshape(args.n_eofs, len(analysis_lat), len(analysis_lon))
    eof_unweighted = eof_weighted / weights[None, :, None]
    eof_sigma = eof_unweighted * pc_std[:, None, None]

    era5_lon_w = np.where(analysis_lon > 180.0, analysis_lon - 360.0, analysis_lon)
    if not np.array_equal(analysis_lat, erai_lat) or not np.array_equal(analysis_lon, erai_lon):
        raise ValueError("ERA5 analysis grid does not match saved ERA-Interim reference grid.")

    pc_df = pd.DataFrame({"date": dates, "water_year": water_year})
    for mode in range(args.n_eofs):
        pc_df[f"PC{mode + 1}"] = pcs[:, mode]
    pc_df.to_csv(output_paths["pc_csv"], index=False)

    summary_df = pd.DataFrame(
        {
            "component": np.arange(1, args.n_eofs + 1),
            "explained_variance": explained_variance[: args.n_eofs],
            "explained_variance_ratio": explained_variance_ratio[: args.n_eofs],
            "cumulative_explained_variance_ratio": cumulative[: args.n_eofs],
            "pc_standard_deviation": pc_std[: args.n_eofs],
            "singular_value": s[: args.n_eofs],
        }
    )
    summary_df.to_csv(output_paths["summary_csv"], index=False)

    eof_ds = xr.Dataset(
        {
            "eof_unweighted_sigma_scaled_m": xr.DataArray(
                eof_sigma,
                dims=("component", "latitude", "longitude"),
                coords={"component": np.arange(1, args.n_eofs + 1), "latitude": analysis_lat, "longitude": analysis_lon},
            ),
            "explained_variance_ratio": xr.DataArray(
                explained_variance_ratio[: args.n_eofs],
                dims=("component",),
                coords={"component": np.arange(1, args.n_eofs + 1)},
            ),
            "explained_variance": xr.DataArray(
                explained_variance[: args.n_eofs],
                dims=("component",),
                coords={"component": np.arange(1, args.n_eofs + 1)},
            ),
            "singular_values": xr.DataArray(
                s[: args.n_eofs],
                dims=("component",),
                coords={"component": np.arange(1, args.n_eofs + 1)},
            ),
            "pc_standard_deviation": xr.DataArray(
                pc_std[: args.n_eofs],
                dims=("component",),
                coords={"component": np.arange(1, args.n_eofs + 1)},
            ),
        }
    )
    write_dataset_netcdf(eof_ds, output_paths["eof_nc"])

    plot_eof_grid(eof_sigma, explained_variance_ratio[: args.n_eofs], era5_lon_w, analysis_lat, domain, "ERA5 Wet-Season Z500 EOF1-EOF6", ccrs, cfeature, plt, output_paths["eof_png"])
    plot_pc_grid(pc_df, plt, output_paths["pc_png"], "ERA5 Wet-Season Daily PCs (gray raw, blue 31-day mean)")

    daily_corr = erai_pc3.merge(pc_df, on="date", how="inner")
    plot_lat_mask = (analysis_lat >= args.plot_lat_min) & (analysis_lat <= args.plot_lat_max)
    plot_lon_mask = (analysis_lon >= args.plot_lon_min) & (analysis_lon <= args.plot_lon_max)
    mode_rows = []
    for mode in range(args.n_eofs):
        era5_map = eof_sigma[mode]
        spatial_full_raw = weighted_spatial_corr(era5_map, erai_map, analysis_lat)
        spatial_plot_raw = weighted_spatial_corr(
            era5_map[np.ix_(plot_lat_mask, plot_lon_mask)],
            erai_map[np.ix_(plot_lat_mask, plot_lon_mask)],
            analysis_lat[plot_lat_mask],
        )
        temporal_raw = float(np.corrcoef(daily_corr[f"PC{mode + 1}"].to_numpy(), daily_corr["ERAI_PC3"].to_numpy())[0, 1])
        raw_values = np.array([spatial_full_raw, spatial_plot_raw, temporal_raw], dtype=float)
        sign_used = 1 if np.nanmean(raw_values) >= 0 else -1
        spatial_full = spatial_full_raw * sign_used
        spatial_plot = spatial_plot_raw * sign_used
        temporal = temporal_raw * sign_used
        composite = float(np.nanmean(np.abs([spatial_full, spatial_plot, temporal])))
        mode_rows.append(
            {
                "ERA5_mode": int(mode + 1),
                "explained_variance_percent": float(explained_variance_ratio[mode] * 100.0),
                "spatial_r_full_domain_vs_ERAInterim_EOF3": spatial_full,
                "spatial_r_plot_window_vs_ERAInterim_EOF3": spatial_plot,
                "temporal_r_vs_ERAInterim_PC3": temporal,
                "sign_used": int(sign_used),
                "composite_abs_mean_score": composite,
                "notes": "",
            }
        )

    mode_df = pd.DataFrame(mode_rows)
    best_idx = mode_df["composite_abs_mean_score"].idxmax()
    best_mode = int(mode_df.loc[best_idx, "ERA5_mode"])
    best_sign = int(mode_df.loc[best_idx, "sign_used"])
    mode_df.loc[best_idx, "notes"] = "Best overall CPM match by mean absolute spatial/temporal correlation."
    mode_df.to_csv(output_paths["mode_summary_csv"], index=False)
    output_paths["mode_summary_json"].write_text(mode_df.to_json(orient="records", indent=2))

    best_map = eof_sigma[best_mode - 1] * best_sign
    plot_single_window(
        best_map,
        explained_variance_ratio[best_mode - 1],
        analysis_lat,
        era5_lon_w,
        args.plot_lat_min,
        args.plot_lat_max,
        args.plot_lon_min - 360.0,
        args.plot_lon_max - 360.0,
        f"Best-Match ERA5 EOF{best_mode} on Figure 1a Window",
        ccrs,
        cfeature,
        plt,
        output_paths["best_window_png"],
    )
    plot_reference_comparison(erai_map, best_map, analysis_lat, era5_lon_w, output_paths["best_match_png"])

    quality = {
        "era5_source": str(args.era5_input_path),
        "expected_wet_season_days_in_range": len(wet_expected),
        "actual_daily_files_found": len(records),
        "excluded_dates": excluded_dates,
        "actual_days_used_for_pca": int(len(dates)),
        "pca_matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "latitude_count": int(len(analysis_lat)),
        "longitude_count": int(len(analysis_lon)),
        "first_timestamp": str(records[0][0]),
        "last_timestamp": str(records[-1][0]),
        "source_units": info["source_units"],
        "level_hpa": args.level_hpa,
    }
    output_paths["quality_json"].write_text(json.dumps(quality, indent=2))

    metadata = {
        "era5_source": str(args.era5_input_path),
        "erai_reference_dir": str(args.erai_reference_dir),
        "analysis_period_start": str(start.date()),
        "analysis_period_end": str(end.date()),
        "domain": domain.__dict__,
        "plot_window": {
            "lat_min": args.plot_lat_min,
            "lat_max": args.plot_lat_max,
            "lon_min_west": 150.0,
            "lon_max_west": 110.0,
        },
        "interpolated_to_1deg": bool(args.interp_to_1deg),
        "drop_leap_day": bool(args.drop_leap_day),
        "best_match_mode": best_mode,
        "best_match_sign": best_sign,
    }
    output_paths["metadata_json"].write_text(json.dumps(metadata, indent=2))

    report = f"""# ERA5 CPM Comparison Report

## Goal

Test whether ERA5 pressure-level Z500 contains the same CPM-like mode as the reproduced ERA-Interim EOF3 baseline.

## Inputs

- ERA5 source: `{args.era5_input_path}`
- ERA-Interim reference directory: `{args.erai_reference_dir}`
- Variable: `Z` at `500 hPa`
- Units converted with `z500_m = geopotential / {GRAVITY}`
- Wet-season period used for both datasets: `{start.date()}` through `{end.date()}`
- EOF/PCA domain: `20N-75N, 90W-170W`
- CPM display window: `25N-60N, 110W-150W`

## ERA5 Variance Explained

{df_to_markdown_table(summary_df)}

## ERA5 Versus ERA-Interim CPM Matching Table

{df_to_markdown_table(mode_df.drop(columns=["composite_abs_mean_score"]))}

## Best Match

- Best ERA5 mode: `EOF{best_mode}` / `PC{best_mode}`
- Sign used relative to saved ERA5 EOF/PC pair: `{best_sign:+d}`
- Full-domain spatial correlation versus ERA-Interim EOF3: `{mode_df.loc[best_idx, 'spatial_r_full_domain_vs_ERAInterim_EOF3']:.3f}`
- Figure-1a-window spatial correlation versus ERA-Interim EOF3: `{mode_df.loc[best_idx, 'spatial_r_plot_window_vs_ERAInterim_EOF3']:.3f}`
- Daily temporal correlation versus ERA-Interim PC3: `{mode_df.loc[best_idx, 'temporal_r_vs_ERAInterim_PC3']:.3f}`

## Interpretation

The best-match ERA5 mode was selected by the largest mean absolute correlation across:

1. weighted spatial correlation on the full EOF/PCA domain;
2. weighted spatial correlation on the Figure 1a CPM display window;
3. daily temporal correlation against ERA-Interim PC3 over overlapping wet-season dates.

All sign choices preserve EOF/PC pairing within ERA5 and were selected only to align ERA5 modes with the ERA-Interim reference sign convention.

## Artifacts

- Numeric summary: `{output_paths['mode_summary_csv']}`
- ERA5 EOF NetCDF: `{output_paths['eof_nc']}`
- ERA5 daily PCs: `{output_paths['pc_csv']}`
- ERA5 EOF1-EOF6 plot: `{output_paths['eof_png']}`
- ERA5 best-match vs ERA-Interim EOF3: `{output_paths['best_match_png']}`
- ERA5 best-match Figure 1a window plot: `{output_paths['best_window_png']}`
"""
    output_paths["report_md"].write_text(report)

    variance_text = ", ".join(f"EOF{int(r.component)}={float(r.explained_variance_ratio)*100:.2f}%" for r in summary_df.itertuples(index=False))
    print(f"ERA5 source: {args.era5_input_path}")
    print(f"Analysis period: {start.date()} to {end.date()}")
    print("Spatial domain: 20.0-75.0N, 90.0-170.0W")
    print(f"PCA matrix shape: {X.shape[0]} x {X.shape[1]}")
    print(f"ERA5 EOF1-EOF6 variance explained: {variance_text}")
    print(f"Best-match ERA5 component: EOF{best_mode}")
    print(f"Best-match temporal correlation vs ERA-Interim PC3: {mode_df.loc[best_idx, 'temporal_r_vs_ERAInterim_PC3']:.3f}")
    print(f"Main artifact directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
