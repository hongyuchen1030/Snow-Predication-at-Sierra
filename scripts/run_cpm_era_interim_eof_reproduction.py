#!/usr/bin/env python3
"""Reproduce the ERA-Interim California Precipitation Mode EOF analysis."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(os.environ.get("PWD", str(Path(__file__).absolute().parents[1])))
DEFAULT_ERA_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/ERA-Interim/ei.oper.an.pl.nc")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_era_interim_reproduction"
FILE_RE = re.compile(r"ei\.oper\.an\.pl\.regn128sc\.(\d{10})\.nc$")
EXPECTED_HOURS = (0, 6, 12, 18)
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
        raise SystemExit(
            "Missing required dependency "
            f"{exc}. Run with numpy, pandas, xarray, matplotlib, and cartopy."
        ) from exc
    return ccrs, cfeature, plt, nc, pd, xr


@dataclass(frozen=True)
class Domain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era-input-path", type=Path, default=DEFAULT_ERA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="1981-11-01")
    parser.add_argument("--end-date", default="2016-04-30")
    parser.add_argument("--lat-min", type=float, default=20.0)
    parser.add_argument("--lat-max", type=float, default=75.0)
    parser.add_argument("--lon-min", type=float, default=190.0)
    parser.add_argument("--lon-max", type=float, default=270.0)
    parser.add_argument("--n-eofs", type=int, default=6)
    parser.add_argument("--anomaly-method", default="calendar_day")
    parser.add_argument("--var-name", default="Z_GDS4_ISBL")
    parser.add_argument("--level-hpa", type=int, default=500)
    parser.add_argument("--interp-to-1deg", action="store_true", default=True)
    parser.add_argument("--no-interp-to-1deg", dest="interp_to_1deg", action="store_false")
    parser.add_argument("--drop-leap-day", action="store_true", default=True)
    parser.add_argument("--keep-leap-day", dest="drop_leap_day", action="store_false")
    parser.add_argument("--flip-sign-components", type=int, nargs="*", default=[])
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--daily-means-path", type=Path)
    parser.add_argument("--daily-anomalies-path", type=Path)
    return parser.parse_args()


def month_starts(start: "pd.Timestamp", end: "pd.Timestamp", pd) -> Iterable["pd.Timestamp"]:
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


def parse_timestamp(path: Path, pd):
    match = FILE_RE.match(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d%H")


def list_files(root: Path, start, end, pd) -> list[tuple["pd.Timestamp", Path]]:
    candidates: list[tuple["pd.Timestamp", Path]] = []
    for month_start in month_starts(start, end, pd):
        month_dir = root / month_start.strftime("%Y%m")
        if not month_dir.exists():
            continue
        for path in sorted(month_dir.glob("*.nc")):
            ts = parse_timestamp(path, pd)
            if ts is None:
                continue
            if ts < start or ts > end + pd.Timedelta(hours=23):
                continue
            if ts.month not in WET_MONTHS:
                continue
            candidates.append((ts, path))
    if not candidates:
        raise FileNotFoundError(f"No ERA-Interim files found under {root} for {start}..{end}.")
    return candidates


def validate_timestamps(records: Sequence[tuple["pd.Timestamp", Path]], start, end, pd) -> dict:
    per_day: dict = defaultdict(list)
    duplicates: list[str] = []
    seen = set()
    for ts, path in records:
        if ts in seen:
            duplicates.append(str(path))
        seen.add(ts)
        per_day[ts.normalize()].append((ts, path))
    if duplicates:
        raise ValueError(f"Found duplicated timestamps: {duplicates[:5]}")

    expected_days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    wet_expected = [day for day in expected_days if day.month in WET_MONTHS]
    missing_days: list[str] = []
    incomplete_days: list[dict] = []
    extra_hours: list[dict] = []
    for day in wet_expected:
        entries = sorted(per_day.get(day, []), key=lambda pair: pair[0])
        hours = tuple(ts.hour for ts, _ in entries)
        if not entries:
            missing_days.append(day.strftime("%Y-%m-%d"))
            continue
        if hours != EXPECTED_HOURS:
            incomplete_days.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "hours_found": list(hours),
                    "expected_hours": list(EXPECTED_HOURS),
                }
            )
            extra = sorted(set(hours) - set(EXPECTED_HOURS))
            if extra:
                extra_hours.append({"date": day.strftime("%Y-%m-%d"), "extra_hours": extra})
    return {
        "per_day": per_day,
        "wet_expected_days": [day.strftime("%Y-%m-%d") for day in wet_expected],
        "missing_days": missing_days,
        "incomplete_days": incomplete_days,
        "extra_hours": extra_hours,
    }


def detect_lat_lon_indices(sample_path: Path, domain: Domain, var_name: str, level_hpa: int, nc):
    with nc.Dataset(sample_path) as ds:
        levels = ds.variables["lv_ISBL0"][:]
        level_matches = np.where(levels == level_hpa)[0]
        if len(level_matches) != 1:
            raise ValueError(f"Expected one exact {level_hpa} hPa level, found indices {level_matches.tolist()}.")
        level_index = int(level_matches[0])
        var = ds.variables[var_name]
        lat = np.asarray(ds.variables["g4_lat_1"][:], dtype=np.float64)
        lon = standardize_lon_360(np.asarray(ds.variables["g4_lon_2"][:], dtype=np.float64))
        lat_mask = (lat >= domain.lat_min) & (lat <= domain.lat_max)
        lon_mask = (lon >= domain.lon_min) & (lon <= domain.lon_max)
        if not lat_mask.any():
            raise ValueError("Latitude domain selection returned no grid cells.")
        if not lon_mask.any():
            raise ValueError("Longitude domain selection returned no grid cells.")
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        subset = np.asarray(var[level_index, lat_idx[0] : lat_idx[-1] + 1, lon_idx[0] : lon_idx[-1] + 1])
        info = {
            "latitudes": lat[lat_idx],
            "longitudes_360": lon[lon_idx],
            "level_index": level_index,
            "level_value_hpa": int(levels[level_index]),
            "lat_indices": lat_idx,
            "lon_indices": lon_idx,
            "var_units": getattr(var, "units", ""),
            "lat_descending": bool(lat_idx[0] < lat_idx[-1] and lat[lat_idx[0]] > lat[lat_idx[-1]]),
            "subset_shape": tuple(int(x) for x in subset.shape),
            "nan_count_sample": int(np.isnan(subset).sum()),
        }
    return info


def read_one_day(task) -> tuple[int, np.ndarray]:
    day_i, paths, level_index, lat_start, lat_stop, lon_start, lon_stop, var_name = task
    import netCDF4 as nc

    arrays = []
    for path in paths:
        with nc.Dataset(path) as ds:
            field = ds.variables[var_name][level_index, lat_start:lat_stop, lon_start:lon_stop]
            arrays.append(np.asarray(field, dtype=np.float64))
    return day_i, np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def read_daily_means(
    dates: Sequence["pd.Timestamp"],
    per_day: dict,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    level_index: int,
    var_name: str,
    nc,
    n_workers: int,
) -> np.ndarray:
    sample_count = len(dates)
    nlat = len(lat_idx)
    nlon = len(lon_idx)
    daily = np.empty((sample_count, nlat, nlon), dtype=np.float32)
    tasks = []
    lat_start, lat_stop = int(lat_idx[0]), int(lat_idx[-1] + 1)
    lon_start, lon_stop = int(lon_idx[0]), int(lon_idx[-1] + 1)
    for day_i, day in enumerate(dates):
        paths = [str(path) for _, path in sorted(per_day[day], key=lambda pair: pair[0])]
        tasks.append((day_i, paths, int(level_index), lat_start, lat_stop, lon_start, lon_stop, var_name))

    print(f"Reading daily means with {n_workers} worker(s).", flush=True)
    if n_workers <= 1:
        iterator = map(read_one_day, tasks)
    else:
        pool = mp.Pool(processes=n_workers)
        iterator = pool.imap(read_one_day, tasks, chunksize=8)
    try:
        for completed, (day_i, day_mean) in enumerate(iterator, start=1):
            daily[day_i] = day_mean
            if completed == 1 or completed % 100 == 0 or completed == sample_count:
                print(
                    f"Reading daily means: {completed}/{sample_count} ({dates[day_i].strftime('%Y-%m-%d')})",
                    flush=True,
                )
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()
    return daily


def calendar_day_anomalies(daily_values: np.ndarray, dates) -> tuple[np.ndarray, dict]:
    month_day = np.array([ts.strftime("%m-%d") for ts in dates], dtype=object)
    unique_md = sorted(set(month_day))
    climatology = np.empty((len(unique_md), daily_values.shape[1], daily_values.shape[2]), dtype=np.float32)
    anomalies = np.empty_like(daily_values, dtype=np.float32)
    counts = {}
    md_to_idx = {md: idx for idx, md in enumerate(unique_md)}
    for md in unique_md:
        mask = month_day == md
        clim = daily_values[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        climatology[md_to_idx[md]] = clim
        anomalies[mask] = daily_values[mask] - clim
        counts[md] = int(mask.sum())
    return anomalies, {
        "month_day": unique_md,
        "climatology": climatology,
        "counts": counts,
    }


def build_daily_dataset(name: str, values: np.ndarray, dates, water_year: np.ndarray, latitudes, longitudes, xr):
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


def create_climatology_dataset(clim_info: dict, latitudes, longitudes, xr):
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


def plot_eof_maps(eof_maps, variance_ratio, longitudes_west, latitudes, domain: Domain, ccrs, cfeature, plt, out_path: Path):
    proj = ccrs.PlateCarree()
    n_modes = eof_maps.shape[0]
    ncols = 2
    nrows = math.ceil(n_modes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.4 * nrows), subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes).ravel()
    vmax = max(float(np.nanmax(np.abs(eof_maps))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    lon2d, lat2d = np.meshgrid(longitudes_west, latitudes)
    for idx, ax in enumerate(axes[:n_modes]):
        data = eof_maps[idx]
        cf = ax.contourf(
            lon2d,
            lat2d,
            data,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            transform=proj,
        )
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
    fig.suptitle("ERA-Interim Wet-Season Z500 EOF1-EOF6", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_eof(data, variance_ratio, longitudes_west, latitudes, domain: Domain, title: str, ccrs, cfeature, plt, out_path: Path):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": proj})
    vmax = max(float(np.nanmax(np.abs(data))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    lon2d, lat2d = np.meshgrid(longitudes_west, latitudes)
    cf = ax.contourf(
        lon2d,
        lat2d,
        data,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
        transform=proj,
    )
    ax.set_extent([domain.lon_min - 360.0, domain.lon_max - 360.0, domain.lat_min, domain.lat_max], crs=proj)
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    try:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35)
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(f"{title} ({variance_ratio * 100:.2f}% variance)")
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("EOF loading scaled to one standard deviation of PC (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pc_grid(pc_df, plt, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    dates = pc_df["date"]
    for idx, ax in enumerate(axes, start=1):
        series = pc_df[f"PC{idx}"]
        smooth = series.rolling(window=31, center=True, min_periods=1).mean()
        ax.plot(dates, series, color="0.6", linewidth=0.5)
        ax.plot(dates, smooth, color="tab:blue", linewidth=1.2)
        ax.set_title(f"PC{idx}")
        ax.axhline(0.0, color="black", linewidth=0.5)
    fig.suptitle("ERA-Interim Wet-Season Daily PCs (gray raw, blue 31-day mean)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pc3(pc_df, plt, out_path: Path):
    fig, ax = plt.subplots(figsize=(15, 4.5))
    dates = pc_df["date"]
    series = pc_df["PC3"]
    smooth = series.rolling(window=31, center=True, min_periods=1).mean()
    ax.plot(dates, series, color="0.65", linewidth=0.6, label="Daily PC3")
    ax.plot(dates, smooth, color="tab:red", linewidth=1.3, label="31-day mean")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("ERA-Interim Wet-Season Daily PC3")
    ax.set_ylabel("PC3")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ccrs, cfeature, plt, nc, pd, xr = require_deps()

    native_daily_path = args.output_dir / "era_interim_z500_daily_wetseason_native_regn128sc.nc"
    daily_means_path = args.daily_means_path or (args.output_dir / "era_interim_z500_daily_wetseason.nc")
    daily_anoms_path = args.daily_anomalies_path or (args.output_dir / "era_interim_z500_daily_anomalies.nc")
    daily_clim_path = args.output_dir / "era_interim_z500_calendar_day_climatology.nc"
    quality_summary_path = args.output_dir / "era_interim_data_quality_summary.json"
    metadata_path = args.output_dir / "era_interim_z500_processing_metadata.json"
    eof_nc_path = args.output_dir / "era_interim_z500_eof1to6.nc"
    pc_csv_path = args.output_dir / "era_interim_z500_pc1to6_daily.csv"
    summary_csv_path = args.output_dir / "era_interim_z500_pca_summary.csv"
    eof_png_path = args.output_dir / "era_interim_z500_eof1to6.png"
    eof3_png_path = args.output_dir / "era_interim_z500_eof3_cpm_candidate.png"
    pc_grid_png_path = args.output_dir / "era_interim_z500_pc1to6_daily.png"
    pc3_png_path = args.output_dir / "era_interim_z500_pc3_daily.png"
    output_paths = [
        native_daily_path,
        daily_means_path,
        daily_anoms_path,
        daily_clim_path,
        quality_summary_path,
        metadata_path,
        eof_nc_path,
        pc_csv_path,
        summary_csv_path,
        eof_png_path,
        eof3_png_path,
        pc_grid_png_path,
        pc3_png_path,
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    domain = Domain(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if args.anomaly_method != "calendar_day":
        raise ValueError("Only --anomaly-method calendar_day is implemented.")
    if args.n_eofs < 1:
        raise ValueError("--n-eofs must be positive.")

    file_records = list_files(args.era_input_path, start, end, pd)
    checks = validate_timestamps(file_records, start, end, pd)
    if checks["missing_days"]:
        raise ValueError(f"Missing wet-season days: {checks['missing_days'][:10]}")
    if checks["incomplete_days"]:
        raise ValueError(f"Incomplete wet-season days: {checks['incomplete_days'][:5]}")

    day_dates = pd.DatetimeIndex(sorted(pd.Timestamp(day) for day in checks["wet_expected_days"]))
    print(f"Resolved {len(file_records)} 6-hourly files across {len(day_dates)} wet-season days.", flush=True)
    sample_path = file_records[0][1]
    subset_info = detect_lat_lon_indices(sample_path, domain, args.var_name, args.level_hpa, nc)
    latitudes = subset_info["latitudes"]
    longitudes_360 = subset_info["longitudes_360"]
    longitudes_west = np.where(longitudes_360 > 180.0, longitudes_360 - 360.0, longitudes_360)

    daily_geopotential = read_daily_means(
        day_dates,
        checks["per_day"],
        subset_info["lat_indices"],
        subset_info["lon_indices"],
        subset_info["level_index"],
        args.var_name,
        nc,
        args.n_workers,
    )
    daily_z500_m = (daily_geopotential / GRAVITY).astype(np.float32)
    print("Completed daily averaging and geopotential-to-height conversion.", flush=True)
    water_year = assign_water_year(day_dates)

    leap_days = [ts.strftime("%Y-%m-%d") for ts in day_dates if ts.month == 2 and ts.day == 29]
    excluded_dates: list[str] = []
    if args.drop_leap_day and leap_days:
        keep = np.array([not (ts.month == 2 and ts.day == 29) for ts in day_dates], dtype=bool)
        excluded_dates = leap_days
        day_dates = day_dates[keep]
        water_year = water_year[keep]
        daily_z500_m = daily_z500_m[keep]

    native_daily_ds = build_daily_dataset(
        "z500_m_daily_native",
        daily_z500_m,
        day_dates,
        water_year,
        latitudes,
        longitudes_360,
        xr,
    )
    native_daily_ds["z500_m_daily_native"].attrs.update(
        {
            "long_name": "Daily mean 500-hPa geopotential height on native ERA-Interim regn128sc grid",
            "units": "m",
            "derived_from": args.var_name,
            "source_units": subset_info["var_units"],
            "conversion": f"{args.var_name} / {GRAVITY}",
            "daily_samples_required": "00,06,12,18 UTC",
        }
    )
    write_dataset_netcdf(native_daily_ds, native_daily_path)

    analysis_values = daily_z500_m
    analysis_latitudes = latitudes
    analysis_longitudes_360 = longitudes_360
    if args.interp_to_1deg:
        target_lats = np.arange(args.lat_min, args.lat_max + 0.001, 1.0, dtype=np.float64)
        target_lons = np.arange(args.lon_min, args.lon_max + 0.001, 1.0, dtype=np.float64)
        analysis_values, analysis_latitudes, analysis_longitudes_360 = interpolate_daily_grid(
            daily_z500_m,
            day_dates,
            water_year,
            latitudes,
            longitudes_360,
            target_lats,
            target_lons,
            xr,
        )
        print("Interpolated daily fields to 1 x 1 degree analysis grid.", flush=True)

    daily_ds = build_daily_dataset(
        "z500_m_daily",
        analysis_values,
        day_dates,
        water_year,
        analysis_latitudes,
        analysis_longitudes_360,
        xr,
    )
    daily_ds["z500_m_daily"].attrs.update(
        {
            "long_name": "Daily mean 500-hPa geopotential height",
            "units": "m",
            "derived_from": args.var_name,
            "source_units": subset_info["var_units"],
            "conversion": f"{args.var_name} / {GRAVITY}",
            "daily_samples_required": "00,06,12,18 UTC",
            "interpolated_to_1deg": str(args.interp_to_1deg),
        }
    )
    write_dataset_netcdf(daily_ds, daily_means_path)

    anomalies, clim_info = calendar_day_anomalies(analysis_values, day_dates)
    print("Constructed calendar-day anomalies.", flush=True)
    anomaly_ds = build_daily_dataset(
        "z500_m_anomaly",
        anomalies,
        day_dates,
        water_year,
        analysis_latitudes,
        analysis_longitudes_360,
        xr,
    )
    anomaly_ds["z500_m_anomaly"].attrs.update(
        {
            "long_name": "Daily 500-hPa geopotential height anomaly",
            "units": "m",
            "anomaly_method": "calendar_day_climatology_over_analysis_period",
            "drop_leap_day": str(args.drop_leap_day),
        }
    )
    write_dataset_netcdf(anomaly_ds, daily_anoms_path)
    write_dataset_netcdf(
        create_climatology_dataset(clim_info, analysis_latitudes, analysis_longitudes_360, xr),
        daily_clim_path,
    )

    latitudes = analysis_latitudes
    longitudes_360 = analysis_longitudes_360
    longitudes_west = np.where(longitudes_360 > 180.0, longitudes_360 - 360.0, longitudes_360)
    weights = np.sqrt(np.cos(np.deg2rad(latitudes))).astype(np.float64)
    weighted_anoms = anomalies.astype(np.float64) * weights[None, :, None]
    if np.isnan(weighted_anoms).any():
        raise ValueError("Unexpected NaNs found in weighted anomalies.")
    X = weighted_anoms.reshape(weighted_anoms.shape[0], -1)
    n_time, n_space = X.shape
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    print("Completed deterministic SVD.", flush=True)
    pcs_dim = u[:, : args.n_eofs] * s[: args.n_eofs]
    explained_variance = (s ** 2) / (n_time - 1)
    ev_sum = explained_variance.sum()
    if ev_sum <= 0.0 or not np.isfinite(ev_sum):
        raise ValueError("Explained variance sum is non-positive; check anomaly construction and input period.")
    explained_variance_ratio = explained_variance / ev_sum
    cumulative_ratio = np.cumsum(explained_variance_ratio)

    eof_weighted_unit = vt[: args.n_eofs].reshape(args.n_eofs, len(latitudes), len(longitudes_360))
    pc_std = pcs_dim.std(axis=0, ddof=1)
    eof_weighted_sigma = eof_weighted_unit * pc_std[:, None, None]
    eof_unweighted_unit = eof_weighted_unit / weights[None, :, None]
    eof_unweighted_sigma = eof_weighted_sigma / weights[None, :, None]

    sign_flips = sorted(set(int(mode) for mode in args.flip_sign_components if 1 <= int(mode) <= args.n_eofs))
    for mode in sign_flips:
        idx = mode - 1
        pcs_dim[:, idx] *= -1.0
        eof_weighted_unit[idx] *= -1.0
        eof_weighted_sigma[idx] *= -1.0
        eof_unweighted_unit[idx] *= -1.0
        eof_unweighted_sigma[idx] *= -1.0

    pc_df = pd.DataFrame({"date": day_dates, "water_year": water_year})
    for mode in range(args.n_eofs):
        pc_df[f"PC{mode + 1}"] = pcs_dim[:, mode]
    pc_df.to_csv(pc_csv_path, index=False)

    summary_df = pd.DataFrame(
        {
            "component": np.arange(1, args.n_eofs + 1),
            "explained_variance": explained_variance[: args.n_eofs],
            "explained_variance_ratio": explained_variance_ratio[: args.n_eofs],
            "cumulative_explained_variance_ratio": cumulative_ratio[: args.n_eofs],
            "pc_standard_deviation": pc_std[: args.n_eofs],
            "singular_value": s[: args.n_eofs],
        }
    )
    summary_df.to_csv(summary_csv_path, index=False)

    eof_ds = xr.Dataset(
        {
            "eof_weighted_unit_norm": xr.DataArray(
                eof_weighted_unit,
                dims=("component", "latitude", "longitude"),
                coords={"component": np.arange(1, args.n_eofs + 1), "latitude": latitudes, "longitude": longitudes_360},
            ),
            "eof_unweighted_unit_norm": xr.DataArray(
                eof_unweighted_unit,
                dims=("component", "latitude", "longitude"),
                coords={"component": np.arange(1, args.n_eofs + 1), "latitude": latitudes, "longitude": longitudes_360},
            ),
            "eof_unweighted_sigma_scaled_m": xr.DataArray(
                eof_unweighted_sigma,
                dims=("component", "latitude", "longitude"),
                coords={"component": np.arange(1, args.n_eofs + 1), "latitude": latitudes, "longitude": longitudes_360},
            ),
            "explained_variance": xr.DataArray(
                explained_variance[: args.n_eofs],
                dims=("component",),
                coords={"component": np.arange(1, args.n_eofs + 1)},
            ),
            "explained_variance_ratio": xr.DataArray(
                explained_variance_ratio[: args.n_eofs],
                dims=("component",),
                coords={"component": np.arange(1, args.n_eofs + 1)},
            ),
            "cumulative_explained_variance_ratio": xr.DataArray(
                cumulative_ratio[: args.n_eofs],
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
            "latitude_weight_sqrt_cos": xr.DataArray(weights, dims=("latitude",), coords={"latitude": latitudes}),
        }
    )
    eof_ds.attrs.update(
        {
            "source_data_path": str(args.era_input_path),
            "source_variable": args.var_name,
            "source_units": subset_info["var_units"],
            "pressure_level_hpa": str(args.level_hpa),
            "geopotential_to_height_conversion": f"divide by {GRAVITY}",
            "analysis_period_start": str(start.date()),
            "analysis_period_end": str(end.date()),
            "season": "November-April",
            "domain_latitude": f"{args.lat_min} to {args.lat_max}",
            "domain_longitude_360": f"{args.lon_min} to {args.lon_max}",
            "weighting": "sqrt(cos(latitude)) applied once before SVD",
            "anomaly_method": "calendar-day climatology over analysis period",
            "drop_leap_day": str(args.drop_leap_day),
            "eof_normalization": "unit-norm EOFs from SVD with dimensional PCs; sigma-scaled physical EOFs saved for plotting",
            "sign_flipped_components": ",".join(str(mode) for mode in sign_flips) or "none",
        }
    )
    write_dataset_netcdf(eof_ds, eof_nc_path)
    print("Saved NetCDF and CSV outputs.", flush=True)

    plot_eof_maps(
        eof_unweighted_sigma,
        explained_variance_ratio[: args.n_eofs],
        longitudes_west,
        latitudes,
        domain,
        ccrs,
        cfeature,
        plt,
        eof_png_path,
    )
    if args.n_eofs >= 3:
        plot_single_eof(
            eof_unweighted_sigma[2],
            explained_variance_ratio[2],
            longitudes_west,
            latitudes,
            domain,
            "EOF3 CPM Candidate",
            ccrs,
            cfeature,
            plt,
            eof3_png_path,
        )
    plot_pc_grid(pc_df, plt, pc_grid_png_path)
    plot_pc3(pc_df, plt, pc3_png_path)
    print("Saved EOF and PC figures.", flush=True)

    quality_summary = {
        "era_source": str(args.era_input_path),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "wet_season_months": sorted(WET_MONTHS),
        "expected_wet_season_days_in_range": len(checks["wet_expected_days"]),
        "actual_complete_days_before_leap_handling": int(len(checks["per_day"])),
        "missing_days": checks["missing_days"],
        "incomplete_days": checks["incomplete_days"],
        "duplicate_timestamp_count": 0,
        "expected_synoptic_hours_per_day": list(EXPECTED_HOURS),
        "excluded_dates": excluded_dates,
        "leap_days_present_before_exclusion": leap_days,
        "actual_days_used_for_pca": int(len(day_dates)),
        "latitude_count": int(len(latitudes)),
        "longitude_count": int(len(longitudes_360)),
        "subset_shape_daily": [int(len(day_dates)), int(len(latitudes)), int(len(longitudes_360))],
        "pca_matrix_shape": [int(n_time), int(n_space)],
        "latitudes_descending": bool(latitudes[0] > latitudes[-1]),
        "longitudes_360_min": float(longitudes_360.min()),
        "longitudes_360_max": float(longitudes_360.max()),
        "sign_flipped_components": sign_flips,
    }
    quality_summary_path.write_text(json.dumps(quality_summary, indent=2))

    metadata = {
        "source_file_count": int(len(file_records)),
        "first_timestamp": str(file_records[0][0]),
        "last_timestamp": str(file_records[-1][0]),
        "sample_file": str(sample_path),
        "variable": args.var_name,
        "pressure_level_hpa": args.level_hpa,
        "resolved_level_index": subset_info["level_index"],
        "resolved_level_value_hpa": subset_info["level_value_hpa"],
        "source_units": subset_info["var_units"],
        "subset_shape": list(subset_info["subset_shape"]),
        "sample_nan_count": subset_info["nan_count_sample"],
        "domain": domain.__dict__,
        "daily_means_path": str(daily_means_path),
        "native_daily_path": str(native_daily_path),
        "daily_anomalies_path": str(daily_anoms_path),
        "daily_climatology_path": str(daily_clim_path),
        "eof_netcdf_path": str(eof_nc_path),
        "pc_csv_path": str(pc_csv_path),
        "summary_csv_path": str(summary_csv_path),
        "plots": {
            "eof1to6": str(eof_png_path),
            "eof3": str(eof3_png_path),
            "pc1to6": str(pc_grid_png_path),
            "pc3": str(pc3_png_path),
        },
        "weighting": "sqrt(cos(latitude))",
        "svd_library": "numpy.linalg.svd",
        "interpolated_to_1deg": bool(args.interp_to_1deg),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    variance_text = ", ".join(
        f"EOF{row.component}={row.explained_variance_ratio * 100:.2f}%"
        for row in summary_df.itertuples(index=False)
    )
    print(f"ERA-Interim source: {args.era_input_path}")
    print(f"Analysis period: {start.date()} to {end.date()}")
    print(
        "Spatial domain: "
        f"{args.lat_min:.1f}-{args.lat_max:.1f}N, "
        f"{360.0 - args.lon_max:.1f}-{360.0 - args.lon_min:.1f}W"
    )
    print("Daily samples used: 00, 06, 12, 18 UTC")
    print(f"PCA matrix shape: {n_time} x {n_space}")
    print(f"EOF1-EOF6 variance explained: {variance_text}")
    print("Does EOF3 visually match the paper CPM?: pending external comparison")
    print(f"Was a sign flip applied?: {'yes: ' + ','.join(map(str, sign_flips)) if sign_flips else 'no'}")
    print("Most CPM-like component: pending external comparison")
    print(f"Main artifact directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
