#!/usr/bin/env python3
"""Build physical daily TaiESM1 predictors for one-day Delta SWE pretraining.

The output intentionally contains physical fields only.  It neither creates a
train/validation split nor normalizes any channel.  Each completed calendar
year is a separate [365, 12, 120, 240] float32 NPY file, allowing safe resume
after an interactive allocation ends.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMIP6_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/CMIP6")
DEFAULT_OUT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1")
TARGET_LAT = np.arange(-89.25, 90.0, 1.5, dtype=np.float64)
TARGET_LON = np.arange(0.75, 360.0, 1.5, dtype=np.float64)
CHANNELS = (
    ("rlut", "rlut", None),
    ("zg_500", "zg", 50000),
    ("zg_50", "zg", 5000),
    ("ta_850", "ta", 85000),
    ("ta_50", "ta", 5000),
    ("ua_850", "ua", 85000),
    ("ua_50", "ua", 5000),
    ("va_850", "va", 85000),
    ("va_50", "va", 5000),
    ("hus_850", "hus", 85000),
    ("psl", "psl", None),
    ("tas", "tas", None),
)
EXPERIMENTS = {
    "historical": (CMIP6_ROOT / "CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1", range(1980, 2015)),
    "ssp370": (CMIP6_ROOT / "ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1", range(2015, 2100)),
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def source_files(root: Path, variable: str) -> list[Path]:
    base = root / "day" / variable
    grids = sorted(p for p in base.iterdir() if p.is_dir())
    if not grids:
        raise FileNotFoundError(f"No grid directories under {base}")
    versions = sorted(p for p in grids[0].iterdir() if p.is_dir())
    if not versions:
        raise FileNotFoundError(f"No version directory under {grids[0]}")
    files = sorted(versions[-1].glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files under {versions[-1]}")
    return files


def source_year_span(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d{4})\d{4}-(\d{4})\d{4}\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse CMIP date span from {path.name}")
    return int(match.group(1)), int(match.group(2))


def write_grid(path: Path) -> None:
    path.write_text(
        "gridtype = lonlat\n"
        "xsize    = 240\nysize    = 120\n"
        "xfirst   = 0.75\nxinc     = 1.5\n"
        "yfirst   = -89.25\nyinc     = 1.5\n",
        encoding="utf-8",
    )


def ensure_weight(out: Path, representative: Path) -> tuple[Path, Path]:
    grid = out / "target_grid_1p5deg.txt"
    weight = out / "weights" / "TaiESM1_day_atmos_bilinear_to_1p5deg.nc"
    if not grid.exists():
        write_grid(grid)
    if not weight.exists():
        weight.parent.mkdir(parents=True, exist_ok=True)
        run(["cdo", "-O", "-L", f"genbil,{grid}", str(representative), str(weight)])
    return grid, weight


def data_var(ds: xr.Dataset, expected: str) -> xr.DataArray:
    if expected in ds.data_vars:
        return ds[expected]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    raise KeyError(f"Cannot identify {expected}; found {list(ds.data_vars)}")


def norm_lon(values: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=np.float64), 360.0)


def validate_grid(da: xr.DataArray) -> None:
    lat = np.asarray(da["lat" if "lat" in da.coords else "latitude"].values)
    lon = norm_lon(da["lon" if "lon" in da.coords else "longitude"].values)
    if da.shape[-2:] != (120, 240) or not np.allclose(lat, TARGET_LAT) or not np.allclose(lon, TARGET_LON):
        raise ValueError(f"Unexpected regridded grid: shape={da.shape[-2:]}")


def pressure_coordinate(da: xr.DataArray) -> str:
    for coord in da.coords:
        attrs = da[coord].attrs
        if attrs.get("standard_name") == "air_pressure" or attrs.get("units") == "Pa":
            return coord
    raise ValueError(f"No pressure coordinate found for {da.name}")


def bilinear_to_target(da: xr.DataArray) -> xr.DataArray:
    """Apply bilinear interpolation on TaiESM1's rectilinear lat/lon grid."""
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    da = da.assign_coords({lon_name: norm_lon(da[lon_name].values)}).sortby(lon_name).sortby(lat_name)
    return da.interp({lat_name: TARGET_LAT, lon_name: TARGET_LON}, method="linear")


def year_output(out: Path, experiment: str, year: int) -> Path:
    return out / "years" / experiment / f"predictors_physical_{year}.npy"


def progress_path(out: Path, experiment: str, year: int, channel: int) -> Path:
    return out / "progress" / experiment / str(year) / f"channel_{channel:02d}.json"


def ensure_year_array(out: Path, experiment: str, year: int) -> np.memmap:
    path = year_output(out, experiment, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(365, 12, 120, 240))
        arr[:] = np.nan
        del arr
    return np.lib.format.open_memmap(path, mode="r+")


def process_source(
    out: Path, grid: Path, weight: Path, experiment: str, channel_index: int, channel_name: str,
    variable: str, level_pa: int | None, source: Path, requested_years: set[int], unit_log: dict[str, str],
) -> None:
    start, end = source_year_span(source)
    overlap = sorted(requested_years.intersection(range(start, end + 1)))
    if not overlap:
        return
    pending = [year for year in overlap if not progress_path(out, experiment, year, channel_index).exists()]
    if not pending:
        return
    with tempfile.TemporaryDirectory(prefix="taiesm1-day-") as temp_name:
        temp_path = Path(temp_name) / "regridded.nc"
        operators = [f"remap,{grid},{weight}"]
        if level_pa is not None:
            # The leading dash makes level selection the nested input operator
            # consumed by CDO's outer remap operator.
            operators.append(f"-sellevel,{level_pa}")
        run(["cdo", "-O", "-L", *operators, str(source), str(temp_path)])
        with xr.open_dataset(temp_path, decode_times=True, use_cftime=True) as ds:
            da = data_var(ds, variable)
            if level_pa is not None:
                plev = pressure_coordinate(da)
                if level_pa not in np.asarray(da[plev].values):
                    raise ValueError(f"{source}: required exact level {level_pa} Pa is absent")
                da = da.sel({plev: level_pa})
            validate_grid(da)
            unit_log.setdefault(channel_name, str(da.attrs.get("units", "")))
            years = np.asarray([int(value.year) for value in da.time.values], dtype=np.int32)
            for year in pending:
                idx = np.flatnonzero(years == year)
                if idx.size != 365:
                    raise ValueError(f"{channel_name} {experiment} {year}: expected 365 days, got {idx.size}")
                values = np.asarray(da.isel(time=idx).values, dtype=np.float32)
                if values.shape != (365, 120, 240):
                    raise ValueError(f"{channel_name} {experiment} {year}: unexpected shape {values.shape}")
                target = ensure_year_array(out, experiment, year)
                target[:, channel_index] = values
                target.flush()
                del target
                p = progress_path(out, experiment, year, channel_index)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"source": str(source), "channel": channel_name, "units": unit_log[channel_name]}) + "\n")
                print(f"completed {experiment} {year} channel={channel_name}", flush=True)


def write_metadata(out: Path, units: dict[str, str]) -> None:
    rows: list[dict[str, object]] = []
    for experiment, (_, years) in EXPERIMENTS.items():
        for year in years:
            path = year_output(out, experiment, year)
            if not path.exists():
                continue
            arr = np.lib.format.open_memmap(path, mode="r")
            if arr.shape != (365, 12, 120, 240) or not np.isfinite(arr).any():
                continue
            valid = np.isfinite(arr).all(axis=(2, 3))
            np.save(out / "years" / experiment / f"validity_mask_{year}.npy", valid)
            for day in range(1, 366):
                rows.append({"model": "TaiESM1", "experiment": experiment, "year": year, "day_of_year": day, "time_index": year * 1000 + day})
    if not rows:
        return
    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out / "metadata.npz",
        model=np.array([row["model"] for row in rows]), experiment=np.array([row["experiment"] for row in rows]),
        year=np.array([row["year"] for row in rows], dtype=np.int16),
        day_of_year=np.array([row["day_of_year"] for row in rows], dtype=np.int16),
        time_index=np.array([row["time_index"] for row in rows], dtype=np.int32),
        channel_names=np.array([name for name, _, _ in CHANNELS]), lat=TARGET_LAT, lon=TARGET_LON,
    )
    (out / "channel_units.json").write_text(json.dumps(units, indent=2) + "\n")
    summary = {"records": len(rows), "channels": [name for name, _, _ in CHANNELS], "shape_per_year": [365, 12, 120, 240], "units": units}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stop-after-channel", type=int, default=None, help="Smoke-test/resume aid; process channels through this zero-based index.")
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    representative = source_files(EXPERIMENTS["historical"][0], "rlut")[0]
    grid, weight = ensure_weight(out, representative)
    (out / "regridding_method.json").write_text(json.dumps({
        "method": "bilinear", "implementation": "CDO remap with climate-utils/2025.01",
        "target_lat": TARGET_LAT.tolist(), "target_lon": TARGET_LON.tolist(),
    }, indent=2) + "\n")
    units: dict[str, str] = {}
    for channel_index, (channel_name, variable, level_pa) in enumerate(CHANNELS):
        if args.stop_after_channel is not None and channel_index > args.stop_after_channel:
            break
        for experiment, (root, years) in EXPERIMENTS.items():
            requested = set(years)
            for source in source_files(root, variable):
                process_source(out, grid, weight, experiment, channel_index, channel_name, variable, level_pa, source, requested, units)
    write_metadata(out, units)
    (out / "README.md").write_text(
        "# TaiESM1 daily 12-variable physical predictors\n\n"
        "Each yearly NPY is `[365, 12, 120, 240]` physical float32 data on the seasonal pipeline's 1.5-degree global grid. "
        "Channel order and calendar-safe metadata are in `metadata.npz`; no normalization or split has been applied.\n"
    )
    print(f"DONE output={out}", flush=True)


if __name__ == "__main__":
    main()
