#!/usr/bin/env python3
"""
Compute ERA5 mean-sea-level-pressure EOF/PC products for candidate SLP domains.
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ERA5_SLP_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5/e5.oper.an.sfc")
PSCRATCH_ROOT = Path(
    os.environ.get(
        "PSCRATCH",
        "/pscratch/sd/h/hyvchen",
    )
).expanduser()
OUTPUT_DIR = PSCRATCH_ROOT / "Snow-Predication-at-Sierra" / "era5_slp_domain_pcs"
COMMON_MONTHLY_CACHE = OUTPUT_DIR / "era5_slp_common_monthly_mean_sep1984_mar2021_lat20_90.nc"
NETCDF_ENGINE = "netcdf4"
FILE_PATTERN = re.compile(r"^\d{6}$")
MONTHS_OF_INTEREST = {9, 10, 11, 12, 1, 2, 3}
ANALYSIS_START_MONTH_KEY = "198409"
ANALYSIS_END_MONTH_KEY = "202103"
LAT_SUPERSET_MIN = 20.0
LAT_SUPERSET_MAX = 90.0
N_MODES = 6
N_WORKERS = max(1, min(16, int(os.environ.get("ERA5_SLP_PCA_N_WORKERS", "16"))))

DOMAIN_SPECS = {
    "PNA_SLP": {
        "lat_min": 20.0,
        "lat_max": 85.0,
        "lon_windows": [(120.0, 240.0)],
        "prefix": "PNA_SLP",
    },
    "NAO_SLP": {
        "lat_min": 20.0,
        "lat_max": 80.0,
        "lon_windows": [(270.0, 360.0), (0.0, 40.0)],
        "prefix": "NAO_SLP",
    },
    "NAM_SLP": {
        "lat_min": 20.0,
        "lat_max": 90.0,
        "lon_windows": [(0.0, 360.0)],
        "prefix": "NAM_SLP",
    },
}


@dataclass(frozen=True)
class RuntimeInfo:
    hostname: str
    slurm_job_id: str


@dataclass(frozen=True)
class DomainSummary:
    domain_name: str
    output_netcdf: str
    output_eof_png: str
    output_pc_png: str
    explained_variance_ratio: List[float]
    n_time: int
    n_valid_gridcells: int
    first_time: str
    last_time: str


def ensure_runtime_on_compute_node() -> RuntimeInfo:
    runtime = RuntimeInfo(
        hostname=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    if not runtime.slurm_job_id or "nid" not in runtime.hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")
    return runtime


def month_dir_to_path(month_key: str) -> Path:
    month_dir = ERA5_SLP_ROOT / month_key
    matches = sorted(month_dir.glob(f"e5.oper.an.sfc.128_151_msl.ll025sc.{month_key}0100_*.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one ERA5 MSL file for {month_key}, found {len(matches)} under {month_dir}")
    return matches[0]


def discover_month_keys() -> List[str]:
    month_keys: List[str] = []
    for path in sorted(ERA5_SLP_ROOT.iterdir()):
        if not path.is_dir() or not FILE_PATTERN.fullmatch(path.name):
            continue
        if path.name < ANALYSIS_START_MONTH_KEY or path.name > ANALYSIS_END_MONTH_KEY:
            continue
        month = int(path.name[4:6])
        if month in MONTHS_OF_INTEREST:
            month_keys.append(path.name)
    if not month_keys:
        raise FileNotFoundError(f"No ERA5 monthly directories found under {ERA5_SLP_ROOT}")
    return month_keys


def load_one_month(month_key: str) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, str]:
    path = month_dir_to_path(month_key)
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        data_var = ds["MSL"].sel(latitude=slice(LAT_SUPERSET_MAX, LAT_SUPERSET_MIN))
        monthly_mean = data_var.mean(dim="time", skipna=True, keep_attrs=True).load()
        field = np.asarray(monthly_mean.values, dtype=np.float32)
        lat = np.asarray(monthly_mean["latitude"].values, dtype=np.float32)
        lon = np.asarray(monthly_mean["longitude"].values, dtype=np.float32)
        units = str(monthly_mean.attrs.get("units", "Pa"))
    return month_key, field, lat, lon, units


def load_common_monthly_means(month_keys: Sequence[str]) -> Tuple[xr.DataArray, Dict[str, str]]:
    fields_by_key: Dict[str, np.ndarray] = {}
    source_files: Dict[str, str] = {}
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    units = "Pa"

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(load_one_month, key): key for key in month_keys}
        for future in as_completed(futures):
            key, field, lat, lon, units = future.result()
            if latitude is None:
                latitude = lat
                longitude = lon
            else:
                if not np.array_equal(latitude, lat) or not np.array_equal(longitude, lon):
                    raise ValueError(f"Latitude/longitude mismatch for {key}")
            fields_by_key[key] = field
            source_files[key] = str(month_dir_to_path(key))
            print(f"Loaded monthly mean MSL for {key}", flush=True)

    assert latitude is not None and longitude is not None
    ordered = [fields_by_key[key] for key in month_keys]
    times = np.asarray([np.datetime64(f"{key[:4]}-{key[4:6]}-01") for key in month_keys], dtype="datetime64[ns]")
    data = xr.DataArray(
        np.stack(ordered, axis=0),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": times,
            "latitude": latitude,
            "longitude": longitude,
        },
        name="MSL",
        attrs={"units": units, "source_dataset": "ERA5"},
    )
    return data, source_files


def load_or_create_common_monthly(month_keys: Sequence[str]) -> Tuple[xr.DataArray, Dict[str, str]]:
    if COMMON_MONTHLY_CACHE.exists():
        with xr.open_dataset(COMMON_MONTHLY_CACHE, engine=NETCDF_ENGINE) as ds:
            data = ds["MSL"].load()
        print(f"Reusing cached monthly mean file: {COMMON_MONTHLY_CACHE}", flush=True)
        source_files = {
            key: str(month_dir_to_path(key))
            for key in month_keys
        }
        return data, source_files

    data, source_files = load_common_monthly_means(month_keys)
    cache_ds = data.to_dataset(name="MSL")
    cache_ds.attrs["source_file_pattern"] = str(ERA5_SLP_ROOT / "YYYYMM" / "e5.oper.an.sfc.128_151_msl.ll025sc.YYYYMM0100_*.nc")
    cache_ds.to_netcdf(
        COMMON_MONTHLY_CACHE,
        engine=NETCDF_ENGINE,
        encoding={
            "MSL": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "chunksizes": [1, data.sizes["latitude"], min(data.sizes["longitude"], 720)],
            }
        },
    )
    print(f"Cached monthly mean file: {COMMON_MONTHLY_CACHE}", flush=True)
    return data, source_files


def subset_domain(common: xr.DataArray, domain_name: str) -> xr.DataArray:
    spec = DOMAIN_SPECS[domain_name]
    subset = common.sel(latitude=slice(spec["lat_max"], spec["lat_min"]))
    pieces: List[xr.DataArray] = []
    for lon_min, lon_max in spec["lon_windows"]:
        if lon_min == 0.0 and lon_max == 360.0:
            pieces.append(subset)
        else:
            lon_slice = subset.sel(longitude=slice(lon_min, lon_max - 0.25 if lon_max == 360.0 else lon_max))
            pieces.append(lon_slice)
    if len(pieces) == 1:
        return pieces[0]
    return xr.concat(pieces, dim="longitude")


def compute_anomalies(domain_data: xr.DataArray) -> xr.DataArray:
    climatology = domain_data.groupby("time.month").mean("time", keep_attrs=True)
    anomaly = (domain_data.groupby("time.month") - climatology).astype(np.float32)
    anomaly.attrs.update(domain_data.attrs)
    anomaly.attrs["anomaly_definition"] = "monthly_mean - month_of_year_climatology"
    return anomaly


def compute_pca(anomaly: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(anomaly["latitude"].values, dtype=np.float64)
    cos_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    weighted = anomaly.values.astype(np.float32) * cos_weights[None, :, None].astype(np.float32)
    n_time, n_lat, n_lon = weighted.shape
    flat = weighted.reshape(n_time, n_lat * n_lon)
    valid_mask = np.all(np.isfinite(flat), axis=0)
    valid = flat[:, valid_mask].astype(np.float64)
    valid -= valid.mean(axis=0, keepdims=True)

    covariance = valid @ valid.T
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals > 0.0
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]
    if eigvals.size < N_MODES:
        raise ValueError(f"Not enough positive eigenvalues to compute {N_MODES} modes.")

    singular_values = np.sqrt(eigvals[:N_MODES])
    pcs = eigvecs[:, :N_MODES] * singular_values[None, :]
    eof_weighted = (valid.T @ eigvecs[:, :N_MODES]) / singular_values[None, :]

    eof_maps = np.full((N_MODES, n_lat * n_lon), np.nan, dtype=np.float64)
    eof_maps[:, valid_mask] = eof_weighted.T
    eof_maps = eof_maps.reshape(N_MODES, n_lat, n_lon)
    with np.errstate(invalid="ignore", divide="ignore"):
        eof_unweighted = eof_maps / cos_weights[None, :, None]

    explained_variance_ratio = eigvals[:N_MODES] / np.sum(eigvals)
    return pcs.astype(np.float32), eof_unweighted.astype(np.float32), explained_variance_ratio.astype(np.float32), valid_mask


def format_time(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def save_domain_netcdf(
    domain_name: str,
    anomaly: xr.DataArray,
    pcs: np.ndarray,
    eofs: np.ndarray,
    explained_variance_ratio: np.ndarray,
    source_files: Dict[str, str],
) -> Path:
    spec = DOMAIN_SPECS[domain_name]
    times = np.asarray(anomaly["time"].values, dtype="datetime64[ns]")
    years = np.asarray([int(str(t)[:4]) for t in times], dtype=np.int32)
    months = np.asarray([int(str(t)[5:7]) for t in times], dtype=np.int32)
    lat = np.asarray(anomaly["latitude"].values, dtype=np.float32)
    lon = np.asarray(anomaly["longitude"].values, dtype=np.float32)

    data_vars: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}
    for mode_idx in range(N_MODES):
        data_vars[f"PC{mode_idx + 1}"] = (("time",), pcs[:, mode_idx])
        data_vars[f"EOF{mode_idx + 1}"] = (("latitude", "longitude"), eofs[mode_idx])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "latitude": lat,
            "longitude": lon,
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
        },
        attrs={
            "domain_name": domain_name,
            "lat_bounds": json.dumps([spec["lat_min"], spec["lat_max"]]),
            "lon_bounds": json.dumps(spec["lon_windows"]),
            "source_file": str(ERA5_SLP_ROOT / "YYYYMM" / "e5.oper.an.sfc.128_151_msl.ll025sc.YYYYMM0100_*.nc"),
            "slp_variable_name": "MSL",
            "slp_units": str(anomaly.attrs.get("units", "Pa")),
            "time_subset_months": json.dumps(sorted(MONTHS_OF_INTEREST)),
            "anomaly_definition": str(anomaly.attrs.get("anomaly_definition", "")),
        },
    )
    ds["year"] = ("time", years)
    ds["month"] = ("time", months)
    ds["explained_variance_ratio"] = (("mode",), explained_variance_ratio)

    path = OUTPUT_DIR / f"{spec['prefix']}_ERA5_PC1_6_monthly.nc"
    encoding: Dict[str, Dict[str, object]] = {}
    for var_name, var in ds.data_vars.items():
        if var_name.startswith("EOF"):
            encoding[var_name] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "chunksizes": [min(len(lat), 181), min(len(lon), 720)],
            }
        elif var_name.startswith("PC"):
            encoding[var_name] = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
        elif var_name == "explained_variance_ratio":
            encoding[var_name] = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
    ds.to_netcdf(path, engine=NETCDF_ENGINE, encoding=encoding)
    return path


def save_eof_plot(domain_name: str, anomaly: xr.DataArray, eofs: np.ndarray, explained_variance_ratio: np.ndarray) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    lon = np.asarray(anomaly["longitude"].values, dtype=float)
    lat = np.asarray(anomaly["latitude"].values, dtype=float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    for mode_idx, ax in enumerate(axes.flat[:N_MODES]):
        field = eofs[mode_idx]
        vmax = np.nanmax(np.abs(field))
        mesh = ax.pcolormesh(lon2d, lat2d, field, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"EOF{mode_idx + 1} ({explained_variance_ratio[mode_idx] * 100.0:.1f}%)")
        ax.set_xlabel("Longitude (degE)")
        ax.set_ylabel("Latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.suptitle(f"{domain_name}: EOF1-EOF6 of ERA5 monthly MSL anomalies", fontsize=14)
    path = OUTPUT_DIR / f"{DOMAIN_SPECS[domain_name]['prefix']}_EOF1_6.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pc_plot(domain_name: str, times: np.ndarray, pcs: np.ndarray) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharex=True)
    x = np.asarray(times, dtype="datetime64[ns]")
    for mode_idx, ax in enumerate(axes.flat[:N_MODES]):
        ax.plot(x, pcs[:, mode_idx], color="#1f4e79", linewidth=1.1)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"PC{mode_idx + 1}")
        ax.set_ylabel("PC amplitude")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time")
    fig.suptitle(f"{domain_name}: PC1-PC6 time series", fontsize=14)
    path = OUTPUT_DIR / f"{DOMAIN_SPECS[domain_name]['prefix']}_PC1_6_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    runtime = ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    month_keys = discover_month_keys()
    print(f"Discovered {len(month_keys)} ERA5 month directories for Sep-Mar processing", flush=True)
    common_monthly, source_files = load_or_create_common_monthly(month_keys)

    summary_rows: List[DomainSummary] = []
    for domain_name in DOMAIN_SPECS:
        print(f"Processing domain {domain_name}", flush=True)
        domain_monthly = subset_domain(common_monthly, domain_name)
        domain_anomaly = compute_anomalies(domain_monthly)
        pcs, eofs, evr, valid_mask = compute_pca(domain_anomaly)
        netcdf_path = save_domain_netcdf(domain_name, domain_anomaly, pcs, eofs, evr, source_files)
        eof_png = save_eof_plot(domain_name, domain_anomaly, eofs, evr)
        pc_png = save_pc_plot(domain_name, np.asarray(domain_anomaly["time"].values), pcs)
        summary_rows.append(
            DomainSummary(
                domain_name=domain_name,
                output_netcdf=str(netcdf_path),
                output_eof_png=str(eof_png),
                output_pc_png=str(pc_png),
                explained_variance_ratio=[float(x) for x in evr.tolist()],
                n_time=int(domain_anomaly.sizes["time"]),
                n_valid_gridcells=int(np.sum(valid_mask)),
                first_time=format_time(domain_anomaly["time"].values[0]),
                last_time=format_time(domain_anomaly["time"].values[-1]),
            )
        )
        print(
            f"{domain_name}: saved {netcdf_path.name}, valid_gridcells={int(np.sum(valid_mask))}, "
            f"EVR1={float(evr[0]):.4f}",
            flush=True,
        )

    summary_payload = {
        "runtime": asdict(runtime),
        "source_root": str(ERA5_SLP_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "months_of_interest": sorted(MONTHS_OF_INTEREST),
        "analysis_month_key_start": ANALYSIS_START_MONTH_KEY,
        "analysis_month_key_end": ANALYSIS_END_MONTH_KEY,
        "n_modes": N_MODES,
        "n_workers": N_WORKERS,
        "domains": [asdict(row) for row in summary_rows],
    }
    summary_path = OUTPUT_DIR / "era5_slp_domain_pcs_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
