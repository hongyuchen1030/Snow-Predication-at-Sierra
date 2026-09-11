#!/usr/bin/env python3
"""
Rerun the Stone et al. (2023) AQM MCA method on each driving GCM's own SST and
WUS-D3 d02 precipitation, instead of HadISST + GPCC. Mirrors the paper's own
GFDL CM2.1 control-run validation (their Table/Fig 5), but for our 4 driving
GCMs under historical and ssp370, treated separately (no pooling).
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_atlantic_pc245_attribution import (  # noqa: E402
    AQM_PRECIP_LAT_MAX,
    AQM_PRECIP_LAT_MIN,
    AQM_PRECIP_LON_MAX,
    AQM_PRECIP_LON_MIN,
    AQM_SST_LAT_MAX,
    AQM_SST_LAT_MIN,
    SEASON_MONTHS_PRECIP,
    SEASON_MONTHS_SST,
    corr_with_series,
    detrend_along_time,
    seasonal_average_label_by_year,
    spatial_metrics,
)
from scripts.wus02_data_audit import build_wus_series  # noqa: E402
from scripts import run_z1z2_plus_amv_k5_loyo as loyo_ref  # noqa: E402
from snow_ml import data_wusd3  # noqa: E402
from snow_ml.data_wusd3 import Wusd3Dataset, WUSD3_FILE_MIDDLE, WUSD3_ROOT  # noqa: E402


data_wusd3.WUSD3_FILE_PREFIX.setdefault("prec", "prec")

CMIP6_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/CMIP6")

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "aqm_gcm_reconstruction"
PATTERNS_DIR = ARTIFACT_DIR / "patterns"
PLOTS_DIR = ARTIFACT_DIR / "plots"
INDICES_DIR = ARTIFACT_DIR / "indices"
LOYO_DIR = ARTIFACT_DIR / "loyo"

REFERENCE_HARMONIZED_NETCDF = (
    PROJECT_ROOT / "artifacts" / "atlantic_pc245_attribution" / "reference_patterns_harmonized.nc"
)
REFERENCE_BOX_LAT_MIN = 0.0
REFERENCE_BOX_LAT_MAX = 70.0
REFERENCE_BOX_LON_MIN = 280.0
REFERENCE_BOX_LON_MAX = 360.0

REGRID_RESOLUTION_DEG = 1.0
REGRID_BUFFER_DEG = 5.0

NINO34_LAT_MIN = -5.0
NINO34_LAT_MAX = 5.0
NINO34_LON_MIN = 190.0
NINO34_LON_MAX = 240.0

WUSD3_PROJECTION_RADIUS_METERS = 6370000.0

PAPER_OBSERVED = {
    "spatial_corr_sst": None,
    "spatial_corr_precip": None,
    "scf_mode2": 0.12,
    "s2_p2_corr": 0.54,
}
PAPER_GFDL = {
    "spatial_corr_sst": 0.59,
    "spatial_corr_precip": 0.78,
    "scf_mode2": 0.04,
    "s2_p2_corr": 0.41,
}


@dataclass(frozen=True)
class GCMSpec:
    name: str
    short: str
    variant: str
    tos_hist_dir: Path
    tos_ssp_dir: Path
    wusd3_hist_id: str
    wusd3_ssp_id: str


GCM_SPECS: List[GCMSpec] = [
    GCMSpec(
        name="EC-Earth3",
        short="ec-earth3",
        variant="r1i1p1f1",
        tos_hist_dir=CMIP6_ROOT / "CMIP/EC-Earth-Consortium/EC-Earth3/historical/r1i1p1f1/Omon/tos/gn",
        tos_ssp_dir=CMIP6_ROOT / "ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp370/r1i1p1f1/Omon/tos/gn",
        wusd3_hist_id="ec-earth3_r1i1p1f1_2_historical_bc",
        wusd3_ssp_id="ec-earth3_r1i1p1f1_2_ssp370_bc",
    ),
    GCMSpec(
        name="MIROC6",
        short="miroc6",
        variant="r1i1p1f1",
        tos_hist_dir=CMIP6_ROOT / "CMIP/MIROC/MIROC6/historical/r1i1p1f1/Omon/tos/gn",
        tos_ssp_dir=CMIP6_ROOT / "ScenarioMIP/MIROC/MIROC6/ssp370/r1i1p1f1/Omon/tos/gn",
        wusd3_hist_id="miroc6_r1i1p1f1_historical_bc",
        wusd3_ssp_id="miroc6_r1i1p1f1_ssp370_bc",
    ),
    GCMSpec(
        name="MPI-ESM1-2-HR",
        short="mpi-esm1-2-hr",
        variant="r3i1p1f1",
        tos_hist_dir=CMIP6_ROOT / "CMIP/MPI-M/MPI-ESM1-2-HR/historical/r3i1p1f1/Omon/tos/gn",
        tos_ssp_dir=CMIP6_ROOT / "ScenarioMIP/DKRZ/MPI-ESM1-2-HR/ssp370/r3i1p1f1/Omon/tos/gn",
        wusd3_hist_id="mpi-esm1-2-hr_r3i1p1f1_historical_bc",
        wusd3_ssp_id="mpi-esm1-2-hr_r3i1p1f1_ssp370_bc",
    ),
    GCMSpec(
        name="TaiESM1",
        short="taiesm1",
        variant="r1i1p1f1",
        tos_hist_dir=CMIP6_ROOT / "CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1/Omon/tos/gn",
        tos_ssp_dir=CMIP6_ROOT / "ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1/Omon/tos/gn",
        wusd3_hist_id="taiesm1_r1i1p1f1_historical_bc",
        wusd3_ssp_id="taiesm1_r1i1p1f1_ssp370_bc",
    ),
]

SCENARIOS = ["historical", "ssp370"]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive or batch compute-node allocation.")


def ensure_dirs() -> None:
    for path in (ARTIFACT_DIR, PATTERNS_DIR, PLOTS_DIR, INDICES_DIR, LOYO_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Bilinear regridding of curvilinear source grids onto a regular target grid,
# reusing one Delaunay triangulation across every timestep and, where the
# native grid is static across experiments, across scenarios too.
# ---------------------------------------------------------------------------


@dataclass
class BilinearRegridder:
    tri: Delaunay
    source_flat_index: np.ndarray
    repeat_count: int
    target_lat: np.ndarray
    target_lon: np.ndarray
    target_lat2d: np.ndarray
    target_lon2d: np.ndarray


def _target_axis(vmin: float, vmax: float, resolution: float) -> np.ndarray:
    start = np.floor(vmin) + resolution / 2.0
    return np.arange(start, vmax, resolution)


def build_regridder(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    global_lon: bool,
    resolution: float = REGRID_RESOLUTION_DEG,
    buffer_deg: float = REGRID_BUFFER_DEG,
) -> BilinearRegridder:
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.mod(np.asarray(lon2d, dtype=float), 360.0)

    in_lat = (lat2d >= lat_min - buffer_deg) & (lat2d <= lat_max + buffer_deg)
    if global_lon:
        in_lon = np.ones_like(in_lat, dtype=bool)
    else:
        in_lon = (lon2d >= lon_min - buffer_deg) & (lon2d <= lon_max + buffer_deg)
    keep = in_lat & in_lon
    if not np.any(keep):
        raise ValueError("No source grid points fall inside the requested domain buffer.")

    source_flat_index = np.flatnonzero(keep.ravel())
    src_lat = lat2d.ravel()[source_flat_index]
    src_lon = lon2d.ravel()[source_flat_index]

    target_lat = _target_axis(lat_min, lat_max, resolution)
    if global_lon:
        target_lon = np.arange(resolution / 2.0, 360.0, resolution)
    else:
        target_lon = _target_axis(lon_min, lon_max, resolution)
    target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)

    if global_lon:
        src_lon_build = np.concatenate([src_lon - 360.0, src_lon, src_lon + 360.0])
        src_lat_build = np.concatenate([src_lat, src_lat, src_lat])
        repeat_count = 3
    else:
        src_lon_build = src_lon
        src_lat_build = src_lat
        repeat_count = 1

    points = np.column_stack([src_lon_build, src_lat_build])
    tri = Delaunay(points)
    return BilinearRegridder(
        tri=tri,
        source_flat_index=source_flat_index,
        repeat_count=repeat_count,
        target_lat=target_lat,
        target_lon=target_lon,
        target_lat2d=target_lat2d,
        target_lon2d=target_lon2d,
    )


def apply_regridder(regridder: BilinearRegridder, values2d: np.ndarray) -> np.ndarray:
    flat = np.asarray(values2d, dtype=float).reshape(-1)[regridder.source_flat_index]
    flat_build = np.concatenate([flat] * regridder.repeat_count) if regridder.repeat_count > 1 else flat
    interp = LinearNDInterpolator(regridder.tri, flat_build)
    return np.asarray(interp(regridder.target_lon2d, regridder.target_lat2d), dtype=float)


# ---------------------------------------------------------------------------
# SST loading: driving GCM's own tos, regridded to a regular 1x1 grid, months
# Nov/Dec/Jan/Feb only.
# ---------------------------------------------------------------------------


def find_version_dir(var_dir: Path) -> Path:
    versions = sorted(p for p in var_dir.glob("v*") if p.is_dir())
    if not versions:
        raise FileNotFoundError(f"No version directory found under {var_dir}")
    return versions[-1]


def find_2d_latlon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_name = next((n for n in ("latitude", "lat") if n in ds.variables), None)
    lon_name = next((n for n in ("longitude", "lon") if n in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise KeyError("Could not locate 2D latitude/longitude in tos dataset.")
    return lat_name, lon_name


def open_tos_dataset(tos_dir: Path) -> xr.Dataset:
    version_dir = find_version_dir(tos_dir)
    files = sorted(str(p) for p in version_dir.glob("tos_Omon_*.nc"))
    if not files:
        raise FileNotFoundError(f"No tos_Omon_*.nc files found under {version_dir}")
    return xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        decode_times=True,
        use_cftime=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={},
    )


def load_gcm_sst_ndjf(
    tos_dir: Path,
    water_year_start: int,
    water_year_end: int,
    regridder: Optional[BilinearRegridder],
    lead_in_tos_dir: Optional[Path] = None,
) -> Tuple[xr.DataArray, BilinearRegridder]:
    ds = open_tos_dataset(tos_dir)
    lat_name, lon_name = find_2d_latlon_names(ds)
    tos = ds["tos"]

    time_start = f"{water_year_start - 1}-11-01"
    time_end = f"{water_year_end}-03-01"
    tos = tos.sel(time=slice(time_start, time_end))
    month = tos["time"].dt.month
    tos = tos.sel(time=month.isin([11, 12, 1, 2]))

    lead_ds = None
    if lead_in_tos_dir is not None:
        lead_ds = open_tos_dataset(lead_in_tos_dir)
        lead_tos = lead_ds["tos"].sel(time=slice(time_start, f"{water_year_start - 1}-12-31"))
        lead_tos = lead_tos.sel(time=lead_tos["time"].dt.month.isin([11, 12]))
        if lead_tos.sizes.get("time", 0) > 0:
            tos = xr.concat([lead_tos, tos], dim="time", coords="minimal", compat="override", join="override")

    tos = tos.load()
    if lead_ds is not None:
        lead_ds.close()

    lat2d = np.asarray(tos[lat_name].values, dtype=float)
    lon2d = np.asarray(tos[lon_name].values, dtype=float)
    if regridder is None:
        regridder = build_regridder(
            lat2d,
            lon2d,
            AQM_SST_LAT_MIN,
            AQM_SST_LAT_MAX,
            0.0,
            360.0,
            global_lon=True,
        )

    years = tos["time"].dt.year.values.astype(int)
    months = tos["time"].dt.month.values.astype(int)
    values = tos.values
    out = np.empty((values.shape[0], regridder.target_lat.size, regridder.target_lon.size), dtype=np.float64)
    for t in range(values.shape[0]):
        out[t] = apply_regridder(regridder, values[t])

    time_index = pd.to_datetime({"year": years, "month": months, "day": 1})
    da = xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": time_index, "lat": regridder.target_lat, "lon": regridder.target_lon},
        name="tos",
    ).sortby("time")
    ds.close()
    return da, regridder


# ---------------------------------------------------------------------------
# Precipitation loading: WUS-D3 d02's own `prec`, regridded to a regular 1x1
# grid, months Dec/Jan/Feb/Mar only. WRF native coordinates are reconstructed
# from the Lambert conformal projection attributes stored in each file.
# ---------------------------------------------------------------------------


def reconstruct_d02_coordinates(path: Path, variable_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        attrs = dict(ds.attrs)
        da = ds[variable_name]
        lat_dim, lon_dim = da.dims[-2], da.dims[-1]
        ny = int(da.sizes[lat_dim])
        nx = int(da.sizes[lon_dim])
    proj = Proj(
        proj="lcc",
        lat_1=float(attrs["TRUELAT1"]),
        lat_2=float(attrs["TRUELAT2"]),
        lat_0=float(attrs["CEN_LAT"]),
        lon_0=float(attrs["STAND_LON"]),
        a=WUSD3_PROJECTION_RADIUS_METERS,
        b=WUSD3_PROJECTION_RADIUS_METERS,
    )
    x_center, y_center = proj(float(attrs["CEN_LON"]), float(attrs["CEN_LAT"]))
    dx = float(attrs["DX"])
    dy = float(attrs["DY"])
    x_values = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx + x_center
    y_values = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dy + y_center
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    longitude, latitude = proj(x_grid, y_grid, inverse=True)
    return latitude.astype(np.float64), longitude.astype(np.float64)


def d02_prec_path(wusd3_dataset_id: str, file_year: int) -> Path:
    dataset = Wusd3Dataset(dataset_id=wusd3_dataset_id, domain="d02", root_dir=WUSD3_ROOT)
    base_dir = data_wusd3.wusd3_dataset_dir(dataset)
    scenario_token = data_wusd3._scenario_token(wusd3_dataset_id)
    model_token = data_wusd3._model_token(wusd3_dataset_id, scenario_token)
    middle = WUSD3_FILE_MIDDLE[scenario_token]
    return base_dir / f"prec.daily.{model_token}.{middle}.d02.{file_year:04d}.nc"


def load_d02_prec_djfm(
    wusd3_dataset_id: str,
    file_year_start: int,
    file_year_end: int,
    regridder: Optional[BilinearRegridder],
) -> Tuple[xr.DataArray, BilinearRegridder]:
    times: List[pd.Timestamp] = []
    fields: List[np.ndarray] = []
    for file_year in range(file_year_start, file_year_end + 1):
        path = d02_prec_path(wusd3_dataset_id, file_year)
        if not path.exists():
            raise FileNotFoundError(f"d02 prec file not found: {path}")
        if regridder is None:
            lat2d, lon2d = reconstruct_d02_coordinates(path, "prec")
            regridder = build_regridder(
                lat2d,
                lon2d,
                AQM_PRECIP_LAT_MIN,
                AQM_PRECIP_LAT_MAX,
                AQM_PRECIP_LON_MIN,
                AQM_PRECIP_LON_MAX,
                global_lon=False,
            )
        with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
            prec = ds["prec"]
            day_year = prec["day"].dt.year.values.astype(int)
            day_month = prec["day"].dt.month.values.astype(int)
            values = prec.values
            for target_year, target_month in (
                (file_year, 12),
                (file_year + 1, 1),
                (file_year + 1, 2),
                (file_year + 1, 3),
            ):
                sel = (day_year == target_year) & (day_month == target_month)
                if not np.any(sel):
                    continue
                month_mean = np.nanmean(values[sel], axis=0)
                times.append(pd.Timestamp(year=target_year, month=target_month, day=1))
                fields.append(apply_regridder(regridder, month_mean))

    order = np.argsort(times)
    time_index = pd.to_datetime([times[i] for i in order])
    stacked = np.stack([fields[i] for i in order], axis=0)
    da = xr.DataArray(
        stacked,
        dims=("time", "lat", "lon"),
        coords={"time": time_index, "lat": regridder.target_lat, "lon": regridder.target_lon},
        name="prec",
    )
    return da, regridder


# ---------------------------------------------------------------------------
# MCA (generalized reproduce_aqm): same cxy/SVD procedure, parameterized on
# already time/domain-restricted seasonal SST and precip fields.
# ---------------------------------------------------------------------------


def run_mca(sst_season: xr.DataArray, precip_season: xr.DataArray) -> Dict[str, object]:
    common_years = np.intersect1d(
        sst_season["time"].dt.year.values.astype(int),
        precip_season["time"].dt.year.values.astype(int),
    )
    sst_season = sst_season.sel(time=sst_season["time"].dt.year.isin(common_years))
    precip_season = precip_season.sel(time=precip_season["time"].dt.year.isin(common_years))

    sst_dt = detrend_along_time(sst_season)
    precip_dt = detrend_along_time(precip_season)

    x_raw = sst_dt.values.reshape(sst_dt.sizes["time"], -1)
    y_raw = precip_dt.values.reshape(precip_dt.sizes["time"], -1)
    xmask = np.isfinite(x_raw).all(axis=0)
    ymask = np.isfinite(y_raw).all(axis=0)
    x = x_raw[:, xmask]
    y = y_raw[:, ymask]

    x_std = np.std(x, axis=0, ddof=1, keepdims=True)
    y_std = np.std(y, axis=0, ddof=1, keepdims=True)
    x_keep = np.isfinite(x_std[0]) & (x_std[0] > 0.0)
    y_keep = np.isfinite(y_std[0]) & (y_std[0] > 0.0)
    x = x[:, x_keep]
    y = y[:, y_keep]
    x_std = x_std[:, x_keep]
    y_std = y_std[:, y_keep]
    x = (x - np.mean(x, axis=0, keepdims=True)) / x_std
    y = (y - np.mean(y, axis=0, keepdims=True)) / y_std

    finite_x = np.isfinite(x).all(axis=0)
    finite_y = np.isfinite(y).all(axis=0)
    x = x[:, finite_x]
    y = y[:, finite_y]
    if x.shape[1] < 2 or y.shape[1] < 2:
        raise ValueError("Too few valid SST or precip grid cells after masking.")

    cxy = (x.T @ y) / float(x.shape[0] - 1)
    u, svals, vt = np.linalg.svd(cxy, full_matrices=False)
    scf = (svals ** 2) / np.sum(svals ** 2)

    s1 = x @ u[:, 0]
    s2 = x @ u[:, 1]
    p1 = y @ vt[0, :]
    p2 = y @ vt[1, :]

    sst_map1 = corr_with_series(sst_dt, s1)
    sst_map2 = corr_with_series(sst_dt, s2)
    precip_map1 = corr_with_series(precip_dt, p1)
    precip_map2 = corr_with_series(precip_dt, p2)

    index_df = pd.DataFrame(
        {
            "date": pd.to_datetime(sst_season["time"].values),
            "water_year": sst_season["time"].dt.year.values.astype(int),
            "S1": s1.astype(float),
            "S2": s2.astype(float),
            "P1": p1.astype(float),
            "P2": p2.astype(float),
        }
    )

    return {
        "sst_map_mode1": sst_map1,
        "sst_map_mode2": sst_map2,
        "precip_map_mode1": precip_map1,
        "precip_map_mode2": precip_map2,
        "scf_mode1": float(scf[0]),
        "scf_mode2": float(scf[1]),
        "index_df": index_df,
        "n_years": int(x.shape[0]),
        "sst_dt_full": sst_dt,
    }


def nino34_correlation(sst_dt_full: xr.DataArray, s1: np.ndarray) -> float:
    box = sst_dt_full.sel(lon=slice(NINO34_LON_MIN, NINO34_LON_MAX))
    box = box.where((box["lat"] >= NINO34_LAT_MIN) & (box["lat"] <= NINO34_LAT_MAX), drop=True)
    weights = np.cos(np.deg2rad(box["lat"]))
    idx = box.weighted(weights).mean(("lat", "lon"), skipna=True).values.astype(float)
    mask = np.isfinite(idx) & np.isfinite(s1)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(idx[mask], s1[mask])[0, 1])


# ---------------------------------------------------------------------------
# Comparison against the reference (observed) AQM pattern.
# ---------------------------------------------------------------------------


def crop_to_reference_box(field: xr.DataArray) -> xr.DataArray:
    cropped = field.sel(
        lat=slice(REFERENCE_BOX_LAT_MIN, REFERENCE_BOX_LAT_MAX),
        lon=slice(REFERENCE_BOX_LON_MIN, REFERENCE_BOX_LON_MAX - 1.0e-9),
    )
    return cropped


def compare_to_reference(gcm_sst_mode2: xr.DataArray, reference_aqm: xr.DataArray) -> Dict[str, float]:
    cropped = crop_to_reference_box(gcm_sst_mode2)
    reference_aligned = reference_aqm.interp(lat=cropped["lat"], lon=cropped["lon"], method="nearest")
    valid_mask = xr.DataArray(
        np.isfinite(cropped.values) & np.isfinite(reference_aligned.values),
        coords={"lat": cropped["lat"], "lon": cropped["lon"]},
        dims=("lat", "lon"),
    )
    return spatial_metrics(cropped, reference_aligned, valid_mask)


# ---------------------------------------------------------------------------
# Step 4: LOYO ridge using this GCM's own S2 index vs its own April-1 Sierra
# SWE, reusing the shared ridge/metric helpers.
# ---------------------------------------------------------------------------


def run_gcm_loyo(index_df: pd.DataFrame, wusd3_dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    swe_df, _ = build_wus_series(wusd3_dataset_id, "d02")
    swe_df = swe_df.rename(columns={"wus_swe_m": "observed_swe"})[["water_year", "observed_swe"]]

    predictor = index_df.loc[index_df["date"].dt.month == 3, ["water_year", "S2"]].rename(
        columns={"S2": "S2_NDJF"}
    )
    table = swe_df.merge(predictor, on="water_year", how="inner").dropna().sort_values("water_year").reset_index(drop=True)
    if len(table) < 10:
        raise ValueError(f"Too few overlapping water years ({len(table)}) for LOYO.")

    # April-1 SWE is a strictly non-negative absolute amount; convert to an
    # anomaly around this GCM's own simulated climatology so sign_accuracy is
    # a meaningful metric, matching the real baseline's SWE-anomaly target.
    table["observed_swe"] = table["observed_swe"] - table["observed_swe"].mean()

    years = table["water_year"].to_numpy(dtype=int)
    y = table["observed_swe"].to_numpy(dtype=float)
    x_all = table[["S2_NDJF"]].to_numpy(dtype=float)

    prediction_rows = []
    for held_idx, held_year in enumerate(years):
        train_mask = np.ones(len(years), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_all[train_mask, :]
        x_test = x_all[~train_mask, :][0]
        y_train = y[train_mask]
        y_test = float(y[held_idx])

        selected_alpha, _ = loyo_ref.inner_loyo_best_alpha(x_train, y_train)
        pred_raw, _, _ = loyo_ref.fit_outer_model(x_train, y_train, x_test, selected_alpha)
        prediction_rows.append(
            {
                "water_year": int(held_year),
                "observed_swe": y_test,
                "predicted_swe": float(pred_raw),
                "selected_ridge_alpha": float(selected_alpha),
            }
        )
    pred_df = pd.DataFrame(prediction_rows).sort_values("water_year").reset_index(drop=True)
    metric_bundle = loyo_ref.compute_metric_bundle(
        pred_df["observed_swe"].to_numpy(dtype=float),
        pred_df["predicted_swe"].to_numpy(dtype=float),
    )
    metrics_df = pd.DataFrame([{"n_years": len(table), **metric_bundle}])
    return pred_df, metrics_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_mode_panel(result: Dict[str, object], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    panels = [
        (axes[0, 0], result["sst_map_mode1"], f"SST mode 1 (SCF1={result['scf_mode1']:.3f})"),
        (axes[0, 1], result["sst_map_mode2"], f"SST mode 2 (SCF2={result['scf_mode2']:.3f})"),
        (axes[1, 0], result["precip_map_mode1"], "Precip mode 1"),
        (axes[1, 1], result["precip_map_mode2"], "Precip mode 2 (AQM analog)"),
    ]
    for ax, field, subtitle in panels:
        vmax = np.nanpercentile(np.abs(field.values), 98)
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = ax.pcolormesh(field["lon"], field["lat"], field, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(subtitle)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_dirs()

    reference_ds = xr.open_dataset(REFERENCE_HARMONIZED_NETCDF)
    reference_aqm = reference_ds["AQM"].load()

    comparison_rows: List[Dict[str, object]] = []
    loyo_metric_rows: List[Dict[str, object]] = []

    for gcm in GCM_SPECS:
        sst_regridder: Optional[BilinearRegridder] = None
        precip_regridder: Optional[BilinearRegridder] = None

        for scenario in SCENARIOS:
            print(f"=== {gcm.name} / {scenario} ===", flush=True)
            if scenario == "historical":
                tos_dir = gcm.tos_hist_dir
                wusd3_id = gcm.wusd3_hist_id
                file_year_start, file_year_end = 1980, 2013
            else:
                tos_dir = gcm.tos_ssp_dir
                wusd3_id = gcm.wusd3_ssp_id
                file_year_start, file_year_end = 2014, 2099
            water_year_start = file_year_start + 1
            water_year_end = file_year_end + 1

            print("Loading precip...", flush=True)
            precip_da, precip_regridder = load_d02_prec_djfm(
                wusd3_id, file_year_start, file_year_end, precip_regridder
            )
            print("Loading SST...", flush=True)
            lead_in_tos_dir = gcm.tos_hist_dir if scenario == "ssp370" else None
            sst_da, sst_regridder = load_gcm_sst_ndjf(
                tos_dir, water_year_start, water_year_end, sst_regridder, lead_in_tos_dir=lead_in_tos_dir
            )

            sst_season = seasonal_average_label_by_year(sst_da, SEASON_MONTHS_SST, 3)
            precip_season = seasonal_average_label_by_year(precip_da, SEASON_MONTHS_PRECIP, 3)

            print("Running MCA...", flush=True)
            result = run_mca(sst_season, precip_season)
            enso_check = nino34_correlation(
                result["sst_dt_full"], result["index_df"]["S1"].to_numpy(dtype=float)
            )

            tag = f"{gcm.short}_{scenario}"
            patterns_ds = xr.Dataset(
                {
                    "sst_mode1": result["sst_map_mode1"],
                    "sst_mode2": result["sst_map_mode2"],
                    "precip_mode1": result["precip_map_mode1"],
                    "precip_mode2": result["precip_map_mode2"],
                }
            )
            patterns_ds.to_netcdf(PATTERNS_DIR / f"{tag}_mca_patterns.nc")
            result["index_df"].to_csv(INDICES_DIR / f"{tag}_s2_index.csv", index=False)
            plot_mode_panel(result, f"{gcm.name} {scenario} MCA modes", PLOTS_DIR / f"{tag}_mode_maps.png")

            comparison = compare_to_reference(result["sst_map_mode2"], reference_aqm)
            s2p2_r = float(
                np.corrcoef(result["index_df"]["S2"].to_numpy(dtype=float), result["index_df"]["P2"].to_numpy(dtype=float))[0, 1]
            )
            comparison_rows.append(
                {
                    "gcm": gcm.name,
                    "scenario": scenario,
                    "n_years": result["n_years"],
                    "spatial_corr_sst_mode2_vs_reference": comparison["signed_spatial_r"],
                    "spatial_corr_sst_mode2_vs_reference_abs": comparison["absolute_spatial_r"],
                    "spatial_corr_precip_mode2_vs_paper": None,
                    "scf_mode1": result["scf_mode1"],
                    "scf_mode2": result["scf_mode2"],
                    "s2_p2_correlation": s2p2_r,
                    "mode1_nino34_correlation_sanity_check": enso_check,
                }
            )

            print("Running Step 4 LOYO...", flush=True)
            try:
                pred_df, metrics_df = run_gcm_loyo(result["index_df"], wusd3_id)
                pred_df.to_csv(LOYO_DIR / f"{tag}_loyo_predictions.csv", index=False)
                metrics_row = metrics_df.iloc[0].to_dict()
                metrics_row.update({"gcm": gcm.name, "scenario": scenario})
                loyo_metric_rows.append(metrics_row)
            except Exception as exc:  # noqa: BLE001
                loyo_metric_rows.append({"gcm": gcm.name, "scenario": scenario, "error": str(exc)})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(ARTIFACT_DIR / "step3_comparison_table.csv", index=False)

    loyo_df = pd.DataFrame(loyo_metric_rows)
    loyo_df.to_csv(ARTIFACT_DIR / "step4_loyo_metrics_table.csv", index=False)

    reference_row = {"source": "paper_observed", **PAPER_OBSERVED}
    gfdl_row = {"source": "paper_gfdl", **PAPER_GFDL}
    pd.DataFrame([reference_row, gfdl_row]).to_csv(ARTIFACT_DIR / "step3_paper_reference_values.csv", index=False)

    print("Done. Artifact directory:", ARTIFACT_DIR, flush=True)


if __name__ == "__main__":
    main()
