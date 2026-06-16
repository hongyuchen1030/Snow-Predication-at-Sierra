#!/usr/bin/env python3
"""
Run the ERA5 SST -> April 1 Sierra SWE LOD analysis with the same setup as the
current Pacific-domain COBE2 diagnostic.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ERA5_SST_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5/e5.oper.an.sfc")
SWE_TARGET_FILE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc")
ARTIFACT_ROOT = Path(
    os.environ.get(
        "ERA5_SIERRA_SWE_LOD_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup",
    )
).expanduser()
PREDICTOR_ROOT = ARTIFACT_ROOT / "predictors"
PREDICTOR_FILE = PREDICTOR_ROOT / "era5_pacific_sst_monthly_anomaly_wy1985_2021_sep1984_mar2021.nc"
PREDICTOR_SUMMARY_FILE = PREDICTOR_ROOT / "era5_pacific_sst_monthly_anomaly_wy1985_2021_sep1984_mar2021_summary.json"
OUTPUT_ROOT = ARTIFACT_ROOT / "lod_analysis"
NETCDF_ENGINE = "netcdf4"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
PACIFIC_LAT_MIN = -10.0
PACIFIC_LAT_MAX = 60.0
PACIFIC_LON_MIN = 120.0
PACIFIC_LON_MAX = 280.0
LAG_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]
K_MAX = 6
DELTA_R2_MIN = 0.02
CORR_MIN = 0.2
ERA5_VARIABLE = "SSTK"


@dataclass(frozen=True)
class ReuseDecision:
    reused_existing_product: bool
    processed_predictor_path: str
    processed_predictor_summary_path: str
    search_roots: list[str]
    search_matches: list[str]
    search_checked_candidates: list[str]
    search_reason: str


def processed_predictor_path(reuse: ReuseDecision) -> Path:
    return Path(reuse.processed_predictor_path)


def processed_predictor_summary_path(reuse: ReuseDecision) -> Path:
    return Path(reuse.processed_predictor_summary_path)


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def month_targets() -> tuple[list[np.datetime64], list[str], list[np.int32], list[str]]:
    times: list[np.datetime64] = []
    lag_names: list[str] = []
    wy_index: list[np.int32] = []
    month_keys: list[str] = []
    for water_year in WATER_YEARS:
        for lag_name, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
            lag_names.append(lag_name)
            wy_index.append(water_year)
            month_keys.append(f"{year:04d}{month:02d}")
    return times, lag_names, wy_index, month_keys


def load_target() -> xr.Dataset:
    ds = xr.open_dataset(SWE_TARGET_FILE, engine=NETCDF_ENGINE).load()
    return ds.sel(water_year=WATER_YEARS)


def build_raw_month_path(month_key: str) -> Path:
    month_dir = ERA5_SST_ROOT / month_key
    matches = sorted(month_dir.glob(f"e5.oper.an.sfc.128_034_sstk.ll025sc.{month_key}0100_*.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one ERA5 SST file for {month_key}, found {len(matches)} under {month_dir}")
    return matches[0]


def search_existing_processed_predictor() -> ReuseDecision:
    search_roots = [
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts",
        str(PROJECT_ROOT / "artifacts"),
    ]
    patterns = ("*era5*sst*.nc", "*sstk*.nc", "*era5*sst*.json")
    all_matches: list[str] = []
    nc_candidates: list[Path] = []

    for root_text in search_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if str(path) not in all_matches:
                    all_matches.append(str(path))
                if path.suffix == ".nc":
                    nc_candidates.append(path)

    checked: list[str] = []
    for path in nc_candidates:
        checked.append(str(path))
        if path.resolve() == PREDICTOR_FILE.resolve() and PREDICTOR_FILE.exists() and PREDICTOR_SUMMARY_FILE.exists():
            try:
                with xr.open_dataset(path, engine=NETCDF_ENGINE) as ds:
                    if (
                        ds.attrs.get("source_dataset") == "ERA5"
                        and ds.attrs.get("variable_name") == ERA5_VARIABLE
                        and float(ds.attrs.get("domain_lat_min")) == PACIFIC_LAT_MIN
                        and float(ds.attrs.get("domain_lat_max")) == PACIFIC_LAT_MAX
                        and float(ds.attrs.get("domain_lon_min_360")) == PACIFIC_LON_MIN
                        and float(ds.attrs.get("domain_lon_max_360")) == PACIFIC_LON_MAX
                        and int(ds.attrs.get("water_year_start")) == WATER_YEAR_START
                        and int(ds.attrs.get("water_year_end")) == WATER_YEAR_END
                    ):
                        summary_candidate = path.with_name(path.stem + "_summary.json")
                        return ReuseDecision(
                            reused_existing_product=True,
                            processed_predictor_path=str(path),
                            processed_predictor_summary_path=str(summary_candidate if summary_candidate.exists() else ""),
                            search_roots=search_roots,
                            search_matches=all_matches,
                            search_checked_candidates=checked,
                            search_reason="Found an existing processed ERA5 Pacific monthly-anomaly predictor with matching domain and water-year metadata.",
                        )
            except Exception:
                continue

    return ReuseDecision(
        reused_existing_product=False,
        processed_predictor_path=str(PREDICTOR_FILE),
        processed_predictor_summary_path=str(PREDICTOR_SUMMARY_FILE),
        search_roots=search_roots,
        search_matches=all_matches,
        search_checked_candidates=checked,
        search_reason="No existing processed ERA5 Pacific monthly-anomaly predictor with the required metadata was found under repo or pscratch artifact trees.",
    )


def compute_monthly_mean_for_month(month_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, str]:
    path = build_raw_month_path(month_key)
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        subset = ds[ERA5_VARIABLE].sel(
            latitude=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            longitude=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        )
        monthly_mean = subset.mean(dim="time", skipna=True, keep_attrs=True).load()
        latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float32)
        longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float32)
        values = np.asarray(monthly_mean.values, dtype=np.float32)
        units = str(monthly_mean.attrs.get("units", "K"))
    return values, latitude, longitude, path, units


def create_processed_predictor(reuse: ReuseDecision) -> None:
    selected_times, lag_names, wy_index, month_keys = month_targets()
    PREDICTOR_ROOT.mkdir(parents=True, exist_ok=True)

    monthly_fields: list[np.ndarray] = []
    source_files: list[str] = []
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    units = "K"
    start = perf_counter()

    for position, month_key in enumerate(month_keys, start=1):
        field, lat, lon, raw_path, units = compute_monthly_mean_for_month(month_key)
        if latitude is None:
            latitude = lat
            longitude = lon
        else:
            if not np.array_equal(latitude, lat) or not np.array_equal(longitude, lon):
                raise ValueError(f"ERA5 grid mismatch at {raw_path}")
        monthly_fields.append(field)
        source_files.append(str(raw_path))
        print(f"[{position:03d}/{len(month_keys):03d}] monthly mean ready for {month_key} from {raw_path.name}", flush=True)

    if latitude is None or longitude is None:
        raise RuntimeError("No ERA5 monthly fields were built.")

    monthly_values = np.stack(monthly_fields, axis=0).astype(np.float32)
    aligned_monthly = monthly_values.reshape(len(WATER_YEARS), len(LAG_SPECS), latitude.size, longitude.size)
    monthly_clim = np.nanmean(aligned_monthly, axis=0, dtype=np.float64).astype(np.float32)
    monthly_anom = (aligned_monthly - monthly_clim[None, :, :, :]).astype(np.float32)
    std_by_lag = np.nanstd(monthly_anom, axis=0, ddof=1, dtype=np.float64)
    shared_valid_ocean = np.all(np.isfinite(aligned_monthly), axis=(0, 1)) & np.all(np.isfinite(std_by_lag) & (std_by_lag > 0.0), axis=0)

    lag_coord = np.asarray(lag_names[: len(LAG_SPECS)], dtype="U3")
    time_coord = np.asarray(selected_times, dtype="datetime64[ns]")
    water_year_coord = WATER_YEARS

    ds = xr.Dataset(
        data_vars={
            "sst_monthly_mean": xr.DataArray(
                monthly_values,
                dims=("time", "latitude", "longitude"),
                coords={"time": time_coord, "latitude": latitude, "longitude": longitude},
                attrs={"units": units, "description": "ERA5 monthly mean SST aggregated from hourly SSTK over the Pacific predictor domain."},
            ),
            "sst_monthly_anomaly": xr.DataArray(
                monthly_anom,
                dims=("water_year", "lag_month", "latitude", "longitude"),
                coords={"water_year": water_year_coord, "lag_month": lag_coord, "latitude": latitude, "longitude": longitude},
                attrs={"units": units, "description": "ERA5 monthly SST anomaly relative to the lag-specific climatology across WY1985-WY2021."},
            ),
            "sst_monthly_climatology": xr.DataArray(
                monthly_clim,
                dims=("lag_month", "latitude", "longitude"),
                coords={"lag_month": lag_coord, "latitude": latitude, "longitude": longitude},
                attrs={"units": units, "description": "ERA5 lag-specific monthly SST climatology for Sep-Mar aligned to water years 1985-2021."},
            ),
            "valid_ocean_mask": xr.DataArray(
                shared_valid_ocean.astype(np.int8),
                dims=("latitude", "longitude"),
                coords={"latitude": latitude, "longitude": longitude},
                attrs={"description": "Shared ocean-only valid mask: finite monthly means in every aligned Sep-Mar sample and positive anomaly std in every lag month.", "flag_values": [0, 1], "flag_meanings": "invalid valid"},
            ),
        },
        attrs={
            "description": "Processed ERA5 Pacific-domain monthly SST predictor product for Sierra SWE LOD.",
            "source_dataset": "ERA5",
            "variable_name": ERA5_VARIABLE,
            "domain_lat_min": PACIFIC_LAT_MIN,
            "domain_lat_max": PACIFIC_LAT_MAX,
            "domain_lon_min_360": PACIFIC_LON_MIN,
            "domain_lon_max_360": PACIFIC_LON_MAX,
            "water_year_start": WATER_YEAR_START,
            "water_year_end": WATER_YEAR_END,
            "lag_months": ",".join(lag_coord.tolist()),
            "month_alignment": "Sep(y-1), Oct(y-1), Nov(y-1), Dec(y-1), Jan(y), Feb(y), Mar(y)",
            "units": units,
            "search_reused_existing_product": int(reuse.reused_existing_product),
        },
    )
    ds.to_netcdf(processed_predictor_path(reuse), engine=NETCDF_ENGINE)

    summary = {
        **asdict(reuse),
        "source_dataset": "ERA5",
        "variable_name": ERA5_VARIABLE,
        "raw_source_root": str(ERA5_SST_ROOT),
        "raw_files_used": source_files,
        "time_coverage_start": str(time_coord[0]),
        "time_coverage_end": str(time_coord[-1]),
        "water_years": water_year_coord.tolist(),
        "lag_months": lag_coord.tolist(),
        "domain": {
            "lat_min": PACIFIC_LAT_MIN,
            "lat_max": PACIFIC_LAT_MAX,
            "lon_min_360": PACIFIC_LON_MIN,
            "lon_max_360": PACIFIC_LON_MAX,
        },
        "monthly_mean_shape": [int(v) for v in monthly_values.shape],
        "monthly_anomaly_shape": [int(v) for v in monthly_anom.shape],
        "shared_valid_ocean_cell_count": int(np.count_nonzero(shared_valid_ocean)),
        "product_created_new": True,
        "runtime_seconds": perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
        "output_path": str(PREDICTOR_FILE),
    }
    processed_predictor_summary_path(reuse).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def load_processed_predictor(reuse: ReuseDecision) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(processed_predictor_path(reuse), engine=NETCDF_ENGINE) as ds:
        monthly_anom = np.asarray(ds["sst_monthly_anomaly"].values, dtype=np.float32)
        shared_valid_ocean = np.asarray(ds["valid_ocean_mask"].values, dtype=bool)
        latitude = np.asarray(ds["latitude"].values, dtype=np.float32)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float32)
    return monthly_anom, shared_valid_ocean, latitude, longitude


def build_standardized_predictor_matrix(
    monthly_anom: np.ndarray,
    shared_valid_ocean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(monthly_anom, axis=0, dtype=np.float64)
    std = np.nanstd(monthly_anom, axis=0, ddof=1, dtype=np.float64)
    valid = shared_valid_ocean[None, :, :] & np.isfinite(std) & (std > 0.0)
    standardized = np.full(monthly_anom.shape, np.nan, dtype=np.float64)
    np.divide(
        monthly_anom - mean[None, :, :, :],
        std[None, :, :, :],
        out=standardized,
        where=valid[None, :, :, :],
    )

    ocean_lat_idx, ocean_lon_idx = np.where(shared_valid_ocean)
    n_ocean = int(ocean_lat_idx.size)
    standardized_by_lag = [standardized[:, lag_index, ocean_lat_idx, ocean_lon_idx] for lag_index in range(len(LAG_SPECS))]
    X = np.concatenate(standardized_by_lag, axis=1).astype(np.float32)
    lag_idx = np.repeat(np.arange(len(LAG_SPECS), dtype=np.int32), n_ocean)
    spatial_idx = np.tile(np.stack([ocean_lat_idx, ocean_lon_idx], axis=1).astype(np.int32), (len(LAG_SPECS), 1))
    return X, lag_idx, spatial_idx, np.full(len(LAG_SPECS), n_ocean, dtype=np.int32)


def centered_correlation_scores(X: np.ndarray, residual: np.ndarray) -> np.ndarray:
    residual_centered = residual - residual.mean()
    residual_std = residual_centered.std(ddof=1)
    if not np.isfinite(residual_std) or residual_std == 0.0:
        return np.zeros(X.shape[1], dtype=np.float64)
    return (X.astype(np.float64).T @ residual_centered.astype(np.float64)) / ((X.shape[0] - 1) * residual_std)


def run_lod(X: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    residual = y.astype(np.float64).copy()
    retained_modes: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    residual_history = [residual.copy()]

    total_sum = float(np.sum(y.astype(np.float64) ** 2))
    previous_r2 = 0.0
    for mode_number in range(1, K_MAX + 1):
        corr_scores = centered_correlation_scores(X, residual)
        q_index = int(np.argmax(np.abs(corr_scores)))
        corr_value = float(corr_scores[q_index])
        if abs(corr_value) < CORR_MIN:
            rows.append(
                {
                    "mode_id": mode_number,
                    "mode_number": mode_number,
                    "selected": False,
                    "stop_reason": f"|corr| < {CORR_MIN}",
                    "candidate_index": q_index,
                    "selected_residual_correlation": corr_value,
                    "corr_with_residual": corr_value,
                }
            )
            break

        u = X[:, q_index].astype(np.float64)
        m_hat = u.copy()
        for previous in retained_modes:
            coeff = float(np.sum(u * previous) / np.sum(previous * previous))
            m_hat = m_hat - coeff * previous
        if mode_number == 1:
            m_hat = u.copy()

        m_hat_mean = float(m_hat.mean())
        m_hat_std = float(m_hat.std(ddof=1))
        if not np.isfinite(m_hat_std) or m_hat_std == 0.0:
            rows.append(
                {
                    "mode_id": mode_number,
                    "mode_number": mode_number,
                    "selected": False,
                    "stop_reason": "orthogonalized mode std == 0",
                    "candidate_index": q_index,
                    "selected_residual_correlation": corr_value,
                    "corr_with_residual": corr_value,
                }
            )
            break

        mode = (m_hat - m_hat_mean) / m_hat_std
        beta = float(np.sum(mode * residual) / np.sum(mode * mode))
        fitted = beta * mode
        new_residual = residual - fitted
        cumulative_r2 = 1.0 - (float(np.sum(new_residual ** 2)) / total_sum)
        delta_r2 = float(cumulative_r2 - previous_r2)
        if delta_r2 < DELTA_R2_MIN:
            rows.append(
                {
                    "mode_id": mode_number,
                    "mode_number": mode_number,
                    "selected": False,
                    "stop_reason": f"delta_r2 < {DELTA_R2_MIN}",
                    "candidate_index": q_index,
                    "selected_residual_correlation": corr_value,
                    "corr_with_residual": corr_value,
                    "beta": beta,
                    "delta_R2": delta_r2,
                    "delta_r2": delta_r2,
                    "cumulative_R2": cumulative_r2,
                    "cumulative_r2": cumulative_r2,
                }
            )
            break

        retained_modes.append(mode.copy())
        residual = new_residual
        residual_history.append(residual.copy())
        previous_r2 = cumulative_r2
        rows.append(
            {
                "mode_id": mode_number,
                "mode_number": mode_number,
                "selected": True,
                "candidate_index": q_index,
                "selected_residual_correlation": corr_value,
                "corr_with_residual": corr_value,
                "beta": beta,
                "delta_R2": delta_r2,
                "delta_r2": delta_r2,
                "cumulative_R2": cumulative_r2,
                "cumulative_r2": cumulative_r2,
                "residual_variance_fraction": float(np.sum(new_residual ** 2) / total_sum),
                "mode_mean_before_standardization": m_hat_mean,
                "mode_std_before_standardization": m_hat_std,
            }
        )
        print(
            f"selected mode {mode_number}: candidate={q_index} corr={corr_value:.4f} beta={beta:.4f} "
            f"delta_r2={delta_r2:.4f} cumulative_r2={cumulative_r2:.4f}",
            flush=True,
        )

    mode_matrix = np.stack(retained_modes, axis=0).astype(np.float32) if retained_modes else np.empty((0, X.shape[0]), dtype=np.float32)
    residual_matrix = np.stack(residual_history, axis=0).astype(np.float32)
    return rows, mode_matrix, residual_matrix


def main() -> None:
    ensure_runtime_on_compute_node()
    PREDICTOR_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    reuse = search_existing_processed_predictor()
    print(json.dumps(asdict(reuse), indent=2), flush=True)
    if not reuse.reused_existing_product:
        create_processed_predictor(reuse)

    target_ds = load_target()
    y = np.asarray(target_ds["sierra_swe_apr1_standardized"].values, dtype=np.float32)
    monthly_anom, shared_valid_ocean, latitude, longitude = load_processed_predictor(reuse)
    print(f"loaded ERA5 monthly anomaly cube shape={monthly_anom.shape}", flush=True)
    print(f"shared valid ocean cells={int(np.count_nonzero(shared_valid_ocean))}", flush=True)

    X, lag_idx, spatial_idx, valid_counts_by_lag = build_standardized_predictor_matrix(monthly_anom, shared_valid_ocean)
    print(f"built predictor matrix shape={X.shape}", flush=True)

    rows, mode_matrix, residual_matrix = run_lod(X, y)
    lag_names = np.asarray([spec[0] for spec in LAG_SPECS], dtype="U3")

    output_rows: list[dict[str, object]] = []
    for row in rows:
        row_out = dict(row)
        if "candidate_index" in row_out:
            q = int(row_out["candidate_index"])
            if 0 <= q < lag_idx.size:
                row_out["lag_month"] = str(lag_names[int(lag_idx[q])])
                row_out["latitude"] = float(latitude[int(spatial_idx[q, 0])])
                row_out["longitude_0_360"] = float(longitude[int(spatial_idx[q, 1])])
                row_out["longitude"] = row_out["longitude_0_360"]
                row_out["ocean_candidate_q"] = q
        output_rows.append(row_out)

    csv_path = OUTPUT_ROOT / "era5_sierra_swe_lod_modes.csv"
    fieldnames = [
        "mode_id",
        "mode_number",
        "selected",
        "stop_reason",
        "ocean_candidate_q",
        "lag_month",
        "latitude",
        "longitude_0_360",
        "selected_residual_correlation",
        "corr_with_residual",
        "beta",
        "delta_R2",
        "delta_r2",
        "cumulative_R2",
        "cumulative_r2",
        "residual_variance_fraction",
        "mode_mean_before_standardization",
        "mode_std_before_standardization",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    mode_count = mode_matrix.shape[0]
    diagnostics = xr.Dataset(
        data_vars={
            "target_standardized": xr.DataArray(y.astype(np.float32), dims=("water_year",), coords={"water_year": WATER_YEARS}, attrs={"units": "1"}),
            "target_anomaly_m": xr.DataArray(np.asarray(target_ds["sierra_swe_apr1_anom_m"].values, dtype=np.float32), dims=("water_year",), coords={"water_year": WATER_YEARS}, attrs={"units": "m"}),
            "lod_mode_series": xr.DataArray(mode_matrix, dims=("mode", "water_year"), coords={"mode": np.arange(1, mode_count + 1, dtype=np.int32), "water_year": WATER_YEARS}, attrs={"units": "1"}),
            "residual_series": xr.DataArray(residual_matrix, dims=("stage", "water_year"), coords={"stage": np.arange(residual_matrix.shape[0], dtype=np.int32), "water_year": WATER_YEARS}, attrs={"units": "1"}),
            "cumulative_reconstruction": xr.DataArray((y.astype(np.float64) - residual_matrix[-1].astype(np.float64)).astype(np.float32), dims=("water_year",), coords={"water_year": WATER_YEARS}, attrs={"units": "1"}),
            "shared_valid_ocean_mask": xr.DataArray(shared_valid_ocean.astype(np.int8), dims=("latitude", "longitude"), coords={"latitude": latitude, "longitude": longitude}, attrs={"flag_values": [0, 1], "flag_meanings": "invalid valid"}),
        },
        attrs={
            "description": "LOD diagnostics for ERA5 SST predictors against April 1 Sierra SWE standardized anomaly.",
            "sst_source": str(processed_predictor_path(reuse)),
            "raw_sst_source_root": str(ERA5_SST_ROOT),
            "swe_target_source": str(SWE_TARGET_FILE),
            "domain_lat_min": PACIFIC_LAT_MIN,
            "domain_lat_max": PACIFIC_LAT_MAX,
            "domain_lon_min_360": PACIFIC_LON_MIN,
            "domain_lon_max_360": PACIFIC_LON_MAX,
            "water_year_start": WATER_YEAR_START,
            "water_year_end": WATER_YEAR_END,
            "lag_months": ",".join(lag_names.tolist()),
            "k_max": K_MAX,
            "delta_r2_min": DELTA_R2_MIN,
            "corr_min": CORR_MIN,
        },
    )
    diagnostics_path = OUTPUT_ROOT / "era5_sierra_swe_lod_diagnostics.nc"
    diagnostics.to_netcdf(diagnostics_path, engine=NETCDF_ENGINE)

    summary = {
        **asdict(reuse),
        "sst_source": str(processed_predictor_path(reuse)),
        "sst_source_summary": str(processed_predictor_summary_path(reuse)),
        "raw_sst_source_root": str(ERA5_SST_ROOT),
        "swe_target_source": str(SWE_TARGET_FILE),
        "water_years": WATER_YEARS.tolist(),
        "n_samples": int(WATER_YEARS.size),
        "lag_months": lag_names.tolist(),
        "domain": {
            "lat_min": PACIFIC_LAT_MIN,
            "lat_max": PACIFIC_LAT_MAX,
            "lon_min_360": PACIFIC_LON_MIN,
            "lon_max_360": PACIFIC_LON_MAX,
        },
        "predictor_matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "valid_candidate_count": int(X.shape[1]),
        "valid_ocean_cell_count": int(np.count_nonzero(shared_valid_ocean)),
        "valid_counts_by_lag": {lag: int(count) for lag, count in zip(lag_names.tolist(), valid_counts_by_lag.tolist())},
        "lod_rows": output_rows,
        "retained_mode_count": int(mode_count),
        "runtime_seconds": perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
        "modes_csv": str(csv_path),
        "diagnostics_nc": str(diagnostics_path),
    }
    summary_path = OUTPUT_ROOT / "era5_sierra_swe_lod_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "modes_csv": str(csv_path), "diagnostics_nc": str(diagnostics_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
