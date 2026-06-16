#!/usr/bin/env python3
"""
Run the COBE2 SST -> April 1 Sierra SWE LOD analysis.
"""

import csv
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
COBE2_MONTHLY_ANOM_REF = PROJECT_ROOT / "artifacts" / "sst_pca" / "cobe2_global_monthly_climatology_anomaly" / "cobe2_global_monthly_clim_sst_eofs.nc"
SWE_TARGET_FILE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc")
OUTPUT_ROOT = Path(
    os.environ.get(
        "COBE2_SIERRA_SWE_LOD_ANALYSIS_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/lod_analysis",
    )
).expanduser()
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


def month_targets() -> tuple[list[np.datetime64], list[str], list[np.int32]]:
    times: list[np.datetime64] = []
    lag_names: list[str] = []
    wy_index: list[np.int32] = []
    for water_year in WATER_YEARS:
        for lag_name, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
            lag_names.append(lag_name)
            wy_index.append(water_year)
    return times, lag_names, wy_index


def load_target() -> xr.Dataset:
    ds = xr.open_dataset(SWE_TARGET_FILE, engine=NETCDF_ENGINE).load()
    ds = ds.sel(water_year=WATER_YEARS)
    return ds


def load_aligned_sst() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_times, _, _ = month_targets()
    with xr.open_dataset(COBE2_SST_FILE, engine=NETCDF_ENGINE) as ds:
        sst = ds["sst"].sel(
            time=selected_times,
            lat=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            lon=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        ).load()
        sst_values = np.asarray(sst.values, dtype=np.float32).reshape(len(WATER_YEARS), len(LAG_SPECS), sst.sizes["lat"], sst.sizes["lon"])
        lat = np.asarray(sst["lat"].values, dtype=np.float32)
        lon = np.asarray(sst["lon"].values, dtype=np.float32)
        return sst_values, lat, lon, np.asarray(selected_times)


def build_standardized_predictor_matrix(
    sst_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    monthly_clim = np.nanmean(sst_values, axis=0, dtype=np.float64)
    anomalies = sst_values - monthly_clim[None, :, :, :]
    mean = np.nanmean(anomalies, axis=0, dtype=np.float64)
    std = np.nanstd(anomalies, axis=0, ddof=1, dtype=np.float64)
    valid = np.all(np.isfinite(anomalies), axis=0) & np.isfinite(std) & (std > 0.0)
    standardized = np.full(anomalies.shape, np.nan, dtype=np.float64)
    np.divide(
        anomalies - mean[None, :, :, :],
        std[None, :, :, :],
        out=standardized,
        where=valid[None, :, :, :],
    )
    lag_idx, lat_idx, lon_idx = np.where(valid)
    X = standardized[:, lag_idx, lat_idx, lon_idx].astype(np.float32)
    valid_counts_by_lag = valid.sum(axis=(1, 2)).astype(np.int32)
    return X, valid, lag_idx.astype(np.int32), np.stack([lat_idx, lon_idx], axis=1).astype(np.int32), valid_counts_by_lag


def centered_correlation_scores(X: np.ndarray, residual: np.ndarray) -> np.ndarray:
    residual_centered = residual - residual.mean()
    residual_std = residual_centered.std(ddof=1)
    if not np.isfinite(residual_std) or residual_std == 0.0:
        return np.zeros(X.shape[1], dtype=np.float64)
    # X columns are already standardized with mean 0 and std 1 over the 37 samples.
    return (X.astype(np.float64).T @ residual_centered.astype(np.float64)) / ((X.shape[0] - 1) * residual_std)


def run_lod(X: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    residual = y.astype(np.float64).copy()
    retained_modes: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    residual_history = [residual.copy()]
    reconstruction = np.zeros_like(residual)

    total_sum = float(np.sum(y.astype(np.float64) ** 2))
    for mode_number in range(1, K_MAX + 1):
        corr_scores = centered_correlation_scores(X, residual)
        q_index = int(np.argmax(np.abs(corr_scores)))
        corr_value = float(corr_scores[q_index])
        if abs(corr_value) < CORR_MIN:
            rows.append(
                {
                    "mode_number": mode_number,
                    "selected": False,
                    "stop_reason": f"|corr| < {CORR_MIN}",
                    "candidate_index": q_index,
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
                    "mode_number": mode_number,
                    "selected": False,
                    "stop_reason": "orthogonalized mode std == 0",
                    "candidate_index": q_index,
                    "corr_with_residual": corr_value,
                }
            )
            break

        mode = (m_hat - m_hat_mean) / m_hat_std
        beta = float(np.sum(mode * residual) / np.sum(mode * mode))
        fitted = beta * mode
        new_residual = residual - fitted
        cumulative_r2 = 1.0 - (float(np.sum(new_residual ** 2)) / total_sum)
        previous_r2 = rows[-1]["cumulative_r2"] if rows and rows[-1].get("selected") else 0.0
        delta_r2 = float(cumulative_r2 - previous_r2)
        if delta_r2 < DELTA_R2_MIN:
            rows.append(
                {
                    "mode_number": mode_number,
                    "selected": False,
                    "stop_reason": f"delta_r2 < {DELTA_R2_MIN}",
                    "candidate_index": q_index,
                    "corr_with_residual": corr_value,
                    "beta": beta,
                    "delta_r2": delta_r2,
                    "cumulative_r2": cumulative_r2,
                }
            )
            break

        retained_modes.append(mode.copy())
        residual = new_residual
        reconstruction = reconstruction + fitted
        residual_history.append(residual.copy())
        rows.append(
            {
                "mode_number": mode_number,
                "selected": True,
                "candidate_index": q_index,
                "corr_with_residual": corr_value,
                "beta": beta,
                "delta_r2": delta_r2,
                "cumulative_r2": cumulative_r2,
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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    target_ds = load_target()
    y = np.asarray(target_ds["sierra_swe_apr1_standardized"].values, dtype=np.float32)
    print(f"loaded target standardized SWE shape={y.shape}", flush=True)

    sst_values, lat, lon, _ = load_aligned_sst()
    print(f"loaded aligned SST shape={sst_values.shape}", flush=True)

    X, valid_mask, lag_idx, spatial_idx, valid_counts_by_lag = build_standardized_predictor_matrix(sst_values)
    print(f"built predictor matrix shape={X.shape}", flush=True)
    print(f"valid ocean cells by lag month={valid_counts_by_lag.tolist()}", flush=True)

    rows, mode_matrix, residual_matrix = run_lod(X, y)
    selected_rows = [row for row in rows if row.get("selected")]

    lag_names = np.asarray([spec[0] for spec in LAG_SPECS], dtype="U3")
    output_rows: list[dict[str, object]] = []
    for row in rows:
        if "candidate_index" in row:
            q = int(row["candidate_index"])
            if 0 <= q < lag_idx.size:
                lag_name = str(lag_names[int(lag_idx[q])])
                lat_value = float(lat[int(spatial_idx[q, 0])])
                lon_value = float(lon[int(spatial_idx[q, 1])])
                row["lag_month"] = lag_name
                row["latitude"] = lat_value
                row["longitude"] = lon_value
                row["ocean_candidate_q"] = q
        output_rows.append(row)

    csv_path = OUTPUT_ROOT / "cobe2_sierra_swe_lod_modes.csv"
    fieldnames = [
        "mode_number",
        "selected",
        "stop_reason",
        "ocean_candidate_q",
        "lag_month",
        "latitude",
        "longitude",
        "corr_with_residual",
        "beta",
        "delta_r2",
        "cumulative_r2",
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
        },
        attrs={
            "description": "LOD diagnostics for COBE2 SST predictors against April 1 Sierra SWE standardized anomaly.",
            "sst_source": str(COBE2_SST_FILE),
            "sst_anomaly_reference": str(COBE2_MONTHLY_ANOM_REF),
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
    diagnostics_path = OUTPUT_ROOT / "cobe2_sierra_swe_lod_diagnostics.nc"
    diagnostics.to_netcdf(diagnostics_path, engine=NETCDF_ENGINE)

    summary = {
        "sst_source": str(COBE2_SST_FILE),
        "sst_anomaly_reference": str(COBE2_MONTHLY_ANOM_REF),
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
        "valid_ocean_cell_count": int(np.any(valid_mask, axis=0).sum()),
        "valid_counts_by_lag": {lag: int(count) for lag, count in zip(lag_names.tolist(), valid_counts_by_lag.tolist())},
        "lod_rows": output_rows,
        "retained_mode_count": int(mode_count),
        "runtime_seconds": perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
        "modes_csv": str(csv_path),
        "diagnostics_nc": str(diagnostics_path),
    }
    summary_path = OUTPUT_ROOT / "cobe2_sierra_swe_lod_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "modes_csv": str(csv_path), "diagnostics_nc": str(diagnostics_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
