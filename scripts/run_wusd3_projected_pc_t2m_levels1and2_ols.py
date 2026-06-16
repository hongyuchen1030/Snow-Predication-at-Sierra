#!/usr/bin/env python3
"""
Mirror the observational COBE2->ERA5 diagnostics for WUS using saved projected PCs.

Level 1:
  one projected pseudo-PC at a time, standardized, univariate OLS
Level 2:
  all six projected pseudo-PCs together, standardized, multivariate OLS

No expensive preprocessing is redone. Inputs are saved projected PCs and saved
monthly WUS-D3 d03 T2 anomaly fields.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
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
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import DEFAULT_SIERRA_REGION


DOMAIN = "d03"
N_MODES = 6
PROJECTED_PC_ROOT = PROJECT_ROOT / "artifacts" / "wus_sst_projected_onto_cobe2_eofs" / DOMAIN
T2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_t2_monthly_anomalies") / DOMAIN
LEVEL1_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_projected_pc_t2m_level1_ols" / DOMAIN
LEVEL2_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_projected_pc_t2m_level2_ols" / DOMAIN


@dataclass(frozen=True)
class Level1SummaryRow:
    mode: int
    pc_std_raw: float
    pc_mean_raw: float
    area_weighted_mean_r2: float
    area_weighted_mean_corr: float
    max_local_r2: float


@dataclass(frozen=True)
class Level2SummaryRow:
    mode: int
    pc_std_raw: float
    pc_mean_raw: float


def discover_dataset_ids() -> List[str]:
    return sorted(path.name for path in PROJECTED_PC_ROOT.iterdir() if path.is_dir())


def projected_pc_file(dataset_id: str) -> Path:
    return PROJECTED_PC_ROOT / dataset_id / "projected_pc_timeseries_and_mask.nc"


def t2_anomaly_file(dataset_id: str) -> Path:
    return T2_ROOT / dataset_id / f"{dataset_id}_{DOMAIN}_t2_monthly_anomaly.nc"


def to_month_start(values: Sequence[np.datetime64]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("datetime64[M]").astype("datetime64[ns]")


def intersect_months(*time_axes: Sequence[np.datetime64]) -> np.ndarray:
    common = to_month_start(time_axes[0])
    for axis in time_axes[1:]:
        common = np.intersect1d(common, to_month_start(axis), assume_unique=False)
    return np.asarray(common, dtype="datetime64[ns]")


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    month_values = to_month_start(time_values)
    index_by_month = {month: idx for idx, month in enumerate(month_values.tolist())}
    return np.asarray(data)[[index_by_month[month] for month in target_months.tolist()]]


def standardize_pc_matrix(pc_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(pc_matrix, dtype=np.float64)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
        raise ValueError("Cannot standardize PCs with non-finite or zero std")
    standardized = (values - means[np.newaxis, :]) / stds[np.newaxis, :]
    return standardized, means, stds


def load_projected_pc(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        values = np.asarray(ds["projected_pc"].values, dtype=np.float64)
    return time, values


def load_t2_anomaly(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        values = np.asarray(ds["t2_anomaly"].values, dtype=np.float64)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return time, values, lat, lon


def area_weighted_mean(field: np.ndarray, latitude_2d: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    latitude = np.asarray(latitude_2d, dtype=np.float64)
    weights = np.cos(np.deg2rad(latitude))
    valid = np.isfinite(values)
    weighted_sum = np.nansum(np.where(valid, values * weights, 0.0))
    weight_sum = np.nansum(np.where(valid, weights, 0.0))
    if not np.isfinite(weight_sum) or weight_sum == 0.0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def compute_beta_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    a = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    denominator = float(np.sum(a ** 2))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("Standardized PC denominator is zero")
    return np.nansum(a[:, np.newaxis, np.newaxis] * y, axis=0) / denominator


def compute_corr_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    a = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    a_centered = a - np.nanmean(a)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        y_centered = y - np.nanmean(y, axis=0, keepdims=True)
    numerator = np.nansum(a_centered[:, np.newaxis, np.newaxis] * y_centered, axis=0)
    denominator = np.sqrt(np.nansum(a_centered ** 2) * np.nansum(y_centered ** 2, axis=0))
    out = np.full(y.shape[1:], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def subset_sierra_region(latitude: np.ndarray, longitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    region = DEFAULT_SIERRA_REGION
    mask = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= region.lat_min)
        & (latitude <= region.lat_max)
        & (longitude >= region.lon_min)
        & (longitude <= region.lon_max)
    )
    row_idx = np.where(np.any(mask, axis=1))[0]
    col_idx = np.where(np.any(mask, axis=0))[0]
    return row_idx, col_idx


def plot_level1_maps(
    output_path: Path,
    dataset_id: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
    rows: List[Level1SummaryRow],
) -> None:
    row_idx, col_idx = subset_sierra_region(latitude, longitude)
    lat = latitude[row_idx[:, None], col_idx[None, :]]
    lon = longitude[row_idx[:, None], col_idx[None, :]]
    beta_crop = beta_maps[:, row_idx[:, None], col_idx[None, :]]
    corr_crop = corr_maps[:, row_idx[:, None], col_idx[None, :]]
    r2_crop = r2_maps[:, row_idx[:, None], col_idx[None, :]]
    fig, axes = plt.subplots(N_MODES, 3, figsize=(15, 24), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.05, hspace=0.52, wspace=0.26)
    for mode_index in range(N_MODES):
        beta_ax, corr_ax, r2_ax = axes[mode_index]
        beta = beta_crop[mode_index]
        beta_vmax = float(np.nanmax(np.abs(beta)))
        if not np.isfinite(beta_vmax) or beta_vmax == 0.0:
            beta_vmax = 1.0
        beta_mesh = beta_ax.pcolormesh(lon, lat, beta, cmap="RdBu_r", shading="auto", vmin=-beta_vmax, vmax=beta_vmax)
        beta_ax.set_title(rf"PC{mode_index + 1} regression $\beta_{{{mode_index + 1}}}(r)$")
        fig.colorbar(beta_mesh, ax=beta_ax, shrink=0.9).set_label(r"$\beta_k(r)$ [K / 1$\sigma$ PC]")

        corr = corr_crop[mode_index]
        corr_mesh = corr_ax.pcolormesh(lon, lat, corr, cmap="RdBu_r", shading="auto", vmin=-1.0, vmax=1.0)
        corr_ax.set_title(
            rf"PC{mode_index + 1} correlation $\rho_{{{mode_index + 1}}}(r)$ | "
            rf"$\overline{{\rho}}={rows[mode_index].area_weighted_mean_corr:.3f}$"
        )
        fig.colorbar(corr_mesh, ax=corr_ax, shrink=0.9).set_label(r"$\rho_k(r)$")

        r2 = r2_crop[mode_index]
        r2_mesh = r2_ax.pcolormesh(lon, lat, r2, cmap="viridis", shading="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(r2))))
        r2_ax.set_title(
            rf"PC{mode_index + 1} explained variance $R_{{{mode_index + 1}}}^2(r)$ | "
            rf"$\overline{{R^2}}={rows[mode_index].area_weighted_mean_r2:.3f}$"
        )
        fig.colorbar(r2_mesh, ax=r2_ax, shrink=0.9).set_label(r"$R_k^2(r)$")

        for ax in (beta_ax, corr_ax, r2_ax):
            ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
            ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
    fig.suptitle(f"{dataset_id} WUS Level 1 OLS: projected PCs -> WUS d03 T2 anomalies", fontsize=15, y=0.992)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def fit_multivariate_regression(predictors: np.ndarray, anomalies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    y_flat = y.reshape(y.shape[0], -1)
    valid_columns = np.isfinite(y_flat).all(axis=0)
    y_valid = y_flat[:, valid_columns]
    b_valid, _, _, _ = np.linalg.lstsq(a, y_valid, rcond=None)
    yhat_valid = a @ b_valid
    residual_sum = np.sum((y_valid - yhat_valid) ** 2, axis=0)
    total_sum = np.sum(y_valid ** 2, axis=0)
    r2_valid = np.full(total_sum.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(total_sum) & (total_sum > 0.0)
    r2_valid[positive] = 1.0 - (residual_sum[positive] / total_sum[positive])
    coefficient_maps = np.full((N_MODES, y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    coefficient_maps.reshape(N_MODES, -1)[:, valid_columns] = b_valid
    r2_map = np.full((y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    r2_map.reshape(-1)[valid_columns] = r2_valid
    return coefficient_maps, r2_map


def plot_level2_coefficients(
    output_path: Path,
    dataset_id: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    coefficient_maps: np.ndarray,
) -> None:
    row_idx, col_idx = subset_sierra_region(latitude, longitude)
    lat = latitude[row_idx[:, None], col_idx[None, :]]
    lon = longitude[row_idx[:, None], col_idx[None, :]]
    coeff_crop = coefficient_maps[:, row_idx[:, None], col_idx[None, :]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(coeff_crop)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    for mode_index, ax in enumerate(axes.ravel()):
        mesh = ax.pcolormesh(lon, lat, coeff_crop[mode_index], cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(rf"PC{mode_index + 1} coefficient $\hat{{B}}_{{{mode_index + 1}}}(r)$")
        ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
        ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.86).set_label(r"$\hat{B}_k(r)$ [K / 1$\sigma$ PC]")
    fig.suptitle(f"{dataset_id} WUS Level 2 OLS: joint PC1-PC6 coefficients", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def process_dataset(dataset_id: str) -> None:
    level1_dir = LEVEL1_ROOT / dataset_id
    level2_dir = LEVEL2_ROOT / dataset_id
    level1_dir.mkdir(parents=True, exist_ok=True)
    level2_dir.mkdir(parents=True, exist_ok=True)

    pc_time, pc_raw = load_projected_pc(projected_pc_file(dataset_id))
    t2_time, t2_anom, latitude, longitude = load_t2_anomaly(t2_anomaly_file(dataset_id))
    overlap_months = intersect_months(pc_time, t2_time)
    pc_raw = select_by_months(pc_time, pc_raw, overlap_months)
    t2_overlap = select_by_months(t2_time, t2_anom, overlap_months)
    pc_std, pc_raw_mean, pc_raw_std = standardize_pc_matrix(pc_raw)

    beta_maps = np.full((N_MODES,) + t2_overlap.shape[1:], np.nan, dtype=np.float64)
    corr_maps = np.full_like(beta_maps, np.nan)
    r2_maps = np.full_like(beta_maps, np.nan)
    level1_rows: List[Level1SummaryRow] = []
    for mode_index in range(N_MODES):
        a = pc_std[:, mode_index]
        beta = compute_beta_map(a, t2_overlap)
        corr = compute_corr_map(a, t2_overlap)
        r2 = corr ** 2
        beta_maps[mode_index] = beta
        corr_maps[mode_index] = corr
        r2_maps[mode_index] = r2
        level1_rows.append(
            Level1SummaryRow(
                mode=mode_index + 1,
                pc_std_raw=float(pc_raw_std[mode_index]),
                pc_mean_raw=float(pc_raw_mean[mode_index]),
                area_weighted_mean_r2=area_weighted_mean(r2, latitude),
                area_weighted_mean_corr=area_weighted_mean(corr, latitude),
                max_local_r2=float(np.nanmax(r2)),
            )
        )

    ds1 = xr.Dataset(
        data_vars={
            "wusd3_t2m_beta": (("mode", "lat2d", "lon2d"), beta_maps.astype(np.float32)),
            "wusd3_t2m_corr": (("mode", "lat2d", "lon2d"), corr_maps.astype(np.float32)),
            "wusd3_t2m_r2": (("mode", "lat2d", "lon2d"), r2_maps.astype(np.float32)),
            "wusd3_projected_pc_standardized": (("time", "mode"), pc_std.astype(np.float32)),
            "wusd3_projected_pc_raw_mean": (("mode",), pc_raw_mean.astype(np.float32)),
            "wusd3_projected_pc_raw_std": (("mode",), pc_raw_std.astype(np.float32)),
        },
        coords={
            "time": overlap_months,
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude.astype(np.float32)),
        },
        attrs={
            "description": "WUS Level 1 OLS diagnostic mirroring observational COBE2->ERA5 Level 1",
            "formula_beta": "beta_k(r) = sum_t[a_k(t) Y(t,r)] / sum_t[a_k(t)^2]",
            "formula_corr": "rho_k(r) = corr(a_k(t), Y(t,r))",
            "formula_r2": "R_k^2(r) = rho_k(r)^2",
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "dataset_id": dataset_id,
            "domain": DOMAIN,
        },
    )
    ds1.to_netcdf(level1_dir / "wusd3_projected_pc_t2m_level1_ols.nc")
    (level1_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "domain": DOMAIN,
                "projected_pc_file": str(projected_pc_file(dataset_id)),
                "t2_anomaly_file": str(t2_anomaly_file(dataset_id)),
                "overlap_start": str(overlap_months[0].astype("datetime64[D]")),
                "overlap_end": str(overlap_months[-1].astype("datetime64[D]")),
                "n_overlap_months": int(overlap_months.size),
                "projection_matrix_shape": [int(pc_std.shape[0]), int(pc_std.shape[1])],
                "t2_grid_shape": [int(latitude.shape[0]), int(latitude.shape[1])],
                "pc_standardized": True,
                "summary_rows": [asdict(row) for row in level1_rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (level1_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(level1_rows[0]).keys()))
        writer.writeheader()
        for row in level1_rows:
            writer.writerow(asdict(row))
    plot_level1_maps(level1_dir / "wusd3_projected_pc_t2m_level1_ols_maps_modes1to6.png", dataset_id, latitude, longitude, beta_maps, corr_maps, r2_maps, level1_rows)

    coefficient_maps, r2_map = fit_multivariate_regression(pc_std, t2_overlap)
    ds2 = xr.Dataset(
        data_vars={
            "wusd3_t2m_multi_pc_beta": (("mode", "lat2d", "lon2d"), coefficient_maps.astype(np.float32)),
            "wusd3_t2m_multi_pc_r2": (("lat2d", "lon2d"), r2_map.astype(np.float32)),
            "wusd3_projected_pc_standardized": (("time", "mode"), pc_std.astype(np.float32)),
            "wusd3_projected_pc_raw_mean": (("mode",), pc_raw_mean.astype(np.float32)),
            "wusd3_projected_pc_raw_std": (("mode",), pc_raw_std.astype(np.float32)),
        },
        coords={
            "time": overlap_months,
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude.astype(np.float32)),
        },
        attrs={
            "description": "WUS Level 2 OLS diagnostic mirroring observational COBE2->ERA5 Level 2",
            "formula_bhat": "B_hat = (A^T A)^(-1) A^T Y",
            "formula_yhat": "Y_hat = A B_hat",
            "formula_r2": "R2(r) = 1 - sum_t[(Y(t,r)-Y_hat(t,r))^2] / sum_t[Y(t,r)^2]",
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "dataset_id": dataset_id,
            "domain": DOMAIN,
        },
    )
    ds2.to_netcdf(level2_dir / "wusd3_projected_pc_t2m_level2_ols.nc")
    (level2_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "domain": DOMAIN,
                "projected_pc_file": str(projected_pc_file(dataset_id)),
                "t2_anomaly_file": str(t2_anomaly_file(dataset_id)),
                "overlap_start": str(overlap_months[0].astype("datetime64[D]")),
                "overlap_end": str(overlap_months[-1].astype("datetime64[D]")),
                "n_overlap_months": int(overlap_months.size),
                "projection_matrix_shape": [int(pc_std.shape[0]), int(pc_std.shape[1])],
                "t2_grid_shape": [int(latitude.shape[0]), int(latitude.shape[1])],
                "pc_standardized": True,
                "mean_r2_joint": area_weighted_mean(r2_map, latitude),
                "projected_pc_std_raw": [float(v) for v in pc_raw_std.tolist()],
                "projected_pc_mean_raw": [float(v) for v in pc_raw_mean.tolist()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_level2_coefficients(level2_dir / "wusd3_projected_pc_t2m_level2_ols_coefficients_modes1to6.png", dataset_id, latitude, longitude, coefficient_maps)
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    mesh = ax.pcolormesh(r2_map, shading="auto", cmap="viridis", vmin=0.0, vmax=max(0.05, float(np.nanmax(r2_map))))
    ax.set_title("WUS Level 2 OLS joint R2")
    ax.set_xlabel("lon2d index")
    ax.set_ylabel("lat2d index")
    fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.savefig(level2_dir / "wusd3_projected_pc_t2m_level2_ols_r2_map.png", dpi=220)
    plt.close(fig)
    print(f"Finished {dataset_id}: Level1->{level1_dir} Level2->{level2_dir}", flush=True)


def main() -> None:
    LEVEL1_ROOT.mkdir(parents=True, exist_ok=True)
    LEVEL2_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_id in discover_dataset_ids():
        process_dataset(dataset_id)


if __name__ == "__main__":
    main()
