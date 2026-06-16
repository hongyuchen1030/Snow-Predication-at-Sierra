#!/usr/bin/env python3
"""
Compare observed COBE2-PC -> ERA5 T2m teleconnections against saved
WUS projected-PC -> WUS-D3 T2m teleconnections.

This script reuses the already saved diagnostic NetCDF outputs and performs
only lightweight interpolation / comparison on a common Sierra land grid.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import DEFAULT_SIERRA_REGION


OBS_HOME_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level1_diagnostic"
OBS_NETCDF_PATH = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_pacific_sierra_t2m_level1_diagnostic/cobe2_pacific_sierra_t2m_level1_diagnostic.nc"
)
WUS_HOME_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_projected_pc_t2m_level1_ols" / "d03"
WUS_NETCDF_ROOT = WUS_HOME_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "wus_vs_observed_sst_t2m_teleconnection_comparison"
SUMMARY_CSV_PATH = OUTPUT_ROOT / "teleconnection_comparison_summary.csv"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "teleconnection_comparison_summary.json"
N_MODES = 6


@dataclass(frozen=True)
class ModeComparisonRow:
    dataset_id: str
    mode: int
    spatial_corr_beta: float
    sign_agreement_beta: float
    beta_amplitude_ratio: float
    mean_beta_obs: float
    mean_beta_wus: float
    mean_r2_obs: float
    mean_r2_wus: float
    mean_rho_obs: float
    mean_rho_wus: float
    spatial_corr_rho: float
    spatial_corr_r2: float
    n_common_land_cells: int
    classification: str


def normalize_lon_to_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(lon, dtype=np.float64), 360.0)


def discover_dataset_ids() -> List[str]:
    if not WUS_HOME_ROOT.exists():
        raise FileNotFoundError(f"Missing WUS diagnostic directory: {WUS_HOME_ROOT}")
    return sorted(path.name for path in WUS_HOME_ROOT.iterdir() if path.is_dir())


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    a1 = np.asarray(a, dtype=np.float64).ravel()
    b1 = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a1) & np.isfinite(b1)
    if int(valid.sum()) < 3:
        return float("nan")
    a2 = a1[valid]
    b2 = b1[valid]
    a_std = float(np.std(a2, ddof=1))
    b_std = float(np.std(b2, ddof=1))
    if a_std == 0.0 or b_std == 0.0 or not np.isfinite(a_std) or not np.isfinite(b_std):
        return float("nan")
    return float(np.corrcoef(a2, b2)[0, 1])


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a1 = np.asarray(a, dtype=np.float64).ravel()
    b1 = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a1) & np.isfinite(b1)
    if int(valid.sum()) == 0:
        return float("nan")
    same_sign = np.signbit(a1[valid]) == np.signbit(b1[valid])
    nonzero = (a1[valid] != 0.0) & (b1[valid] != 0.0)
    if int(nonzero.sum()) == 0:
        return float(np.mean(same_sign))
    return float(np.mean(same_sign[nonzero]))


def classify_mode(row: ModeComparisonRow) -> str:
    if (
        (np.isfinite(row.mean_r2_obs) and row.mean_r2_obs < 0.02)
        and (np.isfinite(row.mean_r2_wus) and row.mean_r2_wus < 0.02)
    ) or (np.isfinite(row.spatial_corr_beta) and abs(row.spatial_corr_beta) < 0.15):
        return "noisy/unreliable"
    if (
        np.isfinite(row.spatial_corr_beta)
        and row.spatial_corr_beta < -0.25
        and np.isfinite(row.sign_agreement_beta)
        and row.sign_agreement_beta < 0.45
    ):
        return "reversed"
    if (
        np.isfinite(row.spatial_corr_beta)
        and row.spatial_corr_beta > 0.25
        and np.isfinite(row.sign_agreement_beta)
        and row.sign_agreement_beta >= 0.60
    ):
        if (
            np.isfinite(row.beta_amplitude_ratio)
            and 0.5 <= row.beta_amplitude_ratio <= 2.0
            and np.isfinite(row.mean_r2_obs)
            and np.isfinite(row.mean_r2_wus)
            and row.mean_r2_wus >= 0.5 * row.mean_r2_obs
        ):
            return "reproduced"
        return "weak"
    return "weak"


def load_obs_dataset() -> xr.Dataset:
    return xr.open_dataset(OBS_NETCDF_PATH)


def load_wus_dataset(dataset_id: str) -> xr.Dataset:
    path = WUS_NETCDF_ROOT / dataset_id / "wusd3_projected_pc_t2m_level1_ols.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing WUS NetCDF for {dataset_id}: {path}")
    return xr.open_dataset(path)


def load_wus_summary(dataset_id: str) -> Dict[str, object]:
    path = WUS_HOME_ROOT / dataset_id / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def subset_wus_sierra(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    region = DEFAULT_SIERRA_REGION
    mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= region.lat_min)
        & (lat <= region.lat_max)
        & (lon >= region.lon_min)
        & (lon <= region.lon_max)
    )
    if not np.any(mask):
        raise ValueError("No WUS d03 points overlap the Sierra comparison region")
    row_idx = np.where(np.any(mask, axis=1))[0]
    col_idx = np.where(np.any(mask, axis=0))[0]
    return row_idx, col_idx, lat[row_idx[:, None], col_idx[None, :]], lon[row_idx[:, None], col_idx[None, :]]


def interpolate_obs_field_to_wus(
    obs_lat: np.ndarray,
    obs_lon_360: np.ndarray,
    obs_field: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (np.asarray(obs_lat, dtype=np.float64), np.asarray(obs_lon_360, dtype=np.float64)),
        np.asarray(obs_field, dtype=np.float64),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack(
        [
            np.asarray(target_lat, dtype=np.float64).ravel(),
            normalize_lon_to_360(np.asarray(target_lon, dtype=np.float64)).ravel(),
        ]
    )
    return interpolator(points).reshape(target_lat.shape)


def mean_on_mask(values: np.ndarray, mask: np.ndarray) -> float:
    subset = np.asarray(values, dtype=np.float64)[mask]
    if subset.size == 0:
        return float("nan")
    return float(np.nanmean(subset))


def std_on_mask(values: np.ndarray, mask: np.ndarray) -> float:
    subset = np.asarray(values, dtype=np.float64)[mask]
    valid = np.isfinite(subset)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.std(subset[valid], ddof=1))


def write_csv(path: Path, rows: Sequence[ModeComparisonRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else [field.name for field in ModeComparisonRow.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_mode_beta_comparison(
    output_path: Path,
    dataset_id: str,
    mode: int,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    beta_obs: np.ndarray,
    beta_wus: np.ndarray,
    beta_diff: np.ndarray,
    row: ModeComparisonRow,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    obs_vmax = float(np.nanmax(np.abs(np.stack([beta_obs, beta_wus], axis=0))))
    if not np.isfinite(obs_vmax) or obs_vmax == 0.0:
        obs_vmax = 1.0
    diff_vmax = float(np.nanmax(np.abs(beta_diff)))
    if not np.isfinite(diff_vmax) or diff_vmax == 0.0:
        diff_vmax = 1.0

    panels = [
        (axes[0], beta_obs, "Observed beta", "RdBu_r", -obs_vmax, obs_vmax, r"$\beta_{\mathrm{obs}}$ [K / 1$\sigma$ PC]"),
        (axes[1], beta_wus, "WUS beta", "RdBu_r", -obs_vmax, obs_vmax, r"$\beta_{\mathrm{wus}}$ [K / 1$\sigma$ PC]"),
        (axes[2], beta_diff, "WUS - observed", "RdBu_r", -diff_vmax, diff_vmax, r"$\Delta \beta$ [K / 1$\sigma$ PC]"),
    ]
    for ax, values, title, cmap, vmin, vmax, cbar_label in panels:
        mesh = ax.pcolormesh(lon2d, lat2d, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
        ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.9).set_label(cbar_label)

    fig.suptitle(
        f"{dataset_id} mode {mode}: beta teleconnection comparison\n"
        f"spatial corr={row.spatial_corr_beta:.3f}, sign agreement={row.sign_agreement_beta:.3f}, "
        f"amplitude ratio={row.beta_amplitude_ratio:.3f}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_summary_bars(output_path: Path, dataset_id: str, rows: Sequence[ModeComparisonRow]) -> None:
    modes = np.array([row.mode for row in rows], dtype=int)
    beta_corr = np.array([row.spatial_corr_beta for row in rows], dtype=np.float64)
    sign_frac = np.array([row.sign_agreement_beta for row in rows], dtype=np.float64)
    amp_ratio = np.array([row.beta_amplitude_ratio for row in rows], dtype=np.float64)
    mean_r2_obs = np.array([row.mean_r2_obs for row in rows], dtype=np.float64)
    mean_r2_wus = np.array([row.mean_r2_wus for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.22, hspace=0.32, wspace=0.16)
    axes = axes.ravel()
    width = 0.38

    axes[0].bar(modes, beta_corr, color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Spatial correlation of beta maps")
    axes[0].set_xlabel("Mode")
    axes[0].set_ylabel("corr")

    axes[1].bar(modes, sign_frac, color="#72b7b2")
    axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Beta sign agreement")
    axes[1].set_xlabel("Mode")
    axes[1].set_ylabel("fraction")

    axes[2].bar(modes, amp_ratio, color="#f58518")
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("Beta amplitude ratio")
    axes[2].set_xlabel("Mode")
    axes[2].set_ylabel("std(WUS) / std(obs)")

    axes[3].bar(modes - width / 2, mean_r2_obs, width=width, label="Observed", color="#54a24b")
    axes[3].bar(modes + width / 2, mean_r2_wus, width=width, label="WUS", color="#e45756")
    axes[3].set_title("Mean $R^2$ over common Sierra land")
    axes[3].set_xlabel("Mode")
    axes[3].set_ylabel("mean $R^2$")
    axes[3].legend()

    caption = "\n".join(
        [
            "How to read the first three panels:",
            "Spatial correlation of beta maps: compares observed vs WUS beta patterns ignoring absolute size.",
            "+1 means very similar, 0 means unrelated, negative means opposite pattern.",
            "Beta sign agreement: fraction of common land cells where observed and WUS beta have the same sign.",
            "1.0 means nearly all cells agree, 0.5 is roughly chance, near 0 means widespread sign reversal.",
            "Beta amplitude ratio = std(beta_wus) / std(beta_obs).",
            "1 means similar spatial amplitude, >1 means WUS is stronger or more contrasty, <1 means weaker or flatter.",
            "Important: amplitude ratio does not tell you whether the pattern is correct.",
            "A mode can have amplitude ratio near 1 and still be bad if spatial correlation is negative.",
            "For example, mode 1 here is strongly negative in spatial correlation, so WUS places warm/cool response opposite to observations.",
        ]
    )

    fig.suptitle(f"{dataset_id}: observed vs WUS teleconnection summary", fontsize=13)
    fig.text(
        0.5,
        0.08,
        caption,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#666666"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compare_dataset(dataset_id: str, obs_ds: xr.Dataset) -> List[ModeComparisonRow]:
    with load_wus_dataset(dataset_id) as wus_ds:
        wus_lat = np.asarray(wus_ds["latitude"].values, dtype=np.float64)
        wus_lon = np.asarray(wus_ds["longitude"].values, dtype=np.float64)
        row_idx, col_idx, sierra_lat, sierra_lon = subset_wus_sierra(wus_lat, wus_lon)
        lon2d = sierra_lon
        lat2d = sierra_lat

        obs_lat = np.asarray(obs_ds["sierra_latitude"].values, dtype=np.float64)
        obs_lon = np.asarray(obs_ds["sierra_longitude"].values, dtype=np.float64)
        obs_beta_all = np.asarray(obs_ds["sierra_era5_t2m_beta"].values, dtype=np.float64)
        obs_rho_all = np.asarray(obs_ds["sierra_era5_t2m_corr"].values, dtype=np.float64)
        obs_r2_all = np.asarray(obs_ds["sierra_era5_t2m_r2"].values, dtype=np.float64)

        wus_beta_all = np.asarray(wus_ds["wusd3_t2m_beta"].values, dtype=np.float64)[:, row_idx[:, None], col_idx[None, :]]
        wus_rho_all = np.asarray(wus_ds["wusd3_t2m_corr"].values, dtype=np.float64)[:, row_idx[:, None], col_idx[None, :]]
        wus_r2_all = np.asarray(wus_ds["wusd3_t2m_r2"].values, dtype=np.float64)[:, row_idx[:, None], col_idx[None, :]]

    dataset_output_dir = OUTPUT_ROOT / dataset_id
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[ModeComparisonRow] = []
    for mode_index in range(N_MODES):
        obs_beta = interpolate_obs_field_to_wus(obs_lat, obs_lon, obs_beta_all[mode_index], lat2d, lon2d)
        obs_rho = interpolate_obs_field_to_wus(obs_lat, obs_lon, obs_rho_all[mode_index], lat2d, lon2d)
        obs_r2 = interpolate_obs_field_to_wus(obs_lat, obs_lon, obs_r2_all[mode_index], lat2d, lon2d)

        wus_beta = np.asarray(wus_beta_all[mode_index], dtype=np.float64)
        wus_rho = np.asarray(wus_rho_all[mode_index], dtype=np.float64)
        wus_r2 = np.asarray(wus_r2_all[mode_index], dtype=np.float64)

        common_mask = (
            np.isfinite(lat2d)
            & np.isfinite(lon2d)
            & np.isfinite(obs_beta)
            & np.isfinite(wus_beta)
            & np.isfinite(obs_rho)
            & np.isfinite(wus_rho)
            & np.isfinite(obs_r2)
            & np.isfinite(wus_r2)
        )

        row = ModeComparisonRow(
            dataset_id=dataset_id,
            mode=mode_index + 1,
            spatial_corr_beta=spatial_corr(np.where(common_mask, obs_beta, np.nan), np.where(common_mask, wus_beta, np.nan)),
            sign_agreement_beta=sign_agreement(np.where(common_mask, obs_beta, np.nan), np.where(common_mask, wus_beta, np.nan)),
            beta_amplitude_ratio=(
                std_on_mask(wus_beta, common_mask) / std_on_mask(obs_beta, common_mask)
                if np.isfinite(std_on_mask(obs_beta, common_mask)) and std_on_mask(obs_beta, common_mask) not in (0.0, -0.0)
                else float("nan")
            ),
            mean_beta_obs=mean_on_mask(obs_beta, common_mask),
            mean_beta_wus=mean_on_mask(wus_beta, common_mask),
            mean_r2_obs=mean_on_mask(obs_r2, common_mask),
            mean_r2_wus=mean_on_mask(wus_r2, common_mask),
            mean_rho_obs=mean_on_mask(obs_rho, common_mask),
            mean_rho_wus=mean_on_mask(wus_rho, common_mask),
            spatial_corr_rho=spatial_corr(np.where(common_mask, obs_rho, np.nan), np.where(common_mask, wus_rho, np.nan)),
            spatial_corr_r2=spatial_corr(np.where(common_mask, obs_r2, np.nan), np.where(common_mask, wus_r2, np.nan)),
            n_common_land_cells=int(np.sum(common_mask)),
            classification="",
        )
        row = ModeComparisonRow(**{**asdict(row), "classification": classify_mode(row)})
        rows.append(row)

        plot_mode_beta_comparison(
            dataset_output_dir / f"mode{mode_index + 1}_beta_comparison.png",
            dataset_id,
            mode_index + 1,
            lon2d,
            lat2d,
            np.where(common_mask, obs_beta, np.nan),
            np.where(common_mask, wus_beta, np.nan),
            np.where(common_mask, wus_beta - obs_beta, np.nan),
            row,
        )

    plot_summary_bars(dataset_output_dir / "summary_bar_plots.png", dataset_id, rows)
    with (dataset_output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)
        handle.write("\n")
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    obs_summary_path = OBS_HOME_DIR / "cobe2_pacific_sierra_t2m_level1_summary.json"
    obs_summary = json.loads(obs_summary_path.read_text(encoding="utf-8"))
    if not bool(obs_summary.get("pc_standardized", False)):
        raise RuntimeError("Observed reference PCs are not standardized according to the saved summary")

    dataset_ids = discover_dataset_ids()
    all_rows: List[ModeComparisonRow] = []
    with load_obs_dataset() as obs_ds:
        for dataset_id in dataset_ids:
            all_rows.extend(compare_dataset(dataset_id, obs_ds))

    write_csv(SUMMARY_CSV_PATH, all_rows)
    summary_payload: Dict[str, object] = {
        "comparison_region": {
            "lat_min": float(DEFAULT_SIERRA_REGION.lat_min),
            "lat_max": float(DEFAULT_SIERRA_REGION.lat_max),
            "lon_min": float(DEFAULT_SIERRA_REGION.lon_min),
            "lon_max": float(DEFAULT_SIERRA_REGION.lon_max),
        },
        "observed_reference_path": str(OBS_NETCDF_PATH),
        "wus_reference_root": str(WUS_NETCDF_ROOT),
        "dataset_ids": dataset_ids,
        "rows": [asdict(row) for row in all_rows],
        "notes": [
            "Observed ERA5-based maps were interpolated onto the WUS d03 Sierra-region grid for comparison only.",
            "Both observed and WUS beta maps are treated as K per 1 sigma PC based on saved pipeline metadata/formulas.",
        ],
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_CSV_PATH}")
    print(f"Wrote {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
