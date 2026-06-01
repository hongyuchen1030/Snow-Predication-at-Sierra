#!/usr/bin/env python3
"""
Run a Level 1-style diagnostic linking projected WUS-on-COBE2 SST PCs to
monthly WUS-D3 d03 overland T2m anomalies.

For each dataset:
1. Load saved projected WUS_D3_PC_k(t) time series.
2. Load saved monthly WUS-D3 d03 t2 anomalies.
3. Align common monthly timestamps.
4. Compute for each mode k:
   - beta_k(x) = sum_t [PC_k(t) * T2_anom(t, x)] / stddev(PC_k)
   - rho_k(x) = corr(PC_k(t), T2_anom(t, x))
   - R_k^2(x) = rho_k(x)^2
5. Save maps, NetCDF, and summary metrics in the style of the Pacific diagnostic.
"""

import argparse
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

from snow_ml.data import DEFAULT_SIERRA_REGION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    ensure_runtime_on_compute_node,
    get_runtime,
    open_dataset_with_fallbacks,
)


EXPERIMENT_NAME = "wusd3_projected_pc_t2m_level1_diagnostic"
DEFAULT_DOMAIN = "d03"
DEFAULT_INPUT_PC_ROOT = PROJECT_ROOT / "artifacts" / "wus_sst_projected_onto_cobe2_eofs"
DEFAULT_INPUT_T2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_t2_monthly_anomalies")
PSCRATCH_OUTPUT_ROOT = Path(
    os.environ.get(
        "WUSD3_PROJECTED_PC_T2M_LEVEL1_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_projected_pc_t2m_level1_diagnostic",
    )
)
HOME_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "wusd3_projected_pc_t2m_level1_diagnostic"
N_MODES = 6


@dataclass(frozen=True)
class SummaryRow:
    mode: int
    pc_std_raw: float
    pc_mean_raw: float
    area_weighted_mean_r2: float
    area_weighted_mean_corr: float
    max_local_r2: float


@dataclass(frozen=True)
class SummaryPayload:
    experiment_name: str
    dataset_id: str
    domain: str
    input_projected_pc_path: str
    input_t2_anomaly_path: str
    output_netcdf_path: str
    output_figure_path: str
    overlap_start: str
    overlap_end: str
    n_overlap_months: int
    n_modes: int
    projected_pc_std: List[float]
    projected_pc_mean: List[float]
    t2_grid_shape: List[int]
    units: str
    slurm_job_id: str
    compute_node: str
    summary_rows: List[Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WUS projected-PC to T2m level-1 diagnostic.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="WUS domain, default d03.")
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Dataset id to process. Repeat for multiple. Defaults to all discovered datasets.",
    )
    parser.add_argument("--projected-pc-root", type=Path, default=DEFAULT_INPUT_PC_ROOT, help="Root of projected PC artifacts.")
    parser.add_argument("--t2-root", type=Path, default=DEFAULT_INPUT_T2_ROOT, help="Root of WUS monthly t2 anomaly artifacts.")
    return parser.parse_args()


def format_month(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def to_month_start(values: Sequence[np.datetime64]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("datetime64[M]").astype("datetime64[ns]")


def intersect_months(*time_axes: Sequence[np.datetime64]) -> np.ndarray:
    common = to_month_start(time_axes[0])
    for axis in time_axes[1:]:
        common = np.intersect1d(common, to_month_start(axis), assume_unique=False)
    return np.asarray(common, dtype="datetime64[ns]")


def build_time_index(month_values: np.ndarray) -> Dict[np.datetime64, int]:
    return {month: index for index, month in enumerate(month_values.tolist())}


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    month_values = to_month_start(time_values)
    index_by_month = build_time_index(month_values)
    return np.asarray(data)[[index_by_month[month] for month in target_months.tolist()]]


def discover_dataset_ids(projected_pc_root: Path, domain: str) -> List[str]:
    domain_root = projected_pc_root / domain
    if not domain_root.exists():
        raise FileNotFoundError("Missing projected PC domain directory: %s" % domain_root)
    return sorted(path.name for path in domain_root.iterdir() if path.is_dir())


def projected_pc_file(projected_pc_root: Path, domain: str, dataset_id: str) -> Path:
    return projected_pc_root / domain / dataset_id / "projected_pc_timeseries_and_mask.nc"


def t2_anomaly_file(t2_root: Path, domain: str, dataset_id: str) -> Path:
    return t2_root / domain / dataset_id / ("%s_%s_t2_monthly_anomaly.nc" % (dataset_id, domain))


def ensure_output_dirs(dataset_id: str, domain: str) -> Tuple[Path, Path]:
    pscratch_dir = PSCRATCH_OUTPUT_ROOT / domain / dataset_id
    home_dir = HOME_OUTPUT_ROOT / domain / dataset_id
    pscratch_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    return pscratch_dir, home_dir


def load_projected_pc(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(path) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        values = np.asarray(ds["projected_pc"].values, dtype=np.float64)
    return time, values


def load_t2_anomaly(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(path) as ds:
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        values = np.asarray(ds["t2_anomaly"].values, dtype=np.float64)
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
    return time, values, latitude, longitude


def compute_beta_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    pc_values = np.asarray(pc, dtype=np.float64)
    if pc_values.ndim != 1:
        raise ValueError("Expected 1D PC series, got %s" % (pc_values.shape,))
    pc_std = float(np.std(pc_values, ddof=1))
    if not np.isfinite(pc_std) or pc_std == 0.0:
        raise ValueError("PC standard deviation is zero or non-finite")
    return np.nansum(pc_values[:, np.newaxis, np.newaxis] * np.asarray(anomalies, dtype=np.float64), axis=0) / pc_std


def compute_corr_map(pc: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    pc_values = np.asarray(pc, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    pc_centered = pc_values - np.nanmean(pc_values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        y_centered = y - np.nanmean(y, axis=0, keepdims=True)
    numerator = np.nansum(pc_centered[:, np.newaxis, np.newaxis] * y_centered, axis=0)
    denominator = np.sqrt(np.nansum(pc_centered ** 2) * np.nansum(y_centered ** 2, axis=0))
    corr = np.full(y.shape[1:], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    corr[valid] = numerator[valid] / denominator[valid]
    return corr


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


def diagnostic_text_box() -> str:
    return "\n".join(
        [
            r"$\beta_k(r)$ is the physical T2m response pattern over the Sierra-region WUS-D3 d03 land grid.",
            r"$\beta_k(r)=\frac{\sum_t a_k(t)Y(t,r)}{\sigma(a_k)}$, "
            r"where $Y(t,r)$ is the WUS-D3 d03 monthly T2m anomaly at time $t$ and Sierra land grid point $r$, "
            r"and $a_k(t)$ is projected WUS-D3 SST pseudo-PC $k$ at time $t$.",
            r"$k$: PC index, for example $k=1$ means PC1. "
            r"$t$: time index, for example one month. "
            r"$r$: one Sierra-region WUS-D3 land grid point.",
            r"$\rho_k(r)=\operatorname{corr}(a_k(t),Y(t,r))$ is the correlation between projected WUS-D3 SST PC $k$ "
            r"and WUS-D3 d03 T2m at grid point $r$.",
            r"$R_k^2(r)=\rho_k(r)^2$ is the fraction of local WUS-D3 d03 T2m variance explained by perfect knowledge of SST PC $k$.",
        ]
    )


def subset_sierra_region(
    latitude: np.ndarray,
    longitude: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    region = DEFAULT_SIERRA_REGION
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    spatial_mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= region.lat_min)
        & (lat <= region.lat_max)
        & (lon >= region.lon_min)
        & (lon <= region.lon_max)
    )
    if not np.any(spatial_mask):
        raise ValueError("No d03 cells found inside the Sierra region bounds")

    row_mask = np.any(spatial_mask, axis=1)
    col_mask = np.any(spatial_mask, axis=0)
    row_idx = np.where(row_mask)[0]
    col_idx = np.where(col_mask)[0]

    lat_crop = lat[row_idx[:, np.newaxis], col_idx[np.newaxis, :]]
    lon_crop = lon[row_idx[:, np.newaxis], col_idx[np.newaxis, :]]

    beta_crop = beta_maps[:, row_idx[:, np.newaxis], col_idx[np.newaxis, :]]
    corr_crop = corr_maps[:, row_idx[:, np.newaxis], col_idx[np.newaxis, :]]
    r2_crop = r2_maps[:, row_idx[:, np.newaxis], col_idx[np.newaxis, :]]

    crop_mask = spatial_mask[row_idx[:, np.newaxis], col_idx[np.newaxis, :]]
    beta_crop = np.where(crop_mask[np.newaxis, :, :], beta_crop, np.nan)
    corr_crop = np.where(crop_mask[np.newaxis, :, :], corr_crop, np.nan)
    r2_crop = np.where(crop_mask[np.newaxis, :, :], r2_crop, np.nan)

    return lat_crop, lon_crop, beta_crop, corr_crop, r2_crop


def compute_mode_maps(
    projected_pc: np.ndarray,
    t2_anomalies: np.ndarray,
    latitude: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SummaryRow]]:
    beta_maps = np.full((N_MODES,) + t2_anomalies.shape[1:], np.nan, dtype=np.float64)
    corr_maps = np.full_like(beta_maps, np.nan)
    r2_maps = np.full_like(beta_maps, np.nan)
    rows: List[SummaryRow] = []

    for mode_index in range(N_MODES):
        pc_values = np.asarray(projected_pc[:, mode_index], dtype=np.float64)
        beta = compute_beta_map(pc_values, t2_anomalies)
        corr = compute_corr_map(pc_values, t2_anomalies)
        r2 = corr ** 2
        beta_maps[mode_index] = beta
        corr_maps[mode_index] = corr
        r2_maps[mode_index] = r2
        rows.append(
            SummaryRow(
                mode=mode_index + 1,
                pc_std_raw=float(np.std(pc_values, ddof=1)),
                pc_mean_raw=float(np.mean(pc_values)),
                area_weighted_mean_r2=area_weighted_mean(r2, latitude),
                area_weighted_mean_corr=area_weighted_mean(corr, latitude),
                max_local_r2=float(np.nanmax(r2)),
            )
        )
    return beta_maps, corr_maps, r2_maps, rows


def save_netcdf(
    path: Path,
    overlap_months: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    beta_maps: np.ndarray,
    corr_maps: np.ndarray,
    r2_maps: np.ndarray,
    projected_pc: np.ndarray,
    rows: List[SummaryRow],
    units: str,
    runtime,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "wusd3_t2m_beta": (("mode", "lat2d", "lon2d"), beta_maps.astype(np.float32)),
            "wusd3_t2m_corr": (("mode", "lat2d", "lon2d"), corr_maps.astype(np.float32)),
            "wusd3_t2m_r2": (("mode", "lat2d", "lon2d"), r2_maps.astype(np.float32)),
            "wusd3_projected_pc": (("time", "mode"), projected_pc.astype(np.float32)),
            "wusd3_area_weighted_mean_r2": (("mode",), np.asarray([row.area_weighted_mean_r2 for row in rows], dtype=np.float32)),
            "wusd3_area_weighted_mean_corr": (("mode",), np.asarray([row.area_weighted_mean_corr for row in rows], dtype=np.float32)),
            "wusd3_max_local_r2": (("mode",), np.asarray([row.max_local_r2 for row in rows], dtype=np.float32)),
        },
        coords={
            "time": overlap_months.astype("datetime64[ns]"),
            "mode": np.arange(1, N_MODES + 1, dtype=np.int32),
            "lat2d": np.arange(latitude.shape[0], dtype=np.int32),
            "lon2d": np.arange(latitude.shape[1], dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), latitude.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), longitude.astype(np.float32)),
        },
        attrs={
            "description": "WUS projected SST PC weighted WUS d03 T2m diagnostics",
            "formula_beta": "WUS_D3_T2M_k(x) = sum_t [WUS_D3_PC_k(t) * WUS_D3_T2M_anom(t, x)] / stddev(WUS_D3_PC_k(t))",
            "formula_corr": "rho_k(x) = corr(WUS_D3_PC_k(t), WUS_D3_T2M_anom(t, x))",
            "formula_r2": "R_k^2(x) = rho_k(x)^2",
            "units": units,
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(path)


def save_summary_csv(path: Path, rows: List[SummaryRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "pc_std_raw",
                "pc_mean_raw",
                "area_weighted_mean_r2",
                "area_weighted_mean_corr",
                "max_local_r2",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.mode,
                    "{:.12g}".format(row.pc_std_raw),
                    "{:.12g}".format(row.pc_mean_raw),
                    "{:.12g}".format(row.area_weighted_mean_r2),
                    "{:.12g}".format(row.area_weighted_mean_corr),
                    "{:.12g}".format(row.max_local_r2),
                ]
            )


def save_summary_json(
    path: Path,
    dataset_id: str,
    domain: str,
    projected_pc_path: Path,
    t2_path: Path,
    netcdf_path: Path,
    figure_path: Path,
    overlap_months: np.ndarray,
    projected_pc: np.ndarray,
    latitude: np.ndarray,
    units: str,
    rows: List[SummaryRow],
    runtime,
) -> None:
    payload = SummaryPayload(
        experiment_name=EXPERIMENT_NAME,
        dataset_id=dataset_id,
        domain=domain,
        input_projected_pc_path=str(projected_pc_path),
        input_t2_anomaly_path=str(t2_path),
        output_netcdf_path=str(netcdf_path),
        output_figure_path=str(figure_path),
        overlap_start=format_month(overlap_months[0]),
        overlap_end=format_month(overlap_months[-1]),
        n_overlap_months=int(overlap_months.size),
        n_modes=N_MODES,
        projected_pc_std=[float(np.std(projected_pc[:, idx], ddof=1)) for idx in range(N_MODES)],
        projected_pc_mean=[float(np.mean(projected_pc[:, idx])) for idx in range(N_MODES)],
        t2_grid_shape=[int(latitude.shape[0]), int(latitude.shape[1])],
        units=units,
        slurm_job_id=runtime.slurm_job_id,
        compute_node=runtime.hostname,
        summary_rows=[asdict(row) for row in rows],
    )
    path.write_text(json.dumps(asdict(payload), indent=2) + "\n", encoding="utf-8")


def plot_maps(path: Path, latitude: np.ndarray, longitude: np.ndarray, beta_maps: np.ndarray, corr_maps: np.ndarray, r2_maps: np.ndarray, rows: List[SummaryRow], dataset_id: str) -> None:
    sierra_latitude, sierra_longitude, sierra_beta, sierra_corr, sierra_r2 = subset_sierra_region(
        latitude, longitude, beta_maps, corr_maps, r2_maps
    )
    fig, axes = plt.subplots(N_MODES, 3, figsize=(15, 24), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.965, bottom=0.17, hspace=0.52, wspace=0.26)

    for mode_index in range(N_MODES):
        beta_ax, corr_ax, r2_ax = axes[mode_index]

        beta = np.asarray(sierra_beta[mode_index], dtype=np.float64)
        beta_vmax = float(np.nanmax(np.abs(beta)))
        if not np.isfinite(beta_vmax) or beta_vmax == 0.0:
            beta_vmax = 1.0
        beta_mesh = beta_ax.pcolormesh(
            sierra_longitude,
            sierra_latitude,
            beta,
            cmap="RdBu_r",
            shading="auto",
            vmin=-beta_vmax,
            vmax=beta_vmax,
        )
        beta_ax.set_title(rf"PC{mode_index + 1} regression $\beta_{{{mode_index + 1}}}(r)$")
        beta_ax.set_ylabel("Latitude")
        fig.colorbar(beta_mesh, ax=beta_ax, shrink=0.9).set_label(r"$\beta_k(r)$ [K / 1$\sigma$ PC]")

        corr = np.asarray(sierra_corr[mode_index], dtype=np.float64)
        corr_mesh = corr_ax.pcolormesh(
            sierra_longitude,
            sierra_latitude,
            corr,
            cmap="RdBu_r",
            shading="auto",
            vmin=-1.0,
            vmax=1.0,
        )
        corr_ax.set_title(
            rf"PC{mode_index + 1} correlation $\rho_{{{mode_index + 1}}}(r)$ | "
            rf"$\overline{{\rho}}={rows[mode_index].area_weighted_mean_corr:.3f}$"
        )
        fig.colorbar(corr_mesh, ax=corr_ax, shrink=0.9).set_label(r"$\rho_k(r)$")

        r2 = np.asarray(sierra_r2[mode_index], dtype=np.float64)
        r2_mesh = r2_ax.pcolormesh(
            sierra_longitude,
            sierra_latitude,
            r2,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max(0.05, float(np.nanmax(r2))),
        )
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

    fig.text(
        0.5,
        0.055,
        diagnostic_text_box(),
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#666666"},
    )
    fig.suptitle("%s projected SST PCs vs WUS d03 T2 anomalies" % dataset_id, fontsize=15, y=0.992)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def process_dataset(dataset_id: str, domain: str, projected_pc_root: Path, t2_root: Path, runtime) -> None:
    projected_pc_path = projected_pc_file(projected_pc_root, domain, dataset_id)
    t2_path = t2_anomaly_file(t2_root, domain, dataset_id)
    if not projected_pc_path.exists():
        raise FileNotFoundError("Missing projected PC file for %s: %s" % (dataset_id, projected_pc_path))
    if not t2_path.exists():
        raise FileNotFoundError("Missing t2 anomaly file for %s: %s" % (dataset_id, t2_path))

    pc_time, projected_pc = load_projected_pc(projected_pc_path)
    t2_time, t2_anomalies, latitude, longitude = load_t2_anomaly(t2_path)
    overlap_months = intersect_months(pc_time, t2_time)
    if overlap_months.size == 0:
        raise ValueError("No common months between projected PCs and t2 anomalies for %s" % dataset_id)

    projected_pc_overlap = select_by_months(pc_time, projected_pc, overlap_months)
    t2_overlap = select_by_months(t2_time, t2_anomalies, overlap_months)
    beta_maps, corr_maps, r2_maps, rows = compute_mode_maps(projected_pc_overlap, t2_overlap, latitude)

    pscratch_dir, home_dir = ensure_output_dirs(dataset_id, domain)
    netcdf_path = pscratch_dir / "wusd3_projected_pc_t2m_level1_diagnostic.nc"
    figure_path = home_dir / "wusd3_projected_pc_t2m_level1_maps_modes1to6.png"
    summary_csv_path = home_dir / "wusd3_projected_pc_t2m_level1_summary.csv"
    summary_json_path = home_dir / "wusd3_projected_pc_t2m_level1_summary.json"

    save_netcdf(netcdf_path, overlap_months, latitude, longitude, beta_maps, corr_maps, r2_maps, projected_pc_overlap, rows, "K", runtime)
    save_summary_csv(summary_csv_path, rows)
    save_summary_json(summary_json_path, dataset_id, domain, projected_pc_path, t2_path, netcdf_path, figure_path, overlap_months, projected_pc_overlap, latitude, "K", rows, runtime)
    plot_maps(figure_path, latitude, longitude, beta_maps, corr_maps, r2_maps, rows, dataset_id)

    print(
        "Finished %s overlap=%s..%s n_months=%d mean_R2_mode1=%.4f"
        % (
            dataset_id,
            format_month(overlap_months[0]),
            format_month(overlap_months[-1]),
            int(overlap_months.size),
            rows[0].area_weighted_mean_r2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)

    dataset_ids = args.dataset_id if args.dataset_id else discover_dataset_ids(args.projected_pc_root, args.domain)
    for dataset_id in dataset_ids:
        process_dataset(dataset_id, args.domain, args.projected_pc_root, args.t2_root, runtime)


if __name__ == "__main__":
    main()
