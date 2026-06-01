#!/usr/bin/env python3
"""
Run a Level 2 sensitivity diagnostic linking Pacific COBE2 SST PCs 1-6
to matched-region ERA5-Land T2m anomalies.
"""

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

from scripts.run_cobe2_pacific_sierra_t2m_level1_diagnostic import (
    COBE2_SST_FILE,
    ERA5_MONTHLY_CLIM_FILE,
    ERA5_MONTHLY_MEAN_FILE,
    LAT_CHUNK,
    LON_CHUNK,
    PACIFIC_SST_REGION_360,
    SIERRA_T2M_REGION_360,
    TIME_CHUNK,
    area_weighted_mean,
    build_time_index,
    ensure_runtime_on_compute_node,
    format_month,
    get_runtime,
    load_pacific_cobe2_pca,
    month_number,
    standardize_pc_matrix,
    subset_era5_region_360,
    to_month_start,
)


EXPERIMENT_NAME = "cobe2_pacific_sierra_t2m_level2_pc1to6"
MODEL_NAME = "PC1_to_PC6"
N_PREDICTORS = 6
PSCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "COBE2_PACIFIC_SIERRA_T2M_LEVEL2_PC1TO6_OUTPUT_DIR",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6",
    )
)
HOME_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6"
NETCDF_FILE = PSCRATCH_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
SUMMARY_CSV_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_summary.csv"
SUMMARY_JSON_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_summary.json"
FIGURE_FILE = HOME_OUTPUT_DIR / "cobe2_pacific_sierra_t2m_level2_pc1to6_maps.png"

ERA5_VARIABLE = "t2m"


@dataclass(frozen=True)
class SummaryRow:
    model_name: str
    n_pcs: int
    overlap_start: str
    overlap_end: str
    n_overlap_months: int
    sierra_area_weighted_mean_r2: float
    sierra_max_local_r2: float
    sierra_min_local_r2: float
    sierra_median_local_r2: float
    pc1_mean_raw: float
    pc2_mean_raw: float
    pc3_mean_raw: float
    pc4_mean_raw: float
    pc5_mean_raw: float
    pc6_mean_raw: float
    pc1_std_raw: float
    pc2_std_raw: float
    pc3_std_raw: float
    pc4_std_raw: float
    pc5_std_raw: float
    pc6_std_raw: float


def ensure_output_dir() -> None:
    PSCRATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HOME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def output_dir_size_text() -> str:
    total_bytes = 0
    for path in PSCRATCH_OUTPUT_DIR.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
    units = ["B", "K", "M", "G", "T", "P"]
    size = float(total_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)}{units[unit_index]}"
    return f"{size:.1f}{units[unit_index]}"


def fit_multivariate_regression(
    predictors: np.ndarray,
    anomalies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != N_PREDICTORS:
        raise ValueError(f"Predictor matrix must have shape (n_time, {N_PREDICTORS})")
    if y.ndim != 3 or y.shape[0] != a.shape[0]:
        raise ValueError("Anomaly cube must have shape (n_time, lat, lon) aligned with predictors")

    y_flat = y.reshape(y.shape[0], -1)
    valid_columns = np.isfinite(y_flat).all(axis=0)
    if not np.any(valid_columns):
        raise ValueError("No all-time-finite matched-region grid points available for regression")

    y_valid = y_flat[:, valid_columns]
    b_valid, _, _, _ = np.linalg.lstsq(a, y_valid, rcond=None)
    yhat_valid = a @ b_valid

    residual_sum = np.sum((y_valid - yhat_valid) ** 2, axis=0)
    total_sum = np.sum(y_valid ** 2, axis=0)
    r2_valid = np.full(total_sum.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(total_sum) & (total_sum > 0.0)
    r2_valid[positive] = 1.0 - (residual_sum[positive] / total_sum[positive])

    coefficient_maps = np.full((N_PREDICTORS, y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    coefficient_maps.reshape(N_PREDICTORS, -1)[:, valid_columns] = b_valid

    r2_map = np.full((y.shape[1], y.shape[2]), np.nan, dtype=np.float64)
    r2_map.reshape(-1)[valid_columns] = r2_valid
    return coefficient_maps, r2_map


def summarize_r2(
    r2_map: np.ndarray,
    latitude: np.ndarray,
    overlap_months: np.ndarray,
    predictor_matrix_raw_mean: np.ndarray,
    predictor_matrix_raw_std: np.ndarray,
    predictor_matrix: np.ndarray,
) -> SummaryRow:
    r2_valid = np.asarray(r2_map, dtype=np.float64)
    finite = np.isfinite(r2_valid)
    return SummaryRow(
        model_name=MODEL_NAME,
        n_pcs=N_PREDICTORS,
        overlap_start=format_month(overlap_months[0]),
        overlap_end=format_month(overlap_months[-1]),
        n_overlap_months=int(overlap_months.size),
        sierra_area_weighted_mean_r2=area_weighted_mean(r2_valid, latitude),
        sierra_max_local_r2=float(np.nanmax(r2_valid)),
        sierra_min_local_r2=float(np.nanmin(r2_valid)),
        sierra_median_local_r2=float(np.nanmedian(r2_valid[finite])) if np.any(finite) else float("nan"),
        pc1_mean_raw=float(predictor_matrix_raw_mean[0]),
        pc2_mean_raw=float(predictor_matrix_raw_mean[1]),
        pc3_mean_raw=float(predictor_matrix_raw_mean[2]),
        pc4_mean_raw=float(predictor_matrix_raw_mean[3]),
        pc5_mean_raw=float(predictor_matrix_raw_mean[4]),
        pc6_mean_raw=float(predictor_matrix_raw_mean[5]),
        pc1_std_raw=float(predictor_matrix_raw_std[0]),
        pc2_std_raw=float(predictor_matrix_raw_std[1]),
        pc3_std_raw=float(predictor_matrix_raw_std[2]),
        pc4_std_raw=float(predictor_matrix_raw_std[3]),
        pc5_std_raw=float(predictor_matrix_raw_std[4]),
        pc6_std_raw=float(predictor_matrix_raw_std[5]),
    )


def save_netcdf(
    pacific: Dict[str, np.ndarray],
    overlap_months: np.ndarray,
    predictor_matrix: np.ndarray,
    predictor_matrix_raw_mean: np.ndarray,
    predictor_matrix_raw_std: np.ndarray,
    sierra_latitude: np.ndarray,
    sierra_longitude: np.ndarray,
    coefficient_maps: np.ndarray,
    r2_map: np.ndarray,
    summary: SummaryRow,
    runtime,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "pacific_cobe2_pc": (("time", "mode"), predictor_matrix.astype(np.float32)),
            "pacific_cobe2_pc_mean_raw": (("mode",), predictor_matrix_raw_mean.astype(np.float32)),
            "pacific_cobe2_pc_std_raw": (("mode",), predictor_matrix_raw_std.astype(np.float32)),
            "pacific_cobe2_eof": (
                ("mode", "pacific_latitude", "pacific_longitude"),
                pacific["eof"][:N_PREDICTORS].astype(np.float32),
            ),
            "sierra_era5_t2m_multi_pc_beta": (
                ("mode", "sierra_latitude", "sierra_longitude"),
                coefficient_maps.astype(np.float32),
            ),
            "sierra_era5_t2m_multi_pc_r2": (
                ("sierra_latitude", "sierra_longitude"),
                r2_map.astype(np.float32),
            ),
            "explained_variance_ratio": (("mode",), pacific["explained_variance_ratio"][:N_PREDICTORS].astype(np.float32)),
            "singular_value": (("mode",), pacific["singular_value"][:N_PREDICTORS].astype(np.float32)),
        },
        coords={
            "time": overlap_months.astype("datetime64[ns]"),
            "mode": np.arange(1, N_PREDICTORS + 1, dtype=np.int32),
            "pacific_latitude": pacific["latitude"].astype(np.float32),
            "pacific_longitude": pacific["longitude"].astype(np.float32),
            "sierra_latitude": sierra_latitude.astype(np.float32),
            "sierra_longitude": sierra_longitude.astype(np.float32),
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "formula_bhat": "B_hat = (A^T A)^(-1) A^T Y",
            "formula_yhat": "Y_hat = A B_hat",
            "formula_r2": "R2(r) = 1 - sum_t[(Y(t,r)-Y_hat(t,r))^2] / sum_t[Y(t,r)^2]",
            "pacific_sst_region_360": json.dumps(PACIFIC_SST_REGION_360.as_dict()),
            "sierra_t2m_region_360": json.dumps(SIERRA_T2M_REGION_360.as_dict()),
            "pc_standardized": "true",
            "pc_standardization": "standardized over overlap months using sample mean and sample std (ddof=1)",
            "time_overlap_start": summary.overlap_start,
            "time_overlap_end": summary.overlap_end,
            "sierra_area_weighted_mean_r2": summary.sierra_area_weighted_mean_r2,
            "sierra_max_local_r2": summary.sierra_max_local_r2,
            "sierra_min_local_r2": summary.sierra_min_local_r2,
            "sierra_median_local_r2": summary.sierra_median_local_r2,
            "slurm_job_id": runtime.slurm_job_id,
            "compute_node": runtime.hostname,
        },
    )
    ds.to_netcdf(
        NETCDF_FILE,
        engine="netcdf4",
        encoding={
            "sierra_era5_t2m_multi_pc_beta": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
            "sierra_era5_t2m_multi_pc_r2": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
            "pacific_cobe2_eof": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)},
        },
    )


def save_summary(summary: SummaryRow, pacific: Dict[str, np.ndarray], sierra_shape: List[int], runtime) -> None:
    fieldnames = [
        "model_name",
        "n_pcs",
        "overlap_start",
        "overlap_end",
        "n_overlap_months",
        "sierra_area_weighted_mean_r2",
        "sierra_max_local_r2",
        "sierra_min_local_r2",
        "sierra_median_local_r2",
        "pc1_mean_raw",
        "pc2_mean_raw",
        "pc3_mean_raw",
        "pc4_mean_raw",
        "pc5_mean_raw",
        "pc6_mean_raw",
        "pc1_std_raw",
        "pc2_std_raw",
        "pc3_std_raw",
        "pc4_std_raw",
        "pc5_std_raw",
        "pc6_std_raw",
    ]
    with SUMMARY_CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerow(
            [
                summary.model_name,
                summary.n_pcs,
                summary.overlap_start,
                summary.overlap_end,
                summary.n_overlap_months,
                "{:.12g}".format(summary.sierra_area_weighted_mean_r2),
                "{:.12g}".format(summary.sierra_max_local_r2),
                "{:.12g}".format(summary.sierra_min_local_r2),
                "{:.12g}".format(summary.sierra_median_local_r2),
                "{:.12g}".format(summary.pc1_mean_raw),
                "{:.12g}".format(summary.pc2_mean_raw),
                "{:.12g}".format(summary.pc3_mean_raw),
                "{:.12g}".format(summary.pc4_mean_raw),
                "{:.12g}".format(summary.pc5_mean_raw),
                "{:.12g}".format(summary.pc6_mean_raw),
                "{:.12g}".format(summary.pc1_std_raw),
                "{:.12g}".format(summary.pc2_std_raw),
                "{:.12g}".format(summary.pc3_std_raw),
                "{:.12g}".format(summary.pc4_std_raw),
                "{:.12g}".format(summary.pc5_std_raw),
                "{:.12g}".format(summary.pc6_std_raw),
            ]
        )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "n_pcs": N_PREDICTORS,
        "pacific_sst_region": PACIFIC_SST_REGION_360.as_dict(),
        "sierra_t2m_region": SIERRA_T2M_REGION_360.as_dict(),
        "input_cobe2_sst_path": str(COBE2_SST_FILE),
        "input_era5_monthly_mean_path": str(ERA5_MONTHLY_MEAN_FILE),
        "input_era5_monthly_climatology_path": str(ERA5_MONTHLY_CLIM_FILE),
        "output_netcdf_path": str(NETCDF_FILE),
        "output_figure_path": str(FIGURE_FILE),
        "pc_standardized": True,
        "pacific_sst_shape": [int(pacific["latitude"].size), int(pacific["longitude"].size)],
        "sierra_t2m_shape": sierra_shape,
        "slurm_job_id": runtime.slurm_job_id,
        "compute_node": runtime.hostname,
        "summary_row": asdict(summary),
        "output_directory_size": output_dir_size_text(),
    }
    SUMMARY_JSON_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_plot_inputs_from_netcdf() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    with xr.open_dataset(NETCDF_FILE, engine="netcdf4", decode_times=True) as ds:
        sierra_latitude = np.asarray(ds["sierra_latitude"].values, dtype=np.float64)
        sierra_longitude = np.asarray(ds["sierra_longitude"].values, dtype=np.float64)
        coefficient_maps = np.asarray(ds["sierra_era5_t2m_multi_pc_beta"].values, dtype=np.float64)
        r2_map = np.asarray(ds["sierra_era5_t2m_multi_pc_r2"].values, dtype=np.float64)
        metrics = {
            "mean_r2": float(ds.attrs.get("sierra_area_weighted_mean_r2", np.nan)),
            "min_r2": float(ds.attrs.get("sierra_min_local_r2", np.nan)),
            "max_r2": float(ds.attrs.get("sierra_max_local_r2", np.nan)),
            "median_r2": float(ds.attrs.get("sierra_median_local_r2", np.nan)),
        }
    return sierra_latitude, sierra_longitude, coefficient_maps, r2_map, metrics


def diagnostic_text_box() -> str:
    return "\n".join(
        [
            r"Sensitivity test: joint Level 2 regression with $A=[\mathrm{PC1},\mathrm{PC2},\mathrm{PC3},\mathrm{PC4},\mathrm{PC5},\mathrm{PC6}]$.",
            r"$\hat{B}=(A^T A)^{-1}A^TY$ and $\hat{Y}=A\hat{B}$.",
            r"Each $\hat{B}_k(r)$ map is the local matched-region T2m response to a one-unit increase in PC$k$, after accounting for the other five PCs.",
            r"PCs are standardized here, so each coefficient is in K per one-standard-deviation increase in the corresponding PC.",
            r"$R^2(r)=1-\frac{\sum_t [Y(t,r)-\hat{Y}(t,r)]^2}{\sum_t [Y(t,r)]^2}$ gives the fraction of local matched-region T2m anomaly variance explained by PC1-PC6 together.",
        ]
    )


def plot_maps(
    sierra_latitude: np.ndarray,
    sierra_longitude: np.ndarray,
    coefficient_maps: np.ndarray,
    r2_map: np.ndarray,
    summary: SummaryRow,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.20, hspace=0.38, wspace=0.28)
    lon2d, lat2d = np.meshgrid(sierra_longitude, sierra_latitude)

    for mode_index in range(N_PREDICTORS):
        ax = axes.flat[mode_index]
        coeff = np.asarray(coefficient_maps[mode_index], dtype=np.float64)
        vmax = float(np.nanmax(np.abs(coeff)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            coeff,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(rf"PC{mode_index + 1} coefficient $\hat{{B}}_{{{mode_index + 1}}}(r)$")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
        ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.84).set_label(r"$\hat{B}_k(r)$ [K / 1$\sigma$ PC]")

    r2_ax = axes[2, 2]
    r2_vmax = max(0.05, float(np.nanmax(r2_map)))
    r2_mesh = r2_ax.pcolormesh(
        lon2d,
        lat2d,
        r2_map,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=r2_vmax,
    )
    r2_ax.set_title(
        rf"Joint explained variance $R^2(r)$ | "
        rf"$\overline{{R^2}}={summary.sierra_area_weighted_mean_r2:.3f}$"
    )
    r2_ax.set_xlabel("Longitude")
    r2_ax.set_ylabel("Latitude")
    r2_ax.set_xlim(SIERRA_T2M_REGION_360.lon_min, SIERRA_T2M_REGION_360.lon_max)
    r2_ax.set_ylim(SIERRA_T2M_REGION_360.lat_min, SIERRA_T2M_REGION_360.lat_max)
    r2_ax.set_aspect("equal", adjustable="box")
    fig.colorbar(r2_mesh, ax=r2_ax, shrink=0.84).set_label(r"$R^2(r)$")

    fig.text(
        0.5,
        0.10,
        diagnostic_text_box(),
        ha="center",
        va="center",
        fontsize=9.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.75", "facecolor": "#f7f7f7", "edgecolor": "#666666"},
    )
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    plot_only = "--plot-only" in sys.argv[1:]
    ensure_output_dir()
    remove_if_exists(FIGURE_FILE)

    if plot_only:
        if not NETCDF_FILE.exists():
            raise FileNotFoundError(f"Plot-only mode requires existing NetCDF: {NETCDF_FILE}")
        sierra_latitude, sierra_longitude, coefficient_maps, r2_map, metrics = load_plot_inputs_from_netcdf()
        summary = SummaryRow(
            model_name=MODEL_NAME,
            n_pcs=N_PREDICTORS,
            overlap_start="",
            overlap_end="",
            n_overlap_months=0,
            sierra_area_weighted_mean_r2=metrics["mean_r2"],
            sierra_max_local_r2=metrics["max_r2"],
            sierra_min_local_r2=metrics["min_r2"],
            sierra_median_local_r2=metrics["median_r2"],
            pc1_mean_raw=float("nan"),
            pc2_mean_raw=float("nan"),
            pc3_mean_raw=float("nan"),
            pc4_mean_raw=float("nan"),
            pc5_mean_raw=float("nan"),
            pc6_mean_raw=float("nan"),
            pc1_std_raw=float("nan"),
            pc2_std_raw=float("nan"),
            pc3_std_raw=float("nan"),
            pc4_std_raw=float("nan"),
            pc5_std_raw=float("nan"),
            pc6_std_raw=float("nan"),
        )
        plot_maps(
            sierra_latitude=sierra_latitude,
            sierra_longitude=sierra_longitude,
            coefficient_maps=coefficient_maps,
            r2_map=r2_map,
            summary=summary,
        )
        print(f"Figure: {FIGURE_FILE}", flush=True)
        return

    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    remove_if_exists(NETCDF_FILE)
    remove_if_exists(SUMMARY_CSV_FILE)
    remove_if_exists(SUMMARY_JSON_FILE)

    pacific = load_pacific_cobe2_pca(PACIFIC_SST_REGION_360)
    monthly_mean_ds = xr.open_dataset(
        ERA5_MONTHLY_MEAN_FILE,
        engine="netcdf4",
        chunks={"time": TIME_CHUNK, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )
    monthly_clim_ds = xr.open_dataset(
        ERA5_MONTHLY_CLIM_FILE,
        engine="netcdf4",
        chunks={"month": 12, "latitude": LAT_CHUNK, "longitude": LON_CHUNK},
        decode_times=True,
    )

    try:
        monthly_mean = subset_era5_region_360(monthly_mean_ds[ERA5_VARIABLE], SIERRA_T2M_REGION_360)
        monthly_clim = subset_era5_region_360(monthly_clim_ds[ERA5_VARIABLE], SIERRA_T2M_REGION_360)
        era5_time = to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        overlap_months = np.intersect1d(pacific["time"], era5_time, assume_unique=False)
        if overlap_months.size == 0:
            raise ValueError("No overlapping months between Pacific COBE2 PC time and matched-region ERA5 time")

        pacific_index = build_time_index(pacific["time"])
        era5_index = build_time_index(era5_time)
        predictor_matrix_raw = np.stack(
            [pacific["pc"][pacific_index[month], :N_PREDICTORS] for month in overlap_months.tolist()],
            axis=0,
        )
        predictor_matrix, predictor_matrix_raw_mean, predictor_matrix_raw_std = standardize_pc_matrix(predictor_matrix_raw)

        sierra_latitude = np.asarray(monthly_mean["latitude"].values, dtype=np.float64)
        sierra_longitude = np.asarray(monthly_mean["longitude"].values, dtype=np.float64)
        anomaly_slices: List[np.ndarray] = []

        print(
            "Computing matched-region ERA5-Land T2m Level 2 PC1-PC6 sensitivity regression for overlap "
            f"{format_month(overlap_months[0])} to {format_month(overlap_months[-1])} "
            f"({int(overlap_months.size)} months)",
            flush=True,
        )
        for step_index, month_value in enumerate(overlap_months.tolist(), start=1):
            era5_time_index = era5_index[month_value]
            monthly_mean_slice = monthly_mean.isel(time=era5_time_index)
            monthly_clim_slice = monthly_clim.sel(month=month_number(month_value))
            anomaly_slice = (monthly_mean_slice - monthly_clim_slice).astype(np.float64).load().values
            anomaly_slices.append(anomaly_slice)
            if step_index == 1 or step_index % 120 == 0 or step_index == overlap_months.size:
                print(
                    f"  processed overlap month {step_index}/{int(overlap_months.size)}: {format_month(month_value)}",
                    flush=True,
                )

        anomalies = np.stack(anomaly_slices, axis=0)
        coefficient_maps, r2_map = fit_multivariate_regression(
            predictors=predictor_matrix,
            anomalies=anomalies,
        )
        summary = summarize_r2(
            r2_map=r2_map,
            latitude=sierra_latitude,
            overlap_months=overlap_months,
            predictor_matrix_raw_mean=predictor_matrix_raw_mean,
            predictor_matrix_raw_std=predictor_matrix_raw_std,
            predictor_matrix=predictor_matrix,
        )
        save_netcdf(
            pacific=pacific,
            overlap_months=overlap_months,
            predictor_matrix=predictor_matrix,
            predictor_matrix_raw_mean=predictor_matrix_raw_mean,
            predictor_matrix_raw_std=predictor_matrix_raw_std,
            sierra_latitude=sierra_latitude,
            sierra_longitude=sierra_longitude,
            coefficient_maps=coefficient_maps,
            r2_map=r2_map,
            summary=summary,
            runtime=runtime,
        )
        save_summary(
            summary=summary,
            pacific=pacific,
            sierra_shape=[int(sierra_latitude.size), int(sierra_longitude.size)],
            runtime=runtime,
        )
        plot_maps(
            sierra_latitude=sierra_latitude,
            sierra_longitude=sierra_longitude,
            coefficient_maps=coefficient_maps,
            r2_map=r2_map,
            summary=summary,
        )
    finally:
        monthly_mean_ds.close()
        monthly_clim_ds.close()

    print(f"Pscratch output directory: {PSCRATCH_OUTPUT_DIR}", flush=True)
    print(f"Home output directory: {HOME_OUTPUT_DIR}", flush=True)
    print(f"NetCDF: {NETCDF_FILE}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV_FILE}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON_FILE}", flush=True)
    print(f"Figure: {FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
