#!/usr/bin/env python3
"""
LOYO diagnostics for patch-mean SST predictors around the fixed full-sample
COBE2 LOD-selected mode-1 and mode-2 reference points.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
TARGET_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
)
FULL37_MODES_CSV = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_modes.csv"
)
FULL37_SUMMARY_JSON = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json"
)
COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo"
)
PATCH_DEFS = {
    "exact_grid_cell": 0.0,
    "5deg": 2.5,
    "10deg": 5.0,
    "15deg": 7.5,
}
MODEL_SPECS = [
    ("Z1_only", [1]),
    ("Z2_only", [2]),
    ("Z1_Z2", [1, 2]),
]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def month_targets() -> list[np.datetime64]:
    times: list[np.datetime64] = []
    for water_year in WATER_YEARS:
        for _, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
    return times


def lag_name_to_index() -> dict[str, int]:
    return {lag_name: idx for idx, (lag_name, _, _) in enumerate(LAG_SPECS)}


def load_target() -> np.ndarray:
    with xr.open_dataset(TARGET_FILE, engine=NETCDF_ENGINE) as ds:
        ds = ds.sel(water_year=WATER_YEARS).load()
        return np.asarray(ds["sierra_swe_apr1_anom_m"].values, dtype=np.float64)


def load_full37_reference_modes() -> dict[int, dict[str, object]]:
    rows: list[dict[str, object]] = []
    if FULL37_MODES_CSV.exists():
        with FULL37_MODES_CSV.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["selected"] != "True":
                    continue
                mode = int(row["mode_number"])
                if mode <= 2:
                    rows.append(
                        {
                            "mode_rank": mode,
                            "lag_month": row["lag_month"],
                            "lat": float(row["latitude"]),
                            "lon": float(row["longitude"]),
                        }
                    )
    else:
        payload = json.loads(FULL37_SUMMARY_JSON.read_text(encoding="utf-8"))
        for row in payload["lod_rows"]:
            if not row.get("selected"):
                continue
            mode = int(row["mode_number"])
            if mode <= 2:
                rows.append(
                    {
                        "mode_rank": mode,
                        "lag_month": str(row["lag_month"]),
                        "lat": float(row["latitude"]),
                        "lon": float(row.get("longitude_0_360", row.get("longitude"))),
                    }
                )
    rows.sort(key=lambda item: int(item["mode_rank"]))
    if [int(item["mode_rank"]) for item in rows] != [1, 2]:
        raise ValueError("Could not recover full-37 selected modes 1 and 2 from saved artifacts.")
    return {int(item["mode_rank"]): item for item in rows}


def load_cobe2_monthly_means() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_times = month_targets()
    with xr.open_dataset(COBE2_SST_FILE, engine=NETCDF_ENGINE) as ds:
        sst = ds["sst"].sel(
            time=selected_times,
            lat=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            lon=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        ).load()
        values = np.asarray(sst.values, dtype=np.float64).reshape(
            len(WATER_YEARS),
            len(LAG_SPECS),
            sst.sizes["lat"],
            sst.sizes["lon"],
        )
        latitude = np.asarray(sst["lat"].values, dtype=np.float64)
        longitude = np.asarray(sst["lon"].values, dtype=np.float64)
    return values, latitude, longitude


def find_coordinate_index(values: np.ndarray, target: float, coord_name: str) -> int:
    idx = int(np.argmin(np.abs(values - target)))
    candidate = float(values[idx])
    if not np.isclose(candidate, float(target), atol=1.0e-6):
        raise ValueError(f"Could not match {coord_name}={target}; nearest value is {candidate}")
    return idx


def build_patch_metadata(
    references: dict[int, dict[str, object]],
    latitude: np.ndarray,
    longitude: np.ndarray,
    sst_values: np.ndarray,
) -> dict[str, dict[int, dict[str, object]]]:
    lag_lookup = lag_name_to_index()
    metadata: dict[str, dict[int, dict[str, object]]] = {}
    for patch_name, half_width_deg in PATCH_DEFS.items():
        metadata[patch_name] = {}
        for mode_rank, ref in references.items():
            center_lat = float(ref["lat"])
            center_lon = float(ref["lon"])
            lag_idx = lag_lookup[str(ref["lag_month"])]
            if patch_name == "exact_grid_cell":
                lat_mask = np.zeros(latitude.shape, dtype=bool)
                lon_mask = np.zeros(longitude.shape, dtype=bool)
                lat_mask[find_coordinate_index(latitude, center_lat, "latitude")] = True
                lon_mask[find_coordinate_index(longitude, center_lon, "longitude")] = True
            else:
                lat_mask = np.abs(latitude - center_lat) <= half_width_deg
                lon_mask = np.abs(longitude - center_lon) <= half_width_deg
            patch_mask = lat_mask[:, None] & lon_mask[None, :]
            valid_any = np.any(np.isfinite(sst_values[:, lag_idx, :, :]) & patch_mask[None, :, :], axis=0)
            n_cells = int(np.count_nonzero(valid_any))
            if n_cells == 0:
                raise ValueError(
                    f"Patch {patch_name} for mode {mode_rank} centered at ({center_lat}, {center_lon}) has zero valid ocean cells."
                )
            metadata[patch_name][mode_rank] = {
                "mode_rank": mode_rank,
                "lag_month": ref["lag_month"],
                "lag_index": lag_idx,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "patch_mask": patch_mask,
                "lat_mask": lat_mask,
                "lon_mask": lon_mask,
                "n_cells": n_cells,
            }
    return metadata


def area_weighted_patch_mean(field2d: np.ndarray, latitudes: np.ndarray, patch_mask: np.ndarray) -> tuple[float, int]:
    lat_weights = np.cos(np.deg2rad(latitudes))[:, None]
    valid = patch_mask & np.isfinite(field2d)
    n_cells = int(np.count_nonzero(valid))
    if n_cells == 0:
        return float("nan"), 0
    weights = np.where(valid, lat_weights, 0.0)
    numerator = float(np.nansum(field2d * weights))
    denominator = float(np.nansum(weights))
    if denominator <= 0.0:
        return float("nan"), 0
    return numerator / denominator, n_cells


def compute_fullsample_patch_predictors(
    sst_values: np.ndarray,
    latitude: np.ndarray,
    patch_metadata: dict[str, dict[int, dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[str, dict[int, np.ndarray]]]:
    monthly_clim = np.nanmean(sst_values, axis=0, dtype=np.float64)
    anomalies = sst_values - monthly_clim[None, :, :, :]
    rows: list[dict[str, object]] = []
    timeseries: dict[str, dict[int, np.ndarray]] = {}
    for patch_name, patch_modes in patch_metadata.items():
        timeseries[patch_name] = {}
        values_by_mode: dict[int, np.ndarray] = {}
        counts_by_mode: dict[int, np.ndarray] = {}
        for mode_rank, meta in patch_modes.items():
            lag_idx = int(meta["lag_index"])
            vals = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)
            counts = np.full(WATER_YEARS.shape, 0, dtype=np.int32)
            for idx in range(WATER_YEARS.size):
                val, n_cells = area_weighted_patch_mean(anomalies[idx, lag_idx, :, :], latitude, meta["patch_mask"])
                vals[idx] = val
                counts[idx] = n_cells
            values_by_mode[mode_rank] = vals
            counts_by_mode[mode_rank] = counts
            timeseries[patch_name][mode_rank] = vals
        for idx, water_year in enumerate(WATER_YEARS.tolist()):
            rows.append(
                {
                    "water_year": water_year,
                    "patch_size": patch_name,
                    "Z1_M1_Jan_lat_-9.5_lon_133.5": float(values_by_mode[1][idx]),
                    "Z2_M2_Oct_lat_0.5_lon_136.5": float(values_by_mode[2][idx]),
                    "n_cells_Z1_patch": int(counts_by_mode[1][idx]),
                    "n_cells_Z2_patch": int(counts_by_mode[2][idx]),
                }
            )
    return rows, timeseries


def standardize_predictors_train_only(
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("One or more predictor columns have non-positive train-fold standard deviation.")
    x_train_std = (x_train_raw - x_mean[None, :]) / x_std[None, :]
    x_test_std = (x_test_raw - x_mean) / x_std
    return x_train_std, x_test_std, x_mean, x_std


def fit_ols_with_intercept(x_train_std: np.ndarray, y_train_raw: np.ndarray, x_test_std: np.ndarray) -> tuple[float, float, np.ndarray]:
    design_train = np.column_stack([np.ones(x_train_std.shape[0], dtype=np.float64), x_train_std])
    coef, *_ = np.linalg.lstsq(design_train, y_train_raw, rcond=None)
    intercept = float(coef[0])
    betas = np.asarray(coef[1:], dtype=np.float64)
    pred = float(intercept + x_test_std @ betas)
    return pred, intercept, betas


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    xx = x[finite]
    yy = y[finite]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def plot_observed_vs_predicted(path: Path, water_years: np.ndarray, observed: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
    ax.plot(water_years, observed, color="black", linewidth=1.8, label="Observed SWE anomaly")
    color_map = {
        "exact_grid_cell": "tab:blue",
        "5deg": "tab:orange",
        "10deg": "tab:green",
        "15deg": "tab:red",
    }
    for patch_name, values in predictions.items():
        ax.plot(water_years, values, linewidth=1.15, label=f"{patch_name} / Z1_Z2", color=color_map[patch_name])
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Full-37 selected patch predictors under LOYO")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metrics_barplot(path: Path, metrics_rows: list[dict[str, object]]) -> None:
    labels = [f"{row['patch_size']}\n{row['model_name']}" for row in metrics_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 10.5), constrained_layout=True)
    for ax, field in zip(axes, ("R2", "RMSE", "sign_accuracy")):
        values = [float(row[field]) for row in metrics_rows]
        ax.bar(x, values, color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(field)
        ax.grid(True, axis="y", linewidth=0.25, color="0.85")
        if field in {"R2", "sign_accuracy"}:
            ax.axhline(0.0 if field == "R2" else 0.5, color="0.5", linewidth=0.8, linestyle="--")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_predictor_timeseries(
    path: Path,
    water_years: np.ndarray,
    timeseries: dict[str, dict[int, np.ndarray]],
    observed_swe: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.5), constrained_layout=True)
    color_map = {
        "exact_grid_cell": "tab:blue",
        "5deg": "tab:orange",
        "10deg": "tab:green",
        "15deg": "tab:red",
    }
    for ax, mode_rank, title in zip(
        axes,
        (1, 2),
        ("Z1 patch predictor around full-37 M1 (Jan, -9.5, 133.5)", "Z2 patch predictor around full-37 M2 (Oct, 0.5, 136.5)"),
    ):
        swe_ax = ax.twinx()
        swe_ax.plot(water_years, observed_swe, label="Observed SWE anomaly", linewidth=1.8, color="black")
        for patch_name, by_mode in timeseries.items():
            ax.plot(water_years, by_mode[mode_rank], label=patch_name, linewidth=1.2, color=color_map[patch_name])
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Patch-mean SST anomaly")
        swe_ax.set_ylabel("Observed SWE anomaly (m)")
        ax.grid(True, linewidth=0.25, color="0.85")
        sst_handles, sst_labels = ax.get_legend_handles_labels()
        swe_handles, swe_labels = swe_ax.get_legend_handles_labels()
        ax.legend(swe_handles + sst_handles, swe_labels + sst_labels, frameon=False, ncol=2)
    axes[-1].set_xlabel("Water year")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def short_conclusion(best_r2: dict[str, object], exact_r2: float, patch5_r2: float, patch10_r2: float, patch15_r2: float) -> str:
    if max(patch5_r2, patch10_r2, patch15_r2) > exact_r2:
        return "Patch Z1_Z2 improves over the exact grid cell, so regional averaging stabilizes the full-37 selected signal."
    if exact_r2 >= max(patch5_r2, patch10_r2, patch15_r2):
        return "The exact grid cell is best, so the selected signal appears localized or brittle to patch averaging."
    if float(best_r2["R2"]) < 0.0:
        return "All patch sizes remain poor, so the full-37 selected signal does not generalize even after regional averaging."
    return "Patch-size effects are mixed; inspect the metric table and time series."


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    target = load_target()
    references = load_full37_reference_modes()
    sst_values, latitude, longitude = load_cobe2_monthly_means()
    patch_metadata = build_patch_metadata(references, latitude, longitude, sst_values)
    predictor_rows, fullsample_timeseries = compute_fullsample_patch_predictors(sst_values, latitude, patch_metadata)

    predictions_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    z1z2_predictions_for_plot: dict[str, np.ndarray] = {}

    for patch_name, patch_modes in patch_metadata.items():
        predictor_by_mode_trainfold = {1: np.full(WATER_YEARS.shape, np.nan, dtype=np.float64), 2: np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)}
        predicted_by_model: dict[str, np.ndarray] = {}
        for model_name, used_modes in MODEL_SPECS:
            preds = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)
            for fold_index, heldout_wy in enumerate(WATER_YEARS.tolist()):
                test_mask = WATER_YEARS == heldout_wy
                train_mask = ~test_mask
                train_sst = sst_values[train_mask]
                test_sst = sst_values[test_mask][0]
                train_target = target[train_mask]
                test_target = float(target[test_mask][0])
                monthly_clim = np.nanmean(train_sst, axis=0, dtype=np.float64)
                train_anom = train_sst - monthly_clim[None, :, :, :]
                test_anom = test_sst - monthly_clim

                train_columns: list[np.ndarray] = []
                test_columns: list[float] = []
                for mode_rank in used_modes:
                    meta = patch_modes[mode_rank]
                    lag_idx = int(meta["lag_index"])
                    train_vals = np.full(train_target.shape, np.nan, dtype=np.float64)
                    for train_idx in range(train_target.size):
                        train_vals[train_idx], _ = area_weighted_patch_mean(
                            train_anom[train_idx, lag_idx, :, :],
                            latitude,
                            meta["patch_mask"],
                        )
                    test_val, _ = area_weighted_patch_mean(test_anom[lag_idx, :, :], latitude, meta["patch_mask"])
                    if np.any(~np.isfinite(train_vals)) or not np.isfinite(test_val):
                        raise ValueError(f"Non-finite patch predictor in patch={patch_name}, mode={mode_rank}, heldout={heldout_wy}")
                    train_columns.append(train_vals)
                    test_columns.append(float(test_val))
                    predictor_by_mode_trainfold[mode_rank][fold_index] = float(test_val)

                x_train_raw = np.column_stack(train_columns).astype(np.float64)
                x_test_raw = np.asarray(test_columns, dtype=np.float64)
                x_train_std, x_test_std, _, _ = standardize_predictors_train_only(x_train_raw, x_test_raw)
                pred, intercept, betas = fit_ols_with_intercept(x_train_std, train_target, x_test_std)
                preds[fold_index] = pred

                beta_z1 = float("nan")
                beta_z2 = float("nan")
                if 1 in used_modes:
                    beta_z1 = float(betas[used_modes.index(1)])
                if 2 in used_modes:
                    beta_z2 = float(betas[used_modes.index(2)])
                sign_correct = float("nan")
                if test_target != 0.0 and pred != 0.0:
                    sign_correct = 1.0 if np.sign(test_target) == np.sign(pred) else 0.0
                predictions_rows.append(
                    {
                        "patch_size": patch_name,
                        "model_name": model_name,
                        "heldout_wy": heldout_wy,
                        "obs_swe": test_target,
                        "pred_swe": pred,
                        "error": float(pred - test_target),
                        "abs_error": float(abs(pred - test_target)),
                        "sign_correct": sign_correct,
                    }
                )
                beta_rows.append(
                    {
                        "patch_size": patch_name,
                        "model_name": model_name,
                        "heldout_wy": heldout_wy,
                        "intercept": intercept,
                        "beta_Z1": beta_z1,
                        "beta_Z2": beta_z2,
                        "sign_beta_Z1": np.nan if np.isnan(beta_z1) else int(np.sign(beta_z1)),
                        "sign_beta_Z2": np.nan if np.isnan(beta_z2) else int(np.sign(beta_z2)),
                    }
                )

            predicted_by_model[model_name] = preds
            metrics_rows.append(
                {
                    "patch_size": patch_name,
                    "model_name": model_name,
                    "num_predictors": len(used_modes),
                    "r": corrcoef_safe(target, preds),
                    "R2": r2_manual(target, preds),
                    "RMSE": rmse(target, preds),
                    "MAE": mae(target, preds),
                    "sign_accuracy": compute_sign_accuracy(target, preds),
                }
            )
        z1z2_predictions_for_plot[patch_name] = predicted_by_model["Z1_Z2"]

    predictors_csv = OUTPUT_ROOT / "full37_patch_predictors.csv"
    predictions_csv = OUTPUT_ROOT / "full37_patch_loyo_predictions.csv"
    metrics_csv = OUTPUT_ROOT / "full37_patch_loyo_metrics.csv"
    beta_csv = OUTPUT_ROOT / "full37_patch_beta_by_fold.csv"
    summary_json = OUTPUT_ROOT / "full37_patch_predictor_summary.json"
    observed_vs_pred_png = OUTPUT_ROOT / "full37_patch_observed_vs_predicted.png"
    metrics_barplot_png = OUTPUT_ROOT / "full37_patch_metrics_barplot.png"
    predictor_timeseries_png = OUTPUT_ROOT / "full37_patch_predictor_timeseries.png"

    write_csv(
        predictors_csv,
        ["water_year", "patch_size", "Z1_M1_Jan_lat_-9.5_lon_133.5", "Z2_M2_Oct_lat_0.5_lon_136.5", "n_cells_Z1_patch", "n_cells_Z2_patch"],
        predictor_rows,
    )
    write_csv(
        predictions_csv,
        ["patch_size", "model_name", "heldout_wy", "obs_swe", "pred_swe", "error", "abs_error", "sign_correct"],
        predictions_rows,
    )
    write_csv(
        metrics_csv,
        ["patch_size", "model_name", "num_predictors", "r", "R2", "RMSE", "MAE", "sign_accuracy"],
        metrics_rows,
    )
    write_csv(
        beta_csv,
        ["patch_size", "model_name", "heldout_wy", "intercept", "beta_Z1", "beta_Z2", "sign_beta_Z1", "sign_beta_Z2"],
        beta_rows,
    )

    plot_observed_vs_predicted(observed_vs_pred_png, WATER_YEARS, target, z1z2_predictions_for_plot)
    plot_metrics_barplot(metrics_barplot_png, metrics_rows)
    plot_predictor_timeseries(predictor_timeseries_png, WATER_YEARS, fullsample_timeseries, target)

    best_r2 = max(metrics_rows, key=lambda row: float(row["R2"]))
    best_rmse = min(metrics_rows, key=lambda row: float(row["RMSE"]))
    best_sign = max(metrics_rows, key=lambda row: float(row["sign_accuracy"]))
    exact_r2 = next(float(row["R2"]) for row in metrics_rows if row["patch_size"] == "exact_grid_cell" and row["model_name"] == "Z1_Z2")
    patch5_r2 = next(float(row["R2"]) for row in metrics_rows if row["patch_size"] == "5deg" and row["model_name"] == "Z1_Z2")
    patch10_r2 = next(float(row["R2"]) for row in metrics_rows if row["patch_size"] == "10deg" and row["model_name"] == "Z1_Z2")
    patch15_r2 = next(float(row["R2"]) for row in metrics_rows if row["patch_size"] == "15deg" and row["model_name"] == "Z1_Z2")
    cell_counts = {
        patch_name: {
            "mode1_n_cells": int(patch_modes[1]["n_cells"]),
            "mode2_n_cells": int(patch_modes[2]["n_cells"]),
        }
        for patch_name, patch_modes in patch_metadata.items()
    }
    summary = {
        "reference_mode_1": references[1],
        "reference_mode_2": references[2],
        "patch_definitions": {
            "exact_grid_cell": "single selected COBE2 grid cell",
            "5deg": "lat/lon half-width 2.5 degrees",
            "10deg": "lat/lon half-width 5 degrees",
            "15deg": "lat/lon half-width 7.5 degrees",
        },
        "patch_cell_counts": cell_counts,
        "water_years": [int(value) for value in WATER_YEARS.tolist()],
        "target_source": str(TARGET_FILE),
        "sst_source": str(COBE2_SST_FILE),
        "full37_reference_source": str(FULL37_MODES_CSV if FULL37_MODES_CSV.exists() else FULL37_SUMMARY_JSON),
        "sst_anomaly_convention_for_loyo": "Train-fold monthly climatology is computed from the 36 training years only, then applied to both train and held-out year.",
        "predictor_timeseries_csv_convention": "full37_patch_predictors.csv uses full-sample monthly climatology only for descriptive patch time series.",
        "train_fold_standardization": "predictors standardized using train years only",
        "intercept_handling": {"intercept": True, "standardization": "train-fold only"},
        "metrics_by_patch_and_model": metrics_rows,
        "best_model_by_R2": best_r2,
        "best_model_by_RMSE": best_rmse,
        "best_model_by_sign_accuracy": best_sign,
        "short_conclusion": short_conclusion(best_r2, exact_r2, patch5_r2, patch10_r2, patch15_r2),
        "outputs": {
            "output_directory": str(OUTPUT_ROOT),
            "full37_patch_predictors_csv": str(predictors_csv),
            "full37_patch_loyo_predictions_csv": str(predictions_csv),
            "full37_patch_loyo_metrics_csv": str(metrics_csv),
            "full37_patch_beta_by_fold_csv": str(beta_csv),
            "full37_patch_predictor_summary_json": str(summary_json),
            "full37_patch_observed_vs_predicted_png": str(observed_vs_pred_png),
            "full37_patch_metrics_barplot_png": str(metrics_barplot_png),
            "full37_patch_predictor_timeseries_png": str(predictor_timeseries_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {OUTPUT_ROOT}")
    print(
        "Reference M1: "
        f"{references[1]['lag_month']}, lat={references[1]['lat']}, lon={references[1]['lon']}"
    )
    print(
        "Reference M2: "
        f"{references[2]['lag_month']}, lat={references[2]['lat']}, lon={references[2]['lon']}"
    )
    print("Patch cell counts:")
    for patch_name, counts in cell_counts.items():
        print(f"  {patch_name}: Z1={counts['mode1_n_cells']}, Z2={counts['mode2_n_cells']}")
    print("Metrics table:")
    for row in metrics_rows:
        print(
            f"  {row['patch_size']} / {row['model_name']}: "
            f"R2={float(row['R2']):.4f}, RMSE={float(row['RMSE']):.4f}, "
            f"MAE={float(row['MAE']):.4f}, sign_accuracy={float(row['sign_accuracy']):.4f}"
        )
    print(f"Best patch/model by R2: {best_r2['patch_size']} / {best_r2['model_name']}")
    print(f"Best patch/model by RMSE: {best_rmse['patch_size']} / {best_rmse['model_name']}")
    print(f"Best patch/model by sign accuracy: {best_sign['patch_size']} / {best_sign['model_name']}")
    print(f"Short conclusion: {summary['short_conclusion']}")


if __name__ == "__main__":
    main()
