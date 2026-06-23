#!/usr/bin/env python3
"""
Run a fixed-selected-LOD-pairs LOYO regression sanity check.

This script uses the position/month pairs selected by the full-sample LOD
diagnostic, keeps those pairs fixed across all outer LOYO folds, and compares:

1. OLS on the fixed selected anomaly columns.
2. A direct fixed-pair LOD-style reconstruction using the same fixed order.
"""

import argparse
import csv
import json
import math
import os
import resource
import sys
import warnings
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "artifacts" / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


class DatasetConfig(object):
    def __init__(self, key, label, summary_file, output_root, cobe2_sst_file=None, era5_predictor_file=None):
        self.key = key
        self.label = label
        self.summary_file = summary_file
        self.output_root = output_root
        self.cobe2_sst_file = cobe2_sst_file
        self.era5_predictor_file = era5_predictor_file


DATASET_CONFIGS = {
    "cobe2": DatasetConfig(
        key="cobe2",
        label="COBE2",
        summary_file=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json"
        ),
        output_root=PROJECT_ROOT / "artifacts" / "fixed_lod_pair_ols_loyo_check" / "cobe2",
        cobe2_sst_file=Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc"),
    ),
    "era5": DatasetConfig(
        key="era5",
        label="ERA5",
        summary_file=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/lod_analysis/era5_sierra_swe_lod_summary.json"
        ),
        output_root=PROJECT_ROOT / "artifacts" / "fixed_lod_pair_ols_loyo_check" / "era5",
        era5_predictor_file=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/predictors/"
            "era5_pacific_sst_monthly_anomaly_wy1985_2021_sep1984_mar2021.nc"
        ),
    ),
}


def peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def month_targets():
    times = []
    for water_year in WATER_YEARS:
        for _, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
    return times


def lag_name_to_index():
    return {lag_name: idx for idx, (lag_name, _, _) in enumerate(LAG_SPECS)}


def load_target():
    with xr.open_dataset(TARGET_FILE, engine=NETCDF_ENGINE) as ds:
        ds = ds.sel(water_year=WATER_YEARS).load()
        target_anom_m = np.asarray(ds["sierra_swe_apr1_anom_m"].values, dtype=np.float64)
        target_std = np.asarray(ds["sierra_swe_apr1_standardized"].values, dtype=np.float64)
    return target_anom_m, target_std


def load_selected_pairs(summary_file):
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing LOD summary file: {summary_file}")
    summary = json.loads(summary_file.read_text())
    selected = [row for row in summary["lod_rows"] if row.get("selected")]
    selected.sort(key=lambda row: int(row["mode_number"]))
    if not selected:
        raise ValueError(f"No selected LOD rows found in {summary_file}")
    return selected


def load_cobe2_monthly_means(config):
    assert config.cobe2_sst_file is not None
    selected_times = month_targets()
    with xr.open_dataset(config.cobe2_sst_file, engine=NETCDF_ENGINE) as ds:
        sst = ds["sst"].sel(
            time=selected_times,
            lat=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            lon=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        ).load()
        values = np.asarray(sst.values, dtype=np.float32).reshape(
            len(WATER_YEARS),
            len(LAG_SPECS),
            sst.sizes["lat"],
            sst.sizes["lon"],
        )
        latitude = np.asarray(sst["lat"].values, dtype=np.float32)
        longitude = np.asarray(sst["lon"].values, dtype=np.float32)
    return values, latitude, longitude


def load_era5_monthly_means(config):
    assert config.era5_predictor_file is not None
    if not config.era5_predictor_file.exists():
        raise FileNotFoundError(f"Missing ERA5 predictor file: {config.era5_predictor_file}")
    with xr.open_dataset(config.era5_predictor_file, engine=NETCDF_ENGINE) as ds:
        if "sst_monthly_mean" not in ds or "latitude" not in ds or "longitude" not in ds:
            raise ValueError(f"ERA5 predictor file is missing expected variables: {config.era5_predictor_file}")
        monthly_mean = np.asarray(ds["sst_monthly_mean"].values, dtype=np.float32)
        latitude = np.asarray(ds["latitude"].values, dtype=np.float32)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float32)
    values = monthly_mean.reshape(len(WATER_YEARS), len(LAG_SPECS), latitude.size, longitude.size)
    return values, latitude, longitude


def load_monthly_means(config):
    if config.key == "cobe2":
        return load_cobe2_monthly_means(config)
    if config.key == "era5":
        return load_era5_monthly_means(config)
    raise ValueError(f"Unsupported dataset key: {config.key}")


def find_coordinate_index(values, target, coord_name):
    idx = int(np.argmin(np.abs(values.astype(np.float64) - float(target))))
    candidate = float(values[idx])
    if not np.isclose(candidate, float(target), atol=1.0e-6):
        raise ValueError(f"Could not match {coord_name}={target} in coordinate array; nearest value is {candidate}")
    return idx


def build_selected_pair_metadata(selected_pairs, latitude, longitude):
    lag_lookup = lag_name_to_index()
    metadata = []
    for pair in selected_pairs:
        lag_name = str(pair["lag_month"])
        lat_value = float(pair["latitude"])
        lon_value = float(pair.get("longitude_0_360", pair.get("longitude")))
        metadata.append(
            {
                "mode_index": int(pair["mode_number"]),
                "lag_month": lag_name,
                "lag_index": lag_lookup[lag_name],
                "latitude": lat_value,
                "longitude_0_360": lon_value,
                "latitude_index": find_coordinate_index(latitude, lat_value, "latitude"),
                "longitude_index": find_coordinate_index(longitude, lon_value, "longitude"),
                "candidate_index": int(pair.get("candidate_index", pair.get("ocean_candidate_q", -1))),
            }
        )
    return metadata


def standardize_target(y_train_raw):
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Training target standard deviation is not positive.")
    return (y_train_raw - y_mean) / y_std, y_mean, y_std


def build_fold_selected_pair_matrix(train_sst, test_sst, pair_metadata):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        monthly_clim = np.nanmean(train_sst, axis=0, dtype=np.float64)
    train_anom = train_sst.astype(np.float64) - monthly_clim[None, :, :, :]
    test_anom = test_sst.astype(np.float64) - monthly_clim

    raw_train_columns = []
    raw_test_values = []
    for pair in pair_metadata:
        lag_idx = int(pair["lag_index"])
        lat_idx = int(pair["latitude_index"])
        lon_idx = int(pair["longitude_index"])
        raw_train_columns.append(train_anom[:, lag_idx, lat_idx, lon_idx])
        raw_test_values.append(float(test_anom[lag_idx, lat_idx, lon_idx]))

    X_train_raw = np.column_stack(raw_train_columns).astype(np.float64)
    x_test_raw = np.asarray(raw_test_values, dtype=np.float64)
    X_mean = np.mean(X_train_raw, axis=0)
    X_std = np.std(X_train_raw, axis=0, ddof=1)
    valid_cols = np.isfinite(X_std) & (X_std > 0.0) & np.all(np.isfinite(X_train_raw), axis=0) & np.isfinite(x_test_raw)
    X_train_std = np.full_like(X_train_raw, np.nan, dtype=np.float64)
    x_test_std = np.full_like(x_test_raw, np.nan, dtype=np.float64)
    X_train_std[:, valid_cols] = (X_train_raw[:, valid_cols] - X_mean[valid_cols]) / X_std[valid_cols]
    x_test_std[valid_cols] = (x_test_raw[valid_cols] - X_mean[valid_cols]) / X_std[valid_cols]
    return X_train_raw, x_test_raw, X_train_std, x_test_std, valid_cols


def fixed_pair_ols_prediction(X_train_std, x_test_std, y_train_std, valid_cols):
    coef_full = np.full(valid_cols.shape, np.nan, dtype=np.float64)
    if not np.any(valid_cols):
        return float("nan"), coef_full
    coef_valid, *_ = np.linalg.lstsq(X_train_std[:, valid_cols], y_train_std, rcond=None)
    coef_full[valid_cols] = coef_valid
    prediction_std = float(x_test_std[valid_cols] @ coef_valid)
    return prediction_std, coef_full


def fixed_pair_direct_lod_prediction(X_train_std, x_test_std, y_train_std, valid_cols):
    beta_full = np.full(valid_cols.shape, np.nan, dtype=np.float64)
    active_indices = np.flatnonzero(valid_cols)
    if active_indices.size == 0:
        return float("nan"), beta_full

    residual = y_train_std.astype(np.float64).copy()
    previous_modes_train = []
    previous_modes_test = []
    prediction_std = 0.0

    for col_idx in active_indices:
        u_train = X_train_std[:, col_idx].astype(np.float64)
        u_test = float(x_test_std[col_idx])
        m_hat_train = u_train.copy()
        m_hat_test = u_test
        for previous_train, previous_test in zip(previous_modes_train, previous_modes_test):
            coeff = float(np.sum(u_train * previous_train) / np.sum(previous_train * previous_train))
            m_hat_train = m_hat_train - coeff * previous_train
            m_hat_test = m_hat_test - coeff * previous_test

        mode_mean = float(np.mean(m_hat_train))
        mode_std = float(np.std(m_hat_train, ddof=1))
        if not np.isfinite(mode_std) or mode_std <= 0.0:
            continue
        mode_train = (m_hat_train - mode_mean) / mode_std
        mode_test = float((m_hat_test - mode_mean) / mode_std)
        beta = float(np.sum(mode_train * residual) / np.sum(mode_train * mode_train))
        beta_full[col_idx] = beta
        residual = residual - beta * mode_train
        prediction_std += beta * mode_test
        previous_modes_train.append(mode_train)
        previous_modes_test.append(mode_test)

    return float(prediction_std), beta_full


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_safe(x, y):
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x, ddof=1) == 0.0 or np.std(y, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r2_manual(y_true, y_pred):
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    return 1.0 - ss_res / ss_tot


def format_metric(value):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def write_predictions_csv(path, rows):
    fieldnames = [
        "year",
        "observed_swe",
        "fixed_pair_ols_loyo_pred",
        "fixed_pair_ols_loyo_pred_std",
        "direct_lod_loyo_pred",
        "direct_lod_loyo_pred_std",
        "prediction_error",
        "absolute_error",
        "ols_minus_direct_lod",
        "active_pair_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_coefficients_csv(path, rows):
    fieldnames = [
        "heldout_year",
        "mode_index",
        "dataset",
        "month",
        "lat",
        "lon",
        "coefficient",
        "direct_lod_beta",
        "predictor_valid_in_fold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def plot_timeseries(path, dataset_label, water_years, observed, ols_pred, direct_pred, metrics, agreement):
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.plot(water_years, observed, color="black", linewidth=1.6, label="Observed SWE anomaly")
    ax.plot(water_years, ols_pred, color="tab:blue", linewidth=1.3, label="Fixed-pair OLS LOYO")
    ax.plot(water_years, direct_pred, color="tab:red", linewidth=1.1, linestyle="--", label="Fixed-pair direct LOD")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title(f"{dataset_label} Fixed LOD Pair OLS LOYO Prediction")
    annotation = (
        f"R2 = {format_metric(metrics['r2'])}   "
        f"r = {format_metric(metrics['corr'])}   "
        f"RMSE = {format_metric(metrics['rmse'])}   "
        f"MAE = {format_metric(metrics['mae'])}\n"
        f"max |OLS - direct LOD| = {format_metric(agreement['max_abs_diff'])}   "
        f"RMSE(OLS - direct LOD) = {format_metric(agreement['rmse_diff'])}"
    )
    ax.text(
        0.01,
        0.99,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_observed_vs_predicted(path, dataset_label, observed, predicted, metrics):
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(observed, predicted, s=46, color="tab:blue", edgecolors="black", linewidths=0.4, alpha=0.85)
    all_values = np.concatenate([observed, predicted])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("Observed SWE anomaly (m)")
    ax.set_ylabel("Fixed-pair OLS LOYO predicted SWE anomaly (m)")
    ax.set_title(f"{dataset_label} Observed vs Fixed LOD Pair OLS LOYO Prediction")
    annotation = (
        f"R2 = {format_metric(metrics['r2'])}\n"
        f"r = {format_metric(metrics['corr'])}\n"
        f"RMSE = {format_metric(metrics['rmse'])}\n"
        f"MAE = {format_metric(metrics['mae'])}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    ax.grid(True, linewidth=0.25, color="0.8")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_method_agreement(path, dataset_label, direct_pred, ols_pred, agreement):
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(direct_pred, ols_pred, s=46, color="tab:green", edgecolors="black", linewidths=0.4, alpha=0.85)
    all_values = np.concatenate([direct_pred, ols_pred])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("Fixed-pair direct LOD LOYO prediction (m)")
    ax.set_ylabel("Fixed-pair OLS LOYO prediction (m)")
    ax.set_title(f"{dataset_label} Fixed-pair OLS vs direct LOD agreement")
    annotation = (
        f"max |diff| = {format_metric(agreement['max_abs_diff'])}\n"
        f"RMSE(diff) = {format_metric(agreement['rmse_diff'])}\n"
        f"r = {format_metric(agreement['corr'])}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    ax.grid(True, linewidth=0.25, color="0.8")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary_markdown(path, config, selected_pairs, metrics, agreement, outputs):
    lines = [
        f"# {config.label} fixed-selected-LOD-pairs LOYO regression check",
        "",
        "- This is the Version A sanity check with full-sample-selected LOD position/month pairs fixed before LOYO.",
        "- No LOD pair reselection is performed inside the outer leave-one-water-year-out folds.",
        "- Predictors and the SWE target are standardized using training years only in each fold.",
        "- The script compares fixed-pair OLS against a fixed-pair direct-LOD reconstruction built from the same ordered pairs.",
        "",
        "## Selected pairs",
        "",
    ]
    for pair in selected_pairs:
        lines.append(
            f"- Mode {int(pair['mode_number'])}: {pair['lag_month']} lat={float(pair['latitude']):.6f} "
            f"lon={float(pair.get('longitude_0_360', pair.get('longitude'))):.6f}"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- OLS LOYO R2: `{metrics['r2']:.6f}`",
            f"- OLS LOYO correlation: `{metrics['corr']:.6f}`",
            f"- OLS LOYO RMSE: `{metrics['rmse']:.6f}` m",
            f"- OLS LOYO MAE: `{metrics['mae']:.6f}` m",
            f"- max |OLS - direct LOD|: `{agreement['max_abs_diff']:.6e}` m",
            f"- RMSE(OLS - direct LOD): `{agreement['rmse_diff']:.6e}` m",
            f"- OLS-vs-direct-LOD correlation: `{agreement['corr']:.6f}`",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_for_dataset(config):
    start = perf_counter()
    config.output_root.mkdir(parents=True, exist_ok=True)

    target_anom_m, target_std_global = load_target()
    monthly_means, latitude, longitude = load_monthly_means(config)
    selected_pairs = load_selected_pairs(config.summary_file)
    pair_metadata = build_selected_pair_metadata(selected_pairs, latitude, longitude)

    prediction_rows = []
    coefficient_rows = []
    ols_pred = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)
    ols_pred_std = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)
    direct_pred = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)
    direct_pred_std = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)

    for fold_idx, held_out_wy in enumerate(WATER_YEARS):
        test_mask = WATER_YEARS == held_out_wy
        train_mask = ~test_mask
        train_sst = monthly_means[train_mask]
        test_sst = monthly_means[test_mask][0]
        y_train_raw = target_anom_m[train_mask]
        y_test_raw = float(target_anom_m[test_mask][0])
        y_train_std, y_mean, y_std = standardize_target(y_train_raw)
        _, _, X_train_std, x_test_std, valid_cols = build_fold_selected_pair_matrix(
            train_sst=train_sst,
            test_sst=test_sst,
            pair_metadata=pair_metadata,
        )

        ols_fold_std, ols_coef = fixed_pair_ols_prediction(X_train_std, x_test_std, y_train_std, valid_cols)
        direct_fold_std, direct_beta = fixed_pair_direct_lod_prediction(X_train_std, x_test_std, y_train_std, valid_cols)

        ols_fold_raw = float(y_mean + y_std * ols_fold_std)
        direct_fold_raw = float(y_mean + y_std * direct_fold_std)

        ols_pred[fold_idx] = ols_fold_raw
        ols_pred_std[fold_idx] = ols_fold_std
        direct_pred[fold_idx] = direct_fold_raw
        direct_pred_std[fold_idx] = direct_fold_std

        prediction_rows.append(
            {
                "year": int(held_out_wy),
                "observed_swe": y_test_raw,
                "fixed_pair_ols_loyo_pred": ols_fold_raw,
                "fixed_pair_ols_loyo_pred_std": ols_fold_std,
                "direct_lod_loyo_pred": direct_fold_raw,
                "direct_lod_loyo_pred_std": direct_fold_std,
                "prediction_error": ols_fold_raw - y_test_raw,
                "absolute_error": abs(ols_fold_raw - y_test_raw),
                "ols_minus_direct_lod": ols_fold_raw - direct_fold_raw,
                "active_pair_count": int(np.count_nonzero(valid_cols)),
            }
        )

        for pair_idx, pair in enumerate(pair_metadata):
            coefficient_rows.append(
                {
                    "heldout_year": int(held_out_wy),
                    "mode_index": int(pair["mode_index"]),
                    "dataset": config.label,
                    "month": str(pair["lag_month"]),
                    "lat": float(pair["latitude"]),
                    "lon": float(pair["longitude_0_360"]),
                    "coefficient": None if math.isnan(float(ols_coef[pair_idx])) else float(ols_coef[pair_idx]),
                    "direct_lod_beta": None if math.isnan(float(direct_beta[pair_idx])) else float(direct_beta[pair_idx]),
                    "predictor_valid_in_fold": bool(valid_cols[pair_idx]),
                }
            )

        print(
            f"{config.label} fold {fold_idx + 1:02d}/{WATER_YEARS.size}: "
            f"WY={int(held_out_wy)} active_pairs={int(np.count_nonzero(valid_cols))} "
            f"obs={y_test_raw:.6f} ols={ols_fold_raw:.6f} direct={direct_fold_raw:.6f}",
            flush=True,
        )

    metrics = {
        "r2": r2_manual(target_anom_m, ols_pred),
        "corr": corrcoef_safe(target_anom_m, ols_pred),
        "rmse": rmse(target_anom_m, ols_pred),
        "mae": mae(target_anom_m, ols_pred),
    }
    agreement = {
        "max_abs_diff": float(np.max(np.abs(ols_pred - direct_pred))),
        "rmse_diff": rmse(ols_pred, direct_pred),
        "corr": corrcoef_safe(ols_pred, direct_pred),
    }

    predictions_csv = config.output_root / f"{config.key}_fixed_lod_pairs_loyo_predictions.csv"
    coefficients_csv = config.output_root / f"{config.key}_fixed_lod_pairs_loyo_coefficients.csv"
    line_chart_png = config.output_root / "fixed_lod_pairs_loyo_line_chart.png"
    scatter_png = config.output_root / "fixed_lod_pairs_loyo_scatter.png"
    agreement_png = config.output_root / "fixed_lod_pairs_method_agreement.png"
    summary_json = config.output_root / f"{config.key}_fixed_lod_pairs_loyo_summary.json"
    summary_md = config.output_root / f"{config.key}_fixed_lod_pairs_loyo_summary.md"

    write_predictions_csv(predictions_csv, prediction_rows)
    write_coefficients_csv(coefficients_csv, coefficient_rows)
    plot_timeseries(line_chart_png, config.label, WATER_YEARS, target_anom_m, ols_pred, direct_pred, metrics, agreement)
    plot_observed_vs_predicted(scatter_png, config.label, target_anom_m, ols_pred, metrics)
    plot_method_agreement(agreement_png, config.label, direct_pred, ols_pred, agreement)

    outputs = {
        "predictions_csv": str(predictions_csv),
        "coefficients_csv": str(coefficients_csv),
        "line_chart_png": str(line_chart_png),
        "scatter_png": str(scatter_png),
        "method_agreement_png": str(agreement_png),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary_payload = {
        "dataset": config.label,
        "water_year_start": WATER_YEAR_START,
        "water_year_end": WATER_YEAR_END,
        "selected_pair_count": len(pair_metadata),
        "selected_pairs": pair_metadata,
        "metrics": metrics,
        "agreement_with_fixed_pair_direct_lod": agreement,
        "target_source": str(TARGET_FILE),
        "selected_pairs_source": str(config.summary_file),
        "outputs": outputs,
        "runtime_seconds": perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
        "observed_swe_standardized_global_reference": target_std_global.tolist(),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(summary_md, config, selected_pairs, metrics, agreement, outputs)
    return summary_payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["cobe2", "era5", "both"],
        default="both",
        help="Which fixed-pair LOYO check to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_keys = ["cobe2", "era5"] if args.dataset == "both" else [args.dataset]
    results = {}
    for key in dataset_keys:
        config = DATASET_CONFIGS[key]
        print(f"starting fixed-selected-LOD-pairs LOYO check for {config.label}", flush=True)
        results[key] = run_for_dataset(config)
    print(json.dumps({key: value["outputs"] for key, value in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
