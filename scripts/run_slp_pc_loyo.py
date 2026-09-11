#!/usr/bin/env python3
"""
Run strict LOYO ridge SWE prediction using ERA5 SLP-domain PCs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "era5_slp_pc_loyo"
TARGET_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)
PSCRATCH_ROOT = Path(os.environ.get("PSCRATCH", "/pscratch/sd/h/hyvchen")).expanduser()
PC_INPUT_DIR = PSCRATCH_ROOT / "Snow-Predication-at-Sierra" / "era5_slp_domain_pcs"

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "slp_pc_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "slp_pc_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "slp_pc_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "slp_pc_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "slp_pc_selected_alpha_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "slp_pc_beta_by_fold.csv"
SUMMARY_JSON = OUTPUT_DIR / "slp_pc_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "slp_pc_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "slp_pc_scatter_three_panel.png"
ERROR_PNG = OUTPUT_DIR / "slp_pc_error_by_year.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
MONTH_SPECS = [("Sep", -1, 9), ("Oct", -1, 10), ("Nov", -1, 11), ("Dec", -1, 12), ("Jan", 0, 1), ("Feb", 0, 2), ("Mar", 0, 3)]
DOMAIN_PREFIX = {"PNA_SLP": "PNA", "NAO_SLP": "NAO", "NAM_SLP": "NAM"}
PC_INPUT_FILES = {
    "PNA_SLP": PC_INPUT_DIR / "PNA_SLP_ERA5_PC1_6_monthly.nc",
    "NAO_SLP": PC_INPUT_DIR / "NAO_SLP_ERA5_PC1_6_monthly.nc",
    "NAM_SLP": PC_INPUT_DIR / "NAM_SLP_ERA5_PC1_6_monthly.nc",
}
MODEL_NAMES = ["PNA_SLP_PC1_6", "NAO_SLP_PC1_6", "NAM_SLP_PC1_6"]
MODEL_TO_DOMAIN = {
    "PNA_SLP_PC1_6": "PNA_SLP",
    "NAO_SLP_PC1_6": "NAO_SLP",
    "NAM_SLP_PC1_6": "NAM_SLP",
}
PERIOD_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    yy = y_true[mask]
    pp = y_pred[mask]
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "mean_error": float(np.mean(error)),
        "median_abs_error": float(np.median(np.abs(error))),
    }


def standardize_train_only(
    x_train_raw: np.ndarray, x_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("Predictor train-fold standard deviation must be positive.")
    return (
        (x_train_raw - x_mean[None, :]) / x_std[None, :],
        (x_test_raw - x_mean) / x_std,
        x_mean,
        x_std,
    )


def standardize_target_train_only(y_train_raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Target train-fold standard deviation must be positive.")
    return (y_train_raw - y_mean) / y_std, y_mean, y_std


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha: Optional[float] = None
    best_mse: Optional[float] = None
    for alpha in ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=float)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            x_inner_train = x_train_raw[mask, :]
            x_inner_test = x_train_raw[~mask, :][0]
            y_inner_train = y_train_raw[mask]
            y_inner_test = float(y_train_raw[~mask][0])
            x_inner_train_std, x_inner_test_std, _, _ = standardize_train_only(x_inner_train, x_inner_test)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train)
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, alpha)
            pred_std = float(x_inner_test_std @ beta_std)
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = mse
    assert best_alpha is not None and best_mse is not None
    return best_alpha, best_mse


def fit_outer_model(
    x_train_raw: np.ndarray, y_train_raw: np.ndarray, x_test_raw: np.ndarray, alpha: float
) -> Tuple[float, np.ndarray, float]:
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
    pred_std = float(x_test_std @ beta_std)
    pred_raw = y_mean + y_std * pred_std
    beta_raw = (y_std / x_std) * beta_std
    intercept_raw = float(y_mean - np.sum(beta_raw * x_mean))
    return pred_raw, beta_raw, intercept_raw


def load_target_table() -> pd.DataFrame:
    table = pd.read_csv(TARGET_TABLE_CSV)
    required = ["water_year", "obs_swe"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required target columns in {TARGET_TABLE_CSV}: {missing}")
    table = table[required].copy()
    table["water_year"] = table["water_year"].astype(int)
    table = table[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    table = table.sort_values("water_year").reset_index(drop=True)
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Target table does not match WY1985--WY2021.")
    return table


def load_domain_pc_frames() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    frames: Dict[str, pd.DataFrame] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    for domain_name, path in PC_INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing PC input file: {path}")
        with xr.open_dataset(path, engine="netcdf4") as ds:
            times = pd.to_datetime(np.asarray(ds["time"].values))
            frame = pd.DataFrame({"time": times})
            for mode_idx in range(1, 7):
                frame[f"PC{mode_idx}"] = np.asarray(ds[f"PC{mode_idx}"].values, dtype=float)
            frames[domain_name] = frame
            metadata[domain_name] = {
                "slp_variable_name": ds.attrs.get("slp_variable_name", "MSL"),
                "slp_units": ds.attrs.get("slp_units", "Pa"),
                "date_coverage": [str(times.min().date()), str(times.max().date())],
                "explained_variance_ratio": [float(x) for x in np.asarray(ds["explained_variance_ratio"].values, dtype=float).tolist()],
                "path": str(path),
            }
    return frames, metadata


def build_predictor_table() -> Tuple[pd.DataFrame, Dict[str, object]]:
    target = load_target_table()
    frames, metadata = load_domain_pc_frames()
    row_dicts: List[Dict[str, object]] = []
    feature_sets: Dict[str, List[str]] = {}

    for water_year in WATER_YEARS:
        row: Dict[str, object] = {
            "water_year": int(water_year),
            "obs_swe": float(target.loc[target["water_year"] == water_year, "obs_swe"].iloc[0]),
        }
        for domain_name, frame in frames.items():
            prefix = DOMAIN_PREFIX[domain_name]
            feature_names: List[str] = []
            for month_label, year_offset, month_num in MONTH_SPECS:
                year = water_year + year_offset
                target_time = pd.Timestamp(year=year, month=month_num, day=1)
                matches = frame.loc[frame["time"] == target_time]
                if len(matches) != 1:
                    raise ValueError(f"Expected one match for {domain_name} {target_time.date()}, found {len(matches)}")
                for mode_idx in range(1, 7):
                    column_name = f"{prefix}_PC{mode_idx}_{month_label}"
                    row[column_name] = float(matches[f"PC{mode_idx}"].iloc[0])
                    feature_names.append(column_name)
            feature_sets[domain_name] = feature_names
        row_dicts.append(row)

    table = pd.DataFrame(row_dicts)
    table = table.sort_values("water_year").reset_index(drop=True)
    table.to_csv(PREDICTOR_TABLE_CSV, index=False)
    return table, {"pc_metadata": metadata, "feature_sets": feature_sets}


def run_model(
    table: pd.DataFrame,
    model_name: str,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    years = table["water_year"].to_numpy(dtype=int)
    obs = table["obs_swe"].to_numpy(dtype=float)
    x_all = table[list(feature_names)].to_numpy(dtype=float)

    prediction_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []

    all_beta_columns = sorted(
        {
            feature
            for model in MODEL_NAMES
            for feature in feature_names_for_model(model)
        }
    )

    preds = np.full(len(years), np.nan, dtype=float)
    selected_alphas = np.full(len(years), np.nan, dtype=float)

    for outer_idx, water_year in enumerate(years):
        mask = np.ones(len(years), dtype=bool)
        mask[outer_idx] = False
        x_train = x_all[mask, :]
        x_test = x_all[~mask, :][0]
        y_train = obs[mask]
        y_test = float(obs[~mask][0])

        selected_alpha, inner_cv_mse = inner_loyo_best_alpha(x_train, y_train)
        pred_raw, beta_raw, intercept_raw = fit_outer_model(x_train, y_train, x_test, selected_alpha)
        preds[outer_idx] = pred_raw
        selected_alphas[outer_idx] = selected_alpha
        error = pred_raw - y_test

        prediction_rows.append(
            {
                "model_name": model_name,
                "heldout_wy": int(water_year),
                "obs_swe": y_test,
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error,
                "residual_obs_minus_pred": -error,
                "abs_error": abs(error),
                "sign_correct": float((np.sign(pred_raw) == np.sign(y_test)) and (pred_raw != 0.0) and (y_test != 0.0)),
                "selected_alpha": selected_alpha,
                "num_predictors": int(len(feature_names)),
                "selected_features": "|".join(feature_names),
            }
        )
        alpha_rows.append(
            {
                "model_name": model_name,
                "heldout_wy": int(water_year),
                "selected_alpha": selected_alpha,
                "inner_cv_mse": inner_cv_mse,
            }
        )
        beta_row = {
            "model_name": model_name,
            "heldout_wy": int(water_year),
            "intercept": intercept_raw,
        }
        for beta_name in all_beta_columns:
            beta_row[f"beta_{beta_name}"] = np.nan
        for feature_name, coef in zip(feature_names, beta_raw.tolist()):
            beta_row[f"beta_{feature_name}"] = float(coef)
        beta_rows.append(beta_row)

        print(
            f"LOYO heldout_WY={int(water_year)} model={model_name} selected_alpha={selected_alpha:g} "
            f"obs={y_test:.6f} pred={pred_raw:.6f}",
            flush=True,
        )

    metrics_bundle = compute_metric_bundle(obs, preds)
    metrics_bundle["num_predictors"] = int(len(feature_names))
    metrics_bundle["model_name"] = model_name
    metrics_bundle["selected_features"] = "|".join(feature_names)
    metrics_bundle["selected_alpha_min"] = float(np.nanmin(selected_alphas))
    metrics_bundle["selected_alpha_max"] = float(np.nanmax(selected_alphas))
    metrics_bundle["selected_alpha_median"] = float(np.nanmedian(selected_alphas))

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(alpha_rows),
        pd.DataFrame(beta_rows),
        metrics_bundle,
    )


def feature_names_for_model(model_name: str) -> List[str]:
    domain_name = MODEL_TO_DOMAIN[model_name]
    prefix = DOMAIN_PREFIX[domain_name]
    return [f"{prefix}_PC{mode_idx}_{month_label}" for month_label, _, _ in MONTH_SPECS for mode_idx in range(1, 7)]


def compute_period_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name in MODEL_NAMES:
        subset = predictions[predictions["model_name"] == model_name].copy()
        years = subset["heldout_wy"].to_numpy(dtype=int)
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        for period_name, selector in PERIOD_SPECS:
            mask = selector(years)
            if int(np.sum(mask)) < 2:
                continue
            metrics = compute_metric_bundle(obs[mask], pred[mask])
            rows.append(
                {
                    "model_name": model_name,
                    "period": period_name,
                    "n_years": int(np.sum(mask)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def plot_observed_vs_predicted(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    base = predictions[predictions["model_name"] == MODEL_NAMES[0]].sort_values("heldout_wy")
    ax.plot(base["heldout_wy"], base["obs_swe"], color="black", linewidth=2.6, label="Observed SWE anomaly")
    colors = {
        "PNA_SLP_PC1_6": "#1b4965",
        "NAO_SLP_PC1_6": "#ca6702",
        "NAM_SLP_PC1_6": "#6a994e",
    }
    for model_name in MODEL_NAMES:
        subset = predictions[predictions["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(subset["heldout_wy"], subset["pred_swe"], linewidth=1.7, label=model_name, color=colors[model_name])
    ax.set_title("Strict LOYO SWE prediction using ERA5 SLP-domain PCs")
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_three_panel(predictions: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharex=True, sharey=True)
    colors = {
        "PNA_SLP_PC1_6": "#1b4965",
        "NAO_SLP_PC1_6": "#ca6702",
        "NAM_SLP_PC1_6": "#6a994e",
    }
    for ax, model_name in zip(axes, MODEL_NAMES):
        subset = predictions[predictions["model_name"] == model_name].copy()
        metric_row = metrics_df.loc[metrics_df["model_name"] == model_name].iloc[0]
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        min_val = min(np.min(obs), np.min(pred))
        max_val = max(np.max(obs), np.max(pred))
        ax.scatter(obs, pred, s=36, color=colors[model_name], alpha=0.85)
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(model_name)
        ax.set_xlabel("Observed SWE anomaly (m)")
        if ax is axes[0]:
            ax.set_ylabel("Predicted SWE anomaly (m)")
        ax.text(
            0.03,
            0.97,
            f"RMSE={metric_row['RMSE']:.3f}\nR2={metric_row['R2']:.3f}\nCorr={metric_row['r']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.grid(alpha=0.25)
    fig.suptitle("Observed vs predicted SWE: ERA5 SLP-domain PC models", fontsize=14)
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_year(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    colors = {
        "PNA_SLP_PC1_6": "#1b4965",
        "NAO_SLP_PC1_6": "#ca6702",
        "NAM_SLP_PC1_6": "#6a994e",
    }
    for model_name in MODEL_NAMES:
        subset = predictions[predictions["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            subset["heldout_wy"],
            subset["error_pred_minus_obs"],
            marker="o",
            linewidth=1.3,
            markersize=3.5,
            color=colors[model_name],
            label=model_name,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Prediction error by water year: ERA5 SLP-domain PC models")
    ax.set_xlabel("Water year")
    ax.set_ylabel("Prediction error (pred - obs) (m)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(ERROR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_alphas(alpha_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        values = alpha_df.loc[alpha_df["model_name"] == model_name, "selected_alpha"].to_numpy(dtype=float)
        summary[model_name] = {
            "min": float(np.nanmin(values)),
            "median": float(np.nanmedian(values)),
            "max": float(np.nanmax(values)),
        }
    return summary


def summarize_coefficients(beta_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for model_name in MODEL_NAMES:
        subset = beta_df[beta_df["model_name"] == model_name]
        coef_cols = [column for column in subset.columns if column.startswith("beta_")]
        rows: List[Dict[str, object]] = []
        for column in coef_cols:
            values = subset[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "feature": column.replace("beta_", "", 1),
                    "mean": float(np.mean(values)),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "sign_positive_fraction": float(np.mean(values > 0.0)),
                }
            )
        rows.sort(key=lambda item: item["mean_abs"], reverse=True)
        summary[model_name] = {"top_coefficients_by_mean_abs": rows[:10]}
    return summary


def short_answer(metrics_df: pd.DataFrame) -> str:
    best_r2_row = metrics_df.sort_values(["R2", "RMSE", "sign_accuracy"], ascending=[False, True, False]).iloc[0]
    best_sign_row = metrics_df.sort_values(["sign_accuracy", "R2"], ascending=[False, False]).iloc[0]
    if best_r2_row["model_name"] == best_sign_row["model_name"]:
        return (
            f"{best_r2_row['model_name']} was best overall by both R2 ({best_r2_row['R2']:.3f}) "
            f"and sign accuracy ({best_r2_row['sign_accuracy']:.3f}) among the three SLP families."
        )
    return (
        f"{best_r2_row['model_name']} was best by R2 ({best_r2_row['R2']:.3f}) and RMSE ({best_r2_row['RMSE']:.3f}), "
        f"while {best_sign_row['model_name']} had the highest sign accuracy ({best_sign_row['sign_accuracy']:.3f})."
    )


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, input_metadata = build_predictor_table()
    prediction_frames: List[pd.DataFrame] = []
    alpha_frames: List[pd.DataFrame] = []
    beta_frames: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []

    for model_name in MODEL_NAMES:
        features = feature_names_for_model(model_name)
        prediction_df, alpha_df, beta_df, metrics_bundle = run_model(table, model_name, features)
        prediction_frames.append(prediction_df)
        alpha_frames.append(alpha_df)
        beta_frames.append(beta_df)
        metric_rows.append(metrics_bundle)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    period_metrics_df = compute_period_metrics(predictions)
    alpha_df = pd.concat(alpha_frames, ignore_index=True)
    beta_df = pd.concat(beta_frames, ignore_index=True)

    predictions.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_metrics_df.to_csv(PERIOD_METRICS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)

    plot_observed_vs_predicted(predictions)
    plot_scatter_three_panel(predictions, metrics_df)
    plot_error_by_year(predictions)

    best_r2_row = metrics_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    best_rmse_row = metrics_df.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    best_sign_row = metrics_df.sort_values(["sign_accuracy", "R2"], ascending=[False, False]).iloc[0]

    summary = {
        "input_files": {
            "target_table": str(TARGET_TABLE_CSV),
            "pc_input_files": {domain: str(path) for domain, path in PC_INPUT_FILES.items()},
        },
        "slp_variable_name": {
            domain: metadata["slp_variable_name"]
            for domain, metadata in input_metadata["pc_metadata"].items()
        },
        "slp_units": {
            domain: metadata["slp_units"]
            for domain, metadata in input_metadata["pc_metadata"].items()
        },
        "date_coverage": {
            domain: metadata["date_coverage"]
            for domain, metadata in input_metadata["pc_metadata"].items()
        },
        "domain_definitions": {
            domain: {
                "feature_prefix": DOMAIN_PREFIX[domain],
                "lat_min": {"PNA_SLP": 20.0, "NAO_SLP": 20.0, "NAM_SLP": 20.0}[domain],
                "lat_max": {"PNA_SLP": 85.0, "NAO_SLP": 80.0, "NAM_SLP": 90.0}[domain],
                "lon_windows_360": {
                    "PNA_SLP": [[120.0, 240.0]],
                    "NAO_SLP": [[270.0, 360.0], [0.0, 40.0]],
                    "NAM_SLP": [[0.0, 360.0]],
                }[domain],
            }
            for domain in PC_INPUT_FILES
        },
        "pc_output_files": {domain: metadata["path"] for domain, metadata in input_metadata["pc_metadata"].items()},
        "explained_variance_ratio_by_domain": {
            domain: metadata["explained_variance_ratio"]
            for domain, metadata in input_metadata["pc_metadata"].items()
        },
        "feature_sets": input_metadata["feature_sets"],
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_metrics_df.to_dict(orient="records"),
        "selected_alpha_summary": summarize_alphas(alpha_df),
        "coefficient_summary": summarize_coefficients(beta_df),
        "best_model_by_R2": {
            "model_name": best_r2_row["model_name"],
            "R2": float(best_r2_row["R2"]),
        },
        "best_model_by_RMSE": {
            "model_name": best_rmse_row["model_name"],
            "RMSE": float(best_rmse_row["RMSE"]),
        },
        "best_model_by_sign_accuracy": {
            "model_name": best_sign_row["model_name"],
            "sign_accuracy": float(best_sign_row["sign_accuracy"]),
        },
        "short_answer": short_answer(metrics_df),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(f"SLP source file family: {PC_INPUT_FILES['PNA_SLP'].name.replace('PNA_SLP_ERA5_PC1_6_monthly.nc', 'e5.oper.an.sfc.128_151_msl.ll025sc.YYYYMM...nc')}", flush=True)
    print(f"PC output directory: {PC_INPUT_DIR}", flush=True)
    print(f"Regression output directory: {OUTPUT_DIR}", flush=True)
    print("Metrics:", flush=True)
    for row in metrics_df.sort_values("model_name").to_dict(orient="records"):
        print(
            f"  {row['model_name']}: R2={row['R2']:.3f}, RMSE={row['RMSE']:.6f}, "
            f"sign_accuracy={row['sign_accuracy']:.3f}",
            flush=True,
        )
    print(f"Best by R2: {best_r2_row['model_name']}", flush=True)
    print(f"Best by RMSE: {best_rmse_row['model_name']}", flush=True)
    print(f"Best by sign_accuracy: {best_sign_row['model_name']}", flush=True)
    print(f"Short answer: {summary['short_answer']}", flush=True)


if __name__ == "__main__":
    main()
