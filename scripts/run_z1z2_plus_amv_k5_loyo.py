#!/usr/bin/env python3
"""
Strict LOYO ridge SWE prediction using oracle Z1/Z2, reduced AMV/AMO K5, and the combined block.
"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
)

PATCH_PREDICTORS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_predictors.csv"
)
BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)
AMV_K5_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_core_plus_forward_addition"
    / "amv_core_plus_forward_predictor_table.csv"
)
AMV_FALLBACK_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_subset_selection_diagnostic"
    / "amv_amo_predictor_table.csv"
)

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "z1z2_amv_k5_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "z1z2_amv_k5_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "z1z2_amv_k5_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "z1z2_amv_k5_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "z1z2_amv_k5_selected_alpha_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "z1z2_amv_k5_beta_by_fold.csv"
SUMMARY_JSON = OUTPUT_DIR / "z1z2_amv_k5_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "z1z2_amv_k5_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "z1z2_amv_k5_scatter_three_panel.png"
ERROR_PNG = OUTPUT_DIR / "z1z2_amv_k5_error_by_year.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)

Z1_NAME = "Z1_M1_Jan_lat_-9.5_lon_133.5"
Z2_NAME = "Z2_M2_Oct_lat_0.5_lon_136.5"
Z1_OUTPUT = "Z1"
Z2_OUTPUT = "Z2"
A_COLUMNS = ["AMV_PC4_Sep", "AMV_PC5_Feb", "AMV_PC2_Feb", "AMV_PC4_Nov", "AMV_PC5_Mar"]

MODEL_FEATURES = {
    "Z1_Z2": [Z1_OUTPUT, Z2_OUTPUT],
    "AMV_AMO_K5": list(A_COLUMNS),
    "Z1_Z2_AMV_AMO_K5": [Z1_OUTPUT, Z2_OUTPUT] + list(A_COLUMNS),
}
PLOT_ORDER = ["Z1_Z2", "AMV_AMO_K5", "Z1_Z2_AMV_AMO_K5"]
PLOT_LABELS = {
    "Z1_Z2": "Z1_Z2",
    "AMV_AMO_K5": "AMV_AMO_K5",
    "Z1_Z2_AMV_AMO_K5": "Z1_Z2_AMV_AMO_K5",
}
DISPLAY_LABELS = {
    "Z1_Z2": r"\(Z_1+Z_2\)",
    "AMV_AMO_K5": "AMV/AMO K5",
    "Z1_Z2_AMV_AMO_K5": r"\(Z_1+Z_2\) + AMV/AMO K5",
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


def standardize_target_train_only(
    y_train_raw: np.ndarray, y_test_raw: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Target train-fold standard deviation must be positive.")
    return (y_train_raw - y_mean) / y_std, y_mean, y_std


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    beta_std = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    return np.asarray(beta_std, dtype=float)


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha: Optional[float] = None
    best_mse: Optional[float] = None
    for alpha in ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=float)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train = x_train_raw[inner_mask, :]
            x_inner_test = x_train_raw[~inner_mask, :][0]
            y_inner_train = y_train_raw[inner_mask]
            y_inner_test = float(y_train_raw[~inner_mask][0])

            x_inner_train_std, x_inner_test_std, _, _ = standardize_train_only(x_inner_train, x_inner_test)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train, np.asarray([y_inner_test]))
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
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw, np.asarray([0.0]))
    beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
    pred_std = float(x_test_std @ beta_std)
    pred_raw = y_mean + y_std * pred_std
    beta_raw = (y_std / x_std) * beta_std
    intercept_raw = float(y_mean - np.sum(beta_raw * x_mean))
    return pred_raw, beta_raw, intercept_raw


def load_predictor_table() -> Tuple[pd.DataFrame, Dict[str, object]]:
    predictors = pd.read_csv(PATCH_PREDICTORS_CSV)
    predictors = predictors[predictors["patch_size"] == "exact_grid_cell"].copy()
    predictors = predictors[["water_year", Z1_NAME, Z2_NAME]].rename(
        columns={Z1_NAME: Z1_OUTPUT, Z2_NAME: Z2_OUTPUT}
    )
    predictors["water_year"] = predictors["water_year"].astype(int)

    base_predictions = pd.read_csv(BASE_PREDICTIONS_CSV)
    base_predictions = base_predictions[
        (base_predictions["patch_size"] == "exact_grid_cell")
        & (base_predictions["model_name"] == "Z1_Z2")
    ].copy()
    base_predictions = base_predictions[["heldout_wy", "obs_swe"]].rename(columns={"heldout_wy": "water_year"})
    base_predictions["water_year"] = base_predictions["water_year"].astype(int)

    a_source = AMV_K5_TABLE_CSV if AMV_K5_TABLE_CSV.exists() else AMV_FALLBACK_CSV
    amv = pd.read_csv(a_source)
    deduped_columns = []
    seen = set()
    for col in amv.columns:
        base = col.split(".")[0]
        if base not in seen:
            deduped_columns.append(col)
            seen.add(base)
    if len(deduped_columns) != len(amv.columns):
        amv = amv[deduped_columns].copy()
        amv.columns = [col.split(".")[0] for col in deduped_columns]
    amv["water_year"] = amv["water_year"].astype(int)
    needed_amv = ["water_year"] + A_COLUMNS
    missing = [col for col in needed_amv if col not in amv.columns]
    if missing:
        raise ValueError(f"Missing AMV K5 columns: {missing}")
    amv = amv[needed_amv].copy()

    table = predictors.merge(base_predictions, on="water_year", how="inner")
    table = table.merge(amv, on="water_year", how="inner")
    table = table.sort_values("water_year").reset_index(drop=True)
    table = table[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Predictor table does not match WY1985--WY2021.")

    input_files = {
        "z_file": str(PATCH_PREDICTORS_CSV),
        "target_file": str(BASE_PREDICTIONS_CSV),
        "amv_file": str(a_source),
        "deduplicated_amv_headers": bool(len(deduped_columns) != len(pd.read_csv(a_source, nrows=0).columns)),
    }
    return table[["water_year", "obs_swe", Z1_OUTPUT, Z2_OUTPUT] + A_COLUMNS].copy(), input_files


def run_models(table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    y = table["obs_swe"].to_numpy(dtype=float)
    prediction_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []

    for model_name, feature_list in MODEL_FEATURES.items():
        x_all = table[feature_list].to_numpy(dtype=float)
        for held_idx, held_year in enumerate(years):
            train_mask = np.ones(len(years), dtype=bool)
            train_mask[held_idx] = False
            x_train = x_all[train_mask, :]
            x_test = x_all[~train_mask, :][0]
            y_train = y[train_mask]
            obs = y[held_idx]

            alpha, inner_cv_mse = inner_loyo_best_alpha(x_train, y_train)
            pred_raw, beta_raw, intercept_raw = fit_outer_model(x_train, y_train, x_test, alpha)
            err = pred_raw - obs
            prediction_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": int(held_year),
                    "obs_swe": float(obs),
                    "pred_swe": float(pred_raw),
                    "error_pred_minus_obs": float(err),
                    "residual_obs_minus_pred": float(-err),
                    "abs_error": float(abs(err)),
                    "sign_correct": float(np.sign(pred_raw) == np.sign(obs)) if obs != 0.0 and pred_raw != 0.0 else np.nan,
                    "selected_alpha": float(alpha),
                    "num_predictors": int(len(feature_list)),
                }
            )
            alpha_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": int(held_year),
                    "selected_alpha": float(alpha),
                    "inner_cv_mse": float(inner_cv_mse),
                }
            )
            beta_row: Dict[str, object] = {
                "model_name": model_name,
                "heldout_wy": int(held_year),
                "intercept": float(intercept_raw),
                "beta_Z1": np.nan,
                "beta_Z2": np.nan,
                "beta_AMV_PC4_Sep": np.nan,
                "beta_AMV_PC5_Feb": np.nan,
                "beta_AMV_PC2_Feb": np.nan,
                "beta_AMV_PC4_Nov": np.nan,
                "beta_AMV_PC5_Mar": np.nan,
            }
            for feature, beta in zip(feature_list, beta_raw):
                key = "beta_" + feature
                beta_row[key] = float(beta)
            beta_rows.append(beta_row)

    return (
        pd.DataFrame(prediction_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
        pd.DataFrame(alpha_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
        pd.DataFrame(beta_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
    )


def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        metrics = compute_metric_bundle(
            sub["obs_swe"].to_numpy(dtype=float),
            sub["pred_swe"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "num_predictors": int(sub["num_predictors"].iloc[0]),
                "r": metrics["r"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_error": metrics["mean_error"],
                "median_abs_error": metrics["median_abs_error"],
            }
        )
    return pd.DataFrame(rows)


def compute_period_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        wy = sub["heldout_wy"].to_numpy(dtype=int)
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        for group_name, selector in PERIOD_SPECS:
            mask = selector(wy)
            metrics = compute_metric_bundle(obs[mask], pred[mask])
            rows.append(
                {
                    "model_name": model_name,
                    "group_name": group_name,
                    "n_years": int(mask.sum()),
                    "r": metrics["r"],
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "sign_accuracy": metrics["sign_accuracy"],
                    "mean_error": metrics["mean_error"],
                    "median_abs_error": metrics["median_abs_error"],
                }
            )
    return pd.DataFrame(rows).sort_values(["model_name", "group_name"]).reset_index(drop=True)


def summarize_alpha(alpha_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for model_name in PLOT_ORDER:
        sub = alpha_df[alpha_df["model_name"] == model_name]
        values = sub["selected_alpha"].to_numpy(dtype=float)
        counts = sub["selected_alpha"].value_counts().sort_index()
        summary[model_name] = {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "value_counts": {str(k): int(v) for k, v in counts.items()},
        }
    return summary


def summarize_coefficients(beta_df: pd.DataFrame) -> Dict[str, object]:
    coef_cols = [
        "beta_Z1",
        "beta_Z2",
        "beta_AMV_PC4_Sep",
        "beta_AMV_PC5_Feb",
        "beta_AMV_PC2_Feb",
        "beta_AMV_PC4_Nov",
        "beta_AMV_PC5_Mar",
    ]
    summary: Dict[str, object] = {}
    for model_name in PLOT_ORDER:
        sub = beta_df[beta_df["model_name"] == model_name]
        model_summary: Dict[str, object] = {}
        for col in coef_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            model_summary[col] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "num_positive": int((vals > 0).sum()),
                "num_negative": int((vals < 0).sum()),
            }
        summary[model_name] = model_summary
    return summary


def make_timeseries_plot(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    base = pred_df[pred_df["model_name"] == "Z1_Z2"].sort_values("heldout_wy")
    ax.plot(base["heldout_wy"], base["obs_swe"], color="black", linewidth=2.4, label="Observed")
    colors = {"Z1_Z2": "#1f77b4", "AMV_AMO_K5": "#ff7f0e", "Z1_Z2_AMV_AMO_K5": "#d62728"}
    labels = {
        "Z1_Z2": "Z1_Z2",
        "AMV_AMO_K5": "AMV_AMO_K5",
        "Z1_Z2_AMV_AMO_K5": "Z1_Z2_AMV_AMO_K5",
    }
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(sub["heldout_wy"], sub["pred_swe"], marker="o", linewidth=1.7, color=colors[model_name], label=labels[model_name])
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO predictions: Z1+Z2, AMV/AMO K5, and combined")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OBS_PRED_PNG, dpi=200)
    plt.close(fig)


def make_scatter_plot(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharex=True, sharey=True)
    all_obs = pred_df["obs_swe"].to_numpy(dtype=float)
    all_pred = pred_df["pred_swe"].to_numpy(dtype=float)
    lo = min(np.min(all_obs), np.min(all_pred))
    hi = max(np.max(all_obs), np.max(all_pred))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lims = (lo - pad, hi + pad)
    metric_lookup = metrics_df.set_index("model_name")
    colors = {"Z1_Z2": "#1f77b4", "AMV_AMO_K5": "#ff7f0e", "Z1_Z2_AMV_AMO_K5": "#d62728"}
    titles = {"Z1_Z2": "Z1_Z2", "AMV_AMO_K5": "AMV_AMO_K5", "Z1_Z2_AMV_AMO_K5": "Z1_Z2_AMV_AMO_K5"}
    for ax, model_name in zip(axes, PLOT_ORDER):
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        row = metric_lookup.loc[model_name]
        ax.scatter(obs, pred, color=colors[model_name], s=48, alpha=0.85)
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(titles[model_name])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.04,
            0.96,
            "RMSE = {:.4f}\n$R^2$ = {:.3f}\nCorr = {:.3f}".format(
                float(row["RMSE"]),
                float(row["R2"]),
                float(row["r"]),
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
    axes[0].set_ylabel("Predicted SWE anomaly (m)")
    for ax in axes:
        ax.set_xlabel("Observed SWE anomaly (m)")
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=200)
    plt.close(fig)


def make_error_plot(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    colors = {"Z1_Z2": "#1f77b4", "AMV_AMO_K5": "#ff7f0e", "Z1_Z2_AMV_AMO_K5": "#d62728"}
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["error_pred_minus_obs"],
            marker="o",
            linewidth=1.6,
            color=colors[model_name],
            label=model_name,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel(r"$\widehat{SWE}-SWE_{\mathrm{obs}}$ (m)")
    ax.set_title("Prediction error by year")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(ERROR_PNG, dpi=200)
    plt.close(fig)


def build_summary(
    input_files: Dict[str, object],
    metrics_df: pd.DataFrame,
    period_metrics_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    beta_df: pd.DataFrame,
) -> Dict[str, object]:
    alpha_summary = summarize_alpha(alpha_df)
    coefficient_summary = summarize_coefficients(beta_df)
    best_r2_row = metrics_df.loc[metrics_df["R2"].astype(float).idxmax()].to_dict()
    best_rmse_row = metrics_df.loc[metrics_df["RMSE"].astype(float).idxmin()].to_dict()
    best_sign_row = metrics_df.loc[metrics_df["sign_accuracy"].astype(float).idxmax()].to_dict()

    metric_lookup = metrics_df.set_index("model_name")
    combined = metric_lookup.loc["Z1_Z2_AMV_AMO_K5"]
    z_only = metric_lookup.loc["Z1_Z2"]
    a_only = metric_lookup.loc["AMV_AMO_K5"]
    combined_vs_components = {
        "beats_Z1_Z2_by_RMSE": bool(float(combined["RMSE"]) < float(z_only["RMSE"])),
        "beats_Z1_Z2_by_R2": bool(float(combined["R2"]) > float(z_only["R2"])),
        "beats_AMV_AMO_K5_by_RMSE": bool(float(combined["RMSE"]) < float(a_only["RMSE"])),
        "beats_AMV_AMO_K5_by_R2": bool(float(combined["R2"]) > float(a_only["R2"])),
        "improves_sign_accuracy_over_Z1_Z2": bool(float(combined["sign_accuracy"]) > float(z_only["sign_accuracy"])),
        "improves_sign_accuracy_over_AMV_AMO_K5": bool(float(combined["sign_accuracy"]) > float(a_only["sign_accuracy"])),
        "delta_RMSE_vs_Z1_Z2": float(combined["RMSE"] - z_only["RMSE"]),
        "delta_RMSE_vs_AMV_AMO_K5": float(combined["RMSE"] - a_only["RMSE"]),
        "delta_R2_vs_Z1_Z2": float(combined["R2"] - z_only["R2"]),
        "delta_R2_vs_AMV_AMO_K5": float(combined["R2"] - a_only["R2"]),
        "delta_sign_accuracy_vs_Z1_Z2": float(combined["sign_accuracy"] - z_only["sign_accuracy"]),
        "delta_sign_accuracy_vs_AMV_AMO_K5": float(combined["sign_accuracy"] - a_only["sign_accuracy"]),
    }

    if float(combined["RMSE"]) < min(float(z_only["RMSE"]), float(a_only["RMSE"])) and float(combined["R2"]) > max(float(z_only["R2"]), float(a_only["R2"])):
        behavior = "The combined model improves on both component models."
    elif float(combined["RMSE"]) > max(float(z_only["RMSE"]), float(a_only["RMSE"])) and float(combined["R2"]) < min(float(z_only["R2"]), float(a_only["R2"])):
        behavior = "The combined model is worse than both component models."
    elif abs(float(combined["RMSE"]) - float(z_only["RMSE"])) < abs(float(combined["RMSE"]) - float(a_only["RMSE"])):
        behavior = "The combined model behaves more like Z1_Z2."
    else:
        behavior = "The combined model behaves more like AMV/AMO K5."

    short_answer = (
        "Combining oracle Z1_Z2 with reduced AMV/AMO K5 "
        + ("improves" if combined_vs_components["beats_Z1_Z2_by_RMSE"] or combined_vs_components["beats_Z1_Z2_by_R2"] else "does not improve")
        + " strict LOYO SWE prediction relative to Z1_Z2 alone. "
        + behavior
    )

    return {
        "input_files": input_files,
        "output_dir": str(OUTPUT_DIR),
        "model_feature_sets": MODEL_FEATURES,
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_metrics_df.to_dict(orient="records"),
        "selected_alpha_summary": alpha_summary,
        "coefficient_summary": coefficient_summary,
        "best_model_by_R2": best_r2_row,
        "best_model_by_RMSE": best_rmse_row,
        "best_model_by_sign_accuracy": best_sign_row,
        "combined_vs_components": combined_vs_components,
        "short_answer": short_answer,
    }


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, input_files = load_predictor_table()
    table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    pred_df, alpha_df, beta_df = run_models(table)
    metrics_df = compute_metrics(pred_df)
    period_metrics_df = compute_period_metrics(pred_df)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_metrics_df.to_csv(PERIOD_METRICS_CSV, index=False)

    make_timeseries_plot(pred_df)
    make_scatter_plot(pred_df, metrics_df)
    make_error_plot(pred_df)

    summary = build_summary(input_files, metrics_df, period_metrics_df, alpha_df, beta_df)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(f"Output directory: {OUTPUT_DIR}")
    print("Metrics:")
    for model_name in PLOT_ORDER:
        row = metrics_df[metrics_df["model_name"] == model_name].iloc[0]
        print(
            f"{model_name}: r={float(row['r']):.6f}, R2={float(row['R2']):.6f}, "
            f"RMSE={float(row['RMSE']):.6f}, MAE={float(row['MAE']):.6f}, "
            f"sign_accuracy={float(row['sign_accuracy']):.6f}"
        )
    print(f"Best model by R2: {summary['best_model_by_R2']['model_name']}")
    print(f"Best model by RMSE: {summary['best_model_by_RMSE']['model_name']}")
    print(f"Best model by sign accuracy: {summary['best_model_by_sign_accuracy']['model_name']}")
    print("Short answer:")
    print(summary["short_answer"])


if __name__ == "__main__":
    main()
