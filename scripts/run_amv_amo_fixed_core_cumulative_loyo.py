#!/usr/bin/env python3
"""
Strict LOYO ridge test for a fixed cumulative AMV/AMO core versus the full 42-feature block.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    / "amv_amo_fixed_core_cumulative_loyo"
)

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "amv_amo_fixed_core_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "amv_amo_fixed_core_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "amv_amo_fixed_core_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "amv_amo_fixed_core_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "amv_amo_fixed_core_selected_alpha_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "amv_amo_fixed_core_beta_by_fold.csv"
SUMMARY_JSON = OUTPUT_DIR / "amv_amo_fixed_core_summary.json"
METRICS_PNG = OUTPUT_DIR / "amv_amo_fixed_core_metrics_by_K.png"
OBS_PRED_PNG = OUTPUT_DIR / "amv_amo_fixed_core_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "amv_amo_fixed_core_scatter.png"
ERROR_PNG = OUTPUT_DIR / "amv_amo_fixed_core_error_by_year.png"

SUBSET_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_subset_selection_diagnostic"
    / "amv_amo_predictor_table.csv"
)
AMV_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)
BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)
FULL_BLOCK_METRICS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "final_sst_only_linear_closure_table"
    / "final_sst_only_loyo_metrics.csv"
)

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
ALL_AMV_COLUMNS = [f"AMV_PC{pc}_{month}" for pc in range(1, 7) for month in MONTHS]
CORE_FEATURES = ["AMV_PC4_Sep", "AMV_PC5_Feb", "AMV_PC2_Feb", "AMV_PC4_Nov"]
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
MODEL_FEATURES = {
    "AMV_core_K1": CORE_FEATURES[:1],
    "AMV_core_K2": CORE_FEATURES[:2],
    "AMV_core_K3": CORE_FEATURES[:3],
    "AMV_core_K4": CORE_FEATURES[:4],
    "AMV_AMO_PC1to6_full": list(ALL_AMV_COLUMNS),
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
    x_train_std = (x_train_raw - x_mean[None, :]) / x_std[None, :]
    x_test_std = (x_test_raw - x_mean) / x_std
    return x_train_std, x_test_std, x_mean, x_std


def standardize_target_train_only(
    y_train_raw: np.ndarray, y_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Target train-fold standard deviation must be positive.")
    y_train_std = (y_train_raw - y_mean) / y_std
    y_test_std = (y_test_raw - y_mean) / y_std
    return y_train_std, y_test_std, y_mean, y_std


def ridge_fit_predict_standardized(
    x_train_std: np.ndarray,
    y_train_std: np.ndarray,
    x_test_std: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    pred_std = x_test_std @ coef
    return pred_std, coef


def evaluate_alpha_grid_loyo(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    n = x.shape[0]
    pred_by_alpha = {alpha: np.full(n, np.nan, dtype=float) for alpha in ALPHA_GRID}
    for held_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[held_idx] = False
        x_train = x[train_mask, :]
        x_test = x[~train_mask, :][0]
        y_train = y[train_mask]
        y_test = y[~train_mask][0]
        x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
        y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train, np.asarray([y_test]))
        for alpha in ALPHA_GRID:
            pred_std, _ = ridge_fit_predict_standardized(
                x_train_std, y_train_std, x_test_std.reshape(1, -1), alpha
            )
            pred_by_alpha[alpha][held_idx] = y_mean + y_std * float(pred_std[0])
    rmse_by_alpha = {alpha: rmse(y, preds) for alpha, preds in pred_by_alpha.items()}
    best_alpha = min(ALPHA_GRID, key=lambda a: (rmse_by_alpha[a], a))
    return float(best_alpha), float(rmse_by_alpha[best_alpha]), pred_by_alpha[best_alpha]


def fit_outer_prediction(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> Tuple[float, np.ndarray, float]:
    x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
    y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train, np.asarray([0.0]))
    pred_std, coef = ridge_fit_predict_standardized(
        x_train_std, y_train_std, x_test_std.reshape(1, -1), alpha
    )
    pred_raw = y_mean + y_std * float(pred_std[0])
    intercept_raw = y_mean - float(np.sum(coef * (np.mean(x_train, axis=0) / np.std(x_train, axis=0, ddof=1))) * y_std)
    return pred_raw, coef, intercept_raw


def build_predictor_table() -> pd.DataFrame:
    if SUBSET_TABLE_CSV.exists():
        table = pd.read_csv(SUBSET_TABLE_CSV)
        needed = ["water_year", "obs_swe"] + CORE_FEATURES + ALL_AMV_COLUMNS
        missing = [c for c in needed if c not in table.columns]
        if missing:
            raise ValueError("Subset predictor table is missing columns: {}".format(missing))
        table = table[needed].copy()
    else:
        base_predictions = pd.read_csv(BASE_PREDICTIONS_CSV)
        base_predictions = base_predictions.loc[
            (base_predictions["patch_size"] == "exact_grid_cell")
            & (base_predictions["model_name"] == "Z1_Z2")
        ].copy()
        targets = (
            base_predictions[["heldout_wy", "obs_swe"]]
            .rename(columns={"heldout_wy": "water_year"})
            .sort_values("water_year")
            .reset_index(drop=True)
        )
        amv = pd.read_csv(AMV_CSV)
        table = targets.merge(amv, on="water_year", how="inner")
        table = table[["water_year", "obs_swe"] + ALL_AMV_COLUMNS].copy()
    table["water_year"] = table["water_year"].astype(int)
    table = table.sort_values("water_year").reset_index(drop=True)
    table = table.loc[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Predictor table does not match WY1985--WY2021.")
    ordered = ["water_year", "obs_swe"] + CORE_FEATURES + [c for c in ALL_AMV_COLUMNS if c not in CORE_FEATURES]
    return table[ordered].copy()


def run_loyo_models(table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    y = table["obs_swe"].to_numpy(dtype=float)
    prediction_rows = []  # type: List[Dict[str, object]]
    alpha_rows = []  # type: List[Dict[str, object]]
    beta_rows = []  # type: List[Dict[str, object]]

    for model_name, feature_list in MODEL_FEATURES.items():
        x_all = table[feature_list].to_numpy(dtype=float)
        for held_idx, held_year in enumerate(years):
            train_mask = np.ones(len(years), dtype=bool)
            train_mask[held_idx] = False
            x_train = x_all[train_mask, :]
            x_test = x_all[~train_mask, :][0]
            y_train = y[train_mask]
            obs = y[held_idx]

            alpha, inner_cv_rmse, _ = evaluate_alpha_grid_loyo(x_train, y_train)
            pred_raw, coef, intercept_raw = fit_outer_prediction(x_train, y_train, x_test, alpha)
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
                    "inner_cv_mse": float(inner_cv_rmse ** 2),
                }
            )
            beta_row = {
                "model_name": model_name,
                "heldout_wy": int(held_year),
                "intercept": float(intercept_raw),
            }
            for feature in CORE_FEATURES:
                beta_row["beta_" + feature] = float(coef[feature_list.index(feature)]) if feature in feature_list else np.nan
            if model_name == "AMV_AMO_PC1to6_full":
                for feature in ALL_AMV_COLUMNS:
                    beta_row["beta_" + feature] = float(coef[feature_list.index(feature)])
            beta_rows.append(beta_row)

    return (
        pd.DataFrame(prediction_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
        pd.DataFrame(alpha_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
        pd.DataFrame(beta_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
    )


def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_FEATURES:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        metrics = compute_metric_bundle(
            sub["obs_swe"].to_numpy(dtype=float), sub["pred_swe"].to_numpy(dtype=float)
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
    for model_name in MODEL_FEATURES:
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
    return pd.DataFrame(rows)


def summarize_alpha(alpha_df: pd.DataFrame) -> Dict[str, object]:
    summary = {}
    for model_name in MODEL_FEATURES:
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
    summary = {}
    for model_name in MODEL_FEATURES:
        sub = beta_df[beta_df["model_name"] == model_name]
        model_summary = {}
        for feature in CORE_FEATURES:
            col = "beta_" + feature
            vals = sub[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            model_summary[feature] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "num_positive": int((vals > 0).sum()),
                "num_negative": int((vals < 0).sum()),
            }
        summary[model_name] = model_summary
    return summary


def load_full_reference_metrics() -> Optional[Dict[str, float]]:
    if not FULL_BLOCK_METRICS_CSV.exists():
        return None
    metrics = pd.read_csv(FULL_BLOCK_METRICS_CSV)
    ref = metrics[metrics["model_name"] == "AMV_AMO_PC1to6_only"]
    if ref.empty:
        return None
    row = ref.iloc[0]
    return {
        "R2": float(row["R2"]),
        "RMSE": float(row["RMSE"]),
        "sign_accuracy": float(row["sign_accuracy"]),
        "r": float(row["r"]),
        "MAE": float(row["MAE"]),
    }


def make_metrics_plot(metrics_df: pd.DataFrame) -> None:
    order = ["AMV_core_K1", "AMV_core_K2", "AMV_core_K3", "AMV_core_K4", "AMV_AMO_PC1to6_full"]
    display = ["K1", "K2", "K3", "K4", "full42"]
    sub = metrics_df.set_index("model_name").loc[order].reset_index()
    x = np.arange(len(order))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    axes[0].plot(x, sub["R2"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("$R^2$")
    axes[0].set_title("Fixed-core cumulative AMV/AMO metrics")
    axes[1].plot(x, sub["RMSE"], marker="o", color="#d62728")
    axes[1].set_ylabel("RMSE (m)")
    axes[2].plot(x, sub["sign_accuracy"], marker="o", color="#2ca02c")
    axes[2].set_ylabel("Sign accuracy")
    axes[2].set_xlabel("Model")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(METRICS_PNG, dpi=200)
    plt.close(fig)


def make_observed_vs_predicted_plot(pred_df: pd.DataFrame) -> None:
    years = WATER_YEARS
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    obs = pred_df[pred_df["model_name"] == "AMV_core_K1"]["obs_swe"].to_numpy(dtype=float)
    ax.plot(years, obs, color="black", linewidth=2.4, label="Observed")
    for model_name, color in [
        ("AMV_core_K1", "#1f77b4"),
        ("AMV_core_K4", "#ff7f0e"),
        ("AMV_AMO_PC1to6_full", "#d62728"),
    ]:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["pred_swe"],
            linewidth=1.8,
            marker="o",
            label=model_name,
            color=color,
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Observed vs predicted for fixed-core cumulative AMV/AMO models")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OBS_PRED_PNG, dpi=200)
    plt.close(fig)


def make_scatter_plot(pred_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharex=True, sharey=True)
    models = ["AMV_core_K1", "AMV_core_K4", "AMV_AMO_PC1to6_full"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    all_obs = pred_df["obs_swe"].to_numpy(dtype=float)
    all_pred = pred_df["pred_swe"].to_numpy(dtype=float)
    lo = min(np.min(all_obs), np.min(all_pred))
    hi = max(np.max(all_obs), np.max(all_pred))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lims = (lo - pad, hi + pad)
    for ax, model_name, color in zip(axes, models, colors):
        sub = pred_df[pred_df["model_name"] == model_name]
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        metrics = compute_metric_bundle(obs, pred)
        ax.scatter(obs, pred, color=color, s=48, alpha=0.85)
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(model_name)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.text(
            0.04,
            0.96,
            "RMSE = {:.4f}\n$R^2$ = {:.3f}\nCorr = {:.3f}".format(
                metrics["RMSE"], metrics["R2"], metrics["r"]
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
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.axhline(0.0, color="black", linewidth=1.2)
    for model_name, color in [
        ("AMV_core_K1", "#1f77b4"),
        ("AMV_core_K4", "#ff7f0e"),
        ("AMV_AMO_PC1to6_full", "#d62728"),
    ]:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["error_pred_minus_obs"],
            linewidth=1.8,
            marker="o",
            label=model_name,
            color=color,
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Prediction error (m)")
    ax.set_title("Prediction error by held-out year for fixed-core AMV/AMO models")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(ERROR_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictor_table = build_predictor_table()
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    pred_df, alpha_df, beta_df = run_loyo_models(predictor_table)
    metrics_df = compute_metrics(pred_df)
    period_df = compute_period_metrics(pred_df)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)

    make_metrics_plot(metrics_df)
    make_observed_vs_predicted_plot(pred_df)
    make_scatter_plot(pred_df)
    make_error_plot(pred_df)

    alpha_summary = summarize_alpha(alpha_df)
    coefficient_summary = summarize_coefficients(beta_df)
    best_by_r2 = metrics_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0].to_dict()
    best_by_rmse = metrics_df.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0].to_dict()
    best_by_sign = metrics_df.sort_values(["sign_accuracy", "RMSE"], ascending=[False, True]).iloc[0].to_dict()
    full_reference = load_full_reference_metrics()

    k4_metrics = metrics_df[metrics_df["model_name"] == "AMV_core_K4"].iloc[0].to_dict()
    full_metrics = metrics_df[metrics_df["model_name"] == "AMV_AMO_PC1to6_full"].iloc[0].to_dict()

    if abs(k4_metrics["RMSE"] - full_metrics["RMSE"]) <= 0.002 and abs(k4_metrics["R2"] - full_metrics["R2"]) <= 0.05:
        short_answer = (
            "The fixed four-feature AMV core captures most of the full 42-column AMV/AMO ridge skill, "
            "so it is a useful low-dimensional AMV representation."
        )
    elif best_by_rmse["model_name"] != "AMV_core_K4":
        short_answer = (
            "The fixed AMV core peaks at K={}; adding later core features does not improve prediction, "
            "and the full 42-feature block remains stronger overall.".format(best_by_rmse["model_name"].split("_K")[-1] if "K" in best_by_rmse["model_name"] else "full42")
        )
    else:
        short_answer = (
            "The fixed four-feature AMV core is interpretable and stable, but it does not reproduce the "
            "full AMV/AMO ridge skill. The full-AMV skill likely draws on distributed information across "
            "many weak predictors."
        )

    summary = {
        "input_files": {
            "predictor_table_source": str(SUBSET_TABLE_CSV if SUBSET_TABLE_CSV.exists() else AMV_CSV),
            "base_predictions_csv": str(BASE_PREDICTIONS_CSV),
            "full_reference_metrics_csv": str(FULL_BLOCK_METRICS_CSV),
        },
        "output_dir": str(OUTPUT_DIR),
        "water_years": WATER_YEARS,
        "fixed_core_features": CORE_FEATURES,
        "model_metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_df.to_dict(orient="records"),
        "selected_alpha_summary": alpha_summary,
        "coefficient_summary": coefficient_summary,
        "best_model_by_R2": best_by_r2,
        "best_model_by_RMSE": best_by_rmse,
        "best_model_by_sign_accuracy": best_by_sign,
        "comparison_to_full_AMV_reference": {
            "computed_full42_metrics": full_metrics,
            "prior_reference_metrics": full_reference,
        },
        "short_answer": short_answer,
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Output directory: {}".format(OUTPUT_DIR))
    print("Core feature order:")
    for idx, feature in enumerate(CORE_FEATURES, start=1):
        print("{}. {}".format(idx, feature))
    print("Metrics:")
    print(metrics_df.to_string(index=False))
    print("Best model by R2:")
    print(best_by_r2)
    print("Best model by RMSE:")
    print(best_by_rmse)
    print("Best model by sign accuracy:")
    print(best_by_sign)
    print("Short answer:")
    print(short_answer)


if __name__ == "__main__":
    main()
