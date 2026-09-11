#!/usr/bin/env python3
"""
Run nested forward feature selection for PNA/NAO/NAM SLP PC-month predictor blocks.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "era5_slp_nested_forward_selection"
INPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "era5_slp_pc_loyo"
PREDICTOR_TABLE_CSV = INPUT_DIR / "slp_pc_predictor_table.csv"
FULL42_PREDICTIONS_CSV = INPUT_DIR / "slp_pc_loyo_predictions.csv"
FULL42_METRICS_CSV = INPUT_DIR / "slp_pc_loyo_metrics.csv"

PREDICTIONS_CSV = OUTPUT_DIR / "slp_nested_forward_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "slp_nested_forward_metrics.csv"
SELECTED_BY_FOLD_CSV = OUTPUT_DIR / "slp_nested_forward_selected_features_by_fold.csv"
SELECTION_PATH_SUMMARY_CSV = OUTPUT_DIR / "slp_nested_forward_selection_path_summary.csv"
SELECTION_FREQ_CSV = OUTPUT_DIR / "slp_nested_forward_selection_frequency.csv"
SUMMARY_JSON = OUTPUT_DIR / "slp_nested_forward_summary.json"
METRICS_BY_K_PNG = OUTPUT_DIR / "slp_nested_forward_metrics_by_K.png"
OBS_PRED_PNG = OUTPUT_DIR / "slp_nested_forward_observed_vs_predicted.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
K_MAX = 5
DOMAIN_PREFIX = {"PNA_SLP": "PNA", "NAO_SLP": "NAO", "NAM_SLP": "NAM"}
DOMAIN_MODELS = {
    "PNA_SLP": "PNA_SLP_PC1_6",
    "NAO_SLP": "NAO_SLP_PC1_6",
    "NAM_SLP": "NAM_SLP_PC1_6",
}
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
N_WORKERS = max(1, min(12, int(os.environ.get("SLP_NESTED_FORWARD_N_WORKERS", "12"))))


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


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha: Optional[float] = None
    best_mse: Optional[float] = None
    best_r2: Optional[float] = None
    for alpha in ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=float)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            x_inner_train = x_train_raw[mask, :]
            x_inner_test = x_train_raw[~mask, :][0]
            y_inner_train = y_train_raw[mask]
            x_inner_train_std, x_inner_test_std, _, _ = standardize_train_only(x_inner_train, x_inner_test)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train)
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, alpha)
            pred_std = float(x_inner_test_std @ beta_std)
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        r2 = float(r2_manual(y_train_raw, preds))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = mse
            best_r2 = r2
    assert best_alpha is not None and best_mse is not None and best_r2 is not None
    return best_alpha, best_mse, best_r2


def fit_outer_prediction(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> float:
    x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train)
    coef = fit_ridge_standardized(x_train_std, y_train_std, alpha)
    pred_std = float(x_test_std @ coef)
    return float(y_mean + y_std * pred_std)


def domain_features(domain_name: str) -> List[str]:
    prefix = DOMAIN_PREFIX[domain_name]
    return [f"{prefix}_PC{pc}_{month}" for month in MONTHS for pc in range(1, 7)]


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(PREDICTOR_TABLE_CSV)
    table["water_year"] = table["water_year"].astype(int)
    table = table[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    table = table.sort_values("water_year").reset_index(drop=True)
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("SLP predictor table does not match WY1985--WY2021.")
    full_preds = pd.read_csv(FULL42_PREDICTIONS_CSV)
    full_metrics = pd.read_csv(FULL42_METRICS_CSV)
    return table, full_preds, full_metrics


def run_outer_fold(
    domain_name: str,
    held_idx: int,
    years: np.ndarray,
    y_all: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    heldout_year = int(years[held_idx])
    outer_train_mask = np.ones(len(years), dtype=bool)
    outer_train_mask[held_idx] = False
    x_outer_train = feature_matrix[outer_train_mask, :]
    y_outer_train = y_all[outer_train_mask]
    x_outer_test = feature_matrix[~outer_train_mask, :][0]
    obs_test = float(y_all[held_idx])

    remaining = list(range(len(feature_names)))
    selected: List[int] = []
    prediction_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    outer_errors_by_k: Dict[int, float] = {}
    final_k_inner_rmse = np.nan

    for step in range(1, K_MAX + 1):
        best = None
        for candidate in remaining:
            candidate_set = selected + [candidate]
            x_candidate = x_outer_train[:, candidate_set]
            alpha, inner_rmse, inner_r2 = inner_loyo_best_alpha(x_candidate, y_outer_train)
            score = (inner_rmse, -inner_r2, alpha, candidate)
            if best is None or score < best["score"]:
                best = {
                    "candidate": candidate,
                    "alpha": alpha,
                    "inner_rmse": inner_rmse,
                    "inner_r2": inner_r2,
                    "score": score,
                }
        assert best is not None
        selected.append(int(best["candidate"]))
        remaining.remove(int(best["candidate"]))
        selected_feature_names = [feature_names[idx] for idx in selected]
        pred_raw = fit_outer_prediction(
            x_outer_train[:, selected],
            y_outer_train,
            x_outer_test[selected],
            float(best["alpha"]),
        )
        err = pred_raw - obs_test
        outer_errors_by_k[step] = float(abs(err))
        prediction_rows.append(
            {
                "domain_name": domain_name,
                "model_name": f"{DOMAIN_MODELS[domain_name]}_nested_K{step}",
                "heldout_wy": heldout_year,
                "K": int(step),
                "num_predictors": int(step),
                "selected_features": "|".join(selected_feature_names),
                "obs_swe": obs_test,
                "pred_swe": float(pred_raw),
                "error_pred_minus_obs": float(err),
                "residual_obs_minus_pred": float(-err),
                "abs_error": float(abs(err)),
                "sign_correct": float(np.sign(pred_raw) == np.sign(obs_test)) if obs_test != 0.0 and pred_raw != 0.0 else np.nan,
                "selected_alpha_final": float(best["alpha"]),
                "inner_cv_RMSE_for_selected_set": float(best["inner_rmse"]),
                "inner_cv_R2_for_selected_set": float(best["inner_r2"]),
            }
        )
        selected_rows.append(
            {
                "domain_name": domain_name,
                "heldout_wy": heldout_year,
                "K": int(step),
                "selected_feature_at_step": feature_names[int(best["candidate"])],
                "selected_features_so_far": "|".join(selected_feature_names),
                "inner_cv_RMSE": float(best["inner_rmse"]),
                "inner_cv_R2": float(best["inner_r2"]),
                "selected_alpha": float(best["alpha"]),
            }
        )
        if step == K_MAX:
            final_k_inner_rmse = float(best["inner_rmse"])

    path_row = {
        "domain_name": domain_name,
        "heldout_wy": heldout_year,
        "selected_path": "|".join(feature_names[idx] for idx in selected),
        "final_K5_inner_cv_RMSE": final_k_inner_rmse,
    }
    for step in range(1, K_MAX + 1):
        path_row[f"K{step}_feature"] = feature_names[selected[step - 1]]
        path_row[f"outer_K{step}_abs_error"] = outer_errors_by_k[step]
    return prediction_rows, selected_rows, path_row


def run_nested_forward_for_domain(table: pd.DataFrame, domain_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = domain_features(domain_name)
    y_all = table["obs_swe"].to_numpy(dtype=float)
    years = table["water_year"].to_numpy(dtype=int)
    feature_matrix = table[feature_names].to_numpy(dtype=float)

    prediction_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    path_rows: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(years))) as executor:
        futures = {
            executor.submit(
                run_outer_fold,
                domain_name,
                held_idx,
                years,
                y_all,
                feature_matrix,
                feature_names,
            ): held_idx
            for held_idx in range(len(years))
        }
        for future in as_completed(futures):
            pred_rows, sel_rows, path_row = future.result()
            prediction_rows.extend(pred_rows)
            selected_rows.extend(sel_rows)
            path_rows.append(path_row)
            print(f"{domain_name}: finished outer fold WY{path_row['heldout_wy']}", flush=True)

    predictions_df = pd.DataFrame(prediction_rows).sort_values(["heldout_wy", "K"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["heldout_wy", "K"]).reset_index(drop=True)
    path_df = pd.DataFrame(path_rows).sort_values("heldout_wy").reset_index(drop=True)
    return predictions_df, selected_df, path_df


def compute_metrics(predictions_df: pd.DataFrame, full_preds: pd.DataFrame, full_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for domain_name, full_model_name in DOMAIN_MODELS.items():
        for k in range(1, K_MAX + 1):
            sub = predictions_df[(predictions_df["domain_name"] == domain_name) & (predictions_df["K"] == k)].copy()
            metrics = compute_metric_bundle(sub["obs_swe"].to_numpy(dtype=float), sub["pred_swe"].to_numpy(dtype=float))
            rows.append(
                {
                    "domain_name": domain_name,
                    "model_name": f"{full_model_name}_nested_K{k}",
                    "comparison_group": f"nested_K{k}",
                    "K": int(k),
                    "num_predictors": int(k),
                    **metrics,
                }
            )
        full_metric_row = full_metrics[full_metrics["model_name"] == full_model_name].iloc[0]
        full_pred_sub = full_preds[full_preds["model_name"] == full_model_name].copy()
        metrics = compute_metric_bundle(full_pred_sub["obs_swe"].to_numpy(dtype=float), full_pred_sub["pred_swe"].to_numpy(dtype=float))
        rows.append(
            {
                "domain_name": domain_name,
                "model_name": full_model_name,
                "comparison_group": "full42",
                "K": 42,
                "num_predictors": int(full_metric_row["num_predictors"]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def compute_selection_frequency(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for domain_name in DOMAIN_MODELS:
        features = domain_features(domain_name)
        domain_selected = selected_df[selected_df["domain_name"] == domain_name].copy()
        for feature in features:
            subset = domain_selected[domain_selected["selected_feature_at_step"] == feature]
            steps = subset["K"].to_numpy(dtype=int)
            any_selected = domain_selected[domain_selected["selected_features_so_far"].str.contains(feature, regex=False)]
            selected_any_years = any_selected["heldout_wy"].nunique()
            rows.append(
                {
                    "domain_name": domain_name,
                    "feature": feature,
                    "selected_at_K1_count": int((steps == 1).sum()),
                    "selected_at_K2_count": int((steps == 2).sum()),
                    "selected_at_K3_count": int((steps == 3).sum()),
                    "selected_at_K4_count": int((steps == 4).sum()),
                    "selected_at_K5_count": int((steps == 5).sum()),
                    "selected_any_count": int(selected_any_years),
                    "selected_any_fraction": float(selected_any_years / len(WATER_YEARS)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["domain_name", "selected_any_count", "selected_at_K1_count", "feature"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def plot_metrics_by_k(metrics_df: pd.DataFrame) -> None:
    domains = list(DOMAIN_MODELS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.0), constrained_layout=True)
    for row_idx, domain_name in enumerate(domains):
        sub = metrics_df[metrics_df["domain_name"] == domain_name].copy()
        nested = sub[sub["comparison_group"].str.startswith("nested_")].sort_values("K")
        full = sub[sub["comparison_group"] == "full42"].iloc[0]

        axes[row_idx, 0].plot(nested["K"], nested["R2"], marker="o", color="#1f77b4")
        axes[row_idx, 0].axhline(float(full["R2"]), color="#d62728", linestyle="--", linewidth=1.3)
        axes[row_idx, 0].set_ylabel(f"{domain_name}\n$R^2$")

        axes[row_idx, 1].plot(nested["K"], nested["RMSE"], marker="o", color="#2ca02c")
        axes[row_idx, 1].axhline(float(full["RMSE"]), color="#d62728", linestyle="--", linewidth=1.3)
        axes[row_idx, 1].set_ylabel("RMSE (m)")

        axes[row_idx, 2].plot(nested["K"], nested["sign_accuracy"], marker="o", color="#9467bd")
        axes[row_idx, 2].axhline(float(full["sign_accuracy"]), color="#d62728", linestyle="--", linewidth=1.3)
        axes[row_idx, 2].set_ylabel("Sign accuracy")

        for col_idx in range(3):
            axes[row_idx, col_idx].set_xticks(range(1, K_MAX + 1))
            axes[row_idx, col_idx].grid(alpha=0.25)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(["Nested $R^2$", "Nested RMSE", "Nested sign accuracy"][col_idx])
            if row_idx == len(domains) - 1:
                axes[row_idx, col_idx].set_xlabel("Nested K")
    fig.savefig(METRICS_BY_K_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_predicted(table: pd.DataFrame, predictions_df: pd.DataFrame, full_preds: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    domains = list(DOMAIN_MODELS.keys())
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 10.0), constrained_layout=True, sharex=True)
    for ax, domain_name in zip(axes, domains):
        obs = table[["water_year", "obs_swe"]].copy()
        domain_metrics = metrics_df[(metrics_df["domain_name"] == domain_name) & (metrics_df["comparison_group"].str.startswith("nested_"))].copy()
        best_k_row = domain_metrics.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
        best_k = int(best_k_row["K"])
        for k in sorted(set([1, best_k, 5])):
            sub = predictions_df[
                (predictions_df["domain_name"] == domain_name) & (predictions_df["K"] == k)
            ][["heldout_wy", "pred_swe"]].rename(columns={"heldout_wy": "water_year", "pred_swe": f"K{k}"})
            obs = obs.merge(sub, on="water_year", how="left")
        full_sub = full_preds[full_preds["model_name"] == DOMAIN_MODELS[domain_name]][["heldout_wy", "pred_swe"]].rename(
            columns={"heldout_wy": "water_year", "pred_swe": "full42"}
        )
        obs = obs.merge(full_sub, on="water_year", how="left")

        ax.plot(obs["water_year"], obs["obs_swe"], color="black", linewidth=2.4, label="Observed")
        colors = {1: "#1f77b4", best_k: "#ff7f0e", 5: "#2ca02c"}
        for k in sorted(set([1, best_k, 5])):
            ax.plot(obs["water_year"], obs[f"K{k}"], linewidth=1.6, label=f"nested K{k}", color=colors[k])
        ax.plot(obs["water_year"], obs["full42"], linewidth=1.7, linestyle="--", color="#d62728", label="full42")
        ax.set_ylabel(f"{domain_name}\nSWE (m)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Held-out water year")
    fig.savefig(OBS_PRED_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(metrics_df: pd.DataFrame, selection_freq_df: pd.DataFrame) -> Dict[str, object]:
    per_domain: Dict[str, object] = {}
    for domain_name in DOMAIN_MODELS:
        sub = metrics_df[metrics_df["domain_name"] == domain_name].copy()
        nested = sub[sub["comparison_group"].str.startswith("nested_")].copy()
        full = sub[sub["comparison_group"] == "full42"].iloc[0].to_dict()
        best_rmse = nested.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0].to_dict()
        best_r2 = nested.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0].to_dict()
        top_features = selection_freq_df[selection_freq_df["domain_name"] == domain_name].head(10).to_dict(orient="records")
        per_domain[domain_name] = {
            "full42": full,
            "best_nested_by_RMSE": best_rmse,
            "best_nested_by_R2": best_r2,
            "top_selection_frequency": top_features,
        }
    return per_domain


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, full_preds, full_metrics = load_inputs()
    all_predictions: List[pd.DataFrame] = []
    all_selected: List[pd.DataFrame] = []
    all_paths: List[pd.DataFrame] = []

    for domain_name in DOMAIN_MODELS:
        print(f"Running nested forward selection for {domain_name}", flush=True)
        predictions_df, selected_df, path_df = run_nested_forward_for_domain(table, domain_name)
        all_predictions.append(predictions_df)
        all_selected.append(selected_df)
        all_paths.append(path_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    selected_df = pd.concat(all_selected, ignore_index=True)
    path_df = pd.concat(all_paths, ignore_index=True)

    full_ref = full_preds[["model_name", "heldout_wy", "obs_swe", "pred_swe", "error_pred_minus_obs", "residual_obs_minus_pred", "abs_error", "sign_correct", "selected_alpha", "num_predictors"]].copy()
    full_ref["domain_name"] = full_ref["model_name"].map({v: k for k, v in DOMAIN_MODELS.items()})
    full_ref["K"] = 42
    full_ref["selected_features"] = "ALL_42"
    full_ref["selected_alpha_final"] = full_ref["selected_alpha"]
    full_ref["inner_cv_RMSE_for_selected_set"] = np.nan
    full_ref["inner_cv_R2_for_selected_set"] = np.nan
    full_ref["model_name"] = full_ref["model_name"]
    full_ref = full_ref[
        [
            "domain_name",
            "model_name",
            "heldout_wy",
            "K",
            "num_predictors",
            "selected_features",
            "obs_swe",
            "pred_swe",
            "error_pred_minus_obs",
            "residual_obs_minus_pred",
            "abs_error",
            "sign_correct",
            "selected_alpha_final",
            "inner_cv_RMSE_for_selected_set",
            "inner_cv_R2_for_selected_set",
        ]
    ]

    predictions_out = pd.concat([predictions_df, full_ref], ignore_index=True).sort_values(["domain_name", "heldout_wy", "K"]).reset_index(drop=True)
    metrics_df = compute_metrics(predictions_df, full_preds, full_metrics)
    selection_freq_df = compute_selection_frequency(selected_df)

    predictions_out.to_csv(PREDICTIONS_CSV, index=False)
    selected_df.to_csv(SELECTED_BY_FOLD_CSV, index=False)
    path_df.to_csv(SELECTION_PATH_SUMMARY_CSV, index=False)
    selection_freq_df.to_csv(SELECTION_FREQ_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)

    plot_metrics_by_k(metrics_df)
    plot_observed_vs_predicted(table, predictions_df, full_preds, metrics_df)

    summary = {
        "input_files": {
            "predictor_table_csv": str(PREDICTOR_TABLE_CSV),
            "full42_predictions_csv": str(FULL42_PREDICTIONS_CSV),
            "full42_metrics_csv": str(FULL42_METRICS_CSV),
        },
        "output_dir": str(OUTPUT_DIR),
        "alpha_grid": ALPHA_GRID.tolist(),
        "nested_K_values": list(range(1, K_MAX + 1)),
        "per_domain": build_summary(metrics_df, selection_freq_df),
        "short_answer": {
            domain_name: (
                f"full42 R2={metrics_df[(metrics_df['domain_name'] == domain_name) & (metrics_df['comparison_group'] == 'full42')]['R2'].iloc[0]:.3f}; "
                f"best nested-by-RMSE is K={int(metrics_df[(metrics_df['domain_name'] == domain_name) & (metrics_df['comparison_group'].str.startswith('nested_'))].sort_values(['RMSE', 'R2'], ascending=[True, False]).iloc[0]['K'])}."
            )
            for domain_name in DOMAIN_MODELS
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    for domain_name in DOMAIN_MODELS:
        sub = metrics_df[metrics_df["domain_name"] == domain_name].copy().sort_values(["K"])
        print(domain_name, flush=True)
        print(sub[["comparison_group", "K", "R2", "RMSE", "sign_accuracy"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
