#!/usr/bin/env python3
"""
Diagnose whether the AMV/AMO PC1-PC6 Sep-Mar block has a stable reduced subset.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_subset_selection_diagnostic"
)

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "amv_amo_predictor_table.csv"
SINGLE_RANKING_CSV = OUTPUT_DIR / "amv_amo_single_variable_loyo_ranking.csv"
SINGLE_PREDICTIONS_CSV = OUTPUT_DIR / "amv_amo_single_variable_loyo_predictions.csv"
NESTED_PREDICTIONS_CSV = OUTPUT_DIR / "amv_amo_nested_forward_predictions.csv"
NESTED_METRICS_CSV = OUTPUT_DIR / "amv_amo_nested_forward_metrics.csv"
SELECTED_BY_FOLD_CSV = OUTPUT_DIR / "amv_amo_nested_forward_selected_features_by_fold.csv"
SELECTION_FREQ_CSV = OUTPUT_DIR / "amv_amo_selection_frequency.csv"
SELECTION_PATH_SUMMARY_CSV = OUTPUT_DIR / "amv_amo_selection_path_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "amv_amo_subset_selection_summary.json"

TOP20_SINGLE_PNG = OUTPUT_DIR / "amv_amo_single_variable_top20.png"
METRICS_BY_K_PNG = OUTPUT_DIR / "amv_amo_nested_forward_metrics_by_K.png"
SELECTION_FREQ_PNG = OUTPUT_DIR / "amv_amo_selection_frequency_top20.png"
OBS_PRED_PNG = OUTPUT_DIR / "amv_amo_nested_forward_observed_vs_predicted.png"

AMV_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
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
FULL_BLOCK_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "final_sst_only_linear_closure_table"
    / "final_sst_only_loyo_predictions.csv"
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
AMV_COLUMNS = [f"AMV_PC{pc}_{month}" for pc in range(1, 7) for month in MONTHS]
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
K_MAX = 5


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("Predictor train-fold standard deviation must be positive.")
    x_train_std = (x_train_raw - x_mean[None, :]) / x_std[None, :]
    x_test_std = (x_test_raw - x_mean) / x_std
    return x_train_std, x_test_std, x_mean, x_std


def standardize_target_train_only(
    y_train_raw: np.ndarray, y_test_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
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
) -> tuple[np.ndarray, np.ndarray]:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    pred_std = x_test_std @ coef
    return pred_std, coef


def evaluate_alpha_grid_loyo(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
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
    best_preds = pred_by_alpha[best_alpha]
    return float(best_alpha), float(rmse_by_alpha[best_alpha]), float(r2_manual(y, best_preds)), best_preds


def fit_outer_prediction(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> tuple[float, np.ndarray]:
    x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
    y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train, np.asarray([0.0]))
    pred_std, coef = ridge_fit_predict_standardized(
        x_train_std, y_train_std, x_test_std.reshape(1, -1), alpha
    )
    pred_raw = y_mean + y_std * float(pred_std[0])
    return pred_raw, coef


def build_predictor_table() -> pd.DataFrame:
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
    targets["water_year"] = targets["water_year"].astype(int)

    amv = pd.read_csv(AMV_CSV)
    amv["water_year"] = amv["water_year"].astype(int)
    missing = [c for c in AMV_COLUMNS if c not in amv.columns]
    if missing:
        raise ValueError(f"Missing expected AMV columns: {missing}")
    amv = amv[["water_year"] + AMV_COLUMNS].sort_values("water_year").reset_index(drop=True)

    merged = targets.merge(amv, on="water_year", how="inner")
    merged = merged.loc[
        (merged["water_year"] >= WATER_YEAR_START) & (merged["water_year"] <= WATER_YEAR_END)
    ].copy()
    if merged["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Merged AMV predictor table does not match WY1985-WY2021.")
    return merged


def run_single_variable_loyo(table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = table["obs_swe"].to_numpy(dtype=float)
    prediction_rows = []  # type: List[Dict[str, object]]
    ranking_rows = []  # type: List[Dict[str, object]]
    for feature in AMV_COLUMNS:
        x = table[[feature]].to_numpy(dtype=float)
        preds = np.full_like(y, np.nan)
        alpha_used = np.full_like(y, np.nan)
        for held_idx, year in enumerate(table["water_year"].to_numpy(dtype=int)):
            train_mask = np.ones(len(table), dtype=bool)
            train_mask[held_idx] = False
            x_train = x[train_mask, :]
            x_test = x[~train_mask, :][0]
            y_train = y[train_mask]
            best_alpha, _, _, _ = evaluate_alpha_grid_loyo(x_train, y_train)
            pred_raw, _ = fit_outer_prediction(x_train, y_train, x_test, best_alpha)
            preds[held_idx] = pred_raw
            alpha_used[held_idx] = best_alpha
            obs = y[held_idx]
            err = pred_raw - obs
            prediction_rows.append(
                {
                    "feature": feature,
                    "heldout_wy": int(year),
                    "obs_swe": float(obs),
                    "pred_swe": float(pred_raw),
                    "error_pred_minus_obs": float(err),
                    "residual_obs_minus_pred": float(-err),
                    "abs_error": float(abs(err)),
                    "sign_correct": float(np.sign(pred_raw) == np.sign(obs)) if obs != 0.0 and pred_raw != 0.0 else np.nan,
                    "selected_alpha": float(best_alpha),
                }
            )
        metrics = compute_metric_bundle(y, preds)
        pc, month = feature.split("_")[1], feature.split("_")[2]
        ranking_rows.append(
            {
                "feature": feature,
                "pc": pc,
                "month": month,
                "num_predictors": 1,
                "r": metrics["r"],
                "abs_r": abs(metrics["r"]) if np.isfinite(metrics["r"]) else np.nan,
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_error": metrics["mean_error"],
            }
        )
    predictions_df = pd.DataFrame(prediction_rows).sort_values(["feature", "heldout_wy"]).reset_index(drop=True)
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(["RMSE", "R2"], ascending=[True, False]).reset_index(drop=True)
    ranking_df["rank_by_R2"] = ranking_df["R2"].rank(method="min", ascending=False).astype(int)
    ranking_df["rank_by_RMSE"] = ranking_df["RMSE"].rank(method="min", ascending=True).astype(int)
    ranking_df["rank_by_abs_r"] = ranking_df["abs_r"].rank(method="min", ascending=False).astype(int)
    ranking_df["rank_by_sign_accuracy"] = ranking_df["sign_accuracy"].rank(method="min", ascending=False).astype(int)
    ranking_df = ranking_df[
        [
            "rank_by_R2",
            "rank_by_RMSE",
            "rank_by_abs_r",
            "rank_by_sign_accuracy",
            "feature",
            "pc",
            "month",
            "num_predictors",
            "r",
            "abs_r",
            "R2",
            "RMSE",
            "MAE",
            "sign_accuracy",
            "mean_error",
        ]
    ]
    return predictions_df, ranking_df


def run_nested_forward_selection(table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_all = table["obs_swe"].to_numpy(dtype=float)
    feature_matrix = table[AMV_COLUMNS].to_numpy(dtype=float)
    years = table["water_year"].to_numpy(dtype=int)

    prediction_rows = []  # type: List[Dict[str, object]]
    selected_rows = []  # type: List[Dict[str, object]]
    path_summary_rows = []  # type: List[Dict[str, object]]

    for held_idx, heldout_year in enumerate(years):
        outer_train_mask = np.ones(len(years), dtype=bool)
        outer_train_mask[held_idx] = False
        x_outer_train = feature_matrix[outer_train_mask, :]
        y_outer_train = y_all[outer_train_mask]
        x_outer_test = feature_matrix[~outer_train_mask, :][0]
        obs_test = y_all[held_idx]

        remaining = list(range(len(AMV_COLUMNS)))
        selected: list[int] = []
        outer_errors_by_k = {}  # type: Dict[int, float]
        final_k5_inner_rmse = np.nan

        for step in range(1, K_MAX + 1):
            best = None
            for candidate in remaining:
                candidate_set = selected + [candidate]
                x_candidate = x_outer_train[:, candidate_set]
                alpha, inner_rmse, inner_r2, _ = evaluate_alpha_grid_loyo(x_candidate, y_outer_train)
                score = (inner_rmse, -inner_r2, alpha, candidate)
                if best is None or score < best["score"]:
                    best = {
                        "candidate": candidate,
                        "candidate_set": candidate_set,
                        "alpha": alpha,
                        "inner_rmse": inner_rmse,
                        "inner_r2": inner_r2,
                        "score": score,
                    }
            assert best is not None
            selected.append(best["candidate"])
            remaining.remove(best["candidate"])
            selected_features = [AMV_COLUMNS[idx] for idx in selected]
            pred_raw, _ = fit_outer_prediction(
                x_outer_train[:, selected],
                y_outer_train,
                x_outer_test[selected],
                best["alpha"],
            )
            err = pred_raw - obs_test
            outer_errors_by_k[step] = float(abs(err))
            prediction_rows.append(
                {
                    "heldout_wy": int(heldout_year),
                    "K": int(step),
                    "selected_features": "|".join(selected_features),
                    "obs_swe": float(obs_test),
                    "pred_swe": float(pred_raw),
                    "error_pred_minus_obs": float(err),
                    "residual_obs_minus_pred": float(-err),
                    "abs_error": float(abs(err)),
                    "sign_correct": float(np.sign(pred_raw) == np.sign(obs_test)) if obs_test != 0.0 and pred_raw != 0.0 else np.nan,
                    "selected_alpha_final": float(best["alpha"]),
                    "inner_cv_RMSE_for_selected_set": float(best["inner_rmse"]),
                }
            )
            selected_rows.append(
                {
                    "heldout_wy": int(heldout_year),
                    "K": int(step),
                    "selected_feature_at_step": AMV_COLUMNS[best["candidate"]],
                    "selected_features_so_far": "|".join(selected_features),
                    "inner_cv_RMSE": float(best["inner_rmse"]),
                    "inner_cv_R2": float(best["inner_r2"]),
                    "selected_alpha": float(best["alpha"]),
                }
            )
            if step == K_MAX:
                final_k5_inner_rmse = float(best["inner_rmse"])

        path_summary_rows.append(
            {
                "heldout_wy": int(heldout_year),
                "K1_feature": AMV_COLUMNS[selected[0]],
                "K2_feature": AMV_COLUMNS[selected[1]],
                "K3_feature": AMV_COLUMNS[selected[2]],
                "K4_feature": AMV_COLUMNS[selected[3]],
                "K5_feature": AMV_COLUMNS[selected[4]],
                "selected_path": "|".join(AMV_COLUMNS[idx] for idx in selected),
                "final_K5_inner_cv_RMSE": final_k5_inner_rmse,
                "outer_K1_abs_error": outer_errors_by_k[1],
                "outer_K2_abs_error": outer_errors_by_k[2],
                "outer_K3_abs_error": outer_errors_by_k[3],
                "outer_K4_abs_error": outer_errors_by_k[4],
                "outer_K5_abs_error": outer_errors_by_k[5],
            }
        )

    predictions_df = pd.DataFrame(prediction_rows).sort_values(["heldout_wy", "K"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["heldout_wy", "K"]).reset_index(drop=True)
    path_df = pd.DataFrame(path_summary_rows).sort_values("heldout_wy").reset_index(drop=True)
    return predictions_df, selected_df, path_df


def compute_nested_metrics(predictions_df: pd.DataFrame, full_block_metrics: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []  # type: List[Dict[str, object]]
    for k in range(1, K_MAX + 1):
        sub = predictions_df[predictions_df["K"] == k].sort_values("heldout_wy")
        metrics = compute_metric_bundle(
            sub["obs_swe"].to_numpy(dtype=float), sub["pred_swe"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "model_name": f"AMV_forward_K{k}",
                "K": int(k),
                "num_predictors": int(k),
                "r": metrics["r"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_error": metrics["mean_error"],
                "median_abs_error": metrics["median_abs_error"],
            }
        )
    if full_block_metrics is not None and not full_block_metrics.empty:
        ref = full_block_metrics.iloc[0]
        rows.append(
            {
                "model_name": "AMV_AMO_PC1to6_full_ridge",
                "K": int(len(AMV_COLUMNS)),
                "num_predictors": int(ref["num_predictors"]),
                "r": float(ref["r"]),
                "R2": float(ref["R2"]),
                "RMSE": float(ref["RMSE"]),
                "MAE": float(ref["MAE"]),
                "sign_accuracy": float(ref["sign_accuracy"]),
                "mean_error": np.nan,
                "median_abs_error": np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_selection_frequency(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in AMV_COLUMNS:
        subset = selected_df[selected_df["selected_feature_at_step"] == feature]
        steps = subset["K"].to_numpy(dtype=int)
        any_selected_steps = selected_df[selected_df["selected_features_so_far"].str.contains(feature, regex=False)]
        selected_any_years = any_selected_steps["heldout_wy"].nunique()
        rows.append(
            {
                "feature": feature,
                "pc": feature.split("_")[1],
                "month": feature.split("_")[2],
                "selected_at_K1_count": int((steps == 1).sum()),
                "selected_at_K2_count": int((steps == 2).sum()),
                "selected_at_K3_count": int((steps == 3).sum()),
                "selected_at_K4_count": int((steps == 4).sum()),
                "selected_at_K5_count": int((steps == 5).sum()),
                "selected_any_count": int(selected_any_years),
                "selected_any_fraction": float(selected_any_years / len(WATER_YEARS)),
                "mean_selection_step": float(np.mean(steps)) if len(steps) else np.nan,
                "median_selection_step": float(np.median(steps)) if len(steps) else np.nan,
            }
        )
    freq_df = pd.DataFrame(rows).sort_values(
        ["selected_any_count", "selected_at_K1_count", "mean_selection_step"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return freq_df


def load_full_block_reference_metrics() -> Optional[pd.DataFrame]:
    if not FULL_BLOCK_METRICS_CSV.exists():
        return None
    metrics = pd.read_csv(FULL_BLOCK_METRICS_CSV)
    ref = metrics[metrics["model_name"] == "AMV_AMO_PC1to6_only"].copy()
    return ref.reset_index(drop=True)


def load_full_block_reference_predictions() -> Optional[pd.DataFrame]:
    if not FULL_BLOCK_PREDICTIONS_CSV.exists():
        return None
    preds = pd.read_csv(FULL_BLOCK_PREDICTIONS_CSV)
    preds = preds[preds["model_name"] == "AMV_AMO_PC1to6_only"].copy()
    if preds.empty:
        return None
    return preds[["heldout_wy", "obs_swe", "pred_swe"]].rename(columns={"pred_swe": "full_ridge_pred"})


def make_top20_single_plot(ranking_df: pd.DataFrame) -> None:
    top20 = ranking_df.sort_values(["RMSE", "R2"], ascending=[True, False]).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.barh(top20["feature"], top20["RMSE"], color="#4c72b0")
    ax.set_xlabel("LOYO RMSE (m)")
    ax.set_title("Top 20 single AMV/AMO predictors by LOYO RMSE")
    fig.tight_layout()
    fig.savefig(TOP20_SINGLE_PNG, dpi=200)
    plt.close(fig)


def make_metrics_by_k_plot(metrics_df: pd.DataFrame) -> None:
    main = metrics_df[metrics_df["model_name"].str.startswith("AMV_forward_")].copy().sort_values("K")
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    axes[0].plot(main["K"], main["R2"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("$R^2$")
    axes[0].set_title("Nested forward AMV-only metrics by selected subset size")
    axes[1].plot(main["K"], main["RMSE"], marker="o", color="#d62728")
    axes[1].set_ylabel("RMSE (m)")
    axes[2].plot(main["K"], main["sign_accuracy"], marker="o", color="#2ca02c")
    axes[2].set_ylabel("Sign accuracy")
    axes[2].set_xlabel("K selected features")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(main["K"])
    fig.tight_layout()
    fig.savefig(METRICS_BY_K_PNG, dpi=200)
    plt.close(fig)


def make_selection_frequency_plot(freq_df: pd.DataFrame) -> None:
    top20 = freq_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.barh(top20["feature"], top20["selected_any_count"], color="#55a868", label="Selected in any step")
    ax.barh(top20["feature"], top20["selected_at_K1_count"], color="#c44e52", label="Selected at K1")
    ax.set_xlabel("Number of outer LOYO folds")
    ax.set_title("Top 20 AMV/AMO features by selection frequency")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(SELECTION_FREQ_PNG, dpi=200)
    plt.close(fig)


def make_observed_vs_predicted_plot(
    table: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    full_block_preds: Optional[pd.DataFrame],
) -> None:
    best_k_row = metrics_df[metrics_df["model_name"].str.startswith("AMV_forward_")].sort_values(
        ["RMSE", "R2"], ascending=[True, False]
    ).iloc[0]
    best_k = int(best_k_row["K"])
    plot_df = table[["water_year", "obs_swe"]].copy()
    for k in sorted(set([1, best_k, 5])):
        sub = predictions_df[predictions_df["K"] == k][["heldout_wy", "pred_swe"]].rename(
            columns={"heldout_wy": "water_year", "pred_swe": f"AMV_forward_K{k}"}
        )
        plot_df = plot_df.merge(sub, on="water_year", how="left")
    if full_block_preds is not None:
        plot_df = plot_df.merge(
            full_block_preds.rename(columns={"heldout_wy": "water_year", "obs_swe": "obs_check"}),
            on="water_year",
            how="left",
        ).drop(columns=["obs_check"])

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.plot(plot_df["water_year"], plot_df["obs_swe"], color="black", linewidth=2.4, label="Observed")
    colors = {1: "#1f77b4", best_k: "#ff7f0e", 5: "#2ca02c"}
    for k in sorted(set([1, best_k, 5])):
        ax.plot(
            plot_df["water_year"],
            plot_df[f"AMV_forward_K{k}"],
            marker="o",
            linewidth=1.8,
            label=f"AMV_forward_K{k}",
            color=colors[k],
        )
    if full_block_preds is not None:
        ax.plot(
            plot_df["water_year"],
            plot_df["full_ridge_pred"],
            linewidth=1.8,
            linestyle="--",
            color="#d62728",
            label="AMV_AMO_PC1to6_full_ridge",
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Observed vs predicted for nested forward-selected AMV-only models")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OBS_PRED_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictor_table = build_predictor_table()
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    single_predictions_df, single_ranking_df = run_single_variable_loyo(predictor_table)
    single_predictions_df.to_csv(SINGLE_PREDICTIONS_CSV, index=False)
    single_ranking_df.to_csv(SINGLE_RANKING_CSV, index=False)

    nested_predictions_df, selected_df, path_df = run_nested_forward_selection(predictor_table)
    nested_predictions_df.to_csv(NESTED_PREDICTIONS_CSV, index=False)
    selected_df.to_csv(SELECTED_BY_FOLD_CSV, index=False)
    path_df.to_csv(SELECTION_PATH_SUMMARY_CSV, index=False)

    full_block_metrics = load_full_block_reference_metrics()
    nested_metrics_df = compute_nested_metrics(nested_predictions_df, full_block_metrics)
    nested_metrics_df.to_csv(NESTED_METRICS_CSV, index=False)

    selection_freq_df = compute_selection_frequency(selected_df)
    selection_freq_df.to_csv(SELECTION_FREQ_CSV, index=False)

    full_block_preds = load_full_block_reference_predictions()

    make_top20_single_plot(single_ranking_df)
    make_metrics_by_k_plot(nested_metrics_df)
    make_selection_frequency_plot(selection_freq_df)
    make_observed_vs_predicted_plot(
        predictor_table, nested_predictions_df, nested_metrics_df, full_block_preds
    )

    nested_only = nested_metrics_df[nested_metrics_df["model_name"].str.startswith("AMV_forward_")].copy()
    best_by_rmse = nested_only.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    best_by_r2 = nested_only.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    best_by_sign = nested_only.sort_values(["sign_accuracy", "RMSE"], ascending=[False, True]).iloc[0]
    stable_features = selection_freq_df[selection_freq_df["selected_any_count"] >= 10]["feature"].tolist()
    very_stable_features = selection_freq_df[selection_freq_df["selected_any_count"] >= 15]["feature"].tolist()

    if int(best_by_rmse["K"]) <= 2 and len(stable_features) > 0:
        recommendation = (
            "The next combined LOD+AMV test should prefer a reduced AMV subset rather than the full "
            "42-feature AMV block."
        )
    elif len(stable_features) == 0:
        recommendation = (
            "Selection is highly unstable, so the next combined LOD+AMV test should be cautious about "
            "adding AMV predictors at all."
        )
    else:
        recommendation = (
            "A modest reduced AMV subset is more defensible than the full AMV block, but stability is mixed."
        )

    summary = {
        "input_files": {
            "amv_csv": str(AMV_CSV),
            "base_predictions_csv": str(BASE_PREDICTIONS_CSV),
            "full_block_predictions_csv": str(FULL_BLOCK_PREDICTIONS_CSV),
            "full_block_metrics_csv": str(FULL_BLOCK_METRICS_CSV),
        },
        "output_dir": str(OUTPUT_DIR),
        "water_years": WATER_YEARS,
        "num_amv_predictors": len(AMV_COLUMNS),
        "single_variable_top10_by_RMSE": single_ranking_df.sort_values(["RMSE", "R2"], ascending=[True, False]).head(10).to_dict(orient="records"),
        "single_variable_top10_by_R2": single_ranking_df.sort_values(["R2", "RMSE"], ascending=[False, True]).head(10).to_dict(orient="records"),
        "single_variable_top10_by_abs_r": single_ranking_df.sort_values(["abs_r", "RMSE"], ascending=[False, True]).head(10).to_dict(orient="records"),
        "nested_forward_metrics_by_K": nested_only.to_dict(orient="records"),
        "best_nested_forward_K_by_RMSE": best_by_rmse.to_dict(),
        "best_nested_forward_K_by_R2": best_by_r2.to_dict(),
        "best_nested_forward_K_by_sign_accuracy": best_by_sign.to_dict(),
        "selection_frequency_top20": selection_freq_df.head(20).to_dict(orient="records"),
        "stable_features": stable_features,
        "very_stable_features": very_stable_features,
        "full_AMV_block_reference_metrics": full_block_metrics.to_dict(orient="records") if full_block_metrics is not None else [],
        "short_answer": (
            f"The strongest individual AMV/AMO predictors are led by "
            f"{single_ranking_df.sort_values(['RMSE', 'R2'], ascending=[True, False]).head(3)['feature'].tolist()}. "
            f"Honest nested forward selection peaks around K={int(best_by_rmse['K'])} by RMSE, "
            f"with stable features {stable_features[:10]}. {recommendation}"
        ),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Output directory: {OUTPUT_DIR}")
    print("Top 10 single-variable AMV predictors by RMSE:")
    print(single_ranking_df.sort_values(["RMSE", "R2"], ascending=[True, False]).head(10).to_string(index=False))
    print("Nested forward metrics by K:")
    print(nested_only.to_string(index=False))
    print(f"Best K by RMSE: K={int(best_by_rmse['K'])}")
    print(f"Best K by R2: K={int(best_by_r2['K'])}")
    print(f"Best K by sign accuracy: K={int(best_by_sign['K'])}")
    print("Top selection frequencies:")
    print(selection_freq_df.head(10).to_string(index=False))
    print("Stable features:")
    print(stable_features)
    print("Short answer:")
    print(summary["short_answer"])


if __name__ == "__main__":
    main()
