#!/usr/bin/env python3
"""
Forward-addition diagnostic starting from the fixed four-feature AMV/AMO core.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    / "amv_amo_core_plus_forward_addition"
)

FIXED_CORE_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_fixed_core_cumulative_loyo"
    / "amv_amo_fixed_core_predictor_table.csv"
)
SUBSET_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_subset_selection_diagnostic"
    / "amv_amo_predictor_table.csv"
)
BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)
AMV_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)

PREDICTOR_TABLE_OUT = OUTPUT_DIR / "amv_core_plus_forward_predictor_table.csv"
SELECTION_PATH_CSV = OUTPUT_DIR / "amv_core_plus_forward_selection_path.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "amv_core_plus_forward_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "amv_core_plus_forward_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "amv_core_plus_forward_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "amv_core_plus_forward_selected_alpha_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "amv_core_plus_forward_beta_by_fold.csv"
SUMMARY_JSON = OUTPUT_DIR / "amv_core_plus_forward_summary.json"
METRICS_PNG = OUTPUT_DIR / "amv_core_plus_forward_metrics_by_K.png"
IMPROVEMENT_PNG = OUTPUT_DIR / "amv_core_plus_forward_improvement_by_step.png"
OBS_PRED_PNG = OUTPUT_DIR / "amv_core_plus_forward_observed_vs_predicted.png"
ERROR_PNG = OUTPUT_DIR / "amv_core_plus_forward_error_by_year.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
ALL_AMV_COLUMNS = [f"AMV_PC{pc}_{month}" for pc in range(1, 7) for month in MONTHS]
CORE_FEATURES = ["AMV_PC4_Sep", "AMV_PC5_Feb", "AMV_PC2_Feb", "AMV_PC4_Nov"]
MEANINGFUL_DELTA_RMSE = 0.001
MEANINGFUL_DELTA_R2 = 0.03
MAX_CONSECUTIVE_NONMEANINGFUL = 2
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
PERIOD_SPECS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
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
    err = pred - obs
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "mean_error": float(np.mean(err)),
        "median_abs_error": float(np.median(np.abs(err))),
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
    coef_std = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    pred_std = x_test_std @ coef_std
    return pred_std, coef_std


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
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train, x_test)
    y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train, np.asarray([0.0]))
    pred_std, coef_std = ridge_fit_predict_standardized(
        x_train_std, y_train_std, x_test_std.reshape(1, -1), alpha
    )
    pred_raw = y_mean + y_std * float(pred_std[0])
    coef_raw = (y_std / x_std) * coef_std
    intercept_raw = float(y_mean - np.sum(coef_raw * x_mean))
    return pred_raw, coef_raw, intercept_raw


def normalize_duplicate_columns(columns: Sequence[str]) -> List[str]:
    return [re.sub(r"\.\d+$", "", col) for col in columns]


def load_fixed_core_table_deduped() -> Optional[pd.DataFrame]:
    if not FIXED_CORE_TABLE_CSV.exists():
        return None
    table = pd.read_csv(FIXED_CORE_TABLE_CSV)
    normalized = normalize_duplicate_columns(table.columns)
    keep_indices = []
    seen = set()
    for idx, col in enumerate(normalized):
        if col not in seen:
            keep_indices.append(idx)
            seen.add(col)
    table = table.iloc[:, keep_indices].copy()
    table.columns = [normalized[idx] for idx in keep_indices]
    needed = ["water_year", "obs_swe"] + ALL_AMV_COLUMNS
    if any(col not in table.columns for col in needed):
        return None
    return table[needed].copy()


def build_predictor_table() -> Tuple[pd.DataFrame, Dict[str, object]]:
    source_note: Dict[str, object] = {"preferred_file": str(FIXED_CORE_TABLE_CSV), "fallback_file": str(SUBSET_TABLE_CSV)}
    table = load_fixed_core_table_deduped()
    if table is not None:
        source_note["used_file"] = str(FIXED_CORE_TABLE_CSV)
        source_note["load_mode"] = "deduplicate_duplicate_headers_keep_first"
    elif SUBSET_TABLE_CSV.exists():
        table = pd.read_csv(SUBSET_TABLE_CSV)
        source_note["used_file"] = str(SUBSET_TABLE_CSV)
        source_note["load_mode"] = "fallback_subset_table"
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
        table = targets.merge(amv[["water_year"] + ALL_AMV_COLUMNS], on="water_year", how="inner")
        source_note["used_file"] = "rebuilt_from_base_predictions_plus_amv_csv"
        source_note["load_mode"] = "merge_base_targets_with_amv"
    needed = ["water_year", "obs_swe"] + ALL_AMV_COLUMNS
    missing = [col for col in needed if col not in table.columns]
    if missing:
        raise ValueError(f"Predictor table is missing expected columns: {missing}")
    table = table[needed].copy()
    table["water_year"] = table["water_year"].astype(int)
    table = table.sort_values("water_year").reset_index(drop=True)
    table = table.loc[
        (table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)
    ].copy()
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Predictor table does not match WY1985--WY2021.")
    return table, source_note


def features_to_string(features: Sequence[str]) -> str:
    return "|".join(features)


def model_name_for_features(features: Sequence[str]) -> str:
    if list(features) == ALL_AMV_COLUMNS:
        return "AMV_AMO_PC1to6_full"
    return f"AMV_core_plus_K{len(features)}"


def evaluate_feature_set(
    table: pd.DataFrame,
    feature_list: Sequence[str],
    step: int,
    model_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    feature_list = list(feature_list)
    model_name = model_name or model_name_for_features(feature_list)
    years = table["water_year"].to_numpy(dtype=int)
    y = table["obs_swe"].to_numpy(dtype=float)
    x_all = table[feature_list].to_numpy(dtype=float)

    prediction_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []

    for held_idx, held_year in enumerate(years):
        train_mask = np.ones(len(years), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_all[train_mask, :]
        x_test = x_all[~train_mask, :][0]
        y_train = y[train_mask]
        obs = y[held_idx]

        alpha, inner_cv_rmse, _ = evaluate_alpha_grid_loyo(x_train, y_train)
        pred_raw, coef_raw, intercept_raw = fit_outer_prediction(x_train, y_train, x_test, alpha)
        err = pred_raw - obs
        prediction_rows.append(
            {
                "model_name": model_name,
                "step": int(step),
                "heldout_wy": int(held_year),
                "obs_swe": float(obs),
                "pred_swe": float(pred_raw),
                "error_pred_minus_obs": float(err),
                "residual_obs_minus_pred": float(-err),
                "abs_error": float(abs(err)),
                "sign_correct": float(np.sign(pred_raw) == np.sign(obs)) if obs != 0.0 and pred_raw != 0.0 else np.nan,
                "selected_alpha": float(alpha),
                "num_predictors": int(len(feature_list)),
                "selected_features": features_to_string(feature_list),
            }
        )
        alpha_rows.append(
            {
                "model_name": model_name,
                "step": int(step),
                "heldout_wy": int(held_year),
                "selected_alpha": float(alpha),
                "inner_cv_mse": float(inner_cv_rmse**2),
            }
        )
        beta_row: Dict[str, object] = {
            "model_name": model_name,
            "step": int(step),
            "heldout_wy": int(held_year),
            "intercept": float(intercept_raw),
        }
        for feature in ALL_AMV_COLUMNS:
            beta_row["beta_" + feature] = float(coef_raw[feature_list.index(feature)]) if feature in feature_list else np.nan
        beta_rows.append(beta_row)

    pred_df = pd.DataFrame(prediction_rows).sort_values("heldout_wy").reset_index(drop=True)
    alpha_df = pd.DataFrame(alpha_rows).sort_values("heldout_wy").reset_index(drop=True)
    beta_df = pd.DataFrame(beta_rows).sort_values("heldout_wy").reset_index(drop=True)
    metrics = compute_metric_bundle(pred_df["obs_swe"].to_numpy(dtype=float), pred_df["pred_swe"].to_numpy(dtype=float))
    metrics_row: Dict[str, object] = {
        "model_name": model_name,
        "step": int(step),
        "num_predictors": int(len(feature_list)),
        "r": metrics["r"],
        "R2": metrics["R2"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "sign_accuracy": metrics["sign_accuracy"],
        "mean_error": metrics["mean_error"],
        "median_abs_error": metrics["median_abs_error"],
        "selected_features": features_to_string(feature_list),
    }
    return pred_df, alpha_df, beta_df, metrics_row


def compute_period_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (model_name, step), sub in pred_df.groupby(["model_name", "step"], sort=False):
        sub = sub.sort_values("heldout_wy")
        wy = sub["heldout_wy"].to_numpy(dtype=int)
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        for group_name, selector in PERIOD_SPECS:
            mask = selector(wy)
            metrics = compute_metric_bundle(obs[mask], pred[mask])
            rows.append(
                {
                    "model_name": model_name,
                    "step": int(step),
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
    return pd.DataFrame(rows).sort_values(["step", "group_name"]).reset_index(drop=True)


def forward_addition(table: pd.DataFrame) -> Dict[str, object]:
    selected = list(CORE_FEATURES)
    remaining = [feature for feature in ALL_AMV_COLUMNS if feature not in selected]
    accepted_models: List[Dict[str, object]] = []
    selection_path_rows: List[Dict[str, object]] = []
    all_predictions: List[pd.DataFrame] = []
    all_alphas: List[pd.DataFrame] = []
    all_betas: List[pd.DataFrame] = []
    all_metrics_rows: List[Dict[str, object]] = []

    k4_step = len(selected)
    pred_df, alpha_df, beta_df, metrics_row = evaluate_feature_set(table, selected, k4_step, "AMV_core_plus_K4")
    accepted_models.append(
        {
            "step": k4_step,
            "model_name": "AMV_core_plus_K4",
            "features": list(selected),
            "metrics": dict(metrics_row),
        }
    )
    all_predictions.append(pred_df)
    all_alphas.append(alpha_df)
    all_betas.append(beta_df)
    all_metrics_rows.append(metrics_row)
    selection_path_rows.append(
        {
            "step": int(k4_step),
            "model_name": "AMV_core_plus_K4",
            "num_predictors": int(len(selected)),
            "selected_features": features_to_string(selected),
            "added_feature": np.nan,
            "previous_RMSE": np.nan,
            "new_RMSE": float(metrics_row["RMSE"]),
            "delta_RMSE": np.nan,
            "previous_R2": np.nan,
            "new_R2": float(metrics_row["R2"]),
            "delta_R2": np.nan,
            "new_MAE": float(metrics_row["MAE"]),
            "new_sign_accuracy": float(metrics_row["sign_accuracy"]),
            "meaningful_gain": np.nan,
            "consecutive_nonmeaningful_after_step": 0,
            "stop_reason": "starting_core",
        }
    )

    previous_metrics = metrics_row
    consecutive_nonmeaningful = 0
    stop_reason = "all_remaining_features_evaluated"

    while remaining and consecutive_nonmeaningful < MAX_CONSECUTIVE_NONMEANINGFUL:
        next_step = len(selected) + 1
        best_trial = None
        for candidate in remaining:
            trial_features = selected + [candidate]
            trial_pred_df, trial_alpha_df, trial_beta_df, trial_metrics = evaluate_feature_set(
                table,
                trial_features,
                next_step,
                f"AMV_core_plus_K{len(trial_features)}",
            )
            score = (
                float(trial_metrics["RMSE"]),
                -float(trial_metrics["R2"]),
                -float(trial_metrics["sign_accuracy"]),
                candidate,
            )
            if best_trial is None or score < best_trial["score"]:
                best_trial = {
                    "candidate": candidate,
                    "features": trial_features,
                    "pred_df": trial_pred_df,
                    "alpha_df": trial_alpha_df,
                    "beta_df": trial_beta_df,
                    "metrics": trial_metrics,
                    "score": score,
                }
        assert best_trial is not None
        selected = list(best_trial["features"])
        remaining.remove(best_trial["candidate"])

        delta_rmse = float(previous_metrics["RMSE"]) - float(best_trial["metrics"]["RMSE"])
        delta_r2 = float(best_trial["metrics"]["R2"]) - float(previous_metrics["R2"])
        meaningful = (delta_rmse >= MEANINGFUL_DELTA_RMSE) or (delta_r2 >= MEANINGFUL_DELTA_R2)
        if meaningful:
            consecutive_nonmeaningful = 0
        else:
            consecutive_nonmeaningful += 1
        if consecutive_nonmeaningful >= MAX_CONSECUTIVE_NONMEANINGFUL:
            stop_reason = "two_consecutive_nonmeaningful_additions"
        elif not remaining:
            stop_reason = "all_remaining_features_evaluated"
        else:
            stop_reason = ""

        model_name = f"AMV_core_plus_K{len(selected)}"
        best_trial["metrics"]["model_name"] = model_name

        accepted_models.append(
            {
                "step": int(len(selected)),
                "model_name": model_name,
                "features": list(selected),
                "metrics": dict(best_trial["metrics"]),
                "added_feature": best_trial["candidate"],
                "delta_RMSE": delta_rmse,
                "delta_R2": delta_r2,
                "meaningful_gain": meaningful,
                "consecutive_nonmeaningful_after_step": consecutive_nonmeaningful,
                "stop_reason": stop_reason,
            }
        )
        all_predictions.append(best_trial["pred_df"])
        all_alphas.append(best_trial["alpha_df"])
        all_betas.append(best_trial["beta_df"])
        all_metrics_rows.append(best_trial["metrics"])
        selection_path_rows.append(
            {
                "step": int(len(selected)),
                "model_name": model_name,
                "num_predictors": int(len(selected)),
                "selected_features": features_to_string(selected),
                "added_feature": best_trial["candidate"],
                "previous_RMSE": float(previous_metrics["RMSE"]),
                "new_RMSE": float(best_trial["metrics"]["RMSE"]),
                "delta_RMSE": float(delta_rmse),
                "previous_R2": float(previous_metrics["R2"]),
                "new_R2": float(best_trial["metrics"]["R2"]),
                "delta_R2": float(delta_r2),
                "new_MAE": float(best_trial["metrics"]["MAE"]),
                "new_sign_accuracy": float(best_trial["metrics"]["sign_accuracy"]),
                "meaningful_gain": bool(meaningful),
                "consecutive_nonmeaningful_after_step": int(consecutive_nonmeaningful),
                "stop_reason": stop_reason,
            }
        )
        previous_metrics = best_trial["metrics"]

    full_step = len(ALL_AMV_COLUMNS)
    full_pred_df, full_alpha_df, full_beta_df, full_metrics_row = evaluate_feature_set(
        table, ALL_AMV_COLUMNS, full_step, "AMV_AMO_PC1to6_full"
    )
    all_predictions.append(full_pred_df)
    all_alphas.append(full_alpha_df)
    all_betas.append(full_beta_df)
    all_metrics_rows.append(full_metrics_row)
    selection_path_rows.append(
        {
            "step": int(full_step),
            "model_name": "AMV_AMO_PC1to6_full",
            "num_predictors": int(len(ALL_AMV_COLUMNS)),
            "selected_features": features_to_string(ALL_AMV_COLUMNS),
            "added_feature": np.nan,
            "previous_RMSE": np.nan,
            "new_RMSE": float(full_metrics_row["RMSE"]),
            "delta_RMSE": np.nan,
            "previous_R2": np.nan,
            "new_R2": float(full_metrics_row["R2"]),
            "delta_R2": np.nan,
            "new_MAE": float(full_metrics_row["MAE"]),
            "new_sign_accuracy": float(full_metrics_row["sign_accuracy"]),
            "meaningful_gain": np.nan,
            "consecutive_nonmeaningful_after_step": np.nan,
            "stop_reason": "reference_full_42",
        }
    )

    predictions_df = pd.concat(all_predictions, ignore_index=True).sort_values(["step", "heldout_wy"]).reset_index(drop=True)
    alpha_df = pd.concat(all_alphas, ignore_index=True).sort_values(["step", "heldout_wy"]).reset_index(drop=True)
    beta_df = pd.concat(all_betas, ignore_index=True).sort_values(["step", "heldout_wy"]).reset_index(drop=True)
    metrics_df = pd.DataFrame(all_metrics_rows).sort_values("step").reset_index(drop=True)
    selection_path_df = pd.DataFrame(selection_path_rows).sort_values("step").reset_index(drop=True)

    return {
        "accepted_models": accepted_models,
        "predictions_df": predictions_df,
        "alpha_df": alpha_df,
        "beta_df": beta_df,
        "metrics_df": metrics_df,
        "selection_path_df": selection_path_df,
        "stop_reason": stop_reason,
    }


def make_metrics_plot(metrics_df: pd.DataFrame, final_step: int) -> None:
    main = metrics_df[metrics_df["model_name"] != "AMV_AMO_PC1to6_full"].copy().sort_values("step")
    full = metrics_df[metrics_df["model_name"] == "AMV_AMO_PC1to6_full"].iloc[0]
    x = main["num_predictors"].to_numpy(dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True)
    series = [("R2", "$R^2$", "#1f77b4"), ("RMSE", "RMSE (m)", "#d62728"), ("sign_accuracy", "Sign accuracy", "#2ca02c")]
    for ax, (col, ylabel, color) in zip(axes, series):
        ax.plot(x, main[col], marker="o", color=color, linewidth=1.8)
        ax.scatter([len(CORE_FEATURES)], [float(main.loc[main["num_predictors"] == len(CORE_FEATURES), col].iloc[0])], color="black", s=55, zorder=3)
        ax.scatter([final_step], [float(main.loc[main["num_predictors"] == final_step, col].iloc[-1])], color="#ff7f0e", s=65, zorder=3)
        ax.scatter([int(full["num_predictors"])], [float(full[col])], color="#9467bd", s=65, marker="D", zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Forward addition from fixed AMV/AMO K4 core")
    axes[-1].set_xlabel("Number of predictors")
    axes[-1].set_xticks(list(main["num_predictors"]) + [42])
    axes[-1].axvline(len(CORE_FEATURES), color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[-1].axvline(final_step, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[-1].axvline(42, color="#9467bd", linestyle=":", linewidth=1.2, alpha=0.8)
    fig.tight_layout()
    fig.savefig(METRICS_PNG, dpi=200)
    plt.close(fig)


def make_improvement_plot(selection_path_df: pd.DataFrame) -> None:
    main = selection_path_df[
        selection_path_df["model_name"].str.startswith("AMV_core_plus_K") & (selection_path_df["step"] > len(CORE_FEATURES))
    ].copy()
    labels = main["added_feature"].tolist()
    x = np.arange(len(main))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    axes[0].bar(x, main["delta_RMSE"], color="#d62728", alpha=0.85)
    axes[0].axhline(MEANINGFUL_DELTA_RMSE, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(r"$\Delta$RMSE (m)")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, main["delta_R2"], color="#1f77b4", alpha=0.85)
    axes[1].axhline(MEANINGFUL_DELTA_R2, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(r"$\Delta R^2$")
    axes[1].set_xlabel("Added feature")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_title("Stepwise forward-addition improvements beyond AMV K4 core")
    fig.tight_layout()
    fig.savefig(IMPROVEMENT_PNG, dpi=200)
    plt.close(fig)


def make_observed_vs_predicted_plot(pred_df: pd.DataFrame, final_model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    k4 = pred_df[pred_df["model_name"] == "AMV_core_plus_K4"].sort_values("heldout_wy")
    final_sub = pred_df[pred_df["model_name"] == final_model_name].sort_values("heldout_wy")
    full = pred_df[pred_df["model_name"] == "AMV_AMO_PC1to6_full"].sort_values("heldout_wy")
    ax.plot(k4["heldout_wy"], k4["obs_swe"], color="black", linewidth=2.5, label="Observed")
    ax.plot(k4["heldout_wy"], k4["pred_swe"], color="#1f77b4", marker="o", linewidth=1.7, label="K4")
    ax.plot(final_sub["heldout_wy"], final_sub["pred_swe"], color="#ff7f0e", marker="o", linewidth=1.7, label=final_model_name)
    ax.plot(full["heldout_wy"], full["pred_swe"], color="#9467bd", marker="o", linewidth=1.7, label="AMV_AMO_PC1to6_full")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Observed vs predicted: K4, forward-expanded model, and full 42-feature AMV")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OBS_PRED_PNG, dpi=200)
    plt.close(fig)


def make_error_by_year_plot(pred_df: pd.DataFrame, final_model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    for model_name, color in [
        ("AMV_core_plus_K4", "#1f77b4"),
        (final_model_name, "#ff7f0e"),
        ("AMV_AMO_PC1to6_full", "#9467bd"),
    ]:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["error_pred_minus_obs"],
            marker="o",
            linewidth=1.6,
            color=color,
            label=model_name,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel(r"$\widehat{SWE} - SWE_{obs}$ (m)")
    ax.set_title("Prediction error by year for K4, forward-expanded model, and full 42-feature AMV")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ERROR_PNG, dpi=200)
    plt.close(fig)


def build_summary(
    source_note: Dict[str, object],
    selection_results: Dict[str, object],
    period_metrics_df: pd.DataFrame,
) -> Dict[str, object]:
    selection_path_df: pd.DataFrame = selection_results["selection_path_df"]
    metrics_df: pd.DataFrame = selection_results["metrics_df"]
    pred_df: pd.DataFrame = selection_results["predictions_df"]
    accepted_models: List[Dict[str, object]] = selection_results["accepted_models"]

    final_selected = accepted_models[-1]
    final_model_name = str(final_selected["model_name"])
    final_step = int(final_selected["step"])
    full_metrics = metrics_df[metrics_df["model_name"] == "AMV_AMO_PC1to6_full"].iloc[0].to_dict()
    nonref_path = selection_path_df[
        selection_path_df["model_name"].str.startswith("AMV_core_plus_K")
    ].copy()
    best_r2_row = metrics_df.loc[metrics_df["R2"].astype(float).idxmax()].to_dict()
    best_rmse_row = metrics_df.loc[metrics_df["RMSE"].astype(float).idxmin()].to_dict()
    best_sign_row = metrics_df.loc[metrics_df["sign_accuracy"].astype(float).idxmax()].to_dict()

    k4 = pred_df[pred_df["model_name"] == "AMV_core_plus_K4"][["heldout_wy", "abs_error", "sign_correct", "pred_swe", "obs_swe"]].copy()
    final_pred = pred_df[pred_df["model_name"] == final_model_name][["heldout_wy", "abs_error", "sign_correct", "pred_swe"]].copy()
    compare = k4.merge(final_pred, on="heldout_wy", suffixes=("_k4", "_final"))
    years_improved = compare.loc[compare["abs_error_final"] < compare["abs_error_k4"], "heldout_wy"].astype(int).tolist()
    years_worsened = compare.loc[compare["abs_error_final"] > compare["abs_error_k4"], "heldout_wy"].astype(int).tolist()

    top_added = nonref_path[
        (nonref_path["step"] > len(CORE_FEATURES)) & nonref_path["added_feature"].notna()
    ][["step", "added_feature", "delta_RMSE", "delta_R2", "meaningful_gain"]].copy()
    top_added = top_added.sort_values(["delta_RMSE", "delta_R2"], ascending=[False, False]).reset_index(drop=True)

    final_metrics = final_selected["metrics"]
    full_gap_rmse = float(final_metrics["RMSE"]) - float(full_metrics["RMSE"])
    full_gap_r2 = float(full_metrics["R2"]) - float(final_metrics["R2"])
    sign_shift = float(final_metrics["sign_accuracy"]) - float(metrics_df[metrics_df["model_name"] == "AMV_core_plus_K4"]["sign_accuracy"].iloc[0])

    meaningful_steps = nonref_path[
        (nonref_path["step"] > len(CORE_FEATURES)) & (nonref_path["meaningful_gain"] == True)  # noqa: E712
    ]
    if meaningful_steps.empty:
        how_many = "no extra predictors pass the meaningful-gain threshold"
    else:
        how_many = f"meaningful gains persist through K{int(meaningful_steps['step'].max())}"

    if full_gap_rmse <= 0.0015 and full_gap_r2 <= 0.04:
        proximity_text = "A small extension beyond K4 recovers most of the full AMV amplitude skill."
    elif float(final_metrics["RMSE"]) < float(metrics_df[metrics_df["model_name"] == "AMV_core_plus_K4"]["RMSE"].iloc[0]):
        proximity_text = "The added predictors recover part of the full AMV amplitude skill, but the remaining advantage is still distributed across more AMV PC-month predictors."
    else:
        proximity_text = "The K4 core already captures most of the stable low-dimensional AMV signal; the full 42-feature advantage appears to come from distributed weak predictors rather than a few extra strong ones."

    added_features_list = [
        str(row["added_feature"])
        for _, row in nonref_path[
            (nonref_path["step"] > len(CORE_FEATURES)) & nonref_path["added_feature"].notna()
        ].iterrows()
    ]
    if added_features_list:
        feature_text = ", ".join(added_features_list[:4])
    else:
        feature_text = "none"
    sign_text = (
        "The forward-expanded model preserves or improves the K4 sign behavior."
        if sign_shift >= 0.0
        else "The forward-expanded model gives up some of K4's sign advantage as it moves toward the full model's amplitude behavior."
    )
    short_answer = (
        f"The extra AMV predictors added beyond K4 are led by {feature_text}. "
        f"{how_many}. {proximity_text} {sign_text}"
    )

    return {
        "input_files": source_note,
        "output_dir": str(OUTPUT_DIR),
        "fixed_starting_core": CORE_FEATURES,
        "feature_universe": ALL_AMV_COLUMNS,
        "stopping_rule": {
            "meaningful_if_delta_RMSE_at_least": MEANINGFUL_DELTA_RMSE,
            "meaningful_if_delta_R2_at_least": MEANINGFUL_DELTA_R2,
            "stop_after_consecutive_nonmeaningful_additions": MAX_CONSECUTIVE_NONMEANINGFUL,
        },
        "selection_path": selection_path_df.to_dict(orient="records"),
        "final_selected_model": {
            "model_name": final_model_name,
            "step": final_step,
            "features": final_selected["features"],
            "metrics": final_metrics,
            "stop_reason": selection_results["stop_reason"],
        },
        "full_42_reference_metrics": full_metrics,
        "metrics_by_step": metrics_df.to_dict(orient="records"),
        "period_metrics": period_metrics_df.to_dict(orient="records"),
        "best_model_by_R2": best_r2_row,
        "best_model_by_RMSE": best_rmse_row,
        "best_model_by_sign_accuracy": best_sign_row,
        "top_added_features": top_added.to_dict(orient="records"),
        "years_improved_by_final_model_relative_to_K4": years_improved,
        "years_worsened_by_final_model_relative_to_K4": years_worsened,
        "short_answer": short_answer,
    }


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, source_note = build_predictor_table()
    table.to_csv(PREDICTOR_TABLE_OUT, index=False)

    selection_results = forward_addition(table)
    predictions_df: pd.DataFrame = selection_results["predictions_df"]
    alpha_df: pd.DataFrame = selection_results["alpha_df"]
    beta_df: pd.DataFrame = selection_results["beta_df"]
    metrics_df: pd.DataFrame = selection_results["metrics_df"]
    selection_path_df: pd.DataFrame = selection_results["selection_path_df"]
    period_metrics_df = compute_period_metrics(predictions_df)

    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_metrics_df.to_csv(PERIOD_METRICS_CSV, index=False)
    selection_path_df.to_csv(SELECTION_PATH_CSV, index=False)

    final_nonref = selection_results["accepted_models"][-1]
    final_model_name = str(final_nonref["model_name"])
    final_step = int(final_nonref["step"])

    make_metrics_plot(metrics_df, final_step)
    make_improvement_plot(selection_path_df)
    make_observed_vs_predicted_plot(predictions_df, final_model_name)
    make_error_by_year_plot(predictions_df, final_model_name)

    summary = build_summary(source_note, selection_results, period_metrics_df)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    path_display = selection_path_df[
        selection_path_df["model_name"].str.startswith("AMV_core_plus_K")
    ][["step", "added_feature", "new_RMSE", "delta_RMSE", "new_R2", "delta_R2", "new_sign_accuracy", "meaningful_gain"]]
    full_ref = metrics_df[metrics_df["model_name"] == "AMV_AMO_PC1to6_full"].iloc[0]

    print(f"Output directory: {OUTPUT_DIR}")
    print("Starting K4 features:")
    for feature in CORE_FEATURES:
        print(f"  - {feature}")
    print("Selection path:")
    print(path_display.to_string(index=False))
    print(f"Stop reason: {selection_results['stop_reason']}")
    print("Full 42 reference:")
    print(
        "  model_name={model_name}, RMSE={RMSE:.6f}, R2={R2:.6f}, sign_accuracy={sign_accuracy:.6f}".format(
            **full_ref.to_dict()
        )
    )
    print("Short answer:")
    print(summary["short_answer"])


if __name__ == "__main__":
    main()
