#!/usr/bin/env python3
"""
Predictor-matrix reconstruction diagnostic for Z and reduced AMV/AMO K5 blocks.
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

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
    / "predictor_matrix_reconstruction_diagnostic"
)

FULL37_PATCH_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_predictors.csv"
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

MATRIX_TABLE_CSV = OUTPUT_DIR / "predictor_matrix_Z_A_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictor_matrix_reconstruction_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "predictor_matrix_reconstruction_metrics.csv"
COEFFICIENTS_CSV = OUTPUT_DIR / "predictor_matrix_reconstruction_coefficients.csv"
SUMMARY_JSON = OUTPUT_DIR / "predictor_matrix_reconstruction_summary.json"
HEATMAP_PNG = OUTPUT_DIR / "reconstruction_metric_heatmap.png"
Z_PANEL_PNG = OUTPUT_DIR / "Z_reconstruction_observed_vs_predicted.png"
A_PANEL_PNG = OUTPUT_DIR / "A_reconstruction_observed_vs_predicted.png"
CROSS_SCATTER_PNG = OUTPUT_DIR / "cross_block_reconstruction_scatter.png"
Z_BLOCK_ERROR_PNG = OUTPUT_DIR / "Z_block_reconstruction_error_norm_by_year.png"
A_BLOCK_ERROR_PNG = OUTPUT_DIR / "A_block_reconstruction_error_norm_by_year.png"
SUMMARY_TABLE_CSV = OUTPUT_DIR / "predictor_matrix_reconstruction_task_summary.csv"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
Z_COLUMNS = ["Z1_M1_Jan_lat_-9.5_lon_133.5", "Z2_M2_Oct_lat_0.5_lon_136.5"]
A_COLUMNS = ["AMV_PC4_Sep", "AMV_PC5_Feb", "AMV_PC2_Feb", "AMV_PC4_Nov", "AMV_PC5_Mar"]
ALPHA_GRID = np.asarray([1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)

TASK_SPECS = [
    ("Z_to_Z", Z_COLUMNS, Z_COLUMNS, "Z"),
    ("ZA_to_Z", Z_COLUMNS + A_COLUMNS, Z_COLUMNS, "Z"),
    ("A_to_A", A_COLUMNS, A_COLUMNS, "A"),
    ("ZA_to_A", Z_COLUMNS + A_COLUMNS, A_COLUMNS, "A"),
    ("A_to_Z", A_COLUMNS, Z_COLUMNS, "Z"),
    ("Z_to_A", Z_COLUMNS, A_COLUMNS, "A"),
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_mean = np.mean(y_train_raw, axis=0)
    y_std = np.std(y_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(y_std)) or np.any(y_std <= 0.0):
        raise ValueError("Target train-fold standard deviation must be positive.")
    return (
        (y_train_raw - y_mean[None, :]) / y_std[None, :],
        (y_test_raw - y_mean) / y_std,
        y_mean,
        y_std,
    )


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


def evaluate_alpha_grid_loyo_multioutput(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    n, q = y.shape
    preds_by_alpha = {alpha: np.full((n, q), np.nan, dtype=float) for alpha in ALPHA_GRID}
    for held_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[held_idx] = False
        x_train = x[train_mask, :]
        x_test = x[~train_mask, :][0]
        y_train = y[train_mask, :]
        y_test = y[~train_mask, :][0]
        x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
        y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train, y_test.reshape(1, -1))
        for alpha in ALPHA_GRID:
            pred_std, _ = ridge_fit_predict_standardized(
                x_train_std,
                y_train_std,
                x_test_std.reshape(1, -1),
                alpha,
            )
            preds_by_alpha[alpha][held_idx, :] = y_mean + y_std * pred_std[0, :]
    fro_rmse_by_alpha = {
        alpha: float(np.sqrt(np.mean((y - preds) ** 2)))
        for alpha, preds in preds_by_alpha.items()
    }
    best_alpha = min(ALPHA_GRID, key=lambda a: (fro_rmse_by_alpha[a], a))
    return float(best_alpha), float(fro_rmse_by_alpha[best_alpha]), preds_by_alpha[best_alpha]


def fit_outer_prediction_multioutput(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train, x_test)
    y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train, np.zeros((1, y_train.shape[1])))
    pred_std, coef_std = ridge_fit_predict_standardized(
        x_train_std,
        y_train_std,
        x_test_std.reshape(1, -1),
        alpha,
    )
    pred_raw = y_mean + y_std * pred_std[0, :]
    coef_raw = coef_std * (y_std[None, :] / x_std[:, None])
    intercept_raw = y_mean - x_mean @ coef_raw
    return pred_raw, coef_raw, intercept_raw


def load_matrix_table() -> Tuple[pd.DataFrame, Dict[str, object]]:
    z_df = pd.read_csv(FULL37_PATCH_CSV)
    z_df = z_df[z_df["patch_size"] == "exact_grid_cell"].copy()
    needed_z = ["water_year"] + Z_COLUMNS
    missing_z = [col for col in needed_z if col not in z_df.columns]
    if missing_z:
        raise ValueError(f"Missing Z columns: {missing_z}")
    z_df = z_df[needed_z].copy()

    a_source = AMV_K5_TABLE_CSV if AMV_K5_TABLE_CSV.exists() else AMV_FALLBACK_CSV
    a_df = pd.read_csv(a_source)
    needed_a = ["water_year"] + A_COLUMNS
    missing_a = [col for col in needed_a if col not in a_df.columns]
    if missing_a:
        raise ValueError(f"Missing A columns: {missing_a}")
    a_df = a_df[needed_a].copy()

    table = z_df.merge(a_df, on="water_year", how="inner")
    table["water_year"] = table["water_year"].astype(int)
    table = table.sort_values("water_year").reset_index(drop=True)
    table = table[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Merged predictor matrix table does not match WY1985--WY2021.")
    return table[["water_year"] + Z_COLUMNS + A_COLUMNS].copy(), {
        "z_file": str(FULL37_PATCH_CSV),
        "a_file": str(a_source),
    }


def run_task(table: pd.DataFrame, task_name: str, input_cols: Sequence[str], target_cols: Sequence[str], target_block: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    x_all = table[list(input_cols)].to_numpy(dtype=float)
    y_all = table[list(target_cols)].to_numpy(dtype=float)

    prediction_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []

    for held_idx, held_year in enumerate(years):
        train_mask = np.ones(len(years), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_all[train_mask, :]
        x_test = x_all[~train_mask, :][0]
        y_train = y_all[train_mask, :]
        y_test = y_all[~train_mask, :][0]

        alpha, inner_fro_rmse, _ = evaluate_alpha_grid_loyo_multioutput(x_train, y_train)
        pred_raw, coef_raw, intercept_raw = fit_outer_prediction_multioutput(x_train, y_train, x_test, alpha)
        alpha_rows.append(
            {
                "task_name": task_name,
                "heldout_wy": int(held_year),
                "selected_alpha": float(alpha),
                "inner_cv_frobenius_rmse": float(inner_fro_rmse),
                "input_cols": "|".join(input_cols),
                "target_cols": "|".join(target_cols),
            }
        )
        for j, target_col in enumerate(target_cols):
            err = float(pred_raw[j] - y_test[j])
            prediction_rows.append(
                {
                    "task_name": task_name,
                    "heldout_wy": int(held_year),
                    "target_col": target_col,
                    "obs_value": float(y_test[j]),
                    "pred_value": float(pred_raw[j]),
                    "error_pred_minus_obs": err,
                    "abs_error": float(abs(err)),
                    "selected_alpha": float(alpha),
                    "input_cols": "|".join(input_cols),
                    "target_cols": "|".join(target_cols),
                }
            )
            coefficient_rows.append(
                {
                    "task_name": task_name,
                    "heldout_wy": int(held_year),
                    "target_col": target_col,
                    "input_col": "__intercept__",
                    "coefficient": float(intercept_raw[j]),
                    "selected_alpha": float(alpha),
                }
            )
            for i, input_col in enumerate(input_cols):
                coefficient_rows.append(
                    {
                        "task_name": task_name,
                        "heldout_wy": int(held_year),
                        "target_col": target_col,
                        "input_col": input_col,
                        "coefficient": float(coef_raw[i, j]),
                        "selected_alpha": float(alpha),
                    }
                )

    pred_df = pd.DataFrame(prediction_rows).sort_values(["task_name", "target_col", "heldout_wy"]).reset_index(drop=True)
    coef_df = pd.DataFrame(coefficient_rows).sort_values(["task_name", "target_col", "heldout_wy", "input_col"]).reset_index(drop=True)
    alpha_df = pd.DataFrame(alpha_rows).sort_values(["task_name", "heldout_wy"]).reset_index(drop=True)
    pred_df["target_block"] = target_block
    return pred_df, coef_df, alpha_df


def compute_metrics(pred_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for task_name, task_pred in pred_df.groupby("task_name", sort=False):
        target_cols = task_pred["target_col"].drop_duplicates().tolist()
        per_target_rmse = []
        per_target_r2 = []
        per_target_mae = []
        for target_col in target_cols:
            sub = task_pred[task_pred["target_col"] == target_col].sort_values("heldout_wy")
            obs = sub["obs_value"].to_numpy(dtype=float)
            pred = sub["pred_value"].to_numpy(dtype=float)
            this_rmse = rmse(obs, pred)
            this_r2 = r2_manual(obs, pred)
            this_mae = mae(obs, pred)
            rows.append(
                {
                    "row_type": "per_target",
                    "task_name": task_name,
                    "target_block": str(sub["target_block"].iloc[0]),
                    "target_col": target_col,
                    "num_input_cols": int(len(sub["input_cols"].iloc[0].split("|"))),
                    "num_target_cols": int(len(target_cols)),
                    "r": corrcoef_safe(obs, pred),
                    "R2": this_r2,
                    "RMSE": this_rmse,
                    "MAE": this_mae,
                    "mean_abs_error": float(np.mean(np.abs(pred - obs))),
                    "median_abs_error": float(np.median(np.abs(pred - obs))),
                    "mean_R2": np.nan,
                    "mean_RMSE": np.nan,
                    "frobenius_RMSE": np.nan,
                    "mean_MAE": np.nan,
                    "max_target_RMSE": np.nan,
                    "selected_alpha_mode_or_median": np.nan,
                }
            )
            per_target_rmse.append(this_rmse)
            per_target_r2.append(this_r2)
            per_target_mae.append(this_mae)

        matrix_obs = []
        matrix_pred = []
        for target_col in target_cols:
            sub = task_pred[task_pred["target_col"] == target_col].sort_values("heldout_wy")
            matrix_obs.append(sub["obs_value"].to_numpy(dtype=float))
            matrix_pred.append(sub["pred_value"].to_numpy(dtype=float))
        obs_mat = np.column_stack(matrix_obs)
        pred_mat = np.column_stack(matrix_pred)
        alpha_values = alpha_df[alpha_df["task_name"] == task_name]["selected_alpha"].to_numpy(dtype=float)
        rows.append(
            {
                "row_type": "aggregate",
                "task_name": task_name,
                "target_block": str(task_pred["target_block"].iloc[0]),
                "target_col": "__aggregate__",
                "num_input_cols": int(len(task_pred["input_cols"].iloc[0].split("|"))),
                "num_target_cols": int(len(target_cols)),
                "r": np.nan,
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "mean_abs_error": np.nan,
                "median_abs_error": np.nan,
                "mean_R2": float(np.mean(per_target_r2)),
                "mean_RMSE": float(np.mean(per_target_rmse)),
                "frobenius_RMSE": float(np.sqrt(np.mean((obs_mat - pred_mat) ** 2))),
                "mean_MAE": float(np.mean(per_target_mae)),
                "max_target_RMSE": float(np.max(per_target_rmse)),
                "selected_alpha_mode_or_median": float(np.median(alpha_values)),
            }
        )
    return pd.DataFrame(rows)


def compute_block_error_by_year(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (task_name, heldout_wy), sub in pred_df.groupby(["task_name", "heldout_wy"], sort=False):
        errs = sub.sort_values("target_col")["error_pred_minus_obs"].to_numpy(dtype=float)
        rows.append(
            {
                "task_name": task_name,
                "heldout_wy": int(heldout_wy),
                "target_block": str(sub["target_block"].iloc[0]),
                "num_target_cols": int(len(sub)),
                "block_error_l2": float(np.sqrt(np.sum(errs ** 2))),
                "block_rmse_within_year": float(np.sqrt(np.mean(errs ** 2))),
            }
        )
    return pd.DataFrame(rows).sort_values(["task_name", "heldout_wy"]).reset_index(drop=True)


def build_task_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    agg = metrics_df[metrics_df["row_type"] == "aggregate"].copy()
    keep = [
        "task_name",
        "target_block",
        "num_input_cols",
        "num_target_cols",
        "mean_R2",
        "mean_RMSE",
        "frobenius_RMSE",
        "mean_MAE",
        "max_target_RMSE",
        "selected_alpha_mode_or_median",
    ]
    return agg[keep].sort_values(
        "task_name",
        key=lambda s: s.map({task: i for i, (task, *_rest) in enumerate(TASK_SPECS)}),
    ).reset_index(drop=True)


def summarize_coefficients(coef_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for task_name in ["ZA_to_Z", "ZA_to_A", "A_to_Z", "Z_to_A"]:
        sub = coef_df[(coef_df["task_name"] == task_name) & (coef_df["input_col"] != "__intercept__")].copy()
        task_summary: Dict[str, object] = {}
        for target_col in sub["target_col"].drop_duplicates():
            target_sub = sub[sub["target_col"] == target_col]
            rows = []
            for input_col in target_sub["input_col"].drop_duplicates():
                vals = target_sub[target_sub["input_col"] == input_col]["coefficient"].to_numpy(dtype=float)
                rows.append(
                    {
                        "input_col": input_col,
                        "mean_abs_coefficient": float(np.mean(np.abs(vals))),
                        "mean_coefficient": float(np.mean(vals)),
                    }
                )
            rows.sort(key=lambda r: (-r["mean_abs_coefficient"], r["input_col"]))
            task_summary[target_col] = rows[:5]
        summary[task_name] = task_summary
    return summary


def make_heatmap(metrics_df: pd.DataFrame) -> None:
    agg = metrics_df[metrics_df["row_type"] == "aggregate"].copy()
    cols = ["mean_R2", "mean_RMSE", "frobenius_RMSE", "max_target_RMSE"]
    mat = agg.set_index("task_name")[cols].loc[[task for task, *_ in TASK_SPECS]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(TASK_SPECS)))
    ax.set_yticklabels([task for task, *_ in TASK_SPECS])
    ax.set_title("Predictor-matrix reconstruction metrics")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=200)
    plt.close(fig)


def make_z_panel(pred_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    for ax, target_col in zip(axes, Z_COLUMNS):
        for task_name, color, label in [
            ("Z_to_Z", "#1f77b4", "Z_to_Z"),
            ("ZA_to_Z", "#ff7f0e", "ZA_to_Z"),
            ("A_to_Z", "#2ca02c", "A_to_Z"),
        ]:
            sub = pred_df[(pred_df["task_name"] == task_name) & (pred_df["target_col"] == target_col)].sort_values("heldout_wy")
            if task_name == "Z_to_Z":
                ax.plot(sub["heldout_wy"], sub["obs_value"], color="black", linewidth=2.2, label="Observed")
            ax.plot(sub["heldout_wy"], sub["pred_value"], marker="o", linewidth=1.5, color=color, label=label)
        ax.set_title(target_col)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, ncol=4)
    axes[-1].set_xlabel("Held-out water year")
    fig.tight_layout()
    fig.savefig(Z_PANEL_PNG, dpi=200)
    plt.close(fig)


def make_a_panel(pred_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(A_COLUMNS), 1, figsize=(12.0, 15.0), sharex=True)
    for ax, target_col in zip(axes, A_COLUMNS):
        for task_name, color, label in [
            ("A_to_A", "#1f77b4", "A_to_A"),
            ("ZA_to_A", "#ff7f0e", "ZA_to_A"),
            ("Z_to_A", "#2ca02c", "Z_to_A"),
        ]:
            sub = pred_df[(pred_df["task_name"] == task_name) & (pred_df["target_col"] == target_col)].sort_values("heldout_wy")
            if task_name == "A_to_A":
                ax.plot(sub["heldout_wy"], sub["obs_value"], color="black", linewidth=2.0, label="Observed")
            ax.plot(sub["heldout_wy"], sub["pred_value"], marker="o", linewidth=1.3, color=color, label=label)
        ax.set_title(target_col)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, ncol=4)
    axes[-1].set_xlabel("Held-out water year")
    fig.tight_layout()
    fig.savefig(A_PANEL_PNG, dpi=200)
    plt.close(fig)


def make_cross_scatter(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    targets = Z_COLUMNS + A_COLUMNS
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.0))
    axes_flat = axes.flatten()
    metric_lookup = metrics_df[metrics_df["row_type"] == "per_target"].set_index(["task_name", "target_col"])
    for ax, target_col in zip(axes_flat, targets):
        task_name = "A_to_Z" if target_col in Z_COLUMNS else "Z_to_A"
        sub = pred_df[(pred_df["task_name"] == task_name) & (pred_df["target_col"] == target_col)].sort_values("heldout_wy")
        obs = sub["obs_value"].to_numpy(dtype=float)
        pred = sub["pred_value"].to_numpy(dtype=float)
        lo = min(np.min(obs), np.min(pred))
        hi = max(np.max(obs), np.max(pred))
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        lims = (lo - pad, hi + pad)
        ax.scatter(obs, pred, s=42, alpha=0.85, color="#4c72b0")
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.1)
        metric_row = metric_lookup.loc[(task_name, target_col)]
        ax.text(
            0.04,
            0.96,
            f"RMSE={float(metric_row['RMSE']):.3f}\nR2={float(metric_row['R2']):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
        )
        ax.set_title(f"{task_name}: {target_col}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
    for ax in axes_flat[len(targets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(CROSS_SCATTER_PNG, dpi=200)
    plt.close(fig)


def make_block_error_plot(block_error_df: pd.DataFrame, target_block: str, output_path: Path, task_order: Sequence[str]) -> None:
    subset = block_error_df[block_error_df["target_block"] == target_block].copy()
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    colors = {
        task_order[0]: "#1f77b4",
        task_order[1]: "#ff7f0e",
        task_order[2]: "#2ca02c",
    }
    for task_name in task_order:
        sub = subset[subset["task_name"] == task_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["block_error_l2"],
            marker="o",
            linewidth=1.6,
            color=colors[task_name],
            label=task_name,
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel(r"$\|Y-\widehat{Y}\|_2$ by year")
    ax.set_title(f"{target_block}-block reconstruction error norm by year")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_summary(
    input_files: Dict[str, object],
    metrics_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    block_error_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
) -> Dict[str, object]:
    agg = metrics_df[metrics_df["row_type"] == "aggregate"].set_index("task_name")
    z_interference = {
        "baseline": agg.loc["Z_to_Z", ["mean_R2", "frobenius_RMSE"]].to_dict(),
        "combined": agg.loc["ZA_to_Z", ["mean_R2", "frobenius_RMSE"]].to_dict(),
        "delta_Z_RMSE": float(agg.loc["ZA_to_Z", "frobenius_RMSE"] - agg.loc["Z_to_Z", "frobenius_RMSE"]),
        "delta_Z_R2": float(agg.loc["ZA_to_Z", "mean_R2"] - agg.loc["Z_to_Z", "mean_R2"]),
    }
    a_interference = {
        "baseline": agg.loc["A_to_A", ["mean_R2", "frobenius_RMSE"]].to_dict(),
        "combined": agg.loc["ZA_to_A", ["mean_R2", "frobenius_RMSE"]].to_dict(),
        "delta_A_RMSE": float(agg.loc["ZA_to_A", "frobenius_RMSE"] - agg.loc["A_to_A", "frobenius_RMSE"]),
        "delta_A_R2": float(agg.loc["ZA_to_A", "mean_R2"] - agg.loc["A_to_A", "mean_R2"]),
    }
    cross_overlap = {
        "A_to_Z": agg.loc["A_to_Z", ["mean_R2", "frobenius_RMSE", "mean_RMSE"]].to_dict(),
        "Z_to_A": agg.loc["Z_to_A", ["mean_R2", "frobenius_RMSE", "mean_RMSE"]].to_dict(),
    }
    coefficient_summary = summarize_coefficients(coef_df)

    z_short = (
        "Adding AMV/AMO K5 does not materially interfere with reconstructing Z when Z is present."
        if abs(z_interference["delta_Z_RMSE"]) < 1.0e-4 and abs(z_interference["delta_Z_R2"]) < 1.0e-3
        else "Adding AMV/AMO K5 changes the Z-block reconstruction when Z is present, indicating some regularized block interference."
    )
    a_short = (
        "Adding Z does not materially interfere with reconstructing the AMV/AMO K5 block when A is present."
        if abs(a_interference["delta_A_RMSE"]) < 1.0e-4 and abs(a_interference["delta_A_R2"]) < 1.0e-3
        else "Adding Z changes the AMV/AMO K5 reconstruction when A is present, indicating some regularized block interference."
    )
    cross_short = (
        "AMV/AMO K5 does not strongly reconstruct Z1/Z2 without Z."
        if float(cross_overlap["A_to_Z"]["mean_R2"]) < 0.5
        else "AMV/AMO K5 contains substantial overlap with Z1/Z2."
    )
    cross_short_2 = (
        "Z1/Z2 do not strongly reconstruct the full K5 AMV block without A."
        if float(cross_overlap["Z_to_A"]["mean_R2"]) < 0.5
        else "Z1/Z2 contain substantial overlap with the K5 AMV block."
    )

    z_yearly = block_error_df[block_error_df["target_block"] == "Z"]
    a_yearly = block_error_df[block_error_df["target_block"] == "A"]
    z_interpret = (
        "Adding AMV/AMO K5 does not interfere with preserving the Z block in a reconstruction task."
        if abs(z_interference["delta_Z_RMSE"]) < 1.0e-4 and abs(z_interference["delta_Z_R2"]) < 1.0e-3
        else "Adding AMV/AMO K5 interferes with preserving the Z block even when Z is present in the input."
    )
    a_interpret = (
        "Adding Z does not interfere with preserving the A block in a reconstruction task."
        if abs(a_interference["delta_A_RMSE"]) < 1.0e-4 and abs(a_interference["delta_A_R2"]) < 1.0e-3
        else "Adding Z interferes with preserving the A block even when A is present in the input."
    )
    return {
        "input_files": input_files,
        "output_dir": str(OUTPUT_DIR),
        "water_years": [WATER_YEAR_START, WATER_YEAR_END],
        "Z_columns": Z_COLUMNS,
        "A_columns": A_COLUMNS,
        "tasks": [
            {
                "task_name": task_name,
                "input_cols": input_cols,
                "target_cols": target_cols,
                "target_block": target_block,
            }
            for task_name, input_cols, target_cols, target_block in TASK_SPECS
        ],
        "metrics_by_task": metrics_df.to_dict(orient="records"),
        "task_summary_table": task_summary_df.to_dict(orient="records"),
        "block_error_by_year": block_error_df.to_dict(orient="records"),
        "Z_block_interference": z_interference,
        "A_block_interference": a_interference,
        "cross_block_overlap": cross_overlap,
        "coefficient_summary": coefficient_summary,
        "short_answer": f"{z_interpret} {a_interpret} {cross_short} {cross_short_2}",
        "final_interpretation": {
            "Z_block": z_interpret,
            "A_block": a_interpret,
            "cross_block_Z_from_A": cross_short,
            "cross_block_A_from_Z": cross_short_2,
            "yearwise_note": {
                "max_Z_block_error_l2": float(z_yearly["block_error_l2"].max()),
                "max_A_block_error_l2": float(a_yearly["block_error_l2"].max()),
            },
        },
    }


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, input_files = load_matrix_table()
    table.to_csv(MATRIX_TABLE_CSV, index=False)

    all_pred = []
    all_coef = []
    all_alpha = []
    for task_name, input_cols, target_cols, target_block in TASK_SPECS:
        pred_df, coef_df, alpha_df = run_task(table, task_name, input_cols, target_cols, target_block)
        all_pred.append(pred_df)
        all_coef.append(coef_df)
        all_alpha.append(alpha_df)

    pred_df = pd.concat(all_pred, ignore_index=True)
    coef_df = pd.concat(all_coef, ignore_index=True)
    alpha_df = pd.concat(all_alpha, ignore_index=True)
    metrics_df = compute_metrics(pred_df, alpha_df)
    block_error_df = compute_block_error_by_year(pred_df)
    task_summary_df = build_task_summary_table(metrics_df)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    coef_df.to_csv(COEFFICIENTS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    block_error_df.to_csv(OUTPUT_DIR / "predictor_matrix_reconstruction_block_error_by_year.csv", index=False)
    task_summary_df.to_csv(SUMMARY_TABLE_CSV, index=False)

    make_heatmap(metrics_df)
    make_block_error_plot(block_error_df, "Z", Z_BLOCK_ERROR_PNG, ["Z_to_Z", "ZA_to_Z", "A_to_Z"])
    make_block_error_plot(block_error_df, "A", A_BLOCK_ERROR_PNG, ["A_to_A", "ZA_to_A", "Z_to_A"])

    summary = build_summary(input_files, metrics_df, coef_df, block_error_df, task_summary_df)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    agg = metrics_df[metrics_df["row_type"] == "aggregate"].set_index("task_name")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Task metrics:")
    for task_name, *_ in TASK_SPECS:
        row = agg.loc[task_name]
        print(
            f"{task_name}: mean_R2={float(row['mean_R2']):.6f}, mean_RMSE={float(row['mean_RMSE']):.6f}, "
            f"frobenius_RMSE={float(row['frobenius_RMSE']):.6f}"
        )
    print("Block interference:")
    print(
        "Z_to_Z vs ZA_to_Z: delta_Z_RMSE={:.6e}, delta_Z_R2={:.6e}".format(
            summary["Z_block_interference"]["delta_Z_RMSE"],
            summary["Z_block_interference"]["delta_Z_R2"],
        )
    )
    print(
        "A_to_A vs ZA_to_A: delta_A_RMSE={:.6e}, delta_A_R2={:.6e}".format(
            summary["A_block_interference"]["delta_A_RMSE"],
            summary["A_block_interference"]["delta_A_R2"],
        )
    )
    print("Cross-block overlap:")
    print(
        "A_to_Z: mean_R2={:.6f}, frobenius_RMSE={:.6f}".format(
            summary["cross_block_overlap"]["A_to_Z"]["mean_R2"],
            summary["cross_block_overlap"]["A_to_Z"]["frobenius_RMSE"],
        )
    )
    print(
        "Z_to_A: mean_R2={:.6f}, frobenius_RMSE={:.6f}".format(
            summary["cross_block_overlap"]["Z_to_A"]["mean_R2"],
            summary["cross_block_overlap"]["Z_to_A"]["frobenius_RMSE"],
        )
    )
    print("Summary table:")
    print(task_summary_df.to_string(index=False))
    print("Short answer:")
    print(summary["short_answer"])


if __name__ == "__main__":
    main()
