#!/usr/bin/env python3
"""
Strict nested-LOYO ridge experiment for exact Z1+Z2 plus AMV PC2-PC4 Sep-Mar predictors.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo"
)
PATCH_PREDICTORS_CSV = BASE_DIR / "full37_patch_predictors.csv"
BASE_PREDICTIONS_CSV = BASE_DIR / "full37_patch_loyo_predictions.csv"
BASE_METRICS_CSV = BASE_DIR / "full37_patch_loyo_metrics.csv"
BASE_BETA_CSV = BASE_DIR / "full37_patch_beta_by_fold.csv"
BASE_SUMMARY_JSON = BASE_DIR / "full37_patch_predictor_summary.json"
AMV_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_plus_AMV_PC2to4_loyo"
)

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "z1_z2_amv_pc2to4_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "z1_z2_amv_pc2to4_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "z1_z2_amv_pc2to4_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "z1_z2_amv_pc2to4_loyo_period_metrics.csv"
BETA_CSV = OUTPUT_DIR / "z1_z2_amv_pc2to4_beta_by_fold.csv"
ALPHA_CSV = OUTPUT_DIR / "z1_z2_amv_pc2to4_selected_alpha_by_fold.csv"
SUMMARY_JSON = OUTPUT_DIR / "z1_z2_amv_pc2to4_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "z1_z2_amv_pc2to4_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "z1_z2_amv_pc2to4_scatter.png"
METRICS_PNG = OUTPUT_DIR / "z1_z2_amv_pc2to4_metrics_comparison.png"
COEF_HEATMAP_PNG = OUTPUT_DIR / "z1_z2_amv_pc2to4_coefficients_heatmap.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
Z1_NAME = "Z1_M1_Jan_lat_-9.5_lon_133.5"
Z2_NAME = "Z2_M2_Oct_lat_0.5_lon_136.5"
Z1_OUTPUT = "Z1"
Z2_OUTPUT = "Z2"
AMV_COLUMNS = ["AMV_PC{}_{}".format(pc, month) for pc in (2, 3, 4) for month in MONTHS]
MODEL_SPECS = [
    ("Z1_Z2", [Z1_OUTPUT, Z2_OUTPUT]),
    ("Z1_Z2_AMV_PC2to4", [Z1_OUTPUT, Z2_OUTPUT] + AMV_COLUMNS),
    ("AMV_PC2to4_only", list(AMV_COLUMNS)),
    ("Z1_Z2_AMV_PC2_only", [Z1_OUTPUT, Z2_OUTPUT] + ["AMV_PC2_{}".format(month) for month in MONTHS]),
    ("Z1_Z2_AMV_PC3_only", [Z1_OUTPUT, Z2_OUTPUT] + ["AMV_PC3_{}".format(month) for month in MONTHS]),
    ("Z1_Z2_AMV_PC4_only", [Z1_OUTPUT, Z2_OUTPUT] + ["AMV_PC4_{}".format(month) for month in MONTHS]),
]
MAIN_PLOT_MODELS = ["Z1_Z2", "Z1_Z2_AMV_PC2to4", "AMV_PC2to4_only"]
GROUP_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]


def ensure_runtime_on_compute_node():
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def corrcoef_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true, y_pred):
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


def rmse(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_sign_accuracy(obs, pred):
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs, pred):
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
    }


def compute_period_metrics(water_years, obs, pred):
    rows = []
    for group_name, selector in GROUP_SPECS:
        mask = selector(water_years)
        yy = obs[mask]
        pp = pred[mask]
        metrics = compute_metric_bundle(yy, pp)
        error = pp - yy
        rows.append(
            {
                "group_name": group_name,
                "n_years": int(mask.sum()),
                "r": metrics["r"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_error": float(np.mean(error)),
                "mean_abs_error": float(np.mean(np.abs(error))),
            }
        )
    return rows


def write_csv(path, fieldnames, rows):
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def load_base_tables():
    predictors = pd.read_csv(PATCH_PREDICTORS_CSV)
    predictors = predictors.loc[predictors["patch_size"] == "exact_grid_cell"].copy()
    predictors = predictors[["water_year", Z1_NAME, Z2_NAME]].sort_values("water_year").reset_index(drop=True)
    predictors["water_year"] = predictors["water_year"].astype(int)
    predictors = predictors.rename(columns={Z1_NAME: Z1_OUTPUT, Z2_NAME: Z2_OUTPUT})

    base_predictions = pd.read_csv(BASE_PREDICTIONS_CSV)
    base_predictions = base_predictions.loc[
        (base_predictions["patch_size"] == "exact_grid_cell") & (base_predictions["model_name"] == "Z1_Z2")
    ].copy()
    base_predictions = base_predictions[["heldout_wy", "obs_swe"]].rename(columns={"heldout_wy": "water_year"})
    base_predictions["water_year"] = base_predictions["water_year"].astype(int)
    base_predictions = base_predictions.sort_values("water_year").reset_index(drop=True)

    amv = pd.read_csv(AMV_CSV)
    amv["water_year"] = amv["water_year"].astype(int)
    amv = amv[["water_year"] + AMV_COLUMNS].sort_values("water_year").reset_index(drop=True)
    return predictors, base_predictions, amv


def build_predictor_table():
    predictors, base_predictions, amv = load_base_tables()
    merged = predictors.merge(base_predictions, on="water_year", how="inner")
    merged = merged.merge(amv, on="water_year", how="inner")
    merged = merged.sort_values("water_year").reset_index(drop=True)
    merged = merged.loc[(merged["water_year"] >= WATER_YEAR_START) & (merged["water_year"] <= WATER_YEAR_END)].copy()
    expected_years = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
    if merged["water_year"].tolist() != expected_years:
        raise ValueError("Predictor table years do not match WY1985-WY2021.")
    return merged[["water_year", "obs_swe", Z1_OUTPUT, Z2_OUTPUT] + AMV_COLUMNS].copy()


def standardize_train_only(x_train_raw, x_test_raw):
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("One or more predictor columns have non-positive train-fold standard deviation.")
    x_train_std = (x_train_raw - x_mean[None, :]) / x_std[None, :]
    x_test_std = (x_test_raw - x_mean) / x_std
    return x_train_std, x_test_std, x_mean, x_std


def standardize_target_train_only(y_train_raw, y_test_raw):
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Train-fold target standard deviation is non-positive.")
    y_train_std = (y_train_raw - y_mean) / y_std
    y_test_std = (float(y_test_raw) - y_mean) / y_std
    return y_train_std, y_test_std, y_mean, y_std


def fit_ridge_standardized(x_train_std, y_train_std, alpha):
    xtx = x_train_std.T @ x_train_std
    ridge = xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64)
    rhs = x_train_std.T @ y_train_std
    beta_std = np.linalg.solve(ridge, rhs)
    return np.asarray(beta_std, dtype=np.float64)


def inner_loyo_best_alpha(x_train_raw, y_train_raw):
    best_alpha = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for alpha in ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train = x_train_raw[inner_mask]
            x_inner_test = x_train_raw[~inner_mask][0]
            y_inner_train = y_train_raw[inner_mask]
            y_inner_test = float(y_train_raw[~inner_mask][0])

            x_inner_train_std, x_inner_test_std, _, _ = standardize_train_only(x_inner_train, x_inner_test)
            y_inner_train_std, _, y_mean, y_std = standardize_target_train_only(y_inner_train, y_inner_test)
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, alpha)
            pred_std = float(x_inner_test_std @ beta_std)
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if (best_mse is None) or (mse < best_mse - 1.0e-15) or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = float(mse)
    return best_alpha, best_mse


def run_nested_loyo_for_model(data_df, model_name, predictor_columns):
    water_years = data_df["water_year"].to_numpy(dtype=np.int32)
    obs = data_df["obs_swe"].to_numpy(dtype=np.float64)
    x_all = data_df[predictor_columns].to_numpy(dtype=np.float64)

    preds = np.full(obs.shape, np.nan, dtype=np.float64)
    selected_alphas = np.full(obs.shape, np.nan, dtype=np.float64)
    inner_cv_mse = np.full(obs.shape, np.nan, dtype=np.float64)
    beta_std_matrix = np.full((len(obs), len(predictor_columns)), np.nan, dtype=np.float64)
    beta_raw_matrix = np.full((len(obs), len(predictor_columns)), np.nan, dtype=np.float64)
    intercept_raw = np.full(len(obs), np.nan, dtype=np.float64)

    prediction_rows = []
    alpha_rows = []
    beta_rows = []

    for fold_index, heldout_wy in enumerate(water_years.tolist()):
        test_mask = water_years == heldout_wy
        train_mask = ~test_mask
        x_train_raw = x_all[train_mask]
        x_test_raw = x_all[test_mask][0]
        y_train_raw = obs[train_mask]
        y_test_raw = float(obs[test_mask][0])

        best_alpha, best_mse = inner_loyo_best_alpha(x_train_raw, y_train_raw)
        x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
        y_train_std, _, y_mean, y_std = standardize_target_train_only(y_train_raw, y_test_raw)
        beta_std = fit_ridge_standardized(x_train_std, y_train_std, best_alpha)
        pred_std = float(x_test_std @ beta_std)
        pred_raw = float(y_mean + y_std * pred_std)

        beta_raw = y_std * beta_std / x_std
        intercept = float(y_mean - np.sum((x_mean / x_std) * y_std * beta_std))

        preds[fold_index] = pred_raw
        selected_alphas[fold_index] = best_alpha
        inner_cv_mse[fold_index] = best_mse
        beta_std_matrix[fold_index, :] = beta_std
        beta_raw_matrix[fold_index, :] = beta_raw
        intercept_raw[fold_index] = intercept

        error_pred_minus_obs = float(pred_raw - y_test_raw)
        residual_obs_minus_pred = float(y_test_raw - pred_raw)
        sign_correct = float("nan")
        if y_test_raw != 0.0 and pred_raw != 0.0:
            sign_correct = 1.0 if np.sign(y_test_raw) == np.sign(pred_raw) else 0.0
        prediction_rows.append(
            {
                "model_name": model_name,
                "heldout_wy": int(heldout_wy),
                "obs_swe": float(y_test_raw),
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error_pred_minus_obs,
                "residual_obs_minus_pred": residual_obs_minus_pred,
                "abs_error": float(abs(error_pred_minus_obs)),
                "sign_correct": sign_correct,
                "selected_alpha": best_alpha,
                "num_predictors": int(len(predictor_columns)),
            }
        )
        alpha_rows.append(
            {
                "model_name": model_name,
                "heldout_wy": int(heldout_wy),
                "selected_alpha": best_alpha,
                "inner_cv_mse": best_mse,
            }
        )
        beta_row = {
            "model_name": model_name,
            "heldout_wy": int(heldout_wy),
            "intercept": intercept,
        }
        for name in [Z1_OUTPUT, Z2_OUTPUT] + AMV_COLUMNS:
            beta_row["beta_" + name] = float("nan")
            beta_row["beta_std_" + name] = float("nan")
        for col_idx, name in enumerate(predictor_columns):
            beta_row["beta_" + name] = float(beta_raw[col_idx])
            beta_row["beta_std_" + name] = float(beta_std[col_idx])
        beta_rows.append(beta_row)

    metric_bundle = compute_metric_bundle(obs, preds)
    metric_row = {
        "model_name": model_name,
        "num_predictors": int(len(predictor_columns)),
        "r": metric_bundle["r"],
        "R2": metric_bundle["R2"],
        "RMSE": metric_bundle["RMSE"],
        "MAE": metric_bundle["MAE"],
        "sign_accuracy": metric_bundle["sign_accuracy"],
    }
    period_rows = []
    for row in compute_period_metrics(water_years, obs, preds):
        row["model_name"] = model_name
        row["num_predictors"] = int(len(predictor_columns))
        period_rows.append(row)

    alpha_summary = {
        "min_alpha": float(np.min(selected_alphas)),
        "median_alpha": float(np.median(selected_alphas)),
        "max_alpha": float(np.max(selected_alphas)),
        "mean_alpha": float(np.mean(selected_alphas)),
        "alpha_counts": {str(alpha): int(np.sum(np.isclose(selected_alphas, alpha))) for alpha in ALPHA_GRID.tolist()},
    }
    coef_summary = {
        "mean_abs_beta_std_by_predictor": {
            name: float(np.nanmean(np.abs(beta_std_matrix[:, idx]))) for idx, name in enumerate(predictor_columns)
        },
        "std_beta_std_by_predictor": {
            name: float(np.nanstd(beta_std_matrix[:, idx], ddof=1)) for idx, name in enumerate(predictor_columns)
        },
    }
    return {
        "preds": preds,
        "prediction_rows": prediction_rows,
        "alpha_rows": alpha_rows,
        "beta_rows": beta_rows,
        "metric_row": metric_row,
        "period_rows": period_rows,
        "alpha_summary": alpha_summary,
        "coef_summary": coef_summary,
        "beta_std_matrix": beta_std_matrix,
        "predictor_columns": predictor_columns,
    }


def plot_observed_vs_predicted(data_df, predictions_by_model):
    fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
    water_years = data_df["water_year"].to_numpy(dtype=np.int32)
    obs = data_df["obs_swe"].to_numpy(dtype=np.float64)
    ax.plot(water_years, obs, color="black", linewidth=1.9, label="Observed SWE anomaly")
    color_map = {
        "Z1_Z2": "tab:blue",
        "Z1_Z2_AMV_PC2to4": "tab:red",
        "AMV_PC2to4_only": "tab:green",
        "Z1_Z2_AMV_PC2_only": "tab:orange",
        "Z1_Z2_AMV_PC3_only": "tab:purple",
        "Z1_Z2_AMV_PC4_only": "tab:brown",
    }
    for model_name in MAIN_PLOT_MODELS:
        ax.plot(water_years, predictions_by_model[model_name], linewidth=1.25, color=color_map[model_name], label=model_name)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO ridge: exact Z1+Z2 with AMV PC2-PC4")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OBS_PRED_PNG, dpi=220)
    plt.close(fig)


def plot_scatter(data_df, predictions_by_model):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    obs = data_df["obs_swe"].to_numpy(dtype=np.float64)
    for ax, model_name in zip(axes, MAIN_PLOT_MODELS):
        pred = predictions_by_model[model_name]
        ax.scatter(obs, pred, color="tab:blue", alpha=0.85)
        lo = float(min(np.min(obs), np.min(pred)))
        hi = float(max(np.max(obs), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=0.9)
        ax.set_title(model_name)
        ax.set_xlabel("Observed SWE anomaly")
        ax.set_ylabel("Predicted SWE anomaly")
        ax.grid(True, linewidth=0.25, color="0.85")
    fig.savefig(SCATTER_PNG, dpi=220)
    plt.close(fig)


def plot_metrics_comparison(metrics_df):
    fields = ["r", "R2", "RMSE", "MAE", "sign_accuracy"]
    fig, axes = plt.subplots(len(fields), 1, figsize=(12.0, 12.0), constrained_layout=True)
    x = np.arange(len(metrics_df))
    labels = metrics_df["model_name"].tolist()
    for ax, field in zip(axes, fields):
        values = metrics_df[field].to_numpy(dtype=float)
        ax.bar(x, values, color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(field)
        ax.grid(True, axis="y", linewidth=0.25, color="0.85")
        if field == "R2":
            ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        if field == "sign_accuracy":
            ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    fig.savefig(METRICS_PNG, dpi=220)
    plt.close(fig)


def plot_coefficients_heatmap(beta_std_matrix, heldout_years, predictor_columns):
    fig, ax = plt.subplots(figsize=(15.0, 8.0), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(beta_std_matrix))) if np.any(np.isfinite(beta_std_matrix)) else 1.0
    image = ax.imshow(beta_std_matrix.T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(heldout_years)))
    ax.set_xticklabels([str(int(y)) for y in heldout_years], rotation=90)
    ax.set_yticks(np.arange(len(predictor_columns)))
    ax.set_yticklabels(predictor_columns)
    ax.set_title("Standardized ridge coefficients by held-out fold: Z1_Z2_AMV_PC2to4")
    fig.colorbar(image, ax=ax, shrink=0.9, label="standardized coefficient")
    fig.savefig(COEF_HEATMAP_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictor_table = build_predictor_table()
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    predictions_rows = []
    alpha_rows = []
    beta_rows = []
    metrics_rows = []
    period_rows = []
    predictions_by_model = {}
    alpha_summaries = {}
    coef_summaries = {}

    for model_name, predictor_columns in MODEL_SPECS:
        result = run_nested_loyo_for_model(predictor_table, model_name, predictor_columns)
        predictions_rows.extend(result["prediction_rows"])
        alpha_rows.extend(result["alpha_rows"])
        beta_rows.extend(result["beta_rows"])
        metrics_rows.append(result["metric_row"])
        period_rows.extend(result["period_rows"])
        predictions_by_model[model_name] = result["preds"]
        alpha_summaries[model_name] = result["alpha_summary"]
        coef_summaries[model_name] = result["coef_summary"]
        if model_name == "Z1_Z2_AMV_PC2to4":
            plot_coefficients_heatmap(
                result["beta_std_matrix"],
                predictor_table["water_year"].to_numpy(dtype=np.int32),
                result["predictor_columns"],
            )

    metrics_df = pd.DataFrame(metrics_rows)
    best_r2 = metrics_df.loc[metrics_df["R2"].idxmax()]
    best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    best_sign = metrics_df.loc[metrics_df["sign_accuracy"].idxmax()]
    metrics_df["best_by_R2"] = metrics_df["model_name"] == str(best_r2["model_name"])
    metrics_df["best_by_RMSE"] = metrics_df["model_name"] == str(best_rmse["model_name"])
    metrics_df["best_by_sign_accuracy"] = metrics_df["model_name"] == str(best_sign["model_name"])
    metrics_df.to_csv(METRICS_CSV, index=False)
    pd.DataFrame(period_rows).to_csv(PERIOD_METRICS_CSV, index=False)
    pd.DataFrame(predictions_rows).to_csv(PREDICTIONS_CSV, index=False)
    pd.DataFrame(alpha_rows).to_csv(ALPHA_CSV, index=False)
    pd.DataFrame(beta_rows).to_csv(BETA_CSV, index=False)

    plot_observed_vs_predicted(predictor_table, predictions_by_model)
    plot_scatter(predictor_table, predictions_by_model)
    plot_metrics_comparison(metrics_df)

    base_metrics = pd.read_csv(BASE_METRICS_CSV)
    base_ref = base_metrics.loc[
        (base_metrics["patch_size"] == "exact_grid_cell") & (base_metrics["model_name"] == "Z1_Z2")
    ].iloc[0].to_dict()
    ridge_ref = metrics_df.loc[metrics_df["model_name"] == "Z1_Z2"].iloc[0]
    combined_ref = metrics_df.loc[metrics_df["model_name"] == "Z1_Z2_AMV_PC2to4"].iloc[0]

    if (float(combined_ref["R2"]) > float(ridge_ref["R2"])) and (float(combined_ref["RMSE"]) < float(ridge_ref["RMSE"])) and (
        float(combined_ref["sign_accuracy"]) >= float(ridge_ref["sign_accuracy"]) - 0.05
    ):
        short_answer = "Adding AMV/AMO PC2--PC4 improves the strict LOYO prediction relative to the ridge Z1_Z2 baseline."
    elif (float(combined_ref["R2"]) > float(ridge_ref["R2"])) or (float(combined_ref["RMSE"]) < float(ridge_ref["RMSE"])):
        short_answer = (
            "Adding AMV/AMO PC2--PC4 gives mixed improvement; the additional predictors may explain some residual structure "
            "but do not robustly improve all skill metrics."
        )
    else:
        short_answer = (
            "Although AMV/AMO PC2--PC4 are associated with the Z1_Z2 residuals in screening, adding them does not improve strict LOYO prediction."
        )

    summary_payload = {
        "input_files": {
            "patch_predictors_csv": str(PATCH_PREDICTORS_CSV),
            "base_predictions_csv": str(BASE_PREDICTIONS_CSV),
            "base_metrics_csv": str(BASE_METRICS_CSV),
            "base_beta_csv": str(BASE_BETA_CSV),
            "base_summary_json": str(BASE_SUMMARY_JSON),
            "amv_csv": str(AMV_CSV),
        },
        "output_dir": str(OUTPUT_DIR),
        "water_years": predictor_table["water_year"].astype(int).tolist(),
        "target": "Observed April 1 Sierra SWE anomaly from exact full37 patch LOYO outputs, WY1985-WY2021",
        "predictor_sets": {model_name: columns for model_name, columns in MODEL_SPECS},
        "regression_method": "strict nested LOYO ridge regression",
        "alpha_grid": ALPHA_GRID.tolist(),
        "standardization": {
            "predictors": "train-fold-only mean/std",
            "target": "train-fold-only mean/std",
            "leakage": "no held-out-year leakage",
        },
        "intercept_handling": "intercept not penalized; ridge fit performed in standardized space and converted back to raw coefficients",
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": pd.DataFrame(period_rows).to_dict(orient="records"),
        "selected_alpha_summary": alpha_summaries,
        "coefficient_stability_summary": coef_summaries,
        "comparison_to_previous_exact_Z1_Z2_OLS_reference": base_ref,
        "short_answer": short_answer,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    print("Output directory: {}".format(OUTPUT_DIR))
    print("Predictor sets:")
    for model_name, predictor_columns in MODEL_SPECS:
        print("- {}: {} predictors".format(model_name, len(predictor_columns)))
    print("Metrics table:")
    print(metrics_df.to_string(index=False))
    print("Best model by R2: {}".format(str(best_r2["model_name"])))
    print("Best model by RMSE: {}".format(str(best_rmse["model_name"])))
    print("Best model by sign accuracy: {}".format(str(best_sign["model_name"])))
    print("Selected alpha summary:")
    print(json.dumps(alpha_summaries, indent=2))
    print("Short answer:")
    print(short_answer)


if __name__ == "__main__":
    main()
