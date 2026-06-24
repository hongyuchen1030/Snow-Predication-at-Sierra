#!/usr/bin/env python3
"""
Assemble the final SST-only strict-LOYO ridge closure table.

This script reuses existing strict-LOYO outputs where available and only runs
the missing full-block models needed for the final closure comparison.
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
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "final_sst_only_linear_closure_table"
)
PREDICTOR_TABLE_CSV = OUTPUT_DIR / "final_sst_only_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "final_sst_only_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "final_sst_only_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "final_sst_only_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "final_sst_only_selected_alpha_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "final_sst_only_beta_by_fold.csv"
CLOSURE_TABLE_TEX = OUTPUT_DIR / "final_sst_only_closure_table.tex"
CLOSURE_TABLE_CSV = OUTPUT_DIR / "final_sst_only_closure_table.csv"
SUMMARY_JSON = OUTPUT_DIR / "final_sst_only_closure_summary.json"
METRICS_PNG = OUTPUT_DIR / "final_sst_only_metrics_comparison.png"
TIMESERIES_PNG = OUTPUT_DIR / "final_sst_only_all_models_timeseries.png"

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
PACIFIC_PC_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6" / "cobe2_pacific_sierra_t2m_level2_pc1to6.nc",
    Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6/"
        "cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
    ),
]
NINO34_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "nino34"
    / "nino34_monthly_wy1985_2021_sep_mar.csv"
)
AMV_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)
DOCS_TEX = PROJECT_ROOT / "docs" / "Current_Status.tex"

PACIFIC_NINO_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_plus_PacificPC_Nino34_loyo"
)
PACIFIC_NINO_PREDICTIONS_CSV = PACIFIC_NINO_DIR / "z1_z2_pacificpc_nino34_loyo_predictions.csv"
PACIFIC_NINO_METRICS_CSV = PACIFIC_NINO_DIR / "z1_z2_pacificpc_nino34_loyo_metrics.csv"
PACIFIC_NINO_PERIOD_CSV = PACIFIC_NINO_DIR / "z1_z2_pacificpc_nino34_loyo_period_metrics.csv"
PACIFIC_NINO_ALPHA_CSV = PACIFIC_NINO_DIR / "z1_z2_pacificpc_nino34_selected_alpha_by_fold.csv"
PACIFIC_NINO_BETA_CSV = PACIFIC_NINO_DIR / "z1_z2_pacificpc_nino34_beta_by_fold.csv"

AMV_PC2TO4_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_plus_AMV_PC2to4_loyo"
)
AMV_PC2TO4_SUMMARY_JSON = AMV_PC2TO4_DIR / "z1_z2_amv_pc2to4_summary.json"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
Z1_NAME = "Z1_M1_Jan_lat_-9.5_lon_133.5"
Z2_NAME = "Z2_M2_Oct_lat_0.5_lon_136.5"
Z1_OUTPUT = "Z1"
Z2_OUTPUT = "Z2"

PACIFIC_COLUMNS = ["Pacific_PC{}_{}".format(pc, month) for pc in range(1, 7) for month in MONTHS]
NINO34_COLUMNS = ["Nino34_{}".format(month) for month in MONTHS]
AMV_COLUMNS = ["AMV_PC{}_{}".format(pc, month) for pc in range(1, 7) for month in MONTHS]

MODEL_ORDER = [
    "Z1_Z2",
    "PacificPC1to6_only",
    "Nino34_only",
    "AMV_AMO_PC1to6_only",
    "PacificPC1to6_Nino34_AMV_AMO_PC1to6",
    "Z1_Z2_PacificPC1to6",
    "Z1_Z2_Nino34",
    "Z1_Z2_AMV_AMO_PC1to6",
]
DISPLAY_LABELS = {
    "Z1_Z2": r"\(Z_1+Z_2\)",
    "PacificPC1to6_only": "Pacific PC1--PC6",
    "Nino34_only": r"Ni\~no 3.4",
    "AMV_AMO_PC1to6_only": "AMV/AMO PC1--PC6",
    "PacificPC1to6_Nino34_AMV_AMO_PC1to6": r"Pacific PC1--PC6 + Ni\~no 3.4 + AMV/AMO PC1--PC6",
    "Z1_Z2_PacificPC1to6": r"\(Z_1+Z_2\) + Pacific PC1--PC6",
    "Z1_Z2_Nino34": r"\(Z_1+Z_2\) + Ni\~no 3.4",
    "Z1_Z2_AMV_AMO_PC1to6": r"\(Z_1+Z_2\) + AMV/AMO PC1--PC6",
}
MODEL_COLUMNS = {
    "Z1_Z2": [Z1_OUTPUT, Z2_OUTPUT],
    "PacificPC1to6_only": list(PACIFIC_COLUMNS),
    "Nino34_only": list(NINO34_COLUMNS),
    "AMV_AMO_PC1to6_only": list(AMV_COLUMNS),
    "PacificPC1to6_Nino34_AMV_AMO_PC1to6": list(PACIFIC_COLUMNS) + list(NINO34_COLUMNS) + list(AMV_COLUMNS),
    "Z1_Z2_PacificPC1to6": [Z1_OUTPUT, Z2_OUTPUT] + list(PACIFIC_COLUMNS),
    "Z1_Z2_Nino34": [Z1_OUTPUT, Z2_OUTPUT] + list(NINO34_COLUMNS),
    "Z1_Z2_AMV_AMO_PC1to6": [Z1_OUTPUT, Z2_OUTPUT] + list(AMV_COLUMNS),
}
REUSE_SOURCE = {
    "Z1_Z2": "exact_Z1_Z2_plus_PacificPC_Nino34_loyo",
    "PacificPC1to6_only": "exact_Z1_Z2_plus_PacificPC_Nino34_loyo",
    "Nino34_only": "exact_Z1_Z2_plus_PacificPC_Nino34_loyo",
    "Z1_Z2_PacificPC1to6": "exact_Z1_Z2_plus_PacificPC_Nino34_loyo",
    "Z1_Z2_Nino34": "exact_Z1_Z2_plus_PacificPC_Nino34_loyo",
}
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


def choose_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No valid source found in candidates: {}".format([str(path) for path in candidates]))


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

    nino = pd.read_csv(NINO34_CSV)
    nino["water_year"] = nino["water_year"].astype(int)
    nino = nino[["water_year"] + NINO34_COLUMNS].sort_values("water_year").reset_index(drop=True)

    amv = pd.read_csv(AMV_CSV)
    amv["water_year"] = amv["water_year"].astype(int)
    amv = amv[["water_year"] + AMV_COLUMNS].sort_values("water_year").reset_index(drop=True)

    return predictors, base_predictions, nino, amv


def build_wy_aligned_pacific_table(water_years):
    pacific_path = choose_existing_path(PACIFIC_PC_CANDIDATES)
    ds = xr.open_dataset(pacific_path)
    if "pacific_cobe2_pc" not in ds:
        raise ValueError("Missing pacific_cobe2_pc in {}".format(pacific_path))
    pc = ds["pacific_cobe2_pc"].load()
    times = pd.to_datetime(ds["time"].to_numpy())
    mode_values = ds["mode"].to_numpy()
    data = pd.DataFrame(pc.to_numpy(), index=times, columns=[int(mode) for mode in mode_values])

    rows = []
    for water_year in water_years:
        row = {"water_year": int(water_year)}
        for month in MONTHS:
            calendar_year = water_year - 1 if month in {"Sep", "Oct", "Nov", "Dec"} else water_year
            timestamp = pd.Timestamp(calendar_year, MONTH_TO_NUMBER[month], 1)
            if timestamp not in data.index:
                raise KeyError("Missing {} in {}".format(timestamp.strftime("%Y-%m"), pacific_path))
            values = data.loc[timestamp]
            for pc_index in range(1, 7):
                row["Pacific_PC{}_{}".format(pc_index, month)] = float(values.loc[pc_index])
        rows.append(row)
    ds.close()
    return pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True), pacific_path


def build_predictor_table():
    predictors, base_predictions, nino, amv = load_base_tables()
    expected_years = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
    pacific, pacific_path = build_wy_aligned_pacific_table(expected_years)

    merged = predictors.merge(base_predictions, on="water_year", how="inner")
    merged = merged.merge(pacific, on="water_year", how="inner")
    merged = merged.merge(nino, on="water_year", how="inner")
    merged = merged.merge(amv, on="water_year", how="inner")
    merged = merged.sort_values("water_year").reset_index(drop=True)
    merged = merged.loc[(merged["water_year"] >= WATER_YEAR_START) & (merged["water_year"] <= WATER_YEAR_END)].copy()
    if merged["water_year"].tolist() != expected_years:
        raise ValueError("Predictor table years do not match WY1985-WY2021.")

    ordered_columns = ["water_year", "obs_swe", Z1_OUTPUT, Z2_OUTPUT] + PACIFIC_COLUMNS + NINO34_COLUMNS + AMV_COLUMNS
    return merged[ordered_columns].copy(), pacific_path


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


def run_nested_loyo_for_model(data_df, model_name, predictor_columns, all_beta_columns):
    water_years = data_df["water_year"].to_numpy(dtype=np.int32)
    obs = data_df["obs_swe"].to_numpy(dtype=np.float64)
    x_all = data_df[predictor_columns].to_numpy(dtype=np.float64)

    preds = np.full(obs.shape, np.nan, dtype=np.float64)
    selected_alphas = np.full(obs.shape, np.nan, dtype=np.float64)

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
        beta_row = {"model_name": model_name, "heldout_wy": int(heldout_wy), "intercept": intercept}
        for name in all_beta_columns:
            beta_row["beta_" + name] = float("nan")
        for col_idx, name in enumerate(predictor_columns):
            beta_row["beta_" + name] = float(beta_raw[col_idx])
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

    return {
        "prediction_rows": prediction_rows,
        "alpha_rows": alpha_rows,
        "beta_rows": beta_rows,
        "metric_row": metric_row,
        "period_rows": period_rows,
    }


def read_reuse_table(path):
    if not path.exists():
        raise FileNotFoundError("Missing reusable result file: {}".format(path))
    return pd.read_csv(path)


def load_reused_model_rows(model_name, all_beta_columns):
    predictions = read_reuse_table(PACIFIC_NINO_PREDICTIONS_CSV)
    metrics = read_reuse_table(PACIFIC_NINO_METRICS_CSV)
    periods = read_reuse_table(PACIFIC_NINO_PERIOD_CSV)
    alphas = read_reuse_table(PACIFIC_NINO_ALPHA_CSV)
    betas = read_reuse_table(PACIFIC_NINO_BETA_CSV)

    prediction_rows = predictions.loc[predictions["model_name"] == model_name].copy()
    metric_rows = metrics.loc[metrics["model_name"] == model_name].copy()
    period_rows = periods.loc[periods["model_name"] == model_name].copy()
    alpha_rows = alphas.loc[alphas["model_name"] == model_name].copy()
    beta_rows = betas.loc[betas["model_name"] == model_name].copy()
    if prediction_rows.empty or metric_rows.empty or period_rows.empty or alpha_rows.empty or beta_rows.empty:
        raise ValueError("Reusable outputs for model {} are incomplete.".format(model_name))

    expected_num_predictors = len(MODEL_COLUMNS[model_name])
    prediction_rows["num_predictors"] = expected_num_predictors
    metric_rows["num_predictors"] = expected_num_predictors
    period_rows["num_predictors"] = expected_num_predictors

    for name in all_beta_columns:
        column = "beta_" + name
        if column not in beta_rows.columns:
            beta_rows[column] = np.nan
    keep_columns = ["model_name", "heldout_wy", "intercept"] + ["beta_" + name for name in all_beta_columns]
    beta_rows = beta_rows[keep_columns].copy()

    metric_row = metric_rows.iloc[0][["model_name", "num_predictors", "r", "R2", "RMSE", "MAE", "sign_accuracy"]].to_dict()

    return {
        "prediction_rows": prediction_rows.to_dict("records"),
        "metric_row": metric_row,
        "period_rows": period_rows.to_dict("records"),
        "alpha_rows": alpha_rows.to_dict("records"),
        "beta_rows": beta_rows.to_dict("records"),
    }


def plot_metrics_comparison(metrics_df):
    fields = ["r", "R2", "RMSE", "MAE", "sign_accuracy"]
    titles = {
        "r": "Correlation",
        "R2": r"$R^2$",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "sign_accuracy": "Sign accuracy",
    }
    labels = [DISPLAY_LABELS[name] for name in metrics_df["model_name"].tolist()]
    x = np.arange(len(metrics_df))

    fig, axes = plt.subplots(len(fields), 1, figsize=(14.0, 13.0), constrained_layout=True)
    for ax, field in zip(axes, fields):
        values = metrics_df[field].to_numpy(dtype=float)
        ax.bar(x, values, color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(titles[field])
        ax.grid(True, axis="y", linewidth=0.25, color="0.85")
        if field == "R2":
            ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        if field == "sign_accuracy":
            ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    fig.savefig(METRICS_PNG, dpi=220)
    plt.close(fig)


def plot_all_model_timeseries(predictions_df):
    fig, ax = plt.subplots(figsize=(15.0, 7.0), constrained_layout=True)

    observed = (
        predictions_df.loc[:, ["heldout_wy", "obs_swe"]]
        .drop_duplicates()
        .sort_values("heldout_wy")
        .reset_index(drop=True)
    )
    ax.plot(
        observed["heldout_wy"].to_numpy(dtype=int),
        observed["obs_swe"].to_numpy(dtype=float),
        color="black",
        linewidth=2.4,
        label="Observed SWE anomaly",
        zorder=4,
    )

    color_values = plt.cm.tab10(np.linspace(0.0, 1.0, len(MODEL_ORDER)))
    for color_value, model_name in zip(color_values, MODEL_ORDER):
        subset = (
            predictions_df.loc[predictions_df["model_name"] == model_name, ["heldout_wy", "pred_swe"]]
            .sort_values("heldout_wy")
            .reset_index(drop=True)
        )
        ax.plot(
            subset["heldout_wy"].to_numpy(dtype=int),
            subset["pred_swe"].to_numpy(dtype=float),
            linewidth=1.5,
            alpha=0.9,
            color=color_value,
            label=DISPLAY_LABELS[model_name],
            zorder=2,
        )

    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO predicted SWE anomaly by SST-only linear model")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(TIMESERIES_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_metric(value, decimals):
    return ("{0:." + str(decimals) + "f}").format(float(value))


def write_closure_table(metrics_df):
    table_df = metrics_df.copy()
    table_df["display_label"] = table_df["model_name"].map(DISPLAY_LABELS)
    table_df = table_df[
        ["model_name", "display_label", "num_predictors", "r", "R2", "RMSE", "MAE", "sign_accuracy", "result_source"]
    ].copy()
    table_df.to_csv(CLOSURE_TABLE_CSV, index=False)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Model & Predictors & \(r\) & \(R^2\) & RMSE & MAE & Sign acc. \\",
        r"\hline",
    ]
    for row in table_df.to_dict("records"):
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["display_label"],
                int(row["num_predictors"]),
                format_metric(row["r"], 3),
                format_metric(row["R2"], 3),
                format_metric(row["RMSE"], 5),
                format_metric(row["MAE"], 5),
                format_metric(row["sign_accuracy"], 3),
            )
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Strict LOYO ridge comparison for SST-only linear predictor blocks. The \(Z_1+Z_2\) model uses the exact full-37-year LOD-selected SST columns as fixed predictors. All PC and index predictors use September--March values.}",
            r"\label{tab:sst_only_linear_closure}",
            r"\end{table}",
        ]
    )
    CLOSURE_TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row_to_dict(metrics_df, field):
    row = metrics_df.iloc[metrics_df[field].idxmax()] if field != "RMSE" else metrics_df.iloc[metrics_df[field].idxmin()]
    return {
        "model_name": row["model_name"],
        "display_label": DISPLAY_LABELS[row["model_name"]],
        field: float(row[field]),
    }


def build_short_answer(best_r2, best_rmse, best_sign):
    if best_r2["model_name"] == "Z1_Z2" and best_rmse["model_name"] == "Z1_Z2":
        return (
            "The fixed Z1+Z2 model remains the best SST-only linear model. Full Pacific PC1--PC6, "
            "Nino34, and AMV/AMO PC1--PC6 blocks do not improve strict LOYO ridge prediction, "
            "either alone or when added to Z1+Z2."
        )

    return (
        "The fixed Z1+Z2 model remains the best low-dimensional fixed-LOD reference and the best "
        "model for sign accuracy, but AMV/AMO PC1--PC6 alone is best by r, R2, RMSE, and MAE. "
        "Adding Pacific, Nino34, or AMV/AMO blocks to Z1+Z2 does not improve the fixed Z1+Z2 baseline."
    )


def build_summary(metrics_df, period_df, input_files, reused_models, newly_run_models, pacific_source_path):
    model_metrics = {}
    for row in metrics_df.to_dict("records"):
        model_metrics[row["model_name"]] = {
            "display_label": DISPLAY_LABELS[row["model_name"]],
            "num_predictors": int(row["num_predictors"]),
            "r": float(row["r"]),
            "R2": float(row["R2"]),
            "RMSE": float(row["RMSE"]),
            "MAE": float(row["MAE"]),
            "sign_accuracy": float(row["sign_accuracy"]),
            "result_source": row["result_source"],
        }

    period_metrics = {}
    for model_name, group_df in period_df.groupby("model_name"):
        period_metrics[model_name] = {}
        for row in group_df.to_dict("records"):
            period_metrics[model_name][row["group_name"]] = {
                "n_years": int(row["n_years"]),
                "r": float(row["r"]),
                "R2": float(row["R2"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
                "sign_accuracy": float(row["sign_accuracy"]),
                "mean_error": float(row["mean_error"]),
                "mean_abs_error": float(row["mean_abs_error"]),
            }

    best_r2 = metric_row_to_dict(metrics_df, "R2")
    best_rmse = metric_row_to_dict(metrics_df, "RMSE")
    best_sign = metric_row_to_dict(metrics_df, "sign_accuracy")

    summary = {
        "input_files": input_files + [str(pacific_source_path)],
        "reused_existing_results": reused_models,
        "newly_run_models": newly_run_models,
        "model_metrics": model_metrics,
        "period_metrics": period_metrics,
        "best_model_by_R2": best_r2,
        "best_model_by_RMSE": best_rmse,
        "best_model_by_sign_accuracy": best_sign,
        "latex_table_path": str(CLOSURE_TABLE_TEX),
        "current_status_tex_updated": False,
        "note_previous_amv_pc2to4": str(AMV_PC2TO4_SUMMARY_JSON) if AMV_PC2TO4_SUMMARY_JSON.exists() else None,
        "short_answer": build_short_answer(best_r2, best_rmse, best_sign),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictor_table, pacific_source_path = build_predictor_table()
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    all_beta_columns = []
    for model_name in MODEL_ORDER:
        for column in MODEL_COLUMNS[model_name]:
            if column not in all_beta_columns:
                all_beta_columns.append(column)

    predictions_rows = []
    metrics_rows = []
    period_rows = []
    alpha_rows = []
    beta_rows = []
    reused_models = []
    newly_run_models = []

    for model_name in MODEL_ORDER:
        if model_name in REUSE_SOURCE:
            result = load_reused_model_rows(model_name, all_beta_columns)
            result_source = "reused:{}".format(REUSE_SOURCE[model_name])
            reused_models.append({"model_name": model_name, "source": REUSE_SOURCE[model_name]})
        else:
            result = run_nested_loyo_for_model(predictor_table, model_name, MODEL_COLUMNS[model_name], all_beta_columns)
            result_source = "new_run:final_sst_only_linear_closure_table"
            newly_run_models.append(model_name)

        for row in result["prediction_rows"]:
            row["result_source"] = result_source
            predictions_rows.append(row)
        metric_row = dict(result["metric_row"])
        metric_row["result_source"] = result_source
        metrics_rows.append(metric_row)
        for row in result["period_rows"]:
            row["result_source"] = result_source
            period_rows.append(row)
        for row in result["alpha_rows"]:
            row["result_source"] = result_source
            alpha_rows.append(row)
        for row in result["beta_rows"]:
            row["result_source"] = result_source
            beta_rows.append(row)

    predictions_df = pd.DataFrame(predictions_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["model_name"] = pd.Categorical(metrics_df["model_name"], categories=MODEL_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("model_name").reset_index(drop=True)
    metrics_df["model_name"] = metrics_df["model_name"].astype(str)
    metrics_df["best_by_R2"] = metrics_df["R2"] == metrics_df["R2"].max()
    metrics_df["best_by_RMSE"] = metrics_df["RMSE"] == metrics_df["RMSE"].min()
    metrics_df["best_by_sign_accuracy"] = metrics_df["sign_accuracy"] == metrics_df["sign_accuracy"].max()

    period_df = pd.DataFrame(period_rows)
    period_df["model_name"] = pd.Categorical(period_df["model_name"], categories=MODEL_ORDER, ordered=True)
    period_df = period_df.sort_values(["model_name", "group_name"]).reset_index(drop=True)
    period_df["model_name"] = period_df["model_name"].astype(str)

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df["model_name"] = pd.Categorical(alpha_df["model_name"], categories=MODEL_ORDER, ordered=True)
    alpha_df = alpha_df.sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    alpha_df["model_name"] = alpha_df["model_name"].astype(str)

    beta_df = pd.DataFrame(beta_rows)
    beta_df["model_name"] = pd.Categorical(beta_df["model_name"], categories=MODEL_ORDER, ordered=True)
    beta_df = beta_df.sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    beta_df["model_name"] = beta_df["model_name"].astype(str)

    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)

    write_closure_table(metrics_df)
    plot_metrics_comparison(metrics_df)
    plot_all_model_timeseries(predictions_df)

    input_files = [
        str(PATCH_PREDICTORS_CSV),
        str(BASE_PREDICTIONS_CSV),
        str(NINO34_CSV),
        str(AMV_CSV),
        str(PACIFIC_NINO_PREDICTIONS_CSV),
        str(PACIFIC_NINO_METRICS_CSV),
        str(PACIFIC_NINO_PERIOD_CSV),
        str(PACIFIC_NINO_ALPHA_CSV),
        str(PACIFIC_NINO_BETA_CSV),
    ]
    summary = build_summary(metrics_df, period_df, input_files, reused_models, newly_run_models, pacific_source_path)

    print("Output directory: {}".format(OUTPUT_DIR))
    print("Newly run models: {}".format(", ".join(newly_run_models) if newly_run_models else "none"))
    print(
        "Reused models: {}".format(
            ", ".join(["{} ({})".format(item["model_name"], item["source"]) for item in reused_models])
            if reused_models
            else "none"
        )
    )
    print("Final closure table: {}".format(CLOSURE_TABLE_TEX))
    print("Updated docs/Current_Status.tex: no")
    print("Short answer:")
    print(summary["short_answer"])


if __name__ == "__main__":
    main()
