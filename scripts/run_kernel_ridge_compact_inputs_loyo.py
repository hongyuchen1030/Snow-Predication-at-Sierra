#!/usr/bin/env python3
"""
Strict LOYO kernel-ridge baselines on compact observed SST inputs for scalar SWE.
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
from sklearn.kernel_ridge import KernelRidge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "kernel_ridge_compact_inputs_loyo"
PREDICTOR_TABLE_CSV = OUTPUT_DIR / "kernel_ridge_compact_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "kernel_ridge_compact_predictions.csv"
HYPERPARAMETERS_CSV = OUTPUT_DIR / "kernel_ridge_compact_selected_hyperparameters.csv"
METRICS_CSV = OUTPUT_DIR / "kernel_ridge_compact_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "kernel_ridge_compact_period_metrics.csv"
README_MD = OUTPUT_DIR / "README.md"
SUMMARY_JSON = OUTPUT_DIR / "kernel_ridge_compact_summary.json"
TIMESERIES_PNG = OUTPUT_DIR / "kernel_ridge_compact_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "kernel_ridge_compact_scatter.png"

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
PACIFIC_PC_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6" / "cobe2_pacific_sierra_t2m_level2_pc1to6.nc",
    Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_pacific_sierra_t2m_level2_pc1to6/"
        "cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
    ),
]

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
Z1_NAME = "Z1_M1_Jan_lat_-9.5_lon_133.5"
Z2_NAME = "Z2_M2_Oct_lat_0.5_lon_136.5"
Z1_OUTPUT = "Z1"
Z2_OUTPUT = "Z2"
A_COLUMNS = ["AMV_PC4_Sep", "AMV_PC5_Feb", "AMV_PC2_Feb", "AMV_PC4_Nov", "AMV_PC5_Mar"]
PACIFIC_COLUMNS = ["Pacific_PC{}_{}".format(pc, month) for pc in range(1, 7) for month in MONTHS]
NINO34_COLUMNS = ["Nino34_{}".format(month) for month in MONTHS]
AMV_COLUMNS = ["AMV_PC{}_{}".format(pc, month) for pc in range(1, 7) for month in MONTHS]
GROUP_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]
RIDGE_ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
KERNEL_ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0], dtype=np.float64)
RBF_GAMMA_GRID = np.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0], dtype=np.float64)
POLY_DEGREE_GRID = [2, 3]
POLY_COEF0_GRID = np.asarray([0.0, 1.0], dtype=np.float64)

FEATURE_SETS = {
    "AMV_AMO_PC1to6": list(AMV_COLUMNS),
    "Z1_Z2_AMV_AMO_K5": [Z1_OUTPUT, Z2_OUTPUT] + list(A_COLUMNS),
    "PacificPC1to6_Nino34_AMV_AMO_PC1to6": list(PACIFIC_COLUMNS) + list(NINO34_COLUMNS) + list(AMV_COLUMNS),
}
DISPLAY_NAMES = {
    "train_mean": "Train mean",
    "ridge__AMV_AMO_PC1to6": "Ridge AMV/AMO PC1-6",
    "kernel_ridge__AMV_AMO_PC1to6": "Kernel ridge AMV/AMO PC1-6",
    "ridge__Z1_Z2_AMV_AMO_K5": "Ridge Z1/Z2 + AMV/AMO K5",
    "kernel_ridge__Z1_Z2_AMV_AMO_K5": "Kernel ridge Z1/Z2 + AMV/AMO K5",
    "ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "Ridge Pacific+Nino34+AMV/AMO",
    "kernel_ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "Kernel ridge Pacific+Nino34+AMV/AMO",
}
PLOT_ORDER = [
    "train_mean",
    "ridge__AMV_AMO_PC1to6",
    "kernel_ridge__AMV_AMO_PC1to6",
    "ridge__Z1_Z2_AMV_AMO_K5",
    "kernel_ridge__Z1_Z2_AMV_AMO_K5",
    "ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6",
    "kernel_ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6",
]


def ensure_runtime_on_compute_node():
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def standardize_target_train_only(y_train_raw):
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Train-fold target standard deviation is non-positive.")
    y_train_std = (y_train_raw - y_mean) / y_std
    return y_train_std, y_mean, y_std


def fit_ridge_standardized(x_train_std, y_train_std, alpha):
    xtx = x_train_std.T @ x_train_std
    ridge = xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64)
    rhs = x_train_std.T @ y_train_std
    beta_std = np.linalg.solve(ridge, rhs)
    return np.asarray(beta_std, dtype=np.float64)


def predict_ridge_raw(x_train_raw, y_train_raw, x_test_raw, alpha):
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
    pred_std = float(x_test_std @ beta_std)
    pred_raw = y_mean + y_std * pred_std
    beta_raw = y_std * beta_std / x_std
    intercept_raw = float(y_mean - np.sum(beta_raw * x_mean))
    return pred_raw, intercept_raw, beta_raw


def kernel_candidate_rows(feature_count):
    gamma_scale = 1.0 / float(max(feature_count, 1))
    rows = []
    for alpha in KERNEL_ALPHA_GRID.tolist():
        for gamma in RBF_GAMMA_GRID.tolist():
            rows.append(
                {
                    "kernel_family": "rbf",
                    "alpha": float(alpha),
                    "gamma": float(gamma * gamma_scale),
                    "degree": np.nan,
                    "coef0": np.nan,
                    "kernel_kwargs": {"kernel": "rbf", "alpha": float(alpha), "gamma": float(gamma * gamma_scale)},
                }
            )
        for degree in POLY_DEGREE_GRID:
            for coef0 in POLY_COEF0_GRID.tolist():
                rows.append(
                    {
                        "kernel_family": "poly",
                        "alpha": float(alpha),
                        "gamma": float(gamma_scale),
                        "degree": int(degree),
                        "coef0": float(coef0),
                        "kernel_kwargs": {
                            "kernel": "poly",
                            "alpha": float(alpha),
                            "gamma": float(gamma_scale),
                            "degree": int(degree),
                            "coef0": float(coef0),
                        },
                    }
                )
    return rows


def inner_loyo_best_ridge_alpha(x_train_raw, y_train_raw):
    best_alpha = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for alpha in RIDGE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            pred_raw, _, _ = predict_ridge_raw(
                x_train_raw[inner_mask],
                y_train_raw[inner_mask],
                x_train_raw[~inner_mask][0],
                float(alpha),
            )
            preds[inner_idx] = pred_raw
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = float(mse)
    return best_alpha, best_mse


def predict_kernel_ridge_raw(x_train_raw, y_train_raw, x_test_raw, candidate):
    x_train_std, x_test_std, _, _ = standardize_train_only(x_train_raw, x_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    model = KernelRidge(**candidate["kernel_kwargs"])
    model.fit(x_train_std, y_train_std)
    pred_std = float(model.predict(x_test_std[None, :])[0])
    pred_raw = float(y_mean + y_std * pred_std)
    return pred_raw, model


def inner_loyo_best_kernel(x_train_raw, y_train_raw):
    best_candidate = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for candidate in kernel_candidate_rows(x_train_raw.shape[1]):
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            pred_raw, _ = predict_kernel_ridge_raw(
                x_train_raw[inner_mask],
                y_train_raw[inner_mask],
                x_train_raw[~inner_mask][0],
                candidate,
            )
            preds[inner_idx] = pred_raw
        mse = float(np.mean((preds - y_train_raw) ** 2))
        alpha = float(candidate["alpha"])
        kernel_rank = 0 if candidate["kernel_family"] == "rbf" else 1
        tie_key = (kernel_rank, alpha, float(candidate["gamma"]), int(candidate["degree"]) if np.isfinite(candidate["degree"]) else 999, float(candidate["coef0"]) if np.isfinite(candidate["coef0"]) else 999.0)
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and tie_key < best_candidate["tie_key"]):
            best_candidate = dict(candidate)
            best_candidate["tie_key"] = tie_key
            best_candidate["inner_cv_mse"] = mse
            best_mse = mse
    return best_candidate


def run_all_models(table):
    water_years = table["water_year"].to_numpy(dtype=np.int32)
    obs = table["obs_swe"].to_numpy(dtype=np.float64)
    prediction_rows = []
    hyper_rows = []

    baseline_preds = np.full(obs.shape, np.nan, dtype=np.float64)
    for fold_index, heldout_wy in enumerate(water_years.tolist()):
        test_mask = water_years == heldout_wy
        train_mask = ~test_mask
        y_train_raw = obs[train_mask]
        y_test_raw = float(obs[test_mask][0])
        pred_raw = float(np.mean(y_train_raw))
        baseline_preds[fold_index] = pred_raw
        error = pred_raw - y_test_raw
        prediction_rows.append(
            {
                "model_name": "train_mean",
                "heldout_wy": int(heldout_wy),
                "obs_swe": y_test_raw,
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error,
                "residual_obs_minus_pred": -error,
                "abs_error": float(abs(error)),
                "sign_correct": float(np.sign(y_test_raw) == np.sign(pred_raw)) if y_test_raw != 0.0 and pred_raw != 0.0 else np.nan,
                "feature_set": "none",
                "model_family": "train_mean",
                "num_predictors": 0,
                "selected_alpha": np.nan,
                "selected_kernel": "mean",
                "selected_gamma": np.nan,
                "selected_degree": np.nan,
                "selected_coef0": np.nan,
            }
        )

    for feature_set_name, predictor_columns in FEATURE_SETS.items():
        x_all = table[predictor_columns].to_numpy(dtype=np.float64)
        for fold_index, heldout_wy in enumerate(water_years.tolist()):
            test_mask = water_years == heldout_wy
            train_mask = ~test_mask
            x_train_raw = x_all[train_mask]
            x_test_raw = x_all[test_mask][0]
            y_train_raw = obs[train_mask]
            y_test_raw = float(obs[test_mask][0])

            ridge_alpha, ridge_mse = inner_loyo_best_ridge_alpha(x_train_raw, y_train_raw)
            ridge_pred, ridge_intercept, ridge_beta = predict_ridge_raw(x_train_raw, y_train_raw, x_test_raw, ridge_alpha)
            ridge_error = ridge_pred - y_test_raw
            ridge_model_name = "ridge__{}".format(feature_set_name)
            prediction_rows.append(
                {
                    "model_name": ridge_model_name,
                    "heldout_wy": int(heldout_wy),
                    "obs_swe": y_test_raw,
                    "pred_swe": ridge_pred,
                    "error_pred_minus_obs": ridge_error,
                    "residual_obs_minus_pred": -ridge_error,
                    "abs_error": float(abs(ridge_error)),
                    "sign_correct": float(np.sign(y_test_raw) == np.sign(ridge_pred)) if y_test_raw != 0.0 and ridge_pred != 0.0 else np.nan,
                    "feature_set": feature_set_name,
                    "model_family": "ridge",
                    "num_predictors": int(len(predictor_columns)),
                    "selected_alpha": float(ridge_alpha),
                    "selected_kernel": "linear",
                    "selected_gamma": np.nan,
                    "selected_degree": np.nan,
                    "selected_coef0": np.nan,
                }
            )
            hyper_rows.append(
                {
                    "model_name": ridge_model_name,
                    "heldout_wy": int(heldout_wy),
                    "feature_set": feature_set_name,
                    "model_family": "ridge",
                    "selected_alpha": float(ridge_alpha),
                    "selected_kernel": "linear",
                    "selected_gamma": np.nan,
                    "selected_degree": np.nan,
                    "selected_coef0": np.nan,
                    "inner_cv_mse": float(ridge_mse),
                    "num_predictors": int(len(predictor_columns)),
                    "intercept": float(ridge_intercept),
                    "beta_l2_norm": float(np.linalg.norm(ridge_beta)),
                }
            )
            print(
                "LOYO heldout_WY={} model={} alpha={} obs={:.6f} pred={:.6f}".format(
                    int(heldout_wy), ridge_model_name, "{:g}".format(float(ridge_alpha)), y_test_raw, ridge_pred
                ),
                flush=True,
            )

            kernel_candidate = inner_loyo_best_kernel(x_train_raw, y_train_raw)
            kernel_pred, kernel_model = predict_kernel_ridge_raw(x_train_raw, y_train_raw, x_test_raw, kernel_candidate)
            kernel_error = kernel_pred - y_test_raw
            kernel_model_name = "kernel_ridge__{}".format(feature_set_name)
            prediction_rows.append(
                {
                    "model_name": kernel_model_name,
                    "heldout_wy": int(heldout_wy),
                    "obs_swe": y_test_raw,
                    "pred_swe": kernel_pred,
                    "error_pred_minus_obs": kernel_error,
                    "residual_obs_minus_pred": -kernel_error,
                    "abs_error": float(abs(kernel_error)),
                    "sign_correct": float(np.sign(y_test_raw) == np.sign(kernel_pred)) if y_test_raw != 0.0 and kernel_pred != 0.0 else np.nan,
                    "feature_set": feature_set_name,
                    "model_family": "kernel_ridge",
                    "num_predictors": int(len(predictor_columns)),
                    "selected_alpha": float(kernel_candidate["alpha"]),
                    "selected_kernel": kernel_candidate["kernel_family"],
                    "selected_gamma": float(kernel_candidate["gamma"]),
                    "selected_degree": kernel_candidate["degree"],
                    "selected_coef0": kernel_candidate["coef0"],
                }
            )
            hyper_rows.append(
                {
                    "model_name": kernel_model_name,
                    "heldout_wy": int(heldout_wy),
                    "feature_set": feature_set_name,
                    "model_family": "kernel_ridge",
                    "selected_alpha": float(kernel_candidate["alpha"]),
                    "selected_kernel": kernel_candidate["kernel_family"],
                    "selected_gamma": float(kernel_candidate["gamma"]),
                    "selected_degree": kernel_candidate["degree"],
                    "selected_coef0": kernel_candidate["coef0"],
                    "inner_cv_mse": float(kernel_candidate["inner_cv_mse"]),
                    "num_predictors": int(len(predictor_columns)),
                    "dual_coef_l2_norm": float(np.linalg.norm(np.asarray(kernel_model.dual_coef_, dtype=np.float64))),
                }
            )
            print(
                "LOYO heldout_WY={} model={} kernel={} alpha={} gamma={:.6g} degree={} coef0={} obs={:.6f} pred={:.6f}".format(
                    int(heldout_wy),
                    kernel_model_name,
                    kernel_candidate["kernel_family"],
                    "{:g}".format(float(kernel_candidate["alpha"])),
                    float(kernel_candidate["gamma"]),
                    "nan" if not np.isfinite(kernel_candidate["degree"]) else int(kernel_candidate["degree"]),
                    "nan" if not np.isfinite(kernel_candidate["coef0"]) else "{:.6g}".format(float(kernel_candidate["coef0"])),
                    y_test_raw,
                    kernel_pred,
                ),
                flush=True,
            )

    pred_df = pd.DataFrame(prediction_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    hyper_df = pd.DataFrame(hyper_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    return pred_df, hyper_df


def compute_metrics(pred_df):
    rows = []
    period_rows = []
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        metrics = compute_metric_bundle(
            sub["obs_swe"].to_numpy(dtype=float),
            sub["pred_swe"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "feature_set": str(sub["feature_set"].iloc[0]),
                "model_family": str(sub["model_family"].iloc[0]),
                "num_predictors": int(sub["num_predictors"].iloc[0]),
                "r": metrics["r"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
            }
        )
        for period_row in compute_period_metrics(
            sub["heldout_wy"].to_numpy(dtype=np.int32),
            sub["obs_swe"].to_numpy(dtype=float),
            sub["pred_swe"].to_numpy(dtype=float),
        ):
            period_rows.append({"model_name": model_name, "display_name": DISPLAY_NAMES[model_name], **period_row})
    return pd.DataFrame(rows), pd.DataFrame(period_rows)


def plot_timeseries(pred_df):
    obs = pred_df[pred_df["model_name"] == "train_mean"].sort_values("heldout_wy")[["heldout_wy", "obs_swe"]]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(obs["heldout_wy"], obs["obs_swe"], color="black", linewidth=2.4, label="Observed SWE anomaly")
    colors = {
        "train_mean": "#7f7f7f",
        "ridge__AMV_AMO_PC1to6": "#1f77b4",
        "kernel_ridge__AMV_AMO_PC1to6": "#6baed6",
        "ridge__Z1_Z2_AMV_AMO_K5": "#d62728",
        "kernel_ridge__Z1_Z2_AMV_AMO_K5": "#ff9896",
        "ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "#2ca02c",
        "kernel_ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "#98df8a",
    }
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["pred_swe"],
            linewidth=1.6,
            color=colors[model_name],
            label=DISPLAY_NAMES[model_name],
        )
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO kernel-ridge baselines on compact observed SST inputs")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(TIMESERIES_PNG, dpi=180)
    plt.close(fig)


def plot_scatter(pred_df, metrics_df):
    models = [name for name in PLOT_ORDER if name != "train_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    obs = pred_df[pred_df["model_name"] == "train_mean"].sort_values("heldout_wy")["obs_swe"].to_numpy(dtype=float)
    limits = [float(np.min(obs)), float(np.max(obs))]
    padding = 0.05 * (limits[1] - limits[0])
    xmin = limits[0] - padding
    xmax = limits[1] + padding
    for ax, model_name in zip(axes.flat, models):
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        pred = sub["pred_swe"].to_numpy(dtype=float)
        metric_row = metrics_df.loc[metrics_df["model_name"] == model_name].iloc[0]
        ax.scatter(obs, pred, color="#1f77b4", s=36, alpha=0.85)
        ax.plot([xmin, xmax], [xmin, xmax], color="black", linestyle="--", linewidth=1.1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel("Observed SWE anomaly (m)")
        ax.set_ylabel("Predicted SWE anomaly (m)")
        ax.set_title(
            "{}\nR2={:.3f} RMSE={:.3f} sign={:.3f}".format(
                DISPLAY_NAMES[model_name],
                float(metric_row["R2"]),
                float(metric_row["RMSE"]),
                float(metric_row["sign_accuracy"]),
            ),
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)
    fig.savefig(SCATTER_PNG, dpi=180)
    plt.close(fig)


def format_metric(value):
    if not np.isfinite(value):
        return "nan"
    return "{:.6f}".format(float(value))


def write_readme(metrics_df, hyper_df, pacific_source_path):
    best_row = metrics_df.sort_values(["R2", "r"], ascending=[False, False]).iloc[0]
    kernel_rows = metrics_df[metrics_df["model_family"] == "kernel_ridge"].copy()
    ridge_rows = metrics_df[metrics_df["model_family"] == "ridge"].copy()
    lines = [
        "# Kernel Ridge Compact-Input LOYO",
        "",
        "This run tested compact observed-data SST predictor blocks under strict outer LOYO with inner-LOYO hyperparameter selection.",
        "",
        "## Inputs tested",
        "",
        "- `AMV_AMO_PC1to6`: 42 Sep-Mar AMV/AMO PC columns.",
        "- `Z1_Z2_AMV_AMO_K5`: oracle `Z1,Z2` plus the selected 5 AMV/AMO columns.",
        "- `PacificPC1to6_Nino34_AMV_AMO_PC1to6`: Pacific PCs, Nino3.4, and AMV/AMO PCs.",
        "",
        "## Headline result",
        "",
        "- Best overall model by LOYO R2: `{}` with `R2={}`, `RMSE={}`, `Pearson_r={}`, `sign_accuracy={}`.".format(
            str(best_row["display_name"]),
            format_metric(best_row["R2"]),
            format_metric(best_row["RMSE"]),
            format_metric(best_row["r"]),
            format_metric(best_row["sign_accuracy"]),
        ),
        "",
        "## Kernel vs linear comparison",
        "",
    ]
    for feature_set_name in FEATURE_SETS:
        ridge_row = ridge_rows.loc[ridge_rows["feature_set"] == feature_set_name].iloc[0]
        kernel_row = kernel_rows.loc[kernel_rows["feature_set"] == feature_set_name].iloc[0]
        comparison = "better" if float(kernel_row["RMSE"]) < float(ridge_row["RMSE"]) else "worse"
        lines.append(
            "- `{}`: kernel ridge is {} than linear ridge by RMSE (`{}` vs `{}`) and has R2 (`{}` vs `{}`).".format(
                feature_set_name,
                comparison,
                format_metric(kernel_row["RMSE"]),
                format_metric(ridge_row["RMSE"]),
                format_metric(kernel_row["R2"]),
                format_metric(ridge_row["R2"]),
            )
        )
    lines.extend(
        [
            "",
            "## Metrics table",
            "",
            metrics_df.to_csv(index=False),
            "",
            "## Source files",
            "",
            "- `PATCH_PREDICTORS_CSV`: `{}`".format(PATCH_PREDICTORS_CSV),
            "- `BASE_PREDICTIONS_CSV`: `{}`".format(BASE_PREDICTIONS_CSV),
            "- `NINO34_CSV`: `{}`".format(NINO34_CSV),
            "- `AMV_CSV`: `{}`".format(AMV_CSV),
            "- `PACIFIC_PC source`: `{}`".format(pacific_source_path),
            "",
            "## Artifacts",
            "",
            "- Metrics: `{}`".format(METRICS_CSV),
            "- Period metrics: `{}`".format(PERIOD_METRICS_CSV),
            "- Predictions: `{}`".format(PREDICTIONS_CSV),
            "- Hyperparameters: `{}`".format(HYPERPARAMETERS_CSV),
            "- Timeseries: `{}`".format(TIMESERIES_PNG),
            "- Scatter: `{}`".format(SCATTER_PNG),
        ]
    )
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    predictor_table, pacific_source_path = build_predictor_table()
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    pred_df, hyper_df = run_all_models(predictor_table)
    metrics_df, period_df = compute_metrics(pred_df)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMETERS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)

    plot_timeseries(pred_df)
    plot_scatter(pred_df, metrics_df)
    write_readme(metrics_df, hyper_df, pacific_source_path)

    summary = {
        "predictor_table_csv": str(PREDICTOR_TABLE_CSV),
        "predictions_csv": str(PREDICTIONS_CSV),
        "hyperparameters_csv": str(HYPERPARAMETERS_CSV),
        "metrics_csv": str(METRICS_CSV),
        "period_metrics_csv": str(PERIOD_METRICS_CSV),
        "timeseries_png": str(TIMESERIES_PNG),
        "scatter_png": str(SCATTER_PNG),
        "readme_md": str(README_MD),
        "best_model_by_r2": metrics_df.sort_values(["R2", "r"], ascending=[False, False]).iloc[0].to_dict(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Predictor table: {}".format(PREDICTOR_TABLE_CSV), flush=True)
    print("Predictions: {}".format(PREDICTIONS_CSV), flush=True)
    print("Metrics: {}".format(METRICS_CSV), flush=True)
    print("README: {}".format(README_MD), flush=True)
    print("Best model by R2: {}".format(summary["best_model_by_r2"]["model_name"]), flush=True)


if __name__ == "__main__":
    main()
