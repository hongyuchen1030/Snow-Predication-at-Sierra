#!/usr/bin/env python3
"""
Strict LOYO random-feature ridge baselines on compact observed SST inputs.
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
from sklearn.kernel_approximation import RBFSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_kernel_ridge_compact_inputs_loyo import (  # noqa: E402
    DISPLAY_NAMES as BASE_DISPLAY_NAMES,
    FEATURE_SETS,
    GROUP_SPECS,
    PREDICTOR_TABLE_CSV as KERNEL_PREDICTOR_TABLE_CSV,
    RIDGE_ALPHA_GRID,
    build_predictor_table,
    compute_metric_bundle,
    compute_period_metrics,
    ensure_runtime_on_compute_node,
    standardize_target_train_only,
    standardize_train_only,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "random_feature_ridge_compact_inputs_loyo"
PREDICTOR_TABLE_CSV = OUTPUT_DIR / "random_feature_compact_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "random_feature_ridge_compact_predictions.csv"
HYPERPARAMETERS_CSV = OUTPUT_DIR / "random_feature_ridge_compact_selected_hyperparameters.csv"
METRICS_CSV = OUTPUT_DIR / "random_feature_ridge_compact_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "random_feature_ridge_compact_period_metrics.csv"
README_MD = OUTPUT_DIR / "README.md"
SUMMARY_JSON = OUTPUT_DIR / "random_feature_ridge_compact_summary.json"
TIMESERIES_PNG = OUTPUT_DIR / "random_feature_ridge_compact_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "random_feature_ridge_compact_scatter.png"

RF_DIMS = [10, 25, 50, 100]
RF_TYPES = ["rff", "relu"]
RBF_GAMMA_GRID = np.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0], dtype=np.float64)
RELU_SCALE_GRID = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=np.float64)

DISPLAY_NAMES = dict(BASE_DISPLAY_NAMES)
DISPLAY_NAMES.update(
    {
        "random_feature_ridge__AMV_AMO_PC1to6": "RF ridge AMV/AMO PC1-6",
        "random_feature_ridge__Z1_Z2_AMV_AMO_K5": "RF ridge Z1/Z2 + AMV/AMO K5",
        "random_feature_ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "RF ridge Pacific+Nino34+AMV/AMO",
    }
)
PLOT_ORDER = [
    "train_mean",
    "ridge__AMV_AMO_PC1to6",
    "random_feature_ridge__AMV_AMO_PC1to6",
    "ridge__Z1_Z2_AMV_AMO_K5",
    "random_feature_ridge__Z1_Z2_AMV_AMO_K5",
    "ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6",
    "random_feature_ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6",
]


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def standardize_allow_constant(x_train, x_test):
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0, ddof=0)
    safe_std = np.where(x_std > 0.0, x_std, 1.0)
    x_train_std = (x_train - x_mean) / safe_std
    x_test_std = (x_test - x_mean) / safe_std
    return x_train_std, x_test_std, x_mean, safe_std


def fit_ridge_standardized(x_train_std, y_train_std, alpha):
    xtx = np.dot(x_train_std.T, x_train_std)
    ridge = xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64)
    rhs = np.dot(x_train_std.T, y_train_std)
    return np.linalg.solve(ridge, rhs)


def predict_ridge_from_features(z_train, y_train_raw, z_test, alpha):
    z_train_std, z_test_std, z_mean, z_std = standardize_allow_constant(z_train, z_test)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(z_train_std, y_train_std, alpha)
    pred_std = float(np.dot(z_test_std, beta_std))
    pred_raw = float(y_mean + y_std * pred_std)
    beta_raw = y_std * beta_std / z_std
    intercept = float(y_mean - np.sum(beta_raw * z_mean))
    return pred_raw, intercept, beta_raw


def build_relu_features(x_train_std, x_test_std, n_components, scale, random_state=0):
    rng = np.random.RandomState(random_state)
    weight = rng.normal(loc=0.0, scale=float(scale) / np.sqrt(max(x_train_std.shape[1], 1)), size=(x_train_std.shape[1], n_components))
    bias = rng.normal(loc=0.0, scale=1.0, size=(n_components,))
    z_train = np.maximum(0.0, np.dot(x_train_std, weight) + bias[None, :])
    z_test = np.maximum(0.0, np.dot(x_test_std, weight) + bias)
    return z_train, z_test, {"weight": weight, "bias": bias}


def build_rff_features(x_train_std, x_test_std, n_components, gamma, random_state=0):
    sampler = RBFSampler(gamma=float(gamma), n_components=int(n_components), random_state=int(random_state))
    z_train = sampler.fit_transform(x_train_std)
    z_test = sampler.transform(x_test_std[None, :])[0]
    return z_train, z_test, sampler


def transform_features(x_train_std, x_test_std, rf_type, n_components, scale_value):
    if rf_type == "rff":
        return build_rff_features(x_train_std, x_test_std, n_components, scale_value)
    if rf_type == "relu":
        return build_relu_features(x_train_std, x_test_std, n_components, scale_value)
    raise ValueError("Unknown random feature type: {}".format(rf_type))


def candidate_rows(feature_count):
    rows = []
    gamma_scale = 1.0 / float(max(feature_count, 1))
    for rf_type in RF_TYPES:
        for n_components in RF_DIMS:
            grid = RBF_GAMMA_GRID.tolist() if rf_type == "rff" else RELU_SCALE_GRID.tolist()
            scale_name = "gamma" if rf_type == "rff" else "relu_scale"
            for scale_value in grid:
                for alpha in RIDGE_ALPHA_GRID.tolist():
                    rows.append(
                        {
                            "rf_type": rf_type,
                            "n_components": int(n_components),
                            "scale_name": scale_name,
                            "scale_value": float(scale_value * gamma_scale if rf_type == "rff" else scale_value),
                            "alpha": float(alpha),
                        }
                    )
    return rows


def choose_best_ridge_alpha_on_features(z_train, y_train_raw):
    best_alpha = None
    best_mse = None
    n_train = z_train.shape[0]
    for alpha in RIDGE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            pred_raw, _, _ = predict_ridge_from_features(
                z_train[inner_mask],
                y_train_raw[inner_mask],
                z_train[~inner_mask][0],
                float(alpha),
            )
            preds[inner_idx] = pred_raw
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = float(mse)
    return best_alpha, best_mse


def inner_loyo_best_random_feature(x_train_raw, y_train_raw):
    best_candidate = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for candidate in candidate_rows(x_train_raw.shape[1]):
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(x_train_raw[inner_mask], x_train_raw[~inner_mask][0])
            z_train, z_valid, _ = transform_features(
                x_inner_train_std,
                x_inner_valid_std,
                candidate["rf_type"],
                candidate["n_components"],
                candidate["scale_value"],
            )
            pred_raw, _, _ = predict_ridge_from_features(
                z_train,
                y_train_raw[inner_mask],
                z_valid,
                candidate["alpha"],
            )
            preds[inner_idx] = pred_raw
        mse = float(np.mean((preds - y_train_raw) ** 2))
        tie_key = (
            0 if candidate["rf_type"] == "rff" else 1,
            int(candidate["n_components"]),
            float(candidate["scale_value"]),
            float(candidate["alpha"]),
        )
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and tie_key < best_candidate["tie_key"]):
            best_candidate = dict(candidate)
            best_candidate["tie_key"] = tie_key
            best_candidate["inner_cv_mse"] = mse
            best_mse = mse
    return best_candidate


def inner_loyo_best_linear_alpha(x_train_raw, y_train_raw):
    best_alpha = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for alpha in RIDGE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            x_train_std, x_valid_std, x_mean, x_std = standardize_train_only(x_train_raw[inner_mask], x_train_raw[~inner_mask][0])
            y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw[inner_mask])
            beta_std = fit_ridge_standardized(x_train_std, y_train_std, float(alpha))
            pred_std = float(np.dot(x_valid_std, beta_std))
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = float(mse)
    return best_alpha, best_mse


def predict_linear_ridge_raw(x_train_raw, y_train_raw, x_test_raw, alpha):
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(x_train_std, y_train_std, float(alpha))
    pred_std = float(np.dot(x_test_std, beta_std))
    pred_raw = float(y_mean + y_std * pred_std)
    beta_raw = y_std * beta_std / x_std
    intercept = float(y_mean - np.sum(beta_raw * x_mean))
    return pred_raw, intercept, beta_raw


def run_all_models(table):
    water_years = table["water_year"].to_numpy(dtype=np.int32)
    obs = table["obs_swe"].to_numpy(dtype=np.float64)
    prediction_rows = []
    hyper_rows = []

    for fold_index, heldout_wy in enumerate(water_years.tolist()):
        test_mask = water_years == heldout_wy
        train_mask = ~test_mask
        y_train_raw = obs[train_mask]
        y_test_raw = float(obs[test_mask][0])
        pred_raw = float(np.mean(y_train_raw))
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
                "rf_type": "mean",
                "rf_dim": np.nan,
                "scale_name": "none",
                "scale_value": np.nan,
            }
        )

    for feature_set_name, predictor_columns in FEATURE_SETS.items():
        x_all = table[predictor_columns].to_numpy(dtype=np.float64)
        for heldout_wy in water_years.tolist():
            test_mask = water_years == heldout_wy
            train_mask = ~test_mask
            x_train_raw = x_all[train_mask]
            x_test_raw = x_all[test_mask][0]
            y_train_raw = obs[train_mask]
            y_test_raw = float(obs[test_mask][0])

            ridge_alpha, ridge_mse = inner_loyo_best_linear_alpha(x_train_raw, y_train_raw)
            ridge_pred, ridge_intercept, ridge_beta = predict_linear_ridge_raw(x_train_raw, y_train_raw, x_test_raw, ridge_alpha)
            ridge_name = "ridge__{}".format(feature_set_name)
            ridge_error = ridge_pred - y_test_raw
            prediction_rows.append(
                {
                    "model_name": ridge_name,
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
                    "rf_type": "linear",
                    "rf_dim": np.nan,
                    "scale_name": "none",
                    "scale_value": np.nan,
                }
            )
            hyper_rows.append(
                {
                    "model_name": ridge_name,
                    "heldout_wy": int(heldout_wy),
                    "feature_set": feature_set_name,
                    "model_family": "ridge",
                    "selected_alpha": float(ridge_alpha),
                    "rf_type": "linear",
                    "rf_dim": np.nan,
                    "scale_name": "none",
                    "scale_value": np.nan,
                    "inner_cv_mse": float(ridge_mse),
                    "num_predictors": int(len(predictor_columns)),
                    "intercept": float(ridge_intercept),
                    "beta_l2_norm": float(np.linalg.norm(ridge_beta)),
                }
            )
            print(
                "LOYO heldout_WY={} model={} alpha={} obs={:.6f} pred={:.6f}".format(
                    int(heldout_wy), ridge_name, "{:g}".format(float(ridge_alpha)), y_test_raw, ridge_pred
                ),
                flush=True,
            )

            best_candidate = inner_loyo_best_random_feature(x_train_raw, y_train_raw)
            x_train_std, x_test_std, _, _ = standardize_train_only(x_train_raw, x_test_raw)
            z_train, z_test, transform_obj = transform_features(
                x_train_std,
                x_test_std,
                best_candidate["rf_type"],
                best_candidate["n_components"],
                best_candidate["scale_value"],
            )
            rf_pred, rf_intercept, rf_beta = predict_ridge_from_features(z_train, y_train_raw, z_test, best_candidate["alpha"])
            rf_name = "random_feature_ridge__{}".format(feature_set_name)
            rf_error = rf_pred - y_test_raw
            prediction_rows.append(
                {
                    "model_name": rf_name,
                    "heldout_wy": int(heldout_wy),
                    "obs_swe": y_test_raw,
                    "pred_swe": rf_pred,
                    "error_pred_minus_obs": rf_error,
                    "residual_obs_minus_pred": -rf_error,
                    "abs_error": float(abs(rf_error)),
                    "sign_correct": float(np.sign(y_test_raw) == np.sign(rf_pred)) if y_test_raw != 0.0 and rf_pred != 0.0 else np.nan,
                    "feature_set": feature_set_name,
                    "model_family": "random_feature_ridge",
                    "num_predictors": int(len(predictor_columns)),
                    "selected_alpha": float(best_candidate["alpha"]),
                    "rf_type": best_candidate["rf_type"],
                    "rf_dim": int(best_candidate["n_components"]),
                    "scale_name": best_candidate["scale_name"],
                    "scale_value": float(best_candidate["scale_value"]),
                }
            )
            hyper_rows.append(
                {
                    "model_name": rf_name,
                    "heldout_wy": int(heldout_wy),
                    "feature_set": feature_set_name,
                    "model_family": "random_feature_ridge",
                    "selected_alpha": float(best_candidate["alpha"]),
                    "rf_type": best_candidate["rf_type"],
                    "rf_dim": int(best_candidate["n_components"]),
                    "scale_name": best_candidate["scale_name"],
                    "scale_value": float(best_candidate["scale_value"]),
                    "inner_cv_mse": float(best_candidate["inner_cv_mse"]),
                    "num_predictors": int(len(predictor_columns)),
                    "intercept": float(rf_intercept),
                    "beta_l2_norm": float(np.linalg.norm(rf_beta)),
                }
            )
            print(
                "LOYO heldout_WY={} model={} rf_type={} rf_dim={} {}={} alpha={} obs={:.6f} pred={:.6f}".format(
                    int(heldout_wy),
                    rf_name,
                    best_candidate["rf_type"],
                    int(best_candidate["n_components"]),
                    best_candidate["scale_name"],
                    "{:.6g}".format(float(best_candidate["scale_value"])),
                    "{:g}".format(float(best_candidate["alpha"])),
                    y_test_raw,
                    rf_pred,
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
        metrics = compute_metric_bundle(sub["obs_swe"].to_numpy(dtype=float), sub["pred_swe"].to_numpy(dtype=float))
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
        for period_row in compute_period_metrics(sub["heldout_wy"].to_numpy(dtype=np.int32), sub["obs_swe"].to_numpy(dtype=float), sub["pred_swe"].to_numpy(dtype=float)):
            period_rows.append({"model_name": model_name, "display_name": DISPLAY_NAMES[model_name], **period_row})
    return pd.DataFrame(rows), pd.DataFrame(period_rows)


def plot_timeseries(pred_df):
    obs = pred_df[pred_df["model_name"] == "train_mean"].sort_values("heldout_wy")[["heldout_wy", "obs_swe"]]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(obs["heldout_wy"], obs["obs_swe"], color="black", linewidth=2.4, label="Observed SWE anomaly")
    colors = {
        "train_mean": "#7f7f7f",
        "ridge__AMV_AMO_PC1to6": "#1f77b4",
        "random_feature_ridge__AMV_AMO_PC1to6": "#6baed6",
        "ridge__Z1_Z2_AMV_AMO_K5": "#d62728",
        "random_feature_ridge__Z1_Z2_AMV_AMO_K5": "#ff9896",
        "ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "#2ca02c",
        "random_feature_ridge__PacificPC1to6_Nino34_AMV_AMO_PC1to6": "#98df8a",
    }
    for model_name in PLOT_ORDER:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(sub["heldout_wy"], sub["pred_swe"], linewidth=1.6, color=colors[model_name], label=DISPLAY_NAMES[model_name])
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO random-feature ridge baselines on compact observed SST inputs")
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


def write_readme(metrics_df):
    best_row = metrics_df.sort_values(["R2", "r"], ascending=[False, False]).iloc[0]
    lines = [
        "# Random Feature Ridge Compact-Input LOYO",
        "",
        "This run tested random-feature ridge baselines on the same compact observed-data SST input families used for the kernel-ridge baseline.",
        "",
        "- `rff` uses random Fourier features.",
        "- `relu` uses random ReLU features.",
        "- `rf_dim` was tuned in `{10,25,50,100}` by inner LOYO.",
        "",
        "Best overall model by LOYO R2: `{}` with `R2={:.6f}`, `RMSE={:.6f}`, `sign_accuracy={:.6f}`.".format(
            str(best_row["display_name"]),
            float(best_row["R2"]),
            float(best_row["RMSE"]),
            float(best_row["sign_accuracy"]),
        ),
        "",
        "See the CSV outputs for exact per-model metrics and selected random-feature hyperparameters.",
        "",
        "- Predictor table source reused from compact-kernel setup: `{}`".format(KERNEL_PREDICTOR_TABLE_CSV),
    ]
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    predictor_table, _ = build_predictor_table()
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)
    pred_df, hyper_df = run_all_models(predictor_table)
    metrics_df, period_df = compute_metrics(pred_df)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMETERS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    plot_timeseries(pred_df)
    plot_scatter(pred_df, metrics_df)
    write_readme(metrics_df)

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
