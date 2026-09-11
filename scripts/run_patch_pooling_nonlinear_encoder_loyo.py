#!/usr/bin/env python3
"""
Strict LOYO patch-pooling nonlinear encoder for scalar April 1 Sierra SWE.
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
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direct_sst_sparse_common import (  # noqa: E402
    MONTHS,
    WATER_YEARS,
    build_raw_sst_cube,
    compute_metric_bundle,
    compute_period_metrics,
    dump_json,
    load_target_table,
    patch_grid_for_size,
    prepare_patch_features,
    standardize_target_train_only,
    standardize_train_only,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "patch_pooling_nonlinear_encoder_loyo"
METADATA_JSON = OUTPUT_DIR / "patch_pooling_encoder_metadata.json"
PREDICTIONS_CSV = OUTPUT_DIR / "patch_pooling_encoder_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "patch_pooling_encoder_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "patch_pooling_encoder_period_metrics.csv"
HYPERPARAMETERS_CSV = OUTPUT_DIR / "patch_pooling_encoder_selected_hyperparameters.csv"
LATENT_CSV = OUTPUT_DIR / "patch_pooling_encoder_latent_coordinates.csv"
PATCH_IMPORTANCE_CSV = OUTPUT_DIR / "patch_pooling_encoder_patch_importance_by_fold.csv"
PATCH_FREQUENCY_CSV = OUTPUT_DIR / "patch_pooling_encoder_patch_frequency.csv"
LATENT_ORACLE_CORR_CSV = OUTPUT_DIR / "patch_pooling_encoder_latent_oracle_correlations.csv"
LATENT_AMV_CORR_CSV = OUTPUT_DIR / "patch_pooling_encoder_latent_amv_correlations.csv"
SUMMARY_JSON = OUTPUT_DIR / "patch_pooling_encoder_summary.json"
README_MD = OUTPUT_DIR / "README.md"
OBS_PRED_PNG = OUTPUT_DIR / "patch_pooling_encoder_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "patch_pooling_encoder_scatter.png"
PATCH_FREQ_PNG = OUTPUT_DIR / "patch_pooling_encoder_patch_frequency_maps.png"
PATCH_IMPORTANCE_PNG = OUTPUT_DIR / "patch_pooling_encoder_mean_importance_maps.png"

AMV_SOURCE = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)
Z_TABLE = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)

PATCH_SIZES = [5, 10]
LATENT_DIMS = [1, 2, 3, 5]
ACTIVATIONS = ["tanh", "relu"]
ALPHA_GRID = np.asarray([1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1], dtype=np.float64)
RIDGE_ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0], dtype=np.float64)
MAX_LATENT_DIM = max(LATENT_DIMS)
TOP_PATCHES_PER_FOLD = 5


def ensure_runtime_on_compute_node():
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_ridge_standardized(x_train_std, y_train_std, alpha):
    xtx = np.dot(x_train_std.T, x_train_std)
    ridge = xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64)
    rhs = np.dot(x_train_std.T, y_train_std)
    return np.linalg.solve(ridge, rhs)


def predict_linear_ridge_raw(x_train_raw, y_train_raw, x_test_raw, alpha):
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(x_train_std, y_train_std, float(alpha))
    pred_std = float(np.dot(x_test_std, beta_std))
    pred_raw = float(y_mean + y_std * pred_std)
    beta_raw = y_std * beta_std / x_std
    intercept = float(y_mean - np.sum(beta_raw * x_mean))
    return pred_raw, intercept, beta_raw


def fit_patch_mlp(x_train_std, y_train_std, latent_dim, activation, alpha):
    model = MLPRegressor(
        hidden_layer_sizes=(int(latent_dim),),
        activation=str(activation),
        solver="lbfgs",
        alpha=float(alpha),
        max_iter=5000,
        random_state=0,
    )
    model.fit(x_train_std, y_train_std)
    return model


def hidden_activation(model, x_std):
    hidden_linear = np.dot(
        np.asarray(x_std, dtype=np.float64),
        np.asarray(model.coefs_[0], dtype=np.float64),
    ) + np.asarray(model.intercepts_[0], dtype=np.float64)[None, :]
    if model.activation == "tanh":
        return np.tanh(hidden_linear)
    if model.activation == "relu":
        return np.maximum(hidden_linear, 0.0)
    raise ValueError("Unsupported activation {}".format(model.activation))


def choose_best_ridge_alpha(x_train_raw, y_train_raw):
    best_alpha = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for alpha in RIDGE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            pred_raw, _, _ = predict_linear_ridge_raw(x_train_raw[mask], y_train_raw[mask], x_train_raw[~mask][0], float(alpha))
            preds[inner_idx] = pred_raw
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = float(mse)
    return best_alpha, best_mse


def choose_best_patch_mlp(x_train_raw, y_train_raw):
    best = None
    best_mse = None
    n_train = x_train_raw.shape[0]
    for latent_dim in LATENT_DIMS:
        for activation in ACTIVATIONS:
            for alpha in ALPHA_GRID.tolist():
                preds = np.full(n_train, np.nan, dtype=np.float64)
                for inner_idx in range(n_train):
                    mask = np.ones(n_train, dtype=bool)
                    mask[inner_idx] = False
                    x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(x_train_raw[mask], x_train_raw[~mask][0])
                    y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw[mask])
                    model = fit_patch_mlp(x_inner_train_std, y_inner_train_std, latent_dim, activation, alpha)
                    pred_std = float(model.predict(x_inner_valid_std[None, :])[0])
                    preds[inner_idx] = float(y_mean + y_std * pred_std)
                mse = float(np.mean((preds - y_train_raw) ** 2))
                tie_key = (int(latent_dim), 0 if activation == "tanh" else 1, float(alpha))
                if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and tie_key < best["tie_key"]):
                    best = {
                        "latent_dim": int(latent_dim),
                        "activation": str(activation),
                        "alpha": float(alpha),
                        "inner_cv_mse": float(mse),
                        "tie_key": tie_key,
                    }
                    best_mse = float(mse)
    return best


def compute_patch_importance(model):
    input_hidden = np.asarray(model.coefs_[0], dtype=np.float64)
    hidden_out = np.asarray(model.coefs_[1], dtype=np.float64).reshape(-1)
    return np.sum(np.abs(input_hidden * hidden_out[None, :]), axis=1)


def plot_patch_panel_maps(arrays_by_size, patch_meta_by_size, out_path, title_prefix, cmap):
    fig, axes = plt.subplots(len(PATCH_SIZES), len(MONTHS), figsize=(2.7 * len(MONTHS), 2.6 * len(PATCH_SIZES)), constrained_layout=True)
    for row_idx, patch_size in enumerate(PATCH_SIZES):
        arr = arrays_by_size[patch_size]
        meta = patch_meta_by_size[patch_size]
        grid_info = patch_grid_for_size(meta["lat_center"].drop_duplicates().to_numpy(), meta["lon_center"].drop_duplicates().to_numpy(), patch_size)
        lat_centers = grid_info["lat_centers"]
        lon_centers = grid_info["lon_centers"]
        lon_plot = np.arange(len(lon_centers) + 1)
        lat_plot = np.arange(len(lat_centers) + 1)
        for month_idx, month_name in enumerate(MONTHS):
            ax = axes[row_idx, month_idx]
            month_arr = arr[month_idx]
            vmax = float(np.nanmax(np.abs(month_arr)))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            mesh = ax.pcolormesh(
                lon_plot,
                lat_plot,
                month_arr,
                shading="auto",
                cmap=cmap,
                vmin=(-vmax if cmap == "coolwarm" else 0.0),
                vmax=vmax,
            )
            ax.set_title("{}deg {}".format(patch_size, month_name))
            ax.set_xlabel("lon patch")
            ax.set_ylabel("lat patch")
            fig.colorbar(mesh, ax=ax, shrink=0.72)
    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_obs_vs_pred(pred_df):
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    base = pred_df[pred_df["model_name"] == "PATCH_RIDGE_5DEG"].sort_values("heldout_wy")
    ax.plot(base["heldout_wy"], base["obs_swe"], color="black", linewidth=2.5, label="Observed SWE anomaly")
    colors = {
        "PATCH_RIDGE_5DEG": "#1f77b4",
        "PATCH_MLP_5DEG": "#6baed6",
        "PATCH_RIDGE_10DEG": "#d62728",
        "PATCH_MLP_10DEG": "#ff9896",
    }
    for model_name in ["PATCH_RIDGE_5DEG", "PATCH_MLP_5DEG", "PATCH_RIDGE_10DEG", "PATCH_MLP_10DEG"]:
        subset = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(subset["heldout_wy"], subset["pred_swe"], color=colors[model_name], linewidth=1.4, label=model_name)
    ax.set_title("Strict LOYO patch-pooling encoder and same-input ridge baselines")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(pred_df, metrics_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True, sharex=True, sharey=True)
    colors = {
        "PATCH_RIDGE_5DEG": "#1f77b4",
        "PATCH_MLP_5DEG": "#6baed6",
        "PATCH_RIDGE_10DEG": "#d62728",
        "PATCH_MLP_10DEG": "#ff9896",
    }
    for ax, model_name in zip(axes.flat, ["PATCH_RIDGE_5DEG", "PATCH_MLP_5DEG", "PATCH_RIDGE_10DEG", "PATCH_MLP_10DEG"]):
        subset = pred_df[pred_df["model_name"] == model_name]
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        metrics = metrics_df[metrics_df["model_name"] == model_name].iloc[0]
        lo = float(min(np.min(obs), np.min(pred)))
        hi = float(max(np.max(obs), np.max(pred)))
        ax.scatter(obs, pred, color=colors[model_name], alpha=0.85, s=38)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_title("{}\nR2={:.3f} RMSE={:.3f} sign={:.3f}".format(model_name, metrics["R2"], metrics["RMSE"], metrics["sign_accuracy"]))
        ax.set_xlabel("Observed SWE anomaly (m)")
        ax.set_ylabel("Predicted SWE anomaly (m)")
        ax.grid(alpha=0.25)
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_auxiliary_tables():
    amv = pd.read_csv(AMV_SOURCE)
    amv["water_year"] = amv["water_year"].astype(int)
    amv = amv.sort_values("water_year").reset_index(drop=True)
    z = pd.read_csv(Z_TABLE)
    z["water_year"] = z["water_year"].astype(int)
    z = z[["water_year", "Z1", "Z2"]].sort_values("water_year").reset_index(drop=True)
    return amv, z


def corrcoef_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def main():
    ensure_runtime_on_compute_node()
    ensure_output_dir()

    target_df = load_target_table()
    raw_cube, lat, lon, metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(metadata["valid_feature_mask"], dtype=bool)
    y_raw = target_df["obs_swe"].to_numpy(dtype=float)
    amv_df, z_df = build_auxiliary_tables()

    dump_json(
        METADATA_JSON,
        {
            "input_sst_file": str(Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")),
            "target_file": str(Z_TABLE),
            "amv_source": str(AMV_SOURCE),
            "domain": metadata["domain"],
            "months": metadata["months"],
            "water_years": WATER_YEARS.astype(int).tolist(),
            "patch_sizes_deg": PATCH_SIZES,
            "latent_dims": LATENT_DIMS,
            "activations": ACTIVATIONS,
        },
    )

    prediction_rows = []
    hyper_rows = []
    latent_rows = []
    importance_rows = []
    metrics_rows = []
    patch_meta_by_size = {}
    frequency_maps = {}
    mean_importance_maps = {}

    for patch_size in PATCH_SIZES:
        years = WATER_YEARS.copy()
        ridge_preds = np.full(years.size, np.nan, dtype=float)
        mlp_preds = np.full(years.size, np.nan, dtype=float)
        latent_holdout = np.full((years.size, MAX_LATENT_DIM), np.nan, dtype=float)
        grid_info = patch_grid_for_size(lat, lon, patch_size)
        lat_count = int(grid_info["lat_count"])
        lon_count = int(grid_info["lon_count"])
        importance_by_fold = np.full((years.size, len(MONTHS), lat_count, lon_count), np.nan, dtype=float)
        selected_by_fold = np.zeros((years.size, len(MONTHS), lat_count, lon_count), dtype=np.int8)
        patch_meta_template = None

        for outer_idx, water_year in enumerate(years):
            mask = np.ones(years.size, dtype=bool)
            mask[outer_idx] = False
            x_train_raw, x_test_raw, patch_meta = prepare_patch_features(raw_cube[mask], raw_cube[~mask][0], valid_feature_mask, lat, lon, patch_size)
            if patch_meta_template is None:
                patch_meta_template = patch_meta.copy()
            y_train = y_raw[mask]
            y_test = float(y_raw[~mask][0])

            ridge_alpha, ridge_inner_mse = choose_best_ridge_alpha(x_train_raw, y_train)
            ridge_pred, ridge_intercept, ridge_beta = predict_linear_ridge_raw(x_train_raw, y_train, x_test_raw, ridge_alpha)
            ridge_preds[outer_idx] = ridge_pred
            ridge_error = ridge_pred - y_test
            prediction_rows.append(
                {
                    "model_name": "PATCH_RIDGE_{}DEG".format(patch_size),
                    "patch_size_deg": int(patch_size),
                    "heldout_wy": int(water_year),
                    "obs_swe": y_test,
                    "pred_swe": ridge_pred,
                    "error_pred_minus_obs": ridge_error,
                    "residual_obs_minus_pred": -ridge_error,
                    "abs_error": abs(ridge_error),
                    "sign_correct": float((np.sign(ridge_pred) == np.sign(y_test)) and (ridge_pred != 0.0) and (y_test != 0.0)),
                }
            )
            hyper_rows.append(
                {
                    "model_name": "PATCH_RIDGE_{}DEG".format(patch_size),
                    "patch_size_deg": int(patch_size),
                    "heldout_wy": int(water_year),
                    "selected_alpha": float(ridge_alpha),
                    "inner_cv_mse": float(ridge_inner_mse),
                    "latent_dim": np.nan,
                    "activation": "linear",
                }
            )

            best = choose_best_patch_mlp(x_train_raw, y_train)
            x_train_std, x_test_std, _, _ = standardize_train_only(x_train_raw, x_test_raw)
            y_train_std, y_mean, y_std = standardize_target_train_only(y_train)
            mlp = fit_patch_mlp(x_train_std, y_train_std, best["latent_dim"], best["activation"], best["alpha"])
            pred_std = float(mlp.predict(x_test_std[None, :])[0])
            pred_raw = float(y_mean + y_std * pred_std)
            mlp_preds[outer_idx] = pred_raw
            mlp_error = pred_raw - y_test
            latent_test = hidden_activation(mlp, x_test_std[None, :])[0]
            latent_holdout[outer_idx, : best["latent_dim"]] = latent_test
            prediction_rows.append(
                {
                    "model_name": "PATCH_MLP_{}DEG".format(patch_size),
                    "patch_size_deg": int(patch_size),
                    "heldout_wy": int(water_year),
                    "obs_swe": y_test,
                    "pred_swe": pred_raw,
                    "error_pred_minus_obs": mlp_error,
                    "residual_obs_minus_pred": -mlp_error,
                    "abs_error": abs(mlp_error),
                    "sign_correct": float((np.sign(pred_raw) == np.sign(y_test)) and (pred_raw != 0.0) and (y_test != 0.0)),
                }
            )
            hyper_rows.append(
                {
                    "model_name": "PATCH_MLP_{}DEG".format(patch_size),
                    "patch_size_deg": int(patch_size),
                    "heldout_wy": int(water_year),
                    "selected_alpha": float(best["alpha"]),
                    "inner_cv_mse": float(best["inner_cv_mse"]),
                    "latent_dim": int(best["latent_dim"]),
                    "activation": best["activation"],
                }
            )
            for dim_idx in range(MAX_LATENT_DIM):
                latent_rows.append(
                    {
                        "model_name": "PATCH_MLP_{}DEG".format(patch_size),
                        "patch_size_deg": int(patch_size),
                        "heldout_wy": int(water_year),
                        "latent_dim_index": int(dim_idx + 1),
                        "latent_value": float(latent_holdout[outer_idx, dim_idx]) if np.isfinite(latent_holdout[outer_idx, dim_idx]) else np.nan,
                    }
                )

            importance = compute_patch_importance(mlp)
            patch_grid = np.full((len(MONTHS), lat_count, lon_count), np.nan, dtype=float)
            for feat_idx, row in patch_meta.iterrows():
                patch_grid[int(row["month_index"]), int(row["lat_patch"]), int(row["lon_patch"])] = importance[feat_idx]
            importance_by_fold[outer_idx] = patch_grid
            top_indices = np.argsort(importance)[-min(TOP_PATCHES_PER_FOLD, importance.size) :]
            for feat_idx in top_indices.tolist():
                row = patch_meta.iloc[feat_idx]
                selected_by_fold[outer_idx, int(row["month_index"]), int(row["lat_patch"]), int(row["lon_patch"])] = 1
            for feat_idx, row in patch_meta.iterrows():
                importance_rows.append(
                    {
                        "model_name": "PATCH_MLP_{}DEG".format(patch_size),
                        "patch_size_deg": int(patch_size),
                        "heldout_wy": int(water_year),
                        "month": row["month"],
                        "month_index": int(row["month_index"]),
                        "lat_patch": int(row["lat_patch"]),
                        "lon_patch": int(row["lon_patch"]),
                        "lat_center": float(row["lat_center"]),
                        "lon_center": float(row["lon_center"]),
                        "importance": float(importance[feat_idx]),
                        "is_top_patch": int(feat_idx in top_indices),
                    }
                )
            print(
                "LOYO heldout_WY={} model=PATCH_MLP_{}DEG latent_dim={} activation={} alpha={} obs={:.6f} pred={:.6f}".format(
                    int(water_year),
                    patch_size,
                    int(best["latent_dim"]),
                    best["activation"],
                    "{:g}".format(float(best["alpha"])),
                    y_test,
                    pred_raw,
                ),
                flush=True,
            )

        patch_meta_by_size[patch_size] = patch_meta_template
        frequency_maps[patch_size] = np.mean(selected_by_fold, axis=0)
        mean_importance_maps[patch_size] = np.nanmean(importance_by_fold, axis=0)
        metrics_rows.append({"model_name": "PATCH_RIDGE_{}DEG".format(patch_size), "patch_size_deg": int(patch_size), **compute_metric_bundle(y_raw, ridge_preds)})
        metrics_rows.append({"model_name": "PATCH_MLP_{}DEG".format(patch_size), "patch_size_deg": int(patch_size), **compute_metric_bundle(y_raw, mlp_preds)})

    pred_df = pd.DataFrame(prediction_rows)
    hyper_df = pd.DataFrame(hyper_rows)
    latent_df = pd.DataFrame(latent_rows)
    importance_df = pd.DataFrame(importance_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    period_df = compute_period_metrics(pred_df, "model_name")

    freq_rows = []
    for patch_size in PATCH_SIZES:
        template = patch_meta_by_size[patch_size][["month", "month_index", "lat_patch", "lon_patch", "lat_center", "lon_center"]].drop_duplicates()
        counts = (
            importance_df[(importance_df["patch_size_deg"] == patch_size) & (importance_df["is_top_patch"] == 1)]
            .groupby(["month", "month_index", "lat_patch", "lon_patch", "lat_center", "lon_center"], as_index=False)
            .size()
            .rename(columns={"size": "selection_count"})
        )
        freq_df = template.merge(counts, on=["month", "month_index", "lat_patch", "lon_patch", "lat_center", "lon_center"], how="left")
        freq_df["selection_count"] = freq_df["selection_count"].fillna(0).astype(int)
        freq_df["selection_frequency"] = freq_df["selection_count"] / float(len(WATER_YEARS))
        freq_df["patch_size_deg"] = int(patch_size)
        freq_rows.extend(freq_df.to_dict(orient="records"))
    freq_df = pd.DataFrame(freq_rows)

    oracle_rows = []
    latent_wide = latent_df.pivot_table(index=["patch_size_deg", "heldout_wy"], columns="latent_dim_index", values="latent_value").reset_index()
    latent_wide.columns = ["patch_size_deg", "heldout_wy"] + ["z{}".format(idx) for idx in range(1, MAX_LATENT_DIM + 1)]
    latent_wide = latent_wide.rename(columns={"heldout_wy": "water_year"})
    merged_oracle = latent_wide.merge(z_df, on="water_year", how="left")
    for patch_size in PATCH_SIZES:
        sub = merged_oracle[merged_oracle["patch_size_deg"] == patch_size]
        for latent_name in ["z{}".format(idx) for idx in range(1, MAX_LATENT_DIM + 1)]:
            oracle_rows.append(
                {
                    "patch_size_deg": int(patch_size),
                    "latent_name": latent_name,
                    "corr_with_Z1": corrcoef_safe(sub[latent_name].to_numpy(dtype=float), sub["Z1"].to_numpy(dtype=float)),
                    "corr_with_Z2": corrcoef_safe(sub[latent_name].to_numpy(dtype=float), sub["Z2"].to_numpy(dtype=float)),
                }
            )
    oracle_df = pd.DataFrame(oracle_rows)

    amv_rows = []
    amv_cols = [col for col in amv_df.columns if col.startswith("AMV_PC")]
    merged_amv = latent_wide.merge(amv_df[["water_year"] + amv_cols], on="water_year", how="left")
    for patch_size in PATCH_SIZES:
        sub = merged_amv[merged_amv["patch_size_deg"] == patch_size]
        for latent_name in ["z{}".format(idx) for idx in range(1, MAX_LATENT_DIM + 1)]:
            corr_pairs = []
            for amv_col in amv_cols:
                corr_pairs.append((amv_col, corrcoef_safe(sub[latent_name].to_numpy(dtype=float), sub[amv_col].to_numpy(dtype=float))))
            corr_pairs = [pair for pair in corr_pairs if np.isfinite(pair[1])]
            if corr_pairs:
                best_name, best_corr = max(corr_pairs, key=lambda pair: abs(pair[1]))
            else:
                best_name, best_corr = "none", np.nan
            amv_rows.append(
                {
                    "patch_size_deg": int(patch_size),
                    "latent_name": latent_name,
                    "top_amv_column": best_name,
                    "top_amv_correlation": best_corr,
                }
            )
    amv_corr_df = pd.DataFrame(amv_rows)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMETERS_CSV, index=False)
    latent_df.to_csv(LATENT_CSV, index=False)
    importance_df.to_csv(PATCH_IMPORTANCE_CSV, index=False)
    freq_df.to_csv(PATCH_FREQUENCY_CSV, index=False)
    oracle_df.to_csv(LATENT_ORACLE_CORR_CSV, index=False)
    amv_corr_df.to_csv(LATENT_AMV_CORR_CSV, index=False)

    plot_obs_vs_pred(pred_df)
    plot_scatter(pred_df, metrics_df)
    plot_patch_panel_maps(frequency_maps, patch_meta_by_size, PATCH_FREQ_PNG, "Patch top-frequency across LOYO folds", "viridis")
    plot_patch_panel_maps(mean_importance_maps, patch_meta_by_size, PATCH_IMPORTANCE_PNG, "Mean patch importance", "coolwarm")

    best_r2 = metrics_df.sort_values("R2", ascending=False).iloc[0]
    best_rmse = metrics_df.sort_values("RMSE", ascending=True).iloc[0]
    summary = {
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_df.to_dict(orient="records"),
        "best_model_by_R2": best_r2.to_dict(),
        "best_model_by_RMSE": best_rmse.to_dict(),
        "top_patch_frequency_rows": freq_df.sort_values("selection_frequency", ascending=False).head(20).to_dict(orient="records"),
        "top_latent_oracle_correlations": oracle_df.to_dict(orient="records"),
        "top_latent_amv_correlations": amv_corr_df.to_dict(orient="records"),
    }
    dump_json(SUMMARY_JSON, summary)
    README_MD.write_text(
        "\n".join(
            [
                "# Patch-Pooling Nonlinear Encoder LOYO",
                "",
                "This run compares same-input ridge and a tiny patch-pooling MLP encoder on 5-degree and 10-degree monthly SST patch summaries.",
                "",
                "Best model by R2: `{}` with `R2={:.6f}`, `RMSE={:.6f}`, `sign_accuracy={:.6f}`.".format(
                    str(best_r2["model_name"]), float(best_r2["R2"]), float(best_r2["RMSE"]), float(best_r2["sign_accuracy"])
                ),
                "",
                "Saved outputs include held-out latent coordinates, patch importance maps, patch top-frequency maps, and correlations between latent coordinates and oracle/AMV predictors.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Patch pooling output directory: {}".format(OUTPUT_DIR), flush=True)
    print("Metrics: {}".format(METRICS_CSV), flush=True)
    print("Latents: {}".format(LATENT_CSV), flush=True)
    print("Best model by R2: {}".format(best_r2["model_name"]), flush=True)


if __name__ == "__main__":
    main()
