#!/usr/bin/env python3
"""
Strict LOYO tiny 15-degree patch MLP for scalar April 1 Sierra SWE.
"""

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direct_sst_sparse_common import (  # noqa: E402
    BASELINE_METRICS,
    MONTHS,
    WATER_YEARS,
    build_raw_sst_cube,
    compute_metric_bundle,
    compute_period_metrics,
    dump_json,
    ensure_runtime_on_compute_node,
    load_target_table,
    patch_grid_for_size,
    prepare_patch_features,
    standardize_target_train_only,
    standardize_train_only,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "patch_mlp_15deg_fast_loyo"
METADATA_JSON = OUTPUT_DIR / "patch_mlp_15deg_metadata.json"
PATCH_FEATURE_TABLE_CSV = OUTPUT_DIR / "patch_mlp_15deg_patch_feature_table.csv"
PREDICTIONS_CHECKPOINT_CSV = OUTPUT_DIR / "patch_mlp_15deg_predictions_checkpoint.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "patch_mlp_15deg_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "patch_mlp_15deg_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "patch_mlp_15deg_period_metrics.csv"
HYPERPARAMS_CSV = OUTPUT_DIR / "patch_mlp_15deg_selected_hyperparams_by_fold.csv"
INNER_CV_CSV = OUTPUT_DIR / "patch_mlp_15deg_inner_cv_results.csv"
WARNINGS_CSV = OUTPUT_DIR / "patch_mlp_15deg_fit_warnings.csv"
SUMMARY_JSON = OUTPUT_DIR / "patch_mlp_15deg_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "patch_mlp_15deg_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "patch_mlp_15deg_scatter.png"
ERROR_PNG = OUTPUT_DIR / "patch_mlp_15deg_error_by_year.png"

PATCH_SIZE_DEG = 15
MODEL_NAME = "PATCH_MLP_15DEG"
CANDIDATE_CONFIGS = [
    {"latent_dim": 1, "activation": "tanh", "alpha": 1.0},
    {"latent_dim": 1, "activation": "tanh", "alpha": 10.0},
    {"latent_dim": 2, "activation": "tanh", "alpha": 1.0},
    {"latent_dim": 2, "activation": "tanh", "alpha": 10.0},
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def deterministic_seed(outer_heldout_wy: int, inner_valid_wy: int, latent_dim: int, alpha: float, stage: str) -> int:
    stage_code = 17 if stage == "outer_refit" else 11
    alpha_code = int(round(alpha * 100))
    return int((outer_heldout_wy * 100000) + ((inner_valid_wy + 1) * 100) + latent_dim * 10 + alpha_code + stage_code)


def build_global_patch_matrix(
    cube: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    grid = patch_grid_for_size(lat, lon, PATCH_SIZE_DEG)
    lat_ids = grid["lat_ids"]
    lon_ids = grid["lon_ids"]
    lat_count = int(grid["lat_count"])
    lon_count = int(grid["lon_count"])
    cos_weights = np.cos(np.deg2rad(lat))
    monthly_clim = np.mean(cube, axis=0)
    anomalies = cube - monthly_clim[None, :, :, :]

    rows = []
    cols = []
    feature_index = 0
    for month_idx, month_name in enumerate(MONTHS):
        for lat_patch in range(lat_count):
            lat_mask = lat_ids == lat_patch
            for lon_patch in range(lon_count):
                lon_mask = lon_ids == lon_patch
                cell_mask = valid_feature_mask[month_idx, :, :] & lat_mask[:, None] & lon_mask[None, :]
                if not np.any(cell_mask):
                    continue
                weights_2d = np.broadcast_to(cos_weights[:, None], cell_mask.shape)
                group_weights = weights_2d[cell_mask]
                weight_sum = float(np.sum(group_weights))
                values = anomalies[:, month_idx, :, :][:, cell_mask]
                patch_mean = np.sum(values * group_weights[None, :], axis=1) / weight_sum
                cols.append(np.asarray(patch_mean, dtype=np.float64))
                rows.append(
                    {
                        "feature_index": int(feature_index),
                        "patch_size_deg": int(PATCH_SIZE_DEG),
                        "month": month_name,
                        "month_index": int(month_idx),
                        "lat_patch": int(lat_patch),
                        "lon_patch": int(lon_patch),
                        "lat_lower": float(-10.0 + PATCH_SIZE_DEG * lat_patch),
                        "lat_upper": float(-10.0 + PATCH_SIZE_DEG * (lat_patch + 1)),
                        "lon_lower": float(120.0 + PATCH_SIZE_DEG * lon_patch),
                        "lon_upper": float(120.0 + PATCH_SIZE_DEG * (lon_patch + 1)),
                        "lat_center": float(grid["lat_centers"][lat_patch]),
                        "lon_center": float(grid["lon_centers"][lon_patch]),
                        "num_cells": int(np.sum(cell_mask)),
                    }
                )
                feature_index += 1

    matrix = np.column_stack(cols).astype(np.float64)
    meta = pd.DataFrame(rows)
    nonfinite_mask = ~np.all(np.isfinite(matrix), axis=0)
    valid_nonfinite = ~nonfinite_mask
    variance = np.var(matrix[:, valid_nonfinite], axis=0, ddof=1) if np.any(valid_nonfinite) else np.asarray([], dtype=np.float64)
    zero_var_mask_valid = variance <= 0.0
    zero_var_mask = np.zeros(matrix.shape[1], dtype=bool)
    zero_var_mask[valid_nonfinite] = zero_var_mask_valid
    final_mask = valid_nonfinite & ~zero_var_mask

    meta["is_nonfinite_removed"] = nonfinite_mask.astype(int)
    meta["is_zero_variance_removed"] = zero_var_mask.astype(int)
    meta["is_final_valid"] = final_mask.astype(int)
    meta["global_std"] = np.std(matrix, axis=0, ddof=1)
    meta["global_mean"] = np.mean(matrix, axis=0)
    return matrix[:, final_mask], meta


def build_metadata(feature_table: pd.DataFrame, final_matrix_shape: tuple[int, int]) -> dict[str, object]:
    lat_bounds = (
        feature_table[["lat_patch", "lat_lower", "lat_upper"]]
        .drop_duplicates()
        .sort_values("lat_patch")
        .to_dict(orient="records")
    )
    lon_bounds = (
        feature_table[["lon_patch", "lon_lower", "lon_upper"]]
        .drop_duplicates()
        .sort_values("lon_patch")
        .to_dict(orient="records")
    )
    return {
        "original_patch_feature_count": int(feature_table.shape[0]),
        "valid_patch_feature_count": int(feature_table["is_final_valid"].sum()),
        "removed_nonfinite_count": int(feature_table["is_nonfinite_removed"].sum()),
        "removed_zero_variance_count": int(feature_table["is_zero_variance_removed"].sum()),
        "final_feature_matrix_shape": [int(final_matrix_shape[0]), int(final_matrix_shape[1])],
        "patch_size_deg": int(PATCH_SIZE_DEG),
        "months": MONTHS,
        "lat_patch_bounds": lat_bounds,
        "lon_patch_bounds": lon_bounds,
    }


def record_warnings(caught, outer_heldout_wy: int, inner_valid_wy: int | None, stage: str, config: dict[str, object], model: MLPRegressor | None) -> list[dict[str, object]]:
    rows = []
    n_iter = float(getattr(model, "n_iter_", np.nan)) if model is not None else np.nan
    loss = float(getattr(model, "loss_", np.nan)) if model is not None and hasattr(model, "loss_") else np.nan
    for item in caught:
        rows.append(
            {
                "outer_heldout_wy": int(outer_heldout_wy),
                "inner_valid_wy": (np.nan if inner_valid_wy is None else int(inner_valid_wy)),
                "stage": stage,
                "latent_dim": int(config["latent_dim"]),
                "activation": str(config["activation"]),
                "alpha": float(config["alpha"]),
                "warning_type": item.category.__name__,
                "warning_message": str(item.message),
                "n_iter": n_iter,
                "loss": loss,
            }
        )
    return rows


def fit_and_predict(
    x_train_std: np.ndarray,
    y_train_std: np.ndarray,
    x_valid_std: np.ndarray,
    outer_heldout_wy: int,
    inner_valid_wy: int | None,
    stage: str,
    config: dict[str, object],
) -> tuple[float, MLPRegressor, list[dict[str, object]]]:
    seed = deterministic_seed(
        outer_heldout_wy=outer_heldout_wy,
        inner_valid_wy=(-1 if inner_valid_wy is None else inner_valid_wy),
        latent_dim=int(config["latent_dim"]),
        alpha=float(config["alpha"]),
        stage=stage,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MLPRegressor(
            hidden_layer_sizes=(int(config["latent_dim"]),),
            activation="tanh",
            solver="adam",
            alpha=float(config["alpha"]),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            tol=1.0e-4,
            random_state=seed,
        )
        model.fit(x_train_std, y_train_std)
        pred_std = float(model.predict(x_valid_std[None, :])[0])
    warning_rows = record_warnings(caught, outer_heldout_wy, inner_valid_wy, stage, config, model)
    return pred_std, model, warning_rows


def choose_config_for_outer_fold(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    train_years: np.ndarray,
    outer_heldout_wy: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    inner_rows = []
    warning_rows = []
    best_config = None
    best_mse = None

    for config in CANDIDATE_CONFIGS:
        squared_errors = []
        for inner_valid_wy in train_years.tolist():
            inner_valid_mask = train_years == inner_valid_wy
            inner_train_mask = ~inner_valid_mask
            x_inner_train_raw = x_train_raw[inner_train_mask]
            x_inner_valid_raw = x_train_raw[inner_valid_mask][0]
            y_inner_train_raw = y_train_raw[inner_train_mask]
            y_inner_valid_raw = float(y_train_raw[inner_valid_mask][0])

            train_std = np.std(x_inner_train_raw, axis=0, ddof=1)
            valid_feature_mask = np.isfinite(train_std) & (train_std > 0.0)
            x_inner_train_clean = x_inner_train_raw[:, valid_feature_mask]
            x_inner_valid_clean = x_inner_valid_raw[valid_feature_mask]
            x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(x_inner_train_clean, x_inner_valid_clean)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train_raw)
            pred_std, model, warning_chunk = fit_and_predict(
                x_train_std=x_inner_train_std,
                y_train_std=y_inner_train_std,
                x_valid_std=x_inner_valid_std,
                outer_heldout_wy=outer_heldout_wy,
                inner_valid_wy=int(inner_valid_wy),
                stage="inner_cv",
                config=config,
            )
            warning_rows.extend(warning_chunk)
            y_valid_std = float((y_inner_valid_raw - y_mean) / y_std)
            se = float((pred_std - y_valid_std) ** 2)
            squared_errors.append(se)
            inner_rows.append(
                {
                    "outer_heldout_wy": int(outer_heldout_wy),
                    "inner_valid_wy": int(inner_valid_wy),
                    "latent_dim": int(config["latent_dim"]),
                    "activation": str(config["activation"]),
                    "alpha": float(config["alpha"]),
                    "inner_squared_error_std": se,
                    "n_features_after_train_cleaning": int(x_inner_train_clean.shape[1]),
                    "n_iter": float(getattr(model, "n_iter_", np.nan)),
                    "loss": float(getattr(model, "loss_", np.nan)),
                }
            )

        mean_mse = float(np.mean(squared_errors))
        tie_key = (int(config["latent_dim"]), float(config["alpha"]))
        if best_mse is None or mean_mse < best_mse - 1.0e-15 or (abs(mean_mse - best_mse) <= 1.0e-15 and tie_key < (int(best_config["latent_dim"]), float(best_config["alpha"]))):
            best_config = dict(config)
            best_config["inner_cv_mse"] = mean_mse
            best_mse = mean_mse

    return best_config, inner_rows, warning_rows


def run_loyo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = load_target_table()
    cube, lat, lon, raw_metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(raw_metadata["valid_feature_mask"], dtype=bool)
    full_matrix, patch_table = build_global_patch_matrix(cube, lat, lon, valid_feature_mask)
    patch_table.to_csv(PATCH_FEATURE_TABLE_CSV, index=False)
    metadata = build_metadata(patch_table, full_matrix.shape)
    dump_json(METADATA_JSON, metadata)

    obs = target["obs_swe"].to_numpy(dtype=np.float64)
    years = target["water_year"].to_numpy(dtype=np.int32)
    prediction_rows = []
    hyper_rows = []
    inner_rows = []
    warning_rows = []

    for heldout_wy in years.tolist():
        test_mask = years == heldout_wy
        train_mask = ~test_mask
        train_years = years[train_mask]
        y_train_raw = obs[train_mask]
        y_test_raw = float(obs[test_mask][0])
        train_cube_raw = cube[train_mask]
        test_cube_raw = cube[test_mask][0]

        x_train_raw, x_test_raw, _ = prepare_patch_features(
            train_cube_raw=train_cube_raw,
            test_cube_raw=test_cube_raw,
            valid_feature_mask=valid_feature_mask,
            lat=lat,
            lon=lon,
            patch_size_deg=PATCH_SIZE_DEG,
        )

        best_config, inner_chunk, warning_chunk = choose_config_for_outer_fold(
            x_train_raw=x_train_raw,
            y_train_raw=y_train_raw,
            train_years=train_years,
            outer_heldout_wy=int(heldout_wy),
        )
        inner_rows.extend(inner_chunk)
        warning_rows.extend(warning_chunk)

        train_std = np.std(x_train_raw, axis=0, ddof=1)
        outer_valid_feature_mask = np.isfinite(train_std) & (train_std > 0.0)
        x_train_clean = x_train_raw[:, outer_valid_feature_mask]
        x_test_clean = x_test_raw[outer_valid_feature_mask]
        x_train_std, x_test_std, _, _ = standardize_train_only(x_train_clean, x_test_clean)
        y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
        pred_std, model, outer_warning_chunk = fit_and_predict(
            x_train_std=x_train_std,
            y_train_std=y_train_std,
            x_valid_std=x_test_std,
            outer_heldout_wy=int(heldout_wy),
            inner_valid_wy=None,
            stage="outer_refit",
            config=best_config,
        )
        warning_rows.extend(outer_warning_chunk)
        pred_raw = float(y_mean + y_std * pred_std)
        error = float(pred_raw - y_test_raw)

        prediction_rows.append(
            {
                "model_name": MODEL_NAME,
                "heldout_wy": int(heldout_wy),
                "obs_swe": y_test_raw,
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error,
                "residual_obs_minus_pred": -error,
                "abs_error": float(abs(error)),
                "sign_correct": float(np.sign(y_test_raw) == np.sign(pred_raw)) if y_test_raw != 0.0 and pred_raw != 0.0 else np.nan,
                "selected_latent_dim": int(best_config["latent_dim"]),
                "selected_activation": str(best_config["activation"]),
                "selected_alpha": float(best_config["alpha"]),
                "inner_cv_mse": float(best_config["inner_cv_mse"]),
            }
        )
        hyper_rows.append(
            {
                "heldout_wy": int(heldout_wy),
                "selected_latent_dim": int(best_config["latent_dim"]),
                "selected_activation": str(best_config["activation"]),
                "selected_alpha": float(best_config["alpha"]),
                "inner_cv_mse": float(best_config["inner_cv_mse"]),
                "outer_train_feature_count": int(x_train_clean.shape[1]),
                "outer_total_patch_count": int(x_train_raw.shape[1]),
                "outer_removed_zero_variance_count": int(np.sum(~outer_valid_feature_mask)),
            }
        )
        pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_CHECKPOINT_CSV, index=False)
        print(
            "LOYO heldout_WY={} model={} latent_dim={} activation={} alpha={} obs={:.6f} pred={:.6f}".format(
                int(heldout_wy),
                MODEL_NAME,
                int(best_config["latent_dim"]),
                str(best_config["activation"]),
                float(best_config["alpha"]),
                y_test_raw,
                pred_raw,
            ),
            flush=True,
        )

    pred_df = pd.DataFrame(prediction_rows).sort_values("heldout_wy").reset_index(drop=True)
    hyper_df = pd.DataFrame(hyper_rows).sort_values("heldout_wy").reset_index(drop=True)
    inner_df = pd.DataFrame(inner_rows).sort_values(["outer_heldout_wy", "latent_dim", "alpha", "inner_valid_wy"]).reset_index(drop=True)
    warn_columns = [
        "outer_heldout_wy",
        "inner_valid_wy",
        "stage",
        "latent_dim",
        "activation",
        "alpha",
        "warning_type",
        "warning_message",
        "n_iter",
        "loss",
    ]
    warn_df = pd.DataFrame(warning_rows, columns=warn_columns)
    return pred_df, hyper_df, inner_df, warn_df


def plot_timeseries(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(pred_df["heldout_wy"], pred_df["obs_swe"], color="black", linewidth=2.2, label="Observed SWE anomaly")
    ax.plot(pred_df["heldout_wy"], pred_df["pred_swe"], color="#1f77b4", linewidth=1.8, label=MODEL_NAME)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly")
    ax.set_title("Strict LOYO 15-degree patch MLP")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(pred_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    obs = pred_df["obs_swe"].to_numpy(dtype=np.float64)
    pred = pred_df["pred_swe"].to_numpy(dtype=np.float64)
    lo = float(min(np.min(obs), np.min(pred)))
    hi = float(max(np.max(obs), np.max(pred)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.scatter(obs, pred, s=45, alpha=0.85, color="#1f77b4")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Observed SWE anomaly")
    ax.set_ylabel("Predicted SWE anomaly")
    ax.set_title("{}\nR2={:.3f}, r={:.3f}".format(MODEL_NAME, metrics["R2"], metrics["r"]))
    ax.grid(alpha=0.25)
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_year(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(pred_df["heldout_wy"], pred_df["error_pred_minus_obs"], color="#d62728", width=0.8, alpha=0.85)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Prediction error")
    ax.set_title("Patch MLP 15-degree LOYO prediction error by year")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(ERROR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def warning_summary_dict(warn_df: pd.DataFrame) -> dict[str, object]:
    if warn_df.empty:
        return {"total_warning_count": 0, "warning_type_counts": {}, "outer_refit_warning_count": 0, "inner_cv_warning_count": 0}
    return {
        "total_warning_count": int(warn_df.shape[0]),
        "warning_type_counts": {str(k): int(v) for k, v in warn_df["warning_type"].value_counts().to_dict().items()},
        "outer_refit_warning_count": int(np.sum(warn_df["stage"] == "outer_refit")),
        "inner_cv_warning_count": int(np.sum(warn_df["stage"] == "inner_cv")),
    }


def build_short_answer(metrics: dict[str, float], hyper_df: pd.DataFrame, warn_df: pd.DataFrame) -> str:
    selected_counts = (
        hyper_df.groupby(["selected_latent_dim", "selected_alpha"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "selected_latent_dim", "selected_alpha"], ascending=[False, True, True])
    )
    top = selected_counts.iloc[0]
    beat_text = "No."
    if np.isfinite(metrics["R2"]) and metrics["R2"] >= BASELINE_METRICS["R2"] - 0.02:
        beat_text = "It approached the baseline."
    if np.isfinite(metrics["R2"]) and metrics["R2"] >= BASELINE_METRICS["R2"]:
        beat_text = "Yes, it beat the baseline."
    return (
        "Did the tiny 15-degree patch MLP finish? Yes. "
        "Did it beat or approach the linear Z1_Z2 + AMV/AMO K5 baseline? {} "
        "Were there convergence warnings? {} "
        "Which latent_dim and alpha were most often selected? latent_dim={} and alpha={}.".format(
            beat_text,
            "Yes." if not warn_df.empty else "No.",
            int(top["selected_latent_dim"]),
            float(top["selected_alpha"]),
        )
    )


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    pred_df, hyper_df, inner_df, warn_df = run_loyo()
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMS_CSV, index=False)
    inner_df.to_csv(INNER_CV_CSV, index=False)
    warn_df.to_csv(WARNINGS_CSV, index=False)

    metrics = compute_metric_bundle(pred_df["obs_swe"].to_numpy(dtype=np.float64), pred_df["pred_swe"].to_numpy(dtype=np.float64))
    metrics_df = pd.DataFrame([{ "model_name": MODEL_NAME, **metrics }])
    metrics_df.to_csv(METRICS_CSV, index=False)

    period_df = compute_period_metrics(pred_df, "model_name")
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)

    plot_timeseries(pred_df)
    plot_scatter(pred_df, metrics)
    plot_error_by_year(pred_df)

    metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    warning_summary = warning_summary_dict(warn_df)
    selected_summary = (
        hyper_df.groupby(["selected_latent_dim", "selected_alpha"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "selected_latent_dim", "selected_alpha"], ascending=[False, True, True])
    )
    summary = {
        "input_files": {
            "target_table": str(PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"),
            "sst_file": "/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc",
        },
        "patch_size_deg": int(PATCH_SIZE_DEG),
        "final_feature_matrix_shape": metadata["final_feature_matrix_shape"],
        "candidate_configs": CANDIDATE_CONFIGS,
        "metrics": metrics,
        "period_metrics": period_df.to_dict(orient="records"),
        "selected_hyperparams_by_fold": hyper_df.to_dict(orient="records"),
        "warning_summary": warning_summary,
        "comparison_to_Z1_Z2_AMV_AMO_K5": {
            "baseline": BASELINE_METRICS,
            "patch_mlp_15deg": {
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "sign_accuracy": metrics["sign_accuracy"],
            },
        },
        "short_answer": build_short_answer(metrics, hyper_df, warn_df),
    }
    dump_json(SUMMARY_JSON, summary)

    print("Output directory: {}".format(OUTPUT_DIR), flush=True)
    print("Final feature matrix shape: {}".format(tuple(summary["final_feature_matrix_shape"])), flush=True)
    print("Metrics:", flush=True)
    print("{}: {}".format(MODEL_NAME, metrics), flush=True)
    print("Selected hyperparameter summary: {}".format(selected_summary.to_dict(orient="records")), flush=True)
    print("Warning summary: {}".format(warning_summary), flush=True)
    print("Comparison to Z1_Z2 + AMV/AMO K5: {}".format(summary["comparison_to_Z1_Z2_AMV_AMO_K5"]), flush=True)
    print("Short answer: {}".format(summary["short_answer"]), flush=True)


if __name__ == "__main__":
    main()
