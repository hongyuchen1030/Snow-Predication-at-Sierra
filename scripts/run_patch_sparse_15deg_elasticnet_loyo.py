#!/usr/bin/env python3
"""
Strict LOYO 15-degree patch-sparse ElasticNet for scalar April 1 Sierra SWE.
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

from scripts.direct_sst_sparse_common import (  # noqa: E402
    BASELINE_METRICS,
    COEF_EPS,
    MONTHS,
    WATER_YEARS,
    build_raw_sst_cube,
    compute_metric_bundle,
    compute_period_metrics,
    dump_json,
    ensure_runtime_on_compute_node,
    fit_elastic_net_standardized,
    load_target_table,
    patch_grid_for_size,
    prepare_patch_features,
    standardize_target_train_only,
    standardize_train_only,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "patch_sparse_15deg_elasticnet_loyo"
METADATA_JSON = OUTPUT_DIR / "patch_sparse_15deg_metadata.json"
PATCH_TABLE_CSV = OUTPUT_DIR / "patch_sparse_15deg_patch_feature_table.csv"
CHECKPOINT_CSV = OUTPUT_DIR / "patch_sparse_15deg_predictions_checkpoint.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "patch_sparse_15deg_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "patch_sparse_15deg_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "patch_sparse_15deg_period_metrics.csv"
HYPERPARAMS_CSV = OUTPUT_DIR / "patch_sparse_15deg_selected_hyperparams_by_fold.csv"
INNER_CV_CSV = OUTPUT_DIR / "patch_sparse_15deg_inner_cv_results.csv"
BETA_CSV = OUTPUT_DIR / "patch_sparse_15deg_beta_by_fold.csv"
SELECTED_PATCHES_CSV = OUTPUT_DIR / "patch_sparse_15deg_selected_patches_by_fold.csv"
SELECTION_FREQ_CSV = OUTPUT_DIR / "patch_sparse_15deg_selection_frequency.csv"
SUMMARY_JSON = OUTPUT_DIR / "patch_sparse_15deg_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "patch_sparse_15deg_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "patch_sparse_15deg_scatter.png"
ERROR_PNG = OUTPUT_DIR / "patch_sparse_15deg_error_by_year.png"
SELECTION_MAP_PNG = OUTPUT_DIR / "patch_sparse_15deg_selection_frequency_maps.png"
MEAN_WEIGHT_MAP_PNG = OUTPUT_DIR / "patch_sparse_15deg_mean_weight_maps.png"
COEF_FOLD_NC = OUTPUT_DIR / "patch_sparse_15deg_coefficients_by_fold.nc"
SELECTION_FREQ_NC = OUTPUT_DIR / "patch_sparse_15deg_selection_frequency.nc"

PATCH_SIZE_DEG = 15
MODEL_NAME = "PATCH_SPARSE_15DEG"
ALPHA_GRID = np.asarray([1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0, 10.0], dtype=np.float64)
L1_RATIO_GRID = np.asarray([0.3, 0.5, 0.7, 0.9, 0.95, 0.99], dtype=np.float64)
PATCH_MLP_15DEG_BASELINE = {"R2": -1.7442, "RMSE": 0.05253, "sign_accuracy": 0.5405}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def deterministic_seed(heldout_wy: int, inner_valid_wy: int, alpha: float, l1_ratio: float) -> int:
    return int(heldout_wy * 100000 + (inner_valid_wy + 1) * 100 + round(alpha * 1000) + round(l1_ratio * 100))


def build_global_patch_matrix(
    cube: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    grid = patch_grid_for_size(lat, lon, PATCH_SIZE_DEG)
    lat_ids = grid["lat_ids"]
    lon_ids = grid["lon_ids"]
    lat_count = int(grid["lat_count"])
    lon_count = int(grid["lon_count"])
    cos_weights = np.cos(np.deg2rad(lat))
    monthly_clim = np.mean(cube, axis=0)
    anomalies = cube - monthly_clim[None, :, :, :]

    feature_rows = []
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
                patch_vals = anomalies[:, month_idx, :, :][:, cell_mask]
                patch_mean = np.sum(patch_vals * group_weights[None, :], axis=1) / weight_sum
                cols.append(np.asarray(patch_mean, dtype=np.float64))
                feature_rows.append(
                    {
                        "patch_id": int(feature_index),
                        "month": month_name,
                        "month_index": int(month_idx),
                        "lat_patch": int(lat_patch),
                        "lon_patch": int(lon_patch),
                        "lat_min": float(-10.0 + PATCH_SIZE_DEG * lat_patch),
                        "lat_max": float(-10.0 + PATCH_SIZE_DEG * (lat_patch + 1)),
                        "lon_min": float(120.0 + PATCH_SIZE_DEG * lon_patch),
                        "lon_max": float(120.0 + PATCH_SIZE_DEG * (lon_patch + 1)),
                        "lat_center": float(grid["lat_centers"][lat_patch]),
                        "lon_center": float(grid["lon_centers"][lon_patch]),
                        "num_cells": int(np.sum(cell_mask)),
                    }
                )
                feature_index += 1

    matrix = np.column_stack(cols).astype(np.float64)
    feature_table = pd.DataFrame(feature_rows)
    nonfinite_mask = ~np.all(np.isfinite(matrix), axis=0)
    valid_nonfinite = ~nonfinite_mask
    variance = np.var(matrix[:, valid_nonfinite], axis=0, ddof=1) if np.any(valid_nonfinite) else np.asarray([], dtype=np.float64)
    zero_var_mask = np.zeros(matrix.shape[1], dtype=bool)
    zero_var_mask[valid_nonfinite] = variance <= 0.0
    final_mask = valid_nonfinite & ~zero_var_mask
    feature_table["is_nonfinite_removed"] = nonfinite_mask.astype(int)
    feature_table["is_zero_variance_removed"] = zero_var_mask.astype(int)
    feature_table["is_final_valid"] = final_mask.astype(int)
    return matrix[:, final_mask], feature_table, final_mask


def build_metadata(feature_table: pd.DataFrame, final_shape: tuple[int, int]) -> dict[str, object]:
    lat_bounds = (
        feature_table[["lat_patch", "lat_min", "lat_max"]]
        .drop_duplicates()
        .sort_values("lat_patch")
        .to_dict(orient="records")
    )
    lon_bounds = (
        feature_table[["lon_patch", "lon_min", "lon_max"]]
        .drop_duplicates()
        .sort_values("lon_patch")
        .to_dict(orient="records")
    )
    return {
        "original_patch_feature_count": int(feature_table.shape[0]),
        "valid_patch_feature_count": int(feature_table["is_final_valid"].sum()),
        "removed_nonfinite_count": int(feature_table["is_nonfinite_removed"].sum()),
        "removed_zero_variance_count": int(feature_table["is_zero_variance_removed"].sum()),
        "final_feature_matrix_shape": [int(final_shape[0]), int(final_shape[1])],
        "patch_size_deg": int(PATCH_SIZE_DEG),
        "months": MONTHS,
        "lat_patch_bounds": lat_bounds,
        "lon_patch_bounds": lon_bounds,
    }


def inner_loyo_scores(x_train_raw: np.ndarray, y_train_raw: np.ndarray, train_years: np.ndarray, outer_heldout_wy: int) -> tuple[pd.DataFrame, float, float]:
    rows = []
    best_alpha = None
    best_l1_ratio = None
    best_mse = None
    best_key = None
    for alpha in ALPHA_GRID.tolist():
        for l1_ratio in L1_RATIO_GRID.tolist():
            squared_errors = []
            selected_counts = []
            for inner_valid_wy in train_years.tolist():
                inner_valid_mask = train_years == inner_valid_wy
                inner_train_mask = ~inner_valid_mask
                x_inner_train_raw = x_train_raw[inner_train_mask]
                x_inner_valid_raw = x_train_raw[inner_valid_mask][0]
                y_inner_train_raw = y_train_raw[inner_train_mask]
                y_inner_valid_raw = float(y_train_raw[inner_valid_mask][0])
                train_std = np.std(x_inner_train_raw, axis=0, ddof=1)
                valid_mask = np.isfinite(train_std) & (train_std > 0.0)
                x_inner_train_clean = x_inner_train_raw[:, valid_mask]
                x_inner_valid_clean = x_inner_valid_raw[valid_mask]
                x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(x_inner_train_clean, x_inner_valid_clean)
                y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train_raw)
                model = fit_elastic_net_standardized(x_inner_train_std, y_inner_train_std, alpha, l1_ratio)
                pred_std = float(model.predict(x_inner_valid_std[None, :])[0])
                valid_std = float((y_inner_valid_raw - y_mean) / y_std)
                squared_errors.append((pred_std - valid_std) ** 2)
                selected_counts.append(int(np.sum(np.abs(np.asarray(model.coef_, dtype=np.float64)) > COEF_EPS)))
            mse = float(np.mean(squared_errors))
            mean_selected = float(np.mean(selected_counts))
            rows.append(
                {
                    "outer_heldout_wy": int(outer_heldout_wy),
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "inner_cv_mse": mse,
                    "mean_num_selected_patches_inner": mean_selected,
                }
            )
            tie_key = (float(alpha), float(l1_ratio))
            if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and tie_key < best_key):
                best_mse = mse
                best_alpha = float(alpha)
                best_l1_ratio = float(l1_ratio)
                best_key = tie_key
    return pd.DataFrame(rows), best_alpha, best_l1_ratio


def plot_observed_vs_predicted(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ordered = pred_df.sort_values("heldout_wy")
    ax.plot(ordered["heldout_wy"], ordered["obs_swe"], color="black", linewidth=2.2, label="Observed SWE anomaly")
    ax.plot(ordered["heldout_wy"], ordered["pred_swe"], color="#1f77b4", linewidth=1.8, label=MODEL_NAME)
    ax.set_title("Strict LOYO 15-degree patch-sparse ElasticNet")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(pred_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 6.3), constrained_layout=True)
    obs = pred_df["obs_swe"].to_numpy(dtype=np.float64)
    pred = pred_df["pred_swe"].to_numpy(dtype=np.float64)
    lo = float(min(np.min(obs), np.min(pred)))
    hi = float(max(np.max(obs), np.max(pred)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.scatter(obs, pred, color="#1f77b4", s=42, alpha=0.85)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.0)
    ax.text(
        0.03,
        0.97,
        "RMSE={:.3f}\nR2={:.3f}\nr={:.3f}".format(metrics["RMSE"], metrics["R2"], metrics["r"]),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_xlabel("Observed SWE anomaly")
    ax.set_ylabel("Predicted SWE anomaly")
    ax.set_title("Patch-sparse 15-degree ElasticNet")
    ax.grid(alpha=0.25)
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_year(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    ordered = pred_df.sort_values("heldout_wy")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(ordered["heldout_wy"], ordered["error_pred_minus_obs"], color="#d62728", alpha=0.85)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Prediction error")
    ax.set_title("Patch-sparse 15-degree ElasticNet prediction error by year")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(ERROR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patch_panel_map(arr: np.ndarray, feature_table: pd.DataFrame, out_path: Path, title: str, cmap: str) -> None:
    grid = patch_grid_for_size(
        feature_table["lat_center"].drop_duplicates().to_numpy(),
        feature_table["lon_center"].drop_duplicates().to_numpy(),
        PATCH_SIZE_DEG,
    )
    lat_centers = grid["lat_centers"]
    lon_centers = grid["lon_centers"]
    fig, axes = plt.subplots(1, len(MONTHS), figsize=(2.8 * len(MONTHS), 3.1), constrained_layout=True)
    for month_idx, month_name in enumerate(MONTHS):
        ax = axes[month_idx]
        month_arr = arr[month_idx]
        vmax = float(np.nanmax(np.abs(month_arr)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = ax.pcolormesh(
            np.arange(len(lon_centers) + 1),
            np.arange(len(lat_centers) + 1),
            month_arr,
            shading="auto",
            cmap=cmap,
            vmin=(-vmax if cmap == "coolwarm" else 0.0),
            vmax=vmax,
        )
        ax.set_title(month_name)
        ax.set_xlabel("lon patch")
        ax.set_ylabel("lat patch")
        fig.colorbar(mesh, ax=ax, shrink=0.72)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def oracle_overlap(selection_df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for oracle_name, month, lat_pt, lon_pt in [
        ("Z1", "Jan", -9.5, 133.5),
        ("Z2", "Oct", 0.5, 136.5),
    ]:
        subset = selection_df[
            (selection_df["month"] == month)
            & (selection_df["lat_min"] <= lat_pt)
            & (selection_df["lat_max"] > lat_pt)
            & (selection_df["lon_min"] <= lon_pt)
            & (selection_df["lon_max"] > lon_pt)
        ].copy()
        if subset.empty:
            rows.append(
                {
                    "oracle_name": oracle_name,
                    "month": month,
                    "lat": float(lat_pt),
                    "lon": float(lon_pt),
                    "found_patch": False,
                }
            )
            continue
        row = subset.iloc[0].to_dict()
        row.update({"oracle_name": oracle_name, "lat": float(lat_pt), "lon": float(lon_pt), "found_patch": True})
        rows.append(row)
    return rows


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()

    target_df = load_target_table()
    raw_cube, lat, lon, raw_metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(raw_metadata["valid_feature_mask"], dtype=bool)
    y_raw = target_df["obs_swe"].to_numpy(dtype=np.float64)
    years = target_df["water_year"].to_numpy(dtype=np.int32)

    global_matrix, feature_table, global_valid_mask = build_global_patch_matrix(raw_cube, lat, lon, valid_feature_mask)
    feature_table.to_csv(PATCH_TABLE_CSV, index=False)
    metadata = build_metadata(feature_table, global_matrix.shape)
    dump_json(METADATA_JSON, metadata)

    prediction_rows = []
    hyper_rows = []
    inner_rows = []
    beta_rows = []
    selected_rows = []

    patch_meta_template = None
    lat_count = None
    lon_count = None
    coef_by_fold = None
    select_by_fold = None

    for outer_idx, heldout_wy in enumerate(years.tolist()):
        test_mask = years == heldout_wy
        train_mask = ~test_mask
        x_train_full, x_test_full, patch_meta = prepare_patch_features(
            raw_cube[train_mask],
            raw_cube[test_mask][0],
            valid_feature_mask,
            lat,
            lon,
            PATCH_SIZE_DEG,
        )
        if patch_meta_template is None:
            patch_meta_template = patch_meta.copy().reset_index(drop=True)
            patch_meta_template["patch_id"] = np.arange(patch_meta_template.shape[0], dtype=int)
            patch_meta_template["lat_min"] = patch_meta_template["lat_patch"].astype(float) * PATCH_SIZE_DEG - 10.0
            patch_meta_template["lat_max"] = patch_meta_template["lat_min"] + PATCH_SIZE_DEG
            patch_meta_template["lon_min"] = patch_meta_template["lon_patch"].astype(float) * PATCH_SIZE_DEG + 120.0
            patch_meta_template["lon_max"] = patch_meta_template["lon_min"] + PATCH_SIZE_DEG
            grid = patch_grid_for_size(lat, lon, PATCH_SIZE_DEG)
            lat_count = len(grid["lat_centers"])
            lon_count = len(grid["lon_centers"])
            coef_by_fold = np.full((len(years), len(MONTHS), lat_count, lon_count), np.nan, dtype=np.float64)
            select_by_fold = np.zeros((len(years), len(MONTHS), lat_count, lon_count), dtype=np.int8)

        x_train_raw = x_train_full[:, global_valid_mask]
        x_test_raw = x_test_full[global_valid_mask]
        y_train_raw = y_raw[train_mask]
        y_test_raw = float(y_raw[test_mask][0])
        train_years = years[train_mask]

        score_df, best_alpha, best_l1_ratio = inner_loyo_scores(x_train_raw, y_train_raw, train_years, int(heldout_wy))
        best_mse = float(score_df[(np.isclose(score_df["alpha"], best_alpha)) & (np.isclose(score_df["l1_ratio"], best_l1_ratio))]["inner_cv_mse"].iloc[0])
        inner_rows.extend(score_df.to_dict(orient="records"))

        train_std = np.std(x_train_raw, axis=0, ddof=1)
        fold_valid_mask = np.isfinite(train_std) & (train_std > 0.0)
        x_train_clean = x_train_raw[:, fold_valid_mask]
        x_test_clean = x_test_raw[fold_valid_mask]
        x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_clean, x_test_clean)
        y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
        model = fit_elastic_net_standardized(x_train_std, y_train_std, best_alpha, best_l1_ratio)
        pred_std = float(model.predict(x_test_std[None, :])[0])
        pred_raw = float(y_mean + y_std * pred_std)
        coef_std_kept = np.asarray(model.coef_, dtype=np.float64)
        coef_raw_kept = (y_std / x_std) * coef_std_kept
        coef_std_full = np.zeros(x_train_raw.shape[1], dtype=np.float64)
        coef_raw_full = np.zeros(x_train_raw.shape[1], dtype=np.float64)
        coef_std_full[fold_valid_mask] = coef_std_kept
        coef_raw_full[fold_valid_mask] = coef_raw_kept
        selected = np.abs(coef_raw_full) > COEF_EPS
        num_selected = int(np.sum(selected))
        error = float(pred_raw - y_test_raw)

        prediction_rows.append(
            {
                "heldout_wy": int(heldout_wy),
                "obs_swe": y_test_raw,
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error,
                "residual_obs_minus_pred": -error,
                "abs_error": float(abs(error)),
                "sign_correct": float(np.sign(pred_raw) == np.sign(y_test_raw)) if pred_raw != 0.0 and y_test_raw != 0.0 else np.nan,
                "selected_alpha": float(best_alpha),
                "selected_l1_ratio": float(best_l1_ratio),
                "inner_cv_mse": best_mse,
                "num_selected_patches": num_selected,
            }
        )
        hyper_rows.append(
            {
                "heldout_wy": int(heldout_wy),
                "selected_alpha": float(best_alpha),
                "selected_l1_ratio": float(best_l1_ratio),
                "inner_cv_mse": best_mse,
                "num_selected_patches": num_selected,
            }
        )

        fold_selected_meta = patch_meta_template.copy()
        fold_selected_meta["coef_standardized"] = coef_std_full
        fold_selected_meta["coef_raw_or_physical"] = coef_raw_full
        fold_selected_meta["selected"] = selected.astype(int)
        fold_selected_meta["heldout_wy"] = int(heldout_wy)
        fold_selected_meta["selected_alpha"] = float(best_alpha)
        fold_selected_meta["selected_l1_ratio"] = float(best_l1_ratio)
        beta_rows.extend(
            fold_selected_meta[
                [
                    "heldout_wy",
                    "patch_id",
                    "month",
                    "lat_min",
                    "lat_max",
                    "lon_min",
                    "lon_max",
                    "coef_standardized",
                    "coef_raw_or_physical",
                    "selected",
                    "selected_alpha",
                    "selected_l1_ratio",
                ]
            ].to_dict(orient="records")
        )

        selected_only = fold_selected_meta[fold_selected_meta["selected"] == 1].copy()
        selected_only["abs_coef_rank"] = selected_only["coef_raw_or_physical"].abs().rank(method="first", ascending=False).astype(int)
        selected_rows.extend(
            selected_only[
                [
                    "heldout_wy",
                    "patch_id",
                    "month",
                    "lat_min",
                    "lat_max",
                    "lon_min",
                    "lon_max",
                    "coef_standardized",
                    "coef_raw_or_physical",
                    "abs_coef_rank",
                ]
            ].to_dict(orient="records")
        )

        patch_coef_grid = np.full((len(MONTHS), lat_count, lon_count), np.nan, dtype=np.float64)
        patch_sel_grid = np.zeros((len(MONTHS), lat_count, lon_count), dtype=np.int8)
        for feat_idx, row in patch_meta_template.iterrows():
            month_idx = int(row["month_index"])
            lat_patch = int(row["lat_patch"])
            lon_patch = int(row["lon_patch"])
            patch_coef_grid[month_idx, lat_patch, lon_patch] = coef_raw_full[feat_idx]
            patch_sel_grid[month_idx, lat_patch, lon_patch] = int(selected[feat_idx])
        coef_by_fold[outer_idx] = patch_coef_grid
        select_by_fold[outer_idx] = patch_sel_grid

        pd.DataFrame(prediction_rows).to_csv(CHECKPOINT_CSV, index=False)
        print(
            "LOYO heldout_WY={} model={} alpha={} l1_ratio={:.2f} obs={:.6f} pred={:.6f} selected={}".format(
                int(heldout_wy),
                MODEL_NAME,
                best_alpha,
                best_l1_ratio,
                y_test_raw,
                pred_raw,
                num_selected,
            ),
            flush=True,
        )

    pred_df = pd.DataFrame(prediction_rows).sort_values("heldout_wy").reset_index(drop=True)
    hyper_df = pd.DataFrame(hyper_rows).sort_values("heldout_wy").reset_index(drop=True)
    inner_df = pd.DataFrame(inner_rows).sort_values(["outer_heldout_wy", "alpha", "l1_ratio"]).reset_index(drop=True)
    beta_df = pd.DataFrame(beta_rows).sort_values(["heldout_wy", "patch_id"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["heldout_wy", "abs_coef_rank"]).reset_index(drop=True)

    metrics = compute_metric_bundle(pred_df["obs_swe"].to_numpy(dtype=np.float64), pred_df["pred_swe"].to_numpy(dtype=np.float64))
    metrics["mean_num_selected_patches"] = float(np.mean(pred_df["num_selected_patches"]))
    metrics["median_num_selected_patches"] = float(np.median(pred_df["num_selected_patches"]))
    metrics_df = pd.DataFrame([{ "model_name": MODEL_NAME, **metrics }])
    period_df = compute_period_metrics(pred_df.assign(model_name=MODEL_NAME), "model_name")

    selection_summary = (
        beta_df.groupby(["patch_id", "month", "lat_min", "lat_max", "lon_min", "lon_max"], as_index=False)
        .agg(
            selection_count=("selected", "sum"),
            mean_coef_standardized=("coef_standardized", "mean"),
            std_coef_standardized=("coef_standardized", "std"),
            mean_abs_coef_standardized=("coef_standardized", lambda x: float(np.mean(np.abs(x)))),
        )
    )
    selection_summary["selection_frequency"] = selection_summary["selection_count"] / float(len(WATER_YEARS))
    selection_summary = selection_summary.sort_values(["selection_frequency", "mean_abs_coef_standardized"], ascending=[False, False]).reset_index(drop=True)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMS_CSV, index=False)
    inner_df.to_csv(INNER_CV_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)
    selected_df.to_csv(SELECTED_PATCHES_CSV, index=False)
    selection_summary.to_csv(SELECTION_FREQ_CSV, index=False)

    coef_ds = xr.Dataset(
        data_vars={
            "coef_raw_or_physical": (("heldout_wy", "month", "lat_patch", "lon_patch"), coef_by_fold.astype(np.float32)),
            "selected_patch": (("heldout_wy", "month", "lat_patch", "lon_patch"), select_by_fold.astype(np.int8)),
        },
        coords={
            "heldout_wy": years.astype(np.int32),
            "month": np.asarray(MONTHS, dtype=object),
            "lat_patch": np.arange(lat_count, dtype=np.int32),
            "lon_patch": np.arange(lon_count, dtype=np.int32),
        },
    )
    coef_ds.to_netcdf(COEF_FOLD_NC)

    selection_freq_map = np.mean(select_by_fold, axis=0)
    mean_weight_map = np.nanmean(coef_by_fold, axis=0)
    std_weight_map = np.nanstd(coef_by_fold, axis=0, ddof=1)
    xr.Dataset(
        data_vars={
            "selection_frequency": (("month", "lat_patch", "lon_patch"), selection_freq_map.astype(np.float32)),
            "mean_coef_raw_or_physical": (("month", "lat_patch", "lon_patch"), mean_weight_map.astype(np.float32)),
            "std_coef_raw_or_physical": (("month", "lat_patch", "lon_patch"), std_weight_map.astype(np.float32)),
        },
        coords={
            "month": np.asarray(MONTHS, dtype=object),
            "lat_patch": np.arange(lat_count, dtype=np.int32),
            "lon_patch": np.arange(lon_count, dtype=np.int32),
        },
    ).to_netcdf(SELECTION_FREQ_NC)

    plot_observed_vs_predicted(pred_df)
    plot_scatter(pred_df, metrics)
    plot_error_by_year(pred_df)
    plot_patch_panel_map(selection_freq_map, patch_meta_template, SELECTION_MAP_PNG, "Patch selection frequency", "viridis")
    plot_patch_panel_map(mean_weight_map, patch_meta_template, MEAN_WEIGHT_MAP_PNG, "Mean coefficient by patch", "coolwarm")

    selected_patch_count_summary = {
        "mean_num_selected_patches": float(np.mean(pred_df["num_selected_patches"])),
        "median_num_selected_patches": float(np.median(pred_df["num_selected_patches"])),
        "max_num_selected_patches": int(np.max(pred_df["num_selected_patches"])),
        "min_num_selected_patches": int(np.min(pred_df["num_selected_patches"])),
    }
    top_selected_patches = selection_summary.head(20).to_dict(orient="records")
    oracle_rows = oracle_overlap(selection_summary)
    short_answer = (
        "Did the 15-degree patch-sparse ElasticNet finish? Yes. "
        "Did it beat the Z1_Z2 + AMV/AMO K5 baseline? {} "
        "Did it beat the failed 15-degree patch MLP? {} "
        "Were the Z1/Z2-containing patches selected frequently? {}.".format(
            "Yes." if metrics["R2"] >= BASELINE_METRICS["R2"] else "No.",
            "Yes." if metrics["R2"] > PATCH_MLP_15DEG_BASELINE["R2"] else "No.",
            "Yes" if any(row.get("selection_frequency", 0.0) >= 0.5 for row in oracle_rows if row.get("found_patch")) else "No",
        )
    )
    summary = {
        "input_files": {
            "target_table": str(PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"),
            "sst_file": "/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc",
        },
        "patch_size_deg": int(PATCH_SIZE_DEG),
        "final_feature_matrix_shape": metadata["final_feature_matrix_shape"],
        "hyperparameter_grid": {
            "alpha_grid": ALPHA_GRID.tolist(),
            "l1_ratio_grid": L1_RATIO_GRID.tolist(),
        },
        "metrics": metrics,
        "period_metrics": period_df.to_dict(orient="records"),
        "selected_hyperparams_by_fold": hyper_df.to_dict(orient="records"),
        "selected_patch_count_summary": selected_patch_count_summary,
        "top_selected_patches": top_selected_patches,
        "oracle_Z1_Z2_patch_overlap": oracle_rows,
        "comparison_to_Z1_Z2_AMV_AMO_K5": {
            "baseline": BASELINE_METRICS,
            "patch_sparse_15deg": {
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "sign_accuracy": metrics["sign_accuracy"],
            },
        },
        "comparison_to_patch_mlp_15deg": {
            "patch_mlp_15deg": PATCH_MLP_15DEG_BASELINE,
            "patch_sparse_15deg": {
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "sign_accuracy": metrics["sign_accuracy"],
            },
        },
        "short_answer": short_answer,
    }
    dump_json(SUMMARY_JSON, summary)

    hyper_summary = (
        hyper_df.groupby(["selected_alpha", "selected_l1_ratio"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "selected_alpha", "selected_l1_ratio"], ascending=[False, True, True])
    )
    print("Output directory: {}".format(OUTPUT_DIR), flush=True)
    print("Final feature matrix shape: {}".format(tuple(metadata["final_feature_matrix_shape"])), flush=True)
    print("Metrics:", flush=True)
    print("{}: {}".format(MODEL_NAME, metrics), flush=True)
    print("Selected hyperparameter summary: {}".format(hyper_summary.to_dict(orient="records")), flush=True)
    print("Selected patch count summary: {}".format(selected_patch_count_summary), flush=True)
    print("Oracle Z1/Z2 patch overlap: {}".format(oracle_rows), flush=True)
    print("Comparison to Z1_Z2 + AMV/AMO K5: {}".format(summary["comparison_to_Z1_Z2_AMV_AMO_K5"]), flush=True)
    print("Comparison to PATCH_MLP_15DEG: {}".format(summary["comparison_to_patch_mlp_15deg"]), flush=True)
    print("Short answer: {}".format(short_answer), flush=True)


if __name__ == "__main__":
    main()
