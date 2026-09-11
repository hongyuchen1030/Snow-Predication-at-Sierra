#!/usr/bin/env python3
"""
Strict LOYO patch-sparse direct-SST encoder baseline for Sierra SWE.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

from scripts.direct_sst_sparse_common import (
    BASELINE_METRICS,
    COEF_EPS,
    MONTHS,
    WATER_YEARS,
    build_raw_sst_cube,
    compute_metric_bundle,
    compute_period_metrics,
    dump_json,
    elastic_net_inner_loyo,
    ensure_runtime_on_compute_node,
    fit_elastic_net_standardized,
    load_target_table,
    patch_grid_for_size,
    prepare_patch_features,
    standardize_target_train_only,
    standardize_train_only,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "direct_sst_patch_sparse_encoder_loyo"
)
METADATA_JSON = OUTPUT_DIR / "direct_sst_patch_sparse_predictor_metadata.json"
PREDICTIONS_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_loyo_period_metrics.csv"
HYPERPARAM_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_selected_hyperparams_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_beta_by_fold.csv"
SELECTED_PATCHES_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_selected_patches_by_fold.csv"
SELECTION_FREQ_CSV = OUTPUT_DIR / "direct_sst_patch_sparse_selection_frequency.csv"
SUMMARY_JSON = OUTPUT_DIR / "direct_sst_patch_sparse_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "direct_sst_patch_sparse_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "direct_sst_patch_sparse_scatter_three_panel.png"
ERROR_PNG = OUTPUT_DIR / "direct_sst_patch_sparse_error_by_year.png"
SELECTION_MAP_PNG = OUTPUT_DIR / "direct_sst_patch_sparse_selection_frequency_maps.png"
MEAN_WEIGHT_MAP_PNG = OUTPUT_DIR / "direct_sst_patch_sparse_mean_weight_maps.png"
COEF_FOLD_NC = OUTPUT_DIR / "patch_sparse_coefficients_by_fold.nc"
SELECTION_FREQ_NC = OUTPUT_DIR / "patch_sparse_selection_frequency.nc"

PATCH_SIZES = [5, 10, 15]
MODEL_NAMES = {5: "PATCH_5DEG", 10: "PATCH_10DEG", 15: "PATCH_15DEG"}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_hyperparams(hyper_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for patch_size, model_name in MODEL_NAMES.items():
        selected = hyper_df[(hyper_df["model_name"] == model_name) & (hyper_df["is_selected"] == 1)].copy()
        alpha_counts = selected["selected_alpha"].value_counts().sort_index()
        l1_counts = selected["selected_l1_ratio"].value_counts().sort_index()
        summary[model_name] = {
            "selected_alpha_mode": float(alpha_counts.idxmax()),
            "selected_alpha_median": float(selected["selected_alpha"].median()),
            "selected_l1_ratio_mode": float(l1_counts.idxmax()),
            "selected_l1_ratio_median": float(selected["selected_l1_ratio"].median()),
        }
    return summary


def plot_observed_vs_predicted(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    base = pred_df[pred_df["model_name"] == MODEL_NAMES[5]].sort_values("heldout_wy")
    ax.plot(base["heldout_wy"], base["obs_swe"], color="black", linewidth=2.5, label="Observed SWE anomaly")
    colors = {5: "#0b3c5d", 10: "#328cc1", 15: "#d9b310"}
    for patch_size, model_name in MODEL_NAMES.items():
        subset = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(subset["heldout_wy"], subset["pred_swe"], linewidth=1.5, color=colors[patch_size], label=model_name)
    ax.set_title("Strict LOYO predictions using patch-sparse direct SST encoders")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_three_panel(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharex=True, sharey=True)
    colors = {5: "#0b3c5d", 10: "#328cc1", 15: "#d9b310"}
    for ax, patch_size in zip(axes, PATCH_SIZES):
        model_name = MODEL_NAMES[patch_size]
        subset = pred_df[pred_df["model_name"] == model_name]
        metrics = metrics_df[metrics_df["model_name"] == model_name].iloc[0]
        obs = subset["obs_swe"].to_numpy(dtype=float)
        pred = subset["pred_swe"].to_numpy(dtype=float)
        min_val = float(min(np.min(obs), np.min(pred)))
        max_val = float(max(np.max(obs), np.max(pred)))
        ax.scatter(obs, pred, color=colors[patch_size], alpha=0.85, s=38)
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(model_name)
        ax.set_xlabel("Observed SWE anomaly (m)")
        ax.text(
            0.03,
            0.97,
            "RMSE={:.3f}\nR2={:.3f}\nr={:.3f}".format(metrics["RMSE"], metrics["R2"], metrics["r"]),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted SWE anomaly (m)")
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_year(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    colors = {5: "#0b3c5d", 10: "#328cc1", 15: "#d9b310"}
    for patch_size in PATCH_SIZES:
        subset = pred_df[pred_df["model_name"] == MODEL_NAMES[patch_size]].sort_values("heldout_wy")
        ax.plot(
            subset["heldout_wy"],
            subset["error_pred_minus_obs"],
            color=colors[patch_size],
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            label=MODEL_NAMES[patch_size],
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Prediction error by year: patch-sparse direct SST encoders")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Prediction error (pred - obs) (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(ERROR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patch_panel_maps(
    arrays_by_patch: Dict[int, np.ndarray],
    patch_meta: Dict[int, pd.DataFrame],
    out_path: Path,
    title_prefix: str,
    cmap: str,
) -> None:
    fig, axes = plt.subplots(len(PATCH_SIZES), len(MONTHS), figsize=(2.7 * len(MONTHS), 2.6 * len(PATCH_SIZES)), constrained_layout=True)
    for row_idx, patch_size in enumerate(PATCH_SIZES):
        arr = arrays_by_patch[patch_size]
        meta = patch_meta[patch_size]
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
            ax.set_title(f"{patch_size}deg {month_name}")
            ax.set_xlabel("lon patch")
            ax.set_ylabel("lat patch")
            fig.colorbar(mesh, ax=ax, shrink=0.72)
    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()

    target_df = load_target_table()
    raw_cube, lat, lon, metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(metadata["valid_feature_mask"], dtype=bool)
    y_raw = target_df["obs_swe"].to_numpy(dtype=float)

    metadata_payload = {
        "input_sst_file": str(Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")),
        "target_file": str(PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"),
        "domain": metadata["domain"],
        "months": metadata["months"],
        "water_years": WATER_YEARS.astype(int).tolist(),
        "original_feature_shape": metadata["feature_shape_before_mask"],
        "original_feature_count": metadata["original_feature_count"],
        "valid_feature_count": metadata["num_valid_features"],
        "removed_feature_count": metadata["removed_feature_count"],
        "final_matrix_shape": metadata["final_matrix_shape"],
        "patch_sizes_deg": PATCH_SIZES,
    }
    dump_json(METADATA_JSON, metadata_payload)

    prediction_rows: List[Dict[str, object]] = []
    hyper_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    metrics_rows: List[Dict[str, object]] = []
    patch_meta_by_size: Dict[int, pd.DataFrame] = {}
    selection_freq_maps: Dict[int, np.ndarray] = {}
    mean_coef_maps: Dict[int, np.ndarray] = {}
    coef_arrays_padded: Dict[int, np.ndarray] = {}
    selected_arrays_padded: Dict[int, np.ndarray] = {}

    max_lat_patch = 0
    max_lon_patch = 0
    for patch_size in PATCH_SIZES:
        grid_info = patch_grid_for_size(lat, lon, patch_size)
        max_lat_patch = max(max_lat_patch, len(grid_info["lat_centers"]))
        max_lon_patch = max(max_lon_patch, len(grid_info["lon_centers"]))

    for patch_size in PATCH_SIZES:
        model_name = MODEL_NAMES[patch_size]
        years = WATER_YEARS.copy()
        preds = np.full(years.size, np.nan, dtype=float)
        num_selected = np.full(years.size, np.nan, dtype=float)

        grid_info = patch_grid_for_size(lat, lon, patch_size)
        lat_count = len(grid_info["lat_centers"])
        lon_count = len(grid_info["lon_centers"])
        coef_by_fold = np.full((years.size, len(MONTHS), lat_count, lon_count), np.nan, dtype=float)
        selected_by_fold = np.zeros((years.size, len(MONTHS), lat_count, lon_count), dtype=np.int8)
        patch_meta_template: Optional[pd.DataFrame] = None

        for outer_idx, water_year in enumerate(years):
            mask = np.ones(years.size, dtype=bool)
            mask[outer_idx] = False
            x_train_raw, x_test_raw, patch_meta = prepare_patch_features(
                raw_cube[mask],
                raw_cube[~mask][0],
                valid_feature_mask,
                lat,
                lon,
                patch_size,
            )
            if patch_meta_template is None:
                patch_meta_template = patch_meta.copy()
            y_train = y_raw[mask]
            y_test = float(y_raw[~mask][0])
            best_alpha, best_l1_ratio, score_rows = elastic_net_inner_loyo(x_train_raw, y_train)
            x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
            y_train_std, y_mean, y_std = standardize_target_train_only(y_train)
            model = fit_elastic_net_standardized(x_train_std, y_train_std, best_alpha, best_l1_ratio)
            pred_std = float(model.predict(x_test_std[None, :])[0])
            pred_raw = float(y_mean + y_std * pred_std)
            coef_std = np.asarray(model.coef_, dtype=float)
            coef_patch = (y_std / x_std) * coef_std
            intercept_raw = float(y_mean + y_std * model.intercept_ - np.sum(coef_patch * x_mean))
            selected = np.abs(coef_patch) > COEF_EPS

            preds[outer_idx] = pred_raw
            num_selected[outer_idx] = int(np.sum(selected))
            error = pred_raw - y_test
            prediction_rows.append(
                {
                    "model_name": model_name,
                    "patch_size_deg": int(patch_size),
                    "heldout_wy": int(water_year),
                    "obs_swe": y_test,
                    "pred_swe": pred_raw,
                    "error_pred_minus_obs": error,
                    "residual_obs_minus_pred": -error,
                    "abs_error": abs(error),
                    "sign_correct": float((np.sign(pred_raw) == np.sign(y_test)) and (pred_raw != 0.0) and (y_test != 0.0)),
                    "selected_alpha": float(best_alpha),
                    "selected_l1_ratio": float(best_l1_ratio),
                    "num_selected_patches": int(np.sum(selected)),
                }
            )
            for row in score_rows:
                hyper_rows.append(
                    {
                        "model_name": model_name,
                        "patch_size_deg": int(patch_size),
                        "heldout_wy": int(water_year),
                        "alpha": float(row["alpha"]),
                        "l1_ratio": float(row["l1_ratio"]),
                        "inner_cv_mse": float(row["mse"]),
                        "selected_alpha": float(best_alpha),
                        "selected_l1_ratio": float(best_l1_ratio),
                        "is_selected": int(np.isclose(row["alpha"], best_alpha) and np.isclose(row["l1_ratio"], best_l1_ratio)),
                    }
                )

            beta_rows.append(
                {
                    "model_name": model_name,
                    "patch_size_deg": int(patch_size),
                    "heldout_wy": int(water_year),
                    "intercept": intercept_raw,
                    "selected_alpha": float(best_alpha),
                    "selected_l1_ratio": float(best_l1_ratio),
                    "num_selected_patches": int(np.sum(selected)),
                }
            )
            patch_grid = np.full((len(MONTHS), lat_count, lon_count), np.nan, dtype=float)
            patch_selected = np.zeros((len(MONTHS), lat_count, lon_count), dtype=np.int8)
            for feat_idx, row in patch_meta.iterrows():
                patch_grid[int(row["month_index"]), int(row["lat_patch"]), int(row["lon_patch"])] = coef_patch[feat_idx]
                patch_selected[int(row["month_index"]), int(row["lat_patch"]), int(row["lon_patch"])] = int(selected[feat_idx])
                if selected[feat_idx]:
                    selected_rows.append(
                        {
                            "model_name": model_name,
                            "patch_size_deg": int(patch_size),
                            "heldout_wy": int(water_year),
                            "month": row["month"],
                            "lat_patch": int(row["lat_patch"]),
                            "lon_patch": int(row["lon_patch"]),
                            "lat_center": float(row["lat_center"]),
                            "lon_center": float(row["lon_center"]),
                            "patch_coef": float(coef_patch[feat_idx]),
                            "selected_alpha": float(best_alpha),
                            "selected_l1_ratio": float(best_l1_ratio),
                        }
                    )
            coef_by_fold[outer_idx] = patch_grid
            selected_by_fold[outer_idx] = patch_selected
            print(
                f"LOYO heldout_WY={int(water_year)} model={model_name} alpha={best_alpha:g} l1_ratio={best_l1_ratio:.2f} "
                f"obs={y_test:.6f} pred={pred_raw:.6f} selected={int(np.sum(selected))}",
                flush=True,
            )

        assert patch_meta_template is not None
        patch_meta_by_size[patch_size] = patch_meta_template
        selection_freq_maps[patch_size] = np.mean(selected_by_fold, axis=0)
        mean_coef_maps[patch_size] = np.nanmean(coef_by_fold, axis=0)
        padded_coef = np.full((years.size, len(MONTHS), max_lat_patch, max_lon_patch), np.nan, dtype=float)
        padded_sel = np.full((years.size, len(MONTHS), max_lat_patch, max_lon_patch), -1, dtype=np.int8)
        padded_coef[:, :, :lat_count, :lon_count] = coef_by_fold
        padded_sel[:, :, :lat_count, :lon_count] = selected_by_fold
        coef_arrays_padded[patch_size] = padded_coef
        selected_arrays_padded[patch_size] = padded_sel

        metrics = compute_metric_bundle(y_raw, preds)
        metrics.update(
            {
                "model_name": model_name,
                "patch_size_deg": int(patch_size),
                "mean_num_selected_patches": float(np.mean(num_selected)),
                "median_num_selected_patches": float(np.median(num_selected)),
            }
        )
        metrics_rows.append(metrics)

    pred_df = pd.DataFrame(prediction_rows)
    hyper_df = pd.DataFrame(hyper_rows)
    beta_df = pd.DataFrame(beta_rows)
    selected_df = pd.DataFrame(selected_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    period_df = compute_period_metrics(pred_df, "model_name")

    freq_rows: List[Dict[str, object]] = []
    for patch_size in PATCH_SIZES:
        model_name = MODEL_NAMES[patch_size]
        sub = selected_df[selected_df["model_name"] == model_name].copy()
        counts = (
            sub.groupby(["month", "lat_patch", "lon_patch", "lat_center", "lon_center"], as_index=False)
            .size()
            .rename(columns={"size": "selection_count"})
            if not sub.empty
            else pd.DataFrame(columns=["month", "lat_patch", "lon_patch", "lat_center", "lon_center", "selection_count"])
        )
        template = patch_meta_by_size[patch_size][["month", "lat_patch", "lon_patch", "lat_center", "lon_center"]].drop_duplicates()
        freq_df = template.merge(counts, on=["month", "lat_patch", "lon_patch", "lat_center", "lon_center"], how="left")
        freq_df["selection_count"] = freq_df["selection_count"].fillna(0).astype(int)
        freq_df["selection_frequency"] = freq_df["selection_count"] / float(len(WATER_YEARS))
        freq_df["model_name"] = model_name
        freq_df["patch_size_deg"] = int(patch_size)
        freq_rows.extend(freq_df.to_dict(orient="records"))
    freq_df = pd.DataFrame(freq_rows)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAM_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)
    selected_df.to_csv(SELECTED_PATCHES_CSV, index=False)
    freq_df.to_csv(SELECTION_FREQ_CSV, index=False)

    coef_stack = np.stack([coef_arrays_padded[patch_size] for patch_size in PATCH_SIZES], axis=0)
    sel_stack = np.stack([selected_arrays_padded[patch_size] for patch_size in PATCH_SIZES], axis=0)
    mean_stack = np.full((len(PATCH_SIZES), len(MONTHS), max_lat_patch, max_lon_patch), np.nan, dtype=float)
    std_stack = np.full_like(mean_stack, np.nan)
    freq_stack = np.full_like(mean_stack, np.nan)
    lat_center_stack = np.full((len(PATCH_SIZES), max_lat_patch), np.nan, dtype=float)
    lon_center_stack = np.full((len(PATCH_SIZES), max_lon_patch), np.nan, dtype=float)
    for idx, patch_size in enumerate(PATCH_SIZES):
        arr = mean_coef_maps[patch_size]
        freq_arr = selection_freq_maps[patch_size]
        mean_stack[idx, :, :arr.shape[1], :arr.shape[2]] = arr
        std_stack[idx, :, :arr.shape[1], :arr.shape[2]] = np.nanstd(coef_arrays_padded[patch_size][:, :, :arr.shape[1], :arr.shape[2]], axis=0, ddof=1)
        freq_stack[idx, :, :freq_arr.shape[1], :freq_arr.shape[2]] = freq_arr
        grid = patch_grid_for_size(lat, lon, patch_size)
        lat_center_stack[idx, : len(grid["lat_centers"])] = grid["lat_centers"]
        lon_center_stack[idx, : len(grid["lon_centers"])] = grid["lon_centers"]

    coef_ds = xr.Dataset(
        data_vars={
            "patch_coef": (("patch_size", "heldout_wy", "month", "lat_patch", "lon_patch"), coef_stack.astype(np.float32)),
            "selected_patch_mask": (("patch_size", "heldout_wy", "month", "lat_patch", "lon_patch"), sel_stack.astype(np.int8)),
        },
        coords={
            "patch_size": np.asarray(PATCH_SIZES, dtype=np.int32),
            "heldout_wy": WATER_YEARS.astype(np.int32),
            "month": np.asarray(MONTHS, dtype=object),
            "lat_patch": np.arange(max_lat_patch, dtype=np.int32),
            "lon_patch": np.arange(max_lon_patch, dtype=np.int32),
        },
        attrs={"description": "Foldwise patch-sparse ElasticNet coefficients for direct SST encoders."},
    )
    coef_ds["lat_patch_center"] = (("patch_size", "lat_patch"), lat_center_stack.astype(np.float32))
    coef_ds["lon_patch_center"] = (("patch_size", "lon_patch"), lon_center_stack.astype(np.float32))
    coef_ds.to_netcdf(COEF_FOLD_NC)

    sel_ds = xr.Dataset(
        data_vars={
            "selection_frequency": (("patch_size", "month", "lat_patch", "lon_patch"), freq_stack.astype(np.float32)),
            "mean_patch_coef": (("patch_size", "month", "lat_patch", "lon_patch"), mean_stack.astype(np.float32)),
            "std_patch_coef": (("patch_size", "month", "lat_patch", "lon_patch"), std_stack.astype(np.float32)),
        },
        coords={
            "patch_size": np.asarray(PATCH_SIZES, dtype=np.int32),
            "month": np.asarray(MONTHS, dtype=object),
            "lat_patch": np.arange(max_lat_patch, dtype=np.int32),
            "lon_patch": np.arange(max_lon_patch, dtype=np.int32),
        },
        attrs={"description": "Patch-sparse direct SST selection frequency and mean coefficient maps."},
    )
    sel_ds["lat_patch_center"] = (("patch_size", "lat_patch"), lat_center_stack.astype(np.float32))
    sel_ds["lon_patch_center"] = (("patch_size", "lon_patch"), lon_center_stack.astype(np.float32))
    sel_ds.to_netcdf(SELECTION_FREQ_NC)

    plot_observed_vs_predicted(pred_df)
    plot_scatter_three_panel(pred_df, metrics_df)
    plot_error_by_year(pred_df)
    plot_patch_panel_maps(selection_freq_maps, patch_meta_by_size, SELECTION_MAP_PNG, "Patch selection frequency", "viridis")
    plot_patch_panel_maps(mean_coef_maps, patch_meta_by_size, MEAN_WEIGHT_MAP_PNG, "Mean patch coefficient", "coolwarm")

    best_r2_row = metrics_df.sort_values("R2", ascending=False).iloc[0]
    best_rmse_row = metrics_df.sort_values("RMSE", ascending=True).iloc[0]
    top_patches = freq_df.sort_values("selection_frequency", ascending=False).head(20)
    oracle_overlap = freq_df[
        ((freq_df["month"] == "Jan") & np.abs(freq_df["lat_center"] - (-9.5)) <= 7.5 & np.abs(freq_df["lon_center"] - 133.5) <= 7.5)
        | ((freq_df["month"] == "Oct") & np.abs(freq_df["lat_center"] - 0.5) <= 7.5 & np.abs(freq_df["lon_center"] - 136.5) <= 7.5)
    ].copy()
    hyper_summary = summarize_hyperparams(hyper_df)
    short_answer = (
        "Patch-sparse direct SST does not provide useful strict LOYO predictability and does not beat the current baseline."
        if float(best_r2_row["R2"]) < 0.2
        else "Patch-sparse direct SST shows some strict LOYO signal but still does not beat the current baseline."
    )
    summary = {
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_df.to_dict(orient="records"),
        "best_patch_size_by_R2": best_r2_row.to_dict(),
        "best_patch_size_by_RMSE": best_rmse_row.to_dict(),
        "selected_hyperparameter_summary": hyper_summary,
        "top_selected_patches": top_patches.to_dict(orient="records"),
        "oracle_overlap": oracle_overlap.to_dict(orient="records"),
        "comparison_to_current_baseline": {
            "baseline_name": "Z1_Z2_AMV_AMO_K5",
            **BASELINE_METRICS,
        },
        "short_answer": short_answer,
    }
    dump_json(SUMMARY_JSON, summary)

    print(f"Patch sparse output directory: {OUTPUT_DIR}", flush=True)
    print("Metrics:", flush=True)
    for row in metrics_df.itertuples(index=False):
        print(
            f"{row.model_name}: R2={row.R2:.6f} RMSE={row.RMSE:.6f} sign_accuracy={row.sign_accuracy:.6f}",
            flush=True,
        )
    print(f"Best patch size by R2: {best_r2_row['model_name']}", flush=True)
    print(f"Best patch size by RMSE: {best_rmse_row['model_name']}", flush=True)
    print(
        "Mean/median selected patches: "
        + "; ".join(
            [
                "{} {:.2f}/{:.2f}".format(row.model_name, row.mean_num_selected_patches, row.median_num_selected_patches)
                for row in metrics_df.itertuples(index=False)
            ]
        ),
        flush=True,
    )
    print(
        "Top selected patches: "
        + "; ".join(
            [
                "{} {} lat_patch={} lon_patch={} freq={:.3f}".format(row.patch_size_deg, row.month, row.lat_patch, row.lon_patch, row.selection_frequency)
                for row in top_patches.head(5).itertuples(index=False)
            ]
        ),
        flush=True,
    )
    print(
        "Comparison to Z1_Z2 + AMV/AMO K5: "
        f"baseline_R2={BASELINE_METRICS['R2']:.6f} baseline_RMSE={BASELINE_METRICS['RMSE']:.6f} "
        f"baseline_sign_accuracy={BASELINE_METRICS['sign_accuracy']:.6f}",
        flush=True,
    )
    print(f"Short answer: {short_answer}", flush=True)


if __name__ == "__main__":
    main()
