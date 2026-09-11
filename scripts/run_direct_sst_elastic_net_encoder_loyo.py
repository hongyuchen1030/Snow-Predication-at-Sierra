#!/usr/bin/env python3
"""
Strict LOYO elastic-net direct-SST encoder baseline for Sierra SWE.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

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
    ALPHA_GRID,
    BASELINE_METRICS,
    COEF_EPS,
    L1_RATIO_GRID,
    MONTHS,
    WATER_YEARS,
    build_raw_sst_cube,
    compute_metric_bundle,
    compute_period_metrics,
    dump_json,
    elastic_net_inner_loyo,
    ensure_runtime_on_compute_node,
    feature_records_dataframe,
    fit_elastic_net_standardized,
    prepare_weighted_features,
    standardize_target_train_only,
    standardize_train_only,
    load_target_table,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "direct_sst_elastic_net_encoder_loyo"
)
METADATA_JSON = OUTPUT_DIR / "direct_sst_elastic_net_predictor_metadata.json"
PREDICTIONS_CSV = OUTPUT_DIR / "direct_sst_elastic_net_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "direct_sst_elastic_net_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "direct_sst_elastic_net_loyo_period_metrics.csv"
HYPERPARAM_CSV = OUTPUT_DIR / "direct_sst_elastic_net_selected_hyperparams_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "direct_sst_elastic_net_beta_by_fold.csv"
SELECTED_CSV = OUTPUT_DIR / "direct_sst_elastic_net_selected_features_by_fold.csv"
SELECTION_FREQ_CSV = OUTPUT_DIR / "direct_sst_elastic_net_selection_frequency.csv"
SUMMARY_JSON = OUTPUT_DIR / "direct_sst_elastic_net_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "direct_sst_elastic_net_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "direct_sst_elastic_net_scatter.png"
ERROR_PNG = OUTPUT_DIR / "direct_sst_elastic_net_error_by_year.png"
SELECTION_MAP_PNG = OUTPUT_DIR / "direct_sst_elastic_net_selection_frequency_maps.png"
MEAN_WEIGHT_MAP_PNG = OUTPUT_DIR / "direct_sst_elastic_net_mean_weight_maps.png"
COEF_FOLD_NC = OUTPUT_DIR / "elastic_net_coefficients_by_fold.nc"
SELECTION_FREQ_NC = OUTPUT_DIR / "elastic_net_selection_frequency.nc"

MODEL_NAME = "DIRECT_SST_ELASTIC_NET"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_hyperparams(hyper_df: pd.DataFrame) -> Dict[str, object]:
    selected = hyper_df[hyper_df["is_selected"] == 1].copy()
    alpha_counts = selected["selected_alpha"].value_counts().sort_index()
    l1_counts = selected["selected_l1_ratio"].value_counts().sort_index()
    return {
        "selected_alpha_mode": float(alpha_counts.idxmax()),
        "selected_alpha_median": float(selected["selected_alpha"].median()),
        "selected_l1_ratio_mode": float(l1_counts.idxmax()),
        "selected_l1_ratio_median": float(selected["selected_l1_ratio"].median()),
        "alpha_counts": {str(float(idx)): int(val) for idx, val in alpha_counts.items()},
        "l1_ratio_counts": {str(float(idx)): int(val) for idx, val in l1_counts.items()},
    }


def plot_observed_vs_predicted(pred_df: pd.DataFrame) -> None:
    pred_df = pred_df.sort_values("heldout_wy")
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(pred_df["heldout_wy"], pred_df["obs_swe"], color="black", linewidth=2.5, label="Observed SWE anomaly")
    ax.plot(pred_df["heldout_wy"], pred_df["pred_swe"], color="#0b3c5d", linewidth=1.6, label="ElasticNet LOYO")
    ax.set_title("Strict LOYO predictions using direct SST ElasticNet encoder")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(OBS_PRED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(pred_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    obs = pred_df["obs_swe"].to_numpy(dtype=float)
    pred = pred_df["pred_swe"].to_numpy(dtype=float)
    min_val = float(min(np.min(obs), np.min(pred)))
    max_val = float(max(np.max(obs), np.max(pred)))
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    ax.scatter(obs, pred, color="#0b3c5d", alpha=0.85, s=40)
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Observed SWE anomaly (m)")
    ax.set_ylabel("Predicted SWE anomaly (m)")
    ax.set_title("Direct SST ElasticNet")
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
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_year(pred_df: pd.DataFrame) -> None:
    pred_df = pred_df.sort_values("heldout_wy")
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(pred_df["heldout_wy"], pred_df["error_pred_minus_obs"], color="#0b3c5d", linewidth=1.4, marker="o", markersize=3.5)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Prediction error by year: direct SST ElasticNet")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Prediction error (pred - obs) (m)")
    ax.grid(alpha=0.25)
    fig.savefig(ERROR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_month_maps(
    values_3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title_prefix: str,
    out_path: Path,
    cmap: str,
) -> None:
    fig, axes = plt.subplots(len(MONTHS), 1, figsize=(10, 2.4 * len(MONTHS)), constrained_layout=True)
    for idx, month_name in enumerate(MONTHS):
        ax = axes[idx]
        arr = values_3d[idx]
        vmax = float(np.nanmax(np.abs(arr)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        mesh = ax.pcolormesh(lon, lat, arr, cmap=cmap, shading="auto", vmin=(-vmax if cmap == "coolwarm" else 0.0), vmax=vmax)
        ax.set_ylabel(month_name)
        ax.set_xlabel("Longitude (E)")
        ax.set_title(f"{title_prefix} {month_name}")
        fig.colorbar(mesh, ax=ax, shrink=0.84)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()

    target_df = load_target_table()
    raw_cube, lat, lon, metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(metadata["valid_feature_mask"], dtype=bool)
    y_raw = target_df["obs_swe"].to_numpy(dtype=float)
    feature_df = feature_records_dataframe(metadata)
    feature_df.to_json(METADATA_JSON, orient="records", indent=2)
    dump_json(
        METADATA_JSON,
        {
            "input_sst_file": str(Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")),
            "target_file": str(target_df.columns.tolist()),
            "domain": metadata["domain"],
            "months": metadata["months"],
            "water_years": WATER_YEARS.astype(int).tolist(),
            "original_feature_shape": metadata["feature_shape_before_mask"],
            "original_feature_count": metadata["original_feature_count"],
            "valid_feature_count": metadata["num_valid_features"],
            "removed_feature_count": metadata["removed_feature_count"],
            "final_matrix_shape": metadata["final_matrix_shape"],
            "area_weighting": metadata["area_weighting"],
            "feature_metadata": feature_df.to_dict(orient="records"),
        },
    )

    years = WATER_YEARS.copy()
    prediction_rows: List[Dict[str, object]] = []
    hyper_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    preds = np.full(years.size, np.nan, dtype=float)
    num_selected = np.full(years.size, np.nan, dtype=float)
    coef_weighted_by_fold = np.full((years.size, len(MONTHS), lat.size, lon.size), np.nan, dtype=float)
    coef_physical_by_fold = np.full_like(coef_weighted_by_fold, np.nan)
    selected_mask_by_fold = np.zeros((years.size, len(MONTHS), lat.size, lon.size), dtype=np.int8)
    lat_sqrt_cos = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    valid_flat = valid_feature_mask.reshape(-1)
    feature_lat_sqrt_cos = np.tile(np.repeat(lat_sqrt_cos, lon.size), len(MONTHS))[valid_flat]

    for outer_idx, water_year in enumerate(years):
        mask = np.ones(years.size, dtype=bool)
        mask[outer_idx] = False
        x_train_raw, x_test_raw = prepare_weighted_features(raw_cube[mask], raw_cube[~mask][0], valid_feature_mask, lat, lon)
        y_train = y_raw[mask]
        y_test = float(y_raw[~mask][0])

        best_alpha, best_l1_ratio, score_rows = elastic_net_inner_loyo(x_train_raw, y_train)
        x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
        y_train_std, y_mean, y_std = standardize_target_train_only(y_train)
        model = fit_elastic_net_standardized(x_train_std, y_train_std, best_alpha, best_l1_ratio)
        pred_std = float(model.predict(x_test_std[None, :])[0])
        pred_raw = float(y_mean + y_std * pred_std)
        coef_std = np.asarray(model.coef_, dtype=float)
        coef_weighted = (y_std / x_std) * coef_std
        intercept_raw = float(y_mean + y_std * model.intercept_ - np.sum(coef_weighted * x_mean))
        coef_physical = coef_weighted * feature_lat_sqrt_cos
        selected = np.abs(coef_weighted) > COEF_EPS

        preds[outer_idx] = pred_raw
        num_selected[outer_idx] = int(np.sum(selected))
        error = pred_raw - y_test
        prediction_rows.append(
            {
                "model_name": MODEL_NAME,
                "heldout_wy": int(water_year),
                "obs_swe": y_test,
                "pred_swe": pred_raw,
                "error_pred_minus_obs": error,
                "residual_obs_minus_pred": -error,
                "abs_error": abs(error),
                "sign_correct": float((np.sign(pred_raw) == np.sign(y_test)) and (pred_raw != 0.0) and (y_test != 0.0)),
                "selected_alpha": float(best_alpha),
                "selected_l1_ratio": float(best_l1_ratio),
                "num_selected_features": int(np.sum(selected)),
            }
        )
        for row in score_rows:
            hyper_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "heldout_wy": int(water_year),
                    "alpha": float(row["alpha"]),
                    "l1_ratio": float(row["l1_ratio"]),
                    "inner_cv_mse": float(row["mse"]),
                    "selected_alpha": float(best_alpha),
                    "selected_l1_ratio": float(best_l1_ratio),
                    "is_selected": int(np.isclose(row["alpha"], best_alpha) and np.isclose(row["l1_ratio"], best_l1_ratio)),
                }
            )

        weighted_grid = np.full(valid_flat.size, np.nan, dtype=float)
        weighted_grid[valid_flat] = coef_weighted
        physical_grid = np.full(valid_flat.size, np.nan, dtype=float)
        physical_grid[valid_flat] = coef_physical
        selected_grid = np.zeros(valid_flat.size, dtype=np.int8)
        selected_grid[valid_flat] = selected.astype(np.int8)
        coef_weighted_by_fold[outer_idx] = weighted_grid.reshape(len(MONTHS), lat.size, lon.size)
        coef_physical_by_fold[outer_idx] = physical_grid.reshape(len(MONTHS), lat.size, lon.size)
        selected_mask_by_fold[outer_idx] = selected_grid.reshape(len(MONTHS), lat.size, lon.size)

        beta_rows.append(
            {
                "model_name": MODEL_NAME,
                "heldout_wy": int(water_year),
                "intercept": intercept_raw,
                "selected_alpha": float(best_alpha),
                "selected_l1_ratio": float(best_l1_ratio),
                "num_selected_features": int(np.sum(selected)),
            }
        )
        selected_idx = np.flatnonzero(selected)
        valid_records = feature_df[feature_df["is_valid"] == 1].reset_index(drop=True)
        for idx in selected_idx.tolist():
            row = valid_records.iloc[idx]
            selected_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "heldout_wy": int(water_year),
                    "month": row["month"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "flat_feature_index": int(row["flat_feature_index"]),
                    "coef_weighted_space": float(coef_weighted[idx]),
                    "coef_physical_space": float(coef_physical[idx]),
                    "selected_alpha": float(best_alpha),
                    "selected_l1_ratio": float(best_l1_ratio),
                }
            )
        print(
            f"LOYO heldout_WY={int(water_year)} model={MODEL_NAME} alpha={best_alpha:g} l1_ratio={best_l1_ratio:.2f} "
            f"obs={y_test:.6f} pred={pred_raw:.6f} selected={int(np.sum(selected))}",
            flush=True,
        )

    pred_df = pd.DataFrame(prediction_rows)
    hyper_df = pd.DataFrame(hyper_rows)
    beta_df = pd.DataFrame(beta_rows)
    selected_df = pd.DataFrame(selected_rows)

    metrics = compute_metric_bundle(y_raw, preds)
    metrics.update(
        {
            "model_name": MODEL_NAME,
            "mean_num_selected_features": float(np.mean(num_selected)),
            "median_num_selected_features": float(np.median(num_selected)),
        }
    )
    metrics_df = pd.DataFrame([metrics])
    period_df = compute_period_metrics(pred_df, "model_name")

    valid_records = feature_df.copy()
    freq_counts = (
        selected_df.groupby(["month", "lat", "lon", "flat_feature_index"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
        if not selected_df.empty
        else pd.DataFrame(columns=["month", "lat", "lon", "flat_feature_index", "selection_count"])
    )
    selection_freq_df = valid_records.merge(
        freq_counts,
        on=["month", "lat", "lon", "flat_feature_index"],
        how="left",
    )
    selection_freq_df["selection_count"] = selection_freq_df["selection_count"].fillna(0).astype(int)
    selection_freq_df = selection_freq_df[selection_freq_df["is_valid"] == 1].copy()
    selection_freq_df["selection_frequency"] = selection_freq_df["selection_count"] / float(len(WATER_YEARS))
    mean_phys = np.nanmean(coef_physical_by_fold, axis=0)
    std_phys = np.nanstd(coef_physical_by_fold, axis=0, ddof=1)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAM_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)
    selected_df.to_csv(SELECTED_CSV, index=False)
    selection_freq_df.to_csv(SELECTION_FREQ_CSV, index=False)

    coef_ds = xr.Dataset(
        data_vars={
            "coef_weighted_space": (("heldout_wy", "month", "lat", "lon"), coef_weighted_by_fold.astype(np.float32)),
            "coef_physical_space": (("heldout_wy", "month", "lat", "lon"), coef_physical_by_fold.astype(np.float32)),
            "selected_mask": (("heldout_wy", "month", "lat", "lon"), selected_mask_by_fold.astype(np.int8)),
        },
        coords={
            "heldout_wy": WATER_YEARS.astype(np.int32),
            "month": np.asarray(MONTHS, dtype=object),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
        },
        attrs={"description": "Foldwise ElasticNet coefficients for the direct SST encoder."},
    )
    coef_ds.to_netcdf(COEF_FOLD_NC)

    freq_grid = np.full((len(MONTHS), lat.size, lon.size), np.nan, dtype=float)
    freq_grid.reshape(-1)[valid_flat] = selection_freq_df["selection_frequency"].to_numpy(dtype=float)
    sel_ds = xr.Dataset(
        data_vars={
            "selection_frequency": (("month", "lat", "lon"), freq_grid.astype(np.float32)),
            "mean_coef_physical_space": (("month", "lat", "lon"), mean_phys.astype(np.float32)),
            "std_coef_physical_space": (("month", "lat", "lon"), std_phys.astype(np.float32)),
        },
        coords={
            "month": np.asarray(MONTHS, dtype=object),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
        },
        attrs={"description": "ElasticNet selection frequency and mean physical-space coefficient maps."},
    )
    sel_ds.to_netcdf(SELECTION_FREQ_NC)

    plot_observed_vs_predicted(pred_df)
    plot_scatter(pred_df, metrics)
    plot_error_by_year(pred_df)
    plot_month_maps(freq_grid, lat, lon, "Selection frequency", SELECTION_MAP_PNG, "viridis")
    plot_month_maps(mean_phys, lat, lon, "Mean physical coefficient", MEAN_WEIGHT_MAP_PNG, "coolwarm")

    top_selected = selection_freq_df.sort_values("selection_frequency", ascending=False).head(20)
    oracle_overlap = selection_freq_df[
        ((selection_freq_df["month"] == "Jan") & np.isclose(selection_freq_df["lat"], -9.5) & np.isclose(selection_freq_df["lon"], 133.5))
        | ((selection_freq_df["month"] == "Oct") & np.isclose(selection_freq_df["lat"], 0.5) & np.isclose(selection_freq_df["lon"], 136.5))
    ].copy()
    alpha_summary = summarize_hyperparams(hyper_df)
    best_vs_baseline = {
        "baseline_name": "Z1_Z2_AMV_AMO_K5",
        **BASELINE_METRICS,
        "delta_R2": float(metrics["R2"] - BASELINE_METRICS["R2"]),
        "delta_RMSE": float(metrics["RMSE"] - BASELINE_METRICS["RMSE"]),
        "delta_sign_accuracy": float(metrics["sign_accuracy"] - BASELINE_METRICS["sign_accuracy"]),
    }
    short_answer = (
        "Elastic net direct SST does not provide useful strict LOYO predictability and does not approach the "
        "Z1_Z2 + AMV/AMO K5 baseline."
        if metrics["R2"] < 0.2
        else "Elastic net direct SST shows some strict LOYO signal but still does not beat the current baseline."
    )
    summary = {
        "input_sst_file": str(Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")),
        "target_file": str(PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"),
        "months": MONTHS,
        "metrics": metrics,
        "period_metrics": period_df.to_dict(orient="records"),
        "best_hyperparameter_summary": alpha_summary,
        "top_selected_features": top_selected.to_dict(orient="records"),
        "oracle_overlap": oracle_overlap.to_dict(orient="records"),
        "comparison_to_current_baseline": best_vs_baseline,
        "short_answer": short_answer,
    }
    dump_json(SUMMARY_JSON, summary)

    print(f"Elastic net output directory: {OUTPUT_DIR}", flush=True)
    print("Metrics:", flush=True)
    print(
        f"{MODEL_NAME}: R2={metrics['R2']:.6f} RMSE={metrics['RMSE']:.6f} "
        f"sign_accuracy={metrics['sign_accuracy']:.6f}",
        flush=True,
    )
    print(f"Best hyperparameter summary: {json.dumps(alpha_summary, sort_keys=True)}", flush=True)
    print(
        f"Mean/median selected feature count: {metrics['mean_num_selected_features']:.2f} / {metrics['median_num_selected_features']:.2f}",
        flush=True,
    )
    print(
        "Top selected features/month-locations: "
        + "; ".join(
            [
                "{} {:.1f}N {:.1f}E freq={:.3f}".format(row.month, row.lat, row.lon, row.selection_frequency)
                for row in top_selected.head(5).itertuples(index=False)
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
