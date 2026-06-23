#!/usr/bin/env python3
"""
Influence and stability diagnostic for the saved exact-grid-cell Z1+Z2 LOYO model.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo"
)
PREDICTIONS_CSV = INPUT_DIR / "full37_patch_loyo_predictions.csv"
METRICS_CSV = INPUT_DIR / "full37_patch_loyo_metrics.csv"
BETA_CSV = INPUT_DIR / "full37_patch_beta_by_fold.csv"
PREDICTORS_CSV = INPUT_DIR / "full37_patch_predictors.csv"
SUMMARY_JSON = INPUT_DIR / "full37_patch_predictor_summary.json"
OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_influence_diagnostic"
)
PATCH_NAME = "exact_grid_cell"
MODEL_NAME = "Z1_Z2"


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    xx = x[finite]
    yy = y[finite]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    return 1.0 - ss_res / ss_tot


def sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def load_filtered_predictions() -> tuple[list[dict[str, object]], dict[str, float]]:
    rows = []
    for row in read_csv(PREDICTIONS_CSV):
        if row["patch_size"] != PATCH_NAME or row["model_name"] != MODEL_NAME:
            continue
        obs = float(row["obs_swe"])
        pred = float(row["pred_swe"])
        err = float(row["error"])
        abs_err = float(row["abs_error"])
        sq_err = err**2
        rows.append(
            {
                "heldout_wy": int(row["heldout_wy"]),
                "obs_swe": obs,
                "pred_swe": pred,
                "error": err,
                "abs_error": abs_err,
                "squared_error": sq_err,
                "sign_correct": float(row["sign_correct"]),
            }
        )
    rows.sort(key=lambda item: int(item["heldout_wy"]))
    if len(rows) != 37:
        raise ValueError(f"Expected 37 filtered prediction rows, found {len(rows)}")
    metrics_row = next(
        row
        for row in read_csv(METRICS_CSV)
        if row["patch_size"] == PATCH_NAME and row["model_name"] == MODEL_NAME
    )
    return rows, {
        "r": float(metrics_row["r"]),
        "R2": float(metrics_row["R2"]),
        "RMSE": float(metrics_row["RMSE"]),
        "MAE": float(metrics_row["MAE"]),
        "sign_accuracy": float(metrics_row["sign_accuracy"]),
    }


def build_fold_error_rows(pred_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    total_sse = float(sum(float(row["squared_error"]) for row in pred_rows))
    by_error = sorted(pred_rows, key=lambda item: float(item["squared_error"]), reverse=True)
    cumulative = 0.0
    out_rows = []
    for row in by_error:
        share = float(row["squared_error"]) / total_sse
        cumulative += share
        out_rows.append(
            {
                **row,
                "SSE_share": share,
                "cumulative_SSE_share": cumulative,
            }
        )
    top_abs = sorted(pred_rows, key=lambda item: float(item["abs_error"]), reverse=True)
    summary = {
        "top_abs_error_years": [int(row["heldout_wy"]) for row in top_abs[:5]],
        "top_squared_error_share_years": [int(row["heldout_wy"]) for row in by_error[:5]],
        "cumulative_SSE_share_top1": float(sum(float(row["SSE_share"]) for row in out_rows[:1])),
        "cumulative_SSE_share_top2": float(sum(float(row["SSE_share"]) for row in out_rows[:2])),
        "cumulative_SSE_share_top3": float(sum(float(row["SSE_share"]) for row in out_rows[:3])),
        "cumulative_SSE_share_top5": float(sum(float(row["SSE_share"]) for row in out_rows[:5])),
    }
    return out_rows, summary


def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": sign_accuracy(obs, pred),
    }


def build_leave_one_prediction_sensitivity(
    pred_rows: list[dict[str, object]],
    all_metrics: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    obs = np.asarray([float(row["obs_swe"]) for row in pred_rows], dtype=np.float64)
    pred = np.asarray([float(row["pred_swe"]) for row in pred_rows], dtype=np.float64)
    years = np.asarray([int(row["heldout_wy"]) for row in pred_rows], dtype=np.int32)
    out_rows = []
    for idx, year in enumerate(years.tolist()):
        mask = np.ones(years.shape, dtype=bool)
        mask[idx] = False
        metrics = compute_metrics(obs[mask], pred[mask])
        out_rows.append(
            {
                "removed_wy": year,
                "n_eval": int(np.count_nonzero(mask)),
                "r_without_year": metrics["r"],
                "R2_without_year": metrics["R2"],
                "RMSE_without_year": metrics["RMSE"],
                "MAE_without_year": metrics["MAE"],
                "sign_accuracy_without_year": metrics["sign_accuracy"],
                "delta_r": metrics["r"] - all_metrics["r"],
                "delta_R2": metrics["R2"] - all_metrics["R2"],
                "delta_RMSE": metrics["RMSE"] - all_metrics["RMSE"],
                "delta_MAE": metrics["MAE"] - all_metrics["MAE"],
            }
        )
    r_values = np.asarray([float(row["r_without_year"]) for row in out_rows], dtype=np.float64)
    r2_values = np.asarray([float(row["R2_without_year"]) for row in out_rows], dtype=np.float64)
    rmse_values = np.asarray([float(row["RMSE_without_year"]) for row in out_rows], dtype=np.float64)
    mae_values = np.asarray([float(row["MAE_without_year"]) for row in out_rows], dtype=np.float64)
    largest_abs_delta_r2 = max(out_rows, key=lambda row: abs(float(row["delta_R2"])))
    largest_abs_delta_r = max(out_rows, key=lambda row: abs(float(row["delta_r"])))
    largest_abs_delta_rmse = max(out_rows, key=lambda row: abs(float(row["delta_RMSE"])))
    summary = {
        "leave_one_metric_ranges": {
            "r_min": float(np.min(r_values)),
            "r_max": float(np.max(r_values)),
            "R2_min": float(np.min(r2_values)),
            "R2_max": float(np.max(r2_values)),
            "RMSE_min": float(np.min(rmse_values)),
            "RMSE_max": float(np.max(rmse_values)),
            "MAE_min": float(np.min(mae_values)),
            "MAE_max": float(np.max(mae_values)),
        },
        "largest_metric_sensitivity_years": {
            "largest_abs_delta_R2": {
                "heldout_wy": int(largest_abs_delta_r2["removed_wy"]),
                "delta_R2": float(largest_abs_delta_r2["delta_R2"]),
            },
            "largest_abs_delta_r": {
                "heldout_wy": int(largest_abs_delta_r["removed_wy"]),
                "delta_r": float(largest_abs_delta_r["delta_r"]),
            },
            "largest_abs_delta_RMSE": {
                "heldout_wy": int(largest_abs_delta_rmse["removed_wy"]),
                "delta_RMSE": float(largest_abs_delta_rmse["delta_RMSE"]),
            },
        },
    }
    return out_rows, summary


def build_refit_influence(pred_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    pred_by_year = {int(row["heldout_wy"]): row for row in pred_rows}
    rows = []
    for row in read_csv(BETA_CSV):
        if row["patch_size"] != PATCH_NAME or row["model_name"] != MODEL_NAME:
            continue
        year = int(row["heldout_wy"])
        pred_row = pred_by_year[year]
        rows.append(
            {
                "heldout_wy": year,
                "intercept": float(row["intercept"]),
                "beta_Z1": float(row["beta_Z1"]),
                "beta_Z2": float(row["beta_Z2"]),
                "sign_beta_Z1": int(float(row["sign_beta_Z1"])),
                "sign_beta_Z2": int(float(row["sign_beta_Z2"])),
                "obs_swe": float(pred_row["obs_swe"]),
                "pred_swe": float(pred_row["pred_swe"]),
                "error": float(pred_row["error"]),
                "abs_error": float(pred_row["abs_error"]),
            }
        )
    rows.sort(key=lambda item: int(item["heldout_wy"]))
    beta_z1 = np.asarray([float(row["beta_Z1"]) for row in rows], dtype=np.float64)
    beta_z2 = np.asarray([float(row["beta_Z2"]) for row in rows], dtype=np.float64)
    abs_error = np.asarray([float(row["abs_error"]) for row in rows], dtype=np.float64)
    sign_stability_z1 = float(np.mean(np.sign(beta_z1) == (1 if np.sum(beta_z1 > 0.0) >= np.sum(beta_z1 < 0.0) else -1)))
    sign_stability_z2 = float(np.mean(np.sign(beta_z2) == (1 if np.sum(beta_z2 > 0.0) >= np.sum(beta_z2 < 0.0) else -1)))
    top5_years = {int(row["heldout_wy"]) for row in sorted(rows, key=lambda item: float(item["abs_error"]), reverse=True)[:5]}
    top5_rows = [row for row in rows if int(row["heldout_wy"]) in top5_years]
    summary = {
        "coefficient_stability": {
            "beta_Z1_summary": {
                "mean_beta_Z1": float(np.mean(beta_z1)),
                "std_beta_Z1": float(np.std(beta_z1, ddof=1)),
                "min_beta_Z1": float(np.min(beta_z1)),
                "max_beta_Z1": float(np.max(beta_z1)),
                "sign_stability_beta_Z1": sign_stability_z1,
            },
            "beta_Z2_summary": {
                "mean_beta_Z2": float(np.mean(beta_z2)),
                "std_beta_Z2": float(np.std(beta_z2, ddof=1)),
                "min_beta_Z2": float(np.min(beta_z2)),
                "max_beta_Z2": float(np.max(beta_z2)),
                "sign_stability_beta_Z2": sign_stability_z2,
            },
            "sign_stability": {
                "beta_Z1_majority_sign": "positive" if np.sum(beta_z1 > 0.0) >= np.sum(beta_z1 < 0.0) else "negative",
                "beta_Z2_majority_sign": "positive" if np.sum(beta_z2 > 0.0) >= np.sum(beta_z2 < 0.0) else "negative",
                "beta_Z1_sign_stability": sign_stability_z1,
                "beta_Z2_sign_stability": sign_stability_z2,
            },
        },
        "error_coefficient_correlations": {
            "corr_abs_error_beta_Z1": corrcoef_safe(abs_error, beta_z1),
            "corr_abs_error_beta_Z2": corrcoef_safe(abs_error, beta_z2),
            "corr_abs_error_abs_beta_Z1": corrcoef_safe(abs_error, np.abs(beta_z1)),
            "corr_abs_error_abs_beta_Z2": corrcoef_safe(abs_error, np.abs(beta_z2)),
        },
        "largest_error_year_coefficient_rows": top5_rows,
    }
    return rows, summary


def build_robust_metric_summary(pred_rows: list[dict[str, object]]) -> dict[str, object]:
    by_abs = sorted(pred_rows, key=lambda item: float(item["abs_error"]), reverse=True)
    obs_all = np.asarray([float(row["obs_swe"]) for row in pred_rows], dtype=np.float64)
    pred_all = np.asarray([float(row["pred_swe"]) for row in pred_rows], dtype=np.float64)
    medae = float(median([float(row["abs_error"]) for row in pred_rows]))
    out = {"median_abs_error": medae}
    for k in (1, 2, 3):
        drop_years = {int(row["heldout_wy"]) for row in by_abs[:k]}
        kept = [row for row in pred_rows if int(row["heldout_wy"]) not in drop_years]
        obs = np.asarray([float(row["obs_swe"]) for row in kept], dtype=np.float64)
        pred = np.asarray([float(row["pred_swe"]) for row in kept], dtype=np.float64)
        out[f"RMSE_drop_top{k}_abs_error"] = rmse(obs, pred)
        out[f"R2_drop_top{k}_abs_error"] = r2_manual(obs, pred)
    return out


def plot_observed_predicted_labeled(path: Path, pred_rows: list[dict[str, object]], all_metrics: dict[str, float]) -> None:
    years = np.asarray([int(row["heldout_wy"]) for row in pred_rows], dtype=np.int32)
    obs = np.asarray([float(row["obs_swe"]) for row in pred_rows], dtype=np.float64)
    pred = np.asarray([float(row["pred_swe"]) for row in pred_rows], dtype=np.float64)
    top5 = sorted(pred_rows, key=lambda item: float(item["abs_error"]), reverse=True)[:5]
    fig, ax = plt.subplots(figsize=(13.0, 5.5), constrained_layout=True)
    ax.plot(years, obs, color="black", linewidth=1.8, label="Observed SWE anomaly")
    ax.plot(years, pred, color="tab:blue", linewidth=1.2, label="Predicted SWE anomaly")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    for row in top5:
        year = int(row["heldout_wy"])
        idx = int(np.where(years == year)[0][0])
        ax.axvline(year, color="tab:red", linewidth=0.7, alpha=0.35)
        ax.text(year, pred[idx], str(year), fontsize=8, color="tab:red", rotation=45, ha="left", va="bottom")
    ax.set_title(
        "exact Z1+Z2 LOYO: "
        f"r={all_metrics['r']:.4f}, R2={all_metrics['R2']:.4f}, RMSE={all_metrics['RMSE']:.4f}, MAE={all_metrics['MAE']:.4f}"
    )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metric_sensitivity(path: Path, sensitivity_rows: list[dict[str, object]], all_metrics: dict[str, float]) -> None:
    years = np.asarray([int(row["removed_wy"]) for row in sensitivity_rows], dtype=np.int32)
    r_values = np.asarray([float(row["r_without_year"]) for row in sensitivity_rows], dtype=np.float64)
    r2_values = np.asarray([float(row["R2_without_year"]) for row in sensitivity_rows], dtype=np.float64)
    rmse_values = np.asarray([float(row["RMSE_without_year"]) for row in sensitivity_rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.5), constrained_layout=True)
    for ax, values, ref, label in zip(
        axes,
        (r_values, r2_values, rmse_values),
        (all_metrics["r"], all_metrics["R2"], all_metrics["RMSE"]),
        ("r without year", "R2 without year", "RMSE without year"),
    ):
        ax.plot(years, values, color="tab:blue", linewidth=1.1)
        ax.scatter(years, values, color="tab:blue", s=18)
        ax.axhline(ref, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.25, color="0.85")
    axes[-1].set_xlabel("Removed evaluation year")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_error_rank_barplot(path: Path, fold_error_rows: list[dict[str, object]]) -> None:
    years = [int(row["heldout_wy"]) for row in fold_error_rows]
    abs_error = [float(row["abs_error"]) for row in fold_error_rows]
    cumulative = [float(row["cumulative_SSE_share"]) for row in fold_error_rows]
    x = np.arange(len(years))
    fig, ax1 = plt.subplots(figsize=(13.0, 5.5), constrained_layout=True)
    ax1.bar(x, abs_error, color="tab:blue", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax1.set_ylabel("Absolute error")
    ax1.set_xlabel("Held-out water year sorted by descending squared error")
    ax1.grid(True, axis="y", linewidth=0.25, color="0.85")
    for idx in range(min(5, len(years))):
        ax1.text(x[idx], abs_error[idx], str(years[idx]), fontsize=8, rotation=45, ha="left", va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="tab:red", linewidth=1.2)
    ax2.set_ylabel("Cumulative SSE share")
    ax2.set_ylim(0.0, 1.02)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_coefficients_by_fold(path: Path, refit_rows: list[dict[str, object]]) -> None:
    years = np.asarray([int(row["heldout_wy"]) for row in refit_rows], dtype=np.int32)
    beta_z1 = np.asarray([float(row["beta_Z1"]) for row in refit_rows], dtype=np.float64)
    beta_z2 = np.asarray([float(row["beta_Z2"]) for row in refit_rows], dtype=np.float64)
    top5 = sorted(refit_rows, key=lambda item: float(item["abs_error"]), reverse=True)[:5]
    fig, ax = plt.subplots(figsize=(13.0, 5.5), constrained_layout=True)
    ax.plot(years, beta_z1, color="tab:blue", linewidth=1.2, label="beta_Z1")
    ax.plot(years, beta_z2, color="tab:orange", linewidth=1.2, label="beta_Z2")
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    for row in top5:
        year = int(row["heldout_wy"])
        idx = int(np.where(years == year)[0][0])
        ax.axvline(year, color="tab:red", linewidth=0.7, alpha=0.35)
        ax.text(year, beta_z1[idx], str(year), fontsize=8, color="tab:red", rotation=45, ha="left", va="bottom")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Coefficient")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_rows, all_metrics = load_filtered_predictions()
    fold_error_rows, fold_error_summary = build_fold_error_rows(pred_rows)
    sensitivity_rows, sensitivity_summary = build_leave_one_prediction_sensitivity(pred_rows, all_metrics)
    refit_rows, refit_summary = build_refit_influence(pred_rows)
    robust_summary = build_robust_metric_summary(pred_rows)

    fold_errors_csv = OUTPUT_DIR / "exact_Z1_Z2_fold_errors.csv"
    sensitivity_csv = OUTPUT_DIR / "exact_Z1_Z2_leave_one_prediction_metric_sensitivity.csv"
    refit_csv = OUTPUT_DIR / "exact_Z1_Z2_leave_one_refit_influence.csv"
    summary_json = OUTPUT_DIR / "exact_Z1_Z2_influence_summary.json"
    labeled_png = OUTPUT_DIR / "exact_Z1_Z2_observed_predicted_error_labeled.png"
    sensitivity_png = OUTPUT_DIR / "exact_Z1_Z2_metric_sensitivity_leave_one_year.png"
    rank_png = OUTPUT_DIR / "exact_Z1_Z2_error_rank_barplot.png"
    coef_png = OUTPUT_DIR / "exact_Z1_Z2_coefficients_by_fold.png"

    write_csv(
        fold_errors_csv,
        [
            "heldout_wy",
            "obs_swe",
            "pred_swe",
            "error",
            "abs_error",
            "squared_error",
            "sign_correct",
            "SSE_share",
            "cumulative_SSE_share",
        ],
        fold_error_rows,
    )
    write_csv(
        sensitivity_csv,
        [
            "removed_wy",
            "n_eval",
            "r_without_year",
            "R2_without_year",
            "RMSE_without_year",
            "MAE_without_year",
            "sign_accuracy_without_year",
            "delta_r",
            "delta_R2",
            "delta_RMSE",
            "delta_MAE",
        ],
        sensitivity_rows,
    )
    write_csv(
        refit_csv,
        [
            "heldout_wy",
            "intercept",
            "beta_Z1",
            "beta_Z2",
            "sign_beta_Z1",
            "sign_beta_Z2",
            "obs_swe",
            "pred_swe",
            "error",
            "abs_error",
        ],
        refit_rows,
    )

    plot_observed_predicted_labeled(labeled_png, pred_rows, all_metrics)
    plot_metric_sensitivity(sensitivity_png, sensitivity_rows, all_metrics)
    plot_error_rank_barplot(rank_png, fold_error_rows)
    plot_coefficients_by_fold(coef_png, refit_rows)

    top3_sse = fold_error_summary["cumulative_SSE_share_top3"]
    r_range = sensitivity_summary["leave_one_metric_ranges"]["r_max"] - sensitivity_summary["leave_one_metric_ranges"]["r_min"]
    r2_range = sensitivity_summary["leave_one_metric_ranges"]["R2_max"] - sensitivity_summary["leave_one_metric_ranges"]["R2_min"]
    if top3_sse >= 0.5 and (r2_range > 0.25 or r_range > 0.2):
        short_conclusion = (
            "The exact Z1+Z2 LOYO skill is strongly influenced by a few years; the apparent skill should be treated as fragile."
        )
    elif top3_sse < 0.4 and sensitivity_summary["leave_one_metric_ranges"]["R2_min"] > 0.0:
        short_conclusion = (
            "The exact Z1+Z2 LOYO skill is not driven by a single influential year; it appears moderately stable across years."
        )
    else:
        short_conclusion = (
            "The exact Z1+Z2 LOYO skill is partly stable but still sensitive to several difficult years; it should be treated as suggestive rather than fully robust."
        )

    summary = {
        "input_files": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "metrics_csv": str(METRICS_CSV),
            "beta_csv": str(BETA_CSV),
            "predictors_csv": str(PREDICTORS_CSV),
            "summary_json": str(SUMMARY_JSON),
        },
        "output_dir": str(OUTPUT_DIR),
        "model_analyzed": {"patch_size": PATCH_NAME, "model_name": MODEL_NAME},
        "all_year_metrics": all_metrics,
        **fold_error_summary,
        **sensitivity_summary,
        **refit_summary,
        "robust_metrics": robust_summary,
        "short_conclusion": short_conclusion,
        "outputs": {
            "exact_Z1_Z2_fold_errors_csv": str(fold_errors_csv),
            "exact_Z1_Z2_leave_one_prediction_metric_sensitivity_csv": str(sensitivity_csv),
            "exact_Z1_Z2_leave_one_refit_influence_csv": str(refit_csv),
            "exact_Z1_Z2_influence_summary_json": str(summary_json),
            "exact_Z1_Z2_observed_predicted_error_labeled_png": str(labeled_png),
            "exact_Z1_Z2_metric_sensitivity_leave_one_year_png": str(sensitivity_png),
            "exact_Z1_Z2_error_rank_barplot_png": str(rank_png),
            "exact_Z1_Z2_coefficients_by_fold_png": str(coef_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Analyzed model: {PATCH_NAME} / {MODEL_NAME}")
    print(
        "All-year metrics: "
        f"r={all_metrics['r']:.4f}, R2={all_metrics['R2']:.4f}, "
        f"RMSE={all_metrics['RMSE']:.5f}, MAE={all_metrics['MAE']:.5f}, "
        f"sign_accuracy={all_metrics['sign_accuracy']:.4f}"
    )
    print("Top 5 absolute-error years:")
    for row in sorted(pred_rows, key=lambda item: float(item["abs_error"]), reverse=True)[:5]:
        print(
            f"  WY{int(row['heldout_wy'])}: abs_error={float(row['abs_error']):.5f}, "
            f"obs={float(row['obs_swe']):.5f}, pred={float(row['pred_swe']):.5f}"
        )
    print(
        "SSE share top 1 / top 2 / top 3 / top 5: "
        f"{fold_error_summary['cumulative_SSE_share_top1']:.4f} / "
        f"{fold_error_summary['cumulative_SSE_share_top2']:.4f} / "
        f"{fold_error_summary['cumulative_SSE_share_top3']:.4f} / "
        f"{fold_error_summary['cumulative_SSE_share_top5']:.4f}"
    )
    ranges = sensitivity_summary["leave_one_metric_ranges"]
    print(
        "Leave-one metric ranges: "
        f"r=[{ranges['r_min']:.4f}, {ranges['r_max']:.4f}], "
        f"R2=[{ranges['R2_min']:.4f}, {ranges['R2_max']:.4f}], "
        f"RMSE=[{ranges['RMSE_min']:.5f}, {ranges['RMSE_max']:.5f}], "
        f"MAE=[{ranges['MAE_min']:.5f}, {ranges['MAE_max']:.5f}]"
    )
    sign_stability = refit_summary["coefficient_stability"]["sign_stability"]
    print(
        "Coefficient sign stability: "
        f"beta_Z1={sign_stability['beta_Z1_sign_stability']:.4f} "
        f"({sign_stability['beta_Z1_majority_sign']}), "
        f"beta_Z2={sign_stability['beta_Z2_sign_stability']:.4f} "
        f"({sign_stability['beta_Z2_majority_sign']})"
    )
    print("Short answer:")
    print(f"Is exact Z1+Z2 stable, or driven by a few influential years? {short_conclusion}")


if __name__ == "__main__":
    main()
