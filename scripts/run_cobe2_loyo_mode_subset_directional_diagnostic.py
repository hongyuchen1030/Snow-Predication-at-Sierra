#!/usr/bin/env python3
"""
Directional diagnostics for saved COBE2 LOYO mode-subset predictions.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "loyo_mode_subset_diagnostic"
    / "loyo_mode_subset_predictions.csv"
)
INPUT_METRICS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "loyo_mode_subset_diagnostic"
    / "loyo_mode_subset_metrics.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "cobe2_loyo_directional_diagnostics"
)
MODEL_ORDER = ["M1_only", "M2_only", "M3_only", "M1_M2", "M1_M2_M3"]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def load_predictions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    by_model: dict[str, list[dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model_name = row["model_name"]
            by_model.setdefault(model_name, []).append(
                {
                    "heldout_wy": int(row["heldout_wy"]),
                    "obs_swe": float(row["obs_swe"]),
                    "pred_swe": float(row["pred_swe"]),
                    "error": float(row["error"]),
                }
            )
    result: dict[str, dict[str, np.ndarray]] = {}
    for model_name, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda item: item["heldout_wy"])
        result[model_name] = {
            "heldout_wy": np.asarray([row["heldout_wy"] for row in rows_sorted], dtype=np.int32),
            "obs_swe": np.asarray([row["obs_swe"] for row in rows_sorted], dtype=np.float64),
            "pred_swe": np.asarray([row["pred_swe"] for row in rows_sorted], dtype=np.float64),
            "error": np.asarray([row["error"] for row in rows_sorted], dtype=np.float64),
        }
    return result


def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    y_true_f = y_true[finite]
    y_pred_f = y_pred[finite]
    if np.std(y_true_f, ddof=1) == 0.0 or np.std(y_pred_f, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true_f, y_pred_f)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def sign_accuracy_details(obs: np.ndarray, pred: np.ndarray, years: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    obs_sel = obs[mask]
    pred_sel = pred[mask]
    years_sel = years[mask]
    finite = np.isfinite(obs_sel) & np.isfinite(pred_sel)
    obs_f = obs_sel[finite]
    pred_f = pred_sel[finite]
    years_f = years_sel[finite]
    zero_obs = obs_f == 0.0
    zero_pred = pred_f == 0.0
    valid_sign = (~zero_obs) & (~zero_pred)
    used_obs = obs_f[valid_sign]
    used_pred = pred_f[valid_sign]
    used_years = years_f[valid_sign]
    if used_obs.size == 0:
        sign_accuracy = float("nan")
        opposite_years: list[int] = []
    else:
        correct = np.sign(used_obs) == np.sign(used_pred)
        sign_accuracy = float(np.mean(correct))
        opposite_years = used_years[~correct].astype(int).tolist()
    return {
        "sign_accuracy": sign_accuracy,
        "n_total": int(mask.sum()),
        "n_used_sign": int(valid_sign.sum()),
        "n_zero_obs": int(zero_obs.sum()),
        "n_zero_pred": int(zero_pred.sum()),
        "opposite_sign_years": opposite_years,
    }


def format_year_list(years: list[int]) -> str:
    return ",".join(str(year) for year in years)


def plot_sign_accuracy(path: Path, metric_rows: list[dict[str, object]]) -> None:
    labels = [str(row["model_subset"]) for row in metric_rows]
    x = np.arange(len(labels))
    width = 0.24
    overall = [float(row["sign_accuracy"]) for row in metric_rows]
    pre = [float(row["sign_accuracy_pre_2010"]) for row in metric_rows]
    post = [float(row["sign_accuracy_post_2010"]) for row in metric_rows]

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.bar(x - width, overall, width=width, label="overall", color="tab:blue")
    ax.bar(x, pre, width=width, label="pre_2010", color="tab:orange")
    ax.bar(x + width, post, width=width, label="post_2010", color="tab:green")
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Sign accuracy")
    ax.set_title("COBE2 LOYO mode-subset directional accuracy")
    ax.grid(True, axis="y", linewidth=0.25, color="0.85")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pre_post_r(path: Path, metric_rows: list[dict[str, object]]) -> None:
    labels = [str(row["model_subset"]) for row in metric_rows]
    x = np.arange(len(labels))
    width = 0.32
    pre = [float(row["r_pre_2010"]) for row in metric_rows]
    post = [float(row["r_post_2010"]) for row in metric_rows]

    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    ax.bar(x - width / 2, pre, width=width, label="pre_2010", color="tab:purple")
    ax.bar(x + width / 2, post, width=width, label="post_2010", color="tab:red")
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Correlation r")
    ax.set_title("COBE2 LOYO pre/post-2010 correlations")
    ax.grid(True, axis="y", linewidth=0.25, color="0.85")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_opposite_sign_highlight(
    path: Path,
    years: np.ndarray,
    observed: np.ndarray,
    model_a_name: str,
    pred_a: np.ndarray,
    opp_a: set[int],
    model_b_name: str,
    pred_b: np.ndarray,
    opp_b: set[int],
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.0), constrained_layout=True)
    ax.plot(years, observed, color="black", linewidth=1.8, label="Observed")
    ax.plot(years, pred_a, color="tab:blue", linewidth=1.2, label=model_a_name)
    ax.plot(years, pred_b, color="tab:red", linewidth=1.2, label=model_b_name)
    ax.axhline(0.0, color="0.5", linewidth=0.8)

    for year in sorted(opp_a):
        idx = int(np.where(years == year)[0][0])
        ax.scatter(year, pred_a[idx], color="tab:blue", s=28, marker="o", zorder=4)
    for year in sorted(opp_b):
        idx = int(np.where(years == year)[0][0])
        ax.scatter(year, pred_b[idx], color="tab:red", s=28, marker="s", zorder=4)

    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Opposite-sign years highlighted for best-RMSE model and M1_M2_M3")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    if not INPUT_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Saved prediction file is missing: {INPUT_PREDICTIONS_CSV}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(INPUT_PREDICTIONS_CSV)

    metric_rows: list[dict[str, object]] = []
    metrics_by_model: dict[str, dict[str, object]] = {}

    for model_name in MODEL_ORDER:
        data = predictions[model_name]
        years = data["heldout_wy"]
        obs = data["obs_swe"]
        pred = data["pred_swe"]
        pre_mask = years <= 2009
        post_mask = years >= 2010

        overall_sign = sign_accuracy_details(obs, pred, years, np.ones(years.shape, dtype=bool))
        pre_sign = sign_accuracy_details(obs, pred, years, pre_mask)
        post_sign = sign_accuracy_details(obs, pred, years, post_mask)

        obs_mean = float(np.mean(obs))
        sse = float(np.sum((obs - pred) ** 2))
        sst = float(np.sum((obs - obs_mean) ** 2))
        row = {
            "model_subset": model_name,
            "r": corrcoef_safe(obs, pred),
            "R2": 1.0 - sse / sst,
            "RMSE": rmse(obs, pred),
            "MAE": mae(obs, pred),
            "sign_accuracy": overall_sign["sign_accuracy"],
            "sign_accuracy_pre_2010": pre_sign["sign_accuracy"],
            "sign_accuracy_post_2010": post_sign["sign_accuracy"],
            "r_pre_2010": corrcoef_safe(obs[pre_mask], pred[pre_mask]),
            "r_post_2010": corrcoef_safe(obs[post_mask], pred[post_mask]),
            "n_total": int(years.size),
            "n_used_sign_total": overall_sign["n_used_sign"],
            "n_zero_obs_total": overall_sign["n_zero_obs"],
            "n_zero_pred_total": overall_sign["n_zero_pred"],
            "n_pre_2010": int(np.count_nonzero(pre_mask)),
            "n_used_sign_pre_2010": pre_sign["n_used_sign"],
            "n_post_2010": int(np.count_nonzero(post_mask)),
            "n_used_sign_post_2010": post_sign["n_used_sign"],
            "opposite_sign_years_all": format_year_list(overall_sign["opposite_sign_years"]),
            "opposite_sign_years_pre_2010": format_year_list(pre_sign["opposite_sign_years"]),
            "opposite_sign_years_post_2010": format_year_list(post_sign["opposite_sign_years"]),
        }
        metric_rows.append(row)
        metrics_by_model[model_name] = row

    metrics_csv = OUTPUT_ROOT / "cobe2_loyo_mode_subset_directional_metrics.csv"
    summary_json = OUTPUT_ROOT / "cobe2_loyo_mode_subset_directional_summary.json"
    sign_acc_png = OUTPUT_ROOT / "cobe2_loyo_mode_subset_sign_accuracy.png"
    pre_post_r_png = OUTPUT_ROOT / "cobe2_loyo_mode_subset_pre_post_r.png"
    highlight_png = OUTPUT_ROOT / "cobe2_loyo_mode_subset_best_and_full_opposite_sign_years.png"

    write_fields = [
        "model_subset",
        "r",
        "R2",
        "RMSE",
        "MAE",
        "sign_accuracy",
        "sign_accuracy_pre_2010",
        "sign_accuracy_post_2010",
        "r_pre_2010",
        "r_post_2010",
        "n_total",
        "n_used_sign_total",
        "n_zero_obs_total",
        "n_zero_pred_total",
        "n_pre_2010",
        "n_used_sign_pre_2010",
        "n_post_2010",
        "n_used_sign_post_2010",
        "opposite_sign_years_all",
        "opposite_sign_years_pre_2010",
        "opposite_sign_years_post_2010",
    ]
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fields)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({name: row.get(name) for name in write_fields})

    plot_sign_accuracy(sign_acc_png, metric_rows)
    plot_pre_post_r(pre_post_r_png, metric_rows)

    best_rmse_model = min(metric_rows, key=lambda row: float(row["RMSE"]))["model_subset"]
    best_sign_model = max(metric_rows, key=lambda row: (-math.inf if math.isnan(float(row["sign_accuracy"])) else float(row["sign_accuracy"])))["model_subset"]
    best_rmse_data = predictions[best_rmse_model]
    full_data = predictions["M1_M2_M3"]
    plot_opposite_sign_highlight(
        highlight_png,
        best_rmse_data["heldout_wy"],
        best_rmse_data["obs_swe"],
        str(best_rmse_model),
        best_rmse_data["pred_swe"],
        set(int(v) for v in filter(None, str(metrics_by_model[best_rmse_model]["opposite_sign_years_all"]).split(","))),
        "M1_M2_M3",
        full_data["pred_swe"],
        set(int(v) for v in filter(None, str(metrics_by_model["M1_M2_M3"]["opposite_sign_years_all"]).split(","))),
    )

    post_2010_drop = {
        model_name: (
            float(row["sign_accuracy_post_2010"]) < float(row["sign_accuracy_pre_2010"])
            if np.isfinite(float(row["sign_accuracy_post_2010"])) and np.isfinite(float(row["sign_accuracy_pre_2010"]))
            else None
        )
        for model_name, row in metrics_by_model.items()
    }
    negative_post_corr_models = [
        model_name for model_name, row in metrics_by_model.items() if np.isfinite(float(row["r_post_2010"])) and float(row["r_post_2010"]) < 0.0
    ]
    useful_directional_models = [
        model_name
        for model_name, row in metrics_by_model.items()
        if np.isfinite(float(row["sign_accuracy"])) and float(row["sign_accuracy"]) > 0.5
    ]

    summary = {
        "input_prediction_file": str(INPUT_PREDICTIONS_CSV),
        "input_metrics_file": str(INPUT_METRICS_CSV),
        "sign_convention": "Directional skill is evaluated on April 1 Sierra SWE anomaly sign using numpy sign on observed and predicted anomaly.",
        "zero_handling_convention": "Years where observed anomaly is zero or predicted anomaly is zero are excluded from the sign-accuracy denominator for that split and counted separately.",
        "metric_table_records": metric_rows,
        "best_relative_model_by_RMSE": best_rmse_model,
        "best_relative_model_by_sign_accuracy": best_sign_model,
        "best_relative_model_note": "Best here is relative among the tested bad-to-poor models, not evidence of useful predictive skill by itself.",
        "models_with_sign_accuracy_above_half": useful_directional_models,
        "any_model_has_useful_directional_skill": bool(useful_directional_models),
        "post_2010_sign_accuracy_worse_than_pre_2010": post_2010_drop,
        "models_with_negative_post_2010_correlation": negative_post_corr_models,
        "interpretation": {
            "best_model_by_RMSE": best_rmse_model,
            "best_model_by_sign_accuracy": best_sign_model,
            "post_2010_drop_summary": post_2010_drop,
            "negative_post_2010_correlation_models": negative_post_corr_models,
            "relative_vs_absolute_note": "The best relative model should not be interpreted as good unless its directional and error metrics are independently useful.",
        },
        "outputs": {
            "output_directory": str(OUTPUT_ROOT),
            "directional_metrics_csv": str(metrics_csv),
            "directional_summary_json": str(summary_json),
            "sign_accuracy_figure": str(sign_acc_png),
            "pre_post_r_figure": str(pre_post_r_png),
            "highlight_figure": str(highlight_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"output directory: {OUTPUT_ROOT}")
    print(f"metric CSV path: {metrics_csv}")
    print(f"summary JSON path: {summary_json}")
    print(f"figure paths: {sign_acc_png}, {pre_post_r_png}, {highlight_png}")
    print(
        "main result: "
        f"best relative RMSE model is {best_rmse_model}, "
        f"best relative sign-accuracy model is {best_sign_model}, "
        f"and negative post-2010 correlation occurs for {', '.join(negative_post_corr_models) if negative_post_corr_models else 'none'}."
    )


if __name__ == "__main__":
    main()
