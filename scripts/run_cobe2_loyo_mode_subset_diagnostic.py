#!/usr/bin/env python3
"""
Compare COBE2 first-3 LOD mode subsets under the saved LOYO reconstruction rule.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
)
SOURCE_FOLD_MODES_CSV = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/loyo_lod_analysis/cobe2_sierra_swe_lod_loyo_fold_modes.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_mode_subset_diagnostic"
MODE_COUNT = 3
MODEL_SPECS = [
    ("M1_only", [1]),
    ("M2_only", [2]),
    ("M3_only", [3]),
    ("M1_M2", [1, 2]),
    ("M1_M2_M3", [1, 2, 3]),
]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def load_target() -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(TARGET_FILE) as ds:
        water_years = np.asarray(ds.variables["water_year"][:], dtype=np.int32)
        target_anom = np.asarray(ds.variables["sierra_swe_apr1_anom_m"][:], dtype=np.float64)
    return water_years, target_anom


def load_selected_first3_rows() -> dict[int, dict[int, dict[str, float | int | str]]]:
    rows_by_wy: dict[int, dict[int, dict[str, float | int | str]]] = {}
    with SOURCE_FOLD_MODES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["selected"] != "True":
                continue
            mode_number = int(row["mode_number"])
            if mode_number > MODE_COUNT:
                continue
            heldout_wy = int(row["held_out_water_year"])
            rows_by_wy.setdefault(heldout_wy, {})[mode_number] = {
                "beta": float(row["beta"]),
                "mode_test_value_standardized": float(row["mode_test_value_standardized"]),
                "lag_month": row["lag_month"],
                "latitude": float(row["latitude"]),
                "longitude_0_360": float(row["longitude_0_360"]),
                "corr_with_residual": float(row["corr_with_residual"]),
            }
    return rows_by_wy


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true, ddof=1) == 0.0 or np.std(y_pred, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def sign_label(sign_value: int) -> str:
    if sign_value > 0:
        return "positive"
    if sign_value < 0:
        return "negative"
    return "zero"


def compute_sign_stability(beta_values: np.ndarray) -> dict[str, object]:
    signs = np.sign(beta_values).astype(np.int32)
    nonzero = signs[signs != 0]
    num_positive = int(np.sum(signs > 0))
    num_negative = int(np.sum(signs < 0))
    num_zero = int(np.sum(signs == 0))
    if nonzero.size == 0:
        majority_sign_value = 0
        sign_stability = float("nan")
    else:
        majority_sign_value = 1 if np.sum(nonzero > 0) >= np.sum(nonzero < 0) else -1
        sign_stability = float(np.mean(signs == majority_sign_value))
    return {
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_zero": num_zero,
        "majority_sign": sign_label(majority_sign_value),
        "majority_sign_value": majority_sign_value,
        "sign_stability": sign_stability,
        "mean_beta": float(np.mean(beta_values)),
        "std_beta": float(np.std(beta_values, ddof=1)),
        "min_beta": float(np.min(beta_values)),
        "max_beta": float(np.max(beta_values)),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def plot_observed_vs_predicted(
    path: Path,
    water_years: np.ndarray,
    observed: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
    ax.plot(water_years, observed, color="black", linewidth=1.8, label="Observed SWE anomaly")
    colors = {
        "M1_only": "tab:blue",
        "M2_only": "tab:orange",
        "M3_only": "tab:green",
        "M1_M2": "tab:red",
        "M1_M2_M3": "tab:purple",
    }
    for model_name, values in predictions_by_model.items():
        ax.plot(water_years, values, linewidth=1.15, label=model_name, color=colors[model_name])
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("COBE2 LOYO prediction from first-3 LOD mode subsets")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metrics_barplot(path: Path, metric_rows: list[dict[str, object]]) -> None:
    model_names = [str(row["model_name"]) for row in metric_rows]
    metric_names = ["r", "R2", "RMSE", "MAE"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(model_names))
    for ax, metric_name in zip(axes, metric_names):
        values = [float(row[metric_name]) for row in metric_rows]
        ax.bar(x, values, color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right")
        ax.set_title(metric_name)
        ax.grid(True, axis="y", linewidth=0.25, color="0.85")
        if metric_name in {"r", "R2"}:
            ax.axhline(0.0, color="0.5", linewidth=0.8)
    fig.suptitle("COBE2 LOYO mode-subset metrics", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def interpretation_from_metrics(metric_rows: list[dict[str, object]]) -> str:
    metrics = {str(row["model_name"]): row for row in metric_rows}
    m1_r2 = float(metrics["M1_only"]["R2"])
    m12_r2 = float(metrics["M1_M2"]["R2"])
    m123_r2 = float(metrics["M1_M2_M3"]["R2"])
    if m1_r2 < 0.0:
        return "Mode 1 alone is not predictively useful under LOYO."
    if m1_r2 > m12_r2 and m1_r2 > m123_r2:
        return "Mode 1 carries the most robust predictive signal; adding later modes likely hurts generalization."
    if m123_r2 > m12_r2 and m12_r2 > m1_r2:
        return "The first 3 modes add useful predictive information cumulatively."
    return "Mode subset results are mixed; inspect fold-level errors and beta stability."


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    water_years, target_anom = load_target()
    rows_by_wy = load_selected_first3_rows()
    observed = target_anom.astype(np.float64)

    predictions_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_name, included_modes in MODEL_SPECS:
        predicted = np.full(water_years.shape, np.nan, dtype=np.float64)
        beta_1_values: list[float] = []
        beta_2_values: list[float] = []
        beta_3_values: list[float] = []

        for idx, heldout_wy in enumerate(water_years.tolist()):
            mode_map = rows_by_wy.get(heldout_wy)
            if mode_map is None or any(mode not in mode_map for mode in range(1, MODE_COUNT + 1)):
                raise ValueError(f"Held-out WY{heldout_wy} is missing one or more of the first 3 saved mode rows.")
            train_mask = water_years != heldout_wy
            train_target = observed[train_mask]
            train_mean = float(np.mean(train_target))
            train_std = float(np.std(train_target, ddof=1))
            pred_std = 0.0

            beta_values_for_row = {1: float("nan"), 2: float("nan"), 3: float("nan")}
            for mode in included_modes:
                row = mode_map[mode]
                beta = float(row["beta"])
                mode_test = float(row["mode_test_value_standardized"])
                pred_std += beta * mode_test
                beta_values_for_row[mode] = beta
            pred_swe = float(train_mean + train_std * pred_std)
            predicted[idx] = pred_swe

            predictions_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": heldout_wy,
                    "obs_swe": float(observed[idx]),
                    "pred_swe": pred_swe,
                    "error": float(pred_swe - observed[idx]),
                }
            )
            beta_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": heldout_wy,
                    "intercept": train_mean,
                    "beta_1": beta_values_for_row[1],
                    "beta_2": beta_values_for_row[2],
                    "beta_3": beta_values_for_row[3],
                    "sign_beta_1": np.nan if np.isnan(beta_values_for_row[1]) else int(np.sign(beta_values_for_row[1])),
                    "sign_beta_2": np.nan if np.isnan(beta_values_for_row[2]) else int(np.sign(beta_values_for_row[2])),
                    "sign_beta_3": np.nan if np.isnan(beta_values_for_row[3]) else int(np.sign(beta_values_for_row[3])),
                }
            )
            if 1 in included_modes:
                beta_1_values.append(beta_values_for_row[1])
            if 2 in included_modes:
                beta_2_values.append(beta_values_for_row[2])
            if 3 in included_modes:
                beta_3_values.append(beta_values_for_row[3])

        predictions_by_model[model_name] = predicted
        observed_mean = float(np.mean(observed))
        sse = float(np.sum((observed - predicted) ** 2))
        sst = float(np.sum((observed - observed_mean) ** 2))
        metric_row = {
            "model_name": model_name,
            "num_modes": len(included_modes),
            "r": corrcoef_safe(observed, predicted),
            "R2": 1.0 - sse / sst,
            "RMSE": rmse(observed, predicted),
            "MAE": mae(observed, predicted),
        }
        metrics_rows.append(metric_row)

        beta_vectors = {1: beta_1_values, 2: beta_2_values, 3: beta_3_values}
        for mode in included_modes:
            stability = compute_sign_stability(np.asarray(beta_vectors[mode], dtype=np.float64))
            sign_rows.append(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "num_positive": stability["num_positive"],
                    "num_negative": stability["num_negative"],
                    "num_zero": stability["num_zero"],
                    "majority_sign": stability["majority_sign"],
                    "sign_stability": stability["sign_stability"],
                    "mean_beta": stability["mean_beta"],
                    "std_beta": stability["std_beta"],
                    "min_beta": stability["min_beta"],
                    "max_beta": stability["max_beta"],
                }
            )

    predictions_csv = OUTPUT_ROOT / "loyo_mode_subset_predictions.csv"
    metrics_csv = OUTPUT_ROOT / "loyo_mode_subset_metrics.csv"
    beta_csv = OUTPUT_ROOT / "loyo_mode_subset_beta_by_fold.csv"
    sign_csv = OUTPUT_ROOT / "loyo_mode_subset_beta_sign_stability.csv"
    line_png = OUTPUT_ROOT / "loyo_mode_subset_observed_vs_predicted.png"
    bar_png = OUTPUT_ROOT / "loyo_mode_subset_metrics_barplot.png"
    summary_json = OUTPUT_ROOT / "loyo_mode_subset_summary.json"

    write_csv(
        predictions_csv,
        ["model_name", "heldout_wy", "obs_swe", "pred_swe", "error"],
        predictions_rows,
    )
    write_csv(
        metrics_csv,
        ["model_name", "num_modes", "r", "R2", "RMSE", "MAE"],
        metrics_rows,
    )
    write_csv(
        beta_csv,
        ["model_name", "heldout_wy", "intercept", "beta_1", "beta_2", "beta_3", "sign_beta_1", "sign_beta_2", "sign_beta_3"],
        beta_rows,
    )
    write_csv(
        sign_csv,
        ["model_name", "mode", "num_positive", "num_negative", "num_zero", "majority_sign", "sign_stability", "mean_beta", "std_beta", "min_beta", "max_beta"],
        sign_rows,
    )
    plot_observed_vs_predicted(line_png, water_years, observed, predictions_by_model)
    plot_metrics_barplot(bar_png, metrics_rows)

    best_by_r2 = max(metrics_rows, key=lambda row: float(row["R2"]))
    best_by_rmse = min(metrics_rows, key=lambda row: float(row["RMSE"]))
    summary = {
        "dataset": "COBE2",
        "diagnostic_name": "LOYO mode subset diagnostic for first-3 LOD modes",
        "source_fold_modes_csv": str(SOURCE_FOLD_MODES_CSV),
        "target_file": str(TARGET_FILE),
        "mode_selection_inside_loyo": True,
        "negative_copies_added": False,
        "regression_convention": "Same saved LOYO first-3-mode reconstruction convention: standardized-trainfold prediction is the sum of beta_k * mode_test_value_standardized over included modes, and raw-space prediction is train_mean + train_std * pred_std.",
        "intercept_convention": "The beta rows store the implied raw-space intercept equal to the train-fold target mean because the saved reconstruction standardizes the target within each training fold.",
        "metrics": metrics_rows,
        "beta_sign_stability": sign_rows,
        "best_model_by_R2": best_by_r2["model_name"],
        "best_model_by_RMSE": best_by_rmse["model_name"],
        "interpretation": interpretation_from_metrics(metrics_rows),
        "outputs": {
            "output_directory": str(OUTPUT_ROOT),
            "loyo_mode_subset_predictions_csv": str(predictions_csv),
            "loyo_mode_subset_metrics_csv": str(metrics_csv),
            "loyo_mode_subset_beta_by_fold_csv": str(beta_csv),
            "loyo_mode_subset_beta_sign_stability_csv": str(sign_csv),
            "loyo_mode_subset_observed_vs_predicted_png": str(line_png),
            "loyo_mode_subset_metrics_barplot_png": str(bar_png),
            "loyo_mode_subset_summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Best model by LOYO R2: {best_by_r2['model_name']}")
    print(f"Best model by LOYO RMSE: {best_by_rmse['model_name']}")
    print("Metrics table:")
    for row in metrics_rows:
        print(
            f"{row['model_name']}: r={float(row['r']):.6f}, "
            f"R2={float(row['R2']):.6f}, RMSE={float(row['RMSE']):.6f}, MAE={float(row['MAE']):.6f}"
        )
    print("Sign stability table:")
    for row in sign_rows:
        print(
            f"{row['model_name']} mode {int(row['mode'])}: "
            f"majority={row['majority_sign']}, stability={float(row['sign_stability']):.6f}, "
            f"+={int(row['num_positive'])}, -={int(row['num_negative'])}, 0={int(row['num_zero'])}"
        )


if __name__ == "__main__":
    main()
