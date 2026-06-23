#!/usr/bin/env python3
"""
Reconstruct first-3-mode Sierra SWE LOYO predictions from saved 6-mode fold outputs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "sierra_swe_lod_first3_mode_prediction"
TARGET_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    fold_modes_csv: Path
    full_predictions_csv: Path
    full_skill_summary_json: Path


DATASETS = [
    DatasetConfig(
        key="cobe2",
        label="COBE2",
        fold_modes_csv=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "cobe2_sierra_swe_lod_setup/loyo_lod_analysis/cobe2_sierra_swe_lod_loyo_fold_modes.csv"
        ),
        full_predictions_csv=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "cobe2_sierra_swe_lod_setup/loyo_lod_analysis/cobe2_sierra_swe_lod_loyo_predictions.csv"
        ),
        full_skill_summary_json=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "cobe2_sierra_swe_lod_setup/loyo_lod_analysis/cobe2_sierra_swe_lod_loyo_skill_summary.json"
        ),
    ),
    DatasetConfig(
        key="era5",
        label="ERA5",
        fold_modes_csv=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/loyo_lod_analysis/era5_sierra_swe_lod_loyo_fold_modes.csv"
        ),
        full_predictions_csv=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/loyo_lod_analysis/era5_sierra_swe_lod_loyo_predictions.csv"
        ),
        full_skill_summary_json=Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/loyo_lod_analysis/era5_sierra_swe_lod_loyo_skill_summary.json"
        ),
    ),
]


def load_target() -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(TARGET_FILE) as ds:
        water_years = np.asarray(ds.variables["water_year"][:], dtype=np.int32)
        target_anom = np.asarray(ds.variables["sierra_swe_apr1_anom_m"][:], dtype=np.float64)
    return water_years, target_anom


def load_full_predictions(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wy = int(row["water_year"])
            out[wy] = {
                "predicted_swe_anom_m": float(row["predicted_swe_anom_m"]),
                "predicted_swe_standardized_trainfold": float(row["predicted_swe_standardized_trainfold"]),
                "n_selected_modes": float(row["n_selected_modes"]),
            }
    return out


def load_selected_first3_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    rows_by_wy: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["selected"] != "True":
                continue
            if int(row["mode_number"]) > 3:
                continue
            wy = int(row["held_out_water_year"])
            rows_by_wy.setdefault(wy, []).append(row)
    return rows_by_wy


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true, ddof=1) == 0.0 or np.std(y_pred, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def plot_scatter(path: Path, dataset_label: str, observed: np.ndarray, predicted: np.ndarray, r2_value: float, corr_value: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(observed, predicted, s=42, color="tab:blue", edgecolors="black", linewidths=0.4, alpha=0.85)
    values = np.concatenate([observed, predicted])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("Observed April 1 Sierra SWE anomaly (m)")
    ax.set_ylabel("Predicted April 1 Sierra SWE anomaly (m)")
    ax.set_title(f"{dataset_label} first-3-mode LOYO reconstruction\nR2={r2_value:.3f}  corr={corr_value:.3f}")
    ax.grid(True, linewidth=0.25, color="0.8")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_timeseries(path: Path, dataset_label: str, water_years: np.ndarray, observed: np.ndarray, predicted: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(water_years, observed, color="black", linewidth=1.5, label="Observed")
    ax.plot(water_years, predicted, color="tab:red", linewidth=1.2, label="Predicted from modes 1-3")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title(f"{dataset_label} first-3-mode LOYO reconstruction")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_summary(path: Path, dataset_label: str, summary: dict[str, object]) -> None:
    metrics = summary["metrics"]
    compare = summary["comparison_to_saved_6mode"]
    lines = [
        f"# {dataset_label} first-3-mode Sierra SWE LOYO reconstruction",
        "",
        "- Source logic: exact reconstruction from the saved 6-mode fold tables.",
        "- For each held-out water year, this keeps only saved selected rows with `mode_number <= 3`.",
        "- The prediction is recomputed as the foldwise sum of `beta * mode_test_value_standardized` over modes 1-3, then transformed back to SWE anomaly units using the same train-fold target mean and standard deviation implied by the saved target series.",
        "",
        "## Metrics",
        "",
        f"- LOYO R2: `{metrics['r2_loyo']:.4f}`",
        f"- LOYO RMSE: `{metrics['rmse_m']:.6f}` m",
        f"- LOYO MAE: `{metrics['mae_m']:.6f}` m",
        f"- LOYO correlation: `{metrics['correlation']:.4f}`",
        "",
        "## Comparison with saved 6-mode LOYO run",
        "",
        f"- Saved 6-mode LOYO R2: `{compare['saved_6mode_r2_loyo']:.4f}`",
        f"- Saved 6-mode LOYO RMSE: `{compare['saved_6mode_rmse_m']:.6f}` m",
        f"- Mean absolute change in predicted anomaly after truncating to 3 modes: `{compare['mean_abs_prediction_change_m']:.6f}` m",
        f"- Maximum absolute change in predicted anomaly after truncating to 3 modes: `{compare['max_abs_prediction_change_m']:.6f}` m",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    water_years, target_anom = load_target()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    for config in DATASETS:
        output_dir = ARTIFACT_ROOT / config.key
        output_dir.mkdir(parents=True, exist_ok=True)

        first3_rows_by_wy = load_selected_first3_rows(config.fold_modes_csv)
        full_predictions = load_full_predictions(config.full_predictions_csv)
        full_skill_summary = json.loads(config.full_skill_summary_json.read_text(encoding="utf-8"))

        prediction_rows: list[dict[str, object]] = []
        fold_subset_rows: list[dict[str, object]] = []
        predicted = np.full(water_years.shape, np.nan, dtype=np.float64)
        full_predicted = np.full(water_years.shape, np.nan, dtype=np.float64)

        for idx, wy in enumerate(water_years.tolist()):
            mode_rows = sorted(first3_rows_by_wy.get(wy, []), key=lambda row: int(row["mode_number"]))
            pred_std = sum(float(row["beta"]) * float(row["mode_test_value_standardized"]) for row in mode_rows)
            train_mask = water_years != wy
            train_target = target_anom[train_mask]
            train_mean = float(np.mean(train_target))
            train_std = float(np.std(train_target, ddof=1))
            pred_raw = float(train_mean + train_std * pred_std)
            predicted[idx] = pred_raw
            full_predicted[idx] = float(full_predictions[wy]["predicted_swe_anom_m"])

            for row in mode_rows:
                fold_subset_rows.append(row)

            prediction_rows.append(
                {
                    "water_year": wy,
                    "observed_swe_anom_m": float(target_anom[idx]),
                    "predicted_swe_anom_m": pred_raw,
                    "predicted_swe_standardized_trainfold": pred_std,
                    "n_selected_modes_used": len(mode_rows),
                    "source_fold_modes_csv": str(config.fold_modes_csv),
                }
            )

        observed = target_anom.astype(np.float64)
        observed_mean = float(np.mean(observed))
        sse = float(np.sum((observed - predicted) ** 2))
        sst = float(np.sum((observed - observed_mean) ** 2))
        r2_loyo = 1.0 - sse / sst
        rmse_value = rmse(observed, predicted)
        mae_value = mae(observed, predicted)
        corr_value = corrcoef_safe(observed, predicted)
        prediction_change = np.abs(predicted - full_predicted)

        base = f"{config.key}_sierra_swe_lod_loyo_first3mode"
        predictions_csv = output_dir / f"{base}_predictions.csv"
        fold_modes_csv = output_dir / f"{base}_fold_modes.csv"
        scatter_png = output_dir / f"{base}_observed_vs_predicted.png"
        timeseries_png = output_dir / f"{base}_timeseries.png"
        summary_json = output_dir / f"{base}_skill_summary.json"
        summary_md = output_dir / f"{base}_summary.md"

        write_csv(
            predictions_csv,
            [
                "water_year",
                "observed_swe_anom_m",
                "predicted_swe_anom_m",
                "predicted_swe_standardized_trainfold",
                "n_selected_modes_used",
                "source_fold_modes_csv",
            ],
            prediction_rows,
        )
        write_csv(
            fold_modes_csv,
            [
                "held_out_water_year",
                "train_year_count",
                "valid_candidate_count",
                "valid_counts_by_lag",
                "mode_number",
                "selected",
                "stop_reason",
                "candidate_index",
                "ocean_candidate_q",
                "lag_month",
                "latitude",
                "longitude_0_360",
                "broad_region",
                "corr_with_residual",
                "beta",
                "delta_r2",
                "cumulative_r2",
                "mode_mean_before_standardization",
                "mode_std_before_standardization",
                "mode_test_value_standardized",
                "orthogonalization_coefficients",
            ],
            fold_subset_rows,
        )
        plot_scatter(scatter_png, config.label, observed, predicted, r2_loyo, corr_value)
        plot_timeseries(timeseries_png, config.label, water_years, observed, predicted)

        summary = {
            "dataset": config.label,
            "target_file": str(TARGET_FILE),
            "source_fold_modes_csv": str(config.fold_modes_csv),
            "reconstruction_rule": "Per fold, keep only selected modes 1-3 from the saved 6-mode LOYO fold table and recompute the held-out prediction.",
            "metrics": {
                "r2_loyo": r2_loyo,
                "rmse_m": rmse_value,
                "mae_m": mae_value,
                "correlation": corr_value,
                "sse": sse,
                "sst": sst,
            },
            "comparison_to_saved_6mode": {
                "saved_6mode_r2_loyo": float(full_skill_summary["metrics"]["r2_loyo"]),
                "saved_6mode_rmse_m": float(full_skill_summary["metrics"]["rmse_m"]),
                "mean_abs_prediction_change_m": float(np.mean(prediction_change)),
                "max_abs_prediction_change_m": float(np.max(prediction_change)),
            },
            "outputs": {
                "predictions_csv": str(predictions_csv),
                "fold_modes_csv": str(fold_modes_csv),
                "scatter_png": str(scatter_png),
                "timeseries_png": str(timeseries_png),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_summary(summary_md, config.label, summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
