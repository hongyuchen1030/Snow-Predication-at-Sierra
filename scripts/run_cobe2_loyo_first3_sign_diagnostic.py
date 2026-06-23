#!/usr/bin/env python3
"""
Build a COBE2 first-3-mode LOYO coefficient-sign diagnostic from saved fold outputs.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_first3_sign_diagnostic"
MODE_COUNT = 3


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def load_target() -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(TARGET_FILE) as ds:
        water_years = np.asarray(ds.variables["water_year"][:], dtype=np.int32)
        target_anom = np.asarray(ds.variables["sierra_swe_apr1_anom_m"][:], dtype=np.float64)
    return water_years, target_anom


def load_selected_first3_rows() -> dict[int, list[dict[str, object]]]:
    rows_by_wy: dict[int, list[dict[str, object]]] = {}
    with SOURCE_FOLD_MODES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["selected"] != "True":
                continue
            mode_number = int(row["mode_number"])
            if mode_number > MODE_COUNT:
                continue
            heldout_wy = int(row["held_out_water_year"])
            parsed = {
                "held_out_water_year": heldout_wy,
                "mode_number": mode_number,
                "lag_month": row["lag_month"],
                "latitude": float(row["latitude"]),
                "longitude_0_360": float(row["longitude_0_360"]),
                "corr_with_residual": float(row["corr_with_residual"]),
                "beta": float(row["beta"]),
                "mode_test_value_standardized": float(row["mode_test_value_standardized"]),
                "selected": True,
            }
            rows_by_wy.setdefault(heldout_wy, []).append(parsed)
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


def compute_sign_summary(beta_values: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode_index in range(MODE_COUNT):
        values = beta_values[:, mode_index]
        signs = np.sign(values).astype(np.int32)
        nonzero = signs[signs != 0]
        num_positive = int(np.sum(signs > 0))
        num_negative = int(np.sum(signs < 0))
        num_zero = int(np.sum(signs == 0))
        if nonzero.size == 0:
            majority_sign = 0
            sign_stability = float("nan")
        else:
            majority_sign = 1 if np.sum(nonzero > 0) >= np.sum(nonzero < 0) else -1
            sign_stability = float(np.mean(signs == majority_sign))
        rows.append(
            {
                "mode": mode_index + 1,
                "num_positive": num_positive,
                "num_negative": num_negative,
                "num_zero": num_zero,
                "majority_sign": sign_label(majority_sign),
                "majority_sign_value": majority_sign,
                "sign_stability": sign_stability,
                "mean_beta": float(np.mean(values)),
                "std_beta": float(np.std(values, ddof=1)),
                "min_beta": float(np.min(values)),
                "max_beta": float(np.max(values)),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def plot_beta_signs(path: Path, water_years: np.ndarray, beta_values: np.ndarray) -> None:
    sign_matrix = np.sign(beta_values).astype(np.int32).T
    fig, ax = plt.subplots(figsize=(13.0, 3.8), constrained_layout=True)
    cmap = ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    image = ax.imshow(sign_matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(water_years.size))
    ax.set_xticklabels(water_years, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(MODE_COUNT))
    ax.set_yticklabels([f"Mode {i}" for i in range(1, MODE_COUNT + 1)])
    ax.set_xlabel("Held-out water year")
    ax.set_title("LOYO coefficient signs for COBE2 first-3 LOD modes")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "+1"])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_beta_values(path: Path, water_years: np.ndarray, beta_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for mode_index in range(MODE_COUNT):
        ax.plot(
            water_years,
            beta_values[:, mode_index],
            linewidth=1.6,
            color=colors[mode_index],
            label=f"beta_{mode_index + 1}",
        )
    ax.axhline(0.0, color="0.4", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Coefficient value")
    ax.set_title("LOYO coefficient values for COBE2 first-3 LOD modes")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_prediction_diagnostic(
    path: Path,
    water_years: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    sign_summary_rows: list[dict[str, object]],
    r_value: float,
    r2_value: float,
    rmse_value: float,
    mae_value: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    ax.plot(water_years, observed, color="black", linewidth=1.6, label="Observed SWE anomaly")
    ax.plot(water_years, predicted, color="tab:red", linewidth=1.3, label="Predicted SWE anomaly")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    stability_text = ", ".join(
        f"M{int(row['mode'])}={row['sign_stability']:.3f}" if np.isfinite(row["sign_stability"]) else f"M{int(row['mode'])}=nan"
        for row in sign_summary_rows
    )
    ax.set_title(
        "COBE2 first-3-mode LOYO prediction with beta-sign diagnostic\n"
        f"sign stability: {stability_text}"
    )
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, loc="upper left")
    metric_text = "\n".join(
        [
            f"r = {r_value:.3f}",
            f"R2 = {r2_value:.3f}",
            f"RMSE = {rmse_value:.5f} m",
            f"MAE = {mae_value:.5f} m",
        ]
    )
    ax.text(
        0.99,
        0.03,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    water_years, target_anom = load_target()
    rows_by_wy = load_selected_first3_rows()

    beta_values = np.full((water_years.size, MODE_COUNT), np.nan, dtype=np.float64)
    predicted = np.full(water_years.shape, np.nan, dtype=np.float64)
    beta_by_fold_rows: list[dict[str, object]] = []
    selected_mode_rows: list[dict[str, object]] = []

    for idx, heldout_wy in enumerate(water_years.tolist()):
        mode_rows = sorted(rows_by_wy.get(heldout_wy, []), key=lambda row: int(row["mode_number"]))
        if len(mode_rows) != MODE_COUNT:
            raise ValueError(f"Held-out WY{heldout_wy} has {len(mode_rows)} selected first-3 mode rows, expected {MODE_COUNT}.")
        train_mask = water_years != heldout_wy
        train_target = target_anom[train_mask]
        train_mean = float(np.mean(train_target))
        train_std = float(np.std(train_target, ddof=1))
        pred_std = 0.0
        for mode_row in mode_rows:
            mode_index = int(mode_row["mode_number"]) - 1
            beta = float(mode_row["beta"])
            beta_values[idx, mode_index] = beta
            pred_std += beta * float(mode_row["mode_test_value_standardized"])
            selected_mode_rows.append(
                {
                    "heldout_wy": heldout_wy,
                    "mode": int(mode_row["mode_number"]),
                    "lag_month": mode_row["lag_month"],
                    "lat": float(mode_row["latitude"]),
                    "lon": float(mode_row["longitude_0_360"]),
                    "selected_corr": float(mode_row["corr_with_residual"]),
                }
            )
        pred_swe = float(train_mean + train_std * pred_std)
        predicted[idx] = pred_swe
        beta_by_fold_rows.append(
            {
                "heldout_wy": heldout_wy,
                "obs_swe": float(target_anom[idx]),
                "pred_swe": pred_swe,
                "beta_1": float(beta_values[idx, 0]),
                "beta_2": float(beta_values[idx, 1]),
                "beta_3": float(beta_values[idx, 2]),
                "sign_beta_1": int(np.sign(beta_values[idx, 0])),
                "sign_beta_2": int(np.sign(beta_values[idx, 1])),
                "sign_beta_3": int(np.sign(beta_values[idx, 2])),
            }
        )

    observed = target_anom.astype(np.float64)
    r_value = corrcoef_safe(observed, predicted)
    observed_mean = float(np.mean(observed))
    sse = float(np.sum((observed - predicted) ** 2))
    sst = float(np.sum((observed - observed_mean) ** 2))
    r2_value = 1.0 - sse / sst
    rmse_value = rmse(observed, predicted)
    mae_value = mae(observed, predicted)

    sign_summary_rows = compute_sign_summary(beta_values)
    sign_flip_frequent = any(
        np.isfinite(row["sign_stability"]) and float(row["sign_stability"]) < 0.8
        for row in sign_summary_rows
    )

    beta_by_fold_csv = OUTPUT_ROOT / "loyo_first3_beta_by_fold.csv"
    sign_summary_csv = OUTPUT_ROOT / "loyo_first3_sign_stability_summary.csv"
    selected_mode_csv = OUTPUT_ROOT / "loyo_first3_selected_mode_metadata_by_fold.csv"
    beta_signs_png = OUTPUT_ROOT / "loyo_first3_beta_signs_by_year.png"
    beta_values_png = OUTPUT_ROOT / "loyo_first3_beta_values_by_year.png"
    prediction_png = OUTPUT_ROOT / "loyo_first3_prediction_with_beta_sign_diagnostic.png"
    summary_json = OUTPUT_ROOT / "loyo_first3_sign_diagnostic_summary.json"

    write_csv(
        beta_by_fold_csv,
        [
            "heldout_wy",
            "obs_swe",
            "pred_swe",
            "beta_1",
            "beta_2",
            "beta_3",
            "sign_beta_1",
            "sign_beta_2",
            "sign_beta_3",
        ],
        beta_by_fold_rows,
    )
    write_csv(
        sign_summary_csv,
        [
            "mode",
            "num_positive",
            "num_negative",
            "num_zero",
            "majority_sign",
            "sign_stability",
            "mean_beta",
            "std_beta",
            "min_beta",
            "max_beta",
        ],
        sign_summary_rows,
    )
    write_csv(
        selected_mode_csv,
        ["heldout_wy", "mode", "lag_month", "lat", "lon", "selected_corr"],
        selected_mode_rows,
    )

    plot_beta_signs(beta_signs_png, water_years, beta_values)
    plot_beta_values(beta_values_png, water_years, beta_values)
    plot_prediction_diagnostic(
        prediction_png,
        water_years,
        observed,
        predicted,
        sign_summary_rows,
        r_value,
        r2_value,
        rmse_value,
        mae_value,
    )

    summary = {
        "dataset": "COBE2",
        "diagnostic_name": "LOYO first-3 coefficient-sign diagnostic",
        "source_fold_modes_csv": str(SOURCE_FOLD_MODES_CSV),
        "target_file": str(TARGET_FILE),
        "mode_selection_inside_loyo": True,
        "mode_selection_rule": "Saved per-fold first-3 selected modes from the existing COBE2 LOYO LOD run were reused exactly.",
        "prediction_rule": "For each fold, pred_std = sum_k(beta_k * mode_test_value_standardized_k) over k=1..3, then pred_swe = train_mean + train_std * pred_std.",
        "leakage_warning": "",
        "metrics": {
            "r": r_value,
            "R2": r2_value,
            "RMSE": rmse_value,
            "MAE": mae_value,
            "sse": sse,
            "sst": sst,
        },
        "sign_stability": sign_summary_rows,
        "coefficient_signs_frequently_flip": sign_flip_frequent,
        "outputs": {
            "output_directory": str(OUTPUT_ROOT),
            "loyo_first3_beta_by_fold_csv": str(beta_by_fold_csv),
            "loyo_first3_sign_stability_summary_csv": str(sign_summary_csv),
            "loyo_first3_selected_mode_metadata_by_fold_csv": str(selected_mode_csv),
            "loyo_first3_beta_signs_by_year_png": str(beta_signs_png),
            "loyo_first3_beta_values_by_year_png": str(beta_values_png),
            "loyo_first3_prediction_with_beta_sign_diagnostic_png": str(prediction_png),
            "loyo_first3_sign_diagnostic_summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    stability_text = ", ".join(
        f"M{int(row['mode'])}={row['sign_stability']:.3f}" if np.isfinite(row["sign_stability"]) else f"M{int(row['mode'])}=nan"
        for row in sign_summary_rows
    )
    print(f"output directory: {OUTPUT_ROOT}")
    print(f"sign stability: {stability_text}")
    print(f"prediction metrics: r={r_value:.6f}, R2={r2_value:.6f}, RMSE={rmse_value:.6f}, MAE={mae_value:.6f}")
    print("mode selection inside LOYO: true")
    print(f"coefficient signs frequently flip: {'yes' if sign_flip_frequent else 'no'}")


if __name__ == "__main__":
    main()
