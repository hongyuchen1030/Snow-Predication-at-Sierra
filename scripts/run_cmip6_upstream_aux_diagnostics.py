from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from snow_ml.cmip6_cnn_experiment import FEATURE_Z_CLIP, build_model, r2_score


PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "cmip6_upstream_aux_v1_diagnostics"
U1_EXPERIMENT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/"
    "cmip6_upstream_aux_v1/experiments/U1_stage2_cpm_aqm"
)
BASE_LATENT_DIR = PROJECT_ROOT / "artifacts" / "cmip6_latent_aux_competition_v1" / "latent_arrays"
BASE_CORE_TABLE = PROJECT_ROOT / "artifacts" / "cmip6_latent_aux_competition_v1" / "linear_probe_core_table.csv"
BASE_CKA = PROJECT_ROOT / "artifacts" / "cmip6_latent_aux_competition_v1" / "cka_matrix.csv"
TARGET_NAMES = ("SWE_label", "CPM_label", "AQM_label")
TARGET_INDEX = {name: idx for idx, name in enumerate(TARGET_NAMES)}
PROBE_ALPHAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)


@dataclass
class RidgeModel:
    alpha: float
    intercept: float
    coef_raw: np.ndarray


def _log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_inputs(
    *,
    physical_data: np.ndarray,
    valid_mask: np.ndarray,
    feature_mu: np.ndarray,
    feature_sigma: np.ndarray,
) -> np.ndarray:
    x = np.asarray(physical_data, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([x, mask], axis=2).astype(np.float32)


def grouped_year_folds(train_years: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_years = np.unique(train_years)
    year_folds = np.array_split(unique_years, min(n_folds, unique_years.size))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for held_out_years in year_folds:
        val_mask = np.isin(train_years, held_out_years)
        train_idx = np.nonzero(~val_mask)[0]
        val_idx = np.nonzero(val_mask)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def fit_ridge_fixed_alpha(*, x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> RidgeModel:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_mean = x_train.mean(axis=0, dtype=np.float64)
    x_scale = x_train.std(axis=0, dtype=np.float64)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)
    x_std = (x_train - x_mean) / x_scale
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    coef_std = np.linalg.solve(
        x_std.T @ x_std + (alpha * np.eye(x_std.shape[1], dtype=np.float64)),
        x_std.T @ y_centered,
    )
    coef_raw = coef_std / x_scale
    intercept = y_mean - float(x_mean @ coef_raw)
    return RidgeModel(alpha=float(alpha), intercept=intercept, coef_raw=coef_raw.astype(np.float64))


def fit_ridge(*, x_train: np.ndarray, y_train: np.ndarray, train_years: np.ndarray) -> RidgeModel:
    folds = grouped_year_folds(train_years)
    best_alpha = float(PROBE_ALPHAS[0])
    best_score = float("inf")
    for alpha in PROBE_ALPHAS:
        fold_errors: list[float] = []
        for fold_train_idx, fold_val_idx in folds:
            model = fit_ridge_fixed_alpha(
                x_train=x_train[fold_train_idx],
                y_train=y_train[fold_train_idx],
                alpha=float(alpha),
            )
            pred = predict_ridge(model, x_train[fold_val_idx])
            fold_errors.append(float(np.mean((pred - y_train[fold_val_idx]) ** 2)))
        if fold_errors:
            score = float(np.mean(fold_errors))
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
    return fit_ridge_fixed_alpha(x_train=x_train, y_train=y_train, alpha=best_alpha)


def predict_ridge(model: RidgeModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ model.coef_raw + model.intercept


def pearson_r_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    denom = math.sqrt(float(np.sum(y_true_centered**2) * np.sum(y_pred_centered**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(y_true_centered * y_pred_centered) / denom)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "r2": r2_score(y_true.astype(np.float32), y_pred.astype(np.float32)),
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "pearson_r": pearson_r_np(y_true, y_pred),
    }


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    xy = x_centered.T @ y_centered
    xx = x_centered.T @ x_centered
    yy = y_centered.T @ y_centered
    numerator = float(np.sum(xy * xy))
    denominator = math.sqrt(float(np.sum(xx * xx)) * float(np.sum(yy * yy)))
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def plot_cka_heatmap(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Updated Final-Z Linear CKA")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bar(
    *,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    categories: list[str],
    series: dict[str, list[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(categories))
    width = 0.8 / max(len(series), 1)
    for idx, (label, values) in enumerate(series.items()):
        ax.bar(x + ((idx - (len(series) - 1) / 2) * width), values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def extract_u1_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    config = load_json(U1_EXPERIMENT / "config.json")
    split = load_json(U1_EXPERIMENT / "split.json")
    norm = np.load(U1_EXPERIMENT / "normalization_stats.npz")
    cache_dir = Path(config["predictor_cache_dir"])
    targets = np.load(cache_dir / "targets.npy")
    physical_data = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    valid_mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    checkpoint = torch.load(U1_EXPERIMENT / "best_checkpoint.pt", map_location="cpu")
    model = build_model(config["architecture"], in_channels_per_month=int(physical_data.shape[2] * 2))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    z_chunks: list[np.ndarray] = []
    f2_chunks: list[np.ndarray] = []
    batch_size = 4
    with torch.no_grad():
        for start in range(0, physical_data.shape[0], batch_size):
            stop = start + batch_size
            batch_inputs = build_inputs(
                physical_data=np.asarray(physical_data[start:stop], dtype=np.float32),
                valid_mask=np.asarray(valid_mask[start:stop], dtype=np.float32),
                feature_mu=np.asarray(norm["feature_mu"], dtype=np.float32),
                feature_sigma=np.asarray(norm["feature_sigma"], dtype=np.float32),
            )
            outputs = model(torch.from_numpy(batch_inputs).to(device))
            z_chunks.append(outputs["z"].detach().cpu().numpy().astype(np.float32))
            f2_chunks.append(outputs["stage2_pooled"].detach().cpu().numpy().astype(np.float32))
    z = np.concatenate(z_chunks, axis=0)
    f2 = np.concatenate(f2_chunks, axis=0)
    np.savez_compressed(
        OUTPUT_ROOT / "latent_arrays" / "U1_stage2_features.npz",
        z=z,
        z_flat=z.reshape(z.shape[0], -1),
        f2=f2,
    )
    return (
        z.reshape(z.shape[0], -1),
        f2,
        targets,
        np.asarray(split["train"], dtype=np.int64),
        np.asarray(split["val"], dtype=np.int64),
        config,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "latent_arrays").mkdir(parents=True, exist_ok=True)

    z_u1, f2_u1, targets, train_idx, val_idx, config = extract_u1_features()
    manifest = read_csv(Path(config["predictor_cache_dir"]) / "manifest.csv")
    water_years = np.asarray([int(row["water_year"]) for row in manifest], dtype=np.int64)
    train_years = water_years[train_idx]

    final_z_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    final_z_summary: dict[str, Any] = {}

    for feature_name, features, rows_store in [
        ("final_z", z_u1, final_z_rows),
        ("stage2", f2_u1, stage2_rows),
    ]:
        final_z_summary[feature_name] = {}
        for target_name in TARGET_NAMES:
            y_train = targets[train_idx, TARGET_INDEX[target_name]].astype(np.float64)
            y_val = targets[val_idx, TARGET_INDEX[target_name]].astype(np.float64)
            model = fit_ridge(x_train=features[train_idx], y_train=y_train, train_years=train_years)
            train_metrics = regression_metrics(y_train, predict_ridge(model, features[train_idx]))
            val_metrics = regression_metrics(y_val, predict_ridge(model, features[val_idx]))
            rows_store.append(
                {
                    "feature_space": feature_name,
                    "target": target_name,
                    "alpha": model.alpha,
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_pearson_r": val_metrics["pearson_r"],
                }
            )
            final_z_summary[feature_name][target_name] = {
                "alpha": model.alpha,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

    write_csv(
        OUTPUT_ROOT / "u1_final_z_probe_results.csv",
        final_z_rows,
        ["feature_space", "target", "alpha", "train_r2", "val_r2", "val_rmse", "val_mae", "val_pearson_r"],
    )
    write_csv(
        OUTPUT_ROOT / "u1_stage2_probe_results.csv",
        stage2_rows,
        ["feature_space", "target", "alpha", "train_r2", "val_r2", "val_rmse", "val_mae", "val_pearson_r"],
    )

    base_core_rows = read_csv(BASE_CORE_TABLE)
    u1_training_summary = load_json(U1_EXPERIMENT / "metrics_summary.json")
    u1_final_lookup = {row["target"]: row for row in final_z_rows}
    comparison_rows = [
        {
            **row,
        }
        for row in base_core_rows
    ]
    comparison_rows.append(
        {
            "encoder": "U1",
            "swe_probe_val_r2": u1_final_lookup["SWE_label"]["val_r2"],
            "swe_probe_rmse": u1_final_lookup["SWE_label"]["val_rmse"],
            "swe_probe_r": u1_final_lookup["SWE_label"]["val_pearson_r"],
            "cpm_probe_val_r2": u1_final_lookup["CPM_label"]["val_r2"],
            "cpm_probe_r": u1_final_lookup["CPM_label"]["val_pearson_r"],
            "aqm_probe_val_r2": u1_final_lookup["AQM_label"]["val_r2"],
            "aqm_probe_r": u1_final_lookup["AQM_label"]["val_pearson_r"],
        }
    )
    write_csv(
        OUTPUT_ROOT / "comparison_probe_table.csv",
        comparison_rows,
        [
            "encoder",
            "swe_probe_val_r2",
            "swe_probe_rmse",
            "swe_probe_r",
            "cpm_probe_val_r2",
            "cpm_probe_r",
            "aqm_probe_val_r2",
            "aqm_probe_r",
        ],
    )

    base_latents = {
        label: np.load(BASE_LATENT_DIR / f"{label}_latents.npz")["z_flat"]
        for label in ("S0", "D1", "P1", "P2")
    }
    labels = ["S0", "D1", "P1", "P2", "U1"]
    all_latents = {**base_latents, "U1": z_u1}
    cka_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            cka_matrix[i, j] = linear_cka(all_latents[label_i][val_idx], all_latents[label_j][val_idx])
    cka_rows = []
    for i, label in enumerate(labels):
        row = {"encoder": label}
        for j, other in enumerate(labels):
            row[other] = float(cka_matrix[i, j])
        cka_rows.append(row)
    write_csv(OUTPUT_ROOT / "updated_cka_matrix.csv", cka_rows, ["encoder", *labels])
    plot_cka_heatmap(OUTPUT_ROOT / "updated_cka_heatmap.png", labels, cka_matrix)

    stage2_lookup = {row["target"]: row for row in stage2_rows}
    comparison_table_rows = [
        {
            "model": "S0",
            "aux_location": "none",
            "swe_val_r2": 0.3754518212143655,
            "swe_rmse": 27.070167541503906,
            "swe_mae": 19.154634475708008,
            "swe_r": 0.6234545111656189,
            "final_z_swe_probe_r2": base_core_rows[0]["swe_probe_val_r2"],
            "final_z_cpm_probe_r2": base_core_rows[0]["cpm_probe_val_r2"],
            "final_z_aqm_probe_r2": base_core_rows[0]["aqm_probe_val_r2"],
        },
        {
            "model": "D1",
            "aux_location": "final_z",
            "swe_val_r2": 0.2030303471510383,
            "swe_rmse": 30.57939338684082,
            "swe_mae": 23.719331741333008,
            "swe_r": 0.45172828435897827,
            "final_z_swe_probe_r2": base_core_rows[1]["swe_probe_val_r2"],
            "final_z_cpm_probe_r2": base_core_rows[1]["cpm_probe_val_r2"],
            "final_z_aqm_probe_r2": base_core_rows[1]["aqm_probe_val_r2"],
        },
        {
            "model": "P1",
            "aux_location": "final_z + PCGrad",
            "swe_val_r2": 0.21919567878075163,
            "swe_rmse": 30.267675399780273,
            "swe_mae": 23.622840881347656,
            "swe_r": 0.492715448141098,
            "final_z_swe_probe_r2": base_core_rows[2]["swe_probe_val_r2"],
            "final_z_cpm_probe_r2": base_core_rows[2]["cpm_probe_val_r2"],
            "final_z_aqm_probe_r2": base_core_rows[2]["aqm_probe_val_r2"],
        },
        {
            "model": "P2",
            "aux_location": "final_z + grad similarity",
            "swe_val_r2": 0.14444766615488225,
            "swe_rmse": 31.683361053466797,
            "swe_mae": 24.91934585571289,
            "swe_r": 0.4643968343734741,
            "final_z_swe_probe_r2": base_core_rows[3]["swe_probe_val_r2"],
            "final_z_cpm_probe_r2": base_core_rows[3]["cpm_probe_val_r2"],
            "final_z_aqm_probe_r2": base_core_rows[3]["aqm_probe_val_r2"],
        },
        {
            "model": "U1",
            "aux_location": "Stage 2",
            "swe_val_r2": u1_training_summary["best_metrics"]["swe_r2"],
            "swe_rmse": u1_training_summary["best_metrics"]["swe_rmse"],
            "swe_mae": u1_training_summary["best_metrics"]["swe_mae"],
            "swe_r": u1_training_summary["best_metrics"]["swe_pearson_r"],
            "final_z_swe_probe_r2": u1_final_lookup["SWE_label"]["val_r2"],
            "final_z_cpm_probe_r2": u1_final_lookup["CPM_label"]["val_r2"],
            "final_z_aqm_probe_r2": u1_final_lookup["AQM_label"]["val_r2"],
        },
    ]
    write_csv(
        OUTPUT_ROOT / "u1_comparison_table.csv",
        comparison_table_rows,
        [
            "model",
            "aux_location",
            "swe_val_r2",
            "swe_rmse",
            "swe_mae",
            "swe_r",
            "final_z_swe_probe_r2",
            "final_z_cpm_probe_r2",
            "final_z_aqm_probe_r2",
        ],
    )

    plot_bar(
        path=OUTPUT_ROOT / "u1_final_z_probe_comparison.png",
        title="Final-Z Probe Validation R² Comparison",
        xlabel="Model",
        ylabel="Validation R²",
        categories=[row["model"] for row in comparison_table_rows],
        series={
            "SWE": [float(row["final_z_swe_probe_r2"]) for row in comparison_table_rows],
            "CPM": [float(row["final_z_cpm_probe_r2"]) for row in comparison_table_rows],
            "AQM": [float(row["final_z_aqm_probe_r2"]) for row in comparison_table_rows],
        },
    )
    plot_bar(
        path=OUTPUT_ROOT / "u1_stage2_probe_results.png",
        title="U1 Stage-2 Probe Validation R²",
        xlabel="Target",
        ylabel="Validation R²",
        categories=["SWE", "CPM", "AQM"],
        series={
            "U1 Stage2": [
                float(stage2_lookup["SWE_label"]["val_r2"]),
                float(stage2_lookup["CPM_label"]["val_r2"]),
                float(stage2_lookup["AQM_label"]["val_r2"]),
            ]
        },
    )

    summary = {
        "u1_experiment_dir": str(U1_EXPERIMENT),
        "u1_training_summary": u1_training_summary,
        "u1_final_z_probe": final_z_summary["final_z"],
        "u1_stage2_probe": final_z_summary["stage2"],
        "updated_cka_labels": labels,
        "updated_cka_matrix": cka_matrix.tolist(),
    }
    with (OUTPUT_ROOT / "u1_diagnostics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    _log(f"Finished U1 upstream diagnostics: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
