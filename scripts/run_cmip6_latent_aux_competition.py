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

from snow_ml.cmip6_cnn_experiment import (
    FEATURE_Z_CLIP,
    build_model,
    r2_score,
    read_manifest,
    set_global_seed,
)


PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "cmip6_latent_aux_competition_v1"
LATENT_ARRAY_DIR = OUTPUT_ROOT / "latent_arrays"

EXPERIMENTS = {
    "S0": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/"
        "cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only"
    ),
    "D1": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/"
        "cmip6_aux_diag_v1/experiments/D1_static_cnn_cpm_aqm_diag"
    ),
    "P1": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/"
        "cmip6_aux_methods_v1/experiments/P1_static_cnn_pcgrad"
    ),
    "P2": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/"
        "cmip6_aux_methods_v1/experiments/P2_static_cnn_gradsim"
    ),
}

TARGET_NAMES = ("SWE_label", "CPM_label", "AQM_label")
TARGET_INDEX = {name: idx for idx, name in enumerate(TARGET_NAMES)}
PROBE_ALPHAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)


@dataclass
class RidgeModel:
    alpha: float
    intercept: float
    coef_raw: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.coef_raw + self.intercept


def _log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_shared_setup() -> tuple[dict[str, Any], dict[str, list[int]], Path]:
    reference_config: dict[str, Any] | None = None
    reference_split: dict[str, list[int]] | None = None
    reference_cache_dir: Path | None = None
    reference_split_payload: str | None = None
    for label, exp_dir in EXPERIMENTS.items():
        config = load_json(exp_dir / "config.json")
        split = load_json(exp_dir / "split.json")
        cache_dir = Path(config["predictor_cache_dir"])
        split_payload = json.dumps(split, sort_keys=True)
        if reference_config is None:
            reference_config = config
            reference_split = split
            reference_cache_dir = cache_dir
            reference_split_payload = split_payload
            continue
        assert cache_dir == reference_cache_dir, f"{label} cache dir mismatch"
        assert split_payload == reference_split_payload, f"{label} split mismatch"
    assert reference_config is not None
    assert reference_split is not None
    assert reference_cache_dir is not None
    return reference_config, reference_split, reference_cache_dir


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
    stacked = np.concatenate([x, mask], axis=2)
    return stacked.astype(np.float32)


def load_encoder_latents(
    *,
    label: str,
    exp_dir: Path,
    architecture: str,
    physical_data: np.memmap,
    valid_mask: np.memmap,
    feature_mu: np.ndarray,
    feature_sigma: np.ndarray,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    config = load_json(exp_dir / "config.json")
    checkpoint = torch.load(exp_dir / "best_checkpoint.pt", map_location="cpu")
    model = build_model(architecture, in_channels_per_month=int(physical_data.shape[2] * 2))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    z_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, physical_data.shape[0], batch_size):
            stop = start + batch_size
            batch_inputs = build_inputs(
                physical_data=np.asarray(physical_data[start:stop], dtype=np.float32),
                valid_mask=np.asarray(valid_mask[start:stop], dtype=np.float32),
                feature_mu=feature_mu,
                feature_sigma=feature_sigma,
            )
            batch = torch.from_numpy(batch_inputs).to(device)
            z = model.encode(batch).detach().cpu().numpy().astype(np.float32)
            z_chunks.append(z)
    z_full = np.concatenate(z_chunks, axis=0)
    z_flat = z_full.reshape(z_full.shape[0], -1)
    np.savez_compressed(
        LATENT_ARRAY_DIR / f"{label}_latents.npz",
        z=z_full,
        z_flat=z_flat,
        architecture=architecture,
        checkpoint_path=str(exp_dir / "best_checkpoint.pt"),
        experiment_dir=str(exp_dir),
        predictor_cache_dir=config["predictor_cache_dir"],
    )
    return z_full, z_flat


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


def fit_ridge(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_years: np.ndarray,
    alphas: np.ndarray = PROBE_ALPHAS,
) -> RidgeModel:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    folds = grouped_year_folds(train_years)
    best_alpha = float(alphas[0])
    best_score = float("inf")
    for alpha in alphas:
        fold_errors: list[float] = []
        for fold_train_idx, fold_val_idx in folds:
            model = fit_ridge_fixed_alpha(
                x_train=x_train[fold_train_idx],
                y_train=y_train[fold_train_idx],
                alpha=float(alpha),
            )
            pred = model.predict(x_train[fold_val_idx])
            fold_errors.append(float(np.mean((pred - y_train[fold_val_idx]) ** 2)))
        if fold_errors:
            score = float(np.mean(fold_errors))
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
    return fit_ridge_fixed_alpha(x_train=x_train, y_train=y_train, alpha=best_alpha)


def fit_ridge_fixed_alpha(*, x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> RidgeModel:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_mean = x_train.mean(axis=0, dtype=np.float64)
    x_scale = x_train.std(axis=0, dtype=np.float64)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)
    x_std = (x_train - x_mean) / x_scale
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    xtx = x_std.T @ x_std
    xty = x_std.T @ y_centered
    coef_std = np.linalg.solve(
        xtx + (alpha * np.eye(x_std.shape[1], dtype=np.float64)),
        xty,
    )
    coef_raw = coef_std / x_scale
    intercept = y_mean - float(x_mean @ coef_raw)
    return RidgeModel(
        alpha=float(alpha),
        intercept=intercept,
        coef_raw=coef_raw.astype(np.float64),
        x_mean=x_mean.astype(np.float64),
        x_scale=x_scale.astype(np.float64),
    )


def predict_ridge(model: RidgeModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ model.coef_raw + model.intercept


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "r2": r2_score(y_true.astype(np.float32), y_pred.astype(np.float32)),
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "pearson_r": pearson_r_np(y_true, y_pred),
    }


def pearson_r_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    denom = math.sqrt(float(np.sum(y_true_centered**2) * np.sum(y_pred_centered**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(y_true_centered * y_pred_centered) / denom)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


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


def remove_single_direction(z: np.ndarray, direction: np.ndarray) -> np.ndarray:
    denom = float(direction @ direction)
    assert denom > 1e-12, "direction norm too small"
    projection = (z @ direction)[:, None] * (direction[None, :] / denom)
    return z - projection


def remove_joint_subspace(z: np.ndarray, directions: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(directions)
    diag_abs = np.abs(np.diag(r))
    keep = diag_abs > 1e-12
    q = q[:, keep]
    assert q.shape[1] > 0, "joint subspace basis is empty"
    return z - (z @ q) @ q.T


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def plot_cka_heatmap(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Linear CKA of Frozen Latent Representations")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_direction_removal(path: Path, rows: list[dict[str, Any]]) -> None:
    encoders = ["S0", "D1", "P1", "P2"]
    reps = ["original", "cpm_removed", "aqm_removed", "cpm_aqm_removed"]
    lookup = {(row["encoder"], row["representation"]): float(row["swe_val_r2"]) for row in rows}
    series = {rep: [lookup[(enc, rep)] for enc in encoders] for rep in reps}
    plot_bar(
        path=path,
        title="SWE Probe Validation R² Before/After Direction Removal",
        xlabel="Encoder",
        ylabel="Validation R²",
        categories=encoders,
        series=series,
    )


def main() -> None:
    set_global_seed(20260818)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LATENT_ARRAY_DIR.mkdir(parents=True, exist_ok=True)

    _, split, cache_dir = verify_shared_setup()
    manifest = read_manifest(cache_dir / "manifest.csv")
    targets = np.load(cache_dir / "targets.npy")
    physical_data = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    valid_mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    water_years = np.asarray([int(row["water_year"]) for row in manifest], dtype=np.int64)
    train_years = water_years[train_idx]
    val_years = water_years[val_idx]
    _log(f"Loaded shared cache with {len(manifest)} samples, train={train_idx.size}, val={val_idx.size}")

    latents_flat: dict[str, np.ndarray] = {}

    for label, exp_dir in EXPERIMENTS.items():
        config = load_json(exp_dir / "config.json")
        norm = np.load(exp_dir / "normalization_stats.npz")
        feature_mu = np.asarray(norm["feature_mu"], dtype=np.float32)
        feature_sigma = np.asarray(norm["feature_sigma"], dtype=np.float32)
        z, z_flat = load_encoder_latents(
            label=label,
            exp_dir=exp_dir,
            architecture=config["architecture"],
            physical_data=physical_data,
            valid_mask=valid_mask,
            feature_mu=feature_mu,
            feature_sigma=feature_sigma,
        )
        latents_flat[label] = z_flat
        _log(f"Extracted latents for {label}: {z.shape} -> {z_flat.shape}")

    linear_probe_rows: list[dict[str, Any]] = []
    probe_summary: dict[str, Any] = {}
    direction_geometry_rows: list[dict[str, Any]] = []
    direction_removal_rows: list[dict[str, Any]] = []
    cached_probe_models: dict[tuple[str, str], RidgeModel] = {}

    for label in EXPERIMENTS:
        z_train = latents_flat[label][train_idx]
        z_val = latents_flat[label][val_idx]
        probe_summary[label] = {"targets": {}}
        target_models: dict[str, RidgeModel] = {}
        for target_name in TARGET_NAMES:
            y_train = targets[train_idx, TARGET_INDEX[target_name]].astype(np.float64)
            y_val = targets[val_idx, TARGET_INDEX[target_name]].astype(np.float64)
            model = fit_ridge(x_train=z_train, y_train=y_train, train_years=train_years)
            target_models[target_name] = model
            cached_probe_models[(label, target_name)] = model
            train_pred = predict_ridge(model, z_train)
            val_pred = predict_ridge(model, z_val)
            train_metrics = regression_metrics(y_train, train_pred)
            val_metrics = regression_metrics(y_val, val_pred)
            row = {
                "encoder": label,
                "target": target_name,
                "alpha": model.alpha,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_pearson_r": val_metrics["pearson_r"],
                "train_pearson_r": train_metrics["pearson_r"],
            }
            linear_probe_rows.append(row)
            probe_summary[label]["targets"][target_name] = {
                "alpha": model.alpha,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

        w_s = target_models["SWE_label"].coef_raw
        w_c = target_models["CPM_label"].coef_raw
        w_a = target_models["AQM_label"].coef_raw
        direction_geometry_rows.extend(
            [
                {
                    "encoder": label,
                    "cosine_name": "cos_wS_wC",
                    "value": cosine_similarity(w_s, w_c),
                },
                {
                    "encoder": label,
                    "cosine_name": "cos_wS_wA",
                    "value": cosine_similarity(w_s, w_a),
                },
                {
                    "encoder": label,
                    "cosine_name": "cos_wC_wA",
                    "value": cosine_similarity(w_c, w_a),
                },
            ]
        )

        representations = {
            "original": latents_flat[label],
            "cpm_removed": remove_single_direction(latents_flat[label], w_c),
            "aqm_removed": remove_single_direction(latents_flat[label], w_a),
            "cpm_aqm_removed": remove_joint_subspace(latents_flat[label], np.column_stack([w_c, w_a])),
        }
        original_val_r2: float | None = None
        for rep_name, rep in representations.items():
            model = fit_ridge(
                x_train=rep[train_idx],
                y_train=targets[train_idx, TARGET_INDEX["SWE_label"]].astype(np.float64),
                train_years=train_years,
            )
            val_pred = predict_ridge(model, rep[val_idx])
            val_metrics = regression_metrics(
                targets[val_idx, TARGET_INDEX["SWE_label"]].astype(np.float64),
                val_pred,
            )
            if rep_name == "original":
                original_val_r2 = val_metrics["r2"]
            assert original_val_r2 is not None or rep_name == "original"
            direction_removal_rows.append(
                {
                    "encoder": label,
                    "representation": rep_name,
                    "alpha": model.alpha,
                    "swe_val_r2": val_metrics["r2"],
                    "swe_val_rmse": val_metrics["rmse"],
                    "swe_val_mae": val_metrics["mae"],
                    "swe_val_r": val_metrics["pearson_r"],
                    "delta_r2_vs_original": 0.0 if rep_name == "original" else (val_metrics["r2"] - float(original_val_r2)),
                }
            )

    cka_labels = list(EXPERIMENTS.keys())
    cka_matrix = np.zeros((len(cka_labels), len(cka_labels)), dtype=np.float64)
    for i, label_i in enumerate(cka_labels):
        for j, label_j in enumerate(cka_labels):
            cka_matrix[i, j] = linear_cka(latents_flat[label_i][val_idx], latents_flat[label_j][val_idx])

    core_probe_table: list[dict[str, Any]] = []
    for label in EXPERIMENTS:
        swe_row = next(row for row in linear_probe_rows if row["encoder"] == label and row["target"] == "SWE_label")
        cpm_row = next(row for row in linear_probe_rows if row["encoder"] == label and row["target"] == "CPM_label")
        aqm_row = next(row for row in linear_probe_rows if row["encoder"] == label and row["target"] == "AQM_label")
        core_probe_table.append(
            {
                "encoder": label,
                "swe_probe_val_r2": swe_row["val_r2"],
                "swe_probe_rmse": swe_row["val_rmse"],
                "swe_probe_r": swe_row["val_pearson_r"],
                "cpm_probe_val_r2": cpm_row["val_r2"],
                "cpm_probe_r": cpm_row["val_pearson_r"],
                "aqm_probe_val_r2": aqm_row["val_r2"],
                "aqm_probe_r": aqm_row["val_pearson_r"],
            }
        )

    write_csv(
        OUTPUT_ROOT / "linear_probe_results.csv",
        linear_probe_rows,
        [
            "encoder",
            "target",
            "alpha",
            "train_r2",
            "val_r2",
            "val_rmse",
            "val_mae",
            "val_pearson_r",
            "train_pearson_r",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "linear_probe_core_table.csv",
        core_probe_table,
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
    with (OUTPUT_ROOT / "linear_probe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(probe_summary, handle, indent=2, sort_keys=True)

    cka_rows = []
    for i, label_i in enumerate(cka_labels):
        row = {"encoder": label_i}
        for j, label_j in enumerate(cka_labels):
            row[label_j] = float(cka_matrix[i, j])
        cka_rows.append(row)
    write_csv(
        OUTPUT_ROOT / "cka_matrix.csv",
        cka_rows,
        ["encoder", *cka_labels],
    )
    plot_cka_heatmap(OUTPUT_ROOT / "cka_heatmap.png", cka_labels, cka_matrix)

    write_csv(
        OUTPUT_ROOT / "direction_geometry.csv",
        direction_geometry_rows,
        ["encoder", "cosine_name", "value"],
    )
    write_csv(
        OUTPUT_ROOT / "direction_removal_results.csv",
        direction_removal_rows,
        [
            "encoder",
            "representation",
            "alpha",
            "swe_val_r2",
            "swe_val_rmse",
            "swe_val_mae",
            "swe_val_r",
            "delta_r2_vs_original",
        ],
    )
    direction_removal_summary = {"rows": direction_removal_rows}
    with (OUTPUT_ROOT / "direction_removal_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(direction_removal_summary, handle, indent=2, sort_keys=True)

    plot_bar(
        path=OUTPUT_ROOT / "frozen_linear_probe_swe_r2.png",
        title="Frozen SWE Probe Validation R² Across Encoders",
        xlabel="Encoder",
        ylabel="Validation R²",
        categories=[row["encoder"] for row in core_probe_table],
        series={"SWE": [float(row["swe_probe_val_r2"]) for row in core_probe_table]},
    )
    plot_bar(
        path=OUTPUT_ROOT / "frozen_linear_probe_cpm_aqm.png",
        title="Frozen CPM/AQM Probe Validation R² Across Encoders",
        xlabel="Encoder",
        ylabel="Validation R²",
        categories=[row["encoder"] for row in core_probe_table],
        series={
            "CPM": [float(row["cpm_probe_val_r2"]) for row in core_probe_table],
            "AQM": [float(row["aqm_probe_val_r2"]) for row in core_probe_table],
        },
    )
    plot_direction_removal(OUTPUT_ROOT / "swe_r2_direction_removal.png", direction_removal_rows)

    # Dedicated single-factor removal plots.
    for rep_name, file_name, title in [
        ("cpm_removed", "swe_r2_cpm_removal.png", "SWE Probe R² Before/After CPM Direction Removal"),
        ("aqm_removed", "swe_r2_aqm_removal.png", "SWE Probe R² Before/After AQM Direction Removal"),
        ("cpm_aqm_removed", "swe_r2_joint_removal.png", "SWE Probe R² Before/After CPM+AQM Subspace Removal"),
    ]:
        rows = [row for row in direction_removal_rows if row["representation"] in {"original", rep_name}]
        lookup = {(row["encoder"], row["representation"]): float(row["swe_val_r2"]) for row in rows}
        categories = list(EXPERIMENTS.keys())
        plot_bar(
            path=OUTPUT_ROOT / file_name,
            title=title,
            xlabel="Encoder",
            ylabel="Validation R²",
            categories=categories,
            series={
                "original": [lookup[(enc, "original")] for enc in categories],
                rep_name: [lookup[(enc, rep_name)] for enc in categories],
            },
        )

    manifest_rows = [
        {
            "sample_index": idx,
            "split": "train" if idx in set(split["train"]) else "val",
            **manifest[idx],
        }
        for idx in range(len(manifest))
    ]
    write_csv(
        LATENT_ARRAY_DIR / "sample_manifest.csv",
        manifest_rows,
        list(manifest_rows[0].keys()),
    )

    analysis_summary = {
        "output_dir": str(OUTPUT_ROOT),
        "experiments": {label: str(path) for label, path in EXPERIMENTS.items()},
        "sample_count": len(manifest),
        "train_count": int(train_idx.size),
        "val_count": int(val_idx.size),
        "train_water_years": sorted({int(year) for year in train_years.tolist()}),
        "val_water_years": sorted({int(year) for year in val_years.tolist()}),
        "core_probe_table": core_probe_table,
        "cka_labels": cka_labels,
        "cka_matrix": cka_matrix.tolist(),
    }
    with (OUTPUT_ROOT / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis_summary, handle, indent=2, sort_keys=True)

    _log(f"Finished latent auxiliary competition diagnostics: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
