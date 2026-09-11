#!/usr/bin/env python3
"""
Strict LOYO undercomplete denoising autoencoder experiment for Sierra SWE.
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direct_sst_sparse_common import (  # noqa: E402
    MONTHS,
    WATER_YEARS,
    build_raw_sst_cube,
    compute_metric_bundle,
    dump_json,
    load_target_table,
    standardize_target_train_only,
    standardize_train_only,
)


BASELINE_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
)
BASELINE_PREDICTIONS_CSV = BASELINE_ARTIFACT_DIR / "z1z2_amv_k5_loyo_predictions.csv"
BASELINE_PREDICTOR_TABLE_CSV = BASELINE_ARTIFACT_DIR / "z1z2_amv_k5_predictor_table.csv"
UPSTREAM_SWE_TARGET_NC = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "undercomplete_dae_sst_loyo_k7_k10_k15"
SST_SOURCE_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")

RIDGE_ALPHA_GRID = np.asarray(
    [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0],
    dtype=np.float64,
)
MAX_LATENT_DIM = 15


class UndercompleteDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, activation_name: str) -> None:
        super().__init__()
        self.activation_name = activation_name
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            make_activation(activation_name),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            make_activation(activation_name),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError("Unsupported activation: {}".format(name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[7, 10, 15])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-prob", type=float, default=0.0)
    parser.add_argument("--corruption", choices=["gaussian", "gaussian_mask"], default="gaussian")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def validate_args(args: argparse.Namespace) -> None:
    bad = [k for k in args.latent_dims if k <= 0 or k > MAX_LATENT_DIM]
    if bad:
        raise ValueError("Latent dimensions must lie in [1, {}]: {}".format(MAX_LATENT_DIM, bad))
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if args.noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if not 0.0 <= args.mask_prob < 1.0:
        raise ValueError("mask_prob must be in [0, 1).")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5).")


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_baseline_predictions() -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred = pd.read_csv(BASELINE_PREDICTIONS_CSV)
    pred = pred[pred["model_name"] == "Z1_Z2_AMV_AMO_K5"].copy()
    pred = pred.rename(columns={"heldout_wy": "water_year", "pred_swe": "y_pred_baseline"})
    pred["water_year"] = pred["water_year"].astype(int)
    pred = pred[["water_year", "obs_swe", "y_pred_baseline"]].sort_values("water_year").reset_index(drop=True)
    pred["squared_error_baseline"] = (pred["y_pred_baseline"] - pred["obs_swe"]) ** 2
    if pred["water_year"].tolist() != WATER_YEARS.tolist():
        raise ValueError("Baseline predictions do not match WY1985--WY2021.")
    metrics = compute_metric_bundle(
        pred["obs_swe"].to_numpy(dtype=float),
        pred["y_pred_baseline"].to_numpy(dtype=float),
    )
    return pred, {
        "mse": float(np.mean(pred["squared_error_baseline"].to_numpy(dtype=float))),
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "r": metrics["r"],
        "r2": metrics["R2"],
        "sign_accuracy": metrics["sign_accuracy"],
    }


def prepare_weighted_features(
    train_cube_raw: np.ndarray,
    test_cube_raw: np.ndarray,
    valid_feature_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_flat = valid_feature_mask.reshape(-1)
    train_valid = train_cube_raw.reshape(train_cube_raw.shape[0], -1)[:, valid_flat]
    test_valid = test_cube_raw.reshape(-1)[valid_flat]
    monthly_clim = np.mean(train_valid, axis=0)
    train_anom = train_valid - monthly_clim[None, :]
    test_anom = test_valid - monthly_clim
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    expanded_weights = np.tile(np.repeat(lat_weights, lon.size), len(MONTHS))[valid_flat]
    x_train = train_anom * expanded_weights[None, :]
    x_test = test_anom * expanded_weights
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
        raise ValueError("Non-finite values entered the weighted SST predictor matrix.")
    return x_train, x_test


def choose_validation_indices(n_train: int, heldout_year: int, seed: int, fraction: float) -> np.ndarray:
    n_val = max(5, int(round(fraction * n_train)))
    n_val = min(n_val, n_train - 1)
    rng = np.random.RandomState(seed * 1000 + heldout_year)
    return np.sort(rng.choice(n_train, size=n_val, replace=False))


def apply_corruption(
    x_clean: torch.Tensor,
    *,
    noise_std: float,
    corruption: str,
    mask_prob: float,
    generator: torch.Generator,
) -> torch.Tensor:
    x_noisy = x_clean
    if noise_std > 0.0:
        noise = torch.randn(
            x_clean.shape,
            generator=generator,
            device=x_clean.device,
            dtype=x_clean.dtype,
        )
        x_noisy = x_noisy + noise_std * noise
    if corruption == "gaussian_mask" and mask_prob > 0.0:
        keep = torch.rand(
            x_clean.shape,
            generator=generator,
            device=x_clean.device,
            dtype=x_clean.dtype,
        ) >= mask_prob
        x_noisy = x_noisy * keep
    return x_noisy


def reconstruction_mse(model: UndercompleteDenoisingAutoencoder, x: np.ndarray, device: torch.device) -> float:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return float(torch.mean((pred - tensor) ** 2).item())


def encode_numpy(model: UndercompleteDenoisingAutoencoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        z = model.encode(tensor)
    return z.cpu().numpy().astype(np.float64)


def train_dae(
    *,
    x_train_std: np.ndarray,
    heldout_year: int,
    seed: int,
    latent_dim: int,
    hidden_dim: int,
    activation: str,
    noise_std: float,
    corruption: str,
    mask_prob: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    validation_fraction: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[UndercompleteDenoisingAutoencoder, Dict[str, float], np.ndarray]:
    n_train = x_train_std.shape[0]
    val_idx = choose_validation_indices(n_train, heldout_year, seed, validation_fraction)
    train_mask = np.ones(n_train, dtype=bool)
    train_mask[val_idx] = False
    x_subtrain = np.asarray(x_train_std[train_mask], dtype=np.float32)
    x_val = np.asarray(x_train_std[val_idx], dtype=np.float32)

    model = UndercompleteDenoisingAutoencoder(
        input_dim=x_train_std.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        activation_name=activation,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_tensor = torch.from_numpy(x_subtrain).to(device)
    val_tensor = torch.from_numpy(x_val).to(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed * 1000 + heldout_year + latent_dim)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_trained = 0
    effective_batch = min(batch_size, x_subtrain.shape[0])

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_tensor.shape[0], generator=generator, device=device)
        for start in range(0, train_tensor.shape[0], effective_batch):
            batch_idx = permutation[start : start + effective_batch]
            clean = train_tensor[batch_idx]
            noisy = apply_corruption(
                clean,
                noise_std=noise_std,
                corruption=corruption,
                mask_prob=mask_prob,
                generator=generator,
            )
            optimizer.zero_grad()
            recon = model(noisy)
            loss = torch.mean((recon - clean) ** 2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = float(torch.mean((val_recon - val_tensor) ** 2).item())
        epochs_trained = epoch
        if val_loss < best_val - 1.0e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    stats = {
        "validation_reconstruction_mse": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
    }
    return model, stats, val_idx


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0], dtype=np.float64), rhs)


def inner_loyo_best_alpha(z_train_raw: np.ndarray, y_train_raw: np.ndarray) -> float:
    n_train = z_train_raw.shape[0]
    best_alpha: Optional[float] = None
    best_mse: Optional[float] = None
    for alpha in RIDGE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            z_inner_train = z_train_raw[mask]
            z_inner_valid = z_train_raw[~mask][0]
            y_inner_train = y_train_raw[mask]
            z_inner_train_std, z_inner_valid_std, _, _ = standardize_train_only(z_inner_train, z_inner_valid)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train)
            beta_std = fit_ridge_standardized(z_inner_train_std, y_inner_train_std, float(alpha))
            pred_std = float(z_inner_valid_std @ beta_std)
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = float(mse)
    if best_alpha is None:
        raise RuntimeError("Failed to select a ridge alpha.")
    return best_alpha


def predict_outer_ridge(z_train_raw: np.ndarray, y_train_raw: np.ndarray, z_test_raw: np.ndarray, alpha: float) -> float:
    z_train_std, z_test_std, _, _ = standardize_train_only(z_train_raw, z_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(z_train_std, y_train_std, alpha)
    pred_std = float(z_test_std @ beta_std)
    return float(y_mean + y_std * pred_std)


def compute_metrics_row(y_obs: np.ndarray, y_pred: np.ndarray, delta_sq: np.ndarray) -> Dict[str, float]:
    metrics = compute_metric_bundle(y_obs, y_pred)
    sq = (y_pred - y_obs) ** 2
    return {
        "mse": float(np.mean(sq)),
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "r": metrics["r"],
        "r2": metrics["R2"],
        "sign_accuracy": metrics["sign_accuracy"],
        "mean_delta_squared_error_vs_baseline": float(np.mean(delta_sq)),
    }


def plot_observed_vs_predicted(obs: np.ndarray, pred: np.ndarray, baseline_pred: np.ndarray, k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(obs, baseline_pred, color="#bdbdbd", alpha=0.6, s=40, label="Baseline")
    ax.scatter(obs, pred, color="#1f77b4", alpha=0.8, s=45, label="DAE k={}".format(k))
    lo = float(min(np.min(obs), np.min(pred), np.min(baseline_pred)))
    hi = float(max(np.max(obs), np.max(pred), np.max(baseline_pred)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed April 1 Sierra SWE anomaly (m)")
    ax.set_ylabel("Predicted anomaly (m)")
    ax.set_title("Observed vs predicted: DAE k={}".format(k))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(years: np.ndarray, obs: np.ndarray, pred: np.ndarray, baseline_pred: np.ndarray, k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(years, obs, color="black", linewidth=2.2, label="Observed")
    ax.plot(years, baseline_pred, color="#9e9e9e", linewidth=1.6, label="Baseline")
    ax.plot(years, pred, color="#1f77b4", linewidth=1.8, label="DAE k={}".format(k))
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO predictions: DAE k={}".format(k))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_comparison(baseline_rmse: float, aggregate_df: pd.DataFrame, out_path: Path) -> None:
    labels = ["baseline"] + ["k={}".format(int(k)) for k in aggregate_df["k"].tolist()]
    values = [baseline_rmse] + aggregate_df["rmse_mean"].tolist()
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(labels, values, color=["#9e9e9e", "#4c78a8", "#72b7b2", "#f58518"][: len(labels)])
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE comparison vs baseline")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_squared_error_by_year(delta_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    colors = {7: "#4c78a8", 10: "#72b7b2", 15: "#f58518"}
    for k, sub in delta_df.groupby("k"):
        ax.plot(
            sub["water_year"].to_numpy(dtype=int),
            sub["delta_squared_error_dae_minus_baseline"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            color=colors.get(int(k), None),
            label="k={}".format(int(k)),
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Water year")
    ax.set_ylabel("Mean delta squared error")
    ax.set_title("Per-year DAE minus baseline squared error")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(baseline_metrics: Dict[str, float], aggregate_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["rmse", "r2", "r", "sign_accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ["#9e9e9e", "#4c78a8", "#72b7b2", "#f58518"]
    for ax, metric in zip(axes.flat, metrics):
        labels = ["baseline"] + ["k={}".format(int(k)) for k in aggregate_df["k"].tolist()]
        values = [baseline_metrics[metric]] + aggregate_df["{}_mean".format(metric)].tolist()
        ax.bar(labels, values, color=colors[: len(labels)])
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_readme(
    *,
    out_path: Path,
    args: argparse.Namespace,
    metadata: Dict[str, object],
    baseline_metrics: Dict[str, float],
    aggregate_df: pd.DataFrame,
) -> None:
    agg = aggregate_df.set_index("k")
    lines = [
        "# Undercomplete DAE SST LOYO experiment",
        "",
        "- source full SST predictor file used: `{}`".format(metadata["source_sst_file"]),
        "- source SWE target file used: `{}`".format(metadata["source_target_file"]),
        "- upstream SWE target provenance: `{}`".format(metadata["upstream_target_file"]),
        "- baseline artifact path: `{}`".format(BASELINE_ARTIFACT_DIR),
        "- exact leakage rule: the held-out water year is excluded from SST anomaly construction, SST standardization, DAE training, DAE validation, DAE early stopping, ridge fitting, ridge lambda selection, and any seed-level aggregation decisions; it is only passed once through the frozen encoder and outer-fold ridge predictor.",
        "- DAE architecture: `P -> {} -> k -> {} -> P` with `{}` activations and linear output.".format(
            args.hidden_dim, args.hidden_dim, args.activation
        ),
        "- denoising corruption rule: `{}` with Gaussian noise std `{}` and mask probability `{}`.".format(
            args.corruption, args.noise_std, args.mask_prob
        ),
        "- ridge lambda selection rule: nested training-only inner LOYO over `{}` with lower-alpha tie-breaks.".format(
            RIDGE_ALPHA_GRID.tolist()
        ),
        "- final baseline RMSE: `{:.6f}`".format(baseline_metrics["rmse"]),
    ]
    for k in args.latent_dims:
        row = agg.loc[int(k)]
        improved = "yes" if float(row["mean_delta_squared_error_vs_baseline_mean"]) < 0.0 else "no"
        lines.extend(
            [
                "- final DAE RMSE for k={}: `{:.6f}`".format(int(k), float(row["rmse_mean"])),
                "- mean delta squared error for k={}: `{:.6f}`".format(
                    int(k), float(row["mean_delta_squared_error_vs_baseline_mean"])
                ),
                "- whether k={} improved over baseline: `{}`".format(int(k), improved),
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_runtime_on_compute_node()
    device = choose_device(args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "dae_loyo_predictions.csv"
    metrics_by_seed_csv = output_dir / "dae_metrics_by_k_seed.csv"
    aggregate_csv = output_dir / "dae_metrics_by_k_aggregate.csv"
    reconstruction_csv = output_dir / "dae_fold_reconstruction.csv"
    latent_csv = output_dir / "dae_latent_codes.csv"
    metadata_json = output_dir / "dae_run_metadata.json"
    readme_md = output_dir / "README.md"

    baseline_pred_df, baseline_metrics = load_baseline_predictions()
    target_df = load_target_table()
    if not np.allclose(
        baseline_pred_df["obs_swe"].to_numpy(dtype=float),
        target_df["obs_swe"].to_numpy(dtype=float),
        equal_nan=False,
    ):
        raise ValueError("Baseline target values and canonical target table do not match.")

    cube, lat, lon, cube_metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(cube_metadata["valid_feature_mask"], dtype=bool)
    y_all = target_df["obs_swe"].to_numpy(dtype=np.float64)
    years = target_df["water_year"].to_numpy(dtype=int)

    prediction_rows: List[Dict[str, object]] = []
    reconstruction_rows: List[Dict[str, object]] = []
    latent_rows: List[Dict[str, object]] = []
    metrics_rows: List[Dict[str, object]] = [
        {
            "model": "baseline_z1z2_amv_k5",
            "k": np.nan,
            "seed": np.nan,
            "mse": baseline_metrics["mse"],
            "rmse": baseline_metrics["rmse"],
            "mae": baseline_metrics["mae"],
            "r": baseline_metrics["r"],
            "r2": baseline_metrics["r2"],
            "sign_accuracy": baseline_metrics["sign_accuracy"],
            "mean_delta_squared_error_vs_baseline": 0.0,
        }
    ]

    for k in args.latent_dims:
        for seed in args.seeds:
            preds = np.full(years.shape, np.nan, dtype=np.float64)
            deltas = np.full(years.shape, np.nan, dtype=np.float64)
            for outer_idx, heldout_year in enumerate(years):
                train_mask = np.ones(years.size, dtype=bool)
                train_mask[outer_idx] = False
                x_train_raw, x_test_raw = prepare_weighted_features(
                    cube[train_mask],
                    cube[outer_idx],
                    valid_feature_mask,
                    lat,
                    lon,
                )
                x_train_std, x_test_std, _, _ = standardize_train_only(x_train_raw, x_test_raw)
                model, dae_stats, _ = train_dae(
                    x_train_std=x_train_std,
                    heldout_year=int(heldout_year),
                    seed=int(seed),
                    latent_dim=int(k),
                    hidden_dim=int(args.hidden_dim),
                    activation=str(args.activation),
                    noise_std=float(args.noise_std),
                    corruption=str(args.corruption),
                    mask_prob=float(args.mask_prob),
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                    max_epochs=int(args.max_epochs),
                    patience=int(args.patience),
                    validation_fraction=float(args.validation_fraction),
                    batch_size=int(args.batch_size),
                    device=device,
                )
                z_train = encode_numpy(model, x_train_std, device)
                z_test = encode_numpy(model, x_test_std[None, :], device)[0]
                ridge_alpha = inner_loyo_best_alpha(z_train, y_all[train_mask])
                y_pred = predict_outer_ridge(z_train, y_all[train_mask], z_test, ridge_alpha)
                baseline_row = baseline_pred_df.iloc[outer_idx]
                baseline_sq = float(baseline_row["squared_error_baseline"])
                dae_sq = float((y_pred - y_all[outer_idx]) ** 2)
                delta_sq = dae_sq - baseline_sq

                preds[outer_idx] = y_pred
                deltas[outer_idx] = delta_sq

                prediction_rows.append(
                    {
                        "water_year": int(heldout_year),
                        "k": int(k),
                        "seed": int(seed),
                        "y_obs": float(y_all[outer_idx]),
                        "y_pred_dae": float(y_pred),
                        "squared_error_dae": dae_sq,
                        "y_pred_baseline": float(baseline_row["y_pred_baseline"]),
                        "squared_error_baseline": baseline_sq,
                        "delta_squared_error_dae_minus_baseline": delta_sq,
                        "selected_ridge_lambda": float(ridge_alpha),
                    }
                )

                train_recon = reconstruction_mse(model, x_train_std, device)
                heldout_recon = reconstruction_mse(model, x_test_std[None, :], device)
                reconstruction_rows.append(
                    {
                        "heldout_year": int(heldout_year),
                        "k": int(k),
                        "seed": int(seed),
                        "train_reconstruction_mse": float(train_recon),
                        "validation_reconstruction_mse": float(dae_stats["validation_reconstruction_mse"]),
                        "heldout_reconstruction_mse": float(heldout_recon),
                        "epochs_trained": int(dae_stats["epochs_trained"]),
                        "best_epoch": int(dae_stats["best_epoch"]),
                    }
                )

                train_years = years[train_mask]
                for row_year, row_z in zip(train_years.tolist(), z_train.tolist()):
                    row = {
                        "heldout_year": int(heldout_year),
                        "water_year": int(row_year),
                        "split_role": "train",
                        "k": int(k),
                        "seed": int(seed),
                    }
                    for latent_idx in range(MAX_LATENT_DIM):
                        row["z{}".format(latent_idx + 1)] = float(row_z[latent_idx]) if latent_idx < k else np.nan
                    latent_rows.append(row)
                heldout_row = {
                    "heldout_year": int(heldout_year),
                    "water_year": int(heldout_year),
                    "split_role": "heldout",
                    "k": int(k),
                    "seed": int(seed),
                }
                for latent_idx in range(MAX_LATENT_DIM):
                    heldout_row["z{}".format(latent_idx + 1)] = float(z_test[latent_idx]) if latent_idx < k else np.nan
                latent_rows.append(heldout_row)

                print(
                    "LOYO heldout_WY={} k={} seed={} ridge_alpha={} pred={:.6f} obs={:.6f}".format(
                        int(heldout_year),
                        int(k),
                        int(seed),
                        "{:g}".format(ridge_alpha),
                        y_pred,
                        y_all[outer_idx],
                    ),
                    flush=True,
                )

            metrics_rows.append(
                {
                    "model": "dae_k{}_seed{}".format(int(k), int(seed)),
                    "k": int(k),
                    "seed": int(seed),
                    **compute_metrics_row(y_all, preds, deltas),
                }
            )

    predictions_df = pd.DataFrame(prediction_rows).sort_values(["k", "seed", "water_year"]).reset_index(drop=True)
    reconstruction_df = pd.DataFrame(reconstruction_rows).sort_values(["k", "seed", "heldout_year"]).reset_index(drop=True)
    latent_df = pd.DataFrame(latent_rows).sort_values(
        ["k", "seed", "heldout_year", "split_role", "water_year"]
    ).reset_index(drop=True)
    metrics_by_seed_df = pd.DataFrame(metrics_rows)

    aggregate_rows: List[Dict[str, object]] = []
    for k, sub in metrics_by_seed_df[metrics_by_seed_df["model"].str.startswith("dae_")].groupby("k"):
        aggregate_rows.append(
            {
                "k": int(k),
                "mse_mean": float(sub["mse"].mean()),
                "mse_median": float(sub["mse"].median()),
                "rmse_mean": float(sub["rmse"].mean()),
                "rmse_median": float(sub["rmse"].median()),
                "mae_mean": float(sub["mae"].mean()),
                "r_mean": float(sub["r"].mean()),
                "r2_mean": float(sub["r2"].mean()),
                "sign_accuracy_mean": float(sub["sign_accuracy"].mean()),
                "mean_delta_squared_error_vs_baseline_mean": float(sub["mean_delta_squared_error_vs_baseline"].mean()),
                "mean_delta_squared_error_vs_baseline_median": float(
                    sub["mean_delta_squared_error_vs_baseline"].median()
                ),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("k").reset_index(drop=True)

    predictions_df.to_csv(predictions_csv, index=False)
    metrics_by_seed_df.to_csv(metrics_by_seed_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)
    reconstruction_df.to_csv(reconstruction_csv, index=False)
    latent_df.to_csv(latent_csv, index=False)

    delta_plot_df = (
        predictions_df.groupby(["k", "water_year"], as_index=False)["delta_squared_error_dae_minus_baseline"].mean()
    )
    plot_delta_squared_error_by_year(delta_plot_df, output_dir / "dae_delta_squared_error_by_year.png")
    plot_rmse_comparison(baseline_metrics["rmse"], aggregate_df, output_dir / "dae_rmse_comparison_vs_baseline.png")
    plot_metrics_summary(baseline_metrics, aggregate_df, output_dir / "dae_metrics_summary.png")

    for k in args.latent_dims:
        sub = predictions_df[predictions_df["k"] == int(k)].copy()
        agg_pred = sub.groupby("water_year", as_index=False).agg(
            y_obs=("y_obs", "first"),
            y_pred_dae=("y_pred_dae", "mean"),
            y_pred_baseline=("y_pred_baseline", "first"),
        )
        plot_observed_vs_predicted(
            agg_pred["y_obs"].to_numpy(dtype=float),
            agg_pred["y_pred_dae"].to_numpy(dtype=float),
            agg_pred["y_pred_baseline"].to_numpy(dtype=float),
            int(k),
            output_dir / "dae_k{}_observed_vs_predicted.png".format(int(k)),
        )
        plot_timeseries(
            agg_pred["water_year"].to_numpy(dtype=int),
            agg_pred["y_obs"].to_numpy(dtype=float),
            agg_pred["y_pred_dae"].to_numpy(dtype=float),
            agg_pred["y_pred_baseline"].to_numpy(dtype=float),
            int(k),
            output_dir / "dae_k{}_timeseries_predictions.png".format(int(k)),
        )

    compact_cube_metadata = {
        "source_variable": cube_metadata.get("source_variable"),
        "months": cube_metadata.get("months"),
        "domain": cube_metadata.get("domain"),
        "feature_shape_before_mask": cube_metadata.get("feature_shape_before_mask"),
        "valid_feature_mask_shape": cube_metadata.get("valid_feature_mask_shape"),
        "original_feature_count": cube_metadata.get("original_feature_count"),
        "num_valid_features": cube_metadata.get("num_valid_features"),
        "removed_feature_count": cube_metadata.get("removed_feature_count"),
        "final_matrix_shape": cube_metadata.get("final_matrix_shape"),
        "anomaly_definition": cube_metadata.get("anomaly_definition"),
        "area_weighting": cube_metadata.get("area_weighting"),
    }
    metadata = {
        "source_sst_file": str(SST_SOURCE_FILE),
        "source_target_file": str(BASELINE_PREDICTOR_TABLE_CSV),
        "upstream_target_file": str(UPSTREAM_SWE_TARGET_NC),
        "baseline_artifact_dir": str(BASELINE_ARTIFACT_DIR),
        "water_years": years.astype(int).tolist(),
        "latent_dims": [int(k) for k in args.latent_dims],
        "seeds": [int(seed) for seed in args.seeds],
        "noise_std": float(args.noise_std),
        "hidden_dim": int(args.hidden_dim),
        "activation": str(args.activation),
        "weight_decay": float(args.weight_decay),
        "learning_rate": float(args.learning_rate),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "validation_fraction": float(args.validation_fraction),
        "batch_size": int(args.batch_size),
        "corruption": str(args.corruption),
        "mask_prob": float(args.mask_prob),
        "ridge_alpha_grid": RIDGE_ALPHA_GRID.tolist(),
        "cube_metadata": compact_cube_metadata,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    dump_json(metadata_json, metadata)
    build_readme(
        out_path=readme_md,
        args=args,
        metadata=metadata,
        baseline_metrics=baseline_metrics,
        aggregate_df=aggregate_df,
    )

    best_row = aggregate_df.sort_values("rmse_mean").iloc[0]
    any_improvement = bool(
        np.any(aggregate_df["mean_delta_squared_error_vs_baseline_mean"].to_numpy(dtype=float) < 0.0)
    )
    agg_by_k = aggregate_df.set_index("k")

    print("baseline RMSE: {:.6f}".format(baseline_metrics["rmse"]), flush=True)
    print("DAE k=7 RMSE: {:.6f}".format(float(agg_by_k.loc[7, "rmse_mean"])), flush=True)
    print("DAE k=10 RMSE: {:.6f}".format(float(agg_by_k.loc[10, "rmse_mean"])), flush=True)
    print("DAE k=15 RMSE: {:.6f}".format(float(agg_by_k.loc[15, "rmse_mean"])), flush=True)
    print(
        "mean delta squared error k=7: {:.6f}".format(
            float(agg_by_k.loc[7, "mean_delta_squared_error_vs_baseline_mean"])
        ),
        flush=True,
    )
    print(
        "mean delta squared error k=10: {:.6f}".format(
            float(agg_by_k.loc[10, "mean_delta_squared_error_vs_baseline_mean"])
        ),
        flush=True,
    )
    print(
        "mean delta squared error k=15: {:.6f}".format(
            float(agg_by_k.loc[15, "mean_delta_squared_error_vs_baseline_mean"])
        ),
        flush=True,
    )
    print("best DAE k by mean RMSE: {}".format(int(best_row["k"])), flush=True)
    print("whether any DAE improves over baseline: {}".format("yes" if any_improvement else "no"), flush=True)
    print("artifact directory: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
