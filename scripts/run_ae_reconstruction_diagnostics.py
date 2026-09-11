#!/usr/bin/env python3
"""Reconstruction-only diagnostics for the undercomplete SST autoencoder."""

from __future__ import annotations

import argparse
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
    dump_json,
    load_target_table,
)

SST_SOURCE_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ae_reconstruction_diagnostics"


class UndercompleteAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, activation_name: str) -> None:
        super().__init__()
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
        return self.decoder(self.encoder(x))


def make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[7, 10, 15, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fallback-hidden-dim", type=int, default=64)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heldout-year", type=int, default=int(WATER_YEARS[0]))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_args(args: argparse.Namespace) -> None:
    if any(k <= 0 for k in args.latent_dims):
        raise ValueError("All latent dimensions must be positive.")
    if args.hidden_dim <= 0 or args.fallback_hidden_dim <= 0:
        raise ValueError("Hidden dimensions must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive.")
    if args.max_epochs <= 0:
        raise ValueError("max-epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.noise_std < 0.0:
        raise ValueError("noise-std must be non-negative.")
    if int(args.heldout_year) not in set(WATER_YEARS.tolist()):
        raise ValueError(f"heldout-year must be one of {WATER_YEARS.tolist()}.")


def build_all_years_weighted_matrix() -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    cube, lat, lon, cube_metadata = build_raw_sst_cube()
    valid_feature_mask = np.asarray(cube_metadata["valid_feature_mask"], dtype=bool)
    valid_flat = valid_feature_mask.reshape(-1)
    cube_flat = cube.reshape(cube.shape[0], -1)[:, valid_flat]
    monthly_clim_all = np.mean(cube_flat, axis=0)
    anomalies_all = cube_flat - monthly_clim_all[None, :]
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    feature_weights = np.tile(np.repeat(lat_weights, lon.size), len(MONTHS))[valid_flat]
    x_all = anomalies_all * feature_weights[None, :]
    if not np.all(np.isfinite(x_all)):
        raise ValueError("All-years SST matrix contains non-finite values.")
    metadata = {
        "source_sst_file": str(SST_SOURCE_FILE),
        "final_matrix_shape": [int(x_all.shape[0]), int(x_all.shape[1])],
        "months": list(MONTHS),
        "water_years": WATER_YEARS.astype(int).tolist(),
        "hidden_dim_requested": None,
        "cube_metadata": {
            "domain": cube_metadata.get("domain"),
            "feature_shape_before_mask": cube_metadata.get("feature_shape_before_mask"),
            "num_valid_features": cube_metadata.get("num_valid_features"),
            "final_matrix_shape": cube_metadata.get("final_matrix_shape"),
            "anomaly_definition": "All-years monthly climatology for Sep-Mar, then area weighting by sqrt(cos(lat)).",
        },
    }
    return x_all.astype(np.float64), WATER_YEARS.astype(int).copy(), metadata


def standardize_all_rows(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1)
    if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("All-row standard deviation must be positive.")
    return (x - mean[None, :]) / std[None, :], mean, std


def standardize_train_only(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=1)
    if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("Train-only standard deviation must be positive.")
    return (x_train - mean[None, :]) / std[None, :], (x_test - mean) / std, mean, std


def reconstruction_mse(model: UndercompleteAutoencoder, x: np.ndarray, device: torch.device) -> float:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        recon = model(tensor)
    return float(torch.mean((recon - tensor) ** 2).item())


def per_row_reconstruction_mse(model: UndercompleteAutoencoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2, dim=1)
    return mse.detach().cpu().numpy().astype(np.float64)


def apply_noise(x: torch.Tensor, noise_std: float, generator: torch.Generator) -> torch.Tensor:
    if noise_std <= 0.0:
        return x
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return x + noise_std * noise


def train_autoencoder(
    *,
    x_train_std: np.ndarray,
    x_eval_std: Optional[np.ndarray],
    latent_dim: int,
    hidden_dim: int,
    activation: str,
    noise_std: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[UndercompleteAutoencoder, pd.DataFrame, Dict[str, float]]:
    model = UndercompleteAutoencoder(
        input_dim=x_train_std.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        activation_name=activation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_tensor = torch.from_numpy(np.asarray(x_train_std, dtype=np.float32)).to(device)
    eval_tensor = None if x_eval_std is None else torch.from_numpy(np.asarray(x_eval_std, dtype=np.float32)).to(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed) * 1000 + int(latent_dim) * 37 + int(hidden_dim))
    effective_batch = min(batch_size, train_tensor.shape[0])

    loss_rows: List[Dict[str, float]] = []
    best_train = float("inf")
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_tensor.shape[0], generator=generator, device=device)
        for start in range(0, train_tensor.shape[0], effective_batch):
            batch_idx = permutation[start : start + effective_batch]
            clean = train_tensor[batch_idx]
            noisy = apply_noise(clean, noise_std=noise_std, generator=generator)
            optimizer.zero_grad()
            recon = model(noisy)
            loss = torch.mean((recon - clean) ** 2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_recon = model(train_tensor)
            train_loss = float(torch.mean((train_recon - train_tensor) ** 2).item())
            if eval_tensor is None:
                eval_loss = np.nan
            else:
                eval_recon = model(eval_tensor)
                eval_loss = float(torch.mean((eval_recon - eval_tensor) ** 2).item())

        if train_loss < best_train - 1.0e-12:
            best_train = train_loss
            best_epoch = epoch

        loss_rows.append(
            {
                "epoch": int(epoch),
                "train_reconstruction_mse": float(train_loss),
                "heldout_reconstruction_mse_if_available": float(eval_loss) if np.isfinite(eval_loss) else np.nan,
            }
        )

    return model, pd.DataFrame(loss_rows), {
        "final_train_reconstruction_mse": float(loss_rows[-1]["train_reconstruction_mse"]),
        "epochs_trained": int(max_epochs),
        "best_epoch_if_available": int(best_epoch),
    }


def plot_full37_final_reconstruction(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for seed, sub in summary_df.groupby("seed", sort=True):
        ax.plot(
            sub["k"].to_numpy(dtype=int),
            sub["final_train_reconstruction_mse"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            alpha=0.75,
            label=f"seed={int(seed)}",
        )
    mean_df = (
        summary_df.groupby("k", as_index=False)["final_train_reconstruction_mse"]
        .mean()
        .sort_values("k")
    )
    ax.plot(
        mean_df["k"].to_numpy(dtype=int),
        mean_df["final_train_reconstruction_mse"].to_numpy(dtype=float),
        color="black",
        linewidth=2.5,
        marker="s",
        label="mean across seeds",
    )
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Final train reconstruction MSE")
    ax.set_title("Full-37 AE reconstruction capacity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_full37_loss_curves(loss_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    summary = (
        loss_df.groupby(["k", "epoch"], as_index=False)["train_reconstruction_mse"]
        .mean()
        .sort_values(["k", "epoch"])
    )
    for k, sub in summary.groupby("k", sort=True):
        ax.plot(
            sub["epoch"].to_numpy(dtype=int),
            sub["train_reconstruction_mse"].to_numpy(dtype=float),
            linewidth=1.8,
            label=f"k={int(k)}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train reconstruction MSE")
    ax.set_title("Full-37 training loss curves by latent dimension")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_full37_by_year(year_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    summary = (
        year_df.groupby(["k", "water_year"], as_index=False)["reconstruction_mse"]
        .mean()
        .sort_values(["k", "water_year"])
    )
    for k, sub in summary.groupby("k", sort=True):
        ax.plot(
            sub["water_year"].to_numpy(dtype=int),
            sub["reconstruction_mse"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=f"k={int(k)}",
        )
    ax.set_xlabel("Water year")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Full-37 per-year reconstruction MSE")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_onefold_train_vs_heldout(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    grouped = summary_df.groupby("k", as_index=False).agg(
        train=("final_train_reconstruction_mse", "mean"),
        heldout=("heldout_reconstruction_mse", "mean"),
    )
    ax.plot(grouped["k"], grouped["train"], marker="o", linewidth=2, label="Train")
    ax.plot(grouped["k"], grouped["heldout"], marker="o", linewidth=2, label="Held-out")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("One-fold overfit: train vs held-out reconstruction")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_onefold_loss_curves(loss_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    summary = loss_df.groupby(["k", "epoch"], as_index=False).agg(
        train_reconstruction_mse=("train_reconstruction_mse", "mean"),
        heldout_reconstruction_mse_if_available=("heldout_reconstruction_mse_if_available", "mean"),
    )
    for k, sub in summary.groupby("k", sort=True):
        axes[0].plot(
            sub["epoch"].to_numpy(dtype=int),
            sub["train_reconstruction_mse"].to_numpy(dtype=float),
            linewidth=1.6,
            label=f"k={int(k)}",
        )
        axes[1].plot(
            sub["epoch"].to_numpy(dtype=int),
            sub["heldout_reconstruction_mse_if_available"].to_numpy(dtype=float),
            linewidth=1.6,
            label=f"k={int(k)}",
        )
    axes[0].set_title("One-fold train loss curves")
    axes[1].set_title("One-fold held-out loss curves")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction MSE")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    out_path: Path,
    *,
    args: argparse.Namespace,
    metadata: Dict[str, object],
    full37_summary_df: pd.DataFrame,
    onefold_summary_df: pd.DataFrame,
) -> None:
    full37_best_k = int(
        full37_summary_df.groupby("k", as_index=False)["final_train_reconstruction_mse"]
        .mean()
        .sort_values("final_train_reconstruction_mse")
        .iloc[0]["k"]
    )
    onefold_agg = onefold_summary_df.groupby("k", as_index=False).agg(
        train=("final_train_reconstruction_mse", "mean"),
        heldout=("heldout_reconstruction_mse", "mean"),
    )
    lines = [
        "# AE reconstruction diagnostics",
        "",
        f"- source SST file: `{metadata['source_sst_file']}`",
        f"- standardized full SST matrix shape: `{metadata['final_matrix_shape'][0]} x {metadata['final_matrix_shape'][1]}`",
        f"- diagnostic artifact directory: `{out_path.parent}`",
        "- this is not a SWE prediction experiment",
        "- this diagnostic intentionally allows overfitting",
        "- Diagnostic 1 trains on all 37 SST years to test AE reconstruction capacity",
        "- Diagnostic 2 trains on 36 SST years with weak regularization to test whether the AE can overfit training rows and how large the left-out reconstruction gap remains",
        "- if reconstruction loss does not decrease with k, the AE architecture/training setup is not effectively using latent capacity",
        "- if training reconstruction improves but held-out reconstruction stays bad, the issue is generalization from tiny N, not basic AE capacity",
        "- full-37 architecture: `P -> {} -> k -> {} -> P` with `{}` activation".format(
            int(args.hidden_dim), int(args.hidden_dim), args.activation
        ),
        "- weak regularization settings: noise_std=`{}`, weight_decay=`{}`, epochs=`{}`".format(
            args.noise_std, args.weight_decay, args.max_epochs
        ),
        f"- one-fold held-out year: `{int(args.heldout_year)}`",
        "",
        "## Summary",
        "",
        f"- best k by full-37 reconstruction: `{full37_best_k}`",
    ]
    for _, row in onefold_agg.sort_values("k").iterrows():
        lines.append(
            "- one-fold k={}: train MSE `{:.6f}`, held-out MSE `{:.6f}`, gap `{:.6f}`".format(
                int(row["k"]),
                float(row["train"]),
                float(row["heldout"]),
                float(row["heldout"] - row["train"]),
            )
        )
    k7 = full37_summary_df.groupby("k", as_index=False)["final_train_reconstruction_mse"].mean().set_index("k")
    if 7 in k7.index and 30 in k7.index and float(k7.loc[30, "final_train_reconstruction_mse"]) >= float(
        k7.loc[7, "final_train_reconstruction_mse"]
    ):
        lines.extend(
            [
                "",
                "## Capacity warning",
                "",
                "- `k=30` did not improve over `k=7` in full-37 reconstruction, which suggests a possible architecture, optimization, or scaling issue.",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_runtime_on_compute_node()
    device = choose_device(args.device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all_raw, water_years, metadata = build_all_years_weighted_matrix()
    metadata["hidden_dim_requested"] = int(args.hidden_dim)
    metadata["torch_version"] = str(torch.__version__)
    metadata["device"] = str(device)
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["latent_dims"] = [int(k) for k in args.latent_dims]
    metadata["seeds"] = [int(seed) for seed in args.seeds]
    metadata["heldout_year"] = int(args.heldout_year)
    metadata["activation"] = str(args.activation)
    metadata["noise_std"] = float(args.noise_std)
    metadata["weight_decay"] = float(args.weight_decay)
    metadata["learning_rate"] = float(args.learning_rate)
    metadata["max_epochs"] = int(args.max_epochs)
    metadata["batch_size"] = int(args.batch_size)

    full37_summary_rows: List[Dict[str, object]] = []
    full37_year_rows: List[Dict[str, object]] = []
    onefold_summary_rows: List[Dict[str, object]] = []
    loss_curve_rows: List[Dict[str, object]] = []

    x_all_std, _, _ = standardize_all_rows(x_all_raw)
    heldout_idx = int(np.where(water_years == int(args.heldout_year))[0][0])
    train_mask = np.ones(x_all_raw.shape[0], dtype=bool)
    train_mask[heldout_idx] = False
    x_onefold_train_raw = x_all_raw[train_mask]
    x_onefold_heldout_raw = x_all_raw[heldout_idx]
    x_onefold_train_std, x_onefold_heldout_std, _, _ = standardize_train_only(
        x_onefold_train_raw, x_onefold_heldout_raw
    )

    for seed in args.seeds:
        for k in args.latent_dims:
            model_full37, full37_loss_df, full37_stats = train_autoencoder(
                x_train_std=x_all_std,
                x_eval_std=None,
                latent_dim=int(k),
                hidden_dim=int(args.hidden_dim),
                activation=str(args.activation),
                noise_std=float(args.noise_std),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                max_epochs=int(args.max_epochs),
                batch_size=int(args.batch_size),
                seed=int(seed),
                device=device,
            )
            full37_summary_rows.append(
                {
                    "diagnostic": "full37_reconstruction",
                    "k": int(k),
                    "seed": int(seed),
                    "final_train_reconstruction_mse": float(full37_stats["final_train_reconstruction_mse"]),
                    "epochs_trained": int(full37_stats["epochs_trained"]),
                    "best_epoch_if_available": int(full37_stats["best_epoch_if_available"]),
                }
            )
            row_mse = per_row_reconstruction_mse(model_full37, x_all_std, device)
            for year, mse in zip(water_years.tolist(), row_mse.tolist()):
                full37_year_rows.append(
                    {
                        "diagnostic": "full37_reconstruction",
                        "k": int(k),
                        "seed": int(seed),
                        "water_year": int(year),
                        "reconstruction_mse": float(mse),
                    }
                )
            for _, row in full37_loss_df.iterrows():
                loss_curve_rows.append(
                    {
                        "diagnostic": "full37_reconstruction",
                        "heldout_year_or_all37": "all37",
                        "k": int(k),
                        "seed": int(seed),
                        "epoch": int(row["epoch"]),
                        "train_reconstruction_mse": float(row["train_reconstruction_mse"]),
                        "heldout_reconstruction_mse_if_available": np.nan,
                    }
                )

            model_onefold, onefold_loss_df, onefold_stats = train_autoencoder(
                x_train_std=x_onefold_train_std,
                x_eval_std=x_onefold_heldout_std[None, :],
                latent_dim=int(k),
                hidden_dim=int(args.hidden_dim),
                activation=str(args.activation),
                noise_std=float(args.noise_std),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                max_epochs=int(args.max_epochs),
                batch_size=int(args.batch_size),
                seed=int(seed),
                device=device,
            )
            heldout_mse = reconstruction_mse(model_onefold, x_onefold_heldout_std[None, :], device)
            onefold_summary_rows.append(
                {
                    "diagnostic": "onefold_overfit",
                    "heldout_year": int(args.heldout_year),
                    "k": int(k),
                    "seed": int(seed),
                    "final_train_reconstruction_mse": float(onefold_stats["final_train_reconstruction_mse"]),
                    "heldout_reconstruction_mse": float(heldout_mse),
                    "heldout_minus_train": float(heldout_mse - onefold_stats["final_train_reconstruction_mse"]),
                    "epochs_trained": int(onefold_stats["epochs_trained"]),
                    "best_epoch_if_available": int(onefold_stats["best_epoch_if_available"]),
                }
            )
            for _, row in onefold_loss_df.iterrows():
                loss_curve_rows.append(
                    {
                        "diagnostic": "onefold_overfit",
                        "heldout_year_or_all37": int(args.heldout_year),
                        "k": int(k),
                        "seed": int(seed),
                        "epoch": int(row["epoch"]),
                        "train_reconstruction_mse": float(row["train_reconstruction_mse"]),
                        "heldout_reconstruction_mse_if_available": float(row["heldout_reconstruction_mse_if_available"]),
                    }
                )

            print(
                "diagnostic k={} seed={} full37_train_mse={:.6f} onefold_train_mse={:.6f} onefold_heldout_mse={:.6f}".format(
                    int(k),
                    int(seed),
                    float(full37_stats["final_train_reconstruction_mse"]),
                    float(onefold_stats["final_train_reconstruction_mse"]),
                    float(heldout_mse),
                ),
                flush=True,
            )

    full37_summary_df = pd.DataFrame(full37_summary_rows).sort_values(["k", "seed"]).reset_index(drop=True)
    full37_year_df = pd.DataFrame(full37_year_rows).sort_values(["k", "seed", "water_year"]).reset_index(drop=True)
    onefold_summary_df = pd.DataFrame(onefold_summary_rows).sort_values(["k", "seed"]).reset_index(drop=True)
    loss_curve_df = pd.DataFrame(loss_curve_rows).sort_values(
        ["diagnostic", "k", "seed", "epoch"]
    ).reset_index(drop=True)

    full37_summary_df.to_csv(output_dir / "ae_full37_reconstruction_summary.csv", index=False)
    full37_year_df.to_csv(output_dir / "ae_full37_reconstruction_by_year.csv", index=False)
    onefold_summary_df.to_csv(output_dir / "ae_onefold_overfit_summary.csv", index=False)
    loss_curve_df.to_csv(output_dir / "ae_reconstruction_diagnostic_loss_curves.csv", index=False)

    plot_full37_final_reconstruction(
        full37_summary_df, output_dir / "ae_full37_final_reconstruction_mse_vs_k.png"
    )
    plot_full37_loss_curves(
        loss_curve_df[loss_curve_df["diagnostic"] == "full37_reconstruction"].copy(),
        output_dir / "ae_full37_loss_curves_by_k.png",
    )
    plot_full37_by_year(
        full37_year_df,
        output_dir / "ae_full37_reconstruction_by_year.png",
    )
    plot_onefold_train_vs_heldout(
        onefold_summary_df,
        output_dir / "ae_onefold_train_vs_heldout_reconstruction_vs_k.png",
    )
    plot_onefold_loss_curves(
        loss_curve_df[loss_curve_df["diagnostic"] == "onefold_overfit"].copy(),
        output_dir / "ae_onefold_loss_curves_by_k.png",
    )

    dump_json(output_dir / "ae_reconstruction_diagnostics_metadata.json", metadata)
    write_readme(
        output_dir / "README.md",
        args=args,
        metadata=metadata,
        full37_summary_df=full37_summary_df,
        onefold_summary_df=onefold_summary_df,
    )

    full37_mean = (
        full37_summary_df.groupby("k", as_index=False)["final_train_reconstruction_mse"]
        .mean()
        .sort_values("k")
    )
    onefold_mean = onefold_summary_df.groupby("k", as_index=False).agg(
        final_train_reconstruction_mse=("final_train_reconstruction_mse", "mean"),
        heldout_reconstruction_mse=("heldout_reconstruction_mse", "mean"),
    ).sort_values("k")
    best_full37_k = int(full37_mean.sort_values("final_train_reconstruction_mse").iloc[0]["k"])
    train_improves = bool(
        np.all(np.diff(full37_mean.sort_values("k")["final_train_reconstruction_mse"].to_numpy(dtype=float)) <= 1.0e-10)
    )
    heldout_improves = bool(
        np.all(np.diff(onefold_mean.sort_values("k")["heldout_reconstruction_mse"].to_numpy(dtype=float)) <= 1.0e-10)
    )

    print("Full-37 reconstruction:", flush=True)
    for _, row in full37_mean.iterrows():
        print(
            "k={} final train reconstruction MSE: {:.6f}".format(
                int(row["k"]), float(row["final_train_reconstruction_mse"])
            ),
            flush=True,
        )
    print(f"best k by full-37 reconstruction: {best_full37_k}", flush=True)
    print("", flush=True)
    print("One-fold overfit reconstruction:", flush=True)
    print(f"heldout year: {int(args.heldout_year)}", flush=True)
    for _, row in onefold_mean.iterrows():
        print(
            "k={} train MSE / heldout MSE: {:.6f} / {:.6f}".format(
                int(row["k"]),
                float(row["final_train_reconstruction_mse"]),
                float(row["heldout_reconstruction_mse"]),
            ),
            flush=True,
        )
    print(f"whether train reconstruction improves with k: {'yes' if train_improves else 'no'}", flush=True)
    print(f"whether heldout reconstruction improves with k: {'yes' if heldout_improves else 'no'}", flush=True)
    print(f"artifact directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
