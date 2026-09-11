#!/usr/bin/env python3
"""Strict LOYO supervised autoencoder on the 91-column ocean-mode predictor table."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_OCEAN_TABLE = (
    PROJECT_ROOT / "artifacts" / "ocean_mode_pyod_autoencoder_loyo" / "ocean_mode_predictor_table.csv"
)
INPUT_BASELINE_TABLE = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)
INPUT_UNSUPERVISED_METRICS = (
    PROJECT_ROOT / "artifacts" / "ocean_mode_pyod_autoencoder_loyo" / "swe_ridge_loyo_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/ocean_mode_supervised_ae_loyo"
)
LOCAL_POINTER_DIR = PROJECT_ROOT / "artifacts" / "ocean_mode_supervised_ae_loyo"
NEAR_ZERO_STD = 1.0e-12
BASELINE_ALPHA_GRID = np.logspace(-6.0, 6.0, 49, dtype=np.float64)
BASELINE_PREDICTIONS_NAME = "baseline_7col_ridge_predictions.csv"
SUPERVISED_PREDICTIONS_NAME = "supervised_ae_loyo_predictions.csv"
SUPERVISED_LATENTS_NAME = "supervised_ae_latent_codes.csv"
SUPERVISED_TRAINING_LOG_NAME = "supervised_ae_training_log.csv"
PRIMARY_LOG_NAME = "run.log"


@dataclass(frozen=True)
class Setting:
    k: int
    lambda_swe: float
    dropout_rate: float
    weight_decay: float
    seed: int

    @property
    def model_name(self) -> str:
        return (
            f"SUPERVISED_AE_k{self.k}_ls{self.lambda_swe:g}_"
            f"d{self.dropout_rate:g}_wd{self.weight_decay:g}_seed{self.seed}"
        )


class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, input_dim),
        )
        self.swe_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.swe_head(z).squeeze(-1)
        return z, x_hat, y_hat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--lambda-swe-grid", nargs="+", type=float, default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--dropout-rates", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1.0e-5, 1.0e-4, 1.0e-3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-pointer-dir", type=Path, default=LOCAL_POINTER_DIR)
    return parser.parse_args()


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside a Perlmutter interactive compute-node allocation.")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_train_only(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (np.abs(std) < NEAR_ZERO_STD), 1.0, std)
    return (train - mean[None, :]) / std[None, :], (test - mean) / std, mean, std


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64), rhs)


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha = float(BASELINE_ALPHA_GRID[0])
    best_mse = float("inf")
    for alpha in BASELINE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(
                x_train_raw[mask], x_train_raw[~mask][0]
            )
            y_inner_train = y_train_raw[mask]
            y_mean = float(np.mean(y_inner_train))
            y_std = float(np.std(y_inner_train, ddof=1))
            if not np.isfinite(y_std) or abs(y_std) < NEAR_ZERO_STD:
                y_std = 1.0
            y_inner_train_std = (y_inner_train - y_mean) / y_std
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, float(alpha))
            preds[inner_idx] = y_mean + y_std * float(x_inner_valid_std @ beta_std)
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and float(alpha) < best_alpha):
            best_alpha = float(alpha)
            best_mse = mse
    return best_alpha, best_mse


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(((np.sign(obs) == np.sign(pred)) & (obs != 0.0) & (pred != 0.0)).astype(float)))


def compute_scalar_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    obs_std = float(np.std(obs, ddof=1))
    pred_std = float(np.std(pred, ddof=1))
    if obs_std < NEAR_ZERO_STD or pred_std < NEAR_ZERO_STD:
        r = float("nan")
    else:
        r = float(np.corrcoef(obs, pred)[0, 1])
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    sse = float(np.sum((obs - pred) ** 2))
    r2 = float("nan") if sst < NEAR_ZERO_STD else float(1.0 - sse / sst)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "r": r,
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "bias": bias,
        "pred_std": pred_std,
        "obs_std": obs_std,
        "n_years": int(obs.size),
    }


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ocean_df = pd.read_csv(INPUT_OCEAN_TABLE).sort_values("water_year").reset_index(drop=True)
    baseline_df = pd.read_csv(INPUT_BASELINE_TABLE).sort_values("water_year").reset_index(drop=True)
    if ocean_df["water_year"].tolist() != baseline_df["water_year"].tolist():
        raise RuntimeError("Water years do not match between ocean-mode table and baseline target table.")
    return ocean_df, baseline_df


def train_supervised_ae_fold(
    x_train_std: np.ndarray,
    y_train_std: np.ndarray,
    x_test_std: np.ndarray,
    *,
    setting: Setting,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(setting.seed)
    model = SupervisedAutoencoder(x_train_std.shape[1], setting.k, setting.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=setting.weight_decay)
    x_train_tensor = torch.tensor(x_train_std, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test_std[None, :], dtype=torch.float32, device=device)

    final_total_loss = float("nan")
    final_recon_loss = float("nan")
    final_swe_loss = float("nan")
    final_train_swe_rmse = float("nan")
    final_train_recon_mse = float("nan")

    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        z_train, x_hat_train, y_hat_train = model(x_train_tensor)
        recon_loss = torch.mean((x_hat_train - x_train_tensor) ** 2)
        swe_loss = torch.mean((y_hat_train - y_train_tensor) ** 2)
        total_loss = recon_loss + setting.lambda_swe * swe_loss
        total_loss.backward()
        optimizer.step()
        final_total_loss = float(total_loss.detach().cpu().item())
        final_recon_loss = float(recon_loss.detach().cpu().item())
        final_swe_loss = float(swe_loss.detach().cpu().item())
        final_train_swe_rmse = float(torch.sqrt(swe_loss).detach().cpu().item())
        final_train_recon_mse = final_recon_loss

    model.eval()
    with torch.no_grad():
        z_train, x_hat_train, y_hat_train = model(x_train_tensor)
        z_test, x_hat_test, y_hat_test = model(x_test_tensor)
    return {
        "z_test": z_test.detach().cpu().numpy().astype(np.float64)[0],
        "y_pred_std": float(y_hat_test.detach().cpu().numpy()[0]),
        "heldout_reconstruction_mse": float(
            np.mean((x_hat_test.detach().cpu().numpy()[0].astype(np.float64) - x_test_std) ** 2)
        ),
        "train_reconstruction_mse": float(
            np.mean((x_hat_train.detach().cpu().numpy().astype(np.float64) - x_train_std) ** 2)
        ),
        "final_total_loss": final_total_loss,
        "final_recon_loss": final_recon_loss,
        "final_swe_loss": final_swe_loss,
        "final_train_swe_rmse": final_train_swe_rmse,
        "final_train_reconstruction_mse": final_train_recon_mse,
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def append_row_csv(path: Path, row: Dict[str, object], columns: Sequence[str]) -> None:
    frame = pd.DataFrame([{column: row.get(column, np.nan) for column in columns}], columns=list(columns))
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def dedupe_checkpoint(df: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=list(subset), keep="last").reset_index(drop=True)


def parse_baseline_log_line(line: str) -> Optional[Dict[str, object]]:
    if not line.startswith("Baseline LOYO heldout_WY="):
        return None
    parts = line.strip().split()
    parsed: Dict[str, object] = {"model_name": "BASELINE_7COL_RIDGE", "inner_cv_mse": np.nan}
    for token in parts[2:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "heldout_WY":
            parsed["water_year"] = int(value)
        elif key == "alpha":
            parsed["alpha_selected"] = float(value)
        elif key == "pred":
            parsed["y_pred"] = float(value)
        elif key == "obs":
            parsed["y_obs"] = float(value)
    if {"water_year", "alpha_selected", "y_pred", "y_obs"} <= parsed.keys():
        return parsed
    return None


def seed_baseline_checkpoint_from_log(output_dir: Path) -> None:
    baseline_path = output_dir / BASELINE_PREDICTIONS_NAME
    if baseline_path.exists():
        return
    run_log_path = output_dir / PRIMARY_LOG_NAME
    if not run_log_path.exists():
        return
    rows: List[Dict[str, object]] = []
    for line in run_log_path.read_text(errors="replace").splitlines():
        parsed = parse_baseline_log_line(line)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        return
    baseline_df = pd.DataFrame(rows).drop_duplicates(subset=["water_year"]).sort_values("water_year").reset_index(drop=True)
    baseline_df.to_csv(baseline_path, index=False)


def create_local_pointer(output_dir: Path, local_pointer_dir: Path) -> None:
    if local_pointer_dir.exists() or local_pointer_dir.is_symlink():
        if local_pointer_dir.is_symlink() and local_pointer_dir.resolve() == output_dir.resolve():
            return
        if local_pointer_dir.is_dir() and not any(local_pointer_dir.iterdir()):
            local_pointer_dir.rmdir()
        else:
            raise RuntimeError(
                f"Local pointer path {local_pointer_dir} already exists and is not the expected symlink."
            )
    local_pointer_dir.symlink_to(output_dir)


def plot_metric_vs_lambda(
    summary_df: pd.DataFrame,
    metric_col: str,
    baseline_value: Optional[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for k_value, sub in summary_df.groupby("k", sort=True):
        sub = sub.sort_values("lambda_swe")
        ax.errorbar(
            sub["lambda_swe"],
            sub[metric_col],
            yerr=sub.get(f"std_{metric_col.replace('mean_', '')}", None),
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=f"k={int(k_value)}",
        )
    if baseline_value is not None and np.isfinite(baseline_value):
        ax.axhline(float(baseline_value), color="#d62728", linestyle="--", linewidth=1.6, label="7-column baseline")
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlabel("lambda_swe")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_timeseries(
    pred_df: pd.DataFrame,
    baseline_pred_df: pd.DataFrame,
    best_model_name: str,
    out_path: Path,
) -> None:
    best = pred_df[pred_df["model_name"] == best_model_name].sort_values("water_year")
    baseline = baseline_pred_df.sort_values("water_year")
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    ax.plot(best["water_year"], best["y_obs"], color="black", linewidth=2.5, label="Observed SWE")
    ax.plot(best["water_year"], best["y_pred"], color="#1f77b4", linewidth=2.0, label="Best supervised AE")
    ax.plot(baseline["water_year"], baseline["y_pred"], color="#d62728", linestyle="--", linewidth=1.8, label="7-column baseline")
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly")
    ax.set_title("Strict LOYO SWE: best supervised AE vs 7-column baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_scatter(
    pred_df: pd.DataFrame,
    baseline_pred_df: pd.DataFrame,
    best_model_name: str,
    out_path: Path,
) -> None:
    best = pred_df[pred_df["model_name"] == best_model_name].sort_values("water_year")
    baseline = baseline_pred_df.sort_values("water_year")
    limits = [
        float(min(best["y_obs"].min(), best["y_pred"].min(), baseline["y_pred"].min())),
        float(max(best["y_obs"].max(), best["y_pred"].max(), baseline["y_pred"].max())),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
    ax.scatter(best["y_obs"], best["y_pred"], s=50, color="#1f77b4", label="Best supervised AE")
    ax.scatter(baseline["y_obs"], baseline["y_pred"], s=50, color="#d62728", marker="x", label="7-column baseline")
    ax.plot(limits, limits, color="black", linewidth=1.2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed SWE")
    ax.set_ylabel("Predicted SWE")
    ax.set_title("Observed vs predicted SWE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tradeoff(metrics_df: pd.DataFrame, out_path: Path) -> None:
    sup = metrics_df[metrics_df["model_name"] != "BASELINE_7COL_RIDGE"].copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        sup["mean_heldout_reconstruction_mse"],
        sup["RMSE"],
        c=sup["lambda_swe"],
        s=40 + 8 * sup["k"].astype(float),
        cmap="viridis",
        alpha=0.85,
    )
    ax.set_xlabel("Mean held-out reconstruction MSE")
    ax.set_ylabel("SWE RMSE")
    ax.set_title("Supervised AE tradeoff: SWE RMSE vs reconstruction MSE")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="lambda_swe")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_report(
    output_dir: Path,
    baseline_metrics: Dict[str, float],
    sup_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    leakage_checks: Dict[str, object],
    unsup_best_row: Optional[pd.Series],
) -> None:
    best_rmse = sup_metrics_df.sort_values(["RMSE", "r"]).iloc[0]
    best_r = sup_metrics_df.sort_values(["r", "RMSE"], ascending=[False, True]).iloc[0]
    best_sign = sup_metrics_df.sort_values(["sign_accuracy", "RMSE"], ascending=[False, True]).iloc[0]
    best_recon = sup_metrics_df.sort_values(["mean_heldout_reconstruction_mse", "RMSE"]).iloc[0]
    best_lambda0 = sup_metrics_df[np.isclose(sup_metrics_df["lambda_swe"], 0.0)].sort_values(["RMSE", "r"]).iloc[0]
    best_lambda = summary_df.sort_values(["mean_RMSE", "mean_r"], ascending=[True, False]).iloc[0]
    best_k = (
        summary_df.groupby("k", as_index=False)["mean_RMSE"]
        .min()
        .sort_values("mean_RMSE")
        .iloc[0]["k"]
    )
    rmse_by_k = (
        summary_df.groupby("k", as_index=False)["mean_RMSE"].min().sort_values("k")
    )
    larger_k_helps = bool(np.all(np.diff(rmse_by_k["mean_RMSE"].to_numpy(dtype=float)) <= 1.0e-12))
    best_lambda0_rmse = float(best_lambda0["RMSE"])
    improved_over_lambda0 = bool(float(best_rmse["RMSE"]) < best_lambda0_rmse - 1.0e-12)
    beat_unsup = False
    if unsup_best_row is not None and np.isfinite(float(unsup_best_row["RMSE"])):
        beat_unsup = bool(float(best_rmse["RMSE"]) < float(unsup_best_row["RMSE"]) - 1.0e-12)
    beat_baseline = bool(float(best_rmse["RMSE"]) < float(baseline_metrics["RMSE"]) - 1.0e-12)
    best_recon_same_as_best_swe = bool(best_recon["model_name"] == best_rmse["model_name"])

    lines = [
        "# Supervised ocean-mode autoencoder LOYO report",
        "",
        "This run is a first strict-LOYO supervised-autoencoder tryout on the 91-column ocean-mode predictor table.",
        "",
        "## Leakage checks",
        "",
    ]
    for key, value in leakage_checks.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Interpretation",
        "",
        f"1. Did adding the supervised SWE term improve over lambda_swe=0? {'Yes.' if improved_over_lambda0 else 'No.'} Best overall RMSE was `{float(best_rmse['RMSE']):.6f}` while best `lambda_swe=0` RMSE was `{best_lambda0_rmse:.6f}`.",
        (
            f"2. Did supervised AE beat the previous unsupervised AE latent ridge? {'Yes.' if beat_unsup else 'No.'} "
            f"Best supervised AE RMSE was `{float(best_rmse['RMSE']):.6f}`; "
            f"best unsupervised AE latent ridge RMSE was `{float(unsup_best_row['RMSE']):.6f}`."
            if unsup_best_row is not None
            else "2. Did supervised AE beat the previous unsupervised AE latent ridge? Unsupervised AE summary was not available."
        ),
        f"3. Did supervised AE beat, match, or lose to the 7-column baseline? {'Beat.' if beat_baseline else 'Lost to the 7-column baseline.'} Baseline RMSE was `{float(baseline_metrics['RMSE']):.6f}` and best supervised-AE RMSE was `{float(best_rmse['RMSE']):.6f}`.",
        f"4. Which lambda_swe worked best? `lambda_swe={float(best_lambda['lambda_swe']):g}` had the best aggregate mean RMSE for `k={int(best_lambda['k'])}`.",
        f"5. Which k worked best? `k={int(best_k)}` by best aggregate mean RMSE across lambda/dropout/weight_decay/seed settings.",
        f"6. Did larger k help SWE prediction? {'Yes.' if larger_k_helps else 'No.'} Best mean-RMSE-by-k sequence was `{', '.join(f'k={int(row.k)} -> {float(row.mean_RMSE):.6f}' for row in rmse_by_k.itertuples())}`.",
        f"7. Did increasing lambda_swe improve SWE prediction while hurting reconstruction? Inspect the lambda-k summary and tradeoff plot. Best SWE model used `lambda_swe={float(best_rmse['lambda_swe']):g}`, while best reconstruction model used `lambda_swe={float(best_recon['lambda_swe']):g}`.",
        f"8. Was there a tradeoff between ocean-mode reconstruction and SWE prediction? {'Yes, the best SWE-prediction setting was different from the best reconstruction setting.' if not best_recon_same_as_best_swe else 'Not strongly; the same setting optimized both.'}",
        f"9. Is the best SWE-prediction setting also the best reconstruction setting? {'Yes.' if best_recon_same_as_best_swe else 'No.'}",
        f"10. Based on this first tryout, should we continue supervised AE tuning? {'Yes, if the supervised term improved over lambda_swe=0 or the unsupervised AE reference, but the fixed 7-column baseline remains the scientific target to beat.' if (improved_over_lambda0 or beat_unsup) else 'Only cautiously. This first tryout did not clearly justify the extra complexity relative to the fixed 7-column baseline.'}",
        "",
        "## Best settings",
        "",
        f"- Baseline 7-column ridge: RMSE `{float(baseline_metrics['RMSE']):.6f}`, r `{float(baseline_metrics['r']):.6f}`, sign_accuracy `{float(baseline_metrics['sign_accuracy']):.6f}`",
        f"- Best supervised AE by RMSE: `{best_rmse['model_name']}`",
        f"- Best supervised AE by r: `{best_r['model_name']}`",
        f"- Best supervised AE by sign accuracy: `{best_sign['model_name']}`",
        f"- Best supervised AE by reconstruction MSE: `{best_recon['model_name']}`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_outputs(
    *,
    output_dir: Path,
    baseline_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    training_log_df: pd.DataFrame,
    years: np.ndarray,
    baseline_metrics: Dict[str, float],
    unsup_best_row: Optional[pd.Series],
    leakage_checks: Dict[str, object],
) -> None:
    prediction_df = prediction_df.drop_duplicates(subset=["model_name", "water_year"], keep="last").copy()
    latent_df = latent_df.drop_duplicates(subset=["model_name", "water_year"], keep="last").copy()
    training_log_df = training_log_df.drop_duplicates(subset=["model_name", "water_year"], keep="last").copy()
    if prediction_df.empty:
        raise RuntimeError("No supervised prediction checkpoint rows were found.")
    metrics_rows: List[Dict[str, object]] = []
    for (model_name, k, lambda_swe, dropout_rate, weight_decay, seed), sub in prediction_df.groupby(
        ["model_name", "k", "lambda_swe", "dropout_rate", "weight_decay", "seed"], dropna=False
    ):
        if sub["water_year"].nunique() != years.size:
            raise RuntimeError(f"Incomplete setting {model_name}: expected {years.size} folds.")
        sub = sub.sort_values("water_year")
        metrics = compute_scalar_metrics(
            sub["y_obs"].to_numpy(dtype=np.float64),
            sub["y_pred"].to_numpy(dtype=np.float64),
        )
        heldout = sub["heldout_reconstruction_mse"].to_numpy(dtype=np.float64)
        train = sub["train_reconstruction_mse"].to_numpy(dtype=np.float64)
        metrics_rows.append(
            {
                "model_name": model_name,
                "k": int(k),
                "lambda_swe": float(lambda_swe),
                "dropout_rate": float(dropout_rate),
                "weight_decay": float(weight_decay),
                "seed": int(seed),
                **metrics,
                "mean_heldout_reconstruction_mse": float(np.mean(heldout)),
                "median_heldout_reconstruction_mse": float(np.median(heldout)),
                "reconstruction_rmse": float(np.sqrt(np.mean(heldout))),
                "mean_train_reconstruction_mse": float(np.mean(train)),
                "reconstruction_generalization_ratio": float(np.mean(heldout / np.where(train > 0.0, train, np.nan))),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["RMSE", "r"], ascending=[True, False]).reset_index(drop=True)
    summary_df = (
        metrics_df.groupby(["k", "lambda_swe"], as_index=False)
        .agg(
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_r=("r", "mean"),
            std_r=("r", "std"),
            mean_sign_accuracy=("sign_accuracy", "mean"),
            std_sign_accuracy=("sign_accuracy", "std"),
            mean_reconstruction_mse=("mean_heldout_reconstruction_mse", "mean"),
            std_reconstruction_mse=("mean_heldout_reconstruction_mse", "std"),
        )
        .sort_values(["k", "lambda_swe"])
        .reset_index(drop=True)
    )

    best_rmse_row = metrics_df.sort_values(["RMSE", "r"]).iloc[0]
    best_r_row = metrics_df.sort_values(["r", "RMSE"], ascending=[False, True]).iloc[0]
    best_sign_row = metrics_df.sort_values(["sign_accuracy", "RMSE"], ascending=[False, True]).iloc[0]
    best_recon_row = metrics_df.sort_values(["mean_heldout_reconstruction_mse", "RMSE"]).iloc[0]
    best_lambda0_row = metrics_df[np.isclose(metrics_df["lambda_swe"], 0.0)].sort_values(["RMSE", "r"]).iloc[0]

    best_models_rows: List[Dict[str, object]] = [
        {"best_model_role": "baseline_7col_ridge", "model_name": "BASELINE_7COL_RIDGE", **baseline_metrics},
        {"best_model_role": "best_supervised_ae_by_RMSE", **best_rmse_row.to_dict()},
        {"best_model_role": "best_supervised_ae_by_r", **best_r_row.to_dict()},
        {"best_model_role": "best_supervised_ae_by_sign_accuracy", **best_sign_row.to_dict()},
        {"best_model_role": "best_supervised_ae_by_reconstruction_MSE", **best_recon_row.to_dict()},
        {"best_model_role": "best_lambda_swe_0_model", **best_lambda0_row.to_dict()},
    ]
    if unsup_best_row is not None:
        best_models_rows.append({"best_model_role": "best_previous_unsupervised_ae", **unsup_best_row.to_dict()})
    best_models_df = pd.DataFrame(best_models_rows)

    save_dataframe(prediction_df.sort_values(["k", "lambda_swe", "dropout_rate", "weight_decay", "seed", "water_year"]).reset_index(drop=True), output_dir / SUPERVISED_PREDICTIONS_NAME)
    save_dataframe(metrics_df, output_dir / "supervised_ae_loyo_metrics.csv")
    save_dataframe(summary_df, output_dir / "supervised_ae_summary_by_lambda_k.csv")
    save_dataframe(best_models_df, output_dir / "supervised_ae_best_models.csv")
    save_dataframe(latent_df.sort_values(["k", "lambda_swe", "dropout_rate", "weight_decay", "seed", "water_year"]).reset_index(drop=True), output_dir / SUPERVISED_LATENTS_NAME)
    save_dataframe(training_log_df.sort_values(["k", "lambda_swe", "dropout_rate", "weight_decay", "seed", "water_year"]).reset_index(drop=True), output_dir / SUPERVISED_TRAINING_LOG_NAME)
    save_dataframe(baseline_df.sort_values("water_year").reset_index(drop=True), output_dir / BASELINE_PREDICTIONS_NAME)

    plot_metric_vs_lambda(
        summary_df,
        "mean_RMSE",
        float(baseline_metrics["RMSE"]),
        "RMSE",
        "Supervised AE SWE RMSE vs lambda_swe",
        output_dir / "supervised_ae_rmse_by_lambda_k.png",
    )
    plot_metric_vs_lambda(
        summary_df,
        "mean_r",
        float(baseline_metrics["r"]),
        "Pearson r",
        "Supervised AE SWE correlation vs lambda_swe",
        output_dir / "supervised_ae_r_by_lambda_k.png",
    )
    plot_metric_vs_lambda(
        summary_df,
        "mean_sign_accuracy",
        float(baseline_metrics["sign_accuracy"]),
        "Sign accuracy",
        "Supervised AE SWE sign accuracy vs lambda_swe",
        output_dir / "supervised_ae_sign_accuracy_by_lambda_k.png",
    )
    plot_metric_vs_lambda(
        summary_df,
        "mean_reconstruction_mse",
        None,
        "Mean held-out reconstruction MSE",
        "Supervised AE reconstruction MSE vs lambda_swe",
        output_dir / "supervised_ae_reconstruction_mse_by_lambda_k.png",
    )
    plot_best_timeseries(prediction_df, baseline_df, str(best_rmse_row["model_name"]), output_dir / "supervised_ae_best_timeseries.png")
    plot_best_scatter(prediction_df, baseline_df, str(best_rmse_row["model_name"]), output_dir / "supervised_ae_best_scatter_obs_vs_pred.png")
    plot_tradeoff(metrics_df, output_dir / "supervised_ae_tradeoff_rmse_vs_reconstruction.png")
    build_report(output_dir, baseline_metrics, metrics_df, summary_df, best_models_df, leakage_checks, unsup_best_row)

    metadata = {
        "script_path": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "output_dir": str(output_dir),
        "input_ocean_table": str(INPUT_OCEAN_TABLE),
        "input_baseline_table": str(INPUT_BASELINE_TABLE),
        "input_unsupervised_metrics": str(INPUT_UNSUPERVISED_METRICS),
        "latent_dims": sorted(prediction_df["k"].dropna().astype(int).unique().tolist()),
        "lambda_swe_grid": sorted(prediction_df["lambda_swe"].dropna().astype(float).unique().tolist()),
        "dropout_rates": sorted(prediction_df["dropout_rate"].dropna().astype(float).unique().tolist()),
        "weight_decays": sorted(prediction_df["weight_decay"].dropna().astype(float).unique().tolist()),
        "seeds": sorted(prediction_df["seed"].dropna().astype(int).unique().tolist()),
        "baseline_alpha_grid": [float(v) for v in BASELINE_ALPHA_GRID.tolist()],
        "baseline_metrics": baseline_metrics,
        "best_supervised_model_by_rmse": best_rmse_row.to_dict(),
        "completed_supervised_fold_rows": int(prediction_df.shape[0]),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_runtime_on_compute_node()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    create_local_pointer(output_dir, args.local_pointer_dir)
    seed_baseline_checkpoint_from_log(output_dir)

    ocean_df, baseline_df = load_data()
    years = ocean_df["water_year"].to_numpy(dtype=int)
    x_features = [col for col in ocean_df.columns if col != "water_year"]
    x_all = ocean_df[x_features].to_numpy(dtype=np.float64)
    y_all = baseline_df["obs_swe"].to_numpy(dtype=np.float64)
    baseline_feature_names = [col for col in baseline_df.columns if col not in {"water_year", "obs_swe"}]
    x_baseline_all = baseline_df[baseline_feature_names].to_numpy(dtype=np.float64)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    settings = [
        Setting(k, lambda_swe, dropout_rate, weight_decay, seed)
        for k in args.latent_dims
        for lambda_swe in args.lambda_swe_grid
        for dropout_rate in args.dropout_rates
        for weight_decay in args.weight_decays
        for seed in args.seeds
    ]

    baseline_path = output_dir / BASELINE_PREDICTIONS_NAME
    prediction_path = output_dir / SUPERVISED_PREDICTIONS_NAME
    latent_path = output_dir / SUPERVISED_LATENTS_NAME
    training_log_path = output_dir / SUPERVISED_TRAINING_LOG_NAME

    baseline_columns = ["model_name", "water_year", "y_obs", "y_pred", "alpha_selected", "inner_cv_mse"]
    prediction_columns = [
        "model_name",
        "water_year",
        "y_obs",
        "y_pred",
        "k",
        "lambda_swe",
        "dropout_rate",
        "weight_decay",
        "seed",
        "heldout_reconstruction_mse",
        "train_reconstruction_mse",
        "final_total_loss",
        "final_recon_loss",
        "final_swe_loss",
    ]
    latent_columns = [
        "model_name",
        "water_year",
        "k",
        "lambda_swe",
        "dropout_rate",
        "weight_decay",
        "seed",
    ] + [f"z{i}" for i in range(1, max(args.latent_dims) + 1)]
    training_log_columns = [
        "model_name",
        "water_year",
        "k",
        "lambda_swe",
        "dropout_rate",
        "weight_decay",
        "seed",
        "epochs_trained",
        "final_total_loss",
        "final_recon_loss",
        "final_swe_loss",
        "final_train_swe_rmse_std",
        "final_train_reconstruction_mse",
    ]

    existing_baseline_df = load_existing_csv(baseline_path)
    completed_baseline_years = set(existing_baseline_df["water_year"].astype(int).tolist()) if not existing_baseline_df.empty else set()
    for heldout_year in years.tolist():
        if int(heldout_year) in completed_baseline_years:
            continue
        train_mask = years != heldout_year
        x_train = x_baseline_all[train_mask]
        x_test = x_baseline_all[~train_mask][0]
        y_train = y_all[train_mask]
        y_test = float(y_all[~train_mask][0])
        alpha, inner_mse = inner_loyo_best_alpha(x_train, y_train)
        x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train, ddof=1))
        if not np.isfinite(y_std) or abs(y_std) < NEAR_ZERO_STD:
            y_std = 1.0
        y_train_std = (y_train - y_mean) / y_std
        beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
        pred = y_mean + y_std * float(x_test_std @ beta_std)
        row = {
            "model_name": "BASELINE_7COL_RIDGE",
            "water_year": int(heldout_year),
            "y_obs": y_test,
            "y_pred": pred,
            "alpha_selected": alpha,
            "inner_cv_mse": inner_mse,
        }
        append_row_csv(baseline_path, row, baseline_columns)
        completed_baseline_years.add(int(heldout_year))
        print(f"Baseline LOYO heldout_WY={heldout_year} alpha={alpha:g} pred={pred:.6f} obs={y_test:.6f}", flush=True)

    baseline_df_out = dedupe_checkpoint(load_existing_csv(baseline_path), ["water_year"]).sort_values("water_year").reset_index(drop=True)
    if baseline_df_out["water_year"].nunique() != years.size:
        raise RuntimeError(f"Baseline checkpoint incomplete: found {baseline_df_out['water_year'].nunique()} of {years.size} years.")
    baseline_metrics = compute_scalar_metrics(
        baseline_df_out["y_obs"].to_numpy(dtype=np.float64),
        baseline_df_out["y_pred"].to_numpy(dtype=np.float64),
    )
    baseline_alphas = baseline_df_out["alpha_selected"].to_numpy(dtype=np.float64)
    baseline_metrics["alpha_selected_median"] = float(np.median(baseline_alphas))
    baseline_metrics["alpha_selected_mode"] = float(pd.Series(baseline_alphas).value_counts().sort_index().idxmax())

    existing_prediction_df = dedupe_checkpoint(load_existing_csv(prediction_path), ["model_name", "water_year"])
    completed_prediction_keys = (
        set(zip(existing_prediction_df["model_name"].astype(str), existing_prediction_df["water_year"].astype(int)))
        if not existing_prediction_df.empty
        else set()
    )

    total_settings = len(settings)
    for setting_idx, setting in enumerate(settings, start=1):
        existing_count = sum((setting.model_name, int(year)) in completed_prediction_keys for year in years.tolist())
        if existing_count == years.size:
            print(f"Skipping completed setting {setting_idx}/{total_settings} {setting.model_name}", flush=True)
            continue
        for outer_idx, heldout_year in enumerate(years.tolist()):
            if (setting.model_name, int(heldout_year)) in completed_prediction_keys:
                continue
            train_mask = years != heldout_year
            x_train = x_all[train_mask]
            x_test = x_all[~train_mask][0]
            y_train = y_all[train_mask]
            y_test = float(y_all[~train_mask][0])

            x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train, ddof=1))
            if not np.isfinite(y_std) or abs(y_std) < NEAR_ZERO_STD:
                y_std = 1.0
            y_train_std = (y_train - y_mean) / y_std

            fold = train_supervised_ae_fold(
                x_train_std,
                y_train_std,
                x_test_std,
                setting=setting,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
            )
            pred = y_mean + y_std * float(fold["y_pred_std"])
            latent_row = {
                "model_name": setting.model_name,
                "water_year": int(heldout_year),
                "k": int(setting.k),
                "lambda_swe": float(setting.lambda_swe),
                "dropout_rate": float(setting.dropout_rate),
                "weight_decay": float(setting.weight_decay),
                "seed": int(setting.seed),
            }
            for latent_idx in range(1, max(args.latent_dims) + 1):
                latent_row[f"z{latent_idx}"] = (
                    float(fold["z_test"][latent_idx - 1]) if latent_idx <= setting.k else np.nan
                )
            append_row_csv(latent_path, latent_row, latent_columns)
            training_row = {
                "model_name": setting.model_name,
                "water_year": int(heldout_year),
                "k": int(setting.k),
                "lambda_swe": float(setting.lambda_swe),
                "dropout_rate": float(setting.dropout_rate),
                "weight_decay": float(setting.weight_decay),
                "seed": int(setting.seed),
                "epochs_trained": int(args.epochs),
                "final_total_loss": float(fold["final_total_loss"]),
                "final_recon_loss": float(fold["final_recon_loss"]),
                "final_swe_loss": float(fold["final_swe_loss"]),
                "final_train_swe_rmse_std": float(fold["final_train_swe_rmse"]),
                "final_train_reconstruction_mse": float(fold["final_train_reconstruction_mse"]),
            }
            append_row_csv(training_log_path, training_row, training_log_columns)
            prediction_row = {
                "model_name": setting.model_name,
                "water_year": int(heldout_year),
                "y_obs": y_test,
                "y_pred": pred,
                "k": int(setting.k),
                "lambda_swe": float(setting.lambda_swe),
                "dropout_rate": float(setting.dropout_rate),
                "weight_decay": float(setting.weight_decay),
                "seed": int(setting.seed),
                "heldout_reconstruction_mse": float(fold["heldout_reconstruction_mse"]),
                "train_reconstruction_mse": float(fold["train_reconstruction_mse"]),
                "final_total_loss": float(fold["final_total_loss"]),
                "final_recon_loss": float(fold["final_recon_loss"]),
                "final_swe_loss": float(fold["final_swe_loss"]),
            }
            append_row_csv(prediction_path, prediction_row, prediction_columns)
            completed_prediction_keys.add((setting.model_name, int(heldout_year)))
            print(
                "Setting {idx}/{total} {name} heldout_WY={wy} pred={pred:.6f} obs={obs:.6f} "
                "heldout_recon={held:.6f} train_recon={train:.6f}".format(
                    idx=setting_idx,
                    total=total_settings,
                    name=setting.model_name,
                    wy=int(heldout_year),
                    pred=pred,
                    obs=y_test,
                    held=float(fold["heldout_reconstruction_mse"]),
                    train=float(fold["train_reconstruction_mse"]),
                ),
                flush=True,
            )

    leakage_checks = {
        "number_of_water_years": int(years.size),
        "X_shape": f"{x_all.shape[0]} x {x_all.shape[1]}",
        "y_target_source_path": str(INPUT_BASELINE_TABLE),
        "heldout_year_absent_from_training_years": True,
        "X_scaler_fit_only_on_training_years": True,
        "y_scaler_fit_only_on_training_years": True,
        "no_heldout_SWE_used_during_training": True,
        "no_all_year_latent_codes_used": True,
        "baseline_ridge_alpha_tuned_only_inside_training_years": True,
        "baseline_alpha_grid": [float(v) for v in BASELINE_ALPHA_GRID.tolist()],
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
    }
    unsup_best_row = None
    if INPUT_UNSUPERVISED_METRICS.exists():
        unsup_df = pd.read_csv(INPUT_UNSUPERVISED_METRICS)
        if not unsup_df.empty:
            unsup_best_row = unsup_df.sort_values(["RMSE", "r"]).iloc[0]

    prediction_df = dedupe_checkpoint(load_existing_csv(prediction_path), ["model_name", "water_year"])
    latent_df = dedupe_checkpoint(load_existing_csv(latent_path), ["model_name", "water_year"])
    training_log_df = dedupe_checkpoint(load_existing_csv(training_log_path), ["model_name", "water_year"])
    prediction_df = prediction_df.drop_duplicates(subset=["model_name", "water_year"], keep="last")
    latent_df = latent_df.drop_duplicates(subset=["model_name", "water_year"], keep="last")
    training_log_df = training_log_df.drop_duplicates(subset=["model_name", "water_year"], keep="last")
    expected_supervised_rows = len(settings) * years.size
    if prediction_df.shape[0] != expected_supervised_rows:
        raise RuntimeError(
            f"Supervised checkpoint incomplete after run: found {prediction_df.shape[0]} rows, expected {expected_supervised_rows}."
        )
    if latent_df.shape[0] != expected_supervised_rows or training_log_df.shape[0] != expected_supervised_rows:
        raise RuntimeError("Latent-code or training-log checkpoint is incomplete after run.")

    build_final_outputs(
        output_dir=output_dir,
        baseline_df=baseline_df_out,
        prediction_df=prediction_df,
        latent_df=latent_df,
        training_log_df=training_log_df,
        years=years,
        baseline_metrics=baseline_metrics,
        unsup_best_row=unsup_best_row,
        leakage_checks=leakage_checks,
    )

    final_metrics_df = pd.read_csv(output_dir / "supervised_ae_loyo_metrics.csv")
    best_rmse_row = final_metrics_df.sort_values(["RMSE", "r"]).iloc[0]
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Baseline RMSE: {float(baseline_metrics['RMSE']):.6f}", flush=True)
    print(f"Best supervised AE by RMSE: {best_rmse_row['model_name']} RMSE={float(best_rmse_row['RMSE']):.6f}", flush=True)


if __name__ == "__main__":
    main()
