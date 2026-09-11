#!/usr/bin/env python3
"""Observational transfer + full reporting for the three percentile-target CNN models
(S0, frozen-S0+attention, end-to-end attention). Inference only - no training, no
fine-tuning, no observational checkpoint selection.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import (  # noqa: E402
    FEATURE_Z_CLIP,
    LATENT_D,
    LATENT_K,
    M1StaticCNN,
    StaticLatentSelfAttentionCNN,
    LatentSelfAttentionBlock,
    spearman_rho as _spearman_rho,
    wet_dry_accuracy as _wet_dry_accuracy,
)

OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
S0_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S0_static_cnn_swe_only")
S1_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S1_static_latent_self_attention")
FROZEN_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/frozen_S0_attention")
UCLA_PERCENTILE_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/ucla_percentile.npz")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1")
IN_CHANNELS_PER_MONTH = 38


def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def pearson_r(y_true, y_pred):
    a, b = y_true - y_true.mean(), y_pred - y_pred.mean()
    return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))


def tercile_accuracy(y_true, y_pred):
    def tercile(v):
        return np.where(v < 1 / 3, 0, np.where(v < 2 / 3, 1, 2))
    return float(np.mean(tercile(y_true) == tercile(y_pred)))


class FrozenS0Encoder(nn.Module):
    def __init__(self, s0: M1StaticCNN) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(s0.backbone)
        self.project = copy.deepcopy(s0.project)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, m, c, h, w = x.shape
        with torch.no_grad():
            features = self.backbone(x.reshape(b, m * c, h, w))
            return self.project(features).reshape(b, LATENT_K, LATENT_D)


class FrozenS0ZAttention(nn.Module):
    def __init__(self, encoder: FrozenS0Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_raw = self.encoder(x)
        attn_in = self.latent_block.norm1(z_raw)
        attn_out, _ = self.latent_block.attn(attn_in, attn_in, attn_in, need_weights=False)
        z_attn = z_raw + attn_out
        z_attn = z_attn + self.latent_block.ffn(self.latent_block.norm2(z_attn))
        return self.swe_head(z_attn.reshape(z_attn.shape[0], -1)).squeeze(-1)


def load_plain(cls, exp_dir: Path, device):
    checkpoint = torch.load(exp_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model = cls(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False) if cls is M1StaticCNN else cls(IN_CHANNELS_PER_MONTH)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, checkpoint["epoch"]


def load_frozen(device):
    s0_checkpoint = torch.load(S0_EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    s0 = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    s0.load_state_dict(s0_checkpoint["model_state_dict"], strict=True)
    encoder = FrozenS0Encoder(s0)
    model = FrozenS0ZAttention(encoder)
    trainable_checkpoint = torch.load(FROZEN_EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    result = model.load_state_dict(trainable_checkpoint["trainable_state_dict"], strict=False)
    expected_encoder_keys = {f"encoder.{k}" for k in model.encoder.state_dict().keys()}
    assert set(result.missing_keys) == expected_encoder_keys
    assert not result.unexpected_keys
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, trainable_checkpoint["epoch"]


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    valid_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    water_years = [int(r["water_year"]) for r in manifest]
    assert water_years == list(range(1985, 2022)), "observational water_year not ascending 1985..2021"
    assert physical.shape == (37, 7, 19, 120, 240)
    assert np.isfinite(physical).sum() > 0

    ucla = np.load(UCLA_PERCENTILE_NPZ)
    ucla_wy = [int(y) for y in ucla["water_year"]]
    assert ucla_wy == water_years, "UCLA water years do not align with observational tensor water years"
    ucla_mm = ucla["ucla_swe_mm"].astype(np.float64)
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)
    assert np.isfinite(ucla_percentile).all()
    print(f"UCLA percentile range: [{ucla_percentile.min():.4f}, {ucla_percentile.max():.4f}]", flush=True)

    def preprocess(feature_mu, feature_sigma):
        x = physical.astype(np.float32)
        x = (x - feature_mu) / feature_sigma
        x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
        x = np.where(valid_mask > 0.5, x, np.nan).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        stacked = np.concatenate([x, valid_mask.astype(np.float32)], axis=2)
        assert np.isfinite(stacked).all()
        return torch.from_numpy(stacked)

    results = {}
    predictions_percentile = {}
    loaders = {
        "S0": (lambda: load_plain(M1StaticCNN, S0_EXP_DIR, device), S0_EXP_DIR),
        "frozen_attention": (lambda: load_frozen(device), FROZEN_EXP_DIR),
        "end_to_end_attention": (lambda: load_plain(StaticLatentSelfAttentionCNN, S1_EXP_DIR, device), S1_EXP_DIR),
    }
    for name, (loader_fn, exp_dir) in loaders.items():
        print(f"=== {name} ===", flush=True)
        stats = np.load(exp_dir / "normalization_stats.npz")
        feature_mu, feature_sigma = stats["feature_mu"], stats["feature_sigma"]
        target_mu, target_sigma = float(stats["target_mu"][0]), float(stats["target_sigma"][0])
        inputs = preprocess(feature_mu, feature_sigma)

        model, epoch = loader_fn()
        params_before = snapshot(model)

        def run_forward():
            with torch.no_grad():
                out = model(inputs.to(device))
                out = out["swe_hat"] if isinstance(out, dict) else out
                return out.detach().cpu().numpy()

        pred1, pred2 = run_forward(), run_forward()
        assert np.allclose(pred1, pred2, atol=1e-6), f"{name}: non-deterministic"
        assert np.isfinite(pred1).all(), f"{name}: non-finite predictions"
        params_after = snapshot(model)
        max_change = max(float((params_before[k] - params_after[k]).abs().max()) for k in params_before)
        assert max_change == 0.0, f"{name}: parameter drift {max_change}"

        pred_percentile = pred1 * target_sigma + target_mu  # model's internal z -> percentile units
        predictions_percentile[name] = pred_percentile

        r2 = r2_score(ucla_percentile, pred_percentile)
        r = pearson_r(ucla_percentile, pred_percentile)
        rho = _spearman_rho(ucla_percentile, pred_percentile)
        wet_dry_acc = _wet_dry_accuracy(ucla_percentile, pred_percentile)
        tercile_acc = tercile_accuracy(ucla_percentile, pred_percentile)
        rmse = float(np.sqrt(np.mean((ucla_percentile - pred_percentile) ** 2)))
        mae = float(np.mean(np.abs(ucla_percentile - pred_percentile)))

        results[name] = {
            "checkpoint": str(exp_dir / "best_checkpoint.pt"),
            "checkpoint_epoch": int(epoch),
            "observational_r2": r2,
            "observational_pearson_r": r,
            "observational_spearman_rho": rho,
            "observational_wet_dry_accuracy": wet_dry_acc,
            "observational_tercile_accuracy": tercile_acc,
            "observational_rmse_percentile": rmse,
            "observational_mae_percentile": mae,
            "max_parameter_change": max_change,
            "deterministic": True,
        }
        print(json.dumps(results[name], indent=2), flush=True)

    # -----------------------------------------------------------------
    # Simulation-side metrics (from saved history/metrics_summary of each run).
    # -----------------------------------------------------------------
    s0_metrics = json.loads((S0_EXP_DIR / "metrics_summary.json").read_text())
    s1_metrics = json.loads((S1_EXP_DIR / "metrics_summary.json").read_text())
    frozen_metrics = json.loads((FROZEN_EXP_DIR / "metrics_summary.json").read_text())
    sim = {
        "S0": {"pearson_r": s0_metrics["best_metrics"]["swe_pearson_r"], "spearman_rho": s0_metrics["best_metrics"]["swe_spearman_rho"], "wet_dry_accuracy": s0_metrics["best_metrics"]["swe_wet_dry_accuracy"], "best_epoch": s0_metrics["best_epoch"]},
        "frozen_attention": {"pearson_r": frozen_metrics["best_metrics"]["swe_pearson_r"], "spearman_rho": frozen_metrics["best_metrics"]["swe_spearman_rho"], "wet_dry_accuracy": frozen_metrics["best_metrics"]["swe_wet_dry_accuracy"], "best_epoch": frozen_metrics["best_epoch"]},
        "end_to_end_attention": {"pearson_r": s1_metrics["best_metrics"]["swe_pearson_r"], "spearman_rho": s1_metrics["best_metrics"]["swe_spearman_rho"], "wet_dry_accuracy": s1_metrics["best_metrics"]["swe_wet_dry_accuracy"], "best_epoch": s1_metrics["best_epoch"]},
    }

    # -----------------------------------------------------------------
    # CSV outputs
    # -----------------------------------------------------------------
    pred_csv = OUTPUT_ROOT / "observational_predictions.csv"
    with pred_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["water_year", "UCLA_SWE_mm", "UCLA_percentile", "S0_predicted_percentile", "frozen_attention_predicted_percentile", "end_to_end_attention_predicted_percentile"])
        for i, wy in enumerate(water_years):
            writer.writerow([wy, f"{ucla_mm[i]:.6f}", f"{ucla_percentile[i]:.6f}", f"{predictions_percentile['S0'][i]:.6f}", f"{predictions_percentile['frozen_attention'][i]:.6f}", f"{predictions_percentile['end_to_end_attention'][i]:.6f}"])
    print(f"wrote {pred_csv}", flush=True)

    final_csv = OUTPUT_ROOT / "final_metrics.csv"
    with final_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Sim val Pearson r", "Sim val Spearman rho", "Sim val wet/dry acc", "Obs Pearson r", "Obs Spearman rho", "Obs wet/dry acc", "Obs tercile acc", "Obs R2"])
        name_labels = {"S0": "S0", "frozen_attention": "Frozen-S0 + attention", "end_to_end_attention": "End-to-end attention"}
        for key, label in name_labels.items():
            writer.writerow([
                label,
                f"{sim[key]['pearson_r']:.6f}", f"{sim[key]['spearman_rho']:.6f}", f"{sim[key]['wet_dry_accuracy']:.6f}",
                f"{results[key]['observational_pearson_r']:.6f}", f"{results[key]['observational_spearman_rho']:.6f}",
                f"{results[key]['observational_wet_dry_accuracy']:.6f}", f"{results[key]['observational_tercile_accuracy']:.6f}",
                f"{results[key]['observational_r2']:.6f}",
            ])
    print(f"wrote {final_csv}", flush=True)

    # -----------------------------------------------------------------
    # Plot 1 (main): percentile time series, x100 for %.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(13, 6), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, marker="o", color="black", linewidth=2.2, label="UCLA observed")
    axis.plot(water_years, predictions_percentile["S0"] * 100, marker="o", markersize=3, color="tab:blue", label="S0")
    axis.plot(water_years, predictions_percentile["frozen_attention"] * 100, marker="o", markersize=3, color="tab:orange", label="Frozen-S0 + attention")
    axis.plot(water_years, predictions_percentile["end_to_end_attention"] * 100, marker="o", markersize=3, color="tab:green", label="End-to-end attention")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--", label="50th percentile")
    axis.set_xlabel("Water year")
    axis.set_ylabel("April 1 Sierra SWE percentile (%)")
    axis.set_title("Observed vs. predicted relative Sierra SWE percentile, WY1985-2021")
    axis.set_ylim(-5, 105)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "main_observational_percentile_timeseries.png")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 2 (shape-only, for visualization only - NOT used for any metric).
    # -----------------------------------------------------------------
    def standardize(v):
        return (v - v.mean()) / v.std()

    fig, axis = plt.subplots(figsize=(13, 6), dpi=160)
    axis.plot(water_years, standardize(ucla_percentile), color="black", linewidth=2.2, label="UCLA observed (standardized)")
    axis.plot(water_years, standardize(predictions_percentile["S0"]), color="tab:blue", label="S0 (standardized)")
    axis.plot(water_years, standardize(predictions_percentile["frozen_attention"]), color="tab:orange", label="Frozen-S0 + attention (standardized)")
    axis.plot(water_years, standardize(predictions_percentile["end_to_end_attention"]), color="tab:green", label="End-to-end attention (standardized)")
    axis.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year")
    axis.set_ylabel("Standardized units (each series independently z-scored)")
    axis.set_title("Shape-only comparison: each series independently standardized for visualization")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "shape_only_observational_comparison.png")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 3: scatter, 3 panels.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160, sharex=True, sharey=True)
    colors = {"S0": "tab:blue", "frozen_attention": "tab:orange", "end_to_end_attention": "tab:green"}
    for axis, key in zip(axes, ("S0", "frozen_attention", "end_to_end_attention"), strict=True):
        axis.scatter(ucla_percentile, predictions_percentile[key], color=colors[key], s=24)
        axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axis.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
        axis.axvline(0.5, color="gray", linewidth=0.6, linestyle=":")
        axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        axis.set_xlabel("Observed percentile")
        axis.set_title(f"{name_labels[key]}\nr={results[key]['observational_pearson_r']:.3f}, rho={results[key]['observational_spearman_rho']:.3f}")
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted percentile")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "observational_scatter.png")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 4: compact skill metrics bar plot.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(9, 5.2), dpi=160)
    keys = ["S0", "frozen_attention", "end_to_end_attention"]
    labels = [name_labels[k] for k in keys]
    x_pos = np.arange(len(keys))
    width = 0.25
    axis.bar(x_pos - width, [results[k]["observational_pearson_r"] for k in keys], width, label="Pearson r", color="tab:blue")
    axis.bar(x_pos, [results[k]["observational_spearman_rho"] for k in keys], width, label="Spearman rho", color="tab:orange")
    axis.bar(x_pos + width, [results[k]["observational_wet_dry_accuracy"] for k in keys], width, label="Wet/dry accuracy", color="tab:green")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
    axis.set_xticks(x_pos); axis.set_xticklabels(labels, rotation=10, ha="right")
    axis.set_title("Observational skill metrics")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "observational_skill_metrics.png")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 5: sim vs obs correlation comparison.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(9, 5.2), dpi=160)
    width = 0.35
    axis.bar(x_pos - width / 2, [sim[k]["pearson_r"] for k in keys], width, label="Simulation validation Pearson r", color="tab:gray")
    axis.bar(x_pos + width / 2, [results[k]["observational_pearson_r"] for k in keys], width, label="Observational Pearson r", color="tab:red")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_pos); axis.set_xticklabels(labels, rotation=10, ha="right")
    axis.set_title("Simulation-validation vs. observational Pearson r")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "simulation_vs_observational_correlation.png")
    plt.close(fig)

    full_report = {
        "s0_checkpoint": str(S0_EXP_DIR / "best_checkpoint.pt"),
        "frozen_attention_checkpoint": str(FROZEN_EXP_DIR / "best_checkpoint.pt"),
        "end_to_end_attention_checkpoint": str(S1_EXP_DIR / "best_checkpoint.pt"),
        "simulation": sim,
        "observational": results,
        "n_observational_samples": 37,
        "water_year_range": [1985, 2021],
    }
    (OUTPUT_ROOT / "full_report.json").write_text(json.dumps(full_report, indent=2) + "\n")
    print(json.dumps(full_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
