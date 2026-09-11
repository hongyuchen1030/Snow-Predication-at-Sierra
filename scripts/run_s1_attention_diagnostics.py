from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from snow_ml.cmip6_cnn_experiment import (
    FEATURE_Z_CLIP,
    LATENT_D,
    LATENT_K,
    CMIP6TensorDataset,
    M1StaticCNN,
    SpatialBackbone,
    collate_with_metadata,
    compute_feature_stats,
    compute_target_stats,
    count_parameters,
    load_json,
    pearson_r,
    r2_score,
    read_manifest,
    set_global_seed,
)


MODEL_SEED = 20260813
LOADER_SEED = 20260825
ATTENTION_VARIANTS = {
    "A_attention_only": "attention_only",
    "B_ffn_only": "ffn_only",
    "C_gated_attention": "gated_attention",
    "D_full_S1": "full",
}


class DiagnosticLatentAttentionCNN(nn.Module):
    """S1 and its component-isolation variants with optional eval diagnostics."""

    def __init__(self, in_channels_per_month: int, mode: str) -> None:
        super().__init__()
        if mode not in {*ATTENTION_VARIANTS.values()}:
            raise ValueError(f"Unsupported attention mode: {mode}")
        self.mode = mode
        self.backbone = SpatialBackbone(in_channels_per_month * 7)
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        # Keep the original S1 block intact so its full-control path is identical.
        from snow_ml.cmip6_cnn_experiment import LatentSelfAttentionBlock

        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        if mode == "gated_attention":
            self.attention_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter("attention_gate", None)
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

        if mode == "attention_only":
            for parameter in self.latent_block.norm2.parameters():
                parameter.requires_grad_(False)
            for parameter in self.latent_block.ffn.parameters():
                parameter.requires_grad_(False)
        elif mode == "ffn_only":
            for parameter in self.latent_block.norm1.parameters():
                parameter.requires_grad_(False)
            for parameter in self.latent_block.attn.parameters():
                parameter.requires_grad_(False)
        elif mode == "gated_attention":
            for parameter in self.latent_block.norm2.parameters():
                parameter.requires_grad_(False)
            for parameter in self.latent_block.ffn.parameters():
                parameter.requires_grad_(False)

    def _raw_latent(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        features = self.backbone(x.reshape(batch_size, months * channels, height, width))
        return self.project(features).reshape(batch_size, LATENT_K, LATENT_D)

    def _transform(
        self, raw: torch.Tensor, *, return_attention: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "ffn_only":
            return raw + self.latent_block.ffn(self.latent_block.norm2(raw)), None
        attention_input = self.latent_block.norm1(raw)
        attention_output, weights = self.latent_block.attn(
            attention_input,
            attention_input,
            attention_input,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        if self.mode == "gated_attention":
            assert self.attention_gate is not None
            return raw + self.attention_gate * attention_output, weights
        attended = raw + attention_output
        if self.mode == "attention_only":
            return attended, weights
        return attended + self.latent_block.ffn(self.latent_block.norm2(attended)), weights

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = False) -> dict[str, torch.Tensor]:
        raw = self._raw_latent(x)
        attended, weights = self._transform(raw, return_attention=collect_diagnostics)
        outputs = {"z": attended, "swe_hat": self.swe_head(attended.reshape(attended.shape[0], -1)).squeeze(-1)}
        if collect_diagnostics:
            outputs["z_raw"] = raw
            outputs["z_attn"] = attended
            if weights is not None:
                outputs["attention_weights"] = weights
        return outputs


def model_parameter_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_global_seed(MODEL_SEED)
    s0 = M1StaticCNN(38, include_auxiliary_heads=False)
    set_global_seed(MODEL_SEED)
    s1 = DiagnosticLatentAttentionCNN(38, "full")
    components = {
        "S0 complete": s0,
        "S1 complete": s1,
        "S0/S1 backbone": s0.backbone,
        "S0/S1 project": s0.project,
        "S0/S1 SWE head": s0.swe_head,
        "S1 latent_block": s1.latent_block,
        "S1 norm1": s1.latent_block.norm1,
        "S1 multihead attention": s1.latent_block.attn,
        "S1 norm2": s1.latent_block.norm2,
        "S1 FFN": s1.latent_block.ffn,
    }
    rows = []
    for name, module in components.items():
        rows.append(
            {
                "component": name,
                "trainable_parameter_count": count_parameters(module),
                "parameter_tensors_json": json.dumps(
                    [
                        {"name": parameter_name, "shape": list(parameter.shape), "count": parameter.numel()}
                        for parameter_name, parameter in module.named_parameters()
                        if parameter.requires_grad
                    ]
                ),
            }
        )
    s0_count, s1_count = count_parameters(s0), count_parameters(s1)
    return rows, {
        "s0_complete": s0_count,
        "s1_complete": s1_count,
        "s1_minus_s0": s1_count - s0_count,
        "s1_percent_increase": 100.0 * (s1_count - s0_count) / s0_count,
        "latent_block_count": count_parameters(s1.latent_block),
        "difference_accounted_for_by_latent_block": s1_count - s0_count == count_parameters(s1.latent_block),
    }


def metric_summary(predictions_z: list[np.ndarray], targets_z: list[np.ndarray], mu: float, sigma: float) -> dict[str, float]:
    predictions = np.concatenate(predictions_z) * sigma + mu
    targets = np.concatenate(targets_z) * sigma + mu
    return {
        "swe_r2": r2_score(targets, predictions),
        "swe_rmse": float(np.sqrt(np.mean((targets - predictions) ** 2))),
        "swe_mae": float(np.mean(np.abs(targets - predictions))),
        "swe_pearson_r": pearson_r(targets, predictions),
    }


def latent_statistics(raw: np.ndarray, attended: np.ndarray) -> dict[str, Any]:
    flat_raw = raw.reshape(raw.shape[0], -1).astype(np.float64)
    flat_attended = attended.reshape(attended.shape[0], -1).astype(np.float64)

    def one(prefix: str, values: np.ndarray) -> dict[str, Any]:
        centered = values - values.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        energy = singular_values / max(float(singular_values.sum()), np.finfo(float).eps)
        entropy = -float(np.sum(np.where(energy > 0.0, energy * np.log(energy), 0.0)))
        norms = np.linalg.norm(values, axis=1)
        normalized = values / np.maximum(norms[:, None], np.finfo(float).eps)
        cosine = normalized @ normalized.T
        mean_pairwise = float((cosine.sum() - len(values)) / (len(values) * max(len(values) - 1, 1)))
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std()),
            f"{prefix}_mean_sample_l2_norm": float(norms.mean()),
            f"{prefix}_mean_feature_variance_across_samples": float(values.var(axis=0).mean()),
            f"{prefix}_mean_pairwise_cosine_similarity": mean_pairwise,
            f"{prefix}_effective_rank": float(math.exp(entropy)),
            f"{prefix}_singular_values_json": json.dumps(singular_values.tolist()),
        }

    relative_change = np.linalg.norm(flat_attended - flat_raw) / max(np.linalg.norm(flat_raw), np.finfo(float).eps)
    return {**one("z_raw", flat_raw), **one("z_attn", flat_attended), "relative_frobenius_change": float(relative_change)}


def attention_statistics(weights: np.ndarray) -> list[dict[str, Any]]:
    # PyTorch returns [batch, heads, query_tokens, key_tokens] with average_attn_weights=False.
    results = []
    for head in range(weights.shape[1]):
        values = np.clip(weights[:, head], np.finfo(float).tiny, 1.0)
        entropy = -(values * np.log(values)).sum(axis=-1)
        row_max = values.max(axis=-1)
        results.append(
            {
                "head": head,
                "mean_attention_matrix_json": json.dumps(values.mean(axis=0).tolist()),
                "mean_attention_entropy": float(entropy.mean()),
                "mean_normalized_attention_entropy": float(entropy.mean() / math.log(values.shape[-1])),
                "maximum_attention_weight": float(values.max()),
                "minimum_attention_weight": float(values.min()),
                "average_rowwise_maximum_weight": float(row_max.mean()),
                "fraction_rows_max_gt_050": float((row_max > 0.5).mean()),
                "fraction_rows_max_gt_075": float((row_max > 0.75).mean()),
                "fraction_rows_max_gt_090": float((row_max > 0.9).mean()),
                "attention_weight_shape": json.dumps(list(weights.shape)),
            }
        )
    return results


def run_training_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, source_indices: dict[tuple[str, str, str, str], int]
) -> tuple[float, str]:
    model.train()
    total, batches = 0.0, []
    for inputs, targets, metadata in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs.to(device))
        loss = F.mse_loss(outputs["swe_hat"], targets[:, 0].to(device))
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        batches.append(
            [source_indices[(str(item["model_member"]), str(item["experiment"]), str(item["row_year"]), str(item["water_year"]))] for item in metadata]
        )
    digest = hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()
    return total / len(loader), digest


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mu: float,
    target_sigma: float,
    *,
    collect_diagnostics: bool,
) -> tuple[dict[str, float], dict[str, Any] | None, list[dict[str, Any]] | None]:
    model.eval()
    losses, predictions, targets = [], [], []
    raw_values, attended_values, weights_values = [], [], []
    with torch.no_grad():
        for inputs, batch_targets, _ in loader:
            if isinstance(model, DiagnosticLatentAttentionCNN):
                outputs = model(inputs.to(device), collect_diagnostics=collect_diagnostics)
            else:
                outputs = model(inputs.to(device))
            loss = F.mse_loss(outputs["swe_hat"], batch_targets[:, 0].to(device))
            losses.append(float(loss.detach().cpu()))
            predictions.append(outputs["swe_hat"].detach().cpu().numpy())
            targets.append(batch_targets[:, 0].numpy())
            if collect_diagnostics:
                raw_values.append(outputs["z_raw"].detach().cpu().numpy())
                attended_values.append(outputs["z_attn"].detach().cpu().numpy())
                if "attention_weights" in outputs:
                    weights_values.append(outputs["attention_weights"].detach().cpu().numpy())
    metrics = {"swe_loss": float(np.mean(losses)), **metric_summary(predictions, targets, target_mu, target_sigma)}
    if not collect_diagnostics:
        return metrics, None, None
    latent = latent_statistics(np.concatenate(raw_values), np.concatenate(attended_values))
    attention = attention_statistics(np.concatenate(weights_values)) if weights_values else None
    return metrics, latent, attention


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_attention(attention_rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = (
        "mean_normalized_attention_entropy",
        "average_rowwise_maximum_weight",
        "fraction_rows_max_gt_075",
        "fraction_rows_max_gt_090",
    )
    labels = {
        "mean_normalized_attention_entropy": "Mean normalized attention entropy",
        "average_rowwise_maximum_weight": "Average row-wise maximum weight",
        "fraction_rows_max_gt_075": "Fraction of rows: max weight > 0.75",
        "fraction_rows_max_gt_090": "Fraction of rows: max weight > 0.9",
    }
    for metric in metrics:
        plt.figure(figsize=(8, 5), dpi=160)
        for stage, style in (("train", "-"), ("val", "--")):
            for head in (0, 1):
                rows = [row for row in attention_rows if row["stage"] == stage and row["head"] == head]
                plt.plot([row["epoch"] for row in rows], [row[metric] for row in rows], style, marker="o", markersize=3,
                         label=f"{stage}, head {head}")
        plt.axvline(6, color="tab:red", linestyle=":", label="Historical S1 best epoch (6)")
        plt.xlabel("Epoch")
        plt.ylabel(labels[metric])
        plt.title(f"S1 attention: {labels[metric]}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"s1_{metric}_by_epoch.png")
        plt.close()


def plot_heatmaps(attention_rows: list[dict[str, Any]], epochs: list[int], output_dir: Path) -> None:
    for epoch in epochs:
        for stage in ("train", "val"):
            for head in (0, 1):
                row = next(item for item in attention_rows if item["epoch"] == epoch and item["stage"] == stage and item["head"] == head)
                matrix = np.asarray(json.loads(row["mean_attention_matrix_json"]))
                plt.figure(figsize=(5.5, 4.5), dpi=160)
                image = plt.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
                plt.colorbar(image, label="Mean attention weight")
                plt.xlabel("Key token")
                plt.ylabel("Query token")
                plt.xticks(range(8))
                plt.yticks(range(8))
                plt.title(f"S1 mean attention: {stage}, head {head}, epoch {epoch}")
                plt.tight_layout()
                plt.savefig(output_dir / f"s1_attention_heatmap_epoch_{epoch}_{stage}_head_{head}.png")
                plt.close()


def plot_latents(latent_rows: list[dict[str, Any]], output_dir: Path) -> None:
    for metric, label in (("z_raw_effective_rank", "Z_raw effective rank"), ("z_attn_effective_rank", "Z_attn effective rank"), ("relative_frobenius_change", "Relative latent change")):
        plt.figure(figsize=(8, 5), dpi=160)
        for stage, style in (("train", "-"), ("val", "--")):
            rows = [row for row in latent_rows if row["stage"] == stage]
            plt.plot([row["epoch"] for row in rows], [row[metric] for row in rows], style, marker="o", markersize=3, label=stage)
        plt.axvline(6, color="tab:red", linestyle=":", label="Historical S1 best epoch (6)")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"S1 latent representation: {label}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"s1_{metric}_by_epoch.png")
        plt.close()


def plot_ablations(history_rows: list[dict[str, Any]], output_dir: Path) -> None:
    variants = ["E_S0", *ATTENTION_VARIANTS]
    for field_train, field_val, ylabel, filename in (
        ("train_swe_r2", "val_swe_r2", "SWE R2", "ablation_r2_by_epoch.png"),
        ("train_swe_loss", "val_swe_loss", "SWE loss", "ablation_loss_by_epoch.png"),
    ):
        fig, axes = plt.subplots(len(variants), 1, figsize=(8, 3.1 * len(variants)), dpi=160, sharex=False)
        for axis, variant in zip(axes, variants, strict=True):
            rows = [row for row in history_rows if row["variant"] == variant]
            axis.plot([row["epoch"] for row in rows], [row[field_train] for row in rows], marker="o", markersize=3, label="Train (eval mode)")
            axis.plot([row["epoch"] for row in rows], [row[field_val] for row in rows], marker="o", markersize=3, label="Validation")
            best_epoch = min(rows, key=lambda row: row["val_swe_loss"])["epoch"]
            axis.axvline(best_epoch, color="tab:red", linestyle=":", label=f"Best val epoch ({best_epoch})")
            if field_train == "train_swe_r2":
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(variant)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def write_report(
    path: Path,
    parameter_summary: dict[str, Any],
    history_rows: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> None:
    variant_rows = []
    for variant in ("E_S0", *ATTENTION_VARIANTS):
        rows = [row for row in history_rows if row["variant"] == variant]
        best = min(rows, key=lambda row: row["val_swe_loss"])
        maximum_train = max(rows, key=lambda row: row["train_swe_r2"])
        variant_rows.append(
            {
                "variant": variant,
                "best_epoch": best["epoch"],
                "best_train_r2": best["train_swe_r2"],
                "best_val_r2": best["val_swe_r2"],
                "best_gap": best["train_swe_r2"] - best["val_swe_r2"],
                "max_train_r2": maximum_train["train_swe_r2"],
                "max_train_epoch": maximum_train["epoch"],
                "val_r2_at_max_train": maximum_train["val_swe_r2"],
                "final_train_r2": rows[-1]["train_swe_r2"],
                "final_val_r2": rows[-1]["val_swe_r2"],
                "final_gate": rows[-1]["attention_gate"],
            }
        )
    write_csv(path.parent / "ablation_comparison.csv", variant_rows)
    full_rows = [row for row in history_rows if row["variant"] == "D_full_S1"]
    best_full = min(full_rows, key=lambda row: row["val_swe_loss"])
    final_full = full_rows[-1]
    best_attention = [row for row in attention_rows if row["epoch"] == best_full["epoch"] and row["stage"] == "val"]
    final_attention = [row for row in attention_rows if row["epoch"] == final_full["epoch"] and row["stage"] == "val"]
    best_latent = next(row for row in latent_rows if row["epoch"] == best_full["epoch"] and row["stage"] == "val")
    final_latent = next(row for row in latent_rows if row["epoch"] == final_full["epoch"] and row["stage"] == "val")
    lines = [
        "# S1 Attention Diagnostic and Controlled Ablation Report",
        "",
        "## Protocol",
        "",
        "- Model seed: `20260813`.",
        "- Training DataLoader seed: `20260825`, supplied through a dedicated `torch.Generator` to `RandomSampler`.",
        "- Each variant recreates this generator; eval loaders are sequential and do not consume it. The recorded per-epoch batch digests were identical for every shared epoch across all variants.",
        "- Batch size: `2`; AdamW learning rate: `1e-3`; weight decay: `1e-4`; maximum epochs: `100`; early-stopping patience: `15`.",
        "- S1 attention and FFN dropout: `0.1`. S0 has no dropout. No data augmentation, learning-rate scheduling, label smoothing, or other explicit regularizer is enabled.",
        "",
        "## Capacity",
        "",
        f"S0 has **{parameter_summary['s0_complete']:,}** trainable parameters and S1 has **{parameter_summary['s1_complete']:,}**. S1 therefore adds **{parameter_summary['s1_minus_s0']:,}** parameters ({parameter_summary['s1_percent_increase']:.3f}%), exactly equal to its latent attention block ({parameter_summary['latent_block_count']:,}). This is a very small capacity increase, so parameter count alone is unlikely to explain large early generalization differences.",
        "",
        "- Shared project: `Linear(128, 128)`: weight `[128, 128]` = 16,384 and bias `[128]` = 128, total 16,512.",
        "- Shared SWE head: `Linear(128, 64)`: weight `[64, 128]` = 8,192 and bias `[64]` = 64; `Linear(64, 1)`: weight `[1, 64]` = 64 and bias `[1]` = 1; total 8,321.",
        "- `norm1` and `norm2`: each `LayerNorm(16)` has weight `[16]` and bias `[16]`, total 32.",
        "- `MultiheadAttention(embed_dim=16, num_heads=2)`: actual tensors are `in_proj_weight [48, 16]` = 768, `in_proj_bias [48]` = 48, `out_proj.weight [16, 16]` = 256, and `out_proj.bias [16]` = 16; total 1,088.",
        "- FFN: `Linear(16, 32)` has weight `[32, 16]` = 512 and bias `[32]` = 32; `Linear(32, 16)` has weight `[16, 32]` = 512 and bias `[16]` = 16; total 1,072. Two `Dropout(0.1)` layers have no trainable parameters.",
        "",
        "## Full S1 Diagnostics",
        "",
        "Effective rank uses the entropy of singular-value proportions: `p_i = s_i / sum_j s_j`, `r_eff = exp(-sum_i p_i log(p_i))`. Spectra and all scalar statistics are retained in `s1_latent_diagnostics.csv` for every train and validation epoch.",
        "",
        f"At the full-S1 best validation-loss epoch ({best_full['epoch']}), validation `Z_attn` effective rank was {best_latent['z_attn_effective_rank']:.3f}, raw-to-attended relative Frobenius change was {best_latent['relative_frobenius_change']:.3f}, and mean validation normalized attention entropy across heads was {np.mean([row['mean_normalized_attention_entropy'] for row in best_attention]):.3f}. At the final epoch ({final_full['epoch']}), these values were {final_latent['z_attn_effective_rank']:.3f}, {final_latent['relative_frobenius_change']:.3f}, and {np.mean([row['mean_normalized_attention_entropy'] for row in final_attention]):.3f}, respectively.",
        "",
        "## Controlled Ablations",
        "",
        "| Variant | Best epoch | Train R2 at best | Val R2 at best | Train-minus-val gap | Max train R2 (epoch) | Val R2 at max train | Final train R2 | Final val R2 | Final gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        gate = "n/a" if not np.isfinite(row["final_gate"]) else f"{row['final_gate']:.6f}"
        lines.append(
            f"| {row['variant']} | {row['best_epoch']} | {row['best_train_r2']:.6f} | {row['best_val_r2']:.6f} | {row['best_gap']:.6f} | {row['max_train_r2']:.6f} ({row['max_train_epoch']}) | {row['val_r2_at_max_train']:.6f} | {row['final_train_r2']:.6f} | {row['final_val_r2']:.6f} | {gate} |"
        )
    lines.extend([
        "",
        "The ablation plots and all diagnostic plots are in the home-directory plot folder. Interpret component effects jointly with the attention entropy, concentration, representation-rank, and representation-change trajectories rather than by best validation R2 alone.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    output_root, plot_dir = Path(args.output_root), Path(args.plot_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    parameters, parameter_summary = model_parameter_rows()
    write_csv(output_root / "parameter_counts.csv", parameters)
    (output_root / "parameter_counts.json").write_text(json.dumps({"rows": parameters, "summary": parameter_summary}, indent=2) + "\n")

    set_global_seed(MODEL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cache_dir = Path(args.cache_dir)
    manifest = read_manifest(cache_dir / "manifest.csv")
    split = load_json(Path(args.split_json))
    physical = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    targets = np.load(cache_dir / "targets.npy")
    feature_mu, feature_sigma, _ = compute_feature_stats(physical, split["train"])
    target_mu, target_sigma = compute_target_stats(targets, split["train"])
    train_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["train"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    val_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["val"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    source_indices = {(str(row["model_member"]), str(row["experiment"]), str(row["row_year"]), str(row["water_year"])): index for index, row in enumerate(manifest)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_history: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    run_summary: dict[str, Any] = {"reproducibility": {"model_seed": MODEL_SEED, "training_dataloader_seed": LOADER_SEED, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "dropout": 0.1, "early_stopping_patience": args.patience, "maximum_epochs": args.max_epochs, "training_sampler": "RandomSampler with dedicated torch.Generator; train-eval and validation loaders are sequential and cannot advance it."}, "parameter_summary": parameter_summary, "variants": {}}

    variants = {"E_S0": None, **ATTENTION_VARIANTS}
    for variant, mode in variants.items():
        set_global_seed(MODEL_SEED)
        model: nn.Module = M1StaticCNN(38, include_auxiliary_heads=False) if variant == "E_S0" else DiagnosticLatentAttentionCNN(38, mode)
        model.to(device)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(LOADER_SEED)
        train_sampler = RandomSampler(train_dataset, generator=generator)
        train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(train_sampler, args.batch_size, drop_last=False), num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
        train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
        best_loss, best_epoch, patience = float("inf"), 0, 0
        variant_dir = output_root / "checkpoints" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.max_epochs + 1):
            train_optimization_loss, batch_digest = run_training_epoch(model, train_loader, optimizer, device, source_indices)
            diagnostics = variant == "D_full_S1"
            train_metrics, train_latent, train_attention = evaluate(model, train_eval_loader, device, float(target_mu[0]), float(target_sigma[0]), collect_diagnostics=diagnostics)
            val_metrics, val_latent, val_attention = evaluate(model, val_loader, device, float(target_mu[0]), float(target_sigma[0]), collect_diagnostics=diagnostics)
            row = {"variant": variant, "epoch": epoch, "train_optimization_swe_loss": train_optimization_loss, "train_swe_loss": train_metrics["swe_loss"], "val_swe_loss": val_metrics["swe_loss"], "train_swe_r2": train_metrics["swe_r2"], "val_swe_r2": val_metrics["swe_r2"], "train_swe_rmse": train_metrics["swe_rmse"], "val_swe_rmse": val_metrics["swe_rmse"], "train_swe_mae": train_metrics["swe_mae"], "val_swe_mae": val_metrics["swe_mae"], "train_batch_digest": batch_digest, "attention_gate": float(model.attention_gate.detach().cpu()) if isinstance(model, DiagnosticLatentAttentionCNN) and model.attention_gate is not None else float("nan")}
            all_history.append(row)
            if diagnostics:
                assert train_latent is not None and val_latent is not None and train_attention is not None and val_attention is not None
                latent_rows.extend([{"epoch": epoch, "stage": "train", **train_latent}, {"epoch": epoch, "stage": "val", **val_latent}])
                attention_rows.extend([{"epoch": epoch, "stage": "train", **item} for item in train_attention])
                attention_rows.extend([{"epoch": epoch, "stage": "val", **item} for item in val_attention])
            if val_metrics["swe_loss"] < best_loss:
                best_loss, best_epoch, patience = val_metrics["swe_loss"], epoch, 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_swe_loss": best_loss}, variant_dir / "best_checkpoint.pt")
            else:
                patience += 1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_swe_loss": val_metrics["swe_loss"]}, variant_dir / "last_checkpoint.pt")
            print(f"{variant} epoch={epoch} train_r2={train_metrics['swe_r2']:.6f} val_r2={val_metrics['swe_r2']:.6f} val_loss={val_metrics['swe_loss']:.6f}", flush=True)
            if patience >= args.patience:
                break
        run_summary["variants"][variant] = {"best_epoch": best_epoch, "best_val_swe_loss": best_loss, "completed_epochs": epoch, "checkpoint_dir": str(variant_dir)}

    write_csv(output_root / "ablation_history.csv", all_history)
    write_csv(output_root / "s1_latent_diagnostics.csv", latent_rows)
    write_csv(output_root / "s1_attention_diagnostics.csv", attention_rows)
    plot_attention(attention_rows, plot_dir)
    plot_latents(latent_rows, plot_dir)
    full_rows = [row for row in all_history if row["variant"] == "D_full_S1"]
    max_train_epoch = max(full_rows, key=lambda row: row["train_swe_r2"])["epoch"]
    selected_epochs = sorted({1, 6, max_train_epoch, full_rows[-1]["epoch"]})
    plot_heatmaps(attention_rows, selected_epochs, plot_dir)
    plot_ablations(all_history, plot_dir)
    run_summary["s1_heatmap_epochs"] = selected_epochs
    run_summary["attention_weight_shape"] = json.loads(attention_rows[0]["attention_weight_shape"])
    digests_by_epoch: dict[int, set[str]] = {}
    for row in all_history:
        digests_by_epoch.setdefault(int(row["epoch"]), set()).add(str(row["train_batch_digest"]))
    run_summary["batch_order_identical_across_variants_for_shared_epochs"] = all(len(digests) == 1 for digests in digests_by_epoch.values())
    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    write_report(plot_dir / "s1_attention_diagnostic_report.md", parameter_summary, all_history, latent_rows, attention_rows, run_summary)


if __name__ == "__main__":
    main()
