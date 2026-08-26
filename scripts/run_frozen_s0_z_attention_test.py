from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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
    LATENT_D,
    LATENT_K,
    CMIP6TensorDataset,
    LatentSelfAttentionBlock,
    M1StaticCNN,
    collate_with_metadata,
    compute_feature_stats,
    compute_target_stats,
    pearson_r,
    r2_score,
    read_manifest,
    set_global_seed,
)


MODEL_SEED = 20260813
LOADER_SEED = 20260826


class FrozenS0Encoder(nn.Module):
    """Exact S0 backbone/project, permanently frozen and evaluated deterministically."""

    def __init__(self, s0: M1StaticCNN) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(s0.backbone)
        self.project = copy.deepcopy(s0.project)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        # Prevent a parent module's train() call from changing frozen-encoder mode.
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        with torch.no_grad():
            features = self.backbone(x.reshape(batch_size, months * channels, height, width))
            return self.project(features).reshape(batch_size, LATENT_K, LATENT_D)


class FrozenS0ZMLP(nn.Module):
    def __init__(self, encoder: FrozenS0Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.swe_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = False) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        outputs = {"swe_hat": self.swe_head(z.reshape(z.shape[0], -1)).squeeze(-1)}
        if collect_diagnostics:
            outputs["z_raw"] = z
        return outputs


class FrozenS0ZAttention(nn.Module):
    def __init__(self, encoder: FrozenS0Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = False) -> dict[str, torch.Tensor]:
        z_raw = self.encoder(x)
        attn_in = self.latent_block.norm1(z_raw)
        attn_out, weights = self.latent_block.attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=collect_diagnostics,
            average_attn_weights=False,
        )
        z_attn = z_raw + attn_out
        z_attn = z_attn + self.latent_block.ffn(self.latent_block.norm2(z_attn))
        outputs = {"swe_hat": self.swe_head(z_attn.reshape(z_attn.shape[0], -1)).squeeze(-1)}
        if collect_diagnostics:
            outputs.update({"z_raw": z_raw, "z_attn": z_attn, "attention_weights": weights})
        return outputs


def tensor_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def maximum_parameter_change(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return max(float((before[name] - after[name]).abs().max()) for name in before)


def summarize(predictions_z: list[np.ndarray], targets_z: list[np.ndarray], mu: float, sigma: float) -> dict[str, float]:
    predicted = np.concatenate(predictions_z) * sigma + mu
    observed = np.concatenate(targets_z) * sigma + mu
    return {
        "swe_r2": r2_score(observed, predicted),
        "swe_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "swe_mae": float(np.mean(np.abs(observed - predicted))),
        "swe_pearson_r": pearson_r(observed, predicted),
    }


def latent_rank(values: np.ndarray) -> float:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    singular = np.linalg.svd(flat - flat.mean(axis=0, keepdims=True), compute_uv=False)
    p = singular / max(float(singular.sum()), np.finfo(float).eps)
    return float(math.exp(-np.sum(np.where(p > 0.0, p * np.log(p), 0.0))))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mu: float,
    target_sigma: float,
    *,
    diagnostics: bool,
) -> tuple[dict[str, float], dict[str, Any] | None, list[dict[str, Any]] | None]:
    model.eval()
    model.encoder.eval()  # type: ignore[attr-defined]
    losses, predictions, targets = [], [], []
    raw, attended, weights = [], [], []
    with torch.no_grad():
        for inputs, batch_targets, _ in loader:
            outputs = model(inputs.to(device), collect_diagnostics=diagnostics)
            loss = F.mse_loss(outputs["swe_hat"], batch_targets[:, 0].to(device))
            losses.append(float(loss.cpu()))
            predictions.append(outputs["swe_hat"].cpu().numpy())
            targets.append(batch_targets[:, 0].numpy())
            if diagnostics:
                raw.append(outputs["z_raw"].cpu().numpy())
                attended.append(outputs["z_attn"].cpu().numpy())
                weights.append(outputs["attention_weights"].cpu().numpy())
    metrics = {"swe_loss": float(np.mean(losses)), **summarize(predictions, targets, target_mu, target_sigma)}
    if not diagnostics:
        return metrics, None, None
    raw_array, attended_array, weight_array = np.concatenate(raw), np.concatenate(attended), np.concatenate(weights)
    latent = {
        "frozen_z_effective_rank": latent_rank(raw_array),
        "z_attn_effective_rank": latent_rank(attended_array),
        "relative_frobenius_change": float(np.linalg.norm(attended_array - raw_array) / np.linalg.norm(raw_array)),
    }
    attention = []
    for head in range(weight_array.shape[1]):
        values = np.clip(weight_array[:, head], np.finfo(float).tiny, 1.0)
        entropy = -(values * np.log(values)).sum(axis=-1)
        row_max = values.max(axis=-1)
        attention.append(
            {
                "head": head,
                "mean_normalized_attention_entropy": float(entropy.mean() / math.log(8)),
                "average_rowwise_maximum_weight": float(row_max.mean()),
                "attention_weight_shape": json.dumps(list(weight_array.shape)),
            }
        )
    return metrics, latent, attention


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, index_by_identity: dict[tuple[str, str, str, str], int]) -> tuple[float, str]:
    model.train()
    model.encoder.eval()  # type: ignore[attr-defined]
    losses, batches = [], []
    for inputs, targets, metadata in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs.to(device))
        loss = F.mse_loss(outputs["swe_hat"], targets[:, 0].to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        batches.append([index_by_identity[(str(row["model_member"]), str(row["experiment"]), str(row["row_year"]), str(row["water_year"]))] for row in metadata])
    return float(np.mean(losses)), hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    for train_field, val_field, label, filename in (
        ("train_swe_r2", "val_swe_r2", "SWE R2", "frozen_s0_z_r2_by_epoch.png"),
        ("train_swe_loss", "val_swe_loss", "SWE loss", "frozen_s0_z_loss_by_epoch.png"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), dpi=160)
        for axis, variant in zip(axes, ("F0_frozen_S0_Z_MLP", "F1_frozen_S0_Z_attention"), strict=True):
            values = [row for row in rows if row["variant"] == variant]
            axis.plot([row["epoch"] for row in values], [row[train_field] for row in values], marker="o", markersize=3, label="Train (eval mode)")
            axis.plot([row["epoch"] for row in values], [row[val_field] for row in values], marker="o", markersize=3, label="Validation")
            best = min(values, key=lambda row: row["val_swe_loss"])
            axis.axvline(best["epoch"], color="tab:red", linestyle="--", label=f"Best validation epoch ({best['epoch']})")
            if train_field == "train_swe_r2":
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
            axis.set_title(variant)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def plot_attention(rows: list[dict[str, Any]], output_dir: Path) -> None:
    for metric, label in (("mean_normalized_attention_entropy", "Normalized attention entropy"), ("average_rowwise_maximum_weight", "Average row-wise maximum attention")):
        plt.figure(figsize=(8, 5), dpi=160)
        for stage, style in (("train", "-"), ("val", "--")):
            for head in (0, 1):
                values = [row for row in rows if row["stage"] == stage and row["head"] == head]
                plt.plot([row["epoch"] for row in values], [row[metric] for row in values], style, marker="o", markersize=3, label=f"{stage}, head {head}")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"F1 frozen-S0-Z attention: {label}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"f1_{metric}_by_epoch.png")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--s0-checkpoint", required=True)
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
    set_global_seed(MODEL_SEED)
    cache_dir = Path(args.cache_dir)
    manifest = read_manifest(cache_dir / "manifest.csv")
    split = json.loads(Path(args.split_json).read_text())
    physical = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    targets = np.load(cache_dir / "targets.npy")
    feature_mu, feature_sigma, _ = compute_feature_stats(physical, split["train"])
    target_mu, target_sigma = compute_target_stats(targets, split["train"])
    train_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["train"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    val_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["val"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    identity_index = {(str(row["model_member"]), str(row["experiment"]), str(row["row_year"]), str(row["water_year"])): index for index, row in enumerate(manifest)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.s0_checkpoint, map_location="cpu", weights_only=False)
    s0 = M1StaticCNN(38, include_auxiliary_heads=False)
    s0.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reference_encoder = FrozenS0Encoder(s0)
    initial_encoder_state = {name: value.detach().cpu().clone() for name, value in reference_encoder.state_dict().items()}
    initial_encoder_digest = tensor_digest(reference_encoder)
    has_dropout = any(isinstance(module, nn.Dropout) for module in reference_encoder.modules())
    has_batchnorm = any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in reference_encoder.modules())
    config = {
        "s0_checkpoint": str(args.s0_checkpoint), "s0_checkpoint_epoch": checkpoint["epoch"],
        "s0_checkpoint_best_val_total_loss": checkpoint["best_val_total_loss"],
        "model_seed": MODEL_SEED, "loader_seed": LOADER_SEED, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "frozen_encoder_parameter_count": sum(parameter.numel() for parameter in reference_encoder.parameters()),
        "encoder_contains_dropout": has_dropout, "encoder_contains_batchnorm": has_batchnorm,
        "encoder_mode": "eval throughout both head-training and evaluation passes",
        "initial_encoder_digest": initial_encoder_digest,
    }
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    all_rows, attention_rows, latent_rows, summary_rows = [], [], [], []
    encoder_digests: dict[str, str] = {}
    for variant in ("F0_frozen_S0_Z_MLP", "F1_frozen_S0_Z_attention"):
        set_global_seed(MODEL_SEED)
        encoder = FrozenS0Encoder(s0)
        model: nn.Module = FrozenS0ZMLP(encoder) if variant.startswith("F0") else FrozenS0ZAttention(encoder)
        model.to(device)
        fixed_inputs = next(iter(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_metadata)))[0].to(device)
        with torch.no_grad():
            fixed_z_before = model.encoder(fixed_inputs).detach().cpu().clone()  # type: ignore[attr-defined]
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(LOADER_SEED)
        train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(RandomSampler(train_dataset, generator=generator), args.batch_size, drop_last=False), num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
        train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
        best_loss, best_epoch, patience = float("inf"), 0, 0
        checkpoint_dir = output_root / "checkpoints" / variant
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.max_epochs + 1):
            optimization_loss, batch_digest = train_epoch(model, train_loader, optimizer, device, identity_index)
            use_diagnostics = variant.startswith("F1")
            train_metrics, train_latent, train_attention = evaluate(model, train_eval_loader, device, float(target_mu[0]), float(target_sigma[0]), diagnostics=use_diagnostics)
            val_metrics, val_latent, val_attention = evaluate(model, val_loader, device, float(target_mu[0]), float(target_sigma[0]), diagnostics=use_diagnostics)
            row = {"variant": variant, "epoch": epoch, "train_optimization_swe_loss": optimization_loss, "train_swe_loss": train_metrics["swe_loss"], "val_swe_loss": val_metrics["swe_loss"], "train_swe_r2": train_metrics["swe_r2"], "val_swe_r2": val_metrics["swe_r2"], "train_swe_rmse": train_metrics["swe_rmse"], "val_swe_rmse": val_metrics["swe_rmse"], "train_swe_mae": train_metrics["swe_mae"], "val_swe_mae": val_metrics["swe_mae"], "train_swe_pearson_r": train_metrics["swe_pearson_r"], "val_swe_pearson_r": val_metrics["swe_pearson_r"], "train_batch_digest": batch_digest}
            all_rows.append(row)
            if use_diagnostics:
                assert train_latent is not None and val_latent is not None and train_attention is not None and val_attention is not None
                latent_rows.extend([{"epoch": epoch, "stage": "train", **train_latent}, {"epoch": epoch, "stage": "val", **val_latent}])
                attention_rows.extend([{"epoch": epoch, "stage": "train", **item} for item in train_attention])
                attention_rows.extend([{"epoch": epoch, "stage": "val", **item} for item in val_attention])
            if val_metrics["swe_loss"] < best_loss:
                best_loss, best_epoch, patience = val_metrics["swe_loss"], epoch, 0
                torch.save({"epoch": epoch, "trainable_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith("encoder.")}, "val_swe_loss": best_loss}, checkpoint_dir / "best_trainable_downstream.pt")
            else:
                patience += 1
            print(f"{variant} epoch={epoch} train_r2={train_metrics['swe_r2']:.6f} val_r2={val_metrics['swe_r2']:.6f} val_loss={val_metrics['swe_loss']:.6f}", flush=True)
            if patience >= args.patience:
                break
        after_state = {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()}  # type: ignore[attr-defined]
        encoder_digests[variant] = tensor_digest(model.encoder)  # type: ignore[attr-defined]
        with torch.no_grad():
            fixed_z_after = model.encoder(fixed_inputs).detach().cpu()  # type: ignore[attr-defined]
        values = [row for row in all_rows if row["variant"] == variant]
        best = min(values, key=lambda item: item["val_swe_loss"])
        max_train = max(values, key=lambda item: item["train_swe_r2"])
        summary_rows.append({"variant": variant, "best_epoch": best_epoch, "best_train_r2": best["train_swe_r2"], "best_val_r2": best["val_swe_r2"], "best_gap": best["train_swe_r2"] - best["val_swe_r2"], "max_train_r2": max_train["train_swe_r2"], "max_train_epoch": max_train["epoch"], "val_r2_at_max_train": max_train["val_swe_r2"], "final_train_r2": values[-1]["train_swe_r2"], "final_val_r2": values[-1]["val_swe_r2"], "max_encoder_parameter_change": maximum_parameter_change(initial_encoder_state, after_state), "max_fixed_batch_z_change": float((fixed_z_before - fixed_z_after).abs().max()), "encoder_all_requires_grad_false": all(not parameter.requires_grad for parameter in model.encoder.parameters())})

    write_csv(output_root / "frozen_s0_z_metrics.csv", all_rows)
    write_csv(output_root / "f1_attention_diagnostics.csv", attention_rows)
    write_csv(output_root / "f1_latent_diagnostics.csv", latent_rows)
    write_csv(output_root / "comparison_summary.csv", summary_rows)
    plot_comparison(all_rows, plot_dir)
    plot_attention(attention_rows, plot_dir)
    same_batches = all(len({row["train_batch_digest"] for row in all_rows if row["epoch"] == epoch}) == 1 for epoch in {int(row["epoch"]) for row in all_rows})
    metadata = {**config, "encoder_digest_by_variant": encoder_digests, "same_encoder_digest_for_f0_f1": len(set(encoder_digests.values())) == 1, "batch_order_identical_for_shared_epochs": same_batches, "attention_weight_shape": json.loads(attention_rows[0]["attention_weight_shape"])}
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = ["# Frozen S0 Z Mechanism-Isolation Test", "", "The frozen encoder is the epoch-27 S0 historical-replay best checkpoint. It contains no dropout and no BatchNorm; it was forced to eval mode throughout. The downstream modules alone were optimized.", "", "| Variant | Best epoch | Train R2 at best | Val R2 at best | R2 gap | Max train R2 (epoch) | Val R2 at max train | Final train R2 | Final val R2 | Max encoder change | Max fixed-Z change |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append("| {variant} | {best_epoch} | {best_train_r2:.6f} | {best_val_r2:.6f} | {best_gap:.6f} | {max_train_r2:.6f} ({max_train_epoch}) | {val_r2_at_max_train:.6f} | {final_train_r2:.6f} | {final_val_r2:.6f} | {max_encoder_parameter_change:.3g} | {max_fixed_batch_z_change:.3g} |".format(**row))
    lines.extend(["", "Interpret F0 and F1 only after verifying `run_metadata.json`: both variants must report identical encoder digests, zero encoder-parameter change, zero fixed-batch-Z change, and identical batch order for shared epochs."])
    (plot_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
