#!/usr/bin/env python3
"""Train ONLY the frozen-S0 + attention downstream model (no F0 plain-MLP branch) on the
percentile SWE target, initializing the frozen encoder from the NEW percentile-trained S0
checkpoint. Adapted from scripts/run_frozen_s0_z_attention_test.py's F1 branch - same
architecture, same freezing/verification logic, same checkpoint-selection-by-val-loss
convention - just a single model, a different encoder source checkpoint, and the
percentile-target predictor cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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
    spearman_rho,
    wet_dry_accuracy,
    read_manifest,
    set_global_seed,
)

MODEL_SEED = 20260813
LOADER_SEED = 20260826


class FrozenS0Encoder(nn.Module):
    def __init__(self, s0: M1StaticCNN) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(s0.backbone)
        self.project = copy.deepcopy(s0.project)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        with torch.no_grad():
            features = self.backbone(x.reshape(batch_size, months * channels, height, width))
            return self.project(features).reshape(batch_size, LATENT_K, LATENT_D)


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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_raw = self.encoder(x)
        attn_in = self.latent_block.norm1(z_raw)
        attn_out, _ = self.latent_block.attn(attn_in, attn_in, attn_in, need_weights=False)
        z_attn = z_raw + attn_out
        z_attn = z_attn + self.latent_block.ffn(self.latent_block.norm2(z_attn))
        return {"swe_hat": self.swe_head(z_attn.reshape(z_attn.shape[0], -1)).squeeze(-1)}


def tensor_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize(predictions_z, targets_z, mu: float, sigma: float) -> dict[str, float]:
    predicted = np.concatenate(predictions_z) * sigma + mu
    observed = np.concatenate(targets_z) * sigma + mu
    return {
        "swe_r2": r2_score(observed, predicted),
        "swe_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "swe_mae": float(np.mean(np.abs(observed - predicted))),
        "swe_pearson_r": pearson_r(observed, predicted),
        "swe_spearman_rho": spearman_rho(observed, predicted),
        "swe_wet_dry_accuracy": wet_dry_accuracy(observed, predicted),
    }


def evaluate(model, loader, device, target_mu: float, target_sigma: float):
    model.eval()
    model.encoder.eval()
    losses, predictions, targets = [], [], []
    with torch.no_grad():
        for inputs, batch_targets, _ in loader:
            outputs = model(inputs.to(device))
            loss = F.mse_loss(outputs["swe_hat"], batch_targets[:, 0].to(device))
            losses.append(float(loss.cpu()))
            predictions.append(outputs["swe_hat"].cpu().numpy())
            targets.append(batch_targets[:, 0].numpy())
    metrics = {"swe_loss": float(np.mean(losses)), **summarize(predictions, targets, target_mu, target_sigma)}
    return metrics


def train_epoch(model, loader, optimizer, device):
    model.train()
    model.encoder.eval()
    losses = []
    for inputs, targets, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs.to(device))
        loss = F.mse_loss(outputs["swe_hat"], targets[:, 0].to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--s0-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    set_global_seed(MODEL_SEED)

    cache_dir = Path(args.cache_dir)
    manifest = read_manifest(cache_dir / "manifest.csv")
    split = json.loads(Path(args.split_json).read_text())
    physical = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    targets = np.load(cache_dir / "targets.npy")
    feature_mu, feature_sigma, _ = compute_feature_stats(physical, split["train"])
    target_mu, target_sigma = compute_target_stats(targets, split["train"])
    np.savez_compressed(output_dir / "normalization_stats.npz", feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)

    train_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["train"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    val_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["val"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.s0_checkpoint, map_location="cpu", weights_only=False)
    s0 = M1StaticCNN(38, include_auxiliary_heads=False)
    s0.load_state_dict(checkpoint["model_state_dict"], strict=True)
    encoder = FrozenS0Encoder(s0)
    initial_encoder_state = {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()}
    initial_encoder_digest = tensor_digest(encoder)

    model = FrozenS0ZAttention(encoder)
    model.to(device)

    fixed_inputs = next(iter(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_metadata)))[0].to(device)
    with torch.no_grad():
        fixed_z_before = model.encoder(fixed_inputs).detach().cpu().clone()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(LOADER_SEED)
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(RandomSampler(train_dataset, generator=generator), args.batch_size, drop_last=False), num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)

    history_rows: list[dict[str, Any]] = []
    best_loss, best_epoch, patience = float("inf"), 0, 0
    for epoch in range(1, args.max_epochs + 1):
        optimization_loss = train_epoch(model, train_loader, optimizer, device)
        train_metrics = evaluate(model, train_eval_loader, device, float(target_mu[0]), float(target_sigma[0]))
        val_metrics = evaluate(model, val_loader, device, float(target_mu[0]), float(target_sigma[0]))
        row = {
            "epoch": epoch,
            "train_optimization_loss": optimization_loss,
            "train_swe_loss": train_metrics["swe_loss"], "val_swe_loss": val_metrics["swe_loss"],
            "train_swe_r2": train_metrics["swe_r2"], "val_swe_r2": val_metrics["swe_r2"],
            "train_swe_rmse": train_metrics["swe_rmse"], "val_swe_rmse": val_metrics["swe_rmse"],
            "train_swe_pearson_r": train_metrics["swe_pearson_r"], "val_swe_pearson_r": val_metrics["swe_pearson_r"],
            "train_swe_spearman_rho": train_metrics["swe_spearman_rho"], "val_swe_spearman_rho": val_metrics["swe_spearman_rho"],
            "train_swe_wet_dry_accuracy": train_metrics["swe_wet_dry_accuracy"], "val_swe_wet_dry_accuracy": val_metrics["swe_wet_dry_accuracy"],
        }
        history_rows.append(row)
        print(f"epoch={epoch} train_r2={train_metrics['swe_r2']:.6f} val_r2={val_metrics['swe_r2']:.6f} "
              f"val_pearson_r={val_metrics['swe_pearson_r']:.6f} val_loss={val_metrics['swe_loss']:.6f}", flush=True)
        if val_metrics["swe_loss"] < best_loss:
            best_loss, best_epoch, patience = val_metrics["swe_loss"], epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "trainable_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith("encoder.")},
                    "val_swe_loss": best_loss,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_checkpoint.pt",
            )
        else:
            patience += 1
        if patience >= args.patience:
            break

    after_encoder_state = {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()}
    encoder_digest_after = tensor_digest(model.encoder)
    max_encoder_change = max(float((initial_encoder_state[name] - after_encoder_state[name]).abs().max()) for name in initial_encoder_state)
    with torch.no_grad():
        fixed_z_after = model.encoder(fixed_inputs).detach().cpu()
    max_fixed_z_change = float((fixed_z_before - fixed_z_after).abs().max())

    import csv as csv_module

    with (output_dir / "history.csv").open("w", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    best_row = min(history_rows, key=lambda r: r["val_swe_loss"])
    metrics_summary = {
        "architecture": "frozen_S0_attention_percentile",
        "s0_checkpoint": str(args.s0_checkpoint),
        "s0_checkpoint_epoch": checkpoint["epoch"],
        "best_epoch": best_epoch,
        "best_val_total_loss": best_loss,
        "best_metrics": {k.replace("val_", ""): v for k, v in best_row.items() if k.startswith("val_")},
        "checkpoint_selection_criterion": "val_swe_loss (minimum) - same convention as existing S0/S1 training pipeline",
        "encoder_all_requires_grad_false": all(not p.requires_grad for p in model.encoder.parameters()),
        "initial_encoder_digest": initial_encoder_digest,
        "final_encoder_digest": encoder_digest_after,
        "encoder_digest_unchanged": initial_encoder_digest == encoder_digest_after,
        "max_encoder_parameter_change": max_encoder_change,
        "max_fixed_batch_latent_change": max_fixed_z_change,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    print(json.dumps(metrics_summary, indent=2), flush=True)
    assert max_encoder_change == 0.0, "encoder parameters changed during downstream training"
    assert max_fixed_z_change == 0.0, "fixed-batch latent changed during downstream training"


if __name__ == "__main__":
    main()
