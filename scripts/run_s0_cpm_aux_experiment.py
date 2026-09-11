#!/usr/bin/env python3
"""Train an S0-family model with an optional CPM auxiliary head, on the existing
percentile SWE target. Self-contained training loop (does not modify
snow_ml.cmip6_cnn_experiment.train_experiment/run_epoch) so existing architectures and
experiments are unaffected. Also usable for the Phase 2 end-to-end-attention+CPM model via
--architecture attention.

CPM@Z: head reads the same flattened Z the SWE head consumes -> gradients reach the whole
encoder (stem, stage1, stage2, stage3, project).

CPM@Stage2: head reads SpatialBackbone.forward_feature_dict()["stage2"] (pre-downsample2,
pre-stage3) directly, global-average-pooled. This branch never touches down2/stage3/project,
so autograd naturally never sends CPM gradient through those modules - no detach() needed,
this is standard multi-consumer autograd (stage2 feeds both the CPM branch and, via
down2->stage3->project, the SWE branch).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import (  # noqa: E402
    LATENT_D,
    LATENT_K,
    MONTH_LABELS,
    CMIP6TensorDataset,
    LatentSelfAttentionBlock,
    SpatialBackbone,
    collate_with_metadata,
    compute_feature_stats,
    compute_target_stats,
    pearson_r,
    r2_score,
    read_manifest,
    set_global_seed,
    spearman_rho,
    wet_dry_accuracy,
)

MODEL_SEED = 20260813
LOADER_SEED = 20260826
IN_CHANNELS_PER_MONTH = 38


class S0CPMHead(nn.Module):
    """S0 pure-CNN family with an optional CPM auxiliary head at Z or at Stage2 output."""

    def __init__(self, cpm_placement: str) -> None:
        super().__init__()
        assert cpm_placement in ("none", "z", "stage2")
        self.cpm_placement = cpm_placement
        self.backbone = SpatialBackbone(IN_CHANNELS_PER_MONTH * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D)
        )
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        if cpm_placement == "z":
            self.cpm_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        elif cpm_placement == "stage2":
            stage2_channels = 64
            self.cpm_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.cpm_head = nn.Sequential(nn.Linear(stage2_channels, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, m, c, h, w = x.shape
        x2d = x.reshape(b, m * c, h, w)
        features = self.backbone.forward_feature_dict(x2d)
        z_flat = self.project(features["stage3"])
        outputs = {"swe_hat": self.swe_head(z_flat).squeeze(-1), "z": z_flat.reshape(b, LATENT_K, LATENT_D)}
        if self.cpm_placement == "z":
            outputs["cpm_hat"] = self.cpm_head(z_flat).squeeze(-1)
        elif self.cpm_placement == "stage2":
            pooled_stage2 = self.cpm_pool(features["stage2"]).flatten(1)
            outputs["cpm_hat"] = self.cpm_head(pooled_stage2).squeeze(-1)
        return outputs


class AttentionCPMHead(nn.Module):
    """End-to-end attention (S1) with the winning CPM placement applied identically."""

    def __init__(self, cpm_placement: str) -> None:
        super().__init__()
        assert cpm_placement in ("z", "stage2")
        self.cpm_placement = cpm_placement
        self.backbone = SpatialBackbone(IN_CHANNELS_PER_MONTH * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D)
        )
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        if cpm_placement == "z":
            self.cpm_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            self.cpm_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.cpm_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, m, c, h, w = x.shape
        x2d = x.reshape(b, m * c, h, w)
        features = self.backbone.forward_feature_dict(x2d)
        latent = self.project(features["stage3"]).reshape(b, LATENT_K, LATENT_D)
        z_attn = self.latent_block(latent)
        z_flat = z_attn.reshape(b, -1)
        outputs = {"swe_hat": self.swe_head(z_flat).squeeze(-1), "z": z_attn}
        if self.cpm_placement == "z":
            outputs["cpm_hat"] = self.cpm_head(z_flat).squeeze(-1)
        else:
            pooled_stage2 = self.cpm_pool(features["stage2"]).flatten(1)
            outputs["cpm_hat"] = self.cpm_head(pooled_stage2).squeeze(-1)
        return outputs


def summarize(pred_z: list[np.ndarray], true_z: list[np.ndarray], mu: float, sigma: float) -> dict[str, float]:
    predicted = np.concatenate(pred_z) * sigma + mu
    observed = np.concatenate(true_z) * sigma + mu
    return {
        "r2": r2_score(observed, predicted),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "pearson_r": pearson_r(observed, predicted),
        "spearman_rho": spearman_rho(observed, predicted),
        "wet_dry_accuracy": wet_dry_accuracy(observed, predicted),
    }


def evaluate(model, loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, has_cpm: bool):
    model.eval()
    losses_total, losses_swe, losses_cpm = [], [], []
    swe_pred, swe_true, cpm_pred, cpm_true = [], [], [], []
    with torch.no_grad():
        for inputs, targets, _ in loader:
            outputs = model(inputs.to(device))
            swe_loss = F.mse_loss(outputs["swe_hat"], targets[:, 0].to(device))
            if has_cpm:
                cpm_loss = F.mse_loss(outputs["cpm_hat"], targets[:, 1].to(device))
                total_loss = swe_loss + 0.1 * cpm_loss
                losses_cpm.append(float(cpm_loss.cpu()))
                cpm_pred.append(outputs["cpm_hat"].cpu().numpy())
                cpm_true.append(targets[:, 1].numpy())
            else:
                total_loss = swe_loss
            losses_total.append(float(total_loss.cpu()))
            losses_swe.append(float(swe_loss.cpu()))
            swe_pred.append(outputs["swe_hat"].cpu().numpy())
            swe_true.append(targets[:, 0].numpy())
    result = {
        "total_loss": float(np.mean(losses_total)),
        "swe_loss": float(np.mean(losses_swe)),
        **{f"swe_{k}": v for k, v in summarize(swe_pred, swe_true, swe_mu, swe_sigma).items()},
    }
    if has_cpm:
        result["cpm_loss"] = float(np.mean(losses_cpm))
        result.update({f"cpm_{k}": v for k, v in summarize(cpm_pred, cpm_true, cpm_mu, cpm_sigma).items() if k in ("r2", "rmse", "pearson_r")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["s0", "attention"], required=True)
    parser.add_argument("--cpm-placement", choices=["none", "z", "stage2"], required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-cpm", type=float, default=0.1)
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
    swe_mu, swe_sigma = float(target_mu[0]), float(target_sigma[0])
    cpm_mu, cpm_sigma = float(target_mu[1]), float(target_sigma[1])
    print(f"swe_mu={swe_mu:.4f} swe_sigma={swe_sigma:.4f} cpm_mu={cpm_mu:.4f} cpm_sigma={cpm_sigma:.4f}", flush=True)

    train_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["train"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    val_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=split["val"], feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_cpm = args.cpm_placement != "none"
    if args.architecture == "s0":
        model = S0CPMHead(args.cpm_placement)
    else:
        assert has_cpm, "attention architecture in this script is only used for the CPM-augmented Phase 2 model"
        model = AttentionCPMHead(args.cpm_placement)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(LOADER_SEED)
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(RandomSampler(train_dataset, generator=generator), args.batch_size, drop_last=False), num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)

    history_rows: list[dict] = []
    best_loss, best_epoch, patience = float("inf"), 0, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        opt_losses = []
        for inputs, tgt, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs.to(device))
            swe_loss = F.mse_loss(outputs["swe_hat"], tgt[:, 0].to(device))
            if has_cpm:
                cpm_loss = F.mse_loss(outputs["cpm_hat"], tgt[:, 1].to(device))
                total_loss = swe_loss + args.lambda_cpm * cpm_loss
            else:
                total_loss = swe_loss
            total_loss.backward()
            optimizer.step()
            opt_losses.append(float(total_loss.detach().cpu()))

        train_metrics = evaluate(model, train_eval_loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, has_cpm)
        val_metrics = evaluate(model, val_loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, has_cpm)
        row = {
            "epoch": epoch, "train_optimization_loss": float(np.mean(opt_losses)),
            "train_total_loss": train_metrics["total_loss"], "val_total_loss": val_metrics["total_loss"],
            "train_swe_loss": train_metrics["swe_loss"], "val_swe_loss": val_metrics["swe_loss"],
            "train_swe_pearson_r": train_metrics["swe_pearson_r"], "val_swe_pearson_r": val_metrics["swe_pearson_r"],
            "train_swe_spearman_rho": train_metrics["swe_spearman_rho"], "val_swe_spearman_rho": val_metrics["swe_spearman_rho"],
            "train_swe_wet_dry_accuracy": train_metrics["swe_wet_dry_accuracy"], "val_swe_wet_dry_accuracy": val_metrics["swe_wet_dry_accuracy"],
            "train_swe_r2": train_metrics["swe_r2"], "val_swe_r2": val_metrics["swe_r2"],
        }
        if has_cpm:
            row.update({
                "train_cpm_loss": train_metrics["cpm_loss"], "val_cpm_loss": val_metrics["cpm_loss"],
                "train_cpm_pearson_r": train_metrics["cpm_pearson_r"], "val_cpm_pearson_r": val_metrics["cpm_pearson_r"],
                "train_cpm_r2": train_metrics["cpm_r2"], "val_cpm_r2": val_metrics["cpm_r2"],
            })
        history_rows.append(row)
        print(f"epoch={epoch} val_swe_r2={val_metrics['swe_r2']:.4f} val_swe_pearson_r={val_metrics['swe_pearson_r']:.4f}"
              + (f" val_cpm_pearson_r={val_metrics['cpm_pearson_r']:.4f}" if has_cpm else "")
              + f" val_total_loss={val_metrics['total_loss']:.4f}", flush=True)

        if val_metrics["total_loss"] < best_loss:
            best_loss, best_epoch, patience = val_metrics["total_loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_total_loss": best_loss, "cpm_placement": args.cpm_placement, "lambda_cpm": args.lambda_cpm if has_cpm else None}, output_dir / "best_checkpoint.pt")
        else:
            patience += 1
        if patience >= args.patience:
            break

    with (output_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    best_row = min(history_rows, key=lambda r: r["val_total_loss"])
    metrics_summary = {
        "architecture": args.architecture, "cpm_placement": args.cpm_placement, "lambda_cpm": args.lambda_cpm if has_cpm else None,
        "best_epoch": best_epoch, "best_val_total_loss": best_loss,
        "best_metrics": {k: v for k, v in best_row.items() if k.startswith("val_")},
        "checkpoint_selection_criterion": "val_total_loss (minimum) - same convention as existing percentile S0/S1 training",
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    print(json.dumps(metrics_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
