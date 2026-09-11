#!/usr/bin/env python3
"""Train S0 percentile-target with TWO auxiliary heads at final Z: CPM and AQM, both
gradients flowing through the complete encoder. Extends the winning CPM@Z design from
cpm_aux_transfer_experiment_v1 with a second, structurally identical AQM head. Same
self-contained training-loop pattern as run_s0_cpm_aux_experiment.py (does not modify
snow_ml.cmip6_cnn_experiment.train_experiment/run_epoch).
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


class S0CPMAQMHeadZ(nn.Module):
    """S0 pure-CNN encoder with CPM and AQM auxiliary heads both at final Z."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = SpatialBackbone(IN_CHANNELS_PER_MONTH * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D)
        )
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.cpm_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.aqm_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, m, c, h, w = x.shape
        features = self.backbone(x.reshape(b, m * c, h, w))
        z_flat = self.project(features)
        return {
            "swe_hat": self.swe_head(z_flat).squeeze(-1),
            "cpm_hat": self.cpm_head(z_flat).squeeze(-1),
            "aqm_hat": self.aqm_head(z_flat).squeeze(-1),
            "z": z_flat.reshape(b, LATENT_K, LATENT_D),
        }


def summarize(pred_z, true_z, mu: float, sigma: float) -> dict[str, float]:
    predicted = np.concatenate(pred_z) * sigma + mu
    observed = np.concatenate(true_z) * sigma + mu
    return {
        "r2": r2_score(observed, predicted), "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "pearson_r": pearson_r(observed, predicted), "spearman_rho": spearman_rho(observed, predicted),
        "wet_dry_accuracy": wet_dry_accuracy(observed, predicted),
    }


def evaluate(model, loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, aqm_mu, aqm_sigma, lambda_cpm, lambda_aqm):
    model.eval()
    losses_total, losses_swe, losses_cpm, losses_aqm = [], [], [], []
    swe_p, swe_t, cpm_p, cpm_t, aqm_p, aqm_t = [], [], [], [], [], []
    with torch.no_grad():
        for inputs, targets, _ in loader:
            out = model(inputs.to(device))
            swe_loss = F.mse_loss(out["swe_hat"], targets[:, 0].to(device))
            cpm_loss = F.mse_loss(out["cpm_hat"], targets[:, 1].to(device))
            aqm_loss = F.mse_loss(out["aqm_hat"], targets[:, 2].to(device))
            total = swe_loss + lambda_cpm * cpm_loss + lambda_aqm * aqm_loss
            losses_total.append(float(total.cpu())); losses_swe.append(float(swe_loss.cpu()))
            losses_cpm.append(float(cpm_loss.cpu())); losses_aqm.append(float(aqm_loss.cpu()))
            swe_p.append(out["swe_hat"].cpu().numpy()); swe_t.append(targets[:, 0].numpy())
            cpm_p.append(out["cpm_hat"].cpu().numpy()); cpm_t.append(targets[:, 1].numpy())
            aqm_p.append(out["aqm_hat"].cpu().numpy()); aqm_t.append(targets[:, 2].numpy())
    result = {"total_loss": float(np.mean(losses_total)), "swe_loss": float(np.mean(losses_swe)), "cpm_loss": float(np.mean(losses_cpm)), "aqm_loss": float(np.mean(losses_aqm))}
    result.update({f"swe_{k}": v for k, v in summarize(swe_p, swe_t, swe_mu, swe_sigma).items()})
    result.update({f"cpm_{k}": v for k, v in summarize(cpm_p, cpm_t, cpm_mu, cpm_sigma).items() if k in ("r2", "pearson_r")})
    result.update({f"aqm_{k}": v for k, v in summarize(aqm_p, aqm_t, aqm_mu, aqm_sigma).items() if k in ("r2", "pearson_r")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-cpm", type=float, default=0.1)
    parser.add_argument("--lambda-aqm", type=float, default=0.1)
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

    assert np.isfinite(targets).all(), "NaN/Inf in targets"
    train_idx, val_idx = split["train"], split["val"]
    assert set(train_idx) & set(val_idx) == set(), "train/val overlap"

    feature_mu, feature_sigma, _ = compute_feature_stats(physical, train_idx)
    target_mu, target_sigma = compute_target_stats(targets, train_idx)
    np.savez_compressed(output_dir / "normalization_stats.npz", feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    swe_mu, swe_sigma = float(target_mu[0]), float(target_sigma[0])
    cpm_mu, cpm_sigma = float(target_mu[1]), float(target_sigma[1])
    aqm_mu, aqm_sigma = float(target_mu[2]), float(target_sigma[2])
    print(f"swe_mu={swe_mu:.4f} swe_sigma={swe_sigma:.4f} cpm_mu={cpm_mu:.4f} cpm_sigma={cpm_sigma:.4f} aqm_mu={aqm_mu:.4f} aqm_sigma={aqm_sigma:.4f}", flush=True)

    train_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=train_idx, feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    val_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=val_idx, feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = S0CPMAQMHeadZ()
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
            out = model(inputs.to(device))
            swe_loss = F.mse_loss(out["swe_hat"], tgt[:, 0].to(device))
            cpm_loss = F.mse_loss(out["cpm_hat"], tgt[:, 1].to(device))
            aqm_loss = F.mse_loss(out["aqm_hat"], tgt[:, 2].to(device))
            total_loss = swe_loss + args.lambda_cpm * cpm_loss + args.lambda_aqm * aqm_loss
            total_loss.backward()
            optimizer.step()
            opt_losses.append(float(total_loss.detach().cpu()))

        train_metrics = evaluate(model, train_eval_loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, aqm_mu, aqm_sigma, args.lambda_cpm, args.lambda_aqm)
        val_metrics = evaluate(model, val_loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, aqm_mu, aqm_sigma, args.lambda_cpm, args.lambda_aqm)
        row = {"epoch": epoch, "train_optimization_loss": float(np.mean(opt_losses))}
        for prefix, m in (("train", train_metrics), ("val", val_metrics)):
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        history_rows.append(row)
        print(f"epoch={epoch} val_swe_r2={val_metrics['swe_r2']:.4f} val_swe_pearson_r={val_metrics['swe_pearson_r']:.4f} "
              f"val_cpm_pearson_r={val_metrics['cpm_pearson_r']:.4f} val_aqm_pearson_r={val_metrics['aqm_pearson_r']:.4f} val_total_loss={val_metrics['total_loss']:.4f}", flush=True)

        if val_metrics["total_loss"] < best_loss:
            best_loss, best_epoch, patience = val_metrics["total_loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_total_loss": best_loss, "lambda_cpm": args.lambda_cpm, "lambda_aqm": args.lambda_aqm}, output_dir / "best_checkpoint.pt")
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
        "architecture": "S0_CPM_AQM_at_Z", "lambda_cpm": args.lambda_cpm, "lambda_aqm": args.lambda_aqm,
        "best_epoch": best_epoch, "best_val_total_loss": best_loss,
        "best_metrics": {k: v for k, v in best_row.items() if k.startswith("val_")},
        "checkpoint_selection_criterion": "val_total_loss (minimum) - same convention as existing percentile/CPM@Z training",
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    print(json.dumps(metrics_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
