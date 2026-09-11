#!/usr/bin/env python3
"""Train an attention SWE head on top of the CPM+AQM@Z encoder, with 3 freeze levels:

frozen  - entire encoder (stem, stage1, down1, stage2, down2, stage3, project) frozen;
          only attention + new swe_head trainable; CPM/AQM heads unused.
partial - stem, stage1, down1, stage2 frozen; down2, stage3, project, attention, swe_head,
          cpm_head, aqm_head all trainable; CPM/AQM loss active.
full    - everything trainable; CPM/AQM loss active.

Encoder/CPM/AQM-head weights are initialized from the existing best
S0_CPM_AQM_at_Z checkpoint (never retrained from scratch). The attention block and its
swe_head are always freshly initialized (they did not exist in that checkpoint).
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
    CMIP6TensorDataset,
    LatentSelfAttentionBlock,
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
from run_s0_cpm_aqm_aux_experiment import S0CPMAQMHeadZ  # noqa: E402

MODEL_SEED = 20260813
LOADER_SEED = 20260826
CPM_AQM_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S0_CPM_AQM_at_Z/best_checkpoint.pt")


class AttentionOnCPMAQMEncoder(nn.Module):
    def __init__(self, freeze_mode: str) -> None:
        super().__init__()
        assert freeze_mode in ("frozen", "partial", "full")
        self.freeze_mode = freeze_mode
        base = S0CPMAQMHeadZ()
        checkpoint = torch.load(CPM_AQM_CHECKPOINT, map_location="cpu", weights_only=False)
        base.load_state_dict(checkpoint["model_state_dict"], strict=True)

        self.backbone = base.backbone
        self.project = base.project
        self.cpm_head = base.cpm_head
        self.aqm_head = base.aqm_head
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

        if freeze_mode == "frozen":
            for module in (self.backbone, self.project, self.cpm_head, self.aqm_head):
                for p in module.parameters():
                    p.requires_grad_(False)
        elif freeze_mode == "partial":
            for module in (self.backbone.stem, self.backbone.stage1, self.backbone.down1, self.backbone.stage2):
                for p in module.parameters():
                    p.requires_grad_(False)
            for module in (self.backbone.down2, self.backbone.stage3, self.project, self.cpm_head, self.aqm_head):
                for p in module.parameters():
                    p.requires_grad_(True)
        else:  # full
            for p in self.parameters():
                p.requires_grad_(True)

    def encoder_eval_if_frozen(self) -> None:
        if self.freeze_mode == "frozen":
            self.backbone.eval()
            self.project.eval()

    def forward(self, x: torch.Tensor, *, use_cpm_aqm: bool) -> dict[str, torch.Tensor]:
        b, m, c, h, w = x.shape
        if self.freeze_mode == "frozen":
            with torch.no_grad():
                features = self.backbone(x.reshape(b, m * c, h, w))
                z_raw = self.project(features).reshape(b, LATENT_K, LATENT_D)
        else:
            features = self.backbone(x.reshape(b, m * c, h, w))
            z_raw = self.project(features).reshape(b, LATENT_K, LATENT_D)
        z_attn = self.latent_block(z_raw)
        z_flat = z_attn.reshape(b, -1)
        out = {"swe_hat": self.swe_head(z_flat).squeeze(-1), "z": z_attn}
        if use_cpm_aqm:
            z_raw_flat = z_raw.reshape(b, -1)
            out["cpm_hat"] = self.cpm_head(z_raw_flat).squeeze(-1)
            out["aqm_hat"] = self.aqm_head(z_raw_flat).squeeze(-1)
        return out


def summarize(pred_z, true_z, mu: float, sigma: float) -> dict[str, float]:
    predicted = np.concatenate(pred_z) * sigma + mu
    observed = np.concatenate(true_z) * sigma + mu
    return {
        "r2": r2_score(observed, predicted), "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "pearson_r": pearson_r(observed, predicted), "spearman_rho": spearman_rho(observed, predicted),
        "wet_dry_accuracy": wet_dry_accuracy(observed, predicted),
    }


def evaluate(model, loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, aqm_mu, aqm_sigma, lambda_cpm, lambda_aqm, use_cpm_aqm):
    model.eval()
    losses_total, losses_swe, losses_cpm, losses_aqm = [], [], [], []
    swe_p, swe_t, cpm_p, cpm_t, aqm_p, aqm_t = [], [], [], [], [], []
    with torch.no_grad():
        for inputs, targets, _ in loader:
            out = model(inputs.to(device), use_cpm_aqm=use_cpm_aqm)
            swe_loss = F.mse_loss(out["swe_hat"], targets[:, 0].to(device))
            total = swe_loss
            if use_cpm_aqm:
                cpm_loss = F.mse_loss(out["cpm_hat"], targets[:, 1].to(device))
                aqm_loss = F.mse_loss(out["aqm_hat"], targets[:, 2].to(device))
                total = swe_loss + lambda_cpm * cpm_loss + lambda_aqm * aqm_loss
                losses_cpm.append(float(cpm_loss.cpu())); losses_aqm.append(float(aqm_loss.cpu()))
                cpm_p.append(out["cpm_hat"].cpu().numpy()); cpm_t.append(targets[:, 1].numpy())
                aqm_p.append(out["aqm_hat"].cpu().numpy()); aqm_t.append(targets[:, 2].numpy())
            losses_total.append(float(total.cpu())); losses_swe.append(float(swe_loss.cpu()))
            swe_p.append(out["swe_hat"].cpu().numpy()); swe_t.append(targets[:, 0].numpy())
    result = {"total_loss": float(np.mean(losses_total)), "swe_loss": float(np.mean(losses_swe))}
    result.update({f"swe_{k}": v for k, v in summarize(swe_p, swe_t, swe_mu, swe_sigma).items()})
    if use_cpm_aqm:
        result["cpm_loss"] = float(np.mean(losses_cpm)); result["aqm_loss"] = float(np.mean(losses_aqm))
        result.update({f"cpm_{k}": v for k, v in summarize(cpm_p, cpm_t, cpm_mu, cpm_sigma).items() if k in ("r2", "pearson_r")})
        result.update({f"aqm_{k}": v for k, v in summarize(aqm_p, aqm_t, aqm_mu, aqm_sigma).items() if k in ("r2", "pearson_r")})
    return result


def snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-mode", choices=["frozen", "partial", "full"], required=True)
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
    use_cpm_aqm = args.freeze_mode != "frozen"

    cache_dir = Path(args.cache_dir)
    manifest = read_manifest(cache_dir / "manifest.csv")
    split = json.loads(Path(args.split_json).read_text())
    physical = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    targets = np.load(cache_dir / "targets.npy")
    assert np.isfinite(targets).all()
    train_idx, val_idx = split["train"], split["val"]
    assert set(train_idx) & set(val_idx) == set()

    feature_mu, feature_sigma, _ = compute_feature_stats(physical, train_idx)
    target_mu, target_sigma = compute_target_stats(targets, train_idx)
    np.savez_compressed(output_dir / "normalization_stats.npz", feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    swe_mu, swe_sigma = float(target_mu[0]), float(target_sigma[0])
    cpm_mu, cpm_sigma = float(target_mu[1]), float(target_sigma[1])
    aqm_mu, aqm_sigma = float(target_mu[2]), float(target_sigma[2])

    train_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=train_idx, feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    val_dataset = CMIP6TensorDataset(physical_data=physical, valid_mask=mask, targets=targets, metadata=manifest, indices=val_idx, feature_mu=feature_mu, feature_sigma=feature_sigma, target_mu=target_mu, target_sigma=target_sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionOnCPMAQMEncoder(args.freeze_mode)
    model.to(device)

    # Sanity: snapshot frozen-part parameters before training.
    frozen_modules = {"frozen": [model.backbone, model.project], "partial": [model.backbone.stem, model.backbone.stage1, model.backbone.down1, model.backbone.stage2]}
    before_frozen = {}
    if args.freeze_mode in frozen_modules:
        for i, mod in enumerate(frozen_modules[args.freeze_mode]):
            before_frozen[i] = snapshot(mod)
    fixed_check_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_metadata)
    fixed_inputs = next(iter(fixed_check_loader))[0].to(device)
    with torch.no_grad():
        model.encoder_eval_if_frozen()
        fixed_z_before = model.project(model.backbone(fixed_inputs.reshape(fixed_inputs.shape[0], -1, fixed_inputs.shape[3], fixed_inputs.shape[4]))).detach().cpu().clone()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(LOADER_SEED)
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(RandomSampler(train_dataset, generator=generator), args.batch_size, drop_last=False), num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_with_metadata)

    history_rows: list[dict] = []
    best_loss, best_epoch, patience = float("inf"), 0, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        model.encoder_eval_if_frozen()
        opt_losses = []
        for inputs, tgt, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(inputs.to(device), use_cpm_aqm=use_cpm_aqm)
            swe_loss = F.mse_loss(out["swe_hat"], tgt[:, 0].to(device))
            total_loss = swe_loss
            if use_cpm_aqm:
                cpm_loss = F.mse_loss(out["cpm_hat"], tgt[:, 1].to(device))
                aqm_loss = F.mse_loss(out["aqm_hat"], tgt[:, 2].to(device))
                total_loss = swe_loss + args.lambda_cpm * cpm_loss + args.lambda_aqm * aqm_loss
            total_loss.backward()
            optimizer.step()
            opt_losses.append(float(total_loss.detach().cpu()))

        train_metrics = evaluate(model, train_eval_loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, aqm_mu, aqm_sigma, args.lambda_cpm, args.lambda_aqm, use_cpm_aqm)
        val_metrics = evaluate(model, val_loader, device, swe_mu, swe_sigma, cpm_mu, cpm_sigma, aqm_mu, aqm_sigma, args.lambda_cpm, args.lambda_aqm, use_cpm_aqm)
        row = {"epoch": epoch, "train_optimization_loss": float(np.mean(opt_losses))}
        for prefix, m in (("train", train_metrics), ("val", val_metrics)):
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        history_rows.append(row)
        msg = f"epoch={epoch} freeze={args.freeze_mode} val_swe_r2={val_metrics['swe_r2']:.4f} val_swe_pearson_r={val_metrics['swe_pearson_r']:.4f}"
        if use_cpm_aqm:
            msg += f" val_cpm_pearson_r={val_metrics['cpm_pearson_r']:.4f} val_aqm_pearson_r={val_metrics['aqm_pearson_r']:.4f}"
        msg += f" val_total_loss={val_metrics['total_loss']:.4f}"
        print(msg, flush=True)

        if val_metrics["total_loss"] < best_loss:
            best_loss, best_epoch, patience = val_metrics["total_loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_total_loss": best_loss, "freeze_mode": args.freeze_mode}, output_dir / "best_checkpoint.pt")
        else:
            patience += 1
        if patience >= args.patience:
            break

    with (output_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    # -----------------------------------------------------------------
    # Post-training sanity checks.
    # -----------------------------------------------------------------
    checks = {}
    if args.freeze_mode in frozen_modules:
        after_frozen = {}
        max_diffs = []
        for i, mod in enumerate(frozen_modules[args.freeze_mode]):
            after_frozen[i] = snapshot(mod)
            max_diffs.append(max(float((before_frozen[i][k] - after_frozen[i][k]).abs().max()) for k in before_frozen[i]))
        checks["max_frozen_parameter_change"] = max(max_diffs)
        with torch.no_grad():
            model.encoder_eval_if_frozen()
            fixed_z_after = model.project(model.backbone(fixed_inputs.reshape(fixed_inputs.shape[0], -1, fixed_inputs.shape[3], fixed_inputs.shape[4]))).detach().cpu()
        checks["max_fixed_z_change"] = float((fixed_z_before - fixed_z_after).abs().max())
        print(f"SANITY: freeze_mode={args.freeze_mode} max_frozen_param_change={checks['max_frozen_parameter_change']} max_fixed_z_change={checks['max_fixed_z_change']}", flush=True)
        assert checks["max_frozen_parameter_change"] == 0.0, "frozen parameters changed"
        # Z is expected to change in "partial" mode (Stage3+project are trainable there) -
        # only "frozen" mode requires the full encoder path, and hence Z, to be unchanged.
        if args.freeze_mode == "frozen":
            assert checks["max_fixed_z_change"] == 0.0, "frozen-path Z changed"
        else:
            assert checks["max_fixed_z_change"] > 0.0, "partial mode: Z did not change even though Stage3+project are trainable - unfreezing may not have taken effect"
    else:
        checks["all_parameters_trainable"] = all(p.requires_grad for p in model.parameters())
        print(f"SANITY: freeze_mode=full all_parameters_trainable={checks['all_parameters_trainable']}", flush=True)

    best_row = min(history_rows, key=lambda r: r["val_total_loss"])
    metrics_summary = {
        "architecture": f"attention_on_cpm_aqm_z_{args.freeze_mode}", "freeze_mode": args.freeze_mode,
        "cpm_aqm_source_checkpoint": str(CPM_AQM_CHECKPOINT),
        "best_epoch": best_epoch, "best_val_total_loss": best_loss,
        "best_metrics": {k: v for k, v in best_row.items() if k.startswith("val_")},
        "sanity_checks": checks,
        "checkpoint_selection_criterion": "val_total_loss (minimum)",
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    print(json.dumps(metrics_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
