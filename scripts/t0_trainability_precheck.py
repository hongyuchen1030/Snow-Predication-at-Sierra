#!/usr/bin/env python3
"""Quick trainability pre-check for the EXISTING T0 implementation: can it
overfit a small (~512 sample) subset of the Oct1-Apr1 training population?
Does not change the architecture, does not implement T1, does not sweep
hyperparameters. Reuses delta_swe_dataset.DeltaSWEDataset and
delta_swe_transformer_t0.T0FactorizedSTTransformer unmodified.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from delta_swe_dataset import DeltaSWEDataset, collate_delta_swe  # noqa: E402
from delta_swe_transformer_t0 import T0FactorizedSTTransformer, count_parameters  # noqa: E402

T0_EXP_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_transformer_t0_v1")
INDEX_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1/data_index/sample_index.npz")
SPLIT_PATH = T0_EXP_ROOT / "split.json"
NORM_DIR = T0_EXP_ROOT / "normalization"
OUT_DIR = T0_EXP_ROOT / "tiny_overfit_precheck"
N_SUBSET = 512
MAX_EPOCHS = 100
BATCH_SIZE = 32
SEED = 20260901


def log_gpu_env() -> None:
    import os

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"SLURMD_NODENAME={os.environ.get('SLURMD_NODENAME')}", flush=True)
    print(f"hostname={subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()}", flush=True)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"],
            capture_output=True, text=True, timeout=30,
        )
        print(smi.stdout, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi failed: {exc}", flush=True)


def load_normalization() -> tuple[np.ndarray, np.ndarray, float, float]:
    feat = np.load(NORM_DIR / "feature_normalization.npz", allow_pickle=True)
    target = json.loads((NORM_DIR / "target_normalization.json").read_text())
    return feat["mean"], feat["std"], target["target_mu_mm"], target["target_sigma_mm"]


def build_stratified_subset(full_train_ds: DeltaSWEDataset, n_target: int, seed: int) -> list[int]:
    delta = full_train_ds.delta_swe
    pos_idx = np.where(delta > 0)[0]
    neg_idx = np.where(delta < 0)[0]
    zero_idx = np.where(delta == 0)[0]
    rng = np.random.default_rng(seed)
    # roughly even thirds, capped by availability
    n_each = n_target // 3
    chosen = []
    for pool in (pos_idx, neg_idx, zero_idx):
        take = min(n_each, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    # top up to exactly n_target from the remaining pool (any category) if short
    remaining_needed = n_target - len(chosen)
    if remaining_needed > 0:
        all_idx = np.arange(len(delta))
        leftover = np.setdiff1d(all_idx, np.array(chosen))
        extra = rng.choice(leftover, size=min(remaining_needed, len(leftover)), replace=False)
        chosen.extend(extra.tolist())
    rng.shuffle(chosen)
    return chosen[:n_target]


def grad_and_param_norm(model: torch.nn.Module, names: dict[str, str]) -> dict[str, dict[str, float]]:
    out = {}
    params = dict(model.named_parameters())
    for label, pname in names.items():
        p = params[pname]
        grad_norm = float(p.grad.norm().item()) if p.grad is not None else None
        grad_finite = bool(torch.isfinite(p.grad).all().item()) if p.grad is not None else None
        param_norm = float(p.detach().norm().item())
        out[label] = {"param_name": pname, "grad_norm": grad_norm, "grad_finite": grad_finite, "param_norm": param_norm}
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 and y_pred.std() > 0 else float("nan")
    return {"rmse_mm": rmse, "mae_mm": mae, "pearson_r": r, "pred_std_mm": float(y_pred.std())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    log_gpu_env()

    split = json.loads(SPLIT_PATH.read_text())
    feature_mean, feature_std, target_mu, target_sigma = load_normalization()
    full_train_ds = DeltaSWEDataset(
        index_path=INDEX_PATH, water_years=set(split["train_water_years"]),
        season_start_month=10, sample_mode="all",
        feature_mean=feature_mean, feature_std=feature_std, target_mu=target_mu, target_sigma=target_sigma,
    )
    print(f"full Oct1-Apr1 train population: {len(full_train_ds)} samples", flush=True)

    subset_rows = build_stratified_subset(full_train_ds, N_SUBSET, SEED)
    subset_delta = full_train_ds.delta_swe[subset_rows]
    stats = {
        "N": len(subset_rows),
        "min_mm": float(subset_delta.min()),
        "max_mm": float(subset_delta.max()),
        "mean_mm": float(subset_delta.mean()),
        "std_mm": float(subset_delta.std(ddof=1)),
        "frac_gt0": float((subset_delta > 0).mean()),
        "frac_lt0": float((subset_delta < 0).mean()),
        "frac_eq0": float((subset_delta == 0).mean()),
    }
    print("=== SUBSET TARGET STATISTICS ===", flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    (OUT_DIR / "subset_stats.json").write_text(json.dumps(stats, indent=2))

    subset_ds = Subset(full_train_ds, subset_rows)
    loader = DataLoader(subset_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                         pin_memory=True, collate_fn=collate_delta_swe, drop_last=False)
    eval_loader = DataLoader(subset_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=True, collate_fn=collate_delta_swe)

    # Existing T0, unchanged except dropout=0 for this diagnostic.
    model = T0FactorizedSTTransformer(d_model=128, num_layers=3, num_heads=4, ffn_ratio=4, dropout=0.0, head_dropout=0.0).to(device)
    n_params = count_parameters(model)
    print(f"T0 parameters (dropout=0): {n_params}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    representative_params = {
        "patch_embed": "patch_embed.proj.weight",
        "temporal_attn_block0": "blocks.0.temporal_attn.attn.in_proj_weight",
        "spatial_attn_block0": "blocks.0.spatial_attn.attn.in_proj_weight",
        "ffn_block0": "blocks.0.ffn.fc1.weight",
        "head": "head.3.weight",
    }
    params_dict = dict(model.named_parameters())
    print("representative parameters resolved:", {k: v for k, v in representative_params.items()}, flush=True)

    param_norms_before = {label: float(params_dict[pname].detach().norm().item()) for label, pname in representative_params.items()}

    epoch_losses = []
    metrics_rows = []
    first_step_grad_report = None
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for step, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
                loss = F.huber_loss(pred, y, delta=1.0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if epoch == 1 and step == 0:
                first_step_grad_report = grad_and_param_norm(model, representative_params)
                print("=== FIRST-STEP GRADIENT REPORT ===", flush=True)
                print(json.dumps(first_step_grad_report, indent=2), flush=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu()) * x.shape[0]
            n_seen += x.shape[0]

        avg_loss = total_loss / n_seen
        epoch_losses.append({"epoch": epoch, "train_loss": avg_loss})

        if epoch == 1:
            print(f"epoch 1 train_loss={avg_loss:.6f}", flush=True)

        if epoch % 10 == 0 or epoch == 1 or epoch == MAX_EPOCHS:
            model.eval()
            all_true, all_pred = [], []
            with torch.no_grad():
                for batch in eval_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        pred = model(x)
                    pred_mm = pred.float().cpu().numpy() * target_sigma + target_mu
                    all_true.append(batch["delta_swe_mm"].numpy())
                    all_pred.append(pred_mm)
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            m = compute_metrics(y_true, y_pred)
            m["epoch"] = epoch
            metrics_rows.append(m)
            print(f"epoch={epoch} train_loss={avg_loss:.6f} rmse={m['rmse_mm']:.4f} mae={m['mae_mm']:.4f} "
                  f"pearson_r={m['pearson_r']:.4f} pred_std={m['pred_std_mm']:.4f}", flush=True)

    runtime_s = time.time() - t_start
    param_norms_after = {label: float(params_dict[pname].detach().norm().item()) for label, pname in representative_params.items()}

    update_table = {}
    for label in representative_params:
        before, after = param_norms_before[label], param_norms_after[label]
        update_table[label] = {"param_norm_before": before, "param_norm_after": after, "abs_change": abs(after - before), "changed": abs(after - before) > 1e-8}
    print("=== PARAMETER UPDATE CHECK (before vs after full run) ===", flush=True)
    print(json.dumps(update_table, indent=2), flush=True)

    with (OUT_DIR / "tiny_overfit_loss.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(epoch_losses)

    with (OUT_DIR / "tiny_overfit_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "rmse_mm", "mae_mm", "pearson_r", "pred_std_mm"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot([e["epoch"] for e in epoch_losses], [e["train_loss"] for e in epoch_losses], "-", color="tab:blue")
    ax.axhline(0.282, color="gray", linestyle="--", linewidth=0.8, label="original full-run train loss (~0.282)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title(f"T0 tiny-subset ({N_SUBSET} samples) overfit check")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tiny_overfit_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color="tab:blue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("true ΔSWE (mm)")
    ax.set_ylabel("predicted ΔSWE (mm)")
    ax.set_title(f"T0 tiny-subset final predictions (epoch {MAX_EPOCHS})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tiny_overfit_pred_vs_true.png")
    plt.close(fig)

    first_loss = epoch_losses[0]["train_loss"]
    best_loss = min(e["train_loss"] for e in epoch_losses)
    final_loss = epoch_losses[-1]["train_loss"]
    final_metrics = metrics_rows[-1]

    original_loss_level = 0.282
    substantial_drop = best_loss < 0.8 * original_loss_level
    r_positive_and_meaningful = final_metrics["pearson_r"] > 0.3
    pred_has_variance = final_metrics["pred_std_mm"] > 0.3 * subset_delta.std(ddof=1)

    verdict = "PASS" if (substantial_drop and r_positive_and_meaningful and pred_has_variance) else "FAIL"

    summary = {
        "subset_stats": stats,
        "epoch1_train_loss": first_loss,
        "best_train_loss": best_loss,
        "final_train_loss": final_loss,
        "final_metrics": final_metrics,
        "first_step_grad_report": first_step_grad_report,
        "param_update_check": update_table,
        "runtime_s": runtime_s,
        "criteria": {
            "substantial_loss_drop_below_80pct_of_0.282": substantial_drop,
            "pearson_r_gt_0.3": r_positive_and_meaningful,
            "pred_std_gt_30pct_of_target_std": pred_has_variance,
        },
        "verdict": verdict,
    }
    (OUT_DIR / "tiny_overfit_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FINAL SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
