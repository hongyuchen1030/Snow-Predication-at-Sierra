#!/usr/bin/env python3
"""Train T0 (ForeSWE-referenced factorized spatiotemporal Transformer) on the
Oct1->Apr1, 7-day -> 1-day Sierra SWE-change task. Train/validation only, no
simulation test split. Reuses DeltaSWEDataset/collate_delta_swe from the
CNN/ConvLSTM baseline pipeline unmodified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from delta_swe_dataset import DeltaSWEDataset, collate_delta_swe  # noqa: E402
from delta_swe_transformer_t0 import T0FactorizedSTTransformer, count_parameters, N_TOKENS, PATCH  # noqa: E402

EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_transformer_t0_v1")
INDEX_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1/data_index/sample_index.npz")
SPLIT_PATH = EXPERIMENT_ROOT / "split.json"
NORM_DIR = EXPERIMENT_ROOT / "normalization"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--benchmark-batches", type=int, default=50)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


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


def build_datasets() -> tuple[DeltaSWEDataset, DeltaSWEDataset]:
    split = json.loads(SPLIT_PATH.read_text())
    feature_mean, feature_std, target_mu, target_sigma = load_normalization()
    common = dict(
        index_path=INDEX_PATH,
        season_start_month=10,  # Oct1 -> Apr1
        sample_mode="all",
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )
    train_ds = DeltaSWEDataset(water_years=set(split["train_water_years"]), **common)
    val_ds = DeltaSWEDataset(water_years=set(split["val_water_years"]), **common)
    return train_ds, val_ds


def compute_metrics(y_true_mm: np.ndarray, y_pred_mm: np.ndarray) -> dict:
    if len(y_true_mm) == 0:
        return {"n": 0}
    err = y_pred_mm - y_true_mm
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_mm - y_true_mm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(y_true_mm, y_pred_mm)[0, 1]) if y_true_mm.std() > 0 and y_pred_mm.std() > 0 else float("nan")
    sign_acc = float(np.mean(np.sign(y_true_mm) == np.sign(y_pred_mm)))
    return {"n": int(len(y_true_mm)), "rmse_mm": rmse, "mae_mm": mae, "r2": r2, "pearson_r": r, "sign_accuracy": sign_acc}


def full_metrics_report(y_true_mm: np.ndarray, y_pred_mm: np.ndarray, active: np.ndarray) -> dict:
    return {
        "overall": compute_metrics(y_true_mm, y_pred_mm),
        "accumulation_gt0": compute_metrics(y_true_mm[y_true_mm > 0], y_pred_mm[y_true_mm > 0]),
        "melt_lt0": compute_metrics(y_true_mm[y_true_mm < 0], y_pred_mm[y_true_mm < 0]),
        "zero_eq0": compute_metrics(y_true_mm[y_true_mm == 0], y_pred_mm[y_true_mm == 0]),
        "snow_active_only": compute_metrics(y_true_mm[active], y_pred_mm[active]),
    }


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, target_mu=0.0, target_sigma=1.0, use_amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    all_true, all_pred, all_active = [], [], []
    t0 = time.time()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
                loss = F.huber_loss(pred, y, delta=1.0)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.shape[0]
        n_batches += x.shape[0]
        pred_mm = pred.detach().float().cpu().numpy() * target_sigma + target_mu
        all_true.append(batch["delta_swe_mm"].numpy())
        all_pred.append(pred_mm)
        all_active.append(batch["active"].numpy())
    elapsed = time.time() - t0
    return (total_loss / max(n_batches, 1), elapsed, np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_active))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    log_gpu_env()

    out_dir = EXPERIMENT_ROOT
    (out_dir / "plots").mkdir(exist_ok=True, parents=True)

    train_ds, val_ds = build_datasets()
    print(f"train={len(train_ds)} val={len(val_ds)} samples", flush=True)
    # Sanity checks (Section 16)
    train_months = sorted(set(train_ds.cal_month.tolist()) | set(val_ds.cal_month.tolist()))
    assert 9 not in train_months, f"September samples present: {train_months}"
    apr_violations = [
        1 for m, d in zip(np.concatenate([train_ds.cal_month, val_ds.cal_month]), np.concatenate([train_ds.cal_day, val_ds.cal_day]))
        if (m == 4 and d > 1) or (5 <= m <= 8)
    ]
    assert len(apr_violations) == 0, f"{len(apr_violations)} Apr2-Aug31 samples present"
    train_wy_set = set(train_ds.water_year.tolist())
    val_wy_set = set(val_ds.water_year.tolist())
    assert len(train_wy_set) == 96, len(train_wy_set)
    assert len(val_wy_set) == 24, len(val_wy_set)
    assert train_wy_set.isdisjoint(val_wy_set), "WY leakage between train/val"
    print(f"sanity OK: no Sep samples, no Apr2-Aug31 samples, {len(train_wy_set)} train WY, {len(val_wy_set)} val WY, no leakage", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_delta_swe, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=collate_delta_swe, persistent_workers=args.num_workers > 0)

    model = T0FactorizedSTTransformer(d_model=128, num_layers=3, num_heads=4, ffn_ratio=4, dropout=0.10, head_dropout=0.05).to(device)
    n_params = count_parameters(model)
    print(f"T0 parameters={n_params}", flush=True)
    if not (500_000 <= n_params <= 2_000_000):
        print(f"WARNING: parameter count {n_params} is far from the expected ~1.1M -- stopping for inspection.", flush=True)
        sys.exit(1)

    _, _, target_mu, target_sigma = load_normalization()

    config = vars(args) | {
        "n_params": n_params, "train_n": len(train_ds), "val_n": len(val_ds),
        "patch": PATCH, "n_tokens": N_TOKENS, "d_model": 128, "num_layers": 3, "num_heads": 4, "ffn_dim": 512,
        "season_start_month": 10, "sample_mode": "all",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "architecture.txt").write_text(repr(model))

    use_amp = (not args.no_amp) and device.type == "cuda"

    if args.benchmark_only:
        print("=== BENCHMARK MODE ===", flush=True)
        t0 = time.time()
        n = 0
        for i, batch in enumerate(train_loader):
            if i == 0:
                t_first = time.time() - t0
            n += 1
            if n >= args.benchmark_batches:
                break
        t_loader = time.time() - t0
        print(f"loader: first_batch={t_first:.3f}s, {n} batches in {t_loader:.3f}s ({t_loader/n:.4f}s/batch)", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
                loss = F.huber_loss(pred, y, delta=1.0)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            n += 1
            if n >= args.benchmark_batches:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        else:
            peak_alloc_gb = peak_reserved_gb = float("nan")
        t_train = time.time() - t0
        print(f"train: {n} batches in {t_train:.3f}s ({t_train/n:.4f}s/batch), peak_alloc={peak_alloc_gb:.2f}GB peak_reserved={peak_reserved_gb:.2f}GB", flush=True)

        t0 = time.time()
        _, train_epoch_time, _, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        _, val_epoch_time, _, _, _ = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        n_batches_per_epoch = len(train_loader)
        est_per_epoch = train_epoch_time + val_epoch_time
        bench_report = {
            "loader_first_batch_s": t_first, "loader_avg_s_per_batch": t_loader / n,
            "train_avg_s_per_batch": t_train / n, "peak_alloc_gb": peak_alloc_gb, "peak_reserved_gb": peak_reserved_gb,
            "one_train_epoch_s": train_epoch_time, "one_val_epoch_s": val_epoch_time,
            "n_train_batches_per_epoch": n_batches_per_epoch, "est_s_per_full_epoch": est_per_epoch,
            "est_time_40_epochs_hours": est_per_epoch * 40 / 3600,
            "est_time_10_25_epochs_hours": [est_per_epoch * 10 / 3600, est_per_epoch * 25 / 3600],
        }
        (out_dir / "benchmark.json").write_text(json.dumps(bench_report, indent=2))
        print(json.dumps(bench_report, indent=2), flush=True)
        print("BENCHMARK_DONE", flush=True)
        return

    # -----------------------------------------------------------------
    # Full training.
    # -----------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()
    log_fh = (out_dir / "train.log").open("w")

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_time, _, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        val_loss, val_time, y_true_val, y_pred_val, _ = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        val_rmse = float(np.sqrt(np.mean((y_pred_val - y_true_val) ** 2)))
        gap = val_loss - train_loss
        scheduler.step(val_loss)

        line = (f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} gap={gap:.6f} "
                f"val_rmse_mm={val_rmse:.4f} train_time={train_time:.1f}s val_time={val_time:.1f}s lr={optimizer.param_groups[0]['lr']:.2e}")
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "generalization_gap": gap, "val_rmse_mm": val_rmse})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "config": config}, out_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch} (best={best_epoch})", flush=True)
                log_fh.write(f"early stopping at epoch {epoch} (best={best_epoch})\n")
                break

    total_train_time = time.time() - train_start
    log_fh.close()
    if device.type == "cuda":
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    else:
        peak_alloc_gb = peak_reserved_gb = float("nan")

    (out_dir / "loss_curve.json").write_text(json.dumps(history, indent=2))
    import csv
    with (out_dir / "loss_curve.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "generalization_gap", "val_rmse_mm"])
        writer.writeheader()
        writer.writerows(history)

    # -----------------------------------------------------------------
    # Evaluate best checkpoint on validation.
    # -----------------------------------------------------------------
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, y_true_val, y_pred_val, active_val = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
    val_report = full_metrics_report(y_true_val, y_pred_val, active_val)
    zero_val = compute_metrics(y_true_val, np.zeros_like(y_true_val))

    final_gap = history[-1]["generalization_gap"]
    train_loss_at_best = next(h["train_loss"] for h in history if h["epoch"] == best_epoch)
    val_loss_at_best = next(h["val_loss"] for h in history if h["epoch"] == best_epoch)

    metrics = {
        "model": "t0_factorized_st_transformer",
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "total_train_time_s": total_train_time,
        "batch_size": args.batch_size,
        "peak_alloc_gb": peak_alloc_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "train_loss_at_best_epoch": train_loss_at_best,
        "val_loss_at_best_epoch": val_loss_at_best,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_generalization_gap": final_gap,
        "val": val_report,
        "zero_change_baseline_val": zero_val,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    np.savez_compressed(
        out_dir / "predictions_val.npz",
        val_true_mm=y_true_val, val_pred_mm=y_pred_val, val_active=active_val,
        val_water_year=val_ds.water_year, val_cal_year=val_ds.cal_year,
        val_cal_month=val_ds.cal_month, val_cal_day=val_ds.cal_day,
        val_swe_t=val_ds.swe_t, val_swe_t1=val_ds.swe_t1,
    )
    print(f"wrote {out_dir / 'predictions_val.npz'}", flush=True)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
