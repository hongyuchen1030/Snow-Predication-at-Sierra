#!/usr/bin/env python3
"""Train B0 (LSTM) or B1 (TCNN) on the 7-day -> 1-day Sierra SWE-change task
(daily_temporal_baselines_v2). Reuses the DeltaSWEDataset pipeline and the
T0 Transformer run's split/normalization artifacts (Oct1-Apr1, 96 train WY /
24 val WY, TaiESM1 12-var predictors -- exact match to this experiment's
protocol, copied unchanged into shared/). Also supports --benchmark-only for
the pre-flight loader/epoch timing check used to pick a batch size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from delta_swe_dataset import DeltaSWEDataset, collate_delta_swe  # noqa: E402
from temporal_baseline_models import build_model, count_parameters  # noqa: E402

EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_temporal_baselines_v2")
BASELINE_INDEX = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1/data_index/sample_index.npz")
SHARED_DIR = EXPERIMENT_ROOT / "shared"
SPLIT_PATH = SHARED_DIR / "split.json"
NORM_DIR = SHARED_DIR / "normalization"

SEASON_START_MONTH = 10  # Oct1 -> Apr1, fixed per protocol
SAMPLE_MODE = "all"


def log_gpu_env() -> None:
    import os
    import subprocess

    print(f"hostname={os.uname().nodename}", flush=True)
    print(f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv"], capture_output=True, text=True, timeout=30)
        print(out.stdout, flush=True)
    except Exception as e:  # pragma: no cover
        print(f"nvidia-smi failed: {e}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "tcnn"], required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=None, help="default: max_epochs (no early stopping, per protocol)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--benchmark-batches", type=int, default=100)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.patience is None:
        args.patience = args.max_epochs
    return args


def load_normalization() -> tuple[np.ndarray, np.ndarray, float, float]:
    feat = np.load(NORM_DIR / "feature_normalization.npz", allow_pickle=True)
    target = json.loads((NORM_DIR / "target_normalization.json").read_text())
    return feat["mean"], feat["std"], target["target_mu_mm"], target["target_sigma_mm"]


def build_datasets() -> tuple[DeltaSWEDataset, DeltaSWEDataset]:
    split = json.loads(SPLIT_PATH.read_text())
    feature_mean, feature_std, target_mu, target_sigma = load_normalization()
    common = dict(
        index_path=BASELINE_INDEX,
        season_start_month=SEASON_START_MONTH,
        sample_mode=SAMPLE_MODE,
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
    if y_true_mm.std() > 0 and y_pred_mm.std() > 0:
        r = float(np.corrcoef(y_true_mm, y_pred_mm)[0, 1])
    else:
        r = float("nan")
    sign_acc = float(np.mean(np.sign(y_true_mm) == np.sign(y_pred_mm)))
    return {
        "n": int(len(y_true_mm)), "rmse_mm": rmse, "mae_mm": mae, "r2": r2,
        "pearson_r": r, "sign_accuracy": sign_acc,
        "pred_mean_mm": float(y_pred_mm.mean()), "pred_std_mm": float(y_pred_mm.std()),
        "target_mean_mm": float(y_true_mm.mean()), "target_std_mm": float(y_true_mm.std()),
    }


def full_metrics_report(y_true_mm: np.ndarray, y_pred_mm: np.ndarray, active: np.ndarray) -> dict:
    return {
        "overall": compute_metrics(y_true_mm, y_pred_mm),
        "accumulation_gt0": compute_metrics(y_true_mm[y_true_mm > 0], y_pred_mm[y_true_mm > 0]),
        "melt_lt0": compute_metrics(y_true_mm[y_true_mm < 0], y_pred_mm[y_true_mm < 0]),
        "zero_eq0": compute_metrics(y_true_mm[y_true_mm == 0], y_pred_mm[y_true_mm == 0]),
        "snow_active_only": compute_metrics(y_true_mm[active], y_pred_mm[active]),
    }


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, target_mu=0.0, target_sigma=1.0, use_amp=True, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    all_true, all_pred, all_active = [], [], []
    t0 = time.time()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
                loss = F.mse_loss(pred, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.shape[0]
        n_samples += x.shape[0]
        pred_mm = pred.detach().float().cpu().numpy() * target_sigma + target_mu
        all_true.append(batch["delta_swe_mm"].numpy())
        all_pred.append(pred_mm)
        all_active.append(batch["active"].numpy())
    elapsed = time.time() - t0
    return (
        total_loss / max(n_samples, 1), elapsed,
        np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_active),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_gpu_env()
    print(f"device={device} model={args.model} season_start_month={SEASON_START_MONTH} sample_mode={SAMPLE_MODE}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets()
    print(f"train={len(train_ds)} val={len(val_ds)} samples", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_delta_swe, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_delta_swe, persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"model={args.model} n_params={n_params}", flush=True)

    _, _, target_mu, target_sigma = load_normalization()

    config = vars(args) | {
        "n_params": n_params, "train_n": len(train_ds), "val_n": len(val_ds),
        "loss": "mse", "optimizer": "adam", "season_start_month": SEASON_START_MONTH, "sample_mode": SAMPLE_MODE,
    }
    if args.model == "tcnn":
        config["tcnn_kernel_size"] = model.kernel_size
        config["tcnn_dilations"] = model.dilations
        config["tcnn_receptive_field"] = model.receptive_field
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

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

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
        train_loss, train_epoch_time, _, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma, grad_clip=args.grad_clip)
        val_loss, val_epoch_time, _, _, _ = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            peak_mem_gb = float("nan")

        n_batches_per_epoch = len(train_loader)
        est_per_epoch = train_epoch_time + val_epoch_time
        bench_report = {
            "loader_first_batch_s": t_first, "loader_avg_s_per_batch": t_loader / n,
            "peak_gpu_mem_gb": peak_mem_gb, "one_train_epoch_s": train_epoch_time, "one_val_epoch_s": val_epoch_time,
            "n_train_batches_per_epoch": n_batches_per_epoch, "est_s_per_full_epoch": est_per_epoch,
            "est_time_50_epochs_hours": est_per_epoch * 50 / 3600,
        }
        (out_dir / "benchmark.json").write_text(json.dumps(bench_report, indent=2))
        print(json.dumps(bench_report, indent=2), flush=True)
        print("BENCHMARK_DONE", flush=True)
        return

    # -----------------------------------------------------------------
    # Full training: Adam, MSE, grad clip=1.0, max_epochs=50, no aggressive
    # early stopping (default patience == max_epochs) so full curves are
    # always produced for diagnosis.
    # -----------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    train_start = time.time()
    log_path = out_dir / "train.log"
    csv_path = out_dir / "train_log.csv"
    log_fh = log_path.open("w")
    csv_fh = csv_path.open("w")
    csv_fh.write("epoch,train_loss,val_loss,val_rmse_mm,train_time_s,val_time_s\n")

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_time, _, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma, grad_clip=args.grad_clip)
        val_loss, val_time, y_true_val, y_pred_val, _ = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        val_rmse = float(np.sqrt(np.mean((y_pred_val - y_true_val) ** 2)))

        line = f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_rmse_mm={val_rmse:.4f} train_time={train_time:.1f}s val_time={val_time:.1f}s"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()
        csv_fh.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_rmse:.4f},{train_time:.2f},{val_time:.2f}\n")
        csv_fh.flush()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse_mm": val_rmse})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "config": config},
                out_dir / "checkpoint_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch} (best={best_epoch})", flush=True)
                log_fh.write(f"early stopping at epoch {epoch} (best={best_epoch})\n")
                break

    total_train_time = time.time() - train_start
    log_fh.close()
    csv_fh.close()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # loss curve plot
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", color="tab:blue", label="train MSE")
    ax.plot(epochs, [h["val_loss"] for h in history], "o-", color="tab:red", label="val MSE")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalized ΔSWE units)")
    ax.set_title(f"B{'0' if args.model=='lstm' else '1'} ({args.model.upper()}): train vs val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Evaluate best checkpoint on validation.
    # -----------------------------------------------------------------
    checkpoint = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, y_true_val, y_pred_val, active_val = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
    val_report = full_metrics_report(y_true_val, y_pred_val, active_val)
    zero_val = compute_metrics(y_true_val, np.zeros_like(y_true_val))

    metrics = {
        "model": args.model,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "train_loss_at_best_epoch": history[best_epoch - 1]["train_loss"],
        "val_loss_at_best_epoch": best_val_loss,
        "total_train_time_s": total_train_time,
        "batch_size": args.batch_size,
        "val": val_report,
        "zero_change_baseline_val": zero_val,
    }
    (out_dir / "metrics_best.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    np.savez_compressed(
        out_dir / "predictions_val.npz",
        val_true_mm=y_true_val, val_pred_mm=y_pred_val, val_active=active_val,
        val_water_year=np.array([val_ds.water_year[i] for i in range(len(val_ds))]),
        val_cal_year=np.array([val_ds.cal_year[i] for i in range(len(val_ds))]),
        val_cal_month=np.array([val_ds.cal_month[i] for i in range(len(val_ds))]),
        val_cal_day=np.array([val_ds.cal_day[i] for i in range(len(val_ds))]),
        val_swe_t=np.array([val_ds.swe_t[i] for i in range(len(val_ds))]),
        val_swe_t1=np.array([val_ds.swe_t1[i] for i in range(len(val_ds))]),
    )
    print(f"wrote {out_dir / 'predictions_val.npz'}", flush=True)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
