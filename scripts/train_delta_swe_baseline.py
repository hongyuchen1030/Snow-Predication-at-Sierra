#!/usr/bin/env python3
"""Train C0 (CNN) or C1 (ConvLSTM) on the 7-day -> 1-day Sierra SWE-change
task. Also supports --benchmark-only for the pre-flight loader/epoch timing
check required before committing to a full run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from delta_swe_dataset import DeltaSWEDataset, collate_delta_swe  # noqa: E402
from delta_swe_models import build_model, count_parameters  # noqa: E402

EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1")
INDEX_PATH = EXPERIMENT_ROOT / "data_index" / "sample_index.npz"
SPLIT_PATH = EXPERIMENT_ROOT / "splits" / "water_year_split.json"
NORM_DIR = EXPERIMENT_ROOT / "normalization"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "convlstm"], required=True)
    p.add_argument("--season-start-month", type=int, default=9, choices=[9, 10, 11, 12])
    p.add_argument("--sample-mode", choices=["all", "snow_active"], default="all")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--benchmark-batches", type=int, default=100)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def load_normalization() -> tuple[np.ndarray, np.ndarray, float, float]:
    feat = np.load(NORM_DIR / "feature_normalization.npz", allow_pickle=True)
    target = json.loads((NORM_DIR / "target_normalization.json").read_text())
    return feat["mean"], feat["std"], target["target_mu_mm"], target["target_sigma_mm"]


def build_datasets(args: argparse.Namespace) -> tuple[DeltaSWEDataset, DeltaSWEDataset, DeltaSWEDataset]:
    split = json.loads(SPLIT_PATH.read_text())
    feature_mean, feature_std, target_mu, target_sigma = load_normalization()
    common = dict(
        index_path=INDEX_PATH,
        season_start_month=args.season_start_month,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )
    train_ds = DeltaSWEDataset(water_years=set(split["train_water_years"]), sample_mode=args.sample_mode, **common)
    val_ds = DeltaSWEDataset(water_years=set(split["val_water_years"]), sample_mode=args.sample_mode, **common)
    test_ds = DeltaSWEDataset(water_years=set(split["test_water_years"]), sample_mode=args.sample_mode, **common)
    return train_ds, val_ds, test_ds


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
    return {"n": int(len(y_true_mm)), "rmse_mm": rmse, "mae_mm": mae, "r2": r2, "pearson_r": r, "sign_accuracy": sign_acc}


def full_metrics_report(y_true_mm: np.ndarray, y_pred_mm: np.ndarray, active: np.ndarray) -> dict:
    report = {"overall": compute_metrics(y_true_mm, y_pred_mm)}
    report["accumulation_gt0"] = compute_metrics(y_true_mm[y_true_mm > 0], y_pred_mm[y_true_mm > 0])
    report["melt_lt0"] = compute_metrics(y_true_mm[y_true_mm < 0], y_pred_mm[y_true_mm < 0])
    report["zero_eq0"] = compute_metrics(y_true_mm[y_true_mm == 0], y_pred_mm[y_true_mm == 0])
    report["snow_active_only"] = compute_metrics(y_true_mm[active], y_pred_mm[active])
    return report


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, target_mu=0.0, target_sigma=1.0, use_amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
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
        true_mm = batch["delta_swe_mm"].numpy()
        all_true.append(true_mm)
        all_pred.append(pred_mm)
        all_active.append(batch["active"].numpy())
    elapsed = time.time() - t0
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    active = np.concatenate(all_active)
    return total_loss / max(n_batches, 1), elapsed, y_true, y_pred, active


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} season_start_month={args.season_start_month} sample_mode={args.sample_mode}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(args)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} samples", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_delta_swe, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_delta_swe, persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_delta_swe,
    )

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"model={args.model} n_params={n_params}", flush=True)

    _, _, target_mu, target_sigma = load_normalization()

    config = vars(args) | {"n_params": n_params, "train_n": len(train_ds), "val_n": len(val_ds), "test_n": len(test_ds)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    use_amp = (not args.no_amp) and device.type == "cuda"

    if args.benchmark_only:
        print("=== BENCHMARK MODE ===", flush=True)
        # loader benchmark: time to fetch benchmark_batches batches
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
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            peak_mem_gb = float("nan")
        t_train = time.time() - t0
        print(f"train: {n} batches in {t_train:.3f}s ({t_train/n:.4f}s/batch), peak_mem={peak_mem_gb:.2f}GB", flush=True)

        # one full epoch timing
        t0 = time.time()
        _, train_epoch_time, _, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        t0v = time.time()
        _, val_epoch_time, _, _, _ = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)

        n_batches_per_epoch = len(train_loader)
        est_per_epoch = train_epoch_time + val_epoch_time
        bench_report = {
            "loader_first_batch_s": t_first,
            "loader_avg_s_per_batch": t_loader / n,
            "train_avg_s_per_batch": t_train / n,
            "peak_gpu_mem_gb": peak_mem_gb,
            "one_train_epoch_s": train_epoch_time,
            "one_val_epoch_s": val_epoch_time,
            "n_train_batches_per_epoch": n_batches_per_epoch,
            "est_s_per_full_epoch": est_per_epoch,
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

    train_start = time.time()
    log_path = out_dir / "train.log"
    log_fh = log_path.open("w")

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_time, _, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        val_loss, val_time, y_true_val, y_pred_val, _ = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
        val_rmse = float(np.sqrt(np.mean((y_pred_val - y_true_val) ** 2)))
        scheduler.step(val_loss)

        line = f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_rmse_mm={val_rmse:.4f} train_time={train_time:.1f}s val_time={val_time:.1f}s lr={optimizer.param_groups[0]['lr']:.2e}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse_mm": val_rmse})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "config": config},
                out_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch} (best={best_epoch})", flush=True)
                log_fh.write(f"early stopping at epoch {epoch} (best={best_epoch})\n")
                break

    total_train_time = time.time() - train_start
    log_fh.close()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # -----------------------------------------------------------------
    # Evaluate best checkpoint on val/test.
    # -----------------------------------------------------------------
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, y_true_val, y_pred_val, active_val = run_epoch(model, val_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)
    _, _, y_true_test, y_pred_test, active_test = run_epoch(model, test_loader, device, use_amp=use_amp, target_mu=target_mu, target_sigma=target_sigma)

    val_report = full_metrics_report(y_true_val, y_pred_val, active_val)
    test_report = full_metrics_report(y_true_test, y_pred_test, active_test)

    zero_val = compute_metrics(y_true_val, np.zeros_like(y_true_val))
    zero_test = compute_metrics(y_true_test, np.zeros_like(y_true_test))

    metrics = {
        "model": args.model,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "total_train_time_s": total_train_time,
        "batch_size": args.batch_size,
        "val": val_report,
        "test": test_report,
        "zero_change_baseline_val": zero_val,
        "zero_change_baseline_test": zero_test,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    np.savez_compressed(
        out_dir / "predictions.npz",
        val_true_mm=y_true_val, val_pred_mm=y_pred_val, val_active=active_val,
        test_true_mm=y_true_test, test_pred_mm=y_pred_test, test_active=active_test,
        test_water_year=np.array([test_ds.water_year[i] for i in range(len(test_ds))]),
        test_cal_year=np.array([test_ds.cal_year[i] for i in range(len(test_ds))]),
        test_cal_month=np.array([test_ds.cal_month[i] for i in range(len(test_ds))]),
        test_cal_day=np.array([test_ds.cal_day[i] for i in range(len(test_ds))]),
        test_swe_t=np.array([test_ds.swe_t[i] for i in range(len(test_ds))]),
        test_swe_t1=np.array([test_ds.swe_t1[i] for i in range(len(test_ds))]),
    )
    print(f"wrote {out_dir / 'predictions.npz'}", flush=True)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
