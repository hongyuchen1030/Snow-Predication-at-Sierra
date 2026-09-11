#!/usr/bin/env python3
"""Diagnostic plots for the completed T0 run: loss curves, generalization
gap, predicted-vs-true scatter, per-WY time series, SWE reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_transformer_t0_v1")
PLOTS = EXP / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

history = json.loads((EXP / "loss_curve.json").read_text())
preds = np.load(EXP / "predictions_val.npz", allow_pickle=True)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
gap = [h["generalization_gap"] for h in history]

# 1. train/val loss
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.plot(epochs, train_loss, "o-", color="tab:blue", label="train Huber loss")
ax.plot(epochs, val_loss, "o-", color="tab:red", label="validation Huber loss")
ax.set_xlabel("epoch")
ax.set_ylabel("Huber loss (normalized ΔSWE units)")
ax.set_title("T0 (factorized ST Transformer): train vs validation loss (raw, unsmoothed)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "loss_curve.png")
plt.close(fig)

# 2. generalization gap
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.plot(epochs, gap, "o-", color="tab:purple")
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set_xlabel("epoch")
ax.set_ylabel("val_loss - train_loss")
ax.set_title("T0: generalization gap per epoch (raw, unsmoothed)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "generalization_gap.png")
plt.close(fig)

# 3. predicted vs true scatter
true_mm = preds["val_true_mm"]
pred_mm = preds["val_pred_mm"]
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.scatter(true_mm, pred_mm, s=6, alpha=0.3, color="tab:blue")
lims = [min(true_mm.min(), pred_mm.min()), max(true_mm.max(), pred_mm.max())]
ax.plot(lims, lims, "k--", linewidth=0.8)
ax.set_xlabel("true ΔSWE (mm)")
ax.set_ylabel("predicted ΔSWE (mm)")
ax.set_title("T0: predicted vs true ΔSWE (validation)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "pred_vs_true_scatter.png")
plt.close(fig)

# 4. time series + SWE reconstruction for a few validation water years
val_wy = preds["val_water_year"]
val_cal_month = preds["val_cal_month"]
val_cal_day = preds["val_cal_day"]
val_swe_t = preds["val_swe_t"]
val_swe_t1 = preds["val_swe_t1"]

unique_wy = sorted(set(val_wy.tolist()))
chosen_wy = unique_wy[: min(4, len(unique_wy))]

fig, axes = plt.subplots(len(chosen_wy), 2, figsize=(13, 4 * len(chosen_wy)), dpi=130, squeeze=False)
for i, wy in enumerate(chosen_wy):
    mask = val_wy == wy
    order = np.argsort(val_cal_month[mask].astype(np.int64) * 100 + val_cal_day[mask].astype(np.int64))
    # sort chronologically within Oct1-Apr1 (months 10,11,12,1,2,3,4 need custom order)
    month_rank = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    idx = np.where(mask)[0]
    idx = sorted(idx, key=lambda j: (month_rank[int(val_cal_month[j])], int(val_cal_day[j])))
    idx = np.array(idx)

    t_true = true_mm[idx]
    t_pred = pred_mm[idx]
    swe_t = val_swe_t[idx]
    swe_t1_true = val_swe_t1[idx]

    axes[i, 0].plot(range(len(idx)), t_true, "o-", color="black", markersize=3, label="true ΔSWE")
    axes[i, 0].plot(range(len(idx)), t_pred, "o-", color="tab:red", markersize=3, alpha=0.7, label="predicted ΔSWE")
    axes[i, 0].set_title(f"WY{wy}: ΔSWE_1d time series")
    axes[i, 0].set_xlabel("day index (Oct1->Apr1)")
    axes[i, 0].set_ylabel("ΔSWE (mm)")
    axes[i, 0].legend(fontsize=8)
    axes[i, 0].grid(alpha=0.3)

    # SWE reconstruction: SWE_hat(t+1) = SWE_hat(t) + pred_delta(t), init with TRUE SWE at first day
    swe_hat = np.zeros(len(idx) + 1)
    swe_hat[0] = swe_t[0]
    for k in range(len(idx)):
        swe_hat[k + 1] = swe_hat[k] + t_pred[k]
    swe_true_series = np.concatenate([[swe_t[0]], swe_t1_true])

    axes[i, 1].plot(range(len(swe_true_series)), swe_true_series, "o-", color="black", markersize=3, label="true SWE")
    axes[i, 1].plot(range(len(swe_hat)), swe_hat, "o-", color="tab:red", markersize=3, alpha=0.7, label="reconstructed SWE (diagnostic)")
    axes[i, 1].set_title(f"WY{wy}: SWE reconstruction from predicted daily ΔSWE (diagnostic only, not autoregressive training)")
    axes[i, 1].set_xlabel("day index (Oct1->Apr1)")
    axes[i, 1].set_ylabel("SWE (mm)")
    axes[i, 1].legend(fontsize=8)
    axes[i, 1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PLOTS / "timeseries_and_swe_reconstruction.png")
plt.close(fig)

print("wrote plots:", list(PLOTS.iterdir()))
