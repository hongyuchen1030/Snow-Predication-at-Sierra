#!/usr/bin/env python3
"""Compare B0 (LSTM) vs B1 (TCNN) after both have finished training.
Reads only small, already-computed artifacts (metrics_best.json, history.json)
-- run on a compute node per the NERSC execution rule (any plotting counts as
computation), even though the inputs are tiny."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_temporal_baselines_v2")
OUT = EXPERIMENT_ROOT / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

models = {"lstm": "B0 LSTM", "tcnn": "B1 TCNN"}
metrics = {}
histories = {}
for m in models:
    metrics[m] = json.loads((EXPERIMENT_ROOT / m / "metrics_best.json").read_text())
    histories[m] = json.loads((EXPERIMENT_ROOT / m / "history.json").read_text())

# comparison/metrics.csv
fields = [
    "model", "n_params", "best_epoch", "train_loss_at_best_epoch", "val_loss_at_best_epoch",
    "val_rmse_mm", "val_mae_mm", "val_r2", "val_pearson_r", "val_sign_accuracy",
    "pred_mean_mm", "pred_std_mm", "target_mean_mm", "target_std_mm",
    "zero_change_rmse_mm", "zero_change_mae_mm", "zero_change_r2",
    "total_train_time_s",
]
with (OUT / "metrics.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(fields)
    for m, label in models.items():
        met = metrics[m]
        ov = met["val"]["overall"]
        zc = met["zero_change_baseline_val"]
        w.writerow([
            label, met["n_params"], met["best_epoch"], met["train_loss_at_best_epoch"], met["val_loss_at_best_epoch"],
            ov["rmse_mm"], ov["mae_mm"], ov["r2"], ov["pearson_r"], ov["sign_accuracy"],
            ov["pred_mean_mm"], ov["pred_std_mm"], ov["target_mean_mm"], ov["target_std_mm"],
            zc["rmse_mm"], zc["mae_mm"], zc["r2"],
            met["total_train_time_s"],
        ])
print(f"wrote {OUT / 'metrics.csv'}", flush=True)

# comparison/loss_curve_comparison.png
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
colors = {"lstm": "tab:blue", "tcnn": "tab:orange"}
for m, label in models.items():
    h = histories[m]
    epochs = [r["epoch"] for r in h]
    axes[0].plot(epochs, [r["train_loss"] for r in h], "-", color=colors[m], label=f"{label} train")
    axes[0].plot(epochs, [r["val_loss"] for r in h], "--", color=colors[m], label=f"{label} val")
    axes[1].plot(epochs, [r["val_loss"] for r in h], "-", color=colors[m], label=f"{label} val")
    axes[1].axvline(metrics[m]["best_epoch"], color=colors[m], linestyle=":", linewidth=0.8)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE loss"); axes[0].set_title("train vs val loss, both models"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val MSE loss"); axes[1].set_title("val loss comparison (best epoch marked)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "loss_curve_comparison.png")
plt.close(fig)
print(f"wrote {OUT / 'loss_curve_comparison.png'}", flush=True)

# comparison/README.md
lines = ["# B0 (LSTM) vs B1 (TCNN) temporal baseline comparison", ""]
lines.append("Reference: Shiheng Duan et al. (2024), \"Using Temporal Deep Learning Models to Estimate Daily Snow Water Equivalent Over the Rocky Mountains\" (https://github.com/ShihengDuan/code-SWE), adapted for T=7 global-atmospheric-field input via a shared per-day spatial encoder.")
lines.append("")
lines.append("| metric | B0 LSTM | B1 TCNN |")
lines.append("|---|---|---|")
for field, fmt in [
    ("n_params", "{}"), ("best_epoch", "{}"), ("train_loss_at_best_epoch", "{:.6f}"), ("val_loss_at_best_epoch", "{:.6f}"),
]:
    lines.append(f"| {field} | {fmt.format(metrics['lstm'][field])} | {fmt.format(metrics['tcnn'][field])} |")
for field, fmt in [
    ("rmse_mm", "{:.4f}"), ("mae_mm", "{:.4f}"), ("r2", "{:.4f}"), ("pearson_r", "{:.4f}"), ("sign_accuracy", "{:.4f}"),
    ("pred_mean_mm", "{:.4f}"), ("pred_std_mm", "{:.4f}"), ("target_mean_mm", "{:.4f}"), ("target_std_mm", "{:.4f}"),
]:
    lines.append(f"| val_{field} | {fmt.format(metrics['lstm']['val']['overall'][field])} | {fmt.format(metrics['tcnn']['val']['overall'][field])} |")
zc_l, zc_t = metrics["lstm"]["zero_change_baseline_val"], metrics["tcnn"]["zero_change_baseline_val"]
lines.append(f"| zero_change_rmse_mm | {zc_l['rmse_mm']:.4f} | {zc_t['rmse_mm']:.4f} |")
lines.append(f"| zero_change_r2 | {zc_l['r2']:.4f} | {zc_t['r2']:.4f} |")
lines.append(f"| total_train_time_s | {metrics['lstm']['total_train_time_s']:.1f} | {metrics['tcnn']['total_train_time_s']:.1f} |")
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(f"wrote {OUT / 'README.md'}", flush=True)
print("COMPARISON_DONE", flush=True)
