#!/usr/bin/env python
"""Diagnostic plots for the round-1 frozen-ClimaX observational probe.
Reads only the already-saved predictions/metrics from run_probe.py."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/climax_frozen_obs_probe_round1"
PRED_DIR = os.path.join(OUT_ROOT, "predictions")
PLOT_DIR = os.path.join(OUT_ROOT, "plots")


def scatter_plot(df: pd.DataFrame, title: str, out_path: str, metrics_row: pd.Series):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["y_true"], df["y_pred"], s=30, alpha=0.8)
    lims = [min(df["y_true"].min(), df["y_pred"].min()), max(df["y_true"].max(), df["y_pred"].max())]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Observed (standardized)")
    ax.set_ylabel("LOYO predicted (standardized)")
    ax.set_title(title)
    txt = f"r={metrics_row['r']:.3f}\nR²={metrics_row['R2']:.3f}\nRMSE={metrics_row['RMSE']:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    metrics = pd.read_csv(os.path.join(OUT_ROOT, "metrics.csv"))

    # A. SWE: best raw-ERA5 probe and best frozen-ClimaX probe
    swe_metrics = metrics[metrics["target"] == "SWE"].sort_values("R2", ascending=False)
    best_raw = swe_metrics[swe_metrics["representation"].str.startswith("Raw-ERA5")].iloc[0]
    best_climax = swe_metrics[swe_metrics["representation"].str.startswith("ClimaX")].iloc[0]

    for tag, row in [("best_raw_era5", best_raw), ("best_climax", best_climax)]:
        df = pd.read_csv(row["predictions_path"])
        scatter_plot(df, f"SWE: {row['representation']}", os.path.join(PLOT_DIR, f"swe_{tag}_obs_vs_pred.png"), row)

    # B. CPM: best ClimaX probe
    cpm_metrics = metrics[(metrics["target"] == "CPM") & (metrics["representation"].str.startswith("ClimaX"))]
    best_cpm = cpm_metrics.sort_values("R2", ascending=False).iloc[0]
    df = pd.read_csv(best_cpm["predictions_path"])
    scatter_plot(df, f"CPM: {best_cpm['representation']}", os.path.join(PLOT_DIR, "cpm_best_climax_obs_vs_pred.png"), best_cpm)

    # C. AQM: best ClimaX probe
    aqm_metrics = metrics[(metrics["target"] == "AQM") & (metrics["representation"].str.startswith("ClimaX"))]
    best_aqm = aqm_metrics.sort_values("R2", ascending=False).iloc[0]
    df = pd.read_csv(best_aqm["predictions_path"])
    scatter_plot(df, f"AQM: {best_aqm['representation']}", os.path.join(PLOT_DIR, "aqm_best_climax_obs_vs_pred.png"), best_aqm)

    # D. compact bar chart: r and R2 across representations and targets
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets = ["SWE", "CPM", "AQM"]
    reps = metrics["representation"].unique().tolist()
    width = 0.2
    x = np.arange(len(targets))
    for ax, metric_name in zip(axes, ["r", "R2"]):
        for i, rep in enumerate(reps):
            vals = []
            for t in targets:
                row = metrics[(metrics["target"] == t) & (metrics["representation"] == rep)]
                vals.append(row[metric_name].iloc[0] if len(row) else np.nan)
            ax.bar(x + i * width, vals, width, label=rep)
        ax.set_xticks(x + width * (len(reps) - 1) / 2)
        ax.set_xticklabels(targets)
        ax.set_ylabel(metric_name)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"Held-out {metric_name} by representation and target")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "comparison_bar_chart.png"), dpi=150)
    plt.close(fig)

    print("Plots written to", PLOT_DIR, flush=True)
    print("best_raw_era5_swe:", best_raw["representation"], flush=True)
    print("best_climax_swe:", best_climax["representation"], flush=True)
    print("best_climax_cpm:", best_cpm["representation"], flush=True)
    print("best_climax_aqm:", best_aqm["representation"], flush=True)


if __name__ == "__main__":
    main()
