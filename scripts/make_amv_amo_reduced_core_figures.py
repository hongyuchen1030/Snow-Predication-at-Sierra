#!/usr/bin/env python3
"""
Create reduced-core AMV/AMO figures from the clean deduplicated forward-addition run.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_core_plus_forward_addition"
)
PREDICTIONS_CSV = INPUT_DIR / "amv_core_plus_forward_loyo_predictions.csv"
METRICS_CSV = INPUT_DIR / "amv_core_plus_forward_loyo_metrics.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"
OBS_PRED_PDF = FIGURES_DIR / "amv_amo_reduced_core_observed_vs_predicted.pdf"
SCATTER_PDF = FIGURES_DIR / "amv_amo_reduced_core_scatter.pdf"

MODELS = [
    ("AMV_core_plus_K4", "K4 starting core", "#1f77b4"),
    ("AMV_core_plus_K5", "K5 reduced core", "#ff7f0e"),
    ("AMV_AMO_PC1to6_full", "Full AMV/AMO PC1--PC6", "#d62728"),
]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def compute_limits(obs_all: np.ndarray, pred_all: np.ndarray) -> tuple[float, float]:
    lo = float(min(np.min(obs_all), np.min(pred_all)))
    hi = float(max(np.max(obs_all), np.max(pred_all)))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    return lo - pad, hi + pad


def make_timeseries(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    k4 = pred_df[pred_df["model_name"] == "AMV_core_plus_K4"].sort_values("heldout_wy")
    ax.plot(
        k4["heldout_wy"],
        k4["obs_swe"],
        color="black",
        linewidth=2.5,
        label="Observed",
    )
    for model_name, label, color in MODELS:
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(
            sub["heldout_wy"],
            sub["pred_swe"],
            color=color,
            marker="o",
            linewidth=1.7,
            label=label,
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Observed vs predicted for reduced-core AMV/AMO models")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OBS_PRED_PDF)
    plt.close(fig)


def make_scatter(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharex=True, sharey=True)
    all_obs = pred_df["obs_swe"].to_numpy(dtype=float)
    all_pred = pred_df["pred_swe"].to_numpy(dtype=float)
    lims = compute_limits(all_obs, all_pred)
    metric_lookup = metrics_df.set_index("model_name")
    for ax, (model_name, label, color) in zip(axes, MODELS):
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        row = metric_lookup.loc[model_name]
        ax.scatter(obs, pred, color=color, s=48, alpha=0.85)
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(label)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.04,
            0.96,
            "RMSE = {:.4f}\n$R^2$ = {:.3f}\nCorr = {:.3f}".format(
                float(row["RMSE"]),
                float(row["R2"]),
                float(row["r"]),
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
    axes[0].set_ylabel("Predicted SWE anomaly (m)")
    for ax in axes:
        ax.set_xlabel("Observed SWE anomaly (m)")
    fig.tight_layout()
    fig.savefig(SCATTER_PDF)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    metrics_df = pd.read_csv(METRICS_CSV)
    pred_df = pred_df[pred_df["model_name"].isin([name for name, _, _ in MODELS])].copy()
    metrics_df = metrics_df[metrics_df["model_name"].isin([name for name, _, _ in MODELS])].copy()
    make_timeseries(pred_df)
    make_scatter(pred_df, metrics_df)
    print(f"Created {OBS_PRED_PDF}")
    print(f"Created {SCATTER_PDF}")


if __name__ == "__main__":
    main()
