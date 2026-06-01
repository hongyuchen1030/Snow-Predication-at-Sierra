#!/usr/bin/env python3
"""
Regenerate a single region-by-month scatter grid from saved LOYO predictions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {
    "ridge": "#2a9d8f",
    "random_forest": "#bc6c25",
    "mlp": "#577590",
}


def scatter_with_identity(ax, observed: np.ndarray, predicted: np.ndarray, color: str, title: str) -> None:
    ax.scatter(observed, predicted, s=28, alpha=0.75, color=color, edgecolors="none")
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if np.any(finite):
        limits = [
            float(np.nanmin(np.concatenate([observed[finite], predicted[finite]]))),
            float(np.nanmax(np.concatenate([observed[finite], predicted[finite]]))),
        ]
        padding = 0.05 * (limits[1] - limits[0] if limits[1] > limits[0] else 1.0)
        lower = limits[0] - padding
        upper = limits[1] + padding
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.1, linestyle="--")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="ridge")
    parser.add_argument("--output-name", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(output_dir / "loyo_predictions_long.csv")
    metrics_df = pd.read_csv(output_dir / "metrics_region_by_month.csv")

    model_df = long_df[long_df["model"] == args.model].copy()
    model_metrics = metrics_df[metrics_df["model"] == args.model].copy()
    if model_df.empty:
        raise ValueError(f"No saved predictions found for model={args.model}")

    region_order = (
        model_df[["region_label", "region_name"]]
        .drop_duplicates()
        .sort_values("region_label")
        .reset_index(drop=True)
    )
    month_order = ["Dec", "Jan", "Feb", "Mar", "Apr"]
    available_months = [month for month in month_order if month in model_df["target_month"].unique().tolist()]

    fig, axes = plt.subplots(
        len(region_order),
        len(available_months),
        figsize=(4.0 * len(available_months), 3.8 * len(region_order)),
        constrained_layout=True,
    )
    if len(region_order) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(available_months) == 1:
        axes = np.expand_dims(axes, axis=1)

    for region_index, region_row in region_order.iterrows():
        for month_index, month_name in enumerate(available_months):
            ax = axes[region_index, month_index]
            subset = model_df[
                (model_df["region_label"] == region_row["region_label"])
                & (model_df["target_month"] == month_name)
            ]
            metric_row = model_metrics[
                (model_metrics["region_label"] == region_row["region_label"])
                & (model_metrics["target_month"] == month_name)
            ].iloc[0]
            scatter_with_identity(
                ax,
                subset["y_true"].to_numpy(dtype=float),
                subset["y_pred"].to_numpy(dtype=float),
                MODEL_COLORS.get(args.model, "#2a9d8f"),
                (
                    f"Region {int(region_row['region_label'])} | {month_name}\n"
                    f"R2={metric_row['r2']:.3f}, corr={metric_row['corr']:.3f}\n"
                    f"RMSE={metric_row['rmse']:.3f}, MAE={metric_row['mae']:.3f}"
                ),
            )
            if region_index == len(region_order) - 1:
                ax.set_xlabel(f"Observed ({month_name})")
            else:
                ax.set_xlabel("")
            if month_index == 0:
                ax.set_ylabel(f"Predicted\n{region_row['region_name']}")
            else:
                ax.set_ylabel("")

    fig.suptitle(f"{args.model} observed vs predicted by region and month", fontsize=14)
    output_name = args.output_name or f"{args.model}_region_month_scatter_grid"
    fig.savefig(plots_dir / f"{output_name}.png", dpi=220)
    fig.savefig(plots_dir / f"{output_name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
