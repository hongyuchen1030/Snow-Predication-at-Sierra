#!/usr/bin/env python3
"""
Reproduce the legacy train-all fixed-alpha "best scatter" plots without
overwriting the current consistent train-all debug outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import scripts.run_s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm as ridge_mod


LEGACY_DEBUG_ALPHA_VALUES = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
DEFAULT_OUTPUT_DIR = (
    ridge_mod.DEFAULT_OUTPUT_DIR
    / "trainall_debug"
    / "legacy_fixed_alpha_best_scatter"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ridge_mod.DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--label-dir", type=Path, default=ridge_mod.DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-months", nargs="+", default=["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"])
    parser.add_argument("--target-months", nargs="+", default=["Jan", "Feb", "Mar"])
    return parser.parse_args()


def compute_fixed_alpha_debug(
    dataset: ridge_mod.DatasetBundle,
    target_months: Sequence[str],
) -> (pd.DataFrame, pd.DataFrame):
    rows_metrics = []
    rows_long = []
    y_true_regionwise = dataset.y_regionwise.reshape(
        dataset.y_regionwise.shape[0],
        len(dataset.region_definitions),
        len(target_months),
    )

    for alpha in LEGACY_DEBUG_ALPHA_VALUES:
        top20_pred = ridge_mod.fit_scaled_model(alpha, dataset.x, dataset.y_top20_all, dataset.x)
        top20_metrics = ridge_mod.metric_block(dataset.y_top20_all.reshape(-1), top20_pred.reshape(-1))
        rows_metrics.append(
            {
                "target_definition": ridge_mod.TOP20_NAME,
                "model_name": "linear_regression" if alpha <= 0.0 else "ridge",
                "debug_alpha": float(alpha),
                "region_label": 0,
                "region_name": ridge_mod.TOP20_NAME,
                **top20_metrics
            }
        )
        for row_index, water_year in enumerate(dataset.water_years.tolist()):
            for month_index, month_name in enumerate(target_months):
                rows_long.append(
                    {
                        "target_definition": ridge_mod.TOP20_NAME,
                        "model_name": "linear_regression" if alpha <= 0.0 else "ridge",
                        "debug_alpha": float(alpha),
                        "water_year": int(water_year),
                        "region_label": 0,
                        "region_name": ridge_mod.TOP20_NAME,
                        "target_month": month_name,
                        "y_true": float(dataset.y_top20_all[row_index, month_index]),
                        "y_pred": float(top20_pred[row_index, month_index]),
                        "error": float(top20_pred[row_index, month_index] - dataset.y_top20_all[row_index, month_index]),
                    }
                )

        for region_index, region in enumerate(dataset.region_definitions):
            region_name = ridge_mod.REGION_NAME_BY_LABEL[region.semantic_label]
            region_pred = ridge_mod.fit_scaled_model(alpha, dataset.x, y_true_regionwise[:, region_index, :], dataset.x)
            region_metrics = ridge_mod.metric_block(y_true_regionwise[:, region_index, :].reshape(-1), region_pred.reshape(-1))
            rows_metrics.append(
                {
                    "target_definition": "regionwise",
                    "model_name": "linear_regression" if alpha <= 0.0 else "ridge",
                    "debug_alpha": float(alpha),
                    "region_label": region.semantic_label,
                    "region_name": region_name,
                    **region_metrics
                }
            )
            for row_index, water_year in enumerate(dataset.water_years.tolist()):
                for month_index, month_name in enumerate(target_months):
                    rows_long.append(
                        {
                            "target_definition": "regionwise",
                            "model_name": "linear_regression" if alpha <= 0.0 else "ridge",
                            "debug_alpha": float(alpha),
                            "water_year": int(water_year),
                            "region_label": region.semantic_label,
                            "region_name": region_name,
                            "target_month": month_name,
                            "y_true": float(y_true_regionwise[row_index, region_index, month_index]),
                            "y_pred": float(region_pred[row_index, month_index]),
                            "error": float(region_pred[row_index, month_index] - y_true_regionwise[row_index, region_index, month_index]),
                        }
                    )

    metrics_df = pd.DataFrame(rows_metrics).sort_values(["target_definition", "region_label", "debug_alpha"]).reset_index(drop=True)
    predictions_df = pd.DataFrame(rows_long).sort_values(
        ["debug_alpha", "target_definition", "region_label", "water_year", "target_month"]
    ).reset_index(drop=True)
    return metrics_df, predictions_df


def plot_legacy_best_scatter(
    plots_dir: Path,
    dataset: ridge_mod.DatasetBundle,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    top20_metrics = metrics_df[metrics_df["target_definition"] == ridge_mod.TOP20_NAME].sort_values("debug_alpha")
    regionwise_metrics = metrics_df[metrics_df["target_definition"] == "regionwise"].sort_values(["region_label", "debug_alpha"])

    best_top20 = top20_metrics.sort_values("r2", ascending=False).iloc[0]
    top20_pred = predictions_df[
        (predictions_df["target_definition"] == ridge_mod.TOP20_NAME)
        & (predictions_df["debug_alpha"] == float(best_top20["debug_alpha"]))
    ]
    fig, ax = ridge_mod.plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    ridge_mod.scatter_with_identity(
        ax,
        top20_pred["y_true"].to_numpy(dtype=np.float64),
        top20_pred["y_pred"].to_numpy(dtype=np.float64),
        ridge_mod.TOP20_COLOR,
        (
            "Legacy fixed-alpha top20_all alpha={:g}\n"
            "R2={:.3f}, corr={:.3f}, RMSE={:.3f}, MAE={:.3f}"
        ).format(
            float(best_top20["debug_alpha"]),
            float(best_top20["r2"]),
            float(best_top20["corr"]),
            float(best_top20["rmse"]),
            float(best_top20["mae"]),
        ),
    )
    ridge_mod.save_fig(fig, plots_dir / "trainall_top20_best_scatter")

    fig, axes = ridge_mod.plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for region_index, region in enumerate(dataset.region_definitions):
        best_region = regionwise_metrics[regionwise_metrics["region_label"] == region.semantic_label].sort_values("r2", ascending=False).iloc[0]
        region_pred = predictions_df[
            (predictions_df["target_definition"] == "regionwise")
            & (predictions_df["region_label"] == region.semantic_label)
            & (predictions_df["debug_alpha"] == float(best_region["debug_alpha"]))
        ]
        ridge_mod.scatter_with_identity(
            axes[region_index],
            region_pred["y_true"].to_numpy(dtype=np.float64),
            region_pred["y_pred"].to_numpy(dtype=np.float64),
            ridge_mod.REGIONWISE_COLOR,
            (
                "{} alpha={:g}\nR2={:.3f}, corr={:.3f}, RMSE={:.3f}, MAE={:.3f}"
            ).format(
                ridge_mod.REGION_NAME_BY_LABEL[region.semantic_label],
                float(best_region["debug_alpha"]),
                float(best_region["r2"]),
                float(best_region["corr"]),
                float(best_region["rmse"]),
                float(best_region["mae"]),
            ),
        )
    ridge_mod.save_fig(fig, plots_dir / "trainall_regionwise_best_scatter")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    route2_netcdf = ridge_mod.base_mod.locate_route2_netcdf(args.artifact_dir)
    label_netcdf = ridge_mod.base_mod.locate_label_netcdf(args.label_dir)
    dataset = ridge_mod.build_supervised_dataset(route2_netcdf, label_netcdf, args.input_months, args.target_months)
    metrics_df, predictions_df = compute_fixed_alpha_debug(dataset, args.target_months)

    metrics_df.to_csv(args.output_dir / "legacy_fixed_alpha_metrics.csv", index=False)
    predictions_df.to_csv(args.output_dir / "legacy_fixed_alpha_predictions_long.csv", index=False)
    plot_legacy_best_scatter(plots_dir, dataset, metrics_df, predictions_df)

    summary = {
        "legacy_debug_alpha_values": LEGACY_DEBUG_ALPHA_VALUES,
        "best_top20_all_by_r2": metrics_df[metrics_df["target_definition"] == ridge_mod.TOP20_NAME]
        .sort_values("r2", ascending=False)
        .iloc[0]
        .to_dict(),
        "best_regionwise_rows_by_r2": metrics_df[metrics_df["target_definition"] == "regionwise"]
        .sort_values(["region_label", "r2"], ascending=[True, False])
        .groupby("region_label", sort=True)
        .head(1)
        .to_dict(orient="records"),
    }
    (args.output_dir / "legacy_fixed_alpha_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
