#!/usr/bin/env python3
"""
Run a regionwise debug Ridge experiment for Sierra top-20% T2m regions.

This debug mode intentionally includes the target year in the training data.
For each water year, the model is fit on all 73 years and then predicts that
same year's target. The goal is to inspect the in-sample ceiling rather than
LOYO generalization skill.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import scripts.run_s2s_pc6_t2m_top20_regions_loyo_ridge_regionwise as ridge_mod


DEFAULT_OUTPUT_DIR = (
    ridge_mod.PROJECT_ROOT
    / "artifacts"
    / "s2s_pc6_t2m_top20_regions_trainall_ridge_regionwise_debug_decinit_fma"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ridge_mod.DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--label-dir", type=Path, default=ridge_mod.DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-months", nargs="+", default=["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    parser.add_argument("--target-months", nargs="+", default=["Feb", "Mar", "Apr"])
    return parser.parse_args()


def run_trainall_debug(
    dataset: ridge_mod.DatasetBundle,
) -> Tuple[ridge_mod.np.ndarray, ridge_mod.pd.DataFrame]:
    water_years = dataset.water_years
    x = dataset.x
    predictions_by_region = {
        region.semantic_label: ridge_mod.np.full_like(
            dataset.y_by_region[region.semantic_label],
            ridge_mod.np.nan,
            dtype=ridge_mod.np.float64,
        )
        for region in dataset.region_definitions
    }
    hyperparameter_rows: List[Dict[str, object]] = []

    print(
        "Starting debug train-all-years ridge sweep across "
        f"{water_years.size} water years and {len(dataset.region_definitions)} regions",
        flush=True,
    )
    print(
        "Debug mode: each prediction year is also included in the training data.",
        flush=True,
    )

    for region in dataset.region_definitions:
        y_region = dataset.y_by_region[region.semantic_label]
        best_alpha = None
        best_score = -ridge_mod.np.inf
        for alpha in ridge_mod.RIDGE_ALPHA_VALUES:
            score = ridge_mod.score_alpha(alpha, x, y_region)
            if score > best_score:
                best_score = score
                best_alpha = alpha
        if best_alpha is None:
            raise RuntimeError(f"No alpha selected for region {region.semantic_label}")

        pred_scaled, _, y_scaler = ridge_mod.fit_ridge_scaled(best_alpha, x, y_region, x)
        predictions_by_region[region.semantic_label][:, :] = y_scaler.inverse_transform(pred_scaled)

        for outer_index, test_water_year in enumerate(water_years):
            hyperparameter_rows.append(
                {
                    "outer_water_year": int(test_water_year),
                    "region_label": region.semantic_label,
                    "region_name": region.semantic_name,
                    "selected_alpha": float(best_alpha),
                    "inner_cv_score_neg_mean_mse_scaled_y": float(best_score),
                    "train_year_included_in_fit": True,
                }
            )
            print(
                f"DEBUG train_all predict_WY={int(test_water_year)} region={region.semantic_label} "
                f"alpha={best_alpha} inner_score={best_score:.6f}",
                flush=True,
            )

    ordered_predictions = ridge_mod.np.concatenate(
        [predictions_by_region[region.semantic_label] for region in dataset.region_definitions],
        axis=1,
    )
    hyperparameter_df = (
        ridge_mod.pd.DataFrame(hyperparameter_rows)
        .sort_values(["region_label", "outer_water_year"])
        .reset_index(drop=True)
    )
    print("Finished debug train-all-years ridge sweep", flush=True)
    return ordered_predictions, hyperparameter_df


def save_run_config_debug(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: ridge_mod.DatasetBundle,
    hyperparameter_df: ridge_mod.pd.DataFrame,
) -> None:
    package_versions = {
        "numpy": ridge_mod.np.__version__,
        "pandas": ridge_mod.pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "matplotlib": ridge_mod.matplotlib.__version__,
        "xarray": ridge_mod.xr.__version__,
        "scipy": __import__("scipy").__version__,
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_path": str(ridge_mod.PROJECT_ROOT),
        "artifact_dir": str(args.artifact_dir),
        "label_dir": str(args.label_dir),
        "output_dir": str(args.output_dir),
        "route2_netcdf": dataset.paths_used["route2_netcdf"],
        "expanded_level2_netcdf": dataset.paths_used["expanded_level2_netcdf"],
        "region_label_file": dataset.paths_used["label_netcdf"],
        "water_years_used": dataset.water_years.tolist(),
        "input_months": list(args.input_months),
        "target_months": list(args.target_months),
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "alpha_grid": ridge_mod.RIDGE_ALPHA_VALUES,
        "cv": "debug_train_all_years_with_inner_cv",
        "mode": "regionwise multi-output ridge debug train-all-years",
        "train_year_included_in_fit": True,
        "t2m_source_note": dataset.t2m_source_note,
        "package_versions": package_versions,
        "selected_hyperparameters_preview": hyperparameter_df.head(12).to_dict(orient="records"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ridge_mod.MODEL_NAME = "ridge_regionwise_trainall_debug"
    ridge_mod.MODEL_DISPLAY = "Ridge Regionwise Train-All Debug"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifact_candidates = ridge_mod.candidate_files_from_dir(args.artifact_dir)
    label_candidates = ridge_mod.candidate_files_from_dir(args.label_dir)
    route2_netcdf = ridge_mod.locate_route2_netcdf(args.artifact_dir)
    label_netcdf = ridge_mod.locate_label_netcdf(args.label_dir)
    relevant_candidates = [
        route2_netcdf,
        ridge_mod.EXPANDED_LEVEL2_NETCDF_FILE,
        label_netcdf,
        ridge_mod.DEFAULT_MONTHLY_ANOMALY_FILE,
        ridge_mod.ERA5_MONTHLY_MEAN_FILE,
        ridge_mod.ERA5_MONTHLY_CLIM_FILE,
        ridge_mod.COBE2_SST_FILE,
    ]
    ridge_mod.print_candidate_group("Candidate files in artifact directory:", artifact_candidates)
    ridge_mod.print_candidate_group("Candidate files in region-label directory:", label_candidates)
    ridge_mod.print_candidate_group("Relevant data files reused from prior scripts:", relevant_candidates)

    dataset = ridge_mod.build_supervised_dataset(route2_netcdf, label_netcdf, args.input_months, args.target_months)
    ridge_mod.print_dataset_summary(dataset)
    ridge_mod.save_dataset_files(dataset, args.output_dir, args.target_months)

    y_pred, hyperparameter_df = run_trainall_debug(dataset)
    hyperparameter_df.to_csv(args.output_dir / "selected_hyperparameters_by_fold.csv", index=False)

    overall_metrics, region_metrics, month_metrics, region_month_metrics, metrics_summary = ridge_mod.compute_all_metrics(
        dataset.y,
        y_pred,
        dataset.region_definitions,
        args.target_months,
    )
    overall_metrics.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    region_metrics.to_csv(args.output_dir / "metrics_by_region.csv", index=False)
    month_metrics.to_csv(args.output_dir / "metrics_by_month.csv", index=False)
    region_month_metrics.to_csv(args.output_dir / "metrics_region_by_month.csv", index=False)
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n", encoding="utf-8")

    ridge_mod.save_predictions(
        args.output_dir,
        dataset.water_years,
        dataset.y,
        y_pred,
        dataset.region_definitions,
        args.target_months,
    )

    ridge_mod.plot_overall_scatter(plots_dir, overall_metrics, dataset.y, y_pred)
    ridge_mod.plot_region_scatter(plots_dir, region_metrics, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    ridge_mod.plot_month_scatter(plots_dir, month_metrics, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    ridge_mod.plot_region_month_scatter_grid(
        plots_dir,
        region_month_metrics,
        dataset.y,
        y_pred,
        dataset.region_definitions,
        args.target_months,
    )
    ridge_mod.plot_time_series(plots_dir, dataset.water_years, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    ridge_mod.plot_region_month_heatmap(plots_dir, region_month_metrics, dataset.region_definitions, args.target_months)
    ridge_mod.plot_selected_alpha(plots_dir, hyperparameter_df, dataset.region_definitions)

    save_run_config_debug(args.output_dir, args, dataset, hyperparameter_df)
    ridge_mod.print_final_summary(args.output_dir, dataset, overall_metrics, region_metrics, month_metrics, hyperparameter_df)


if __name__ == "__main__":
    main()
