from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import (
    DEFAULT_MODEL_REGION,
    add_region_args,
    get_regional_swe_grid_definition,
    region_from_args,
    region_to_dict,
)
from snow_ml.pc_baseline import (
    align_features_and_targets,
    assemble_feature_table,
    discover_swe_water_years,
    fit_swe_pca,
    load_apr1_swe_maps,
    predictor_cutoff_date,
    print_region_check,
    run_loyo,
    save_aligned_dataset_csv,
    save_metrics,
    save_predictions_csv,
    save_reconstruction_diagnostics,
    save_target_metadata,
    save_target_pca_npz,
    save_targets_csv,
)


def parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {text!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict April 1 SWE principal-component scores from yearly scalar predictors."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Water years to consider. If omitted, use all discovered SWE files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "pc_baseline",
        help="Directory for PC baseline CSV and JSON outputs.",
    )
    parser.add_argument("--predict-pc2", type=parse_bool, default=True)
    parser.add_argument(
        "--lead-months",
        type=int,
        default=0,
        help="Months of lead time relative to the current March 31 predictor cutoff. "
        "0 keeps the existing behavior, 1 uses the end of February, 2 uses the end of January, "
        "and 3 uses the end of December.",
    )
    parser.add_argument(
        "--lead-months-list",
        nargs="+",
        type=int,
        default=None,
        help="Optional experiment mode: run multiple lead times and write one subdirectory per lead.",
    )
    parser.add_argument(
        "--dec31-mode",
        choices=("stats", "pcs"),
        default="stats",
        help="Use Dec 31 SWE spatial mean/std or projection onto April 1 EOFs.",
    )
    parser.add_argument("--include-seasonal-features", type=parse_bool, default=True)
    parser.add_argument("--enable-predictor-pca", type=parse_bool, default=True)
    parser.add_argument("--save-reconstruction-diagnostics", type=parse_bool, default=False)
    parser.add_argument(
        "--save-plots",
        type=parse_bool,
        default=False,
        help="Run scripts/plot_pc_baseline_results.py after each baseline run.",
    )
    parser.add_argument(
        "--region-check-only",
        type=parse_bool,
        default=False,
        help="Print coordinate-only region diagnostics and exit before loading fields.",
    )
    add_region_args(parser)
    return parser.parse_args()


def run_single_experiment(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    lead_months: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    region = region_from_args(args, default=DEFAULT_MODEL_REGION)
    if region is None:
        raise ValueError("This baseline requires a finite model region.")

    candidate_years = sorted(args.years if args.years is not None else discover_swe_water_years())
    if not candidate_years:
        raise RuntimeError("No candidate SWE water years found.")

    print(f"candidate SWE water years: {candidate_years}", flush=True)
    print(f"requested region: {region_to_dict(region)}", flush=True)
    print(f"coarsen factor: {args.coarsen_factor}", flush=True)
    print(f"Dec 31 feature mode: {args.dec31_mode}", flush=True)
    print(f"lead months: {lead_months}", flush=True)
    print(f"example predictor cutoff date: {predictor_cutoff_date(candidate_years[0], lead_months)}", flush=True)
    print(f"predict PC2: {args.predict_pc2}", flush=True)
    print(f"include seasonal features: {args.include_seasonal_features}", flush=True)
    print(f"enable predictor PCA tuning: {args.enable_predictor_pca}", flush=True)

    grid = get_regional_swe_grid_definition(
        candidate_years[0],
        region,
        args.coarsen_factor,
    )
    print(f"effective SWE grid region: {region_to_dict(grid.effective_region)}", flush=True)
    print(f"coarsened SWE grid shape: {grid.grid_shape}", flush=True)
    print_region_check(
        expected_region=region,
        grid=grid,
        sample_year=candidate_years[0],
    )
    if args.region_check_only:
        print("region check only requested; exiting before target/feature loading", flush=True)
        return

    target_maps, target_drops = load_apr1_swe_maps(
        candidate_years,
        grid=grid,
        region=region,
        coarsen_factor=args.coarsen_factor,
    )
    print(f"available target years before alignment: {sorted(target_maps)}", flush=True)
    if target_drops:
        print(f"dropped target years before alignment: {target_drops}", flush=True)

    preliminary_feature_rows, preliminary_feature_drops = assemble_feature_table(
        candidate_years,
        grid=grid,
        region=region,
        dec31_mode="stats",
        include_seasonal_features=args.include_seasonal_features,
        target_pca=None,
        lead_months=lead_months,
    )
    preliminary_feature_years = [int(row["water_year"]) for row in preliminary_feature_rows]
    print(f"available feature years before alignment: {preliminary_feature_years}", flush=True)
    if preliminary_feature_drops:
        print(f"dropped feature years before alignment: {preliminary_feature_drops}", flush=True)

    usable_for_pca = sorted(set(target_maps).intersection(preliminary_feature_years))
    dropped_after_prealign = {
        year: "missing target or preliminary required predictors"
        for year in candidate_years
        if year not in usable_for_pca
    }
    if dropped_after_prealign:
        print(f"dropped years before target PCA: {dropped_after_prealign}", flush=True)
    target_pca = fit_swe_pca(
        {year: target_maps[year] for year in usable_for_pca},
        n_components=2 if args.predict_pc2 else 1,
        grid=grid,
    )

    if args.dec31_mode == "pcs":
        feature_rows, feature_drops = assemble_feature_table(
            target_pca.years,
            grid=grid,
            region=region,
            dec31_mode="pcs",
            include_seasonal_features=args.include_seasonal_features,
            target_pca=target_pca,
            lead_months=lead_months,
        )
    else:
        feature_rows = [
            row for row in preliminary_feature_rows
            if int(row["water_year"]) in set(target_pca.years)
        ]
        feature_drops = {}

    if feature_drops:
        print(f"dropped final feature years: {feature_drops}", flush=True)

    aligned_rows = align_features_and_targets(
        feature_rows,
        target_pca,
        predict_pc2=args.predict_pc2,
    )
    if not aligned_rows:
        raise RuntimeError("No aligned feature/target rows remain.")

    target_columns = ["pc1"] + (["pc2"] if args.predict_pc2 else [])
    feature_columns = [
        name for name in aligned_rows[0]
        if name not in {"water_year", *target_columns}
    ]
    aligned_years = [int(row["water_year"]) for row in aligned_rows]

    print(f"final aligned years: {aligned_years}", flush=True)
    print(f"number of features: {len(feature_columns)}", flush=True)
    print(f"feature columns used: {feature_columns}", flush=True)
    print(f"PC2 enabled: {args.predict_pc2}", flush=True)

    save_targets_csv(output_dir / "targets.csv", target_pca, predict_pc2=args.predict_pc2)
    save_target_metadata(output_dir / "target_pca_metadata.json", target_pca)
    save_target_pca_npz(output_dir / "target_pca_arrays.npz", target_pca)
    save_aligned_dataset_csv(
        output_dir / "aligned_dataset.csv",
        aligned_rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    prediction_rows, metrics = run_loyo(
        aligned_rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
        enable_predictor_pca=args.enable_predictor_pca,
    )
    save_predictions_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        target_columns=target_columns,
    )
    save_metrics(output_dir / "metrics.json", metrics)

    if args.save_reconstruction_diagnostics:
        save_reconstruction_diagnostics(
            output_dir / "reconstruction_diagnostics.csv",
            prediction_rows,
            target_pca,
        )

    run_metadata = {
        "lead_months": int(lead_months),
        "predictor_cutoff_dates_by_year": {
            str(year): predictor_cutoff_date(int(year), lead_months).isoformat()
            for year in aligned_years
        },
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "aligned_years": aligned_years,
        "predict_pc2": bool(args.predict_pc2),
        "dec31_mode": args.dec31_mode,
        "include_seasonal_features": bool(args.include_seasonal_features),
        "enable_predictor_pca": bool(args.enable_predictor_pca),
        "region": region_to_dict(region),
        "coarsen_factor": int(args.coarsen_factor),
        "target_pca_years": target_pca.years,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )

    if args.save_plots:
        _run_plot_script(output_dir)

    print(f"saved outputs to {output_dir}", flush=True)
    print(f"final metrics: {metrics}", flush=True)
    return {
        "lead_months": int(lead_months),
        "output_dir": str(output_dir),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "aligned_years": aligned_years,
        "cutoff_example": predictor_cutoff_date(aligned_years[0], lead_months).isoformat(),
    }


def save_leadtime_summary(
    output_dir: Path,
    run_summaries: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {"runs": run_summaries, "targets": {}}
    for summary in sorted(run_summaries, key=lambda item: int(item["lead_months"])):
        lead_months = int(summary["lead_months"])
        for target_name, metric_values in summary["metrics"].items():
            row = {
                "lead_months": lead_months,
                "target": target_name,
                "pearson_correlation": metric_values.get("pearson_correlation"),
                "r2": metric_values.get("r2"),
                "rmse": metric_values.get("rmse"),
                "mae": metric_values.get("mae"),
                "feature_count": len(summary["feature_columns"]),
                "year_count": len(summary["aligned_years"]),
                "cutoff_example": summary["cutoff_example"],
                "output_dir": summary["output_dir"],
            }
            rows.append(row)
            summary_json["targets"].setdefault(target_name, []).append(row)

    csv_header = [
        "lead_months",
        "target",
        "pearson_correlation",
        "r2",
        "rmse",
        "mae",
        "feature_count",
        "year_count",
        "cutoff_example",
        "output_dir",
    ]
    csv_lines = [",".join(csv_header)]
    for row in rows:
        csv_lines.append(",".join(str(row[name]) for name in csv_header))
    (output_dir / "leadtime_summary.csv").write_text("\n".join(csv_lines) + "\n")
    (output_dir / "leadtime_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
    )
    _save_skill_vs_lead_plots(output_dir, rows)


def _run_plot_script(output_dir: Path) -> None:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_pc_baseline_results.py"),
        "--input-dir",
        str(output_dir),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True, env=env)


def _save_skill_vs_lead_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("XDG_CACHE_HOME", "/tmp")
        os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
        os.environ.setdefault("XDG_CACHE_HOME", env["XDG_CACHE_HOME"])
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: could not create lead-time skill plots: {exc}", flush=True)
        return

    for target_name in ("pc1", "pc2"):
        target_rows = [row for row in rows if row["target"] == target_name]
        if not target_rows:
            continue
        target_rows.sort(key=lambda item: int(item["lead_months"]))
        lead = [int(row["lead_months"]) for row in target_rows]
        corr = [float(row["pearson_correlation"]) for row in target_rows]
        r2 = [float(row["r2"]) for row in target_rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(lead, corr, marker="o", label="Pearson r")
        ax.plot(lead, r2, marker="o", label="R2")
        ax.set_title(f"{target_name.upper()} skill vs lead time")
        ax.set_xlabel("Lead time (months)")
        ax.set_ylabel("Skill")
        ax.set_xticks(lead)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{target_name}_skill_vs_lead.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.lead_months < 0:
        raise ValueError(f"--lead-months must be >= 0, got {args.lead_months}")

    if args.lead_months_list is None:
        run_single_experiment(args, output_dir=args.output_dir, lead_months=args.lead_months)
        return

    unique_leads = sorted({int(value) for value in args.lead_months_list})
    if any(value < 0 for value in unique_leads):
        raise ValueError(f"--lead-months-list must contain only non-negative values, got {unique_leads}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    for lead_months in unique_leads:
        lead_dir = args.output_dir / f"lead_{lead_months}_month"
        print(f"=== RUN LEAD {lead_months} MONTH ===", flush=True)
        run_summaries.append(
            run_single_experiment(args, output_dir=lead_dir, lead_months=lead_months)
        )
    save_leadtime_summary(args.output_dir, run_summaries)
    print(f"saved lead-time summary outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
