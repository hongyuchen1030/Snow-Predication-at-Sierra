from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import DEFAULT_COARSEN_FACTOR, DEFAULT_MODEL_REGION, SST_MONTHLY_MEAN_PATH, region_to_dict
from snow_ml.data_wusd3 import (
    DEFAULT_WUSD3_DATASET_ID,
    DEFAULT_WUSD3_DOMAIN,
    Wusd3Dataset,
    default_wusd3_dataset,
    discover_wusd3_dataset_ids,
    discover_wusd3_water_years,
    get_wusd3_grid_definition,
    load_wusd3_snapshot,
    load_wusd3_target_swe_map,
)
from snow_ml.pc_baseline import (
    PREDICTOR_PCA_OPTIONS,
    RIDGE_ALPHAS,
    align_features_and_targets,
    build_precip_features,
    build_sst_features,
    build_temperature_features,
    fit_swe_pca,
    predictor_cutoff_date,
    run_loyo,
    save_aligned_dataset_csv,
    save_metrics,
    save_predictions_csv,
    save_target_metadata,
    save_target_pca_npz,
    save_targets_csv,
)


STRICT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "pc_leadtime_wusd3_strict"
STRICT_REPORT_PATH = STRICT_OUTPUT_DIR / "strict_comparability_report.json"


def parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, got %r" % text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict WUS-D3 version of the UCLA PC lead-time baseline."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_WUSD3_DATASET_ID,
        help="Historical WUS-D3 member id under the daily root.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_WUSD3_DOMAIN,
        help="WUS-D3 domain. Strict baseline expects d02.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Optional water years override. If omitted, use all discovered WUS-D3 years.",
    )
    parser.add_argument(
        "--lead-months-list",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Lead months to run. Strict baseline expects 1 2 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STRICT_OUTPUT_DIR,
        help="Output directory for the strict WUS-D3 lead-time baseline.",
    )
    parser.add_argument(
        "--save-plots",
        type=parse_bool,
        default=True,
        help="Create lead-time skill plots and per-lead plots like the UCLA baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Wusd3Dataset(
        dataset_id=args.dataset_id,
        domain=args.domain,
        root_dir=default_wusd3_dataset().root_dir,
    )
    if dataset.domain != "d02":
        raise ValueError("Strict WUS-D3 baseline requires domain d02, got %r." % dataset.domain)
    available_dataset_ids = discover_wusd3_dataset_ids(root_dir=dataset.root_dir)
    if dataset.dataset_id not in available_dataset_ids:
        raise ValueError(
            "dataset_id %r not found. Available ids include: %s"
            % (dataset.dataset_id, available_dataset_ids)
        )
    if dataset.dataset_id.endswith("ssp370_bc"):
        raise ValueError("Strict WUS-D3 baseline requires a historical member, got %r." % dataset.dataset_id)

    requested_leads = sorted({int(value) for value in args.lead_months_list})
    if requested_leads != [1, 2, 3]:
        raise ValueError("Strict WUS-D3 baseline requires lead-months-list 1 2 3, got %s." % requested_leads)

    candidate_years = sorted(args.years if args.years is not None else discover_wusd3_water_years(dataset))
    if not candidate_years:
        raise RuntimeError("No candidate WUS-D3 water years found.")

    print("WUS-D3 strict dataset_id: %s" % dataset.dataset_id, flush=True)
    print("WUS-D3 strict domain: %s" % dataset.domain, flush=True)
    print("candidate WUS-D3 water years: %s" % candidate_years, flush=True)
    print("strict region: %s" % region_to_dict(DEFAULT_MODEL_REGION), flush=True)
    print("strict coarsen factor: %s" % DEFAULT_COARSEN_FACTOR, flush=True)

    grid = get_wusd3_grid_definition(
        dataset,
        water_year=candidate_years[0],
        region=DEFAULT_MODEL_REGION,
        coarsen_factor=DEFAULT_COARSEN_FACTOR,
    )
    print("effective WUS-D3 grid region: %s" % region_to_dict(grid.effective_region), flush=True)
    print("coarsened WUS-D3 grid shape: %s" % (grid.grid_shape,), flush=True)

    target_maps, target_drops = load_apr1_wusd3_maps(dataset, candidate_years, grid=grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    strict_report: dict[str, Any] = {
        "wusd3_member_used": dataset.dataset_id,
        "years_included": candidate_years,
        "target_variable": "snow",
        "target_pca_method": "April 1 WUS-D3 maps, mean-centered only, PCA(n_components=2), valid cells finite in every usable year.",
        "target_pca_global_or_fold_local": "global",
        "sst_included": True,
        "sst_source_file": str(SST_MONTHLY_MEAN_PATH),
        "era5_land_included": True,
        "era5_variables_used": ["t2m", "tp"],
        "ridge_alpha_tuned": True,
        "alpha_grid": list(RIDGE_ALPHAS),
        "predictor_pca_enabled": True,
        "predictor_pca_options": list(PREDICTOR_PCA_OPTIONS),
        "region": region_to_dict(DEFAULT_MODEL_REGION),
        "coarsen_factor": int(DEFAULT_COARSEN_FACTOR),
        "feature_list_per_lead": {},
        "feature_count_per_lead": {},
        "aligned_years": {},
        "missing_or_failed_feature_groups": {
            "target_drops": target_drops,
            "preliminary_feature_drops_for_pca_alignment": {},
        },
    }

    for lead_months in requested_leads:
        lead_dir = args.output_dir / ("lead_%d_month" % lead_months)
        print("=== STRICT RUN LEAD %d MONTH ===" % lead_months, flush=True)
        summary, lead_failures = run_single_experiment(
            dataset,
            grid=grid,
            target_maps=target_maps,
            output_dir=lead_dir,
            lead_months=lead_months,
            save_plots=args.save_plots,
        )
        strict_report["feature_list_per_lead"]["lead_%d_month" % lead_months] = summary["feature_columns"]
        strict_report["feature_count_per_lead"]["lead_%d_month" % lead_months] = len(summary["feature_columns"])
        strict_report["aligned_years"]["lead_%d_month" % lead_months] = summary["aligned_years"]
        strict_report["missing_or_failed_feature_groups"]["lead_%d_month" % lead_months] = lead_failures
        _validate_strict_feature_count(lead_months, len(summary["feature_columns"]))
        run_summaries.append(summary)

    save_leadtime_summary(args.output_dir, run_summaries)
    STRICT_REPORT_PATH.write_text(json.dumps(strict_report, indent=2, sort_keys=True) + "\n")
    print("saved strict comparability report to %s" % STRICT_REPORT_PATH, flush=True)


def load_apr1_wusd3_maps(
    dataset: Wusd3Dataset,
    years: list[int],
    *,
    grid,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    maps = {}
    dropped = {}
    for water_year in sorted(years):
        try:
            field = load_wusd3_target_swe_map(
                dataset,
                water_year=water_year,
                swe_grid=grid,
                fill_missing=False,
            )
            values = np.asarray(field.values, dtype=np.float32)
            if not np.isfinite(values).any():
                dropped[water_year] = "April 1 WUS-D3 target map has no finite cells"
                continue
            maps[water_year] = values
            print(
                "target map WY%d: shape=%s finite_cells=%d"
                % (water_year, tuple(values.shape), int(np.isfinite(values).sum())),
                flush=True,
            )
        except Exception as exc:
            dropped[water_year] = "April 1 WUS-D3 load failed: %s" % exc
            print("drop target WY%d: %s" % (water_year, dropped[water_year]), flush=True)
    return maps, dropped


def build_wusd3_dec31_features(
    dataset: Wusd3Dataset,
    water_year: int,
    *,
    grid,
) -> dict[str, float]:
    field = load_wusd3_snapshot(
        dataset,
        water_year=water_year,
        snapshot_date=date(water_year - 1, 12, 31),
        swe_grid=grid,
        fill_missing=False,
    )
    values = np.asarray(field.values, dtype=np.float32)
    if not np.isfinite(values).any():
        raise ValueError("No finite Dec 31 WUS-D3 SWE values for WY%d" % water_year)
    return {
        "dec31_swe_mean": float(np.nanmean(values)),
        "dec31_swe_std": float(np.nanstd(values)),
    }


def assemble_wusd3_feature_table(
    dataset: Wusd3Dataset,
    years: list[int],
    *,
    grid,
    lead_months: int,
) -> tuple[list[dict[str, float]], dict[int, str]]:
    rows = []
    dropped = {}
    for water_year in sorted(years):
        print("build strict WUS-D3 feature row WY%d" % water_year, flush=True)
        try:
            cutoff_date = predictor_cutoff_date(water_year, lead_months)
            row = {"water_year": float(water_year)}
            row.update(build_wusd3_dec31_features(dataset, water_year, grid=grid))
            row.update(
                build_temperature_features(
                    water_year,
                    region=DEFAULT_MODEL_REGION,
                    include_seasonal_features=True,
                    cutoff_date=cutoff_date,
                )
            )
            row.update(
                build_precip_features(
                    water_year,
                    region=DEFAULT_MODEL_REGION,
                    include_seasonal_features=True,
                    cutoff_date=cutoff_date,
                )
            )
            row.update(
                build_sst_features(
                    water_year,
                    region=DEFAULT_MODEL_REGION,
                    cutoff_date=cutoff_date,
                )
            )
            bad_columns = [
                name for name, value in row.items()
                if name != "water_year" and not np.isfinite(float(value))
            ]
            if bad_columns:
                dropped[water_year] = "non-finite features: %s" % bad_columns
                continue
            rows.append(row)
        except Exception as exc:
            dropped[water_year] = str(exc)
            print("drop strict feature WY%d: %s" % (water_year, exc), flush=True)
    return rows, dropped


def run_single_experiment(
    dataset: Wusd3Dataset,
    *,
    grid,
    target_maps: dict[int, np.ndarray],
    output_dir: Path,
    lead_months: int,
    save_plots: bool,
) -> tuple[dict[str, Any], dict[int, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preliminary_feature_rows, preliminary_feature_drops = assemble_wusd3_feature_table(
        dataset,
        sorted(target_maps),
        grid=grid,
        lead_months=lead_months,
    )
    preliminary_feature_years = [int(row["water_year"]) for row in preliminary_feature_rows]
    usable_for_pca = sorted(set(target_maps).intersection(preliminary_feature_years))
    if not usable_for_pca:
        raise RuntimeError(
            "No overlapping WUS-D3 target years and preliminary feature years remain for lead %d." % lead_months
        )
    target_pca = fit_swe_pca(
        {year: target_maps[year] for year in usable_for_pca},
        n_components=2,
        grid=grid,
    )
    usable_years = set(target_pca.years)
    feature_rows = [
        row for row in preliminary_feature_rows
        if int(row["water_year"]) in usable_years
    ]
    feature_drops = dict(preliminary_feature_drops)
    for year in sorted(target_maps):
        if year not in usable_for_pca:
            feature_drops.setdefault(year, "missing target or preliminary required predictors")
    aligned_rows = align_features_and_targets(
        feature_rows,
        target_pca,
        predict_pc2=True,
    )
    if not aligned_rows:
        raise RuntimeError("No aligned strict WUS-D3 feature/target rows remain.")

    target_columns = ["pc1", "pc2"]
    feature_columns = [
        name for name in aligned_rows[0]
        if name not in {"water_year", "pc1", "pc2"}
    ]
    aligned_years = [int(row["water_year"]) for row in aligned_rows]

    save_targets_csv(output_dir / "targets.csv", target_pca, predict_pc2=True)
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
        enable_predictor_pca=True,
    )
    save_predictions_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        target_columns=target_columns,
    )
    save_metrics(output_dir / "metrics.json", metrics)

    run_metadata = {
        "dataset_id": dataset.dataset_id,
        "domain": dataset.domain,
        "lead_months": int(lead_months),
        "predictor_cutoff_dates_by_year": {
            str(year): predictor_cutoff_date(int(year), lead_months).isoformat()
            for year in aligned_years
        },
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "aligned_years": aligned_years,
        "predict_pc2": True,
        "dec31_mode": "stats",
        "include_seasonal_features": True,
        "enable_predictor_pca": True,
        "region": region_to_dict(DEFAULT_MODEL_REGION),
        "coarsen_factor": int(DEFAULT_COARSEN_FACTOR),
        "target_pca_years": target_pca.years,
        "strict_reference": "UCLA pc_leadtime_baseline audit behavior",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )
    if save_plots:
        _run_plot_script(output_dir)
    print("saved strict WUS-D3 outputs to %s" % output_dir, flush=True)
    print("final strict metrics: %s" % metrics, flush=True)
    return (
        {
            "lead_months": int(lead_months),
            "output_dir": str(output_dir),
            "metrics": metrics,
            "feature_columns": feature_columns,
            "aligned_years": aligned_years,
            "cutoff_example": predictor_cutoff_date(aligned_years[0], lead_months).isoformat(),
        },
        feature_drops,
    )


def save_leadtime_summary(
    output_dir: Path,
    run_summaries: list[dict[str, Any]],
) -> None:
    rows = []
    summary_json = {"runs": run_summaries, "targets": {}}
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
        print("warning: could not create strict lead-time skill plots: %s" % exc, flush=True)
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
        ax.set_title("%s skill vs lead time" % target_name.upper())
        ax.set_xlabel("Lead time (months)")
        ax.set_ylabel("Skill")
        ax.set_xticks(lead)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / ("%s_skill_vs_lead.png" % target_name), dpi=150)
        plt.close(fig)


def _validate_strict_feature_count(lead_months: int, feature_count: int) -> None:
    expected = 16 if lead_months in (1, 2) else 13
    if feature_count != expected:
        raise RuntimeError(
            "Strict WUS-D3 feature count mismatch for lead %d: expected %d, got %d."
            % (lead_months, expected, feature_count)
        )


if __name__ == "__main__":
    main()
