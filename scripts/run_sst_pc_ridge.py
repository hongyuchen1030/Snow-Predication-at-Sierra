#!/usr/bin/env python3
"""
Run the SST-PC-only ridge experiment against WUS-D3 SWE PCs.
"""

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import DEFAULT_COARSEN_FACTOR, DEFAULT_MODEL_REGION
from snow_ml.data_wusd3 import (
    Wusd3Dataset,
    WUSD3_SWE_VARIABLE,
    default_wusd3_dataset,
    discover_wusd3_water_years,
    get_wusd3_grid_definition,
    variable_path_for_file_year,
)


SST_PC_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "model_sst_pcs"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_pc_ridge"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "leadtime_summary.csv"
REPORT_FILE = OUTPUT_DIR / "sst_pc_ridge_report.md"

LEADS = (1, 2, 3)
TARGET_PCS = ("PC1", "PC2")
FEATURE_COLUMNS = ("SST_PC1_mean", "SST_PC2_mean", "SST_PC3_mean", "SST_PC4_mean", "SST_PC5_mean")
TARGET_DATE_MM_DD = "04-01"
PCA_COMPONENTS = 2
RIDGE_ALPHA = 1.0


@dataclass(frozen=True)
class ModelTargetPca:
    years: List[int]
    scores: np.ndarray
    grid_shape: Tuple[int, int]
    valid_cell_count: int
    feature_mask: np.ndarray


def get_tmux_session_name() -> str:
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#S"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def discover_models() -> List[str]:
    models = []
    for path in sorted(SST_PC_DIR.glob("*_sst_pcs.csv")):
        models.append(path.stem.replace("_sst_pcs", ""))
    if not models:
        raise FileNotFoundError("No SST PC files found under {}".format(SST_PC_DIR))
    return models


def parse_iso_date(text: str) -> date:
    year_text, month_text, day_text = text.split("-")
    return date(int(year_text), int(month_text), int(day_text))


def cutoff_date_for_lead(target_year: int, lead_months: int) -> date:
    if lead_months == 1:
        return date(target_year, 3, 1) - timedelta(days=1)
    if lead_months == 2:
        return date(target_year, 2, 1) - timedelta(days=1)
    if lead_months == 3:
        return date(target_year, 1, 1) - timedelta(days=1)
    raise ValueError("Unsupported lead {}, expected one of {}".format(lead_months, LEADS))


def water_year_start(target_year: int) -> date:
    return date(target_year - 1, 10, 1)


def load_sst_pc_series(path: Path) -> Tuple[List[date], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("No rows found in {}".format(path))

    times = [parse_iso_date(row["time"]) for row in rows]
    values = np.array(
        [
            [float(row["SST_PC{}".format(mode)]) for mode in range(1, 6)]
            for row in rows
        ],
        dtype=np.float64,
    )
    return times, values


def build_sst_feature_table(times: Sequence[date], values: np.ndarray, lead_months: int) -> Dict[int, Dict[str, object]]:
    feature_rows = {}
    all_years = sorted({timestamp.year for timestamp in times} | {timestamp.year + 1 for timestamp in times if timestamp.month >= 10})
    for target_year in all_years:
        start_date = water_year_start(target_year)
        cutoff_date = cutoff_date_for_lead(target_year, lead_months)
        selected_indices = [
            index
            for index, timestamp in enumerate(times)
            if start_date <= timestamp <= cutoff_date
        ]
        if not selected_indices:
            continue
        selected_values = values[selected_indices]
        feature_vector = selected_values.mean(axis=0)
        if feature_vector.shape[0] != 5:
            raise ValueError("Expected exactly 5 SST features, got {}".format(feature_vector.shape[0]))
        feature_rows[target_year] = {
            "feature_vector": feature_vector,
            "months_used": [times[index].isoformat() for index in selected_indices],
            "start_date": start_date.isoformat(),
            "cutoff_date": cutoff_date.isoformat(),
        }
    return feature_rows


def fit_model_target_pca(dataset_id: str, years: Sequence[int]) -> ModelTargetPca:
    dataset = Wusd3Dataset(
        dataset_id=dataset_id,
        domain="d02",
        root_dir=default_wusd3_dataset().root_dir,
    )
    selected_years = sorted(int(year) for year in years)
    if len(selected_years) < PCA_COMPONENTS:
        raise ValueError("Need at least {} years for PCA, got {}".format(PCA_COMPONENTS, len(selected_years)))

    swe_grid = get_wusd3_grid_definition(
        dataset,
        water_year=selected_years[0],
        region=DEFAULT_MODEL_REGION,
        coarsen_factor=DEFAULT_COARSEN_FACTOR,
    )
    fields = []
    for water_year in selected_years:
        field = load_apr1_swe_map(dataset, water_year=water_year, swe_grid=swe_grid)
        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Expected 2D April 1 SWE map for {} WY{}, got {}".format(dataset_id, water_year, values.shape))
        fields.append(values)
        print(
            "loaded April 1 SWE model={} year={} shape={} finite_cells={}".format(
                dataset_id,
                water_year,
                values.shape,
                int(np.isfinite(values).sum()),
            ),
            flush=True,
        )

    cube = np.stack(fields, axis=0)
    full_matrix = cube.reshape(cube.shape[0], -1)
    valid_cell_mask = np.isfinite(full_matrix).all(axis=0)
    valid_cell_count = int(valid_cell_mask.sum())
    if valid_cell_count == 0:
        raise ValueError("No all-year finite SWE cells for {}".format(dataset_id))

    matrix = full_matrix[:, valid_cell_mask].astype(np.float64)
    mean_vector = matrix.mean(axis=0)
    std_vector = matrix.std(axis=0, ddof=0)
    std_vector = np.where(std_vector == 0.0, 1.0, std_vector)
    standardized = (matrix - mean_vector) / std_vector
    u_matrix, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    scores = (u_matrix[:, :PCA_COMPONENTS] * singular_values[:PCA_COMPONENTS]).astype(np.float64)

    print(
        "target PCA model={} years={} matrix_shape={} standardized_shape={} valid_cells={}".format(
            dataset_id,
            selected_years,
            tuple(full_matrix.shape),
            tuple(standardized.shape),
            valid_cell_count,
        ),
        flush=True,
    )
    return ModelTargetPca(
        years=selected_years,
        scores=scores,
        grid_shape=(int(cube.shape[1]), int(cube.shape[2])),
        valid_cell_count=valid_cell_count,
        feature_mask=valid_cell_mask,
    )


def load_apr1_swe_map(dataset: Wusd3Dataset, *, water_year: int, swe_grid) -> xr.DataArray:
    snapshot_text = "{}-{}".format(water_year, TARGET_DATE_MM_DD)
    path = variable_path_for_file_year(dataset, "swe", water_year - 1)
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        swe = ds[WUSD3_SWE_VARIABLE].sel(day=snapshot_text)
        if "day" in swe.dims:
            swe = swe.isel(day=0, drop=True)
        swe = swe.astype("float32")
        swe = swe.where(np.isfinite(swe))
        swe = swe.isel(lat2d=swe_grid.row_slice, lon2d=swe_grid.col_slice)
        swe = swe.isel(
            lat2d=slice(0, swe_grid.trimmed_shape[0]),
            lon2d=slice(0, swe_grid.trimmed_shape[1]),
        )
        swe = swe.where(swe_grid.region_mask)
        swe = swe.coarsen(
            lat2d=swe_grid.coarsen_factor,
            lon2d=swe_grid.coarsen_factor,
            boundary="trim",
        ).mean().load()
    swe.name = "wusd3_swe_{}".format(snapshot_text)
    return swe


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_leave_one_year_out(
    model_name: str,
    lead_months: int,
    years: Sequence[int],
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
) -> List[Dict[str, object]]:
    if x_matrix.shape[1] != 5:
        raise ValueError("Expected exactly 5 features, got {}".format(x_matrix.shape[1]))
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError("Feature/target row mismatch: {} vs {}".format(x_matrix.shape[0], y_matrix.shape[0]))

    rows = []
    for target_index, target_name in enumerate(TARGET_PCS):
        y_values = y_matrix[:, target_index]
        for test_index, water_year in enumerate(years):
            train_mask = np.arange(len(years)) != test_index
            x_train = x_matrix[train_mask]
            x_test = x_matrix[test_index : test_index + 1]
            y_train = y_values[train_mask]
            x_mean = x_train.mean(axis=0)
            x_std = x_train.std(axis=0, ddof=0)
            x_std = np.where(x_std == 0.0, 1.0, x_std)
            x_train_scaled = (x_train - x_mean) / x_std
            x_test_scaled = (x_test - x_mean) / x_std
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(x_train_scaled, y_train)
            y_pred = float(model.predict(x_test_scaled)[0])
            rows.append(
                {
                    "model": model_name,
                    "lead_months": int(lead_months),
                    "target_pc": target_name,
                    "year": int(water_year),
                    "y_true": float(y_values[test_index]),
                    "y_pred": y_pred,
                }
            )
            print(
                "loyo model={} lead={} target={} year={} true={:.6g} pred={:.6g}".format(
                    model_name,
                    lead_months,
                    target_name,
                    int(water_year),
                    float(y_values[test_index]),
                    y_pred,
                ),
                flush=True,
            )
    return rows


def summarize_predictions(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows = []
    grouping_keys = []
    for row in rows:
        grouping_keys.append((str(row["model"]), int(row["lead_months"]), str(row["target_pc"])))
        grouping_keys.append(("ALL", int(row["lead_months"]), str(row["target_pc"])))

    for model_name, lead_months, target_pc in sorted(set(grouping_keys)):
        selected = [
            row for row in rows
            if int(row["lead_months"]) == lead_months
            and str(row["target_pc"]) == target_pc
            and (model_name == "ALL" or str(row["model"]) == model_name)
        ]
        y_true = np.array([float(row["y_true"]) for row in selected], dtype=np.float64)
        y_pred = np.array([float(row["y_pred"]) for row in selected], dtype=np.float64)
        summary_rows.append(
            {
                "model": model_name,
                "lead_months": lead_months,
                "target_pc": target_pc,
                "n_predictions": int(y_true.shape[0]),
                "correlation": correlation(y_true, y_pred),
                "r2": r2_score_manual(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
            }
        )
    return summary_rows


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "lead_months", "target_pc", "year", "y_true", "y_pred"])
        for row in rows:
            writer.writerow([
                row["model"],
                row["lead_months"],
                row["target_pc"],
                row["year"],
                "{:.12g}".format(float(row["y_true"])),
                "{:.12g}".format(float(row["y_pred"])),
            ])


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "lead_months", "target_pc", "n_predictions", "correlation", "r2", "rmse", "mae"])
        for row in rows:
            writer.writerow([
                row["model"],
                row["lead_months"],
                row["target_pc"],
                row["n_predictions"],
                "{:.12g}".format(float(row["correlation"])) if np.isfinite(float(row["correlation"])) else "nan",
                "{:.12g}".format(float(row["r2"])) if np.isfinite(float(row["r2"])) else "nan",
                "{:.12g}".format(float(row["rmse"])),
                "{:.12g}".format(float(row["mae"])),
            ])


def write_report(
    path: Path,
    runtime: Dict[str, str],
    year_counts: Sequence[Dict[str, object]],
    summary_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# SST-PC Ridge Experiment",
        "",
        "## Runtime",
        "",
        "- tmux session: `{}`".format(runtime["tmux_session"] or "not detected"),
        "- hostname: `{}`".format(runtime["hostname"]),
        "- Slurm job ID: `{}`".format(runtime["slurm_job_id"]),
        "",
        "## Setup",
        "",
        "- predictors: water-year-to-date means of SST_PC1..SST_PC5",
        "- target date: April 1",
        "- lead cutoffs: 1 month = Feb end, 2 months = Jan end, 3 months = Dec end",
        "- validation: leave-one-year-out, re-fit ridge every held-out year",
        "",
        "## Years Used",
        "",
        "| Model | SWE Years | SST Years (lead 1) | Final Years (lead 1) | Final Years (lead 2) | Final Years (lead 3) |",
        "|-------|-----------|--------------------|----------------------|----------------------|----------------------|",
    ]

    for row in year_counts:
        lines.append(
            "| {model} | {swe_years} | {sst_years_lead1} | {final_years_lead1} | {final_years_lead2} | {final_years_lead3} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Pooled Metrics",
            "",
            "| Lead | Target | N | Correlation | R2 | RMSE | MAE |",
            "|------|--------|---|-------------|----|------|-----|",
        ]
    )
    pooled_rows = [row for row in summary_rows if row["model"] == "ALL"]
    for row in sorted(pooled_rows, key=lambda item: (int(item["lead_months"]), str(item["target_pc"]))):
        lines.append(
            "| {lead_months} | {target_pc} | {n_predictions} | {correlation:.6f} | {r2:.6f} | {rmse:.6f} | {mae:.6f} |".format(
                **row
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runtime = {
        "tmux_session": get_tmux_session_name(),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    if not runtime["slurm_job_id"] or "nid" not in runtime["hostname"]:
        raise RuntimeError("Do not run this script on a login node; active interactive compute allocation required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models()
    all_prediction_rows = []
    year_count_rows = []

    print("models discovered from SST PCs: {}".format(models), flush=True)
    for model_name in models:
        dataset = Wusd3Dataset(
            dataset_id=model_name,
            domain="d02",
            root_dir=default_wusd3_dataset().root_dir,
        )
        swe_years = sorted(discover_wusd3_water_years(dataset))
        sst_times, sst_values = load_sst_pc_series(SST_PC_DIR / "{}_sst_pcs.csv".format(model_name))

        lead_feature_tables = {lead: build_sst_feature_table(sst_times, sst_values, lead) for lead in LEADS}
        all_final_years = sorted(set(swe_years) & set(lead_feature_tables[1].keys()) & set(lead_feature_tables[2].keys()) & set(lead_feature_tables[3].keys()))
        if len(all_final_years) < PCA_COMPONENTS:
            raise RuntimeError("Not enough overlapping years for {}: {}".format(model_name, all_final_years))

        target_pca = fit_model_target_pca(model_name, all_final_years)
        year_to_target = {
            year: target_pca.scores[index, :2]
            for index, year in enumerate(target_pca.years)
        }

        year_count_row = {
            "model": model_name,
            "swe_years": len(swe_years),
            "sst_years_lead1": len(lead_feature_tables[1]),
            "final_years_lead1": 0,
            "final_years_lead2": 0,
            "final_years_lead3": 0,
        }

        for lead_months in LEADS:
            matched_sst_years = sorted(lead_feature_tables[lead_months].keys())
            matched_swe_years = sorted(year for year in swe_years if year in year_to_target)
            final_years = sorted(year for year in matched_swe_years if year in lead_feature_tables[lead_months])
            if len(final_years) < 2:
                raise RuntimeError(
                    "Need at least 2 final years for LOYO model={} lead={}, got {}".format(
                        model_name,
                        lead_months,
                        final_years,
                    )
                )

            x_matrix = np.array(
                [lead_feature_tables[lead_months][year]["feature_vector"] for year in final_years],
                dtype=np.float64,
            )
            y_matrix = np.array([year_to_target[year] for year in final_years], dtype=np.float64)
            if x_matrix.shape[1] != 5:
                raise ValueError("Expected exactly 5 features for {} lead {}, got {}".format(model_name, lead_months, x_matrix.shape[1]))

            year_count_row["final_years_lead{}".format(lead_months)] = len(final_years)

            print("=" * 80, flush=True)
            print("model={} lead={} target_date={}".format(model_name, lead_months, TARGET_DATE_MM_DD), flush=True)
            print("available years: {}".format(swe_years), flush=True)
            print("matched SST years: {}".format(matched_sst_years), flush=True)
            print("matched SWE years: {}".format(matched_swe_years), flush=True)
            print("final year count: {}".format(len(final_years)), flush=True)
            print("final years: {}".format(final_years), flush=True)
            print("feature matrix shape: {}".format(tuple(x_matrix.shape)), flush=True)
            print("target shape: {}".format(tuple(y_matrix.shape)), flush=True)
            print(
                "example months used for first year {}: {}".format(
                    final_years[0],
                    lead_feature_tables[lead_months][final_years[0]]["months_used"],
                ),
                flush=True,
            )

            all_prediction_rows.extend(
                run_leave_one_year_out(
                    model_name=model_name,
                    lead_months=lead_months,
                    years=final_years,
                    x_matrix=x_matrix,
                    y_matrix=y_matrix,
                )
            )

        year_count_rows.append(year_count_row)

    summary_rows = summarize_predictions(all_prediction_rows)
    write_predictions_csv(PREDICTIONS_FILE, all_prediction_rows)
    write_summary_csv(SUMMARY_FILE, summary_rows)
    write_report(REPORT_FILE, runtime, year_count_rows, summary_rows)

    print("saved predictions: {}".format(PREDICTIONS_FILE), flush=True)
    print("saved summary: {}".format(SUMMARY_FILE), flush=True)
    print("saved report: {}".format(REPORT_FILE), flush=True)


if __name__ == "__main__":
    main()
