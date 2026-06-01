#!/usr/bin/env python3
"""
Run April-1-only short-lead leave-one-water-year-out ridge regression using
existing SST PCs and daily WUS-D3 d01 SWE PCs.
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data_wusd3 import Wusd3Dataset, discover_wusd3_file_years, variable_path_for_file_year


SST_PC_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "model_sst_pcs"
WRFINPUT_D01 = Path("/global/cfs/projectdirs/m3522/cmip6/WUS-D3/wrfinput_d01")
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "short_lead_pc_predictability"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_FILE = OUTPUT_DIR / "short_lead_pc_predictability_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "short_lead_pc_predictions.csv"
REPORT_FILE = OUTPUT_DIR / "short_lead_pc_predictability_report.md"
LEAD_DAYS = (1, 2, 3)
RIDGE_ALPHA = 1.0
SWE_N_COMPONENTS = 2
APRIL_TARGET_MONTH = 4
APRIL_TARGET_DAY = 1
MARCH_SST_MONTH = 3
MARCH_SST_DAY = 31


@dataclass(frozen=True)
class SwePcSeries:
    dates: List[date]
    pcs: np.ndarray
    valid_land_cell_count: int
    land_cell_count: int
    singular_values: np.ndarray


def parse_iso_date(text: str) -> date:
    year_text, month_text, day_text = text.split("-")
    return date(int(year_text), int(month_text), int(day_text))


def water_year_for_date(timestamp: date) -> int:
    if timestamp.month >= 10:
        return timestamp.year + 1
    return timestamp.year


def discover_models() -> List[str]:
    models = []
    for path in sorted(SST_PC_DIR.glob("*_sst_pcs.csv")):
        models.append(path.stem.replace("_sst_pcs", ""))
    if not models:
        raise FileNotFoundError(f"No SST PC files found under {SST_PC_DIR}")
    return models


def load_wrf_landmask() -> np.ndarray:
    ds = xr.open_dataset(WRFINPUT_D01, engine="netcdf4")
    try:
        if "LANDMASK" in ds:
            mask = np.asarray(ds["LANDMASK"].isel(Time=0).values, dtype=np.float32)
            land_mask = mask == 1.0
            source_name = "LANDMASK"
        elif "XLAND" in ds:
            mask = np.asarray(ds["XLAND"].isel(Time=0).values, dtype=np.float32)
            land_mask = mask == 1.0
            source_name = "XLAND"
        else:
            raise KeyError("Expected LANDMASK or XLAND in wrfinput_d01")
    finally:
        ds.close()

    print(
        f"Using {source_name} for d01 land mask: {int(land_mask.sum())} land cells, "
        f"{int(land_mask.size - land_mask.sum())} non-land cells",
        flush=True,
    )
    return land_mask


def load_sst_pc_series(path: Path) -> Tuple[List[date], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    dates = [parse_iso_date(row["time"]) for row in rows]
    values = np.array(
        [[float(row["SST_PC1"]), float(row["SST_PC2"])] for row in rows],
        dtype=np.float64,
    )
    return dates, values


def load_daily_snow_matrix(model_name: str) -> Tuple[List[date], np.ndarray]:
    dataset = Wusd3Dataset(
        dataset_id=model_name,
        domain="d01",
        root_dir=Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3"),
    )
    file_years = discover_wusd3_file_years(dataset, variable_key="swe")
    if not file_years:
        raise FileNotFoundError(f"No d01 snow files found for {model_name}")

    all_dates: List[date] = []
    all_fields: List[np.ndarray] = []
    for file_year in file_years:
        path = variable_path_for_file_year(dataset, "swe", file_year)
        print(f"Loading d01 snow file: {path}", flush=True)
        with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
            if "snow" not in ds.data_vars:
                raise ValueError(f"Expected 'snow' in {path}, got {list(ds.data_vars)}")
            day_values = np.asarray(ds["day"].values)
            snow_values = np.asarray(ds["snow"].values, dtype=np.float32)
        if snow_values.ndim != 3:
            raise ValueError(f"Expected snow values shape (day, lat2d, lon2d), got {snow_values.shape}")
        file_dates = [parse_iso_date(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D")) for value in day_values]
        all_dates.extend(file_dates)
        all_fields.append(snow_values)

    cube = np.concatenate(all_fields, axis=0).astype(np.float32)
    print(f"Combined snow cube for {model_name}: {cube.shape}", flush=True)
    return all_dates, cube


def compute_daily_swe_pcs(model_name: str, land_mask: np.ndarray) -> SwePcSeries:
    dates, cube = load_daily_snow_matrix(model_name)
    if cube.shape[1:] != land_mask.shape:
        raise ValueError(f"Snow grid {cube.shape[1:]} does not match land mask {land_mask.shape}")

    flattened = cube.reshape(cube.shape[0], -1)
    valid_mask = land_mask.reshape(-1) & np.isfinite(flattened).all(axis=0)
    valid_land_cell_count = int(valid_mask.sum())
    if valid_land_cell_count < SWE_N_COMPONENTS:
        raise ValueError(
            f"Need at least {SWE_N_COMPONENTS} valid land cells for {model_name}, got {valid_land_cell_count}"
        )

    x_matrix = flattened[:, valid_mask].astype(np.float32, copy=False)
    print(
        f"SWE PCA matrix for {model_name}: {x_matrix.shape}, valid_land_cells={valid_land_cell_count}",
        flush=True,
    )
    u_matrix, singular_values, _ = randomized_svd(
        x_matrix,
        n_components=SWE_N_COMPONENTS,
        n_iter=5,
        random_state=0,
    )
    pcs = (u_matrix * singular_values[np.newaxis, :]).astype(np.float64)
    print(
        f"SWE PCs for {model_name}: pcs_shape={pcs.shape}, singular_values="
        f"{[float(value) for value in singular_values]}",
        flush=True,
    )
    return SwePcSeries(
        dates=dates,
        pcs=pcs,
        valid_land_cell_count=valid_land_cell_count,
        land_cell_count=int(land_mask.sum()),
        singular_values=singular_values.astype(np.float64),
    )


def build_samples(
    model_name: str,
    sst_dates: Sequence[date],
    sst_values: np.ndarray,
    swe_series: SwePcSeries,
    lead_days: int,
) -> List[Dict[str, object]]:
    swe_by_date = {swe_date: swe_series.pcs[index] for index, swe_date in enumerate(swe_series.dates)}
    sst_by_date = {sst_date: sst_values[index] for index, sst_date in enumerate(sst_dates)}

    candidate_years = sorted(
        {
            swe_date.year
            for swe_date in swe_series.dates
            if swe_date.month == APRIL_TARGET_MONTH and swe_date.day == APRIL_TARGET_DAY
        }
    )

    samples: List[Dict[str, object]] = []
    for target_year in candidate_years:
        target_date = date(target_year, APRIL_TARGET_MONTH, APRIL_TARGET_DAY)
        sst_date = date(target_year, MARCH_SST_MONTH, MARCH_SST_DAY)
        swe_feature_date = target_date - timedelta(days=lead_days)
        if sst_date not in sst_by_date:
            continue
        if swe_feature_date not in swe_by_date:
            continue
        if target_date not in swe_by_date:
            continue
        swe_now = swe_by_date[swe_feature_date]
        swe_target = swe_by_date[target_date]
        sst_now = sst_by_date[sst_date]
        feature_vector = np.array(
            [
                float(sst_now[0]),
                float(sst_now[1]),
                float(swe_now[0]),
                float(swe_now[1]),
            ],
            dtype=np.float64,
        )
        samples.append(
            {
                "model": model_name,
                "date": swe_feature_date,
                "sst_date": sst_date,
                "target_date": target_date,
                "water_year": water_year_for_date(target_date),
                "features": feature_vector,
                "target_value": float(swe_target[0]),
            }
        )

    if not samples:
        raise RuntimeError(f"No aligned samples found for model={model_name} lead_days={lead_days}")
    print(
        f"April-1 samples for model={model_name} lead_days={lead_days}: {len(samples)} "
        f"water_years={sorted(int(sample['water_year']) for sample in samples)[:5]}...",
        flush=True,
    )
    return samples


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


def run_leave_one_water_year_out(samples: Sequence[Dict[str, object]], lead_days: int) -> List[Dict[str, object]]:
    water_years = sorted({int(sample["water_year"]) for sample in samples})
    rows: List[Dict[str, object]] = []
    for test_year in water_years:
        train_samples = [sample for sample in samples if int(sample["water_year"]) != test_year]
        test_samples = [sample for sample in samples if int(sample["water_year"]) == test_year]
        if not train_samples or not test_samples:
            continue

        x_train = np.stack([sample["features"] for sample in train_samples], axis=0)
        y_train = np.array([float(sample["target_value"]) for sample in train_samples], dtype=np.float64)
        x_test = np.stack([sample["features"] for sample in test_samples], axis=0)

        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0, ddof=0)
        x_std = np.where(x_std == 0.0, 1.0, x_std)

        x_train_scaled = (x_train - x_mean) / x_std
        x_test_scaled = (x_test - x_mean) / x_std

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        for sample, pred_value in zip(test_samples, y_pred):
            rows.append(
                {
                    "model": sample["model"],
                    "lead_days": int(lead_days),
                    "water_year": int(sample["water_year"]),
                    "date": sample["date"].isoformat(),
                    "sst_date": sample["sst_date"].isoformat(),
                    "target_date": sample["target_date"].isoformat(),
                    "SST_PC1": float(sample["features"][0]),
                    "SST_PC2": float(sample["features"][1]),
                    "SWE_PC1": float(sample["features"][2]),
                    "SWE_PC2": float(sample["features"][3]),
                    "true_SWE_PC1_target": float(sample["target_value"]),
                    "predicted_SWE_PC1_target": float(pred_value),
                }
            )
        print(
            f"LOYO model={test_samples[0]['model']} lead_days={lead_days} "
            f"held_out_water_year={test_year} n_test={len(test_samples)}",
            flush=True,
        )
    return rows


def summarize_predictions(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    model_names = sorted({str(row["model"]) for row in rows})
    for model_name in model_names:
        for lead_days in LEAD_DAYS:
            selected = [
                row for row in rows
                if str(row["model"]) == model_name and int(row["lead_days"]) == lead_days
            ]
            if not selected:
                continue
            y_true = np.array([float(row["true_SWE_PC1_target"]) for row in selected], dtype=np.float64)
            y_pred = np.array([float(row["predicted_SWE_PC1_target"]) for row in selected], dtype=np.float64)
            summary_rows.append(
                {
                    "model": model_name,
                    "lead_days": int(lead_days),
                    "n_predictions": int(len(selected)),
                    "n_water_years": int(len({int(row['water_year']) for row in selected})),
                    "correlation": correlation(y_true, y_pred),
                    "rmse": rmse(y_true, y_pred),
                    "r2": r2_score_manual(y_true, y_pred),
                }
            )
    return summary_rows


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "lead_days",
                "water_year",
                "date",
                "sst_date",
                "target_date",
                "SST_PC1",
                "SST_PC2",
                "SWE_PC1",
                "SWE_PC2",
                "true_SWE_PC1_target",
                "predicted_SWE_PC1_target",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["lead_days"],
                    row["water_year"],
                    row["date"],
                    row["sst_date"],
                    row["target_date"],
                    "{:.12g}".format(float(row["SST_PC1"])),
                    "{:.12g}".format(float(row["SST_PC2"])),
                    "{:.12g}".format(float(row["SWE_PC1"])),
                    "{:.12g}".format(float(row["SWE_PC2"])),
                    "{:.12g}".format(float(row["true_SWE_PC1_target"])),
                    "{:.12g}".format(float(row["predicted_SWE_PC1_target"])),
                ]
            )


def write_results_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "lead_days", "n_predictions", "n_water_years", "correlation", "rmse", "r2"])
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["lead_days"],
                    row["n_predictions"],
                    row["n_water_years"],
                    "{:.12g}".format(float(row["correlation"])) if np.isfinite(float(row["correlation"])) else "nan",
                    "{:.12g}".format(float(row["rmse"])),
                    "{:.12g}".format(float(row["r2"])) if np.isfinite(float(row["r2"])) else "nan",
                ]
            )


def plot_predictions(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["date"])
    dates = [parse_iso_date(str(row["date"])) for row in sorted_rows]
    y_true = np.array([float(row["true_SWE_PC1_target"]) for row in sorted_rows], dtype=np.float64)
    y_pred = np.array([float(row["predicted_SWE_PC1_target"]) for row in sorted_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, y_true, label="True SWE_PC1(April 1)", linewidth=1.5)
    ax.plot(dates, y_pred, label="Predicted SWE_PC1(April 1)", linewidth=1.2)
    ax.set_xlabel("Feature date")
    ax.set_ylabel("SWE PC1")
    ax.set_title(output_path.stem.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary(results_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = sorted({str(row["model"]) for row in results_rows})
    for model_name in model_names:
        selected = sorted(
            [row for row in results_rows if str(row["model"]) == model_name],
            key=lambda row: int(row["lead_days"]),
        )
        ax.plot(
            [int(row["lead_days"]) for row in selected],
            [float(row["correlation"]) for row in selected],
            marker="o",
            label=model_name,
        )
    ax.set_xlabel("Lead day")
    ax.set_ylabel("Correlation")
    ax.set_title("Short-lead SWE PC1 skill by model")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(
    path: Path,
    runtime: Dict[str, str],
    sst_paths: Dict[str, str],
    swe_meta: Dict[str, Dict[str, object]],
    results_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Short-Lead PC Predictability",
        "",
        "## Runtime",
        "",
        f"- hostname: `{runtime['hostname']}`",
        f"- Slurm job ID: `{runtime['slurm_job_id']}`",
        "",
        "## Setup",
        "",
        "- existing SST PC files were reused from `artifacts/sst_pca/model_sst_pcs/`",
        "- SWE PCs were computed from WUS-D3 `d01` `snow` using the WRF `LANDMASK` land cells only",
        "- SWE PC method: uncentered randomized SVD on the daily land-only matrix",
        "- one sample per water year, targeting April 1 only",
        "- SST predictors use the March monthly SST PCs dated March 31 for all three leads",
        "- SWE predictors use March 31, March 30, and March 29 for leads 1, 2, and 3 respectively",
        "- predictor features: `SST_PC1(March 31)`, `SST_PC2(March 31)`, `SWE_PC1(t)`, `SWE_PC2(t)`",
        "- target: `SWE_PC1(April 1)`",
        "- validation: leave-one-water-year-out ridge regression with no shuffling",
        f"- ridge alpha: `{RIDGE_ALPHA}`",
        "",
        "## Reused SST PC Inputs",
        "",
    ]

    for model_name in sorted(sst_paths):
        lines.append(f"- `{model_name}`: `{sst_paths[model_name]}`")

    lines.extend(
        [
            "",
            "## SWE PCA Metadata",
            "",
            "| Model | Valid Land Cells | Total Land Cells | Singular Value 1 | Singular Value 2 |",
            "|-------|------------------|------------------|------------------|------------------|",
        ]
    )

    for model_name in sorted(swe_meta):
        row = swe_meta[model_name]
        lines.append(
            f"| {model_name} | {row['valid_land_cell_count']} | {row['land_cell_count']} | "
            f"{row['singular_value_1']:.6e} | {row['singular_value_2']:.6e} |"
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Model | Lead Day | N Predictions | N Water Years | Correlation | RMSE | R2 |",
            "|-------|----------|---------------|---------------|-------------|------|----|",
        ]
    )

    for row in sorted(results_rows, key=lambda item: (str(item["model"]), int(item["lead_days"]))):
        lines.append(
            f"| {row['model']} | {row['lead_days']} | {row['n_predictions']} | {row['n_water_years']} | "
            f"{row['correlation']:.6f} | {row['rmse']:.6f} | {row['r2']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runtime = {
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    if not runtime["slurm_job_id"] or "nid" not in runtime["hostname"]:
        raise RuntimeError("Do not run this script on a login node; active compute allocation required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    models = discover_models()
    expected_sst_paths = {
        model_name: str(SST_PC_DIR / f"{model_name}_sst_pcs.csv")
        for model_name in models
    }
    missing = [path for path in expected_sst_paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing existing SST PC files: {missing}")

    land_mask = load_wrf_landmask()
    all_prediction_rows: List[Dict[str, object]] = []
    swe_meta: Dict[str, Dict[str, object]] = {}

    for model_name in models:
        print("=" * 80, flush=True)
        print(f"Processing model: {model_name}", flush=True)
        print("=" * 80, flush=True)
        sst_dates, sst_values = load_sst_pc_series(SST_PC_DIR / f"{model_name}_sst_pcs.csv")
        swe_series = compute_daily_swe_pcs(model_name, land_mask)
        swe_meta[model_name] = {
            "valid_land_cell_count": swe_series.valid_land_cell_count,
            "land_cell_count": swe_series.land_cell_count,
            "singular_value_1": float(swe_series.singular_values[0]),
            "singular_value_2": float(swe_series.singular_values[1]),
        }

        for lead_days in LEAD_DAYS:
            samples = build_samples(model_name, sst_dates, sst_values, swe_series, lead_days)
            prediction_rows = run_leave_one_water_year_out(samples, lead_days)
            all_prediction_rows.extend(prediction_rows)

    results_rows = summarize_predictions(all_prediction_rows)
    write_predictions_csv(PREDICTIONS_FILE, all_prediction_rows)
    write_results_csv(RESULTS_FILE, results_rows)

    for model_name in models:
        for lead_days in LEAD_DAYS:
            selected = [
                row for row in all_prediction_rows
                if str(row["model"]) == model_name and int(row["lead_days"]) == lead_days
            ]
            plot_predictions(
                selected,
                PLOTS_DIR / f"{model_name}_lead{lead_days}_prediction_timeseries.png",
            )

    plot_summary(results_rows, PLOTS_DIR / "lead_skill_summary.png")
    write_report(REPORT_FILE, runtime, expected_sst_paths, swe_meta, results_rows)

    print(f"Saved results CSV: {RESULTS_FILE}", flush=True)
    print(f"Saved predictions CSV: {PREDICTIONS_FILE}", flush=True)
    print(f"Saved report: {REPORT_FILE}", flush=True)
    print(f"Saved plots under: {PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
