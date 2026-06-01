#!/usr/bin/env python3
"""
Run a lightweight LOYO Random Forest diagnostic for the scalar JFM mean
California-box land T2m response using Jun-Nov SST PC1-PC6 predictors.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.run_s2s_pc6_t2m_top20_regions_loyo_models as base_mod


DEFAULT_DATASET_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm"
    / "ridge_target_sensitivity_dataset.csv"
)
DEFAULT_LABEL_NETCDF = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
    / "top20_region_labels"
    / "cleaned_top20_region_labels.nc"
)
DEFAULT_MONTHLY_ANOMALY_FILE = base_mod.DEFAULT_MONTHLY_ANOMALY_FILE
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "s2s_pc6_t2m_top20_land_jfm_loyo_random_forest"

INPUT_MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
TARGET_MONTHS = ["Jan", "Feb", "Mar"]
TARGET_MONTH_TO_NUMBER = {"Jan": 1, "Feb": 2, "Mar": 3}

PREDICTIONS_FILE = "loyo_predictions.csv"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
FEATURE_IMPORTANCE_BY_MONTH_FILE = "feature_importance_by_month.csv"
FEATURE_IMPORTANCE_BY_PC_FILE = "feature_importance_by_pc.csv"
METRICS_FILE = "metrics.json"
RUN_CONFIG_FILE = "run_config.json"
SCATTER_PLOT_FILE = "predicted_vs_actual_scatter.png"
TIME_SERIES_PLOT_FILE = "predicted_vs_actual_timeseries.png"
IMPORTANCE_MONTH_PLOT_FILE = "feature_importance_by_month.png"
IMPORTANCE_PC_PLOT_FILE = "feature_importance_by_pc.png"


@dataclass(frozen=True)
class DatasetBundle:
    water_years: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_reference_from_dataset: np.ndarray
    feature_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--label-netcdf", type=Path, default=DEFAULT_LABEL_NETCDF)
    parser.add_argument("--monthly-anomaly-file", type=Path, default=DEFAULT_MONTHLY_ANOMALY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser.parse_args()


def month_feature_names(input_months: Sequence[str]) -> List[str]:
    return [f"{month_name}_PC{pc_index}" for month_name in input_months for pc_index in range(1, 7)]


def weighted_mask_mean(field_2d: np.ndarray, mask_2d: np.ndarray, latitude: np.ndarray) -> float:
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, np.newaxis], mask_2d.shape)
    valid = mask_2d & np.isfinite(field_2d)
    if not np.any(valid):
        return float("nan")
    numerator = np.sum(field_2d[valid] * weights[valid])
    denominator = np.sum(weights[valid])
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def resolve_n_jobs(requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, int(requested))
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value:
        return max(1, int(slurm_value))
    return max(1, min(8, os.cpu_count() or 1))


def init_model(n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=n_jobs,
    )


def load_predictor_dataset(dataset_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Predictor dataset CSV not found: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    feature_names = month_feature_names(INPUT_MONTHS)
    required_columns = ["water_year"] + feature_names + [f"top20_all_{month_name}" for month_name in TARGET_MONTHS]
    missing_columns = [name for name in required_columns if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset CSV is missing required columns: {missing_columns}")

    water_years = df["water_year"].to_numpy(dtype=np.int32)
    x = df[feature_names].to_numpy(dtype=np.float64)
    y_reference = df[[f"top20_all_{month_name}" for month_name in TARGET_MONTHS]].mean(axis=1).to_numpy(dtype=np.float64)
    return water_years, x, y_reference, feature_names


def load_top20_mask(label_netcdf: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not label_netcdf.exists():
        raise FileNotFoundError(f"Top-20 label NetCDF not found: {label_netcdf}")
    with xr.open_dataset(label_netcdf, engine="netcdf4") as ds:
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
        if "selected_top20_mask" in ds:
            mask = np.asarray(ds["selected_top20_mask"].values, dtype=np.int8) > 0
        elif "cleaned_region_label" in ds:
            mask = np.asarray(ds["cleaned_region_label"].values, dtype=np.int32) > 0
        else:
            raise ValueError("Label NetCDF is missing both selected_top20_mask and cleaned_region_label")
    return latitude, longitude, mask


def build_scalar_target(
    monthly_anomaly_file: Path,
    label_latitude: np.ndarray,
    label_longitude: np.ndarray,
    top20_mask: np.ndarray,
    water_years: np.ndarray,
) -> np.ndarray:
    try:
        return build_scalar_target_from_monthly_anomaly(
            monthly_anomaly_file=monthly_anomaly_file,
            label_latitude=label_latitude,
            label_longitude=label_longitude,
            top20_mask=top20_mask,
            water_years=water_years,
        )
    except ValueError as exc:
        print(
            "monthly anomaly target path was unusable; falling back to monthly mean minus climatology: "
            f"{exc}",
            flush=True,
        )
        return build_scalar_target_from_mean_climatology(
            label_latitude=label_latitude,
            label_longitude=label_longitude,
            top20_mask=top20_mask,
            water_years=water_years,
        )


def build_scalar_target_from_monthly_anomaly(
    monthly_anomaly_file: Path,
    label_latitude: np.ndarray,
    label_longitude: np.ndarray,
    top20_mask: np.ndarray,
    water_years: np.ndarray,
) -> np.ndarray:
    if not monthly_anomaly_file.exists():
        raise FileNotFoundError(f"Monthly anomaly file not found: {monthly_anomaly_file}")

    with xr.open_dataset(monthly_anomaly_file, engine="netcdf4") as ds:
        if "t2m" not in ds:
            raise ValueError(f"Expected variable 't2m' in {monthly_anomaly_file}")
        monthly_anomaly = base_mod.subset_era5_region_360(ds["t2m"], base_mod.PACIFIC_SST_REGION_360).sel(
            latitude=slice(float(label_latitude.min()), float(label_latitude.max())),
            longitude=slice(float(label_longitude.min()), float(label_longitude.max())),
        )
        if monthly_anomaly.shape[1:] != top20_mask.shape:
            raise ValueError(
                f"Monthly anomaly subset shape {monthly_anomaly.shape[1:]} does not match top20 mask shape {top20_mask.shape}"
            )
        monthly_anomaly = monthly_anomaly.load()
        time_values = base_mod.to_month_start(np.asarray(monthly_anomaly["time"].values, dtype="datetime64[ns]"))
        month_lookup = base_mod.build_month_lookup(time_values)

        targets: List[float] = []
        for water_year in water_years.tolist():
            month_values = []
            for month_name in TARGET_MONTHS:
                month_number = TARGET_MONTH_TO_NUMBER[month_name]
                key = base_mod.month_key(base_mod.month_start(int(water_year), month_number))
                if key not in month_lookup:
                    raise ValueError(f"Missing target month {key} for water year {water_year}")
                field = np.asarray(monthly_anomaly.isel(time=month_lookup[key]).values, dtype=np.float64)
                month_values.append(weighted_mask_mean(field, top20_mask, label_latitude))
            target_value = float(np.mean(month_values))
            if not np.isfinite(target_value):
                raise ValueError(f"Non-finite JFM target for water year {water_year}")
            targets.append(target_value)

    return np.asarray(targets, dtype=np.float64)


def build_scalar_target_from_mean_climatology(
    label_latitude: np.ndarray,
    label_longitude: np.ndarray,
    top20_mask: np.ndarray,
    water_years: np.ndarray,
) -> np.ndarray:
    with xr.open_dataset(base_mod.ERA5_MONTHLY_MEAN_FILE, engine="netcdf4") as mean_ds, xr.open_dataset(
        base_mod.ERA5_MONTHLY_CLIM_FILE,
        engine="netcdf4",
    ) as clim_ds:
        monthly_mean = base_mod.subset_era5_region_360(mean_ds["t2m"], base_mod.PACIFIC_SST_REGION_360).sel(
            latitude=slice(float(label_latitude.min()), float(label_latitude.max())),
            longitude=slice(float(label_longitude.min()), float(label_longitude.max())),
        )
        monthly_clim = base_mod.subset_era5_region_360(clim_ds["t2m"], base_mod.PACIFIC_SST_REGION_360).sel(
            latitude=slice(float(label_latitude.min()), float(label_latitude.max())),
            longitude=slice(float(label_longitude.min()), float(label_longitude.max())),
        )
        if monthly_mean.shape[1:] != top20_mask.shape:
            raise ValueError(
                f"Monthly mean subset shape {monthly_mean.shape[1:]} does not match top20 mask shape {top20_mask.shape}"
            )
        if monthly_clim.shape[1:] != top20_mask.shape:
            raise ValueError(
                f"Monthly climatology subset shape {monthly_clim.shape[1:]} does not match top20 mask shape {top20_mask.shape}"
            )

        monthly_mean = monthly_mean.load()
        monthly_clim = monthly_clim.load()
        time_values = base_mod.to_month_start(np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]"))
        month_lookup = base_mod.build_month_lookup(time_values)

        targets: List[float] = []
        for water_year in water_years.tolist():
            month_values = []
            for month_name in TARGET_MONTHS:
                month_number = TARGET_MONTH_TO_NUMBER[month_name]
                key = base_mod.month_key(base_mod.month_start(int(water_year), month_number))
                if key not in month_lookup:
                    raise ValueError(f"Missing target month {key} for water year {water_year}")
                time_index = month_lookup[key]
                field = np.asarray(monthly_mean.isel(time=time_index).values, dtype=np.float64)
                clim_field = np.asarray(monthly_clim.isel(month=month_number - 1).values, dtype=np.float64)
                month_values.append(weighted_mask_mean(field - clim_field, top20_mask, label_latitude))
            target_value = float(np.mean(month_values))
            if not np.isfinite(target_value):
                raise ValueError(f"Non-finite JFM target from mean/climatology for water year {water_year}")
            targets.append(target_value)

    return np.asarray(targets, dtype=np.float64)


def build_dataset(args: argparse.Namespace) -> DatasetBundle:
    water_years, x, y_reference, feature_names = load_predictor_dataset(args.dataset_csv)
    label_latitude, label_longitude, top20_mask = load_top20_mask(args.label_netcdf)
    y = build_scalar_target(
        args.monthly_anomaly_file,
        label_latitude,
        label_longitude,
        top20_mask,
        water_years,
    )

    max_abs_diff = float(np.max(np.abs(y - y_reference)))
    print(f"dataset-vs-anomaly JFM target max_abs_diff: {max_abs_diff:.6e}", flush=True)
    if max_abs_diff > 1.0e-6:
        raise ValueError(
            "Monthly-anomaly-derived JFM target does not match the prepared dataset reference "
            f"(max_abs_diff={max_abs_diff:.6e})."
        )

    return DatasetBundle(
        water_years=water_years,
        x=x,
        y=y,
        y_reference_from_dataset=y_reference,
        feature_names=feature_names,
    )


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, output_dir: Path, n_jobs: int, dataset: DatasetBundle) -> None:
    payload = {
        "dataset_csv": str(args.dataset_csv),
        "label_netcdf": str(args.label_netcdf),
        "monthly_anomaly_file": str(args.monthly_anomaly_file),
        "output_dir": str(output_dir),
        "n_jobs": int(n_jobs),
        "rf_params": {
            "n_estimators": 300,
            "max_depth": 3,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "random_state": 42,
        },
        "input_months": INPUT_MONTHS,
        "target_definition": "JFM mean T2m anomaly over selected top-20 land EVR cells",
        "n_water_years": int(dataset.water_years.size),
        "first_water_year": int(dataset.water_years.min()),
        "last_water_year": int(dataset.water_years.max()),
    }
    (output_dir / RUN_CONFIG_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        return pd.DataFrame(columns=["water_year", "y_true", "y_pred"])
    df = pd.read_csv(predictions_path)
    required = ["water_year", "y_true", "y_pred"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Existing predictions file is missing required columns: {missing}")
    df = df[required].copy()
    df["water_year"] = df["water_year"].astype(int)
    df = df.sort_values("water_year").drop_duplicates(subset=["water_year"], keep="last").reset_index(drop=True)
    return df


def append_prediction_row(predictions_path: Path, row: Dict[str, object]) -> None:
    file_exists = predictions_path.exists()
    with predictions_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["water_year", "y_true", "y_pred"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def validate_existing_predictions(dataset: DatasetBundle, predictions_df: pd.DataFrame) -> None:
    if predictions_df.empty:
        return
    expected_years = set(dataset.water_years.tolist())
    invalid_years = sorted(set(predictions_df["water_year"].tolist()) - expected_years)
    if invalid_years:
        raise ValueError(f"Predictions file contains water years not present in dataset: {invalid_years}")

    y_by_year = {int(year): float(value) for year, value in zip(dataset.water_years.tolist(), dataset.y.tolist())}
    for row in predictions_df.itertuples(index=False):
        expected_true = y_by_year[int(row.water_year)]
        if not math.isclose(float(row.y_true), expected_true, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"Existing y_true mismatch for water year {int(row.water_year)}: "
                f"file={float(row.y_true):.12g} expected={expected_true:.12g}"
            )


def run_loyo(dataset: DatasetBundle, output_dir: Path, n_jobs: int) -> pd.DataFrame:
    predictions_path = output_dir / PREDICTIONS_FILE
    existing_predictions = load_predictions(predictions_path)
    validate_existing_predictions(dataset, existing_predictions)
    completed_years = set(existing_predictions["water_year"].tolist())

    print(f"number of completed LOYO predictions: {len(completed_years)}", flush=True)

    for water_year in dataset.water_years.tolist():
        if int(water_year) in completed_years:
            continue
        train_mask = dataset.water_years != int(water_year)
        x_train = dataset.x[train_mask, :]
        y_train = dataset.y[train_mask]
        x_test = dataset.x[~train_mask, :]
        y_test = dataset.y[~train_mask]
        if x_test.shape[0] != 1:
            raise ValueError(f"Expected exactly one held-out sample for water year {water_year}")

        model = init_model(n_jobs)
        model.fit(x_train, y_train)
        y_pred = float(model.predict(x_test)[0])
        row = {
            "water_year": int(water_year),
            "y_true": float(y_test[0]),
            "y_pred": y_pred,
        }
        append_prediction_row(predictions_path, row)
        completed_years.add(int(water_year))
        print(
            f"completed LOYO prediction for WY{int(water_year)}: y_true={float(y_test[0]):.6f} y_pred={y_pred:.6f}",
            flush=True,
        )

    final_predictions = load_predictions(predictions_path)
    validate_existing_predictions(dataset, final_predictions)
    final_predictions = final_predictions.sort_values("water_year").reset_index(drop=True)
    final_predictions.to_csv(predictions_path, index=False)
    return final_predictions


def safe_float(value: float) -> Optional[float]:
    return None if not np.isfinite(value) else float(value)


def compute_metrics(predictions_df: pd.DataFrame, n_total: int) -> Dict[str, object]:
    n_completed = int(len(predictions_df))
    y_true = predictions_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = predictions_df["y_pred"].to_numpy(dtype=np.float64)

    metrics: Dict[str, object] = {
        "n_completed": n_completed,
        "n_total": int(n_total),
        "all_years_completed": bool(n_completed == n_total),
        "r2": None,
        "pearson_correlation": None,
        "rmse": None,
        "mae": None,
    }
    if n_completed == 0:
        return metrics

    metrics["rmse"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

    if n_completed >= 2:
        metrics["r2"] = safe_float(float(r2_score(y_true, y_pred)))
        corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0.0 and np.std(y_pred) > 0.0 else float("nan")
        metrics["pearson_correlation"] = safe_float(float(corr))

    return metrics


def save_metrics(metrics: Dict[str, object], output_dir: Path) -> None:
    (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def compute_feature_importance(
    dataset: DatasetBundle,
    completed_years: Sequence[int],
    n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for water_year in completed_years:
        train_mask = dataset.water_years != int(water_year)
        model = init_model(n_jobs)
        model.fit(dataset.x[train_mask, :], dataset.y[train_mask])
        for feature_name, importance in zip(dataset.feature_names, model.feature_importances_):
            month_name, pc_label = feature_name.split("_")
            rows.append(
                {
                    "held_out_water_year": int(water_year),
                    "feature_name": feature_name,
                    "month": month_name,
                    "pc": pc_label,
                    "importance": float(importance),
                }
            )

    importance_df = pd.DataFrame(rows)
    if importance_df.empty:
        empty_month = pd.DataFrame(columns=["month", "importance_mean", "importance_std", "n_completed"])
        empty_pc = pd.DataFrame(columns=["pc", "importance_mean", "importance_std", "n_completed"])
        return importance_df, empty_month, empty_pc

    by_month = (
        importance_df.groupby(["held_out_water_year", "month"], as_index=False)["importance"]
        .sum()
        .groupby("month")
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"), n_completed=("importance", "count"))
        .reset_index()
    )
    by_pc = (
        importance_df.groupby(["held_out_water_year", "pc"], as_index=False)["importance"]
        .sum()
        .groupby("pc")
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"), n_completed=("importance", "count"))
        .reset_index()
    )
    by_month["importance_std"] = by_month["importance_std"].fillna(0.0)
    by_pc["importance_std"] = by_pc["importance_std"].fillna(0.0)
    return importance_df, by_month, by_pc


def save_feature_importance_outputs(
    importance_df: pd.DataFrame,
    by_month: pd.DataFrame,
    by_pc: pd.DataFrame,
    output_dir: Path,
) -> None:
    importance_df.to_csv(output_dir / FEATURE_IMPORTANCE_FILE, index=False)
    by_month.to_csv(output_dir / FEATURE_IMPORTANCE_BY_MONTH_FILE, index=False)
    by_pc.to_csv(output_dir / FEATURE_IMPORTANCE_BY_PC_FILE, index=False)


def make_scatter_plot(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    if predictions_df.empty:
        return
    y_true = predictions_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = predictions_df["y_pred"].to_numpy(dtype=np.float64)
    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(y_true, y_pred, s=48, color="#355070", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#bc6c25", linewidth=1.2)
    ax.set_xlabel("Actual JFM mean T2m anomaly")
    ax.set_ylabel("Predicted JFM mean T2m anomaly")
    ax.set_title(f"LOYO Random Forest: predicted vs actual (n={len(predictions_df)})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / SCATTER_PLOT_FILE, dpi=160)
    plt.close(fig)


def make_time_series_plot(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    if predictions_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.plot(predictions_df["water_year"], predictions_df["y_true"], marker="o", linewidth=1.6, color="#355070", label="Actual")
    ax.plot(predictions_df["water_year"], predictions_df["y_pred"], marker="o", linewidth=1.6, color="#bc6c25", label="Predicted")
    ax.set_xlabel("Water year")
    ax.set_ylabel("JFM mean T2m anomaly")
    ax.set_title("LOYO Random Forest by water year")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / TIME_SERIES_PLOT_FILE, dpi=160)
    plt.close(fig)


def make_importance_plot(
    table: pd.DataFrame,
    x_column: str,
    output_path: Path,
    title: str,
    color: str,
) -> None:
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(table[x_column].astype(str), table["importance_mean"], color=color, alpha=0.9)
    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel("Mean feature importance")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_plots(predictions_df: pd.DataFrame, by_month: pd.DataFrame, by_pc: pd.DataFrame, output_dir: Path) -> None:
    make_scatter_plot(predictions_df, output_dir)
    make_time_series_plot(predictions_df, output_dir)
    month_order = pd.Categorical(by_month["month"], categories=INPUT_MONTHS, ordered=True) if not by_month.empty else None
    if month_order is not None:
        by_month = by_month.assign(_month_order=month_order).sort_values("_month_order").drop(columns="_month_order")
    pc_order = pd.Categorical(by_pc["pc"], categories=[f"PC{i}" for i in range(1, 7)], ordered=True) if not by_pc.empty else None
    if pc_order is not None:
        by_pc = by_pc.assign(_pc_order=pc_order).sort_values("_pc_order").drop(columns="_pc_order")
    make_importance_plot(
        by_month,
        x_column="month",
        output_path=output_dir / IMPORTANCE_MONTH_PLOT_FILE,
        title="Random Forest feature importance by month",
        color="#6d597a",
    )
    make_importance_plot(
        by_pc,
        x_column="pc",
        output_path=output_dir / IMPORTANCE_PC_PLOT_FILE,
        title="Random Forest feature importance by PC",
        color="#2a9d8f",
    )


def main() -> None:
    args = parse_args()
    n_jobs = resolve_n_jobs(args.n_jobs)
    ensure_output_dir(args.output_dir)

    dataset = build_dataset(args)
    save_run_config(args, args.output_dir, n_jobs, dataset)

    print(f"X shape: {dataset.x.shape}", flush=True)
    print(f"Y shape: {dataset.y.shape}", flush=True)
    print(f"number of water years: {dataset.water_years.size}", flush=True)
    print(f"first water year: {int(dataset.water_years.min())}", flush=True)
    print(f"last water year: {int(dataset.water_years.max())}", flush=True)
    print(f"output directory: {args.output_dir}", flush=True)

    predictions_df = run_loyo(dataset, args.output_dir, n_jobs)
    print(f"number of completed LOYO predictions: {len(predictions_df)}", flush=True)

    metrics = compute_metrics(predictions_df, n_total=int(dataset.water_years.size))
    save_metrics(metrics, args.output_dir)

    completed_years = predictions_df["water_year"].astype(int).tolist()
    importance_df, by_month, by_pc = compute_feature_importance(dataset, completed_years, n_jobs)
    save_feature_importance_outputs(importance_df, by_month, by_pc, args.output_dir)
    save_plots(predictions_df, by_month, by_pc, args.output_dir)


if __name__ == "__main__":
    main()
