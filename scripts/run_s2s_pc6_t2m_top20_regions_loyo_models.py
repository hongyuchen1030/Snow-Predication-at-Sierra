#!/usr/bin/env python3
"""
Run LOYO Ridge / Random Forest / MLP experiments for Sierra top-20% T2m regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_cobe2_global_sst_eof_reproduction import COBE2_SST_FILE
from scripts.run_cobe2_pacific_sierra_t2m_level1_diagnostic import (
    ERA5_MONTHLY_CLIM_FILE,
    ERA5_MONTHLY_MEAN_FILE,
    PACIFIC_SST_REGION_360,
    subset_era5_region_360,
    to_month_start,
)
from scripts.run_cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only import (
    EXPANDED_LEVEL2_NETCDF_FILE,
)


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
DEFAULT_LABEL_DIR = DEFAULT_ARTIFACT_DIR / "top20_region_labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "s2s_pc6_t2m_top20_regions_loyo_ridge_rf_mlp"
DEFAULT_ROUTE2_NETCDF = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only/"
    "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only.nc"
)
DEFAULT_MONTHLY_ANOMALY_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "era5land_t2m_monthly_anomalies/era5land_t2m_monthly_anomaly.nc"
)
SEARCH_SUFFIXES = (".nc", ".npz", ".npy", ".csv", ".pkl", ".json")
INPUT_MONTH_TO_NUMBER = {"Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11}
TARGET_MONTH_TO_NUMBER = {"Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4}
SEMANTIC_REGION_NAMES = {
    1: "coastal/left group",
    2: "inland/right group",
    3: "northern/top group",
}
RIDGE_GRID = [{"alpha": value} for value in [0.01, 0.1, 1.0, 10.0, 100.0]]
RF_GRID = [
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    for n_estimators in [200, 500]
    for max_depth in [2, 3, 4, None]
    for min_samples_leaf in [1, 2, 4]
    for max_features in ["sqrt", 0.5, 1.0]
]
MLP_GRID = [
    {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "activation": activation,
    }
    for hidden_layer_sizes in [(16,), (32,), (32, 16)]
    for alpha in [1e-4, 1e-3, 1e-2]
    for activation in ["relu", "tanh"]
]
MODEL_ORDER = ["ridge", "random_forest", "mlp"]
MODEL_DISPLAY = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "mlp": "MLPRegressor",
}
MODEL_COLORS = {
    "ridge": "#355070",
    "random_forest": "#b56576",
    "mlp": "#2a9d8f",
}


@dataclass(frozen=True)
class RegionDefinition:
    semantic_label: int
    semantic_name: str
    source_cleaned_label: int
    centroid_lat: float
    centroid_lon: float
    size: int


@dataclass(frozen=True)
class DatasetBundle:
    water_years: np.ndarray
    x: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    dropped_years: List[Dict[str, object]]
    region_definitions: List[RegionDefinition]
    paths_used: Dict[str, str]
    t2m_source_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-months", nargs="+", default=["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"])
    parser.add_argument("--target-months", nargs="+", default=["Dec", "Jan", "Feb", "Mar", "Apr"])
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    return parser.parse_args()


def month_start(year: int, month: int) -> np.datetime64:
    return np.datetime64(f"{year:04d}-{month:02d}-01", "ns")


def month_key(value: np.datetime64) -> str:
    month_value = np.asarray(value, dtype="datetime64[M]")
    return str(np.datetime_as_string(month_value, unit="M"))


def build_month_lookup(values: np.ndarray) -> Dict[str, int]:
    return {month_key(value): index for index, value in enumerate(np.asarray(values, dtype="datetime64[ns]"))}


def candidate_files_from_dir(directory: Path) -> List[Path]:
    results: List[Path] = []
    if not directory.exists():
        return results
    for suffix in SEARCH_SUFFIXES:
        results.extend(sorted(directory.rglob(f"*{suffix}")))
    return sorted(set(results))


def print_candidate_group(title: str, paths: Sequence[Path]) -> None:
    print(title, flush=True)
    if not paths:
        print("  none found", flush=True)
        return
    for path in paths:
        print(f"  {path}", flush=True)


def locate_route2_netcdf(artifact_dir: Path) -> Path:
    local_candidates = sorted(artifact_dir.rglob("*.nc"))
    preferred = [path for path in local_candidates if "high_predictability_sierra_only" in path.name]
    if preferred:
        return preferred[0]
    if DEFAULT_ROUTE2_NETCDF.exists():
        return DEFAULT_ROUTE2_NETCDF
    raise FileNotFoundError("Could not locate the Route 2 Sierra-only high-predictability NetCDF.")


def locate_label_netcdf(label_dir: Path) -> Path:
    preferred = label_dir / "cleaned_top20_region_labels.nc"
    if preferred.exists():
        return preferred
    candidates = sorted(label_dir.rglob("*.nc"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not locate cleaned top-20 region labels NetCDF.")


def locate_era5_monthly_anomaly_file() -> Path | None:
    return DEFAULT_MONTHLY_ANOMALY_FILE if DEFAULT_MONTHLY_ANOMALY_FILE.exists() else None


def infer_semantic_region_order(cleaned_labels: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> List[RegionDefinition]:
    cleaned_ids = [int(value) for value in np.unique(cleaned_labels) if int(value) > 0]
    if len(cleaned_ids) != 3:
        raise ValueError(f"Expected exactly 3 cleaned labels, found {cleaned_ids}")

    raw_rows = []
    for cleaned_id in cleaned_ids:
        indices = np.argwhere(cleaned_labels == cleaned_id)
        raw_rows.append(
            {
                "cleaned_id": cleaned_id,
                "size": int(indices.shape[0]),
                "centroid_lat": float(latitude[indices[:, 0]].mean()),
                "centroid_lon": float(longitude[indices[:, 1]].mean()),
            }
        )

    northern = max(raw_rows, key=lambda row: (row["centroid_lat"], -row["centroid_lon"]))
    remaining = [row for row in raw_rows if row["cleaned_id"] != northern["cleaned_id"]]
    coastal = min(remaining, key=lambda row: row["centroid_lon"])
    inland = max(remaining, key=lambda row: row["centroid_lon"])

    semantic_rows = [
        RegionDefinition(
            semantic_label=1,
            semantic_name=SEMANTIC_REGION_NAMES[1],
            source_cleaned_label=int(coastal["cleaned_id"]),
            centroid_lat=float(coastal["centroid_lat"]),
            centroid_lon=float(coastal["centroid_lon"]),
            size=int(coastal["size"]),
        ),
        RegionDefinition(
            semantic_label=2,
            semantic_name=SEMANTIC_REGION_NAMES[2],
            source_cleaned_label=int(inland["cleaned_id"]),
            centroid_lat=float(inland["centroid_lat"]),
            centroid_lon=float(inland["centroid_lon"]),
            size=int(inland["size"]),
        ),
        RegionDefinition(
            semantic_label=3,
            semantic_name=SEMANTIC_REGION_NAMES[3],
            source_cleaned_label=int(northern["cleaned_id"]),
            centroid_lat=float(northern["centroid_lat"]),
            centroid_lon=float(northern["centroid_lon"]),
            size=int(northern["size"]),
        ),
    ]
    return semantic_rows


def load_label_mask(label_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[RegionDefinition]]:
    with xr.open_dataset(label_path, engine="netcdf4") as ds:
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
        cleaned = np.asarray(ds["cleaned_region_label"].values, dtype=np.int32)
    regions = infer_semantic_region_order(cleaned, latitude, longitude)
    return latitude, longitude, cleaned, regions


def load_pacific_pc_timeseries(level2_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(level2_path, engine="netcdf4") as ds:
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        pc_values = np.asarray(ds["pacific_cobe2_pc"].values, dtype=np.float64)
    if pc_values.shape[1] < 6:
        raise ValueError(f"Expected at least 6 SST PCs, got shape {pc_values.shape}")
    return time_values, pc_values[:, :6]


def load_era5_monthly_region_targets(
    label_latitude: np.ndarray,
    label_longitude: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    cleaned_labels: np.ndarray,
) -> Tuple[Dict[str, Dict[int, float]], str]:
    with xr.open_dataset(ERA5_MONTHLY_MEAN_FILE, engine="netcdf4") as mean_ds, xr.open_dataset(
        ERA5_MONTHLY_CLIM_FILE,
        engine="netcdf4",
    ) as clim_ds:
        monthly_mean = subset_era5_region_360(mean_ds["t2m"], PACIFIC_SST_REGION_360).sel(
            latitude=slice(float(label_latitude.min()), float(label_latitude.max())),
            longitude=slice(float(label_longitude.min()), float(label_longitude.max())),
        )
        monthly_clim = subset_era5_region_360(clim_ds["t2m"], PACIFIC_SST_REGION_360).sel(
            latitude=slice(float(label_latitude.min()), float(label_latitude.max())),
            longitude=slice(float(label_longitude.min()), float(label_longitude.max())),
        )
        if monthly_mean.shape[1:] != (label_latitude.size, label_longitude.size):
            raise ValueError(
                f"Monthly mean subset shape {monthly_mean.shape[1:]} does not match label grid "
                f"{(label_latitude.size, label_longitude.size)}"
            )
        if monthly_clim.shape[1:] != (label_latitude.size, label_longitude.size):
            raise ValueError(
                f"Monthly climatology subset shape {monthly_clim.shape[1:]} does not match label grid "
                f"{(label_latitude.size, label_longitude.size)}"
            )
        time_values = np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]")
        month_numbers = np.asarray(monthly_mean["time"].dt.month.values, dtype=np.int32)
        climatology_values = np.asarray(monthly_clim.values, dtype=np.float64)

        region_masks = {
            region.semantic_label: cleaned_labels == region.source_cleaned_label for region in region_definitions
        }
        monthly_targets: Dict[str, Dict[int, float]] = {}
        for index, (time_value, month_value) in enumerate(zip(time_values, month_numbers.tolist())):
            monthly_mean_slice = np.asarray(monthly_mean.isel(time=index).load().values, dtype=np.float64)
            monthly_clim_slice = climatology_values[month_value - 1, :, :]
            anomaly_slice = monthly_mean_slice - monthly_clim_slice
            monthly_targets[month_key(time_value)] = {
                region_label: weighted_regional_mean(anomaly_slice, region_masks[region_label], label_latitude)
                for region_label in [1, 2, 3]
            }
            if index == 0 or (index + 1) % 120 == 0 or index + 1 == time_values.size:
                print(
                    f"  processed ERA5 monthly target {index + 1}/{time_values.size}: {month_key(time_value)}",
                    flush=True,
                )
    return monthly_targets, "computed anomalies from monthly mean minus monthly climatology"


def weighted_regional_mean(field_2d: np.ndarray, mask_2d: np.ndarray, latitude: np.ndarray) -> float:
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, np.newaxis], mask_2d.shape)
    valid = mask_2d & np.isfinite(field_2d)
    if not np.any(valid):
        return float("nan")
    weighted_sum = np.sum(field_2d[valid] * weights[valid])
    weight_sum = np.sum(weights[valid])
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def build_feature_names(input_months: Sequence[str]) -> List[str]:
    names: List[str] = []
    for month_name in input_months:
        for pc_index in range(1, 7):
            names.append(f"{month_name}_PC{pc_index}")
    return names


def build_target_names(target_months: Sequence[str]) -> List[str]:
    names: List[str] = []
    for region_label in [1, 2, 3]:
        for month_name in target_months:
            names.append(f"region{region_label}_{month_name}")
    return names


def build_supervised_dataset(
    route2_netcdf: Path,
    label_netcdf: Path,
    input_months: Sequence[str],
    target_months: Sequence[str],
) -> DatasetBundle:
    print(f"Loading cleaned region labels from {label_netcdf}", flush=True)
    label_latitude, label_longitude, cleaned_labels, region_definitions = load_label_mask(label_netcdf)
    print(f"Loading SST PC1-PC6 from {EXPANDED_LEVEL2_NETCDF_FILE}", flush=True)
    pc_time, pc_values = load_pacific_pc_timeseries(EXPANDED_LEVEL2_NETCDF_FILE)
    print("Loading ERA5-Land monthly mean/climatology and computing regional T2m anomaly targets", flush=True)
    era5_monthly_targets, t2m_source_note = load_era5_monthly_region_targets(
        label_latitude,
        label_longitude,
        region_definitions,
        cleaned_labels,
    )
    pc_time = to_month_start(pc_time)
    pc_index = build_month_lookup(pc_time)

    common_times = sorted(set(pc_index).intersection(set(era5_monthly_targets)))
    if not common_times:
        raise ValueError("No overlapping months between SST PCs and ERA5 anomalies.")

    first_year = int(min(common_times)[:4])
    last_year = int(max(common_times)[:4])
    candidate_water_years = list(range(first_year + 1, last_year + 1))
    print(
        f"Building supervised dataset across candidate water years {candidate_water_years[0]}-{candidate_water_years[-1]} "
        f"({len(candidate_water_years)} candidates)",
        flush=True,
    )

    feature_names = build_feature_names(input_months)
    target_names = build_target_names(target_months)

    rows: List[Dict[str, object]] = []
    dropped_years: List[Dict[str, object]] = []
    for water_year in candidate_water_years:
        input_month_dates = [month_start(water_year - 1, INPUT_MONTH_TO_NUMBER[name]) for name in input_months]
        target_month_dates = []
        for name in target_months:
            month_number = TARGET_MONTH_TO_NUMBER[name]
            target_year = water_year - 1 if month_number == 12 else water_year
            target_month_dates.append(month_start(target_year, month_number))
        input_keys = [month_key(value) for value in input_month_dates]
        target_keys = [month_key(value) for value in target_month_dates]

        missing_inputs = [value for value in input_keys if value not in pc_index]
        missing_targets = [value for value in target_keys if value not in era5_monthly_targets]
        if missing_inputs or missing_targets:
            dropped_years.append(
                {
                    "water_year": water_year,
                    "reason": {
                        "missing_inputs": missing_inputs,
                        "missing_targets": missing_targets,
                    },
                }
            )
            continue

        x_matrix = np.stack([pc_values[pc_index[value], :] for value in input_keys], axis=0)
        x_vector = x_matrix.reshape(-1)

        y_values: List[float] = []
        for region_label in [1, 2, 3]:
            for target_value in target_keys:
                y_values.append(float(era5_monthly_targets[target_value][region_label]))
        y_vector = np.asarray(y_values, dtype=np.float64)

        if np.isnan(x_vector).any() or np.isnan(y_vector).any():
            dropped_years.append(
                {
                    "water_year": water_year,
                    "reason": {
                        "input_nan_count": int(np.isnan(x_vector).sum()),
                        "target_nan_count": int(np.isnan(y_vector).sum()),
                    },
                }
            )
            continue

        row: Dict[str, object] = {"water_year": water_year}
        row.update({name: float(value) for name, value in zip(feature_names, x_vector.tolist())})
        row.update({name: float(value) for name, value in zip(target_names, y_vector.tolist())})
        rows.append(row)

    if not rows:
        preview = dropped_years[:5]
        raise ValueError(f"No complete water years were available after screening. First dropped rows: {preview}")

    df = pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)
    print(f"Finished dataset assembly with {len(df)} usable water years", flush=True)
    x = df[feature_names].to_numpy(dtype=np.float64)
    y = df[target_names].to_numpy(dtype=np.float64)
    water_years = df["water_year"].to_numpy(dtype=np.int32)
    return DatasetBundle(
        water_years=water_years,
        x=x,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        dropped_years=dropped_years,
        region_definitions=region_definitions,
        paths_used={
            "route2_netcdf": str(route2_netcdf),
            "expanded_level2_netcdf": str(EXPANDED_LEVEL2_NETCDF_FILE),
            "label_netcdf": str(label_netcdf),
            "monthly_anomaly_file": str(DEFAULT_MONTHLY_ANOMALY_FILE),
            "monthly_mean_file": str(ERA5_MONTHLY_MEAN_FILE),
            "monthly_climatology_file": str(ERA5_MONTHLY_CLIM_FILE),
            "cobe2_sst_file": str(COBE2_SST_FILE),
        },
        t2m_source_note=t2m_source_note,
    )


def init_model(model_name: str, params: Dict[str, object]):
    if model_name == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=0,
            n_jobs=-1,
        )
    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            alpha=float(params["alpha"]),
            activation=str(params["activation"]),
            solver="lbfgs",
            max_iter=5000,
            random_state=0,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def parameter_grid(model_name: str) -> List[Dict[str, object]]:
    if model_name == "ridge":
        return RIDGE_GRID
    if model_name == "random_forest":
        return RF_GRID
    if model_name == "mlp":
        return MLP_GRID
    raise ValueError(f"Unsupported model: {model_name}")


def inner_cv_splitter(n_train: int):
    if n_train <= 8:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_train), shuffle=False)


def fit_scaled_model(
    model_name: str,
    params: Dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    warning_rows: List[Dict[str, object]],
    context: Dict[str, object],
) -> Tuple[np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_eval_scaled = x_scaler.transform(x_eval)
    model = init_model(model_name, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train_scaled, y_train_scaled)
    for item in caught:
        if issubclass(item.category, Warning):
            warning_rows.append(
                {
                    "model": model_name,
                    "stage": context["stage"],
                    "outer_water_year": context.get("outer_water_year"),
                    "params_json": json.dumps(params, sort_keys=True),
                    "warning_category": item.category.__name__,
                    "warning_message": str(item.message),
                }
            )
    pred_scaled = model.predict(x_eval_scaled)
    pred_scaled = np.asarray(pred_scaled, dtype=np.float64)
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled[:, np.newaxis]
    return pred_scaled, x_scaler, y_scaler


def score_candidate(
    model_name: str,
    params: Dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    warning_rows: List[Dict[str, object]],
    outer_water_year: int,
) -> float:
    splitter = inner_cv_splitter(x_train.shape[0])
    fold_scores: List[float] = []
    for inner_train_index, inner_valid_index in splitter.split(x_train):
        pred_scaled, _, y_scaler = fit_scaled_model(
            model_name,
            params,
            x_train[inner_train_index],
            y_train[inner_train_index],
            x_train[inner_valid_index],
            warning_rows,
            {"stage": "inner_cv", "outer_water_year": outer_water_year},
        )
        y_valid_scaled = y_scaler.transform(y_train[inner_valid_index])
        fold_scores.append(-mean_squared_error(y_valid_scaled, pred_scaled))
    return float(np.mean(fold_scores))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return float("nan")
    if np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": pearson_corr(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_loyo(
    dataset: DatasetBundle,
    models: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    x = dataset.x
    y = dataset.y
    water_years = dataset.water_years
    predictions = {model_name: np.full_like(y, np.nan, dtype=np.float64) for model_name in models}
    hyperparameter_rows: List[Dict[str, object]] = []
    warning_rows: List[Dict[str, object]] = []

    for model_name in models:
        print(f"Starting LOYO sweep for model={model_name} across {water_years.size} water years", flush=True)
        for outer_index, test_water_year in enumerate(water_years):
            train_mask = water_years != test_water_year
            x_train = x[train_mask]
            y_train = y[train_mask]
            x_test = x[~train_mask]
            if x_test.shape[0] != 1:
                raise ValueError("Expected exactly one held-out sample per LOYO fold.")

            best_params = None
            best_score = -np.inf
            for params in parameter_grid(model_name):
                score = score_candidate(model_name, params, x_train, y_train, warning_rows, int(test_water_year))
                if score > best_score:
                    best_score = score
                    best_params = params
            if best_params is None:
                raise RuntimeError(f"No hyperparameters selected for {model_name} WY{int(test_water_year)}")

            pred_scaled, _, y_scaler = fit_scaled_model(
                model_name,
                best_params,
                x_train,
                y_train,
                x_test,
                warning_rows,
                {"stage": "outer_fit", "outer_water_year": int(test_water_year)},
            )
            predictions[model_name][outer_index, :] = y_scaler.inverse_transform(pred_scaled)[0]
            hyperparameter_rows.append(
                {
                    "outer_water_year": int(test_water_year),
                    "model": model_name,
                    "selected_params_json": json.dumps(best_params, sort_keys=True),
                    "inner_cv_score_neg_mean_mse_scaled_y": float(best_score),
                }
            )
            print(
                f"LOYO held_out_WY={int(test_water_year)} model={model_name} best_params={best_params} "
                f"inner_score={best_score:.6f}",
                flush=True,
            )
        print(f"Finished LOYO sweep for model={model_name}", flush=True)

    hyperparameter_df = pd.DataFrame(hyperparameter_rows).sort_values(["model", "outer_water_year"]).reset_index(drop=True)
    warning_df = pd.DataFrame(warning_rows)
    return predictions, hyperparameter_df, warning_df


def reshape_targets(y: np.ndarray, n_regions: int = 3, n_months: int = 5) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(y.shape[0], n_regions, n_months)


def compute_all_metrics(
    water_years: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    overall_rows: List[Dict[str, object]] = []
    region_rows: List[Dict[str, object]] = []
    month_rows: List[Dict[str, object]] = []
    region_month_rows: List[Dict[str, object]] = []
    summary_json: Dict[str, object] = {"water_years": water_years.tolist(), "models": {}}

    y_true_reshaped = reshape_targets(y_true)
    for model_name, y_pred in predictions.items():
        y_pred_reshaped = reshape_targets(y_pred)
        overall = compute_metric_block(y_true.reshape(-1), y_pred.reshape(-1))
        overall_rows.append({"model": model_name, **overall})

        model_payload: Dict[str, object] = {"overall": overall, "by_region": [], "by_month": [], "region_by_month": []}
        for region_index, region in enumerate(region_definitions):
            block = compute_metric_block(
                y_true_reshaped[:, region_index, :].reshape(-1),
                y_pred_reshaped[:, region_index, :].reshape(-1),
            )
            row = {
                "model": model_name,
                "region_label": region.semantic_label,
                "region_name": region.semantic_name,
                **block,
            }
            region_rows.append(row)
            model_payload["by_region"].append(row)

        for month_index, month_name in enumerate(target_months):
            block = compute_metric_block(
                y_true_reshaped[:, :, month_index].reshape(-1),
                y_pred_reshaped[:, :, month_index].reshape(-1),
            )
            row = {"model": model_name, "target_month": month_name, **block}
            month_rows.append(row)
            model_payload["by_month"].append(row)

        for region_index, region in enumerate(region_definitions):
            for month_index, month_name in enumerate(target_months):
                block = compute_metric_block(
                    y_true_reshaped[:, region_index, month_index],
                    y_pred_reshaped[:, region_index, month_index],
                )
                row = {
                    "model": model_name,
                    "region_label": region.semantic_label,
                    "region_name": region.semantic_name,
                    "target_month": month_name,
                    **block,
                }
                region_month_rows.append(row)
                model_payload["region_by_month"].append(row)
        summary_json["models"][model_name] = model_payload

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(region_rows),
        pd.DataFrame(month_rows),
        pd.DataFrame(region_month_rows),
        summary_json,
    )


def save_dataset_files(dataset: DatasetBundle, output_dir: Path) -> Path:
    df = pd.DataFrame(
        np.column_stack([dataset.water_years[:, np.newaxis], dataset.x, dataset.y]),
        columns=["water_year"] + dataset.feature_names + dataset.target_names,
    )
    df["water_year"] = df["water_year"].astype(int)
    dataset_csv = output_dir / "s2s_pc6_t2m_top20_regions_dataset.csv"
    df.to_csv(dataset_csv, index=False)
    np.save(output_dir / "X.npy", dataset.x)
    np.save(output_dir / "Y.npy", dataset.y)
    np.save(output_dir / "water_years.npy", dataset.water_years)
    (output_dir / "feature_names.json").write_text(json.dumps(dataset.feature_names, indent=2) + "\n", encoding="utf-8")
    (output_dir / "target_names.json").write_text(json.dumps(dataset.target_names, indent=2) + "\n", encoding="utf-8")
    return dataset_csv


def save_predictions(
    output_dir: Path,
    water_years: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> Tuple[Path, Path]:
    rows_long: List[Dict[str, object]] = []
    rows_wide: List[Dict[str, object]] = []
    for model_name, y_pred in predictions.items():
        true_reshaped = reshape_targets(y_true)
        pred_reshaped = reshape_targets(y_pred)
        for row_index, water_year in enumerate(water_years.tolist()):
            row_wide: Dict[str, object] = {"model": model_name, "water_year": int(water_year)}
            for region_index, region in enumerate(region_definitions):
                for month_index, month_name in enumerate(target_months):
                    column_key = f"region{region.semantic_label}_{month_name}"
                    y_true_value = float(true_reshaped[row_index, region_index, month_index])
                    y_pred_value = float(pred_reshaped[row_index, region_index, month_index])
                    rows_long.append(
                        {
                            "model": model_name,
                            "water_year": int(water_year),
                            "region_label": region.semantic_label,
                            "region_name": region.semantic_name,
                            "target_month": month_name,
                            "y_true": y_true_value,
                            "y_pred": y_pred_value,
                            "error": y_pred_value - y_true_value,
                        }
                    )
                    row_wide[f"{column_key}_true"] = y_true_value
                    row_wide[f"{column_key}_pred"] = y_pred_value
                    row_wide[f"{column_key}_error"] = y_pred_value - y_true_value
            rows_wide.append(row_wide)

    long_path = output_dir / "loyo_predictions_long.csv"
    wide_path = output_dir / "loyo_predictions_wide.csv"
    pd.DataFrame(rows_long).to_csv(long_path, index=False)
    pd.DataFrame(rows_wide).to_csv(wide_path, index=False)
    np.save(output_dir / "y_true.npy", y_true)
    if "ridge" in predictions:
        np.save(output_dir / "y_pred_ridge.npy", predictions["ridge"])
    if "random_forest" in predictions:
        np.save(output_dir / "y_pred_random_forest.npy", predictions["random_forest"])
    if "mlp" in predictions:
        np.save(output_dir / "y_pred_mlp.npy", predictions["mlp"])
    return long_path, wide_path


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
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.2, linestyle="--")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed T2m anomaly")
    ax.set_ylabel("Predicted T2m anomaly")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_region_month_scatter_grid(
    plots_dir: Path,
    region_month_metrics: pd.DataFrame,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    y_true_reshaped = reshape_targets(y_true)
    for model_name, y_pred in predictions.items():
        pred_reshaped = reshape_targets(y_pred)
        fig, axes = plt.subplots(
            len(region_definitions),
            len(target_months),
            figsize=(4.0 * len(target_months), 3.8 * len(region_definitions)),
            constrained_layout=True,
        )
        for region_index, region in enumerate(region_definitions):
            for month_index, month_name in enumerate(target_months):
                ax = axes[region_index, month_index]
                metrics_row = region_month_metrics[
                    (region_month_metrics["model"] == model_name)
                    & (region_month_metrics["region_label"] == region.semantic_label)
                    & (region_month_metrics["target_month"] == month_name)
                ].iloc[0]
                observed = y_true_reshaped[:, region_index, month_index]
                predicted = pred_reshaped[:, region_index, month_index]
                scatter_with_identity(
                    ax,
                    observed,
                    predicted,
                    MODEL_COLORS[model_name],
                    (
                        f"Region {region.semantic_label} | {month_name}\n"
                        f"R2={metrics_row['r2']:.3f}, corr={metrics_row['corr']:.3f}\n"
                        f"RMSE={metrics_row['rmse']:.3f}, MAE={metrics_row['mae']:.3f}"
                    ),
                )
                if region_index == len(region_definitions) - 1:
                    ax.set_xlabel(f"Observed ({month_name})")
                else:
                    ax.set_xlabel("")
                if month_index == 0:
                    ax.set_ylabel(f"Predicted\n{region.semantic_name}")
                else:
                    ax.set_ylabel("")
        fig.suptitle(f"{MODEL_DISPLAY[model_name]} observed vs predicted by region and month", fontsize=14)
        fig.savefig(plots_dir / f"{model_name}_region_month_scatter_grid.png", dpi=220)
        fig.savefig(plots_dir / f"{model_name}_region_month_scatter_grid.pdf")
        plt.close(fig)


def plot_time_series(
    plots_dir: Path,
    water_years: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    y_true_reshaped = reshape_targets(y_true)
    for model_name, y_pred in predictions.items():
        pred_reshaped = reshape_targets(y_pred)
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
        for region_index, region in enumerate(region_definitions):
            ax = axes[region_index]
            for month_index, month_name in enumerate(target_months):
                color = plt.cm.tab10(month_index)
                ax.plot(
                    water_years,
                    y_true_reshaped[:, region_index, month_index],
                    color=color,
                    linewidth=1.8,
                    label=f"{month_name} observed" if region_index == 0 else None,
                )
                ax.plot(
                    water_years,
                    pred_reshaped[:, region_index, month_index],
                    color=color,
                    linewidth=1.4,
                    linestyle="--",
                    label=f"{month_name} predicted" if region_index == 0 else None,
                )
            ax.set_title(f"Region {region.semantic_label}: {region.semantic_name}")
            ax.set_ylabel("T2m anomaly")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Water year")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
        fig.savefig(plots_dir / f"{model_name}_time_series_by_region.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: DatasetBundle,
    models: Sequence[str],
    hyperparameter_df: pd.DataFrame,
) -> None:
    package_versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "matplotlib": matplotlib.__version__,
        "xarray": xr.__version__,
        "scipy": __import__("scipy").__version__,
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_path": str(PROJECT_ROOT),
        "artifact_dir": str(args.artifact_dir),
        "label_dir": str(args.label_dir),
        "output_dir": str(args.output_dir),
        "route2_netcdf": dataset.paths_used["route2_netcdf"],
        "expanded_level2_netcdf": dataset.paths_used["expanded_level2_netcdf"],
        "region_label_file": dataset.paths_used["label_netcdf"],
        "water_years_used": dataset.water_years.tolist(),
        "input_months": list(args.input_months),
        "target_months": list(args.target_months),
        "models": list(models),
        "t2m_source_note": dataset.t2m_source_note,
        "model_hyperparameter_grids": {
            "ridge": RIDGE_GRID,
            "random_forest": RF_GRID,
            "mlp": MLP_GRID,
        },
        "inner_cv_scoring": "negative mean squared error on training-fold standardized Y",
        "package_versions": package_versions,
        "selected_hyperparameters_preview": hyperparameter_df.head(10).to_dict(orient="records"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_warning_log(output_dir: Path, warning_df: pd.DataFrame) -> None:
    warning_path = output_dir / "mlp_warnings.log"
    if warning_df.empty:
        warning_path.write_text("No warnings captured.\n", encoding="utf-8")
        return
    lines = []
    for row in warning_df.itertuples(index=False):
        lines.append(
            f"model={row.model} stage={row.stage} outer_water_year={row.outer_water_year} "
            f"params={row.params_json} {row.warning_category}: {row.warning_message}"
        )
    warning_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_dataset_summary(dataset: DatasetBundle) -> None:
    print(f"number of water years: {dataset.water_years.size}", flush=True)
    print(f"first water year: {int(dataset.water_years.min())}", flush=True)
    print(f"last water year: {int(dataset.water_years.max())}", flush=True)
    print(f"X shape: {dataset.x.shape}", flush=True)
    print(f"Y shape: {dataset.y.shape}", flush=True)
    print(f"T2m anomaly source: {dataset.t2m_source_note}", flush=True)
    if dataset.dropped_years:
        print("years skipped and why:", flush=True)
        for row in dataset.dropped_years:
            print(f"  WY{row['water_year']}: {row['reason']}", flush=True)
    else:
        print("years skipped and why: none", flush=True)
    print(
        f"missing value counts: X={int(np.isnan(dataset.x).sum())}, Y={int(np.isnan(dataset.y).sum())}",
        flush=True,
    )
    print("semantic region mapping used for targets:", flush=True)
    for region in dataset.region_definitions:
        print(
            f"  region{region.semantic_label}={region.semantic_name} "
            f"<- cleaned_label {region.source_cleaned_label} size={region.size} "
            f"centroid=({region.centroid_lat:.3f}, {region.centroid_lon:.3f})",
            flush=True,
        )


def print_final_summary(
    output_dir: Path,
    dataset_csv: Path,
    plots_dir: Path,
    overall_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
) -> None:
    print("Experiment completed.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Dataset file: {dataset_csv}", flush=True)
    print(
        "Metrics files: "
        f"{output_dir / 'metrics_overall.csv'}, "
        f"{output_dir / 'metrics_by_region.csv'}, "
        f"{output_dir / 'metrics_by_month.csv'}, "
        f"{output_dir / 'metrics_region_by_month.csv'}",
        flush=True,
    )
    print(
        "Prediction files: "
        f"{output_dir / 'loyo_predictions_long.csv'}, "
        f"{output_dir / 'loyo_predictions_wide.csv'}",
        flush=True,
    )
    print(f"Plot directory: {plots_dir}", flush=True)
    print("Compact summary:", flush=True)
    print("model, overall_R2, overall_corr, overall_RMSE, overall_MAE", flush=True)
    for row in overall_metrics.sort_values("model").itertuples(index=False):
        print(f"{row.model}, {row.r2:.4f}, {row.corr:.4f}, {row.rmse:.4f}, {row.mae:.4f}", flush=True)

    best_overall = overall_metrics.sort_values("r2", ascending=False).iloc[0]
    print(
        f"Best model by overall R2: {best_overall['model']} ({best_overall['r2']:.4f})",
        flush=True,
    )
    print("Best model by region:", flush=True)
    for region_label in sorted(region_metrics["region_label"].unique()):
        region_block = region_metrics[region_metrics["region_label"] == region_label].sort_values("r2", ascending=False).iloc[0]
        print(
            f"  region{int(region_label)} {region_block['region_name']}: "
            f"{region_block['model']} ({region_block['r2']:.4f})",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifact_candidates = candidate_files_from_dir(args.artifact_dir)
    label_candidates = candidate_files_from_dir(args.label_dir)
    route2_netcdf = locate_route2_netcdf(args.artifact_dir)
    label_netcdf = locate_label_netcdf(args.label_dir)
    relevant_candidates = [
        route2_netcdf,
        EXPANDED_LEVEL2_NETCDF_FILE,
        label_netcdf,
        DEFAULT_MONTHLY_ANOMALY_FILE,
        ERA5_MONTHLY_MEAN_FILE,
        ERA5_MONTHLY_CLIM_FILE,
        COBE2_SST_FILE,
    ]
    print_candidate_group("Candidate files in artifact directory:", artifact_candidates)
    print_candidate_group("Candidate files in region-label directory:", label_candidates)
    print_candidate_group("Relevant data files reused from prior scripts:", relevant_candidates)

    dataset = build_supervised_dataset(route2_netcdf, label_netcdf, args.input_months, args.target_months)
    print_dataset_summary(dataset)
    dataset_csv = save_dataset_files(dataset, args.output_dir)

    predictions, hyperparameter_df, warning_df = run_loyo(dataset, args.models)
    hyperparameter_df.to_csv(args.output_dir / "selected_hyperparameters_by_fold.csv", index=False)
    write_warning_log(args.output_dir, warning_df)

    overall_metrics, region_metrics, month_metrics, region_month_metrics, metrics_summary = compute_all_metrics(
        dataset.water_years,
        dataset.y,
        predictions,
        dataset.region_definitions,
        args.target_months,
    )
    overall_metrics.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    region_metrics.to_csv(args.output_dir / "metrics_by_region.csv", index=False)
    month_metrics.to_csv(args.output_dir / "metrics_by_month.csv", index=False)
    region_month_metrics.to_csv(args.output_dir / "metrics_region_by_month.csv", index=False)
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n", encoding="utf-8")

    save_predictions(args.output_dir, dataset.water_years, dataset.y, predictions, dataset.region_definitions, args.target_months)

    plot_region_month_scatter_grid(
        plots_dir,
        region_month_metrics,
        dataset.y,
        predictions,
        dataset.region_definitions,
        args.target_months,
    )

    save_run_config(args.output_dir, args, dataset, args.models, hyperparameter_df)
    print_final_summary(args.output_dir, dataset_csv, plots_dir, overall_metrics, region_metrics)


if __name__ == "__main__":
    main()
