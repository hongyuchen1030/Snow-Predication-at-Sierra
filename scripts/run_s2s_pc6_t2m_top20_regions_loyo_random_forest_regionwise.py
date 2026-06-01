#!/usr/bin/env python3
"""
Run a regionwise LOYO Random Forest experiment for Sierra top-20% T2m regions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut


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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "s2s_pc6_t2m_top20_regions_loyo_random_forest_regionwise_decinit_fma"
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
INPUT_MONTH_TO_NUMBER = {"Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
TARGET_MONTH_TO_NUMBER = {"Feb": 2, "Mar": 3, "Apr": 4}
SEMANTIC_REGION_NAMES = {
    1: "coastal/left group",
    2: "inland/right group",
    3: "northern/top group",
}
MODEL_NAME = "random_forest_regionwise"
MODEL_DISPLAY = "Random Forest Regionwise"
MODEL_COLOR = "#bc6c25"
RF_PARAM_GRID: List[Dict[str, object]] = [
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": True,
    }
    for n_estimators in [300, 600]
    for max_depth in [2, 3, 4, 5, None]
    for min_samples_leaf in [1, 2, 4, 6]
    for max_features in ["sqrt", 0.5, 1.0]
]


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
    y_by_region: Dict[int, np.ndarray]
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
    parser.add_argument("--input-months", nargs="+", default=["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    parser.add_argument("--target-months", nargs="+", default=["Feb", "Mar", "Apr"])
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


def infer_semantic_region_order(cleaned_labels: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> List[RegionDefinition]:
    cleaned_ids = [int(value) for value in np.unique(cleaned_labels) if int(value) > 0]
    if len(cleaned_ids) != 3:
        raise ValueError(f"Expected exactly 3 cleaned labels, found {cleaned_ids}")

    raw_rows: List[Dict[str, object]] = []
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

    return [
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
                region.semantic_label: weighted_regional_mean(
                    anomaly_slice,
                    region_masks[region.semantic_label],
                    label_latitude,
                )
                for region in region_definitions
            }
            if index == 0 or (index + 1) % 120 == 0 or index + 1 == time_values.size:
                print(
                    f"  processed ERA5 monthly target {index + 1}/{time_values.size}: {month_key(time_value)}",
                    flush=True,
                )
    return monthly_targets, "computed anomalies from monthly mean minus monthly climatology"


def build_feature_names(input_months: Sequence[str]) -> List[str]:
    names: List[str] = []
    for month_name in input_months:
        for pc_index in range(1, 7):
            names.append(f"{month_name}_PC{pc_index}")
    return names


def build_target_names(region_definitions: Sequence[RegionDefinition], target_months: Sequence[str]) -> List[str]:
    names: List[str] = []
    for region in region_definitions:
        for month_name in target_months:
            names.append(f"region{region.semantic_label}_{month_name}")
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
    target_names = build_target_names(region_definitions, target_months)
    rows: List[Dict[str, object]] = []
    dropped_years: List[Dict[str, object]] = []

    for water_year in candidate_water_years:
        input_month_dates = [month_start(water_year - 1, INPUT_MONTH_TO_NUMBER[name]) for name in input_months]
        target_month_dates = [month_start(water_year, TARGET_MONTH_TO_NUMBER[name]) for name in target_months]
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
        for region in region_definitions:
            for target_key in target_keys:
                y_values.append(float(era5_monthly_targets[target_key][region.semantic_label]))
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
    y_by_region = {
        region.semantic_label: df[[f"region{region.semantic_label}_{month_name}" for month_name in target_months]].to_numpy(dtype=np.float64)
        for region in region_definitions
    }
    return DatasetBundle(
        water_years=water_years,
        x=x,
        y=y,
        y_by_region=y_by_region,
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


def inner_cv_splitter(n_train: int):
    if n_train <= 8:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_train), shuffle=False)


def fit_random_forest(
    params: Dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        bootstrap=bool(params["bootstrap"]),
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_eval), dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    return pred, importances


def score_params(params: Dict[str, object], x_train: np.ndarray, y_train: np.ndarray) -> float:
    splitter = inner_cv_splitter(x_train.shape[0])
    fold_scores: List[float] = []
    for inner_train_index, inner_valid_index in splitter.split(x_train):
        pred, _ = fit_random_forest(
            params,
            x_train[inner_train_index],
            y_train[inner_train_index],
            x_train[inner_valid_index],
        )
        fold_scores.append(-mean_squared_error(y_train[inner_valid_index], pred))
    return float(np.mean(fold_scores))


def run_loyo_regionwise(dataset: DatasetBundle) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    water_years = dataset.water_years
    x = dataset.x
    predictions_by_region = {
        region.semantic_label: np.full_like(dataset.y_by_region[region.semantic_label], np.nan, dtype=np.float64)
        for region in dataset.region_definitions
    }
    hyperparameter_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    print("Random Forest is trained on unscaled features and unscaled targets.", flush=True)
    print(
        f"Starting regionwise LOYO random-forest sweep across {water_years.size} water years and {len(dataset.region_definitions)} regions",
        flush=True,
    )
    for outer_index, test_water_year in enumerate(water_years):
        train_mask = water_years != test_water_year
        x_train = x[train_mask]
        x_test = x[~train_mask]
        if x_test.shape[0] != 1:
            raise ValueError("Expected exactly one held-out sample per LOYO fold.")

        for region in dataset.region_definitions:
            y_region = dataset.y_by_region[region.semantic_label]
            y_train = y_region[train_mask]
            best_params = None
            best_score = -np.inf
            for params in RF_PARAM_GRID:
                score = score_params(params, x_train, y_train)
                if score > best_score:
                    best_score = score
                    best_params = params
            if best_params is None:
                raise RuntimeError(f"No hyperparameters selected for region {region.semantic_label} WY{int(test_water_year)}")
            pred, importances = fit_random_forest(best_params, x_train, y_train, x_test)
            predictions_by_region[region.semantic_label][outer_index, :] = pred[0]
            hyperparameter_rows.append(
                {
                    "outer_water_year": int(test_water_year),
                    "region_label": region.semantic_label,
                    "region_name": region.semantic_name,
                    "selected_params_json": json.dumps(best_params, sort_keys=True),
                    "inner_cv_score_neg_mean_mse": float(best_score),
                    "n_estimators": int(best_params["n_estimators"]),
                    "max_depth": "None" if best_params["max_depth"] is None else int(best_params["max_depth"]),
                    "min_samples_leaf": int(best_params["min_samples_leaf"]),
                    "max_features": str(best_params["max_features"]),
                    "bootstrap": bool(best_params["bootstrap"]),
                }
            )
            for feature_name, importance in zip(dataset.feature_names, importances.tolist()):
                importance_rows.append(
                    {
                        "outer_water_year": int(test_water_year),
                        "region_label": region.semantic_label,
                        "region_name": region.semantic_name,
                        "feature_name": feature_name,
                        "importance": float(importance),
                    }
                )
            print(
                f"LOYO held_out_WY={int(test_water_year)} region={region.semantic_label} params={best_params} "
                f"inner_score={best_score:.6f}",
                flush=True,
            )

    ordered_predictions = np.concatenate(
        [predictions_by_region[region.semantic_label] for region in dataset.region_definitions],
        axis=1,
    )
    hyperparameter_df = pd.DataFrame(hyperparameter_rows).sort_values(["region_label", "outer_water_year"]).reset_index(drop=True)
    importance_df = pd.DataFrame(importance_rows)
    print("Finished regionwise LOYO random-forest sweep", flush=True)
    return ordered_predictions, hyperparameter_df, importance_df


def reshape_targets(y: np.ndarray, n_regions: int, n_months: int) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(y.shape[0], n_regions, n_months)


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


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    n_regions = len(region_definitions)
    n_months = len(target_months)
    y_true_reshaped = reshape_targets(y_true, n_regions=n_regions, n_months=n_months)
    y_pred_reshaped = reshape_targets(y_pred, n_regions=n_regions, n_months=n_months)

    overall = compute_metric_block(y_true.reshape(-1), y_pred.reshape(-1))
    overall_df = pd.DataFrame([{"model": MODEL_NAME, **overall}])

    region_rows: List[Dict[str, object]] = []
    for region_index, region in enumerate(region_definitions):
        block = compute_metric_block(
            y_true_reshaped[:, region_index, :].reshape(-1),
            y_pred_reshaped[:, region_index, :].reshape(-1),
        )
        region_rows.append(
            {
                "model": MODEL_NAME,
                "region_label": region.semantic_label,
                "region_name": region.semantic_name,
                **block,
            }
        )
    region_df = pd.DataFrame(region_rows)

    month_rows: List[Dict[str, object]] = []
    for month_index, month_name in enumerate(target_months):
        block = compute_metric_block(
            y_true_reshaped[:, :, month_index].reshape(-1),
            y_pred_reshaped[:, :, month_index].reshape(-1),
        )
        month_rows.append({"model": MODEL_NAME, "target_month": month_name, **block})
    month_df = pd.DataFrame(month_rows)

    region_month_rows: List[Dict[str, object]] = []
    for region_index, region in enumerate(region_definitions):
        for month_index, month_name in enumerate(target_months):
            block = compute_metric_block(
                y_true_reshaped[:, region_index, month_index],
                y_pred_reshaped[:, region_index, month_index],
            )
            region_month_rows.append(
                {
                    "model": MODEL_NAME,
                    "region_label": region.semantic_label,
                    "region_name": region.semantic_name,
                    "target_month": month_name,
                    **block,
                }
            )
    region_month_df = pd.DataFrame(region_month_rows)

    summary_json = {
        "model": MODEL_NAME,
        "overall": overall,
        "by_region": region_rows,
        "by_month": month_rows,
        "region_by_month": region_month_rows,
    }
    return overall_df, region_df, month_df, region_month_df, summary_json


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
    y_pred: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> Tuple[Path, Path]:
    n_regions = len(region_definitions)
    n_months = len(target_months)
    true_reshaped = reshape_targets(y_true, n_regions=n_regions, n_months=n_months)
    pred_reshaped = reshape_targets(y_pred, n_regions=n_regions, n_months=n_months)
    rows_long: List[Dict[str, object]] = []
    rows_wide: List[Dict[str, object]] = []

    for row_index, water_year in enumerate(water_years.tolist()):
        row_wide: Dict[str, object] = {"model": MODEL_NAME, "water_year": int(water_year)}
        for region_index, region in enumerate(region_definitions):
            for month_index, month_name in enumerate(target_months):
                column_key = f"region{region.semantic_label}_{month_name}"
                y_true_value = float(true_reshaped[row_index, region_index, month_index])
                y_pred_value = float(pred_reshaped[row_index, region_index, month_index])
                rows_long.append(
                    {
                        "model": MODEL_NAME,
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
    np.save(output_dir / "y_pred_random_forest_regionwise.npy", y_pred)
    return long_path, wide_path


def save_feature_importance_summary(
    output_dir: Path,
    importance_df: pd.DataFrame,
    region_definitions: Sequence[RegionDefinition],
) -> pd.DataFrame:
    grouped = (
        importance_df.groupby(["region_label", "feature_name"], as_index=False)["importance"]
        .agg(mean_importance="mean", std_importance="std")
        .fillna({"std_importance": 0.0})
    )
    region_name_map = {region.semantic_label: region.semantic_name for region in region_definitions}
    grouped["region_name"] = grouped["region_label"].map(region_name_map)
    grouped = grouped[["region_label", "region_name", "feature_name", "mean_importance", "std_importance"]]
    grouped.to_csv(output_dir / "feature_importance_by_region.csv", index=False)
    return grouped


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
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Observed T2m anomaly")
    ax.set_ylabel("Predicted T2m anomaly")
    ax.set_title(title)


def plot_overall_scatter(plots_dir: Path, overall_metrics: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    row = overall_metrics.iloc[0]
    fig, ax = plt.subplots(figsize=(6.3, 6.1), constrained_layout=True)
    scatter_with_identity(
        ax,
        y_true.reshape(-1),
        y_pred.reshape(-1),
        MODEL_COLOR,
        f"Overall pooled\nR2={row['r2']:.3f}, corr={row['corr']:.3f}, RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}",
    )
    fig.savefig(plots_dir / "random_forest_regionwise_overall_scatter.png", dpi=220)
    fig.savefig(plots_dir / "random_forest_regionwise_overall_scatter.pdf")
    plt.close(fig)


def plot_region_scatter(
    plots_dir: Path,
    region_metrics: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    y_true_reshaped = reshape_targets(y_true, n_regions=len(region_definitions), n_months=len(target_months))
    y_pred_reshaped = reshape_targets(y_pred, n_regions=len(region_definitions), n_months=len(target_months))
    fig, axes = plt.subplots(1, len(region_definitions), figsize=(5.0 * len(region_definitions), 4.8), constrained_layout=True)
    for region_index, region in enumerate(region_definitions):
        row = region_metrics[region_metrics["region_label"] == region.semantic_label].iloc[0]
        scatter_with_identity(
            axes[region_index],
            y_true_reshaped[:, region_index, :].reshape(-1),
            y_pred_reshaped[:, region_index, :].reshape(-1),
            MODEL_COLOR,
            (
                f"Region {region.semantic_label}: {region.semantic_name}\n"
                f"R2={row['r2']:.3f}, corr={row['corr']:.3f}, RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}"
            ),
        )
    fig.savefig(plots_dir / "random_forest_regionwise_regions_scatter.png", dpi=220)
    fig.savefig(plots_dir / "random_forest_regionwise_regions_scatter.pdf")
    plt.close(fig)


def plot_month_scatter(
    plots_dir: Path,
    month_metrics: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    y_true_reshaped = reshape_targets(y_true, n_regions=len(region_definitions), n_months=len(target_months))
    y_pred_reshaped = reshape_targets(y_pred, n_regions=len(region_definitions), n_months=len(target_months))
    fig, axes = plt.subplots(1, len(target_months), figsize=(5.0 * len(target_months), 4.8), constrained_layout=True)
    for month_index, month_name in enumerate(target_months):
        row = month_metrics[month_metrics["target_month"] == month_name].iloc[0]
        scatter_with_identity(
            axes[month_index],
            y_true_reshaped[:, :, month_index].reshape(-1),
            y_pred_reshaped[:, :, month_index].reshape(-1),
            MODEL_COLOR,
            f"{month_name}\nR2={row['r2']:.3f}, corr={row['corr']:.3f}, RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}",
        )
    fig.savefig(plots_dir / "random_forest_regionwise_months_scatter.png", dpi=220)
    fig.savefig(plots_dir / "random_forest_regionwise_months_scatter.pdf")
    plt.close(fig)


def plot_region_month_scatter_grid(
    plots_dir: Path,
    region_month_metrics: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    y_true_reshaped = reshape_targets(y_true, n_regions=len(region_definitions), n_months=len(target_months))
    y_pred_reshaped = reshape_targets(y_pred, n_regions=len(region_definitions), n_months=len(target_months))
    fig, axes = plt.subplots(
        len(region_definitions),
        len(target_months),
        figsize=(4.0 * len(target_months), 3.8 * len(region_definitions)),
        constrained_layout=True,
    )
    for region_index, region in enumerate(region_definitions):
        for month_index, month_name in enumerate(target_months):
            row = region_month_metrics[
                (region_month_metrics["region_label"] == region.semantic_label)
                & (region_month_metrics["target_month"] == month_name)
            ].iloc[0]
            ax = axes[region_index, month_index]
            scatter_with_identity(
                ax,
                y_true_reshaped[:, region_index, month_index],
                y_pred_reshaped[:, region_index, month_index],
                MODEL_COLOR,
                (
                    f"Region {region.semantic_label} | {month_name}\n"
                    f"R2={row['r2']:.3f}, corr={row['corr']:.3f}\n"
                    f"RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}"
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
    fig.suptitle("Regionwise Random Forest observed vs predicted by region and month", fontsize=14)
    fig.savefig(plots_dir / "random_forest_regionwise_region_month_scatter_grid.png", dpi=220)
    fig.savefig(plots_dir / "random_forest_regionwise_region_month_scatter_grid.pdf")
    plt.close(fig)


def plot_time_series(
    plots_dir: Path,
    water_years: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    y_true_reshaped = reshape_targets(y_true, n_regions=len(region_definitions), n_months=len(target_months))
    y_pred_reshaped = reshape_targets(y_pred, n_regions=len(region_definitions), n_months=len(target_months))
    for region_index, region in enumerate(region_definitions):
        fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
        for month_index, month_name in enumerate(target_months):
            color = plt.cm.tab10(month_index)
            ax.plot(
                water_years,
                y_true_reshaped[:, region_index, month_index],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=f"{month_name} observed",
            )
            ax.plot(
                water_years,
                y_pred_reshaped[:, region_index, month_index],
                color=color,
                linewidth=1.4,
                linestyle="--",
                marker="x",
                markersize=3.5,
                label=f"{month_name} predicted",
            )
        ax.set_title(f"Region {region.semantic_label}: {region.semantic_name}")
        ax.set_xlabel("Water year")
        ax.set_ylabel("T2m anomaly")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=9)
        fig.savefig(plots_dir / f"random_forest_regionwise_time_series_region{region.semantic_label}.png", dpi=220)
        fig.savefig(plots_dir / f"random_forest_regionwise_time_series_region{region.semantic_label}.pdf")
        plt.close(fig)


def plot_region_month_heatmap(
    plots_dir: Path,
    region_month_metrics: pd.DataFrame,
    region_definitions: Sequence[RegionDefinition],
    target_months: Sequence[str],
) -> None:
    pivot = (
        region_month_metrics.pivot(index="region_label", columns="target_month", values="r2")
        .loc[[region.semantic_label for region in region_definitions], list(target_months)]
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    image = ax.imshow(pivot.values, cmap="coolwarm", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(target_months)))
    ax.set_xticklabels(list(target_months))
    ax.set_yticks(np.arange(len(region_definitions)))
    ax.set_yticklabels([f"Region {region.semantic_label}" for region in region_definitions])
    ax.set_title("Regionwise Random Forest region-by-month R2")
    for row_index in range(pivot.shape[0]):
        for col_index in range(pivot.shape[1]):
            ax.text(col_index, row_index, f"{pivot.values[row_index, col_index]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.9).set_label("R2")
    fig.savefig(plots_dir / "random_forest_regionwise_region_month_r2_heatmap.png", dpi=220)
    fig.savefig(plots_dir / "random_forest_regionwise_region_month_r2_heatmap.pdf")
    plt.close(fig)


def plot_feature_importance(
    plots_dir: Path,
    feature_importance_df: pd.DataFrame,
    region_definitions: Sequence[RegionDefinition],
) -> None:
    for region in region_definitions:
        subset = feature_importance_df[feature_importance_df["region_label"] == region.semantic_label].sort_values(
            "mean_importance",
            ascending=False,
        )
        fig, ax = plt.subplots(figsize=(10.5, 9.5), constrained_layout=True)
        ax.barh(subset["feature_name"], subset["mean_importance"], xerr=subset["std_importance"], color=MODEL_COLOR, alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Mean feature importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Region {region.semantic_label}: {region.semantic_name}")
        ax.grid(True, axis="x", alpha=0.25)
        fig.savefig(plots_dir / f"random_forest_regionwise_feature_importance_region{region.semantic_label}.png", dpi=220)
        fig.savefig(plots_dir / f"random_forest_regionwise_feature_importance_region{region.semantic_label}.pdf")
        plt.close(fig)


def plot_selected_hyperparameters(
    plots_dir: Path,
    hyperparameter_df: pd.DataFrame,
    region_definitions: Sequence[RegionDefinition],
) -> None:
    param_specs = [
        ("n_estimators", "n_estimators"),
        ("max_depth", "max_depth"),
        ("min_samples_leaf", "min_samples_leaf"),
        ("max_features", "max_features"),
    ]
    fig, axes = plt.subplots(len(region_definitions), len(param_specs), figsize=(4.0 * len(param_specs), 3.5 * len(region_definitions)), constrained_layout=True)
    if len(region_definitions) == 1:
        axes = np.expand_dims(axes, axis=0)
    for region_index, region in enumerate(region_definitions):
        subset = hyperparameter_df[hyperparameter_df["region_label"] == region.semantic_label]
        for param_index, (column_name, label) in enumerate(param_specs):
            ax = axes[region_index, param_index]
            counts = subset[column_name].astype(str).value_counts().sort_index()
            x_values = np.arange(counts.shape[0])
            ax.bar(x_values, counts.values, color=MODEL_COLOR, alpha=0.85)
            ax.set_xticks(x_values)
            ax.set_xticklabels(counts.index.tolist(), rotation=45, ha="right")
            if region_index == 0:
                ax.set_title(label)
            if param_index == 0:
                ax.set_ylabel(f"Region {region.semantic_label}\nFold count")
            else:
                ax.set_ylabel("")
            ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(plots_dir / "random_forest_regionwise_selected_hyperparameters.png", dpi=220)
    fig.savefig(plots_dir / "random_forest_regionwise_selected_hyperparameters.pdf")
    plt.close(fig)


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: DatasetBundle,
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
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "hyperparameter_grid": RF_PARAM_GRID,
        "cv": "leave-one-water-year-out",
        "mode": "regionwise multi-output random forest",
        "x_scaling": "none",
        "y_scaling": "none",
        "t2m_source_note": dataset.t2m_source_note,
        "package_versions": package_versions,
        "selected_hyperparameters_preview": hyperparameter_df.head(12).to_dict(orient="records"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_dataset_summary(dataset: DatasetBundle) -> None:
    print(f"number of water years: {dataset.water_years.size}", flush=True)
    print(f"first water year: {int(dataset.water_years.min())}", flush=True)
    print(f"last water year: {int(dataset.water_years.max())}", flush=True)
    print(f"X shape: {dataset.x.shape}", flush=True)
    print(f"Y shape: {dataset.y.shape}", flush=True)
    print(f"T2m anomaly source: {dataset.t2m_source_note}", flush=True)
    print("Random Forest uses raw X and raw Y without scaling.", flush=True)
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
    dataset: DatasetBundle,
    overall_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    month_metrics: pd.DataFrame,
    hyperparameter_df: pd.DataFrame,
) -> None:
    overall_row = overall_metrics.iloc[0]
    print("Experiment completed.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Dataset shape: X = {dataset.x.shape}, Y = {dataset.y.shape}", flush=True)
    print(
        f"Water years used: {int(dataset.water_years.min())}-{int(dataset.water_years.max())} "
        f"({dataset.water_years.size} total)",
        flush=True,
    )
    print("Overall metrics summary:", flush=True)
    print(
        f"  R2={overall_row['r2']:.4f}, corr={overall_row['corr']:.4f}, "
        f"RMSE={overall_row['rmse']:.4f}, MAE={overall_row['mae']:.4f}",
        flush=True,
    )
    print("Metrics by region:", flush=True)
    for row in region_metrics.itertuples(index=False):
        print(
            f"  region{row.region_label}, R2={row.r2:.4f}, corr={row.corr:.4f}, RMSE={row.rmse:.4f}, MAE={row.mae:.4f}",
            flush=True,
        )
    print("Metrics by month:", flush=True)
    for row in month_metrics.itertuples(index=False):
        print(
            f"  {row.target_month}, R2={row.r2:.4f}, corr={row.corr:.4f}, RMSE={row.rmse:.4f}, MAE={row.mae:.4f}",
            flush=True,
        )
    print("Selected hyperparameter summary by region:", flush=True)
    for region in dataset.region_definitions:
        subset = hyperparameter_df[hyperparameter_df["region_label"] == region.semantic_label]
        n_est = subset["n_estimators"].astype(str).value_counts().sort_index()
        depth = subset["max_depth"].astype(str).value_counts().sort_index()
        leaf = subset["min_samples_leaf"].astype(str).value_counts().sort_index()
        mfeat = subset["max_features"].astype(str).value_counts().sort_index()
        print(f"  region{region.semantic_label}:", flush=True)
        print(f"    n_estimators: {', '.join([f'{k}:{v}' for k, v in n_est.items()])}", flush=True)
        print(f"    max_depth: {', '.join([f'{k}:{v}' for k, v in depth.items()])}", flush=True)
        print(f"    min_samples_leaf: {', '.join([f'{k}:{v}' for k, v in leaf.items()])}", flush=True)
        print(f"    max_features: {', '.join([f'{k}:{v}' for k, v in mfeat.items()])}", flush=True)


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
    save_dataset_files(dataset, args.output_dir)

    y_pred, hyperparameter_df, raw_importance_df = run_loyo_regionwise(dataset)
    hyperparameter_df.to_csv(args.output_dir / "selected_hyperparameters_by_fold.csv", index=False)
    feature_importance_df = save_feature_importance_summary(args.output_dir, raw_importance_df, dataset.region_definitions)

    overall_metrics, region_metrics, month_metrics, region_month_metrics, metrics_summary = compute_all_metrics(
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

    save_predictions(
        args.output_dir,
        dataset.water_years,
        dataset.y,
        y_pred,
        dataset.region_definitions,
        args.target_months,
    )

    plot_overall_scatter(plots_dir, overall_metrics, dataset.y, y_pred)
    plot_region_month_scatter_grid(plots_dir, region_month_metrics, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    plot_region_scatter(plots_dir, region_metrics, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    plot_month_scatter(plots_dir, month_metrics, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    plot_time_series(plots_dir, dataset.water_years, dataset.y, y_pred, dataset.region_definitions, args.target_months)
    plot_region_month_heatmap(plots_dir, region_month_metrics, dataset.region_definitions, args.target_months)
    plot_feature_importance(plots_dir, feature_importance_df, dataset.region_definitions)
    plot_selected_hyperparameters(plots_dir, hyperparameter_df, dataset.region_definitions)

    save_run_config(args.output_dir, args, dataset, hyperparameter_df)
    print_final_summary(args.output_dir, dataset, overall_metrics, region_metrics, month_metrics, hyperparameter_df)


if __name__ == "__main__":
    main()
