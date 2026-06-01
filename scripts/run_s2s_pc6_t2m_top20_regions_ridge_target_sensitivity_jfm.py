#!/usr/bin/env python3
"""
Run the corrected Ridge-only target sensitivity experiment for Sierra top-20
predictability cells using Jun-Nov SST PCs and Jan-Mar T2m anomaly targets.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.run_s2s_pc6_t2m_top20_regions_loyo_models as base_mod


DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
)
DEFAULT_LABEL_DIR = DEFAULT_ARTIFACT_DIR / "top20_region_labels"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm"
)
INPUT_MONTH_TO_NUMBER = {"Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11}
TARGET_MONTH_TO_NUMBER = {"Jan": 1, "Feb": 2, "Mar": 3}
RIDGE_ALPHA_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
REGION_NAME_BY_LABEL = {1: "coastal_left", 2: "inland_right", 3: "northern_top"}
TOP20_NAME = "top20_all"
TOP20_COLOR = "#355070"
REGIONWISE_COLOR = "#6d597a"


@dataclass(frozen=True)
class DatasetBundle:
    water_years: np.ndarray
    x: np.ndarray
    y_top20_all: np.ndarray
    y_regionwise: np.ndarray
    feature_names: List[str]
    target_names_top20_all: List[str]
    target_names_regionwise: List[str]
    region_definitions: List[base_mod.RegionDefinition]
    alignment_rows: List[Dict[str, object]]
    dropped_years: List[Dict[str, object]]
    paths_used: Dict[str, str]
    t2m_source_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-months", nargs="+", default=["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"])
    parser.add_argument("--target-months", nargs="+", default=["Jan", "Feb", "Mar"])
    parser.add_argument("--train-all-debug", action="store_true")
    return parser.parse_args()


def build_feature_names(input_months: Sequence[str]) -> List[str]:
    return [f"{month_name}_PC{pc_index}" for month_name in input_months for pc_index in range(1, 7)]


def build_target_names_top20_all(target_months: Sequence[str]) -> List[str]:
    return [f"{TOP20_NAME}_{month_name}" for month_name in target_months]


def build_target_names_regionwise(
    region_definitions: Sequence[base_mod.RegionDefinition],
    target_months: Sequence[str],
) -> List[str]:
    names: List[str] = []
    for region in region_definitions:
        region_name = REGION_NAME_BY_LABEL[region.semantic_label]
        for month_name in target_months:
            names.append(f"{region_name}_{month_name}")
    return names


def weighted_mask_mean(field_2d: np.ndarray, mask_2d: np.ndarray, latitude: np.ndarray) -> float:
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, np.newaxis], mask_2d.shape)
    valid = mask_2d & np.isfinite(field_2d)
    if not np.any(valid):
        return float("nan")
    weighted_sum = np.sum(field_2d[valid] * weights[valid])
    weight_sum = np.sum(weights[valid])
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def load_monthly_targets(
    label_latitude: np.ndarray,
    label_longitude: np.ndarray,
    cleaned_labels: np.ndarray,
    region_definitions: Sequence[base_mod.RegionDefinition],
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], str]:
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
        if monthly_mean.shape[1:] != cleaned_labels.shape:
            raise ValueError(
                f"Monthly mean subset shape {monthly_mean.shape[1:]} does not match label grid {cleaned_labels.shape}"
            )
        if monthly_clim.shape[1:] != cleaned_labels.shape:
            raise ValueError(
                f"Monthly climatology subset shape {monthly_clim.shape[1:]} does not match label grid {cleaned_labels.shape}"
            )

        time_values = np.asarray(monthly_mean["time"].values, dtype="datetime64[ns]")
        month_numbers = np.asarray(monthly_mean["time"].dt.month.values, dtype=np.int32)
        climatology_values = np.asarray(monthly_clim.values, dtype=np.float64)

        top20_mask = cleaned_labels > 0
        region_masks = {
            region.semantic_label: cleaned_labels == region.source_cleaned_label for region in region_definitions
        }
        monthly_top20_all: Dict[str, float] = {}
        monthly_regionwise: Dict[str, Dict[int, float]] = {}

        for index, (time_value, month_value) in enumerate(zip(time_values, month_numbers.tolist())):
            monthly_mean_slice = np.asarray(monthly_mean.isel(time=index).load().values, dtype=np.float64)
            monthly_clim_slice = climatology_values[month_value - 1, :, :]
            anomaly_slice = monthly_mean_slice - monthly_clim_slice
            month_id = base_mod.month_key(time_value)
            monthly_top20_all[month_id] = weighted_mask_mean(anomaly_slice, top20_mask, label_latitude)
            monthly_regionwise[month_id] = {
                region.semantic_label: weighted_mask_mean(
                    anomaly_slice,
                    region_masks[region.semantic_label],
                    label_latitude,
                )
                for region in region_definitions
            }
            if index == 0 or (index + 1) % 120 == 0 or index + 1 == time_values.size:
                print(
                    f"  processed ERA5 monthly target {index + 1}/{time_values.size}: {month_id}",
                    flush=True,
                )

    return monthly_top20_all, monthly_regionwise, "computed anomalies from monthly mean minus monthly climatology"


def build_supervised_dataset(
    route2_netcdf: Path,
    label_netcdf: Path,
    input_months: Sequence[str],
    target_months: Sequence[str],
) -> DatasetBundle:
    print(f"Loading cleaned region labels from {label_netcdf}", flush=True)
    label_latitude, label_longitude, cleaned_labels, region_definitions = base_mod.load_label_mask(label_netcdf)
    print(f"Loading SST PC1-PC6 from {base_mod.EXPANDED_LEVEL2_NETCDF_FILE}", flush=True)
    pc_time, pc_values = base_mod.load_pacific_pc_timeseries(base_mod.EXPANDED_LEVEL2_NETCDF_FILE)
    print("Loading ERA5-Land monthly mean/climatology and computing corrected T2m anomaly targets", flush=True)
    monthly_top20_all, monthly_regionwise, t2m_source_note = load_monthly_targets(
        label_latitude,
        label_longitude,
        cleaned_labels,
        region_definitions,
    )

    pc_time = base_mod.to_month_start(pc_time)
    pc_index = base_mod.build_month_lookup(pc_time)
    common_times = sorted(set(pc_index).intersection(set(monthly_top20_all)))
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
    target_names_top20_all = build_target_names_top20_all(target_months)
    target_names_regionwise = build_target_names_regionwise(region_definitions, target_months)

    rows: List[Dict[str, object]] = []
    alignment_rows: List[Dict[str, object]] = []
    dropped_years: List[Dict[str, object]] = []

    for water_year in candidate_water_years:
        input_month_dates = [base_mod.month_start(water_year - 1, INPUT_MONTH_TO_NUMBER[name]) for name in input_months]
        target_month_dates = [base_mod.month_start(water_year, TARGET_MONTH_TO_NUMBER[name]) for name in target_months]
        input_keys = [base_mod.month_key(value) for value in input_month_dates]
        target_keys = [base_mod.month_key(value) for value in target_month_dates]
        alignment_rows.append(
            {
                "water_year": water_year,
                "input_months": ", ".join(input_keys),
                "target_months": ", ".join(target_keys),
            }
        )

        missing_inputs = [value for value in input_keys if value not in pc_index]
        missing_targets = [value for value in target_keys if value not in monthly_top20_all]
        if missing_inputs or missing_targets:
            dropped_years.append(
                {
                    "water_year": water_year,
                    "reason": {"missing_inputs": missing_inputs, "missing_targets": missing_targets},
                }
            )
            continue

        x_matrix = np.stack([pc_values[pc_index[value], :] for value in input_keys], axis=0)
        x_vector = x_matrix.reshape(-1)
        y_top20_all = np.asarray([monthly_top20_all[key] for key in target_keys], dtype=np.float64)
        y_regionwise_values: List[float] = []
        for region in region_definitions:
            for target_key in target_keys:
                y_regionwise_values.append(float(monthly_regionwise[target_key][region.semantic_label]))
        y_regionwise = np.asarray(y_regionwise_values, dtype=np.float64)

        if np.isnan(x_vector).any() or np.isnan(y_top20_all).any() or np.isnan(y_regionwise).any():
            dropped_years.append(
                {
                    "water_year": water_year,
                    "reason": {
                        "input_nan_count": int(np.isnan(x_vector).sum()),
                        "target_nan_count_top20_all": int(np.isnan(y_top20_all).sum()),
                        "target_nan_count_regionwise": int(np.isnan(y_regionwise).sum()),
                    },
                }
            )
            continue

        row: Dict[str, object] = {"water_year": water_year}
        row.update({name: float(value) for name, value in zip(feature_names, x_vector.tolist())})
        row.update({name: float(value) for name, value in zip(target_names_top20_all, y_top20_all.tolist())})
        row.update({name: float(value) for name, value in zip(target_names_regionwise, y_regionwise.tolist())})
        rows.append(row)

    if not rows:
        raise ValueError(f"No complete water years were available after screening. First drops: {dropped_years[:5]}")

    df = pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)
    print(f"Finished dataset assembly with {len(df)} usable water years", flush=True)

    return DatasetBundle(
        water_years=df["water_year"].to_numpy(dtype=np.int32),
        x=df[feature_names].to_numpy(dtype=np.float64),
        y_top20_all=df[target_names_top20_all].to_numpy(dtype=np.float64),
        y_regionwise=df[target_names_regionwise].to_numpy(dtype=np.float64),
        feature_names=feature_names,
        target_names_top20_all=target_names_top20_all,
        target_names_regionwise=target_names_regionwise,
        region_definitions=region_definitions,
        alignment_rows=alignment_rows,
        dropped_years=dropped_years,
        paths_used={
            "route2_netcdf": str(route2_netcdf),
            "expanded_level2_netcdf": str(base_mod.EXPANDED_LEVEL2_NETCDF_FILE),
            "label_netcdf": str(label_netcdf),
            "monthly_mean_file": str(base_mod.ERA5_MONTHLY_MEAN_FILE),
            "monthly_climatology_file": str(base_mod.ERA5_MONTHLY_CLIM_FILE),
            "cobe2_sst_file": str(base_mod.COBE2_SST_FILE),
        },
        t2m_source_note=t2m_source_note,
    )


def print_dataset_summary(dataset: DatasetBundle) -> None:
    print(f"number of water years: {dataset.water_years.size}", flush=True)
    print(f"first water year: {int(dataset.water_years.min())}", flush=True)
    print(f"last water year: {int(dataset.water_years.max())}", flush=True)
    print(f"X shape: {dataset.x.shape}", flush=True)
    print(f"Y_top20_all shape: {dataset.y_top20_all.shape}", flush=True)
    print(f"Y_regionwise shape: {dataset.y_regionwise.shape}", flush=True)
    print(f"T2m anomaly source: {dataset.t2m_source_note}", flush=True)
    print(
        f"missing value counts: X={int(np.isnan(dataset.x).sum())}, "
        f"Y_top20_all={int(np.isnan(dataset.y_top20_all).sum())}, "
        f"Y_regionwise={int(np.isnan(dataset.y_regionwise).sum())}",
        flush=True,
    )
    print("semantic region mapping used for targets:", flush=True)
    for region in dataset.region_definitions:
        print(
            f"  region{region.semantic_label}={REGION_NAME_BY_LABEL[region.semantic_label]} "
            f"<- cleaned_label {region.source_cleaned_label} size={region.size} "
            f"centroid=({region.centroid_lat:.3f}, {region.centroid_lon:.3f})",
            flush=True,
        )
    print("water-year alignment examples:", flush=True)
    for row in dataset.alignment_rows[:5]:
        print(
            f"  WY{row['water_year']}: inputs [{row['input_months']}] -> targets [{row['target_months']}]",
            flush=True,
        )
    if dataset.dropped_years:
        print("years skipped and why:", flush=True)
        for row in dataset.dropped_years[:10]:
            print(f"  WY{row['water_year']}: {row['reason']}", flush=True)
    else:
        print("years skipped and why: none", flush=True)


def save_dataset_files(dataset: DatasetBundle, output_dir: Path) -> None:
    df = pd.DataFrame(
        np.column_stack(
            [
                dataset.water_years[:, np.newaxis],
                dataset.x,
                dataset.y_top20_all,
                dataset.y_regionwise,
            ]
        ),
        columns=[
            "water_year",
            *dataset.feature_names,
            *dataset.target_names_top20_all,
            *dataset.target_names_regionwise,
        ],
    )
    df["water_year"] = df["water_year"].astype(int)
    df.to_csv(output_dir / "ridge_target_sensitivity_dataset.csv", index=False)
    pd.DataFrame(dataset.alignment_rows).to_csv(output_dir / "water_year_alignment_check.csv", index=False)
    np.save(output_dir / "X.npy", dataset.x)
    np.save(output_dir / "Y_top20_all.npy", dataset.y_top20_all)
    np.save(output_dir / "Y_regionwise.npy", dataset.y_regionwise)
    np.save(output_dir / "water_years.npy", dataset.water_years)
    (output_dir / "feature_names.json").write_text(json.dumps(dataset.feature_names, indent=2) + "\n", encoding="utf-8")
    (output_dir / "target_names_top20_all.json").write_text(
        json.dumps(dataset.target_names_top20_all, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "target_names_regionwise.json").write_text(
        json.dumps(dataset.target_names_regionwise, indent=2) + "\n",
        encoding="utf-8",
    )


def inner_cv_splitter(n_train: int):
    if n_train <= 8:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_train), shuffle=False)


def fit_scaled_model(
    alpha: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_eval_scaled = x_scaler.transform(x_eval)
    if float(alpha) <= 0.0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=float(alpha))
    model.fit(x_train_scaled, y_train_scaled)
    pred_scaled = np.asarray(model.predict(x_eval_scaled), dtype=np.float64)
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled[:, np.newaxis]
    return y_scaler.inverse_transform(pred_scaled)


def score_alpha(alpha: float, x_train: np.ndarray, y_train: np.ndarray) -> float:
    splitter = inner_cv_splitter(x_train.shape[0])
    fold_scores: List[float] = []
    for inner_train_index, inner_valid_index in splitter.split(x_train):
        x_inner_train = x_train[inner_train_index]
        y_inner_train = y_train[inner_train_index]
        x_inner_valid = x_train[inner_valid_index]
        y_inner_valid = y_train[inner_valid_index]

        x_scaler = StandardScaler().fit(x_inner_train)
        y_scaler = StandardScaler().fit(y_inner_train)
        x_inner_train_scaled = x_scaler.transform(x_inner_train)
        y_inner_train_scaled = y_scaler.transform(y_inner_train)
        x_inner_valid_scaled = x_scaler.transform(x_inner_valid)
        y_inner_valid_scaled = y_scaler.transform(y_inner_valid)

        if float(alpha) <= 0.0:
            model = LinearRegression()
        else:
            model = Ridge(alpha=float(alpha))
        model.fit(x_inner_train_scaled, y_inner_train_scaled)
        pred_scaled = np.asarray(model.predict(x_inner_valid_scaled), dtype=np.float64)
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled[:, np.newaxis]
        fold_scores.append(-mean_squared_error(y_inner_valid_scaled, pred_scaled))
    return float(np.mean(fold_scores))


def select_alpha(x_train: np.ndarray, y_train: np.ndarray, alpha_grid: Sequence[float]) -> Tuple[float, float]:
    best_alpha = None
    best_score = -np.inf
    for alpha in alpha_grid:
        score = score_alpha(alpha, x_train, y_train)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("No alpha selected.")
    return best_alpha, float(best_score)


def run_loyo(dataset: DatasetBundle, target_months: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    water_years = dataset.water_years
    y_pred_top20_all = np.full_like(dataset.y_top20_all, np.nan, dtype=np.float64)
    y_pred_regionwise = np.full_like(dataset.y_regionwise, np.nan, dtype=np.float64)
    hyperparameter_rows: List[Dict[str, object]] = []

    print(f"Starting LOYO Ridge sweep across {water_years.size} water years", flush=True)
    for outer_index, test_water_year in enumerate(water_years):
        train_mask = water_years != test_water_year
        x_train = dataset.x[train_mask]
        x_test = dataset.x[~train_mask]
        if x_test.shape[0] != 1:
            raise ValueError("Expected exactly one held-out water year per fold.")

        top20_alpha, top20_score = select_alpha(x_train, dataset.y_top20_all[train_mask], RIDGE_ALPHA_VALUES)
        y_pred_top20_all[outer_index, :] = fit_scaled_model(top20_alpha, x_train, dataset.y_top20_all[train_mask], x_test)[0]
        hyperparameter_rows.append(
            {
                "target_definition": TOP20_NAME,
                "model_name": "ridge",
                "outer_water_year": int(test_water_year),
                "region_label": 0,
                "region_name": TOP20_NAME,
                "selected_alpha": float(top20_alpha),
                "inner_cv_score_neg_mean_mse_scaled_y": float(top20_score),
            }
        )
        print(
            f"LOYO held_out_WY={int(test_water_year)} target_definition={TOP20_NAME} "
            f"alpha={top20_alpha:g} inner_score={top20_score:.6f}",
            flush=True,
        )

        y_regionwise_true = dataset.y_regionwise.reshape(dataset.y_regionwise.shape[0], len(dataset.region_definitions), len(target_months))
        y_regionwise_pred = y_pred_regionwise.reshape(y_pred_regionwise.shape[0], len(dataset.region_definitions), len(target_months))
        for region_index, region in enumerate(dataset.region_definitions):
            best_alpha, best_score = select_alpha(x_train, y_regionwise_true[train_mask, region_index, :], RIDGE_ALPHA_VALUES)
            y_regionwise_pred[outer_index, region_index, :] = fit_scaled_model(
                best_alpha,
                x_train,
                y_regionwise_true[train_mask, region_index, :],
                x_test,
            )[0]
            hyperparameter_rows.append(
                {
                    "target_definition": "regionwise",
                    "model_name": "ridge",
                    "outer_water_year": int(test_water_year),
                    "region_label": region.semantic_label,
                    "region_name": REGION_NAME_BY_LABEL[region.semantic_label],
                    "selected_alpha": float(best_alpha),
                    "inner_cv_score_neg_mean_mse_scaled_y": float(best_score),
                }
            )
            print(
                f"LOYO held_out_WY={int(test_water_year)} target_definition=regionwise "
                f"region={region.semantic_label} alpha={best_alpha:g} inner_score={best_score:.6f}",
                flush=True,
            )

    hyperparameter_df = pd.DataFrame(hyperparameter_rows).sort_values(
        ["target_definition", "region_label", "outer_water_year"]
    ).reset_index(drop=True)
    return y_pred_top20_all, y_pred_regionwise, hyperparameter_df


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": pearson_corr(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def compute_metrics(
    dataset: DatasetBundle,
    y_pred_top20_all: np.ndarray,
    y_pred_regionwise: np.ndarray,
    target_months: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    overall_rows: List[Dict[str, object]] = []
    month_rows: List[Dict[str, object]] = []
    region_rows: List[Dict[str, object]] = []
    region_month_rows: List[Dict[str, object]] = []

    overall_rows.append(
        {
            "target_definition": TOP20_NAME,
            "model_name": "ridge",
            **metric_block(dataset.y_top20_all.reshape(-1), y_pred_top20_all.reshape(-1)),
        }
    )
    for month_index, month_name in enumerate(target_months):
        month_rows.append(
            {
                "target_definition": TOP20_NAME,
                "model_name": "ridge",
                "target_month": month_name,
                **metric_block(dataset.y_top20_all[:, month_index], y_pred_top20_all[:, month_index]),
            }
        )
    region_rows.append(
        {
            "target_definition": TOP20_NAME,
            "model_name": "ridge",
            "region_label": 0,
            "region_name": TOP20_NAME,
            **metric_block(dataset.y_top20_all.reshape(-1), y_pred_top20_all.reshape(-1)),
        }
    )

    n_regions = len(dataset.region_definitions)
    n_months = len(target_months)
    y_true_regionwise = dataset.y_regionwise.reshape(dataset.y_regionwise.shape[0], n_regions, n_months)
    y_pred_regionwise_3d = y_pred_regionwise.reshape(y_pred_regionwise.shape[0], n_regions, n_months)

    overall_rows.append(
        {
            "target_definition": "regionwise",
            "model_name": "ridge",
            **metric_block(dataset.y_regionwise.reshape(-1), y_pred_regionwise.reshape(-1)),
        }
    )
    for region_index, region in enumerate(dataset.region_definitions):
        region_rows.append(
            {
                "target_definition": "regionwise",
                "model_name": "ridge",
                "region_label": region.semantic_label,
                "region_name": REGION_NAME_BY_LABEL[region.semantic_label],
                **metric_block(
                    y_true_regionwise[:, region_index, :].reshape(-1),
                    y_pred_regionwise_3d[:, region_index, :].reshape(-1),
                ),
            }
        )
    for month_index, month_name in enumerate(target_months):
        month_rows.append(
            {
                "target_definition": "regionwise",
                "model_name": "ridge",
                "target_month": month_name,
                **metric_block(
                    y_true_regionwise[:, :, month_index].reshape(-1),
                    y_pred_regionwise_3d[:, :, month_index].reshape(-1),
                ),
            }
        )
    for region_index, region in enumerate(dataset.region_definitions):
        for month_index, month_name in enumerate(target_months):
            region_month_rows.append(
                {
                    "target_definition": "regionwise",
                    "model_name": "ridge",
                    "region_label": region.semantic_label,
                    "region_name": REGION_NAME_BY_LABEL[region.semantic_label],
                    "target_month": month_name,
                    **metric_block(
                        y_true_regionwise[:, region_index, month_index],
                        y_pred_regionwise_3d[:, region_index, month_index],
                    ),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    month_df = pd.DataFrame(month_rows)
    region_df = pd.DataFrame(region_rows)
    region_month_df = pd.DataFrame(region_month_rows)

    best_target_row = overall_df.sort_values("r2", ascending=False).iloc[0]
    regionwise_only = region_df[region_df["target_definition"] == "regionwise"].sort_values("r2", ascending=False)
    best_region_row = regionwise_only.iloc[0]
    summary = {
        "overall": overall_rows,
        "by_month": month_rows,
        "by_region": region_rows,
        "region_by_month": region_month_rows,
        "best_target_definition_by_loyo_r2": {
            "target_definition": best_target_row["target_definition"],
            "r2": float(best_target_row["r2"]),
        },
        "best_region_by_pooled_jfm_loyo_r2": {
            "region_label": int(best_region_row["region_label"]),
            "region_name": best_region_row["region_name"],
            "r2": float(best_region_row["r2"]),
        },
    }
    return overall_df, month_df, region_df, region_month_df, summary


def save_predictions(
    output_dir: Path,
    dataset: DatasetBundle,
    y_pred_top20_all: np.ndarray,
    y_pred_regionwise: np.ndarray,
    target_months: Sequence[str],
) -> None:
    rows_long: List[Dict[str, object]] = []
    rows_wide: List[Dict[str, object]] = []

    y_true_regionwise = dataset.y_regionwise.reshape(dataset.y_regionwise.shape[0], len(dataset.region_definitions), len(target_months))
    y_pred_regionwise_3d = y_pred_regionwise.reshape(y_pred_regionwise.shape[0], len(dataset.region_definitions), len(target_months))

    for row_index, water_year in enumerate(dataset.water_years.tolist()):
        row_wide: Dict[str, object] = {"model_name": "ridge", "water_year": int(water_year)}
        for month_index, month_name in enumerate(target_months):
            y_true_value = float(dataset.y_top20_all[row_index, month_index])
            y_pred_value = float(y_pred_top20_all[row_index, month_index])
            rows_long.append(
                {
                    "target_definition": TOP20_NAME,
                    "model_name": "ridge",
                    "water_year": int(water_year),
                    "region_label": 0,
                    "region_name": TOP20_NAME,
                    "target_month": month_name,
                    "y_true": y_true_value,
                    "y_pred": y_pred_value,
                    "error": y_pred_value - y_true_value,
                }
            )
            prefix = f"{TOP20_NAME}_{month_name}"
            row_wide[f"{prefix}_true"] = y_true_value
            row_wide[f"{prefix}_pred"] = y_pred_value
            row_wide[f"{prefix}_error"] = y_pred_value - y_true_value

        for region_index, region in enumerate(dataset.region_definitions):
            region_name = REGION_NAME_BY_LABEL[region.semantic_label]
            for month_index, month_name in enumerate(target_months):
                y_true_value = float(y_true_regionwise[row_index, region_index, month_index])
                y_pred_value = float(y_pred_regionwise_3d[row_index, region_index, month_index])
                rows_long.append(
                    {
                        "target_definition": "regionwise",
                        "model_name": "ridge",
                        "water_year": int(water_year),
                        "region_label": region.semantic_label,
                        "region_name": region_name,
                        "target_month": month_name,
                        "y_true": y_true_value,
                        "y_pred": y_pred_value,
                        "error": y_pred_value - y_true_value,
                    }
                )
                prefix = f"{region_name}_{month_name}"
                row_wide[f"{prefix}_true"] = y_true_value
                row_wide[f"{prefix}_pred"] = y_pred_value
                row_wide[f"{prefix}_error"] = y_pred_value - y_true_value
        rows_wide.append(row_wide)

    pd.DataFrame(rows_long).to_csv(output_dir / "loyo_predictions_long.csv", index=False)
    pd.DataFrame(rows_wide).to_csv(output_dir / "loyo_predictions_wide.csv", index=False)
    np.save(output_dir / "y_true_top20_all.npy", dataset.y_top20_all)
    np.save(output_dir / "y_pred_top20_all.npy", y_pred_top20_all)
    np.save(output_dir / "y_true_regionwise.npy", dataset.y_regionwise)
    np.save(output_dir / "y_pred_regionwise.npy", y_pred_regionwise)


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


def save_fig(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=220)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_debug_outputs(
    plots_dir: Path,
    dataset: DatasetBundle,
    y_pred_top20_all: np.ndarray,
    y_pred_regionwise: np.ndarray,
    metrics_df: pd.DataFrame,
    hyperparameter_df: pd.DataFrame,
    target_months: Sequence[str],
) -> None:
    top20_metrics = metrics_df[metrics_df["target_definition"] == TOP20_NAME].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    scatter_with_identity(
        ax,
        dataset.y_top20_all.reshape(-1),
        y_pred_top20_all.reshape(-1),
        TOP20_COLOR,
        (
            f"Train-all debug top20_all alpha={float(hyperparameter_df[hyperparameter_df['region_label'] == 0]['selected_alpha'].iloc[0]):g}\n"
            f"R2={top20_metrics['r2']:.3f}, corr={top20_metrics['corr']:.3f}, "
            f"RMSE={top20_metrics['rmse']:.3f}, MAE={top20_metrics['mae']:.3f}"
        ),
    )
    save_fig(fig, plots_dir / "trainall_top20_scatter")

    n_regions = len(dataset.region_definitions)
    n_months = len(target_months)
    y_true_regionwise = dataset.y_regionwise.reshape(dataset.y_regionwise.shape[0], n_regions, n_months)
    y_pred_regionwise_3d = y_pred_regionwise.reshape(y_pred_regionwise.shape[0], n_regions, n_months)
    regionwise_metrics = metrics_df[metrics_df["target_definition"] == "regionwise"].sort_values("region_label")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for region_index, region in enumerate(dataset.region_definitions):
        region_metrics = regionwise_metrics[regionwise_metrics["region_label"] == region.semantic_label].iloc[0]
        selected_alpha = float(
            hyperparameter_df[hyperparameter_df["region_label"] == region.semantic_label]["selected_alpha"].iloc[0]
        )
        scatter_with_identity(
            axes[region_index],
            y_true_regionwise[:, region_index, :].reshape(-1),
            y_pred_regionwise_3d[:, region_index, :].reshape(-1),
            REGIONWISE_COLOR,
            (
                f"{REGION_NAME_BY_LABEL[region.semantic_label]} alpha={selected_alpha:g}\n"
                f"R2={region_metrics['r2']:.3f}, corr={region_metrics['corr']:.3f}, "
                f"RMSE={region_metrics['rmse']:.3f}, MAE={region_metrics['mae']:.3f}"
            ),
        )
    save_fig(fig, plots_dir / "trainall_regionwise_scatter")

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8), constrained_layout=True)
    panels = [(TOP20_NAME, 0, TOP20_NAME)] + [("regionwise", region.semantic_label, REGION_NAME_BY_LABEL[region.semantic_label]) for region in dataset.region_definitions]
    for ax, (target_definition, region_label, title) in zip(axes, panels):
        subset = hyperparameter_df[
            (hyperparameter_df["target_definition"] == target_definition)
            & (hyperparameter_df["region_label"] == region_label)
        ].iloc[0]
        ax.bar([0], [float(subset["selected_alpha"])], color=TOP20_COLOR if region_label == 0 else REGIONWISE_COLOR, alpha=0.85)
        ax.set_xticks([0])
        ax.set_xticklabels([title], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Selected alpha")
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(0, float(subset["selected_alpha"]), "{:g}".format(float(subset["selected_alpha"])), ha="center", va="bottom", fontsize=10)
    save_fig(fig, plots_dir / "trainall_selected_alpha_bar")

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    best_rows = [top20_metrics] + [regionwise_metrics[regionwise_metrics["region_label"] == region.semantic_label].iloc[0] for region in dataset.region_definitions]
    labels = [TOP20_NAME] + [REGION_NAME_BY_LABEL[region.semantic_label] for region in dataset.region_definitions]
    values = [float(row["r2"]) for row in best_rows]
    bars = ax.bar(np.arange(len(labels)), values, color=[TOP20_COLOR] + [REGIONWISE_COLOR] * len(dataset.region_definitions))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Train-all debug R2")
    ax.set_title("Train-all debug R2 by target")
    ax.grid(True, axis="y", alpha=0.25)
    alpha_values = [float(hyperparameter_df[hyperparameter_df["region_label"] == 0]["selected_alpha"].iloc[0])] + [
        float(hyperparameter_df[hyperparameter_df["region_label"] == region.semantic_label]["selected_alpha"].iloc[0])
        for region in dataset.region_definitions
    ]
    for bar, value, alpha in zip(bars, values, alpha_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, "{:.2f}\na={:g}".format(value, alpha), ha="center", va="bottom", fontsize=9)
    save_fig(fig, plots_dir / "trainall_r2_bar")


def plot_outputs(
    plots_dir: Path,
    dataset: DatasetBundle,
    y_pred_top20_all: np.ndarray,
    y_pred_regionwise: np.ndarray,
    overall_df: pd.DataFrame,
    month_df: pd.DataFrame,
    region_df: pd.DataFrame,
    region_month_df: pd.DataFrame,
    hyperparameter_df: pd.DataFrame,
    target_months: Sequence[str],
) -> None:
    top20_overall = overall_df[overall_df["target_definition"] == TOP20_NAME].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    scatter_with_identity(
        ax,
        dataset.y_top20_all.reshape(-1),
        y_pred_top20_all.reshape(-1),
        TOP20_COLOR,
        (
            f"All-top20 pooled JFM\nR2={top20_overall['r2']:.3f}, corr={top20_overall['corr']:.3f}, "
            f"RMSE={top20_overall['rmse']:.3f}, MAE={top20_overall['mae']:.3f}"
        ),
    )
    save_fig(fig, plots_dir / "top20_all_scatter_pooled")

    top20_month_df = month_df[month_df["target_definition"] == TOP20_NAME]
    fig, axes = plt.subplots(1, len(target_months), figsize=(5.0 * len(target_months), 4.8), constrained_layout=True)
    for month_index, month_name in enumerate(target_months):
        row = top20_month_df[top20_month_df["target_month"] == month_name].iloc[0]
        scatter_with_identity(
            axes[month_index],
            dataset.y_top20_all[:, month_index],
            y_pred_top20_all[:, month_index],
            TOP20_COLOR,
            f"{month_name}\nR2={row['r2']:.3f}, corr={row['corr']:.3f}, RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}",
        )
    save_fig(fig, plots_dir / "top20_all_month_scatter")

    n_regions = len(dataset.region_definitions)
    n_months = len(target_months)
    y_true_regionwise = dataset.y_regionwise.reshape(dataset.y_regionwise.shape[0], n_regions, n_months)
    y_pred_regionwise_3d = y_pred_regionwise.reshape(y_pred_regionwise.shape[0], n_regions, n_months)

    fig, axes = plt.subplots(n_regions, n_months, figsize=(4.2 * n_months, 3.9 * n_regions), constrained_layout=True)
    for region_index, region in enumerate(dataset.region_definitions):
        for month_index, month_name in enumerate(target_months):
            row = region_month_df[
                (region_month_df["region_label"] == region.semantic_label)
                & (region_month_df["target_month"] == month_name)
            ].iloc[0]
            scatter_with_identity(
                axes[region_index, month_index],
                y_true_regionwise[:, region_index, month_index],
                y_pred_regionwise_3d[:, region_index, month_index],
                REGIONWISE_COLOR,
                (
                    f"{REGION_NAME_BY_LABEL[region.semantic_label]} | {month_name}\n"
                    f"R2={row['r2']:.3f}, corr={row['corr']:.3f}\n"
                    f"RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}"
                ),
            )
    save_fig(fig, plots_dir / "regionwise_region_month_scatter_grid")

    fig, axes = plt.subplots(1, n_regions, figsize=(5.1 * n_regions, 4.8), constrained_layout=True)
    regionwise_region_df = region_df[region_df["target_definition"] == "regionwise"]
    for region_index, region in enumerate(dataset.region_definitions):
        row = regionwise_region_df[regionwise_region_df["region_label"] == region.semantic_label].iloc[0]
        scatter_with_identity(
            axes[region_index],
            y_true_regionwise[:, region_index, :].reshape(-1),
            y_pred_regionwise_3d[:, region_index, :].reshape(-1),
            REGIONWISE_COLOR,
            (
                f"{REGION_NAME_BY_LABEL[region.semantic_label]}\n"
                f"R2={row['r2']:.3f}, corr={row['corr']:.3f}, RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}"
            ),
        )
    save_fig(fig, plots_dir / "regionwise_region_pooled_scatter")

    regionwise_month_df = month_df[month_df["target_definition"] == "regionwise"]
    fig, axes = plt.subplots(1, len(target_months), figsize=(5.0 * len(target_months), 4.8), constrained_layout=True)
    for month_index, month_name in enumerate(target_months):
        row = regionwise_month_df[regionwise_month_df["target_month"] == month_name].iloc[0]
        scatter_with_identity(
            axes[month_index],
            y_true_regionwise[:, :, month_index].reshape(-1),
            y_pred_regionwise_3d[:, :, month_index].reshape(-1),
            REGIONWISE_COLOR,
            f"{month_name}\nR2={row['r2']:.3f}, corr={row['corr']:.3f}, RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}",
        )
    save_fig(fig, plots_dir / "regionwise_month_pooled_scatter")

    fig, ax = plt.subplots(figsize=(12.6, 4.9), constrained_layout=True)
    for month_index, month_name in enumerate(target_months):
        color = plt.cm.tab10(month_index)
        ax.plot(dataset.water_years, dataset.y_top20_all[:, month_index], color=color, marker="o", linewidth=1.8, label=f"{month_name} observed")
        ax.plot(dataset.water_years, y_pred_top20_all[:, month_index], color=color, marker="x", linestyle="--", linewidth=1.4, label=f"{month_name} predicted")
    ax.set_title("All-top20 observed vs predicted across water years")
    ax.set_xlabel("Water year")
    ax.set_ylabel("T2m anomaly")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    save_fig(fig, plots_dir / "top20_all_time_series")

    for region_index, region in enumerate(dataset.region_definitions):
        fig, ax = plt.subplots(figsize=(12.6, 4.9), constrained_layout=True)
        for month_index, month_name in enumerate(target_months):
            color = plt.cm.tab10(month_index)
            ax.plot(
                dataset.water_years,
                y_true_regionwise[:, region_index, month_index],
                color=color,
                marker="o",
                linewidth=1.8,
                label=f"{month_name} observed",
            )
            ax.plot(
                dataset.water_years,
                y_pred_regionwise_3d[:, region_index, month_index],
                color=color,
                marker="x",
                linestyle="--",
                linewidth=1.4,
                label=f"{month_name} predicted",
            )
        ax.set_title(f"{REGION_NAME_BY_LABEL[region.semantic_label]} observed vs predicted across water years")
        ax.set_xlabel("Water year")
        ax.set_ylabel("T2m anomaly")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=9)
        save_fig(fig, plots_dir / f"regionwise_time_series_region{region.semantic_label}")

    region_r2_rows = regionwise_region_df.sort_values("region_label")
    labels = [TOP20_NAME] + region_r2_rows["region_name"].tolist()
    r2_values = [float(top20_overall["r2"])] + region_r2_rows["r2"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(labels)), r2_values, color=[TOP20_COLOR] + [REGIONWISE_COLOR] * len(region_r2_rows))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("LOYO pooled JFM R2")
    ax.set_title("R2 comparison by target definition")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    save_fig(fig, plots_dir / "r2_comparison_bar")

    alpha_summary = (
        hyperparameter_df.groupby(["target_definition", "region_label", "region_name", "selected_alpha"])
        .size()
        .reset_index(name="fold_count")
        .sort_values(["target_definition", "region_label", "selected_alpha"])
    )
    alpha_summary.to_csv(plots_dir / "selected_alpha_summary_table.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.6), constrained_layout=True)
    panel_rows = [
        (TOP20_NAME, 0, TOP20_NAME),
        ("regionwise", 1, REGION_NAME_BY_LABEL[1]),
        ("regionwise", 2, REGION_NAME_BY_LABEL[2]),
        ("regionwise", 3, REGION_NAME_BY_LABEL[3]),
    ]
    for ax, (target_definition, region_label, region_name) in zip(axes, panel_rows):
        subset = alpha_summary[
            (alpha_summary["target_definition"] == target_definition) & (alpha_summary["region_label"] == region_label)
        ]
        if subset.empty:
            ax.set_title(region_name)
            ax.text(0.5, 0.5, "No folds", ha="center", va="center")
            ax.axis("off")
            continue
        x_values = np.arange(subset.shape[0])
        ax.bar(x_values, subset["fold_count"].to_numpy(), color=TOP20_COLOR if region_label == 0 else REGIONWISE_COLOR)
        ax.set_xticks(x_values)
        ax.set_xticklabels([f"{value:g}" for value in subset["selected_alpha"]], rotation=45, ha="right")
        ax.set_xlabel("Selected alpha")
        ax.set_ylabel("Fold count")
        ax.set_title(region_name)
        ax.grid(True, axis="y", alpha=0.25)
    save_fig(fig, plots_dir / "selected_alpha_distribution")


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: DatasetBundle,
    hyperparameter_df: Optional[pd.DataFrame],
) -> None:
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
        "target_names_top20_all": dataset.target_names_top20_all,
        "target_names_regionwise": dataset.target_names_regionwise,
        "alpha_grid": RIDGE_ALPHA_VALUES,
        "cv": "inner_cv_on_all_years_then_refit_all_years" if args.train_all_debug else "leave-one-water-year-out",
        "mode": "ridge_target_sensitivity_jfm_debug" if args.train_all_debug else "ridge_target_sensitivity_jfm",
        "t2m_source_note": dataset.t2m_source_note,
    }
    if hyperparameter_df is not None and not hyperparameter_df.empty:
        payload["selected_hyperparameters_preview"] = hyperparameter_df.head(12).to_dict(orient="records")
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_final_summary(
    output_dir: Path,
    dataset: DatasetBundle,
    overall_df: pd.DataFrame,
    region_df: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    print("Experiment completed.", flush=True)
    print("Cleanup completed.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Dataset shapes: X={dataset.x.shape}, Y_top20_all={dataset.y_top20_all.shape}, Y_regionwise={dataset.y_regionwise.shape}", flush=True)
    print(
        f"Water years used: {int(dataset.water_years.min())}-{int(dataset.water_years.max())} "
        f"({dataset.water_years.size} total)",
        flush=True,
    )
    print("Top20-all metrics summary.", flush=True)
    top20_row = overall_df[overall_df["target_definition"] == TOP20_NAME].iloc[0]
    print(
        f"  R2={top20_row['r2']:.4f}, corr={top20_row['corr']:.4f}, RMSE={top20_row['rmse']:.4f}, MAE={top20_row['mae']:.4f}",
        flush=True,
    )
    print("Regionwise metrics summary.", flush=True)
    for row in region_df[region_df["target_definition"] == "regionwise"].itertuples(index=False):
        print(
            f"  region{row.region_label} {row.region_name}: R2={row.r2:.4f}, corr={row.corr:.4f}, "
            f"RMSE={row.rmse:.4f}, MAE={row.mae:.4f}",
            flush=True,
        )
    best_target = summary["best_target_definition_by_loyo_r2"]
    best_region = summary["best_region_by_pooled_jfm_loyo_r2"]
    print(
        f"Best target definition by LOYO R2: {best_target['target_definition']} ({best_target['r2']:.4f})",
        flush=True,
    )
    print(
        f"Best region by pooled JFM LOYO R2: region{best_region['region_label']} "
        f"{best_region['region_name']} ({best_region['r2']:.4f})",
        flush=True,
    )


def run_train_all_debug(
    dataset: DatasetBundle,
    output_dir: Path,
    target_months: Sequence[str],
) -> None:
    print("Starting train-all debug mode", flush=True)
    hyperparameter_rows: List[Dict[str, object]] = []
    rows_long: List[Dict[str, object]] = []
    rows_metrics: List[Dict[str, object]] = []
    y_true_regionwise = dataset.y_regionwise.reshape(dataset.y_regionwise.shape[0], len(dataset.region_definitions), len(target_months))
    top20_alpha, top20_score = select_alpha(dataset.x, dataset.y_top20_all, RIDGE_ALPHA_VALUES)
    y_pred_top20_all = fit_scaled_model(top20_alpha, dataset.x, dataset.y_top20_all, dataset.x)
    hyperparameter_rows.append(
        {
            "target_definition": TOP20_NAME,
            "model_name": "ridge",
            "region_label": 0,
            "region_name": TOP20_NAME,
            "selected_alpha": float(top20_alpha),
            "inner_cv_score_neg_mean_mse_scaled_y": float(top20_score),
        }
    )
    top20_metrics = metric_block(dataset.y_top20_all.reshape(-1), y_pred_top20_all.reshape(-1))
    rows_metrics.append(
        {
            "target_definition": TOP20_NAME,
            "model_name": "ridge",
            "region_label": 0,
            "region_name": TOP20_NAME,
            **top20_metrics,
        }
    )
    for row_index, water_year in enumerate(dataset.water_years.tolist()):
        for month_index, month_name in enumerate(target_months):
            rows_long.append(
                {
                    "target_definition": TOP20_NAME,
                    "model_name": "ridge",
                    "selected_alpha": float(top20_alpha),
                    "water_year": int(water_year),
                    "region_label": 0,
                    "region_name": TOP20_NAME,
                    "target_month": month_name,
                    "y_true": float(dataset.y_top20_all[row_index, month_index]),
                    "y_pred": float(y_pred_top20_all[row_index, month_index]),
                    "error": float(y_pred_top20_all[row_index, month_index] - dataset.y_top20_all[row_index, month_index]),
                }
            )
    print(
        f"TRAIN_ALL_DEBUG target_definition={TOP20_NAME} alpha={top20_alpha:g} "
        f"R2={top20_metrics['r2']:.4f} corr={top20_metrics['corr']:.4f}",
        flush=True,
    )

    y_pred_regionwise = np.full_like(dataset.y_regionwise, np.nan, dtype=np.float64)
    y_pred_regionwise_3d = y_pred_regionwise.reshape(y_pred_regionwise.shape[0], len(dataset.region_definitions), len(target_months))
    for region_index, region in enumerate(dataset.region_definitions):
        region_name = REGION_NAME_BY_LABEL[region.semantic_label]
        region_alpha, region_score = select_alpha(dataset.x, y_true_regionwise[:, region_index, :], RIDGE_ALPHA_VALUES)
        region_pred = fit_scaled_model(region_alpha, dataset.x, y_true_regionwise[:, region_index, :], dataset.x)
        y_pred_regionwise_3d[:, region_index, :] = region_pred
        hyperparameter_rows.append(
            {
                "target_definition": "regionwise",
                "model_name": "ridge",
                "region_label": region.semantic_label,
                "region_name": region_name,
                "selected_alpha": float(region_alpha),
                "inner_cv_score_neg_mean_mse_scaled_y": float(region_score),
            }
        )
        region_metrics = metric_block(y_true_regionwise[:, region_index, :].reshape(-1), region_pred.reshape(-1))
        rows_metrics.append(
            {
                "target_definition": "regionwise",
                "model_name": "ridge",
                "region_label": region.semantic_label,
                "region_name": region_name,
                **region_metrics,
            }
        )
        for row_index, water_year in enumerate(dataset.water_years.tolist()):
            for month_index, month_name in enumerate(target_months):
                rows_long.append(
                    {
                        "target_definition": "regionwise",
                        "model_name": "ridge",
                        "selected_alpha": float(region_alpha),
                        "water_year": int(water_year),
                        "region_label": region.semantic_label,
                        "region_name": region_name,
                        "target_month": month_name,
                        "y_true": float(y_true_regionwise[row_index, region_index, month_index]),
                        "y_pred": float(region_pred[row_index, month_index]),
                        "error": float(region_pred[row_index, month_index] - y_true_regionwise[row_index, region_index, month_index]),
                    }
                )
        print(
            f"TRAIN_ALL_DEBUG target_definition=regionwise region={region.semantic_label} "
            f"alpha={region_alpha:g} R2={region_metrics['r2']:.4f} corr={region_metrics['corr']:.4f}",
            flush=True,
        )

    hyperparameter_df = pd.DataFrame(hyperparameter_rows).sort_values(["target_definition", "region_label"]).reset_index(drop=True)
    metrics_df = pd.DataFrame(rows_metrics).sort_values(["target_definition", "region_label"]).reset_index(drop=True)
    predictions_df = pd.DataFrame(rows_long).sort_values(
        ["target_definition", "region_label", "water_year", "target_month"]
    ).reset_index(drop=True)
    hyperparameter_df.to_csv(output_dir / "selected_hyperparameters_trainall.csv", index=False)
    metrics_df.to_csv(output_dir / "trainall_debug_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "trainall_debug_predictions_long.csv", index=False)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_debug_outputs(plots_dir, dataset, y_pred_top20_all, y_pred_regionwise, metrics_df, hyperparameter_df, target_months)
    (output_dir / "trainall_debug_summary.json").write_text(
        json.dumps(
            {
                "alpha_grid": RIDGE_ALPHA_VALUES,
                "selected_hyperparameters": hyperparameter_df.to_dict(orient="records"),
                "metrics": metrics_df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Finished train-all debug mode", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    if not args.train_all_debug:
        plots_dir.mkdir(parents=True, exist_ok=True)

    artifact_candidates = base_mod.candidate_files_from_dir(args.artifact_dir)
    label_candidates = base_mod.candidate_files_from_dir(args.label_dir)
    route2_netcdf = base_mod.locate_route2_netcdf(args.artifact_dir)
    label_netcdf = base_mod.locate_label_netcdf(args.label_dir)
    base_mod.print_candidate_group("Candidate files in artifact directory:", artifact_candidates)
    base_mod.print_candidate_group("Candidate files in region-label directory:", label_candidates)

    dataset = build_supervised_dataset(route2_netcdf, label_netcdf, args.input_months, args.target_months)
    print_dataset_summary(dataset)
    save_dataset_files(dataset, args.output_dir)

    if args.train_all_debug:
        save_run_config(args.output_dir, args, dataset, hyperparameter_df=None)
        run_train_all_debug(dataset, args.output_dir, args.target_months)
        return

    y_pred_top20_all, y_pred_regionwise, hyperparameter_df = run_loyo(dataset, args.target_months)
    hyperparameter_df.to_csv(args.output_dir / "selected_hyperparameters_by_fold.csv", index=False)

    overall_df, month_df, region_df, region_month_df, summary = compute_metrics(
        dataset,
        y_pred_top20_all,
        y_pred_regionwise,
        args.target_months,
    )
    overall_df.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    month_df.to_csv(args.output_dir / "metrics_by_month.csv", index=False)
    region_df.to_csv(args.output_dir / "metrics_by_region.csv", index=False)
    region_month_df.to_csv(args.output_dir / "metrics_region_by_month.csv", index=False)
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save_predictions(args.output_dir, dataset, y_pred_top20_all, y_pred_regionwise, args.target_months)
    plot_outputs(
        plots_dir,
        dataset,
        y_pred_top20_all,
        y_pred_regionwise,
        overall_df,
        month_df,
        region_df,
        region_month_df,
        hyperparameter_df,
        args.target_months,
    )
    save_run_config(args.output_dir, args, dataset, hyperparameter_df)
    print_final_summary(args.output_dir, dataset, overall_df, region_df, summary)


if __name__ == "__main__":
    main()
