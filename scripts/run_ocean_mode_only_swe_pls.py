#!/usr/bin/env python3
"""
Run the basin-based ocean-mode-only Sierra SWE PLS baseline with strict LOYO.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.cross_decomposition import PLSRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import (  # noqa: E402
    DEFAULT_SIERRA_REGION,
    SWE_MISSING_VALUE,
    SWE_VARIABLE,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)


PACIFIC_PC_PATH = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_pacific_sierra_t2m_level2_pc1to6/cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
)
NINO34_CSV_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "nino34"
    / "nino34_monthly_wy1985_2021_sep_mar.csv"
)
AMV_AMO_CSV_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)
REGION_MASK_NPZ_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_target_spatial_diagnostic"
    / "basin_assignment_grid_wy2021.npz"
)
REGION_MASK_NC_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_target_spatial_diagnostic"
    / "basin_assignment_grid_wy2021.nc"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "ocean_mode_only_pls"
PREDICTOR_TABLE_CSV = OUTPUT_DIR / "ocean_mode_predictors_wy1985_2021.csv"
TARGET_TABLE_CSV = OUTPUT_DIR / "sierra_apr1_swe_north_central_south_wy1985_2021.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "loyo_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "loyo_metrics.json"
SELECTED_COMPONENTS_CSV = OUTPUT_DIR / "selected_components_by_fold.csv"
COEFFICIENTS_CSV = OUTPUT_DIR / "pls_coefficients_by_fold.csv"
LOADINGS_WEIGHTS_NC = OUTPUT_DIR / "pls_loadings_weights_by_fold.nc"
TIMESERIES_PNG = OUTPUT_DIR / "observed_vs_predicted_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "observed_vs_predicted_scatter.png"
COMPONENTS_PNG = OUTPUT_DIR / "selected_components_by_fold.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
MONTH_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]
PLS_COMPONENT_GRID = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
NETCDF_ENGINE = "netcdf4"
MEAN_STAT_INDEX = 0
REGION_CODES = {"North": 1.0, "Central": 2.0, "South": 3.0}
REGION_KEYS = ("North", "Central", "South")


@dataclass(frozen=True)
class RegionTargetSeries:
    key: str
    observed_m: np.ndarray


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def get_swe_coordinate_names() -> Tuple[str, str]:
    swe_grid = get_regional_swe_grid_definition(
        water_year=int(WATER_YEAR_START),
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )
    return str(swe_grid.latitude.dims[0]), str(swe_grid.longitude.dims[0])


def load_region_assignment_mask() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(REGION_MASK_NPZ_PATH, allow_pickle=True)
    lat = np.asarray(payload["lat"], dtype=np.float64)
    lon = np.asarray(payload["lon"], dtype=np.float64)
    assignment = np.asarray(payload["assignment"], dtype=np.float64)
    if assignment.shape != (lat.size, lon.size):
        raise ValueError(
            "Region assignment shape does not match lat/lon sizes: "
            f"{assignment.shape} vs {(lat.size, lon.size)}"
        )
    finite_codes = np.unique(assignment[np.isfinite(assignment)])
    expected_codes = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    if not np.array_equal(finite_codes, expected_codes):
        raise ValueError(f"Unexpected region codes in assignment mask: {finite_codes}")
    return lat, lon, assignment


def load_apr1_swe_for_water_year(
    water_year: int,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_name: str,
    lon_name: str,
) -> np.ndarray:
    path = swe_file_for_water_year(water_year)
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        swe = ds[SWE_VARIABLE]
        if "Stats" in swe.dims:
            swe = swe.isel(Stats=MEAN_STAT_INDEX, drop=True)
        swe = swe.sel(time=np.datetime64(f"{water_year:04d}-04-01"))
        swe = swe.sel(
            {
                lat_name: xr.DataArray(lat, dims=(lat_name,)),
                lon_name: xr.DataArray(lon, dims=(lon_name,)),
            }
        )
        values = np.asarray(swe.values, dtype=np.float64)
    values[values == SWE_MISSING_VALUE] = np.nan
    return values


def compute_region_weighted_mean(values: np.ndarray, region_mask: np.ndarray, lat_weights_2d: np.ndarray) -> float:
    valid = np.isfinite(values) & region_mask
    if not np.any(valid):
        return float("nan")
    weights = np.where(valid, lat_weights_2d, 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float("nan")
    return float(np.sum(np.where(valid, values, 0.0) * weights) / total_weight)


def build_target_table() -> Tuple[List[RegionTargetSeries], List[Dict[str, Any]]]:
    lat, lon, assignment = load_region_assignment_mask()
    lat_name, lon_name = get_swe_coordinate_names()
    lat_weights = np.cos(np.deg2rad(lat))
    lat_weights_2d = np.broadcast_to(lat_weights[:, None], assignment.shape)
    rows: List[Dict[str, Any]] = []
    region_values: Dict[str, List[float]] = {key: [] for key in REGION_KEYS}

    for water_year in WATER_YEARS:
        swe = load_apr1_swe_for_water_year(int(water_year), lat, lon, lat_name, lon_name)
        row: Dict[str, Any] = {"water_year": int(water_year)}
        for region_key in REGION_KEYS:
            region_mask = assignment == REGION_CODES[region_key]
            value = compute_region_weighted_mean(swe, region_mask, lat_weights_2d)
            row[f"SWE_{region_key}"] = value
            region_values[region_key].append(value)
        rows.append(row)
        print(f"processed April 1 SWE targets for WY{int(water_year)}", flush=True)

    with TARGET_TABLE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["water_year", "SWE_North", "SWE_Central", "SWE_South"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "water_year": row["water_year"],
                    "SWE_North": f"{row['SWE_North']:.12g}",
                    "SWE_Central": f"{row['SWE_Central']:.12g}",
                    "SWE_South": f"{row['SWE_South']:.12g}",
                }
            )

    targets = [
        RegionTargetSeries(key=region_key, observed_m=np.asarray(region_values[region_key], dtype=np.float64))
        for region_key in REGION_KEYS
    ]
    return targets, rows


def load_pacific_predictors() -> Tuple[List[str], np.ndarray]:
    with xr.open_dataset(PACIFIC_PC_PATH, engine=NETCDF_ENGINE) as ds:
        times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        pcs = np.asarray(ds["pacific_cobe2_pc"].values, dtype=np.float64)

    time_to_index = {
        str(np.datetime_as_string(value, unit="D")): idx
        for idx, value in enumerate(times)
    }
    columns: List[str] = []
    rows = np.full((WATER_YEARS.size, len(MONTH_SPECS) * 6), np.nan, dtype=np.float64)
    for wy_idx, water_year in enumerate(WATER_YEARS):
        col_idx = 0
        for month_name, year_offset, month in MONTH_SPECS:
            key = f"{int(water_year + year_offset):04d}-{month:02d}-01"
            time_idx = time_to_index.get(key)
            if time_idx is None:
                raise KeyError(f"Missing Pacific PC timestamp {key}")
            for mode_idx in range(6):
                rows[wy_idx, col_idx] = float(pcs[time_idx, mode_idx])
                if wy_idx == 0:
                    columns.append(f"Pacific_PC{mode_idx + 1}_{month_name}")
                col_idx += 1
    return columns, rows


def load_csv_predictor_table(path: Path, expected_prefix: str) -> Tuple[List[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or fieldnames[0] != "water_year":
            raise ValueError(f"Unexpected header in {path}")
        columns = fieldnames[1:]
        if not all(column.startswith(expected_prefix) for column in columns):
            raise ValueError(f"Unexpected column prefix in {path}: {columns}")
        rows: List[List[float]] = []
        water_years: List[int] = []
        for row in reader:
            water_years.append(int(row["water_year"]))
            rows.append([float(row[column]) for column in columns])
    if water_years != WATER_YEARS.tolist():
        raise ValueError(f"Water years in {path} do not match WY1985--WY2021")
    return columns, np.asarray(rows, dtype=np.float64)


def build_predictor_matrix() -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    pacific_columns, pacific = load_pacific_predictors()
    nino_columns, nino = load_csv_predictor_table(NINO34_CSV_PATH, "Nino34_")
    amv_columns, amv = load_csv_predictor_table(AMV_AMO_CSV_PATH, "AMV_PC")
    columns = pacific_columns + nino_columns + amv_columns
    matrix = np.concatenate([pacific, nino, amv], axis=1)
    if matrix.shape != (WATER_YEARS.size, len(columns)):
        raise ValueError(f"Unexpected predictor matrix shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Predictor matrix contains non-finite values")

    rows: List[Dict[str, Any]] = []
    with PREDICTOR_TABLE_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["water_year"] + columns
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, water_year in enumerate(WATER_YEARS):
            row = {"water_year": int(water_year)}
            for col_idx, column in enumerate(columns):
                row[column] = float(matrix[row_idx, col_idx])
            writer.writerow(
                {key: (f"{value:.12g}" if key != "water_year" else value) for key, value in row.items()}
            )
            rows.append(row)

    return columns, matrix, rows


def standardize_training_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=1)
    std = np.where(std == 0.0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def standardize_training_target(y_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train, ddof=1))
    if std == 0.0:
        std = 1.0
    return (y_train - mean) / std, mean, std


def fit_pls_model(x_train_std: np.ndarray, y_train_std: np.ndarray, n_components: int) -> PLSRegression:
    model = PLSRegression(n_components=int(n_components), scale=False)
    model.fit(x_train_std, y_train_std[:, None])
    return model


def choose_component_grid(n_train_samples: int, n_features: int) -> np.ndarray:
    max_components = min(int(PLS_COMPONENT_GRID.max()), n_train_samples - 1, n_features)
    return PLS_COMPONENT_GRID[PLS_COMPONENT_GRID <= max_components]


def select_components_inner_loyo(x_train_std: np.ndarray, y_train_std: np.ndarray) -> int:
    candidate_grid = choose_component_grid(x_train_std.shape[0], x_train_std.shape[1])
    best_components = int(candidate_grid[0])
    best_mse = float("inf")
    n_samples = x_train_std.shape[0]
    tolerance = 1.0e-12
    for n_components in candidate_grid:
        preds = np.full(n_samples, np.nan, dtype=np.float64)
        for inner_idx in range(n_samples):
            inner_mask = np.ones(n_samples, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train = x_train_std[inner_mask]
            y_inner_train = y_train_std[inner_mask]
            component_cap = min(int(n_components), x_inner_train.shape[0] - 1, x_inner_train.shape[1])
            model = fit_pls_model(x_inner_train, y_inner_train, component_cap)
            preds[inner_idx] = float(model.predict(x_train_std[~inner_mask]).ravel()[0])
        mse = float(np.mean((y_train_std - preds) ** 2))
        if mse < (best_mse - tolerance) or (abs(mse - best_mse) <= tolerance and int(n_components) < best_components):
            best_mse = mse
            best_components = int(n_components)
    return best_components


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_loyo_pls(
    feature_names: Sequence[str],
    x: np.ndarray,
    targets: Sequence[RegionTargetSeries],
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    xr.Dataset,
    Dict[str, Dict[str, float]],
    List[Dict[str, Any]],
]:
    n_samples, n_features = x.shape
    region_names = [target.key for target in targets]
    predictions_by_region: Dict[str, np.ndarray] = {
        target.key: np.full(n_samples, np.nan, dtype=np.float64) for target in targets
    }
    selected_components = np.full((len(targets), n_samples), np.nan, dtype=np.float64)
    coef_array = np.full((len(targets), n_samples, n_features), np.nan, dtype=np.float64)
    x_weights_array = np.full((len(targets), n_samples, n_features, int(PLS_COMPONENT_GRID.max())), np.nan, dtype=np.float64)
    x_loadings_array = np.full_like(x_weights_array, np.nan)
    x_rotations_array = np.full_like(x_weights_array, np.nan)
    x_scores_array = np.full((len(targets), n_samples, n_samples - 1, int(PLS_COMPONENT_GRID.max())), np.nan, dtype=np.float64)
    y_loadings_array = np.full((len(targets), n_samples, int(PLS_COMPONENT_GRID.max())), np.nan, dtype=np.float64)
    feature_mean_array = np.full((n_samples, n_features), np.nan, dtype=np.float64)
    feature_std_array = np.full((n_samples, n_features), np.nan, dtype=np.float64)
    target_mean_array = np.full((len(targets), n_samples), np.nan, dtype=np.float64)
    target_std_array = np.full((len(targets), n_samples), np.nan, dtype=np.float64)
    coefficient_rows: List[Dict[str, Any]] = []

    for outer_idx, water_year in enumerate(WATER_YEARS):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[outer_idx] = False
        test_mask = ~train_mask
        x_train_std, x_test_std, x_mean, x_std = standardize_training_features(x[train_mask], x[test_mask])
        feature_mean_array[outer_idx, :] = x_mean
        feature_std_array[outer_idx, :] = x_std

        for region_idx, target in enumerate(targets):
            y = target.observed_m
            y_train_std, y_mean, y_std = standardize_training_target(y[train_mask])
            n_components = select_components_inner_loyo(x_train_std, y_train_std)
            model = fit_pls_model(x_train_std, y_train_std, n_components)
            y_pred_std = float(model.predict(x_test_std).ravel()[0])
            y_pred = y_pred_std * y_std + y_mean

            predictions_by_region[target.key][outer_idx] = y_pred
            selected_components[region_idx, outer_idx] = float(n_components)
            coef_array[region_idx, outer_idx, :] = model.coef_.ravel()
            x_weights_array[region_idx, outer_idx, :, :n_components] = model.x_weights_
            x_loadings_array[region_idx, outer_idx, :, :n_components] = model.x_loadings_
            x_rotations_array[region_idx, outer_idx, :, :n_components] = model.x_rotations_
            x_scores_array[region_idx, outer_idx, :, :n_components] = model.x_scores_
            y_loadings_array[region_idx, outer_idx, :n_components] = model.y_loadings_.ravel()
            target_mean_array[region_idx, outer_idx] = y_mean
            target_std_array[region_idx, outer_idx] = y_std

            coefficient_row: Dict[str, Any] = {
                "water_year": int(water_year),
                "region": target.key,
                "n_components": int(n_components),
            }
            for feature_idx, feature_name in enumerate(feature_names):
                coefficient_row[feature_name] = float(model.coef_.ravel()[feature_idx])
            coefficient_rows.append(coefficient_row)

            print(
                f"LOYO WY{int(water_year)} region={target.key} "
                f"components={n_components} obs={float(y[outer_idx]):.6f} pred={y_pred:.6f}",
                flush=True,
            )

    metrics: Dict[str, Dict[str, float]] = {}
    for target in targets:
        observed = target.observed_m
        predicted = predictions_by_region[target.key]
        metrics[target.key] = {
            "R2_LOYO": r2_score_manual(observed, predicted),
            "RMSE": rmse(observed, predicted),
            "MAE": mae(observed, predicted),
            "Pearson_r": pearson_r(observed, predicted),
            "n_years": float(n_samples),
            "target_mean": float(np.mean(observed)),
            "target_std": float(np.std(observed, ddof=1)),
        }

    loadings_ds = xr.Dataset(
        data_vars={
            "pls_beta_standardized": (
                ("region", "water_year", "feature"),
                coef_array.astype(np.float32),
            ),
            "selected_n_components": (
                ("region", "water_year"),
                selected_components.astype(np.float32),
            ),
            "x_weights": (
                ("region", "water_year", "feature", "component"),
                x_weights_array.astype(np.float32),
            ),
            "x_loadings": (
                ("region", "water_year", "feature", "component"),
                x_loadings_array.astype(np.float32),
            ),
            "x_rotations": (
                ("region", "water_year", "feature", "component"),
                x_rotations_array.astype(np.float32),
            ),
            "x_scores": (
                ("region", "water_year", "train_sample", "component"),
                x_scores_array.astype(np.float32),
            ),
            "y_loadings": (
                ("region", "water_year", "component"),
                y_loadings_array.astype(np.float32),
            ),
            "feature_train_mean": (
                ("water_year", "feature"),
                feature_mean_array.astype(np.float32),
            ),
            "feature_train_std": (
                ("water_year", "feature"),
                feature_std_array.astype(np.float32),
            ),
            "target_train_mean": (
                ("region", "water_year"),
                target_mean_array.astype(np.float32),
            ),
            "target_train_std": (
                ("region", "water_year"),
                target_std_array.astype(np.float32),
            ),
        },
        coords={
            "region": np.asarray(region_names, dtype=object),
            "water_year": WATER_YEARS.astype(np.int32),
            "feature": np.asarray(list(feature_names), dtype=object),
            "component": np.arange(1, int(PLS_COMPONENT_GRID.max()) + 1, dtype=np.int32),
            "train_sample": np.arange(1, n_samples, dtype=np.int32),
        },
        attrs={
            "description": "Outer-LOYO PLS foldwise coefficients, loadings, and standardization metadata",
            "pls_component_grid": json.dumps(PLS_COMPONENT_GRID.tolist()),
            "target_definition": "Basin-based North, Central, and South Sierra April 1 SWE means",
            "region_mask_npz": str(REGION_MASK_NPZ_PATH),
            "region_mask_nc": str(REGION_MASK_NC_PATH),
            "coefficient_space": "Standardized predictor and standardized training-target space",
        },
    )
    return predictions_by_region, selected_components, loadings_ds, metrics, coefficient_rows


def write_predictions_csv(
    targets: Sequence[RegionTargetSeries],
    predictions_by_region: Dict[str, np.ndarray],
    selected_components: np.ndarray,
) -> None:
    target_map = {target.key: target for target in targets}
    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "water_year",
            "SWE_North_obs",
            "SWE_North_pred",
            "SWE_North_residual",
            "K_North",
            "SWE_Central_obs",
            "SWE_Central_pred",
            "SWE_Central_residual",
            "K_Central",
            "SWE_South_obs",
            "SWE_South_pred",
            "SWE_South_residual",
            "K_South",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, water_year in enumerate(WATER_YEARS):
            north_obs = float(target_map["North"].observed_m[idx])
            north_pred = float(predictions_by_region["North"][idx])
            central_obs = float(target_map["Central"].observed_m[idx])
            central_pred = float(predictions_by_region["Central"][idx])
            south_obs = float(target_map["South"].observed_m[idx])
            south_pred = float(predictions_by_region["South"][idx])
            writer.writerow(
                {
                    "water_year": int(water_year),
                    "SWE_North_obs": f"{north_obs:.12g}",
                    "SWE_North_pred": f"{north_pred:.12g}",
                    "SWE_North_residual": f"{(north_obs - north_pred):.12g}",
                    "K_North": int(selected_components[0, idx]),
                    "SWE_Central_obs": f"{central_obs:.12g}",
                    "SWE_Central_pred": f"{central_pred:.12g}",
                    "SWE_Central_residual": f"{(central_obs - central_pred):.12g}",
                    "K_Central": int(selected_components[1, idx]),
                    "SWE_South_obs": f"{south_obs:.12g}",
                    "SWE_South_pred": f"{south_pred:.12g}",
                    "SWE_South_residual": f"{(south_obs - south_pred):.12g}",
                    "K_South": int(selected_components[2, idx]),
                }
            )


def write_selected_components_csv(selected_components: np.ndarray) -> None:
    with SELECTED_COMPONENTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["water_year", "K_North", "K_Central", "K_South"],
        )
        writer.writeheader()
        for idx, water_year in enumerate(WATER_YEARS):
            writer.writerow(
                {
                    "water_year": int(water_year),
                    "K_North": int(selected_components[0, idx]),
                    "K_Central": int(selected_components[1, idx]),
                    "K_South": int(selected_components[2, idx]),
                }
            )


def write_coefficients_csv(feature_names: Sequence[str], coefficient_rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ["water_year", "region", "n_components"] + list(feature_names)
    with COEFFICIENTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in coefficient_rows:
            out_row = {
                key: (f"{value:.12g}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(out_row)


def write_metrics_json(
    feature_names: Sequence[str],
    metrics: Dict[str, Dict[str, float]],
    runtime_seconds: float,
) -> None:
    payload = {
        "experiment": "ocean_mode_only_pls",
        "script_path": str(Path(__file__).resolve()),
        "output_folder": str(OUTPUT_DIR),
        "water_year_start": int(WATER_YEAR_START),
        "water_year_end": int(WATER_YEAR_END),
        "n_years": int(WATER_YEARS.size),
        "pls_component_grid": PLS_COMPONENT_GRID.tolist(),
        "predictor_count": int(len(feature_names)),
        "predictor_table_path": str(PREDICTOR_TABLE_CSV),
        "target_table_path": str(TARGET_TABLE_CSV),
        "region_mask_npz": str(REGION_MASK_NPZ_PATH),
        "region_mask_nc": str(REGION_MASK_NC_PATH),
        "source_predictors": {
            "pacific_pc_netcdf": str(PACIFIC_PC_PATH),
            "nino34_csv": str(NINO34_CSV_PATH),
            "amv_amo_csv": str(AMV_AMO_CSV_PATH),
        },
        "regions": metrics,
        "artifacts": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "metrics_json": str(METRICS_JSON),
            "selected_components_csv": str(SELECTED_COMPONENTS_CSV),
            "coefficients_csv": str(COEFFICIENTS_CSV),
            "loadings_weights_netcdf": str(LOADINGS_WEIGHTS_NC),
            "timeseries_png": str(TIMESERIES_PNG),
            "scatter_png": str(SCATTER_PNG),
            "selected_components_png": str(COMPONENTS_PNG),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peak_memory_mb": peak_memory_mb(),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_timeseries(
    targets: Sequence[RegionTargetSeries],
    predictions_by_region: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = {"observed": "#1f4e79", "predicted": "#c76d06", "mean": "#777777"}
    for ax, target in zip(axes, targets):
        observed = target.observed_m
        predicted = predictions_by_region[target.key]
        region_metrics = metrics[target.key]
        ax.plot(WATER_YEARS, observed, color=colors["observed"], marker="o", linewidth=1.8, label="Observed")
        ax.plot(
            WATER_YEARS,
            predicted,
            color=colors["predicted"],
            marker="s",
            linewidth=1.8,
            label="LOYO predicted",
        )
        ax.axhline(np.mean(observed), color=colors["mean"], linestyle="--", linewidth=1.0, label="Observed mean")
        ax.set_ylabel("April 1 SWE (m)")
        ax.set_title(
            f"{target.key} Sierra | "
            f"R2={region_metrics['R2_LOYO']:.3f}, "
            f"RMSE={region_metrics['RMSE']:.3f} m, "
            f"MAE={region_metrics['MAE']:.3f} m, "
            f"r={region_metrics['Pearson_r']:.3f}"
        )
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("Water year")
    fig.suptitle("Ocean-mode-only LOYO PLS: observed vs predicted April 1 Sierra SWE", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(TIMESERIES_PNG, dpi=220)
    plt.close(fig)


def plot_scatter(
    targets: Sequence[RegionTargetSeries],
    predictions_by_region: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, target in zip(axes, targets):
        observed = target.observed_m
        predicted = predictions_by_region[target.key]
        region_metrics = metrics[target.key]
        vmin = float(min(np.min(observed), np.min(predicted)))
        vmax = float(max(np.max(observed), np.max(predicted)))
        ax.scatter(observed, predicted, s=28, color="#246a73", alpha=0.85)
        ax.plot([vmin, vmax], [vmin, vmax], color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Observed April 1 SWE (m)")
        ax.set_ylabel("Predicted April 1 SWE (m)")
        ax.set_title(
            f"{target.key} Sierra\n"
            f"R2={region_metrics['R2_LOYO']:.3f}, "
            f"RMSE={region_metrics['RMSE']:.3f} m, "
            f"MAE={region_metrics['MAE']:.3f} m, "
            f"r={region_metrics['Pearson_r']:.3f}"
        )
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=220)
    plt.close(fig)


def plot_selected_components(selected_components: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    bins = np.arange(0.5, int(PLS_COMPONENT_GRID.max()) + 1.6, 1.0)
    for ax, region_key, region_values in zip(axes, REGION_KEYS, selected_components):
        region_int = region_values.astype(int)
        counts = [int(np.sum(region_int == k)) for k in PLS_COMPONENT_GRID]
        ax.bar(PLS_COMPONENT_GRID, counts, color="#4c78a8", width=0.7)
        ax.set_xticks(PLS_COMPONENT_GRID)
        ax.set_xlabel("Selected PLS components")
        ax.set_title(f"{region_key} Sierra")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Outer-fold count")
    fig.suptitle("Selected PLS components across outer LOYO folds", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(COMPONENTS_PNG, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    start = perf_counter()

    feature_names, x, _ = build_predictor_matrix()
    targets, _ = build_target_table()
    predictions_by_region, selected_components, loadings_ds, metrics, coefficient_rows = run_loyo_pls(
        feature_names,
        x,
        targets,
    )

    write_predictions_csv(targets, predictions_by_region, selected_components)
    write_selected_components_csv(selected_components)
    write_coefficients_csv(feature_names, coefficient_rows)
    loadings_ds.to_netcdf(LOADINGS_WEIGHTS_NC, engine=NETCDF_ENGINE)
    plot_timeseries(targets, predictions_by_region, metrics)
    plot_scatter(targets, predictions_by_region, metrics)
    plot_selected_components(selected_components)

    runtime_seconds = perf_counter() - start
    write_metrics_json(feature_names, metrics, runtime_seconds)

    print(f"Wrote target table: {TARGET_TABLE_CSV}", flush=True)
    print(f"Wrote predictor table: {PREDICTOR_TABLE_CSV}", flush=True)
    print(f"Wrote predictions: {PREDICTIONS_CSV}", flush=True)
    print(f"Wrote metrics: {METRICS_JSON}", flush=True)
    print(f"Wrote selected components: {SELECTED_COMPONENTS_CSV}", flush=True)
    print(f"Wrote coefficients CSV: {COEFFICIENTS_CSV}", flush=True)
    print(f"Wrote loadings/weights NetCDF: {LOADINGS_WEIGHTS_NC}", flush=True)
    print(f"Wrote time-series figure: {TIMESERIES_PNG}", flush=True)
    print(f"Wrote scatter figure: {SCATTER_PNG}", flush=True)
    print(f"Wrote component-summary figure: {COMPONENTS_PNG}", flush=True)


if __name__ == "__main__":
    main()
