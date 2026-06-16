#!/usr/bin/env python3
"""
Run the ocean-mode-only Sierra SWE ridge baseline with LOYO cross-validation.
"""

import csv
import json
import os
import resource
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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
    RegionBounds,
    get_regional_swe_grid_definition,
    swe_file_for_water_year,
)


PACIFIC_PC_PATH = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_pacific_sierra_t2m_level2_pc1to6/cobe2_pacific_sierra_t2m_level2_pc1to6.nc"
)
NINO34_PATH = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "nino34" / "nino34_monthly_wy1985_2021_sep_mar.nc"
AMV_AMO_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_eofs_pc1to6.nc"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "ocean_mode_only_ridge"
PREDICTIONS_CSV = OUTPUT_DIR / "loyo_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "loyo_metrics.json"
COEFFICIENTS_NC = OUTPUT_DIR / "loyo_coefficients.nc"
TIMESERIES_PNG = OUTPUT_DIR / "observed_vs_predicted_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "predicted_vs_observed_scatter.png"
SUMMARY_JSON = OUTPUT_DIR / "run_summary.json"

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
RIDGE_ALPHAS = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
NETCDF_ENGINE = "netcdf4"
MEAN_STAT_INDEX = 0
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class TargetRegion:
    key: str
    title: str
    short_name: str
    region_bounds: RegionBounds


@dataclass(frozen=True)
class RegionTargetSeries:
    region: TargetRegion
    mean_m: np.ndarray
    anom_m: np.ndarray


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


def infer_coordinate_bounds(values: xr.DataArray) -> np.ndarray:
    centers = np.asarray(values.values, dtype=np.float64)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    bounds = np.empty(centers.size + 1, dtype=np.float64)
    bounds[1:-1] = midpoints
    bounds[0] = centers[0] - (midpoints[0] - centers[0])
    bounds[-1] = centers[-1] + (centers[-1] - midpoints[-1])
    return bounds


def cell_area_from_bounds(latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
    lat_bounds_deg = infer_coordinate_bounds(latitude)
    lon_bounds_deg = infer_coordinate_bounds(longitude)
    lat_bounds_rad = np.deg2rad(lat_bounds_deg)
    lon_bounds_rad = np.deg2rad(lon_bounds_deg)
    lat_band = np.abs(np.sin(lat_bounds_rad[1:]) - np.sin(lat_bounds_rad[:-1]))
    lon_band = np.abs(lon_bounds_rad[1:] - lon_bounds_rad[:-1])
    area = (EARTH_RADIUS_M**2) * lat_band[:, None] * lon_band[None, :]
    return xr.DataArray(
        area.astype(np.float64),
        dims=(latitude.dims[0], longitude.dims[0]),
        coords={latitude.dims[0]: latitude, longitude.dims[0]: longitude},
        name="grid_cell_area_m2",
    )


def build_subregions() -> List[TargetRegion]:
    lat_edges = np.linspace(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max, 4)
    return [
        TargetRegion(
            key="south",
            title="South Sierra",
            short_name="S",
            region_bounds=RegionBounds(
                lat_min=float(lat_edges[0]),
                lat_max=float(lat_edges[1]),
                lon_min=DEFAULT_SIERRA_REGION.lon_min,
                lon_max=DEFAULT_SIERRA_REGION.lon_max,
            ),
        ),
        TargetRegion(
            key="middle",
            title="Middle Sierra",
            short_name="M",
            region_bounds=RegionBounds(
                lat_min=float(lat_edges[1]),
                lat_max=float(lat_edges[2]),
                lon_min=DEFAULT_SIERRA_REGION.lon_min,
                lon_max=DEFAULT_SIERRA_REGION.lon_max,
            ),
        ),
        TargetRegion(
            key="north",
            title="North Sierra",
            short_name="N",
            region_bounds=RegionBounds(
                lat_min=float(lat_edges[2]),
                lat_max=float(lat_edges[3]),
                lon_min=DEFAULT_SIERRA_REGION.lon_min,
                lon_max=DEFAULT_SIERRA_REGION.lon_max,
            ),
        ),
    ]


def build_region_weight_masks() -> Tuple[xr.DataArray, List[Tuple[TargetRegion, xr.DataArray]]]:
    swe_grid = get_regional_swe_grid_definition(
        water_year=WATER_YEAR_START,
        region=DEFAULT_SIERRA_REGION,
        coarsen_factor=1,
    )
    latitude = swe_grid.latitude
    longitude = swe_grid.longitude
    area = cell_area_from_bounds(latitude, longitude)
    lat2d, lon2d = xr.broadcast(latitude, longitude)

    full_mask = (
        (lat2d >= DEFAULT_SIERRA_REGION.lat_min)
        & (lat2d <= DEFAULT_SIERRA_REGION.lat_max)
        & (lon2d >= DEFAULT_SIERRA_REGION.lon_min)
        & (lon2d <= DEFAULT_SIERRA_REGION.lon_max)
    ).astype(np.float64)

    regions = build_subregions()
    masks: List[Tuple[TargetRegion, xr.DataArray]] = []
    for idx, region in enumerate(regions):
        upper_inclusive = idx == (len(regions) - 1)
        lat_mask = lat2d >= region.region_bounds.lat_min
        if upper_inclusive:
            lat_mask = lat_mask & (lat2d <= region.region_bounds.lat_max)
        else:
            lat_mask = lat_mask & (lat2d < region.region_bounds.lat_max)
        mask = (
            lat_mask
            & (lon2d >= region.region_bounds.lon_min)
            & (lon2d <= region.region_bounds.lon_max)
            & (full_mask > 0.0)
        ).astype(np.float64)
        masks.append((region, (mask * area).rename(f"{region.key}_weight_m2")))
    return area, masks


def apr1_region_mean_for_water_year(water_year: int, weights: xr.DataArray) -> float:
    path = swe_file_for_water_year(water_year)
    target_date = f"{water_year}-04-01"
    lat_name, lon_name = weights.dims
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_times=True) as ds:
        swe = ds[SWE_VARIABLE].isel(Stats=MEAN_STAT_INDEX, drop=True)
        swe = swe.sel(time=np.datetime64(target_date))
        swe = swe.sel({lat_name: weights[lat_name], lon_name: weights[lon_name]})
        values = np.asarray(swe.where(swe != SWE_MISSING_VALUE).values, dtype=np.float64)
    weight_values = np.asarray(weights.values, dtype=np.float64)
    valid = np.isfinite(values)
    numerator = np.where(valid, values * weight_values, 0.0).sum()
    denominator = np.where(valid, weight_values, 0.0).sum()
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def build_apr1_targets() -> List[RegionTargetSeries]:
    _, region_weights = build_region_weight_masks()
    means_by_region: Dict[str, List[float]] = {region.key: [] for region, _ in region_weights}
    for water_year in WATER_YEARS:
        for region, weights in region_weights:
            value = apr1_region_mean_for_water_year(int(water_year), weights)
            means_by_region[region.key].append(value)
        print(f"processed Apr 1 SWE targets for WY{water_year}", flush=True)

    outputs: List[RegionTargetSeries] = []
    for region, _ in region_weights:
        mean_m = np.asarray(means_by_region[region.key], dtype=np.float64)
        anom_m = mean_m - float(np.nanmean(mean_m))
        outputs.append(RegionTargetSeries(region=region, mean_m=mean_m, anom_m=anom_m))
    return outputs


def monthly_keys_for_water_years() -> List[str]:
    keys: List[str] = []
    for water_year in WATER_YEARS:
        for _, year_offset, month in MONTH_SPECS:
            keys.append(f"{int(water_year + year_offset):04d}-{month:02d}-01")
    return keys


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
            if key not in time_to_index:
                raise KeyError(f"Missing Pacific PC timestamp {key}")
            time_idx = time_to_index[key]
            for mode_idx in range(6):
                rows[wy_idx, col_idx] = float(pcs[time_idx, mode_idx])
                columns.append(f"Pacific_PC{mode_idx + 1}_{month_name}") if wy_idx == 0 else None
                col_idx += 1
    return columns, rows


def load_nino34_predictors() -> Tuple[List[str], np.ndarray]:
    with xr.open_dataset(NINO34_PATH, engine=NETCDF_ENGINE) as ds:
        water_years = np.asarray(ds["water_year"].values, dtype=np.int32)
        anomalies = np.asarray(ds["nino34_index_anomaly"].values, dtype=np.float64)
        months = [str(value) for value in ds["month"].values.tolist()]
    if not np.array_equal(water_years, WATER_YEARS):
        raise ValueError("Niño 3.4 water years do not match WY1985--WY2021")
    return [f"Nino34_{month}" for month in months], anomalies


def load_amv_predictors() -> Tuple[List[str], np.ndarray]:
    with xr.open_dataset(AMV_AMO_PATH, engine=NETCDF_ENGINE) as ds:
        times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        pcs = np.asarray(ds["pc"].values, dtype=np.float64)

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
            if key not in time_to_index:
                raise KeyError(f"Missing AMV/AMO PC timestamp {key}")
            time_idx = time_to_index[key]
            for mode_idx in range(6):
                rows[wy_idx, col_idx] = float(pcs[time_idx, mode_idx])
                columns.append(f"AMV_PC{mode_idx + 1}_{month_name}") if wy_idx == 0 else None
                col_idx += 1
    return columns, rows


def build_feature_matrix() -> Tuple[List[str], np.ndarray]:
    pacific_columns, pacific = load_pacific_predictors()
    nino_columns, nino = load_nino34_predictors()
    amv_columns, amv = load_amv_predictors()
    columns = pacific_columns + nino_columns + amv_columns
    matrix = np.concatenate([pacific, nino, amv], axis=1)
    if matrix.shape != (WATER_YEARS.size, len(columns)):
        raise ValueError(f"Unexpected feature matrix shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return columns, matrix


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


def fit_ridge_coefficients(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    ridge = gram + alpha * np.eye(gram.shape[0], dtype=np.float64)
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(ridge, rhs)


def predict_with_coefficients(x_std: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return x_std @ beta


def select_alpha_inner_loyo(x_train_std: np.ndarray, y_train_std: np.ndarray) -> float:
    best_alpha = float(RIDGE_ALPHAS[0])
    best_mse = float("inf")
    n_samples = x_train_std.shape[0]
    for alpha in RIDGE_ALPHAS:
        preds = np.full(n_samples, np.nan, dtype=np.float64)
        for inner_idx in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[inner_idx] = False
            beta = fit_ridge_coefficients(x_train_std[mask], y_train_std[mask], float(alpha))
            preds[inner_idx] = float(predict_with_coefficients(x_train_std[~mask], beta)[0])
        mse = float(np.mean((y_train_std - preds) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
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


def run_lowo_ridge(
    feature_names: Sequence[str],
    x: np.ndarray,
    targets: Sequence[RegionTargetSeries],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], xr.Dataset]:
    n_samples, n_features = x.shape
    region_keys = [target.region.key for target in targets]

    predictions_by_region: Dict[str, np.ndarray] = {}
    selected_alpha = np.full((len(targets), n_samples), np.nan, dtype=np.float64)
    beta_array = np.full((len(targets), n_samples, n_features), np.nan, dtype=np.float64)
    feature_mean_array = np.full((n_samples, n_features), np.nan, dtype=np.float64)
    feature_std_array = np.full((n_samples, n_features), np.nan, dtype=np.float64)
    target_mean_array = np.full((len(targets), n_samples), np.nan, dtype=np.float64)
    target_std_array = np.full((len(targets), n_samples), np.nan, dtype=np.float64)

    for outer_idx in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[outer_idx] = False
        test_mask = ~train_mask
        x_train_std, x_test_std, feat_mean, feat_std = standardize_training_features(x[train_mask], x[test_mask])
        feature_mean_array[outer_idx, :] = feat_mean
        feature_std_array[outer_idx, :] = feat_std

        for region_idx, target in enumerate(targets):
            y = target.mean_m
            y_train_std, y_mean, y_std = standardize_training_target(y[train_mask])
            alpha = select_alpha_inner_loyo(x_train_std, y_train_std)
            beta = fit_ridge_coefficients(x_train_std, y_train_std, alpha)
            y_pred_std = float(predict_with_coefficients(x_test_std, beta)[0])
            y_pred = y_pred_std * y_std + y_mean

            region_key = target.region.key
            if region_key not in predictions_by_region:
                predictions_by_region[region_key] = np.full(n_samples, np.nan, dtype=np.float64)
            predictions_by_region[region_key][outer_idx] = y_pred
            selected_alpha[region_idx, outer_idx] = alpha
            beta_array[region_idx, outer_idx, :] = beta
            target_mean_array[region_idx, outer_idx] = y_mean
            target_std_array[region_idx, outer_idx] = y_std
            print(
                f"LOYO WY{int(WATER_YEARS[outer_idx])} region={region_key} alpha={alpha:g} "
                f"obs={float(y[outer_idx]):.6f} pred={float(y_pred):.6f}",
                flush=True,
            )

    metrics: Dict[str, object] = {"regions": {}}
    for target in targets:
        region_key = target.region.key
        observed = target.mean_m
        predicted = predictions_by_region[region_key]
        metrics["regions"][region_key] = {
            "title": target.region.title,
            "r2_loyo": r2_score_manual(observed, predicted),
            "rmse_m": rmse(observed, predicted),
            "mae_m": mae(observed, predicted),
            "pearson_r": correlation(observed, predicted),
            "observed_mean_m": float(np.mean(observed)),
        }

    coeff_ds = xr.Dataset(
        data_vars={
            "selected_alpha": (("region", "water_year"), selected_alpha.astype(np.float32)),
            "beta_standardized_feature_space": (
                ("region", "water_year", "feature"),
                beta_array.astype(np.float32),
            ),
            "feature_mean": (("water_year", "feature"), feature_mean_array.astype(np.float32)),
            "feature_std": (("water_year", "feature"), feature_std_array.astype(np.float32)),
            "target_mean_m": (("region", "water_year"), target_mean_array.astype(np.float32)),
            "target_std_m": (("region", "water_year"), target_std_array.astype(np.float32)),
        },
        coords={
            "region": np.asarray(region_keys, dtype=object),
            "water_year": WATER_YEARS.astype(np.int32),
            "feature": np.asarray(list(feature_names), dtype=object),
        },
        attrs={
            "description": "Ocean-mode-only SWE ridge LOYO coefficients and foldwise standardization metadata",
            "ridge_alpha_grid": json.dumps(RIDGE_ALPHAS.tolist()),
            "feature_definition": "Pacific PC1-6 Sep-Mar, Niño 3.4 Sep-Mar, AMV PC1-6 Sep-Mar",
            "coefficient_space": "beta for standardized feature space and standardized training target",
        },
    )
    return predictions_by_region, metrics, coeff_ds


def write_predictions_csv(targets: Sequence[RegionTargetSeries], predictions_by_region: Dict[str, np.ndarray]) -> None:
    fieldnames = [
        "water_year",
        "north_observed_m",
        "north_predicted_m",
        "north_residual_m",
        "middle_observed_m",
        "middle_predicted_m",
        "middle_residual_m",
        "south_observed_m",
        "south_predicted_m",
        "south_residual_m",
    ]
    by_key = {target.region.key: target for target in targets}
    with PREDICTIONS_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for idx, water_year in enumerate(WATER_YEARS):
            north_obs = float(by_key["north"].mean_m[idx])
            north_pred = float(predictions_by_region["north"][idx])
            middle_obs = float(by_key["middle"].mean_m[idx])
            middle_pred = float(predictions_by_region["middle"][idx])
            south_obs = float(by_key["south"].mean_m[idx])
            south_pred = float(predictions_by_region["south"][idx])
            writer.writerow(
                [
                    int(water_year),
                    f"{north_obs:.12g}",
                    f"{north_pred:.12g}",
                    f"{(north_obs - north_pred):.12g}",
                    f"{middle_obs:.12g}",
                    f"{middle_pred:.12g}",
                    f"{(middle_obs - middle_pred):.12g}",
                    f"{south_obs:.12g}",
                    f"{south_pred:.12g}",
                    f"{(south_obs - south_pred):.12g}",
                ]
            )


def plot_timeseries(targets: Sequence[RegionTargetSeries], predictions_by_region: Dict[str, np.ndarray], metrics: Dict[str, object]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    order = ["north", "middle", "south"]
    colors = {"observed": "#1f4e79", "predicted": "#c76d06", "mean": "#777777"}
    target_map = {target.region.key: target for target in targets}
    for ax, region_key in zip(axes, order):
        target = target_map[region_key]
        observed = target.mean_m
        predicted = predictions_by_region[region_key]
        region_metrics = metrics["regions"][region_key]
        ax.plot(WATER_YEARS, observed, color=colors["observed"], marker="o", linewidth=1.8, label="Observed")
        ax.plot(WATER_YEARS, predicted, color=colors["predicted"], marker="s", linewidth=1.8, label="LOYO predicted")
        ax.axhline(np.mean(observed), color=colors["mean"], linestyle="--", linewidth=1.0, label="Observed mean")
        ax.set_ylabel("April 1 SWE (m)")
        ax.set_title(
            f"{target.region.title} | "
            f"R2={region_metrics['r2_loyo']:.3f}, "
            f"RMSE={region_metrics['rmse_m']:.3f} m, "
            f"MAE={region_metrics['mae_m']:.3f} m, "
            f"r={region_metrics['pearson_r']:.3f}"
        )
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("Water year")
    fig.suptitle("Ocean-mode-only LOYO ridge: observed vs predicted April 1 Sierra SWE", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(TIMESERIES_PNG, dpi=220)
    plt.close(fig)


def plot_scatter(targets: Sequence[RegionTargetSeries], predictions_by_region: Dict[str, np.ndarray], metrics: Dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=False)
    order = ["north", "middle", "south"]
    target_map = {target.region.key: target for target in targets}
    for ax, region_key in zip(axes, order):
        target = target_map[region_key]
        observed = target.mean_m
        predicted = predictions_by_region[region_key]
        region_metrics = metrics["regions"][region_key]
        vmin = float(min(np.min(observed), np.min(predicted)))
        vmax = float(max(np.max(observed), np.max(predicted)))
        ax.scatter(observed, predicted, s=28, color="#246a73", alpha=0.85)
        ax.plot([vmin, vmax], [vmin, vmax], color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Observed April 1 SWE (m)")
        ax.set_ylabel("Predicted April 1 SWE (m)")
        ax.set_title(f"{target.region.title}\nR2={region_metrics['r2_loyo']:.3f}, r={region_metrics['pearson_r']:.3f}")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=220)
    plt.close(fig)


def write_metrics_json(
    feature_names: Sequence[str],
    targets: Sequence[RegionTargetSeries],
    metrics: Dict[str, object],
    runtime_seconds: float,
) -> None:
    region_payload = {}
    for target in targets:
        region_payload[target.region.key] = {
            **metrics["regions"][target.region.key],
            "subregion_bounds": asdict(target.region.region_bounds),
        }
    payload = {
        "experiment": "ocean_mode_only_swe_ridge",
        "water_year_start": WATER_YEAR_START,
        "water_year_end": WATER_YEAR_END,
        "n_samples": int(WATER_YEARS.size),
        "predictor_count": int(len(feature_names)),
        "ridge_alpha_grid": RIDGE_ALPHAS.tolist(),
        "predictor_columns": list(feature_names),
        "subregion_definition": (
            "North/Middle/South Sierra are defined here as three equal-width latitude bands "
            "within the existing Sierra box 35N-42N, 122.5W-118W."
        ),
        "regions": region_payload,
        "source_predictors": {
            "pacific_pc_netcdf": str(PACIFIC_PC_PATH),
            "nino34_netcdf": str(NINO34_PATH),
            "amv_amo_netcdf": str(AMV_AMO_PATH),
        },
        "artifacts": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "metrics_json": str(METRICS_JSON),
            "coefficients_netcdf": str(COEFFICIENTS_NC),
            "timeseries_png": str(TIMESERIES_PNG),
            "scatter_png": str(SCATTER_PNG),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peak_memory_mb": peak_memory_mb(),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_run_summary(feature_names: Sequence[str], targets: Sequence[RegionTargetSeries], runtime_seconds: float) -> None:
    payload = {
        "experiment": "ocean_mode_only_swe_ridge",
        "water_years": WATER_YEARS.tolist(),
        "month_sequence": [name for name, _, _ in MONTH_SPECS],
        "predictor_columns": list(feature_names),
        "predictor_groups": {
            "pacific_pc_columns": 42,
            "nino34_columns": 7,
            "amv_pc_columns": 42,
        },
        "target_regions": [
            {
                "key": target.region.key,
                "title": target.region.title,
                "bounds": asdict(target.region.region_bounds),
            }
            for target in targets
        ],
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb(),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    start = perf_counter()

    feature_names, x = build_feature_matrix()
    targets = build_apr1_targets()
    predictions_by_region, metrics, coeff_ds = run_lowo_ridge(feature_names, x, targets)

    write_predictions_csv(targets, predictions_by_region)
    coeff_ds.to_netcdf(COEFFICIENTS_NC, engine=NETCDF_ENGINE)
    runtime_seconds = perf_counter() - start
    write_metrics_json(feature_names, targets, metrics, runtime_seconds)
    write_run_summary(feature_names, targets, runtime_seconds)
    plot_timeseries(targets, predictions_by_region, metrics)
    plot_scatter(targets, predictions_by_region, metrics)

    print(f"Wrote predictions: {PREDICTIONS_CSV}", flush=True)
    print(f"Wrote metrics: {METRICS_JSON}", flush=True)
    print(f"Wrote coefficients: {COEFFICIENTS_NC}", flush=True)
    print(f"Wrote time-series figure: {TIMESERIES_PNG}", flush=True)
    print(f"Wrote scatter figure: {SCATTER_PNG}", flush=True)
    print(f"Wrote summary: {SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
