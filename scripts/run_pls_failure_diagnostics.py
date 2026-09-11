#!/usr/bin/env python3
"""
Diagnose why vanilla PLS failed for scalar April 1 Sierra-average SWE.
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cross_decomposition import PLSRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "pls_failure_diagnostics"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
BASELINE_SUMMARY_CSV = OUTPUT_DIR / "baseline_summary.csv"
README_MD = OUTPUT_DIR / "README.md"

AMV_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)
PATCH_PREDICTORS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_predictors.csv"
)
BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)
SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=int)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
Z1_NAME = "Z1_M1_Jan_lat_-9.5_lon_133.5"
Z2_NAME = "Z2_M2_Oct_lat_0.5_lon_136.5"
Z_COLUMNS = ["Z1", "Z2"]
AMV_LAT_MIN = 0.0
AMV_LAT_MAX = 70.0
AMV_LON_MIN_360 = 280.0
AMV_LON_MAX_360 = 360.0
NETCDF_ENGINES = [None, "netcdf4", "h5netcdf", "scipy"]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    kind: str
    columns: Optional[Tuple[str, ...]] = None
    region_label: Optional[str] = None


EXPERIMENTS: Tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="amv_amo_42pcs_only",
        description="AMV/AMO 42 PCs only",
        kind="table",
        columns=tuple(f"AMV_PC{pc}_{month}" for month in MONTHS for pc in range(1, 7)),
    ),
    ExperimentSpec(
        name="amv_amo_42pcs_plus_oracle_z1z2",
        description="AMV/AMO 42 PCs plus oracle Z1, Z2",
        kind="table",
        columns=("Z1", "Z2") + tuple(f"AMV_PC{pc}_{month}" for month in MONTHS for pc in range(1, 7)),
    ),
    ExperimentSpec(
        name="raw_sst_amv_amo_region",
        description="Raw SST restricted to AMV/AMO region",
        kind="raw_sst",
        region_label="amv_amo_region",
    ),
    ExperimentSpec(
        name="raw_sst_full_passive",
        description="Full passive raw SST",
        kind="raw_sst",
        region_label="full_passive",
    ),
)


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    last_error = None  # type: Optional[Exception]
    for engine in NETCDF_ENGINES:
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Unable to open dataset {path}: {last_error}")


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    yy = y_true[mask]
    pp = y_pred[mask]
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "Pearson_r": corrcoef_safe(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
    }


def standardize_training_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=1)
    std = np.where(std == 0.0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


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
    max_components = min(10, n_train_samples - 1, n_features)
    return np.arange(1, max_components + 1, dtype=int)


def load_scalar_target_table() -> pd.DataFrame:
    predictors = pd.read_csv(PATCH_PREDICTORS_CSV)
    predictors = predictors.loc[predictors["patch_size"] == "exact_grid_cell", ["water_year", Z1_NAME, Z2_NAME]].copy()
    predictors = predictors.rename(columns={Z1_NAME: "Z1", Z2_NAME: "Z2"})
    predictors["water_year"] = predictors["water_year"].astype(int)

    targets = pd.read_csv(BASE_PREDICTIONS_CSV)
    targets = targets.loc[
        (targets["patch_size"] == "exact_grid_cell") & (targets["model_name"] == "Z1_Z2"),
        ["heldout_wy", "obs_swe"],
    ].copy()
    targets = targets.rename(columns={"heldout_wy": "water_year"})
    targets["water_year"] = targets["water_year"].astype(int)

    amv = pd.read_csv(AMV_CSV)
    amv["water_year"] = amv["water_year"].astype(int)
    amv_columns = [column for column in amv.columns if column.startswith("AMV_PC")]
    if len(amv_columns) != 42:
        raise ValueError(f"Expected 42 AMV/AMO predictor columns, found {len(amv_columns)}")

    table = predictors.merge(targets, on="water_year", how="inner")
    table = table.merge(amv[["water_year"] + amv_columns], on="water_year", how="inner")
    table = table.sort_values("water_year").reset_index(drop=True)
    if table["water_year"].tolist() != WATER_YEARS.tolist():
        raise ValueError("Predictor/target table does not match WY1985--WY2021.")
    return table


def load_global_sst_cube() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_times: List[np.datetime64] = []
    for water_year in WATER_YEARS:
        for month_name in MONTHS:
            calendar_year = water_year - 1 if month_name in {"Sep", "Oct", "Nov", "Dec"} else water_year
            selected_times.append(np.datetime64(f"{calendar_year:04d}-{MONTH_TO_NUMBER[month_name]:02d}-01"))

    with open_dataset_with_fallbacks(SST_FILE) as ds:
        if "sst" not in ds:
            raise KeyError(f"Expected variable 'sst' in {SST_FILE}")
        sst = ds["sst"].sel(time=selected_times).load()
        cube = np.asarray(sst.values, dtype=np.float32).reshape(
            len(WATER_YEARS), len(MONTHS), sst.sizes["lat"], sst.sizes["lon"]
        )
        lat = np.asarray(sst["lat"].values, dtype=np.float64)
        lon = np.asarray(sst["lon"].values, dtype=np.float64)
        missing_value = sst.attrs.get("missing_value")
        fill_value = sst.attrs.get("_FillValue")
        if missing_value is not None:
            cube = np.where(cube == float(missing_value), np.nan, cube)
        if fill_value is not None:
            cube = np.where(cube == float(fill_value), np.nan, cube)
    return np.asarray(cube, dtype=np.float32), lat, lon


def build_region_valid_mask(
    cube: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    region_label: str,
) -> np.ndarray:
    if region_label == "full_passive":
        spatial_mask = np.ones((lat.size, lon.size), dtype=bool)
    elif region_label == "amv_amo_region":
        spatial_mask = (
            (lat[:, None] >= AMV_LAT_MIN)
            & (lat[:, None] <= AMV_LAT_MAX)
            & (lon[None, :] >= AMV_LON_MIN_360)
            & (lon[None, :] < AMV_LON_MAX_360)
        )
    else:
        raise ValueError(f"Unknown region label: {region_label}")

    valid_mask = np.all(np.isfinite(cube), axis=0) & spatial_mask[None, :, :]
    if not np.any(valid_mask):
        raise ValueError(f"No valid SST features found for region {region_label}")
    return valid_mask


def prepare_weighted_features(
    train_cube_raw: np.ndarray,
    test_cube_raw: np.ndarray,
    valid_feature_mask: np.ndarray,
    lat_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        monthly_clim = np.nanmean(train_cube_raw, axis=0, dtype=np.float64)
    train_anom = np.asarray(train_cube_raw, dtype=np.float64) - monthly_clim[None, :, :, :]
    test_anom = np.asarray(test_cube_raw, dtype=np.float64) - monthly_clim
    train_weighted = train_anom * lat_weights[None, None, :, None]
    test_weighted = test_anom * lat_weights[None, :, None]
    valid_flat = valid_feature_mask.reshape(-1)
    x_train = train_weighted.reshape(train_weighted.shape[0], -1)[:, valid_flat]
    x_test = test_weighted.reshape(-1)[valid_flat]
    return x_train, x_test


def select_components_inner_loyo_matrix(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> int:
    candidate_grid = choose_component_grid(x_train_raw.shape[0], x_train_raw.shape[1])
    best_components = int(candidate_grid[0])
    best_mse = float("inf")
    tolerance = 1.0e-12
    n_samples = x_train_raw.shape[0]
    for n_components in candidate_grid:
        sq_errors = np.full(n_samples, np.nan, dtype=np.float64)
        for inner_idx in range(n_samples):
            inner_mask = np.ones(n_samples, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train = x_train_raw[inner_mask]
            x_inner_valid = x_train_raw[~inner_mask]
            y_inner_train = y_train_raw[inner_mask]
            y_inner_valid = float(y_train_raw[~inner_mask][0])
            x_inner_train_std, x_inner_valid_std = standardize_training_features(x_inner_train, x_inner_valid)
            y_inner_train_std, y_mean, y_std = standardize_training_target(y_inner_train)
            model = fit_pls_model(x_inner_train_std, y_inner_train_std, int(n_components))
            pred_std = float(model.predict(x_inner_valid_std).ravel()[0])
            pred_raw = y_mean + y_std * pred_std
            sq_errors[inner_idx] = (pred_raw - y_inner_valid) ** 2
        mse = float(np.mean(sq_errors))
        if mse < (best_mse - tolerance) or (abs(mse - best_mse) <= tolerance and int(n_components) < best_components):
            best_mse = mse
            best_components = int(n_components)
    return best_components


def select_components_inner_loyo_raw_sst(
    train_cube_raw: np.ndarray,
    y_train_raw: np.ndarray,
    valid_feature_mask: np.ndarray,
    lat_weights: np.ndarray,
) -> int:
    feature_count = int(np.sum(valid_feature_mask))
    candidate_grid = choose_component_grid(train_cube_raw.shape[0], feature_count)
    best_components = int(candidate_grid[0])
    best_mse = float("inf")
    tolerance = 1.0e-12
    n_samples = train_cube_raw.shape[0]
    for n_components in candidate_grid:
        sq_errors = np.full(n_samples, np.nan, dtype=np.float64)
        for inner_idx in range(n_samples):
            inner_mask = np.ones(n_samples, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train, x_inner_valid = prepare_weighted_features(
                train_cube_raw[inner_mask],
                train_cube_raw[~inner_mask][0],
                valid_feature_mask,
                lat_weights,
            )
            y_inner_train = y_train_raw[inner_mask]
            y_inner_valid = float(y_train_raw[~inner_mask][0])
            x_inner_train_std, x_inner_valid_std = standardize_training_features(x_inner_train, x_inner_valid[None, :])
            y_inner_train_std, y_mean, y_std = standardize_training_target(y_inner_train)
            model = fit_pls_model(x_inner_train_std, y_inner_train_std, int(n_components))
            pred_std = float(model.predict(x_inner_valid_std).ravel()[0])
            pred_raw = y_mean + y_std * pred_std
            sq_errors[inner_idx] = (pred_raw - y_inner_valid) ** 2
        mse = float(np.mean(sq_errors))
        if mse < (best_mse - tolerance) or (abs(mse - best_mse) <= tolerance and int(n_components) < best_components):
            best_mse = mse
            best_components = int(n_components)
    return best_components


def run_table_experiment(spec: ExperimentSpec, table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    if spec.columns is None:
        raise ValueError(f"Missing columns for table experiment {spec.name}")
    years = table["water_year"].to_numpy(dtype=int)
    y_all = table["obs_swe"].to_numpy(dtype=np.float64)
    x_all = table[list(spec.columns)].to_numpy(dtype=np.float64)
    predictions: List[Dict[str, Union[float, int]]] = []
    baseline_predictions = np.full(years.size, np.nan, dtype=np.float64)
    model_predictions = np.full(years.size, np.nan, dtype=np.float64)

    for outer_idx, water_year in enumerate(years):
        train_mask = np.ones(years.size, dtype=bool)
        train_mask[outer_idx] = False
        x_train_raw = x_all[train_mask]
        x_test_raw = x_all[~train_mask]
        y_train_raw = y_all[train_mask]
        y_test = float(y_all[outer_idx])

        x_train_std, x_test_std = standardize_training_features(x_train_raw, x_test_raw)
        y_train_std, y_mean, y_std = standardize_training_target(y_train_raw)
        n_components = select_components_inner_loyo_matrix(x_train_raw, y_train_raw)
        model = fit_pls_model(x_train_std, y_train_std, n_components)
        pred_std = float(model.predict(x_test_std).ravel()[0])
        pred_raw = y_mean + y_std * pred_std
        baseline_pred = float(np.mean(y_train_raw))

        baseline_predictions[outer_idx] = baseline_pred
        model_predictions[outer_idx] = pred_raw
        predictions.append(
            {
                "water_year": int(water_year),
                "y_true": y_test,
                "y_pred": pred_raw,
                "residual": y_test - pred_raw,
                "selected_n_components": int(n_components),
            }
        )
        print(
            f"{spec.name}: LOYO heldout_WY={int(water_year)} n_components={int(n_components)} "
            f"obs={y_test:.6f} pred={pred_raw:.6f} baseline={baseline_pred:.6f}",
            flush=True,
        )

    pred_df = pd.DataFrame(predictions)
    metrics = compute_metric_bundle(y_all, model_predictions)
    metrics["n_features"] = float(x_all.shape[1])
    metrics["mean_selected_components"] = float(pred_df["selected_n_components"].mean())
    baseline_metrics = compute_metric_bundle(y_all, baseline_predictions)
    return pred_df, metrics, baseline_metrics


def run_raw_sst_experiment(
    spec: ExperimentSpec,
    y_all: np.ndarray,
    raw_cube: np.ndarray,
    lat: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    years = WATER_YEARS.copy()
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    predictions: List[Dict[str, Union[float, int]]] = []
    baseline_predictions = np.full(years.size, np.nan, dtype=np.float64)
    model_predictions = np.full(years.size, np.nan, dtype=np.float64)

    for outer_idx, water_year in enumerate(years):
        train_mask = np.ones(years.size, dtype=bool)
        train_mask[outer_idx] = False
        train_cube = raw_cube[train_mask]
        test_cube = raw_cube[~train_mask][0]
        y_train_raw = y_all[train_mask]
        y_test = float(y_all[outer_idx])

        x_train_raw, x_test_raw = prepare_weighted_features(train_cube, test_cube, valid_feature_mask, lat_weights)
        x_train_std, x_test_std = standardize_training_features(x_train_raw, x_test_raw[None, :])
        y_train_std, y_mean, y_std = standardize_training_target(y_train_raw)
        n_components = select_components_inner_loyo_raw_sst(train_cube, y_train_raw, valid_feature_mask, lat_weights)
        model = fit_pls_model(x_train_std, y_train_std, n_components)
        pred_std = float(model.predict(x_test_std).ravel()[0])
        pred_raw = y_mean + y_std * pred_std
        baseline_pred = float(np.mean(y_train_raw))

        baseline_predictions[outer_idx] = baseline_pred
        model_predictions[outer_idx] = pred_raw
        predictions.append(
            {
                "water_year": int(water_year),
                "y_true": y_test,
                "y_pred": pred_raw,
                "residual": y_test - pred_raw,
                "selected_n_components": int(n_components),
            }
        )
        print(
            f"{spec.name}: LOYO heldout_WY={int(water_year)} n_components={int(n_components)} "
            f"obs={y_test:.6f} pred={pred_raw:.6f} baseline={baseline_pred:.6f}",
            flush=True,
        )

    pred_df = pd.DataFrame(predictions)
    metrics = compute_metric_bundle(y_all, model_predictions)
    metrics["n_features"] = float(np.sum(valid_feature_mask))
    metrics["mean_selected_components"] = float(pred_df["selected_n_components"].mean())
    baseline_metrics = compute_metric_bundle(y_all, baseline_predictions)
    return pred_df, metrics, baseline_metrics


def make_obs_vs_pred_plot(pred_df: pd.DataFrame, experiment_name: str, out_path: Path, metrics: Dict[str, float]) -> None:
    years = pred_df["water_year"].to_numpy(dtype=int)
    obs = pred_df["y_true"].to_numpy(dtype=float)
    pred = pred_df["y_pred"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(years, obs, color="black", linewidth=2.2, label="Observed")
    ax.plot(years, pred, color="#1f77b4", linewidth=1.8, label="PLS LOYO")
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra-average SWE anomaly (m)")
    ax.set_title(
        f"{experiment_name}\nR2={metrics['R2']:.3f} RMSE={metrics['RMSE']:.4f} "
        f"r={metrics['Pearson_r']:.3f} sign={metrics['sign_accuracy']:.3f}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def conclusion_line(condition: bool, yes_text: str, no_text: str) -> str:
    return yes_text if condition else no_text


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if np.isfinite(value):
                    values.append(f"{value:.6f}")
                else:
                    values.append("nan")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + rows)


def build_readme(
    summary_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_paths: Dict[str, Dict[str, str]],
) -> str:
    summary_lookup = summary_df.set_index("experiment")
    baseline_lookup = baseline_df.set_index("experiment")
    amv_only = summary_lookup.loc["amv_amo_42pcs_only"]
    amv_plus_z = summary_lookup.loc["amv_amo_42pcs_plus_oracle_z1z2"]
    amv_raw = summary_lookup.loc["raw_sst_amv_amo_region"]
    full_raw = summary_lookup.loc["raw_sst_full_passive"]

    lines = [
        "# PLS Failure Diagnostics",
        "",
        "This diagnostic ran four strict outer-LOYO scalar April 1 Sierra-average SWE experiments with train-fold-only preprocessing, inner-LOYO tuning of `n_components`, and inverse-transformed held-out predictions.",
        "",
        "## Headline answers",
        "",
        f"- Does PLS work on AMV/AMO PCs only? {conclusion_line(float(amv_only['R2']) > 0.0, 'Yes, it shows positive LOYO skill.', 'No, it does not show useful positive LOYO skill.')}",
        f"- Does PLS recover the strong baseline when oracle Z1,Z2 are added? {conclusion_line(float(amv_plus_z['R2']) > float(amv_only['R2']) and float(amv_plus_z['RMSE']) < float(amv_only['RMSE']), 'Yes, adding oracle Z1,Z2 materially improves the PLS result.', 'No, adding oracle Z1,Z2 does not recover a strong LOYO baseline.')}",
        f"- Does raw AMV/AMO-region SST fail compared with AMV/AMO PCs? {conclusion_line(float(amv_raw['R2']) < float(amv_only['R2']) or float(amv_raw['RMSE']) > float(amv_only['RMSE']), 'Yes, the raw regional SST is weaker than the AMV/AMO PCs.', 'No, the raw regional SST is not weaker than the AMV/AMO PCs.')}",
        f"- Does full passive SST fail worse than region-restricted SST? {conclusion_line(float(full_raw['R2']) < float(amv_raw['R2']) or float(full_raw['RMSE']) > float(amv_raw['RMSE']), 'Yes, the full passive field is worse than the AMV/AMO-region raw SST.', 'No, the full passive field is not worse than the AMV/AMO-region raw SST.')}",
        "",
        "## Interpretation",
        "",
    ]

    if float(amv_only["R2"]) > 0.0 and float(amv_plus_z["R2"]) > float(amv_only["R2"]):
        implementation_text = "The basic PLS implementation appears to work on low-dimensional curated predictors."
    else:
        implementation_text = "The low-dimensional curated-predictor cases do not clearly rescue PLS, so implementation risk remains non-trivial."
    if float(full_raw["R2"]) < float(amv_only["R2"]) and float(full_raw["R2"]) <= float(amv_raw["R2"]):
        domain_text = "The strongest failure signature is consistent with full-domain noise and excessive raw-grid dimensionality."
    else:
        domain_text = "The main limitation is not uniquely tied to the full-domain raw-grid case."
    if float(amv_only["R2"]) > float(amv_raw["R2"]):
        pca_text = "The AMV/AMO PC representation is more useful than feeding the same regional field in raw-grid form, which supports the need for regional PCA or similarly structured compression."
    else:
        pca_text = "Regional PCA is not clearly helping relative to the raw regional field in this diagnostic."

    lines.extend(
        [
            f"- {implementation_text}",
            f"- {domain_text}",
            f"- {pca_text}",
            "",
            "## PLS summary metrics",
            "",
            dataframe_to_markdown(summary_df),
            "",
            "## Train-mean baseline metrics",
            "",
            dataframe_to_markdown(baseline_df),
            "",
            "## Artifact paths",
            "",
        ]
    )
    for experiment_name, paths in output_paths.items():
        lines.append(f"- `{experiment_name}` predictions: `{paths['predictions']}`")
        lines.append(f"- `{experiment_name}` plot: `{paths['plot']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()

    predictor_table = load_scalar_target_table()
    y_all = predictor_table["obs_swe"].to_numpy(dtype=np.float64)
    raw_cube, lat, lon = load_global_sst_cube()
    amv_valid_mask = build_region_valid_mask(raw_cube, lat, lon, "amv_amo_region")
    full_valid_mask = build_region_valid_mask(raw_cube, lat, lon, "full_passive")

    summary_rows: List[Dict[str, Union[float, str]]] = []
    baseline_rows: List[Dict[str, Union[float, str]]] = []
    output_paths: Dict[str, Dict[str, str]] = {}

    for spec in EXPERIMENTS:
        if spec.kind == "table":
            pred_df, metrics, baseline_metrics = run_table_experiment(spec, predictor_table)
        elif spec.kind == "raw_sst":
            valid_mask = amv_valid_mask if spec.region_label == "amv_amo_region" else full_valid_mask
            pred_df, metrics, baseline_metrics = run_raw_sst_experiment(spec, y_all, raw_cube, lat, valid_mask)
        else:
            raise ValueError(f"Unknown experiment kind {spec.kind}")

        pred_path = OUTPUT_DIR / f"predictions_{spec.name}.csv"
        plot_path = OUTPUT_DIR / f"obs_vs_pred_{spec.name}.png"
        pred_df.to_csv(pred_path, index=False)
        make_obs_vs_pred_plot(pred_df, spec.description, plot_path, metrics)

        summary_rows.append(
            {
                "experiment": spec.name,
                "n_features": int(metrics["n_features"]),
                "mean_selected_components": float(metrics["mean_selected_components"]),
                "R2": float(metrics["R2"]),
                "RMSE": float(metrics["RMSE"]),
                "MAE": float(metrics["MAE"]),
                "Pearson_r": float(metrics["Pearson_r"]),
                "sign_accuracy": float(metrics["sign_accuracy"]),
            }
        )
        baseline_rows.append(
            {
                "experiment": spec.name,
                "R2": float(baseline_metrics["R2"]),
                "RMSE": float(baseline_metrics["RMSE"]),
                "MAE": float(baseline_metrics["MAE"]),
                "Pearson_r": float(baseline_metrics["Pearson_r"]),
                "sign_accuracy": float(baseline_metrics["sign_accuracy"]),
            }
        )
        output_paths[spec.name] = {"predictions": str(pred_path), "plot": str(plot_path)}

    summary_df = pd.DataFrame(summary_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    baseline_df.to_csv(BASELINE_SUMMARY_CSV, index=False)
    README_MD.write_text(build_readme(summary_df, baseline_df, output_paths), encoding="utf-8")

    payload = {
        "summary_csv": str(SUMMARY_CSV),
        "baseline_summary_csv": str(BASELINE_SUMMARY_CSV),
        "readme": str(README_MD),
        "outputs": output_paths,
        "sst_source": str(SST_FILE),
        "amv_domain": {
            "lat_min": AMV_LAT_MIN,
            "lat_max": AMV_LAT_MAX,
            "lon_min_360": AMV_LON_MIN_360,
            "lon_max_360_exclusive": AMV_LON_MAX_360,
        },
        "global_grid": {
            "n_lat": int(lat.size),
            "n_lon": int(lon.size),
            "n_features_amv_region": int(np.sum(amv_valid_mask)),
            "n_features_full_passive": int(np.sum(full_valid_mask)),
        },
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"summary_csv={SUMMARY_CSV}", flush=True)
    print(f"baseline_summary_csv={BASELINE_SUMMARY_CSV}", flush=True)
    print(f"readme={README_MD}", flush=True)
    for row in summary_rows:
        print(
            f"{row['experiment']}: n_features={row['n_features']} mean_selected_components={row['mean_selected_components']:.3f} "
            f"R2={row['R2']:.6f} RMSE={row['RMSE']:.6f} MAE={row['MAE']:.6f} "
            f"Pearson_r={row['Pearson_r']:.6f} sign_accuracy={row['sign_accuracy']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
