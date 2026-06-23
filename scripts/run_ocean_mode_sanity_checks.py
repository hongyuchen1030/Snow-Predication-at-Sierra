#!/usr/bin/env python3
"""
Run train-all-years ocean-mode sanity checks for Sierra April 1 SWE.
"""

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
from sklearn.cross_decomposition import PLSRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "ocean_mode_sanity_checks"

SOURCE_RIDGE_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "ocean_mode_only_ridge"
SOURCE_PLS_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "ocean_mode_only_pls"

PREDICTOR_TABLE_CSV = SOURCE_RIDGE_DIR / "ocean_mode_predictors_wy1985_2021.csv"
TARGET_TABLE_CSV = SOURCE_RIDGE_DIR / "sierra_apr1_swe_north_central_south_wy1985_2021.csv"

IN_SAMPLE_PREDICTIONS_CSV = OUTPUT_DIR / "in_sample_predictions.csv"
IN_SAMPLE_METRICS_JSON = OUTPUT_DIR / "in_sample_metrics.json"
SHUFFLED_NULL_METRICS_CSV = OUTPUT_DIR / "shuffled_null_metrics.csv"
TIMESERIES_PNG = OUTPUT_DIR / "observed_vs_fitted_timeseries.png"

RIDGE_LAMBDAS = np.asarray([0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0], dtype=np.float64)
PLS_COMPONENT_GRID = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
RNG_SEED = 20260618
N_PERMUTATIONS = 1000
REGION_KEYS = ("North", "Central", "South")
WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)


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


def load_csv_matrix(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or fieldnames[0] != "water_year":
            raise ValueError(f"Unexpected header in {path}")
        columns = fieldnames[1:]
        years: List[int] = []
        rows: List[List[float]] = []
        for row in reader:
            years.append(int(row["water_year"]))
            rows.append([float(row[column]) for column in columns])
    if years != WATER_YEARS.tolist():
        raise ValueError(f"Water years in {path} do not match WY1985--WY2021")
    return columns, np.asarray(rows, dtype=np.float64)


def verify_matching_source_tables() -> None:
    for filename in (
        "ocean_mode_predictors_wy1985_2021.csv",
        "sierra_apr1_swe_north_central_south_wy1985_2021.csv",
    ):
        ridge_text = (SOURCE_RIDGE_DIR / filename).read_text(encoding="utf-8")
        pls_text = (SOURCE_PLS_DIR / filename).read_text(encoding="utf-8")
        if ridge_text != pls_text:
            raise ValueError(f"Ridge and PLS source tables differ for {filename}")


def load_predictors() -> Tuple[List[str], np.ndarray]:
    return load_csv_matrix(PREDICTOR_TABLE_CSV)


def load_targets() -> List[RegionTargetSeries]:
    columns, matrix = load_csv_matrix(TARGET_TABLE_CSV)
    expected = [f"SWE_{region_key}" for region_key in REGION_KEYS]
    if columns != expected:
        raise ValueError(f"Unexpected target columns in {TARGET_TABLE_CSV}: {columns}")
    return [
        RegionTargetSeries(key=region_key, observed_m=matrix[:, idx].astype(np.float64))
        for idx, region_key in enumerate(REGION_KEYS)
    ]


def standardize_all_years_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std, mean, std


def standardize_all_years_target(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1))
    if std == 0.0:
        std = 1.0
    return (y - mean) / std, mean, std


def fit_ridge_coefficients(x_std: np.ndarray, y_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_std.T @ x_std
    rhs = x_std.T @ y_std
    if alpha == 0.0:
        beta, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)
        return beta
    ridge = gram + alpha * np.eye(gram.shape[0], dtype=np.float64)
    return np.linalg.solve(ridge, rhs)


def fit_pls_model(x_std: np.ndarray, y_std: np.ndarray, n_components: int) -> PLSRegression:
    model = PLSRegression(n_components=int(n_components), scale=False)
    model.fit(x_std, y_std[:, None])
    return model


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


def compute_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": r2_score_manual(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "Pearson_r": pearson_r(y_true, y_pred),
    }


def run_real_data_sweeps(
    feature_names: Sequence[str],
    x: np.ndarray,
    targets: Sequence[RegionTargetSeries],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, np.ndarray]]:
    x_std, x_mean, x_stddev = standardize_all_years_features(x)
    prediction_rows: List[Dict[str, Any]] = []
    metrics_payload: Dict[str, Any] = {
        "ridge": {},
        "pls": {},
        "best_config_by_region": {},
        "feature_standardization": {
            "n_features": int(len(feature_names)),
            "feature_names": list(feature_names),
            "feature_mean": x_mean.tolist(),
            "feature_std_ddof1": x_stddev.tolist(),
        },
    }
    best_series: Dict[str, np.ndarray] = {}

    for target in targets:
        y = target.observed_m
        y_std, y_mean, y_stddev = standardize_all_years_target(y)
        region_best: Dict[str, Any] = {"ridge": None, "pls": None}
        region_best_r2 = {"ridge": -np.inf, "pls": -np.inf}
        region_ridge_metrics: List[Dict[str, Any]] = []
        region_pls_metrics: List[Dict[str, Any]] = []

        for alpha in RIDGE_LAMBDAS:
            beta = fit_ridge_coefficients(x_std, y_std, float(alpha))
            fitted_std = x_std @ beta
            fitted = fitted_std * y_stddev + y_mean
            metric_bundle = compute_metric_bundle(y, fitted)
            region_ridge_metrics.append(
                {
                    "lambda": float(alpha),
                    **metric_bundle,
                }
            )
            for year_idx, water_year in enumerate(WATER_YEARS):
                prediction_rows.append(
                    {
                        "water_year": int(water_year),
                        "region": target.key,
                        "model_family": "ridge",
                        "config_name": "lambda",
                        "config_value": float(alpha),
                        "observed": float(y[year_idx]),
                        "fitted": float(fitted[year_idx]),
                        "residual": float(y[year_idx] - fitted[year_idx]),
                    }
                )
            if metric_bundle["R2"] > region_best_r2["ridge"]:
                region_best_r2["ridge"] = metric_bundle["R2"]
                region_best["ridge"] = {
                    "config_name": "lambda",
                    "config_value": float(alpha),
                    **metric_bundle,
                }
                best_series[f"{target.key}_ridge"] = fitted.copy()
            print(
                f"real-data ridge region={target.key} lambda={alpha:g} "
                f"R2={metric_bundle['R2']:.6f} RMSE={metric_bundle['RMSE']:.6f}",
                flush=True,
            )

        for n_components in PLS_COMPONENT_GRID:
            model = fit_pls_model(x_std, y_std, int(n_components))
            fitted_std = model.predict(x_std).ravel()
            fitted = fitted_std * y_stddev + y_mean
            metric_bundle = compute_metric_bundle(y, fitted)
            region_pls_metrics.append(
                {
                    "n_components": int(n_components),
                    **metric_bundle,
                }
            )
            for year_idx, water_year in enumerate(WATER_YEARS):
                prediction_rows.append(
                    {
                        "water_year": int(water_year),
                        "region": target.key,
                        "model_family": "pls",
                        "config_name": "n_components",
                        "config_value": int(n_components),
                        "observed": float(y[year_idx]),
                        "fitted": float(fitted[year_idx]),
                        "residual": float(y[year_idx] - fitted[year_idx]),
                    }
                )
            if metric_bundle["R2"] > region_best_r2["pls"]:
                region_best_r2["pls"] = metric_bundle["R2"]
                region_best["pls"] = {
                    "config_name": "n_components",
                    "config_value": int(n_components),
                    **metric_bundle,
                }
                best_series[f"{target.key}_pls"] = fitted.copy()
            print(
                f"real-data pls region={target.key} K={int(n_components)} "
                f"R2={metric_bundle['R2']:.6f} RMSE={metric_bundle['RMSE']:.6f}",
                flush=True,
            )

        metrics_payload["ridge"][target.key] = {
            "target_mean": y_mean,
            "target_std_ddof1": y_stddev,
            "configs": region_ridge_metrics,
        }
        metrics_payload["pls"][target.key] = {
            "target_mean": y_mean,
            "target_std_ddof1": y_stddev,
            "configs": region_pls_metrics,
        }
        metrics_payload["best_config_by_region"][target.key] = region_best

    return prediction_rows, metrics_payload, best_series


def run_shuffled_null_sweeps(
    x: np.ndarray,
    targets: Sequence[RegionTargetSeries],
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    x_std, _, _ = standardize_all_years_features(x)
    rows: List[Dict[str, Any]] = []

    for target in targets:
        y = target.observed_m
        for permutation_idx in range(N_PERMUTATIONS):
            permuted = y[rng.permutation(y.size)]
            y_perm_std, y_perm_mean, y_perm_stddev = standardize_all_years_target(permuted)

            for alpha in RIDGE_LAMBDAS:
                beta = fit_ridge_coefficients(x_std, y_perm_std, float(alpha))
                fitted = (x_std @ beta) * y_perm_stddev + y_perm_mean
                metric_bundle = compute_metric_bundle(permuted, fitted)
                rows.append(
                    {
                        "region": target.key,
                        "model_family": "ridge",
                        "config_name": "lambda",
                        "config_value": float(alpha),
                        "permutation": permutation_idx + 1,
                        **metric_bundle,
                    }
                )

            for n_components in PLS_COMPONENT_GRID:
                model = fit_pls_model(x_std, y_perm_std, int(n_components))
                fitted = model.predict(x_std).ravel() * y_perm_stddev + y_perm_mean
                metric_bundle = compute_metric_bundle(permuted, fitted)
                rows.append(
                    {
                        "region": target.key,
                        "model_family": "pls",
                        "config_name": "n_components",
                        "config_value": int(n_components),
                        "permutation": permutation_idx + 1,
                        **metric_bundle,
                    }
                )

            if (permutation_idx + 1) % 100 == 0:
                print(
                    f"completed shuffled nulls region={target.key} "
                    f"permutations={permutation_idx + 1}/{N_PERMUTATIONS}",
                    flush=True,
                )

    return rows


def summarize_null_rows(null_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    grouped: Dict[Tuple[str, str, str, float], List[float]] = {}
    for row in null_rows:
        key = (
            str(row["region"]),
            str(row["model_family"]),
            str(row["config_name"]),
            float(row["config_value"]),
        )
        grouped.setdefault(key, []).append(float(row["R2"]))
    for (region, model_family, config_name, config_value), values in sorted(grouped.items()):
        summary.setdefault(region, {}).setdefault(model_family, []).append(
            {
                config_name: config_value,
                "mean_R2": float(np.mean(values)),
                "std_R2": float(np.std(values, ddof=1)),
                "p05_R2": float(np.quantile(values, 0.05)),
                "p50_R2": float(np.quantile(values, 0.50)),
                "p95_R2": float(np.quantile(values, 0.95)),
            }
        )
    return summary


def write_predictions_csv(rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "water_year",
        "region",
        "model_family",
        "config_name",
        "config_value",
        "observed",
        "fitted",
        "residual",
    ]
    with IN_SAMPLE_PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "water_year": int(row["water_year"]),
                    "region": row["region"],
                    "model_family": row["model_family"],
                    "config_name": row["config_name"],
                    "config_value": f"{float(row['config_value']):.12g}",
                    "observed": f"{float(row['observed']):.12g}",
                    "fitted": f"{float(row['fitted']):.12g}",
                    "residual": f"{float(row['residual']):.12g}",
                }
            )


def write_null_metrics_csv(rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "region",
        "model_family",
        "config_name",
        "config_value",
        "permutation",
        "R2",
        "RMSE",
        "MAE",
        "Pearson_r",
    ]
    with SHUFFLED_NULL_METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "region": row["region"],
                    "model_family": row["model_family"],
                    "config_name": row["config_name"],
                    "config_value": f"{float(row['config_value']):.12g}",
                    "permutation": int(row["permutation"]),
                    "R2": f"{float(row['R2']):.12g}",
                    "RMSE": f"{float(row['RMSE']):.12g}",
                    "MAE": f"{float(row['MAE']):.12g}",
                    "Pearson_r": f"{float(row['Pearson_r']):.12g}",
                }
            )


def write_metrics_json(
    feature_names: Sequence[str],
    real_metrics: Dict[str, Any],
    null_rows: Sequence[Dict[str, Any]],
    runtime_seconds: float,
) -> None:
    payload = {
        "experiment": "ocean_mode_sanity_checks",
        "script_path": str(Path(__file__).resolve()),
        "output_folder": str(OUTPUT_DIR),
        "water_year_start": int(WATER_YEAR_START),
        "water_year_end": int(WATER_YEAR_END),
        "n_years": int(WATER_YEARS.size),
        "n_permutations": int(N_PERMUTATIONS),
        "rng_seed": int(RNG_SEED),
        "ridge_lambda_grid": RIDGE_LAMBDAS.tolist(),
        "pls_component_grid": PLS_COMPONENT_GRID.tolist(),
        "predictor_count": int(len(feature_names)),
        "source_tables": {
            "predictor_table_path": str(PREDICTOR_TABLE_CSV),
            "target_table_path": str(TARGET_TABLE_CSV),
            "matching_pls_predictor_table_path": str(SOURCE_PLS_DIR / PREDICTOR_TABLE_CSV.name),
            "matching_pls_target_table_path": str(SOURCE_PLS_DIR / TARGET_TABLE_CSV.name),
        },
        "real_data": real_metrics,
        "shuffled_null_summary": summarize_null_rows(null_rows),
        "artifacts": {
            "in_sample_predictions_csv": str(IN_SAMPLE_PREDICTIONS_CSV),
            "in_sample_metrics_json": str(IN_SAMPLE_METRICS_JSON),
            "shuffled_null_metrics_csv": str(SHUFFLED_NULL_METRICS_CSV),
            "timeseries_png": str(TIMESERIES_PNG),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peak_memory_mb": peak_memory_mb(),
    }
    IN_SAMPLE_METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_timeseries(
    targets: Sequence[RegionTargetSeries],
    best_series: Dict[str, np.ndarray],
    real_metrics: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = {"observed": "#1f4e79", "ridge": "#c76d06", "pls": "#2a7f62"}
    for ax, target in zip(axes, targets):
        ridge_best = real_metrics["best_config_by_region"][target.key]["ridge"]
        pls_best = real_metrics["best_config_by_region"][target.key]["pls"]
        ax.plot(WATER_YEARS, target.observed_m, color=colors["observed"], marker="o", linewidth=1.8, label="Observed")
        ax.plot(
            WATER_YEARS,
            best_series[f"{target.key}_ridge"],
            color=colors["ridge"],
            marker="s",
            linewidth=1.6,
            label=f"Best ridge (lambda={ridge_best['config_value']:.4g})",
        )
        ax.plot(
            WATER_YEARS,
            best_series[f"{target.key}_pls"],
            color=colors["pls"],
            marker="^",
            linewidth=1.6,
            label=f"Best PLS (K={int(pls_best['config_value'])})",
        )
        ax.set_ylabel("April 1 SWE (m)")
        ax.set_title(
            f"{target.key} Sierra | "
            f"ridge R2={ridge_best['R2']:.3f}, "
            f"PLS R2={pls_best['R2']:.3f}"
        )
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("Water year")
    fig.suptitle("Ocean-mode sanity checks: observed vs fitted April 1 Sierra SWE", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(TIMESERIES_PNG, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    ensure_output_dir()
    verify_matching_source_tables()
    start = perf_counter()

    feature_names, x = load_predictors()
    targets = load_targets()

    real_prediction_rows, real_metrics, best_series = run_real_data_sweeps(feature_names, x, targets)
    null_rows = run_shuffled_null_sweeps(x, targets)

    write_predictions_csv(real_prediction_rows)
    write_null_metrics_csv(null_rows)
    plot_timeseries(targets, best_series, real_metrics)

    runtime_seconds = perf_counter() - start
    write_metrics_json(feature_names, real_metrics, null_rows, runtime_seconds)

    print(f"Wrote predictions: {IN_SAMPLE_PREDICTIONS_CSV}", flush=True)
    print(f"Wrote metrics: {IN_SAMPLE_METRICS_JSON}", flush=True)
    print(f"Wrote null metrics: {SHUFFLED_NULL_METRICS_CSV}", flush=True)
    print(f"Wrote time-series figure: {TIMESERIES_PNG}", flush=True)


if __name__ == "__main__":
    main()
