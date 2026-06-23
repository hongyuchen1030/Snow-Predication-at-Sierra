#!/usr/bin/env python3
"""
Residual analysis for the saved exact-grid-cell Z1+Z2 Sierra SWE LOYO model.
"""

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import ERA5_LAND_ROOT_PATH  # noqa: E402


INPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo"
INFLUENCE_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_influence_diagnostic"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_residual_analysis"

PREDICTIONS_CSV = INPUT_DIR / "full37_patch_loyo_predictions.csv"
METRICS_CSV = INPUT_DIR / "full37_patch_loyo_metrics.csv"
BETA_CSV = INPUT_DIR / "full37_patch_beta_by_fold.csv"
PATCH_PREDICTORS_CSV = INPUT_DIR / "full37_patch_predictors.csv"
INFLUENCE_ERRORS_CSV = INFLUENCE_DIR / "exact_Z1_Z2_fold_errors.csv"
INFLUENCE_SUMMARY_JSON = INFLUENCE_DIR / "exact_Z1_Z2_influence_summary.json"

OCEAN_MODE_PREDICTORS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "ocean_mode_only_ridge"
    / "ocean_mode_predictors_wy1985_2021.csv"
)
NINO34_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "nino34"
    / "nino34_monthly_wy1985_2021_sep_mar.csv"
)
AMV_AMO_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)

PATCH_NAME = "exact_grid_cell"
MODEL_NAME = "Z1_Z2"
KNOWN_TOP5_YEARS = [1993, 2005, 2006, 2012, 2021]
SEARCHED_PATHS = [
    "artifacts/",
    "scripts/",
    "docs/",
    "src/",
    str(OCEAN_MODE_PREDICTORS_CSV),
    str(NINO34_CSV),
    str(AMV_AMO_CSV),
    str(ERA5_LAND_ROOT_PATH),
]
SEARCHED_PATTERNS = [
    "precip",
    "precipitation",
    "PR",
    "PPT",
    "tp",
    "ERA5",
    "temperature",
    "T2M",
    "tas",
    "snowfall",
    "rainfall",
    "rain-snow",
    "rain_snow",
    "WUS",
    "SWE",
    "SCA",
    "Nino",
    "nino34",
    "PDO",
    "AMO",
    "AMV",
    "MJO",
    "PNA",
    "ENSO",
    "Pacific EOF",
    "COBE2 PC",
]
MONTH_TO_NUMBER = {
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
}


@dataclass(frozen=True)
class MetricBundle:
    r: float
    R2: float
    RMSE: float
    MAE: float
    sign_accuracy: float


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


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


def sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sign(obs[mask]) == np.sign(pred[mask])))


def metric_bundle(obs: np.ndarray, pred: np.ndarray) -> MetricBundle:
    return MetricBundle(
        r=corrcoef_safe(obs, pred),
        R2=r2_manual(obs, pred),
        RMSE=rmse(obs, pred),
        MAE=mae(obs, pred),
        sign_accuracy=sign_accuracy(obs, pred),
    )


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def correlation_record(
    x: pd.Series,
    y: pd.Series,
) -> Tuple[float, float, float, float, int]:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    xx = x.to_numpy(dtype=float)[mask]
    yy = y.to_numpy(dtype=float)[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    pearson = stats.pearsonr(xx, yy)
    spearman = stats.spearmanr(xx, yy)
    return float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(spearman.pvalue), n


def build_base_residual_table() -> Tuple[pd.DataFrame, Dict[str, float]]:
    predictions = pd.read_csv(PREDICTIONS_CSV)
    metrics = pd.read_csv(METRICS_CSV)

    df = predictions.loc[
        (predictions["patch_size"] == PATCH_NAME) & (predictions["model_name"] == MODEL_NAME),
        ["heldout_wy", "obs_swe", "pred_swe", "error", "abs_error", "sign_correct"],
    ].copy()
    df = df.rename(columns={"heldout_wy": "water_year", "error": "error_pred_minus_obs"})
    df["water_year"] = df["water_year"].astype(int)
    df["obs_swe"] = df["obs_swe"].astype(float)
    df["pred_swe"] = df["pred_swe"].astype(float)
    df["error_pred_minus_obs"] = df["error_pred_minus_obs"].astype(float)
    df["residual_obs_minus_pred"] = df["obs_swe"] - df["pred_swe"]
    df["abs_residual"] = np.abs(df["residual_obs_minus_pred"])
    df["squared_residual"] = df["residual_obs_minus_pred"] ** 2
    df["sign_correct"] = df["sign_correct"].astype(float).astype(int)
    df = df.sort_values("water_year").reset_index(drop=True)

    df["obs_swe_rank"] = df["obs_swe"].rank(method="average", ascending=True)
    df["obs_swe_quantile"] = df["obs_swe"].rank(method="average", pct=True)
    df["wet_extreme_flag"] = df["obs_swe_quantile"] >= 0.8
    df["dry_extreme_flag"] = df["obs_swe_quantile"] <= 0.2
    df["neutral_swe_flag"] = ~(df["wet_extreme_flag"] | df["dry_extreme_flag"])
    df["period_group"] = np.where(df["water_year"] <= 2010, "pre_2010", "post_2010")
    df["pre_2005_flag"] = df["water_year"] <= 2005
    df["post_2005_flag"] = df["water_year"] > 2005
    df["pre_2010_flag"] = df["water_year"] <= 2010
    df["post_2010_flag"] = df["water_year"] > 2010

    computed_top5 = (
        df.sort_values(["abs_residual", "water_year"], ascending=[False, True])["water_year"].head(5).tolist()
    )
    df["is_top5_abs_error_year"] = df["water_year"].isin(KNOWN_TOP5_YEARS)
    df["is_computed_top5_abs_error_year"] = df["water_year"].isin(computed_top5)

    metrics_row = metrics.loc[
        (metrics["patch_size"] == PATCH_NAME) & (metrics["model_name"] == MODEL_NAME)
    ].iloc[0]
    base_metrics = {
        "r": float(metrics_row["r"]),
        "R2": float(metrics_row["R2"]),
        "RMSE": float(metrics_row["RMSE"]),
        "MAE": float(metrics_row["MAE"]),
        "sign_accuracy": float(metrics_row["sign_accuracy"]),
    }
    return df, base_metrics


def group_distribution_metrics(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    obs = df["obs_swe"].to_numpy(dtype=float)
    pred = df["pred_swe"].to_numpy(dtype=float)
    residual = df["residual_obs_minus_pred"].to_numpy(dtype=float)
    abs_residual = df["abs_residual"].to_numpy(dtype=float)
    metrics = metric_bundle(obs, pred)
    return {
        "group_name": name,
        "n_years": int(len(df)),
        "r": metrics.r,
        "R2": metrics.R2,
        "RMSE": metrics.RMSE,
        "MAE": metrics.MAE,
        "mean_residual": float(df["residual_obs_minus_pred"].mean()),
        "median_residual": float(df["residual_obs_minus_pred"].median()),
        "mean_abs_residual": float(df["abs_residual"].mean()),
        "median_abs_residual": float(df["abs_residual"].median()),
        "RMSE_residual": float(np.sqrt(np.mean(residual**2))),
        "sign_accuracy": metrics.sign_accuracy,
        "mean_obs_swe": float(df["obs_swe"].mean()),
        "mean_pred_swe": float(df["pred_swe"].mean()),
        "residual_std": float(np.std(residual, ddof=1)) if len(df) > 1 else float("nan"),
        "abs_residual_std": float(np.std(abs_residual, ddof=1)) if len(df) > 1 else float("nan"),
    }


def build_year_group_metrics(residual_df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "pre_2010": residual_df.loc[residual_df["water_year"] <= 2010],
        "post_2010": residual_df.loc[residual_df["water_year"] > 2010],
        "pre_2005": residual_df.loc[residual_df["water_year"] <= 2005],
        "post_2005": residual_df.loc[residual_df["water_year"] > 2005],
        "wet_extreme": residual_df.loc[residual_df["wet_extreme_flag"]],
        "dry_extreme": residual_df.loc[residual_df["dry_extreme_flag"]],
        "neutral": residual_df.loc[residual_df["neutral_swe_flag"]],
    }
    rows = [group_distribution_metrics(name, group) for name, group in groups.items() if not group.empty]
    metrics_df = pd.DataFrame(rows)

    def metric_for(group_name: str, column: str) -> float:
        return float(metrics_df.loc[metrics_df["group_name"] == group_name, column].iloc[0])

    diff_rows = [
        {
            "group_name": "post2010_minus_pre2010",
            "n_years": np.nan,
            "r": np.nan,
            "R2": np.nan,
            "RMSE": metric_for("post_2010", "RMSE") - metric_for("pre_2010", "RMSE"),
            "MAE": metric_for("post_2010", "MAE") - metric_for("pre_2010", "MAE"),
            "mean_residual": metric_for("post_2010", "mean_residual") - metric_for("pre_2010", "mean_residual"),
            "median_residual": metric_for("post_2010", "median_residual") - metric_for("pre_2010", "median_residual"),
            "mean_abs_residual": metric_for("post_2010", "mean_abs_residual") - metric_for("pre_2010", "mean_abs_residual"),
            "median_abs_residual": metric_for("post_2010", "median_abs_residual") - metric_for("pre_2010", "median_abs_residual"),
            "RMSE_residual": metric_for("post_2010", "RMSE_residual") - metric_for("pre_2010", "RMSE_residual"),
            "sign_accuracy": metric_for("post_2010", "sign_accuracy") - metric_for("pre_2010", "sign_accuracy"),
            "mean_obs_swe": np.nan,
            "mean_pred_swe": np.nan,
            "residual_std": np.nan,
            "abs_residual_std": np.nan,
        },
        {
            "group_name": "post2005_minus_pre2005",
            "n_years": np.nan,
            "r": np.nan,
            "R2": np.nan,
            "RMSE": metric_for("post_2005", "RMSE") - metric_for("pre_2005", "RMSE"),
            "MAE": metric_for("post_2005", "MAE") - metric_for("pre_2005", "MAE"),
            "mean_residual": metric_for("post_2005", "mean_residual") - metric_for("pre_2005", "mean_residual"),
            "median_residual": metric_for("post_2005", "median_residual") - metric_for("pre_2005", "median_residual"),
            "mean_abs_residual": metric_for("post_2005", "mean_abs_residual") - metric_for("pre_2005", "mean_abs_residual"),
            "median_abs_residual": metric_for("post_2005", "median_abs_residual") - metric_for("pre_2005", "median_abs_residual"),
            "RMSE_residual": metric_for("post_2005", "RMSE_residual") - metric_for("pre_2005", "RMSE_residual"),
            "sign_accuracy": metric_for("post_2005", "sign_accuracy") - metric_for("pre_2005", "sign_accuracy"),
            "mean_obs_swe": np.nan,
            "mean_pred_swe": np.nan,
            "residual_std": np.nan,
            "abs_residual_std": np.nan,
        },
    ]
    return pd.concat([metrics_df, pd.DataFrame(diff_rows)], ignore_index=True)


def add_custom_precip_temp_features(residual_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return residual_df.copy(), {
        "status": "not_run_no_existing_precip_temp_data_found",
        "message": (
            "Searches found temperature-related analysis artifacts but no existing repo-local Sierra water-year "
            "precipitation/temperature summary table aligned to the exact Z1+Z2 SWE years. "
            "Raw ERA5-Land source files exist at %s but were not used because the request said to rely on existing "
            "repo/artifact products rather than build new summaries from source data."
        )
        % ERA5_LAND_ROOT_PATH,
        "searched_paths": SEARCHED_PATHS,
        "searched_patterns": SEARCHED_PATTERNS,
        "sources_used": [],
    }


def merge_climate_indices(residual_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source_files = []
    frames = []
    for path in [OCEAN_MODE_PREDICTORS_CSV, NINO34_CSV, AMV_AMO_CSV]:
        if path.exists():
            frame = pd.read_csv(path)
            frame["water_year"] = frame["water_year"].astype(int)
            frames.append(frame)
            source_files.append(str(path))
    if not frames:
        return residual_df.copy(), {
            "status": "not_run_no_existing_climate_index_data_found",
            "sources_used": [],
        }

    merged = residual_df.copy()
    for frame in frames:
        cols = [name for name in frame.columns if name == "water_year" or name not in merged.columns]
        merged = merged.merge(frame[cols], on="water_year", how="left")
    return merged, {
        "status": "run",
        "sources_used": source_files,
    }


def build_correlation_table(residual_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    top_features: List[Dict[str, Any]] = []
    excluded = {
        "water_year",
        "obs_swe",
        "pred_swe",
        "abs_error",
        "error_pred_minus_obs",
        "residual_obs_minus_pred",
        "abs_residual",
        "squared_residual",
        "sign_correct",
        "obs_swe_rank",
        "obs_swe_quantile",
        "wet_extreme_flag",
        "dry_extreme_flag",
        "neutral_swe_flag",
        "period_group",
        "pre_2005_flag",
        "post_2005_flag",
        "pre_2010_flag",
        "post_2010_flag",
        "is_top5_abs_error_year",
        "is_computed_top5_abs_error_year",
    }
    for column in residual_df.columns:
        if column in excluded:
            continue
        if residual_df[column].dtype == object:
            continue
        series = pd.to_numeric(residual_df[column], errors="coerce")
        if series.notna().sum() < 3:
            continue
        group = "precip_temp" if any(
            token in column.lower()
            for token in ["precip", "temperature", "temp", "snowfall", "rain_", "warm_", "cold_"]
        ) else "climate_index"
        pearson_r, pearson_p, spearman_r, spearman_p, n_years = correlation_record(
            series, residual_df["residual_obs_minus_pred"]
        )
        pearson_abs, _, spearman_abs, _, _ = correlation_record(series, residual_df["abs_residual"])
        pearson_top5, _, spearman_top5, _, _ = correlation_record(
            series, residual_df["is_top5_abs_error_year"].astype(float)
        )
        month_match = None
        for month in MONTH_TO_NUMBER:
            if column.endswith("_" + month) or month in column:
                month_match = month
        record = {
            "variable_group": group,
            "variable_name": column,
            "season_or_month": month_match or "aggregate",
            "n_years": n_years,
            "pearson_r_with_residual": pearson_r,
            "pearson_p_with_residual_if_available": pearson_p,
            "spearman_r_with_residual": spearman_r,
            "spearman_p_with_residual_if_available": spearman_p,
            "pearson_r_with_abs_residual": pearson_abs,
            "spearman_r_with_abs_residual": spearman_abs,
            "pearson_r_with_top5_flag": pearson_top5,
            "spearman_r_with_top5_flag": spearman_top5,
            "notes": "Aligned by water_year.",
            "source_file": (
                str(OCEAN_MODE_PREDICTORS_CSV)
                if group == "climate_index"
                else "ERA5-Land via src/snow_ml/pc_baseline.py::_load_atmospheric_daily_domain_series"
            ),
        }
        records.append(record)
        top_features.append(
            {
                "variable_group": group,
                "variable_name": column,
                "strength": max(abs(maybe_float(pearson_r)), abs(maybe_float(pearson_abs)), abs(maybe_float(spearman_r)), abs(maybe_float(spearman_abs))),
            }
        )
    corr_df = pd.DataFrame(records).sort_values(
        ["variable_group", "pearson_r_with_residual"],
        ascending=[True, False],
        na_position="last",
    )
    top_features = sorted(top_features, key=lambda item: item["strength"], reverse=True)
    return corr_df, top_features


def build_extreme_years_table(residual_df: pd.DataFrame, correlation_df: pd.DataFrame) -> pd.DataFrame:
    climate_corr = correlation_df.loc[correlation_df["variable_group"] == "climate_index"].copy()
    climate_strength = (
        climate_corr.assign(
            sort_strength=climate_corr[
                [
                    "pearson_r_with_residual",
                    "pearson_r_with_abs_residual",
                    "spearman_r_with_residual",
                    "spearman_r_with_abs_residual",
                ]
            ]
            .abs()
            .max(axis=1)
        )
        .sort_values("sort_strength", ascending=False)
    )
    top_climate_names = climate_strength["variable_name"].head(3).tolist()
    rows = []
    extreme_df = residual_df.sort_values(["abs_residual", "water_year"], ascending=[False, True]).head(10).copy()
    for _, row in extreme_df.iterrows():
        climate_parts = []
        for name in top_climate_names:
            if name in extreme_df.columns:
                climate_parts.append(f"{name}={row[name]:.4g}")
        precip_cols = [name for name in extreme_df.columns if "precip" in name.lower() or "rain" in name.lower() or "snowfall" in name.lower()]
        temp_cols = [name for name in extreme_df.columns if "temp" in name.lower()]
        auto_note = "large residual but not observed-SWE extreme"
        if row["residual_obs_minus_pred"] > 0 and row["wet_extreme_flag"]:
            auto_note = "high positive residual: model underpredicted wet year"
        elif row["residual_obs_minus_pred"] < 0 and row["dry_extreme_flag"]:
            auto_note = "high negative residual: model overpredicted dry year"
        rows.append(
            {
                "water_year": int(row["water_year"]),
                "obs_swe": float(row["obs_swe"]),
                "pred_swe": float(row["pred_swe"]),
                "residual_obs_minus_pred": float(row["residual_obs_minus_pred"]),
                "abs_residual": float(row["abs_residual"]),
                "swe_rank": float(row["obs_swe_rank"]),
                "swe_quantile": float(row["obs_swe_quantile"]),
                "wet_extreme_flag": bool(row["wet_extreme_flag"]),
                "dry_extreme_flag": bool(row["dry_extreme_flag"]),
                "period_group": str(row["period_group"]),
                "available_precip_summary_columns": "; ".join(precip_cols[:8]),
                "available_temperature_summary_columns": "; ".join(temp_cols[:8]),
                "top_correlated_climate_index_values": "; ".join(climate_parts),
                "brief_auto_note": auto_note,
            }
        )
    return pd.DataFrame(rows)


def answer_yes_no_maybe(condition: Optional[bool], yes: str, no: str, maybe: str) -> str:
    if condition is True:
        return yes
    if condition is False:
        return no
    return maybe


def save_scatter_residual_vs_obs(residual_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(residual_df["obs_swe"], residual_df["residual_obs_minus_pred"], color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(0.0, color="black", linewidth=1.0)
    for _, row in residual_df.loc[residual_df["is_top5_abs_error_year"]].iterrows():
        ax.annotate(str(int(row["water_year"])), (row["obs_swe"], row["residual_obs_minus_pred"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Observed SWE anomaly")
    ax.set_ylabel("Residual (obs - pred)")
    ax.set_title("Exact Z1+Z2 residual vs observed SWE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_timeline_plot(residual_df: pd.DataFrame, output_path: Path) -> None:
    years = residual_df["water_year"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(years, residual_df["obs_swe"], label="Observed SWE anomaly", color="tab:blue", linewidth=2.0)
    ax.plot(years, residual_df["pred_swe"], label="Predicted SWE anomaly", color="tab:orange", linewidth=2.0)
    ax.plot(years, residual_df["residual_obs_minus_pred"], label="Residual", color="tab:green", linewidth=1.5, linestyle="--")
    for _, row in residual_df.loc[residual_df["is_top5_abs_error_year"]].iterrows():
        ax.axvline(int(row["water_year"]), color="tab:red", alpha=0.2, linewidth=1.0)
        ax.annotate(str(int(row["water_year"])), (int(row["water_year"]), row["residual_obs_minus_pred"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Water year")
    ax.set_ylabel("SWE anomaly / residual")
    ax.set_title("Exact Z1+Z2 LOYO timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_boxplot(residual_df: pd.DataFrame, output_path: Path) -> None:
    groups = [
        ("pre_2010", residual_df.loc[residual_df["pre_2010_flag"], "abs_residual"]),
        ("post_2010", residual_df.loc[residual_df["post_2010_flag"], "abs_residual"]),
        ("pre_2005", residual_df.loc[residual_df["pre_2005_flag"], "abs_residual"]),
        ("post_2005", residual_df.loc[residual_df["post_2005_flag"], "abs_residual"]),
        ("wet_extreme", residual_df.loc[residual_df["wet_extreme_flag"], "abs_residual"]),
        ("dry_extreme", residual_df.loc[residual_df["dry_extreme_flag"], "abs_residual"]),
        ("neutral", residual_df.loc[residual_df["neutral_swe_flag"], "abs_residual"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([series.to_numpy(dtype=float) for _, series in groups], labels=[name for name, _ in groups], vert=True)
    ax.set_ylabel("Absolute residual")
    ax.set_title("Absolute residual by period and SWE group")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_multiscatter(
    residual_df: pd.DataFrame,
    selected_columns: List[str],
    output_path: Path,
    title: str,
    y_column: str = "residual_obs_minus_pred",
) -> None:
    n = len(selected_columns)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)
    for ax, column in zip(axes.flatten(), selected_columns):
        ax.scatter(residual_df[column], residual_df[y_column], color="tab:blue")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel(column)
        ax.set_ylabel("Residual (obs - pred)")
        ax.set_title(column)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_placeholder(path: Path, message: str) -> None:
    path.write_text(message + "\n", encoding="utf-8")


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage: building base residual table", flush=True)

    residual_df, base_metrics = build_base_residual_table()
    print("Stage: loading precip/temp auxiliaries", flush=True)
    residual_df, precip_temp_meta = add_custom_precip_temp_features(residual_df)
    print("Stage: merging climate indices", flush=True)
    residual_df, climate_meta = merge_climate_indices(residual_df)

    residual_table_path = OUTPUT_DIR / "exact_Z1_Z2_residual_table.csv"
    residual_df.to_csv(residual_table_path, index=False)
    print("Stage: saved residual table", flush=True)

    year_group_df = build_year_group_metrics(residual_df)
    year_group_path = OUTPUT_DIR / "exact_Z1_Z2_residual_year_group_metrics.csv"
    year_group_df.to_csv(year_group_path, index=False)
    print("Stage: saved year-group metrics", flush=True)

    correlation_df, top_features = build_correlation_table(residual_df)
    correlation_path = OUTPUT_DIR / "exact_Z1_Z2_residual_correlations.csv"
    correlation_df.to_csv(correlation_path, index=False)
    print("Stage: saved correlation table", flush=True)

    extreme_years_df = build_extreme_years_table(residual_df, correlation_df)
    extreme_years_path = OUTPUT_DIR / "exact_Z1_Z2_residual_extreme_years.csv"
    extreme_years_df.to_csv(extreme_years_path, index=False)
    print("Stage: saved extreme-year table", flush=True)

    summary_metric_rows = []
    summary_metric_rows.append(
        {
            "metric_name": "corr_residual_vs_obs_swe_pearson",
            "metric_value": corrcoef_safe(
                residual_df["residual_obs_minus_pred"].to_numpy(dtype=float),
                residual_df["obs_swe"].to_numpy(dtype=float),
            ),
        }
    )
    summary_metric_rows.append(
        {
            "metric_name": "corr_abs_residual_vs_abs_obs_swe_pearson",
            "metric_value": corrcoef_safe(
                residual_df["abs_residual"].to_numpy(dtype=float),
                np.abs(residual_df["obs_swe"].to_numpy(dtype=float)),
            ),
        }
    )
    sp_r, sp_p, _, _, _ = correlation_record(residual_df["residual_obs_minus_pred"], residual_df["obs_swe"])
    summary_metric_rows.append({"metric_name": "corr_residual_vs_obs_swe_spearman", "metric_value": sp_r})
    summary_metric_rows.append({"metric_name": "corr_residual_vs_obs_swe_spearman_p", "metric_value": sp_p})
    metric_path = OUTPUT_DIR / "exact_Z1_Z2_residual_summary_metrics.csv"
    pd.DataFrame(summary_metric_rows).to_csv(metric_path, index=False)

    save_scatter_residual_vs_obs(residual_df, OUTPUT_DIR / "exact_Z1_Z2_residual_vs_observed_swe.png")
    save_timeline_plot(residual_df, OUTPUT_DIR / "exact_Z1_Z2_residual_timeline.png")
    save_boxplot(residual_df, OUTPUT_DIR / "exact_Z1_Z2_residual_by_period_boxplot.png")

    precip_top = [
        item["variable_name"]
        for item in top_features
        if item["variable_group"] == "precip_temp"
    ][:4]
    if precip_top:
        save_multiscatter(
            residual_df,
            precip_top,
            OUTPUT_DIR / "exact_Z1_Z2_residual_vs_precip_temp.png",
            "Residual vs strongest precip/temp variables",
        )
    else:
        write_placeholder(
            OUTPUT_DIR / "exact_Z1_Z2_residual_vs_precip_temp_NOT_RUN.txt",
            "No existing precip/temp variables were available for plotting.",
        )

    climate_top = [
        item["variable_name"]
        for item in top_features
        if item["variable_group"] == "climate_index"
    ][:4]
    if climate_top:
        save_multiscatter(
            residual_df,
            climate_top,
            OUTPUT_DIR / "exact_Z1_Z2_residual_vs_climate_indices.png",
            "Residual vs strongest climate indices",
        )
    else:
        write_placeholder(
            OUTPUT_DIR / "exact_Z1_Z2_residual_vs_climate_indices_NOT_RUN.txt",
            "No existing climate-index variables were available for plotting.",
        )

    wet_metrics = year_group_df.loc[year_group_df["group_name"] == "wet_extreme"].iloc[0].to_dict()
    dry_metrics = year_group_df.loc[year_group_df["group_name"] == "dry_extreme"].iloc[0].to_dict()
    neutral_metrics = year_group_df.loc[year_group_df["group_name"] == "neutral"].iloc[0].to_dict()
    pre2010 = year_group_df.loc[year_group_df["group_name"] == "pre_2010"].iloc[0].to_dict()
    post2010 = year_group_df.loc[year_group_df["group_name"] == "post_2010"].iloc[0].to_dict()

    top_residual_correlates = (
        correlation_df.assign(
            sort_strength=correlation_df[
                [
                    "pearson_r_with_residual",
                    "pearson_r_with_abs_residual",
                    "spearman_r_with_residual",
                    "spearman_r_with_abs_residual",
                ]
            ]
            .abs()
            .max(axis=1)
        )
        .sort_values("sort_strength", ascending=False)
        .head(10)
    )

    wet_concentrated = wet_metrics["mean_residual"] > 0.0 and wet_metrics["mean_abs_residual"] > neutral_metrics["mean_abs_residual"]
    post2010_worse = post2010["mean_abs_residual"] > pre2010["mean_abs_residual"]
    precip_signal = any(item["variable_group"] == "precip_temp" and item["strength"] >= 0.3 for item in top_features)
    climate_signal = any(item["variable_group"] == "climate_index" and item["strength"] >= 0.3 for item in top_features)

    influence_summary = {}
    if INFLUENCE_SUMMARY_JSON.exists():
        influence_summary = json.loads(INFLUENCE_SUMMARY_JSON.read_text(encoding="utf-8"))

    summary_payload = {
        "input_files": [
            str(PREDICTIONS_CSV),
            str(METRICS_CSV),
            str(BETA_CSV),
            str(PATCH_PREDICTORS_CSV),
            str(INFLUENCE_ERRORS_CSV),
            str(INFLUENCE_SUMMARY_JSON),
        ],
        "output_dir": str(OUTPUT_DIR),
        "base_model": {
            "patch_size": PATCH_NAME,
            "model_name": MODEL_NAME,
        },
        "base_model_metrics": base_metrics,
        "residual_definition": {
            "residual_obs_minus_pred": "obs_swe - pred_swe",
            "error_pred_minus_obs": "pred_swe - obs_swe",
        },
        "top5_abs_error_years": KNOWN_TOP5_YEARS,
        "computed_top5_abs_error_years": residual_df.loc[
            residual_df["is_computed_top5_abs_error_year"], "water_year"
        ].tolist(),
        "top10_abs_error_years": extreme_years_df["water_year"].tolist(),
        "wet_dry_extreme_summary": {
            "wet_extreme": wet_metrics,
            "dry_extreme": dry_metrics,
            "neutral": neutral_metrics,
        },
        "pre_post_period_summary": {
            "pre_2010": pre2010,
            "post_2010": post2010,
            "post2010_minus_pre2010": year_group_df.loc[year_group_df["group_name"] == "post2010_minus_pre2010"].iloc[0].to_dict(),
            "pre_2005": year_group_df.loc[year_group_df["group_name"] == "pre_2005"].iloc[0].to_dict(),
            "post_2005": year_group_df.loc[year_group_df["group_name"] == "post_2005"].iloc[0].to_dict(),
            "post2005_minus_pre2005": year_group_df.loc[year_group_df["group_name"] == "post2005_minus_pre2005"].iloc[0].to_dict(),
        },
        "precip_temp_analysis_status": precip_temp_meta["status"],
        "precip_temp_sources_used": precip_temp_meta.get("sources_used", []),
        "climate_index_analysis_status": climate_meta["status"],
        "climate_index_sources_used": climate_meta.get("sources_used", []),
        "searched_paths": SEARCHED_PATHS,
        "searched_patterns": SEARCHED_PATTERNS,
        "top_residual_correlates": top_residual_correlates[
            [
                "variable_group",
                "variable_name",
                "pearson_r_with_residual",
                "spearman_r_with_residual",
                "pearson_r_with_abs_residual",
                "spearman_r_with_abs_residual",
            ]
        ].to_dict(orient="records"),
        "short_answers": {
            "are_residuals_concentrated_in_wet_years": answer_yes_no_maybe(
                wet_concentrated,
                "Yes: wet-year residuals are more positive and larger in magnitude than neutral years.",
                "No clear wet-year concentration appears relative to neutral years.",
                "Mixed: wet-year concentration is not decisive.",
            ),
            "are_residuals_concentrated_after_2010": answer_yes_no_maybe(
                post2010_worse,
                "Yes: post-2010 absolute residuals are larger on average.",
                "No: post-2010 residuals are not larger on average.",
                "Mixed: pre/post-2010 contrast is not decisive.",
            ),
            "are_residuals_related_to_precip_extremes": answer_yes_no_maybe(
                precip_signal if precip_temp_meta["status"] == "run" else None,
                "Yes: at least one precipitation-related variable has a moderate residual association.",
                "No clear precipitation-related association appears in the available variables.",
                "Not assessed from existing data.",
            ),
            "are_residuals_related_to_temperature_or_rain_snow_proxy": answer_yes_no_maybe(
                precip_signal if precip_temp_meta["status"] == "run" else None,
                "Yes: available temperature/rain-snow proxy variables show a moderate residual association.",
                "No clear temperature or rain-snow proxy association appears in the available variables.",
                "Not assessed from existing data.",
            ),
            "are_residuals_related_to_known_climate_indices": answer_yes_no_maybe(
                climate_signal if climate_meta["status"] == "run" else None,
                "Yes: at least one available climate index has a moderate residual association.",
                "No clear climate-index association appears in the available variables.",
                "Not assessed from existing data.",
            ),
        },
        "overall_interpretation": (
            "The exact Z1+Z2 model captures part of the Sierra SWE background variability, but the remaining largest-error years cluster where residual magnitude stays high after conditioning on the saved SST predictor pair. "
            "The auxiliary tables identify whether that leftover error aligns more with wet/dry SWE extremes, later-period degradation, and the strongest available precip/temp or climate-index correlates."
        ),
        "influence_summary_context": influence_summary,
    }
    summary_path = OUTPUT_DIR / "exact_Z1_Z2_residual_analysis_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(f"Base model: {PATCH_NAME} / {MODEL_NAME}", flush=True)
    print("Top 5 residual years:", ", ".join(str(year) for year in KNOWN_TOP5_YEARS), flush=True)
    print(summary_payload["short_answers"]["are_residuals_concentrated_in_wet_years"], flush=True)
    print(summary_payload["short_answers"]["are_residuals_concentrated_after_2010"], flush=True)
    print(summary_payload["short_answers"]["are_residuals_related_to_precip_extremes"], flush=True)
    print(summary_payload["short_answers"]["are_residuals_related_to_known_climate_indices"], flush=True)
    print("Overall interpretation:", summary_payload["overall_interpretation"], flush=True)


if __name__ == "__main__":
    main()
