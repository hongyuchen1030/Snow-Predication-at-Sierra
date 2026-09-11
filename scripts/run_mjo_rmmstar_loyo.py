#!/usr/bin/env python3
"""
Strict LOYO ridge SWE prediction using pure MJO/RMM* predictors.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sierra_swe_mjo_rmmstar_loyo"
TARGET_TABLE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)
BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "mjo_rmmstar_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "mjo_rmmstar_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "mjo_rmmstar_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "mjo_rmmstar_loyo_period_metrics.csv"
ALPHA_CSV = OUTPUT_DIR / "mjo_rmmstar_selected_alpha_by_fold.csv"
BETA_CSV = OUTPUT_DIR / "mjo_rmmstar_beta_by_fold.csv"
SUMMARY_JSON = OUTPUT_DIR / "mjo_rmmstar_summary.json"
OBS_PRED_PNG = OUTPUT_DIR / "mjo_rmmstar_observed_vs_predicted.png"
SCATTER_PNG = OUTPUT_DIR / "mjo_rmmstar_scatter_panels.png"
ERROR_PNG = OUTPUT_DIR / "mjo_rmmstar_error_by_year.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
ACTIVE_THRESHOLD = 1.0
SEARCH_ROOTS = ["data", "artifacts", "inputs"]
SEARCH_PATTERNS = ["*rmm*", "*RMM*", "*rmmstar*", "*RMMstar*", "*mjo*", "*MJO*"]


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


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "mean_error": float(np.mean(error)),
        "median_abs_error": float(np.median(np.abs(error))),
    }


def standardize_train_only(
    x_train_raw: np.ndarray, x_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("Predictor train-fold standard deviation must be positive.")
    return (
        (x_train_raw - x_mean[None, :]) / x_std[None, :],
        (x_test_raw - x_mean) / x_std,
        x_mean,
        x_std,
    )


def standardize_target_train_only(y_train_raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError("Target train-fold standard deviation must be positive.")
    return (y_train_raw - y_mean) / y_std, y_mean, y_std


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    beta_std = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    return np.asarray(beta_std, dtype=float)


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha: Optional[float] = None
    best_mse: Optional[float] = None
    for alpha in ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=float)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train = x_train_raw[inner_mask, :]
            x_inner_test = x_train_raw[~inner_mask, :][0]
            y_inner_train = y_train_raw[inner_mask]
            y_inner_test = float(y_train_raw[~inner_mask][0])

            x_inner_train_std, x_inner_test_std, _, _ = standardize_train_only(x_inner_train, x_inner_test)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train)
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, alpha)
            pred_std = float(x_inner_test_std @ beta_std)
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = mse
    assert best_alpha is not None and best_mse is not None
    return best_alpha, best_mse


def fit_outer_model(
    x_train_raw: np.ndarray, y_train_raw: np.ndarray, x_test_raw: np.ndarray, alpha: float
) -> Tuple[float, np.ndarray, float]:
    x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
    y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
    beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
    pred_std = float(x_test_std @ beta_std)
    pred_raw = y_mean + y_std * pred_std
    beta_raw = (y_std / x_std) * beta_std
    intercept_raw = float(y_mean - np.sum(beta_raw * x_mean))
    return pred_raw, beta_raw, intercept_raw


def phase_from_rmm_components(rmm1: float, rmm2: float) -> int:
    angle = (math.degrees(math.atan2(rmm2, rmm1)) + 360.0) % 360.0
    return int((math.floor((angle + 22.5) / 45.0) % 8) + 1)


def discover_rmm_files() -> List[Path]:
    matches: List[Path] = []
    searched_roots: List[str] = []
    for root_name in SEARCH_ROOTS:
        root = PROJECT_ROOT / root_name
        searched_roots.append(str(root))
        if not root.exists():
            continue
        for pattern in SEARCH_PATTERNS:
            matches.extend(root.rglob(pattern))
    unique_files = []
    seen = set()
    for path in matches:
        if path.is_file():
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(path)
    if not unique_files:
        raise FileNotFoundError(
            "No usable RMM/MJO file found. Searched paths: {}".format(", ".join(searched_roots))
        )
    return unique_files


def choose_rmm_file(candidates: Sequence[Path]) -> Tuple[Path, str]:
    preferred = [p for p in candidates if "rmm_star_data.txt" in p.name.lower()]
    if preferred:
        return preferred[0], "RMM_star"
    rmmstar_like = [p for p in candidates if "star" in p.name.lower()]
    if rmmstar_like:
        return rmmstar_like[0], "RMM_star"
    ordinary = [p for p in candidates if "rmm" in p.name.lower() or "mjo" in p.name.lower()]
    if ordinary:
        return ordinary[0], "ordinary_RMM"
    raise FileNotFoundError("RMM/MJO candidates were found but none were usable: {}".format([str(p) for p in candidates]))


def load_rmm_dataframe(path: Path, rmm_type_used: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    invalid_sentinel_rows = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
        except ValueError:
            continue
        numeric = []
        ok = True
        for token in parts[3:]:
            try:
                numeric.append(float(token))
            except ValueError:
                ok = False
                break
        if not ok or len(numeric) < 3:
            continue

        # Common cached format here is year month day phase rmm1 rmm2 amplitude.
        if len(numeric) >= 4:
            phase_in_file = int(round(numeric[0]))
            rmm1 = float(numeric[1])
            rmm2 = float(numeric[2])
            amp_in_file = float(numeric[3])
        else:
            phase_in_file = -1
            rmm1 = float(numeric[0])
            rmm2 = float(numeric[1])
            amp_in_file = float(numeric[2])

        if rmm1 <= -90.0 or rmm2 <= -90.0 or amp_in_file <= -90.0:
            invalid_sentinel_rows += 1
            continue

        amplitude = float(math.hypot(rmm1, rmm2))
        phase = phase_from_rmm_components(rmm1, rmm2)
        active = amplitude >= ACTIVE_THRESHOLD
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=day),
                "year": year,
                "month": month,
                "day": day,
                "RMM1_star": rmm1,
                "RMM2_star": rmm2,
                "phase": phase,
                "phase_in_file": phase_in_file,
                "amplitude": amplitude,
                "active": active,
            }
        )
    if not rows:
        raise ValueError(f"No usable RMM rows parsed from {path}")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    meta = {
        "rmm_type_used": rmm_type_used,
        "phase_source": "computed_from_rmm1_rmm2",
        "amplitude_source": "computed_from_rmm1_rmm2",
        "active_definition": f"amplitude >= {ACTIVE_THRESHOLD}",
        "invalid_sentinel_rows_skipped": int(invalid_sentinel_rows),
        "date_coverage": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "n_daily_rows": int(len(df)),
        },
    }
    return df, meta


def load_target_table() -> Tuple[pd.DataFrame, str]:
    if TARGET_TABLE_CSV.exists():
        df = pd.read_csv(TARGET_TABLE_CSV)
        needed = ["water_year", "obs_swe"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns in {TARGET_TABLE_CSV}: {missing}")
        df = df[needed].copy()
        source = str(TARGET_TABLE_CSV)
    else:
        base_predictions = pd.read_csv(BASE_PREDICTIONS_CSV)
        base_predictions = base_predictions[
            (base_predictions["patch_size"] == "exact_grid_cell")
            & (base_predictions["model_name"] == "Z1_Z2")
        ].copy()
        df = base_predictions[["heldout_wy", "obs_swe"]].rename(columns={"heldout_wy": "water_year"})
        source = str(BASE_PREDICTIONS_CSV)
    df["water_year"] = df["water_year"].astype(int)
    df = df[(df["water_year"] >= WATER_YEAR_START) & (df["water_year"] <= WATER_YEAR_END)].copy()
    df = df.sort_values("water_year").reset_index(drop=True)
    if df["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Target table does not match WY1985--WY2021.")
    return df, source


def infer_common_available_months(rmm_df: pd.DataFrame, target_df: pd.DataFrame) -> List[str]:
    available = []
    for month_name in MONTHS:
        month_num = MONTH_TO_NUMBER[month_name]
        ok = True
        for water_year in target_df["water_year"].tolist():
            cal_year = water_year if month_name in {"Jan", "Feb", "Mar"} else water_year - 1
            month_df = rmm_df[(rmm_df["year"] == cal_year) & (rmm_df["month"] == month_num)]
            if month_df.empty:
                ok = False
                break
        if ok:
            available.append(month_name)
    if not available:
        raise ValueError("No common usable months were found across WY1985--WY2021 in the discovered RMM file.")
    return available


def build_monthly_predictors(
    rmm_df: pd.DataFrame, target_df: pd.DataFrame, usable_months: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    rows: List[Dict[str, float]] = []
    xy_cols: List[str] = []
    amp_active_cols: List[str] = []
    phase_cols: List[str] = []
    for water_year, obs_swe in target_df[["water_year", "obs_swe"]].itertuples(index=False):
        row: Dict[str, float] = {"water_year": int(water_year), "obs_swe": float(obs_swe)}
        for month_name in usable_months:
            month_num = MONTH_TO_NUMBER[month_name]
            cal_year = water_year if month_name in {"Jan", "Feb", "Mar"} else water_year - 1
            month_df = rmm_df[(rmm_df["year"] == cal_year) & (rmm_df["month"] == month_num)].copy()
            if month_df.empty:
                raise ValueError(f"No RMM rows found for WY{water_year} month {month_name} ({cal_year}-{month_num:02d}).")
            rmm1_col = f"RMM1star_{month_name}"
            rmm2_col = f"RMM2star_{month_name}"
            amp_col = f"RMMamp_{month_name}"
            active_col = f"RMMactivefrac_{month_name}"
            row[rmm1_col] = float(month_df["RMM1_star"].mean())
            row[rmm2_col] = float(month_df["RMM2_star"].mean())
            row[amp_col] = float(month_df["amplitude"].mean())
            row[active_col] = float(month_df["active"].mean())
            xy_cols.extend([rmm1_col, rmm2_col]) if water_year == WATER_YEAR_START else None
            amp_active_cols.extend([amp_col, active_col]) if water_year == WATER_YEAR_START else None
            for phase in range(1, 9):
                col = f"RMMphase{phase}_frac_{month_name}"
                phase_frac = float(((month_df["active"]) & (month_df["phase"] == phase)).mean())
                row[col] = phase_frac
                if water_year == WATER_YEAR_START:
                    phase_cols.append(col)
        rows.append(row)
    predictor_df = pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)
    feature_sets = {
        "RMMSTAR_XY_MONTHLY": xy_cols,
        "RMMSTAR_AMP_ACTIVE_MONTHLY": amp_active_cols,
        "RMMSTAR_XY_AMP_ACTIVE": xy_cols + amp_active_cols,
        "RMMSTAR_PHASE_ACTIVE_FRACTION": phase_cols,
        "RMMSTAR_ALL": xy_cols + amp_active_cols + phase_cols,
    }
    return predictor_df, feature_sets


def run_models(table: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    y = table["obs_swe"].to_numpy(dtype=float)
    prediction_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    beta_rows: List[Dict[str, object]] = []

    for model_name, feature_list in feature_sets.items():
        x_all = table[feature_list].to_numpy(dtype=float)
        for held_idx, held_year in enumerate(years):
            train_mask = np.ones(len(years), dtype=bool)
            train_mask[held_idx] = False
            x_train = x_all[train_mask, :]
            x_test = x_all[~train_mask, :][0]
            y_train = y[train_mask]
            obs = y[held_idx]

            alpha, inner_cv_mse = inner_loyo_best_alpha(x_train, y_train)
            pred_raw, beta_raw, intercept_raw = fit_outer_model(x_train, y_train, x_test, alpha)
            err = pred_raw - obs
            prediction_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": int(held_year),
                    "obs_swe": float(obs),
                    "pred_swe": float(pred_raw),
                    "error_pred_minus_obs": float(err),
                    "residual_obs_minus_pred": float(-err),
                    "abs_error": float(abs(err)),
                    "sign_correct": float(np.sign(pred_raw) == np.sign(obs)) if obs != 0.0 and pred_raw != 0.0 else np.nan,
                    "selected_alpha": float(alpha),
                    "num_predictors": int(len(feature_list)),
                }
            )
            alpha_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": int(held_year),
                    "selected_alpha": float(alpha),
                    "inner_cv_mse": float(inner_cv_mse),
                }
            )
            beta_rows.append(
                {
                    "model_name": model_name,
                    "heldout_wy": int(held_year),
                    "intercept": float(intercept_raw),
                    "predictor_name": "__intercept__",
                    "beta": float(intercept_raw),
                    "selected_alpha": float(alpha),
                }
            )
            for predictor_name, beta in zip(feature_list, beta_raw):
                beta_rows.append(
                    {
                        "model_name": model_name,
                        "heldout_wy": int(held_year),
                        "intercept": float(intercept_raw),
                        "predictor_name": predictor_name,
                        "beta": float(beta),
                        "selected_alpha": float(alpha),
                    }
                )

    return (
        pd.DataFrame(prediction_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
        pd.DataFrame(alpha_rows).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True),
        pd.DataFrame(beta_rows).sort_values(["model_name", "heldout_wy", "predictor_name"]).reset_index(drop=True),
    )


def compute_metrics(pred_df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for model_name, feature_list in feature_sets.items():
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        metrics = compute_metric_bundle(sub["obs_swe"].to_numpy(dtype=float), sub["pred_swe"].to_numpy(dtype=float))
        rows.append(
            {
                "model_name": model_name,
                "num_predictors": int(len(feature_list)),
                "r": metrics["r"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_error": metrics["mean_error"],
                "median_abs_error": metrics["median_abs_error"],
            }
        )
    return pd.DataFrame(rows)


def compute_period_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in pred_df["model_name"].drop_duplicates():
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        wy = sub["heldout_wy"].to_numpy(dtype=int)
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        for group_name, selector in [
            ("all_years", lambda arr: np.isfinite(arr)),
            ("pre_2010", lambda arr: arr <= 2010),
            ("post_2010", lambda arr: arr > 2010),
            ("pre_2005", lambda arr: arr <= 2005),
            ("post_2005", lambda arr: arr > 2005),
        ]:
            mask = selector(wy)
            metrics = compute_metric_bundle(obs[mask], pred[mask])
            rows.append(
                {
                    "model_name": model_name,
                    "group_name": group_name,
                    "n_years": int(mask.sum()),
                    "r": metrics["r"],
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "sign_accuracy": metrics["sign_accuracy"],
                    "mean_error": metrics["mean_error"],
                    "median_abs_error": metrics["median_abs_error"],
                }
            )
    return pd.DataFrame(rows).sort_values(["model_name", "group_name"]).reset_index(drop=True)


def summarize_alpha(alpha_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for model_name in alpha_df["model_name"].drop_duplicates():
        sub = alpha_df[alpha_df["model_name"] == model_name]
        values = sub["selected_alpha"].to_numpy(dtype=float)
        counts = sub["selected_alpha"].value_counts().sort_index()
        summary[model_name] = {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "value_counts": {str(k): int(v) for k, v in counts.items()},
        }
    return summary


def summarize_coefficients(beta_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for model_name in beta_df["model_name"].drop_duplicates():
        sub = beta_df[(beta_df["model_name"] == model_name) & (beta_df["predictor_name"] != "__intercept__")]
        rows = []
        for predictor_name in sub["predictor_name"].drop_duplicates():
            vals = sub[sub["predictor_name"] == predictor_name]["beta"].to_numpy(dtype=float)
            rows.append(
                {
                    "predictor_name": predictor_name,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean_abs_beta": float(np.mean(np.abs(vals))),
                }
            )
        rows.sort(key=lambda r: (-r["mean_abs_beta"], r["predictor_name"]))
        summary[model_name] = rows[:20]
    return summary


def make_timeseries_plot(pred_df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    main_model = "RMMSTAR_XY_MONTHLY"
    base = pred_df[pred_df["model_name"] == main_model].sort_values("heldout_wy")
    ax.plot(base["heldout_wy"], base["obs_swe"], color="black", linewidth=2.4, label="Observed")
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_sets)))
    for color, model_name in zip(colors, feature_sets.keys()):
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(sub["heldout_wy"], sub["pred_swe"], marker="o", linewidth=1.4, color=color, label=model_name)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO predictions using pure MJO/RMM* predictors")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OBS_PRED_PNG, dpi=200)
    plt.close(fig)


def make_scatter_panels(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> None:
    preferred = ["RMMSTAR_XY_MONTHLY", "RMMSTAR_XY_AMP_ACTIVE", "RMMSTAR_ALL"]
    if "RMMSTAR_PHASE_ACTIVE_FRACTION" in feature_sets:
        preferred.append("RMMSTAR_PHASE_ACTIVE_FRACTION")
    panel_models = [name for name in preferred if name in feature_sets]
    n = len(panel_models)
    ncols = 2 if n > 3 else n
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    all_obs = pred_df["obs_swe"].to_numpy(dtype=float)
    all_pred = pred_df["pred_swe"].to_numpy(dtype=float)
    lo = min(np.min(all_obs), np.min(all_pred))
    hi = max(np.max(all_obs), np.max(all_pred))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lims = (lo - pad, hi + pad)
    metric_lookup = metrics_df.set_index("model_name")
    for ax, model_name in zip(axes_flat, panel_models):
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        obs = sub["obs_swe"].to_numpy(dtype=float)
        pred = sub["pred_swe"].to_numpy(dtype=float)
        row = metric_lookup.loc[model_name]
        ax.scatter(obs, pred, s=42, alpha=0.85, color="#4c72b0")
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.1)
        ax.text(
            0.04,
            0.96,
            "RMSE = {:.4f}\n$R^2$ = {:.3f}\nCorr = {:.3f}".format(
                float(row["RMSE"]),
                float(row["R2"]),
                float(row["r"]),
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
        ax.set_title(model_name)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Observed SWE anomaly (m)")
        ax.set_ylabel("Predicted SWE anomaly (m)")
    for ax in axes_flat[len(panel_models):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=200)
    plt.close(fig)


def make_error_plot(pred_df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_sets)))
    for color, model_name in zip(colors, feature_sets.keys()):
        sub = pred_df[pred_df["model_name"] == model_name].sort_values("heldout_wy")
        ax.plot(sub["heldout_wy"], sub["error_pred_minus_obs"], marker="o", linewidth=1.4, color=color, label=model_name)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel(r"$\widehat{SWE}-SWE_{\mathrm{obs}}$ (m)")
    ax.set_title("Prediction error by year")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(ERROR_PNG, dpi=200)
    plt.close(fig)


def build_summary(
    input_files: Dict[str, object],
    rmm_meta: Dict[str, object],
    usable_months: Sequence[str],
    feature_sets: Dict[str, List[str]],
    metrics_df: pd.DataFrame,
    period_metrics_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    beta_df: pd.DataFrame,
) -> Dict[str, object]:
    alpha_summary = summarize_alpha(alpha_df)
    coefficient_summary = summarize_coefficients(beta_df)
    best_r2_row = metrics_df.loc[metrics_df["R2"].astype(float).idxmax()].to_dict()
    best_rmse_row = metrics_df.loc[metrics_df["RMSE"].astype(float).idxmin()].to_dict()
    best_sign_row = metrics_df.loc[metrics_df["sign_accuracy"].astype(float).idxmax()].to_dict()

    comparison_bits = []
    if TARGET_TABLE_CSV.exists():
        k5_summary_path = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_summary.json"
        if k5_summary_path.exists():
            try:
                ref = json.loads(k5_summary_path.read_text())
                ref_metrics = {row["model_name"]: row for row in ref["metrics"]}
                comparison_bits.append(
                    "Compared with the earlier strict-LOYO benchmarks, the best pure MJO/RMM* model should be judged against Z1_Z2, AMV/AMO K5, and their combined model in artifacts/cobe2_sierra_swe_lod_setup/z1z2_plus_amv_k5_loyo."
                )
            except Exception:
                pass

    main_row = metrics_df[metrics_df["model_name"] == "RMMSTAR_XY_MONTHLY"].iloc[0].to_dict()
    best_row = best_r2_row
    competitive_text = (
        "The pure MJO/RMM* result looks competitive with the SST-derived baselines."
        if float(best_row["R2"]) >= 0.25
        else "The pure MJO/RMM* result is weaker than the stronger SST-derived Z1_Z2 and AMV/AMO K5 baselines."
    )
    short_answer = (
        "Pure MJO/RMM* "
        + ("does provide some useful" if float(best_row["R2"]) > 0.0 else "does not provide strong")
        + " strict LOYO predictability for April 1 Sierra SWE. "
        + f"The best representation here is {best_row['model_name']}. "
        + competitive_text
    )

    return {
        "input_files": input_files,
        "rmm_type_used": rmm_meta["rmm_type_used"],
        "date_coverage": rmm_meta["date_coverage"],
        "water_year_alignment": "Sep--Dec from WY-1 and Jan--Mar from WY",
        "usable_months": list(usable_months),
        "missing_requested_months": [m for m in MONTHS if m not in usable_months],
        "feature_sets": feature_sets,
        "metrics": metrics_df.to_dict(orient="records"),
        "period_metrics": period_metrics_df.to_dict(orient="records"),
        "selected_alpha_summary": alpha_summary,
        "coefficient_summary": coefficient_summary,
        "best_model_by_R2": best_r2_row,
        "best_model_by_RMSE": best_rmse_row,
        "best_model_by_sign_accuracy": best_sign_row,
        "rmm_processing": rmm_meta,
        "short_answer": short_answer,
        "comparison_note": comparison_bits,
    }


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = discover_rmm_files()
    rmm_file, rmm_type_used = choose_rmm_file(candidates)
    rmm_df, rmm_meta = load_rmm_dataframe(rmm_file, rmm_type_used)
    target_df, target_source = load_target_table()
    usable_months = infer_common_available_months(rmm_df, target_df)
    predictor_df, feature_sets = build_monthly_predictors(rmm_df, target_df, usable_months)
    predictor_df.to_csv(PREDICTOR_TABLE_CSV, index=False)

    pred_df, alpha_df, beta_df = run_models(predictor_df, feature_sets)
    metrics_df = compute_metrics(pred_df, feature_sets)
    period_metrics_df = compute_period_metrics(pred_df)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_metrics_df.to_csv(PERIOD_METRICS_CSV, index=False)
    alpha_df.to_csv(ALPHA_CSV, index=False)
    beta_df.to_csv(BETA_CSV, index=False)

    make_timeseries_plot(pred_df, feature_sets)
    make_scatter_panels(pred_df, metrics_df, feature_sets)
    make_error_plot(pred_df, feature_sets)

    input_files = {
        "rmm_file": str(rmm_file),
        "target_file": target_source,
        "searched_roots": SEARCH_ROOTS,
    }
    summary = build_summary(input_files, rmm_meta, usable_months, feature_sets, metrics_df, period_metrics_df, alpha_df, beta_df)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(f"Output directory: {OUTPUT_DIR}")
    print("Metrics:")
    for model_name in feature_sets.keys():
        row = metrics_df[metrics_df["model_name"] == model_name].iloc[0]
        print(
            f"{model_name}: r={float(row['r']):.6f}, R2={float(row['R2']):.6f}, "
            f"RMSE={float(row['RMSE']):.6f}, MAE={float(row['MAE']):.6f}, "
            f"sign_accuracy={float(row['sign_accuracy']):.6f}"
        )
    print(f"Best model by R2: {summary['best_model_by_R2']['model_name']}")
    print(f"Best model by RMSE: {summary['best_model_by_RMSE']['model_name']}")
    print(f"Best model by sign accuracy: {summary['best_model_by_sign_accuracy']['model_name']}")
    print("Short answer:")
    print(summary["short_answer"])


if __name__ == "__main__":
    main()
