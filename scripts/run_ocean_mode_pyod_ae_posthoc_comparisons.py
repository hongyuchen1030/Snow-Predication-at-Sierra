#!/usr/bin/env python3
"""Posthoc SWE and reconstruction comparisons for the completed PyOD ocean-mode AE run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ocean_mode_pyod_autoencoder_loyo"
BASELINE_7COL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "z1z2_plus_amv_k5_loyo"
    / "z1z2_amv_k5_predictor_table.csv"
)
ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
PCA_COMPONENTS = [3, 5, 7, 10]
REPORT_TITLE_1 = "Downstream SWE ridge comparison"
REPORT_TITLE_2 = "Ocean-mode reconstruction comparison against 7-column baseline"
NEAR_ZERO_STD = 1.0e-12


@dataclass(frozen=True)
class SettingKey:
    model_family: str
    model_name: str
    k: Optional[int] = None
    dropout_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    seed: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def append_or_replace_report_section(report_path: Path, section_title: str, section_lines: Sequence[str]) -> None:
    body = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    header = f"## {section_title}"
    replacement = "\n".join([header, ""] + list(section_lines)).rstrip() + "\n"
    if header not in body:
        if body and not body.endswith("\n"):
            body += "\n"
        if body:
            body = body.rstrip() + "\n\n"
        report_path.write_text(body + replacement, encoding="utf-8")
        return
    start = body.index(header)
    next_header = body.find("\n## ", start + len(header))
    if next_header == -1:
        updated = body[:start].rstrip() + "\n\n" + replacement
    else:
        updated = body[:start].rstrip() + "\n\n" + replacement + "\n" + body[next_header + 1 :].lstrip()
    report_path.write_text(updated, encoding="utf-8")


def standardize_train_only(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (np.abs(std) < NEAR_ZERO_STD), 1.0, std)
    return (x_train - mean[None, :]) / std[None, :], (x_test - mean) / std, mean, std


def fit_ridge_multioutput(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64), rhs)


def fit_ridge_scalar(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    return fit_ridge_multioutput(x_train_std, y_train_std[:, None], alpha)[:, 0]


def loocv_mse_for_alpha(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> float:
    y_mat = np.asarray(y_train_std, dtype=np.float64)
    if y_mat.ndim == 1:
        y_mat = y_mat[:, None]
    xtx = x_train_std.T @ x_train_std
    inv_term = np.linalg.inv(xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64))
    beta = inv_term @ x_train_std.T @ y_mat
    fitted = x_train_std @ beta
    hat_diag = np.sum((x_train_std @ inv_term) * x_train_std, axis=1)
    denom = 1.0 - hat_diag
    denom = np.where(np.abs(denom) < 1.0e-10, np.sign(denom) * 1.0e-10 + (denom == 0.0) * 1.0e-10, denom)
    residual = y_mat - fitted
    loo_residual = residual / denom[:, None]
    return float(np.mean(loo_residual**2))


def inner_loyo_best_alpha_scalar(x_train_std: np.ndarray, y_train_std: np.ndarray) -> Tuple[float, float]:
    best_alpha = float(ALPHA_GRID[0])
    best_mse = float("inf")
    for alpha in ALPHA_GRID.tolist():
        current_mse = loocv_mse_for_alpha(x_train_std, y_train_std, float(alpha))
        if current_mse < best_mse - 1.0e-15 or (
            abs(current_mse - best_mse) <= 1.0e-15 and float(alpha) < best_alpha
        ):
            best_alpha = float(alpha)
            best_mse = current_mse
    return best_alpha, best_mse


def inner_loyo_best_alpha_multioutput(x_train_std: np.ndarray, y_train_std: np.ndarray) -> Tuple[float, float]:
    best_alpha = float(ALPHA_GRID[0])
    best_mse = float("inf")
    for alpha in ALPHA_GRID.tolist():
        current_mse = loocv_mse_for_alpha(x_train_std, y_train_std, float(alpha))
        if current_mse < best_mse - 1.0e-15 or (
            abs(current_mse - best_mse) <= 1.0e-15 and float(alpha) < best_alpha
        ):
            best_alpha = float(alpha)
            best_mse = current_mse
    return best_alpha, best_mse


def compute_scalar_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    obs_std = float(np.std(obs, ddof=1))
    pred_std = float(np.std(pred, ddof=1))
    if obs_std < NEAR_ZERO_STD or pred_std < NEAR_ZERO_STD:
        r = float("nan")
    else:
        r = float(np.corrcoef(obs, pred)[0, 1])
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    sse = float(np.sum((obs - pred) ** 2))
    r2 = float("nan") if sst < NEAR_ZERO_STD else float(1.0 - sse / sst)
    sign_accuracy = float(
        np.mean(((np.sign(obs) == np.sign(pred)) & (obs != 0.0) & (pred != 0.0)).astype(np.float64))
    )
    return {
        "r": r,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "sign_accuracy": sign_accuracy,
        "mean_bias": bias,
        "prediction_std": pred_std,
        "observed_std": obs_std,
        "n_years": int(obs.size),
    }


def format_setting_fields(key: SettingKey) -> Dict[str, object]:
    return {
        "model_name": key.model_name,
        "model_family": key.model_family,
        "k": key.k if key.k is not None else np.nan,
        "dropout_rate": key.dropout_rate if key.dropout_rate is not None else np.nan,
        "weight_decay": key.weight_decay if key.weight_decay is not None else np.nan,
        "seed": key.seed if key.seed is not None else np.nan,
    }


def load_inputs(output_dir: Path) -> Dict[str, pd.DataFrame]:
    paths = {
        "baseline": BASELINE_7COL_PATH,
        "ocean": output_dir / "ocean_mode_predictor_table.csv",
        "latent": output_dir / "pyod_ae_latent_codes.csv",
        "ae_fold": output_dir / "pyod_ae_fold_metrics.csv",
        "native_resid": output_dir / "pyod_ae_reconstruction_residuals.csv",
        "report": output_dir / "REPORT.md",
    }
    for name, path in paths.items():
        if name != "report" and not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")
    baseline_df = pd.read_csv(paths["baseline"])
    ocean_df = pd.read_csv(paths["ocean"])
    latent_df = pd.read_csv(paths["latent"])
    ae_fold_df = pd.read_csv(paths["ae_fold"])
    native_resid_df = pd.read_csv(paths["native_resid"])
    return {
        "baseline": baseline_df,
        "ocean": ocean_df,
        "latent": latent_df,
        "ae_fold": ae_fold_df,
        "native_resid": native_resid_df,
    }


def validate_latent_artifact(latent_df: pd.DataFrame, years: Sequence[int]) -> None:
    group_cols = ["k", "seed", "dropout_rate", "weight_decay", "heldout_year"]
    expected_years = set(int(v) for v in years)
    for keys, sub in latent_df.groupby(group_cols, dropna=False):
        heldout_year = int(keys[-1])
        train = sub[sub["split_role"] == "train"]
        heldout = sub[sub["split_role"] == "heldout"]
        if len(heldout) != 1 or int(heldout.iloc[0]["water_year"]) != heldout_year:
            raise RuntimeError(
                f"Latent artifact is not leakage-safe for setting {keys}: expected one held-out row."
            )
        train_years = set(int(v) for v in train["water_year"].tolist())
        if train.shape[0] != len(years) - 1 or train_years != (expected_years - {heldout_year}):
            raise RuntimeError(
                f"Latent artifact is not leakage-safe for setting {keys}: training years mismatch."
            )


def build_foldsafe_latent_lookup(latent_df: pd.DataFrame) -> Tuple[Dict[Tuple[int, float, float, int], Dict[int, Dict[str, np.ndarray]]], int]:
    latent_cols = [col for col in latent_df.columns if col.startswith("z")]
    max_k = len(latent_cols)
    lookup: Dict[Tuple[int, float, float, int], Dict[int, Dict[str, np.ndarray]]] = {}
    for (k, dropout_rate, weight_decay, seed), setting_df in latent_df.groupby(
        ["k", "dropout_rate", "weight_decay", "seed"], dropna=False
    ):
        per_year: Dict[int, Dict[str, np.ndarray]] = {}
        for heldout_year, fold_df in setting_df.groupby("heldout_year", dropna=False):
            use_cols = latent_cols[: int(k)]
            train_df = fold_df[fold_df["split_role"] == "train"].sort_values("water_year")
            heldout_df = fold_df[fold_df["split_role"] == "heldout"].sort_values("water_year")
            per_year[int(heldout_year)] = {
                "train_years": train_df["water_year"].to_numpy(dtype=int),
                "train_z": train_df[use_cols].to_numpy(dtype=np.float64),
                "heldout_z": heldout_df[use_cols].to_numpy(dtype=np.float64)[0],
            }
        lookup[(int(k), float(dropout_rate), float(weight_decay), int(seed))] = per_year
    return lookup, max_k


def pca_scores_from_outer_fold(x_train_raw: np.ndarray, x_test_raw: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x_train_std, x_test_std, _, _ = standardize_train_only(x_train_raw, x_test_raw)
    _, _, vt = np.linalg.svd(x_train_std, full_matrices=False)
    components = vt[:k].T
    return x_train_std @ components, x_test_std @ components


def evaluate_swe_representation(
    key: SettingKey,
    years: np.ndarray,
    y_obs: np.ndarray,
    rep_getter,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    pred_rows: List[Dict[str, object]] = []
    preds = np.full(years.shape, np.nan, dtype=np.float64)
    alphas = np.full(years.shape, np.nan, dtype=np.float64)
    print(f"Starting SWE evaluation for {key.model_name}", flush=True)
    for outer_idx, heldout_year in enumerate(years.tolist()):
        train_mask = years != heldout_year
        y_train = y_obs[train_mask]
        y_test = float(y_obs[~train_mask][0])
        x_train_raw, x_test_raw = rep_getter(int(heldout_year))
        x_train_std, x_test_std, _, _ = standardize_train_only(x_train_raw, x_test_raw)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train, ddof=1))
        if not np.isfinite(y_std) or abs(y_std) < NEAR_ZERO_STD:
            y_std = 1.0
        y_train_std = (y_train - y_mean) / y_std
        alpha, inner_mse = inner_loyo_best_alpha_scalar(x_train_std, y_train_std)
        beta_std = fit_ridge_scalar(x_train_std, y_train_std, alpha)
        pred = y_mean + y_std * float(x_test_std @ beta_std)
        preds[outer_idx] = pred
        alphas[outer_idx] = alpha
        pred_rows.append(
            {
                **format_setting_fields(key),
                "water_year": int(heldout_year),
                "y_obs": y_test,
                "y_pred": pred,
                "error_pred_minus_obs": pred - y_test,
                "abs_error": abs(pred - y_test),
                "sign_correct": float((np.sign(pred) == np.sign(y_test)) and (pred != 0.0) and (y_test != 0.0)),
                "alpha_selected": alpha,
                "inner_cv_mse": inner_mse,
            }
        )
    metrics = compute_scalar_metrics(y_obs, preds)
    metrics_row = {
        **format_setting_fields(key),
        **metrics,
        "alpha_selected_median": float(np.nanmedian(alphas)),
        "alpha_selected_mode": float(pd.Series(alphas).value_counts().sort_index().idxmax()),
    }
    print(
        f"Finished SWE evaluation for {key.model_name}: RMSE={metrics_row['RMSE']:.6f}, r={metrics_row['r']:.6f}",
        flush=True,
    )
    return pred_rows, metrics_row


def evaluate_reconstruction_representation(
    key: SettingKey,
    years: np.ndarray,
    x_target_raw: np.ndarray,
    rep_getter,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    pred_rows: List[Dict[str, object]] = []
    heldout_mse: List[float] = []
    train_mse: List[float] = []
    heldout_year_to_mse: Dict[int, float] = {}
    selected_alphas: List[float] = []
    target_names = None
    print(f"Starting reconstruction evaluation for {key.model_name}", flush=True)
    for outer_idx, heldout_year in enumerate(years.tolist()):
        train_mask = years != heldout_year
        x_train_target = x_target_raw[train_mask]
        x_test_target = x_target_raw[~train_mask][0]
        x_train_rep_raw, x_test_rep_raw, target_names = rep_getter(int(heldout_year))
        x_train_rep_std, x_test_rep_std, _, _ = standardize_train_only(x_train_rep_raw, x_test_rep_raw)
        x_train_target_std, x_test_target_std, _, _ = standardize_train_only(x_train_target, x_test_target)
        alpha, inner_mse = inner_loyo_best_alpha_multioutput(x_train_rep_std, x_train_target_std)
        beta_std = fit_ridge_multioutput(x_train_rep_std, x_train_target_std, alpha)
        train_pred_std = x_train_rep_std @ beta_std
        heldout_pred_std = x_test_rep_std @ beta_std
        fold_train_mse = float(np.mean((x_train_target_std - train_pred_std) ** 2))
        fold_test_mse = float(np.mean((x_test_target_std - heldout_pred_std) ** 2))
        train_mse.append(fold_train_mse)
        heldout_mse.append(fold_test_mse)
        heldout_year_to_mse[int(heldout_year)] = fold_test_mse
        selected_alphas.append(alpha)
        for target_idx, target_name in enumerate(target_names):
            pred_rows.append(
                {
                    **format_setting_fields(key),
                    "water_year": int(heldout_year),
                    "target_column": target_name,
                    "x_obs_standardized": float(x_test_target_std[target_idx]),
                    "x_pred_standardized": float(heldout_pred_std[target_idx]),
                    "alpha_selected": alpha,
                    "inner_cv_mse": inner_mse,
                }
            )
    heldout_mse_array = np.asarray(heldout_mse, dtype=np.float64)
    train_mse_array = np.asarray(train_mse, dtype=np.float64)
    ratio = heldout_mse_array / np.where(train_mse_array > 0.0, train_mse_array, np.nan)
    metrics_row = {
        **format_setting_fields(key),
        "mean_heldout_reconstruction_mse": float(np.mean(heldout_mse_array)),
        "median_heldout_reconstruction_mse": float(np.median(heldout_mse_array)),
        "heldout_reconstruction_rmse": float(np.sqrt(np.mean(heldout_mse_array))),
        "mean_training_reconstruction_mse": float(np.mean(train_mse_array)),
        "mean_generalization_ratio": float(np.nanmean(ratio)),
        "alpha_selected_median": float(np.median(selected_alphas)),
        "alpha_selected_mode": float(pd.Series(selected_alphas).value_counts().sort_index().idxmax()),
        "best_heldout_year": int(min(heldout_year_to_mse, key=heldout_year_to_mse.get)),
        "worst_heldout_year": int(max(heldout_year_to_mse, key=heldout_year_to_mse.get)),
    }
    print(
        "Finished reconstruction evaluation for {name}: heldout_MSE={mse:.6f}, ratio={ratio:.6f}".format(
            name=key.model_name,
            mse=metrics_row["mean_heldout_reconstruction_mse"],
            ratio=metrics_row["mean_generalization_ratio"],
        ),
        flush=True,
    )
    return pred_rows, metrics_row


def evaluate_null_reconstruction(
    years: np.ndarray,
    x_target_raw: np.ndarray,
    target_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    key = SettingKey(model_family="null_reconstruction", model_name="NULL_MEAN_RECONSTRUCTION")
    pred_rows: List[Dict[str, object]] = []
    heldout_mse: List[float] = []
    for heldout_year in years.tolist():
        train_mask = years != heldout_year
        x_train_target = x_target_raw[train_mask]
        x_test_target = x_target_raw[~train_mask][0]
        _, x_test_target_std, _, _ = standardize_train_only(x_train_target, x_test_target)
        heldout_pred_std = np.zeros_like(x_test_target_std)
        heldout_mse.append(float(np.mean((x_test_target_std - heldout_pred_std) ** 2)))
        for target_idx, target_name in enumerate(target_names):
            pred_rows.append(
                {
                    **format_setting_fields(key),
                    "water_year": int(heldout_year),
                    "target_column": target_name,
                    "x_obs_standardized": float(x_test_target_std[target_idx]),
                    "x_pred_standardized": 0.0,
                    "alpha_selected": np.nan,
                    "inner_cv_mse": np.nan,
                }
            )
    heldout_array = np.asarray(heldout_mse, dtype=np.float64)
    metrics_row = {
        **format_setting_fields(key),
        "mean_heldout_reconstruction_mse": float(np.mean(heldout_array)),
        "median_heldout_reconstruction_mse": float(np.median(heldout_array)),
        "heldout_reconstruction_rmse": float(np.sqrt(np.mean(heldout_array))),
        "mean_training_reconstruction_mse": np.nan,
        "mean_generalization_ratio": np.nan,
        "alpha_selected_median": np.nan,
        "alpha_selected_mode": np.nan,
        "best_heldout_year": int(years[np.argmin(heldout_array)]),
        "worst_heldout_year": int(years[np.argmax(heldout_array)]),
    }
    return pred_rows, metrics_row


def build_swe_summary_by_setting(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = metrics_df.copy()
    summary["rmse_rank"] = summary["RMSE"].rank(method="min", ascending=True)
    summary["r_rank"] = summary["r"].rank(method="min", ascending=False)
    summary["sign_accuracy_rank"] = summary["sign_accuracy"].rank(method="min", ascending=False)
    return summary.sort_values(["rmse_rank", "r_rank", "model_family", "model_name"]).reset_index(drop=True)


def build_reconstruction_summary_by_k(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ae = metrics_df[metrics_df["model_family"] == "ae_latent_ridge_decoder"].copy()
    rows: List[Dict[str, object]] = []
    for k_value, sub in ae.groupby("k", dropna=False):
        rows.append(
            {
                "k": int(k_value),
                "mean_heldout_reconstruction_mse_across_settings": float(
                    sub["mean_heldout_reconstruction_mse"].mean()
                ),
                "median_heldout_reconstruction_mse_across_settings": float(
                    sub["mean_heldout_reconstruction_mse"].median()
                ),
                "std_heldout_reconstruction_mse_across_settings": float(
                    sub["mean_heldout_reconstruction_mse"].std(ddof=1)
                ),
                "best_heldout_reconstruction_mse": float(sub["mean_heldout_reconstruction_mse"].min()),
                "mean_training_reconstruction_mse": float(sub["mean_training_reconstruction_mse"].mean()),
                "mean_generalization_ratio": float(sub["mean_generalization_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def summarize_native_decoder(ae_fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (k, dropout_rate, weight_decay, seed), sub in ae_fold_df.groupby(
        ["k", "dropout_rate", "weight_decay", "seed"], dropna=False
    ):
        rows.append(
            {
                **format_setting_fields(
                    SettingKey(
                        model_family="native_ae_decoder_reconstruction",
                        model_name=f"NATIVE_AE_DECODER_k{int(k)}_d{float(dropout_rate):g}_wd{float(weight_decay):g}_seed{int(seed)}",
                        k=int(k),
                        dropout_rate=float(dropout_rate),
                        weight_decay=float(weight_decay),
                        seed=int(seed),
                    )
                ),
                "mean_heldout_reconstruction_mse": float(sub["test_recon_mse"].mean()),
                "median_heldout_reconstruction_mse": float(sub["test_recon_mse"].median()),
                "heldout_reconstruction_rmse": float(np.sqrt(sub["test_recon_mse"].mean())),
                "mean_training_reconstruction_mse": float(sub["train_recon_mse"].mean()),
                "mean_generalization_ratio": float(sub["generalization_ratio"].mean()),
                "alpha_selected_median": np.nan,
                "alpha_selected_mode": np.nan,
                "best_heldout_year": int(sub.sort_values("test_recon_mse").iloc[0]["heldout_year"]),
                "worst_heldout_year": int(sub.sort_values("test_recon_mse", ascending=False).iloc[0]["heldout_year"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "dropout_rate", "weight_decay", "seed"]).reset_index(drop=True)


def plot_swe_rmse_by_k(metrics_df: pd.DataFrame, out_path: Path) -> None:
    baseline_rmse = float(metrics_df.loc[metrics_df["model_family"] == "baseline_7col_swe", "RMSE"].iloc[0])
    ae = metrics_df[metrics_df["model_family"] == "ae_latent_swe"].copy()
    summary = ae.groupby("k", as_index=False)["RMSE"].min().sort_values("k")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(summary["k"], summary["RMSE"], marker="o", linewidth=2.0, label="Best AE setting per k")
    ax.axhline(baseline_rmse, color="#d62728", linestyle="--", linewidth=1.8, label="7-column baseline")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("SWE RMSE")
    ax.set_title("Strict LOYO SWE RMSE: AE latent ridge vs 7-column baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_swe_r_by_k(metrics_df: pd.DataFrame, out_path: Path) -> None:
    baseline_r = float(metrics_df.loc[metrics_df["model_family"] == "baseline_7col_swe", "r"].iloc[0])
    ae = metrics_df[metrics_df["model_family"] == "ae_latent_swe"].copy()
    summary = ae.groupby("k", as_index=False)["r"].max().sort_values("k")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(summary["k"], summary["r"], marker="o", linewidth=2.0, label="Best AE setting per k")
    ax.axhline(baseline_r, color="#d62728", linestyle="--", linewidth=1.8, label="7-column baseline")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Pearson r")
    ax.set_title("Strict LOYO SWE correlation: AE latent ridge vs 7-column baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_swe_timeseries(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path) -> None:
    best_ae_name = metrics_df[metrics_df["model_family"] == "ae_latent_swe"].sort_values("RMSE").iloc[0]["model_name"]
    best_ae = pred_df[pred_df["model_name"] == best_ae_name].sort_values("water_year")
    baseline = pred_df[pred_df["model_family"] == "baseline_7col_swe"].sort_values("water_year")
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    ax.plot(best_ae["water_year"], best_ae["y_obs"], color="black", linewidth=2.4, label="Observed SWE")
    ax.plot(best_ae["water_year"], best_ae["y_pred"], color="#1f77b4", linewidth=2.0, label="Best AE latent ridge")
    ax.plot(baseline["water_year"], baseline["y_pred"], color="#d62728", linewidth=1.8, linestyle="--", label="7-column baseline")
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly")
    ax.set_title("Strict LOYO SWE predictions: best AE latent ridge vs 7-column baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_swe_scatter(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path) -> None:
    best_ae_name = metrics_df[metrics_df["model_family"] == "ae_latent_swe"].sort_values("RMSE").iloc[0]["model_name"]
    best_ae = pred_df[pred_df["model_name"] == best_ae_name].sort_values("water_year")
    baseline = pred_df[pred_df["model_family"] == "baseline_7col_swe"].sort_values("water_year")
    fig, ax = plt.subplots(figsize=(5.6, 5.6), constrained_layout=True)
    ax.scatter(best_ae["y_obs"], best_ae["y_pred"], s=50, color="#1f77b4", label="Best AE latent ridge")
    ax.scatter(baseline["y_obs"], baseline["y_pred"], s=50, color="#d62728", marker="x", label="7-column baseline")
    limits = [
        float(min(best_ae["y_obs"].min(), best_ae["y_pred"].min(), baseline["y_pred"].min())),
        float(max(best_ae["y_obs"].max(), best_ae["y_pred"].max(), baseline["y_pred"].max())),
    ]
    ax.plot(limits, limits, color="black", linewidth=1.2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed SWE")
    ax.set_ylabel("Predicted SWE")
    ax.set_title("Strict LOYO SWE: observed vs predicted")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_recon_mse_by_k(metrics_df: pd.DataFrame, out_path: Path) -> None:
    ae = metrics_df[metrics_df["model_family"] == "ae_latent_ridge_decoder"].copy()
    baseline = float(
        metrics_df.loc[metrics_df["model_family"] == "baseline_7col_ridge_decoder", "mean_heldout_reconstruction_mse"].iloc[0]
    )
    null_mse = float(
        metrics_df.loc[metrics_df["model_family"] == "null_reconstruction", "mean_heldout_reconstruction_mse"].iloc[0]
    )
    summary = ae.groupby("k", as_index=False)["mean_heldout_reconstruction_mse"].min().sort_values("k")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(summary["k"], summary["mean_heldout_reconstruction_mse"], marker="o", linewidth=2.0, label="Best AE ridge decoder per k")
    ax.axhline(baseline, color="#d62728", linestyle="--", linewidth=1.8, label="7-column ridge decoder")
    ax.axhline(null_mse, color="#7f7f7f", linestyle=":", linewidth=1.8, label="Null mean reconstruction")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Held-out reconstruction MSE")
    ax.set_title("Ridge-decoder reconstruction of 91-column ocean-mode table")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_recon_relative_improvement(metrics_df: pd.DataFrame, out_path: Path) -> None:
    ae = metrics_df[metrics_df["model_family"] == "ae_latent_ridge_decoder"].copy()
    baseline_imp = float(
        metrics_df.loc[metrics_df["model_family"] == "baseline_7col_ridge_decoder", "relative_improvement_vs_null"].iloc[0]
    )
    summary = ae.groupby("k", as_index=False)["relative_improvement_vs_null"].max().sort_values("k")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(summary["k"], summary["relative_improvement_vs_null"], marker="o", linewidth=2.0, label="Best AE ridge decoder per k")
    ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.8, label="Null baseline")
    ax.axhline(baseline_imp, color="#d62728", linestyle="--", linewidth=1.8, label="7-column ridge decoder")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Relative improvement vs null")
    ax.set_title("Relative reconstruction improvement over null mean baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_recon_generalization_ratio(metrics_df: pd.DataFrame, out_path: Path) -> None:
    ae = metrics_df[metrics_df["model_family"] == "ae_latent_ridge_decoder"].copy()
    baseline = float(
        metrics_df.loc[metrics_df["model_family"] == "baseline_7col_ridge_decoder", "mean_generalization_ratio"].iloc[0]
    )
    summary = ae.groupby("k", as_index=False)["mean_generalization_ratio"].min().sort_values("k")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(summary["k"], summary["mean_generalization_ratio"], marker="o", linewidth=2.0, label="Best AE ratio per k")
    ax.axhline(baseline, color="#d62728", linestyle="--", linewidth=1.8, label="7-column ridge decoder")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Held-out / train reconstruction ratio")
    ax.set_title("Ridge-decoder reconstruction generalization ratio by k")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    inputs = load_inputs(output_dir)
    baseline_df = inputs["baseline"].sort_values("water_year").reset_index(drop=True)
    ocean_df = inputs["ocean"].sort_values("water_year").reset_index(drop=True)
    latent_df = inputs["latent"].copy()
    ae_fold_df = inputs["ae_fold"].copy()
    native_resid_df = inputs["native_resid"].copy()

    years = baseline_df["water_year"].to_numpy(dtype=int)
    if years.tolist() != ocean_df["water_year"].to_numpy(dtype=int).tolist():
        raise RuntimeError("Water years do not match between baseline table and ocean-mode table.")
    validate_latent_artifact(latent_df, years.tolist())

    y_obs = baseline_df["obs_swe"].to_numpy(dtype=np.float64)
    baseline_features = [col for col in baseline_df.columns if col not in {"water_year", "obs_swe"}]
    ocean_features = [col for col in ocean_df.columns if col != "water_year"]
    x_baseline = baseline_df[baseline_features].to_numpy(dtype=np.float64)
    x_ocean = ocean_df[ocean_features].to_numpy(dtype=np.float64)
    latent_lookup, _ = build_foldsafe_latent_lookup(latent_df)

    swe_pred_rows: List[Dict[str, object]] = []
    swe_metric_rows: List[Dict[str, object]] = []

    baseline_key = SettingKey(model_family="baseline_7col_swe", model_name="BASELINE_7COL_SWE")
    rows, metrics = evaluate_swe_representation(
        baseline_key,
        years,
        y_obs,
        lambda heldout_year: (
            x_baseline[years != heldout_year],
            x_baseline[years == heldout_year][0],
        ),
    )
    swe_pred_rows.extend(rows)
    swe_metric_rows.append(metrics)

    full_key = SettingKey(model_family="full_ocean_mode_swe", model_name="FULL_91COL_OCEAN_MODE_SWE")
    rows, metrics = evaluate_swe_representation(
        full_key,
        years,
        y_obs,
        lambda heldout_year: (
            x_ocean[years != heldout_year],
            x_ocean[years == heldout_year][0],
        ),
    )
    swe_pred_rows.extend(rows)
    swe_metric_rows.append(metrics)

    for k in PCA_COMPONENTS:
        pca_key = SettingKey(model_family="pca_swe", model_name=f"PCA{k}_SWE", k=int(k))
        rows, metrics = evaluate_swe_representation(
            pca_key,
            years,
            y_obs,
            lambda heldout_year, k_value=k: pca_scores_from_outer_fold(
                x_ocean[years != heldout_year], x_ocean[years == heldout_year][0], int(k_value)
            ),
        )
        swe_pred_rows.extend(rows)
        swe_metric_rows.append(metrics)

    for (k, dropout_rate, weight_decay, seed), per_year in latent_lookup.items():
        key = SettingKey(
            model_family="ae_latent_swe",
            model_name=f"AE_LATENT_k{k}_d{dropout_rate:g}_wd{weight_decay:g}_seed{seed}",
            k=int(k),
            dropout_rate=float(dropout_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )
        rows, metrics = evaluate_swe_representation(
            key,
            years,
            y_obs,
            lambda heldout_year, lookup=per_year: (
                lookup[int(heldout_year)]["train_z"],
                lookup[int(heldout_year)]["heldout_z"],
            ),
        )
        swe_pred_rows.extend(rows)
        swe_metric_rows.append(metrics)

    swe_pred_df = pd.DataFrame(swe_pred_rows).sort_values(
        ["model_family", "k", "dropout_rate", "weight_decay", "seed", "water_year"],
        na_position="last",
    ).reset_index(drop=True)
    swe_metrics_df = pd.DataFrame(swe_metric_rows).sort_values(
        ["model_family", "RMSE", "r"], na_position="last"
    ).reset_index(drop=True)
    swe_summary_df = build_swe_summary_by_setting(swe_metrics_df)

    best_rmse_row = swe_metrics_df.sort_values(["RMSE", "r"], na_position="last").iloc[0]
    best_r_row = swe_metrics_df.sort_values(["r", "RMSE"], ascending=[False, True], na_position="last").iloc[0]
    best_sign_row = swe_metrics_df.sort_values(
        ["sign_accuracy", "RMSE"], ascending=[False, True], na_position="last"
    ).iloc[0]
    best_ae_rmse_row = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].sort_values(
        ["RMSE", "r"]
    ).iloc[0]
    best_ae_r_row = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].sort_values(
        ["r", "RMSE"], ascending=[False, True]
    ).iloc[0]
    best_ae_sign_row = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].sort_values(
        ["sign_accuracy", "RMSE"], ascending=[False, True]
    ).iloc[0]
    swe_best_models_df = pd.DataFrame(
        [
            {**swe_metrics_df[swe_metrics_df["model_family"] == "baseline_7col_swe"].iloc[0].to_dict(), "best_model_role": "baseline_7col"},
            {**best_ae_rmse_row.to_dict(), "best_model_role": "best_ae_by_rmse"},
            {**best_ae_r_row.to_dict(), "best_model_role": "best_ae_by_r"},
            {**best_ae_sign_row.to_dict(), "best_model_role": "best_ae_by_sign_accuracy"},
            {**swe_metrics_df[swe_metrics_df["model_family"] == "full_ocean_mode_swe"].iloc[0].to_dict(), "best_model_role": "full_91col_control"},
            {**swe_metrics_df[swe_metrics_df["model_family"] == "pca_swe"].sort_values("RMSE").iloc[0].to_dict(), "best_model_role": "best_pca_control"},
            {**best_rmse_row.to_dict(), "best_model_role": "best_overall_by_rmse"},
            {**best_r_row.to_dict(), "best_model_role": "best_overall_by_r"},
            {**best_sign_row.to_dict(), "best_model_role": "best_overall_by_sign_accuracy"},
        ]
    )

    recon_pred_rows: List[Dict[str, object]] = []
    recon_metric_rows: List[Dict[str, object]] = []

    rows, null_metrics = evaluate_null_reconstruction(years, x_ocean, ocean_features)
    recon_pred_rows.extend(rows)
    recon_metric_rows.append(null_metrics)
    null_mean_mse = float(null_metrics["mean_heldout_reconstruction_mse"])

    baseline_recon_key = SettingKey(
        model_family="baseline_7col_ridge_decoder",
        model_name="BASELINE_7COL_RIDGE_DECODER"
    )
    rows, metrics = evaluate_reconstruction_representation(
        baseline_recon_key,
        years,
        x_ocean,
        lambda heldout_year: (
            x_baseline[years != heldout_year],
            x_baseline[years == heldout_year][0],
            ocean_features,
        ),
    )
    recon_pred_rows.extend(rows)
    recon_metric_rows.append(metrics)

    for k in PCA_COMPONENTS:
        pca_key = SettingKey(model_family="pca_ridge_decoder", model_name=f"PCA{k}_RIDGE_DECODER", k=int(k))
        rows, metrics = evaluate_reconstruction_representation(
            pca_key,
            years,
            x_ocean,
            lambda heldout_year, k_value=k: (*pca_scores_from_outer_fold(
                x_ocean[years != heldout_year], x_ocean[years == heldout_year][0], int(k_value)
            ), ocean_features),
        )
        recon_pred_rows.extend(rows)
        recon_metric_rows.append(metrics)

    for (k, dropout_rate, weight_decay, seed), per_year in latent_lookup.items():
        key = SettingKey(
            model_family="ae_latent_ridge_decoder",
            model_name=f"AE_LATENT_RIDGE_DECODER_k{k}_d{dropout_rate:g}_wd{weight_decay:g}_seed{seed}",
            k=int(k),
            dropout_rate=float(dropout_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )
        rows, metrics = evaluate_reconstruction_representation(
            key,
            years,
            x_ocean,
            lambda heldout_year, lookup=per_year: (
                lookup[int(heldout_year)]["train_z"],
                lookup[int(heldout_year)]["heldout_z"],
                ocean_features,
            ),
        )
        recon_pred_rows.extend(rows)
        recon_metric_rows.append(metrics)

    native_pred_df = native_resid_df.rename(
        columns={
            "heldout_year": "water_year",
            "feature_name": "target_column",
            "x_standardized": "x_obs_standardized",
            "x_reconstructed": "x_pred_standardized",
            "weight_decay": "weight_decay",
        }
    ).copy()
    native_pred_df["model_family"] = "native_ae_decoder_reconstruction"
    native_pred_df["model_name"] = native_pred_df.apply(
        lambda row: "NATIVE_AE_DECODER_k{0}_d{1:g}_wd{2:g}_seed{3}".format(
            int(row["k"]),
            float(row["dropout_rate"]),
            float(row["weight_decay"]),
            int(row["seed"]),
        ),
        axis=1,
    )
    native_pred_df["alpha_selected"] = np.nan
    native_pred_df["inner_cv_mse"] = np.nan
    native_pred_df = native_pred_df[
        [
            "model_name",
            "model_family",
            "water_year",
            "target_column",
            "x_obs_standardized",
            "x_pred_standardized",
            "k",
            "dropout_rate",
            "weight_decay",
            "seed",
            "alpha_selected",
            "inner_cv_mse",
        ]
    ]
    recon_pred_rows.extend(native_pred_df.to_dict(orient="records"))
    native_metrics_df = summarize_native_decoder(ae_fold_df)
    recon_metric_rows.extend(native_metrics_df.to_dict(orient="records"))

    recon_pred_df = pd.DataFrame(recon_pred_rows).sort_values(
        ["model_family", "k", "dropout_rate", "weight_decay", "seed", "water_year", "target_column"],
        na_position="last",
    ).reset_index(drop=True)
    recon_metrics_df = pd.DataFrame(recon_metric_rows).reset_index(drop=True)
    recon_metrics_df["relative_improvement_vs_null"] = 1.0 - (
        recon_metrics_df["mean_heldout_reconstruction_mse"] / null_mean_mse
    )
    recon_metrics_df = recon_metrics_df.sort_values(
        ["model_family", "mean_heldout_reconstruction_mse", "mean_generalization_ratio"],
        na_position="last",
    ).reset_index(drop=True)
    recon_summary_k_df = build_reconstruction_summary_by_k(recon_metrics_df)

    recon_best_models_df = pd.DataFrame(
        [
            {**recon_metrics_df[recon_metrics_df["model_family"] == "null_reconstruction"].iloc[0].to_dict(), "best_model_role": "null_baseline"},
            {**recon_metrics_df[recon_metrics_df["model_family"] == "baseline_7col_ridge_decoder"].iloc[0].to_dict(), "best_model_role": "best_7col_ridge_decoder"},
            {**recon_metrics_df[recon_metrics_df["model_family"] == "ae_latent_ridge_decoder"].sort_values("mean_heldout_reconstruction_mse").iloc[0].to_dict(), "best_model_role": "best_ae_ridge_decoder_by_mse"},
            {**recon_metrics_df[recon_metrics_df["model_family"] == "ae_latent_ridge_decoder"].sort_values("mean_generalization_ratio").iloc[0].to_dict(), "best_model_role": "best_ae_ridge_decoder_by_generalization_ratio"},
            {**recon_metrics_df[recon_metrics_df["model_family"] == "pca_ridge_decoder"].sort_values("mean_heldout_reconstruction_mse").iloc[0].to_dict(), "best_model_role": "best_pca_ridge_decoder"},
            {**recon_metrics_df[recon_metrics_df["model_family"] == "native_ae_decoder_reconstruction"].sort_values("mean_heldout_reconstruction_mse").iloc[0].to_dict(), "best_model_role": "best_native_ae_decoder"},
        ]
    )

    swe_pred_df.to_csv(output_dir / "swe_ridge_loyo_predictions.csv", index=False)
    swe_metrics_df.to_csv(output_dir / "swe_ridge_loyo_metrics.csv", index=False)
    swe_summary_df.to_csv(output_dir / "swe_ridge_loyo_summary_by_setting.csv", index=False)
    swe_best_models_df.to_csv(output_dir / "swe_ridge_best_models.csv", index=False)
    recon_pred_df.to_csv(output_dir / "ridge_decoder_reconstruction_predictions.csv", index=False)
    recon_metrics_df.to_csv(output_dir / "ridge_decoder_reconstruction_metrics.csv", index=False)
    recon_summary_k_df.to_csv(output_dir / "ridge_decoder_reconstruction_summary_by_k.csv", index=False)
    recon_best_models_df.to_csv(output_dir / "ridge_decoder_reconstruction_best_models.csv", index=False)

    plot_swe_rmse_by_k(swe_metrics_df, output_dir / "swe_ridge_ae_vs_7col_rmse_by_k.png")
    plot_swe_r_by_k(swe_metrics_df, output_dir / "swe_ridge_ae_vs_7col_r_by_k.png")
    plot_best_swe_timeseries(swe_pred_df, swe_metrics_df, output_dir / "swe_ridge_best_timeseries.png")
    plot_best_swe_scatter(swe_pred_df, swe_metrics_df, output_dir / "swe_ridge_best_scatter_obs_vs_pred.png")
    plot_recon_mse_by_k(recon_metrics_df, output_dir / "ridge_decoder_reconstruction_mse_by_k_vs_7col.png")
    plot_recon_relative_improvement(
        recon_metrics_df, output_dir / "ridge_decoder_reconstruction_relative_improvement_vs_null.png"
    )
    plot_recon_generalization_ratio(
        recon_metrics_df, output_dir / "ridge_decoder_reconstruction_generalization_ratio_by_k.png"
    )

    baseline_swe = swe_metrics_df[swe_metrics_df["model_family"] == "baseline_7col_swe"].iloc[0]
    best_ae_swe_rmse = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].sort_values("RMSE").iloc[0]
    best_ae_swe_r = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].sort_values(
        ["r", "RMSE"], ascending=[False, True]
    ).iloc[0]
    best_ae_swe_sign = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].sort_values(
        ["sign_accuracy", "RMSE"], ascending=[False, True]
    ).iloc[0]
    swe_by_k_best = swe_metrics_df[swe_metrics_df["model_family"] == "ae_latent_swe"].groupby("k", as_index=False)["RMSE"].min()
    swe_k_improves = bool(np.all(np.diff(swe_by_k_best.sort_values("k")["RMSE"].to_numpy(dtype=np.float64)) <= 1.0e-12))

    baseline_recon = recon_metrics_df[recon_metrics_df["model_family"] == "baseline_7col_ridge_decoder"].iloc[0]
    best_ae_recon = recon_metrics_df[recon_metrics_df["model_family"] == "ae_latent_ridge_decoder"].sort_values(
        "mean_heldout_reconstruction_mse"
    ).iloc[0]
    best_ae_recon_ratio = recon_metrics_df[
        recon_metrics_df["model_family"] == "ae_latent_ridge_decoder"
    ].sort_values("mean_generalization_ratio").iloc[0]
    best_native_recon = recon_metrics_df[
        recon_metrics_df["model_family"] == "native_ae_decoder_reconstruction"
    ].sort_values("mean_heldout_reconstruction_mse").iloc[0]
    recon_by_k_best = recon_metrics_df[
        recon_metrics_df["model_family"] == "ae_latent_ridge_decoder"
    ].groupby("k", as_index=False)["mean_heldout_reconstruction_mse"].min()
    recon_k_improves = bool(
        np.all(np.diff(recon_by_k_best.sort_values("k")["mean_heldout_reconstruction_mse"].to_numpy(dtype=np.float64)) <= 1.0e-12)
    )
    best_recon_setting_matches_native = bool(
        (best_ae_recon["k"] == best_native_recon["k"])
        and np.isclose(best_ae_recon["dropout_rate"], best_native_recon["dropout_rate"])
        and np.isclose(best_ae_recon["weight_decay"], best_native_recon["weight_decay"])
        and np.isclose(best_ae_recon["seed"], best_native_recon["seed"])
    )
    best_recon_k = int(best_ae_recon["k"])
    best_swe_k = int(best_ae_swe_rmse["k"])

    swe_lines = [
        f"The strict LOYO downstream SWE target was taken directly from `{BASELINE_7COL_PATH}` column `obs_swe`, matching the existing 7-column Sierra baseline exactly.",
        "The best AE latent ridge model by RMSE was `{name}` with `k={k}`, `dropout_rate={dropout}`, `weight_decay={wd}`, `seed={seed}`, giving `RMSE={rmse:.6f}` and `r={r:.6f}`.".format(
            name=best_ae_swe_rmse["model_name"],
            k=int(best_ae_swe_rmse["k"]),
            dropout=float(best_ae_swe_rmse["dropout_rate"]),
            wd=float(best_ae_swe_rmse["weight_decay"]),
            seed=int(best_ae_swe_rmse["seed"]),
            rmse=float(best_ae_swe_rmse["RMSE"]),
            r=float(best_ae_swe_rmse["r"]),
        ),
        "The best AE latent ridge model by correlation was `{name}` with `r={r:.6f}`; the best AE model by sign accuracy was `{sign_name}` with sign accuracy `{sign_acc:.6f}`.".format(
            name=best_ae_swe_r["model_name"],
            r=float(best_ae_swe_r["r"]),
            sign_name=best_ae_swe_sign["model_name"],
            sign_acc=float(best_ae_swe_sign["sign_accuracy"]),
        ),
        "Compared with the fixed 7-column baseline (`RMSE={base_rmse:.6f}`, `r={base_r:.6f}`, `sign_accuracy={base_sign:.6f}`), AE latent ridge `{result}` by RMSE.".format(
            base_rmse=float(baseline_swe["RMSE"]),
            base_r=float(baseline_swe["r"]),
            base_sign=float(baseline_swe["sign_accuracy"]),
            result="beat the baseline" if float(best_ae_swe_rmse["RMSE"]) < float(baseline_swe["RMSE"]) - 1.0e-12 else (
                "matched the baseline within tolerance" if abs(float(best_ae_swe_rmse["RMSE"]) - float(baseline_swe["RMSE"])) <= 1.0e-12 else "lost to the baseline"
            ),
        ),
        "The best reconstruction `k` was `{best_recon_k}` while the best SWE-prediction `k` by RMSE was `{best_swe_k}`, so reconstruction quality and SWE prediction skill are `{alignment}`.".format(
            best_recon_k=best_recon_k,
            best_swe_k=best_swe_k,
            alignment="aligned at the same k" if best_recon_k == best_swe_k else "not aligned"
        ),
        "There `{trend}` evidence that larger `k` improves SWE prediction: best-AE RMSE by k `{rmse_by_k}`.".format(
            trend="is" if swe_k_improves else "is not",
            rmse_by_k=", ".join(
                f"k={int(row.k)} -> {float(row.RMSE):.6f}" for row in swe_by_k_best.sort_values("k").itertuples()
            ),
        ),
    ]

    recon_lines = [
        "This section evaluates `representation -> 91-column ocean-mode table` using a ridge decoder fitted inside each outer LOYO fold only. It is separate from the SWE prediction question.",
        "The best AE latent ridge-decoder reconstruction model was `{name}` with mean held-out reconstruction MSE `{mse:.6f}` and mean generalization ratio `{ratio:.6f}`.".format(
            name=best_ae_recon["model_name"],
            mse=float(best_ae_recon["mean_heldout_reconstruction_mse"]),
            ratio=float(best_ae_recon["mean_generalization_ratio"]),
        ),
        "The fixed 7-column ridge decoder had mean held-out reconstruction MSE `{base_mse:.6f}`, while the null mean-reconstruction baseline had `{null_mse:.6f}`. The 7-column baseline `{base_vs_null}` the null baseline, and the best AE latent ridge decoder `{ae_vs_base}` the 7-column ridge decoder.".format(
            base_mse=float(baseline_recon["mean_heldout_reconstruction_mse"]),
            null_mse=float(null_mean_mse),
            base_vs_null="outperformed" if float(baseline_recon["mean_heldout_reconstruction_mse"]) < null_mean_mse - 1.0e-12 else "did not outperform",
            ae_vs_base="outperformed" if float(best_ae_recon["mean_heldout_reconstruction_mse"]) < float(baseline_recon["mean_heldout_reconstruction_mse"]) - 1.0e-12 else "did not outperform",
        ),
        "There `{trend}` evidence that larger `k` improves ridge-decoder reconstruction: best AE reconstruction MSE by k `{mse_by_k}`.".format(
            trend="is" if recon_k_improves else "is not",
            mse_by_k=", ".join(
                f"k={int(row.k)} -> {float(row.mean_heldout_reconstruction_mse):.6f}"
                for row in recon_by_k_best.sort_values("k").itertuples()
            ),
        ),
        "The best native PyOD decoder setting `{native_name}` `{native_match}` the best AE ridge-decoder reconstruction setting.".format(
            native_name=best_native_recon["model_name"],
            native_match="matches" if best_recon_setting_matches_native else "does not match",
        ),
        "The best AE ridge-decoder reconstruction setting and the best AE SWE-prediction setting are `{alignment}`, which reinforces that better ocean-mode reconstruction is not proof of better SWE prediction.".format(
            alignment="the same" if best_recon_k == best_swe_k else "different"
        ),
    ]

    append_or_replace_report_section(output_dir / "REPORT.md", REPORT_TITLE_1, swe_lines)
    append_or_replace_report_section(output_dir / "REPORT.md", REPORT_TITLE_2, recon_lines)

    metadata = {
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "baseline_7col_path": str(BASELINE_7COL_PATH),
        "swe_target_source": str(BASELINE_7COL_PATH),
        "swe_target_column": "obs_swe",
        "ridge_alpha_grid": ALPHA_GRID.tolist(),
        "pca_components": PCA_COMPONENTS,
        "latent_artifact_foldsafe_verified": True,
        "best_ae_swe_by_rmse": best_ae_swe_rmse.to_dict(),
        "best_ae_reconstruction_by_mse": best_ae_recon.to_dict(),
    }
    (output_dir / "posthoc_comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"output directory: {output_dir}")
    print(f"SWE baseline RMSE: {float(baseline_swe['RMSE']):.6f}")
    print(f"Best AE SWE RMSE: {float(best_ae_swe_rmse['RMSE']):.6f} ({best_ae_swe_rmse['model_name']})")
    print(f"Best AE SWE r: {float(best_ae_swe_r['r']):.6f} ({best_ae_swe_r['model_name']})")
    print(
        "Best AE reconstruction MSE: "
        f"{float(best_ae_recon['mean_heldout_reconstruction_mse']):.6f} ({best_ae_recon['model_name']})"
    )
    print(
        "7-column ridge-decoder reconstruction MSE: "
        f"{float(baseline_recon['mean_heldout_reconstruction_mse']):.6f}"
    )
    print(f"Null reconstruction MSE: {null_mean_mse:.6f}")


if __name__ == "__main__":
    main()
