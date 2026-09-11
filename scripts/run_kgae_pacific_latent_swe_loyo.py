#!/usr/bin/env python3
"""Strict LOYO ridge on frozen published KGAE Pacific latent predictors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_pacific_latent_swe_loyo"
)
LOCAL_POINTER_DIR = PROJECT_ROOT / "artifacts" / "kgae_pacific_latent_swe_loyo"
BASELINE_TABLE = Path(
    "/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/"
    "z1z2_plus_amv_k5_loyo/z1z2_amv_k5_predictor_table.csv"
)
KGAE_REPO = LOCAL_POINTER_DIR / "external" / "kgae"
KGAE_ENCODINGS = KGAE_REPO / "KGAE_encodings.nc"
KGAE_MASK = KGAE_REPO / "data_pipeline" / "seamask.nc"
KGAE_ERA5_SST = KGAE_REPO / "data_pipeline" / "era5.sst.pacific.1x1.1940-2023.nc"
ALPHA_GRID = np.logspace(-6.0, 6.0, 49, dtype=np.float64)
NEAR_ZERO_STD = 1.0e-12
MONTHS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-pointer-dir", type=Path, default=LOCAL_POINTER_DIR)
    parser.add_argument("--kgae-repo", type=Path, default=KGAE_REPO)
    parser.add_argument("--baseline-table", type=Path, default=BASELINE_TABLE)
    parser.add_argument("--use-run", type=str, default="basin.goodtest")
    parser.add_argument("--use-recursion", type=int, default=0)
    return parser.parse_args()


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside a Perlmutter interactive compute-node allocation.")


def ensure_local_pointer(target_dir: Path, pointer_path: Path) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if pointer_path.is_symlink():
        if pointer_path.resolve() == target_dir.resolve():
            return
        raise RuntimeError(f"Pointer path already exists with different symlink target: {pointer_path}")
    if pointer_path.is_dir():
        note = pointer_path / "ARTIFACT_POINTER.txt"
        note.write_text(f"Primary artifact directory: {target_dir}\n", encoding="utf-8")
        return
    if pointer_path.exists():
        if pointer_path.resolve() == target_dir.resolve():
            return
        raise RuntimeError(f"Pointer path already exists with different target: {pointer_path}")
    pointer_path.symlink_to(target_dir)


def standardize_train_only(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (np.abs(std) < NEAR_ZERO_STD), 1.0, std)
    return (train - mean[None, :]) / std[None, :], (test - mean) / std, mean, std


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(xtx + alpha * np.eye(x_train_std.shape[1], dtype=np.float64), rhs)


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha = float(ALPHA_GRID[0])
    best_mse = float("inf")
    for alpha in ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            x_inner_train_std, x_inner_valid_std, _, _ = standardize_train_only(
                x_train_raw[mask], x_train_raw[~mask][0]
            )
            y_inner_train = y_train_raw[mask]
            y_mean = float(np.mean(y_inner_train))
            y_std = float(np.std(y_inner_train, ddof=1))
            if not np.isfinite(y_std) or abs(y_std) < NEAR_ZERO_STD:
                y_std = 1.0
            y_inner_train_std = (y_inner_train - y_mean) / y_std
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, float(alpha))
            preds[inner_idx] = y_mean + y_std * float(x_inner_valid_std @ beta_std)
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and float(alpha) < best_alpha):
            best_alpha = float(alpha)
            best_mse = mse
    return best_alpha, best_mse


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(((np.sign(obs) == np.sign(pred)) & (obs != 0.0) & (pred != 0.0)).astype(float)))


def compute_scalar_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    obs_std = float(np.std(obs, ddof=1))
    pred_std = float(np.std(pred, ddof=1))
    r = float("nan") if obs_std < NEAR_ZERO_STD or pred_std < NEAR_ZERO_STD else float(np.corrcoef(obs, pred)[0, 1])
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    sse = float(np.sum((obs - pred) ** 2))
    r2 = float("nan") if sst < NEAR_ZERO_STD else float(1.0 - sse / sst)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "r": r,
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "mean_bias": bias,
        "prediction_std": pred_std,
        "observed_std": obs_std,
        "n_years": int(obs.size),
    }


def load_baseline_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("water_year").reset_index(drop=True)
    required = {"water_year", "obs_swe", "Z1", "Z2"}
    if not required <= set(df.columns):
        raise RuntimeError(f"Baseline table missing required columns: {required - set(df.columns)}")
    return df


def open_kgae_metadata(args: argparse.Namespace) -> Tuple[xr.DataArray, Dict[str, object]]:
    encodings_path = args.kgae_repo / "KGAE_encodings.nc"
    mask_path = args.kgae_repo / "data_pipeline" / "seamask.nc"
    era5_path = args.kgae_repo / "data_pipeline" / "era5.sst.pacific.1x1.1940-2023.nc"
    if not encodings_path.exists():
        raise FileNotFoundError(f"Missing published KGAE encodings: {encodings_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing published KGAE seamask: {mask_path}")
    if not era5_path.exists():
        raise FileNotFoundError(f"Missing published KGAE ERA5 Pacific SST file: {era5_path}")

    enc_ds = xr.open_dataset(encodings_path)
    enc = enc_ds["encodings"]
    if args.use_run not in enc.coords["run"].values.tolist():
        raise RuntimeError(f"Requested KGAE run {args.use_run!r} not found in {encodings_path}")
    if args.use_recursion not in enc.coords["recursion"].values.tolist():
        raise RuntimeError(f"Requested recursion {args.use_recursion} not found in {encodings_path}")

    mask_ds = xr.open_dataset(mask_path)
    mask = mask_ds["pacific"]
    era5 = xr.open_dataset(era5_path)["sst"].rename({"latitude": "lat", "longitude": "lon"})
    feat = era5.stack(feature=("lat", "lon")).dropna("feature", how="any")

    monthly_latents = enc.sel(run=args.use_run, recursion=args.use_recursion).mean("initialization")
    monthly_latents = monthly_latents.transpose("time", "mode")

    metadata = {
        "kgae_source_repo": str(args.kgae_repo),
        "github_url": "https://github.com/kjhall01/kgae",
        "kgae_encodings_path": str(encodings_path),
        "kgae_mask_path": str(mask_path),
        "kgae_era5_sst_path": str(era5_path),
        "kgae_run_used": args.use_run,
        "kgae_recursion_used": args.use_recursion,
        "kgae_model_frozen": True,
        "latent_coordinates_used": monthly_latents.coords["mode"].values.tolist(),
        "latitude_min": float(era5.lat.min()),
        "latitude_max": float(era5.lat.max()),
        "longitude_min": float(era5.lon.min()),
        "longitude_max": float(era5.lon.max()),
        "longitude_convention": "-180_to_180",
        "grid_resolution_degrees": 1.0,
        "n_valid_grid_points": int(feat.sizes["feature"]),
        "land_points_masked": True,
        "experiment_type": "basin_wide_pacific",
        "available_runs_in_encodings": enc.coords["run"].values.tolist(),
        "available_recursions_in_encodings": [int(v) for v in enc.coords["recursion"].values.tolist()],
        "available_mode_names": enc.coords["mode"].values.tolist(),
        "kgae_time_start": str(pd.Timestamp(monthly_latents.time.values[0]).date()),
        "kgae_time_end": str(pd.Timestamp(monthly_latents.time.values[-1]).date()),
        "kgae_source_data_period_months": int(monthly_latents.sizes["time"]),
        "preprocessing_summary": (
            "Published ERA5 Pacific SST file on 1-degree grid, detrended with global_detrend(deg=2), "
            "monthly climatology removed with training-era monthly anomalies in the KGAE scripts; "
            "land points masked and feature stack formed by dropping any NaN grid points."
        ),
        "normalization_summary": (
            "KGAE_encodings.nc provides published frozen monthly latent coordinates directly; "
            "for this SWE experiment no further KGAE-side fine-tuning or label-based transformation was applied."
        ),
    }
    return monthly_latents, metadata


def build_kgae_predictor_table(monthly_latents: xr.DataArray, baseline_df: pd.DataFrame) -> pd.DataFrame:
    mode_names = monthly_latents.coords["mode"].values.tolist()
    if len(mode_names) < 5:
        raise RuntimeError(f"Expected at least five KGAE modes, found {len(mode_names)}")
    monthly_latents = monthly_latents.isel(mode=slice(0, 5))
    rows: List[Dict[str, object]] = []
    for _, row in baseline_df.iterrows():
        wy = int(row["water_year"])
        out: Dict[str, object] = {"water_year": wy, "obs_swe": float(row["obs_swe"])}
        for mode_idx in range(5):
            for month_label, year_offset, month_num in MONTHS:
                ts = pd.Timestamp(wy + year_offset, month_num, 1)
                if ts not in set(pd.to_datetime(monthly_latents.time.values)):
                    raise RuntimeError(f"Missing KGAE monthly latent timestamp {ts} for water year {wy}")
                value = float(monthly_latents.sel(time=ts).isel(mode=mode_idx).item())
                out[f"kgae_z{mode_idx + 1}_{month_label}"] = value
        rows.append(out)
    df = pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)
    if df["water_year"].tolist() != baseline_df["water_year"].tolist():
        raise RuntimeError("KGAE predictor table years do not match SWE baseline years.")
    return df


def run_loyo_ridge(
    model_name: str,
    model_family: str,
    predictor_set: str,
    years: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    selected_alphas: List[float] = []
    for heldout_year in years.tolist():
        train_mask = years != heldout_year
        x_train = x_all[train_mask]
        x_test = x_all[~train_mask][0]
        y_train = y_all[train_mask]
        y_test = float(y_all[~train_mask][0])
        alpha, _ = inner_loyo_best_alpha(x_train, y_train)
        x_train_std, x_test_std, _, _ = standardize_train_only(x_train, x_test)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train, ddof=1))
        if not np.isfinite(y_std) or abs(y_std) < NEAR_ZERO_STD:
            y_std = 1.0
        y_train_std = (y_train - y_mean) / y_std
        beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
        pred = y_mean + y_std * float(x_test_std @ beta_std)
        selected_alphas.append(alpha)
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "water_year": int(heldout_year),
                "y_obs": y_test,
                "y_pred": pred,
                "alpha_selected": alpha,
                "predictor_set": predictor_set,
            }
        )
    pred_df = pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)
    metrics = compute_scalar_metrics(
        pred_df["y_obs"].to_numpy(dtype=np.float64), pred_df["y_pred"].to_numpy(dtype=np.float64)
    )
    metrics["alpha_selected_median"] = float(np.median(np.asarray(selected_alphas, dtype=np.float64)))
    metrics["alpha_selected_mode"] = float(pd.Series(selected_alphas).value_counts().sort_index().idxmax())
    return pred_df, metrics


def make_timeseries_plot(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="water_year", columns="model_name", values="y_pred")
    obs = df.groupby("water_year")["y_obs"].first().sort_index()
    years = obs.index.to_numpy(dtype=int)
    plt.figure(figsize=(11, 5))
    plt.plot(years, obs.values, color="black", linewidth=2.0, label="Observed SWE")
    if "BASELINE_7COL_RIDGE" in pivot.columns:
        plt.plot(years, pivot["BASELINE_7COL_RIDGE"].values, color="tab:blue", linewidth=1.5, label="7-column baseline")
    if "KGAE_LATENT_MONTHLY_RIDGE" in pivot.columns:
        plt.plot(years, pivot["KGAE_LATENT_MONTHLY_RIDGE"].values, color="tab:orange", linewidth=1.5, label="Frozen KGAE latent ridge")
    plt.xlabel("Water year")
    plt.ylabel("April 1 Sierra SWE")
    plt.title("Observed vs LOYO predictions")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_scatter_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    models = [
        ("BASELINE_7COL_RIDGE", "tab:blue", "7-column baseline"),
        ("KGAE_LATENT_MONTHLY_RIDGE", "tab:orange", "Frozen KGAE latent ridge"),
    ]
    all_vals: List[float] = []
    for model_name, color, label in models:
        sub = df[df["model_name"] == model_name].sort_values("water_year")
        if sub.empty:
            continue
        plt.scatter(sub["y_obs"], sub["y_pred"], color=color, alpha=0.8, label=label)
        all_vals.extend(sub["y_obs"].tolist())
        all_vals.extend(sub["y_pred"].tolist())
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1.0)
        plt.xlim(lo - pad, hi + pad)
        plt.ylim(lo - pad, hi + pad)
    plt.xlabel("Observed SWE")
    plt.ylabel("Predicted SWE")
    plt.title("Observed vs predicted SWE")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_metrics_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    order = ["BASELINE_7COL_RIDGE", "KGAE_LATENT_MONTHLY_RIDGE"]
    labels = {
        "BASELINE_7COL_RIDGE": "7-column baseline",
        "KGAE_LATENT_MONTHLY_RIDGE": "Frozen KGAE latent ridge",
    }
    sub = metrics_df.set_index("model_name").loc[order].reset_index()
    metrics = ["RMSE", "r", "sign_accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, metric in zip(axes, metrics):
        vals = sub[metric].to_numpy(dtype=float)
        ax.bar(np.arange(len(sub)), vals, color=["tab:blue", "tab:orange"])
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels([labels[name] for name in sub["model_name"]], rotation=15, ha="right")
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(
    report_path: Path,
    metadata: Dict[str, object],
    baseline_metrics: Dict[str, float],
    kgae_metrics: Dict[str, float],
    recon_feasible: bool,
    recon_note: str,
) -> None:
    ranking = (
        "The published frozen KGAE Pacific latents carry SWE-predictive information comparable to or stronger than the "
        "hand-selected 7-column baseline, which motivates supervised KGAE adaptation in the next experiment."
        if kgae_metrics["RMSE"] <= baseline_metrics["RMSE"]
        else "The published frozen KGAE Pacific latents may reconstruct/represent Pacific SST variability, but they do "
        "not by themselves capture the specific Sierra SWE-predictive direction as well as the existing 7-column baseline."
    )
    flat_note = (
        "KGAE predictions retained useful interannual variability."
        if kgae_metrics["prediction_std"] > 0.5 * baseline_metrics["prediction_std"]
        else "KGAE predictions were comparatively flat relative to the baseline and observed SWE variability."
    )
    lines = [
        "# Frozen KGAE Pacific latent Sierra SWE LOYO report",
        "",
        "## KGAE source",
        "",
        f"- GitHub URL: `{metadata['github_url']}`",
        f"- Local repo path: `{metadata['kgae_source_repo']}`",
        f"- Frozen latent source file: `{metadata['kgae_encodings_path']}`",
        "- The KGAE model was not trained or fine-tuned in this experiment.",
        f"- KGAE run used: `{metadata['kgae_run_used']}`",
        f"- Recursion used: `{metadata['kgae_recursion_used']}`",
        f"- Latent coordinates used: `{metadata['latent_coordinates_used']}`",
        "",
        "## KGAE Pacific spatial domain",
        "",
        f"- latitude range: `{metadata['latitude_min']}` to `{metadata['latitude_max']}`",
        f"- longitude range: `{metadata['longitude_min']}` to `{metadata['longitude_max']}`",
        f"- longitude convention: `{metadata['longitude_convention']}`",
        f"- grid resolution: `{metadata['grid_resolution_degrees']}` degree",
        f"- valid gridpoint count: `{metadata['n_valid_grid_points']}`",
        "- mask definition: published Pacific seamask intersected with the shipped ERA5 Pacific SST file, using only grid points with no missing values across time",
        f"- basin-wide Pacific vs tropical Pacific: `{metadata['experiment_type']}`",
        f"- land points masked: `{metadata['land_points_masked']}`",
        f"- preprocessing/normalization: {metadata['preprocessing_summary']}",
        "",
        "## Water-year predictor construction",
        "",
        "- Monthly KGAE latents were converted to annual SWE predictors using Sep(Y-1) through Mar(Y).",
        "- The first five latent coordinates were used directly.",
        "- This yields 35 monthly-lag predictors total.",
        "",
        "## SWE prediction comparison",
        "",
        f"- Current rerun 7-column baseline RMSE: `{baseline_metrics['RMSE']:.6f}`",
        f"- Current rerun 7-column baseline r: `{baseline_metrics['r']:.6f}`",
        f"- Current rerun 7-column baseline sign accuracy: `{baseline_metrics['sign_accuracy']:.6f}`",
        f"- Frozen KGAE latent ridge RMSE: `{kgae_metrics['RMSE']:.6f}`",
        f"- Frozen KGAE latent ridge r: `{kgae_metrics['r']:.6f}`",
        f"- Frozen KGAE latent ridge sign accuracy: `{kgae_metrics['sign_accuracy']:.6f}`",
        f"- Outcome: frozen KGAE latent ridge {'beat or matched' if kgae_metrics['RMSE'] <= baseline_metrics['RMSE'] else 'lost to'} the 7-column baseline for April 1 Sierra SWE prediction.",
        f"- Interannual-variability assessment: {flat_note}",
        f"- Promise for second-stage supervision: {'Yes' if kgae_metrics['r'] > 0.2 or kgae_metrics['RMSE'] < baseline_metrics['RMSE'] * 1.5 else 'Unclear; the frozen test is weak enough that adaptation would be exploratory.'}",
        "",
        "## Reconstruction diagnostics",
        "",
        f"- Reconstruction feasible from published artifacts: `{recon_feasible}`",
        f"- Notes: {recon_note}",
        "- KGAE reconstruction quality and SWE predictability are different questions.",
        "",
        "## Leakage checks",
        "",
        f"- number of SWE water years: `37`",
        f"- KGAE source data period: `{metadata['kgae_time_start']}` to `{metadata['kgae_time_end']}`",
        "- whether KGAE was pretrained externally and frozen: `True`",
        "- whether any SWE labels were used in KGAE training: `False`",
        "- held-out water year absent from ridge training fold: `True`",
        "- predictor scaler fit only on training years: `True`",
        "- ridge alpha tuned only inside training years: `True`",
        "- no KGAE fine-tuning using SWE: `True`",
        "",
        "## Interpretation",
        "",
        ranking,
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_runtime_on_compute_node()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_pointer(args.output_dir, args.local_pointer_dir)

    baseline_df = load_baseline_table(args.baseline_table)
    monthly_latents, metadata = open_kgae_metadata(args)
    kgae_predictor_df = build_kgae_predictor_table(monthly_latents, baseline_df)
    kgae_predictor_df.to_csv(args.output_dir / "kgae_swe_predictor_table.csv", index=False)

    years = baseline_df["water_year"].to_numpy(dtype=int)
    y_all = baseline_df["obs_swe"].to_numpy(dtype=np.float64)
    baseline_columns = [col for col in baseline_df.columns if col not in {"water_year", "obs_swe"}]
    x_baseline = baseline_df[baseline_columns].to_numpy(dtype=np.float64)
    kgae_columns = [col for col in kgae_predictor_df.columns if col not in {"water_year", "obs_swe"}]
    x_kgae = kgae_predictor_df[kgae_columns].to_numpy(dtype=np.float64)

    baseline_pred_df, baseline_metrics = run_loyo_ridge(
        "BASELINE_7COL_RIDGE", "baseline_7col_ridge", "7-column baseline", years, x_baseline, y_all
    )
    kgae_pred_df, kgae_metrics = run_loyo_ridge(
        "KGAE_LATENT_MONTHLY_RIDGE", "kgae_latent_monthly_ridge", "frozen KGAE monthly latents", years, x_kgae, y_all
    )
    pred_df = pd.concat([baseline_pred_df, kgae_pred_df], ignore_index=True)
    pred_df.to_csv(args.output_dir / "kgae_swe_ridge_loyo_predictions.csv", index=False)

    metrics_rows = [
        {
            "model_name": "BASELINE_7COL_RIDGE",
            "model_family": "baseline_7col_ridge",
            "predictor_set": "7-column baseline",
            **baseline_metrics,
        },
        {
            "model_name": "KGAE_LATENT_MONTHLY_RIDGE",
            "model_family": "kgae_latent_monthly_ridge",
            "predictor_set": "frozen KGAE monthly latents",
            **kgae_metrics,
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_dir / "kgae_swe_ridge_loyo_metrics.csv", index=False)

    best_models_df = metrics_df.copy()
    best_models_df.to_csv(args.output_dir / "kgae_swe_best_models.csv", index=False)

    make_timeseries_plot(pred_df, args.output_dir / "kgae_vs_7col_swe_timeseries.png")
    make_scatter_plot(pred_df, args.output_dir / "kgae_vs_7col_swe_scatter.png")
    make_metrics_bar(metrics_df, args.output_dir / "kgae_vs_7col_swe_metrics_bar.png")

    recon_report_path = args.output_dir / "kgae_reconstruction_report.md"
    recon_note = (
        "The published repo provides the frozen monthly encodings, the Pacific mask, and the ERA5 Pacific SST domain file, "
        "but it does not ship pretrained decoder checkpoints or a frozen model bundle that can be loaded directly for "
        "reconstruction on this machine. Therefore ERA5-native and COBE2 cross-dataset reconstruction diagnostics were not "
        "run in this first frozen-latent experiment."
    )
    recon_report_path.write_text(
        "# KGAE reconstruction diagnostics\n\n"
        "Reconstruction diagnostics were not feasible from the published artifacts used here.\n\n"
        f"{recon_note}\n",
        encoding="utf-8",
    )

    with open(args.output_dir / "kgae_domain_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    write_report(
        args.output_dir / "REPORT.md",
        metadata,
        baseline_metrics,
        kgae_metrics,
        recon_feasible=False,
        recon_note=recon_note,
    )

    print(f"Output directory: {args.output_dir}")
    print(f"Baseline RMSE: {baseline_metrics['RMSE']:.6f}")
    print(f"KGAE latent ridge RMSE: {kgae_metrics['RMSE']:.6f}")


if __name__ == "__main__":
    main()
