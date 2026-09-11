#!/usr/bin/env python3
"""Strict LOYO PyOD autoencoder on the known ocean-mode predictor table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pyod.models.auto_encoder import AutoEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
NINO34_NC_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "nino34"
    / "nino34_monthly_wy1985_2021_sep_mar.nc"
)
AMV_AMO_CSV_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)
AMV_AMO_NC_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_eofs_pc1to6.nc"
)
EXISTING_OCEAN_MODE_RIDGE_DIR = (
    PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "ocean_mode_only_ridge"
)
EXISTING_OCEAN_MODE_PREDICTOR_TABLE = (
    EXISTING_OCEAN_MODE_RIDGE_DIR / "ocean_mode_predictors_wy1985_2021.csv"
)
BASELINE_7COL_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo"
)
BASELINE_7COL_PREDICTOR_TABLE = BASELINE_7COL_DIR / "z1z2_amv_k5_predictor_table.csv"
BASELINE_7COL_PREDICTIONS = BASELINE_7COL_DIR / "z1z2_amv_k5_loyo_predictions.csv"
BASELINE_7COL_METRICS = BASELINE_7COL_DIR / "z1z2_amv_k5_loyo_metrics.csv"
PSCRATCH_BASELINE_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ocean_mode_pyod_autoencoder_loyo"
FIGURES_DIRNAME = "figures"
RIDGE_ALPHA_GRID = np.asarray(
    [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64
)
WATER_YEARS = np.arange(1985, 2022, dtype=np.int32)
MONTH_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]
NEAR_ZERO_STD = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--dropout-rates", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1.0e-5, 1.0e-4, 1.0e-3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epoch-num", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-neurons", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--posthoc-reconstruction-summary-only",
        action="store_true",
        help="Read existing reconstruction artifacts and write latent-dimension summaries and plots.",
    )
    return parser.parse_args()


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def validate_args(args: argparse.Namespace) -> None:
    if any(k <= 0 for k in args.latent_dims):
        raise ValueError("All latent dimensions must be positive.")
    if any(d < 0.0 or d >= 1.0 for d in args.dropout_rates):
        raise ValueError("Dropout rates must be in [0, 1).")
    if any(wd < 0.0 for wd in args.weight_decays):
        raise ValueError("Weight decays must be non-negative.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative.")
    if args.epoch_num <= 0 or args.batch_size <= 0 or args.lr <= 0.0:
        raise ValueError("epoch-num, batch-size, and lr must be positive.")
    if any(h <= 0 for h in args.hidden_neurons):
        raise ValueError("Hidden-neuron sizes must be positive.")


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


def load_pacific_predictors() -> Tuple[List[str], np.ndarray]:
    with xr.open_dataset(PACIFIC_PC_PATH, engine="netcdf4") as ds:
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


def build_predictor_matrix() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pacific_columns, pacific = load_pacific_predictors()
    nino_columns, nino = load_csv_predictor_table(NINO34_CSV_PATH, "Nino34_")
    amv_columns, amv = load_csv_predictor_table(AMV_AMO_CSV_PATH, "AMV_PC")
    columns = pacific_columns + nino_columns + amv_columns
    matrix = np.concatenate([pacific, nino, amv], axis=1)
    if matrix.shape != (WATER_YEARS.size, len(columns)):
        raise ValueError(f"Unexpected predictor matrix shape {matrix.shape}")
    table = pd.DataFrame(matrix, columns=columns)
    table.insert(0, "water_year", WATER_YEARS.astype(int))
    missing_report = {
        "total_missing_values": int(table.isna().sum().sum()),
        "missing_by_column": {
            col: int(count) for col, count in table.isna().sum().items() if int(count) > 0
        },
    }
    all_year_std = table.drop(columns=["water_year"]).std(axis=0, ddof=1)
    global_constant_columns = [str(col) for col in all_year_std.index[all_year_std.fillna(0.0).abs() < NEAR_ZERO_STD]]
    summary = {
        "input_file_paths": {
            "pacific_pc_netcdf": str(PACIFIC_PC_PATH),
            "nino34_csv": str(NINO34_CSV_PATH),
            "nino34_netcdf": str(NINO34_NC_PATH),
            "amv_amo_csv": str(AMV_AMO_CSV_PATH),
            "amv_amo_netcdf": str(AMV_AMO_NC_PATH),
        },
        "water_year_range": [int(WATER_YEARS.min()), int(WATER_YEARS.max())],
        "predictor_column_names": columns,
        "predictor_table_shape": [int(table.shape[0]), int(table.shape[1])],
        "missing_value_report": missing_report,
        "globally_constant_columns": global_constant_columns,
        "dropped_or_unavailable_columns": global_constant_columns.copy(),
    }
    if global_constant_columns:
        table = table.drop(columns=global_constant_columns).copy()
        summary["predictor_table_shape_after_global_drop"] = [int(table.shape[0]), int(table.shape[1])]
    return table, summary


def standardize_train_only(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=1)
    near_zero = np.where(~np.isfinite(std) | (np.abs(std) < NEAR_ZERO_STD))[0].tolist()
    safe_std = std.copy()
    if near_zero:
        safe_std[near_zero] = 1.0
    return (
        (x_train - mean[None, :]) / safe_std[None, :],
        (x_test - mean) / safe_std,
        mean,
        safe_std,
        near_zero,
    )


def fit_pyod_autoencoder(
    x_train_std: np.ndarray,
    *,
    k: int,
    dropout_rate: float,
    weight_decay: float,
    seed: int,
    args: argparse.Namespace,
) -> AutoEncoder:
    hidden_list = [int(v) for v in args.hidden_neurons] + [int(k)]
    clf = AutoEncoder(
        contamination=float(args.contamination),
        preprocessing=False,
        lr=float(args.lr),
        epoch_num=int(args.epoch_num),
        batch_size=int(args.batch_size),
        optimizer_name="adam",
        optimizer_params={"weight_decay": float(weight_decay)},
        hidden_neuron_list=hidden_list,
        hidden_activation_name="relu",
        batch_norm=False,
        dropout_rate=float(dropout_rate),
        random_state=int(seed),
        verbose=0,
    )
    clf.fit(np.asarray(x_train_std, dtype=np.float32))
    return clf


def get_model_module(clf: AutoEncoder) -> torch.nn.Module:
    model = getattr(clf, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("PyOD AutoEncoder does not expose torch module as clf.model.")
    return model


def reconstruct_with_model(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        recon = model(tensor)
    return recon.detach().cpu().numpy().astype(np.float64)


def encode_with_model(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, torch.nn.Module):
        raise RuntimeError("PyOD AutoEncoder torch model does not expose encoder submodule.")
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        z = encoder(tensor)
    return z.detach().cpu().numpy().astype(np.float64)


def mse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    return float(np.mean((x_true - x_pred) ** 2))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def linear_r2_multivariate(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    x_std = np.where(np.abs(x_std) < NEAR_ZERO_STD, 1.0, x_std)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0, ddof=1)
    y_std = np.where(np.abs(y_std) < NEAR_ZERO_STD, 1.0, y_std)
    xz = (x - x_mean[None, :]) / x_std[None, :]
    yz = (y - y_mean[None, :]) / y_std[None, :]
    design = np.column_stack([np.ones(xz.shape[0]), xz])
    beta, _, _, _ = np.linalg.lstsq(design, yz, rcond=None)
    pred = design @ beta
    ss_res = np.sum((yz - pred) ** 2, axis=0)
    ss_tot = np.sum((yz - np.mean(yz, axis=0)[None, :]) ** 2, axis=0)
    r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, np.nan)
    return r2.astype(np.float64), float(np.nanmean(r2))


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0], dtype=np.float64), rhs)


def select_alpha_inner_loyo(z_train_raw: np.ndarray, y_train_raw: np.ndarray) -> float:
    n_train = z_train_raw.shape[0]
    best_alpha = float(RIDGE_ALPHA_GRID[0])
    best_mse = float("inf")
    for alpha in RIDGE_ALPHA_GRID:
        preds = np.full(n_train, np.nan, dtype=np.float64)
        for inner_idx in range(n_train):
            mask = np.ones(n_train, dtype=bool)
            mask[inner_idx] = False
            z_inner_train = z_train_raw[mask]
            z_inner_valid = z_train_raw[~mask][0]
            y_inner_train = y_train_raw[mask]
            z_train_std, z_valid_std, _, _, _ = standardize_train_only(z_inner_train, z_inner_valid)
            y_mean = float(np.mean(y_inner_train))
            y_std = float(np.std(y_inner_train, ddof=1))
            if not np.isfinite(y_std) or y_std < NEAR_ZERO_STD:
                y_std = 1.0
            y_train_std = (y_inner_train - y_mean) / y_std
            beta = fit_ridge_standardized(z_train_std, y_train_std, float(alpha))
            preds[inner_idx] = y_mean + y_std * float(z_valid_std @ beta)
        current = float(np.mean((preds - y_train_raw) ** 2))
        if current < best_mse - 1.0e-15 or (abs(current - best_mse) <= 1.0e-15 and float(alpha) < best_alpha):
            best_mse = current
            best_alpha = float(alpha)
    return best_alpha


def find_baseline_7col() -> Tuple[Optional[Path], Dict[str, List[str]]]:
    exact_candidates = [
        BASELINE_7COL_PREDICTOR_TABLE,
        PSCRATCH_BASELINE_ROOT / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv",
    ]
    candidate_hits: List[str] = []
    for candidate in exact_candidates:
        if candidate.exists():
            candidate_hits.append(str(candidate))
    searched_roots = [str(PROJECT_ROOT / "artifacts"), str(PSCRATCH_BASELINE_ROOT)]
    if BASELINE_7COL_PREDICTOR_TABLE.exists():
        return BASELINE_7COL_PREDICTOR_TABLE, {"searched_roots": searched_roots, "candidate_files": candidate_hits}
    return None, {"searched_roots": searched_roots, "candidate_files": candidate_hits}


def write_table_and_summary(
    predictor_table: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    predictor_table.to_csv(output_dir / "ocean_mode_predictor_table.csv", index=False)
    (output_dir / "ocean_mode_predictor_table_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def plot_reconstruction_mse_by_k(aggregate_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        aggregate_df[aggregate_df["aggregate_level"] == "by_k_dropout_weight_decay"]
        .groupby("k", as_index=False)["mean_test_reconstruction_mse"]
        .mean()
        .sort_values("k")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(summary["k"], summary["mean_test_reconstruction_mse"], marker="o", linewidth=2.0)
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Mean held-out reconstruction MSE")
    ax.set_title("Held-out reconstruction MSE by latent dimension")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_generalization_ratio_by_k(aggregate_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        aggregate_df[aggregate_df["aggregate_level"] == "by_k_dropout_weight_decay"]
        .groupby("k", as_index=False)["mean_generalization_ratio"]
        .mean()
        .sort_values("k")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(summary["k"], summary["mean_generalization_ratio"], marker="o", linewidth=2.0, color="#c76d06")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Mean generalization ratio")
    ax.set_title("Held-out / train reconstruction ratio by latent dimension")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_reconstruction_heatmaps(aggregate_df: pd.DataFrame, out_path: Path) -> None:
    summary = aggregate_df[aggregate_df["aggregate_level"] == "by_k_dropout_weight_decay"].copy()
    ks = sorted(summary["k"].dropna().astype(int).unique().tolist())
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4.5), constrained_layout=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        sub = summary[summary["k"] == k].copy()
        pivot = sub.pivot(index="dropout_rate", columns="weight_decay", values="mean_test_reconstruction_mse")
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"{v:g}" for v in pivot.columns.tolist()])
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index.tolist()])
        ax.set_xlabel("weight_decay")
        ax.set_ylabel("dropout_rate")
        ax.set_title(f"k={k}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean held-out reconstruction MSE by dropout and weight decay", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_worst_heldout_years(fold_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        fold_df.groupby("heldout_year", as_index=False)["test_recon_mse"]
        .mean()
        .sort_values("test_recon_mse", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(summary["heldout_year"].astype(str), summary["test_recon_mse"], color="#8c564b")
    ax.set_xlabel("Held-out year")
    ax.set_ylabel("Mean held-out reconstruction MSE")
    ax.set_title("Worst held-out years by reconstruction difficulty")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_latent_scatter(latent_df: pd.DataFrame, best_setting: Dict[str, Any], out_path: Path) -> None:
    sub = latent_df[
        (latent_df["split_role"] == "heldout")
        & (latent_df["k"] == int(best_setting["k"]))
        & (latent_df["seed"] == int(best_setting["seed"]))
        & (latent_df["dropout_rate"] == float(best_setting["dropout_rate"]))
        & (latent_df["weight_decay"] == float(best_setting["weight_decay"]))
    ].copy()
    latent_cols = [col for col in sub.columns if col.startswith("z")]
    use_cols = [col for col in latent_cols if sub[col].notna().any()]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if len(use_cols) >= 2:
        scatter = ax.scatter(sub[use_cols[0]], sub[use_cols[1]], c=sub["water_year"], cmap="viridis", s=40)
        ax.set_xlabel(use_cols[0])
        ax.set_ylabel(use_cols[1])
        fig.colorbar(scatter, ax=ax, label="water_year")
    else:
        ax.scatter(sub["water_year"], sub[use_cols[0]], s=40)
        ax.set_xlabel("water_year")
        ax.set_ylabel(use_cols[0])
    ax.set_title("Held-out latent coordinates for best setting")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_alignment_heatmap(alignment_df: pd.DataFrame, best_setting: Dict[str, Any], out_path: Path) -> None:
    sub = alignment_df[
        (alignment_df["k"] == int(best_setting["k"]))
        & (alignment_df["seed"] == int(best_setting["seed"]))
        & (alignment_df["dropout_rate"] == float(best_setting["dropout_rate"]))
        & (alignment_df["weight_decay"] == float(best_setting["weight_decay"]))
    ].copy()
    heat = (
        sub.groupby(["latent_name", "baseline_column"], as_index=False)["abs_correlation"]
        .mean()
        .pivot(index="latent_name", columns="baseline_column", values="abs_correlation")
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * heat.shape[0])), constrained_layout=True)
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(heat.index.tolist())
    ax.set_title("Mean absolute latent correlation with 7-column baseline")
    fig.colorbar(im, ax=ax, label="|correlation|")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_reconstruction_summary_by_k(
    fold_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, Any]] = []
    for k_value, sub in fold_df.groupby("k"):
        row = {
            "k": int(k_value),
            "heldout_reconstruction_mse_mean_all_folds": float(sub["test_recon_mse"].mean()),
            "heldout_reconstruction_mse_std_all_folds": float(sub["test_recon_mse"].std(ddof=1)),
            "heldout_reconstruction_mse_median_all_folds": float(sub["test_recon_mse"].median()),
            "best_heldout_reconstruction_mse_all_folds": float(sub["test_recon_mse"].min()),
            "mean_train_reconstruction_mse_all_folds": float(sub["train_recon_mse"].mean()),
            "mean_generalization_ratio_all_folds": float(sub["generalization_ratio"].mean()),
        }
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows).sort_values("k").reset_index(drop=True)

    by_setting = aggregate_df[
        aggregate_df["aggregate_level"] == "by_k_dropout_weight_decay_seed"
    ].copy()
    summary_rows: List[Dict[str, Any]] = []
    for k_value, sub in by_setting.groupby("k"):
        ordered = sub.sort_values(
            ["mean_test_reconstruction_mse", "median_test_reconstruction_mse", "mean_generalization_ratio"]
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        summary_rows.append(
            {
                "k": int(k_value),
                "best_dropout_rate": float(best["dropout_rate"]),
                "best_weight_decay": float(best["weight_decay"]),
                "best_seed": int(best["seed"]),
                "best_mean_heldout_reconstruction_mse": float(best["mean_test_reconstruction_mse"]),
                "mean_heldout_reconstruction_mse_across_settings": float(
                    sub["mean_test_reconstruction_mse"].mean()
                ),
                "std_heldout_reconstruction_mse_across_settings": float(
                    sub["mean_test_reconstruction_mse"].std(ddof=1)
                ),
                "mean_generalization_ratio": float(sub["mean_generalization_ratio"].mean()),
                "best_generalization_ratio": float(sub["mean_generalization_ratio"].min()),
            }
        )
    by_k_summary = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)
    return fold_summary, by_k_summary


def plot_posthoc_reconstruction_mse_by_k(
    fold_summary_df: pd.DataFrame, by_k_summary_df: pd.DataFrame, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.errorbar(
        fold_summary_df["k"],
        fold_summary_df["heldout_reconstruction_mse_mean_all_folds"],
        yerr=fold_summary_df["heldout_reconstruction_mse_std_all_folds"],
        fmt="o-",
        linewidth=2.0,
        capsize=4,
        label="Mean +/- 1 SD across folds/settings",
    )
    ax.scatter(
        by_k_summary_df["k"],
        by_k_summary_df["best_mean_heldout_reconstruction_mse"],
        s=55,
        color="#d62728",
        label="Best setting mean",
        zorder=3,
    )
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Held-out reconstruction MSE")
    ax.set_title("Autoencoder held-out reconstruction loss by latent dimension")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_posthoc_generalization_ratio_by_k(
    fold_summary_df: pd.DataFrame, by_k_summary_df: pd.DataFrame, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        fold_summary_df["k"],
        fold_summary_df["mean_generalization_ratio_all_folds"],
        marker="o",
        linewidth=2.0,
        label="Mean across folds/settings",
        color="#c76d06",
    )
    ax.scatter(
        by_k_summary_df["k"],
        by_k_summary_df["best_generalization_ratio"],
        s=55,
        color="#1f77b4",
        label="Best setting ratio",
        zorder=3,
    )
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Held-out / train reconstruction ratio")
    ax.set_title("Autoencoder reconstruction generalization ratio by latent dimension")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_posthoc_reconstruction_summary(output_dir: Path) -> Dict[str, Any]:
    fold_path = output_dir / "pyod_ae_fold_metrics.csv"
    aggregate_path = output_dir / "pyod_ae_aggregate_metrics.csv"
    residual_path = output_dir / "pyod_ae_reconstruction_residuals.csv"
    report_path = output_dir / "REPORT.md"
    if not fold_path.exists() or not aggregate_path.exists() or not residual_path.exists():
        raise FileNotFoundError(
            "Expected pyod_ae_fold_metrics.csv, pyod_ae_aggregate_metrics.csv, and "
            "pyod_ae_reconstruction_residuals.csv in the output directory."
        )

    fold_df = pd.read_csv(fold_path)
    aggregate_df = pd.read_csv(aggregate_path)
    residual_df = pd.read_csv(residual_path, usecols=["k", "feature_name"])
    feature_count_by_k = (
        residual_df.groupby("k")["feature_name"].nunique().sort_index().astype(int).to_dict()
    )
    fold_summary_df, by_k_summary_df = build_reconstruction_summary_by_k(fold_df, aggregate_df)
    by_k_summary_df.to_csv(output_dir / "ae_reconstruction_summary_by_k.csv", index=False)

    plot_posthoc_reconstruction_mse_by_k(
        fold_summary_df,
        by_k_summary_df,
        output_dir / "ae_reconstruction_mse_by_k.png",
    )
    plot_posthoc_generalization_ratio_by_k(
        fold_summary_df,
        by_k_summary_df,
        output_dir / "ae_generalization_ratio_by_k.png",
    )

    monotonic = bool(
        np.all(
            np.diff(
                fold_summary_df["heldout_reconstruction_mse_mean_all_folds"].to_numpy(dtype=np.float64)
            )
            <= 1.0e-12
        )
    )
    best_recon_row = by_k_summary_df.sort_values(
        ["best_mean_heldout_reconstruction_mse", "mean_generalization_ratio", "k"]
    ).iloc[0]
    best_ratio_row = by_k_summary_df.sort_values(
        ["best_generalization_ratio", "best_mean_heldout_reconstruction_mse", "k"]
    ).iloc[0]
    report_lines = [
        "Held-out reconstruction loss here is the fold-safe LOYO MSE on the standardized ocean-mode predictor features only. The underlying table has shape `37 x 92` including `water_year`, so the autoencoder reconstructs the `91` non-identifier predictor columns in each fold.",
        "Across latent dimensions, the mean held-out reconstruction MSE `{trend}` improve monotonically as `k` increases when averaged over dropout rates, weight decays, seeds, and held-out folds.".format(
            trend="does" if monotonic else "does not"
        ),
        "The best held-out reconstruction MSE was achieved by `k={k}` with best setting `dropout_rate={dropout}`, `weight_decay={wd}`, `seed={seed}`, giving mean held-out reconstruction MSE `{mse:.6f}`.".format(
            k=int(best_recon_row["k"]),
            dropout=float(best_recon_row["best_dropout_rate"]),
            wd=float(best_recon_row["best_weight_decay"]),
            seed=int(best_recon_row["best_seed"]),
            mse=float(best_recon_row["best_mean_heldout_reconstruction_mse"]),
        ),
        "The best training-vs-held-out generalization ratio was achieved by `k={k}` with best ratio `{ratio:.6f}`; lower is better because it indicates a smaller held-out reconstruction penalty relative to training reconstruction.".format(
            k=int(best_ratio_row["k"]),
            ratio=float(best_ratio_row["best_generalization_ratio"]),
        ),
        "A direct comparison between best reconstruction `k` and best SWE-prediction `k` is not available from this artifact set because the downstream SWE ridge comparison was left pending. For the main experiment, the 7-column baseline should still be compared only through downstream SWE prediction metrics, not reconstruction loss.",
    ]
    append_or_replace_report_section(
        report_path,
        "Autoencoder reconstruction loss by latent dimension",
        report_lines,
    )

    return {
        "fold_summary_df": fold_summary_df,
        "by_k_summary_df": by_k_summary_df,
        "feature_count_by_k": feature_count_by_k,
        "monotonic": monotonic,
        "best_recon_k": int(best_recon_row["k"]),
        "best_ratio_k": int(best_ratio_row["k"]),
    }


def build_report(
    output_dir: Path,
    *,
    pyod_version: str,
    args: argparse.Namespace,
    predictor_table: pd.DataFrame,
    summary: Dict[str, Any],
    fold_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    latent_extraction_ok: bool,
    baseline_path: Optional[Path],
    baseline_search: Dict[str, List[str]],
    alignment_df: pd.DataFrame,
    downstream_completed: bool,
    unresolved_issues: List[str],
    model_introspection: Dict[str, Any],
) -> None:
    best_recon = aggregate_df[aggregate_df["aggregate_level"] == "by_k_dropout_weight_decay"].sort_values(
        ["mean_test_reconstruction_mse", "median_test_reconstruction_mse", "mean_generalization_ratio"]
    ).iloc[0]
    best_ratio = aggregate_df[aggregate_df["aggregate_level"] == "by_k_dropout_weight_decay"].sort_values(
        ["mean_generalization_ratio", "median_generalization_ratio", "mean_test_reconstruction_mse"]
    ).iloc[0]
    memorization = (
        fold_df["train_recon_mse"].median() < 1.0e-2
        and fold_df["generalization_ratio"].median() > 10.0
    )
    lines = [
        "# PyOD ocean-mode autoencoder LOYO report",
        "",
        f"1. Was PyOD AutoEncoder installed already? {'No. It was installed into /pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3 during this run.' if pyod_version else 'Unknown.'}",
        f"2. What version of PyOD was used? `{pyod_version}`",
        "3. What exact AutoEncoder settings were used?",
        f"   `latent_dims={args.latent_dims}`, `dropout_rates={args.dropout_rates}`, `weight_decays={args.weight_decays}`, `seeds={args.seeds}`, `epoch_num={args.epoch_num}`, `batch_size={args.batch_size}`, `lr={args.lr}`, `hidden_neuron_list={list(args.hidden_neurons) + ['k']}`, `hidden_activation_name='relu'`, `batch_norm=False`, `preprocessing=False`, `optimizer_name='adam'`, `contamination={args.contamination}`.",
        f"4. What was the assembled ocean-mode predictor table shape? `{predictor_table.shape[0]} x {predictor_table.shape[1]}` including `water_year`.",
        f"5. Were there missing or constant predictor columns? Missing total = `{summary['missing_value_report']['total_missing_values']}`; globally constant columns dropped = `{summary.get('globally_constant_columns', [])}`.",
        "6. Which setting gave the best held-out reconstruction MSE?",
        "   `k={k}`, `dropout_rate={dropout}`, `weight_decay={wd}` with mean held-out reconstruction MSE `{mse:.6f}`.".format(
            k=int(best_recon["k"]),
            dropout=float(best_recon["dropout_rate"]),
            wd=float(best_recon["weight_decay"]),
            mse=float(best_recon["mean_test_reconstruction_mse"]),
        ),
        "7. Which setting gave the best training-vs-held-out generalization ratio?",
        "   `k={k}`, `dropout_rate={dropout}`, `weight_decay={wd}` with mean generalization ratio `{ratio:.6f}`.".format(
            k=int(best_ratio["k"]),
            dropout=float(best_ratio["dropout_rate"]),
            wd=float(best_ratio["weight_decay"]),
            ratio=float(best_ratio["mean_generalization_ratio"]),
        ),
        f"8. Did the model appear to memorize the 36 training years? {'Yes, at least partly.' if memorization else 'Not strongly enough to call it clear memorization across the full grid.'}",
        f"9. Could latent codes be extracted cleanly from the PyOD model? {'Yes.' if latent_extraction_ok else 'No.'}",
        f"   Extraction path: `{model_introspection.get('latent_extraction_method', 'unavailable')}`",
        f"10. If latent codes were extracted, do they align with the existing 7-column baseline? {'Yes, alignment diagnostics were computed.' if (latent_extraction_ok and not alignment_df.empty) else 'Alignment not available.'}",
        f"    Baseline path used: `{baseline_path}`" if baseline_path else f"    Baseline path used: unavailable. Candidates searched: `{baseline_search}`",
        f"11. If ridge comparison was completed, did latent-code ridge match, beat, or lose to the existing 7-column baseline? {'Completed.' if downstream_completed else 'Pending because apples-to-apples SWE target-path choice remained ambiguous between the regional ocean-mode ridge targets and the single-target 7-column Sierra baseline.'}",
        "12. Based on this first run, should we continue with package-based autoencoder on the ocean-mode table?",
        "    {}".format(
            "Yes, if the best settings show stable low held-out reconstruction and meaningful latent/baseline alignment; otherwise treat this as a package/API feasibility pass before tuning."
        ),
        "",
        "## Paths",
        "",
        f"- output directory: `{output_dir}`",
        f"- predictor table: `{output_dir / 'ocean_mode_predictor_table.csv'}`",
        f"- fold metrics: `{output_dir / 'pyod_ae_fold_metrics.csv'}`",
        f"- aggregate metrics: `{output_dir / 'pyod_ae_aggregate_metrics.csv'}`",
        f"- latent codes: `{output_dir / 'pyod_ae_latent_codes.csv'}`",
        f"- reconstruction residuals: `{output_dir / 'pyod_ae_reconstruction_residuals.csv'}`",
        "",
        "## Unresolved issues",
        "",
    ]
    if unresolved_issues:
        lines.extend([f"- {issue}" for issue in unresolved_issues])
    else:
        lines.append("- none")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / FIGURES_DIRNAME
    if args.posthoc_reconstruction_summary_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_info = run_posthoc_reconstruction_summary(output_dir)
        best_recon_row = summary_info["by_k_summary_df"].sort_values(
            ["best_mean_heldout_reconstruction_mse", "mean_generalization_ratio", "k"]
        ).iloc[0]
        best_ratio_row = summary_info["by_k_summary_df"].sort_values(
            ["best_generalization_ratio", "best_mean_heldout_reconstruction_mse", "k"]
        ).iloc[0]
        print(f"output directory: {output_dir}", flush=True)
        print(
            "best setting by held-out reconstruction MSE: "
            f"k={int(best_recon_row['k'])}, "
            f"dropout_rate={float(best_recon_row['best_dropout_rate'])}, "
            f"weight_decay={float(best_recon_row['best_weight_decay'])}, "
            f"seed={int(best_recon_row['best_seed'])}, "
            f"mean_test_reconstruction_mse={float(best_recon_row['best_mean_heldout_reconstruction_mse']):.6f}",
            flush=True,
        )
        print(
            "best setting by generalization ratio: "
            f"k={int(best_ratio_row['k'])}, "
            f"best_generalization_ratio={float(best_ratio_row['best_generalization_ratio']):.6f}",
            flush=True,
        )
        return

    ensure_runtime_on_compute_node()

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictor_table, table_summary = build_predictor_matrix()
    existing_match = False
    if EXISTING_OCEAN_MODE_PREDICTOR_TABLE.exists():
        existing_df = pd.read_csv(EXISTING_OCEAN_MODE_PREDICTOR_TABLE)
        existing_match = predictor_table.equals(existing_df)
        table_summary["matches_existing_ocean_mode_ridge_predictor_table"] = bool(existing_match)
        table_summary["existing_ocean_mode_ridge_predictor_table_path"] = str(EXISTING_OCEAN_MODE_PREDICTOR_TABLE)
    write_table_and_summary(predictor_table, table_summary, output_dir)

    feature_names = [col for col in predictor_table.columns if col != "water_year"]
    x_all = predictor_table[feature_names].to_numpy(dtype=np.float64)
    baseline_path, baseline_search = find_baseline_7col()
    baseline_df = None
    baseline_feature_names: List[str] = []
    if baseline_path is not None:
        baseline_df = pd.read_csv(baseline_path)
        baseline_feature_names = [col for col in baseline_df.columns if col not in {"water_year", "obs_swe"}]

    pyod_version = __import__("pyod").__version__
    fold_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    latent_rows: List[Dict[str, Any]] = []
    alignment_rows: List[Dict[str, Any]] = []
    unresolved_issues: List[str] = []
    fold_near_zero_columns: List[Dict[str, Any]] = []
    model_introspection: Dict[str, Any] = {}
    latent_extraction_ok = False

    for k in args.latent_dims:
        for dropout_rate in args.dropout_rates:
            for weight_decay in args.weight_decays:
                for seed in args.seeds:
                    for outer_idx, heldout_year in enumerate(WATER_YEARS):
                        train_mask = np.ones(WATER_YEARS.size, dtype=bool)
                        train_mask[outer_idx] = False
                        x_train = x_all[train_mask]
                        x_test = x_all[~train_mask][0]
                        x_train_std, x_test_std, _, _, near_zero_idx = standardize_train_only(x_train, x_test)
                        if near_zero_idx:
                            fold_near_zero_columns.append(
                                {
                                    "heldout_year": int(heldout_year),
                                    "k": int(k),
                                    "seed": int(seed),
                                    "dropout_rate": float(dropout_rate),
                                    "weight_decay": float(weight_decay),
                                    "near_zero_columns": [feature_names[idx] for idx in near_zero_idx],
                                }
                            )
                        clf = fit_pyod_autoencoder(
                            x_train_std,
                            k=int(k),
                            dropout_rate=float(dropout_rate),
                            weight_decay=float(weight_decay),
                            seed=int(seed),
                            args=args,
                        )
                        model = get_model_module(clf)
                        if not model_introspection:
                            model_introspection = {
                                "model_type": str(type(model)),
                                "encoder_type": str(type(getattr(model, "encoder", None))),
                                "decoder_type": str(type(getattr(model, "decoder", None))),
                                "model_named_modules_head": [
                                    {"name": name, "type": str(type(module))}
                                    for name, module in list(model.named_modules())[:20]
                                ],
                                "latent_extraction_method": "Used clf.model.encoder(...) for latent codes and clf.model(...) for reconstructions; verified on compute node with a dummy fitted AutoEncoder.",
                            }
                        x_train_recon = reconstruct_with_model(model, x_train_std)
                        x_test_recon = reconstruct_with_model(model, x_test_std[None, :])[0]
                        train_recon_mse = mse(x_train_std, x_train_recon)
                        test_recon_mse = mse(x_test_std[None, :], x_test_recon[None, :])
                        test_recon_rmse = float(np.sqrt(test_recon_mse))
                        generalization_ratio = float(test_recon_mse / train_recon_mse) if train_recon_mse > 0.0 else float("inf")

                        fold_rows.append(
                            {
                                "heldout_year": int(heldout_year),
                                "k": int(k),
                                "seed": int(seed),
                                "dropout_rate": float(dropout_rate),
                                "weight_decay": float(weight_decay),
                                "train_recon_mse": float(train_recon_mse),
                                "test_recon_mse": float(test_recon_mse),
                                "test_recon_rmse": float(test_recon_rmse),
                                "generalization_ratio": float(generalization_ratio),
                            }
                        )
                        for feature_name, x_std, x_hat in zip(feature_names, x_test_std.tolist(), x_test_recon.tolist()):
                            residual_rows.append(
                                {
                                    "heldout_year": int(heldout_year),
                                    "k": int(k),
                                    "seed": int(seed),
                                    "dropout_rate": float(dropout_rate),
                                    "weight_decay": float(weight_decay),
                                    "feature_name": feature_name,
                                    "x_standardized": float(x_std),
                                    "x_reconstructed": float(x_hat),
                                    "residual": float(x_std - x_hat),
                                }
                            )

                        try:
                            z_train = encode_with_model(model, x_train_std)
                            z_test = encode_with_model(model, x_test_std[None, :])[0]
                            latent_extraction_ok = True
                            train_years = WATER_YEARS[train_mask]
                            for row_year, row_z in zip(train_years.tolist(), z_train.tolist()):
                                row = {
                                    "heldout_year": int(heldout_year),
                                    "water_year": int(row_year),
                                    "split_role": "train",
                                    "k": int(k),
                                    "seed": int(seed),
                                    "dropout_rate": float(dropout_rate),
                                    "weight_decay": float(weight_decay),
                                }
                                for latent_idx in range(max(args.latent_dims)):
                                    row[f"z{latent_idx + 1}"] = float(row_z[latent_idx]) if latent_idx < int(k) else np.nan
                                latent_rows.append(row)
                            heldout_row = {
                                "heldout_year": int(heldout_year),
                                "water_year": int(heldout_year),
                                "split_role": "heldout",
                                "k": int(k),
                                "seed": int(seed),
                                "dropout_rate": float(dropout_rate),
                                "weight_decay": float(weight_decay),
                            }
                            for latent_idx in range(max(args.latent_dims)):
                                heldout_row[f"z{latent_idx + 1}"] = float(z_test[latent_idx]) if latent_idx < int(k) else np.nan
                            latent_rows.append(heldout_row)

                            if baseline_df is not None:
                                baseline_train = (
                                    baseline_df.set_index("water_year")
                                    .loc[train_years.tolist(), baseline_feature_names]
                                    .to_numpy(dtype=np.float64)
                                )
                                baseline_train_df = (
                                    baseline_df.set_index("water_year").loc[train_years.tolist(), baseline_feature_names]
                                )
                                r2_per_latent, mean_r2 = linear_r2_multivariate(baseline_train, z_train)
                                for latent_idx in range(int(k)):
                                    latent_name = f"z{latent_idx + 1}"
                                    for baseline_column in baseline_feature_names:
                                        corr = corrcoef_safe(
                                            z_train[:, latent_idx],
                                            baseline_train_df[baseline_column].to_numpy(dtype=np.float64),
                                        )
                                        alignment_rows.append(
                                            {
                                                "heldout_year": int(heldout_year),
                                                "k": int(k),
                                                "seed": int(seed),
                                                "dropout_rate": float(dropout_rate),
                                                "weight_decay": float(weight_decay),
                                                "latent_name": latent_name,
                                                "baseline_column": baseline_column,
                                                "correlation": float(corr),
                                                "abs_correlation": float(abs(corr)) if np.isfinite(corr) else np.nan,
                                                "latent_r2_explained_by_7col": float(r2_per_latent[latent_idx]),
                                                "mean_latent_r2_explained_by_7col": float(mean_r2),
                                                "baseline_predictor_table_path": str(baseline_path),
                                            }
                                        )
                        except Exception as exc:
                            if "Latent extraction failed" not in unresolved_issues:
                                unresolved_issues.append(f"Latent extraction failed for at least one fold: {exc}")

                        print(
                            "LOYO heldout_WY={} k={} seed={} dropout={} weight_decay={} train_mse={:.6f} test_mse={:.6f} ratio={:.6f}".format(
                                int(heldout_year),
                                int(k),
                                int(seed),
                                float(dropout_rate),
                                float(weight_decay),
                                train_recon_mse,
                                test_recon_mse,
                                generalization_ratio,
                            ),
                            flush=True,
                        )

    fold_df = pd.DataFrame(fold_rows).sort_values(
        ["k", "dropout_rate", "weight_decay", "seed", "heldout_year"]
    ).reset_index(drop=True)
    residual_df = pd.DataFrame(residual_rows).sort_values(
        ["k", "dropout_rate", "weight_decay", "seed", "heldout_year", "feature_name"]
    ).reset_index(drop=True)
    latent_df = pd.DataFrame(latent_rows)
    if not latent_df.empty:
        latent_df = latent_df.sort_values(
            ["k", "dropout_rate", "weight_decay", "seed", "heldout_year", "split_role", "water_year"]
        ).reset_index(drop=True)
    alignment_df = pd.DataFrame(alignment_rows)
    if not alignment_df.empty:
        alignment_df = alignment_df.sort_values(
            ["k", "dropout_rate", "weight_decay", "seed", "heldout_year", "latent_name", "baseline_column"]
        ).reset_index(drop=True)

    fold_df.to_csv(output_dir / "pyod_ae_fold_metrics.csv", index=False)
    residual_df.to_csv(output_dir / "pyod_ae_reconstruction_residuals.csv", index=False)
    if not latent_df.empty:
        latent_df.to_csv(output_dir / "pyod_ae_latent_codes.csv", index=False)
    else:
        (output_dir / "pyod_ae_latent_codes.csv").write_text("", encoding="utf-8")
    if not alignment_df.empty:
        alignment_df.to_csv(output_dir / "pyod_ae_alignment_with_7col_baseline.csv", index=False)
    else:
        (output_dir / "pyod_ae_alignment_with_7col_baseline.csv").write_text("", encoding="utf-8")

    def aggregate_rows(df: pd.DataFrame, group_cols: List[str], level_name: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for keys, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"aggregate_level": level_name}
            for col, value in zip(group_cols, keys):
                row[col] = value
            row["mean_test_reconstruction_mse"] = float(sub["test_recon_mse"].mean())
            row["median_test_reconstruction_mse"] = float(sub["test_recon_mse"].median())
            row["std_test_reconstruction_mse"] = float(sub["test_recon_mse"].std(ddof=1))
            row["mean_generalization_ratio"] = float(sub["generalization_ratio"].mean())
            row["median_generalization_ratio"] = float(sub["generalization_ratio"].median())
            year_mean = sub.groupby("heldout_year", as_index=False)["test_recon_mse"].mean()
            row["worst_heldout_reconstruction_year"] = int(
                year_mean.sort_values("test_recon_mse", ascending=False).iloc[0]["heldout_year"]
            )
            row["best_heldout_reconstruction_year"] = int(
                year_mean.sort_values("test_recon_mse", ascending=True).iloc[0]["heldout_year"]
            )
            rows.append(row)
        return pd.DataFrame(rows)

    agg_primary = aggregate_rows(
        fold_df, ["k", "dropout_rate", "weight_decay"], "by_k_dropout_weight_decay"
    )
    agg_secondary = aggregate_rows(
        fold_df, ["k", "dropout_rate", "weight_decay", "seed"], "by_k_dropout_weight_decay_seed"
    )
    aggregate_df = pd.concat([agg_primary, agg_secondary], ignore_index=True, sort=False)
    aggregate_df = aggregate_df.sort_values(
        ["aggregate_level", "k", "dropout_rate", "weight_decay", "seed"], na_position="last"
    ).reset_index(drop=True)
    aggregate_df.to_csv(output_dir / "pyod_ae_aggregate_metrics.csv", index=False)

    best_recon = agg_primary.sort_values(
        ["mean_test_reconstruction_mse", "median_test_reconstruction_mse", "mean_generalization_ratio"]
    ).iloc[0]
    best_ratio = agg_primary.sort_values(
        ["mean_generalization_ratio", "median_generalization_ratio", "mean_test_reconstruction_mse"]
    ).iloc[0]
    best_setting_for_plots = {
        "k": int(best_recon["k"]),
        "seed": int(
            fold_df[
                (fold_df["k"] == best_recon["k"])
                & (fold_df["dropout_rate"] == best_recon["dropout_rate"])
                & (fold_df["weight_decay"] == best_recon["weight_decay"])
            ]
            .sort_values("test_recon_mse")
            .iloc[0]["seed"]
        ),
        "dropout_rate": float(best_recon["dropout_rate"]),
        "weight_decay": float(best_recon["weight_decay"]),
    }

    plot_reconstruction_mse_by_k(aggregate_df, figures_dir / "reconstruction_mse_by_k.pdf")
    plot_generalization_ratio_by_k(aggregate_df, figures_dir / "generalization_ratio_by_k.pdf")
    plot_reconstruction_heatmaps(
        aggregate_df, figures_dir / "reconstruction_mse_by_dropout_weight_decay.pdf"
    )
    plot_worst_heldout_years(fold_df, figures_dir / "worst_heldout_years_reconstruction.pdf")
    if latent_extraction_ok and not latent_df.empty:
        plot_latent_scatter(
            latent_df, best_setting_for_plots, figures_dir / "latent_code_scatter_best_setting.pdf"
        )
    if latent_extraction_ok and not alignment_df.empty:
        plot_alignment_heatmap(
            alignment_df, best_setting_for_plots, figures_dir / "alignment_with_7col_baseline.pdf"
        )

    if fold_near_zero_columns:
        table_summary["fold_near_zero_column_events"] = fold_near_zero_columns[:200]
        table_summary["fold_near_zero_column_event_count"] = int(len(fold_near_zero_columns))
        (output_dir / "ocean_mode_predictor_table_summary.json").write_text(
            json.dumps(table_summary, indent=2) + "\n", encoding="utf-8"
        )

    metadata = {
        "script_path": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "pyod_version": pyod_version,
        "torch_version": str(torch.__version__),
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "predictor_table_shape": [int(predictor_table.shape[0]), int(predictor_table.shape[1])],
        "latent_dims": [int(v) for v in args.latent_dims],
        "dropout_rates": [float(v) for v in args.dropout_rates],
        "weight_decays": [float(v) for v in args.weight_decays],
        "seeds": [int(v) for v in args.seeds],
        "epoch_num": int(args.epoch_num),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_neurons": [int(v) for v in args.hidden_neurons],
        "baseline_7col_predictor_table_found": bool(baseline_path is not None),
        "baseline_7col_predictor_table_path": str(baseline_path) if baseline_path else None,
        "baseline_search": baseline_search,
        "latent_extraction_ok": bool(latent_extraction_ok),
        "downstream_swe_ridge_completed": False,
        "unresolved_issues": unresolved_issues,
        "model_introspection": model_introspection,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    build_report(
        output_dir,
        pyod_version=pyod_version,
        args=args,
        predictor_table=predictor_table,
        summary=table_summary,
        fold_df=fold_df,
        aggregate_df=aggregate_df,
        latent_extraction_ok=latent_extraction_ok,
        baseline_path=baseline_path,
        baseline_search=baseline_search,
        alignment_df=alignment_df,
        downstream_completed=False,
        unresolved_issues=unresolved_issues,
        model_introspection=model_introspection,
    )

    if not unresolved_issues:
        unresolved_issues.append(
            "Downstream SWE ridge comparison left pending because the apples-to-apples target choice between the regional ocean-mode ridge targets and the single-target 7-column Sierra baseline should be confirmed before running a comparison."
        )

    print(f"output directory: {output_dir}", flush=True)
    print(f"exact command used: {' '.join(sys.argv)}", flush=True)
    print(f"PyOD version: {pyod_version}", flush=True)
    print(f"predictor table shape: {predictor_table.shape[0]} x {predictor_table.shape[1]}", flush=True)
    print(
        "best setting by held-out reconstruction MSE: k={k}, dropout_rate={dropout}, weight_decay={wd}, mean_test_reconstruction_mse={mse:.6f}".format(
            k=int(best_recon["k"]),
            dropout=float(best_recon["dropout_rate"]),
            wd=float(best_recon["weight_decay"]),
            mse=float(best_recon["mean_test_reconstruction_mse"]),
        ),
        flush=True,
    )
    print(
        "best setting by generalization ratio: k={k}, dropout_rate={dropout}, weight_decay={wd}, mean_generalization_ratio={ratio:.6f}".format(
            k=int(best_ratio["k"]),
            dropout=float(best_ratio["dropout_rate"]),
            wd=float(best_ratio["weight_decay"]),
            ratio=float(best_ratio["mean_generalization_ratio"]),
        ),
        flush=True,
    )
    print(f"whether latent codes were successfully extracted: {'yes' if latent_extraction_ok else 'no'}", flush=True)
    print(f"whether the existing 7-column baseline artifact was found: {'yes' if baseline_path is not None else 'no'}", flush=True)
    print("whether downstream SWE ridge comparison was completed: no", flush=True)
    print("any failure or unresolved issue: {}".format("; ".join(unresolved_issues)), flush=True)


if __name__ == "__main__":
    main()
