#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "kgae_more_latents_swe_loyo"
PSCRATCH_ARTIFACT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_more_latents_swe_loyo")

BASELINE_TABLE = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"
PACIFIC_PC_TABLE = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_plus_PacificPC_Nino34_loyo"
    / "z1_z2_pacificpc_nino34_predictor_table.csv"
)
EXISTING_METRICS_CSV = PSCRATCH_ARTIFACT_DIR / "kgae_more_latents_swe_ridge_loyo_metrics.csv"
EXISTING_PREDICTIONS_CSV = PSCRATCH_ARTIFACT_DIR / "kgae_more_latents_swe_ridge_loyo_predictions.csv"
EXISTING_REPORT_MD = HOME_ARTIFACT_DIR / "REPORT.md"

ALPHAS = np.logspace(-6, 6, 49)
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]

OUT_METRICS = "kgae_more_latents_swe_ridge_loyo_metrics_pacific_fair.csv"
OUT_PREDICTIONS = "kgae_more_latents_swe_ridge_loyo_predictions_pacific_fair.csv"
OUT_BEST_MODELS = "kgae_more_latents_best_models_pacific_fair.csv"
OUT_SUMMARY = "kgae_more_latents_pacific_fair_summary.md"
OUT_RMSE = "kgae_more_latents_pacific_fair_rmse.png"
OUT_R = "kgae_more_latents_pacific_fair_r.png"
OUT_SIGN = "kgae_more_latents_pacific_fair_sign_accuracy.png"
OUT_TS = "kgae_more_latents_pacific_fair_timeseries.png"
OUT_SCATTER = "kgae_more_latents_pacific_fair_scatter.png"

BASELINE_NAME = "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC"
Z12_NAME = "Z1_Z2_PACIFIC_LOD_RIDGE"
PC6_NAME = "PACIFIC_PC6_RIDGE"
PC16_NAME = "PACIFIC_PC1TO6_RIDGE"
KGAE_MODELS = [
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_prediction_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_obs = df["y_obs"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    err = y_pred - y_obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    obs_std = float(np.std(y_obs, ddof=0))
    pred_std = float(np.std(y_pred, ddof=0))
    r = float(np.corrcoef(y_obs, y_pred)[0, 1]) if len(y_obs) > 1 else np.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    sign_accuracy = float(np.mean(np.sign(y_obs) == np.sign(y_pred)))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "r": r,
        "sign_accuracy": sign_accuracy,
        "bias": bias,
        "pred_std": pred_std,
        "obs_std": obs_std,
        "number_of_years": int(len(y_obs)),
    }


def select_ridge_alpha_inner_loyo(x_train: np.ndarray, y_train: np.ndarray) -> float:
    best_alpha = None
    best_mse = np.inf
    n = x_train.shape[0]
    for alpha in ALPHAS:
        se = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mu = x_train[mask].mean(axis=0)
            sigma = x_train[mask].std(axis=0)
            sigma[sigma == 0.0] = 1.0
            xtr = (x_train[mask] - mu) / sigma
            xva = (x_train[i : i + 1] - mu) / sigma
            gram = xtr.T @ xtr
            rhs = xtr.T @ y_train[mask]
            beta = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
            pred = float((xva @ beta)[0])
            se.append((pred - y_train[i]) ** 2)
        cur = float(np.mean(se))
        if cur < best_mse:
            best_mse = cur
            best_alpha = float(alpha)
    return float(best_alpha)


def run_loyo_ridge(model_name: str, model_family: str, predictor_df: pd.DataFrame, feature_cols: list[str], k_value: float | None) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    data = predictor_df.sort_values("water_year").reset_index(drop=True)
    x_all = data[feature_cols].to_numpy(dtype=float)
    y_all = data["obs_swe"].to_numpy(dtype=float)
    years = data["water_year"].to_numpy(dtype=int)
    for i, wy in enumerate(years):
        mask = np.ones(len(years), dtype=bool)
        mask[i] = False
        alpha = select_ridge_alpha_inner_loyo(x_all[mask], y_all[mask])
        mu = x_all[mask].mean(axis=0)
        sigma = x_all[mask].std(axis=0)
        sigma[sigma == 0.0] = 1.0
        xtr = (x_all[mask] - mu) / sigma
        xte = (x_all[i : i + 1] - mu) / sigma
        gram = xtr.T @ xtr
        rhs = xtr.T @ y_all[mask]
        beta = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
        pred = float((xte @ beta)[0])
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "k": float(k_value) if k_value is not None else np.nan,
                "water_year": int(wy),
                "y_obs": float(y_all[i]),
                "y_pred": pred,
                "alpha_selected": alpha,
            }
        )
    pred_df = pd.DataFrame(rows)
    metrics = {"model_name": model_name, "model_family": model_family}
    metrics.update(compute_prediction_metrics(pred_df))
    return pred_df, metrics


def build_z1_z2_baseline() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(BASELINE_TABLE).sort_values("water_year").reset_index(drop=True)
    cols = ["water_year", "obs_swe", "Z1", "Z2"]
    return df[cols].copy(), ["Z1", "Z2"]


def build_pacific_pc_baseline() -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(PACIFIC_PC_TABLE).sort_values("water_year").reset_index(drop=True)
    pc6_cols = [f"Pacific_PC6_{month}" for month in MONTHS if f"Pacific_PC6_{month}" in df.columns]
    if len(pc6_cols) == len(MONTHS):
        cols = ["water_year", "obs_swe"] + pc6_cols
        return df[cols].copy(), pc6_cols, PC6_NAME
    pc_cols = [
        f"Pacific_PC{pc}_{month}"
        for pc in range(1, 7)
        for month in MONTHS
        if f"Pacific_PC{pc}_{month}" in df.columns
    ]
    if not pc_cols:
        raise ValueError("No clear Pacific PC columns found for a Pacific-only baseline.")
    cols = ["water_year", "obs_swe"] + pc_cols
    return df[cols].copy(), pc_cols, PC16_NAME


def load_existing_kgae_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(EXISTING_METRICS_CSV)
    preds = pd.read_csv(EXISTING_PREDICTIONS_CSV)
    metrics = metrics.copy()
    preds = preds.copy()
    metrics.loc[metrics["model_name"] == "BASELINE_7COL_RIDGE", "model_name"] = BASELINE_NAME
    metrics.loc[metrics["model_family"] == "BASELINE_7COL_RIDGE", "model_family"] = BASELINE_NAME
    preds.loc[preds["model_name"] == "BASELINE_7COL_RIDGE", "model_name"] = BASELINE_NAME
    preds.loc[preds["model_family"] == "BASELINE_7COL_RIDGE", "model_family"] = BASELINE_NAME
    keep_models = [BASELINE_NAME] + KGAE_MODELS
    metrics = metrics.loc[metrics["model_name"].isin(keep_models)].copy()
    preds = preds.loc[preds["model_name"].isin(keep_models)].copy()
    return metrics, preds


def metric_bar_plot(metrics_df: pd.DataFrame, value_col: str, ylabel: str, output_path: Path, order: list[str], display_names: dict[str, str]) -> None:
    plot_df = metrics_df.set_index("model_name").loc[order].reset_index()
    plt.figure(figsize=(8.8, 4.6))
    plt.bar(np.arange(len(plot_df)), plot_df[value_col].to_numpy(dtype=float), color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"][: len(plot_df)])
    plt.xticks(np.arange(len(plot_df)), [display_names[name] for name in plot_df["model_name"]], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_timeseries(pred_df: pd.DataFrame, best_pacific_name: str, best_kgae_name: str, output_path: Path, display_names: dict[str, str]) -> None:
    sel = pred_df.loc[pred_df["model_name"].isin([BASELINE_NAME, best_pacific_name, best_kgae_name])].copy()
    pivot = sel.pivot(index="water_year", columns="model_name", values="y_pred").sort_index()
    obs = sel.groupby("water_year")["y_obs"].first().sort_index()
    plt.figure(figsize=(10.5, 4.8))
    plt.plot(obs.index, obs.values, color="black", linewidth=2.2, label="Observed SWE")
    for model_name, color in [(BASELINE_NAME, "#4c78a8"), (best_pacific_name, "#f58518"), (best_kgae_name, "#54a24b")]:
        plt.plot(pivot.index, pivot[model_name].values, linewidth=1.8, marker="o", label=display_names[model_name], color=color)
    plt.xlabel("Water year")
    plt.ylabel("April 1 SWE anomaly (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_scatter(pred_df: pd.DataFrame, best_pacific_name: str, best_kgae_name: str, output_path: Path, display_names: dict[str, str]) -> None:
    models = [best_pacific_name, best_kgae_name, BASELINE_NAME]
    colors = {best_pacific_name: "#f58518", best_kgae_name: "#54a24b", BASELINE_NAME: "#4c78a8"}
    sel = pred_df.loc[pred_df["model_name"].isin(models)].copy()
    lo = min(sel["y_obs"].min(), sel["y_pred"].min())
    hi = max(sel["y_obs"].max(), sel["y_pred"].max())
    plt.figure(figsize=(6.2, 6.2))
    for model_name in models:
        sub = sel.loc[sel["model_name"] == model_name]
        plt.scatter(sub["y_obs"], sub["y_pred"], alpha=0.8, s=36, label=display_names[model_name], color=colors[model_name])
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    plt.xlabel("Observed SWE anomaly (m)")
    plt.ylabel("Predicted SWE anomaly (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_best_models_table(metrics_df: pd.DataFrame, pacific_model_name: str) -> pd.DataFrame:
    best_pacific = metrics_df.loc[metrics_df["model_name"].isin([Z12_NAME, pacific_model_name])].sort_values(["RMSE", "MAE"]).iloc[0]
    best_kgae = metrics_df.loc[metrics_df["model_name"].isin(KGAE_MODELS)].sort_values(["RMSE", "MAE"]).iloc[0]
    z12_row = metrics_df.loc[metrics_df["model_name"] == Z12_NAME].iloc[0]
    pc_row = metrics_df.loc[metrics_df["model_name"] == pacific_model_name].iloc[0]
    baseline = metrics_df.loc[metrics_df["model_name"] == BASELINE_NAME].iloc[0]
    rows = [
        {"best_model_role": "best_overall_benchmark", **baseline.to_dict()},
        {"best_model_role": "best_pacific_only_baseline", **best_pacific.to_dict()},
        {"best_model_role": "best_kgae_larger_latent_model", **best_kgae.to_dict()},
        {"best_model_role": "z1_z2_pacific_baseline", **z12_row.to_dict()},
        {"best_model_role": "pacific_pc_baseline", **pc_row.to_dict()},
    ]
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    header = [str(col) for col in df.columns.tolist()]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [len(col) for col in header]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    line_header = "| " + " | ".join(text.ljust(widths[idx]) for idx, text in enumerate(header)) + " |"
    line_sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(header))) + " |"
    line_rows = ["| " + " | ".join(text.ljust(widths[idx]) for idx, text in enumerate(row)) + " |" for row in rows]
    return "\n".join([line_header, line_sep] + line_rows)


def build_summary(metrics_df: pd.DataFrame, pacific_model_name: str, output_path: Path, display_names: dict[str, str]) -> None:
    def row(name: str) -> pd.Series:
        return metrics_df.loc[metrics_df["model_name"] == name].iloc[0]

    baseline = row(BASELINE_NAME)
    z12 = row(Z12_NAME)
    pac = row(pacific_model_name)
    k5 = row("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5")
    k10 = row("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10")
    best_pacific = metrics_df.loc[metrics_df["model_name"].isin([Z12_NAME, pacific_model_name])].sort_values(["RMSE", "MAE"]).iloc[0]
    best_kgae = metrics_df.loc[metrics_df["model_name"].isin(KGAE_MODELS)].sort_values(["RMSE", "MAE"]).iloc[0]

    lines = [
        "# KGAE More-Latents Pacific-Only Fair Summary",
        "",
        "## Main correction",
        "",
        "- The original 7-column baseline is kept as the current best overall benchmark, but it is not Pacific-only.",
        "- It is labeled here as `BASELINE_7COL_PACIFIC_PLUS_ATLANTIC` because it contains Pacific `Z1`, `Z2` plus five Atlantic AMV/AMO/AQM-related predictors.",
        "- The fair KGAE comparison is therefore against Pacific-only ridge baselines.",
        "",
        "## Models in this patch",
        "",
        f"- `{BASELINE_NAME}`",
        f"- `{Z12_NAME}`",
        f"- `{pacific_model_name}`",
        "- `KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5`",
        "- `KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10`",
        "",
        "## Direct answers",
        "",
        f"- How does KGAE k=5 compare to Z1/Z2? RMSE `{k5['RMSE']:.6f}` vs `{z12['RMSE']:.6f}`, r `{k5['r']:.6f}` vs `{z12['r']:.6f}`, sign accuracy `{k5['sign_accuracy']:.6f}` vs `{z12['sign_accuracy']:.6f}`.",
        f"- How does KGAE k=10 compare to Z1/Z2? RMSE `{k10['RMSE']:.6f}` vs `{z12['RMSE']:.6f}`, r `{k10['r']:.6f}` vs `{z12['r']:.6f}`, sign accuracy `{k10['sign_accuracy']:.6f}` vs `{z12['sign_accuracy']:.6f}`.",
        f"- How does KGAE k=5 compare to {pacific_model_name}? RMSE `{k5['RMSE']:.6f}` vs `{pac['RMSE']:.6f}`, r `{k5['r']:.6f}` vs `{pac['r']:.6f}`, sign accuracy `{k5['sign_accuracy']:.6f}` vs `{pac['sign_accuracy']:.6f}`.",
        f"- How does KGAE k=10 compare to {pacific_model_name}? RMSE `{k10['RMSE']:.6f}` vs `{pac['RMSE']:.6f}`, r `{k10['r']:.6f}` vs `{pac['r']:.6f}`, sign accuracy `{k10['sign_accuracy']:.6f}` vs `{pac['sign_accuracy']:.6f}`.",
        f"- Does k=10 improve over k=5? RMSE {'yes' if k10['RMSE'] < k5['RMSE'] else 'no'}, Pearson r {'yes' if k10['r'] > k5['r'] else 'no'}, sign accuracy {'yes' if k10['sign_accuracy'] > k5['sign_accuracy'] else 'no'}.",
        f"- Does k=10 approach or beat the Pacific-only baselines? {'No' if k10['RMSE'] > min(z12['RMSE'], pac['RMSE']) else 'Yes'}; best Pacific-only RMSE is `{best_pacific['RMSE']:.6f}` and k=10 RMSE is `{k10['RMSE']:.6f}`.",
        f"- How far are all Pacific-only models from the full 7-column Pacific+Atlantic baseline? Best Pacific-only RMSE gap to benchmark is `{best_pacific['RMSE'] - baseline['RMSE']:+.6f}`, best KGAE RMSE gap is `{best_kgae['RMSE'] - baseline['RMSE']:+.6f}`.",
        "",
        "## Fair-comparison interpretation",
        "",
    ]

    if (best_kgae["RMSE"] <= best_pacific["RMSE"]) and (best_kgae["RMSE"] > baseline["RMSE"]):
        lines.append(
            "- KGAE is competitive as a Pacific-only representation, but it cannot recover the Atlantic/AQM information present in the full 7-column baseline."
        )
    elif (best_kgae["RMSE"] > best_pacific["RMSE"]) and (best_kgae["RMSE"] > baseline["RMSE"]):
        if (k10["RMSE"] < z12["RMSE"]) and (k10["RMSE"] > pac["RMSE"]):
            lines.append(
                "- KGAE captures some useful Pacific SST structure but still misses or weakens the Pacific PC direction most associated with SWE."
            )
        else:
            lines.append(
                "- Increasing KGAE latent dimension improved PC6-like alignment but did not produce a Pacific-only representation as predictive as the existing Pacific-only baselines."
            )
    else:
        lines.append("- The Pacific-only comparisons should be read directly from the metrics table; no stronger claim is warranted than the measured LOYO skill.")

    lines.extend(
        [
            "",
            "## Metrics snapshot",
            "",
            dataframe_to_markdown(
                metrics_df[["model_name", "RMSE", "MAE", "R2", "r", "sign_accuracy", "bias", "pred_std", "obs_std", "number_of_years"]]
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def patch_report(report_path: Path, metrics_df: pd.DataFrame, pacific_model_name: str) -> None:
    existing = report_path.read_text() if report_path.exists() else "# KGAE More-Latents Report\n"
    marker = "## Pacific-only fair comparison"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n"
    else:
        existing = existing.rstrip() + "\n\n"

    def row(name: str) -> pd.Series:
        return metrics_df.loc[metrics_df["model_name"] == name].iloc[0]

    baseline = row(BASELINE_NAME)
    z12 = row(Z12_NAME)
    pac = row(pacific_model_name)
    k5 = row("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5")
    k10 = row("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10")
    best_pacific = metrics_df.loc[metrics_df["model_name"].isin([Z12_NAME, pacific_model_name])].sort_values(["RMSE", "MAE"]).iloc[0]
    best_kgae = metrics_df.loc[metrics_df["model_name"].isin(KGAE_MODELS)].sort_values(["RMSE", "MAE"]).iloc[0]

    lines = [
        existing,
        "## Pacific-only fair comparison",
        "",
        "- The 7-column baseline is the current best overall benchmark but it is not Pacific-only.",
        "- In this patched comparison it is labeled `BASELINE_7COL_PACIFIC_PLUS_ATLANTIC`.",
        "- KGAE is Pacific-only.",
        f"- Therefore, the fair KGAE comparison is against `{Z12_NAME}` and `{pacific_model_name}`.",
        f"- `{pacific_model_name}` is not part of the 7-column baseline.",
        f"- `{pacific_model_name}` comes from the separate Pacific PC/Nino predictor table and is used here only as a Pacific-side diagnostic/reference.",
        "- The five non-Z1/Z2 columns in the 7-column baseline are Atlantic AMV/AMO/AQM-related predictors.",
        "",
        "Patched Pacific-only metrics:",
        "",
        f"- `{BASELINE_NAME}`: RMSE `{baseline['RMSE']:.6f}`, R2 `{baseline['R2']:.6f}`, r `{baseline['r']:.6f}`, sign accuracy `{baseline['sign_accuracy']:.6f}`",
        f"- `{Z12_NAME}`: RMSE `{z12['RMSE']:.6f}`, R2 `{z12['R2']:.6f}`, r `{z12['r']:.6f}`, sign accuracy `{z12['sign_accuracy']:.6f}`",
        f"- `{pacific_model_name}`: RMSE `{pac['RMSE']:.6f}`, R2 `{pac['R2']:.6f}`, r `{pac['r']:.6f}`, sign accuracy `{pac['sign_accuracy']:.6f}`",
        f"- `KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5`: RMSE `{k5['RMSE']:.6f}`, R2 `{k5['R2']:.6f}`, r `{k5['r']:.6f}`, sign accuracy `{k5['sign_accuracy']:.6f}`",
        f"- `KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10`: RMSE `{k10['RMSE']:.6f}`, R2 `{k10['R2']:.6f}`, r `{k10['r']:.6f}`, sign accuracy `{k10['sign_accuracy']:.6f}`",
        "",
        "Interpretation:",
        "",
    ]

    if (best_kgae["RMSE"] <= best_pacific["RMSE"]) and (best_kgae["RMSE"] > baseline["RMSE"]):
        lines.append("KGAE is competitive as a Pacific-only representation, but it cannot recover the Atlantic/AQM information present in the full 7-column baseline.")
    elif (best_kgae["RMSE"] > best_pacific["RMSE"]) and (best_kgae["RMSE"] > baseline["RMSE"]):
        if (k10["RMSE"] < z12["RMSE"]) and (k10["RMSE"] > pac["RMSE"]):
            lines.append("KGAE captures some useful Pacific SST structure but still misses or weakens the Pacific PC direction most associated with SWE.")
        else:
            lines.append("Increasing KGAE latent dimension improved PC6-like alignment but did not produce a Pacific-only representation as predictive as the existing Pacific-only baselines.")
    else:
        lines.append("The patched Pacific-only comparison should be read directly from the reported strict-LOYO metrics without a stronger claim.")

    report_path.write_text("\n".join(lines) + "\n")


def write_outputs(pscratch_dir: Path, home_dir: Path, metrics_df: pd.DataFrame, pred_df: pd.DataFrame, best_models_df: pd.DataFrame, summary_text_path: Path, figures: list[str]) -> None:
    metrics_df.to_csv(pscratch_dir / OUT_METRICS, index=False)
    pred_df.to_csv(pscratch_dir / OUT_PREDICTIONS, index=False)
    best_models_df.to_csv(pscratch_dir / OUT_BEST_MODELS, index=False)
    for name in [OUT_METRICS, OUT_PREDICTIONS, OUT_BEST_MODELS]:
        src = pscratch_dir / name
        dst = home_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    dst_summary = home_dir / OUT_SUMMARY
    if dst_summary.exists() or dst_summary.is_symlink():
        dst_summary.unlink()
    dst_summary.write_text(summary_text_path.read_text())
    for fig in figures:
        src = pscratch_dir / fig
        dst = home_dir / fig
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            dst.write_bytes(src.read_bytes())


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    ensure_dir(PSCRATCH_ARTIFACT_DIR)
    ensure_dir(HOME_ARTIFACT_DIR)

    existing_metrics, existing_preds = load_existing_kgae_outputs()
    z12_df, z12_features = build_z1_z2_baseline()
    pac_df, pac_features, pacific_model_name = build_pacific_pc_baseline()

    z12_pred, z12_metrics = run_loyo_ridge(Z12_NAME, Z12_NAME, z12_df, z12_features, None)
    pac_pred, pac_metrics = run_loyo_ridge(pacific_model_name, pacific_model_name, pac_df, pac_features, 6.0 if pacific_model_name == PC6_NAME else np.nan)

    metrics_rows = [row.to_dict() for _, row in existing_metrics.iterrows()]
    metrics_rows.extend([z12_metrics, pac_metrics])
    metrics_df = pd.DataFrame(metrics_rows)

    pred_df = pd.concat([existing_preds, z12_pred, pac_pred], ignore_index=True)

    model_order = [BASELINE_NAME, Z12_NAME, pacific_model_name] + KGAE_MODELS
    metrics_df["model_name"] = pd.Categorical(metrics_df["model_name"], categories=model_order, ordered=True)
    metrics_df = metrics_df.sort_values("model_name").reset_index(drop=True)
    pred_df["model_name"] = pd.Categorical(pred_df["model_name"], categories=model_order, ordered=True)
    pred_df = pred_df.sort_values(["model_name", "water_year"]).reset_index(drop=True)

    best_models_df = build_best_models_table(metrics_df, pacific_model_name)

    display_names = {
        BASELINE_NAME: "7-col Pacific+Atlantic",
        Z12_NAME: "Z1/Z2 Pacific",
        pacific_model_name: "Pacific PC6" if pacific_model_name == PC6_NAME else "Pacific PC1-6",
        "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5": "KGAE k=5",
        "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10": "KGAE k=10",
    }
    metric_bar_plot(metrics_df, "RMSE", "RMSE (m)", PSCRATCH_ARTIFACT_DIR / OUT_RMSE, model_order, display_names)
    metric_bar_plot(metrics_df, "r", "Pearson r", PSCRATCH_ARTIFACT_DIR / OUT_R, model_order, display_names)
    metric_bar_plot(metrics_df, "sign_accuracy", "Sign accuracy", PSCRATCH_ARTIFACT_DIR / OUT_SIGN, model_order, display_names)

    best_pacific_name = metrics_df.loc[metrics_df["model_name"].isin([Z12_NAME, pacific_model_name])].sort_values(["RMSE", "MAE"]).iloc[0]["model_name"]
    best_kgae_name = metrics_df.loc[metrics_df["model_name"].isin(KGAE_MODELS)].sort_values(["RMSE", "MAE"]).iloc[0]["model_name"]
    plot_timeseries(pred_df, best_pacific_name, best_kgae_name, PSCRATCH_ARTIFACT_DIR / OUT_TS, display_names)
    plot_scatter(pred_df, best_pacific_name, best_kgae_name, PSCRATCH_ARTIFACT_DIR / OUT_SCATTER, display_names)

    summary_path = PSCRATCH_ARTIFACT_DIR / OUT_SUMMARY
    build_summary(metrics_df, pacific_model_name, summary_path, display_names)
    patch_report(EXISTING_REPORT_MD, metrics_df, pacific_model_name)

    write_outputs(
        PSCRATCH_ARTIFACT_DIR,
        HOME_ARTIFACT_DIR,
        metrics_df,
        pred_df,
        best_models_df,
        summary_path,
        [OUT_RMSE, OUT_R, OUT_SIGN, OUT_TS, OUT_SCATTER],
    )
    print("Wrote Pacific-fair KGAE patch outputs to", PSCRATCH_ARTIFACT_DIR)


if __name__ == "__main__":
    main()
