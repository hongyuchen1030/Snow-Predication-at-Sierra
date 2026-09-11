#!/usr/bin/env python3
"""Compare 32-year AQM, CPM, and AQM+CPM LOYO SWE predictability."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_z1z2_plus_amv_k5_loyo as ref  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_aqm_swe_32yr"
AQM_DIR = PROJECT_ROOT / "artifacts" / "aqm_index_loyo"
CPM_DIR = PROJECT_ROOT / "artifacts" / "cpm_swe_attribution"

AQM_TABLE_CSV = AQM_DIR / "aqm_swe_predictor_table.csv"
AQM_SOURCE_JSON = AQM_DIR / "aqm_source_and_method.json"
CPM_OVERLAP_CSV = CPM_DIR / "cpm_swe_overlap_table.csv"
CPM_REPORT_MD = CPM_DIR / "report.md"

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "cpm_aqm_32yr_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "cpm_aqm_32yr_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "cpm_aqm_32yr_loyo_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "cpm_aqm_32yr_summary.json"
REPORT_MD = OUTPUT_DIR / "report.md"
COMPARISON_FIGURE_PNG = OUTPUT_DIR / "cpm_aqm_32yr_loyo_comparison.png"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2016

MODEL_FEATURES = {
    "AQM_only": ["AQM_NDJF"],
    "CPM_only": ["CPM_NovApr"],
    "AQM_plus_CPM": ["AQM_NDJF", "CPM_NovApr"],
}
MODEL_LABELS = {
    "AQM_only": "AQM only",
    "CPM_only": "CPM only",
    "AQM_plus_CPM": "AQM + CPM",
}
MODEL_COLORS = {
    "AQM_only": "#d55e00",
    "CPM_only": "#0072b2",
    "AQM_plus_CPM": "#009e73",
}


def load_table() -> tuple[pd.DataFrame, dict]:
    aqm = pd.read_csv(AQM_TABLE_CSV)
    aqm["water_year"] = aqm["water_year"].astype(int)
    aqm = aqm[(aqm["water_year"] >= WATER_YEAR_START) & (aqm["water_year"] <= WATER_YEAR_END)].copy()
    needed_aqm = ["water_year", "observed_swe", "AQM_NDJF"]
    missing = [col for col in needed_aqm if col not in aqm.columns]
    if missing:
        raise ValueError(f"Missing AQM columns: {missing}")

    cpm = pd.read_csv(CPM_OVERLAP_CSV)
    cpm["water_year"] = cpm["water_year"].astype(int)
    needed_cpm = ["water_year", "obs_swe", "cpm_pc3_nov_apr_mean"]
    missing = [col for col in needed_cpm if col not in cpm.columns]
    if missing:
        raise ValueError(f"Missing CPM columns: {missing}")
    cpm = cpm[(cpm["water_year"] >= WATER_YEAR_START) & (cpm["water_year"] <= WATER_YEAR_END)].copy()
    cpm = cpm.rename(columns={"obs_swe": "observed_swe_cpm", "cpm_pc3_nov_apr_mean": "CPM_NovApr"})

    merged = aqm.merge(cpm[["water_year", "observed_swe_cpm", "CPM_NovApr"]], on="water_year", how="inner").sort_values("water_year").reset_index(drop=True)
    if len(merged) != (WATER_YEAR_END - WATER_YEAR_START + 1):
        raise ValueError(f"Expected 32 overlap years, found {len(merged)}")
    swe_diff = np.abs(merged["observed_swe"] - merged["observed_swe_cpm"])
    if np.nanmax(swe_diff) > 1.0e-12:
        raise ValueError("AQM and CPM tables do not share the same observed SWE target.")
    merged = merged.drop(columns=["observed_swe_cpm"])
    aqm_meta = json.loads(AQM_SOURCE_JSON.read_text())
    return merged, aqm_meta


def run_loyo_for_model(table: pd.DataFrame, feature_columns: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    y = table["observed_swe"].to_numpy(dtype=float)
    x_all = table[feature_columns].to_numpy(dtype=float)

    prediction_rows = []
    for held_idx, held_year in enumerate(years):
        train_mask = np.ones(len(years), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_all[train_mask, :]
        x_test = x_all[~train_mask, :][0]
        y_train = y[train_mask]
        y_test = float(y[held_idx])

        selected_alpha, _ = ref.inner_loyo_best_alpha(x_train, y_train)
        pred_raw, beta_raw, intercept_raw = ref.fit_outer_model(x_train, y_train, x_test, selected_alpha)
        row = {
            "model_name": model_name,
            "water_year": int(held_year),
            "observed_swe": y_test,
            "predicted_swe": float(pred_raw),
            "selected_ridge_alpha": float(selected_alpha),
            "intercept": float(intercept_raw),
        }
        for beta_idx, beta in enumerate(beta_raw):
            row[f"beta_{feature_columns[beta_idx]}"] = float(beta)
        prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows).sort_values("water_year").reset_index(drop=True)
    metrics = ref.compute_metric_bundle(pred_df["observed_swe"].to_numpy(dtype=float), pred_df["predicted_swe"].to_numpy(dtype=float))
    metrics_df = pd.DataFrame([{"model_name": model_name, "num_predictors": len(feature_columns), "n_years": len(pred_df), **metrics}])
    return pred_df, metrics_df


def make_comparison_figure(panels: list[tuple[str, pd.DataFrame, float]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
    for col, (model_name, pred_df, r2) in enumerate(panels):
        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]
        years = pred_df["water_year"].to_numpy(dtype=int)
        obs = pred_df["observed_swe"].to_numpy(dtype=float)
        pred = pred_df["predicted_swe"].to_numpy(dtype=float)

        ax_line = axes[0, col]
        ax_line.plot(years, obs, color="black", linewidth=2.2, label="Observed SWE")
        ax_line.plot(years, pred, color=color, marker="o", linewidth=1.6, label="Predicted SWE")
        ax_line.set_title(label)
        ax_line.set_xlabel("Water year")
        ax_line.set_ylabel("April 1 Sierra SWE anomaly")
        ax_line.grid(True, alpha=0.25)
        ax_line.legend(frameon=True, fontsize=9)

        ax_scatter = axes[1, col]
        lo = min(obs.min(), pred.min())
        hi = max(obs.max(), pred.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        lims = (lo - pad, hi + pad)
        ax_scatter.scatter(obs, pred, color=color, s=48, alpha=0.85)
        ax_scatter.plot(lims, lims, color="black", linestyle="--", linewidth=1.1)
        ax_scatter.set_xlim(lims)
        ax_scatter.set_ylim(lims)
        ax_scatter.set_xlabel("Observed SWE anomaly")
        ax_scatter.set_ylabel("Predicted SWE anomaly")
        ax_scatter.grid(True, alpha=0.2)
        ax_scatter.text(
            0.04,
            0.96,
            f"$R^2$ = {r2:.3f}",
            transform=ax_scatter.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
    fig.suptitle("32-year LOYO April 1 Sierra SWE prediction: AQM vs CPM vs AQM+CPM")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(COMPARISON_FIGURE_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table, aqm_meta = load_table()
    table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    all_pred_rows = []
    all_metrics_rows = []
    for model_name, feature_columns in MODEL_FEATURES.items():
        pred_df, metrics_df = run_loyo_for_model(table, feature_columns, model_name)
        all_pred_rows.append(pred_df)
        all_metrics_rows.append(metrics_df)

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    metrics = pd.concat(all_metrics_rows, ignore_index=True)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    metrics.to_csv(METRICS_CSV, index=False)

    panels = []
    for model_name in ["AQM_only", "CPM_only", "AQM_plus_CPM"]:
        pred_df = predictions[predictions["model_name"] == model_name].copy()
        r2 = float(metrics.loc[metrics["model_name"] == model_name, "R2"].iloc[0])
        panels.append((model_name, pred_df, r2))
    make_comparison_figure(panels)

    summary = {
        "years": list(range(WATER_YEAR_START, WATER_YEAR_END + 1)),
        "n_years": WATER_YEAR_END - WATER_YEAR_START + 1,
        "aqm_source": str(AQM_TABLE_CSV),
        "aqm_source_description": aqm_meta["source_description"],
        "cpm_source": str(CPM_OVERLAP_CSV),
        "cpm_report": str(CPM_REPORT_MD),
        "models": {row["model_name"]: row for row in metrics.to_dict(orient="records")},
        "figure": str(COMPARISON_FIGURE_PNG),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    report_lines = [
        "# 32-year CPM + AQM SWE Comparison",
        "",
        f"- AQM source: `{AQM_TABLE_CSV}`",
        f"- CPM source: `{CPM_OVERLAP_CSV}`",
        f"- Years: `WY{WATER_YEAR_START}-WY{WATER_YEAR_END}`",
        f"- n: `{WATER_YEAR_END - WATER_YEAR_START + 1}`",
        "",
        "## LOYO metrics",
        "",
        "| Model | r | R2 | RMSE | MAE | sign_accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name in ["AQM_only", "CPM_only", "AQM_plus_CPM"]:
        row = metrics.loc[metrics["model_name"] == model_name].iloc[0]
        report_lines.append(
            f"| {MODEL_LABELS[model_name]} | {float(row['r']):.6f} | {float(row['R2']):.6f} | {float(row['RMSE']):.6f} | {float(row['MAE']):.6f} | {float(row['sign_accuracy']):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Predictor table: `{PREDICTOR_TABLE_CSV}`",
            f"- LOYO predictions: `{PREDICTIONS_CSV}`",
            f"- LOYO metrics: `{METRICS_CSV}`",
            f"- Comparison figure: `{COMPARISON_FIGURE_PNG}`",
        ]
    )
    REPORT_MD.write_text("\n".join(report_lines) + "\n")

    print(f"Predictor table: {PREDICTOR_TABLE_CSV}")
    for model_name in ["AQM_only", "CPM_only", "AQM_plus_CPM"]:
        row = metrics.loc[metrics["model_name"] == model_name].iloc[0]
        print(f"{MODEL_LABELS[model_name]}: r={float(row['r']):.6f}, R2={float(row['R2']):.6f}, RMSE={float(row['RMSE']):.6f}")
    print(f"Figure: {COMPARISON_FIGURE_PNG}")


if __name__ == "__main__":
    main()
