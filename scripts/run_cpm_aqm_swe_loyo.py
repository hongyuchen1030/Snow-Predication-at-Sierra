#!/usr/bin/env python3
"""
Strict LOYO ridge experiments using the wet-season CPM (ERA5 Z500 PC3 Nov-Apr
mean) index, alone and combined with the existing AQM_NDJF index, against the
real 37-year UCLA SWE target. Mirrors scripts/run_aqm_index_loyo.py, reusing
the same shared ridge/metric helpers.
"""

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


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_aqm_swe_loyo"

AQM_LOYO_DIR = PROJECT_ROOT / "artifacts" / "aqm_index_loyo"
AQM_PREDICTOR_TABLE_CSV = AQM_LOYO_DIR / "aqm_swe_predictor_table.csv"
AQM_PREDICTIONS_CSV = AQM_LOYO_DIR / "aqm_seasonal_loyo_predictions.csv"
AQM_METRICS_CSV = AQM_LOYO_DIR / "aqm_seasonal_loyo_metrics.csv"

CPM_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cpm_era5_reproduction_wy2021"
CPM_DAILY_PC_CSV = CPM_ARTIFACT_DIR / "era5_z500_pc1to6_daily.csv"
CPM_METADATA_JSON = CPM_ARTIFACT_DIR / "era5_z500_processing_metadata.json"

PREDICTOR_TABLE_CSV = OUTPUT_DIR / "cpm_aqm_swe_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "cpm_aqm_swe_loyo_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "cpm_aqm_swe_loyo_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "cpm_aqm_swe_summary.json"
COMPARISON_FIGURE_PNG = OUTPUT_DIR / "aqm_cpm_combined_loyo_comparison.png"
README_PATH = OUTPUT_DIR / "README.md"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))

MODEL_FEATURES = {
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


def load_aqm_predictor_table() -> pd.DataFrame:
    df = pd.read_csv(AQM_PREDICTOR_TABLE_CSV)
    df["water_year"] = df["water_year"].astype(int)
    needed = ["water_year", "observed_swe", "AQM_NDJF"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in AQM predictor table: {missing}")
    return df[needed].copy()


def load_cpm_seasonal_index() -> pd.DataFrame:
    metadata = json.loads(CPM_METADATA_JSON.read_text())
    if int(metadata["best_match_mode"]) != 3:
        raise ValueError(
            f"CPM best-match mode changed to {metadata['best_match_mode']} in the extended "
            "ERA5 refit; PC3 can no longer be assumed to be the CPM analog without re-checking."
        )
    sign = int(metadata["best_match_sign"])

    daily = pd.read_csv(CPM_DAILY_PC_CSV, parse_dates=["date"])
    daily["water_year"] = daily["water_year"].astype(int)
    daily["CPM_pc3"] = sign * daily["PC3"].astype(float)
    seasonal = (
        daily.groupby("water_year", as_index=False)["CPM_pc3"]
        .mean()
        .rename(columns={"CPM_pc3": "CPM_NovApr"})
    )
    return seasonal


def build_predictor_table() -> pd.DataFrame:
    aqm = load_aqm_predictor_table()
    cpm = load_cpm_seasonal_index()
    table = aqm.merge(cpm, on="water_year", how="inner").sort_values("water_year").reset_index(drop=True)
    table = table[(table["water_year"] >= WATER_YEAR_START) & (table["water_year"] <= WATER_YEAR_END)].copy()
    if table["water_year"].tolist() != WATER_YEARS:
        missing = sorted(set(WATER_YEARS) - set(table["water_year"].tolist()))
        raise ValueError(f"Predictor table does not cover WY{WATER_YEAR_START}-{WATER_YEAR_END}; missing {missing}")
    return table


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
        pred_raw, _, _ = ref.fit_outer_model(x_train, y_train, x_test, selected_alpha)
        prediction_rows.append(
            {
                "model_name": model_name,
                "water_year": int(held_year),
                "observed_swe": y_test,
                "predicted_swe": float(pred_raw),
                "selected_ridge_alpha": float(selected_alpha),
            }
        )
    pred_df = pd.DataFrame(prediction_rows).sort_values("water_year").reset_index(drop=True)
    metric_bundle = ref.compute_metric_bundle(
        pred_df["observed_swe"].to_numpy(dtype=float),
        pred_df["predicted_swe"].to_numpy(dtype=float),
    )
    metrics_df = pd.DataFrame(
        [{"model_name": model_name, "num_predictors": len(feature_columns), "n_years": len(pred_df), **metric_bundle}]
    )
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
        ax_line.plot(years, obs, color="black", linewidth=2.2, label="Observed")
        ax_line.plot(years, pred, color=color, marker="o", linewidth=1.6, label="Predicted")
        ax_line.set_title(label)
        ax_line.set_xlabel("Water year")
        ax_line.set_ylabel("April 1 Sierra SWE anomaly (m)")
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
        ax_scatter.set_xlabel("Observed SWE anomaly (m)")
        ax_scatter.set_ylabel("Predicted SWE anomaly (m)")
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
    fig.suptitle("Strict LOYO prediction of April 1 Sierra SWE: AQM vs CPM vs AQM+CPM")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(COMPARISON_FIGURE_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    ref.ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building CPM+AQM predictor table...", flush=True)
    table = build_predictor_table()
    table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    all_pred_rows = []
    all_metrics_rows = []
    for model_name, feature_columns in MODEL_FEATURES.items():
        print(f"Running LOYO for {model_name}...", flush=True)
        pred_df, metrics_df = run_loyo_for_model(table, feature_columns, model_name)
        all_pred_rows.append(pred_df)
        all_metrics_rows.append(metrics_df)

    combined_pred_df = pd.concat(all_pred_rows, ignore_index=True)
    combined_metrics_df = pd.concat(all_metrics_rows, ignore_index=True)
    combined_pred_df.to_csv(PREDICTIONS_CSV, index=False)
    combined_metrics_df.to_csv(METRICS_CSV, index=False)

    aqm_pred_df = pd.read_csv(AQM_PREDICTIONS_CSV)
    aqm_metrics_df = pd.read_csv(AQM_METRICS_CSV)
    aqm_r2 = float(aqm_metrics_df.iloc[0]["R2"])

    cpm_metrics_row = combined_metrics_df[combined_metrics_df["model_name"] == "CPM_only"].iloc[0]
    combo_metrics_row = combined_metrics_df[combined_metrics_df["model_name"] == "AQM_plus_CPM"].iloc[0]
    cpm_pred_df = combined_pred_df[combined_pred_df["model_name"] == "CPM_only"].copy()
    combo_pred_df = combined_pred_df[combined_pred_df["model_name"] == "AQM_plus_CPM"].copy()
    cpm_r2 = float(cpm_metrics_row["R2"])
    combo_r2 = float(combo_metrics_row["R2"])

    print("Building comparison figure...", flush=True)
    make_comparison_figure(
        [
            ("AQM_only", aqm_pred_df, aqm_r2),
            ("CPM_only", cpm_pred_df, cpm_r2),
            ("AQM_plus_CPM", combo_pred_df, combo_r2),
        ]
    )

    summary = {
        "water_years": WATER_YEARS,
        "n_years": len(WATER_YEARS),
        "aqm_source": str(AQM_PREDICTOR_TABLE_CSV),
        "cpm_source": str(CPM_DAILY_PC_CSV),
        "cpm_best_match_mode": 3,
        "models": {
            "AQM_only": aqm_metrics_df.iloc[0].to_dict(),
            "CPM_only": combined_metrics_df[combined_metrics_df["model_name"] == "CPM_only"].iloc[0].to_dict(),
            "AQM_plus_CPM": combined_metrics_df[combined_metrics_df["model_name"] == "AQM_plus_CPM"].iloc[0].to_dict(),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    README_PATH.write_text(
        "\n".join(
            [
                "# CPM and AQM+CPM LOYO prediction of real Sierra SWE",
                "",
                "Mirrors `artifacts/aqm_index_loyo` (AQM_NDJF only). Adds the wet-season CPM "
                "index (ERA5 Z500 PC3, Nov-Apr mean, from `artifacts/cpm_era5_reproduction_wy2021`, "
                "extended from the original 1981-2016 validation run so it covers the full "
                f"WY{WATER_YEAR_START}-{WATER_YEAR_END} real UCLA SWE record) alone, and combined with AQM_NDJF.",
                "",
                "## Results",
                "",
                f"- AQM only: r={float(aqm_metrics_df.iloc[0]['r']):.4f}, R2={aqm_r2:.4f}, sign_accuracy={float(aqm_metrics_df.iloc[0]['sign_accuracy']):.4f}",
                f"- CPM only: r={float(cpm_metrics_row['r']):.4f}, R2={cpm_r2:.4f}, sign_accuracy={float(cpm_metrics_row['sign_accuracy']):.4f}",
                f"- AQM + CPM: r={float(combo_metrics_row['r']):.4f}, R2={combo_r2:.4f}, sign_accuracy={float(combo_metrics_row['sign_accuracy']):.4f}",
                "",
                f"See `{COMPARISON_FIGURE_PNG.name}` for the 3x2 panel comparison.",
            ]
        )
        + "\n"
    )

    print("Done. Artifact directory:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
