#!/usr/bin/env python3
"""
Strict LOYO ridge experiment using the published NOAA monthly AMV/AMO index only.
"""

import json
import os
import sys
from datetime import datetime, timezone
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

from scripts import run_z1z2_plus_amv_k5_loyo as ref


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "noaa_amv_index_loyo"
RAW_DATA_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/data/noaa_amv")
RAW_DATA_PATH = RAW_DATA_DIR / "ersst.v5.amo.dat"

README_PATH = OUTPUT_DIR / "README.md"
SOURCE_METADATA_JSON = OUTPUT_DIR / "noaa_amv_source_metadata.json"
MONTHLY_PROCESSED_CSV = OUTPUT_DIR / "noaa_amv_monthly_processed.csv"
PREDICTOR_TABLE_CSV = OUTPUT_DIR / "noaa_amv_swe_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "noaa_amv_loyo_predictions.csv"
HYPERPARAMETERS_CSV = OUTPUT_DIR / "noaa_amv_loyo_selected_hyperparameters.csv"
METRICS_CSV = OUTPUT_DIR / "noaa_amv_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "noaa_amv_loyo_period_metrics.csv"
TIMESERIES_PNG = OUTPUT_DIR / "noaa_amv_loyo_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "noaa_amv_loyo_observed_vs_predicted.png"
AUDIT_JSON = OUTPUT_DIR / "noaa_amv_loyo_audit.json"

BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)
FINAL_SST_ONLY_METRICS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "final_sst_only_linear_closure_table"
    / "final_sst_only_loyo_metrics.csv"
)
FINAL_SST_ONLY_PERIOD_METRICS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "final_sst_only_linear_closure_table"
    / "final_sst_only_loyo_period_metrics.csv"
)

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
MONTH_SPECS = [
    ("NOAA_AMV_Sep", 9, -1),
    ("NOAA_AMV_Oct", 10, -1),
    ("NOAA_AMV_Nov", 11, -1),
    ("NOAA_AMV_Dec", 12, -1),
    ("NOAA_AMV_Jan", 1, 0),
    ("NOAA_AMV_Feb", 2, 0),
    ("NOAA_AMV_Mar", 3, 0),
]
FEATURE_COLUMNS = [name for name, _, _ in MONTH_SPECS]
MODEL_NAME = "NOAA_AMV_index_only"

SOURCE_PAGE_URL = "https://psl.noaa.gov/data/timeseries/AMO/"
DATA_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.amo.dat"
NCEI_README_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/Readme"


def load_noaa_monthly_index() -> tuple[pd.DataFrame, dict]:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("Missing raw NOAA file: {}".format(RAW_DATA_PATH))

    df = pd.read_csv(
        RAW_DATA_PATH,
        delim_whitespace=True,
        skiprows=2,
        names=["year", "month", "noaa_amv"],
    )
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["noaa_amv"] = df["noaa_amv"].astype(float)
    df["date"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": np.ones(len(df), dtype=int),
        }
    )
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["missing_flag"] = df["noaa_amv"].isin([-99.99, -999.0, -999.99])
    if bool(df["missing_flag"].any()):
        raise ValueError("Raw NOAA AMV file contains missing-value sentinels within parsed rows.")

    meta = {
        "raw_data_path": str(RAW_DATA_PATH),
        "source_page_url": SOURCE_PAGE_URL,
        "data_url": DATA_URL,
        "ncei_readme_url": NCEI_README_URL,
        "download_access_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_name": "ERSST AMO (North Atlantic 0-60N SSTA) Index",
        "dataset_version": "ERSST v5",
        "host_agency": "NOAA/NCEI",
        "time_frequency": "monthly",
        "coverage_start": df["date"].min().strftime("%Y-%m"),
        "coverage_end": df["date"].max().strftime("%Y-%m"),
        "smoothing": "none documented in the raw file; used as the published monthly series",
        "detrending": "not documented in the accessible NCEI file header or Readme; treated as published monthly SSTA series with no additional detrending applied here",
        "missing_value_convention": "No sentinel values detected in the downloaded file; checked against -99.99, -999, and -999.99",
        "sign_convention": "Positive values indicate warmer-than-average North Atlantic SST anomalies, i.e. the positive AMV/AMO phase.",
        "provenance_note": (
            "NOAA PSL's AMO page points users to this NOAA/NCEI ERSSTv5 time series. "
            "The NCEI directory Readme says these indices are generated for user convenience "
            "and are not the official NOAA indices used for climate monitoring."
        ),
    }
    return df[["date", "year", "month", "noaa_amv"]].copy(), meta


def load_reference_target_series() -> pd.DataFrame:
    base_predictions = pd.read_csv(BASE_PREDICTIONS_CSV)
    base_predictions = base_predictions.loc[
        (base_predictions["patch_size"] == "exact_grid_cell") & (base_predictions["model_name"] == "Z1_Z2")
    ].copy()
    base_predictions = (
        base_predictions[["heldout_wy", "obs_swe"]]
        .rename(columns={"heldout_wy": "water_year", "obs_swe": "observed_swe"})
        .sort_values("water_year")
        .reset_index(drop=True)
    )
    base_predictions["water_year"] = base_predictions["water_year"].astype(int)
    base_predictions = base_predictions.loc[
        (base_predictions["water_year"] >= WATER_YEAR_START) & (base_predictions["water_year"] <= WATER_YEAR_END)
    ].copy()
    if base_predictions["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Reference target series does not match WY1985-WY2021.")
    return base_predictions


def build_predictor_table(monthly_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    indexed = monthly_df.set_index(["year", "month"])["noaa_amv"]
    rows = []
    missing_pairs = []
    for water_year in WATER_YEARS:
        row = {"water_year": int(water_year)}
        for name, month, year_offset in MONTH_SPECS:
            calendar_year = water_year + year_offset
            key = (calendar_year, month)
            if key not in indexed.index:
                missing_pairs.append({"water_year": int(water_year), "calendar_year": int(calendar_year), "month": int(month)})
                row[name] = np.nan
            else:
                row[name] = float(indexed.loc[key])
        rows.append(row)
    predictors = pd.DataFrame(rows)
    table = target_df.merge(predictors, on="water_year", how="left")
    missing_years = table.loc[table[FEATURE_COLUMNS].isna().any(axis=1), "water_year"].astype(int).tolist()
    if missing_years:
        raise ValueError("Missing NOAA AMV predictors for water years: {}".format(missing_years))
    if table["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Predictor table does not match WY1985-WY2021.")
    mapping = {
        "water_year_definition": "April 1 Sierra SWE target for water year Y",
        "month_mapping": {
            "NOAA_AMV_Sep": "September of calendar year Y-1",
            "NOAA_AMV_Oct": "October of calendar year Y-1",
            "NOAA_AMV_Nov": "November of calendar year Y-1",
            "NOAA_AMV_Dec": "December of calendar year Y-1",
            "NOAA_AMV_Jan": "January of calendar year Y",
            "NOAA_AMV_Feb": "February of calendar year Y",
            "NOAA_AMV_Mar": "March of calendar year Y",
        },
        "missing_month_pairs": missing_pairs,
        "missing_water_years": missing_years,
        "n_usable_years": int(len(table)),
    }
    return table[["water_year", "observed_swe"] + FEATURE_COLUMNS].copy(), mapping


def run_loyo(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    y = table["observed_swe"].to_numpy(dtype=float)
    x_all = table[FEATURE_COLUMNS].to_numpy(dtype=float)

    prediction_rows = []
    hyper_rows = []
    preds = np.full(len(years), np.nan, dtype=float)

    for held_idx, held_year in enumerate(years):
        train_mask = np.ones(len(years), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_all[train_mask, :]
        x_test = x_all[~train_mask, :][0]
        y_train = y[train_mask]
        y_test = float(y[held_idx])

        selected_alpha, inner_cv_mse = ref.inner_loyo_best_alpha(x_train, y_train)
        pred_raw, beta_raw, intercept_raw = ref.fit_outer_model(x_train, y_train, x_test, selected_alpha)
        preds[held_idx] = float(pred_raw)
        residual = float(y_test - pred_raw)
        error = float(pred_raw - y_test)
        sign_correct = np.nan
        if y_test != 0.0 and pred_raw != 0.0:
            sign_correct = float(np.sign(y_test) == np.sign(pred_raw))

        prediction_rows.append(
            {
                "water_year": int(held_year),
                "observed_swe": float(y_test),
                "predicted_swe": float(pred_raw),
                "residual": residual,
                "error_pred_minus_obs": error,
                "abs_error": float(abs(error)),
                "selected_ridge_alpha": float(selected_alpha),
                "sign_correct": sign_correct,
                "intercept": float(intercept_raw),
            }
        )
        hyper_row = {
            "water_year": int(held_year),
            "selected_ridge_alpha": float(selected_alpha),
            "inner_cv_mse": float(inner_cv_mse),
            "intercept": float(intercept_raw),
        }
        for name, beta in zip(FEATURE_COLUMNS, beta_raw):
            hyper_row["beta_" + name] = float(beta)
        hyper_rows.append(hyper_row)

    pred_df = pd.DataFrame(prediction_rows).sort_values("water_year").reset_index(drop=True)
    hyper_df = pd.DataFrame(hyper_rows).sort_values("water_year").reset_index(drop=True)

    metric_bundle = ref.compute_metric_bundle(
        pred_df["observed_swe"].to_numpy(dtype=float),
        pred_df["predicted_swe"].to_numpy(dtype=float),
    )
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "num_predictors": len(FEATURE_COLUMNS),
                "r": metric_bundle["r"],
                "R2": metric_bundle["R2"],
                "RMSE": metric_bundle["RMSE"],
                "MAE": metric_bundle["MAE"],
                "sign_accuracy": metric_bundle["sign_accuracy"],
                "mean_error": metric_bundle["mean_error"],
                "median_abs_error": metric_bundle["median_abs_error"],
            }
        ]
    )

    period_rows = []
    obs = pred_df["observed_swe"].to_numpy(dtype=float)
    pred = pred_df["predicted_swe"].to_numpy(dtype=float)
    for group_name, selector in ref.PERIOD_SPECS:
        mask = selector(years)
        bundle = ref.compute_metric_bundle(obs[mask], pred[mask])
        period_rows.append(
            {
                "model_name": MODEL_NAME,
                "group_name": group_name,
                "n_years": int(mask.sum()),
                "r": bundle["r"],
                "R2": bundle["R2"],
                "RMSE": bundle["RMSE"],
                "MAE": bundle["MAE"],
                "sign_accuracy": bundle["sign_accuracy"],
                "mean_error": bundle["mean_error"],
                "median_abs_error": bundle["median_abs_error"],
            }
        )
    period_df = pd.DataFrame(period_rows).sort_values("group_name").reset_index(drop=True)
    return pred_df, hyper_df, metrics_df, period_df


def make_timeseries_plot(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.plot(
        pred_df["water_year"],
        pred_df["observed_swe"],
        color="black",
        linewidth=2.4,
        label="Observed",
    )
    ax.plot(
        pred_df["water_year"],
        pred_df["predicted_swe"],
        color="#1f77b4",
        marker="o",
        linewidth=1.8,
        label="NOAA AMV index only",
    )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO prediction using the published NOAA monthly AMV index")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(TIMESERIES_PNG, dpi=200)
    plt.close(fig)


def make_scatter_plot(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    obs = pred_df["observed_swe"].to_numpy(dtype=float)
    pred = pred_df["predicted_swe"].to_numpy(dtype=float)
    lo = min(np.min(obs), np.min(pred))
    hi = max(np.max(obs), np.max(pred))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lims = (lo - pad, hi + pad)

    row = metrics_df.iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(obs, pred, color="#1f77b4", s=52, alpha=0.85)
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed SWE anomaly (m)")
    ax.set_ylabel("Predicted SWE anomaly (m)")
    ax.set_title("Observed vs predicted SWE: NOAA AMV index only")
    ax.grid(True, alpha=0.2)
    ax.text(
        0.04,
        0.96,
        "RMSE = {:.4f}\n$R^2$ = {:.3f}\nCorr = {:.3f}\nSign = {:.3f}".format(
            float(row["RMSE"]),
            float(row["R2"]),
            float(row["r"]),
            float(row["sign_accuracy"]),
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=200)
    plt.close(fig)


def run_reference_audit() -> dict:
    table, _ = ref.load_predictor_table()
    rerun_pred_df, rerun_alpha_df, _, = ref.run_models(table)
    rerun_metrics_df = ref.compute_metrics(rerun_pred_df)

    saved_pred_df = pd.read_csv(ref.PREDICTIONS_CSV).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    saved_alpha_df = pd.read_csv(ref.ALPHA_CSV).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    saved_metrics_df = pd.read_csv(ref.METRICS_CSV).sort_values("model_name").reset_index(drop=True)
    rerun_metrics_df = rerun_metrics_df.sort_values("model_name").reset_index(drop=True)

    pred_diff = float(np.max(np.abs(rerun_pred_df["pred_swe"].to_numpy(dtype=float) - saved_pred_df["pred_swe"].to_numpy(dtype=float))))
    alpha_diff = float(
        np.max(np.abs(rerun_alpha_df["selected_alpha"].to_numpy(dtype=float) - saved_alpha_df["selected_alpha"].to_numpy(dtype=float)))
    )
    metric_diff = {}
    for key in ["r", "R2", "RMSE", "MAE", "sign_accuracy"]:
        metric_diff[key] = float(
            np.max(np.abs(rerun_metrics_df[key].to_numpy(dtype=float) - saved_metrics_df[key].to_numpy(dtype=float)))
        )

    return {
        "reference_script": str(Path(ref.__file__).resolve()),
        "reference_output_dir": str(ref.OUTPUT_DIR),
        "water_years_match": bool(table["water_year"].astype(int).tolist() == WATER_YEARS),
        "alpha_grid_match": [float(x) for x in ref.ALPHA_GRID.tolist()],
        "prediction_max_abs_diff": pred_diff,
        "selected_alpha_max_abs_diff": alpha_diff,
        "metric_max_abs_diff": metric_diff,
        "reproduces_saved_reference_outputs": bool(
            pred_diff <= 1.0e-12
            and alpha_diff <= 1.0e-12
            and all(value <= 1.0e-12 for value in metric_diff.values())
        ),
    }


def load_comparison_metrics() -> dict:
    metrics_df = pd.read_csv(FINAL_SST_ONLY_METRICS_CSV)
    period_df = pd.read_csv(FINAL_SST_ONLY_PERIOD_METRICS_CSV)
    overall = metrics_df.loc[metrics_df["model_name"] == "AMV_AMO_PC1to6_only"].iloc[0].to_dict()
    period = (
        period_df.loc[period_df["model_name"] == "AMV_AMO_PC1to6_only"]
        .sort_values("group_name")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    return {"overall": overall, "period_metrics": period}


def write_readme(
    source_meta: dict,
    mapping_meta: dict,
    metrics_df: pd.DataFrame,
    period_df: pd.DataFrame,
    audit: dict,
    comparison: dict,
) -> None:
    overall = metrics_df.iloc[0].to_dict()
    lines = [
        "# NOAA monthly AMV index -> April 1 Sierra SWE",
        "",
        "This artifact runs the published NOAA-hosted monthly AMV/AMO index as the only predictor block for strict LOYO ridge prediction of April 1 Sierra SWE.",
        "",
        "## Data source",
        "",
        "- Raw file: `{}`".format(RAW_DATA_PATH),
        "- NOAA source page: `{}`".format(SOURCE_PAGE_URL),
        "- NOAA data URL: `{}`".format(DATA_URL),
        "- Dataset/version: {} / {}".format(source_meta["product_name"], source_meta["dataset_version"]),
        "- Coverage used here: {} to {}".format(source_meta["coverage_start"], source_meta["coverage_end"]),
        "- Frequency: {}".format(source_meta["time_frequency"]),
        "- Detrending note: {}".format(source_meta["detrending"]),
        "- Smoothing note: {}".format(source_meta["smoothing"]),
        "- Caveat: {}".format(source_meta["provenance_note"]),
        "",
        "## Water-year mapping",
        "",
    ]
    for key in FEATURE_COLUMNS:
        lines.append("- {}: {}".format(key, mapping_meta["month_mapping"][key]))
    lines.extend(
        [
            "",
            "## Predictor set",
            "",
            "- Predictors: {}".format(", ".join(FEATURE_COLUMNS)),
            "- Usable water years: {}".format(mapping_meta["n_usable_years"]),
            "- Missing water years: {}".format(mapping_meta["missing_water_years"]),
            "",
            "## Ridge configuration reused from the reference workflow",
            "",
            "- Outer split: leave one water year out over WY1985-WY2021",
            "- Predictor standardization: train-fold mean/std only",
            "- Target standardization: train-fold mean/std only",
            "- Alpha grid: {}".format(", ".join("{:g}".format(x) for x in ref.ALPHA_GRID.tolist())),
            "- Inner selection: leave one training year out, choose lowest MSE, break ties toward smaller alpha",
            "- Raw-space reconstruction: use the reference script's coefficient and intercept reconstruction",
            "",
            "## NOAA-index result",
            "",
            "- Overall: r={r:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, sign_accuracy={sign_accuracy:.6f}".format(**overall),
            "",
            "## Period metrics",
            "",
        ]
    )
    for _, row in period_df.iterrows():
        lines.append(
            "- {} (n={}): r={:.6f}, R2={:.6f}, RMSE={:.6f}, MAE={:.6f}, sign_accuracy={:.6f}".format(
                row["group_name"],
                int(row["n_years"]),
                float(row["r"]),
                float(row["R2"]),
                float(row["RMSE"]),
                float(row["MAE"]),
                float(row["sign_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reference-pipeline audit",
            "",
            "- Reference outputs reproduced exactly: {}".format(audit["reproduces_saved_reference_outputs"]),
            "- Max abs prediction diff vs saved reference: {:.3e}".format(audit["prediction_max_abs_diff"]),
            "- Max abs selected-alpha diff vs saved reference: {:.3e}".format(audit["selected_alpha_max_abs_diff"]),
            "",
            "## Factual comparison to the existing North Atlantic-PC model",
            "",
            "- Existing `AMV_AMO_PC1to6_only` overall metrics: r={r:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, sign_accuracy={sign_accuracy:.6f}".format(
                **comparison["overall"]
            ),
            "- This comparison is statistical only: the NOAA index is a published scalar monthly series, while the existing model uses 42 monthly North Atlantic EOF-PC predictors.",
        ]
    )
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ref.ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_df, source_meta = load_noaa_monthly_index()
    target_df = load_reference_target_series()
    predictor_table, mapping_meta = build_predictor_table(monthly_df, target_df)

    monthly_df.to_csv(MONTHLY_PROCESSED_CSV, index=False)
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    pred_df, hyper_df, metrics_df, period_df = run_loyo(predictor_table)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMETERS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)

    make_timeseries_plot(pred_df)
    make_scatter_plot(pred_df, metrics_df)

    audit = run_reference_audit()
    comparison = load_comparison_metrics()

    source_meta["water_year_mapping"] = mapping_meta
    source_meta["predictor_columns"] = FEATURE_COLUMNS
    SOURCE_METADATA_JSON.write_text(json.dumps(source_meta, indent=2), encoding="utf-8")
    AUDIT_JSON.write_text(json.dumps({"reference_audit": audit, "comparison_to_amv_pc_model": comparison}, indent=2), encoding="utf-8")
    write_readme(source_meta, mapping_meta, metrics_df, period_df, audit, comparison)

    overall = metrics_df.iloc[0]
    print("Output directory: {}".format(OUTPUT_DIR))
    print("Usable years: {}".format(mapping_meta["n_usable_years"]))
    print(
        "Metrics: r={:.6f}, R2={:.6f}, RMSE={:.6f}, MAE={:.6f}, sign_accuracy={:.6f}".format(
            float(overall["r"]),
            float(overall["R2"]),
            float(overall["RMSE"]),
            float(overall["MAE"]),
            float(overall["sign_accuracy"]),
        )
    )
    print("Reference audit reproduced saved outputs: {}".format(audit["reproduces_saved_reference_outputs"]))


if __name__ == "__main__":
    main()
