#!/usr/bin/env python3
"""
Quick sanity check comparing honest nested LOYO M1+M2 against oracle full37 exact Z1+Z2.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NESTED_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_mode_subset_diagnostic"
NESTED_PREDICTIONS_CSV = NESTED_DIR / "loyo_mode_subset_predictions.csv"
NESTED_METRICS_CSV = NESTED_DIR / "loyo_mode_subset_metrics.csv"
NESTED_SUMMARY_JSON = NESTED_DIR / "loyo_mode_subset_summary.json"

ORACLE_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo"
ORACLE_PREDICTIONS_CSV = ORACLE_DIR / "full37_patch_loyo_predictions.csv"
ORACLE_METRICS_CSV = ORACLE_DIR / "full37_patch_loyo_metrics.csv"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "quick_nested_lod_m1_m2_sanity_check"
PREDICTIONS_CSV = OUTPUT_DIR / "nested_lod_m1_m2_sanity_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "nested_lod_m1_m2_sanity_metrics.csv"
COMPARISON_CSV = OUTPUT_DIR / "nested_lod_m1_m2_vs_oracle_z1_z2_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "nested_lod_m1_m2_sanity_summary.json"
TIMESERIES_PNG = OUTPUT_DIR / "nested_lod_m1_m2_vs_oracle_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "nested_lod_m1_m2_vs_oracle_scatter.png"

GROUP_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]


def ensure_runtime_on_compute_node():
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def corrcoef_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true, y_pred):
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


def rmse(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_sign_accuracy(obs, pred):
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs, pred):
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
    }


def compute_period_metrics(water_years, obs, pred):
    rows = []
    for group_name, selector in GROUP_SPECS:
        mask = selector(water_years)
        yy = obs[mask]
        pp = pred[mask]
        metrics = compute_metric_bundle(yy, pp)
        error = pp - yy
        rows.append(
            {
                "group_name": group_name,
                "n_years": int(mask.sum()),
                "r": metrics["r"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_error": float(np.mean(error)),
                "mean_abs_error": float(np.mean(np.abs(error))),
            }
        )
    return rows


def load_nested_m1_m2():
    pred_df = pd.read_csv(NESTED_PREDICTIONS_CSV)
    metrics_df = pd.read_csv(NESTED_METRICS_CSV)
    summary = json.loads(NESTED_SUMMARY_JSON.read_text())
    if "M1_M2" not in pred_df["model_name"].unique():
        raise ValueError("M1_M2 not found in nested predictions CSV.")
    if "M1_M2" not in metrics_df["model_name"].unique():
        raise ValueError("M1_M2 not found in nested metrics CSV.")
    nested_pred = pred_df.loc[pred_df["model_name"] == "M1_M2"].copy()
    nested_pred = nested_pred.rename(columns={"error": "error_pred_minus_obs"})
    nested_pred["residual_obs_minus_pred"] = nested_pred["obs_swe"] - nested_pred["pred_swe"]
    nested_pred["abs_error"] = np.abs(nested_pred["error_pred_minus_obs"])
    nested_pred["sign_correct"] = np.where(
        (nested_pred["obs_swe"] != 0.0) & (nested_pred["pred_swe"] != 0.0),
        (np.sign(nested_pred["obs_swe"]) == np.sign(nested_pred["pred_swe"])).astype(float),
        np.nan,
    )
    nested_pred["model_label"] = "nested_LOYO_M1_M2"
    nested_pred = nested_pred[
        [
            "model_label",
            "heldout_wy",
            "obs_swe",
            "pred_swe",
            "error_pred_minus_obs",
            "residual_obs_minus_pred",
            "abs_error",
            "sign_correct",
        ]
    ].sort_values("heldout_wy").reset_index(drop=True)
    return nested_pred, metrics_df.loc[metrics_df["model_name"] == "M1_M2"].iloc[0].to_dict(), summary


def load_oracle_z1_z2():
    pred_df = pd.read_csv(ORACLE_PREDICTIONS_CSV)
    metrics_df = pd.read_csv(ORACLE_METRICS_CSV)
    pred_df = pred_df.loc[
        (pred_df["patch_size"] == "exact_grid_cell") & (pred_df["model_name"] == "Z1_Z2")
    ].copy()
    pred_df = pred_df.rename(
        columns={
            "heldout_wy": "heldout_wy",
            "error": "error_pred_minus_obs",
        }
    )
    pred_df["residual_obs_minus_pred"] = pred_df["obs_swe"] - pred_df["pred_swe"]
    pred_df["model_label"] = "oracle_full37_Z1_Z2"
    pred_df = pred_df[
        [
            "model_label",
            "heldout_wy",
            "obs_swe",
            "pred_swe",
            "error_pred_minus_obs",
            "abs_error",
            "sign_correct",
            "residual_obs_minus_pred",
        ]
    ].sort_values("heldout_wy").reset_index(drop=True)
    metric_row = metrics_df.loc[
        (metrics_df["patch_size"] == "exact_grid_cell") & (metrics_df["model_name"] == "Z1_Z2")
    ].iloc[0].to_dict()
    return pred_df, metric_row


def draw_timeseries(nested_df, oracle_df):
    fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
    years = nested_df["heldout_wy"].to_numpy(dtype=np.int32)
    ax.plot(years, nested_df["obs_swe"].to_numpy(dtype=np.float64), color="black", linewidth=1.9, label="Observed SWE anomaly")
    ax.plot(years, nested_df["pred_swe"].to_numpy(dtype=np.float64), color="tab:red", linewidth=1.2, label="nested_LOYO_M1_M2")
    ax.plot(years, oracle_df["pred_swe"].to_numpy(dtype=np.float64), color="tab:blue", linewidth=1.2, label="oracle_full37_Z1_Z2")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Nested LOD M1+M2 vs oracle full37 exact Z1+Z2")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(TIMESERIES_PNG, dpi=220)
    plt.close(fig)


def draw_scatter(nested_df, oracle_df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, df, title, color in [
        (axes[0], nested_df, "nested_LOYO_M1_M2", "tab:red"),
        (axes[1], oracle_df, "oracle_full37_Z1_Z2", "tab:blue"),
    ]:
        obs = df["obs_swe"].to_numpy(dtype=np.float64)
        pred = df["pred_swe"].to_numpy(dtype=np.float64)
        ax.scatter(obs, pred, color=color, alpha=0.85)
        lo = float(min(np.min(obs), np.min(pred)))
        hi = float(max(np.max(obs), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Observed SWE anomaly")
        ax.set_ylabel("Predicted SWE anomaly")
        ax.grid(True, linewidth=0.25, color="0.85")
    fig.savefig(SCATTER_PNG, dpi=220)
    plt.close(fig)


def main():
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nested_df, nested_metric_row_raw, nested_summary = load_nested_m1_m2()
    oracle_df, oracle_metric_row_raw = load_oracle_z1_z2()

    merged_check = nested_df[["heldout_wy"]].merge(oracle_df[["heldout_wy"]], on="heldout_wy", how="outer", indicator=True)
    if not (merged_check["_merge"] == "both").all():
        raise ValueError("Nested and oracle prediction years do not align.")

    nested_obs = nested_df["obs_swe"].to_numpy(dtype=np.float64)
    nested_pred = nested_df["pred_swe"].to_numpy(dtype=np.float64)
    oracle_obs = oracle_df["obs_swe"].to_numpy(dtype=np.float64)
    oracle_pred = oracle_df["pred_swe"].to_numpy(dtype=np.float64)
    years = nested_df["heldout_wy"].to_numpy(dtype=np.int32)

    nested_metrics = compute_metric_bundle(nested_obs, nested_pred)
    oracle_metrics = compute_metric_bundle(oracle_obs, oracle_pred)
    nested_period_rows = compute_period_metrics(years, nested_obs, nested_pred)
    oracle_period_rows = compute_period_metrics(years, oracle_obs, oracle_pred)

    nested_predictions_out = nested_df.copy()
    oracle_predictions_out = oracle_df.copy()
    predictions_out = pd.concat([nested_predictions_out, oracle_predictions_out], ignore_index=True)
    predictions_out.to_csv(PREDICTIONS_CSV, index=False)

    metrics_rows = []
    for model_label, metric_bundle, source_metric_row in [
        ("nested_LOYO_M1_M2", nested_metrics, nested_metric_row_raw),
        ("oracle_full37_Z1_Z2", oracle_metrics, oracle_metric_row_raw),
    ]:
        metrics_rows.append(
            {
                "model_label": model_label,
                "r": metric_bundle["r"],
                "R2": metric_bundle["R2"],
                "RMSE": metric_bundle["RMSE"],
                "MAE": metric_bundle["MAE"],
                "sign_accuracy": metric_bundle["sign_accuracy"],
                "source_metrics_row": json.dumps(source_metric_row),
            }
        )
    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)

    comparison_rows = [
        {
            "model_label": "nested_LOYO_M1_M2",
            "feature_selection": "nested per held-out fold",
            "mode_selection_inside_loyo": bool(nested_summary.get("mode_selection_inside_loyo", False)),
            "uses_full37_selected_locations": False,
            "r": nested_metrics["r"],
            "R2": nested_metrics["R2"],
            "RMSE": nested_metrics["RMSE"],
            "MAE": nested_metrics["MAE"],
            "sign_accuracy": nested_metrics["sign_accuracy"],
            "notes": "Extracted from existing saved M1_M2 nested LOYO subset diagnostic.",
        },
        {
            "model_label": "oracle_full37_Z1_Z2",
            "feature_selection": "fixed full37 selected exact locations",
            "mode_selection_inside_loyo": False,
            "uses_full37_selected_locations": True,
            "r": oracle_metrics["r"],
            "R2": oracle_metrics["R2"],
            "RMSE": oracle_metrics["RMSE"],
            "MAE": oracle_metrics["MAE"],
            "sign_accuracy": oracle_metrics["sign_accuracy"],
            "notes": "Oracle reference using full37 selected exact Z1/Z2 locations.",
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_CSV, index=False)

    draw_timeseries(nested_df, oracle_df)
    draw_scatter(nested_df, oracle_df)

    r2_gap = float(nested_metrics["R2"] - oracle_metrics["R2"])
    rmse_gap = float(nested_metrics["RMSE"] - oracle_metrics["RMSE"])
    if (nested_metrics["R2"] >= oracle_metrics["R2"] - 0.05) and (nested_metrics["RMSE"] <= oracle_metrics["RMSE"] + 0.005):
        short_answer = (
            "The real nested LOD M1+M2 retains most of the oracle full-37 Z1+Z2 skill, so the feature discovery itself appears reasonably stable."
        )
    elif (nested_metrics["R2"] < oracle_metrics["R2"] - 0.15) or (nested_metrics["RMSE"] > oracle_metrics["RMSE"] + 0.01):
        short_answer = (
            "The real nested LOD M1+M2 does not retain the oracle full-37 Z1+Z2 skill. "
            "The apparent Z1+Z2 predictability depends on knowing the full-37 selected feature locations, so feature-selection instability remains the main limitation."
        )
    else:
        short_answer = "The real nested LOD M1+M2 retains some but not all of the oracle full-37 Z1+Z2 skill."

    summary_payload = {
        "input_files": {
            "nested_predictions_csv": str(NESTED_PREDICTIONS_CSV),
            "nested_metrics_csv": str(NESTED_METRICS_CSV),
            "nested_summary_json": str(NESTED_SUMMARY_JSON),
            "oracle_predictions_csv": str(ORACLE_PREDICTIONS_CSV),
            "oracle_metrics_csv": str(ORACLE_METRICS_CSV),
        },
        "output_dir": str(OUTPUT_DIR),
        "whether_existing_nested_m1_m2_was_found": True,
        "mode_selection_inside_loyo": bool(nested_summary.get("mode_selection_inside_loyo", False)),
        "nested_m1_m2_metrics": nested_metrics,
        "nested_m1_m2_period_metrics": nested_period_rows,
        "oracle_z1_z2_metrics": oracle_metrics,
        "oracle_z1_z2_period_metrics": oracle_period_rows,
        "main_comparison": {
            "r2_gap_nested_minus_oracle": r2_gap,
            "rmse_gap_nested_minus_oracle": rmse_gap,
            "nested_model_label": "nested_LOYO_M1_M2",
            "oracle_model_label": "oracle_full37_Z1_Z2",
        },
        "short_answer": short_answer,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    print("Output directory: {}".format(OUTPUT_DIR))
    print("Nested M1+M2 metrics:")
    print(json.dumps(nested_metrics, indent=2))
    print("Oracle Z1+Z2 metrics:")
    print(json.dumps(oracle_metrics, indent=2))
    print("Short answer:")
    print(short_answer)


if __name__ == "__main__":
    main()
