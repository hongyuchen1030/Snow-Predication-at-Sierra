#!/usr/bin/env python3
"""Selected-column year dependency diagnostic for COBE2 Sierra SWE LOD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECIAL_YEARS = [2011, 2015, 2019, 1993, 2006, 2005, 2012, 2021]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def compute_metrics(obs: Iterable[float], pred: Iterable[float]) -> Dict[str, float]:
    obs_arr = np.asarray(list(obs), dtype=float)
    pred_arr = np.asarray(list(pred), dtype=float)
    n = int(obs_arr.size)
    if n == 0:
        return {
            "n": 0,
            "r": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "sign_accuracy": np.nan,
            "mean_abs_error": np.nan,
            "median_abs_error": np.nan,
            "mean_error": np.nan,
        }

    err = pred_arr - obs_arr
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    sign_accuracy = float(np.mean(np.sign(pred_arr) == np.sign(obs_arr)))
    mean_error = float(np.mean(err))
    mean_abs_error = float(np.mean(np.abs(err)))
    median_abs_error = float(np.median(np.abs(err)))
    denom = float(np.sum((obs_arr - np.mean(obs_arr)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom != 0.0 else np.nan
    r = (
        float(np.corrcoef(obs_arr, pred_arr)[0, 1])
        if n > 1 and np.std(obs_arr) > 0 and np.std(pred_arr) > 0
        else np.nan
    )
    return {
        "n": n,
        "r": r,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "sign_accuracy": sign_accuracy,
        "mean_abs_error": mean_abs_error,
        "median_abs_error": median_abs_error,
        "mean_error": mean_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/cobe2_sierra_swe_lod_setup"),
        help="Base directory containing existing LOD artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/cobe2_sierra_swe_lod_setup/selected_column_year_dependency_diagnostic"
        ),
        help="Output directory for diagnostic products.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nested_modes_path = (
        base_dir / "nested_loyo_mode_stability_audit" / "nested_loyo_selected_modes_by_fold.csv"
    )
    nested_preds_path = (
        base_dir / "loyo_mode_subset_diagnostic" / "loyo_mode_subset_predictions.csv"
    )
    oracle_preds_path = (
        base_dir
        / "full37_selected_patch_predictor_loyo"
        / "full37_patch_loyo_predictions.csv"
    )
    full37_patch_predictors_path = (
        base_dir
        / "full37_selected_patch_predictor_loyo"
        / "full37_patch_predictors.csv"
    )
    selection_match_by_fold_path = (
        base_dir
        / "selection_instability_error_diagnostic"
        / "selection_match_by_fold.csv"
    )
    selection_summary_path = (
        base_dir
        / "selection_instability_error_diagnostic"
        / "selection_match_diagnostic_summary.json"
    )

    with selection_summary_path.open() as f:
        selection_summary = json.load(f)

    full37_reference_modes = selection_summary["full37_reference_modes"]
    z1_ref = full37_reference_modes["1"]
    z2_ref = full37_reference_modes["2"]

    nested_modes = pd.read_csv(nested_modes_path)
    nested_modes = nested_modes[nested_modes["mode"].isin([1, 2])].copy()
    nested_modes["lag_month"] = nested_modes["lag_month"].astype(str)
    mode1 = nested_modes[nested_modes["mode"] == 1].copy()
    mode2 = nested_modes[nested_modes["mode"] == 2].copy()

    mode1 = mode1.rename(
        columns={
            "lag_month": "M1_month",
            "lat": "M1_lat",
            "lon": "M1_lon",
            "selected_corr": "M1_corr",
            "delta_R2": "M1_delta_R2",
            "pred_swe": "nested_pred_M1_M2",
            "obs_swe": "obs_swe",
        }
    )[
        [
            "heldout_wy",
            "M1_month",
            "M1_lat",
            "M1_lon",
            "M1_corr",
            "M1_delta_R2",
            "nested_pred_M1_M2",
            "obs_swe",
        ]
    ]
    mode2 = mode2.rename(
        columns={
            "lag_month": "M2_month",
            "lat": "M2_lat",
            "lon": "M2_lon",
            "selected_corr": "M2_corr",
            "delta_R2": "M2_delta_R2",
        }
    )[
        ["heldout_wy", "M2_month", "M2_lat", "M2_lon", "M2_corr", "M2_delta_R2"]
    ]

    by_fold = mode1.merge(mode2, on="heldout_wy", how="inner", validate="one_to_one")

    nested_preds = pd.read_csv(nested_preds_path)
    nested_preds = nested_preds[nested_preds["model_name"] == "M1_M2"].copy()
    nested_preds = nested_preds.rename(
        columns={
            "pred_swe": "nested_pred_M1_M2_from_predictions",
            "error": "nested_error_M1_M2",
        }
    )
    nested_preds["nested_abs_error_M1_M2"] = nested_preds["nested_error_M1_M2"].abs()
    nested_preds["nested_sign_correct_M1_M2"] = (
        np.sign(nested_preds["nested_pred_M1_M2_from_predictions"])
        == np.sign(nested_preds["obs_swe"])
    ).astype(int)
    nested_preds = nested_preds[
        [
            "heldout_wy",
            "obs_swe",
            "nested_pred_M1_M2_from_predictions",
            "nested_error_M1_M2",
            "nested_abs_error_M1_M2",
            "nested_sign_correct_M1_M2",
        ]
    ]

    by_fold = by_fold.merge(
        nested_preds,
        on=["heldout_wy", "obs_swe"],
        how="left",
        validate="one_to_one",
    )
    by_fold["nested_pred_M1_M2"] = by_fold["nested_pred_M1_M2_from_predictions"].fillna(
        by_fold["nested_pred_M1_M2"]
    )
    by_fold = by_fold.drop(columns=["nested_pred_M1_M2_from_predictions"])

    oracle_preds = pd.read_csv(oracle_preds_path)
    oracle_preds = oracle_preds[
        (oracle_preds["patch_size"] == "exact_grid_cell")
        & (oracle_preds["model_name"] == "Z1_Z2")
    ].copy()
    oracle_preds = oracle_preds.rename(
        columns={
            "pred_swe": "oracle_pred_Z1_Z2",
            "error": "oracle_error_Z1_Z2",
            "abs_error": "oracle_abs_error_Z1_Z2",
            "sign_correct": "oracle_sign_correct_Z1_Z2",
        }
    )[
        [
            "heldout_wy",
            "oracle_pred_Z1_Z2",
            "oracle_error_Z1_Z2",
            "oracle_abs_error_Z1_Z2",
            "oracle_sign_correct_Z1_Z2",
        ]
    ]
    by_fold = by_fold.merge(oracle_preds, on="heldout_wy", how="left", validate="one_to_one")

    full37_patch_predictors = pd.read_csv(full37_patch_predictors_path)
    predictor_cols = [
        c
        for c in full37_patch_predictors.columns
        if c.startswith("Z1_") or c.startswith("Z2_")
    ]
    if len(predictor_cols) != 2:
        raise RuntimeError(
            "Expected exactly two full37 exact-grid predictor columns, found "
            f"{predictor_cols}"
        )

    # The exact-grid predictor names encode the full-37 references.
    by_fold["Z1_month"] = z1_ref["lag_month"]
    by_fold["Z1_lat"] = z1_ref["lat"]
    by_fold["Z1_lon"] = z1_ref["lon"]
    by_fold["Z2_month"] = z2_ref["lag_month"]
    by_fold["Z2_lat"] = z2_ref["lat"]
    by_fold["Z2_lon"] = z2_ref["lon"]

    by_fold["M1_exact_Z1"] = (
        (by_fold["M1_month"] == by_fold["Z1_month"])
        & np.isclose(by_fold["M1_lat"], by_fold["Z1_lat"])
        & np.isclose(by_fold["M1_lon"], by_fold["Z1_lon"])
    )
    by_fold["M2_exact_Z2"] = (
        (by_fold["M2_month"] == by_fold["Z2_month"])
        & np.isclose(by_fold["M2_lat"], by_fold["Z2_lat"])
        & np.isclose(by_fold["M2_lon"], by_fold["Z2_lon"])
    )
    by_fold["ordered_both_exact"] = by_fold["M1_exact_Z1"] & by_fold["M2_exact_Z2"]
    by_fold["any_exact"] = by_fold["M1_exact_Z1"] | by_fold["M2_exact_Z2"]
    by_fold["num_exact"] = (
        by_fold["M1_exact_Z1"].astype(int) + by_fold["M2_exact_Z2"].astype(int)
    )
    by_fold["M1_same_month_Z1"] = by_fold["M1_month"] == by_fold["Z1_month"]
    by_fold["M2_same_month_Z2"] = by_fold["M2_month"] == by_fold["Z2_month"]

    by_fold["M1_distance_to_Z1_km"] = by_fold.apply(
        lambda row: haversine_km(row["M1_lat"], row["M1_lon"], row["Z1_lat"], row["Z1_lon"]),
        axis=1,
    )
    by_fold["M2_distance_to_Z2_km"] = by_fold.apply(
        lambda row: haversine_km(row["M2_lat"], row["M2_lon"], row["Z2_lat"], row["Z2_lon"]),
        axis=1,
    )
    by_fold["max_distance_to_reference_km"] = by_fold[
        ["M1_distance_to_Z1_km", "M2_distance_to_Z2_km"]
    ].max(axis=1)
    by_fold["mean_distance_to_reference_km"] = by_fold[
        ["M1_distance_to_Z1_km", "M2_distance_to_Z2_km"]
    ].mean(axis=1)

    def unordered_match(row: pd.Series) -> bool:
        selected = {
            (row["M1_month"], float(row["M1_lat"]), float(row["M1_lon"])),
            (row["M2_month"], float(row["M2_lat"]), float(row["M2_lon"])),
        }
        reference = {
            (row["Z1_month"], float(row["Z1_lat"]), float(row["Z1_lon"])),
            (row["Z2_month"], float(row["Z2_lat"]), float(row["Z2_lon"])),
        }
        return selected == reference

    by_fold["unordered_both_exact"] = by_fold.apply(unordered_match, axis=1)
    by_fold["notes"] = np.where(
        by_fold["ordered_both_exact"],
        "ordered exact match to full-37 Z1/Z2",
        np.where(
            by_fold["unordered_both_exact"],
            "unordered exact set match only",
            "nested fold changed at least one reference column",
        ),
    )

    by_fold["dependency_label"] = np.where(
        by_fold["ordered_both_exact"],
        "removing_year_keeps_Z1_Z2",
        np.where(
            by_fold["num_exact"] == 1,
            "removing_year_loses_one_reference_column",
            "removing_year_loses_both_reference_columns",
        ),
    )

    nested_top5_years = (
        by_fold.nlargest(5, "nested_abs_error_M1_M2")["heldout_wy"].astype(int).tolist()
    )
    oracle_top5_years = (
        by_fold.nlargest(5, "oracle_abs_error_Z1_Z2")["heldout_wy"].astype(int).tolist()
    )
    by_fold["is_top5_nested_error"] = by_fold["heldout_wy"].isin(nested_top5_years)
    by_fold["is_top5_oracle_error"] = by_fold["heldout_wy"].isin(oracle_top5_years)
    by_fold["is_structural_selection_failure"] = ~by_fold["ordered_both_exact"]
    by_fold["is_prediction_failure"] = by_fold["is_top5_nested_error"]
    by_fold["interpretation"] = np.select(
        [
            by_fold["is_structural_selection_failure"] & by_fold["is_prediction_failure"],
            by_fold["is_structural_selection_failure"],
            by_fold["is_prediction_failure"],
        ],
        [
            "both structural selection failure and hard prediction year",
            "structural selection failure but not top prediction error",
            "hard prediction year but selected columns still match",
        ],
        default="neither top structural nor top prediction failure",
    )

    ordered_columns = [
        "heldout_wy",
        "obs_swe",
        "nested_pred_M1_M2",
        "nested_error_M1_M2",
        "nested_abs_error_M1_M2",
        "nested_sign_correct_M1_M2",
        "oracle_pred_Z1_Z2",
        "oracle_error_Z1_Z2",
        "oracle_abs_error_Z1_Z2",
        "oracle_sign_correct_Z1_Z2",
        "M1_month",
        "M1_lat",
        "M1_lon",
        "M1_corr",
        "M1_delta_R2",
        "M2_month",
        "M2_lat",
        "M2_lon",
        "M2_corr",
        "M2_delta_R2",
        "Z1_month",
        "Z1_lat",
        "Z1_lon",
        "Z2_month",
        "Z2_lat",
        "Z2_lon",
        "M1_exact_Z1",
        "M2_exact_Z2",
        "ordered_both_exact",
        "any_exact",
        "num_exact",
        "M1_same_month_Z1",
        "M2_same_month_Z2",
        "M1_distance_to_Z1_km",
        "M2_distance_to_Z2_km",
        "max_distance_to_reference_km",
        "mean_distance_to_reference_km",
        "unordered_both_exact",
        "notes",
    ]
    by_fold = by_fold.sort_values("heldout_wy").reset_index(drop=True)
    by_fold[ordered_columns].to_csv(
        output_dir / "selected_column_year_dependency_by_fold.csv", index=False
    )

    group_specs: List[Tuple[str, object, pd.Series]] = [
        ("ordered_both_exact", True, by_fold["ordered_both_exact"]),
        ("ordered_both_exact", False, ~by_fold["ordered_both_exact"]),
        ("num_exact", 0, by_fold["num_exact"] == 0),
        ("num_exact", 1, by_fold["num_exact"] == 1),
        ("num_exact", 2, by_fold["num_exact"] == 2),
        ("M1_exact_Z1", True, by_fold["M1_exact_Z1"]),
        ("M1_exact_Z1", False, ~by_fold["M1_exact_Z1"]),
        ("M2_exact_Z2", True, by_fold["M2_exact_Z2"]),
        ("M2_exact_Z2", False, ~by_fold["M2_exact_Z2"]),
        (
            "both_same_month",
            True,
            by_fold["M1_same_month_Z1"] & by_fold["M2_same_month_Z2"],
        ),
        (
            "both_same_month",
            False,
            ~(by_fold["M1_same_month_Z1"] & by_fold["M2_same_month_Z2"]),
        ),
    ]

    group_rows: List[Dict[str, object]] = []
    for group_name, group_value, mask in group_specs:
        subset = by_fold.loc[mask]
        metrics = compute_metrics(subset["obs_swe"], subset["nested_pred_M1_M2"])
        group_rows.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                **metrics,
            }
        )
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(output_dir / "selected_column_year_dependency_error_groups.csv", index=False)

    metrics_df = by_fold[
        [
            "heldout_wy",
            "dependency_label",
            "num_exact",
            "ordered_both_exact",
            "nested_abs_error_M1_M2",
            "oracle_abs_error_Z1_Z2",
            "is_top5_nested_error",
            "is_top5_oracle_error",
            "is_structural_selection_failure",
            "is_prediction_failure",
            "interpretation",
        ]
    ].copy()
    metrics_df.to_csv(output_dir / "selected_column_year_dependency_metrics.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    color_map = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}
    marker_map = {0: "x", 1: "s", 2: "o"}

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for num_exact in [0, 1, 2]:
        subset = by_fold[by_fold["num_exact"] == num_exact]
        ax.scatter(
            subset["max_distance_to_reference_km"],
            subset["nested_abs_error_M1_M2"],
            color=color_map[num_exact],
            marker=marker_map[num_exact],
            s=60,
            alpha=0.9,
            label=f"num_exact = {num_exact}",
        )
    for _, row in by_fold[by_fold["heldout_wy"].isin(SPECIAL_YEARS)].iterrows():
        ax.annotate(
            str(int(row["heldout_wy"])),
            (row["max_distance_to_reference_km"], row["nested_abs_error_M1_M2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Max distance to full-37 reference columns (km)")
    ax.set_ylabel("Nested |prediction error| for M1+M2 (m)")
    ax.set_title("Selected-column mismatch vs nested M1+M2 error")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_column_match_vs_error.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
    years = by_fold["heldout_wy"].astype(int)
    ax1.plot(years, by_fold["nested_abs_error_M1_M2"], color="#1f77b4", linewidth=2.2, label="Nested M1+M2 abs error")
    ax1.plot(years, by_fold["oracle_abs_error_Z1_Z2"], color="#444444", linewidth=1.8, linestyle="--", label="Oracle Z1+Z2 abs error")
    ax1.set_xlabel("Held-out water year")
    ax1.set_ylabel("Absolute error (m)")
    ax2 = ax1.twinx()
    ax2.scatter(
        years,
        by_fold["num_exact"],
        c=by_fold["ordered_both_exact"].map({True: "#2ca02c", False: "#d62728"}),
        marker="o",
        s=48,
        label="num_exact / ordered match",
        zorder=3,
    )
    ax2.set_ylabel("Number of exact Z1/Z2 matches")
    ax2.set_yticks([0, 1, 2])
    for _, row in by_fold[by_fold["heldout_wy"].isin(SPECIAL_YEARS)].iterrows():
        ax1.annotate(
            str(int(row["heldout_wy"])),
            (row["heldout_wy"], row["nested_abs_error_M1_M2"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.set_title("Held-out year dependency timeline")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_column_dependency_timeline.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
    bar_colors = by_fold["num_exact"].map(color_map)
    ax1.bar(years, by_fold["num_exact"], color=bar_colors, width=0.8)
    ax1.set_xlabel("Held-out water year")
    ax1.set_ylabel("Number of exact matches to full-37 Z1/Z2")
    ax1.set_yticks([0, 1, 2])
    ax2 = ax1.twinx()
    ax2.plot(years, by_fold["nested_abs_error_M1_M2"], color="black", linewidth=2.0, marker="o", markersize=4.0)
    ax2.set_ylabel("Nested M1+M2 abs error (m)")
    for _, row in by_fold[by_fold["heldout_wy"].isin(SPECIAL_YEARS)].iterrows():
        ax2.annotate(
            str(int(row["heldout_wy"])),
            (row["heldout_wy"], row["nested_abs_error_M1_M2"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax1.set_title("Effect of removing each year on selected reference columns")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_column_removed_year_effect.png", dpi=200)
    plt.close(fig)

    def plot_mode_map(mode_prefix: str, ref: Dict[str, object], output_name: str) -> None:
        lat_col = f"{mode_prefix}_lat"
        lon_col = f"{mode_prefix}_lon"
        exact_col = f"{mode_prefix}_exact_Z{1 if mode_prefix == 'M1' else 2}"
        month_col = f"{mode_prefix}_month"
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        exact_subset = by_fold[by_fold[exact_col]]
        mismatch_subset = by_fold[~by_fold[exact_col]]
        ax.scatter(
            exact_subset[lon_col],
            exact_subset[lat_col],
            color="#2ca02c",
            marker="o",
            s=46,
            alpha=0.85,
            label="Exact-match folds",
        )
        ax.scatter(
            mismatch_subset[lon_col],
            mismatch_subset[lat_col],
            color="#d62728",
            marker="x",
            s=58,
            alpha=0.9,
            label="Mismatch folds",
        )
        ax.scatter(
            [ref["lon"]],
            [ref["lat"]],
            color="gold",
            edgecolor="black",
            marker="*",
            s=220,
            linewidth=0.8,
            label=f"Full-37 reference {mode_prefix.replace('M', 'Z')}",
            zorder=4,
        )
        for _, row in by_fold[by_fold["heldout_wy"].isin(SPECIAL_YEARS)].iterrows():
            ax.annotate(
                f"{int(row['heldout_wy'])}-{row[month_col]}",
                (row[lon_col], row[lat_col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{mode_prefix} selected locations by held-out year")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / output_name, dpi=200)
        plt.close(fig)

    plot_mode_map("M1", z1_ref, "selected_column_mismatch_map_mode1.png")
    plot_mode_map("M2", z2_ref, "selected_column_mismatch_map_mode2.png")

    years_losing_both = (
        by_fold.loc[by_fold["num_exact"] == 0, "heldout_wy"].astype(int).tolist()
    )
    years_losing_one = (
        by_fold.loc[by_fold["num_exact"] == 1, "heldout_wy"].astype(int).tolist()
    )

    special_years_report = []
    for year in SPECIAL_YEARS:
        row = by_fold.loc[by_fold["heldout_wy"] == year]
        if row.empty:
            continue
        rec = row.iloc[0]
        special_years_report.append(
            {
                "heldout_wy": int(rec["heldout_wy"]),
                "dependency_label": rec["dependency_label"],
                "M1_exact_Z1": bool(rec["M1_exact_Z1"]),
                "M2_exact_Z2": bool(rec["M2_exact_Z2"]),
                "num_exact": int(rec["num_exact"]),
                "nested_abs_error_M1_M2": float(rec["nested_abs_error_M1_M2"]),
                "oracle_abs_error_Z1_Z2": float(rec["oracle_abs_error_Z1_Z2"]),
                "interpretation": rec["interpretation"],
            }
        )

    nested_top5_mismatch_count = int(
        by_fold.loc[by_fold["is_top5_nested_error"], "is_structural_selection_failure"].sum()
    )
    if nested_top5_mismatch_count >= 3:
        short_answer = (
            "The nested M1+M2 errors have mixed causes leaning strongly toward "
            "feature-selection instability: several of the worst held-out years lose "
            "one or both full-37 reference columns when that year is removed."
        )
    elif nested_top5_mismatch_count <= 1:
        short_answer = (
            "The worst nested M1+M2 prediction errors are not primarily caused by "
            "losing the full-37 reference columns. The hardest years stay hard even "
            "when the selected columns still match."
        )
    else:
        short_answer = (
            "The nested M1+M2 errors have mixed causes: some years are hard because "
            "the selected columns change when the year is removed, while others remain "
            "hard even when the reference columns are retained."
        )

    summary = {
        "input_files": {
            "nested_selected_modes_csv": str(nested_modes_path),
            "nested_predictions_csv": str(nested_preds_path),
            "oracle_predictions_csv": str(oracle_preds_path),
            "full37_patch_predictors_csv": str(full37_patch_predictors_path),
            "selection_match_by_fold_csv": str(selection_match_by_fold_path),
            "selection_summary_json": str(selection_summary_path),
        },
        "output_dir": str(output_dir),
        "full37_Z1": z1_ref,
        "full37_Z2": z2_ref,
        "num_folds": int(len(by_fold)),
        "num_M1_exact_Z1": int(by_fold["M1_exact_Z1"].sum()),
        "num_M2_exact_Z2": int(by_fold["M2_exact_Z2"].sum()),
        "num_ordered_both_exact": int(by_fold["ordered_both_exact"].sum()),
        "num_unordered_both_exact": int(by_fold["unordered_both_exact"].sum()),
        "error_metrics_by_match_group": group_df.to_dict(orient="records"),
        "top5_nested_error_years": nested_top5_years,
        "top5_oracle_error_years": oracle_top5_years,
        "years_losing_both_reference_columns": years_losing_both,
        "years_losing_one_reference_column": years_losing_one,
        "special_years_report": special_years_report,
        "short_answer": short_answer,
    }
    with (output_dir / "selected_column_year_dependency_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Output directory: {output_dir}")
    print("Focused dependency summary:")
    print(
        by_fold[
            [
                "heldout_wy",
                "dependency_label",
                "num_exact",
                "nested_abs_error_M1_M2",
                "oracle_abs_error_Z1_Z2",
            ]
        ]
        .to_string(index=False)
    )
    print("Top 5 nested-error years:", nested_top5_years)
    print("Top 5 oracle-error years:", oracle_top5_years)
    print("Short answer:", short_answer)


if __name__ == "__main__":
    main()
