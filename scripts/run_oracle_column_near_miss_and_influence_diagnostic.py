#!/usr/bin/env python3
"""Refined near-miss and oracle influence diagnostic for COBE2 LOD columns."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECIAL_YEARS = [2011, 2015, 2019, 1993, 2006]
QUALITY_ORDER = ["exact", "near_miss", "acceptable_same_month", "ambiguous", "wrong"]
QUALITY_COLORS = {
    "exact": "#2ca02c",
    "near_miss": "#1f77b4",
    "acceptable_same_month": "#ff7f0e",
    "ambiguous": "#9467bd",
    "wrong": "#d62728",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    denom = float(np.sum((obs_arr - np.mean(obs_arr)) ** 2))
    return {
        "n": n,
        "r": float(np.corrcoef(obs_arr, pred_arr)[0, 1])
        if n > 1 and np.std(obs_arr) > 0 and np.std(pred_arr) > 0
        else np.nan,
        "R2": float(1.0 - np.sum(err**2) / denom) if denom != 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "sign_accuracy": float(np.mean(np.sign(pred_arr) == np.sign(obs_arr))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "median_abs_error": float(np.median(np.abs(err))),
        "mean_error": float(np.mean(err)),
    }


def corr_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def linear_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    return y - fitted


def classify_selection(distance_km: float, same_month: bool, exact_match: bool) -> Dict[str, object]:
    near_miss = same_month and distance_km <= 500.0 and not exact_match
    same_month_moderate = same_month and 500.0 < distance_km <= 1500.0
    same_month_far = same_month and distance_km > 1500.0
    wrong_month_near = (not same_month) and distance_km <= 500.0
    wrong_month_moderate = (not same_month) and 500.0 < distance_km <= 1500.0
    true_wrong_column = (not same_month) and distance_km > 1500.0

    if exact_match:
        quality = "exact"
    elif same_month and distance_km <= 500.0:
        quality = "near_miss"
    elif same_month and distance_km <= 1500.0:
        quality = "acceptable_same_month"
    elif (not same_month) and distance_km > 1500.0:
        quality = "wrong"
    else:
        quality = "ambiguous"

    return {
        "near_miss": near_miss,
        "same_month_moderate": same_month_moderate,
        "same_month_far": same_month_far,
        "wrong_month_near": wrong_month_near,
        "wrong_month_moderate": wrong_month_moderate,
        "true_wrong_column": true_wrong_column,
        "selection_quality": quality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/cobe2_sierra_swe_lod_setup"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/cobe2_sierra_swe_lod_setup/oracle_column_near_miss_and_influence_diagnostic"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dependency_path = (
        base_dir
        / "selected_column_year_dependency_diagnostic"
        / "selected_column_year_dependency_by_fold.csv"
    )
    predictors_path = (
        base_dir
        / "full37_selected_patch_predictor_loyo"
        / "full37_patch_predictors.csv"
    )
    summary_path = (
        base_dir
        / "selection_instability_error_diagnostic"
        / "selection_match_diagnostic_summary.json"
    )

    by_fold = pd.read_csv(dependency_path)
    predictors = pd.read_csv(predictors_path)
    predictors = predictors[predictors["patch_size"] == "exact_grid_cell"].copy()
    with summary_path.open() as f:
        selection_summary = json.load(f)

    z1_ref = selection_summary["full37_reference_modes"]["1"]
    z2_ref = selection_summary["full37_reference_modes"]["2"]

    by_fold["M1_distance_to_Z1_km"] = by_fold.apply(
        lambda row: haversine_km(row["M1_lat"], row["M1_lon"], z1_ref["lat"], z1_ref["lon"]),
        axis=1,
    )
    by_fold["M2_distance_to_Z2_km"] = by_fold.apply(
        lambda row: haversine_km(row["M2_lat"], row["M2_lon"], z2_ref["lat"], z2_ref["lon"]),
        axis=1,
    )
    by_fold["M1_same_month_Z1"] = by_fold["M1_month"] == z1_ref["lag_month"]
    by_fold["M2_same_month_Z2"] = by_fold["M2_month"] == z2_ref["lag_month"]

    m1_class = by_fold.apply(
        lambda row: classify_selection(
            row["M1_distance_to_Z1_km"], bool(row["M1_same_month_Z1"]), bool(row["M1_exact_Z1"])
        ),
        axis=1,
        result_type="expand",
    )
    m2_class = by_fold.apply(
        lambda row: classify_selection(
            row["M2_distance_to_Z2_km"], bool(row["M2_same_month_Z2"]), bool(row["M2_exact_Z2"])
        ),
        axis=1,
        result_type="expand",
    )

    by_fold["M1_selection_quality"] = m1_class["selection_quality"]
    by_fold["M2_selection_quality"] = m2_class["selection_quality"]
    for col in [
        "near_miss",
        "same_month_moderate",
        "same_month_far",
        "wrong_month_near",
        "wrong_month_moderate",
        "true_wrong_column",
    ]:
        by_fold[f"M1_{col}"] = m1_class[col]
        by_fold[f"M2_{col}"] = m2_class[col]

    by_fold["both_exact"] = by_fold["M1_selection_quality"].eq("exact") & by_fold[
        "M2_selection_quality"
    ].eq("exact")
    by_fold["both_exact_or_near"] = by_fold["M1_selection_quality"].isin(
        ["exact", "near_miss"]
    ) & by_fold["M2_selection_quality"].isin(["exact", "near_miss"])
    by_fold["both_acceptable"] = by_fold["M1_selection_quality"].isin(
        ["exact", "near_miss", "acceptable_same_month"]
    ) & by_fold["M2_selection_quality"].isin(["exact", "near_miss", "acceptable_same_month"])
    by_fold["any_wrong"] = by_fold["M1_selection_quality"].eq("wrong") | by_fold[
        "M2_selection_quality"
    ].eq("wrong")
    by_fold["both_wrong"] = by_fold["M1_selection_quality"].eq("wrong") & by_fold[
        "M2_selection_quality"
    ].eq("wrong")
    by_fold["same_month_count"] = (
        by_fold["M1_same_month_Z1"].astype(int) + by_fold["M2_same_month_Z2"].astype(int)
    )
    by_fold["max_distance_km"] = by_fold[
        ["M1_distance_to_Z1_km", "M2_distance_to_Z2_km"]
    ].max(axis=1)
    by_fold["mean_distance_km"] = by_fold[
        ["M1_distance_to_Z1_km", "M2_distance_to_Z2_km"]
    ].mean(axis=1)

    near_cols = [
        "heldout_wy",
        "obs_swe",
        "nested_pred_M1_M2",
        "nested_abs_error_M1_M2",
        "oracle_pred_Z1_Z2",
        "oracle_abs_error_Z1_Z2",
        "M1_month",
        "M1_lat",
        "M1_lon",
        "M1_distance_to_Z1_km",
        "M1_same_month_Z1",
        "M1_selection_quality",
        "M2_month",
        "M2_lat",
        "M2_lon",
        "M2_distance_to_Z2_km",
        "M2_same_month_Z2",
        "M2_selection_quality",
        "both_exact",
        "both_exact_or_near",
        "both_acceptable",
        "any_wrong",
        "both_wrong",
        "same_month_count",
        "max_distance_km",
        "mean_distance_km",
    ]
    by_fold[near_cols].sort_values("heldout_wy").to_csv(
        output_dir / "oracle_column_near_miss_by_fold.csv", index=False
    )

    wrong_years = by_fold[by_fold["any_wrong"]].copy()
    wrong_years["interpretation"] = np.select(
        [
            wrong_years["both_wrong"],
            wrong_years["M1_selection_quality"].eq("wrong"),
            wrong_years["M2_selection_quality"].eq("wrong"),
        ],
        [
            "both modes selected different month and far location",
            "mode 1 selected different month and far location",
            "mode 2 selected different month and far location",
        ],
        default="not true wrong column",
    )
    wrong_cols = [
        "heldout_wy",
        "nested_abs_error_M1_M2",
        "oracle_abs_error_Z1_Z2",
        "M1_selection_quality",
        "M1_month",
        "M1_lat",
        "M1_lon",
        "M1_distance_to_Z1_km",
        "M2_selection_quality",
        "M2_month",
        "M2_lat",
        "M2_lon",
        "M2_distance_to_Z2_km",
        "same_month_count",
        "interpretation",
    ]
    wrong_years = wrong_years.sort_values("nested_abs_error_M1_M2", ascending=False)
    wrong_years[wrong_cols].to_csv(
        output_dir / "oracle_column_wrong_selection_years.csv", index=False
    )

    group_specs = []
    for col in ["both_exact", "both_exact_or_near", "both_acceptable", "any_wrong", "both_wrong"]:
        for val in [True, False]:
            group_specs.append((col, str(val), by_fold[col] == val))
    for val in [0, 1, 2]:
        group_specs.append(("same_month_count", str(val), by_fold["same_month_count"] == val))

    group_rows: List[Dict[str, object]] = []
    for group_name, group_value, mask in group_specs:
        subset = by_fold.loc[mask]
        metrics = compute_metrics(subset["obs_swe"], subset["nested_pred_M1_M2"])
        group_rows.append({"group_name": group_name, "group_value": group_value, **metrics})
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(output_dir / "oracle_column_near_miss_error_groups.csv", index=False)

    # Leave-one-year influence using saved oracle columns.
    z1_col = [c for c in predictors.columns if c.startswith("Z1_")][0]
    z2_col = [c for c in predictors.columns if c.startswith("Z2_")][0]
    merged = by_fold.merge(
        predictors[["water_year", z1_col, z2_col]],
        left_on="heldout_wy",
        right_on="water_year",
        how="left",
        validate="one_to_one",
    ).sort_values("heldout_wy")
    merged = merged.rename(columns={z1_col: "oracle_Z1_value", z2_col: "oracle_Z2_value"})

    obs_all = merged["obs_swe"].to_numpy(dtype=float)
    z1_all = merged["oracle_Z1_value"].to_numpy(dtype=float)
    z2_all = merged["oracle_Z2_value"].to_numpy(dtype=float)
    z1_corr_full = corr_or_nan(z1_all, obs_all)
    resid_full = linear_residual(obs_all, z1_all)
    z2_corr_full = corr_or_nan(z2_all, resid_full)

    influence_rows = []
    for idx, row in merged.iterrows():
        mask = merged["heldout_wy"] != row["heldout_wy"]
        train = merged.loc[mask]
        y_train = train["obs_swe"].to_numpy(dtype=float)
        z1_train = train["oracle_Z1_value"].to_numpy(dtype=float)
        z2_train = train["oracle_Z2_value"].to_numpy(dtype=float)
        z1_corr_wo = corr_or_nan(z1_train, y_train)
        resid_wo = linear_residual(y_train, z1_train)
        z2_corr_wo = corr_or_nan(z2_train, resid_wo)
        influence_rows.append(
            {
                "heldout_wy": int(row["heldout_wy"]),
                "Z1_corr_full37": z1_corr_full,
                "Z1_corr_without_y": z1_corr_wo,
                "delta_Z1_corr": z1_corr_wo - z1_corr_full,
                "selected_M1_abs_corr": abs(float(row["M1_corr"])),
                "oracle_Z1_abs_corr_without_y": abs(z1_corr_wo),
                "Z1_score_gap": abs(float(row["M1_corr"])) - abs(z1_corr_wo),
                "M1_selection_quality": row["M1_selection_quality"],
                "Z2_corr_full37": z2_corr_full,
                "Z2_corr_without_y": z2_corr_wo,
                "delta_Z2_corr": z2_corr_wo - z2_corr_full,
                "selected_M2_abs_corr": abs(float(row["M2_corr"])),
                "oracle_Z2_abs_corr_without_y": abs(z2_corr_wo),
                "Z2_score_gap": abs(float(row["M2_corr"])) - abs(z2_corr_wo),
                "M2_selection_quality": row["M2_selection_quality"],
                "both_acceptable": bool(row["both_acceptable"]),
                "any_wrong": bool(row["any_wrong"]),
                "nested_abs_error_M1_M2": float(row["nested_abs_error_M1_M2"]),
                "oracle_abs_error_Z1_Z2": float(row["oracle_abs_error_Z1_Z2"]),
            }
        )
    influence_df = pd.DataFrame(influence_rows).sort_values("heldout_wy")
    z1_q25 = influence_df["delta_Z1_corr"].quantile(0.25)
    z2_q25 = influence_df["delta_Z2_corr"].quantile(0.25)
    influence_df["large_negative_delta_Z1"] = influence_df["delta_Z1_corr"] <= z1_q25
    influence_df["large_negative_delta_Z2"] = influence_df["delta_Z2_corr"] <= z2_q25
    influence_df["Z1_structural_year"] = influence_df["large_negative_delta_Z1"] | influence_df[
        "M1_selection_quality"
    ].eq("wrong")
    influence_df["Z2_structural_year"] = influence_df["large_negative_delta_Z2"] | influence_df[
        "M2_selection_quality"
    ].eq("wrong")
    influence_df.to_csv(output_dir / "oracle_column_influence_by_year.csv", index=False)

    # Plots
    plt.style.use("seaborn-v0_8-whitegrid")

    def mode_map(mode_prefix: str, ref: Dict[str, object], out_name: str) -> None:
        lat_col = f"{mode_prefix}_lat"
        lon_col = f"{mode_prefix}_lon"
        qual_col = f"{mode_prefix}_selection_quality"
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        for quality in QUALITY_ORDER:
            subset = by_fold[by_fold[qual_col] == quality]
            if subset.empty:
                continue
            ax.scatter(
                subset[lon_col],
                subset[lat_col],
                color=QUALITY_COLORS[quality],
                s=56,
                alpha=0.9,
                label=quality,
            )
        wrong_subset = by_fold[by_fold[qual_col] == "wrong"]
        for _, row in wrong_subset.iterrows():
            ax.annotate(
                str(int(row["heldout_wy"])),
                (row[lon_col], row[lat_col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.scatter(
            [ref["lon"]],
            [ref["lat"]],
            color="gold",
            edgecolor="black",
            marker="*",
            s=220,
            linewidth=0.8,
            label=f"Oracle {mode_prefix.replace('M', 'Z')}",
            zorder=4,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{mode_prefix} oracle-column near-miss map")
        ax.legend(frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / out_name, dpi=200)
        plt.close(fig)

    mode_map("M1", z1_ref, "mode1_near_miss_map.png")
    mode_map("M2", z2_ref, "mode2_near_miss_map.png")

    combined_quality = np.select(
        [
            by_fold["both_exact"],
            by_fold["both_exact_or_near"],
            by_fold["both_acceptable"],
            by_fold["any_wrong"],
        ],
        ["exact", "near_miss", "acceptable_same_month", "wrong"],
        default="ambiguous",
    )
    by_fold["combined_selection_quality"] = combined_quality

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    years = by_fold["heldout_wy"].astype(int)
    ax.plot(years, by_fold["nested_abs_error_M1_M2"], color="black", linewidth=2.0, label="Nested M1+M2 abs error")
    for quality in QUALITY_ORDER:
        subset = by_fold[by_fold["combined_selection_quality"] == quality]
        if subset.empty:
            continue
        ax.scatter(
            subset["heldout_wy"],
            subset["nested_abs_error_M1_M2"],
            color=QUALITY_COLORS[quality],
            s=58,
            label=quality,
            zorder=3,
        )
    for _, row in by_fold[by_fold["heldout_wy"].isin(SPECIAL_YEARS)].iterrows():
        ax.annotate(
            str(int(row["heldout_wy"])),
            (row["heldout_wy"], row["nested_abs_error_M1_M2"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Nested absolute error (m)")
    ax.set_title("Near miss quality vs nested M1+M2 error")
    ax.legend(frameon=True, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "near_miss_vs_error_timeline.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(
        influence_df["heldout_wy"],
        influence_df["delta_Z1_corr"],
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        label="delta_Z1_corr",
    )
    ax.plot(
        influence_df["heldout_wy"],
        influence_df["delta_Z2_corr"],
        color="#ff7f0e",
        linewidth=2.0,
        marker="s",
        label="delta_Z2_corr",
    )
    wrong_inf = influence_df[influence_df["any_wrong"]]
    ax.scatter(
        wrong_inf["heldout_wy"],
        wrong_inf[["delta_Z1_corr", "delta_Z2_corr"]].min(axis=1),
        color="#d62728",
        marker="x",
        s=70,
        label="Any wrong selection",
        zorder=4,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Leave-one-year change in oracle correlation")
    ax.set_title("Oracle column leave-one-year influence")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_column_leave_one_influence.png", dpi=200)
    plt.close(fig)

    m1_counts = by_fold["M1_selection_quality"].value_counts().reindex(QUALITY_ORDER, fill_value=0)
    m2_counts = by_fold["M2_selection_quality"].value_counts().reindex(QUALITY_ORDER, fill_value=0)
    wrong_m1_years = by_fold.loc[by_fold["M1_selection_quality"] == "wrong", "heldout_wy"].astype(int).tolist()
    wrong_m2_years = by_fold.loc[by_fold["M2_selection_quality"] == "wrong", "heldout_wy"].astype(int).tolist()
    any_wrong_years = by_fold.loc[by_fold["any_wrong"], "heldout_wy"].astype(int).tolist()
    near_miss_years = by_fold.loc[
        (by_fold["M1_selection_quality"] == "near_miss")
        | (by_fold["M2_selection_quality"] == "near_miss"),
        "heldout_wy",
    ].astype(int).tolist()
    acceptable_years = by_fold.loc[
        (by_fold["M1_selection_quality"] == "acceptable_same_month")
        | (by_fold["M2_selection_quality"] == "acceptable_same_month"),
        "heldout_wy",
    ].astype(int).tolist()

    influence_summary = {
        "Z1_q25_delta_corr": float(z1_q25),
        "Z2_q25_delta_corr": float(z2_q25),
        "Z1_structural_years": influence_df.loc[influence_df["Z1_structural_year"], "heldout_wy"].astype(int).tolist(),
        "Z2_structural_years": influence_df.loc[influence_df["Z2_structural_year"], "heldout_wy"].astype(int).tolist(),
        "most_negative_Z1_delta_years": influence_df.nsmallest(5, "delta_Z1_corr")["heldout_wy"].astype(int).tolist(),
        "most_negative_Z2_delta_years": influence_df.nsmallest(5, "delta_Z2_corr")["heldout_wy"].astype(int).tolist(),
    }

    if len(any_wrong_years) <= max(4, len(near_miss_years)):
        first_sentence = (
            "When exact Z1/Z2 are not selected, the nested choices are often still near misses "
            "or acceptable same-month selections rather than true wrong columns."
        )
    else:
        first_sentence = (
            "When exact Z1/Z2 are not selected, a substantial fraction of folds jump to truly "
            "wrong different-month and far-away columns."
        )
    short_answer = (
        f"{first_sentence} True wrong-column years are M1={wrong_m1_years} and "
        f"M2={wrong_m2_years}, with any wrong column in years {any_wrong_years}. "
        f"The high-error years are concentrated in structurally unstable folds: "
        f"{by_fold.nlargest(5, 'nested_abs_error_M1_M2')['heldout_wy'].astype(int).tolist()} "
        f"contain several true wrong or non-acceptable selections, while the key structural "
        f"influence years for keeping Z1/Z2 competitive are highlighted by the large negative "
        f"leave-one-year deltas: Z1 {influence_summary['most_negative_Z1_delta_years']}, "
        f"Z2 {influence_summary['most_negative_Z2_delta_years']}."
    )

    summary = {
        "full37_Z1": z1_ref,
        "full37_Z2": z2_ref,
        "num_folds": int(len(by_fold)),
        "counts_by_M1_selection_quality": {k: int(v) for k, v in m1_counts.items()},
        "counts_by_M2_selection_quality": {k: int(v) for k, v in m2_counts.items()},
        "num_both_exact": int(by_fold["both_exact"].sum()),
        "num_both_exact_or_near": int(by_fold["both_exact_or_near"].sum()),
        "num_both_acceptable": int(by_fold["both_acceptable"].sum()),
        "num_any_wrong": int(by_fold["any_wrong"].sum()),
        "num_both_wrong": int(by_fold["both_wrong"].sum()),
        "wrong_selection_years": any_wrong_years,
        "near_miss_years": near_miss_years,
        "acceptable_same_month_years": acceptable_years,
        "error_metrics_by_quality_group": group_df.to_dict(orient="records"),
        "oracle_column_influence_summary": influence_summary,
        "short_answer": short_answer,
    }
    with (output_dir / "oracle_column_near_miss_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Output directory: {output_dir}")
    print("M1 quality counts:")
    print(m1_counts.to_string())
    print("M2 quality counts:")
    print(m2_counts.to_string())
    print("Years with true wrong M1:")
    print(wrong_m1_years)
    print("Years with true wrong M2:")
    print(wrong_m2_years)
    print("Years with any true wrong column:")
    print(any_wrong_years)
    print("Error by quality group:")
    print(group_df.to_string(index=False))
    print("Oracle influence summary:")
    print(json.dumps(influence_summary, indent=2))
    print("Short answer:")
    print(short_answer)


if __name__ == "__main__":
    main()
