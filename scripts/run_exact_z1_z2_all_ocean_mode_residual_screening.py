#!/usr/bin/env python3
"""
Residual screening for Pacific PC, Nino34, and AMV/AMO families against exact Z1+Z2 SWE residuals.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


RESIDUAL_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_residual_analysis"
RESIDUAL_TABLE_CSV = RESIDUAL_DIR / "exact_Z1_Z2_residual_table.csv"
RESIDUAL_SUMMARY_JSON = RESIDUAL_DIR / "exact_Z1_Z2_residual_analysis_summary.json"

PACIFIC_PC_SOURCE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "sst_pca"
    / "cobe2_global_monthly_climatology_anomaly"
    / "cobe2_global_monthly_clim_sst_eofs.nc"
)
NINO34_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "nino34"
    / "nino34_monthly_wy1985_2021_sep_mar.csv"
)
AMV_AMO_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "swe_climate_mode_baseline"
    / "amv_amo"
    / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
)

PACIFIC_SCREENING_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_pacific_pc_residual_screening"
)
PACIFIC_SCREENING_CORR_CSV = PACIFIC_SCREENING_DIR / "pacific_pc_residual_screening_correlations.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_all_ocean_mode_residual_screening"
)

CORRELATIONS_CSV = OUTPUT_DIR / "all_ocean_mode_residual_screening_correlations.csv"
TOP_RANKED_CSV = OUTPUT_DIR / "all_ocean_mode_residual_screening_top_ranked.csv"
FAMILY_COMPARISON_CSV = OUTPUT_DIR / "all_ocean_mode_family_comparison.csv"
TOP_YEAR_VALUES_CSV = OUTPUT_DIR / "all_ocean_mode_top_residual_year_values.csv"
TOP_YEAR_FLAGS_CSV = OUTPUT_DIR / "all_ocean_mode_top_residual_year_extreme_flags.csv"
SUMMARY_JSON = OUTPUT_DIR / "all_ocean_mode_residual_screening_summary.json"
FAMILY_BARPLOT_PNG = OUTPUT_DIR / "all_ocean_mode_family_best_correlations_barplot.png"
RESIDUAL_HEATMAPS_PNG = OUTPUT_DIR / "all_ocean_mode_residual_correlation_heatmaps.png"
ABS_RESIDUAL_HEATMAPS_PNG = OUTPUT_DIR / "all_ocean_mode_abs_residual_correlation_heatmaps.png"
TOP_YEAR_Z_PNG = OUTPUT_DIR / "all_ocean_mode_top_residual_year_zscore_heatmap.png"
TOP_SCATTER_PNG = OUTPUT_DIR / "all_ocean_mode_residual_scatter_top_candidates.png"

MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
FAMILY_ORDER = ["Pacific_PC", "Nino34", "AMV_AMO"]
EXPECTED_RESIDUAL_COLUMNS = ["water_year", "obs_swe", "pred_swe", "residual_obs_minus_pred", "abs_residual"]
RANKING_SPECS = [
    ("abs_pearson_residual", "pearson_r_with_residual"),
    ("abs_spearman_residual", "spearman_r_with_residual"),
    ("abs_pearson_abs_residual", "pearson_r_with_abs_residual"),
    ("abs_spearman_abs_residual", "spearman_r_with_abs_residual"),
]


def ensure_runtime_on_compute_node():
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def classify_strength(abs_corr):
    if not np.isfinite(abs_corr):
        return "unknown"
    if abs_corr >= 0.6:
        return "strong"
    if abs_corr >= 0.4:
        return "moderate"
    if abs_corr >= 0.25:
        return "weak-to-moderate"
    return "weak"


def pearson_and_spearman(x, y):
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    n = int(mask.sum())
    result = {
        "n_years": n,
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_r": float("nan"),
        "spearman_p": float("nan"),
    }
    if n < 3:
        return result
    xx = x_arr[mask]
    yy = y_arr[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return result
    if stats is None:
        result["pearson_r"] = float(np.corrcoef(xx, yy)[0, 1])
        ranks_x = pd.Series(xx).rank(method="average").to_numpy(dtype=float)
        ranks_y = pd.Series(yy).rank(method="average").to_numpy(dtype=float)
        result["spearman_r"] = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        return result
    pearson = stats.pearsonr(xx, yy)
    spearman = stats.spearmanr(xx, yy)
    result["pearson_r"] = float(pearson.statistic)
    result["pearson_p"] = float(pearson.pvalue)
    result["spearman_r"] = float(spearman.statistic)
    result["spearman_p"] = float(spearman.pvalue)
    return result


def load_base_residuals():
    missing = [str(path) for path in [RESIDUAL_TABLE_CSV, RESIDUAL_SUMMARY_JSON] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required base residual files: {}".format(missing))
    df = pd.read_csv(RESIDUAL_TABLE_CSV)
    column_mapping = {name: name for name in EXPECTED_RESIDUAL_COLUMNS}
    for required in EXPECTED_RESIDUAL_COLUMNS:
        if required not in df.columns:
            raise ValueError("Residual table missing required column: {}".format(required))
    df["water_year"] = df["water_year"].astype(int)
    return df.sort_values("water_year").reset_index(drop=True), column_mapping, json.loads(RESIDUAL_SUMMARY_JSON.read_text())


def build_wy_aligned_pc_table(netcdf_path, water_years, prefix):
    ds = xr.open_dataset(netcdf_path)
    if "pc" not in ds:
        raise ValueError("Missing pc in {}".format(netcdf_path))
    pc = ds["pc"].load()
    times = pd.to_datetime(ds["time"].to_numpy())
    mode_values = ds["mode"].to_numpy()
    data = pd.DataFrame(pc.to_numpy(), index=times, columns=[int(m) for m in mode_values])
    month_to_number = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
    rows = []
    for water_year in water_years:
        row = {"water_year": int(water_year)}
        for month in MONTHS:
            calendar_year = water_year - 1 if month in {"Sep", "Oct", "Nov", "Dec"} else water_year
            timestamp = pd.Timestamp(calendar_year, month_to_number[month], 1)
            if timestamp not in data.index:
                raise KeyError("Missing {} in {}".format(timestamp.strftime("%Y-%m"), netcdf_path))
            values = data.loc[timestamp]
            for pc_index in range(1, 7):
                row["{}_PC{}_{}".format(prefix, pc_index, month)] = float(values.loc[pc_index])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)


def build_family_long_table(wide_df, family, source_file):
    value_columns = [name for name in wide_df.columns if name != "water_year"]
    rows = []
    for _, row in wide_df.iterrows():
        water_year = int(row["water_year"])
        for column in value_columns:
            value = float(row[column])
            if family == "Nino34":
                month = column.split("_", 1)[1]
                pc_or_index = "Nino34"
            else:
                left, month = column.rsplit("_", 1)
                pc_or_index = left.split("_")[1]
            rows.append(
                {
                    "water_year": water_year,
                    "family": family,
                    "variable_name": column,
                    "pc_or_index": pc_or_index,
                    "month": month,
                    "value": value,
                    "source_file": str(source_file),
                }
            )
    return pd.DataFrame(rows)


def load_family_tables(residual_df):
    source_files_by_family = {}
    wide_by_family = {}
    long_frames = []

    pacific_wide = build_wy_aligned_pc_table(PACIFIC_PC_SOURCE_FILE, residual_df["water_year"].tolist(), "Pacific")
    source_files_by_family["Pacific_PC"] = str(PACIFIC_PC_SOURCE_FILE)
    wide_by_family["Pacific_PC"] = pacific_wide
    long_frames.append(build_family_long_table(pacific_wide, "Pacific_PC", PACIFIC_PC_SOURCE_FILE))

    if not NINO34_CSV.exists():
        raise FileNotFoundError("Missing Nino34 source file: {}".format(NINO34_CSV))
    nino_wide = pd.read_csv(NINO34_CSV).sort_values("water_year").reset_index(drop=True)
    nino_wide["water_year"] = nino_wide["water_year"].astype(int)
    source_files_by_family["Nino34"] = str(NINO34_CSV)
    wide_by_family["Nino34"] = nino_wide
    long_frames.append(build_family_long_table(nino_wide, "Nino34", NINO34_CSV))

    if not AMV_AMO_CSV.exists():
        raise FileNotFoundError("Missing AMV/AMO source file: {}".format(AMV_AMO_CSV))
    amv_wide = pd.read_csv(AMV_AMO_CSV).sort_values("water_year").reset_index(drop=True)
    amv_wide["water_year"] = amv_wide["water_year"].astype(int)
    source_files_by_family["AMV_AMO"] = str(AMV_AMO_CSV)
    wide_by_family["AMV_AMO"] = amv_wide
    long_frames.append(build_family_long_table(amv_wide, "AMV_AMO", AMV_AMO_CSV))

    return pd.concat(long_frames, ignore_index=True), wide_by_family, source_files_by_family


def build_correlation_table(residual_df, family_long_df):
    rows = []
    grouped = family_long_df.groupby(["family", "variable_name", "pc_or_index", "month", "source_file"], sort=False)
    for (family, variable_name, pc_or_index, month, source_file), group in grouped:
        merged = residual_df[EXPECTED_RESIDUAL_COLUMNS].merge(
            group[["water_year", "value"]], on="water_year", how="inner"
        )
        residual_stats = pearson_and_spearman(merged["value"], merged["residual_obs_minus_pred"])
        abs_stats = pearson_and_spearman(merged["value"], merged["abs_residual"])
        rows.append(
            {
                "family": family,
                "variable_name": variable_name,
                "pc_or_index": pc_or_index,
                "month": month,
                "n_years": residual_stats["n_years"],
                "pearson_r_with_residual": residual_stats["pearson_r"],
                "pearson_p_with_residual": residual_stats["pearson_p"],
                "spearman_r_with_residual": residual_stats["spearman_r"],
                "spearman_p_with_residual": residual_stats["spearman_p"],
                "pearson_r_with_abs_residual": abs_stats["pearson_r"],
                "pearson_p_with_abs_residual": abs_stats["pearson_p"],
                "spearman_r_with_abs_residual": abs_stats["spearman_r"],
                "spearman_p_with_abs_residual": abs_stats["spearman_p"],
                "source_file": source_file,
            }
        )
    return pd.DataFrame(rows)


def build_top_ranked_table(correlations_df):
    ranked_frames = []
    for ranking_type, corr_col in RANKING_SPECS:
        ranked = correlations_df.copy()
        ranked["ranking_type"] = ranking_type
        ranked["correlation_value"] = ranked[corr_col]
        ranked["abs_correlation_value"] = ranked[corr_col].abs()
        ranked = ranked.sort_values(["abs_correlation_value", "family", "variable_name"], ascending=[False, True, True]).head(20)
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_frames.append(
            ranked[
                [
                    "ranking_type",
                    "rank",
                    "family",
                    "variable_name",
                    "pc_or_index",
                    "month",
                    "correlation_value",
                    "abs_correlation_value",
                    "n_years",
                    "source_file",
                ]
            ]
        )
    return pd.concat(ranked_frames, ignore_index=True)


def best_row_for_family(group, column):
    idx = group[column].abs().idxmax()
    return group.loc[idx]


def build_family_comparison(correlations_df):
    rows = []
    for family in FAMILY_ORDER:
        group = correlations_df.loc[correlations_df["family"] == family].copy()
        best_pearson_residual = best_row_for_family(group, "pearson_r_with_residual")
        best_spearman_residual = best_row_for_family(group, "spearman_r_with_residual")
        best_pearson_abs = best_row_for_family(group, "pearson_r_with_abs_residual")
        best_spearman_abs = best_row_for_family(group, "spearman_r_with_abs_residual")
        rows.append(
            {
                "family": family,
                "n_predictors": int(len(group)),
                "best_abs_pearson_residual": float(abs(best_pearson_residual["pearson_r_with_residual"])),
                "best_abs_pearson_residual_variable": best_pearson_residual["variable_name"],
                "best_abs_spearman_residual": float(abs(best_spearman_residual["spearman_r_with_residual"])),
                "best_abs_spearman_residual_variable": best_spearman_residual["variable_name"],
                "best_abs_pearson_abs_residual": float(abs(best_pearson_abs["pearson_r_with_abs_residual"])),
                "best_abs_pearson_abs_residual_variable": best_pearson_abs["variable_name"],
                "best_abs_spearman_abs_residual": float(abs(best_spearman_abs["spearman_r_with_abs_residual"])),
                "best_abs_spearman_abs_residual_variable": best_spearman_abs["variable_name"],
                "mean_abs_pearson_residual": float(group["pearson_r_with_residual"].abs().mean()),
                "mean_abs_spearman_residual": float(group["spearman_r_with_residual"].abs().mean()),
                "mean_abs_pearson_abs_residual": float(group["pearson_r_with_abs_residual"].abs().mean()),
                "mean_abs_spearman_abs_residual": float(group["spearman_r_with_abs_residual"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_top_year_tables(residual_df, family_long_df, top_residual_years):
    selected = residual_df.loc[residual_df["water_year"].isin(top_residual_years), EXPECTED_RESIDUAL_COLUMNS].copy()
    joined = selected.merge(family_long_df, on="water_year", how="inner")
    value_rows = []
    flag_rows = []
    for family in FAMILY_ORDER:
        family_group = joined.loc[joined["family"] == family].copy()
        if family_group.empty:
            continue
        family_reference = family_long_df.loc[family_long_df["family"] == family].copy()
        stats_by_var = {}
        for variable_name, group in family_reference.groupby("variable_name"):
            values = group["value"].to_numpy(dtype=float)
            stats_by_var[variable_name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
                "q80": float(np.nanquantile(np.abs(values), 0.8)),
                "q90": float(np.nanquantile(np.abs(values), 0.9)),
            }
        for water_year in top_residual_years:
            year_group = family_group.loc[family_group["water_year"] == water_year].copy()
            if year_group.empty:
                continue
            year_group = year_group.sort_values(["family", "variable_name"]).reset_index(drop=True)
            abs_z_records = []
            n1 = 0
            n15 = 0
            n2 = 0
            n20 = 0
            n10 = 0
            for _, row in year_group.iterrows():
                ref = stats_by_var[row["variable_name"]]
                zscore = float((row["value"] - ref["mean"]) / ref["std"]) if ref["std"] > 0.0 else float("nan")
                top20 = bool(np.isfinite(row["value"]) and abs(row["value"]) >= ref["q80"])
                top10 = bool(np.isfinite(row["value"]) and abs(row["value"]) >= ref["q90"])
                if np.isfinite(zscore) and abs(zscore) >= 1.0:
                    n1 += 1
                if np.isfinite(zscore) and abs(zscore) >= 1.5:
                    n15 += 1
                if np.isfinite(zscore) and abs(zscore) >= 2.0:
                    n2 += 1
                if top20:
                    n20 += 1
                if top10:
                    n10 += 1
                value_rows.append(
                    {
                        "water_year": int(water_year),
                        "family": family,
                        "variable_name": row["variable_name"],
                        "pc_or_index": row["pc_or_index"],
                        "month": row["month"],
                        "raw_value": float(row["value"]),
                        "standardized_zscore": zscore,
                        "residual_obs_minus_pred": float(row["residual_obs_minus_pred"]),
                        "abs_residual": float(row["abs_residual"]),
                        "obs_swe": float(row["obs_swe"]),
                        "pred_swe": float(row["pred_swe"]),
                    }
                )
                abs_z_records.append((row["variable_name"], abs(zscore) if np.isfinite(zscore) else float("-inf")))
            abs_z_records.sort(key=lambda item: (-item[1], item[0]))
            top10_vars = [name for name, _ in abs_z_records[:10]]
            flag_rows.append(
                {
                    "water_year": int(water_year),
                    "family": family,
                    "num_variables_abs_z_ge_1": int(n1),
                    "num_variables_abs_z_ge_1p5": int(n15),
                    "num_variables_abs_z_ge_2": int(n2),
                    "num_variables_top_20pct_abs": int(n20),
                    "num_variables_top_10pct_abs": int(n10),
                    "top_10_largest_abs_z_variables": "; ".join(top10_vars),
                }
            )
    return pd.DataFrame(value_rows), pd.DataFrame(flag_rows)


def draw_family_best_correlations_barplot(family_comparison_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(family_comparison_df))
    width = 0.38
    left = family_comparison_df["best_abs_pearson_residual"].to_numpy(dtype=float)
    right = family_comparison_df["best_abs_pearson_abs_residual"].to_numpy(dtype=float)
    axes[0].bar(x, left, color="#1f77b4")
    axes[0].set_title("Best |Pearson r| with residual")
    axes[1].bar(x, right, color="#d62728")
    axes[1].set_title("Best |Pearson r| with |residual|")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(family_comparison_df["family"].tolist(), rotation=0)
        ax.set_ylim(0.0, max(0.4, float(np.nanmax([left.max(), right.max()])) + 0.05))
        ax.set_ylabel("|correlation|")
    fig.tight_layout()
    fig.savefig(FAMILY_BARPLOT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def family_heatmap_matrix(correlations_df, family, value_column):
    group = correlations_df.loc[correlations_df["family"] == family].copy()
    if family == "Nino34":
        row = []
        for month in MONTHS:
            match = group.loc[group["month"] == month, value_column]
            row.append(float(match.iloc[0]) if not match.empty else float("nan"))
        return np.asarray([row], dtype=float), ["Nino34"]
    rows = []
    labels = []
    for pc in range(1, 7):
        row = []
        label = "PC{}".format(pc)
        for month in MONTHS:
            match = group.loc[(group["pc_or_index"] == label) & (group["month"] == month), value_column]
            row.append(float(match.iloc[0]) if not match.empty else float("nan"))
        rows.append(row)
        labels.append(label)
    return np.asarray(rows, dtype=float), labels


def draw_heatmap_grid(correlations_df, value_column, title_suffix, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        matrix, labels = family_heatmap_matrix(correlations_df, family, value_column)
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-0.75, vmax=0.75)
        ax.set_title("{}\n{}".format(family, title_suffix))
        ax.set_xticks(np.arange(len(MONTHS)))
        ax.set_xticklabels(MONTHS)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = "nan" if not np.isfinite(value) else "{:.2f}".format(value)
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Pearson r")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_top_year_zscore_heatmap(top_year_values_df, top_ranked_df):
    selected_vars = top_ranked_df.loc[top_ranked_df["ranking_type"] == "abs_pearson_residual", "variable_name"].drop_duplicates().head(20).tolist()
    subset = top_year_values_df.loc[top_year_values_df["variable_name"].isin(selected_vars)].copy()
    if subset.empty:
        return
    subset["row_label"] = subset["family"] + "_" + subset["water_year"].astype(str)
    pivot = subset.pivot(index="row_label", columns="variable_name", values="standardized_zscore")
    ordered_rows = []
    for year in sorted(top_year_values_df["water_year"].unique().tolist()):
        for family in FAMILY_ORDER:
            label = "{}_{}".format(family, year)
            if label in pivot.index:
                ordered_rows.append(label)
    pivot = pivot.reindex(index=ordered_rows, columns=selected_vars)
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(selected_vars)), 7))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-3.0, vmax=3.0)
    ax.set_xticks(np.arange(len(selected_vars)))
    ax.set_xticklabels(selected_vars, rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Top residual-year z-scores for top-ranked variables")
    fig.colorbar(image, ax=ax, shrink=0.85, label="z-score")
    fig.tight_layout()
    fig.savefig(TOP_YEAR_Z_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_top_scatter(correlations_df, residual_df, family_long_df):
    top_candidates = correlations_df.copy()
    top_candidates["abs_corr"] = top_candidates["pearson_r_with_residual"].abs()
    top_candidates = top_candidates.sort_values(["abs_corr", "family", "variable_name"], ascending=[False, True, True]).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    top_years = (
        residual_df[["water_year", "abs_residual"]]
        .sort_values(["abs_residual", "water_year"], ascending=[False, True])
        .head(5)["water_year"]
        .tolist()
    )
    for ax, (_, candidate) in zip(axes.ravel(), top_candidates.iterrows()):
        merged = residual_df[EXPECTED_RESIDUAL_COLUMNS].merge(
            family_long_df.loc[family_long_df["variable_name"] == candidate["variable_name"], ["water_year", "value"]],
            on="water_year",
            how="inner",
        )
        ax.scatter(merged["value"], merged["residual_obs_minus_pred"], color="#1f77b4", alpha=0.85)
        for year in top_years:
            row = merged.loc[merged["water_year"] == year]
            if row.empty:
                continue
            xval = float(row.iloc[0]["value"])
            yval = float(row.iloc[0]["residual_obs_minus_pred"])
            ax.scatter(xval, yval, color="#d62728", s=30)
            ax.text(xval, yval, str(year), fontsize=8, ha="left", va="bottom")
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_title(candidate["variable_name"])
        ax.set_xlabel("Predictor value")
        ax.set_ylabel("Residual obs - pred")
    for ax in axes.ravel()[len(top_candidates) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(TOP_SCATTER_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reuse_pacific_screening_if_compatible():
    if not PACIFIC_SCREENING_CORR_CSV.exists():
        return {"reused": False, "reason": "missing Pacific screening correlation CSV"}
    try:
        pacific_corr = pd.read_csv(PACIFIC_SCREENING_CORR_CSV)
    except Exception as exc:
        return {"reused": False, "reason": "failed to read Pacific screening outputs: {}".format(exc)}
    if "source_file" not in pacific_corr.columns:
        return {"reused": False, "reason": "Pacific screening CSV missing source_file"}
    source_files = sorted(set(pacific_corr["source_file"].astype(str).tolist()))
    expected = str(PACIFIC_PC_SOURCE_FILE)
    if source_files != [expected]:
        return {"reused": False, "reason": "Pacific screening source mismatch: {}".format(source_files)}
    return {"reused": True, "reason": "Pacific-only screening already compatible with canonical Pacific EOF/PC source"}


def build_short_answer(family_comparison_df, top_year_flags_df):
    best_family_row = family_comparison_df.sort_values(
        ["best_abs_pearson_residual", "best_abs_pearson_abs_residual"], ascending=[False, False]
    ).iloc[0]
    best_corr = float(max(best_family_row["best_abs_pearson_residual"], best_family_row["best_abs_pearson_abs_residual"]))
    strength = classify_strength(best_corr)
    high_year_flag_summary = top_year_flags_df.groupby("family")[
        ["num_variables_abs_z_ge_1", "num_variables_abs_z_ge_1p5", "num_variables_abs_z_ge_2"]
    ].mean()
    strongest_family = best_family_row["family"]
    top_counts = high_year_flag_summary.loc[strongest_family].to_dict() if strongest_family in high_year_flag_summary.index else {}
    unusual = "yes" if top_counts and (top_counts.get("num_variables_abs_z_ge_1p5", 0.0) > 0.0) else "limited"
    return (
        "{} shows the strongest residual association overall. The strongest family-level association is {} "
        "(best absolute correlation {:.3f}). High-residual years show {} evidence of unusually large/small values "
        "within at least one family, but this is still screening only and does not establish predictive improvement."
    ).format(strongest_family, strength, best_corr, unusual)


def main():
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    residual_df, residual_column_mapping, residual_summary = load_base_residuals()
    if len(residual_df) != 37:
        raise ValueError("Expected 37 residual years, found {}".format(len(residual_df)))
    top_residual_years = (
        residual_df[["water_year", "abs_residual"]]
        .sort_values(["abs_residual", "water_year"], ascending=[False, True])
        .head(5)["water_year"]
        .astype(int)
        .tolist()
    )

    pacific_reuse = reuse_pacific_screening_if_compatible()
    family_long_df, wide_by_family, source_files_by_family = load_family_tables(residual_df)
    correlations_df = build_correlation_table(residual_df, family_long_df)
    top_ranked_df = build_top_ranked_table(correlations_df)
    family_comparison_df = build_family_comparison(correlations_df)
    top_year_values_df, top_year_flags_df = build_top_year_tables(residual_df, family_long_df, top_residual_years)

    correlations_df.to_csv(CORRELATIONS_CSV, index=False)
    top_ranked_df.to_csv(TOP_RANKED_CSV, index=False)
    family_comparison_df.to_csv(FAMILY_COMPARISON_CSV, index=False)
    top_year_values_df.to_csv(TOP_YEAR_VALUES_CSV, index=False)
    top_year_flags_df.to_csv(TOP_YEAR_FLAGS_CSV, index=False)

    draw_family_best_correlations_barplot(family_comparison_df)
    draw_heatmap_grid(correlations_df, "pearson_r_with_residual", "Pearson r with residual", RESIDUAL_HEATMAPS_PNG)
    draw_heatmap_grid(correlations_df, "pearson_r_with_abs_residual", "Pearson r with |residual|", ABS_RESIDUAL_HEATMAPS_PNG)
    draw_top_year_zscore_heatmap(top_year_values_df, top_ranked_df)
    draw_top_scatter(correlations_df, residual_df, family_long_df)

    family_comparison_records = family_comparison_df.to_dict(orient="records")
    top_residual_year_extreme_summary = top_year_flags_df.to_dict(orient="records")
    short_answer = build_short_answer(family_comparison_df, top_year_flags_df)
    strongest_family_row = family_comparison_df.sort_values(
        ["best_abs_pearson_residual", "best_abs_pearson_abs_residual"], ascending=[False, False]
    ).iloc[0]
    next_step = (
        "Use the strongest family and variables from this screening to design a small, additive follow-up diagnostic "
        "such as partial-correlation checks or a held-out block-addition test, without yet claiming improved prediction."
    )

    summary_payload = {
        "base_residual_file": str(RESIDUAL_TABLE_CSV),
        "base_residual_summary_json": str(RESIDUAL_SUMMARY_JSON),
        "residual_column_mapping": residual_column_mapping,
        "output_dir": str(OUTPUT_DIR),
        "source_files_by_family": source_files_by_family,
        "water_years": residual_df["water_year"].astype(int).tolist(),
        "n_years": int(len(residual_df)),
        "n_predictors_by_family": {
            family: int(len([c for c in wide_by_family[family].columns if c != "water_year"])) for family in FAMILY_ORDER
        },
        "top5_abs_residual_years": top_residual_years,
        "family_comparison": family_comparison_records,
        "top_predictors_by_abs_pearson_residual": top_ranked_df.loc[
            top_ranked_df["ranking_type"] == "abs_pearson_residual"
        ].to_dict(orient="records"),
        "top_predictors_by_abs_spearman_residual": top_ranked_df.loc[
            top_ranked_df["ranking_type"] == "abs_spearman_residual"
        ].to_dict(orient="records"),
        "top_predictors_by_abs_pearson_abs_residual": top_ranked_df.loc[
            top_ranked_df["ranking_type"] == "abs_pearson_abs_residual"
        ].to_dict(orient="records"),
        "top_predictors_by_abs_spearman_abs_residual": top_ranked_df.loc[
            top_ranked_df["ranking_type"] == "abs_spearman_abs_residual"
        ].to_dict(orient="records"),
        "top_residual_year_extreme_summary": top_residual_year_extreme_summary,
        "scipy_available_for_pvalues": bool(stats is not None),
        "pacific_screening_reuse_status": pacific_reuse,
        "short_answer": short_answer,
        "next_recommended_step": next_step,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    print("Output directory: {}".format(OUTPUT_DIR))
    print("Residual file verified: yes")
    print("Families screened:")
    for family in FAMILY_ORDER:
        print(
            "- {}: {} predictors, {}".format(
                family,
                len([c for c in wide_by_family[family].columns if c != "water_year"]),
                source_files_by_family[family],
            )
        )
    print("Family comparison:")
    print(family_comparison_df.to_string(index=False))
    print("Top residual correlations:")
    print(
        top_ranked_df.loc[top_ranked_df["ranking_type"] == "abs_pearson_residual"][
            ["rank", "family", "variable_name", "correlation_value"]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("Top abs-residual correlations:")
    print(
        top_ranked_df.loc[top_ranked_df["ranking_type"] == "abs_pearson_abs_residual"][
            ["rank", "family", "variable_name", "correlation_value"]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("Short answer:")
    print(short_answer)
    print("Next recommended step:")
    print(next_step)


if __name__ == "__main__":
    main()
