#!/usr/bin/env python3
"""
Focused Pacific PC residual screening for the saved exact-grid-cell Z1+Z2 SWE model.
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_pacific_pc_residual_screening"
)

CORRELATIONS_CSV = OUTPUT_DIR / "pacific_pc_residual_screening_correlations.csv"
TOP_RANKED_CSV = OUTPUT_DIR / "pacific_pc_residual_screening_top_ranked.csv"
TOP_YEAR_VALUES_CSV = OUTPUT_DIR / "pacific_pc_top_residual_year_values.csv"
TOP_YEAR_FLAGS_CSV = OUTPUT_DIR / "pacific_pc_top_residual_year_extreme_flags.csv"
SUMMARY_JSON = OUTPUT_DIR / "pacific_pc_residual_screening_summary.json"
RESIDUAL_HEATMAP_PNG = OUTPUT_DIR / "pacific_pc_residual_correlation_heatmap.png"
ABS_RESIDUAL_HEATMAP_PNG = OUTPUT_DIR / "pacific_pc_abs_residual_correlation_heatmap.png"
TOP_YEAR_Z_HEATMAP_PNG = OUTPUT_DIR / "pacific_pc_top_residual_year_zscore_heatmap.png"
TOP_SCATTER_PNG = OUTPUT_DIR / "pacific_pc_residual_scatter_top_candidates.png"

MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
TOP_RESIDUAL_YEARS = [1993, 2005, 2006, 2012, 2021]
TOP_K_PER_RANKING = 15
EXPECTED_BASE_COLUMNS = [
    "water_year",
    "obs_swe",
    "pred_swe",
    "residual_obs_minus_pred",
    "abs_residual",
]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def pearson_and_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
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


def load_base_residuals() -> Tuple[pd.DataFrame, Dict[str, object]]:
    missing = [str(path) for path in [RESIDUAL_TABLE_CSV, RESIDUAL_SUMMARY_JSON] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required base residual files: {missing}")
    df = pd.read_csv(RESIDUAL_TABLE_CSV)
    missing_columns = [name for name in EXPECTED_BASE_COLUMNS if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Residual table missing required columns: {missing_columns}")
    df["water_year"] = df["water_year"].astype(int)
    summary = json.loads(RESIDUAL_SUMMARY_JSON.read_text())
    return df.sort_values("water_year").reset_index(drop=True), summary


def build_wy_aligned_pc_table(netcdf_path: Path, water_years: Iterable[int]) -> pd.DataFrame:
    ds = xr.open_dataset(netcdf_path)
    if "pc" not in ds:
        raise ValueError(f"Missing pc in {netcdf_path}")
    pc = ds["pc"].load()
    times = pd.to_datetime(ds["time"].to_numpy())
    mode_values = ds["mode"].to_numpy()
    data = pd.DataFrame(pc.to_numpy(), index=times, columns=[int(m) for m in mode_values])
    month_to_number = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
    rows: List[Dict[str, float]] = []
    for water_year in water_years:
        row = {"water_year": int(water_year)}
        for month in MONTHS:
            calendar_year = water_year - 1 if month in {"Sep", "Oct", "Nov", "Dec"} else water_year
            timestamp = pd.Timestamp(calendar_year, month_to_number[month], 1)
            if timestamp not in data.index:
                raise KeyError(f"Missing {timestamp:%Y-%m} in {netcdf_path}")
            values = data.loc[timestamp]
            for pc_index in range(1, 7):
                row[f"Pacific_PC{pc_index}_{month}"] = float(values.loc[pc_index])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)


def verify_residual_table_alignment(
    residual_df: pd.DataFrame,
    aligned_pc_df: pd.DataFrame,
) -> Dict[str, object]:
    pc_columns = [f"Pacific_PC{pc}_{month}" for month in MONTHS for pc in range(1, 7)]
    missing_columns = [name for name in pc_columns if name not in residual_df.columns]
    comparison = {
        "residual_table_has_all_42_pc_columns": not missing_columns,
        "missing_pc_columns_in_residual_table": missing_columns,
        "max_abs_diff_vs_canonical_netcdf": float("nan"),
        "allclose_vs_canonical_netcdf": False,
    }
    if missing_columns:
        return comparison
    merged = residual_df[["water_year", *pc_columns]].merge(aligned_pc_df, on="water_year", suffixes=("_residual", "_canonical"))
    diffs = []
    for column in pc_columns:
        left = merged[f"{column}_residual"].to_numpy(dtype=float)
        right = merged[f"{column}_canonical"].to_numpy(dtype=float)
        diffs.append(np.max(np.abs(left - right)))
    max_abs_diff = float(np.max(diffs)) if diffs else float("nan")
    comparison["max_abs_diff_vs_canonical_netcdf"] = max_abs_diff
    comparison["allclose_vs_canonical_netcdf"] = bool(np.isfinite(max_abs_diff) and max_abs_diff <= 1.0e-6)
    return comparison


def build_correlation_table(screen_df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    rows = []
    for month in MONTHS:
        for pc in range(1, 7):
            column = f"Pacific_PC{pc}_{month}"
            residual_stats = pearson_and_spearman(screen_df[column], screen_df["residual_obs_minus_pred"])
            abs_stats = pearson_and_spearman(screen_df[column], screen_df["abs_residual"])
            rows.append(
                {
                    "variable_name": column,
                    "pc": pc,
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
                    "source_file": str(source_file),
                }
            )
    return pd.DataFrame(rows)


def build_top_ranked_table(correlations_df: pd.DataFrame) -> pd.DataFrame:
    ranking_specs = [
        ("abs_pearson_r_with_residual", "pearson_r_with_residual"),
        ("abs_spearman_r_with_residual", "spearman_r_with_residual"),
        ("abs_pearson_r_with_abs_residual", "pearson_r_with_abs_residual"),
        ("abs_spearman_r_with_abs_residual", "spearman_r_with_abs_residual"),
    ]
    ranked_frames = []
    for ranking_type, column in ranking_specs:
        ranked = correlations_df.copy()
        ranked["ranking_type"] = ranking_type
        ranked["ranking_value"] = ranked[column].abs()
        ranked = ranked.sort_values(["ranking_value", "variable_name"], ascending=[False, True]).head(TOP_K_PER_RANKING)
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_frames.append(ranked)
    return pd.concat(ranked_frames, ignore_index=True)


def build_top_year_tables(screen_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    flag_rows = []
    for month in MONTHS:
        for pc in range(1, 7):
            column = f"Pacific_PC{pc}_{month}"
            values = screen_df[column].to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1))
            abs_values = np.abs(values)
            q80 = float(np.nanquantile(abs_values, 0.8))
            q90 = float(np.nanquantile(abs_values, 0.9))
            for year in TOP_RESIDUAL_YEARS:
                row = screen_df.loc[screen_df["water_year"] == year]
                if row.empty:
                    raise KeyError(f"Missing top residual year {year} in screening table")
                value = float(row.iloc[0][column])
                zscore = float((value - mean) / std) if std > 0.0 else float("nan")
                base_payload = {
                    "water_year": year,
                    "variable_name": column,
                    "pc": pc,
                    "month": month,
                    "pc_value": value,
                    "zscore_within_37_year_sample": zscore,
                    "sample_mean": mean,
                    "sample_std": std,
                    "abs_value_threshold_80pct": q80,
                    "abs_value_threshold_90pct": q90,
                }
                long_rows.append(base_payload)
                flag_rows.append(
                    {
                        **base_payload,
                        "abs_z_ge_1_0": bool(np.isfinite(zscore) and abs(zscore) >= 1.0),
                        "abs_z_ge_1_5": bool(np.isfinite(zscore) and abs(zscore) >= 1.5),
                        "abs_z_ge_2_0": bool(np.isfinite(zscore) and abs(zscore) >= 2.0),
                        "top_20_percent_abs_value": bool(np.isfinite(value) and abs(value) >= q80),
                        "top_10_percent_abs_value": bool(np.isfinite(value) and abs(value) >= q90),
                    }
                )
    return pd.DataFrame(long_rows), pd.DataFrame(flag_rows)


def draw_heatmap(
    values: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "coolwarm",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    matrix = values.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(values.columns.tolist(), rotation=0)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(values.index.tolist())
    ax.set_title(title)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            cell = matrix[row_index, col_index]
            label = "nan" if not np.isfinite(cell) else f"{cell:.2f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=ax, shrink=0.9, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_top_candidate_scatter(screen_df: pd.DataFrame, top_ranked_df: pd.DataFrame) -> None:
    top_candidates = (
        top_ranked_df.loc[top_ranked_df["ranking_type"] == "abs_pearson_r_with_residual", "variable_name"]
        .drop_duplicates()
        .head(4)
        .tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for ax, column in zip(axes.ravel(), top_candidates):
        x = screen_df[column].to_numpy(dtype=float)
        y = screen_df["residual_obs_minus_pred"].to_numpy(dtype=float)
        ax.scatter(x, y, color="#1f77b4", alpha=0.85)
        for year in TOP_RESIDUAL_YEARS:
            row = screen_df.loc[screen_df["water_year"] == year].iloc[0]
            ax.scatter(float(row[column]), float(row["residual_obs_minus_pred"]), color="#d62728", s=28)
            ax.text(float(row[column]), float(row["residual_obs_minus_pred"]), str(year), fontsize=8, ha="left", va="bottom")
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_title(column)
        ax.set_xlabel("Pacific PC value")
        ax.set_ylabel("Residual obs - pred")
    for ax in axes.ravel()[len(top_candidates) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(TOP_SCATTER_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    residual_df, residual_summary = load_base_residuals()
    base_metrics = residual_summary.get("base_model_metrics", {})
    top5_abs = (
        residual_df[["water_year", "abs_residual"]]
        .sort_values(["abs_residual", "water_year"], ascending=[False, True])
        .head(5)
        .to_dict(orient="records")
    )

    if not PACIFIC_PC_SOURCE_FILE.exists():
        raise FileNotFoundError("Missing canonical Pacific EOF/PC source file: {}".format(PACIFIC_PC_SOURCE_FILE))
    aligned_pc_df = build_wy_aligned_pc_table(PACIFIC_PC_SOURCE_FILE, residual_df["water_year"].tolist())
    alignment_check = verify_residual_table_alignment(residual_df, aligned_pc_df)

    screen_df = residual_df[EXPECTED_BASE_COLUMNS].merge(aligned_pc_df, on="water_year", how="inner")
    if len(screen_df) != len(residual_df):
        raise ValueError(f"Expected {len(residual_df)} aligned years, found {len(screen_df)}")

    correlations_df = build_correlation_table(screen_df, PACIFIC_PC_SOURCE_FILE)
    top_ranked_df = build_top_ranked_table(correlations_df)
    top_year_values_df, top_year_flags_df = build_top_year_tables(screen_df)

    correlations_df.to_csv(CORRELATIONS_CSV, index=False)
    top_ranked_df.to_csv(TOP_RANKED_CSV, index=False)
    top_year_values_df.to_csv(TOP_YEAR_VALUES_CSV, index=False)
    top_year_flags_df.to_csv(TOP_YEAR_FLAGS_CSV, index=False)

    residual_heatmap = (
        correlations_df.pivot(index="pc", columns="month", values="pearson_r_with_residual")
        .reindex(index=range(1, 7), columns=MONTHS)
    )
    abs_residual_heatmap = (
        correlations_df.pivot(index="pc", columns="month", values="pearson_r_with_abs_residual")
        .reindex(index=range(1, 7), columns=MONTHS)
    )
    zscore_heatmap = (
        top_year_values_df.assign(year_label=lambda df: df["water_year"].astype(str))
        .pivot(index="year_label", columns="variable_name", values="zscore_within_37_year_sample")
        .reindex(index=[str(year) for year in TOP_RESIDUAL_YEARS])
    )

    draw_heatmap(
        residual_heatmap,
        title="Pearson r: Pacific PCs vs Z1+Z2 residual",
        colorbar_label="Pearson r",
        output_path=RESIDUAL_HEATMAP_PNG,
        vmin=-0.75,
        vmax=0.75,
    )
    draw_heatmap(
        abs_residual_heatmap,
        title="Pearson r: Pacific PCs vs |Z1+Z2 residual|",
        colorbar_label="Pearson r",
        output_path=ABS_RESIDUAL_HEATMAP_PNG,
        vmin=-0.75,
        vmax=0.75,
    )
    draw_heatmap(
        zscore_heatmap,
        title="Top residual-year Pacific PC z-scores",
        colorbar_label="z-score",
        output_path=TOP_YEAR_Z_HEATMAP_PNG,
        vmin=-3.0,
        vmax=3.0,
    )
    draw_top_candidate_scatter(screen_df, top_ranked_df)

    summary_payload = {
        "base_residual_verification": {
            "residual_table_csv": str(RESIDUAL_TABLE_CSV),
            "residual_summary_json": str(RESIDUAL_SUMMARY_JSON),
            "residual_table_columns_used": EXPECTED_BASE_COLUMNS,
            "number_of_years": int(len(residual_df)),
            "water_year_min": int(residual_df["water_year"].min()),
            "water_year_max": int(residual_df["water_year"].max()),
            "top5_abs_residual_years": top5_abs,
            "base_model_metrics_from_json": base_metrics,
        },
        "pacific_pc_source": {
            "canonical_eof_pc_file": str(PACIFIC_PC_SOURCE_FILE),
            "source_variable": "pc(time, mode)",
            "scipy_available_for_pvalues": stats is not None,
            "water_year_alignment_rule": "WY y uses Sep(y-1), Oct(y-1), Nov(y-1), Dec(y-1), Jan(y), Feb(y), Mar(y)",
            "alignment_check_against_residual_table": alignment_check,
        },
        "top_ranked_predictors": {
            ranking_type: (
                top_ranked_df.loc[top_ranked_df["ranking_type"] == ranking_type, ["rank", "variable_name", "ranking_value"]]
                .to_dict(orient="records")
            )
            for ranking_type in sorted(top_ranked_df["ranking_type"].unique())
        },
        "requested_top_residual_years": TOP_RESIDUAL_YEARS,
        "output_dir": str(OUTPUT_DIR),
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
