#!/usr/bin/env python3
"""Run CPM-to-SWE attribution analysis using saved ERA5 CPM artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(os.environ.get("PWD", str(Path(__file__).absolute().parents[1])))
DEFAULT_CPM_DIR = PROJECT_ROOT / "artifacts" / "cpm_era5_reproduction"
DEFAULT_SWE_TABLE = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo" / "z1z2_amv_k5_predictor_table.csv"
DEFAULT_UPSTREAM_SUMMARY = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo" / "full37_patch_predictor_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_swe_attribution"


def require_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from scipy import stats
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing required dependency: {exc}") from exc
    return plt, pd, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpm-artifact-dir", type=Path, default=DEFAULT_CPM_DIR)
    parser.add_argument("--swe-table-path", type=Path, default=DEFAULT_SWE_TABLE)
    parser.add_argument("--upstream-summary-path", type=Path, default=DEFAULT_UPSTREAM_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-water-year", type=int, default=1985)
    parser.add_argument("--end-water-year", type=int, default=2016)
    parser.add_argument("--run-monthly-analysis", action="store_true", default=True)
    parser.add_argument("--skip-monthly-analysis", dest="run_monthly_analysis", action="store_false")
    parser.add_argument("--run-robustness", action="store_true", default=True)
    parser.add_argument("--skip-robustness", dest="run_robustness", action="store_false")
    return parser.parse_args()


def fisher_ci(r: float, n: int, stats) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(0.975)
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x, ddof=1)


def linear_detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return corrcoef_safe(x[:-1], x[1:])


def effective_sample_size(n: int, r1_x: float, r1_y: float) -> float:
    if not np.isfinite(r1_x) or not np.isfinite(r1_y):
        return float(n)
    denom = 1.0 + r1_x * r1_y
    num = 1.0 - r1_x * r1_y
    if denom <= 0.0:
        return float(n)
    neff = n * num / denom
    return float(min(max(neff, 3.0), n))


def bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.asarray(pvals)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * m / rank)
        q[i] = val
        prev = val
    out = np.empty(m, dtype=float)
    out[order] = q
    return out.tolist()


def ols_fit(X: np.ndarray, y: np.ndarray, stats):
    n = len(y)
    p = X.shape[1]
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    sigma2 = rss / (n - p)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=n - p))
    hat = X @ xtx_inv @ X.T
    leverage = np.diag(hat)
    cooks = (resid**2 / (p * sigma2)) * (leverage / (1.0 - leverage) ** 2)
    return {
        "beta": beta,
        "fitted": fitted,
        "resid": resid,
        "r2": r2,
        "adj_r2": adj_r2,
        "se": se,
        "pvals": pvals,
        "cooks_distance": cooks,
    }


def df_to_markdown(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def plot_scatter(x, y, xlabel, ylabel, title, stats_text, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    ax.scatter(x, y, color="tab:blue", alpha=0.85)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xx, intercept + slope * xx, color="tab:red", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_standardized_timeseries(df, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    years = df["water_year"].to_numpy()
    for col, color, label in [
        ("cpm_pc3_nov_apr_mean", "tab:green", "CPM"),
        ("obs_swe", "tab:blue", "Observed SWE"),
    ]:
        ax.plot(years, zscore(df[col].to_numpy()), marker="o", linewidth=1.5, label=label, color=color)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Water year")
    ax.set_ylabel("Standardized value")
    ax.set_title("Standardized wet-season CPM and observed SWE (visualization only)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loyo_scatter(obs, pred, stats_text, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(obs, pred, color="tab:purple")
    lo = min(np.min(obs), np.min(pred))
    hi = max(np.max(obs), np.max(pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    coef = np.polyfit(obs, pred, 1)
    xx = np.linspace(lo, hi, 100)
    ax.plot(xx, coef[1] + coef[0] * xx, color="tab:red", linewidth=1.2)
    ax.set_xlabel("Observed SWE")
    ax.set_ylabel("LOYO predicted SWE")
    ax.set_title("LOYO CPM-only SWE model")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_correlations(monthly_df, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(monthly_df))
    ax.bar(x, monthly_df["pearson_r"], color="teal")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_df["month"])
    ax.set_ylabel("Pearson r")
    ax.set_title("Monthly CPM vs observed SWE correlations")
    for xi, row in zip(x, monthly_df.itertuples(index=False)):
        ax.text(xi, row.pearson_r, f"p={row.p_value:.3f}\nq={row.q_value_bh:.3f}", ha="center", va="bottom" if row.pearson_r >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    plt, pd, stats = require_deps()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    upstream_summary = json.loads(args.upstream_summary_path.read_text())
    swe_table = pd.read_csv(args.swe_table_path)
    pc_df = pd.read_csv(args.cpm_artifact_dir / "era5_z500_pc1to6_daily.csv", parse_dates=["date"])

    swe = swe_table[["water_year", "obs_swe"]].copy()
    swe["water_year"] = swe["water_year"].astype(int)
    swe = swe[(swe["water_year"] >= args.start_water_year) & (swe["water_year"] <= args.end_water_year)].sort_values("water_year")
    swe.to_csv(args.output_dir / "observed_swe_by_water_year.csv", index=False)

    cpm_daily = pc_df[["date", "water_year", "PC3"]].copy().rename(columns={"PC3": "cpm_pc3_daily"})
    annual_cpm = (
        cpm_daily.groupby("water_year", as_index=False)
        .agg(
            cpm_pc3_nov_apr_mean=("cpm_pc3_daily", "mean"),
            number_of_days=("cpm_pc3_daily", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .sort_values("water_year")
    )
    annual_cpm = annual_cpm[(annual_cpm["water_year"] >= args.start_water_year) & (annual_cpm["water_year"] <= args.end_water_year)]
    annual_cpm.to_csv(args.output_dir / "cpm_pc3_nov_apr_by_water_year.csv", index=False)

    overlap = swe.merge(annual_cpm, on="water_year", how="inner").sort_values("water_year").reset_index(drop=True)
    overlap.to_csv(args.output_dir / "cpm_swe_overlap_table.csv", index=False)

    years = overlap["water_year"].to_numpy()
    swe_values = overlap["obs_swe"].to_numpy(dtype=float)
    cpm_values = overlap["cpm_pc3_nov_apr_mean"].to_numpy(dtype=float)
    n = len(overlap)
    if n < 5:
        raise ValueError("Too few overlap years for CPM-to-SWE attribution analysis.")

    pearson_r, pearson_p = stats.pearsonr(cpm_values, swe_values)
    spearman_rho, spearman_p = stats.spearmanr(cpm_values, swe_values)
    slope, intercept, _, _, _ = stats.linregress(cpm_values, swe_values)
    ci_lo, ci_hi = fisher_ci(float(pearson_r), n, stats)

    plot_scatter(
        cpm_values,
        swe_values,
        "Wet-season CPM (PC3 Nov-Apr mean)",
        "Observed SWE",
        "CPM vs observed SWE",
        f"r={pearson_r:.3f}\np={pearson_p:.3g}\nn={n}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        args.output_dir / "cpm_vs_swe_scatter.png",
        plt,
    )
    plot_standardized_timeseries(overlap[["water_year", "cpm_pc3_nov_apr_mean", "obs_swe"]], args.output_dir / "cpm_swe_standardized_timeseries.png", plt)

    X = np.column_stack([np.ones(n), cpm_values])
    fit = ols_fit(X, swe_values, stats)
    beta0, beta1 = fit["beta"]
    loyo_preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xi = np.column_stack([np.ones(mask.sum()), cpm_values[mask]])
        yi = swe_values[mask]
        beta_i = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        loyo_preds.append(float(np.array([1.0, cpm_values[i]]) @ beta_i))
    loyo_preds = np.asarray(loyo_preds)
    loyo_r = corrcoef_safe(swe_values, loyo_preds)
    loyo_r2 = 1.0 - np.sum((swe_values - loyo_preds) ** 2) / np.sum((swe_values - np.mean(swe_values)) ** 2)
    loyo_rmse = float(np.sqrt(np.mean((swe_values - loyo_preds) ** 2)))
    loyo_mae = float(np.mean(np.abs(swe_values - loyo_preds)))
    pd.DataFrame({"water_year": years, "obs_swe": swe_values, "pred_swe": loyo_preds}).to_csv(args.output_dir / "cpm_swe_loyo_predictions.csv", index=False)
    plot_loyo_scatter(
        swe_values,
        loyo_preds,
        f"r={loyo_r:.3f}\nR²={loyo_r2:.3f}\nRMSE={loyo_rmse:.4f}\nMAE={loyo_mae:.4f}",
        args.output_dir / "cpm_swe_loyo_scatter.png",
        plt,
    )

    model_summary = pd.DataFrame(
        [
            {
                "model": "SWE ~ CPM",
                "intercept": beta0,
                "beta_cpm": beta1,
                "se_intercept": fit["se"][0],
                "se_beta_cpm": fit["se"][1],
                "p_intercept": fit["pvals"][0],
                "p_beta_cpm": fit["pvals"][1],
                "in_sample_R2": fit["r2"],
                "adjusted_R2": fit["adj_r2"],
                "LOYO_r": loyo_r,
                "LOYO_R2": loyo_r2,
                "LOYO_RMSE": loyo_rmse,
                "LOYO_MAE": loyo_mae,
            }
        ]
    )
    model_summary.to_csv(args.output_dir / "cpm_swe_model_summary.csv", index=False)

    robustness_rows = []
    if args.run_robustness:
        detrended_r = corrcoef_safe(linear_detrend(cpm_values), linear_detrend(swe_values))
        loo_rs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            loo_rs.append(corrcoef_safe(cpm_values[mask], swe_values[mask]))
        loo_rs = np.asarray(loo_rs)
        influence_idx = int(np.nanargmax(np.abs(loo_rs - pearson_r)))
        r1_cpm = lag1_autocorr(cpm_values)
        r1_swe = lag1_autocorr(swe_values)
        neff = effective_sample_size(n, r1_cpm, r1_swe)
        if abs(pearson_r) < 1.0:
            t_eff = pearson_r * math.sqrt((neff - 2.0) / max(1.0e-12, 1.0 - pearson_r**2))
            p_eff = float(2.0 * (1.0 - stats.t.cdf(abs(t_eff), df=max(1.0, neff - 2.0))))
        else:
            p_eff = 0.0
        robustness_rows.append(
            {
                "raw_r": pearson_r,
                "detrended_r": detrended_r,
                "leave_one_out_min_r": float(np.nanmin(loo_rs)),
                "leave_one_out_max_r": float(np.nanmax(loo_rs)),
                "most_influential_year": int(years[influence_idx]),
                "max_leave_one_out_change_in_r": float(np.nanmax(np.abs(loo_rs - pearson_r))),
                "max_cooks_distance": float(np.nanmax(fit["cooks_distance"])),
                "lag1_autocorr_cpm": r1_cpm,
                "lag1_autocorr_swe": r1_swe,
                "effective_sample_size": neff,
                "effective_sample_size_p_value": p_eff,
            }
        )
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(args.output_dir / "cpm_swe_robustness_summary.csv", index=False)

    monthly_df = pd.DataFrame()
    if args.run_monthly_analysis:
        cpm_daily["month"] = cpm_daily["date"].dt.month
        month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
        monthly_rows = []
        for month in [11, 12, 1, 2, 3, 4]:
            sub = cpm_daily[cpm_daily["month"] == month]
            month_mean = sub.groupby("water_year", as_index=False)["cpm_pc3_daily"].mean().rename(columns={"cpm_pc3_daily": "cpm_month_mean"})
            joined = swe.merge(month_mean, on="water_year", how="inner").sort_values("water_year")
            r, p = stats.pearsonr(joined["cpm_month_mean"], joined["obs_swe"])
            monthly_rows.append({"month": month_names[month], "n": int(len(joined)), "pearson_r": float(r), "p_value": float(p)})
        monthly_df = pd.DataFrame(monthly_rows)
        monthly_df["q_value_bh"] = bh_adjust(monthly_df["p_value"].tolist())
        monthly_df.to_csv(args.output_dir / "cpm_monthly_swe_correlations.csv", index=False)
        plot_monthly_correlations(monthly_df, args.output_dir / "cpm_monthly_swe_correlations.png", plt)

    source_audit = f"""# CPM to SWE Source Audit

- CPM artifact directory: `{args.cpm_artifact_dir}`
- CPM daily PC source: `{args.cpm_artifact_dir / 'era5_z500_pc1to6_daily.csv'}`
- CPM definition used here: ERA5 wet-season `PC3`, aggregated to one November-April mean per water year
- SWE source table: `{args.swe_table_path}`
- SWE column used here: `obs_swe`
- Upstream summary inspected: `{args.upstream_summary_path}`
- Upstream target source recorded in metadata: `{upstream_summary['target_source']}`
- SWE target interpretation from saved pipeline: Sierra April 1 SWE anomaly, standardized, aligned by water year
- Shared overlap used here: `WY{int(years[0])}-WY{int(years[-1])}` with `n={n}`
- Water-year alignment:
  - CPM for WY `y` = mean daily PC3 from `Nov(y-1)` through `Apr(y)`
  - SWE for WY `y` = saved `obs_swe` target for April 1 of WY `y`
- No SWE retraining or predictor reselection was performed in this run.
"""
    (args.output_dir / "cpm_swe_source_audit.md").write_text(source_audit)

    primary_table = pd.DataFrame(
        [
            {
                "Predictor": "CPM",
                "Years": f"WY{int(years[0])}-WY{int(years[-1])}",
                "n": int(n),
                "Pearson r": float(pearson_r),
                "p-value": float(pearson_p),
                "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                "Spearman rho": float(spearman_rho),
                "Detrended r": float(robustness_df['detrended_r'].iloc[0]) if not robustness_df.empty else float("nan"),
            }
        ]
    )
    joint_table = pd.DataFrame(
        [
            {
                "Model": "SWE ~ CPM",
                "In-sample R²": float(fit["r2"]),
                "Adjusted R²": float(fit["adj_r2"]),
                "LOYO r": float(loyo_r),
                "LOYO R²": float(loyo_r2),
                "LOYO RMSE": float(loyo_rmse),
                "LOYO MAE": float(loyo_mae),
            }
        ]
    )

    report = f"""# CPM to SWE Attribution Report

## Inputs

- CPM artifact directory: `{args.cpm_artifact_dir}`
- SWE source table: `{args.swe_table_path}`
- Upstream metadata file: `{args.upstream_summary_path}`

## Source audit

- CPM index used here: wet-season ERA5 `PC3` from the completed CPM reproduction
- Annual CPM definition: mean daily `PC3` over November-April for each water year
- SWE target used here: `obs_swe` from the canonical saved predictor table
- Upstream target source recorded in metadata: `{upstream_summary['target_source']}`
- The saved target is interpreted from project metadata as the standardized Sierra April 1 SWE anomaly by water year

## Overlap

- Overlap years: `WY{int(years[0])}-WY{int(years[-1])}`
- Number of paired years: `{n}`
- Annual CPM table: `{args.output_dir / 'cpm_pc3_nov_apr_by_water_year.csv'}`
- SWE table: `{args.output_dir / 'observed_swe_by_water_year.csv'}`
- Overlap table: `{args.output_dir / 'cpm_swe_overlap_table.csv'}`

## Primary result table

{df_to_markdown(primary_table)}

## Model result table

{df_to_markdown(joint_table)}

## Single-predictor model details

- intercept = {beta0:.6f}
- beta_cpm = {beta1:.6f}
- SE(intercept) = {fit['se'][0]:.6f}
- SE(beta_cpm) = {fit['se'][1]:.6f}
- p(beta_cpm) = {fit['pvals'][1]:.6f}
- Spearman p-value = {spearman_p:.6f}

## Robustness diagnostics

{df_to_markdown(robustness_df) if not robustness_df.empty else 'Not run.'}

## Monthly timing analysis

{df_to_markdown(monthly_df) if not monthly_df.empty else 'Not run.'}

## Interpretation

- This is a direct CPM-to-SWE attribution check, mirroring the temporal correlation and regression style used earlier.
- The result should be interpreted as whether wet-season CPM amplitude is associated with the Sierra SWE target over the overlapping years.
- No new EOF selection, no SWE-based reselection of climate predictors, and no retraining of the prior SST-based SWE models were introduced here.

## Artifact paths

- Source audit: `{args.output_dir / 'cpm_swe_source_audit.md'}`
- Scatter plot: `{args.output_dir / 'cpm_vs_swe_scatter.png'}`
- Standardized time series: `{args.output_dir / 'cpm_swe_standardized_timeseries.png'}`
- LOYO predictions: `{args.output_dir / 'cpm_swe_loyo_predictions.csv'}`
- LOYO scatter: `{args.output_dir / 'cpm_swe_loyo_scatter.png'}`
- Model summary: `{args.output_dir / 'cpm_swe_model_summary.csv'}`
- Robustness summary: `{args.output_dir / 'cpm_swe_robustness_summary.csv'}`
- Monthly timing CSV: `{args.output_dir / 'cpm_monthly_swe_correlations.csv'}`
- Monthly timing plot: `{args.output_dir / 'cpm_monthly_swe_correlations.png'}`

## Direct conclusion

- Does CPM show evidence of association with SWE?
  - Answer from the raw annual correlation above.
- Is the evidence stable in leave-one-year-out checks?
  - See the LOYO and robustness diagnostics above.
"""
    (args.output_dir / "report.md").write_text(report)

    print(f"CPM source table: {args.cpm_artifact_dir / 'era5_z500_pc1to6_daily.csv'}")
    print(f"SWE source table: {args.swe_table_path}")
    print(f"Overlap years: WY{int(years[0])}-WY{int(years[-1])}")
    print(f"Number of years: {n}")
    print(f"r(CPM, SWE): {pearson_r:.6f}")
    print(f"Spearman rho(CPM, SWE): {spearman_rho:.6f}")
    print(f"LOYO correlation: {loyo_r:.6f}")
    print(f"Main artifact directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
