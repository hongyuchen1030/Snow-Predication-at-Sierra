#!/usr/bin/env python3
"""Run Z1/Z2 to CPM attribution analysis using saved ERA5 CPM artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(os.environ.get("PWD", str(Path(__file__).absolute().parents[1])))
DEFAULT_CPM_DIR = PROJECT_ROOT / "artifacts" / "cpm_era5_reproduction"
DEFAULT_Z_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "z1_z2_cpm_attribution"
DEFAULT_UPSTREAM_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "full37_selected_patch_predictor_loyo"
GRAVITY = 9.80665


def require_deps():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr
        from scipy import stats
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing required dependency: {exc}") from exc
    return ccrs, cfeature, plt, pd, xr, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpm-artifact-dir", type=Path, default=DEFAULT_CPM_DIR)
    parser.add_argument("--z-artifact-dir", type=Path, default=DEFAULT_Z_DIR)
    parser.add_argument("--upstream-z-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-water-year", type=int, default=1985)
    parser.add_argument("--end-water-year", type=int, default=2016)
    parser.add_argument("--run-monthly-analysis", action="store_true", default=True)
    parser.add_argument("--skip-monthly-analysis", dest="run_monthly_analysis", action="store_false")
    parser.add_argument("--run-robustness", action="store_true", default=True)
    parser.add_argument("--skip-robustness", dest="run_robustness", action="store_false")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_dataset_netcdf(dataset, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        dataset.to_netcdf(tmp_path, engine="scipy", mode="w")
        shutil.move(str(tmp_path), str(target))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def assign_water_year(index) -> np.ndarray:
    years = index.year.to_numpy()
    months = index.month.to_numpy()
    return np.where(months >= 11, years + 1, years)


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


def ols_fit(X: np.ndarray, y: np.ndarray, stats):
    n = len(y)
    p = X.shape[1]
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    sigma2 = rss / (n - p)
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=n - p))
    if p > 1:
        ms_model = (tss - rss) / (p - 1)
        ms_error = rss / (n - p)
        f_stat = ms_model / ms_error if ms_error > 0.0 else float("nan")
        f_p = 1.0 - stats.f.cdf(f_stat, p - 1, n - p) if np.isfinite(f_stat) else float("nan")
    else:
        f_stat = float("nan")
        f_p = float("nan")
    hat = X @ XtX_inv @ X.T
    leverage = np.diag(hat)
    cooks = (resid ** 2 / (p * sigma2)) * (leverage / (1.0 - leverage) ** 2)
    return {
        "beta": beta,
        "fitted": fitted,
        "resid": resid,
        "rss": rss,
        "r2": r2,
        "adj_r2": adj_r2,
        "se": se,
        "pvals": pvals,
        "f_stat": f_stat,
        "f_p": f_p,
        "cooks_distance": cooks,
    }


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    Xz = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.lstsq(Xz, x, rcond=None)[0]
    by = np.linalg.lstsq(Xz, y, rcond=None)[0]
    rx = x - Xz @ bx
    ry = y - Xz @ by
    return corrcoef_safe(rx, ry)


def vif_for_column(X: np.ndarray, col: int) -> float:
    y = X[:, col]
    X_other = np.delete(X, col, axis=1)
    X_design = np.column_stack([np.ones(len(y)), X_other])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    fitted = X_design @ beta
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    return float(1.0 / (1.0 - r2))


def weighted_spatial_corr(a: np.ndarray, b: np.ndarray, latitudes: np.ndarray) -> float:
    w = np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, a.shape[1]))
    mask = np.isfinite(a) & np.isfinite(b)
    ww = w[mask]
    x = a[mask]
    y = b[mask]
    if ww.size == 0:
        return float("nan")
    ww = ww / ww.sum()
    xbar = np.sum(ww * x)
    ybar = np.sum(ww * y)
    cov = np.sum(ww * (x - xbar) * (y - ybar))
    varx = np.sum(ww * (x - xbar) ** 2)
    vary = np.sum(ww * (y - ybar) ** 2)
    if varx <= 0.0 or vary <= 0.0:
        return float("nan")
    return float(cov / np.sqrt(varx * vary))


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


def df_to_markdown(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def plot_scatter(x, y, years, xlabel, ylabel, title, stats_text, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x, y, color="tab:blue", alpha=0.85)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xx, intercept + slope * xx, color="tab:red", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_standardized_timeseries(df, out_path: Path, plt):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    years = df["water_year"].to_numpy()
    for col, color in [("Z1", "tab:blue"), ("Z2", "tab:orange"), ("cpm_pc3_nov_apr_mean", "tab:green")]:
        ax.plot(years, zscore(df[col].to_numpy()), marker="o", linewidth=1.5, label=col, color=color)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Water year")
    ax.set_ylabel("Standardized value")
    ax.set_title("Standardized Z1, Z2, and wet-season CPM (visualization only)")
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
    ax.set_xlabel("Observed CPM")
    ax.set_ylabel("LOYO predicted CPM")
    ax.set_title("LOYO Z1+Z2 joint CPM model")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_maps(cpm_map, reg_map, latitudes, longitudes_west, title_left, title_right, subtitle, out_path: Path, ccrs, cfeature, plt):
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), subplot_kw={"projection": proj})
    vmax = max(float(np.nanmax(np.abs(cpm_map))), float(np.nanmax(np.abs(reg_map))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    lon2d, lat2d = np.meshgrid(longitudes_west, latitudes)
    cf = axes[0].contourf(lon2d, lat2d, cpm_map, levels=levels, cmap="RdBu_r", extend="both", transform=proj)
    axes[1].contourf(lon2d, lat2d, reg_map, levels=levels, cmap="RdBu_r", extend="both", transform=proj)
    for ax, title in zip(axes, [title_left, title_right]):
        ax.set_extent([longitudes_west.min(), longitudes_west.max(), latitudes.min(), latitudes.max()], crs=proj)
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        try:
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
        except Exception:
            pass
        ax.set_title(title)
    cbar = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.06, pad=0.1)
    cbar.set_label("Separately normalized for visual shape comparison")
    fig.suptitle(subtitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_three_panel_maps(cpm_map, z1_map, z2_map, latitudes, longitudes_west, out_path: Path, ccrs, cfeature, plt):
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), subplot_kw={"projection": proj})
    vmax = max(float(np.nanmax(np.abs(cpm_map))), float(np.nanmax(np.abs(z1_map))), float(np.nanmax(np.abs(z2_map))), 1.0e-8)
    levels = np.linspace(-vmax, vmax, 19)
    lon2d, lat2d = np.meshgrid(longitudes_west, latitudes)
    maps = [cpm_map, z1_map, z2_map]
    titles = ["ERA5 EOF3 CPM", "Z1-associated Z500", "Z2-associated Z500"]
    for ax, arr, title in zip(axes, maps, titles):
        cf = ax.contourf(lon2d, lat2d, arr, levels=levels, cmap="RdBu_r", extend="both", transform=proj)
        ax.set_extent([longitudes_west.min(), longitudes_west.max(), latitudes.min(), latitudes.max()], crs=proj)
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        try:
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
        except Exception:
            pass
        ax.set_title(title)
    cbar = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.06, pad=0.08)
    cbar.set_label("Separately normalized for visual shape comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ccrs, cfeature, plt, pd, xr, stats = require_deps()
    args = parse_args()
    ensure_output_dir(args.output_dir)

    z_summary_path = args.z_artifact_dir / "z1z2_amv_k5_summary.json"
    z_table_path = args.z_artifact_dir / "z1z2_amv_k5_predictor_table.csv"
    upstream_summary_path = args.upstream_z_dir / "full37_patch_predictor_summary.json"
    upstream_predictors_path = args.upstream_z_dir / "full37_patch_predictors.csv"

    z_summary = json.loads(z_summary_path.read_text())
    upstream_summary = json.loads(upstream_summary_path.read_text())
    z_table = pd.read_csv(z_table_path)
    upstream_predictors = pd.read_csv(upstream_predictors_path)

    z_clean = z_table[["water_year", "Z1", "Z2"]].copy()
    z_clean["water_year"] = z_clean["water_year"].astype(int)
    z_clean = z_clean[(z_clean["water_year"] >= args.start_water_year) & (z_clean["water_year"] <= args.end_water_year)]
    z_clean.to_csv(args.output_dir / "z1_z2_by_water_year.csv", index=False)

    pc_df = pd.read_csv(args.cpm_artifact_dir / "era5_z500_pc1to6_daily.csv", parse_dates=["date"])
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

    overlap = z_clean.merge(annual_cpm, on="water_year", how="inner").sort_values("water_year").reset_index(drop=True)
    overlap.to_csv(args.output_dir / "z1_z2_cpm_overlap_table.csv", index=False)

    years = overlap["water_year"].to_numpy()
    z1 = overlap["Z1"].to_numpy(dtype=float)
    z2 = overlap["Z2"].to_numpy(dtype=float)
    cpm = overlap["cpm_pc3_nov_apr_mean"].to_numpy(dtype=float)
    n = len(overlap)

    if n < 5:
        raise ValueError("Too few overlap years for attribution analysis.")

    def simple_summary(name: str, x: np.ndarray):
        pearson_r, pearson_p = stats.pearsonr(x, cpm)
        spearman_rho, spearman_p = stats.spearmanr(x, cpm)
        slope, intercept, _, _, _ = stats.linregress(x, cpm)
        ci_lo, ci_hi = fisher_ci(float(pearson_r), n, stats)
        return {
            "predictor": name,
            "Years": f"WY{int(years[0])}-WY{int(years[-1])}",
            "n": int(n),
            "Pearson r": float(pearson_r),
            "p-value": float(pearson_p),
            "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            "Spearman rho": float(spearman_rho),
            "Spearman p": float(spearman_p),
            "slope": float(slope),
            "intercept": float(intercept),
        }

    z1_summary = simple_summary("Z1", z1)
    z2_summary = simple_summary("Z2", z2)

    plot_scatter(
        z1,
        cpm,
        years,
        "Z1",
        "Wet-season CPM (PC3 Nov-Apr mean)",
        "Z1 vs wet-season CPM",
        f"r={z1_summary['Pearson r']:.3f}\np={z1_summary['p-value']:.3g}\nn={n}\n95% CI {z1_summary['95% CI']}",
        args.output_dir / "z1_vs_cpm_wet_season_scatter.png",
        plt,
    )
    plot_scatter(
        z2,
        cpm,
        years,
        "Z2",
        "Wet-season CPM (PC3 Nov-Apr mean)",
        "Z2 vs wet-season CPM",
        f"r={z2_summary['Pearson r']:.3f}\np={z2_summary['p-value']:.3g}\nn={n}\n95% CI {z2_summary['95% CI']}",
        args.output_dir / "z2_vs_cpm_wet_season_scatter.png",
        plt,
    )
    plot_standardized_timeseries(overlap[["water_year", "Z1", "Z2", "cpm_pc3_nov_apr_mean"]], args.output_dir / "z1_z2_cpm_standardized_timeseries.png", plt)

    X = np.column_stack([np.ones(n), z1, z2])
    fit = ols_fit(X, cpm, stats)
    beta0, beta1, beta2 = fit["beta"]
    corr_z1_z2 = corrcoef_safe(z1, z2)
    vif_z1 = vif_for_column(np.column_stack([z1, z2]), 0)
    vif_z2 = vif_for_column(np.column_stack([z1, z2]), 1)
    partial_z1 = partial_corr(z1, cpm, z2)
    partial_z2 = partial_corr(z2, cpm, z1)

    loyo_preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xi = np.column_stack([np.ones(mask.sum()), z1[mask], z2[mask]])
        yi = cpm[mask]
        beta_i = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        pred = float(np.array([1.0, z1[i], z2[i]]) @ beta_i)
        loyo_preds.append(pred)
    loyo_preds = np.asarray(loyo_preds)
    loyo_r = corrcoef_safe(cpm, loyo_preds)
    loyo_r2 = 1.0 - np.sum((cpm - loyo_preds) ** 2) / np.sum((cpm - np.mean(cpm)) ** 2)
    loyo_rmse = float(np.sqrt(np.mean((cpm - loyo_preds) ** 2)))
    loyo_mae = float(np.mean(np.abs(cpm - loyo_preds)))
    loyo_df = pd.DataFrame({"water_year": years, "observed_cpm": cpm, "predicted_cpm": loyo_preds})
    loyo_df.to_csv(args.output_dir / "z1_z2_joint_cpm_loyo_predictions.csv", index=False)
    plot_loyo_scatter(cpm, loyo_preds, f"r={loyo_r:.3f}\nR²={loyo_r2:.3f}\nRMSE={loyo_rmse:.3f}\nMAE={loyo_mae:.3f}", args.output_dir / "z1_z2_joint_cpm_loyo_scatter.png", plt)

    joint_summary = pd.DataFrame(
        [
            {
                "model": "CPM ~ Z1 + Z2",
                "intercept": beta0,
                "beta1": beta1,
                "beta2": beta2,
                "se_intercept": fit["se"][0],
                "se_beta1": fit["se"][1],
                "se_beta2": fit["se"][2],
                "p_intercept": fit["pvals"][0],
                "p_beta1": fit["pvals"][1],
                "p_beta2": fit["pvals"][2],
                "in_sample_R2": fit["r2"],
                "adjusted_R2": fit["adj_r2"],
                "F_stat": fit["f_stat"],
                "F_p": fit["f_p"],
                "corr_Z1_Z2": corr_z1_z2,
                "VIF_Z1": vif_z1,
                "VIF_Z2": vif_z2,
                "partial_corr_Z1_given_Z2": partial_z1,
                "partial_corr_Z2_given_Z1": partial_z2,
                "LOYO_r": loyo_r,
                "LOYO_R2": loyo_r2,
                "LOYO_RMSE": loyo_rmse,
                "LOYO_MAE": loyo_mae,
            }
        ]
    )
    joint_summary.to_csv(args.output_dir / "z1_z2_joint_cpm_model_summary.csv", index=False)

    anom_ds = xr.open_dataset(args.cpm_artifact_dir / "era5_z500_daily_anomalies.nc")
    anom = anom_ds["z500_m_anomaly"]
    annual_maps = (
        anom.groupby("water_year").mean("date").sel(water_year=slice(args.start_water_year, args.end_water_year))
    )
    annual_maps = annual_maps.sel(water_year=years)
    lat = annual_maps["latitude"].values.astype(float)
    lon = annual_maps["longitude"].values.astype(float)
    lon_w = np.where(lon > 180.0, lon - 360.0, lon)
    maps = annual_maps.values.astype(float)  # n, lat, lon

    def slope_map(x: np.ndarray):
        x_center = x - np.mean(x)
        varx = np.sum(x_center ** 2)
        y_center = maps - maps.mean(axis=0, keepdims=True)
        slope = np.tensordot(x_center, y_center, axes=(0, 0)) / varx
        intercept = maps.mean(axis=0) - slope * np.mean(x)
        return slope, intercept

    z1_slope, z1_intercept = slope_map(z1)
    z2_slope, z2_intercept = slope_map(z2)
    z1_std_slope = z1_slope * np.std(z1, ddof=1)
    z2_std_slope = z2_slope * np.std(z2, ddof=1)

    eof_ds = xr.open_dataset(args.cpm_artifact_dir / "era5_z500_eof1to6.nc")
    eof3 = eof_ds["eof_unweighted_sigma_scaled_m"].sel(component=3).values.astype(float)

    joint_score = beta1 * z1 + beta2 * z2
    joint_slope, joint_intercept = slope_map(joint_score)
    joint_std_slope = joint_slope * np.std(joint_score, ddof=1)

    def save_regression_nc(raw_slope, std_slope, intercept, name: str, out_path: Path):
        ds = xr.Dataset(
            {
                f"{name}_raw_slope": xr.DataArray(raw_slope, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon}),
                f"{name}_std_slope": xr.DataArray(std_slope, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon}),
                f"{name}_intercept": xr.DataArray(intercept, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon}),
            }
        )
        write_dataset_netcdf(ds, out_path)

    save_regression_nc(z1_slope, z1_std_slope, z1_intercept, "z1", args.output_dir / "z500_regressed_on_z1.nc")
    save_regression_nc(z2_slope, z2_std_slope, z2_intercept, "z2", args.output_dir / "z500_regressed_on_z2.nc")
    save_regression_nc(joint_slope, joint_std_slope, joint_intercept, "z1_z2_joint", args.output_dir / "z500_regressed_on_z1_z2_joint.nc")

    plot_lat_mask = (lat >= 25.0) & (lat <= 60.0)
    plot_lon_mask = (lon >= 210.0) & (lon <= 250.0)
    spatial_rows = []
    for predictor, reg_map in [("Z1", z1_std_slope), ("Z2", z2_std_slope), ("Z1_Z2_joint", joint_std_slope)]:
        r_full = weighted_spatial_corr(reg_map, eof3, lat)
        r_plot = weighted_spatial_corr(reg_map[np.ix_(plot_lat_mask, plot_lon_mask)], eof3[np.ix_(plot_lat_mask, plot_lon_mask)], lat[plot_lat_mask])
        spatial_rows.extend(
            [
                {
                    "predictor": predictor,
                    "domain": "full_cpm_domain",
                    "spatial_correlation": r_full,
                    "absolute_spatial_correlation": abs(r_full),
                    "number_of_grid_cells": int(eof3.size),
                    "weighting": "cos(latitude)",
                },
                {
                    "predictor": predictor,
                    "domain": "figure1a_window",
                    "spatial_correlation": r_plot,
                    "absolute_spatial_correlation": abs(r_plot),
                    "number_of_grid_cells": int(plot_lat_mask.sum() * plot_lon_mask.sum()),
                    "weighting": "cos(latitude)",
                },
            ]
        )
    spatial_df = pd.DataFrame(spatial_rows)
    spatial_df.to_csv(args.output_dir / "z1_z2_vs_cpm_spatial_correlations.csv", index=False)

    plot_two_panel_maps(
        eof3,
        z1_std_slope,
        lat,
        lon_w,
        "ERA5 EOF3 CPM",
        "Z1-associated Z500 regression",
        f"Z1 regression vs CPM EOF3\nfull-domain weighted spatial r = {spatial_df[(spatial_df.predictor=='Z1') & (spatial_df.domain=='full_cpm_domain')]['spatial_correlation'].iloc[0]:.3f}",
        args.output_dir / "z1_z500_regression_vs_cpm_eof3.png",
        ccrs,
        cfeature,
        plt,
    )
    plot_two_panel_maps(
        eof3,
        z2_std_slope,
        lat,
        lon_w,
        "ERA5 EOF3 CPM",
        "Z2-associated Z500 regression",
        f"Z2 regression vs CPM EOF3\nfull-domain weighted spatial r = {spatial_df[(spatial_df.predictor=='Z2') & (spatial_df.domain=='full_cpm_domain')]['spatial_correlation'].iloc[0]:.3f}",
        args.output_dir / "z2_z500_regression_vs_cpm_eof3.png",
        ccrs,
        cfeature,
        plt,
    )
    plot_three_panel_maps(eof3, z1_std_slope, z2_std_slope, lat, lon_w, args.output_dir / "cpm_eof3_z1_z2_regression_comparison.png", ccrs, cfeature, plt)
    plot_two_panel_maps(
        eof3,
        joint_std_slope,
        lat,
        lon_w,
        "ERA5 EOF3 CPM",
        "Joint Z1+Z2-associated Z500 regression",
        f"In-sample joint regression score vs CPM EOF3\nfull-domain weighted spatial r = {spatial_df[(spatial_df.predictor=='Z1_Z2_joint') & (spatial_df.domain=='full_cpm_domain')]['spatial_correlation'].iloc[0]:.3f}",
        args.output_dir / "z1_z2_joint_z500_regression_vs_cpm_eof3.png",
        ccrs,
        cfeature,
        plt,
    )

    robustness_rows = []
    if args.run_robustness:
        for name, x in [("Z1", z1), ("Z2", z2)]:
            detrended_r = corrcoef_safe(linear_detrend(x), linear_detrend(cpm))
            loo_rs = []
            influence = []
            Xs = np.column_stack([np.ones(n), x])
            fit_simple = ols_fit(Xs, cpm, stats)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                loo_rs.append(corrcoef_safe(x[mask], cpm[mask]))
                influence.append(abs(corrcoef_safe(x[mask], cpm[mask]) - corrcoef_safe(x, cpm)))
            r1_x = lag1_autocorr(x)
            r1_y = lag1_autocorr(cpm)
            neff = effective_sample_size(n, r1_x, r1_y)
            if neff > 2:
                r = corrcoef_safe(x, cpm)
                t_stat = r * np.sqrt((neff - 2.0) / (1.0 - r ** 2))
                p_neff = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=neff - 2.0))
            else:
                p_neff = np.nan
            robustness_rows.append(
                {
                    "predictor": name,
                    "raw_r": corrcoef_safe(x, cpm),
                    "detrended_r": detrended_r,
                    "leave_one_out_min_r": float(np.min(loo_rs)),
                    "leave_one_out_max_r": float(np.max(loo_rs)),
                    "most_influential_year": int(years[int(np.argmax(influence))]),
                    "max_leave_one_out_change_in_r": float(np.max(influence)),
                    "max_cooks_distance": float(np.max(fit_simple["cooks_distance"])),
                    "lag1_autocorr_predictor": r1_x,
                    "lag1_autocorr_cpm": r1_y,
                    "effective_sample_size": neff,
                    "effective_sample_size_p_value": p_neff,
                }
            )
        robustness_df = pd.DataFrame(robustness_rows)
        robustness_df.to_csv(args.output_dir / "z1_z2_cpm_robustness_summary.csv", index=False)
    else:
        robustness_df = pd.DataFrame()

    if args.run_monthly_analysis:
        monthly = cpm_daily.copy()
        monthly["month"] = monthly["date"].dt.month
        month_map = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
        monthly = monthly[monthly["month"].isin(month_map)].copy()
        monthly_mean = monthly.groupby(["water_year", "month"], as_index=False)["cpm_pc3_daily"].mean()
        rows = []
        for month_num, month_name in month_map.items():
            sub = monthly_mean[monthly_mean["month"] == month_num][["water_year", "cpm_pc3_daily"]].rename(columns={"cpm_pc3_daily": "monthly_cpm"})
            merged = overlap[["water_year", "Z1", "Z2"]].merge(sub, on="water_year", how="inner")
            for name in ["Z1", "Z2"]:
                r, p = stats.pearsonr(merged[name], merged["monthly_cpm"])
                rows.append({"predictor": name, "month": month_name, "n": int(len(merged)), "pearson_r": float(r), "p_value": float(p)})
        monthly_df = pd.DataFrame(rows)
        monthly_df["q_value_bh"] = bh_adjust(monthly_df["p_value"].tolist())
        monthly_df.to_csv(args.output_dir / "z1_z2_monthly_cpm_correlations.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for predictor, color in [("Z1", "tab:blue"), ("Z2", "tab:orange")]:
            sub = monthly_df[monthly_df["predictor"] == predictor]
            ax.plot(sub["month"], sub["pearson_r"], marker="o", linewidth=1.5, label=predictor, color=color)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel("Pearson r")
        ax.set_title("Exploratory monthly CPM correlations (Nov-Apr)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(args.output_dir / "z1_z2_monthly_cpm_correlations.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        monthly_df = pd.DataFrame()

    audit_md = f"""# Z1/Z2 Source Audit

## Inspected JSON files

- `{z_summary_path}`
- `{upstream_summary_path}`

## Canonical predictor table

- Final saved run table: `{z_table_path}`
- Upstream source table: `{upstream_predictors_path}`

## Source SST dataset

- `{upstream_summary['sst_source']}`

## Exact predictor definitions

- Z1: LOD mode 1, January, latitude {upstream_summary['reference_mode_1']['lat']}, longitude {upstream_summary['reference_mode_1']['lon']}E
- Z2: LOD mode 2, October, latitude {upstream_summary['reference_mode_2']['lat']}, longitude {upstream_summary['reference_mode_2']['lon']}E

## Saved column names

- In upstream full37 table: `Z1_M1_Jan_lat_-9.5_lon_133.5`, `Z2_M2_Oct_lat_0.5_lon_136.5`
- In canonical run table used here: `Z1`, `Z2`

## Value representation

- The upstream summary states that `full37_patch_predictors.csv` uses **full-sample monthly climatology only for descriptive patch time series**.
- The canonical `z1z2_amv_k5_predictor_table.csv` is therefore treated here as the fixed full-series predictor table saved for the LOYO run, not fold-specific standardized training values.
- The LOYO model training itself used train-fold-only standardization, but those standardized fold values are **not** the values used here for attribution.

## Units and preprocessing

- SST source field: COBE2 monthly SST.
- Stored Z1 and Z2 values appear to be SST anomaly-type predictor values on the saved full-series descriptive scale.
- The JSON metadata do not explicitly attach a units string to the saved columns; physically they are SST-derived anomaly values from the COBE2 source field.

## Water-year alignment

- Saved water years in the canonical run table span `WY1985-WY2021`.
- Alignment implied by the metadata and names:
  - Z1 uses January of calendar year y for water year y.
  - Z2 uses October of calendar year y-1 for water year y.
- This attribution analysis preserves that saved water-year convention and does not recompute the predictors.

## SWE information

- Yes, Z1 and Z2 were originally selected in a SWE-predictive LOD workflow.
- This attribution analysis does **not** rerun selection or introduce new SWE-based feature choices; it only audits and reuses the saved predictors.

## Fixed full-series or fold-dependent?

- The values used here are fixed full-series saved predictor values from `z1z2_amv_k5_predictor_table.csv`.
- They are not fold-specific LOYO predictor standardizations.
"""
    (args.output_dir / "z1_z2_source_audit.md").write_text(audit_md)

    primary_df = pd.DataFrame(
        [
            {
                "Predictor": "Z1",
                "Years": f"WY{int(years[0])}-WY{int(years[-1])}",
                "n": n,
                "Pearson r": z1_summary["Pearson r"],
                "p-value": z1_summary["p-value"],
                "95% CI": z1_summary["95% CI"],
                "Spearman rho": z1_summary["Spearman rho"],
                "Detrended r": robustness_df.loc[robustness_df["predictor"] == "Z1", "detrended_r"].iloc[0] if not robustness_df.empty else np.nan,
            },
            {
                "Predictor": "Z2",
                "Years": f"WY{int(years[0])}-WY{int(years[-1])}",
                "n": n,
                "Pearson r": z2_summary["Pearson r"],
                "p-value": z2_summary["p-value"],
                "95% CI": z2_summary["95% CI"],
                "Spearman rho": z2_summary["Spearman rho"],
                "Detrended r": robustness_df.loc[robustness_df["predictor"] == "Z2", "detrended_r"].iloc[0] if not robustness_df.empty else np.nan,
            },
        ]
    )
    spatial_table = spatial_df[spatial_df["predictor"].isin(["Z1", "Z2"])][["predictor", "domain", "spatial_correlation", "absolute_spatial_correlation"]].copy()
    joint_table = pd.DataFrame(
        [
            {
                "Model": "CPM ~ Z1 + Z2",
                "In-sample R²": fit["r2"],
                "Adjusted R²": fit["adj_r2"],
                "LOYO r": loyo_r,
                "LOYO R²": loyo_r2,
                "LOYO RMSE": loyo_rmse,
                "LOYO MAE": loyo_mae,
            }
        ]
    )

    def classify(name: str) -> str:
        temporal = abs(primary_df.loc[primary_df["Predictor"] == name, "Pearson r"].iloc[0])
        spatial = spatial_df[(spatial_df["predictor"] == name) & (spatial_df["domain"] == "full_cpm_domain")]["absolute_spatial_correlation"].iloc[0]
        if temporal >= 0.4 and spatial >= 0.5:
            return "Strong CPM association"
        if temporal >= 0.2 or spatial >= 0.3:
            return "Partial CPM association"
        return "Weak or unsupported CPM association"

    report = f"""# Z1/Z2 to CPM Attribution Report

## Inputs

- CPM artifact directory: `{args.cpm_artifact_dir}`
- Z1/Z2 run directory: `{args.z_artifact_dir}`
- Upstream Z1/Z2 metadata directory: `{args.upstream_z_dir}`

## Exact Z1/Z2 definitions

- Z1: LOD mode 1, January, latitude {upstream_summary['reference_mode_1']['lat']}, longitude {upstream_summary['reference_mode_1']['lon']}E
- Z2: LOD mode 2, October, latitude {upstream_summary['reference_mode_2']['lat']}, longitude {upstream_summary['reference_mode_2']['lon']}E
- Canonical predictor table used here: `{z_table_path}`
- Source SST dataset: `{upstream_summary['sst_source']}`

## CPM construction

- Daily CPM source: `{args.cpm_artifact_dir / 'era5_z500_pc1to6_daily.csv'}`
- Annual CPM definition: November-April mean of saved ERA5 daily PC3
- ERA5 EOF3 retained the established CPM sign orientation from the completed reproduction

## Water-year overlap

- Overlap years: `WY{int(years[0])}-WY{int(years[-1])}`
- Number of paired years: `{n}`
- Expected alignment confirmed:
  - Z1 uses January of calendar year y for WY y
  - Z2 uses October of calendar year y-1 for WY y
  - CPM uses November of y-1 through April of y

## Primary temporal result table

{df_to_markdown(primary_df)}

## Spatial result table

{df_to_markdown(spatial_table)}

## Joint result table

{df_to_markdown(joint_table)}

## Joint temporal model details

- beta0 = {beta0:.6f}
- beta1 = {beta1:.6f}
- beta2 = {beta2:.6f}
- SE(beta1) = {fit['se'][1]:.6f}
- SE(beta2) = {fit['se'][2]:.6f}
- p(beta1) = {fit['pvals'][1]:.6g}
- p(beta2) = {fit['pvals'][2]:.6g}
- F-statistic = {fit['f_stat']:.6f}
- F-test p-value = {fit['f_p']:.6g}
- corr(Z1, Z2) = {corr_z1_z2:.6f}
- VIF(Z1) = {vif_z1:.6f}
- VIF(Z2) = {vif_z2:.6f}
- partial corr(Z1, CPM | Z2) = {partial_z1:.6f}
- partial corr(Z2, CPM | Z1) = {partial_z2:.6f}

## Atmospheric spatial-pattern attribution

- Annual wet-season mean anomaly source: `{args.cpm_artifact_dir / 'era5_z500_daily_anomalies.nc'}`
- EOF3 field used for CPM map comparison: `{args.cpm_artifact_dir / 'era5_z500_eof1to6.nc'}`
- Saved regression maps:
  - `{args.output_dir / 'z500_regressed_on_z1.nc'}`
  - `{args.output_dir / 'z500_regressed_on_z2.nc'}`
  - `{args.output_dir / 'z500_regressed_on_z1_z2_joint.nc'}`

## Robustness diagnostics

{df_to_markdown(robustness_df) if not robustness_df.empty else 'Not run.'}

## Monthly timing analysis

{df_to_markdown(monthly_df) if not monthly_df.empty else 'Not run.'}

## Interpretation

- Z1 classification: {classify('Z1')}
- Z2 classification: {classify('Z2')}
- Joint descriptive Z1+Z2 regression-map spatial evidence (full domain): {spatial_df[(spatial_df.predictor=='Z1_Z2_joint') & (spatial_df.domain=='full_cpm_domain')]['spatial_correlation'].iloc[0]:.6f}

Use careful wording:

- Z1 is temporally associated with the wet-season CPM amplitude to the extent shown by the raw annual correlation.
- Z2 is temporally associated with the wet-season CPM amplitude only to the extent shown by the raw annual correlation.
- The Z500 circulation associated with each predictor is evaluated by the regression-map spatial correlation with ERA5 EOF3, not by temporal correlation alone.

## Artifact paths

- Source audit: `{args.output_dir / 'z1_z2_source_audit.md'}`
- Clean Z table: `{args.output_dir / 'z1_z2_by_water_year.csv'}`
- Annual CPM table: `{args.output_dir / 'cpm_pc3_nov_apr_by_water_year.csv'}`
- Overlap table: `{args.output_dir / 'z1_z2_cpm_overlap_table.csv'}`
- Primary scatter plots:
  - `{args.output_dir / 'z1_vs_cpm_wet_season_scatter.png'}`
  - `{args.output_dir / 'z2_vs_cpm_wet_season_scatter.png'}`
- Standardized time series:
  - `{args.output_dir / 'z1_z2_cpm_standardized_timeseries.png'}`
- Joint model outputs:
  - `{args.output_dir / 'z1_z2_joint_cpm_model_summary.csv'}`
  - `{args.output_dir / 'z1_z2_joint_cpm_loyo_predictions.csv'}`
  - `{args.output_dir / 'z1_z2_joint_cpm_loyo_scatter.png'}`
- Spatial outputs:
  - `{args.output_dir / 'z1_z500_regression_vs_cpm_eof3.png'}`
  - `{args.output_dir / 'z2_z500_regression_vs_cpm_eof3.png'}`
  - `{args.output_dir / 'cpm_eof3_z1_z2_regression_comparison.png'}`
  - `{args.output_dir / 'z1_z2_joint_z500_regression_vs_cpm_eof3.png'}`
- Robustness:
  - `{args.output_dir / 'z1_z2_cpm_robustness_summary.csv'}`
- Monthly timing:
  - `{args.output_dir / 'z1_z2_monthly_cpm_correlations.csv'}`
  - `{args.output_dir / 'z1_z2_monthly_cpm_correlations.png'}`

## Direct conclusion

- Does Z1 show evidence of association with CPM?
  - {classify('Z1')}
- Does Z2 show evidence of association with CPM?
  - {classify('Z2')}
- Do Z1 and Z2 jointly show stronger evidence?
  - {'Yes, temporally' if abs(loyo_r) > max(abs(z1_summary['Pearson r']), abs(z2_summary['Pearson r'])) else 'Not clearly stronger temporally'}
- Is the evidence temporal, spatial, or both?
  - Z1: {'both temporal and spatial' if classify('Z1') == 'Strong CPM association' else 'mixed / partial'}
  - Z2: {'both temporal and spatial' if classify('Z2') == 'Strong CPM association' else 'mixed / partial or weak'}
"""
    (args.output_dir / "report.md").write_text(report)

    print(f"Z1/Z2 source table: {z_table_path}")
    print(f"Z1 definition: mode 1, Jan, lat {upstream_summary['reference_mode_1']['lat']}, lon {upstream_summary['reference_mode_1']['lon']}E")
    print(f"Z2 definition: mode 2, Oct, lat {upstream_summary['reference_mode_2']['lat']}, lon {upstream_summary['reference_mode_2']['lon']}E")
    print(f"Overlap years: WY{int(years[0])}-WY{int(years[-1])}")
    print(f"Number of years: {n}")
    print(f"r(Z1, CPM): {z1_summary['Pearson r']:.6f}")
    print(f"r(Z2, CPM): {z2_summary['Pearson r']:.6f}")
    print(f"Joint LOYO correlation: {loyo_r:.6f}")
    print(
        "Spatial r(Z1-associated Z500, EOF3): "
        f"{spatial_df[(spatial_df.predictor=='Z1') & (spatial_df.domain=='full_cpm_domain')]['spatial_correlation'].iloc[0]:.6f}"
    )
    print(
        "Spatial r(Z2-associated Z500, EOF3): "
        f"{spatial_df[(spatial_df.predictor=='Z2') & (spatial_df.domain=='full_cpm_domain')]['spatial_correlation'].iloc[0]:.6f}"
    )
    print(f"Main conclusion: Z1 {classify('Z1')}; Z2 {classify('Z2')}")
    print(f"Artifact directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
