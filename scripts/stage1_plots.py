"""
Stage 1 required plots (WY2005-2020 dedicated verification):
  A. yearly April-1 SWE lines (obs vs Snow-17) per region
  B. predicted-vs-observed April-1 scatter per region, 1:1 line, r/RMSE/bias
  C. full seasonal daily trajectories for representative low/near-normal/high
     held-out years, selected using OBSERVED SWE only
  D. all held-out years' daily obs-vs-predicted trajectories (small multiples)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1_pipeline_common import (
    REGIONS, HOLDOUT_2020_START, HOLDOUT_2020_END, full_metric_set, OUT_DIR,
)

ALL_REGIONS = REGIONS + ["Whole_Sierra"]
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_apr1_2005_2020():
    return pd.read_csv(OUT_DIR / "heldout_2005_2020" / "heldout_yearly_apr1_2005_2020.csv")


def load_daily():
    pred_csv = OUT_DIR / "calibration" / "predicted_regional_swe_calibrated_wy1985_2021.csv"
    pred_df = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
    obs_raw = pd.read_csv(OUT_DIR / "forcing_cache" / "observed_regional_swe_wy1985_2021.csv",
                           index_col=0, parse_dates=True)
    weights = json.loads((OUT_DIR / "calibration" / "whole_sierra_weights.json").read_text())
    from stage1_pipeline_common import combine_whole_sierra
    obs_df = obs_raw.copy()
    obs_df["Whole_Sierra"] = combine_whole_sierra({r: obs_raw[r] for r in REGIONS}, weights)
    return pred_df, obs_df


def plot_A_yearly_lines(apr1_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, region in zip(axes.ravel(), ALL_REGIONS):
        sub = apr1_df[apr1_df["region"] == region].sort_values("water_year")
        ax.plot(sub["water_year"], sub["observed_apr1_swe_mm"], "o-", label="Observed", color="tab:blue")
        ax.plot(sub["water_year"], sub["predicted_apr1_swe_mm"], "s--", label="Snow-17 (frozen theta)", color="tab:orange")
        ax.set_title(region)
        ax.set_ylabel("April-1 SWE (mm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Water Year")
    fig.suptitle("Plot A: April-1 SWE, observed vs Snow-17 (held-out WY2005-2020)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plotA_yearly_april1_lines.png", dpi=150)
    plt.close(fig)


def plot_B_scatter(apr1_df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, region in zip(axes.ravel(), ALL_REGIONS):
        sub = apr1_df[apr1_df["region"] == region]
        o = sub["observed_apr1_swe_mm"].values
        p = sub["predicted_apr1_swe_mm"].values
        valid = np.isfinite(o) & np.isfinite(p)
        m = full_metric_set(o, p)
        ax.scatter(o[valid], p[valid], color="tab:blue")
        lims = [0, max(np.nanmax(o[valid]) if valid.any() else 1, np.nanmax(p[valid]) if valid.any() else 1) * 1.1]
        ax.plot(lims, lims, "k--", lw=1, label="1:1")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Observed April-1 SWE (mm)")
        ax.set_ylabel("Predicted April-1 SWE (mm)")
        ax.set_title(f"{region}\nr={m['pearson_r']:.2f}, RMSE={m['rmse']:.1f}mm, bias={m['mean_bias']:.1f}mm")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Plot B: Predicted vs Observed April-1 SWE (held-out WY2005-2020)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plotB_april1_scatter.png", dpi=150)
    plt.close(fig)


def select_representative_years(apr1_df, region="Whole_Sierra"):
    """Select low/near-normal/high snow years using OBSERVED April-1 SWE only."""
    sub = apr1_df[apr1_df["region"] == region].dropna(subset=["observed_apr1_swe_mm"]).sort_values("observed_apr1_swe_mm")
    if len(sub) < 3:
        return {}
    low_wy = int(sub.iloc[0]["water_year"])
    high_wy = int(sub.iloc[-1]["water_year"])
    median_val = sub["observed_apr1_swe_mm"].median()
    normal_wy = int(sub.iloc[(sub["observed_apr1_swe_mm"] - median_val).abs().argsort().iloc[0]]["water_year"])
    return {"low": low_wy, "near_normal": normal_wy, "high": high_wy}


def plot_C_representative_years(pred_df, obs_df, apr1_df):
    reps = select_representative_years(apr1_df)
    if not reps:
        print("Not enough years for representative selection; skipping Plot C")
        return
    fig, axes = plt.subplots(len(reps), 1, figsize=(11, 3.2 * len(reps)), sharex=False)
    if len(reps) == 1:
        axes = [axes]
    for ax, (label, wy) in zip(axes, reps.items()):
        start = pd.Timestamp(year=wy - 1, month=10, day=1)
        end = pd.Timestamp(year=wy, month=9, day=30)
        for region, color in zip(ALL_REGIONS, ["tab:green", "tab:purple", "tab:brown", "black"]):
            o = obs_df[region].loc[(obs_df.index >= start) & (obs_df.index <= end)]
            p = pred_df[region].loc[(pred_df.index >= start) & (pred_df.index <= end)]
            lw = 2.2 if region == "Whole_Sierra" else 1.2
            ax.plot(o.index, o.values, color=color, lw=lw, label=f"{region} obs")
            ax.plot(p.index, p.values, color=color, lw=lw, ls="--", label=f"{region} pred")
        ax.set_title(f"WY{wy} ({label} observed whole-Sierra April-1 snow)")
        ax.set_ylabel("SWE (mm)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=4, loc="upper right")
    fig.suptitle("Plot C: Representative held-out years (selected from OBSERVED SWE only)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plotC_representative_years.png", dpi=150)
    plt.close(fig)
    (PLOT_DIR / "plotC_selected_years.json").write_text(json.dumps(reps, indent=2) + "\n")


def plot_D_all_years(pred_df, obs_df):
    years = list(range(2005, 2021))
    ncols = 4
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharey=False)
    axes = axes.ravel()
    for ax, wy in zip(axes, years):
        start = pd.Timestamp(year=wy - 1, month=10, day=1)
        end = pd.Timestamp(year=wy, month=9, day=30)
        o = obs_df["Whole_Sierra"].loc[(obs_df.index >= start) & (obs_df.index <= end)]
        p = pred_df["Whole_Sierra"].loc[(pred_df.index >= start) & (pred_df.index <= end)]
        ax.plot(o.index, o.values, color="tab:blue", lw=1.2, label="obs")
        ax.plot(p.index, p.values, color="tab:orange", lw=1.2, ls="--", label="pred")
        ax.set_title(f"WY{wy}", fontsize=9)
        ax.tick_params(labelsize=6, rotation=30)
        ax.grid(alpha=0.3)
    for ax in axes[len(years):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle("Plot D: All held-out years, Whole-Sierra daily obs vs predicted (WY2005-2020)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plotD_all_years_whole_sierra.png", dpi=150)
    plt.close(fig)

    # Also per-region multi-page style: one fig per region, all years overlaid as small multiples
    for region in REGIONS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharey=False)
        axes = axes.ravel()
        for ax, wy in zip(axes, years):
            start = pd.Timestamp(year=wy - 1, month=10, day=1)
            end = pd.Timestamp(year=wy, month=9, day=30)
            o = obs_df[region].loc[(obs_df.index >= start) & (obs_df.index <= end)]
            p = pred_df[region].loc[(pred_df.index >= start) & (pred_df.index <= end)]
            ax.plot(o.index, o.values, color="tab:blue", lw=1.2, label="obs")
            ax.plot(p.index, p.values, color="tab:orange", lw=1.2, ls="--", label="pred")
            ax.set_title(f"WY{wy}", fontsize=9)
            ax.tick_params(labelsize=6, rotation=30)
            ax.grid(alpha=0.3)
        for ax in axes[len(years):]:
            ax.axis("off")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Plot D: All held-out years, {region} daily obs vs predicted (WY2005-2020)")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"plotD_all_years_{region.lower()}.png", dpi=150)
        plt.close(fig)


def main():
    apr1_df = load_apr1_2005_2020()
    pred_df, obs_df = load_daily()
    plot_A_yearly_lines(apr1_df)
    plot_B_scatter(apr1_df)
    plot_C_representative_years(pred_df, obs_df, apr1_df)
    plot_D_all_years(pred_df, obs_df)
    print(f"All plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
