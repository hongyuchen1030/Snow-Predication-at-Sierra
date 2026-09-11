"""
Stage 1 held-out evaluation. Reads the ALREADY-COMPUTED continuous calibrated
trajectory (predicted_regional_swe_calibrated_wy1985_2021.csv, produced by
stage1_calibration_diagnostics.py using frozen theta) and the observed regional
SWE cache -- no recalibration, no re-running Snow17, no held-out SWE used for
any parameter or model choice.

Produces, for BOTH windows (WY2005-2020 dedicated, and WY2005-2021 official):
  - per-year, per-region April-1 obs/pred/abs-error/signed-error/pct-error
  - per-year, per-region seasonal peak magnitude/date/peak-date-error (years
    selected for later plotting are chosen from OBSERVED peak values only)
  - daily-trajectory NSE/RMSE/MAE/Pearson-r/mean-bias per region + whole-Sierra
  - whole-Sierra April-1 summary metrics (Pearson r, R^2, RMSE, MAE, mean bias,
    sign/anomaly agreement)
  - a secondary standardized-anomaly diagnostic (z-score using WY1985-2004
    calibration-period mean/std per region -- a leakage-safe climatology since
    it uses only the calibration period, never WY2005+ years)
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from stage1_pipeline_common import (
    REGIONS, CAL_START, CAL_END, HOLDOUT_2020_START, HOLDOUT_2020_END, HOLDOUT_2021_END,
    water_year, april1_values, seasonal_peak, full_metric_set, whole_sierra_weight,
    combine_whole_sierra, load_region_data, OUT_DIR,
)

ALL_SERIES_REGIONS = REGIONS + ["Whole_Sierra"]


def build_yearly_apr1_table(pred_df, obs_df, years):
    rows = []
    for region in ALL_SERIES_REGIONS:
        pred_apr1 = april1_values(pred_df[region])
        obs_apr1 = april1_values(obs_df[region])
        for wy in years:
            o = obs_apr1.get(wy, np.nan)
            p = pred_apr1.get(wy, np.nan)
            abs_err = p - o if np.isfinite(o) and np.isfinite(p) else np.nan
            pct_err = (abs_err / o * 100.0) if np.isfinite(o) and o != 0 and np.isfinite(abs_err) else np.nan
            pk_o, pk_o_date = seasonal_peak(obs_df[region], wy)
            pk_p, pk_p_date = seasonal_peak(pred_df[region], wy)
            peak_date_err_days = np.nan
            if pk_o_date is not None and pk_p_date is not None:
                peak_date_err_days = (pk_p_date - pk_o_date).days
            rows.append({
                "water_year": wy, "region": region,
                "observed_apr1_swe_mm": o, "predicted_apr1_swe_mm": p,
                "abs_error_mm": abs_err, "signed_error_mm": abs_err, "pct_error": pct_err,
                "observed_peak_swe_mm": pk_o, "observed_peak_date": str(pk_o_date) if pk_o_date is not None else None,
                "predicted_peak_swe_mm": pk_p, "predicted_peak_date": str(pk_p_date) if pk_p_date is not None else None,
                "peak_date_error_days": peak_date_err_days,
            })
    return pd.DataFrame(rows)


def daily_trajectory_metrics(pred_df, obs_df, start, end):
    out = {}
    for region in ALL_SERIES_REGIONS:
        p = pred_df[region].loc[(pred_df.index >= start) & (pred_df.index <= end)]
        o = obs_df[region].reindex(p.index)
        out[region] = full_metric_set(o.values, p.values)
    return out


def whole_sierra_apr1_summary(apr1_df):
    sub = apr1_df[apr1_df["region"] == "Whole_Sierra"].sort_values("water_year")
    obs = sub["observed_apr1_swe_mm"].values
    pred = sub["predicted_apr1_swe_mm"].values
    m = full_metric_set(obs, pred)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() >= 2:
        obs_anom = obs[valid] - np.nanmean(obs[valid])
        pred_anom = pred[valid] - np.nanmean(pred[valid])
        sign_agree = float(np.mean(np.sign(obs_anom) == np.sign(pred_anom)))
    else:
        sign_agree = np.nan
    m["sign_anomaly_agreement_fraction"] = sign_agree
    return m


def standardized_anomaly_diagnostic(pred_df, obs_df, start, end):
    """Leakage-safe: climatology (mean/std) computed ONLY from WY1985-2004
    calibration period, applied to standardize both obs and pred in the
    held-out window. Never uses WY2005+ data to define the normalization."""
    out = {}
    for region in ALL_SERIES_REGIONS:
        obs_cal = obs_df[region].loc[(obs_df.index >= CAL_START) & (obs_df.index <= CAL_END)]
        mu, sigma = float(np.nanmean(obs_cal.values)), float(np.nanstd(obs_cal.values))
        if sigma <= 0 or not np.isfinite(sigma):
            out[region] = {"note": "degenerate calibration-period std, skipped"}
            continue
        p = pred_df[region].loc[(pred_df.index >= start) & (pred_df.index <= end)]
        o = obs_df[region].reindex(p.index)
        o_z = (o.values - mu) / sigma
        p_z = (p.values - mu) / sigma
        out[region] = full_metric_set(o_z, p_z)
        out[region]["climatology_mean_mm"] = mu
        out[region]["climatology_std_mm"] = sigma
        out[region]["climatology_period"] = "WY1985-2004 (calibration period only)"
    return out


def run_window(pred_df, obs_df, start, end, years, label, out_prefix):
    apr1_df = build_yearly_apr1_table(pred_df, obs_df, years)
    daily_metrics = daily_trajectory_metrics(pred_df, obs_df, start, end)
    whole_apr1 = whole_sierra_apr1_summary(apr1_df)
    anomaly = standardized_anomaly_diagnostic(pred_df, obs_df, start, end)

    apr1_csv = OUT_DIR / out_prefix / f"heldout_yearly_apr1_{label}.csv"
    apr1_df.to_csv(apr1_csv, index=False)

    result = {
        "window": label,
        "start": str(start), "end": str(end),
        "water_years": years,
        "daily_trajectory_metrics": daily_metrics,
        "whole_sierra_april1_summary": whole_apr1,
        "standardized_anomaly_diagnostic_leakage_safe": anomaly,
        "yearly_apr1_csv": str(apr1_csv),
    }
    out_json = OUT_DIR / out_prefix / f"heldout_{label}_metrics.json"
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"Saved {out_json}")
    print(f"Saved {apr1_csv}")
    return result


def main():
    t0 = time.time()
    pred_csv = OUT_DIR / "calibration" / "predicted_regional_swe_calibrated_wy1985_2021.csv"
    pred_df = pd.read_csv(pred_csv, index_col=0, parse_dates=True)

    obs_raw = pd.read_csv(OUT_DIR / "forcing_cache" / "observed_regional_swe_wy1985_2021.csv",
                           index_col=0, parse_dates=True)
    # Whole-Sierra observed uses the same occupancy weights as predicted whole-Sierra
    weights = json.loads((OUT_DIR / "calibration" / "whole_sierra_weights.json").read_text())
    obs_whole = combine_whole_sierra({r: obs_raw[r] for r in REGIONS}, weights)
    obs_df = obs_raw.copy()
    obs_df["Whole_Sierra"] = obs_whole

    years_2020 = list(range(2005, 2021))
    years_2021 = list(range(2005, 2022))

    r2020 = run_window(pred_df, obs_df, HOLDOUT_2020_START, HOLDOUT_2020_END,
                        years_2020, "2005_2020", "heldout_2005_2020")
    r2021 = run_window(pred_df, obs_df, HOLDOUT_2020_START, HOLDOUT_2021_END,
                        years_2021, "2005_2021", "heldout_2005_2021")

    # also save the raw daily obs+pred regional series for the held-out window (for plotting)
    daily_out = pred_df.loc[(pred_df.index >= HOLDOUT_2020_START) & (pred_df.index <= HOLDOUT_2021_END)].copy()
    daily_out.columns = [f"pred_{c}" for c in daily_out.columns]
    obs_slice = obs_df.reindex(daily_out.index)
    for c in ALL_SERIES_REGIONS:
        daily_out[f"obs_{c}"] = obs_slice[c]
    daily_out.to_csv(OUT_DIR / "heldout_2005_2021" / "heldout_daily_regional_swe.csv")
    print(f"Saved heldout_daily_regional_swe.csv ({len(daily_out)} rows)")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
