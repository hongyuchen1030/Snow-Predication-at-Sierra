"""
Stage 1 calibration-period (WY1985-2004) diagnostics: NSE/RMSE/MAE/Pearson-r/bias
for North/Central/South/Whole_Sierra, using the FROZEN calibrated theta, compared
against the fixed midpoint baseline. WY1985-2004 only -- no held-out (WY2005+) SWE
is read or used anywhere in this script.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from stage1_pipeline_common import (
    REGIONS, CAL_START, CAL_END, run_region_continuous, whole_sierra_weight,
    combine_whole_sierra, full_metric_set, load_theta, OUT_DIR,
)


def main():
    t0 = time.time()
    regional_series_cal = {}
    data_by_region = {}
    obs_by_region = {}
    thetas = {}

    for region in REGIONS:
        theta = load_theta(region)
        thetas[region] = theta
        params = theta["calibrated_parameters"]
        print(f"[calibrated] running {region} with theta...", flush=True)
        tidx, swe, data = run_region_continuous(region, params)
        series = pd.Series(swe, index=tidx)
        regional_series_cal[region] = series
        data_by_region[region] = data
        obs_by_region[region] = data["obs"][region]
        print(f"[calibrated] {region} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    weights = whole_sierra_weight(data_by_region)
    whole_cal = combine_whole_sierra(regional_series_cal, weights)

    # Slice to calibration window WY1985-2004
    def cal_slice(s):
        return s.loc[(s.index >= CAL_START) & (s.index <= CAL_END)]

    obs_whole = combine_whole_sierra(obs_by_region, weights)

    metrics_calibrated = {}
    for region in REGIONS:
        sim = cal_slice(regional_series_cal[region])
        obs = obs_by_region[region].reindex(sim.index)
        metrics_calibrated[region] = full_metric_set(obs.values, sim.values)
    sim_w = cal_slice(whole_cal)
    obs_w = obs_whole.reindex(sim_w.index)
    metrics_calibrated["Whole_Sierra"] = full_metric_set(obs_w.values, sim_w.values)

    # Midpoint baseline (precomputed continuous run)
    mp_csv = OUT_DIR / "calibration" / "predicted_regional_swe_midpoint_wy1985_2021.csv"
    mp_df = pd.read_csv(mp_csv, index_col=0, parse_dates=True)
    metrics_midpoint = {}
    for region in REGIONS + ["Whole_Sierra"]:
        sim = cal_slice(mp_df[region])
        if region == "Whole_Sierra":
            obs = obs_whole.reindex(sim.index)
        else:
            obs = obs_by_region[region].reindex(sim.index)
        metrics_midpoint[region] = full_metric_set(obs.values, sim.values)

    out = {
        "calibration_period": "WY1985-2004",
        "objective_used_for_calibration": "1 - NSE on full daily regional SWE trajectory",
        "metrics_calibrated_theta": metrics_calibrated,
        "metrics_midpoint_baseline": metrics_midpoint,
        "improvement_nse_calibrated_minus_midpoint": {
            r: metrics_calibrated[r]["nse"] - metrics_midpoint[r]["nse"]
            for r in REGIONS + ["Whole_Sierra"]
        },
        "frozen_theta_files": {r: str(OUT_DIR / "frozen_parameters" / f"theta_{r[0]}.json") for r in REGIONS},
        "whole_sierra_weights": weights,
    }
    out_json = OUT_DIR / "calibration" / "calibration_metrics.json"
    out_json.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps(out, indent=2, default=float))
    print(f"Saved {out_json}")

    # Also save the calibrated continuous full-period trajectory for reuse by held-out scripts
    full_df = pd.DataFrame(regional_series_cal)
    full_df["Whole_Sierra"] = whole_cal
    full_out = OUT_DIR / "calibration" / "predicted_regional_swe_calibrated_wy1985_2021.csv"
    full_df.to_csv(full_out)
    print(f"Saved continuous calibrated trajectory: {full_out} ({len(full_df)} rows)")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
