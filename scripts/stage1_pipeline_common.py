"""
Shared functions for Stage 1 post-calibration analysis:
- continuous full-period (WY1985 spin-up through WY2021) Snow-17 simulation per region
- occupancy-weighted whole-Sierra combination of the 3 regional trajectories
- standard hydrologic metrics (NSE, RMSE, MAE, Pearson r, bias)
- water-year helpers (April-1 extraction, seasonal peak extraction)

Reuses load_region_data / run_region_snow17 / constants from stage1_calibrate.py
verbatim -- does not reimplement forcing extraction, elevation correction, or
occupancy weighting logic.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd

from stage1_calibrate import (
    load_region_data, run_region_snow17, REGION_CODE, FIXED_PARAMS, BOUNDS,
    SPINUP_START, CAL_START, CAL_END, OUT_DIR, CACHE_DIR,
)

REGIONS = ["North", "Central", "South"]

# Full continuous simulation window: spin-up start through end of WY2021.
HOLDOUT_2020_START = pd.Timestamp("2004-10-01")  # WY2005 start
HOLDOUT_2020_END = pd.Timestamp("2020-09-30")    # WY2020 end
HOLDOUT_2021_END = pd.Timestamp("2021-09-30")    # WY2021 end

MIDPOINT_PARAMS = {k: (lo + hi) / 2 for k, (lo, hi) in BOUNDS.items()}


def water_year(ts):
    """WY = calendar year of the following Oct1..Sep30 span's end (WY starts Oct 1 of Y-1)."""
    ts = pd.Timestamp(ts)
    return ts.year + 1 if ts.month >= 10 else ts.year


def run_region_continuous(region_name, params, data=None):
    """One continuous Snow17 run per cell from SPINUP_START through HOLDOUT_2021_END,
    zero initial state at SPINUP_START, no re-zeroing at water-year boundaries.
    Returns (time_index, regional_swe_array)."""
    if data is None:
        data = load_region_data(region_name)
    sl = (data["common_index"] >= SPINUP_START) & (data["common_index"] <= HOLDOUT_2021_END)
    regional = run_region_snow17(data, params, sl)
    time_index = data["common_index"][sl]
    return time_index, regional, data


def whole_sierra_weight(data_by_region):
    """Total occupancy weight per region = sum of w_cells (native-pixel occupancy
    count matching that region) across all of that region's ERA5 cells. Used to
    combine regional SWE trajectories into a single whole-Sierra trajectory:
    SWE_sierra(t) = sum_r(W_r * SWE_r(t)) / sum_r(W_r)."""
    return {r: float(data_by_region[r]["w_cells"].sum()) for r in REGIONS}


def combine_whole_sierra(regional_series, weights):
    """regional_series: dict region -> pd.Series (aligned index). weights: dict region -> float."""
    df = pd.DataFrame(regional_series)
    w = np.array([weights[r] for r in df.columns])
    return pd.Series((df.values * w[None, :]).sum(axis=1) / w.sum(), index=df.index)


def nse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64); sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return np.nan
    o, s = obs[valid], sim[valid]
    denom = np.sum((o - o.mean()) ** 2)
    return np.nan if denom <= 0 else 1.0 - np.sum((o - s) ** 2) / denom


def rmse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64); sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((obs[valid] - sim[valid]) ** 2)))


def mae(obs, sim):
    obs = np.asarray(obs, dtype=np.float64); sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return np.nan
    return float(np.mean(np.abs(obs[valid] - sim[valid])))


def pearson_r(obs, sim):
    obs = np.asarray(obs, dtype=np.float64); sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2 or np.std(obs[valid]) == 0 or np.std(sim[valid]) == 0:
        return np.nan
    return float(np.corrcoef(obs[valid], sim[valid])[0, 1])


def mean_bias(obs, sim):
    obs = np.asarray(obs, dtype=np.float64); sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return np.nan
    return float(np.mean(sim[valid] - obs[valid]))


def full_metric_set(obs, sim):
    return {
        "nse": nse(obs, sim), "rmse": rmse(obs, sim), "mae": mae(obs, sim),
        "pearson_r": pearson_r(obs, sim), "mean_bias": mean_bias(obs, sim),
        "r_squared": (lambda r: r ** 2 if np.isfinite(r) else np.nan)(pearson_r(obs, sim)),
        "n_valid": int((np.isfinite(np.asarray(obs, dtype=np.float64)) &
                         np.isfinite(np.asarray(sim, dtype=np.float64))).sum()),
    }


def april1_values(series):
    """series: pd.Series with DatetimeIndex. Returns dict wy -> value at that WY's Apr-1
    (April falls in the WY's own calendar year, e.g. WY2005 Apr1 = 2005-04-01)."""
    out = {}
    for wy in sorted(set(water_year(t) for t in series.index)):
        target = pd.Timestamp(year=wy, month=4, day=1)
        if target in series.index:
            out[wy] = float(series.loc[target])
    return out


def seasonal_peak(series, wy):
    """Peak value and date within WY (Oct1 of wy-1 through Sep30 of wy)."""
    start = pd.Timestamp(year=wy - 1, month=10, day=1)
    end = pd.Timestamp(year=wy, month=9, day=30)
    sub = series.loc[(series.index >= start) & (series.index <= end)]
    sub = sub.dropna()
    if len(sub) == 0:
        return np.nan, None
    idx = sub.idxmax()
    return float(sub.loc[idx]), idx


def load_theta(region_name):
    import json
    p = OUT_DIR / "frozen_parameters" / f"theta_{region_name[0]}.json"
    return json.loads(p.read_text())
