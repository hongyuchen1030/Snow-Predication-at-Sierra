#!/usr/bin/env python3
"""Build the sample index for the 7-day -> 1-day Sierra SWE-change task.

Does NOT materialize any 7-day window tensors. For every candidate target day
t it records only pointers (experiment, calendar year, day-of-year) into the
existing yearly predictor arrays for t-6..t, plus SWE(t) and SWE(t+1) (cheap
scalars), plus metadata (water_year, date, active flag). Training scripts use
this index plus mmap'd yearly arrays to build batches lazily.

Calendar handling: both the predictor archive (TaiESM1 CMIP6 `day` table,
calendar=noleap) and the WUS-D3 daily SWE target (calendar=365_day, a CF
synonym of noleap) use the SAME no-leap calendar with no exceptions. This
script converts every day in both datasets to an absolute integer serial
number (`serial = year*365 + day_of_year_from_jan1 - 1`), which makes
"consecutive day" and "same day across datasets" both trivial equality/+1
checks -- no pandas/Gregorian date arithmetic is used anywhere.

Most-permissive index: built once assuming a Sep-1 season start (the most
inclusive choice). Every row carries the target date's (month, day) and
water_year, so a training script can filter to Oct/Nov/Dec-start seasons
cheaply at load time without rebuilding this index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
PREDICTOR_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1")
SWE_NPZ = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.npz"

EXPERIMENT_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1"
OUT_DIR = Path(EXPERIMENT_ROOT) / "data_index"

CUM_DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]  # Jan..Dec, noleap
N_HISTORY_DAYS = 7


def serial(year: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """day_of_year is 1-365, Jan1-based, noleap calendar."""
    return year.astype(np.int64) * 365 + (day_of_year.astype(np.int64) - 1)


def month_day_from_doy(day_of_year: int) -> tuple[int, int]:
    for month in range(11, -1, -1):
        if day_of_year > CUM_DAYS_BEFORE_MONTH[month]:
            return month + 1, day_of_year - CUM_DAYS_BEFORE_MONTH[month]
    raise ValueError(day_of_year)


def water_year_of(cal_year: int, month: int) -> int:
    return cal_year + 1 if month >= 9 else cal_year


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Predictor metadata.
    # -----------------------------------------------------------------
    pmeta = np.load(PREDICTOR_ROOT / "metadata.npz", allow_pickle=True)
    p_year = pmeta["year"].astype(np.int64)
    p_doy = pmeta["day_of_year"].astype(np.int64)
    p_experiment = pmeta["experiment"]
    n_pred = len(p_year)
    print(f"predictor metadata: {n_pred} records, experiments={sorted(set(p_experiment.tolist()))}", flush=True)

    p_serial = serial(p_year, p_doy)
    assert len(set(p_serial.tolist())) == n_pred, "duplicate predictor days"

    # index: serial -> (experiment, year, doy) for locating the row in the yearly npy
    serial_to_pred = {int(s): (str(e), int(y), int(d)) for s, e, y, d in zip(p_serial, p_experiment, p_year, p_doy)}

    # verify within-file day_of_year is exactly 1..365 in order (no internal gaps),
    # and that consecutive calendar years chain with serial+1 (no cross-file gap).
    order = np.argsort(p_serial)
    sorted_serial = p_serial[order]
    gaps = np.where(np.diff(sorted_serial) != 1)[0]
    print(f"predictor serial gaps (should be 0): {len(gaps)}", flush=True)
    if len(gaps):
        for g in gaps[:5]:
            print(f"  gap at serial {sorted_serial[g]} -> {sorted_serial[g+1]}", flush=True)

    # -----------------------------------------------------------------
    # 2. SWE target (full year-round series, already validated).
    # -----------------------------------------------------------------
    smeta = np.load(SWE_NPZ, allow_pickle=True)
    s_year = smeta["year"].astype(np.int64)
    s_month = smeta["month"].astype(np.int64)
    s_day = smeta["day"].astype(np.int64)
    s_swe = smeta["swe_mm"].astype(np.float64)
    n_swe = len(s_year)
    s_doy = np.array([CUM_DAYS_BEFORE_MONTH[m - 1] + d for m, d in zip(s_month, s_day)], dtype=np.int64)
    s_serial = serial(s_year, s_doy)
    assert len(set(s_serial.tolist())) == n_swe, "duplicate SWE days"
    serial_to_swe = {int(s): float(v) for s, v in zip(s_serial, s_swe)}
    print(f"SWE target: {n_swe} records", flush=True)

    # -----------------------------------------------------------------
    # 3. Build candidate target-day samples.
    #    For every predictor day t with a full 7-day back-history (t-6..t)
    #    present with no gap, and SWE(t), SWE(t+1) both present.
    # -----------------------------------------------------------------
    rows = []
    n_no_history = 0
    n_history_gap = 0
    n_swe_missing = 0
    for s_t in sorted_serial.tolist():
        history_serials = [s_t - k for k in range(N_HISTORY_DAYS - 1, -1, -1)]  # t-6..t
        if not all(hs in serial_to_pred for hs in history_serials):
            n_no_history += 1
            continue
        # contiguity is already guaranteed globally (checked above), but verify locally too
        if any(history_serials[i + 1] - history_serials[i] != 1 for i in range(len(history_serials) - 1)):
            n_history_gap += 1
            continue
        swe_t = serial_to_swe.get(s_t)
        swe_t1 = serial_to_swe.get(s_t + 1)
        if swe_t is None or swe_t1 is None or not (np.isfinite(swe_t) and np.isfinite(swe_t1)):
            n_swe_missing += 1
            continue

        exp_t, year_t, doy_t = serial_to_pred[s_t]
        month_t, day_t = month_day_from_doy(doy_t)
        wy = water_year_of(year_t, month_t)
        active = bool(swe_t > 0.0 or swe_t1 > 0.0)

        history_pointers = [serial_to_pred[hs] for hs in history_serials]  # list of (experiment, year, doy)

        rows.append(
            {
                "serial_t": s_t,
                "water_year": wy,
                "cal_year": year_t,
                "cal_month": month_t,
                "cal_day": day_t,
                "experiment_t": exp_t,
                "swe_t": swe_t,
                "swe_t1": swe_t1,
                "delta_swe": swe_t1 - swe_t,
                "active": active,
                "history": history_pointers,  # 7 entries, oldest first (t-6..t)
            }
        )

    print(
        f"candidate target days: {len(sorted_serial)} | "
        f"kept={len(rows)} | dropped_no_history={n_no_history} | "
        f"dropped_history_gap={n_history_gap} | dropped_swe_missing={n_swe_missing}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 4. Save index. Two representations: a flat NPZ for fast array access
    #    at training time, and a JSON for readability/debugging.
    # -----------------------------------------------------------------
    n = len(rows)
    water_year = np.array([r["water_year"] for r in rows], dtype=np.int32)
    cal_year = np.array([r["cal_year"] for r in rows], dtype=np.int32)
    cal_month = np.array([r["cal_month"] for r in rows], dtype=np.int8)
    cal_day = np.array([r["cal_day"] for r in rows], dtype=np.int8)
    experiment_t = np.array([r["experiment_t"] for r in rows])
    swe_t = np.array([r["swe_t"] for r in rows], dtype=np.float64)
    swe_t1 = np.array([r["swe_t1"] for r in rows], dtype=np.float64)
    delta_swe = np.array([r["delta_swe"] for r in rows], dtype=np.float64)
    active = np.array([r["active"] for r in rows], dtype=bool)
    serial_t = np.array([r["serial_t"] for r in rows], dtype=np.int64)

    # history pointers: (n, 7, 3) as [experiment_code(0/1), year, doy] -- store
    # experiment as 0=historical/1=ssp370 to keep this a clean numeric array.
    exp_code_map = {"historical": 0, "ssp370": 1}
    history_year = np.zeros((n, N_HISTORY_DAYS), dtype=np.int32)
    history_doy = np.zeros((n, N_HISTORY_DAYS), dtype=np.int16)
    history_exp = np.zeros((n, N_HISTORY_DAYS), dtype=np.int8)
    for i, r in enumerate(rows):
        for k, (exp, year, doy) in enumerate(r["history"]):
            history_year[i, k] = year
            history_doy[i, k] = doy
            history_exp[i, k] = exp_code_map[exp]

    np.savez_compressed(
        OUT_DIR / "sample_index.npz",
        serial_t=serial_t,
        water_year=water_year,
        cal_year=cal_year,
        cal_month=cal_month,
        cal_day=cal_day,
        experiment_t=experiment_t,
        swe_t=swe_t,
        swe_t1=swe_t1,
        delta_swe=delta_swe,
        active=active,
        history_year=history_year,
        history_doy=history_doy,
        history_exp=history_exp,
        exp_code_map=json.dumps(exp_code_map),
    )
    print(f"wrote {OUT_DIR / 'sample_index.npz'}", flush=True)

    # -----------------------------------------------------------------
    # 5. Report counts per season-start choice and sample mode.
    # -----------------------------------------------------------------
    def in_season(month: np.ndarray, day: np.ndarray, start_month: int) -> np.ndarray:
        fall = (month >= start_month) & (month <= 12)
        winter = (month >= 1) & (month <= 3)
        apr1 = (month == 4) & (day == 1)
        return fall | winter | apr1

    summary = {"n_water_years": int(len(set(water_year.tolist()))), "total_candidate_samples": n, "by_season_start": {}}
    for start_month, label in ((9, "Sep1"), (10, "Oct1"), (11, "Nov1"), (12, "Dec1")):
        mask = in_season(cal_month, cal_day, start_month)
        n_all = int(mask.sum())
        n_active = int((mask & active).sum())
        summary["by_season_start"][label] = {
            "start_month": start_month,
            "N_all": n_all,
            "N_active": n_active,
            "N_zero_to_zero": n_all - n_active,
            "active_fraction": n_active / n_all if n_all else None,
        }
        print(f"season {label:6s}: N_all={n_all:6d} N_active={n_active:6d} active_frac={n_active/n_all:.4f}", flush=True)

    (OUT_DIR / "sample_index_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'sample_index_summary.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
