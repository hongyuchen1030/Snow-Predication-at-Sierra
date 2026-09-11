#!/usr/bin/env python3
"""Pre-training audit for daily_temporal_baselines_v2 (B0 LSTM / B1 TCNN).

Verifies, on N randomly selected Oct1-Apr1 / train-water-year samples, that:
  1. the 7 history-day pointers are exactly 7 CONSECUTIVE calendar days
     (serial = year*365 + doy - 1, strictly +1 each step) ending on the
     target day t;
  2. delta_swe stored in the index equals swe_t1 - swe_t;
  3. swe_t and swe_t1 independently match the raw
     artifacts/taiesm1_daily_sierra_swe.csv record for that exact date
     (catches drift between the index and the source SWE artifact, not just
     internal self-consistency);
  4. the actual predictor arrays for all 7 history days load, are finite,
     and have the expected [12,120,240] shape (the same mmap access path
     DeltaSWEDataset uses at __getitem__ time).

Must run on a compute node (loads real predictor + SWE arrays). Writes
daily_temporal_baselines_v2/shared/sample_audit.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
BASELINE_INDEX = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1/data_index/sample_index.npz")
PREDICTOR_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1")
SWE_CSV = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.csv"
SPLIT_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_temporal_baselines_v2/shared/split.json")
OUT_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_temporal_baselines_v2/shared/sample_audit.json")

N_AUDIT = 25
SEED = 20260901
N_HISTORY_DAYS = 7
CUM_DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def in_season_mask(month: np.ndarray, day: np.ndarray, start_month: int) -> np.ndarray:
    fall = (month >= start_month) & (month <= 12)
    winter = (month >= 1) & (month <= 3)
    apr1 = (month == 4) & (day == 1)
    return fall | winter | apr1


def serial(year: int, doy: int) -> int:
    return year * 365 + (doy - 1)


def load_swe_csv() -> dict:
    lookup = {}
    with SWE_CSV.open() as fh:
        for row in csv.DictReader(fh):
            key = (int(row["cal_year"]), int(row["cal_month"]), int(row["cal_day"]))
            lookup[key] = float(row["swe_mm"])
    return lookup


def month_day_from_doy(doy: int) -> tuple[int, int]:
    for month in range(11, -1, -1):
        if doy > CUM_DAYS_BEFORE_MONTH[month]:
            return month + 1, doy - CUM_DAYS_BEFORE_MONTH[month]
    raise ValueError(doy)


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    train_wy = set(split["train_water_years"])

    idx = np.load(BASELINE_INDEX, allow_pickle=True)
    water_year = idx["water_year"]
    cal_month = idx["cal_month"]
    cal_day = idx["cal_day"]
    cal_year = idx["cal_year"]
    serial_t = idx["serial_t"]
    swe_t = idx["swe_t"]
    swe_t1 = idx["swe_t1"]
    delta_swe = idx["delta_swe"]
    history_year = idx["history_year"]
    history_doy = idx["history_doy"]
    history_exp = idx["history_exp"]
    exp_code_map = json.loads(str(idx["exp_code_map"]))
    code_to_exp = {v: k for k, v in exp_code_map.items()}

    mask = np.isin(water_year, list(train_wy)) & in_season_mask(cal_month, cal_day, 10)
    candidate_rows = np.where(mask)[0]
    print(f"candidate Oct1-Apr1 / train-WY rows: {len(candidate_rows)}", flush=True)

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(candidate_rows, size=min(N_AUDIT, len(candidate_rows)), replace=False)

    swe_lookup = load_swe_csv()

    results = []
    n_pass = 0
    for i in chosen.tolist():
        problems = []

        # 1. history contiguity: 7 consecutive days ending at target day t.
        h_serials = [serial(int(history_year[i, k]), int(history_doy[i, k])) for k in range(N_HISTORY_DAYS)]
        steps_ok = all(h_serials[k + 1] - h_serials[k] == 1 for k in range(N_HISTORY_DAYS - 1))
        ends_at_t = h_serials[-1] == int(serial_t[i])
        if not steps_ok:
            problems.append(f"history days not consecutive: serials={h_serials}")
        if not ends_at_t:
            problems.append(f"history does not end at target day t: last={h_serials[-1]} serial_t={int(serial_t[i])}")

        # 2. delta_swe consistency.
        recomputed_delta = float(swe_t1[i] - swe_t[i])
        delta_ok = abs(recomputed_delta - float(delta_swe[i])) < 1e-9
        if not delta_ok:
            problems.append(f"delta_swe mismatch: stored={float(delta_swe[i])} recomputed={recomputed_delta}")

        # 3. cross-check swe_t / swe_t1 against the raw SWE csv artifact.
        key_t = (int(cal_year[i]), int(cal_month[i]), int(cal_day[i]))
        csv_swe_t = swe_lookup.get(key_t)
        # date of t+1: derive from serial_t+1
        s_t1 = int(serial_t[i]) + 1
        y_t1, doy_t1 = s_t1 // 365, (s_t1 % 365) + 1
        m_t1, d_t1 = month_day_from_doy(doy_t1)
        key_t1 = (y_t1, m_t1, d_t1)
        csv_swe_t1 = swe_lookup.get(key_t1)
        csv_ok = (
            csv_swe_t is not None and csv_swe_t1 is not None
            and abs(csv_swe_t - float(swe_t[i])) < 1e-9
            and abs(csv_swe_t1 - float(swe_t1[i])) < 1e-9
        )
        if not csv_ok:
            problems.append(f"csv cross-check failed: index(swe_t={float(swe_t[i])}, swe_t1={float(swe_t1[i])}) csv(swe_t={csv_swe_t}, swe_t1={csv_swe_t1}) key_t={key_t} key_t1={key_t1}")

        # 4. load actual predictor arrays for all 7 history days, check shape/finiteness.
        shapes_ok, finite_ok = True, True
        for k in range(N_HISTORY_DAYS):
            exp_name = code_to_exp[int(history_exp[i, k])]
            year = int(history_year[i, k])
            doy = int(history_doy[i, k])
            arr = np.load(PREDICTOR_ROOT / "years" / exp_name / f"predictors_physical_{year}.npy", mmap_mode="r")
            day_slice = np.asarray(arr[doy - 1])
            if day_slice.shape != (12, 120, 240):
                shapes_ok = False
                problems.append(f"bad predictor shape at k={k}: {day_slice.shape}")
            if not np.isfinite(day_slice).all():
                finite_ok = False
                problems.append(f"non-finite predictor values at k={k}")

        row_pass = steps_ok and ends_at_t and delta_ok and csv_ok and shapes_ok and finite_ok
        n_pass += int(row_pass)
        results.append(
            {
                "row": int(i),
                "water_year": int(water_year[i]),
                "target_date": f"{int(cal_year[i])}-{int(cal_month[i]):02d}-{int(cal_day[i]):02d}",
                "history_serials": h_serials,
                "swe_t_mm": float(swe_t[i]),
                "swe_t1_mm": float(swe_t1[i]),
                "delta_swe_mm": float(delta_swe[i]),
                "pass": bool(row_pass),
                "problems": problems,
            }
        )

    summary = {
        "n_audited": len(results),
        "n_pass": n_pass,
        "all_pass": n_pass == len(results),
        "seed": SEED,
        "season_start_month": 10,
        "note": "audits 7-day history contiguity/end-at-t, delta_swe=swe_t1-swe_t, cross-check vs artifacts/taiesm1_daily_sierra_swe.csv, and raw predictor-array shape/finiteness, on randomly chosen Oct1-Apr1 samples from the 96 train water years.",
        "rows": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print(f"wrote {OUT_PATH}", flush=True)
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
