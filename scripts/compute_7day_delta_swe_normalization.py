#!/usr/bin/env python3
"""Compute predictor and target normalization statistics from TRAINING water
years only (Sep1-Apr1 window, the most permissive season -- a superset of
every narrower season start choice, so one normalization is reusable across
all season configs). Per-channel, per-pixel mean/std, matching the
scientifically sensible part of the existing seasonal CNN's normalization
(spatial climatology varies strongly by location for these variables), but
simplified: no per-month axis (every day is treated the same way here).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
PREDICTOR_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1")
EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1")
INDEX_PATH = EXPERIMENT_ROOT / "data_index" / "sample_index.npz"
SPLIT_PATH = EXPERIMENT_ROOT / "splits" / "water_year_split.json"
OUT_DIR = EXPERIMENT_ROOT / "normalization"

N_HISTORY_DAYS = 7
N_CHANNELS = 12


def in_season_mask(month: np.ndarray, day: np.ndarray, start_month: int = 9) -> np.ndarray:
    fall = (month >= start_month) & (month <= 12)
    winter = (month >= 1) & (month <= 3)
    apr1 = (month == 4) & (day == 1)
    return fall | winter | apr1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text())
    train_wy = set(split["train_water_years"])

    idx = np.load(INDEX_PATH, allow_pickle=True)
    water_year = idx["water_year"]
    cal_month = idx["cal_month"]
    cal_day = idx["cal_day"]
    history_year = idx["history_year"]
    history_doy = idx["history_doy"]
    history_exp = idx["history_exp"]
    exp_code_map = json.loads(str(idx["exp_code_map"]))
    code_to_exp = {v: k for k, v in exp_code_map.items()}

    train_mask = np.isin(water_year, list(train_wy)) & in_season_mask(cal_month, cal_day, start_month=9)
    train_rows = np.where(train_mask)[0]
    print(f"training samples used for normalization (Sep1-Apr1, train WYs): {len(train_rows)}", flush=True)

    # collect the UNIQUE set of predictor days actually referenced by these
    # training samples' 7-day windows (avoid re-reading the same day many times).
    unique_days: set[tuple[int, int, int]] = set()
    for row in train_rows:
        for k in range(N_HISTORY_DAYS):
            unique_days.add((int(history_exp[row, k]), int(history_year[row, k]), int(history_doy[row, k])))
    print(f"unique predictor days to scan: {len(unique_days)}", flush=True)

    # group by (experiment, year) so each yearly npy is opened (mmap) once.
    by_file: dict[tuple[int, int], list[int]] = {}
    for exp_code, year, doy in unique_days:
        by_file.setdefault((exp_code, year), []).append(doy)

    sum_ = np.zeros((N_CHANNELS, 120, 240), dtype=np.float64)
    sumsq = np.zeros((N_CHANNELS, 120, 240), dtype=np.float64)
    count = 0

    for (exp_code, year), doy_list in sorted(by_file.items()):
        exp_name = code_to_exp[exp_code]
        path = PREDICTOR_ROOT / "years" / exp_name / f"predictors_physical_{year}.npy"
        arr = np.load(path, mmap_mode="r")  # [365, 12, 120, 240]
        rows = np.asarray(sorted(doy_list)) - 1  # doy 1-indexed -> row 0-indexed
        chunk = np.asarray(arr[rows], dtype=np.float64)  # [n_days_this_year, 12, 120, 240]
        sum_ += chunk.sum(axis=0)
        sumsq += (chunk ** 2).sum(axis=0)
        count += chunk.shape[0]
        print(f"  scanned {exp_name} {year}: {chunk.shape[0]} days (running total {count})", flush=True)

    mean = sum_ / count
    var = sumsq / count - mean ** 2
    var = np.clip(var, a_min=1e-8, a_max=None)
    std = np.sqrt(var)

    channel_names = list(np.load(PREDICTOR_ROOT / "metadata.npz", allow_pickle=True)["channel_names"])
    np.savez_compressed(
        OUT_DIR / "feature_normalization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        channel_names=np.array(channel_names),
        n_days_used=count,
    )
    print(f"wrote {OUT_DIR / 'feature_normalization.npz'} (mean/std shape={mean.shape}, n_days_used={count})", flush=True)

    # -----------------------------------------------------------------
    # Target normalization stats (train samples only, Sep1-Apr1 season).
    # -----------------------------------------------------------------
    delta_swe = idx["delta_swe"]
    train_delta = delta_swe[train_rows]
    target_mu = float(train_delta.mean())
    target_sigma = float(train_delta.std(ddof=1))
    target_stats = {
        "target_mu_mm": target_mu,
        "target_sigma_mm": target_sigma,
        "n_train_samples_sep1_apr1": int(len(train_rows)),
        "note": "delta_swe = SWE(t+1) - SWE(t), mm; computed from training water years, Sep1-Apr1 season only",
    }
    (OUT_DIR / "target_normalization.json").write_text(json.dumps(target_stats, indent=2))
    print(json.dumps(target_stats, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
