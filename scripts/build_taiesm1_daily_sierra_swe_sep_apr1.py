#!/usr/bin/env python3
"""Filter the already-built, already-validated TaiESM1 daily Sierra SWE
series (artifacts/taiesm1_daily_sierra_swe.{csv,npz}) down to the Sep 1 ->
Apr 1 seasonal window for every water year. Does NOT touch the raw WUS-D3
files or recompute the spatial reduction.

Water year convention (explicit): Sep 1 of calendar year Y through Apr 1 of
calendar year Y+1 belongs to WY (Y+1). Each source file (file_year=Y, WUS
water-style Sep(Y)-Aug(Y+1)) directly supplies the Sep1(Y)-Apr1(Y+1) slice
for WY(Y+1) at the start of its own 365-day record.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.csv"
IN_NPZ = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.npz"
OUT_CSV = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe_sep_apr1.csv"
OUT_NPZ = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe_sep_apr1.npz"
OUT_SUMMARY = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe_sep_apr1_summary.json"


def in_seasonal_window(month: int, day: int) -> bool:
    # Sep(9), Oct(10), Nov(11), Dec(12), Jan(1), Feb(2), Mar(3) in full;
    # April only day 1.
    if month in (9, 10, 11, 12, 1, 2, 3):
        return True
    if month == 4 and day == 1:
        return True
    return False


def main() -> None:
    with IN_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    print(f"loaded {len(rows)} source daily records from {IN_CSV}", flush=True)

    kept = []
    for r in rows:
        month = int(r["cal_month"])
        day = int(r["cal_day"])
        if not in_seasonal_window(month, day):
            continue
        file_year = int(r["file_year"])
        water_year = file_year + 1
        kept.append(
            {
                "model": r["model"],
                "experiment": r["experiment"],
                "water_year": water_year,
                "file_year": file_year,
                "cal_year": int(r["cal_year"]),
                "cal_month": month,
                "cal_day": day,
                "date_key": f"{int(r['cal_year']):04d}-{month:02d}-{day:02d}",
                "day_of_year_source": int(r["day_of_year"]),
                "swe_mm": float(r["swe_mm"]),
            }
        )

    # sort by water_year then chronological order within the window
    kept.sort(key=lambda r: (r["water_year"], r["day_of_year_source"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model", "experiment", "water_year", "file_year", "cal_year",
                "cal_month", "cal_day", "date_key", "day_of_year_source", "swe_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(kept)
    print(f"wrote {OUT_CSV}", flush=True)

    water_year = np.array([r["water_year"] for r in kept], dtype=np.int32)
    year = np.array([r["cal_year"] for r in kept], dtype=np.int32)
    month = np.array([r["cal_month"] for r in kept], dtype=np.int8)
    day = np.array([r["cal_day"] for r in kept], dtype=np.int8)
    date_key = np.array([r["date_key"] for r in kept])
    day_of_year = np.array([r["day_of_year_source"] for r in kept], dtype=np.int16)
    swe_mm = np.array([r["swe_mm"] for r in kept], dtype=np.float64)
    experiment_arr = np.array([r["experiment"] for r in kept])

    np.savez_compressed(
        OUT_NPZ,
        water_year=water_year,
        year=year,
        month=month,
        day=day,
        date_key=date_key,
        day_of_year=day_of_year,
        swe_mm=swe_mm,
        experiment=experiment_arr,
        model=np.array(["TaiESM1"] * len(kept)),
    )
    print(f"wrote {OUT_NPZ}", flush=True)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    water_years_present = sorted(set(water_year.tolist()))
    n_water_years = len(water_years_present)
    counts_per_wy = {}
    for wy in water_years_present:
        counts_per_wy[wy] = int((water_year == wy).sum())
    distinct_counts = set(counts_per_wy.values())

    total_records = len(kept)
    total_delta_pairs = sum(max(0, c - 1) for c in counts_per_wy.values())

    first = kept[0]
    last = kept[-1]

    # verification 6: no Apr2-Aug31 dates remain
    violations = [
        r for r in kept
        if not in_seasonal_window(r["cal_month"], r["cal_day"])
    ]

    # verification 7: Mar31 -> Apr1 exists for every complete water year
    by_wy = {}
    for r in kept:
        by_wy.setdefault(r["water_year"], []).append(r)
    missing_mar31_apr1 = []
    for wy, recs in by_wy.items():
        has_mar31 = any(r["cal_month"] == 3 and r["cal_day"] == 31 for r in recs)
        has_apr1 = any(r["cal_month"] == 4 and r["cal_day"] == 1 for r in recs)
        if not (has_mar31 and has_apr1):
            missing_mar31_apr1.append({"water_year": wy, "has_mar31": has_mar31, "has_apr1": has_apr1})

    report = {
        "source_csv": str(IN_CSV),
        "source_npz": str(IN_NPZ),
        "1_n_water_years": n_water_years,
        "1_water_year_range": [min(water_years_present), max(water_years_present)],
        "2_records_per_water_year": {
            "distinct_values": sorted(distinct_counts),
            "uniform": len(distinct_counts) == 1,
            "expected_if_365day_calendar": 213,
        },
        "3_total_sep1_apr1_records": total_records,
        "4_total_possible_1day_delta_pairs": total_delta_pairs,
        "5_first_retained_date": f"{first['cal_year']:04d}-{first['cal_month']:02d}-{first['cal_day']:02d} (WY{first['water_year']})",
        "5_last_retained_date": f"{last['cal_year']:04d}-{last['cal_month']:02d}-{last['cal_day']:02d} (WY{last['water_year']})",
        "6_apr2_aug31_violations": len(violations),
        "7_water_years_missing_mar31_or_apr1": missing_mar31_apr1,
    }
    OUT_SUMMARY.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"\nwrote {OUT_SUMMARY}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
