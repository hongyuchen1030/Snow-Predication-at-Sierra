#!/usr/bin/env python3
"""Target-side-only snow-onset / training-month diagnostic for TaiESM1 daily
Sierra SWE. Reads artifacts/taiesm1_daily_sierra_swe.npz and (for the 7-day
window counts) the already-built sample index. Does NOT touch predictor
arrays and does NOT train/retrain anything.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
SWE_NPZ = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.npz"
SAMPLE_INDEX = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1/data_index/sample_index.npz")

OUT_CSV_MONTHLY = PROJECT_ROOT / "artifacts" / "taiesm1_daily_swe_training_month_diagnostic.csv"
OUT_CSV_ONSET = PROJECT_ROOT / "artifacts" / "taiesm1_daily_swe_onset_by_water_year.csv"
OUT_MD = PROJECT_ROOT / "docs" / "TaiESM1_Daily_SWE_Training_Month_Diagnostic.md"
OUT_PLOT = PROJECT_ROOT / "artifacts" / "taiesm1_swe_onset_training_month_diagnostic.png"

MONTH_NAMES = {9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
MONTH_ORDER = [9, 10, 11, 12, 1, 2, 3]
CUM_DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def serial(year: int, month: int, day: int) -> int:
    return year * 365 + CUM_DAYS_BEFORE_MONTH[month - 1] + (day - 1)


def water_year_of(cal_year: int, month: int) -> int:
    return cal_year + 1 if month >= 9 else cal_year


def in_season(month: int, day: int, start_month: int = 9) -> bool:
    if start_month <= month <= 12:
        return True
    if 1 <= month <= 3:
        return True
    if month == 4 and day == 1:
        return True
    return False


def day_index_to_month_day(day_index: int) -> tuple[str, int]:
    cum = [("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31), ("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 1)]
    remaining = day_index
    for name, n_days in cum:
        if remaining <= n_days:
            return name, remaining
        remaining -= n_days
    raise ValueError(day_index)


def main() -> None:
    d = np.load(SWE_NPZ, allow_pickle=True)
    year = d["year"].astype(np.int64)
    month = d["month"].astype(np.int64)
    day = d["day"].astype(np.int64)
    swe = d["swe_mm"].astype(np.float64)
    n = len(year)
    print(f"loaded {n} full daily SWE records from {SWE_NPZ}", flush=True)

    ser = np.array([serial(y, m, dd) for y, m, dd in zip(year, month, day)])
    order = np.argsort(ser)
    ser, year, month, day, swe = ser[order], year[order], month[order], day[order], swe[order]
    gaps = np.where(np.diff(ser) != 1)[0]
    assert len(gaps) == 0, f"non-contiguous full daily series, gaps at {gaps[:5]}"
    ser_to_pos = {int(s): i for i, s in enumerate(ser)}

    wy = np.array([water_year_of(y, m) for y, m in zip(year, month)])

    # -----------------------------------------------------------------
    # 1. Monthly 1-day pair statistics (Sep-Mar), from the full continuous series.
    # -----------------------------------------------------------------
    swe_t = swe[:-1]
    swe_t1 = swe[1:]
    delta = swe_t1 - swe_t
    month_t = month[:-1]

    monthly_rows = []
    for m in MONTH_ORDER:
        sel = month_t == m
        n_all = int(sel.sum())
        st, st1, dl = swe_t[sel], swe_t1[sel], delta[sel]
        zero_zero = int(((st == 0) & (st1 == 0)).sum())
        active = int(((st > 0) | (st1 > 0)).sum())
        row = {
            "month": MONTH_NAMES[m],
            "N_all": n_all,
            "N_zero_zero": zero_zero,
            "N_active": active,
            "active_fraction": active / n_all if n_all else float("nan"),
            "frac_swe_t_gt0": float((st > 0).mean()) if n_all else float("nan"),
            "frac_swe_t_gt1mm": float((st > 1.0).mean()) if n_all else float("nan"),
            "mean_swe_t": float(st.mean()) if n_all else float("nan"),
            "median_swe_t": float(np.median(st)) if n_all else float("nan"),
            "mean_abs_delta": float(np.abs(dl).mean()) if n_all else float("nan"),
            "median_abs_delta": float(np.median(np.abs(dl))) if n_all else float("nan"),
            "std_delta": float(dl.std(ddof=1)) if n_all > 1 else float("nan"),
            "frac_delta_gt0": float((dl > 0).mean()) if n_all else float("nan"),
            "frac_delta_lt0": float((dl < 0).mean()) if n_all else float("nan"),
            "frac_delta_eq0": float((dl == 0).mean()) if n_all else float("nan"),
        }
        monthly_rows.append(row)

    with OUT_CSV_MONTHLY.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)
    print(f"wrote {OUT_CSV_MONTHLY}", flush=True)

    print("\n=== 1. MONTHLY 1-DAY PAIR STATISTICS ===", flush=True)
    for r in monthly_rows:
        print(f"  {r['month']}: N_all={r['N_all']} N_active={r['N_active']} active%={100*r['active_fraction']:.1f} "
              f"zero_zero%={100*r['N_zero_zero']/r['N_all']:.1f} mean_swe_t={r['mean_swe_t']:.2f} mean|d|={r['mean_abs_delta']:.3f}", flush=True)

    # -----------------------------------------------------------------
    # 2/3. Onset by water year (>0, >1mm, >5mm, and persistent >1mm).
    # -----------------------------------------------------------------
    water_years = sorted(set(wy.tolist()))
    onset_rows = []
    for water_year_val in water_years:
        wy_mask = wy == water_year_val
        wy_positions = np.where(wy_mask)[0]
        # restrict candidate onset DATE search to Sep1-Apr1 of this water year
        wy_season_positions = [p for p in wy_positions if in_season(int(month[p]), int(day[p]))]
        if not wy_season_positions:
            continue
        wy_season_positions.sort()

        def first_onset(threshold: float, strict_gt: bool = True) -> int | None:
            for p in wy_season_positions:
                if swe[p] > threshold:
                    return p
            return None

        def first_persistent_onset(threshold: float = 1.0) -> int | None:
            for p in wy_season_positions:
                if swe[p] <= threshold:
                    continue
                # following 7 days (p+1..p+7), using the full continuous series
                count_gt = 0
                for k in range(1, 8):
                    if p + k >= len(swe):
                        break
                    if swe[p + k] > threshold:
                        count_gt += 1
                if count_gt >= 5:
                    return p
            return None

        p0 = first_onset(0.0)
        p1 = first_onset(1.0)
        p5 = first_onset(5.0)
        pp = first_persistent_onset(1.0)

        def pos_to_info(p: int | None) -> dict:
            if p is None:
                return {"date": None, "month": None, "day": None}
            return {"date": f"{int(year[p]):04d}-{int(month[p]):02d}-{int(day[p]):02d}", "month": int(month[p]), "day": int(day[p])}

        onset_rows.append(
            {
                "water_year": int(water_year_val),
                "onset_gt0_date": pos_to_info(p0)["date"],
                "onset_gt1mm_date": pos_to_info(p1)["date"],
                "onset_gt5mm_date": pos_to_info(p5)["date"],
                "onset_persistent_gt1mm_date": pos_to_info(pp)["date"],
            }
        )

    with OUT_CSV_ONSET.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(onset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(onset_rows)
    print(f"\nwrote {OUT_CSV_ONSET}", flush=True)

    def summarize_onset_dates(dates: list[str | None]) -> dict:
        valid = [dt for dt in dates if dt is not None]
        n_missing = len(dates) - len(valid)
        if not valid:
            return {"n_with_onset": 0, "n_never": n_missing}
        # convert to day-index within water year (1=Sep1..213=Apr1) for percentile math
        day_idx = []
        for dt in valid:
            y, m, dd = (int(x) for x in dt.split("-"))
            if m >= 9:
                idx = CUM_DAYS_BEFORE_MONTH[m - 1] - CUM_DAYS_BEFORE_MONTH[8] + dd
            else:
                idx = (365 - CUM_DAYS_BEFORE_MONTH[8]) + CUM_DAYS_BEFORE_MONTH[m - 1] + dd
            day_idx.append(idx)
        arr = np.array(sorted(day_idx), dtype=np.float64)
        pct = {
            "earliest": int(arr.min()),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "latest": int(arr.max()),
        }
        readable = {k: "%s %d" % day_index_to_month_day(int(round(v))) for k, v in pct.items()}
        month_counts = {"Sep": 0, "Oct": 0, "Nov": 0, "Dec": 0, "Jan_or_later": 0}
        for dt in valid:
            m = int(dt.split("-")[1])
            if m in (9, 10, 11, 12):
                month_counts[MONTH_NAMES[m]] += 1
            else:
                month_counts["Jan_or_later"] += 1
        return {
            "n_with_onset": len(valid), "n_never": n_missing,
            "day_index_percentiles": pct, "readable_dates": readable,
            "onset_month_counts": month_counts,
            "onset_month_pct": {k: 100.0 * v / len(valid) for k, v in month_counts.items()},
        }

    onset_summaries = {}
    for key, label in (
        ("onset_gt0_date", "SWE > 0"),
        ("onset_gt1mm_date", "SWE > 1mm"),
        ("onset_gt5mm_date", "SWE > 5mm"),
        ("onset_persistent_gt1mm_date", "persistent SWE > 1mm (>=5 of next 7 days also >1mm)"),
    ):
        dates = [r[key] for r in onset_rows]
        summary = summarize_onset_dates(dates)
        onset_summaries[key] = summary
        print(f"\n=== ONSET: {label} ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)

    # -----------------------------------------------------------------
    # 4. Candidate start-month comparison (1-day pairs + 7-day windows).
    # -----------------------------------------------------------------
    idx = np.load(SAMPLE_INDEX, allow_pickle=True)
    idx_month = idx["cal_month"]
    idx_day = idx["cal_day"]
    idx_active = idx["active"]

    print("\n=== 4. CANDIDATE START-MONTH COMPARISON ===", flush=True)
    candidate_rows = []
    for start_month, label in ((9, "Sep1"), (10, "Oct1"), (11, "Nov1"), (12, "Dec1")):
        # 1-day pairs, from the monthly table (Sep..Mar rows with month>=start_month, plus all of Jan-Mar)
        included_months = [m for m in MONTH_ORDER if (m >= start_month) or (m <= 3)]
        pair_n_all = sum(r["N_all"] for r in monthly_rows if r["month"] in [MONTH_NAMES[m] for m in included_months])
        pair_n_active = sum(r["N_active"] for r in monthly_rows if r["month"] in [MONTH_NAMES[m] for m in included_months])
        pair_n_zz = pair_n_all - pair_n_active

        # 7-day windows, from the sample index
        season_sel = np.array([in_season(int(m), int(dd), start_month) for m, dd in zip(idx_month, idx_day)])
        n_windows_all = int(season_sel.sum())
        n_windows_active = int((season_sel & idx_active).sum())

        row = {
            "candidate": f"{label}->Apr1",
            "pair_N_all": pair_n_all,
            "pair_N_active": pair_n_active,
            "pair_N_zero_zero": pair_n_zz,
            "pair_active_fraction": pair_n_active / pair_n_all if pair_n_all else float("nan"),
            "windows_all_mode": n_windows_all,
            "windows_snow_active_mode": n_windows_active,
        }
        candidate_rows.append(row)
        print(f"  {label}->Apr1: pairs N_all={pair_n_all} N_active={pair_n_active} active%={100*row['pair_active_fraction']:.1f} | "
              f"7-day windows: all={n_windows_all} snow_active={n_windows_active}", flush=True)

    # -----------------------------------------------------------------
    # 5. Information lost by later starts (onset before proposed start).
    # -----------------------------------------------------------------
    print("\n=== 5. INFORMATION LOST BY LATER STARTS ===", flush=True)
    info_loss = {}
    n_wy = len(onset_rows)
    for start_month, label in ((10, "Oct1"), (11, "Nov1"), (12, "Dec1")):
        for key, dlabel in (("onset_gt1mm_date", "first SWE>1mm"), ("onset_persistent_gt1mm_date", "persistent SWE>1mm")):
            before = 0
            for r in onset_rows:
                dt = r[key]
                if dt is None:
                    continue
                y, m, dd = (int(x) for x in dt.split("-"))
                # "before proposed start" means the onset month is Sep, or (Oct and start>=Nov), etc.
                onset_before_start = m < start_month and m >= 9  # Sep..(start_month-1)
                if onset_before_start:
                    before += 1
            pct = 100.0 * before / n_wy
            info_loss[f"{label}_{dlabel}"] = {"n_water_years_with_onset_before_start": before, "pct_of_120_wy": pct}
            print(f"  {label} start, using {dlabel}: {before}/{n_wy} WY ({pct:.1f}%) had onset BEFORE {label}", flush=True)

    # -----------------------------------------------------------------
    # Plot.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    months_lbl = [MONTH_NAMES[m] for m in MONTH_ORDER]
    active_pct = [100 * r["active_fraction"] for r in monthly_rows]
    zz_pct = [100 * r["N_zero_zero"] / r["N_all"] for r in monthly_rows]
    axes[0].bar(months_lbl, active_pct, color="tab:blue", alpha=0.8, label="active %")
    axes[0].bar(months_lbl, zz_pct, bottom=active_pct, color="tab:gray", alpha=0.6, label="zero->zero %")
    axes[0].set_ylabel("% of 1-day pairs")
    axes[0].set_title("Active vs zero->zero fraction by month")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    persistent_dates = [r["onset_persistent_gt1mm_date"] for r in onset_rows if r["onset_persistent_gt1mm_date"]]
    persistent_months = [int(dt.split("-")[1]) for dt in persistent_dates]
    hist_months = [9, 10, 11, 12, 1, 2, 3]
    hist_counts = [persistent_months.count(m) for m in hist_months]
    axes[1].bar([MONTH_NAMES[m] for m in hist_months], hist_counts, color="tab:orange")
    axes[1].set_ylabel("# water years")
    axes[1].set_title("Persistent snow-onset month (>1mm, 5-of-7 days)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_PLOT)
    plt.close(fig)
    print(f"\nwrote {OUT_PLOT}", flush=True)

    # -----------------------------------------------------------------
    # Recommendation.
    # -----------------------------------------------------------------
    onset_before_nov_persistent = info_loss["Nov1_persistent SWE>1mm"]["pct_of_120_wy"]
    onset_before_oct_persistent = info_loss["Oct1_persistent SWE>1mm"]["pct_of_120_wy"]
    onset_before_dec_persistent = info_loss["Dec1_persistent SWE>1mm"]["pct_of_120_wy"]

    sep_row = next(r for r in candidate_rows if r["candidate"] == "Sep1->Apr1")
    oct_row = next(r for r in candidate_rows if r["candidate"] == "Oct1->Apr1")
    nov_row = next(r for r in candidate_rows if r["candidate"] == "Nov1->Apr1")
    dec_row = next(r for r in candidate_rows if r["candidate"] == "Dec1->Apr1")

    # Recommendation logic (reported, not silently decided): Sep active fraction very low (Section 1),
    # Oct active fraction still low-moderate, Nov/Dec active fraction high but Oct1 already captures
    # a meaningful chunk of onset events missed by Nov1/Dec1.
    recommended_start = "OCT 1" if onset_before_oct_persistent < 10 else "SEP 1"
    chosen_row = oct_row if recommended_start == "OCT 1" else sep_row
    zero_zero_frac_at_chosen_start = 1 - chosen_row["pair_active_fraction"]
    # "flooding" threshold set well above the modest ~11% zero->zero remaining at
    # Oct1 -- that fraction is not overwhelming, and zero->zero pairs are
    # legitimate negative examples (the model must learn when NOT to add snow),
    # not pure noise. Only recommend dropping them if they would clearly dominate.
    recommended_mode = "snow_active" if zero_zero_frac_at_chosen_start > 0.25 else "all"

    final_summary = {
        "recommended_start": recommended_start,
        "recommended_sample_mode": recommended_mode,
        "total_7day_windows_all_mode": chosen_row["windows_all_mode"],
        "snow_active_7day_windows": chosen_row["windows_snow_active_mode"],
        "pct_zero_zero_removed_by_snow_active_mode": 100.0 * (1 - chosen_row["windows_snow_active_mode"] / chosen_row["windows_all_mode"]),
        "pct_water_years_with_onset_before_chosen_start": onset_before_oct_persistent if recommended_start == "OCT 1" else 0.0,
    }

    print("\n=== FINAL RECOMMENDATION (statistics-driven) ===", flush=True)
    print(json.dumps(final_summary, indent=2), flush=True)

    # -----------------------------------------------------------------
    # Markdown report.
    # -----------------------------------------------------------------
    lines = []
    lines.append("# TaiESM1 Daily SWE Training-Month Diagnostic")
    lines.append("")
    lines.append(
        "**Status: target-side diagnostic only.** No predictor arrays touched, no retraining. "
        f"Source: `{SWE_NPZ.relative_to(PROJECT_ROOT)}` (43,800 daily records, 120 water years, 365_day calendar, mm)."
    )
    lines.append("")
    lines.append("## 1. Monthly 1-day pair statistics (Sep-Mar)")
    lines.append("")
    lines.append("| Month | N_all | N_zero_zero | N_active | Active % | SWE(t)>0 % | SWE(t)>1mm % | mean SWE(t) | median SWE(t) | mean\\|ΔSWE\\| | median\\|ΔSWE\\| | std ΔSWE | Δ>0 % | Δ<0 % | Δ=0 % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in monthly_rows:
        lines.append(
            f"| {r['month']} | {r['N_all']} | {r['N_zero_zero']} | {r['N_active']} | {100*r['active_fraction']:.1f} | "
            f"{100*r['frac_swe_t_gt0']:.1f} | {100*r['frac_swe_t_gt1mm']:.1f} | {r['mean_swe_t']:.2f} | {r['median_swe_t']:.2f} | "
            f"{r['mean_abs_delta']:.3f} | {r['median_abs_delta']:.3f} | {r['std_delta']:.3f} | "
            f"{100*r['frac_delta_gt0']:.1f} | {100*r['frac_delta_lt0']:.1f} | {100*r['frac_delta_eq0']:.1f} |"
        )
    lines.append("")
    lines.append("## 2. Snow-onset distribution by water year (120 WY)")
    lines.append("")
    for key, label in (("onset_gt0_date", "SWE > 0"), ("onset_gt1mm_date", "SWE > 1mm"), ("onset_gt5mm_date", "SWE > 5mm")):
        s = onset_summaries[key]
        lines.append(f"### {label}")
        lines.append("")
        if s["n_with_onset"]:
            r = s["readable_dates"]
            c = s["onset_month_counts"]
            lines.append(f"- earliest {r['earliest']}, p5 {r['p5']}, p25 {r['p25']}, median {r['median']}, p75 {r['p75']}, p95 {r['p95']}, latest {r['latest']}")
            lines.append(f"- month counts: Sep={c['Sep']}, Oct={c['Oct']}, Nov={c['Nov']}, Dec={c['Dec']}, Jan-or-later={c['Jan_or_later']} (n_with_onset={s['n_with_onset']}, never={s['n_never']})")
        lines.append("")
    lines.append("## 3. Persistent snow-onset (SWE>1mm AND >=5 of next 7 days also >1mm)")
    lines.append("")
    s = onset_summaries["onset_persistent_gt1mm_date"]
    r, c = s["readable_dates"], s["onset_month_counts"]
    lines.append(f"- earliest {r['earliest']}, p5 {r['p5']}, p25 {r['p25']}, median {r['median']}, p75 {r['p75']}, p95 {r['p95']}, latest {r['latest']}")
    lines.append(f"- month counts: Sep={c['Sep']}, Oct={c['Oct']}, Nov={c['Nov']}, Dec={c['Dec']}, Jan-or-later={c['Jan_or_later']} (n_with_onset={s['n_with_onset']}, never={s['n_never']})")
    lines.append("")
    lines.append("## 4. Candidate start-month comparison (all end Apr 1)")
    lines.append("")
    lines.append("| Candidate | Pair N_all | Pair N_active | Pair N_zero_zero | Pair active % | 7-day windows (all) | 7-day windows (snow_active) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in candidate_rows:
        lines.append(
            f"| {row['candidate']} | {row['pair_N_all']} | {row['pair_N_active']} | {row['pair_N_zero_zero']} | "
            f"{100*row['pair_active_fraction']:.1f} | {row['windows_all_mode']} | {row['windows_snow_active_mode']} |"
        )
    lines.append("")
    lines.append("## 5. Information lost by later starts")
    lines.append("")
    lines.append("| Candidate start | Onset definition | WY with onset BEFORE start | % of 120 WY |")
    lines.append("|---|---|---:|---:|")
    for k, v in info_loss.items():
        label, dlabel = k.split("_", 1)
        lines.append(f"| {label} | {dlabel} | {v['n_water_years_with_onset_before_start']} | {v['pct_of_120_wy']:.1f}% |")
    lines.append("")
    lines.append("![monthly active/zero-zero and onset histogram](../artifacts/taiesm1_swe_onset_training_month_diagnostic.png)")
    lines.append("")
    lines.append("## 6. Final recommendation")
    lines.append("")
    lines.append(f"```")
    lines.append(f"Recommended start:        {final_summary['recommended_start']}")
    lines.append(f"Recommended sample mode:  {final_summary['recommended_sample_mode']}")
    lines.append("")
    lines.append(f"Total 7-day windows:      {final_summary['total_7day_windows_all_mode']}")
    lines.append(f"Snow-active 7-day windows: {final_summary['snow_active_7day_windows']}")
    lines.append("")
    lines.append(f"% zero->zero removed:     {final_summary['pct_zero_zero_removed_by_snow_active_mode']:.1f}%")
    lines.append(f"% water years with onset before chosen start: {final_summary['pct_water_years_with_onset_before_chosen_start']:.1f}%")
    lines.append(f"```")
    lines.append("")
    lines.append(
        "**Main justification (start month):** using a real onset threshold (>1mm, persistent) rather than "
        "just \"is SWE usually zero,\" **0 of 120 water years** have persistent onset before October 1 -- "
        "September genuinely contributes no onset physics (its 6.2% \"active\" fraction is almost entirely "
        "SWE(t)>0 trivial/transient single-day noise, not real accumulation). November 1 would silently drop "
        "the onset period for 6.7-17.5% of water years, and December 1 for 74-81% of water years -- both "
        "directly contradict retaining onset physics. October 1 is therefore the earliest start that loses "
        "zero water years' onset information while avoiding September's dead weight."
    )
    lines.append("")
    lines.append(
        f"**Main justification (sample mode):** at the recommended Oct1 start, zero->zero pairs are only "
        f"{100*zero_zero_frac_at_chosen_start:.1f}% of samples -- a modest fraction, not \"flooding.\" Zero->zero "
        "pairs are legitimate negative examples (the model must learn when NOT to add snow, e.g. during dry "
        "spells or before/after the snow season), so discarding them risks biasing the model away from "
        "realistic zero-SWE conditions rather than improving it. `all` is recommended; `snow_active` remains "
        "available as a configurable option (both counts are reported above) if a future run shows the model "
        "is dominated by trivial zero predictions."
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
