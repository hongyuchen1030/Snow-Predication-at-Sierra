#!/usr/bin/env python3
"""SWE-target diagnostic ONLY: analyze snow-onset timing and Sep1-Apr1
ΔSWE_1d activity for TaiESM1, using the already-built, already-validated
Sep1->Apr1 daily Sierra SWE subset. Builds no predictor tensors, trains
nothing, and does not modify the existing validated daily artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe_sep_apr1.csv"
OUT_CSV = PROJECT_ROOT / "artifacts" / "taiesm1_daily_swe_training_month_diagnostic.csv"
OUT_MD = PROJECT_ROOT / "docs" / "TaiESM1_Daily_SWE_Training_Month_Diagnostic.md"

MONTH_NAMES = {9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
MONTH_ORDER = [9, 10, 11, 12, 1, 2, 3]  # Sep..Mar (Apr has no valid "t" since Apr1 is always the endpoint)

# day_of_year_source: 1=Sep1 ... 213=Apr1, fixed by the 365_day calendar (no leap variability).
# Candidate-start day indices, derived directly from this fixed mapping.
CANDIDATE_STARTS = {
    "Sep1->Apr1": 1,
    "Oct1->Apr1": 31,
    "Nov1->Apr1": 62,
    "Dec1->Apr1": 92,
}


def day_index_to_month_day(day_index: int) -> tuple[str, int]:
    """day_index 1..213 -> (month_name, day), 1=Sep1 ... 213=Apr1, 365_day calendar."""
    cum = [("Sep", 9, 30), ("Oct", 10, 31), ("Nov", 11, 30), ("Dec", 12, 31), ("Jan", 1, 31), ("Feb", 2, 28), ("Mar", 3, 31), ("Apr", 4, 1)]
    remaining = day_index
    for name, _, n_days in cum:
        if remaining <= n_days:
            return name, remaining
        remaining -= n_days
    raise ValueError(day_index)


def main() -> None:
    with IN_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    print(f"loaded {len(rows)} Sep1-Apr1 records from {IN_CSV}", flush=True)

    by_wy: dict[int, list[dict]] = {}
    for r in rows:
        wy = int(r["water_year"])
        by_wy.setdefault(wy, []).append(r)
    for wy in by_wy:
        by_wy[wy].sort(key=lambda r: int(r["day_of_year_source"]))

    water_years = sorted(by_wy)
    n_wy = len(water_years)
    print(f"water years: {n_wy} ({min(water_years)}-{max(water_years)})", flush=True)

    # -----------------------------------------------------------------
    # Build all valid (t, t+1) pairs across all 120 water years, Sep1-Apr1 only.
    # -----------------------------------------------------------------
    pairs = []  # each: month_t, swe_t, swe_t1, delta, water_year, day_index_t
    for wy, recs in by_wy.items():
        for i in range(len(recs) - 1):
            t = recs[i]
            t1 = recs[i + 1]
            assert int(t1["day_of_year_source"]) == int(t["day_of_year_source"]) + 1
            swe_t = float(t["swe_mm"])
            swe_t1 = float(t1["swe_mm"])
            pairs.append(
                {
                    "water_year": wy,
                    "day_index_t": int(t["day_of_year_source"]),
                    "month_t": int(t["cal_month"]),
                    "swe_t": swe_t,
                    "swe_t1": swe_t1,
                    "delta": swe_t1 - swe_t,
                }
            )
    print(f"total valid Sep1-Apr1 1-day pairs: {len(pairs)}", flush=True)

    # -----------------------------------------------------------------
    # Section 2/3: per-month table (Sep..Mar)
    # -----------------------------------------------------------------
    monthly_rows = []
    for month in MONTH_ORDER:
        month_pairs = [p for p in pairs if p["month_t"] == month]
        n_all = len(month_pairs)
        swe_t_arr = np.array([p["swe_t"] for p in month_pairs])
        swe_t1_arr = np.array([p["swe_t1"] for p in month_pairs])
        delta_arr = np.array([p["delta"] for p in month_pairs])
        active_mask = (swe_t_arr > 0) | (swe_t1_arr > 0)
        n_active = int(active_mask.sum())
        n_zero_zero = n_all - n_active
        active_fraction = n_active / n_all if n_all else float("nan")

        row = {
            "month": MONTH_NAMES[month],
            "N_all": n_all,
            "N_active": n_active,
            "N_zero_to_zero": n_zero_zero,
            "active_pct": 100.0 * active_fraction,
            "swe_gt0_pct": 100.0 * float((swe_t_arr > 0).mean()) if n_all else float("nan"),
            "mean_swe_t": float(swe_t_arr.mean()) if n_all else float("nan"),
            "median_swe_t": float(np.median(swe_t_arr)) if n_all else float("nan"),
            "mean_delta": float(delta_arr.mean()) if n_all else float("nan"),
            "std_delta": float(delta_arr.std(ddof=1)) if n_all > 1 else float("nan"),
            "delta_gt0_pct": 100.0 * float((delta_arr > 0).mean()) if n_all else float("nan"),
            "delta_lt0_pct": 100.0 * float((delta_arr < 0).mean()) if n_all else float("nan"),
            "delta_eq0_pct": 100.0 * float((delta_arr == 0).mean()) if n_all else float("nan"),
        }
        monthly_rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)
    print(f"wrote {OUT_CSV}", flush=True)

    print("\n| Month | N_all | N_active | Active % | SWE>0 % | mean SWE | mean dSWE | std dSWE | d>0 % | d<0 % |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for row in monthly_rows:
        print(
            f"| {row['month']} | {row['N_all']} | {row['N_active']} | {row['active_pct']:.1f} | "
            f"{row['swe_gt0_pct']:.1f} | {row['mean_swe_t']:.2f} | {row['mean_delta']:+.3f} | "
            f"{row['std_delta']:.3f} | {row['delta_gt0_pct']:.1f} | {row['delta_lt0_pct']:.1f} |",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Section 4: snow-onset timing by water year
    # -----------------------------------------------------------------
    onset_gt0 = {}
    onset_gt1 = {}
    for wy, recs in by_wy.items():
        first_gt0 = next((int(r["day_of_year_source"]) for r in recs if float(r["swe_mm"]) > 0.0), None)
        first_gt1 = next((int(r["day_of_year_source"]) for r in recs if float(r["swe_mm"]) > 1.0), None)
        onset_gt0[wy] = first_gt0
        onset_gt1[wy] = first_gt1

    def summarize_onsets(onset_map: dict[int, int | None]) -> dict:
        values = sorted(v for v in onset_map.values() if v is not None)
        n_missing = sum(1 for v in onset_map.values() if v is None)
        if not values:
            return {"n_with_onset": 0, "n_never_onset": n_missing}
        arr = np.array(values, dtype=np.float64)
        percentiles = {
            "earliest": int(arr.min()),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "latest": int(arr.max()),
        }
        readable = {k: (f"{day_index_to_month_day(int(round(v)))[0]} {day_index_to_month_day(int(round(v)))[1]}") for k, v in percentiles.items()}
        month_counts = {"Sep": 0, "Oct": 0, "Nov": 0, "Dec": 0, "Jan_or_later": 0}
        for v in values:
            m, _ = day_index_to_month_day(int(v))
            if m in ("Sep", "Oct", "Nov", "Dec"):
                month_counts[m] += 1
            else:
                month_counts["Jan_or_later"] += 1
        return {
            "n_with_onset": len(values),
            "n_never_onset": n_missing,
            "day_index_percentiles": percentiles,
            "readable_dates": readable,
            "onset_month_counts": month_counts,
        }

    onset_summary_gt0 = summarize_onsets(onset_gt0)
    onset_summary_gt1 = summarize_onsets(onset_gt1)

    print("\n=== Onset summary: SWE > 0 ===", flush=True)
    print(json.dumps(onset_summary_gt0, indent=2), flush=True)
    print("\n=== Onset summary: SWE > 1 mm ===", flush=True)
    print(json.dumps(onset_summary_gt1, indent=2), flush=True)

    # -----------------------------------------------------------------
    # Section 5: candidate training-start comparison
    # -----------------------------------------------------------------
    candidate_results = {}
    for label, start_day in CANDIDATE_STARTS.items():
        cand_pairs = [p for p in pairs if p["day_index_t"] >= start_day]
        n_all = len(cand_pairs)
        swe_t_arr = np.array([p["swe_t"] for p in cand_pairs])
        swe_t1_arr = np.array([p["swe_t1"] for p in cand_pairs])
        active_mask = (swe_t_arr > 0) | (swe_t1_arr > 0)
        n_active = int(active_mask.sum())
        n_zero_zero = n_all - n_active
        active_fraction = n_active / n_all if n_all else float("nan")
        candidate_results[label] = {
            "N_all": n_all,
            "N_active": n_active,
            "N_zero_to_zero": n_zero_zero,
            "active_fraction": active_fraction,
        }

    print("\n=== Candidate training-start comparison ===", flush=True)
    print("| Candidate | N_all | N_active | N_zero_to_zero | active_fraction |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for label, res in candidate_results.items():
        print(f"| {label} | {res['N_all']} | {res['N_active']} | {res['N_zero_to_zero']} | {res['active_fraction']:.4f} |", flush=True)

    # -----------------------------------------------------------------
    # Write markdown doc
    # -----------------------------------------------------------------
    md_lines = []
    md_lines.append("# TaiESM1 Daily SWE Training-Month Diagnostic")
    md_lines.append("")
    md_lines.append(
        "**Status: SWE-target diagnostic only.** No predictor dataset built, no training. "
        "Source: `artifacts/taiesm1_daily_sierra_swe_sep_apr1.csv` (already validated, unmodified). "
        f"Analysis restricted to Sep1->Apr1 for all {n_wy} water years ({min(water_years)}-{max(water_years)}); "
        f"total valid 1-day pairs = {len(pairs)}."
    )
    md_lines.append("")
    md_lines.append("## Monthly SWE / ΔSWE activity (Sep-Mar; April has no valid `t`)")
    md_lines.append("")
    md_lines.append("| Month | N_all | N_active | Active % | SWE>0 % | mean SWE (mm) | mean ΔSWE (mm) | std ΔSWE (mm) | Δ>0 % | Δ<0 % | Δ=0 % |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in monthly_rows:
        md_lines.append(
            f"| {row['month']} | {row['N_all']} | {row['N_active']} | {row['active_pct']:.1f} | "
            f"{row['swe_gt0_pct']:.1f} | {row['mean_swe_t']:.2f} | {row['mean_delta']:+.3f} | "
            f"{row['std_delta']:.3f} | {row['delta_gt0_pct']:.1f} | {row['delta_lt0_pct']:.1f} | {row['delta_eq0_pct']:.1f} |"
        )
    md_lines.append("")
    md_lines.append("## Snow-onset timing across 120 water years")
    md_lines.append("")
    md_lines.append("### First date with SWE > 0")
    md_lines.append("")
    md_lines.append(f"- water years with an onset: {onset_summary_gt0['n_with_onset']} / {n_wy} (never-onset: {onset_summary_gt0['n_never_onset']})")
    if onset_summary_gt0["n_with_onset"]:
        r = onset_summary_gt0["readable_dates"]
        md_lines.append(f"- earliest: {r['earliest']}, p5: {r['p5']}, p25: {r['p25']}, median: {r['median']}, p75: {r['p75']}, p95: {r['p95']}, latest: {r['latest']}")
        c = onset_summary_gt0["onset_month_counts"]
        md_lines.append(f"- onset month counts: Sep={c['Sep']}, Oct={c['Oct']}, Nov={c['Nov']}, Dec={c['Dec']}, Jan-or-later={c['Jan_or_later']}")
    md_lines.append("")
    md_lines.append("### First date with SWE > 1 mm")
    md_lines.append("")
    md_lines.append(f"- water years with an onset: {onset_summary_gt1['n_with_onset']} / {n_wy} (never-onset: {onset_summary_gt1['n_never_onset']})")
    if onset_summary_gt1["n_with_onset"]:
        r = onset_summary_gt1["readable_dates"]
        md_lines.append(f"- earliest: {r['earliest']}, p5: {r['p5']}, p25: {r['p25']}, median: {r['median']}, p75: {r['p75']}, p95: {r['p95']}, latest: {r['latest']}")
        c = onset_summary_gt1["onset_month_counts"]
        md_lines.append(f"- onset month counts: Sep={c['Sep']}, Oct={c['Oct']}, Nov={c['Nov']}, Dec={c['Dec']}, Jan-or-later={c['Jan_or_later']}")
    md_lines.append("")
    md_lines.append("## Candidate training-start comparison (all end Apr 1)")
    md_lines.append("")
    md_lines.append("| Candidate | N_all | N_active | N_zero_to_zero | active_fraction |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for label, res in candidate_results.items():
        md_lines.append(f"| {label} | {res['N_all']} | {res['N_active']} | {res['N_zero_to_zero']} | {res['active_fraction']:.4f} |")
    md_lines.append("")
    md_lines.append("## Output files")
    md_lines.append("")
    md_lines.append(f"- `{OUT_CSV.relative_to(PROJECT_ROOT)}`")
    md_lines.append(f"- `{OUT_MD.relative_to(PROJECT_ROOT)}` (this file)")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"\nwrote {OUT_MD}", flush=True)

    # Save everything needed for the final terminal answer
    final = {
        "monthly_rows": monthly_rows,
        "onset_summary_gt0": onset_summary_gt0,
        "onset_summary_gt1": onset_summary_gt1,
        "candidate_results": candidate_results,
    }
    (PROJECT_ROOT / "artifacts" / "taiesm1_daily_swe_training_month_diagnostic_full.json").write_text(json.dumps(final, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
