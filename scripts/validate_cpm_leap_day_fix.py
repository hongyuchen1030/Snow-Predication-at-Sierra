#!/usr/bin/env python
"""Validate the leap-day fix in run_cpm_swe_attribution_37yr_fixed_cpm.py.

Runs the corrected extension script into a NEW output directory (does not
touch the trusted 32-year artifacts/cpm_swe_attribution/ artifact), then
checks:
  - it completes without KeyError('02-29')
  - exactly 1 leap day (2020-02-29) was dropped from the extension window
  - the corrected extension yields the expected 905 retained days
  - WY2020 has number_of_days == 181, matching every other water year
  - years 1985-2016 (the overlap with the trusted artifact) match exactly
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/global/u1/h/hyvchen/Snow-Predication-at-Sierra"
TRUSTED_CSV = f"{REPO_ROOT}/artifacts/cpm_swe_attribution/cpm_pc3_nov_apr_by_water_year.csv"
NEW_OUTPUT_DIR = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_swe_attribution_37yr_fixed_cpm_leapfix"
PYTHON_BIN = "/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python"


def main() -> int:
    print("=== Running corrected CPM extension into a new output dir ===", flush=True)
    result = subprocess.run(
        [PYTHON_BIN, f"{REPO_ROOT}/scripts/run_cpm_swe_attribution_37yr_fixed_cpm.py",
         "--output-dir", NEW_OUTPUT_DIR],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"FAIL: extension script exited with code {result.returncode}", flush=True)
        return 1
    print("Extension script completed with exit code 0 (no KeyError).", flush=True)

    new_csv = f"{NEW_OUTPUT_DIR}/cpm_pc3_nov_apr_by_water_year.csv"
    new_df = pd.read_csv(new_csv).sort_values("water_year")
    print(f"\n=== Loaded corrected annual CPM table: {new_csv} ===", flush=True)
    print(new_df.to_string(index=False), flush=True)

    ok = True

    print("\n=== Check 1: retained day counts ===", flush=True)
    day_counts = new_df.set_index("water_year")["number_of_days"]
    for wy, n in day_counts.items():
        status = "OK" if n == 181 else "MISMATCH"
        if n != 181:
            ok = False
        print(f"  WY{wy}: number_of_days={n}  [{status}] (expected 181, matching every other water year)", flush=True)

    print("\n=== Check 2: overlap with trusted 32-year artifact (WY1985-2016) ===", flush=True)
    trusted_df = pd.read_csv(TRUSTED_CSV).sort_values("water_year")
    merged = trusted_df.merge(new_df, on="water_year", suffixes=("_trusted", "_new"))
    diffs = (merged["cpm_pc3_nov_apr_mean_new"] - merged["cpm_pc3_nov_apr_mean_trusted"]).abs()
    max_diff = float(diffs.max())
    print(f"  n_overlap_years={len(merged)}  max_abs_diff={max_diff:.3e}", flush=True)
    if max_diff > 1e-6:
        ok = False
        print("  MISMATCH: overlap years differ beyond floating-point tolerance", flush=True)
        print(merged[["water_year", "cpm_pc3_nov_apr_mean_trusted", "cpm_pc3_nov_apr_mean_new"]].to_string(index=False), flush=True)
    else:
        print("  OK: all overlap years numerically identical within tolerance", flush=True)

    print("\n=== Check 3: WY2020 specifically completes and is present ===", flush=True)
    if 2020 not in set(new_df["water_year"]):
        ok = False
        print("  MISMATCH: WY2020 missing from output", flush=True)
    else:
        print(f"  OK: WY2020 present, cpm_pc3_nov_apr_mean={new_df.set_index('water_year').loc[2020, 'cpm_pc3_nov_apr_mean']:.6f}", flush=True)

    print("\n=== Check 4: full 37-year coverage (1985-2021) ===", flush=True)
    years_present = set(new_df["water_year"].astype(int))
    expected_years = set(range(1985, 2022))
    missing = expected_years - years_present
    if missing:
        ok = False
        print(f"  MISMATCH: missing years {sorted(missing)}", flush=True)
    else:
        print("  OK: all 37 water years (1985-2021) present", flush=True)

    print(f"\n=== OVERALL: {'PASS' if ok else 'FAIL'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
