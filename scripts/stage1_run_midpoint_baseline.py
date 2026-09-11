"""
Run the continuous full-period (WY1985 spinup through WY2021) Snow17 simulation
with the FIXED MIDPOINT parameter vector (same for all 3 regions), for use as the
calibration-diagnostics baseline comparison (per Stage 1 spec: "compare against
the midpoint/default smoke-test parameter vector"). Does NOT touch held-out SWE
for any parameter decision -- midpoint is a fixed, pre-registered baseline, not fit.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from stage1_pipeline_common import (
    REGIONS, MIDPOINT_PARAMS, run_region_continuous, whole_sierra_weight,
    combine_whole_sierra, OUT_DIR, load_region_data,
)

def main():
    t0 = time.time()
    regional_series = {}
    data_by_region = {}
    for region in REGIONS:
        print(f"[midpoint] running {region}...", flush=True)
        tidx, swe, data = run_region_continuous(region, MIDPOINT_PARAMS)
        regional_series[region] = pd.Series(swe, index=tidx)
        data_by_region[region] = data
        print(f"[midpoint] {region} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    weights = whole_sierra_weight(data_by_region)
    whole = combine_whole_sierra(regional_series, weights)

    df = pd.DataFrame(regional_series)
    df["Whole_Sierra"] = whole
    out_csv = OUT_DIR / "calibration" / "predicted_regional_swe_midpoint_wy1985_2021.csv"
    df.to_csv(out_csv)
    print(f"Saved {out_csv} ({len(df)} rows)", flush=True)
    (OUT_DIR / "calibration" / "whole_sierra_weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    print(f"whole_sierra_weights = {weights}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
