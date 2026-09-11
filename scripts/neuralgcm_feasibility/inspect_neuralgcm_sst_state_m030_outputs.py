#!/usr/bin/env python3
"""Inspect completed NeuralGCM WY2021 SST-state ensemble outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
OUTPUT_BASE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs")

JSON_OUT = REPORT_DIR / "neuralgcm_wy2021_sst_state_m030_output_inspection.json"
TXT_OUT = REPORT_DIR / "neuralgcm_wy2021_sst_state_m030_output_inspection.txt"

STATES = ("actual", "reversed_pacific")
VARIABLES = [
    "precipitation_cumulative_mean",
    "evaporation",
    "temperature_850hPa",
    "u_component_of_wind_850hPa",
    "v_component_of_wind_850hPa",
    "specific_humidity_850hPa",
]


def stats(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    return {
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "min": float(np.nanmin(values)) if finite.any() else None,
        "max": float(np.nanmax(values)) if finite.any() else None,
        "mean": float(np.nanmean(values)) if finite.any() else None,
    }


def main() -> None:
    report = {"states": {}}
    lines = ["NeuralGCM WY2021 SST-state M=30 output inspection", ""]

    for state in STATES:
        state_root = OUTPUT_BASE / f"wy2021_{state}_m030"
        state_report = {"output_root": str(state_root), "members": {}}
        lines.extend([f"{state}:", f"  output_root: {state_root}"])
        for member_dir in sorted(state_root.glob("member_*")):
            output_file = member_dir / "predictions_6h.nc"
            if not output_file.exists():
                continue
            with xr.open_dataset(output_file) as ds:
                member_report = {
                    "path": str(output_file),
                    "size_bytes": int(output_file.stat().st_size),
                    "dims": {k: int(v) for k, v in ds.sizes.items()},
                    "coords": list(ds.coords),
                    "time_start": str(np.asarray(ds.time.values[0]).item()),
                    "time_end": str(np.asarray(ds.time.values[-1]).item()),
                    "time_count": int(ds.sizes["time"]),
                    "variables": {},
                }
                for name in VARIABLES:
                    if name not in ds:
                        member_report["variables"][name] = {"missing": True}
                        continue
                    da = ds[name]
                    member_report["variables"][name] = {
                        "dims": list(da.dims),
                        "shape": [int(v) for v in da.shape],
                        "stats": stats(np.asarray(da.values)),
                    }
                state_report["members"][member_dir.name] = member_report
        report["states"][state] = state_report
        lines.append(f"  completed_members: {len(state_report['members'])}")

    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
