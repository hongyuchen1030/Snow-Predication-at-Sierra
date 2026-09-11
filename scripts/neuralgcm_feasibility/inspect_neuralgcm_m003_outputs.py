#!/usr/bin/env python3
"""Inspect NeuralGCM M=3 stochastic ensemble outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
OUTPUT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs/wy2021_actual_m003")
JSON_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_output_inspection.json"
TXT_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_m003_output_inspection.txt"

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
    report = {"output_root": str(OUTPUT_ROOT), "members": {}}
    lines = ["NeuralGCM WY2021 actual M=3 output inspection", f"output_root: {OUTPUT_ROOT}", ""]

    for member_dir in sorted(OUTPUT_ROOT.glob("member_*")):
        output_file = member_dir / "predictions_6h.nc"
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
            lines.extend(
                [
                    f"{member_dir.name}:",
                    f"  path: {output_file}",
                    f"  dims: {member_report['dims']}",
                    f"  time_start: {member_report['time_start']}",
                    f"  time_end: {member_report['time_end']}",
                ]
            )
            for name in VARIABLES:
                if name not in ds:
                    member_report["variables"][name] = {"missing": True}
                    lines.append(f"  {name}: missing")
                    continue
                da = ds[name]
                variable_stats = stats(np.asarray(da.values))
                member_report["variables"][name] = {
                    "dims": list(da.dims),
                    "shape": [int(v) for v in da.shape],
                    "stats": variable_stats,
                }
                lines.append(f"  {name}: {variable_stats}")
            report["members"][member_dir.name] = member_report
            lines.append("")

    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
