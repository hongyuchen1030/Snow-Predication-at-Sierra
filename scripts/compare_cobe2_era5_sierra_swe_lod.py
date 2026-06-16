#!/usr/bin/env python3
"""
Compare the existing COBE2 Sierra SWE LOD diagnostic against the ERA5 SST run.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


COBE2_SUMMARY = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json")
ERA5_SUMMARY = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup/lod_analysis/era5_sierra_swe_lod_summary.json")
OUTPUT_ROOT = Path(
    os.environ.get(
        "COBE2_ERA5_SIERRA_SWE_LOD_COMPARISON_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_era5_sst_sierra_swe_lod_comparison",
    )
).expanduser()


def load_selected_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    rows = [row for row in payload["lod_rows"] if row.get("selected")]
    return rows


def broad_region(lat: float, lon: float) -> str:
    if lat < 10.0:
        lat_band = "tropical"
    elif lat < 30.0:
        lat_band = "subtropical"
    else:
        lat_band = "midlatitude"

    if lon < 170.0:
        lon_band = "western"
    elif lon < 230.0:
        lon_band = "central"
    else:
        lon_band = "eastern"
    return f"{lat_band}_{lon_band}_pacific"


def distance_degrees(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    return math.sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2)


def row_value(row: dict[str, object], preferred: str, fallback: str) -> float:
    if preferred in row:
        return float(row[preferred])
    return float(row[fallback])


def extract_curve(rows: list[dict[str, object]], preferred: str, fallback: str) -> list[float]:
    return [row_value(row, preferred, fallback) for row in rows]


def compare_rows(cobe2_rows: list[dict[str, object]], era5_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    max_modes = max(len(cobe2_rows), len(era5_rows))
    comparison_rows: list[dict[str, object]] = []
    for mode_number in range(1, max_modes + 1):
        cobe2 = cobe2_rows[mode_number - 1] if mode_number <= len(cobe2_rows) else None
        era5 = era5_rows[mode_number - 1] if mode_number <= len(era5_rows) else None

        row: dict[str, object] = {"mode_id": mode_number}
        if cobe2 is not None:
            row.update(
                {
                    "cobe2_lag_month": cobe2.get("lag_month"),
                    "cobe2_latitude": cobe2.get("latitude"),
                    "cobe2_longitude_0_360": cobe2.get("longitude_0_360", cobe2.get("longitude")),
                    "cobe2_broad_region": broad_region(float(cobe2["latitude"]), float(cobe2.get("longitude_0_360", cobe2["longitude"]))),
                    "cobe2_delta_R2": row_value(cobe2, "delta_R2", "delta_r2"),
                    "cobe2_cumulative_R2": row_value(cobe2, "cumulative_R2", "cumulative_r2"),
                }
            )
        if era5 is not None:
            row.update(
                {
                    "era5_lag_month": era5.get("lag_month"),
                    "era5_latitude": era5.get("latitude"),
                    "era5_longitude_0_360": era5.get("longitude_0_360", era5.get("longitude")),
                    "era5_broad_region": broad_region(float(era5["latitude"]), float(era5.get("longitude_0_360", era5["longitude"]))),
                    "era5_delta_R2": row_value(era5, "delta_R2", "delta_r2"),
                    "era5_cumulative_R2": row_value(era5, "cumulative_R2", "cumulative_r2"),
                }
            )

        if cobe2 is not None and era5 is not None:
            row["same_lag_month"] = bool(cobe2.get("lag_month") == era5.get("lag_month"))
            row["same_broad_region"] = bool(row["cobe2_broad_region"] == row["era5_broad_region"])
            row["location_distance_degrees"] = distance_degrees(
                float(cobe2["latitude"]),
                float(cobe2.get("longitude_0_360", cobe2["longitude"])),
                float(era5["latitude"]),
                float(era5.get("longitude_0_360", era5["longitude"])),
            )
        comparison_rows.append(row)
    return comparison_rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cobe2_rows = load_selected_rows(COBE2_SUMMARY)
    era5_rows = load_selected_rows(ERA5_SUMMARY)
    comparison_rows = compare_rows(cobe2_rows, era5_rows)

    comparison_csv = OUTPUT_ROOT / "cobe2_era5_lod_mode_comparison.csv"
    fieldnames = [
        "mode_id",
        "cobe2_lag_month",
        "cobe2_latitude",
        "cobe2_longitude_0_360",
        "cobe2_broad_region",
        "cobe2_delta_R2",
        "cobe2_cumulative_R2",
        "era5_lag_month",
        "era5_latitude",
        "era5_longitude_0_360",
        "era5_broad_region",
        "era5_delta_R2",
        "era5_cumulative_R2",
        "same_lag_month",
        "same_broad_region",
        "location_distance_degrees",
    ]
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    matched_first3 = [row for row in comparison_rows[:3] if row.get("same_broad_region")]
    first3_cobe2_delta = extract_curve(cobe2_rows[:3], "delta_R2", "delta_r2")
    first3_era5_delta = extract_curve(era5_rows[:3], "delta_R2", "delta_r2")
    later_cobe2_delta = extract_curve(cobe2_rows[3:], "delta_R2", "delta_r2")
    later_era5_delta = extract_curve(era5_rows[3:], "delta_R2", "delta_r2")

    summary = {
        "cobe2_summary_path": str(COBE2_SUMMARY),
        "era5_summary_path": str(ERA5_SUMMARY),
        "comparison_csv": str(comparison_csv),
        "cobe2_selected_lag_months": [row["lag_month"] for row in cobe2_rows],
        "era5_selected_lag_months": [row["lag_month"] for row in era5_rows],
        "cobe2_selected_locations": [
            {"mode_id": int(row.get("mode_id", row["mode_number"])), "latitude": float(row["latitude"]), "longitude_0_360": float(row.get("longitude_0_360", row["longitude"]))}
            for row in cobe2_rows
        ],
        "era5_selected_locations": [
            {"mode_id": int(row.get("mode_id", row["mode_number"])), "latitude": float(row["latitude"]), "longitude_0_360": float(row.get("longitude_0_360", row["longitude"]))}
            for row in era5_rows
        ],
        "mode_pairs": comparison_rows,
        "cobe2_delta_R2_curve": extract_curve(cobe2_rows, "delta_R2", "delta_r2"),
        "era5_delta_R2_curve": extract_curve(era5_rows, "delta_R2", "delta_r2"),
        "cobe2_cumulative_R2_curve": extract_curve(cobe2_rows, "cumulative_R2", "cumulative_r2"),
        "era5_cumulative_R2_curve": extract_curve(era5_rows, "cumulative_R2", "cumulative_r2"),
        "leading_mode_similarity": {
            "first_three_same_broad_region_count": len(matched_first3),
            "first_three_same_broad_region_mode_ids": [int(row["mode_id"]) for row in matched_first3],
            "first_three_more_stable_than_later_by_delta_R2": {
                "cobe2": bool(np.mean(first3_cobe2_delta) > np.mean(later_cobe2_delta)) if later_cobe2_delta else True,
                "era5": bool(np.mean(first3_era5_delta) > np.mean(later_era5_delta)) if later_era5_delta else True,
            },
        },
        "interpretation": {
            "leading_modes_similar_broad_regions": len(matched_first3) >= 1,
            "leading_modes_more_stable_than_later_modes": bool(
                (not later_cobe2_delta or np.mean(first3_cobe2_delta) > np.mean(later_cobe2_delta))
                and (not later_era5_delta or np.mean(first3_era5_delta) > np.mean(later_era5_delta))
            ),
            "notes": [
                "Broad-region similarity uses a coarse Pacific binning: tropical/subtropical/midlatitude crossed with western/central/eastern longitude sectors.",
                "Stability is summarized here by whether the mean delta_R2 over modes 1-3 exceeds the mean delta_R2 over later retained modes in each dataset.",
            ],
        },
    }

    summary_path = OUTPUT_ROOT / "cobe2_era5_lod_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison_csv": str(comparison_csv), "summary_json": str(summary_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
