#!/usr/bin/env python3
"""
Audit COBE2 Case A vs Case B LOYO setups using saved repo artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NESTED_FOLD_MODES_CSV = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/loyo_lod_analysis/cobe2_sierra_swe_lod_loyo_fold_modes.csv"
)
NESTED_PREDICTIONS_CSV = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_mode_subset_diagnostic" / "loyo_mode_subset_predictions.csv"
NESTED_METRICS_CSV = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_mode_subset_diagnostic" / "loyo_mode_subset_metrics.csv"
NESTED_SUMMARY_JSON = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_mode_subset_diagnostic" / "loyo_mode_subset_summary.json"
SIGN_DIAG_SUMMARY_JSON = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_first3_sign_diagnostic" / "loyo_first3_sign_diagnostic_summary.json"
SIGN_DIAG_SELECTED_CSV = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "loyo_first3_sign_diagnostic" / "loyo_first3_selected_mode_metadata_by_fold.csv"
DIR_SUMMARY_JSON = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "cobe2_loyo_directional_diagnostics" / "cobe2_loyo_mode_subset_directional_summary.json"

FIXED_SUMMARY_JSON = PROJECT_ROOT / "artifacts" / "fixed_lod_pair_ols_loyo_check" / "cobe2" / "cobe2_fixed_lod_pairs_loyo_summary.json"
FIXED_SUMMARY_MD = PROJECT_ROOT / "artifacts" / "fixed_lod_pair_ols_loyo_check" / "cobe2" / "cobe2_fixed_lod_pairs_loyo_summary.md"

NESTED_AUDIT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "nested_loyo_mode_stability_audit"
CASE_COMPARE_DIR = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "caseA_caseB_loyo_audit"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def build_nested_selected_modes_table() -> tuple[list[dict[str, object]], dict[str, object]]:
    fold_rows = read_csv_rows(NESTED_FOLD_MODES_CSV)
    pred_rows = read_csv_rows(NESTED_PREDICTIONS_CSV)
    pred_map = {
        (row["model_name"], int(row["heldout_wy"])): row
        for row in pred_rows
    }
    selected_rows: list[dict[str, object]] = []
    mode_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    lag_counts: dict[tuple[int, str], int] = {}
    region_counts: dict[tuple[int, str, str], int] = {}

    for row in fold_rows:
        if row["selected"] != "True":
            continue
        mode = int(row["mode_number"])
        if mode > 3:
            continue
        heldout_wy = int(row["held_out_water_year"])
        pred_row = pred_map[("M1_M2_M3", heldout_wy)]
        out = {
            "heldout_wy": heldout_wy,
            "mode": mode,
            "lag_month": row["lag_month"],
            "lat": float(row["latitude"]),
            "lon": float(row["longitude_0_360"]),
            "selected_corr": float(row["corr_with_residual"]),
            "delta_R2": float(row["delta_r2"]),
            "cumulative_R2": float(row["cumulative_r2"]),
            "beta": float(row["beta"]),
            "pred_swe": float(pred_row["pred_swe"]),
            "obs_swe": float(pred_row["obs_swe"]),
        }
        selected_rows.append(out)
        mode_counts[mode] += 1
        lag_counts[(mode, str(row["lag_month"]))] = lag_counts.get((mode, str(row["lag_month"])), 0) + 1
        region_key = (mode, row["lag_month"], f"{float(row['latitude']):.2f},{float(row['longitude_0_360']):.2f}")
        region_counts[region_key] = region_counts.get(region_key, 0) + 1

    summary = {
        "source_fold_modes_csv": str(NESTED_FOLD_MODES_CSV),
        "source_predictions_csv": str(NESTED_PREDICTIONS_CSV),
        "row_count": len(selected_rows),
        "heldout_year_count": len({row["heldout_wy"] for row in selected_rows}),
        "mode_counts": mode_counts,
        "top_repeated_locations_by_mode": {},
    }
    for mode in (1, 2, 3):
        candidates = [
            {"lag_month": lag, "latlon": latlon, "count": count}
            for (m, lag, latlon), count in region_counts.items()
            if m == mode
        ]
        candidates.sort(key=lambda item: item["count"], reverse=True)
        summary["top_repeated_locations_by_mode"][f"mode_{mode}"] = candidates[:5]
    return selected_rows, summary


def build_case_comparison() -> tuple[list[dict[str, object]], dict[str, object]]:
    fixed = read_json(FIXED_SUMMARY_JSON)
    nested_metrics = {row["model_name"]: row for row in read_csv_rows(NESTED_METRICS_CSV)}

    rows = [
        {
            "case": "A_like_fixed_fullsample_pairs",
            "script": "scripts/run_fixed_selected_lod_pairs_loyo_check.py",
            "output_dir": str(FIXED_SUMMARY_JSON.parent),
            "mode_selection_inside_loyo": False,
            "r": float(fixed["metrics"]["corr"]),
            "R2": float(fixed["metrics"]["r2"]),
            "RMSE": float(fixed["metrics"]["rmse"]),
            "MAE": float(fixed["metrics"]["mae"]),
            "notes": "Full-sample-selected LOD position/month pairs fixed before LOYO; uses 6 fixed pairs, so this is Case A-like but not the exact same first-3 nested subset pipeline.",
        },
        {
            "case": "B_nested_first3_subset_current_bad",
            "script": "scripts/run_cobe2_loyo_mode_subset_diagnostic.py",
            "output_dir": str(NESTED_SUMMARY_JSON.parent),
            "mode_selection_inside_loyo": True,
            "r": float(nested_metrics["M1_M2_M3"]["r"]),
            "R2": float(nested_metrics["M1_M2_M3"]["R2"]),
            "RMSE": float(nested_metrics["M1_M2_M3"]["RMSE"]),
            "MAE": float(nested_metrics["M1_M2_M3"]["MAE"]),
            "notes": "Saved per-fold selected modes from nested COBE2 LOYO LOD; metrics shown for the current bad M1_M2_M3 subset diagnostic.",
        },
    ]
    summary = {
        "comparison_note": "Case A row is the closest fixed-selection COBE2 analogue found in the repo. It fixes 6 full-sample-selected LOD pairs, not the exact nested first-3 orthogonal-mode subset used by the current bad diagnostics.",
        "rows": rows,
    }
    return rows, summary


def main() -> None:
    nested_selected_rows, nested_summary = build_nested_selected_modes_table()
    NESTED_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    nested_csv = NESTED_AUDIT_DIR / "nested_loyo_selected_modes_by_fold.csv"
    nested_json = NESTED_AUDIT_DIR / "nested_loyo_mode_stability_summary.json"
    write_csv(
        nested_csv,
        ["heldout_wy", "mode", "lag_month", "lat", "lon", "selected_corr", "delta_R2", "cumulative_R2", "beta", "pred_swe", "obs_swe"],
        nested_selected_rows,
    )
    nested_json.write_text(json.dumps(nested_summary, indent=2) + "\n", encoding="utf-8")

    compare_rows, compare_summary = build_case_comparison()
    CASE_COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    compare_csv = CASE_COMPARE_DIR / "caseA_caseB_loyo_comparison.csv"
    compare_json = CASE_COMPARE_DIR / "caseA_caseB_loyo_audit_summary.json"
    write_csv(
        compare_csv,
        ["case", "script", "output_dir", "mode_selection_inside_loyo", "r", "R2", "RMSE", "MAE", "notes"],
        compare_rows,
    )
    compare_json.write_text(json.dumps(compare_summary, indent=2) + "\n", encoding="utf-8")

    print(f"nested audit CSV: {nested_csv}")
    print(f"nested audit JSON: {nested_json}")
    print(f"case compare CSV: {compare_csv}")
    print(f"case compare JSON: {compare_json}")


if __name__ == "__main__":
    main()
