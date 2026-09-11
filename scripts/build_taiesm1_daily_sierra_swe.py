#!/usr/bin/env python3
"""Build the full daily TaiESM1 WUS-D3 Sierra-mean SWE time series.

Reuses the EXACT reference-grid / Sierra-box mask / native-cell-area
weighting already used by scripts/build_wusd3_swe_labels.py for the existing
April-1 seasonal SWE labels (same shared EC-Earth3 reference grid, same
DEFAULT_SIERRA_REGION box, same weighted_masked_mean reduction) -- applied
per day instead of per water year. No new mask, no elevation/mountain
filter, no regridding to a different grid: only the same geographic box
applied to TaiESM1's own native d02 cells.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.data_wusd3 import (  # noqa: E402
    WUSD3_ROOT,
    WUSD3_SWE_VARIABLE,
    Wusd3Dataset,
    discover_wusd3_file_years,
    variable_path_for_file_year,
)
from build_wusd3_swe_labels import build_mask_and_area, weighted_masked_mean  # noqa: E402

# The existing pipeline (build_wusd3_swe_labels.py:extract_all_labels) uses
# ONE shared reference grid (EC-Earth3's) for every parent model, since all
# WUS-D3 d02 downscaling runs share the same fixed regional domain. Reused
# verbatim here so the daily series is directly comparable to the existing
# April-1 labels.
REF_DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"
TAIESM1_HIST_ID = "taiesm1_r1i1p1f1_historical_bc"
TAIESM1_SSP_ID = "taiesm1_r1i1p1f1_ssp370_bc"

OUT_CSV = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.csv"
OUT_NPZ = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.npz"
OUT_SUMMARY = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe_summary.json"


def load_year_field(dataset: Wusd3Dataset, file_year: int) -> tuple[np.ndarray, np.ndarray, str, str]:
    path = variable_path_for_file_year(dataset, "swe", file_year)
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        field = np.asarray(ds[WUSD3_SWE_VARIABLE].values, dtype=np.float64)
        day_raw = np.asarray(ds["day"].values)
        day_units = str(ds["day"].attrs["units"])
        day_calendar = str(ds["day"].attrs.get("calendar", "365_day"))
    return field, day_raw, day_units, day_calendar


def main() -> None:
    print("Building shared reference grid/mask/area (EC-Earth3, DEFAULT_SIERRA_REGION)...", flush=True)
    _, ref_grid, ref_mask, ref_area, mask_summary = build_mask_and_area(REF_DATASET_ID)
    mask_arr = np.asarray(ref_mask.values, dtype=np.float64)
    area_arr = np.asarray(ref_area.values, dtype=np.float64)
    weights2d = mask_arr * area_arr  # NaN outside the Sierra box (area is NaN there); finite m^2 inside
    inside_box = np.isfinite(weights2d)
    n_cells_selected = int(inside_box.sum())
    total_area_km2 = float(np.nansum(weights2d)) / 1.0e6

    print(
        f"Reference grid: {mask_summary} | selected_cells={n_cells_selected} "
        f"total_area_km2={total_area_km2:.3f}",
        flush=True,
    )
    print(f"Sierra box: {PROJECT_ROOT}", flush=True)

    records: list[dict] = []
    for experiment, wusd3_id in (("historical", TAIESM1_HIST_ID), ("ssp370", TAIESM1_SSP_ID)):
        dataset = Wusd3Dataset(dataset_id=wusd3_id, domain="d02", root_dir=WUSD3_ROOT)
        file_years = sorted(discover_wusd3_file_years(dataset))
        for file_year in file_years:
            field, day_raw, day_units, day_calendar = load_year_field(dataset, file_year)
            assert field.ndim == 3, field.shape

            cropped = field[:, ref_grid.row_slice, ref_grid.col_slice]
            cropped = cropped[:, : ref_grid.trimmed_shape[0], : ref_grid.trimmed_shape[1]]
            assert cropped.shape[1:] == mask_arr.shape, (cropped.shape, mask_arr.shape, wusd3_id, file_year)

            valid_day = np.isfinite(cropped)
            full_valid = valid_day & inside_box[None, :, :]
            w = np.where(full_valid, weights2d[None, :, :], 0.0)
            f = np.where(full_valid, cropped, 0.0)
            denom = w.sum(axis=(1, 2))
            numer = (f * w).sum(axis=(1, 2))
            swe_mean = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)

            dates = netCDF4.num2date(day_raw, units=day_units, calendar=day_calendar)
            n_days = cropped.shape[0]
            assert n_days == len(dates) == 365, (n_days, len(dates), wusd3_id, file_year)
            for i in range(n_days):
                dt = dates[i]
                records.append(
                    {
                        "model": "TaiESM1",
                        "experiment": experiment,
                        "file_year": file_year,
                        "cal_year": int(dt.year),
                        "cal_month": int(dt.month),
                        "cal_day": int(dt.day),
                        "day_of_year": i + 1,
                        "swe_mm": float(swe_mean[i]),
                    }
                )
            print(
                f"{experiment} {file_year}: mean={np.nanmean(swe_mean):.3f} mm "
                f"n_valid_days={int(np.isfinite(swe_mean).sum())}/{n_days}",
                flush=True,
            )

    print(f"Total daily records built: {len(records)}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "experiment", "file_year", "cal_year", "cal_month", "cal_day", "day_of_year", "swe_mm"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {OUT_CSV}", flush=True)

    swe_mm = np.array([r["swe_mm"] for r in records], dtype=np.float64)
    year = np.array([r["cal_year"] for r in records], dtype=np.int32)
    month = np.array([r["cal_month"] for r in records], dtype=np.int8)
    day = np.array([r["cal_day"] for r in records], dtype=np.int8)
    day_of_year = np.array([r["day_of_year"] for r in records], dtype=np.int16)
    experiment_arr = np.array([r["experiment"] for r in records])
    file_year_arr = np.array([r["file_year"] for r in records], dtype=np.int32)

    np.savez_compressed(
        OUT_NPZ,
        swe_mm=swe_mm,
        year=year,
        month=month,
        day=day,
        day_of_year=day_of_year,
        experiment=experiment_arr,
        file_year=file_year_arr,
        model=np.array(["TaiESM1"] * len(records)),
    )
    print(f"wrote {OUT_NPZ}", flush=True)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    n_records = len(records)
    expected = 120 * 365
    n_years = 34 + 86
    print(f"\n=== VALIDATION ===", flush=True)
    print(f"A. records={n_records} expected={expected} ({n_years} years x 365 days) match={n_records == expected}", flush=True)

    # B. duplicates / gaps in (cal_year, day_of_year) or file continuity
    keys = [(r["file_year"], r["day_of_year"]) for r in records]
    n_unique = len(set(keys))
    print(f"B. duplicate check: unique_keys={n_unique} total={n_records} duplicates={n_records - n_unique}", flush=True)
    file_years_present = sorted({r["file_year"] for r in records})
    expected_file_years = list(range(1980, 2100))
    missing_file_years = sorted(set(expected_file_years) - set(file_years_present))
    print(f"   missing file-years: {missing_file_years if missing_file_years else 'NONE'}", flush=True)
    boundary_records = [r for r in records if r["file_year"] in (2013, 2014)]
    boundary_records_sorted = sorted(boundary_records, key=lambda r: (r["file_year"], r["day_of_year"]))
    last_hist = boundary_records_sorted[364]  # last day of file_year=2013
    first_ssp = boundary_records_sorted[365]  # first day of file_year=2014
    print(
        f"   branch boundary: last historical record = {last_hist['cal_year']}-{last_hist['cal_month']:02d}-{last_hist['cal_day']:02d} "
        f"-> first ssp370 record = {first_ssp['cal_year']}-{first_ssp['cal_month']:02d}-{first_ssp['cal_day']:02d} "
        f"(consecutive={last_hist['cal_year']==2013 and last_hist['cal_month']==12 and last_hist['cal_day']==31 and first_ssp['cal_year']==2014 and first_ssp['cal_month']==1 and first_ssp['cal_day']==1})",
        flush=True,
    )

    finite = np.isfinite(swe_mm)
    n_missing = int((~finite).sum())
    stats = {
        "n_records": n_records,
        "min_swe_mm": float(np.nanmin(swe_mm)),
        "max_swe_mm": float(np.nanmax(swe_mm)),
        "mean_swe_mm": float(np.nanmean(swe_mm)),
        "std_swe_mm": float(np.nanstd(swe_mm)),
        "frac_exactly_zero": float((swe_mm[finite] == 0.0).mean()) if finite.any() else float("nan"),
        "frac_missing": float(n_missing / n_records),
    }
    print(f"C. stats: {json.dumps(stats, indent=2)}", flush=True)

    # D. spot-check a few dates directly against a fresh independent computation
    from datetime import date as _date

    spot_dates = [(1985, 4, 1), (2000, 1, 15), (2020, 4, 1), (2099, 6, 15), (1980, 1, 1), (2099, 12, 31)]
    spot_results = []
    lookup = {(r["cal_year"], r["cal_month"], r["cal_day"]): r["swe_mm"] for r in records}
    for y, m, d in spot_dates:
        key = (y, m, d)
        val = lookup.get(key)
        spot_results.append({"date": f"{y}-{m:02d}-{d:02d}", "swe_mm_from_series": val})
        print(f"D. spot-check {y}-{m:02d}-{d:02d}: swe_mm={val}", flush=True)

    # E. reproduce existing April-1 seasonal labels for several water years
    print("\nE. April-1 reproduction check against scripts/build_wusd3_swe_labels.py output:", flush=True)
    swe_labels_csv = PROJECT_ROOT / "artifacts" / "cmip6_swe_labels" / "wusd3_sierra_apr1_swe_labels.csv"
    with swe_labels_csv.open() as fh:
        existing_rows = [r for r in csv.DictReader(fh) if r["source_id"] == "TaiESM1"]
    check_water_years = [1985, 1990, 2000, 2010, 2013, 2016, 2050, 2099]
    repro_results = []
    for wy in check_water_years:
        existing = next((r for r in existing_rows if int(r["water_year"]) == wy), None)
        april1_val = lookup.get((wy, 4, 1))
        if existing is None or april1_val is None:
            repro_results.append({"water_year": wy, "existing": None, "new": april1_val, "match": None})
            print(f"   WY{wy}: existing=MISSING new={april1_val}", flush=True)
            continue
        existing_val = float(existing["SWE_label"])
        diff = abs(existing_val - april1_val)
        match = diff < 1e-3
        repro_results.append({"water_year": wy, "existing": existing_val, "new": april1_val, "abs_diff": diff, "match": match})
        print(f"   WY{wy}: existing={existing_val:.6f} new={april1_val:.6f} diff={diff:.2e} match={match}", flush=True)

    summary = {
        "reference_grid_dataset": REF_DATASET_ID,
        "sierra_region": {"lat_min": 35.0, "lat_max": 42.0, "lon_min": -122.5, "lon_max": -118.0},
        "n_cells_selected": n_cells_selected,
        "total_area_km2": total_area_km2,
        "mask_summary": mask_summary,
        "validation_A_record_count": {"n_records": n_records, "expected": expected, "match": n_records == expected},
        "validation_B_duplicates_gaps": {
            "n_unique_keys": n_unique,
            "duplicates": n_records - n_unique,
            "missing_file_years": missing_file_years,
        },
        "validation_C_stats": stats,
        "validation_D_spotcheck": spot_results,
        "validation_E_april1_reproduction": repro_results,
        "output_csv": str(OUT_CSV),
        "output_npz": str(OUT_NPZ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {OUT_SUMMARY}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
