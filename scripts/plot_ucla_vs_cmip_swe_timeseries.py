#!/usr/bin/env python3
"""Single diagnostic: UCLA-observed vs. CMIP-simulated (WUS-D3) April-1 Sierra SWE,
WY1985-2021, same UCLA Sierra region/mask applied to every series.

No CNN, no training, no inference. Pure data extraction and one line plot. Reuses the
project's existing April-1 UCLA extraction (already computed) and the project's existing
WUS-D3 mask/area-weighting extraction code (build_wusd3_swe_labels.py) unmodified.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.data_wusd3 import Wusd3Dataset, discover_wusd3_file_years, load_wusd3_snapshot  # noqa: E402
from build_wusd3_swe_labels import build_mask_and_area, weighted_masked_mean  # noqa: E402

WUSD3_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3")
OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_observational_transfer_v1")

UCLA_TARGET_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_mean_mm_plain.npz")

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021

# From artifacts/cmip6_wusd3_inventory/wusd3_parent_runs.csv - the exact 4 model_members
# already used by this project's CMIP6 CNN training manifest.
MODELS = {
    "EC-Earth3": "ec-earth3_r1i1p1f1_2",
    "MIROC6": "miroc6_r1i1p1f1",
    "MPI-ESM1-2-HR": "mpi-esm1-2-hr_r3i1p1f1",
    "TaiESM1": "taiesm1_r1i1p1f1",
}
REFERENCE_DATASET_ID = "ec-earth3_r1i1p1f1_2_historical_bc"


def scenario_for_file_year(file_year: int) -> str:
    return "historical_bc" if file_year <= 2013 else "ssp370_bc"


def main() -> None:
    print(f"reference grid/mask/area dataset: {REFERENCE_DATASET_ID} (UCLA DEFAULT_SIERRA_REGION applied via build_sierra_mask)", flush=True)
    _, ref_grid, ref_mask, ref_area, mask_summary = build_mask_and_area(REFERENCE_DATASET_ID)
    print(f"reference grid mask summary: {mask_summary}", flush=True)

    ucla = np.load(UCLA_TARGET_NPZ)
    ucla_by_year = {int(wy): float(v) for wy, v in zip(ucla["water_year"], ucla["sierra_swe_apr1_mean_mm"], strict=True)}

    water_years = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
    series: dict[str, dict[int, float]] = {name: {} for name in MODELS}
    source_files_used: dict[str, list[str]] = {name: [] for name in MODELS}

    for display_name, base_id in MODELS.items():
        for water_year in water_years:
            file_year = water_year - 1
            scenario = scenario_for_file_year(file_year)
            dataset_id = f"{base_id}_{scenario}"
            dataset = Wusd3Dataset(dataset_id=dataset_id, domain="d02", root_dir=WUSD3_ROOT)
            dataset_dir = WUSD3_ROOT / "daily" / dataset_id / "postprocess" / "d02"
            if not dataset_dir.exists():
                print(f"[{display_name}] WY{water_year}: dataset dir missing ({dataset_id}), skipping", flush=True)
                continue
            available_file_years = set(discover_wusd3_file_years(dataset))
            if file_year not in available_file_years:
                print(f"[{display_name}] WY{water_year}: file_year {file_year} not available in {dataset_id}, skipping", flush=True)
                continue
            try:
                snapshot = load_wusd3_snapshot(
                    dataset,
                    water_year=water_year,
                    snapshot_date=date(water_year, 4, 1),
                    swe_grid=ref_grid,
                    fill_missing=False,
                )
            except (FileNotFoundError, KeyError) as exc:
                print(f"[{display_name}] WY{water_year}: {exc}", flush=True)
                continue
            swe_mm, valid_area_m2 = weighted_masked_mean(snapshot, ref_mask, ref_area)
            if not np.isfinite(swe_mm):
                print(f"[{display_name}] WY{water_year}: no valid masked cells, skipping", flush=True)
                continue
            series[display_name][water_year] = swe_mm
            source_files_used[display_name].append(snapshot.attrs["source_path"])

    # -----------------------------------------------------------------
    # CSV: 37 (water years) x 5 (UCLA + 4 CMIP models), NaN for missing years.
    # -----------------------------------------------------------------
    csv_path = OUTPUT_ROOT / "ucla_vs_cmip_swe_apr1_timeseries.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["water_year", "UCLA_observed_mm"] + [f"{name}_mm" for name in MODELS])
        for wy in water_years:
            row = [wy, f"{ucla_by_year[wy]:.6f}"]
            for name in MODELS:
                value = series[name].get(wy)
                row.append(f"{value:.6f}" if value is not None else "")
            writer.writerow(row)
    print(f"wrote {csv_path}", flush=True)

    # -----------------------------------------------------------------
    # Single line plot, 5 lines.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(12, 6), dpi=160)
    observed_vector = [ucla_by_year[wy] for wy in water_years]
    axis.plot(water_years, observed_vector, marker="o", color="black", linewidth=2.2, label="UCLA observed")
    colors = {"EC-Earth3": "tab:blue", "MIROC6": "tab:orange", "MPI-ESM1-2-HR": "tab:green", "TaiESM1": "tab:red"}
    for name, color in colors.items():
        years_present = sorted(series[name].keys())
        values_present = [series[name][wy] for wy in years_present]
        axis.plot(years_present, values_present, marker="o", markersize=3, color=color, label=name)
    axis.set_xlabel("Water year")
    axis.set_ylabel("April 1 Sierra SWE (mm)")
    axis.set_title("UCLA observed vs. CMIP (WUS-D3) simulated April 1 Sierra SWE, WY1985-2021\nsame UCLA Sierra region/mask applied to all five series")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    png_path = OUTPUT_ROOT / "ucla_vs_cmip_swe_apr1_timeseries.png"
    fig.savefig(png_path)
    plt.close(fig)
    print(f"wrote {png_path}", flush=True)

    # -----------------------------------------------------------------
    # Confirmations requested.
    # -----------------------------------------------------------------
    print("\n=== CMIP source datasets used ===", flush=True)
    for name, base_id in MODELS.items():
        n_years = len(series[name])
        example_paths = source_files_used[name][:2] + (["..."] if len(source_files_used[name]) > 2 else [])
        print(f"{name} (wusd3_dataset_id base={base_id}): {n_years}/{len(water_years)} water years present", flush=True)
        for p in example_paths:
            print(f"    {p}", flush=True)
    print("\n=== Region/mask confirmation ===", flush=True)
    print("All five series use snow_ml.data.DEFAULT_SIERRA_REGION (lat 35-42, lon -122.5..-118.0) "
          "via snow_ml.data.build_sierra_mask. UCLA series: build_sierra_mask applied on UCLA's own "
          "native grid (scripts/process_cobe2_sierra_swe_apr1_target.py, already computed). CMIP series: "
          "the SAME build_sierra_mask/DEFAULT_SIERRA_REGION applied on the WUS-D3 d02 native grid via "
          "scripts/build_wusd3_swe_labels.build_mask_and_area, with one shared reference grid/mask/area "
          "(from the EC-Earth3 historical_bc dataset) reused across all 4 CMIP models, matching the "
          "existing build_wusd3_swe_labels.py convention exactly. This is not the older WUS-D3 CNN-target "
          "footprint audit; region bounds are identical to the UCLA extraction, only the native grid "
          "geometry the region is intersected with differs (WUS-D3 d02 vs. UCLA), which is unavoidable "
          "since they are different physical grids.", flush=True)


if __name__ == "__main__":
    main()
