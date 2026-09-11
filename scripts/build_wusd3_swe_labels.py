#!/usr/bin/env python3
"""Build canonical WUS-D3 d02 April-1 Sierra SWE scalar labels."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data import DEFAULT_SIERRA_REGION, build_sierra_mask  # noqa: E402
from snow_ml.data_wusd3 import (  # noqa: E402
    WUSD3_SWE_VARIABLE,
    Wusd3Dataset,
    discover_wusd3_file_years,
    get_wusd3_grid_definition,
    load_wusd3_snapshot,
    variable_path_for_file_year,
)


ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cmip6_swe_labels"
PARENT_RUNS_CSV = PROJECT_ROOT / "artifacts" / "cmip6_wusd3_inventory" / "wusd3_parent_runs.csv"
AUX_LABELS_CSV = PROJECT_ROOT / "artifacts" / "cmip6_aux_labels" / "cmip6_auxiliary_labels.csv"
OUT_CSV = ARTIFACT_DIR / "wusd3_sierra_apr1_swe_labels.csv"
OUT_FULL_CSV = ARTIFACT_DIR / "wusd3_sierra_apr1_swe_labels_full_available.csv"
OUT_JSON = ARTIFACT_DIR / "wusd3_sierra_apr1_swe_labels_summary.json"
QA_CSV = ARTIFACT_DIR / "wusd3_sierra_apr1_swe_manual_checks.csv"


def area_from_bounds_2d(latitudes: np.ndarray, longitudes: np.ndarray, radius_m: float = 6_371_000.0) -> np.ndarray:
    nlat, nlon = latitudes.shape
    areas = np.full((nlat, nlon), np.nan, dtype=np.float64)
    for i in range(nlat):
        for j in range(nlon):
            center_lat = latitudes[i, j]
            center_lon = longitudes[i, j]
            if not np.isfinite(center_lat) or not np.isfinite(center_lon):
                continue
            if nlat == 1:
                dlat = 0.0
            elif i == 0:
                dlat = abs(latitudes[i + 1, j] - center_lat)
            elif i == nlat - 1:
                dlat = abs(center_lat - latitudes[i - 1, j])
            else:
                dlat = 0.5 * abs(latitudes[i + 1, j] - latitudes[i - 1, j])
            if nlon == 1:
                dlon = 0.0
            elif j == 0:
                dlon = abs(longitudes[i, j + 1] - center_lon)
            elif j == nlon - 1:
                dlon = abs(center_lon - longitudes[i, j - 1])
            else:
                dlon = 0.5 * abs(longitudes[i, j + 1] - longitudes[i, j - 1])
            lat_north = np.deg2rad(center_lat + 0.5 * dlat)
            lat_south = np.deg2rad(center_lat - 0.5 * dlat)
            lon_east = np.deg2rad(center_lon + 0.5 * dlon)
            lon_west = np.deg2rad(center_lon - 0.5 * dlon)
            areas[i, j] = (radius_m**2) * abs(np.sin(lat_north) - np.sin(lat_south)) * abs(lon_east - lon_west)
    return areas


def infer_grid_cell_areas(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("WUS-D3 d02 latitude/longitude must both be 2D.")
    areas = area_from_bounds_2d(
        np.asarray(lat.values, dtype=np.float64),
        np.asarray(lon.values, dtype=np.float64),
    )
    return xr.DataArray(areas, dims=lat.dims, coords=lat.coords, name="cell_area_m2")


def weighted_masked_mean(field: xr.DataArray, mask: xr.DataArray, area: xr.DataArray) -> tuple[float, float]:
    valid = np.isfinite(field)
    weights = mask.astype(np.float64) * area.astype(np.float64)
    weights = weights.where(valid)
    denom = float(weights.sum(skipna=True).item())
    if denom <= 0.0:
        return float("nan"), float("nan")
    numer = float((field.astype(np.float64) * weights).sum(skipna=True).item())
    return numer / denom, denom


def infer_experiment_from_dataset_id(dataset_id: str) -> str:
    return "ssp370" if dataset_id.endswith("_ssp370_bc") else "historical"


def strip_wusd3_dataset_id(dataset_id: str) -> str:
    if dataset_id.endswith("_historical_bc"):
        return dataset_id[: -len("_historical_bc")]
    if dataset_id.endswith("_ssp370_bc"):
        return dataset_id[: -len("_ssp370_bc")]
    return dataset_id


def load_parent_run_map() -> pd.DataFrame:
    return pd.read_csv(PARENT_RUNS_CSV)


def build_mask_and_area(dataset_id: str) -> tuple[object, xr.DataArray, xr.DataArray, xr.DataArray, dict[str, object]]:
    dataset = Wusd3Dataset(dataset_id=dataset_id, domain="d02", root_dir=Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3"))
    grid = get_wusd3_grid_definition(dataset, water_year=1985, region=DEFAULT_SIERRA_REGION, coarsen_factor=1)
    mask = build_sierra_mask(grid, region=DEFAULT_SIERRA_REGION).astype(np.float64)
    area = infer_grid_cell_areas(grid.fine_latitude, grid.fine_longitude)
    weights = (mask * area).rename("sierra_weight_m2")
    positive = weights > 0.0
    summary = {
        "grid_shape": [int(grid.fine_latitude.shape[0]), int(grid.fine_latitude.shape[1])],
        "selected_cell_count": int(positive.sum().item()),
        "total_selected_area_m2": float(weights.sum(skipna=True).item()),
        "lat_min_selected": float(grid.fine_latitude.where(positive).min(skipna=True).item()),
        "lat_max_selected": float(grid.fine_latitude.where(positive).max(skipna=True).item()),
        "lon_min_selected": float(grid.fine_longitude.where(positive).min(skipna=True).item()),
        "lon_max_selected": float(grid.fine_longitude.where(positive).max(skipna=True).item()),
        "mask_type": str(mask.attrs.get("mask_type", "unknown")),
    }
    return dataset, grid, mask, area, summary


def extract_all_labels() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    parent_runs = load_parent_run_map()
    ref_dataset_id = "ec-earth3_r1i1p1f1_2_historical_bc"
    _, ref_grid, ref_mask, ref_area, mask_summary = build_mask_and_area(ref_dataset_id)

    rows: list[dict[str, object]] = []
    manual_checks: list[dict[str, object]] = []
    source_paths: dict[str, str] = {}

    for parent in parent_runs.itertuples(index=False):
        base_id = str(parent.wusd3_dataset_id)
        for scenario_suffix in ("historical_bc", "ssp370_bc"):
            dataset_id = f"{base_id}_{scenario_suffix}"
            experiment = "historical" if scenario_suffix == "historical_bc" else "ssp370"
            dataset = Wusd3Dataset(dataset_id=dataset_id, domain="d02", root_dir=Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3"))
            years = discover_wusd3_file_years(dataset)
            for file_year in years:
                water_year = int(file_year) + 1
                snapshot = load_wusd3_snapshot(
                    dataset,
                    water_year=water_year,
                    snapshot_date=date(water_year, 4, 1),
                    swe_grid=ref_grid,
                    fill_missing=False,
                )
                swe_mm, valid_area_m2 = weighted_masked_mean(snapshot, ref_mask, ref_area)
                source_path = snapshot.attrs.get("source_path", str(variable_path_for_file_year(dataset, "swe", file_year)))
                rows.append(
                    {
                        "source_id": str(parent.source_id),
                        "member_id": str(parent.member_id),
                        "model_member": f"{parent.source_id}:{parent.member_id}",
                        "experiment": experiment,
                        "row_year": file_year,
                        "water_year": water_year,
                        "SWE_label": swe_mm,
                        "SWE_units": "mm",
                        "source_path": source_path,
                        "valid_selected_area_m2": valid_area_m2,
                    }
                )
                source_paths[f"{dataset_id}:{file_year}"] = source_path
            # Manual QA for a few representative years per scenario when available.
            subset_years = []
            if years:
                subset_years = [years[0], years[min(len(years) - 1, len(years) // 2)], years[-1]]
                subset_years = sorted(set(subset_years))
            for file_year in subset_years:
                water_year = file_year + 1
                snapshot = load_wusd3_snapshot(
                    dataset,
                    water_year=water_year,
                    snapshot_date=date(water_year, 4, 1),
                    swe_grid=ref_grid,
                    fill_missing=False,
                )
                swe_mm, valid_area_m2 = weighted_masked_mean(snapshot, ref_mask, ref_area)
                manual_checks.append(
                    {
                        "dataset_id": dataset_id,
                        "source_id": str(parent.source_id),
                        "member_id": str(parent.member_id),
                        "experiment": experiment,
                        "row_year": file_year,
                        "water_year": water_year,
                        "apr1_date": f"{water_year:04d}-04-01",
                        "scalar_swe_mm": swe_mm,
                        "valid_selected_area_m2": valid_area_m2,
                        "min_selected_swe_mm": float(snapshot.where(ref_mask > 0).min(skipna=True).item()),
                        "max_selected_swe_mm": float(snapshot.where(ref_mask > 0).max(skipna=True).item()),
                        "source_path": snapshot.attrs.get("source_path"),
                    }
                )

    full_df = pd.DataFrame(rows).sort_values(["source_id", "experiment", "row_year"]).reset_index(drop=True)
    full_df.insert(0, "wusd3_sample_full", np.arange(len(full_df), dtype=int))
    manual_df = pd.DataFrame(manual_checks).sort_values(["dataset_id", "row_year"]).reset_index(drop=True)

    aux_df = pd.read_csv(AUX_LABELS_CSV)
    join_cols = ["source_id", "member_id", "model_member", "experiment", "row_year", "water_year"]
    overlap_df = aux_df[join_cols + ["sample"]].merge(full_df, on=join_cols, how="inner")
    overlap_df = overlap_df[
        ["sample", "source_id", "member_id", "model_member", "experiment", "row_year", "water_year", "SWE_label", "SWE_units", "source_path", "valid_selected_area_m2", "wusd3_sample_full"]
    ].sort_values("sample").reset_index(drop=True)

    summary = {
        "sierra_region_definition_recovered_from": str(PROJECT_ROOT / "docs" / "Current_Status.tex"),
        "sierra_region_bounds": asdict(DEFAULT_SIERRA_REGION),
        "wusd3_mask_construction": "Applied the canonical geographic Sierra box to native WUS-D3 d02 reconstructed lat/lon coordinates using snow_ml.data.build_sierra_mask with coarsen_factor=1, then weighted by native cell area.",
        "selected_cell_count": int(mask_summary["selected_cell_count"]),
        "total_selected_area_m2": float(mask_summary["total_selected_area_m2"]),
        "selected_area_km2": float(mask_summary["total_selected_area_m2"]) / 1.0e6,
        "selected_lat_lon_bounds_on_d02": {
            "lat_min": float(mask_summary["lat_min_selected"]),
            "lat_max": float(mask_summary["lat_max_selected"]),
            "lon_min": float(mask_summary["lon_min_selected"]),
            "lon_max": float(mask_summary["lon_max_selected"]),
        },
        "mask_type": str(mask_summary["mask_type"]),
        "wusd3_variable": WUSD3_SWE_VARIABLE,
        "wusd3_units": "mm",
        "temporal_resolution": "daily",
        "april1_extraction_rule": "Exact day match on daily WUS-D3 d02 file: April 1 of water year Y within file year Y-1.",
        "area_weighting_method": "Native WUS-D3 d02 cell area from reconstructed 2D lat/lon geometry using spherical quadrilateral approximation; missing SWE excluded from numerator and denominator.",
        "final_label_count": int(len(overlap_df)),
        "full_available_label_count": int(len(full_df)),
        "overlap_with_cmip6_aux_rows": int(len(overlap_df)),
        "composition_by_model_experiment": {
            f"{source_id}|{experiment}": int(count)
            for (source_id, experiment), count in full_df.groupby(["source_id", "experiment"]).size().items()
        },
        "canonical_table_note": "The canonical SWE-label table is restricted to the exact 394-row overlap with the CMIP6 predictor/auxiliary manifest so its `sample` column matches the existing CNN manifest. A full 480-row WUS availability table is saved separately.",
        "manual_check_rows": int(len(manual_df)),
        "source_paths_example": {k: source_paths[k] for k in list(source_paths)[:6]},
    }
    return overlap_df, full_df, summary, manual_df


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    labels_df, full_df, summary, manual_df = extract_all_labels()
    labels_df.to_csv(OUT_CSV, index=False)
    full_df.to_csv(OUT_FULL_CSV, index=False)
    manual_df.to_csv(QA_CSV, index=False)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(labels_df)} SWE labels to {OUT_CSV}")
    print(f"Overlap with CMIP6 auxiliary labels: {summary['overlap_with_cmip6_aux_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
