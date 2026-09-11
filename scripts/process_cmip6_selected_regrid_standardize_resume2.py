#!/usr/bin/env python3
"""Resume-friendly CMIP6 regrid + standardize driver without mixed-calendar merge writes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_base_module():
    script_path = Path(__file__).resolve().with_name("process_cmip6_selected_regrid_standardize.py")
    spec = importlib.util.spec_from_file_location("cmip6_regrid_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import base module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(default_output_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--field", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_var_dataset(base, path: Path, variable_id: str) -> xr.DataArray:
    ds = base.open_selected_dataset(path, variable_id)
    da = base.select_data_var(ds, variable_id).astype(np.float32)
    return da


def build_lookup(da: xr.DataArray) -> dict[tuple[int, int], int]:
    years = da["time"].dt.year.values
    months = da["time"].dt.month.values
    return {(int(y), int(m)): idx for idx, (y, m) in enumerate(zip(years, months, strict=False))}


def build_model_year_rows_from_parts(base, field_inputs: dict[str, dict[str, Path | None]], variable_id: str):
    samples: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    for model_key, parts in field_inputs.items():
        hist_path = parts.get("historical")
        ssp_path = parts.get("ssp370")
        hist_ds = base.open_selected_dataset(hist_path, variable_id) if hist_path else None
        ssp_ds = base.open_selected_dataset(ssp_path, variable_id) if ssp_path else None
        hist_da = base.select_data_var(hist_ds, variable_id).astype(np.float32) if hist_ds is not None else None
        ssp_da = base.select_data_var(ssp_ds, variable_id).astype(np.float32) if ssp_ds is not None else None
        hist_lookup = build_lookup(hist_da) if hist_da is not None else {}
        ssp_lookup = build_lookup(ssp_da) if ssp_da is not None else {}

        for start_year in range(1980, 2100):
            month_keys = [(start_year, 9), (start_year, 10), (start_year, 11), (start_year, 12)] + [
                (start_year + 1, m) for m in range(1, 9)
            ]
            pieces: list[np.ndarray] = []
            missing: list[tuple[int, int]] = []
            for year, month in month_keys:
                use_hist = year < 2015
                if use_hist and hist_da is not None and (year, month) in hist_lookup:
                    pieces.append(np.asarray(hist_da.isel(time=hist_lookup[(year, month)]).values, dtype=np.float32))
                elif (not use_hist) and ssp_da is not None and (year, month) in ssp_lookup:
                    pieces.append(np.asarray(ssp_da.isel(time=ssp_lookup[(year, month)]).values, dtype=np.float32))
                else:
                    missing.append((year, month))
            if missing:
                gaps.append({"model": model_key, "row_year": start_year, "missing": missing})
                continue
            samples.append(np.stack(pieces, axis=0))
            rows.append({"model": model_key, "row_year": start_year})

        if hist_ds is not None:
            hist_ds.close()
        if ssp_ds is not None:
            ssp_ds.close()

    if not samples:
        raise RuntimeError(f"No model-year rows built for {variable_id}")
    return np.stack(samples, axis=0), rows, gaps


def main() -> None:
    base = load_base_module()
    args = parse_args(base.DEFAULT_OUTPUT_ROOT)
    output_root = args.output_root
    fields = [spec for spec in base.FIELD_SPECS if args.field is None or spec.field_id in set(args.field)]
    base.ensure_dir(output_root)
    grid_file = output_root / "grid_1p5deg.txt"
    base.write_target_grid(grid_file)

    manifest_rows: list[dict[str, object]] = []
    weight_manifest: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for spec in fields:
        field_inputs: dict[str, dict[str, Path | None]] = {}
        for parent in base.PARENT_RUNS:
            weight_path = base.ensure_weight_file(output_root, grid_file, parent, spec, args.overwrite)
            weight_manifest.append(
                {
                    "model": parent.model,
                    "member_id": parent.member_id,
                    "family": spec.source_family,
                    "method": spec.method,
                    "weight_file": str(weight_path),
                }
            )
            hist_path = base.merge_select_and_regrid(output_root, grid_file, parent, spec, "historical", args.overwrite)
            ssp_path = base.merge_select_and_regrid(output_root, grid_file, parent, spec, "ssp370", args.overwrite)
            field_inputs[f"{parent.model}:{parent.member_id}"] = {"historical": hist_path, "ssp370": ssp_path}

            for experiment, path in (("historical", hist_path), ("ssp370", ssp_path)):
                if path is None:
                    continue
                with xr.open_dataset(path, decode_times=True, use_cftime=True) as ds_out:
                    out_var = base.select_data_var(ds_out, spec.variable_id)
                    grid_check = base.validate_target_grid(out_var)
                    arr = np.asarray(out_var.values, dtype=np.float64)
                    validation_rows.append(
                        {
                            "field": spec.field_id,
                            "model": parent.model,
                            "experiment": experiment,
                            **grid_check,
                            "time_count": int(out_var.sizes["time"]),
                            "units": out_var.attrs.get("units", ""),
                            "nan_any": bool(np.isnan(arr).any()),
                            "all_nan_any_time": bool(np.any(np.all(np.isnan(arr), axis=(-2, -1)))),
                            "min": float(np.nanmin(arr)),
                            "max": float(np.nanmax(arr)),
                            "mean": float(np.nanmean(arr)),
                        }
                    )

        values, rows, gaps = build_model_year_rows_from_parts(base, field_inputs, spec.variable_id)
        raw_field_path = output_root / "raw" / f"{spec.output_name}_1p5deg_model_years.nc"
        std_field_path = output_root / "standardized" / f"{spec.output_name}_1p5deg_model_years_standardized.nc"
        stats_field_path = output_root / "normalization_stats" / f"{spec.output_name}_1p5deg_stats.nc"

        base.save_field_raw_dataset(raw_field_path, spec.output_name, values, rows)
        standardized, mu, sigma, valid_count, zero_var = base.standardize_values(values)
        base.save_standardized_dataset(std_field_path, spec.output_name, standardized, rows)
        base.save_stats_dataset(stats_field_path, spec.output_name, mu, sigma, valid_count, zero_var)

        manifest_rows.append(
            {
                "field": spec.field_id,
                "level_pa": spec.level_pa,
                "depth_m": spec.depth_m,
                "regridding_method": spec.method,
                "weight_file": " | ".join(
                    sorted(
                        {
                            str(base.weight_file_path(output_root, parent, spec.source_family, spec.method))
                            for parent in base.PARENT_RUNS
                        }
                    )
                ),
                "raw_regridded_output": str(raw_field_path),
                "standardized_output": str(std_field_path),
                "stats_output": str(stats_field_path),
                "valid_rows_years": int(values.shape[0]),
                "nan_mask_behavior_verified": bool(np.array_equal(np.isnan(values), np.isnan(standardized))),
                "zero_variance_columns": int(np.sum(zero_var)),
                "qa_status": "ok",
                "status": "ok",
                "gaps_preview": json.dumps(gaps[:20]),
                "raw_gb": round(base.path_storage_gb(raw_field_path), 3),
                "std_gb": round(base.path_storage_gb(std_field_path), 3),
            }
        )

    artifact_dir = base.PROJECT_ROOT / "artifacts" / "cmip6_regrid_standardize"
    base.ensure_dir(artifact_dir)
    base.upsert_csv_rows(artifact_dir / "field_summary.csv", manifest_rows, ["field"])
    base.upsert_csv_rows(
        artifact_dir / "weight_manifest.csv",
        list(pd.DataFrame(weight_manifest).drop_duplicates().to_dict("records")),
        ["model", "member_id", "family", "method", "weight_file"],
    )
    base.upsert_csv_rows(
        artifact_dir / "validation_checks.csv",
        validation_rows,
        ["field", "model", "experiment"],
    )

    existing_summary: dict[str, object] = {}
    run_summary_path = artifact_dir / "run_summary.json"
    if run_summary_path.exists():
        existing_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    prior_fields = set(existing_summary.get("fields_processed", []))
    prior_fields.update(spec.field_id for spec in fields)
    existing_summary.update(
        {
            "output_root": str(output_root),
            "fields_processed": sorted(prior_fields),
            "thetao_status": (
                "thetao_50m and thetao_100m processed; MPI-ESM1-2-HR ssp370 remains unavailable locally"
                if any(spec.variable_id == "thetao" for spec in fields)
                else existing_summary.get(
                    "thetao_status",
                    "pending vertical-coordinate decision before horizontal/tensor harmonization",
                )
            ),
        }
    )
    run_summary_path.write_text(json.dumps(existing_summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
