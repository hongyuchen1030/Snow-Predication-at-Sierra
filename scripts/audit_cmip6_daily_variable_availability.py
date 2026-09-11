#!/usr/bin/env python3
"""Ground-truth audit of daily/sub-daily CMIP6 availability for the 19 existing
seasonal-CNN input variables, across all 4 parent models.

This does NOT read our already-processed monthly files. It inspects the raw
ESGF-mirrored CMIP6 archive directly: for every CMIP6 table_id directory that
exists under each parent model's historical/ssp370 root, it checks whether each
physical variable_id has data there, opens one representative NetCDF file, and
records its actual global-attribute metadata (source_id, experiment_id,
variant_label, table_id, frequency, grid_label) plus the vertical coordinate
values (pressure level / ocean depth) and time coverage read directly from that
file. No frequency is inferred from directory/table names alone.

Outputs (under artifacts/cmip6_daily_availability_audit/):
  - table_scan_raw.csv          one row per (variable_id, model, experiment, table_id) found on disk
  - variable_model_availability_19x4.csv   the requested 19-variable x 4-model table
  - audit_summary.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import netCDF4
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts" / "cmip6_daily_availability_audit"

# Reuse the exact PARENT_RUNS / FIELD_SPECS the existing seasonal pipeline uses,
# instead of re-declaring them, so this audit cannot silently drift from what the
# CNN pipeline actually consumes.
spec_module_path = PROJECT_ROOT / "scripts" / "process_cmip6_selected_regrid_standardize.py"
spec = importlib.util.spec_from_file_location("process_cmip6_selected_regrid_standardize", spec_module_path)
pipeline_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline_module
spec.loader.exec_module(pipeline_module)  # type: ignore[union-attr]

PARENT_RUNS = pipeline_module.PARENT_RUNS
FIELD_SPECS = pipeline_module.FIELD_SPECS
EXPERIMENTS = ("historical", "ssp370")


def parent_root(parent, experiment: str) -> Path:
    return parent.hist_root if experiment == "historical" else parent.ssp_root


def list_table_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def representative_file_for(root: Path, table_id: str, variable_id: str) -> Path | None:
    base = root / table_id / variable_id
    if not base.exists():
        return None
    grid_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    for grid_dir in grid_dirs:
        version_dirs = sorted(p for p in grid_dir.iterdir() if p.is_dir())
        if not version_dirs:
            continue
        latest_version = version_dirs[-1]
        nc_files = sorted(latest_version.glob("*.nc"))
        if nc_files:
            return nc_files[0]
    return None


def read_time_coverage(ds: netCDF4.Dataset) -> tuple[str | None, str | None, int | None]:
    if "time" not in ds.variables:
        return None, None, None
    time_var = ds.variables["time"]
    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    values = np.asarray(time_var[:])
    if values.size == 0 or units is None:
        return None, None, int(values.size)
    try:
        start = netCDF4.num2date(values[0], units=units, calendar=calendar)
        end = netCDF4.num2date(values[-1], units=units, calendar=calendar)
        return str(start), str(end), int(values.size)
    except Exception:
        return f"raw:{values[0]}", f"raw:{values[-1]}", int(values.size)


def find_vertical_coordinate(ds: netCDF4.Dataset, variable_id: str) -> tuple[str | None, np.ndarray | None, str | None]:
    if variable_id not in ds.variables:
        return None, None, None
    dims = ds.variables[variable_id].dimensions
    for dim_name in dims:
        if dim_name not in ds.variables:
            continue
        var = ds.variables[dim_name]
        standard_name = str(getattr(var, "standard_name", "")).lower()
        axis = str(getattr(var, "axis", "")).upper()
        units = str(getattr(var, "units", "")).lower()
        positive = str(getattr(var, "positive", "")).lower()
        is_pressure = standard_name == "air_pressure" or (axis == "Z" and units == "pa")
        is_depth = standard_name in {"depth", "ocean_sigma_z"} or (
            axis == "Z" and (units in {"m", "meter", "meters", "metre", "metres"} or positive == "down")
        )
        if is_pressure or is_depth:
            kind = "pressure_pa" if is_pressure else "depth_m"
            return dim_name, np.asarray(var[:], dtype=np.float64), kind
    return None, None, None


def inspect_file(path: Path, variable_id: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = netCDF4.Dataset(path, mode="r")
    try:
        global_attrs = {name: ds.getncattr(name) for name in ds.ncattrs()}
        start, end, n_time = read_time_coverage(ds)
        vcoord_name, vcoord_values, vcoord_kind = find_vertical_coordinate(ds, variable_id)
        var_attrs = {}
        if variable_id in ds.variables:
            var_attrs = {name: ds.variables[variable_id].getncattr(name) for name in ds.variables[variable_id].ncattrs()}
        return {
            "file_path": str(path),
            "source_id": global_attrs.get("source_id"),
            "institution_id": global_attrs.get("institution_id"),
            "activity_id": global_attrs.get("activity_id"),
            "experiment_id": global_attrs.get("experiment_id"),
            "variant_label": global_attrs.get("variant_label"),
            "table_id": global_attrs.get("table_id"),
            "frequency": global_attrs.get("frequency"),
            "grid_label": global_attrs.get("grid_label"),
            "variable_id_in_file": variable_id if variable_id in ds.variables else None,
            "variable_units": var_attrs.get("units"),
            "vertical_coord_name": vcoord_name,
            "vertical_coord_kind": vcoord_kind,
            "vertical_coord_values": vcoord_values.tolist() if vcoord_values is not None else None,
            "time_start": start,
            "time_end": end,
            "n_time_steps": n_time,
        }
    finally:
        ds.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_variable_ids = sorted({spec_.variable_id for spec_ in FIELD_SPECS})

    # scan[(variable_id, model, experiment)][table_id] = inspect_file(...) or {"error": ...}
    scan: dict[tuple[str, str, str], dict[str, dict]] = {}

    for parent in PARENT_RUNS:
        for experiment in EXPERIMENTS:
            root = parent_root(parent, experiment)
            table_ids = list_table_ids(root)
            print(f"[{parent.model} | {experiment}] tables on disk: {table_ids}", flush=True)
            for variable_id in unique_variable_ids:
                key = (variable_id, parent.model, experiment)
                scan.setdefault(key, {})
                for table_id in table_ids:
                    rep_file = representative_file_for(root, table_id, variable_id)
                    if rep_file is None:
                        continue
                    try:
                        meta = inspect_file(rep_file, variable_id)
                        meta["parent_member_id_expected"] = parent.member_id
                        meta["member_matches_expected"] = meta.get("variant_label") == parent.member_id
                        scan[key][table_id] = meta
                        print(
                            f"    found {variable_id} in {table_id}: "
                            f"frequency={meta.get('frequency')} variant={meta.get('variant_label')}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        scan[key][table_id] = {"error": str(exc), "file_path": str(rep_file)}
                        print(f"    ERROR reading {rep_file}: {exc}", flush=True)

    # ---- flat raw scan CSV ----
    raw_rows: list[dict] = []
    for (variable_id, model, experiment), table_map in scan.items():
        for table_id, meta in table_map.items():
            row = {"variable_id": variable_id, "model": model, "experiment": experiment, "table_id_dir": table_id}
            row.update(meta)
            raw_rows.append(row)

    raw_csv_path = OUT_DIR / "table_scan_raw.csv"
    if raw_rows:
        all_keys: list[str] = []
        for row in raw_rows:
            for k in row.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with raw_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_keys)
            writer.writeheader()
            for row in raw_rows:
                serializable = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}
                writer.writerow(serializable)

    # ---- 19 (field) x 4 (model) availability table ----
    final_rows: list[dict] = []
    for field_spec in FIELD_SPECS:
        for parent in PARENT_RUNS:
            monthly_meta = scan.get((field_spec.variable_id, parent.model, "historical"), {}).get(field_spec.table_id)
            monthly_meta_ssp = scan.get((field_spec.variable_id, parent.model, "ssp370"), {}).get(field_spec.table_id)

            daily_candidates: list[tuple[str, dict, dict | None]] = []
            for experiment in EXPERIMENTS:
                table_map = scan.get((field_spec.variable_id, parent.model, experiment), {})
                for table_id, meta in table_map.items():
                    if table_id == field_spec.table_id:
                        continue
                    if "error" in meta:
                        continue
                    if str(meta.get("frequency", "")).lower() not in {"day", "daily", "d1"}:
                        continue
                    # verify the specific level (if any) is present in this candidate table's file
                    level_ok = True
                    if field_spec.level_pa is not None:
                        level_ok = meta.get("vertical_coord_kind") == "pressure_pa" and meta.get(
                            "vertical_coord_values"
                        ) is not None and any(
                            abs(float(v) - float(field_spec.level_pa)) < 1.0 for v in meta["vertical_coord_values"]
                        )
                    elif field_spec.depth_m is not None:
                        vcvals = meta.get("vertical_coord_values")
                        level_ok = meta.get("vertical_coord_kind") == "depth_m" and vcvals is not None and (
                            min(vcvals) <= field_spec.depth_m <= max(vcvals)
                        )
                    daily_candidates.append((table_id, meta, {"experiment": experiment, "level_ok": level_ok}))

            hist_daily = [c for c in daily_candidates if c[2]["experiment"] == "historical" and c[2]["level_ok"]]
            ssp_daily = [c for c in daily_candidates if c[2]["experiment"] == "ssp370" and c[2]["level_ok"]]
            best = hist_daily[0] if hist_daily else (daily_candidates[0] if daily_candidates else None)

            def dataset_id(meta: dict, table_id: str) -> str | None:
                if not meta or "error" in meta:
                    return None
                parts = [
                    "CMIP6",
                    meta.get("activity_id"),
                    meta.get("institution_id"),
                    meta.get("source_id"),
                    meta.get("experiment_id"),
                    meta.get("variant_label"),
                    table_id,
                    field_spec.variable_id,
                    meta.get("grid_label"),
                ]
                if any(p is None for p in parts):
                    return None
                return ".".join(str(p) for p in parts)

            final_rows.append(
                {
                    "field_name": field_spec.field_id,
                    "variable_id": field_spec.variable_id,
                    "level_pa": field_spec.level_pa,
                    "depth_m": field_spec.depth_m,
                    "model": parent.model,
                    "monthly_table_id_used": field_spec.table_id,
                    "monthly_frequency_confirmed": monthly_meta.get("frequency") if monthly_meta else None,
                    "monthly_source_id": monthly_meta.get("source_id") if monthly_meta else None,
                    "monthly_variant_label": monthly_meta.get("variant_label") if monthly_meta else None,
                    "monthly_historical_available": monthly_meta is not None and "error" not in monthly_meta,
                    "monthly_ssp370_available": monthly_meta_ssp is not None and "error" not in monthly_meta_ssp,
                    "monthly_historical_time_range": (
                        f"{monthly_meta.get('time_start')} .. {monthly_meta.get('time_end')}" if monthly_meta else None
                    ),
                    "monthly_ssp370_time_range": (
                        f"{monthly_meta_ssp.get('time_start')} .. {monthly_meta_ssp.get('time_end')}"
                        if monthly_meta_ssp
                        else None
                    ),
                    "daily_equivalent_available": best is not None,
                    "daily_table_id": best[0] if best else None,
                    "daily_historical_available": len(hist_daily) > 0,
                    "daily_ssp370_available": len(ssp_daily) > 0,
                    "daily_exact_dataset_id_historical": (
                        dataset_id(hist_daily[0][1], hist_daily[0][0]) if hist_daily else None
                    ),
                    "daily_exact_dataset_id_ssp370": dataset_id(ssp_daily[0][1], ssp_daily[0][0]) if ssp_daily else None,
                    "daily_member_matches_monthly_member": best[1].get("member_matches_expected") if best else None,
                    "daily_variant_label": best[1].get("variant_label") if best else None,
                    "daily_frequency_confirmed": best[1].get("frequency") if best else None,
                    "daily_level_present": best[2]["level_ok"] if best else None,
                    "daily_time_range": (f"{best[1].get('time_start')} .. {best[1].get('time_end')}" if best else None),
                }
            )

    final_csv_path = OUT_DIR / "variable_model_availability_19x4.csv"
    with final_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "n_field_specs": len(FIELD_SPECS),
        "n_parent_models": len(PARENT_RUNS),
        "n_rows_19x4_table": len(final_rows),
        "n_raw_scan_rows": len(raw_rows),
        "daily_available_row_count": sum(1 for r in final_rows if r["daily_equivalent_available"]),
        "daily_available_both_experiments_row_count": sum(
            1 for r in final_rows if r["daily_historical_available"] and r["daily_ssp370_available"]
        ),
        "output_files": {
            "raw_scan_csv": str(raw_csv_path),
            "final_19x4_csv": str(final_csv_path),
        },
    }
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
