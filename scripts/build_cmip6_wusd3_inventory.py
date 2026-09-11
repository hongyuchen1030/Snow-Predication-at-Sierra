#!/usr/bin/env python3
"""Build a CMIP6 inventory for the four WUS-D3 parent GCMs.

This script prefers the local NERSC CMIP6 holdings and records explicit notes
when a required branch is not present locally.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xarray as xr
from cftime import num2date


CMIP6_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/CMIP6")
WUSD3_DAILY_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily")
OUT_DIR = Path("artifacts/cmip6_wusd3_inventory")


@dataclass(frozen=True)
class ParentRun:
    wusd3_dataset_id: str
    source_id: str
    institution_id: str
    member_id: str
    historical_experiment: str
    ssp370_experiment: str
    hist_activity: str
    ssp_activity: str
    hist_source_path: Path
    ssp_source_path: Path
    mapping_note: str
    evidence: str
    mapping_confidence: str


PARENT_RUNS = [
    ParentRun(
        wusd3_dataset_id="ec-earth3_r1i1p1f1_2",
        source_id="EC-Earth3",
        institution_id="EC-Earth-Consortium",
        member_id="r102i1p1f1",
        historical_experiment="historical",
        ssp370_experiment="ssp370",
        hist_activity="CMIP",
        ssp_activity="ScenarioMIP",
        hist_source_path=CMIP6_ROOT
        / "CMIP"
        / "EC-Earth-Consortium"
        / "EC-Earth3"
        / "historical"
        / "r102i1p1f1",
        ssp_source_path=CMIP6_ROOT
        / "ScenarioMIP"
        / "EC-Earth-Consortium"
        / "EC-Earth3"
        / "ssp370"
        / "r102i1p1f1",
        mapping_note=(
            "WUS-D3 names this run ec-earth3_r1i1p1f1_2. The local CMIP6 EC-Earth3 "
            "members are indexed as r101i1p1f1, r102i1p1f1, ...; this script maps "
            "the WUS suffix _2 to r102i1p1f1, but that mapping should be treated as "
            "high-confidence inference unless stronger provenance is found."
        ),
        evidence=(
            "WUS-D3 datalake dataset ids; local EC-Earth3 CMIP6 member directory "
            "layout under CMIP and ScenarioMIP"
        ),
        mapping_confidence="inferred_high",
    ),
    ParentRun(
        wusd3_dataset_id="miroc6_r1i1p1f1",
        source_id="MIROC6",
        institution_id="MIROC",
        member_id="r1i1p1f1",
        historical_experiment="historical",
        ssp370_experiment="ssp370",
        hist_activity="CMIP",
        ssp_activity="ScenarioMIP",
        hist_source_path=CMIP6_ROOT / "CMIP" / "MIROC" / "MIROC6" / "historical" / "r1i1p1f1",
        ssp_source_path=CMIP6_ROOT / "ScenarioMIP" / "MIROC" / "MIROC6" / "ssp370" / "r1i1p1f1",
        mapping_note="Direct match between WUS-D3 dataset name and local CMIP6 member id.",
        evidence="WUS-D3 datalake dataset ids; local CMIP6 MIROC6 directory layout",
        mapping_confidence="verified_local_name_match",
    ),
    ParentRun(
        wusd3_dataset_id="mpi-esm1-2-hr_r3i1p1f1",
        source_id="MPI-ESM1-2-HR",
        institution_id="MPI-M",
        member_id="r3i1p1f1",
        historical_experiment="historical",
        ssp370_experiment="ssp370",
        hist_activity="CMIP",
        ssp_activity="ScenarioMIP",
        hist_source_path=CMIP6_ROOT
        / "CMIP"
        / "MPI-M"
        / "MPI-ESM1-2-HR"
        / "historical"
        / "r3i1p1f1",
        ssp_source_path=CMIP6_ROOT
        / "ScenarioMIP"
        / "MPI-M"
        / "MPI-ESM1-2-HR"
        / "ssp370"
        / "r3i1p1f1",
        mapping_note=(
            "Historical member matches directly. The expected local ScenarioMIP "
            "ssp370 branch is missing under the standard NERSC CMIP6 mirror path."
        ),
        evidence=(
            "WUS-D3 datalake dataset ids; local historical CMIP6 MPI-ESM1-2-HR "
            "directory layout; missing local ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370"
        ),
        mapping_confidence="historical_verified_scenario_local_gap",
    ),
    ParentRun(
        wusd3_dataset_id="taiesm1_r1i1p1f1",
        source_id="TaiESM1",
        institution_id="AS-RCEC",
        member_id="r1i1p1f1",
        historical_experiment="historical",
        ssp370_experiment="ssp370",
        hist_activity="CMIP",
        ssp_activity="ScenarioMIP",
        hist_source_path=CMIP6_ROOT / "CMIP" / "AS-RCEC" / "TaiESM1" / "historical" / "r1i1p1f1",
        ssp_source_path=CMIP6_ROOT / "ScenarioMIP" / "AS-RCEC" / "TaiESM1" / "ssp370" / "r1i1p1f1",
        mapping_note="Direct match between WUS-D3 dataset name and local CMIP6 member id.",
        evidence="WUS-D3 datalake dataset ids; local CMIP6 TaiESM1 directory layout",
        mapping_confidence="verified_local_name_match",
    ),
]


VERTICAL_DIM_NAMES = [
    "plev",
    "lev",
    "alev",
    "olevel",
    "olevhalf",
    "depth",
    "sdepth",
    "rho",
    "sigma",
]


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_wusd3_years(dataset_id: str) -> tuple[int | None, int | None, int]:
    snow_dir = WUSD3_DAILY_ROOT / dataset_id / "postprocess" / "d02"
    years = []
    if snow_dir.exists():
        for path in snow_dir.glob("snow*.nc"):
            token = path.stem.split(".")[-1]
            if token.isdigit():
                years.append(int(token))
    if not years:
        return None, None, 0
    years = sorted(set(years))
    return years[0], years[-1], len(years)


def to_iso(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def read_time_bounds(nc_path: Path) -> tuple[str | None, str | None, str | None]:
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as exc:
        return None, None, f"time_read_error: {exc}"
    try:
        if "time" not in ds:
            return None, None, "missing_time_coordinate"
        time_var = ds["time"]
        units = time_var.attrs.get("units")
        calendar = time_var.attrs.get("calendar", "standard")
        if units is None or time_var.size == 0:
            return None, None, "time_units_missing_or_empty"
        start = num2date(time_var.values[0], units, calendar=calendar)
        end = num2date(time_var.values[-1], units, calendar=calendar)
        return to_iso(start), to_iso(end), None
    finally:
        ds.close()


def summarize_variable_file(nc_path: Path, variable_id: str) -> dict[str, str]:
    summary: dict[str, str] = {
        "standard_name": "",
        "long_name": "",
        "units": "",
        "dimensions": "",
        "vertical_coordinate": "",
        "grid_label": "",
        "attribute_error": "",
    }
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as exc:
        summary["attribute_error"] = f"open_error: {exc}"
        return summary
    try:
        if variable_id not in ds.variables:
            summary["attribute_error"] = f"missing_variable:{variable_id}"
            return summary
        var = ds[variable_id]
        summary["standard_name"] = str(var.attrs.get("standard_name", ""))
        summary["long_name"] = str(var.attrs.get("long_name", ""))
        summary["units"] = str(var.attrs.get("units", ""))
        dims = list(var.dims)
        summary["dimensions"] = "|".join(dims)
        vertical_dims = [dim for dim in dims if dim in VERTICAL_DIM_NAMES]
        summary["vertical_coordinate"] = "|".join(vertical_dims)
        summary["grid_label"] = str(ds.attrs.get("grid_label", ""))
        return summary
    finally:
        ds.close()


def coverage_ok(experiment_id: str, start: str | None, end: str | None) -> bool:
    if not start or not end:
        return False
    if experiment_id == "historical":
        return start <= "1980-09-01" and end >= "2014-08-31"
    if experiment_id == "ssp370":
        return start <= "2015-01-01" and end >= "2100-08-31"
    return False


def combined_water_year_ok(hist_end: str | None, ssp_start: str | None, ssp_end: str | None) -> bool:
    if not hist_end or not ssp_start or not ssp_end:
        return False
    return hist_end >= "2014-12-01" and ssp_start <= "2015-01-01" and ssp_end >= "2100-08-31"


def collect_inventory(parent: ParentRun, experiment_id: str, root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        rows.append(
            {
                "wusd3_dataset_id": parent.wusd3_dataset_id,
                "source_id": parent.source_id,
                "institution_id": parent.institution_id,
                "member_id": parent.member_id,
                "experiment_id": experiment_id,
                "table_id": "",
                "variable_id": "",
                "grid_label": "",
                "version": "",
                "standard_name": "",
                "long_name": "",
                "units": "",
                "dimensions": "",
                "vertical_coordinate": "",
                "n_files": "0",
                "time_start": "",
                "time_end": "",
                "coverage_ok_for_wusd3": "False",
                "local_archive_status": "missing_local_branch",
                "note": f"Expected local path does not exist: {root}",
            }
        )
        return rows

    for table_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        table_id = table_dir.name
        for variable_dir in sorted(path for path in table_dir.iterdir() if path.is_dir()):
            variable_id = variable_dir.name
            for grid_dir in sorted(path for path in variable_dir.iterdir() if path.is_dir()):
                grid_label = grid_dir.name
                for version_dir in sorted(path for path in grid_dir.iterdir() if path.is_dir()):
                    files = sorted(version_dir.glob("*.nc"))
                    if not files:
                        continue
                    first_start, _, first_err = read_time_bounds(files[0])
                    _, last_end, last_err = read_time_bounds(files[-1])
                    meta = summarize_variable_file(files[0], variable_id)
                    note_parts = [part for part in [first_err, last_err, meta["attribute_error"]] if part]
                    rows.append(
                        {
                            "wusd3_dataset_id": parent.wusd3_dataset_id,
                            "source_id": parent.source_id,
                            "institution_id": parent.institution_id,
                            "member_id": parent.member_id,
                            "experiment_id": experiment_id,
                            "table_id": table_id,
                            "variable_id": variable_id,
                            "grid_label": grid_label or meta["grid_label"],
                            "version": version_dir.name,
                            "standard_name": meta["standard_name"],
                            "long_name": meta["long_name"],
                            "units": meta["units"],
                            "dimensions": meta["dimensions"],
                            "vertical_coordinate": meta["vertical_coordinate"],
                            "n_files": str(len(files)),
                            "time_start": first_start or "",
                            "time_end": last_end or "",
                            "coverage_ok_for_wusd3": str(
                                coverage_ok(experiment_id, first_start, last_end)
                            ),
                            "local_archive_status": "present_local_branch",
                            "note": " | ".join(note_parts),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parent_rows() -> list[dict[str, str]]:
    rows = []
    for parent in PARENT_RUNS:
        hist_start, hist_end, hist_n = list_wusd3_years(f"{parent.wusd3_dataset_id}_historical_bc")
        ssp_start, ssp_end, ssp_n = list_wusd3_years(f"{parent.wusd3_dataset_id}_ssp370_bc")
        rows.append(
            {
                "wusd3_dataset_id": parent.wusd3_dataset_id,
                "source_id": parent.source_id,
                "institution_id": parent.institution_id,
                "member_id": parent.member_id,
                "historical_experiment": parent.historical_experiment,
                "ssp370_experiment": parent.ssp370_experiment,
                "grid_label": "",
                "historical_wusd3_file_year_start": str(hist_start or ""),
                "historical_wusd3_file_year_end": str(hist_end or ""),
                "historical_wusd3_n_years": str(hist_n),
                "ssp370_wusd3_file_year_start": str(ssp_start or ""),
                "ssp370_wusd3_file_year_end": str(ssp_end or ""),
                "ssp370_wusd3_n_years": str(ssp_n),
                "mapping_confidence": parent.mapping_confidence,
                "evidence": parent.evidence,
                "note": parent.mapping_note,
            }
        )
    return rows


def build_shared_outputs(full_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    model_experiment_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in full_rows:
        if row["variable_id"]:
            key = (row["source_id"], row["member_id"], row["experiment_id"])
            model_experiment_rows[key].append(row)

    per_model_var: dict[tuple[str, str], dict[str, dict[str, list[dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in full_rows:
        if row["variable_id"]:
            model_key = (row["source_id"], row["member_id"])
            per_model_var[model_key][row["variable_id"]][row["experiment_id"]].append(row)

    matrix_map: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    shared_rows: list[dict[str, str]] = []
    all_models = [(p.source_id, p.member_id) for p in PARENT_RUNS]

    for variable_id in sorted({row["variable_id"] for row in full_rows if row["variable_id"]}):
        model_notes = {}
        all_four = True
        tables = set()
        standard_names = set()
        long_names = set()
        units = set()
        vertical_notes = set()
        grid_notes = set()
        hist_ok_all = True
        ssp_ok_all = True
        combined_ok_all = True

        for source_id, member_id in all_models:
            model_label = f"{source_id}:{member_id}"
            exp_map = per_model_var[(source_id, member_id)].get(variable_id, {})
            hist_rows = exp_map.get("historical", [])
            ssp_rows = exp_map.get("ssp370", [])
            hist_present = bool(hist_rows)
            ssp_present = bool(ssp_rows)
            hist_ok = any(row["coverage_ok_for_wusd3"] == "True" for row in hist_rows)
            ssp_ok = any(row["coverage_ok_for_wusd3"] == "True" for row in ssp_rows)
            hist_end = max((row["time_end"] for row in hist_rows if row["time_end"]), default=None)
            ssp_start = min((row["time_start"] for row in ssp_rows if row["time_start"]), default=None)
            ssp_end = max((row["time_end"] for row in ssp_rows if row["time_end"]), default=None)
            combined_ok = combined_water_year_ok(hist_end, ssp_start, ssp_end)

            note = []
            if hist_present:
                note.append("hist")
            if ssp_present:
                note.append("ssp370")
            if hist_ok:
                note.append("hist_ok")
            if ssp_ok:
                note.append("ssp_ok")
            if combined_ok:
                note.append("combined_ok")
            if not ssp_present:
                note.append("ssp_missing")
            if not hist_present:
                note.append("hist_missing")

            model_notes[model_label] = ",".join(note)
            matrix_map[variable_id][model_label] = {
                "available": "True" if hist_present or ssp_present else "False",
                "note": model_notes[model_label],
            }

            if not (hist_present and ssp_present and combined_ok):
                all_four = False
            hist_ok_all = hist_ok_all and hist_ok
            ssp_ok_all = ssp_ok_all and ssp_ok
            combined_ok_all = combined_ok_all and combined_ok

            for row in hist_rows + ssp_rows:
                if row["table_id"]:
                    tables.add(row["table_id"])
                if row["standard_name"]:
                    standard_names.add(row["standard_name"])
                if row["long_name"]:
                    long_names.add(row["long_name"])
                if row["units"]:
                    units.add(row["units"])
                if row["vertical_coordinate"]:
                    vertical_notes.add(f"{model_label}:{row['vertical_coordinate']}")
                if row["grid_label"]:
                    grid_notes.add(f"{model_label}:{row['grid_label']}")

        if all_four:
            shared_rows.append(
                {
                    "variable_id": variable_id,
                    "standard_name": " | ".join(sorted(standard_names)),
                    "long_name": " | ".join(sorted(long_names)),
                    "units": " | ".join(sorted(units)),
                    "available_models": "; ".join(sorted(model_notes)),
                    "tables_or_frequencies": " | ".join(sorted(tables)),
                    "historical_coverage_ok": str(hist_ok_all),
                    "ssp370_coverage_ok": str(ssp_ok_all),
                    "combined_water_year_coverage_ok": str(combined_ok_all),
                    "vertical_coordinate_notes": " | ".join(sorted(vertical_notes)),
                    "grid_difference_notes": " | ".join(sorted(grid_notes)),
                    "notes": json.dumps(model_notes, sort_keys=True),
                }
            )

    matrix_rows: list[dict[str, str]] = []
    model_columns = [f"{p.source_id}:{p.member_id}" for p in PARENT_RUNS]
    for variable_id in sorted(matrix_map):
        row = {"variable_id": variable_id}
        shared = True
        for column in model_columns:
            note = matrix_map[variable_id].get(column, {}).get("note", "")
            row[column] = note
            if "combined_ok" not in note:
                shared = False
        row["shared_all_four"] = str(shared)
        matrix_rows.append(row)
    return shared_rows, matrix_rows, matrix_map


def main() -> None:
    ensure_out_dir()

    parent_rows = build_parent_rows()
    write_csv(
        OUT_DIR / "wusd3_parent_runs.csv",
        parent_rows,
        [
            "wusd3_dataset_id",
            "source_id",
            "institution_id",
            "member_id",
            "historical_experiment",
            "ssp370_experiment",
            "grid_label",
            "historical_wusd3_file_year_start",
            "historical_wusd3_file_year_end",
            "historical_wusd3_n_years",
            "ssp370_wusd3_file_year_start",
            "ssp370_wusd3_file_year_end",
            "ssp370_wusd3_n_years",
            "mapping_confidence",
            "evidence",
            "note",
        ],
    )

    full_rows: list[dict[str, str]] = []
    for parent in PARENT_RUNS:
        full_rows.extend(collect_inventory(parent, "historical", parent.hist_source_path))
        full_rows.extend(collect_inventory(parent, "ssp370", parent.ssp_source_path))

    write_csv(
        OUT_DIR / "cmip6_full_inventory.csv",
        full_rows,
        [
            "wusd3_dataset_id",
            "source_id",
            "institution_id",
            "member_id",
            "experiment_id",
            "table_id",
            "variable_id",
            "grid_label",
            "version",
            "standard_name",
            "long_name",
            "units",
            "dimensions",
            "vertical_coordinate",
            "n_files",
            "time_start",
            "time_end",
            "coverage_ok_for_wusd3",
            "local_archive_status",
            "note",
        ],
    )

    shared_rows, matrix_rows, _ = build_shared_outputs(full_rows)
    write_csv(
        OUT_DIR / "cmip6_shared_variables.csv",
        shared_rows,
        [
            "variable_id",
            "standard_name",
            "long_name",
            "units",
            "available_models",
            "tables_or_frequencies",
            "historical_coverage_ok",
            "ssp370_coverage_ok",
            "combined_water_year_coverage_ok",
            "vertical_coordinate_notes",
            "grid_difference_notes",
            "notes",
        ],
    )

    write_csv(
        OUT_DIR / "cmip6_inventory_matrix.csv",
        matrix_rows,
        [
            "variable_id",
            *(f"{p.source_id}:{p.member_id}" for p in PARENT_RUNS),
            "shared_all_four",
        ],
    )

    summary = {
        "parent_runs": parent_rows,
        "n_full_rows": len(full_rows),
        "n_shared_variables": len(shared_rows),
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / "inventory_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
