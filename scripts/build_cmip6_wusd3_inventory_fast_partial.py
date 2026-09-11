#!/usr/bin/env python3
"""Build a fast partial CMIP6 inventory for the four WUS-D3 parent GCMs.

This mode is designed to keep progress moving when one local archive branch is
missing. It:
1. scans the local directory trees without opening NetCDF files,
2. derives time coverage from CMIP6 filenames,
3. writes placeholder rows for missing local branches, and
4. produces provisional shared-variable outputs that explicitly mark the
   missing MPI-ESM1-2-HR ssp370 branch.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


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
        hist_source_path=CMIP6_ROOT / "CMIP" / "EC-Earth-Consortium" / "EC-Earth3" / "historical" / "r102i1p1f1",
        ssp_source_path=CMIP6_ROOT / "ScenarioMIP" / "EC-Earth-Consortium" / "EC-Earth3" / "ssp370" / "r102i1p1f1",
        mapping_note=(
            "WUS ec-earth3_r1i1p1f1_2 is currently mapped to EC-Earth3 r102i1p1f1 "
            "by local member naming convention. This remains a high-confidence "
            "inference until direct provenance is confirmed."
        ),
        evidence="WUS-D3 dataset ids and local EC-Earth3 CMIP6 member directories",
        mapping_confidence="inferred_high",
    ),
    ParentRun(
        wusd3_dataset_id="miroc6_r1i1p1f1",
        source_id="MIROC6",
        institution_id="MIROC",
        member_id="r1i1p1f1",
        historical_experiment="historical",
        ssp370_experiment="ssp370",
        hist_source_path=CMIP6_ROOT / "CMIP" / "MIROC" / "MIROC6" / "historical" / "r1i1p1f1",
        ssp_source_path=CMIP6_ROOT / "ScenarioMIP" / "MIROC" / "MIROC6" / "ssp370" / "r1i1p1f1",
        mapping_note="Direct WUS-to-CMIP6 local name match.",
        evidence="WUS-D3 dataset ids and local MIROC6 CMIP6 directories",
        mapping_confidence="verified_local_name_match",
    ),
    ParentRun(
        wusd3_dataset_id="mpi-esm1-2-hr_r3i1p1f1",
        source_id="MPI-ESM1-2-HR",
        institution_id="MPI-M",
        member_id="r3i1p1f1",
        historical_experiment="historical",
        ssp370_experiment="ssp370",
        hist_source_path=CMIP6_ROOT / "CMIP" / "MPI-M" / "MPI-ESM1-2-HR" / "historical" / "r3i1p1f1",
        ssp_source_path=CMIP6_ROOT / "ScenarioMIP" / "MPI-M" / "MPI-ESM1-2-HR" / "ssp370" / "r3i1p1f1",
        mapping_note=(
            "Historical member is verified locally. The expected local CMIP6 "
            "ScenarioMIP MPI-ESM1-2-HR ssp370 r3i1p1f1 branch is currently missing."
        ),
        evidence=(
            "WUS-D3 dataset ids, local historical MPI-ESM1-2-HR branch, and absent "
            "standard ScenarioMIP ssp370 branch under the local mirror"
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
        hist_source_path=CMIP6_ROOT / "CMIP" / "AS-RCEC" / "TaiESM1" / "historical" / "r1i1p1f1",
        ssp_source_path=CMIP6_ROOT / "ScenarioMIP" / "AS-RCEC" / "TaiESM1" / "ssp370" / "r1i1p1f1",
        mapping_note="Direct WUS-to-CMIP6 local name match.",
        evidence="WUS-D3 dataset ids and local TaiESM1 CMIP6 directories",
        mapping_confidence="verified_local_name_match",
    ),
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


def parse_time_token(token: str) -> str | None:
    if len(token) == 6 and token.isdigit():
        return f"{token[0:4]}-{token[4:6]}-01"
    if len(token) == 8 and token.isdigit():
        return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"
    return None


def read_time_bounds_from_filename(nc_path: Path) -> tuple[str | None, str | None, str | None]:
    stem = nc_path.stem
    timerange = stem.split("_")[-1]
    if "-" not in timerange:
        return None, None, "filename_time_range_unparsed"
    start_token, end_token = timerange.split("-", 1)
    start = parse_time_token(start_token)
    end = parse_time_token(end_token)
    if start and end:
        return start, end, None
    return None, None, f"filename_time_range_unparsed:{timerange}"


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
        for variable_dir in sorted(path for path in table_dir.iterdir() if path.is_dir()):
            for grid_dir in sorted(path for path in variable_dir.iterdir() if path.is_dir()):
                for version_dir in sorted(path for path in grid_dir.iterdir() if path.is_dir()):
                    files = sorted(version_dir.glob("*.nc"))
                    if not files:
                        continue
                    first_start, _, first_err = read_time_bounds_from_filename(files[0])
                    _, last_end, last_err = read_time_bounds_from_filename(files[-1])
                    note_parts = [part for part in [first_err, last_err, "fast_partial_metadata_not_opened"] if part]
                    rows.append(
                        {
                            "wusd3_dataset_id": parent.wusd3_dataset_id,
                            "source_id": parent.source_id,
                            "institution_id": parent.institution_id,
                            "member_id": parent.member_id,
                            "experiment_id": experiment_id,
                            "table_id": table_dir.name,
                            "variable_id": variable_dir.name,
                            "grid_label": grid_dir.name,
                            "version": version_dir.name,
                            "standard_name": "",
                            "long_name": "",
                            "units": "",
                            "dimensions": "",
                            "vertical_coordinate": "",
                            "n_files": str(len(files)),
                            "time_start": first_start or "",
                            "time_end": last_end or "",
                            "coverage_ok_for_wusd3": str(coverage_ok(experiment_id, first_start, last_end)),
                            "local_archive_status": "present_local_branch",
                            "note": " | ".join(note_parts),
                        }
                    )
    return rows


def build_shared_outputs(full_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    per_model_var: dict[tuple[str, str], dict[str, dict[str, list[dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in full_rows:
        if row["variable_id"]:
            per_model_var[(row["source_id"], row["member_id"])][row["variable_id"]][row["experiment_id"]].append(row)

    all_models = [(p.source_id, p.member_id) for p in PARENT_RUNS]
    model_columns = [f"{p.source_id}:{p.member_id}" for p in PARENT_RUNS]
    shared_rows: list[dict[str, str]] = []
    matrix_rows: list[dict[str, str]] = []

    for variable_id in sorted({row["variable_id"] for row in full_rows if row["variable_id"]}):
        hist_ok_all = True
        ssp_ok_all = True
        combined_ok_all = True
        confirmed_all = True
        provisional_all = True
        tables = set()
        grid_notes = set()
        model_notes: dict[str, str] = {}

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
            missing_only_local_gap = source_id == "MPI-ESM1-2-HR" and hist_present and not ssp_present

            note_parts = []
            if hist_present:
                note_parts.append("hist")
            if ssp_present:
                note_parts.append("ssp370")
            if hist_ok:
                note_parts.append("hist_ok")
            if ssp_ok:
                note_parts.append("ssp_ok")
            if combined_ok:
                note_parts.append("combined_ok")
            if not ssp_present:
                note_parts.append("ssp_missing")
            if not hist_present:
                note_parts.append("hist_missing")
            if missing_only_local_gap:
                note_parts.append("pending_local_mpi_ssp370_gap")
            model_notes[model_label] = ",".join(note_parts)

            if not (hist_present and ssp_present and combined_ok):
                confirmed_all = False
            if not ((hist_present and ssp_present and combined_ok) or missing_only_local_gap):
                provisional_all = False

            hist_ok_all = hist_ok_all and hist_ok
            ssp_ok_all = ssp_ok_all and ssp_ok
            combined_ok_all = combined_ok_all and combined_ok

            for row in hist_rows + ssp_rows:
                if row["table_id"]:
                    tables.add(row["table_id"])
                if row["grid_label"]:
                    grid_notes.add(f"{model_label}:{row['grid_label']}")

        if confirmed_all or provisional_all:
            shared_rows.append(
                {
                    "variable_id": variable_id,
                    "standard_name": "",
                    "long_name": "",
                    "units": "",
                    "available_models": "; ".join(f"{k}={v}" for k, v in sorted(model_notes.items())),
                    "tables_or_frequencies": " | ".join(sorted(tables)),
                    "historical_coverage_ok": str(hist_ok_all),
                    "ssp370_coverage_ok": str(ssp_ok_all),
                    "combined_water_year_coverage_ok": str(combined_ok_all),
                    "confirmed_shared_all_four": str(confirmed_all),
                    "provisional_shared_pending_missing_mpi_ssp370": str(provisional_all and not confirmed_all),
                    "vertical_coordinate_notes": "",
                    "grid_difference_notes": " | ".join(sorted(grid_notes)),
                    "notes": json.dumps(model_notes, sort_keys=True),
                }
            )

        matrix_row = {"variable_id": variable_id}
        shared_all_four = True
        provisional_pending = False
        for column in model_columns:
            matrix_row[column] = model_notes.get(column, "")
            if "combined_ok" not in matrix_row[column]:
                shared_all_four = False
            if "pending_local_mpi_ssp370_gap" in matrix_row[column]:
                provisional_pending = True
        matrix_row["shared_all_four"] = str(shared_all_four)
        matrix_row["provisional_shared_pending_missing_mpi_ssp370"] = str(provisional_pending and not shared_all_four)
        matrix_rows.append(matrix_row)

    return shared_rows, matrix_rows


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

    shared_rows, matrix_rows = build_shared_outputs(full_rows)
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
            "confirmed_shared_all_four",
            "provisional_shared_pending_missing_mpi_ssp370",
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
            "provisional_shared_pending_missing_mpi_ssp370",
        ],
    )

    summary = {
        "scan_mode": "fast_partial_filename_based",
        "n_parent_runs": len(parent_rows),
        "n_full_inventory_rows": len(full_rows),
        "n_shared_rows": len(shared_rows),
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / "inventory_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
