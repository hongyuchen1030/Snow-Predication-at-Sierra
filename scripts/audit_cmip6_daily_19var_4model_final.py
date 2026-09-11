#!/usr/bin/env python3
"""Final simple 19x4 daily-availability audit for the 4 WUS-D3 parent CMIP6
models, reusing all metadata already gathered by the two prior audit passes
(no re-scanning of the archive; only a corrected re-analysis).

For each of the 11 unique physical variables x 4 models x {historical,ssp370},
restrict candidate tables to ones with a genuine fixed vertical coordinate
(plev for pressure fields, true depth for thetao; any table for surface
fields) -- this is what caught the MPI CFday hybrid-sigma false positive.
Then, for every candidate table, glob every chunk file on disk and parse the
CMIP6-standard start/end date range directly from each filename (no netCDF
open needed for this step -- fast and gives real interior-gap detection),
and report the LONGEST continuous sub-range (allowing <=32 day tolerance
between consecutive chunks) plus any gaps.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import date, timedelta
from pathlib import Path

CMIP6_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/CMIP6")

PARENT_ROOTS = {
    ("EC-Earth3", "historical"): CMIP6_ROOT / "CMIP/EC-Earth-Consortium/EC-Earth3/historical/r102i1p1f1",
    ("EC-Earth3", "ssp370"): CMIP6_ROOT / "ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp370/r102i1p1f1",
    ("MIROC6", "historical"): CMIP6_ROOT / "CMIP/MIROC/MIROC6/historical/r1i1p1f1",
    ("MIROC6", "ssp370"): CMIP6_ROOT / "ScenarioMIP/MIROC/MIROC6/ssp370/r1i1p1f1",
    ("MPI-ESM1-2-HR", "historical"): CMIP6_ROOT / "CMIP/MPI-M/MPI-ESM1-2-HR/historical/r3i1p1f1",
    ("MPI-ESM1-2-HR", "ssp370"): CMIP6_ROOT / "ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/r3i1p1f1",  # absent
    ("TaiESM1", "historical"): CMIP6_ROOT / "CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1",
    ("TaiESM1", "ssp370"): CMIP6_ROOT / "ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1",
}

RAW_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_daily_availability_audit/table_scan_raw.csv")

PRESSURE_VARS = {"zg", "ta", "ua", "va", "hus"}
SURFACE_VARS = {"rlut", "psl", "tas", "siconc", "tos"}
DEPTH_VARS = {"thetao"}
LAND_VARS = {"mrso"}

DATE_RE = re.compile(r"_(\d{6,8})-(\d{6,8})\.nc$")


def parse_fname_date(token: str) -> date:
    if len(token) == 8:
        return date(int(token[0:4]), int(token[4:6]), int(token[6:8]))
    if len(token) == 6:
        return date(int(token[0:4]), int(token[4:6]), 1)
    raise ValueError(token)


def chunk_ranges(files: list[Path]) -> list[tuple[date, date]]:
    out = []
    for f in files:
        m = DATE_RE.search(f.name)
        if not m:
            continue
        out.append((parse_fname_date(m.group(1)), parse_fname_date(m.group(2))))
    return sorted(out)


def all_segments(ranges: list[tuple[date, date]], tol_days: int = 32) -> tuple[list[tuple[date, date]], list[str]]:
    if not ranges:
        return [], []
    segments = []
    seg_start, seg_end = ranges[0]
    gaps = []
    for start, end in ranges[1:]:
        if (start - seg_end).days <= tol_days:
            seg_end = max(seg_end, end)
        else:
            segments.append((seg_start, seg_end))
            gaps.append(f"gap {seg_end}->{start}")
            seg_start, seg_end = start, end
    segments.append((seg_start, seg_end))
    return segments, gaps


def main() -> None:
    with RAW_CSV.open() as fh:
        raw_rows = list(csv.DictReader(fh))

    # (variable_id, model, experiment) -> list of (table, vertical_coord_kind)
    tables_by_key: dict[tuple, list[tuple[str, str | None]]] = {}
    for r in raw_rows:
        if r.get("frequency", "").lower() != "day":
            continue
        key = (r["variable_id"], r["model"], r["experiment"])
        tables_by_key.setdefault(key, []).append((r["table_id_dir"], r.get("vertical_coord_kind")))

    variables = sorted(PRESSURE_VARS | SURFACE_VARS | DEPTH_VARS | LAND_VARS)
    models = ["EC-Earth3", "MIROC6", "MPI-ESM1-2-HR", "TaiESM1"]

    results = {}
    for model in models:
        for exp in ("historical", "ssp370"):
            root = PARENT_ROOTS[(model, exp)]
            req_start, req_end = (date(1980, 9, 1), date(2014, 12, 31)) if exp == "historical" else (date(2015, 1, 1), date(2100, 8, 31))
            for var in variables:
                key = (var, model, exp)
                candidates = tables_by_key.get(key, [])
                best_choice = None  # (table, seg_start, seg_end, all_segments, gaps, n_files, overlaps_required)
                for table, vkind in candidates:
                    if var in PRESSURE_VARS and vkind != "pressure_pa":
                        continue
                    if var in DEPTH_VARS and vkind != "depth_m":
                        continue
                    base = root / table / var
                    if not base.exists():
                        continue
                    files: list[Path] = []
                    for grid_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                        version_dirs = sorted(p for p in grid_dir.iterdir() if p.is_dir())
                        if not version_dirs:
                            continue
                        latest_version = version_dirs[-1]
                        files = sorted(latest_version.glob("*.nc"))
                        if files:
                            break
                    ranges = chunk_ranges(files)
                    segments, gaps = all_segments(ranges)
                    if not segments:
                        continue
                    # Prefer the longest segment that actually overlaps the
                    # required water-year window; fall back to the longest
                    # segment overall if none overlap.
                    overlapping = [s for s in segments if s[0] <= req_end and s[1] >= req_start]
                    pool = overlapping if overlapping else segments
                    seg_start, seg_end = max(pool, key=lambda s: (s[1] - s[0]).days)
                    span_days = (seg_end - seg_start).days
                    candidate_rank = (bool(overlapping), span_days)
                    if best_choice is None or candidate_rank > best_choice["rank"]:
                        best_choice = {
                            "table": table, "start": seg_start, "end": seg_end,
                            "gaps": gaps, "n_files": len(files), "n_segments": len(segments),
                            "overlaps_required": bool(overlapping), "rank": candidate_rank,
                        }
                if best_choice is None:
                    results[key] = None
                else:
                    b = best_choice
                    results[key] = {
                        "table": b["table"], "start": str(b["start"]), "end": str(b["end"]),
                        "gaps": b["gaps"], "n_files": b["n_files"], "n_segments": b["n_segments"],
                        "overlaps_required": b["overlaps_required"],
                    }
                    flag = "OK" if b["overlaps_required"] else "NO-OVERLAP-WITH-NEEDED-WINDOW"
                    print(f"{model:16s} {exp:10s} {var:6s} -> {b['table']:8s} {b['start']}..{b['end']} "
                          f"n_files={b['n_files']} n_segments={b['n_segments']} {flag}", flush=True)

    out_path = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_daily_availability_audit/final_19x4_longest_continuous.json")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, out_path.open("w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
