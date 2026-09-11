#!/usr/bin/env python3
"""Simple per-variable, per-model, ALL-years daily-availability listing for
TaiESM1 and MPI-ESM1-2-HR, restricted to the 12 selected predictors. Reuses
the same file-glob convention (single grid_label, latest version only) and
the same genuine-fixed-pressure-level table choices already verified in the
prior audits. Every disconnected chunk of years is reported; none are
discarded or reduced to "longest continuous."
"""
from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

CMIP6_ROOT = Path("/global/cfs/projectdirs/m3522/cmip6/CMIP6")

ROOTS = {
    ("TaiESM1", "historical"): CMIP6_ROOT / "CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1",
    ("TaiESM1", "ssp370"): CMIP6_ROOT / "ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1",
    ("MPI-ESM1-2-HR", "historical"): CMIP6_ROOT / "CMIP/MPI-M/MPI-ESM1-2-HR/historical/r3i1p1f1",
    ("MPI-ESM1-2-HR", "ssp370"): CMIP6_ROOT / "ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/r3i1p1f1",  # absent on disk
}

# field -> (variable_id, table) using only the already-verified genuine
# fixed-pressure-level (non-hybrid-sigma) table for each model.
FIELD_TABLE = {
    "TaiESM1": {
        "rlut": ("rlut", "day"), "zg_500": ("zg", "day"), "zg_50": ("zg", "day"),
        "ta_850": ("ta", "day"), "ta_50": ("ta", "day"),
        "ua_850": ("ua", "day"), "ua_50": ("ua", "day"),
        "va_850": ("va", "day"), "va_50": ("va", "day"),
        "hus_850": ("hus", "day"), "psl": ("psl", "day"), "tas": ("tas", "day"),
    },
    "MPI-ESM1-2-HR": {
        "rlut": ("rlut", "day"), "zg_500": ("zg", "EdayZ"), "zg_50": ("zg", "EdayZ"),
        "ta_850": ("ta", "Eday"), "ta_50": ("ta", "Eday"),
        "ua_850": ("ua", "day"), "ua_50": ("ua", "day"),  # CFday hybrid-sigma excluded
        "va_850": ("va", "EdayZ"), "va_50": ("va", "EdayZ"),
        "hus_850": ("hus", "EdayZ"), "psl": ("psl", "day"), "tas": ("tas", "day"),
    },
}

FIELD_ORDER = ["rlut", "zg_500", "zg_50", "ta_850", "ta_50", "ua_850", "ua_50",
               "va_850", "va_50", "hus_850", "psl", "tas"]
MODELS = ["TaiESM1", "MPI-ESM1-2-HR"]

DATE_RE = re.compile(r"_(\d{6,8})-(\d{6,8})\.nc$")


def parse_fname_date(token: str) -> date:
    if len(token) == 8:
        return date(int(token[0:4]), int(token[4:6]), int(token[6:8]))
    return date(int(token[0:4]), int(token[4:6]), 1)


def chunk_ranges(files: list[Path]) -> list[tuple[date, date]]:
    out = []
    for f in files:
        m = DATE_RE.search(f.name)
        if m:
            out.append((parse_fname_date(m.group(1)), parse_fname_date(m.group(2))))
    return sorted(out)


def merge_segments(ranges: list[tuple[date, date]], tol_days: int = 32) -> list[tuple[int, int]]:
    if not ranges:
        return []
    segments = []
    seg_start, seg_end = ranges[0]
    for start, end in ranges[1:]:
        if (start - seg_end).days <= tol_days:
            seg_end = max(seg_end, end)
        else:
            segments.append((seg_start.year, seg_end.year))
            seg_start, seg_end = start, end
    segments.append((seg_start.year, seg_end.year))
    return segments


def years_for(model: str, exp: str, variable_id: str, table: str) -> list[tuple[int, int]]:
    root = ROOTS[(model, exp)]
    base = root / table / variable_id
    if not base.exists():
        return []
    files: list[Path] = []
    for grid_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        version_dirs = sorted(p for p in grid_dir.iterdir() if p.is_dir())
        if not version_dirs:
            continue
        latest_version = version_dirs[-1]
        files = sorted(latest_version.glob("*.nc"))
        if files:
            break
    return merge_segments(chunk_ranges(files))


def format_ranges(segments: list[tuple[int, int]]) -> str:
    if not segments:
        return "NONE"
    parts = []
    for a, b in segments:
        parts.append(str(a) if a == b else f"{a}–{b}")
    return ", ".join(parts)


def main() -> None:
    rows = []
    print(f"| Model | Variable | Available years |")
    print(f"|-------|----------|-----------------|")
    for model in MODELS:
        for field in FIELD_ORDER:
            variable_id, table = FIELD_TABLE[model][field]
            hist_segments = years_for(model, "historical", variable_id, table)
            ssp_segments = years_for(model, "ssp370", variable_id, table)
            hist_str = format_ranges(hist_segments)
            ssp_str = format_ranges(ssp_segments)
            if hist_str == "NONE" and ssp_str == "NONE":
                cell = "NONE"
            elif ssp_str == "NONE":
                cell = f"historical: {hist_str}"
            elif hist_str == "NONE":
                cell = f"ssp370: {ssp_str}"
            else:
                cell = f"historical: {hist_str}; ssp370: {ssp_str}"
            print(f"| {model} | {field} | {cell} |")
            rows.append({
                "model": model, "variable": field, "table": table,
                "historical_years": hist_str, "ssp370_years": ssp_str,
                "combined": cell,
            })

    csv_path = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/docs/TaiESM1_MPI_12Var_Daily_Years.csv")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "variable", "table", "historical_years", "ssp370_years", "combined"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
