#!/usr/bin/env python3
"""Inspect ACE2 forcing, initial-condition, and output datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


ALIASES = {
    "precipitation": ["PRATEsfc", "precip", "precipitation", "tp"],
    "temperature": ["TMP2m", "t2m", "tas", "temperature"],
    "pressure": ["PRESsfc", "SLP", "PRMSL", "msl", "sp"],
    "geopotential": ["HGT500", "Z500", "z", "gh"],
    "u_wind": ["UGRD10m", "UGRD", "u10", "ua", "u"],
    "v_wind": ["VGRD10m", "VGRD", "v10", "va", "v"],
    "humidity": ["Q2m", "RH2m", "q", "r", "specific_humidity", "relative_humidity"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forcing", type=Path, nargs="*", default=[])
    parser.add_argument("--initial-condition", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--report-txt",
        type=Path,
        required=True,
        help="Path to the human-readable text report.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        required=True,
        help="Path to the machine-readable JSON report.",
    )
    return parser.parse_args()


def infer_timestep(ds: xr.Dataset) -> str | None:
    if "time" not in ds.coords or ds.sizes.get("time", 0) < 2:
        return None
    values = ds.indexes["time"]
    delta = values[1] - values[0]
    return str(delta)


def find_aliases(variable_names: list[str]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    lowered = {name.lower(): name for name in variable_names}
    for label, aliases in ALIASES.items():
        match = None
        for alias in aliases:
            if alias.lower() in lowered:
                match = lowered[alias.lower()]
                break
        mapping[label] = match
    return mapping


def summarize_dataset(path: Path) -> dict[str, Any]:
    with xr.open_dataset(path) as ds:
        data_vars = list(ds.data_vars)
        coords = list(ds.coords)
        dims = {name: int(size) for name, size in ds.sizes.items()}
        summary = {
            "path": str(path),
            "data_vars": data_vars,
            "coords": coords,
            "dims": dims,
            "time_step": infer_timestep(ds),
            "aliases": find_aliases(data_vars),
            "var_dims": {name: list(ds[name].dims) for name in data_vars},
            "coord_preview": {},
        }
        for coord in coords:
            values = ds[coord].values
            if np.issubdtype(np.asarray(values).dtype, np.datetime64):
                preview = [str(v) for v in values[: min(3, len(values))]]
            else:
                preview = np.asarray(values[: min(3, len(values))]).tolist()
            summary["coord_preview"][coord] = preview
        return summary


def build_report(paths: list[Path]) -> list[dict[str, Any]]:
    return [summarize_dataset(path) for path in paths]


def render_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_name in ("forcing", "initial_condition", "output"):
        lines.append(f"[{section_name}]")
        entries = report.get(section_name, [])
        if not entries:
            lines.append("  none")
            lines.append("")
            continue
        for entry in entries:
            lines.append(f"  path: {entry['path']}")
            lines.append(f"  dims: {entry['dims']}")
            lines.append(f"  coords: {entry['coords']}")
            lines.append(f"  data_vars ({len(entry['data_vars'])}): {entry['data_vars']}")
            lines.append(f"  time_step: {entry['time_step']}")
            lines.append(f"  aliases: {entry['aliases']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = {
        "forcing": build_report(args.forcing),
        "initial_condition": build_report(args.initial_condition),
        "output": build_report(args.output),
    }

    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.write_text(render_text_report(report), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render_text_report(report))


if __name__ == "__main__":
    main()
