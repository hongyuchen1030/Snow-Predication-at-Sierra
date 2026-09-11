#!/usr/bin/env python3
"""Compare ACE2 ensemble members for a short same-IC inference test."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


TARGET_VARS = ("PRATEsfc", "TMP2m", "VGRD10m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing monthly_mean_predictions.nc or per-member subdirectories.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def open_member_arrays(input_dir: Path) -> tuple[dict[str, xr.Dataset], dict[str, Any]]:
    monthly_path = input_dir / "monthly_mean_predictions.nc"
    if monthly_path.exists():
        ds = xr.open_dataset(monthly_path)
        if "sample" not in ds.dims:
            raise ValueError(f"{monthly_path} has no sample dimension to compare.")
        members = {
            f"member_{sample_idx:03d}": ds.isel(sample=sample_idx).load()
            for sample_idx in range(int(ds.sizes["sample"]))
        }
        inventory = {
            "mode": "single_file_sample_dimension",
            "monthly_mean_predictions_path": str(monthly_path),
            "n_members": len(members),
        }
        ds.close()
        return members, inventory

    member_files = sorted(input_dir.glob("member_*/monthly_mean_predictions.nc"))
    if not member_files:
        raise FileNotFoundError(
            f"No monthly_mean_predictions.nc found in {input_dir} or member_* subdirectories."
        )
    members = {
        path.parent.name: xr.open_dataset(path).load()
        for path in member_files
    }
    inventory = {
        "mode": "per_member_directories",
        "member_paths": {path.parent.name: str(path) for path in member_files},
        "n_members": len(members),
    }
    return members, inventory


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    if np.allclose(a, a.flat[0]) or np.allclose(b, b.flat[0]):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_difference(
    a: xr.DataArray,
    b: xr.DataArray,
    tolerance: float,
) -> dict[str, Any]:
    av = np.asarray(a.values, dtype=np.float64).ravel()
    bv = np.asarray(b.values, dtype=np.float64).ravel()
    diff = av - bv
    return {
        "shape": list(a.shape),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "mean_abs_difference": float(np.mean(np.abs(diff))),
        "rmse_difference": float(np.sqrt(np.mean(diff**2))),
        "correlation": correlation(av, bv),
        "exactly_identical": bool(np.array_equal(av, bv)),
        "identical_within_tolerance": bool(np.allclose(av, bv, atol=tolerance, rtol=0.0)),
    }


def compare_members(
    members: dict[str, xr.Dataset],
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "member_names": list(members.keys()),
        "variables": {},
        "pairwise_results": [],
    }

    for left_name, right_name in combinations(members.keys(), 2):
        left_ds = members[left_name]
        right_ds = members[right_name]
        pair_entry = {"member_a": left_name, "member_b": right_name, "variables": {}}
        for var_name in TARGET_VARS:
            if var_name not in left_ds.data_vars or var_name not in right_ds.data_vars:
                continue
            stats = summarize_difference(left_ds[var_name], right_ds[var_name], tolerance)
            row = {
                "member_a": left_name,
                "member_b": right_name,
                "variable": var_name,
                **stats,
            }
            rows.append(row)
            pair_entry["variables"][var_name] = stats
            summary["variables"].setdefault(var_name, []).append(row)
        summary["pairwise_results"].append(pair_entry)

    return rows, summary


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        members, inventory = open_member_arrays(args.input_dir)
        rows, summary = compare_members(members, tolerance=args.tolerance)
        report = {
            "status": "ok",
            "input_dir": str(args.input_dir),
            "inventory": inventory,
            "tolerance": args.tolerance,
            "summary": summary,
        }
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(pd.DataFrame(rows).to_csv(index=False).strip())
        print(json.dumps(report, indent=2))
    except Exception as exc:  # pragma: no cover - failure reporting path
        failure_row = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        report = {
            "status": "error",
            "input_dir": str(args.input_dir),
            "tolerance": args.tolerance,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        pd.DataFrame([failure_row]).to_csv(args.output_csv, index=False)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(pd.DataFrame([failure_row]).to_csv(index=False).strip())
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
