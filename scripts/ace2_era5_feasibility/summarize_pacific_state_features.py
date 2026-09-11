#!/usr/bin/env python3
"""Combine per-state ACE2 feature rows into one comparison table and summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import math

import pandas as pd


TARGET_PREFIXES = [
    "SIERRA_BOX__PRATEsfc",
    "CA_NV_BOX__PRATEsfc",
    "WEST_US_BOX__PRATEsfc",
    "SIERRA_BOX__TMP2m",
    "CA_NV_BOX__TMP2m",
    "WEST_US_BOX__TMP2m",
    "SIERRA_BOX__VGRD10m",
    "CA_NV_BOX__VGRD10m",
    "WEST_US_BOX__VGRD10m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--reversed", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_row(path: Path, state: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {path}")
    df = df.copy()
    df.insert(0, "SST_state", state)
    return df


def diff_row(lhs: pd.Series, rhs: pd.Series, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"SST_state": label}
    for column in lhs.index:
        if column == "SST_state":
            continue
        if pd.api.types.is_numeric_dtype(type(lhs[column])) and pd.api.types.is_numeric_dtype(type(rhs[column])):
            out[column] = float(lhs[column] - rhs[column])
        else:
            out[column] = ""
    return out


def main() -> None:
    args = parse_args()
    actual = load_row(args.actual, "actual")
    neutral = load_row(args.neutral, "neutral")
    reversed_df = load_row(args.reversed, "reversed")

    combined = pd.concat([actual, neutral, reversed_df], ignore_index=True)
    actual_row = combined.iloc[0]
    neutral_row = combined.iloc[1]
    reversed_row = combined.iloc[2]

    diff_rows = pd.DataFrame(
        [
            diff_row(actual_row, neutral_row, "diff_actual_minus_neutral"),
            diff_row(reversed_row, neutral_row, "diff_reversed_minus_neutral"),
            diff_row(actual_row, reversed_row, "diff_actual_minus_reversed"),
        ]
    )
    final_df = pd.concat([combined, diff_rows], ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    summary = {
        "states": ["actual", "neutral", "reversed"],
        "source_feature_files": {
            "actual": str(args.actual),
            "neutral": str(args.neutral),
            "reversed": str(args.reversed),
        },
        "state_validity": {},
        "selected_metrics": {},
    }
    for _, row in combined.iterrows():
        state = row["SST_state"]
        finite_metric_count = 0
        nan_metric_count = 0
        for column, value in row.items():
            if column == "SST_state":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isnan(value):
                    nan_metric_count += 1
                else:
                    finite_metric_count += 1
        summary["state_validity"][state] = {
            "finite_numeric_metric_count": finite_metric_count,
            "nan_numeric_metric_count": nan_metric_count,
            "all_tracked_numeric_metrics_nan": finite_metric_count == 0 and nan_metric_count > 0,
        }
    for prefix in TARGET_PREFIXES:
        metric_name = f"{prefix}__mean_over_available_time"
        if metric_name in final_df.columns:
            summary["selected_metrics"][metric_name] = {
                row["SST_state"]: row[metric_name]
                for _, row in final_df.iterrows()
                if pd.notna(row[metric_name]) and row["SST_state"]
            }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(final_df.to_csv(index=False).strip())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
