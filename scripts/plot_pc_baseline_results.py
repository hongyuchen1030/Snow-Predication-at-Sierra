from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_PC1_COLUMNS = ("water_year", "pc1_true", "pc1_pred")
PC2_COLUMNS = ("pc2_true", "pc2_pred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved PC baseline predictions.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts") / "pc_baseline",
        help="Directory containing predictions.csv and optional metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to --input-dir.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_PC1_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} is missing required PC1 columns: {missing}. "
                f"Found columns: {fieldnames}"
            )

        rows: list[dict[str, float]] = []
        for row_index, raw in enumerate(reader, start=2):
            parsed: dict[str, float] = {}
            for name, value in raw.items():
                if value is None or value == "":
                    parsed[name] = float("nan")
                    continue
                try:
                    parsed[name] = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Could not parse numeric value in {path}:{row_index}, "
                        f"column {name!r}: {value!r}"
                    ) from exc
            rows.append(parsed)

    rows.sort(key=lambda item: int(item["water_year"]))
    return rows


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"metrics.json not found at {path}; plotting without metrics", flush=True)
        return {}
    with path.open() as handle:
        return json.load(handle)


def plot_time_series(
    rows: list[dict[str, float]],
    *,
    pc_name: str,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    true_name = f"{pc_name}_true"
    pred_name = f"{pc_name}_pred"
    clean = _drop_nan_rows(rows, [true_name, pred_name], label=f"{pc_name} trend")
    years = _column(clean, "water_year")
    true = _column(clean, true_name)
    pred = _column(clean, pred_name)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(years, true, marker="o", label=f"{pc_name.upper()} true")
    ax.plot(years, pred, marker="o", label=f"{pc_name.upper()} predicted")
    ax.set_title(_title_with_metrics(f"True vs predicted {pc_name.upper()} over time", pc_name, metrics))
    ax.set_xlabel("Water year")
    ax.set_ylabel(f"{pc_name.upper()} value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter(
    rows: list[dict[str, float]],
    *,
    pc_name: str,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    true_name = f"{pc_name}_true"
    pred_name = f"{pc_name}_pred"
    clean = _drop_nan_rows(rows, [true_name, pred_name], label=f"{pc_name} scatter")
    true = _column(clean, true_name)
    pred = _column(clean, pred_name)
    lo = float(np.nanmin(np.concatenate([true, pred])))
    hi = float(np.nanmax(np.concatenate([true, pred])))

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(true, pred)
    ax.plot([lo, hi], [lo, hi], label="1:1 reference")
    ax.set_title(_title_with_metrics(f"{pc_name.upper()} true vs predicted", pc_name, metrics))
    ax.set_xlabel(f"{pc_name.upper()} true")
    ax.set_ylabel(f"{pc_name.upper()} predicted")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    rows: list[dict[str, float]],
    *,
    pc_name: str,
    output_path: Path,
) -> None:
    true_name = f"{pc_name}_true"
    pred_name = f"{pc_name}_pred"
    clean = _drop_nan_rows(rows, [true_name, pred_name], label=f"{pc_name} residual")
    years = _column(clean, "water_year")
    residual = _column(clean, pred_name) - _column(clean, true_name)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(years, residual, marker="o", label=f"{pc_name.upper()} residual")
    ax.axhline(0.0, label="zero error")
    ax.set_title(f"{pc_name.upper()} residual over time")
    ax.set_xlabel("Water year")
    ax.set_ylabel(f"{pc_name.upper()} predicted - true")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_predictions(input_dir / "predictions.csv")
    metrics = load_metrics(input_dir / "metrics.json")
    has_pc2 = all(name in rows[0] for name in PC2_COLUMNS) if rows else False

    print_summary(rows, has_pc2=has_pc2, metrics=metrics)

    plot_time_series(rows, pc_name="pc1", metrics=metrics, output_path=output_dir / "pc1_trend.png")
    plot_scatter(rows, pc_name="pc1", metrics=metrics, output_path=output_dir / "pc1_scatter.png")
    plot_residuals(rows, pc_name="pc1", output_path=output_dir / "pc1_residual.png")

    if has_pc2:
        plot_time_series(rows, pc_name="pc2", metrics=metrics, output_path=output_dir / "pc2_trend.png")
        plot_scatter(rows, pc_name="pc2", metrics=metrics, output_path=output_dir / "pc2_scatter.png")
        plot_residuals(rows, pc_name="pc2", output_path=output_dir / "pc2_residual.png")
    else:
        print("PC2 columns not found; skipped PC2 plots", flush=True)

    print(f"saved plots to {output_dir}", flush=True)


def print_summary(
    rows: list[dict[str, float]],
    *,
    has_pc2: bool,
    metrics: dict[str, Any],
) -> None:
    if not rows:
        raise ValueError("predictions.csv contains no data rows")
    years = [int(row["water_year"]) for row in rows]
    print(f"year range: {min(years)}-{max(years)}", flush=True)
    print(f"rows loaded: {len(rows)}", flush=True)
    print(f"PC2 found: {has_pc2}", flush=True)
    for pc_name in ("pc1", "pc2"):
        if pc_name in metrics:
            print(f"{pc_name.upper()} metrics: {metrics[pc_name]}", flush=True)


def _drop_nan_rows(
    rows: list[dict[str, float]],
    columns: list[str],
    *,
    label: str,
) -> list[dict[str, float]]:
    clean = [
        row for row in rows
        if all(np.isfinite(row.get(column, float("nan"))) for column in columns)
    ]
    dropped = len(rows) - len(clean)
    if dropped:
        print(f"warning: dropped {dropped} rows with NaNs for {label}", flush=True)
    if not clean:
        raise ValueError(f"No finite rows available for {label}")
    return clean


def _column(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=np.float64)


def _title_with_metrics(base: str, pc_name: str, metrics: dict[str, Any]) -> str:
    values = metrics.get(pc_name)
    if not isinstance(values, dict):
        return base
    corr = values.get("pearson_correlation")
    rmse = values.get("rmse")
    parts = []
    if corr is not None:
        parts.append(f"r={float(corr):.3g}")
    if rmse is not None:
        parts.append(f"RMSE={float(rmse):.3g}")
    if not parts:
        return base
    return f"{base} ({', '.join(parts)})"


if __name__ == "__main__":
    main()
