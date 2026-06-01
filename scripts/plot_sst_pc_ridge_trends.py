#!/usr/bin/env python3
"""
Plot SST-PC ridge actual vs predicted SWE PC trends.
"""

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = PROJECT_ROOT / "artifacts" / "sst_pc_ridge" / "predictions.csv"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "sst_pc_ridge" / "plots"

TARGETS = ("PC1", "PC2")
LEADS = (1, 2, 3)


def load_predictions(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "model": row["model"],
                    "lead_months": int(row["lead_months"]),
                    "target_pc": row["target_pc"],
                    "year": int(row["year"]),
                    "y_true": float(row["y_true"]),
                    "y_pred": float(row["y_pred"]),
                }
            )
    if not rows:
        raise ValueError("No prediction rows found in {}".format(path))
    return rows


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def safe_metric_text(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return "{:.3f}".format(value)


def sort_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda row: (int(row["year"]), str(row["model"])))


def aggregate_pooled_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)  # type: Dict[int, List[Dict[str, object]]]
    for row in rows:
        grouped[int(row["year"])].append(row)

    pooled = []
    for year in sorted(grouped):
        year_rows = grouped[year]
        pooled.append(
            {
                "model": "ALL",
                "lead_months": int(year_rows[0]["lead_months"]),
                "target_pc": str(year_rows[0]["target_pc"]),
                "year": year,
                "y_true": float(np.mean([float(item["y_true"]) for item in year_rows])),
                "y_pred": float(np.mean([float(item["y_pred"]) for item in year_rows])),
            }
        )
    return pooled


def plot_rows(rows: Sequence[Dict[str, object]], *, title_prefix: str, output_path: Path) -> Tuple[float, float]:
    ordered = sort_rows(rows)
    years = np.array([int(row["year"]) for row in ordered], dtype=np.int32)
    y_true = np.array([float(row["y_true"]) for row in ordered], dtype=np.float64)
    y_pred = np.array([float(row["y_pred"]) for row in ordered], dtype=np.float64)
    corr = correlation(y_true, y_pred)
    r2 = r2_score_manual(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(years, y_true, color="#1f4e79", linewidth=2.0, label="Actual")
    ax.plot(years, y_pred, color="#d95f02", linewidth=2.0, label="Predicted")
    ax.set_xlabel("Year")
    ax.set_ylabel("SWE PC value")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    ax.set_title(
        "{}\nCorr={}, R²={}".format(
            title_prefix,
            safe_metric_text(corr),
            safe_metric_text(r2),
        )
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return corr, r2


def main() -> None:
    rows = load_predictions(PREDICTIONS_FILE)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    models = sorted({str(row["model"]) for row in rows})
    generated_files = []
    metric_rows = []

    for lead_months in LEADS:
        for target_pc in TARGETS:
            subset = [
                row for row in rows
                if int(row["lead_months"]) == lead_months and str(row["target_pc"]) == target_pc
            ]
            pooled_rows = aggregate_pooled_rows(subset)
            pooled_output = PLOTS_DIR / "all_models_{}_lead{}.png".format(target_pc.lower(), lead_months)
            corr, r2 = plot_rows(
                pooled_rows,
                title_prefix="All Models Mean Trend: {} lead {}".format(target_pc, lead_months),
                output_path=pooled_output,
            )
            generated_files.append(pooled_output)
            metric_rows.append(
                {
                    "plot_type": "ALL",
                    "lead_months": lead_months,
                    "target_pc": target_pc,
                    "correlation": corr,
                    "r2": r2,
                    "output_path": str(pooled_output),
                }
            )

            for model_name in models:
                model_rows = [
                    row for row in subset
                    if str(row["model"]) == model_name
                ]
                output_path = PLOTS_DIR / "{}_{}_lead{}.png".format(
                    model_name,
                    target_pc.lower(),
                    lead_months,
                )
                corr, r2 = plot_rows(
                    model_rows,
                    title_prefix="{}: {} lead {}".format(model_name, target_pc, lead_months),
                    output_path=output_path,
                )
                generated_files.append(output_path)
                metric_rows.append(
                    {
                        "plot_type": model_name,
                        "lead_months": lead_months,
                        "target_pc": target_pc,
                        "correlation": corr,
                        "r2": r2,
                        "output_path": str(output_path),
                    }
                )

    best_row = None
    best_score = -math.inf
    for row in metric_rows:
        if row["plot_type"] != "ALL":
            continue
        corr = float(row["correlation"])
        if np.isfinite(corr) and corr > best_score:
            best_score = corr
            best_row = row

    print("Generated {} plot files in {}".format(len(generated_files), PLOTS_DIR), flush=True)
    for path in generated_files:
        print(path, flush=True)
    if best_row is not None:
        print(
            "Best pooled visual match: {} lead {} (corr={}, r2={})".format(
                best_row["target_pc"],
                best_row["lead_months"],
                safe_metric_text(float(best_row["correlation"])),
                safe_metric_text(float(best_row["r2"])),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
