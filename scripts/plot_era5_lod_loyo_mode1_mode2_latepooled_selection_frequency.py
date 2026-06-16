#!/usr/bin/env python3
"""
Create an ERA5 LOYO LOD mode-stability plot with mode 1, mode 2, and pooled modes 3-6.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = Path(
    os.environ.get(
        "ERA5_LOYO_FOLD_MODES_CSV",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup/loyo_lod_analysis/era5_sierra_swe_lod_loyo_fold_modes.csv",
    )
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get(
        "ERA5_LOYO_MODE_STABILITY_OUTPUT_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup/loyo_lod_analysis",
    )
).expanduser()

PACIFIC_LAT_MIN = -10.0
PACIFIC_LAT_MAX = 60.0
PACIFIC_LON_MIN = 120.0
PACIFIC_LON_MAX = 280.0
TOTAL_FOLDS = 37


def load_selected_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            selected_value = str(row.get("selected", "")).strip().lower()
            if selected_value != "true":
                continue
            rows.append(
                {
                    "held_out_water_year": int(row["held_out_water_year"]),
                    "mode_number": int(row["mode_number"]),
                    "latitude": float(row["latitude"]),
                    "longitude_0_360": float(row["longitude_0_360"]),
                }
            )
    return rows


def count_locations(rows: list[dict[str, object]]) -> dict[tuple[float, float], int]:
    counts: dict[tuple[float, float], int] = {}
    for row in rows:
        key = (float(row["latitude"]), float(row["longitude_0_360"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def counts_to_rows(group: str, counts: dict[tuple[float, float], int], denominator: int) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for (latitude, longitude), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        output_rows.append(
            {
                "group": group,
                "latitude_bin_or_selected_latitude": latitude,
                "longitude_bin_or_selected_longitude": longitude,
                "selection_count": count,
                "fraction_of_possible_folds": count / denominator if denominator > 0 else float("nan"),
            }
        )
    return output_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "group",
        "latitude_bin_or_selected_latitude",
        "longitude_bin_or_selected_longitude",
        "selection_count",
        "fraction_of_possible_folds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def panel_scatter(ax: plt.Axes, counts: dict[tuple[float, float], int], title: str, vmax: int):
    ax.set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
    ax.set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
    ax.set_xticks(np.arange(120.0, 281.0, 20.0))
    ax.set_yticks(np.arange(-10.0, 61.0, 10.0))
    ax.set_xlabel("Longitude (0 to 360)")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.25, color="0.82")
    ax.set_title(title)
    ax.set_facecolor("white")
    if not counts:
        return None

    lats = np.array([key[0] for key in counts], dtype=np.float64)
    lons = np.array([key[1] for key in counts], dtype=np.float64)
    freqs = np.array([counts[key] for key in counts], dtype=np.float64)
    sizes = 36.0 + 34.0 * np.sqrt(freqs)
    scatter = ax.scatter(
        lons,
        lats,
        c=freqs,
        s=sizes,
        cmap="viridis",
        vmin=0.0,
        vmax=float(max(vmax, 1)),
        edgecolors="black",
        linewidths=0.45,
    )
    return scatter


def make_plot(path_png: Path, path_pdf: Path, grouped_counts: list[tuple[str, dict[tuple[float, float], int]]]) -> None:
    vmax = 0
    for _, counts in grouped_counts:
        if counts:
            vmax = max(vmax, max(counts.values()))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    for ax, (title, counts) in zip(axes, grouped_counts):
        scatter = panel_scatter(ax, counts, title, vmax)
        if scatter is not None:
            fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.04, label="selection count")
    fig.savefig(path_png, dpi=240)
    fig.savefig(path_pdf)
    plt.close(fig)


def write_note(path: Path, pooled_total: int, mode1_counts: dict[tuple[float, float], int], mode2_counts: dict[tuple[float, float], int], pooled_counts: dict[tuple[float, float], int]) -> None:
    mode1_top = max(mode1_counts.values()) if mode1_counts else 0
    mode2_top = max(mode2_counts.values()) if mode2_counts else 0
    pooled_top = max(pooled_counts.values()) if pooled_counts else 0
    pooled_behavior = "recurring regions" if pooled_top >= 5 else "scattered later-mode behavior"
    mode1_stability = "rank-stable" if mode1_top >= 0.5 * TOTAL_FOLDS else "not strongly rank-stable"
    mode2_stability = (
        "more rank-stable than later modes, although less spatially concentrated than mode 1"
        if mode2_top >= 0.3 * TOTAL_FOLDS
        else "not strongly rank-stable at the exact-gridcell level"
    )
    lines = [
        "# ERA5 LOYO Mode-Stability Interpretation",
        "",
        f"- Mode 1 is shown separately because it appears {mode1_stability} across LOYO folds.",
        f"- Mode 2 is shown separately because it appears {mode2_stability} across LOYO folds.",
        "- Modes 3--6 are pooled because later LOD modes may reorder across LOYO folds even when they reflect related residual SST signals.",
        "- The pooled panel therefore asks whether later selected modes repeatedly return to the same broad SST regions after allowing rank reordering.",
        "- Clear clusters in the pooled panel indicate later residual signals that are spatially recurring but not necessarily rank-stable.",
        "- Scattered points in the pooled panel indicate unstable later-mode selection.",
        f"- In this ERA5 LOYO result, the pooled modes 3--6 panel is most consistent with **{pooled_behavior}**.",
        f"- The pooled modes 3--6 denominator for `fraction_of_possible_folds` is the total number of actually selected later modes across all folds: `{pooled_total}`.",
        "- The plotted locations come directly from the saved LOYO fold-mode table; no LOD rerun was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_selected_rows(INPUT_CSV)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mode1_rows = [row for row in rows if int(row["mode_number"]) == 1]
    mode2_rows = [row for row in rows if int(row["mode_number"]) == 2]
    pooled_late_rows = [row for row in rows if 3 <= int(row["mode_number"]) <= 6]

    mode1_counts = count_locations(mode1_rows)
    mode2_counts = count_locations(mode2_rows)
    pooled_late_counts = count_locations(pooled_late_rows)

    csv_rows: list[dict[str, object]] = []
    csv_rows.extend(counts_to_rows("mode1", mode1_counts, TOTAL_FOLDS))
    csv_rows.extend(counts_to_rows("mode2", mode2_counts, TOTAL_FOLDS))
    csv_rows.extend(counts_to_rows("modes3to6_pooled", pooled_late_counts, len(pooled_late_rows)))

    summary_csv = OUTPUT_ROOT / "era5_lod_loyo_mode1_mode2_latepooled_selection_frequency.csv"
    plot_png = OUTPUT_ROOT / "era5_lod_loyo_mode1_mode2_latepooled_selection_frequency.png"
    plot_pdf = OUTPUT_ROOT / "era5_lod_loyo_mode1_mode2_latepooled_selection_frequency.pdf"
    note_md = OUTPUT_ROOT / "era5_lod_loyo_mode_stability_interpretation.md"

    write_summary_csv(summary_csv, csv_rows)
    make_plot(
        plot_png,
        plot_pdf,
        [
            ("Mode 1 selection frequency", mode1_counts),
            ("Mode 2 selection frequency", mode2_counts),
            ("Pooled modes 3--6 selection frequency", pooled_late_counts),
        ],
    )
    write_note(note_md, len(pooled_late_rows), mode1_counts, mode2_counts, pooled_late_counts)

    print(summary_csv)
    print(plot_png)
    print(plot_pdf)
    print(note_md)


if __name__ == "__main__":
    main()
