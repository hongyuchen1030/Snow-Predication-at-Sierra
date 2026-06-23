#!/usr/bin/env python3
"""
Plot nested-LOYO LOD selected-location maps for COBE2 and ERA5.

This script reuses the saved fold-by-fold nested LOYO LOD selections, compares
them against the corresponding full-sample selected locations, and writes:

1. One map per dataset and mode.
2. A summary CSV with stability metrics.
3. A detailed CSV with distance-to-full diagnostics.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "artifacts" / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


NETCDF_ENGINE = "netcdf4"
PACIFIC_LAT_MIN = -10.0
PACIFIC_LAT_MAX = 60.0
PACIFIC_LON_MIN = 120.0
PACIFIC_LON_MAX = 280.0
TOTAL_FOLDS = 37
WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
LAG_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "nested_loyo_lod_selected_location_map_diagnostic"

COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
ERA5_PREDICTOR_FILE = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "era5_sierra_swe_lod_setup/predictors/era5_pacific_sst_monthly_anomaly_wy1985_2021_sep1984_mar2021.nc"
)

DATASET_CONFIGS = {
    "COBE2": {
        "label": "COBE2",
        "fold_modes_csv": Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "cobe2_sierra_swe_lod_setup/loyo_lod_analysis/cobe2_sierra_swe_lod_loyo_fold_modes.csv"
        ),
        "full_summary_json": Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json"
        ),
    },
    "ERA5": {
        "label": "ERA5",
        "fold_modes_csv": Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/loyo_lod_analysis/era5_sierra_swe_lod_loyo_fold_modes.csv"
        ),
        "full_summary_json": Path(
            "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
            "era5_sierra_swe_lod_setup/lod_analysis/era5_sierra_swe_lod_summary.json"
        ),
    },
}


def month_targets():
    times = []
    for water_year in WATER_YEARS:
        for _, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64("{:04d}-{:02d}-01".format(year, month)))
    return times


def load_cobe2_grid():
    selected_times = month_targets()
    with xr.open_dataset(COBE2_SST_FILE, engine=NETCDF_ENGINE) as ds:
        sst = ds["sst"].sel(
            time=selected_times[: len(LAG_SPECS)],
            lat=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            lon=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        ).load()
        latitude = np.asarray(sst["lat"].values, dtype=np.float64)
        longitude = np.asarray(sst["lon"].values, dtype=np.float64)
    return latitude, longitude


def load_era5_grid():
    with xr.open_dataset(ERA5_PREDICTOR_FILE, engine=NETCDF_ENGINE) as ds:
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
    return latitude, longitude


def load_dataset_grids():
    return {
        "COBE2": load_cobe2_grid(),
        "ERA5": load_era5_grid(),
    }


def find_coordinate_index(values, target, coord_name):
    idx = int(np.argmin(np.abs(values - float(target))))
    nearest = float(values[idx])
    if not np.isclose(nearest, float(target), atol=1.0e-6):
        raise ValueError("Could not match {}={} in coordinate array; nearest={}".format(coord_name, target, nearest))
    return idx


def haversine_km(lon1, lat1, lon2, lat2):
    radius_km = 6371.0
    lon1_rad = np.deg2rad(lon1)
    lat1_rad = np.deg2rad(lat1)
    lon2_rad = np.deg2rad(lon2)
    lat2_rad = np.deg2rad(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


def load_nested_rows(dataset, csv_path, latitude, longitude):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            selected_value = str(row.get("selected", "")).strip().lower()
            if selected_value != "true":
                continue
            lat_value = float(row["latitude"])
            lon_value = float(row["longitude_0_360"])
            rows.append(
                {
                    "heldout_year": int(row["held_out_water_year"]),
                    "mode_index": int(row["mode_number"]),
                    "dataset": dataset,
                    "month": str(row["lag_month"]),
                    "lat": lat_value,
                    "lon": lon_value,
                    "grid_i": find_coordinate_index(latitude, lat_value, "latitude"),
                    "grid_j": find_coordinate_index(longitude, lon_value, "longitude"),
                    "column_id": int(row.get("candidate_index", row.get("ocean_candidate_q", -1))),
                    "corr": float(row["corr_with_residual"]),
                    "delta_r2": float(row["delta_r2"]),
                    "r2": float(row["cumulative_r2"]),
                }
            )
    return rows


def load_full_rows(dataset, summary_path, latitude, longitude):
    summary = json.loads(summary_path.read_text())
    rows = []
    for row in summary["lod_rows"]:
        if not row.get("selected"):
            continue
        lat_value = float(row["latitude"])
        lon_value = float(row.get("longitude_0_360", row.get("longitude")))
        rows.append(
            {
                "mode_index": int(row["mode_number"]),
                "dataset": dataset,
                "month": str(row["lag_month"]),
                "lat": lat_value,
                "lon": lon_value,
                "grid_i": find_coordinate_index(latitude, lat_value, "latitude"),
                "grid_j": find_coordinate_index(longitude, lon_value, "longitude"),
                "column_id": int(row.get("candidate_index", row.get("ocean_candidate_q", -1))),
                "corr": float(row["corr_with_residual"]),
                "delta_r2": float(row["delta_r2"]),
                "r2": float(row["cumulative_r2"]),
            }
        )
    return rows


def add_distance_to_full(nested_rows, full_rows):
    full_lookup = {}
    for row in full_rows:
        full_lookup[(row["dataset"], row["mode_index"])] = row

    detailed_rows = []
    for row in nested_rows:
        full_row = full_lookup[(row["dataset"], row["mode_index"])]
        distance = float(haversine_km(row["lon"], row["lat"], full_row["lon"], full_row["lat"]))
        output_row = dict(row)
        output_row["full_month"] = full_row["month"]
        output_row["full_lat"] = full_row["lat"]
        output_row["full_lon"] = full_row["lon"]
        output_row["full_grid_i"] = full_row["grid_i"]
        output_row["full_grid_j"] = full_row["grid_j"]
        output_row["full_column_id"] = full_row["column_id"]
        output_row["distance_to_full_km"] = distance
        output_row["exact_full_column_match"] = int(row["column_id"] == full_row["column_id"])
        output_row["same_month_as_full"] = int(str(row["month"]) == str(full_row["month"]))
        detailed_rows.append(output_row)
    return detailed_rows


def plot_mode_map(dataset, mode_index, mode_rows, full_row, output_path, total_possible_folds):
    mode_rows_sorted = sorted(mode_rows, key=lambda item: item["heldout_year"])
    longitudes = np.array([row["lon"] for row in mode_rows_sorted], dtype=np.float64)
    latitudes = np.array([row["lat"] for row in mode_rows_sorted], dtype=np.float64)
    years = np.array([row["heldout_year"] for row in mode_rows_sorted], dtype=np.float64)
    exact_matches = int(sum(row["exact_full_column_match"] for row in mode_rows_sorted))
    same_month_matches = int(sum(row["same_month_as_full"] for row in mode_rows_sorted))
    distances = np.array([row["distance_to_full_km"] for row in mode_rows_sorted], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12.0, 5.8), constrained_layout=True)
    ax.set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
    ax.set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
    ax.set_xticks(np.arange(120.0, 281.0, 20.0))
    ax.set_yticks(np.arange(-10.0, 61.0, 10.0))
    ax.set_xlabel("Longitude (0 to 360)")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.25, color="0.82")
    ax.set_facecolor("white")

    scatter = ax.scatter(
        longitudes,
        latitudes,
        c=years,
        cmap="viridis",
        s=38,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )
    for idx, row in enumerate(mode_rows_sorted):
        dx = 0.45 * ((idx % 3) - 1)
        dy = 0.45 * (((idx // 3) % 3) - 1)
        ax.text(
            float(row["lon"]) + dx,
            float(row["lat"]) + dy,
            str(row["month"]),
            fontsize=6.5,
            ha="center",
            va="center",
            zorder=4,
        )

    ax.scatter(
        float(full_row["lon"]),
        float(full_row["lat"]),
        marker="*",
        s=220,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    ax.text(
        float(full_row["lon"]) + 1.0,
        float(full_row["lat"]) + 1.0,
        "Full: {}".format(full_row["month"]),
        fontsize=9,
        fontweight="bold",
        zorder=6,
    )

    ax.set_title("{} Mode {} Nested LOYO Selected Locations".format(dataset, mode_index))
    subtitle = "Dots: selected location in each held-out fold; labels: selected month; star: full-37 selection"
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    stats_text = (
        "Exact full-column match: {}/{}\n"
        "Same month as full: {}/{}\n"
        "Median distance to full: {:.0f} km\n"
        "Mean distance to full: {:.0f} km"
    ).format(
        exact_matches,
        total_possible_folds,
        same_month_matches,
        total_possible_folds,
        float(np.nanmedian(distances)),
        float(np.nanmean(distances)),
    )
    ax.text(
        0.01,
        0.99,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.88},
    )

    dot_proxy = mlines.Line2D([], [], color="tab:blue", marker="o", linestyle="None", markersize=6, label="Nested LOYO selected points")
    text_proxy = mlines.Line2D([], [], color="none", marker=None, linestyle="None", label="Text labels = selected month")
    star_proxy = mlines.Line2D([], [], color="black", marker="*", linestyle="None", markersize=11, label="Full-37 selected point")
    ax.legend(handles=[dot_proxy, text_proxy, star_proxy], loc="lower right", frameon=True)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label("Held-out water year")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "dataset": dataset,
        "mode_index": mode_index,
        "n_exact_full_match": exact_matches,
        "n_same_month_as_full": same_month_matches,
        "median_distance_to_full_km": float(np.nanmedian(distances)),
        "mean_distance_to_full_km": float(np.nanmean(distances)),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_summary_markdown(path, summary_rows, output_dir):
    lines = [
        "# Nested LOYO LOD Selected-Location Map Diagnostic",
        "",
        "- The maps were produced from the saved nested LOYO fold-mode tables and the full-sample LOD summaries.",
        "- No LOD rerun was performed.",
        "- Each figure shows the selected location for each held-out fold, month labels, and the full-sample black-star reference.",
        "",
        "## Outputs",
        "",
        "- Summary CSV: `{}`".format(output_dir / "nested_loyo_selection_stability_summary.csv"),
        "- Detailed CSV: `{}`".format(output_dir / "nested_loyo_selection_with_distance_to_full.csv"),
        "",
        "## Stability summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- {} mode {}: exact={}/{} same_month={}/{} median_dist={:.0f} km mean_dist={:.0f} km".format(
                row["dataset"],
                row["mode_index"],
                row["n_exact_full_match"],
                TOTAL_FOLDS,
                row["n_same_month_as_full"],
                TOTAL_FOLDS,
                row["median_distance_to_full_km"],
                row["mean_distance_to_full_km"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    grids = load_dataset_grids()

    nested_rows_all = []
    full_rows_all = []
    for dataset, config in DATASET_CONFIGS.items():
        latitude, longitude = grids[dataset]
        nested_rows_all.extend(load_nested_rows(dataset, config["fold_modes_csv"], latitude, longitude))
        full_rows_all.extend(load_full_rows(dataset, config["full_summary_json"], latitude, longitude))

    detailed_rows = add_distance_to_full(nested_rows_all, full_rows_all)
    full_lookup = {}
    for row in full_rows_all:
        full_lookup[(row["dataset"], row["mode_index"])] = row

    summary_rows = []
    for dataset in sorted(DATASET_CONFIGS):
        dataset_rows = [row for row in detailed_rows if row["dataset"] == dataset]
        mode_indices = sorted({int(row["mode_index"]) for row in dataset_rows})
        for mode_index in mode_indices:
            mode_rows = [row for row in dataset_rows if int(row["mode_index"]) == mode_index]
            output_png = OUTPUT_ROOT / "{}_mode{}_nested_loyo_selected_locations.png".format(dataset, mode_index)
            summary_rows.append(
                plot_mode_map(
                    dataset=dataset,
                    mode_index=mode_index,
                    mode_rows=mode_rows,
                    full_row=full_lookup[(dataset, mode_index)],
                    output_path=output_png,
                    total_possible_folds=TOTAL_FOLDS,
                )
            )

    summary_rows = sorted(summary_rows, key=lambda row: (row["dataset"], int(row["mode_index"])))
    summary_csv = OUTPUT_ROOT / "nested_loyo_selection_stability_summary.csv"
    detailed_csv = OUTPUT_ROOT / "nested_loyo_selection_with_distance_to_full.csv"
    summary_md = OUTPUT_ROOT / "nested_loyo_selection_stability_summary.md"

    write_csv(
        summary_csv,
        summary_rows,
        [
            "dataset",
            "mode_index",
            "n_exact_full_match",
            "n_same_month_as_full",
            "median_distance_to_full_km",
            "mean_distance_to_full_km",
        ],
    )
    write_csv(
        detailed_csv,
        detailed_rows,
        [
            "heldout_year",
            "mode_index",
            "dataset",
            "month",
            "lat",
            "lon",
            "grid_i",
            "grid_j",
            "column_id",
            "corr",
            "delta_r2",
            "r2",
            "full_month",
            "full_lat",
            "full_lon",
            "full_grid_i",
            "full_grid_j",
            "full_column_id",
            "distance_to_full_km",
            "exact_full_column_match",
            "same_month_as_full",
        ],
    )
    write_summary_markdown(summary_md, summary_rows, OUTPUT_ROOT)

    print(summary_csv)
    print(detailed_csv)
    print(summary_md)
    for dataset in sorted(DATASET_CONFIGS):
        for mode_index in sorted({row["mode_index"] for row in summary_rows if row["dataset"] == dataset}):
            print(OUTPUT_ROOT / "{}_mode{}_nested_loyo_selected_locations.png".format(dataset, mode_index))


if __name__ == "__main__":
    main()
