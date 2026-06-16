#!/usr/bin/env python3
"""
Generate North Atlantic AMV/AMO EOF-PC predictor products for the Sierra SWE baseline.
"""

import csv
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_cobe2_global_sst_eof_reproduction import (
    COBE2_SST_FILE,
    compute_latitude_sqrt_cos_weights,
    format_date,
)
from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    compute_monthly_climatology_anomalies,
    ensure_runtime_on_compute_node,
    get_runtime,
    open_dataset_with_fallbacks,
)


EXPERIMENT_NAME = "amv_amo_cobe2_north_atlantic_pc1to6_for_swe"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "swe_climate_mode_baseline" / "amv_amo"
EOF_NETCDF_PATH = OUTPUT_DIR / "amv_amo_cobe2_north_atlantic_eofs_pc1to6.nc"
PREDICTOR_CSV_PATH = OUTPUT_DIR / "amv_amo_cobe2_north_atlantic_pc1to6_wy1985_2021_sep_mar.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "amv_amo_cobe2_north_atlantic_pc1to6_summary.json"
EOF_FIGURE_PATH = OUTPUT_DIR / "amv_amo_cobe2_north_atlantic_eofs_modes1to6.png"

VARIABLE_NAME = "sst"
N_SAVED_MODES = 6
LAT_MIN = 0.0
LAT_MAX = 70.0
LON_MIN_360 = 280.0
LON_MAX_360 = 360.0
WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
MONTH_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]


@dataclass(frozen=True)
class AtlanticWeightedEofResult:
    source_file: str
    variable_name: str
    time: np.ndarray
    latitude: np.ndarray
    longitude_360: np.ndarray
    climatology: np.ndarray
    anomalies: np.ndarray
    eofs: np.ndarray
    pcs: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    valid_cell_mask: np.ndarray
    latitude_weights: np.ndarray
    sign_flips_applied: np.ndarray


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def deduplicate_cyclic_longitudes(longitude_360: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(longitude_360, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    keep_indices: List[int] = []
    seen = set()
    for index, value in enumerate(lon):
        key = round(float(value), 8)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(index)
    keep = np.asarray(keep_indices, dtype=np.int64)
    return lon[keep], data[:, :, keep]


def load_north_atlantic_cobe2_sst() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        if VARIABLE_NAME not in ds:
            raise KeyError("Expected SST variable in COBE2 monthly file")
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        values = np.asarray(ds[VARIABLE_NAME].values, dtype=np.float64)
        missing_value = float(ds[VARIABLE_NAME].attrs.get("missing_value", 1.0e20))

    values = np.where(values >= missing_value, np.nan, values)
    longitude_360 = np.mod(longitude, 360.0)
    sort_idx = np.argsort(longitude_360)
    longitude_sorted = longitude_360[sort_idx]
    values_sorted = values[:, :, sort_idx]
    longitude_sorted, values_sorted = deduplicate_cyclic_longitudes(longitude_sorted, values_sorted)

    lat_mask = (latitude >= LAT_MIN) & (latitude <= LAT_MAX)
    lon_mask = (longitude_sorted >= LON_MIN_360) & (longitude_sorted < LON_MAX_360)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError("North Atlantic AMV/AMO domain is empty on the COBE2 grid")

    return (
        time_values,
        latitude[lat_mask],
        longitude_sorted[lon_mask],
        values_sorted[:, lat_mask, :][:, :, lon_mask],
    )


def solve_weighted_eofs(
    time_values: np.ndarray,
    latitude: np.ndarray,
    longitude_360: np.ndarray,
    values: np.ndarray,
) -> AtlanticWeightedEofResult:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        climatology, anomalies = compute_monthly_climatology_anomalies(values, time_values)

    anomalies_flat = anomalies.reshape(anomalies.shape[0], -1)
    valid_cell_mask_flat = np.isfinite(anomalies_flat).all(axis=0)
    n_valid_cells = int(valid_cell_mask_flat.sum())
    if n_valid_cells < N_SAVED_MODES:
        raise ValueError("Need at least six valid Atlantic ocean cells for EOF analysis")

    lat_weights = compute_latitude_sqrt_cos_weights(latitude)
    weights_2d = np.broadcast_to(lat_weights[:, np.newaxis], (latitude.size, longitude_360.size))
    weights_flat = weights_2d.reshape(-1)[valid_cell_mask_flat]

    anomaly_matrix = anomalies_flat[:, valid_cell_mask_flat]
    weighted_matrix = anomaly_matrix * weights_flat[np.newaxis, :]
    gram_matrix = weighted_matrix @ weighted_matrix.T
    eigenvalues, u_matrix = np.linalg.eigh(gram_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    u_matrix = u_matrix[:, order]

    singular_values_all = np.sqrt(eigenvalues)
    total_weighted_variance = float(np.sum(singular_values_all ** 2))
    if total_weighted_variance <= 0.0:
        raise ValueError("Total weighted Atlantic anomaly variance is zero")
    explained_variance_ratio_all = (singular_values_all ** 2) / total_weighted_variance

    pcs = (u_matrix[:, :N_SAVED_MODES] * singular_values_all[:N_SAVED_MODES]).astype(np.float64)

    weighted_eofs_valid = np.zeros((N_SAVED_MODES, n_valid_cells), dtype=np.float64)
    for mode_index in range(N_SAVED_MODES):
        singular_value = singular_values_all[mode_index]
        if singular_value <= 0.0:
            continue
        weighted_eofs_valid[mode_index] = (weighted_matrix.T @ u_matrix[:, mode_index]) / singular_value

    unweighted_eofs_valid = np.full_like(weighted_eofs_valid, np.nan)
    positive_weight = weights_flat > 0.0
    unweighted_eofs_valid[:, positive_weight] = (
        weighted_eofs_valid[:, positive_weight] / weights_flat[np.newaxis, positive_weight]
    )

    eof_grid = np.full((N_SAVED_MODES, latitude.size, longitude_360.size), np.nan, dtype=np.float64)
    eof_grid.reshape(N_SAVED_MODES, -1)[:, valid_cell_mask_flat] = unweighted_eofs_valid
    valid_mask_2d = np.zeros((latitude.size, longitude_360.size), dtype=bool)
    valid_mask_2d.reshape(-1)[valid_cell_mask_flat] = True

    return AtlanticWeightedEofResult(
        source_file=str(COBE2_SST_FILE),
        variable_name=VARIABLE_NAME,
        time=np.asarray(time_values, dtype="datetime64[ns]"),
        latitude=latitude.astype(np.float32),
        longitude_360=longitude_360.astype(np.float32),
        climatology=climatology.astype(np.float32),
        anomalies=anomalies.astype(np.float32),
        eofs=eof_grid.astype(np.float32),
        pcs=pcs.astype(np.float32),
        singular_values=singular_values_all.astype(np.float64),
        explained_variance_ratio=explained_variance_ratio_all.astype(np.float64),
        valid_cell_mask=valid_mask_2d,
        latitude_weights=lat_weights.astype(np.float32),
        sign_flips_applied=np.ones(N_SAVED_MODES, dtype=np.int8),
    )


def save_eof_dataset(result: AtlanticWeightedEofResult) -> None:
    ds = xr.Dataset(
        data_vars={
            "eof": (("mode", "lat", "lon"), result.eofs[:N_SAVED_MODES]),
            "pc": (("time", "mode"), result.pcs[:, :N_SAVED_MODES]),
            "singular_value": (("mode",), result.singular_values[:N_SAVED_MODES].astype(np.float32)),
            "explained_variance_ratio": (
                ("mode",),
                result.explained_variance_ratio[:N_SAVED_MODES].astype(np.float32),
            ),
            "valid_mask": (("lat", "lon"), result.valid_cell_mask),
        },
        coords={
            "mode": np.arange(1, N_SAVED_MODES + 1, dtype=np.int32),
            "time": result.time,
            "lat": result.latitude,
            "lon": result.longitude_360,
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "description": "COBE2 North Atlantic monthly-climatology SST EOF product for the Sierra SWE AMV/AMO baseline",
            "source_file": result.source_file,
            "variable_name": result.variable_name,
            "domain_latitude": "0N to 70N",
            "domain_longitude_360": "280E to <360E",
            "domain_equivalent_west_longitude": "80W to 0W",
            "monthly_climatology_removed": "true",
            "additional_time_mean_centering_applied": "false",
            "latitude_weighting": "sqrt(cos(lat)) applied before EOF solve; EOF maps saved in unweighted SST-loading units",
            "sign_convention": "EOF/PC sign is arbitrary; no sign-normalization or physical sign interpretation was applied",
            "cyclic_longitude_endpoint_handling": "duplicate cyclic longitudes were removed before Atlantic subsetting",
        },
    )
    ds.to_netcdf(EOF_NETCDF_PATH)


def build_predictor_table(result: AtlanticWeightedEofResult) -> Tuple[List[str], np.ndarray]:
    time_to_index = {}
    for idx, value in enumerate(np.asarray(result.time, dtype="datetime64[ns]")):
        time_to_index[str(np.datetime_as_string(value, unit="D"))] = idx

    columns = ["water_year"]
    for month_name, _, _ in MONTH_SPECS:
        for mode in range(1, N_SAVED_MODES + 1):
            columns.append("AMV_PC%d_%s" % (mode, month_name))

    rows = np.full((WATER_YEARS.size, len(columns)), np.nan, dtype=np.float64)
    rows[:, 0] = WATER_YEARS.astype(np.float64)
    for wy_idx, water_year in enumerate(WATER_YEARS):
        col_idx = 1
        for month_name, year_offset, month in MONTH_SPECS:
            year = int(water_year + year_offset)
            key = "%04d-%02d-01" % (year, month)
            if key not in time_to_index:
                raise KeyError("Missing Atlantic PC timestamp %s in EOF output" % key)
            time_idx = time_to_index[key]
            for mode_idx in range(N_SAVED_MODES):
                rows[wy_idx, col_idx] = float(result.pcs[time_idx, mode_idx])
                col_idx += 1
    return columns, rows


def write_predictor_csv(columns: List[str], rows: np.ndarray) -> None:
    with PREDICTOR_CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [int(row[0])] + ["{:.12g}".format(float(value)) for value in row[1:]]
            )


def save_summary_json(result: AtlanticWeightedEofResult, columns: List[str]) -> None:
    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "source_file": result.source_file,
        "variable_name": result.variable_name,
        "full_time_start": format_date(result.time[0]),
        "full_time_end": format_date(result.time[-1]),
        "n_time_full_record": int(result.time.size),
        "domain_lat_min": LAT_MIN,
        "domain_lat_max": LAT_MAX,
        "domain_lon_min_360": LON_MIN_360,
        "domain_lon_max_360_exclusive": LON_MAX_360,
        "domain_equivalent_west_longitude": "80W to 0W",
        "monthly_climatology_removed": True,
        "additional_time_mean_centering_applied": False,
        "weighting_formula": "sqrt(cos(lat))",
        "valid_cell_rule": "grid cell retained only if anomaly is finite at every monthly time step",
        "sign_convention": "EOF sign is arbitrary; no sign-normalization applied",
        "n_lat": int(result.latitude.size),
        "n_lon": int(result.longitude_360.size),
        "n_valid_cells": int(result.valid_cell_mask.sum()),
        "explained_variance_ratio_mode1to6": [float(value) for value in evr.tolist()],
        "cumulative_explained_variance_ratio_mode1to6": [float(value) for value in np.cumsum(evr).tolist()],
        "singular_values_mode1to6": [float(value) for value in result.singular_values[:N_SAVED_MODES].tolist()],
        "predictor_water_year_start": WATER_YEAR_START,
        "predictor_water_year_end": WATER_YEAR_END,
        "predictor_month_sequence": [item[0] for item in MONTH_SPECS],
        "predictor_columns": columns[1:],
        "output_eof_netcdf": str(EOF_NETCDF_PATH),
        "output_predictor_csv": str(PREDICTOR_CSV_PATH),
        "output_figure": str(EOF_FIGURE_PATH),
        "runtime_hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_eofs_and_pcs(result: AtlanticWeightedEofResult) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    lon2d, lat2d = np.meshgrid(result.longitude_360, result.latitude)
    for mode_index, ax in enumerate(axes.flat):
        eof = np.asarray(result.eofs[mode_index], dtype=np.float64)
        finite = np.isfinite(eof)
        if not np.any(finite):
            ax.set_visible(False)
            continue
        vmax = float(np.nanmax(np.abs(eof[finite])))
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            eof,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlim(float(result.longitude_360[0]), float(result.longitude_360[-1]))
        ax.set_ylim(float(result.latitude[0]), float(result.latitude[-1]))
        ax.set_xlabel("Longitude (0..360)")
        ax.set_ylabel("Latitude")
        ax.set_title(
            "EOF%d | EVR=%.3f" % (mode_index + 1, float(result.explained_variance_ratio[mode_index]))
        )
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.86)
        cbar.set_label("SST loading")
    fig.suptitle("COBE2 North Atlantic SST EOFs 1--6 for the AMV/AMO baseline", fontsize=14)
    fig.savefig(EOF_FIGURE_PATH, dpi=180)
    plt.close(fig)


def print_stdout_summary(result: AtlanticWeightedEofResult) -> None:
    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    print("Output directory: %s" % OUTPUT_DIR, flush=True)
    print(
        "Atlantic EOF EVR1-6: %s"
        % ", ".join(["%.6f" % float(value) for value in evr.tolist()]),
        flush=True,
    )
    print("EOF NetCDF: %s" % EOF_NETCDF_PATH, flush=True)
    print("Predictor CSV: %s" % PREDICTOR_CSV_PATH, flush=True)
    print("Summary JSON: %s" % SUMMARY_JSON_PATH, flush=True)
    print("EOF figure: %s" % EOF_FIGURE_PATH, flush=True)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()

    time_values, latitude, longitude_360, values = load_north_atlantic_cobe2_sst()
    result = solve_weighted_eofs(time_values, latitude, longitude_360, values)
    save_eof_dataset(result)
    columns, rows = build_predictor_table(result)
    write_predictor_csv(columns, rows)
    save_summary_json(result, columns)
    plot_eofs_and_pcs(result)
    print_stdout_summary(result)


if __name__ == "__main__":
    main()
