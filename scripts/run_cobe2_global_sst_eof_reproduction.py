#!/usr/bin/env python3
"""
Run an additive global COBE2 monthly-climatology SST EOF reproduction diagnostic.

This experiment:
1. Loads the full global COBE2 monthly SST dataset.
2. Converts longitude to [-180, 180] and sorts it for coherent global plotting.
3. Removes the month-of-year climatology to form SST anomalies.
4. Applies sqrt(cos(lat)) area weighting before the EOF solve.
5. Computes the full singular-value spectrum from the time-space Gram matrix.
6. Reconstructs the leading saved EOF maps and PCs.
7. Writes NetCDF, summary JSON, ENSO validation CSV, and an EOF/PC figure.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

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

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    COBE2_SST_FILE,
    compute_monthly_climatology_anomalies,
    compute_monthly_series_anomalies,
    compute_multiple_regression_skill,
    compute_weighted_regional_mean,
    ensure_runtime_on_compute_node,
    get_runtime,
    normalize_longitude_to_minus180_180,
    open_dataset_with_fallbacks,
    spatial_correlation,
)


EXPERIMENT_NAME = "cobe2_global_monthly_climatology_eof_reproduction"
DATASET_ID = "COBE2_global"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "cobe2_global_monthly_climatology_anomaly"
EOF_FILE = OUTPUT_DIR / "cobe2_global_monthly_clim_sst_eofs.nc"
SUMMARY_FILE = OUTPUT_DIR / "cobe2_global_monthly_clim_sst_summary.json"
FIGURE_FILE = OUTPUT_DIR / "cobe2_global_monthly_clim_sst_eof_pc_reproduction.png"
PACIFIC_CENTERED_FIGURE_FILE = OUTPUT_DIR / "cobe2_global_monthly_clim_sst_eof_pc_reproduction_pacific_centered.png"
PACIFIC_CENTERED_SIGN_ALIGNED_FIGURE_FILE = (
    OUTPUT_DIR / "cobe2_global_monthly_clim_sst_eof_pc_reproduction_pacific_centered_sign_aligned.png"
)
ENSO_VALIDATION_CSV = OUTPUT_DIR / "cobe2_global_enso_pc_correlation_validation.csv"
METHOD_SUMMARY_FILE = OUTPUT_DIR / "cobe2_global_eof_reproduction_method_summary.md"

VARIABLE_NAME = "sst"
N_SAVED_MODES = 6
NINO34_LAT_MIN = -5.0
NINO34_LAT_MAX = 5.0
NINO34_LON_MIN = -170.0
NINO34_LON_MAX = -120.0


@dataclass(frozen=True)
class GlobalWeightedEofResult:
    dataset_id: str
    source_file: str
    variable_name: str
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    climatology: np.ndarray
    anomalies: np.ndarray
    eofs: np.ndarray
    pcs: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    valid_cell_mask: np.ndarray
    latitude_weights: np.ndarray
    sign_flips_applied: np.ndarray


def format_date(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_global_cobe2_sst() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        if VARIABLE_NAME not in ds:
            raise KeyError(f"Expected variable {VARIABLE_NAME!r} in {COBE2_SST_FILE}")
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = normalize_longitude_to_minus180_180(np.asarray(ds["lon"].values, dtype=np.float64))
        values = np.asarray(ds[VARIABLE_NAME].values, dtype=np.float64)
        missing_value = float(ds[VARIABLE_NAME].attrs.get("missing_value", 1.0e20))

    lon_sort_idx = np.argsort(longitude)
    longitude_sorted = longitude[lon_sort_idx]
    values_sorted = values[:, :, lon_sort_idx]
    values_sorted = np.where(values_sorted >= missing_value, np.nan, values_sorted)
    return time_values, latitude, longitude_sorted, values_sorted


def compute_latitude_sqrt_cos_weights(latitude: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = np.clip(cos_lat, 0.0, None)
    return np.sqrt(cos_lat)


def solve_weighted_eofs(
    time_values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
) -> GlobalWeightedEofResult:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        climatology, anomalies = compute_monthly_climatology_anomalies(values, time_values)
    anomalies_flat = anomalies.reshape(anomalies.shape[0], -1)
    valid_cell_mask_flat = np.isfinite(anomalies_flat).all(axis=0)
    n_valid_cells = int(valid_cell_mask_flat.sum())
    if n_valid_cells < N_SAVED_MODES:
        raise ValueError(f"Need at least {N_SAVED_MODES} all-time-finite ocean cells, got {n_valid_cells}")

    lat_weights = compute_latitude_sqrt_cos_weights(latitude)
    weights_2d = np.broadcast_to(lat_weights[:, np.newaxis], (latitude.size, longitude.size))
    weights_flat = weights_2d.reshape(-1)[valid_cell_mask_flat]

    # Weighting convention:
    # 1. Build monthly-climatology anomalies A(time, space).
    # 2. Multiply each spatial column by sqrt(cos(lat)) to form X = A * W.
    # 3. Solve the EOF problem on X so the variance metric is area-weighted.
    # 4. Reconstruct weighted EOF vectors in V, then divide by sqrt(cos(lat))
    #    to map EOFs back to unweighted SST-loading units for plotting.
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
        raise ValueError("Total weighted anomaly variance is zero")
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

    eof_grid = np.full((N_SAVED_MODES, latitude.size, longitude.size), np.nan, dtype=np.float64)
    eof_grid.reshape(N_SAVED_MODES, -1)[:, valid_cell_mask_flat] = unweighted_eofs_valid

    valid_mask_2d = np.zeros((latitude.size, longitude.size), dtype=bool)
    valid_mask_2d.reshape(-1)[valid_cell_mask_flat] = True

    sign_flips_applied = np.ones(N_SAVED_MODES, dtype=np.int8)
    return GlobalWeightedEofResult(
        dataset_id=DATASET_ID,
        source_file=str(COBE2_SST_FILE),
        variable_name=VARIABLE_NAME,
        time=np.asarray(time_values, dtype="datetime64[ns]"),
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        climatology=climatology.astype(np.float32),
        anomalies=anomalies.astype(np.float32),
        eofs=eof_grid.astype(np.float32),
        pcs=pcs.astype(np.float32),
        singular_values=singular_values_all.astype(np.float64),
        explained_variance_ratio=explained_variance_ratio_all.astype(np.float64),
        valid_cell_mask=valid_mask_2d,
        latitude_weights=lat_weights.astype(np.float32),
        sign_flips_applied=sign_flips_applied,
    )


def compute_nino34_index_from_cobe2(
    time_values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    regional_mean = compute_weighted_regional_mean(
        values,
        latitude,
        longitude,
        NINO34_LAT_MIN,
        NINO34_LAT_MAX,
        NINO34_LON_MIN,
        NINO34_LON_MAX,
    )
    anomalies = compute_monthly_series_anomalies(regional_mean, time_values)
    return time_values, anomalies


def align_pc_and_index(
    pc_time: np.ndarray,
    pc_values: np.ndarray,
    index_time: np.ndarray,
    index_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_time, pc_idx, index_idx = np.intersect1d(
        np.asarray(pc_time, dtype="datetime64[ns]"),
        np.asarray(index_time, dtype="datetime64[ns]"),
        assume_unique=False,
        return_indices=True,
    )
    if common_time.size == 0:
        return common_time, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    pc_aligned = np.asarray(pc_values, dtype=np.float64)[pc_idx]
    index_aligned = np.asarray(index_values, dtype=np.float64)[index_idx]
    finite = np.isfinite(pc_aligned) & np.isfinite(index_aligned)
    return common_time[finite], pc_aligned[finite], index_aligned[finite]


def interpret_enso_signal(pc_abs_corrs: np.ndarray, pc23_r: float, pc123_r: float) -> str:
    dominant_mode = int(np.argmax(pc_abs_corrs)) + 1
    dominant_corr = float(np.max(pc_abs_corrs))
    if dominant_mode == 1 and dominant_corr >= 0.7:
        return "PC1 is the most ENSO-related leading mode."
    if dominant_mode == 2 and dominant_corr >= 0.7:
        return "PC2 is the most ENSO-related leading mode."
    if dominant_mode == 3 and dominant_corr >= 0.7:
        return "PC3 is the most ENSO-related leading mode."
    if pc_abs_corrs[1] >= 0.4 and pc_abs_corrs[2] >= 0.4 and pc23_r >= 0.7:
        return "ENSO is represented most cleanly by the combined PC2-PC3 subspace."
    if np.isfinite(pc123_r) and np.isfinite(pc23_r) and pc123_r > pc23_r + 0.05:
        return "ENSO-related variability is distributed across PC1-PC3."
    return f"PC{dominant_mode} is the most ENSO-related of the first three modes."


def save_eof_dataset(result: GlobalWeightedEofResult, output_path: Path) -> None:
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
            "lon": result.longitude,
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "description": "COBE2 global monthly-climatology SST weighted EOF reproduction",
            "source_file": result.source_file,
            "variable_name": result.variable_name,
            "domain": "global",
            "monthly_climatology_removed": "true",
            "additional_global_time_mean_centering_applied": "false",
            "latitude_weighting": "sqrt(cos(lat)) applied before EOF solve; EOF maps saved in unweighted SST-loading units",
        },
    )
    ds.to_netcdf(output_path)


def save_summary_json(result: GlobalWeightedEofResult, output_path: Path) -> None:
    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    sv = np.asarray(result.singular_values[:N_SAVED_MODES], dtype=np.float64)
    payload: Dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "input_file_path": result.source_file,
        "variable_name": result.variable_name,
        "domain": "global",
        "n_time": int(result.time.size),
        "n_valid_cells": int(result.valid_cell_mask.sum()),
        "monthly_climatology_removed": True,
        "additional_global_time_mean_centering_applied": False,
        "latitude_weighting_applied": True,
        "latitude_weighting_formula": "sqrt(cos(lat))",
        "weighting_implementation": (
            "Monthly-climatology anomaly columns were multiplied by sqrt(cos(lat)) before the EOF solve. "
            "The solve used the full time-space Gram matrix X X^T to recover all singular values exactly. "
            "Saved EOF maps were converted back to unweighted SST-loading units by dividing each weighted EOF "
            "column by sqrt(cos(lat)) at that grid cell."
        ),
        "time_start": format_date(result.time[0]),
        "time_end": format_date(result.time[-1]),
        "n_lat": int(result.latitude.size),
        "n_lon": int(result.longitude.size),
        "singular_values_eof1_eof3": [float(value) for value in sv],
        "evr_eof1_eof3": [float(value) for value in evr],
        "cumulative_evr_eof1_eof3": float(np.sum(evr)),
        "sign_flips_applied_for_plotting": [int(value) for value in result.sign_flips_applied.tolist()],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def try_import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.util as cutil

        return ccrs, cutil
    except Exception:
        return None, None


def add_cyclic_if_available(field: np.ndarray, longitude: np.ndarray, cutil) -> Tuple[np.ndarray, np.ndarray]:
    if cutil is None:
        return field, longitude
    cyclic_field, cyclic_lon = cutil.add_cyclic_point(field, coord=longitude)
    return cyclic_field, cyclic_lon


def maybe_roll_longitude_for_pacific_centered(
    field: np.ndarray,
    longitude: np.ndarray,
    pacific_centered: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not pacific_centered:
        return field, longitude
    lon_360 = np.mod(np.asarray(longitude, dtype=np.float64), 360.0)
    order = np.argsort(lon_360)
    return np.asarray(field, dtype=np.float64)[:, order], lon_360[order]


def build_visual_sign_aligned_result(result: GlobalWeightedEofResult) -> Tuple[GlobalWeightedEofResult, np.ndarray]:
    flips = np.ones(N_SAVED_MODES, dtype=np.int8)

    eof1 = np.asarray(result.eofs[0], dtype=np.float64)
    pc1 = np.asarray(result.pcs[:, 0], dtype=np.float64)
    valid1 = np.isfinite(eof1) & np.asarray(result.valid_cell_mask, dtype=bool)
    eof1_mean = float(np.nanmean(eof1[valid1])) if np.any(valid1) else 0.0
    pc1_trend = float(np.polyfit(np.arange(pc1.size, dtype=np.float64), pc1, 1)[0]) if pc1.size >= 2 else 0.0
    mode1_score = 0
    mode1_score += 1 if eof1_mean >= 0.0 else -1
    mode1_score += 1 if pc1_trend >= 0.0 else -1
    if mode1_score < 0 or (mode1_score == 0 and eof1_mean < 0.0):
        flips[0] = -1

    lat = np.asarray(result.latitude, dtype=np.float64)
    lon = np.asarray(result.longitude, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon, lat)
    nino_mask = (
        (lat2d >= NINO34_LAT_MIN)
        & (lat2d <= NINO34_LAT_MAX)
        & (lon2d >= NINO34_LON_MIN)
        & (lon2d <= NINO34_LON_MAX)
    )
    for mode_index in (1, 2):
        field = np.asarray(result.eofs[mode_index], dtype=np.float64)
        nino_values = field[nino_mask & np.isfinite(field)]
        nino_mean = float(np.nanmean(nino_values)) if nino_values.size > 0 else 0.0
        if nino_mean < 0.0:
            flips[mode_index] = -1

    aligned_eofs = np.asarray(result.eofs, dtype=np.float64).copy()
    aligned_pcs = np.asarray(result.pcs, dtype=np.float64).copy()
    for mode_index in range(N_SAVED_MODES):
        aligned_eofs[mode_index] *= flips[mode_index]
        aligned_pcs[:, mode_index] *= flips[mode_index]

    aligned_result = GlobalWeightedEofResult(
        dataset_id=result.dataset_id,
        source_file=result.source_file,
        variable_name=result.variable_name,
        time=result.time.copy(),
        latitude=result.latitude.copy(),
        longitude=result.longitude.copy(),
        climatology=result.climatology.copy(),
        anomalies=result.anomalies.copy(),
        eofs=aligned_eofs.astype(np.float32),
        pcs=aligned_pcs.astype(np.float32),
        singular_values=result.singular_values.copy(),
        explained_variance_ratio=result.explained_variance_ratio.copy(),
        valid_cell_mask=result.valid_cell_mask.copy(),
        latitude_weights=result.latitude_weights.copy(),
        sign_flips_applied=flips.copy(),
    )
    return aligned_result, flips


def plot_eof_pc_reproduction(
    result: GlobalWeightedEofResult,
    output_path: Path,
    *,
    pacific_centered: bool,
    title_suffix: str,
) -> None:
    ccrs, cutil = try_import_cartopy()
    lon = np.asarray(result.longitude, dtype=np.float64)
    lat = np.asarray(result.latitude, dtype=np.float64)
    time_values = np.asarray(result.time, dtype="datetime64[ns]")
    time_plot = np.array(time_values.astype("datetime64[D]").astype(object))
    central_longitude = 180.0 if pacific_centered else 0.0

    if ccrs is not None:
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        axes = np.empty((N_SAVED_MODES, 2), dtype=object)
        for row in range(N_SAVED_MODES):
            axes[row, 0] = fig.add_subplot(
                N_SAVED_MODES,
                2,
                2 * row + 1,
                projection=ccrs.Robinson(central_longitude=central_longitude),
            )
            axes[row, 1] = fig.add_subplot(N_SAVED_MODES, 2, 2 * row + 2)
    else:
        fig, axes = plt.subplots(N_SAVED_MODES, 2, figsize=(16, 12), constrained_layout=True)

    for mode_index in range(N_SAVED_MODES):
        eof_field = np.asarray(result.eofs[mode_index], dtype=np.float64)
        pc_values = np.asarray(result.pcs[:, mode_index], dtype=np.float64)
        evr_pct = 100.0 * float(result.explained_variance_ratio[mode_index])

        map_ax = axes[mode_index, 0]
        vmax = float(np.nanmax(np.abs(eof_field)))
        vmax = 1.0 if not np.isfinite(vmax) or vmax == 0.0 else vmax

        if ccrs is not None:
            eof_plot, lon_plot = add_cyclic_if_available(eof_field, lon, cutil)
            lon2d, lat2d = np.meshgrid(lon_plot, lat)
            mesh = map_ax.pcolormesh(
                lon2d,
                lat2d,
                eof_plot,
                cmap="RdBu_r",
                shading="auto",
                vmin=-vmax,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            map_ax.coastlines(linewidth=0.6)
            map_ax.set_global()
        else:
            eof_plot, lon_plot = maybe_roll_longitude_for_pacific_centered(eof_field, lon, pacific_centered)
            lon2d, lat2d = np.meshgrid(lon, lat)
            if pacific_centered:
                lon2d, lat2d = np.meshgrid(lon_plot, lat)
            else:
                lon2d, lat2d = np.meshgrid(lon, lat)
            mesh = map_ax.pcolormesh(
                lon2d,
                lat2d,
                eof_plot,
                cmap="RdBu_r",
                shading="auto",
                vmin=-vmax,
                vmax=vmax,
            )
            map_ax.set_xlim(float(lon2d.min()), float(lon2d.max()))
            map_ax.set_ylim(float(lat.min()), float(lat.max()))
            map_ax.set_xlabel("Longitude")
            map_ax.set_ylabel("Latitude")

        map_ax.set_title(f"EOF{mode_index + 1} ({evr_pct:.1f}%)")
        fig.colorbar(mesh, ax=map_ax, fraction=0.046, pad=0.04)

        pc_ax = axes[mode_index, 1]
        pc_ax.axhline(0.0, color="0.4", linewidth=0.8)
        pc_ax.plot(time_plot, pc_values, color="black", linewidth=1.0)
        pc_ax.fill_between(time_plot, 0.0, pc_values, where=pc_values >= 0.0, color="red", alpha=0.35)
        pc_ax.fill_between(time_plot, 0.0, pc_values, where=pc_values < 0.0, color="blue", alpha=0.35)
        pc_ax.set_title(f"PC{mode_index + 1}")
        pc_ax.set_ylabel("Amplitude")
        if mode_index == N_SAVED_MODES - 1:
            pc_ax.set_xlabel("Time")
        pc_ax.xaxis.set_major_locator(mdates.YearLocator(base=20))
        pc_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        pc_ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"COBE2 Global Monthly-Climatology SST EOF/PC Reproduction{title_suffix}", fontsize=16)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_method_summary(
    result: GlobalWeightedEofResult,
    validation_row: Dict[str, Any],
    visual_flips: np.ndarray,
    output_path: Path,
) -> None:
    lat_step = abs(float(np.median(np.diff(np.asarray(result.latitude, dtype=np.float64))))) if result.latitude.size > 1 else float("nan")
    lon_step = abs(float(np.median(np.diff(np.asarray(result.longitude, dtype=np.float64))))) if result.longitude.size > 1 else float("nan")
    evr = 100.0 * np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    sv = np.asarray(result.singular_values[:N_SAVED_MODES], dtype=np.float64)
    lines = [
        "# COBE2 Global SST EOF Reproduction Method Summary",
        "",
        "## A. Input data",
        f"- COBE2 file path: `{result.source_file}`",
        f"- SST variable name: `{result.variable_name}`",
        f"- Time range used: {format_date(result.time[0])} to {format_date(result.time[-1])}",
        f"- Number of monthly time steps: {int(result.time.size)}",
        f"- Latitude/longitude grid description: `{int(result.latitude.size)}` latitudes by `{int(result.longitude.size)}` longitudes, approximately `{lat_step:.1f}° x {lon_step:.1f}°` global grid, longitudes normalized to `[-180, 180]` and sorted for plotting/processing.",
        "- Full global ocean domain was used.",
        "- Missing/land cells were converted to `NaN`, and EOF analysis kept only grid cells that were finite at every monthly time step.",
        "",
        "## B. Anomaly definition",
        "- We computed month-of-year climatology anomalies.",
        "- For each calendar month, the long-term mean field for that month was computed.",
        "- Each monthly field was replaced by `SST anomaly(t, lat, lon) = SST(t, lat, lon) - climatology[month(t), lat, lon]`.",
        "- No additional global or time-mean subtraction was applied after monthly climatology removal.",
        "- The analysis was not detrended.",
        "- Monthly data were used directly; annual means were not used.",
        "",
        "## C. Area weighting",
        "- Latitude weighting was used.",
        "- Exact formula: `weighted anomaly = anomaly * sqrt(cos(latitude))`.",
        "- This is standard for global gridded EOF analysis because grid-cell areas shrink toward the poles.",
        "- The EOF solve was performed on the weighted anomaly matrix, and the saved/plotted EOF maps were converted back to unweighted SST-loading form by dividing by `sqrt(cos(latitude))` at each grid cell.",
        "",
        "## D. PCA/SVD computation",
        "- Data matrix shape: `X = time x valid spatial cells`.",
        f"- In this run: `X = {int(result.time.size)} x {int(result.valid_cell_mask.sum())}`.",
        "- Valid-cell rule: keep cells finite over all time steps.",
        "- SVD/EOF solve: `X = U S V^T` conceptually, with the full singular-value spectrum recovered exactly from the time-space Gram matrix `X X^T`.",
        "- PC definition: `PC_i(t) = U[:, i] * S[i]`.",
        "- EOF definition: `EOF_i = V^T[i, :]` reshaped back to lat-lon.",
        "- EVR definition: `EVR_i = S_i^2 / sum_j S_j^2`.",
        f"- Number of modes saved/plotted: `{N_SAVED_MODES}`.",
        "",
        "## E. Results",
        f"- EOF1 EVR = {evr[0]:.1f}%",
        f"- EOF2 EVR = {evr[1]:.1f}%",
        f"- EOF3 EVR = {evr[2]:.1f}%",
        f"- Cumulative EOF1-EOF3 EVR = {100.0 * float(np.sum(result.explained_variance_ratio[:N_SAVED_MODES])):.1f}%",
        f"- Singular value EOF1 = {sv[0]:.6f}",
        f"- Singular value EOF2 = {sv[1]:.6f}",
        f"- Singular value EOF3 = {sv[2]:.6f}",
        f"- corr(PC1, Niño3.4) = {float(validation_row['pc1_nino34_corr']):.6f}",
        f"- corr(PC2, Niño3.4) = {float(validation_row['pc2_nino34_corr']):.6f}",
        f"- corr(PC3, Niño3.4) = {float(validation_row['pc3_nino34_corr']):.6f}",
        f"- PC2+PC3 multiple R = {float(validation_row['pc23_enso_multiple_r']):.6f}",
        f"- PC1+PC2+PC3 multiple R = {float(validation_row['pc123_enso_multiple_r']):.6f}",
        f"- Visual-comparison sign flips applied only in the sign-aligned Pacific-centered figure: `EOF/PC multipliers = {visual_flips.tolist()}`.",
        "",
        "## F. Comparison question for Paul",
        (
            "Our reproduced global COBE2 EOF plot has the expected broad trend-like EOF1 and a tropical-Pacific "
            "structure in the leading modes, but the EVR percentages differ from the reference figure: our "
            "EOF1/EOF2/EOF3 EVRs are 31.8%, 14.7%, and 3.3%, while the reference figure reports 24.4%, 12.4%, "
            "and 3.8%. Since we do not know the exact settings used in the reference plot, could you confirm "
            "whether that plot used the same dataset, time period, monthly vs annual anomalies, climatology "
            "baseline, detrending, latitude weighting, and EOF scaling/sign convention?"
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_enso_validation_csv(
    result: GlobalWeightedEofResult,
    nino_time: np.ndarray,
    nino_index: np.ndarray,
    output_path: Path,
) -> Dict[str, Any]:
    pcs_aligned = []
    common_time = np.asarray(result.time, dtype="datetime64[ns]")
    nino_aligned_reference = None
    for mode_index in range(N_SAVED_MODES):
        aligned_time, pc_aligned, nino_aligned = align_pc_and_index(
            common_time,
            result.pcs[:, mode_index],
            nino_time,
            nino_index,
        )
        if mode_index == 0:
            common_time = aligned_time
            nino_aligned_reference = nino_aligned
        else:
            if aligned_time.shape != common_time.shape or np.any(aligned_time != common_time):
                raise ValueError("PC and Nino3.4 time alignment mismatch across modes")
        pcs_aligned.append(pc_aligned)

    pc1 = pcs_aligned[0]
    pc2 = pcs_aligned[1]
    pc3 = pcs_aligned[2]
    nino = np.asarray(nino_aligned_reference, dtype=np.float64)

    corr1 = spatial_correlation(pc1, nino)
    corr2 = spatial_correlation(pc2, nino)
    corr3 = spatial_correlation(pc3, nino)
    pc23_r2, pc23_r = compute_multiple_regression_skill(np.column_stack([pc2, pc3]), nino)
    pc123_r2, pc123_r = compute_multiple_regression_skill(np.column_stack([pc1, pc2, pc3]), nino)

    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    sv = np.asarray(result.singular_values[:N_SAVED_MODES], dtype=np.float64)
    pc_abs_corrs = np.array([abs(corr1), abs(corr2), abs(corr3)], dtype=np.float64)
    interpretation = interpret_enso_signal(pc_abs_corrs, pc23_r, pc123_r)
    notes = (
        "Nino3.4 index computed from the same COBE2 monthly SST file after month-of-year climatology removal, "
        "with cosine-latitude area weighting over 5S-5N and 170W-120W."
    )
    row = {
        "experiment": EXPERIMENT_NAME,
        "n_time": int(common_time.size),
        "n_valid_cells": int(result.valid_cell_mask.sum()),
        "evr1": float(evr[0]),
        "evr2": float(evr[1]),
        "evr3": float(evr[2]),
        "cumulative_evr123": float(np.sum(evr)),
        "singular_value1": float(sv[0]),
        "singular_value2": float(sv[1]),
        "singular_value3": float(sv[2]),
        "pc1_nino34_corr": float(corr1),
        "pc1_nino34_abs_corr": float(abs(corr1)),
        "pc2_nino34_corr": float(corr2),
        "pc2_nino34_abs_corr": float(abs(corr2)),
        "pc3_nino34_corr": float(corr3),
        "pc3_nino34_abs_corr": float(abs(corr3)),
        "pc23_enso_r2": float(pc23_r2),
        "pc23_enso_multiple_r": float(pc23_r),
        "pc123_enso_r2": float(pc123_r2),
        "pc123_enso_multiple_r": float(pc123_r),
        "interpretation": interpretation,
        "notes": notes,
    }

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "n_time",
                "n_valid_cells",
                "evr1",
                "evr2",
                "evr3",
                "cumulative_evr123",
                "singular_value1",
                "singular_value2",
                "singular_value3",
                "pc1_nino34_corr",
                "pc1_nino34_abs_corr",
                "pc2_nino34_corr",
                "pc2_nino34_abs_corr",
                "pc3_nino34_corr",
                "pc3_nino34_abs_corr",
                "pc23_enso_r2",
                "pc23_enso_multiple_r",
                "pc123_enso_r2",
                "pc123_enso_multiple_r",
                "interpretation",
                "notes",
            ]
        )
        writer.writerow(
            [
                row["experiment"],
                row["n_time"],
                row["n_valid_cells"],
                f"{row['evr1']:.12g}",
                f"{row['evr2']:.12g}",
                f"{row['evr3']:.12g}",
                f"{row['cumulative_evr123']:.12g}",
                f"{row['singular_value1']:.12g}",
                f"{row['singular_value2']:.12g}",
                f"{row['singular_value3']:.12g}",
                f"{row['pc1_nino34_corr']:.12g}",
                f"{row['pc1_nino34_abs_corr']:.12g}",
                f"{row['pc2_nino34_corr']:.12g}",
                f"{row['pc2_nino34_abs_corr']:.12g}",
                f"{row['pc3_nino34_corr']:.12g}",
                f"{row['pc3_nino34_abs_corr']:.12g}",
                f"{row['pc23_enso_r2']:.12g}",
                f"{row['pc23_enso_multiple_r']:.12g}",
                f"{row['pc123_enso_r2']:.12g}",
                f"{row['pc123_enso_multiple_r']:.12g}",
                row["interpretation"],
                row["notes"],
            ]
        )
    return row


def print_stdout_summary(result: GlobalWeightedEofResult, validation_row: Dict[str, Any]) -> None:
    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    singular_values = np.asarray(result.singular_values[:N_SAVED_MODES], dtype=np.float64)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(
        "EVR EOF1-EOF3: "
        f"{evr[0]:.6f}, {evr[1]:.6f}, {evr[2]:.6f} | cumulative={np.sum(evr):.6f}",
        flush=True,
    )
    print(
        "Singular values EOF1-EOF3: "
        f"{singular_values[0]:.6f}, {singular_values[1]:.6f}, {singular_values[2]:.6f}",
        flush=True,
    )
    print(
        "PC vs Nino3.4 correlations: "
        f"PC1={validation_row['pc1_nino34_corr']:.6f}, "
        f"PC2={validation_row['pc2_nino34_corr']:.6f}, "
        f"PC3={validation_row['pc3_nino34_corr']:.6f}",
        flush=True,
    )
    print(f"PC2+PC3 multiple R: {validation_row['pc23_enso_multiple_r']:.6f}", flush=True)
    print(f"PC1+PC2+PC3 multiple R: {validation_row['pc123_enso_multiple_r']:.6f}", flush=True)
    print(f"Interpretation: {validation_row['interpretation']}", flush=True)
    print(f"Pacific-centered figure: {PACIFIC_CENTERED_FIGURE_FILE}", flush=True)
    print(f"Sign-aligned figure: {PACIFIC_CENTERED_SIGN_ALIGNED_FIGURE_FILE}", flush=True)
    print(f"Method summary: {METHOD_SUMMARY_FILE}", flush=True)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()

    time_values, latitude, longitude, values = load_global_cobe2_sst()
    result = solve_weighted_eofs(time_values, latitude, longitude, values)
    save_eof_dataset(result, EOF_FILE)
    save_summary_json(result, SUMMARY_FILE)
    plot_eof_pc_reproduction(result, FIGURE_FILE, pacific_centered=False, title_suffix="")
    plot_eof_pc_reproduction(
        result,
        PACIFIC_CENTERED_FIGURE_FILE,
        pacific_centered=True,
        title_suffix=" (Pacific-centered)",
    )

    nino_time, nino_index = compute_nino34_index_from_cobe2(time_values, latitude, longitude, values)
    validation_row = write_enso_validation_csv(result, nino_time, nino_index, ENSO_VALIDATION_CSV)
    visual_result, visual_flips = build_visual_sign_aligned_result(result)
    plot_eof_pc_reproduction(
        visual_result,
        PACIFIC_CENTERED_SIGN_ALIGNED_FIGURE_FILE,
        pacific_centered=True,
        title_suffix=" (Pacific-centered, visual sign-aligned)",
    )
    write_method_summary(result, validation_row, visual_flips, METHOD_SUMMARY_FILE)
    print_stdout_summary(result, validation_row)


if __name__ == "__main__":
    main()
