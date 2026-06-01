#!/usr/bin/env python3
"""
Run an additive COBE2 monthly-climatology EOF diagnostic over an extended Pacific domain.

This experiment:
1. Loads COBE2 monthly SST.
2. Normalizes longitude to [-180, 180] and crops an extended Pacific domain.
3. Removes the month-of-year climatology.
4. Computes SVD-based EOFs/PCs.
5. Saves the leading three EOF maps, PCs, singular values, EVRs, summary JSON, figure, and ENSO validation CSV.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    compute_eof_result,
    compute_monthly_series_anomalies,
    compute_multiple_regression_skill,
    compute_weighted_regional_mean,
    ensure_runtime_on_compute_node,
    get_runtime,
    normalize_longitude_to_minus180_180,
    open_dataset_with_fallbacks,
    spatial_correlation,
)


EXPERIMENT_NAME = "cobe2_extended_pacific_monthly_climatology_eof"
DATASET_ID = "COBE2_extended_pacific"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "cobe2_extended_pacific_monthly_climatology_anomaly"
EOF_FILE = OUTPUT_DIR / "cobe2_extended_pacific_monthly_clim_sst_eofs.nc"
SUMMARY_FILE = OUTPUT_DIR / "cobe2_extended_pacific_monthly_clim_sst_summary.json"
FIGURE_FILE = OUTPUT_DIR / "cobe2_extended_pacific_monthly_clim_sst_eof123.png"
ENSO_VALIDATION_CSV = OUTPUT_DIR / "cobe2_extended_pacific_enso_pc_correlation_validation.csv"

LAT_MIN = -10.0
LAT_MAX = 45.0
LON_MIN = -180.0
LON_MAX = -110.0
NINO34_LAT_MIN = -5.0
NINO34_LAT_MAX = 5.0
NINO34_LON_MIN = -170.0
NINO34_LON_MAX = -120.0
N_SAVED_MODES = 3


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_cobe2_region(ds: xr.Dataset, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sst = np.asarray(ds["sst"].values, dtype=np.float64)
    lat_orig = np.asarray(ds["lat"].values, dtype=np.float64)
    lon_orig = normalize_longitude_to_minus180_180(np.asarray(ds["lon"].values, dtype=np.float64))
    lon_sort_idx = np.argsort(lon_orig)
    lon_sorted = lon_orig[lon_sort_idx]
    sst_sorted = sst[:, :, lon_sort_idx]

    lat_mask = (lat_orig >= lat_min) & (lat_orig <= lat_max)
    lon_mask = (lon_sorted >= lon_min) & (lon_sorted <= lon_max)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError("Extended Pacific crop is empty for requested domain")

    lat_crop = lat_orig[lat_mask]
    lon_crop = lon_sorted[lon_mask]
    sst_crop = sst_sorted[:, lat_mask, :][:, :, lon_mask]
    missing_value = float(ds["sst"].attrs.get("missing_value", 1.0e20))
    sst_crop = np.where(sst_crop >= missing_value, np.nan, sst_crop)
    return lat_crop, lon_crop, sst_crop


def load_extended_pacific_result():
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        lat_crop, lon_crop, sst_crop = crop_cobe2_region(ds, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    return compute_eof_result(
        dataset_id=DATASET_ID,
        source_file=str(COBE2_SST_FILE),
        time_values=time_values,
        latitude=lat_crop,
        longitude=lon_crop,
        values=sst_crop,
        notes=[
            "month-of-year climatology removed before EOF analysis",
            "domain extends over the Pacific to include the full Nino3.4 region",
        ],
    )


def compute_nino34_index_from_cobe2() -> Tuple[np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(COBE2_SST_FILE) as ds:
        time_values = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = normalize_longitude_to_minus180_180(np.asarray(ds["lon"].values, dtype=np.float64))
        lon_sort_idx = np.argsort(longitude)
        longitude = longitude[lon_sort_idx]
        values = np.asarray(ds["sst"].values, dtype=np.float64)[:, :, lon_sort_idx]
        missing_value = float(ds["sst"].attrs.get("missing_value", 1.0e20))
        values = np.where(values >= missing_value, np.nan, values)

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


def align_pc_and_index(pc_time: np.ndarray, pc_values: np.ndarray, index_time: np.ndarray, index_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def interpret_enso_signal(pc1_abs: float, pc2_abs: float, pc3_abs: float, pc23_r: float, pc123_r: float) -> str:
    if pc2_abs >= 0.7 and pc2_abs > pc3_abs:
        return "ENSO primarily aligned with PC2."
    if pc3_abs >= 0.7 and pc3_abs > pc2_abs:
        return "ENSO primarily aligned with PC3."
    if pc2_abs >= 0.4 and pc3_abs >= 0.4 and pc23_r >= 0.7:
        return "ENSO represented in combined PC2-PC3 subspace."
    if np.isfinite(pc123_r) and np.isfinite(pc23_r) and pc123_r >= pc23_r + 0.1:
        return "ENSO-related variability is distributed across the leading PCs, including PC1."
    if max(pc1_abs, pc2_abs, pc3_abs, pc23_r, pc123_r) < 0.4:
        return "no clear ENSO relationship in leading EOFs."
    return "ENSO-related variability is distributed across the leading PCs, including PC1."


def save_eof_dataset(result, output_path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "eof": (("mode", "lat", "lon"), result.eofs[:N_SAVED_MODES]),
            "pc": (("time", "mode"), result.pcs[:, :N_SAVED_MODES]),
            "singular_value": (("mode",), result.singular_values[:N_SAVED_MODES]),
            "explained_variance_ratio": (("mode",), result.explained_variance_ratio[:N_SAVED_MODES]),
            "valid_cell_mask": (("lat", "lon"), result.valid_cell_mask),
        },
        coords={
            "mode": np.arange(1, N_SAVED_MODES + 1, dtype=np.int32),
            "time": result.time,
            "lat": result.latitude,
            "lon": result.longitude,
        },
        attrs={
            "experiment_name": EXPERIMENT_NAME,
            "description": "COBE2 extended-Pacific monthly-climatology SST EOF diagnostics",
            "source_file": result.source_file,
        },
    )
    ds.to_netcdf(output_path)


def save_summary_json(result, n_time: int, output_path: Path) -> None:
    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "dataset_id": result.dataset_id,
        "source_file": result.source_file,
        "domain_bounds": {
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
        },
        "n_time": int(n_time),
        "n_valid_cells": int(result.valid_cell_mask.sum()),
        "singular_values": [float(value) for value in result.singular_values[:N_SAVED_MODES]],
        "explained_variance_ratio": [float(value) for value in evr],
        "cumulative_explained_variance_ratio": [float(value) for value in np.cumsum(evr)],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_eof_maps(result, output_path: Path) -> None:
    lon = np.asarray(result.longitude, dtype=np.float64)
    lat = np.asarray(result.latitude, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(1, N_SAVED_MODES, figsize=(16, 5), constrained_layout=True)
    for mode_index, ax in enumerate(np.atleast_1d(axes)):
        field = np.asarray(result.eofs[mode_index], dtype=np.float64)
        vmax = float(np.nanmax(np.abs(field)))
        vmax = 1.0 if not np.isfinite(vmax) or vmax == 0.0 else vmax
        mesh = ax.pcolormesh(lon2d, lat2d, field, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.add_patch(
            Rectangle(
                (NINO34_LON_MIN, NINO34_LAT_MIN),
                NINO34_LON_MAX - NINO34_LON_MIN,
                NINO34_LAT_MAX - NINO34_LAT_MIN,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                linestyle="--",
            )
        )
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel("Longitude")
        if mode_index == 0:
            ax.set_ylabel("Latitude")
        ax.set_title(f"EOF{mode_index + 1} | EVR={result.explained_variance_ratio[mode_index]:.3f}")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("COBE2 extended-Pacific monthly-climatology SST EOFs\nDashed box: Nino 3.4 region")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_enso_validation_csv(result, nino_time: np.ndarray, nino_index: np.ndarray, output_path: Path) -> Dict[str, Any]:
    common_time = np.asarray(result.time, dtype="datetime64[ns]")
    pcs_aligned = []
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
    nino = nino_aligned_reference

    corr1 = spatial_correlation(pc1, nino)
    corr2 = spatial_correlation(pc2, nino)
    corr3 = spatial_correlation(pc3, nino)
    pc23_r2, pc23_r = compute_multiple_regression_skill(np.column_stack([pc2, pc3]), nino)
    pc123_r2, pc123_r = compute_multiple_regression_skill(np.column_stack([pc1, pc2, pc3]), nino)

    evr = np.asarray(result.explained_variance_ratio[:N_SAVED_MODES], dtype=np.float64)
    sv = np.asarray(result.singular_values[:N_SAVED_MODES], dtype=np.float64)
    interpretation = interpret_enso_signal(abs(corr1), abs(corr2), abs(corr3), pc23_r, pc123_r)
    notes = "Nino3.4 index computed from COBE2 monthly SST anomalies with cosine-latitude area weighting."
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


def print_stdout_summary(result, validation_row: Dict[str, Any]) -> None:
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
        f"PC1={validation_row['pc1_nino34_corr']:.6f} (abs={validation_row['pc1_nino34_abs_corr']:.6f}), "
        f"PC2={validation_row['pc2_nino34_corr']:.6f} (abs={validation_row['pc2_nino34_abs_corr']:.6f}), "
        f"PC3={validation_row['pc3_nino34_corr']:.6f} (abs={validation_row['pc3_nino34_abs_corr']:.6f})",
        flush=True,
    )
    print(f"PC2+PC3 multiple R: {validation_row['pc23_enso_multiple_r']:.6f}", flush=True)
    print(f"PC1+PC2+PC3 multiple R: {validation_row['pc123_enso_multiple_r']:.6f}", flush=True)
    print(f"Interpretation: {validation_row['interpretation']}", flush=True)


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dir()

    result = load_extended_pacific_result()
    save_eof_dataset(result, EOF_FILE)
    save_summary_json(result, int(result.time.shape[0]), SUMMARY_FILE)
    plot_eof_maps(result, FIGURE_FILE)

    nino_time, nino_index = compute_nino34_index_from_cobe2()
    validation_row = write_enso_validation_csv(result, nino_time, nino_index, ENSO_VALIDATION_CSV)
    print_stdout_summary(result, validation_row)


if __name__ == "__main__":
    main()
