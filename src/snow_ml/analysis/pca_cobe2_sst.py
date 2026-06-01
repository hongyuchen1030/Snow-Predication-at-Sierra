"""
COBE2 SST PCA analysis.

This module performs PCA on COBE2 SST anomalies over a regional domain.
The workflow is:
  1. Load COBE2 SST from the provided file
  2. Normalize longitude from 0-360 to -180-180 if needed
  3. Crop to the requested region
  4. Compute the mean SST field (temporal mean at each grid point)
  5. Compute anomalies (SST - mean) for each time step
  6. Flatten anomalies into a matrix (time x space)
  7. Remove grid points with missing values
  8. Run PCA on the centered anomaly matrix
  9. Save EOFs, PCs, mean field, and metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Cobe2SstPcaResult:
    """Container for COBE2 SST PCA results."""

    time: np.ndarray  # shape (n_time,)
    latitude: np.ndarray  # shape (n_lat_crop,)
    longitude: np.ndarray  # shape (n_lon_crop,)
    grid_shape: tuple[int, int]  # (n_lat_crop, n_lon_crop)
    mean_field: np.ndarray  # shape (n_lat_crop, n_lon_crop) - temporal mean SST
    anomalies: np.ndarray  # shape (n_time, n_lat_crop, n_lon_crop) - SST anomalies
    eofs: np.ndarray  # shape (n_components, n_lat_crop, n_lon_crop) - principal components
    pcs: np.ndarray  # shape (n_time, n_components) - principal component time series
    explained_variance_ratio: np.ndarray  # shape (n_components,)
    valid_cell_count: int
    valid_cell_mask: np.ndarray  # shape (n_lat_crop, n_lon_crop)
    metadata: dict


def normalize_longitude_to_minus180_180(lon: np.ndarray) -> np.ndarray:
    """Convert longitude from 0-360 convention to -180-180 convention."""
    lon_norm = lon.copy()
    lon_norm = np.where(lon_norm > 180, lon_norm - 360, lon_norm)
    return lon_norm


def run_cobe2_sst_pca(
    *,
    cobe2_sst_file: Path | str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_components: int = 5,
) -> Cobe2SstPcaResult:
    """
    Perform PCA on COBE2 SST anomalies in a regional domain.

    Parameters
    ----------
    cobe2_sst_file : Path or str
        Path to COBE2 SST NetCDF file (sst.mon.mean.nc).
    lat_min : float
        Minimum latitude (southern bound).
    lat_max : float
        Maximum latitude (northern bound).
    lon_min : float
        Minimum longitude (western bound, -180 to 180 convention).
    lon_max : float
        Maximum longitude (eastern bound, -180 to 180 convention).
    n_components : int
        Number of PCA components to retain.

    Returns
    -------
    Cobe2SstPcaResult
        Result object containing EOFs, PCs, mean field, and metadata.
    """
    cobe2_sst_file = Path(cobe2_sst_file)
    print(f"Loading COBE2 SST from: {cobe2_sst_file}", flush=True)

    # Load the dataset
    ds = xr.open_dataset(cobe2_sst_file)
    sst_var = ds["sst"]
    lat_orig = ds["lat"].values
    lon_orig = ds["lon"].values
    time_orig = ds["time"].values

    print(f"COBE2 original dimensions: time={len(time_orig)}, lat={len(lat_orig)}, lon={len(lon_orig)}", flush=True)
    print(f"Original latitude range: {lat_orig.min():.1f} to {lat_orig.max():.1f}", flush=True)
    print(f"Original longitude range: {lon_orig.min():.1f} to {lon_orig.max():.1f}", flush=True)

    # Normalize longitude
    lon_normalized = normalize_longitude_to_minus180_180(lon_orig)
    print(f"Normalized longitude range: {lon_normalized.min():.1f} to {lon_normalized.max():.1f}", flush=True)

    # Create sorted longitude for cropping
    lon_sort_idx = np.argsort(lon_normalized)
    lon_sorted = lon_normalized[lon_sort_idx]
    sst_sorted_lon = sst_var.values[:, :, lon_sort_idx]  # (time, lat, lon)

    # Crop to requested region
    lat_mask = (lat_orig >= lat_min) & (lat_orig <= lat_max)
    lon_mask = (lon_sorted >= lon_min) & (lon_sorted <= lon_max)

    lat_crop_idx = np.where(lat_mask)[0]
    lon_crop_idx = np.where(lon_mask)[0]

    lat_crop = lat_orig[lat_crop_idx]
    lon_crop = lon_sorted[lon_crop_idx]

    # Subset SST
    sst_crop = sst_sorted_lon[:, lat_crop_idx, :][:, :, lon_crop_idx]  # (time, lat, lon)

    n_time, n_lat, n_lon = sst_crop.shape
    print(f"Cropped grid shape: ({n_lat}, {n_lon})", flush=True)
    print(f"Cropped latitude range: {lat_crop.min():.1f} to {lat_crop.max():.1f}", flush=True)
    print(f"Cropped longitude range: {lon_crop.min():.1f} to {lon_crop.max():.1f}", flush=True)
    print(f"Time steps in cropped domain: {n_time}", flush=True)
    print(f"COBE2 time range: {time_orig[0]} to {time_orig[-1]}", flush=True)

    # Convert missing values to NaN
    missing_value = float(ds["sst"].attrs.get("missing_value", 1.0e20))
    print(f"Missing value marker: {missing_value}", flush=True)
    sst_crop = np.where(sst_crop >= missing_value, np.nan, sst_crop)

    # Count missing values
    missing_per_timestep = np.isnan(sst_crop).sum(axis=(1, 2))
    print(f"Missing values per time step: min={missing_per_timestep.min()}, max={missing_per_timestep.max()}", flush=True)

    # Compute temporal mean (ignoring NaN)
    mean_field = np.nanmean(sst_crop, axis=0)  # (lat, lon)
    finite_mean_mask = np.isfinite(mean_field)
    n_finite_mean = int(finite_mean_mask.sum())
    print(f"Grid points with finite mean: {n_finite_mean} / {n_lat * n_lon}", flush=True)

    # Compute anomalies: subtract mean from each time step
    # Broadcast mean across time
    anomalies = sst_crop - mean_field[None, :, :]  # (time, lat, lon)
    print(f"Anomaly array shape: {anomalies.shape}", flush=True)

    # Flatten anomalies: (time, lat * lon)
    anomalies_flat = anomalies.reshape(n_time, -1)

    # Build valid cell mask: cells that have finite values at every time step
    valid_cell_mask_flat = np.isfinite(anomalies_flat).all(axis=0)
    valid_cell_count = int(valid_cell_mask_flat.sum())
    print(f"Grid cells finite for every time step: {valid_cell_count} / {n_lat * n_lon}", flush=True)

    if valid_cell_count == 0:
        raise ValueError("No grid cells are finite for every time step.")

    # Select only valid cells for PCA
    anomalies_pca = anomalies_flat[:, valid_cell_mask_flat].astype(np.float64)
    print(f"PCA matrix shape (time x space): {anomalies_pca.shape}", flush=True)

    # Center the anomalies (should be near-zero already, but ensure for numerical stability)
    anomalies_centered = anomalies_pca - anomalies_pca.mean(axis=0)
    centering_max_abs = float(np.abs(anomalies_centered.mean(axis=0)).max())
    print(f"Centered matrix column mean max abs: {centering_max_abs:.6e}", flush=True)

    # Perform PCA
    print(f"Running PCA with {n_components} components...", flush=True)
    pca = PCA(n_components=n_components, svd_solver="full")
    pcs = pca.fit_transform(anomalies_centered)  # (time, n_components)
    eofs_flat = pca.components_.astype(np.float32)  # (n_components, n_space)

    print(f"PCA fit complete.", flush=True)
    print(f"PC scores shape: {pcs.shape}", flush=True)
    print(f"EOF components shape: {eofs_flat.shape}", flush=True)

    # Unflatten EOFs back to spatial grid
    eofs_grid = np.full((n_components, n_lat, n_lon), np.nan, dtype=np.float32)
    for i in range(n_components):
        eofs_grid[i, :, :].flat[valid_cell_mask_flat] = eofs_flat[i, :]

    # Unflatten mean field
    mean_field_final = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    mean_field_final.flat[valid_cell_mask_flat] = anomalies_pca.mean(axis=0)

    # Unflatten valid_cell_mask to 2D
    valid_cell_mask_2d = np.zeros((n_lat, n_lon), dtype=bool)
    valid_cell_mask_2d.flat[valid_cell_mask_flat] = True

    # Print explained variance
    explained_var = pca.explained_variance_ratio_.astype(np.float32)
    cumsum_var = np.cumsum(explained_var)
    print(f"Explained variance ratio (first 5): {explained_var[:5]}", flush=True)
    print(f"Cumulative explained variance (first 5): {cumsum_var[:5]}", flush=True)

    # Metadata
    metadata = {
        "experiment_name": "cobe2_sst_pca",
        "description": (
            "PCA on COBE2 SST anomalies over a regional domain. "
            "Anomalies are computed as SST - temporal_mean_SST. "
            "PCA is fit only on grid cells with finite values for every time step."
        ),
        "source_file": str(cobe2_sst_file),
        "missing_value_marker": float(missing_value),
        "region": {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
        },
        "effective_region": {
            "lat_min": float(lat_crop.min()),
            "lat_max": float(lat_crop.max()),
            "lon_min": float(lon_crop.min()),
            "lon_max": float(lon_crop.max()),
        },
        "grid_shape": [int(n_lat), int(n_lon)],
        "n_time_steps": int(n_time),
        "time_start": str(np.datetime_as_string(np.asarray(time_orig[0], dtype="datetime64[ns]"), unit="D")),
        "time_end": str(np.datetime_as_string(np.asarray(time_orig[-1], dtype="datetime64[ns]"), unit="D")),
        "n_valid_cells": int(valid_cell_count),
        "n_components": int(n_components),
        "explained_variance_ratio": explained_var.tolist(),
        "cumulative_explained_variance_ratio": cumsum_var.tolist(),
    }

    result = Cobe2SstPcaResult(
        time=np.asarray(time_orig, dtype="datetime64[ns]"),
        latitude=lat_crop.astype(np.float32),
        longitude=lon_crop.astype(np.float32),
        grid_shape=(int(n_lat), int(n_lon)),
        mean_field=mean_field_final,
        anomalies=anomalies.astype(np.float32),
        eofs=eofs_grid,
        pcs=pcs.astype(np.float32),
        explained_variance_ratio=explained_var,
        valid_cell_count=valid_cell_count,
        valid_cell_mask=valid_cell_mask_2d,
        metadata=metadata,
    )

    ds.close()
    return result


def save_cobe2_sst_pca_results(result: Cobe2SstPcaResult, output_dir: Path) -> None:
    """
    Save PCA results to disk.

    Outputs:
      - cobe2_sst_mean.nc: Mean SST field
      - cobe2_sst_anomalies.nc: SST anomalies
      - cobe2_sst_eofs.nc: EOF spatial patterns
      - cobe2_sst_pcs.csv: PC time series
      - cobe2_sst_pca_summary.json: Metadata and explained variance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mean field
    mean_ds = xr.Dataset(
        {
            "sst_mean": (["lat", "lon"], result.mean_field),
        },
        coords={
            "lat": result.latitude,
            "lon": result.longitude,
        },
    )
    mean_ds.attrs = {"description": "COBE2 mean SST field"}
    mean_file = output_dir / "cobe2_sst_mean.nc"
    mean_ds.to_netcdf(mean_file)
    print(f"Saved mean SST: {mean_file}", flush=True)

    # Save anomalies
    anomalies_ds = xr.Dataset(
        {
            "sst_anomaly": (["time", "lat", "lon"], result.anomalies),
        },
        coords={
            "time": result.time,
            "lat": result.latitude,
            "lon": result.longitude,
        },
    )
    anomalies_ds.attrs = {"description": "COBE2 SST anomalies (SST - mean)"}
    anomalies_file = output_dir / "cobe2_sst_anomalies.nc"
    anomalies_ds.to_netcdf(anomalies_file)
    print(f"Saved anomalies: {anomalies_file}", flush=True)

    # Save EOFs
    eofs_ds = xr.Dataset(
        {
            "eof": (["component", "lat", "lon"], result.eofs),
            "valid_cell_mask": (["lat", "lon"], result.valid_cell_mask),
        },
        coords={
            "component": np.arange(result.eofs.shape[0]),
            "lat": result.latitude,
            "lon": result.longitude,
        },
    )
    eofs_ds.attrs = {
        "description": "COBE2 SST EOF patterns from PCA",
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
    }
    eofs_file = output_dir / "cobe2_sst_eofs.nc"
    eofs_ds.to_netcdf(eofs_file)
    print(f"Saved EOFs: {eofs_file}", flush=True)

    # Save PC time series
    import pandas as pd

    pc_df = pd.DataFrame(
        result.pcs,
        columns=[f"PC{i+1}" for i in range(result.pcs.shape[1])],
    )
    pc_df.insert(0, "time", np.asarray(result.time, dtype="datetime64[M]"))
    pc_file = output_dir / "cobe2_sst_pcs.csv"
    pc_df.to_csv(pc_file, index=False)
    print(f"Saved PC time series: {pc_file}", flush=True)

    # Save metadata and summary
    summary_file = output_dir / "cobe2_sst_pca_summary.json"
    with open(summary_file, "w") as f:
        json.dump(result.metadata, f, indent=2)
    print(f"Saved summary: {summary_file}", flush=True)
