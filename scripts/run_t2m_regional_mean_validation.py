#!/usr/bin/env python3
"""
Regional-mean March t2m anomaly validation between WUS models and ERA5.

This script:
1. Reads pre-computed March t2m anomaly fields
2. Computes regional mean over valid land cells for each year
3. Aligns years 1980-2013
4. Computes metrics: correlation, RMSE, mean bias
5. Saves results and plots
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import linregress

# Paths
ANOMALY_ROOT = Path("artifacts/t2m_anomaly_validation")
WUS_ANOMALY_DIR = ANOMALY_ROOT / "wus_t2m_march_anomalies"
ERA5_ANOMALY_FILE = ANOMALY_ROOT / "era5_t2m_march_anomalies" / "era5_land_historical_overlap_march_t2m_anomalies_on_wusd3_d01.nc"
OUTPUT_ROOT = Path("artifacts/t2m_regional_mean_validation")
PLOTS_DIR = OUTPUT_ROOT / "plots"
RESULTS_CSV_PATH = OUTPUT_ROOT / "regional_mean_validation_results.csv"
TIMESERIES_CSV_PATH = OUTPUT_ROOT / "regional_mean_timeseries.csv"
REPORT_PATH = OUTPUT_ROOT / "regional_mean_validation_report.md"

# Year range
YEAR_START = 1980
YEAR_END = 2013
TARGET_YEARS = list(range(YEAR_START, YEAR_END + 1))


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    """Open netCDF with fallback engines."""
    errors = []
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}: {exc}")
    raise RuntimeError(f"Failed to open {path}: {'; '.join(errors)}")


def compute_regional_mean(anomalies: xr.DataArray, landmask: xr.DataArray) -> xr.DataArray:
    """
    Compute regional mean over valid land cells.
    
    Args:
        anomalies: (year, lat*, lon*) anomaly field
        landmask: (lat*, lon*) binary landmask (1=land, 0=ocean/masked)
    
    Returns:
        (year,) regional mean time series
    """
    # Apply landmask and compute spatial mean
    # Handle both 'lat2d'/'lon2d' and 'lat'/'lon' dimension names
    spatial_dims = []
    for dim in anomalies.dims:
        if 'lat' in dim or 'lon' in dim:
            spatial_dims.append(dim)
    
    if not spatial_dims:
        raise ValueError(f"No spatial dimensions found in anomalies: {anomalies.dims}")
    
    masked = anomalies.where(landmask == 1)
    regional_mean = masked.mean(dim=spatial_dims, skipna=True)
    return regional_mean


def load_wus_anomalies_historical() -> Dict[str, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Load WUS historical anomaly files.
    
    Returns:
        Dict mapping model name to (anomalies, landmask)
    """
    results = {}
    
    for path in sorted(WUS_ANOMALY_DIR.glob("*_historical_*.nc")):
        print(f"Loading WUS: {path.name}")
        with open_dataset_with_fallbacks(path) as ds:
            # Extract model name from filename
            # e.g., "ec-earth3_r1i1p1f1_2_historical_bc_historical_march_t2_anomalies.nc"
            model_name = path.stem
            
            # Get anomaly variable - look for 2D spatial anomaly data
            # Check both 'lat'/'lon' and 'lat2d'/'lon2d' dimension names
            data_vars = {}
            for k, v in ds.data_vars.items():
                dims = v.dims
                if ('lat2d' in dims and 'lon2d' in dims) or ('lat' in dims and 'lon' in dims):
                    if 'year' in dims:  # Must have time dimension
                        data_vars[k] = v
            
            if not data_vars:
                raise ValueError(f"No spatial data variables found in {path}")
            
            # Use the first spatial data variable (usually the anomaly)
            anomaly_var = list(data_vars.keys())[0]
            anomalies = ds[anomaly_var].load()
            
            # Get landmask if available
            if "landmask" in ds.coords:
                landmask = ds["landmask"].load()
            elif "landmask" in ds.data_vars:
                landmask = ds["landmask"].load()
            else:
                # Create landmask from non-NaN values
                landmask = (~anomalies.isel(year=0).isnull()).astype(int)
            
            results[model_name] = (anomalies, landmask)
    
    return results


def load_era5_anomalies() -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Load ERA5 anomalies.
    
    Returns:
        (anomalies, landmask)
    """
    print(f"Loading ERA5: {ERA5_ANOMALY_FILE.name}")
    if not ERA5_ANOMALY_FILE.exists():
        raise FileNotFoundError(f"ERA5 anomaly file not found: {ERA5_ANOMALY_FILE}")
    
    with open_dataset_with_fallbacks(ERA5_ANOMALY_FILE) as ds:
        # Get main data variable with spatial dimensions
        # Check both 'lat'/'lon' and 'lat2d'/'lon2d' dimension names
        data_vars = {}
        for k, v in ds.data_vars.items():
            dims = v.dims
            if ('lat2d' in dims and 'lon2d' in dims) or ('lat' in dims and 'lon' in dims):
                if 'year' in dims:  # Must have time dimension
                    data_vars[k] = v
        
        if not data_vars:
            raise ValueError(f"No spatial data variables found in {ERA5_ANOMALY_FILE}")
        
        anomaly_var = list(data_vars.keys())[0]
        anomalies = ds[anomaly_var].load()
        
        # Get landmask if available
        if "landmask" in ds.coords:
            landmask = ds["landmask"].load()
        elif "landmask" in ds.data_vars:
            landmask = ds["landmask"].load()
        else:
            # Create landmask from non-NaN values
            landmask = (~anomalies.isel(year=0).isnull()).astype(int)
        
        return anomalies, landmask


def align_years(anomalies: xr.DataArray, source_years: List[int]) -> xr.DataArray:
    """
    Select only target years and ensure they match.
    
    Args:
        anomalies: Input anomaly array
        source_years: List of years in the source data
    
    Returns:
        Anomalies for target years only
    """
    available = [y for y in TARGET_YEARS if y in source_years]
    if not available:
        raise ValueError(f"No overlap with target years {TARGET_YEARS}")
    
    try:
        # Try to select by year coordinate if it exists
        aligned = anomalies.sel(year=available)
    except KeyError:
        # Fall back to positional indexing if year coordinate doesn't exist
        indices = [source_years.index(y) for y in available]
        aligned = anomalies.isel(year=indices)
        aligned["year"] = available
    
    return aligned


def compute_metrics(
    wus_mean: xr.DataArray,
    era5_mean: xr.DataArray,
) -> Dict[str, float]:
    """
    Compute validation metrics.
    
    Args:
        wus_mean: Regional mean time series from WUS model
        era5_mean: Regional mean time series from ERA5
    
    Returns:
        Dict with correlation, rmse, and mean_bias
    """
    # Convert to numpy for computation
    wus_vals = np.asarray(wus_mean.values, dtype=float)
    era5_vals = np.asarray(era5_mean.values, dtype=float)
    
    # Remove NaN values
    mask = ~(np.isnan(wus_vals) | np.isnan(era5_vals))
    wus_clean = wus_vals[mask]
    era5_clean = era5_vals[mask]
    
    if len(wus_clean) < 2:
        return {
            "correlation": np.nan,
            "rmse": np.nan,
            "mean_bias": np.nan,
            "valid_years": 0,
        }
    
    # Correlation
    corr = np.corrcoef(wus_clean, era5_clean)[0, 1]
    
    # RMSE
    rmse = np.sqrt(np.mean((wus_clean - era5_clean) ** 2))
    
    # Mean bias
    bias = np.mean(wus_clean - era5_clean)
    
    return {
        "correlation": float(corr),
        "rmse": float(rmse),
        "mean_bias": float(bias),
        "valid_years": int(len(wus_clean)),
    }


def create_timeseries_plot(
    model_name: str,
    years: List[int],
    wus_mean: xr.DataArray,
    era5_mean: xr.DataArray,
    metrics: Dict[str, float],
) -> Path:
    """Create time-series comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(years, wus_mean.values, 'o-', label='WUS Model', linewidth=2, markersize=4)
    ax.plot(years, era5_mean.values, 's-', label='ERA5', linewidth=2, markersize=4)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('March t2m Anomaly (K)', fontsize=12)
    ax.set_title(f'{model_name} - March T2m Anomaly Time Series\n' +
                f'Correlation: {metrics["correlation"]:.3f}, ' +
                f'RMSE: {metrics["rmse"]:.3f} K, ' +
                f'Bias: {metrics["mean_bias"]:.3f} K', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / f"{model_name}_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_scatter_plot(
    model_name: str,
    wus_mean: xr.DataArray,
    era5_mean: xr.DataArray,
    metrics: Dict[str, float],
) -> Path:
    """Create scatter plot comparison."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(era5_mean.values, wus_mean.values, s=100, alpha=0.6, edgecolors='k')
    
    # Add 1:1 line
    min_val = min(era5_mean.min().values, wus_mean.min().values)
    max_val = max(era5_mean.max().values, wus_mean.max().values)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1')
    
    ax.set_xlabel('ERA5 Anomaly (K)', fontsize=12)
    ax.set_ylabel('WUS Model Anomaly (K)', fontsize=12)
    ax.set_title(f'{model_name} - March T2m Anomaly Scatter\n' +
                f'Correlation: {metrics["correlation"]:.3f}', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / f"{model_name}_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_correlation_summary_plot(
    model_correlations: Dict[str, float],
) -> Path:
    """Create summary bar plot of correlations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(model_correlations.keys())
    correlations = [model_correlations[m] for m in models]
    
    bars = ax.bar(range(len(models)), correlations, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{corr:.3f}',
               ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Regional-Mean March T2m Anomaly: WUS vs ERA5 Correlations', fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    ax.set_ylim([-1, 1])
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "correlations_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def save_results_csv(
    all_results: List[Dict[str, object]],
) -> None:
    """Save summary metrics to CSV."""
    if not all_results:
        print("No results to save")
        return
    
    with open(RESULTS_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Saved results to {RESULTS_CSV_PATH}")


def save_timeseries_csv(
    timeseries_data: Dict[str, Dict[str, List]],
) -> None:
    """Save regional mean time series to CSV."""
    # Flatten the nested structure
    rows = []
    for model_name, data in timeseries_data.items():
        for year, wus_val, era5_val in zip(
            data["years"],
            data["wus_mean"],
            data["era5_mean"],
        ):
            rows.append({
                "model": model_name,
                "year": year,
                "wus_mean": wus_val,
                "era5_mean": era5_val,
            })
    
    if rows:
        with open(TIMESERIES_CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved time series to {TIMESERIES_CSV_PATH}")


def write_validation_report(
    all_results: List[Dict[str, object]],
    model_dirs: Dict[str, Path],
) -> None:
    """Write markdown validation report."""
    lines = [
        "# Regional-Mean March T2m Anomaly Validation Report",
        "",
        "## Overview",
        f"This report presents regional-mean validation of historical WUS model March t2m anomalies",
        f"against ERA5-Land, aligned for years {YEAR_START}-{YEAR_END}.",
        "",
        "## Input Data",
        f"- WUS anomalies: `{WUS_ANOMALY_DIR}`",
        f"- ERA5 anomaly: `{ERA5_ANOMALY_FILE.name}`",
        "",
        "## Metrics",
        "- **Correlation**: Pearson correlation coefficient between WUS and ERA5 regional means",
        "- **RMSE**: Root mean squared error between WUS and ERA5 regional means (K)",
        "- **Mean Bias**: Mean difference WUS - ERA5 (K)",
        "",
        "## Results Summary",
        "",
        "| Model | Correlation | RMSE (K) | Mean Bias (K) | Valid Years |",
        "|-------|-------------|----------|---------------|-------------|",
    ]
    
    for result in sorted(all_results, key=lambda x: str(x["model"])):
        lines.append(
            f"| {result['model']} | {result['correlation']:.4f} | "
            f"{result['rmse']:.4f} | {result['mean_bias']:.4f} | {result['valid_years']} |"
        )
    
    lines.extend([
        "",
        "## Plots",
        "- `correlations_summary.png`: Bar plot of correlations across all models",
        "- `{model}_timeseries.png`: Time-series comparison for each model",
        "- `{model}_scatter.png`: Scatter plot comparison for each model",
        "",
        "## Data Files",
        f"- `{RESULTS_CSV_PATH.name}`: Summary metrics",
        f"- `{TIMESERIES_CSV_PATH.name}`: Year-by-year regional means",
    ])
    
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Wrote validation report to {REPORT_PATH}")


def main():
    print(f"Starting regional-mean t2m anomaly validation", flush=True)
    print(f"Target years: {YEAR_START}-{YEAR_END}", flush=True)
    
    # Ensure output dirs exist
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load anomalies
    wus_anomalies_dict = load_wus_anomalies_historical()
    era5_anomalies, era5_landmask = load_era5_anomalies()
    
    print(f"Found {len(wus_anomalies_dict)} WUS historical models", flush=True)
    
    # Get ERA5 years from coordinates or infer
    if "year" in era5_anomalies.dims:
        era5_years = era5_anomalies["year"].values.tolist()
    else:
        raise ValueError("Cannot determine ERA5 years from dataset")
    
    print(f"ERA5 available years: {sorted(era5_years)}", flush=True)
    
    # Perform validation
    all_results: List[Dict[str, object]] = []
    model_correlations: Dict[str, float] = {}
    timeseries_data: Dict[str, Dict[str, List]] = {}
    
    for model_name, (wus_anom, wus_landmask) in sorted(wus_anomalies_dict.items()):
        print(f"\nProcessing {model_name}", flush=True)
        
        # Get WUS years from coordinates or infer
        if "year" in wus_anom.dims:
            wus_years = wus_anom["year"].values.tolist()
        else:
            raise ValueError(f"Cannot determine WUS years from {model_name} dataset")
        
        print(f"  WUS available years: {sorted(wus_years)}", flush=True)
        
        # Align years
        wus_aligned = align_years(wus_anom, wus_years)
        era5_aligned = align_years(era5_anomalies, era5_years)
        aligned_years = list(wus_aligned["year"].values)
        
        print(f"  Aligned years: {sorted(aligned_years)} (n={len(aligned_years)})", flush=True)
        
        # Compute regional means
        wus_mean = compute_regional_mean(wus_aligned, wus_landmask)
        era5_mean = compute_regional_mean(era5_aligned, era5_landmask)
        
        # Compute metrics
        metrics = compute_metrics(wus_mean, era5_mean)
        print(f"  Correlation: {metrics['correlation']:.4f}", flush=True)
        print(f"  RMSE: {metrics['rmse']:.4f} K", flush=True)
        print(f"  Mean Bias: {metrics['mean_bias']:.4f} K", flush=True)
        
        # Store results
        all_results.append({
            "model": model_name,
            "correlation": metrics["correlation"],
            "rmse": metrics["rmse"],
            "mean_bias": metrics["mean_bias"],
            "valid_years": metrics["valid_years"],
        })
        
        model_correlations[model_name] = metrics["correlation"]
        timeseries_data[model_name] = {
            "years": aligned_years,
            "wus_mean": wus_mean.values.tolist(),
            "era5_mean": era5_mean.values.tolist(),
        }
        
        # Create plots
        print(f"  Creating time-series plot", flush=True)
        create_timeseries_plot(model_name, aligned_years, wus_mean, era5_mean, metrics)
        
        print(f"  Creating scatter plot", flush=True)
        create_scatter_plot(model_name, wus_mean, era5_mean, metrics)
    
    # Create summary plot
    print(f"\nCreating correlation summary plot", flush=True)
    create_correlation_summary_plot(model_correlations)
    
    # Save results
    print(f"\nSaving results", flush=True)
    save_results_csv(all_results)
    save_timeseries_csv(timeseries_data)
    write_validation_report(all_results, {m: Path(f"model_{m}") for m in wus_anomalies_dict.keys()})
    
    print(f"\nCompleted regional-mean validation", flush=True)
    print(f"Results saved to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
