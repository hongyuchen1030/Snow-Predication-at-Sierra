#!/usr/bin/env python3
"""Validate the completed ACE2 common-ocean Pacific anomaly-reversal experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
ACE2_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2")
ACE2_ASSETS = ACE2_ROOT / "assets"
ACE2_OUTPUTS = ACE2_ROOT / "outputs"
REPORT_DIR = REPO / "artifacts" / "ace2_era5_feasibility"

OUTPUT_PATHS = {
    "actual": ACE2_OUTPUTS / "pacific_anomaly_reversal_2020_common_ocean" / "actual_season" / "monthly_mean_predictions.nc",
    "neutral": ACE2_OUTPUTS / "pacific_anomaly_reversal_2020_common_ocean" / "neutral_season" / "monthly_mean_predictions.nc",
    "reversed": ACE2_OUTPUTS / "pacific_anomaly_reversal_2020_common_ocean" / "reversed_season" / "monthly_mean_predictions.nc",
}

FORCING_PATHS = {
    "actual": [
        ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020_common_ocean" / "actual" / "forcing_2020.nc",
        ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020_common_ocean" / "actual" / "forcing_2021.nc",
    ],
    "neutral": [
        ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020_common_ocean" / "neutral" / "forcing_2020.nc",
        ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020_common_ocean" / "neutral" / "forcing_2021.nc",
    ],
    "reversed": [
        ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020_common_ocean" / "reversed" / "forcing_2020.nc",
        ACE2_ASSETS / "forcing_perturbations" / "pacific_anomaly_reversal_2020_common_ocean" / "reversed" / "forcing_2021.nc",
    ],
}

PRISM_PATHS = [
    Path("/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_2020.nc"),
    Path("/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_2021.nc"),
]

REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
    "CA_NV_BOX": {"lat_min": 32.0, "lat_max": 43.0, "lon_min_360": 234.0, "lon_max_360": 246.0},
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
}

PACIFIC_DOMAIN = {
    "lat_min": -10.0,
    "lat_max": 60.0,
    "lon_min_360": 120.0,
    "lon_max_360": 280.0,
}

ACE2_START = np.datetime64("2020-09-01T00:00:00")
ACE2_END = np.datetime64("2021-03-31T18:00:00")
MONTH_LABELS = ["2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02", "2021-03"]
SECONDS_PER_6H = 6 * 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--actual-output", type=Path, default=OUTPUT_PATHS["actual"])
    parser.add_argument("--neutral-output", type=Path, default=OUTPUT_PATHS["neutral"])
    parser.add_argument("--reversed-output", type=Path, default=OUTPUT_PATHS["reversed"])
    parser.add_argument("--actual-forcing", nargs=2, type=Path, default=FORCING_PATHS["actual"])
    parser.add_argument("--neutral-forcing", nargs=2, type=Path, default=FORCING_PATHS["neutral"])
    parser.add_argument("--reversed-forcing", nargs=2, type=Path, default=FORCING_PATHS["reversed"])
    parser.add_argument("--prism-paths", nargs="+", type=Path, default=PRISM_PATHS)
    return parser.parse_args()


def lon_convention(values: np.ndarray) -> str:
    return "0_to_360" if float(np.nanmin(values)) >= 0.0 else "minus180_to_180"


def convert_bounds(bounds: dict[str, float], convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"] - 360.0 if bounds["lon_min_360"] > 180.0 else bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"] - 360.0 if bounds["lon_max_360"] > 180.0 else bounds["lon_max_360"]
    return lon_min, lon_max


def region_subset(da: xr.DataArray, lat_name: str, lon_name: str, bounds: dict[str, float]) -> xr.DataArray:
    convention = lon_convention(np.asarray(da[lon_name].values))
    lon_min, lon_max = convert_bounds(bounds, convention)
    lat_values = np.asarray(da[lat_name].values)
    lon_values = np.asarray(da[lon_name].values)
    lat_slice = slice(bounds["lat_min"], bounds["lat_max"]) if lat_values[0] <= lat_values[-1] else slice(bounds["lat_max"], bounds["lat_min"])
    lon_slice = slice(lon_min, lon_max) if lon_values[0] <= lon_values[-1] else slice(lon_max, lon_min)
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def weighted_spatial_mean(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    weights = xr.DataArray(np.cos(np.deg2rad(da[lat_name].values)), coords={lat_name: da[lat_name]}, dims=(lat_name,))
    spatial_dims = [dim for dim in da.dims if dim not in {"time", "sample"}]
    return da.weighted(weights).mean(dim=spatial_dims, skipna=True)


def detect_prate_conversion(ds: xr.Dataset) -> dict[str, Any]:
    prate = ds["PRATEsfc"]
    units = str(prate.attrs.get("units", ""))
    counts_name = "counts" if "counts" in ds else ("counts" if "counts" in ds.coords else None)
    valid_time = ds["valid_time"] if "valid_time" in ds.coords else None
    result: dict[str, Any] = {
        "units": units,
        "counts_available": counts_name is not None,
        "valid_time_available": valid_time is not None,
        "conversion_method": None,
        "seconds_per_month": [],
        "assumption": None,
        "reliable": False,
    }
    if counts_name is not None:
        counts = np.asarray(ds[counts_name].values, dtype=float).reshape(-1)
        seconds = counts * SECONDS_PER_6H
        result["conversion_method"] = "counts_x_6h"
        result["seconds_per_month"] = [float(v) for v in seconds.tolist()]
        result["assumption"] = "counts represent the number of 6-hour source steps contributing to each monthly mean."
        result["reliable"] = True
        return result
    result["conversion_method"] = "calendar_month_seconds"
    month_seconds = [30 * 86400, 31 * 86400, 30 * 86400, 31 * 86400, 31 * 86400, 28 * 86400, 31 * 86400]
    result["seconds_per_month"] = [float(v) for v in month_seconds]
    result["assumption"] = "counts metadata missing; using calendar-month seconds for Sep 2020 through Mar 2021."
    result["reliable"] = False
    return result


def ace2_monthly_precip_totals(ds: xr.Dataset) -> xr.DataArray:
    info = detect_prate_conversion(ds)
    seconds = xr.DataArray(np.asarray(info["seconds_per_month"], dtype=np.float64), coords={"time": ds["time"]}, dims=("time",))
    prate = ds["PRATEsfc"].isel(sample=0).astype(np.float64)
    return prate * seconds


def ace2_monthly_means(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    return ds[var_name].isel(sample=0).astype(np.float64)


def month_labels_from_valid_time(ds: xr.Dataset) -> list[str]:
    valid = np.asarray(ds["valid_time"].values).reshape(-1)
    return [str(v)[:7] for v in valid]


def compute_region_series_ace2(ds: xr.Dataset, var_name: str, precip_totals_mm: xr.DataArray | None = None) -> dict[str, dict[str, Any]]:
    lat_name, lon_name = "lat", "lon"
    out: dict[str, dict[str, Any]] = {}
    for region_name, bounds in REGIONS.items():
        if var_name == "PRATEsfc" and precip_totals_mm is not None:
            subset = region_subset(precip_totals_mm, lat_name, lon_name, bounds)
            series = weighted_spatial_mean(subset, lat_name)
            monthly_values = [float(v) for v in np.asarray(series.values, dtype=float)]
            out[region_name] = {
                "monthly_total_mm": dict(zip(MONTH_LABELS, monthly_values, strict=True)),
                "sep_mar_total_mm": float(np.nansum(monthly_values)),
                "sep_mar_mean_prate": float(np.nanmean(weighted_spatial_mean(region_subset(ace2_monthly_means(ds, "PRATEsfc"), lat_name, lon_name, bounds), lat_name).values)),
            }
        else:
            subset = region_subset(ace2_monthly_means(ds, var_name), lat_name, lon_name, bounds)
            series = weighted_spatial_mean(subset, lat_name)
            monthly_values = [float(v) for v in np.asarray(series.values, dtype=float)]
            out[region_name] = {
                "monthly_mean": dict(zip(MONTH_LABELS, monthly_values, strict=True)),
                "sep_mar_mean": float(np.nanmean(monthly_values)),
            }
    return out


def load_prism_observed(prism_paths: list[Path]) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
    existing = [path for path in prism_paths if path.exists()]
    report: dict[str, Any] = {
        "source_selected": None,
        "status": "missing",
        "paths_checked": [str(path) for path in prism_paths],
    }
    if len(existing) < 2:
        report["blocker"] = "PRISM 2020/2021 files not both available locally."
        return report, None

    datasets = [xr.open_dataset(path) for path in existing]
    try:
        data_arrays = []
        for ds in datasets:
            if "PPT" not in ds.data_vars:
                raise KeyError("Expected PRISM precipitation variable 'PPT'.")
            data_arrays.append(ds["PPT"])
        data = xr.concat(data_arrays, dim="time").sortby("time")
        data = data.sel(time=slice("2020-09-01", "2021-03-31")).astype(np.float64)
        monthly = data.resample(time="MS").sum()
        monthly = monthly.sel(time=monthly["time"].dt.strftime("%Y-%m").isin(MONTH_LABELS))
        month_labels = [str(v)[:7] for v in np.asarray(monthly["time"].values)]
        if month_labels != MONTH_LABELS:
            raise ValueError(f"Unexpected PRISM month labels: {month_labels}")
        template = datasets[0]
        report.update(
            {
                "source_selected": "PRISM daily precipitation",
                "status": "available",
                "file_paths": [str(path) for path in existing],
                "variable_name": "PPT",
                "units": str(template["PPT"].attrs.get("units", "")),
                "time_coverage_used": ["2020-09-01", "2021-03-31"],
                "grid_domain": {
                    "lat_min": float(template["lat"].min()),
                    "lat_max": float(template["lat"].max()),
                    "lon_min": float(template["lon"].min()),
                    "lon_max": float(template["lon"].max()),
                },
                "conversion_used": "daily PRISM PPT is already in mm; monthly totals are calendar-month sums of daily values.",
                "classification": "observed",
            }
        )

        regional: dict[str, dict[str, Any]] = {}
        for region_name, bounds in REGIONS.items():
            subset = region_subset(monthly, "lat", "lon", bounds)
            series = weighted_spatial_mean(subset, "lat")
            monthly_values = [float(v) for v in np.asarray(series.values, dtype=float)]
            regional[region_name] = {
                "monthly_total_mm": dict(zip(MONTH_LABELS, monthly_values, strict=True)),
                "sep_mar_total_mm": float(np.nansum(monthly_values)),
            }
    finally:
        for ds in datasets:
            ds.close()
    return report, regional


def compare_monthly_series(model_values: list[float], observed_values: list[float]) -> dict[str, Any]:
    model = np.asarray(model_values, dtype=float)
    obs = np.asarray(observed_values, dtype=float)
    diff = model - obs
    corr = float(np.corrcoef(model, obs)[0, 1]) if model.size >= 2 else None
    return {
        "monthly_correlation": corr,
        "rmse_mm": float(np.sqrt(np.nanmean(diff ** 2))),
        "bias_mm": float(np.nanmean(diff)),
    }


def seasonal_mean_surface_temperature(paths: list[Path]) -> xr.DataArray:
    datasets = [xr.open_dataset(path) for path in paths]
    try:
        combined = xr.concat([ds["surface_temperature"] for ds in datasets], dim="time")
        seasonal = combined.sel(time=slice(ACE2_START, ACE2_END)).mean(dim="time", skipna=True)
        return seasonal.astype(np.float64)
    finally:
        for ds in datasets:
            ds.close()


def subset_pacific(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    lat_vals = np.asarray(da[lat_name].values)
    lon_vals = np.asarray(da[lon_name].values)
    lat_slice = slice(PACIFIC_DOMAIN["lat_min"], PACIFIC_DOMAIN["lat_max"]) if lat_vals[0] <= lat_vals[-1] else slice(PACIFIC_DOMAIN["lat_max"], PACIFIC_DOMAIN["lat_min"])
    lon_slice = slice(PACIFIC_DOMAIN["lon_min_360"], PACIFIC_DOMAIN["lon_max_360"]) if lon_vals[0] <= lon_vals[-1] else slice(PACIFIC_DOMAIN["lon_max_360"], PACIFIC_DOMAIN["lon_min_360"])
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def finite_stats(da: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(da.values, dtype=float)
    finite = np.isfinite(values)
    return {
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "min": float(np.nanmin(values)) if finite.any() else None,
        "max": float(np.nanmax(values)) if finite.any() else None,
        "mean": float(np.nanmean(values)) if finite.any() else None,
        "std": float(np.nanstd(values)) if finite.any() else None,
    }


def signed_area_fraction(da: xr.DataArray) -> dict[str, float]:
    values = np.asarray(da.values, dtype=float)
    lats = np.asarray(da[da.dims[-2]].values, dtype=float)
    lat_weights = np.cos(np.deg2rad(lats))
    weights_2d = np.repeat(lat_weights[:, None], values.shape[-1], axis=1)
    finite = np.isfinite(values)
    total = float(weights_2d[finite].sum())
    if total == 0.0:
        return {"positive_fraction": 0.0, "negative_fraction": 0.0}
    return {
        "positive_fraction": float(weights_2d[(values > 0) & finite].sum() / total),
        "negative_fraction": float(weights_2d[(values < 0) & finite].sum() / total),
    }


def plot_three_panel_maps(fields: dict[str, xr.DataArray], lat_name: str, lon_name: str, output_png: Path, output_pdf: Path, title_prefix: str, colorbar_label: str, cmap: str, symmetric: bool, extent: dict[str, float]) -> dict[str, Any]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(field.values, dtype=float) for field in fields.values()]
    finite_values = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if np.isfinite(arr).any()])
    if finite_values.size == 0:
        raise ValueError("No finite values available for plotting.")
    vmax = float(np.nanmax(np.abs(finite_values))) if symmetric else float(np.nanmax(finite_values))
    vmin = -vmax if symmetric else float(np.nanmin(finite_values))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    stats: dict[str, Any] = {}
    for ax, (name, field) in zip(axes, fields.items(), strict=True):
        lon_vals = np.asarray(field[lon_name].values)
        lat_vals = np.asarray(field[lat_name].values)
        image = ax.pcolormesh(lon_vals, lat_vals, np.asarray(field.values, dtype=float), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        lon_min = extent.get("lon_min", extent.get("lon_min_360"))
        lon_max = extent.get("lon_max", extent.get("lon_max_360"))
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(extent["lat_min"], extent["lat_max"])
        ax.set_title(f"{title_prefix}\n{name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.25, color="0.7", alpha=0.5)
        stats[name] = finite_stats(field)
        stats[name]["signed_area_fraction"] = signed_area_fraction(field)
    fig.colorbar(image, ax=axes, shrink=0.9, label=colorbar_label)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)
    return stats


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "actual": args.actual_output,
        "neutral": args.neutral_output,
        "reversed": args.reversed_output,
    }
    forcings = {
        "actual": list(args.actual_forcing),
        "neutral": list(args.neutral_forcing),
        "reversed": list(args.reversed_forcing),
    }

    output_datasets = {name: xr.open_dataset(path) for name, path in outputs.items()}
    try:
        actual_ds = output_datasets["actual"]
        month_labels = month_labels_from_valid_time(actual_ds)
        if month_labels != MONTH_LABELS:
            raise ValueError(f"Unexpected ACE2 month labels: {month_labels}")

        prate_info = detect_prate_conversion(actual_ds)
        precip_totals = {name: ace2_monthly_precip_totals(ds) for name, ds in output_datasets.items()}
        region_data = {
            name: {
                "PRATEsfc": compute_region_series_ace2(ds, "PRATEsfc", precip_totals_mm=precip_totals[name]),
                "TMP2m": compute_region_series_ace2(ds, "TMP2m"),
                "VGRD10m": compute_region_series_ace2(ds, "VGRD10m"),
            }
            for name, ds in output_datasets.items()
        }

        observed_report, observed_regions = load_prism_observed(list(args.prism_paths))
        observed_md = report_dir / "observed_precip_source_2020_validation_report.md"
        observed_json = report_dir / "observed_precip_source_2020_validation_report.json"
        observed_json.write_text(json.dumps(observed_report, indent=2) + "\n", encoding="utf-8")
        observed_lines = [
            "# Observed Precipitation Source Report",
            "",
            f"- Status: `{observed_report['status']}`",
            f"- Source selected: `{observed_report.get('source_selected')}`",
            f"- Paths: `{observed_report.get('file_paths', observed_report.get('paths_checked'))}`",
        ]
        if observed_report["status"] == "available":
            observed_lines.extend(
                [
                    f"- Variable name: `{observed_report['variable_name']}`",
                    f"- Units: `{observed_report['units']}`",
                    f"- Time coverage used: `{observed_report['time_coverage_used'][0]}` to `{observed_report['time_coverage_used'][1]}`",
                    f"- Conversion used: {observed_report['conversion_used']}",
                    f"- Classification: `{observed_report['classification']}`",
                ]
            )
        else:
            observed_lines.append(f"- Blocker: {observed_report.get('blocker', 'No local observed/reanalysis precipitation source found.')}")
        write_markdown(observed_md, observed_lines)

        comparison_rows: list[dict[str, Any]] = []
        comparison_json: dict[str, Any] = {"regions": {}}
        if observed_regions is not None:
            for region_name in REGIONS:
                model_monthly = [region_data["actual"]["PRATEsfc"][region_name]["monthly_total_mm"][label] for label in MONTH_LABELS]
                obs_monthly = [observed_regions[region_name]["monthly_total_mm"][label] for label in MONTH_LABELS]
                diagnostics = compare_monthly_series(model_monthly, obs_monthly)
                row = {
                    "region": region_name,
                    "ace2_actual_sep_mar_total_mm": float(np.sum(model_monthly)),
                    "observed_sep_mar_total_mm": float(np.sum(obs_monthly)),
                    "ace2_minus_observed_mm": float(np.sum(model_monthly) - np.sum(obs_monthly)),
                    "ace2_over_observed": float(np.sum(model_monthly) / np.sum(obs_monthly)) if np.sum(obs_monthly) != 0 else np.nan,
                    "monthly_correlation": diagnostics["monthly_correlation"],
                    "rmse_mm": diagnostics["rmse_mm"],
                    "bias_mm": diagnostics["bias_mm"],
                }
                for label, model_value, obs_value in zip(MONTH_LABELS, model_monthly, obs_monthly, strict=True):
                    row[f"ace2_{label}_mm"] = model_value
                    row[f"observed_{label}_mm"] = obs_value
                comparison_rows.append(row)
                comparison_json["regions"][region_name] = row
        comparison_csv_path = report_dir / "common_ocean_actual_vs_observed_precip_2020.csv"
        comparison_json_path = report_dir / "common_ocean_actual_vs_observed_precip_2020.json"
        pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)
        comparison_json_path.write_text(json.dumps(comparison_json, indent=2) + "\n", encoding="utf-8")

        seasonal_sst = {name: subset_pacific(seasonal_mean_surface_temperature(paths), "latitude", "longitude") for name, paths in forcings.items()}
        sst_diffs = {
            "actual - neutral": seasonal_sst["actual"] - seasonal_sst["neutral"],
            "reversed - neutral": seasonal_sst["reversed"] - seasonal_sst["neutral"],
            "actual - reversed": seasonal_sst["actual"] - seasonal_sst["reversed"],
        }
        sst_stats = plot_three_panel_maps(
            sst_diffs,
            lat_name="latitude",
            lon_name="longitude",
            output_png=report_dir / "common_ocean_sst_anomaly_maps_2020.png",
            output_pdf=report_dir / "common_ocean_sst_anomaly_maps_2020.pdf",
            title_prefix="Sep-Mar mean SST anomaly (K)",
            colorbar_label="K",
            cmap="RdBu_r",
            symmetric=True,
            extent=PACIFIC_DOMAIN,
        )
        (report_dir / "common_ocean_sst_anomaly_summary.json").write_text(json.dumps({"seasonal_mean_stats": sst_stats}, indent=2) + "\n", encoding="utf-8")

        seasonal_precip = {name: precip_totals[name].sum(dim="time", skipna=True) for name in precip_totals}
        west_us_extent = {"lat_min": 25.0, "lat_max": 55.0, "lon_min": 220.0, "lon_max": 260.0}
        precip_diffs = {
            "actual - neutral": seasonal_precip["actual"] - seasonal_precip["neutral"],
            "reversed - neutral": seasonal_precip["reversed"] - seasonal_precip["neutral"],
            "actual - reversed": seasonal_precip["actual"] - seasonal_precip["reversed"],
        }
        precip_stats = plot_three_panel_maps(
            precip_diffs,
            lat_name="lat",
            lon_name="lon",
            output_png=report_dir / "common_ocean_precip_response_maps_2020.png",
            output_pdf=report_dir / "common_ocean_precip_response_maps_2020.pdf",
            title_prefix="Sep-Mar total precipitation response (mm)",
            colorbar_label="mm",
            cmap="BrBG",
            symmetric=True,
            extent=west_us_extent,
        )
        (report_dir / "common_ocean_precip_response_summary.json").write_text(json.dumps({"seasonal_total_stats": precip_stats}, indent=2) + "\n", encoding="utf-8")

        regional_rows: list[dict[str, Any]] = []
        for region_name in REGIONS:
            actual_total = region_data["actual"]["PRATEsfc"][region_name]["sep_mar_total_mm"]
            neutral_total = region_data["neutral"]["PRATEsfc"][region_name]["sep_mar_total_mm"]
            reversed_total = region_data["reversed"]["PRATEsfc"][region_name]["sep_mar_total_mm"]
            regional_rows.append(
                {
                    "region": region_name,
                    "actual_total_mm": actual_total,
                    "neutral_total_mm": neutral_total,
                    "reversed_total_mm": reversed_total,
                    "actual_minus_neutral_mm": actual_total - neutral_total,
                    "reversed_minus_neutral_mm": reversed_total - neutral_total,
                    "actual_minus_reversed_mm": actual_total - reversed_total,
                    "actual_over_neutral": actual_total / neutral_total if neutral_total != 0 else np.nan,
                    "reversed_over_neutral": reversed_total / neutral_total if neutral_total != 0 else np.nan,
                    "actual_over_reversed": actual_total / reversed_total if reversed_total != 0 else np.nan,
                }
            )
        regional_df = pd.DataFrame(regional_rows)
        regional_df.to_csv(report_dir / "common_ocean_regional_precip_response_2020.csv", index=False)

        sierra = regional_df.loc[regional_df["region"] == "SIERRA_BOX"].iloc[0].to_dict()
        coherence_note = "tentatively coherent"
        if precip_stats["actual - neutral"]["signed_area_fraction"]["positive_fraction"] < 0.55 and precip_stats["actual - neutral"]["signed_area_fraction"]["negative_fraction"] < 0.55:
            coherence_note = "mixed-sign / less coherent"

        report_lines = [
            "# Milestone 9 Common-Ocean Validation Report",
            "",
            "## Conversion",
            f"- `PRATEsfc` units: `{prate_info['units']}`.",
            f"- Conversion method: `{prate_info['conversion_method']}`.",
            f"- Seconds represented by each ACE2 month: `{prate_info['seconds_per_month']}`.",
            f"- Assumption: {prate_info['assumption']}",
            "",
            "## Observed Comparison",
            f"- Observed/reference source: `{observed_report.get('source_selected')}` with status `{observed_report['status']}`.",
        ]
        if observed_regions is not None:
            for row in comparison_rows:
                report_lines.append(
                    f"- {row['region']}: ACE2 actual `{row['ace2_actual_sep_mar_total_mm']:.1f}` mm vs observed `{row['observed_sep_mar_total_mm']:.1f}` mm, ratio `{row['ace2_over_observed']:.3f}`, monthly corr `{row['monthly_correlation']:.3f}`, RMSE `{row['rmse_mm']:.1f}` mm."
                )
        else:
            report_lines.append(f"- Blocker: {observed_report.get('blocker')}")
        report_lines.extend(
            [
                "",
                "## Sierra Outcomes",
                f"- Sierra Sep-Mar total precipitation: actual `{sierra['actual_total_mm']:.1f}` mm, neutral `{sierra['neutral_total_mm']:.1f}` mm, reversed `{sierra['reversed_total_mm']:.1f}` mm.",
                f"- `actual - neutral`: `{sierra['actual_minus_neutral_mm']:.1f}` mm.",
                f"- `reversed - neutral`: `{sierra['reversed_minus_neutral_mm']:.1f}` mm.",
                f"- `actual - reversed`: `{sierra['actual_minus_reversed_mm']:.1f}` mm.",
                "",
                "## Map Checks",
                f"- SST anomaly maps are finite for all three comparisons: `{all(v['nan_count'] == 0 for v in sst_stats.values())}`.",
                f"- Precipitation response maps are finite for all three comparisons: `{all(v['nan_count'] == 0 for v in precip_stats.values())}`.",
                f"- Spatial coherence check on `actual - neutral`: `{coherence_note}` based on sign-weighted area fractions `{precip_stats['actual - neutral']['signed_area_fraction']}`.",
                "- Physical plausibility: the response is usable as a first diagnostic, but it still needs broader year-to-year validation before treating the SST-state experiment as production-ready.",
                "",
                "## Next Step",
                "- Recommended next experiment: repeat the common-ocean actual/neutral/reversed validation on a second diagnostic year before broadening amplitudes or regions.",
            ]
        )
        write_markdown(report_dir / "milestone9_common_ocean_validation_report.md", report_lines)

    finally:
        for ds in output_datasets.values():
            ds.close()


if __name__ == "__main__":
    main()
