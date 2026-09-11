#!/usr/bin/env python3
"""Validate and aggregate common-ocean Pacific anomaly-reversal runs across years."""

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

SECONDS_PER_6H = 6 * 3600
STATE_NAMES = ("actual", "neutral", "reversed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--enso-json", type=Path, default=None, help="Optional JSON mapping IC year or water year to ENSO label.")
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


def month_labels_for_year(year: int) -> list[str]:
    next_year = year + 1
    return [f"{year}-09", f"{year}-10", f"{year}-11", f"{year}-12", f"{next_year}-01", f"{next_year}-02", f"{next_year}-03"]


def detect_prate_conversion(ds: xr.Dataset) -> dict[str, Any]:
    prate = ds["PRATEsfc"]
    units = str(prate.attrs.get("units", ""))
    counts_name = "counts" if "counts" in ds else ("counts" if "counts" in ds.coords else None)
    result: dict[str, Any] = {
        "units": units,
        "counts_available": counts_name is not None,
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
    result["assumption"] = "counts metadata missing; calendar-month seconds would be required."
    return result


def ace2_monthly_precip_totals(ds: xr.Dataset) -> xr.DataArray:
    info = detect_prate_conversion(ds)
    seconds = xr.DataArray(np.asarray(info["seconds_per_month"], dtype=np.float64), coords={"time": ds["time"]}, dims=("time",))
    return ds["PRATEsfc"].isel(sample=0).astype(np.float64) * seconds


def ace2_monthly_means(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    return ds[var_name].isel(sample=0).astype(np.float64)


def compute_region_series(ds: xr.Dataset, month_labels: list[str], var_name: str, precip_totals_mm: xr.DataArray | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    lat_name, lon_name = "lat", "lon"
    for region_name, bounds in REGIONS.items():
        if var_name == "PRATEsfc" and precip_totals_mm is not None:
            total_subset = region_subset(precip_totals_mm, lat_name, lon_name, bounds)
            total_series = weighted_spatial_mean(total_subset, lat_name)
            prate_subset = region_subset(ace2_monthly_means(ds, "PRATEsfc"), lat_name, lon_name, bounds)
            prate_series = weighted_spatial_mean(prate_subset, lat_name)
            monthly_total_values = [float(v) for v in np.asarray(total_series.values, dtype=float)]
            monthly_prate_values = [float(v) for v in np.asarray(prate_series.values, dtype=float)]
            out[region_name] = {
                "monthly_total_mm": dict(zip(month_labels, monthly_total_values, strict=True)),
                "monthly_mean_prate": dict(zip(month_labels, monthly_prate_values, strict=True)),
                "sep_mar_total_mm": float(np.nansum(monthly_total_values)),
                "sep_mar_mean_prate": float(np.nanmean(monthly_prate_values)),
            }
        else:
            subset = region_subset(ace2_monthly_means(ds, var_name), lat_name, lon_name, bounds)
            series = weighted_spatial_mean(subset, lat_name)
            monthly_values = [float(v) for v in np.asarray(series.values, dtype=float)]
            out[region_name] = {
                "monthly_mean": dict(zip(month_labels, monthly_values, strict=True)),
                "sep_mar_mean": float(np.nanmean(monthly_values)),
            }
    return out


def output_path(year: int, state: str) -> Path:
    return ACE2_OUTPUTS / f"pacific_anomaly_reversal_{year}_common_ocean" / f"{state}_season" / "monthly_mean_predictions.nc"


def forcing_paths(year: int, state: str) -> list[Path]:
    root = ACE2_ASSETS / "forcing_perturbations" / f"pacific_anomaly_reversal_{year}_common_ocean" / state
    return [root / f"forcing_{year}.nc", root / f"forcing_{year + 1}.nc"]


def validation_paths(year: int) -> tuple[Path, Path]:
    return (
        REPORT_DIR / f"pacific_anomaly_reversal_{year}_common_ocean_validation.csv",
        REPORT_DIR / f"pacific_anomaly_reversal_{year}_common_ocean_validation.json",
    )


def seasonal_mean_surface_temperature(paths: list[Path], year: int) -> xr.DataArray:
    start = np.datetime64(f"{year}-09-01T00:00:00")
    end = np.datetime64(f"{year + 1}-03-31T18:00:00")
    datasets = [xr.open_dataset(path) for path in paths]
    try:
        combined = xr.concat([ds["surface_temperature"] for ds in datasets], dim="time")
        seasonal = combined.sel(time=slice(start, end)).mean(dim="time", skipna=True)
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


def load_prism_for_year(year: int, report_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
    prism_paths = [
        Path(f"/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_{year}.nc"),
        Path(f"/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_{year + 1}.nc"),
    ]
    existing = [path for path in prism_paths if path.exists()]
    report: dict[str, Any] = {
        "ic_year": year,
        "water_year": year + 1,
        "status": "missing",
        "paths_checked": [str(path) for path in prism_paths],
    }
    if len(existing) < 2:
        report["blocker"] = f"PRISM files for {year} and {year + 1} are not both available locally."
        return report, None

    datasets = [xr.open_dataset(path) for path in existing]
    try:
        arrays = []
        for ds in datasets:
            if "PPT" not in ds.data_vars:
                raise KeyError("Expected PRISM precipitation variable 'PPT'.")
            arrays.append(ds["PPT"])
        data = xr.concat(arrays, dim="time").sortby("time")
        data = data.sel(time=slice(f"{year}-09-01", f"{year + 1}-03-31")).astype(np.float64)
        monthly = data.resample(time="MS").sum()
        month_labels = [str(v)[:7] for v in np.asarray(monthly["time"].values)]
        expected = month_labels_for_year(year)
        if month_labels != expected:
            raise ValueError(f"Unexpected PRISM month labels for {year}: {month_labels}")
        template = datasets[0]
        report.update(
            {
                "status": "available",
                "source_selected": "PRISM daily precipitation",
                "file_paths": [str(path) for path in existing],
                "variable_name": "PPT",
                "units": str(template["PPT"].attrs.get("units", "")),
                "time_coverage_used": [f"{year}-09-01", f"{year + 1}-03-31"],
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
                "monthly_total_mm": dict(zip(expected, monthly_values, strict=True)),
                "sep_mar_total_mm": float(np.nansum(monthly_values)),
            }
    finally:
        for ds in datasets:
            ds.close()

    (report_dir / f"observed_precip_source_{year}_validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Observed Precipitation Source Report for {year}",
        "",
        f"- Status: `{report['status']}`",
        f"- Source selected: `{report.get('source_selected')}`",
        f"- Paths: `{report.get('file_paths', report.get('paths_checked'))}`",
    ]
    if report["status"] == "available":
        lines.extend(
            [
                f"- Variable name: `{report['variable_name']}`",
                f"- Units: `{report['units']}`",
                f"- Time coverage used: `{report['time_coverage_used'][0]}` to `{report['time_coverage_used'][1]}`",
                f"- Conversion used: {report['conversion_used']}",
            ]
        )
    else:
        lines.append(f"- Blocker: {report.get('blocker')}")
    (report_dir / f"observed_precip_source_{year}_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, regional


def load_enso_labels(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {str(k): str(v) for k, v in payload.items()}


def plot_sierra_totals(df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sierra = df[df["region"] == "SIERRA_BOX"].copy()
    if sierra.empty:
        return
    years = sierra["IC_year"].astype(int).tolist()
    x = np.arange(len(years))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    ax.bar(x - width, sierra["actual_total_mm"], width=width, label="actual")
    ax.bar(x, sierra["neutral_total_mm"], width=width, label="neutral")
    ax.bar(x + width, sierra["reversed_total_mm"], width=width, label="reversed")
    prism = sierra["PRISM_total_mm_if_available"].astype(float)
    if prism.notna().any():
        ax.plot(x, prism.values, color="black", marker="o", linewidth=1.5, label="PRISM")
    ax.set_xticks(x, [str(year) for year in years])
    ax.set_ylabel("Sep-Mar precipitation total (mm)")
    ax.set_xlabel("IC year")
    ax.set_title("Sierra common-ocean SST-state response by year")
    ax.grid(True, axis="y", linewidth=0.3, color="0.75", alpha=0.6)
    ax.legend()
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_response_differences(df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sierra = df[df["region"] == "SIERRA_BOX"].copy()
    if sierra.empty:
        return
    years = sierra["IC_year"].astype(int).tolist()
    x = np.arange(len(years))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    ax.bar(x - width, sierra["actual_minus_neutral_mm"], width=width, label="actual - neutral")
    ax.bar(x, sierra["reversed_minus_neutral_mm"], width=width, label="reversed - neutral")
    ax.bar(x + width, sierra["actual_minus_reversed_mm"], width=width, label="actual - reversed")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [str(year) for year in years])
    ax.set_ylabel("Difference in Sep-Mar precipitation total (mm)")
    ax.set_xlabel("IC year")
    ax.set_title("Sierra SST-state precipitation differences by year")
    ax.grid(True, axis="y", linewidth=0.3, color="0.75", alpha=0.6)
    ax.legend()
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    enso_labels = load_enso_labels(args.enso_json)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {"years": {}}
    completed_years: list[int] = []
    failed_years: dict[int, str] = {}

    for year in args.years:
        year_key = str(year)
        aggregate_json["years"][year_key] = {}
        outputs = {state: output_path(year, state) for state in STATE_NAMES}
        missing_outputs = [str(path) for path in outputs.values() if not path.exists()]
        if missing_outputs:
            failed_years[year] = f"Missing output files: {missing_outputs}"
            aggregate_json["years"][year_key]["status"] = "missing_outputs"
            aggregate_json["years"][year_key]["missing_outputs"] = missing_outputs
            continue

        validation_csv, validation_json = validation_paths(year)
        validation_ok = validation_csv.exists() and validation_json.exists()
        month_labels = month_labels_for_year(year)
        datasets = {state: xr.open_dataset(path) for state, path in outputs.items()}
        try:
            prate_info = detect_prate_conversion(datasets["actual"])
            if not prate_info["counts_available"]:
                failed_years[year] = "PRATE counts metadata missing."
                aggregate_json["years"][year_key]["status"] = "missing_counts"
                continue

            precip_totals = {state: ace2_monthly_precip_totals(ds) for state, ds in datasets.items()}
            region_data = {
                state: {
                    "PRATEsfc": compute_region_series(ds, month_labels, "PRATEsfc", precip_totals_mm=precip_totals[state]),
                    "TMP2m": compute_region_series(ds, month_labels, "TMP2m"),
                    "VGRD10m": compute_region_series(ds, month_labels, "VGRD10m"),
                }
                for state, ds in datasets.items()
            }

            observed_report, observed_regions = load_prism_for_year(year, report_dir)
            comparison_rows: list[dict[str, Any]] = []
            if observed_regions is not None:
                for region_name in REGIONS:
                    model_monthly = [region_data["actual"]["PRATEsfc"][region_name]["monthly_total_mm"][label] for label in month_labels]
                    obs_monthly = [observed_regions[region_name]["monthly_total_mm"][label] for label in month_labels]
                    diagnostics = compare_monthly_series(model_monthly, obs_monthly)
                    comparison_rows.append(
                        {
                            "region": region_name,
                            "ace2_actual_sep_mar_total_mm": float(np.sum(model_monthly)),
                            "observed_sep_mar_total_mm": float(np.sum(obs_monthly)),
                            "ace2_minus_observed_mm": float(np.sum(model_monthly) - np.sum(obs_monthly)),
                            "ace2_over_observed": float(np.sum(model_monthly) / np.sum(obs_monthly)) if np.sum(obs_monthly) != 0 else np.nan,
                            "monthly_correlation": diagnostics["monthly_correlation"],
                            "rmse_mm": diagnostics["rmse_mm"],
                            "bias_mm": diagnostics["bias_mm"],
                        }
                    )
            pd.DataFrame(comparison_rows).to_csv(report_dir / f"common_ocean_actual_vs_observed_precip_{year}.csv", index=False)
            (report_dir / f"common_ocean_actual_vs_observed_precip_{year}.json").write_text(json.dumps({"regions": comparison_rows}, indent=2) + "\n", encoding="utf-8")

            seasonal_sst = {state: subset_pacific(seasonal_mean_surface_temperature(forcing_paths(year, state), year), "latitude", "longitude") for state in STATE_NAMES}
            sst_stats = {
                "actual_minus_neutral": finite_stats(seasonal_sst["actual"] - seasonal_sst["neutral"]),
                "reversed_minus_neutral": finite_stats(seasonal_sst["reversed"] - seasonal_sst["neutral"]),
                "actual_minus_reversed": finite_stats(seasonal_sst["actual"] - seasonal_sst["reversed"]),
            }
            seasonal_precip = {state: precip_totals[state].sum(dim="time", skipna=True) for state in STATE_NAMES}
            precip_stats = {
                "actual_minus_neutral": finite_stats(seasonal_precip["actual"] - seasonal_precip["neutral"]),
                "reversed_minus_neutral": finite_stats(seasonal_precip["reversed"] - seasonal_precip["neutral"]),
                "actual_minus_reversed": finite_stats(seasonal_precip["actual"] - seasonal_precip["reversed"]),
            }

            for region_name in REGIONS:
                actual_total = region_data["actual"]["PRATEsfc"][region_name]["sep_mar_total_mm"]
                neutral_total = region_data["neutral"]["PRATEsfc"][region_name]["sep_mar_total_mm"]
                reversed_total = region_data["reversed"]["PRATEsfc"][region_name]["sep_mar_total_mm"]
                observed_total = None
                observed_corr = None
                observed_bias = None
                if observed_regions is not None:
                    observed_total = observed_regions[region_name]["sep_mar_total_mm"]
                    row_match = next((row for row in comparison_rows if row["region"] == region_name), None)
                    if row_match is not None:
                        observed_corr = row_match["monthly_correlation"]
                        observed_bias = row_match["ace2_minus_observed_mm"]
                row = {
                    "IC_year": year,
                    "water_year": year + 1,
                    "ENSO_label_if_available": enso_labels.get(str(year), enso_labels.get(str(year + 1), "")),
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
                    "PRISM_total_mm_if_available": observed_total,
                    "actual_minus_PRISM_mm_if_available": observed_bias,
                    "monthly_corr_actual_vs_PRISM_if_available": observed_corr,
                }
                for state in STATE_NAMES:
                    row[f"{state}_sep_mar_mean_PRATEsfc"] = region_data[state]["PRATEsfc"][region_name]["sep_mar_mean_prate"]
                    row[f"{state}_sep_mar_mean_TMP2m"] = region_data[state]["TMP2m"][region_name]["sep_mar_mean"]
                    row[f"{state}_sep_mar_mean_VGRD10m"] = region_data[state]["VGRD10m"][region_name]["sep_mar_mean"]
                    for label, value in region_data[state]["PRATEsfc"][region_name]["monthly_total_mm"].items():
                        row[f"{state}_{label}_total_mm"] = value
                aggregate_rows.append(row)

            completed_years.append(year)
            aggregate_json["years"][year_key] = {
                "status": "completed",
                "validation_files_present": validation_ok,
                "prate_conversion": prate_info,
                "sst_response_stats": sst_stats,
                "precip_response_stats": precip_stats,
                "observed_status": observed_report["status"],
            }
        finally:
            for ds in datasets.values():
                ds.close()

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_csv = report_dir / "common_ocean_four_year_precip_response.csv"
    aggregate_json_path = report_dir / "common_ocean_four_year_precip_response.json"
    aggregate_df.to_csv(aggregate_csv, index=False)
    aggregate_json_path.write_text(
        json.dumps(
            {
                "completed_years": completed_years,
                "failed_years": failed_years,
                "years": aggregate_json["years"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_sierra_totals(
        aggregate_df,
        report_dir / "common_ocean_four_year_sierra_precip_response.png",
        report_dir / "common_ocean_four_year_sierra_precip_response.pdf",
    )
    plot_response_differences(
        aggregate_df,
        report_dir / "common_ocean_four_year_response_differences.png",
        report_dir / "common_ocean_four_year_response_differences.pdf",
    )

    lines = [
        "# Milestone 10 Four-Year Common-Ocean Experiment Report",
        "",
        f"- Years requested: `{args.years}`",
        f"- Years completed: `{completed_years}`",
        f"- Years failed or pending: `{failed_years}`",
        "",
    ]
    if not aggregate_df.empty:
        sierra = aggregate_df[aggregate_df["region"] == "SIERRA_BOX"].copy()
        lines.append("## Sierra Totals")
        for _, row in sierra.iterrows():
            lines.append(
                f"- {int(row['IC_year'])}: actual `{row['actual_total_mm']:.1f}` mm, neutral `{row['neutral_total_mm']:.1f}` mm, reversed `{row['reversed_total_mm']:.1f}` mm, PRISM `{row['PRISM_total_mm_if_available']}`."
            )
        lines.extend(
            [
                "",
                "## Direction Check",
            ]
        )
        an_mean = float(sierra["actual_minus_neutral_mm"].mean()) if not sierra.empty else np.nan
        rn_mean = float(sierra["reversed_minus_neutral_mm"].mean()) if not sierra.empty else np.nan
        ar_mean = float(sierra["actual_minus_reversed_mm"].mean()) if not sierra.empty else np.nan
        lines.append(f"- Mean `actual - neutral` across completed Sierra cases: `{an_mean:.1f}` mm.")
        lines.append(f"- Mean `reversed - neutral` across completed Sierra cases: `{rn_mean:.1f}` mm.")
        lines.append(f"- Mean `actual - reversed` across completed Sierra cases: `{ar_mean:.1f}` mm.")
        if rn_mean > an_mean:
            lines.append("- In the completed sample, the reversed Pacific state tends wetter than the neutral state more strongly than the actual state does.")
        else:
            lines.append("- In the completed sample, the actual Pacific state tends as wet or wetter relative to neutral than the reversed state.")
        lines.append("- Recommendation: continue only if at least two non-2020 years finish cleanly; otherwise the scientific signal is still too under-sampled.")
    else:
        lines.append("- No nonempty completed-year aggregate table was produced.")
    (report_dir / "milestone10_four_year_common_ocean_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
