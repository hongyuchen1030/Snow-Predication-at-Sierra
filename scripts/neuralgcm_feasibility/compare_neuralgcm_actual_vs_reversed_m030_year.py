#!/usr/bin/env python3
"""Compare actual vs reversed-Pacific NeuralGCM Sierra precipitation distributions for one year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-year", type=int, required=True)
    return parser.parse_args()


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var = (((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))) / (len(a) + len(b) - 2)
    return float(np.sqrt(var))


def sign_flip_pvalue(deltas: np.ndarray, n_samples: int = 20000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(deltas)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, len(deltas)))
    simulated = np.abs((signs * deltas[None, :]).mean(axis=1))
    return float((np.sum(simulated >= observed) + 1) / (n_samples + 1))


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def lon_convention(values: np.ndarray) -> str:
    return "0_to_360" if float(np.nanmin(values)) >= 0.0 else "minus180_to_180"


def convert_bounds(bounds: dict, convention: str) -> tuple[float, float]:
    if convention == "0_to_360":
        return bounds["lon_min_360"], bounds["lon_max_360"]
    lon_min = bounds["lon_min_360"] - 360.0 if bounds["lon_min_360"] > 180.0 else bounds["lon_min_360"]
    lon_max = bounds["lon_max_360"] - 360.0 if bounds["lon_max_360"] > 180.0 else bounds["lon_max_360"]
    return lon_min, lon_max


def subset_region(da: xr.DataArray, bounds: dict, lat_name: str, lon_name: str) -> xr.DataArray:
    lat_vals = np.asarray(da[lat_name].values)
    lon_vals = np.asarray(da[lon_name].values)
    lon_min, lon_max = convert_bounds(bounds, lon_convention(lon_vals))
    lat_slice = slice(bounds["lat_min"], bounds["lat_max"]) if lat_vals[0] <= lat_vals[-1] else slice(bounds["lat_max"], bounds["lat_min"])
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def weighted_mean(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    weights = xr.DataArray(np.cos(np.deg2rad(da[lat_name].values)), coords={lat_name: da[lat_name]}, dims=(lat_name,))
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    return da.weighted(weights).mean(dim=spatial_dims, skipna=True)


def compute_prism_sierra_total_mm(water_year: int) -> dict | None:
    year0 = water_year - 1
    year1 = water_year
    prism_prev = Path(f"/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_{year0}.nc")
    prism_curr = Path(f"/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_{year1}.nc")
    if not prism_prev.exists() or not prism_curr.exists():
        return None

    datasets = [xr.open_dataset(prism_prev), xr.open_dataset(prism_curr)]
    try:
        daily = xr.concat([ds["PPT"] for ds in datasets], dim="time").sortby("time")
        daily = daily.sel(time=slice(f"{year0}-09-01", f"{year1}-03-31"))
        monthly = daily.resample(time="MS").sum()
        daily_subset = subset_region(daily, REGIONS["SIERRA_BOX"], "lat", "lon")
        daily_series = weighted_mean(daily_subset, "lat")
        monthly_subset = subset_region(monthly, REGIONS["SIERRA_BOX"], "lat", "lon")
        monthly_series = weighted_mean(monthly_subset, "lat")
        daily_values = np.asarray(daily_series.values, dtype=float)
        monthly_values = np.asarray(monthly_series.values, dtype=float)
        total_mm = float(np.nansum(daily_values))
        return {
            "source": "PRISM",
            "water_year": water_year,
            "region": "SIERRA_BOX",
            "units": "mm",
            "monthly_totals_mm": {str(np.datetime_as_string(t, unit='M')): float(v) for t, v in zip(monthly_series.time.values, monthly_values, strict=True)},
            "sep_mar_total_mm": total_mm,
            "averaging_method": "cosine-latitude-weighted regional mean over SIERRA_BOX, then summed over days",
        }
    finally:
        for ds in datasets:
            ds.close()


def main() -> None:
    args = parse_args()
    feature_csv = REPORT_DIR / f"neuralgcm_wy{args.water_year}_sst_state_m030_member_features.csv"
    csv_out = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_distribution_comparison.csv"
    json_out = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_distribution_comparison.json"
    hist_png = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_histogram.png"
    ecdf_png = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_ecdf.png"
    distribution_png = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_distribution.png"
    paired_png = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_paired_delta.png"
    prism_json = REPORT_DIR / f"neuralgcm_wy{args.water_year}_sierra_observed_precip_reference.json"
    report_md = REPORT_DIR / f"neuralgcm_wy{args.water_year}_actual_vs_reversed_m030_sierra_report.md"

    df = pd.read_csv(feature_csv)
    sierra = df[df["region"] == "SIERRA_BOX"].copy()
    actual = sierra[sierra["state"] == "actual"].sort_values("rng_key")
    reversed_df = sierra[sierra["state"] == "reversed_pacific"].sort_values("rng_key")
    paired = actual.merge(reversed_df, on=["rng_key", "region"], suffixes=("_actual", "_reversed"))

    actual_vals = actual["sep_mar_total_precip_mm"].astype(float).to_numpy()
    reversed_vals = reversed_df["sep_mar_total_precip_mm"].astype(float).to_numpy()
    deltas = paired["sep_mar_total_precip_mm_reversed"].to_numpy() - paired["sep_mar_total_precip_mm_actual"].to_numpy()
    pooled = pooled_std(actual_vals, reversed_vals)
    delta_mean = float(np.mean(reversed_vals) - np.mean(actual_vals))

    row = {
        "water_year": args.water_year,
        "region": "SIERRA_BOX",
        "n_actual": int(len(actual_vals)),
        "n_reversed": int(len(reversed_vals)),
        "mean_actual": float(np.mean(actual_vals)),
        "std_actual": float(np.std(actual_vals, ddof=0)),
        "min_actual": float(np.min(actual_vals)),
        "max_actual": float(np.max(actual_vals)),
        "q05_actual": quantile(actual_vals, 0.05),
        "q25_actual": quantile(actual_vals, 0.25),
        "q50_actual": quantile(actual_vals, 0.50),
        "q75_actual": quantile(actual_vals, 0.75),
        "q95_actual": quantile(actual_vals, 0.95),
        "mean_reversed": float(np.mean(reversed_vals)),
        "std_reversed": float(np.std(reversed_vals, ddof=0)),
        "min_reversed": float(np.min(reversed_vals)),
        "max_reversed": float(np.max(reversed_vals)),
        "q05_reversed": quantile(reversed_vals, 0.05),
        "q25_reversed": quantile(reversed_vals, 0.25),
        "q50_reversed": quantile(reversed_vals, 0.50),
        "q75_reversed": quantile(reversed_vals, 0.75),
        "q95_reversed": quantile(reversed_vals, 0.95),
        "delta_mean": delta_mean,
        "standardized_delta": float(delta_mean / pooled) if np.isfinite(pooled) and pooled != 0.0 else None,
        "paired_delta_mean": float(np.mean(deltas)),
        "paired_delta_std": float(np.std(deltas, ddof=0)),
        "paired_delta_min": float(np.min(deltas)),
        "paired_delta_max": float(np.max(deltas)),
        "fraction_paired_deltas_gt_zero": float(np.mean(deltas > 0)),
        "probability_reversed_gt_actual": float(np.mean(deltas > 0)),
        "paired_sign_flip_pvalue_two_sided": sign_flip_pvalue(deltas),
    }
    prism_reference = compute_prism_sierra_total_mm(args.water_year)
    if prism_reference is None:
        prism_status = {
            "available": False,
            "blocker": f"Missing one or both PRISM files: /global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_{args.water_year - 1}.nc and /global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_{args.water_year}.nc",
        }
    else:
        prism_status = {
            "available": True,
            "sep_mar_total_mm": prism_reference["sep_mar_total_mm"],
            "inside_actual_range": bool(row["min_actual"] <= prism_reference["sep_mar_total_mm"] <= row["max_actual"]),
            "inside_actual_q05_q95": bool(row["q05_actual"] <= prism_reference["sep_mar_total_mm"] <= row["q95_actual"]),
        }
        row["prism_sep_mar_total_mm"] = prism_reference["sep_mar_total_mm"]
        row["prism_inside_actual_range"] = prism_status["inside_actual_range"]
        row["prism_inside_actual_q05_q95"] = prism_status["inside_actual_q05_q95"]

    pd.DataFrame([row]).to_csv(csv_out, index=False)
    json_out.write_text(
        json.dumps(
            {
                "precipitation_units": "mm",
                "precipitation_formula": "1000 * (area_weighted_final_cumulative_precip_m - area_weighted_initial_cumulative_precip_m)",
                "summary": row,
                "observed_precip_reference": prism_reference,
                "prism_status": prism_status,
                "paired_detail": {
                    "rng_keys": paired["rng_key"].astype(int).tolist(),
                    "actual_totals_mm": paired["sep_mar_total_precip_mm_actual"].astype(float).tolist(),
                    "reversed_totals_mm": paired["sep_mar_total_precip_mm_reversed"].astype(float).tolist(),
                    "paired_by_rng_key_delta_mm": deltas.astype(float).tolist(),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if prism_reference is not None:
        prism_json.write_text(json.dumps(prism_reference, indent=2) + "\n", encoding="utf-8")

    bins = max(8, int(np.sqrt(max(len(actual_vals), len(reversed_vals)))))
    plt.figure(figsize=(8, 5))
    plt.hist(actual_vals, bins=bins, alpha=0.75, label="actual SST", density=False, color="#1f77b4")
    if prism_reference is not None:
        plt.axvline(prism_reference["sep_mar_total_mm"], color="gray", linewidth=2, linestyle="--", label="PRISM observed")
    plt.xlabel("Sep-Mar Sierra precipitation (mm)")
    plt.ylabel("member count")
    plt.title(f"WY{args.water_year} actual SST ensemble: Sierra precipitation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_png, dpi=150)
    plt.close()

    xa, ya = ecdf(actual_vals)
    xr_, yr_ = ecdf(reversed_vals)
    plt.figure(figsize=(8, 5))
    plt.step(xa, ya, where="post", label="actual")
    plt.step(xr_, yr_, where="post", label="reversed_pacific")
    if prism_reference is not None:
        plt.axvline(prism_reference["sep_mar_total_mm"], color="gray", linewidth=2, linestyle="--", label="PRISM observed actual world")
    plt.xlabel("Sep-Mar Sierra precipitation (mm)")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ecdf_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(actual_vals, bins=bins, alpha=0.55, label="actual SST", color="#1f77b4")
    plt.hist(reversed_vals, bins=bins, alpha=0.55, label="reversed Pacific SST", color="#d62728")
    if prism_reference is not None:
        plt.axvline(prism_reference["sep_mar_total_mm"], color="gray", linewidth=2, linestyle="--", label="PRISM observed actual world")
    plt.xlabel("Sep-Mar Sierra precipitation (mm)")
    plt.ylabel("member count")
    plt.title(f"WY{args.water_year} actual vs reversed-Pacific Sierra precipitation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(distribution_png, dpi=150)
    plt.close()

    ordered_deltas = np.sort(deltas)
    plt.figure(figsize=(8, 5))
    plt.axvline(0.0, color="black", linewidth=1)
    plt.scatter(ordered_deltas, np.arange(1, len(ordered_deltas) + 1), color="#444444", s=25)
    plt.xlabel("Delta precipitation = reversed - actual (mm)")
    plt.ylabel("sorted member rank")
    plt.tight_layout()
    plt.savefig(paired_png, dpi=150)
    plt.close()

    prism_lines = []
    if prism_reference is not None:
        prism_lines = [
            f"- PRISM observed WY{args.water_year} Sierra Sep-Mar precipitation: `{prism_reference['sep_mar_total_mm']:.3f} mm`",
            f"- PRISM inside actual ensemble range: `{prism_status['inside_actual_range']}`",
            f"- PRISM inside actual 5-95% interval: `{prism_status['inside_actual_q05_q95']}`",
        ]
    else:
        prism_lines = [f"- PRISM observed WY{args.water_year} Sierra Sep-Mar precipitation: unavailable", f"  blocker: `{prism_status['blocker']}`"]

    report_md.write_text(
        "\n".join(
            [
                f"## NeuralGCM WY{args.water_year} corrected Sierra precipitation comparison",
                "",
                "This report uses existing NeuralGCM outputs only. No model rerun was performed.",
                "",
                "### Precipitation formula",
                "",
                "- `precipitation_cumulative_mean` is treated as cumulative precipitation depth in meters.",
                "- Sierra seasonal total in mm = `1000 * (area_weighted_final_value - area_weighted_initial_value)`.",
                "- Monthly totals in mm = `1000 * (month_end_cumulative - previous_month_end_cumulative)`.",
                "",
                "### Corrected Sierra summary",
                "",
                f"- actual ensemble mean/std: `{row['mean_actual']:.3f} / {row['std_actual']:.3f} mm`",
                f"- actual q05/q25/q50/q75/q95: `{row['q05_actual']:.3f} / {row['q25_actual']:.3f} / {row['q50_actual']:.3f} / {row['q75_actual']:.3f} / {row['q95_actual']:.3f} mm`",
                f"- reversed ensemble mean/std: `{row['mean_reversed']:.3f} / {row['std_reversed']:.3f} mm`",
                f"- reversed q05/q25/q50/q75/q95: `{row['q05_reversed']:.3f} / {row['q25_reversed']:.3f} / {row['q50_reversed']:.3f} / {row['q75_reversed']:.3f} / {row['q95_reversed']:.3f} mm`",
                f"- corrected `delta_mean = reversed - actual`: `{row['delta_mean']:.3f} mm`",
                f"- corrected `standardized_delta`: `{row['standardized_delta']:.6f}`",
                f"- fraction of paired deltas > 0: `{row['fraction_paired_deltas_gt_zero']:.6f}`",
                "",
                "### PRISM comparison",
                "",
                *prism_lines,
                "",
                "### Interpretation",
                "",
                f"- Reversed Pacific SST still shifts Sierra precipitation downward after corrected extraction: `{row['delta_mean'] < 0}`",
                "- PRISM validates only the actual-SST world, not the reversed-SST counterfactual.",
                "",
                "### Artifacts",
                "",
                f"- [member_features_csv]({feature_csv.as_posix()})",
                f"- [distribution_comparison_csv]({csv_out.as_posix()})",
                f"- [distribution_summary_json]({json_out.as_posix()})",
                f"- [prism_reference_json]({prism_json.as_posix()})" if prism_reference is not None else "- prism_reference_json: unavailable",
                f"- [actual_validation_histogram]({hist_png.as_posix()})",
                f"- [actual_vs_reversed_distribution_plot]({distribution_png.as_posix()})",
                f"- [paired_delta_plot]({paired_png.as_posix()})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
