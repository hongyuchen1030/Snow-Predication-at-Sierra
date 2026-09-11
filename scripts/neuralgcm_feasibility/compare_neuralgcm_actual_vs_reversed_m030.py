#!/usr/bin/env python3
"""Compare actual vs reversed-Pacific NeuralGCM precipitation distributions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
FEATURE_CSV = REPORT_DIR / "neuralgcm_wy2021_sst_state_m030_member_features.csv"

CSV_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_vs_reversed_m030_distribution_comparison.csv"
JSON_OUT = REPORT_DIR / "neuralgcm_wy2021_actual_vs_reversed_m030_distribution_comparison.json"

HIST_PNG = REPORT_DIR / "neuralgcm_wy2021_actual_vs_reversed_m030_sierra_histogram.png"
ECDF_PNG = REPORT_DIR / "neuralgcm_wy2021_actual_vs_reversed_m030_sierra_ecdf.png"
PAIRED_PNG = REPORT_DIR / "neuralgcm_wy2021_actual_vs_reversed_m030_paired_delta.png"
BOX_PNG = REPORT_DIR / "neuralgcm_wy2021_actual_vs_reversed_m030_boxplot_by_region.png"

PRISM_CSV = REPORT_DIR / "neuralgcm_wy2021_actual_m030_prism_validation.csv"
PRISM_JSON = REPORT_DIR / "neuralgcm_wy2021_actual_m030_prism_validation.json"
PRISM_PNG = REPORT_DIR / "neuralgcm_wy2021_actual_m030_prism_validation.png"

PRISM_2020 = Path("/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_2020.nc")
PRISM_2021 = Path("/global/cfs/projectdirs/m3522/datalake/PRISM/PRISM_PPT_2021.nc")

REGIONS = {
    "SIERRA_BOX": {"lat_min": 35.0, "lat_max": 42.5, "lon_min_360": 235.0, "lon_max_360": 243.0},
    "CA_NV_BOX": {"lat_min": 32.0, "lat_max": 43.0, "lon_min_360": 234.0, "lon_max_360": 246.0},
    "WEST_US_BOX": {"lat_min": 30.0, "lat_max": 50.0, "lon_min_360": 225.0, "lon_max_360": 255.0},
}
ACTUAL_OUTPUT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_neuralgcm/outputs/wy2021_actual_m030")


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


def prism_validation(actual_df: pd.DataFrame) -> list[dict]:
    if not PRISM_2020.exists() or not PRISM_2021.exists():
        return []
    datasets = [xr.open_dataset(PRISM_2020), xr.open_dataset(PRISM_2021)]
    try:
        monthly = xr.concat([ds["PPT"] for ds in datasets], dim="time").sortby("time")
        monthly = monthly.sel(time=slice("2020-09-01", "2021-03-31")).resample(time="MS").sum()
        rows = []
        for region_name, bounds in REGIONS.items():
            subset = subset_region(monthly, bounds, "lat", "lon")
            series = weighted_mean(subset, "lat")
            prism_monthly = np.asarray(series.values, dtype=float)
            prism_total = float(prism_monthly.sum())
            region_df = actual_df[actual_df["region"] == region_name].sort_values("rng_key")
            actual_totals = region_df["sep_mar_total_precip_native_units"].astype(float).to_numpy()
            rows.append(
                {
                    "region": region_name,
                    "prism_total_mm": prism_total,
                    "actual_n": int(len(actual_totals)),
                    "actual_min_native_units": float(actual_totals.min()) if len(actual_totals) else None,
                    "actual_q05_native_units": quantile(actual_totals, 0.05) if len(actual_totals) else None,
                    "actual_q50_native_units": quantile(actual_totals, 0.50) if len(actual_totals) else None,
                    "actual_q95_native_units": quantile(actual_totals, 0.95) if len(actual_totals) else None,
                    "actual_max_native_units": float(actual_totals.max()) if len(actual_totals) else None,
                    "prism_within_actual_range_native_vs_mm_not_directly_comparable": None,
                    "note": "PRISM is mm while NeuralGCM precipitation remains in native cumulative units; use only as a coarse separate reference until units are resolved.",
                }
            )
    finally:
        for ds in datasets:
            ds.close()
    return rows


def main() -> None:
    df = pd.read_csv(FEATURE_CSV)
    summary_rows = []
    paired_detail = {}

    for region_name in REGIONS:
        actual = df[(df["state"] == "actual") & (df["region"] == region_name)].sort_values("rng_key")
        reversed_df = df[(df["state"] == "reversed_pacific") & (df["region"] == region_name)].sort_values("rng_key")
        paired = actual.merge(
            reversed_df,
            on=["rng_key", "region"],
            suffixes=("_actual", "_reversed"),
        )

        actual_vals = actual["sep_mar_total_precip_native_units"].astype(float).to_numpy()
        reversed_vals = reversed_df["sep_mar_total_precip_native_units"].astype(float).to_numpy()
        deltas = paired["sep_mar_total_precip_native_units_reversed"].to_numpy() - paired["sep_mar_total_precip_native_units_actual"].to_numpy()
        pooled = pooled_std(actual_vals, reversed_vals)
        delta_mean = float(np.mean(reversed_vals) - np.mean(actual_vals))
        summary_rows.append(
            {
                "region": region_name,
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
                "paired_delta_mean": float(np.mean(deltas)) if len(deltas) else None,
                "paired_delta_std": float(np.std(deltas, ddof=0)) if len(deltas) else None,
                "paired_delta_min": float(np.min(deltas)) if len(deltas) else None,
                "paired_delta_max": float(np.max(deltas)) if len(deltas) else None,
                "fraction_paired_deltas_gt_zero": float(np.mean(deltas > 0)) if len(deltas) else None,
                "probability_reversed_gt_actual": float(np.mean(deltas > 0)) if len(deltas) else None,
                "paired_sign_flip_pvalue_two_sided": sign_flip_pvalue(deltas) if len(deltas) else None,
            }
        )
        paired_detail[region_name] = {
            "rng_keys": paired["rng_key"].astype(int).tolist(),
            "actual_totals_native_units": paired["sep_mar_total_precip_native_units_actual"].astype(float).tolist(),
            "reversed_totals_native_units": paired["sep_mar_total_precip_native_units_reversed"].astype(float).tolist(),
            "paired_by_rng_key_delta_native_units": deltas.astype(float).tolist(),
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CSV_OUT, index=False)
    JSON_OUT.write_text(
        json.dumps(
            {
                "native_precipitation_units_caveat": "NeuralGCM precipitation_cumulative_mean remains in native cumulative model units here; no mm claim is made.",
                "summary": summary_rows,
                "paired_detail": paired_detail,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sierra_actual = df[(df["state"] == "actual") & (df["region"] == "SIERRA_BOX")]["sep_mar_total_precip_native_units"].astype(float).to_numpy()
    sierra_reversed = df[(df["state"] == "reversed_pacific") & (df["region"] == "SIERRA_BOX")]["sep_mar_total_precip_native_units"].astype(float).to_numpy()
    bins = max(8, int(np.sqrt(max(len(sierra_actual), len(sierra_reversed)))))
    plt.figure(figsize=(8, 5))
    plt.hist(sierra_actual, bins=bins, alpha=0.55, label="actual", density=False)
    plt.hist(sierra_reversed, bins=bins, alpha=0.55, label="reversed_pacific", density=False)
    plt.xlabel("Sierra Sep-Mar precipitation (native cumulative units)")
    plt.ylabel("member count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=150)
    plt.close()

    xa, ya = ecdf(sierra_actual)
    xr_, yr_ = ecdf(sierra_reversed)
    plt.figure(figsize=(8, 5))
    plt.step(xa, ya, where="post", label="actual")
    plt.step(xr_, yr_, where="post", label="reversed_pacific")
    plt.xlabel("Sierra Sep-Mar precipitation (native cumulative units)")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ECDF_PNG, dpi=150)
    plt.close()

    paired_sierra = df[df["region"] == "SIERRA_BOX"].pivot(index="rng_key", columns="state", values="sep_mar_total_precip_native_units").sort_index()
    paired_delta = paired_sierra["reversed_pacific"] - paired_sierra["actual"]
    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(paired_delta.index.to_numpy(), paired_delta.to_numpy(), marker="o")
    plt.xlabel("rng_key")
    plt.ylabel("reversed - actual Sierra precipitation (native cumulative units)")
    plt.tight_layout()
    plt.savefig(PAIRED_PNG, dpi=150)
    plt.close()

    plot_df = df[df["region"].isin(REGIONS)].copy()
    plot_df["group"] = plot_df["region"] + " | " + plot_df["state"]
    ordered_groups = []
    for region_name in REGIONS:
        ordered_groups.extend([f"{region_name} | actual", f"{region_name} | reversed_pacific"])
    data = [plot_df[plot_df["group"] == group]["sep_mar_total_precip_native_units"].astype(float).to_numpy() for group in ordered_groups]
    plt.figure(figsize=(11, 5))
    plt.boxplot(data, tick_labels=ordered_groups, vert=True)
    plt.ylabel("Sep-Mar precipitation (native cumulative units)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(BOX_PNG, dpi=150)
    plt.close()

    prism_rows = prism_validation(df[df["state"] == "actual"])
    if prism_rows:
        prism_df = pd.DataFrame(prism_rows)
        prism_df.to_csv(PRISM_CSV, index=False)
        PRISM_JSON.write_text(json.dumps({"rows": prism_rows}, indent=2) + "\n", encoding="utf-8")
        plt.figure(figsize=(8, 5))
        x = np.arange(len(prism_df))
        plt.bar(x - 0.18, prism_df["actual_q50_native_units"], width=0.36, label="actual ensemble median (native units)")
        plt.bar(x + 0.18, prism_df["prism_total_mm"], width=0.36, label="PRISM total (mm)")
        plt.xticks(x, prism_df["region"])
        plt.ylabel("Native units or mm")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PRISM_PNG, dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
