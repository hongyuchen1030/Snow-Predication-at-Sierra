#!/usr/bin/env python3
"""Phase 1 WY1998 SWE translator feasibility using existing NeuralGCM outputs only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
REPORT_DIR = REPO / "artifacts" / "neuralgcm_feasibility"
SCRATCH_ROOT = Path("/pscratch/sd/h/hyvchen")
NEURALGCM_ROOT = SCRATCH_ROOT / "Snow-Predication-at-Sierra_neuralgcm"
OUTPUT_ROOT = NEURALGCM_ROOT / "outputs"
PREP_ROOT = NEURALGCM_ROOT / "assets" / "prepared_inputs"

UCLA_SWE_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/UCLA_WUS_SNOWv1")
ERA5_LAND_T2M_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5-Land/2m_temperature")

WATER_YEAR = 1998
YEAR0 = WATER_YEAR - 1
YEAR1 = WATER_YEAR
START_DATE = f"{YEAR0}-09-01"
END_DATE = f"{YEAR1}-03-31"
ACTUAL_OUTPUT_DIR = OUTPUT_ROOT / "wy1998_actual_m030"
REVERSED_OUTPUT_DIR = OUTPUT_ROOT / "wy1998_reversed_pacific_m030"
PREDICTION_FILE_NAME = "predictions_6h.nc"

SIERRA_BOX = {
    "lat_min": 35.0,
    "lat_max": 42.0,
    "lon_min_360": 237.5,
    "lon_max_360": 242.0,
}

WEATHER_FORCING_CSV = REPORT_DIR / "neuralgcm_wy1998_swe_phase1_sierra_daily_forcing.csv"
DAILY_SWE_CSV = REPORT_DIR / "neuralgcm_wy1998_swe_phase1_daily_swe_timeseries.csv"
MEMBER_RESULTS_CSV = REPORT_DIR / "neuralgcm_wy1998_snow17_prms_swe_member_results.csv"
COMPARISON_JSON = REPORT_DIR / "neuralgcm_wy1998_snow17_prms_swe_distribution_comparison.json"
FINAL_REPORT_MD = REPORT_DIR / "neuralgcm_wy1998_snow17_prms_swe_report.md"
AVAILABLE_MODELS_MD = REPORT_DIR / "swe_translator_available_models_report.md"
FORCING_VAR_MD = REPORT_DIR / "neuralgcm_swe_forcing_variable_report.md"
ACTUAL_HIST_PNG = REPORT_DIR / "neuralgcm_wy1998_swe_actual_histogram.png"
ACTUAL_VS_REVERSED_PNG = REPORT_DIR / "neuralgcm_wy1998_swe_actual_vs_reversed.png"
PAIRED_DELTA_PNG = REPORT_DIR / "neuralgcm_wy1998_swe_paired_delta.png"


@dataclass(frozen=True)
class TranslatorParams:
    snowfall_temp_c: float = 0.0
    rain_temp_c: float = 2.0
    melt_temp_c: float = 0.0
    degree_day_factor_mm_per_c_day: float = 3.0


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var = (((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))) / (len(a) + len(b) - 2)
    return float(np.sqrt(var))


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


def load_era5_land_daily_sierra_t2m_c() -> pd.Series:
    yearly_paths = [
        ERA5_LAND_T2M_ROOT / f"ERA5_{YEAR0}_2m_temperature.nc",
        ERA5_LAND_T2M_ROOT / f"ERA5_{YEAR1}_2m_temperature.nc",
    ]
    daily_parts: list[pd.Series] = []
    for path in yearly_paths:
        with xr.open_dataset(path) as ds:
            regional = subset_region(ds["t2m"], SIERRA_BOX, "latitude", "longitude")
            regional = regional.sel(time=slice(START_DATE, END_DATE))
            series_k = weighted_mean(regional, "latitude")
            daily_k = series_k.resample(time="1D").mean().load()
            daily_parts.append(
                pd.Series(
                    data=np.asarray(daily_k.values, dtype=float) - 273.15,
                    index=pd.to_datetime(daily_k.time.values),
                    name="era5land_t2m_c",
                )
            )
    return pd.concat(daily_parts).sort_index().loc[START_DATE:END_DATE]


def load_ucla_april1_sierra_swe_mm() -> float:
    path = UCLA_SWE_ROOT / f"WUS_UCLA_SR_v01_ALL_0_agg_16_WY{WATER_YEAR}_SD_SWE_SCA_POST.nc"
    with xr.open_dataset(path) as ds:
        swe = ds["SWE_Post"]
        if "Stats" in swe.dims:
            swe = swe.isel(Stats=0, drop=True)
        swe = swe.sel(time=np.datetime64(f"{WATER_YEAR}-04-01"))
        regional = subset_region(swe, SIERRA_BOX, "Latitude", "Longitude")
        weights = xr.DataArray(np.cos(np.deg2rad(regional["Latitude"].values)), coords={"Latitude": regional["Latitude"]}, dims=("Latitude",))
        mean_m = regional.weighted(weights).mean(dim=("Latitude", "Longitude"), skipna=True)
        return float(mean_m.item()) * 1000.0


def daily_precip_from_cumulative_mm(series_m: xr.DataArray) -> pd.Series:
    ordered = series_m.sortby("time")
    values_m = np.asarray(ordered.values, dtype=float)
    if values_m.size == 0:
        raise ValueError("Empty cumulative precipitation series.")
    increments_mm = 1000.0 * np.diff(values_m, prepend=values_m[0])
    increments_mm[0] = 1000.0 * max(values_m[0], 0.0)
    increments_mm = np.clip(increments_mm, 0.0, None)
    six_hour = pd.Series(increments_mm, index=pd.to_datetime(ordered.time.values), name="precip_mm_6h")
    return six_hour.resample("1D").sum()


def daily_temp850_c(series_k: xr.DataArray) -> pd.Series:
    ordered = series_k.sortby("time")
    return pd.Series(
        data=np.asarray(ordered.values, dtype=float) - 273.15,
        index=pd.to_datetime(ordered.time.values),
        name="temperature_850hPa_c",
    ).resample("1D").mean()


def snow_fraction_from_temp(temp_c: float, params: TranslatorParams) -> float:
    if temp_c <= params.snowfall_temp_c:
        return 1.0
    if temp_c >= params.rain_temp_c:
        return 0.0
    span = params.rain_temp_c - params.snowfall_temp_c
    return float((params.rain_temp_c - temp_c) / span)


def run_temp_index_translator(precip_mm: pd.Series, temp_c: pd.Series, params: TranslatorParams) -> pd.Series:
    common_index = precip_mm.index.intersection(temp_c.index)
    precip_vals = precip_mm.reindex(common_index).fillna(0.0).to_numpy(dtype=float)
    temp_vals = temp_c.reindex(common_index).to_numpy(dtype=float)
    swe = np.zeros(common_index.size, dtype=float)
    state = 0.0
    for i, (p_mm, t_c) in enumerate(zip(precip_vals, temp_vals, strict=True)):
        snow_frac = snow_fraction_from_temp(float(t_c), params)
        snowfall = p_mm * snow_frac
        melt = params.degree_day_factor_mm_per_c_day * max(float(t_c) - params.melt_temp_c, 0.0)
        state = max(state + snowfall - melt, 0.0)
        swe[i] = state
    return pd.Series(swe, index=common_index, name="swe_mm")


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_mm": float(np.mean(values)),
        "std_mm": float(np.std(values, ddof=0)),
        "min_mm": float(np.min(values)),
        "max_mm": float(np.max(values)),
        "q05_mm": quantile(values, 0.05),
        "q25_mm": quantile(values, 0.25),
        "q50_mm": quantile(values, 0.50),
        "q75_mm": quantile(values, 0.75),
        "q95_mm": quantile(values, 0.95),
    }


def iter_member_files(state: str) -> list[tuple[str, Path]]:
    root = ACTUAL_OUTPUT_DIR if state == "actual" else REVERSED_OUTPUT_DIR
    return [(member_dir.name, member_dir / PREDICTION_FILE_NAME) for member_dir in sorted(root.glob("member_*")) if (member_dir / PREDICTION_FILE_NAME).exists()]


def make_reports(
    params: TranslatorParams,
    available_models_text: str,
    forcing_report_text: str,
    member_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    comparison: dict,
) -> None:
    AVAILABLE_MODELS_MD.write_text(available_models_text, encoding="utf-8")
    FORCING_VAR_MD.write_text(forcing_report_text, encoding="utf-8")
    MEMBER_RESULTS_CSV.write_text(member_df.to_csv(index=False), encoding="utf-8")
    DAILY_SWE_CSV.write_text(daily_df.to_csv(index=False), encoding="utf-8")
    COMPARISON_JSON.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    actual_vals = member_df.loc[member_df["state"] == "actual", "apr1_proxy_swe_mm"].to_numpy(dtype=float)
    reversed_vals = member_df.loc[member_df["state"] == "reversed_pacific", "apr1_proxy_swe_mm"].to_numpy(dtype=float)
    ucla_mm = float(comparison["ucla_observed_april1_swe_mm"])
    bins = max(8, int(np.sqrt(max(len(actual_vals), len(reversed_vals)))))

    plt.figure(figsize=(8, 5))
    plt.hist(actual_vals, bins=bins, alpha=0.8, color="#1f77b4", label="actual SST")
    plt.axvline(ucla_mm, color="gray", linestyle="--", linewidth=2, label="UCLA observed April 1 SWE")
    plt.xlabel("April 1 Sierra SWE proxy (mm SWE)")
    plt.ylabel("member count")
    plt.title("WY1998 actual SST SWE ensemble")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_HIST_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(actual_vals, bins=bins, alpha=0.55, color="#1f77b4", label="actual SST")
    plt.hist(reversed_vals, bins=bins, alpha=0.55, color="#d62728", label="reversed Pacific SST")
    plt.axvline(ucla_mm, color="gray", linestyle="--", linewidth=2, label="UCLA observed actual world")
    plt.xlabel("April 1 Sierra SWE proxy (mm SWE)")
    plt.ylabel("member count")
    plt.title("WY1998 actual vs reversed-Pacific SWE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_VS_REVERSED_PNG, dpi=150)
    plt.close()

    paired = (
        member_df[["state", "member_id", "rng_key", "apr1_proxy_swe_mm"]]
        .pivot(index="rng_key", columns="state", values="apr1_proxy_swe_mm")
        .sort_index()
    )
    deltas = np.sort(paired["reversed_pacific"].to_numpy(dtype=float) - paired["actual"].to_numpy(dtype=float))
    plt.figure(figsize=(8, 5))
    plt.axvline(0.0, color="black", linewidth=1)
    plt.scatter(deltas, np.arange(1, len(deltas) + 1), color="#444444", s=25)
    plt.xlabel("Delta SWE = reversed - actual (mm SWE)")
    plt.ylabel("sorted member rank")
    plt.tight_layout()
    plt.savefig(PAIRED_DELTA_PNG, dpi=150)
    plt.close()

    summary = comparison["distribution_summary"]
    lines = [
        "## NeuralGCM WY1998 Phase 1 SWE translator feasibility",
        "",
        "This report uses existing NeuralGCM WY1998 actual/reversed outputs only. No SST ensemble rerun was performed.",
        "",
        "### Translator used",
        "",
        f"- translator: `{comparison['translator_used']}`",
        "- translator type: uncalibrated temperature-index bucket feasibility fallback",
        "- reason fallback was used: no local/offline usable SNOW-17/PRMS/pywatershed implementation was available in the current environment",
        "",
        "### Temperature source",
        "",
        f"- near-surface temperature available in saved NeuralGCM outputs: `{comparison['near_surface_temp_in_neuralgcm_outputs']}`",
        f"- only saved NeuralGCM temperature diagnostic: `{comparison['saved_neuralgcm_temperature_diagnostic']}`",
        f"- Phase 1 temperature source actually used: `{comparison['phase1_temperature_source_used']}`",
        f"- NeuralGCM rerun needed for native 2m temperature output: `{comparison['needs_neuralgcm_rerun_for_2m_temperature']}`",
        "",
        "### Sierra SWE results",
        "",
        f"- UCLA observed April 1 SWE: `{comparison['ucla_observed_april1_swe_mm']:.3f} mm SWE`",
        f"- actual mean/std: `{summary['actual']['mean_mm']:.3f} / {summary['actual']['std_mm']:.3f} mm SWE`",
        f"- actual q05/q25/q50/q75/q95: `{summary['actual']['q05_mm']:.3f} / {summary['actual']['q25_mm']:.3f} / {summary['actual']['q50_mm']:.3f} / {summary['actual']['q75_mm']:.3f} / {summary['actual']['q95_mm']:.3f}`",
        f"- UCLA inside actual range: `{comparison['ucla_inside_actual_range']}`",
        f"- UCLA inside actual 5-95% interval: `{comparison['ucla_inside_actual_q05_q95']}`",
        f"- reversed mean/std: `{summary['reversed_pacific']['mean_mm']:.3f} / {summary['reversed_pacific']['std_mm']:.3f} mm SWE`",
        f"- reversed q05/q25/q50/q75/q95: `{summary['reversed_pacific']['q05_mm']:.3f} / {summary['reversed_pacific']['q25_mm']:.3f} / {summary['reversed_pacific']['q50_mm']:.3f} / {summary['reversed_pacific']['q75_mm']:.3f} / {summary['reversed_pacific']['q95_mm']:.3f}`",
        f"- delta_mean = reversed - actual: `{comparison['delta_mean_mm']:.3f} mm SWE`",
        f"- standardized_delta: `{comparison['standardized_delta']:.6f}`",
        f"- fraction of paired deltas > 0: `{comparison['fraction_paired_deltas_gt_zero']:.6f}`",
        f"- reversed SST shifts SWE downward: `{comparison['delta_mean_mm'] < 0.0}`",
        "",
        "### Caveats",
        "",
        "- This is not a packaged SNOW-17/PRMS run. It is a feasibility fallback because no offline SNOW-17/PRMS implementation was available.",
        "- Temperature forcing comes from local ERA5-Land 2m temperature, so the actual/reversed SWE differences in this Phase 1 pass are driven by NeuralGCM precipitation differences, not by a member-specific near-surface temperature response.",
        "- The SWE value reported here is an April 1 proxy based on the last available NeuralGCM season day through 1998-03-31.",
        "- This first pass uses the full Sierra box only. North/Central/South subregions were not yet translated on the NeuralGCM grid.",
        "",
        "### Next step",
        "",
        "- Best next step: rerun the same NeuralGCM SST-state ensembles with 2m temperature saved, then swap the fallback translator for a real SNOW-17/PRMS-style implementation once one is staged locally or installed offline.",
    ]
    FINAL_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    params = TranslatorParams()

    era5land_temp_c = load_era5_land_daily_sierra_t2m_c()
    ucla_april1_swe_mm = load_ucla_april1_sierra_swe_mm()

    forcing_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []

    for state in ("actual", "reversed_pacific"):
        for member_id, path in iter_member_files(state):
            rng_key = int(member_id.split("_")[-1])
            with xr.open_dataset(path) as ds:
                precip = ds["precipitation_cumulative_mean"]
                if "surface" in precip.dims:
                    precip = precip.isel(surface=0)
                precip_region = subset_region(precip, SIERRA_BOX, "latitude", "longitude")
                precip_series = weighted_mean(precip_region, "latitude")
                daily_precip_mm = daily_precip_from_cumulative_mm(precip_series)

                temp850_daily_c = None
                if "temperature_850hPa" in ds:
                    temp850 = ds["temperature_850hPa"]
                    if "surface" in temp850.dims:
                        temp850 = temp850.isel(surface=0)
                    temp850_region = subset_region(temp850, SIERRA_BOX, "latitude", "longitude")
                    temp850_series = weighted_mean(temp850_region, "latitude")
                    temp850_daily_c = daily_temp850_c(temp850_series)

            swe_daily_mm = run_temp_index_translator(daily_precip_mm, era5land_temp_c, params)
            for day in swe_daily_mm.index:
                forcing_rows.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "state": state,
                        "member_id": member_id,
                        "rng_key": rng_key,
                        "sierra_precip_mm_day": float(daily_precip_mm.get(day, np.nan)),
                        "era5land_t2m_c_day": float(era5land_temp_c.get(day, np.nan)),
                        "temperature_850hPa_c_day": float(temp850_daily_c.get(day, np.nan)) if temp850_daily_c is not None else np.nan,
                    }
                )
                daily_rows.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "state": state,
                        "member_id": member_id,
                        "rng_key": rng_key,
                        "precip_mm_day": float(daily_precip_mm.get(day, np.nan)),
                        "temperature_c_day": float(era5land_temp_c.get(day, np.nan)),
                        "swe_mm": float(swe_daily_mm.loc[day]),
                    }
                )

            member_rows.append(
                {
                    "water_year": WATER_YEAR,
                    "state": state,
                    "member_id": member_id,
                    "rng_key": rng_key,
                    "region": "SIERRA_BOX",
                    "translator": "fallback_uncalibrated_temperature_index",
                    "precip_source": "NeuralGCM precipitation_cumulative_mean increments",
                    "temperature_source": "ERA5-Land 2m temperature",
                    "apr1_proxy_swe_mm": float(swe_daily_mm.iloc[-1]),
                    "peak_swe_mm": float(swe_daily_mm.max()),
                    "final_available_date": swe_daily_mm.index[-1].strftime("%Y-%m-%d"),
                    "daily_steps": int(swe_daily_mm.size),
                }
            )

    forcing_df = pd.DataFrame(forcing_rows).sort_values(["state", "member_id", "date"]).reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values(["state", "member_id", "date"]).reset_index(drop=True)
    member_df = pd.DataFrame(member_rows).sort_values(["state", "member_id"]).reset_index(drop=True)

    WEATHER_FORCING_CSV.write_text(forcing_df.to_csv(index=False), encoding="utf-8")

    actual_vals = member_df.loc[member_df["state"] == "actual", "apr1_proxy_swe_mm"].to_numpy(dtype=float)
    reversed_vals = member_df.loc[member_df["state"] == "reversed_pacific", "apr1_proxy_swe_mm"].to_numpy(dtype=float)
    paired = (
        member_df[["state", "rng_key", "apr1_proxy_swe_mm"]]
        .pivot(index="rng_key", columns="state", values="apr1_proxy_swe_mm")
        .sort_index()
    )
    deltas = paired["reversed_pacific"].to_numpy(dtype=float) - paired["actual"].to_numpy(dtype=float)
    pooled = pooled_std(actual_vals, reversed_vals)

    comparison = {
        "water_year": WATER_YEAR,
        "translator_used": "fallback_uncalibrated_temperature_index",
        "translator_parameters": {
            "snowfall_temp_c": params.snowfall_temp_c,
            "rain_temp_c": params.rain_temp_c,
            "melt_temp_c": params.melt_temp_c,
            "degree_day_factor_mm_per_c_day": params.degree_day_factor_mm_per_c_day,
        },
        "available_model_status": {
            "pywatershed_importable": False,
            "prms_importable": False,
            "snow17_importable": False,
            "offline_install_attempt": "pywatershed install attempt failed because PyPI DNS/network access is unavailable in the current environment",
        },
        "near_surface_temp_in_neuralgcm_outputs": False,
        "saved_neuralgcm_temperature_diagnostic": "temperature_850hPa",
        "phase1_temperature_source_used": "ERA5-Land 2m temperature daily Sierra mean",
        "needs_neuralgcm_rerun_for_2m_temperature": True,
        "swe_units": "mm SWE",
        "ucla_observed_april1_swe_mm": ucla_april1_swe_mm,
        "ucla_inside_actual_range": bool(actual_vals.min() <= ucla_april1_swe_mm <= actual_vals.max()),
        "ucla_inside_actual_q05_q95": bool(quantile(actual_vals, 0.05) <= ucla_april1_swe_mm <= quantile(actual_vals, 0.95)),
        "delta_mean_mm": float(np.mean(reversed_vals) - np.mean(actual_vals)),
        "standardized_delta": float((np.mean(reversed_vals) - np.mean(actual_vals)) / pooled) if np.isfinite(pooled) and pooled != 0.0 else None,
        "fraction_paired_deltas_gt_zero": float(np.mean(deltas > 0.0)),
        "distribution_summary": {
            "actual": summarize_distribution(actual_vals),
            "reversed_pacific": summarize_distribution(reversed_vals),
        },
        "paired_detail": {
            "rng_keys": paired.index.astype(int).tolist(),
            "actual_swe_mm": paired["actual"].astype(float).tolist(),
            "reversed_swe_mm": paired["reversed_pacific"].astype(float).tolist(),
            "delta_swe_mm": deltas.astype(float).tolist(),
        },
        "caveats": [
            "No local/offline packaged SNOW-17/PRMS/pywatershed implementation was available.",
            "This Phase 1 run uses ERA5-Land Sierra 2m temperature rather than NeuralGCM member-specific near-surface temperature.",
            "Only the full Sierra box was translated in this first pass.",
            "April 1 SWE is approximated using the last available model day through 1998-03-31.",
        ],
    }

    available_models_text = "\n".join(
        [
            "## SWE translator available models survey",
            "",
            "- `pywatershed`: not importable in the current NeuralGCM venv.",
            "- `prms`: not importable.",
            "- `snow17`: not importable.",
            "- other obvious local package names checked: `hydromt`, `pysnobal`; not importable.",
            "- temporary offline install attempt: `pywatershed` install failed because the environment cannot resolve `pypi.org`.",
            "",
            "### Existing project translator code",
            "",
            "- Existing repo code covers UCLA SWE target loading, Sierra-area averaging, and North/Central/South basin masks.",
            "- No pre-existing SWE translator model implementation was found in the repo.",
            "",
            "### Feasibility fallback used in this turn",
            "",
            "- implementation used: `fallback_uncalibrated_temperature_index`",
            "- required inputs: daily precipitation totals in mm, daily air temperature in C",
            "- parameters: snowfall/rain partition thresholds and degree-day melt factor",
            "- output SWE units: `mm SWE`",
            "- supported forcing geometry in this fallback: basin-aggregated / Sierra-area mean",
            "- precipitation + temperature only: `Yes`",
            "",
            "### Bottom line",
            "",
            "- No packaged SNOW-17/PRMS-style implementation is currently usable offline in this environment.",
            "- This turn therefore uses a clearly labeled temperature-index feasibility fallback rather than claiming a real SNOW-17/PRMS run.",
            "",
        ]
    )

    forcing_report_text = "\n".join(
        [
            "## NeuralGCM SWE forcing variable report",
            "",
            f"- saved prediction file inspected: `{ACTUAL_OUTPUT_DIR / 'member_000' / PREDICTION_FILE_NAME}`",
            f"- prepared forcing file inspected: `{PREP_ROOT / 'wy1998_actual_m001' / 'forcing_regridded_6h.nc'}`",
            f"- prepared initial condition file inspected: `{PREP_ROOT / 'wy1998_actual_m001' / 'initial_condition_regridded.nc'}`",
            "",
            "### Current NeuralGCM outputs",
            "",
            "- precipitation available: `precipitation_cumulative_mean`",
            "- precipitation interpretation: cumulative precipitation depth in meters",
            "- near-surface temperature available: `No`",
            "- saved temperature diagnostic: `temperature_850hPa` only",
            "- other saved diagnostics: `evaporation`, `u_component_of_wind_850hPa`, `v_component_of_wind_850hPa`, `specific_humidity_850hPa`",
            "",
            "### Prepared inputs",
            "",
            "- prepared forcing variables: `sea_surface_temperature`, `sea_ice_cover`",
            "- prepared forcing does not contain 2m air temperature",
            "- prepared initial condition contains 3D pressure-level `temperature`, not a saved 2m temperature field",
            "",
            "### Safest Phase 1 temperature source",
            "",
            "- safest no-rerun temperature source for Phase 1: local ERA5-Land `t2m` over the Sierra box",
            "- rationale: it is true near-surface temperature, already local, and directly usable by a temperature-index translator",
            "- downside: actual and reversed SST states then share the same temperature forcing in Phase 1",
            "",
            "### Rerun need",
            "",
            "- If the translator must use NeuralGCM-native near-surface temperature per member/state, then NeuralGCM needs to be rerun with a 2m temperature diagnostic saved.",
            "",
        ]
    )

    make_reports(params, available_models_text, forcing_report_text, member_df, daily_df, comparison)


if __name__ == "__main__":
    main()
