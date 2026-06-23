#!/usr/bin/env python3
"""
Run COBE2-only leave-one-water-year-out LOD for April 1 Sierra SWE.
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
SWE_TARGET_FILE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc")
INSAMPLE_SUMMARY_FILE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json")
OUTPUT_ROOT = Path(
    os.environ.get(
        "COBE2_SIERRA_SWE_LOYO_LOD_ANALYSIS_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/loyo_lod_analysis",
    )
).expanduser()
NETCDF_ENGINE = "netcdf4"

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = np.arange(WATER_YEAR_START, WATER_YEAR_END + 1, dtype=np.int32)
PACIFIC_LAT_MIN = -10.0
PACIFIC_LAT_MAX = 60.0
PACIFIC_LON_MIN = 120.0
PACIFIC_LON_MAX = 280.0
LAG_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]
DEFAULT_K_MAX = 6
DELTA_R2_MIN = 0.02
CORR_MIN = 0.2


def read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


K_MAX = read_positive_int_env("SIERRA_SWE_LOD_K_MAX", DEFAULT_K_MAX)


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def peak_memory_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def month_targets() -> list[np.datetime64]:
    times: list[np.datetime64] = []
    for water_year in WATER_YEARS:
        for _, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
    return times


def load_target() -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(SWE_TARGET_FILE, engine=NETCDF_ENGINE) as ds:
        ds = ds.sel(water_year=WATER_YEARS).load()
        target_anom_m = np.asarray(ds["sierra_swe_apr1_anom_m"].values, dtype=np.float64)
        target_std = np.asarray(ds["sierra_swe_apr1_standardized"].values, dtype=np.float64)
    return target_anom_m, target_std


def load_aligned_sst() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_times = month_targets()
    with xr.open_dataset(COBE2_SST_FILE, engine=NETCDF_ENGINE) as ds:
        sst = ds["sst"].sel(
            time=selected_times,
            lat=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            lon=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        ).load()
        values = np.asarray(sst.values, dtype=np.float32).reshape(len(WATER_YEARS), len(LAG_SPECS), sst.sizes["lat"], sst.sizes["lon"])
        latitude = np.asarray(sst["lat"].values, dtype=np.float32)
        longitude = np.asarray(sst["lon"].values, dtype=np.float32)
    return values, latitude, longitude


def build_fold_predictors(
    train_sst: np.ndarray,
    test_sst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    monthly_clim = np.nanmean(train_sst, axis=0, dtype=np.float64)
    train_anom = train_sst - monthly_clim[None, :, :, :]
    test_anom = test_sst - monthly_clim
    mean = np.nanmean(train_anom, axis=0, dtype=np.float64)
    std = np.nanstd(train_anom, axis=0, ddof=1, dtype=np.float64)
    valid = np.all(np.isfinite(train_anom), axis=0) & np.isfinite(std) & (std > 0.0)

    train_standardized = np.full(train_anom.shape, np.nan, dtype=np.float64)
    np.divide(
        train_anom - mean[None, :, :, :],
        std[None, :, :, :],
        out=train_standardized,
        where=valid[None, :, :, :],
    )
    test_standardized = np.full(test_anom.shape, np.nan, dtype=np.float64)
    np.divide(
        test_anom - mean,
        std,
        out=test_standardized,
        where=valid,
    )

    lag_idx, lat_idx, lon_idx = np.where(valid)
    X_train = train_standardized[:, lag_idx, lat_idx, lon_idx].astype(np.float32)
    x_test = test_standardized[lag_idx, lat_idx, lon_idx].astype(np.float64)
    spatial_idx = np.stack([lat_idx, lon_idx], axis=1).astype(np.int32)
    valid_counts_by_lag = valid.sum(axis=(1, 2)).astype(np.int32)
    return X_train, x_test, lag_idx.astype(np.int32), spatial_idx, valid_counts_by_lag


def centered_correlation_scores(X: np.ndarray, residual: np.ndarray) -> np.ndarray:
    residual_centered = residual - residual.mean()
    residual_std = residual_centered.std(ddof=1)
    if not np.isfinite(residual_std) or residual_std == 0.0:
        return np.zeros(X.shape[1], dtype=np.float64)
    return (X.astype(np.float64).T @ residual_centered.astype(np.float64)) / ((X.shape[0] - 1) * residual_std)


def standardize_train_target(y_train_raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(y_train_raw))
    std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Training target standard deviation is not positive.")
    return (y_train_raw - mean) / std, mean, std


def broad_region(lat: float, lon: float) -> str:
    if lat < 10.0:
        lat_band = "tropical"
    elif lat < 30.0:
        lat_band = "subtropical"
    else:
        lat_band = "midlatitude"

    if lon < 170.0:
        lon_band = "western"
    elif lon < 230.0:
        lon_band = "central"
    else:
        lon_band = "eastern"
    return f"{lat_band}_{lon_band}_pacific"


def run_lod_fold(
    X_train: np.ndarray,
    x_test: np.ndarray,
    y_train_std: np.ndarray,
    train_water_years: np.ndarray,
    held_out_wy: int,
    latitude: np.ndarray,
    longitude: np.ndarray,
    lag_idx: np.ndarray,
    spatial_idx: np.ndarray,
    valid_counts_by_lag: np.ndarray,
) -> tuple[float, list[dict[str, object]], str]:
    residual = y_train_std.astype(np.float64).copy()
    total_sum = float(np.sum(y_train_std.astype(np.float64) ** 2))
    retained_modes_train: list[np.ndarray] = []
    retained_modes_test: list[float] = []
    fold_rows: list[dict[str, object]] = []
    prediction_std = 0.0
    previous_r2 = 0.0
    stop_reason = "reached_k_max"
    lag_names = np.asarray([spec[0] for spec in LAG_SPECS], dtype="U3")

    for mode_number in range(1, K_MAX + 1):
        corr_scores = centered_correlation_scores(X_train, residual)
        q_index = int(np.argmax(np.abs(corr_scores)))
        corr_value = float(corr_scores[q_index])
        base_row = {
            "held_out_water_year": int(held_out_wy),
            "train_year_count": int(train_water_years.size),
            "valid_candidate_count": int(X_train.shape[1]),
            "valid_counts_by_lag": ",".join(str(int(v)) for v in valid_counts_by_lag.tolist()),
            "mode_number": mode_number,
            "selected": False,
            "candidate_index": q_index,
            "corr_with_residual": corr_value,
        }
        if abs(corr_value) < CORR_MIN:
            base_row["stop_reason"] = f"|corr| < {CORR_MIN}"
            stop_reason = str(base_row["stop_reason"])
            fold_rows.append(base_row)
            break

        u_train = X_train[:, q_index].astype(np.float64)
        u_test = float(x_test[q_index])
        m_hat_train = u_train.copy()
        m_hat_test = u_test
        orthogonalization_coeffs: list[float] = []
        for previous_train, previous_test in zip(retained_modes_train, retained_modes_test):
            coeff = float(np.sum(u_train * previous_train) / np.sum(previous_train * previous_train))
            m_hat_train = m_hat_train - coeff * previous_train
            m_hat_test = m_hat_test - coeff * previous_test
            orthogonalization_coeffs.append(coeff)
        if mode_number == 1:
            m_hat_train = u_train.copy()
            m_hat_test = u_test

        m_hat_mean = float(m_hat_train.mean())
        m_hat_std = float(m_hat_train.std(ddof=1))
        if not np.isfinite(m_hat_std) or m_hat_std == 0.0:
            base_row["stop_reason"] = "orthogonalized mode std == 0"
            stop_reason = str(base_row["stop_reason"])
            fold_rows.append(base_row)
            break

        mode_train = (m_hat_train - m_hat_mean) / m_hat_std
        mode_test = float((m_hat_test - m_hat_mean) / m_hat_std)
        beta = float(np.sum(mode_train * residual) / np.sum(mode_train * mode_train))
        new_residual = residual - beta * mode_train
        cumulative_r2 = 1.0 - (float(np.sum(new_residual ** 2)) / total_sum)
        delta_r2 = float(cumulative_r2 - previous_r2)

        row = dict(base_row)
        row.update(
            {
                "beta": beta,
                "delta_r2": delta_r2,
                "cumulative_r2": cumulative_r2,
                "mode_mean_before_standardization": m_hat_mean,
                "mode_std_before_standardization": m_hat_std,
                "mode_test_value_standardized": mode_test,
                "orthogonalization_coefficients": json.dumps(orthogonalization_coeffs),
                "selected": False,
            }
        )

        if delta_r2 < DELTA_R2_MIN:
            row["stop_reason"] = f"delta_r2 < {DELTA_R2_MIN}"
            stop_reason = str(row["stop_reason"])
            fold_rows.append(row)
            break

        row["selected"] = True
        row["stop_reason"] = ""
        lag_name = str(lag_names[int(lag_idx[q_index])])
        lat_value = float(latitude[int(spatial_idx[q_index, 0])])
        lon_value = float(longitude[int(spatial_idx[q_index, 1])])
        row["lag_month"] = lag_name
        row["latitude"] = lat_value
        row["longitude_0_360"] = lon_value
        row["ocean_candidate_q"] = q_index
        row["broad_region"] = broad_region(lat_value, lon_value)
        fold_rows.append(row)

        retained_modes_train.append(mode_train.copy())
        retained_modes_test.append(mode_test)
        prediction_std += beta * mode_test
        residual = new_residual
        previous_r2 = cumulative_r2

    return prediction_std, fold_rows, stop_reason


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true, ddof=1) == 0.0 or np.std(y_pred, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def plot_scatter(path: Path, observed: np.ndarray, predicted: np.ndarray, r2_loyo: float, corr: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(observed, predicted, s=42, color="tab:blue", edgecolors="black", linewidths=0.4, alpha=0.85)
    all_values = np.concatenate([observed, predicted])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("Observed April 1 Sierra SWE anomaly (m)")
    ax.set_ylabel("LOYO predicted April 1 Sierra SWE anomaly (m)")
    ax.set_title(f"COBE2-only LOYO LOD: observed vs predicted\nR2={r2_loyo:.3f}  corr={corr:.3f}")
    ax.grid(True, linewidth=0.25, color="0.8")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_timeseries(path: Path, water_years: np.ndarray, observed: np.ndarray, predicted: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(water_years, observed, color="black", linewidth=1.5, label="Observed")
    ax.plot(water_years, predicted, color="tab:red", linewidth=1.2, label="LOYO predicted")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("COBE2-only LOYO LOD by water year")
    ax.grid(True, linewidth=0.25, color="0.85")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_selection_frequency_map(path: Path, fold_rows: list[dict[str, object]]) -> None:
    selected_rows = [row for row in fold_rows if row.get("selected")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for mode_number in range(1, 7):
        ax = axes[(mode_number - 1) // 3, (mode_number - 1) % 3]
        mode_rows = [row for row in selected_rows if int(row["mode_number"]) == mode_number]
        counts: dict[tuple[float, float], int] = {}
        for row in mode_rows:
            key = (float(row["latitude"]), float(row["longitude_0_360"]))
            counts[key] = counts.get(key, 0) + 1
        ax.set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
        ax.set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
        ax.set_xticks(np.arange(120, 281, 20))
        ax.set_yticks(np.arange(-10, 61, 10))
        ax.grid(True, linewidth=0.2, color="0.8")
        ax.set_title(f"Mode {mode_number} selection frequency")
        ax.set_xlabel("Longitude (0 to 360)")
        ax.set_ylabel("Latitude")
        if counts:
            lats = np.array([key[0] for key in counts], dtype=np.float64)
            lons = np.array([key[1] for key in counts], dtype=np.float64)
            freqs = np.array([counts[key] for key in counts], dtype=np.float64)
            scatter = ax.scatter(lons, lats, c=freqs, s=30.0 + 28.0 * freqs, cmap="viridis", edgecolors="black", linewidths=0.4)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="fold count")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_predictions_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "water_year",
        "observed_swe_anom_m",
        "predicted_swe_anom_m",
        "observed_swe_standardized_global",
        "predicted_swe_standardized_trainfold",
        "n_selected_modes",
        "fold_stop_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_fold_modes_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "held_out_water_year",
        "train_year_count",
        "valid_candidate_count",
        "valid_counts_by_lag",
        "mode_number",
        "selected",
        "stop_reason",
        "candidate_index",
        "ocean_candidate_q",
        "lag_month",
        "latitude",
        "longitude_0_360",
        "broad_region",
        "corr_with_residual",
        "beta",
        "delta_r2",
        "cumulative_r2",
        "mode_mean_before_standardization",
        "mode_std_before_standardization",
        "mode_test_value_standardized",
        "orthogonalization_coefficients",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def build_mode_stability_rows(fold_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_rows = [row for row in fold_rows if row.get("selected")]
    output_rows: list[dict[str, object]] = []
    for mode_number in range(1, K_MAX + 1):
        mode_rows = [row for row in selected_rows if int(row["mode_number"]) == mode_number]
        total = len(mode_rows)
        lag_counts: dict[str, int] = {}
        region_counts: dict[str, int] = {}
        for row in mode_rows:
            lag = str(row["lag_month"])
            region = str(row["broad_region"])
            lag_counts[lag] = lag_counts.get(lag, 0) + 1
            region_counts[region] = region_counts.get(region, 0) + 1
        for lag, count in sorted(lag_counts.items()):
            output_rows.append(
                {
                    "summary_type": "lag_month_frequency",
                    "mode_number": mode_number,
                    "category": lag,
                    "count": count,
                    "fraction": count / total if total else float("nan"),
                }
            )
        for region, count in sorted(region_counts.items()):
            output_rows.append(
                {
                    "summary_type": "broad_region_frequency",
                    "mode_number": mode_number,
                    "category": region,
                    "count": count,
                    "fraction": count / total if total else float("nan"),
                }
            )
    return output_rows


def write_mode_stability_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["summary_type", "mode_number", "category", "count", "fraction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def repeated_region_summary(fold_rows: list[dict[str, object]], mode_number: int) -> dict[str, object]:
    selected_rows = [row for row in fold_rows if row.get("selected") and int(row["mode_number"]) == mode_number]
    total = len(selected_rows)
    counts: dict[str, int] = {}
    for row in selected_rows:
        region = str(row["broad_region"])
        counts[region] = counts.get(region, 0) + 1
    if not counts:
        return {"mode_number": mode_number, "selected_fold_count": 0, "top_region": "", "top_region_count": 0, "top_region_fraction": float("nan")}
    top_region, top_count = max(counts.items(), key=lambda item: item[1])
    return {
        "mode_number": mode_number,
        "selected_fold_count": total,
        "top_region": top_region,
        "top_region_count": top_count,
        "top_region_fraction": top_count / total,
    }


def write_summary_markdown(
    path: Path,
    skill_summary: dict[str, object],
    observed: np.ndarray,
    predicted: np.ndarray,
    repeated_regions: list[dict[str, object]],
    selection_map_path: Path,
) -> None:
    lines = [
        "# COBE2-only LOYO LOD Summary",
        "",
        "- This run is **COBE2-only**.",
        "- Mode selection and coefficient fitting are done **inside each training fold only**.",
        "- The held-out year is **not used** for mode selection, target standardization, predictor standardization, or coefficient fitting.",
        "- The reported `R2_LOYO` is the **single final LOYO test R2** computed across all 37 held-out predictions.",
        "- No fold-level test R2 values were computed or averaged.",
        "",
        "## Skill",
        "",
        f"- Water years: `{WATER_YEAR_START}--{WATER_YEAR_END}` ({len(observed)} folds)",
        f"- LOYO run mode cap (`K_max`): `{skill_summary['k_max']}`",
        f"- In-sample 6-mode cumulative R2 from the existing COBE2 diagnostic: `{skill_summary['in_sample_reference']['full_run_cumulative_r2_mode6']:.4f}`",
        f"- Final LOYO test R2: `{skill_summary['metrics']['r2_loyo']:.4f}`",
        f"- LOYO RMSE: `{skill_summary['metrics']['rmse_m']:.6f}` m",
        f"- LOYO MAE: `{skill_summary['metrics']['mae_m']:.6f}` m",
        f"- LOYO correlation: `{skill_summary['metrics']['correlation']:.4f}`",
        f"- Mean selected mode count per fold: `{skill_summary['mode_selection']['mean_selected_modes_per_fold']:.3f}`",
        f"- Maximum selected mode count in any fold: `{skill_summary['mode_selection']['max_selected_modes_per_fold']}`",
        "",
        "## Does High In-sample R2 Survive?",
        "",
        f"- Verdict: **{skill_summary['interpretation']['generalization_verdict']}**",
        f"- The LOYO test R2 is `{skill_summary['metrics']['r2_loyo']:.4f}`, compared with in-sample `{skill_summary['in_sample_reference']['full_run_cumulative_r2_mode6']:.4f}` from the full COBE2 6-mode fit.",
        "",
        "## Mode Stability",
        "",
        "- Raw fold-by-fold mode selections are saved in `cobe2_sierra_swe_lod_loyo_fold_modes.csv`.",
        "- Stability summaries across folds are saved in `cobe2_sierra_swe_lod_loyo_mode_stability.csv`.",
        f"- Selection-frequency map figure: `{selection_map_path.name}`",
        "",
        "### Repeated broad-region selection for modes 1-3",
        "",
    ]
    for item in repeated_regions:
        frac = item["top_region_fraction"]
        frac_text = "nan" if not np.isfinite(frac) else f"{frac:.3f}"
        lines.append(
            f"- Mode {item['mode_number']}: top region `{item['top_region']}` selected in "
            f"{item['top_region_count']}/{item['selected_fold_count']}` selected folds "
            f"(fraction `{frac_text}`)."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `cobe2_sierra_swe_lod_loyo_predictions.csv`",
            "- `cobe2_sierra_swe_lod_loyo_fold_modes.csv`",
            "- `cobe2_sierra_swe_lod_loyo_skill_summary.json`",
            "- `cobe2_sierra_swe_lod_loyo_mode_stability.csv`",
            "- `cobe2_sierra_swe_lod_loyo_observed_vs_predicted.png`",
            "- `cobe2_sierra_swe_lod_loyo_timeseries.png`",
            "- `cobe2_sierra_swe_lod_loyo_summary.md`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    print("starting COBE2-only LOYO LOD", flush=True)
    print(f"output_root={OUTPUT_ROOT}", flush=True)
    print(f"k_max={K_MAX}", flush=True)
    print("loading April 1 Sierra SWE target...", flush=True)
    target_anom_m, target_std_global = load_target()
    print(f"loaded target arrays shape={target_anom_m.shape}", flush=True)
    print("loading aligned COBE2 SST predictors...", flush=True)
    sst_values, latitude, longitude = load_aligned_sst()
    print(f"loaded aligned COBE2 SST shape={sst_values.shape}", flush=True)
    insample_summary = json.loads(INSAMPLE_SUMMARY_FILE.read_text())

    prediction_rows: list[dict[str, object]] = []
    fold_mode_rows: list[dict[str, object]] = []
    predicted_values = np.full(WATER_YEARS.shape, np.nan, dtype=np.float64)

    print(f"Starting COBE2-only LOYO LOD across {WATER_YEARS.size} folds", flush=True)
    for fold_index, held_out_wy in enumerate(WATER_YEARS):
        test_mask = WATER_YEARS == held_out_wy
        train_mask = ~test_mask
        train_sst = sst_values[train_mask]
        test_sst = sst_values[test_mask][0]
        y_train_raw = target_anom_m[train_mask]
        y_test_raw = float(target_anom_m[test_mask][0])
        train_water_years = WATER_YEARS[train_mask]

        X_train, x_test, lag_idx, spatial_idx, valid_counts_by_lag = build_fold_predictors(train_sst, test_sst)
        y_train_std, y_train_mean, y_train_stddev = standardize_train_target(y_train_raw)
        predicted_std, fold_rows, stop_reason = run_lod_fold(
            X_train=X_train,
            x_test=x_test,
            y_train_std=y_train_std,
            train_water_years=train_water_years,
            held_out_wy=int(held_out_wy),
            latitude=latitude,
            longitude=longitude,
            lag_idx=lag_idx,
            spatial_idx=spatial_idx,
            valid_counts_by_lag=valid_counts_by_lag,
        )
        predicted_raw = float(y_train_mean + y_train_stddev * predicted_std)
        predicted_values[fold_index] = predicted_raw
        selected_count = sum(1 for row in fold_rows if row.get("selected"))
        prediction_rows.append(
            {
                "water_year": int(held_out_wy),
                "observed_swe_anom_m": y_test_raw,
                "predicted_swe_anom_m": predicted_raw,
                "observed_swe_standardized_global": float(target_std_global[test_mask][0]),
                "predicted_swe_standardized_trainfold": predicted_std,
                "n_selected_modes": selected_count,
                "fold_stop_reason": stop_reason,
            }
        )
        fold_mode_rows.extend(fold_rows)
        print(
            f"fold {fold_index + 1:02d}/{WATER_YEARS.size}: held_out_WY={int(held_out_wy)} "
            f"selected_modes={selected_count} y_true={y_test_raw:.6f} y_pred={predicted_raw:.6f} stop={stop_reason}",
            flush=True,
        )

    observed = target_anom_m.astype(np.float64)
    predicted = predicted_values.astype(np.float64)
    observed_mean = float(np.mean(observed))
    sse = float(np.sum((observed - predicted) ** 2))
    sst = float(np.sum((observed - observed_mean) ** 2))
    r2_loyo = 1.0 - sse / sst
    rmse_value = rmse(observed, predicted)
    mae_value = mae(observed, predicted)
    corr_value = corrcoef_safe(observed, predicted)

    predictions_csv = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_predictions.csv"
    fold_modes_csv = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_fold_modes.csv"
    skill_summary_json = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_skill_summary.json"
    mode_stability_csv = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_mode_stability.csv"
    scatter_png = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_observed_vs_predicted.png"
    timeseries_png = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_timeseries.png"
    summary_md = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_summary.md"
    selection_map_png = OUTPUT_ROOT / "cobe2_sierra_swe_lod_loyo_selection_frequency_map.png"

    write_predictions_csv(predictions_csv, prediction_rows)
    write_fold_modes_csv(fold_modes_csv, fold_mode_rows)
    plot_scatter(scatter_png, observed, predicted, r2_loyo, corr_value)
    plot_timeseries(timeseries_png, WATER_YEARS, observed, predicted)
    plot_selection_frequency_map(selection_map_png, fold_mode_rows)

    stability_rows = build_mode_stability_rows(fold_mode_rows)
    write_mode_stability_csv(mode_stability_csv, stability_rows)
    repeated_regions = [repeated_region_summary(fold_mode_rows, mode_number) for mode_number in (1, 2, 3)]

    insample_r2 = float(insample_summary["lod_rows"][-1]["cumulative_r2"])
    if r2_loyo >= 0.75 * insample_r2:
        verdict = "high in-sample R2 survives reasonably well in train/test evaluation"
    elif r2_loyo > 0.0:
        verdict = "high in-sample R2 weakens substantially but remains positive in train/test evaluation"
    else:
        verdict = "high in-sample R2 does not survive train/test evaluation"

    selected_counts = [int(row["n_selected_modes"]) for row in prediction_rows]
    skill_summary = {
        "dataset": "COBE2 only",
        "target_definition": "April 1 Sierra SWE anomaly (m), water years 1985-2021",
        "predictor_definition": "COBE2 monthly SST anomalies Sep-Mar over Pacific box lat -10..60 lon 120..280 in 0..360",
        "cross_validation": "leave-one-water-year-out",
        "fold_count": int(WATER_YEARS.size),
        "k_max": K_MAX,
        "stopping_criteria": {
            "delta_r2_min": DELTA_R2_MIN,
            "corr_min": CORR_MIN,
        },
        "metrics": {
            "r2_loyo": r2_loyo,
            "rmse_m": rmse_value,
            "mae_m": mae_value,
            "correlation": corr_value,
            "sse": sse,
            "sst": sst,
        },
        "mode_selection": {
            "mean_selected_modes_per_fold": float(np.mean(selected_counts)),
            "max_selected_modes_per_fold": int(np.max(selected_counts)),
            "min_selected_modes_per_fold": int(np.min(selected_counts)),
        },
        "in_sample_reference": {
            "full_run_cumulative_r2_mode6": insample_r2,
            "summary_path": str(INSAMPLE_SUMMARY_FILE),
        },
        "interpretation": {
            "generalization_verdict": verdict,
            "mode1_repeated_region": repeated_regions[0],
            "mode2_repeated_region": repeated_regions[1],
            "mode3_repeated_region": repeated_regions[2],
        },
        "outputs": {
            "predictions_csv": str(predictions_csv),
            "fold_modes_csv": str(fold_modes_csv),
            "skill_summary_json": str(skill_summary_json),
            "mode_stability_csv": str(mode_stability_csv),
            "scatter_png": str(scatter_png),
            "timeseries_png": str(timeseries_png),
            "selection_frequency_map_png": str(selection_map_png),
            "summary_md": str(summary_md),
        },
        "runtime_seconds": perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
    }
    skill_summary_json.write_text(json.dumps(skill_summary, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(summary_md, skill_summary, observed, predicted, repeated_regions, selection_map_png)
    print(json.dumps(skill_summary["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
