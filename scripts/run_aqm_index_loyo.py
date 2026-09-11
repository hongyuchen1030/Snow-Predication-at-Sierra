#!/usr/bin/env python3
"""
Strict LOYO ridge experiment using the independently reconstructed Stone et al.
Atlantic Quadpole Mode seasonal index only.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_z1z2_plus_amv_k5_loyo as ref


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "aqm_index_loyo"

README_PATH = OUTPUT_DIR / "README.md"
SOURCE_METADATA_JSON = OUTPUT_DIR / "aqm_source_and_method.json"
INDEX_FULL_RECORD_CSV = OUTPUT_DIR / "aqm_index_full_record.csv"
PREDICTOR_TABLE_CSV = OUTPUT_DIR / "aqm_swe_predictor_table.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "aqm_seasonal_loyo_predictions.csv"
HYPERPARAMETERS_CSV = OUTPUT_DIR / "aqm_seasonal_loyo_selected_hyperparameters.csv"
METRICS_CSV = OUTPUT_DIR / "aqm_seasonal_loyo_metrics.csv"
PERIOD_METRICS_CSV = OUTPUT_DIR / "aqm_seasonal_loyo_period_metrics.csv"
TIMESERIES_PNG = OUTPUT_DIR / "aqm_seasonal_loyo_timeseries.png"
SCATTER_PNG = OUTPUT_DIR / "aqm_seasonal_loyo_observed_vs_predicted.png"
AUDIT_JSON = OUTPUT_DIR / "aqm_loyo_audit.json"

BASE_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "full37_selected_patch_predictor_loyo"
    / "full37_patch_loyo_predictions.csv"
)
AQM_REFERENCE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "atlantic_pc245_attribution"
    / "reference_indices_monthly_and_seasonal.csv"
)
AQM_PROVENANCE_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "atlantic_pc245_attribution"
    / "reference_pattern_provenance.csv"
)
AQM_METHOD_MD = (
    PROJECT_ROOT
    / "artifacts"
    / "atlantic_pc245_attribution"
    / "method_reconstruction.md"
)
AQM_DECISION_MD = (
    PROJECT_ROOT
    / "artifacts"
    / "atlantic_pc245_attribution"
    / "pc245_attribution_decision.md"
)
AQM_TEMPORAL_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "atlantic_pc245_attribution"
    / "pc245_temporal_similarity.csv"
)
NOAA_AMV_METRICS_CSV = PROJECT_ROOT / "artifacts" / "noaa_amv_index_loyo" / "noaa_amv_loyo_metrics.csv"
NOAA_AMV_PERIOD_METRICS_CSV = (
    PROJECT_ROOT / "artifacts" / "noaa_amv_index_loyo" / "noaa_amv_loyo_period_metrics.csv"
)
AMV_CORE_METRICS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_core_plus_forward_addition"
    / "amv_core_plus_forward_loyo_metrics.csv"
)
SINGLE_VARIABLE_RANKING_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "amv_amo_subset_selection_diagnostic"
    / "amv_amo_single_variable_loyo_ranking.csv"
)

WATER_YEAR_START = 1985
WATER_YEAR_END = 2021
WATER_YEARS = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
FEATURE_COLUMNS = ["AQM_NDJF"]
MODEL_NAME = "AQM_seasonal_index_only"
ATTRIBUTION_SCRIPT = PROJECT_ROOT / "scripts" / "run_atlantic_pc245_attribution.py"
EXACT_COMMAND = "salloc -N 1 -n 1 -c 32 -t 04:00:00 -C cpu -q interactive -A m2637"


def load_reference_target_series() -> pd.DataFrame:
    base_predictions = pd.read_csv(BASE_PREDICTIONS_CSV)
    base_predictions = base_predictions.loc[
        (base_predictions["patch_size"] == "exact_grid_cell") & (base_predictions["model_name"] == "Z1_Z2")
    ].copy()
    base_predictions = (
        base_predictions[["heldout_wy", "obs_swe"]]
        .rename(columns={"heldout_wy": "water_year", "obs_swe": "observed_swe"})
        .sort_values("water_year")
        .reset_index(drop=True)
    )
    base_predictions["water_year"] = base_predictions["water_year"].astype(int)
    base_predictions = base_predictions.loc[
        (base_predictions["water_year"] >= WATER_YEAR_START) & (base_predictions["water_year"] <= WATER_YEAR_END)
    ].copy()
    if base_predictions["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Reference target series does not match WY1985-WY2021.")
    return base_predictions


def load_aqm_index() -> tuple[pd.DataFrame, dict]:
    full_df = pd.read_csv(AQM_REFERENCE_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    need = ["date", "AQM_S2_NDJF"]
    missing = [col for col in need if col not in full_df.columns]
    if missing:
        raise ValueError("Missing AQM columns in attribution reference CSV: {}".format(missing))

    aqm_df = full_df[need].copy()
    aqm_df["aqm_defined"] = aqm_df["AQM_S2_NDJF"].notna()
    aqm_df["water_year"] = aqm_df["date"].dt.year.astype(int)
    aqm_df["month"] = aqm_df["date"].dt.month.astype(int)
    aqm_df["calendar_label"] = aqm_df["date"].dt.strftime("%Y-%m-%d")

    provenance_df = pd.read_csv(AQM_PROVENANCE_CSV)
    aqm_row = provenance_df.loc[provenance_df["reference_mode"] == "AQM"]
    if aqm_row.empty:
        raise ValueError("Could not find AQM provenance row.")
    aqm_row = aqm_row.iloc[0]

    meta = {
        "source_paper": "Stone et al. 2023, npj Climate and Atmospheric Science 6:139",
        "source_mode_name": "Atlantic Quadpole Mode",
        "reused_existing_index": True,
        "reused_index_column": "AQM_S2_NDJF",
        "reused_from_artifact": str(AQM_REFERENCE_CSV),
        "producer_script": str(ATTRIBUTION_SCRIPT),
        "supporting_files": {
            "method_reconstruction": str(AQM_METHOD_MD),
            "temporal_similarity": str(AQM_TEMPORAL_CSV),
            "attribution_decision": str(AQM_DECISION_MD),
            "reference_pattern_provenance": str(AQM_PROVENANCE_CSV),
        },
        "source_kind": aqm_row["source_kind"],
        "source_description": aqm_row["source_description"],
        "preprocessing": aqm_row["preprocessing"],
        "raw_input_path_or_url": aqm_row["raw_input_path_or_url"],
        "sst_dataset": "HadISST v3",
        "precip_dataset": "GPCC monthly precipitation",
        "sst_domain": "North Atlantic domain used in the Stone et al. AQM reproduction; see attribution artifact and source script.",
        "season_definition": "NDJF seasonal SST index labeled by the February/March-year season year",
        "sst_months_used": ["Nov", "Dec", "Jan", "Feb"],
        "precip_months_used_in_original_mode_derivation": ["Dec", "Jan", "Feb", "Mar"],
        "index_frequency": "seasonal",
        "index_date_labeling": "The reproduced AQM seasonal index is stored on March 1 of year Y for the Nov(Y-1)-Feb(Y) season.",
        "detrending": "Linear detrending at each grid point before MCA, as documented in the attribution reconstruction.",
        "global_mean_removed": False,
        "spatial_pattern_method": "Mode-2 homogeneous SST correlation map from lagged MCA of NDJF SST and DJFM precipitation.",
        "scalar_index_method": "S2 = standardized NDJF SST field projected onto the SST-side mode-2 MCA singular vector from the reproduction script.",
        "sign_convention": "Positive values indicate the warm AQM orientation defined by positive S2.",
        "uses_sierra_swe": False,
        "leakage_statement": (
            "The reused AQM index was reconstructed from HadISST and GPCC only, over a long climate record, "
            "with no Sierra SWE, no ridge coefficients, no selected predictor list, and no optimization against the Sierra target."
        ),
        "coverage_start": aqm_df["date"].min().strftime("%Y-%m-%d"),
        "coverage_end": aqm_df["date"].max().strftime("%Y-%m-%d"),
        "access_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if aqm_df.loc[aqm_df["aqm_defined"], "water_year"].max() < WATER_YEAR_END:
        print("Extending AQM seasonal index tail through WY{} using fixed-loading projection...".format(WATER_YEAR_END), flush=True)
        extended_df, validation = extend_aqm_index_from_fixed_loading(aqm_df, WATER_YEAR_END)
        aqm_df = aqm_df.merge(extended_df, on="date", how="left")
        fill_mask = aqm_df["AQM_S2_NDJF"].isna() & aqm_df["AQM_S2_NDJF_reprojected"].notna()
        aqm_df.loc[fill_mask, "AQM_S2_NDJF"] = aqm_df.loc[fill_mask, "AQM_S2_NDJF_reprojected"]
        aqm_df["aqm_defined"] = aqm_df["AQM_S2_NDJF"].notna()
        meta["reused_existing_index"] = False
        meta["reused_index_column"] = "AQM_S2_NDJF with fixed-loading projection fill for missing tail years"
        meta["tail_extension"] = {
            "used": True,
            "reason": "Saved attribution CSV stopped before WY2021 because the original GPCC-backed AQM reconstruction period ended early.",
            "method": (
                "Recomputed the Stone-style SST-side mode-2 MCA loading from the original HadISST+GPCC training period, "
                "then projected later NDJF HadISST seasons onto that fixed loading with training-period detrending and standardization."
            ),
            "validation_against_saved_overlap": validation,
            "filled_dates": aqm_df.loc[fill_mask, "date"].dt.strftime("%Y-%m-%d").tolist(),
        }
    else:
        meta["tail_extension"] = {"used": False}
    return aqm_df, meta


def _linear_fit_and_standardize(
    values: np.ndarray, time_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(time_index, dtype=float)
    y = np.asarray(values, dtype=float)
    x_center = float(np.mean(x))
    x_var = float(np.sum((x - x_center) ** 2))
    if x_var <= 0.0:
        raise ValueError("Time axis variance must be positive for detrending.")
    y_mean = np.mean(y, axis=0)
    slopes = np.sum((x[:, None] - x_center) * (y - y_mean[None, :]), axis=0) / x_var
    intercepts = y_mean - slopes * x_center
    detrended = y - (slopes[None, :] * x[:, None] + intercepts[None, :])
    mu = np.mean(detrended, axis=0)
    sigma = np.std(detrended, axis=0, ddof=1)
    keep = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(mu)
    if not np.any(keep):
        raise ValueError("No valid standardized columns remain after detrending.")
    return detrended[:, keep], mu[keep], sigma[keep], slopes[keep], intercepts[keep], keep


def extend_aqm_index_from_fixed_loading(saved_aqm_df: pd.DataFrame, target_year_end: int) -> tuple[pd.DataFrame, dict]:
    from scripts import run_atlantic_pc245_attribution as attribution

    print("Loading HadISST and GPCC for fixed-loading AQM extension...", flush=True)
    hadisst = attribution.load_hadisst_sst()
    gpcc = attribution.load_gpcc_precip()

    train_sst = attribution.latitude_slice(
        hadisst.sel(time=slice("1890-11-01", "2019-02-28")),
        attribution.AQM_SST_LAT_MIN,
        attribution.AQM_SST_LAT_MAX,
    )
    full_sst = attribution.latitude_slice(
        hadisst.sel(time=slice("1890-11-01", f"{target_year_end}-02-28")),
        attribution.AQM_SST_LAT_MIN,
        attribution.AQM_SST_LAT_MAX,
    )
    train_precip = attribution.latitude_slice(
        gpcc.sel(time=slice("1890-12-01", "2019-03-31"), lon=slice(attribution.AQM_PRECIP_LON_MIN, attribution.AQM_PRECIP_LON_MAX)),
        attribution.AQM_PRECIP_LAT_MIN,
        attribution.AQM_PRECIP_LAT_MAX,
    )

    train_sst_season = attribution.seasonal_average_label_by_year(train_sst, attribution.SEASON_MONTHS_SST, 3)
    full_sst_season = attribution.seasonal_average_label_by_year(full_sst, attribution.SEASON_MONTHS_SST, 3)
    train_precip_season = attribution.seasonal_average_label_by_year(train_precip, attribution.SEASON_MONTHS_PRECIP, 3)
    print(
        "Constructed seasonal fields: train_sst={}, full_sst={}, train_precip={}".format(
            train_sst_season.sizes["time"],
            full_sst_season.sizes["time"],
            train_precip_season.sizes["time"],
        ),
        flush=True,
    )

    common_train_years = np.intersect1d(
        train_sst_season["time"].dt.year.values.astype(int),
        train_precip_season["time"].dt.year.values.astype(int),
    )
    train_sst_season = train_sst_season.sel(time=train_sst_season["time"].dt.year.isin(common_train_years))
    train_precip_season = train_precip_season.sel(time=train_precip_season["time"].dt.year.isin(common_train_years))

    full_years = full_sst_season["time"].dt.year.values.astype(int)
    full_sst_season = full_sst_season.sel(time=full_sst_season["time"].dt.year.isin(full_years[full_years <= target_year_end]))

    x_train_raw = train_sst_season.values.reshape(train_sst_season.sizes["time"], -1)
    y_train_raw = train_precip_season.values.reshape(train_precip_season.sizes["time"], -1)
    x_full_raw = full_sst_season.values.reshape(full_sst_season.sizes["time"], -1)

    xmask = np.isfinite(x_train_raw).all(axis=0)
    ymask = np.isfinite(y_train_raw).all(axis=0)
    x_train = x_train_raw[:, xmask]
    x_full = x_full_raw[:, xmask]
    y_train = y_train_raw[:, ymask]

    train_years = train_sst_season["time"].dt.year.values.astype(int)
    full_years = full_sst_season["time"].dt.year.values.astype(int)
    t_train = train_years - int(train_years.min())
    t_full = full_years - int(train_years.min())

    x_train_dt, x_mu, x_sigma, x_slopes, x_intercepts, x_keep = _linear_fit_and_standardize(x_train, t_train)
    y_train_dt, y_mu, y_sigma, _, _, _ = _linear_fit_and_standardize(y_train, t_train)
    x_full = x_full[:, x_keep]

    x_train_std = (x_train_dt - x_mu[None, :]) / x_sigma[None, :]
    y_train_std = (y_train_dt - y_mu[None, :]) / y_sigma[None, :]

    cxy = (x_train_std.T @ y_train_std) / float(x_train_std.shape[0] - 1)
    u, _, _ = np.linalg.svd(cxy, full_matrices=False)
    sst_mode2_loading = u[:, 1]
    print(
        "Fixed-loading extension matrices: x_train={}, x_full={}, y_train={}".format(
            x_train_std.shape,
            x_full.shape,
            y_train_std.shape,
        ),
        flush=True,
    )

    x_full_dt = x_full - (x_slopes[None, :] * t_full[:, None] + x_intercepts[None, :])
    x_full_std = (x_full_dt - x_mu[None, :]) / x_sigma[None, :]
    s2_full = x_full_std @ sst_mode2_loading

    full_index_df = pd.DataFrame(
        {
            "date": pd.to_datetime(full_sst_season["time"].values),
            "AQM_S2_NDJF_reprojected": s2_full.astype(float),
        }
    )

    overlap = saved_aqm_df.merge(full_index_df, on="date", how="inner")
    overlap = overlap.loc[overlap["AQM_S2_NDJF"].notna()].copy()
    overlap_diff = overlap["AQM_S2_NDJF_reprojected"] - overlap["AQM_S2_NDJF"]
    validation = {
        "projection_basis_training_period_start": str(pd.Timestamp(train_sst_season["time"].values[0]).date()),
        "projection_basis_training_period_end": str(pd.Timestamp(train_sst_season["time"].values[-1]).date()),
        "full_projection_period_end": str(pd.Timestamp(full_sst_season["time"].values[-1]).date()),
        "overlap_count": int(len(overlap)),
        "overlap_max_abs_diff": float(np.max(np.abs(overlap_diff))) if len(overlap) else None,
        "overlap_rmse": float(np.sqrt(np.mean(overlap_diff**2))) if len(overlap) else None,
        "overlap_correlation": float(np.corrcoef(overlap["AQM_S2_NDJF"], overlap["AQM_S2_NDJF_reprojected"])[0, 1]) if len(overlap) > 1 else None,
        "filled_projection_years": [int(year) for year in full_index_df.loc[full_index_df["date"].dt.year > overlap["date"].dt.year.max(), "date"].dt.year.tolist()] if len(overlap) else [],
    }
    return full_index_df, validation


def build_predictor_table(aqm_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    seasonal = aqm_df.loc[aqm_df["aqm_defined"]].copy()
    seasonal = seasonal.loc[seasonal["month"] == 3, ["water_year", "AQM_S2_NDJF", "date"]].copy()
    seasonal = seasonal.rename(columns={"AQM_S2_NDJF": "AQM_NDJF", "date": "aqm_label_date"})
    seasonal["water_year"] = seasonal["water_year"].astype(int)

    merged = target_df.merge(seasonal, on="water_year", how="left")
    missing_years = merged.loc[merged["AQM_NDJF"].isna(), "water_year"].astype(int).tolist()
    if missing_years:
        raise ValueError("Missing AQM seasonal values for water years: {}".format(missing_years))
    if merged["water_year"].tolist() != WATER_YEARS:
        raise ValueError("Predictor table does not match WY1985-WY2021.")

    mapping = {
        "water_year_definition": "April 1 Sierra SWE anomaly target for water year Y",
        "predictor_name": "AQM_NDJF",
        "source_index_name": "AQM_S2_NDJF",
        "calendar_mapping": "AQM_NDJF for water year Y uses November and December of Y-1 plus January and February of Y.",
        "index_labeling": "Stored at March 1 of water year Y in the attribution artifact.",
        "usable_water_years": int(len(merged)),
        "missing_water_years": missing_years,
        "water_years": [int(x) for x in merged["water_year"].tolist()],
        "aqm_label_dates": merged["aqm_label_date"].dt.strftime("%Y-%m-%d").tolist(),
        "monthly_model_run": False,
        "monthly_model_reason": (
            "The attribution artifact provides a documented fixed seasonal Stone-style AQM scalar index, "
            "but not a separately documented fixed monthly scalar AQM basis. To avoid redefining the AQM "
            "predictor after the fact, this experiment runs only the primary seasonal index test."
        ),
    }
    result = merged[["water_year", "observed_swe", "AQM_NDJF"]].copy()
    return result, mapping


def run_loyo(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = table["water_year"].to_numpy(dtype=int)
    y = table["observed_swe"].to_numpy(dtype=float)
    x_all = table[FEATURE_COLUMNS].to_numpy(dtype=float)

    prediction_rows = []
    hyper_rows = []

    for held_idx, held_year in enumerate(years):
        train_mask = np.ones(len(years), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_all[train_mask, :]
        x_test = x_all[~train_mask, :][0]
        y_train = y[train_mask]
        y_test = float(y[held_idx])

        selected_alpha, inner_cv_mse = ref.inner_loyo_best_alpha(x_train, y_train)
        pred_raw, beta_raw, intercept_raw = ref.fit_outer_model(x_train, y_train, x_test, selected_alpha)

        residual = float(y_test - pred_raw)
        error = float(pred_raw - y_test)
        sign_correct = np.nan
        if y_test != 0.0 and pred_raw != 0.0:
            sign_correct = float(np.sign(y_test) == np.sign(pred_raw))

        prediction_rows.append(
            {
                "water_year": int(held_year),
                "observed_swe": float(y_test),
                "predicted_swe": float(pred_raw),
                "residual": residual,
                "error_pred_minus_obs": error,
                "abs_error": float(abs(error)),
                "selected_ridge_alpha": float(selected_alpha),
                "sign_correct": sign_correct,
                "intercept": float(intercept_raw),
            }
        )
        hyper_rows.append(
            {
                "water_year": int(held_year),
                "selected_ridge_alpha": float(selected_alpha),
                "inner_cv_mse": float(inner_cv_mse),
                "intercept": float(intercept_raw),
                "beta_AQM_NDJF": float(beta_raw[0]),
            }
        )

    pred_df = pd.DataFrame(prediction_rows).sort_values("water_year").reset_index(drop=True)
    hyper_df = pd.DataFrame(hyper_rows).sort_values("water_year").reset_index(drop=True)

    metric_bundle = ref.compute_metric_bundle(
        pred_df["observed_swe"].to_numpy(dtype=float),
        pred_df["predicted_swe"].to_numpy(dtype=float),
    )
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "num_predictors": len(FEATURE_COLUMNS),
                "r": metric_bundle["r"],
                "R2": metric_bundle["R2"],
                "RMSE": metric_bundle["RMSE"],
                "MAE": metric_bundle["MAE"],
                "sign_accuracy": metric_bundle["sign_accuracy"],
                "mean_error": metric_bundle["mean_error"],
                "median_abs_error": metric_bundle["median_abs_error"],
            }
        ]
    )

    period_rows = []
    obs = pred_df["observed_swe"].to_numpy(dtype=float)
    pred = pred_df["predicted_swe"].to_numpy(dtype=float)
    for group_name, selector in ref.PERIOD_SPECS:
        mask = selector(years)
        bundle = ref.compute_metric_bundle(obs[mask], pred[mask])
        period_rows.append(
            {
                "model_name": MODEL_NAME,
                "group_name": group_name,
                "n_years": int(mask.sum()),
                "r": bundle["r"],
                "R2": bundle["R2"],
                "RMSE": bundle["RMSE"],
                "MAE": bundle["MAE"],
                "sign_accuracy": bundle["sign_accuracy"],
                "mean_error": bundle["mean_error"],
                "median_abs_error": bundle["median_abs_error"],
            }
        )
    period_df = pd.DataFrame(period_rows).sort_values("group_name").reset_index(drop=True)
    return pred_df, hyper_df, metrics_df, period_df


def make_timeseries_plot(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.plot(pred_df["water_year"], pred_df["observed_swe"], color="black", linewidth=2.4, label="Observed")
    ax.plot(
        pred_df["water_year"],
        pred_df["predicted_swe"],
        color="#d55e00",
        marker="o",
        linewidth=1.8,
        label="AQM NDJF index only",
    )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("April 1 Sierra SWE anomaly (m)")
    ax.set_title("Strict LOYO prediction using the independent seasonal AQM index")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(TIMESERIES_PNG, dpi=200)
    plt.close(fig)


def make_scatter_plot(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    obs = pred_df["observed_swe"].to_numpy(dtype=float)
    pred = pred_df["predicted_swe"].to_numpy(dtype=float)
    lo = min(np.min(obs), np.min(pred))
    hi = max(np.max(obs), np.max(pred))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lims = (lo - pad, hi + pad)

    row = metrics_df.iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(obs, pred, color="#d55e00", s=52, alpha=0.85)
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed SWE anomaly (m)")
    ax.set_ylabel("Predicted SWE anomaly (m)")
    ax.set_title("Observed vs predicted SWE: seasonal AQM index only")
    ax.grid(True, alpha=0.2)
    ax.text(
        0.04,
        0.96,
        "RMSE = {:.4f}\n$R^2$ = {:.3f}\nCorr = {:.3f}\nSign = {:.3f}".format(
            float(row["RMSE"]),
            float(row["R2"]),
            float(row["r"]),
            float(row["sign_accuracy"]),
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=200)
    plt.close(fig)


def run_reference_audit() -> dict:
    table, _ = ref.load_predictor_table()
    rerun_pred_df, rerun_alpha_df, _ = ref.run_models(table)
    rerun_metrics_df = ref.compute_metrics(rerun_pred_df)

    saved_pred_df = pd.read_csv(ref.PREDICTIONS_CSV).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    saved_alpha_df = pd.read_csv(ref.ALPHA_CSV).sort_values(["model_name", "heldout_wy"]).reset_index(drop=True)
    saved_metrics_df = pd.read_csv(ref.METRICS_CSV).sort_values("model_name").reset_index(drop=True)
    rerun_metrics_df = rerun_metrics_df.sort_values("model_name").reset_index(drop=True)

    pred_diff = float(
        np.max(np.abs(rerun_pred_df["pred_swe"].to_numpy(dtype=float) - saved_pred_df["pred_swe"].to_numpy(dtype=float)))
    )
    alpha_diff = float(
        np.max(
            np.abs(
                rerun_alpha_df["selected_alpha"].to_numpy(dtype=float)
                - saved_alpha_df["selected_alpha"].to_numpy(dtype=float)
            )
        )
    )
    metric_diff = {}
    for key in ["r", "R2", "RMSE", "MAE", "sign_accuracy"]:
        metric_diff[key] = float(
            np.max(np.abs(rerun_metrics_df[key].to_numpy(dtype=float) - saved_metrics_df[key].to_numpy(dtype=float)))
        )

    return {
        "reference_script": str(Path(ref.__file__).resolve()),
        "reference_output_dir": str(ref.OUTPUT_DIR),
        "water_years_match": bool(table["water_year"].astype(int).tolist() == WATER_YEARS),
        "alpha_grid_match": [float(x) for x in ref.ALPHA_GRID.tolist()],
        "prediction_max_abs_diff": pred_diff,
        "selected_alpha_max_abs_diff": alpha_diff,
        "metric_max_abs_diff": metric_diff,
        "reproduces_saved_reference_outputs": bool(
            pred_diff <= 1.0e-12
            and alpha_diff <= 1.0e-12
            and all(value <= 1.0e-12 for value in metric_diff.values())
        ),
    }


def load_comparison_metrics() -> dict:
    noaa_overall = pd.read_csv(NOAA_AMV_METRICS_CSV).iloc[0].to_dict()
    noaa_period = pd.read_csv(NOAA_AMV_PERIOD_METRICS_CSV).sort_values("group_name").to_dict(orient="records")

    amv_metrics = pd.read_csv(AMV_CORE_METRICS_CSV)
    full_pc = amv_metrics.loc[amv_metrics["model_name"] == "AMV_AMO_PC1to6_full"].iloc[0].to_dict()
    k5 = amv_metrics.loc[amv_metrics["model_name"] == "AMV_core_plus_K5"].iloc[0].to_dict()

    single_var = pd.read_csv(SINGLE_VARIABLE_RANKING_CSV)
    pc2_feb = single_var.loc[single_var["feature"] == "AMV_PC2_Feb"]
    pc2_feb_dict = None if pc2_feb.empty else pc2_feb.iloc[0].to_dict()

    return {
        "noaa_amv_overall": noaa_overall,
        "noaa_amv_period_metrics": noaa_period,
        "amv_pc1to6_full_overall": full_pc,
        "amv_selected_k5_overall": k5,
        "existing_pc2_feb_single_predictor": pc2_feb_dict,
    }


def determine_conclusion(overall: dict) -> str:
    r = float(overall["r"])
    r2 = float(overall["R2"])
    sign_accuracy = float(overall["sign_accuracy"])
    if r > 0.0 and r2 > 0.0 and sign_accuracy >= 0.5:
        return "An independently defined AQM index contains out-of-sample predictive information for April 1 Sierra Nevada SWE."
    return (
        "Although Atlantic EOF2 is spatially and temporally consistent with AQM, the independently defined scalar "
        "AQM index does not reproduce the April 1 SWE predictability under strict LOYO."
    )


def write_readme(
    source_meta: dict,
    mapping_meta: dict,
    metrics_df: pd.DataFrame,
    period_df: pd.DataFrame,
    audit: dict,
    comparison: dict,
) -> None:
    overall = metrics_df.iloc[0].to_dict()
    conclusion = determine_conclusion(overall)
    lines = [
        "# Independent AQM index -> April 1 Sierra SWE",
        "",
        "This artifact runs the independently reconstructed Stone et al. seasonal Atlantic Quadpole Mode index as the only predictor block for strict LOYO ridge prediction of April 1 Sierra SWE.",
        "",
        "## Compute session",
        "",
        "- tmux session: `aqm_index_loyo`",
        "- Slurm job ID: `{}`".format(os.environ.get("SLURM_JOB_ID", "")),
        "- Hostname: `{}`".format(os.uname().nodename),
        "- Python: `{}`".format(sys.executable),
        "- Interactive allocation command: `{}`".format(EXACT_COMMAND),
        "",
        "## AQM source and method",
        "",
        "- Source paper: {}".format(source_meta["source_paper"]),
        "- Source description: {}".format(source_meta["source_description"]),
        "- Reused existing index: {}".format(source_meta["reused_existing_index"]),
        "- Reused index column: `{}`".format(source_meta["reused_index_column"]),
        "- Seasonal definition: {}".format(source_meta["season_definition"]),
        "- Preprocessing: {}".format(source_meta["preprocessing"]),
        "- Sign convention: {}".format(source_meta["sign_convention"]),
        "- SWE independence: {}".format(source_meta["leakage_statement"]),
        "",
        "## Water-year mapping",
        "",
        "- `AQM_NDJF`: {}".format(mapping_meta["calendar_mapping"]),
        "- Index label date: {}".format(mapping_meta["index_labeling"]),
        "- Water years used: {}-{}".format(WATER_YEAR_START, WATER_YEAR_END),
        "- Missing water years: {}".format(mapping_meta["missing_water_years"]),
        "",
        "## Predictor set",
        "",
        "- Predictors: {}".format(", ".join(FEATURE_COLUMNS)),
        "- Usable water years: {}".format(mapping_meta["usable_water_years"]),
        "- Monthly AQM model run: {}".format(mapping_meta["monthly_model_run"]),
        "- Monthly-model note: {}".format(mapping_meta["monthly_model_reason"]),
        "",
        "## Ridge configuration reused from the reference workflow",
        "",
        "- Outer split: leave one water year out over WY1985-WY2021",
        "- Predictor standardization: train-fold mean/std only",
        "- Target standardization: train-fold mean/std only",
        "- Alpha grid: {}".format(", ".join("{:g}".format(x) for x in ref.ALPHA_GRID.tolist())),
        "- Inner selection: leave one training year out, choose lowest MSE, break ties toward smaller alpha",
        "- Raw-space reconstruction: use the reference script's coefficient and intercept reconstruction",
        "",
        "## Seasonal AQM result",
        "",
        "- Overall: r={r:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, sign_accuracy={sign_accuracy:.6f}".format(
            **overall
        ),
        "- Conclusion: {}".format(conclusion),
        "",
        "## Period metrics",
        "",
    ]
    for _, row in period_df.iterrows():
        lines.append(
            "- {} (n={}): r={:.6f}, R2={:.6f}, RMSE={:.6f}, MAE={:.6f}, sign_accuracy={:.6f}".format(
                row["group_name"],
                int(row["n_years"]),
                float(row["r"]),
                float(row["R2"]),
                float(row["RMSE"]),
                float(row["MAE"]),
                float(row["sign_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reference-pipeline audit",
            "",
            "- Reference outputs reproduced exactly: {}".format(audit["reproduces_saved_reference_outputs"]),
            "- Max abs prediction diff vs saved reference: {:.3e}".format(audit["prediction_max_abs_diff"]),
            "- Max abs selected-alpha diff vs saved reference: {:.3e}".format(audit["selected_alpha_max_abs_diff"]),
            "",
            "## Factual comparisons",
            "",
            "- NOAA AMV index only: r={r:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, sign_accuracy={sign_accuracy:.6f}".format(
                **comparison["noaa_amv_overall"]
            ),
            "- Full Atlantic PC1-PC6 model: r={r:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, sign_accuracy={sign_accuracy:.6f}".format(
                **comparison["amv_pc1to6_full_overall"]
            ),
            "- Selected Atlantic K5 model: r={r:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, sign_accuracy={sign_accuracy:.6f}".format(
                **comparison["amv_selected_k5_overall"]
            ),
        ]
    )
    if comparison["existing_pc2_feb_single_predictor"] is not None:
        row = comparison["existing_pc2_feb_single_predictor"]
        lines.append(
            "- Existing single-predictor `AMV_PC2_Feb`: r={:.6f}, R2={:.6f}, RMSE={:.6f}, MAE={:.6f}, sign_accuracy={:.6f}".format(
                float(row["r"]),
                float(row["R2"]),
                float(row["RMSE"]),
                float(row["MAE"]),
                float(row["sign_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `{}`".format(SOURCE_METADATA_JSON),
            "- `{}`".format(INDEX_FULL_RECORD_CSV),
            "- `{}`".format(PREDICTOR_TABLE_CSV),
            "- `{}`".format(PREDICTIONS_CSV),
            "- `{}`".format(HYPERPARAMETERS_CSV),
            "- `{}`".format(METRICS_CSV),
            "- `{}`".format(PERIOD_METRICS_CSV),
            "- `{}`".format(TIMESERIES_PNG),
            "- `{}`".format(SCATTER_PNG),
            "- `{}`".format(AUDIT_JSON),
        ]
    )
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ref.ensure_runtime_on_compute_node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading reference SWE target and AQM index...", flush=True)
    target_df = load_reference_target_series()
    aqm_df, source_meta = load_aqm_index()
    print("Building seasonal predictor table...", flush=True)
    predictor_table, mapping_meta = build_predictor_table(aqm_df, target_df)

    aqm_df.to_csv(INDEX_FULL_RECORD_CSV, index=False)
    predictor_table.to_csv(PREDICTOR_TABLE_CSV, index=False)

    print("Running strict LOYO ridge...", flush=True)
    pred_df, hyper_df, metrics_df, period_df = run_loyo(predictor_table)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    hyper_df.to_csv(HYPERPARAMETERS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    period_df.to_csv(PERIOD_METRICS_CSV, index=False)

    make_timeseries_plot(pred_df)
    make_scatter_plot(pred_df, metrics_df)

    print("Running reference-pipeline audit...", flush=True)
    audit = run_reference_audit()
    comparison = load_comparison_metrics()

    source_meta["water_year_mapping"] = mapping_meta
    source_meta["compute_session"] = {
        "tmux_session": "aqm_index_loyo",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "interactive_command": EXACT_COMMAND,
    }
    SOURCE_METADATA_JSON.write_text(json.dumps(source_meta, indent=2) + "\n", encoding="utf-8")

    audit_payload = {
        "target_series_identical_to_reference": bool(
            predictor_table["water_year"].astype(int).tolist() == WATER_YEARS
            and int(len(predictor_table)) == len(WATER_YEARS)
        ),
        "predictor_columns": FEATURE_COLUMNS,
        "outer_lowo_years": WATER_YEARS,
        "reference_audit": audit,
        "aqm_construction_independent_of_swe": True,
        "aqm_reused_existing_index": True,
        "monthly_model_run": False,
        "monthly_model_reason": mapping_meta["monthly_model_reason"],
        "comparisons": comparison,
        "conclusion": determine_conclusion(metrics_df.iloc[0].to_dict()),
    }
    AUDIT_JSON.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    write_readme(source_meta, mapping_meta, metrics_df, period_df, audit, comparison)

    overall = metrics_df.iloc[0]
    print("Wrote AQM seasonal LOYO artifact to {}".format(OUTPUT_DIR), flush=True)
    print(
        "Metrics: r={:.6f}, R2={:.6f}, RMSE={:.6f}, MAE={:.6f}, sign_accuracy={:.6f}".format(
            float(overall["r"]),
            float(overall["R2"]),
            float(overall["RMSE"]),
            float(overall["MAE"]),
            float(overall["sign_accuracy"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
