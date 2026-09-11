#!/usr/bin/env python
"""Round-1 frozen-representation linear probe (simplified, direct-Ridge only):
does pretrained ClimaX H contain linearly accessible information about
April-1 Sierra SWE, CPM, or AQM?

Per explicit scope: NO PCA anywhere in this round (neither on the raw ERA5
fields nor on the ClimaX pooled representation). Direct Ridge only, since
p=1024 >> n=37 is an appropriate regime for Ridge without dimensionality
reduction. SWE runs first and is reported immediately; CPM/AQM follow using
their existing trusted artifacts (CPM: the 32-year artifact, no extension
attempt in this script at all).

Reads existing artifacts read-only:
  - climax_era5_march25_obs_latents/H_march25_obs_latents.npy   (37,2048,1024)
  - climax_era5_march25_obs_latents/paired_dataset.npz          (SWE anomaly target)
  - climax_era5_march25_obs_latents/regridded_1p40625deg/*.nc   (raw ERA5 physical fields, pre-normalization)
  - artifacts/cpm_swe_attribution/cpm_pc3_nov_apr_by_water_year.csv (CPM, trusted 32yr)
  - artifacts/aqm_index_loyo/aqm_index_full_record.csv          (AQM)

No fine-tuning, no MLP, no nonlinear probe, no PCA. Strict leakage-safe LOYO:
StandardScaler and Ridge alpha selection fit ONLY on the 36 training years
of each outer fold.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = "/global/u1/h/hyvchen/Snow-Predication-at-Sierra"
LATENT_DIR = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents"
REGRID_DIR = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/regridded_1p40625deg"
OUT_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/climax_frozen_obs_probe_round1"
PRED_DIR = os.path.join(OUT_ROOT, "predictions")
PLOT_DIR = os.path.join(OUT_ROOT, "plots")

CPM_32YR_CSV = os.path.join(REPO_ROOT, "artifacts/cpm_swe_attribution/cpm_pc3_nov_apr_by_water_year.csv")
AQM_CSV = os.path.join(REPO_ROOT, "artifacts/aqm_index_loyo/aqm_index_full_record.csv")
BASELINE_README = os.path.join(REPO_ROOT, "artifacts/kernel_ridge_compact_inputs_loyo/README.md")

WATER_YEARS = list(range(1985, 2022))
RAW_FIELD_VARKEYS = ["zg_500", "ta_850", "ua_850", "va_850", "hus_850", "tas"]

ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def ensure_dirs():
    for d in (PRED_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Targets (no CPM extension attempted here -- separate task, uses trusted
# 32-year artifact directly, per explicit instruction)
# ---------------------------------------------------------------------------

def load_targets():
    paired = np.load(os.path.join(LATENT_DIR, "paired_dataset.npz"), allow_pickle=True)
    swe_wy = paired["water_year"].astype(int)
    assert list(swe_wy) == WATER_YEARS
    swe_standardized = paired["swe_apr1_standardized"].astype(np.float64)
    swe_anom_mm = paired["swe_apr1_anom_mm"].astype(np.float64)

    cpm_df = pd.read_csv(CPM_32YR_CSV).sort_values("water_year")
    cpm_years_available = cpm_df["water_year"].astype(int).tolist()
    cpm_raw = cpm_df["cpm_pc3_nov_apr_mean"].to_numpy(dtype=np.float64)
    cpm_standardized = (cpm_raw - cpm_raw.mean()) / cpm_raw.std(ddof=0)

    aqm_full = pd.read_csv(AQM_CSV)
    aqm_full["water_year"] = aqm_full["water_year"].astype(int)
    aqm_full["month"] = aqm_full["month"].astype(int)
    aqm_march = aqm_full[(aqm_full["month"] == 3) & (aqm_full["water_year"].isin(WATER_YEARS))].sort_values("water_year")
    assert list(aqm_march["water_year"]) == WATER_YEARS
    aqm_raw = aqm_march["AQM_S2_NDJF"].to_numpy(dtype=np.float64)
    aqm_standardized = (aqm_raw - aqm_raw.mean()) / aqm_raw.std(ddof=0)

    return {
        "SWE": {
            "water_years": WATER_YEARS, "y_standardized": swe_standardized,
            "y_physical_anom_mm": swe_anom_mm,
            "source_artifact": os.path.join(LATENT_DIR, "paired_dataset.npz")
            + " (from sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc, unmodified)",
        },
        "CPM": {
            "water_years": cpm_years_available, "y_standardized": cpm_standardized,
            "y_physical_anom_mm": None, "source_artifact": CPM_32YR_CSV,
            "note": "trusted 32-year artifact (WY1985-2016); the known leap-day extension bug is a separate task",
        },
        "AQM": {
            "water_years": WATER_YEARS, "y_standardized": aqm_standardized,
            "y_physical_anom_mm": None, "source_artifact": AQM_CSV,
        },
    }


def build_audit_table(targets: dict) -> pd.DataFrame:
    rows = []
    for name, t in targets.items():
        y = t["y_standardized"]
        rows.append({
            "target": name, "source_artifact": t["source_artifact"], "shape": len(y),
            "water_years": f"{t['water_years'][0]}-{t['water_years'][-1]} (n={len(t['water_years'])})",
            "mean": float(np.mean(y)), "std": float(np.std(y, ddof=0)),
            "min": float(np.min(y)), "max": float(np.max(y)),
        })
    return pd.DataFrame(rows)


def load_climax_pooled() -> np.ndarray:
    H = np.load(os.path.join(LATENT_DIR, "H_march25_obs_latents.npy"))
    assert H.shape == (37, 2048, 1024), H.shape
    Z = H.mean(axis=1)
    assert Z.shape == (37, 1024)
    return Z


def load_raw_fields() -> dict[str, np.ndarray]:
    fields = {}
    for vk in RAW_FIELD_VARKEYS:
        ds = xr.open_dataset(os.path.join(REGRID_DIR, f"{vk}_march25_1p40625deg.nc"))
        arr = ds[vk].values.astype(np.float64)
        assert arr.shape[0] == 37, (vk, arr.shape)
        fields[vk] = arr
        ds.close()
    return fields


def raw_global_summary_features(fields: dict[str, np.ndarray], lat: np.ndarray) -> tuple[np.ndarray, list[str]]:
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()
    feats, names = [], []
    for vk in RAW_FIELD_VARKEYS:
        arr = fields[vk]
        weighted_mean_map = np.average(arr, axis=1, weights=w)
        gmean = weighted_mean_map.mean(axis=1)
        dev = arr - gmean[:, None, None]
        weighted_var_map = np.average(dev ** 2, axis=1, weights=w)
        gstd = np.sqrt(weighted_var_map.mean(axis=1))
        feats += [gmean, gstd]
        names += [f"{vk}_gmean", f"{vk}_gstd"]
    return np.stack(feats, axis=1), names


def inner_loo_select_alpha(X_train: np.ndarray, y_train: np.ndarray, alphas: list[float]) -> float:
    n = X_train.shape[0]
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            scaler = StandardScaler().fit(X_train[mask])
            model = Ridge(alpha=alpha).fit(scaler.transform(X_train[mask]), y_train[mask])
            preds[i] = model.predict(scaler.transform(X_train[i:i + 1]))[0]
        ss_res = np.sum((y_train - preds) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        score = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def run_loyo_direct_ridge(X: np.ndarray, y: np.ndarray, years: list[int]) -> pd.DataFrame:
    n = len(years)
    rows = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]
        alpha = inner_loo_select_alpha(X_train, y_train, ALPHA_GRID)
        scaler = StandardScaler().fit(X_train)
        model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)
        y_pred = model.predict(scaler.transform(X[i:i + 1]))[0]
        rows.append({"water_year": years[i], "y_true": y[i], "y_pred": y_pred, "selected_alpha": alpha})
    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, sign_accuracy: bool = False) -> dict:
    r, _ = scipy_stats.pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    n = len(y_true)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(max(n - 3, 1))
    ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    out = {"r": float(r), "r_ci_lo": float(ci_lo), "r_ci_hi": float(ci_hi),
           "R2": float(r2), "RMSE": rmse, "MAE": mae, "n": n}
    if sign_accuracy:
        out["sign_accuracy"] = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return out


def mode_of(series: pd.Series):
    vc = series.value_counts()
    return vc.index[0] if len(vc) else None


def run_target(target_name: str, tinfo: dict, representations: dict, all_metrics: list) -> None:
    y_full = tinfo["y_standardized"]
    target_years = tinfo["water_years"]
    year_index = {wy: i for i, wy in enumerate(WATER_YEARS)}
    idx = [year_index[wy] for wy in target_years]

    for rep_name, X_full in representations.items():
        X = X_full[idx]
        y = y_full
        print(f"=== LOYO: target={target_name} representation={rep_name} n={len(y)} ===", flush=True)
        preds_df = run_loyo_direct_ridge(X, y, target_years)
        preds_path = os.path.join(PRED_DIR, f"{target_name}__{rep_name}.csv")
        preds_df.to_csv(preds_path, index=False)
        m = compute_metrics(preds_df["y_true"].to_numpy(), preds_df["y_pred"].to_numpy(),
                             sign_accuracy=(target_name == "SWE"))
        m.update({"target": target_name, "representation": rep_name, "n_years": len(y),
                  "modal_alpha": mode_of(preds_df["selected_alpha"]), "predictions_path": preds_path})
        all_metrics.append(m)
        line = f"  r={m['r']:.4f} R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f}"
        if "sign_accuracy" in m:
            line += f" sign_acc={m['sign_accuracy']:.4f}"
        print(line, flush=True)


def main():
    ensure_dirs()

    print("=== Load targets (SWE / CPM / AQM); NO CPM extension attempted (separate task) ===", flush=True)
    targets = load_targets()
    audit_df = build_audit_table(targets)
    audit_df.to_csv(os.path.join(OUT_ROOT, "target_audit_table.csv"), index=False)
    print(audit_df.to_string(index=False), flush=True)

    print("=== Load ClimaX global-mean-pooled representation ===", flush=True)
    Z_climax = load_climax_pooled()
    print(f"  Z_climax: shape={Z_climax.shape} dtype={Z_climax.dtype} "
          f"nan={int(np.isnan(Z_climax).sum())} inf={int(np.isinf(Z_climax).sum())} "
          f"mean={Z_climax.mean():.6f} std={Z_climax.std():.6f} "
          f"min={Z_climax.min():.6f} max={Z_climax.max():.6f}", flush=True)

    print("=== Build raw ERA5 global-summary baseline (12 features, no PCA) ===", flush=True)
    fields = load_raw_fields()
    sample_ds = xr.open_dataset(os.path.join(REGRID_DIR, "tas_march25_1p40625deg.nc"))
    lat = sample_ds["lat"].values
    sample_ds.close()
    X_raw_global, raw_global_names = raw_global_summary_features(fields, lat)
    print(f"  Raw-ERA5 Global Summary: shape={X_raw_global.shape} features={raw_global_names}", flush=True)

    representations = {
        "Raw-ERA5-GlobalSummary": X_raw_global,
        "ClimaX-H-Direct": Z_climax,
    }

    all_metrics: list[dict] = []

    print("\n" + "#" * 70, flush=True)
    print("# PRIORITY: SWE FIRST -- reporting immediately once done", flush=True)
    print("#" * 70, flush=True)
    run_target("SWE", targets["SWE"], representations, all_metrics)
    swe_metrics_df = pd.DataFrame([m for m in all_metrics if m["target"] == "SWE"])
    swe_metrics_df.to_csv(os.path.join(OUT_ROOT, "metrics_SWE_only.csv"), index=False)
    print("\n=== SWE RESULT (immediate) ===", flush=True)
    print(swe_metrics_df.to_string(index=False), flush=True)

    print("\n" + "#" * 70, flush=True)
    print("# CPM (trusted 32-year artifact, no extension attempted)", flush=True)
    print("#" * 70, flush=True)
    run_target("CPM", targets["CPM"], representations, all_metrics)

    print("\n" + "#" * 70, flush=True)
    print("# AQM", flush=True)
    print("#" * 70, flush=True)
    run_target("AQM", targets["AQM"], representations, all_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUT_ROOT, "metrics.csv"), index=False)

    print("\n=== Existing observational SWE benchmark (reference only) ===", flush=True)
    benchmark_note = {
        "source_artifact": BASELINE_README,
        "model_name": "Ridge Z1/Z2 + AMV/AMO K5",
        "R2": 0.495860, "RMSE": 0.022516, "Pearson_r": 0.710375, "MAE": 0.018787, "sign_accuracy": 0.783784,
        "note": "Different feature set (Z1/Z2 + AMV/AMO PCs), NOT the March-25 ClimaX/raw-ERA5 comparison "
                "in this task -- included for context only, not mixed into the comparison table.",
    }
    print(json.dumps(benchmark_note, indent=2), flush=True)

    print("=== Leakage audit ===", flush=True)
    leakage_audit = [
        "No SWE/CPM/AQM values used in constructing H (built purely from the 10 approved ERA5 variables).",
        "No held-out target used in normalization: ClimaX input normalization stats were fixed at extraction "
        "time from the 37 ERA5 snapshots only; StandardScaler for THIS probe is fit fresh inside each outer "
        "LOYO training fold only.",
        "No PCA used anywhere in this round (removed per explicit instruction).",
        "No feature selection using target values: features (global-mean pooling, raw global summary) are "
        "fixed transforms of X only, never functions of y.",
        "No Ridge alpha chosen from held-out predictions: alpha selected via inner leave-one-out on the "
        "outer training fold only; the outer test year is never seen during selection.",
        "All years appear exactly once as outer test year (verified per representation/target).",
        "Output predictions aligned back to correct water_year via an explicit column, not positional inference.",
        "CPM: no extension attempted in this script; used the trusted 32-year (WY1985-2016) artifact as-is, "
        "per explicit instruction that the leap-day extension fix is a separate task.",
    ]
    with open(os.path.join(OUT_ROOT, "leakage_audit.txt"), "w") as f:
        f.write("\n".join(f"- {line}" for line in leakage_audit))

    for rep_name in representations:
        for target_name, tinfo in targets.items():
            df = pd.read_csv(os.path.join(PRED_DIR, f"{target_name}__{rep_name}.csv"))
            assert df["water_year"].nunique() == len(df) == len(tinfo["water_years"])

    probe_config = {
        "alpha_grid": ALPHA_GRID,
        "cv_scheme": "outer LOYO (leave-one-year-out); inner leave-one-out on the outer-training fold for alpha selection",
        "representations": list(representations.keys()),
        "raw_field_variables": RAW_FIELD_VARKEYS,
        "raw_global_summary_features": raw_global_names,
        "latitude_weighting": "cosine-latitude weighting applied to the global mean/std of raw ERA5 fields",
        "pca_used": False,
        "targets": {k: {"water_years": v["water_years"], "source_artifact": v["source_artifact"]} for k, v in targets.items()},
    }
    with open(os.path.join(OUT_ROOT, "probe_config.json"), "w") as f:
        json.dump(probe_config, f, indent=2, default=str)

    summary = {
        "source_artifacts": {
            "H_latents": os.path.join(LATENT_DIR, "H_march25_obs_latents.npy"),
            "paired_dataset": os.path.join(LATENT_DIR, "paired_dataset.npz"),
            "raw_regridded_fields_dir": REGRID_DIR,
        },
        "target_audit_table": audit_df.to_dict(orient="records"),
        "climax_pooled_shape": list(Z_climax.shape),
        "climax_pooled_stats": {"mean": float(Z_climax.mean()), "std": float(Z_climax.std()),
                                 "min": float(Z_climax.min()), "max": float(Z_climax.max()),
                                 "n_nan": int(np.isnan(Z_climax).sum()), "n_inf": int(np.isinf(Z_climax).sum())},
        "existing_observational_swe_benchmark_reference_only": benchmark_note,
        "metrics_csv": os.path.join(OUT_ROOT, "metrics.csv"),
        "output_root": OUT_ROOT,
    }
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== DONE ===", flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
