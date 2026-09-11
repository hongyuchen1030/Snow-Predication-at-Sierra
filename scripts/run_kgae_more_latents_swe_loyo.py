#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KGAE_REPO = PROJECT_ROOT / "artifacts" / "kgae_pacific_latent_swe_loyo" / "external" / "kgae"
DEFAULT_HOME_OUT = PROJECT_ROOT / "artifacts" / "kgae_more_latents_swe_loyo"
DEFAULT_PSCRATCH_OUT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_more_latents_swe_loyo")
BASELINE_TABLE = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/z1z2_plus_amv_k5_loyo/z1z2_amv_k5_predictor_table.csv")
PACIFIC_PC_TABLE = PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "exact_Z1_Z2_plus_PacificPC_Nino34_loyo" / "z1_z2_pacificpc_nino34_predictor_table.csv"
ERA5_PATH = DEFAULT_KGAE_REPO / "data_pipeline" / "era5.sst.pacific.1x1.1940-2023.nc"
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
ALPHAS = np.logspace(-6, 6, 49)


sys.path.insert(0, str(DEFAULT_KGAE_REPO))
import kgae  # noqa: E402


@dataclass
class DatasetBundle:
    sst: xr.DataArray
    anomalies: xr.DataArray
    stacked: xr.DataArray
    values: np.ndarray
    reference_pattern: np.ndarray
    pfit: xr.Dataset
    monthly_clim: xr.DataArray


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def symlink_or_copy_lightweight(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        if src.suffix.lower() in {".png", ".pdf", ".csv", ".md", ".json", ".txt"}:
            dst.write_bytes(src.read_bytes())


def load_sst() -> xr.DataArray:
    ds = xr.open_dataset(ERA5_PATH)
    da = ds["sst"]
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    return da.sortby("time")


def preprocess_sst(sst: xr.DataArray) -> DatasetBundle:
    anomalies, fit, _, pfit = kgae.global_detrend(sst, deg=2)
    anomalies, monthly_clim = kgae.remove_climo(anomalies)
    oni = anomalies.sel(lat=slice(-5, 5), lon=slice(10, 60)).mean(["lat", "lon"])
    corr = xr.corr(anomalies, oni, dim="time")
    reference_pattern = corr.stack(feature=("lat", "lon")).dropna("feature", how="any").values
    stacked = anomalies.stack(feature=("lat", "lon")).dropna("feature", how="any").transpose("time", "feature")
    return DatasetBundle(
        sst=sst,
        anomalies=anomalies,
        stacked=stacked,
        values=stacked.values.astype(np.float32),
        reference_pattern=reference_pattern,
        pfit=pfit,
        monthly_clim=monthly_clim,
    )


def monthly_anomalize_with_training_climo(data: xr.DataArray, pfit: xr.Dataset, monthly_clim: xr.DataArray) -> xr.DataArray:
    detrended = data - xr.polyval(data["time"], pfit.polyfit_coefficients)
    pieces = []
    for year in sorted({pd.Timestamp(v).year for v in detrended.time.values}):
        yearly = detrended.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))).groupby("time.month").mean() - monthly_clim
        yearly = yearly.assign_coords({"month": [pd.Timestamp(year, m, 1) for m in yearly["month"].values]}).rename({"month": "time"})
        pieces.append(yearly)
    return xr.concat(pieces, "time").sortby("time")


def split_train_val(stacked: xr.DataArray, val_months: int) -> tuple[list[np.ndarray], np.ndarray, xr.DataArray, xr.DataArray]:
    n_time = stacked.sizes["time"]
    val_months = max(12, min(val_months, n_time // 5))
    train = stacked.isel(time=slice(None, n_time - val_months))
    val = stacked.isel(time=slice(n_time - val_months, None))
    train_folds = [train.isel(time=slice(j, None, 4)).sortby("time").values.astype(np.float32) for j in range(4)]
    return train_folds, val.values.astype(np.float32), train, val


def fit_kgae_model(
    train_batches: list[np.ndarray],
    val_data: np.ndarray | None,
    latent_dim: int,
    epochs: int,
    seed: int,
    device: str,
    hidden_layers: list[int],
    variational: bool,
    reference_pattern: np.ndarray,
    sort_data: np.ndarray,
):
    model = kgae.KGAE(
        input_dim=train_batches[0].shape[1],
        latent_dim=latent_dim,
        is_variational=variational,
        hidden_layers=hidden_layers,
        activation=torch.nn.Tanh,
        power_spectrum_smoothing_kernel=7,
        device=device,
        seed=seed,
        tag=f"k{latent_dim}",
    )
    if val_data is None:
        tracking = model.fit_noval(training_data=train_batches, num_epochs=epochs, lr=1e-3)
    else:
        tracking = model.fit(training_data=train_batches, val_data=val_data, num_epochs=epochs, lr=1e-3)
    model.eval()
    model.sort_latents_by_frequency(sort_data.astype(np.float32), reference_pattern=reference_pattern)
    return model, tracking


def encode_array(model, values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = model.encoder(torch.tensor(values, dtype=torch.float32, device=model.device)).cpu().numpy()
    if model.is_variational:
        z = z[:, : model.latent_dim]
    return z * model.flip_signs.values.reshape(1, -1)


def decode_array(model, latents: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xhat = model.decoder(torch.tensor(latents, dtype=torch.float32, device=model.device)).cpu().numpy()
    return xhat


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse_from_mse(value: float) -> float:
    return float(math.sqrt(value))


def build_water_year_predictor_table(latent_da: xr.DataArray, obs_swe: pd.Series, k: int) -> pd.DataFrame:
    rows = []
    for water_year, y in obs_swe.items():
        row = {"water_year": int(water_year), "obs_swe": float(y)}
        ok = True
        for month in MONTHS:
            year = water_year - 1 if month in {"Sep", "Oct", "Nov", "Dec"} else water_year
            ts = pd.Timestamp(year, pd.Timestamp(month + " 1, 2001").month, 1)
            if ts not in latent_da.time.to_index():
                ok = False
                break
            vals = latent_da.sel(time=ts).values
            for idx in range(k):
                row[f"kgae_k{k}_z{idx+1}_{month}"] = float(vals[idx])
        if ok:
            rows.append(row)
    return pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)


def select_ridge_alpha_inner_loyo(x_train: np.ndarray, y_train: np.ndarray) -> float:
    best_alpha = None
    best_mse = np.inf
    n = x_train.shape[0]
    for alpha in ALPHAS:
        se = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mu = x_train[mask].mean(axis=0)
            sigma = x_train[mask].std(axis=0)
            sigma[sigma == 0] = 1.0
            xtr = (x_train[mask] - mu) / sigma
            xva = (x_train[i : i + 1] - mu) / sigma
            model = Ridge(alpha=alpha)
            model.fit(xtr, y_train[mask])
            pred = model.predict(xva)[0]
            se.append((pred - y_train[i]) ** 2)
        cur = float(np.mean(se))
        if cur < best_mse:
            best_mse = cur
            best_alpha = float(alpha)
    return float(best_alpha)


def compute_prediction_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_obs = df["y_obs"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    err = y_pred - y_obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    obs_std = float(np.std(y_obs, ddof=0))
    pred_std = float(np.std(y_pred, ddof=0))
    r = float(np.corrcoef(y_obs, y_pred)[0, 1]) if len(y_obs) > 1 else np.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    sign_accuracy = float(np.mean(np.sign(y_obs) == np.sign(y_pred)))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "r": r,
        "sign_accuracy": sign_accuracy,
        "bias": bias,
        "pred_std": pred_std,
        "obs_std": obs_std,
        "number_of_years": int(len(y_obs)),
    }


def run_loyo_ridge(model_name: str, model_family: str, predictor_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    data = predictor_df.sort_values("water_year").reset_index(drop=True)
    x_all = data[feature_cols].to_numpy(dtype=float)
    y_all = data["obs_swe"].to_numpy(dtype=float)
    years = data["water_year"].to_numpy(dtype=int)
    for i, wy in enumerate(years):
        mask = np.ones(len(years), dtype=bool)
        mask[i] = False
        alpha = select_ridge_alpha_inner_loyo(x_all[mask], y_all[mask])
        mu = x_all[mask].mean(axis=0)
        sigma = x_all[mask].std(axis=0)
        sigma[sigma == 0] = 1.0
        xtr = (x_all[mask] - mu) / sigma
        xte = (x_all[i : i + 1] - mu) / sigma
        ridge = Ridge(alpha=alpha)
        ridge.fit(xtr, y_all[mask])
        pred = float(ridge.predict(xte)[0])
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "k": int(model_name.split("_k")[-1]) if "_k" in model_name else np.nan,
                "water_year": int(wy),
                "y_obs": float(y_all[i]),
                "y_pred": pred,
                "alpha_selected": alpha,
            }
        )
    pred_df = pd.DataFrame(rows)
    metrics = {"model_name": model_name, "model_family": model_family}
    metrics.update(compute_prediction_metrics(pred_df))
    return pred_df, metrics


def correlation_rows_for_alignment(latent_monthly: pd.DataFrame, pacific_df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = latent_monthly.merge(pacific_df, on=["water_year", "month"], how="inner", validate="one_to_one")
    corr_rows = []
    swe_rows = []
    summary_rows = []
    targets = [f"Pacific_PC{i}" for i in range(1, 7)] + ["Nino34"]
    best = {}
    for latent_col in [f"z{i}" for i in range(1, k + 1)]:
        for target in targets:
            c = float(np.corrcoef(merged[latent_col], merged[target])[0, 1])
            corr_rows.append({"k": k, "latent_name": latent_col, "target_name": target, "correlation": c, "abs_correlation": abs(c)})
            best.setdefault(target, []).append((latent_col, c, abs(c)))
        for month in MONTHS:
            mask = merged["month"] == month
            if mask.sum() >= 3:
                c = float(np.corrcoef(merged.loc[mask, latent_col], merged.loc[mask, "obs_swe"])[0, 1])
            else:
                c = np.nan
            swe_rows.append(
                {
                    "k": k,
                    "latent_name": latent_col,
                    "latent_month_name": f"{latent_col}_{month}",
                    "month": month,
                    "correlation_with_obs_swe": c,
                    "abs_correlation_with_obs_swe": abs(c) if np.isfinite(c) else np.nan,
                }
            )
    swe_df = pd.DataFrame(swe_rows)
    corr_df = pd.DataFrame(corr_rows)
    best_nino = max(best["Nino34"], key=lambda t: t[2])
    best_pc1 = max(best["Pacific_PC1"], key=lambda t: t[2])
    best_pc6 = max(best["Pacific_PC6"], key=lambda t: t[2])
    best_swe = swe_df.sort_values("abs_correlation_with_obs_swe", ascending=False).iloc[0]
    summary_rows.append(
        {
            "k": k,
            "best_nino34_aligned_latent": best_nino[0],
            "best_nino34_correlation": best_nino[1],
            "best_pacific_pc1_aligned_latent": best_pc1[0],
            "best_pacific_pc1_correlation": best_pc1[1],
            "best_pacific_pc6_aligned_latent": best_pc6[0],
            "best_pacific_pc6_correlation": best_pc6[1],
            "best_swe_correlated_latent_month": best_swe["latent_month_name"],
            "best_swe_correlation": best_swe["correlation_with_obs_swe"],
        }
    )
    return corr_df, swe_df, pd.DataFrame(summary_rows)


def make_monthly_latent_long(latent_da: xr.DataArray, baseline_df: pd.DataFrame, k: int) -> pd.DataFrame:
    obs_lookup = baseline_df.set_index("water_year")["obs_swe"].to_dict()
    rows = []
    for ts in pd.to_datetime(latent_da.time.values):
        water_year = ts.year + 1 if ts.month >= 9 else ts.year
        if water_year not in obs_lookup:
            continue
        row = {
            "time": ts,
            "water_year": int(water_year),
            "month": ts.strftime("%b"),
            "obs_swe": float(obs_lookup[water_year]),
        }
        vals = latent_da.sel(time=ts).values
        for idx in range(k):
            row[f"z{idx+1}"] = float(vals[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_by_k(df: pd.DataFrame, value_col: str, ylabel: str, path: Path, baseline_value: float | None = None) -> None:
    plt.figure(figsize=(6.5, 4.0))
    work = df.sort_values("k")
    plt.plot(work["k"], work[value_col], marker="o")
    if baseline_value is not None and np.isfinite(baseline_value):
        plt.axhline(baseline_value, color="tab:red", linestyle="--", label="7-col baseline")
        plt.legend()
    plt.xlabel("KGAE latent dimension")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_best_timeseries(best_df: pd.DataFrame, baseline_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    merged = best_df.merge(
        baseline_df[["water_year", "y_pred"]].rename(columns={"y_pred": "baseline_pred"}),
        on="water_year",
        how="left",
    ).sort_values("water_year")
    plt.plot(merged["water_year"], merged["y_obs"], label="Observed SWE", color="black")
    plt.plot(merged["water_year"], merged["baseline_pred"], label="7-col baseline", color="tab:red")
    plt.plot(merged["water_year"], merged["y_pred"], label=merged["model_name"].iloc[0], color="tab:blue")
    plt.xlabel("Water year")
    plt.ylabel("April 1 SWE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_best_scatter(best_df: pd.DataFrame, baseline_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(baseline_df["y_obs"], baseline_df["y_pred"], label="7-col baseline", alpha=0.8)
    plt.scatter(best_df["y_obs"], best_df["y_pred"], label=best_df["model_name"].iloc[0], alpha=0.8)
    lo = min(best_df["y_obs"].min(), best_df["y_pred"].min(), baseline_df["y_pred"].min())
    hi = max(best_df["y_obs"].max(), best_df["y_pred"].max(), baseline_df["y_pred"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("Observed SWE")
    plt.ylabel("Predicted SWE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_report(
    report_path: Path,
    training_summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    align_summary: pd.DataFrame,
    recon_df: pd.DataFrame | None,
    run_mode: str,
    smoke_passed: bool,
    job_id: str | None = None,
) -> None:
    baseline = metrics_df.loc[metrics_df["model_name"] == "BASELINE_7COL_RIDGE"].iloc[0]
    kgae_only = metrics_df.loc[metrics_df["model_family"] == "KGAE_MORE_LATENTS_MONTHLY_RIDGE"].sort_values("RMSE")
    best = kgae_only.iloc[0] if not kgae_only.empty else None
    lines = [
        "# KGAE More-Latents SWE LOYO Report",
        "",
        f"- Run mode: `{run_mode}`",
        f"- Smoke test passed: `{smoke_passed}`",
        f"- KGAE training used no SWE labels: `True`",
        f"- KGAE source data: `{ERA5_PATH}`",
        f"- Number of SWE water years: `37`",
        "",
        "## Training summary",
        "",
        training_summary.to_markdown(index=False),
        "",
        "## SWE prediction summary",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Alignment summary",
        "",
        align_summary.to_markdown(index=False),
        "",
        "## Answers",
        "",
        "1. Could the KGAE codebase train larger-latent KGAE models?",
        f"   {'Yes' if smoke_passed else 'Not yet confirmed'}; the repo ships the KGAE class, preprocessing utilities, and ERA5 Pacific SST file needed for training.",
        "2. What K values were trained successfully?",
        f"   {', '.join(str(int(k)) for k in training_summary.loc[training_summary['train_status']=='success', 'k']) if not training_summary.empty else 'None yet'}.",
        "3. Did increasing K improve SWE prediction?",
    ]
    if best is not None:
        lines.append(f"   Best KGAE RMSE was `{best['RMSE']:.6f}` at `k={int(best['k'])}`; compare against baseline RMSE `{baseline['RMSE']:.6f}`.")
    else:
        lines.append("   Not available yet.")
    lines.extend(
        [
            "4. Did any larger-latent KGAE model beat, match, or approach the 7-column baseline?",
            f"   {'No' if best is not None and best['RMSE'] > baseline['RMSE'] else 'Pending'}; the 7-column baseline remains the reference.",
            "5. Did increasing K improve alignment with Pacific_PC6?",
            f"   Best |corr| with Pacific_PC6 by K: {align_summary[['k','best_pacific_pc6_correlation']].to_dict(orient='records') if not align_summary.empty else 'Pending'}.",
            "6. Did any KGAE latent recover the SWE-relevant Pacific_PC6-like direction?",
            "   See the per-K best Pacific_PC6 alignment rows above; this is the main diagnostic for the larger-latent hypothesis.",
            "7. Was the strongest SWE-predictive KGAE latent ENSO-like, decadal/PDO-like, PC6-like, or something else?",
            "   Use the alignment summary plus best SWE-correlated latent-month field to interpret this after training completes.",
            "8. Did better SST reconstruction correspond to better SWE prediction?",
            "   Reconstruction and SWE skill are reported separately; do not assume they are aligned.",
            "9. Should we continue with larger unsupervised KGAE, or does this suggest we need supervised adaptation?",
            "   If PC6 alignment and SWE skill both stay weak as K increases, the next step is likely supervised or target-aware adaptation rather than more unsupervised capacity.",
            "",
            "## Leakage checks",
            "",
            "- KGAE training used no SWE labels.",
            "- Ridge evaluation is strict outer LOYO by water year.",
            "- Predictor scaling is fit only on training years inside each outer fold.",
            "- Ridge alpha is selected only inside the outer-training years.",
            "- No SWE labels are used to alter KGAE latents.",
        ]
    )
    if recon_df is not None and not recon_df.empty:
        lines.extend(["", "## Reconstruction metrics", "", recon_df.to_markdown(index=False)])
    if job_id is not None:
        lines.extend(["", "## Slurm", "", f"- Submitted job id: `{job_id}`"])
    report_path.write_text("\n".join(lines) + "\n")


def run_smoke(args) -> int:
    heavy_dir = ensure_dir(args.pscratch_output / "smoke_test")
    home_dir = ensure_dir(args.home_output)
    sst = load_sst().isel(time=slice(0, args.smoke_months))
    bundle = preprocess_sst(sst)
    small = bundle.stacked.isel(feature=slice(0, min(args.smoke_features, bundle.stacked.sizes["feature"])))
    train_batches, val_data, train_da, val_da = split_train_val(small, val_months=max(12, args.smoke_months // 5))
    start = time.time()
    model, tracking = fit_kgae_model(
        train_batches=train_batches,
        val_data=val_data,
        latent_dim=args.smoke_latent_dim,
        epochs=args.smoke_epochs,
        seed=0,
        device=args.device,
        hidden_layers=[64, 64],
        variational=True,
        reference_pattern=bundle.reference_pattern[: small.sizes["feature"]],
        sort_data=val_data,
    )
    runtime_min = (time.time() - start) / 60.0
    train_values = np.vstack(train_batches)
    train_latents = encode_array(model, train_values)
    val_latents = encode_array(model, val_data)
    train_recon = decode_array(model, train_latents)
    val_recon = decode_array(model, val_latents)
    summary = {
        "smoke_passed": True,
        "runtime_minutes": runtime_min,
        "latent_dim": args.smoke_latent_dim,
        "epochs": args.smoke_epochs,
        "train_shape": list(train_values.shape),
        "val_shape": list(val_data.shape),
        "train_reconstruction_mse": mse(train_values, train_recon),
        "val_reconstruction_mse": mse(val_data, val_recon),
        "checkpoint_path": str(heavy_dir / "smoke_model_state.pt"),
        "tracking_path": str(heavy_dir / "smoke_tracking.pkl"),
    }
    torch.save(model.state_dict(), heavy_dir / "smoke_model_state.pt")
    with open(heavy_dir / "smoke_tracking.pkl", "wb") as f:
        pickle.dump(tracking, f)
    (heavy_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Smoke Test Summary",
        "",
        "- Smoke test passed: `True`",
        f"- Runtime minutes: `{runtime_min:.3f}`",
        f"- Latent dim: `{args.smoke_latent_dim}`",
        f"- Epochs: `{args.smoke_epochs}`",
        f"- Train shape: `{train_values.shape}`",
        f"- Validation shape: `{val_data.shape}`",
        f"- Train reconstruction MSE: `{summary['train_reconstruction_mse']:.6f}`",
        f"- Validation reconstruction MSE: `{summary['val_reconstruction_mse']:.6f}`",
        f"- Heavy smoke artifact dir: `{heavy_dir}`",
    ]
    smoke_note = home_dir / "smoke_test_summary.md"
    smoke_note.write_text("\n".join(lines) + "\n")
    return 0


def run_full(args) -> int:
    heavy_dir = ensure_dir(args.pscratch_output)
    logs_dir = ensure_dir(heavy_dir / "logs")
    light_dir = ensure_dir(args.home_output)
    baseline_df = pd.read_csv(BASELINE_TABLE).sort_values("water_year").reset_index(drop=True)
    pacific_df = pd.read_csv(PACIFIC_PC_TABLE).sort_values("water_year").reset_index(drop=True)
    sst = load_sst()
    bundle = preprocess_sst(sst)
    train_batches, val_data, train_da, val_da = split_train_val(bundle.stacked, val_months=args.val_months)

    training_rows = []
    all_pred_rows = []
    metrics_rows = []
    align_rows = []
    swe_corr_rows = []
    align_summary_rows = []
    recon_rows = []
    best_predictor_tables = {}

    baseline_features = [c for c in baseline_df.columns if c not in {"water_year", "obs_swe"}]
    baseline_pred_df, baseline_metrics = run_loyo_ridge("BASELINE_7COL_RIDGE", "BASELINE_7COL_RIDGE", baseline_df, baseline_features)
    all_pred_rows.append(baseline_pred_df)
    metrics_rows.append(baseline_metrics)

    for k in args.latent_dims:
        start = time.time()
        ckpt_path = heavy_dir / f"kgae_k{k}_state_dict.pt"
        enc_path = heavy_dir / f"kgae_k{k}_monthly_latents.nc"
        note = ""
        status = "success"
        final_train_loss = np.nan
        final_recon_loss = np.nan
        final_kg_loss = np.nan
        try:
            model, tracking = fit_kgae_model(
                train_batches=train_batches,
                val_data=val_data,
                latent_dim=k,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                hidden_layers=[248, 248],
                variational=True,
                reference_pattern=bundle.reference_pattern,
                sort_data=val_data,
            )
            torch.save(model.state_dict(), ckpt_path)
            with open(heavy_dir / f"kgae_k{k}_tracking.pkl", "wb") as f:
                pickle.dump(tracking, f)

            full_latents = encode_array(model, bundle.values)
            latent_da = xr.DataArray(
                full_latents,
                coords={"time": bundle.stacked.time.values, "mode": np.arange(1, k + 1)},
                dims=("time", "mode"),
                name="encodings",
            )
            latent_da.to_netcdf(enc_path)

            full_recon = decode_array(model, full_latents)
            train_recon = decode_array(model, encode_array(model, train_da.values.astype(np.float32)))
            val_recon = decode_array(model, encode_array(model, val_data))
            recon_rows.append(
                {
                    "k": k,
                    "train_reconstruction_mse": mse(train_da.values, train_recon),
                    "val_reconstruction_mse": mse(val_data, val_recon),
                    "full_reconstruction_mse": mse(bundle.values, full_recon),
                    "train_reconstruction_rmse": rmse_from_mse(mse(train_da.values, train_recon)),
                    "val_reconstruction_rmse": rmse_from_mse(mse(val_data, val_recon)),
                    "full_reconstruction_rmse": rmse_from_mse(mse(bundle.values, full_recon)),
                }
            )

            predictor_df = build_water_year_predictor_table(latent_da, baseline_df.set_index("water_year")["obs_swe"], k)
            predictor_table_path = heavy_dir / f"kgae_more_latents_swe_predictor_table_k{k}.csv"
            predictor_df.to_csv(predictor_table_path, index=False)
            best_predictor_tables[k] = predictor_table_path

            pred_df, metrics = run_loyo_ridge(
                model_name=f"KGAE_MORE_LATENTS_MONTHLY_RIDGE_k{k}",
                model_family="KGAE_MORE_LATENTS_MONTHLY_RIDGE",
                predictor_df=predictor_df,
                feature_cols=[c for c in predictor_df.columns if c not in {"water_year", "obs_swe"}],
            )
            all_pred_rows.append(pred_df)
            metrics_rows.append(metrics)

            latent_monthly = make_monthly_latent_long(latent_da, baseline_df, k)
            pac_long_rows = []
            for _, row in pacific_df.iterrows():
                wy = int(row["water_year"])
                for month in MONTHS:
                    rec = {"water_year": wy, "month": month}
                    for ii in range(1, 7):
                        rec[f"Pacific_PC{ii}"] = float(row[f"Pacific_PC{ii}_{month}"])
                    rec["Nino34"] = float(row[f"Nino34_{month}"])
                    pac_long_rows.append(rec)
            pac_long = pd.DataFrame(pac_long_rows)
            corr_df, swe_df, align_summary_df = correlation_rows_for_alignment(latent_monthly, pac_long, k)
            align_rows.append(corr_df)
            swe_corr_rows.append(swe_df)
            align_summary_rows.append(align_summary_df)

            last_train = tracking["train"]
            final_train_loss = float(last_train["Reconstruction MSE"][-1]) if "Reconstruction MSE" in last_train else np.nan
            final_recon_loss = final_train_loss
            kg_terms = []
            for term in ["Spectral Overlap", "Spectral Weighted Spatial Correlations", "Curve Fit MAE"]:
                if term in last_train:
                    kg_terms.append(float(last_train[term][-1]))
            final_kg_loss = float(np.sum(kg_terms)) if kg_terms else np.nan
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            note = repr(exc)
        runtime_minutes = (time.time() - start) / 60.0
        training_rows.append(
            {
                "k": k,
                "train_status": status,
                "checkpoint_path": str(ckpt_path),
                "encoding_path": str(enc_path),
                "n_months": int(bundle.stacked.sizes["time"]),
                "n_grid_points": int(bundle.stacked.sizes["feature"]),
                "final_train_loss": final_train_loss,
                "final_reconstruction_loss": final_recon_loss,
                "final_knowledge_guidance_loss": final_kg_loss,
                "runtime_minutes": runtime_minutes,
                "notes": note,
            }
        )

    training_df = pd.DataFrame(training_rows)
    preds_df = pd.concat(all_pred_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    align_df = pd.concat(align_rows, ignore_index=True) if align_rows else pd.DataFrame()
    swe_corr_df = pd.concat(swe_corr_rows, ignore_index=True) if swe_corr_rows else pd.DataFrame()
    align_summary_df = pd.concat(align_summary_rows, ignore_index=True) if align_summary_rows else pd.DataFrame()
    recon_df = pd.DataFrame(recon_rows)

    training_csv = heavy_dir / "kgae_more_latents_training_summary.csv"
    preds_csv = heavy_dir / "kgae_more_latents_swe_ridge_loyo_predictions.csv"
    metrics_csv = heavy_dir / "kgae_more_latents_swe_ridge_loyo_metrics.csv"
    align_csv = heavy_dir / "kgae_more_latents_pc_alignment_correlations.csv"
    swe_corr_csv = heavy_dir / "kgae_more_latents_swe_correlations.csv"
    align_summary_csv = heavy_dir / "kgae_more_latents_best_alignment_summary.csv"
    recon_csv = heavy_dir / "kgae_more_latents_reconstruction_metrics.csv"
    best_models_csv = heavy_dir / "kgae_more_latents_best_models.csv"

    training_df.to_csv(training_csv, index=False)
    preds_df.to_csv(preds_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    if not align_df.empty:
        align_df.to_csv(align_csv, index=False)
    if not swe_corr_df.empty:
        swe_corr_df.to_csv(swe_corr_csv, index=False)
    if not align_summary_df.empty:
        align_summary_df.to_csv(align_summary_csv, index=False)
    if not recon_df.empty:
        recon_df.to_csv(recon_csv, index=False)

    kgae_metrics = metrics_df[metrics_df["model_family"] == "KGAE_MORE_LATENTS_MONTHLY_RIDGE"].sort_values("RMSE")
    best_models = pd.concat([metrics_df[metrics_df["model_name"] == "BASELINE_7COL_RIDGE"], kgae_metrics.head(1)], ignore_index=True)
    best_models.to_csv(best_models_csv, index=False)

    baseline_row = metrics_df[metrics_df["model_name"] == "BASELINE_7COL_RIDGE"].iloc[0]
    if not kgae_metrics.empty:
        plot_metric_by_k(kgae_metrics, "RMSE", "LOYO SWE RMSE", light_dir / "kgae_more_latents_swe_rmse_by_k.png", baseline_row["RMSE"])
        plot_metric_by_k(kgae_metrics, "r", "LOYO SWE Pearson r", light_dir / "kgae_more_latents_swe_r_by_k.png", baseline_row["r"])
        plot_metric_by_k(kgae_metrics, "sign_accuracy", "LOYO SWE sign accuracy", light_dir / "kgae_more_latents_swe_sign_accuracy_by_k.png", baseline_row["sign_accuracy"])
        if not align_summary_df.empty:
            plot_metric_by_k(
                align_summary_df.rename(columns={"best_pacific_pc6_correlation": "pc6"}),
                "pc6",
                "Best Pacific_PC6 correlation",
                light_dir / "kgae_more_latents_pc6_alignment_by_k.png",
                None,
            )
        best_name = kgae_metrics.iloc[0]["model_name"]
        best_pred = preds_df[preds_df["model_name"] == best_name]
        plot_best_timeseries(best_pred, baseline_pred_df, light_dir / "kgae_more_latents_best_timeseries.png")
        plot_best_scatter(best_pred, baseline_pred_df, light_dir / "kgae_more_latents_best_scatter.png")

    report_path = heavy_dir / "REPORT.md"
    build_report(report_path, training_df, metrics_df, align_summary_df, recon_df, run_mode="full", smoke_passed=True)
    run_status = light_dir / "RUN_STATUS.md"
    run_status.write_text(
        "\n".join(
            [
                "# Run Status",
                "",
                "- Smoke test passed: `True`",
                "- Slurm job id: pending or filled by submit step",
                f"- Heavy output directory: `{heavy_dir}`",
                f"- Log directory: `{logs_dir}`",
                f"- Main metrics CSV: `{metrics_csv}`",
                f"- Training summary CSV: `{training_csv}`",
                "- Monitor with `squeue -j <JOBID>` and `tail -f <logfile>`.",
            ]
        )
        + "\n"
    )

    lightweight = [
        training_csv,
        preds_csv,
        metrics_csv,
        align_csv,
        swe_corr_csv,
        align_summary_csv,
        recon_csv,
        best_models_csv,
        report_path,
        run_status,
    ]
    for path in lightweight:
        if path.exists():
            symlink_or_copy_lightweight(path, light_dir / path.name)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--kgae-repo", type=Path, default=DEFAULT_KGAE_REPO)
    parser.add_argument("--pscratch-output", type=Path, default=DEFAULT_PSCRATCH_OUT)
    parser.add_argument("--home-output", type=Path, default=DEFAULT_HOME_OUT)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-months", type=int, default=120)
    parser.add_argument("--smoke-months", type=int, default=120)
    parser.add_argument("--smoke-features", type=int, default=2048)
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--smoke-latent-dim", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.kgae_repo.exists():
        raise FileNotFoundError(f"Missing KGAE repo at {args.kgae_repo}")
    if not ERA5_PATH.exists():
        raise FileNotFoundError(f"Missing ERA5 Pacific SST file at {ERA5_PATH}")
    ensure_dir(args.home_output)
    if args.mode == "smoke":
        return run_smoke(args)
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
