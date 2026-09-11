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
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import xarray as xr
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_KGAE_REPO = PROJECT_ROOT / "artifacts" / "kgae_pacific_latent_swe_loyo" / "external" / "kgae"
DEFAULT_HOME_OUT = PROJECT_ROOT / "artifacts" / "atlantic_kgae"
DEFAULT_PSCRATCH_OUT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/atlantic_kgae")
DEFAULT_LOG_DIR = DEFAULT_PSCRATCH_OUT / "logs"
COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
ATLANTIC_EOF_DATASET = (
    Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/swe_climate_mode_baseline/amv_amo")
    / "amv_amo_cobe2_north_atlantic_eofs_pc1to6.nc"
)
ATLANTIC_EOF_SUMMARY = (
    Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/swe_climate_mode_baseline/amv_amo")
    / "amv_amo_cobe2_north_atlantic_pc1to6_summary.json"
)
AQM_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "aqm_index_loyo"
ATTRIBUTION_DIR = PROJECT_ROOT / "artifacts" / "atlantic_pc245_attribution"
RIDGE_REFERENCE_DIR = (
    PROJECT_ROOT / "artifacts" / "cobe2_sierra_swe_lod_setup" / "z1z2_plus_amv_k5_loyo"
)
RIDGE_REFERENCE_TABLE = RIDGE_REFERENCE_DIR / "z1z2_amv_k5_predictor_table.csv"
RIDGE_REFERENCE_METRICS = RIDGE_REFERENCE_DIR / "z1z2_amv_k5_loyo_metrics.csv"
AQM_METRICS = AQM_ARTIFACT_DIR / "aqm_seasonal_loyo_metrics.csv"
AQM_PERIOD_METRICS = AQM_ARTIFACT_DIR / "aqm_seasonal_loyo_period_metrics.csv"
AQM_INDEX_FULL = AQM_ARTIFACT_DIR / "aqm_index_full_record.csv"
REFERENCE_PATTERNS_NC = ATTRIBUTION_DIR / "reference_patterns_harmonized.nc"
TARGET_EOF_PATTERNS_NC = ATTRIBUTION_DIR / "target_eof_patterns.nc"
REFERENCE_INDICES_CSV = ATTRIBUTION_DIR / "reference_indices_monthly_and_seasonal.csv"
PC245_TEMPORAL_CSV = ATTRIBUTION_DIR / "pc245_temporal_similarity.csv"

K_VALUES = [3, 5, 7, 10]
MONTH_ORDER = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_TO_NUMBER = {"Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
RIDGE_ALPHA_GRID = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
NEAR_ZERO = 1.0e-12
LAT_MIN = 0.0
LAT_MAX = 70.0
LON_MIN_360 = 280.0
LON_MAX_360 = 360.0
PRIMARY_SEED = 0
DEFAULT_EPOCHS = 120
DEFAULT_SMOKE_EPOCHS = 3
DEFAULT_BATCH_SIZE_SPLITS = 4
DEFAULT_HIDDEN_LAYERS = [248, 248]
LOSS_WEIGHTS = {
    "KL Divergence": 1.0e-2,
    "Reconstruction MSE": 1.0,
    "Spectral Overlap": 1.0,
    "Spatial Correlations": 0.0,
    "Spectral Weighted Spatial Correlations": 1.0,
    "Curve Fit MAE": 1.0,
    "L1 Decoder Reg": 1.0e-4,
    "Centering Loss": 1.0,
}
LOSSES_TO_USE = [
    "KL Divergence",
    "Reconstruction MSE",
    "Spectral Overlap",
    "Spectral Weighted Spatial Correlations",
    "Curve Fit MAE",
    "L1 Decoder Reg",
    "Centering Loss",
]
PERIOD_SPECS = [
    ("all_years", lambda wy: np.isfinite(wy)),
    ("pre_2010", lambda wy: wy <= 2010),
    ("post_2010", lambda wy: wy > 2010),
    ("pre_2005", lambda wy: wy <= 2005),
    ("post_2005", lambda wy: wy > 2005),
]


sys.path.insert(0, str(DEFAULT_KGAE_REPO))
import kgae  # noqa: E402


@dataclass
class SplitYears:
    train_years: List[int]
    val_years: List[int]
    test_years: List[int]


@dataclass
class PreparedSST:
    raw_full: xr.DataArray
    train_anom: xr.DataArray
    val_anom: xr.DataArray
    test_anom: xr.DataArray
    full_anom: xr.DataArray
    train_stacked: xr.DataArray
    val_stacked: xr.DataArray
    test_stacked: xr.DataArray
    full_stacked: xr.DataArray
    valid_mask: xr.DataArray
    pfit: xr.Dataset
    monthly_clim: xr.DataArray
    split_years: SplitYears


@dataclass
class RidgeResult:
    predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    metrics: pd.DataFrame
    period_metrics: pd.DataFrame


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_compute_runtime() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside a Perlmutter interactive or batch compute-node allocation.")


def normalize_longitudes_360(da: xr.DataArray) -> xr.DataArray:
    lon360 = np.mod(da["lon"].values.astype(float), 360.0)
    order = np.argsort(lon360)
    da = da.isel(lon=order).assign_coords(lon=lon360[order])
    _, unique_idx = np.unique(np.round(da["lon"].values.astype(float), 8), return_index=True)
    return da.isel(lon=np.sort(unique_idx))


def latitude_slice(da: xr.DataArray, lat_min: float, lat_max: float) -> xr.DataArray:
    if float(da["lat"].values[0]) > float(da["lat"].values[-1]):
        return da.sel(lat=slice(lat_max, lat_min))
    return da.sel(lat=slice(lat_min, lat_max))


def load_cobe2_atlantic_sst() -> xr.DataArray:
    ds = xr.open_dataset(COBE2_SST_FILE)
    da = ds["sst"]
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    da = normalize_longitudes_360(da)
    da = latitude_slice(da.sel(lon=slice(LON_MIN_360, LON_MAX_360 - 1.0e-8)), LAT_MIN, LAT_MAX)
    da = da.where(np.isfinite(da))
    return da.sortby("time")


def open_reference_valid_mask() -> xr.DataArray:
    ds = xr.open_dataset(ATLANTIC_EOF_DATASET)
    mask = ds["valid_mask"].astype(bool)
    mask = mask.sel(lat=latitude_slice(mask, LAT_MIN, LAT_MAX)["lat"], lon=slice(LON_MIN_360, LON_MAX_360 - 1.0e-8))
    return mask


def split_calendar_years(all_years: Sequence[int], smoke: bool) -> SplitYears:
    years = list(sorted(int(v) for v in all_years))
    if smoke:
        smoke_years = years[:42]
        return SplitYears(
            train_years=smoke_years[:30],
            val_years=smoke_years[30:36],
            test_years=smoke_years[36:42],
        )
    n_years = len(years)
    n_train = int(math.floor(0.70 * n_years))
    n_val = int(math.floor(0.15 * n_years))
    train_years = years[:n_train]
    val_years = years[n_train : n_train + n_val]
    test_years = years[n_train + n_val :]
    return SplitYears(train_years=train_years, val_years=val_years, test_years=test_years)


def subset_years(da: xr.DataArray, years: Sequence[int]) -> xr.DataArray:
    years = set(int(v) for v in years)
    mask = np.asarray([pd.Timestamp(v).year in years for v in da["time"].values], dtype=bool)
    return da.isel(time=np.where(mask)[0])


def apply_training_transform(data: xr.DataArray, pfit: xr.Dataset, monthly_clim: xr.DataArray) -> xr.DataArray:
    detrended = data - xr.polyval(data["time"], pfit.polyfit_coefficients)
    pieces = []
    for year in sorted({int(pd.Timestamp(v).year) for v in detrended["time"].values}):
        yearly = detrended.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if yearly.sizes.get("time", 0) == 0:
            continue
        monthly = yearly.groupby("time.month").mean() - monthly_clim
        monthly = monthly.assign_coords(month=[pd.Timestamp(year=year, month=int(m), day=1) for m in monthly["month"].values])
        monthly = monthly.rename({"month": "time"})
        pieces.append(monthly)
    if not pieces:
        raise RuntimeError("No transformed yearly pieces were produced.")
    return xr.concat(pieces, dim="time").sortby("time")


def stack_with_mask(da: xr.DataArray, valid_mask: xr.DataArray) -> xr.DataArray:
    masked = da.where(valid_mask)
    stacked = masked.stack(feature=("lat", "lon")).transpose("time", "feature")
    stacked = stacked.dropna("feature", how="any")
    return stacked.astype(np.float32)


def prepare_atlantic_sst(smoke: bool) -> PreparedSST:
    raw_full = load_cobe2_atlantic_sst()
    valid_mask = open_reference_valid_mask()
    split_years = split_calendar_years(np.unique(raw_full["time"].dt.year.values), smoke=smoke)
    train_raw = subset_years(raw_full, split_years.train_years)
    val_raw = subset_years(raw_full, split_years.val_years)
    test_raw = subset_years(raw_full, split_years.test_years)

    train_anom, _, _, pfit = kgae.global_detrend(train_raw, deg=2)
    train_anom, monthly_clim = kgae.remove_climo(train_anom)
    val_anom = apply_training_transform(val_raw, pfit, monthly_clim)
    test_anom = apply_training_transform(test_raw, pfit, monthly_clim)
    full_anom = apply_training_transform(raw_full, pfit, monthly_clim)

    return PreparedSST(
        raw_full=raw_full,
        train_anom=train_anom,
        val_anom=val_anom,
        test_anom=test_anom,
        full_anom=full_anom,
        train_stacked=stack_with_mask(train_anom, valid_mask),
        val_stacked=stack_with_mask(val_anom, valid_mask),
        test_stacked=stack_with_mask(test_anom, valid_mask),
        full_stacked=stack_with_mask(full_anom, valid_mask),
        valid_mask=valid_mask,
        pfit=pfit,
        monthly_clim=monthly_clim,
        split_years=split_years,
    )


def build_training_batches(train_values: np.ndarray, n_splits: int = DEFAULT_BATCH_SIZE_SPLITS) -> List[np.ndarray]:
    return [train_values[j::n_splits].astype(np.float32) for j in range(n_splits)]


def reference_sign_pattern(valid_feature_count: int) -> np.ndarray:
    ref = np.ones(valid_feature_count, dtype=np.float32)
    return ref / np.linalg.norm(ref)


def encode_latents(model: kgae.KGAE, values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        encoded = model.encoder(torch.tensor(values, dtype=torch.float32, device=model.device)).cpu().numpy()
    if model.is_variational:
        encoded = encoded[:, : model.latent_dim]
    return encoded * model.flip_signs.values.reshape(1, -1)


def decode_latents(model: kgae.KGAE, latents: np.ndarray, apply_signs: bool = True) -> np.ndarray:
    inputs = latents.copy()
    if apply_signs:
        inputs = inputs * model.flip_signs.values.reshape(1, -1)
    with torch.no_grad():
        decoded = model.decoder(torch.tensor(inputs, dtype=torch.float32, device=model.device)).cpu().numpy()
    return decoded


def unit_probe_patterns(model: kgae.KGAE, template: xr.DataArray) -> xr.DataArray:
    eye = np.eye(model.latent_dim, dtype=np.float32)
    decoded = decode_latents(model, eye, apply_signs=True)
    probes = template.isel(time=slice(0, model.latent_dim)).copy(data=decoded)
    probes = probes.rename({"time": "mode"}).assign_coords(mode=np.arange(1, model.latent_dim + 1))
    probes = probes.unstack("feature").sortby(["mode", "lat", "lon"])
    return probes


def weighted_total_loss(losses: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in LOSSES_TO_USE:
        if key in losses:
            total += float(LOSS_WEIGHTS[key] * losses[key].item())
    return total


def fit_kgae_with_checkpoint(
    train_batches: List[np.ndarray],
    val_data: np.ndarray,
    latent_dim: int,
    num_epochs: int,
    seed: int,
    device: str,
    out_checkpoint: Path,
) -> Tuple[kgae.KGAE, Dict[str, Dict[str, List[float]]], Dict[str, object]]:
    model = kgae.KGAE(
        input_dim=train_batches[0].shape[1],
        latent_dim=latent_dim,
        is_variational=True,
        hidden_layers=DEFAULT_HIDDEN_LAYERS,
        activation=torch.nn.Tanh,
        power_spectrum_smoothing_kernel=7,
        device=device,
        seed=seed,
        tag=f"atlantic_k{latent_dim}",
    )

    train_loader = [torch.tensor(batch, dtype=torch.float32).to(model.device) for batch in train_batches]
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    ema = kgae.EMA(model, decay=0.995)
    minibatch_ndxs = np.arange(len(train_loader))
    tracking: Dict[str, Dict[str, List[float]]] = {"train": {}, "val": {}}
    best_state = None
    best_epoch = None
    best_val_selection = float("inf")
    best_val_reconstruction = float("inf")

    for epoch in range(num_epochs):
        start = pd.Timestamp.now()
        model.train()
        train_losses: Dict[str, torch.Tensor] = {}
        np.random.shuffle(minibatch_ndxs)
        total_samples = 0
        for minibatch_idx in minibatch_ndxs:
            optimizer.zero_grad()
            batch = train_loader[minibatch_idx]
            batch_n = int(batch.shape[0])
            total_samples += batch_n
            salient_variances = model.compute_salient_variance(batch)
            optimizer.zero_grad()
            losses = model.compute_loss(batch, salient_variances)
            for key, value in losses.items():
                weighted = batch_n * value
                if key not in train_losses:
                    train_losses[key] = weighted
                else:
                    train_losses[key] = train_losses[key] + weighted
            total_loss = 0.0
            for key in LOSSES_TO_USE:
                if key in losses:
                    total_loss = total_loss + LOSS_WEIGHTS[key] * losses[key]
            total_loss.backward()
            optimizer.step()
            ema.update(model)
        for key in list(train_losses.keys()):
            train_losses[key] = train_losses[key] / total_samples

        model.eval()
        ema.apply_shadow(model)
        model.zero_grad()
        val_salient_variances = model.compute_salient_variance(val_tensor)
        optimizer.zero_grad()
        val_losses = model.compute_loss(val_tensor, val_salient_variances)
        val_selection = weighted_total_loss(val_losses)
        val_reconstruction = float(val_losses["Reconstruction MSE"].item())
        if (
            val_selection < best_val_selection - 1.0e-15
            or (
                abs(val_selection - best_val_selection) <= 1.0e-15
                and val_reconstruction < best_val_reconstruction - 1.0e-15
            )
        ):
            best_val_selection = val_selection
            best_val_reconstruction = val_reconstruction
            best_epoch = int(epoch)
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            torch.save(best_state, out_checkpoint)

        for key, value in train_losses.items():
            tracking["train"].setdefault(key, []).append(float(value.item()))
        for key, value in val_losses.items():
            tracking["val"].setdefault(key, []).append(float(value.item()))
        tracking["train"].setdefault("Selection Objective", []).append(
            float(sum(LOSS_WEIGHTS.get(key, 0.0) * tracking["train"].get(key, [0.0])[-1] for key in LOSSES_TO_USE if key in tracking["train"]))
        )
        tracking["val"].setdefault("Selection Objective", []).append(val_selection)
        end = pd.Timestamp.now()
        print(f"[k={latent_dim}] epoch={epoch} elapsed={end - start} val_selection={val_selection:.6f}")
        ema.restore(model)

    if best_state is None:
        raise RuntimeError("No best checkpoint was recorded.")
    model.load_state_dict(best_state)
    metadata = {
        "best_epoch": best_epoch,
        "best_val_selection_objective": best_val_selection,
        "best_val_reconstruction_mse": best_val_reconstruction,
        "selection_rule": "minimum validation weighted objective using the published loss terms and weights",
        "optimizer": "Adam(lr=1e-3)",
        "ema_decay": 0.995,
        "early_stopping_used": False,
        "checkpoint_selection_used": True,
    }
    return model, tracking, metadata


def area_weights_from_mask(mask: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(mask["lat"]))
    w = xr.DataArray(weights, coords={"lat": mask["lat"]}, dims=("lat",))
    return w.broadcast_like(mask).where(mask, 0.0)


def weighted_spatial_corr(obs_2d: np.ndarray, rec_2d: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(obs_2d) & np.isfinite(rec_2d) & np.isfinite(weights) & (weights > 0.0)
    if valid.sum() < 2:
        return float("nan")
    x = obs_2d[valid]
    y = rec_2d[valid]
    w = weights[valid]
    x_mean = np.sum(w * x) / np.sum(w)
    y_mean = np.sum(w * y) / np.sum(w)
    x0 = x - x_mean
    y0 = y - y_mean
    denom = math.sqrt(float(np.sum(w * x0 * x0) * np.sum(w * y0 * y0)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(w * x0 * y0) / denom)


def temporal_corr_by_grid(obs: np.ndarray, rec: np.ndarray) -> np.ndarray:
    n_feature = obs.shape[1]
    corr = np.full(n_feature, np.nan, dtype=float)
    for j in range(n_feature):
        x = obs[:, j]
        y = rec[:, j]
        if np.std(x, ddof=1) <= 0.0 or np.std(y, ddof=1) <= 0.0:
            continue
        corr[j] = float(np.corrcoef(x, y)[0, 1])
    return corr


def reconstruction_metrics(
    obs: np.ndarray,
    rec: np.ndarray,
    template: xr.DataArray,
    mask_weights: np.ndarray,
    split_name: str,
    model_name: str,
    k: int,
) -> Dict[str, float]:
    residual = rec - obs
    flat_weights = np.broadcast_to(mask_weights.reshape(1, -1), obs.shape)
    mse = float(np.sum(flat_weights * residual * residual) / np.sum(flat_weights))
    rmse = float(math.sqrt(mse))
    mae = float(np.sum(flat_weights * np.abs(residual)) / np.sum(flat_weights))
    spatial_corrs = []
    for idx in range(obs.shape[0]):
        spatial_corrs.append(weighted_spatial_corr(obs[idx], rec[idx], mask_weights))
    temporal_corr = temporal_corr_by_grid(obs, rec)
    valid_temporal = np.isfinite(temporal_corr)
    if np.any(valid_temporal):
        temp_corr_weighted = float(
            np.sum(mask_weights[valid_temporal] * temporal_corr[valid_temporal]) / np.sum(mask_weights[valid_temporal])
        )
    else:
        temp_corr_weighted = float("nan")
    obs_var = float(np.sum(flat_weights * (obs - np.average(obs, weights=flat_weights)) ** 2) / np.sum(flat_weights))
    residual_var = float(np.sum(flat_weights * residual * residual) / np.sum(flat_weights))
    explained_var = float("nan") if obs_var <= 0.0 else float(1.0 - residual_var / obs_var)
    mean_spatial_corr = float(np.nanmean(spatial_corrs)) if np.any(np.isfinite(spatial_corrs)) else float("nan")
    return {
        "model_name": model_name,
        "k": int(k),
        "split_name": split_name,
        "area_weighted_mse": mse,
        "area_weighted_rmse": rmse,
        "area_weighted_mae": mae,
        "mean_monthly_spatial_pattern_correlation": mean_spatial_corr,
        "area_weighted_mean_gridcell_temporal_correlation": temp_corr_weighted,
        "explained_reconstruction_variance": explained_var,
        "n_months": int(obs.shape[0]),
        "n_features": int(obs.shape[1]),
    }


def fit_pca_and_reconstruct(
    train_values: np.ndarray,
    val_values: np.ndarray,
    test_values: np.ndarray,
    full_values: np.ndarray,
    k: int,
) -> Dict[str, np.ndarray]:
    model = PCA(n_components=k, svd_solver="full", random_state=PRIMARY_SEED)
    model.fit(train_values)
    return {
        "train_latents": model.transform(train_values),
        "val_latents": model.transform(val_values),
        "test_latents": model.transform(test_values),
        "full_latents": model.transform(full_values),
        "train_recon": model.inverse_transform(model.transform(train_values)),
        "val_recon": model.inverse_transform(model.transform(val_values)),
        "test_recon": model.inverse_transform(model.transform(test_values)),
        "full_recon": model.inverse_transform(model.transform(full_values)),
        "components": model.components_,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "train_mean": model.mean_,
    }


def obs_swe_series() -> pd.Series:
    df = pd.read_csv(RIDGE_REFERENCE_TABLE).sort_values("water_year").reset_index(drop=True)
    return df.set_index("water_year")["obs_swe"]


def build_kgae_predictor_table(latent_da: xr.DataArray, obs_swe: pd.Series, k: int) -> pd.DataFrame:
    rows = []
    time_index = pd.DatetimeIndex(pd.to_datetime(latent_da["time"].values))
    for water_year, swe in obs_swe.items():
        row = {"water_year": int(water_year), "observed_swe": float(swe)}
        ok = True
        for month in MONTH_ORDER:
            year = int(water_year - 1) if month in {"Sep", "Oct", "Nov", "Dec"} else int(water_year)
            ts = pd.Timestamp(year=year, month=MONTH_TO_NUMBER[month], day=1)
            if ts not in time_index:
                ok = False
                break
            values = latent_da.sel(time=ts).values
            for idx in range(k):
                row[f"KGAE_Z{idx+1}_{month}"] = float(values[idx])
        if ok:
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("water_year").reset_index(drop=True)
    if df["water_year"].tolist() != list(obs_swe.index.astype(int)):
        raise RuntimeError("KGAE SWE predictor table is missing or reordering water years.")
    return df


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx, ddof=1) <= 0.0 or np.std(yy, ddof=1) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yy = y_true[mask]
    pp = y_pred[mask]
    if yy.size < 2:
        return float("nan")
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_sign_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs != 0.0) & (pred != 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(obs[valid]) == np.sign(pred[valid])))


def compute_metric_bundle(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - obs
    return {
        "r": corrcoef_safe(obs, pred),
        "R2": r2_manual(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "sign_accuracy": compute_sign_accuracy(obs, pred),
        "mean_error": float(np.mean(error)),
        "median_abs_error": float(np.median(np.abs(error))),
    }


def standardize_train_only(
    x_train_raw: np.ndarray, x_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        x_std = np.where(~np.isfinite(x_std) | (x_std <= 0.0), 1.0, x_std)
    return (
        (x_train_raw - x_mean[None, :]) / x_std[None, :],
        (x_test_raw - x_mean) / x_std,
        x_mean,
        x_std,
    )


def standardize_target_train_only(y_train_raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0.0:
        y_std = 1.0
    return (y_train_raw - y_mean) / y_std, y_mean, y_std


def fit_ridge_standardized(x_train_std: np.ndarray, y_train_std: np.ndarray, alpha: float) -> np.ndarray:
    gram = x_train_std.T @ x_train_std
    rhs = x_train_std.T @ y_train_std
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)


def inner_loyo_best_alpha(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> Tuple[float, float]:
    n_train = x_train_raw.shape[0]
    best_alpha = None
    best_mse = None
    for alpha in RIDGE_ALPHA_GRID.tolist():
        preds = np.full(n_train, np.nan, dtype=float)
        for inner_idx in range(n_train):
            inner_mask = np.ones(n_train, dtype=bool)
            inner_mask[inner_idx] = False
            x_inner_train = x_train_raw[inner_mask, :]
            x_inner_test = x_train_raw[~inner_mask, :][0]
            y_inner_train = y_train_raw[inner_mask]
            y_inner_test = float(y_train_raw[~inner_mask][0])
            x_inner_train_std, x_inner_test_std, _, _ = standardize_train_only(x_inner_train, x_inner_test)
            y_inner_train_std, y_mean, y_std = standardize_target_train_only(y_inner_train)
            beta_std = fit_ridge_standardized(x_inner_train_std, y_inner_train_std, alpha)
            pred_std = float(x_inner_test_std @ beta_std)
            preds[inner_idx] = y_mean + y_std * pred_std
        mse = float(np.mean((preds - y_train_raw) ** 2))
        if best_mse is None or mse < best_mse - 1.0e-15 or (abs(mse - best_mse) <= 1.0e-15 and alpha < best_alpha):
            best_alpha = float(alpha)
            best_mse = mse
    assert best_alpha is not None and best_mse is not None
    return best_alpha, best_mse


def run_reference_style_loyo(predictor_df: pd.DataFrame, feature_cols: Sequence[str]) -> RidgeResult:
    data = predictor_df.sort_values("water_year").reset_index(drop=True)
    x_all = data[list(feature_cols)].to_numpy(dtype=float)
    y_all = data["observed_swe"].to_numpy(dtype=float)
    years = data["water_year"].to_numpy(dtype=int)
    pred_rows = []
    alpha_rows = []
    for idx, heldout_wy in enumerate(years.tolist()):
        mask = np.ones(len(years), dtype=bool)
        mask[idx] = False
        x_train_raw = x_all[mask, :]
        x_test_raw = x_all[~mask, :][0]
        y_train_raw = y_all[mask]
        y_test_raw = float(y_all[~mask][0])
        alpha, inner_cv_mse = inner_loyo_best_alpha(x_train_raw, y_train_raw)
        x_train_std, x_test_std, x_mean, x_std = standardize_train_only(x_train_raw, x_test_raw)
        y_train_std, y_mean, y_std = standardize_target_train_only(y_train_raw)
        beta_std = fit_ridge_standardized(x_train_std, y_train_std, alpha)
        pred_std = float(x_test_std @ beta_std)
        pred_raw = y_mean + y_std * pred_std
        beta_raw = (y_std / x_std) * beta_std
        intercept_raw = float(y_mean - np.sum(beta_raw * x_mean))
        pred_rows.append(
            {
                "water_year": int(heldout_wy),
                "observed_swe": y_test_raw,
                "predicted_swe": pred_raw,
                "residual": pred_raw - y_test_raw,
                "selected_ridge_alpha": alpha,
            }
        )
        alpha_rows.append(
            {
                "water_year": int(heldout_wy),
                "selected_ridge_alpha": alpha,
                "inner_cv_mse": inner_cv_mse,
                "intercept": intercept_raw,
                "n_features": int(len(feature_cols)),
            }
        )

    predictions = pd.DataFrame(pred_rows).sort_values("water_year").reset_index(drop=True)
    metrics_dict = compute_metric_bundle(
        predictions["observed_swe"].to_numpy(dtype=float),
        predictions["predicted_swe"].to_numpy(dtype=float),
    )
    metrics = pd.DataFrame(
        [
            {
                "model_name": "Atlantic_KGAE_only",
                "num_predictors": int(len(feature_cols)),
                **metrics_dict,
            }
        ]
    )

    period_rows = []
    for group_name, selector in PERIOD_SPECS:
        mask = np.asarray([selector(int(wy)) for wy in predictions["water_year"].to_numpy(dtype=int)], dtype=bool)
        sub = predictions.loc[mask].copy()
        if sub.empty:
            continue
        group_metrics = compute_metric_bundle(
            sub["observed_swe"].to_numpy(dtype=float),
            sub["predicted_swe"].to_numpy(dtype=float),
        )
        period_rows.append(
            {
                "model_name": "Atlantic_KGAE_only",
                "group_name": group_name,
                "n_years": int(sub.shape[0]),
                **group_metrics,
            }
        )
    return RidgeResult(
        predictions=predictions,
        selected_hyperparameters=pd.DataFrame(alpha_rows).sort_values("water_year").reset_index(drop=True),
        metrics=metrics,
        period_metrics=pd.DataFrame(period_rows),
    )


def save_history_csv(tracking: Dict[str, Dict[str, List[float]]], path: Path) -> None:
    train_keys = sorted(tracking["train"].keys())
    val_keys = sorted(tracking["val"].keys())
    n_epochs = max(len(v) for v in tracking["val"].values())
    rows = []
    for epoch in range(n_epochs):
        row = {"epoch": epoch}
        for key in train_keys:
            row[f"train_{key}"] = tracking["train"][key][epoch]
        for key in val_keys:
            row[f"val_{key}"] = tracking["val"][key][epoch]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def to_unstacked_da(values: np.ndarray, stacked_template: xr.DataArray, name: str) -> xr.DataArray:
    da = xr.DataArray(values, coords=stacked_template.coords, dims=stacked_template.dims, name=name)
    return da.unstack("feature").transpose("time", "lat", "lon").sortby("time")


def power_spectrum_summary(latent_da: xr.DataArray, k: int) -> pd.DataFrame:
    values = np.asarray(latent_da.values, dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(values.shape[0], d=1.0)
    coeff = np.fft.rfft(centered, axis=0)
    power = (coeff * np.conj(coeff)).real
    power = power / power.sum(axis=0, keepdims=True)
    rows = []
    for idx in range(k):
        j = int(np.nanargmax(power[1:, idx]) + 1) if power.shape[0] > 1 else 0
        dominant_freq = float(freqs[j])
        dominant_period = float("inf") if dominant_freq <= 0.0 else float(1.0 / dominant_freq)
        rows.append(
            {
                "k": int(k),
                "latent_name": f"Z{idx+1}",
                "variance": float(np.var(values[:, idx], ddof=1)),
                "dominant_frequency_cycles_per_month": dominant_freq,
                "dominant_period_months": dominant_period,
            }
        )
    return pd.DataFrame(rows)


def load_reference_patterns() -> xr.Dataset:
    return xr.open_dataset(REFERENCE_PATTERNS_NC)


def load_target_eofs() -> xr.Dataset:
    return xr.open_dataset(TARGET_EOF_PATTERNS_NC)


def spatial_compare(target: xr.DataArray, reference: xr.DataArray) -> Dict[str, float]:
    target, reference = xr.align(target, reference, join="inner")
    w = np.cos(np.deg2rad(target["lat"])).broadcast_like(target)
    mask = np.isfinite(target.values) & np.isfinite(reference.values) & np.isfinite(w.values)
    if mask.sum() < 2:
        return {
            "signed_spatial_r": float("nan"),
            "absolute_spatial_r": float("nan"),
            "demeaned_spatial_r": float("nan"),
            "pattern_congruence": float("nan"),
            "normalized_rmse": float("nan"),
            "optimal_sign": 1,
            "valid_grid_cells": int(mask.sum()),
        }
    x = target.values[mask]
    y = reference.values[mask]
    weights = w.values[mask]
    weights = weights / weights.sum()

    def weighted_corr(a: np.ndarray, b: np.ndarray) -> float:
        a_mean = np.sum(weights * a)
        b_mean = np.sum(weights * b)
        a0 = a - a_mean
        b0 = b - b_mean
        denom = math.sqrt(float(np.sum(weights * a0 * a0) * np.sum(weights * b0 * b0)))
        if denom <= 0.0:
            return float("nan")
        return float(np.sum(weights * a0 * b0) / denom)

    signed_r = weighted_corr(x, y)
    alt_r = weighted_corr(x, -y)
    optimal_sign = 1 if abs(signed_r) >= abs(alt_r) else -1
    yy = y * optimal_sign
    signed_r = weighted_corr(x, yy)
    x_std = (x - np.sum(weights * x)) / math.sqrt(float(np.sum(weights * (x - np.sum(weights * x)) ** 2)))
    y_std = (yy - np.sum(weights * yy)) / math.sqrt(float(np.sum(weights * (yy - np.sum(weights * yy)) ** 2)))
    congruence = float(np.sum(weights * x_std * y_std))
    rmse_norm = float(np.sqrt(np.sum(weights * (x_std - y_std) ** 2)))
    demeaned_r = weighted_corr(x - np.sum(weights * x), yy - np.sum(weights * yy))
    return {
        "signed_spatial_r": signed_r,
        "absolute_spatial_r": abs(signed_r),
        "demeaned_spatial_r": demeaned_r,
        "pattern_congruence": congruence,
        "normalized_rmse": rmse_norm,
        "optimal_sign": int(optimal_sign),
        "valid_grid_cells": int(mask.sum()),
    }


def latent_attribution_tables(
    probe_patterns: xr.DataArray,
    latent_da: xr.DataArray,
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference_patterns = load_reference_patterns()
    target_eofs = load_target_eofs()
    reference_indices = pd.read_csv(REFERENCE_INDICES_CSV, parse_dates=["date"])

    spatial_rows = []
    temporal_rows = []
    for mode_idx in range(k):
        latent_name = f"Z{mode_idx+1}"
        target = probe_patterns.sel(mode=mode_idx + 1)
        for ref_name in list(reference_patterns.data_vars):
            compare = spatial_compare(target, reference_patterns[ref_name])
            spatial_rows.append(
                {
                    "k": int(k),
                    "latent_name": latent_name,
                    "reference_mode": ref_name,
                    **compare,
                }
            )
        for eof_name in list(target_eofs.data_vars):
            compare = spatial_compare(target, target_eofs[eof_name])
            spatial_rows.append(
                {
                    "k": int(k),
                    "latent_name": latent_name,
                    "reference_mode": eof_name,
                    **compare,
                }
            )

        latent_series = pd.DataFrame({"date": pd.to_datetime(latent_da["time"].values), "latent": latent_da.sel(mode=mode_idx + 1).values})
        monthly_cols = ["NOAA_AMV", "NAO", "NorthAtlantic_basin_mean", "Global_mean_SST", "PC2", "PC4", "PC5"]
        merged = latent_series.merge(reference_indices[["date"] + monthly_cols], on="date", how="left")
        for col in monthly_cols:
            r = corrcoef_safe(merged["latent"].to_numpy(dtype=float), merged[col].to_numpy(dtype=float))
            temporal_rows.append(
                {
                    "k": int(k),
                    "latent_name": latent_name,
                    "comparison_name": col,
                    "timescale": "monthly",
                    "correlation": r,
                    "n_samples": int(np.isfinite(merged["latent"]) & np.isfinite(merged[col])).sum(),
                }
            )
        seasonal = merged[merged["date"].dt.month.isin([11, 12, 1, 2])].copy()
        seasonal["season_year"] = np.where(seasonal["date"].dt.month >= 11, seasonal["date"].dt.year + 1, seasonal["date"].dt.year)
        latent_ndjf = seasonal.groupby("season_year")["latent"].mean().reset_index()
        aqm = reference_indices.loc[reference_indices["date"].dt.month == 3, ["date", "AQM_S2_NDJF"]].copy()
        aqm["season_year"] = aqm["date"].dt.year
        seasonal_merged = latent_ndjf.merge(aqm[["season_year", "AQM_S2_NDJF"]], on="season_year", how="inner")
        temporal_rows.append(
            {
                "k": int(k),
                "latent_name": latent_name,
                "comparison_name": "AQM_S2_NDJF",
                "timescale": "NDJF",
                "correlation": corrcoef_safe(
                    seasonal_merged["latent"].to_numpy(dtype=float),
                    seasonal_merged["AQM_S2_NDJF"].to_numpy(dtype=float),
                ),
                "n_samples": int(seasonal_merged.shape[0]),
            }
        )
    return pd.DataFrame(spatial_rows), pd.DataFrame(temporal_rows)


def read_saved_metric(path: Path, model_name: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    row = df.loc[df["model_name"] == model_name].iloc[0].to_dict()
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v) for k, v in row.items()}


def comparison_table(kgae_metrics_rows: List[Dict[str, object]]) -> pd.DataFrame:
    comparison_rows = []
    full_pc = read_saved_metric(AQM_ARTIFACT_DIR / "aqm_loyo_audit.json", "AMV_AMO_PC1to6_only") if False else None
    amv_pc = pd.read_csv(AQM_ARTIFACT_DIR / "aqm_loyo_audit.json") if False else None
    aqm_metrics = pd.read_csv(AQM_METRICS).iloc[0].to_dict()
    comparison_rows.append(
        {
            "model_name": "Atlantic_PCA_full_PC1to6",
            "model_family": "Atlantic_PCA_saved",
            "num_predictors": 42,
            "r": 0.6334877378429052,
            "R2": 0.3387300543701577,
            "RMSE": 0.0257871232551144,
            "MAE": 0.019394638502849698,
            "sign_accuracy": 0.6486486486486487,
            "source": str(AQM_ARTIFACT_DIR / "README.md"),
        }
    )
    comparison_rows.append(
        {
            "model_name": "Atlantic_selected_K5_PC_month",
            "model_family": "Atlantic_PCA_saved",
            "num_predictors": 5,
            "r": 0.5619277782743538,
            "R2": 0.2998398559841624,
            "RMSE": 0.0265345786890728,
            "MAE": 0.021749,
            "sign_accuracy": 0.7567567567567568,
            "source": str(AQM_ARTIFACT_DIR / "README.md"),
        }
    )
    comparison_rows.append(
        {
            "model_name": "AQM_seasonal_index_only",
            "model_family": "AQM_saved",
            "num_predictors": int(aqm_metrics["num_predictors"]),
            "r": float(aqm_metrics["r"]),
            "R2": float(aqm_metrics["R2"]),
            "RMSE": float(aqm_metrics["RMSE"]),
            "MAE": float(aqm_metrics["MAE"]),
            "sign_accuracy": float(aqm_metrics["sign_accuracy"]),
            "source": str(AQM_METRICS),
        }
    )
    comparison_rows.extend(kgae_metrics_rows)
    return pd.DataFrame(comparison_rows)


def make_reconstruction_vs_k_figure(kgae_df: pd.DataFrame, pca_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    kgae_test = kgae_df[kgae_df["split_name"] == "test"].sort_values("k")
    pca_test = pca_df[pca_df["split_name"] == "test"].sort_values("k")
    plt.plot(kgae_test["k"], kgae_test["area_weighted_rmse"], marker="o", label="Atlantic KGAE")
    plt.plot(pca_test["k"], pca_test["area_weighted_rmse"], marker="s", label="Atlantic PCA")
    plt.xlabel("Latent dimension k")
    plt.ylabel("Held-out SST RMSE")
    plt.title("Held-out Atlantic SST reconstruction vs k")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_reconstruction_split_figure(kgae_df: pd.DataFrame, out_path: Path) -> None:
    splits = ["train", "validation", "test"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    width = 0.22
    ks = sorted(kgae_df["k"].unique().tolist())
    x = np.arange(len(ks))
    for offset, split in enumerate(splits):
        sub = kgae_df[kgae_df["split_name"] == split].sort_values("k")
        ax.bar(x + (offset - 1) * width, sub["area_weighted_rmse"], width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Area-weighted SST RMSE")
    ax.set_title("Atlantic KGAE reconstruction by split")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_kgae_vs_pca_figure(kgae_df: pd.DataFrame, pca_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ks = sorted(kgae_df["k"].unique().tolist())
    width = 0.35
    x = np.arange(len(ks))
    kgae_vals = kgae_df[kgae_df["split_name"] == "test"].sort_values("k")["area_weighted_rmse"].to_numpy(dtype=float)
    pca_vals = pca_df[pca_df["split_name"] == "test"].sort_values("k")["area_weighted_rmse"].to_numpy(dtype=float)
    ax.bar(x - width / 2, kgae_vals, width=width, label="Atlantic KGAE")
    ax.bar(x + width / 2, pca_vals, width=width, label="Atlantic PCA")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Test SST RMSE")
    ax.set_title("Atlantic KGAE vs Atlantic PCA")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_example_maps(obs_da: xr.DataArray, rec_da: xr.DataArray, out_path: Path, title_prefix: str) -> None:
    indices = [0, obs_da.sizes["time"] // 2, obs_da.sizes["time"] - 1]
    fig, axes = plt.subplots(len(indices), 2, figsize=(9, 3.2 * len(indices)))
    vmax = float(np.nanmax(np.abs(obs_da.values)))
    for row, idx in enumerate(indices):
        obs_da.isel(time=idx).plot(ax=axes[row, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax, add_colorbar=False)
        rec_da.isel(time=idx).plot(ax=axes[row, 1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, add_colorbar=False)
        axes[row, 0].set_title(f"{title_prefix} obs {pd.Timestamp(obs_da['time'].values[idx]).date()}")
        axes[row, 1].set_title(f"{title_prefix} recon {pd.Timestamp(obs_da['time'].values[idx]).date()}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_residual_maps(resid_da: xr.DataArray, out_path: Path) -> None:
    indices = [0, resid_da.sizes["time"] // 2, resid_da.sizes["time"] - 1]
    fig, axes = plt.subplots(1, len(indices), figsize=(4.2 * len(indices), 3.8))
    vmax = float(np.nanmax(np.abs(resid_da.values)))
    for col, idx in enumerate(indices):
        resid_da.isel(time=idx).plot(ax=axes[col], cmap="RdBu_r", vmin=-vmax, vmax=vmax, add_colorbar=False)
        axes[col].set_title(str(pd.Timestamp(resid_da["time"].values[idx]).date()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_power_spectra_figure(latent_da: xr.DataArray, out_path: Path) -> None:
    values = latent_da.to_numpy(dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(values.shape[0], d=1.0)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power = power / power.sum(axis=0, keepdims=True)
    k = values.shape[1]
    fig, axes = plt.subplots(k, 1, figsize=(7.5, 2.0 * k), sharex=True)
    if k == 1:
        axes = [axes]
    for idx in range(k):
        axes[idx].plot(freqs, power[:, idx], color="tab:blue")
        axes[idx].set_ylabel(f"Z{idx+1}")
    axes[-1].set_xlabel("Cycles per month")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_heatmap(table: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    pivot = table.pivot(index="latent_name", columns="reference_mode", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(1.2 * max(4, pivot.shape[1]), 0.8 * max(4, pivot.shape[0])))
    im = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_swe_skill_figure(kgae_comp: pd.DataFrame, out_path: Path) -> None:
    df = kgae_comp[kgae_comp["model_family"] == "Atlantic_KGAE"].sort_values("k")
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(df["k"], df["R2"], marker="o", label="KGAE R2")
    plt.axhline(0.3387300543701577, color="tab:blue", linestyle="--", label="Atlantic PCA PC1-6")
    plt.axhline(0.2998398559841624, color="tab:green", linestyle="--", label="Selected Atlantic K5")
    plt.axhline(0.09738540851660227, color="tab:red", linestyle="--", label="AQM index")
    plt.xlabel("Latent dimension k")
    plt.ylabel("LOYO SWE R2")
    plt.title("Atlantic KGAE SWE skill vs k")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_best_swe_timeseries(pred_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.plot(pred_df["water_year"], pred_df["observed_swe"], color="black", label="Observed SWE")
    plt.plot(pred_df["water_year"], pred_df["predicted_swe"], color="tab:orange", label="Atlantic KGAE")
    plt.xlabel("Water year")
    plt.ylabel("April 1 SWE")
    plt.title("Best Atlantic KGAE LOYO timeseries")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_best_swe_scatter(pred_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(5.0, 5.0))
    plt.scatter(pred_df["observed_swe"], pred_df["predicted_swe"], color="tab:orange", alpha=0.85)
    lo = float(min(pred_df["observed_swe"].min(), pred_df["predicted_swe"].min()))
    hi = float(max(pred_df["observed_swe"].max(), pred_df["predicted_swe"].max()))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("Observed SWE")
    plt.ylabel("Predicted SWE")
    plt.title("Best Atlantic KGAE observed vs predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_combo_figure(reconstruction_df: pd.DataFrame, swe_comp_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    test_df = reconstruction_df[reconstruction_df["split_name"] == "test"].sort_values("k")
    axes[0].plot(test_df["k"], test_df["area_weighted_rmse"], marker="o", label="KGAE")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Test SST RMSE")
    axes[0].set_title("Reconstruction")

    swe_df = swe_comp_df[swe_comp_df["model_family"] == "Atlantic_KGAE"].sort_values("k")
    axes[1].plot(swe_df["k"], swe_df["RMSE"], marker="o", label="KGAE")
    axes[1].axhline(0.0257871232551144, color="tab:blue", linestyle="--", label="Atlantic PCA PC1-6")
    axes[1].axhline(0.030127609854625342, color="tab:red", linestyle="--", label="AQM")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("LOYO SWE RMSE")
    axes[1].set_title("SWE prediction")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_method_reconstruction(path: Path, split_years: SplitYears, smoke: bool) -> None:
    text = "\n".join(
        [
            "# Atlantic KGAE method reconstruction",
            "",
            "## Published KGAE implementation",
            "",
            f"- Local KGAE clone: `{DEFAULT_KGAE_REPO}`",
            "- GitHub repository: `https://github.com/kjhall01/kgae`",
            "- Local commit: `df40f04d7546db2654b2c742c888a82b50fe06d9`",
            "- Repository README identifies the method as Hall et al. (2025, submitted/under review).",
            "- Model class reused directly: `kgae.kgae.KGAE`.",
            "- Input tensor layout: one monthly SST anomaly map per sample, flattened to `(time, feature)` after masking ocean points.",
            "- Encoder architecture: feed-forward MLP `input -> 248 -> 248 -> 2*k` with `Tanh` activations and Xavier-normal layer initialization.",
            "- Decoder architecture: mirrored MLP `k -> 248 -> 248 -> input` with `Tanh` activations.",
            "- Variational setting: `is_variational=True`.",
            "- Frequency decomposition: handled by the latent power-spectrum losses inside `KGAE.compute_loss` plus post-training `sort_latents_by_frequency` ordering.",
            "- Reconstruction loss: mean squared error in feature space.",
            "- Additional knowledge-guided losses: KL divergence, spectral overlap, spectral-weighted spatial-correlation penalty, curve-fit MAE, decoder regularization, and latent centering loss.",
            "- Loss weights: KL `1e-2`, Reconstruction `1`, Spectral Overlap `1`, Spectral Weighted Spatial Correlations `1`, Curve Fit MAE `1`, L1 Decoder Reg `1e-4`, Centering Loss `1`.",
            "- Optimizer: Adam with learning rate `1e-3`.",
            "- EMA: exponential moving average with decay `0.995`.",
            "- Published preprocessing reused: `kgae.global_detrend(..., deg=2)` followed by `kgae.remove_climo(...)`.",
            "",
            "## Atlantic adaptation",
            "",
            "- SST dataset: COBE2 monthly SST from `/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc`.",
            "- Atlantic domain: `0-70N`, `280-360E` (`80W-0`).",
            "- Longitude convention: `0..360` with duplicate cyclic endpoint removed.",
            "- Ocean mask: exact `valid_mask` from the saved Atlantic EOF dataset so the KGAE grid matches the Atlantic EOF experiment.",
            "- KGAE preprocessing change relative to the Atlantic EOF experiment: the Atlantic EOF experiment used monthly-climatology anomalies without detrending, while the published KGAE method explicitly removes a basin-mean quadratic trend before monthly climatology removal. That published KGAE preprocessing was retained here and documented rather than replaced.",
            "- Sign orientation after frequency sorting: anchored to a uniform positive Atlantic basin anomaly pattern to avoid importing a Pacific-specific sign anchor.",
            "",
            "## Temporal split",
            "",
            f"- Smoke mode: `{smoke}`",
            f"- Training years: `{split_years.train_years[0]}` to `{split_years.train_years[-1]}`",
            f"- Validation years: `{split_years.val_years[0]}` to `{split_years.val_years[-1]}`",
            f"- Test years: `{split_years.test_years[0]}` to `{split_years.test_years[-1]}`",
            "- All months from a calendar year remain in the same split.",
            "- Final SST test block is untouched during optimization and checkpoint selection.",
            "",
            "## SWE regression",
            "",
            "- The frozen encoder is used to encode Sep-Mar Atlantic SST for WY1985-WY2021.",
            "- Predictor table columns are `KGAE_Zi_Month` with month identity preserved.",
            "- Ridge alpha grid matches the reference Atlantic SWE workflow exactly: `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]`.",
            "- Standardization, LOYO splitting, tie-breaking, intercept reconstruction, and metric formulas follow the reference ridge workflow.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def write_readme(
    path: Path,
    split_years: SplitYears,
    smoke_metadata: Dict[str, object] | None,
    slurm_job_id: str | None,
) -> None:
    lines = [
        "# Atlantic KGAE experiment",
        "",
        "- Artifact directory: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/atlantic_kgae`",
        "- Heavy working directory: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/atlantic_kgae`",
        "- K values: `3, 5, 7, 10`",
        "- SST split:",
        f"  - train `{split_years.train_years[0]}` to `{split_years.train_years[-1]}`",
        f"  - validation `{split_years.val_years[0]}` to `{split_years.val_years[-1]}`",
        f"  - test `{split_years.test_years[0]}` to `{split_years.test_years[-1]}`",
        "- Atlantic domain: `0-70N`, `80W-0` on the COBE2 grid and Atlantic EOF valid mask.",
        "- Published KGAE repo reused directly: `https://github.com/kjhall01/kgae` at local commit `df40f04d7546db2654b2c742c888a82b50fe06d9`.",
    ]
    if smoke_metadata is not None:
        lines.extend(
            [
                "",
                "## Smoke test",
                "",
                f"- Passed: `{smoke_metadata.get('smoke_passed')}`",
                f"- Compute node: `{smoke_metadata.get('hostname')}`",
                f"- Interactive job id: `{smoke_metadata.get('slurm_job_id')}`",
            ]
        )
    if slurm_job_id is not None:
        lines.extend(["", "## Production batch", "", f"- Slurm job id: `{slurm_job_id}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_smoke_report(path: Path, smoke_summary: Dict[str, object]) -> None:
    lines = [
        "# Atlantic KGAE smoke test report",
        "",
        f"- Smoke test passed: `{smoke_summary['smoke_passed']}`",
        f"- tmux session: `{smoke_summary['tmux_session']}`",
        f"- Slurm job id: `{smoke_summary['slurm_job_id']}`",
        f"- Hostname: `{smoke_summary['hostname']}`",
        f"- Python executable: `{smoke_summary['python']}`",
        f"- KGAE device: `{smoke_summary['device']}`",
        f"- Latent dimension: `{smoke_summary['latent_dim']}`",
        f"- Epochs: `{smoke_summary['epochs']}`",
        f"- Train years: `{smoke_summary['train_years']}`",
        f"- Validation years: `{smoke_summary['val_years']}`",
        f"- Test years: `{smoke_summary['test_years']}`",
        f"- Checks completed: `{', '.join(smoke_summary['checks_completed'])}`",
        f"- Heavy smoke directory: `{smoke_summary['heavy_output_dir']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_python_env_report(python_path: Path) -> Dict[str, str]:
    import subprocess

    code = (
        "import sys, torch, xarray as xr, numpy as np, pandas as pd; "
        "print(sys.executable); print(torch.__version__); print(xr.__version__); print(np.__version__); print(pd.__version__)"
    )
    out = subprocess.check_output([str(python_path), "-c", code], text=True)
    lines = out.strip().splitlines()
    return {
        "python": lines[0],
        "torch": lines[1],
        "xarray": lines[2],
        "numpy": lines[3],
        "pandas": lines[4],
    }


def run_smoke(args: argparse.Namespace) -> int:
    ensure_compute_runtime()
    prepared = prepare_atlantic_sst(smoke=True)
    heavy_dir = ensure_dir(args.pscratch_output / "smoke_test")
    ensure_dir(args.home_output)
    train_batches = build_training_batches(prepared.train_stacked.values)
    model, tracking, fit_meta = fit_kgae_with_checkpoint(
        train_batches=train_batches,
        val_data=prepared.val_stacked.values,
        latent_dim=args.smoke_latent_dim,
        num_epochs=args.smoke_epochs,
        seed=PRIMARY_SEED,
        device=args.device,
        out_checkpoint=heavy_dir / "smoke_best_model.pt",
    )
    model.sort_latents_by_frequency(
        prepared.val_stacked.values.astype(np.float32),
        reference_pattern=reference_sign_pattern(prepared.val_stacked.sizes["feature"]),
    )
    torch.save(model.state_dict(), heavy_dir / "smoke_model_state.pt")
    with open(heavy_dir / "smoke_tracking.pkl", "wb") as handle:
        pickle.dump(tracking, handle)

    reloaded = kgae.KGAE(
        input_dim=prepared.train_stacked.sizes["feature"],
        latent_dim=args.smoke_latent_dim,
        is_variational=True,
        hidden_layers=DEFAULT_HIDDEN_LAYERS,
        activation=torch.nn.Tanh,
        power_spectrum_smoothing_kernel=7,
        device=args.device,
        seed=PRIMARY_SEED,
        tag="atlantic_kgae_smoke_reload",
    )
    reloaded.load_state_dict(torch.load(heavy_dir / "smoke_model_state.pt", map_location=args.device))
    reloaded.flip_signs = model.flip_signs

    test_latents = encode_latents(reloaded, prepared.test_stacked.values)
    test_recon = decode_latents(reloaded, test_latents)
    latent_da = xr.DataArray(
        encode_latents(reloaded, prepared.full_stacked.values),
        coords={"time": prepared.full_stacked["time"].values, "mode": np.arange(1, args.smoke_latent_dim + 1)},
        dims=("time", "mode"),
        name="atlantic_kgae_smoke_latents",
    )
    latent_da.to_netcdf(heavy_dir / "smoke_monthly_latents.nc")
    predictor_table = build_kgae_predictor_table(latent_da, obs_swe_series(), args.smoke_latent_dim)
    predictor_table.to_csv(heavy_dir / "smoke_swe_predictor_table.csv", index=False)
    ridge_result = run_reference_style_loyo(
        predictor_table,
        [c for c in predictor_table.columns if c not in {"water_year", "observed_swe"}],
    )
    ridge_result.predictions.to_csv(heavy_dir / "smoke_swe_loyo_predictions.csv", index=False)
    ridge_result.metrics.to_csv(heavy_dir / "smoke_swe_loyo_metrics.csv", index=False)
    ridge_result.selected_hyperparameters.to_csv(heavy_dir / "smoke_swe_loyo_selected_hyperparameters.csv", index=False)
    ridge_result.period_metrics.to_csv(heavy_dir / "smoke_swe_loyo_period_metrics.csv", index=False)
    test_recon_da = to_unstacked_da(test_recon, prepared.test_stacked, "smoke_test_reconstruction")
    test_recon_da.to_netcdf(heavy_dir / "smoke_test_reconstruction.nc")

    smoke_summary = {
        "smoke_passed": True,
        "tmux_session": "atlantic_kgae_smoke",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "device": args.device,
        "latent_dim": int(args.smoke_latent_dim),
        "epochs": int(args.smoke_epochs),
        "train_years": [prepared.split_years.train_years[0], prepared.split_years.train_years[-1]],
        "val_years": [prepared.split_years.val_years[0], prepared.split_years.val_years[-1]],
        "test_years": [prepared.split_years.test_years[0], prepared.split_years.test_years[-1]],
        "best_epoch": fit_meta["best_epoch"],
        "checks_completed": [
            "load Atlantic COBE2 SST",
            "construct published-style KGAE anomalies",
            "create contiguous train/validation/test blocks",
            "instantiate Atlantic KGAE",
            "run forward pass",
            "calculate all KGAE loss terms",
            "run backward pass and optimizer update",
            "save checkpoint",
            "reload checkpoint",
            "reconstruct held-out SST",
            "encode Sep-Mar Atlantic SST",
            "build annual WY1985-WY2021 predictor table",
            "run minimal reference-style LOYO ridge",
            "write smoke output files to pscratch",
        ],
        "heavy_output_dir": str(heavy_dir),
    }
    (heavy_dir / "smoke_summary.json").write_text(json.dumps(smoke_summary, indent=2) + "\n", encoding="utf-8")
    write_smoke_report(args.home_output / "smoke_test_report.md", smoke_summary)
    print(json.dumps(smoke_summary, indent=2))
    return 0


def run_full(args: argparse.Namespace) -> int:
    ensure_compute_runtime()
    prepared = prepare_atlantic_sst(smoke=False)
    ensure_dir(args.home_output)
    heavy_dir = ensure_dir(args.pscratch_output)
    ensure_dir(DEFAULT_LOG_DIR)
    figures_dir = ensure_dir(args.home_output / "figures")
    checkpoints_dir = ensure_dir(args.home_output / "checkpoints")
    latents_dir = ensure_dir(args.home_output / "latents")
    recon_dir = ensure_dir(args.home_output / "reconstructions")
    metrics_dir = ensure_dir(args.home_output / "metrics")
    training_history_dir = ensure_dir(args.home_output / "training_history")

    write_method_reconstruction(args.home_output / "method_and_kgae_reconstruction.md", prepared.split_years, smoke=False)
    split_payload = {
        "train_years": prepared.split_years.train_years,
        "validation_years": prepared.split_years.val_years,
        "test_years": prepared.split_years.test_years,
        "full_record_start": str(pd.Timestamp(prepared.raw_full["time"].values[0]).date()),
        "full_record_end": str(pd.Timestamp(prepared.raw_full["time"].values[-1]).date()),
        "train_months": int(prepared.train_stacked.sizes["time"]),
        "validation_months": int(prepared.val_stacked.sizes["time"]),
        "test_months": int(prepared.test_stacked.sizes["time"]),
        "swe_year_overlap_statement": "KGAE SST training is SWE-independent, so overlap with WY1985-WY2021 is not target leakage.",
    }
    (args.home_output / "data_split.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    config_payload = {
        "kgae_repo": str(DEFAULT_KGAE_REPO),
        "kgae_git_remote": "https://github.com/kjhall01/kgae",
        "kgae_git_commit": "df40f04d7546db2654b2c742c888a82b50fe06d9",
        "device": args.device,
        "epochs": int(args.epochs),
        "hidden_layers": DEFAULT_HIDDEN_LAYERS,
        "loss_weights": LOSS_WEIGHTS,
        "losses_to_use": LOSSES_TO_USE,
        "optimizer": {"name": "Adam", "lr": 1.0e-3},
        "ema_decay": 0.995,
        "ridge_alpha_grid": RIDGE_ALPHA_GRID.tolist(),
        "latent_dimensions": K_VALUES,
        "seed": PRIMARY_SEED,
    }
    (args.home_output / "training_configuration.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    weights_2d = area_weights_from_mask(prepared.valid_mask).values
    flat_weights = weights_2d[prepared.valid_mask.values]
    obs_swe = obs_swe_series()
    train_batches = build_training_batches(prepared.train_stacked.values)
    model_inventory_rows = []
    kgae_recon_rows = []
    pca_recon_rows = []
    latent_power_rows = []
    latent_attr_rows = []
    comparison_rows = []
    best_prediction_df = None
    best_skill_row = None
    best_recon_rmse = None
    best_recon_k = None
    best_probe_patterns = None
    best_latent_da = None
    best_test_recon_da = None
    best_test_resid_da = None

    pca_period_rows = []
    for k in K_VALUES:
        pca_result = fit_pca_and_reconstruct(
            prepared.train_stacked.values,
            prepared.val_stacked.values,
            prepared.test_stacked.values,
            prepared.full_stacked.values,
            k,
        )
        for split_name, obs_values, rec_values in [
            ("train", prepared.train_stacked.values, pca_result["train_recon"]),
            ("validation", prepared.val_stacked.values, pca_result["val_recon"]),
            ("test", prepared.test_stacked.values, pca_result["test_recon"]),
        ]:
            pca_period_rows.append(
                reconstruction_metrics(
                    obs_values,
                    rec_values,
                    prepared.train_stacked if split_name == "train" else prepared.val_stacked if split_name == "validation" else prepared.test_stacked,
                    flat_weights,
                    split_name,
                    "Atlantic_PCA",
                    k,
                )
            )
    pca_recon_df = pd.DataFrame(pca_period_rows)
    pca_recon_df.to_csv(args.home_output / "pca_reconstruction_metrics.csv", index=False)

    for k in K_VALUES:
        checkpoint_path = heavy_dir / "checkpoints" / f"k{k}" / "best_model.pt"
        ensure_dir(checkpoint_path.parent)
        model, tracking, fit_meta = fit_kgae_with_checkpoint(
            train_batches=train_batches,
            val_data=prepared.val_stacked.values,
            latent_dim=k,
            num_epochs=args.epochs,
            seed=PRIMARY_SEED,
            device=args.device,
            out_checkpoint=checkpoint_path,
        )
        model.sort_latents_by_frequency(
            prepared.val_stacked.values.astype(np.float32),
            reference_pattern=reference_sign_pattern(prepared.val_stacked.sizes["feature"]),
        )
        torch.save(model.state_dict(), checkpoint_path)
        save_history_csv(tracking, training_history_dir / f"k{k}_training_history.csv")

        full_latents = encode_latents(model, prepared.full_stacked.values)
        latent_da = xr.DataArray(
            full_latents,
            coords={"time": prepared.full_stacked["time"].values, "mode": np.arange(1, k + 1)},
            dims=("time", "mode"),
            name=f"k{k}_monthly_latents",
        )
        monthly_latent_csv = latents_dir / f"k{k}_monthly_latents.csv"
        latent_df = pd.DataFrame(full_latents, columns=[f"Z{i}" for i in range(1, k + 1)])
        latent_df.insert(0, "date", pd.to_datetime(prepared.full_stacked["time"].values))
        latent_df.to_csv(monthly_latent_csv, index=False)

        test_latents = encode_latents(model, prepared.test_stacked.values)
        train_latents = encode_latents(model, prepared.train_stacked.values)
        val_latents = encode_latents(model, prepared.val_stacked.values)
        train_recon = decode_latents(model, train_latents)
        val_recon = decode_latents(model, val_latents)
        test_recon = decode_latents(model, test_latents)
        full_recon = decode_latents(model, full_latents)
        probe_patterns = unit_probe_patterns(model, prepared.full_stacked)

        test_recon_da = to_unstacked_da(test_recon, prepared.test_stacked, f"k{k}_test_reconstruction")
        test_resid_da = to_unstacked_da(test_recon - prepared.test_stacked.values, prepared.test_stacked, f"k{k}_test_residuals")
        test_recon_da.to_netcdf(recon_dir / f"k{k}_test_reconstruction.nc")
        test_resid_da.to_netcdf(recon_dir / f"k{k}_test_residuals.nc")

        split_metric_rows = []
        for split_name, obs_values, rec_values, template in [
            ("train", prepared.train_stacked.values, train_recon, prepared.train_stacked),
            ("validation", prepared.val_stacked.values, val_recon, prepared.val_stacked),
            ("test", prepared.test_stacked.values, test_recon, prepared.test_stacked),
        ]:
            split_metric_rows.append(
                reconstruction_metrics(obs_values, rec_values, template, flat_weights, split_name, "Atlantic_KGAE", k)
            )
        split_metric_df = pd.DataFrame(split_metric_rows)
        split_metric_df["generalization_ratio_test_to_train_rmse"] = (
            split_metric_df.loc[split_metric_df["split_name"] == "test", "area_weighted_rmse"].iloc[0]
            / split_metric_df.loc[split_metric_df["split_name"] == "train", "area_weighted_rmse"].iloc[0]
        )
        split_metric_df.to_csv(metrics_dir / f"k{k}_reconstruction_metrics.csv", index=False)
        kgae_recon_rows.extend(split_metric_df.to_dict(orient="records"))

        latent_power_rows.extend(power_spectrum_summary(latent_da, k).to_dict(orient="records"))
        spatial_attr_df, temporal_attr_df = latent_attribution_tables(probe_patterns, latent_da, k)
        attr_df = spatial_attr_df.merge(
            temporal_attr_df.groupby(["k", "latent_name"])["correlation"].max().rename("best_temporal_correlation").reset_index(),
            on=["k", "latent_name"],
            how="left",
        )
        attr_df["attribution_source"] = "Atlantic KGAE latent decoder probe vs saved Atlantic references"
        latent_attr_rows.extend(attr_df.to_dict(orient="records"))

        predictor_table = build_kgae_predictor_table(latent_da, obs_swe, k)
        predictor_table.to_csv(latents_dir / f"k{k}_swe_predictor_table.csv", index=False)
        ridge_result = run_reference_style_loyo(
            predictor_table,
            [c for c in predictor_table.columns if c not in {"water_year", "observed_swe"}],
        )
        ridge_result.predictions.to_csv(metrics_dir / f"k{k}_swe_loyo_predictions.csv", index=False)
        ridge_result.selected_hyperparameters.to_csv(metrics_dir / f"k{k}_swe_loyo_selected_hyperparameters.csv", index=False)
        ridge_result.metrics.to_csv(metrics_dir / f"k{k}_swe_loyo_metrics.csv", index=False)
        ridge_result.period_metrics.to_csv(metrics_dir / f"k{k}_swe_loyo_period_metrics.csv", index=False)

        metrics_row = ridge_result.metrics.iloc[0].to_dict()
        metrics_row.update({"model_name": f"Atlantic_KGAE_k{k}", "model_family": "Atlantic_KGAE", "k": int(k)})
        comparison_rows.append(metrics_row)

        model_inventory_rows.append(
            {
                "model_name": f"Atlantic_KGAE_k{k}",
                "model_family": "Atlantic_KGAE",
                "k": int(k),
                "checkpoint_path": str(checkpoint_path),
                "monthly_latents_csv": str(monthly_latent_csv),
                "predictor_table_csv": str(latents_dir / f"k{k}_swe_predictor_table.csv"),
                "best_epoch": int(fit_meta["best_epoch"]),
                "best_val_selection_objective": float(fit_meta["best_val_selection_objective"]),
                "best_val_reconstruction_mse": float(fit_meta["best_val_reconstruction_mse"]),
            }
        )

        test_rmse = split_metric_df.loc[split_metric_df["split_name"] == "test", "area_weighted_rmse"].iloc[0]
        if best_recon_rmse is None or test_rmse < best_recon_rmse:
            best_recon_rmse = float(test_rmse)
            best_recon_k = int(k)
            best_probe_patterns = probe_patterns
            best_latent_da = latent_da
            best_test_recon_da = test_recon_da
            best_test_resid_da = test_resid_da
        if best_skill_row is None or float(metrics_row["RMSE"]) < float(best_skill_row["RMSE"]):
            best_skill_row = metrics_row
            best_prediction_df = ridge_result.predictions.copy()

    model_inventory_df = pd.DataFrame(model_inventory_rows)
    model_inventory_df.to_csv(args.home_output / "model_inventory.csv", index=False)

    kgae_recon_df = pd.DataFrame(kgae_recon_rows)
    kgae_recon_df.to_csv(args.home_output / "reconstruction_metrics_by_period.csv", index=False)
    overall_recon = (
        kgae_recon_df[kgae_recon_df["split_name"] == "test"][["model_name", "k", "area_weighted_mse", "area_weighted_rmse", "area_weighted_mae", "mean_monthly_spatial_pattern_correlation", "area_weighted_mean_gridcell_temporal_correlation", "explained_reconstruction_variance", "generalization_ratio_test_to_train_rmse"]]
        .copy()
        .rename(columns={"model_name": "model_family"})
    )
    overall_recon.to_csv(args.home_output / "reconstruction_metrics.csv", index=False)

    latent_power_df = pd.DataFrame(latent_power_rows)
    latent_power_df.to_csv(args.home_output / "latent_frequency_summary.csv", index=False)
    latent_attr_df = pd.DataFrame(latent_attr_rows)
    latent_attr_df.to_csv(args.home_output / "latent_physical_attribution.csv", index=False)

    swe_comp_df = comparison_table(comparison_rows)
    swe_comp_df.to_csv(args.home_output / "swe_model_comparison.csv", index=False)

    if best_recon_k is None or best_probe_patterns is None or best_latent_da is None or best_test_recon_da is None or best_test_resid_da is None or best_prediction_df is None or best_skill_row is None:
        raise RuntimeError("Best-model tracking failed.")

    make_reconstruction_vs_k_figure(kgae_recon_df, pca_recon_df, figures_dir / "reconstruction_vs_k.png")
    make_reconstruction_split_figure(kgae_recon_df, figures_dir / "reconstruction_train_validation_test.png")
    make_kgae_vs_pca_figure(kgae_recon_df, pca_recon_df, figures_dir / "kgae_vs_pca_reconstruction.png")
    make_example_maps(to_unstacked_da(prepared.test_stacked.values, prepared.test_stacked, "observed_test"), best_test_recon_da, figures_dir / "test_reconstruction_examples.png", "Atlantic")
    make_residual_maps(best_test_resid_da, figures_dir / "test_residual_examples.png")
    make_power_spectra_figure(best_latent_da, figures_dir / "latent_power_spectra.png")
    best_spatial_attr, best_temporal_attr = latent_attribution_tables(best_probe_patterns, best_latent_da, best_recon_k)
    make_heatmap(best_spatial_attr, "signed_spatial_r", figures_dir / "latent_attribution_heatmap.png", f"Best-k spatial attribution (k={best_recon_k})")
    make_swe_skill_figure(swe_comp_df, figures_dir / "swe_skill_vs_k.png")
    make_best_swe_timeseries(best_prediction_df, figures_dir / "best_k_swe_timeseries.png")
    make_best_swe_scatter(best_prediction_df, figures_dir / "best_k_swe_observed_vs_predicted.png")
    make_combo_figure(kgae_recon_df, swe_comp_df, figures_dir / "model_comparison_reconstruction_and_swe.png")

    run_audit = {
        "tmux_session": "atlantic_kgae_smoke",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "device": args.device,
        "train_months": int(prepared.train_stacked.sizes["time"]),
        "validation_months": int(prepared.val_stacked.sizes["time"]),
        "test_months": int(prepared.test_stacked.sizes["time"]),
        "full_months": int(prepared.full_stacked.sizes["time"]),
        "n_features": int(prepared.full_stacked.sizes["feature"]),
        "best_reconstruction_k": int(best_recon_k),
        "best_swe_k": int(best_skill_row["k"]),
        "no_swe_in_kgae_training": True,
        "encoder_frozen_for_swe": True,
    }
    (args.home_output / "run_audit.json").write_text(json.dumps(run_audit, indent=2) + "\n", encoding="utf-8")
    (heavy_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "finished_utc": pd.Timestamp.utcnow().isoformat(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": os.uname().nodename,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_readme(args.home_output / "README.md", prepared.split_years, None, os.environ.get("SLURM_JOB_ID"))
    print("Atlantic KGAE production run completed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--home-output", type=Path, default=DEFAULT_HOME_OUT)
    parser.add_argument("--pscratch-output", type=Path, default=DEFAULT_PSCRATCH_OUT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--smoke-epochs", type=int, default=DEFAULT_SMOKE_EPOCHS)
    parser.add_argument("--smoke-latent-dim", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return run_smoke(args)
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
