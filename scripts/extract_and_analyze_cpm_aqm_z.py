#!/usr/bin/env python3
"""Extract Z/swe_hat/cpm_hat/aqm_hat for S0+CPM@Z+AQM@Z on simulation + observations, then
run the full comparison against S0 baseline and S0+CPM@Z: probes, joint CPM/AQM geometry,
retrieval, observational metrics, final table, and plots. Frozen inference only.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import FEATURE_Z_CLIP  # noqa: E402
from run_s0_cpm_aqm_aux_experiment import S0CPMAQMHeadZ  # noqa: E402

CMIP_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
PERCENTILE_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/data_cache")
OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
SPLIT_JSON = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/split.json")
EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S0_CPM_AQM_at_Z")

CPM_ERA5_PC_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_era5_reproduction_wy2021/era5_z500_pc1to6_daily.csv")
AQM_FULL_RECORD_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/aqm_index_loyo/aqm_index_full_record.csv")
UCLA_PERCENTILE_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/ucla_percentile.npz")
BASELINE_PRED_CSV = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/observational_predictions.csv")
BASELINE_LATENTS_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cnn_cpm_aqm_latent_geometry_v1/cnn_latents.npz")
CPM_Z_OUTPUTS_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_aux_transfer_experiment_v1/outputs_S0_CPM_Z.npz")
CPM_Z_OBS_SWE_COL = "S0_CPM_at_Z"
MATCHED_STATE_PRED_CSV = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/matched_climate_state_swe_diagnostic_v1/matched_climate_state_results_K20.csv")
MATCHED_STATE_REPORT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/matched_climate_state_swe_diagnostic_v1/full_report.json")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_aqm_aux_transfer_experiment_v1")
IN_CHANNELS_PER_MONTH = 38
K_ALL = (10, 20, 30)


def pearson_r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def spearman_rho(a, b):
    return pearson_r(rankdata(a), rankdata(b))


def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def wet_dry_accuracy(y_true, y_pred):
    return float(np.mean((y_true >= 0.5) == (y_pred >= 0.5)))


def tercile_accuracy(y_true, y_pred):
    def tercile(v):
        return np.where(v < 1 / 3, 0, np.where(v < 2 / 3, 1, 2))
    return float(np.mean(tercile(y_true) == tercile(y_pred)))


def ridge_fit(x_train, y_train, alpha=1.0):
    x_mean, x_std = x_train.mean(0), x_train.std(0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    xs = (x_train - x_mean) / x_std
    y_mean = float(y_train.mean())
    beta = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ (y_train - y_mean))
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "beta": beta}


def ridge_predict(model, x):
    return (x - model["x_mean"]) / model["x_std"] @ model["beta"] + model["y_mean"]


def knn_indices(query, pool, k):
    d2 = ((query[:, None, :] - pool[None, :, :]) ** 2).sum(axis=2)
    return np.argsort(d2, axis=1, kind="mergesort")[:, :k]


def load_observed_cpm() -> dict[int, float]:
    with CPM_ERA5_PC_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    by_wy: dict[int, list[float]] = {}
    for row in rows:
        month = int(row["date"][5:7])
        wy = int(row["water_year"])
        if month in (11, 12, 1, 2, 3):
            by_wy.setdefault(wy, []).append(float(row["PC3"]))
    return {wy: float(np.mean(v)) for wy, v in by_wy.items()}


def load_observed_aqm() -> dict[int, float]:
    with AQM_FULL_RECORD_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    out: dict[int, float] = {}
    for row in rows:
        if row["AQM_S2_NDJF_reprojected"] in (None, ""):
            continue
        out[int(row["water_year"])] = float(row["AQM_S2_NDJF_reprojected"])
    return out


def preprocess(physical, mask, feature_mu, feature_sigma) -> torch.Tensor:
    x = physical.astype(np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    stacked = np.concatenate([x, mask.astype(np.float32)], axis=2)
    assert np.isfinite(stacked).all()
    return torch.from_numpy(stacked)


def run_in_batches(model, inputs, device, batch_size=8):
    z_out, swe_out, cpm_out, aqm_out = [], [], [], []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            chunk = inputs[start : start + batch_size].to(device)
            out = model(chunk)
            z_out.append(out["z"].detach().cpu().numpy())
            swe_out.append(out["swe_hat"].detach().cpu().numpy())
            cpm_out.append(out["cpm_hat"].detach().cpu().numpy())
            aqm_out.append(out["aqm_hat"].detach().cpu().numpy())
    return {
        "z": np.concatenate(z_out, axis=0), "swe_hat": np.concatenate(swe_out, axis=0),
        "cpm_hat": np.concatenate(cpm_out, axis=0), "aqm_hat": np.concatenate(aqm_out, axis=0),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # -----------------------------------------------------------------
    # Extraction.
    # -----------------------------------------------------------------
    stats = np.load(EXP_DIR / "normalization_stats.npz")
    checkpoint = torch.load(EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model = S0CPMAQMHeadZ()
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    physical = np.load(CMIP_CACHE_DIR / "inputs_physical.npy")
    mask = np.load(CMIP_CACHE_DIR / "inputs_valid_mask.npy")
    sim_inputs = preprocess(physical, mask, stats["feature_mu"], stats["feature_sigma"])
    sim_out = run_in_batches(model, sim_inputs, device)

    obs_physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    obs_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        obs_manifest = list(csv.DictReader(fh))
    obs_water_years = [int(r["water_year"]) for r in obs_manifest]
    assert obs_water_years == list(range(1985, 2022))
    obs_inputs = preprocess(obs_physical, obs_mask, stats["feature_mu"], stats["feature_sigma"])
    obs_out = run_in_batches(model, obs_inputs, device)
    obs_out2 = run_in_batches(model, obs_inputs, device)
    assert np.allclose(obs_out["swe_hat"], obs_out2["swe_hat"], atol=1e-6), "non-deterministic"
    assert np.isfinite(obs_out["swe_hat"]).all() and np.isfinite(obs_out["cpm_hat"]).all() and np.isfinite(obs_out["aqm_hat"]).all()
    print("extraction + determinism/finite checks passed", flush=True)

    np.savez(OUTPUT_ROOT / "outputs_S0_CPM_AQM_Z.npz", z_sim=sim_out["z"], swe_hat_sim=sim_out["swe_hat"], cpm_hat_sim=sim_out["cpm_hat"], aqm_hat_sim=sim_out["aqm_hat"],
             z_obs=obs_out["z"], swe_hat_obs=obs_out["swe_hat"], cpm_hat_obs=obs_out["cpm_hat"], aqm_hat_obs=obs_out["aqm_hat"],
             obs_water_years=np.asarray(obs_water_years, dtype=np.int32), target_mu=stats["target_mu"], target_sigma=stats["target_sigma"])

    # -----------------------------------------------------------------
    # Load reference data.
    # -----------------------------------------------------------------
    with (CMIP_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    cmip_targets = np.load(CMIP_CACHE_DIR / "targets.npy")
    percentile_targets = np.load(PERCENTILE_CACHE_DIR / "targets.npy")
    n_samples = len(manifest)
    raw_cpm = cmip_targets[:, 1].astype(np.float64)
    raw_aqm = cmip_targets[:, 2].astype(np.float64)
    sim_swe_percentile = percentile_targets[:, 0].astype(np.float64)

    split = json.loads(SPLIT_JSON.read_text())
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    cpm_mu, cpm_sigma = float(raw_cpm[train_idx].mean()), float(raw_cpm[train_idx].std())
    aqm_mu, aqm_sigma = float(raw_aqm[train_idx].mean()), float(raw_aqm[train_idx].std())
    state = np.stack([(raw_cpm - cpm_mu) / cpm_sigma, (raw_aqm - aqm_mu) / aqm_sigma], axis=1)
    print(f"CPM train stats: mu={cpm_mu:.4f} sigma={cpm_sigma:.4f}; AQM train stats: mu={aqm_mu:.4f} sigma={aqm_sigma:.4f}", flush=True)

    water_years = list(range(1985, 2022))
    ucla = np.load(UCLA_PERCENTILE_NPZ)
    assert [int(y) for y in ucla["water_year"]] == water_years
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)
    observed_cpm_raw = load_observed_cpm()
    observed_aqm_raw = load_observed_aqm()
    assert all(wy in observed_cpm_raw for wy in water_years) and all(wy in observed_aqm_raw for wy in water_years)
    era5_cpm_vec = np.asarray([observed_cpm_raw[wy] for wy in water_years])
    era5_aqm_vec = np.asarray([observed_aqm_raw[wy] for wy in water_years])
    obs_state = np.stack([(era5_cpm_vec - cpm_mu) / cpm_sigma, (era5_aqm_vec - aqm_mu) / aqm_sigma], axis=1)

    with BASELINE_PRED_CSV.open() as fh:
        baseline_rows = list(csv.DictReader(fh))
    s0_baseline_swe_obs = np.asarray([float(r["S0_predicted_percentile"]) for r in baseline_rows])
    baseline_latents = np.load(BASELINE_LATENTS_NPZ)
    z_s0_sim = baseline_latents["z_s0_sim"].reshape(n_samples, -1)
    z_s0_obs = baseline_latents["z_s0_obs"].reshape(37, -1)

    cpm_z_data = np.load(CPM_Z_OUTPUTS_NPZ)
    z_cpmz_sim = cpm_z_data["z_sim"].reshape(n_samples, -1)
    z_cpmz_obs = cpm_z_data["z_obs"].reshape(37, -1)
    cpmz_swe_mu, cpmz_swe_sigma = float(cpm_z_data["target_mu"][0]), float(cpm_z_data["target_sigma"][0])
    cpmz_cpm_mu, cpmz_cpm_sigma = float(cpm_z_data["target_mu"][1]), float(cpm_z_data["target_sigma"][1])
    cpmz_swe_obs = cpm_z_data["swe_hat_obs"] * cpmz_swe_sigma + cpmz_swe_mu
    cpmz_cpm_obs = cpm_z_data["cpm_hat_obs"] * cpmz_cpm_sigma + cpmz_cpm_mu

    swe_mu, swe_sigma = float(stats["target_mu"][0]), float(stats["target_sigma"][0])
    cpm_head_mu, cpm_head_sigma = float(stats["target_mu"][1]), float(stats["target_sigma"][1])
    aqm_head_mu, aqm_head_sigma = float(stats["target_mu"][2]), float(stats["target_sigma"][2])
    new_swe_obs = obs_out["swe_hat"] * swe_sigma + swe_mu
    new_cpm_obs = obs_out["cpm_hat"] * cpm_head_sigma + cpm_head_mu
    new_aqm_obs = obs_out["aqm_hat"] * aqm_head_sigma + aqm_head_mu
    z_new_sim = sim_out["z"].reshape(n_samples, -1)
    z_new_obs = obs_out["z"].reshape(37, -1)

    models = {
        "S0_baseline": {"z_sim": z_s0_sim, "z_obs": z_s0_obs},
        "S0_CPM_at_Z": {"z_sim": z_cpmz_sim, "z_obs": z_cpmz_obs},
        "S0_CPM_AQM_at_Z": {"z_sim": z_new_sim, "z_obs": z_new_obs},
    }

    # -----------------------------------------------------------------
    # Probes (CPM and AQM separately).
    # -----------------------------------------------------------------
    probe_rows = []
    for name, cfg in models.items():
        z_sim, z_obs = cfg["z_sim"], cfg["z_obs"]
        for target_name, target_full, target_obs_full in (("CPM", raw_cpm, era5_cpm_vec), ("AQM", raw_aqm, era5_aqm_vec)):
            probe = ridge_fit(z_sim[train_idx], target_full[train_idx], alpha=1.0)
            for split_name, z_query, target_true in (("simulation_validation", z_sim[val_idx], target_full[val_idx]), ("observational", z_obs, target_obs_full)):
                pred = ridge_predict(probe, z_query)
                probe_rows.append({"model": name, "target": target_name, "eval_split": split_name, "pearson_r": pearson_r(target_true, pred), "r2": r2_score(target_true, pred), "rmse": rmse(target_true, pred)})
    with (OUTPUT_ROOT / "probe_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(probe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_rows)
    print(f"wrote {OUTPUT_ROOT / 'probe_metrics.csv'}", flush=True)

    # -----------------------------------------------------------------
    # Joint CPM/AQM geometry.
    # -----------------------------------------------------------------
    geometry_rows = []
    for name, cfg in models.items():
        z_sim = cfg["z_sim"]
        ca_dist = np.sqrt(((state[val_idx][:, None, :] - state[train_idx][None, :, :]) ** 2).sum(axis=2))
        latent_dist = np.sqrt(((z_sim[val_idx][:, None, :] - z_sim[train_idx][None, :, :]) ** 2).sum(axis=2))
        rho = spearman_rho(ca_dist.ravel(), latent_dist.ravel())
        geometry_rows.append({"model": name, "K": "", "spearman_rho_joint_geometry": rho, "mean_neighbor_overlap_fraction": ""})
        ca_nn_all = np.argsort(ca_dist, axis=1, kind="mergesort")
        latent_nn_all = np.argsort(latent_dist, axis=1, kind="mergesort")
        for k in K_ALL:
            fracs = [len(set(ca_nn_all[i, :k].tolist()) & set(latent_nn_all[i, :k].tolist())) / k for i in range(len(val_idx))]
            geometry_rows.append({"model": name, "K": k, "spearman_rho_joint_geometry": "", "mean_neighbor_overlap_fraction": float(np.mean(fracs))})
    with (OUTPUT_ROOT / "geometry_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(geometry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(geometry_rows)
    print(f"wrote {OUTPUT_ROOT / 'geometry_metrics.csv'}", flush=True)

    # -----------------------------------------------------------------
    # Retrieval.
    # -----------------------------------------------------------------
    retrieval_rows = []
    retrieval_obs_preds: dict[str, np.ndarray] = {}
    for name, cfg in models.items():
        z_sim, z_obs = cfg["z_sim"], cfg["z_obs"]
        for eval_name, query_z, query_true in (("simulation_validation", z_sim[val_idx], sim_swe_percentile[val_idx]), ("observational", z_obs, ucla_percentile)):
            for k in K_ALL:
                nn_idx = knn_indices(query_z, z_sim[train_idx], k)
                preds = np.median(sim_swe_percentile[train_idx][nn_idx], axis=1)
                retrieval_rows.append({"model": name, "eval_split": eval_name, "K": k, "pearson_r": pearson_r(query_true, preds), "spearman_rho": spearman_rho(query_true, preds), "wet_dry_accuracy": wet_dry_accuracy(query_true, preds), "rmse_percentile": rmse(query_true, preds)})
                if eval_name == "observational" and k == 20:
                    retrieval_obs_preds[name] = preds
    with (OUTPUT_ROOT / "retrieval_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(retrieval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(retrieval_rows)
    print(f"wrote {OUTPUT_ROOT / 'retrieval_metrics.csv'}", flush=True)

    # -----------------------------------------------------------------
    # Observational metrics (direct model heads) for the new model.
    # -----------------------------------------------------------------
    obs_rows = [{
        "model": "S0_CPM_AQM_at_Z",
        "swe_pearson_r": pearson_r(ucla_percentile, new_swe_obs), "swe_spearman_rho": spearman_rho(ucla_percentile, new_swe_obs),
        "swe_wet_dry_accuracy": wet_dry_accuracy(ucla_percentile, new_swe_obs), "swe_tercile_accuracy": tercile_accuracy(ucla_percentile, new_swe_obs),
        "swe_r2": r2_score(ucla_percentile, new_swe_obs), "swe_rmse": rmse(ucla_percentile, new_swe_obs), "swe_mae": mae(ucla_percentile, new_swe_obs),
        "cpm_pearson_r": pearson_r(era5_cpm_vec, new_cpm_obs), "cpm_r2": r2_score(era5_cpm_vec, new_cpm_obs), "cpm_rmse": rmse(era5_cpm_vec, new_cpm_obs),
        "aqm_pearson_r": pearson_r(era5_aqm_vec, new_aqm_obs), "aqm_r2": r2_score(era5_aqm_vec, new_aqm_obs), "aqm_rmse": rmse(era5_aqm_vec, new_aqm_obs),
    }]
    with (OUTPUT_ROOT / "observational_metrics_cpm_aqm_z.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_rows)
    print(f"wrote {OUTPUT_ROOT / 'observational_metrics_cpm_aqm_z.csv'}", flush=True)
    print(json.dumps(obs_rows, indent=2), flush=True)

    # -----------------------------------------------------------------
    # Training history copy.
    # -----------------------------------------------------------------
    (OUTPUT_ROOT / "training_history_cpm_aqm_z.csv").write_text((EXP_DIR / "history.csv").read_text())
    print(f"wrote {OUTPUT_ROOT / 'training_history_cpm_aqm_z.csv'}", flush=True)

    # -----------------------------------------------------------------
    # Matched-state benchmark (K=20) for reference row.
    # -----------------------------------------------------------------
    with MATCHED_STATE_PRED_CSV.open() as fh:
        matched_rows = list(csv.DictReader(fh))
    matched_median = np.asarray([float(r["matched_median_SWE_percentile"]) for r in matched_rows])
    matched_report = json.loads(MATCHED_STATE_REPORT.read_text())
    matched_k20 = next(r for r in matched_report["sensitivity_by_K"] if r["K"] == 20)

    # -----------------------------------------------------------------
    # Final comparison table.
    # -----------------------------------------------------------------
    def geom_val(name):
        return next(r["spearman_rho_joint_geometry"] for r in geometry_rows if r["model"] == name and r["spearman_rho_joint_geometry"] != "")

    def retrieval_val(name):
        return next(r["pearson_r"] for r in retrieval_rows if r["model"] == name and r["eval_split"] == "observational" and r["K"] == 20)

    def cpm_obs_r(name):
        return next(r["pearson_r"] for r in probe_rows if r["model"] == name and r["target"] == "CPM" and r["eval_split"] == "observational")

    def aqm_obs_r(name):
        return next(r["pearson_r"] for r in probe_rows if r["model"] == name and r["target"] == "AQM" and r["eval_split"] == "observational")

    s0_baseline_r = pearson_r(ucla_percentile, s0_baseline_swe_obs)
    s0_baseline_rho = spearman_rho(ucla_percentile, s0_baseline_swe_obs)
    s0_baseline_wd = wet_dry_accuracy(ucla_percentile, s0_baseline_swe_obs)
    cpmz_r = pearson_r(ucla_percentile, cpmz_swe_obs)
    cpmz_rho = spearman_rho(ucla_percentile, cpmz_swe_obs)
    cpmz_wd = wet_dry_accuracy(ucla_percentile, cpmz_swe_obs)
    cpmz_cpm_r = pearson_r(era5_cpm_vec, cpmz_cpm_obs)

    final_rows = [
        {"model": "S0_baseline", "obs_swe_r": s0_baseline_r, "obs_swe_rho": s0_baseline_rho, "obs_wet_dry": s0_baseline_wd, "obs_cpm_r": "", "obs_aqm_r": "", "joint_geometry_corr": geom_val("S0_baseline"), "obs_latent_retrieval_r": retrieval_val("S0_baseline")},
        {"model": "S0_CPM_at_Z", "obs_swe_r": cpmz_r, "obs_swe_rho": cpmz_rho, "obs_wet_dry": cpmz_wd, "obs_cpm_r": cpmz_cpm_r, "obs_aqm_r": "", "joint_geometry_corr": geom_val("S0_CPM_at_Z"), "obs_latent_retrieval_r": retrieval_val("S0_CPM_at_Z")},
        {"model": "S0_CPM_AQM_at_Z", "obs_swe_r": obs_rows[0]["swe_pearson_r"], "obs_swe_rho": obs_rows[0]["swe_spearman_rho"], "obs_wet_dry": obs_rows[0]["swe_wet_dry_accuracy"], "obs_cpm_r": obs_rows[0]["cpm_pearson_r"], "obs_aqm_r": obs_rows[0]["aqm_pearson_r"], "joint_geometry_corr": geom_val("S0_CPM_AQM_at_Z"), "obs_latent_retrieval_r": retrieval_val("S0_CPM_AQM_at_Z")},
        {"model": "CPM_AQM_matched_state_benchmark (non-neural, K=20)", "obs_swe_r": matched_k20["pearson_r"], "obs_swe_rho": matched_k20["spearman_rho"], "obs_wet_dry": matched_k20["wet_dry_accuracy"], "obs_cpm_r": "", "obs_aqm_r": "", "joint_geometry_corr": "", "obs_latent_retrieval_r": ""},
    ]
    with (OUTPUT_ROOT / "final_comparison.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"wrote {OUTPUT_ROOT / 'final_comparison.csv'}", flush=True)
    print(json.dumps(final_rows, indent=2, default=str), flush=True)

    # -----------------------------------------------------------------
    # Plots.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.2, label="UCLA observed")
    axis.plot(water_years, s0_baseline_swe_obs * 100, marker="o", markersize=3, color="tab:blue", label="S0 baseline")
    axis.plot(water_years, cpmz_swe_obs * 100, marker="o", markersize=3, color="tab:orange", label="S0 + CPM@Z")
    axis.plot(water_years, new_swe_obs * 100, marker="o", markersize=3, color="tab:red", label="S0 + CPM@Z + AQM@Z")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year"); axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_title("Observational SWE percentile: S0 baseline vs. CPM@Z vs. CPM+AQM@Z")
    axis.legend(fontsize=9); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_swe_timeseries.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, era5_cpm_vec, color="black", marker="o", markersize=4, linewidth=2.2, label="ERA5 CPM (observed)")
    axis.plot(water_years, cpmz_cpm_obs, marker="o", markersize=3, color="tab:orange", label="S0+CPM@Z prediction")
    axis.plot(water_years, new_cpm_obs, marker="o", markersize=3, color="tab:red", label="S0+CPM@Z+AQM@Z prediction")
    axis.set_xlabel("Water year"); axis.set_ylabel("CPM index")
    axis.set_title("Observational CPM comparison")
    axis.legend(fontsize=9); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_cpm_comparison.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, era5_aqm_vec, color="black", marker="o", markersize=4, linewidth=2.2, label="Observed AQM")
    axis.plot(water_years, new_aqm_obs, marker="o", markersize=3, color="tab:red", label="S0+CPM@Z+AQM@Z prediction")
    axis.set_xlabel("Water year"); axis.set_ylabel("AQM index")
    axis.set_title("Observational AQM comparison")
    axis.legend(fontsize=9); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_aqm_comparison.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), dpi=160)
    names = ["S0_baseline", "S0_CPM_at_Z", "S0_CPM_AQM_at_Z"]
    vals = [geom_val(n) for n in names]
    axis.bar(names, vals, color=["tab:blue", "tab:orange", "tab:red"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Spearman rho (joint CPM/AQM distance vs. latent distance)")
    axis.set_title("Joint CPM/AQM geometry preservation")
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "joint_geometry_comparison.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.2, label="UCLA observed")
    axis.plot(water_years, matched_median * 100, marker="o", markersize=3, color="tab:purple", label="CPM/AQM matched-state benchmark (K=20)")
    axis.plot(water_years, retrieval_obs_preds["S0_baseline"] * 100, marker="o", markersize=3, color="tab:blue", label="S0 latent retrieval")
    axis.plot(water_years, retrieval_obs_preds["S0_CPM_at_Z"] * 100, marker="o", markersize=3, color="tab:orange", label="S0+CPM@Z latent retrieval")
    axis.plot(water_years, retrieval_obs_preds["S0_CPM_AQM_at_Z"] * 100, marker="o", markersize=3, color="tab:red", label="S0+CPM+AQM@Z latent retrieval")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year"); axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_title("Observational latent-retrieval SWE percentile vs. CPM/AQM matched-state benchmark")
    axis.legend(fontsize=8); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_latent_retrieval_comparison.png"); plt.close(fig)

    print("done", flush=True)


if __name__ == "__main__":
    main()
