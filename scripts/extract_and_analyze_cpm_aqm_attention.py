#!/usr/bin/env python3
"""Extract Z/swe_hat/cpm_hat/aqm_hat for the 3 attention-on-CPM+AQM@Z models (frozen,
partial, full freeze levels) on simulation + observations, then run the full comparison
against the S0+CPM+AQM@Z baseline (reused from cpm_aqm_aux_transfer_experiment_v1/):
probes, joint CPM/AQM geometry, retrieval, observational metrics, final table, and plots.
Frozen inference only - no training.
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
from run_cpm_aqm_attention_experiment import AttentionOnCPMAQMEncoder  # noqa: E402

CMIP_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
PERCENTILE_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/data_cache")
OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
SPLIT_JSON = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/split.json")

EXP_DIRS = {
    "frozen": Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/CPM_AQM_attention_frozen"),
    "partial": Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/CPM_AQM_attention_partial"),
    "full": Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/CPM_AQM_attention_full"),
}

CPM_ERA5_PC_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_era5_reproduction_wy2021/era5_z500_pc1to6_daily.csv")
AQM_FULL_RECORD_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/aqm_index_loyo/aqm_index_full_record.csv")
UCLA_PERCENTILE_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/ucla_percentile.npz")
BASELINE_CPM_AQM_Z_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_aqm_aux_transfer_experiment_v1/outputs_S0_CPM_AQM_Z.npz")
MATCHED_STATE_PRED_CSV = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/matched_climate_state_swe_diagnostic_v1/matched_climate_state_results_K20.csv")
MATCHED_STATE_REPORT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/matched_climate_state_swe_diagnostic_v1/full_report.json")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_aqm_attention_combination_v1")
K_ALL = (10, 20, 30)

MODEL_LABELS = {
    "S0_CPM_AQM_at_Z": "S0+CPM+AQM@Z baseline (no attention)",
    "attention_frozen": "CPM+AQM frozen encoder + attention",
    "attention_partial": "CPM+AQM partial unfreeze + attention",
    "attention_full": "CPM+AQM full end-to-end + attention",
}
TRAINABLE_SCOPE = {
    "S0_CPM_AQM_at_Z": "full encoder (original training)",
    "attention_frozen": "attention + swe_head only",
    "attention_partial": "down2, stage3, project, attention, swe_head, cpm_head, aqm_head",
    "attention_full": "everything",
}


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


def run_in_batches(model, inputs, device, use_cpm_aqm, batch_size=8):
    z_out, swe_out, cpm_out, aqm_out = [], [], [], []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            chunk = inputs[start : start + batch_size].to(device)
            out = model(chunk, use_cpm_aqm=use_cpm_aqm)
            z_out.append(out["z"].detach().cpu().numpy())
            swe_out.append(out["swe_hat"].detach().cpu().numpy())
            if use_cpm_aqm:
                cpm_out.append(out["cpm_hat"].detach().cpu().numpy())
                aqm_out.append(out["aqm_hat"].detach().cpu().numpy())
    result = {"z": np.concatenate(z_out, axis=0), "swe_hat": np.concatenate(swe_out, axis=0)}
    if use_cpm_aqm:
        result["cpm_hat"] = np.concatenate(cpm_out, axis=0)
        result["aqm_hat"] = np.concatenate(aqm_out, axis=0)
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # -----------------------------------------------------------------
    # Load reference data (shared across all models).
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

    water_years = list(range(1985, 2022))
    ucla = np.load(UCLA_PERCENTILE_NPZ)
    assert [int(y) for y in ucla["water_year"]] == water_years
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)
    observed_cpm_raw = load_observed_cpm()
    observed_aqm_raw = load_observed_aqm()
    assert all(wy in observed_cpm_raw for wy in water_years) and all(wy in observed_aqm_raw for wy in water_years)
    era5_cpm_vec = np.asarray([observed_cpm_raw[wy] for wy in water_years])
    era5_aqm_vec = np.asarray([observed_aqm_raw[wy] for wy in water_years])

    obs_physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    obs_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        obs_manifest = list(csv.DictReader(fh))
    obs_water_years = [int(r["water_year"]) for r in obs_manifest]
    assert obs_water_years == water_years

    physical = np.load(CMIP_CACHE_DIR / "inputs_physical.npy")
    mask = np.load(CMIP_CACHE_DIR / "inputs_valid_mask.npy")

    # -----------------------------------------------------------------
    # Baseline (reused, not re-extracted).
    # -----------------------------------------------------------------
    baseline_data = np.load(BASELINE_CPM_AQM_Z_NPZ)
    baseline_swe_mu, baseline_swe_sigma = float(baseline_data["target_mu"][0]), float(baseline_data["target_sigma"][0])
    baseline_cpm_mu, baseline_cpm_sigma = float(baseline_data["target_mu"][1]), float(baseline_data["target_sigma"][1])
    baseline_aqm_mu, baseline_aqm_sigma = float(baseline_data["target_mu"][2]), float(baseline_data["target_sigma"][2])
    models_z = {
        "S0_CPM_AQM_at_Z": {
            "z_sim": baseline_data["z_sim"].reshape(n_samples, -1),
            "z_obs": baseline_data["z_obs"].reshape(37, -1),
        }
    }
    obs_swe = {"S0_CPM_AQM_at_Z": baseline_data["swe_hat_obs"] * baseline_swe_sigma + baseline_swe_mu}
    obs_cpm = {"S0_CPM_AQM_at_Z": baseline_data["cpm_hat_obs"] * baseline_cpm_sigma + baseline_cpm_mu}
    obs_aqm = {"S0_CPM_AQM_at_Z": baseline_data["aqm_hat_obs"] * baseline_aqm_sigma + baseline_aqm_mu}

    # -----------------------------------------------------------------
    # Extraction for the 3 attention models.
    # -----------------------------------------------------------------
    for mode, exp_dir in EXP_DIRS.items():
        use_cpm_aqm = mode != "frozen"
        stats = np.load(exp_dir / "normalization_stats.npz")
        checkpoint = torch.load(exp_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        model = AttentionOnCPMAQMEncoder(mode)
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        sim_inputs = preprocess(physical, mask, stats["feature_mu"], stats["feature_sigma"])
        sim_out = run_in_batches(model, sim_inputs, device, use_cpm_aqm)

        obs_inputs = preprocess(obs_physical, obs_mask, stats["feature_mu"], stats["feature_sigma"])
        obs_out = run_in_batches(model, obs_inputs, device, use_cpm_aqm)
        obs_out2 = run_in_batches(model, obs_inputs, device, use_cpm_aqm)
        assert np.allclose(obs_out["swe_hat"], obs_out2["swe_hat"], atol=1e-6), f"{mode}: non-deterministic"
        assert np.isfinite(obs_out["swe_hat"]).all(), f"{mode}: non-finite swe"
        if use_cpm_aqm:
            assert np.isfinite(obs_out["cpm_hat"]).all() and np.isfinite(obs_out["aqm_hat"]).all(), f"{mode}: non-finite cpm/aqm"
        print(f"{mode}: extraction + determinism/finite checks passed", flush=True)

        save_kwargs = dict(
            z_sim=sim_out["z"], swe_hat_sim=sim_out["swe_hat"],
            z_obs=obs_out["z"], swe_hat_obs=obs_out["swe_hat"],
            obs_water_years=np.asarray(obs_water_years, dtype=np.int32),
            target_mu=stats["target_mu"], target_sigma=stats["target_sigma"],
        )
        if use_cpm_aqm:
            save_kwargs.update(cpm_hat_sim=sim_out["cpm_hat"], aqm_hat_sim=sim_out["aqm_hat"],
                                cpm_hat_obs=obs_out["cpm_hat"], aqm_hat_obs=obs_out["aqm_hat"])
        np.savez(OUTPUT_ROOT / f"outputs_attention_{mode}.npz", **save_kwargs)

        model_key = f"attention_{mode}"
        models_z[model_key] = {"z_sim": sim_out["z"].reshape(n_samples, -1), "z_obs": obs_out["z"].reshape(37, -1)}
        swe_mu, swe_sigma = float(stats["target_mu"][0]), float(stats["target_sigma"][0])
        obs_swe[model_key] = obs_out["swe_hat"] * swe_sigma + swe_mu
        if use_cpm_aqm:
            cpm_head_mu, cpm_head_sigma = float(stats["target_mu"][1]), float(stats["target_sigma"][1])
            aqm_head_mu, aqm_head_sigma = float(stats["target_mu"][2]), float(stats["target_sigma"][2])
            obs_cpm[model_key] = obs_out["cpm_hat"] * cpm_head_sigma + cpm_head_mu
            obs_aqm[model_key] = obs_out["aqm_hat"] * aqm_head_sigma + aqm_head_mu

        (OUTPUT_ROOT / f"training_history_{mode}_attention.csv").write_text((exp_dir / "history.csv").read_text())

    model_order = ["S0_CPM_AQM_at_Z", "attention_frozen", "attention_partial", "attention_full"]

    # -----------------------------------------------------------------
    # Probes (CPM and AQM separately), fit on sim-train, eval on sim-val + obs.
    # -----------------------------------------------------------------
    probe_rows = []
    for name in model_order:
        z_sim, z_obs = models_z[name]["z_sim"], models_z[name]["z_obs"]
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
    for name in model_order:
        z_sim = models_z[name]["z_sim"]
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
    # Latent nearest-neighbor SWE retrieval.
    # -----------------------------------------------------------------
    retrieval_rows = []
    retrieval_obs_preds: dict[str, np.ndarray] = {}
    for name in model_order:
        z_sim, z_obs = models_z[name]["z_sim"], models_z[name]["z_obs"]
        for eval_name, query_z, query_true in (("simulation_validation", z_sim[val_idx], sim_swe_percentile[val_idx]), ("observational", z_obs, ucla_percentile)):
            for k in K_ALL:
                nn_idx = knn_indices(query_z, z_sim[train_idx], k)
                preds = np.median(sim_swe_percentile[train_idx][nn_idx], axis=1)
                retrieval_rows.append({"model": name, "eval_split": eval_name, "K": k, "pearson_r": pearson_r(query_true, preds), "spearman_rho": spearman_rho(query_true, preds), "wet_dry_accuracy": wet_dry_accuracy(query_true, preds), "rmse_percentile": rmse(query_true, preds)})
                if eval_name == "observational" and k == 20:
                    retrieval_obs_preds[name] = preds
    with (OUTPUT_ROOT / "latent_retrieval_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(retrieval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(retrieval_rows)
    print(f"wrote {OUTPUT_ROOT / 'latent_retrieval_metrics.csv'}", flush=True)

    # -----------------------------------------------------------------
    # Direct-head observational metrics.
    # -----------------------------------------------------------------
    obs_rows = []
    for name in model_order:
        row = {
            "model": name,
            "swe_pearson_r": pearson_r(ucla_percentile, obs_swe[name]), "swe_spearman_rho": spearman_rho(ucla_percentile, obs_swe[name]),
            "swe_wet_dry_accuracy": wet_dry_accuracy(ucla_percentile, obs_swe[name]), "swe_tercile_accuracy": tercile_accuracy(ucla_percentile, obs_swe[name]),
            "swe_r2": r2_score(ucla_percentile, obs_swe[name]), "swe_rmse": rmse(ucla_percentile, obs_swe[name]), "swe_mae": mae(ucla_percentile, obs_swe[name]),
        }
        if name in obs_cpm:
            row.update(cpm_pearson_r=pearson_r(era5_cpm_vec, obs_cpm[name]), cpm_r2=r2_score(era5_cpm_vec, obs_cpm[name]), cpm_rmse=rmse(era5_cpm_vec, obs_cpm[name]))
            row.update(aqm_pearson_r=pearson_r(era5_aqm_vec, obs_aqm[name]), aqm_r2=r2_score(era5_aqm_vec, obs_aqm[name]), aqm_rmse=rmse(era5_aqm_vec, obs_aqm[name]))
        else:
            row.update(cpm_pearson_r="", cpm_r2="", cpm_rmse="", aqm_pearson_r="", aqm_r2="", aqm_rmse="")
        obs_rows.append(row)
    with (OUTPUT_ROOT / "observational_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_rows)
    print(f"wrote {OUTPUT_ROOT / 'observational_metrics.csv'}", flush=True)
    print(json.dumps(obs_rows, indent=2, default=str), flush=True)

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

    def sim_swe_r(name):
        return next(r["pearson_r"] for r in retrieval_rows if r["model"] == name and r["eval_split"] == "simulation_validation" and r["K"] == 20)

    obs_by_name = {r["model"]: r for r in obs_rows}
    final_rows = []
    for name in model_order:
        r = obs_by_name[name]
        final_rows.append({
            "model": MODEL_LABELS[name], "trainable_encoder_scope": TRAINABLE_SCOPE[name],
            "sim_swe_r": sim_swe_r(name), "obs_swe_r": r["swe_pearson_r"], "obs_swe_rho": r["swe_spearman_rho"],
            "obs_wet_dry": r["swe_wet_dry_accuracy"], "obs_latent_retrieval_r": retrieval_val(name),
            "joint_cpm_aqm_geometry": geom_val(name),
        })
    final_rows.append({
        "model": "CPM/AQM matched-state benchmark (non-neural, K=20)", "trainable_encoder_scope": "n/a",
        "sim_swe_r": "", "obs_swe_r": matched_k20["pearson_r"], "obs_swe_rho": matched_k20["spearman_rho"],
        "obs_wet_dry": matched_k20["wet_dry_accuracy"], "obs_latent_retrieval_r": "", "joint_cpm_aqm_geometry": "",
    })
    with (OUTPUT_ROOT / "final_comparison.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"wrote {OUTPUT_ROOT / 'final_comparison.csv'}", flush=True)
    print(json.dumps(final_rows, indent=2, default=str), flush=True)

    # -----------------------------------------------------------------
    # Plots.
    # -----------------------------------------------------------------
    colors = {"S0_CPM_AQM_at_Z": "tab:red", "attention_frozen": "tab:blue", "attention_partial": "tab:orange", "attention_full": "tab:green"}

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.2, label="UCLA observed")
    for name in model_order:
        axis.plot(water_years, obs_swe[name] * 100, marker="o", markersize=3, color=colors[name], label=MODEL_LABELS[name])
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year"); axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_title("Observational SWE percentile: CPM+AQM baseline vs. attention variants")
    axis.legend(fontsize=8); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_swe_timeseries.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    ucla_z = (ucla_percentile - ucla_percentile.mean()) / ucla_percentile.std()
    axis.plot(water_years, ucla_z, color="black", marker="o", markersize=4, linewidth=2.2, label="UCLA observed (standardized)")
    for name in model_order:
        v = obs_swe[name]
        vz = (v - v.mean()) / v.std()
        axis.plot(water_years, vz, marker="o", markersize=3, color=colors[name], label=f"{MODEL_LABELS[name]} (standardized)")
    axis.set_xlabel("Water year"); axis.set_ylabel("Standardized anomaly (shape only - not used for metrics)")
    axis.set_title("Shape-only standardized comparison (SCALE REMOVED - visual trend-matching only)")
    axis.legend(fontsize=8); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "shape_only_standardized_comparison.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=160)
    metric_labels = ["Pearson r", "Spearman rho", "Wet/dry acc"]
    x = np.arange(len(metric_labels))
    width = 0.2
    for i, name in enumerate(model_order):
        r = obs_by_name[name]
        vals = [r["swe_pearson_r"], r["swe_spearman_rho"], r["swe_wet_dry_accuracy"]]
        axis.bar(x + (i - 1.5) * width, vals, width, label=MODEL_LABELS[name], color=colors[name])
    axis.set_xticks(x); axis.set_xticklabels(metric_labels)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Score"); axis.set_title("Observational skill comparison (direct heads)")
    axis.legend(fontsize=8); axis.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_skill_comparison.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.2, label="UCLA observed")
    for name in model_order:
        axis.plot(water_years, retrieval_obs_preds[name] * 100, marker="o", markersize=3, color=colors[name], label=f"{MODEL_LABELS[name]} (latent retrieval)")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year"); axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_title("Observational latent-retrieval SWE percentile comparison (K=20)")
    axis.legend(fontsize=8); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "observational_latent_retrieval_comparison.png"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), dpi=160)
    names_disp = [MODEL_LABELS[n] for n in model_order]
    vals = [geom_val(n) for n in model_order]
    axis.bar(names_disp, vals, color=[colors[n] for n in model_order])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Spearman rho (joint CPM/AQM distance vs. latent distance)")
    axis.set_title("Joint CPM/AQM geometry preservation")
    axis.set_xticklabels(names_disp, rotation=20, ha="right")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "geometry_comparison.png"); plt.close(fig)

    print("done", flush=True)


if __name__ == "__main__":
    main()
