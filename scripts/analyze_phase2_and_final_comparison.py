#!/usr/bin/env python3
"""Phase 2 analysis (attention baseline vs. attention+CPM@Z) and final 6-model comparison
assembly. No training; reads already-extracted latents/predictions/metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMIP_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
PERCENTILE_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/data_cache")
SPLIT_JSON = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/split.json")
CPM_ERA5_PC_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_era5_reproduction_wy2021/era5_z500_pc1to6_daily.csv")
UCLA_PERCENTILE_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/ucla_percentile.npz")
BASELINE_PRED_CSV = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/observational_predictions.csv")
BASELINE_LATENTS_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cnn_cpm_aqm_latent_geometry_v1/cnn_latents.npz")
S0_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S0_static_cnn_swe_only")
S1_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S1_static_latent_self_attention")
FROZEN_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/frozen_S0_attention")
MATCHED_STATE_REPORT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/matched_climate_state_swe_diagnostic_v1/full_report.json")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_aux_transfer_experiment_v1")
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


def main() -> None:
    with (CMIP_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    cmip_targets = np.load(CMIP_CACHE_DIR / "targets.npy")
    percentile_targets = np.load(PERCENTILE_CACHE_DIR / "targets.npy")
    n_samples = len(manifest)
    raw_cpm = cmip_targets[:, 1].astype(np.float64)
    sim_swe_percentile = percentile_targets[:, 0].astype(np.float64)

    split = json.loads(SPLIT_JSON.read_text())
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    cpm_mu, cpm_sigma = float(raw_cpm[train_idx].mean()), float(raw_cpm[train_idx].std())
    cpm_std = (raw_cpm - cpm_mu) / cpm_sigma

    water_years = list(range(1985, 2022))
    ucla = np.load(UCLA_PERCENTILE_NPZ)
    assert [int(y) for y in ucla["water_year"]] == water_years
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)
    observed_cpm_raw = load_observed_cpm()
    era5_cpm_vec = np.asarray([observed_cpm_raw[wy] for wy in water_years])

    with BASELINE_PRED_CSV.open() as fh:
        baseline_rows = list(csv.DictReader(fh))
    attention_baseline_swe_obs = np.asarray([float(r["end_to_end_attention_predicted_percentile"]) for r in baseline_rows])
    s0_baseline_swe_obs = np.asarray([float(r["S0_predicted_percentile"]) for r in baseline_rows])
    frozen_swe_obs = np.asarray([float(r["frozen_attention_predicted_percentile"]) for r in baseline_rows])

    baseline_latents = np.load(BASELINE_LATENTS_NPZ)
    z_attn_baseline_sim = baseline_latents["z_s1_sim"].reshape(n_samples, -1)
    z_attn_baseline_obs = baseline_latents["z_s1_obs"].reshape(37, -1)

    cpm_data = np.load(OUTPUT_ROOT / "outputs_attention_CPM_Z.npz")
    swe_mu, swe_sigma = float(cpm_data["target_mu"][0]), float(cpm_data["target_sigma"][0])
    cpm_head_mu, cpm_head_sigma = float(cpm_data["target_mu"][1]), float(cpm_data["target_sigma"][1])
    z_attn_cpm_sim = cpm_data["z_sim"].reshape(n_samples, -1)
    z_attn_cpm_obs = cpm_data["z_obs"].reshape(37, -1)
    attn_cpm_swe_obs = cpm_data["swe_hat_obs"] * swe_sigma + swe_mu
    attn_cpm_cpm_obs = cpm_data["cpm_hat_obs"] * cpm_head_sigma + cpm_head_mu

    phase2_models = {
        "attention_baseline": {"z_sim": z_attn_baseline_sim, "z_obs": z_attn_baseline_obs, "has_cpm_head": False, "swe_obs": attention_baseline_swe_obs, "cpm_obs": None},
        "attention_CPM_at_Z": {"z_sim": z_attn_cpm_sim, "z_obs": z_attn_cpm_obs, "has_cpm_head": True, "swe_obs": attn_cpm_swe_obs, "cpm_obs": attn_cpm_cpm_obs},
    }

    probe_geom_rows = []
    obs_swe_rows = []
    obs_cpm_rows = []
    for name, cfg in phase2_models.items():
        z_sim, z_obs = cfg["z_sim"], cfg["z_obs"]
        probe = ridge_fit(z_sim[train_idx], raw_cpm[train_idx], alpha=1.0)
        for split_name, z_query, target_true in (("simulation_validation", z_sim[val_idx], raw_cpm[val_idx]), ("observational", z_obs, era5_cpm_vec)):
            pred = ridge_predict(probe, z_query)
            probe_geom_rows.append({"model": name, "eval_split": split_name, "metric": "probe", "pearson_r": pearson_r(target_true, pred), "r2": r2_score(target_true, pred), "rmse": rmse(target_true, pred)})

        cpm_dist = np.abs(cpm_std[val_idx][:, None] - cpm_std[train_idx][None, :])
        latent_dist = np.sqrt(((z_sim[val_idx][:, None, :] - z_sim[train_idx][None, :, :]) ** 2).sum(axis=2))
        rho = spearman_rho(cpm_dist.ravel(), latent_dist.ravel())
        probe_geom_rows.append({"model": name, "eval_split": "simulation_validation", "metric": "geometry_spearman_rho", "pearson_r": rho, "r2": None, "rmse": None})

        for eval_name, query_z, query_true in (("simulation_validation", z_sim[val_idx], sim_swe_percentile[val_idx]), ("observational", z_obs, ucla_percentile)):
            nn_idx = knn_indices(query_z, z_sim[train_idx], 20)
            preds = np.median(sim_swe_percentile[train_idx][nn_idx], axis=1)
            probe_geom_rows.append({"model": name, "eval_split": eval_name, "metric": "retrieval_K20_pearson_r", "pearson_r": pearson_r(query_true, preds), "r2": None, "rmse": None})

        swe_obs = cfg["swe_obs"]
        obs_swe_rows.append({
            "model": name, "pearson_r": pearson_r(ucla_percentile, swe_obs), "spearman_rho": spearman_rho(ucla_percentile, swe_obs),
            "wet_dry_accuracy": wet_dry_accuracy(ucla_percentile, swe_obs), "tercile_accuracy": tercile_accuracy(ucla_percentile, swe_obs),
            "r2": r2_score(ucla_percentile, swe_obs), "rmse": rmse(ucla_percentile, swe_obs),
        })
        if cfg["has_cpm_head"]:
            obs_cpm_rows.append({"model": name, "pearson_r": pearson_r(era5_cpm_vec, cfg["cpm_obs"]), "r2": r2_score(era5_cpm_vec, cfg["cpm_obs"]), "rmse": rmse(era5_cpm_vec, cfg["cpm_obs"])})

    phase2_train_csv = OUTPUT_ROOT / "phase2_training_metrics.csv"
    with (S1_EXP_DIR / "metrics_summary.json").open() if False else open("/dev/null") as _:
        pass
    with phase2_train_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "best_epoch", "val_swe_r2", "val_swe_pearson_r", "val_swe_spearman_rho", "val_swe_wet_dry_accuracy", "val_cpm_pearson_r"])
        s1_summary = json.loads((S1_EXP_DIR / "metrics_summary.json").read_text())
        writer.writerow(["attention_baseline", s1_summary["best_epoch"], s1_summary["best_metrics"]["swe_r2"], s1_summary["best_metrics"]["swe_pearson_r"], s1_summary["best_metrics"]["swe_spearman_rho"], s1_summary["best_metrics"]["swe_wet_dry_accuracy"], ""])
        cpm_summary = json.loads(Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/attention_CPM_at_Z/metrics_summary.json").read_text())
        bm = cpm_summary["best_metrics"]
        writer.writerow(["attention_CPM_at_Z", cpm_summary["best_epoch"], bm["val_swe_r2"], bm["val_swe_pearson_r"], bm["val_swe_spearman_rho"], bm["val_swe_wet_dry_accuracy"], bm.get("val_cpm_pearson_r", "")])
    print(f"wrote {phase2_train_csv}", flush=True)

    phase2_obs_csv = OUTPUT_ROOT / "phase2_observational_metrics.csv"
    with phase2_obs_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_swe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_swe_rows)
    print(f"wrote {phase2_obs_csv}", flush=True)

    phase2_probe_geom_csv = OUTPUT_ROOT / "phase2_probe_geometry_metrics.csv"
    with phase2_probe_geom_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(probe_geom_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_geom_rows)
    print(f"wrote {phase2_probe_geom_csv}", flush=True)
    print(json.dumps({"phase2_obs_swe": obs_swe_rows, "phase2_obs_cpm": obs_cpm_rows, "phase2_probe_geom": probe_geom_rows}, indent=2), flush=True)

    # -----------------------------------------------------------------
    # Final 6-model comparison table.
    # -----------------------------------------------------------------
    s0_summary = json.loads((S0_EXP_DIR / "metrics_summary.json").read_text())
    s0_cpm_z_summary = json.loads(Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S0_CPM_at_Z/metrics_summary.json").read_text())
    frozen_summary = json.loads((FROZEN_EXP_DIR / "metrics_summary.json").read_text())
    s1_summary = json.loads((S1_EXP_DIR / "metrics_summary.json").read_text())
    attn_cpm_summary = json.loads(Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/attention_CPM_at_Z/metrics_summary.json").read_text())

    with open(OUTPUT_ROOT / "phase1_observational_swe_metrics.csv") as fh:
        p1_obs_swe = {r["model"]: r for r in csv.DictReader(fh)}
    with open(OUTPUT_ROOT / "phase1_observational_cpm_metrics.csv") as fh:
        p1_obs_cpm = {r["model"]: r for r in csv.DictReader(fh)}
    with open(OUTPUT_ROOT / "phase1_geometry_metrics.csv") as fh:
        p1_geom_rows_all = list(csv.DictReader(fh))
        p1_geom = {r["model"]: r["spearman_rho_cpm_distance_correlation"] for r in p1_geom_rows_all if r.get("spearman_rho_cpm_distance_correlation")}
    with open(OUTPUT_ROOT / "phase1_retrieval_metrics.csv") as fh:
        p1_retrieval_all = list(csv.DictReader(fh))
        p1_retrieval_obs_k20 = {r["model"]: r["pearson_r"] for r in p1_retrieval_all if r["eval_split"] == "observational" and r["K"] == "20"}

    frozen_obs_r = pearson_r(ucla_percentile, frozen_swe_obs)
    frozen_obs_rho = spearman_rho(ucla_percentile, frozen_swe_obs)
    frozen_obs_wd = wet_dry_accuracy(ucla_percentile, frozen_swe_obs)

    attn_geom = next((r["pearson_r"] for r in probe_geom_rows if r["model"] == "attention_baseline" and r["metric"] == "geometry_spearman_rho"), None)
    attn_cpm_geom = next((r["pearson_r"] for r in probe_geom_rows if r["model"] == "attention_CPM_at_Z" and r["metric"] == "geometry_spearman_rho"), None)
    attn_retrieval_obs = next((r["pearson_r"] for r in probe_geom_rows if r["model"] == "attention_baseline" and r["metric"] == "retrieval_K20_pearson_r" and r["eval_split"] == "observational"), None)
    attn_cpm_retrieval_obs = next((r["pearson_r"] for r in probe_geom_rows if r["model"] == "attention_CPM_at_Z" and r["metric"] == "retrieval_K20_pearson_r" and r["eval_split"] == "observational"), None)

    final_rows = [
        {"model": "S0_baseline", "cpm_placement": "none", "sim_swe_r": s0_summary["best_metrics"]["swe_pearson_r"], "sim_swe_rho": s0_summary["best_metrics"]["swe_spearman_rho"], "sim_wet_dry": s0_summary["best_metrics"]["swe_wet_dry_accuracy"], "obs_swe_r": p1_obs_swe["S0_baseline"]["pearson_r"], "obs_swe_rho": p1_obs_swe["S0_baseline"]["spearman_rho"], "obs_wet_dry": p1_obs_swe["S0_baseline"]["wet_dry_accuracy"], "obs_cpm_r": "", "cpm_geometry_corr": p1_geom.get("S0_baseline", ""), "obs_latent_retrieval_r": p1_retrieval_obs_k20.get("S0_baseline", "")},
        {"model": "S0_CPM_at_Z", "cpm_placement": "Z", "sim_swe_r": s0_cpm_z_summary["best_metrics"]["val_swe_pearson_r"], "sim_swe_rho": s0_cpm_z_summary["best_metrics"]["val_swe_spearman_rho"], "sim_wet_dry": s0_cpm_z_summary["best_metrics"]["val_swe_wet_dry_accuracy"], "obs_swe_r": p1_obs_swe["S0_CPM_at_Z"]["pearson_r"], "obs_swe_rho": p1_obs_swe["S0_CPM_at_Z"]["spearman_rho"], "obs_wet_dry": p1_obs_swe["S0_CPM_at_Z"]["wet_dry_accuracy"], "obs_cpm_r": p1_obs_cpm["S0_CPM_at_Z"]["pearson_r"], "cpm_geometry_corr": p1_geom.get("S0_CPM_at_Z", ""), "obs_latent_retrieval_r": p1_retrieval_obs_k20.get("S0_CPM_at_Z", "")},
        {"model": "S0_CPM_at_Stage2", "cpm_placement": "Stage2", "sim_swe_r": None, "sim_swe_rho": None, "sim_wet_dry": None, "obs_swe_r": p1_obs_swe["S0_CPM_at_Stage2"]["pearson_r"], "obs_swe_rho": p1_obs_swe["S0_CPM_at_Stage2"]["spearman_rho"], "obs_wet_dry": p1_obs_swe["S0_CPM_at_Stage2"]["wet_dry_accuracy"], "obs_cpm_r": p1_obs_cpm["S0_CPM_at_Stage2"]["pearson_r"], "cpm_geometry_corr": p1_geom.get("S0_CPM_at_Stage2", ""), "obs_latent_retrieval_r": p1_retrieval_obs_k20.get("S0_CPM_at_Stage2", "")},
        {"model": "Frozen_S0_attention_reference", "cpm_placement": "none (frozen encoder)", "sim_swe_r": frozen_summary["best_metrics"]["swe_pearson_r"], "sim_swe_rho": frozen_summary["best_metrics"]["swe_spearman_rho"], "sim_wet_dry": frozen_summary["best_metrics"]["swe_wet_dry_accuracy"], "obs_swe_r": frozen_obs_r, "obs_swe_rho": frozen_obs_rho, "obs_wet_dry": frozen_obs_wd, "obs_cpm_r": "", "cpm_geometry_corr": "", "obs_latent_retrieval_r": ""},
        {"model": "attention_baseline", "cpm_placement": "none", "sim_swe_r": s1_summary["best_metrics"]["swe_pearson_r"], "sim_swe_rho": s1_summary["best_metrics"]["swe_spearman_rho"], "sim_wet_dry": s1_summary["best_metrics"]["swe_wet_dry_accuracy"], "obs_swe_r": obs_swe_rows[0]["pearson_r"], "obs_swe_rho": obs_swe_rows[0]["spearman_rho"], "obs_wet_dry": obs_swe_rows[0]["wet_dry_accuracy"], "obs_cpm_r": "", "cpm_geometry_corr": attn_geom, "obs_latent_retrieval_r": attn_retrieval_obs},
        {"model": "attention_CPM_at_Z", "cpm_placement": "Z", "sim_swe_r": attn_cpm_summary["best_metrics"]["val_swe_pearson_r"], "sim_swe_rho": attn_cpm_summary["best_metrics"]["val_swe_spearman_rho"], "sim_wet_dry": attn_cpm_summary["best_metrics"]["val_swe_wet_dry_accuracy"], "obs_swe_r": obs_swe_rows[1]["pearson_r"], "obs_swe_rho": obs_swe_rows[1]["spearman_rho"], "obs_wet_dry": obs_swe_rows[1]["wet_dry_accuracy"], "obs_cpm_r": obs_cpm_rows[0]["pearson_r"], "cpm_geometry_corr": attn_cpm_geom, "obs_latent_retrieval_r": attn_cpm_retrieval_obs},
    ]
    final_csv = OUTPUT_ROOT / "final_comparison.csv"
    with final_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"wrote {final_csv}", flush=True)
    print(json.dumps(final_rows, indent=2, default=str), flush=True)

    # -----------------------------------------------------------------
    # Final plots.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.2, label="UCLA observed")
    axis.plot(water_years, s0_baseline_swe_obs * 100, marker="o", markersize=3, color="tab:blue", label="S0 baseline")
    s0_cpm_z_data = np.load(OUTPUT_ROOT / "outputs_S0_CPM_Z.npz")
    s0_cpm_z_swe_obs = s0_cpm_z_data["swe_hat_obs"] * float(s0_cpm_z_data["target_sigma"][0]) + float(s0_cpm_z_data["target_mu"][0])
    axis.plot(water_years, s0_cpm_z_swe_obs * 100, marker="o", markersize=3, color="tab:orange", label="Winning S0+CPM@Z")
    axis.plot(water_years, frozen_swe_obs * 100, marker="o", markersize=3, color="tab:purple", label="Frozen-S0+attention (reference)")
    axis.plot(water_years, attention_baseline_swe_obs * 100, marker="o", markersize=3, color="tab:green", label="End-to-end attention")
    axis.plot(water_years, attn_cpm_swe_obs * 100, marker="o", markersize=3, color="tab:red", label="End-to-end attention+CPM@Z")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year"); axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_title("Final observational SWE comparison: all 6 models")
    axis.legend(fontsize=8, ncol=2); axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "final_observational_swe_comparison.png")
    plt.close(fig)
    print(f"wrote {OUTPUT_ROOT / 'final_observational_swe_comparison.png'}", flush=True)

    fig, axis = plt.subplots(figsize=(11, 5.5), dpi=160)
    names = [r["model"] for r in final_rows]
    x_pos = np.arange(len(names))
    width = 0.25
    obs_r = [float(r["obs_swe_r"]) for r in final_rows]
    obs_rho = [float(r["obs_swe_rho"]) for r in final_rows]
    obs_wd = [float(r["obs_wet_dry"]) for r in final_rows]
    axis.bar(x_pos - width, obs_r, width, label="Obs SWE Pearson r", color="tab:blue")
    axis.bar(x_pos, obs_rho, width, label="Obs SWE Spearman rho", color="tab:orange")
    axis.bar(x_pos + width, obs_wd, width, label="Obs SWE wet/dry acc", color="tab:green")
    obs_cpm_vals = [float(r["obs_cpm_r"]) if r["obs_cpm_r"] not in ("", None) else np.nan for r in final_rows]
    axis.scatter(x_pos, obs_cpm_vals, color="tab:red", marker="D", s=50, label="Obs CPM Pearson r (where applicable)", zorder=5)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
    axis.set_xticks(x_pos); axis.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axis.set_title("Final skill-metric comparison (observational)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "final_skill_metric_comparison.png")
    plt.close(fig)
    print(f"wrote {OUTPUT_ROOT / 'final_skill_metric_comparison.png'}", flush=True)

    # Extend CPM geometry comparison bar plot to include attention models.
    fig, axis = plt.subplots(figsize=(9, 5), dpi=160)
    geom_names = ["S0_baseline", "S0_CPM_at_Z", "S0_CPM_at_Stage2", "attention_baseline", "attention_CPM_at_Z"]
    geom_vals = [float(p1_geom.get(n, attn_geom if n == "attention_baseline" else attn_cpm_geom)) for n in geom_names]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    axis.bar(geom_names, geom_vals, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Spearman rho (CPM distance vs. latent distance)")
    axis.set_title("CPM geometry preservation, all 5 CNN encoders")
    axis.set_xticklabels(geom_names, rotation=20, ha="right", fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "cpm_geometry_comparison_all_models.png")
    plt.close(fig)
    print(f"wrote {OUTPUT_ROOT / 'cpm_geometry_comparison_all_models.png'}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
