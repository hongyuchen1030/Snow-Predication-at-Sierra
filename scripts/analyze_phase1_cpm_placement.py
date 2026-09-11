#!/usr/bin/env python3
"""Phase 1 analysis: CPM decodability, CPM geometry preservation, and SWE-relevant latent
retrieval for S0 baseline, S0+CPM@Z, and S0+CPM@Stage2 - both in simulation and on the 37
observational years. No training happens here; reads already-extracted latents/predictions.
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
BASELINE_S0_PRED_CSV = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/observational_predictions.csv")
BASELINE_LATENTS_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cnn_cpm_aqm_latent_geometry_v1/cnn_latents.npz")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cpm_aux_transfer_experiment_v1")
K_ALL = (10, 20, 30)
K_PRIMARY = 20


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

    ucla = np.load(UCLA_PERCENTILE_NPZ)
    water_years = list(range(1985, 2022))
    assert [int(y) for y in ucla["water_year"]] == water_years
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)
    observed_cpm_raw = load_observed_cpm()
    assert all(wy in observed_cpm_raw for wy in water_years)
    obs_cpm_std = np.asarray([(observed_cpm_raw[wy] - cpm_mu) / cpm_sigma for wy in water_years])

    with BASELINE_S0_PRED_CSV.open() as fh:
        baseline_pred_rows = list(csv.DictReader(fh))
    baseline_swe_obs = np.asarray([float(r["S0_predicted_percentile"]) for r in baseline_pred_rows])

    baseline_latents = np.load(BASELINE_LATENTS_NPZ)
    z_baseline_sim = baseline_latents["z_s0_sim"].reshape(n_samples, -1)
    z_baseline_obs = baseline_latents["z_s0_obs"].reshape(37, -1)

    model_configs = {
        "S0_baseline": {"z_sim": z_baseline_sim, "z_obs": z_baseline_obs, "has_cpm_head": False, "swe_obs": baseline_swe_obs, "cpm_obs": None},
    }
    for name, path in (("S0_CPM_at_Z", "outputs_S0_CPM_Z.npz"), ("S0_CPM_at_Stage2", "outputs_S0_CPM_Stage2.npz")):
        data = np.load(OUTPUT_ROOT / path)
        swe_mu, swe_sigma = float(data["target_mu"][0]), float(data["target_sigma"][0])
        cpm_head_mu, cpm_head_sigma = float(data["target_mu"][1]), float(data["target_sigma"][1])
        model_configs[name] = {
            "z_sim": data["z_sim"].reshape(n_samples, -1), "z_obs": data["z_obs"].reshape(37, -1),
            "has_cpm_head": True,
            "swe_obs": data["swe_hat_obs"] * swe_sigma + swe_mu,
            "cpm_obs": data["cpm_hat_obs"] * cpm_head_sigma + cpm_head_mu,
        }

    probe_rows, geometry_rows, retrieval_rows, obs_swe_rows, obs_cpm_rows = [], [], [], [], []
    for name, cfg in model_configs.items():
        z_sim, z_obs = cfg["z_sim"], cfg["z_obs"]

        # --- CPM decodability probe ---
        probe = ridge_fit(z_sim[train_idx], raw_cpm[train_idx], alpha=1.0)
        for split_name, z_query, target_true in (("simulation_validation", z_sim[val_idx], raw_cpm[val_idx]), ("observational", z_obs, np.asarray([observed_cpm_raw[wy] for wy in water_years]))):
            pred = ridge_predict(probe, z_query)
            probe_rows.append({"model": name, "eval_split": split_name, "pearson_r": pearson_r(target_true, pred), "r2": r2_score(target_true, pred), "rmse": rmse(target_true, pred)})

        # --- CPM geometry preservation (1D CPM distance) ---
        cpm_dist = np.abs(cpm_std[val_idx][:, None] - cpm_std[train_idx][None, :])
        latent_dist = np.sqrt(((z_sim[val_idx][:, None, :] - z_sim[train_idx][None, :, :]) ** 2).sum(axis=2))
        rho = spearman_rho(cpm_dist.ravel(), latent_dist.ravel())
        geometry_rows.append({"model": name, "spearman_rho_cpm_distance_correlation": rho})
        cpm_nn_all = np.argsort(cpm_dist, axis=1, kind="mergesort")
        latent_nn_all = np.argsort(latent_dist, axis=1, kind="mergesort")
        for k in K_ALL:
            fracs = [len(set(cpm_nn_all[i, :k].tolist()) & set(latent_nn_all[i, :k].tolist())) / k for i in range(len(val_idx))]
            geometry_rows.append({"model": name, "K": k, "mean_neighbor_overlap_fraction": float(np.mean(fracs))})

        # --- Retrieval: latent NN median SWE percentile, val and obs ---
        for eval_name, query_z, query_true in (("simulation_validation", z_sim[val_idx], sim_swe_percentile[val_idx]), ("observational", z_obs, ucla_percentile)):
            for k in K_ALL:
                nn_idx = knn_indices(query_z, z_sim[train_idx], k)
                preds = np.median(sim_swe_percentile[train_idx][nn_idx], axis=1)
                retrieval_rows.append({
                    "model": name, "eval_split": eval_name, "K": k,
                    "pearson_r": pearson_r(query_true, preds), "spearman_rho": spearman_rho(query_true, preds),
                    "wet_dry_accuracy": wet_dry_accuracy(query_true, preds), "rmse_percentile": rmse(query_true, preds),
                })

        # --- Observational SWE metrics (direct model prediction, not retrieval) ---
        swe_obs = cfg["swe_obs"]
        obs_swe_rows.append({
            "model": name, "pearson_r": pearson_r(ucla_percentile, swe_obs), "spearman_rho": spearman_rho(ucla_percentile, swe_obs),
            "wet_dry_accuracy": wet_dry_accuracy(ucla_percentile, swe_obs), "tercile_accuracy": tercile_accuracy(ucla_percentile, swe_obs),
            "r2": r2_score(ucla_percentile, swe_obs), "rmse": rmse(ucla_percentile, swe_obs), "mae": mae(ucla_percentile, swe_obs),
        })

        # --- Observational CPM metrics (direct model prediction, only if model has a CPM head) ---
        if cfg["has_cpm_head"]:
            cpm_obs_pred = cfg["cpm_obs"]
            cpm_obs_true = np.asarray([observed_cpm_raw[wy] for wy in water_years])
            obs_cpm_rows.append({"model": name, "pearson_r": pearson_r(cpm_obs_true, cpm_obs_pred), "r2": r2_score(cpm_obs_true, cpm_obs_pred), "rmse": rmse(cpm_obs_true, cpm_obs_pred)})

    for prefix, rows in (("phase1_probe_metrics.csv", probe_rows), ("phase1_geometry_metrics.csv", geometry_rows), ("phase1_retrieval_metrics.csv", retrieval_rows)):
        path = OUTPUT_ROOT / prefix
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}", flush=True)

    obs_swe_csv = OUTPUT_ROOT / "phase1_observational_swe_metrics.csv"
    with obs_swe_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_swe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_swe_rows)
    print(f"wrote {obs_swe_csv}", flush=True)

    obs_cpm_csv = OUTPUT_ROOT / "phase1_observational_cpm_metrics.csv"
    with obs_cpm_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_cpm_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_cpm_rows)
    print(f"wrote {obs_cpm_csv}", flush=True)

    print(json.dumps({"probe": probe_rows, "geometry": geometry_rows, "retrieval": retrieval_rows, "obs_swe": obs_swe_rows, "obs_cpm": obs_cpm_rows}, indent=2), flush=True)

    # -----------------------------------------------------------------
    # Plots.
    # -----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(13, 6), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.0, label="UCLA observed")
    for name, color in (("S0_baseline", "tab:blue"), ("S0_CPM_at_Z", "tab:orange"), ("S0_CPM_at_Stage2", "tab:green")):
        axis.plot(water_years, model_configs[name]["swe_obs"] * 100, marker="o", markersize=3, color=color, label=name)
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year"); axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_title("Phase 1: observational SWE percentile - S0 baseline vs. CPM placements")
    axis.legend(fontsize=9); axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "phase1_observational_swe_timeseries.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6), dpi=160)
    era5_cpm_vec = np.asarray([observed_cpm_raw[wy] for wy in water_years])
    axis.plot(water_years, era5_cpm_vec, color="black", marker="o", markersize=4, linewidth=2.0, label="ERA5 CPM (observed)")
    axis.plot(water_years, model_configs["S0_CPM_at_Z"]["cpm_obs"], marker="o", markersize=3, color="tab:orange", label="CPM@Z prediction")
    axis.plot(water_years, model_configs["S0_CPM_at_Stage2"]["cpm_obs"], marker="o", markersize=3, color="tab:green", label="CPM@Stage2 prediction")
    axis.set_xlabel("Water year"); axis.set_ylabel("CPM index")
    axis.set_title("Phase 1: observational CPM - ERA5 vs. predicted")
    axis.legend(fontsize=9); axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "phase1_observational_cpm_timeseries.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), dpi=160)
    names = ["S0_baseline", "S0_CPM_at_Z", "S0_CPM_at_Stage2"]
    geom_by_model = {name: next(r["spearman_rho_cpm_distance_correlation"] for r in geometry_rows if r["model"] == name and "spearman_rho_cpm_distance_correlation" in r) for name in names}
    axis.bar(names, [geom_by_model[n] for n in names], color=["tab:blue", "tab:orange", "tab:green"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Spearman rho (CPM distance vs. latent distance)")
    axis.set_title("Phase 1: CPM geometry preservation")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "phase1_cpm_geometry_comparison.png")
    plt.close(fig)

    print("done", flush=True)


if __name__ == "__main__":
    main()
