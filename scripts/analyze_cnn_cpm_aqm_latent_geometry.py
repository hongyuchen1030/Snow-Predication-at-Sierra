#!/usr/bin/env python3
"""CPU-side analysis: ridge decodability probes, CPM/AQM-geometry preservation, and
nearest-neighbor retrieval skill comparison (CPM/AQM space vs. S0 latent vs.
end-to-end-attention latent), for both simulation validation and the 37 observational
years. Reads latents already extracted by extract_percentile_cnn_latents.py - no neural
network forward pass happens in this script.
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
AQM_FULL_RECORD_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/aqm_index_loyo/aqm_index_full_record.csv")
UCLA_PERCENTILE_NPZ = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1/ucla_percentile.npz")
OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cnn_cpm_aqm_latent_geometry_v1")
LATENTS_NPZ = OUTPUT_ROOT / "cnn_latents.npz"

WATER_YEAR_START, WATER_YEAR_END = 1985, 2021
K_PRIMARY = 20
K_ALL = (10, 20, 30)


# ---------------------------------------------------------------------------
# Metric helpers.
# ---------------------------------------------------------------------------
def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_r(rankdata(a), rankdata(b))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def wet_dry_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true >= 0.5) == (y_pred >= 0.5)))


def ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> dict:
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    xs = (x_train - x_mean) / x_std
    y_mean = float(y_train.mean())
    yc = y_train - y_mean
    n_features = xs.shape[1]
    beta = np.linalg.solve(xs.T @ xs + alpha * np.eye(n_features), xs.T @ yc)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "beta": beta}


def ridge_predict(model: dict, x: np.ndarray) -> np.ndarray:
    xs = (x - model["x_mean"]) / model["x_std"]
    return xs @ model["beta"] + model["y_mean"]


def pca_2d(x_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    components = vt[:2]
    return (x_all - mean) @ components.T


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


def knn_indices(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    """query: (Nq, D), pool: (Np, D) -> (Nq, k) indices into pool, nearest first."""
    d2 = ((query[:, None, :] - pool[None, :, :]) ** 2).sum(axis=2)
    return np.argsort(d2, axis=1, kind="mergesort")[:, :k]


def main() -> None:
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
    print(f"train n={len(train_idx)} val n={len(val_idx)} total n={n_samples}", flush=True)

    cpm_mu, cpm_sigma = float(raw_cpm[train_idx].mean()), float(raw_cpm[train_idx].std())
    aqm_mu, aqm_sigma = float(raw_aqm[train_idx].mean()), float(raw_aqm[train_idx].std())
    state = np.stack([(raw_cpm - cpm_mu) / cpm_sigma, (raw_aqm - aqm_mu) / aqm_sigma], axis=1)

    latents = np.load(LATENTS_NPZ)
    z_s0_sim = latents["z_s0_sim"].reshape(n_samples, -1)
    z_s1_sim = latents["z_s1_sim"].reshape(n_samples, -1)
    z_s0_obs = latents["z_s0_obs"]
    z_s1_obs = latents["z_s1_obs"]
    obs_water_years = [int(y) for y in latents["obs_water_years"]]
    frozen_max_diff = float(latents["frozen_identity_max_diff"])
    frozen_identity_verified = bool(latents["frozen_identity_digest_match"])
    z_s0_obs_flat = z_s0_obs.reshape(len(obs_water_years), -1)
    z_s1_obs_flat = z_s1_obs.reshape(len(obs_water_years), -1)
    print(f"z_s0_sim shape={z_s0_sim.shape} z_s1_sim shape={z_s1_sim.shape}", flush=True)
    print(f"frozen-S0+attention encoder identity: max_diff={frozen_max_diff} verified={frozen_identity_verified}", flush=True)

    # -----------------------------------------------------------------
    # STEP 1: ridge decodability probes, Z -> CPM and Z -> AQM.
    # -----------------------------------------------------------------
    probe_rows = []
    for enc_name, z in (("S0", z_s0_sim), ("end_to_end_attention", z_s1_sim)):
        for target_name, target in (("CPM", raw_cpm), ("AQM", raw_aqm)):
            model = ridge_fit(z[train_idx], target[train_idx], alpha=1.0)
            pred_val = ridge_predict(model, z[val_idx])
            probe_rows.append({
                "encoder": enc_name, "target": target_name,
                "pearson_r": pearson_r(target[val_idx], pred_val),
                "r2": r2_score(target[val_idx], pred_val),
                "rmse": rmse(target[val_idx], pred_val),
            })
    probe_csv = OUTPUT_ROOT / "probe_metrics.csv"
    with probe_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(probe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_rows)
    print(f"wrote {probe_csv}", flush=True)
    for row in probe_rows:
        print(row, flush=True)

    # -----------------------------------------------------------------
    # STEP 2: geometry preservation - Spearman on pairwise distances + KNN overlap.
    # -----------------------------------------------------------------
    geometry_rows = []
    overlap_rows = []
    pca_data = {}
    for enc_name, z in (("S0", z_s0_sim), ("end_to_end_attention", z_s1_sim)):
        state_dist = np.sqrt(((state[val_idx][:, None, :] - state[train_idx][None, :, :]) ** 2).sum(axis=2))
        latent_dist = np.sqrt(((z[val_idx][:, None, :] - z[train_idx][None, :, :]) ** 2).sum(axis=2))
        rho = spearman_rho(state_dist.ravel(), latent_dist.ravel())
        geometry_rows.append({"encoder": enc_name, "spearman_rho_distance_correlation": rho})

        state_nn_all = np.argsort(state_dist, axis=1, kind="mergesort")
        latent_nn_all = np.argsort(latent_dist, axis=1, kind="mergesort")
        for k in K_ALL:
            state_nn = state_nn_all[:, :k]
            latent_nn = latent_nn_all[:, :k]
            fractions = [len(set(state_nn[i].tolist()) & set(latent_nn[i].tolist())) / k for i in range(len(val_idx))]
            overlap_rows.append({"encoder": enc_name, "K": k, "mean_neighbor_overlap_fraction": float(np.mean(fractions))})

        pca_data[enc_name] = pca_2d(z[train_idx], z)

    geometry_csv = OUTPUT_ROOT / "geometry_metrics.csv"
    with geometry_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(geometry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(geometry_rows)
    print(f"wrote {geometry_csv}", flush=True)
    for row in geometry_rows:
        print(row, flush=True)

    overlap_csv = OUTPUT_ROOT / "neighbor_overlap_by_k.csv"
    with overlap_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(overlap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overlap_rows)
    print(f"wrote {overlap_csv}", flush=True)
    for row in overlap_rows:
        print(row, flush=True)

    # PCA plots.
    for enc_name in ("S0", "end_to_end_attention"):
        coords = pca_data[enc_name]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=160)
        for axis, (color_vals, label) in zip(axes, ((raw_cpm, "CPM"), (raw_aqm, "AQM"), (sim_swe_percentile, "SWE percentile")), strict=True):
            sc = axis.scatter(coords[:, 0], coords[:, 1], c=color_vals, cmap="RdBu_r" if label != "SWE percentile" else "viridis", s=18)
            axis.set_title(f"{enc_name} latent PCA, colored by {label}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            fig.colorbar(sc, ax=axis, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / f"pca_{enc_name}_latent.png")
        plt.close(fig)
        print(f"wrote {OUTPUT_ROOT / f'pca_{enc_name}_latent.png'}", flush=True)

    # -----------------------------------------------------------------
    # STEP 3: simulation-side retrieval skill comparison.
    # -----------------------------------------------------------------
    sim_retrieval_rows = []

    def retrieve_and_score(query_space: np.ndarray, pool_space: np.ndarray, pool_targets: np.ndarray, query_targets: np.ndarray, k: int) -> dict:
        nn_idx = knn_indices(query_space, pool_space, k)
        preds = np.median(pool_targets[nn_idx], axis=1)
        return {
            "pearson_r": pearson_r(query_targets, preds),
            "spearman_rho": spearman_rho(query_targets, preds),
            "wet_dry_accuracy": wet_dry_accuracy(query_targets, preds),
            "rmse_percentile": rmse(query_targets, preds),
        }

    for method_name, query_space, pool_space in (
        ("CPM_AQM", state[val_idx], state[train_idx]),
        ("S0_latent", z_s0_sim[val_idx], z_s0_sim[train_idx]),
        ("end_to_end_attention_latent", z_s1_sim[val_idx], z_s1_sim[train_idx]),
    ):
        for k in K_ALL:
            metrics = retrieve_and_score(query_space, pool_space, sim_swe_percentile[train_idx], sim_swe_percentile[val_idx], k)
            sim_retrieval_rows.append({"method": method_name, "K": k, **metrics})

    sim_retrieval_csv = OUTPUT_ROOT / "simulation_retrieval_metrics.csv"
    with sim_retrieval_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sim_retrieval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sim_retrieval_rows)
    print(f"wrote {sim_retrieval_csv}", flush=True)
    for row in sim_retrieval_rows:
        print(row, flush=True)

    # -----------------------------------------------------------------
    # OBSERVATIONAL EXTENSION.
    # -----------------------------------------------------------------
    ucla = np.load(UCLA_PERCENTILE_NPZ)
    ucla_wy = [int(y) for y in ucla["water_year"]]
    water_years = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
    assert ucla_wy == water_years
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)

    observed_cpm_raw = load_observed_cpm()
    observed_aqm_raw = load_observed_aqm()
    assert all(wy in observed_cpm_raw for wy in water_years)
    assert all(wy in observed_aqm_raw for wy in water_years)
    obs_state = np.stack([
        np.asarray([(observed_cpm_raw[wy] - cpm_mu) / cpm_sigma for wy in water_years]),
        np.asarray([(observed_aqm_raw[wy] - aqm_mu) / aqm_sigma for wy in water_years]),
    ], axis=1)
    assert obs_water_years == water_years

    obs_retrieval_rows = []
    obs_predictions: dict[str, np.ndarray] = {}
    for method_name, query_space, pool_space in (
        ("CPM_AQM", obs_state, state[train_idx]),
        ("S0_latent", z_s0_obs_flat, z_s0_sim[train_idx]),
        ("end_to_end_attention_latent", z_s1_obs_flat, z_s1_sim[train_idx]),
    ):
        for k in K_ALL:
            nn_idx = knn_indices(query_space, pool_space, k)
            preds = np.median(sim_swe_percentile[train_idx][nn_idx], axis=1)
            metrics = {
                "pearson_r": pearson_r(ucla_percentile, preds),
                "spearman_rho": spearman_rho(ucla_percentile, preds),
                "wet_dry_accuracy": wet_dry_accuracy(ucla_percentile, preds),
                "rmse_percentile": rmse(ucla_percentile, preds),
            }
            obs_retrieval_rows.append({"method": method_name, "K": k, **metrics})
            if k == K_PRIMARY:
                obs_predictions[method_name] = preds

    obs_retrieval_csv = OUTPUT_ROOT / "observational_retrieval_metrics.csv"
    with obs_retrieval_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_retrieval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_retrieval_rows)
    print(f"wrote {obs_retrieval_csv}", flush=True)
    for row in obs_retrieval_rows:
        print(row, flush=True)

    obs_pred_csv = OUTPUT_ROOT / "observational_retrieval_predictions.csv"
    with obs_pred_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["water_year", "UCLA_percentile", "CPM_AQM_matched_median", "S0_latent_matched_median", "end_to_end_attention_latent_matched_median"])
        for i, wy in enumerate(water_years):
            writer.writerow([wy, f"{ucla_percentile[i]:.6f}", f"{obs_predictions['CPM_AQM'][i]:.6f}", f"{obs_predictions['S0_latent'][i]:.6f}", f"{obs_predictions['end_to_end_attention_latent'][i]:.6f}"])
    print(f"wrote {obs_pred_csv}", flush=True)

    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.plot(water_years, ucla_percentile * 100, color="black", marker="o", markersize=4, linewidth=2.0, label="UCLA observed")
    axis.plot(water_years, obs_predictions["CPM_AQM"] * 100, color="tab:red", marker="o", markersize=3, label="CPM/AQM matched-state median (K=20)")
    axis.plot(water_years, obs_predictions["S0_latent"] * 100, color="tab:blue", marker="o", markersize=3, label="S0-latent matched-state median (K=20)")
    axis.plot(water_years, obs_predictions["end_to_end_attention_latent"] * 100, color="tab:green", marker="o", markersize=3, label="End-to-end-attention-latent matched-state median (K=20)")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year")
    axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_ylim(-5, 105)
    axis.set_title("Observed vs. matched-state SWE: CPM/AQM space vs. CNN latent spaces")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    obs_plot_path = OUTPUT_ROOT / "observational_matched_state_comparison.png"
    fig.savefig(obs_plot_path)
    plt.close(fig)
    print(f"wrote {obs_plot_path}", flush=True)

    full_report = {
        "frozen_attention_encoder_identity": {"max_diff": frozen_max_diff, "verified_identical_to_S0": frozen_identity_verified},
        "probe_metrics": probe_rows,
        "geometry_metrics": geometry_rows,
        "neighbor_overlap_by_k": overlap_rows,
        "simulation_retrieval_metrics": sim_retrieval_rows,
        "observational_retrieval_metrics": obs_retrieval_rows,
        "note_on_observational_pool": "Per this task's explicit instruction, the observational nearest-neighbor reference pool is the 318 simulation TRAINING samples only, not the full 393-sample train+val pool used in the earlier standalone matched_climate_state_swe_diagnostic_v1 (which used all 393). CPM/AQM metrics therefore differ slightly between the two analyses; this is expected and documented.",
    }
    (OUTPUT_ROOT / "full_report.json").write_text(json.dumps(full_report, indent=2) + "\n")
    print(json.dumps(full_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
