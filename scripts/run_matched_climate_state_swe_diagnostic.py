#!/usr/bin/env python3
"""Matched-climate-state diagnostic: for each observed water year, find the K nearest
simulated (CPM, AQM) climate states across all four CMIP parent models (regardless of
calendar year), and compare their simulated Sierra SWE percentile against the observed
UCLA Sierra SWE percentile. No CNN training or inference anywhere in this script.

CPM/AQM: reused directly from the existing CMIP6_label targets (columns 1,2 of
cmip6_cnn_architecture_screen_v1/data_cache/targets.npy - built by
scripts/build_cmip6_aux_labels.py) for simulations, and from the existing extended
observational reproductions for UCLA/ERA5-equivalent years:
  - CPM: artifacts/cpm_era5_reproduction_wy2021/era5_z500_pc1to6_daily.csv (PC3), Nov-Mar
    mean by water year - identical method/pattern to the original cpm_era5_reproduction,
    just extended through WY2021 (verified: only the analysis_period_end metadata differs).
  - AQM: artifacts/aqm_index_loyo/aqm_index_full_record.csv, AQM_S2_NDJF_reprojected
    column, which already documents itself as "AQM_S2_NDJF with fixed-loading projection
    fill for missing tail years" - the exact existing extension of the same AQM_S2_NDJF
    definition used by scripts/build_cmip6_aux_labels.py.

SWE percentile: reused directly from cmip6_cnn_swe_percentile_v1/data_cache/targets.npy
(column 0, built by scripts/build_parent_model_percentile_swe_target.py, train-only
empirical CDF per parent model) for simulations, and artifacts/s0_attention_swe_percentile_v1/
ucla_percentile.npz for observations - both already computed, not rebuilt here.
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
OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/matched_climate_state_swe_diagnostic_v1")

WATER_YEAR_START, WATER_YEAR_END = 1985, 2021
K_PRIMARY = 20
K_SENSITIVITY = (10, 30)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    def rankdata(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(1, len(x) + 1)
        return ranks
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def load_observed_cpm() -> dict[int, float]:
    with CPM_ERA5_PC_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    by_wy: dict[int, list[float]] = {}
    for row in rows:
        month = int(row["date"][5:7])
        wy = int(row["water_year"])
        if month in (11, 12, 1, 2, 3):
            by_wy.setdefault(wy, []).append(float(row["PC3"]))
    return {wy: float(np.mean(values)) for wy, values in by_wy.items()}


def load_observed_aqm() -> dict[int, float]:
    with AQM_FULL_RECORD_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    out: dict[int, float] = {}
    for row in rows:
        if row["AQM_S2_NDJF_reprojected"] in (None, ""):
            continue
        wy = int(row["water_year"])
        out[wy] = float(row["AQM_S2_NDJF_reprojected"])
    return out


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    with (CMIP_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    cmip_targets = np.load(CMIP_CACHE_DIR / "targets.npy")
    percentile_targets = np.load(PERCENTILE_CACHE_DIR / "targets.npy")
    assert cmip_targets.shape[0] == len(manifest) == percentile_targets.shape[0]

    raw_cpm = cmip_targets[:, 1].astype(np.float64)
    raw_aqm = cmip_targets[:, 2].astype(np.float64)
    sim_swe_percentile = percentile_targets[:, 0].astype(np.float64)
    sim_model_member = np.asarray([row["model_member"] for row in manifest])
    sim_row_year = np.asarray([int(row["row_year"]) for row in manifest])
    n_samples = len(manifest)

    for name, count in zip(*np.unique(sim_model_member, return_counts=True), strict=True):
        pass
    unique_models, model_counts = np.unique(sim_model_member, return_counts=True)
    print("=== simulation sample pool size by parent model ===", flush=True)
    for m, c in zip(unique_models, model_counts, strict=True):
        print(f"{m}: {int(c)} samples", flush=True)
    print(f"total simulation pool: {n_samples} samples", flush=True)

    split = json.loads(SPLIT_JSON.read_text())
    train_idx = np.asarray(split["train"], dtype=np.int64)
    cpm_mu, cpm_sigma = float(raw_cpm[train_idx].mean()), float(raw_cpm[train_idx].std())
    aqm_mu, aqm_sigma = float(raw_aqm[train_idx].mean()), float(raw_aqm[train_idx].std())
    print(f"training-derived standardization: CPM mu={cpm_mu:.4f} sigma={cpm_sigma:.4f}; AQM mu={aqm_mu:.4f} sigma={aqm_sigma:.4f}", flush=True)

    sim_cpm_std = (raw_cpm - cpm_mu) / cpm_sigma
    sim_aqm_std = (raw_aqm - aqm_mu) / aqm_sigma
    assert np.isfinite(sim_cpm_std).all() and np.isfinite(sim_aqm_std).all()

    observed_cpm_raw = load_observed_cpm()
    observed_aqm_raw = load_observed_aqm()
    water_years = list(range(WATER_YEAR_START, WATER_YEAR_END + 1))
    missing_cpm = [wy for wy in water_years if wy not in observed_cpm_raw]
    missing_aqm = [wy for wy in water_years if wy not in observed_aqm_raw]
    print(f"observed CPM coverage: {len(water_years) - len(missing_cpm)}/{len(water_years)} (missing: {missing_cpm})", flush=True)
    print(f"observed AQM coverage: {len(water_years) - len(missing_aqm)}/{len(water_years)} (missing: {missing_aqm})", flush=True)
    assert not missing_cpm and not missing_aqm, "observed CPM/AQM must cover all 37 water years"

    ucla = np.load(UCLA_PERCENTILE_NPZ)
    ucla_wy = [int(y) for y in ucla["water_year"]]
    assert ucla_wy == water_years
    ucla_mm = ucla["ucla_swe_mm"].astype(np.float64)
    ucla_percentile = ucla["ucla_percentile"].astype(np.float64)

    observed_cpm_std = np.asarray([(observed_cpm_raw[wy] - cpm_mu) / cpm_sigma for wy in water_years])
    observed_aqm_std = np.asarray([(observed_aqm_raw[wy] - aqm_mu) / aqm_sigma for wy in water_years])
    assert np.isfinite(observed_cpm_std).all() and np.isfinite(observed_aqm_std).all()

    def matched_stats(k: int):
        rows = []
        for i, wy in enumerate(water_years):
            d = np.sqrt((sim_cpm_std - observed_cpm_std[i]) ** 2 + (sim_aqm_std - observed_aqm_std[i]) ** 2)
            nearest = np.argsort(d, kind="mergesort")[:k]
            matched_pct = sim_swe_percentile[nearest]
            same_sign = np.mean((matched_pct >= 0.5) == (ucla_percentile[i] >= 0.5))
            rows.append({
                "water_year": wy,
                "observed_CPM": observed_cpm_raw[wy],
                "observed_AQM": observed_aqm_raw[wy],
                "observed_SWE_percentile": ucla_percentile[i],
                "observed_SWE_mm": ucla_mm[i],
                "matched_median_SWE_percentile": float(np.median(matched_pct)),
                "matched_q25_SWE_percentile": float(np.percentile(matched_pct, 25)),
                "matched_q75_SWE_percentile": float(np.percentile(matched_pct, 75)),
                "same_sign_fraction": float(same_sign),
                "mean_state_distance": float(d[nearest].mean()),
                "matched_ids": [f"{sim_model_member[j]}:{sim_row_year[j]}" for j in nearest],
            })
        return rows

    rows_k20 = matched_stats(K_PRIMARY)
    observed_arr = np.asarray([r["observed_SWE_percentile"] for r in rows_k20])
    median_arr = np.asarray([r["matched_median_SWE_percentile"] for r in rows_k20])

    main_csv = OUTPUT_ROOT / "matched_climate_state_results_K20.csv"
    with main_csv.open("w", newline="") as fh:
        fieldnames = ["water_year", "observed_CPM", "observed_AQM", "observed_SWE_percentile", "observed_SWE_mm", "matched_median_SWE_percentile", "matched_q25_SWE_percentile", "matched_q75_SWE_percentile", "same_sign_fraction", "mean_state_distance", "matched_ids"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_k20:
            row = dict(r)
            row["matched_ids"] = ";".join(row["matched_ids"])
            writer.writerow(row)
    print(f"wrote {main_csv}", flush=True)

    def aggregate(rows, k):
        obs = np.asarray([r["observed_SWE_percentile"] for r in rows])
        med = np.asarray([r["matched_median_SWE_percentile"] for r in rows])
        same_sign = np.asarray([r["same_sign_fraction"] for r in rows])
        return {
            "K": k,
            "pearson_r": pearson_r(obs, med),
            "spearman_rho": spearman_rho(obs, med),
            "wet_dry_accuracy": float(np.mean((obs >= 0.5) == (med >= 0.5))),
            "mean_same_sign_fraction": float(same_sign.mean()),
            "rmse_percentile": float(np.sqrt(np.mean((obs - med) ** 2))),
        }

    agg_k20 = aggregate(rows_k20, K_PRIMARY)
    sensitivity_rows = [agg_k20] + [aggregate(matched_stats(k), k) for k in K_SENSITIVITY]
    sens_csv = OUTPUT_ROOT / "sensitivity_metrics_by_K.csv"
    with sens_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sensitivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(f"wrote {sens_csv}", flush=True)
    print(json.dumps(sensitivity_rows, indent=2), flush=True)

    # Main plot.
    q25_arr = np.asarray([r["matched_q25_SWE_percentile"] for r in rows_k20]) * 100
    q75_arr = np.asarray([r["matched_q75_SWE_percentile"] for r in rows_k20]) * 100
    fig, axis = plt.subplots(figsize=(13, 6.3), dpi=160)
    axis.fill_between(water_years, q25_arr, q75_arr, color="tab:red", alpha=0.2, label="Matched simulated 25th-75th percentile")
    axis.plot(water_years, median_arr * 100, color="tab:red", marker="o", markersize=3, label="Matched simulated median (K=20)")
    axis.plot(water_years, observed_arr * 100, color="black", marker="o", markersize=4, linewidth=2.0, label="UCLA observed")
    axis.axhline(50.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Water year")
    axis.set_ylabel("Sierra SWE percentile (%)")
    axis.set_ylim(-5, 105)
    axis.set_title("Observed SWE vs. matched WUS-D3 climate-state SWE response")
    fig.text(0.5, 0.965, "Each observed year is matched to the 20 nearest simulated years in standardized CPM-AQM space", ha="center", fontsize=9, style="italic")
    axis.legend(fontsize=9, loc="upper right")
    axis.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = OUTPUT_ROOT / "matched_climate_state_swe_response.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"wrote {plot_path}", flush=True)

    full_report = {
        "cpm_code_reused": "scripts/build_cmip6_aux_labels.py:build_cpm_labels_from_cmip6 (simulation, via targets.npy col 1); artifacts/cpm_era5_reproduction_wy2021/era5_z500_pc1to6_daily.csv PC3 Nov-Mar mean (observations, same fixed EOF3 pattern extended through WY2021)",
        "aqm_code_reused": "scripts/build_cmip6_aux_labels.py:build_aqm_labels_from_cmip6 (simulation, via targets.npy col 2); artifacts/aqm_index_loyo/aqm_index_full_record.csv AQM_S2_NDJF_reprojected (observations, documented extension of the same AQM_S2_NDJF definition)",
        "swe_percentile_reused": "cmip6_cnn_swe_percentile_v1/data_cache/targets.npy col 0 (simulation, train-only empirical CDF per parent model); artifacts/s0_attention_swe_percentile_v1/ucla_percentile.npz (observations)",
        "simulation_pool_size_by_model": {str(m): int(c) for m, c in zip(unique_models, model_counts, strict=True)},
        "simulation_pool_total": n_samples,
        "standardization": {"cpm_mu": cpm_mu, "cpm_sigma": cpm_sigma, "aqm_mu": aqm_mu, "aqm_sigma": aqm_sigma, "source": "training-split-only, same split.json as CNN experiments"},
        "sensitivity_by_K": sensitivity_rows,
        "main_plot": str(plot_path),
        "main_csv": str(main_csv),
    }
    (OUTPUT_ROOT / "full_report.json").write_text(json.dumps(full_report, indent=2) + "\n")
    print(json.dumps(full_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
