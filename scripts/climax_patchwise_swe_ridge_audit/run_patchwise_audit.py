#!/usr/bin/env python
"""Patch-local frozen-ClimaX SWE audit: independent Ridge probes per spatial
patch, strict nested LOYO, 1000-permutation empirical null, BH-FDR, spatial
coherence (connected components + Moran's I). No PCA, no attention, no
Captum/SHAP, no CPM/AQM, no joint-patch model.

Uses the validated exact closed-form nested-ridge-LOO engine
(fast_ridge_loyo.py) instead of brute-force refitting -- this is required to
make 2048 patches x 1000 permutations tractable; see that module's docstring
and validate_fast_ridge_loyo.py for the equivalence proof, which must PASS
before this script is trusted.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SCRIPT_DIR = "/global/u1/h/hyvchen/Snow-Predication-at-Sierra/scripts/climax_patchwise_swe_ridge_audit"
sys.path.insert(0, SCRIPT_DIR)
from fast_ridge_loyo import ALPHA_GRID, nested_loyo_fast  # noqa: E402
from patch_geometry import N_LAT_PATCH, N_LON_PATCH, N_PATCHES, build_patch_metadata  # noqa: E402

LATENT_DIR = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents"
OUT_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/climax_patchwise_swe_ridge_audit"
PRED_DIR = os.path.join(OUT_ROOT, "predictions")
PLOT_DIR = os.path.join(OUT_ROOT, "plots")

N_PERMUTATIONS = 1000
RANDOM_SEED = 20260909
WATER_YEARS = list(range(1985, 2022))


def ensure_dirs():
    for d in (PRED_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, _ = scipy_stats.pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    sign_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {"r": float(r) if np.isfinite(r) else 0.0, "R2": float(r2), "RMSE": rmse, "MAE": mae,
            "sign_accuracy": sign_acc}


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def connected_components_8neighbor_wrap(mask_2d: np.ndarray) -> tuple[np.ndarray, int]:
    """8-neighbor connectivity on the (32,64) patch grid, with longitude
    wraparound (col 63 adjacent to col 0); latitude does not wrap."""
    n_rows, n_cols = mask_2d.shape
    labels = -np.ones((n_rows, n_cols), dtype=int)
    current_label = 0
    for r0 in range(n_rows):
        for c0 in range(n_cols):
            if not mask_2d[r0, c0] or labels[r0, c0] != -1:
                continue
            stack = [(r0, c0)]
            labels[r0, c0] = current_label
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = r + dr
                        nc = (c + dc) % n_cols  # longitude wraparound
                        if 0 <= nr < n_rows and mask_2d[nr, nc] and labels[nr, nc] == -1:
                            labels[nr, nc] = current_label
                            stack.append((nr, nc))
            current_label += 1
    return labels, current_label


def build_adjacency_weights(n_rows: int, n_cols: int) -> np.ndarray:
    """Row-standardized queen (8-neighbor) adjacency for Moran's I, patch
    index p = row*n_cols+col, with longitude wraparound."""
    n = n_rows * n_cols
    W = np.zeros((n, n), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            p = r * n_cols + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, (c + dc) % n_cols
                    if 0 <= nr < n_rows:
                        W[p, nr * n_cols + nc] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    n = len(values)
    z = values - values.mean()
    num = z @ (W @ z)
    den = np.sum(z ** 2)
    S0 = W.sum()
    if den == 0 or S0 == 0:
        return 0.0
    return float((n / S0) * (num / den))


def main():
    t_start = time.time()
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    print("=== Step 1: load H, SWE target; verify 37-year alignment ===", flush=True)
    H = np.load(os.path.join(LATENT_DIR, "H_march25_obs_latents.npy"))
    assert H.shape == (37, 2048, 1024), H.shape
    paired = np.load(os.path.join(LATENT_DIR, "paired_dataset.npz"), allow_pickle=True)
    wy = paired["water_year"].astype(int)
    assert list(wy) == WATER_YEARS, "H and SWE target water-year order must match exactly"
    y_swe = paired["swe_apr1_standardized"].astype(np.float64)
    print(f"  H shape={H.shape}, y_swe shape={y_swe.shape}, water_years={wy[0]}-{wy[-1]} (n={len(wy)})", flush=True)
    print(f"  NaN in H: {int(np.isnan(H).sum())}, Inf in H: {int(np.isinf(H).sum())}", flush=True)
    assert np.isnan(H).sum() == 0 and np.isinf(H).sum() == 0

    print("=== Step 2: build patch geographic metadata (verified token ordering) ===", flush=True)
    meta_df = build_patch_metadata()
    meta_df.to_csv(os.path.join(OUT_ROOT, "patch_metadata.csv"), index=False)
    n_sierra = int(meta_df["intersects_sierra"].sum())
    print(f"  {len(meta_df)} patches; {n_sierra} intersect the Sierra box (35-42N, -122.5..-118)", flush=True)

    print(f"=== Step 3: generate {N_PERMUTATIONS} label permutations (seed={RANDOM_SEED}) ===", flush=True)
    perm_indices = np.stack([rng.permutation(37) for _ in range(N_PERMUTATIONS)], axis=0)  # (1000, 37)
    Y_full = np.empty((37, 1 + N_PERMUTATIONS), dtype=np.float64)
    Y_full[:, 0] = y_swe
    for k in range(N_PERMUTATIONS):
        Y_full[:, 1 + k] = y_swe[perm_indices[k]]

    print("=== Step 4: patch-wise nested LOYO Ridge, real + all permutations, all 2048 patches ===", flush=True)
    real_metrics = []
    real_preds_all = np.empty((2048, 37), dtype=np.float32)
    perm_r = np.empty((2048, N_PERMUTATIONS), dtype=np.float32)
    perm_r2 = np.empty((2048, N_PERMUTATIONS), dtype=np.float32)

    t_loop = time.time()
    for p in range(N_PATCHES):
        X_p = H[:, p, :].astype(np.float64)
        out = nested_loyo_fast(X_p, Y_full, ALPHA_GRID)  # (37, 1001)

        y_pred_real = out[:, 0]
        real_preds_all[p] = y_pred_real.astype(np.float32)
        m = compute_metrics(y_swe, y_pred_real)
        m["patch_id"] = p
        real_metrics.append(m)

        for k in range(N_PERMUTATIONS):
            y_true_perm = Y_full[:, 1 + k]
            y_pred_perm = out[:, 1 + k]
            ss_tot = np.sum((y_true_perm - y_true_perm.mean()) ** 2)
            r, _ = scipy_stats.pearsonr(y_true_perm, y_pred_perm)
            r = r if np.isfinite(r) else 0.0
            r2 = 1 - np.sum((y_true_perm - y_pred_perm) ** 2) / ss_tot if ss_tot > 0 else np.nan
            perm_r[p, k] = r
            perm_r2[p, k] = r2

        if (p + 1) % 200 == 0 or p == 0 or p == N_PATCHES - 1:
            elapsed = time.time() - t_loop
            rate = (p + 1) / elapsed
            eta = (N_PATCHES - p - 1) / rate if rate > 0 else float("nan")
            print(f"  patch {p + 1}/{N_PATCHES} done, elapsed={elapsed:.1f}s, "
                  f"rate={rate:.2f} patches/s, ETA={eta:.1f}s", flush=True)

    print(f"  Patch sweep total time: {time.time() - t_loop:.1f}s", flush=True)

    real_df = pd.DataFrame(real_metrics).merge(meta_df, on="patch_id")

    print("=== Step 5: empirical p-values (two-sided, |r| primary statistic) and BH-FDR ===", flush=True)
    abs_r_real = real_df["r"].to_numpy(dtype=np.float64)
    abs_r_perm = np.abs(perm_r)  # (2048, 1000)
    pvals = np.empty(N_PATCHES)
    for p in range(N_PATCHES):
        pvals[p] = (1 + np.sum(abs_r_perm[p] >= abs(abs_r_real[p]))) / (N_PERMUTATIONS + 1)
    real_df["empirical_p"] = pvals
    real_df["fdr_q"] = benjamini_hochberg(pvals)

    n_uncorrected_p05 = int((pvals < 0.05).sum())
    n_fdr_q05 = int((real_df["fdr_q"] < 0.05).sum())
    n_fdr_q10 = int((real_df["fdr_q"] < 0.10).sum())

    print(f"  uncorrected p<0.05: {n_uncorrected_p05}", flush=True)
    print(f"  FDR q<0.05: {n_fdr_q05}", flush=True)
    print(f"  FDR q<0.10: {n_fdr_q10}", flush=True)

    print("=== Step 6: family-wise max-statistic permutation test ===", flush=True)
    null_max_r = perm_r.max(axis=0)          # (1000,)
    null_max_abs_r = np.abs(perm_r).max(axis=0)
    null_max_r2 = np.nanmax(perm_r2, axis=0)

    real_max_r = float(real_df["r"].max())
    real_max_abs_r = float(real_df["r"].abs().max())
    real_max_r2 = float(real_df["R2"].max())

    fwe_p_max_r = (1 + np.sum(null_max_r >= real_max_r)) / (N_PERMUTATIONS + 1)
    fwe_p_max_abs_r = (1 + np.sum(null_max_abs_r >= real_max_abs_r)) / (N_PERMUTATIONS + 1)
    fwe_p_max_r2 = (1 + np.sum(null_max_r2 >= real_max_r2)) / (N_PERMUTATIONS + 1)

    print(f"  real max r={real_max_r:.4f}, null max-r mean={null_max_r.mean():.4f}, "
          f"FWE p={fwe_p_max_r:.4f}", flush=True)
    print(f"  real max |r|={real_max_abs_r:.4f}, null max-|r| mean={null_max_abs_r.mean():.4f}, "
          f"FWE p={fwe_p_max_abs_r:.4f}", flush=True)
    print(f"  real max R2={real_max_r2:.4f}, null max-R2 mean={null_max_r2.mean():.4f}, "
          f"FWE p={fwe_p_max_r2:.4f}", flush=True)

    print("=== Step 7: spatial coherence (connected components + Moran's I) ===", flush=True)
    if n_fdr_q05 > 0:
        sig_mask_flat = (real_df.sort_values("patch_id")["fdr_q"] < 0.05).to_numpy()
        sig_basis = "FDR q<0.05"
    else:
        thresh = real_df["r"].abs().quantile(0.95)
        sig_mask_flat = (real_df.sort_values("patch_id")["r"].abs() >= thresh).to_numpy()
        sig_basis = f"exploratory top-5% |r| (threshold |r|>={thresh:.4f}); NO patch survived FDR q<0.05"
    sig_mask_2d = sig_mask_flat.reshape(N_LAT_PATCH, N_LON_PATCH)

    labels_2d, n_components = connected_components_8neighbor_wrap(sig_mask_2d)
    component_sizes = []
    component_locations = []
    for comp_id in range(n_components):
        comp_mask = labels_2d == comp_id
        size = int(comp_mask.sum())
        rows, cols = np.where(comp_mask)
        patch_ids = [int(r * N_LON_PATCH + c) for r, c in zip(rows, cols)]
        sub = meta_df[meta_df["patch_id"].isin(patch_ids)]
        component_sizes.append(size)
        component_locations.append({
            "component_id": comp_id, "size": size,
            "lat_range": [float(sub["lat_min"].min()), float(sub["lat_max"].max())],
            "lon_range": [float(sub["lon_min"].min()), float(sub["lon_max"].max())],
            "patch_ids": patch_ids,
        })
    largest_component_size = max(component_sizes) if component_sizes else 0
    print(f"  significance basis: {sig_basis}", flush=True)
    print(f"  n_significant_patches={int(sig_mask_flat.sum())}, n_connected_components={n_components}, "
          f"largest_component_size={largest_component_size}", flush=True)

    W_adj = build_adjacency_weights(N_LAT_PATCH, N_LON_PATCH)
    r_map_flat = real_df.sort_values("patch_id")["r"].to_numpy(dtype=np.float64)
    real_moran_i = morans_i(r_map_flat, W_adj)

    n_moran_perm = 999
    moran_null = np.empty(n_moran_perm)
    for k in range(n_moran_perm):
        shuffled = rng.permutation(r_map_flat)
        moran_null[k] = morans_i(shuffled, W_adj)
    moran_p = (1 + np.sum(np.abs(moran_null) >= abs(real_moran_i))) / (n_moran_perm + 1)
    print(f"  Moran's I (real r-map) = {real_moran_i:.4f}, permutation p = {moran_p:.4f} "
          f"(n_perm={n_moran_perm}, seed={RANDOM_SEED})", flush=True)

    print("=== Step 8: save predictions, metrics, permutation summary ===", flush=True)
    np.savez(
        os.path.join(PRED_DIR, "patch_predictions.npz"),
        water_year=np.asarray(WATER_YEARS, dtype=np.int32),
        y_true=y_swe.astype(np.float32),
        y_pred=real_preds_all,  # (2048, 37)
    )
    real_df_sorted = real_df.sort_values("patch_id").reset_index(drop=True)
    real_df_sorted.to_csv(os.path.join(OUT_ROOT, "patch_metrics.csv"), index=False)
    np.savez(
        os.path.join(OUT_ROOT, "permutation_summary.npz"),
        perm_r=perm_r, perm_r2=perm_r2,
        null_max_r=null_max_r, null_max_abs_r=null_max_abs_r, null_max_r2=null_max_r2,
        perm_indices=perm_indices, random_seed=RANDOM_SEED,
    )

    print("=== Step 9: figures ===", flush=True)
    make_figures(real_df_sorted, sig_mask_2d, labels_2d, n_components)

    print("=== Step 10: ranked tables ===", flush=True)
    top20_r = real_df_sorted.reindex(real_df_sorted["r"].abs().sort_values(ascending=False).index).head(20)
    top20_r2 = real_df_sorted.sort_values("R2", ascending=False).head(20)
    top20_r.to_csv(os.path.join(OUT_ROOT, "top20_by_abs_pearson_r.csv"), index=False)
    top20_r2.to_csv(os.path.join(OUT_ROOT, "top20_by_R2.csv"), index=False)

    sierra_df = real_df_sorted[real_df_sorted["intersects_sierra"]]
    sierra_df.to_csv(os.path.join(OUT_ROOT, "sierra_patch_results.csv"), index=False)

    print("=== Step 11: leakage/statistical-protocol audit ===", flush=True)
    for p in [0, 1024, 2047]:
        df_check = pd.read_csv(os.path.join(PRED_DIR if False else OUT_ROOT, "patch_metrics.csv"))
    audit_lines = [
        "No SWE information entered H extraction (H built purely from ERA5 atmospheric fields, verified in prior task).",
        "No global target-based feature selection: each patch's 1024-dim vector is used as-is; no feature was chosen "
        "based on correlation with SWE outside the per-patch model itself.",
        "No PCA anywhere in this audit.",
        "No held-out year used in scaler fitting: StandardScaler statistics (mean/std) are computed from the "
        "36 training years only, inside standardize_train_test(), called fresh per outer fold.",
        "No held-out year used in alpha selection: alpha is chosen via the closed-form LOO identity evaluated "
        "strictly on the 36 training years (loo_sse_all_alphas operates on Y_train, the held-out row is excluded "
        "before this function is ever called).",
        "Each of the 37 years is predicted exactly once per patch (nested_loyo_fast's outer loop covers "
        "range(37) exactly once, verified by predictions array shape (2048,37) with no NaNs).",
        "The permutation procedure uses the EXACT SAME closed-form pipeline (nested_loyo_fast) as the real "
        "labels -- real and permuted labels are columns of the same Y_full matrix processed together, so there "
        "is no separate/weaker code path for the null.",
        f"n_nan_in_real_predictions={int(np.isnan(real_preds_all).sum())} (must be 0).",
    ]
    with open(os.path.join(OUT_ROOT, "leakage_audit.txt"), "w") as f:
        f.write("\n".join(f"- {line}" for line in audit_lines))
    assert np.isnan(real_preds_all).sum() == 0

    print("=== Step 12: final summary + Q&A ===", flush=True)
    strongest_row = real_df_sorted.reindex(real_df_sorted["r"].abs().sort_values(ascending=False).index).iloc[0]
    strongest_p = int(strongest_row["patch_id"])
    strongest_survives_fdr = bool(strongest_row["fdr_q"] < 0.05)
    strongest_survives_fwe = bool(fwe_p_max_abs_r < 0.05)

    sierra_best = sierra_df.reindex(sierra_df["r"].abs().sort_values(ascending=False).index).iloc[0] if len(sierra_df) else None

    if strongest_survives_fdr and n_components >= 1 and largest_component_size >= 2:
        interpretation = "A"
    elif (n_uncorrected_p05 > 0) and (n_fdr_q05 == 0 or not strongest_survives_fwe):
        interpretation = "B"
    else:
        interpretation = "C"

    qa = {
        "Q1_strongest_patch": {
            "patch_id": strongest_p, "row": int(strongest_row["row"]), "col": int(strongest_row["col"]),
            "lat_center": float(strongest_row["lat_center"]), "lon_center": float(strongest_row["lon_center"]),
            "r": float(strongest_row["r"]), "R2": float(strongest_row["R2"]), "RMSE": float(strongest_row["RMSE"]),
            "MAE": float(strongest_row["MAE"]), "sign_accuracy": float(strongest_row["sign_accuracy"]),
            "empirical_p": float(strongest_row["empirical_p"]), "fdr_q": float(strongest_row["fdr_q"]),
            "intersects_sierra": bool(strongest_row["intersects_sierra"]),
        },
        "Q2_stronger_than_chance": {
            "survives_fdr_q05": strongest_survives_fdr,
            "survives_family_wise_max_stat_p05": strongest_survives_fwe,
            "family_wise_p_max_abs_r": float(fwe_p_max_abs_r),
        },
        "Q3_spatial_clusters": {
            "n_significant_patches": int(sig_mask_flat.sum()), "significance_basis": sig_basis,
            "n_connected_components": n_components, "largest_component_size": largest_component_size,
            "morans_i": real_moran_i, "morans_i_permutation_p": float(moran_p),
        },
        "Q4_sierra_vs_remote": {
            "n_sierra_patches": n_sierra,
            "sierra_best_abs_r": float(sierra_best["r"]) if sierra_best is not None else None,
            "sierra_best_patch_id": int(sierra_best["patch_id"]) if sierra_best is not None else None,
            "global_best_abs_r": float(strongest_row["r"]),
            "global_best_is_in_sierra": bool(strongest_row["intersects_sierra"]),
        },
        "Q5_interpretation": interpretation,
    }

    summary = {
        "n_patches": N_PATCHES, "n_permutations": N_PERMUTATIONS, "random_seed": RANDOM_SEED,
        "alpha_grid": ALPHA_GRID.tolist(),
        "computational_method": "exact closed-form nested-ridge-LOO (fast_ridge_loyo.py), validated against "
                                 "brute-force sklearn refitting in validate_fast_ridge_loyo.py (must show PASS "
                                 "in validate.out before these results are trusted)",
        "n_uncorrected_p05": n_uncorrected_p05, "n_fdr_q05": n_fdr_q05, "n_fdr_q10": n_fdr_q10,
        "family_wise_p_max_r": float(fwe_p_max_r), "family_wise_p_max_abs_r": float(fwe_p_max_abs_r),
        "family_wise_p_max_r2": float(fwe_p_max_r2),
        "morans_i": real_moran_i, "morans_i_p": float(moran_p),
        "qa": qa,
        "elapsed_seconds": time.time() - t_start,
        "output_root": OUT_ROOT,
    }
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    spatial_coherence = {
        "significance_basis": sig_basis, "n_significant_patches": int(sig_mask_flat.sum()),
        "n_connected_components": n_components, "largest_component_size": largest_component_size,
        "components": component_locations,
        "morans_i": real_moran_i, "morans_i_permutation_p": float(moran_p), "morans_i_n_permutations": n_moran_perm,
    }
    with open(os.path.join(OUT_ROOT, "spatial_coherence.json"), "w") as f:
        json.dump(spatial_coherence, f, indent=2, default=str)

    print(json.dumps(qa, indent=2, default=str), flush=True)
    print(f"\n=== DONE in {time.time() - t_start:.1f}s ===", flush=True)


def make_figures(real_df: pd.DataFrame, sig_mask_2d: np.ndarray, labels_2d: np.ndarray, n_components: int):
    row = real_df.sort_values("patch_id")["row"].to_numpy().reshape(N_LAT_PATCH, N_LON_PATCH)
    lat_2d = real_df.sort_values("patch_id")["lat_center"].to_numpy().reshape(N_LAT_PATCH, N_LON_PATCH)
    lon_2d = real_df.sort_values("patch_id")["lon_center"].to_numpy().reshape(N_LAT_PATCH, N_LON_PATCH)
    r_2d = real_df.sort_values("patch_id")["r"].to_numpy().reshape(N_LAT_PATCH, N_LON_PATCH)
    r2_2d = real_df.sort_values("patch_id")["R2"].to_numpy().reshape(N_LAT_PATCH, N_LON_PATCH)
    lat_edges = np.linspace(-90, 90, N_LAT_PATCH + 1)
    lon_edges = np.linspace(-180, 180, N_LON_PATCH + 1)

    def sierra_box(ax):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-122.5, 35), 4.5, 7, fill=False, edgecolor="lime", linewidth=2, zorder=5))

    # Figure 1: r map
    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = np.abs(r_2d).max()
    im = ax.pcolormesh(lon_edges, lat_edges, r_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    sierra_box(ax)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Figure 1: Patch-wise LOYO Pearson r (real SWE)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig1_pearson_r_map.png"), dpi=150)
    plt.close(fig)

    # Figure 2: R2 map
    fig, ax = plt.subplots(figsize=(12, 6))
    vmax2 = max(abs(np.nanmin(r2_2d)), abs(np.nanmax(r2_2d)))
    im = ax.pcolormesh(lon_edges, lat_edges, r2_2d, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, shading="flat")
    sierra_box(ax)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Figure 2: Patch-wise LOYO R² (real SWE)")
    fig.colorbar(im, ax=ax, label="R²")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig2_r2_map.png"), dpi=150)
    plt.close(fig)

    # Figure 3: significance map
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(lon_edges, lat_edges, sig_mask_2d.astype(int), cmap="Greys", vmin=0, vmax=1, shading="flat")
    sierra_box(ax)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Figure 3: Patch significance map")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig3_significance_map.png"), dpi=150)
    plt.close(fig)

    # Figure 4: connected components
    fig, ax = plt.subplots(figsize=(12, 6))
    comp_display = np.where(labels_2d >= 0, labels_2d, np.nan)
    im = ax.pcolormesh(lon_edges, lat_edges, comp_display, cmap="tab20", shading="flat")
    sierra_box(ax)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Figure 4: Spatial connected components (n={n_components})")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig4_connected_components.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
