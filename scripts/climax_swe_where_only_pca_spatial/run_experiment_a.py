"""Experiment A -- WHERE-only baseline: target-independent PCA (per-patch,
fit on pooled training patch vectors, no labels involved) collapses each
patch's 1024-d embedding to K components, then a single Ridge regression
over the flattened (2048*K)-d "where did which compressed signal occur"
vector predicts April-1 Sierra SWE standardized anomaly. This model can
learn WHERE (which of the 2048 patches matter, via the Ridge coefficients
on that patch's K slots) but NOT WHAT within a patch beyond the K unsupervised
PCA directions shared identically by every patch -- it cannot up/down-weight
a task-relevant direction inside a patch's 1024-d embedding differently from
any other patch. Compare against Experiment B (WHAT+WHERE, rank-1 shared
latent projection) to isolate whether SWE skill (if any) comes from spatial
selection alone or requires task-adapted "what" as well.

Strict LOYO leakage safety: PCA, standardization, K selection, and alpha
selection are all fit using ONLY the 36 outer-training years of each fold;
the held-out year touches nothing but the final prediction. PCA is fit on
reshaped [36*2048, 1024] pooled patch vectors (unsupervised -- no SWE label
enters this step at all), so it is identical whether the downstream label
column is the real SWE anomaly or a label permutation; this lets one PCA fit
per outer fold be reused across the real labels AND all 1000 permutation
columns, keeping the permutation-null sweep cheap. Ridge alpha and K
selection reuse the exact closed-form nested-LOO machinery validated in
scripts/climax_patchwise_swe_ridge_audit/fast_ridge_loyo.py (see that
module's docstring for the standardization/intercept conventions, which are
followed identically here), extended with an outer sweep over K.

No fine-tuning of ClimaX, no attention, no MLP, no Captum/SHAP -- Ridge and
PCA only, as specified.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, "/global/u1/h/hyvchen/Snow-Predication-at-Sierra/scripts/climax_patchwise_swe_ridge_audit")
from fast_ridge_loyo import (  # noqa: E402
    standardize_train_test,
    gram_eigendecomposition,
    loo_sse_all_alphas,
    predict_test_point_grouped_by_alpha,
)
from patch_geometry import build_patch_metadata, N_LAT_PATCH, N_LON_PATCH, N_PATCHES  # noqa: E402

H_PATH = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents/H_march25_obs_latents.npy"
PAIRED_PATH = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents/paired_dataset.npz"
OUT_DIR_BASE = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/climax_swe_where_only_pca_spatial"

ALPHA_GRID_A = np.array([1e-2, 1e-1, 1, 10, 100, 1000, 10000], dtype=np.float64)
K_GRID = [1, 2, 3, 5, 8]
K_MAX = max(K_GRID)
PCA_SEED = 0
PERM_SEED = 20260908

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
N_PERM = 5 if SMOKE_TEST else 1000
SMOKE_N_FOLDS = 3


def load_data():
    H = np.load(H_PATH).astype(np.float64)  # (37, 2048, 1024)
    paired = np.load(PAIRED_PATH)
    y = paired["swe_apr1_standardized"].astype(np.float64)  # (37,)
    water_year = paired["water_year"]
    assert H.shape == (37, N_PATCHES, 1024), H.shape
    assert y.shape == (37,)
    return H, y, water_year


def build_permutation_labels(y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Returns Y_all (n, 1+n_perm): column 0 = real y, columns 1.. = y under
    a fixed random permutation of the year->label mapping, one permutation
    per column, reused identically across every outer fold (standard label-
    permutation test convention)."""
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    Y_all = np.empty((n, 1 + n_perm), dtype=np.float64)
    Y_all[:, 0] = y
    for j in range(n_perm):
        perm = rng.permutation(n)
        Y_all[:, 1 + j] = y[perm]
    return Y_all


def fit_pca_project_all_years(H: np.ndarray, train_idx: np.ndarray, k_max: int, seed: int) -> np.ndarray:
    """Fits PCA(n_components=k_max) on the pooled [len(train_idx)*2048, 1024]
    training patch vectors ONLY (unsupervised -- no labels), then projects
    ALL 37 years' patches through it. Returns H_proj (37, 2048, k_max)."""
    n_years_all, n_patches, d = H.shape
    X_train_patches = H[train_idx].reshape(-1, d)
    pca = PCA(n_components=k_max, svd_solver="randomized", random_state=seed)
    pca.fit(X_train_patches)
    H_flat = H.reshape(-1, d)
    H_proj_flat = pca.transform(H_flat)  # (37*2048, k_max)
    return H_proj_flat.reshape(n_years_all, n_patches, k_max)


def run_outer_fold(H_proj_all: np.ndarray, Y_all: np.ndarray, train_idx: np.ndarray, test_idx: int):
    """For one outer fold: sweeps K in K_GRID, selects (K, alpha) per label
    column via inner-LOO score on the 36 training years (closed-form, no
    refitting), predicts the held-out year for every column. Returns
    final_pred (n_cols,), chosen_K (n_cols,), chosen_alpha (n_cols,)."""
    n_cols = Y_all.shape[1]
    per_K = {}
    ss_tot = None
    for K in K_GRID:
        X_full_K = H_proj_all[:, :, :K].reshape(H_proj_all.shape[0], -1)  # (37, 2048*K)
        X_train_K = X_full_K[train_idx]
        x_test_K = X_full_K[test_idx]
        X_train_std, x_test_std = standardize_train_test(X_train_K, x_test_K)
        eigvals, eigvecs = gram_eigendecomposition(X_train_std)

        Y_train = Y_all[train_idx]  # (36, n_cols)
        y_train_mean = Y_train.mean(axis=0, keepdims=True)
        Y_train_c = Y_train - y_train_mean
        if ss_tot is None:
            ss_tot = np.sum(Y_train_c ** 2, axis=0)  # same for every K (Y unchanged)

        loo_sse = loo_sse_all_alphas(eigvals, eigvecs, Y_train_c, ALPHA_GRID_A)  # (n_alpha, n_cols)
        best_alpha_idx = np.argmin(loo_sse, axis=0)  # (n_cols,)
        best_sse = loo_sse[best_alpha_idx, np.arange(n_cols)]
        score = 1.0 - best_sse / np.where(ss_tot > 1e-12, ss_tot, 1.0)

        per_K[K] = dict(
            eigvals=eigvals, eigvecs=eigvecs, X_train_std=X_train_std, x_test_std=x_test_std,
            Y_train_c=Y_train_c, y_train_mean=y_train_mean,
            best_alpha=ALPHA_GRID_A[best_alpha_idx], score=score,
        )

    scores_stack = np.stack([per_K[K]["score"] for K in K_GRID], axis=0)  # (n_K, n_cols)
    chosen_K_idx = np.argmax(scores_stack, axis=0)  # (n_cols,) ties -> smallest K (first max)
    chosen_K = np.array([K_GRID[i] for i in chosen_K_idx])

    final_pred = np.empty(n_cols, dtype=np.float64)
    chosen_alpha = np.empty(n_cols, dtype=np.float64)
    for k_idx, K in enumerate(K_GRID):
        cols = np.where(chosen_K_idx == k_idx)[0]
        if len(cols) == 0:
            continue
        d = per_K[K]
        preds = predict_test_point_grouped_by_alpha(
            d["eigvals"], d["eigvecs"], d["X_train_std"], d["x_test_std"],
            d["Y_train_c"][:, cols], d["best_alpha"][cols], ALPHA_GRID_A,
        )
        final_pred[cols] = preds + d["y_train_mean"][0, cols]
        chosen_alpha[cols] = d["best_alpha"][cols]

    return final_pred, chosen_K, chosen_alpha


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-12 else 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    sign_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return dict(r=r, R2=r2, RMSE=rmse, MAE=mae, sign_accuracy=sign_acc, n=n)


def descriptive_full_fit_spatial_map(H: np.ndarray, y: np.ndarray):
    """Descriptive, IN-SAMPLE (not out-of-sample) fit using all 37 years, for
    the spatial coefficient visualization only. PCA + K/alpha selected via a
    single LOO sweep over all 37 years (same closed-form machinery, n=37
    instead of 36), then one final Ridge fit on all 37 years at the selected
    (K, alpha). Returns beta (2048, K_selected) and K_selected, alpha_selected."""
    n = H.shape[0]
    X_train_patches = H.reshape(-1, H.shape[-1])
    pca = PCA(n_components=K_MAX, svd_solver="randomized", random_state=PCA_SEED)
    pca.fit(X_train_patches)
    H_proj = pca.transform(X_train_patches).reshape(n, N_PATCHES, K_MAX)

    y_col = y[:, None]
    best_score, best_K, best_alpha = -np.inf, None, None
    best_state = None
    for K in K_GRID:
        X_full_K = H_proj[:, :, :K].reshape(n, -1)
        mu = X_full_K.mean(axis=0)
        sigma = X_full_K.std(axis=0, ddof=0)
        sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
        X_std = (X_full_K - mu) / sigma_safe
        eigvals, eigvecs = gram_eigendecomposition(X_std)
        y_mean = y_col.mean(axis=0, keepdims=True)
        y_c = y_col - y_mean
        ss_tot = np.sum(y_c ** 2)
        loo_sse = loo_sse_all_alphas(eigvals, eigvecs, y_c, ALPHA_GRID_A)[:, 0]
        best_idx = int(np.argmin(loo_sse))
        score = 1.0 - loo_sse[best_idx] / max(ss_tot, 1e-12)
        if score > best_score:
            best_score, best_K, best_alpha = score, K, ALPHA_GRID_A[best_idx]
            best_state = dict(X_std=X_std, mu=mu, sigma_safe=sigma_safe, y_c=y_c, y_mean=y_mean)

    # Final ridge fit on all 37 years at (best_K, best_alpha), primal solve
    # (feasible here: p = 2048*best_K but done once, not in a hot loop).
    X_std = best_state["X_std"]
    y_c = best_state["y_c"]
    p = X_std.shape[1]
    A = X_std.T @ X_std + best_alpha * np.eye(p)
    b = X_std.T @ y_c[:, 0]
    beta_std = np.linalg.solve(A, b)  # coefficients in standardized-feature space
    beta = (beta_std / best_state["sigma_safe"]).reshape(N_PATCHES, best_K)  # unstandardize
    return beta, best_K, float(best_alpha), float(best_score)


def main() -> int:
    t0 = time.time()
    OUT_DIR = f"{OUT_DIR_BASE}/_smoke_test" if SMOKE_TEST else OUT_DIR_BASE
    if SMOKE_TEST:
        os.makedirs(f"{OUT_DIR}/predictions", exist_ok=True)
        print(f"SMOKE_TEST=1: writing to {OUT_DIR}, {SMOKE_N_FOLDS} folds, {N_PERM} perms", flush=True)
    print("Loading data...", flush=True)
    H, y, water_year = load_data()
    n = H.shape[0]
    print(f"H shape {H.shape}, y shape {y.shape}, elapsed {time.time()-t0:.1f}s", flush=True)

    Y_all = build_permutation_labels(y, N_PERM, PERM_SEED)  # (37, 1001)
    n_cols = Y_all.shape[1]

    oof_pred = np.full((n, n_cols), np.nan, dtype=np.float64)
    chosen_K_real = np.zeros(n, dtype=int)
    chosen_alpha_real = np.zeros(n, dtype=np.float64)

    fold_indices = range(SMOKE_N_FOLDS) if SMOKE_TEST else range(n)
    for i in fold_indices:
        train_idx = np.array([j for j in range(n) if j != i])
        H_proj_all = fit_pca_project_all_years(H, train_idx, K_MAX, PCA_SEED)
        final_pred, chosen_K, chosen_alpha = run_outer_fold(H_proj_all, Y_all, train_idx, i)
        oof_pred[i] = final_pred
        chosen_K_real[i] = chosen_K[0]
        chosen_alpha_real[i] = chosen_alpha[0]
        print(f"  fold {i+1}/{n} (WY{water_year[i]}) done, K*={chosen_K[0]}, alpha*={chosen_alpha[0]:g}, "
              f"elapsed {time.time()-t0:.1f}s", flush=True)

    fold_idx_arr = np.array(list(fold_indices))
    y_pred_real = oof_pred[fold_idx_arr, 0]
    metrics_real = compute_metrics(y[fold_idx_arr], y_pred_real)
    print(f"\nReal-label metrics ({'SMOKE TEST subset' if SMOKE_TEST else 'full'}): {metrics_real}", flush=True)

    r_perm = np.empty(N_PERM, dtype=np.float64)
    for j in range(N_PERM):
        y_perm_true = Y_all[fold_idx_arr, 1 + j]
        y_perm_pred = oof_pred[fold_idx_arr, 1 + j]
        r_perm[j] = np.corrcoef(y_perm_true, y_perm_pred)[0, 1] if np.std(y_perm_pred) > 1e-12 else 0.0
    r_perm = np.nan_to_num(r_perm, nan=0.0)
    r_real = metrics_real["r"]
    emp_p = (1 + np.sum(np.abs(r_perm) >= np.abs(r_real))) / (N_PERM + 1)
    print(f"Permutation test: r_real={r_real:.4f}, null |r| mean={np.mean(np.abs(r_perm)):.4f}, "
          f"empirical p={emp_p:.4f}", flush=True)

    # -------- descriptive full-fit spatial map (in-sample, for viz only) --------
    print("\nFitting descriptive full-37-year model for spatial coefficient map "
          "(IN-SAMPLE, NOT out-of-sample; visualization only)...", flush=True)
    beta, desc_K, desc_alpha, desc_loo_r2 = descriptive_full_fit_spatial_map(H, y)
    spatial_mag = np.sqrt(np.sum(beta ** 2, axis=1))  # (2048,) aggregate magnitude
    print(f"  descriptive fit: K={desc_K}, alpha={desc_alpha:g}, inner-LOO R2={desc_loo_r2:.4f}", flush=True)

    # -------- save artifacts --------
    patch_meta = build_patch_metadata()
    oof_df = pd.DataFrame({
        "water_year": water_year[fold_idx_arr], "y_true": y[fold_idx_arr], "y_pred": y_pred_real,
        "chosen_K": chosen_K_real[fold_idx_arr], "chosen_alpha": chosen_alpha_real[fold_idx_arr],
    })
    oof_df.to_csv(f"{OUT_DIR}/predictions/oof_predictions.csv", index=False)

    np.savez(f"{OUT_DIR}/predictions/permutation_null.npz", r_perm=r_perm, r_real=r_real, emp_p=emp_p)

    spatial_df = patch_meta.copy()
    spatial_df["spatial_coef_magnitude"] = spatial_mag
    for k in range(desc_K):
        spatial_df[f"beta_pc{k}"] = beta[:, k]
    spatial_df.to_csv(f"{OUT_DIR}/predictions/spatial_coefficient_map.csv", index=False)
    np.save(f"{OUT_DIR}/predictions/spatial_beta_2048xK.npy", beta)

    summary = dict(
        experiment="A_where_only_pca_spatial",
        description="Target-independent per-fold PCA (K components, unsupervised) + "
                     "flattened 2048*K Ridge; learns WHERE only, not WHAT within a patch.",
        n=n, metrics_real=metrics_real,
        permutation=dict(n_perm=N_PERM, r_real=r_real,
                          null_abs_r_mean=float(np.mean(np.abs(r_perm))),
                          null_abs_r_p95=float(np.percentile(np.abs(r_perm), 95)),
                          empirical_p=float(emp_p)),
        K_grid=K_GRID, alpha_grid=ALPHA_GRID_A.tolist(),
        chosen_K_distribution={int(k): int(np.sum(chosen_K_real[fold_idx_arr] == k)) for k in K_GRID},
        chosen_K_per_fold=chosen_K_real[fold_idx_arr].tolist(),
        chosen_alpha_per_fold=chosen_alpha_real[fold_idx_arr].tolist(),
        smoke_test=SMOKE_TEST,
        descriptive_full_fit=dict(
            note="IN-SAMPLE fit on all 37 years, for spatial coefficient visualization ONLY; "
                 "NOT an out-of-sample estimate.",
            K=desc_K, alpha=desc_alpha, inner_loo_r2_all37=desc_loo_r2,
        ),
        elapsed_seconds=time.time() - t0,
    )
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Total elapsed {time.time()-t0:.1f}s. Artifacts written to {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
