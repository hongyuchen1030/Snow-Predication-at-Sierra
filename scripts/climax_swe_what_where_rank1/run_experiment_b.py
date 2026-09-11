"""Experiment B -- WHAT+WHERE rank-1 factorized model.

    s_{y,p} = w . H[y,p,:]            (w in R^1024, SHARED across all 2048
                                        patches -- the "WHAT" direction)
    yhat_y  = b + sum_p a_p * s_{y,p} = b + a . (H[y] @ w)
                                       (a in R^2048 -- the "WHERE" weights)

The effective per-patch coefficient matrix W[p,d] = a_p * w_d is rank-1
constrained: every patch reads out the SAME direction w of its 1024-d
embedding (unlike Experiment A's shared unsupervised PCA basis, w here is
task-adapted via regression, but still identical across space), and only the
per-patch scalar a_p varies. This isolates whether SWE skill requires a
patch-specific "what" (full-rank, e.g. attention) or a single shared "what"
direction plus spatial reweighting suffices.

Bilinear in (a, w) -> solved via alternating ridge regression, each half-step
using the exact closed-form ridge dual solve (n_train << feature dim, so
inverting the n x n Gram matrix is cheap; O(p^3) primal solves are not
attempted). Scale ambiguity (a*c, w/c predict identically) is resolved by
unit-normalizing w after every w-step; the next a-step's ridge refit absorbs
whatever scale is needed, so this does not bias the fit.

No hidden neural layers, no attention, no MLP, no fine-tuning of ClimaX --
alternating ridge regression only, as specified.

Hyperparameter/init selection cost: a literal nested-LOO of the full ALS
(rerunning ALS to convergence for every left-out training point, for every
hyperparameter combo, for every outer fold) is computationally intractable
here. Instead: (1) for each of a small (lambda_a, lambda_w) grid, ALS is run
to convergence on all 36 outer-training years, from several deterministic
initializations, keeping the init with lowest final training loss;
(2) that combo is scored by an APPROXIMATE leave-one-out criterion: closed-
form ridge-LOO on the converged a-given-w ridge step only (w held fixed at
its converged value), not a full ALS refit per left-out point. This is
documented explicitly as an approximation to true nested LOO, analogous to
the standardize-once-per-outer-fold simplification documented in
scripts/climax_patchwise_swe_ridge_audit/fast_ridge_loyo.py.

Permutation testing: to keep 1000 permutations tractable, each outer fold's
(lambda_a, lambda_w, init) are selected ONCE from the real-label nested
procedure and held FIXED across all permutations of that fold; only the ALS
ridge-fitting itself (the part that actually depends on the label vector) is
rerun per permutation, with fewer iterations (10 vs 15) since exact
convergence tracking is not needed for a null distribution. This is a
documented efficiency simplification, not part of the real-data result.
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
from fast_ridge_loyo import gram_eigendecomposition, loo_sse_all_alphas  # noqa: E402
from patch_geometry import build_patch_metadata, N_LAT_PATCH, N_LON_PATCH, N_PATCHES  # noqa: E402

H_PATH = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents/H_march25_obs_latents.npy"
PAIRED_PATH = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents/paired_dataset.npz"
OUT_DIR_BASE = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/climax_swe_what_where_rank1"

LAMBDA_A_GRID = [1.0, 10.0, 100.0]
LAMBDA_W_GRID = [1.0, 10.0, 100.0]
N_ITER_REAL = 15
# Timed empirically (smoke test, pre-matmul-optimization): the real-data
# nested sweep (9 combos x 3 inits x <=15 iters) costs ~27.6s/fold, so a full
# 37-fold pass is ~17 min -- fine. But the permutation loop reruns a full ALS
# fit per (permutation, fold), and even at only 10 iters/fit, 1000 perms x 37
# folds was projected at >10 hours -- far beyond the 4-hour interactive
# window. N_ITER_PERM and N_PERM are reduced accordingly (from Experiment
# A's 1000, which stays feasible there because A's PCA step is label-
# independent and its Ridge step is closed-form/non-iterative). Hyper-
# parameters and init are also fixed per fold from the real-data fit (see
# module docstring) rather than re-swept per permutation.
N_ITER_PERM = 8
TOL = 1e-6
PERM_SEED = 20260908
INIT_SEEDS = {"random_seed1": 1, "random_seed2": 2}  # + "first_pc" (label-free, data-derived)

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
N_PERM = 5 if SMOKE_TEST else 200
SMOKE_N_FOLDS = 3


def load_data():
    H = np.load(H_PATH).astype(np.float64)
    paired = np.load(PAIRED_PATH)
    y = paired["swe_apr1_standardized"].astype(np.float64)
    water_year = paired["water_year"]
    assert H.shape == (37, N_PATCHES, 1024), H.shape
    return H, y, water_year


def build_permutation_labels(y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    Y_all = np.empty((n, 1 + n_perm), dtype=np.float64)
    Y_all[:, 0] = y
    for j in range(n_perm):
        Y_all[:, 1 + j] = y[rng.permutation(n)]
    return Y_all


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
    return mu, sigma_safe, (X - mu) / sigma_safe


def ridge_full_solve(eigvals, eigvecs, y_c, lam):
    """coef in the standardized-feature space that generated eigvals/eigvecs,
    for the FULL (non-LOO) fit at penalty lam."""
    inv_diag = 1.0 / (eigvals + lam)
    alpha_coef = eigvecs @ (inv_diag * (eigvecs.T @ y_c))
    return alpha_coef  # (n,) dual coefficients; caller contracts with X_std.T


def first_pc_init(H_train: np.ndarray, seed: int) -> np.ndarray:
    X = H_train.reshape(-1, H_train.shape[-1])
    pca = PCA(n_components=1, svd_solver="randomized", random_state=seed)
    pca.fit(X)
    return pca.components_[0]


def make_inits(H_train: np.ndarray) -> dict:
    inits = {"first_pc": first_pc_init(H_train, seed=0)}
    for name, seed in INIT_SEEDS.items():
        rng = np.random.default_rng(seed)
        v = rng.normal(size=H_train.shape[-1])
        inits[name] = v / np.linalg.norm(v)
    return inits


def als_fit(H_train: np.ndarray, y_train: np.ndarray, lambda_a: float, lambda_w: float,
            w_init: np.ndarray, n_iter: int, tol: float = TOL):
    w = w_init / max(np.linalg.norm(w_init), 1e-12)
    prev_obj = None
    a_raw = b = mu_s = sigma_s = eigvals_s = eigvecs_s = S_std = a_std = None
    for _ in range(n_iter):
        S_train = H_train @ w  # (n,2048,1024) @ (1024,) -> (n,2048), BLAS matmul
        mu_s, sigma_s, S_std = standardize_fit(S_train)
        y_mean = y_train.mean()
        y_c = y_train - y_mean
        eigvals_s, eigvecs_s = gram_eigendecomposition(S_std)
        alpha_coef = ridge_full_solve(eigvals_s, eigvecs_s, y_c, lambda_a)
        a_std = S_std.T @ alpha_coef
        a_raw = a_std / sigma_s
        b = y_mean - mu_s @ a_raw

        Z_train = H_train.transpose(0, 2, 1) @ a_raw  # (n,1024,2048) @ (2048,) -> (n,1024), BLAS matmul
        mu_z, sigma_z, Z_std = standardize_fit(Z_train)
        eigvals_z, eigvecs_z = gram_eigendecomposition(Z_std)
        alpha_coef_w = ridge_full_solve(eigvals_z, eigvecs_z, y_c, lambda_w)
        w_std = Z_std.T @ alpha_coef_w
        w_raw = w_std / sigma_z
        w = w_raw / max(np.linalg.norm(w_raw), 1e-12)

        pred_train = S_train @ a_raw + b
        obj = float(np.sum((y_train - pred_train) ** 2) + lambda_a * np.sum(a_std ** 2)
                     + lambda_w * np.sum(w_std ** 2))
        if prev_obj is not None and abs(prev_obj - obj) < tol * max(abs(prev_obj), 1e-8):
            prev_obj = obj
            break
        prev_obj = obj

    # Final recompute of (a, b) at the converged w, for a consistent state to
    # return (mirrors the a-step already run this iteration, cheap).
    S_train = H_train @ w
    mu_s, sigma_s, S_std = standardize_fit(S_train)
    y_mean = y_train.mean()
    y_c = y_train - y_mean
    eigvals_s, eigvecs_s = gram_eigendecomposition(S_std)
    alpha_coef = ridge_full_solve(eigvals_s, eigvecs_s, y_c, lambda_a)
    a_std = S_std.T @ alpha_coef
    a_raw = a_std / sigma_s
    b = y_mean - mu_s @ a_raw

    return dict(w=w, a=a_raw, b=b, final_obj=prev_obj,
                eigvals_s=eigvals_s, eigvecs_s=eigvecs_s, y_c=y_c)


def predict_als(model: dict, H_test: np.ndarray) -> float:
    s_test = H_test @ model["w"]
    return float(s_test @ model["a"] + model["b"])


def approx_loo_score(model: dict, lambda_a: float) -> float:
    """Approximate hyperparameter-selection score: closed-form ridge-LOO of
    the converged a-given-w step only (w frozen), NOT a full ALS refit per
    left-out point -- see module docstring."""
    loo_sse = loo_sse_all_alphas(model["eigvals_s"], model["eigvecs_s"], model["y_c"][:, None],
                                  np.array([lambda_a]))[0, 0]
    ss_tot = np.sum(model["y_c"] ** 2)
    return 1.0 - loo_sse / max(ss_tot, 1e-12)


def fit_fold_nested(H_train: np.ndarray, y_train: np.ndarray):
    """Sweeps (lambda_a, lambda_w) x 3 inits, returns the best model plus
    per-combo diagnostics (init final_obj values, for stability reporting)."""
    inits = make_inits(H_train)
    best = None
    diagnostics = []
    for lambda_a in LAMBDA_A_GRID:
        for lambda_w in LAMBDA_W_GRID:
            init_results = {}
            best_for_combo = None
            for name, w0 in inits.items():
                m = als_fit(H_train, y_train, lambda_a, lambda_w, w0, N_ITER_REAL)
                init_results[name] = m["final_obj"]
                if best_for_combo is None or m["final_obj"] < best_for_combo[1]["final_obj"]:
                    best_for_combo = (name, m)
            score = approx_loo_score(best_for_combo[1], lambda_a)
            diagnostics.append(dict(lambda_a=lambda_a, lambda_w=lambda_w,
                                     best_init=best_for_combo[0], score=score,
                                     init_final_obj=init_results))
            if best is None or score > best["score"]:
                best = dict(lambda_a=lambda_a, lambda_w=lambda_w, init_name=best_for_combo[0],
                            model=best_for_combo[1], score=score, w_init=inits[best_for_combo[0]])
    return best, diagnostics


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


def main() -> int:
    t0 = time.time()
    OUT_DIR = f"{OUT_DIR_BASE}/_smoke_test" if SMOKE_TEST else OUT_DIR_BASE
    if SMOKE_TEST:
        os.makedirs(f"{OUT_DIR}/predictions", exist_ok=True)
        print(f"SMOKE_TEST=1: writing to {OUT_DIR}, {SMOKE_N_FOLDS} folds, {N_PERM} perms", flush=True)
    print("Loading data...", flush=True)
    H, y, water_year = load_data()
    n = H.shape[0]
    print(f"H shape {H.shape}, elapsed {time.time()-t0:.1f}s", flush=True)

    Y_all = build_permutation_labels(y, N_PERM, PERM_SEED)

    y_pred_real = np.full(n, np.nan)
    fold_lambda_a = np.zeros(n)
    fold_lambda_w = np.zeros(n)
    fold_init_name = [None] * n
    fold_w = np.zeros((n, 1024))
    fold_a = np.zeros((n, N_PATCHES))
    fold_diagnostics = []

    fold_indices = range(SMOKE_N_FOLDS) if SMOKE_TEST else range(n)
    fold_idx_arr = np.array(list(fold_indices))
    for i in fold_indices:
        train_idx = np.array([j for j in range(n) if j != i])
        H_train, y_train = H[train_idx], y[train_idx]
        best, diag = fit_fold_nested(H_train, y_train)
        y_pred_real[i] = predict_als(best["model"], H[i])
        fold_lambda_a[i] = best["lambda_a"]
        fold_lambda_w[i] = best["lambda_w"]
        fold_init_name[i] = best["init_name"]
        fold_w[i] = best["model"]["w"]
        fold_a[i] = best["model"]["a"]
        fold_diagnostics.append(dict(fold=i, water_year=int(water_year[i]),
                                      lambda_a=best["lambda_a"], lambda_w=best["lambda_w"],
                                      init_name=best["init_name"], score=best["score"],
                                      combos=diag))
        print(f"  fold {i+1}/{n} (WY{water_year[i]}) done: lambda_a={best['lambda_a']:g}, "
              f"lambda_w={best['lambda_w']:g}, init={best['init_name']}, score={best['score']:.3f}, "
              f"pred={y_pred_real[i]:.3f} true={y[i]:.3f}, elapsed {time.time()-t0:.1f}s", flush=True)

    metrics_real = compute_metrics(y[fold_idx_arr], y_pred_real[fold_idx_arr])
    print(f"\nReal-label metrics ({'SMOKE TEST subset' if SMOKE_TEST else 'full'}): {metrics_real}", flush=True)

    # -------- stability diagnostics (real-label fold models) --------
    def pairwise_cos_stats(V: np.ndarray) -> dict:
        if V.shape[0] < 2:
            return dict(mean_abs_cos=float("nan"), median_abs_cos=float("nan"),
                        min_abs_cos=float("nan"), max_abs_cos=float("nan"))
        Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        C = Vn @ Vn.T
        iu = np.triu_indices(V.shape[0], k=1)
        vals = np.abs(C[iu])
        return dict(mean_abs_cos=float(np.mean(vals)), median_abs_cos=float(np.median(vals)),
                    min_abs_cos=float(np.min(vals)), max_abs_cos=float(np.max(vals)))

    w_stability = pairwise_cos_stats(fold_w[fold_idx_arr])
    a_stability = pairwise_cos_stats(fold_a[fold_idx_arr])
    print(f"w stability (WHAT, cosine sim across 37 folds): {w_stability}", flush=True)
    print(f"a stability (WHERE, cosine sim across 37 folds): {a_stability}", flush=True)

    # -------- safety checkpoint: save the real (non-permutation) results NOW,
    # before the expensive permutation loop, so a walltime timeout there does
    # not lose the already-computed out-of-sample fold results. Overwritten
    # with the full summary (permutation fields added) at the end. --------
    os.makedirs(f"{OUT_DIR}/predictions", exist_ok=True)
    fold_init_name_arr_ckpt = np.array(fold_init_name, dtype=object)
    pd.DataFrame({
        "water_year": water_year[fold_idx_arr], "y_true": y[fold_idx_arr],
        "y_pred": y_pred_real[fold_idx_arr],
        "lambda_a": fold_lambda_a[fold_idx_arr], "lambda_w": fold_lambda_w[fold_idx_arr],
        "init_name": fold_init_name_arr_ckpt[fold_idx_arr],
    }).to_csv(f"{OUT_DIR}/predictions/oof_predictions.csv", index=False)
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(dict(
            experiment="B_what_where_rank1", status="CHECKPOINT_real_fit_only_permutation_pending",
            n=n, metrics_real=metrics_real, stability=dict(w_what=w_stability, a_where=a_stability),
            elapsed_seconds=time.time() - t0,
        ), f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"  [checkpoint written at {time.time()-t0:.1f}s: real fit + stability, "
          f"permutation test not yet run]", flush=True)

    # -------- permutation test: fixed per-fold hyperparams/init, rerun ALS --------
    print("\nRunning permutation null (fixed per-fold lambda/init, rerunning ALS fit only)...",
          flush=True)
    inits_by_fold = {}
    for i in fold_indices:
        train_idx = np.array([j for j in range(n) if j != i])
        H_train = H[train_idx]
        inits_by_fold[i] = make_inits(H_train)[fold_init_name[i]]

    r_perm = np.empty(N_PERM)
    for j in range(N_PERM):
        y_perm_pred = np.full(n, np.nan)
        for i in fold_indices:
            train_idx = np.array([k for k in range(n) if k != i])
            H_train = H[train_idx]
            y_perm_train = Y_all[train_idx, 1 + j]
            m = als_fit(H_train, y_perm_train, fold_lambda_a[i], fold_lambda_w[i],
                        inits_by_fold[i], N_ITER_PERM)
            y_perm_pred[i] = predict_als(m, H[i])
        y_perm_true = Y_all[fold_idx_arr, 1 + j]
        r_perm[j] = (np.corrcoef(y_perm_true, y_perm_pred[fold_idx_arr])[0, 1]
                     if np.std(y_perm_pred[fold_idx_arr]) > 1e-12 else 0.0)
        if (j + 1) % 50 == 0:
            print(f"  permutation {j+1}/{N_PERM} done, elapsed {time.time()-t0:.1f}s", flush=True)
            np.savez(f"{OUT_DIR}/predictions/permutation_null_partial.npz",
                     r_perm_so_far=r_perm[:j + 1], n_perm_done=j + 1, r_real=metrics_real["r"])
    r_perm = np.nan_to_num(r_perm, nan=0.0)
    r_real = metrics_real["r"]
    emp_p = (1 + np.sum(np.abs(r_perm) >= np.abs(r_real))) / (N_PERM + 1)
    print(f"Permutation test: r_real={r_real:.4f}, null |r| mean={np.mean(np.abs(r_perm)):.4f}, "
          f"empirical p={emp_p:.4f}", flush=True)

    # -------- descriptive full-37-year fit for spatial map (IN-SAMPLE, viz only) --------
    print("\nFitting descriptive full-37-year model for spatial map "
          "(IN-SAMPLE, NOT out-of-sample; visualization only)...", flush=True)
    best_full, _ = fit_fold_nested(H, y)
    a_full = best_full["model"]["a"]
    w_full = best_full["model"]["w"]

    # -------- save artifacts --------
    patch_meta = build_patch_metadata()
    fold_init_name_arr = np.array(fold_init_name, dtype=object)
    oof_df = pd.DataFrame({
        "water_year": water_year[fold_idx_arr], "y_true": y[fold_idx_arr],
        "y_pred": y_pred_real[fold_idx_arr],
        "lambda_a": fold_lambda_a[fold_idx_arr], "lambda_w": fold_lambda_w[fold_idx_arr],
        "init_name": fold_init_name_arr[fold_idx_arr],
    })
    oof_df.to_csv(f"{OUT_DIR}/predictions/oof_predictions.csv", index=False)
    np.savez(f"{OUT_DIR}/predictions/permutation_null.npz", r_perm=r_perm, r_real=r_real, emp_p=emp_p)
    np.savez(f"{OUT_DIR}/predictions/fold_w_a.npz",
             fold_w=fold_w[fold_idx_arr], fold_a=fold_a[fold_idx_arr])

    spatial_df = patch_meta.copy()
    spatial_df["a_weight_descriptive"] = a_full
    top_pos = spatial_df.nlargest(15, "a_weight_descriptive")[
        ["patch_id", "row", "col", "lat_center", "lon_center", "a_weight_descriptive", "intersects_sierra"]]
    top_neg = spatial_df.nsmallest(15, "a_weight_descriptive")[
        ["patch_id", "row", "col", "lat_center", "lon_center", "a_weight_descriptive", "intersects_sierra"]]
    spatial_df.to_csv(f"{OUT_DIR}/predictions/spatial_weight_map.csv", index=False)
    np.save(f"{OUT_DIR}/predictions/w_descriptive_1024.npy", w_full)

    with open(f"{OUT_DIR}/fold_diagnostics.json", "w") as f:
        json.dump(fold_diagnostics, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    summary = dict(
        experiment="B_what_where_rank1",
        description="Rank-1 bilinear factorization s=w.H[y,p,:], yhat=b+sum_p a_p*s -- "
                     "shared task-adapted WHAT (w) x per-patch WHERE (a), via alternating ridge.",
        n=n, metrics_real=metrics_real,
        permutation=dict(n_perm=N_PERM, r_real=r_real,
                          null_abs_r_mean=float(np.mean(np.abs(r_perm))),
                          null_abs_r_p95=float(np.percentile(np.abs(r_perm), 95)),
                          empirical_p=float(emp_p)),
        stability=dict(w_what=w_stability, a_where=a_stability),
        lambda_a_grid=LAMBDA_A_GRID, lambda_w_grid=LAMBDA_W_GRID,
        chosen_lambda_a_per_fold=fold_lambda_a[fold_idx_arr].tolist(),
        chosen_lambda_w_per_fold=fold_lambda_w[fold_idx_arr].tolist(),
        chosen_init_per_fold=fold_init_name_arr[fold_idx_arr].tolist(),
        smoke_test=SMOKE_TEST,
        descriptive_full_fit=dict(
            note="IN-SAMPLE fit on all 37 years, for spatial weight visualization ONLY; "
                 "NOT an out-of-sample estimate.",
            lambda_a=best_full["lambda_a"], lambda_w=best_full["lambda_w"], init=best_full["init_name"],
            top_positive_patches=top_pos.to_dict(orient="records"),
            top_negative_patches=top_neg.to_dict(orient="records"),
        ),
        elapsed_seconds=time.time() - t0,
    )
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    print(f"\nDone. Total elapsed {time.time()-t0:.1f}s. Artifacts written to {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
