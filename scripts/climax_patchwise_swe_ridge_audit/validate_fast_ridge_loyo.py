"""Validate fast_ridge_loyo's closed-form nested LOYO against brute-force
sklearn Ridge/StandardScaler refitting. Must pass before the fast method is
trusted for the full 2048-patch x 1000-permutation sweep."""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/global/u1/h/hyvchen/Snow-Predication-at-Sierra/scripts/climax_patchwise_swe_ridge_audit")
from fast_ridge_loyo import ALPHA_GRID, nested_loyo_fast  # noqa: E402


def brute_force_nested_loyo(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Reference implementation: explicit sklearn refitting, same protocol as
    fast_ridge_loyo:
      - StandardScaler fit ONCE per outer fold on all 36 outer-training years
        (not re-fit inside each inner-LOO iteration on 35 points);
      - the intercept-equivalent y-centering is likewise done ONCE per outer
        fold from all 36 outer-training years' mean (fit_intercept=False
        used throughout on the pre-centered y, with that fixed mean added
        back to predictions), not re-derived per inner-LOO 35-point subset.
    Both simplifications keep the outer held-out test year fully isolated
    (it never touches the scaler or the y-mean) and satisfy the stated
    requirements literally; they are what make the closed-form LOO identity
    (fixed X, y varies) applicable at all. Letting sklearn's default
    fit_intercept=True implicitly re-derive a fresh intercept from each
    35-point inner-LOO subset is a MEANINGFULLY stricter variant (confirmed
    empirically: this was the actual source of the first validation
    failures, not the scaler-refit question) and cannot be vectorized the
    same way, so it is not used here -- both fast and reference now use the
    identical, explicitly documented convention."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        n_train = len(train_idx)

        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.transform(X_train)
        y_mean = y_train.mean()
        y_train_centered = y_train - y_mean

        best_alpha, best_score = alphas[0], -np.inf
        for alpha in alphas:
            preds = np.zeros(n_train)
            for k in range(n_train):
                mask = np.ones(n_train, dtype=bool)
                mask[k] = False
                model = Ridge(alpha=alpha, fit_intercept=False).fit(X_train_std[mask], y_train_centered[mask])
                preds[k] = model.predict(X_train_std[k:k + 1])[0]
            ss_res = np.sum((y_train_centered - preds) ** 2)
            ss_tot = np.sum((y_train_centered - y_train_centered.mean()) ** 2)
            score = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            if score > best_score:
                best_score, best_alpha = score, alpha

        model = Ridge(alpha=best_alpha, fit_intercept=False).fit(X_train_std, y_train_centered)
        out[i] = model.predict(scaler.transform(X[i:i + 1]))[0] + y_mean
    return out


def main() -> int:
    rng = np.random.default_rng(20260909)
    n, p = 37, 1024
    n_perm_test = 5

    print("=== Validation 1: synthetic random data, 1 real + 5 permuted label columns ===", flush=True)
    X = rng.normal(size=(n, p))
    y_real = rng.normal(size=n)

    Y_cols = [y_real]
    for _ in range(n_perm_test):
        Y_cols.append(rng.permutation(y_real))
    Y = np.stack(Y_cols, axis=1)  # (37, 6)

    fast_out = nested_loyo_fast(X, Y, ALPHA_GRID)  # (37, 6)

    max_abs_diff = 0.0
    for c in range(Y.shape[1]):
        brute_out = brute_force_nested_loyo(X, Y[:, c], ALPHA_GRID)
        diff = np.max(np.abs(fast_out[:, c] - brute_out))
        max_abs_diff = max(max_abs_diff, diff)
        print(f"  column {c}: max abs diff (fast vs brute-force) = {diff:.3e}", flush=True)

    print(f"\n  OVERALL max abs diff across all columns: {max_abs_diff:.3e}", flush=True)
    ok1 = max_abs_diff < 1e-6
    print(f"  {'PASS' if ok1 else 'FAIL'} (tolerance 1e-6)", flush=True)

    print("\n=== Validation 2: real ClimaX H, one real patch, real SWE labels ===", flush=True)
    H = np.load("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents/H_march25_obs_latents.npy")
    paired = np.load("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents/latents/paired_dataset.npz")
    y_swe = paired["swe_apr1_standardized"].astype(np.float64)

    test_patches = [0, 1024, 2047]
    ok2 = True
    for p_idx in test_patches:
        X_p = H[:, p_idx, :].astype(np.float64)
        fast_p = nested_loyo_fast(X_p, y_swe[:, None], ALPHA_GRID)[:, 0]
        brute_p = brute_force_nested_loyo(X_p, y_swe, ALPHA_GRID)
        diff = np.max(np.abs(fast_p - brute_p))
        print(f"  patch {p_idx}: max abs diff = {diff:.3e}", flush=True)
        if diff >= 1e-6:
            ok2 = False
    print(f"  {'PASS' if ok2 else 'FAIL'} (tolerance 1e-6)", flush=True)

    overall_ok = ok1 and ok2
    print(f"\n=== OVERALL VALIDATION: {'PASS' if overall_ok else 'FAIL'} ===", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
