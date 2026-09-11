"""Debug round 3: test against REAL sklearn.Ridge (fit_intercept=True
default), not a self-consistent hand-rolled no-intercept reference. This is
the test that should have been run from the start."""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, "/global/u1/h/hyvchen/Snow-Predication-at-Sierra/scripts/climax_patchwise_swe_ridge_audit")
from fast_ridge_loyo import nested_loyo_fast, ALPHA_GRID  # noqa: E402
from validate_fast_ridge_loyo import brute_force_nested_loyo  # noqa: E402

print("=== Test: n=37, p=1024 synthetic data, fast (intercept-corrected) vs sklearn.Ridge brute-force ===")
rng = np.random.default_rng(20260909)
n, p = 37, 1024
X = rng.normal(size=(n, p))
y_real = rng.normal(size=n) + 5.0  # deliberately non-zero-mean, to stress-test the intercept fix

Y_cols = [y_real]
for _ in range(5):
    Y_cols.append(rng.permutation(y_real))
Y = np.stack(Y_cols, axis=1)

fast_out = nested_loyo_fast(X, Y, ALPHA_GRID)

max_abs_diff = 0.0
for c in range(Y.shape[1]):
    brute_out = brute_force_nested_loyo(X, Y[:, c], ALPHA_GRID)
    diff = np.max(np.abs(fast_out[:, c] - brute_out))
    max_abs_diff = max(max_abs_diff, diff)
    print(f"  column {c}: max abs diff = {diff:.3e}")
print(f"OVERALL max abs diff: {max_abs_diff:.3e}  ->  {'PASS' if max_abs_diff < 1e-6 else 'FAIL'}")

print("\n=== Sanity: does sklearn.Ridge's default fit_intercept actually matter here? ===")
Xtr, ytr = X[1:], y_real[1:]
r_with = Ridge(alpha=10.0, fit_intercept=True).fit(Xtr, ytr)
r_without = Ridge(alpha=10.0, fit_intercept=False).fit(Xtr, ytr)
xt = X[0:1]
print("pred with intercept:   ", r_with.predict(xt)[0])
print("pred without intercept:", r_without.predict(xt)[0])
print("y_real mean:", y_real.mean())
