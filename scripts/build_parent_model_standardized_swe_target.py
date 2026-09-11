#!/usr/bin/env python3
"""Build a new CMIP6 CNN predictor cache whose SWE target is replaced by a
parent-model-relative standardized anomaly:

    y[m,t] = (SWE[m,t] - mu_m) / sigma_m

mu_m, sigma_m computed from TRAINING-split samples of parent model m only, using the
existing raw SWE_label column (already extracted with the current UCLA Sierra
region/mask via build_wusd3_swe_labels.build_mask_and_area / DEFAULT_SIERRA_REGION -
not re-extracted here). Physical predictor and valid-mask arrays are symlinked, not
copied, to avoid duplicating heavy data on $HOME or $PSCRATCH needlessly.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

SOURCE_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
SPLIT_JSON = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/split.json")
NEW_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_std_anomaly_v1/data_cache")
UCLA_TARGET_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_mean_mm_plain.npz")
OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_std_anomaly_v1")

EXPECTED_MODELS = {"EC-Earth3:r102i1p1f1", "MIROC6:r1i1p1f1", "MPI-ESM1-2-HR:r3i1p1f1", "TaiESM1:r1i1p1f1"}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    NEW_CACHE_DIR.mkdir(parents=True, exist_ok=False)

    with (SOURCE_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    targets = np.load(SOURCE_CACHE_DIR / "targets.npy")
    target_names = json.loads((SOURCE_CACHE_DIR / "target_names.json").read_text())
    assert target_names[0] == "SWE_label"
    n_samples = len(manifest)
    assert targets.shape[0] == n_samples

    split = json.loads(SPLIT_JSON.read_text())
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    assert set(train_idx.tolist()) & set(val_idx.tolist()) == set(), "train/val overlap"

    model_member = np.asarray([row["model_member"] for row in manifest])
    unique_models = sorted(set(model_member.tolist()))
    print(f"unique parent models in manifest: {unique_models}", flush=True)
    assert set(unique_models) == EXPECTED_MODELS, f"unexpected model set: {unique_models}"

    raw_swe = targets[:, 0].astype(np.float64)
    assert np.isfinite(raw_swe).all(), "non-finite raw SWE_label in source cache"

    new_swe = np.full(n_samples, np.nan, dtype=np.float64)
    normalization_rows = []
    print("\n=== per-model train/val sample counts and raw SWE stats (train-only) ===", flush=True)
    for model in sorted(EXPECTED_MODELS):
        model_mask = model_member == model
        train_mask_m = model_mask & np.isin(np.arange(n_samples), train_idx)
        val_mask_m = model_mask & np.isin(np.arange(n_samples), val_idx)
        n_train_m = int(train_mask_m.sum())
        n_val_m = int(val_mask_m.sum())
        assert n_train_m > 0, f"no training samples for {model}"
        train_swe_m = raw_swe[train_mask_m]
        mu_m = float(train_swe_m.mean())
        sigma_m = float(train_swe_m.std())
        assert sigma_m > 1e-9, f"degenerate sigma for {model}"
        print(f"{model}: n_train={n_train_m} n_val={n_val_m} mu_m={mu_m:.4f}mm sigma_m={sigma_m:.4f}mm", flush=True)

        both_mask_m = model_mask
        new_swe[both_mask_m] = (raw_swe[both_mask_m] - mu_m) / sigma_m

        # Verify standardized TRAIN targets for this model are ~mean 0, std 1.
        standardized_train = (raw_swe[train_mask_m] - mu_m) / sigma_m
        assert abs(float(standardized_train.mean())) < 1e-6, f"{model} standardized train mean not ~0"
        assert abs(float(standardized_train.std()) - 1.0) < 1e-6, f"{model} standardized train std not ~1"

        normalization_rows.append({
            "domain_model": model,
            "mu_swe_mm": mu_m,
            "sigma_swe_mm": sigma_m,
            "number_of_training_samples": n_train_m,
            "number_of_validation_samples": n_val_m,
        })

    assert np.isfinite(new_swe).all(), "NaN/Inf remains after standardization - some sample not covered by a parent model"
    print("\nno NaN/Inf in standardized target: OK", flush=True)

    # UCLA observational mu/sigma (for documentation + observational anomaly target).
    ucla = np.load(UCLA_TARGET_NPZ)
    ucla_swe_mm = ucla["sierra_swe_apr1_mean_mm"].astype(np.float64)
    assert ucla_swe_mm.shape[0] == 37
    mu_obs = float(ucla_swe_mm.mean())
    sigma_obs = float(ucla_swe_mm.std())
    print(f"UCLA: n=37 mu_obs={mu_obs:.4f}mm sigma_obs={sigma_obs:.4f}mm", flush=True)
    normalization_rows.append({
        "domain_model": "UCLA_observational",
        "mu_swe_mm": mu_obs,
        "sigma_swe_mm": sigma_obs,
        "number_of_training_samples": 0,
        "number_of_validation_samples": 37,
    })

    # Write target_normalization.csv
    norm_csv = OUTPUT_ROOT / "target_normalization.csv"
    with norm_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(normalization_rows[0].keys()))
        writer.writeheader()
        writer.writerows(normalization_rows)
    print(f"wrote {norm_csv}", flush=True)

    # Write new targets.npy (SWE_label column replaced; CPM_label/AQM_label unchanged).
    new_targets = targets.copy()
    new_targets[:, 0] = new_swe.astype(np.float32)
    np.save(NEW_CACHE_DIR / "targets.npy", new_targets)
    (NEW_CACHE_DIR / "target_names.json").write_text(json.dumps(target_names) + "\n")

    # Symlink heavy arrays (do not duplicate).
    for fname in ("inputs_physical.npy", "inputs_valid_mask.npy"):
        (NEW_CACHE_DIR / fname).symlink_to(SOURCE_CACHE_DIR / fname)
    # Copy small files.
    (NEW_CACHE_DIR / "manifest.csv").write_text((SOURCE_CACHE_DIR / "manifest.csv").read_text())
    if (SOURCE_CACHE_DIR / "predictor_inventory.json").exists():
        (NEW_CACHE_DIR / "predictor_inventory.json").write_text((SOURCE_CACHE_DIR / "predictor_inventory.json").read_text())

    print(f"\nwrote new predictor cache: {NEW_CACHE_DIR}", flush=True)
    print(f"reused split: {SPLIT_JSON}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
