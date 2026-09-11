#!/usr/bin/env python3
"""Build a new CMIP6 CNN predictor cache whose SWE target is replaced by an empirical
percentile within each CMIP parent model's own TRAINING-year SWE distribution:

    p[m,t] = empirical_CDF_m(SWE[m,t])

empirical_CDF_m fitted ONLY on parent model m's training-split raw SWE (already extracted
with the current UCLA Sierra region/mask via build_wusd3_swe_labels.build_mask_and_area).
Validation-year percentiles are obtained by interpolating against that same training-fitted
CDF, never by re-ranking within validation data. Plotting-position convention:
p_i = (rank_i - 0.5) / N for the N training points, linearly interpolated for any other value
and clipped to the training endpoints (so percentiles never hit exact 0 or 1).
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
NEW_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/data_cache")
UCLA_TARGET_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_mean_mm_plain.npz")
OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_percentile_v1")

EXPECTED_MODELS = {"EC-Earth3:r102i1p1f1", "MIROC6:r1i1p1f1", "MPI-ESM1-2-HR:r3i1p1f1", "TaiESM1:r1i1p1f1"}


def plotting_position_percentile(train_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    order = np.argsort(train_values, kind="mergesort")
    sorted_values = train_values[order]
    n = len(train_values)
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(query_values, sorted_values, plotting_positions)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    NEW_CACHE_DIR.mkdir(parents=True, exist_ok=False)

    with (SOURCE_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    targets = np.load(SOURCE_CACHE_DIR / "targets.npy")
    target_names = json.loads((SOURCE_CACHE_DIR / "target_names.json").read_text())
    assert target_names[0] == "SWE_label"
    n_samples = len(manifest)

    split = json.loads(SPLIT_JSON.read_text())
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()

    model_member = np.asarray([row["model_member"] for row in manifest])
    unique_models = sorted(set(model_member.tolist()))
    print(f"unique parent models in manifest: {unique_models}", flush=True)
    assert set(unique_models) == EXPECTED_MODELS

    raw_swe = targets[:, 0].astype(np.float64)
    assert np.isfinite(raw_swe).all()

    new_percentile = np.full(n_samples, np.nan, dtype=np.float64)
    metadata_rows = []
    print("\n=== per-model train/val sample counts (train-only CDF fit) ===", flush=True)
    for model in sorted(EXPECTED_MODELS):
        model_mask = model_member == model
        train_mask_m = model_mask & np.isin(np.arange(n_samples), train_idx)
        val_mask_m = model_mask & np.isin(np.arange(n_samples), val_idx)
        n_train_m, n_val_m = int(train_mask_m.sum()), int(val_mask_m.sum())
        assert n_train_m > 0, f"no training samples for {model}"
        train_swe_m = raw_swe[train_mask_m]

        # Apply the TRAINING-fitted CDF to every sample of this model (train and val).
        query = raw_swe[model_mask]
        p = plotting_position_percentile(train_swe_m, query)
        new_percentile[model_mask] = p

        p_train = plotting_position_percentile(train_swe_m, train_swe_m)
        print(f"{model}: n_train={n_train_m} n_val={n_val_m} "
              f"train_swe_min={train_swe_m.min():.2f}mm max={train_swe_m.max():.2f}mm "
              f"p_train_min={p_train.min():.4f} p_train_max={p_train.max():.4f} p_train_mean={p_train.mean():.4f}", flush=True)
        assert p_train.min() > 0.0 and p_train.max() < 1.0

        metadata_rows.append({
            "domain_model": model,
            "n_training_samples": n_train_m,
            "n_validation_samples": n_val_m,
            "raw_swe_train_min_mm": float(train_swe_m.min()),
            "raw_swe_train_max_mm": float(train_swe_m.max()),
            "plotting_position_convention": "p_i = (rank_i - 0.5) / N, linearly interpolated for query values, clipped to [0.5/N, (N-0.5)/N]",
        })

    assert np.isfinite(new_percentile).all()
    assert new_percentile.min() > 0.0 and new_percentile.max() < 1.0, "percentile target hit exact 0 or 1 endpoint"
    print(f"\nall percentile targets finite and strictly in (0,1): min={new_percentile.min():.6f} max={new_percentile.max():.6f}", flush=True)

    # UCLA observational percentile: own empirical CDF over all 37 observed years.
    ucla = np.load(UCLA_TARGET_NPZ)
    ucla_wy = ucla["water_year"].astype(np.int32)
    ucla_swe_mm = ucla["sierra_swe_apr1_mean_mm"].astype(np.float64)
    assert ucla_swe_mm.shape[0] == 37
    ucla_percentile = plotting_position_percentile(ucla_swe_mm, ucla_swe_mm)
    assert ucla_percentile.min() > 0.0 and ucla_percentile.max() < 1.0
    print(f"UCLA: n=37 percentile range [{ucla_percentile.min():.4f}, {ucla_percentile.max():.4f}]", flush=True)
    np.savez(
        OUTPUT_ROOT / "ucla_percentile.npz",
        water_year=ucla_wy,
        ucla_swe_mm=ucla_swe_mm,
        ucla_percentile=ucla_percentile,
    )

    metadata_rows.append({
        "domain_model": "UCLA_observational",
        "n_training_samples": 0,
        "n_validation_samples": 37,
        "raw_swe_train_min_mm": float(ucla_swe_mm.min()),
        "raw_swe_train_max_mm": float(ucla_swe_mm.max()),
        "plotting_position_convention": "p_i = (rank_i - 0.5) / 37 over all 37 observed years (no train/val split for observations)",
    })
    meta_csv = OUTPUT_ROOT / "parent_model_percentile_metadata.csv"
    with meta_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"wrote {meta_csv}", flush=True)

    definition_md = OUTPUT_ROOT / "percentile_target_definition.md"
    definition_md.write_text(
        "# Percentile SWE Target Definition\n\n"
        "For each CMIP parent model m, an empirical CDF is fit to that model's April-1 "
        "Sierra SWE (mm) over its TRAINING-split samples only, using the plotting-position "
        "convention `p_i = (rank_i - 0.5) / N` for the N training points. Every sample of "
        "model m - both training and validation - is then converted to a percentile by "
        "linearly interpolating its raw SWE value against this training-fitted CDF "
        "(`numpy.interp`), clipped at the training endpoints `[0.5/N, (N-0.5)/N]` so no "
        "percentile is ever exactly 0 or 1.\n\n"
        "UCLA observational percentiles use the same plotting-position convention applied "
        "to all 37 WY1985-2021 observed values as a single fixed reference distribution "
        "(no train/validation split exists for observations).\n\n"
        "Raw SWE (mm) itself comes from the existing project extraction "
        "(`scripts/build_wusd3_swe_labels.py`, `build_mask_and_area`/`weighted_masked_mean`), "
        "using the current UCLA Sierra region/mask - unchanged from the recent "
        "UCLA-vs-CMIP comparison. Only the final transformation from mm to a [0,1] "
        "percentile is new in this experiment.\n"
    )
    print(f"wrote {definition_md}", flush=True)

    # Write new targets.npy (SWE_label column replaced by percentile; others unchanged).
    new_targets = targets.copy()
    new_targets[:, 0] = new_percentile.astype(np.float32)
    np.save(NEW_CACHE_DIR / "targets.npy", new_targets)
    (NEW_CACHE_DIR / "target_names.json").write_text(json.dumps(target_names) + "\n")
    for fname in ("inputs_physical.npy", "inputs_valid_mask.npy"):
        (NEW_CACHE_DIR / fname).symlink_to(SOURCE_CACHE_DIR / fname)
    (NEW_CACHE_DIR / "manifest.csv").write_text((SOURCE_CACHE_DIR / "manifest.csv").read_text())
    if (SOURCE_CACHE_DIR / "predictor_inventory.json").exists():
        (NEW_CACHE_DIR / "predictor_inventory.json").write_text((SOURCE_CACHE_DIR / "predictor_inventory.json").read_text())

    print(f"\nwrote new predictor cache: {NEW_CACHE_DIR}", flush=True)
    print(f"reused split: {SPLIT_JSON}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
