#!/usr/bin/env python3
"""Deterministic train/val/test split by WHOLE water year for the 7-day
delta-SWE experiment.

The existing seasonal-CNN split (cmip6_cnn_s0_historical_replay_v1/.../
split_verification.json, seed 20260813) was inspected: it covers all 120
TaiESM1 water years (96 train / 24 val) but is only an 80/20 train/val split
with no test partition, so it is not directly reusable for this experiment's
80/10/10 requirement. A new deterministic split is built here instead, using
a different fixed seed to avoid any accidental collision with that split's
identity, and is documented explicitly.
"""
import json
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1")
OUT_DIR = EXPERIMENT_ROOT / "splits"
SEED = 20260901
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# remainder -> test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    water_years = list(range(1981, 2101))  # TaiESM1: WY1981-WY2100, 120 total
    rng = np.random.default_rng(SEED)
    shuffled = water_years.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    train_wy = sorted(shuffled[:n_train])
    val_wy = sorted(shuffled[n_train:n_train + n_val])
    test_wy = sorted(shuffled[n_train + n_val:])

    assert set(train_wy) | set(val_wy) | set(test_wy) == set(water_years)
    assert not (set(train_wy) & set(val_wy))
    assert not (set(train_wy) & set(test_wy))
    assert not (set(val_wy) & set(test_wy))

    split = {
        "seed": SEED,
        "method": "deterministic whole-water-year shuffle, np.random.default_rng(seed)",
        "train_fraction_target": TRAIN_FRAC,
        "val_fraction_target": VAL_FRAC,
        "note_on_existing_split": (
            "cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/split_verification.json "
            "(seed 20260813) covers all 120 TaiESM1 water years but is only an 80/20 train/val split with no "
            "test partition -- inspected and NOT reused because this experiment needs an 80/10/10 3-way split."
        ),
        "train_water_years": train_wy,
        "val_water_years": val_wy,
        "test_water_years": test_wy,
        "n_train": len(train_wy),
        "n_val": len(val_wy),
        "n_test": len(test_wy),
    }
    out_path = OUT_DIR / "water_year_split.json"
    out_path.write_text(json.dumps(split, indent=2))
    print(json.dumps({k: v for k, v in split.items() if k not in ("train_water_years", "val_water_years", "test_water_years")}, indent=2))
    print(f"train WY ({len(train_wy)}): {train_wy}")
    print(f"val WY ({len(val_wy)}): {val_wy}")
    print(f"test WY ({len(test_wy)}): {test_wy}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
