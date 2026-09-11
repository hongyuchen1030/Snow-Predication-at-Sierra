#!/usr/bin/env python3
"""Forward-pass smoke test: run the frozen S0/S1/S2/U1 CMIP6 CNN checkpoints on the new
ERA5/ERA5-Land/EN4 observational cache built by assemble_era5_cmip6_obs_cache.py.

For each checkpoint, applies that checkpoint's own normalization_stats.npz (never recomputed),
runs the exact CMIP6TensorDataset preprocessing (see build_inputs below, mirrored verbatim from
scripts/run_cmip6_upstream_aux_diagnostics.py:63-76), and asserts no NaN/Inf and a successful
forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from snow_ml.cmip6_cnn_experiment import FEATURE_Z_CLIP, build_model  # noqa: E402

CHECKPOINTS = {
    "S0": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only"
    ),
    "S1": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S1_static_latent_self_attention"
    ),
    "S2": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S2_static_swe_token"
    ),
    "U1": Path(
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_upstream_aux_v1/experiments/U1_stage2_cpm_aqm"
    ),
}


def build_inputs(*, physical_data: np.ndarray, valid_mask: np.ndarray, feature_mu: np.ndarray, feature_sigma: np.ndarray) -> np.ndarray:
    x = np.asarray(physical_data, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([x, mask], axis=2).astype(np.float32)


def run_checkpoint_smoke_test(
    name: str,
    experiment_dir: Path,
    physical: np.ndarray,
    mask: np.ndarray,
    water_years: np.ndarray,
    n_probe: int | None,
) -> dict:
    config = json.loads((experiment_dir / "config.json").read_text())
    norm = np.load(experiment_dir / "normalization_stats.npz")
    feature_mu = np.asarray(norm["feature_mu"], dtype=np.float32)
    feature_sigma = np.asarray(norm["feature_sigma"], dtype=np.float32)

    if feature_mu.shape != physical.shape[1:]:
        raise ValueError(f"{name}: feature_mu shape {feature_mu.shape} != per-sample physical shape {physical.shape[1:]}")
    if feature_sigma.shape != physical.shape[1:]:
        raise ValueError(f"{name}: feature_sigma shape {feature_sigma.shape} != per-sample physical shape {physical.shape[1:]}")

    model = build_model(config["architecture"], in_channels_per_month=int(physical.shape[2] * 2))
    checkpoint = torch.load(experiment_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n = physical.shape[0] if n_probe is None else min(n_probe, physical.shape[0])
    phys_slice = physical[:n]
    mask_slice = mask[:n]

    valid_bool = mask_slice.astype(bool)
    pre_norm_finite_where_valid = bool(np.isfinite(phys_slice[valid_bool]).all()) if valid_bool.any() else True

    stacked = build_inputs(physical_data=phys_slice, valid_mask=mask_slice, feature_mu=feature_mu, feature_sigma=feature_sigma)
    post_finite = bool(np.isfinite(stacked).all())
    if not post_finite:
        raise AssertionError(f"{name}: NaN/Inf present in model input tensor after preprocessing")

    with torch.no_grad():
        outputs = model(torch.from_numpy(stacked).to(device))

    result: dict = {
        "checkpoint": name,
        "architecture": config["architecture"],
        "n_samples_probed": int(n),
        "water_years_probed": [int(w) for w in water_years[:n]],
        "input_shape": list(stacked.shape),
        "pre_normalization_finite_where_valid": pre_norm_finite_where_valid,
        "post_preprocessing_finite": post_finite,
    }
    all_outputs_finite = True
    for key, tensor in outputs.items():
        arr = tensor.detach().cpu().numpy()
        finite = bool(np.isfinite(arr).all())
        all_outputs_finite = all_outputs_finite and finite
        result[f"{key}_shape"] = list(arr.shape)
        result[f"{key}_finite"] = finite
        if not finite:
            result[f"{key}_nan_count"] = int(np.isnan(arr).sum())
            result[f"{key}_inf_count"] = int(np.isinf(arr).sum())
        elif arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            flat = arr.reshape(-1)
            result[f"{key}_sample_stats"] = {"min": float(flat.min()), "max": float(flat.max()), "mean": float(flat.mean())}

    result["status"] = "PASS" if (pre_norm_finite_where_valid and post_finite and all_outputs_finite) else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-cache-dir",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational",
    )
    parser.add_argument("--n-probe", type=int, default=None, help="Limit to first N water years (default: all 37)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cache_dir = Path(args.data_cache_dir)
    physical = np.load(cache_dir / "inputs_physical.npy")
    mask = np.load(cache_dir / "inputs_valid_mask.npy")
    summary = json.loads((cache_dir / "summary.json").read_text())
    water_years = np.asarray(summary["water_years"], dtype=np.int64)

    print(f"Loaded observational cache: physical={physical.shape} mask={mask.shape} water_years={water_years.tolist()}", flush=True)

    results = []
    for name, experiment_dir in CHECKPOINTS.items():
        print(f"\n=== {name} ({experiment_dir.name}) ===", flush=True)
        result = run_checkpoint_smoke_test(name, experiment_dir, physical, mask, water_years, args.n_probe)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

    overall_pass = all(r["status"] == "PASS" for r in results)
    out_path = Path(args.output) if args.output else cache_dir / "smoke_test_results.json"
    out_path.write_text(json.dumps({"overall_status": "PASS" if overall_pass else "FAIL", "results": results}, indent=2))
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
