#!/usr/bin/env python3
"""Extract Z, swe_hat, and cpm_hat (where present) for a trained CPM-auxiliary model
(S0+CPM@Z, S0+CPM@Stage2, or attention+CPM) over simulation train+val and the 37
observational years. Frozen inference only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import FEATURE_Z_CLIP  # noqa: E402
from run_s0_cpm_aux_experiment import S0CPMHead, AttentionCPMHead  # noqa: E402

CMIP_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")


def preprocess(physical, mask, feature_mu, feature_sigma) -> torch.Tensor:
    x = physical.astype(np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    stacked = np.concatenate([x, mask.astype(np.float32)], axis=2)
    assert np.isfinite(stacked).all()
    return torch.from_numpy(stacked)


def run_in_batches(model, inputs, device, batch_size=8):
    z_out, swe_out, cpm_out = [], [], []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            chunk = inputs[start : start + batch_size].to(device)
            out = model(chunk)
            z_out.append(out["z"].detach().cpu().numpy())
            swe_out.append(out["swe_hat"].detach().cpu().numpy())
            if "cpm_hat" in out:
                cpm_out.append(out["cpm_hat"].detach().cpu().numpy())
    result = {"z": np.concatenate(z_out, axis=0), "swe_hat": np.concatenate(swe_out, axis=0)}
    if cpm_out:
        result["cpm_hat"] = np.concatenate(cpm_out, axis=0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["s0", "attention"], required=True)
    parser.add_argument("--cpm-placement", choices=["z", "stage2"], required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    stats = np.load(exp_dir / "normalization_stats.npz")
    checkpoint = torch.load(exp_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model_cls = S0CPMHead if args.architecture == "s0" else AttentionCPMHead
    model = model_cls(args.cpm_placement)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    physical = np.load(CMIP_CACHE_DIR / "inputs_physical.npy")
    mask = np.load(CMIP_CACHE_DIR / "inputs_valid_mask.npy")
    sim_inputs = preprocess(physical, mask, stats["feature_mu"], stats["feature_sigma"])
    sim_out = run_in_batches(model, sim_inputs, device)
    print(f"sim z shape={sim_out['z'].shape}", flush=True)

    obs_physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    obs_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        obs_manifest = list(csv.DictReader(fh))
    obs_water_years = [int(r["water_year"]) for r in obs_manifest]
    obs_inputs = preprocess(obs_physical, obs_mask, stats["feature_mu"], stats["feature_sigma"])
    obs_out = run_in_batches(model, obs_inputs, device)
    print(f"obs z shape={obs_out['z'].shape}", flush=True)

    # Determinism + finite checks.
    obs_out2 = run_in_batches(model, obs_inputs, device)
    assert np.allclose(obs_out["swe_hat"], obs_out2["swe_hat"], atol=1e-6)
    assert np.isfinite(obs_out["swe_hat"]).all()
    if "cpm_hat" in obs_out:
        assert np.allclose(obs_out["cpm_hat"], obs_out2["cpm_hat"], atol=1e-6)
        assert np.isfinite(obs_out["cpm_hat"]).all()

    save_dict = {
        "z_sim": sim_out["z"], "swe_hat_sim": sim_out["swe_hat"],
        "z_obs": obs_out["z"], "swe_hat_obs": obs_out["swe_hat"],
        "obs_water_years": np.asarray(obs_water_years, dtype=np.int32),
        "target_mu": stats["target_mu"], "target_sigma": stats["target_sigma"],
    }
    if "cpm_hat" in sim_out:
        save_dict["cpm_hat_sim"] = sim_out["cpm_hat"]
        save_dict["cpm_hat_obs"] = obs_out["cpm_hat"]
    np.savez(args.output_npz, **save_dict)
    print(f"wrote {args.output_npz}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
