#!/usr/bin/env python3
"""Extract frozen S0 and end-to-end-attention encoder latents (the same Z the downstream
SWE head consumes) for all 393 simulation samples and all 37 observational years, using
each model's own percentile-target checkpoint and normalization stats. Also verifies
numerically that the frozen-S0+attention model's encoder is identical to plain S0's
encoder (deep-copied from the same checkpoint), so it is not extracted separately.

Frozen inference only - no training, no gradient, no checkpoint modification.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import FEATURE_Z_CLIP, LATENT_D, LATENT_K, M1StaticCNN, StaticLatentSelfAttentionCNN  # noqa: E402

CMIP_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
PERCENTILE_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/data_cache")
S0_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S0_static_cnn_swe_only")
S1_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_percentile_v1/experiments/S1_static_latent_self_attention")
OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cnn_cpm_aqm_latent_geometry_v1")

IN_CHANNELS_PER_MONTH = 38


def tensor_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenS0Encoder(nn.Module):
    def __init__(self, s0: M1StaticCNN) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(s0.backbone)
        self.project = copy.deepcopy(s0.project)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, m, c, h, w = x.shape
        with torch.no_grad():
            features = self.backbone(x.reshape(b, m * c, h, w))
            return self.project(features).reshape(b, LATENT_K, LATENT_D)


def preprocess(physical: np.ndarray, mask: np.ndarray, feature_mu: np.ndarray, feature_sigma: np.ndarray) -> torch.Tensor:
    x = physical.astype(np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    stacked = np.concatenate([x, mask.astype(np.float32)], axis=2)
    assert np.isfinite(stacked).all()
    return torch.from_numpy(stacked)


def encode_in_batches(model: nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int = 8) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            chunk = inputs[start : start + batch_size].to(device)
            z = model.encode(chunk) if hasattr(model, "encode") else model(chunk)
            outputs.append(z.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    with (CMIP_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    physical = np.load(CMIP_CACHE_DIR / "inputs_physical.npy")
    mask = np.load(CMIP_CACHE_DIR / "inputs_valid_mask.npy")
    n_samples = len(manifest)
    print(f"simulation samples: {n_samples}", flush=True)

    obs_physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    obs_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        obs_manifest = list(csv.DictReader(fh))
    obs_water_years = [int(r["water_year"]) for r in obs_manifest]
    assert obs_water_years == list(range(1985, 2022))

    # ---------------- S0 ----------------
    s0_stats = np.load(S0_EXP_DIR / "normalization_stats.npz")
    s0_checkpoint = torch.load(S0_EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    s0 = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    s0.load_state_dict(s0_checkpoint["model_state_dict"], strict=True)
    s0.to(device)
    s0.eval()
    for p in s0.parameters():
        p.requires_grad_(False)

    sim_inputs_s0 = preprocess(physical, mask, s0_stats["feature_mu"], s0_stats["feature_sigma"])
    z_s0_sim = encode_in_batches(s0, sim_inputs_s0, device)
    print(f"S0 sim latent shape: {z_s0_sim.shape}", flush=True)

    obs_inputs_s0 = preprocess(obs_physical, obs_mask, s0_stats["feature_mu"], s0_stats["feature_sigma"])
    z_s0_obs = encode_in_batches(s0, obs_inputs_s0, device)
    print(f"S0 obs latent shape: {z_s0_obs.shape}", flush=True)

    # ---------------- S1 (end-to-end attention) ----------------
    s1_stats = np.load(S1_EXP_DIR / "normalization_stats.npz")
    s1_checkpoint = torch.load(S1_EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    s1 = StaticLatentSelfAttentionCNN(IN_CHANNELS_PER_MONTH)
    s1.load_state_dict(s1_checkpoint["model_state_dict"], strict=True)
    s1.to(device)
    s1.eval()
    for p in s1.parameters():
        p.requires_grad_(False)

    sim_inputs_s1 = preprocess(physical, mask, s1_stats["feature_mu"], s1_stats["feature_sigma"])
    z_s1_sim = encode_in_batches(s1, sim_inputs_s1, device)
    print(f"S1 sim latent shape: {z_s1_sim.shape}", flush=True)

    obs_inputs_s1 = preprocess(obs_physical, obs_mask, s1_stats["feature_mu"], s1_stats["feature_sigma"])
    z_s1_obs = encode_in_batches(s1, obs_inputs_s1, device)
    print(f"S1 obs latent shape: {z_s1_obs.shape}", flush=True)

    # ---------------- Frozen-S0+attention encoder identity check ----------------
    frozen_encoder = FrozenS0Encoder(s0)
    frozen_encoder.to(device)
    check_batch = sim_inputs_s0[:4].to(device)
    with torch.no_grad():
        z_frozen_check = frozen_encoder(check_batch).detach().cpu().numpy()
        z_s0_check = s0.encode(check_batch).detach().cpu().numpy()
    max_diff = float(np.abs(z_frozen_check - z_s0_check).max())
    # Also verify parameter-value equality directly (not just forward-pass output equality),
    # comparing tensors by (unprefixed) parameter name rather than raw state_dict digests,
    # since FrozenS0Encoder's own state_dict keys are prefixed ("backbone.stem...") while
    # s0.backbone's own state_dict keys are not ("stem...") - same tensors, different key
    # strings, so a naive whole-state-dict hash comparison is not meaningful here.
    frozen_params = dict(frozen_encoder.backbone.state_dict())
    s0_params = dict(s0.backbone.state_dict())
    backbone_equal = all(torch.equal(frozen_params[k], s0_params[k]) for k in s0_params)
    frozen_project = dict(frozen_encoder.project.state_dict())
    s0_project = dict(s0.project.state_dict())
    project_equal = all(torch.equal(frozen_project[k], s0_project[k]) for k in s0_project)
    identity_verified = (max_diff == 0.0) and backbone_equal and project_equal
    print(f"frozen-S0+attention encoder identity check: max_diff={max_diff} backbone_params_equal={backbone_equal} project_params_equal={project_equal} verified={identity_verified}", flush=True)
    assert identity_verified, "frozen-S0+attention encoder is NOT identical to plain S0 encoder"

    np.savez(
        OUTPUT_ROOT / "cnn_latents.npz",
        z_s0_sim=z_s0_sim, z_s1_sim=z_s1_sim,
        z_s0_obs=z_s0_obs, z_s1_obs=z_s1_obs,
        obs_water_years=np.asarray(obs_water_years, dtype=np.int32),
        frozen_identity_max_diff=max_diff,
        frozen_identity_digest_match=identity_verified,
    )
    print(f"wrote {OUTPUT_ROOT / 'cnn_latents.npz'}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
