#!/usr/bin/env python3
"""Simple line-plot comparison: UCLA-observed April-1 Sierra SWE vs. frozen inference from
S0/S1/S2/S3 (the four base historical-replay architectures) and the two frozen-encoder
diagnostic variants (F0, F1), all run on the same 37-water-year observational tensor.

No training. Every model is loaded from its exact saved checkpoint, frozen, eval() only.
"""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import (  # noqa: E402
    FEATURE_Z_CLIP,
    LATENT_D,
    LATENT_K,
    M1StaticCNN,
    StaticLatentSelfAttentionCNN,
    StaticSWETokenAttentionCNN,
    StaticResidualGatedAttentionCNN,
    LatentSelfAttentionBlock,
)

OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
NORMALIZATION_STATS_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/normalization_stats.npz")
UCLA_TARGET_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_mean_mm_plain.npz")

S0_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt")
S1_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/S1_static_latent_self_attention/best_checkpoint.pt")
S2_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/S2_static_swe_token/best_checkpoint.pt")
S3_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/S3_static_residual_gated_attention/best_checkpoint.pt")
F0_TRAINABLE_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1/checkpoints/F0_frozen_S0_Z_MLP/best_trainable_downstream.pt")
F1_TRAINABLE_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1/checkpoints/F1_frozen_S0_Z_attention/best_trainable_downstream.pt")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_observational_transfer_v1")
IN_CHANNELS_PER_MONTH = 38


class FrozenS0Encoder(nn.Module):
    def __init__(self, s0: M1StaticCNN) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(s0.backbone)
        self.project = copy.deepcopy(s0.project)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        with torch.no_grad():
            features = self.backbone(x.reshape(batch_size, months * channels, height, width))
            return self.project(features).reshape(batch_size, LATENT_K, LATENT_D)


class FrozenS0ZMLP(nn.Module):
    def __init__(self, encoder: FrozenS0Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.swe_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.swe_head(z.reshape(z.shape[0], -1)).squeeze(-1)


class FrozenS0ZAttention(nn.Module):
    def __init__(self, encoder: FrozenS0Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_raw = self.encoder(x)
        attn_in = self.latent_block.norm1(z_raw)
        attn_out, _ = self.latent_block.attn(attn_in, attn_in, attn_in, need_weights=False)
        z_attn = z_raw + attn_out
        z_attn = z_attn + self.latent_block.ffn(self.latent_block.norm2(z_attn))
        return self.swe_head(z_attn.reshape(z_attn.shape[0], -1)).squeeze(-1)


def load_plain(cls, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = cls(IN_CHANNELS_PER_MONTH)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_s0(device: torch.device):
    checkpoint = torch.load(S0_CHECKPOINT, map_location="cpu", weights_only=False)
    model = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_frozen(head_cls, trainable_checkpoint_path: Path, device: torch.device):
    s0_checkpoint = torch.load(S0_CHECKPOINT, map_location="cpu", weights_only=False)
    s0 = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    s0.load_state_dict(s0_checkpoint["model_state_dict"], strict=True)
    encoder = FrozenS0Encoder(s0)
    model = head_cls(encoder)
    trainable_checkpoint = torch.load(trainable_checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(trainable_checkpoint["trainable_state_dict"], strict=False)
    expected_encoder_keys = {f"encoder.{k}" for k in model.encoder.state_dict().keys()}
    assert set(result.missing_keys) == expected_encoder_keys
    assert not result.unexpected_keys
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    valid_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    water_years = [int(row["water_year"]) for row in manifest]
    assert water_years == list(range(1985, 2022))

    target_npz = np.load(UCLA_TARGET_NPZ)
    observed = {int(wy): float(v) for wy, v in zip(target_npz["water_year"], target_npz["sierra_swe_apr1_mean_mm"], strict=True)}
    observed_vector = np.asarray([observed[wy] for wy in water_years], dtype=np.float64)

    stats = np.load(NORMALIZATION_STATS_PATH)
    feature_mu, feature_sigma = stats["feature_mu"], stats["feature_sigma"]
    swe_mu, swe_sigma = float(stats["target_mu"][0]), float(stats["target_sigma"][0])

    x = physical.astype(np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(valid_mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    stacked = np.concatenate([x, valid_mask.astype(np.float32)], axis=2)
    inputs = torch.from_numpy(stacked)

    def infer(model) -> np.ndarray:
        with torch.no_grad():
            out = model(inputs.to(device))
            if isinstance(out, dict):
                out = out["swe_hat"]
            return out.detach().cpu().numpy() * swe_sigma + swe_mu

    predictions: dict[str, np.ndarray] = {}
    print("=== S0 ===", flush=True)
    predictions["S0"] = infer(load_s0(device))
    print("=== S1 ===", flush=True)
    predictions["S1"] = infer(load_plain(StaticLatentSelfAttentionCNN, S1_CHECKPOINT, device))
    print("=== S2 ===", flush=True)
    predictions["S2"] = infer(load_plain(StaticSWETokenAttentionCNN, S2_CHECKPOINT, device))
    print("=== S3 ===", flush=True)
    predictions["S3"] = infer(load_plain(StaticResidualGatedAttentionCNN, S3_CHECKPOINT, device))
    print("=== F0_frozen_S0_Z_MLP ===", flush=True)
    predictions["F0_frozen_S0_Z_MLP"] = infer(load_frozen(FrozenS0ZMLP, F0_TRAINABLE_CHECKPOINT, device))
    print("=== F1_frozen_S0_Z_attention ===", flush=True)
    predictions["F1_frozen_S0_Z_attention"] = infer(load_frozen(FrozenS0ZAttention, F1_TRAINABLE_CHECKPOINT, device))

    for name, values in predictions.items():
        assert np.isfinite(values).all(), name

    # Save the combined predictions table.
    all_csv = OUTPUT_ROOT / "all_seven_models_predictions.csv"
    with all_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["water_year", "observed_swe_mm"] + list(predictions.keys())
        writer.writerow(header)
        for i, wy in enumerate(water_years):
            writer.writerow([wy, f"{observed_vector[i]:.6f}"] + [f"{predictions[n][i]:.6f}" for n in predictions])

    # Plot 1: observed + S0,S1,S2,S3 (5 lines).
    fig, axis = plt.subplots(figsize=(12, 5.5), dpi=160)
    axis.plot(water_years, observed_vector, marker="o", color="black", linewidth=2.0, label="Observed (UCLA)")
    for name, color in (("S0", "tab:blue"), ("S1", "tab:orange"), ("S2", "tab:green"), ("S3", "tab:red")):
        axis.plot(water_years, predictions[name], marker="o", markersize=3, color=color, label=name)
    axis.set_xlabel("Water year")
    axis.set_ylabel("April 1 Sierra SWE (mm)")
    axis.set_title("Observed vs. S0-S3 simulation-trained models (frozen inference on observations)")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "lineplot_observed_vs_S0_S1_S2_S3.png")
    plt.close(fig)

    # Plot 2: observed + S0,S1,S2,S3 + F0,F1 (7 lines).
    fig, axis = plt.subplots(figsize=(12, 5.5), dpi=160)
    axis.plot(water_years, observed_vector, marker="o", color="black", linewidth=2.0, label="Observed (UCLA)")
    colors = {
        "S0": "tab:blue", "S1": "tab:orange", "S2": "tab:green", "S3": "tab:red",
        "F0_frozen_S0_Z_MLP": "tab:purple", "F1_frozen_S0_Z_attention": "tab:brown",
    }
    for name, color in colors.items():
        axis.plot(water_years, predictions[name], marker="o", markersize=3, color=color, label=name)
    axis.set_xlabel("Water year")
    axis.set_ylabel("April 1 Sierra SWE (mm)")
    axis.set_title("Observed vs. all 6 models: S0-S3 plus frozen-encoder CNN variants (frozen inference on observations)")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "lineplot_observed_vs_all_six_models.png")
    plt.close(fig)

    print(f"wrote {all_csv}", flush=True)
    print(f"wrote {OUTPUT_ROOT / 'lineplot_observed_vs_S0_S1_S2_S3.png'}", flush=True)
    print(f"wrote {OUTPUT_ROOT / 'lineplot_observed_vs_all_six_models.png'}", flush=True)


if __name__ == "__main__":
    main()
