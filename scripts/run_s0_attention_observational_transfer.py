#!/usr/bin/env python3
"""Observational-transfer evaluation for the three already-trained Sierra SWE CNN models.

No training happens here. Every model is loaded from its exact saved checkpoint, frozen
(requires_grad=False, eval()), and run once under torch.no_grad() over the 37-water-year
observational predictor stack. Simulation-training normalization statistics (never
observational statistics) are reused verbatim from the saved normalization_stats.npz.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
    LatentSelfAttentionBlock,
    pearson_r,
    r2_score,
)

# ---------------------------------------------------------------------------
# Fixed paths (all pre-existing; nothing here is rebuilt).
# ---------------------------------------------------------------------------
OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
S0_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/best_checkpoint.pt")
S1_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/S1_static_latent_self_attention/best_checkpoint.pt")
F1_TRAINABLE_CHECKPOINT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1/checkpoints/F1_frozen_S0_Z_attention/best_trainable_downstream.pt")
NORMALIZATION_STATS_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only/normalization_stats.npz")
UCLA_TARGET_FILE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_observational_transfer_v1")

IN_CHANNELS_PER_MONTH = 38  # 19 physical + 19 mask, matching training exactly.

# Simulation-validation R2 recovered from each experiment's own saved metrics (section H).
SIMULATION_VAL_R2 = {
    "S0": 0.375452,  # cmip6_cnn_s0_historical_replay_v1/.../metrics_summary.json best_metrics.swe_r2
    "frozen_s0_attention": 0.374623,  # artifacts/frozen_s0_z_attention_test_v1/README.md, F1 row
    "end_to_end_attention": 0.239554,  # cmip6_cnn_s1_s3_historical_replay_v1/.../metrics_summary.json best_metrics.swe_r2
}


def tensor_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot_params(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def max_param_change(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return max(float((before[name] - after[name]).abs().max()) for name in before)


# ---------------------------------------------------------------------------
# Model reconstruction (architectures copied by reference from snow_ml.cmip6_cnn_experiment
# and scripts/run_frozen_s0_z_attention_test.py; not redefined/altered here).
# ---------------------------------------------------------------------------
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


def load_s0(device: torch.device) -> tuple[M1StaticCNN, dict]:
    checkpoint = torch.load(S0_CHECKPOINT, map_location="cpu", weights_only=False)
    model = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, {"epoch": checkpoint["epoch"], "checkpoint": str(S0_CHECKPOINT)}


def load_s1(device: torch.device) -> tuple[StaticLatentSelfAttentionCNN, dict]:
    checkpoint = torch.load(S1_CHECKPOINT, map_location="cpu", weights_only=False)
    model = StaticLatentSelfAttentionCNN(IN_CHANNELS_PER_MONTH)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, {"epoch": checkpoint["epoch"], "checkpoint": str(S1_CHECKPOINT)}


def load_frozen_s0_attention(device: torch.device) -> tuple[FrozenS0ZAttention, dict]:
    s0_checkpoint = torch.load(S0_CHECKPOINT, map_location="cpu", weights_only=False)
    s0 = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    s0.load_state_dict(s0_checkpoint["model_state_dict"], strict=True)
    encoder = FrozenS0Encoder(s0)
    model = FrozenS0ZAttention(encoder)
    trainable_checkpoint = torch.load(F1_TRAINABLE_CHECKPOINT, map_location="cpu", weights_only=False)
    trainable_state = trainable_checkpoint["trainable_state_dict"]
    result = model.load_state_dict(trainable_state, strict=False)
    # Encoder keys are intentionally absent from the trainable checkpoint (frozen, loaded
    # separately from the S0 checkpoint above) - that is the only allowed "missing" set.
    unexplained_missing = [k for k in result.missing_keys if not k.startswith("encoder.")]
    assert not unexplained_missing, f"Unexplained missing keys: {unexplained_missing}"
    assert not result.unexpected_keys, f"Unexpected keys: {result.unexpected_keys}"
    encoder_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    expected_encoder_keys = {f"encoder.{k}" for k in model.encoder.state_dict().keys()}
    assert set(encoder_missing) == expected_encoder_keys, (
        f"Encoder keys mismatch vs frozen encoder state dict: "
        f"only_in_missing={set(encoder_missing) - expected_encoder_keys}, "
        f"only_in_expected={expected_encoder_keys - set(encoder_missing)}"
    )
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, {
        "epoch": trainable_checkpoint["epoch"],
        "checkpoint": f"encoder={S0_CHECKPOINT} (epoch {s0_checkpoint['epoch']}) + head={F1_TRAINABLE_CHECKPOINT}",
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)  # do not overwrite previous artifacts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # -----------------------------------------------------------------
    # B. Observational predictor data - load and verify metadata.
    # -----------------------------------------------------------------
    physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    valid_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    import csv

    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    with (OBS_CACHE_DIR / "summary.json").open() as fh:
        obs_summary = json.load(fh)

    assert physical.shape == (37, 7, 19, 120, 240), physical.shape
    assert valid_mask.shape == (37, 7, 19, 120, 240), valid_mask.shape
    assert physical.dtype == np.float32, physical.dtype
    manifest_water_years = [int(row["water_year"]) for row in manifest]
    assert manifest_water_years == list(range(1985, 2022)), "manifest water_year not ascending 1985..2021"
    assert obs_summary["month_labels"] == ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    assert obs_summary["predictor_fields"] == [
        "tos", "siconc", "rlut", "zg_500", "ta_850", "ua_850", "va_850", "ua_200", "va_200",
        "hus_850", "psl", "tas", "zg_50", "ta_50", "ua_50", "va_50", "mrso", "thetao_50m", "thetao_100m",
    ]
    finite_physical_frac = float(np.isfinite(physical).mean())
    print(f"physical: shape={physical.shape} dtype={physical.dtype} finite_frac={finite_physical_frac:.4f}", flush=True)
    print("Values are raw physical units (not normalized) - confirmed by generation code "
          "(assemble_era5_cmip6_obs_cache.py writes CDO/xarray output directly, no z-scoring).", flush=True)

    # -----------------------------------------------------------------
    # C. Observational SWE target - load existing UCLA April-1 Sierra extraction.
    # Read via the plain .npz sibling (extracted once from the original .nc using an
    # xarray-capable environment) so this torch-only inference script needs no xarray.
    # -----------------------------------------------------------------
    target_npz = np.load(UCLA_TARGET_FILE.with_name("sierra_swe_apr1_mean_mm_plain.npz"))
    target_water_years = [int(y) for y in target_npz["water_year"]]
    assert len(target_water_years) == 37
    assert target_water_years == list(range(1985, 2022)), "UCLA target water_year not ascending 1985..2021"
    observed_swe_mm = {int(wy): float(v) for wy, v in zip(target_water_years, target_npz["sierra_swe_apr1_mean_mm"], strict=True)}
    assert all(np.isfinite(v) for v in observed_swe_mm.values())
    assert set(observed_swe_mm.keys()) == set(manifest_water_years), "predictor/target water-year sets differ"
    print(f"observed SWE target: {len(observed_swe_mm)} water years, {UCLA_TARGET_FILE}", flush=True)

    # -----------------------------------------------------------------
    # D. Reuse simulation-training normalization exactly.
    # -----------------------------------------------------------------
    stats = np.load(NORMALIZATION_STATS_PATH)
    feature_mu, feature_sigma = stats["feature_mu"], stats["feature_sigma"]
    target_mu, target_sigma = stats["target_mu"], stats["target_sigma"]
    assert feature_mu.shape == (7, 19, 120, 240)
    assert target_mu.shape == (3,)
    swe_target_mu, swe_target_sigma = float(target_mu[0]), float(target_sigma[0])
    print(f"normalization stats: {NORMALIZATION_STATS_PATH} swe_target_mu={swe_target_mu:.4f} swe_target_sigma={swe_target_sigma:.4f}", flush=True)

    x = physical.astype(np.float32)
    x = (x - feature_mu) / feature_sigma
    x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x = np.where(valid_mask > 0.5, x, np.nan).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    stacked = np.concatenate([x, valid_mask.astype(np.float32)], axis=2)  # channel axis
    assert stacked.shape == (37, 7, 38, 120, 240)
    assert np.isfinite(stacked).all(), "NaN/Inf present after preprocessing"
    inputs = torch.from_numpy(stacked)
    print(f"preprocessed inputs: shape={tuple(inputs.shape)} finite_ok=True", flush=True)

    # Scale check (diagnostic only - not used to alter anything).
    scale_summary = {
        "obs_normalized_input_mean": float(stacked[:, :, :19].mean()),
        "obs_normalized_input_std": float(stacked[:, :, :19].std()),
        "obs_normalized_input_min": float(stacked[:, :, :19].min()),
        "obs_normalized_input_max": float(stacked[:, :, :19].max()),
        "note": "computed after clip to +/-{:.1f}; compare against simulation train-split z-scores which are ~N(0,1) by construction".format(FEATURE_Z_CLIP),
    }

    ordered_water_years = manifest_water_years  # already verified ascending 1985..2021
    observed_vector = np.asarray([observed_swe_mm[wy] for wy in ordered_water_years], dtype=np.float64)

    # -----------------------------------------------------------------
    # E + F. Load each model, run frozen inference, sanity-check.
    # -----------------------------------------------------------------
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    checkpoint_info: dict[str, dict] = {}

    loaders = {
        "S0": load_s0,
        "frozen_s0_attention": load_frozen_s0_attention,
        "end_to_end_attention": load_s1,
    }

    for name, loader in loaders.items():
        print(f"=== {name} ===", flush=True)
        model, info = loader(device)
        checkpoint_info[name] = info

        params_before = snapshot_params(model)
        digest_before = tensor_digest(model)

        def run_forward() -> np.ndarray:
            with torch.no_grad():
                inp = inputs.to(device)
                if name == "S0":
                    out = model(inp)["swe_hat"]
                elif name == "end_to_end_attention":
                    out = model(inp)["swe_hat"]
                else:
                    out = model(inp)
                return out.detach().cpu().numpy()

        pred_z_1 = run_forward()
        pred_z_2 = run_forward()
        assert np.allclose(pred_z_1, pred_z_2, atol=1e-6), f"{name}: non-deterministic eval-mode output"
        assert np.isfinite(pred_z_1).all(), f"{name}: non-finite predictions"

        params_after = snapshot_params(model)
        digest_after = tensor_digest(model)
        assert digest_before == digest_after, f"{name}: encoder/parameters changed during inference"
        max_change = max_param_change(params_before, params_after)
        assert max_change == 0.0, f"{name}: parameter max change {max_change} != 0"

        pred_physical = pred_z_1 * swe_target_sigma + swe_target_mu
        predictions[name] = pred_physical

        r2 = r2_score(observed_vector, pred_physical)
        r = pearson_r(observed_vector, pred_physical)
        rmse = float(np.sqrt(np.mean((observed_vector - pred_physical) ** 2)))
        mae = float(np.mean(np.abs(observed_vector - pred_physical)))
        obs_anom_sign = np.sign(observed_vector - swe_target_mu)
        pred_anom_sign = np.sign(pred_physical - swe_target_mu)
        sign_acc = float(np.mean(obs_anom_sign == pred_anom_sign))

        results[name] = {
            "checkpoint": info["checkpoint"],
            "checkpoint_epoch": info["epoch"],
            "simulation_val_r2": SIMULATION_VAL_R2[name],
            "observational_r2": r2,
            "observational_pearson_r": r,
            "observational_rmse_mm": rmse,
            "observational_mae_mm": mae,
            "observational_sign_accuracy": sign_acc,
            "max_parameter_change_during_inference": max_change,
            "deterministic": True,
        }
        print(json.dumps(results[name], indent=2), flush=True)

    # -----------------------------------------------------------------
    # I. Outputs
    # -----------------------------------------------------------------
    import csv as csv_module

    predictions_csv = OUTPUT_ROOT / "observational_predictions.csv"
    with predictions_csv.open("w", newline="") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["water_year", "observed_swe_mm", "s0_prediction_mm", "frozen_s0_attention_prediction_mm", "end_to_end_attention_prediction_mm"])
        for i, wy in enumerate(ordered_water_years):
            writer.writerow([
                wy,
                f"{observed_vector[i]:.6f}",
                f"{predictions['S0'][i]:.6f}",
                f"{predictions['frozen_s0_attention'][i]:.6f}",
                f"{predictions['end_to_end_attention'][i]:.6f}",
            ])

    metrics_csv = OUTPUT_ROOT / "observational_metrics.csv"
    with metrics_csv.open("w", newline="") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["model", "checkpoint", "simulation_val_r2", "observational_r2", "observational_pearson_r", "observational_rmse_mm", "observational_mae_mm", "observational_sign_accuracy"])
        for name in ("S0", "frozen_s0_attention", "end_to_end_attention"):
            row = results[name]
            writer.writerow([name, row["checkpoint"], row["simulation_val_r2"], row["observational_r2"], row["observational_pearson_r"], row["observational_rmse_mm"], row["observational_mae_mm"], row["observational_sign_accuracy"]])

    # Diagnostic plots.
    fig, axis = plt.subplots(figsize=(11, 5), dpi=160)
    axis.plot(ordered_water_years, observed_vector, marker="o", color="black", label="Observed (UCLA)")
    for name, color in (("S0", "tab:blue"), ("frozen_s0_attention", "tab:orange"), ("end_to_end_attention", "tab:green")):
        axis.plot(ordered_water_years, predictions[name], marker="o", markersize=3, color=color, label=name)
    axis.set_xlabel("Water year")
    axis.set_ylabel("April 1 Sierra SWE (mm)")
    axis.set_title("Observational transfer: 37-year SWE time series")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "observational_timeseries.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160, sharex=True, sharey=True)
    for axis, (name, color) in zip(axes, (("S0", "tab:blue"), ("frozen_s0_attention", "tab:orange"), ("end_to_end_attention", "tab:green")), strict=True):
        axis.scatter(observed_vector, predictions[name], color=color, s=20)
        lims = [min(observed_vector.min(), predictions[name].min()), max(observed_vector.max(), predictions[name].max())]
        axis.plot(lims, lims, "k--", linewidth=0.8)
        axis.set_xlabel("Observed SWE (mm)")
        axis.set_title(f"{name}\nR2={results[name]['observational_r2']:.3f}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted SWE (mm)")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "observational_scatter.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), dpi=160)
    names = ["S0", "frozen_s0_attention", "end_to_end_attention"]
    sim_r2 = [SIMULATION_VAL_R2[n] for n in names]
    obs_r2 = [results[n]["observational_r2"] for n in names]
    x_pos = np.arange(len(names))
    width = 0.35
    axis.bar(x_pos - width / 2, sim_r2, width, label="Simulation validation R2", color="tab:gray")
    axis.bar(x_pos + width / 2, obs_r2, width, label="Observational R2", color="tab:red")
    axis.set_xticks(x_pos)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("R2")
    axis.set_title("Simulation vs observational R2")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "simulation_vs_observational_r2.png")
    plt.close(fig)

    # Ranking comparison.
    sim_ranking = sorted(names, key=lambda n: SIMULATION_VAL_R2[n], reverse=True)
    obs_ranking = sorted(names, key=lambda n: results[n]["observational_r2"], reverse=True)
    ranking_unchanged = sim_ranking == obs_ranking

    full_report = {
        "checkpoints": checkpoint_info,
        "normalization_stats_path": str(NORMALIZATION_STATS_PATH),
        "observational_cache_dir": str(OBS_CACHE_DIR),
        "ucla_target_file": str(UCLA_TARGET_FILE),
        "n_samples": 37,
        "water_year_range": [1985, 2021],
        "results": results,
        "simulation_ranking_best_to_worst": sim_ranking,
        "observational_ranking_best_to_worst": obs_ranking,
        "ranking_unchanged": ranking_unchanged,
        "scale_check": scale_summary,
        "obs_predictor_finite_fraction": finite_physical_frac,
    }
    (OUTPUT_ROOT / "full_report.json").write_text(json.dumps(full_report, indent=2) + "\n")
    print(json.dumps(full_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
