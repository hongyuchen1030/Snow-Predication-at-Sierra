#!/usr/bin/env python3
"""Diagnostic ONLY: compare the simulation-trained CNN SWE target distribution
against the observed UCLA WUS-SR SWE distribution, both computed over the same
UCLA Sierra-region footprint (DEFAULT_SIERRA_REGION + build_sierra_mask +
spherical area weighting, applied natively to each dataset's own grid).

No training. No checkpoint modification. No XAI / predictor-side diagnostics.
Every model is loaded frozen (requires_grad=False, eval()) and run once under
torch.no_grad(); saved predictions are reused wherever they already exist.
"""

from __future__ import annotations

import copy
import csv
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import (  # noqa: E402
    FEATURE_Z_CLIP,
    LATENT_D,
    LATENT_K,
    CMIP6TensorDataset,
    M1StaticCNN,
    StaticLatentSelfAttentionCNN,
    LatentSelfAttentionBlock,
    collate_with_metadata,
    pearson_r,
    r2_score,
    read_manifest,
)

# ---------------------------------------------------------------------------
# Fixed, pre-existing paths. Nothing here is rebuilt from scratch.
# ---------------------------------------------------------------------------
CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache")
S0_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s0_historical_replay_v1/experiments/S0_static_cnn_swe_only")
S1_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_s1_s3_historical_replay_v1/experiments/S1_static_latent_self_attention")
HEAD_SCREEN_SPLIT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only/split.json")
FROZEN_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/frozen_s0_z_attention_test_v1")
F0_CHECKPOINT = FROZEN_DIR / "checkpoints/F0_frozen_S0_Z_MLP/best_trainable_downstream.pt"
F1_CHECKPOINT = FROZEN_DIR / "checkpoints/F1_frozen_S0_Z_attention/best_trainable_downstream.pt"
OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
UCLA_MEAN_MM_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_mean_mm_plain.npz")
EXISTING_OBS_PRED_CSV = PROJECT_ROOT / "artifacts/s0_attention_observational_transfer_v1/observational_predictions.csv"
SIM_SWE_LABELS_CSV = PROJECT_ROOT / "artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels.csv"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/swe_sim_vs_ucla_distribution_v1"

IN_CHANNELS_PER_MONTH = 38  # 19 physical + 19 mask, matching training exactly.
MODEL_ORDER = ["S0", "F0_frozen_S0_Z_MLP", "F1_frozen_S0_Z_attention", "S1_end_to_end_attention"]
MODEL_COLOR = {"S0": "tab:blue", "F0_frozen_S0_Z_MLP": "tab:purple", "F1_frozen_S0_Z_attention": "tab:orange", "S1_end_to_end_attention": "tab:green"}


# ---------------------------------------------------------------------------
# Model classes copied by reference from snow_ml.cmip6_cnn_experiment,
# scripts/run_frozen_s0_z_attention_test.py, and
# scripts/run_s0_attention_observational_transfer.py. Not redefined/altered.
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


def tensor_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_s0_module() -> M1StaticCNN:
    checkpoint = torch.load(S0_EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model = M1StaticCNN(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, checkpoint["epoch"]


def load_frozen_head(checkpoint_path: Path, head_cls, s0_module: M1StaticCNN):
    encoder = FrozenS0Encoder(s0_module)
    model = head_cls(encoder)
    trainable_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(trainable_checkpoint["trainable_state_dict"], strict=False)
    unexplained_missing = [k for k in result.missing_keys if not k.startswith("encoder.")]
    assert not unexplained_missing, f"Unexplained missing keys for {checkpoint_path}: {unexplained_missing}"
    assert not result.unexpected_keys, f"Unexpected keys for {checkpoint_path}: {result.unexpected_keys}"
    expected_encoder_keys = {f"encoder.{k}" for k in model.encoder.state_dict().keys()}
    encoder_missing = {k for k in result.missing_keys if k.startswith("encoder.")}
    assert encoder_missing == expected_encoder_keys
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, trainable_checkpoint["epoch"]


def stats_block(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n_samples": int(values.size),
        "mean_mm": float(values.mean()),
        "std_mm": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "min_mm": float(values.min()),
        "max_mm": float(values.max()),
        "median_mm": float(p50),
        "p5_mm": float(p5),
        "p25_mm": float(p25),
        "p75_mm": float(p75),
        "p95_mm": float(p95),
        "iqr_mm": float(p75 - p25),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"output_root={OUTPUT_ROOT}", flush=True)

    # -----------------------------------------------------------------
    # 1. Manifest + split — reuse exactly, no new split created.
    # -----------------------------------------------------------------
    manifest = read_manifest(CACHE_DIR / "manifest.csv")
    n_total = len(manifest)
    print(f"manifest rows: {n_total} ({CACHE_DIR / 'manifest.csv'})", flush=True)

    split_s0 = json.loads((S0_EXP_DIR / "split.json").read_text())
    split_s1 = json.loads((S1_EXP_DIR / "split.json").read_text())
    split_head_screen = json.loads(HEAD_SCREEN_SPLIT.read_text())
    assert sorted(split_s0["train"]) == sorted(split_s1["train"]) == sorted(split_head_screen["train"])
    assert sorted(split_s0["val"]) == sorted(split_s1["val"]) == sorted(split_head_screen["val"])
    train_idx = sorted(split_s0["train"])
    val_idx = sorted(split_s0["val"])
    assert len(set(train_idx)) == len(train_idx) and len(set(val_idx)) == len(val_idx), "duplicated sample indices"
    assert set(train_idx).isdisjoint(val_idx), "train/val overlap"
    assert set(train_idx) | set(val_idx) == set(range(n_total)), "split does not cover the full manifest"
    print(f"split verified identical across S0/{S1_EXP_DIR.name}/head_screen: train={len(train_idx)} val={len(val_idx)}", flush=True)

    swe_label_mm = np.array([float(row["SWE_label"]) for row in manifest], dtype=np.float64)
    sim_train_swe_mm = swe_label_mm[train_idx]
    sim_val_swe_mm = swe_label_mm[val_idx]

    # Cross-check against the canonical WUS-D3 Sierra-region label table (same
    # DEFAULT_SIERRA_REGION + build_sierra_mask + area-weighted-mean recipe
    # used for the UCLA target below, applied natively to the WUS-D3 grid).
    with SIM_SWE_LABELS_CSV.open() as fh:
        sim_labels_rows = list(csv.DictReader(fh))
    join_key = lambda row: (row["model_member"], row["experiment"], row["row_year"])  # noqa: E731
    sim_labels_by_key = {join_key(row): float(row["SWE_label"]) for row in sim_labels_rows}
    mismatches = 0
    for row in manifest:
        key = join_key(row)
        if key not in sim_labels_by_key:
            mismatches += 1
            continue
        if abs(sim_labels_by_key[key] - float(row["SWE_label"])) > 1.0e-6:
            mismatches += 1
    assert mismatches == 0, f"{mismatches} manifest rows disagree with {SIM_SWE_LABELS_CSV}"
    print(f"simulation SWE cross-check against {SIM_SWE_LABELS_CSV}: {n_total - mismatches}/{n_total} rows match exactly", flush=True)

    # -----------------------------------------------------------------
    # 2. UCLA observational SWE, WY1985-2021 (37 values), same region/mask.
    # -----------------------------------------------------------------
    ucla = np.load(UCLA_MEAN_MM_NPZ)
    ucla_years = [int(y) for y in ucla["water_year"]]
    ucla_mm = np.asarray(ucla["sierra_swe_apr1_mean_mm"], dtype=np.float64)
    assert ucla_years == list(range(1985, 2022)), "UCLA target water years not ascending 1985..2021"
    assert len(ucla_years) == 37
    assert np.isfinite(ucla_mm).all()
    print(f"UCLA observational SWE: {len(ucla_years)} water years, {UCLA_MEAN_MM_NPZ}", flush=True)

    # -----------------------------------------------------------------
    # 3. Distribution statistics + CSV outputs.
    # -----------------------------------------------------------------
    groups = {
        "simulation_train": sim_train_swe_mm,
        "simulation_validation": sim_val_swe_mm,
        "ucla_observation_wy1985_2021": ucla_mm,
    }
    summary_rows = []
    for name, values in groups.items():
        row = {"group": name, **stats_block(values)}
        summary_rows.append(row)
    sigma_obs = next(r["std_mm"] for r in summary_rows if r["group"] == "ucla_observation_wy1985_2021")
    sigma_train = next(r["std_mm"] for r in summary_rows if r["group"] == "simulation_train")
    sigma_val = next(r["std_mm"] for r in summary_rows if r["group"] == "simulation_validation")
    variance_ratios = {
        "sigma_obs_over_sigma_sim_train": sigma_obs / sigma_train,
        "sigma_obs_over_sigma_sim_val": sigma_obs / sigma_val,
    }
    print(json.dumps({"summary": summary_rows, "variance_ratios": variance_ratios}, indent=2), flush=True)

    with (OUTPUT_ROOT / "swe_distribution_summary.csv").open("w", newline="") as fh:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (OUTPUT_ROOT / "ucla_observed_swe_wy1985_2021.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["water_year", "sierra_swe_apr1_mean_mm"])
        for wy, val in zip(ucla_years, ucla_mm, strict=True):
            writer.writerow([wy, f"{val:.6f}"])

    with (OUTPUT_ROOT / "simulation_train_swe_ucla_region.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "model_member", "experiment", "row_year", "water_year", "swe_label_mm"])
        for i in train_idx:
            row = manifest[i]
            writer.writerow([i, row["model_member"], row["experiment"], row["row_year"], row["water_year"], f"{float(row['SWE_label']):.6f}"])

    with (OUTPUT_ROOT / "simulation_validation_swe_ucla_region.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "model_member", "experiment", "row_year", "water_year", "swe_label_mm"])
        for i in val_idx:
            row = manifest[i]
            writer.writerow([i, row["model_member"], row["experiment"], row["row_year"], row["water_year"], f"{float(row['SWE_label']):.6f}"])

    # -----------------------------------------------------------------
    # 4. Model predictions. Reuse saved predictions wherever they exist;
    #    run frozen inference only where nothing was saved (F0, F1 on
    #    simulation validation; F0 on the 37 observational years).
    # -----------------------------------------------------------------
    s0_module, s0_epoch = load_s0_module()
    f0_module, f0_epoch = load_frozen_head(F0_CHECKPOINT, FrozenS0ZMLP, s0_module)
    f1_module, f1_epoch = load_frozen_head(F1_CHECKPOINT, FrozenS0ZAttention, s0_module)

    s0_stats = np.load(S0_EXP_DIR / "normalization_stats.npz")
    s0_swe_mu, s0_swe_sigma = float(s0_stats["target_mu"][0]), float(s0_stats["target_sigma"][0])
    feature_mu, feature_sigma = s0_stats["feature_mu"], s0_stats["feature_sigma"]

    physical = np.load(CACHE_DIR / "inputs_physical.npy", mmap_mode="r")
    valid_mask = np.load(CACHE_DIR / "inputs_valid_mask.npy", mmap_mode="r")
    targets_arr = np.load(CACHE_DIR / "targets.npy")

    val_dataset = CMIP6TensorDataset(
        physical_data=physical,
        valid_mask=valid_mask,
        targets=targets_arr,
        metadata=manifest,
        indices=val_idx,
        feature_mu=feature_mu,
        feature_sigma=feature_sigma,
        target_mu=s0_stats["target_mu"],
        target_sigma=s0_stats["target_sigma"],
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_idx), shuffle=False, collate_fn=collate_with_metadata)
    val_inputs, val_targets_z, val_meta = next(iter(val_loader))

    digest_before = {"F0": tensor_digest(f0_module), "F1": tensor_digest(f1_module), "S0_encoder": tensor_digest(s0_module.backbone) + tensor_digest(s0_module.project)}

    with torch.no_grad():
        f0_val_z_1 = f0_module(val_inputs).numpy()
        f1_val_z_1 = f1_module(val_inputs).numpy()
        f0_val_z_2 = f0_module(val_inputs).numpy()
        f1_val_z_2 = f1_module(val_inputs).numpy()
    assert np.allclose(f0_val_z_1, f0_val_z_2, atol=1e-6) and np.allclose(f1_val_z_1, f1_val_z_2, atol=1e-6), "non-deterministic eval-mode output"

    digest_after = {"F0": tensor_digest(f0_module), "F1": tensor_digest(f1_module), "S0_encoder": tensor_digest(s0_module.backbone) + tensor_digest(s0_module.project)}
    assert digest_before == digest_after, "model parameters changed during inference"

    val_true_mm = val_targets_z[:, 0].numpy() * s0_swe_sigma + s0_swe_mu
    f0_val_pred_mm = f0_val_z_1 * s0_swe_sigma + s0_swe_mu
    f1_val_pred_mm = f1_val_z_1 * s0_swe_sigma + s0_swe_mu
    val_water_years = np.array([int(m["water_year"]) for m in val_meta])
    val_sample_indices = np.array([int(m["sample_index"]) for m in val_meta])
    assert np.allclose(val_true_mm, swe_label_mm[val_sample_indices], atol=1e-2), "val true SWE mismatch vs manifest SWE_label"

    # S0, S1 validation predictions were saved at training time - reuse verbatim.
    def load_saved_val_predictions(exp_dir: Path) -> dict[int, dict[str, float]]:
        rows = json.loads((exp_dir / "best_validation_predictions.json").read_text())
        stats = np.load(exp_dir / "normalization_stats.npz")
        mu, sigma = float(stats["target_mu"][0]), float(stats["target_sigma"][0])
        out = {}
        for row in rows:
            meta = row["metadata"]
            wy = int(meta["water_year"])
            out[wy] = {
                "true_mm": float(meta["SWE_label"]),
                "pred_mm": row["swe_hat_z"] * sigma + mu,
            }
        return out

    s0_val_saved = load_saved_val_predictions(S0_EXP_DIR)
    s1_val_saved = load_saved_val_predictions(S1_EXP_DIR)
    assert set(s0_val_saved) == set(int(wy) for wy in val_water_years)
    assert set(s1_val_saved) == set(int(wy) for wy in val_water_years)

    val_predictions_by_model = {
        "S0": {int(wy): {"true_mm": s0_val_saved[int(wy)]["true_mm"], "pred_mm": s0_val_saved[int(wy)]["pred_mm"]} for wy in val_water_years},
        "F0_frozen_S0_Z_MLP": {int(wy): {"true_mm": float(t), "pred_mm": float(p)} for wy, t, p in zip(val_water_years, val_true_mm, f0_val_pred_mm, strict=True)},
        "F1_frozen_S0_Z_attention": {int(wy): {"true_mm": float(t), "pred_mm": float(p)} for wy, t, p in zip(val_water_years, val_true_mm, f1_val_pred_mm, strict=True)},
        "S1_end_to_end_attention": {int(wy): {"true_mm": s1_val_saved[int(wy)]["true_mm"], "pred_mm": s1_val_saved[int(wy)]["pred_mm"]} for wy in val_water_years},
    }

    with (OUTPUT_ROOT / "simulation_validation_model_predictions.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "water_year", "swe_true_mm"] + [f"{m}_pred_mm" for m in MODEL_ORDER])
        for idx, wy in zip(val_sample_indices, val_water_years, strict=True):
            true_val = val_predictions_by_model["S0"][int(wy)]["true_mm"]
            row = [int(idx), int(wy), f"{true_val:.6f}"]
            for m in MODEL_ORDER:
                row.append(f"{val_predictions_by_model[m][int(wy)]['pred_mm']:.6f}")
            writer.writerow(row)

    # Observational (37 UCLA years). S0/F1/S1 predictions already saved by the
    # existing observational-transfer run; only F0 needs a fresh forward pass.
    with EXISTING_OBS_PRED_CSV.open() as fh:
        existing_obs_rows = list(csv.DictReader(fh))
    assert [int(r["water_year"]) for r in existing_obs_rows] == ucla_years
    obs_pred_by_model = {
        "S0": np.array([float(r["s0_prediction_mm"]) for r in existing_obs_rows]),
        "F1_frozen_S0_Z_attention": np.array([float(r["frozen_s0_attention_prediction_mm"]) for r in existing_obs_rows]),
        "S1_end_to_end_attention": np.array([float(r["end_to_end_attention_prediction_mm"]) for r in existing_obs_rows]),
    }
    observed_from_saved_run = np.array([float(r["observed_swe_mm"]) for r in existing_obs_rows])
    assert np.allclose(observed_from_saved_run, ucla_mm, atol=1e-4), "existing obs-transfer observed vector disagrees with UCLA npz"

    physical_obs = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    mask_obs = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    assert physical_obs.shape == (37, 7, 19, 120, 240)
    x_obs = physical_obs.astype(np.float32)
    x_obs = (x_obs - feature_mu) / feature_sigma
    x_obs = np.clip(x_obs, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
    x_obs = np.where(mask_obs > 0.5, x_obs, np.nan).astype(np.float32)
    x_obs = np.nan_to_num(x_obs, nan=0.0, posinf=0.0, neginf=0.0)
    stacked_obs = np.concatenate([x_obs, mask_obs.astype(np.float32)], axis=2)
    obs_inputs = torch.from_numpy(stacked_obs)

    with torch.no_grad():
        f0_obs_z_1 = f0_module(obs_inputs).numpy()
        f0_obs_z_2 = f0_module(obs_inputs).numpy()
    assert np.allclose(f0_obs_z_1, f0_obs_z_2, atol=1e-6)
    digest_after_obs = tensor_digest(f0_module)
    assert digest_after_obs == digest_after["F0"], "F0 parameters changed during observational inference"
    obs_pred_by_model["F0_frozen_S0_Z_MLP"] = f0_obs_z_1 * s0_swe_sigma + s0_swe_mu

    with (OUTPUT_ROOT / "observational_model_predictions.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["water_year", "swe_observed_mm"] + [f"{m}_pred_mm" for m in MODEL_ORDER])
        for i, wy in enumerate(ucla_years):
            row = [wy, f"{ucla_mm[i]:.6f}"]
            for m in MODEL_ORDER:
                row.append(f"{obs_pred_by_model[m][i]:.6f}")
            writer.writerow(row)

    # -----------------------------------------------------------------
    # 5. Plots.
    # -----------------------------------------------------------------
    x_min = min(sim_train_swe_mm.min(), sim_val_swe_mm.min(), ucla_mm.min())
    x_max = max(sim_train_swe_mm.max(), sim_val_swe_mm.max(), ucla_mm.max())
    bins = np.linspace(x_min, x_max, 30)

    # Plot A: overlaid histograms.
    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    axis.hist(sim_train_swe_mm, bins=bins, alpha=0.45, density=True, label=f"Simulation train (n={sim_train_swe_mm.size})", color="tab:blue")
    axis.hist(sim_val_swe_mm, bins=bins, alpha=0.45, density=True, label=f"Simulation validation (n={sim_val_swe_mm.size})", color="tab:orange")
    axis.hist(ucla_mm, bins=bins, alpha=0.45, density=True, label=f"UCLA observation (n={ucla_mm.size})", color="black")
    axis.set_xlabel("April 1 Sierra SWE (mm), UCLA region")
    axis.set_ylabel("Density")
    axis.set_title("SWE target distribution: simulation vs. UCLA observation")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "plot_A_distribution_comparison_histogram.png")
    plt.close(fig)

    # Plot B: box/violin comparison.
    fig, axis = plt.subplots(figsize=(7, 6), dpi=160)
    data = [sim_train_swe_mm, sim_val_swe_mm, ucla_mm]
    labels = ["Sim train", "Sim val", "UCLA obs"]
    parts = axis.violinplot(data, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], ("tab:blue", "tab:orange", "black"), strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.25)
    axis.boxplot(data, labels=labels, showfliers=True, widths=0.25)
    axis.set_ylabel("April 1 Sierra SWE (mm), UCLA region")
    axis.set_title("SWE target distribution comparison")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "plot_B_distribution_comparison_boxplot.png")
    plt.close(fig)

    # Plot C: UCLA 37-year time series.
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    axis.plot(ucla_years, ucla_mm, marker="o", color="black")
    axis.set_xlabel("Water year")
    axis.set_ylabel("April 1 Sierra SWE (mm)")
    axis.set_title("UCLA observed Sierra SWE, WY1985-2021")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "plot_C_ucla_timeseries.png")
    plt.close(fig)

    # Plot D: one figure per model, two panels (sim val | obs), shared y-axis.
    all_val_true = np.array([val_predictions_by_model["S0"][int(wy)]["true_mm"] for wy in val_water_years])
    for model_name in MODEL_ORDER:
        val_pred = np.array([val_predictions_by_model[model_name][int(wy)]["pred_mm"] for wy in val_water_years])
        obs_pred = obs_pred_by_model[model_name]
        y_min = min(all_val_true.min(), val_pred.min(), ucla_mm.min(), obs_pred.min())
        y_max = max(all_val_true.max(), val_pred.max(), ucla_mm.max(), obs_pred.max())
        pad = 0.05 * (y_max - y_min)

        fig, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
        order = np.argsort(val_water_years)
        axis_left.plot(np.arange(len(val_water_years)), all_val_true[order], "o-", color="black", markersize=4, label="True (simulation val)")
        axis_left.plot(np.arange(len(val_water_years)), val_pred[order], "o-", color=MODEL_COLOR[model_name], markersize=4, label=f"{model_name} prediction")
        axis_left.set_xlabel("Simulation validation sample (sorted by water year)")
        axis_left.set_ylabel("SWE (mm)")
        axis_left.set_title(f"Simulation validation (n={len(val_water_years)})")
        axis_left.set_ylim(y_min - pad, y_max + pad)
        axis_left.legend(fontsize=8)
        axis_left.grid(alpha=0.25)

        axis_right.plot(ucla_years, ucla_mm, "o-", color="black", markersize=4, label="Observed (UCLA)")
        axis_right.plot(ucla_years, obs_pred, "o-", color=MODEL_COLOR[model_name], markersize=4, label=f"{model_name} prediction")
        axis_right.set_xlabel("Water year")
        axis_right.set_title("UCLA observational, WY1985-2021 (n=37)")
        axis_right.set_ylim(y_min - pad, y_max + pad)
        axis_right.legend(fontsize=8)
        axis_right.grid(alpha=0.25)

        fig.suptitle(f"{model_name}: simulation-validation vs. observational SWE (same y-axis, mm)")
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / f"plot_D_{model_name}_sim_vs_obs.png")
        plt.close(fig)

    # -----------------------------------------------------------------
    # 6. Final report / README inputs.
    # -----------------------------------------------------------------
    report = {
        "ucla_region_definition": {"lat_min": 35.0, "lat_max": 42.0, "lon_min": -122.5, "lon_max": -118.0, "source": "snow_ml.data.DEFAULT_SIERRA_REGION"},
        "ucla_mask_source": "snow_ml.data.build_sierra_mask (boolean lat/lon-box mask, coarsen_factor=1) via scripts/process_cobe2_sierra_swe_apr1_target.py",
        "simulation_mask_source": "snow_ml.data.build_sierra_mask (same function, same DEFAULT_SIERRA_REGION, coarsen_factor=1) applied natively to the WUS-D3 d02 grid via scripts/build_wusd3_swe_labels.py / build_mask_and_area",
        "area_weighting": "A_i = R^2 |sin(lat_north)-sin(lat_south)| * |lon_east-lon_west|, R=6,371,000 m, applied identically in both scripts",
        "known_footprint_area_difference": "artifacts/cnn_swe_target_spatial_audit/cnn_wusd3_vs_ucla_ace_target_summary.json: WUS-D3 selected area is 61.7% of the UCLA selected area within the identical lat/lon box (188,244 km^2 vs 304,961 km^2), because the two products' native grids resolve the mountain/coast footprint differently; no cell-index matching was used, per the same-geographic-footprint requirement.",
        "simulation_swe_source": str(SIM_SWE_LABELS_CSV),
        "simulation_predictor_split_source": str(S0_EXP_DIR / "split.json"),
        "ucla_swe_source": str(UCLA_MEAN_MM_NPZ),
        "n_total_samples": n_total,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "summary_stats": summary_rows,
        "variance_ratios": variance_ratios,
        "checkpoints": {
            "S0": {"path": str(S0_EXP_DIR / "best_checkpoint.pt"), "epoch": s0_epoch},
            "F0_frozen_S0_Z_MLP": {"path": str(F0_CHECKPOINT), "epoch": f0_epoch, "note": "F0 val/obs predictions computed here by frozen inference (not previously saved)"},
            "F1_frozen_S0_Z_attention": {"path": str(F1_CHECKPOINT), "epoch": f1_epoch, "note": "F1 val predictions computed here; F1 obs predictions reused from artifacts/s0_attention_observational_transfer_v1"},
            "S1_end_to_end_attention": {"path": str(S1_EXP_DIR / "best_checkpoint.pt"), "note": "val/obs predictions reused from saved artifacts"},
        },
    }
    (OUTPUT_ROOT / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
