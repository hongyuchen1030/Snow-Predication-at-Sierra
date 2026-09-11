#!/usr/bin/env python3
"""Observational transfer for the two standardized-anomaly-target CNN models (S0 and
end-to-end S1 attention), plus final reporting. No training, no fine-tuning here -
inference only, on the already-trained checkpoints from
run_cmip6_cnn_experiment.py --predictor-cache-dir cmip6_cnn_swe_std_anomaly_v1/data_cache.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from snow_ml.cmip6_cnn_experiment import (  # noqa: E402
    FEATURE_Z_CLIP,
    M1StaticCNN,
    StaticLatentSelfAttentionCNN,
)

OBS_CACHE_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_obs_transfer_v1/data_cache_observational")
S0_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_std_anomaly_v1/experiments/S0_static_cnn_swe_only")
S1_EXP_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_std_anomaly_v1/experiments/S1_static_latent_self_attention")
UCLA_TARGET_NPZ = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_mean_mm_plain.npz")

OUTPUT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/s0_attention_swe_std_anomaly_v1")
IN_CHANNELS_PER_MONTH = 38


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = y_true - y_true.mean(), y_pred - y_pred.mean()
    return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))


def load_model(cls, exp_dir: Path, device: torch.device):
    checkpoint = torch.load(exp_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model = cls(IN_CHANNELS_PER_MONTH, include_auxiliary_heads=False) if cls is M1StaticCNN else cls(IN_CHANNELS_PER_MONTH)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, checkpoint["epoch"]


def snapshot_params(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # -----------------------------------------------------------------
    # Observational predictor tensor.
    # -----------------------------------------------------------------
    physical = np.load(OBS_CACHE_DIR / "inputs_physical.npy")
    valid_mask = np.load(OBS_CACHE_DIR / "inputs_valid_mask.npy")
    with (OBS_CACHE_DIR / "manifest.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    water_years = [int(r["water_year"]) for r in manifest]
    assert water_years == list(range(1985, 2022)), "observational water_year not ascending 1985..2021"
    assert physical.shape == (37, 7, 19, 120, 240)

    # -----------------------------------------------------------------
    # UCLA observational target: y_obs = (SWE_obs - mu_obs) / sigma_obs.
    # -----------------------------------------------------------------
    ucla = np.load(UCLA_TARGET_NPZ)
    ucla_wy = [int(y) for y in ucla["water_year"]]
    ucla_mm_by_year = {wy: float(v) for wy, v in zip(ucla_wy, ucla["sierra_swe_apr1_mean_mm"], strict=True)}
    assert set(ucla_mm_by_year.keys()) == set(water_years)
    observed_mm = np.asarray([ucla_mm_by_year[wy] for wy in water_years], dtype=np.float64)
    mu_obs, sigma_obs = float(observed_mm.mean()), float(observed_mm.std())
    observed_anomaly = (observed_mm - mu_obs) / sigma_obs
    assert np.isfinite(observed_anomaly).all()
    print(f"UCLA: mu_obs={mu_obs:.4f}mm sigma_obs={sigma_obs:.4f}mm", flush=True)

    # -----------------------------------------------------------------
    # Preprocess observational predictors using EACH model's own saved feature stats
    # (identical between S0/S1 since physical predictors are unchanged; using each
    # model's own file for exactness).
    # -----------------------------------------------------------------
    def preprocess(feature_mu, feature_sigma):
        x = physical.astype(np.float32)
        x = (x - feature_mu) / feature_sigma
        x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
        x = np.where(valid_mask > 0.5, x, np.nan).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        stacked = np.concatenate([x, valid_mask.astype(np.float32)], axis=2)
        assert np.isfinite(stacked).all()
        return torch.from_numpy(stacked)

    results = {}
    predictions_anomaly = {}
    for name, cls, exp_dir in (("S0", M1StaticCNN, S0_EXP_DIR), ("attention", StaticLatentSelfAttentionCNN, S1_EXP_DIR)):
        print(f"=== {name} ===", flush=True)
        stats = np.load(exp_dir / "normalization_stats.npz")
        feature_mu, feature_sigma = stats["feature_mu"], stats["feature_sigma"]
        target_mu, target_sigma = float(stats["target_mu"][0]), float(stats["target_sigma"][0])
        inputs = preprocess(feature_mu, feature_sigma)

        model, epoch = load_model(cls, exp_dir, device)
        params_before = snapshot_params(model)

        def run_forward():
            with torch.no_grad():
                out = model(inputs.to(device))
                out = out["swe_hat"] if isinstance(out, dict) else out
                return out.detach().cpu().numpy()

        pred1, pred2 = run_forward(), run_forward()
        assert np.allclose(pred1, pred2, atol=1e-6), f"{name}: non-deterministic"
        assert np.isfinite(pred1).all()
        params_after = snapshot_params(model)
        max_change = max(float((params_before[k] - params_after[k]).abs().max()) for k in params_before)
        assert max_change == 0.0, f"{name}: parameter drift {max_change}"

        pred_anomaly = pred1 * target_sigma + target_mu  # model's internal z -> standardized-anomaly units
        predictions_anomaly[name] = pred_anomaly

        r2 = r2_score(observed_anomaly, pred_anomaly)
        r = pearson_r(observed_anomaly, pred_anomaly)
        rmse = float(np.sqrt(np.mean((observed_anomaly - pred_anomaly) ** 2)))
        mae = float(np.mean(np.abs(observed_anomaly - pred_anomaly)))
        results[name] = {
            "checkpoint": str(exp_dir / "best_checkpoint.pt"),
            "checkpoint_epoch": int(epoch),
            "observational_r2": r2,
            "observational_pearson_r": r,
            "observational_rmse_anomaly": rmse,
            "observational_mae_anomaly": mae,
            "max_parameter_change": max_change,
            "deterministic": True,
        }
        print(json.dumps(results[name], indent=2), flush=True)

    # -----------------------------------------------------------------
    # Simulation validation metrics (already saved by training).
    # -----------------------------------------------------------------
    s0_metrics = json.loads((S0_EXP_DIR / "metrics_summary.json").read_text())
    s1_metrics = json.loads((S1_EXP_DIR / "metrics_summary.json").read_text())
    sim_val_r2 = {"S0": s0_metrics["best_metrics"]["swe_r2"], "attention": s1_metrics["best_metrics"]["swe_r2"]}
    sim_best_epoch = {"S0": s0_metrics["best_epoch"], "attention": s1_metrics["best_epoch"]}

    # -----------------------------------------------------------------
    # Outputs.
    # -----------------------------------------------------------------
    pred_csv = OUTPUT_ROOT / "observational_predictions.csv"
    with pred_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["water_year", "UCLA_SWE_mm", "UCLA_standardized_anomaly", "S0_predicted_anomaly", "attention_predicted_anomaly"])
        for i, wy in enumerate(water_years):
            writer.writerow([wy, f"{observed_mm[i]:.6f}", f"{observed_anomaly[i]:.6f}", f"{predictions_anomaly['S0'][i]:.6f}", f"{predictions_anomaly['attention'][i]:.6f}"])
    print(f"wrote {pred_csv}", flush=True)

    final_csv = OUTPUT_ROOT / "final_metrics.csv"
    with final_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "checkpoint", "checkpoint_epoch", "simulation_val_r2", "observational_r2", "observational_pearson_r", "observational_rmse_anomaly", "observational_mae_anomaly"])
        for name in ("S0", "attention"):
            row = results[name]
            writer.writerow([name, row["checkpoint"], row["checkpoint_epoch"], sim_val_r2[name], row["observational_r2"], row["observational_pearson_r"], row["observational_rmse_anomaly"], row["observational_mae_anomaly"]])
    print(f"wrote {final_csv}", flush=True)

    # Plot 1 & 2: train/val R2 vs epoch, per model.
    for name, exp_dir, filename in (("S0", S0_EXP_DIR, "r2_vs_epoch_S0.png"), ("attention", S1_EXP_DIR, "r2_vs_epoch_attention.png")):
        with (exp_dir / "history.csv").open() as fh:
            history = list(csv.DictReader(fh))
        epochs = [int(r["epoch"]) for r in history]
        train_r2 = [float(r["train_swe_r2"]) for r in history]
        val_r2 = [float(r["swe_r2"]) for r in history]
        fig, axis = plt.subplots(figsize=(8, 5), dpi=160)
        axis.plot(epochs, train_r2, marker="o", markersize=3, label="Train (eval mode)")
        axis.plot(epochs, val_r2, marker="o", markersize=3, label="Validation")
        axis.axvline(sim_best_epoch[name], color="tab:red", linestyle="--", label=f"Best epoch ({sim_best_epoch[name]})")
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("SWE R2 (standardized-anomaly target)")
        axis.set_title(f"{name}: train/val R2 vs epoch")
        axis.legend(fontsize=9)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / filename)
        plt.close(fig)
        print(f"wrote {OUTPUT_ROOT / filename}", flush=True)

    # Plot 3: observed vs predicted anomaly time series.
    fig, axis = plt.subplots(figsize=(12, 5.5), dpi=160)
    axis.plot(water_years, observed_anomaly, marker="o", color="black", linewidth=2.0, label="UCLA observed (standardized anomaly)")
    axis.plot(water_years, predictions_anomaly["S0"], marker="o", markersize=3, color="tab:blue", label="S0")
    axis.plot(water_years, predictions_anomaly["attention"], marker="o", markersize=3, color="tab:orange", label="End-to-end attention")
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Water year")
    axis.set_ylabel("Standardized SWE anomaly")
    axis.set_title("UCLA observed vs. predicted standardized April-1 Sierra SWE anomaly, WY1985-2021")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "observed_vs_predicted_anomaly_timeseries.png")
    plt.close(fig)

    # Plot 4: scatter, per model.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=160, sharex=True, sharey=True)
    for axis, name, color in zip(axes, ("S0", "attention"), ("tab:blue", "tab:orange"), strict=True):
        axis.scatter(observed_anomaly, predictions_anomaly[name], color=color, s=24)
        lims = [-2.5, 3.0]
        axis.plot(lims, lims, "k--", linewidth=0.8)
        axis.set_xlim(lims); axis.set_ylim(lims)
        axis.set_xlabel("Observed standardized anomaly")
        axis.set_title(f"{name}\nR2={results[name]['observational_r2']:.3f}, r={results[name]['observational_pearson_r']:.3f}")
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted standardized anomaly")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "observed_vs_predicted_anomaly_scatter.png")
    plt.close(fig)

    # Plot 5: compact sim-vs-obs R2 comparison.
    fig, axis = plt.subplots(figsize=(6.5, 5), dpi=160)
    names = ["S0", "attention"]
    x_pos = np.arange(len(names))
    width = 0.35
    axis.bar(x_pos - width / 2, [sim_val_r2[n] for n in names], width, label="Simulation validation R2", color="tab:gray")
    axis.bar(x_pos + width / 2, [results[n]["observational_r2"] for n in names], width, label="Observational R2", color="tab:red")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_pos); axis.set_xticklabels(names)
    axis.set_ylabel("R2 (standardized-anomaly target)")
    axis.set_title("Simulation vs. observational R2")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "simulation_vs_observational_r2.png")
    plt.close(fig)

    full_report = {
        "s0_architecture_source": "src/snow_ml/cmip6_cnn_experiment.py:M1StaticCNN, trained via scripts/run_cmip6_cnn_experiment.py --architecture S0_static_cnn_swe_only",
        "attention_architecture_source": "src/snow_ml/cmip6_cnn_experiment.py:StaticLatentSelfAttentionCNN, trained via scripts/run_cmip6_cnn_experiment.py --architecture S1_static_latent_self_attention",
        "s0_checkpoint": str(S0_EXP_DIR / "best_checkpoint.pt"),
        "attention_checkpoint": str(S1_EXP_DIR / "best_checkpoint.pt"),
        "s0_best_epoch": sim_best_epoch["S0"],
        "attention_best_epoch": sim_best_epoch["attention"],
        "s0_sim_val_r2": sim_val_r2["S0"],
        "attention_sim_val_r2": sim_val_r2["attention"],
        "mu_obs_mm": mu_obs,
        "sigma_obs_mm": sigma_obs,
        "results": results,
    }
    (OUTPUT_ROOT / "full_report.json").write_text(json.dumps(full_report, indent=2) + "\n")
    print(json.dumps(full_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
