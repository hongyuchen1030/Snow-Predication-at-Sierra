from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from snow_ml.cmip6_cnn_experiment import (
    CMIP6TensorDataset,
    ExperimentConfig,
    build_model,
    collate_with_metadata,
    compute_feature_stats,
    compute_losses,
    compute_target_stats,
    load_json,
    load_or_build_split,
    read_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_16var_baseline_v1")
CACHE_DIR = RUN_ROOT / "data_cache"
EXPERIMENT_DIR = RUN_ROOT / "experiments" / "S0_16var_static_cnn_swe_only"
SMOKE_DIR = EXPERIMENT_DIR / "smoke_test"
DIAGNOSTIC_DIR = PROJECT_ROOT / "artifacts" / "cmip6_cnn_16var_baseline_v1_diagnostics"
BASELINE_EXPERIMENT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments/S0_static_cnn_swe_only"
)
BASELINE_CACHE_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache"
)
BASELINE_CONFIG_PATH = BASELINE_EXPERIMENT_DIR / "config.json"
BASELINE_SPLIT_PATH = BASELINE_EXPERIMENT_DIR / "split.json"
BASELINE_MANIFEST_PATH = BASELINE_CACHE_DIR / "manifest.csv"
BASELINE_METRICS = {
    "model": "S0-19",
    "physical_predictors": 19,
    "cnn_input_channels": 266,
    "best_epoch": 27,
    "train_r2": 0.426,
    "val_r2": 0.375,
    "val_rmse": 27.07,
    "val_mae": 19.15,
    "val_r": 0.623,
}


def log(message: str) -> None:
    print(message, flush=True)


def run_command(args: list[str]) -> None:
    log(f"RUN {' '.join(args)}")
    subprocess.run(args, check=True)


def load_baseline_config() -> dict[str, object]:
    return load_json(BASELINE_CONFIG_PATH)


def prepare_cache(skip_if_present: bool) -> None:
    required = [
        CACHE_DIR / "manifest.csv",
        CACHE_DIR / "inputs_physical.npy",
        CACHE_DIR / "inputs_valid_mask.npy",
        CACHE_DIR / "targets.npy",
        CACHE_DIR / "predictor_inventory.json",
    ]
    if skip_if_present and all(path.exists() for path in required):
        log(f"Using existing prepared cache at {CACHE_DIR}")
        return
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_cmip6_cnn_experiment.py"),
            "--predictor-set",
            "obs_compatible_16",
            "--output-dir",
            str(CACHE_DIR),
        ]
    )


def build_datasets(config: dict[str, object]) -> tuple[CMIP6TensorDataset, CMIP6TensorDataset, dict[str, list[int]], np.ndarray, np.ndarray]:
    manifest_rows = read_manifest(CACHE_DIR / "manifest.csv")
    split, _ = load_or_build_split(
        manifest_rows,
        seed=int(config["seed"]),
        split_json_path=str(BASELINE_SPLIT_PATH),
        split_manifest_path=str(BASELINE_MANIFEST_PATH),
    )
    physical_data = np.load(CACHE_DIR / "inputs_physical.npy", mmap_mode="r")
    valid_mask = np.load(CACHE_DIR / "inputs_valid_mask.npy", mmap_mode="r")
    targets = np.load(CACHE_DIR / "targets.npy")
    feature_mu, feature_sigma, _ = compute_feature_stats(physical_data, split["train"])
    target_mu, target_sigma = compute_target_stats(targets, split["train"])
    train_dataset = CMIP6TensorDataset(
        physical_data=physical_data,
        valid_mask=valid_mask,
        targets=targets,
        metadata=manifest_rows,
        indices=split["train"],
        feature_mu=feature_mu,
        feature_sigma=feature_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )
    val_dataset = CMIP6TensorDataset(
        physical_data=physical_data,
        valid_mask=valid_mask,
        targets=targets,
        metadata=manifest_rows,
        indices=split["val"],
        feature_mu=feature_mu,
        feature_sigma=feature_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )
    return train_dataset, val_dataset, split, target_mu, target_sigma


def run_smoke_test(*, require_gpu: bool) -> None:
    baseline_config = load_baseline_config()
    predictor_inventory = load_json(CACHE_DIR / "predictor_inventory.json")
    log(f"Retained predictors ({predictor_inventory['predictor_count']}): {predictor_inventory['predictor_fields']}")

    train_dataset, _, split, target_mu, target_sigma = build_datasets(baseline_config)
    loader = DataLoader(
        train_dataset,
        batch_size=int(baseline_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_metadata,
    )

    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Smoke test requested with --require-gpu but CUDA is not available in this session.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("S0_static_cnn_swe_only", in_channels_per_month=int(predictor_inventory["stacked_channel_count"]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(baseline_config["learning_rate"]),
        weight_decay=float(baseline_config["weight_decay"]),
    )

    batch_inputs, batch_targets, batch_metadata = next(iter(loader))
    log(f"Smoke batch stacked shape: {tuple(batch_inputs.shape)}")
    flattened_shape = (
        batch_inputs.shape[0],
        batch_inputs.shape[1] * batch_inputs.shape[2],
        batch_inputs.shape[3],
        batch_inputs.shape[4],
    )
    log(f"Smoke batch flattened CNN shape: {flattened_shape}")

    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch_inputs)
    losses = compute_losses(outputs, batch_targets, swe_only=True, kendall_model=None)
    if not torch.isfinite(losses["total_loss"]).all():
        raise RuntimeError("Smoke test produced a non-finite total loss.")
    losses["total_loss"].backward()
    optimizer.step()

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    smoke_history_path = SMOKE_DIR / "history.csv"
    with smoke_history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_size",
                "input_shape",
                "flattened_input_shape",
                "loss",
                "train_rows",
                "val_rows",
                "device",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "batch_size": int(batch_inputs.shape[0]),
                "input_shape": list(batch_inputs.shape),
                "flattened_input_shape": list(flattened_shape),
                "loss": float(losses["total_loss"].detach().cpu().item()),
                "train_rows": len(split["train"]),
                "val_rows": len(split["val"]),
                "device": str(device),
            }
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_metadata": batch_metadata,
            "target_mu": target_mu,
            "target_sigma": target_sigma,
        },
        SMOKE_DIR / "smoke_checkpoint.pt",
    )
    summary = {
        "retained_predictors": predictor_inventory["predictor_fields"],
        "predictor_count": predictor_inventory["predictor_count"],
        "mask_count": predictor_inventory["mask_count"],
        "stacked_batch_shape": list(batch_inputs.shape),
        "flattened_cnn_shape": list(flattened_shape),
        "forward_succeeded": True,
        "loss_finite": True,
        "backward_succeeded": True,
        "optimizer_step_succeeded": True,
        "checkpoint_written": True,
        "history_written": True,
        "device": str(device),
    }
    (SMOKE_DIR / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"Smoke test completed on {device}. Outputs written to {SMOKE_DIR}")


def run_training() -> None:
    baseline_config = load_baseline_config()
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_cmip6_cnn_experiment.py"),
            "--architecture",
            "S0_static_cnn_swe_only",
            "--predictor-cache-dir",
            str(CACHE_DIR),
            "--output-dir",
            str(EXPERIMENT_DIR),
            "--seed",
            str(int(baseline_config["seed"])),
            "--batch-size",
            str(int(baseline_config["batch_size"])),
            "--max-epochs",
            str(int(baseline_config["max_epochs"])),
            "--early-stopping-patience",
            str(int(baseline_config["early_stopping_patience"])),
            "--learning-rate",
            str(float(baseline_config["learning_rate"])),
            "--weight-decay",
            str(float(baseline_config["weight_decay"])),
            "--num-workers",
            str(int(baseline_config["num_workers"])),
            "--split-json",
            str(BASELINE_SPLIT_PATH),
            "--split-manifest",
            str(BASELINE_MANIFEST_PATH),
        ]
        + ([] if bool(baseline_config["amp"]) else ["--disable-amp"])
    )


def make_plots(history_path: Path) -> None:
    history = np.genfromtxt(history_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    epochs = history["epoch"]
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    def save_plot(y1: np.ndarray, y2: np.ndarray | None, title: str, ylabel: str, output_name: str, label1: str, label2: str | None = None) -> None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, y1, label=label1, linewidth=2)
        if y2 is not None and label2 is not None:
            plt.plot(epochs, y2, label=label2, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(DIAGNOSTIC_DIR / output_name, dpi=150)
        plt.close()

    save_plot(history["swe_loss"], history["val_swe_loss"], "S0-16 SWE loss by epoch", "Loss", "s0_16_loss.png", "Train", "Validation")
    save_plot(history["train_swe_r2"], history["swe_r2"], "S0-16 R^2 by epoch", "R^2", "s0_16_r2.png", "Train", "Validation")
    save_plot(history["swe_rmse"], None, "S0-16 validation RMSE by epoch", "RMSE", "s0_16_val_rmse.png", "Validation RMSE")
    save_plot(history["swe_mae"], None, "S0-16 validation MAE by epoch", "MAE", "s0_16_val_mae.png", "Validation MAE")


def write_comparison_report() -> None:
    summary = load_json(EXPERIMENT_DIR / "metrics_summary.json")
    make_plots(EXPERIMENT_DIR / "history.csv")
    best_metrics = summary["best_metrics"]
    best_train_metrics = summary["best_train_metrics"]
    comparison_rows = [
        {
            "Model": "S0-19",
            "Physical predictors": BASELINE_METRICS["physical_predictors"],
            "CNN input channels": BASELINE_METRICS["cnn_input_channels"],
            "Best epoch": BASELINE_METRICS["best_epoch"],
            "Train R2": BASELINE_METRICS["train_r2"],
            "Val R2": BASELINE_METRICS["val_r2"],
            "Val RMSE": BASELINE_METRICS["val_rmse"],
            "Val MAE": BASELINE_METRICS["val_mae"],
            "Val r": BASELINE_METRICS["val_r"],
        },
        {
            "Model": "S0-16",
            "Physical predictors": int(summary["predictor_count"]),
            "CNN input channels": int(summary["cnn_input_channels"]),
            "Best epoch": int(summary["best_epoch"]),
            "Train R2": float(best_train_metrics["swe_r2"]),
            "Val R2": float(best_metrics["swe_r2"]),
            "Val RMSE": float(best_metrics["swe_rmse"]),
            "Val MAE": float(best_metrics["swe_mae"]),
            "Val r": float(best_metrics["swe_pearson_r"]),
        },
    ]
    comparison_csv = DIAGNOSTIC_DIR / "s0_19_vs_s0_16_comparison.csv"
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    deltas = {
        "delta_r2": float(best_metrics["swe_r2"]) - float(BASELINE_METRICS["val_r2"]),
        "delta_rmse": float(best_metrics["swe_rmse"]) - float(BASELINE_METRICS["val_rmse"]),
        "delta_mae": float(best_metrics["swe_mae"]) - float(BASELINE_METRICS["val_mae"]),
        "delta_r": float(best_metrics["swe_pearson_r"]) - float(BASELINE_METRICS["val_r"]),
    }
    payload = {
        "baseline_s0_19": BASELINE_METRICS,
        "s0_16": comparison_rows[1],
        "deltas": deltas,
        "comparison_csv": str(comparison_csv),
        "diagnostic_dir": str(DIAGNOSTIC_DIR),
    }
    (DIAGNOSTIC_DIR / "s0_16_comparison_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare", "smoke", "train", "report", "all"),
        default="prepare",
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    if args.mode in {"prepare", "smoke", "train", "all"}:
        prepare_cache(skip_if_present=not args.force_prepare)
    if args.mode in {"smoke", "all"}:
        run_smoke_test(require_gpu=args.require_gpu)
    if args.mode in {"train", "all"}:
        run_training()
    if args.mode in {"report", "all", "train"} and (EXPERIMENT_DIR / "metrics_summary.json").exists():
        write_comparison_report()


if __name__ == "__main__":
    main()
