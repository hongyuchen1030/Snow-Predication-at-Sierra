from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_SPECS = [
    ("S0_static_cnn_swe_only", "S0", "Flatten + MLP"),
    ("S1_static_latent_self_attention", "S1", "Self-attention + flatten + MLP"),
    ("S2_static_swe_token", "S2", "SWE/CLS token"),
    ("S3_static_residual_gated_attention", "S3", "MLP + gated residual attention"),
]


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def read_summary(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def classify_behavior(label: str, best_row: dict[str, float], history: list[dict[str, float]]) -> str:
    train_r2 = float(best_row["train_swe_r2"])
    val_r2 = float(best_row["swe_r2"])
    gap = train_r2 - val_r2
    best_epoch = int(best_row["epoch"])
    alpha = float(best_row.get("alpha", 0.0))
    if label == "S3" and abs(alpha) < 0.02:
        return "attention branch not contributing"
    if best_epoch <= 5 and val_r2 < 0.1:
        return "early saturation"
    if gap > 0.2 and val_r2 > 0.1:
        return "overfitting"
    if train_r2 < 0.15 and val_r2 < 0.15:
        return "underfitting"
    val_series = [float(row["swe_r2"]) for row in history]
    if len(val_series) >= 6:
        recent = val_series[-6:]
        if max(recent) - min(recent) > 0.2 and val_r2 < 0.2:
            return "unstable optimization"
    return "healthy fit"


def plot_two_series(
    rows: list[dict[str, float]],
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    train_values = [float(row[train_key]) for row in rows]
    val_values = [float(row[val_key]) for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_values, label="train", linewidth=2)
    plt.plot(epochs, val_values, label="validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-root",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_swe_head_screen_v1/diagnostics",
    )
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}

    for architecture, label, head_name in MODEL_SPECS:
        experiment_dir = experiments_root / architecture
        history = read_history(experiment_dir / "history.csv")
        summary = read_summary(experiment_dir / "metrics_summary.json")
        best_epoch = int(summary["best_epoch"])
        best_row = next(row for row in history if int(row["epoch"]) == best_epoch)
        interpretation = classify_behavior(label, best_row, history)
        histories[label] = history

        comparison_rows.append(
            {
                "model": label,
                "architecture": architecture,
                "swe_head": head_name,
                "best_epoch": best_epoch,
                "train_r2_at_best_epoch": float(best_row["train_swe_r2"]),
                "val_r2_at_best_epoch": float(best_row["swe_r2"]),
                "r2_gap": float(best_row["train_swe_r2"] - best_row["swe_r2"]),
                "train_loss": float(best_row["total_loss"]),
                "val_loss": float(best_row["val_total_loss"]),
                "val_rmse": float(best_row["swe_rmse"]),
                "val_mae": float(best_row["swe_mae"]),
                "val_r": float(best_row["swe_pearson_r"]),
                "parameter_count": int(summary["parameter_count"]),
                "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
                "walltime_seconds": float(summary["walltime_seconds"]),
                "alpha_at_best_epoch": float(best_row.get("alpha", 0.0)),
                "final_alpha": float(summary.get("alpha", 0.0)),
                "interpretation": interpretation,
                "history_path": str(experiment_dir / "history.csv"),
                "metrics_summary_path": str(experiment_dir / "metrics_summary.json"),
                "checkpoint_path": str(experiment_dir / "best_checkpoint.pt"),
            }
        )

        plot_two_series(
            history,
            "total_loss",
            "val_total_loss",
            "SWE loss",
            f"{label} train vs validation loss",
            output_dir / f"{label.lower()}_train_vs_val_loss.png",
        )
        plot_two_series(
            history,
            "train_swe_r2",
            "swe_r2",
            "R^2",
            f"{label} train vs validation R^2",
            output_dir / f"{label.lower()}_train_vs_val_r2.png",
        )

        if label == "S3" and "alpha" in history[0]:
            epochs = [int(row["epoch"]) for row in history]
            alphas = [float(row["alpha"]) for row in history]
            plt.figure(figsize=(7, 4.5))
            plt.plot(epochs, alphas, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("alpha")
            plt.title("S3 learned gate alpha by epoch")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "s3_alpha_by_epoch.png", dpi=160)
            plt.close()

    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        plt.plot(
            [int(row["epoch"]) for row in history],
            [float(row["val_total_loss"]) for row in history],
            label=label,
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("S0-S3 validation loss by epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "combined_val_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        plt.plot(
            [int(row["epoch"]) for row in history],
            [float(row["swe_r2"]) for row in history],
            label=label,
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation R^2")
    plt.title("S0-S3 validation R^2 by epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "combined_val_r2.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        plt.plot(
            [int(row["epoch"]) for row in history],
            [float(row["train_swe_r2"]) for row in history],
            label=label,
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training R^2")
    plt.title("S0-S3 training R^2 by epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "combined_train_r2.png", dpi=160)
    plt.close()

    comparison_csv = output_dir / "s0_s3_diagnostic_table.csv"
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (output_dir / "s0_s3_diagnostic_summary.json").open("w") as handle:
        json.dump({"rows": comparison_rows}, handle, indent=2)


if __name__ == "__main__":
    main()
