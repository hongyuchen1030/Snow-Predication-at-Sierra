from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODELS = {
    "S0": "S0_static_cnn_swe_only",
    "S1": "S1_static_latent_self_attention",
    "S2": "S2_static_swe_token",
    "S3": "S3_static_residual_gated_attention",
}


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-root", required=True)
    parser.add_argument("--s1-s3-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r2_dir = output_dir / "r2_plots"
    loss_dir = output_dir / "loss_plots"
    r2_dir.mkdir(exist_ok=True)
    loss_dir.mkdir(exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for model, architecture in MODELS.items():
        experiment_dir = Path(args.s0_root if model == "S0" else args.s1_s3_root) / architecture
        history = read_history(experiment_dir / "history.csv")
        summary = json.loads((experiment_dir / "metrics_summary.json").read_text())
        best_epoch = int(summary["best_epoch"])

        rows: list[dict[str, Any]] = []
        for row in history:
            extracted = {
                "model": model,
                "epoch": int(row["epoch"]),
                "corrected_train_swe_r2": float(row["train_swe_r2"]),
                "val_swe_r2": float(row["swe_r2"]),
                "train_minus_val_r2": float(row["train_swe_r2"]) - float(row["swe_r2"]),
                "train_swe_rmse": float(row["train_swe_rmse"]),
                "val_swe_rmse": float(row["swe_rmse"]),
                "train_swe_mae": float(row["train_swe_mae"]),
                "val_swe_mae": float(row["swe_mae"]),
                "train_swe_loss": float(row["train_eval_swe_loss"]),
                "val_swe_loss": float(row["val_swe_loss"]),
                "training_loss_source": "train_eval_swe_loss",
            }
            rows.append(extracted)
            combined_rows.append(extracted)

        best_row = next(row for row in rows if row["epoch"] == best_epoch)
        max_train_row = max(rows, key=lambda row: row["corrected_train_swe_r2"])
        final_row = rows[-1]
        summaries.append(
            {
                "model": model,
                "best_validation_epoch": best_epoch,
                "corrected_train_r2_at_best_epoch": best_row["corrected_train_swe_r2"],
                "validation_r2_at_best_epoch": best_row["val_swe_r2"],
                "train_minus_val_r2_at_best_epoch": best_row["train_minus_val_r2"],
                "maximum_corrected_train_r2": max_train_row["corrected_train_swe_r2"],
                "epoch_of_maximum_corrected_train_r2": max_train_row["epoch"],
                "validation_r2_at_maximum_train_r2": max_train_row["val_swe_r2"],
                "final_corrected_train_r2": final_row["corrected_train_swe_r2"],
                "final_validation_r2": final_row["val_swe_r2"],
            }
        )

        epochs = [row["epoch"] for row in rows]
        plt.figure(figsize=(8, 5), dpi=160)
        plt.plot(epochs, [row["corrected_train_swe_r2"] for row in rows], marker="o", markersize=3,
                 linewidth=1.8, label="Corrected train R2 (eval mode)")
        plt.plot(epochs, [row["val_swe_r2"] for row in rows], marker="o", markersize=3,
                 linewidth=1.8, label="Validation R2")
        plt.axhline(0.0, color="black", linewidth=0.9, linestyle=":", label="R2 = 0")
        plt.axvline(best_epoch, color="tab:red", linewidth=1.1, linestyle="--",
                    label=f"Best validation epoch ({best_epoch})")
        plt.title(f"{model}: SWE R2 by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWE R2")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(r2_dir / f"{model}_swe_r2_by_epoch.png")
        plt.close()

        plt.figure(figsize=(8, 5), dpi=160)
        plt.plot(epochs, [row["train_swe_loss"] for row in rows], marker="o", markersize=3,
                 linewidth=1.8, label="Training SWE loss (eval mode)")
        plt.plot(epochs, [row["val_swe_loss"] for row in rows], marker="o", markersize=3,
                 linewidth=1.8, label="Validation SWE loss")
        plt.axvline(best_epoch, color="tab:red", linewidth=1.1, linestyle="--",
                    label=f"Best validation epoch ({best_epoch})")
        plt.title(f"{model}: SWE Loss by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWE loss")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(loss_dir / f"{model}_swe_loss_by_epoch.png")
        plt.close()

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "combined_epoch_metrics.csv", combined_rows)
    write_csv(output_dir / "trajectory_numerical_summary.csv", summaries)
    lines = [
        "# Corrected Historical Replay Trajectories: S0-S3",
        "",
        "All training R2, RMSE, MAE, and training loss values below come from the post-epoch `model.eval()` training-set pass. The training loss source is `train_eval_swe_loss`, so it is directly comparable with `val_swe_loss`; no optimization-pass loss was substituted.",
        "",
        "| Model | Best validation epoch | Corrected train R2 at best | Validation R2 at best | Train minus val R2 | Max corrected train R2 | Epoch of max train R2 | Validation R2 at max train R2 | Final corrected train R2 | Final validation R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {best_validation_epoch} | {corrected_train_r2_at_best_epoch:.6f} | "
            "{validation_r2_at_best_epoch:.6f} | {train_minus_val_r2_at_best_epoch:.6f} | "
            "{maximum_corrected_train_r2:.6f} | {epoch_of_maximum_corrected_train_r2} | "
            "{validation_r2_at_maximum_train_r2:.6f} | {final_corrected_train_r2:.6f} | "
            "{final_validation_r2:.6f} |".format(**row)
        )
    lines.extend([
        "",
        "## Trajectory Reading",
        "",
        "- S0: corrected training R2 continues to improve after the validation-loss-selected best epoch, while validation R2 declines by the final epoch.",
        "- S1: corrected training R2 keeps rising well beyond the best validation epoch, whereas validation R2 deteriorates sharply and becomes negative by the final epoch.",
        "- S2: corrected training R2 remains effectively zero or negative throughout, indicating the model did not establish meaningful training-set explanatory skill under eval-mode inference.",
        "- S3: corrected training R2 continues improving after the best validation epoch; validation R2 later declines substantially, consistent with overfitting rather than a collapse of training fit.",
    ])
    (output_dir / "trajectory_summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "trajectory_numerical_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
