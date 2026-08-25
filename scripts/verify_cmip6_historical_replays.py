from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARCHITECTURES = (
    "S0_static_cnn_swe_only",
    "S1_static_latent_self_attention",
    "S2_static_swe_token",
    "S3_static_residual_gated_attention",
)


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", required=True)
    parser.add_argument("--s0-replay-root", required=True)
    parser.add_argument("--s1-s3-replay-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    historical_root = Path(args.historical_root)
    s0_replay_root = Path(args.s0_replay_root)
    s1_s3_replay_root = Path(args.s1_s3_replay_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        historical_dir = historical_root / architecture
        replay_dir = (
            s0_replay_root / architecture
            if architecture == "S0_static_cnn_swe_only"
            else s1_s3_replay_root / architecture
        )
        historical_history = read_history(historical_dir / "history.csv")
        replay_history = read_history(replay_dir / "history.csv")
        historical_summary = json.loads((historical_dir / "metrics_summary.json").read_text())
        replay_summary = json.loads((replay_dir / "metrics_summary.json").read_text())
        replay_by_epoch = {int(row["epoch"]): row for row in replay_history}

        for historical_row in historical_history:
            epoch = int(historical_row["epoch"])
            replay_row = replay_by_epoch.get(epoch)
            if replay_row is None:
                raise RuntimeError(f"{architecture}: replay has no epoch {epoch}")
            historical_loss = float(historical_row["val_total_loss"])
            replay_loss = float(replay_row["val_total_loss"])
            historical_r2 = float(historical_row["val_swe_r2"])
            replay_r2 = float(replay_row["val_swe_r2"])
            epoch_rows.append(
                {
                    "model": architecture.split("_")[0],
                    "architecture": architecture,
                    "epoch": epoch,
                    "historical_val_total_loss": historical_loss,
                    "replay_val_total_loss": replay_loss,
                    "abs_val_total_loss_difference": abs(historical_loss - replay_loss),
                    "historical_val_swe_r2": historical_r2,
                    "replay_val_swe_r2": replay_r2,
                    "abs_val_swe_r2_difference": abs(historical_r2 - replay_r2),
                }
            )

        model_epoch_rows = [row for row in epoch_rows if row["architecture"] == architecture]
        historical_best_epoch = int(historical_summary["best_epoch"])
        replay_best_epoch = int(replay_summary["best_epoch"])
        historical_best_row = next(row for row in historical_history if int(row["epoch"]) == historical_best_epoch)
        replay_at_historical_best = replay_by_epoch[historical_best_epoch]
        historical_train_r2 = float(historical_best_row["train_swe_r2"])
        corrected_train_r2 = float(replay_at_historical_best["train_swe_r2"])
        replay_val_r2 = float(replay_at_historical_best["val_swe_r2"])
        summary_rows.append(
            {
                "model": architecture.split("_")[0],
                "architecture": architecture,
                "historical_best_epoch": historical_best_epoch,
                "replay_best_epoch": replay_best_epoch,
                "historical_train_r2": historical_train_r2,
                "corrected_train_r2": corrected_train_r2,
                "historical_val_r2": float(historical_best_row["val_swe_r2"]),
                "replay_val_r2": replay_val_r2,
                "corrected_r2_gap": corrected_train_r2 - replay_val_r2,
                "max_abs_val_loss_difference": max(row["abs_val_total_loss_difference"] for row in model_epoch_rows),
                "max_abs_val_r2_difference": max(row["abs_val_swe_r2_difference"] for row in model_epoch_rows),
                "historical_trajectory_reproduced": historical_best_epoch == replay_best_epoch
                and max(row["abs_val_total_loss_difference"] for row in model_epoch_rows) == 0.0
                and max(row["abs_val_swe_r2_difference"] for row in model_epoch_rows) == 0.0,
            }
        )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "epoch_by_epoch_validation_comparison.csv", epoch_rows)
    write_csv(output_dir / "historical_replay_summary.csv", summary_rows)
    (output_dir / "historical_replay_verification.json").write_text(
        json.dumps({"epoch_by_epoch": epoch_rows, "summary": summary_rows}, indent=2) + "\n"
    )
    lines = [
        "# CMIP6 S0-S3 Historical Replay Verification",
        "",
        "| Model | Historical best epoch | Replay best epoch | Historical train R2 | Corrected train R2 | Historical val R2 | Replay val R2 | Corrected R2 gap | Max abs val loss diff | Max abs val R2 diff | Historical trajectory reproduced? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {historical_best_epoch} | {replay_best_epoch} | {historical_train_r2:.12g} | "
            "{corrected_train_r2:.12g} | {historical_val_r2:.12g} | {replay_val_r2:.12g} | "
            "{corrected_r2_gap:.12g} | {max_abs_val_loss_difference:.12g} | "
            "{max_abs_val_r2_difference:.12g} | {historical_trajectory_reproduced} |".format(**row)
        )
    (output_dir / "historical_replay_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
