from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATTERN = re.compile(
    r"^(E_S0|A_attention_only|B_ffn_only) epoch=(\d+) "
    r"train_r2=([-+0-9.eE]+) val_r2=([-+0-9.eE]+) val_loss=([-+0-9.eE]+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--parameter-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in Path(args.log).read_text().splitlines():
        match = PATTERN.match(line.strip())
        if match is None:
            continue
        variant, epoch, train_r2, val_r2, val_loss = match.groups()
        rows.append(
            {
                "variant": variant,
                "epoch": int(epoch),
                "corrected_train_swe_r2": float(train_r2),
                "val_swe_r2": float(val_r2),
                "train_minus_val_r2": float(train_r2) - float(val_r2),
                "val_swe_loss": float(val_loss),
            }
        )
    if not rows:
        raise RuntimeError("No completed E/A/B epoch records were found in the log")
    with (output_dir / "partial_completed_ablation_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    variants = ("E_S0", "A_attention_only", "B_ffn_only")
    for field, ylabel, filename in (
        ("corrected_train_swe_r2", "SWE R2", "partial_train_validation_r2.png"),
        ("val_swe_loss", "Validation SWE loss", "partial_validation_loss.png"),
    ):
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), dpi=160, sharex=False)
        for axis, variant in zip(axes, variants, strict=True):
            values = [row for row in rows if row["variant"] == variant]
            if field == "corrected_train_swe_r2":
                axis.plot([row["epoch"] for row in values], [row[field] for row in values], marker="o", markersize=3, label="Train R2 (eval mode)")
                axis.plot([row["epoch"] for row in values], [row["val_swe_r2"] for row in values], marker="o", markersize=3, label="Validation R2")
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
            else:
                axis.plot([row["epoch"] for row in values], [row[field] for row in values], marker="o", markersize=3, label="Validation SWE loss")
            best = min(values, key=lambda row: row["val_swe_loss"])
            axis.axvline(best["epoch"], color="tab:red", linestyle="--", label=f"Best validation epoch ({best['epoch']})")
            axis.set_title(variant)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)

    summary_rows = []
    for variant in variants:
        values = [row for row in rows if row["variant"] == variant]
        best = min(values, key=lambda row: row["val_swe_loss"])
        maximum_train = max(values, key=lambda row: row["corrected_train_swe_r2"])
        summary_rows.append(
            {
                "variant": variant,
                "completed_epochs": len(values),
                "best_validation_epoch": best["epoch"],
                "train_r2_at_best": best["corrected_train_swe_r2"],
                "val_r2_at_best": best["val_swe_r2"],
                "r2_gap_at_best": best["train_minus_val_r2"],
                "max_train_r2": maximum_train["corrected_train_swe_r2"],
                "epoch_of_max_train_r2": maximum_train["epoch"],
                "val_r2_at_max_train_r2": maximum_train["val_swe_r2"],
                "final_train_r2": values[-1]["corrected_train_swe_r2"],
                "final_val_r2": values[-1]["val_swe_r2"],
            }
        )
    with (output_dir / "partial_completed_ablation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    parameters = json.loads(Path(args.parameter_json).read_text())["summary"]
    lines = [
        "# Partial S1 Attention Ablation Report",
        "",
        "This report covers only the completed variants: `E_S0`, `A_attention_only`, and `B_ffn_only`. `C_gated_attention` is still running and `D_full_S1` diagnostics have not started, so no conclusion about attention entropy, latent rank, gates, or final component attribution is made here.",
        "",
        "## Capacity and Regularization",
        "",
        f"S0: {parameters['s0_complete']:,} trainable parameters. S1: {parameters['s1_complete']:,}. Difference: {parameters['s1_minus_s0']:,} ({parameters['s1_percent_increase']:.3f}%), exactly the latent block. S1 uses `Dropout(0.1)` in multi-head attention and two FFN locations; all variants use AdamW weight decay `1e-4`.",
        "",
        "## Completed Variants",
        "",
        "| Variant | Completed epochs | Best epoch | Train R2 at best | Val R2 at best | Gap at best | Max train R2 (epoch) | Val R2 at max train | Final train R2 | Final val R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {completed_epochs} | {best_validation_epoch} | {train_r2_at_best:.6f} | {val_r2_at_best:.6f} | {r2_gap_at_best:.6f} | {max_train_r2:.6f} ({epoch_of_max_train_r2}) | {val_r2_at_max_train_r2:.6f} | {final_train_r2:.6f} | {final_val_r2:.6f} |".format(**row)
        )
    lines.extend([
        "",
        "The partial evidence already shows that both attention-only and FFN-only variants can keep improving train R2 after their validation-loss-selected optimum. The full diagnosis remains pending until the gated and original-full-S1 controls finish and the per-epoch latent/attention diagnostics are written.",
        "",
        "Note: the live training script writes its full eval-mode train and validation loss history only after all variants finish. This partial report therefore includes validation loss and both corrected eval-mode R2 series, but does not substitute optimization-step loss for the still-unwritten eval-mode training loss.",
    ])
    (output_dir / "partial_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
