from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_SPECS = [
    ("D1_static_cnn_cpm_aqm_diag", "D1"),
    ("D2_static_cnn_cpm_only", "D2"),
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


def save_two_line_plot(
    rows: list[dict[str, float]],
    key_a: str,
    key_b: str,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(epochs, [float(row[key_a]) for row in rows], label=label_a, linewidth=2)
    plt.plot(epochs, [float(row[key_b]) for row in rows], label=label_b, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_multi_line_plot(
    rows: list[dict[str, float]],
    keys: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(7.5, 4.5))
    for key, label in zip(keys, labels, strict=True):
        if key not in rows[0]:
            continue
        plt.plot(epochs, [float(row[key]) for row in rows], label=label, linewidth=2)
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
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_aux_diag_v1/experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_aux_diag_v1_diagnostics",
    )
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for architecture, label in MODEL_SPECS:
        experiment_dir = experiments_root / architecture
        history = read_history(experiment_dir / "history.csv")
        summary = read_summary(experiment_dir / "metrics_summary.json")
        best_epoch = int(summary["best_epoch"])
        best_row = next(row for row in history if int(row["epoch"]) == best_epoch)

        save_two_line_plot(
            history,
            "swe_loss",
            "val_swe_loss",
            "train SWE loss",
            "val SWE loss",
            f"{label} SWE loss",
            "Loss",
            output_dir / f"{label.lower()}_swe_loss.png",
        )
        save_two_line_plot(
            history,
            "train_swe_r2",
            "swe_r2",
            "train SWE R^2",
            "val SWE R^2",
            f"{label} SWE R^2",
            "R^2",
            output_dir / f"{label.lower()}_swe_r2.png",
        )

        loss_keys = ["swe_loss", "cpm_loss", "aqm_loss"]
        loss_labels = ["SWE", "CPM", "AQM"]
        if label == "D2":
            loss_keys = ["swe_loss", "cpm_loss"]
            loss_labels = ["SWE", "CPM"]
        save_multi_line_plot(
            history,
            loss_keys,
            loss_labels,
            f"{label} train task losses",
            "Loss",
            output_dir / f"{label.lower()}_task_losses.png",
        )

        grad_norm_keys = ["grad_norm_swe", "grad_norm_cpm", "grad_norm_aqm"]
        grad_norm_labels = ["||g_SWE||", "||g_CPM||", "||g_AQM||"]
        if label == "D2":
            grad_norm_keys = ["grad_norm_swe", "grad_norm_cpm"]
            grad_norm_labels = ["||g_SWE||", "||g_CPM||"]
        save_multi_line_plot(
            history,
            grad_norm_keys,
            grad_norm_labels,
            f"{label} gradient norms",
            "Gradient norm",
            output_dir / f"{label.lower()}_gradient_norms.png",
        )

        save_multi_line_plot(
            history,
            ["grad_cos_swe_cpm"],
            ["cos(SWE, CPM)"],
            f"{label} cosine(SWE, CPM)",
            "Cosine similarity",
            output_dir / f"{label.lower()}_cos_swe_cpm.png",
        )
        save_multi_line_plot(
            history,
            ["grad_frac_negative_swe_cpm"],
            ["frac negative SWE-CPM"],
            f"{label} negative SWE-CPM fraction",
            "Fraction",
            output_dir / f"{label.lower()}_neg_frac_swe_cpm.png",
        )

        if label == "D1":
            save_multi_line_plot(
                history,
                ["grad_cos_swe_aqm"],
                ["cos(SWE, AQM)"],
                f"{label} cosine(SWE, AQM)",
                "Cosine similarity",
                output_dir / f"{label.lower()}_cos_swe_aqm.png",
            )
            save_multi_line_plot(
                history,
                ["grad_frac_negative_swe_aqm"],
                ["frac negative SWE-AQM"],
                f"{label} negative SWE-AQM fraction",
                "Fraction",
                output_dir / f"{label.lower()}_neg_frac_swe_aqm.png",
            )

        summary_rows.append(
            {
                "model": label,
                "architecture": architecture,
                "best_epoch": best_epoch,
                "train_swe_r2": float(best_row["train_swe_r2"]),
                "val_swe_r2": float(best_row["swe_r2"]),
                "val_rmse": float(best_row["swe_rmse"]),
                "val_mae": float(best_row["swe_mae"]),
                "val_r": float(best_row["swe_pearson_r"]),
                "train_swe_loss": float(best_row["swe_loss"]),
                "val_swe_loss": float(best_row["val_swe_loss"]),
                "train_cpm_loss": float(best_row["cpm_loss"]),
                "val_cpm_loss": float(best_row["val_cpm_loss"]),
                "train_aqm_loss": float(best_row["aqm_loss"]),
                "val_aqm_loss": float(best_row["val_aqm_loss"]),
                "cpm_r": float(best_row["cpm_pearson_r"]),
                "aqm_r": float(best_row["aqm_pearson_r"]),
                "grad_norm_swe": float(best_row.get("grad_norm_swe", float("nan"))),
                "grad_norm_cpm": float(best_row.get("grad_norm_cpm", float("nan"))),
                "grad_norm_aqm": float(best_row.get("grad_norm_aqm", float("nan"))),
                "mean_cos_swe_cpm": float(best_row.get("grad_cos_swe_cpm", float("nan"))),
                "frac_negative_swe_cpm": float(best_row.get("grad_frac_negative_swe_cpm", float("nan"))),
                "mean_cos_swe_aqm": float(best_row.get("grad_cos_swe_aqm", float("nan"))),
                "frac_negative_swe_aqm": float(best_row.get("grad_frac_negative_swe_aqm", float("nan"))),
                "parameter_count": int(summary["parameter_count"]),
                "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
                "walltime_seconds": float(summary["walltime_seconds"]),
                "history_path": str(experiment_dir / "history.csv"),
                "metrics_summary_path": str(experiment_dir / "metrics_summary.json"),
                "checkpoint_path": str(experiment_dir / "best_checkpoint.pt"),
            }
        )

    with (output_dir / "d1_d2_diagnostic_summary.json").open("w") as handle:
        json.dump({"rows": summary_rows}, handle, indent=2)
    with (output_dir / "d1_d2_diagnostic_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
