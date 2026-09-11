from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def save_plot(
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
        "--experiment-dir",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_kendall_aux_v1/experiments/K1_static_cnn_kendall_aux",
    )
    parser.add_argument(
        "--output-dir",
        default="/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_kendall_aux_v1_diagnostics",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = read_history(experiment_dir / "history.csv")

    save_plot(
        history,
        ["swe_loss", "val_swe_loss"],
        ["train SWE loss", "val SWE loss"],
        "Kendall SWE loss",
        "Loss",
        output_dir / "kendall_swe_loss.png",
    )
    save_plot(
        history,
        ["train_swe_r2", "swe_r2"],
        ["train SWE R^2", "val SWE R^2"],
        "Kendall SWE R^2",
        "R^2",
        output_dir / "kendall_swe_r2.png",
    )
    save_plot(
        history,
        ["swe_loss", "cpm_loss", "aqm_loss"],
        ["SWE raw loss", "CPM raw loss", "AQM raw loss"],
        "Kendall raw task losses",
        "Loss",
        output_dir / "kendall_raw_task_losses.png",
    )
    save_plot(
        history,
        ["s_swe", "s_cpm", "s_aqm"],
        ["s_SWE", "s_CPM", "s_AQM"],
        "Kendall log variances",
        "log variance",
        output_dir / "kendall_log_variances.png",
    )
    save_plot(
        history,
        ["w_swe", "w_cpm", "w_aqm"],
        ["w_SWE", "w_CPM", "w_AQM"],
        "Kendall effective task weights",
        "Weight",
        output_dir / "kendall_effective_weights.png",
    )
    save_plot(
        history,
        ["w_rel_swe", "w_rel_cpm", "w_rel_aqm"],
        ["rel SWE", "rel CPM", "rel AQM"],
        "Kendall normalized relative task weights",
        "Relative weight",
        output_dir / "kendall_relative_weights.png",
    )
    save_plot(
        history,
        ["grad_norm_swe", "grad_norm_cpm", "grad_norm_aqm"],
        ["||g_SWE||", "||g_CPM||", "||g_AQM||"],
        "Kendall raw gradient norms",
        "Gradient norm",
        output_dir / "kendall_raw_gradient_norms.png",
    )
    save_plot(
        history,
        ["weighted_grad_norm_swe", "weighted_grad_norm_cpm", "weighted_grad_norm_aqm"],
        ["w_SWE||g_SWE||", "w_CPM||g_CPM||", "w_AQM||g_AQM||"],
        "Kendall weighted gradient magnitudes",
        "Weighted gradient magnitude",
        output_dir / "kendall_weighted_gradient_norms.png",
    )
    save_plot(
        history,
        ["grad_cos_swe_cpm", "grad_cos_swe_aqm"],
        ["cos(SWE, CPM)", "cos(SWE, AQM)"],
        "Kendall SWE auxiliary gradient cosine",
        "Cosine similarity",
        output_dir / "kendall_gradient_cosines.png",
    )
    save_plot(
        history,
        ["grad_frac_negative_swe_cpm", "grad_frac_negative_swe_aqm"],
        ["frac negative SWE-CPM", "frac negative SWE-AQM"],
        "Kendall conflicting batch fraction",
        "Fraction",
        output_dir / "kendall_conflict_fraction.png",
    )

    summary = {
        "history_path": str(experiment_dir / "history.csv"),
        "metrics_summary_path": str(experiment_dir / "metrics_summary.json"),
        "plot_dir": str(output_dir),
    }
    with (output_dir / "kendall_plot_manifest.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
