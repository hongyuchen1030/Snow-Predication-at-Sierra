#!/usr/bin/env python3
"""Summarize reconstruction quality for completed DAE SST LOYO artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DAE reconstruction quality from existing artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Completed undercomplete DAE artifact directory.",
    )
    return parser.parse_args()


def append_readme_section(readme_path: Path) -> None:
    section_title = "## Reconstruction-quality interpretation"
    section_body = "\n".join(
        [
            section_title,
            "",
            "- `heldout_reconstruction_mse` measures how well the k-dimensional latent code reconstructs SST for water years that were excluded from DAE training.",
            "- If reconstruction improves with larger `k`, the encoder is preserving more of the full SST field information.",
            "- This reconstruction view is separate from Sierra SWE predictability.",
            "- The current SWE results show that better SST reconstruction does not automatically translate into better held-out SWE prediction.",
            "- In each LOYO fold, the autoencoder training objective is the reconstruction loss over the 36 training years only.",
            "- The held-out reconstruction loss is not the training objective; it is a generalization diagnostic for the left-out SST year.",
            "",
        ]
    )

    existing = readme_path.read_text()
    if section_title in existing:
        prefix = existing.split(section_title)[0].rstrip() + "\n\n"
        readme_path.write_text(prefix + section_body)
    else:
        separator = "\n\n" if not existing.endswith("\n\n") else ""
        readme_path.write_text(existing + separator + section_body)


def make_summary_table(recon_df: pd.DataFrame) -> pd.DataFrame:
    grouped = recon_df.groupby("k", sort=True)
    summary = grouped.agg(
        heldout_reconstruction_mse_mean=("heldout_reconstruction_mse", "mean"),
        heldout_reconstruction_mse_median=("heldout_reconstruction_mse", "median"),
        heldout_reconstruction_mse_std=("heldout_reconstruction_mse", "std"),
        heldout_reconstruction_mse_min=("heldout_reconstruction_mse", "min"),
        heldout_reconstruction_mse_max=("heldout_reconstruction_mse", "max"),
        train_reconstruction_mse_mean=("train_reconstruction_mse", "mean"),
        validation_reconstruction_mse_mean=("validation_reconstruction_mse", "mean"),
        mean_epochs_trained=("epochs_trained", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
    )
    return summary.reset_index()


def make_loss_decomposition_by_k_seed(recon_df: pd.DataFrame) -> pd.DataFrame:
    df = recon_df.copy()
    df["heldout_minus_train"] = (
        df["heldout_reconstruction_mse"] - df["train_reconstruction_mse"]
    )
    df["validation_minus_train"] = (
        df["validation_reconstruction_mse"] - df["train_reconstruction_mse"]
    )

    grouped = df.groupby(["k", "seed"], sort=True)
    summary = grouped.agg(
        train_mse_mean=("train_reconstruction_mse", "mean"),
        train_mse_median=("train_reconstruction_mse", "median"),
        train_mse_std=("train_reconstruction_mse", "std"),
        validation_mse_mean=("validation_reconstruction_mse", "mean"),
        validation_mse_median=("validation_reconstruction_mse", "median"),
        validation_mse_std=("validation_reconstruction_mse", "std"),
        heldout_mse_mean=("heldout_reconstruction_mse", "mean"),
        heldout_mse_median=("heldout_reconstruction_mse", "median"),
        heldout_mse_std=("heldout_reconstruction_mse", "std"),
        heldout_minus_train_mean=("heldout_minus_train", "mean"),
        validation_minus_train_mean=("validation_minus_train", "mean"),
    )
    return summary.reset_index()


def make_loss_decomposition_by_k(loss_by_k_seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = loss_by_k_seed_df.groupby("k", sort=True)
    summary = grouped.agg(
        train_mse_mean_across_seeds=("train_mse_mean", "mean"),
        validation_mse_mean_across_seeds=("validation_mse_mean", "mean"),
        heldout_mse_mean_across_seeds=("heldout_mse_mean", "mean"),
        heldout_minus_train_mean_across_seeds=("heldout_minus_train_mean", "mean"),
        validation_minus_train_mean_across_seeds=(
            "validation_minus_train_mean",
            "mean",
        ),
    )
    return summary.reset_index()


def plot_heldout_vs_k(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        summary_df["k"],
        summary_df["heldout_reconstruction_mse_mean"],
        yerr=summary_df["heldout_reconstruction_mse_std"],
        fmt="-o",
        capsize=4,
        linewidth=2,
        color="#1f77b4",
    )
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Held-out reconstruction MSE")
    ax.set_title("Held-out SST reconstruction vs latent dimension")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_train_val_heldout(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        summary_df["k"],
        summary_df["train_reconstruction_mse_mean"],
        marker="o",
        linewidth=2,
        label="Train",
        color="#2ca02c",
    )
    ax.plot(
        summary_df["k"],
        summary_df["validation_reconstruction_mse_mean"],
        marker="o",
        linewidth=2,
        label="Validation",
        color="#ff7f0e",
    )
    ax.plot(
        summary_df["k"],
        summary_df["heldout_reconstruction_mse_mean"],
        marker="o",
        linewidth=2,
        label="Held-out",
        color="#1f77b4",
    )
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Train, validation, and held-out reconstruction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_by_year(recon_df: pd.DataFrame, output_path: Path) -> None:
    yearly = (
        recon_df.groupby(["heldout_year", "k"], sort=True)["heldout_reconstruction_mse"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, group in yearly.groupby("k", sort=True):
        ax.plot(
            group["heldout_year"],
            group["heldout_reconstruction_mse"],
            marker="o",
            linewidth=1.8,
            label=f"k={k}",
        )
    ax.set_xlabel("Held-out water year")
    ax.set_ylabel("Mean held-out reconstruction MSE")
    ax.set_title("Held-out reconstruction MSE by year")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_by_seed(recon_df: pd.DataFrame, output_path: Path) -> None:
    seed_summary = (
        recon_df.groupby(["seed", "k"], sort=True)["heldout_reconstruction_mse"]
        .mean()
        .reset_index()
    )
    seeds = sorted(seed_summary["seed"].unique())
    ks = sorted(seed_summary["k"].unique())
    x = np.arange(len(seeds))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.linspace(-width, width, num=len(ks))
    for offset, k in zip(offsets, ks):
        group = seed_summary[seed_summary["k"] == k].sort_values("seed")
        ax.bar(
            x + offset,
            group["heldout_reconstruction_mse"],
            width=width,
            label=f"k={k}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean held-out reconstruction MSE")
    ax.set_title("Held-out reconstruction MSE by seed")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_loss_decomposition_by_k(
    loss_by_k_seed_df: pd.DataFrame, output_path: Path
) -> None:
    grouped = loss_by_k_seed_df.groupby("k", sort=True)
    summary = grouped.agg(
        train_mean=("train_mse_mean", "mean"),
        train_std=("train_mse_mean", "std"),
        validation_mean=("validation_mse_mean", "mean"),
        validation_std=("validation_mse_mean", "std"),
        heldout_mean=("heldout_mse_mean", "mean"),
        heldout_std=("heldout_mse_mean", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["k"],
        summary["train_mean"],
        yerr=summary["train_std"].fillna(0.0),
        fmt="-o",
        linewidth=2,
        capsize=4,
        label="Train",
        color="#2ca02c",
    )
    ax.errorbar(
        summary["k"],
        summary["validation_mean"],
        yerr=summary["validation_std"].fillna(0.0),
        fmt="-o",
        linewidth=2,
        capsize=4,
        label="Validation",
        color="#ff7f0e",
    )
    ax.errorbar(
        summary["k"],
        summary["heldout_mean"],
        yerr=summary["heldout_std"].fillna(0.0),
        fmt="-o",
        linewidth=2,
        capsize=4,
        label="Held-out",
        color="#1f77b4",
    )
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Mean reconstruction MSE across LOYO folds")
    ax.set_title("DAE reconstruction-loss decomposition by k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_generalization_gap_by_k(
    loss_by_k_seed_df: pd.DataFrame, output_path: Path
) -> None:
    grouped = loss_by_k_seed_df.groupby("k", sort=True)
    summary = grouped.agg(
        heldout_gap_mean=("heldout_minus_train_mean", "mean"),
        heldout_gap_std=("heldout_minus_train_mean", "std"),
        validation_gap_mean=("validation_minus_train_mean", "mean"),
        validation_gap_std=("validation_minus_train_mean", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["k"],
        summary["heldout_gap_mean"],
        yerr=summary["heldout_gap_std"].fillna(0.0),
        fmt="-o",
        linewidth=2,
        capsize=4,
        label="Held-out minus train",
        color="#d62728",
    )
    ax.errorbar(
        summary["k"],
        summary["validation_gap_mean"],
        yerr=summary["validation_gap_std"].fillna(0.0),
        fmt="-o",
        linewidth=2,
        capsize=4,
        label="Validation minus train",
        color="#9467bd",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("Generalization gap")
    ax.set_title("DAE reconstruction generalization gap by k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()

    recon_path = artifact_dir / "dae_fold_reconstruction.csv"
    metrics_path = artifact_dir / "dae_metrics_by_k_aggregate.csv"
    readme_path = artifact_dir / "README.md"

    recon_df = pd.read_csv(recon_path)
    metrics_df = pd.read_csv(metrics_path)

    summary_df = make_summary_table(recon_df)
    summary_df.to_csv(artifact_dir / "dae_reconstruction_summary_by_k.csv", index=False)

    loss_by_k_seed_df = make_loss_decomposition_by_k_seed(recon_df)
    loss_by_k_seed_df.to_csv(
        artifact_dir / "dae_reconstruction_loss_decomposition_by_k_seed.csv",
        index=False,
    )

    loss_by_k_df = make_loss_decomposition_by_k(loss_by_k_seed_df)
    loss_by_k_df.to_csv(
        artifact_dir / "dae_reconstruction_loss_decomposition_by_k.csv",
        index=False,
    )

    prediction_vs_reconstruction_df = summary_df[
        [
            "k",
            "heldout_reconstruction_mse_mean",
            "heldout_reconstruction_mse_median",
        ]
    ].merge(
        metrics_df[
            [
                "k",
                "rmse_mean",
                "rmse_median",
                "r_mean",
                "r2_mean",
                "mean_delta_squared_error_vs_baseline_mean",
            ]
        ].rename(
            columns={
                "rmse_mean": "swe_rmse_mean",
                "rmse_median": "swe_rmse_median",
                "r_mean": "swe_r_mean",
                "r2_mean": "swe_r2_mean",
            }
        ),
        on="k",
        how="left",
    )
    prediction_vs_reconstruction_df.to_csv(
        artifact_dir / "dae_prediction_vs_reconstruction_summary.csv",
        index=False,
    )

    plot_heldout_vs_k(
        summary_df, artifact_dir / "dae_heldout_reconstruction_mse_vs_k.png"
    )
    plot_train_val_heldout(
        summary_df,
        artifact_dir / "dae_train_validation_heldout_reconstruction_vs_k.png",
    )
    plot_by_year(
        recon_df, artifact_dir / "dae_heldout_reconstruction_mse_by_year.png"
    )
    plot_by_seed(
        recon_df, artifact_dir / "dae_heldout_reconstruction_mse_by_seed.png"
    )
    plot_loss_decomposition_by_k(
        loss_by_k_seed_df,
        artifact_dir / "dae_train_validation_heldout_loss_decomposition_by_k.png",
    )
    plot_generalization_gap_by_k(
        loss_by_k_seed_df,
        artifact_dir / "dae_reconstruction_generalization_gap_by_k.png",
    )

    append_readme_section(readme_path)

    best_recon_k = int(
        summary_df.loc[
            summary_df["heldout_reconstruction_mse_mean"].idxmin(), "k"
        ]
    )
    best_swe_k = int(metrics_df.loc[metrics_df["rmse_mean"].idxmin(), "k"])
    recon_rank = summary_df.sort_values("heldout_reconstruction_mse_mean")["k"].tolist()
    swe_rank = metrics_df.sort_values("rmse_mean")["k"].tolist()

    print("DAE reconstruction summary:")
    for _, row in summary_df.sort_values("k").iterrows():
        print(
            f"k={int(row['k'])} heldout reconstruction MSE mean: "
            f"{row['heldout_reconstruction_mse_mean']:.6f}"
        )
    print(f"best k by reconstruction: {best_recon_k}")
    print(f"best k by SWE RMSE: {best_swe_k}")
    print(
        "whether reconstruction ranking matches SWE ranking: "
        f"{'yes' if recon_rank == swe_rank else 'no'}"
    )


if __name__ == "__main__":
    main()
