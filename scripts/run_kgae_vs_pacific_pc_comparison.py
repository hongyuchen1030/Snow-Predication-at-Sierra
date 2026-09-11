#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
KGAE_MODE_NAMES = {
    1: "Decadal",
    2: "Interannual",
    3: "Quasibiennial",
    4: "HF1",
    5: "HF2",
}


def corr_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x_use = x[mask]
    y_use = y[mask]
    if np.allclose(np.std(x_use), 0.0) or np.allclose(np.std(y_use), 0.0):
        return np.nan
    return float(np.corrcoef(x_use, y_use)[0, 1])


def build_long_table(kgae_df: pd.DataFrame, pacific_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = kgae_df.merge(
        pacific_df,
        on=["water_year", "obs_swe"],
        how="inner",
        validate="one_to_one",
    )
    for _, row in merged.iterrows():
        for month in MONTHS:
            out = {
                "water_year": int(row["water_year"]),
                "month": month,
                "obs_swe": float(row["obs_swe"]),
            }
            for idx in range(1, 6):
                out[f"kgae_z{idx}"] = float(row[f"kgae_z{idx}_{month}"])
            for idx in range(1, 7):
                out[f"Pacific_PC{idx}"] = float(row[f"Pacific_PC{idx}_{month}"])
            if f"Nino34_{month}" in row.index:
                out["Nino34"] = float(row[f"Nino34_{month}"])
            rows.append(out)
    return pd.DataFrame(rows)


def make_heatmap(corr_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = corr_df.pivot(index="kgae_mode_name", columns="pacific_predictor", values="correlation")
    plot_df = plot_df.loc[
        [KGAE_MODE_NAMES[i] for i in range(1, 6)],
        [c for c in plot_df.columns if c.startswith("Pacific_PC")] + ([c for c in plot_df.columns if c == "Nino34"]),
    ]

    plt.figure(figsize=(9, 4.8))
    sns.heatmap(
        plot_df,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("KGAE vs Pacific PC Monthly Correlations\nmatched by water year and month")
    plt.xlabel("Pacific predictor family")
    plt.ylabel("KGAE latent")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_swe_correlations(kgae_df: pd.DataFrame, pacific_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx in range(1, 6):
        for month in MONTHS:
            col = f"kgae_z{idx}_{month}"
            corr = corr_or_nan(kgae_df[col].to_numpy(), kgae_df["obs_swe"].to_numpy())
            rows.append(
                {
                    "family": "KGAE",
                    "feature_name": col,
                    "mode_name": KGAE_MODE_NAMES[idx],
                    "month": month,
                    "correlation_with_obs_swe": corr,
                    "abs_correlation_with_obs_swe": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    for idx in range(1, 7):
        for month in MONTHS:
            col = f"Pacific_PC{idx}_{month}"
            corr = corr_or_nan(pacific_df[col].to_numpy(), pacific_df["obs_swe"].to_numpy())
            rows.append(
                {
                    "family": "Pacific_PC",
                    "feature_name": col,
                    "mode_name": f"Pacific_PC{idx}",
                    "month": month,
                    "correlation_with_obs_swe": corr,
                    "abs_correlation_with_obs_swe": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    if "Nino34_Sep" in pacific_df.columns:
        for month in MONTHS:
            col = f"Nino34_{month}"
            corr = corr_or_nan(pacific_df[col].to_numpy(), pacific_df["obs_swe"].to_numpy())
            rows.append(
                {
                    "family": "Nino34",
                    "feature_name": col,
                    "mode_name": "Nino34",
                    "month": month,
                    "correlation_with_obs_swe": corr,
                    "abs_correlation_with_obs_swe": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kgae-table",
        type=Path,
        default=Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_pacific_latent_swe_loyo/kgae_swe_predictor_table.csv"),
    )
    parser.add_argument(
        "--pacific-table",
        type=Path,
        default=Path("artifacts/cobe2_sierra_swe_lod_setup/exact_Z1_Z2_plus_PacificPC_Nino34_loyo/z1_z2_pacificpc_nino34_predictor_table.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kgae_pacific_latent_swe_loyo"),
    )
    parser.add_argument(
        "--home-output-dir",
        type=Path,
        default=Path("artifacts/kgae_pacific_latent_swe_loyo"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.home_output_dir.mkdir(parents=True, exist_ok=True)

    kgae_df = pd.read_csv(args.kgae_table)
    pacific_df = pd.read_csv(args.pacific_table)

    long_df = build_long_table(kgae_df, pacific_df)

    corr_rows = []
    pacific_predictors = [f"Pacific_PC{i}" for i in range(1, 7)]
    if "Nino34" in long_df.columns:
        pacific_predictors.append("Nino34")
    for idx in range(1, 6):
        kgae_col = f"kgae_z{idx}"
        mode_name = KGAE_MODE_NAMES[idx]
        for pacific_name in pacific_predictors:
            corr = corr_or_nan(long_df[kgae_col].to_numpy(), long_df[pacific_name].to_numpy())
            corr_rows.append(
                {
                    "kgae_mode_index": idx,
                    "kgae_mode_name": mode_name,
                    "pacific_predictor": pacific_name,
                    "correlation": corr,
                    "abs_correlation": abs(corr) if np.isfinite(corr) else np.nan,
                    "n_samples": int(np.isfinite(long_df[kgae_col].to_numpy()).sum()),
                }
            )
    corr_df = pd.DataFrame(corr_rows).sort_values(["kgae_mode_index", "pacific_predictor"])
    corr_csv = args.output_dir / "kgae_vs_pacific_pc_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)

    swe_corr_df = summarize_swe_correlations(kgae_df, pacific_df).sort_values(
        ["family", "abs_correlation_with_obs_swe"], ascending=[True, False]
    )
    swe_corr_csv = args.output_dir / "kgae_vs_pacific_pc_swe_correlations.csv"
    swe_corr_df.to_csv(swe_corr_csv, index=False)

    heatmap_path = args.output_dir / "kgae_vs_pacific_pc_correlation_heatmap.png"
    make_heatmap(corr_df, heatmap_path)

    for src in [corr_csv, swe_corr_csv, heatmap_path]:
        dst = args.home_output_dir / src.name
        if dst.resolve() == src.resolve():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    nino_corr = corr_df[corr_df["pacific_predictor"] == "Nino34"].sort_values("abs_correlation", ascending=False)
    enso_like_pc = (
        pacific_df[[f"Pacific_PC{i}_{m}" for i in range(1, 7) for m in MONTHS] + [f"Nino34_{m}" for m in MONTHS]]
        .pipe(
            lambda df: pd.DataFrame(
                [
                    {
                        "pacific_pc": f"Pacific_PC{i}",
                        "corr_with_nino34": corr_or_nan(
                            np.concatenate([pacific_df[f"Pacific_PC{i}_{m}"].to_numpy() for m in MONTHS]),
                            np.concatenate([pacific_df[f"Nino34_{m}"].to_numpy() for m in MONTHS]),
                        ),
                    }
                    for i in range(1, 7)
                ]
            )
        )
        .assign(abs_corr_with_nino34=lambda df: df["corr_with_nino34"].abs())
        .sort_values("abs_corr_with_nino34", ascending=False)
    )
    pdo_like_candidate = corr_df[corr_df["kgae_mode_name"] == "Decadal"].sort_values(
        "abs_correlation", ascending=False
    )

    print("Authoritative Pacific table:", args.pacific_table)
    print("KGAE table:", args.kgae_table)
    if not nino_corr.empty:
        top = nino_corr.iloc[0]
        print(
            "Strongest KGAE-Nino34 match:",
            top["kgae_mode_name"],
            f"corr={top['correlation']:.6f}",
        )
    top_enso_pc = enso_like_pc.iloc[0]
    print(
        "Most Nino34-aligned Pacific PC:",
        top_enso_pc["pacific_pc"],
        f"corr={top_enso_pc['corr_with_nino34']:.6f}",
    )
    top_pdo = pdo_like_candidate.iloc[0]
    print(
        "Best decadal/PDO-like Pacific candidate via KGAE-Decadal alignment:",
        top_pdo["pacific_predictor"],
        f"corr={top_pdo['correlation']:.6f}",
    )
    print("Wrote:", corr_csv)
    print("Wrote:", swe_corr_csv)
    print("Wrote:", heatmap_path)


if __name__ == "__main__":
    main()
