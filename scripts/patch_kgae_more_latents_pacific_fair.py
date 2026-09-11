#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()
FINAL_HOME_OUT = PROJECT_ROOT / "artifacts" / "kgae_more_latents_swe_loyo"
HOME_OUT = Path("/tmp/kgae_more_latents_swe_loyo_patch")
HEAVY_OUT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_more_latents_swe_loyo")

BASELINE_TABLE = Path(
    "/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/"
    "z1z2_plus_amv_k5_loyo/z1z2_amv_k5_predictor_table.csv"
)
PACIFIC_TABLE = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_plus_PacificPC_Nino34_loyo"
    / "z1_z2_pacificpc_nino34_predictor_table.csv"
)
PACIFIC_PRED = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_plus_PacificPC_Nino34_loyo"
    / "z1_z2_pacificpc_nino34_loyo_predictions.csv"
)
PILOT_PRED = HEAVY_OUT / "kgae_more_latents_swe_ridge_loyo_predictions.csv"
PILOT_METRICS = HEAVY_OUT / "kgae_more_latents_swe_ridge_loyo_metrics.csv"
REPORT_MD = HOME_OUT / "REPORT.md"
ALPHAS = np.logspace(-6, 6, 49)

MODEL_ORDER = [
    "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC",
    "Z1_Z2_PACIFIC_LOD_RIDGE",
    "PACIFIC_PC6_RIDGE",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10",
]

DISPLAY_NAME = {
    "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC": "7-col Pacific+Atlantic",
    "Z1_Z2_PACIFIC_LOD_RIDGE": "Z1/Z2 Pacific LOD",
    "PACIFIC_PC6_RIDGE": "Pacific PC6",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5": "KGAE k=5",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10": "KGAE k=10",
}

COLOR = {
    "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC": "#c44e52",
    "Z1_Z2_PACIFIC_LOD_RIDGE": "#55a868",
    "PACIFIC_PC6_RIDGE": "#8172b3",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5": "#4c72b0",
    "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10": "#1f4e8c",
}


def compute_prediction_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_obs = df["y_obs"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    err = y_pred - y_obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    obs_std = float(np.std(y_obs, ddof=0))
    pred_std = float(np.std(y_pred, ddof=0))
    r = float(np.corrcoef(y_obs, y_pred)[0, 1]) if len(y_obs) > 1 else np.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    sign_accuracy = float(np.mean(np.sign(y_obs) == np.sign(y_pred)))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "r": r,
        "sign_accuracy": sign_accuracy,
        "bias": bias,
        "pred_std": pred_std,
        "obs_std": obs_std,
        "number_of_years": int(len(y_obs)),
    }


def select_ridge_alpha_inner_loyo(x_train: np.ndarray, y_train: np.ndarray) -> float:
    n = x_train.shape[0]
    alpha_errors = np.zeros_like(ALPHAS, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xtr = x_train[mask]
        ytr = y_train[mask]
        mu = xtr.mean(axis=0)
        sigma = xtr.std(axis=0)
        sigma[sigma == 0] = 1.0
        xs = (xtr - mu) / sigma
        xt = (x_train[i : i + 1] - mu) / sigma
        y_mu = float(np.mean(ytr))
        rhs = xs.T @ (ytr - y_mu)
        evals, evecs = np.linalg.eigh(xs.T @ xs)
        proj = evecs.T @ rhs
        xt_proj = xt.reshape(-1) @ evecs
        preds = y_mu + np.sum((xt_proj * proj)[None, :] / (evals[None, :] + ALPHAS[:, None]), axis=1)
        alpha_errors += (preds - float(y_train[i])) ** 2
    best_idx = int(np.argmin(alpha_errors / n))
    return float(ALPHAS[best_idx])


def ridge_predict_closed_form(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> float:
    x_mu = x_train.mean(axis=0)
    x_sigma = x_train.std(axis=0)
    x_sigma[x_sigma == 0] = 1.0
    xs = (x_train - x_mu) / x_sigma
    xt = (x_test - x_mu) / x_sigma
    y_mu = float(np.mean(y_train))
    yc = y_train - y_mu
    gram = xs.T @ xs
    coef = np.linalg.solve(gram + alpha * np.eye(xs.shape[1]), xs.T @ yc)
    return float((y_mu + xt @ coef).reshape(-1)[0])


def run_loyo_ridge(
    model_name: str,
    model_family: str,
    predictor_set: str,
    predictor_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    data = predictor_df.sort_values("water_year").reset_index(drop=True)
    x_all = data[feature_cols].to_numpy(dtype=float)
    y_all = data["obs_swe"].to_numpy(dtype=float)
    years = data["water_year"].to_numpy(dtype=int)
    for i, wy in enumerate(years):
        print(f"  outer fold {i+1}/{len(years)} heldout={wy}", flush=True)
        mask = np.ones(len(years), dtype=bool)
        mask[i] = False
        alpha = select_ridge_alpha_inner_loyo(x_all[mask], y_all[mask])
        pred = ridge_predict_closed_form(x_all[mask], y_all[mask], x_all[i : i + 1], alpha)
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "water_year": int(wy),
                "y_obs": float(y_all[i]),
                "y_pred": pred,
                "alpha_selected": alpha,
                "predictor_set": predictor_set,
                "k": int(model_name.rsplit("_k", 1)[1]) if "_k" in model_name else np.nan,
            }
        )
    pred_df = pd.DataFrame(rows)
    metrics = {
        "model_name": model_name,
        "model_family": model_family,
        "predictor_set": predictor_set,
        "k": int(model_name.rsplit("_k", 1)[1]) if "_k" in model_name else np.nan,
    }
    metrics.update(compute_prediction_metrics(pred_df))
    return pred_df, metrics


def load_existing_baseline_predictions() -> pd.DataFrame:
    pilot_pred = pd.read_csv(PILOT_PRED)
    base = pilot_pred.loc[pilot_pred["model_name"] == "BASELINE_7COL_RIDGE"].copy()
    base = base.rename(columns={"k": "k"})
    base["model_name"] = "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC"
    base["model_family"] = "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC"
    base["predictor_set"] = "Z1,Z2 plus five Atlantic AMV/AMO/AQM predictors"
    return base[
        ["model_name", "model_family", "water_year", "y_obs", "y_pred", "alpha_selected", "predictor_set", "k"]
    ].sort_values("water_year")


def load_existing_kgae_predictions() -> pd.DataFrame:
    pilot_pred = pd.read_csv(PILOT_PRED)
    kgae = pilot_pred.loc[
        pilot_pred["model_name"].isin(
            ["KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5", "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10"]
        )
    ].copy()
    kgae["predictor_set"] = kgae["model_name"].map(
        {
            "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5": "Frozen retrained KGAE monthly latents, k=5",
            "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10": "Frozen retrained KGAE monthly latents, k=10",
        }
    )
    return kgae[
        ["model_name", "model_family", "water_year", "y_obs", "y_pred", "alpha_selected", "predictor_set", "k"]
    ].sort_values(["model_name", "water_year"])


def load_existing_z1z2_predictions() -> tuple[pd.DataFrame, dict[str, float]]:
    pred = pd.read_csv(PACIFIC_PRED)
    pred = pred.loc[pred["model_name"] == "Z1_Z2"].copy()
    pred["model_name"] = "Z1_Z2_PACIFIC_LOD_RIDGE"
    pred["model_family"] = "PACIFIC_ONLY_BASELINE"
    pred["water_year"] = pred["heldout_wy"].astype(int)
    pred["y_obs"] = pred["obs_swe"].astype(float)
    pred["y_pred"] = pred["pred_swe"].astype(float)
    pred["alpha_selected"] = pred["selected_alpha"].astype(float)
    pred["predictor_set"] = "Z1,Z2 only"
    pred["k"] = np.nan
    pred = pred[
        ["model_name", "model_family", "water_year", "y_obs", "y_pred", "alpha_selected", "predictor_set", "k"]
    ].sort_values("water_year")
    metrics = {
        "model_name": "Z1_Z2_PACIFIC_LOD_RIDGE",
        "model_family": "PACIFIC_ONLY_BASELINE",
        "predictor_set": "Z1,Z2 only",
        "k": np.nan,
    }
    metrics.update(compute_prediction_metrics(pred))
    return pred, metrics


def build_metrics_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in pred_df.groupby("model_name", sort=False):
        meta = group.iloc[0][["model_name", "model_family", "predictor_set", "k"]].to_dict()
        meta.update(compute_prediction_metrics(group))
        rows.append(meta)
    out = pd.DataFrame(rows)
    out["order"] = out["model_name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def add_value_labels(ax: plt.Axes, values: np.ndarray) -> None:
    ymin, ymax = ax.get_ylim()
    pad = 0.02 * (ymax - ymin if ymax != ymin else 1.0)
    for i, val in enumerate(values):
        ax.text(i, val + pad, f"{val:.3f}", ha="center", va="bottom", fontsize=9)


def bar_plot(metrics_df: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path) -> None:
    work = metrics_df.copy()
    xs = np.arange(len(work))
    colors = [COLOR[name] for name in work["model_name"]]
    labels = [DISPLAY_NAME[name] for name in work["model_name"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    vals = work[value_col].to_numpy(dtype=float)
    ax.bar(xs, vals, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    add_value_labels(ax, vals)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def timeseries_plot(
    obs: pd.Series,
    benchmark: pd.DataFrame,
    pacific_best: pd.DataFrame,
    kgae_best: pd.DataFrame,
    metrics_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(obs.index, obs.values, color="black", linewidth=2.0, label="Observed SWE")
    for df in [benchmark, pacific_best, kgae_best]:
        name = df["model_name"].iloc[0]
        row = metrics_df.loc[metrics_df["model_name"] == name].iloc[0]
        ax.plot(
            df["water_year"],
            df["y_pred"],
            color=COLOR[name],
            linewidth=1.8,
            label=f"{DISPLAY_NAME[name]} (RMSE={row['RMSE']:.4f}, R^2={row['R2']:.3f})",
        )
    ax.set_xlabel("Water year")
    ax.set_ylabel("April 1 Sierra SWE")
    ax.set_title("Pacific-fair LOYO SWE comparison")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_plot(
    benchmark: pd.DataFrame,
    pacific_best: pd.DataFrame,
    kgae_best: pd.DataFrame,
    metrics_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    groups = [benchmark, pacific_best, kgae_best]
    all_obs = pd.concat([g["y_obs"] for g in groups], ignore_index=True)
    all_pred = pd.concat([g["y_pred"] for g in groups], ignore_index=True)
    lo = float(min(all_obs.min(), all_pred.min()))
    hi = float(max(all_obs.max(), all_pred.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    for df in groups:
        name = df["model_name"].iloc[0]
        row = metrics_df.loc[metrics_df["model_name"] == name].iloc[0]
        ax.scatter(
            df["y_obs"],
            df["y_pred"],
            color=COLOR[name],
            alpha=0.78,
            label=f"{DISPLAY_NAME[name]} (RMSE={row['RMSE']:.4f}, R^2={row['R2']:.3f})",
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed SWE")
    ax.set_ylabel("Predicted SWE")
    ax.set_title("Pacific-fair LOYO SWE scatter")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def patch_report(summary_body: str) -> None:
    source_report = FINAL_HOME_OUT / "REPORT.md"
    existing = source_report.read_text() if source_report.exists() else "# KGAE More-Latents Report\n"
    header = "## Pacific-only fair comparison"
    if header in existing:
        existing = existing.split(header)[0].rstrip() + "\n\n"
    REPORT_MD.write_text(existing + summary_body.rstrip() + "\n")


def format_compare(lhs_name: str, lhs: pd.Series, rhs_name: str, rhs: pd.Series) -> str:
    verdict = "beats" if lhs["RMSE"] < rhs["RMSE"] else "loses to"
    return (
        f"- {DISPLAY_NAME[lhs_name]} {verdict} {DISPLAY_NAME[rhs_name]}: "
        f"RMSE {lhs['RMSE']:.6f} vs {rhs['RMSE']:.6f}; "
        f"r {lhs['r']:.6f} vs {rhs['r']:.6f}; "
        f"sign accuracy {lhs['sign_accuracy']:.6f} vs {rhs['sign_accuracy']:.6f}."
    )


def main() -> None:
    print("Loading source artifacts...", flush=True)
    HOME_OUT.mkdir(parents=True, exist_ok=True)
    baseline_table = pd.read_csv(BASELINE_TABLE).sort_values("water_year").reset_index(drop=True)
    pacific_table = pd.read_csv(PACIFIC_TABLE).sort_values("water_year").reset_index(drop=True)

    baseline_pred = load_existing_baseline_predictions()
    kgae_pred = load_existing_kgae_predictions()
    z1z2_pred, _ = load_existing_z1z2_predictions()

    pc6_cols = [f"Pacific_PC6_{month}" for month in ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]]
    if any(col not in pacific_table.columns for col in pc6_cols):
        missing = [col for col in pc6_cols if col not in pacific_table.columns]
        raise ValueError(f"Missing Pacific PC6 columns: {missing}")
    pc6_df = pacific_table[["water_year", "obs_swe"] + pc6_cols].copy()
    print("Running pure Pacific_PC6 LOYO ridge...", flush=True)
    pc6_pred, _ = run_loyo_ridge(
        model_name="PACIFIC_PC6_RIDGE",
        model_family="PACIFIC_ONLY_BASELINE",
        predictor_set="Pacific_PC6 monthly Sep-Mar only",
        predictor_df=pc6_df,
        feature_cols=pc6_cols,
    )

    pred_fair = pd.concat([baseline_pred, z1z2_pred, pc6_pred, kgae_pred], ignore_index=True)
    print("Writing merged fair-comparison tables...", flush=True)
    pred_fair["order"] = pred_fair["model_name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    pred_fair = pred_fair.sort_values(["order", "water_year"]).drop(columns="order").reset_index(drop=True)
    metrics_fair = build_metrics_from_predictions(pred_fair)

    pred_fair.to_csv(HOME_OUT / "kgae_more_latents_swe_ridge_loyo_predictions_pacific_fair.csv", index=False)
    metrics_fair.to_csv(HOME_OUT / "kgae_more_latents_swe_ridge_loyo_metrics_pacific_fair.csv", index=False)

    benchmark = metrics_fair.loc[metrics_fair["model_name"] == "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC"].iloc[0]
    z1z2 = metrics_fair.loc[metrics_fair["model_name"] == "Z1_Z2_PACIFIC_LOD_RIDGE"].iloc[0]
    pc6 = metrics_fair.loc[metrics_fair["model_name"] == "PACIFIC_PC6_RIDGE"].iloc[0]
    k5 = metrics_fair.loc[metrics_fair["model_name"] == "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5"].iloc[0]
    k10 = metrics_fair.loc[metrics_fair["model_name"] == "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10"].iloc[0]

    pacific_only = metrics_fair.loc[
        metrics_fair["model_name"].isin(["Z1_Z2_PACIFIC_LOD_RIDGE", "PACIFIC_PC6_RIDGE"])
    ].sort_values("RMSE")
    best_pacific = pacific_only.iloc[0]
    best_kgae = metrics_fair.loc[
        metrics_fair["model_name"].isin(["KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5", "KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10"])
    ].sort_values("RMSE").iloc[0]

    best_models = pd.DataFrame(
        [
            {"model_role": "best_overall_benchmark", **benchmark.to_dict()},
            {"model_role": "best_pacific_only_baseline", **best_pacific.to_dict()},
            {"model_role": "best_kgae_larger_latent_model", **best_kgae.to_dict()},
            {"model_role": "z1_z2_pacific_baseline", **z1z2.to_dict()},
            {"model_role": "pacific_pc6_baseline", **pc6.to_dict()},
        ]
    )
    best_models.to_csv(HOME_OUT / "kgae_more_latents_best_models_pacific_fair.csv", index=False)

    summary_lines = [
        "## Pacific-only fair comparison",
        "",
        "- The 7-column baseline is the current best overall benchmark, but it is not Pacific-only.",
        "- KGAE is Pacific-only, so the fair comparison is against `Z1_Z2_PACIFIC_LOD_RIDGE` and `PACIFIC_PC6_RIDGE`.",
        "- `Pacific_PC6` is not part of the 7-column baseline. It comes from the separate Pacific PC/Nino predictor table and is used here only as a Pacific-side diagnostic/reference.",
        "- The five non-`Z1`/`Z2` columns in the 7-column baseline are Atlantic AMV/AMO/AQM-related predictors.",
        "",
        "Results:",
        format_compare("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5", k5, "Z1_Z2_PACIFIC_LOD_RIDGE", z1z2),
        format_compare("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10", k10, "Z1_Z2_PACIFIC_LOD_RIDGE", z1z2),
        format_compare("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k5", k5, "PACIFIC_PC6_RIDGE", pc6),
        format_compare("KGAE_MORE_LATENTS_MONTHLY_RIDGE_k10", k10, "PACIFIC_PC6_RIDGE", pc6),
        (
            f"- KGAE k=10 {'improves over' if k10['RMSE'] < k5['RMSE'] else 'does not improve over'} "
            f"k=5 in RMSE: {k10['RMSE']:.6f} vs {k5['RMSE']:.6f}."
        ),
        (
            f"- KGAE k=10 {'improves over' if k10['r'] > k5['r'] else 'does not improve over'} "
            f"k=5 in Pearson r: {k10['r']:.6f} vs {k5['r']:.6f}."
        ),
        (
            f"- KGAE k=10 {'improves over' if k10['sign_accuracy'] > k5['sign_accuracy'] else 'does not improve over'} "
            f"k=5 in sign accuracy: {k10['sign_accuracy']:.6f} vs {k5['sign_accuracy']:.6f}."
        ),
        (
            f"- Best Pacific-only baseline: {DISPLAY_NAME[best_pacific['model_name']]} "
            f"(RMSE {best_pacific['RMSE']:.6f}, r {best_pacific['r']:.6f})."
        ),
        (
            f"- Best KGAE model: {DISPLAY_NAME[best_kgae['model_name']]} "
            f"(RMSE {best_kgae['RMSE']:.6f}, r {best_kgae['r']:.6f})."
        ),
        (
            f"- RMSE gap to the full 7-column Pacific+Atlantic benchmark: "
            f"Z1/Z2 {z1z2['RMSE'] - benchmark['RMSE']:.6f}; "
            f"Pacific PC6 {pc6['RMSE'] - benchmark['RMSE']:.6f}; "
            f"best KGAE {best_kgae['RMSE'] - benchmark['RMSE']:.6f}."
        ),
        "",
        "Interpretation:",
    ]

    if best_kgae["RMSE"] < best_pacific["RMSE"]:
        summary_lines.append(
            "- KGAE is competitive as a Pacific-only representation, but it still cannot recover the Atlantic/AQM information present in the full 7-column baseline."
        )
    else:
        summary_lines.append(
            "- Increasing KGAE latent dimension improved PC6-like alignment, but it did not produce a Pacific-only representation as predictive as the existing Pacific-only baselines."
        )

    summary_text = "\n".join(summary_lines) + "\n"
    (HOME_OUT / "kgae_more_latents_pacific_fair_summary.md").write_text(summary_text)
    patch_report(summary_text)

    print("Rendering figures...", flush=True)
    obs = baseline_table.set_index("water_year")["obs_swe"]
    benchmark_pred = pred_fair.loc[pred_fair["model_name"] == "BASELINE_7COL_PACIFIC_PLUS_ATLANTIC"]
    pacific_best_pred = pred_fair.loc[pred_fair["model_name"] == best_pacific["model_name"]]
    kgae_best_pred = pred_fair.loc[pred_fair["model_name"] == best_kgae["model_name"]]

    bar_plot(
        metrics_fair,
        value_col="RMSE",
        title="Pacific-fair LOYO comparison: RMSE",
        ylabel="RMSE",
        path=HOME_OUT / "kgae_more_latents_pacific_fair_rmse.png",
    )
    bar_plot(
        metrics_fair,
        value_col="r",
        title="Pacific-fair LOYO comparison: Pearson r",
        ylabel="Pearson r",
        path=HOME_OUT / "kgae_more_latents_pacific_fair_r.png",
    )
    bar_plot(
        metrics_fair,
        value_col="sign_accuracy",
        title="Pacific-fair LOYO comparison: sign accuracy",
        ylabel="Sign accuracy",
        path=HOME_OUT / "kgae_more_latents_pacific_fair_sign_accuracy.png",
    )
    timeseries_plot(
        obs=obs,
        benchmark=benchmark_pred,
        pacific_best=pacific_best_pred,
        kgae_best=kgae_best_pred,
        metrics_df=metrics_fair,
        path=HOME_OUT / "kgae_more_latents_pacific_fair_timeseries.png",
    )
    scatter_plot(
        benchmark=benchmark_pred,
        pacific_best=pacific_best_pred,
        kgae_best=kgae_best_pred,
        metrics_df=metrics_fair,
        path=HOME_OUT / "kgae_more_latents_pacific_fair_scatter.png",
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
