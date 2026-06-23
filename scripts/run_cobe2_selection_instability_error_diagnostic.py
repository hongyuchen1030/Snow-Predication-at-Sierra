#!/usr/bin/env python3
"""
Diagnose whether poor nested COBE2 LOYO prediction is associated with
full-sample vs foldwise selected-mode mismatch.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL37_SUMMARY_JSON = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json"
)
NESTED_SELECTED_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "nested_loyo_mode_stability_audit"
    / "nested_loyo_selected_modes_by_fold.csv"
)
PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "loyo_mode_subset_diagnostic"
    / "loyo_mode_subset_predictions.csv"
)
METRICS_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "loyo_mode_subset_diagnostic"
    / "loyo_mode_subset_metrics.csv"
)
BETA_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "loyo_mode_subset_diagnostic"
    / "loyo_mode_subset_beta_by_fold.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "selection_instability_error_diagnostic"
)
MODEL_SPECS = [
    ("M1_only", [1]),
    ("M2_only", [2]),
    ("M3_only", [3]),
    ("M1_M2", [1, 2]),
    ("M1_M2_M3", [1, 2, 3]),
]


def ensure_runtime_on_compute_node() -> None:
    hostname = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") or "nid" not in hostname:
        raise RuntimeError("Run this script inside an interactive compute-node allocation.")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def to_bool_or_nan(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return 1.0 if bool(value) else 0.0


def sign_correct_or_nan(obs: float, pred: float) -> float:
    if not math.isfinite(obs) or not math.isfinite(pred) or obs == 0.0 or pred == 0.0:
        return float("nan")
    return 1.0 if math.copysign(1.0, obs) == math.copysign(1.0, pred) else 0.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(a)))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    xx = x[finite]
    yy = y[finite]
    if np.std(xx, ddof=1) == 0.0 or np.std(yy, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def rankdata_average(values: np.ndarray) -> np.ndarray:
    ranks = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return ranks
    vals = values[finite]
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    sorted_ranks = np.empty(sorted_vals.shape, dtype=np.float64)
    start = 0
    while start < sorted_vals.size:
        end = start + 1
        while end < sorted_vals.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        sorted_ranks[start:end] = avg_rank
        start = end
    back = np.empty(order.shape, dtype=np.float64)
    back[order] = sorted_ranks
    ranks[finite] = back
    return ranks


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    return corrcoef_safe(rankdata_average(x), rankdata_average(y))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 3:
        return float("nan")
    y_mean = float(np.mean(y_true))
    sst = float(np.sum((y_true - y_mean) ** 2))
    if sst == 0.0:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst


def load_full37_modes() -> dict[int, dict[str, object]]:
    payload = json.loads(FULL37_SUMMARY_JSON.read_text(encoding="utf-8"))
    selected = [row for row in payload["lod_rows"] if row.get("selected")]
    out: dict[int, dict[str, object]] = {}
    for row in selected:
        mode = int(row["mode_number"])
        if mode > 3:
            continue
        out[mode] = {
            "mode": mode,
            "lag_month": str(row["lag_month"]),
            "lat": float(row["latitude"]),
            "lon": float(row.get("longitude_0_360", row.get("longitude"))),
            "corr": float(row.get("selected_residual_correlation", row.get("corr_with_residual"))),
            "delta_R2": float(row["delta_r2"]),
            "cumulative_R2": float(row["cumulative_r2"]),
        }
    if sorted(out) != [1, 2, 3]:
        raise ValueError("Full-sample summary is missing one or more of modes 1-3.")
    return out


def load_nested_by_fold() -> dict[tuple[int, int], dict[str, object]]:
    out: dict[tuple[int, int], dict[str, object]] = {}
    for row in read_csv(NESTED_SELECTED_CSV):
        key = (int(row["heldout_wy"]), int(row["mode"]))
        out[key] = {
            "heldout_wy": int(row["heldout_wy"]),
            "mode": int(row["mode"]),
            "lag_month": row["lag_month"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "selected_corr": float(row["selected_corr"]),
            "delta_R2": float(row["delta_R2"]),
            "cumulative_R2": float(row["cumulative_R2"]),
            "beta": float(row["beta"]),
            "pred_swe": float(row["pred_swe"]),
            "obs_swe": float(row["obs_swe"]),
        }
    return out


def load_predictions() -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(PREDICTIONS_CSV):
        key = (row["model_name"], int(row["heldout_wy"]))
        out[key] = {
            "obs_swe": float(row["obs_swe"]),
            "pred_swe": float(row["pred_swe"]),
            "error": float(row["error"]),
        }
    return out


def load_metrics_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(METRICS_CSV):
        rows.append(
            {
                "model_name": row["model_name"],
                "num_modes": int(row["num_modes"]),
                "r": float(row["r"]),
                "R2": float(row["R2"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
            }
        )
    return rows


def load_beta_rows() -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in read_csv(BETA_CSV):
        keys.add((row["model_name"], int(row["heldout_wy"])))
    return keys


def format_group_value(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def conditional_metrics(rows: list[dict[str, object]], group_field: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        raw = row[group_field]
        label = format_group_value(raw)
        if label == "NaN":
            continue
        grouped.setdefault(label, []).append(row)

    metric_rows: list[dict[str, object]] = []
    for label, members in sorted(grouped.items(), key=lambda item: item[0]):
        obs = np.asarray([float(item["obs_swe"]) for item in members], dtype=np.float64)
        pred = np.asarray([float(item["pred_swe"]) for item in members], dtype=np.float64)
        err = np.asarray([float(item["error"]) for item in members], dtype=np.float64)
        abs_err = np.asarray([float(item["abs_error"]) for item in members], dtype=np.float64)
        sign_correct = np.asarray([float(item["sign_correct"]) for item in members], dtype=np.float64)
        valid_sign = np.isfinite(sign_correct)
        n = len(members)
        metric_rows.append(
            {
                "group_value": label,
                "n_folds": n,
                "r": corrcoef_safe(obs, pred) if n >= 3 else float("nan"),
                "R2": r2_safe(obs, pred) if n >= 3 else float("nan"),
                "RMSE": rmse(obs, pred),
                "MAE": mae(obs, pred),
                "mean_abs_error": float(np.mean(abs_err)),
                "median_abs_error": float(median(abs_err.tolist())),
                "sign_accuracy": float(np.mean(sign_correct[valid_sign])) if np.any(valid_sign) else float("nan"),
                "mean_error": float(np.mean(err)),
            }
        )
    return metric_rows


def build_join_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[int, dict[str, object]]]:
    full37 = load_full37_modes()
    nested = load_nested_by_fold()
    predictions = load_predictions()
    beta_keys = load_beta_rows()

    rows: list[dict[str, object]] = []
    per_model: dict[str, list[dict[str, object]]] = {name: [] for name, _ in MODEL_SPECS}
    heldout_years = sorted({year for _, year in predictions})
    for model_name, used_modes in MODEL_SPECS:
        for heldout_wy in heldout_years:
            key = (model_name, heldout_wy)
            if key not in predictions:
                raise ValueError(f"Missing prediction row for {key}.")
            if key not in beta_keys:
                raise ValueError(f"Missing beta row for {key}.")

            pred_row = predictions[key]
            exact_by_mode: dict[int, float] = {}
            dist_by_mode: dict[int, float] = {}
            same_month_by_mode: dict[int, float] = {}
            for mode in (1, 2, 3):
                if mode not in used_modes:
                    exact_by_mode[mode] = float("nan")
                    dist_by_mode[mode] = float("nan")
                    same_month_by_mode[mode] = float("nan")
                    continue
                fold_mode = nested[(heldout_wy, mode)]
                ref_mode = full37[mode]
                exact_match = (
                    fold_mode["lag_month"] == ref_mode["lag_month"]
                    and fold_mode["lat"] == ref_mode["lat"]
                    and fold_mode["lon"] == ref_mode["lon"]
                )
                exact_by_mode[mode] = to_bool_or_nan(exact_match)
                same_month_by_mode[mode] = to_bool_or_nan(fold_mode["lag_month"] == ref_mode["lag_month"])
                dist_by_mode[mode] = haversine_km(
                    float(fold_mode["lat"]),
                    float(fold_mode["lon"]),
                    float(ref_mode["lat"]),
                    float(ref_mode["lon"]),
                )

            used_exact = np.asarray([exact_by_mode[m] for m in used_modes], dtype=np.float64)
            used_dist = np.asarray([dist_by_mode[m] for m in used_modes], dtype=np.float64)
            num_modes_used = len(used_modes)
            num_modes_matched = int(np.nansum(used_exact))
            all_modes_match = num_modes_matched == num_modes_used
            any_mode_match = num_modes_matched > 0
            if model_name == "M1_M2":
                leading_mode_match: float = exact_by_mode[1]
            elif model_name == "M1_M2_M3":
                leading_mode_match = exact_by_mode[1]
            elif model_name == "M1_only":
                leading_mode_match = exact_by_mode[1]
            else:
                leading_mode_match = float("nan")

            row = {
                "model_name": model_name,
                "heldout_wy": heldout_wy,
                "obs_swe": pred_row["obs_swe"],
                "pred_swe": pred_row["pred_swe"],
                "error": pred_row["error"],
                "abs_error": abs(pred_row["error"]),
                "sign_correct": sign_correct_or_nan(pred_row["obs_swe"], pred_row["pred_swe"]),
                "num_modes_used": num_modes_used,
                "mode_1_exact_match": exact_by_mode[1],
                "mode_2_exact_match": exact_by_mode[2],
                "mode_3_exact_match": exact_by_mode[3],
                "num_modes_matched": num_modes_matched,
                "all_modes_match": to_bool_or_nan(all_modes_match),
                "any_mode_match": to_bool_or_nan(any_mode_match),
                "leading_mode_match": leading_mode_match,
                "mode_1_distance_km_to_full": dist_by_mode[1],
                "mode_2_distance_km_to_full": dist_by_mode[2],
                "mode_3_distance_km_to_full": dist_by_mode[3],
                "mode_1_same_month_as_full": same_month_by_mode[1],
                "mode_2_same_month_as_full": same_month_by_mode[2],
                "mode_3_same_month_as_full": same_month_by_mode[3],
                "mean_distance_to_full_km": float(np.nanmean(used_dist)),
                "max_distance_to_full_km": float(np.nanmax(used_dist)),
            }
            rows.append(row)
            per_model[model_name].append(row)
    return rows, per_model, full37


def build_conditional_metrics(per_model: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model_name, rows in per_model.items():
        for group_field in ("all_modes_match", "any_mode_match", "leading_mode_match", "num_modes_matched"):
            for metric_row in conditional_metrics(rows, group_field):
                out.append({"model_name": model_name, "grouping_variable": group_field, **metric_row})
    return out


def build_correlation_rows(per_model: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model_name, rows in per_model.items():
        abs_error = np.asarray([float(row["abs_error"]) for row in rows], dtype=np.float64)
        for predictor_field in ("mean_distance_to_full_km", "max_distance_to_full_km", "num_modes_matched"):
            predictor = np.asarray([float(row[predictor_field]) for row in rows], dtype=np.float64)
            out.append(
                {
                    "model_name": model_name,
                    "predictor_field": predictor_field,
                    "n_folds": int(np.count_nonzero(np.isfinite(abs_error) & np.isfinite(predictor))),
                    "pearson_r": corrcoef_safe(abs_error, predictor),
                    "spearman_rho": spearman_safe(abs_error, predictor),
                }
            )
    return out


def plot_boxplots(path: Path, per_model: dict[str, list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(len(MODEL_SPECS), 1, figsize=(11.0, 15.0), constrained_layout=True)
    for ax, (model_name, used_modes) in zip(axes, MODEL_SPECS):
        rows = per_model[model_name]
        groups: list[tuple[str, list[float]]] = []
        if len(used_modes) == 1:
            matched = [float(row["abs_error"]) for row in rows if float(row["all_modes_match"]) == 1.0]
            off = [float(row["abs_error"]) for row in rows if float(row["all_modes_match"]) == 0.0]
            if matched:
                groups.append(("match", matched))
            if off:
                groups.append(("off-mode", off))
        else:
            for k in range(len(used_modes) + 1):
                vals = [float(row["abs_error"]) for row in rows if int(row["num_modes_matched"]) == k]
                if vals:
                    groups.append((f"{k} matched", vals))
        ax.boxplot([values for _, values in groups], tick_labels=[label for label, _ in groups], patch_artist=True)
        ax.set_title(model_name)
        ax.set_ylabel("Absolute error (m)")
        ax.grid(True, axis="y", linewidth=0.25, color="0.85")
    fig.suptitle("Absolute prediction error grouped by selection-match status", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_distance_scatter(path: Path, per_model: dict[str, list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(len(MODEL_SPECS), 1, figsize=(10.5, 15.0), constrained_layout=True)
    year_min = min(int(min(row["heldout_wy"] for row in rows)) for rows in per_model.values())
    year_max = max(int(max(row["heldout_wy"] for row in rows)) for rows in per_model.values())
    norm = plt.Normalize(year_min, year_max)
    for ax, (model_name, _) in zip(axes, MODEL_SPECS):
        rows = per_model[model_name]
        x = np.asarray([float(row["mean_distance_to_full_km"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["abs_error"]) for row in rows], dtype=np.float64)
        years = np.asarray([int(row["heldout_wy"]) for row in rows], dtype=np.int32)
        scatter = ax.scatter(x, y, c=years, cmap="viridis", norm=norm, s=32, edgecolors="none")
        ax.set_title(model_name)
        ax.set_xlabel("Mean distance to full-37 selected mode(s) (km)")
        ax.set_ylabel("Absolute error (m)")
        ax.grid(True, linewidth=0.25, color="0.85")
    fig.colorbar(scatter, ax=axes, label="Held-out water year")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_timeline(path: Path, per_model: dict[str, list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(len(MODEL_SPECS), 1, figsize=(12.0, 15.0), constrained_layout=True)
    for ax, (model_name, used_modes) in zip(axes, MODEL_SPECS):
        rows = sorted(per_model[model_name], key=lambda item: int(item["heldout_wy"]))
        years = np.asarray([int(row["heldout_wy"]) for row in rows], dtype=np.int32)
        abs_error = np.asarray([float(row["abs_error"]) for row in rows], dtype=np.float64)
        ax.plot(years, abs_error, color="0.65", linewidth=1.0)
        if len(used_modes) == 1:
            colors = ["tab:green" if float(row["all_modes_match"]) == 1.0 else "tab:red" for row in rows]
            ax.scatter(years, abs_error, c=colors, s=30)
        else:
            matched = np.asarray([int(row["num_modes_matched"]) for row in rows], dtype=np.int32)
            scatter = ax.scatter(years, abs_error, c=matched, cmap="plasma", vmin=0, vmax=len(used_modes), s=30)
            fig.colorbar(scatter, ax=ax, label="num_modes_matched")
        ax.set_title(model_name)
        ax.set_ylabel("Absolute error (m)")
        ax.grid(True, linewidth=0.25, color="0.85")
    axes[-1].set_xlabel("Held-out water year")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_match_effects(
    conditional_rows: list[dict[str, object]], overall_metrics: list[dict[str, object]]
) -> dict[str, object]:
    overall_by_model = {str(row["model_name"]): row for row in overall_metrics}
    details: dict[str, object] = {}
    better_count = 0
    poor_even_when_matched_count = 0
    mixed_count = 0
    small_group_count = 0

    def pick_row(model_name: str, grouping_variable: str, accepted_values: set[str]) -> dict[str, object] | None:
        for row in conditional_rows:
            if row["model_name"] != model_name or row["grouping_variable"] != grouping_variable:
                continue
            if str(row["group_value"]) in accepted_values:
                return row
        return None

    for model_name, used_modes in MODEL_SPECS:
        overall = overall_by_model[model_name]
        true_row = pick_row(model_name, "all_modes_match", {"1", "True"})
        false_row = pick_row(model_name, "all_modes_match", {"0", "False"})
        matched_ref = true_row or pick_row(model_name, "num_modes_matched", {str(len(used_modes))})

        if matched_ref is None or false_row is None:
            label = "mixed_or_missing_groups"
            mixed_count += 1
        else:
            matched_n = int(matched_ref["n_folds"])
            off_n = int(false_row["n_folds"])
            rmse_better = float(matched_ref["RMSE"]) < float(false_row["RMSE"])
            mae_better = float(matched_ref["MAE"]) < float(false_row["MAE"])
            sign_better = (
                math.isnan(float(matched_ref["sign_accuracy"]))
                or math.isnan(float(false_row["sign_accuracy"]))
                or float(matched_ref["sign_accuracy"]) >= float(false_row["sign_accuracy"])
            )
            matched_still_poor = (
                float(matched_ref["RMSE"]) >= 0.85 * float(overall["RMSE"])
                or (math.isfinite(float(matched_ref["sign_accuracy"])) and float(matched_ref["sign_accuracy"]) <= 0.6)
            )
            if matched_n < 5 or off_n < 5:
                label = "small_groups_but_matched_better" if (rmse_better and mae_better and sign_better) else "small_groups_mixed"
                small_group_count += 1
            elif rmse_better and mae_better and sign_better and not matched_still_poor:
                label = "matched_clearly_better"
                better_count += 1
            elif rmse_better and mae_better and sign_better:
                label = "matched_better_but_still_not_reliably_good"
                poor_even_when_matched_count += 1
            elif matched_still_poor:
                label = "matched_still_poor"
                poor_even_when_matched_count += 1
            else:
                label = "mixed"
                mixed_count += 1
        details[model_name] = {
            "assessment": label,
            "matched_reference": matched_ref,
            "offmode_reference": false_row,
        }

    if better_count >= 3 and poor_even_when_matched_count == 0 and small_group_count <= 1:
        conclusion = (
            "Prediction failure is strongly associated with feature-selection instability: "
            "folds that rediscover the full-37 selected modes predict better than folds that jump to off-mode SST locations/months."
        )
    elif poor_even_when_matched_count >= 3 and better_count <= 1:
        conclusion = (
            "Prediction failure is not mainly explained by exact feature-selection instability: "
            "even folds that rediscover the full-37 selected modes do not predict reliably."
        )
    else:
        conclusion = (
            "Feature-selection instability explains part of the failure for some mode subsets, "
            "but not the full failure. Inspect mode-specific results."
        )
    return {
        "per_model_assessment": details,
        "better_count": better_count,
        "poor_even_when_matched_count": poor_even_when_matched_count,
        "mixed_count": mixed_count,
        "small_group_count": small_group_count,
        "conclusion": conclusion,
    }


def main() -> None:
    ensure_runtime_on_compute_node()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    joined_rows, per_model, full37 = build_join_rows()
    overall_metrics = load_metrics_rows()
    conditional_rows = build_conditional_metrics(per_model)
    corr_rows = build_correlation_rows(per_model)

    match_csv = OUTPUT_ROOT / "selection_match_by_fold.csv"
    conditional_csv = OUTPUT_ROOT / "selection_match_conditional_metrics.csv"
    corr_csv = OUTPUT_ROOT / "selection_distance_error_correlations.csv"
    summary_json = OUTPUT_ROOT / "selection_match_diagnostic_summary.json"
    boxplot_png = OUTPUT_ROOT / "selection_match_error_boxplots.png"
    scatter_png = OUTPUT_ROOT / "selection_distance_vs_abs_error.png"
    timeline_png = OUTPUT_ROOT / "selection_match_timeline.png"

    write_csv(
        match_csv,
        [
            "model_name",
            "heldout_wy",
            "obs_swe",
            "pred_swe",
            "error",
            "abs_error",
            "sign_correct",
            "num_modes_used",
            "mode_1_exact_match",
            "mode_2_exact_match",
            "mode_3_exact_match",
            "num_modes_matched",
            "all_modes_match",
            "any_mode_match",
            "leading_mode_match",
            "mode_1_distance_km_to_full",
            "mode_2_distance_km_to_full",
            "mode_3_distance_km_to_full",
            "mode_1_same_month_as_full",
            "mode_2_same_month_as_full",
            "mode_3_same_month_as_full",
            "mean_distance_to_full_km",
            "max_distance_to_full_km",
        ],
        joined_rows,
    )
    write_csv(
        conditional_csv,
        [
            "model_name",
            "grouping_variable",
            "group_value",
            "n_folds",
            "r",
            "R2",
            "RMSE",
            "MAE",
            "mean_abs_error",
            "median_abs_error",
            "sign_accuracy",
            "mean_error",
        ],
        conditional_rows,
    )
    write_csv(
        corr_csv,
        ["model_name", "predictor_field", "n_folds", "pearson_r", "spearman_rho"],
        corr_rows,
    )

    plot_boxplots(boxplot_png, per_model)
    plot_distance_scatter(scatter_png, per_model)
    plot_timeline(timeline_png, per_model)

    interpretation = summarize_match_effects(conditional_rows, overall_metrics)
    summary = {
        "input_files": {
            "full37_summary_json": str(FULL37_SUMMARY_JSON),
            "nested_selected_modes_csv": str(NESTED_SELECTED_CSV),
            "predictions_csv": str(PREDICTIONS_CSV),
            "metrics_csv": str(METRICS_CSV),
            "beta_csv": str(BETA_CSV),
        },
        "full37_reference_modes": full37,
        "overall_metrics": overall_metrics,
        "row_count": len(joined_rows),
        "conditional_metric_row_count": len(conditional_rows),
        "distance_correlation_row_count": len(corr_rows),
        "interpretation": interpretation,
        "outputs": {
            "selection_match_by_fold_csv": str(match_csv),
            "selection_match_conditional_metrics_csv": str(conditional_csv),
            "selection_distance_error_correlations_csv": str(corr_csv),
            "selection_match_diagnostic_summary_json": str(summary_json),
            "selection_match_error_boxplots_png": str(boxplot_png),
            "selection_distance_vs_abs_error_png": str(scatter_png),
            "selection_match_timeline_png": str(timeline_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Joined table: {match_csv}")
    print(f"Conditional metrics: {conditional_csv}")
    print(f"Distance correlations: {corr_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Figures: {boxplot_png}, {scatter_png}, {timeline_png}")
    print(f"Conclusion: {interpretation['conclusion']}")


if __name__ == "__main__":
    main()
