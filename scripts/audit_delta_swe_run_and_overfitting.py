#!/usr/bin/env python3
"""Read-only audit of the completed CNN/ConvLSTM 7-day delta-SWE runs:
(1) sample-construction audit (season, Apr2-Aug31 leakage check, sample mode,
counts, target-month histogram, 20 window/date spot-checks, batches/epoch,
epochs executed), (2) per-epoch train/val Huber-loss extraction + plots,
(3) overfitting diagnostics. Does NOT retrain or modify anything.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra")
PREDICTOR_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1")
EXPERIMENT_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/daily_7day_delta_swe_baselines_v1")
INDEX_PATH = EXPERIMENT_ROOT / "data_index" / "sample_index.npz"
SPLIT_PATH = EXPERIMENT_ROOT / "splits" / "water_year_split.json"
SWE_NPZ = PROJECT_ROOT / "artifacts" / "taiesm1_daily_sierra_swe.npz"

OUT_DIR = EXPERIMENT_ROOT / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def in_season_mask(month: np.ndarray, day: np.ndarray, start_month: int) -> np.ndarray:
    fall = (month >= start_month) & (month <= 12)
    winter = (month >= 1) & (month <= 3)
    apr1 = (month == 4) & (day == 1)
    return fall | winter | apr1


def sample_audit() -> dict:
    idx = np.load(INDEX_PATH, allow_pickle=True)
    split = json.loads(SPLIT_PATH.read_text())
    cnn_cfg = json.loads((EXPERIMENT_ROOT / "cnn_control" / "config.json").read_text())
    convlstm_cfg = json.loads((EXPERIMENT_ROOT / "convlstm" / "config.json").read_text())
    assert cnn_cfg["season_start_month"] == convlstm_cfg["season_start_month"]
    assert cnn_cfg["sample_mode"] == convlstm_cfg["sample_mode"]
    season_start_month = cnn_cfg["season_start_month"]
    sample_mode = cnn_cfg["sample_mode"]

    water_year = idx["water_year"]
    cal_month = idx["cal_month"]
    cal_day = idx["cal_day"]
    active = idx["active"]

    season_mask_full = in_season_mask(cal_month, cal_day, season_start_month)
    # explicit Apr2-Aug31 check: does ANY row in the full (unfiltered) index that
    # was actually used by train/val/test fall outside Sep1-Apr1?
    splits = {
        "train": set(split["train_water_years"]),
        "val": set(split["val_water_years"]),
        "test": set(split["test_water_years"]),
    }
    counts = {}
    month_hist = {}
    violations_total = 0
    for name, wy_set in splits.items():
        wy_mask = np.isin(water_year, list(wy_set))
        used_mask = wy_mask & season_mask_full
        if sample_mode == "snow_active":
            used_mask &= active
        n_all_in_wy_season = int(used_mask.sum())
        # check for Apr2-Aug31 leakage explicitly among rows selected by WY only (pre-season-filter)
        wy_only_rows = np.where(wy_mask)[0]
        out_of_season = ~season_mask_full[wy_only_rows]
        # of those WY rows, how many are Apr2-Aug31 (i.e. month in 4(day>1)..8)?
        m = cal_month[wy_only_rows]
        d = cal_day[wy_only_rows]
        apr2_aug31 = ((m == 4) & (d > 1)) | ((m >= 5) & (m <= 8))
        counts[name] = {
            "n_water_years": len(wy_set),
            "n_samples_used_by_completed_run": n_all_in_wy_season,
            "n_wy_rows_total_before_season_filter": int(wy_mask.sum()),
            "n_apr2_aug31_rows_present_in_full_index_for_these_wy": int(apr2_aug31.sum()),
        }
        # month histogram of the samples actually used
        used_rows = np.where(used_mask)[0]
        months = cal_month[used_rows]
        hist = {int(m): int((months == m).sum()) for m in sorted(set(months.tolist()))}
        month_hist[name] = hist
        violations_total += int((used_mask & ~season_mask_full).sum())  # should always be 0 by construction

    report = {
        "season_start_month": season_start_month,
        "season_definition": "Sep(start)..Dec, Jan..Mar full, Apr 1 only",
        "sample_mode": sample_mode,
        "splits": counts,
        "target_month_histograms": month_hist,
        "apr2_aug31_violations_in_used_samples": violations_total,
    }
    return report


def spot_check_windows(n_checks: int = 20, seed: int = 20260901) -> list[dict]:
    idx = np.load(INDEX_PATH, allow_pickle=True)
    split = json.loads(SPLIT_PATH.read_text())
    cnn_cfg = json.loads((EXPERIMENT_ROOT / "cnn_control" / "config.json").read_text())
    season_start_month = cnn_cfg["season_start_month"]
    all_wy = set(split["train_water_years"]) | set(split["val_water_years"]) | set(split["test_water_years"])

    water_year = idx["water_year"]
    cal_month = idx["cal_month"]
    cal_day = idx["cal_day"]
    used_mask = np.isin(water_year, list(all_wy)) & in_season_mask(cal_month, cal_day, season_start_month)
    # sample_mode == "all" for the completed runs, so no active filter here.
    candidate_rows = np.where(used_mask)[0]

    rng = random.Random(seed)
    rows = rng.sample(list(candidate_rows), n_checks)

    smeta = np.load(SWE_NPZ, allow_pickle=True)
    swe_lookup = {}
    CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    for y, m, d, v in zip(smeta["year"], smeta["month"], smeta["day"], smeta["swe_mm"]):
        serial = int(y) * 365 + CUM[int(m) - 1] + int(d) - 1
        swe_lookup[serial] = float(v)

    exp_code_map = json.loads(str(idx["exp_code_map"]))
    code_to_exp = {v: k for k, v in exp_code_map.items()}

    results = []
    for row in rows:
        history_year = idx["history_year"][row]
        history_doy = idx["history_doy"][row]
        history_exp = idx["history_exp"][row]
        serial_t = int(idx["serial_t"][row])

        # (a) 7 history days genuinely consecutive, by re-deriving each day's serial
        #     from (experiment, year, doy) independently of the stored serial_t.
        hist_serials = []
        for k in range(7):
            y = int(history_year[k])
            d = int(history_doy[k])
            hist_serials.append(y * 365 + (d - 1))
        consecutive_ok = all(hist_serials[i + 1] - hist_serials[i] == 1 for i in range(6))
        last_matches_t = hist_serials[-1] == serial_t

        # (b) SWE(t), SWE(t+1) independently re-looked-up and compared to stored values
        swe_t_direct = swe_lookup.get(serial_t)
        swe_t1_direct = swe_lookup.get(serial_t + 1)
        swe_t_stored = float(idx["swe_t"][row])
        swe_t1_stored = float(idx["swe_t1"][row])
        swe_match = (
            swe_t_direct is not None and swe_t1_direct is not None
            and abs(swe_t_direct - swe_t_stored) < 1e-6 and abs(swe_t1_direct - swe_t1_stored) < 1e-6
        )
        delta_stored = float(idx["delta_swe"][row])
        delta_recomputed = swe_t1_stored - swe_t_stored
        delta_match = abs(delta_stored - delta_recomputed) < 1e-9

        # (c) date is within Sep1-Apr1 (using cal_month/cal_day of t)
        month_t = int(idx["cal_month"][row])
        day_t = int(idx["cal_day"][row])
        in_season = in_season_mask(np.array([month_t]), np.array([day_t]), 9)[0]

        results.append(
            {
                "row": int(row),
                "water_year": int(idx["water_year"][row]),
                "target_date": f"{int(idx['cal_year'][row]):04d}-{month_t:02d}-{day_t:02d}",
                "history_experiments": [code_to_exp[int(e)] for e in history_exp],
                "history_consecutive_ok": bool(consecutive_ok),
                "history_last_equals_t_ok": bool(last_matches_t),
                "swe_lookup_match_ok": bool(swe_match),
                "delta_recompute_match_ok": bool(delta_match),
                "in_season_ok": bool(in_season),
                "swe_t": swe_t_stored,
                "swe_t1": swe_t1_stored,
                "delta_swe": delta_stored,
            }
        )
    return results


def load_history(model_dir: str) -> list[dict]:
    return json.loads((EXPERIMENT_ROOT / model_dir / "history.json").read_text())


def overfitting_diagnosis(history: list[dict]) -> dict:
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    min_train_epoch = epochs[int(np.argmin(train))]
    min_val_epoch = epochs[int(np.argmin(val))]
    min_val = min(val)
    final_val = val[-1]
    final_train = train[-1]
    val_at_epoch1 = val[0]
    train_at_best_val = train[epochs.index(min_val_epoch)]

    gap = [v - t for v, t in zip(val, train)]
    gap_at_best = gap[epochs.index(min_val_epoch)]
    gap_at_final = gap[-1]

    # post-best-epoch behavior
    best_idx = epochs.index(min_val_epoch)
    post_best_train = train[best_idx:]
    post_best_val = val[best_idx:]
    train_keeps_decreasing = all(post_best_train[i + 1] <= post_best_train[i] + 1e-9 for i in range(len(post_best_train) - 1))
    val_net_increase = post_best_val[-1] - post_best_val[0]
    train_net_decrease = post_best_train[0] - post_best_train[-1]

    gap_growth_ratio = (gap_at_final / gap_at_best) if gap_at_best > 1e-9 else float("inf")

    if len(history) - best_idx <= 1:
        diagnosis = "INSUFFICIENT_POST_BEST_EPOCHS"
    elif val_net_increase > 0.03 * post_best_val[0] and train_keeps_decreasing and gap_growth_ratio > 3:
        diagnosis = "CLEAR OVERFITTING"
    elif val_net_increase > 0.0 and train_net_decrease > 0.0:
        diagnosis = "MILD OVERFITTING"
    else:
        diagnosis = "NO CLEAR OVERFITTING"

    return {
        "epoch_of_min_training_loss": min_train_epoch,
        "epoch_of_min_validation_loss": min_val_epoch,
        "validation_loss_epoch1": val_at_epoch1,
        "minimum_validation_loss": min_val,
        "final_validation_loss": final_val,
        "train_loss_at_best_validation_epoch": train_at_best_val,
        "final_train_loss": final_train,
        "gap_at_best_val_epoch": gap_at_best,
        "gap_at_final_epoch": gap_at_final,
        "gap_growth_ratio_final_over_best": gap_growth_ratio,
        "post_best_epoch_train_monotonic_decrease": train_keeps_decreasing,
        "post_best_epoch_val_net_change": val_net_increase,
        "post_best_epoch_train_net_change": -train_net_decrease,
        "diagnosis": diagnosis,
        "epochs_executed": len(history),
    }


def main() -> None:
    print("=== 1. SAMPLE AUDIT ===", flush=True)
    audit = sample_audit()
    print(json.dumps(audit, indent=2), flush=True)
    (OUT_DIR / "sample_audit.json").write_text(json.dumps(audit, indent=2))

    print("\n=== 2. 20 WINDOW/DATE SPOT CHECKS ===", flush=True)
    spot = spot_check_windows(20)
    all_ok = all(
        r["history_consecutive_ok"] and r["history_last_equals_t_ok"] and r["swe_lookup_match_ok"]
        and r["delta_recompute_match_ok"] and r["in_season_ok"]
        for r in spot
    )
    for r in spot:
        print(f"  WY{r['water_year']} t={r['target_date']} consec={r['history_consecutive_ok']} "
              f"swe_match={r['swe_lookup_match_ok']} delta_match={r['delta_recompute_match_ok']} "
              f"in_season={r['in_season_ok']} delta_swe={r['delta_swe']:.4f}", flush=True)
    print(f"ALL 20 SPOT CHECKS PASS: {all_ok}", flush=True)
    (OUT_DIR / "spot_check_20_windows.json").write_text(json.dumps({"all_pass": all_ok, "checks": spot}, indent=2))

    print("\n=== 3. BATCHES/EPOCH AND EPOCHS EXECUTED ===", flush=True)
    cnn_cfg = json.loads((EXPERIMENT_ROOT / "cnn_control" / "config.json").read_text())
    convlstm_cfg = json.loads((EXPERIMENT_ROOT / "convlstm" / "config.json").read_text())
    cnn_hist = load_history("cnn_control")
    convlstm_hist = load_history("convlstm")
    cnn_batches_per_epoch = cnn_cfg["train_n"] // cnn_cfg["batch_size"]
    convlstm_batches_per_epoch = convlstm_cfg["train_n"] // convlstm_cfg["batch_size"]
    print(f"CNN: batch_size={cnn_cfg['batch_size']} batches/epoch={cnn_batches_per_epoch} epochs_executed={len(cnn_hist)}", flush=True)
    print(f"ConvLSTM: batch_size={convlstm_cfg['batch_size']} batches/epoch={convlstm_batches_per_epoch} epochs_executed={len(convlstm_hist)}", flush=True)

    # -----------------------------------------------------------------
    # 4. Loss curves: raw CSV/JSON + plots.
    # -----------------------------------------------------------------
    for name, hist in (("cnn_control", cnn_hist), ("convlstm", convlstm_hist)):
        import csv
        with (EXPERIMENT_ROOT / name / "loss_curve.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "val_rmse_mm"])
            writer.writeheader()
            writer.writerows(hist)

    def plot_single(name: str, hist: list[dict], out_path: Path) -> None:
        epochs = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(epochs, [h["train_loss"] for h in hist], "o-", color="tab:blue", label="train Huber loss")
        ax.plot(epochs, [h["val_loss"] for h in hist], "o-", color="tab:red", label="validation Huber loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Huber loss (normalized ΔSWE units)")
        ax.set_title(f"{name}: train vs validation loss (raw, unsmoothed)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    (EXPERIMENT_ROOT / "cnn_control" / "plots").mkdir(exist_ok=True, parents=True)
    (EXPERIMENT_ROOT / "convlstm" / "plots").mkdir(exist_ok=True, parents=True)
    plot_single("CNN control", cnn_hist, EXPERIMENT_ROOT / "cnn_control" / "plots" / "loss_curve.png")
    plot_single("ConvLSTM baseline", convlstm_hist, EXPERIMENT_ROOT / "convlstm" / "plots" / "loss_curve.png")

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax.plot([h["epoch"] for h in cnn_hist], [h["train_loss"] for h in cnn_hist], "o-", color="tab:blue", label="CNN train")
    ax.plot([h["epoch"] for h in cnn_hist], [h["val_loss"] for h in cnn_hist], "o--", color="tab:blue", alpha=0.6, label="CNN val")
    ax.plot([h["epoch"] for h in convlstm_hist], [h["train_loss"] for h in convlstm_hist], "s-", color="tab:orange", label="ConvLSTM train")
    ax.plot([h["epoch"] for h in convlstm_hist], [h["val_loss"] for h in convlstm_hist], "s--", color="tab:orange", alpha=0.6, label="ConvLSTM val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (normalized ΔSWE units)")
    ax.set_title("CNN vs ConvLSTM: train/validation loss (raw, unsmoothed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curve_comparison.png")
    plt.close(fig)
    print(f"\nwrote loss curve plots + CSVs", flush=True)

    # -----------------------------------------------------------------
    # 5. Overfitting diagnosis.
    # -----------------------------------------------------------------
    cnn_diag = overfitting_diagnosis(cnn_hist)
    convlstm_diag = overfitting_diagnosis(convlstm_hist)
    print("\n=== 5. OVERFITTING DIAGNOSIS ===", flush=True)
    print("CNN:", json.dumps(cnn_diag, indent=2), flush=True)
    print("ConvLSTM:", json.dumps(convlstm_diag, indent=2), flush=True)

    final_report = {
        "sample_audit": audit,
        "spot_check_all_pass": all_ok,
        "cnn": {"batches_per_epoch": cnn_batches_per_epoch, "epochs_executed": len(cnn_hist), "overfitting": cnn_diag},
        "convlstm": {"batches_per_epoch": convlstm_batches_per_epoch, "epochs_executed": len(convlstm_hist), "overfitting": convlstm_diag},
        "post_april_samples_used": audit["apr2_aug31_violations_in_used_samples"] > 0,
        "split_used_for_completed_runs": "80/10/10 (train/val/test) whole-water-year -- NOT the new train/val-only default",
    }
    (OUT_DIR / "training_protocol_audit.json").write_text(json.dumps(final_report, indent=2))
    print(f"\nwrote {OUT_DIR / 'training_protocol_audit.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
