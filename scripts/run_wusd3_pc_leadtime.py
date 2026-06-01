import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.data_wusd3 import (
    DEFAULT_WUSD3_DATASET_ID,
    DEFAULT_WUSD3_DOMAIN,
    Wusd3Dataset,
    default_wusd3_dataset,
    discover_wusd3_dataset_ids,
    discover_wusd3_water_years,
    file_year_for_date,
    inspect_wusd3_file,
    load_domain_series,
    load_snapshot,
    variable_path_for_file_year,
)
from snow_ml.pc_baseline import (
    compute_metrics,
    predictor_cutoff_date,
    save_aligned_dataset_csv,
    save_metrics,
    save_predictions_csv,
    save_targets_csv,
)


TARGET_MONTH_DAY = "04-01"
PCA_COMPONENTS = 5


@dataclass(frozen=True)
class Wusd3TargetPca:
    years: List[int]
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    mean_vector: np.ndarray
    scale_vector: np.ndarray
    valid_cell_mask: np.ndarray
    grid_shape: Tuple[int, int]
    metadata: Dict[str, Any]


def parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {text!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WUS-D3 PCA + lead-time baseline on one historical D02 dataset."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_WUSD3_DATASET_ID,
        help="WUS-D3 daily dataset id under the historical daily root.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_WUSD3_DOMAIN,
        help="WUS-D3 postprocess domain, default d02.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Optional water years to use. If omitted, use all discovered WUS-D3 water years.",
    )
    parser.add_argument(
        "--dec31-mode",
        choices=("pcs", "stats"),
        default="pcs",
        help="Use Dec 31 SWE projected into April 1 PCA space, or simple domain mean/std.",
    )
    parser.add_argument(
        "--include-sst",
        type=parse_bool,
        default=False,
        help="Reserved placeholder. Current WUS-D3 baseline skips SST unless explicitly added later.",
    )
    parser.add_argument(
        "--lead-months-list",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Lead months to run, typically 1 2 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "pc_leadtime_wusd3",
        help="Directory for UCLA-format lead-time outputs.",
    )
    parser.add_argument(
        "--pca-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "wusd3_pca",
        help="Directory for the standalone WUS-D3 PCA artifacts.",
    )
    parser.add_argument(
        "--inspect-only",
        type=parse_bool,
        default=False,
        help="Inspect one representative WUS-D3 file and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Wusd3Dataset(
        dataset_id=args.dataset_id,
        domain=args.domain,
        root_dir=default_wusd3_dataset().root_dir,
    )
    if args.inspect_only:
        run_inspection(dataset)
        return

    available_dataset_ids = discover_wusd3_dataset_ids(root_dir=dataset.root_dir)
    if dataset.dataset_id not in available_dataset_ids:
        raise ValueError(
            f"dataset_id {dataset.dataset_id!r} not found. "
            f"Available ids include: {available_dataset_ids}"
        )

    candidate_years = sorted(args.years if args.years is not None else discover_wusd3_water_years(dataset))
    if not candidate_years:
        raise RuntimeError("No WUS-D3 water years discovered.")

    print(f"WUS-D3 dataset_id: {dataset.dataset_id}", flush=True)
    print(f"WUS-D3 domain: {dataset.domain}", flush=True)
    print(f"candidate water years: {candidate_years}", flush=True)
    print(f"Dec 31 mode: {args.dec31_mode}", flush=True)
    print(f"include_sst: {args.include_sst}", flush=True)

    target_maps, target_dates = load_apr1_swe_maps(dataset, candidate_years)
    target_pca = fit_standardized_wusd3_pca(
        target_maps,
        dataset=dataset,
        target_dates=target_dates,
        n_components=PCA_COMPONENTS,
    )
    save_pca_artifacts(args.pca_output_dir, target_pca)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: List[Dict[str, Any]] = []
    for lead_months in sorted({int(value) for value in args.lead_months_list}):
        if lead_months < 0:
            raise ValueError(f"Lead months must be >= 0, got {lead_months}")
        lead_dir = args.output_dir / f"lead_{lead_months}_month"
        print(f"=== RUN LEAD {lead_months} MONTH ===", flush=True)
        summary = run_single_experiment(
            dataset,
            target_pca=target_pca,
            output_dir=lead_dir,
            lead_months=lead_months,
            dec31_mode=args.dec31_mode,
        )
        run_summaries.append(summary)

    save_leadtime_summary(args.output_dir, run_summaries)
    print(f"saved lead-time summary outputs to {args.output_dir}", flush=True)


def run_inspection(dataset: Wusd3Dataset) -> None:
    water_years = discover_wusd3_water_years(dataset)
    if not water_years:
        raise RuntimeError("No WUS-D3 files found for inspection.")
    example_water_year = int(water_years[0])
    example_path = variable_path_for_file_year(
        dataset,
        variable_key="swe",
        file_year=example_water_year - 1,
    )
    path_summary = inspect_wusd3_file(example_path)
    print(json.dumps(path_summary, indent=2, sort_keys=True, default=str), flush=True)


def load_apr1_swe_maps(
    dataset: Wusd3Dataset,
    water_years: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    maps: Dict[int, np.ndarray] = {}
    target_dates: Dict[int, str] = {}
    for water_year in water_years:
        target_date = date(water_year, 4, 1)
        field = load_snapshot(dataset, variable_key="swe", snapshot_date=target_date)
        values = np.asarray(field.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                f"Expected WUS-D3 SWE snapshot to be 2D on lat2d/lon2d, got shape {values.shape}"
            )
        if not np.isfinite(values).any():
            raise ValueError(f"No finite WUS-D3 SWE values on {target_date.isoformat()}")
        maps[water_year] = values
        target_dates[water_year] = target_date.isoformat()
        print(
            f"loaded April 1 SWE WY{water_year}: file_year={file_year_for_date(target_date)} "
            f"shape={values.shape} finite_cells={int(np.isfinite(values).sum())}",
            flush=True,
        )
    return maps, target_dates


def fit_standardized_wusd3_pca(
    target_maps: Dict[int, np.ndarray],
    *,
    dataset: Wusd3Dataset,
    target_dates: Dict[int, str],
    n_components: int,
) -> Wusd3TargetPca:
    years = sorted(target_maps)
    cube = np.stack([target_maps[year] for year in years], axis=0)
    full_matrix = cube.reshape(cube.shape[0], -1)
    valid_cell_mask = np.isfinite(full_matrix).all(axis=0)
    if int(valid_cell_mask.sum()) == 0:
        raise ValueError("No WUS-D3 SWE grid cells are finite across every selected year.")

    matrix = full_matrix[:, valid_cell_mask].astype(np.float64)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(standardized).astype(np.float32)

    metadata = {
        "dataset_id": dataset.dataset_id,
        "domain": dataset.domain,
        "target_definition": "April 1 WUS-D3 snow field by water year.",
        "sample_definition": (
            "One sample is one water year. Each map is WUS-D3 D02 snow selected on April 1, "
            "flattened after an all-year finite-cell mask, standardized by pixel, then sent to PCA."
        ),
        "swe_variable": "snow",
        "n_components": int(n_components),
        "years": years,
        "target_dates": [target_dates[year] for year in years],
        "grid_shape": list(cube.shape[1:]),
        "full_matrix_shape": list(full_matrix.shape),
        "pca_matrix_shape": list(standardized.shape),
        "valid_cell_count": int(valid_cell_mask.sum()),
        "pca_standardization": "Per-grid-cell standardization across years with sklearn StandardScaler.",
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
    }
    print(f"WUS-D3 PCA matrix shape: {tuple(standardized.shape)}", flush=True)
    print(
        "WUS-D3 PCA explained variance ratio: "
        f"{[float(value) for value in pca.explained_variance_ratio_]}",
        flush=True,
    )
    return Wusd3TargetPca(
        years=years,
        scores=scores,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        mean_vector=scaler.mean_.astype(np.float32),
        scale_vector=scaler.scale_.astype(np.float32),
        valid_cell_mask=valid_cell_mask,
        grid_shape=(int(cube.shape[1]), int(cube.shape[2])),
        metadata=metadata,
    )


def save_pca_artifacts(output_dir: Path, target_pca: Wusd3TargetPca) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pcs_header = ["water_year"] + [f"pc{i + 1}" for i in range(target_pca.scores.shape[1])]
    pcs_lines = [",".join(pcs_header)]
    for row_index, year in enumerate(target_pca.years):
        row = [str(year)] + [str(float(value)) for value in target_pca.scores[row_index]]
        pcs_lines.append(",".join(row))
    (output_dir / "pcs.csv").write_text("\n".join(pcs_lines) + "\n")

    explained_lines = ["component,explained_variance_ratio"]
    for index, value in enumerate(target_pca.explained_variance_ratio, start=1):
        explained_lines.append(f"pc{index},{float(value)}")
    (output_dir / "explained_variance.csv").write_text("\n".join(explained_lines) + "\n")

    np.savez_compressed(
        output_dir / "wusd3_pca_arrays.npz",
        years=np.asarray(target_pca.years, dtype=np.int32),
        scores=target_pca.scores,
        explained_variance_ratio=target_pca.explained_variance_ratio,
        components=target_pca.components,
        mean_vector=target_pca.mean_vector,
        scale_vector=target_pca.scale_vector,
        valid_cell_mask=target_pca.valid_cell_mask.reshape(target_pca.grid_shape).astype(np.uint8),
        metadata_json=np.asarray(json.dumps(target_pca.metadata), dtype=str),
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(target_pca.metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"saved WUS-D3 PCA artifacts to {output_dir}", flush=True)


def run_single_experiment(
    dataset: Wusd3Dataset,
    *,
    target_pca: Wusd3TargetPca,
    output_dir: Path,
    lead_months: int,
    dec31_mode: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = assemble_feature_table(
        dataset,
        target_pca=target_pca,
        lead_months=lead_months,
        dec31_mode=dec31_mode,
    )
    aligned_rows = align_features_and_targets(feature_rows, target_pca)
    if not aligned_rows:
        raise RuntimeError("No aligned WUS-D3 feature/target rows remain.")

    target_columns = ["pc1", "pc2"]
    feature_columns = [
        name for name in aligned_rows[0]
        if name not in {"water_year", *target_columns}
    ]
    aligned_years = [int(row["water_year"]) for row in aligned_rows]

    print(f"final aligned years: {aligned_years}", flush=True)
    print(f"number of features: {len(feature_columns)}", flush=True)
    print(f"feature columns used: {feature_columns}", flush=True)

    save_targets_csv(output_dir / "targets.csv", target_pca, predict_pc2=True)
    save_target_metadata(output_dir / "target_pca_metadata.json", target_pca)
    save_target_pca_npz(output_dir / "target_pca_arrays.npz", target_pca)
    save_aligned_dataset_csv(
        output_dir / "aligned_dataset.csv",
        aligned_rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    prediction_rows, metrics = run_fixed_alpha_loyo(
        aligned_rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
    save_predictions_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        target_columns=target_columns,
    )
    save_metrics(output_dir / "metrics.json", metrics)
    run_metadata = {
        "dataset_id": dataset.dataset_id,
        "domain": dataset.domain,
        "lead_months": int(lead_months),
        "predictor_cutoff_dates_by_year": {
            str(year): predictor_cutoff_date(int(year), lead_months).isoformat()
            for year in aligned_years
        },
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "aligned_years": aligned_years,
        "dec31_mode": dec31_mode,
        "alpha": 1.0,
        "target_pca_years": target_pca.years,
        "pca_components_fit": int(target_pca.components.shape[0]),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"saved outputs to {output_dir}", flush=True)
    print(f"final metrics: {metrics}", flush=True)
    return {
        "lead_months": int(lead_months),
        "output_dir": str(output_dir),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "aligned_years": aligned_years,
        "cutoff_example": predictor_cutoff_date(aligned_years[0], lead_months).isoformat(),
    }


def assemble_feature_table(
    dataset: Wusd3Dataset,
    *,
    target_pca: Wusd3TargetPca,
    lead_months: int,
    dec31_mode: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for water_year in target_pca.years:
        cutoff_date = predictor_cutoff_date(water_year, lead_months)
        print(f"build feature row WY{water_year} cutoff={cutoff_date.isoformat()}", flush=True)
        row: Dict[str, float] = {"water_year": float(water_year)}
        row.update(build_dec31_features(dataset, water_year=water_year, target_pca=target_pca, mode=dec31_mode))
        row.update(build_temperature_features(dataset, cutoff_date=cutoff_date))
        row.update(build_precip_features(dataset, cutoff_date=cutoff_date))
        bad_columns = [
            name for name, value in row.items()
            if name != "water_year" and not np.isfinite(float(value))
        ]
        if bad_columns:
            raise ValueError(f"Non-finite WUS-D3 features for WY{water_year}: {bad_columns}")
        rows.append(row)
    return rows


def build_dec31_features(
    dataset: Wusd3Dataset,
    *,
    water_year: int,
    target_pca: Wusd3TargetPca,
    mode: str,
) -> Dict[str, float]:
    field = load_snapshot(dataset, variable_key="swe", snapshot_date=date(water_year - 1, 12, 31))
    values = np.asarray(field.values, dtype=np.float32)
    if mode == "stats":
        finite = values[np.isfinite(values)]
        return {
            "dec31_swe_mean": float(finite.mean()),
            "dec31_swe_std": float(finite.std()),
        }
    if mode != "pcs":
        raise ValueError(f"Unsupported Dec 31 mode: {mode}")
    flat = values.reshape(-1)[target_pca.valid_cell_mask].astype(np.float32)
    if not np.isfinite(flat).all():
        raise ValueError(f"Dec 31 WUS-D3 SWE has missing values inside the April 1 PCA mask for WY{water_year}")
    standardized = (flat - target_pca.mean_vector) / target_pca.scale_vector
    projected = standardized @ target_pca.components.T
    return {
        "dec31_pc1": float(projected[0]),
        "dec31_pc2": float(projected[1]),
    }


def build_temperature_features(
    dataset: Wusd3Dataset,
    *,
    cutoff_date: date,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for days in (30, 60, 90):
        start_date = cutoff_date - timedelta(days=days - 1)
        series = load_domain_series(
            dataset,
            variable_key="t2m",
            start_date=start_date,
            end_date=cutoff_date,
        )
        values = np.asarray(series.values, dtype=np.float64)
        features[f"t2m_mean_last_{days}d"] = float(values.mean())
        features[f"t2m_std_last_{days}d"] = float(values.std())
    return features


def build_precip_features(
    dataset: Wusd3Dataset,
    *,
    cutoff_date: date,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for days in (30, 60, 90):
        start_date = cutoff_date - timedelta(days=days - 1)
        series = load_domain_series(
            dataset,
            variable_key="tp",
            start_date=start_date,
            end_date=cutoff_date,
        )
        values = np.asarray(series.values, dtype=np.float64)
        features[f"tp_sum_last_{days}d"] = float(values.sum())
    return features


def align_features_and_targets(
    feature_rows: List[Dict[str, float]],
    target_pca: Wusd3TargetPca,
) -> List[Dict[str, float]]:
    year_to_index = {year: index for index, year in enumerate(target_pca.years)}
    aligned: List[Dict[str, float]] = []
    for row in feature_rows:
        water_year = int(row["water_year"])
        index = year_to_index[water_year]
        combined = dict(row)
        combined["pc1"] = float(target_pca.scores[index, 0])
        combined["pc2"] = float(target_pca.scores[index, 1])
        aligned.append(combined)
    return aligned


def run_fixed_alpha_loyo(
    rows: List[Dict[str, float]],
    *,
    feature_columns: List[str],
    target_columns: List[str],
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    years = np.asarray([int(row["water_year"]) for row in rows], dtype=np.int32)
    x = np.asarray([[float(row[name]) for name in feature_columns] for row in rows], dtype=np.float64)
    predictions: Dict[int, Dict[str, float]] = {
        int(year): {"water_year": float(year)}
        for year in years
    }
    for target_name in target_columns:
        y = np.asarray([float(row[target_name]) for row in rows], dtype=np.float64)
        for test_index, test_year in enumerate(years):
            train_mask = np.arange(len(rows)) != test_index
            x_train = x[train_mask]
            y_train = y[train_mask]
            x_test = x[test_index : test_index + 1]
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model = Ridge(alpha=1.0)
            model.fit(x_train_scaled, y_train)
            predicted = float(model.predict(x_test_scaled)[0])
            predictions[int(test_year)][f"{target_name}_true"] = float(y[test_index])
            predictions[int(test_year)][f"{target_name}_pred"] = predicted
            predictions[int(test_year)][f"{target_name}_abs_error"] = abs(predicted - float(y[test_index]))
            print(
                f"outer fold target={target_name} test_year={int(test_year)} "
                f"alpha=1.0 true={float(y[test_index]):.6g} pred={predicted:.6g}",
                flush=True,
            )
    prediction_rows = [predictions[int(year)] for year in years]
    metrics = compute_metrics(prediction_rows, target_columns)
    return prediction_rows, metrics


def save_target_metadata(path: Path, target_pca: Wusd3TargetPca) -> None:
    path.write_text(json.dumps(target_pca.metadata, indent=2, sort_keys=True) + "\n")


def save_target_pca_npz(path: Path, target_pca: Wusd3TargetPca) -> None:
    component_maps = np.full(
        (target_pca.components.shape[0], target_pca.valid_cell_mask.size),
        np.nan,
        dtype=np.float32,
    )
    component_maps[:, target_pca.valid_cell_mask] = target_pca.components
    component_maps = component_maps.reshape((target_pca.components.shape[0],) + target_pca.grid_shape)
    mean_map = np.full(target_pca.valid_cell_mask.size, np.nan, dtype=np.float32)
    mean_map[target_pca.valid_cell_mask] = target_pca.mean_vector
    mean_map = mean_map.reshape(target_pca.grid_shape)
    scale_map = np.full(target_pca.valid_cell_mask.size, np.nan, dtype=np.float32)
    scale_map[target_pca.valid_cell_mask] = target_pca.scale_vector
    scale_map = scale_map.reshape(target_pca.grid_shape)
    np.savez_compressed(
        path,
        years=np.asarray(target_pca.years, dtype=np.int32),
        scores=target_pca.scores[:, :2],
        explained_variance_ratio=target_pca.explained_variance_ratio,
        components=target_pca.components,
        component_maps=component_maps,
        mean_vector=target_pca.mean_vector,
        scale_vector=target_pca.scale_vector,
        mean_map=mean_map,
        scale_map=scale_map,
        valid_cell_mask=target_pca.valid_cell_mask.reshape(target_pca.grid_shape).astype(np.uint8),
        metadata_json=np.asarray(json.dumps(target_pca.metadata), dtype=str),
    )


def save_leadtime_summary(
    output_dir: Path,
    run_summaries: List[Dict[str, Any]],
) -> None:
    rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {"runs": run_summaries, "targets": {}}
    for summary in sorted(run_summaries, key=lambda item: int(item["lead_months"])):
        lead_months = int(summary["lead_months"])
        for target_name, metric_values in summary["metrics"].items():
            row = {
                "lead_months": lead_months,
                "target": target_name,
                "pearson_correlation": metric_values.get("pearson_correlation"),
                "r2": metric_values.get("r2"),
                "rmse": metric_values.get("rmse"),
                "mae": metric_values.get("mae"),
                "feature_count": len(summary["feature_columns"]),
                "year_count": len(summary["aligned_years"]),
                "cutoff_example": summary["cutoff_example"],
                "output_dir": summary["output_dir"],
            }
            rows.append(row)
            summary_json["targets"].setdefault(target_name, []).append(row)

    csv_header = [
        "lead_months",
        "target",
        "pearson_correlation",
        "r2",
        "rmse",
        "mae",
        "feature_count",
        "year_count",
        "cutoff_example",
        "output_dir",
    ]
    csv_lines = [",".join(csv_header)]
    for row in rows:
        csv_lines.append(",".join(str(row[name]) for name in csv_header))
    (output_dir / "leadtime_summary.csv").write_text("\n".join(csv_lines) + "\n")
    (output_dir / "leadtime_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
    )
    save_skill_vs_lead_plots(output_dir, rows)


def save_skill_vs_lead_plots(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: could not create lead-time skill plots: {exc}", flush=True)
        return

    for target_name in ("pc1", "pc2"):
        target_rows = [row for row in rows if row["target"] == target_name]
        if not target_rows:
            continue
        target_rows.sort(key=lambda item: int(item["lead_months"]))
        lead = [int(row["lead_months"]) for row in target_rows]
        corr = [float(row["pearson_correlation"]) for row in target_rows]
        r2 = [float(row["r2"]) for row in target_rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(lead, corr, marker="o", label="Pearson r")
        ax.plot(lead, r2, marker="o", label="R2")
        ax.set_title(f"{target_name.upper()} skill vs lead time")
        ax.set_xlabel("Lead time (months)")
        ax.set_ylabel("Skill")
        ax.set_xticks(lead)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{target_name}_skill_vs_lead.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
