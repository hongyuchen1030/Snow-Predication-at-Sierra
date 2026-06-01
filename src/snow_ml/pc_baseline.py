from __future__ import annotations

import calendar
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from snow_ml.data import (
    ERA5_DAILY_REDUCTIONS,
    ForecastConfig,
    RegionBounds,
    SST_MONTHLY_MEAN_PATH,
    SWE_ROOT_PATH,
    SweGridDefinition,
    era5_land_yearly_file,
    load_swe_snapshot,
    load_target_swe_map,
    region_to_dict,
)


TARGET_MMDD = "04-01"
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PREDICTOR_PCA_OPTIONS = (None, 3, 5, 8)


@dataclass(frozen=True)
class SweTargetPca:
    years: list[int]
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    mean_vector: np.ndarray
    valid_cell_mask: np.ndarray
    grid_shape: tuple[int, int]
    true_domain_mean: dict[int, float]
    metadata: dict[str, object]


def discover_swe_water_years() -> list[int]:
    pattern = re.compile(r"_WY(\d{4})_SD_SWE_SCA_POST\.nc$")
    years: list[int] = []
    for path in sorted(SWE_ROOT_PATH.glob("*.nc")):
        match = pattern.search(path.name)
        if match:
            years.append(int(match.group(1)))
    return years


def print_region_check(
    *,
    expected_region: RegionBounds,
    grid: SweGridDefinition,
    sample_year: int,
) -> None:
    print("=== REGION CHECK ===", flush=True)
    print(
        "Expected:",
        expected_region.lat_min,
        expected_region.lat_max,
        expected_region.lon_min,
        expected_region.lon_max,
        flush=True,
    )
    print(
        "SWE actual lat range:",
        _coord_min(grid.fine_latitude),
        _coord_max(grid.fine_latitude),
        flush=True,
    )
    print(
        "SWE actual lon range:",
        _coord_min(grid.fine_longitude),
        _coord_max(grid.fine_longitude),
        flush=True,
    )
    print(
        "SWE requested region:",
        region_to_dict(grid.requested_region),
        flush=True,
    )
    print(
        "SWE effective region:",
        region_to_dict(grid.effective_region),
        flush=True,
    )
    _print_atmospheric_region_check("t2m", sample_year, expected_region)
    _print_atmospheric_region_check("tp", sample_year, expected_region)
    _print_sst_region_check(expected_region)
    print("=== END REGION CHECK ===", flush=True)


def load_apr1_swe_maps(
    years: Iterable[int],
    *,
    grid: SweGridDefinition,
    region: RegionBounds,
    coarsen_factor: int,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    maps: dict[int, np.ndarray] = {}
    dropped: dict[int, str] = {}
    for water_year in sorted(years):
        config = ForecastConfig(
            water_year=water_year,
            target_month_day=TARGET_MMDD,
            region=region,
            coarsen_factor=coarsen_factor,
        )
        try:
            field = load_target_swe_map(config, swe_grid=grid, fill_missing=False)
            values = np.asarray(field.values, dtype=np.float32)
            if not np.isfinite(values).any():
                dropped[water_year] = "April 1 SWE map has no finite cells"
                continue
            maps[water_year] = values
            print(
                f"target map WY{water_year}: shape={tuple(values.shape)} "
                f"finite_cells={int(np.isfinite(values).sum())}",
                flush=True,
            )
        except Exception as exc:
            dropped[water_year] = f"April 1 SWE load failed: {exc}"
            print(f"drop target WY{water_year}: {dropped[water_year]}", flush=True)
    return maps, dropped


def fit_swe_pca(
    target_maps: dict[int, np.ndarray],
    *,
    n_components: int,
    grid: SweGridDefinition,
) -> SweTargetPca:
    years = sorted(target_maps)
    if len(years) < n_components:
        raise ValueError(
            f"Need at least {n_components} usable target years, got {len(years)}."
        )

    cube = np.stack([target_maps[year] for year in years], axis=0)
    full_matrix = cube.reshape(cube.shape[0], -1)
    valid_cell_mask = np.isfinite(full_matrix).all(axis=0)
    if int(valid_cell_mask.sum()) == 0:
        raise ValueError("No April 1 SWE grid cells are finite for every usable year.")

    matrix = full_matrix[:, valid_cell_mask].astype(np.float64)
    mean_vector = matrix.mean(axis=0)
    centered = matrix - mean_vector
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(centered).astype(np.float32)
    true_domain_mean = {
        year: float(np.nanmean(target_maps[year].reshape(-1)[valid_cell_mask]))
        for year in years
    }

    metadata = {
        "target_definition": "April 1 SWE_Post mean field by water year.",
        "pca_centering": "Column means over usable years and all-year-finite grid cells.",
        "years": years,
        "n_components": int(n_components),
        "explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "grid_shape": list(grid.grid_shape),
        "valid_cell_count": int(valid_cell_mask.sum()),
        "requested_region": region_to_dict(grid.requested_region),
        "effective_region": region_to_dict(grid.effective_region),
        "coarsen_factor": int(grid.coarsen_factor),
    }
    print(f"target PCA years: {years}", flush=True)
    print(f"target PCA matrix shape: {tuple(matrix.shape)}", flush=True)
    print(
        "target PCA explained variance ratio: "
        f"{[float(value) for value in pca.explained_variance_ratio_]}",
        flush=True,
    )
    return SweTargetPca(
        years=years,
        scores=scores,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        mean_vector=mean_vector.astype(np.float32),
        valid_cell_mask=valid_cell_mask,
        grid_shape=tuple(int(size) for size in grid.grid_shape),
        true_domain_mean=true_domain_mean,
        metadata=metadata,
    )


def build_dec31_features(
    water_year: int,
    *,
    grid: SweGridDefinition,
    mode: str,
    target_pca: SweTargetPca | None = None,
) -> dict[str, float]:
    field = load_swe_snapshot(
        water_year,
        date(water_year - 1, 12, 31),
        stat_name="mean",
        swe_grid=grid,
        fill_missing=False,
    )
    values = np.asarray(field.values, dtype=np.float32)
    if mode == "stats":
        return {
            "dec31_swe_mean": _finite_mean(values),
            "dec31_swe_std": _finite_std(values),
        }
    if mode != "pcs":
        raise ValueError(f"Unsupported Dec 31 mode: {mode}")
    if target_pca is None:
        raise ValueError("target_pca is required when dec31-mode is pcs")
    flat = values.reshape(-1)[target_pca.valid_cell_mask].astype(np.float32)
    if not np.isfinite(flat).all():
        raise ValueError("Dec 31 SWE has missing values inside the April 1 PCA mask")
    centered = flat - target_pca.mean_vector
    projected = centered @ target_pca.components.T
    output = {"dec31_pc1": float(projected[0])}
    if target_pca.components.shape[0] > 1:
        output["dec31_pc2"] = float(projected[1])
    return output


def build_temperature_features(
    water_year: int,
    *,
    region: RegionBounds,
    include_seasonal_features: bool,
    cutoff_date: date,
) -> dict[str, float]:
    windows: list[tuple[str, date, date]] = [
        ("t2m_mean_last_30d", _window_start_from_end_date(cutoff_date, 30), cutoff_date),
        ("t2m_mean_last_60d", _window_start_from_end_date(cutoff_date, 60), cutoff_date),
        ("t2m_mean_last_90d", _window_start_from_end_date(cutoff_date, 90), cutoff_date),
    ]
    if include_seasonal_features:
        windows.extend([
            ("t2m_mean_DJF", date(water_year - 1, 12, 1), date(water_year, 3, 1) - timedelta(days=1)),
            ("t2m_mean_JFM", date(water_year, 1, 1), date(water_year, 3, 31)),
        ])
    active_windows = _clip_named_windows_to_cutoff(windows, cutoff_date)
    daily = _load_atmospheric_daily_domain_series(
        "t2m",
        min(start for _, start, _ in active_windows),
        cutoff_date,
        region,
    )
    return {
        name: _series_window_mean(daily, start, end)
        for name, start, end in active_windows
    }


def build_precip_features(
    water_year: int,
    *,
    region: RegionBounds,
    include_seasonal_features: bool,
    cutoff_date: date,
) -> dict[str, float]:
    windows: list[tuple[str, date, date]] = [
        ("tp_sum_last_30d", _window_start_from_end_date(cutoff_date, 30), cutoff_date),
        ("tp_sum_last_60d", _window_start_from_end_date(cutoff_date, 60), cutoff_date),
        ("tp_sum_last_90d", _window_start_from_end_date(cutoff_date, 90), cutoff_date),
    ]
    if include_seasonal_features:
        windows.extend([
            ("tp_sum_DJF", date(water_year - 1, 12, 1), date(water_year, 3, 1) - timedelta(days=1)),
            ("tp_sum_JFM", date(water_year, 1, 1), date(water_year, 3, 31)),
        ])
    active_windows = _clip_named_windows_to_cutoff(windows, cutoff_date)
    daily = _load_atmospheric_daily_domain_series(
        "tp",
        min(start for _, start, _ in active_windows),
        cutoff_date,
        region,
    )
    return {
        name: _series_window_sum(daily, start, end)
        for name, start, end in active_windows
    }


def build_sst_features(
    water_year: int,
    *,
    region: RegionBounds,
    cutoff_date: date,
) -> dict[str, float]:
    windows: list[tuple[str, date, date]] = [
        ("sst_mean_last_90d", _window_start_from_end_date(cutoff_date, 90), cutoff_date),
        ("sst_mean_last_180d", _window_start_from_end_date(cutoff_date, 180), cutoff_date),
        ("sst_mean_DJF", date(water_year - 1, 12, 1), date(water_year, 3, 1) - timedelta(days=1)),
        ("sst_mean_JFM", date(water_year, 1, 1), date(water_year, 3, 31)),
    ]
    active_windows = _clip_named_windows_to_cutoff(windows, cutoff_date)
    series = _load_sst_monthly_domain_series(
        min(start for _, start, _ in active_windows),
        cutoff_date,
        region,
    )
    return {
        name: _series_window_mean(series, start, end)
        for name, start, end in active_windows
    }


def assemble_feature_table(
    years: Iterable[int],
    *,
    grid: SweGridDefinition,
    region: RegionBounds,
    dec31_mode: str,
    include_seasonal_features: bool,
    target_pca: SweTargetPca | None,
    lead_months: int = 0,
) -> tuple[list[dict[str, float]], dict[int, str]]:
    rows: list[dict[str, float]] = []
    dropped: dict[int, str] = {}
    for water_year in sorted(years):
        print(f"build feature row WY{water_year}", flush=True)
        try:
            cutoff_date = predictor_cutoff_date(water_year, lead_months)
            row: dict[str, float] = {"water_year": float(water_year)}
            row.update(
                build_dec31_features(
                    water_year,
                    grid=grid,
                    mode=dec31_mode,
                    target_pca=target_pca,
                )
            )
            row.update(
                build_temperature_features(
                    water_year,
                    region=region,
                    include_seasonal_features=include_seasonal_features,
                    cutoff_date=cutoff_date,
                )
            )
            row.update(
                build_precip_features(
                    water_year,
                    region=region,
                    include_seasonal_features=include_seasonal_features,
                    cutoff_date=cutoff_date,
                )
            )
            row.update(build_sst_features(water_year, region=region, cutoff_date=cutoff_date))
            bad_columns = [
                name for name, value in row.items()
                if name != "water_year" and not np.isfinite(float(value))
            ]
            if bad_columns:
                dropped[water_year] = f"non-finite features: {bad_columns}"
                continue
            rows.append(row)
        except Exception as exc:
            dropped[water_year] = str(exc)
            print(f"drop feature WY{water_year}: {exc}", flush=True)
    return rows, dropped


def align_features_and_targets(
    feature_rows: list[dict[str, float]],
    target_pca: SweTargetPca,
    *,
    predict_pc2: bool,
) -> list[dict[str, float]]:
    score_by_year = {
        year: target_pca.scores[index]
        for index, year in enumerate(target_pca.years)
    }
    aligned: list[dict[str, float]] = []
    for row in feature_rows:
        water_year = int(row["water_year"])
        if water_year not in score_by_year:
            continue
        combined = dict(row)
        combined["water_year"] = float(water_year)
        combined["pc1"] = float(score_by_year[water_year][0])
        if predict_pc2:
            combined["pc2"] = float(score_by_year[water_year][1])
        aligned.append(combined)
    return sorted(aligned, key=lambda item: int(item["water_year"]))


def run_loyo(
    rows: list[dict[str, float]],
    *,
    feature_columns: list[str],
    target_columns: list[str],
    enable_predictor_pca: bool,
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    years = np.asarray([int(row["water_year"]) for row in rows], dtype=np.int32)
    x = np.asarray([[float(row[name]) for name in feature_columns] for row in rows], dtype=np.float64)
    predictions: dict[int, dict[str, float]] = {
        int(year): {"water_year": float(year)}
        for year in years
    }

    for target_name in target_columns:
        y = np.asarray([float(row[target_name]) for row in rows], dtype=np.float64)
        for test_index, test_year in enumerate(years):
            train_indices = np.asarray([idx for idx in range(len(rows)) if idx != test_index], dtype=np.int32)
            best_alpha, best_pca = tune_hyperparameters(
                x[train_indices],
                y[train_indices],
                enable_predictor_pca=enable_predictor_pca,
            )
            predicted = fit_predict_one_fold(
                x[train_indices],
                y[train_indices],
                x[test_index : test_index + 1],
                alpha=best_alpha,
                predictor_pca_components=best_pca,
            )
            prefix = target_name
            predictions[int(test_year)][f"{prefix}_true"] = float(y[test_index])
            predictions[int(test_year)][f"{prefix}_pred"] = float(predicted[0])
            predictions[int(test_year)][f"{prefix}_abs_error"] = abs(float(predicted[0]) - float(y[test_index]))
            print(
                f"outer fold target={target_name} test_year={int(test_year)} "
                f"alpha={best_alpha} predictor_pca={best_pca} "
                f"true={float(y[test_index]):.6g} pred={float(predicted[0]):.6g}",
                flush=True,
            )

    prediction_rows = [predictions[int(year)] for year in years]
    metrics = compute_metrics(prediction_rows, target_columns)
    return prediction_rows, metrics


def tune_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    *,
    enable_predictor_pca: bool,
) -> tuple[float, int | None]:
    if x.shape[0] < 3:
        return 1.0, None

    pca_options = PREDICTOR_PCA_OPTIONS if enable_predictor_pca else (None,)
    best_score = math.inf
    best_alpha = 1.0
    best_pca: int | None = None
    splits = _inner_splits(x.shape[0])
    for alpha in RIDGE_ALPHAS:
        for pca_components in pca_options:
            if pca_components is not None and pca_components > min(x.shape[1], x.shape[0] - 1):
                continue
            fold_errors: list[float] = []
            for train_index, val_index in splits:
                if pca_components is not None and pca_components > min(x.shape[1], len(train_index)):
                    fold_errors = []
                    break
                pred = fit_predict_one_fold(
                    x[train_index],
                    y[train_index],
                    x[val_index],
                    alpha=alpha,
                    predictor_pca_components=pca_components,
                )
                fold_errors.extend((pred - y[val_index]).tolist())
            if not fold_errors:
                continue
            rmse = float(np.sqrt(np.mean(np.square(fold_errors))))
            if rmse < best_score:
                best_score = rmse
                best_alpha = float(alpha)
                best_pca = pca_components
    return best_alpha, best_pca


def fit_predict_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
    predictor_pca_components: int | None,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    if predictor_pca_components is not None:
        pca = PCA(n_components=predictor_pca_components, svd_solver="full")
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)
    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)
    return np.asarray(model.predict(x_test_scaled), dtype=np.float64)


def compute_metrics(
    prediction_rows: list[dict[str, float]],
    target_columns: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for target_name in target_columns:
        true = np.asarray([row[f"{target_name}_true"] for row in prediction_rows], dtype=np.float64)
        pred = np.asarray([row[f"{target_name}_pred"] for row in prediction_rows], dtype=np.float64)
        err = pred - true
        metrics[target_name] = {
            "pearson_correlation": _pearson(true, pred),
            "r2": _r2(true, pred),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "mae": float(np.mean(np.abs(err))),
        }
    return metrics


def save_targets_csv(path: Path, target_pca: SweTargetPca, *, predict_pc2: bool) -> None:
    header = ["water_year", "pc1"] + (["pc2"] if predict_pc2 else [])
    rows = []
    for index, year in enumerate(target_pca.years):
        row = [year, float(target_pca.scores[index, 0])]
        if predict_pc2:
            row.append(float(target_pca.scores[index, 1]))
        rows.append(row)
    _write_csv(path, header, rows)


def save_aligned_dataset_csv(
    path: Path,
    rows: list[dict[str, float]],
    *,
    feature_columns: list[str],
    target_columns: list[str],
) -> None:
    header = ["water_year"] + feature_columns + target_columns
    table = [[_format_csv_value(row[name]) for name in header] for row in rows]
    _write_csv(path, header, table)


def save_predictions_csv(
    path: Path,
    rows: list[dict[str, float]],
    *,
    target_columns: list[str],
) -> None:
    header = ["water_year"]
    for target_name in target_columns:
        header.extend([
            f"{target_name}_true",
            f"{target_name}_pred",
            f"{target_name}_abs_error",
        ])
    table = [[_format_csv_value(row[name]) for name in header] for row in rows]
    _write_csv(path, header, table)


def save_target_metadata(path: Path, target_pca: SweTargetPca) -> None:
    path.write_text(json.dumps(target_pca.metadata, indent=2, sort_keys=True) + "\n")


def save_target_pca_npz(path: Path, target_pca: SweTargetPca) -> None:
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
    np.savez_compressed(
        path,
        years=np.asarray(target_pca.years, dtype=np.int32),
        scores=target_pca.scores,
        explained_variance_ratio=target_pca.explained_variance_ratio,
        components=target_pca.components,
        component_maps=component_maps,
        mean_vector=target_pca.mean_vector,
        mean_map=mean_map,
        valid_cell_mask=target_pca.valid_cell_mask.reshape(target_pca.grid_shape).astype(np.uint8),
        metadata_json=np.asarray(json.dumps(target_pca.metadata), dtype=str),
    )


def save_metrics(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    path.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n")


def save_reconstruction_diagnostics(
    path: Path,
    prediction_rows: list[dict[str, float]],
    target_pca: SweTargetPca,
) -> None:
    if target_pca.components.shape[0] < 2:
        raise ValueError("Reconstruction diagnostics require PC1 and PC2.")
    rows = []
    for row in prediction_rows:
        water_year = int(row["water_year"])
        predicted_scores = np.asarray([row["pc1_pred"], row["pc2_pred"]], dtype=np.float32)
        reconstructed = target_pca.mean_vector + predicted_scores @ target_pca.components[:2]
        rows.append(
            [
                water_year,
                target_pca.true_domain_mean[water_year],
                float(np.nanmean(reconstructed)),
                abs(float(np.nanmean(reconstructed)) - target_pca.true_domain_mean[water_year]),
            ]
        )
    _write_csv(
        path,
        [
            "water_year",
            "true_apr1_domain_mean_swe",
            "reconstructed_apr1_domain_mean_swe",
            "abs_error",
        ],
        rows,
    )


def _load_atmospheric_daily_domain_series(
    variable_name: str,
    start: date,
    end: date,
    region: RegionBounds,
) -> xr.DataArray:
    pieces: list[xr.DataArray] = []
    for year in range(start.year, end.year + 1):
        path = era5_land_yearly_file(variable_name, year)
        selection_start = np.datetime64(max(start, date(year, 1, 1)).isoformat())
        selection_end = np.datetime64(min(end, date(year, 12, 31)).isoformat()) + np.timedelta64(23, "h")
        with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
            field = ds[variable_name].sel(time=slice(selection_start, selection_end))
            if int(field.sizes.get("time", 0)) == 0:
                continue
            field = _prepare_latlon_field(field)
            field = _subset_latlon_field(field, region)
            lat_name, lon_name = _latlon_names(field)
            domain = field.mean(dim=[lat_name, lon_name], skipna=True)
            if ERA5_DAILY_REDUCTIONS[variable_name] == "mean":
                daily = domain.resample(time="1D").mean()
            else:
                daily = domain.resample(time="1D").sum()
            pieces.append(daily.load())
    if not pieces:
        raise RuntimeError(f"No {variable_name} data found between {start} and {end}")
    combined = xr.concat(pieces, dim="time").sortby("time")
    return combined.sel(time=slice(np.datetime64(start.isoformat()), np.datetime64(end.isoformat())))


def _load_sst_monthly_domain_series(
    start: date,
    end: date,
    region: RegionBounds,
) -> xr.DataArray:
    with xr.open_dataset(SST_MONTHLY_MEAN_PATH, engine="netcdf4", decode_times=True) as ds:
        sst = ds["sst"].sel(time=slice(np.datetime64(start.isoformat()), np.datetime64(end.isoformat())))
        if int(sst.sizes.get("time", 0)) == 0:
            raise RuntimeError(f"No SST data found between {start} and {end}")
        sst = _prepare_latlon_field(sst)
        sst = _subset_latlon_field(sst, region)
        lat_name, lon_name = _latlon_names(sst)
        return sst.mean(dim=[lat_name, lon_name], skipna=True).load()


def _prepare_latlon_field(field: xr.DataArray) -> xr.DataArray:
    lat_name, lon_name = _latlon_names(field)
    longitude = field[lon_name]
    if longitude.ndim == 1:
        field = field.assign_coords({lon_name: xr.where(longitude > 180.0, longitude - 360.0, longitude)})
    return field.sortby(lat_name).sortby(lon_name)


def _subset_latlon_field(field: xr.DataArray, region: RegionBounds) -> xr.DataArray:
    lat_name, lon_name = _latlon_names(field)
    subset = field.sel(
        {
            lat_name: slice(region.lat_min, region.lat_max),
            lon_name: slice(region.lon_min, region.lon_max),
        }
    )
    if int(subset.sizes.get(lat_name, 0)) == 0 or int(subset.sizes.get(lon_name, 0)) == 0:
        raise ValueError(f"Region {region.as_dict()} does not overlap {field.name} coordinates")
    return subset


def _latlon_names(field: xr.DataArray) -> tuple[str, str]:
    lat_name = next((name for name in ("latitude", "lat", "Latitude") if name in field.coords), None)
    lon_name = next((name for name in ("longitude", "lon", "Longitude") if name in field.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"Could not infer latitude/longitude names from coords {list(field.coords)}")
    return lat_name, lon_name


def _window_start_date(water_year: int, days: int) -> date:
    return date(water_year, 4, 1) - timedelta(days=days)


def _window_start_from_end_date(end_date: date, days: int) -> date:
    return end_date - timedelta(days=days - 1)


def _seasonal_start_date(water_year: int) -> date:
    return date(water_year - 1, 12, 1)


def predictor_cutoff_date(water_year: int, lead_months: int) -> date:
    if lead_months < 0:
        raise ValueError(f"lead_months must be >= 0, got {lead_months}")
    year = water_year
    month = 3 - lead_months
    while month <= 0:
        month += 12
        year -= 1
    day = calendar.monthrange(year, month)[1]
    return date(year, month, day)


def _clip_named_windows_to_cutoff(
    windows: list[tuple[str, date, date]],
    cutoff_date: date,
) -> list[tuple[str, date, date]]:
    clipped: list[tuple[str, date, date]] = []
    for name, start, end in windows:
        if start > cutoff_date:
            continue
        clipped.append((name, start, min(end, cutoff_date)))
    if not clipped:
        raise RuntimeError(f"No predictor windows remain after applying cutoff {cutoff_date.isoformat()}")
    return clipped


def _series_window_mean(series: xr.DataArray, start: date, end: date) -> float:
    selected = _select_series_window(series, start, end)
    return float(selected.mean(skipna=True).item())


def _series_window_sum(series: xr.DataArray, start: date, end: date) -> float:
    selected = _select_series_window(series, start, end)
    return float(selected.sum(skipna=True).item())


def _select_series_window(series: xr.DataArray, start: date, end: date) -> xr.DataArray:
    selected = series.sel(time=slice(np.datetime64(start.isoformat()), np.datetime64(end.isoformat())))
    if int(selected.sizes.get("time", 0)) == 0:
        raise RuntimeError(f"No data in window {start} to {end}")
    return selected


def _inner_splits(sample_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if sample_count <= 8:
        return [
            (
                np.asarray([j for j in range(sample_count) if j != i], dtype=np.int32),
                np.asarray([i], dtype=np.int32),
            )
            for i in range(sample_count)
        ]
    kfold = KFold(n_splits=min(5, sample_count), shuffle=False)
    return [
        (np.asarray(train_index, dtype=np.int32), np.asarray(val_index, dtype=np.int32))
        for train_index, val_index in kfold.split(np.arange(sample_count))
    ]


def _pearson(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size < 2 or np.std(true) == 0.0 or np.std(pred) == 0.0:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size == 0:
        return float("nan")
    denominator = float(np.sum(np.square(true - np.mean(true))))
    if denominator == 0.0:
        return float("nan")
    numerator = float(np.sum(np.square(true - pred)))
    return float(1.0 - numerator / denominator)


def _finite_mean(values: np.ndarray) -> float:
    if not np.isfinite(values).any():
        raise ValueError("No finite values available for mean")
    return float(np.nanmean(values))


def _finite_std(values: np.ndarray) -> float:
    if not np.isfinite(values).any():
        raise ValueError("No finite values available for std")
    return float(np.nanstd(values))


def _format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _write_csv(path: Path, header: list[str], rows: list[Iterable[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _print_atmospheric_region_check(
    variable_name: str,
    sample_year: int,
    expected_region: RegionBounds,
) -> None:
    path = era5_land_yearly_file(variable_name, sample_year)
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        field = _prepare_latlon_field(ds[variable_name])
        subset = _subset_latlon_field(field, expected_region)
        lat_name, lon_name = _latlon_names(subset)
        label = f"ERA5 {variable_name}"
        print(
            f"{label} actual lat range:",
            _coord_min(subset[lat_name]),
            _coord_max(subset[lat_name]),
            flush=True,
        )
        print(
            f"{label} actual lon range:",
            _coord_min(subset[lon_name]),
            _coord_max(subset[lon_name]),
            flush=True,
        )


def _print_sst_region_check(expected_region: RegionBounds) -> None:
    with xr.open_dataset(SST_MONTHLY_MEAN_PATH, engine="netcdf4", decode_times=True) as ds:
        field = _prepare_latlon_field(ds["sst"])
        subset = _subset_latlon_field(field, expected_region)
        lat_name, lon_name = _latlon_names(subset)
        print(
            "SST actual lat range:",
            _coord_min(subset[lat_name]),
            _coord_max(subset[lat_name]),
            flush=True,
        )
        print(
            "SST actual lon range:",
            _coord_min(subset[lon_name]),
            _coord_max(subset[lon_name]),
            flush=True,
        )


def _coord_min(values: xr.DataArray) -> float:
    return float(np.nanmin(np.asarray(values.values, dtype=np.float64)))


def _coord_max(values: xr.DataArray) -> float:
    return float(np.nanmax(np.asarray(values.values, dtype=np.float64)))


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
