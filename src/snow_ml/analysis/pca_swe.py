from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from snow_ml.data import (
    DEFAULT_MODEL_REGION,
    DEFAULT_COARSEN_FACTOR,
    SWE_ROOT_PATH,
    ForecastConfig,
    RegionBounds,
    get_regional_swe_grid_definition,
    load_target_swe_map,
    region_to_dict,
)

DEFAULT_PCA_TARGET_MONTH_DAY = "04-01"
DEFAULT_PCA_COMPONENTS = 5


@dataclass(frozen=True)
class SwePcaResult:
    years: np.ndarray
    target_dates: list[str]
    matrix_shape: tuple[int, int]
    full_matrix_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    valid_cell_count: int
    explained_variance_ratio: np.ndarray
    scores: np.ndarray
    components: np.ndarray
    component_maps: np.ndarray
    mean_map: np.ndarray
    valid_cell_mask: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    metadata: dict[str, object]


def discover_swe_water_years() -> list[int]:
    pattern = re.compile(r"_WY(\d{4})_SD_SWE_SCA_POST\.nc$")
    years: list[int] = []
    for path in sorted(SWE_ROOT_PATH.glob("*.nc")):
        match = pattern.search(path.name)
        if match:
            years.append(int(match.group(1)))
    return years


def run_swe_pca(
    *,
    years: list[int] | None,
    target_month_day: str,
    n_components: int,
    output_dir: Path,
    region: RegionBounds | None = DEFAULT_MODEL_REGION,
    coarsen_factor: int = DEFAULT_COARSEN_FACTOR,
) -> SwePcaResult:
    selected_years = sorted(years if years is not None else discover_swe_water_years())
    if len(selected_years) < n_components:
        raise ValueError(
            f"PCA needs at least n_components years. Got {len(selected_years)} years "
            f"for n_components={n_components}."
        )

    print(f"SWE PCA years: {selected_years}", flush=True)
    print(f"SWE PCA target month-day: {target_month_day}", flush=True)
    print("SWE PCA sample definition: SWE_Post mean statistic on the target month-day for each water year.", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = get_regional_swe_grid_definition(
        selected_years[0],
        region,
        coarsen_factor,
    )

    fields: list[np.ndarray] = []
    target_dates: list[str] = []
    for water_year in selected_years:
        config = ForecastConfig(
            water_year=water_year,
            target_month_day=target_month_day,
            region=region or DEFAULT_MODEL_REGION,
            coarsen_factor=coarsen_factor,
        )
        field = load_target_swe_map(config, swe_grid=grid, fill_missing=False)
        values = np.asarray(field.values, dtype=np.float32)
        fields.append(values)
        target_dates.append(str(field.name).replace("swe_mean_", ""))
        finite_count = int(np.isfinite(values).sum())
        print(
            f"loaded WY{water_year} date={target_dates[-1]} "
            f"field_shape={tuple(values.shape)} finite_cells={finite_count}",
            flush=True,
        )

    cube = np.stack(fields, axis=0)
    full_matrix = cube.reshape(cube.shape[0], -1)
    valid_cell_mask = np.isfinite(full_matrix).all(axis=0)
    valid_cell_count = int(valid_cell_mask.sum())
    if valid_cell_count == 0:
        raise ValueError("No grid cells are finite for every selected water year.")

    matrix = full_matrix[:, valid_cell_mask].astype(np.float64)
    matrix_mean = matrix.mean(axis=0)
    centered = matrix - matrix_mean
    print(f"full SWE matrix shape (years, flattened grid): {tuple(full_matrix.shape)}", flush=True)
    print(f"PCA matrix shape after all-year finite-cell mask: {tuple(matrix.shape)}", flush=True)
    print(f"mean-centered matrix column mean max abs: {float(np.abs(centered.mean(axis=0)).max()):.6e}", flush=True)

    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(centered)
    components = pca.components_.astype(np.float32)

    component_maps = _unflatten_valid_columns(
        components,
        valid_cell_mask=valid_cell_mask,
        grid_shape=grid.grid_shape,
    )
    mean_map = _unflatten_valid_columns(
        matrix_mean[None, :].astype(np.float32),
        valid_cell_mask=valid_cell_mask,
        grid_shape=grid.grid_shape,
    )[0]

    metadata = {
        "experiment_name": "swe_yearly_pca_baseline",
        "sample_definition": (
            "One sample is one water year. Each map is SWE_Post selected on the requested "
            "target month-day, using Stats index 0 mean, then cropped/coarsened to the SWE grid."
        ),
        "target_month_day": target_month_day,
        "target_dates": target_dates,
        "years": selected_years,
        "n_components": n_components,
        "swe_variable": "SWE_Post",
        "swe_statistic_name": "mean",
        "swe_statistic_index": 0,
        "missing_value_policy": (
            "Fit PCA only on flattened grid cells finite in every selected year; maps are "
            "reshaped to the full grid with NaN outside that all-year-valid mask."
        ),
        "requested_region": region_to_dict(grid.requested_region),
        "effective_region": region_to_dict(grid.effective_region),
        "coarsen_factor": int(grid.coarsen_factor),
        "grid_shape": list(grid.grid_shape),
        "full_matrix_shape": list(full_matrix.shape),
        "pca_matrix_shape": list(matrix.shape),
        "valid_cell_count": valid_cell_count,
        "latitude_name": grid.latitude_name,
        "longitude_name": grid.longitude_name,
    }

    result = SwePcaResult(
        years=np.asarray(selected_years, dtype=np.int32),
        target_dates=target_dates,
        matrix_shape=tuple(int(size) for size in matrix.shape),
        full_matrix_shape=tuple(int(size) for size in full_matrix.shape),
        grid_shape=tuple(int(size) for size in grid.grid_shape),
        valid_cell_count=valid_cell_count,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        scores=scores.astype(np.float32),
        components=components,
        component_maps=component_maps,
        mean_map=mean_map.astype(np.float32),
        valid_cell_mask=valid_cell_mask.reshape(grid.grid_shape),
        latitude=np.asarray(grid.latitude.values, dtype=np.float32),
        longitude=np.asarray(grid.longitude.values, dtype=np.float32),
        metadata=metadata,
    )
    save_swe_pca_result(result, output_dir)
    return result


def save_swe_pca_result(result: SwePcaResult, output_dir: Path) -> None:
    np.savez_compressed(
        output_dir / "swe_pca_results.npz",
        years=result.years,
        explained_variance_ratio=result.explained_variance_ratio,
        scores=result.scores,
        components=result.components,
        component_maps=result.component_maps,
        mean_map=result.mean_map,
        valid_cell_mask=result.valid_cell_mask.astype(np.uint8),
        latitude=result.latitude,
        longitude=result.longitude,
        metadata_json=np.asarray(json.dumps(result.metadata), dtype=str),
    )
    _write_explained_variance_csv(
        output_dir / "explained_variance_ratio.csv",
        result.explained_variance_ratio,
    )
    _write_scores_csv(output_dir / "pc_scores_by_year.csv", result.years, result.scores)
    (output_dir / "metadata.json").write_text(json.dumps(result.metadata, indent=2, sort_keys=True) + "\n")
    plot_spatial_modes(
        output_dir / "pca_spatial_modes.png",
        result.component_maps,
        latitude=result.latitude,
        longitude=result.longitude,
        explained_variance_ratio=result.explained_variance_ratio,
    )
    plot_scores(output_dir / "pc_scores_by_year.png", result.years, result.scores)
    print(f"saved PCA outputs to {output_dir}", flush=True)


def _unflatten_valid_columns(
    values: np.ndarray,
    *,
    valid_cell_mask: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    output = np.full((values.shape[0], valid_cell_mask.size), np.nan, dtype=np.float32)
    output[:, valid_cell_mask] = values.astype(np.float32)
    return output.reshape((values.shape[0],) + grid_shape)


def _write_explained_variance_csv(path: Path, explained_variance_ratio: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "explained_variance_ratio"])
        for index, value in enumerate(explained_variance_ratio, start=1):
            writer.writerow([index, f"{float(value):.10g}"])


def _write_scores_csv(path: Path, years: np.ndarray, scores: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["water_year"] + [f"PC{index}" for index in range(1, scores.shape[1] + 1)])
        for year, row in zip(years, scores, strict=True):
            writer.writerow([int(year)] + [f"{float(value):.10g}" for value in row])


def plot_spatial_modes(
    path: Path,
    component_maps: np.ndarray,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    explained_variance_ratio: np.ndarray,
) -> None:
    n_components = component_maps.shape[0]
    fig, axes = plt.subplots(1, n_components, figsize=(4.0 * n_components, 4.0), constrained_layout=True)
    if n_components == 1:
        axes = [axes]
    extent = [float(np.nanmin(longitude)), float(np.nanmax(longitude)), float(np.nanmin(latitude)), float(np.nanmax(latitude))]
    for index, ax in enumerate(axes):
        values = component_maps[index]
        finite = values[np.isfinite(values)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        image = ax.imshow(
            values,
            origin="upper",
            extent=extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"PC{index + 1} ({100.0 * explained_variance_ratio[index]:.1f}%)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(path: Path, years: np.ndarray, scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for index in range(scores.shape[1]):
        ax.plot(years, scores[:, index], marker="o", linewidth=1.5, label=f"PC{index + 1}")
    ax.set_xlabel("Water year")
    ax.set_ylabel("PC score")
    ax.legend(ncol=min(scores.shape[1], 5))
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)
