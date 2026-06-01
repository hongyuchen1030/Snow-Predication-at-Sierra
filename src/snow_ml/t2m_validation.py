import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


WUSD3_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily")
WUSD3_WRFINPUT_D01 = Path("/global/cfs/projectdirs/m3522/cmip6/WUS-D3/wrfinput_d01")
ERA5_LAND_T2M_ROOT = Path("/global/cfs/projectdirs/m3522/datalake/ERA5-Land/2m_temperature")
OUTPUT_ROOT = Path("artifacts/t2m_anomaly_validation")
WUS_OUTPUT_DIR = OUTPUT_ROOT / "wus_t2m_march_anomalies"
ERA5_OUTPUT_DIR = OUTPUT_ROOT / "era5_t2m_march_anomalies"
PLOTS_DIR = OUTPUT_ROOT / "plots"
RESULTS_CSV_PATH = OUTPUT_ROOT / "t2m_validation_results.csv"
REPORT_PATH = OUTPUT_ROOT / "t2m_validation_report.md"

TARGET_DOMAIN = "d01"
WUS_VARIABLE = "t2"
ERA5_VARIABLE = "t2m"
LANDMASK_VARIABLE = "LANDMASK"
LANDMASK_LATITUDE = "XLAT"
LANDMASK_LONGITUDE = "XLONG"
ERA5_YEAR_PATTERN = re.compile(r"ERA5_(\d{4})_2m_temperature\.nc$")
HISTORICAL_TOKEN = "_historical_"
MARCH_MONTH = 3
ERA5_MARGIN_DEGREES = 1.0


@dataclass(frozen=True)
class GridDefinition:
    latitude: xr.DataArray
    longitude: xr.DataArray
    landmask: xr.DataArray

    @property
    def latitude_bounds(self) -> Tuple[float, float]:
        values = np.asarray(self.latitude.values, dtype=np.float64)
        return float(np.nanmin(values)), float(np.nanmax(values))

    @property
    def longitude_bounds(self) -> Tuple[float, float]:
        values = np.asarray(self.longitude.values, dtype=np.float64)
        return float(np.nanmin(values)), float(np.nanmax(values))

    @property
    def land_latitude_bounds(self) -> Tuple[float, float]:
        values = np.asarray(self.latitude.values, dtype=np.float64)
        mask = np.asarray(self.landmask.values) == 1
        return float(np.nanmin(values[mask])), float(np.nanmax(values[mask]))

    @property
    def land_longitude_bounds(self) -> Tuple[float, float]:
        values = np.asarray(self.longitude.values, dtype=np.float64)
        mask = np.asarray(self.landmask.values) == 1
        return float(np.nanmin(values[mask])), float(np.nanmax(values[mask]))


@dataclass(frozen=True)
class DatasetAnomalyResult:
    dataset_id: str
    scenario: str
    years: List[int]
    file_path: Path


@dataclass(frozen=True)
class ValidationSummary:
    dataset_id: str
    overlap_years: List[int]
    mean_spatial_correlation: float
    mean_rmse: float
    mean_bias: float


def run_t2m_anomaly_validation() -> None:
    print("starting March t2m anomaly validation", flush=True)
    ensure_output_dirs()
    grid = load_wusd3_d01_grid()
    dataset_ids = discover_wusd3_dataset_ids()
    print("discovered dataset ids:", dataset_ids, flush=True)

    era5_years = discover_era5_years()
    historical_years = sorted(
        year
        for year in era5_years
        if all(
            year in discover_wusd3_t2_years(dataset_id)
            for dataset_id in dataset_ids
            if HISTORICAL_TOKEN in dataset_id
        )
    )
    print("historical overlap years:", historical_years, flush=True)

    era5_anomalies = build_era5_march_anomalies_on_wusd3_grid(
        years=historical_years,
        grid=grid,
    )
    era5_output_path = save_era5_anomalies(era5_anomalies, historical_years)
    print(f"saved ERA5 anomalies to {era5_output_path}", flush=True)

    anomaly_results: List[DatasetAnomalyResult] = []
    validation_rows: List[Dict[str, object]] = []
    validation_summaries: List[ValidationSummary] = []

    for dataset_id in dataset_ids:
        scenario = "historical" if HISTORICAL_TOKEN in dataset_id else "ssp370"
        print(f"processing {dataset_id}", flush=True)
        years = discover_wusd3_t2_years(dataset_id)
        anomalies = build_wusd3_march_anomalies(
            dataset_id=dataset_id,
            years=years,
            grid=grid,
        )
        output_path = save_wusd3_anomalies(dataset_id, scenario, anomalies, years)
        anomaly_results.append(
            DatasetAnomalyResult(
                dataset_id=dataset_id,
                scenario=scenario,
                years=years,
                file_path=output_path,
            )
        )
        print(f"saved WUS anomalies to {output_path}", flush=True)

        if scenario != "historical":
            continue

        overlap_years = [year for year in years if year in historical_years]
        summary = validate_dataset_against_era5(
            dataset_id=dataset_id,
            wus_anomalies=anomalies.sel(year=overlap_years),
            era5_anomalies=era5_anomalies.sel(year=overlap_years),
            grid=grid,
            rows=validation_rows,
        )
        validation_summaries.append(summary)
        create_dataset_plots(
            dataset_id=dataset_id,
            overlap_years=overlap_years,
            wus_anomalies=anomalies.sel(year=overlap_years),
            era5_anomalies=era5_anomalies.sel(year=overlap_years),
            grid=grid,
            summary=summary,
        )

    write_validation_results_csv(validation_rows)
    create_summary_plot(validation_summaries)
    write_validation_report(
        anomaly_results=anomaly_results,
        era5_output_path=era5_output_path,
        validation_summaries=validation_summaries,
        historical_years=historical_years,
    )
    print(f"wrote {RESULTS_CSV_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, WUS_OUTPUT_DIR, ERA5_OUTPUT_DIR, PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def discover_wusd3_dataset_ids() -> List[str]:
    dataset_ids = sorted(path.name for path in WUSD3_ROOT.iterdir() if path.is_dir())
    if not dataset_ids:
        raise FileNotFoundError(f"no WUS-D3 dataset directories found under {WUSD3_ROOT}")
    return dataset_ids


def discover_wusd3_t2_years(dataset_id: str) -> List[int]:
    pattern = re.compile(r"\.d01\.(\d{4})\.nc$")
    paths = sorted((WUSD3_ROOT / dataset_id / "postprocess" / TARGET_DOMAIN).glob("t2.daily.*.nc"))
    years: List[int] = []
    for path in paths:
        match = pattern.search(path.name)
        if match:
            years.append(int(match.group(1)))
    if not years:
        raise FileNotFoundError(f"no WUS-D3 t2 files found for {dataset_id}")
    return years


def discover_era5_years() -> List[int]:
    years: List[int] = []
    for path in sorted(ERA5_LAND_T2M_ROOT.glob("ERA5_*_2m_temperature.nc")):
        match = ERA5_YEAR_PATTERN.fullmatch(path.name)
        if match:
            years.append(int(match.group(1)))
    if not years:
        raise FileNotFoundError(f"no ERA5-Land t2m files found under {ERA5_LAND_T2M_ROOT}")
    return years


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    errors: List[str] = []
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs: Dict[str, object] = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:  # pragma: no cover - debug path
            errors.append(f"{engine or 'default'}: {exc}")
    joined = "; ".join(errors)
    raise RuntimeError(f"failed to open {path} with xarray engines: {joined}")


def load_wusd3_d01_grid() -> GridDefinition:
    print(f"loading WUS d01 grid from {WUSD3_WRFINPUT_D01}", flush=True)
    with open_dataset_with_fallbacks(WUSD3_WRFINPUT_D01) as ds:
        latitude = ds[LANDMASK_LATITUDE].isel(Time=0).rename({"south_north": "lat2d", "west_east": "lon2d"})
        longitude = ds[LANDMASK_LONGITUDE].isel(Time=0).rename({"south_north": "lat2d", "west_east": "lon2d"})
        landmask = ds[LANDMASK_VARIABLE].isel(Time=0).rename({"south_north": "lat2d", "west_east": "lon2d"})
        landmask = xr.where(landmask == 1, 1, 0).astype(np.int8)
    grid = GridDefinition(latitude=latitude.load(), longitude=longitude.load(), landmask=landmask.load())
    print("grid latitude bounds:", tuple(round(value, 3) for value in grid.latitude_bounds), flush=True)
    print("grid longitude bounds:", tuple(round(value, 3) for value in grid.longitude_bounds), flush=True)
    print("land latitude bounds:", tuple(round(value, 3) for value in grid.land_latitude_bounds), flush=True)
    print("land longitude bounds:", tuple(round(value, 3) for value in grid.land_longitude_bounds), flush=True)
    print("land cells:", int(np.count_nonzero(grid.landmask.values == 1)), flush=True)
    return grid


def build_wusd3_march_anomalies(
    dataset_id: str,
    years: List[int],
    grid: GridDefinition,
) -> xr.DataArray:
    fields: List[xr.DataArray] = []
    for year in years:
        field = load_wusd3_march_mean_field(dataset_id, year, grid)
        fields.append(field)
    stacked = xr.concat(fields, dim=xr.IndexVariable("year", years))
    climatology = stacked.mean(dim="year", skipna=True)
    anomalies = (stacked - climatology).astype(np.float32)
    anomalies.name = "wus_t2_march_anomaly"
    anomalies.attrs["units"] = "K"
    anomalies.attrs["description"] = "Monthly March mean 2-meter temperature anomaly on WUS-D3 d01 land cells"
    anomalies = anomalies.assign_coords(latitude=grid.latitude, longitude=grid.longitude, landmask=grid.landmask)
    return anomalies


def build_era5_march_anomalies_on_wusd3_grid(
    years: List[int],
    grid: GridDefinition,
) -> xr.DataArray:
    fields: List[xr.DataArray] = []
    for year in years:
        field = load_era5_march_mean_field_on_wusd3_grid(year, grid)
        fields.append(field)
    stacked = xr.concat(fields, dim=xr.IndexVariable("year", years))
    climatology = stacked.mean(dim="year", skipna=True)
    anomalies = (stacked - climatology).astype(np.float32)
    anomalies.name = "era5_t2m_march_anomaly"
    anomalies.attrs["units"] = "K"
    anomalies.attrs["description"] = "ERA5-Land monthly March mean 2m temperature anomaly regridded to WUS-D3 d01 land cells"
    anomalies = anomalies.assign_coords(latitude=grid.latitude, longitude=grid.longitude, landmask=grid.landmask)
    return anomalies


def load_wusd3_march_mean_field(
    dataset_id: str,
    year: int,
    grid: GridDefinition,
) -> xr.DataArray:
    path = wusd3_t2_file_path(dataset_id, year)
    print(f"loading WUS March mean: {path}", flush=True)
    with open_dataset_with_fallbacks(path) as ds:
        field = ds[WUS_VARIABLE].sel(day=ds["day"].dt.month == MARCH_MONTH).mean(dim="day", skipna=True)
        field = field.rename({"lat2d": "lat2d", "lon2d": "lon2d"}).astype(np.float32).load()
    field = field.where(grid.landmask == 1)
    field.name = "wus_t2_march_mean"
    field.attrs["source_path"] = str(path)
    field.attrs["year"] = int(year)
    field = field.assign_coords(latitude=grid.latitude, longitude=grid.longitude, landmask=grid.landmask)
    return field


def load_era5_march_mean_field_on_wusd3_grid(
    year: int,
    grid: GridDefinition,
) -> xr.DataArray:
    path = ERA5_LAND_T2M_ROOT / f"ERA5_{year}_2m_temperature.nc"
    if not path.exists():
        raise FileNotFoundError(f"ERA5-Land file not found: {path}")
    print(f"loading ERA5 March mean: {path}", flush=True)
    with open_dataset_with_fallbacks(path) as ds:
        subset = subset_era5_to_wusd3_bounds(ds[ERA5_VARIABLE], grid)
        monthly_march = subset.sel(time=subset["time"].dt.month == MARCH_MONTH).mean(dim="time", skipna=True)
        monthly_march = monthly_march.load()
        regridded = regrid_era5_field_to_wusd3_grid(monthly_march, grid)
    regridded = regridded.where(grid.landmask == 1)
    regridded.name = "era5_t2m_march_mean"
    regridded.attrs["source_path"] = str(path)
    regridded.attrs["year"] = int(year)
    return regridded


def subset_era5_to_wusd3_bounds(field: xr.DataArray, grid: GridDefinition) -> xr.DataArray:
    lat_min, lat_max = grid.land_latitude_bounds
    lon_min, lon_max = grid.land_longitude_bounds
    target_lon_min = lon_min % 360.0
    target_lon_max = lon_max % 360.0

    latitude = field["latitude"]
    longitude = field["longitude"]

    lat_slice = slice(lat_max + ERA5_MARGIN_DEGREES, lat_min - ERA5_MARGIN_DEGREES)
    subset = field.sel(latitude=lat_slice)
    if float(subset.sizes["latitude"]) == 0:
        lat_slice = slice(lat_min - ERA5_MARGIN_DEGREES, lat_max + ERA5_MARGIN_DEGREES)
        subset = field.sel(latitude=lat_slice)

    lon_mask = (
        (longitude >= target_lon_min - ERA5_MARGIN_DEGREES)
        & (longitude <= target_lon_max + ERA5_MARGIN_DEGREES)
    )
    subset = subset.sel(longitude=longitude[lon_mask])
    if float(subset.sizes["longitude"]) == 0:
        raise ValueError("ERA5 longitude subset is empty for WUS grid bounds")

    print(
        "ERA5 subset sizes:",
        {
            "time": int(subset.sizes["time"]),
            "latitude": int(subset.sizes["latitude"]),
            "longitude": int(subset.sizes["longitude"]),
        },
        flush=True,
    )
    return subset


def regrid_era5_field_to_wusd3_grid(field: xr.DataArray, grid: GridDefinition) -> xr.DataArray:
    source_latitudes = np.asarray(field["latitude"].values, dtype=np.float64)
    source_longitudes = np.asarray(field["longitude"].values, dtype=np.float64)
    source_values = np.asarray(field.values, dtype=np.float64)

    if source_latitudes[0] > source_latitudes[-1]:
        source_latitudes = source_latitudes[::-1]
        source_values = source_values[::-1, :]

    target_latitudes = np.asarray(grid.latitude.values, dtype=np.float64)
    target_longitudes = np.mod(np.asarray(grid.longitude.values, dtype=np.float64), 360.0)

    interpolator = RegularGridInterpolator(
        (source_latitudes, source_longitudes),
        source_values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack([target_latitudes.reshape(-1), target_longitudes.reshape(-1)])
    values = interpolator(points).reshape(target_latitudes.shape).astype(np.float32)
    return xr.DataArray(
        values,
        dims=("lat2d", "lon2d"),
        coords={
            "latitude": grid.latitude,
            "longitude": grid.longitude,
            "landmask": grid.landmask,
        },
        name="era5_t2m_on_wusd3_grid",
    )


def save_wusd3_anomalies(
    dataset_id: str,
    scenario: str,
    anomalies: xr.DataArray,
    years: List[int],
) -> Path:
    path = WUS_OUTPUT_DIR / f"{dataset_id}_{scenario}_march_t2_anomalies.nc"
    dataset = anomalies.to_dataset(name=anomalies.name)
    dataset.attrs["dataset_id"] = dataset_id
    dataset.attrs["scenario"] = scenario
    dataset.attrs["years"] = ",".join(str(year) for year in years)
    dataset.attrs["landmask_definition"] = "LANDMASK == 1 is land, LANDMASK == 0 is water"
    dataset.to_netcdf(path)
    return path


def save_era5_anomalies(anomalies: xr.DataArray, years: List[int]) -> Path:
    path = ERA5_OUTPUT_DIR / "era5_land_historical_overlap_march_t2m_anomalies_on_wusd3_d01.nc"
    dataset = anomalies.to_dataset(name=anomalies.name)
    dataset.attrs["comparison_years"] = ",".join(str(year) for year in years)
    dataset.attrs["regridding"] = "ERA5-Land bilinear interpolation onto WUS-D3 d01 native grid"
    dataset.attrs["landmask_definition"] = "WUS-D3 LANDMASK == 1 is land, LANDMASK == 0 is water"
    dataset.to_netcdf(path)
    return path


def validate_dataset_against_era5(
    dataset_id: str,
    wus_anomalies: xr.DataArray,
    era5_anomalies: xr.DataArray,
    grid: GridDefinition,
    rows: List[Dict[str, object]],
) -> ValidationSummary:
    correlations: List[float] = []
    rmses: List[float] = []
    biases: List[float] = []

    for year in wus_anomalies["year"].values.tolist():
        wus_field = wus_anomalies.sel(year=year)
        era5_field = era5_anomalies.sel(year=year)
        correlation, rmse, bias = compute_spatial_metrics(wus_field, era5_field, grid.landmask)
        rows.append(
            {
                "dataset_id": dataset_id,
                "scenario": "historical",
                "year": int(year),
                "spatial_correlation": correlation,
                "rmse": rmse,
                "mean_bias": bias,
            }
        )
        correlations.append(correlation)
        rmses.append(rmse)
        biases.append(bias)

    summary = ValidationSummary(
        dataset_id=dataset_id,
        overlap_years=[int(year) for year in wus_anomalies["year"].values.tolist()],
        mean_spatial_correlation=float(np.nanmean(correlations)),
        mean_rmse=float(np.nanmean(rmses)),
        mean_bias=float(np.nanmean(biases)),
    )
    rows.append(
        {
            "dataset_id": dataset_id,
            "scenario": "historical",
            "year": "ALL",
            "spatial_correlation": summary.mean_spatial_correlation,
            "rmse": summary.mean_rmse,
            "mean_bias": summary.mean_bias,
        }
    )
    print(
        f"validation summary {dataset_id}: corr={summary.mean_spatial_correlation:.4f} "
        f"rmse={summary.mean_rmse:.4f} bias={summary.mean_bias:.4f}",
        flush=True,
    )
    return summary


def compute_spatial_metrics(
    wus_field: xr.DataArray,
    era5_field: xr.DataArray,
    landmask: xr.DataArray,
) -> Tuple[float, float, float]:
    wus_values = np.asarray(wus_field.values, dtype=np.float64)
    era5_values = np.asarray(era5_field.values, dtype=np.float64)
    valid = (
        np.asarray(landmask.values) == 1
    ) & np.isfinite(wus_values) & np.isfinite(era5_values)

    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")

    wus_valid = wus_values[valid]
    era5_valid = era5_values[valid]
    if wus_valid.size < 2:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(wus_valid, era5_valid)[0, 1])
    difference = wus_valid - era5_valid
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    bias = float(np.mean(difference))
    return correlation, rmse, bias


def write_validation_results_csv(rows: List[Dict[str, object]]) -> None:
    fieldnames = ["dataset_id", "scenario", "year", "spatial_correlation", "rmse", "mean_bias"]
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_dataset_plots(
    dataset_id: str,
    overlap_years: List[int],
    wus_anomalies: xr.DataArray,
    era5_anomalies: xr.DataArray,
    grid: GridDefinition,
    summary: ValidationSummary,
) -> None:
    example_years = choose_example_years(overlap_years)
    plot_example_maps(dataset_id, example_years, wus_anomalies, era5_anomalies, grid)
    plot_domain_mean_timeseries(dataset_id, overlap_years, wus_anomalies, era5_anomalies, grid)
    plot_histogram_comparison(dataset_id, wus_anomalies, era5_anomalies, grid)
    plot_yearly_metric_series(dataset_id, overlap_years, wus_anomalies, era5_anomalies, grid, summary)


def choose_example_years(years: List[int]) -> List[int]:
    if not years:
        return []
    indices = sorted({0, len(years) // 2, len(years) - 1})
    return [years[index] for index in indices]


def plot_example_maps(
    dataset_id: str,
    years: List[int],
    wus_anomalies: xr.DataArray,
    era5_anomalies: xr.DataArray,
    grid: GridDefinition,
) -> None:
    if not years:
        return
    figure, axes = plt.subplots(len(years), 3, figsize=(15, 4 * len(years)), constrained_layout=True)
    if len(years) == 1:
        axes = np.asarray([axes])

    latitude = np.asarray(grid.latitude.values)
    longitude = np.asarray(grid.longitude.values)
    vmax = max(
        float(np.nanmax(np.abs(wus_anomalies.sel(year=years).values))),
        float(np.nanmax(np.abs(era5_anomalies.sel(year=years).values))),
        0.5,
    )
    for row_index, year in enumerate(years):
        wus_field = wus_anomalies.sel(year=year)
        era5_field = era5_anomalies.sel(year=year)
        diff = wus_field - era5_field
        fields = [wus_field, era5_field, diff]
        titles = [
            f"{dataset_id} WUS anomaly {year}",
            f"ERA5 anomaly {year}",
            f"WUS - ERA5 difference {year}",
        ]
        for col_index, (field, title) in enumerate(zip(fields, titles)):
            axis = axes[row_index, col_index]
            image = axis.pcolormesh(longitude, latitude, field.values, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            figure.colorbar(image, ax=axis, shrink=0.85, label="K")
    figure.savefig(PLOTS_DIR / f"{dataset_id}_anomaly_examples.png", dpi=150)
    plt.close(figure)


def plot_domain_mean_timeseries(
    dataset_id: str,
    years: List[int],
    wus_anomalies: xr.DataArray,
    era5_anomalies: xr.DataArray,
    grid: GridDefinition,
) -> None:
    wus_series = compute_domain_mean_series(wus_anomalies, grid.landmask)
    era5_series = compute_domain_mean_series(era5_anomalies, grid.landmask)
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.plot(years, wus_series, marker="o", label=f"{dataset_id} WUS")
    axis.plot(years, era5_series, marker="o", label="ERA5-Land")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(f"March domain-mean anomaly comparison: {dataset_id}")
    axis.set_xlabel("Year")
    axis.set_ylabel("K")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(PLOTS_DIR / f"{dataset_id}_yearly_anomaly_comparison.png", dpi=150)
    plt.close(figure)


def plot_histogram_comparison(
    dataset_id: str,
    wus_anomalies: xr.DataArray,
    era5_anomalies: xr.DataArray,
    grid: GridDefinition,
) -> None:
    wus_values = flatten_land_values(wus_anomalies, grid.landmask)
    era5_values = flatten_land_values(era5_anomalies, grid.landmask)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.hist(wus_values, bins=40, alpha=0.6, density=True, label=f"{dataset_id} WUS")
    axis.hist(era5_values, bins=40, alpha=0.6, density=True, label="ERA5-Land")
    axis.set_title(f"March anomaly histogram comparison: {dataset_id}")
    axis.set_xlabel("K")
    axis.set_ylabel("Density")
    axis.legend()
    figure.savefig(PLOTS_DIR / f"{dataset_id}_histogram_comparison.png", dpi=150)
    plt.close(figure)


def plot_yearly_metric_series(
    dataset_id: str,
    years: List[int],
    wus_anomalies: xr.DataArray,
    era5_anomalies: xr.DataArray,
    grid: GridDefinition,
    summary: ValidationSummary,
) -> None:
    correlations: List[float] = []
    rmses: List[float] = []
    biases: List[float] = []
    for year in years:
        correlation, rmse, bias = compute_spatial_metrics(
            wus_anomalies.sel(year=year),
            era5_anomalies.sel(year=year),
            grid.landmask,
        )
        correlations.append(correlation)
        rmses.append(rmse)
        biases.append(bias)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(years, correlations, marker="o")
    axes[0].axhline(summary.mean_spatial_correlation, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Correlation")
    axes[0].set_title(f"Yearly spatial metrics: {dataset_id}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(years, rmses, marker="o", color="tab:red")
    axes[1].axhline(summary.mean_rmse, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("RMSE (K)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(years, biases, marker="o", color="tab:green")
    axes[2].axhline(summary.mean_bias, color="black", linestyle="--", linewidth=0.9)
    axes[2].set_ylabel("Bias (K)")
    axes[2].set_xlabel("Year")
    axes[2].grid(True, alpha=0.3)

    figure.savefig(PLOTS_DIR / f"{dataset_id}_metric_summary.png", dpi=150)
    plt.close(figure)


def create_summary_plot(validation_summaries: List[ValidationSummary]) -> None:
    if not validation_summaries:
        return
    dataset_ids = [summary.dataset_id for summary in validation_summaries]
    correlations = [summary.mean_spatial_correlation for summary in validation_summaries]
    rmses = [summary.mean_rmse for summary in validation_summaries]
    biases = [summary.mean_bias for summary in validation_summaries]

    x = np.arange(len(dataset_ids))
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].bar(x, correlations, color="tab:blue")
    axes[0].set_ylabel("Mean correlation")
    axes[0].set_title("Historical March anomaly validation summary")

    axes[1].bar(x, rmses, color="tab:red")
    axes[1].set_ylabel("Mean RMSE (K)")

    axes[2].bar(x, biases, color="tab:green")
    axes[2].set_ylabel("Mean bias (K)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dataset_ids, rotation=20, ha="right")

    figure.savefig(PLOTS_DIR / "historical_correlation_summary.png", dpi=150)
    plt.close(figure)


def write_validation_report(
    anomaly_results: List[DatasetAnomalyResult],
    era5_output_path: Path,
    validation_summaries: List[ValidationSummary],
    historical_years: List[int],
) -> None:
    lines: List[str] = []
    lines.append("# T2M anomaly validation report")
    lines.append("")
    lines.append("## Years processed")
    lines.append("")
    lines.append(f"- Historical overlap comparison years: {historical_years[0]}-{historical_years[-1]} ({len(historical_years)} years)")
    for result in anomaly_results:
        lines.append(
            f"- {result.dataset_id}: {result.years[0]}-{result.years[-1]} "
            f"({len(result.years)} years), saved to `{result.file_path}`"
        )
    lines.append(f"- ERA5 historical-overlap anomalies saved to `{era5_output_path}`")
    lines.append("")
    lines.append("## Anomaly methodology")
    lines.append("")
    lines.append("- WUS-D3 d01 variable: `t2`, daily data aggregated to one March mean field per year.")
    lines.append("- ERA5-Land variable: `t2m`, hourly data aggregated to one March mean field per year.")
    lines.append("- ERA5-Land was subset to the WUS d01 latitude/longitude envelope, then bilinearly interpolated onto the WUS d01 native grid.")
    lines.append("- LANDMASK from `wrfinput_d01` was applied after regridding. Only cells with `LANDMASK == 1` were retained.")
    lines.append("- For each dataset separately, March climatology was computed as the mean March field across all available years in that dataset, then yearly anomalies were computed as March mean minus climatology.")
    lines.append("- Historical ERA5 comparison used the shared overlap years across all historical WUS datasets.")
    lines.append("")
    lines.append("## Validation summary")
    lines.append("")
    for summary in validation_summaries:
        consistency = classify_consistency(summary)
        lines.append(
            f"- {summary.dataset_id}: mean spatial correlation={summary.mean_spatial_correlation:.3f}, "
            f"mean RMSE={summary.mean_rmse:.3f} K, mean bias={summary.mean_bias:.3f} K. "
            f"Assessment: {consistency}."
        )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if validation_summaries:
        mean_corr = float(np.nanmean([summary.mean_spatial_correlation for summary in validation_summaries]))
        mean_rmse = float(np.nanmean([summary.mean_rmse for summary in validation_summaries]))
        mean_bias = float(np.nanmean([summary.mean_bias for summary in validation_summaries]))
        lines.append(
            f"Across the historical models, the mean spatial correlation is {mean_corr:.3f}, "
            f"the mean RMSE is {mean_rmse:.3f} K, and the mean bias is {mean_bias:.3f} K."
        )
        lines.append(
            f"Based on these diagnostics, the March WUS-D3 t2 anomalies are "
            f"{classify_overall_suitability(mean_corr, mean_rmse, mean_bias)} "
            f"for the SST -> t2m prediction experiment."
        )
    else:
        lines.append("No historical validation summaries were generated.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_consistency(summary: ValidationSummary) -> str:
    if summary.mean_spatial_correlation >= 0.7 and summary.mean_rmse <= 2.0 and abs(summary.mean_bias) <= 0.5:
        return "strong historical consistency with ERA5-Land"
    if summary.mean_spatial_correlation >= 0.5 and summary.mean_rmse <= 3.0 and abs(summary.mean_bias) <= 1.0:
        return "reasonable historical consistency with ERA5-Land"
    return "weak or biased historical consistency; inspect the anomaly maps and yearly diagnostics carefully"


def classify_overall_suitability(mean_corr: float, mean_rmse: float, mean_bias: float) -> str:
    if mean_corr >= 0.7 and mean_rmse <= 2.0 and abs(mean_bias) <= 0.5:
        return "well aligned and suitable"
    if mean_corr >= 0.5 and mean_rmse <= 3.0 and abs(mean_bias) <= 1.0:
        return "good enough for a baseline experiment, with some caution"
    return "not yet reliable enough without additional bias and structure checks"


def compute_domain_mean_series(fields: xr.DataArray, landmask: xr.DataArray) -> np.ndarray:
    values = np.asarray(fields.values, dtype=np.float64)
    mask = np.asarray(landmask.values) == 1
    series: List[float] = []
    for index in range(values.shape[0]):
        valid = mask & np.isfinite(values[index])
        if not np.any(valid):
            series.append(float("nan"))
        else:
            series.append(float(np.mean(values[index][valid])))
    return np.asarray(series, dtype=np.float64)


def flatten_land_values(fields: xr.DataArray, landmask: xr.DataArray) -> np.ndarray:
    values = np.asarray(fields.values, dtype=np.float64)
    mask = np.asarray(landmask.values) == 1
    flat = values[:, mask]
    return flat[np.isfinite(flat)]


def wusd3_t2_file_path(dataset_id: str, year: int) -> Path:
    base_dir = WUSD3_ROOT / dataset_id / "postprocess" / TARGET_DOMAIN
    matches = sorted(base_dir.glob(f"t2.daily.*.{TARGET_DOMAIN}.{year:04d}.nc"))
    if not matches:
        raise FileNotFoundError(f"no WUS-D3 t2 file found for {dataset_id} {year}")
    if len(matches) > 1:
        raise RuntimeError(f"expected one WUS-D3 t2 file for {dataset_id} {year}, found {len(matches)}")
    return matches[0]
