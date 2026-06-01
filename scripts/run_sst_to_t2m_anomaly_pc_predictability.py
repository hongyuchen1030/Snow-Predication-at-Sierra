#!/usr/bin/env python3
"""
Predict March overland T2M anomaly PCs from February SST anomaly PCs.

Workflow:
1. Load anomaly-based COBE2 SST EOFs from artifacts/sst_pca/cobe2/.
2. Project each model SST anomaly field onto the COBE2 anomaly EOF basis.
3. Load saved March T2M anomaly cubes on WUS-D3 d01 land cells.
4. Compute land-only March T2M EOFs / PCs from anomaly fields.
5. Run leave-one-year-out ridge regression to predict T2M_PC1/2 from SST_PC1/2.
6. Save metrics, timeseries, SST/T2M PC artifacts, plots, and a markdown report.
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


COBE2_EOF_FILE = PROJECT_ROOT / "artifacts" / "sst_pca" / "cobe2" / "cobe2_sst_eofs.nc"
MODEL_SST_ANOMALY_ROOT = PROJECT_ROOT / "artifacts" / "sst_pca" / "model_sst_anomalies"
MODEL_SST_ANOM_PC_DIR = PROJECT_ROOT / "artifacts" / "sst_pca" / "model_sst_anomaly_pcs"
T2M_ANOMALY_DIR = PROJECT_ROOT / "artifacts" / "t2m_anomaly_validation" / "wus_t2m_march_anomalies"
WRFINPUT_D01 = Path("/global/cfs/projectdirs/m3522/cmip6/WUS-D3/wrfinput_d01")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sst_to_t2m_anomaly_pc_predictability"
OUTPUT_DIR = Path(os.environ.get("SST_TO_T2M_ANOM_PC_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
PLOTS_DIR = OUTPUT_DIR / "plots"
EOFS_DIR = OUTPUT_DIR / "EOFs"
PCS_DIR = OUTPUT_DIR / "PCs"

RESULTS_FILE = OUTPUT_DIR / "prediction_results.csv"
TIMESERIES_FILE = OUTPUT_DIR / "prediction_timeseries.csv"
REPORT_FILE = OUTPUT_DIR / "prediction_report.md"
SST_PROJECTION_SUMMARY_FILE = OUTPUT_DIR / "sst_anomaly_projection_summary.json"

PREVIOUS_RAW_RESULTS_FILE = (
    PROJECT_ROOT / "artifacts" / "sst_to_t2m_pc_predictability" / "prediction_results.csv"
)

RIDGE_ALPHA = 1.0
SST_MONTH = 2
TARGET_MONTH = 3
N_SST_COMPONENTS = 5
N_TARGET_COMPONENTS = 2
HISTORICAL_SUFFIX = "_historical_bc"
ANOMALY_SUFFIX = "_historical_march_t2_anomalies.nc"


@dataclass(frozen=True)
class RuntimeInfo:
    hostname: str
    slurm_job_id: str


@dataclass(frozen=True)
class SstEofReference:
    latitude: np.ndarray
    longitude: np.ndarray
    eofs: np.ndarray
    valid_mask: np.ndarray
    denominators: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class T2mPcaResult:
    years: List[int]
    pcs: np.ndarray
    eofs: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    valid_mask: np.ndarray
    land_mask: np.ndarray
    grid_shape: Tuple[int, int]
    latitude: np.ndarray
    longitude: np.ndarray


@dataclass(frozen=True)
class FoldPrediction:
    model: str
    year: int
    target_pc: str
    feature_time: date
    target_month: str
    sst_pc1: float
    sst_pc2: float
    true_value: float
    predicted_value: float


def parse_iso_date(text: str) -> date:
    year_text, month_text, day_text = text.split("-")
    return date(int(year_text), int(month_text), int(day_text))


def get_runtime() -> RuntimeInfo:
    return RuntimeInfo(
        hostname=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )


def ensure_runtime_on_compute_node(runtime: RuntimeInfo) -> None:
    if not runtime.slurm_job_id or "nid" not in runtime.hostname:
        raise RuntimeError(
            "Do not run this script on a login node; active interactive compute allocation required."
        )


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, PLOTS_DIR, EOFS_DIR, PCS_DIR, MODEL_SST_ANOM_PC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def open_dataset_with_fallbacks(path: Path) -> xr.Dataset:
    errors: List[str] = []
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs: Dict[str, object] = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}: {exc}")
    raise RuntimeError(f"failed to open {path}: {'; '.join(errors)}")


def discover_historical_models() -> List[str]:
    models: List[str] = []
    for path in sorted(T2M_ANOMALY_DIR.glob(f"*{HISTORICAL_SUFFIX}{ANOMALY_SUFFIX}")):
        model_name = path.name[: -len(ANOMALY_SUFFIX)]
        models.append(model_name)
    if not models:
        raise FileNotFoundError(f"No historical March t2 anomaly files found in {T2M_ANOMALY_DIR}")
    return models


def load_wrf_landmask() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset_with_fallbacks(WRFINPUT_D01) as ds:
        landmask = np.asarray(ds["LANDMASK"].isel(Time=0).values, dtype=np.int8)
        latitude = np.asarray(ds["XLAT"].isel(Time=0).values, dtype=np.float32)
        longitude = np.asarray(ds["XLONG"].isel(Time=0).values, dtype=np.float32)
    return landmask == 1, latitude, longitude


def load_sst_eof_reference() -> SstEofReference:
    with open_dataset_with_fallbacks(COBE2_EOF_FILE) as ds:
        eofs = np.asarray(ds["eof"].values[:N_SST_COMPONENTS], dtype=np.float64)
        valid_mask = np.asarray(ds["valid_cell_mask"].values, dtype=bool)
        latitude = np.asarray(ds["lat"].values, dtype=np.float64)
        longitude = np.asarray(ds["lon"].values, dtype=np.float64)
        explained_variance_ratio = np.asarray(
            ds.attrs.get("explained_variance_ratio", [float("nan")] * N_SST_COMPONENTS),
            dtype=np.float64,
        )[:N_SST_COMPONENTS]

    if eofs.shape[1:] != valid_mask.shape:
        raise ValueError(f"EOF grid {eofs.shape[1:]} does not match valid mask {valid_mask.shape}")
    denominators = np.array(
        [float(np.dot(eofs[index, valid_mask], eofs[index, valid_mask])) for index in range(N_SST_COMPONENTS)],
        dtype=np.float64,
    )
    if np.any(denominators == 0.0):
        raise ValueError(f"Encountered zero EOF denominator(s): {denominators.tolist()}")
    return SstEofReference(
        latitude=latitude,
        longitude=longitude,
        eofs=eofs,
        valid_mask=valid_mask,
        denominators=denominators,
        explained_variance_ratio=explained_variance_ratio,
    )


def detect_sst_anomaly_var(dataset: xr.Dataset) -> str:
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if "time" in dims and "lat" in dims and "lon" in dims:
            return name
    raise ValueError(f"Could not find SST anomaly variable in {list(dataset.data_vars)}")


def project_sst_anomaly_pcs(
    data_3d: np.ndarray,
    reference: SstEofReference,
) -> np.ndarray:
    if data_3d.ndim != 3:
        raise ValueError(f"Expected data with 3 dimensions (time, lat, lon), got shape {data_3d.shape}")
    if data_3d.shape[1:] != reference.valid_mask.shape:
        raise ValueError(
            f"Grid shape mismatch: expected {reference.valid_mask.shape}, got {data_3d.shape[1:]}"
        )

    values = np.asarray(data_3d[:, reference.valid_mask], dtype=np.float64)
    if not np.isfinite(values).all():
        bad_count = int(np.size(values) - np.isfinite(values).sum())
        raise ValueError(f"Projection input contains {bad_count} non-finite values on the valid anomaly mask")

    pcs = np.empty((data_3d.shape[0], N_SST_COMPONENTS), dtype=np.float64)
    for mode_index in range(N_SST_COMPONENTS):
        eof_values = reference.eofs[mode_index, reference.valid_mask]
        pcs[:, mode_index] = (values @ eof_values) / reference.denominators[mode_index]
    return pcs


def write_sst_pc_csv(path: Path, times: Sequence[str], pcs: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "SST_PC1", "SST_PC2", "SST_PC3", "SST_PC4", "SST_PC5"])
        for time_value, row in zip(times, pcs):
            writer.writerow([time_value] + ["{:.12g}".format(float(value)) for value in row[:N_SST_COMPONENTS]])


def build_or_reuse_model_sst_anomaly_pcs(
    model_name: str,
    reference: SstEofReference,
) -> Dict[str, object]:
    output_path = MODEL_SST_ANOM_PC_DIR / f"{model_name}_sst_anomaly_pcs.csv"
    input_path = MODEL_SST_ANOMALY_ROOT / model_name / f"{model_name}_tskin_anomaly.nc"
    reused = output_path.exists()

    with open_dataset_with_fallbacks(input_path) as ds:
        anomaly_var = detect_sst_anomaly_var(ds)
        anomaly = ds[anomaly_var].load()
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        if not np.array_equal(lat, reference.latitude):
            raise ValueError(f"SST anomaly latitude grid does not match COBE2 EOF grid for {model_name}")
        if not np.array_equal(lon, reference.longitude):
            raise ValueError(f"SST anomaly longitude grid does not match COBE2 EOF grid for {model_name}")
        time_values = [
            np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D")
            for value in np.asarray(ds["time"].values)
        ]

    pcs = project_sst_anomaly_pcs(np.asarray(anomaly.values, dtype=np.float64), reference)
    write_sst_pc_csv(output_path, time_values, pcs)
    return {
        "model": model_name,
        "input_file": str(input_path),
        "output_csv": str(output_path),
        "reused_existing_path": reused,
        "n_monthly_samples": int(pcs.shape[0]),
        "time_start": time_values[0],
        "time_end": time_values[-1],
        "pc_ranges": [
            {
                "component": f"SST_PC{mode_index + 1}",
                "min": float(pcs[:, mode_index].min()),
                "max": float(pcs[:, mode_index].max()),
            }
            for mode_index in range(N_SST_COMPONENTS)
        ],
    }


def load_february_sst_samples(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    samples: List[Dict[str, object]] = []
    for row in rows:
        timestamp = parse_iso_date(row["time"])
        if timestamp.month != SST_MONTH:
            continue
        samples.append(
            {
                "year": int(timestamp.year),
                "time": timestamp,
                "features": np.array(
                    [float(row["SST_PC1"]), float(row["SST_PC2"])],
                    dtype=np.float64,
                ),
            }
        )
    if not samples:
        raise ValueError(f"No February SST rows found in {path}")
    return samples


def detect_spatial_anomaly_var(dataset: xr.Dataset) -> str:
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if "year" in dims and "lat2d" in dims and "lon2d" in dims:
            return name
    raise ValueError(f"Could not find anomaly variable with dims (year, lat2d, lon2d) in {list(dataset.data_vars)}")


def compute_t2m_pca(anomaly_path: Path, wrf_land_mask: np.ndarray) -> T2mPcaResult:
    with open_dataset_with_fallbacks(anomaly_path) as ds:
        anomaly_var = detect_spatial_anomaly_var(ds)
        anomalies = ds[anomaly_var].load()
        years = [int(value) for value in anomalies["year"].values.tolist()]
        if "landmask" in ds.coords:
            anomaly_landmask = np.asarray(ds["landmask"].values, dtype=np.int8) == 1
        elif "landmask" in ds.data_vars:
            anomaly_landmask = np.asarray(ds["landmask"].values, dtype=np.int8) == 1
        else:
            anomaly_landmask = wrf_land_mask
        latitude = np.asarray(ds["latitude"].values, dtype=np.float32)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float32)

    values = np.asarray(anomalies.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected anomaly cube (year, lat2d, lon2d), got {values.shape}")
    if values.shape[1:] != wrf_land_mask.shape:
        raise ValueError(f"Anomaly grid {values.shape[1:]} does not match WRF land mask {wrf_land_mask.shape}")

    land_mask = wrf_land_mask & anomaly_landmask
    flat = values.reshape(values.shape[0], -1)
    valid_mask = land_mask.reshape(-1) & np.isfinite(flat).all(axis=0)
    valid_count = int(valid_mask.sum())
    if valid_count < N_TARGET_COMPONENTS:
        raise ValueError(f"Need at least {N_TARGET_COMPONENTS} valid land cells, got {valid_count}")

    matrix = flat[:, valid_mask]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    pcs = (u_matrix[:, :N_TARGET_COMPONENTS] * singular_values[:N_TARGET_COMPONENTS]).astype(np.float64)
    eofs = vt_matrix[:N_TARGET_COMPONENTS, :].astype(np.float64)
    variance = singular_values ** 2
    explained_variance_ratio = (variance / variance.sum()).astype(np.float64)

    return T2mPcaResult(
        years=years,
        pcs=pcs,
        eofs=eofs,
        explained_variance_ratio=explained_variance_ratio,
        singular_values=singular_values.astype(np.float64),
        valid_mask=valid_mask,
        land_mask=land_mask,
        grid_shape=(int(values.shape[1]), int(values.shape[2])),
        latitude=latitude,
        longitude=longitude,
    )


def build_samples(
    model_name: str,
    sst_samples: Sequence[Dict[str, object]],
    t2m_pca: T2mPcaResult,
) -> List[Dict[str, object]]:
    features_by_year = {int(row["year"]): row for row in sst_samples}
    samples: List[Dict[str, object]] = []
    for index, year in enumerate(t2m_pca.years):
        feature_row = features_by_year.get(int(year))
        if feature_row is None:
            continue
        samples.append(
            {
                "model": model_name,
                "year": int(year),
                "feature_time": feature_row["time"],
                "target_month": f"{year:04d}-{TARGET_MONTH:02d}",
                "features": np.asarray(feature_row["features"], dtype=np.float64),
                "targets": t2m_pca.pcs[index, :N_TARGET_COMPONENTS].astype(np.float64),
            }
        )
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 overlapping years for {model_name}, got {len(samples)}")
    return samples


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def run_leave_one_year_out(samples: Sequence[Dict[str, object]]) -> List[FoldPrediction]:
    years = sorted({int(sample["year"]) for sample in samples})
    rows: List[FoldPrediction] = []

    for target_index in range(N_TARGET_COMPONENTS):
        target_pc_name = f"T2M_PC{target_index + 1}"
        for test_year in years:
            train_samples = [sample for sample in samples if int(sample["year"]) != test_year]
            test_samples = [sample for sample in samples if int(sample["year"]) == test_year]
            if not train_samples or not test_samples:
                continue

            x_train = np.stack([sample["features"] for sample in train_samples], axis=0)
            y_train = np.array(
                [float(sample["targets"][target_index]) for sample in train_samples],
                dtype=np.float64,
            )
            x_test = np.stack([sample["features"] for sample in test_samples], axis=0)

            x_mean = x_train.mean(axis=0)
            x_std = x_train.std(axis=0, ddof=0)
            x_std = np.where(x_std == 0.0, 1.0, x_std)

            x_train_scaled = (x_train - x_mean) / x_std
            x_test_scaled = (x_test - x_mean) / x_std

            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)

            for sample, pred_value in zip(test_samples, y_pred):
                rows.append(
                    FoldPrediction(
                        model=str(sample["model"]),
                        year=int(sample["year"]),
                        target_pc=target_pc_name,
                        feature_time=sample["feature_time"],
                        target_month=str(sample["target_month"]),
                        sst_pc1=float(sample["features"][0]),
                        sst_pc2=float(sample["features"][1]),
                        true_value=float(sample["targets"][target_index]),
                        predicted_value=float(pred_value),
                    )
                )
    return rows


def summarize_predictions(
    predictions: Sequence[FoldPrediction],
    explained_variance: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model_names = sorted({prediction.model for prediction in predictions})

    for model_name in model_names + ["ALL"]:
        for target_pc in ("T2M_PC1", "T2M_PC2"):
            selected = [
                prediction
                for prediction in predictions
                if prediction.target_pc == target_pc and (model_name == "ALL" or prediction.model == model_name)
            ]
            if not selected:
                continue
            y_true = np.array([prediction.true_value for prediction in selected], dtype=np.float64)
            y_pred = np.array([prediction.predicted_value for prediction in selected], dtype=np.float64)
            component_index = int(target_pc[-1]) - 1
            rows.append(
                {
                    "model": model_name,
                    "target_pc": target_pc,
                    "n_predictions": int(len(selected)),
                    "n_years": int(len({prediction.year for prediction in selected})),
                    "correlation": correlation(y_true, y_pred),
                    "rmse": rmse(y_true, y_pred),
                    "r2": r2_score_manual(y_true, y_pred),
                    "explained_variance_ratio": (
                        float("nan")
                        if model_name == "ALL"
                        else float(explained_variance[model_name][component_index])
                    ),
                }
            )
    return rows


def write_prediction_results(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "target_pc",
                "n_predictions",
                "n_years",
                "correlation",
                "rmse",
                "r2",
                "explained_variance_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["target_pc"],
                    row["n_predictions"],
                    row["n_years"],
                    "{:.12g}".format(float(row["correlation"])) if np.isfinite(float(row["correlation"])) else "nan",
                    "{:.12g}".format(float(row["rmse"])),
                    "{:.12g}".format(float(row["r2"])) if np.isfinite(float(row["r2"])) else "nan",
                    (
                        "{:.12g}".format(float(row["explained_variance_ratio"]))
                        if np.isfinite(float(row["explained_variance_ratio"]))
                        else "nan"
                    ),
                ]
            )


def write_prediction_timeseries(path: Path, rows: Sequence[FoldPrediction]) -> None:
    sorted_rows = sorted(rows, key=lambda row: (row.model, row.target_pc, row.year))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "year",
                "target_pc",
                "feature_time",
                "target_month",
                "SST_PC1",
                "SST_PC2",
                "true_value",
                "predicted_value",
                "residual",
            ]
        )
        for row in sorted_rows:
            writer.writerow(
                [
                    row.model,
                    row.year,
                    row.target_pc,
                    row.feature_time.isoformat(),
                    row.target_month,
                    "{:.12g}".format(row.sst_pc1),
                    "{:.12g}".format(row.sst_pc2),
                    "{:.12g}".format(row.true_value),
                    "{:.12g}".format(row.predicted_value),
                    "{:.12g}".format(row.predicted_value - row.true_value),
                ]
            )


def save_model_pcs(path: Path, years: Sequence[int], pcs: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "T2M_PC1", "T2M_PC2"])
        for year, values in zip(years, pcs[:, :N_TARGET_COMPONENTS]):
            writer.writerow(
                [
                    int(year),
                    "{:.12g}".format(float(values[0])),
                    "{:.12g}".format(float(values[1])),
                ]
            )


def save_model_eofs(path: Path, result: T2mPcaResult) -> None:
    eof_cube = np.full((N_TARGET_COMPONENTS, result.valid_mask.shape[0]), np.nan, dtype=np.float32)
    eof_cube[:, result.valid_mask] = result.eofs[:N_TARGET_COMPONENTS, :].astype(np.float32)
    eof_cube = eof_cube.reshape((N_TARGET_COMPONENTS,) + result.grid_shape)

    dataset = xr.Dataset(
        data_vars={
            "t2m_eof": (
                ("component", "lat2d", "lon2d"),
                eof_cube,
                {
                    "description": "Land-only March T2M anomaly EOFs from SVD of anomaly fields",
                    "component_definition": "Right singular vectors for valid land cells",
                },
            ),
            "explained_variance_ratio": (
                ("component_full",),
                result.explained_variance_ratio.astype(np.float32),
            ),
            "singular_values": (
                ("component_full",),
                result.singular_values.astype(np.float32),
            ),
            "landmask": (
                ("lat2d", "lon2d"),
                result.land_mask.astype(np.int8),
            ),
        },
        coords={
            "component": np.arange(1, N_TARGET_COMPONENTS + 1, dtype=np.int32),
            "component_full": np.arange(1, result.explained_variance_ratio.shape[0] + 1, dtype=np.int32),
            "year": np.array(result.years, dtype=np.int32),
            "latitude": (("lat2d", "lon2d"), result.latitude.astype(np.float32)),
            "longitude": (("lat2d", "lon2d"), result.longitude.astype(np.float32)),
        },
        attrs={
            "pca_method": "SVD on March mean t2 anomaly fields over all-year-finite land cells",
            "input_definition": "WUS-D3 d01 March mean t2 anomalies, land cells only",
        },
    )
    dataset.to_netcdf(path)


def plot_model_timeseries(model_name: str, rows: Sequence[FoldPrediction], target_pc: str, path: Path) -> None:
    selected = sorted(
        [row for row in rows if row.model == model_name and row.target_pc == target_pc],
        key=lambda row: row.year,
    )
    years = [row.year for row in selected]
    truth = np.array([row.true_value for row in selected], dtype=np.float64)
    pred = np.array([row.predicted_value for row in selected], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, truth, label=f"True {target_pc}", linewidth=1.6)
    ax.plot(years, pred, label=f"Predicted {target_pc}", linewidth=1.3)
    ax.set_xlabel("Year")
    ax.set_ylabel(target_pc)
    ax.set_title(f"{model_name} February SST anomaly -> March {target_pc}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pooled_timeseries(rows: Sequence[FoldPrediction], target_pc: str, path: Path) -> None:
    selected = sorted(
        [row for row in rows if row.target_pc == target_pc],
        key=lambda row: (row.model, row.year),
    )
    model_names = sorted({row.model for row in selected})
    fig, axes = plt.subplots(
        len(model_names),
        1,
        figsize=(11, max(3.2 * len(model_names), 4.0)),
        sharex=True,
        constrained_layout=True,
    )
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        model_rows = [row for row in selected if row.model == model_name]
        years = [row.year for row in model_rows]
        truth = np.array([row.true_value for row in model_rows], dtype=np.float64)
        pred = np.array([row.predicted_value for row in model_rows], dtype=np.float64)
        corr = correlation(truth, pred)
        ax.plot(years, truth, label="True", linewidth=1.6)
        ax.plot(years, pred, label="Predicted", linewidth=1.3)
        ax.set_ylabel(target_pc)
        ax.set_title(f"{model_name}\ncorr={corr:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Year")
    fig.suptitle(f"February SST anomaly -> March {target_pc} temporal evolution by model")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_scatter(model_name: str, rows: Sequence[FoldPrediction], target_pc: str, path: Path) -> None:
    selected = [row for row in rows if row.model == model_name and row.target_pc == target_pc]
    truth = np.array([row.true_value for row in selected], dtype=np.float64)
    pred = np.array([row.predicted_value for row in selected], dtype=np.float64)
    corr = correlation(truth, pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(truth, pred, s=28, alpha=0.8)
    lower = float(min(truth.min(), pred.min()))
    upper = float(max(truth.max(), pred.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel(f"True {target_pc}")
    ax.set_ylabel(f"Predicted {target_pc}")
    ax.set_title(f"{model_name} {target_pc} scatter\ncorr={corr:.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_eofs(model_name: str, result: T2mPcaResult, path: Path) -> None:
    eof_cube = np.full((N_TARGET_COMPONENTS, result.valid_mask.shape[0]), np.nan, dtype=np.float32)
    eof_cube[:, result.valid_mask] = result.eofs[:N_TARGET_COMPONENTS, :].astype(np.float32)
    eof_cube = eof_cube.reshape((N_TARGET_COMPONENTS,) + result.grid_shape)

    fig, axes = plt.subplots(1, N_TARGET_COMPONENTS, figsize=(12, 4.5), constrained_layout=True)
    if N_TARGET_COMPONENTS == 1:
        axes = [axes]
    for index, ax in enumerate(axes):
        panel = eof_cube[index]
        vmax = float(np.nanmax(np.abs(panel)))
        vmax = 1.0 if vmax == 0.0 or not np.isfinite(vmax) else vmax
        image = ax.imshow(panel, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(f"EOF {index + 1}\nEVR={result.explained_variance_ratio[index]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model_name} March T2M anomaly EOFs")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_summary(summary_rows: Sequence[Dict[str, object]], path: Path) -> None:
    selected = [row for row in summary_rows if row["model"] != "ALL"]
    model_names = sorted({str(row["model"]) for row in selected})
    pc1 = [
        float(next(row["correlation"] for row in selected if row["model"] == model_name and row["target_pc"] == "T2M_PC1"))
        for model_name in model_names
    ]
    pc2 = [
        float(next(row["correlation"] for row in selected if row["model"] == model_name and row["target_pc"] == "T2M_PC2"))
        for model_name in model_names
    ]

    x = np.arange(len(model_names), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, pc1, width=width, label="T2M_PC1")
    ax.bar(x + width / 2, pc2, width=width, label="T2M_PC2")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Correlation")
    ax.set_title("February SST anomaly -> March T2M PC correlation by model")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_r2_summary(summary_rows: Sequence[Dict[str, object]], path: Path) -> None:
    selected = [row for row in summary_rows if row["model"] != "ALL"]
    model_names = sorted({str(row["model"]) for row in selected})
    pc1 = [
        float(next(row["r2"] for row in selected if row["model"] == model_name and row["target_pc"] == "T2M_PC1"))
        for model_name in model_names
    ]
    pc2 = [
        float(next(row["r2"] for row in selected if row["model"] == model_name and row["target_pc"] == "T2M_PC2"))
        for model_name in model_names
    ]

    x = np.arange(len(model_names), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, pc1, width=width, label="T2M_PC1")
    ax.bar(x + width / 2, pc2, width=width, label="T2M_PC2")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("February SST anomaly -> March T2M PC R2 by model")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_previous_raw_summary() -> Dict[Tuple[str, str], Dict[str, float]]:
    if not PREVIOUS_RAW_RESULTS_FILE.exists():
        return {}
    with PREVIOUS_RAW_RESULTS_FILE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        summary[(row["model"], row["target_pc"])] = {
            "correlation": float(row["correlation"]) if row["correlation"] != "nan" else float("nan"),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]) if row["r2"] != "nan" else float("nan"),
        }
    return summary


def write_report(
    path: Path,
    runtime: RuntimeInfo,
    models: Sequence[str],
    explained_variance: Dict[str, np.ndarray],
    summary_rows: Sequence[Dict[str, object]],
    sst_projection_rows: Sequence[Dict[str, object]],
) -> None:
    pooled_rows = {row["target_pc"]: row for row in summary_rows if row["model"] == "ALL"}
    mean_pc1_corr = float(np.mean([row["correlation"] for row in summary_rows if row["model"] != "ALL" and row["target_pc"] == "T2M_PC1"]))
    mean_pc2_corr = float(np.mean([row["correlation"] for row in summary_rows if row["model"] != "ALL" and row["target_pc"] == "T2M_PC2"]))
    mean_pc1_r2 = float(np.mean([row["r2"] for row in summary_rows if row["model"] != "ALL" and row["target_pc"] == "T2M_PC1"]))
    mean_pc2_r2 = float(np.mean([row["r2"] for row in summary_rows if row["model"] != "ALL" and row["target_pc"] == "T2M_PC2"]))
    previous_raw = load_previous_raw_summary()

    better_pc = "PC1" if pooled_rows["T2M_PC1"]["correlation"] >= pooled_rows["T2M_PC2"]["correlation"] else "PC2"
    lines = [
        "# SST Anomaly -> March T2M Anomaly PC Predictability",
        "",
        "## Runtime",
        "",
        f"- hostname: `{runtime.hostname}`",
        f"- Slurm job ID: `{runtime.slurm_job_id}`",
        "",
        "## Setup",
        "",
        "- This run uses anomaly-consistent preprocessing.",
        "- SST predictors are February SST anomaly PCs projected from `artifacts/sst_pca/cobe2/cobe2_sst_eofs.nc` onto model `*_tskin_anomaly.nc` fields.",
        "- T2M targets are March overland T2M anomaly PCs computed from `artifacts/t2m_anomaly_validation/wus_t2m_march_anomalies/`.",
        "- Validation used leave-one-year-out ridge regression with one held-out year per fold.",
        f"- Ridge alpha: `{RIDGE_ALPHA}`.",
        "",
        "## Scientific Interpretation",
        "",
        "- This experiment tests whether February SST anomalies predict interannual March overland temperature variability rather than the climatological mean state.",
        "- The primary diagnostics are the yearly predicted-vs-true PC trajectory plots and the scatter plots for `T2M_PC1` and `T2M_PC2`.",
        f"- Pooled skill is `{pooled_rows['T2M_PC1']['correlation']:.3f}` correlation / `{pooled_rows['T2M_PC1']['r2']:.3f}` R2 for `T2M_PC1`, and `{pooled_rows['T2M_PC2']['correlation']:.3f}` / `{pooled_rows['T2M_PC2']['r2']:.3f}` for `T2M_PC2`.",
        f"- `T2M_{better_pc}` is more predictable in this anomaly-consistent configuration.",
        f"- Across models, mean correlation is `{mean_pc1_corr:.3f}` for `T2M_PC1` and `{mean_pc2_corr:.3f}` for `T2M_PC2`; mean R2 is `{mean_pc1_r2:.3f}` and `{mean_pc2_r2:.3f}` respectively.",
        "",
        "## Time-Series Diagnostics",
        "",
        "- Per-model yearly line plots are written to `plots/*_t2m_pc1_timeseries.png` and `plots/*_t2m_pc2_timeseries.png`.",
        "- Pooled multi-panel trajectory plots are written to `plots/pooled_t2m_pc1_timeseries.png` and `plots/pooled_t2m_pc2_timeseries.png`.",
        "- Year-by-year predicted and true PC values are tabulated in `prediction_timeseries.csv`.",
        "",
        "## SST Anomaly Projection Inputs",
        "",
    ]

    for row in sst_projection_rows:
        lines.append(
            f"- `{row['model']}`: `{row['input_file']}` -> `{row['output_csv']}` "
            f"(reused existing path before overwrite: `{row['reused_existing_path']}`)"
        )

    lines.extend(
        [
            "",
            "## Model Coverage",
            "",
        ]
    )

    for model_name in models:
        lines.append(
            f"- `{model_name}`: first two T2M explained-variance ratios = "
            f"`{explained_variance[model_name][0]:.3f}`, `{explained_variance[model_name][1]:.3f}`"
        )

    if previous_raw:
        lines.extend(
            [
                "",
                "## Comparison With Previous Raw-SST Run",
                "",
                "| Model | Target PC | Raw Corr | Anomaly Corr | Raw R2 | Anomaly R2 |",
                "|-------|-----------|----------|--------------|--------|------------|",
            ]
        )
        for row in sorted(summary_rows, key=lambda item: (str(item["model"]), str(item["target_pc"]))):
            key = (str(row["model"]), str(row["target_pc"]))
            if key not in previous_raw:
                continue
            raw = previous_raw[key]
            lines.append(
                f"| {row['model']} | {row['target_pc']} | "
                f"{raw['correlation']:.6f} | {float(row['correlation']):.6f} | "
                f"{raw['r2']:.6f} | {float(row['r2']):.6f} |"
            )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Model | Target PC | N Predictions | N Years | Correlation | RMSE | R2 | Explained Variance Ratio |",
            "|-------|-----------|---------------|---------|-------------|------|----|--------------------------|",
        ]
    )

    for row in sorted(summary_rows, key=lambda item: (str(item["model"]), str(item["target_pc"]))):
        evr_text = (
            f"{float(row['explained_variance_ratio']):.6f}"
            if np.isfinite(float(row["explained_variance_ratio"]))
            else "nan"
        )
        lines.append(
            f"| {row['model']} | {row['target_pc']} | {row['n_predictions']} | {row['n_years']} | "
            f"{float(row['correlation']):.6f} | {float(row['rmse']):.6f} | {float(row['r2']):.6f} | {evr_text} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)
    ensure_output_dirs()

    wrf_land_mask, _, _ = load_wrf_landmask()
    sst_reference = load_sst_eof_reference()
    model_names = discover_historical_models()
    sst_projection_rows = []
    for model_name in model_names:
        sst_projection_rows.append(build_or_reuse_model_sst_anomaly_pcs(model_name, sst_reference))

    SST_PROJECTION_SUMMARY_FILE.write_text(json.dumps(sst_projection_rows, indent=2) + "\n", encoding="utf-8")

    all_predictions: List[FoldPrediction] = []
    explained_variance: Dict[str, np.ndarray] = {}

    for model_name in model_names:
        sst_path = MODEL_SST_ANOM_PC_DIR / f"{model_name}_sst_anomaly_pcs.csv"
        anomaly_path = T2M_ANOMALY_DIR / f"{model_name}_historical_march_t2_anomalies.nc"
        sst_samples = load_february_sst_samples(sst_path)
        t2m_pca = compute_t2m_pca(anomaly_path, wrf_land_mask)
        samples = build_samples(model_name, sst_samples, t2m_pca)
        predictions = run_leave_one_year_out(samples)
        all_predictions.extend(predictions)
        explained_variance[model_name] = t2m_pca.explained_variance_ratio.copy()

        save_model_pcs(PCS_DIR / f"{model_name}_march_t2m_pcs.csv", t2m_pca.years, t2m_pca.pcs)
        save_model_eofs(EOFS_DIR / f"{model_name}_march_t2m_eofs.nc", t2m_pca)

        for target_pc in ("T2M_PC1", "T2M_PC2"):
            plot_model_timeseries(
                model_name,
                predictions,
                target_pc,
                PLOTS_DIR / f"{model_name}_{target_pc.lower()}_timeseries.png",
            )
            plot_model_scatter(
                model_name,
                predictions,
                target_pc,
                PLOTS_DIR / f"{model_name}_{target_pc.lower()}_scatter.png",
            )

        plot_model_eofs(model_name, t2m_pca, PLOTS_DIR / f"{model_name}_march_t2m_eofs.png")

    summary_rows = summarize_predictions(all_predictions, explained_variance)
    write_prediction_results(RESULTS_FILE, summary_rows)
    write_prediction_timeseries(TIMESERIES_FILE, all_predictions)
    plot_correlation_summary(summary_rows, PLOTS_DIR / "correlation_summary.png")
    plot_r2_summary(summary_rows, PLOTS_DIR / "r2_summary.png")
    plot_pooled_timeseries(all_predictions, "T2M_PC1", PLOTS_DIR / "pooled_t2m_pc1_timeseries.png")
    plot_pooled_timeseries(all_predictions, "T2M_PC2", PLOTS_DIR / "pooled_t2m_pc2_timeseries.png")
    write_report(REPORT_FILE, runtime, model_names, explained_variance, summary_rows, sst_projection_rows)

    print(f"wrote {RESULTS_FILE}", flush=True)
    print(f"wrote {TIMESERIES_FILE}", flush=True)
    print(f"wrote {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    main()
