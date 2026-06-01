#!/usr/bin/env python3
"""
Project reusable monthly WUS-D3 SST anomalies onto the fixed COBE2 EOF basis.

This script:
1. Reuses saved WUS-D3 monthly SST anomaly files on the COBE2 grid.
2. Projects each dataset onto the fixed COBE2 EOF basis using the same
   sqrt(cos(lat)) weighting convention used in the COBE2 EOF build.
3. Computes overlap-month correlations against the saved COBE2 PCs when an
   observed overlap exists.
4. Saves projected pseudo-PC time series and Sierra-box EOF figures.

Projection is always done on the full shared COBE2 grid. The Sierra-box plots
only crop the saved EOF maps for visualization.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_sst_monthly_climatology_eof_diagnostics import (
    ensure_runtime_on_compute_node,
    get_runtime,
    open_dataset_with_fallbacks,
)
from snow_ml.data import DEFAULT_SIERRA_REGION


N_MODES_DEFAULT = 6
DEFAULT_DOMAIN = "d03"
COBE2_EOF_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "sst_pca"
    / "cobe2_global_monthly_climatology_anomaly"
    / "cobe2_global_monthly_clim_sst_eofs.nc"
)
DEFAULT_INPUT_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/wusd3_sst_on_cobe2_grid_monthly"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "wus_sst_projected_onto_cobe2_eofs"


@dataclass(frozen=True)
class Cobe2Reference:
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    eof: np.ndarray
    pc: np.ndarray
    valid_mask: np.ndarray
    explained_variance_ratio: np.ndarray
    weighting_note: str


@dataclass(frozen=True)
class DatasetProjection:
    dataset_id: str
    domain: str
    time: np.ndarray
    projected_pc: np.ndarray
    shared_mask: np.ndarray
    overlap_months: np.ndarray
    overlap_correlations: np.ndarray
    projected_pc_std: np.ndarray
    input_file: str
    anomaly_variable: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project reusable WUS-D3 SST anomalies onto COBE2 EOFs.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="WUS-D3 domain to process, default d03.")
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Dataset id to process. Repeat flag to pass multiple datasets. Defaults to all discovered datasets.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root of remapped WUS SST files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--n-modes", type=int, default=N_MODES_DEFAULT, help="Number of EOF modes to project.")
    return parser.parse_args()


def format_date(value: np.datetime64) -> str:
    return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))


def to_month_start(times: Sequence[np.datetime64]) -> np.ndarray:
    values = np.asarray(times, dtype="datetime64[ns]")
    return values.astype("datetime64[M]").astype("datetime64[ns]")


def intersect_months(*time_axes: Sequence[np.datetime64]) -> np.ndarray:
    common = to_month_start(time_axes[0])
    for axis in time_axes[1:]:
        common = np.intersect1d(common, to_month_start(axis), assume_unique=False)
    return np.asarray(common, dtype="datetime64[ns]")


def select_by_months(time_values: Sequence[np.datetime64], data: np.ndarray, target_months: np.ndarray) -> np.ndarray:
    month_values = to_month_start(time_values)
    index_by_month = {month: idx for idx, month in enumerate(month_values.tolist())}
    return np.asarray(data)[[index_by_month[month] for month in target_months.tolist()]]


def discover_dataset_ids(input_root: Path, domain: str) -> List[str]:
    domain_root = input_root / domain
    if not domain_root.exists():
        raise FileNotFoundError("Missing domain directory: %s" % domain_root)
    dataset_ids = sorted(path.name for path in domain_root.iterdir() if path.is_dir())
    if not dataset_ids:
        raise FileNotFoundError("No dataset directories found under %s" % domain_root)
    return dataset_ids


def anomaly_file_for_dataset(input_root: Path, domain: str, dataset_id: str) -> Path:
    return input_root / domain / dataset_id / (
        "%s_%s_tskin_on_cobe2_grid_monthly_anomaly.nc" % (dataset_id, domain)
    )


def load_cobe2_reference(n_modes: int) -> Cobe2Reference:
    with open_dataset_with_fallbacks(COBE2_EOF_FILE) as ds:
        weighting_note = str(ds.attrs.get("latitude_weighting", "sqrt(cos(lat)) weighting"))
        return Cobe2Reference(
            time=np.asarray(ds["time"].values, dtype="datetime64[ns]"),
            latitude=np.asarray(ds["lat"].values, dtype=np.float64),
            longitude=np.asarray(ds["lon"].values, dtype=np.float64),
            eof=np.asarray(ds["eof"].values[:n_modes], dtype=np.float64),
            pc=np.asarray(ds["pc"].values[:, :n_modes], dtype=np.float64),
            valid_mask=np.asarray(ds["valid_mask"].values, dtype=bool),
            explained_variance_ratio=np.asarray(ds["explained_variance_ratio"].values[:n_modes], dtype=np.float64),
            weighting_note=weighting_note,
        )


def load_wus_sst_anomaly(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    with open_dataset_with_fallbacks(path) as ds:
        if "tskin_anomaly" in ds.data_vars:
            variable_name = "tskin_anomaly"
        elif "tskin" in ds.data_vars:
            variable_name = "tskin"
        else:
            raise ValueError("Expected tskin anomaly variable in %s, found %s" % (path, list(ds.data_vars)))
        values = np.asarray(ds[variable_name].values, dtype=np.float64)
        time = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    return time, values, variable_name


def compute_projection(
    dataset_id: str,
    domain: str,
    cobe2: Cobe2Reference,
    anomaly_path: Path,
) -> DatasetProjection:
    time, values, anomaly_variable = load_wus_sst_anomaly(anomaly_path)
    shared_mask = cobe2.valid_mask & np.isfinite(values).all(axis=0)
    shared_count = int(np.count_nonzero(shared_mask))
    if shared_count == 0:
        raise ValueError("No shared valid cells remain for dataset %s" % dataset_id)

    lat_weights_1d = np.sqrt(np.clip(np.cos(np.deg2rad(cobe2.latitude)), 0.0, None))
    weights_2d = np.broadcast_to(lat_weights_1d[:, np.newaxis], shared_mask.shape)
    weights_flat = weights_2d[shared_mask]

    weighted_anom = np.asarray(values[:, shared_mask], dtype=np.float64) * weights_flat[np.newaxis, :]
    weighted_eof = np.asarray(cobe2.eof[:, shared_mask], dtype=np.float64) * weights_flat[np.newaxis, :]
    projected_pc = weighted_anom @ weighted_eof.T

    overlap_months = intersect_months(time, cobe2.time)
    correlations = np.full(cobe2.eof.shape[0], np.nan, dtype=np.float64)
    if overlap_months.size > 0:
        projected_overlap = select_by_months(time, projected_pc, overlap_months)
        cobe2_pc_overlap = select_by_months(cobe2.time, cobe2.pc, overlap_months)
        for mode_index in range(cobe2.eof.shape[0]):
            left = projected_overlap[:, mode_index]
            right = cobe2_pc_overlap[:, mode_index]
            finite = np.isfinite(left) & np.isfinite(right)
            if np.count_nonzero(finite) >= 3:
                correlations[mode_index] = np.corrcoef(left[finite], right[finite])[0, 1]

    return DatasetProjection(
        dataset_id=dataset_id,
        domain=domain,
        time=time,
        projected_pc=projected_pc.astype(np.float32),
        shared_mask=shared_mask,
        overlap_months=overlap_months,
        overlap_correlations=correlations,
        projected_pc_std=np.std(projected_pc, axis=0, ddof=1),
        input_file=str(anomaly_path),
        anomaly_variable=anomaly_variable,
    )


def output_dir_for_dataset(output_root: Path, domain: str, dataset_id: str) -> Path:
    return output_root / domain / dataset_id


def sierra_box_indices(cobe2: Cobe2Reference) -> Tuple[np.ndarray, np.ndarray]:
    lon_360 = np.mod(cobe2.longitude, 360.0)
    lon_min = DEFAULT_SIERRA_REGION.lon_min % 360.0
    lon_max = DEFAULT_SIERRA_REGION.lon_max % 360.0
    lat_idx = np.where(
        (cobe2.latitude >= DEFAULT_SIERRA_REGION.lat_min)
        & (cobe2.latitude <= DEFAULT_SIERRA_REGION.lat_max)
    )[0]
    lon_idx = np.where((lon_360 >= lon_min) & (lon_360 <= lon_max))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Sierra box does not intersect the COBE2 grid")
    return lat_idx, lon_idx


def plot_projected_pcs(output_dir: Path, result: DatasetProjection) -> None:
    n_modes = result.projected_pc.shape[1]
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2.0 * n_modes), sharex=True, constrained_layout=True)
    if n_modes == 1:
        axes = [axes]
    time_plot = result.time.astype("datetime64[ns]")
    for mode_index, ax in enumerate(axes):
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax.plot(time_plot, result.projected_pc[:, mode_index], color="black", linewidth=1.0)
        ax.set_ylabel("PC%d" % (mode_index + 1))
        ax.set_title(
            "Projected PC%d std=%.3f overlap corr=%.3f"
            % (
                mode_index + 1,
                float(result.projected_pc_std[mode_index]),
                float(result.overlap_correlations[mode_index]),
            )
        )
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("%s %s WUS-on-COBE2 pseudo-PCs" % (result.dataset_id, result.domain), fontsize=14)
    fig.savefig(output_dir / "projected_pc_timeseries_modes1to6.png", dpi=220)
    plt.close(fig)


def plot_sierra_eofs(output_dir: Path, cobe2: Cobe2Reference, result: DatasetProjection) -> None:
    lat_idx, lon_idx = sierra_box_indices(cobe2)
    lon_360 = np.mod(cobe2.longitude, 360.0)
    lon_plot = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)
    lon_subset = lon_plot[lon_idx]
    lat_subset = cobe2.latitude[lat_idx]
    lon2d, lat2d = np.meshgrid(lon_subset, lat_subset)

    eof_subset = cobe2.eof[:, lat_idx, :][:, :, lon_idx]
    shared_subset = result.shared_mask[lat_idx, :][:, lon_idx]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for mode_index, ax in enumerate(axes_flat[: eof_subset.shape[0]]):
        field = np.asarray(eof_subset[mode_index], dtype=np.float64)
        vmax = float(np.nanmax(np.abs(field)))
        vmax = 1.0 if not np.isfinite(vmax) or vmax == 0.0 else vmax
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            field,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlim(DEFAULT_SIERRA_REGION.lon_min, DEFAULT_SIERRA_REGION.lon_max)
        ax.set_ylim(DEFAULT_SIERRA_REGION.lat_min, DEFAULT_SIERRA_REGION.lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            "EOF%d Sierra box (EVR %.2f%%)" % (
                mode_index + 1,
                100.0 * float(cobe2.explained_variance_ratio[mode_index]),
            )
        )
        if np.any(shared_subset):
            ax.contour(
                lon2d,
                lat2d,
                shared_subset.astype(np.int8),
                levels=[0.5],
                colors="black",
                linewidths=0.8,
            )
        fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.suptitle(
        "%s %s COBE2 EOFs 1-6 cropped to Sierra box\nblack outline shows shared cells used in the projection"
        % (result.dataset_id, result.domain),
        fontsize=14,
    )
    fig.savefig(output_dir / "cobe2_eofs_modes1to6_sierra_box.png", dpi=220)
    plt.close(fig)


def save_projection_netcdf(output_dir: Path, cobe2: Cobe2Reference, result: DatasetProjection) -> None:
    ds = xr.Dataset(
        data_vars={
            "projected_pc": (("time", "mode"), result.projected_pc.astype(np.float32)),
            "projection_shared_mask": (("lat", "lon"), result.shared_mask.astype(np.int8)),
            "cobe2_eof": (
                ("mode", "lat", "lon"),
                cobe2.eof.astype(np.float32),
            ),
        },
        coords={
            "time": result.time.astype("datetime64[ns]"),
            "mode": np.arange(1, result.projected_pc.shape[1] + 1, dtype=np.int32),
            "lat": cobe2.latitude.astype(np.float32),
            "lon": cobe2.longitude.astype(np.float32),
        },
        attrs={
            "dataset_id": result.dataset_id,
            "domain": result.domain,
            "description": "WUS-D3 SST anomalies projected onto fixed COBE2 global EOFs",
            "projection_weighting": cobe2.weighting_note,
        },
    )
    ds.to_netcdf(output_dir / "projected_pc_timeseries_and_mask.nc")


def save_projection_outputs(output_root: Path, cobe2: Cobe2Reference, result: DatasetProjection) -> Dict[str, object]:
    output_dir = output_dir_for_dataset(output_root, result.domain, result.dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "projected_pc_timeseries.npy", result.projected_pc)
    with (output_dir / "projected_pc_timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time"] + ["PC%d" % idx for idx in range(1, result.projected_pc.shape[1] + 1)])
        for time_value, row in zip(result.time, result.projected_pc):
            writer.writerow([format_date(time_value)] + ["%.12g" % float(value) for value in row])

    metadata = {
        "dataset_id": result.dataset_id,
        "domain": result.domain,
        "input_file": result.input_file,
        "anomaly_variable": result.anomaly_variable,
        "time_start": format_date(result.time[0]),
        "time_end": format_date(result.time[-1]),
        "n_time": int(result.time.size),
        "n_modes": int(result.projected_pc.shape[1]),
        "projection_shared_cell_count": int(np.count_nonzero(result.shared_mask)),
        "projection_shared_grid_shape": [int(result.shared_mask.shape[0]), int(result.shared_mask.shape[1])],
        "projection_weighting": cobe2.weighting_note,
        "cobe2_eof_file": str(COBE2_EOF_FILE),
        "overlap_with_cobe2_months": int(result.overlap_months.size),
        "overlap_start": None if result.overlap_months.size == 0 else format_date(result.overlap_months[0]),
        "overlap_end": None if result.overlap_months.size == 0 else format_date(result.overlap_months[-1]),
        "overlap_correlations": [float(value) for value in result.overlap_correlations.tolist()],
        "projected_pc_std": [float(value) for value in result.projected_pc_std.tolist()],
        "sierra_plot_region": asdict(DEFAULT_SIERRA_REGION),
        "projection_note": "Projection used the whole shared remapped COBE2-grid SST field; EOF plots are Sierra-box crops only.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    plot_projected_pcs(output_dir, result)
    plot_sierra_eofs(output_dir, cobe2, result)
    save_projection_netcdf(output_dir, cobe2, result)
    return metadata


def save_summary_csv(output_root: Path, domain: str, rows: List[Dict[str, object]], n_modes: int) -> None:
    summary_dir = output_root / domain
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "projection_summary.csv"
    fieldnames = [
        "dataset_id",
        "domain",
        "time_start",
        "time_end",
        "n_time",
        "projection_shared_cell_count",
        "overlap_with_cobe2_months",
        "overlap_start",
        "overlap_end",
    ] + ["pc%d_corr_with_cobe2" % idx for idx in range(1, n_modes + 1)]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "dataset_id": row["dataset_id"],
                "domain": row["domain"],
                "time_start": row["time_start"],
                "time_end": row["time_end"],
                "n_time": row["n_time"],
                "projection_shared_cell_count": row["projection_shared_cell_count"],
                "overlap_with_cobe2_months": row["overlap_with_cobe2_months"],
                "overlap_start": row["overlap_start"],
                "overlap_end": row["overlap_end"],
            }
            for idx, value in enumerate(row["overlap_correlations"], start=1):
                out["pc%d_corr_with_cobe2" % idx] = "%.12g" % float(value)
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    runtime = get_runtime()
    ensure_runtime_on_compute_node(runtime)

    cobe2 = load_cobe2_reference(args.n_modes)
    dataset_ids = args.dataset_id if args.dataset_id else discover_dataset_ids(args.input_root, args.domain)
    print("Projecting datasets on domain %s: %s" % (args.domain, ", ".join(dataset_ids)), flush=True)

    summary_rows: List[Dict[str, object]] = []
    for dataset_id in dataset_ids:
        anomaly_path = anomaly_file_for_dataset(args.input_root, args.domain, dataset_id)
        if not anomaly_path.exists():
            raise FileNotFoundError("Missing anomaly file for %s: %s" % (dataset_id, anomaly_path))
        print("Projecting %s from %s" % (dataset_id, anomaly_path), flush=True)
        result = compute_projection(dataset_id, args.domain, cobe2, anomaly_path)
        metadata = save_projection_outputs(args.output_root, cobe2, result)
        summary_rows.append(metadata)
        print(
            "Finished %s time=%s..%s n_time=%d shared_cells=%d overlap=%d"
            % (
                dataset_id,
                metadata["time_start"],
                metadata["time_end"],
                metadata["n_time"],
                metadata["projection_shared_cell_count"],
                metadata["overlap_with_cobe2_months"],
            ),
            flush=True,
        )

    save_summary_csv(args.output_root, args.domain, summary_rows, args.n_modes)
    print("Wrote summary CSV under %s" % (args.output_root / args.domain), flush=True)


if __name__ == "__main__":
    main()
