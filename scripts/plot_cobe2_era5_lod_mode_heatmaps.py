#!/usr/bin/env python3
"""
Create paper-style COBE2 vs ERA5 SST LOD mode heat maps using existing LOD outputs only.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COBE2_SST_FILE = Path("/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc")
COBE2_SUMMARY_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_summary.json")
COBE2_DIAGNOSTICS_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/lod_analysis/cobe2_sierra_swe_lod_diagnostics.nc")
ERA5_SUMMARY_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup/lod_analysis/era5_sierra_swe_lod_summary.json")
ERA5_DIAGNOSTICS_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup/lod_analysis/era5_sierra_swe_lod_diagnostics.nc")
ERA5_PREDICTOR_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/era5_sierra_swe_lod_setup/predictors/era5_pacific_sst_monthly_anomaly_wy1985_2021_sep1984_mar2021.nc")
COMPARISON_SUMMARY_PATH = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_era5_sst_sierra_swe_lod_comparison/cobe2_era5_lod_comparison_summary.json")
OUTPUT_DIR = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_era5_sst_sierra_swe_lod_comparison")
PNG_PATH = OUTPUT_DIR / "cobe2_vs_era5_lod_modes_1to6_heatmaps.png"
PDF_PATH = OUTPUT_DIR / "cobe2_vs_era5_lod_modes_1to6_heatmaps.pdf"
NOTE_PATH = OUTPUT_DIR / "cobe2_vs_era5_lod_modes_visual_note.md"
NETCDF_ENGINE = "netcdf4"

PACIFIC_LAT_MIN = -10.0
PACIFIC_LAT_MAX = 60.0
PACIFIC_LON_MIN = 120.0
PACIFIC_LON_MAX = 280.0
LAG_SPECS = [
    ("Sep", -1, 9),
    ("Oct", -1, 10),
    ("Nov", -1, 11),
    ("Dec", -1, 12),
    ("Jan", 0, 1),
    ("Feb", 0, 2),
    ("Mar", 0, 3),
]
LAG_INDEX_BY_NAME = {name: index for index, (name, _, _) in enumerate(LAG_SPECS)}
CMAP = "RdBu_r"
CORR_LIMIT = 1.0


@dataclass(frozen=True)
class ModeRow:
    mode_number: int
    lag_month: str
    latitude: float
    longitude_0_360: float
    corr_with_residual: float
    delta_r2: float
    cumulative_r2: float


def compute_corr_map(residual: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
    r = np.asarray(residual, dtype=np.float64)
    y = np.asarray(anomalies, dtype=np.float64)
    r_centered = r - np.nanmean(r)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        y_centered = y - np.nanmean(y, axis=0, keepdims=True)
    numerator = np.nansum(r_centered[:, np.newaxis, np.newaxis] * y_centered, axis=0)
    denominator = np.sqrt(np.nansum(r_centered ** 2) * np.nansum(y_centered ** 2, axis=0))
    corr = np.full(y.shape[1:], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    corr[valid] = numerator[valid] / denominator[valid]
    return corr


def load_mode_rows(summary_path: Path) -> list[ModeRow]:
    payload = json.loads(summary_path.read_text())
    rows: list[ModeRow] = []
    for row in payload["lod_rows"]:
        if not row.get("selected"):
            continue
        rows.append(
            ModeRow(
                mode_number=int(row.get("mode_number", row.get("mode_id"))),
                lag_month=str(row["lag_month"]),
                latitude=float(row["latitude"]),
                longitude_0_360=float(row.get("longitude_0_360", row["longitude"])),
                corr_with_residual=float(row.get("selected_residual_correlation", row["corr_with_residual"])),
                delta_r2=float(row.get("delta_R2", row.get("delta_r2"))),
                cumulative_r2=float(row.get("cumulative_R2", row.get("cumulative_r2"))),
            )
        )
    rows.sort(key=lambda row: row.mode_number)
    return rows


def load_residual_series(path: Path) -> np.ndarray:
    with xr.open_dataset(path, engine=NETCDF_ENGINE) as ds:
        return np.asarray(ds["residual_series"].values, dtype=np.float64)


def month_targets() -> list[np.datetime64]:
    water_years = np.arange(1985, 2022, dtype=np.int32)
    times: list[np.datetime64] = []
    for water_year in water_years:
        for _, year_offset, month in LAG_SPECS:
            year = int(water_year + year_offset)
            times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
    return times


def load_cobe2_anomalies() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_times = month_targets()
    with xr.open_dataset(COBE2_SST_FILE, engine=NETCDF_ENGINE) as ds:
        sst = ds["sst"].sel(
            time=selected_times,
            lat=slice(PACIFIC_LAT_MAX, PACIFIC_LAT_MIN),
            lon=slice(PACIFIC_LON_MIN, PACIFIC_LON_MAX),
        ).load()
        values = np.asarray(sst.values, dtype=np.float32).reshape(37, len(LAG_SPECS), sst.sizes["lat"], sst.sizes["lon"])
        lat = np.asarray(sst["lat"].values, dtype=np.float32)
        lon = np.asarray(sst["lon"].values, dtype=np.float32)
    climatology = np.nanmean(values, axis=0, dtype=np.float64)
    anomalies = (values - climatology[None, :, :, :]).astype(np.float32)
    return anomalies, lat, lon


def load_era5_anomalies() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(ERA5_PREDICTOR_PATH, engine=NETCDF_ENGINE) as ds:
        anomalies = np.asarray(ds["sst_monthly_anomaly"].values, dtype=np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
        valid_mask = np.asarray(ds["valid_ocean_mask"].values, dtype=bool)
    return anomalies, lat, lon, valid_mask


def load_comparison_summary() -> dict[str, object]:
    return json.loads(COMPARISON_SUMMARY_PATH.read_text())


def build_maps(
    anomalies: np.ndarray,
    residual_series: np.ndarray,
    rows: list[ModeRow],
    extra_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for row in rows:
        lag_index = LAG_INDEX_BY_NAME[row.lag_month]
        residual_before = residual_series[row.mode_number - 1]
        field = anomalies[:, lag_index, :, :]
        corr_map = compute_corr_map(residual_before, field)
        if extra_mask is not None:
            corr_map = np.where(extra_mask, corr_map, np.nan)
        maps.append(corr_map)
    return maps


def plot_panel(ax, lat, lon, field, row: ModeRow, dataset_name: str):
    lon2d, lat2d = np.meshgrid(lon, lat)
    mesh = ax.pcolormesh(lon2d, lat2d, field, shading="auto", cmap=CMAP, vmin=-CORR_LIMIT, vmax=CORR_LIMIT)
    ax.set_xlim(PACIFIC_LON_MIN, PACIFIC_LON_MAX)
    ax.set_ylim(PACIFIC_LAT_MIN, PACIFIC_LAT_MAX)
    ax.scatter([row.longitude_0_360], [row.latitude], s=75, marker="*", color="black", edgecolors="white", linewidths=0.5, zorder=5)
    ax.set_xticks(np.arange(120, 281, 20))
    ax.set_yticks(np.arange(-10, 61, 10))
    ax.grid(True, linewidth=0.2, color="0.7", alpha=0.5)
    ax.set_xlabel("Longitude (0 to 360)")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{dataset_name} Mode {row.mode_number} | {row.lag_month} | "
        f"({row.latitude:.2f}, {row.longitude_0_360:.2f})\n"
        f"corr={row.corr_with_residual:+.3f}  dR2={row.delta_r2:.3f}  R2={row.cumulative_r2:.3f}",
        fontsize=9,
    )
    return mesh


def write_note(comparison: dict[str, object]) -> None:
    lines = [
        "# COBE2 vs ERA5 LOD Mode Heat Maps",
        "",
        "- The plotted field in each panel is the SST residual-correlation heat map for that LOD mode:",
        "  `corr(SST_anomaly[:, selected_lag_month_k, lat, lon], SWE_residual_before_mode_k)`.",
        "- The residual used for mode `k` is the residual before that mode is extracted, not the final residual after all modes.",
        "- COBE2 panels are plotted on the native COBE2 grid and ERA5 panels are plotted on the native ERA5 grid.",
        "- No regridding was applied for this visualization.",
        "- The selected LOD grid cell is marked on top of each heat map with a black star.",
        f"- All panels use a fixed diverging correlation color scale with limits `[-{CORR_LIMIT:.0f}, {CORR_LIMIT:.0f}]` centered at zero.",
        "- Land and missing SST cells are masked as white/transparent according to each dataset's native ocean mask and finite-sample availability.",
        "",
        "## Broad Comparison",
        "",
        "- The first few modes do not show a clean one-to-one shared broad SST structure between COBE2 and ERA5.",
        "- In the saved comparison summary, none of the first three mode pairs land in the same broad Pacific region classification.",
        "- A later mode does show broader agreement: mode 6 in both datasets falls in the midlatitude western Pacific sector.",
        "",
        "## Caveats",
        "",
        "- ERA5 has a much finer native grid and a larger native ocean mask than COBE2, so the heat maps differ in spatial texture and local sharpness even before any scientific interpretation.",
        "- Because the two datasets use different native grids and ocean masks, visually similar broad structures may still differ in exact peak location or compactness.",
        "- The selected marker is only the single SST predictor chosen by the iterative LOD step; the surrounding heat map shows the broader residual-associated SST structure at that stage.",
        "",
        "## Comparison Summary Reference",
        "",
        f"- Source summary: `{COMPARISON_SUMMARY_PATH}`",
        f"- leading_modes_similar_broad_regions: `{comparison['interpretation']['leading_modes_similar_broad_regions']}`",
        f"- first_three_same_broad_region_count: `{comparison['leading_mode_similarity']['first_three_same_broad_region_count']}`",
    ]
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cobe2_rows = load_mode_rows(COBE2_SUMMARY_PATH)
    era5_rows = load_mode_rows(ERA5_SUMMARY_PATH)
    if len(cobe2_rows) < 6 or len(era5_rows) < 6:
        raise ValueError("Expected six retained modes for both COBE2 and ERA5.")

    cobe2_residuals = load_residual_series(COBE2_DIAGNOSTICS_PATH)
    era5_residuals = load_residual_series(ERA5_DIAGNOSTICS_PATH)
    cobe2_anom, cobe2_lat, cobe2_lon = load_cobe2_anomalies()
    era5_anom, era5_lat, era5_lon, era5_valid_mask = load_era5_anomalies()

    cobe2_maps = build_maps(cobe2_anom, cobe2_residuals, cobe2_rows[:6])
    era5_maps = build_maps(era5_anom, era5_residuals, era5_rows[:6], extra_mask=era5_valid_mask)

    fig, axes = plt.subplots(6, 2, figsize=(16, 24), constrained_layout=True)

    meshes = []
    for mode_index in range(6):
        meshes.append(plot_panel(axes[mode_index, 0], cobe2_lat, cobe2_lon, cobe2_maps[mode_index], cobe2_rows[mode_index], "COBE2"))
        meshes.append(plot_panel(axes[mode_index, 1], era5_lat, era5_lon, era5_maps[mode_index], era5_rows[mode_index], "ERA5"))

    fig.suptitle(
        "COBE2 vs ERA5 SST LOD Residual-Correlation Heat Maps\n"
        "Each panel shows corr(SST anomaly at selected lag month, SWE residual before that mode)",
        fontsize=16,
    )
    cbar = fig.colorbar(meshes[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Residual correlation")
    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)
    plt.close(fig)

    comparison = load_comparison_summary()
    write_note(comparison)
    print(json.dumps({"png": str(PNG_PATH), "pdf": str(PDF_PATH), "note": str(NOTE_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
