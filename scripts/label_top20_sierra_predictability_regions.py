#!/usr/bin/env python3
"""
Label top-20% Sierra high-predictability grid cells into regional groups.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "artifacts" / "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "top20_region_labels"
DEFAULT_NETCDF = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/"
    "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only/"
    "cobe2_pacific_sierra_t2m_level2_pc1to6_route2_high_predictability_sierra_only.nc"
)
SEARCH_SUFFIXES = (".nc", ".npz", ".npy", ".csv", ".pkl", ".json")
RAW_LABELS_NPY = "raw_connected_component_labels.npy"
CLEANED_LABELS_NPY = "cleaned_top20_region_labels.npy"
RAW_LABELS_NETCDF = "raw_connected_component_labels.nc"
CLEANED_LABELS_NETCDF = "cleaned_top20_region_labels.nc"
LABEL_TABLE_CSV = "top20_region_label_table.csv"
SUMMARY_JSON = "top20_region_labels_summary.json"
DIAGNOSTIC_PNG = "top20_sierra_region_labels_diagnostic.png"
DIAGNOSTIC_PDF = "top20_sierra_region_labels_diagnostic.pdf"


@dataclass(frozen=True)
class ComponentInfo:
    raw_label: int
    size: int
    centroid_lat: float
    centroid_lon: float


@dataclass(frozen=True)
class RegionInfo:
    cleaned_label: int
    label_name: str
    size: int
    centroid_lat: float
    centroid_lon: float
    source_major_component: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--netcdf-path", type=Path, default=None)
    parser.add_argument("--top-percent", type=int, default=20)
    parser.add_argument(
        "--target-major-components",
        type=int,
        default=3,
        help="Number of major components to preserve before reassigning smaller pieces.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Grid connectivity for connected components.",
    )
    parser.add_argument(
        "--min-major-size",
        type=int,
        default=0,
        help="Optional minimum component size for major components. If 0, only rank by size.",
    )
    return parser.parse_args()


def find_candidate_files(input_dir: Path, exclude_dirs: Iterable[Path] = ()) -> List[Path]:
    excluded = {path.resolve() for path in exclude_dirs}
    candidates: List[Path] = []
    for suffix in SEARCH_SUFFIXES:
        for path in sorted(input_dir.rglob(f"*{suffix}")):
            if any(parent == excluded_path for excluded_path in excluded for parent in [path.resolve(), *path.resolve().parents]):
                continue
            candidates.append(path)
    return sorted(set(candidates))


def print_candidate_files(candidates: Sequence[Path]) -> None:
    print("Candidate files in input artifact directory:", flush=True)
    if not candidates:
        print("  none found", flush=True)
        return
    for path in candidates:
        print(f"  {path}", flush=True)


def infer_netcdf_path(input_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    local_candidates = sorted(input_dir.rglob("*.nc"))
    preferred_local = [path for path in local_candidates if "high_predictability_sierra_only" in path.name]
    if preferred_local:
        return preferred_local[0]
    if DEFAULT_NETCDF.exists():
        return DEFAULT_NETCDF
    if local_candidates:
        return local_candidates[0]
    raise FileNotFoundError(
        "No NetCDF candidate was found in the input directory and the expected /pscratch source file is unavailable."
    )


def select_top_percent_mask(
    fine_r2_map: np.ndarray,
    valid_sierra_mask: np.ndarray,
    top_percent: int,
) -> Tuple[np.ndarray, float]:
    values = np.asarray(fine_r2_map, dtype=np.float64)[valid_sierra_mask]
    if values.size == 0:
        raise ValueError("No valid Sierra R2 values found for top-percent selection")
    threshold = float(np.nanpercentile(values, 100 - top_percent))
    mask = valid_sierra_mask & np.isfinite(fine_r2_map) & (fine_r2_map >= threshold)
    return mask, threshold


def load_top20_mask(netcdf_path: Path, top_percent: int) -> Dict[str, np.ndarray | float | str]:
    with xr.open_dataset(netcdf_path, engine="netcdf4") as ds:
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
        fine_r2_map = np.asarray(ds["fine_local_r2"].values, dtype=np.float64)
        valid_sierra_mask = np.asarray(ds["valid_sierra_mask"].values).astype(bool)
        available_top_percents = [int(value) for value in np.asarray(ds["top_percent"].values)]

        selection_source = "reconstructed_from_fine_local_r2_and_valid_sierra_mask"
        threshold = float("nan")
        if "group_mask" in ds and top_percent in available_top_percents:
            selected_mask = np.asarray(ds["group_mask"].sel(top_percent=top_percent).values).astype(bool)
            selection_source = f"group_mask[top_percent={top_percent}] from source NetCDF"
            if "r2_threshold_used_within_sierra" in ds:
                threshold = float(ds["r2_threshold_used_within_sierra"].sel(top_percent=top_percent).values)
        else:
            selected_mask, threshold = select_top_percent_mask(fine_r2_map, valid_sierra_mask, top_percent)

        if not np.any(selected_mask):
            raise ValueError(f"Top-{top_percent}% selected mask is empty")

        reconstructed_mask, reconstructed_threshold = select_top_percent_mask(fine_r2_map, valid_sierra_mask, top_percent)
        exact_match = bool(np.array_equal(selected_mask, reconstructed_mask))
        if not np.isfinite(threshold):
            threshold = reconstructed_threshold

    return {
        "latitude": latitude,
        "longitude": longitude,
        "fine_r2_map": fine_r2_map,
        "valid_sierra_mask": valid_sierra_mask,
        "selected_mask": selected_mask,
        "selection_source": selection_source,
        "r2_threshold": threshold,
        "reconstruction_exact_match": exact_match,
        "available_top_percents": np.asarray(available_top_percents, dtype=np.int32),
    }


def connected_components(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.int8)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labels, n_components = ndimage.label(mask, structure=structure)
    return labels.astype(np.int32), int(n_components)


def component_infos(raw_labels: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> List[ComponentInfo]:
    infos: List[ComponentInfo] = []
    for raw_label in range(1, int(raw_labels.max()) + 1):
        indices = np.argwhere(raw_labels == raw_label)
        if indices.size == 0:
            continue
        infos.append(
            ComponentInfo(
                raw_label=raw_label,
                size=int(indices.shape[0]),
                centroid_lat=float(latitude[indices[:, 0]].mean()),
                centroid_lon=float(longitude[indices[:, 1]].mean()),
            )
        )
    infos.sort(key=lambda info: (-info.size, info.raw_label))
    return infos


def choose_major_components(
    infos: Sequence[ComponentInfo],
    target_major_components: int,
    min_major_size: int,
) -> List[ComponentInfo]:
    eligible = [info for info in infos if info.size >= min_major_size]
    if len(eligible) >= target_major_components:
        return eligible[:target_major_components]
    return list(infos[: min(target_major_components, len(infos))])


def assign_region_names(major_infos: Sequence[ComponentInfo]) -> Dict[int, Tuple[int, str]]:
    if not major_infos:
        raise ValueError("No major components are available for region naming")

    mapping: Dict[int, Tuple[int, str]] = {}
    northern = max(major_infos, key=lambda info: (info.centroid_lat, -info.centroid_lon))
    remaining = [info for info in major_infos if info.raw_label != northern.raw_label]

    if remaining:
        remaining_by_lon = sorted(remaining, key=lambda info: (info.centroid_lon, -info.centroid_lat))
        left = remaining_by_lon[0]
        right = remaining_by_lon[-1]
        if right.raw_label == left.raw_label:
            mapping[right.raw_label] = (1, "Region 1: inland/right group")
        else:
            mapping[right.raw_label] = (1, "Region 1: inland/right group")
            mapping[left.raw_label] = (2, "Region 2: coastal/left group")
        middle_candidates = [info for info in remaining_by_lon[1:-1] if info.raw_label not in mapping]
        next_label = 4
        for info in middle_candidates:
            mapping[info.raw_label] = (next_label, f"Region {next_label}: extra component")
            next_label += 1

    mapping[northern.raw_label] = (3, "Region 3: northern/top group")

    assigned_labels = {cleaned_label for cleaned_label, _ in mapping.values()}
    next_label = 4
    for info in sorted(major_infos, key=lambda item: (item.centroid_lat, item.centroid_lon)):
        if info.raw_label in mapping:
            continue
        while next_label in assigned_labels:
            next_label += 1
        mapping[info.raw_label] = (next_label, f"Region {next_label}: extra component")
        assigned_labels.add(next_label)
        next_label += 1
    return mapping


def squared_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    scale = np.cos(np.deg2rad((lat1 + lat2) / 2.0))
    return (lat1 - lat2) ** 2 + ((lon1 - lon2) * scale) ** 2


def pairwise_squared_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_2d = lat1[:, np.newaxis]
    lon1_2d = lon1[:, np.newaxis]
    lat2_2d = lat2[np.newaxis, :]
    lon2_2d = lon2[np.newaxis, :]
    scale = np.cos(np.deg2rad((lat1_2d + lat2_2d) / 2.0))
    return (lat1_2d - lat2_2d) ** 2 + ((lon1_2d - lon2_2d) * scale) ** 2


def clean_labels(
    raw_labels: np.ndarray,
    infos: Sequence[ComponentInfo],
    major_infos: Sequence[ComponentInfo],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> Tuple[np.ndarray, List[RegionInfo], Dict[int, int]]:
    if not major_infos:
        raise ValueError("At least one major component is required for cleaning")

    major_mapping = assign_region_names(major_infos)
    cleaned_labels = np.zeros_like(raw_labels, dtype=np.int32)
    raw_to_cleaned: Dict[int, int] = {}
    coastal_raw_label = next(
        raw_label
        for raw_label, (_, region_name) in major_mapping.items()
        if region_name == "Region 2: coastal/left group"
    )
    coastal_cleaned_label = major_mapping[coastal_raw_label][0]
    inland_raw_label = next(
        raw_label
        for raw_label, (_, region_name) in major_mapping.items()
        if region_name == "Region 1: inland/right group"
    )
    inland_centroid_lat = next(info.centroid_lat for info in major_infos if info.raw_label == inland_raw_label)
    inland_centroid_lon = next(info.centroid_lon for info in major_infos if info.raw_label == inland_raw_label)
    major_component_points: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for info in major_infos:
        cleaned_label, _ = major_mapping[info.raw_label]
        raw_to_cleaned[info.raw_label] = cleaned_label
        cleaned_labels[raw_labels == info.raw_label] = cleaned_label
        major_indices = np.argwhere(raw_labels == info.raw_label)
        major_component_points[info.raw_label] = (latitude[major_indices[:, 0]], longitude[major_indices[:, 1]])

    for info in infos:
        if info.raw_label in raw_to_cleaned:
            continue
        component_indices = np.argwhere(raw_labels == info.raw_label)
        if component_indices.size == 0:
            continue

        # Domain-specific override: the southern detached coastal-mode fragment
        # should be merged into the coastal group rather than the inland group.
        if info.centroid_lat < inland_centroid_lat and info.centroid_lon < inland_centroid_lon:
            cleaned_label = coastal_cleaned_label
            raw_to_cleaned[info.raw_label] = cleaned_label
            cleaned_labels[raw_labels == info.raw_label] = cleaned_label
            continue

        lat_values = latitude[component_indices[:, 0]]
        lon_values = longitude[component_indices[:, 1]]
        distances = []
        for major_info in major_infos:
            major_lat, major_lon = major_component_points[major_info.raw_label]
            dist = pairwise_squared_distance(lat_values, lon_values, major_lat, major_lon).min(axis=1).mean()
            distances.append((dist, major_info.raw_label))
        _, nearest_raw_label = min(distances, key=lambda item: (item[0], item[1]))
        cleaned_label = raw_to_cleaned[nearest_raw_label]
        raw_to_cleaned[info.raw_label] = cleaned_label
        cleaned_labels[raw_labels == info.raw_label] = cleaned_label

    regions: List[RegionInfo] = []
    for major_info in major_infos:
        cleaned_label, region_name = major_mapping[major_info.raw_label]
        region_indices = np.argwhere(cleaned_labels == cleaned_label)
        if region_indices.size == 0:
            continue
        regions.append(
            RegionInfo(
                cleaned_label=cleaned_label,
                label_name=region_name,
                size=int(region_indices.shape[0]),
                centroid_lat=float(latitude[region_indices[:, 0]].mean()),
                centroid_lon=float(longitude[region_indices[:, 1]].mean()),
                source_major_component=major_info.raw_label,
            )
        )
    regions.sort(key=lambda region: region.cleaned_label)
    return cleaned_labels, regions, raw_to_cleaned


def save_label_arrays(
    output_dir: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    selected_mask: np.ndarray,
    raw_labels: np.ndarray,
    cleaned_labels: np.ndarray,
) -> None:
    np.save(output_dir / RAW_LABELS_NPY, raw_labels)
    np.save(output_dir / CLEANED_LABELS_NPY, cleaned_labels)

    raw_ds = xr.Dataset(
        data_vars={
            "selected_top20_mask": (("latitude", "longitude"), selected_mask.astype(np.int8)),
            "raw_component_label": (("latitude", "longitude"), raw_labels.astype(np.int32)),
        },
        coords={"latitude": latitude.astype(np.float32), "longitude": longitude.astype(np.float32)},
    )
    raw_ds.to_netcdf(output_dir / RAW_LABELS_NETCDF, engine="netcdf4")

    cleaned_ds = xr.Dataset(
        data_vars={
            "selected_top20_mask": (("latitude", "longitude"), selected_mask.astype(np.int8)),
            "cleaned_region_label": (("latitude", "longitude"), cleaned_labels.astype(np.int32)),
        },
        coords={"latitude": latitude.astype(np.float32), "longitude": longitude.astype(np.float32)},
    )
    cleaned_ds.to_netcdf(output_dir / CLEANED_LABELS_NETCDF, engine="netcdf4")


def save_label_table(
    output_dir: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    selected_mask: np.ndarray,
    raw_labels: np.ndarray,
    cleaned_labels: np.ndarray,
) -> None:
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    with (output_dir / LABEL_TABLE_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lat", "lon", "selected_top20", "raw_component_label", "cleaned_region_label"])
        for row_index in range(selected_mask.shape[0]):
            for col_index in range(selected_mask.shape[1]):
                writer.writerow(
                    [
                        f"{lat2d[row_index, col_index]:.6f}",
                        f"{lon2d[row_index, col_index]:.6f}",
                        int(selected_mask[row_index, col_index]),
                        int(raw_labels[row_index, col_index]),
                        int(cleaned_labels[row_index, col_index]),
                    ]
                )


def categorical_colormap(n_labels: int) -> matplotlib.colors.Colormap:
    if n_labels <= 3:
        colors = ["#2a9d8f", "#bc6c25", "#577590"]
    else:
        base = plt.cm.get_cmap("tab10", n_labels)
        colors = [base(index) for index in range(n_labels)]
    return matplotlib.colors.ListedColormap(colors)


def plot_diagnostics(
    output_dir: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    selected_mask: np.ndarray,
    raw_labels: np.ndarray,
    cleaned_labels: np.ndarray,
    regions: Sequence[RegionInfo],
) -> None:
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    selected_mask_plot = np.where(selected_mask, 1.0, np.nan)
    raw_plot = np.where(selected_mask, raw_labels.astype(float), np.nan)
    cleaned_plot = np.where(selected_mask, cleaned_labels.astype(float), np.nan)
    selected_indices = np.argwhere(selected_mask)
    lat_min = float(latitude[selected_indices[:, 0]].min()) - 0.4
    lat_max = float(latitude[selected_indices[:, 0]].max()) + 0.4
    lon_min = float(longitude[selected_indices[:, 1]].min()) - 0.4
    lon_max = float(longitude[selected_indices[:, 1]].max()) + 0.4

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)

    selected_cmap = matplotlib.colors.ListedColormap(["#2a9d8f"])
    axes[0].pcolormesh(
        lon2d,
        lat2d,
        selected_mask_plot,
        cmap=selected_cmap,
        shading="auto",
        vmin=1.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[0].set_title("Selected top 20% within Sierra")

    raw_count = int(raw_labels.max())
    raw_cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(raw_count, 1))
    raw_mesh = axes[1].pcolormesh(
        lon2d,
        lat2d,
        raw_plot,
        cmap=raw_cmap,
        shading="auto",
        vmin=1,
        vmax=max(raw_count, 1),
        rasterized=True,
    )
    axes[1].set_title("Raw connected-component labels")
    fig.colorbar(raw_mesh, ax=axes[1], shrink=0.88, pad=0.02).set_label("raw component")

    cleaned_region_ids = sorted(region.cleaned_label for region in regions)
    cleaned_cmap = categorical_colormap(max(len(cleaned_region_ids), 1))
    cleaned_mesh = axes[2].pcolormesh(
        lon2d,
        lat2d,
        cleaned_plot,
        cmap=cleaned_cmap,
        shading="auto",
        vmin=min(cleaned_region_ids) if cleaned_region_ids else 1,
        vmax=max(cleaned_region_ids) if cleaned_region_ids else 1,
        rasterized=True,
    )
    axes[2].set_title("Cleaned final region labels")
    fig.colorbar(cleaned_mesh, ax=axes[2], shrink=0.88, pad=0.02).set_label("cleaned region")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_facecolor("#e9ecef")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    for region in regions:
        axes[2].scatter(region.centroid_lon, region.centroid_lat, marker="x", s=90, color="black", linewidths=2)
        axes[2].text(
            region.centroid_lon + 0.06,
            region.centroid_lat + 0.04,
            f"{region.cleaned_label}",
            fontsize=10,
            weight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    legend_lines = "\n".join([f"{region.cleaned_label}: {region.label_name.split(': ', 1)[1]}" for region in regions])
    axes[2].text(
        0.02,
        0.98,
        legend_lines,
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.85},
    )

    fig.savefig(output_dir / DIAGNOSTIC_PNG, dpi=220)
    fig.savefig(output_dir / DIAGNOSTIC_PDF)
    plt.close(fig)


def save_summary_json(
    output_dir: Path,
    netcdf_path: Path,
    top_percent: int,
    target_major_components: int,
    connectivity: int,
    threshold: float,
    selection_source: str,
    reconstruction_exact_match: bool,
    infos: Sequence[ComponentInfo],
    major_infos: Sequence[ComponentInfo],
    raw_to_cleaned: Dict[int, int],
    regions: Sequence[RegionInfo],
) -> None:
    payload = {
        "source_netcdf": str(netcdf_path),
        "top_percent": top_percent,
        "target_major_components": target_major_components,
        "connectivity": connectivity,
        "r2_threshold_used_within_sierra": threshold,
        "selection_source": selection_source,
        "reconstruction_exact_match": reconstruction_exact_match,
        "raw_components": [
            {
                "raw_label": info.raw_label,
                "size": info.size,
                "centroid_lat": info.centroid_lat,
                "centroid_lon": info.centroid_lon,
            }
            for info in infos
        ],
        "major_components": [info.raw_label for info in major_infos],
        "raw_to_cleaned_mapping": {str(raw_label): cleaned_label for raw_label, cleaned_label in raw_to_cleaned.items()},
        "cleaned_regions": [
            {
                "cleaned_label": region.cleaned_label,
                "label_name": region.label_name,
                "size": region.size,
                "centroid_lat": region.centroid_lat,
                "centroid_lon": region.centroid_lon,
                "source_major_component": region.source_major_component,
            }
            for region in regions
        ],
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_summary(
    output_dir: Path,
    selected_mask: np.ndarray,
    infos: Sequence[ComponentInfo],
    regions: Sequence[RegionInfo],
) -> None:
    print(f"number of selected top20 grid cells: {int(np.count_nonzero(selected_mask))}", flush=True)
    print(f"number of raw connected components: {len(infos)}", flush=True)
    print("size of each raw component:", flush=True)
    for info in infos:
        print(
            f"  raw_component={info.raw_label} size={info.size} "
            f"centroid_lat={info.centroid_lat:.4f} centroid_lon={info.centroid_lon:.4f}",
            flush=True,
        )
    print(f"number of cleaned final regions: {len(regions)}", flush=True)
    print("size and centroid of each cleaned region:", flush=True)
    for region in regions:
        print(
            f"  cleaned_region={region.cleaned_label} name='{region.label_name}' size={region.size} "
            f"centroid_lat={region.centroid_lat:.4f} centroid_lon={region.centroid_lon:.4f}",
            flush=True,
        )
    print("output paths of all saved files:", flush=True)
    for filename in [
        RAW_LABELS_NPY,
        CLEANED_LABELS_NPY,
        RAW_LABELS_NETCDF,
        CLEANED_LABELS_NETCDF,
        LABEL_TABLE_CSV,
        SUMMARY_JSON,
        DIAGNOSTIC_PNG,
        DIAGNOSTIC_PDF,
    ]:
        print(f"  {output_dir / filename}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_candidate_files(args.input_dir, exclude_dirs=[args.output_dir])
    print_candidate_files(candidates)

    netcdf_path = infer_netcdf_path(args.input_dir, args.netcdf_path)
    print(f"Using source NetCDF: {netcdf_path}", flush=True)

    loaded = load_top20_mask(netcdf_path, args.top_percent)
    latitude = loaded["latitude"]
    longitude = loaded["longitude"]
    selected_mask = loaded["selected_mask"]

    raw_labels, n_components = connected_components(selected_mask, args.connectivity)
    if n_components <= 0:
        raise ValueError("Connected-component labeling produced zero components")

    infos = component_infos(raw_labels, latitude, longitude)
    major_infos = choose_major_components(infos, args.target_major_components, args.min_major_size)
    cleaned_labels, regions, raw_to_cleaned = clean_labels(raw_labels, infos, major_infos, latitude, longitude)

    save_summary_json(
        args.output_dir,
        netcdf_path,
        args.top_percent,
        args.target_major_components,
        args.connectivity,
        float(loaded["r2_threshold"]),
        str(loaded["selection_source"]),
        bool(loaded["reconstruction_exact_match"]),
        infos,
        major_infos,
        raw_to_cleaned,
        regions,
    )
    save_label_arrays(args.output_dir, latitude, longitude, selected_mask, raw_labels, cleaned_labels)
    save_label_table(args.output_dir, latitude, longitude, selected_mask, raw_labels, cleaned_labels)
    plot_diagnostics(args.output_dir, latitude, longitude, selected_mask, raw_labels, cleaned_labels, regions)

    print(f"Selection source: {loaded['selection_source']}", flush=True)
    print(f"R2 threshold used within Sierra for top {args.top_percent}%: {float(loaded['r2_threshold']):.6f}", flush=True)
    print(f"Reconstructed mask exact match with stored mask: {bool(loaded['reconstruction_exact_match'])}", flush=True)
    print_summary(args.output_dir, selected_mask, infos, regions)


if __name__ == "__main__":
    main()
