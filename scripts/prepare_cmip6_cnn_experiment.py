from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


MONTH_LABELS = ("Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar")
PREDICTOR_SPECS = (
    ("tos", "tos_1p5deg_model_years.nc", "tos"),
    ("siconc", "siconc_1p5deg_model_years.nc", "siconc"),
    ("rlut", "rlut_1p5deg_model_years.nc", "rlut"),
    ("zg_500", "zg_500hPa_1p5deg_model_years.nc", "zg_500hPa"),
    ("ta_850", "ta_850hPa_1p5deg_model_years.nc", "ta_850hPa"),
    ("ua_850", "ua_850hPa_1p5deg_model_years.nc", "ua_850hPa"),
    ("va_850", "va_850hPa_1p5deg_model_years.nc", "va_850hPa"),
    ("ua_200", "ua_200hPa_1p5deg_model_years.nc", "ua_200hPa"),
    ("va_200", "va_200hPa_1p5deg_model_years.nc", "va_200hPa"),
    ("hus_850", "hus_850hPa_1p5deg_model_years.nc", "hus_850hPa"),
    ("psl", "psl_1p5deg_model_years.nc", "psl"),
    ("tas", "tas_1p5deg_model_years.nc", "tas"),
    ("zg_50", "zg_50hPa_1p5deg_model_years.nc", "zg_50hPa"),
    ("ta_50", "ta_50hPa_1p5deg_model_years.nc", "ta_50hPa"),
    ("ua_50", "ua_50hPa_1p5deg_model_years.nc", "ua_50hPa"),
    ("va_50", "va_50hPa_1p5deg_model_years.nc", "va_50hPa"),
    ("mrso", "mrso_1p5deg_model_years.nc", "mrso"),
    ("thetao_50m", "thetao_50m_1p5deg_model_years.nc", "thetao_50m"),
    ("thetao_100m", "thetao_100m_1p5deg_model_years.nc", "thetao_100m"),
)
OBS_TRANSFER_16_PREDICTORS = (
    "tos",
    "siconc",
    "rlut",
    "zg_500",
    "ta_850",
    "ua_850",
    "va_850",
    "ua_200",
    "va_200",
    "hus_850",
    "psl",
    "tas",
    "zg_50",
    "ta_50",
    "ua_50",
    "va_50",
)
OBS_TRANSFER_17_PREDICTORS = OBS_TRANSFER_16_PREDICTORS + ("mrso",)


def infer_experiment(row_year: int) -> str:
    return "historical" if row_year <= 2013 else "ssp370"


def parse_model_member(model_member: str) -> tuple[str, str]:
    source_id, member_id = model_member.split(":", 1)
    return source_id, member_id


def load_field_index(raw_path: Path, variable_name: str) -> tuple[dict[tuple[str, int], int], xr.Dataset]:
    ds = xr.open_dataset(raw_path)
    mapping = {
        (str(model_member), int(row_year)): int(sample_index)
        for sample_index, (model_member, row_year) in enumerate(
            zip(ds["model"].values.astype(str), ds["row_year"].values.astype(int), strict=True)
        )
    }
    if tuple(ds["month_label"].values.tolist())[: len(MONTH_LABELS)] != MONTH_LABELS:
        raise ValueError(f"{raw_path} has unexpected month labels: {ds['month_label'].values}")
    if variable_name not in ds.data_vars:
        raise ValueError(f"{raw_path} does not contain expected variable {variable_name}")
    return mapping, ds


def select_predictor_specs(predictor_set: str) -> tuple[tuple[str, str, str], ...]:
    if predictor_set == "all19":
        return PREDICTOR_SPECS
    if predictor_set == "obs_compatible_16":
        selected = tuple(spec for spec in PREDICTOR_SPECS if spec[0] in OBS_TRANSFER_16_PREDICTORS)
        if len(selected) != len(OBS_TRANSFER_16_PREDICTORS):
            raise ValueError(
                f"Expected {len(OBS_TRANSFER_16_PREDICTORS)} predictors for obs_compatible_16, found {len(selected)}"
            )
        return selected
    if predictor_set == "obs_compatible_17":
        selected = tuple(spec for spec in PREDICTOR_SPECS if spec[0] in OBS_TRANSFER_17_PREDICTORS)
        if len(selected) != len(OBS_TRANSFER_17_PREDICTORS):
            raise ValueError(
                f"Expected {len(OBS_TRANSFER_17_PREDICTORS)} predictors for obs_compatible_17, found {len(selected)}"
            )
        return selected
    raise ValueError(f"Unknown predictor set: {predictor_set}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_regridded_1p5deg/raw",
    )
    parser.add_argument(
        "--aux-labels",
        default="artifacts/cmip6_aux_labels/cmip6_auxiliary_labels.csv",
    )
    parser.add_argument(
        "--swe-labels",
        default="artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache",
    )
    parser.add_argument(
        "--predictor-set",
        choices=("all19", "obs_compatible_16", "obs_compatible_17"),
        default="all19",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_specs = select_predictor_specs(args.predictor_set)

    aux_df = pd.read_csv(args.aux_labels)
    swe_df = pd.read_csv(args.swe_labels)
    merged = aux_df.merge(
        swe_df,
        on=["source_id", "member_id", "model_member", "experiment", "row_year", "water_year"],
        how="inner",
        suffixes=("_aux", "_swe"),
    )

    field_indexes: dict[str, dict[tuple[str, int], int]] = {}
    field_datasets: dict[str, xr.Dataset] = {}
    intersection_keys = {(str(row.model_member), int(row.row_year)) for row in merged.itertuples()}
    file_rows: list[dict[str, object]] = []
    for field_name, file_name, variable_name in predictor_specs:
        mapping, ds = load_field_index(raw_dir / file_name, variable_name)
        field_indexes[field_name] = mapping
        field_datasets[field_name] = ds
        available_keys = set(mapping)
        intersection_keys &= available_keys
        file_rows.append(
            {
                "field": field_name,
                "file_name": file_name,
                "variable_name": variable_name,
                "sample_count": len(mapping),
            }
        )

    manifest_df = merged[
        merged.apply(lambda row: (str(row["model_member"]), int(row["row_year"])) in intersection_keys, axis=1)
    ].copy()
    manifest_df = manifest_df.sort_values(["model_member", "row_year"]).reset_index(drop=True)
    manifest_df["sample_index"] = np.arange(len(manifest_df), dtype=np.int64)

    first_ds = next(iter(field_datasets.values()))
    n_samples = len(manifest_df)
    n_months = len(MONTH_LABELS)
    n_fields = len(predictor_specs)
    n_lat = int(first_ds.sizes["lat"])
    n_lon = int(first_ds.sizes["lon"])

    physical_path = output_dir / "inputs_physical.npy"
    mask_path = output_dir / "inputs_valid_mask.npy"
    targets_path = output_dir / "targets.npy"

    physical_memmap = np.lib.format.open_memmap(
        physical_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_months, n_fields, n_lat, n_lon),
    )
    mask_memmap = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_samples, n_months, n_fields, n_lat, n_lon),
    )
    targets = np.zeros((n_samples, 3), dtype=np.float32)

    for sample_idx, row in enumerate(manifest_df.itertuples()):
        key = (str(row.model_member), int(row.row_year))
        targets[sample_idx, 0] = float(row.SWE_label)
        targets[sample_idx, 1] = float(row.CPM_label)
        targets[sample_idx, 2] = float(row.AQM_label)
        for field_index, (field_name, _file_name, variable_name) in enumerate(predictor_specs):
            ds = field_datasets[field_name]
            source_index = field_indexes[field_name][key]
            array = ds[variable_name].isel(sample=source_index, month_in_model_year=slice(0, n_months)).values.astype(
                np.float32
            )
            physical_memmap[sample_idx, :, field_index] = array
            mask_memmap[sample_idx, :, field_index] = np.isfinite(array).astype(np.uint8)

    np.save(targets_path, targets)

    manifest_path = output_dir / "manifest.csv"
    manifest_columns = [
        "sample_index",
        "source_id",
        "member_id",
        "model_member",
        "experiment",
        "row_year",
        "water_year",
        "SWE_label",
        "CPM_label",
        "AQM_label",
    ]
    manifest_df.to_csv(manifest_path, columns=manifest_columns, index=False)

    summary = {
        "output_dir": str(output_dir),
        "predictor_set": args.predictor_set,
        "sample_count": int(n_samples),
        "month_labels": list(MONTH_LABELS),
        "predictor_fields": [row["field"] for row in file_rows],
        "physical_shape": [int(n_samples), int(n_months), int(n_fields), int(n_lat), int(n_lon)],
        "valid_mask_shape": [int(n_samples), int(n_months), int(n_fields), int(n_lat), int(n_lon)],
        "stacked_input_shape_per_sample": [int(n_months), int(n_fields * 2), int(n_lat), int(n_lon)],
        "flattened_cnn_input_shape_per_sample": [int(n_months * n_fields * 2), int(n_lat), int(n_lon)],
        "target_names": ["SWE_label", "CPM_label", "AQM_label"],
        "dropped_rows_due_to_predictor_intersection": int(len(merged) - len(manifest_df)),
        "field_files": file_rows,
        "sample_counts_by_model": manifest_df["model_member"].value_counts().sort_index().to_dict(),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with (output_dir / "target_names.json").open("w") as handle:
        json.dump(summary["target_names"], handle, indent=2)

    predictor_inventory = {
        "predictor_set": args.predictor_set,
        "predictor_fields": [row["field"] for row in file_rows],
        "predictor_count": int(n_fields),
        "mask_count": int(n_fields),
        "stacked_channel_count": int(n_fields * 2),
        "months": list(MONTH_LABELS),
        "stacked_tensor_shape_per_sample": [int(n_months), int(n_fields * 2), int(n_lat), int(n_lon)],
        "flattened_cnn_input_shape_per_sample": [int(n_months * n_fields * 2), int(n_lat), int(n_lon)],
        "removed_predictors": sorted(set(row[0] for row in PREDICTOR_SPECS) - set(summary["predictor_fields"])),
    }
    with (output_dir / "predictor_inventory.json").open("w") as handle:
        json.dump(predictor_inventory, handle, indent=2, sort_keys=True)

    for ds in field_datasets.values():
        ds.close()

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote physical inputs: {physical_path}")
    print(f"Wrote valid mask: {mask_path}")
    print(f"Wrote targets: {targets_path}")
    print(f"Sample count: {n_samples}")


if __name__ == "__main__":
    main()
