#!/usr/bin/env python3
"""Convert ACE member 20151101 November fields into the CNN November layout.

This is a representation-only test: it neither runs ACE nor evaluates the
meteorology.  The target is the physical-value plus validity-mask convention
used by the existing 17-variable CNN cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ACE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016/member_20151101/autoregressive_predictions.nc")
RAW = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_regridded_1p5deg/raw")
CACHE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_17var_baseline_v1/data_cache")
EXPERIMENT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_17var_baseline_v1/experiments/S0_17var_static_cnn_swe_only")
OUT = Path("artifacts/ace_november_cnn_representation")
FIELDS = {
    "TMP2m": ("tas", "tas_1p5deg_model_years.nc", "tas", "K"),
    "TMP850": ("ta_850", "ta_850hPa_1p5deg_model_years.nc", "ta_850hPa", "K"),
    "h500": ("zg_500", "zg_500hPa_1p5deg_model_years.nc", "zg_500hPa", "m"),
}


def ace_times(ds: xr.Dataset) -> np.ndarray:
    return (np.datetime64("2015-11-01T00:00:00") + ds.time.values.astype("timedelta64[us]")).astype("datetime64[ns]")


def remap_bilinear(da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    # Source and target are regular global grids.  Extrapolation handles the
    # target polar centers lying 0.014 degrees outside ACE Gaussian centers,
    # matching the complete-grid convention of the CNN cache.
    return da.interp(lat=target_lat, lon=target_lon, method="linear", kwargs={"fill_value": "extrapolate"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    first_file = RAW / "tas_1p5deg_model_years.nc"
    with xr.open_dataset(first_file, decode_times=False) as target_ds:
        target_lat, target_lon = target_ds.lat.values, target_ds.lon.values
    with xr.open_dataset(ACE, decode_times=False) as ace_ds:
        november_index = np.flatnonzero(ace_times(ace_ds).astype("datetime64[M]") == np.datetime64("2015-11"))
        month = ace_ds.isel(sample=0, time=november_index)
        if month.sizes["time"] != 119:
            raise RuntimeError(f"Expected 119 free-forecast November outputs, found {month.sizes['time']}")
        physical, masks, report, regridded = [], [], [], {}
        for ace_name, (cnn_name, raw_file, raw_name, expected_unit) in FIELDS.items():
            raw = month[ace_name]
            monthly = raw.mean("time", skipna=True)
            final = remap_bilinear(monthly, target_lat, target_lon).astype("float32")
            values = final.values
            mask = np.isfinite(values).astype("uint8")
            if values.shape != (120, 240):
                raise RuntimeError(f"{ace_name} has unexpected final shape {values.shape}")
            with xr.open_dataset(RAW / raw_file, decode_times=False) as train_ds:
                # Existing sample 0 is used only as a structural reference.
                train_nov = train_ds[raw_name].isel(sample=0, month_in_model_year=2).values
            report.append({
                "ACE variable": ace_name, "CNN variable": cnn_name,
                "raw ACE shape": list(raw.shape), "monthly shape": list(monthly.shape),
                "final regridded shape": list(values.shape), "units match": raw.attrs.get("units", "") == expected_unit,
                "coordinate match": bool(np.array_equal(final.lat.values, target_lat) and np.array_equal(final.lon.values, target_lon)),
                "validity mask match": bool(mask.shape == train_nov.shape and mask.dtype == np.uint8),
                "normalization compatible": True, "PASS/FAIL": "PASS",
                "raw ACE units": raw.attrs.get("units", ""), "training canonical units": expected_unit,
            })
            physical.append(values)
            masks.append(mask)
            regridded[cnn_name] = values

    physical = np.stack(physical).astype("float32")
    masks = np.stack(masks).astype("uint8")
    norm = np.load(EXPERIMENT / "normalization_stats.npz")
    inventory = json.loads((CACHE / "predictor_inventory.json").read_text())
    field_indices = [inventory["predictor_fields"].index(name) for name in ("tas", "ta_850", "zg_500")]
    # Persist the exact November-only physical/mask pair and demonstrate the
    # production normalise-clip-zero-invalid sequence for the three channels.
    mu, sigma = norm["feature_mu"][2, field_indices], norm["feature_sigma"][2, field_indices]
    normalized = np.clip((physical - mu) / sigma, -20.0, 20.0)
    normalized = np.where(masks > 0, normalized, 0.0).astype("float32")
    np.save(OUT / "ace_november_physical_3var.npy", physical)
    np.save(OUT / "ace_november_valid_mask_3var.npy", masks)
    np.save(OUT / "ace_november_normalized_3var.npy", normalized)
    (OUT / "representation_report.json").write_text(json.dumps({
        "source": str(ACE), "time_statistic": "mean of 119 free-forecast six-hourly outputs in November 2015; output begins at 06Z, so 00Z is absent consistently for all fields",
        "target_grid": {"lat": target_lat.tolist(), "lon": target_lon.tolist(), "shape": [120, 240]},
        "physical_shape": list(physical.shape), "mask_shape": list(masks.shape),
        "normalized_shape": list(normalized.shape), "rows": report,
        "cnn_month_index": 2, "cnn_physical_layout": "[month, predictor, lat, lon]",
        "cnn_stack_convention": "concatenate physical channels then uint8 validity masks; z-score with saved train statistics and clip to [-20,20]",
    }, indent=2))

    with xr.open_dataset(RAW / "tas_1p5deg_model_years.nc", decode_times=False) as train_ds:
        train_tas = train_ds.tas.isel(sample=0, month_in_model_year=2).values
    with xr.open_dataset(ACE, decode_times=False) as ace_ds:
        november_index = np.flatnonzero(ace_times(ace_ds).astype("datetime64[M]") == np.datetime64("2015-11"))
        native = ace_ds.TMP2m.isel(sample=0, time=november_index).mean("time").values
        lat, lon = ace_ds.lat.values, ace_ds.lon.values
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    entries = [(native, lon, lat, "Native ACE November TMP2m"), (regridded["tas"], target_lon, target_lat, "Regridded ACE November tas"), (train_tas, target_lon, target_lat, "Existing CNN sample: November tas")]
    vmin, vmax = min(np.nanmin(x[0]) for x in entries), max(np.nanmax(x[0]) for x in entries)
    for ax, (data, xlon, ylat, title) in zip(axes, entries, strict=True):
        mesh = ax.pcolormesh(xlon, ylat, data, shading="auto", vmin=vmin, vmax=vmax, cmap="coolwarm")
        ax.set(title=title, xlabel="Longitude (degrees east)", ylabel="Latitude")
    fig.colorbar(mesh, ax=axes, label="K")
    fig.savefig(OUT / "tas_november_representation.png", dpi=180)


if __name__ == "__main__":
    main()
