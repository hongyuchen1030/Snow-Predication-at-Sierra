#!/usr/bin/env python3
"""Build the November-only ACE5 eight-variable pretraining representation.

The production CMIP6 workflow uses CDO bilinear remapping.  CDO is unavailable
in this environment; for ACE's regular global lat/lon grid this uses linear
bilinear interpolation onto the same target centers and records that fact.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
RAW = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_regridded_1p5deg/raw")
CACHE = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_17var_baseline_v1/data_cache")
EXPERIMENT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_17var_baseline_v1/experiments/S0_17var_static_cnn_swe_only")
OUT = Path("artifacts/ace5_november_8var")
MEMBERS = ["20151101", "20151102", "20151103", "20151104", "20151105"]
CHANNELS = ["tas", "ta_850", "zg_500", "rlut", "ua_850", "va_850", "ua_200", "va_200"]
DIRECT = {"tas": "TMP2m", "ta_850": "TMP850", "zg_500": "h500", "rlut": "ULWRFtoa"}
WIND = {"ua_850": ("eastward_wind", 85000.0), "va_850": ("northward_wind", 85000.0), "ua_200": ("eastward_wind", 20000.0), "va_200": ("northward_wind", 20000.0)}
UNITS = {"tas": "K", "ta_850": "K", "zg_500": "m", "rlut": "W/m**2", "ua_850": "m s**-1", "va_850": "m s**-1", "ua_200": "m s**-1", "va_200": "m s**-1"}


def times_from_output(ds: xr.Dataset, start: str) -> np.ndarray:
    return (np.datetime64(start) + ds.time.values.astype("timedelta64[us]")).astype("datetime64[ns]")


def layer_pressure_centers(ps: np.ndarray, ak: np.ndarray, bk: np.ndarray) -> np.ndarray:
    """Log-mean pressure for each finite-volume layer from its interfaces."""
    interface = ak[:, None, None, None] + bk[:, None, None, None] * ps[None, :, :, :]
    top, bottom = interface[:-1], interface[1:]
    # The target levels use layers 1--7; retain a finite arithmetic fallback
    # for layer 0, whose top interface is 0 Pa.
    with np.errstate(divide="ignore", invalid="ignore"):
        centers = (bottom - top) / (np.log(bottom) - np.log(top))
    centers = np.where(np.isfinite(centers), centers, 0.5 * (top + bottom))
    return centers


def interpolate_log_pressure(values: np.ndarray, center_p: np.ndarray, target_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate eight layer values at target_p without vertical extrapolation."""
    # Input shape is [time, layer, lat, lon]; pressures share that layout.
    ntime, nlayer, nlat, nlon = values.shape
    out = np.full((ntime, nlat, nlon), np.nan, dtype=np.float64)
    valid = np.zeros((ntime, nlat, nlon), dtype=bool)
    lp = np.log(center_p)
    for k in range(nlayer - 1):
        lo, hi = center_p[:, k], center_p[:, k + 1]
        bracket = (lo <= target_p) & (target_p <= hi)
        if not bracket.any():
            continue
        frac = (np.log(target_p) - lp[:, k]) / (lp[:, k + 1] - lp[:, k])
        candidate = values[:, k] + frac * (values[:, k + 1] - values[:, k])
        out = np.where(bracket, candidate, out)
        valid |= bracket
    return out, valid


def remap(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    return da.interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": "extrapolate"})


def stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    good = values[mask.astype(bool)]
    return {"finite_fraction": float(np.isfinite(values).mean()), "min": float(np.nanmin(good)), "max": float(np.nanmax(good)), "mean": float(np.nanmean(good)), "std": float(np.nanstd(good)), "validity_mask_fraction": float(mask.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="*", choices=MEMBERS, default=MEMBERS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(RAW / "tas_1p5deg_model_years.nc", decode_times=False) as d:
        target_lat, target_lon = d.lat.values, d.lon.values
    with xr.open_dataset(ROOT / "forcing_2015_2016" / "forcing_2015.nc", decode_times=False) as d:
        ak = np.array([d[f"ak_{i}"].item() for i in range(9)], dtype=np.float64)
        bk = np.array([d[f"bk_{i}"].item() for i in range(9)], dtype=np.float64)
    inventory = json.loads((CACHE / "predictor_inventory.json").read_text())
    indices = [inventory["predictor_fields"].index(x) for x in CHANNELS]
    norm = np.load(EXPERIMENT / "normalization_stats.npz")
    mu, sigma = norm["feature_mu"][2, indices], norm["feature_sigma"][2, indices]
    combined, combined_masks, combined_norm, summary_rows, member_meta = [], [], [], [], []
    for member in args.members:
        start = f"{member[:4]}-{member[4:6]}-{member[6:]}T00:00:00"
        source = ROOT / f"member_{member}" / "autoregressive_predictions.nc"
        with xr.open_dataset(source, decode_times=False) as ds:
            times = times_from_output(ds, start)
            take = np.flatnonzero(times.astype("datetime64[M]") == np.datetime64("2015-11"))
            if len(take) == 0:
                raise RuntimeError(f"{member} has no November output")
            fields: dict[str, np.ndarray] = {}
            validity: dict[str, np.ndarray] = {}
            raw_shape = [len(take), 180, 360]
            for channel, ace_name in DIRECT.items():
                values = ds[ace_name].isel(sample=0, time=take).values.astype(np.float64)
                fields[channel] = np.nanmean(values, axis=0)
                validity[channel] = np.isfinite(fields[channel])
            ps = ds.PRESsfc.isel(sample=0, time=take).values.astype(np.float64)
            center_p = layer_pressure_centers(ps, ak, bk).transpose(1, 0, 2, 3)
            for channel, (stem, target_p) in WIND.items():
                layers = np.stack([ds[f"{stem}_{i}"].isel(sample=0, time=take).values for i in range(8)], axis=1).astype(np.float64)
                instantaneous, valid = interpolate_log_pressure(layers, center_p, target_p)
                below_surface = ps < target_p
                fields[channel] = np.nanmean(instantaneous, axis=0)
                validity[channel] = np.isfinite(fields[channel])
                member_meta.append({"member": member, "channel": channel, "target_pressure_pa": target_p, "interpolation": "linear in log pressure between log-mean hybrid layer-center pressures", "native_valid_fraction_over_time_space": float(valid.mean()), "below_surface_fraction": float(below_surface.mean()), "vertical_extrapolation_count": int((~valid & ~below_surface).sum())})
            physical, masks = [], []
            for channel in CHANNELS:
                native = xr.DataArray(fields[channel], dims=("lat", "lon"), coords={"lat": ds.lat.values, "lon": ds.lon.values})
                mapped = remap(native, target_lat, target_lon).values.astype("float32")
                mask = np.isfinite(mapped).astype("uint8")
                physical.append(mapped); masks.append(mask)
                row = {"member": member, "channel": channel, "units": UNITS[channel], "raw_source_shape": raw_shape, "november_timesteps": int(len(take)), "monthly_native_shape": [180, 360], "final_grid_shape": [120, 240]}
                row.update(stats(mapped, mask)); summary_rows.append(row)
        physical = np.stack(physical); masks = np.stack(masks)
        normalized = np.clip((physical - mu) / sigma, -20.0, 20.0)
        normalized = np.where(masks > 0, normalized, 0.0).astype("float32")
        member_dir = OUT / f"member_{member}"; member_dir.mkdir(exist_ok=True)
        np.save(member_dir / "physical.npy", physical); np.save(member_dir / "valid_mask.npy", masks); np.save(member_dir / "normalized.npy", normalized)
        metadata = {"member": member, "initialization": start, "source": str(source), "forecast_timestamps_used": [str(times[i]) for i in take], "n_six_hourly_samples": int(len(take)), "channels": CHANNELS, "units": UNITS, "target_grid": {"lat_first": float(target_lat[0]), "lat_last": float(target_lat[-1]), "lon_first": float(target_lon[0]), "lon_last": float(target_lon[-1]), "shape": [120, 240]}, "remapping": "regular-grid bilinear interpolation equivalent to CDO bilinear; CDO executable unavailable", "normalization": "saved S0 November train stats; z-score then clip [-20,20], invalid values set to zero", "vertical_coordinate": {"interface_equation": "p_interface = ak + bk * PRESsfc", "interpolation": "log-pressure interpolation; no extrapolation; 850 hPa below-surface/unsupported points masked"}, "wind_diagnostics": [x for x in member_meta if x["member"] == member]}
        (member_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        combined.append(physical); combined_masks.append(masks); combined_norm.append(normalized)
    if args.members != MEMBERS:
        return
    combined = np.stack(combined); combined_masks = np.stack(combined_masks); combined_norm = np.stack(combined_norm)
    np.save(OUT / "ACE5_November_physical.npy", combined); np.save(OUT / "ACE5_November_valid_mask.npy", combined_masks); np.save(OUT / "ACE5_November_normalized.npy", combined_norm)
    pd.DataFrame(summary_rows).to_csv(OUT / "channel_summary.csv", index=False)
    pd.DataFrame(member_meta).to_csv(OUT / "wind_interpolation_diagnostics.csv", index=False)
    spread = np.nanstd(combined, axis=0)
    np.save(OUT / "ACE5_November_ensemble_std.npy", spread)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    for idx, (ax, channel) in enumerate(zip(axes.ravel(), CHANNELS, strict=True)):
        mesh = ax.pcolormesh(target_lon, target_lat, spread[idx], shading="auto", cmap="magma")
        ax.set(title=channel, xlabel="Longitude", ylabel="Latitude"); fig.colorbar(mesh, ax=ax, shrink=.8)
    fig.suptitle("ACE5 November ensemble standard deviation on the CNN grid")
    fig.savefig(OUT / "ACE5_November_ensemble_spread_8var.png", dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
