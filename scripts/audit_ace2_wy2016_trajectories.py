#!/usr/bin/env python3
"""Read-only quality audit for the existing ACE2-ERA5 WY2016 lag ensemble.

The output files are already on the ERA5 1-degree Gaussian grid.  This script
therefore compares the saved forecasts to the source ERA5 archive without
regridding, and writes only derived diagnostics below the repository artifacts
directory.
"""
from __future__ import annotations

import json
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr


ACE_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
ERA5_ZARR = Path("/global/cfs/projectdirs/m4581/aimip/ERA5_initialization/2024-06-20-era5-1deg-8layer-1940-2022.zarr")
OUT = Path("artifacts/ace2_wy2016_trajectory_audit")
MEMBERS = ["20151101", "20151102", "20151103", "20151104", "20151105"]
STARTS = {member: pd.Timestamp(f"{member[:4]}-{member[4:6]}-{member[6:]}T00:00:00") for member in MEMBERS}
COMMON_END = pd.Timestamp("2016-04-01T00:00:00")
# The only saved forecast fields that faithfully match current CNN predictors.
COMPARE = {"TMP2m": "TMP2m", "TMP850": "TMP850", "h500": "h500", "ULWRFtoa": "ULWRFtoa", "PRATEsfc": "PRATEsfc"}
CNN_MAP = {
    "tos": ("not available", "", "ACE has no prognostic ocean surface temperature output"),
    "siconc": ("not available", "", "No prognostic sea-ice concentration output"),
    "rlut": ("exact, pending sign metadata check", "ULWRFtoa", "TOA upward longwave flux; retain only if sign matches CMIP rlut"),
    "zg_500": ("exact", "h500", "500-hPa geopotential height in metres"),
    "ta_850": ("exact", "TMP850", "850-hPa air temperature in K"),
    "ua_850": ("not defensibly identified", "eastward_wind_0..7", "Saved model-level winds lack verified pressure-level mapping"),
    "va_850": ("not defensibly identified", "northward_wind_0..7", "Saved model-level winds lack verified pressure-level mapping"),
    "ua_200": ("not defensibly identified", "eastward_wind_0..7", "Saved model-level winds lack verified pressure-level mapping"),
    "va_200": ("not defensibly identified", "northward_wind_0..7", "Saved model-level winds lack verified pressure-level mapping"),
    "hus_850": ("not defensibly identified", "specific_total_water_0..7", "Total water is not verified 850-hPa specific humidity"),
    "psl": ("not available", "PRESsfc", "Surface pressure is not sea-level pressure"),
    "tas": ("exact", "TMP2m", "2-m air temperature in K"),
    "zg_50": ("not available", "", "No 50-hPa height saved"),
    "ta_50": ("not available", "", "No 50-hPa temperature saved"),
    "ua_50": ("not available", "", "No 50-hPa zonal wind saved"),
    "va_50": ("not available", "", "No 50-hPa meridional wind saved"),
    "mrso": ("not available", "", "No prognostic soil-moisture output saved"),
    "thetao_50m": ("not available", "", "No subsurface ocean-temperature output saved"),
    "thetao_100m": ("not available", "", "No subsurface ocean-temperature output saved"),
}


def weights(lat: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat))[:, None]


def weighted_mean(a: np.ndarray, w: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
    if region is not None:
        w = w * region
    good = np.isfinite(a)
    numerator = np.nansum(a * w, axis=(-2, -1))
    denominator = np.sum(w * good, axis=(-2, -1))
    return numerator / denominator


def spatial_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        x, y = a[i].ravel(), b[i].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2:
            out[i] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out


def calendar_times(raw: np.ndarray, start: pd.Timestamp) -> pd.DatetimeIndex:
    # ACE prediction files store elapsed microseconds and exclude the initial state.
    return pd.DatetimeIndex(start + pd.to_timedelta(raw.astype("int64"), unit="us"))


def era5_indices(times: pd.DatetimeIndex) -> np.ndarray:
    """Return source-array indices, validated against the archive time coordinate."""
    origin = pd.Timestamp("1940-01-01T12:00:00")
    return ((times - origin).total_seconds() / 21600).astype("int64")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Opening the full group exhausts metadata resources on login nodes.  The
    # five direct arrays are sufficient and preserve native grid alignment.
    ref = {name: zarr.open_array(str(ERA5_ZARR / name), mode="r") for name in COMPARE.values()}
    ref_lat = zarr.open_array(str(ERA5_ZARR / "latitude"), mode="r")[:]
    ref_lon = zarr.open_array(str(ERA5_ZARR / "longitude"), mode="r")[:]
    records, trajectories, inventories = [], [], {}
    mean_bias_maps: list[np.ndarray] = []
    for member in MEMBERS:
        print(f"AUDIT member={member} open", flush=True)
        path = ACE_ROOT / f"member_{member}" / "autoregressive_predictions.nc"
        cfg = ACE_ROOT / "configs" / f"member_{member}.yaml"
        ds = xr.open_dataset(path, decode_times=False)
        times = calendar_times(ds.time.values, STARTS[member])
        lat, lon = ds.lat.values, ds.lon.values
        if not (np.allclose(lat, ref_lat, atol=1e-5) and np.allclose(lon, ref_lon, atol=1e-7)):
            raise RuntimeError("ACE and source ERA5 grids do not exactly match")
        w = weights(lat)
        region = ((lat[:, None] >= 35.0) & (lat[:, None] <= 42.0) & (lon[None, :] >= 237.5) & (lon[None, :] <= 242.0))
        variables = {}
        for name in ds.data_vars:
            values = ds[name].values
            variables[name] = {"units": ds[name].attrs.get("units", ""), "shape": list(values.shape), "nonfinite": int((~np.isfinite(values)).sum())}
        print(f"AUDIT member={member} inventory complete", flush=True)
        inventories[member] = {
            "path": str(path), "size_bytes": path.stat().st_size, "init": str(STARTS[member]), "final": str(times[-1]),
            "steps": len(times), "cadence_hours": float((times[1] - times[0]).total_seconds() / 3600),
            "grid": {"lat": len(lat), "lon": len(lon), "lat_first": float(lat[0]), "lat_last": float(lat[-1]), "lon_first": float(lon[0]), "lon_last": float(lon[-1])},
            "config": str(cfg), "initial_condition": str(ACE_ROOT / "initial_conditions" / f"ic_{STARTS[member]:%Y-%m-%d}.nc"),
            "variables": variables,
        }
        common = times <= COMMON_END
        for ace_name, era_name in COMPARE.items():
            print(f"AUDIT member={member} variable={ace_name}", flush=True)
            a = np.asarray(ds[ace_name].values, dtype="float64")
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            ref_index = era5_indices(times)
            if not np.all(np.diff(ref_index) == 1):
                raise RuntimeError("Expected contiguous six-hour ERA5 verification times")
            b = np.asarray(ref[era_name][ref_index[0] : ref_index[-1] + 1, :, :], dtype="float64")
            d = a - b
            global_a, global_b = weighted_mean(a, w), weighted_mean(b, w)
            reg_a, reg_b = weighted_mean(a, w, region), weighted_mean(b, w, region)
            row = {
                "member": member, "variable": ace_name, "units": ds[ace_name].attrs.get("units", ""),
                "n_times": len(times), "mean_bias": float(np.nanmean(d)), "rmse": float(np.sqrt(np.nanmean(d ** 2))),
                "spatial_corr_mean": float(np.nanmean(spatial_corr(a, b))),
                "global_temporal_corr": float(np.corrcoef(global_a, global_b)[0, 1]),
                "region_mean_bias": float(np.nanmean(d[:, region])), "region_rmse": float(np.sqrt(np.nanmean(d[:, region] ** 2))),
                "region_temporal_corr": float(np.corrcoef(reg_a, reg_b)[0, 1]),
            }
            records.append(row)
            for i, time in enumerate(times):
                trajectories.append({"member": member, "time": str(time), "lead_days": (time - STARTS[member]).total_seconds() / 86400,
                    "variable": ace_name, "global_ace": global_a[i], "global_era5": global_b[i], "region_ace": reg_a[i], "region_era5": reg_b[i],
                    "global_bias": global_a[i] - global_b[i], "region_bias": reg_a[i] - reg_b[i], "global_rmse": float(np.sqrt(np.nanmean(d[i] ** 2))), "region_rmse": float(np.sqrt(np.nanmean(d[i, region] ** 2)))} )
            if ace_name == "TMP2m":
                mean_bias_maps.append(np.nanmean(d[common], axis=0))
            del a, b, d
            gc.collect()
        ds.close()
        print(f"AUDIT member={member} complete", flush=True)
    pd.DataFrame(records).to_csv(OUT / "member_variable_metrics.csv", index=False)
    traj = pd.DataFrame(trajectories)
    traj.to_csv(OUT / "time_series_metrics.csv", index=False)
    (OUT / "member_inventory.json").write_text(json.dumps(inventories, indent=2, sort_keys=True))
    (OUT / "cnn_variable_mapping.json").write_text(json.dumps(CNN_MAP, indent=2, sort_keys=True))

    # Common calendar dates are used only for ensemble spread, not for lead-error curves.
    common = traj[(pd.to_datetime(traj.time) >= pd.Timestamp("2015-11-05T06")) & (pd.to_datetime(traj.time) <= COMMON_END)]
    spread = common.groupby(["time", "variable"]).agg(global_ensemble_std=("global_ace", "std"), region_ensemble_std=("region_ace", "std"), n=("member", "count")).reset_index()
    spread.to_csv(OUT / "ensemble_spread.csv", index=False)

    tmp = traj[traj.variable == "TMP2m"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for member, x in tmp.groupby("member"):
        ax.plot(x.lead_days, x.region_bias, label=member)
    ax.axhline(0, color="black", lw=0.8); ax.set(xlabel="Forecast lead (days)", ylabel="ACE - ERA5 TMP2m (K)", title="CNN target rectangle: 35-42N, 122.5-118W")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "tmp2m_region_bias_by_lead.png", dpi=160); plt.close(fig)

    pr = traj[traj.variable == "PRATEsfc"].copy()
    pr["dt_seconds"] = 21600.0
    for col in ("region_ace", "region_era5"):
        pr[col] = pr[col] * pr.dt_seconds
    fig, ax = plt.subplots(figsize=(9, 4))
    for member, x in pr.groupby("member"):
        x = x.sort_values("lead_days"); ax.plot(x.lead_days, x.region_ace.cumsum(), label=f"ACE {member}")
    x = pr[pr.member == "20151101"].sort_values("lead_days"); ax.plot(x.lead_days, x.region_era5.cumsum(), color="black", lw=2, label="ERA5")
    ax.set(xlabel="Forecast lead (days)", ylabel="Accumulated precipitation (mm)", title="CNN target rectangle"); ax.legend(ncol=3, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "precip_region_accumulation.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for variable, x in spread.groupby("variable"):
        if variable != "PRATEsfc": ax.plot(pd.to_datetime(x.time), x.region_ensemble_std, label=variable)
    ax.set(xlabel="Calendar date", ylabel="Ensemble standard deviation", title="Regional lag-ensemble spread"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "ensemble_spread.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for variable, x in traj.groupby("variable"):
        ax.plot(x.lead_days, x.region_rmse, ".", ms=1, alpha=.25, label=variable)
    ax.set(xlabel="Forecast lead (days)", ylabel="Regional RMSE", title="ACE error growth by lead time"); ax.legend(ncol=3, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "regional_error_growth.png", dpi=160); plt.close(fig)

    lat_slice = (lat >= 32) & (lat <= 45); lon_slice = (lon >= 232) & (lon <= 248)
    mean_map = np.nanmean(mean_bias_maps, axis=0)
    fig, ax = plt.subplots(figsize=(7, 4)); m = ax.pcolormesh(lon[lon_slice] - 360, lat[lat_slice], mean_map[np.ix_(lat_slice, lon_slice)], cmap="coolwarm")
    ax.add_patch(plt.Rectangle((-122.5, 35), 4.5, 7, fill=False, ec="black", lw=1.5)); ax.set(xlabel="Longitude", ylabel="Latitude", title="Mean ACE - ERA5 TMP2m bias (all members, available shared leads)")
    fig.colorbar(m, ax=ax, label="K"); fig.tight_layout(); fig.savefig(OUT / "tmp2m_bias_map_target_region.png", dpi=160); plt.close(fig)

    print(OUT)


if __name__ == "__main__":
    main()
