#!/usr/bin/env python
"""Frozen-ClimaX observational (ERA5) latent extraction, March 25 snapshot only.

Minimal observational experiment, per explicit scope: for each water year
WY1985-WY2021, recover the genuine daily-mean global ERA5 state (10 approved
ClimaX variables) for March 25 of that water year, from the raw ERA5 hourly
archive (not interpolated/derived from any monthly product), regrid to the
official ClimaX 1.40625deg/128x256 grid, run the frozen pretrained ClimaX
encoder, and pair each resulting H_y with the existing April-1 Sierra SWE
anomaly target for that same water year. 37 paired samples total.

No SWE/CPM/AQM correlation, no probing, no other dates, no daily UCLA SWE.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import torch
import xarray as xr

REPO_ROOT = "/global/u1/h/hyvchen/Snow-Predication-at-Sierra"
CLIMAX_SRC = os.path.join(REPO_ROOT, "thirdparties/ClimaX/src")
sys.path.insert(0, CLIMAX_SRC)

from climax.arch import ClimaX  # noqa: E402

ERA5_ROOT = "/global/cfs/projectdirs/m3522/datalake/ERA5"
AN_SFC_ROOT = os.path.join(ERA5_ROOT, "e5.oper.an.sfc")
AN_PL_ROOT = os.path.join(ERA5_ROOT, "e5.oper.an.pl")

PSCRATCH_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_era5_march25_obs_latents"
NATIVE_DIR = os.path.join(PSCRATCH_ROOT, "native_0p25deg")
REGRID_DIR = os.path.join(PSCRATCH_ROOT, "regridded_1p40625deg")
LATENT_DIR = os.path.join(PSCRATCH_ROOT, "latents")
GRID_FILE = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_frozen_mpi_smoke_test/climax_1p40625deg_grid/target_grid.txt"
CKPT_PATH = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_frozen_mpi_smoke_test/checkpoints/1.40625deg.ckpt"
HOME_ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts/climax_era5_march25_obs_latents")

SWE_ANOM_NC = (
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/"
    "targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc"
)

WATER_YEARS = list(range(1985, 2022))  # WY1985 .. WY2021, 37 samples

# field -> (era5 var_code, file_varname, level_hpa or None, source_family)
FIELD_SOURCE = {
    "zg_500": ("128_129_z", "Z", 500.0, "an.pl"),
    "zg_50": ("128_129_z", "Z", 50.0, "an.pl"),
    "ta_850": ("128_130_t", "T", 850.0, "an.pl"),
    "ta_50": ("128_130_t", "T", 50.0, "an.pl"),
    "ua_850": ("128_131_u", "U", 850.0, "an.pl"),
    "ua_50": ("128_131_u", "U", 50.0, "an.pl"),
    "va_850": ("128_132_v", "V", 850.0, "an.pl"),
    "va_50": ("128_132_v", "V", 50.0, "an.pl"),
    "hus_850": ("128_133_q", "Q", 850.0, "an.pl"),
    "tas": ("128_167_2t", "VAR_2T", None, "an.sfc"),
}
AN_PL_WIND_VAR_CODES = {"128_131_u", "128_132_v"}

# Approved 10-variable subset -> ClimaX vocabulary name.
VAR_TO_CLIMAX = {
    "zg_500": "geopotential_500",
    "zg_50": "geopotential_50",
    "ta_850": "temperature_850",
    "ta_50": "temperature_50",
    "ua_850": "u_component_of_wind_850",
    "ua_50": "u_component_of_wind_50",
    "va_850": "v_component_of_wind_850",
    "va_50": "v_component_of_wind_50",
    "hus_850": "specific_humidity_850",
    "tas": "2m_temperature",
}
VARKEYS = list(VAR_TO_CLIMAX.keys())

# Unlike CMIP6 zg (geopotential HEIGHT, m -- needs *9.807 for ClimaX), ERA5's own Z
# variable is already genuine geopotential (m**2 s**-2, confirmed directly from the
# raw file's units/long_name attrs) -- the same units ClimaX's "geopotential_*"
# expects. No unit conversion is applied to any of the 10 variables here.

CLIMAX_DEFAULT_VARS = [
    "land_sea_mask", "orography", "lattitude",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_50", "geopotential_250", "geopotential_500", "geopotential_600",
    "geopotential_700", "geopotential_850", "geopotential_925",
    "u_component_of_wind_50", "u_component_of_wind_250", "u_component_of_wind_500",
    "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850",
    "u_component_of_wind_925",
    "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500",
    "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850",
    "v_component_of_wind_925",
    "temperature_50", "temperature_250", "temperature_500", "temperature_600",
    "temperature_700", "temperature_850", "temperature_925",
    "relative_humidity_50", "relative_humidity_250", "relative_humidity_500",
    "relative_humidity_600", "relative_humidity_700", "relative_humidity_850",
    "relative_humidity_925",
    "specific_humidity_50", "specific_humidity_250", "specific_humidity_500",
    "specific_humidity_600", "specific_humidity_700", "specific_humidity_850",
    "specific_humidity_925",
]
assert len(CLIMAX_DEFAULT_VARS) == 48

IMG_SIZE = [128, 256]
PATCH_SIZE = 4
EMBED_DIM = 1024
DEPTH = 8
DECODER_DEPTH = 2
NUM_HEADS = 16
MLP_RATIO = 4.0


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_an_pl_day(var_code: str, file_varname: str, level_hpa: float, target_date: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_tag = "ll025uv" if var_code in AN_PL_WIND_VAR_CODES else "ll025sc"
    month_dir = os.path.join(AN_PL_ROOT, f"{target_date.year:04d}{target_date.month:02d}")
    ymd = f"{target_date.year:04d}{target_date.month:02d}{target_date.day:02d}"
    fname = f"e5.oper.an.pl.{var_code}.{grid_tag}.{ymd}00_{ymd}23.nc"
    path = os.path.join(month_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected ERA5 an.pl day file not found: {path}")
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        da = ds[file_varname].sel(level=level_hpa)
        actual_level = float(np.asarray(da["level"].values))
        if abs(actual_level - level_hpa) > 1e-6:
            raise ValueError(f"Level mismatch: wanted {level_hpa}, found {actual_level} in {path}")
        vals = np.asarray(da.values, dtype=np.float64)  # (24, lat, lon)
        assert vals.shape[0] == 24, f"expected 24 hourly steps, found {vals.shape[0]} in {path}"
        daily_mean = np.nanmean(vals, axis=0).astype(np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return daily_mean, lat, lon


def load_an_sfc_day(var_code: str, file_varname: str, target_date: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    month_dir = os.path.join(AN_SFC_ROOT, f"{target_date.year:04d}{target_date.month:02d}")
    import glob
    matches = sorted(glob.glob(os.path.join(month_dir, f"e5.oper.an.sfc.{var_code}.ll025sc.{target_date.year:04d}{target_date.month:02d}*.nc")))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one an.sfc month file for {var_code} {target_date}, found {len(matches)}")
    path = matches[0]
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as ds:
        day_slice = ds[file_varname].sel(time=ds["time"].dt.day == target_date.day)
        n_hours = day_slice.sizes["time"]
        if n_hours != 24:
            raise ValueError(f"Expected 24 hourly steps for {target_date} in {path}, found {n_hours}")
        daily_mean = np.asarray(day_slice.mean(dim="time", skipna=True).values, dtype=np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return daily_mean, lat, lon


def build_native_march25_stack() -> dict[str, str]:
    """One NetCDF per variable, 37 timesteps (March 25 of each WY1985-2021),
    on ERA5's native 0.25deg global grid, no unit conversion."""
    os.makedirs(NATIVE_DIR, exist_ok=True)
    out_paths: dict[str, str] = {}
    for vk in VARKEYS:
        var_code, file_varname, level_hpa, family = FIELD_SOURCE[vk]
        out_path = os.path.join(NATIVE_DIR, f"{vk}_march25_native.nc")
        out_paths[vk] = out_path
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue
        fields = []
        lat_ref = lon_ref = None
        for wy in WATER_YEARS:
            target_date = date(wy, 3, 25)
            if family == "an.pl":
                field, lat, lon = load_an_pl_day(var_code, file_varname, level_hpa, target_date)
            else:
                field, lat, lon = load_an_sfc_day(var_code, file_varname, target_date)
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            else:
                assert np.array_equal(lat, lat_ref) and np.array_equal(lon, lon_ref)
            fields.append(field)
            print(f"  [{vk}] WY{wy} ({target_date.isoformat()}): "
                  f"mean={np.nanmean(field):.4f} nan={int(np.isnan(field).sum())}", flush=True)
        stacked = np.stack(fields, axis=0)
        times = np.array([f"{wy}-03-25" for wy in WATER_YEARS], dtype="datetime64[D]")
        da = xr.DataArray(
            stacked, dims=("time", "lat", "lon"),
            coords={"time": times, "lat": lat_ref, "lon": lon_ref}, name=vk,
        )
        ds_out = da.to_dataset()
        ds_out["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
        ds_out["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
        ds_out.to_netcdf(out_path, engine="netcdf4")
        print(f"  wrote {out_path}", flush=True)
    return out_paths


def regrid_to_climax_grid(native_paths: dict[str, str]) -> dict[str, str]:
    os.makedirs(REGRID_DIR, exist_ok=True)
    out_paths: dict[str, str] = {}
    for vk, infile in native_paths.items():
        outfile = os.path.join(REGRID_DIR, f"{vk}_march25_1p40625deg.nc")
        out_paths[vk] = outfile
        if os.path.exists(outfile):
            print(f"[skip] {outfile} already exists")
            continue
        run(["cdo", "-O", f"remapbil,{GRID_FILE}", infile, outfile])
    return out_paths


def verify_grid(lat: np.ndarray, lon: np.ndarray) -> None:
    expected_lat = -90 + 1.40625 / 2 + np.arange(128) * 1.40625
    expected_lon = np.arange(256) * 1.40625
    assert lat.shape[0] == 128 and lon.shape[0] == 256, (lat.shape, lon.shape)
    assert np.allclose(lat, expected_lat, atol=1e-6), "lat grid mismatch vs ClimaX target grid"
    assert np.allclose(lon, expected_lon, atol=1e-6), "lon grid mismatch vs ClimaX target grid"
    assert lat[0] < lat[-1], "lat must be ascending south-to-north (ClimaX convention)"


def build_model() -> torch.nn.Module:
    model = ClimaX(
        default_vars=CLIMAX_DEFAULT_VARS, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM, depth=DEPTH, decoder_depth=DECODER_DEPTH,
        num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, drop_path=0.0, drop_rate=0.0,
        parallel_patch_embed=False,
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    raw_sd = ckpt["state_dict"]
    rename = {
        "net.channel_embed": "net.var_embed",
        "net.channel_query": "net.var_query",
        "net.channel_agg.in_proj_weight": "net.var_agg.in_proj_weight",
        "net.channel_agg.in_proj_bias": "net.var_agg.in_proj_bias",
        "net.channel_agg.out_proj.weight": "net.var_agg.out_proj.weight",
        "net.channel_agg.out_proj.bias": "net.var_agg.out_proj.bias",
    }
    sd = {}
    for k, v in raw_sd.items():
        k2 = rename.get(k, k)
        assert k2.startswith("net.")
        sd[k2[len("net."):]] = v
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def param_sum(model: torch.nn.Module) -> float:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).sum().item()


def main() -> None:
    os.makedirs(LATENT_DIR, exist_ok=True)
    os.makedirs(HOME_ARTIFACT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    print("=== Step 1: build native 0.25deg March-25 stacks (37 WY, 10 vars, genuine daily mean from raw hourly ERA5) ===", flush=True)
    native_paths = build_native_march25_stack()

    print("=== Step 2: CDO regrid native 0.25deg -> ClimaX 1.40625deg/128x256 ===", flush=True)
    regridded_paths = regrid_to_climax_grid(native_paths)

    print("=== Step 3: load regridded fields, verify grid, compute normalization stats over the 37 samples ===", flush=True)
    stats = {}
    fields = {}
    lat_ref = lon_ref = None
    n_nan_total = 0
    n_total = 0
    for vk in VARKEYS:
        ds = xr.open_dataset(regridded_paths[vk])
        da = ds[vk]
        lat = ds["lat"].values
        lon = ds["lon"].values
        verify_grid(lat, lon)
        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        else:
            assert np.allclose(lat, lat_ref) and np.allclose(lon, lon_ref)
        arr = da.values.astype(np.float64)
        assert arr.shape[0] == len(WATER_YEARS), (vk, arr.shape)
        n_nan_total += int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
        n_total += arr.size
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        stats[vk] = {
            "mean": mean, "std": std,
            "stats_source": "computed over these same 37 March-25 ERA5 global snapshots (WY1985-2021); "
                             "no downstream SWE/CPM/AQM/obs label used",
        }
        fields[vk] = arr.astype(np.float32)
        ds.close()
        print(f"  {vk}: mean={mean:.6f} std={std:.6f}", flush=True)

    print(f"  total input NaN/Inf: {n_nan_total} / {n_total}", flush=True)
    assert n_nan_total == 0, f"found {n_nan_total} NaN/Inf values in regridded ERA5 March-25 input"

    n_samples = fields[VARKEYS[0]].shape[0]
    assert n_samples == 37 == len(WATER_YEARS)

    print("=== Step 4: build normalized input tensor [37,10,128,256] ===", flush=True)
    x = np.zeros((n_samples, len(VARKEYS), 128, 256), dtype=np.float32)
    for i, vk in enumerate(VARKEYS):
        mean, std = stats[vk]["mean"], stats[vk]["std"]
        x[:, i] = (fields[vk] - mean) / std
    variables = [VAR_TO_CLIMAX[vk] for vk in VARKEYS]
    print("  input tensor shape:", x.shape, " ClimaX variable order:", variables, flush=True)

    print("=== Step 5: build frozen ClimaX model, load official checkpoint ===", flush=True)
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == 0
    param_before = param_sum(model)
    print(f"  total params: {n_params:,}  trainable: {n_trainable} (must be 0)  param_sum={param_before}", flush=True)

    print("=== Step 6: run frozen encoder for all 37 samples ===", flush=True)
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        lead_times = torch.zeros(x_t.shape[0], device=device, dtype=x_t.dtype)
        H = model.forward_encoder(x_t.to(device), lead_times, variables).detach().cpu().numpy().astype(np.float32)
    print(f"  H shape: {H.shape} (expect (37, 2048, 1024))", flush=True)
    assert H.shape == (37, 2048, 1024), H.shape
    assert not np.isnan(H).any() and not np.isinf(H).any()

    param_after = param_sum(model)
    assert param_before == param_after, "model parameters changed during forward pass!"

    print("=== Step 7: determinism + distinctness checks ===", flush=True)
    with torch.no_grad():
        lt1 = torch.zeros(1, device=device, dtype=x_t.dtype)
        h1 = model.forward_encoder(x_t[0:1].to(device), lt1, variables).detach().cpu().numpy()
        h2 = model.forward_encoder(x_t[0:1].to(device), lt1, variables).detach().cpu().numpy()
    det_diff = float(np.max(np.abs(h1 - h2)))
    print(f"  determinism (repeat inference, WY1985): max abs diff = {det_diff:.3e}", flush=True)
    assert det_diff == 0.0

    diffs = []
    for i in range(1, n_samples):
        d = float(np.max(np.abs(H[0] - H[i])))
        diffs.append(d)
    min_pairwise_diff = min(diffs)
    print(f"  distinctness: min max-abs-diff between WY1985 and any other year = {min_pairwise_diff:.6f}", flush=True)
    assert min_pairwise_diff > 0.0, "some year produced an identical latent to WY1985 -- investigate"

    param_after_checks = param_sum(model)
    assert param_before == param_after_checks

    print("=== Step 8: load existing April-1 SWE anomaly target and align by water_year ===", flush=True)
    swe_ds = xr.open_dataset(SWE_ANOM_NC)
    swe_water_years = swe_ds["water_year"].values.astype(int)
    assert list(swe_water_years) == WATER_YEARS, (list(swe_water_years), WATER_YEARS)
    swe_anom_mm = swe_ds["sierra_swe_apr1_anom_mm"].values.astype(np.float32)
    swe_standardized = swe_ds["sierra_swe_apr1_standardized"].values.astype(np.float32)
    swe_mean_mm = swe_ds["sierra_swe_apr1_mean_mm"].values.astype(np.float32)
    apr1_target_dates = swe_ds["target_date"].values
    swe_ds.close()

    march25_dates = np.array([date(wy, 3, 25).isoformat() for wy in WATER_YEARS])

    print("=== Step 9: save paired dataset ===", flush=True)
    np.save(os.path.join(LATENT_DIR, "H_march25_obs_latents.npy"), H)
    np.savez(
        os.path.join(LATENT_DIR, "paired_dataset.npz"),
        water_year=np.asarray(WATER_YEARS, dtype=np.int32),
        march25_date=march25_dates,
        apr1_target_date=apr1_target_dates,
        swe_apr1_mean_mm=swe_mean_mm,
        swe_apr1_anom_mm=swe_anom_mm,
        swe_apr1_standardized=swe_standardized,
    )

    summary = {
        "scope": {
            "source": "ERA5 (raw hourly analysis, NCAR RDA ds633.0 mirror)",
            "date_per_water_year": "March 25",
            "water_years": [WATER_YEARS[0], WATER_YEARS[-1]],
            "n_samples": n_samples,
            "no_taiesm1": True, "no_daily_ucla_swe": True, "no_other_dates": True,
        },
        "checkpoint": "https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt",
        "variables_our_name_to_climax_name": VAR_TO_CLIMAX,
        "unit_conversion_note": (
            "None applied to any of the 10 variables. In particular, ERA5's own Z "
            "(geopotential) is natively m**2 s**-2 (confirmed from file units/long_name "
            "attrs), i.e. already ClimaX's expected 'geopotential' unit -- unlike CMIP6's "
            "zg (geopotential HEIGHT, m), ERA5 Z needs no *9.807 conversion."
        ),
        "excluded_variables": {"rlut": "no ClimaX vocabulary match; excluded, not substituted",
                                "psl": "no ClimaX vocabulary match; excluded, not substituted"},
        "daily_mean_construction": "genuine 24-hour mean from raw ERA5 hourly analysis files "
                                    "(e5.oper.an.pl per-day files for pressure-level vars, "
                                    "e5.oper.an.sfc per-month files sliced to the target day "
                                    "for tas) -- not interpolated or derived from any monthly product",
        "native_grid": "ERA5 0.25deg global (721 x 1440, lat -90..90, lon 0..359.75)",
        "target_grid": "1.40625deg, 128x256 (ClimaX pretraining grid)",
        "regrid_method": "CDO remapbil (bilinear), same grid definition/method as validated prior ClimaX runs",
        "cropping": "none -- full global ERA5 state supplied to ClimaX, per approved scope",
        "normalization": {
            "methodology": "per-variable global mean/std over space+time (ClimaX's own published methodology)",
            "fit_window": "the same 37 March-25 ERA5 global snapshots used for extraction (WY1985-2021) "
                           "-- no ssp370/obs-label/downstream-target leakage",
            "per_variable_stats": stats,
        },
        "lead_time_assumption": "lead_times=0 for all samples (encoder-only nowcast, matches validated prior runs)",
        "extraction_point": "model.forward_encoder(x, lead_times, variables) -- final transformer block output, before prediction head",
        "input_tensor_shape": list(x.shape),
        "H_shape": list(H.shape),
        "H_shape_meaning": "[water_year sample, token (32x64 flattened patch grid), embed_dim]",
        "n_total_params": n_params,
        "n_trainable_params": n_trainable,
        "param_sum_before": param_before,
        "param_sum_after_forward": param_after,
        "param_sum_after_checks": param_after_checks,
        "determinism_max_abs_diff": det_diff,
        "min_pairwise_distinctness_max_abs_diff": min_pairwise_diff,
        "H_mean": float(H.mean()), "H_std": float(H.std()),
        "H_min": float(H.min()), "H_max": float(H.max()),
        "H_frac_nan": float(np.isnan(H).mean()), "H_frac_inf": float(np.isinf(H).mean()),
        "swe_anomaly_target_source": SWE_ANOM_NC,
        "swe_anomaly_target_note": "existing April-1 Sierra SWE anomaly artifact, unmodified; "
                                    "paired by water_year (March 25 of WY_y with the Apr-1 target of the same WY_y)",
        "output_paths": {
            "H_latents": os.path.join(LATENT_DIR, "H_march25_obs_latents.npy"),
            "paired_dataset": os.path.join(LATENT_DIR, "paired_dataset.npz"),
        },
        "device": str(device),
    }
    with open(os.path.join(LATENT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(HOME_ARTIFACT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
