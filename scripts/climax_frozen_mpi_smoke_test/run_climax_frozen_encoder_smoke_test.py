#!/usr/bin/env python
"""Frozen-ClimaX MPI smoke test.

Reconstructs a valid input for the official pretrained ClimaX 1.40625deg
checkpoint from ONE year of our existing MPI-ESM1-2-HR data using the six
overlapping variables (tas, zg500, ta850, ua850, va850, hus850), runs the
frozen encoder (no training, no gradient updates), and saves the full
spatial latent representation H_t plus provenance metadata.

This script performs NO training, NO fine-tuning, and NO probing against
SWE/CPM/AQM. It only validates: our MPI data -> frozen ClimaX encoder -> H_t.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import xarray as xr

REPO_ROOT = "/global/u1/h/hyvchen/Snow-Predication-at-Sierra"
CLIMAX_SRC = os.path.join(REPO_ROOT, "thirdparties/ClimaX/src")
sys.path.insert(0, CLIMAX_SRC)

from climax.arch import ClimaX  # noqa: E402

PSCRATCH_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_frozen_mpi_smoke_test"
RAW_DIR = os.path.join(PSCRATCH_ROOT, "raw_mpi_native")
GRID_FILE = os.path.join(PSCRATCH_ROOT, "climax_1p40625deg_grid/target_grid.txt")
REGRID_DIR = os.path.join(PSCRATCH_ROOT, "regridded")
CKPT_PATH = os.path.join(PSCRATCH_ROOT, "checkpoints/1.40625deg.ckpt")
OUT_DIR = os.path.join(PSCRATCH_ROOT, "artifact")
HOME_ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts/climax_frozen_mpi_smoke_test")

STATS_YEARS = [1950, 1951, 1952, 1953, 1954]  # full locally-cached 5-yr chunk (day table)
TEST_YEAR = 1952  # the single smoke-test year (must NOT be used alone for normalization)

MPI_SOURCE = {
    # varkey: (native file, cmip var name, table_id, plev_pa or None)
    "tas": (
        "tas_day_MPI-ESM1-2-HR_historical_r3i1p1f1_gn_19500101-19541231.nc",
        "tas", "day", None,
    ),
    "zg500": (
        "zg_day_MPI-ESM1-2-HR_historical_r3i1p1f1_gn_19500101-19541231.nc",
        "zg", "day", 50000.0,
    ),
    "ta850": (
        "ta_day_MPI-ESM1-2-HR_historical_r3i1p1f1_gn_19500101-19541231.nc",
        "ta", "day", 85000.0,
    ),
    "ua850": (
        "ua_day_MPI-ESM1-2-HR_historical_r3i1p1f1_gn_19500101-19541231.nc",
        "ua", "day", 85000.0,
    ),
    "va850": (
        "va_day_MPI-ESM1-2-HR_historical_r3i1p1f1_gn_19500101-19541231.nc",
        "va", "day", 85000.0,
    ),
    "hus850": (
        "hus_day_MPI-ESM1-2-HR_historical_r3i1p1f1_gn_19500101-19541231.nc",
        "hus", "day", 85000.0,
    ),
}

# our_var -> official ClimaX default_vars name (see configs/pretrain_climax.yaml)
CLIMAX_VAR_NAME = {
    "tas": "2m_temperature",
    "zg500": "geopotential_500",
    "ta850": "temperature_850",
    "ua850": "u_component_of_wind_850",
    "va850": "v_component_of_wind_850",
    "hus850": "specific_humidity_850",
}

# Exact ClimaX pretraining default_vars list (configs/pretrain_climax.yaml, main branch,
# commit fetched 2026-09-04). Order matters: it defines the checkpoint's per-variable
# token-embedding / channel-embedding indices.
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

ZG_TO_Z_GRAVITY = 9.807  # exact constant used by ClimaX's own src/data_preprocessing/regrid.py

IMG_SIZE = [128, 256]
PATCH_SIZE = 4
EMBED_DIM = 1024
DEPTH = 8
DECODER_DEPTH = 2
NUM_HEADS = 16
MLP_RATIO = 4.0


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def regrid_variable(varkey):
    """Select the six required years at the requested pressure level (if any) from
    the native MPI grid, then bilinear-regrid to ClimaX's 1.40625deg/128x256 grid
    using the exact target-grid definition from ClimaX's own regrid.py
    (lat=-90+ddeg/2:ddeg:90, lon=0:ddeg:360, periodic bilinear)."""
    fname, cmip_var, table, plev_pa = MPI_SOURCE[varkey]
    infile = os.path.join(RAW_DIR, fname)
    outfile = os.path.join(REGRID_DIR, f"{varkey}_1p40625deg_1950-1954.nc")
    if os.path.exists(outfile):
        print(f"[skip] {outfile} already exists")
        return outfile

    years_arg = ",".join(str(y) for y in STATS_YEARS)
    if plev_pa is not None:
        cmd = [
            "cdo", "-O",
            f"remapbil,{GRID_FILE}",
            f"-sellevel,{plev_pa}",
            f"-selyear,{years_arg}",
            f"-selname,{cmip_var}",
            infile, outfile,
        ]
    else:
        cmd = [
            "cdo", "-O",
            f"remapbil,{GRID_FILE}",
            f"-selyear,{years_arg}",
            f"-selname,{cmip_var}",
            infile, outfile,
        ]
    run(cmd)
    return outfile


def load_regridded(varkey):
    fname, cmip_var, table, plev_pa = MPI_SOURCE[varkey]
    outfile = os.path.join(REGRID_DIR, f"{varkey}_1p40625deg_1950-1954.nc")
    ds = xr.open_dataset(outfile)
    da = ds[cmip_var]
    if "plev" in da.dims:
        da = da.squeeze("plev", drop=True)
    if varkey == "zg500":
        da = da * ZG_TO_Z_GRAVITY
        da.attrs["units"] = "m2 s-2"
        da.attrs["converted_from"] = "zg (geopotential height, m) * 9.807 -> z (geopotential, m2 s-2)"
    return da, ds.lat.values, ds.lon.values


def verify_grid(lat, lon):
    expected_lat = -90 + 1.40625 / 2 + np.arange(128) * 1.40625
    expected_lon = np.arange(256) * 1.40625
    assert lat.shape[0] == 128 and lon.shape[0] == 256, (lat.shape, lon.shape)
    assert np.allclose(lat, expected_lat, atol=1e-6), "lat grid mismatch vs ClimaX target grid"
    assert np.allclose(lon, expected_lon, atol=1e-6), "lon grid mismatch vs ClimaX target grid"
    assert lat[0] < lat[-1], "lat must be ascending south-to-north (ClimaX convention)"
    return True


def select_test_year(da, year):
    sub = da.sel(time=da["time.year"] == year)
    n = sub.sizes["time"]
    if n == 366:
        sub = sub.isel(time=slice(0, 365))  # ClimaX convention: drop the trailing day of a leap year
    return sub


def compute_stats(da):
    arr = da.values.astype(np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return mean, std


def build_model():
    model = ClimaX(
        default_vars=CLIMAX_DEFAULT_VARS,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        decoder_depth=DECODER_DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        drop_path=0.0,
        drop_rate=0.0,
        parallel_patch_embed=False,
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    raw_sd = ckpt["state_dict"]

    # Official checkpoint compatibility adapter (documented, not a change to arch.py):
    # the released 1.40625deg.ckpt was saved with an earlier naming convention
    # ("channel_embed"/"channel_query"/"channel_agg") than the current main-branch
    # arch.py ("var_embed"/"var_query"/"var_agg"). This is confirmed by ClimaX's own
    # src/climax/utils/pos_embed.py::interpolate_channel_embed, which explicitly
    # looks for the key "net.channel_embed" in a loaded checkpoint -- i.e. this
    # renaming is expected/known in the official repo itself, not a bug in our copy.
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


def main():
    os.makedirs(REGRID_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(HOME_ARTIFACT_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    varkeys = list(MPI_SOURCE.keys())

    print("=== Step 1: CDO regrid (native -> 1.40625deg, 1950-1954) ===", flush=True)
    for vk in varkeys:
        regrid_variable(vk)

    print("=== Step 2: load, verify grid, compute stats, select test year ===", flush=True)
    stats = {}
    test_year_fields = {}
    lat_ref = lon_ref = None
    n_nan_input = 0
    n_total_input = 0
    for vk in varkeys:
        da, lat, lon = load_regridded(vk)
        verify_grid(lat, lon)
        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        else:
            assert np.allclose(lat, lat_ref) and np.allclose(lon, lon_ref)

        mean, std = compute_stats(da)
        stats[vk] = {"mean": mean, "std": std, "n_years": STATS_YEARS, "n_timesteps": int(da.sizes["time"])}

        test_da = select_test_year(da, TEST_YEAR)
        arr = test_da.values.astype(np.float32)
        n_nan_input += int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
        n_total_input += arr.size
        test_year_fields[vk] = {"raw": arr, "time": test_da["time"].values}
        print(f"  {vk}: mean={mean:.6f} std={std:.6f} n_days_test_year={arr.shape[0]}", flush=True)

    assert n_nan_input == 0, f"found {n_nan_input} NaN/Inf values in reconstructed input"

    n_days = test_year_fields[varkeys[0]]["time"].shape[0]
    for vk in varkeys:
        assert test_year_fields[vk]["raw"].shape[0] == n_days, "timestep count mismatch across variables"

    print("=== Step 3: normalize with 5-year (non-single-year) statistics ===", flush=True)
    x = np.zeros((n_days, len(varkeys), 128, 256), dtype=np.float32)
    for i, vk in enumerate(varkeys):
        raw = test_year_fields[vk]["raw"]
        mean, std = stats[vk]["mean"], stats[vk]["std"]
        x[:, i] = (raw - mean) / std

    variables = [CLIMAX_VAR_NAME[vk] for vk in varkeys]
    print("  input tensor shape:", x.shape, " variables (ClimaX names):", variables, flush=True)

    print("=== Step 4: build frozen ClimaX model, load official 1.40625deg checkpoint ===", flush=True)
    model = build_model()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {n_params:,}  trainable params: {n_trainable:,} (must be 0)", flush=True)
    assert n_trainable == 0

    param_hash_before = torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).sum().item()

    print("=== Step 5: run frozen encoder (forward_encoder), full test year, no_grad ===", flush=True)
    batch_size = 16
    H_chunks = []
    x_t = torch.from_numpy(x)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n_days, batch_size):
            xb = x_t[s:s + batch_size].to(device)
            lead_times = torch.zeros(xb.shape[0], device=device, dtype=xb.dtype)
            hb = model.forward_encoder(xb, lead_times, variables)
            H_chunks.append(hb.detach().cpu().numpy().astype(np.float32))
    H = np.concatenate(H_chunks, axis=0)
    elapsed = time.time() - t0
    print(f"  H shape: {H.shape}  dtype: {H.dtype}  elapsed: {elapsed:.1f}s", flush=True)

    param_hash_after = torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).sum().item()
    assert param_hash_before == param_hash_after, "model parameters changed during forward pass!"

    print("=== Step 6: determinism check (single sample, run twice) ===", flush=True)
    with torch.no_grad():
        xb = x_t[0:1].to(device)
        lead_times = torch.zeros(1, device=device, dtype=xb.dtype)
        h1 = model.forward_encoder(xb, lead_times, variables).detach().cpu().numpy()
        h2 = model.forward_encoder(xb, lead_times, variables).detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(h1 - h2)))
    print(f"  max abs diff between two runs of the same sample: {max_abs_diff:.3e}", flush=True)
    assert max_abs_diff == 0.0, "encoder is not deterministic in eval mode!"

    print("=== Step 7: save artifact ===", flush=True)
    n_frac_nan = float(np.isnan(H).mean())
    n_frac_inf = float(np.isinf(H).mean())
    summary = {
        "checkpoint": "https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt",
        "checkpoint_naming_note": "state_dict used official channel_embed/channel_query/channel_agg keys, "
                                   "renamed to var_embed/var_query/var_agg to match current arch.py main branch "
                                   "(confirmed expected by ClimaX's own utils/pos_embed.py::interpolate_channel_embed)",
        "source_model": "MPI-ESM1-2-HR",
        "source_experiment": "historical",
        "source_member": "r3i1p1f1",
        "source_member_note": "ClimaX's own MPI-ESM pretraining recipe (snakemake_configs/MPI-ESM) used "
                               "r1i1p1f1, 6hrPlevPt (6-hourly instantaneous). Our project's existing MPI-ESM1-2-HR "
                               "holding uses r3i1p1f1. Different ensemble member: same model/forcing, different "
                               "internal-variability realization. Not the exact states ClimaX trained on.",
        "table_id": "day (daily-mean, fixed plev8 pressure levels incl. 500/850 hPa; NOT the 6-hourly "
                    "instantaneous fields ClimaX's MPI-ESM pretraining recipe actually used)",
        "temporal_resolution_mismatch": "MPI local holding is DAILY MEAN. ClimaX's own MPI-ESM CMIP6 pretraining "
                                         "used 6-hourly INSTANTANEOUS (6hrPlevPt) fields. This is a genuine, "
                                         "documented mismatch -- daily-mean, not monthly-mean, so we proceeded "
                                         "per task instructions, but this is not a bit-exact reproduction of the "
                                         "pretraining input distribution.",
        "test_year": TEST_YEAR,
        "stats_years": STATS_YEARS,
        "normalization_note": "Official per-dataset normalize_mean.npz/normalize_std.npz used for ClimaX's own "
                               "CMIP6 pretraining are NOT publicly released (checked: HuggingFace tungnd/climax "
                               "repo contains only *.ckpt files; no companion stats found in GitHub issues/repo). "
                               "Stats here were computed using ClimaX's own published methodology (per-variable "
                               "global mean/std over space+time) applied to the full locally available 1950-1954 "
                               "5-year window (NOT the single smoke-test year 1952) at the same 1.40625deg grid. "
                               "This is a documented, non-single-year approximation, not the original checkpoint's "
                               "exact training statistics.",
        "variables_our_name_to_climax_name": CLIMAX_VAR_NAME,
        "zg_to_z_conversion": f"z = zg * {ZG_TO_Z_GRAVITY} (exact constant from ClimaX src/data_preprocessing/regrid.py)",
        "per_variable_stats": stats,
        "native_grid": "192x384 (MPI-ESM1-2-HR atmosphere grid gn)",
        "target_grid": "1.40625deg, 128x256 (ClimaX pretraining grid)",
        "regrid_method": "CDO remapbil (bilinear), target grid lat=-90+ddeg/2:ddeg:90 lon=0:ddeg:360, matching "
                          "ClimaX src/data_preprocessing/regrid.py's xESMF bilinear grid definition. CDO's SCRIP "
                          "bilinear and xESMF's ESMF-based bilinear are the same interpolation family but not "
                          "bit-identical implementations -- documented deviation from the official preprocessing "
                          "pipeline.",
        "lead_time_assumption": "lead_times=0 for all samples (ClimaX's forward_encoder requires this argument "
                                 "architecturally; since we extract the input state's own representation rather "
                                 "than forecasting a future state, 0 is used as the neutral 'nowcast' choice).",
        "input_tensor_shape": list(x.shape),
        "H_shape": list(H.shape),
        "H_shape_meaning": "[time (days in test year), token (flattened 32x64 spatial patch grid, "
                            "row-major over (lat_patch, lon_patch)), embed_dim]",
        "H_dtype": str(H.dtype),
        "H_min": float(H.min()),
        "H_max": float(H.max()),
        "H_mean": float(H.mean()),
        "H_std": float(H.std()),
        "H_frac_nan": n_frac_nan,
        "H_frac_inf": n_frac_inf,
        "determinism_max_abs_diff": max_abs_diff,
        "n_trainable_params": n_trainable,
        "n_total_params": n_params,
        "param_sum_before": param_hash_before,
        "param_sum_after": param_hash_after,
        "patch_grid": {"n_patch_lat": 32, "n_patch_lon": 64, "patch_size_deg": 1.40625 * 4},
        "device": str(device),
        "forward_pass_seconds": elapsed,
    }

    np.save(os.path.join(OUT_DIR, "H_latent.npy"), H)
    np.save(os.path.join(OUT_DIR, "input_tensor_normalized.npy"), x)
    np.save(os.path.join(OUT_DIR, "timestamps.npy"), test_year_fields[varkeys[0]]["time"])
    with open(os.path.join(OUT_DIR, "lat.npy"), "wb") as f:
        np.save(f, lat_ref)
    with open(os.path.join(OUT_DIR, "lon.npy"), "wb") as f:
        np.save(f, lon_ref)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # small, non-duplicated summary copy in the repo's artifact area (per storage contract)
    with open(os.path.join(HOME_ARTIFACT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
