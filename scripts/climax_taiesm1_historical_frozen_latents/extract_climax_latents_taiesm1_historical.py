#!/usr/bin/env python
"""Frozen-ClimaX TaiESM1-historical latent extraction.

Scope (explicitly approved): TaiESM1, historical only, 1980-2014 (35 years,
12775 daily samples), 10 ClimaX-mappable variables only (rlut and psl
excluded, not substituted). Reuses the validated MPI smoke-test's model
loading, checkpoint-renaming, and forward_encoder extraction pattern
verbatim. Performs NO training, NO fine-tuning, NO forecasting, and NO
SWE/CPM/AQM correlation. Uses lead_times=0 throughout (encoder-only, nowcast
convention) exactly as the smoke test did.

Source data: already-processed TaiESM1 daily 12-variable 1.5deg (120x240)
physical arrays at
/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1/.
Not re-derived from raw CMIP6.
"""
from __future__ import annotations

import argparse
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

SOURCE_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1"
PSCRATCH_ROOT = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_taiesm1_historical_frozen_latents"
NETCDF_1P5_DIR = os.path.join(PSCRATCH_ROOT, "netcdf_1p5deg")
REGRID_DIR = os.path.join(PSCRATCH_ROOT, "regridded_1p40625deg")
LATENT_DIR = os.path.join(PSCRATCH_ROOT, "latents")
GRID_FILE = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_frozen_mpi_smoke_test/climax_1p40625deg_grid/target_grid.txt"
CKPT_PATH = "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/climax_frozen_mpi_smoke_test/checkpoints/1.40625deg.ckpt"
HOME_ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts/climax_taiesm1_historical_frozen_latents")

YEARS = list(range(1980, 2015))  # historical only, 35 years, per approved scope
ALL_CHANNELS = ["rlut", "zg_500", "zg_50", "ta_850", "ta_50", "ua_850", "ua_50",
                 "va_850", "va_50", "hus_850", "psl", "tas"]
# Approved 10-variable subset -> ClimaX vocabulary name. rlut, psl excluded, not substituted.
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
VARKEYS = list(VAR_TO_CLIMAX.keys())  # fixed order used throughout
ZG_TO_Z_GRAVITY = 9.807  # exact constant used by the validated MPI smoke test / ClimaX's own regrid.py

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
BATCH_SIZE = 16


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_source_year(year: int) -> tuple[np.ndarray, np.ndarray]:
    experiment = "historical"
    pred_path = os.path.join(SOURCE_ROOT, "years", experiment, f"predictors_physical_{year}.npy")
    mask_path = os.path.join(SOURCE_ROOT, "years", experiment, f"validity_mask_{year}.npy")
    pred = np.load(pred_path)  # [365, 12, 120, 240]
    mask = np.load(mask_path)  # [365, 12] per-day/per-channel validity flag
    assert pred.shape == (365, 12, 120, 240), (year, pred.shape)
    assert mask.shape == (365, 12), (year, mask.shape)
    return pred, mask


def build_native_netcdf() -> dict[str, str]:
    """Write one NetCDF per approved variable, all 35 historical years
    concatenated (12775 timesteps), on the existing 1.5deg 120x240 grid,
    with zg unit conversion applied. Returns varkey -> netcdf path."""
    meta = np.load(os.path.join(SOURCE_ROOT, "metadata.npz"), allow_pickle=True)
    lat = meta["lat"].astype(np.float64)
    lon = meta["lon"].astype(np.float64)
    channel_names = list(meta["channel_names"])
    assert channel_names == ALL_CHANNELS, (channel_names, ALL_CHANNELS)
    chan_idx = {c: i for i, c in enumerate(ALL_CHANNELS)}

    os.makedirs(NETCDF_1P5_DIR, exist_ok=True)
    out_paths: dict[str, str] = {}

    # Build a single 365_day (noleap) time axis for 1980-01-01..2014-12-31.
    times = xr.cftime_range(start="1980-01-01", periods=len(YEARS) * 365, freq="D", calendar="noleap")

    for vk in VARKEYS:
        out_path = os.path.join(NETCDF_1P5_DIR, f"{vk}_1p5deg_1980-2014.nc")
        out_paths[vk] = out_path
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue
        idx = chan_idx[vk]
        year_arrays = []
        total_invalid_days = 0
        for year in YEARS:
            pred, mask = load_source_year(year)
            arr = pred[:, idx, :, :].astype(np.float64)  # [365,120,240]
            invalid_days = int((~mask[:, idx].astype(bool)).sum())  # mask[:, idx]: [365] per-day validity flag for this channel
            if invalid_days:
                print(f"  WARNING: {vk} {year} has {invalid_days} day(s) flagged invalid by validity_mask", flush=True)
            total_invalid_days += invalid_days
            year_arrays.append(arr)
        full = np.concatenate(year_arrays, axis=0)  # [12775,120,240]
        assert full.shape[0] == len(YEARS) * 365
        assert total_invalid_days == 0, f"{vk}: {total_invalid_days} day(s) flagged invalid across 1980-2014 -- stop, do not silently proceed"
        if vk in ("zg_500", "zg_50"):
            full = full * ZG_TO_Z_GRAVITY

        da = xr.DataArray(
            full.astype(np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": lat, "lon": lon},
            name=vk,
        )
        da.attrs["invalid_mask_flagged_cells"] = int(total_invalid_days)
        ds = da.to_dataset()
        # CF attributes so CDO recognizes this as a lonlat grid (xarray's default
        # to_netcdf omits them, which makes remapbil fail with "Unsupported
        # generic coordinates").
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
        ds.to_netcdf(out_path, engine="netcdf4")
        print(f"  wrote {out_path}  invalid_mask_flagged_cells={total_invalid_days}")
    return out_paths


def regrid_to_climax_grid(native_paths: dict[str, str]) -> dict[str, str]:
    os.makedirs(REGRID_DIR, exist_ok=True)
    out_paths: dict[str, str] = {}
    for vk, infile in native_paths.items():
        outfile = os.path.join(REGRID_DIR, f"{vk}_1p40625deg_1980-2014.nc")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true",
                         help="Run only the small deterministic validation subset, then stop.")
    args = parser.parse_args()

    os.makedirs(LATENT_DIR, exist_ok=True)
    os.makedirs(HOME_ARTIFACT_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    print("=== Step 1: verify source dataset (365 days/year, all 35 years present) ===", flush=True)
    meta = np.load(os.path.join(SOURCE_ROOT, "metadata.npz"), allow_pickle=True)
    for year in YEARS:
        pred_path = os.path.join(SOURCE_ROOT, "years", "historical", f"predictors_physical_{year}.npy")
        mask_path = os.path.join(SOURCE_ROOT, "years", "historical", f"validity_mask_{year}.npy")
        assert os.path.exists(pred_path), f"missing {pred_path}"
        assert os.path.exists(mask_path), f"missing {mask_path}"
    print(f"  confirmed {len(YEARS)} years present, each expected [365,12,120,240]", flush=True)

    print("=== Step 2: build native-grid (1.5deg) NetCDFs for the 10 approved variables ===", flush=True)
    native_paths = build_native_netcdf()

    print("=== Step 3: CDO regrid 120x240 (1.5deg) -> 128x256 (1.40625deg, ClimaX grid) ===", flush=True)
    regridded_paths = regrid_to_climax_grid(native_paths)

    print("=== Step 4: load regridded fields, verify grid, compute historical-only normalization stats ===", flush=True)
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
        assert arr.shape[0] == len(YEARS) * 365, (vk, arr.shape)
        n_nan_total += int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
        n_total += arr.size
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        stats[vk] = {
            "mean": mean, "std": std,
            "stats_source": "TaiESM1 historical 1980-2014 only (no ssp370, no obs/labels)",
            "n_timesteps": int(arr.shape[0]),
        }
        fields[vk] = arr.astype(np.float32)
        ds.close()
        print(f"  {vk}: mean={mean:.6f} std={std:.6f} n_timesteps={arr.shape[0]}", flush=True)

    print(f"  total input NaN/Inf across all 10 variables, all 35 years: {n_nan_total} / {n_total}", flush=True)
    assert n_nan_total == 0, f"found {n_nan_total} NaN/Inf values in regridded TaiESM1 historical input"

    n_days_total = fields[VARKEYS[0]]["shape"] if False else fields[VARKEYS[0]].shape[0]
    for vk in VARKEYS:
        assert fields[vk].shape[0] == n_days_total, "timestep count mismatch across variables"
    assert n_days_total == len(YEARS) * 365 == 12775, n_days_total

    print("=== Step 5: build normalized full input tensor [N,10,128,256] ===", flush=True)
    x_full = np.zeros((n_days_total, len(VARKEYS), 128, 256), dtype=np.float32)
    for i, vk in enumerate(VARKEYS):
        mean, std = stats[vk]["mean"], stats[vk]["std"]
        x_full[:, i] = (fields[vk] - mean) / std
    variables = [VAR_TO_CLIMAX[vk] for vk in VARKEYS]
    print("  full input tensor shape:", x_full.shape, " ClimaX variable order:", variables, flush=True)

    print("=== Step 6: build frozen ClimaX model, load official 1.40625deg checkpoint ===", flush=True)
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {n_params:,}  trainable params: {n_trainable:,} (must be 0)", flush=True)
    assert n_trainable == 0
    param_before = param_sum(model)

    print("=== Step 7: SMALL VALIDATION SUBSET (first 8 days of 1980) ===", flush=True)
    n_val = 8
    xv = torch.from_numpy(x_full[:n_val])
    with torch.no_grad():
        lead_times = torch.zeros(xv.shape[0], device=device, dtype=xv.dtype)
        hv = model.forward_encoder(xv.to(device), lead_times, variables)
    hv_np = hv.detach().cpu().numpy().astype(np.float32)
    print(f"  validation H shape: {hv_np.shape} (expect ({n_val}, 2048, 1024))", flush=True)
    assert hv_np.shape == (n_val, 2048, 1024), hv_np.shape
    assert not np.isnan(hv_np).any(), "NaN in validation latent"
    assert not np.isinf(hv_np).any(), "Inf in validation latent"
    with torch.no_grad():
        lead_times1 = torch.zeros(1, device=device, dtype=xv.dtype)
        h1 = model.forward_encoder(xv[0:1].to(device), lead_times1, variables).detach().cpu().numpy()
        h2 = model.forward_encoder(xv[0:1].to(device), lead_times1, variables).detach().cpu().numpy()
    det_diff = float(np.max(np.abs(h1 - h2)))
    print(f"  determinism check (repeat inference, same day): max abs diff = {det_diff:.3e}", flush=True)
    assert det_diff == 0.0, "encoder not deterministic in eval mode"
    diff_days = float(np.max(np.abs(hv_np[0] - hv_np[1])))
    print(f"  distinctness check (day0 vs day1 of validation subset): max abs diff = {diff_days:.6f}", flush=True)
    assert diff_days > 0.0, "different days produced identical latents -- investigate before continuing"
    param_after_val = param_sum(model)
    assert param_before == param_after_val, "model parameters changed during validation forward pass!"
    print("  VALIDATION PASSED: shape, dtype, NaN/Inf, determinism, distinctness, frozen-checksum all OK", flush=True)

    validation_summary = {
        "n_validation_days": n_val,
        "validation_H_shape": list(hv_np.shape),
        "determinism_max_abs_diff": det_diff,
        "distinctness_max_abs_diff_day0_vs_day1": diff_days,
        "param_sum_before": param_before,
        "param_sum_after_validation": param_after_val,
        "variables_in_order": variables,
    }
    with open(os.path.join(LATENT_DIR, "validation_summary.json"), "w") as f:
        json.dump(validation_summary, f, indent=2)

    if args.validate_only:
        print("=== --validate-only set: stopping after validation, per instructions ===", flush=True)
        return

    print("=== Step 8: FULL extraction, all 35 historical years, saved year-by-year ===", flush=True)
    day_offset = 0
    per_year_report = []
    x_t = torch.from_numpy(x_full)
    for year in YEARS:
        t0 = time.time()
        year_slice = slice(day_offset, day_offset + 365)
        H_chunks = []
        with torch.no_grad():
            for s in range(0, 365, BATCH_SIZE):
                xb = x_t[year_slice][s:s + BATCH_SIZE].to(device)
                lead_times = torch.zeros(xb.shape[0], device=device, dtype=xb.dtype)
                hb = model.forward_encoder(xb, lead_times, variables)
                H_chunks.append(hb.detach().cpu().numpy().astype(np.float32))
        H_year = np.concatenate(H_chunks, axis=0)
        assert H_year.shape == (365, 2048, 1024), (year, H_year.shape)

        # Exact calendar dates from the same noleap cftime axis used for regridding.
        year_times = xr.cftime_range(start=f"{year}-01-01", periods=365, freq="D", calendar="noleap")
        timestamps = np.array([t.isoformat() for t in year_times])

        out_path = os.path.join(LATENT_DIR, f"TaiESM1_historical_{year}_H_latent.npy")
        np.save(out_path, H_year)
        np.save(os.path.join(LATENT_DIR, f"TaiESM1_historical_{year}_timestamps.npy"), timestamps)

        elapsed = time.time() - t0
        per_year_report.append({
            "year": year,
            "n_days": 365,
            "H_shape": list(H_year.shape),
            "H_mean": float(H_year.mean()),
            "H_std": float(H_year.std()),
            "H_min": float(H_year.min()),
            "H_max": float(H_year.max()),
            "n_nan": int(np.isnan(H_year).sum()),
            "n_inf": int(np.isinf(H_year).sum()),
            "elapsed_seconds": elapsed,
            "output_path": out_path,
        })
        print(f"  year {year}: H_shape={H_year.shape} mean={H_year.mean():.6f} std={H_year.std():.6f} "
              f"nan={int(np.isnan(H_year).sum())} inf={int(np.isinf(H_year).sum())} elapsed={elapsed:.1f}s",
              flush=True)
        day_offset += 365

    param_after_full = param_sum(model)
    assert param_before == param_after_full, "model parameters changed during full extraction!"

    total_days = sum(r["n_days"] for r in per_year_report)
    total_bytes = sum(os.path.getsize(r["output_path"]) for r in per_year_report)

    summary = {
        "scope": {
            "model": "TaiESM1", "institution_id": "AS-RCEC", "member_id": "r1i1p1f1",
            "experiment": "historical", "years": [YEARS[0], YEARS[-1]],
            "no_ssp370": True, "no_mpi": True,
        },
        "checkpoint": "https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt",
        "variables_our_name_to_climax_name": VAR_TO_CLIMAX,
        "excluded_variables": {"rlut": "no ClimaX vocabulary match; excluded, not substituted",
                                "psl": "no ClimaX vocabulary match; excluded, not substituted"},
        "zg_to_z_conversion": f"z = zg * {ZG_TO_Z_GRAVITY}",
        "source_dataset": {
            "path": SOURCE_ROOT,
            "native_grid": "120x240, 1.5deg (already-processed, not raw CMIP6)",
        },
        "target_grid": "1.40625deg, 128x256 (ClimaX pretraining grid)",
        "regrid_method": "CDO remapbil (bilinear), same grid definition and method as validated MPI smoke test",
        "normalization": {
            "methodology": "per-variable global mean/std over space+time (ClimaX's own published methodology)",
            "fit_window": "TaiESM1 historical 1980-2014 only (35 years, all 12775 days) -- no ssp370, no obs, no labels",
            "per_variable_stats": stats,
        },
        "lead_time_assumption": "lead_times=0 for all samples (encoder-only nowcast, matches validated smoke test)",
        "extraction_point": "model.forward_encoder(x, lead_times, variables) -- final transformer block output, before prediction head",
        "input_tensor_shape_per_day": [len(VARKEYS), 128, 256],
        "H_shape_per_day": [2048, 1024],
        "H_shape_meaning": "[token (flattened 32x64 spatial patch grid), embed_dim]",
        "n_processed_days": total_days,
        "expected_days": len(YEARS) * 365,
        "n_total_params": n_params,
        "n_trainable_params": n_trainable,
        "param_sum_before": param_before,
        "param_sum_after_validation": param_after_val,
        "param_sum_after_full_extraction": param_after_full,
        "total_output_bytes": total_bytes,
        "total_output_gib": total_bytes / (1024 ** 3),
        "per_year_report": per_year_report,
        "validation_summary": validation_summary,
        "device": str(device),
    }
    with open(os.path.join(LATENT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(HOME_ARTIFACT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"=== DONE: {total_days} days processed, {total_bytes / (1024**3):.2f} GiB written ===", flush=True)


if __name__ == "__main__":
    main()
