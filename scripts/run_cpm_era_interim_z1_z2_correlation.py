#!/usr/bin/env python3
"""Run the ERA-Interim CPM reproduction and Z1/Z2 correlation workflow.

This script is ready for use once an ERA-Interim input file is available in a
Python environment that provides numpy, pandas, xarray, scipy, and matplotlib.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTOR_TABLE = (
    PROJECT_ROOT
    / "artifacts"
    / "cobe2_sierra_swe_lod_setup"
    / "exact_Z1_Z2_plus_PacificPC_Nino34_loyo"
    / "z1_z2_pacificpc_nino34_predictor_table.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cpm_era_interim_reproduction"


def require_deps():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr
        from scipy import stats
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required Python dependency: "
            f"{exc}. Run this script in a scientific Python environment with "
            "numpy, pandas, xarray, scipy, and matplotlib."
        ) from exc
    return pd, xr, stats, plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era-input-path", type=Path, required=True)
    parser.add_argument("--predictor-table-path", type=Path, default=DEFAULT_PREDICTOR_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="1982-01-01")
    parser.add_argument("--end-date", default="2016-12-31")
    parser.add_argument("--lat-min", type=float, default=20.0)
    parser.add_argument("--lat-max", type=float, default=75.0)
    parser.add_argument("--lon-min", type=float, default=190.0)
    parser.add_argument("--lon-max", type=float, default=270.0)
    parser.add_argument("--n-eofs", type=int, default=6)
    parser.add_argument("--anomaly-method", default="calendar_day")
    parser.add_argument("--drop-leap-day", action="store_true", default=True)
    return parser.parse_args()


def standardize_lon_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon, 360.0)


def detect_coord_name(dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in dataset.coords:
            return name
        if name in dataset.dims:
            return name
    raise KeyError(f"Could not find coordinate among {tuple(candidates)}")


def detect_var_name(dataset) -> str:
    preferred = [
        "z500",
        "zg",
        "z",
        "gh",
        "geopotential_height",
        "geopotential",
    ]
    data_vars = list(dataset.data_vars)
    for name in preferred:
        if name in dataset.data_vars:
            return name
    if len(data_vars) == 1:
        return data_vars[0]
    raise KeyError(f"Could not identify Z500 variable from data vars: {data_vars}")


def assign_water_year(index) -> np.ndarray:
    years = index.year.to_numpy()
    months = index.month.to_numpy()
    return np.where(months >= 11, years + 1, years)


def fisher_ci(r: float, n: int, stats) -> Tuple[float, float]:
    if n <= 3:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(0.975)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def compute_summary(name: str, x: np.ndarray, y: np.ndarray, stats):
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    ci_low, ci_high = fisher_ci(float(pearson_r), len(x), stats)
    return {
        "predictor": name,
        "n": len(x),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_ci_low": ci_low,
        "pearson_ci_high": ci_high,
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def main() -> int:
    args = parse_args()
    pd, xr, stats, plt = require_deps()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.era_input_path)
    time_name = detect_coord_name(ds, ("time", "valid_time", "date"))
    lat_name = detect_coord_name(ds, ("lat", "latitude", "y"))
    lon_name = detect_coord_name(ds, ("lon", "longitude", "x"))
    var_name = detect_var_name(ds)

    field = ds[var_name]
    if var_name == "z" or "geopotential" in var_name.lower():
        field = field / 9.80665
    field = field.rename("z500_m")

    field = field.assign_coords({lon_name: standardize_lon_360(field[lon_name].values)})
    field = field.sortby(lon_name).sortby(lat_name)
    field = field.sel({lat_name: slice(args.lat_min, args.lat_max), lon_name: slice(args.lon_min, args.lon_max)})
    field = field.sel({time_name: slice(args.start_date, args.end_date)})

    # Aggregate sub-daily data to daily means when needed.
    if np.nanmedian(np.diff(field[time_name].values).astype("timedelta64[h]").astype(float)) < 24.0:
        field = field.resample({time_name: "1D"}).mean()

    dates = field[time_name].to_index()
    wet_mask = np.isin(dates.month, [11, 12, 1, 2, 3, 4])
    field = field.isel({time_name: wet_mask})
    dates = field[time_name].to_index()

    if args.drop_leap_day:
        keep = ~((dates.month == 2) & (dates.day == 29))
        field = field.isel({time_name: keep})
        dates = field[time_name].to_index()

    if args.anomaly_method != "calendar_day":
        raise ValueError("Only anomaly method 'calendar_day' is implemented.")

    month_day = pd.Index([f"{m:02d}-{d:02d}" for m, d in zip(dates.month, dates.day)], name="month_day")
    field = field.assign_coords(month_day=(time_name, month_day))
    clim = field.groupby("month_day").mean(time_name)
    anom = field.groupby("month_day") - clim

    lat_radians = np.deg2rad(anom[lat_name].values)
    weights = np.sqrt(np.cos(lat_radians))
    weight_da = xr.DataArray(weights, coords={lat_name: anom[lat_name]}, dims=(lat_name,))
    weighted = anom * weight_da

    stacked = weighted.stack(space=(lat_name, lon_name)).transpose(time_name, "space")
    valid_space = ~np.any(np.isnan(stacked.values), axis=0)
    X = stacked.values[:, valid_space]

    u, s, vt = np.linalg.svd(X, full_matrices=False)
    pcs = u[:, : args.n_eofs] * s[: args.n_eofs]
    evr = (s ** 2) / np.sum(s ** 2)

    flat_weights = np.repeat(weights[:, None], anom.sizes[lon_name], axis=1).reshape(-1)[valid_space]
    eof_unweighted_valid = vt[: args.n_eofs, :] / flat_weights[None, :]

    # Restore EOFs to the full spatial grid with NaNs at invalid cells.
    eof_full = np.full((args.n_eofs, stacked.sizes["space"]), np.nan, dtype=float)
    eof_full[:, valid_space] = eof_unweighted_valid
    eof_maps = (
        xr.DataArray(
            eof_full,
            dims=("mode", "space"),
            coords={"mode": np.arange(1, args.n_eofs + 1), "space": stacked["space"]},
        )
        .unstack("space")
        .rename("eof_unweighted")
    )

    pc_df = pd.DataFrame({"date": dates})
    for mode_idx in range(args.n_eofs):
        pc_df[f"PC{mode_idx + 1}"] = pcs[:, mode_idx]

    pc_df["water_year"] = assign_water_year(pc_df["date"].dt)
    cpm_daily = pc_df[["date", "water_year", "PC3"]].rename(columns={"PC3": "cpm_pc3_daily"})
    cpm_daily.to_csv(args.output_dir / "cpm_pc3_daily.csv", index=False)

    annual = (
        cpm_daily.groupby("water_year", as_index=False)
        .agg(
            cpm_pc3_nov_apr_mean=("cpm_pc3_daily", "mean"),
            number_of_days=("cpm_pc3_daily", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .sort_values("water_year")
    )
    annual.to_csv(args.output_dir / "cpm_pc3_nov_apr_by_water_year.csv", index=False)

    predictors = pd.read_csv(args.predictor_table_path)
    if len(predictors["water_year"]) != len(set(predictors["water_year"])):
        raise ValueError("Predictor table has duplicate water years.")
    overlap = predictors[["water_year", "Z1", "Z2"]].merge(annual, on="water_year", how="inner")
    overlap.to_csv(args.output_dir / "z1_z2_cpm_overlap_table.csv", index=False)

    summaries = [
        compute_summary("Z1", overlap["Z1"].to_numpy(), overlap["cpm_pc3_nov_apr_mean"].to_numpy(), stats),
        compute_summary("Z2", overlap["Z2"].to_numpy(), overlap["cpm_pc3_nov_apr_mean"].to_numpy(), stats),
    ]
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "correlation_summary.csv", index=False
    )

    eof_ds = xr.Dataset(
        {
            "eof_unweighted": eof_maps,
            "explained_variance_ratio": xr.DataArray(
                evr[: args.n_eofs], dims=("mode",), coords={"mode": np.arange(1, args.n_eofs + 1)}
            ),
            "singular_values": xr.DataArray(
                s[: args.n_eofs], dims=("mode",), coords={"mode": np.arange(1, args.n_eofs + 1)}
            ),
        }
    )
    eof_ds.to_netcdf(args.output_dir / "eofs_and_metadata.nc")

    metadata = {
        "era_input_path": str(args.era_input_path.resolve()),
        "predictor_table_path": str(args.predictor_table_path.resolve()),
        "time_name": time_name,
        "lat_name": lat_name,
        "lon_name": lon_name,
        "var_name": var_name,
        "converted_geopotential_to_height": bool(var_name == "z" or "geopotential" in var_name.lower()),
        "domain": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "anomaly_method": args.anomaly_method,
        "drop_leap_day": args.drop_leap_day,
        "overlap_years": overlap["water_year"].tolist(),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Minimal diagnostic plots.
    for mode in range(1, args.n_eofs + 1):
        fig, ax = plt.subplots(figsize=(8, 4))
        eof_maps.sel(mode=mode).plot(ax=ax)
        ax.set_title(f"EOF{mode}")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"eof{mode}.png", dpi=150)
        plt.close(fig)

    for predictor in ("Z1", "Z2"):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(overlap[predictor], overlap["cpm_pc3_nov_apr_mean"])
        summary = next(item for item in summaries if item["predictor"] == predictor)
        x = overlap[predictor].to_numpy()
        ax.plot(x, summary["slope"] * x + summary["intercept"], color="tab:red")
        ax.set_xlabel(predictor)
        ax.set_ylabel("CPM PC3 Nov-Apr mean")
        ax.set_title(
            f"{predictor} vs CPM\nr={summary['pearson_r']:.3f}, p={summary['pearson_p']:.3g}, n={summary['n']}"
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{predictor.lower()}_vs_cpm_scatter.png", dpi=150)
        plt.close(fig)

    print(f"Wrote CPM reproduction outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
