#!/usr/bin/env python3
"""For every daily-available (variable, model, experiment, table) combination
already confirmed in artifacts/cmip6_daily_availability_audit/table_scan_raw.csv,
open the FIRST and LAST chunk file on disk (sorted) to report the true full
first/last timestamp, not just one representative decade chunk.
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import netCDF4
import numpy as np

RAW_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_daily_availability_audit/table_scan_raw.csv")
OUT_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_daily_availability_audit/table_scan_full_time_coverage.csv")


def read_time_edge(ds: netCDF4.Dataset, *, first: bool) -> str | None:
    if "time" not in ds.variables:
        return None
    time_var = ds.variables["time"]
    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    values = np.asarray(time_var[:])
    if values.size == 0 or units is None:
        return None
    value = values[0] if first else values[-1]
    try:
        return str(netCDF4.num2date(value, units=units, calendar=calendar))
    except Exception:
        return f"raw:{value}"


def main() -> None:
    with RAW_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    daily_rows = [r for r in rows if str(r.get("frequency", "")).lower() == "day" and r.get("file_path")]
    print(f"{len(daily_rows)} daily (variable, model, experiment, table) rows to expand", flush=True)

    out_rows = []
    for row in daily_rows:
        first_path = Path(row["file_path"])
        variable_dir = first_path.parent  # .../<grid_label>/<version>/
        nc_files = sorted(variable_dir.glob("*.nc"))
        if not nc_files:
            continue
        first_file, last_file = nc_files[0], nc_files[-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_first = netCDF4.Dataset(first_file, mode="r")
        first_ts = read_time_edge(ds_first, first=True)
        ds_first.close()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_last = netCDF4.Dataset(last_file, mode="r")
        last_ts = read_time_edge(ds_last, first=False)
        ds_last.close()

        out_rows.append(
            {
                "variable_id": row["variable_id"],
                "model": row["model"],
                "experiment": row["experiment"],
                "table_id_dir": row["table_id_dir"],
                "n_chunk_files": len(nc_files),
                "first_file": str(first_file),
                "last_file": str(last_file),
                "full_first_timestamp": first_ts,
                "full_last_timestamp": last_ts,
            }
        )
        print(f"{row['variable_id']:10s} {row['model']:16s} {row['experiment']:10s} {row['table_id_dir']:8s} "
              f"n_files={len(nc_files):3d} {first_ts} .. {last_ts}", flush=True)

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
