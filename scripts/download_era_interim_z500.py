#!/usr/bin/env python3
"""Prepare an ERA-Interim Z500 acquisition request for the CPM reproduction.

This helper does not fabricate authenticated UCAR GDEX download URLs. It writes
an explicit request specification and exits with a clear blocker message unless a
local ERA-Interim input file has already been supplied out of band.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("artifacts/cpm_era_interim_reproduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the request specification will be written.",
    )
    parser.add_argument(
        "--local-input-path",
        type=Path,
        default=None,
        help="Optional already-downloaded ERA-Interim file path to record instead of blocking.",
    )
    parser.add_argument(
        "--start-date",
        default="1981-11-01",
        help="Earliest timestamp needed to support wet-season means. Default supports WY1982-WY2016.",
    )
    parser.add_argument(
        "--end-date",
        default="2016-04-30",
        help="Latest wet-season date needed for the paper-period reproduction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request_spec = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "provider": "UCAR GDEX",
            "dataset_id": "d627000",
            "dataset_title": "ERA-Interim Project",
            "legacy_dataset_url": "https://rda.ucar.edu/datasets/ds627.0/",
            "current_dataset_url": "https://gdex.ucar.edu/datasets/d627000/",
            "product": "Analysis",
            "format": "WMO_GRIB1",
        },
        "variable_request": {
            "target_variable": "500 hPa geopotential or geopotential height (Z500)",
            "preferred_output_units": "meters geopotential height",
            "conversion_if_needed": "z500_m = geopotential / 9.80665",
        },
        "time_request": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "native_sampling": "6-hourly",
            "paper_period_note": "Paper analyzes 1982-2016 and aggregates ERA-Interim to daily data.",
        },
        "analysis_domain": {
            "lat_min": 20.0,
            "lat_max": 75.0,
            "lon_west": 170.0,
            "lon_east": 90.0,
            "lon_note": "Paper states 90W-170W for PCA; internal analysis can convert to 190E-270E.",
        },
        "blocker": {
            "status": "authenticated_download_required",
            "message": (
                "The GDEX metadata pages are reachable, but the file-level data-access workflow "
                "is sign-in gated in this environment. No authenticated download URL was fabricated."
            ),
        },
    }

    if args.local_input_path is not None:
        request_spec["local_input_path"] = str(args.local_input_path.resolve())
        request_spec["blocker"] = {
            "status": "resolved_with_local_input",
            "message": "A local ERA-Interim input path was supplied manually.",
        }

    spec_path = args.output_dir / "era_interim_z500_request_spec.json"
    spec_path.write_text(json.dumps(request_spec, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote request specification to {spec_path}")

    if args.local_input_path is None:
        print(
            "ERA-Interim automatic download is still blocked by authenticated GDEX access. "
            "Provide a local file path with --local-input-path after download."
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
