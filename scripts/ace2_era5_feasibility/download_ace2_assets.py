#!/usr/bin/env python3
"""Download a minimal ACE2-ERA5 asset set from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="allenai/ACE2-ERA5",
        help="Hugging Face repository containing the checkpoint and sample data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the snapshot subset will be stored.",
    )
    parser.add_argument(
        "--ic-year",
        default="2020",
        help="Initial-condition sample year to download (e.g. 2020).",
    )
    parser.add_argument(
        "--forcing-years",
        nargs="+",
        default=["2020"],
        help="Forcing years to download.",
    )
    parser.add_argument(
        "--include-training-artifacts",
        action="store_true",
        help="Also download training_validation_data/time_mean.nc if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "README.md",
        "ace2_era5_ckpt.tar",
        "inference_config.yaml",
        f"initial_conditions/ic_{args.ic_year}.nc",
    ]
    allow_patterns.extend(f"forcing_data/forcing_{year}.nc" for year in args.forcing_years)
    if args.include_training_artifacts:
        allow_patterns.append("training_validation_data/time_mean.nc")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.output_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )

    manifest = {
        "repo_id": args.repo_id,
        "snapshot_path": snapshot_path,
        "allow_patterns": allow_patterns,
    }
    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
