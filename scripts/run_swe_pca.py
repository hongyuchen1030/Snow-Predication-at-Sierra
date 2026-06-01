from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snow_ml.analysis.pca_swe import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_PCA_TARGET_MONTH_DAY,
    run_swe_pca,
)
from snow_ml.data import add_region_args, region_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone PCA baseline on yearly SWE maps.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Water years to include. If omitted, use all discovered SWE files.",
    )
    parser.add_argument(
        "--target-mmdd",
        default=DEFAULT_PCA_TARGET_MONTH_DAY,
        help="Target month-day used as the one SWE map per water year, for example 04-01.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="Number of PCA components.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "experiments" / "swe_pca_mmdd0401",
        help="Directory for PCA arrays, metadata, CSV summaries, and plots.",
    )
    add_region_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = region_from_args(args)
    result = run_swe_pca(
        years=args.years,
        target_month_day=args.target_mmdd,
        n_components=args.n_components,
        output_dir=args.output_dir,
        region=region,
        coarsen_factor=args.coarsen_factor,
    )
    print(f"final full matrix shape (years, flattened grid): {result.full_matrix_shape}")
    print(f"final PCA matrix shape (years, all-year-valid cells): {result.matrix_shape}")
    print(f"final component maps shape (components, H, W): {tuple(result.component_maps.shape)}")
    print(f"final scores shape (years, components): {tuple(result.scores.shape)}")
    print(f"explained variance ratio: {[float(value) for value in result.explained_variance_ratio]}")


if __name__ == "__main__":
    main()
