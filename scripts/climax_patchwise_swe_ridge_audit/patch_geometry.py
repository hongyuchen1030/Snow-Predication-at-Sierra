"""Exact geographic reconstruction of ClimaX's 2048 patch tokens.

Token ordering verified directly from source, not assumed:
  - timm.models.vision_transformer.PatchEmbed.forward(): x = self.proj(x)
    (Conv2d -> [B, D, H_patch, W_patch]), then x.flatten(2).transpose(1, 2)
    -- PyTorch's flatten(2) is C-order (row-major) over the last two dims,
    so patch index = h_patch_row * W_patch + w_patch_col.
  - climax/arch.py forward_encoder() calls token_embeds[id](x[:, i:i+1]) per
    variable (each producing [B, L, D] in that same order), stacks over
    variables, aggregates over variables per-token (aggregate_variables:
    einsum "bvld->blvd" then flatten(0,1) -- mixes V, never reorders L), adds
    pos_embed (built via get_2d_sincos_pos_embed(..., H_patch, W_patch, ...),
    i.e. the same H-then-W grid convention), and runs the token sequence
    through standard self-attention blocks, which never reorder tokens.
  - So the saved H's patch axis is unambiguously:
        patch_id = lat_row * 64 + lon_col,   lat_row in [0,32), lon_col in [0,64)
    where lat_row=0 is the SOUTHERNMOST band (grid ascends south->north, per
    the extraction script's own verify_grid: "lat must be ascending
    south-to-north") and lon_col=0 is the prime-meridian-adjacent band
    (grid is 0..358.59375, ascending, standard 0-360 convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_LAT_PATCH = 32
N_LON_PATCH = 64
N_PATCHES = N_LAT_PATCH * N_LON_PATCH
PATCH_SIZE_CELLS = 4
GRID_RES_DEG = 1.40625

SIERRA_LAT_MIN, SIERRA_LAT_MAX = 35.0, 42.0
SIERRA_LON_MIN, SIERRA_LON_MAX = -122.5, -118.0  # -180..180 convention


def climax_grid_lat_lon() -> tuple[np.ndarray, np.ndarray]:
    """The exact 128x256 grid-cell centers ClimaX operates on (verified
    against climax_frozen_mpi_smoke_test's own verify_grid() formula)."""
    lat = -90 + GRID_RES_DEG / 2 + np.arange(128) * GRID_RES_DEG
    lon = np.arange(256) * GRID_RES_DEG  # 0..358.59375, ascending
    return lat, lon


def lon_360_to_180(lon_360: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(lon_360) + 180.0) % 360.0) - 180.0


def build_patch_metadata() -> pd.DataFrame:
    lat_grid, lon_grid = climax_grid_lat_lon()
    half = GRID_RES_DEG / 2
    rows = []
    for lat_row in range(N_LAT_PATCH):
        cell_lats = lat_grid[lat_row * PATCH_SIZE_CELLS:(lat_row + 1) * PATCH_SIZE_CELLS]
        lat_min = float(cell_lats.min() - half)
        lat_max = float(cell_lats.max() + half)
        lat_center = float(cell_lats.mean())
        for lon_col in range(N_LON_PATCH):
            cell_lons = lon_grid[lon_col * PATCH_SIZE_CELLS:(lon_col + 1) * PATCH_SIZE_CELLS]
            lon_min_360 = float(cell_lons.min() - half)
            lon_max_360 = float(cell_lons.max() + half)
            lon_center_360 = float(cell_lons.mean())

            lon_min_180 = float(lon_360_to_180(lon_min_360))
            lon_max_180 = float(lon_360_to_180(lon_max_360))
            lon_center_180 = float(lon_360_to_180(lon_center_360))

            patch_id = lat_row * N_LON_PATCH + lon_col
            rows.append({
                "patch_id": patch_id, "row": lat_row, "col": lon_col,
                "lat_center": lat_center, "lon_center": lon_center_180,
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min_180, "lon_max": lon_max_180,
            })
    df = pd.DataFrame(rows)
    # Sierra intersection: box does not straddle the 0/360 meridian, and no
    # ClimaX patch straddles it either (256/4=64 divides evenly), so a plain
    # interval-overlap test in -180..180 is correct here.
    df["intersects_sierra"] = (
        (df["lat_max"] >= SIERRA_LAT_MIN) & (df["lat_min"] <= SIERRA_LAT_MAX) &
        (df["lon_max"] >= SIERRA_LON_MIN) & (df["lon_min"] <= SIERRA_LON_MAX)
    )
    assert len(df) == N_PATCHES
    assert df["patch_id"].nunique() == N_PATCHES
    return df
