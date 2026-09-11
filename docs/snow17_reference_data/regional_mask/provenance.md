# Sierra North/Central/South mask — KNN-filled provenance

## Lineage

1. **Seed assignment (unchanged).** Source:
   `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/home_migrated_20260828/swe_target_spatial_diagnostic/basin_assignment_grid_wy2021.npz`,
   produced by `scripts/plot_sierra_swe_huc8_basin_assignment_test.py`. This is the
   original USGS WBD HUC8 keyword-based classification (CDEC/DWR grouping:
   North = Trinity through Feather & Truckee; Central = Yuba/Tahoe through
   Merced/Walker; South = San Joaquin/Mono through Kern), applied to the WY2021
   active April-1 SWE footprint (>0.05 m). This seed assignment — 110,231 cells
   (North=29,912, Central=46,572, South=33,747) — is used **verbatim** as the
   starting point here. No seed-assigned cell's label was changed.

2. **Diagnosis.** The 16,918 cells left unassigned by the seed classification
   were traced to their containing HUC8 polygon (exact point-in-polygon test,
   not a bounding-box approximation) and split into two exclusive groups:
   - **Category A (12,719 cells): confirmed outside the intended Sierra Nevada**
     region — Great Basin Nevada, Modoc Plateau, Klamath Basin, and San Joaquin
     Valley-floor HUC8s that happen to fall inside the generous rectangular SWE
     extraction box (35–42°N, −122.5 to −118°W). These must remain unassigned.
   - **Category B (4,199 cells): confirmed geographically inside the Sierra
     Nevada**, whose containing HUC8 name simply didn't match any keyword in
     the seed classifier: `Crowley Lake` (4,080), `Fresno River` (53),
     `Upper Deer-Upper White` (30), `Upper Poso` (30), `Upper Bear` (6).

3. **Fill method (this artifact).** Category-B cells only were filled by
   **K=16 nearest-neighbor majority vote**, using a `scipy.spatial.cKDTree`
   built **exclusively from already-assigned seed cells** (Category-A cells are
   never inserted into the tree, so they cannot influence any vote). For each
   Category-B cell, the plurality label among its 16 nearest seed-assigned
   neighbors (longitude scaled by cos(mean latitude) for a locally accurate
   distance metric) is assigned; ties would be broken by the single nearest
   neighbor's label (0 ties occurred in this run). **The North/Central/South
   label was never derived from HUC8 identity or name** — HUC8 names were used
   only to reproduce the already-diagnosed Category-A/B eligibility split
   (i.e., which cells to touch), not to decide what label a touched cell
   receives.

## Result

| | count |
|---|---|
| Seed-assigned (unchanged) | 110,231 |
| Category-B filled (this step) | 4,199 |
| — filled North | 0 |
| — filled Central | 57 |
| — filled South | 4,142 |
| Category-A (still NaN, outside Sierra) | 12,719 |
| Remaining Sierra-interior NaN cells | 0 |

## Verification (post hoc only, not used to decide labels)

| HUC8 | n cells | KNN-assigned label(s) |
|---|---|---|
| Crowley Lake | 4,080 | South: 4,080 (100%) |
| Fresno River | 53 | Central: 51, South: 2 |
| Upper Deer-Upper White | 30 | South: 30 (100%) |
| Upper Poso | 30 | South: 30 (100%) |
| Upper Bear | 6 | Central: 6 (100%) |

## Files

- `basin_assignment_grid_wy2021_knn_filled.npz` / `.nc` — final mask
  (`assignment`: 1=North, 2=Central, 3=South, NaN=outside Sierra or no
  WY2021 active-SWE footprint). Also retains `seed_assignment` (pre-fill) and
  `category_b_filled_mask` for auditability.
- `fill_summary.json` — machine-readable version of the numbers above.
- `before_after_knn_fill.png` — full-domain before/after map.
- `crowley_lake_verification.png` — close-up of the Crowley Lake cluster
  against its filled surroundings.
- Script: `scripts/fill_sierra_basin_gaps_knn.py` (this run's exact code).

## What this artifact is not

This does **not** modify, recalibrate, or re-run the seed HUC8/CDEC
classification, and it does **not** touch the Category-A exclusion boundary.
It is purely a gap-fill over cells already confirmed (in the prior diagnostic
turn) to be inside the Sierra Nevada. The original
`basin_assignment_grid_wy2021.npz` diagnostic artifact was left untouched.
