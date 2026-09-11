# Short-Timescale Daily Sierra SWE Target Dataset

**Status: documentation/audit only.** No predictor tensors were built, nothing
was trained, and no existing validated SWE artifact was modified. Every
number below was read directly from the actual CSV/NPZ/JSON files, not
assumed from the task description.

## 1. Canonical target files

All paths are relative to `/global/homes/h/hyvchen/Snow-Predication-at-Sierra`.

| File | Role |
|---|---|
| `artifacts/taiesm1_daily_sierra_swe_sep_apr1.csv` | **Canonical file for short-timescale training** — the Sep1→Apr1-only subset, ready to index by water year and date. |
| `artifacts/taiesm1_daily_sierra_swe_sep_apr1.npz` | Same subset, machine-readable arrays — **use this for fast array-based loading in a training script.** |
| `artifacts/taiesm1_daily_sierra_swe_sep_apr1_summary.json` | Validation summary for the subset (record counts, date bounds, gap/leakage checks). |
| `artifacts/taiesm1_daily_sierra_swe.csv` | Source: the full year-round (Sep1→Aug31) daily series, all 120 water years. Kept for provenance/reference; **not** the training file (it still contains the Apr2–Aug31 melt/summer season the short-timescale experiment excludes). |
| `artifacts/taiesm1_daily_sierra_swe.npz` | Array form of the full year-round series. |
| `artifacts/taiesm1_daily_sierra_swe_summary.json` | Validation summary for the full series (includes the spatial-mask provenance used by both series — see §2). |

For short-timescale training, use **`taiesm1_daily_sierra_swe_sep_apr1.csv`** or **`taiesm1_daily_sierra_swe_sep_apr1.npz`** — not the full-year files.

## 2. Geographic target definition

- **Sierra region:** `lat 35.0–42.0`, `lon -122.5 to -118.0` (`snow_ml.data.DEFAULT_SIERRA_REGION`) — read directly from `taiesm1_daily_sierra_swe_summary.json`'s `sierra_region` field. This is the same box used by the project's existing UCLA observational target and existing WUS-D3 April-1 seasonal labels.
- **Spatial mask/domain:** a shared **EC-Earth3** WUS-D3 d02 reference grid (`reference_grid_dataset: "ec-earth3_r1i1p1f1_2_historical_bc"`), per the existing pipeline's convention of using one common reference grid for all four parent models (same fixed regional WRF domain). `mask_type: "fractional"`. Grid shape `93 × 83`, of which **3,362 cells** fall inside the Sierra box, totaling **188,244.5 km²** (`total_area_km2` in the summary JSON; selected-cell lat/lon bounds: `35.070–41.932 N`, `-122.416 to -118.088`).
- **Spatial reduction method:** area-weighted spatial mean over the selected cells (native WUS-D3 cell area × Sierra box mask, `weighted_masked_mean`) — a single scalar Sierra-mean SWE value per day. No elevation/mountain classification, no regridding to a different grid — same geographic box applied directly to TaiESM1's own native cells, exactly as documented for the SWE-build step.
- **Units:** `mm` (SWE), confirmed from the source `snow` variable's own NetCDF units attribute (also carried through unchanged — the `swe_mm` column/array name reflects this).

## 3. Model and experiment origin

- **Parent model:** TaiESM1 (WUS-D3-downscaled), member `r1i1p1f1` — confirmed via the `model` column/array (`"TaiESM1"` for every record) and the underlying build pipeline's `PARENT_RUNS` definition.
- **Historical years:** water years **1981–2014** (34 water years) — `experiment == "historical"`.
- **SSP years:** water years **2015–2100** (86 water years) — `experiment == "ssp370"`.
- **Experiment labels:** exactly two values appear in the `experiment` field: `"historical"` and `"ssp370"`. No overlap between the two water-year sets (verified: empty intersection).
- **Total WY coverage:** **120 water years**, WY1981 through WY2100 — 34 historical + 86 ssp370 = 120, matching the record count exactly.

## 4. Temporal coverage

- **First retained date:** `1980-09-01` (start of WY1981).
- **Last retained date:** `2100-04-01` (end of WY2100).
- **Months included:** September, October, November, December, January, February, March in full, plus **April 1 only** (no April 2 onward, no May–August at all).
- **Records per water year:** exactly **213** for every one of the 120 water years (uniform — confirmed, not just assumed: `records_per_water_year.distinct_values == [213]`, `uniform: true`). This is a fixed constant because the source calendar is `365_day` (no leap-year variability): Sep(30)+Oct(31)+Nov(30)+Dec(31)+Jan(31)+Feb(28)+Mar(31)+Apr1(1) = 213.
- **Number of water years:** **120**.
- **Total records:** **25,560** (120 × 213) — confirmed by direct row count (25,561 CSV lines − 1 header = 25,560).
- **Calendar:** `365_day` (no-leap) — every year, including calendar leap years, has exactly 365 days; there is never a Feb 29.
- **Water-year convention:** Sep 1 of calendar year Y through Apr 1 of calendar year Y+1 belongs to **WY(Y+1)**. Equivalently, each WUS-D3 source file is named by `file_year` (spanning Sep1(file_year)→Aug31(file_year+1)); `water_year = file_year + 1`, and this dataset retains only the first 213 days of each such file.

## 5. File organization

### CSV (`taiesm1_daily_sierra_swe_sep_apr1.csv`)

Columns, in order, exactly as they appear in the file header:

| Column | Meaning |
|---|---|
| `model` | Always `"TaiESM1"`. |
| `experiment` | `"historical"` or `"ssp370"`. |
| `water_year` | Integer WY per the convention in §4 (1981–2100). |
| `file_year` | The underlying WUS-D3 source file's year label (`water_year - 1`). |
| `cal_year` | Actual calendar year of this record (decoded from the source NetCDF `day`/time coordinate via `cftime`, not inferred from the filename). |
| `cal_month` | Calendar month (1–12). |
| `cal_day` | Calendar day of month. |
| `date_key` | `"YYYY-MM-DD"` string built from `cal_year`/`cal_month`/`cal_day` — convenient join/lookup key. |
| `day_of_year_source` | Position within the water year's 213-day retained window, 1 = Sep 1 … 213 = Apr 1. |
| `swe_mm` | Sierra-mean area-weighted SWE for that day, in mm. |

### NPZ (`taiesm1_daily_sierra_swe_sep_apr1.npz`)

All arrays are length **25,560**, one entry per daily record, in the same row order as the CSV (confirmed by inspecting the file directly):

| Array | dtype | Meaning |
|---|---|---|
| `water_year` | `int32` | Same as CSV `water_year`. |
| `year` | `int32` | Same as CSV `cal_year`. |
| `month` | `int8` | Same as CSV `cal_month`. |
| `day` | `int8` | Same as CSV `cal_day`. |
| `date_key` | `<U10` (string) | Same as CSV `date_key`, e.g. `"1980-09-01"`. |
| `day_of_year` | `int16` | Same as CSV `day_of_year_source`, 1–213. |
| `swe_mm` | `float64` | Same as CSV `swe_mm`. |
| `experiment` | `<U10` (string) | `"historical"` or `"ssp370"`. |
| `model` | `<U7` (string) | Always `"TaiESM1"`. |

(For reference, the full year-round `taiesm1_daily_sierra_swe.npz` has the same shape/dtype pattern but 43,800 entries, no `water_year`/`date_key` array, and a `day_of_year` that runs 1–365 instead of 1–213 — it is not the file to index for short-timescale training.)

## 6. Training target indexing

To retrieve the SWE target for a given **(water_year, date)`**:

- **CSV:** filter rows where `water_year == <WY>` and `date_key == "<YYYY-MM-DD>"` (or `cal_month`/`cal_day`), then read `swe_mm`. Rows are already sorted by `water_year`, then chronologically within the window (`day_of_year_source` ascending).
- **NPZ:** build a boolean mask `(water_year_arr == WY) & (day_of_year_arr == d)` (where `d` is 1–213, mapped from the target calendar date via the fixed Sep1→Apr1 sequence in §4), or precompute a `{(water_year, day_of_year): row_index}` lookup dict once at load time for O(1) access.
- **For a sequence window** ending at day index `d` within a given water year: take the contiguous slice `day_of_year ∈ [d-k+1, d]` **within that same `water_year`** — since each water year's 213 days are stored contiguously and in order, this is a simple slice, but the boundary must be checked explicitly (`d - k + 1 >= 1`) to avoid reading into the previous water year's rows, since water years are stored back-to-back in the same array with no separator.
- **1-day target** `ΔSWE_1d(t) = SWE(t+1) - SWE(t)`: valid only where `day_of_year(t) < 213` (i.e. `t` is not Apr 1) and `t+1` is the very next row within the same `water_year` — do not chain across the `day_of_year=213 → day_of_year=1` boundary between one water year and the next, since those are not temporally adjacent calendar dates in this subset (day 213 of WY_n is Apr 1, day 1 of WY_(n+1) is the following Sep 1, ~5 months later).

## 7. Validation checks

- **Daily resolution — confirmed.** Each retained record is one calendar day (`365_day` calendar, no monthly or seasonal aggregation); 213 discrete daily rows per water year, not a single seasonal scalar.
- **Complete Sep1–Apr1 coverage — confirmed.** Every one of the 120 water years has exactly 213 records, first = Sep 1, last = Apr 1 of the following calendar year.
- **No Apr2–Aug31 leakage — confirmed.** `taiesm1_daily_sierra_swe_sep_apr1_summary.json` reports `"6_apr2_aug31_violations": 0` — verified by direct inspection, zero records with month/day outside the Sep1–Apr1 window.
- **Mar31→Apr1 present every WY — confirmed.** `taiesm1_daily_sierra_swe_sep_apr1_summary.json` reports `"7_water_years_missing_mar31_or_apr1": []` — every one of the 120 water years has both dates present.
- **No cross-water-year chaining — confirmed by construction.** Records are grouped and ordered strictly within `water_year`; the last record of one WY (Apr 1) and the first record of the next WY (the following Sep 1) are adjacent rows in storage but are **not** calendar-adjacent days, so a training script must gate on `water_year` equality (§6) before forming any 1-day (or longer) pair — this dataset does not do that gating for you; it only guarantees the underlying daily values are gap-free and correctly labeled.

---

**Report:** `docs/Short_Time_Daily_Sierra_SWE_Target_Dataset.md`
**Canonical CSV:** `artifacts/taiesm1_daily_sierra_swe_sep_apr1.csv`
**Canonical NPZ:** `artifacts/taiesm1_daily_sierra_swe_sep_apr1.npz`

`TaiESM1 daily Sierra SWE target: WY1981-WY2100 (120 water years), Sep1-Apr1 within each WY, 213 records/WY, 25560 total records, mm.`
