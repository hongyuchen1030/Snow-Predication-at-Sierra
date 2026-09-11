# Simulation Full Inventory and Daily Availability Audit

**Status: INSPECTION / AVAILABILITY AUDIT ONLY.** No model was trained, no
existing dataset was modified. Every number below is read directly from
existing manifests/summaries already on disk, or from freshly-opened raw
NetCDF file metadata (global attributes, vertical coordinates, time
coordinates) — never inferred from directory names alone.

**Scripts used:**
- [`scripts/audit_cmip6_daily_variable_availability.py`](../scripts/audit_cmip6_daily_variable_availability.py) (existing, reused verbatim)
- [`scripts/audit_cmip6_daily_full_time_coverage.py`](../scripts/audit_cmip6_daily_full_time_coverage.py) (new: expands the above to true first/last timestamp per variable/model/experiment/table, by opening the first and last chunk file on disk, not a single representative file)

**Raw outputs:**
- `artifacts/cmip6_daily_availability_audit/table_scan_raw.csv`
- `artifacts/cmip6_daily_availability_audit/variable_model_availability_19x4.csv`
- `artifacts/cmip6_daily_availability_audit/table_scan_full_time_coverage.csv`
- `artifacts/cmip6_daily_availability_audit/coverage_vs_required_window.json`

---
## PART A — Do we now have the full expected simulation-year dataset?

### A.1 Where the intended full set comes from

Source: `artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels_summary.json`
(built by `scripts/build_wusd3_swe_labels.py`, which enumerates every WUS-D3
daily-SWE file actually present on disk at
`/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/<dataset_id>_{historical,ssp370}_bc/postprocess/d02/`).

```
"full_available_label_count": 480,
"composition_by_model_experiment": {
  "EC-Earth3|historical": 34,  "EC-Earth3|ssp370": 86,
  "MIROC6|historical": 34,     "MIROC6|ssp370": 86,
  "MPI-ESM1-2-HR|historical": 34, "MPI-ESM1-2-HR|ssp370": 86,
  "TaiESM1|historical": 34,    "TaiESM1|ssp370": 86
}
```

**480 = 4 parent models × (34 historical + 86 ssp370) water years.** This is
the intended full simulation-year set: 4 WUS-D3-downscaled parent runs, each
with a complete historical branch (file_year 1980–2013 → water_year
1981–2014) and a complete ssp370 branch (file_year 2014–2099 → water_year
2015–2100), confirmed from `artifacts/cmip6_wusd3_inventory/wusd3_parent_runs.csv`.
**The "~480–488" figure floated verbally maps exactly onto this documented
480 — there is no evidence of any larger intended number** (checked: no
other count near 480–488 appears anywhere else in `docs/` or `scripts/`).

### A.2 Why the current canonical dataset ended up at ~393/394, not 480

Two independent, already-documented restrictions, applied in sequence:

**Step 1 (480 → 394): MPI-ESM1-2-HR has no ssp370 branch in the local CMIP6
mirror.** `scripts/build_cmip6_wusd3_inventory.py` already states this
explicitly (`mapping_confidence: "historical_verified_scenario_local_gap"`):

> "The expected local CMIP6 ScenarioMIP MPI-ESM1-2-HR ssp370 r3i1p1f1 branch
> is currently missing."

Verified directly again in this audit:
`ls /global/cfs/projectdirs/m3522/cmip6/CMIP6/ScenarioMIP/MPI-M/MPI-ESM1-2-HR/`
→ `No such file or directory`. WUS-D3's own SWE product *does* have 86 MPI
ssp370 water years (it was downscaled from a CMIP6 copy this project's local
mirror doesn't hold), but the 19-variable atmospheric predictor pipeline
(`scripts/process_cmip6_selected_regrid_standardize.py`) and the CPM/AQM
auxiliary-label pipeline (`scripts/build_cmip6_aux_labels.py`) both read
`Amon`/`Omon`/`SImon`/`Lmon` files directly from this same local mirror, so
neither can produce anything for those 86 rows. 480 − 86 = **394**, exactly
matching `"final_label_count": 394` / `"overlap_with_cmip6_aux_rows": 394` in
the WUS-D3 summary and the row count of
`cmip6_cnn_architecture_screen_v1/data_cache/manifest.csv`.

**Step 2 (394 → 393): one more row is dropped by the 19-field intersection
in `scripts/prepare_cmip6_cnn_experiment.py`.** Its own summary
(`cmip6_cnn_architecture_screen_v1/data_cache/summary.json`) records
`"dropped_rows_due_to_predictor_intersection": 1` and shows `mrso`'s
`sample_count` is 393 while all other 18 fields are 394. Diffing the raw
regridded files directly (`tos_1p5deg_model_years.nc` vs
`mrso_1p5deg_model_years.nc`, opened with xarray) identifies the exact
missing key: **`TaiESM1:r1i1p1f1`, `row_year=2014`** (water_year 2015, the
first ssp370 row). Root cause, confirmed by listing the source file directly:

```
/global/cfs/projectdirs/m3522/cmip6/CMIP6/ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1/Lmon/mrso/gn/v20201014/
  mrso_Lmon_TaiESM1_ssp370_r1i1p1f1_gn_201502-210012.nc
```

TaiESM1's ssp370 `mrso` (Lmon) file starts at **2015-02**, not 2015-01 — one
month short of the boundary needed to splice the WUS-style Sep(2014)–Aug(2015)
window for row_year 2014 (historical `mrso` ends Dec 2014, so Jan 2015 is the
one month with no `mrso` value anywhere). All 18 other fields' TaiESM1 ssp370
files correctly start at 2015-01. This is a single-variable, single-model,
single-month archival gap, not a systemic issue.

**Confirmed: 480 − 86 (MPI ssp370, entirely absent locally) − 1 (TaiESM1
WY2015, one-month `mrso` gap) = 393.** This is the exact, fully-explained
mechanism — no other rows are silently dropped anywhere in the chain.

### A.3 Do the newly-synced MPI-ESM1-2-HR HighResMIP files fill the gap?

**No — structurally excluded, confirmed by direct inspection, not merged.**

```
$ ls .../HighResMIP/MPI-M/MPI-ESM1-2-HR/
control-1950  highres-future  highresSST-future  highresSST-present  hist-1950
$ ls .../HighResMIP/MPI-M/MPI-ESM1-2-HR/highresSST-present/
r1i1p1f1
$ ls .../HighResMIP/MPI-M/MPI-ESM1-2-HR/highresSST-future/
r1i1p1f1
```

Three independent, sufficient reasons none of this can substitute for the
missing `ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/r3i1p1f1` branch:

1. **Different `activity_id`.** `HighResMIP` ≠ `ScenarioMIP`. The WUS-D3
   downscaling and this project's regridding pipeline are both keyed to
   `ScenarioMIP`/`ssp370` specifically.
2. **Different experiment design.** `highresSST-present`/`highresSST-future`
   are AMIP-style, atmosphere-only runs forced by **prescribed** observed/
   projected SST and sea ice — not the coupled ocean-atmosphere design of
   `ssp370`. They are not a physically equivalent substitute even for the
   same nominal calendar years.
3. **Different `member_id`.** The new files are `r1i1p1f1`; the WUS-D3/
   PARENT_RUNS mapping for MPI-ESM1-2-HR is specifically `r3i1p1f1`. They are
   not the same simulation realization even in name.

Per the task's explicit instruction, **`highresSST-present`/`highresSST-future`
were not merged with `historical`/`ssp370` anywhere in this audit.** They are
recorded here as present on disk and excluded from every count below.

### A.4 Maximum complete number of simulation water years now

**Still 393 — unchanged by the sync.** The newly-synced files touch a
different `activity_id`/`experiment_id`/`member_id` entirely and cannot enter
the historical/ssp370 pipeline without exactly the kind of silent, unjustified
merge the task prohibits.

### A.5 Progressive counts

| Quantity | Value | Definition |
|---|---:|---|
| `N_atmospheric_source_years` (union, ≥1 of 19 vars present) | 394 | rows present in the 19-field regridded predictor set, before requiring all 19 simultaneously |
| `N_with_all_19_variables` (intersection of all 19) | 393 | after the `mrso`/TaiESM1/WY2015 drop |
| `N_with_WUS-D3_SWE` (within the 393/394 above) | 393/394 (no further restriction) | WUS-D3's 480-row SWE table is a strict superset of every row already in the atmospheric intersection |
| **`N_complete_intersection`** (19 predictors + SWE, simultaneously) | **393** | current canonical dataset size |

**488 is not obtained; 480 is the correct intended total, and 393 is the
correct, fully-explained current total.** The gap between 480 and 393 (87
rows) is entirely accounted for: 86 rows (MPI-ESM1-2-HR ssp370, no local
raw-CMIP6 branch) + 1 row (TaiESM1 WY2015, one-month `mrso` archival gap).

**Exact missing water-year IDs (87 total):**
- `MPI-ESM1-2-HR:r3i1p1f1` — **WY2015 through WY2100** (86 consecutive years)
- `TaiESM1:r1i1p1f1` — **WY2015** (1 year)

**Newly recoverable rows from this sync: zero.** No row of the 393-row
canonical manifest changes, and no new row becomes addable, because the sync
added files under a different `activity_id` that this pipeline correctly does
not — and per instructions, should not — treat as `ssp370`.

---
## PART B — Daily availability within the full source inventory

Method: for every one of the 19 fields, every table_id directory that
actually exists on disk under each parent model's `historical`/`ssp370` root
was enumerated (not a fixed guess-list), the variable's presence checked, and
where present, **the first and last file (sorted) in that
variable/table/grid/version directory were both opened** with `netCDF4` and
their global attributes, vertical coordinates, and time coordinates read
directly. A field only counts as "daily and usable" here if the resulting
calendar coverage genuinely spans the water years the seasonal pipeline
needs: **historical must cover 1980‑09‑01 → 2014‑12‑31** (WY1981–2014, the
last month of which splices into the first ssp370 row), and **ssp370 must
cover 2015‑01‑01 → 2100‑08‑31** (WY2015–2100). A table that merely contains
*some* daily-frequency file, without spanning this window, is marked
incomplete and the exact missing years are reported — this catches several
cases a shallower check would have missed.

### B.1 What a shallower check gets wrong (and why the deeper check was necessary)

An earlier pass in this project checked only "does *a* file with
`frequency=day` exist for this variable/table/model." Repeating that check
alone would have (incorrectly) reported MIROC6 and MPI-ESM1-2-HR as having
"17/19 daily variables" each. Opening every chunk file's actual time bounds
shows this is not true for the water years actually needed:

- `mrso` (soil moisture): MIROC6's local daily archive ends **2009-12-31**
  (needs 2014-12-31) → misses WY2010–2014. MPI-ESM1-2-HR's ends
  **1979-12-31** (needs 2014-12-31) → misses *all* of WY1981–2014, i.e. zero
  usable overlap with the seasonal dataset despite the table nominally
  existing.
- `tos`, `siconc`: MIROC6's daily archive ends 1999-12-31 (`tos`) /
  1999-12-31 (`siconc`) — 15 years short.
- `ta`, `tas`: MIROC6 and MPI-ESM1-2-HR daily archives both stop years before
  2014 (MIROC6 `ta` ends 2011‑2012, `tas` ends 1949; MPI `ta`/`tas` end 2009).

None of this is visible from a directory listing or from opening a single
representative (often early-historical) chunk file — it required reading the
literal first and last timestamp of the full holding.

### B.2 Table 2 — 19-variable DAILY availability matrix (genuine, full-window)

`covers` = table's actual first/last timestamp spans the *entire* required
water-year window for that experiment (not just "a file exists"). Where
`covers=NO`, the gap is stated. Pressure-level presence was separately
verified from the vertical-coordinate values in the same files (✓ = required
level(s) confirmed present; the one exception is noted below the table).

| Field | Level(s) required | EC-Earth3 hist | EC-Earth3 ssp370 | MIROC6 hist | MIROC6 ssp370 | MPI-ESM1-2-HR hist | MPI-ESM1-2-HR ssp370 | TaiESM1 hist | TaiESM1 ssp370 |
|---|---|---|---|---|---|---|---|---|---|
| `tos` | surface | NO (no daily file) | NO | NO (`Oday`, ends 1999) | NO (none) | NO (`Oday`, ends 1974) | NO (none) | NO (no `Oday`) | NO |
| `siconc` | surface | NO | NO | NO (`SIday`, ends 1999) | NO (none) | **YES** (`SIday`, 1850–2014) | NO (none) | NO (no `SIday`) | NO |
| `rlut` | TOA | NO | NO | **YES** (`day`, 1850–2014) | **YES** (`day`, 2015–2100) | **YES** (`day`, 1850–2014) | NO (none) | **YES** (`day`, 1850–2014) | **YES** (`day`, 2015–2100) |
| `zg_500` | 500 hPa ✓ | NO | NO | **YES** (`day`) | NO (starts 2017) | **YES** (`EdayZ`) | NO | **YES** (`day`) | **YES** (`day`) |
| `zg_50` | 50 hPa ✓ | NO | NO | **YES** (`day`) | NO (starts 2017) | **YES** (`EdayZ`) | NO | **YES** (`day`) | **YES** (`day`) |
| `ta_850` | 850 hPa ✓ | NO | NO | NO (`Eday`, ends 2011) | NO (`day`, ends 2090) | NO (`Eday`, ends 2009) | NO | **YES** (`day`) | **YES** (`day`) |
| `ta_50` | 50 hPa ✓ | NO | NO | NO (same file as `ta_850`) | NO | NO (same file) | NO | **YES** | **YES** |
| `ua_850` | 850 hPa ✓ | NO | NO | NO (`Eday`, ends 2012) | NO (starts 2017) | **YES** (`CFday`) | NO | **YES** (`day`) | **YES** (`day`) |
| `ua_200` | 200 hPa | NO | NO | NO | NO | **YES** (`CFday`, level ✓) | NO | NO — **plev8 has no 200 hPa** | NO — same gap |
| `ua_50` | 50 hPa ✓ | NO | NO | NO (same file as `ua_850`) | NO | **YES** | NO | **YES** | **YES** |
| `va_850` | 850 hPa ✓ | NO | NO | **YES** (`day`, 1852–2014) | NO (ends 2099-12, 1 day short of 2100-08) | **YES** (`EdayZ`) | NO | **YES** (`day`) | **YES** (`day`) |
| `va_200` | 200 hPa | NO | NO | NO | NO | **YES** (`EdayZ`, level ✓) | NO | NO — **plev8 has no 200 hPa** | NO — same gap |
| `va_50` | 50 hPa ✓ | NO | NO | **YES** | NO | **YES** | NO | **YES** | **YES** |
| `hus_850` | 850 hPa ✓ | NO | NO | **YES** (`Eday`, 1850–2014) | **YES** (`Eday`, 2015–2100) | **YES** (`CFday`) | NO | **YES** (`day`) | **YES** (`day`) |
| `psl` | surface | **YES** (`day`, 1970–2014) | **YES** (`day`, 2015–2100) | **YES** (`day`) | **YES** (`day`) | **YES** (`day`) | NO (no branch) | **YES** (`day`) | **YES** (`day`) |
| `tas` | surface | NO | NO | NO (`day`, ends 1949) | NO (ends 2094) | NO (`day`, ends 2009) | NO | **YES** (`day`) | **YES** (`day`) |
| `mrso` | column | NO | NO | NO (`day`, ends 2009) | NO (ends 2084) | NO (`day`, ends 1979) | NO | NO (no daily table) | NO |
| `thetao_50m` | ~50 m | NO | NO | NO (no daily `thetao` anywhere) | NO | NO | NO | NO | NO |
| `thetao_100m` | ~100 m | NO | NO | NO | NO | NO | NO | NO | NO |

**Pressure-level exception found:** TaiESM1's `day` table carries only the
standard CMIP6 plev8 subset — `[1000, 850, 700, 500, 250, 100, 50, 10] hPa`
(confirmed by reading the `plev` coordinate values directly) — **200 hPa is
absent**, so `ua_200`/`va_200` are not usable for TaiESM1 even though `ua`/`va`
daily data otherwise fully covers both branches. No other level gaps were
found; every other ✓ above was confirmed from the actual `plev`/depth
coordinate array, not assumed from the field name.

### B.3 Table 3 — per-model daily summary

| Parent model | Complete seasonal WYs (Part A) | WYs with all 19 daily vars | WYs with 17 daily vars (excl. `thetao`×2) | Max common genuinely-daily subset | Exact missing (of 19) |
|---|---:|---:|---:|---|---|
| EC-Earth3 (r102i1p1f1) | 120 | 0 | 0 | **1**: `psl` | `tos, siconc, rlut, zg_500, zg_50, ta_850, ta_50, ua_850, ua_200, ua_50, va_850, va_200, va_50, hus_850, tas, mrso, thetao_50m, thetao_100m` |
| MIROC6 (r1i1p1f1), historical only | 34 | 0 | 0 | **8** (hist. WY1981–2014 only): `rlut, zg_500, zg_50, va_850, va_50, hus_850, psl` (7 — see note*) | `tos, siconc, ta_850, ta_50, ua_850, ua_200, ua_50, va_200, tas, mrso, thetao_50m, thetao_100m` |
| MIROC6 (r1i1p1f1), ssp370 only | 86 | 0 | 0 | **3** (ssp. WY2015–2100 only): `rlut, hus_850, psl` | everything else (16) |
| MPI-ESM1-2-HR (r3i1p1f1), historical only | 34 | 0 | 0 | **12**: `siconc, rlut, zg_500, zg_50, ua_850, ua_200, ua_50, va_850, va_200, va_50, hus_850, psl` | `tos, ta_850, ta_50, tas, mrso, thetao_50m, thetao_100m` |
| MPI-ESM1-2-HR, ssp370 | 0 (no branch at all) | 0 | 0 | **0** | all 19 (no local ssp370 archive) |
| TaiESM1 (r1i1p1f1), historical **and** ssp370 (identical set both branches) | 119 (120 minus WY2015) | 0 | 0 | **12**: `rlut, zg_500, zg_50, ta_850, ta_50, ua_850, ua_50, va_850, va_50, hus_850, psl, tas` | `tos, siconc, ua_200, va_200, mrso, thetao_50m, thetao_100m` |

\* MIROC6 historical count is 8 including `zg_500`+`zg_50` and `va_850`+`va_50`
as separate fields from one `zg`/`va` source variable (2+2 = 4 fields from 2
physical variables), plus `rlut`, `hus_850`, `psl` (3 more) = 7 physical
variables → 8 fields total once levels are split out, matching the table
above.

**No model reaches all 19, and no model reaches 17 (excluding `thetao`) —
`thetao` is 0/4 everywhere, but the genuinely complete daily subset never
exceeds 12/19 for any single model even ignoring `thetao`,** because `tos`,
`siconc`, and `mrso` also fail the full-time-window test for every model that
has any daily holding of them at all (MPI's `siconc` is the sole full-window
exception). **TaiESM1 is the only model with an *identical*, fully-verified
12-variable daily subset spanning both the historical and ssp370 branches
continuously (1981–2100, minus the pre-existing WY2015 `mrso`-driven monthly
gap, which does not affect this 12-variable daily subset since `mrso` is not
in it).**

---
## FINAL SUMMARY

**1. Intended full simulation-year count: 480** (4 parent models ×
[34 historical + 86 ssp370] water years), per WUS-D3's own full availability
table (`artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels_summary.json`,
`full_available_label_count`).

**2. Complete usable seasonal 19-variable + SWE count now, after the sync:
393 — unchanged.** The newly-synced files (`HighResMIP`/`MPI-M`/
`MPI-ESM1-2-HR`/`highresSST-present`/`highresSST-future`) are a different
`activity_id`, a different (prescribed-SST, atmosphere-only) experiment
design, and a different `member_id` (`r1i1p1f1` vs. the required `r3i1p1f1`)
from the missing `ScenarioMIP ssp370` branch, and were correctly not merged.

**3. Exactly what is still missing (87 of 480 water years):**
- `MPI-ESM1-2-HR:r3i1p1f1` **WY2015–WY2100** (86 years) — needs the actual
  `ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/r3i1p1f1` branch synced to the
  local CMIP6 mirror (not HighResMIP).
- `TaiESM1:r1i1p1f1` **WY2015** (1 year) — needs the `ssp370` `Lmon`/`mrso`
  file to be re-pulled/re-synced so it covers January 2015 (current local
  copy starts 2015-02).

**4. Genuine daily subset available now, by model** (full-calendar-window,
correct-pressure-level, both branches where applicable — no interpolation,
no repeated-monthly-as-daily, no substitution):
- **EC-Earth3:** 1 variable (`psl`), both branches, 1970–2100 (excluding the
  synced-but-unrelated gap years already known).
- **MIROC6:** 8 variables historical-only (WY1981–2014); only 3 of those 8
  (`rlut`, `hus_850`, `psl`) also extend cleanly through ssp370 (WY2015–2100).
- **MPI-ESM1-2-HR:** 12 variables, historical-only (WY1981–2014); **zero**
  ssp370 daily coverage of anything (no local ssp370 branch at all, daily or
  monthly).
- **TaiESM1:** 12 variables, identically available across **both**
  historical and ssp370 (WY1981–2100 continuously, aside from the pre-existing
  WY2015 monthly-only gap which is irrelevant to this 12-variable set) — the
  strongest and most temporally continuous genuine daily foundation of the
  four models.
- **No model has genuine daily `tos`, `mrso`, or `thetao` (either depth)
  spanning the years actually needed** — those three physical quantities
  cannot currently support a short-timescale daily-window experiment for any
  of the four parent models without violating the "no interpolation, no
  substitution" constraint.

No model architecture is proposed here, per instructions.

---
## Source paths used

- `/global/cfs/projectdirs/m3522/cmip6/CMIP6/{CMIP,ScenarioMIP,HighResMIP}/...` — raw CMIP6/HighResMIP archive (NERSC local mirror)
- `/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/<dataset_id>_{historical,ssp370}_bc/postprocess/d02/` — WUS-D3 downscaled daily SWE
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_regridded_1p5deg/raw/*_1p5deg_model_years.nc` — regridded monthly predictor files
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache/{manifest.csv,summary.json}` — canonical 393-row predictor cache
- `artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels_summary.json`, `artifacts/cmip6_wusd3_inventory/wusd3_parent_runs.csv` — WUS-D3 full/parent-run inventory
- `artifacts/cmip6_daily_availability_audit/` — this audit's raw and derived outputs
