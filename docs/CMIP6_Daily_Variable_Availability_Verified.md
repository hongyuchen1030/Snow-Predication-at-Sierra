# CMIP6 Daily-Resolution Availability — Ground-Truth Verification

**Purpose:** verify, from actual NetCDF file metadata (not directory-name listings and
not our already-processed monthly output), whether daily/sub-daily CMIP6 equivalents
of the existing 19 seasonal-CNN input variables exist for all 4 parent models. This
supersedes the "checked with `ls`" pass in
[Short_Timescale_7day_Dataset_Audit.md](Short_Timescale_7day_Dataset_Audit.md) §4,
which turned out to be too optimistic for EC-Earth3 (see §4 below).

**Method / reproducibility:** [scripts/audit_cmip6_daily_variable_availability.py](../scripts/audit_cmip6_daily_variable_availability.py).
Imports the live `PARENT_RUNS`/`FIELD_SPECS` from
`scripts/process_cmip6_selected_regrid_standardize.py` (no re-declared variable list).
For each of the 4 parent models × 2 experiments, it lists every CMIP6 table_id
directory actually present on disk, and for every one of the 12 unique physical
`variable_id`s used by our 19 fields, opens one representative NetCDF file (if any
exist under that table/variable path) with `netCDF4` and reads its **global
attributes directly** (`source_id`, `experiment_id`, `variant_label`, `table_id`,
`frequency`, `grid_label`), its **vertical coordinate values** (pressure level in Pa
or ocean depth in m, read from the file, not assumed), and its **time coordinate**
(decoded start/end). Level presence for a specific field (e.g. `ua_200`) is checked
against the vertical-coordinate values actually read from that candidate daily file,
not the field's name. Run: `/pscratch/sd/h/hyvchen/conda_envs/swe_torch_cpu3/bin/python
scripts/audit_cmip6_daily_variable_availability.py`.

**Raw outputs** (machine-readable, full detail):
- `artifacts/cmip6_daily_availability_audit/table_scan_raw.csv` — 217 rows, every
  (variable_id, model, experiment, table_id) combination actually found on disk, with
  full metadata per row.
- `artifacts/cmip6_daily_availability_audit/variable_model_availability_19x4.csv` — the
  76-row (19 fields × 4 models) table described below.
- `artifacts/cmip6_daily_availability_audit/audit_summary.json`

## 1. Monthly source metadata, verified from file attributes (not filenames)

For every one of the 19 fields, the file actually used by
`process_cmip6_selected_regrid_standardize.py` (`FIELD_SPECS[i].table_id`) was opened
and its global attributes read directly. Every one of the 19×4 = 76 (field, model)
combinations resolved to a real file with `frequency=mon` (`Amon`/`Omon`/`SImon`/`Lmon`
per the table already listed in `Short_Timescale_7day_Dataset_Audit.md` §2), the
expected `source_id`, and `variant_label` exactly matching the member ID declared in
`PARENT_RUNS` (`EC-Earth3:r102i1p1f1`, `MIROC6:r1i1p1f1`, `MPI-ESM1-2-HR:r3i1p1f1`,
`TaiESM1:r1i1p1f1`) — see `monthly_*` columns of
`variable_model_availability_19x4.csv`. One pre-existing, already-documented gap
confirmed independently here: **MPI-ESM1-2-HR has no `ssp370` data at all on this
local CMIP6 mirror** — `ls .../ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/` returns
nothing — matching the note already in
[CMIP6_Variable_Selection_Table.md](CMIP6_Variable_Selection_Table.md) line 6. This
is unrelated to daily-vs-monthly resolution; it affects the existing monthly pipeline
identically.

## 2. Daily-equivalent search — every table_id on disk was checked, not a guessed list

Rather than only checking `day`/`Oday`/`SIday`/`Eday` by name, the script enumerated
**every** table_id directory that actually exists under each parent model's root
(e.g. MIROC6 has `3hr, 6hrLev, 6hrPlev, 6hrPlevPt, AERmon, Amon, CFday, CFmon, E3hr,
Eday, EdayZ, Emon, EmonZ, LImon, Lmon, Oday, Odec, Ofx, Omon, SIday, SImon, day, fx`)
and, for each one containing the target `variable_id`, opened a file and read its
`frequency` attribute to confirm `day`. Relevant daily tables actually used by our 19
variables turned out to be `day`, `Oday`, `SIday`, `Eday`, `EdayZ` (all confirmed via
the `frequency` attribute, not the table name).

## 3. The 19-variable × 4-model daily availability table

`daily_historical` / `daily_ssp370` = a genuinely daily (`frequency=day`) file exists
for that exact variable **and** the exact pressure level / depth used by the field.
`FULL_MATCH` = daily coverage exists for every experiment (historical / ssp370) where
the field is *currently* used at monthly resolution (i.e. holding the pre-existing
MPI-ESM1-2-HR ssp370 gap constant, since that gap already exists in the monthly
baseline).

| Field | EC-Earth3 | MIROC6 | MPI-ESM1-2-HR | TaiESM1 | Models fully daily-covered |
|---|---|---|---|---|---|
| `tos` | ❌ no daily file (empty dir) | ⚠️ daily-hist only (`Oday`), no ssp370 daily | ✅ (no ssp370 branch anyway) | ❌ no `Oday` table at all | 1/4 |
| `siconc` | ❌ | ⚠️ daily-hist only (`SIday`), no ssp370 daily | ✅ (no ssp370 branch anyway) | ❌ no `SIday` table at all | 1/4 |
| `rlut` | ❌ no daily file (not in `day` dir) | ✅ (`day`) | ✅ (`day`, no ssp370 branch anyway) | ✅ (`day`) | 3/4 |
| `zg_500` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`EdayZ`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `ta_850` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `ua_850` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `va_850` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `ua_200` | ❌ empty daily dir | ⚠️ daily-hist only, no ssp370 daily | ✅ (`Eday`, no ssp370 branch) | ❌ daily `day` table has **plev8 only** (1000/850/700/500/250/100/50/10 hPa) — no 200 hPa | 1/4 |
| `va_200` | ❌ empty daily dir | ⚠️ daily-hist only, no ssp370 daily | ✅ (`Eday`, no ssp370 branch) | ❌ same plev8 gap as `ua_200` | 1/4 |
| `hus_850` | ❌ not in daily tables at all | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `psl` | ✅ (`day`) | ✅ (`day`) | ✅ (`day`, no ssp370 branch) | ✅ (`day`) | **4/4** |
| `tas` | ❌ empty daily dir | ✅ (`day`) | ✅ (`day`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `zg_50` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`EdayZ`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `ta_50` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `ua_50` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `va_50` | ❌ empty daily dir | ✅ (`Eday`) | ✅ (`Eday`, no ssp370 branch) | ✅ (`day`) | 3/4 |
| `mrso` | ❌ not in daily tables at all | ✅ (`day`) | ✅ (`day`, no ssp370 branch) | ❌ not in daily tables at all | 2/4 |
| `thetao_50m` | ❌ | ❌ (`thetao` absent from `Oday`) | ❌ (`thetao` absent from `Oday`) | ❌ no `Oday` table | **0/4** |
| `thetao_100m` | ❌ | ❌ | ❌ | ❌ | **0/4** |

**Per-model totals** (fields with real daily-historical data out of 19):
EC-Earth3 **1/19**, MIROC6 **17/19**, MPI-ESM1-2-HR **17/19**, TaiESM1 **12/19**.

## 4. Correction to the earlier `ls`-based pass

The earlier audit ([Short_Timescale_7day_Dataset_Audit.md](Short_Timescale_7day_Dataset_Audit.md) §4)
checked only whether a `day`/`Oday`/`SIday` directory *existed and listed the variable
name*, and reported EC-Earth3 as daily-available for `zg`, `ta`, `ua`, `va`, `psl`,
`tas`. Opening the actual files now shows that is wrong for 5 of those 6: EC-Earth3's
`day/ta`, `day/tas`, `day/ua`, `day/va`, `day/zg` directories exist as **empty stubs**
(zero grid/version subdirectories, zero `.nc` files) on this local mirror — only
`day/psl` actually contains data. **EC-Earth3 has real daily data for only 1 of the 19
fields (`psl`)**, not 6. This is the reason the task asked to verify from file
metadata rather than trust directory listings.

## 5. Answer to the question this task asked

**No** — a physically consistent daily version of the existing 19-channel dataset
cannot be constructed across all 4 parent models. Only `psl` has full daily coverage
on all 4 models. `thetao_50m`/`thetao_100m` have zero daily coverage on any of the 4
models under any table. EC-Earth3 in particular has essentially no usable daily data
locally (1/19). No variable substitution has been made — this is a verification
result only, per the task's request.
