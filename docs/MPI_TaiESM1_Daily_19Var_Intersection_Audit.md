# MPI-ESM1-2-HR × TaiESM1 — Daily 19-Variable Intersection Audit

**Status: AVAILABILITY AUDIT ONLY.** No training samples were constructed, no
model was trained or modified, no new variable (e.g. `pr`) was added, no
related variable was substituted for another (e.g. `huss`↔`hus`,
`tos`↔`thetao`), no monthly value was repeated or interpolated as daily.

**Models / members** (from the project's existing `PARENT_RUNS` in
`scripts/process_cmip6_selected_regrid_standardize.py`, i.e. the exact
member already tied to the WUS-D3/CMIP6 pipeline):

| Model | institution_id | member_id | historical root | ssp370 root |
|---|---|---|---|---|
| MPI-ESM1-2-HR | MPI-M | `r3i1p1f1` | `.../CMIP/MPI-M/MPI-ESM1-2-HR/historical/r3i1p1f1` | `.../ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/r3i1p1f1` — **absent locally** (re-confirmed: `ls` → no such directory) |
| TaiESM1 | AS-RCEC | `r1i1p1f1` | `.../CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1` | `.../ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1` |

Per instructions, MPI-ESM1-2-HR's contribution is historical-only; no ssp370
branch was manufactured or substituted for it.

**Method:** every candidate table_id directory actually on disk for each of
the 11 unique physical variables behind the 19 fields was enumerated; where a
variable was present, **both the first and last chunk file (sorted) were
opened directly** with `netCDF4` to read global attributes, the vertical
coordinate, and the true first/last timestamp of the full local holding (not
one representative file). A field counts as genuinely daily only if a
candidate table (a) has `frequency=day` confirmed from the file itself, (b)
carries a **true fixed-value vertical coordinate** matching the required
pressure level(s)/depth (see §0 below — this caught a real trap), and (c)
that table's actual first/last timestamp spans the full required water-year
window: **historical 1980‑09‑01→2014‑12‑31**, **ssp370 2015‑01‑01→2100‑08‑31**
(sufficient to cover every Nov 1–Mar 31 window within each covered year).

Scripts: [`scripts/audit_cmip6_daily_variable_availability.py`](../scripts/audit_cmip6_daily_variable_availability.py),
[`scripts/audit_cmip6_daily_full_time_coverage.py`](../scripts/audit_cmip6_daily_full_time_coverage.py)
(both reused verbatim from the prior full-inventory audit — no new file scans
were needed, only a corrected re-analysis of the same raw metadata, see §0).

## 0. A trap found and corrected: hybrid-sigma levels are not pressure levels

MPI-ESM1-2-HR's `CFday` table carries `ua` and `hus` on dimensionless model
**hybrid-sigma levels** (`lev`, `standard_name=atmosphere_hybrid_sigma_pressure_coordinate`,
with `ap`/`b`/`ps` needed to reconstruct actual pressure = `ap + b·ps`,
which varies in time and space) — **not** a fixed `plev` coordinate. This
table nominally spans 1850–2014 (looks like "full coverage"), but it cannot
supply "ua at 850/200/50 hPa" without an additional vertical-interpolation
step this audit was told not to perform. The genuinely fixed-pressure-level
table for MPI `ua` (`EdayZ`, confirmed `plev=[...,850,...,200,...,50,...]
hPa` by reading the coordinate directly) only reaches **1999-12-31** — 15
years short of what's needed. This was cross-checked against every other
variable; `hus`, `va`, `zg` all have a genuine fixed-`plev` table (`Eday`/
`EdayZ`) that *does* reach 2014-12-31, so only `ua` (and `ta`, separately, for
coverage reasons below) is affected by this trap. **`ua_850`/`ua_200`/`ua_50`
are excluded from MPI's usable set for this reason**, not merely a coverage
shortfall — using `CFday` would require interpolation, which is explicitly
disallowed.

## 1. The 19-row table

`P/D verified?` = required pressure level(s)/depth confirmed present in the
**genuinely usable** table (fixed `plev`/`depth`, not hybrid-sigma).
"partial" coverage windows are stated exactly; a field is only counted
"daily?=YES" below if it covers the *entire* required window.

| Predictor | MPI hist daily? | MPI hist coverage | Tai hist daily? | Tai hist coverage | Tai ssp370 daily? | Tai ssp370 coverage | P/D verified? | Units | table_id (MPI / Tai) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `tos` | NO | `Oday` 1855–1974 (partial, misses WY1981–2014 entirely) | NO | no daily `Oday` table at all | NO | — | n/a | °C | Oday / — | Excluded both models |
| `siconc` | **YES** | `SIday` 1850–2014 (full) | NO | no daily `SIday` table at all | NO | — | n/a (surface) | % | SIday / — | MPI-only; TaiESM1 has no daily sea-ice table |
| `rlut` | **YES** | `day` 1850–2014 (full) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | n/a (TOA) | W m⁻² | day / day | **In common set** |
| `zg_500` | **YES** | `EdayZ` 1850–2014 (full) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 500 hPa confirmed in both | m | EdayZ / day | **In common set** |
| `ta_850` | NO | best fixed-plev table (`Eday`) 1985–2009 only (partial) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 850 hPa (TaiESM1) | K | Eday(partial) / day | MPI partial only |
| `ua_850` | NO | best fixed-plev table (`EdayZ`) 1865–1999 only (partial); `CFday` reaches 2014 but is **hybrid-sigma**, rejected (§0) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 850 hPa (TaiESM1) | m s⁻¹ | EdayZ(partial)/CFday(hybrid, rejected) / day | MPI partial only |
| `va_850` | **YES** | `EdayZ` 1850–2014 (full) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 850 hPa confirmed both | m s⁻¹ | EdayZ / day | **In common set** |
| `ua_200` | NO | same trap as `ua_850` — fixed-plev only to 1999 | NO | plev8 in `day` table has **no 200 hPa** (`[1000,850,700,500,250,100,50,10]`) | NO | same plev8 gap | n/a | m s⁻¹ | EdayZ(partial)/CFday(rejected) / day (no 200 hPa) | Doubly excluded |
| `va_200` | **YES** | `EdayZ` 1850–2014 (full), 200 hPa confirmed | NO | plev8 has no 200 hPa | NO | same plev8 gap | ✓ (MPI only) | m s⁻¹ | EdayZ / day (no 200 hPa) | TaiESM1-side gap excludes it |
| `ua_50` | NO | same trap as `ua_850` | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 50 hPa (TaiESM1) | m s⁻¹ | EdayZ(partial) / day | MPI partial only |
| `va_50` | **YES** | `EdayZ` 1850–2014 (full) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 50 hPa confirmed both | m s⁻¹ | EdayZ / day | **In common set** |
| `hus_850` | **YES** | `Eday` 1975–2014 (full — starts before required 1980) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 850 hPa confirmed both | 1 (kg/kg) | Eday / day | **In common set** |
| `psl` | **YES** | `day` 1850–2014 (full) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | n/a (surface) | Pa | day / day | **In common set** |
| `tas` | NO | `day` 1865–2009 only (partial, misses WY2010–2014) | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | n/a (surface) | K | day(partial) / day | MPI partial only |
| `zg_50` | **YES** | `EdayZ` 1850–2014 (full), 50 hPa confirmed | **YES** | `day` 1850–2014 (full) | **YES** | `day` 2015–2100 (full) | ✓ 50 hPa confirmed both | m | EdayZ / day | **In common set** |
| `ta_50` | NO | same partial `Eday` window as `ta_850` (1985–2009) | **YES** | `day` 1850–2014 (full), 50 hPa confirmed | **YES** | `day` 2015–2100 (full) | ✓ (TaiESM1) | K | Eday(partial) / day | MPI partial only |
| `mrso` | NO | `day` 1875–1979 only (misses all of WY1981–2014) | NO | no daily table at all | NO | — | n/a | kg m⁻² | day(partial) / — | Excluded both |
| `thetao_50m` | NO | no daily `thetao` product at any table | NO | no daily `thetao` product at any table | NO | — | n/a | — | — | Excluded both |
| `thetao_100m` | NO | same as above | NO | same as above | NO | — | n/a | — | — | Excluded both |

## 2. Usable sets by branch

**A. MPI historical usable daily set** (full 1980-09→2014-12 coverage, true
fixed level where required):
`{siconc, rlut, zg_500, zg_50, va_850, va_200, va_50, hus_850, psl}` — **9 fields**

**B. TaiESM1 historical usable daily set:**
`{rlut, zg_500, zg_50, ta_850, ta_50, ua_850, ua_50, va_850, va_50, hus_850, psl, tas}` — **12 fields**
(`ua_200`/`va_200` excluded: TaiESM1's `day`-table `plev8` set has no 200 hPa level.)

**C. TaiESM1 ssp370 usable daily set:**
identical to B — `{rlut, zg_500, zg_50, ta_850, ta_50, ua_850, ua_50, va_850, va_50, hus_850, psl, tas}` — **12 fields**, confirmed full 2015-01→2100-08 coverage for every one of them.

**D. Exact intersection**

```
V_common = V_MPI_historical ∩ V_TaiESM1_historical ∩ V_original_19
         = {siconc, rlut, zg_500, zg_50, va_850, va_200, va_50, hus_850, psl}
           ∩ {rlut, zg_500, zg_50, ta_850, ta_50, ua_850, ua_50, va_850, va_50, hus_850, psl, tas}
         = {rlut, zg_500, zg_50, va_850, va_50, hus_850, psl}
```

**V_common = 7 variables.** (`siconc` and `va_200` drop out because TaiESM1
has no genuine daily coverage of either; everything TaiESM1-only —
`ta_850, ta_50, ua_850, ua_50, tas` — drops out because MPI's fixed-pressure
daily archive for `ta`/`ua`/`tas` either stops early or (for `ua`) is only
available on hybrid-sigma levels.)

**Does every V_common variable continue through TaiESM1 ssp370?** Yes — all
7 (`rlut, zg_500, zg_50, va_850, va_50, hus_850, psl`) are confirmed with
full 2015-01→2100-08 daily coverage on TaiESM1 ssp370 (Table 1, columns 6–7).
Therefore:

```
V_common_historical = V_common_full
                     = {rlut, zg_500, zg_50, va_850, va_50, hus_850, psl}
```

No two-tier split is needed — the full 7-variable common set survives into
TaiESM1's ssp370 branch unchanged.

## 3. Sample-count implication (water years, not samples — no tensors built)

WUS-D3 daily SWE availability per model/branch, from
`artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels_summary.json`
(`composition_by_model_experiment`): MPI historical 34, TaiESM1 historical
34, TaiESM1 ssp370 86 — all three fully contained within the atmospheric
coverage windows confirmed above, so no additional SWE-side restriction
applies (WUS-D3's SWE holding is a strict superset here; the previously
known TaiESM1-ssp370 `mrso` monthly gap is irrelevant to this 7-variable set
since `mrso` is not in it).

| Dataset | Models/branches | Variable set | WY range(s) | WY count |
|---|---|---|---|---|
| **A** | MPI historical + TaiESM1 historical | `V_common_historical` (7 ch.) | MPI: WY1981–2014 (34); TaiESM1: WY1981–2014 (34) | **68** |
| **B** | MPI historical + TaiESM1 historical + TaiESM1 ssp370 | `V_common_full` (7 ch., same set) | MPI: WY1981–2014 (34); TaiESM1: WY1981–2014 (34) + WY2015–2100 (86) | **154** |

The trade-off is explicit and quantified: staying at 7 channels (rather than
attempting a larger, model-inconsistent channel set) buys roughly **2.3×
more water years** in Dataset B (154 vs. 68) at no cost in channel count
between A and B — the channel set does not shrink when ssp370 is added,
because every V_common variable already extends cleanly through TaiESM1's
ssp370 archive.

---
## FINAL ANSWER

1. **`V_common_historical`** = `{rlut, zg_500, zg_50, va_850, va_50, hus_850, psl}`
2. **Number of channels in `V_common_historical`: 7**
3. **Complete MPI historical WY count: 34** (WY1981–2014)
4. **Complete TaiESM1 historical WY count: 34** (WY1981–2014)
5. **`V_common_full`** = `{rlut, zg_500, zg_50, va_850, va_50, hus_850, psl}` (identical to `V_common_historical` — full continuity confirmed)
6. **Number of channels in `V_common_full`: 7**
7. **Complete TaiESM1 ssp370 WY count: 86** (WY2015–2100)
8. **Total usable WY count for Dataset A: 68** (34 MPI + 34 TaiESM1, historical only)
9. **Total usable WY count for Dataset B: 154** (34 + 34 + 86)
10. **Excluded variables from the original 19, with exact reason:**

| Variable | Reason for exclusion |
|---|---|
| `tos` | No daily `Oday` table for TaiESM1 at all; MPI's `Oday` archive ends 1974 (misses all of WY1981–2014). |
| `siconc` | MPI has full daily `SIday` coverage (1850–2014), but TaiESM1 has no daily `SIday` table at all. |
| `ta_850`, `ta_50` | MPI's only genuine fixed-pressure-level `ta` table (`Eday`) covers only 1985–2009 — misses both the start (1980–1985) and the tail (2010–2014) of the required window. Full at TaiESM1. |
| `ua_850`, `ua_50` | MPI's only genuine fixed-pressure-level `ua` table (`EdayZ`) stops at 1999; the longer-reaching `CFday` table is on **hybrid-sigma model levels**, not true pressure levels, and cannot supply 850/50 hPa without interpolation (disallowed). Full at TaiESM1. |
| `ua_200`, `va_200` | Doubly excluded: MPI's `ua_200` has the same hybrid-sigma/1999-cutoff problem as `ua_850`; **both** `ua_200` and `va_200` are absent from TaiESM1 entirely because TaiESM1's daily `plev8` level set (`1000,850,700,500,250,100,50,10` hPa) has no 200 hPa level at all. (`va_200` is otherwise fully available and level-verified at MPI — only the TaiESM1 side is the blocker.) |
| `tas` | MPI's only daily `tas` table ends 2009-12-31 (misses WY2010–2014). Full at TaiESM1. |
| `mrso` | No daily table at all for TaiESM1; MPI's daily `mrso` archive ends 1979-12-31 (misses all of WY1981–2014). |
| `thetao_50m`, `thetao_100m` | No genuine daily `thetao` product exists for either model, at any table — confirmed by direct directory/file inspection, not inferred. |

No ML architecture is recommended. No new variable was added. No related
variable was substituted. No interpolation (temporal or vertical) was
performed. No dataset tensors were constructed.

## Source paths used

- `/global/cfs/projectdirs/m3522/cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r3i1p1f1/{day,Eday,EdayZ,CFday,SIday,Oday}/...`
- `/global/cfs/projectdirs/m3522/cmip6/CMIP6/ScenarioMIP/MPI-M/MPI-ESM1-2-HR/ssp370/` — confirmed absent
- `/global/cfs/projectdirs/m3522/cmip6/CMIP6/CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1/day/...`
- `/global/cfs/projectdirs/m3522/cmip6/CMIP6/ScenarioMIP/AS-RCEC/TaiESM1/ssp370/r1i1p1f1/day/...`
- `artifacts/cmip6_swe_labels/wusd3_sierra_apr1_swe_labels_summary.json` (WUS-D3 SWE availability by model/branch)
- `artifacts/cmip6_daily_availability_audit/{table_scan_raw.csv,table_scan_full_time_coverage.csv}` (this audit's underlying evidence)
