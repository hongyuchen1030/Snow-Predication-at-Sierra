# CMIP6 Daily 19-Variable × 4-Model Availability Audit

**Status: METADATA/FILE-AVAILABILITY AUDIT ONLY.** No dataset was designed,
no architecture recommended, no tensors built, no vertical or temporal
interpolation performed, no variable derived or substituted, no model set
reduced. Executed entirely on the login node (targeted `ls`/glob + `netCDF4`
metadata reads only — no GPU, no recursive filesystem scan).

**Models / members** (existing WUS-D3/CMIP6 pipeline definitions, from
`PARENT_RUNS` in `scripts/process_cmip6_selected_regrid_standardize.py`):

| Model | institution_id | member_id |
|---|---|---|
| EC-Earth3 | EC-Earth-Consortium | `r102i1p1f1` |
| MIROC6 | MIROC | `r1i1p1f1` |
| MPI-ESM1-2-HR | MPI-M | `r3i1p1f1` |
| TaiESM1 | AS-RCEC | `r1i1p1f1` |

**Method:** for every one of the 11 unique physical variables behind the 19
fields, every candidate daily table (`day`, `Eday`, `EdayZ`, `CFday`, `Oday`,
`SIday`) actually present on disk for each model was identified from prior
metadata scans (`artifacts/cmip6_daily_availability_audit/table_scan_raw.csv`
— global attributes, vertical-coordinate type, read directly from opened
files). Candidates whose vertical coordinate is **not** a genuine fixed
pressure/depth value (i.e. hybrid-sigma model levels) were excluded outright
— no vertical interpolation was performed to "fix" them. For every remaining
candidate, **every chunk filename on disk was parsed for its exact date
range** (`scripts/audit_cmip6_daily_19var_4model_final.py`, single grid_label
+ latest version directory only, matching the project's existing
`find_source_files` convention) and the file set was segmented into
genuinely continuous runs (gap tolerance 32 days). The reported range is the
**longest continuous run that overlaps** the water years for which WUS-D3
daily SWE exists (historical: WY1981–2014 / calendar 1980‑09→2014‑12; ssp370:
WY2015–2100 / calendar 2015‑01→2100‑08) — not simply "first file to last
file," which would silently hide interior gaps.

## 0. Two false-positive traps found and avoided

**(a) Hybrid-sigma levels ≠ fixed pressure levels (carried over from the
prior audit).** MPI-ESM1-2-HR's `CFday` table stores `ua`/`hus` on
dimensionless hybrid-sigma model levels (`ap + b·ps`), not a fixed `plev`
coordinate. Nominally it spans 1850–2014, but cannot supply "850 hPa" without
vertical interpolation — excluded per instructions. The genuine fixed-`plev`
table for MPI `ua` (`Eday`) covers a much smaller, sparse range (see table).

**(b) Many "daily" archives on this local mirror are sparse spot-check
holdings, not continuous records — this is new and more consequential than
(a).** Checking only the first and last file on disk (as an earlier pass in
this project did) makes archives like MIROC6/MPI's `Eday`/`EdayZ`/`SIday`/
`Oday` holdings for `ua`, `va`, `ta`, `zg`, `tos`, `siconc` look like they
span "1850–2014" when in fact they are **isolated 1–5-year chunks scattered
every 5–40 years**. Example — MPI's `EdayZ`/`zg` directory contains exactly
these files and nothing else:

```
zg_EdayZ_..._18500101-18541231.nc   zg_EdayZ_..._19000101-19041231.nc
zg_EdayZ_..._18550101-18591231.nc   zg_EdayZ_..._19750101-19791231.nc
zg_EdayZ_..._18600101-18641231.nc   zg_EdayZ_..._19800101-19841231.nc
zg_EdayZ_..._18850101-18891231.nc   zg_EdayZ_..._20000101-20041231.nc
                                      zg_EdayZ_..._20100101-20141231.nc
```

Only the **1975–1984** chunk (10 years, out of the 34 needed) is a genuinely
continuous run overlapping WY1981–2014. This pattern recurs for most
pressure-level variables at MIROC6 and MPI (see table) — **only `psl`,
`rlut`, and `hus_850` are genuinely fully continuous across the entire
needed window at MIROC6 and MPI**; everything else there is sparse. This
materially revises figures reported in the two prior audit documents in this
project for `zg`, `va`, `siconc` at MPI, which had been evaluated only by
first/last-file boundary and are corrected here.

## 1. The 19×4 matrix

`YES (sparse/partial)` = genuine daily data with the correct level exists,
but only as isolated chunks — the range shown is the single longest
chunk that overlaps the needed water years, not the full archive span.
Machine-readable version: [`docs/CMIP6_Daily_19Var_4Model_Availability_Audit.csv`](CMIP6_Daily_19Var_4Model_Availability_Audit.csv).

| Variable | EC-Earth3 | MIROC6 | MPI-ESM1-2-HR | TaiESM1 |
|---|---|---|---|---|
| `tos` | NO — no daily files present at all | YES — `Oday` — 1980–1999 (sparse) | NO — `Oday` chunks pre-1905 only; none overlap WY1981–2014 | NO — no daily `Oday` table |
| `siconc` | NO — no daily files at all | YES — `SIday` — 1980–1999 (sparse) | YES — `SIday` — 2000–2004 (sparse) | NO — no daily `SIday` table |
| `rlut` | NO — no daily files at all | YES — `day` — 1850–2014 hist / 2015–2100 ssp (fully continuous) | YES — `day` — 1850–2014 (fully continuous) | YES — `day` — 1850–2014 hist / 2015–2100 ssp (fully continuous) |
| `zg_500` | NO — `day/zg` dir empty | YES — `Eday` — 2000–2001 hist / 2025–2028 ssp (sparse); 500 hPa confirmed | YES — `EdayZ` — 1975–1984 (sparse); 500 hPa confirmed | YES — `day` — 1850–2014 / 2015–2100 (fully continuous); 500 hPa confirmed |
| `ta_850` | NO — `day/ta` dir empty | YES — `Eday` — 2003–2004 hist / 2073–2078 ssp (sparse); 850 hPa confirmed | YES — `Eday` — 1985–1999 (sparse); 850 hPa confirmed | YES — `day` — full continuous; 850 hPa confirmed |
| `ua_850` | NO — `day/ua` dir empty | YES — `Eday` — 1985–1987 hist / 2051–2057 ssp (sparse); 850 hPa confirmed | YES — `day` — 1980–1984 (sparse); 850 hPa confirmed (`CFday` reaches 2014 but is hybrid-sigma — rejected) | YES — `day` — full continuous; 850 hPa confirmed |
| `va_850` | NO — `day/va` dir empty | YES — `Eday` — 1998–1999 hist / 2078–2082 ssp (sparse); 850 hPa confirmed | YES — `EdayZ` — 1980–1984 (sparse); 850 hPa confirmed | YES — `day` — full continuous; 850 hPa confirmed |
| `ua_200` | NO — `day/ua` dir empty | YES — `Eday` — 1985–1987 (sparse); 200 hPa confirmed | YES — `Eday` — 1985–1989 (sparse); 200 hPa confirmed (`day`/plev8 lacks 200 hPa) | NO — daily only on plev8 `[1000,850,700,500,250,100,50,10]` hPa — **no 200 hPa at any coverage** |
| `va_200` | NO — `day/va` dir empty | YES — `Eday` — 1998–1999 (sparse); 200 hPa confirmed | YES — `EdayZ` — 1980–1984 (sparse); 200 hPa confirmed | NO — same plev8 gap, no 200 hPa |
| `hus_850` | NO — no `day/hus` dir (only unrelated `huss`) | YES — `Eday` — 1850–2014 hist / 2015–2100 ssp (fully continuous); 850 hPa confirmed | YES — `EdayZ` — 1850–2014 (fully continuous); 850 hPa confirmed | YES — `day` — full continuous; 850 hPa confirmed |
| `psl` | YES — `day` — 1970–2014 hist / 2015–2100 ssp (fully continuous) | YES — `day` — 1850–2014 / 2015–2100 (fully continuous) | YES — `day` — 1850–2014 (fully continuous) | YES — `day` — 1850–2014 / 2015–2100 (fully continuous) |
| `tas` | NO — `day/tas` dir empty | NO — historical chunks (1890–99, 1940–49) entirely pre-1980; only ssp370 (2025–2064) overlaps anything | YES — `day` — 1980–1984 (sparse) | YES — `day` — full continuous |
| `zg_50` | NO — `day/zg` dir empty | YES — `Eday` — 2000–2001 / 2025–2028 (sparse); 50 hPa confirmed | YES — `EdayZ` — 1975–1984 (sparse); 50 hPa confirmed | YES — `day` — full continuous; 50 hPa confirmed |
| `ta_50` | NO — `day/ta` dir empty | YES — `Eday` — 2003–2004 / 2073–2078 (sparse); 50 hPa confirmed | YES — `Eday` — 1985–1999 (sparse); 50 hPa confirmed | YES — `day` — full continuous; 50 hPa confirmed |
| `ua_50` | NO — `day/ua` dir empty | YES — `Eday` — 1985–1987 / 2051–2057 (sparse); 50 hPa confirmed | YES — `day` — 1980–1984 (sparse); 50 hPa confirmed | YES — `day` — full continuous; 50 hPa confirmed |
| `va_50` | NO — `day/va` dir empty | YES — `Eday` — 1998–1999 / 2078–2082 (sparse); 50 hPa confirmed | YES — `EdayZ` — 1980–1984 (sparse); 50 hPa confirmed | YES — `day` — full continuous; 50 hPa confirmed |
| `mrso` | NO — no daily table at all | YES — `day` — 2000–2009 hist / 2065–2084 ssp (sparse) | NO — chunks (1875–1979) entirely pre-1980, no overlap | NO — no daily table at all |
| `thetao_50m` | NO — no daily `thetao` product at any table | NO — same | NO — same | NO — same |
| `thetao_100m` | NO — no daily `thetao` product at any table | NO — same | NO — same | NO — same |

## 2. Derived summary

`V_all4` and the per-model sets below use the table's YES/NO exactly as
written — i.e. a variable counts as available if genuine daily data with the
correct level exists and overlaps the needed water years **at all**,
including the sparse/partial cases (per the task's own definition, item 5:
"usable temporal coverage overlapping the simulation years"). Where a
variable is only sparse, that is preserved from Section 1, not hidden here.

```
V_all4 = {psl}                                              (1 variable)

V_EC     = {psl}                                             (1)
V_MIROC  = {hus_850, mrso, psl, rlut, siconc, ta_50, ta_850,
            tos, ua_200, ua_50, ua_850, va_200, va_50, va_850,
            zg_50, zg_500}                                   (16)
V_MPI    = {hus_850, psl, rlut, siconc, ta_50, ta_850, tas,
            ua_200, ua_50, ua_850, va_200, va_50, va_850,
            zg_50, zg_500}                                   (15)
V_Tai    = {hus_850, psl, rlut, ta_50, ta_850, tas, ua_50,
            ua_850, va_50, va_850, zg_50, zg_500}             (12)
```

**Number of daily-available variables per model:** EC-Earth3 **1**, MIROC6
**16**, MPI-ESM1-2-HR **15**, TaiESM1 **12**.

**Pairwise intersections:**

```
EC-Earth3 ∩ MIROC6          = {psl}                                      (1)
EC-Earth3 ∩ MPI-ESM1-2-HR   = {psl}                                      (1)
EC-Earth3 ∩ TaiESM1         = {psl}                                      (1)
MIROC6 ∩ MPI-ESM1-2-HR      = {hus_850, psl, rlut, siconc, ta_50, ta_850,
                                ua_200, ua_50, ua_850, va_200, va_50,
                                va_850, zg_50, zg_500}                    (14)
MIROC6 ∩ TaiESM1            = {hus_850, psl, rlut, ta_50, ta_850, ua_50,
                                ua_850, va_50, va_850, zg_50, zg_500}     (11)
MPI-ESM1-2-HR ∩ TaiESM1     = {hus_850, psl, rlut, ta_50, ta_850, tas,
                                ua_50, ua_850, va_50, va_850, zg_50,
                                zg_500}                                   (12)
```

**Important caveat for whoever reads these counts next:** most of the
entries contributing to the MIROC6/MPI-ESM1-2-HR numbers above are the
**sparse** kind (isolated few-year chunks), not continuous archives. The
subset that is genuinely continuous across the *entire* needed window at
every model that has it at all is much smaller: `{psl}` (all 4 models),
plus `{rlut, hus_850}` fully continuous at MIROC6, MPI-ESM1-2-HR, and
TaiESM1 (not EC-Earth3), plus TaiESM1's own additional 9 fully-continuous
fields (`zg_500, zg_50, ta_850, ta_50, ua_850, ua_50, va_850, va_50, tas`).
No decision about which years/variables to actually use is made here, per
instructions — this distinction is reported so that decision can be made
correctly.

## Output files

- `docs/CMIP6_Daily_19Var_4Model_Availability_Audit.md` (this file)
- `docs/CMIP6_Daily_19Var_4Model_Availability_Audit.csv`
- `artifacts/cmip6_daily_availability_audit/final_19x4_longest_continuous.json` (raw per-variable/model/experiment segment data)
- Scripts: `scripts/audit_cmip6_daily_19var_4model_final.py`, `scripts/build_19x4_final_matrix.py`
