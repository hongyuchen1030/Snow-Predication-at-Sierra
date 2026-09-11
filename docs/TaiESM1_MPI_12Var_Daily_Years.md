# TaiESM1 / MPI-ESM1-2-HR — All Available Daily Years, 12 Selected Predictors

**Status: SIMPLE AVAILABILITY LISTING ONLY.** No intersections across
variables were computed, no requirement that different variables share the
same years, no continuity requirement, no sparse/isolated years discarded,
no dataset design, no training-window counting, no recommendation, no new
variable added. This lists, independently per variable, **every** calendar
year for which genuinely daily data (fixed pressure level, not hybrid-sigma)
exists on disk — not just the longest run, not just years overlapping any
particular target window.

**Method:** for each variable, the same already-verified genuine
fixed-pressure-level table is used (rejecting hybrid-sigma products exactly
as before — MPI's `ua`/`hus` `CFday` table remains excluded). Every chunk
filename actually on disk (single `grid_label`, latest version directory
only, matching the project's existing `find_source_files` convention) was
parsed for its exact start/end date; consecutive chunks separated by ≤32
days were merged into one range, and every resulting disconnected range is
listed. Source: [`scripts/build_taiesm1_mpi_12var_years.py`](../scripts/build_taiesm1_mpi_12var_years.py), reusing metadata already gathered in the prior daily-availability audits.

Machine-readable version: [`docs/TaiESM1_MPI_12Var_Daily_Years.csv`](TaiESM1_MPI_12Var_Daily_Years.csv).

| Model | Variable | Available years |
|-------|----------|-----------------|
| TaiESM1 | rlut | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | zg_500 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | zg_50 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | ta_850 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | ta_50 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | ua_850 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | ua_50 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | va_850 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | va_50 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | hus_850 | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | psl | historical: 1850–2014; ssp370: 2015–2100 |
| TaiESM1 | tas | historical: 1850–2014; ssp370: 2015–2100 |
| MPI-ESM1-2-HR | rlut | historical: 1850–2014 |
| MPI-ESM1-2-HR | zg_500 | historical: 1850–1864, 1885–1889, 1900–1904, 1975–1984, 2000–2004, 2010–2014 |
| MPI-ESM1-2-HR | zg_50 | historical: 1850–1864, 1885–1889, 1900–1904, 1975–1984, 2000–2004, 2010–2014 |
| MPI-ESM1-2-HR | ta_850 | historical: 1985–1999, 2005–2009 |
| MPI-ESM1-2-HR | ta_50 | historical: 1985–1999, 2005–2009 |
| MPI-ESM1-2-HR | ua_850 | historical: 1910–1914, 1950–1954, 1980–1984 |
| MPI-ESM1-2-HR | ua_50 | historical: 1910–1914, 1950–1954, 1980–1984 |
| MPI-ESM1-2-HR | va_850 | historical: 1865–1869, 1890–1894, 1920–1924, 1980–1984, 2010–2014 |
| MPI-ESM1-2-HR | va_50 | historical: 1865–1869, 1890–1894, 1920–1924, 1980–1984, 2010–2014 |
| MPI-ESM1-2-HR | hus_850 | historical: 1850–2014 |
| MPI-ESM1-2-HR | psl | historical: 1850–2014 |
| MPI-ESM1-2-HR | tas | historical: 1865–1869, 1945–1959, 1980–1984, 2005–2009 |

MPI-ESM1-2-HR has no local ssp370 archive at all (already established in
prior audits), so every MPI row is historical-only.

## Output files

- `docs/TaiESM1_MPI_12Var_Daily_Years.md` (this file)
- `docs/TaiESM1_MPI_12Var_Daily_Years.csv`
