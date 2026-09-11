#!/usr/bin/env python3
"""Assemble the final 19-variable x 4-model daily-availability matrix from
the verified findings of this audit (all facts sourced from netCDF metadata
opened directly earlier in this session -- see docs/CMIP6_Daily_19Var_4Model_Availability_Audit.md
for the narrative). Writes the required markdown table + CSV.
"""
import csv
from pathlib import Path

OUT_MD_TABLE = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cmip6_daily_availability_audit/final_19x4_table_fragment.md")
OUT_CSV = Path("/global/u1/h/hyvchen/Snow-Predication-at-Sierra/docs/CMIP6_Daily_19Var_4Model_Availability_Audit.csv")

MODELS = ["EC-Earth3", "MIROC6", "MPI-ESM1-2-HR", "TaiESM1"]

# cell = (available: bool, table, hist_start, hist_end, ssp_start, ssp_end, level_check, notes)
# "sparse" flags mean: genuine daily+level exists, but only as isolated
# multi-year chunks, not a continuous archive; start/end given are the single
# best chunk that overlaps the WY1981-2014 / WY2015-2100 windows.
DATA = {
"tos": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "no daily files present at all (dir absent)"),
 "MIROC6": (True, "Oday", "1980", "1999", None, None, "n/a (surface)", "sparse: 3 chunks total (1930-39,1980-99,...); only 1980-1999 overlaps needed window"),
 "MPI-ESM1-2-HR": (False, "Oday", None, None, None, None, "n/a", "sparse, chunks pre-1905 only (e.g. 1895-1904); none overlap WY1981-2014"),
 "TaiESM1": (False, "-", None, None, None, None, "n/a", "no daily Oday table exists"),
},
"siconc": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "no daily files present at all"),
 "MIROC6": (True, "SIday", "1980", "1999", None, None, "n/a (surface)", "sparse: 5 fragments total; only 1980-1999 chunk overlaps"),
 "MPI-ESM1-2-HR": (True, "SIday", "2000", "2004", None, None, "n/a (surface)", "sparse: 4 fragments (1850-54,1895-99,2000-04,2010-14); longest single overlapping chunk is 2000-2004 (2010-2014 is a separate, shorter, later chunk)"),
 "TaiESM1": (False, "-", None, None, None, None, "n/a", "no daily SIday table exists"),
},
"rlut": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "no daily files present at all"),
 "MIROC6": (True, "day", "1850", "2014", "2015", "2100", "n/a (TOA)", "fully continuous both branches, no gaps"),
 "MPI-ESM1-2-HR": (True, "day", "1850", "2014", None, None, "n/a (TOA)", "fully continuous historical, no gaps; no local ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "n/a (TOA)", "fully continuous both branches, no gaps"),
},
"zg_500": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/zg directory exists but is empty (0 files)"),
 "MIROC6": (True, "Eday", "2000", "2001", "2025", "2028", "500 hPa confirmed (plev19)", "sparse: 27 fragments hist / 19 ssp; only these small chunks overlap"),
 "MPI-ESM1-2-HR": (True, "EdayZ", "1975", "1984", None, None, "500 hPa confirmed (plev19)", "sparse: 6 fragments; 1975-1984 is the longest overlapping chunk; no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "500 hPa confirmed (plev8)", "fully continuous both branches"),
},
"zg_50": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/zg directory exists but is empty"),
 "MIROC6": (True, "Eday", "2000", "2001", "2025", "2028", "50 hPa confirmed (plev19)", "same sparse Eday file set as zg_500"),
 "MPI-ESM1-2-HR": (True, "EdayZ", "1975", "1984", None, None, "50 hPa confirmed (plev19)", "same sparse EdayZ file set as zg_500"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "50 hPa confirmed (plev8)", "fully continuous both branches"),
},
"ta_850": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/ta directory exists but is empty"),
 "MIROC6": (True, "Eday", "2003", "2004", "2073", "2078", "850 hPa confirmed (plev19)", "sparse: 24 fragments hist / 22 ssp; tiny overlapping chunks"),
 "MPI-ESM1-2-HR": (True, "Eday", "1985", "1999", None, None, "850 hPa confirmed (plev19)", "sparse: 2 fragments; 1985-1999 is the overlapping one; no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "850 hPa confirmed (plev8)", "fully continuous both branches"),
},
"ta_50": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/ta directory exists but is empty"),
 "MIROC6": (True, "Eday", "2003", "2004", "2073", "2078", "50 hPa confirmed (plev19)", "same sparse Eday file set as ta_850"),
 "MPI-ESM1-2-HR": (True, "Eday", "1985", "1999", None, None, "50 hPa confirmed (plev19)", "same sparse Eday file set as ta_850"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "50 hPa confirmed (plev8)", "fully continuous both branches"),
},
"ua_850": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/ua directory exists but is empty"),
 "MIROC6": (True, "Eday", "1985", "1987", "2051", "2057", "850 hPa confirmed (plev19)", "sparse: 28 fragments hist / 18 ssp; tiny overlapping chunks"),
 "MPI-ESM1-2-HR": (True, "day", "1980", "1984", None, None, "850 hPa confirmed (plev8)", "sparse: 3 fragments; CFday reaches 1850-2014 but is HYBRID-SIGMA (rejected, see Sec.0); no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "850 hPa confirmed (plev8)", "fully continuous both branches"),
},
"ua_50": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/ua directory exists but is empty"),
 "MIROC6": (True, "Eday", "1985", "1987", "2051", "2057", "50 hPa confirmed (plev19)", "same sparse Eday file set as ua_850"),
 "MPI-ESM1-2-HR": (True, "day", "1980", "1984", None, None, "50 hPa confirmed (plev8)", "same sparse day file set as ua_850; CFday hybrid-sigma rejected"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "50 hPa confirmed (plev8)", "fully continuous both branches"),
},
"ua_200": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/ua directory exists but is empty"),
 "MIROC6": (True, "Eday", "1985", "1987", None, None, "200 hPa confirmed (plev19)", "sparse; ssp370 200hPa overlap not confirmed usable (plev19 Eday not present for ssp370, only day/plev8)"),
 "MPI-ESM1-2-HR": (True, "Eday", "1985", "1989", None, None, "200 hPa confirmed (plev19)", "sparse: day table (1980-84 chunk) lacks 200 hPa (plev8); Eday's only overlapping chunk is 1985-1989; CFday hybrid-sigma rejected"),
 "TaiESM1": (False, "day", None, None, None, None, "200 hPa ABSENT from plev8", "only table available is day/plev8=[1000,850,700,500,250,100,50,10]hPa - no 200hPa at any coverage"),
},
"va_850": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/va directory exists but is empty"),
 "MIROC6": (True, "Eday", "1998", "1999", "2078", "2082", "850 hPa confirmed (plev19)", "sparse: 24 fragments hist / 22 ssp"),
 "MPI-ESM1-2-HR": (True, "EdayZ", "1980", "1984", None, None, "850 hPa confirmed (plev19)", "sparse: 5 fragments; no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "850 hPa confirmed (plev8)", "fully continuous both branches"),
},
"va_50": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/va directory exists but is empty"),
 "MIROC6": (True, "Eday", "1998", "1999", "2078", "2082", "50 hPa confirmed (plev19)", "same sparse Eday file set as va_850"),
 "MPI-ESM1-2-HR": (True, "EdayZ", "1980", "1984", None, None, "50 hPa confirmed (plev19)", "same sparse EdayZ file set as va_850"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "50 hPa confirmed (plev8)", "fully continuous both branches"),
},
"va_200": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/va directory exists but is empty"),
 "MIROC6": (True, "Eday", "1998", "1999", None, None, "200 hPa confirmed (plev19)", "sparse; usable ssp370 200hPa chunk not confirmed"),
 "MPI-ESM1-2-HR": (True, "EdayZ", "1980", "1984", None, None, "200 hPa confirmed (plev19)", "same sparse EdayZ file set as va_850; no ssp370 branch"),
 "TaiESM1": (False, "day", None, None, None, None, "200 hPa ABSENT from plev8", "only table is day/plev8, no 200hPa at any coverage"),
},
"hus_850": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/hus directory does not exist (only huss, unrelated variable)"),
 "MIROC6": (True, "Eday", "1850", "2014", "2015", "2100", "850 hPa confirmed (plev19)", "fully continuous both branches, no gaps"),
 "MPI-ESM1-2-HR": (True, "EdayZ", "1850", "2014", None, None, "850 hPa confirmed (plev19)", "fully continuous historical, no gaps; no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "850 hPa confirmed (plev8)", "fully continuous both branches"),
},
"psl": {
 "EC-Earth3": (True, "day", "1970", "2014", "2015", "2100", "n/a (surface)", "fully continuous both branches (only variable EC-Earth3 has any real daily data for)"),
 "MIROC6": (True, "day", "1850", "2014", "2015", "2100", "n/a (surface)", "fully continuous both branches"),
 "MPI-ESM1-2-HR": (True, "day", "1850", "2014", None, None, "n/a (surface)", "fully continuous historical; no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "n/a (surface)", "fully continuous both branches"),
},
"tas": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "day/tas directory exists but is empty"),
 "MIROC6": (False, "day", None, None, "2025", "2064", "n/a", "historical chunks (1890-99,1940-49) are entirely pre-1980, no overlap with needed window; ssp370 2025-2064 chunk does overlap"),
 "MPI-ESM1-2-HR": (True, "day", "1980", "1984", None, None, "n/a (surface)", "sparse: 4 fragments; 1980-1984 overlaps; no ssp370 branch"),
 "TaiESM1": (True, "day", "1850", "2014", "2015", "2100", "n/a (surface)", "fully continuous both branches"),
},
"mrso": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "no daily mrso table at all"),
 "MIROC6": (True, "day", "2000", "2009", "2065", "2084", "n/a (column)", "sparse: 3 fragments hist / 2 ssp; only these overlap"),
 "MPI-ESM1-2-HR": (False, "day", None, None, None, None, "n/a", "chunks (1875-79...1940-44) entirely pre-1980, no overlap with WY1981-2014"),
 "TaiESM1": (False, "-", None, None, None, None, "n/a", "no daily mrso table at all"),
},
"thetao_50m": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
 "MIROC6": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
 "MPI-ESM1-2-HR": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
 "TaiESM1": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
},
"thetao_100m": {
 "EC-Earth3": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
 "MIROC6": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
 "MPI-ESM1-2-HR": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
 "TaiESM1": (False, "-", None, None, None, None, "n/a", "no daily thetao product exists at any table"),
},
}

VARORDER = ["tos","siconc","rlut","zg_500","ta_850","ua_850","va_850","ua_200","va_200",
            "hus_850","psl","tas","zg_50","ta_50","ua_50","va_50","mrso","thetao_50m","thetao_100m"]

def cell_text(v):
    ok, table, hs, he, ss, se, level, notes = v
    if not ok:
        if table != "-":
            return f"NO — {table} — no usable overlap with WY1981-2014/2015-2100 ({notes})"
        return f"NO — {notes}"
    branch = f"{hs}-{he}" if hs else "no historical"
    if ss:
        branch += f" hist / {ss}-{se} ssp"
    sparse_flag = " (sparse/partial)" if "sparse" in notes or "chunk" in notes else ""
    return f"YES — {table} — {branch}{sparse_flag}"

# Markdown table
lines = ["| Variable | " + " | ".join(MODELS) + " |", "|---|" + "|".join(["---"]*len(MODELS)) + "|"]
for var in VARORDER:
    row = [var] + [cell_text(DATA[var][m]) for m in MODELS]
    lines.append("| " + " | ".join(row) + " |")
OUT_MD_TABLE.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# CSV
with OUT_CSV.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["variable","model","daily_available","table","historical_start","historical_end","scenario","scenario_start","scenario_end","level_or_depth_check","notes"])
    for var in VARORDER:
        for m in MODELS:
            ok, table, hs, he, ss, se, level, notes = DATA[var][m]
            writer.writerow([var, m, "YES" if ok else "NO", table, hs or "", he or "",
                              "ssp370" if ss else "", ss or "", se or "", level, notes])
print(f"\nwrote {OUT_CSV}")
print(f"wrote {OUT_MD_TABLE}")
