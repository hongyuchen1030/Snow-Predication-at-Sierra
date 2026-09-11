# Snow-17 Methodology & Readiness Record

**Purpose.** This document has two jobs:

1. It is the single authoritative record of every Snow-17 methodological
   decision resolved for this project's observational/reanalysis physics
   baseline (unchanged role from earlier versions of this document).
2. **It is the configuration reference for any Sierra P/T → SWE translation
   experiment run with Snow-17** — a 1-day forecast, a 3-day forecast, a
   7-day forecast, any other short lead time, or a different weather
   generator entirely (ACE2, CMIP6, or something else). This document alone,
   without any chat history, should be enough to correctly configure,
   initialize, force, and interpret Snow-17 for any such experiment.

It replaces the old requirements table in
[`docs/intern/permultter_snow17.md`](intern/permultter_snow17.md) (lines
291-316) as the current-methodology reference for that scope. The old file is
left in place, unmodified, as historical intern-onboarding material — see
[Provenance of this document](#provenance-of-this-document) at the end for
exactly what it still is and is not.

Last consolidated: 2026-09-11 (restructured around the A/B/C/D distinction
below; Stage 1 calibration/validation results folded in from
`docs/Snow17_Stage1_SCEUA_Validation_Report.md`).

---

## 0. Read this first — the one distinction that matters most

> **Forecast lead time does not determine Snow-17 configuration.**

Snow-17 is a stateful physical translator:

$$
P(t),\ T(t),\ S(t_0) \;\xrightarrow{\text{Snow-17}}\; SWE(t)
$$

where $S(t_0)$ is the complete Snow-17 internal state at the start of your
forecast window. Whether your atmospheric experiment predicts **1 day, 3
days, 7 days, 14 days,** or anything else does **not**, by itself, change:

- Snow-17 parameter definitions
- fixed parameter values
- calibrated regional parameter vectors
- elevation correction
- terrain representation
- regional (North/Central/South) classification
- Snow-17 state variables
- units
- spatial aggregation rules

**What changes between experiments is:** the forecast initialization date
$t_0$, and the length/source of the generated P/T forcing supplied *after*
$t_0$. Everything else below in **Part A** is invariant across every Sierra
Snow-17 experiment this project runs, regardless of lead time.

### How this document is organized

| Part | Content | Changes with forecast lead time? |
|---|---|---|
| **A** | Invariant Sierra Snow-17 configuration | No |
| **B** | Forecast-start state-initialization procedure | Only the start date $t_0$ moves; the procedure itself does not |
| **C** | Experiment-specific atmospheric forcing choices | Yes — this is where lead time / weather-generator choice actually lives |
| **D** | Hongyu's Stage-1 calibration/validation experiment (a specific, completed instance of the above) | N/A — historical record |

Throughout, statements are tagged with one of:

- **[UNIVERSAL]** — true of the vendored `tonic` Snow-17 implementation itself, independent of this project
- **[SIERRA-STANDARD]** — a project-wide decision that applies to every Sierra Snow-17 experiment (this project's own standard, not upstream `tonic` behavior)
- **[STAGE-1]** — specific to Hongyu's WY1985-2021 ERA5-Land validation experiment (Part D); not necessarily applicable to your experiment's calibration/data split
- **[ACE2-SPECIFIC]** — specific to ACE2-generated forcing; not applicable if you are using ERA5-Land or another source

### Which workflow do I need?

Most Sierra P/T→SWE experiments should **not** recalibrate Snow-17. Two
workflows:

**Workflow A — use the validated frozen translator (default; use this unless you have a specific reason not to):**
```
1. Prepare your P/T forcing (Part C)
2. Establish the Snow-17 state at your forecast start date t0 (Part B)
3. Apply the frozen theta_N / theta_C / theta_S parameter vectors (§A.2, §D.1)
4. Run Snow-17 forward through your generated forecast P/T
5. Aggregate/report SWE (§A.4/§A.5)
```

**Workflow B — recalibrate Snow-17 (only for an explicit Snow-17 methodology experiment):**
```
Follow the calibration design in Part D (§D.2-D.4): SCE-UA, calibration
data, 1-NSE objective, calibration/held-out split, leakage rules.
```
If you are not deliberately testing a new calibration methodology, you want
**Workflow A**.

---

# PART A — Invariant Sierra Snow-17 Configuration

Everything in this part is the same regardless of your forecast lead time or
which weather-generation experiment you are running.

## A.1 Implementation **[UNIVERSAL / SIERRA-STANDARD]**

**Final choice:** [UW-Hydro `tonic`](https://github.com/UW-Hydro/tonic/blob/master/tonic/models/snow17/snow17.py)
`snow17()`, vendored locally at:
- `scripts/vendor/snow17.py` — **use this one for production runs.**
- `scripts/vendor/snow17_instrumented.py` — additive-only copy that also
  returns the five internal state trajectories per timestep (upstream only
  returns `model_swe`/`outflow`); use for diagnostics only.

**Rationale:** established, citable, minimal implementation whose exposed
parameter set was cross-checked directly against its source (not assumed).
It exposes exactly:

```
scf, rvs, uadj, mbase, mfmax, mfmin, tipm, nmf, plwhc, pxtemp, pxtemp1, pxtemp2
```

(12 parameters total — see the full table in §A.2) and has **no** `DAYGM`,
`SI`, or `ADC` argument anywhere in its signature or body (confirmed by
reading the full vendored source, not just the signature). This matches
Shiheng Duan's own email: *"DAYGM is omitted and I didn't recognize ADC or
SI."*

**Upstream bug found and fixed, in both vendored copies:** the upstream
ripeness branch read
```python
elif ((qw >= deficit) and
      ait((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx))):
```
`ait` is a float at that point in the loop; calling it as a function raises
`TypeError` whenever this branch is reached. Fixed to a plain boolean AND
(matching the surrounding NWS SNOW-17 ripeness logic, Anderson 1973):
```python
elif (qw >= deficit) and ((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx)):
```
No other physics line changed. **Both `scripts/vendor/snow17.py` and
`scripts/vendor/snow17_instrumented.py` now have this fix** — confirmed
identical at the relevant line, and exercised successfully across all 694
Sierra cells in the §D.1 smoke test with no errors. **If you vendor your own
copy of `tonic` Snow-17 from scratch, check for and fix this bug — do not
assume a fresh download is already patched.**

## A.2 The 12 Snow-17 parameters **[UNIVERSAL definitions / SIERRA-STANDARD values]**

Snow-17 (this `tonic` implementation) takes exactly 12 named parameters. 7
are calibrated per Sierra region by this project; 5 are fixed. **The
definitions of all 12 are invariant** — what differs project-to-project is
only which values you plug in.

**Calibrated (7), Shiheng Duan (2024) Appendix B, Table B1 ranges** — source
verified directly from the local PDF:
`docs/papers/Water Resources Research - 2024 - Duan - Using Temporal Deep
Learning Models to Estimate Daily Snow Water Equivalent Over.pdf`. Full
citation: Duan, S., Ullrich, P., Risser, M., & Rhoades, A. (2024). Using
temporal deep learning models to estimate daily snow water equivalent over
the Rocky Mountains. *Water Resources Research*, 60, e2023WR035009.
https://doi.org/10.1029/2023WR035009

| Parameter | Description | Unit | Calibration bounds (Shiheng Appendix B) | Frozen Stage-1 value (§D.1) — North / Central / South |
|---|---|---|---|---|
| SCF | Gage catch deficit multiplying factor | — | 0.9 – 1.2 | 1.091 / 1.128 / **1.200 (bound)** |
| MFMAX | Maximum melt factor, non-rain periods | mm·°C⁻¹·6hr⁻¹* | 0.5 – 1.3 | 0.757 / **1.296 (bound)** / 1.026 |
| MFMIN | Minimum melt factor, non-rain periods | mm·°C⁻¹·6hr⁻¹* | 0.1 – 0.6 | 0.249 / **0.600 (bound)** / **0.600 (bound)** |
| UADJ | Average wind function, rain-on-snow periods | mm·mb⁻¹ | 0.05 – 0.2 | **0.0500 (bound)** / **0.0503 (bound)** / **0.0500 (bound)** |
| PXTEMP | Temperature separating rain/snow for the energy-budget rain term | °C | 0.0 – 2.0 | 0.255 / 1.098 / 1.176 |
| PXTEMP1 | Lower limit temp, snow/transition boundary | °C | −2.0 – 0.0 | **-1.999 (bound)** / **-1.997 (bound)** / -0.404 |
| PXTEMP2 | Upper limit temp, transition/rain boundary | °C | 0.0 – 4.0 | 3.453 / **3.994 (bound)** / **3.982 (bound)** |

\* See §A.7 — the `6hr` in the textbook unit is a normalization baked into
`melt_function`'s own `dt/6` scaling, **not** a requirement that your
timestep be 6 hours. `MFMAX`/`MFMIN` values in the table above are used
correctly regardless of your chosen `dt`.

**Distinct roles (paper's own text, Appendix B):** *"The snow-rain partition
uses a linear transition scheme, which involves PXTEMP1 and PXTEMP2, while
PXTEMP is only used for the rain temperature for the energy budget."*
PXTEMP1/PXTEMP2 are not redundant with PXTEMP.

**Fixed (5)**, verified directly from the vendored `tonic` function signature
(`scripts/vendor/snow17.py`, matching upstream commit `b862801`, Feb 25 2015):

| Parameter | Value | Basis |
|---|---|---|
| RVS | 1.0 | Rain-vs-snow option 1 = linear transition between PXTEMP1/PXTEMP2 |
| MBASE | 1.0 | Base temp above which melt occurs, °C |
| TIPM | 0.1 | Docstring cites NWS Anderson Manual recommended range 0.1-0.2 for deep snowpack |
| NMF | 0.15 | **Docstring bug**: text says "default 0.04" (copy-paste error from the adjacent `plwhc` docstring); the **operative** default, from the function signature itself, is 0.15 |
| PLWHC | 0.04 | Liquid water holding capacity fraction |

Shiheng did **not** document these five anywhere — his Appendix B lists only
the seven he calibrated, and no supplementary code (Zenodo 7818315,
`github.com/ShihengDuan/code-SWE`) contains a SNOW-17 parameter file
(confirmed by inspection). This project fixes them to the documented `tonic`
defaults as the only citable, non-fabricated option — **not** a claim that
Shiheng used exactly these values.

**Using the frozen Stage-1 values (Workflow A):** load
`theta_N.json` / `theta_C.json` / `theta_S.json` directly — do not retype
the numbers above. See §D.1 for exact paths and the full JSON schema
(includes optimizer settings, seed, git commit, convergence diagnostics).
**Using the calibration bounds (Workflow B):** the bounds column above is
what a fresh SCE-UA run should search over — see §D.2-D.3 for the exact
optimizer configuration used.

## A.3 Snow-17 state and the restart-state fact **[UNIVERSAL — verified from source]**

**The internal states that determine continuation** (verified from the
vendored source, not assumed):

| State | Meaning |
|---|---|
| `ait` | Antecedent temperature index |
| `w_qx` | Liquid water holding capacity |
| `w_q` | Liquid water currently held in the snowpack |
| `w_i` | Ice-portion SWE |
| `deficit` | Heat deficit (negative heat storage) |

`model_swe = w_i + w_q` at every timestep — **but the reverse is not true**:

> $\boxed{\text{SWE alone is not the complete Snow-17 state.}}$
>
> Two snowpacks can have identical SWE but different `deficit`/`w_q`/`ait`
> values (e.g. a cold, dry snowpack vs. one that just had a rain-on-snow
> event) and will melt completely differently going forward. You cannot
> reconstruct the full state from a single SWE number.

**Critical fact, verified directly by reading `snow17()` line-by-line
(both vendored copies):** the function hardcodes all five states to `0.0` at
its own top (lines 116-126 of `scripts/vendor/snow17.py`) on *every call*,
and **only returns `model_swe`/`outflow` arrays for the whole input series**
— it does not accept an initial-state argument, and it does not return the
final state. `snow17_instrumented.py` is identical in this respect: it
*additionally returns* the five state trajectories for diagnostic viewing,
but it **still initializes them to zero internally and does not accept them
as input**, either.

**Consequence — this settles the state-continuation question:**

> $\boxed{\text{No restart-state API exists in either vendored copy.}}$
>
> You **cannot** serialize Snow-17 state at some date, save it, and later
> inject it into a fresh call. The only currently-supported way to continue
> Snow-17 across a forecast start date is:
>
> **B (mandatory with this implementation): run one continuous Snow-17
> trajectory** that contains your historical/initialization period
> immediately followed by your generated forecast period, in a single call
> to `snow17()`, over a single unbroken `time`/`prec`/`tair` array. Do not
> reset state at the forecast start date — because there is no "reset,"
> there is only "one call from zero, or a different call from zero." The
> only way to arrive at forecast time with nonzero state is to have run
> Snow-17 continuously from a zero-state start *through* that date within
> the same call.

If you need true checkpoint/restart (e.g. to avoid re-running a long
spin-up for every forecast start date), that requires modifying the
vendored function to accept/return state explicitly — **not currently
implemented anywhere in this repo**. Until that exists, Option B (one
continuous run) is what every script in this project actually does (see
§A.5, §D.1's `run_region_continuous`/`run_region_snow17`).

## A.4 Regional parameter assignment **[SIERRA-STANDARD]**

Every Snow-17 grid cell gets exactly one of three shared 7-parameter
vectors — `theta_N`, `theta_C`, or `theta_S` — never a per-cell fit. The
region a cell belongs to is a **fixed geographic classification**,
independent of forecast lead time, forecast start date, or which weather
model is driving the forcing.

**Finalized mask — use this copy:**
`docs/snow17_reference_data/regional_mask/basin_assignment_grid_wy2021_knn_filled.npz`
/ `.nc`, plus `before_after_knn_fill.png`, `crowley_lake_verification.png`,
`fill_summary.json`, `provenance.md` in the same folder. (Perlmutter home
directories are not externally readable, and only `docs/` and `scripts/` are
synced to GitHub, so this full-size copy lives under `docs/` rather than
`artifacts/` specifically so it travels with the repo. Canonical/original
location, same content: `artifacts/sierra_basin_assignment_knn_filled/`.)

**Original geographic definition (CDEC/DWR basin grouping):** North =
Trinity through Feather & Truckee; Central = Yuba & Tahoe through Merced &
Walker; South = San Joaquin & Mono through Kern.

**Provenance in brief** (full detail unchanged from earlier versions of this
document — see §A.4-detail below): USGS WBD HUC8 polygon keyword-matching
seeded 110,231 native cells (North=29,912, Central=46,572, South=33,747);
16,918 unassigned active-footprint cells were resolved by K=16
nearest-neighbor majority vote over *already-assigned* cells only, using
**geography alone, never any year's SWE values** — this separation is
deliberate and load-bearing: the mask does not change if you run a
different water year, a different forecast experiment, or a different
weather model. Aggregated onto the ERA5-Land 0.1° grid by majority vote:
North=293, Central=224, South=177, Outside=2,572 cells (47 mixed cells
resolved by majority vote, deterministic tie-break North<Central<South).

**Filename note:** despite the historical `..._wy2021...` filename, this
mask is a fixed geographic product, not a per-year one.

<details>
<summary>A.4-detail — KNN gap-fill categories (click to expand; unchanged from prior version of this document)</summary>

- **Category A (12,719 cells): genuinely outside the Sierra** — Great Basin
  Nevada, Modoc Plateau, Klamath Basin, San Joaquin Valley-floor HUC8s inside
  the generous rectangular SWE extraction box but not Sierra terrain. Left
  unassigned (NaN), not touched by the fill.
- **Category B (4,199 cells): genuinely inside the Sierra**, missed only by
  HUC8-name keyword matching — dominated by the Crowley Lake HUC8 (4,080
  cells, the Long Valley/Mono Basin area — the "Mono" component of the CDEC
  South Sierra definition), plus Fresno River (53), Upper Poso (30), Upper
  Deer-Upper White (30), Upper Bear (6). Filled by K=16 NN majority vote
  (Category-A cells never enter the neighbor pool): 0 North / 57 Central /
  4,142 South filled, 0 ties. Crowley Lake resolved 100% South.
- Seed assignments were never changed — verified bitwise-identical in the
  fill script's own checks.

</details>

| Status | RESOLVED / IMPLEMENTED / VERIFIED |
|---|---|

## A.5 Spatial working grid and aggregation **[SIERRA-STANDARD]**

**Authoritative grid for the currently validated Sierra translator: ERA5-Land
native 0.1°×0.1°.** Artifact:
`artifacts/snow17_era5land_working_grid/era5land_working_grid.npz` / `.nc`,
`summary.json`, `provenance.md`.

**Use this grid unless you have an explicit scientific reason to build and
validate a different Snow-17 discretization.** Do not casually upsample
coarse meteorological forcing to a finer Snow-17 grid — the project's own
earlier "coarsen-8" proposal was rejected for exactly this reason (~2.8×
finer than ERA5-Land, would duplicate one ERA5 value across ~5.5 neighboring
cells with no new information; see `artifacts/snow17_grid_alignment_audit/`
for the full superseded analysis). This exact grid is not mandatory if a
given experiment genuinely requires another one — but
$P$, $T$, terrain, region, and Snow-17 state must all refer to the *same*
physically consistent spatial unit; do not mix a forcing grid with a
different terrain/region grid without re-deriving the mapping.

**Regional SWE aggregation — occupancy/area-weighted, not equal-weighted:**
$$
SWE_r(t) = \frac{\sum_i w_{ir}\,SWE_i(t)}{\sum_i w_{ir}}
$$
where $w_{ir}$ is the count of native, region-$r$-assigned SWE pixels inside
ERA5-Land cell $i$ (the `occupancy` field in `era5land_working_grid.npz`).
**Do not** compute a flat/equal-weighted mean of ERA5-Land cell values for a
region — a WY2021 consistency check found this produces a **−28.6% to
−32.6%** low bias (North/Central/South) relative to the true native-pixel
mean, a mean-of-cell-means artifact, not a coding bug (documented in
`artifacts/snow17_era5land_working_grid/summary.json`).

**Whole-Sierra combination** (used in §D.1): the same equation extended
across regions, with $W_r$ = total occupancy weight per region (sum of
$w_{ir}$ over all of that region's cells) as the region-level weight.

| Status | RESOLVED / IMPLEMENTED / VERIFIED (§D.1 smoke test + §D.2 production calibration) |
|---|---|

## A.6 Elevation correction — generic rule, then the Sierra instantiation **[UNIVERSAL rule / SIERRA-STANDARD instantiation]**

**Generic rule — read this before plugging any forcing source into the
formula below:**
$$
\boxed{\Delta z = z_{\text{target terrain}} - z_{\text{elevation represented by the forcing}}}
$$
The correction only makes physical sense if $z_{\text{forcing}}$ is the
elevation that the *specific forcing product you are using* actually
represents on its own grid — **not automatically ERA5-Land's orography**.
If you substitute a different weather model's temperature into the formula
below while keeping `z_ERA5Land` in the denominator term, you will be
correcting against the wrong reference surface.

- **ERA5-Land forcing → use ERA5-Land's own orography** (this is what Stage
  1 does; see below).
- **ACE2 on an ERA5/ERA5-Land-compatible grid** → determine and document the
  elevation actually represented by *that* forcing grid before reusing this
  formula (see `HGTsfc` in `docs/ACE2_ERA5_WY2016_Capability_Audit.md`,
  which is delivered as part of ACE2's own forcing set — do not assume it
  equals ERA5-Land's grid elevation without checking). See also
  `docs/ACE2_Snow17_Scientific_Validity_Rule.md`, which independently
  requires a documented elevation-correction manifest before any production
  ACE2→Snow-17 run.
- **Another weather model** → determine its forcing-grid/reference orography
  first; do not skip this step.

**Sierra project's ERA5-Land instantiation of the rule (Stage 1, and the
default for any ERA5-Land-forced experiment):**
$$
T_{\rm corrected} = T_{\rm ERA5Land} + (-4.5\ {\rm K/km})\,\frac{z_{\rm DEM}-z_{\rm ERA5Land}}{1000}
$$
where $z_{\rm DEM}$ is the USGS NED 1/3 arc-second DEM (§A.6-terrain below)
and $z_{\rm ERA5Land}$ is ERA5-Land's own coarse orography. Sign verified
against Dutra et al. (2020) Eq. 1 (Γ=DT/DZ, negative Γ ⇒ colder with
height): since $\Delta z$ is positive on average (DEM terrain sits above
ERA5-Land's smoothed orography), this produces a **colder** correction at
higher terrain — the physically expected direction. Implemented and applied
throughout §D.1's production forcing cache (`stage1_cache_forcing.py`).

**Elevation mismatch, quantified** (`artifacts/snow17_elevation_mismatch_audit/elevation_stats.json`):

| domain | mean Δz | median Δz | fraction ≥1000 m |
|---|---|---|---|
| Full Sierra (n=694) | +263.3 m | +216.9 m | 1.30% |
| North (n=293) | +241.2 m | +223.6 m | 0.0% |
| Central (n=224) | +216.1 m | +163.0 m | 0.0% |
| South (n=177) | +359.8 m | +281.7 m | 5.08% |

**Why −4.5 K/km specifically (not −6.5, not a variable ELR):** Dutra, E.,
Fernandes, A., Trigo, I. F., et al. (2020). Environmental Lapse Rate for
High-Resolution Land Surface Downscaling: An Application to ERA5. *Earth and
Space Science*, 7(3). https://doi.org/10.1029/2019EA000984. Local copy:
`docs/papers/Earth and Space Science - 2020 - Dutra - Environmental Lapse
Rate for High‐Resolution Land Surface Downscaling  An.pdf` (note: the filename
contains a Unicode hyphen `‐` U+2010 in "High‐Resolution", not a plain ASCII
hyphen — copy-paste the exact filename rather than retyping it, or the path
will silently fail to resolve).

−4.5 K/km (`clrO` in the paper) is Dutra's own observationally-derived
optimal constant for a western-US domain (125-100°W, 30-50°N, includes the
Sierra Nevada; GHCN station regression, 2,941 stations, Eq. 2) — not a
generic textbook default. It explicitly outperforms −6.5 K/km, which the
paper calls "too high" for this region, and the paper's own Conclusions
state there is "no significant added value of the variable ELR when
compared with the constant ELR" for land-surface variables closest to
Snow-17's use case (soil temperature, snow depth). Full feasibility audit
(why Dutra's exact model-level method wasn't used — no local ERA5
model-level archive; the pressure-level approximation was tested and
explicitly rejected, not merely deprioritized, since it captures only 1-3 of
Dutra's 9 native levels at representative Sierra cells, worst at highest
elevation): `artifacts/dutra2020_elr_feasibility_audit/` (`literature_method.md`,
`pressure_level_approximation_feasibility.md`, `recommendation.md`,
`feasibility_summary.json`, `provenance.md`).

**This lapse-rate value (−4.5 K/km) may remain the Sierra literature-supported
baseline for any experiment unless a specific experiment provides a better
justified alternative** — it is not tied to ERA5-Land specifically, only the
*use of ERA5-Land's orography as the reference surface* is.

### Terrain / DEM **[SIERRA-STANDARD]**

**Canonical cache (confirmed intact):**
`/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/swe_target_spatial_diagnostic/external_data/usgs_dem/`
— 51 GeoTIFF tiles, USGS NED 1/3 arc-second (~10 m), dated 2026-06-16. (An
earlier belief that this cache was purged was mistaken — two *other*,
non-canonical paths were checked and found empty; the canonical path had it
the whole time. Do not repeat the "DEM cache was purged" claim.)

| Status | RESOLVED / IMPLEMENTED / VERIFIED |
|---|---|

## A.7 Snow-17 timestep vs. forecast lead time **[UNIVERSAL distinction]**

**These are not the same thing, and conflating them is an easy mistake to
make:**

$$
\boxed{\texttt{dt (Snow-17's integration timestep)} \;\neq\; \texttt{forecast lead time}}
$$

A 7-day forecast may be driven by **28 × 6-hourly** forcing steps (`dt=6`)
or by **7 × daily** forcing steps (`dt=24`) — the *lead time* is how long
the forecast covers; the *dt* is how the physical model integrates the
forcing you give it. `snow17()`'s `dt` argument (hours) must match the
actual cadence of your `time`/`prec`/`tair` arrays — the function asserts
`time.shape == prec.shape == tair.shape` and integrates step-by-step at
whatever `dt` you pass; it does not resample for you.

**What Stage 1 (and the spin-up experiment and smoke test) actually use,
verified directly from the code (`DT_HOURS = 24` in
`scripts/stage1_calibrate.py`, `scripts/snow17_smoke_test_wy1985.py`, and
`scripts/spinup_experiment.py`): daily forcing, `dt=24`.**

**This is explicitly flagged, not silently reconciled**, against the older
document (`docs/intern/permultter_snow17.md`), which discusses
6-hourly forcing in its temperature-construction section — that section
describes a *different, Phase-2/ACE2-oriented* procedure (§C.2 below), not
this project's current daily-timestep observational baseline. If your
experiment uses ACE2's native 6-hourly cadence, you can and should run
Snow-17 at `dt=6` directly on that cadence — do not force it to daily
unless you have a specific reason to (and if you do resample, document
exactly how precipitation depth and temperature are aggregated/interpolated
across the resample; nothing in this repo currently does that for Snow-17
production forcing).

**A useful, source-verified detail for whichever `dt` you choose:** the
"6hr" embedded in `MFMAX`/`MFMIN`'s textbook units is not a hidden
requirement that `dt=6` — `melt_function()` already scales its output by
`dt/6` (`scripts/vendor/snow17.py:352`), and the antecedent-temperature-index
decay (`tipm_dt`) is separately scaled the same way (`snow17.py:143`). The
calibrated `MFMAX`/`MFMIN` values in §A.2 are used correctly by the code at
whatever `dt` you pass — you do not need to manually rescale them.

## A.8 What Snow-17 itself requires (P and T), independent of source **[UNIVERSAL]**

**Precipitation:** Snow-17 requires **precipitation depth (mm) per model
timestep**, not an arbitrary rate or a cumulative archive value. See §C for
exactly how to get there from ERA5-Land, ACE2, or another source.

**Temperature:** Snow-17 requires **air temperature in °C, at the same
timestep as precipitation** (same array length, asserted by the function).
See §C for source-specific unit conversions and whether/how elevation
correction should be applied.

---

# PART B — Forecast-Start State-Initialization Procedure

This section is the generic recipe to follow, regardless of lead time.
**Read §A.3 first** — the reason this procedure works the way it does is
that Snow-17 has no restart-state API.

## B.1 What "1-year spin-up" actually means (and does not mean)

**Do not interpret the validated 1-year spin-up rule as meaning a short-lead
weather model has to predict a full year.** The spin-up is a *state
initialization* step run on historical/reanalysis forcing **before** your
forecast starts — it is completely separate from, and has nothing to do
with, the length of your generated forecast.

For a forecast beginning at $t_0$:
$$
\boxed{\text{historical/reanalysis } P,T \;\xrightarrow{\text{Snow-17}}\; S(t_0)}
$$
then
$$
\boxed{\text{generated } P,T \text{ from } t_0 \text{ onward} \;\xrightarrow{\text{Snow-17}[S(t_0)]}\; SWE}
$$
Because there is no restart API (§A.3), "$\text{Snow-17}[S(t_0)]$" in
practice means: one continuous `snow17()` call whose input arrays span from
before $t_0$ through your forecast end, with state accumulated naturally by
running through $t_0$ rather than being reset there.

**Example — predicting March 25 → April 1:**
```
preceding forcing history
       |
Snow-17 spin-up / state evolution
       |
Snow-17 state on March 25   (implicit -- it's just where the running loop is)
       |
generated P/T for March 25-April 1
       |
Snow-17 continued from the March-25 point, SAME continuous call
       |
April-1 SWE
```
Do **not** zero-initialize Snow-17 on March 25 — accumulated winter snow and
its thermal/liquid states already exist by then, and a March-25 cold start
would discard that.

## B.2 Where the "1 year" rule comes from

**Experiment:** `artifacts/snow17_spinup_convergence_test/` —
`provenance.md`, `convergence_summary_by_run.csv`,
`state_convergence_at_1984-10-01.csv`,
`swe_diff_trajectory_post_1984-10-01.csv`, `plots/`. Script:
`scripts/spinup_experiment.py`. Compute: job 58137126, node nid004220
(account m2637).

**Design:** 9 representative ERA5-Land cells (North/Central/South ×
low/medium/high elevation). Zero-state cold starts at 4 candidate spin-up
lengths (6-month, 1-year, 2-year, 3-year before 1984-10-01) were each
compared against a 5-year reference spin-up, tracking all five internal
states plus `model_swe`.

**Result, verified from `convergence_summary_by_run.csv`:** at 1984-10-01,
the **1-year, 2-year, and 3-year** runs match the 5-year reference **exactly
(all diffs = 0.0)** for every tracked state. The 6-month run has small
nonzero residuals (max `model_swe` diff 0.335 mm, max `w_q` diff 0.675 mm).

**Physical explanation:** the Sierra's dry summer removes the seasonal
snowpack entirely every year, resetting all five Snow-17 states to/near
zero annually — so a 1-year forcing-driven spin-up is sufficient regardless
of the artificial zero-state cold start, because the *true* state one year
prior is itself very close to zero in the Sierra's climate. **This physical
argument is Sierra-specific** (a domain without a reliable annual
snow-free reset would need a different, likely longer, spin-up rule — this
1-year number is not a universal Snow-17 constant, it is this project's
validated Sierra-specific result).

**Caveat, still open:** this spin-up test used *uncorrected* ERA5-Land
temperature (the elevation correction, §A.6, was decided after this
experiment ran). The physical argument is not expected to be sensitive to a
systematic ~1-2 K offset, and it is *substantially* (not formally) supported
by the §D.1 smoke test, which used elevation-corrected T throughout a 1-year
spin-up and produced finite, physically sane, correctly-seasonal SWE — but
a dedicated re-run of the exact spin-up convergence test with corrected T
has not been done. Treat as low-risk but open.

## B.3 Generic initialization recipe

```
1. Choose your forecast start date t0.

2. Start Snow-17 at least 1 year before t0 (the validated Sierra rule;
   see B.2's caveat above if your experiment's domain/assumptions differ
   materially from the Sierra dry-summer-reset case).

3. Drive that pre-forecast period using the best available physically
   consistent meteorological forcing (ERA5-Land for the current
   observational baseline; see Part C for what "physically consistent"
   means for your source).

4. Because there is no restart-state API (A.3), do NOT try to save/reload
   state at t0. Instead:

5. Continue the SAME continuous snow17() call from before t0 through your
   generated forecast's end -- i.e., concatenate your spin-up-period P/T
   arrays with your generated-forecast P/T arrays into one time/prec/tair
   triple, and make one call.
```

## B.4 Worked example — 7-day forecast, March 25 → April 1

Even though not every experiment uses a 7-day forecast, this example makes
the initialization logic concrete.

```
Scientific target:        April-1 SWE
Forecast initialization:  March 25
Forecast lead:             7 days
```

Procedure:
```
1. Start Snow-17 approximately one year before March 25 (e.g. the previous
   October 1, following the WY-spinup convention in B.2) using validated
   historical/reanalysis forcing (ERA5-Land P/T + elevation correction,
   Part C.1) for state initialization.

2. Run continuously through March 25 -- one unbroken snow17() call.

3. At March 25, the running state IS the "state on March 25" -- there is
   nothing to explicitly "save," because you have not stopped the call.

4. Concatenate the experiment's generated P/T sequence (March 25 - April 1)
   onto the SAME time/prec/tair arrays used in steps 1-2.

5. Call snow17() once, over the full concatenated array (spin-up period +
   forecast period). Do not make two separate calls.

6. Read model_swe on April 1 (the last index of the combined array, or
   wherever April 1 falls in the combined time index).
```

**If you need to run many forecast start dates efficiently** (e.g. many
March-25-style forecasts across years) and re-running the full historical
spin-up for each one is expensive: the currently validated pattern in this
repo (§D.1, `stage1_pipeline_common.py::run_region_continuous`) is to run
**one long continuous simulation** covering the entire historical span once,
and then read off the state implicitly by slicing the resulting SWE/output
arrays at whatever dates you need — this works because Snow-17's state
trajectory *is* the sequence of calls, not a separate object you can extract
mid-stream. If your experiment truly needs many independent forecast start
dates with different generated-forecast branches after each one (i.e. a
tree of forecasts sharing a common history), you will need to either (a)
re-run the shared-history portion once per branch (simple, still correct,
possibly slow), or (b) modify the vendored function to expose/accept state
explicitly (not implemented in this repo — see §A.3).

---

# PART C — Experiment-Specific Atmospheric Forcing Choices

This is where forecast lead time and weather-generator choice actually
matter: not to Snow-17's configuration (Part A), but to what forcing you
feed it and how you prepare that forcing.

## C.1 ERA5-Land — the current observational baseline **[SIERRA-STANDARD]**

**Precipitation — `tp` (total_precipitation):**
`/global/cfs/projectdirs/m3522/datalake/ERA5-Land/total_precipitation/ERA5_{year}_total_precipitation.nc`,
native units **m**, native cadence hourly.

**Bug found and fixed:** ERA5-Land `tp` in this archive is **accumulated
since 00 UTC, resetting every calendar day** — not a sequence of independent
per-hour depths (confirmed by direct inspection: values rise monotonically
from ~0 at 01:00 to the full daily total at the *next* day's 00:00, then
reset). A naive `resample('D').sum()` over 24 already-cumulative values
over-counts by roughly an order of magnitude — this was caught by a smoke
test producing a physically impossible **1284.8 mm/day** maximum.

**Corrected rule:** take the value at hour 00:00 (the pre-reset, fully
accumulated total) and attribute it to the *preceding* calendar day, across
the full concatenated multi-year hourly series (handles the Dec 31→Jan 1
boundary correctly too). After the fix, max daily total across all test
cells/years (1979-1985) is **102.4 mm/day** — a plausible atmospheric-river
extreme. `t2m` is an instantaneous field; a plain hourly mean per calendar
day is correct for it as-is.

**Reference implementation:** `scripts/spinup_experiment.py`
(`extract_forcing_all_years`); also reused verbatim in
`scripts/stage1_cache_forcing.py`.

**Production warning, still open:** `src/snow_ml/data.py`'s
`ERA5_DAILY_REDUCTIONS = {"tp": "sum"}` convention is exactly the
naive-sum pattern that caused this bug and **has not been checked/fixed
there** as of this writing. Before trusting any output of that convention
for `tp`, confirm whether it has been fixed to use the reset-aware rule
above. This applies to every future use of ERA5-Land (or plain ERA5) `tp`
in this repository, not just Snow-17.

**Temperature:** ERA5-Land 2-m air temperature, native 0.1° grid, K→°C
conversion, then the §A.6 elevation correction using ERA5-Land's own
orography as $z_{\text{forcing}}$.

## C.2 ACE2-generated forcing **[ACE2-SPECIFIC — not yet used in any production Snow-17 run in this repo]**

**Native cadence and grid:** ACE2's checkpoint runs at a **6-hour timestep**
on a 180×360 1-degree Gaussian grid (verified from
`docs/ACE2_ERA5_WY2016_Capability_Audit.md`). This is coarser and at a
different cadence than the ERA5-Land 0.1°/daily baseline (§C.1) — do not
assume you can drop ACE2 output directly into the ERA5-Land pipeline without
re-deriving the grid mapping and elevation reference (see §A.6's generic
rule).

**Precipitation — `PRATEsfc`:** units `kg m⁻² s⁻¹` (a rate), per the same
audit. Converting to Snow-17's required mm-per-timestep depth at ACE2's
native 6-hour cadence:
$$
P_{6h} = PRATEsfc \times 21600
$$
(21600 s = 6 h; 1 kg/m² of water = 1 mm depth, so this conversion is
dimensionally exact). **This conversion has not yet been run through a
production Snow-17 forcing pipeline in this repo** — verify it against
whatever specific ACE2 output file you are using before trusting it
scientifically, per `docs/ACE2_Snow17_Scientific_Validity_Rule.md`'s
production gate (no tutorial/example/placeholder values as scientific
input; a documented manifest is required before any production run).

**Temperature:** ACE2 outputs relevant temperature/state fields at the same
6-hour cadence; elevation correction requires first determining what
elevation ACE2's own grid represents (its forcing files provide `HGTsfc`;
confirm this is the correct reference before reusing the ERA5-Land-specific
formula instantiation in §A.6 as-is).

**Known non-production artifact, explicitly flagged as invalid for science:**
`/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016/snow17_member_20151101_provisional`
is a software-execution demonstration only (NOAA-OWP `ex1` example
parameters, a zero cold start, a pressure-derived elevation estimate — see
`docs/ACE2_Snow17_Scientific_Validity_Rule.md`). **Its SWE values must not be
used for scientific analysis, validation, or training.**

## C.3 CMIP6-recentered historical procedure (Phase 2, re-scoped) **[EXPERIMENT-SPECIFIC — not the current baseline]**

The old procedure in `docs/intern/permultter_snow17.md` §"Constructing the
2-m Temperature Forcing" (lines 318-438) — recentering a 6-hourly
ACE-ERA5/NeuralGCM-generated anomaly pattern around a matching CMIP6 daily
mean, $T_{\mathrm{adjusted},i} = \mu_{\mathrm{CMIP}} +
(T_{\mathrm{ML},i}-\mu_{\mathrm{ML}})$ — belongs to a **different,
later/parallel experiment**, not the current ERA5-Land observational
baseline (§C.1) and not yet evaluated as future ACE2/CMIP6 methodology. It
is preserved as historical/possibly-future methodology, not deleted, and is
**not** the procedure you should reach for by default — do not treat it as
"the" universal Snow-17 temperature procedure.

## C.4 What Snow-17 needs vs. what your source outputs — quick lookup

| Item | Snow-17 requires | ERA5-Land (§C.1) | ACE2 (§C.2) |
|---|---|---|---|
| Precipitation | depth (mm) per timestep | `tp`, reset-aware hour-00 extraction | `PRATEsfc` (kg/m²/s) × 21600 for 6h steps |
| Temperature | °C per timestep | `t2m` (K→°C) + elevation correction vs. ERA5-Land orography | native temperature field + elevation correction vs. ACE2's own reference orography (verify first) |
| Timestep | must match your `dt` argument | daily (`dt=24`), this project's current baseline | 6-hourly (`dt=6`) native |
| Elevation reference | $z_{\text{forcing}}$ in §A.6's formula | ERA5-Land orography | ACE2's own grid orography (`HGTsfc`) — confirm, don't assume |

This table should answer: *what does Snow-17 itself require? what does a
given source model actually output? what exact conversion connects them?*

---

# PART D — Validated Stage-1 Reference Experiment

Everything in this part is Hongyu's specific, completed WY1985-2021
ERA5-Land experiment — a concrete instance of Parts A-C, not a second set of
universal rules. If you are using Workflow A (§0), the only things you need
from this section are the frozen theta paths (§D.1) and the known
limitations (§D.5). If you are recalibrating (Workflow B), this section is
your design template.

## D.1 What was run, and where the outputs live

**Full report:** `docs/Snow17_Stage1_SCEUA_Validation_Report.md`.

**Design:** SCE-UA (SPOTPY) calibration on WY1985-2004 daily regional SWE
trajectories (not April-1 alone), North/Central/South independently, 7
calibrated + 5 fixed parameters per region (§A.2), on the full 694-cell
ERA5-Land working grid (§A.5), with occupancy-weighted regional aggregation
(§A.5), −4.5 K/km elevation correction (§A.6), 1-year zero-state spin-up
(§B.2), daily timestep (§A.7). Regions frozen, then evaluated **with no
recalibration** on held-out WY2005-2020 and WY2005-2021.

**Frozen parameter files — use this copy:**
```
docs/snow17_reference_data/frozen_parameters/theta_N.json
docs/snow17_reference_data/frozen_parameters/theta_C.json
docs/snow17_reference_data/frozen_parameters/theta_S.json
```
Full JSON: calibrated values, fixed values, bounds, objective, calibration
NSE, optimizer seed/settings, evaluation count, convergence diagnostics, git
commit, forcing artifact path. (Canonical/production location, same content,
also read directly by `scripts/stage1_calibrate.py`'s `OUT_DIR` at
calibration time: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/snow17_stage1_scua_validation/frozen_parameters/`
— not reachable without Perlmutter access; the `docs/` copy above is the one
that ships via GitHub.)

**Independent reproducibility check:** re-running
`Snow17RegionSetup.simulation()`/`objectivefunction()` directly with each
saved `calibrated_parameters` vector (no optimizer involved) reproduced the
recorded 1-NSE exactly for all three regions (North 0.055148, Central
0.043014, South 0.055383).

**Production code (reuse these rather than reimplementing):**
- `scripts/stage1_cache_forcing.py` — reset-aware ERA5-Land tp/t2m extraction + elevation correction, cached once
- `scripts/stage1_cache_observed_swe.py` — observed regional SWE cache
- `scripts/stage1_pipeline_common.py` — continuous full-period simulation, whole-Sierra combination, metrics (reusable across experiments)
- `scripts/stage1_calibrate.py` — SCE-UA calibration driver (Workflow B template)
- `scripts/stage1_calibration_diagnostics.py`, `scripts/stage1_heldout_evaluation.py`, `scripts/stage1_plots.py` — post-calibration diagnostics/plots
- `scripts/stage1_verify_theta.py` — independent reproducibility check

## D.2 Calibration objective **[STAGE-1 decision, reusable default for Workflow B]**

SCE-UA via SPOTPY, objective = minimize `(1 − NSE)`, computed on the **full
daily regional SWE trajectory**, not April-1 alone. Mathematically
equivalent to minimizing SSE/MSE for a fixed calibration dataset (the NSE
denominator — observation variance — is constant per region).

SPOTPY configuration actually used: `parallel="mpc"`, `dbformat="ram"` (see
note below), seed=42, `ngs=7`, `kstop=5`, `pcento=peps=0.01`, requested 2000
repetitions/region (South converged in exactly 2000, Central in 2000, North
in 1995).

**SPOTPY implementation note (not a Snow-17 fact, but will bite you if
reusing this code):** `dbformat="csv"` combined with `parallel="mpc"` was
found, in this SPOTPY installation, to produce a corrupted/header-less
output file that `getdata()` cannot parse correctly. Fix used:
`dbformat="ram"`, writing the convergence CSV manually from the returned
structured array.

## D.3 Calibration / held-out split **[STAGE-1 decision]**

```
calibration          = WY1985-2004
held-out evaluation   = WY2005-2021 (dedicated WY2005-2020 verification first)
```

A single frozen calibration window with a completely untouched, contiguous
held-out block — not a LOYO (leave-one-year-out) scheme. Snow-17 is a
calibrated physical model here, not a statistical model being evaluated for
LOYO-style generalization. Observational SWE was used freely during
WY1985-2004 fitting; WY2005-2021 never influenced parameter values, bounds,
objective choice, or stopping criteria — verified in
`artifacts/snow17_stage1_scua_validation/leakage_audit.md`, written and
passed **before** any held-out metric was computed.

## D.4 Results

**Held-out whole-Sierra skill (WY2005-2020, before WY2021 was added):**
daily NSE = 0.934, RMSE = 65.4 mm, Pearson r = 0.982; April-1 NSE = 0.931,
r = 0.991, sign/anomaly agreement across years = 93.75%. Adding WY2021
leaves these numbers essentially unchanged (daily NSE 0.934, April-1 r
0.991).

**Calibration improved on the fixed-midpoint baseline in every region**
(NSE, WY1985-2004): North 0.913→0.945, Central 0.941→0.957, South
0.895→0.945, Whole-Sierra 0.932→0.958.

Full 9-question answer set, all four required plot types, and the complete
metrics tables: `docs/Snow17_Stage1_SCEUA_Validation_Report.md`.

## D.5 Known limitations (read before using the frozen theta in a new experiment)

- **Systematic wet bias in low-snow years, Central & South regions:**
  April-1 SWE overpredicted by ~30-40% on average in the driest held-out
  years (vs. ~7-9% in wet years); North stays nearly unbiased in normal
  years but *underpredicts* the single most extreme drought year (WY2015).
- **UADJ saturates at its lower calibration bound (0.05) in all three
  regions** — a consistent signal (not noise) that the pre-registered
  bound range may not bracket the true optimum for this domain; several
  other parameters (SCF, MFMIN, PXTEMP1, PXTEMP2) also saturate in
  South/Central specifically (see §A.2's table, bound-hit values marked).
- **Melt-timing lag** in low-snow years: predicted SWE persists somewhat
  longer than observed after the true melt-out date in dry years (mean
  |peak-date error| 17-22 days across regions).

**Verdict (from the full report):**
```
B. Snow-17 has partial held-out skill but important limitations must be
   documented before ACE2 coupling.
```
This is not a blanket endorsement to use the frozen theta uncritically for
drought-year analysis specifically — see the full report for the
region/year breakdown before drawing conclusions in that regime.

---

# Appendix I — Master parameter/forcing lookup table

Consolidates §A.2 (parameters) and §A.8/§C (forcing) into one lookup.

| Item | What Snow-17 requires | Sierra project choice | Does forecast lead change it? | Action | Source/path |
|---|---|---|---|---|---|
| SCF | calibrated, 0.9-1.2 | frozen per region | No | load theta file (§D.1) | `docs/snow17_reference_data/frozen_parameters/theta_{N,C,S}.json` |
| MFMAX | calibrated, 0.5-1.3 | frozen per region | No | load theta file | same |
| MFMIN | calibrated, 0.1-0.6 | frozen per region | No | load theta file | same |
| UADJ | calibrated, 0.05-0.2 | frozen per region | No | load theta file | same |
| PXTEMP | calibrated, 0.0-2.0 | frozen per region | No | load theta file | same |
| PXTEMP1 | calibrated, -2.0-0.0 | frozen per region | No | load theta file | same |
| PXTEMP2 | calibrated, 0.0-4.0 | frozen per region | No | load theta file | same |
| RVS | fixed | 1.0 | No | hardcode | `scripts/vendor/snow17.py` |
| MBASE | fixed | 1.0 | No | hardcode | same |
| TIPM | fixed | 0.1 | No | hardcode | same |
| NMF | fixed | 0.15 (docstring says 0.04 — bug, ignore it) | No | hardcode | same |
| PLWHC | fixed | 0.04 | No | hardcode | same |
| Precipitation | depth (mm) per timestep | source-specific conversion (§C.4) | No (only the *length* supplied changes) | convert correctly per source | §C.1/§C.2 |
| Temperature | °C per timestep, elevation-corrected | ERA5-Land t2m + −4.5 K/km vs. ERA5-Land orography (default) | No | apply §A.6's generic rule for your source | §A.6, §C.1/§C.2 |
| Forecast initial state $S(t_0)$ | complete internal state (5 variables, §A.3) | 1-year zero-state spin-up, then continuous run | **Yes — $t_0$ itself moves** | run one continuous `snow17()` call spanning spin-up + forecast (§B) | §A.3, §B |
| Region assignment | one of 3 shared vectors per cell | KNN-filled geographic mask | No | look up cell's region, apply matching theta | §A.4, `docs/snow17_reference_data/regional_mask/` |
| Spatial aggregation | occupancy-weighted regional mean | §A.5 formula | No | use occupancy weights, not equal-weighting | §A.5 |

---

## Superseded / Resolved Historical Blockers

| Old blocker (as it reads in `docs/intern/permultter_snow17.md`) | Replaced by |
|---|---|
| "Complete calibrated Sierra parameter package... **Current major blocker**... archived CNRFC NFDC1 Snow-17 configuration" | Abandoned as a target. Replaced by: 7 parameters calibrated per-region via SCE-UA (§A.2/§D.2/§D.3), 5 parameters fixed to documented `tonic` defaults (§A.2). NFDC1 archive search (`artifacts/nfdc1_snow17_configuration_retrieval/`) retained as evidence of unavailability, not an active blocker. |
| DAYGM — "Missing" | SUPERSEDED — not a `tonic` input at all (§A.1). |
| SI — "Missing" | SUPERSEDED — not a `tonic` input at all (§A.1). |
| ADC — "Missing... Obtain the complete calibrated ADC curve" | SUPERSEDED — not a `tonic` input at all (§A.1). |
| "Elevation-zone definitions and adjustments... Not yet obtained" | SUPERSEDED — replaced by the single-elevation-per-cell residual correction in §A.6; no multi-zone structure is part of this implementation. |
| "Initial SWE... WUS-D3 d02 `snow` from the corresponding simulation and preceding completed day" | SUPERSEDED — replaced by 1-year forcing-driven zero-state spin-up (§B), continuous execution (§A.3). No prior simulated or observed SWE is used for initialization. |
| "Other Snow-17 initial internal states... Not yet fully resolved" | RESOLVED — all five tracked states (§A.3/§B.2) converge from zero within 1 year in the Sierra's dry-summer climate; the states themselves and their restart limitations are now fully documented (§A.3). |
| Coarsen-8 working grid proposal | SUPERSEDED — ERA5-Land native 0.1° grid (§A.5). |
| Arbitrary 6.5 K/km lapse rate | SUPERSEDED — −4.5 K/km, literature-derived (§A.6); 6.5 K/km explicitly shown inferior in the source paper. |
| "Daily Dutra ELR" as a hard requirement | Resolved via feasibility audit, not implemented exactly — a defensible literature-optimal constant substitutes for it (§A.6), with the exact daily model-level method documented as infeasible with current local data holdings. |
| Pressure-level approximation of Dutra's method | Explicitly rejected, not merely deprioritized (§A.6) — shown to fail hardest at high Sierra elevation. |
| Unknown SCE-UA objective | RESOLVED — `1 − NSE` (§D.2). |
| Unknown fixed five parameters | RESOLVED (§A.2). |
| "DEM cache was purged" (an incorrect statement briefly made during this project's own recent work) | RESOLVED/corrected — the canonical cache was intact throughout (§A.6). |
| ACE2/CMIP6-recentered temperature construction procedure, previously presented as *the* Snow-17 temperature procedure | Re-scoped, not deleted — applies only to a separate Phase-2/experiment-specific case (§C.3), explicitly not the current observational baseline or a universal requirement. |
| Only SPOTPY/no calibration code existed | RESOLVED — full SCE-UA calibration run, frozen theta produced, held-out validated (§D). |

---

## Final Readiness Matrix

| Component | Final decision | Status | Source |
|---|---|---|---|
| Snow-17 implementation | UW-Hydro `tonic`, vendored, upstream bug fixed in both copies | VERIFIED | §A.1 |
| Restart/state capability | No restart API exists; one continuous run required | VERIFIED (source-read) | §A.3 |
| Working grid | ERA5-Land native 0.1° | IMPLEMENTED | §A.5 |
| Observational SWE target | UCLA WUS-SR `SWE_Post` | RESOLVED | `config/paths.py` `SWE_ROOT` |
| Regional mask | KNN-filled HUC8 mask, aggregated to ERA5-Land grid | IMPLEMENTED / VERIFIED | §A.4, `docs/snow17_reference_data/regional_mask/` |
| Spatial aggregation | Occupancy-weighted regional mean | VERIFIED (production) | §A.5, §D.1 |
| P forcing (ERA5-Land) | `tp`, reset-aware daily extraction | RESOLVED (bug+fix), production | §C.1; still PENDING in `src/snow_ml/data.py` |
| P forcing (ACE2) | `PRATEsfc` × 21600 for 6h steps | DOCUMENTED, NOT YET PRODUCTION-RUN | §C.2 |
| T forcing | ERA5-Land `t2m`, K→°C | RESOLVED, production | §C.1 |
| Elevation correction | −4.5 K/km, residual, generic rule + ERA5-Land instantiation | RESOLVED, production | §A.6 |
| Spin-up | 1 year, forcing-driven, zero start, continuous execution | RESOLVED / VERIFIED, production | §B.2, §D.1 |
| Timestep | daily (`dt=24`) for the current baseline; not a Snow-17 requirement | RESOLVED / VERIFIED | §A.7 |
| Seven calibrated parameters | SCF/MFMAX/MFMIN/UADJ/PXTEMP/PXTEMP1/PXTEMP2, Appendix B ranges | RESOLVED (values), FROZEN (Stage-1 vectors) | §A.2, §D.1 |
| Five fixed parameters | RVS/MBASE/TIPM/NMF/PLWHC, `tonic` defaults | RESOLVED / VERIFIED | §A.2 |
| Optimizer | SCE-UA via SPOTPY | RESOLVED, RUN | §D.2 |
| Objective | minimize `1 − NSE`, full daily trajectory | RESOLVED, RUN | §D.2 |
| Calibration period | WY1985-2004 | RESOLVED, RUN | §D.3 |
| Evaluation period | WY2005-2021, held out | RESOLVED, RUN, VALIDATED | §D.3/§D.4 |
| Frozen theta_N/C/S | exist, reproducibility-verified | DONE | §D.1, `docs/snow17_reference_data/frozen_parameters/` |
| ACE2 coupling | deferred; forcing conversion documented but not production-run | OPEN | §C.2 |

---

## What remains genuinely OPEN (documented explicitly, not silently absorbed)

1. **`src/snow_ml/data.py`'s `ERA5_DAILY_REDUCTIONS` `tp` convention** — not
   yet checked/fixed against the accumulation-bug finding (§C.1). Stage 1's
   own extraction code bypasses this module and is correct; any *other*
   script that uses `ERA5_DAILY_REDUCTIONS={"tp": "sum"}` directly should be
   checked before trusting its `tp` output.
2. **ACE2 forcing conversion (§C.2) is documented but not yet exercised in
   any production Snow-17 run** — the only existing ACE2→Snow-17 output in
   this repo is explicitly a non-scientific software demonstration (§C.2).
3. **True Snow-17 state checkpoint/restart** is not implemented — every
   experiment must re-run a continuous trajectory from spin-up through
   forecast end (§A.3/§B.3). If you need many independent forecast branches
   sharing history, you will need to either re-run the shared portion per
   branch or modify the vendored function (not done here).
4. **Re-verification of the 1-year spin-up conclusion under
   elevation-corrected T** — substantially, not formally, addressed (§B.2's
   caveat).
5. **Occupancy-weighted regional aggregation** is implemented in two places
   (`scripts/snow17_smoke_test_wy1985.py` inline, and
   `scripts/stage1_pipeline_common.py` as a reusable function) — prefer the
   latter for new work rather than re-copying the smoke test's inline
   version.
6. **A verbally-recalled "~5.6%" figure** for cells with ≥1000 m elevation
   mismatch does not exactly match either the Full-Sierra (1.30%) or South
   (5.08%) figures verified from `elevation_stats.json` — flagged, not
   reconciled; the artifact numbers are authoritative.
7. **ACE2 coupling itself (Phase 2)** is explicitly deferred — Stage 1 did
   not use ACE2, CMIP6, or NeuralGCM forcing, and this document does not
   claim the Snow-17 configuration above has been validated end-to-end with
   any generated (non-reanalysis) weather source yet.

---

## Provenance of this document

- Old table location: `docs/intern/permultter_snow17.md`, lines 291-316 (the
  requirements table) and 318-438 (the ACE2/CMIP6 temperature construction
  procedure, now §C.3). **Left unmodified** — it is intern-onboarding
  material, now superseded in scope by this document.
- An identical copy of the same old table also exists at
  `docs/intern/Catherine/Task_Gaoyinying.md` (line 421 area) — **not
  modified**, out of scope, flagged here for awareness.
- **`docs/snow17_reference_data/`** (added 2026-09-11): full-size copies of
  the regional mask (`regional_mask/` — npz, nc, both fill-diagnostic pngs,
  `fill_summary.json`, `provenance.md`) and the three frozen parameter files
  (`frozen_parameters/theta_{N,C,S}.json`). These exist here, duplicated from
  their canonical `artifacts/`/pscratch locations, **specifically because
  Perlmutter home directories are not externally readable and only `docs/`
  and `scripts/` are synced to this project's GitHub** — the GitHub copy of
  this repo would otherwise not include either file. Every reference to
  these two artifacts elsewhere in this document points at the
  `docs/snow17_reference_data/` copy, not the canonical one, for that
  reason.
- This document and its companion machine-readable manifest
  (`docs/Snow17_Methodology_and_Readiness.json`) were built by tracing every
  artifact/log/script cited above directly (including re-reading the
  vendored `snow17()` source line-by-line to verify the restart-state claim
  in §A.3, and re-verifying every referenced file path exists as of
  2026-09-11); no section was reconstructed from memory alone.
- **2026-09-11 restructuring**: reorganized around the A(invariant
  config)/B(state init)/C(experiment-specific forcing)/D(Stage-1 record)
  distinction, so the document holds up across different forecast lead times
  and weather generators without needing to be re-derived per experiment.
  No scientific decision was changed in this pass —
  only organization, the addition of the restart-state analysis (§A.3, newly
  derived from source but not previously written down), the generic
  elevation-correction rule (§A.6), the timestep-vs-lead-time distinction
  (§A.7), the ACE2 forcing conversion table (§C.2/§C.4), and the Stage-1
  results (§D, previously a separate §17 addendum, now fully integrated).
  One path-fidelity fix: the Dutra PDF filename uses a Unicode hyphen
  (`High‐Resolution`), corrected in §A.6 from an earlier plain-ASCII-hyphen
  transcription that would not resolve if copy-pasted literally.
