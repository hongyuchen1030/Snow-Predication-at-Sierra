# CMIP6 Predictor Selection for WUS-D3 Parent Models
The selection is organized in three stages:
1. **Shared candidate pool:** CMIP6 variables physically available across the four WUS-D3 parent models.
2. **Long-term predictability sources:** Paul's conceptual sources of seasonal/long-lead predictability mapped onto the shared CMIP6 fields.
3. **Final first-input selection:** the subset selected for the first encoder input matrix.
For this screening step, the missing `MPI-ESM1-2-HR ssp370` branch is assumed to have the same shared-variable status as its historical branch.
---
# 1. Shared CMIP6 Candidate Pool
| Physical quantity | CMIP6 variable | Shared options available | Notes |
|---|---|---|---|
| Sea surface temperature | `tos` | Surface-only | Shared SST field; suitable for Pacific/AQM-style SST work. |
| Geopotential height | `zg` | `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1 hPa` | Pressure-level field; `500 hPa` is available. |
| Air temperature | `ta` | `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1 hPa` | Pressure-level field. |
| Eastward wind | `ua` | `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1 hPa` | Pressure-level field. |
| Northward wind | `va` | `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1 hPa` | Pressure-level field. |
| Specific humidity aloft | `hus` | `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1 hPa` | Pressure-level field. |
| Near-surface air temperature | `tas` | Surface-only | Shared surface field. |
| Daily maximum near-surface air temperature | `tasmax` | Surface-only | Shared daily surface extreme. |
| Daily minimum near-surface air temperature | `tasmin` | Surface-only | Shared daily surface extreme. |
| Sea level pressure | `psl` | Surface-only | Shared surface pressure field. |
| Near-surface specific humidity | `huss` | Surface-only | Shared surface humidity field. |
| Near-surface relative humidity | `hurs` | Surface-only | Shared surface humidity field. |
| Precipitation | `pr` | Surface-only | Shared precipitation flux field. |
| Snowfall flux | `prsn` | Surface-only | Shared snowfall flux field. |
| Total precipitable water | `prw` | Column-integrated | Shared column water-vapor field. |
| TOA outgoing longwave radiation | `rlut` | TOA radiative flux | Best shared OLR-type field for MJO-style diagnostics. |
| TOA outgoing shortwave radiation | `rsut` | TOA radiative flux | Shared TOA radiative field. |
| TOA incoming shortwave radiation | `rsdt` | TOA radiative flux | Shared TOA radiative field. |
| Surface downwelling shortwave radiation | `rsds` | Surface-only | Shared surface radiative field. |
| Surface upwelling shortwave radiation | `rsus` | Surface-only | Shared surface radiative field. |
| Surface downwelling longwave radiation | `rlds` | Surface-only | Shared surface radiative field. |
| Surface upwelling longwave radiation | `rlus` | Surface-only | Shared surface radiative field. |
| Latent heat flux | `hfls` | Surface-only | Shared surface turbulent-flux field. |
| Sensible heat flux | `hfss` | Surface-only | Shared surface turbulent-flux field. |
| Surface wind speed | `sfcWind` | Surface-only | Shared surface wind-speed field. |
| Total soil moisture content | `mrso` | Column-integrated soil moisture | Shared monthly soil-moisture field. |
| Surface snow amount | `snw` | Surface-only | Shared snow-mass field. |
| Sea-ice concentration | `siconc` | Surface-only | Shared sea-ice concentration field. |
| Sea-water salinity | `so` | Depth-dependent; no exact shared native depth list | Requires common target depths after interpolation. |
| Sea-water potential temperature | `thetao` | Depth-dependent; no exact shared native depth list | Suitable source field for constructing standardized upper-ocean heat content. |
| Ocean x-velocity | `uo` | Depth-dependent; no exact shared native depth list | Requires common target depths after interpolation. |
| Ocean y-velocity | `vo` | Depth-dependent; no exact shared native depth list | Requires common target depths after interpolation. |
| Sea surface height above geoid | `zos` | Surface-only | Shared ocean surface-height field. |
| Cloud area fraction | `clt` | Column-integrated | Shared cloud-cover field. |
| Ice water path | `clivi` | Column-integrated | Shared cloud-ice path field. |
| Liquid water path | `clwvi` | Column-integrated | Shared cloud-liquid path field. |
| Evaporation | `evspsbl` | Surface-only | Shared evaporation-flux field. |
---
# 2. Paul's Long-Term Predictability Sources vs. Shared CMIP6 Fields
| Source category | Paul's predictability source | Shared CMIP6 field(s) / construction | Status for our 4 models | Decision / notes |
|---|---|---|---|---|
| **Atmosphere** | Madden–Julian Oscillation (MJO) | `rlut` for tropical convection/OLR; normally combined with `ua/va @ 850/200 hPa` | **Partial** | `rlut` is available at daily frequency historically, but shared daily `ua/va` are not available across all four. Do not construct a standard MJO index in the first input matrix. |
| **Atmosphere** | Synoptic conditions | `zg`, `psl`, `ta`, `ua`, `va`, `hus` | **Available** | Represent with a compact set of physically motivated levels rather than all 19 pressure levels. |
| **Atmosphere** | Stratospheric conditions | `zg`, `ta`, `ua`, `va` at ~`100–10 hPa` | **Available** | Shared stratospheric levels exist. Keep as a later extension/ablation rather than expanding the first matrix immediately. |
| **Atmosphere** | Deviations in atmospheric angular momentum | Primarily derived from `ua` plus atmospheric mass/pressure information | **Partial / not yet verified** | Do not construct AAM in the first matrix until all required fields and formulation are verified. |
| **Land** | Soil moisture | `mrso` | **Available** | Include as the primary shared land-memory predictor. |
| **Land** | Snow cover | `snc` | **Missing** | Exact snow-cover fraction is not shared. `snw` is shared but represents snow amount rather than fractional snow cover and creates potential target leakage for April-1 SWE. Do not include initially. |
| **Land** | Vegetation | `lai`, vegetation-fraction variables | **Missing** | No compatible vegetation diagnostic is shared across all four parent models. |
| **Land** | Groundwater | No shared groundwater diagnostic identified | **Missing** | No consistent groundwater-state field identified across all four models. |
| **Ocean** | Modes of variability — ENSO, PDO, AMV, AQM, etc. | `tos` → derive SST patterns / indices | **Available** | Core long-memory ocean predictor. Include. |
| **Ocean** | Pattern of ocean heat content — surface + depth | `thetao` → vertically interpolate and integrate over a common upper-ocean depth | **Available** | Include as a derived upper-ocean heat-content field rather than selecting an arbitrary native `thetao` depth. |
| **Ocean** | Sea ice | `siconc` | **Available** | Shared monthly sea-ice concentration. Include as an ocean/cryosphere memory candidate. |
---
# Final First-Input CMIP6 Variable and Level Selection

| CMIP6 field / derived field | Selected level / construction | Predictability source represented | Selection rationale |
|---|---|---|---|
| `tos` | Surface SST | Ocean modes: ENSO, PDO, AMV/AQM, Pacific SST variability | **Include.** Core slowly varying ocean-state field and directly supports the established Pacific SST and AQM hypotheses. Stone et al. (2023) derives AQM from SST–precipitation coupled variability. |
| `thetao` | **Upper-ocean heat-content representation derived after vertical interpolation** | Ocean heat-content memory | **Include.** Paul explicitly identifies patterns of ocean heat content as a long-term predictability source. Do not select an arbitrary native depth; standardize the four ocean vertical grids and construct an upper-ocean thermal/heat-content representation. |
| `siconc` | Surface sea-ice concentration | Sea-ice memory | **Include.** Directly corresponds to Paul's sea-ice predictability source and is shared across all four models. |
| `rlut` | TOA outgoing longwave radiation | MJO / tropical convection | **Include.** `rlut` alone cannot reconstruct the canonical MJO, but it contains the tropical convective/OLR component of MJO variability. Partial representation is sufficient for inclusion in the encoder. |
| `zg` | **500 hPa** | CPM / mid-tropospheric circulation / synoptic state | **Include.** `zg@500 hPa` directly gives Z500 and preserves the CPM definition. Chen et al. (2021) defines CPM from 500-hPa geopotential-height variability. |
| `ta` | **850 hPa** | Lower-tropospheric thermal state | **Include.** T850 represents lower-tropospheric air-mass temperature relevant to storm thermodynamics and rain/snow conditions. Sanuy et al. (2024) uses T850 with Z500 and sea-level pressure; Dong et al. uses 850/500/200-hPa temperature. |
| `ua` | **850 hPa** | Low-level circulation / transport; partial MJO information | **Include.** U850 represents low-level atmospheric transport and is also one of the standard circulation components associated with MJO diagnostics. Li et al. (2022) identifies U850 as a useful hydroclimate predictor. |
| `va` | **850 hPa** | Low-level circulation / meridional transport | **Include.** Completes the low-level horizontal wind vector and captures meridional transport that cannot be represented by `ua` alone. |
| `ua` | **200 hPa** | Upper-level jet / tropical teleconnections / partial MJO information | **Include.** U200 captures upper-tropospheric circulation and is one of the standard MJO circulation components. Li et al. (2022) identifies U200 as a useful precipitation predictor. |
| `va` | **200 hPa** | Upper-level jet / meridional circulation | **Include.** Completes the upper-level horizontal wind vector and allows the encoder to learn jet and teleconnection structure. |
| `hus` | **850 hPa** | Lower-tropospheric moisture state | **Include.** Represents atmospheric moisture available for transport and precipitation. Dong et al. (2025) uses specific humidity at 850/500/200 hPa; Zhang et al. (2023) uses 1000/850/500 hPa. |
| `psl` | Sea level | Large-scale surface circulation / synoptic state | **Include.** Complements Z500 by describing surface pressure systems. MSLP is commonly used together with Z500 and T850 in circulation/hydroclimate analyses. |
| `tas` | Near-surface | Surface thermal state | **Include.** Represents near-surface temperature relevant to snow/rain partitioning and snowpack thermodynamics; distinct from `ta850`. |
| `mrso` | Column-integrated total soil moisture | Land memory | **Include.** Directly corresponds to Paul's soil-moisture predictability source and is the clean shared land-memory variable across all four models. |
| `zg` | **50 hPa** | Stratospheric circulation | **Include.** Paul explicitly identifies stratospheric conditions as a long-term predictability source. One representative lower/mid-stratospheric level gives the encoder access to this signal without adding every stratospheric pressure level. |
| `ta` | **50 hPa** | Stratospheric thermal state | **Include.** Captures stratospheric temperature variability associated with changes in the polar vortex and stratosphere–troposphere coupling. |
| `ua` | **50 hPa** | Stratospheric zonal circulation / partial atmospheric-angular-momentum information | **Include.** Provides stratospheric zonal-flow information and retains part of the wind information relevant to angular-momentum variability even though we are not explicitly constructing AAM. |
| `va` | **50 hPa** | Stratospheric meridional circulation | **Include.** Completes the stratospheric horizontal circulation state rather than providing only zonal wind. |

## Final first-input field list

### Ocean memory
- `tos`
- `thetao → upper-ocean heat-content representation`
- `siconc`

### Tropical / intraseasonal signal
- `rlut`

### Tropospheric circulation and thermodynamic state
- `zg @ 500 hPa`
- `ta @ 850 hPa`
- `ua @ 850 hPa`
- `va @ 850 hPa`
- `ua @ 200 hPa`
- `va @ 200 hPa`
- `hus @ 850 hPa`
- `psl`
- `tas`

### Stratospheric state
- `zg @ 50 hPa`
- `ta @ 50 hPa`
- `ua @ 50 hPa`
- `va @ 50 hPa`

### Land memory
- `mrso`

**Total: 18 input fields / derived fields.**

## Not included in the first matrix

| Variable | Reason for exclusion |
|---|---|
| `pr`, `prsn` | Strongly relevant to SWE, but inclusion depends on forecast cutoff because in-season precipitation/snowfall can directly construct the April-1 SWE target. Resolve temporal leakage before adding them. |
| `snw` | Available, but it is already snow mass and therefore especially vulnerable to target leakage. |
| `huss`, `hurs` | Useful candidates, but initially redundant with `hus@850` and `tas`; retain for later ablation. |
| `prw` | Potentially useful integrated-moisture predictor, but initially redundant with `hus@850`; retain for ablation. |
| `tasmax`, `tasmin` | `tas` provides the initial surface-temperature channel. Extremes can be tested later. |
| `so` | Ocean salinity may contribute to ocean-memory structure but adds substantial vertical harmonization without a current SWE-specific hypothesis. |
| `uo`, `vo` | Ocean currents potentially carry long memory but no well-justified target depth or first-pass SWE mechanism has been established. |
| `zos` | Potential ocean-memory proxy but weaker motivation than SST and upper-ocean heat content. |
| Cloud / radiation variables other than `rlut` | Available but not directly tied to one of the selected long-lead predictability mechanisms strongly enough to enter the first matrix. |
| Vegetation | No compatible shared field across all four models. |
| Groundwater | No compatible shared diagnostic across all four models. |

## Selection principle

A field does **not** need to reconstruct an entire named climate index by itself to be included. If it contains a physically meaningful component of a known long-term predictability source, it can be supplied to the encoder and the encoder can learn useful combinations of those signals.

Therefore, for example, `rlut`, `ua@850`, and `ua@200` are retained even though the available data are insufficient to reconstruct a canonical daily MJO index consistently across all four models.