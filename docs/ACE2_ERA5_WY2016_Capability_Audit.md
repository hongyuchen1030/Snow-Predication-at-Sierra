# ACE2-ERA5 WY2016 Capability Audit

Scope: read-only inspection of the exact checkpoint and WY2016 configuration.
Checkpoint: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/assets/checkpoints/ace2_era5_ckpt.tar`.
The checkpoint has 44 inputs and 50 outputs on the 180x360 1-degree Gaussian grid at a 6-hour timestep.

## Exact output capability

The saved `member_20151101/autoregressive_predictions.nc` contains every one of the checkpoint's 50 `out_names`; it is not a reduced output selection.  The YAML writer requested unsupported extra names (`TMP500`, `TMP200`, `Q850/Q500/Q200`, pressure-level winds, additional heights, `DPT2m`, and column water), which are not checkpoint outputs and therefore were not saved.

### Prognostic state: ensemble-predicted

All have dimensions `(sample,time,lat,lon)`.  Units are K for temperature, `kg kg-1` for total water, `m s-1` for model-layer winds, Pa for `PRESsfc`, K for surface and 2-m temperature, and `kg kg-1` / `m s-1` for `Q2m` / 10-m winds.

- `PRESsfc`, `surface_temperature`, `TMP2m`, `Q2m`, `UGRD10m`, `VGRD10m`
- `air_temperature_0` through `air_temperature_7`
- `specific_total_water_0` through `specific_total_water_7`
- `eastward_wind_0` through `eastward_wind_7`
- `northward_wind_0` through `northward_wind_7`

### Diagnostic outputs: diagnostic-predicted

All have dimensions `(sample,time,lat,lon)`.  The units are `W m-2` for fluxes, `kg m-2 s-1` for `PRATEsfc`, K for `TMP850`, m for `h500`, and `kg m-2 s-1` for the moisture-advection tendency.

- `LHTFLsfc`, `SHTFLsfc`, `PRATEsfc`
- `ULWRFsfc`, `ULWRFtoa`, `DLWRFsfc`, `DSWRFsfc`, `USWRFsfc`, `USWRFtoa`
- `tendency_of_total_water_path_due_to_advection`, `TMP850`, `h500`

`ULWRFtoa` is explicitly `Upward LW radiative flux at TOA`; it matches the physical direction and units of CMIP6 `rlut`, with no sign reversal.

### Prescribed forcing: not ensemble inflation

The 2015/2016 forcing files provide `DSWRFtoa`, `global_mean_co2`, `sea_ice_fraction`, `ocean_fraction`, `land_fraction`, `HGTsfc`, and hybrid coefficients `ak_0..8`, `bk_0..8`.  Of these, only `DSWRFtoa` is a next-step forcing.  The other forcing-only inputs are supplied to each step as input-only names.  They are identical across the five lag members because each config uses the same forcing directory.

`sea_ice_fraction` is therefore prescribed, not a new ensemble realization.  `surface_temperature` is an output/state despite also being present in the forcing files; it is not an input-only forcing in this checkpoint.

### Static fields: not ensemble inflation

`HGTsfc` and `land_fraction` are time-invariant.  The hybrid coordinate has nine interfaces
`p_interface(k) = ak_k + bk_k * PRESsfc`, defining eight layers.  `ocean_fraction` is effectively a prescribed mask/forcing rather than a predicted ocean state.

## 19-channel CNN mapping

| CNN variable | ACE source | Availability | Type | Transformation | Confidence |
|---|---|---|---|---|---|
| `tos` | none | unavailable | unavailable | Skin temperature is not ocean SST | high |
| `siconc` | `sea_ice_fraction` | prescribed | prescribed | no transformation, but identical forcing sequence | high |
| `rlut` | `ULWRFtoa` | direct | diagnostic-predicted | none | high |
| `zg_500` | `h500` | direct | diagnostic-predicted | none | high |
| `ta_850` | `TMP850` | direct | diagnostic-predicted | none | high |
| `ua_850` | `eastward_wind_0..7` | derived | ensemble-predicted | log-pressure interpolation using hybrid interfaces | moderate |
| `va_850` | `northward_wind_0..7` | derived | ensemble-predicted | log-pressure interpolation using hybrid interfaces | moderate |
| `ua_200` | `eastward_wind_0..7` | derived | ensemble-predicted | log-pressure interpolation using hybrid interfaces | moderate |
| `va_200` | `northward_wind_0..7` | derived | ensemble-predicted | log-pressure interpolation using hybrid interfaces | moderate |
| `hus_850` | `specific_total_water_0..7` | unavailable | unavailable | Total water is not water-vapor specific humidity | high |
| `psl` | `PRESsfc`, `HGTsfc`, `TMP2m`, `Q2m` | uncertain | derived candidate | sea-level reduction requires a validated method | low |
| `tas` | `TMP2m` | direct | ensemble-predicted | none | high |
| `zg_50` | state plus hybrid coordinate | uncertain | derived candidate | hydrostatic integration; no predicted geopotential layers | low |
| `ta_50` | `air_temperature_0..7` | uncertain | derived candidate | log-pressure interpolation; only top two layers constrain 50 hPa | low |
| `ua_50` | `eastward_wind_0..7` | uncertain | derived candidate | same coarse top-layer interpolation | low |
| `va_50` | `northward_wind_0..7` | uncertain | derived candidate | same coarse top-layer interpolation | low |
| `mrso` | none | unavailable | unavailable | no soil-moisture state/output | high |
| `thetao_50m` | none | unavailable | unavailable | no subsurface ocean state/output | high |
| `thetao_100m` | none | unavailable | unavailable | no subsurface ocean state/output | high |

For pressure reconstruction, form interface pressures at every timestep from `PRESsfc`, calculate layer-center pressure from adjacent interfaces, and interpolate the relevant model-layer value linearly in `log(p)`.  Mask 850-hPa values wherever surface pressure is below 850 hPa.  200 hPa lies well within the modeled atmosphere.  Fifty hPa lies within the vertical domain but is constrained only by the upper two coarse layers; it must not be included without validation.

## Additional useful predicted fields

`PRATEsfc`, `Q2m`, 10-m winds, model-layer temperature/water/winds, surface pressure, skin temperature, radiative fluxes, turbulent fluxes, and moisture-advection tendency are valid ensemble-predicted or diagnostic-predicted auxiliary targets.  They can support representation learning, but they are not substitutes for unavailable ocean, soil, or water-vapor channels.

## Tiers

- **Tier 1:** `tas`, `ta_850`, `zg_500`, `rlut`.
- **Tier 2:** `ua_850`, `va_850`, `ua_200`, `va_200`, after validated hybrid-coordinate interpolation and below-ground masking.
- **Tier 3:** `tos`, `siconc` (prescribed), `hus_850`, `psl` (until validated), all 50-hPa fields (until validated), `mrso`, `thetao_50m`, `thetao_100m`.

**Maximum defensible ACE pretraining subset: 8 variables**: `tas`, `ta_850`, `zg_500`, `rlut`, `ua_850`, `va_850`, `ua_200`, `va_200`.
