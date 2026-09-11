# ACE2-ERA5 to Snow-17 Scientific Validity Rule

## Production inputs

Production ACE2-ERA5 to Snow-17 experiments must not use tutorial, example,
default, synthetic, placeholder, or climatological values as scientific input.
Every meteorological input, Snow-17 parameter, terrain field, lapse-rate
treatment, and initial state must be either:

1. an observed, reanalysis, or model field appropriate to the represented
   quantity;
2. a parameterization calibrated for, or independently established for, the
   relevant Sierra Nevada operational unit; or
3. a documented, scientifically defensible transformation of such data.

If an input cannot meet one of these criteria, it is an unresolved scientific
input. The production workflow must stop and report it; it must not substitute
a demonstration value.

## Existing provisional output

`/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016/snow17_member_20151101_provisional`
is a software-execution demonstration only. It used the NOAA-OWP `ex1` example
parameters, a zero cold start, and a pressure-derived elevation estimate. Its
SWE values are not Sierra forecasts and must not be used for scientific
analysis, validation, or training.

## Required production gate

Before any production Snow-17 integration, retain a manifest that names the
source, geographic applicability, temporal applicability, units, and any
transformations for:

- precipitation and 2 m air temperature;
- each Snow-17 parameter and the areal-depletion curve;
- Snow-17 initial/carryover state;
- Sierra-intersection terrain height and ACE grid terrain height; and
- the temperature elevation correction, including its lapse-rate source.

The Snow-17 runner must reject runs that do not provide this manifest.
