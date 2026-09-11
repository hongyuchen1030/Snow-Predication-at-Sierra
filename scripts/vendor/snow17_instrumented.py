#!/usr/local/bin/python
"""
Instrumented copy of UW-Hydro/tonic's snow17.py
(https://github.com/UW-Hydro/tonic/blob/master/tonic/models/snow17/snow17.py),
for the WY1985 spin-up-duration convergence diagnostic.

Two, and only two, changes from the upstream file:

1. BUG FIX (upstream typo, not a scientific/parameter change): the upstream
   file's ripeness branch at (in the original file) line 263 reads

       elif ((qw >= deficit) and
             ait((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx))):

   `ait` is a float at this point in the loop; calling it as a function
   raises `TypeError: 'float' object is not callable` whenever this branch
   is reached. Cross-checked against the surrounding NWS SNOW-17 ripeness
   logic (Anderson, 1973) and the other two branches of the same if/elif/else:
   the clear intent is a plain boolean AND of two conditions, i.e. the stray
   `ait(` should just be `(`. Fixed below; no other line of physics changed.

2. INSTRUMENTATION (additive only): the upstream function returns only
   `model_swe` and `outflow`. This diagnostic needs the five internal state
   variables (ait, w_qx, w_q, w_i, deficit) at every timestep, not just the
   final SWE, so the function below returns those as five additional arrays.
   No physics/parameter/control-flow line is changed beyond fix (1) above.
"""
from __future__ import print_function, division
import numpy as np


def snow17_instrumented(time, prec, tair, lat=50, elevation=0, dt=24, scf=1.0, rvs=1,
                         uadj=0.04, mbase=1.0, mfmax=1.05, mfmin=0.6, tipm=0.1, nmf=0.15,
                         plwhc=0.04, pxtemp=1.0, pxtemp1=-1.0, pxtemp2=3.0):
    """Same signature/physics as tonic.models.snow17.snow17.snow17, plus
    per-timestep internal state trajectories in the return tuple."""

    time = np.asarray(time)
    prec = np.asarray(prec)
    tair = np.asarray(tair)
    assert time.shape == prec.shape == tair.shape

    ait = 0.0
    w_qx = 0.0
    w_q = 0.0
    w_i = 0.0
    deficit = 0.0

    nsteps = len(time)
    model_swe = np.zeros(nsteps)
    outflow = np.zeros(nsteps)
    ait_out = np.zeros(nsteps)
    w_qx_out = np.zeros(nsteps)
    w_q_out = np.zeros(nsteps)
    w_i_out = np.zeros(nsteps)
    deficit_out = np.zeros(nsteps)

    stefan = 6.12 * (10 ** (-10))
    p_atm = 33.86 * (29.9 - (0.335 * elevation / 100) +
                     (0.00022 * ((elevation / 100) ** 2.4)))

    transitionx = [pxtemp1, pxtemp2]
    transitiony = [1.0, 0.0]

    tipm_dt = 1.0 - ((1.0 - tipm) ** (dt / 6))

    for i, t in enumerate(time):
        mf = melt_function(t, dt, lat, mfmax, mfmin)

        t_air_mean = tair[i]
        precip = prec[i]

        if rvs == 0:
            if t_air_mean <= pxtemp:
                fracsnow = 1.0
            else:
                fracsnow = 0.0
        elif rvs == 1:
            if t_air_mean <= pxtemp1:
                fracsnow = 1.0
            elif t_air_mean >= pxtemp2:
                fracsnow = 0.0
            else:
                fracsnow = np.interp(t_air_mean, transitionx, transitiony)
        elif rvs == 2:
            fracsnow = 1.0
        else:
            raise ValueError('Invalid rain vs snow option')

        fracrain = 1.0 - fracsnow

        pn = precip * fracsnow * scf
        w_i += pn
        e = 0.0
        rain = fracrain * precip

        if t_air_mean < 0.0:
            t_snow_new = t_air_mean
            delta_hd_snow = - (t_snow_new * pn) / (80 / 0.5)
            t_rain = pxtemp
        else:
            t_snow_new = 0.0
            delta_hd_snow = 0.0
            t_rain = t_air_mean

        if pn > (1.5 * dt):
            ait = t_snow_new
        else:
            ait = ait + tipm_dt * (t_air_mean - ait)
        if ait > 0:
            ait = 0

        delta_hd_t = nmf * (dt / 6.0) * ((mf) / mfmax) * (ait - t_snow_new)

        e_sat = 2.7489 * (10 ** 8) * np.exp(
            (-4278.63 / (t_air_mean + 242.792)))
        if rain > (0.25 * dt):
            m_ros1 = np.maximum(
                stefan * dt * (((t_air_mean + 273) ** 4) - (273 ** 4)), 0.0)
            m_ros2 = np.maximum((0.0125 * rain * t_rain), 0.0)
            m_ros3 = np.maximum((8.5 * uadj *
                                (dt / 6.0) *
                                (((0.9 * e_sat) - 6.11) +
                                 (0.00057 * p_atm * t_air_mean))),
                                0.0)
            m_ros = m_ros1 + m_ros2 + m_ros3
        else:
            m_ros = 0.0

        if rain <= (0.25 * dt) and (t_air_mean > mbase):
            m_nr = (mf * (t_air_mean - mbase)) + (0.0125 * rain * t_rain)
        else:
            m_nr = 0.0

        melt = m_ros + m_nr
        if melt <= 0:
            melt = 0.0

        if melt < w_i:
            w_i = w_i - melt
        else:
            melt = w_i + w_q
            w_i = 0.0

        qw = melt + rain
        w_qx = plwhc * w_i
        deficit += delta_hd_snow + delta_hd_t

        if deficit < 0:
            deficit = 0.0
        elif deficit > 0.33 * w_i:
            deficit = 0.33 * w_i

        if w_i > 0.0:
            if (qw + w_q) > ((deficit * (1 + plwhc)) + w_qx):
                e = qw + w_q - w_qx - (deficit * (1 + plwhc))
                w_q = w_qx
                w_i = w_i + deficit
                deficit = 0.0
            elif (qw >= deficit) and ((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx)):
                # (bug-fixed condition -- see module docstring item 1)
                e = 0.0
                w_q = w_q + qw - deficit
                w_i = w_i + deficit
                deficit = 0.0
            else:
                e = 0.0
                w_i = w_i + qw
                deficit = deficit - qw
            swe = w_i + w_q
        else:
            e = qw
            swe = 0

        if deficit == 0:
            ait = 0

        model_swe[i] = swe
        outflow[i] = e
        ait_out[i] = ait
        w_qx_out[i] = w_qx
        w_q_out[i] = w_q
        w_i_out[i] = w_i
        deficit_out[i] = deficit

    return model_swe, outflow, ait_out, w_qx_out, w_q_out, w_i_out, deficit_out


def melt_function(t, dt, lat, mfmax, mfmin):
    tt = t.timetuple()
    jday = tt[-2]
    n_mar21 = jday - 80
    days = 365

    sv = (0.5 * np.sin((n_mar21 * 2 * np.pi) / days)) + 0.5
    if lat < 54:
        av = 1.0
    else:
        if jday <= 77 or jday >= 267:
            av = 0.0
        elif jday >= 117 and jday <= 227:
            av = 1.0
        elif jday >= 78 and jday <= 116:
            av = np.interp(jday, [78, 116], [0, 1])
        elif jday >= 228 and jday <= 266:
            av = np.interp(jday, [228, 266], [1, 0])
        else:
            av = 1.0
    meltf = (dt / 6) * ((sv * av * (mfmax - mfmin)) + mfmin)
    return meltf
