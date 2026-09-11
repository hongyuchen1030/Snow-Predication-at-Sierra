# TaiESM1 Daily SWE Training-Month Diagnostic

**Status: target-side diagnostic only.** No predictor arrays touched, no retraining. Source: `artifacts/taiesm1_daily_sierra_swe.npz` (43,800 daily records, 120 water years, 365_day calendar, mm).

## 1. Monthly 1-day pair statistics (Sep-Mar)

| Month | N_all | N_zero_zero | N_active | Active % | SWE(t)>0 % | SWE(t)>1mm % | mean SWE(t) | median SWE(t) | mean\|ΔSWE\| | median\|ΔSWE\| | std ΔSWE | Δ>0 % | Δ<0 % | Δ=0 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sep | 3600 | 3378 | 222 | 6.2 | 3.9 | 0.0 | 0.00 | 0.00 | 0.000 | 0.000 | 0.007 | 3.1 | 3.1 | 93.8 |
| Oct | 3720 | 2050 | 1670 | 44.9 | 38.1 | 2.6 | 0.12 | 0.00 | 0.044 | 0.000 | 0.287 | 18.5 | 26.4 | 55.1 |
| Nov | 3600 | 321 | 3279 | 91.1 | 88.1 | 35.4 | 2.60 | 0.29 | 0.379 | 0.057 | 0.983 | 39.2 | 51.8 | 8.9 |
| Dec | 3720 | 28 | 3692 | 99.2 | 98.8 | 83.7 | 14.66 | 8.05 | 1.012 | 0.293 | 2.007 | 45.2 | 54.1 | 0.8 |
| Jan | 3720 | 0 | 3720 | 100.0 | 100.0 | 97.6 | 33.89 | 26.96 | 1.074 | 0.485 | 1.898 | 42.4 | 57.6 | 0.0 |
| Feb | 3360 | 0 | 3360 | 100.0 | 100.0 | 99.3 | 47.49 | 39.73 | 1.249 | 0.653 | 2.073 | 41.5 | 58.5 | 0.0 |
| Mar | 3720 | 0 | 3720 | 100.0 | 100.0 | 99.0 | 53.54 | 46.99 | 0.939 | 0.640 | 1.410 | 31.4 | 68.6 | 0.0 |

## 2. Snow-onset distribution by water year (120 WY)

### SWE > 0

- earliest Sep 1, p5 Sep 5, p25 Sep 24, median Oct 6, p75 Oct 18, p95 Oct 30, latest Nov 16
- month counts: Sep=44, Oct=71, Nov=5, Dec=0, Jan-or-later=0 (n_with_onset=120, never=0)

### SWE > 1mm

- earliest Oct 12, p5 Oct 18, p25 Nov 5, median Nov 13, p75 Nov 28, p95 Dec 24, latest Jan 12
- month counts: Sep=0, Oct=21, Nov=76, Dec=19, Jan-or-later=4 (n_with_onset=120, never=0)

### SWE > 5mm

- earliest Oct 25, p5 Nov 9, p25 Nov 20, median Dec 4, p75 Dec 17, p95 Jan 9, latest Feb 6
- month counts: Sep=0, Oct=2, Nov=49, Dec=54, Jan-or-later=15 (n_with_onset=120, never=0)

## 3. Persistent snow-onset (SWE>1mm AND >=5 of next 7 days also >1mm)

- earliest Oct 12, p5 Oct 24, p25 Nov 11, median Nov 19, p75 Dec 2, p95 Dec 30, latest Jan 12
- month counts: Sep=0, Oct=8, Nov=81, Dec=26, Jan-or-later=5 (n_with_onset=120, never=0)

## 4. Candidate start-month comparison (all end Apr 1)

| Candidate | Pair N_all | Pair N_active | Pair N_zero_zero | Pair active % | 7-day windows (all) | 7-day windows (snow_active) |
|---|---:|---:|---:|---:|---:|---:|
| Sep1->Apr1 | 25440 | 19663 | 5777 | 77.3 | 25469 | 19692 |
| Oct1->Apr1 | 21840 | 19441 | 2399 | 89.0 | 21869 | 19470 |
| Nov1->Apr1 | 18120 | 17771 | 349 | 98.1 | 18149 | 17800 |
| Dec1->Apr1 | 14520 | 14492 | 28 | 99.8 | 14549 | 14521 |

## 5. Information lost by later starts

| Candidate start | Onset definition | WY with onset BEFORE start | % of 120 WY |
|---|---|---:|---:|
| Oct1 | first SWE>1mm | 0 | 0.0% |
| Oct1 | persistent SWE>1mm | 0 | 0.0% |
| Nov1 | first SWE>1mm | 21 | 17.5% |
| Nov1 | persistent SWE>1mm | 8 | 6.7% |
| Dec1 | first SWE>1mm | 97 | 80.8% |
| Dec1 | persistent SWE>1mm | 89 | 74.2% |

![monthly active/zero-zero and onset histogram](../artifacts/taiesm1_swe_onset_training_month_diagnostic.png)

## 6. Final recommendation

```
Recommended start:        OCT 1
Recommended sample mode:  all

Total 7-day windows:      21869
Snow-active 7-day windows: 19470

% zero->zero removed:     11.0%
% water years with onset before chosen start: 0.0%
```

**Main justification (start month):** using a real onset threshold (>1mm, persistent) rather than just "is SWE usually zero," **0 of 120 water years** have persistent onset before October 1 -- September genuinely contributes no onset physics (its 6.2% "active" fraction is almost entirely SWE(t)>0 trivial/transient single-day noise, not real accumulation). November 1 would silently drop the onset period for 6.7-17.5% of water years, and December 1 for 74-81% of water years -- both directly contradict retaining onset physics. October 1 is therefore the earliest start that loses zero water years' onset information while avoiding September's dead weight.

**Main justification (sample mode):** at the recommended Oct1 start, zero->zero pairs are only 11.0% of samples -- a modest fraction, not "flooding." Zero->zero pairs are legitimate negative examples (the model must learn when NOT to add snow, e.g. during dry spells or before/after the snow season), so discarding them risks biasing the model away from realistic zero-SWE conditions rather than improving it. `all` is recommended; `snow_active` remains available as a configurable option (both counts are reported above) if a future run shows the model is dominated by trivial zero predictions.
