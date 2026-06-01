Todo:

1. Validate the COBE2 PCs:  EOF() dot time_series_of_SST, should equals PCs(COBE2)
2. Try to predict the overland temperatures (t2m in wus, it should match each wus model, can compare it with the ERA5 t2m, it should be anomaly), still using WUS dataset, send Paul the EOFs and PCs (from COBE2). Predict monthly avg March t2m given the March 1st sst.



| Model | Correlation | RMSE (K) | Mean Bias (K) | Valid Years |
|-------|-------------|----------|---------------|-------------|
| ec-earth3_r1i1p1f1_2_historical_bc_historical_march_t2_anomalies | 0.1553 | 1.9393 | 0.0000 | 34 |
| miroc6_r1i1p1f1_historical_bc_historical_march_t2_anomalies | 0.2265 | 1.8361 | -0.0000 | 34 |
| mpi-esm1-2-hr_r3i1p1f1_historical_bc_historical_march_t2_anomalies | -0.1662 | 2.5598 | 0.0000 | 34 |
| taiesm1_r1i1p1f1_historical_bc_historical_march_t2_anomalies | 0.0745 | 2.1624 | -0.0000 | 34 |

For the regional-mean March t2m anomaly validation, the workflow was:

1. Load WUS-D3 d01 `t2` for each historical model:

* EC-Earth3
* MIROC6
* MPI-ESM1-2-HR
* TaiESM1

2. Load ERA5-Land `t2m`.

3. Use the WRF `LANDMASK` from:

```text id="c04u7m"
/global/cfs/projectdirs/m3522/cmip6/WUS-D3/wrfinput_d01
```

Keep only:

* `LANDMASK == 1`
* land cells only

4. Regrid ERA5-Land `t2m` onto the native WUS d01 grid.

So:

* WUS grid stays fixed
* ERA5 is interpolated onto WUS

5. For each year:

* compute March monthly mean temperature field

This produces:

* one March mean temperature map per year

for:

* WUS
* ERA5

6. For each dataset separately:

* compute March climatology over all overlapping historical years (1980–2013)

Meaning:

```text id="t3y4r0"
March climatology =
mean of all March temperature fields
```

7. Compute anomalies:

```text id="9gx87v"
March anomaly(year) =
March mean(year)
-
March climatology
```

This was done separately for:

* each WUS model
* ERA5

So the anomalies are relative to each dataset’s own climatology.

8. For each year:

* spatially average the anomaly field over all valid land cells

This converts:

```text id="7ynsxt"
2D anomaly map
```

into:

```text id="0g8j0q"
1 scalar regional-mean anomaly
```

for both:

* WUS
* ERA5

9. Compare the two yearly time series:

```text id="tb3bjt"
WUS regional-mean anomaly(year)
vs
ERA5 regional-mean anomaly(year)
```

over:

* 1980–2013
* 34 years total

10. Compute metrics:

* correlation
* RMSE
* mean bias

The reported values therefore measure:

how similarly the WUS models and ERA5 describe year-to-year March regional temperature anomalies over land.

3. Back to the original SWE, provide the 1 day lead predictivity (0.95 predicitiy expected and then try 1 day, two day, see how quickly it falls off). Make sure the input is SST and SWE. 



For the EOF2/EOF3 mode-mixing check, I focused on the 2×2 spatial correlation block between COBE2 EOF2/EOF3 and WUS EOF2/EOF3. Each value below is the absolute Pearson spatial correlation between two EOF loading maps, after allowing for EOF sign flips.

| Model | corr(COBE2 EOF2, WUS EOF2) | corr(COBE2 EOF2, WUS EOF3) | corr(COBE2 EOF3, WUS EOF2) | corr(COBE2 EOF3, WUS EOF3) |
|---|---:|---:|---:|---:|
| taiesm1_r1i1p1f1_historical_bc | 0.674 | 0.627 | 0.607 | 0.729 |
| ec-earth3_r1i1p1f1_2_historical_bc | 0.722 | 0.206 | 0.564 | 0.603 |
| miroc6_r1i1p1f1_ssp370_bc | 0.691 | 0.245 | 0.530 | 0.729 |
| taiesm1_r1i1p1f1_ssp370_bc | 0.744 | 0.370 | 0.467 | 0.839 |
| mpi-esm1-2-hr_r3i1p1f1_historical_bc | 0.670 | 0.367 | 0.451 | 0.788 |
| ec-earth3_r1i1p1f1_2_ssp370_bc | 0.719 | 0.162 | 0.456 | 0.700 |
| miroc6_r1i1p1f1_historical_bc | 0.739 | 0.348 | 0.440 | 0.863 |
| mpi-esm1-2-hr_r3i1p1f1_ssp370_bc | 0.724 | 0.208 | 0.310 | 0.870 |

The diagonal correlations, corr(COBE2 EOF2, WUS EOF2) and corr(COBE2 EOF3, WUS EOF3), measure the same-numbered EOF agreement. The off-diagonal correlations, corr(COBE2 EOF2, WUS EOF3) and corr(COBE2 EOF3, WUS EOF2), measure possible EOF2/EOF3 mixing.

The table does not show a simple EOF2/EOF3 swap, because the diagonal correlations are still generally larger than the off-diagonal correlations. However, some models have relatively large off-diagonal correlations, which indicates partial EOF2/EOF3 mixing. The strongest case is taiesm1_r1i1p1f1_historical_bc, where all four correlations in the EOF2/EOF3 block are relatively close. ec-earth3_r1i1p1f1_2_historical_bc and miroc6_r1i1p1f1_ssp370_bc also show notable mixing, especially through the corr(COBE2 EOF3, WUS EOF2) term.