# Baseline for the Best Ridge Regression Model

The best predictive performance for April 1 Sierra Nevada SWE is obtained using the seven-column baseline. This predictor set consists of two “oracle” columns selected using LOD modes from the Pacific SST region and five columns selected through a greedy search over North Atlantic SST principal components. Each row represents one water year, resulting in a $37 \times 7$ predictor matrix. The definitions of these predictors and the corresponding experimental setup are described below.

## Pacific LOD Predictors

For the LOD component, we define the oracle predictors $Z_1$ and $Z_2$ as the first two SST columns selected by applying LOD to the complete 37-year dataset.

The Pacific SST domain is defined using COBE2 data over:

* $10^\circ\text{S}–60^\circ\text{N}$
* $120^\circ\text{E}–280^\circ\text{E}$ in the $0^\circ–360^\circ$ longitude convention

$Z_1$ is the column selected by the full-sample first LOD mode, and $Z_2$ is the column selected by the full-sample second LOD mode. We focus on the first two modes because they are the only LOD modes that exhibit meaningful location and month stability across the leave-one-year-out (LOYO) folds.


Z1 = mode 1, month Jan, latitude -9.5, longitude 133.5
Z2 = mode 2, month Oct, latitude 0.5, longitude 136.5

## North Atlantic PCA Predictors

PCA is applied to COBE2 SST within the North Atlantic domain:

* $0^\circ–70^\circ\text{N}$
* $80^\circ\text{W}–0^\circ$

This domain is referred to as the AMO region in [this paper](https://doi.org/10.1038/s43247-024-01594-2).

The complete PC1–PC6 predictor block from September through March provides useful standalone predictability for April 1 Sierra Nevada SWE. Because this full block contains 42 predictors, we tested whether a smaller subset could retain most of its predictive signal.

The reduced five-predictor AMV/AMO core, referred to as K5, is:

{
  AMV_PC4_Sep,
  AMV_PC5_Feb,
  AMV_PC2_Feb,
  AMV_PC4_Nov,
  AMV_PC5_Mar
}

Baseline Results

![Three-panel scatter plot for the Z1, Z2, and AMV K5 baseline](../artifacts/cobe2_sierra_swe_lod_setup/z1z2_plus_amv_k5_loyo/z1z2_amv_k5_scatter_three_panel.png)

![Observed versus predicted April 1 Sierra Nevada SWE for the seven-column baseline](../artifacts/cobe2_sierra_swe_lod_setup/z1z2_plus_amv_k5_loyo/z1z2_amv_k5_observed_vs_predicted.png)




# SST AutoEncoder Work for Pacific SSTs


## AutoEncoder from the scratch 

### Raw Pacific-Region SST AutoEncoder

We first tried a simple autoencoder on the Pacific SST region.

- source full SST predictor file used: `/global/cfs/projectdirs/m3522/datalake/COBE2/sst.mon.mean.nc`
- source SWE target file used: `/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/z1z2_plus_amv_k5_loyo/z1z2_amv_k5_predictor_table.csv`
- upstream SWE target provenance: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/targets/sierra_swe_apr1_anomaly_standardized_wy1985_2021.nc`
- baseline artifact path: `/global/u1/h/hyvchen/Snow-Predication-at-Sierra/artifacts/cobe2_sierra_swe_lod_setup/z1z2_plus_amv_k5_loyo`
- exact leakage rule: the held-out water year is excluded from SST anomaly construction, SST standardization, DAE training, DAE validation, DAE early stopping, ridge fitting, ridge lambda selection, and any seed-level aggregation decisions; it is only passed once through the frozen encoder and outer-fold ridge predictor.
- DAE architecture: `P -> 64 -> k -> 64 -> P` with `tanh` activations and a linear output layer.
- denoising corruption rule: `gaussian` with Gaussian noise std `0.05` and mask probability `0.0`
- ridge lambda selection rule: nested training-only inner LOYO over `[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]` with lower-alpha tie-breaks
- final baseline RMSE: `0.022516`
- mean delta squared error for `k=7`: `0.000563`
- mean delta squared error for `k=10`: `0.000593`
- final DAE RMSE for `k=15`: `0.033012`
- mean delta squared error for `k=15`: `0.000583`
- best `k` by full-37 reconstruction: `10`
- one-fold `k=7`: train MSE `0.012481`, held-out MSE `0.679131`, gap `0.666650`
- one-fold `k=10`: train MSE `0.010869`, held-out MSE `0.812124`, gap `0.801255`
- one-fold `k=15`: train MSE `0.000160`, held-out MSE `0.782250`, gap `0.782090`
- one-fold `k=30`: train MSE `0.002327`, held-out MSE `0.796444`, gap `0.794117`

The direct-SST autoencoder has enough capacity to reconstruct, or even memorize, the available Pacific SST training years, but that reconstruction does not generalize to the held-out year. Increasing the latent dimension improves training reconstruction without improving held-out reconstruction. This suggests that the main limitation is severe generalization failure caused by the very small number of independent years, rather than insufficient autoencoder capacity.

### Run the AutoEncoder on Climate-Mode PCs

Because the raw-SST autoencoder suffers from severe overfitting, we next asked a narrower question: can a nonlinear low-dimensional latent representation reconstruct the combined yearly vector of Pacific PCs, Niño 3.4, and Atlantic PCs for an unseen year?

The input climate-mode blocks were:

| Mode / index | Variable | Domain | Method | Expected predictor columns | Status |
| --- | --- | --- | --- | --- | --- |
| COBE2 Pacific SST modes | SST | $10^\circ\mathrm{S}$--$60^\circ\mathrm{N}$, $120^\circ\mathrm{E}$--$280^\circ\mathrm{E}$ in $0..360$, equivalently $120^\circ\mathrm{E}$--$80^\circ\mathrm{W}$ | EOF1--EOF6 / PC1--PC6 from the existing COBE2 broad Pacific SST EOF workflow | Sep--Mar Pacific SST PC1--PC6 values. These modes may contain basin-wide warming, ENSO-like variability, and PDO-like variability, but those interpretations should only be assigned after inspecting the EOF maps and PC correlations. | Existing artifacts; reused |
| Niño 3.4 | SST | $5^\circ\mathrm{S}$--$5^\circ\mathrm{N}$, $190^\circ\mathrm{E}$--$240^\circ\mathrm{E}$ in $0..360$, equivalently $170^\circ\mathrm{W}$--$120^\circ\mathrm{W}$ | Area-weighted domain average of monthly SST anomalies | `Nino34_Sep`, `Nino34_Oct`, `Nino34_Nov`, `Nino34_Dec`, `Nino34_Jan`, `Nino34_Feb`, `Nino34_Mar` | Generated |
| AMV / AMO | SST | $0^\circ$--$70^\circ\mathrm{N}$, $280^\circ\mathrm{E}$--$360^\circ\mathrm{E}$ in $0..360$, equivalently $80^\circ\mathrm{W}$--$0^\circ$ | Separate EOF1--EOF6 / PC1--PC6 analysis over the North Atlantic SST anomaly field; this does not reuse the Pacific EOF basis | Sep--Mar AMV/AMO PC1--PC6 values | Generated |

This produces a predictor table with shape $37 \times 91$.

The autoencoder used here was `pyod.models.auto_encoder.AutoEncoder` from `pyod` (`pyod_version=3.6.1`, `torch_version=2.12.1`) with the following settings: hidden layers `[32, 16, k]`, `k \in \{3, 5, 7, 10\}`, ReLU activation, dropout `{0.05, 0.1, 0.2}`, weight decay `{1e-5, 1e-4, 1e-3}`, 200 epochs, batch size `36`, learning rate `1e-3`, and contamination `0.1`.

The AE extracted a limited amount of common nonlinear or redundant structure from the combined climate-mode table, but it did not learn a compact latent representation that reliably reconstructs the detailed PC-month state of an unseen year.

### Add Supervision to the Loss Term

We then turned the model into a supervised autoencoder.

The conventional autoencoder loss minimizes only reconstruction error. The supervised autoencoder adds an SWE-prediction loss:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{reconstruction}}
+
\lambda_{\mathrm{SWE}}
\frac{1}{N}
\sum_t
\left(y_t-\widehat{y}_t\right)^2.
$$

Equivalently,

$$
\boxed{
\mathcal{L}
=
\operatorname{MSE}(X,\widehat{X})
+
\lambda_{\mathrm{SWE}}
\operatorname{MSE}(y,\widehat{y})
}
$$

The best supervised model used:

$$
k=5,\qquad
\lambda_{\mathrm{SWE}}=0.03,
$$

with dropout `0.05` and weight decay $10^{-4}$.

Its LOYO SWE performance was:

$$
\mathrm{RMSE}=0.031392,
\qquad
r=0.287,
\qquad
\text{sign accuracy}=0.568.
$$

The previous unsupervised AE latent model had:

$$
\mathrm{RMSE}=0.031903,
\qquad
r=0.097,
\qquad
\text{sign accuracy}=0.486.
$$

So adding the SWE loss produced a modest but real improvement, especially in correlation:

$$
r: 0.097 \rightarrow 0.287.
$$

This indicates that the supervised term successfully changed, or "rotated," the latent representation toward directions more related to SWE. The dominant directions needed to reconstruct the full climate-mode table were not automatically the directions most useful for SWE prediction.

![Best supervised AE observed-versus-predicted scatter](../artifacts/ocean_mode_supervised_ae_loyo/supervised_ae_best_scatter_obs_vs_pred.png)

![Best supervised AE time series](../artifacts/ocean_mode_supervised_ae_loyo/supervised_ae_best_timeseries.png)

What we have so far:

| Model | RMSE | $R^2$ | Sign accuracy |
| --- | ---: | ---: | ---: |
| 7-column baseline | 0.022750 | 0.496 | 0.784 |
| Supervised AE on climate-mode inputs | 0.031392 | 0.020 | 0.568 |
| Unsupervised AE on climate-mode inputs | 0.031903 | -0.012 | 0.486 |

In short,

$$
\text{best climate-state reconstruction}
\neq
\text{best SWE prediction}.
$$


## The KGAE: Knowledge Guided AutoEncoder

This is the pretrained AutoEncoder from the paper from https://arxiv.org/pdf/2508.08490

The pretrained Knowledge-Guided AutoEncoder was developed to separate monthly Pacific SST variability across different timescales. Its custom loss combines SST reconstruction with spectral and spatial constraints that encourage distinct latent modes. The paper interprets three dominant modes as decadal PDO/NPMM-like variability, interannual ENSO-like variability, and quasibiennial ENSO/TBO-like variability.

The input is stacked in Month: each row represent a month, And tried to directly use their existed latents to test:1. if it has good predictability withe April 1st SWE using the ridge regression models.
/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_more_latents_swe_loyo/kgae_more_latents_best_timeseries.png

/global/homes/h/hyvchen/Snow-Predication-at-Sierra/artifacts/kgae_more_latents_swe_loyo/kgae_more_latents_best_scatter.png


We applied the pretrained encoder to our Pacific SST data and tested its latent coordinates as predictors of April 1 Sierra SWE using strict-LOYO ridge regression. The pretrained KGAE latents showed no useful out-of-sample SWE predictability.

The KGAE latents were correlated with some of the Pacific PCA-derived climate variability (January Pacific PC6), confirming that they capture recognizable Pacific SST structure. However, they were only weakly correlated with the Pacific PC-month direction that showed the strongest SWE predictability, particularly January Pacific PC6.

Therefore, the dominant timescale-separated Pacific modes learned by the pretrained KGAE do not appear to contain the specific SST direction responsible for our strongest SWE signal. This does not prove that SWE is unrelated to ENSO, PDO, or other Pacific climate modes; it shows only that these pretrained KGAE representations do not recover the SWE-predictive structure identified in our COBE2 analysis.


# SST in Atlantic Ocean


## Atlantic Quadpole Mode (AQM) and SWE Predictability

We shifted attention from the Pacific to the North Atlantic because the Atlantic EOF/PC predictors showed unexpectedly stronger SWE predictability.

Conventional AMV did not explain this result. The independent NOAA ERSSTv5 AMV/AMO index performed poorly:

$$
r = -0.457, \qquad R^2 = -0.165.
$$

None of the successful PC2, PC4, or PC5 patterns was convincingly identifiable as conventional basin-wide AMV.

Instead, the Atlantic PCs appear to be more closely related to the Atlantic Quadpole Mode (AQM) defined in [this paper](https://www.nature.com/articles/s41612-023-00471-7). The reconstructed AQM used here was obtained from HadISST SST and GPCC precipitation through lagged maximum covariance analysis, so it was constructed independently of Sierra SWE.

### PC2 is strongly AQM-like

Among the predictive Atlantic PCs, EOF2 showed the clearest physical identity.

The area-weighted spatial Pearson correlation was calculated as

$$
r_{\mathrm{spatial}}
=
\frac{
\sum_p w_p
\left(E_p-\bar{E}_w\right)
\left(A_p-\bar{A}_w\right)
}{
\sqrt{
\sum_p w_p\left(E_p-\bar{E}_w\right)^2
}
\sqrt{
\sum_p w_p\left(A_p-\bar{A}_w\right)^2
}
},
$$

where:

* $E_p$ is the EOF2 value at grid cell $p$.
* $A_p$ is the AQM pattern value at the same grid cell.
* $w_p = \cos(\phi_p)$ is the latitude-area weight.
* $\bar{E}_w$ and $\bar{A}_w$ are the weighted spatial means.

The reproduced AQM pattern gave

$$
\left|r_{\mathrm{spatial}}\right| = 0.787.
$$

Its temporal coordinate was also related to the independently reconstructed AQM index. The ordinary temporal correlation was

$$
r_{\mathrm{temporal}}
=
\frac{
\sum_t
\left(\mathrm{PC2}_t-\overline{\mathrm{PC2}}\right)
\left(\mathrm{AQM}_t-\overline{\mathrm{AQM}}\right)
}{
\sqrt{
\sum_t
\left(\mathrm{PC2}_t-\overline{\mathrm{PC2}}\right)^2
}
\sqrt{
\sum_t
\left(\mathrm{AQM}_t-\overline{\mathrm{AQM}}\right)^2
}
}.
$$

By contrast, PC4 and PC5 were weaker mixed Atlantic patterns. They contained partial AQM- or NAO-related structure, but they could not be assigned confidently to a single recognized mode.

We then tested the independently defined Stone-style NDJF AQM index as the only predictor of April 1 Sierra SWE. The strict LOYO result was

$$
r = 0.314, \qquad R^2 = 0.097, \qquad \mathrm{RMSE} = 0.0301\ \mathrm{m},
$$

with sign accuracy equal to $0.649$.

![LOYO time series using the NDJF AQM index](../artifacts/aqm_index_loyo/aqm_seasonal_loyo_timeseries.png)

![Observed versus predicted SWE using the NDJF AQM index](../artifacts/aqm_index_loyo/aqm_seasonal_loyo_observed_vs_predicted.png)

Therefore, we are now running the KGAE model on the Atlantic Ocean to test whether it can recover latent structures with strong SWE predictability.


# AI forecase model for percipitations


For ACE2-ERA5, we found that it is technically possible to run an atmospheric simulation under one prescribed SST forcing setup. We mainly tested the provided free year, which appears to be a weak ENSO case. We also flipped the SST anomaly sign to create a weak La Niña-like forcing, but the predicted precipitation did not change very much. So ACE2-ERA5 worked as a feasibility test, but it did not give us a strong SST-forced precipitation contrast in this initial setup.

NeuralGCM was easier to use for this experiment because we could generate stochastic ensembles using different random seeds. I used WY1998, a strong ENSO year, and generated 30 ensemble members under the actual SST forcing. Then I flipped the Pacific SST anomaly sign to create a reversed/La Niña-like counterfactual and generated another 30-member ensemble.

![NeuralGCM Sierra precipitation distribution under actual versus reversed Pacific SST forcing for WY1998](../artifacts/neuralgcm_feasibility/neuralgcm_wy1998_actual_vs_reversed_m030_sierra_distribution.png)
