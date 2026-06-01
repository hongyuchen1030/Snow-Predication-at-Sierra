1. CURRENT CODE AUDIT

  - Reuse as-is
  - The 66-channel date-conditioned feature layout stayed unchanged.
  - The existing non-SWE loading and interpolation path stayed intact conceptually; it now just targets the coarsened regional
    SWE grid.
  - The UCLA target choice stayed unchanged: SWE_Post, Stats index 0 (mean).
  - Modify
  - src/snow_ml/data.py:84 was the right place to add coarsening because it already owned the SWE grid definition.
  - src/snow_ml/features.py:110 was the right place to switch sample building from binary masks to fractional masks and to
    carry coarse-grid metadata.
  - scripts/preprocess_training_data.py:31, scripts/train_unet.py:28, scripts/run_inference_and_postprocess.py:23, and
    scripts/inspect_sierra_mask.py:28 needed CLI and metadata updates for --coarsen-factor.
  - Replace
  - The prior “cropped regional SWE grid = final model grid” assumption was replaced by “crop first, trim, coarsen, then use
    the coarsened SWE grid as the model grid.”
  - Binary saved masks were replaced with fractional masks.
  - Sierra post-processing moved from boolean subset statistics to weighted sums/means.

  2. FILES CHANGED

  - src/snow_ml/data.py:54
      - Added DEFAULT_COARSEN_FACTOR=8, coarsen_factor in ForecastConfig, expanded SweGridDefinition to carry cropped/trimmed/
        coarse shapes plus fine and coarse coordinates, and added crop-trim-coarsen helpers.
      - Why: this is now the single shared regional/coarsened grid API.
  - src/snow_ml/data.py:259
      - get_swe_grid_definition(...) now does: crop regional SWE grid, trim edges to divisibility, coarsen coordinates by
        block mean, and record metadata.
      - Why: the model grid must be the coarsened cropped SWE grid.
  - src/snow_ml/data.py:325
      - SWE snapshots now subset on the fine cropped SWE region and coarsen by mean for SWE fields.
      - Why: SWE inputs/targets must live on the coarse model grid.
  - src/snow_ml/data.py:369
      - build_valid_swe_mask(...) now builds fine-grid finite coverage then coarsens by block average, producing fractional
        weights.
      - Why: this matches the exact mask construction you specified.
  - src/snow_ml/data.py:479
      - build_sierra_mask(...) now builds the fine-grid Sierra box mask then coarsens by block-average.
      - Why: Sierra mask is now fractional area coverage.
  - src/snow_ml/features.py:110
      - Sample building now uses get_regional_swe_grid_definition(..., coarsen_factor), constructs coarse target plus
        fractional masks, and saves crop/trim/coarse metadata.
      - Why: one sample is now indexed by (water_year, target_mmdd, region, coarsen_factor).
  - src/snow_ml/train.py:88
      - Training already had masked MSE from the last pass; it now naturally uses the saved fractional valid_mask.
      - Why: your loss formula is unchanged and already correct for fractional weights.
  - scripts/preprocess_training_data.py:50
      - Added --coarsen-factor, switched grid diagnostics to cropped/trimmed/coarsened shapes, and saved fractional masks plus
        coarsening metadata.
      - Why: preprocessing defines the final model domain and mask format.
  - scripts/train_unet.py:50
      - Added coarsen-factor consistency checking and saved coarse-grid metadata in training_summary.json.
      - Why: prevent mixing artifacts with different spatial resolutions.
  - scripts/run_inference_and_postprocess.py:117
      - Inference now treats both masks as floats and computes Sierra weighted totals/means with the fractional Sierra mask.
      - Why: post-processing now follows the specified area-weighted design.
  - scripts/inspect_sierra_mask.py:36
      - Inspection now reports cropped/trimmed/coarsened shapes, coarsen factor, and fractional mask summaries.
      - Why: this is the simplest validation step for the new grid path.

  3. COARSENING API

  - Region bounds are still specified exactly as before through:
      - --lat-min
      - --lat-max
      - --lon-min
      - --lon-max
  - New coarsening option:
      - --coarsen-factor
      - default: 8
  - Shared API location:
      - RegionBounds and ForecastConfig in src/snow_ml/data.py:54
      - add_region_args(...) now also adds --coarsen-factor in src/snow_ml/data.py:114
  - Grid construction order:
      1. Crop SWE grid to requested regional box
      2. Trim edge rows/cols to divisibility by coarsen_factor
      3. Coarsen trimmed SWE grid by block mean
      4. Use that coarsened grid as the common model grid
  - Saved metadata now includes:
      - requested_region
      - effective_region
      - original_cropped_grid_shape
      - trimmed_grid_shape
      - coarsened_grid_shape
      - coarsen_factor
      - mask_type = fractional
      - trimming_policy = trim
  - Default-region expected shape at factor 8, verified on the real SWE file:
      - original cropped regional grid: 2363 x 2476
      - trimmed regional grid: 2360 x 2472
      - coarsened grid: 295 x 309

  4. FINAL PIPELINE SUMMARY

  - The common model grid is now the coarsened cropped SWE grid, not the raw cropped SWE grid.
  - Non-SWE fields SST, t2m, tp, and terrain are interpolated directly onto that coarse SWE grid.
  - SWE-derived fields Dec 31 SWE, target SWE, valid mask, and Sierra mask start on the fine cropped SWE grid, then are
    coarsened.
  - The 66-channel date-conditioned input design is unchanged.
  - Training uses fractional valid-mask-weighted MSE:
      - (((pred - target) ** 2) * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
  - Sierra post-processing uses fractional Sierra-mask weights:
      - total: (field * sierra_mask).sum()
      - mean: (field * sierra_mask).sum() / sierra_mask.sum().clamp_min(1.0)

  5. RUN ORDER

      1. Inspect regional coarse grid and fractional masks

  module purge
  module load python
  module load conda
  conda activate uxarray_build
  cd ~/Snow-Predication-at-Sierra
  PYTHONPATH=src:. python scripts/inspect_sierra_mask.py \
    --water-year 2014 \
    --target-mmdd 04-01 \
    --lat-min 32.5 \
    --lat-max 43.0 \
    --lon-min -134.5 \
    --lon-max -114.0 \
    --coarsen-factor 8

  - Output:
      - artifacts/mask_inspection/sierra_mask_summary.json
  - Expected shapes for default region/factor 8:
      - cropped: 2363 x 2476
      - trimmed: 2360 x 2472
      - coarse: 295 x 309
      2. Preprocess

  module purge
  module load python
  module load conda
  conda activate uxarray_build
  cd ~/Snow-Predication-at-Sierra
  PYTHONPATH=src:. python scripts/preprocess_training_data.py \
    --years 2017 \
    --target-mmdd 04-01 \
    --lat-min 32.5 \
    --lat-max 43.0 \
    --lon-min -134.5 \
    --lon-max -114.0 \
    --coarsen-factor 1
    --history-years 2

  - Output:
      - artifacts/preprocessed/grid_and_mask.npz
      - artifacts/preprocessed/grid_and_mask_summary.json
      - artifacts/preprocessed/wy<year>_<mmdd>.npz
  - Expected sample shapes:
      - inputs = (66, 295, 309)
      - target = (1, 295, 309)
      - valid_mask = (1, 295, 309)
      - sierra_mask = (1, 295, 309)
      3. Train

  module purge
  module load pytorch
  cd ~/Snow-Predication-at-Sierra
  PYTHONPATH=src:. python scripts/train_unet.py \
    --input-dir artifacts/preprocessed \
    --output-dir artifacts/training \
    --epochs 5 \
    --batch-size 1 \
    --lat-min 32.5 \
    --lat-max 43.0 \
    --lon-min -134.5 \
    --lon-max -114.0 \
    --coarsen-factor 1

  - Output:
      - artifacts/training/last_checkpoint.pt
      - artifacts/training/loss_history.csv
      - artifacts/training/split.json
      - artifacts/training/training_summary.json
      4. Inference/post-processing

  module purge
  module load pytorch
  cd ~/Snow-Predication-at-Sierra
  PYTHONPATH=src:. python scripts/run_inference_and_postprocess.py \
    --input-dir artifacts/preprocessed \
    --training-dir artifacts/training \
    --output-dir artifacts/inference \
    --lat-min 32.5 \
    --lat-max 43.0 \
    --lon-min -134.5 \
    --lon-max -114.0 \
    --coarsen-factor 8

  - Output:
      - artifacts/inference/per_sample_metrics.csv
      - artifacts/inference/target_month_day_summary.csv
      - artifacts/inference/inference_summary.json

  6. ASSUMPTIONS / OPEN CHOICES

  - The requested Pacific extent still exceeds the actual SWE longitude coverage, so effective_region is clipped by the SWE
    file extent. For the verified default request, effective coarse-grid support starts at lon=-125.0.
  - Coarsening for SWE maps uses block mean. I kept that consistent for target SWE and Dec 31 SWE.
  - Fractional masks are not thresholded anywhere by default.
  - Verified here:
      - syntax checks passed
      - real-data inspection passed
      - default region/factor-8 coarse shape verified as 295 x 309
  - Not run here:
      - full torch training/inference, because torch is still not importable in the current inspection runtime.





TODO:
- Try the PCA model and learn the coefficients of different vectors, and see how the SWE is spatially or dimensionally related.Do a dimensional reduction first to avoid overfitting because we dont have too much training data.

- Whether or not we can predict heavy snow year, normal snow year, less snow year. Need to label the training data as well. 


