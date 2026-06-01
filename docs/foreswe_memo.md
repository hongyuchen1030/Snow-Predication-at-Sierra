For future runs, set outputs/checkpoints to $SCRATCH, or symlink:

rm -rf ~/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/outputs
rm -rf ~/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/checkpoints

ln -s $SCRATCH/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/outputs \
      ~/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/outputs

ln -s $SCRATCH/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/checkpoints \
      ~/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/checkpoints