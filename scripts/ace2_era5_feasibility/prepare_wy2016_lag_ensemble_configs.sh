#!/usr/bin/env bash
set -euo pipefail

ACE2_ROOT="${ACE2_ROOT:-/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2}"
ROOT="${ACE2_ROOT}/lag_ensemble_wy2016"
CONFIG_DIR="${ROOT}/configs"
FORCING_DIR="${ACE2_ROOT}/assets/forcing"
CHECKPOINT="${ACE2_ROOT}/assets/checkpoints/ace2_era5_ckpt.tar"
IC_DIR="${ROOT}/initial_conditions"

mkdir -p "${CONFIG_DIR}"
for member_date in 2015-11-01 2015-11-02 2015-11-03 2015-11-04 2015-11-05; do
  member_tag="$(echo "${member_date}" | tr -d -)"
  experiment_dir="${ROOT}/member_${member_tag}"
  config_path="${CONFIG_DIR}/member_${member_tag}.yaml"
  mkdir -p "${experiment_dir}"
  cat > "${config_path}" <<EOF
experiment_dir: ${experiment_dir}

n_forward_steps: 608

forward_steps_in_memory: 40

checkpoint_path: ${CHECKPOINT}

logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true
  project: ace

initial_condition:
  path: ${IC_DIR}/ic_${member_date}.nc
  start_indices:
    times:
      - "${member_date}T00:00:00"

forcing_loader:
  dataset:
    data_path: ${FORCING_DIR}
  num_data_workers: 2

data_writer:
  save_prediction_files: true
  save_monthly_files: true
  names:
    - PRESsfc
    - surface_temperature
    - TMP2m
    - Q2m
    - UGRD10m
    - VGRD10m
    - air_temperature_0
    - specific_total_water_0
    - eastward_wind_0
    - northward_wind_0
    - air_temperature_1
    - specific_total_water_1
    - eastward_wind_1
    - northward_wind_1
    - air_temperature_2
    - specific_total_water_2
    - eastward_wind_2
    - northward_wind_2
    - air_temperature_3
    - specific_total_water_3
    - eastward_wind_3
    - northward_wind_3
    - air_temperature_4
    - specific_total_water_4
    - eastward_wind_4
    - northward_wind_4
    - air_temperature_5
    - specific_total_water_5
    - eastward_wind_5
    - northward_wind_5
    - air_temperature_6
    - specific_total_water_6
    - eastward_wind_6
    - northward_wind_6
    - air_temperature_7
    - specific_total_water_7
    - eastward_wind_7
    - northward_wind_7
    - PRATEsfc
    - TMP850
    - TMP500
    - TMP200
    - Q850
    - Q500
    - Q200
    - UGRD850
    - VGRD850
    - UGRD500
    - VGRD500
    - UGRD200
    - VGRD200
    - h1000
    - h850
    - h700
    - h500
    - h300
    - h250
    - h200
    - ULWRFtoa
    - USWRFtoa
    - DLWRFsfc
    - DSWRFsfc
    - ULWRFsfc
    - USWRFsfc
    - LHTFLsfc
    - SHTFLsfc
    - DPT2m
    - total_column_water_vapour
    - tendency_of_total_water_path_due_to_advection
EOF
done

printf '%s\n' "${CONFIG_DIR}"/member_*.yaml
