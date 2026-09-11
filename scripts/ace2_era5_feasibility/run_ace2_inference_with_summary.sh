#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <config_path> <output_dir> <summary_json> <summary_txt>" >&2
  exit 2
fi

CONFIG_PATH="$1"
OUTPUT_DIR="$2"
SUMMARY_JSON="$3"
SUMMARY_TXT="$4"

REPO="${REPO:-/global/homes/h/hyvchen/Snow-Predication-at-Sierra}"
ACE2_ROOT="${ACE2_ROOT:-/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2}"
ACE2_RUNTIME="${ACE2_RUNTIME:-${ACE2_ROOT}/runtime}"

cd "${REPO}"
source "${ACE2_RUNTIME}/venv_full/bin/activate"
unset PYTHONPATH
export PIP_CACHE_DIR="${ACE2_RUNTIME}/pip_cache"
export TMPDIR="${ACE2_RUNTIME}/tmp"
export MPLCONFIGDIR="${ACE2_RUNTIME}/mpl_cache"
export XDG_CACHE_HOME="${ACE2_RUNTIME}/xdg_cache"

mkdir -p "$(dirname "${SUMMARY_JSON}")" "$(dirname "${SUMMARY_TXT}")"

START_EPOCH="$(date -u +%s)"
START_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
HOSTNAME_NOW="$(hostname)"
SLURM_ID="${SLURM_JOB_ID:-}"

set +e
python -m fme.ace.validate_config "${CONFIG_PATH}" --config_type inference
VALIDATE_EXIT=$?
if [[ ${VALIDATE_EXIT} -eq 0 ]]; then
  python -m fme.ace.inference "${CONFIG_PATH}"
  RUN_EXIT=$?
else
  RUN_EXIT=${VALIDATE_EXIT}
fi
set -e

END_EPOCH="$(date -u +%s)"
END_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
WALL_SECONDS="$((END_EPOCH - START_EPOCH))"

python - "$CONFIG_PATH" "$OUTPUT_DIR" "$SUMMARY_JSON" "$SUMMARY_TXT" "$START_ISO" "$END_ISO" "$WALL_SECONDS" "$HOSTNAME_NOW" "$SLURM_ID" "$RUN_EXIT" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
summary_txt = Path(sys.argv[4])
start_iso = sys.argv[5]
end_iso = sys.argv[6]
wall_seconds = int(sys.argv[7])
hostname_now = sys.argv[8]
slurm_id = sys.argv[9]
run_exit = int(sys.argv[10])

inventory = []
total_size = 0
if output_dir.exists():
    for child in sorted(output_dir.iterdir()):
        if child.is_file() or child.is_symlink():
            size = child.stat().st_size
            inventory.append({"name": child.name, "size_bytes": size})
            total_size += size

inference_log = output_dir / "inference_out.log"
steps_per_second = None
inference_duration_seconds = None
if inference_log.exists():
    text = inference_log.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "steps/s" in line or "steps/second" in line:
            tokens = line.replace(",", " ").split()
            for idx, token in enumerate(tokens):
                if token in {"steps/s", "steps/second"} and idx > 0:
                    try:
                        steps_per_second = float(tokens[idx - 1])
                    except ValueError:
                        pass
        if "Total inference duration:" in line or "inference duration:" in line:
            try:
                inference_duration_seconds = float(line.rsplit(":", 1)[1].strip().split()[0])
            except Exception:
                pass

summary = {
    "config_path": str(config_path),
    "output_dir": str(output_dir),
    "slurm_job_id": slurm_id or None,
    "node": hostname_now,
    "start_time_utc": start_iso,
    "end_time_utc": end_iso,
    "wall_seconds": wall_seconds,
    "exit_code": run_exit,
    "status": "success" if run_exit == 0 else "failed",
    "output_file_inventory": inventory,
    "total_output_size_bytes": total_size,
    "inference_duration_seconds": inference_duration_seconds,
    "steps_per_second": steps_per_second,
}

summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    f"config_path: {summary['config_path']}",
    f"output_dir: {summary['output_dir']}",
    f"slurm_job_id: {summary['slurm_job_id']}",
    f"node: {summary['node']}",
    f"start_time_utc: {summary['start_time_utc']}",
    f"end_time_utc: {summary['end_time_utc']}",
    f"wall_seconds: {summary['wall_seconds']}",
    f"exit_code: {summary['exit_code']}",
    f"status: {summary['status']}",
    f"inference_duration_seconds: {summary['inference_duration_seconds']}",
    f"steps_per_second: {summary['steps_per_second']}",
    f"total_output_size_bytes: {summary['total_output_size_bytes']}",
    "output_file_inventory:",
]
for entry in inventory:
    lines.append(f"  - {entry['name']}: {entry['size_bytes']}")
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

exit "${RUN_EXIT}"
