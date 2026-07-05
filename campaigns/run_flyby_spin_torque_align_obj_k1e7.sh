#!/usr/bin/env bash
# OBJ DEM: fixed k_c and spin period; sweep torque-align vs flyby orbit angular momentum.
# See campaigns/flyby_spin_torque_align_obj_k1e7.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

_torque_align_sym=$(
  strings sobol/phantomsetup 2>/dev/null | grep -c apophis_spin_torque_align_deg || true
)
if [[ "${_torque_align_sym:-0}" -lt 1 ]]; then
  echo "[ERROR] phantomsetup lacks apophis_spin_torque_align_deg — rebuild: cd sobol && make setup && make" >&2
  exit 1
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
  echo "[INFO] Dry-run mode (no phantom execution)"
fi

LOG="${LOG:-sobol_mass_runs/flyby_spin_torque_align_obj_k1e7_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting flyby_spin_torque_align_obj_k1e7 at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --num-samples 24 \
  --seed 43 \
  --batch-label flyby_spin_torque_align_obj_k1e7 \
  --use-dem-fixed true \
  --np-apophis 500 \
  --use-shape-crop-fixed true \
  --kc-fixed 1e7 \
  --spin-period-fixed 2.05 \
  --spin-torque-align-min 0 \
  --spin-torque-align-max 180 \
  --tmax-hours 108 \
  --dtmax-hours 0.5 \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  "${DRY_RUN_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
