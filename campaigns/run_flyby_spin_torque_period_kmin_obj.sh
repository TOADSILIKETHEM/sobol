#!/usr/bin/env bash
# Flyby OBJ DEM: k_c=3.5e6 (minimal cohesion), 2D sweep spin period × torque-align angle.
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

LOG="${LOG:-sobol_mass_runs/flyby_spin_torque_period_kmin_obj_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting flyby_spin_torque_period_kmin_obj at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --num-samples 64 \
  --seed 44 \
  --batch-label flyby_spin_torque_period_kmin_obj \
  --use-dem-fixed true \
  --np-apophis 500 \
  --use-shape-crop-fixed true \
  --kc-fixed 3.5e6 \
  --spin-period-min 1.3 \
  --spin-period-max 2.5 \
  --spin-torque-align-min 0 \
  --spin-torque-align-max 180 \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  "${DRY_RUN_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
