#!/usr/bin/env bash
# Flyby DEM sweep: fixed k_c=1e7, vary spin period + axis (obliquity, azimuth).
# See campaigns/flyby_spin_angle_k1e7.md for rationale and parameter clarifications.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
  echo "[INFO] Dry-run mode (no phantom execution)"
fi

LOG="${LOG:-sobol_mass_runs/flyby_spin_angle_k1e7_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting flyby_spin_angle_k1e7 at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --num-samples 32 \
  --seed 42 \
  --batch-label flyby_spin_angle_k1e7 \
  --use-dem-fixed true \
  --np-apophis 500 \
  --kc-fixed 1e7 \
  --spin-period-min 1.2 \
  --spin-period-max 3.2 \
  --spin-obliquity-min 0 \
  --spin-obliquity-max 180 \
  --spin-azimuth-min 0 \
  --spin-azimuth-max 360 \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  "${DRY_RUN_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
