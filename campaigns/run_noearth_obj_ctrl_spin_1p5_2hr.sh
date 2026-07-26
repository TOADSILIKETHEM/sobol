#!/usr/bin/env bash
# Priority 2: matched np=1000 OBJ no-Earth control, spin 1.52–2.05 hr, kc=1e7.
# Spin axis ~177° via ecliptic obl/az in sobol/sobol.setup (torque_align=-1; Earth absent).
# Default cleanup ON (binary dumps removed after metrics extraction).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

_spin_sym=$(
  strings sobol/phantomsetup 2>/dev/null | grep -c apophis_spin_period || true
)
if [[ "${_spin_sym:-0}" -lt 1 ]]; then
  echo "[ERROR] phantomsetup lacks apophis_spin_period — rebuild: cd sobol && make setup && make" >&2
  exit 1
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
  echo "[INFO] Dry-run mode (no phantom execution)"
fi

LOG="${LOG:-sobol_mass_runs/noearth_obj_ctrl_spin_1p5_2hr_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting noearth_obj_ctrl_spin_1p5_2hr at $(date -Is)" | tee -a "$LOG"
echo "[INFO] Cleanup enabled (default; omit --no-cleanup)" | tee -a "$LOG"

python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --num-samples 16 \
  --seed 43 \
  --batch-label noearth_obj_ctrl_spin_1p5_2hr \
  --use-dem-fixed true \
  --np-apophis 1000 \
  --use-shape-crop-fixed true \
  --kc-fixed 1e7 \
  --apophis-only-fixed true \
  --tmax-hours 108 \
  --dtmax-hours 0.5 \
  --spin-period-min 1.52 \
  --spin-period-max 2.05 \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  "${DRY_RUN_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
