#!/usr/bin/env bash
# No-Earth DEM control: sweep spin period 1.5–2.05 hr, apophis_only=T.
# Companion to flyby_spin_earth (2–4 hr) and the 1.55/1.60/2.0 hr single runs.
# np=500 sphere lattice, kc=1e7, 4.5 day tmax, 30 min dumps.
# Establishes the intrinsic spin disruption threshold without tidal forcing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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

LOG="${LOG:-sobol_mass_runs/noearth_ctrl_spin_1p5_2hr_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting noearth_ctrl_spin_1p5_2hr at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --num-samples 16 \
  --seed 42 \
  --batch-label noearth_ctrl_spin_1p5_2hr \
  --use-dem-fixed true \
  --np-apophis 500 \
  --kc-fixed 1e7 \
  --spin-period-min 1.50 \
  --spin-period-max 2.05 \
  --apophis-only-fixed true \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  "${DRY_RUN_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
