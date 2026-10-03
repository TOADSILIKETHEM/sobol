#!/usr/bin/env bash
# Intrinsic spin threshold: no Earth, sphere DEM, tmax=12 hr, spin 0.66–5 hr.
# Runs spin_kc0_12hr then spin_kc1e7_12hr (§4.3 RESULTS_SECTION_STRUCTURE.md).
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

COMMON=(
  --base-dir sobol
  --output-root sobol_mass_runs
  --prefix sobol
  --num-samples 16
  --seed 42
  --use-dem-fixed true
  --np-apophis 500
  --spin-period-min 0.66
  --spin-period-max 5
  --tmax-hours 12
  --apophis-only-fixed true
  --ephemeris-cache-dir sobol
  --jobs 2
  "${DRY_RUN_ARGS[@]}"
)

LOG="${LOG:-sobol_mass_runs/spin_kc_intrinsic_12hr_stdout.log}"
mkdir -p sobol_mass_runs

run_batch() {
  local label=$1
  local kc=$2
  shift 2
  echo "[INFO] Starting ${label} (kc=${kc}) at $(date -Is)" | tee -a "$LOG"
  python3 sobol/run_mass_sobol_phantom.py \
    "${COMMON[@]}" \
    --batch-label "$label" \
    --kc-fixed "$kc" \
    "$@" \
    2>&1 | tee -a "$LOG"
}

run_batch spin_kc0_12hr 0 "$@"
run_batch spin_kc1e7_12hr 1e7 "$@"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
echo "[INFO] Plot: python3 sobol/Analysis/plot_spin_kc_intrinsic_threshold.py" | tee -a "$LOG"
