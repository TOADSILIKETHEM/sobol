#!/usr/bin/env bash
# np_apophis resolution study: Regimes A (OBJ no-Earth marginal), B (OBJ Earth opposite),
# C (sphere no-Earth null). 18 runs total; cleanup ON (default).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

NP_LIST=(250 375 500 750 1000 1500)
NP_LIST_CSV="${NP_LIST[*]}"

_torque_align_sym=$(
  strings sobol/phantomsetup 2>/dev/null | grep -c apophis_spin_torque_align_deg || true
)
if [[ "${_torque_align_sym:-0}" -lt 1 ]]; then
  echo "[ERROR] phantomsetup lacks apophis_spin_torque_align_deg — rebuild: cd sobol && make setup && make" >&2
  exit 1
fi

if ! grep -q 'MAXPTMASS=2000' sobol/Makefile 2>/dev/null; then
  echo "[WARN] sobol/Makefile may lack MAXPTMASS=2000 — np=1500 needs MAXPTMASS>=2000" >&2
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
  echo "[INFO] Dry-run mode (no phantom execution)"
fi

LOG="${LOG:-sobol_mass_runs/np_sensitivity_stdout.log}"
mkdir -p sobol_mass_runs

COMMON=(
  --base-dir sobol
  --output-root sobol_mass_runs
  --prefix sobol
  --use-dem-fixed true
  --kc-fixed 1e7
  --tmax-hours 108
  --dtmax-hours 0.5
  --ephemeris-cache-dir sobol
  --jobs 2
  "${DRY_RUN_ARGS[@]}"
)

run_np_list_batch() {
  local label=$1
  shift
  echo "[INFO] === ${label} at $(date -Is) ===" | tee -a "$LOG"
  python3 sobol/run_mass_sobol_phantom.py \
    "${COMMON[@]}" \
    --batch-label "$label" \
    --np-apophis-list "${NP_LIST[@]}" \
    "$@" \
    2>&1 | tee -a "$LOG"
}

stage_earth_opposite_setup() {
  local stage="${REPO_ROOT}/sobol/staging_np_sens_earth"
  mkdir -p "$stage"
  cp "${REPO_ROOT}/sobol/sobol.setup" "${stage}/sobol.setup"
  cp "${REPO_ROOT}/sobol/sobol.in" "${stage}/sobol.in"
  python3 - <<'PY'
import re
import sys
from pathlib import Path
p = Path("sobol/staging_np_sens_earth/sobol.setup")
text = p.read_text(encoding="utf-8")

def rep(key: str, val: str) -> None:
    global text
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^!]*)(!.*)?$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        raise SystemExit(f"key not found: {key}")
    comment = m.group(3) or ""
    text = pat.sub(f"{m.group(1)}{val}{(' ' + comment.strip()) if comment else ''}", text, count=1)

rep("apophis_only", "          F")
rep("apophis_spin_period", "       1.520")
rep("apophis_spin_torque_align_deg", "  177.500")
p.write_text(text, encoding="utf-8")
print(f"[INFO] Staged Earth-opposite setup: {p}", file=sys.stderr)
PY
  echo "$stage"
}

echo "[INFO] Starting np_sensitivity full package (18 runs) at $(date -Is)" | tee "$LOG"
echo "[INFO] np_apophis values: ${NP_LIST_CSV}" | tee -a "$LOG"
echo "[INFO] Cleanup enabled (default; no --no-cleanup)" | tee -a "$LOG"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS}" | tee -a "$LOG"

# Regime A — OBJ no-Earth marginal (Priority 2 analogue at P=1.55 hr)
run_np_list_batch np_sens_obj_noearth_p155 \
  --use-shape-crop-fixed true \
  --apophis-only-fixed true \
  --spin-period-fixed 1.55

# Regime B — OBJ Earth opposite (~177.5 deg vs h_hat)
EARTH_STAGE="$(stage_earth_opposite_setup)"
echo "[INFO] === np_sens_obj_earth_opposite at $(date -Is) ===" | tee -a "$LOG"
python3 sobol/run_mass_sobol_phantom.py \
  --base-dir "$EARTH_STAGE" \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --batch-label np_sens_obj_earth_opposite \
  --np-apophis-list "${NP_LIST[@]}" \
  --use-dem-fixed true \
  --use-shape-crop-fixed true \
  --kc-fixed 1e7 \
  --spin-period-fixed 1.52 \
  --tmax-hours 108 \
  --dtmax-hours 0.5 \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  "${DRY_RUN_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"

# Regime C — sphere no-Earth null at P=2.0 hr
run_np_list_batch np_sens_sphere_noearth_p200 \
  --use-shape-crop-fixed false \
  --apophis-only-fixed true \
  --spin-period-fixed 2.0

echo "[INFO] Finished np_sensitivity full package at $(date -Is)" | tee -a "$LOG"
