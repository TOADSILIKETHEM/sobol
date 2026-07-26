#!/usr/bin/env bash
# np_apophis resolution study v2 — full package (~47 runs).
# Packages: P1 dense Earth-opposite, P2 σ_c-constant (Earth + no-Earth),
#           P3D dense no-Earth, P3E sphere null @ P=1.55, P3F near-aligned Earth, P3G np=2000.
# Cleanup ON (default). On partial failure, auto-resumes each batch before continuing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
JOBS="${JOBS:-2}"
LOG="${LOG:-sobol_mass_runs/np_sensitivity_v2_stdout.log}"
STATE="${STATE:-sobol_mass_runs/np_sensitivity_v2_state.tsv}"

mkdir -p sobol_mass_runs

COMMON=(
  --output-root sobol_mass_runs
  --prefix sobol
  --use-dem-fixed true
  --kc-fixed 1e7
  --tmax-hours 108
  --dtmax-hours 0.5
  --ephemeris-cache-dir sobol
  --jobs "$JOBS"
)

NP_DENSE=(300 350 400 425 450 475 500 525 550 575 600 650 700 800 900 1000 1200)
NP_COARSE=(300 400 500 600 750 1000 1200)
NP_BREADTH=(300 500 750 1000)

log() { echo "[INFO] $*" | tee -a "$LOG"; }
warn() { echo "[WARN] $*" | tee -a "$LOG"; }
err() { echo "[ERROR] $*" | tee -a "$LOG" >&2; }

batch_done() {
  local label=$1
  grep -q "^${label}\tDONE$" "$STATE" 2>/dev/null
}

mark_done() {
  local label=$1
  grep -v "^${label}	" "$STATE" 2>/dev/null > "${STATE}.tmp" || true
  echo -e "${label}\tDONE" >> "${STATE}.tmp"
  mv "${STATE}.tmp" "$STATE"
}

latest_batch_dir() {
  local label=$1
  ls -td "${REPO_ROOT}/sobol_mass_runs/sobol_"*"_${label}" 2>/dev/null | head -1
}

count_failed_runs() {
  local batch_dir=$1
  python3 - <<PY
import csv
from pathlib import Path
p = Path("${batch_dir}") / "sobol_mass_outputs.csv"
if not p.is_file():
    print(9999)
    raise SystemExit
rows = list(csv.DictReader(p.open()))
expected = len(list(csv.DictReader((Path("${batch_dir}")/"sobol_mass_samples.csv").open())))
ok = sum(1 for r in rows if r.get("status") == "ok")
print(max(0, expected - ok))
PY
}

stage_earth_setup() {
  local stage=$1
  local torque_align=$2
  local spin_period=$3
  mkdir -p "$stage"
  cp "${REPO_ROOT}/sobol/sobol.setup" "${stage}/sobol.setup"
  cp "${REPO_ROOT}/sobol/sobol.in" "${stage}/sobol.in"
  python3 - <<PY
import re
import sys
from pathlib import Path
p = Path("${stage}/sobol.setup")
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
rep("apophis_spin_period", "       ${spin_period}")
rep("apophis_spin_torque_align_deg", "  ${torque_align}")
p.write_text(text, encoding="utf-8")
print(f"Staged Earth setup torque_align=${torque_align} P=${spin_period} -> {p}", file=sys.stderr)
PY
  echo "$stage"
}

run_np_batch_with_resume() {
  local label=$1
  local base_dir=$2
  shift 2
  local -a extra_flags=("$@")

  if batch_done "$label"; then
    log "Skipping ${label} (marked DONE in ${STATE})"
    return 0
  fi

  log "=== Batch ${label} at $(date -Is) ==="
  set +e
  OMP_NUM_THREADS="$OMP_NUM_THREADS" python3 sobol/run_mass_sobol_phantom.py \
    --base-dir "$base_dir" \
    "${COMMON[@]}" \
    --batch-label "$label" \
    "${extra_flags[@]}" \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e

  local batch_dir
  batch_dir="$(latest_batch_dir "$label")"
  if [[ -z "${batch_dir}" || ! -d "${batch_dir}" ]]; then
    err "Batch ${label}: no output directory found after launch"
    return 1
  fi

  local attempts=0
  while [[ $(count_failed_runs "$batch_dir") -gt 0 ]]; do
    attempts=$((attempts + 1))
    if [[ $attempts -gt 5 ]]; then
      err "Batch ${label}: still has failures after ${attempts} resume attempts — stopping package"
      return 1
    fi
    warn "Batch ${label}: resuming failures (attempt ${attempts}) in ${batch_dir}"
    set +e
    OMP_NUM_THREADS="$OMP_NUM_THREADS" python3 sobol/resume_batch.py \
      --batch-dir "$batch_dir" \
      --base-dir "$base_dir" \
      "${COMMON[@]}" \
      "${extra_flags[@]}" \
      2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
      warn "Batch ${label}: resume_batch exited ${rc}"
    fi
    batch_dir="$(latest_batch_dir "$label")"
  done

  mark_done "$label"
  log "Batch ${label} complete → ${batch_dir}"
  return 0
}

preflight_binaries() {
  local sym=0
  for bin in sobol/bin/phantomsetup sobol/phantomsetup; do
    if [[ -x "$bin" ]]; then
      sym=$(strings "$bin" 2>/dev/null | grep -c apophis_spin_torque_align_deg || true)
      if [[ "${sym}" -ge 1 ]]; then
        break
      fi
    fi
  done
  if [[ "${sym}" -lt 1 ]]; then
    err "phantomsetup lacks spin support — run: cd sobol && make setup && make"
    exit 1
  fi
  local maxpt
  maxpt=$(grep -oP 'MAXPTMASS=\K[0-9]+' sobol/Makefile | head -1 || echo 0)
  if [[ "${maxpt}" -lt 2500 ]]; then
    warn "MAXPTMASS=${maxpt} < 2500 — rebuilding PHANTOM for np=2000"
    (cd sobol && make setup && make) 2>&1 | tee -a "$LOG"
  elif ! grep -q 'MAXPTMASS=3000' sobol/phantom_version 2>/dev/null; then
    warn "Rebuilding PHANTOM with MAXPTMASS=3000 for np=2000 arm"
    (cd sobol && make setup && make) 2>&1 | tee -a "$LOG"
  fi
}

# --- staging ---
EARTH_OPPOSITE="$(stage_earth_setup sobol/staging_np_sens_v2_earth_opposite 177.500 1.520)"
EARTH_ALIGNED="$(stage_earth_setup sobol/staging_np_sens_v2_earth_aligned 11.1870651506 1.520)"

touch "$STATE"
log "Starting np_sensitivity_v2 full package at $(date -Is)"
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} JOBS=${JOBS}"
log "Cleanup enabled (default; no --no-cleanup)"
preflight_binaries

# P1 — dense Regime B (fixed kc)
run_np_batch_with_resume np_sens_v2_obj_earth_opposite_dense "$EARTH_OPPOSITE" \
  --use-shape-crop-fixed true \
  --spin-period-fixed 1.52 \
  --np-apophis-list "${NP_DENSE[@]}"

# P2a — σ_c-constant Earth opposite
run_np_batch_with_resume np_sens_v2_obj_earth_opposite_sigmac "$EARTH_OPPOSITE" \
  --use-shape-crop-fixed true \
  --spin-period-fixed 1.52 \
  --kc-scale-ref-np 500 \
  --np-apophis-list "${NP_COARSE[@]}"

# P2b — σ_c-constant OBJ no-Earth @ P=1.55
run_np_batch_with_resume np_sens_v2_obj_noearth_p155_sigmac sobol \
  --use-shape-crop-fixed true \
  --apophis-only-fixed true \
  --spin-period-fixed 1.55 \
  --kc-scale-ref-np 500 \
  --np-apophis-list "${NP_COARSE[@]}"

# P3D — dense OBJ no-Earth @ P=1.55 (fixed kc)
run_np_batch_with_resume np_sens_v2_obj_noearth_p155_dense sobol \
  --use-shape-crop-fixed true \
  --apophis-only-fixed true \
  --spin-period-fixed 1.55 \
  --np-apophis-list "${NP_COARSE[@]}"

# P3E — sphere null @ P=1.55
run_np_batch_with_resume np_sens_v2_sphere_noearth_p155 sobol \
  --use-shape-crop-fixed false \
  --apophis-only-fixed true \
  --spin-period-fixed 1.55 \
  --np-apophis-list "${NP_BREADTH[@]}"

# P3F — OBJ Earth near-aligned (~11°)
run_np_batch_with_resume np_sens_v2_obj_earth_aligned "$EARTH_ALIGNED" \
  --use-shape-crop-fixed true \
  --spin-period-fixed 1.52 \
  --np-apophis-list "${NP_BREADTH[@]}"

# P3G — upper resolution bound
run_np_batch_with_resume np_sens_v2_obj_earth_opposite_np2000 "$EARTH_OPPOSITE" \
  --use-shape-crop-fixed true \
  --spin-period-fixed 1.52 \
  --np-apophis-list 2000

log "Finished np_sensitivity_v2 full package at $(date -Is)"
log "State file: ${STATE}"
log "Plots: python3 sobol/Analysis/plot_np_sensitivity.py (update for v2 batches)"
