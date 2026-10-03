#!/usr/bin/env bash
# np × spin-period grid campaign (Tier A + B): P1 intrinsic OBJ no-Earth, P2 σ_c-constant,
# P3 Earth opposite subset. Cleanup ON (default). Auto-resumes partial batches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
JOBS="${JOBS:-2}"
LOG="${LOG:-sobol_mass_runs/np_spin_grid_stdout.log}"
STATE="${STATE:-sobol_mass_runs/np_spin_grid_state.tsv}"

mkdir -p sobol_mass_runs

COMMON=(
  --output-root sobol_mass_runs
  --prefix sobol
  --use-dem-fixed true
  --kt-fixed 1e7
  --tmax-hours 108
  --dtmax-hours 0.5
  --ephemeris-cache-dir sobol
  --jobs "$JOBS"
)

NP_GRID=(400 500 600 750 1000)
SPIN_GRID=(1.45 1.55 1.65 1.75 1.85 1.95 2.05)
NP_EARTH=(400 500 750 1000)
SPIN_EARTH=(1.45 1.55 1.65 1.75 1.95 2.05)

log() { echo "[INFO] $*" | tee -a "$LOG"; }
warn() { echo "[WARN] $*" | tee -a "$LOG" >&2; }
err() { echo "[ERROR] $*" | tee -a "$LOG" >&2; }

batch_done() {
  local label=$1
  grep -q "^${label}\tDONE$" "$STATE" 2>/dev/null
}

mark_done() {
  local label=$1
  grep -v "^${label}\t" "$STATE" 2>/dev/null > "${STATE}.tmp" || true
  echo -e "${label}\tDONE" >> "${STATE}.tmp"
  mv "${STATE}.tmp" "$STATE"
}

latest_batch_dir() {
  local label=$1
  ls -td "${REPO_ROOT}/sobol_mass_runs/sobol_"*_"${label}" 2>/dev/null | head -1
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

stage_earth_opposite_setup() {
  local stage=$1
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
rep("apophis_spin_torque_align_deg", "  177.500")
p.write_text(text, encoding="utf-8")
print(f"Staged Earth-opposite setup -> {p}", file=sys.stderr)
PY
  echo "$stage"
}

run_grid_batch_with_resume() {
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
  local kt=0
  for bin in sobol/bin/phantomsetup sobol/phantomsetup; do
    if [[ -x "$bin" ]]; then
      sym=$(strings "$bin" 2>/dev/null | grep -c apophis_spin_torque_align_deg || true)
      kt=$(strings "$bin" 2>/dev/null | grep -c 'kt_cgs' || true)
      if [[ "${sym}" -ge 1 && "${kt}" -ge 1 ]]; then
        break
      fi
    fi
  done
  if [[ "${sym}" -lt 1 ]]; then
    err "phantomsetup lacks spin support — run: cd sobol && make setup && make"
    exit 1
  fi
  if [[ "${kt}" -lt 1 ]]; then
    err "phantomsetup lacks kt_cgs — rebuild: cd sobol && make setup && make"
    exit 1
  fi
}

EARTH_OPPOSITE="$(stage_earth_opposite_setup sobol/staging_np_spin_grid_earth_opposite)"

touch "$STATE"
log "Starting np_spin_grid Tier A+B at $(date -Is)"
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} JOBS=${JOBS}"
log "Cleanup enabled (default; no --no-cleanup)"
preflight_binaries

# P1 — Tier A: OBJ no-Earth ~177° (obl/az in template), fixed kc
run_grid_batch_with_resume np_spin_p1_noearth_obj_kc sobol \
  --use-shape-crop-fixed true \
  --apophis-only-fixed true \
  --np-apophis-list "${NP_GRID[@]}" \
  --spin-period-list "${SPIN_GRID[@]}"

# P2 — Tier B: OBJ no-Earth, σ_c-constant
run_grid_batch_with_resume np_spin_p2_noearth_obj_sigmac sobol \
  --use-shape-crop-fixed true \
  --apophis-only-fixed true \
  --kt-scale-ref-np 500 \
  --np-apophis-list "${NP_GRID[@]}" \
  --spin-period-list "${SPIN_GRID[@]}"

# P3 — Tier B: OBJ Earth opposite ~177°
run_grid_batch_with_resume np_spin_p3_earth_opposite_kc "$EARTH_OPPOSITE" \
  --use-shape-crop-fixed true \
  --np-apophis-list "${NP_EARTH[@]}" \
  --spin-period-list "${SPIN_EARTH[@]}"

log "Finished np_spin_grid Tier A+B at $(date -Is)"
log "State file: ${STATE}"
log "94 runs total (35 + 35 + 24)"
