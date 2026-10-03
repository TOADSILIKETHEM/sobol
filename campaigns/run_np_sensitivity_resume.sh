#!/usr/bin/env bash
# Resume np sensitivity after partial failure: Regime A run 6, then B + C.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

REGIME_A_BATCH="sobol_mass_runs/sobol_20260621_170343_np_sens_obj_noearth_p155"
LOG="${LOG:-sobol_mass_runs/np_sensitivity_resume_stdout.log}"
NP_LIST=(250 375 500 750 1000 1500)

echo "[INFO] Resuming np_sensitivity at $(date -Is)" | tee "$LOG"

echo "[INFO] Regime A: retry run 6 (np=1500)" | tee -a "$LOG"
OMP_NUM_THREADS=2 python3 sobol/resume_batch.py \
  --batch-dir "$REGIME_A_BATCH" \
  --run-ids 6 \
  --base-dir sobol \
  --prefix sobol \
  --num-samples 6 \
  --use-dem-fixed true \
  --use-shape-crop-fixed true \
  --apophis-only-fixed true \
  --kc-fixed 1e7 \
  --spin-period-fixed 1.55 \
  --np-apophis-list 250 375 500 750 1000 1500 \
  --tmax-hours 108 \
  --dtmax-hours 0.5 \
  --ephemeris-cache-dir sobol \
  --jobs 1 \
  2>&1 | tee -a "$LOG"

# Regime B — staged Earth-opposite setup
EARTH_STAGE="${REPO_ROOT}/sobol/staging_np_sens_earth"
mkdir -p "$EARTH_STAGE"
cp "${REPO_ROOT}/sobol/sobol.setup" "${EARTH_STAGE}/sobol.setup"
cp "${REPO_ROOT}/sobol/sobol.in" "${EARTH_STAGE}/sobol.in"
python3 - <<'PY'
import re
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
PY

echo "[INFO] Regime B: np_sens_obj_earth_opposite" | tee -a "$LOG"
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
  2>&1 | tee -a "$LOG"

echo "[INFO] Regime C: np_sens_sphere_noearth_p200" | tee -a "$LOG"
python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol \
  --output-root sobol_mass_runs \
  --prefix sobol \
  --batch-label np_sens_sphere_noearth_p200 \
  --np-apophis-list "${NP_LIST[@]}" \
  --use-dem-fixed true \
  --use-shape-crop-fixed false \
  --apophis-only-fixed true \
  --kc-fixed 1e7 \
  --spin-period-fixed 2.0 \
  --tmax-hours 108 \
  --dtmax-hours 0.5 \
  --ephemeris-cache-dir sobol \
  --jobs 2 \
  2>&1 | tee -a "$LOG"

echo "[INFO] Finished np_sensitivity resume at $(date -Is)" | tee -a "$LOG"
