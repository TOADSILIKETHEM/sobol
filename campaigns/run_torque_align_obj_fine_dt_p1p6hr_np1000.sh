#!/usr/bin/env bash
# OBJ-cropped opposite_near_h breakup re-run: spin 1.6 hr, np=1000, 5-min dumps.
# Same torque-align angle (177.5°) and kc=1e7 as the 1.52hr/np500 reference batch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

LOG="${LOG:-sobol_mass_runs/torque_align_obj_fine_dt_p1p6hr_np1000_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting torque_align_obj_fine_dt_p1p6hr_np1000 at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_torque_align_blender_reruns.py \
  --use-shape-crop \
  --case opposite \
  --spin-period 1.6 \
  --np-apophis 1000 \
  --dtmax-hours "$(python3 -c 'print(5/60)')" \
  --batch-label torque_align_obj_fine_dt_p1p6hr_np1000 \
  "$@" 2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
