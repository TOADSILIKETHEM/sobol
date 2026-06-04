#!/usr/bin/env bash
# OBJ-cropped opposite_near_h breakup re-run with finer dumps for Blender (5 min vs 30 min).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

LOG="${LOG:-sobol_mass_runs/torque_align_obj_breakup_fine_dt_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting torque_align_obj_breakup_fine_dt at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_torque_align_blender_reruns.py \
  --use-shape-crop \
  --case opposite \
  --dtmax-hours "$(python3 -c 'print(5/60)')" \
  --batch-label torque_align_obj_breakup_fine_dt \
  "$@" 2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
