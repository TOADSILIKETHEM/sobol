#!/usr/bin/env bash
# Re-run torque-align runs 4 & 19 (near +h / near -h) with literature OBJ cropping + dumps for Blender.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

LOG="${LOG:-sobol_mass_runs/torque_align_blender_vis_obj_stdout.log}"
mkdir -p sobol_mass_runs

echo "[INFO] Starting torque_align_blender_vis_obj at $(date -Is)" | tee -a "$LOG"

python3 sobol/run_torque_align_blender_reruns.py \
  --use-shape-crop \
  --batch-label torque_align_blender_vis_obj \
  "$@" 2>&1 | tee -a "$LOG"

echo "[INFO] Finished at $(date -Is)" | tee -a "$LOG"
