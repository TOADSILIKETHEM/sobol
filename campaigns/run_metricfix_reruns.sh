#!/usr/bin/env bash
# Re-run thesis campaigns after intrinsic-spin metric fix (6 h window + intactness gates).
# Creates new timestamped batch dirs; plot scripts pick latest suffix automatically.
#
# Tier 1 — spin metrics in thesis plots (~165 runs, ~8–10 h):
#   np_sensitivity_v2 (47), np_spin_grid (94), flyby_spin30 (24)
# Tier 2 — supporting intrinsic / threshold (~49 runs, ~3–4 h):
#   noearth_obj_ctrl (16), spin_kc0+kc1e7 (32), fine_dt p2hr (1)
#
# Skipped (dispersion-only or already re-extracted from .ev):
#   flyby_spin_torque_period_kmin_obj, fine_dt p155/p1p6hr/breakup/nospin, sphere noearth_ctrl
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export JOBS="${JOBS:-2}"

MASTER_LOG="${MASTER_LOG:-sobol_mass_runs/metricfix_rerun_stdout.log}"
mkdir -p sobol_mass_runs

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

log "=== metricfix rerun package start ==="
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} JOBS=${JOBS}"

# Fresh state files — do not skip batches marked DONE in prior campaigns
export STATE="sobol_mass_runs/metricfix_np_sensitivity_v2_state.tsv"
: > "$STATE"
log "Tier 1a: np_sensitivity_v2 (47 runs)"
bash sobol/campaigns/run_np_sensitivity_v2.sh 2>&1 | tee -a "$MASTER_LOG"

export STATE="sobol_mass_runs/metricfix_np_spin_grid_state.tsv"
: > "$STATE"
log "Tier 1b: np_spin_grid (94 runs)"
bash sobol/campaigns/run_np_spin_grid.sh 2>&1 | tee -a "$MASTER_LOG"

log "Tier 1c: flyby_spin30_torque_align_k1e7 (24 runs)"
bash sobol/campaigns/run_flyby_spin30_torque_align_k1e7.sh 2>&1 | tee -a "$MASTER_LOG"

log "Tier 2a: noearth_obj_ctrl_spin_1p5_2hr (16 runs)"
bash sobol/campaigns/run_noearth_obj_ctrl_spin_1p5_2hr.sh 2>&1 | tee -a "$MASTER_LOG"

log "Tier 2b: spin_kc intrinsic 12 hr (32 runs)"
bash sobol/campaigns/run_spin_kc_intrinsic_12hr.sh 2>&1 | tee -a "$MASTER_LOG"

log "Tier 2c: torque_align_obj_fine_dt_p2hr_np1000 (1 run)"
bash sobol/campaigns/run_torque_align_obj_fine_dt_p2hr_np1000.sh 2>&1 | tee -a "$MASTER_LOG"

log "=== metricfix rerun package complete ==="
log "Regenerate plots:"
log "  python3 sobol/Analysis/plot_np_sensitivity.py"
log "  python3 sobol/Analysis/plot_np_spin_grid.py"
log "  python3 sobol/Analysis/plot_flyby_spin30_torque_align.py"
log "  python3 sobol/Analysis/plot_spin_disruption_threshold.py"
