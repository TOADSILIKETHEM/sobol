# Torque-align Blender re-runs with OBJ cropping (`torque_align_blender_vis_obj`)

## Purpose

Re-run the two torque-align extremes from `flyby_spin_torque_align_k1e7` (Sobol runs **4** and **19**) with **full dumps kept** and **literature bilobed mesh** initial geometry (not a sphere lattice).

## Cases

| run_dir | Label | `apophis_spin_torque_align_deg` | Role |
|---------|--------|----------------------------------|------|
| `run_0001` | `aligned_near_h` | ~11.2° | Near **+ĥ** (spin AM ∥ relative orbital AM) — **intact** |
| `run_0002` | `opposite_near_h` | ~177.5° | Near **−ĥ** (spin AM anti-parallel to orbital AM) — **breakup** |

Spin frame: `docs/METRICS.md` § *Spin axis* and § *Terminology: “+ĥ aligned” ≠ aligned with tidal force* (why near +**ĥ** is stable and near −**ĥ** breaks up at *P* ≈ 1.52 hr — not a sign error). Campaign notes: `sobol/campaigns/flyby_spin_torque_align_k1e7.md`.

## Fixed parameters

| Parameter | Value |
|-----------|--------|
| Earth flyby | On |
| DEM | `use_dem=T`, `np_apophis=500` |
| Shape | `Shapes/apophis.shape` + `apophis_v233s7.obj`, `scale_r_apophis≈1.205` |
| k_c | **10⁷ dyne/cm** |
| Spin period | **1.52 h** |
| t_max | 4.5 days (template) |

## Launch

```bash
bash sobol/campaigns/run_torque_align_blender_vis_obj.sh
```

Equivalent:

```bash
python3 sobol/run_torque_align_blender_reruns.py \
  --use-shape-crop \
  --batch-label torque_align_blender_vis_obj
```

Sphere comparison (no OBJ): `bash sobol/campaigns/run_torque_align_blender_vis.sh`.

## Reference batch (completed)

`sobol_mass_runs/sobol_20260601_171137_torque_align_blender_vis_obj/`

Example metrics (`sobol_mass_outputs.csv`):

- run_0001: `dispersion_ratio` ≈ 1.28, `unbound_fraction` ≈ 0.016  
- run_0002: `dispersion_ratio` ≈ 54, `unbound_fraction` ≈ 0.93  

## Blender pipeline

1. Convert dumps (WSL): `sobol/Analysis/run_demtocsv_batch.py` → `Code/DEMCSVs/torque_align_obj/run_0001_*` and `run_0002_*`.
2. Blender: `Code/BlenderConvert/DEMGrainsBlender.py` — set `GRAINS_CSV_DIR` / `BODIES_CSV_DIR` to the chosen `run_XXXX_*_output` folder.

Full commands: `CLAUDE.md` § *Windows post-processing and visualisation*.
