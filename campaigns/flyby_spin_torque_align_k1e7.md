# Flyby spin–torque alignment campaign (`flyby_spin_torque_align_k1e7`)

## Science goal

Test whether **spin aligned vs opposite to the flyby orbital angular momentum** changes DEM breakup, with cohesion and spin rate fixed in the marginal regime.

Unlike ecliptic `apophis_spin_obliquity` / `apophis_spin_azimuth`, this campaign uses **`apophis_spin_torque_align_deg`** in `phantomsetup`:

- At setup epoch, compute Earth–Apophis relative position **r** and velocity **v** (sink 4 vs DEM centroid).
- **ĥ** = unit(**r** × **v**) — direction of relative orbital angular momentum.
- **r̂** = unit(**r**).
- Spin axis **ŝ** = rotate **ĥ** about **r̂** by `torque_align_deg` (Rodrigues):
  - **0°** → **ŝ = +ĥ** (spin parallel to flyby orbit angular momentum)
  - **180°** → **ŝ = −ĥ** (opposite / retrograde about the orbit normal)

This is the natural frame for “aligned vs opposite” to the sense of the encounter orbit, not raw ecliptic angles.

## Fixed parameters

| Parameter | Value |
|-----------|--------|
| Earth flyby | On |
| DEM | `use_dem=T`, `np_apophis=500` |
| k_c | **10⁷ dyne/cm** (`--kc-fixed 1e7`) |
| Spin period | **1.52 h** (`--spin-period-fixed 1.52`) — near P3 threshold at k_c=10⁷ |
| t_max | 4.5 days (template) |

## Varying parameter (1 Sobol dimension, 24 samples)

| Dimension | Min | Max |
|-----------|-----|-----|
| `apophis_spin_torque_align_deg` | **0°** | **180°** |

Includes endpoints 0° (aligned +h) and 180° (opposite −h).

## Rebuild required

New setup key needs current `phantomsetup`:

```bash
cd sobol && make setup && make
strings sobol/phantomsetup | grep -i torque_align
```

## Launch

```bash
bash sobol/campaigns/run_flyby_spin_torque_align_k1e7.sh
```

## Outputs

`sobol_mass_runs/sobol_<timestamp>_flyby_spin_torque_align_k1e7/`

Key columns: `apophis_spin_torque_align_deg`, `apophis_spin_period`, `dispersion_ratio`, `unbound_fraction`.

## Blender re-runs (two extremes)

From this batch, Sobol runs **4** and **19** are re-run for visualization (near +h / near −h):

| Launcher | Shape | Batch label |
|----------|--------|-------------|
| `sobol/campaigns/run_torque_align_blender_vis.sh` | Sphere | `torque_align_blender_vis` |
| `sobol/campaigns/run_torque_align_blender_vis_obj.sh` | OBJ (`--use-shape-crop`) | `torque_align_blender_vis_obj` |

See `sobol/run_torque_align_blender_reruns.py` and `sobol/campaigns/torque_align_blender_vis_obj.md`. Blender CSV handoff: `CLAUDE.md` § *Windows post-processing and visualisation*.
