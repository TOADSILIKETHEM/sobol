# Flyby OBJ torque-align tidal spin (`flyby_spin_torque_align_obj_k1e7`)

## Science goal

Show that **spin–orbit tidal torque** changes post-flyby spin period in a sign-consistent way across
`apophis_spin_torque_align_deg` ∈ [0°, 180°] on **OBJ-cropped** DEM (no sphere lattice confound).

Primary plot metric: **post − approach** (%). Secondary: post − intrinsic.

**Not** a claim about alignment with tidal *stretch* (along **r̂**). See `docs/METRICS.md`.

## Fixed parameters

| Parameter | Value |
|-----------|--------|
| Earth flyby | On |
| DEM | `use_dem=T`, `np_apophis=500` |
| Shape | OBJ crop (`--use-shape-crop-fixed true`) |
| k_c | **10⁷ dyne/cm** (`--kc-fixed 1e7`) |
| Spin period | **2.05 h** (`--spin-period-fixed 2.05`) — P3 grid shows intact at ~177°; faster than 30 hr for measurable tides |
| t_max / dt | **108 h**, **0.5 h** dumps |

## Varying parameter (24 samples, seed 43)

| Dimension | Min | Max |
|-----------|-----|-----|
| `apophis_spin_torque_align_deg` | 0° | 180° |

## Scout (`flyby_spin_torque_align_obj_scout`)

8 samples, same fixed params — validates intactness + intrinsic flatness before 24-run commit.

## Acceptance (automated)

`python3 sobol/Analysis/verify_obj_torque_spin_trend.py <batch_dir>`:

- ≥ **20/24** runs: `dispersion_ratio < 1.15` and `unbound_fraction < 0.01`
- Intrinsic spread across angles **< 8%** (OBJ settling; not sphere 3% lattice target)
- `post − approach` at lowest angle vs highest angle differ by **≥ 0.3%** with **opposite signs**

## Outputs

`sobol_mass_runs/sobol_<timestamp>_flyby_spin_torque_align_obj_k1e7/`

Plot: `python3 sobol/Analysis/plot_flyby_spin_torque_align_obj.py`
→ `sobol_mass_runs/plots/flyby_spin_torque_align_obj_k1e7.png`
