# Flyby spin-axis angle campaign (`flyby_spin_angle_k1e7`)

## Science goal

Test whether **spin axis orientation** (obliquity + azimuth in ecliptic coordinates) changes DEM breakup or spreading during the **2029 Earth flyby**, with cohesion fixed at literature-central **k_c = 10⁷ dyne/cm** (~20 Pa bulk scaling).

This is **not** the same as modelling tidal spin-up during the encounter: initial spin is imposed once at t = 0 via `apply_apophis_spin`. The flyby tidal field can redistribute stress, but there is no separate “tidal rotation rate” parameter—only whether the **pre-existing** spin vector is aligned or anti-aligned with directions that favour neck tension vs compression.

## Fixed parameters (match prior flyby DEM batches)

| Parameter | Value | Notes |
|-----------|--------|--------|
| Earth | On (`apophis_only=F`) | Full solar system + flyby |
| DEM | `use_dem=T`, `np_apophis=500` | |
| k_c | **10⁷ dyne/cm** | `--kc-fixed 1e7` |
| t_max | 4.5 days | Template `sobol.setup` |
| Epoch | 2029-04-10 | Cached ephemeris in `sobol/` |
| Shape | Sphere lattice (blank `apophis_shape_file`) | Same as `flyby_spin_kc_p3` |
| Spin obliquity/azimuth (reference) | 0°, 0° | Pole along ecliptic north; all prior sweeps used this |

## Varying parameters (3 Sobol dimensions, 32 samples)

| Dimension | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| `apophis_spin_period` | **1.2 h** | **3.2 h** | Brackets marginal → stable at k_c ~ 10⁷ (P3: disruption only below ~1.1 h at weak k_c; flyby_spin_earth spreading at 2–4 h with k_c=0) |
| `apophis_spin_obliquity` | **0°** | **180°** | Spin axis from ecliptic north to south |
| `apophis_spin_azimuth` | **0°** | **360°** | Axis direction in ecliptic plane |

## Spin sense (prograde vs retrograde)

PHANTOM uses **ω = 2π/T > 0** always. **Retrograde** spin about the same geometric axis is obtained by **flipping the spin axis** (e.g. obliquity 0° → 180° for a z-aligned pole), not by negative period.

For in-plane axes (obliquity ≈ 90°), azimuth and azimuth + 180° are opposite senses.

## Parameters that are **not** in this campaign (clarifications)

1. **Observed Apophis spin pole** — Real pole orientation (~30.6 h period) is a specific (obliquity, azimuth) point, not sampled here; add a dedicated run if needed.
2. **Tidal spin-up rate** — Not implemented; compare axis alignment only.
3. **Spin period sign** — Not supported (`period ≤ 0` disables spin).
4. **Bilobed shape** — This batch uses spherical packing like P3; neck–axis coupling needs `--shape-file` + `--use-shape-crop-fixed true` in a follow-up.
5. **k_c sweep** — Fixed at 10⁷; use P3 for spin×k_c.

## Launch

From repo root:

```bash
bash sobol/campaigns/run_flyby_spin_angle_k1e7.sh
```

Dry-run only: `DRY_RUN=1 bash sobol/campaigns/run_flyby_spin_angle_k1e7.sh`

## Outputs

`sobol_mass_runs/sobol_<timestamp>_flyby_spin_angle_k1e7/`

Columns: `apophis_spin_period`, `apophis_spin_obliquity`, `apophis_spin_azimuth`, `kc_cgs`, `dispersion_ratio`, `unbound_fraction`, `closest_approach_km`.
