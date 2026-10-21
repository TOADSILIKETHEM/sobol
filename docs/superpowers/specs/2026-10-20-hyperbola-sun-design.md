# Real 2029 Sun in hyperbola renders: design

Date: 2026-10-20. Repos touched: `Code/` (Blender scripts, pure-Python helper), `sobol/` (tests, this spec).

## Intent

Hyperbola runs (`--encounter hyperbola`, `phantomflyby`) have no Sun: the bodies CSV holds only `Earth` and `Apophis`. That is correct for the physics. For rendering it is not: the lamp shines from the fixed, arbitrary `APOPHIS_ONLY_SUN_DIRECTION = (1,0,0)`, there is no visible Sun, and Earth's time of day is arbitrary. When the camera turns toward the light there is nothing there.

The renders are meant to be realistic. The Sun must sit where it really was relative to Apophis's path in April 2029, at its true apparent size, with realistic brightness and glare, and Earth must show the real 2029 sunlit side. Legibility outranks realism: brightness and glare are the first things to be toned down if they make Apophis or Earth hard to read.

**Primary target:** the `auto` camera in the instance path (the default TUI render: tracks Apophis, Earth-side view; `DEMCinematicSetup.setup_camera`, `use_earth` branch).

## Goals

1. **Real Sun direction** in hyperbola renders, from the 2029 geometry carried onto the stand-in hyperbola (method below). The sun lamp (`DEM_SunLight`) uses it in place of `(1,0,0)`.
2. **Visible Sun** at true angular radius (0.2666 deg) in that direction, visible to the camera, occluded correctly by Earth.
3. **Brightness and glare**: disk brightness consistent with the lamp, plus a compositor glare/bloom so the Sun reads as the Sun. Both tunable by module constants; defaults chosen for legibility first (see Constraints).
4. **Real 2029 Earth orientation** in hyperbola renders: the sunlit hemisphere and continents match the real flyby at the matching time.
5. **Ephemeris runs get the same Sun visual** (shared code). The current ephemeris Sun sphere is ~4.8x too large (radius 10000 BU at 0.003 BU/km, so ~1.28 deg angular radius vs real 0.267 deg); it moves to the new true-size visual. Ephemeris lamp direction and Earth orientation already come from the Sun row and stay as they are.

## Non-goals

- No change to the simulation, `phantomflyby`, `encounter.py`, or the bodies CSV format.
- No Moon in hyperbola renders.
- `apophis_only` runs (no Earth row) keep `APOPHIS_ONLY_SUN_DIRECTION`; no visible Sun added there.
- The per-sphere hero script `DEMGrainsBlenderEarthCam.py` (not a TUI path) gets no hyperbola features (direction mapping, Earth orientation, glare) in this change. **Exception (Claude's call):** `sobol/tests/test_instance_parity.py` compares `Sun_Visual` / `Sun_Body` `matrix_world` between the hero script and the instance path on an ephemeris fixture, so the hero script's ephemeris Sun visual takes the same true-size placement and sizing (ideally by importing the shared constants/function rather than its own copies at lines 112-115). The parity test must pass unchanged.
- Lens flares (ghosts, streaks, starburst).

## Anti-goals (fails even if it technically works)

- Glare or exposure that turns Apophis into an unreadable silhouette or washes out Earth in the `auto` shot.
- The Sun lighting the scene twice (lamp + emissive body both illuminating grains/Earth), or casting the emissive body's shadow.
- A hard-coded guess at the hyperbola's frame convention that silently breaks if `moddump_earthflyby.f90` or the sign of the orbit changes.
- Changing how existing ephemeris renders are lit (only the Sun visual's size/brightness changes there).

## Real 2029 geometry (decided)

Source: JPL Horizons API, `CENTER='500@399'` (geocentric), `REF_PLANE='ECLIPTIC'` (Ecliptic of J2000.0), `VECTORS`, km and km/s, queried 2026-10-20. Closest approach at **2029-04-13 21:46:00 TDB** (about 21:44:51 UTC), |r| = 38,011 km.

| Vector (J2000 ecliptic) | Value |
|---|---|
| Apophis geocentric position r_CA (km) | (-19187.2234811485, 32258.7096137982, 6007.9605574541) |
| Apophis geocentric velocity v_CA (km/s) | (6.3321929251, 3.4051973509, 1.8444680692) |
| Geocentric Sun unit vector s_CA | (0.9147059482, 0.4041200656, -0.0000295845) |

Define the real pericentre frame: x = r_CA / |r_CA| (pericentre direction, Earth to Apophis), z = (r_CA x v_CA) normalised (orbit normal), y = z x x (direction of motion at pericentre). The Sun in that frame is **(-0.1188, 0.9655, 0.2318)**: 15 deg from the direction of motion, phase angle 83 deg.

These constants go in code with a provenance comment and a small regeneration script (Horizons query + the frame computation above), so they can be re-derived and audited.

**Sun motion over the encounter:** the geocentric Sun direction turns about the ecliptic pole at about 0.9856 deg/day (~1 deg over a 48 h window, ~27 px at 1920 px across 70 deg). Builder's call: rotate s_CA about the J2000 ecliptic pole by that rate times (t_real - t_CA) per frame, rather than holding it constant.

## Mapping the 2029 geometry onto a hyperbola run (decided)

The stand-in hyperbola has its own frame (`moddump_earthflyby.f90` builds it in the x-y plane, `arg_peri = 0`, `incl = 0`) and its own rp / v_inf, so the turn angle differs from the real flyby. Mapping rule agreed with Max: **align the real pericentre direction and orbit normal with the run's pericentre direction and orbit normal**, then read off the Sun.

- Measure the run's frame **from the run's own bodies CSVs** (Apophis minus Earth positions over all frames), not from an assumed moddump convention:
  - orbit normal z_sim: from positions (e.g. the sum of r_i x r_{i+1}, normalised), sign set by the direction of travel;
  - pericentre direction x_sim and pericentre time t_CA_sim: from the closest approach. Dumps can be sparse (the `flyby_fixture` has 5 frames; Mia's runs ~51), so the builder decides how to get these accurately (e.g. a Kepler conic fit in the orbital plane using Earth's mass from the CSV `m` column, or interpolation) with a target error of 0.5 deg or less on direction.
- Rotation R maps the real frame (x, y, z) to the sim frame (x_sim, y_sim, z_sim). Sun direction in the scene = R s(t_real).
- Time mapping: t_real = t_CA_real + (time_s - t_CA_sim), time_s from the bodies CSV (TDB seconds).
- Hyperbola `time_s` does not start at 0 (Mia's run starts at 23489 s, 50000 km out), and pericentre can fall between frames ~700 s apart, so the closest-approach frame alone is not enough; a fit or interpolation is required.
- Detection: a bodies dir with an `Earth` row and no `Sun` row is a hyperbola run (the existing convention in `DEMCinematicSetup`). Fewer than the frames the method needs: fall back to the current fixed direction with a printed warning, never crash a render.

All of this goes in a new pure-Python module next to `earth_orientation.py` (no `bpy`), so it is unit-testable in `sobol/tests`.

## Earth orientation for hyperbola runs (decided)

Currently `keyframe_bodies_and_earth` spins Earth from `EARTH_EPOCH_TDB + time_s` and prints "Earth time-of-day: arbitrary". For hyperbola runs use the real orientation at t_real, carried into the sim frame by the same rotation: q_scene = q_R x `earth_orientation.earth_quaternion(t_real, ..., frame_rotation_deg=0)`. Keep the unwrapped-GMST handling and `SpinStepCheck`. Check: at t_real = t_CA_real the subsolar point from `ecliptic_to_geographic` matches the real one for 2029-04-13 21:46 TDB.

## Visible Sun (decided in outline; numbers are the builder's)

- One shared function in `DEMCinematicSetup.py` replacing `ensure_sun_visual`'s placement/sizing, used by hyperbola and ephemeris runs, called from both bulk paths (`DEMBulkInstanceBlender.py` already calls it; `DEMBulkVizBlender.py` currently does not and should).
- Emissive sphere (the approach Max already knows from ephemeris renders) placed at the Apophis core + D x sun_direction each frame, with radius D x tan(0.2666 deg). D well beyond Earth (Earth is at true angular size: 0.2 BU/km, so up to ~1e5 BU away at a 4e5 km start) and inside `CAM_CLIP_END` (2e7 BU); e.g. D = 5e6 BU. Parallax from the 150 BU camera offset is negligible.
- Ray visibility: camera only. Not diffuse, glossy, transmission, volume or shadow, so the lamp alone lights the scene (no double lighting, no shadow from the sphere). The lamp's `SUN_LAMP_ANGLE` (0.00931 rad, the full angular diameter) stays.
- Brightness: start from the radiance consistent with the lamp (disk radiance = irradiance / solid angle; the builder confirms how Blender's emission strength maps to radiance), so the Sun clips to white under AgX like a real exposure. One constant scales it for legibility.

## Glare (decided in outline)

- Compositor Glare node (bloom or fog glow; builder picks what Blender 5.2 offers and what looks right), threshold above the brightest lit surface so only the Sun blooms. Applied in the headless render path (`DEMHeadlessRender.py`), where the AgX view transform is set.
- **Scope (Claude's call):** glare is added only when the scene has a `Sun_Visual` (hyperbola and ephemeris runs); `apophis_only` and runs without bodies get none. Because ephemeris renders gain glare, the threshold must keep Earth's sunglint, clouds and fill-lit grains from blooming; check one ephemeris still before and after.
- Constants: enable flag, type, threshold, size/strength. **Legibility rule:** at the default settings, in the `auto` shot frames where the Sun is in view next to Earth, Apophis's grains and Earth's limb must stay readable. If not, reduce glare first, then disk brightness.
- Blender 5.x compositor API differs from 4.x (node groups); confirm against the installed 5.2.1.

## Expected look in the `auto` shot (estimated, to be confirmed)

Hand estimate for rp 8000 km, v_inf 5.9 km/s: on the inbound leg the camera looks back toward Earth and the Sun is ~18 deg off the view axis, so it is in the 70 deg frame near Earth; Earth shows as a crescent; Apophis is back-lit (dark side to camera; the existing camera fill light keeps it readable). Near pericentre the Sun is ~112 deg off-axis, out of frame. The builder confirms this on real data.

## Other camera modes and the composite path (Claude's call)

The real Sun direction, visible Sun and Earth orientation apply to every camera mode and to both bulk paths (instance and composite), since they are scene facts, not camera choices. The legibility check is done on the instance `auto` shot only. `earth_chase`, `top_down` and `earth_fixed` are not tuned in this change; the builder renders one still of `earth_chase` on the verification run against the existing `run_0001_render_instance_grains_earth_chase_slowmo5` and reports any silhouette problem rather than fixing it. The composite path (no camera fill light today) gets no new fill light, so composite lighting is unchanged apart from direction.

## Constraints

- Cycles only (no EEVEE reasoning). Blender 5.2.1, Python 3.13; `bpy` code validated headless with `--python-exit-code 1`.
- Renders must stay self-contained and portable (`honours_paths.py` fallbacks; no new assets).
- Shared Earth/Moon/Sun code lives in `DEMCinematicSetup.py`; geometry maths in a pure-Python module with tests in `sobol/tests` (pattern: `earth_orientation.py` + `test_earth_orientation.py`).
- No commit/push until Max has tested.

## Verification

- Unit tests (pure Python): real-frame constants reproduce (-0.1188, 0.9655, 0.2318); the mapping recovers a known rotation on a synthetic hyperbola (sparse and dense sampling, both orbit senses); Sun direction for an ephemeris-like case is unchanged; Earth subsolar point at t_CA matches the real one.
- Render check on real hyperbola data: `DEMCSVs/mia_full_B_s1_cb_x0.7/run_0001_*` (51 frames, already has `auto` and `earth_chase` renders to compare), `auto` camera, a few frames on the inbound leg (Sun in frame) and around pericentre. Max judges realism vs legibility on these stills before any full render.
- One ephemeris run render to confirm the true-size Sun and unchanged lighting.

## Left to the builder

Module and function names; exact frame-measurement method within the 0.5 deg target; D; emission strength mapping; glare node type and numbers (within the legibility rule); how the headless driver exposes the glare constants; the regeneration script's location.
