# Real 2029 Sun in Hyperbola Renders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hyperbola renders get the real 2029 Sun (direction, true apparent size, brightness, glare) and the real 2029 sunlit Earth; ephemeris renders get the same true-size Sun visual.

**Architecture:** A new pure-Python module `Code/BlenderConvert/sun_geometry.py` (no `bpy`, like `earth_orientation.py`) holds the Horizons constants, fits the run's hyperbola from its bodies CSVs (linear conic least squares), and maps the real pericentre frame onto it. `DEMCinematicSetup.py` uses it to key the lamp, a camera-only emissive Sun sphere and Earth's orientation; `DEMHeadlessRender.py` adds compositor glare when a Sun visual exists.

**Tech Stack:** Python 3 + numpy (pure module, tested in WSL pytest), Blender 5.2.1 `bpy` / Cycles (Windows `blender.exe`, driven from WSL tests), JPL Horizons API (stdlib `urllib`, regeneration script only).

**Spec:** `sobol/docs/superpowers/specs/2026-10-20-hyperbola-sun-design.md`

## Global Constraints

- Cycles only; Blender 5.2.1, Python 3.13; validate headless with `--python-exit-code 1`.
- No change to the simulation, `phantomflyby`, `encounter.py`, or the bodies CSV format.
- `apophis_only` runs (no `Earth` row) keep `APOPHIS_ONLY_SUN_DIRECTION` and get no Sun visual or glare.
- Hero script `DEMGrainsBlenderEarthCam.py`: only the ephemeris Sun visual's placement/sizing changes (parity); no hyperbola features.
- Ephemeris lamp direction and Earth orientation stay exactly as now.
- No new assets; renders stay portable (`honours_paths.py` fallbacks).
- **No commits or pushes.** Max commits after testing. Both repos already have uncommitted changes in files this plan touches (`DEMCinematicSetup.py`, `DEMHeadlessRender.py`, `DEMBulkVizBlender.py`, `DEMBulkInstanceBlender.py`, sobol tests): edit on top of them, never stash, revert or `git checkout` them.
- Tests: `cd sobol && python3 -m pytest tests/<file> -v`. Blender tests skip without `blender.exe`; they must run (not skip) on Max's laptop.

## Review Focus

1. **Short or one-sided runs** (tmax ends before pericentre, or only 2 bodies CSVs): the conic fit needs no pericentre sample; fewer than 3 frames returns `None` and the render falls back to the fixed direction with a printed warning. Test in Task 2.
2. **Other orbit orientations** (retrograde, pericentre not along -y, non-unit `perturber_earth_masses`): the mapping must come from data, not moddump conventions. Synthetic tests in Task 2 cover both senses, a rotated plane and 2 Earth masses.
3. **Manifest windows** (`bodies_start_index` > 0, tail-only instance manifests): the fit reads every CSV in the bodies dir, not the window. Test in Task 3.
4. **Ephemeris and `apophis_only` regressions**: a `Sun` row means ephemeris (no hyperbola mapping, lamp keys unchanged); no `Earth` row means fixed direction and no Sun visual. Tests in Task 3.
5. **Earth occluding the Sun**: the Sun sphere must sit beyond Earth in every frame; if Earth's scene distance ever exceeds `SUN_VISUAL_DISTANCE_BU / 10`, print a warning. Test in Task 3.

---

### Task 1: Real 2029 constants, pericentre frame and Sun visual helpers

**Files:**
- Create: `Code/BlenderConvert/sun_geometry.py`
- Create: `Code/BlenderConvert/regen_sun_constants.py`
- Test: `sobol/tests/test_sun_geometry.py`

**Interfaces:**
- Produces (all in `sun_geometry`):
  - `CA_EPOCH_TDB = '2029-04-13 21:46:00'` (str, parseable by `earth_orientation.julian_date`)
  - `R_CA_KM = (-19187.2234811485, 32258.7096137982, 6007.9605574541)`
  - `V_CA_KMS = (6.3321929251, 3.4051973509, 1.8444680692)`
  - `SUN_CA_UNIT = (0.9147059482, 0.4041200656, -0.0000295845)` (geocentric, J2000 ecliptic)
  - `SUN_DISTANCE_CA_KM = 150035116.9`, `SUN_RADIUS_KM = 695700.0`
  - `SUN_ANGULAR_RADIUS_RAD = math.asin(SUN_RADIUS_KM / SUN_DISTANCE_CA_KM)` (= 0.004637 rad, 0.2657 deg; the spec's 0.2666 deg is the 1 AU mean, this is the 2029-04-13 value)
  - `SUN_ECLIPTIC_RATE_RAD_S = 2*math.pi / (365.256363 * 86400.0)`
  - `SUN_VISUAL_DISTANCE_BU = 1.5e7` (inside `CAM_CLIP_END` 2e7; parallax from `earth_fixed` camera at ~8e4 BU is <= 0.3 deg, accepted)
  - `pericentre_frame(r, v) -> np.ndarray` shape (3, 3), rows x (r unit), y (z cross x), z ((r cross v) unit)
  - `sun_direction_j2000(dt_s: float) -> np.ndarray` (3,) unit: `SUN_CA_UNIT` rotated about +z by `SUN_ECLIPTIC_RATE_RAD_S * dt_s` (dt from closest approach)
  - `sun_visual_radius_bu() -> float` = `SUN_VISUAL_DISTANCE_BU * math.tan(SUN_ANGULAR_RADIUS_RAD)`
  - `sun_visual_location(direction) -> tuple[float, float, float]` = `SUN_VISUAL_DISTANCE_BU * unit(direction)` (Apophis-centred scene)
  - `sun_disk_emission_strength(lamp_irradiance: float) -> float` = `SUN_EMISSION_CALIBRATION * lamp_irradiance / SUN_ANGULAR_RADIUS_RAD**2 * SUN_DISK_BRIGHTNESS_SCALE`, with `SUN_EMISSION_CALIBRATION = 1.0` (Task 5 measures and pins it) and `SUN_DISK_BRIGHTNESS_SCALE = 1.0` (legibility knob)

- [ ] **Step 1: Write the failing tests** in `sobol/tests/test_sun_geometry.py` (import pattern as `test_earth_orientation.py`: `sys.path.insert(0, str(CODE / 'BlenderConvert'))`; `import sun_geometry as sg`)

```python
def test_real_sun_in_pericentre_frame():
    F = sg.pericentre_frame(sg.R_CA_KM, sg.V_CA_KMS)
    np.testing.assert_allclose(F @ np.array(sg.SUN_CA_UNIT), [-0.1188, 0.9655, 0.2318], atol=5e-4)
    np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-12)
    assert np.linalg.norm(sg.R_CA_KM) == pytest.approx(38011.4, abs=0.1)

def test_sun_direction_turns_one_degree_per_day():
    a, b = sg.sun_direction_j2000(0.0), sg.sun_direction_j2000(86400.0)
    np.testing.assert_allclose(a, sg.SUN_CA_UNIT, atol=1e-12)
    assert math.degrees(math.acos(a @ b)) == pytest.approx(0.98561, abs=1e-4)

def test_sun_visual_true_angular_size():
    assert math.atan(sg.sun_visual_radius_bu() / sg.SUN_VISUAL_DISTANCE_BU) == pytest.approx(0.004637, abs=2e-6)
    loc = sg.sun_visual_location((2.0, 0.0, 0.0))
    assert loc == pytest.approx((1.5e7, 0.0, 0.0))

def test_emission_strength_scales_with_lamp():
    assert sg.sun_disk_emission_strength(20.0) == pytest.approx(2 * sg.sun_disk_emission_strength(10.0))
```

- [ ] **Step 2: Run** `cd sobol && python3 -m pytest tests/test_sun_geometry.py -v`. Expected: FAIL, `ModuleNotFoundError: sun_geometry`.

- [ ] **Step 3: Implement the constants and functions above in `Code/BlenderConvert/sun_geometry.py`.** Module docstring in the style of `earth_orientation.py`: frames (J2000 ecliptic, real pericentre frame, scene frame), Horizons provenance (API, `CENTER='500@399'`, `REF_PLANE='ECLIPTIC'`, `VECTORS`, km and km/s, queried 2026-10-20, closest approach 2029-04-13 21:46:00 TDB, 38,011 km), and "regenerate with `regen_sun_constants.py`".

- [ ] **Step 4: Write `Code/BlenderConvert/regen_sun_constants.py`**: stdlib only (`urllib`); queries Horizons `VECTORS` for `COMMAND='99942'` and `'10'`, `CENTER='500@399'`, `REF_PLANE='ECLIPTIC'`, `OUT_UNITS='KM-S'`, `VEC_TABLE='2'`, `CSV_FORMAT='YES'`, 1-minute steps over 2029-04-13 21:30–22:00 TDB; takes the minimum-|r| row; prints the constant lines for `sun_geometry.py` and the Sun in `pericentre_frame` (imports `sun_geometry.pericentre_frame`). No test (network); run it once and confirm its output matches the constants.

- [ ] **Step 5: Run** the tests again. Expected: 4 PASS. Run `python3 Code/BlenderConvert/regen_sun_constants.py`; expected printout matches the Step 3 constants to the digits given and Sun-in-frame `(-0.1188, 0.9655, 0.2318)`.

- [ ] **Step 6: Checkpoint** (no commit). `git -C Code status --short` shows the two new files.

---

### Task 2: Fit the run's hyperbola and map the 2029 geometry onto it

**Files:**
- Modify: `Code/BlenderConvert/sun_geometry.py`
- Test: `sobol/tests/test_sun_geometry.py`

**Interfaces:**
- Consumes: Task 1 constants, `pericentre_frame`, `sun_direction_j2000`; `earth_orientation.earth_quaternion(time_s, epoch, frame_rotation_deg)`.
- Produces:
  - `GM_SUN_KM3_S2 = 1.32712440018e11` (bodies CSV `m` is in solar masses)
  - `@dataclass(frozen=True) HyperbolaFrame`: `rotation: np.ndarray` (3x3, J2000 ecliptic → scene), `t_ca_sim_s: float`, `rp_km: float`, `ecc: float`
  - `fit_hyperbola_frame(times_s, rel_km, mu_km3_s2) -> HyperbolaFrame | None` (`rel_km` = Apophis − Earth, shape (n, 3); `None` when n < 3)
  - `hyperbola_frame_from_bodies_dir(bodies_dir: str) -> HyperbolaFrame | None`: reads **every** `*.csv` (sorted) with stdlib `csv`; `None` when the first CSV has a `Sun` row, has no `Earth` row, or fewer than 3 files have both `Earth` and `Apophis`; `mu = Earth m * GM_SUN_KM3_S2`
  - `sun_direction_scene(frame: HyperbolaFrame, time_s: float) -> np.ndarray` (3,) unit = `frame.rotation @ sun_direction_j2000(time_s - frame.t_ca_sim_s)`
  - `earth_quaternion_scene(frame: HyperbolaFrame, time_s: float) -> tuple[float, float, float, float]` (w, x, y, z) = q(rotation) ⊗ `earth_orientation.earth_quaternion(time_s - frame.t_ca_sim_s, CA_EPOCH_TDB, frame_rotation_deg=0.0)`
  - `matrix_to_quaternion(m) -> tuple` (w, x, y, z), w >= 0

**Fit algorithm (pinned; it reproduced Mia's run to 1e-7 km and t_peri to 0.04 s):**

```text
z  = unit( sum_i r_i x r_{i+1} )            # time-ordered, so the sense of travel sets the sign
u  = unit( r_0 - (r_0.z) z ),  w = z x u     # any in-plane basis
solve least squares  |r_i| = p - ex*(r_i.u) - ey*(r_i.w)   for (p, ex, ey)   # Kepler conic, linear
e  = hypot(ex, ey);  x_sim = unit(ex*u + ey*w);  rp = p / (1 + e)
y_sim = z x x_sim
a = p / (1 - e^2);  n = sqrt(mu / |a|^3)
per sample: theta = atan2(r.y_sim, r.x_sim); F = 2 atanh( sqrt((e-1)/(e+1)) tan(theta/2) ); M = e sinh F - F
t_ca_sim = mean(t_i - M_i / n)
rotation = S.T @ pericentre_frame(R_CA_KM, V_CA_KMS)    with S rows (x_sim, y_sim, z)
```

- [ ] **Step 1: Write the failing tests.** Add a synthetic Kepler-hyperbola generator to the test file (Newton solve of `e sinh F - F = n t`, then `x = a(e - cosh F)`, `y = a sqrt(e^2-1) sinh F`, rotated by a given 3x3 matrix) and a Rodrigues helper `_rotation_about(axis, angle)`:

```python
def _synthetic(rp, vinf, mu, rot, times, retro=False):
    """Kepler hyperbola about the origin: pericentre on +x, normal +z (or -z if retro), then rotated."""
    a = mu / vinf**2; e = 1 + rp * vinf**2 / mu; n = math.sqrt(mu / a**3)
    out = []
    for t in times:
        M = n * t; F = math.asinh(M / e)
        for _ in range(50):
            F -= (e * math.sinh(F) - F - M) / (e * math.cosh(F) - 1)
        x = a * (e - math.cosh(F)); y = a * math.sqrt(e * e - 1) * math.sinh(F)
        out.append(rot @ np.array([x, -y if retro else y, 0.0]))
    return np.array(out)

@pytest.mark.parametrize("retro", [False, True])
@pytest.mark.parametrize("n_frames", [5, 51])
def test_fit_recovers_synthetic_hyperbola(retro, n_frames):
    rot = _rotation_about((1, 2, 3), 0.7)          # helper: Rodrigues
    mu = 2 * 398600.4                               # 2 Earth masses
    times = np.linspace(-30000, 20000, n_frames) + 12345.0
    rel = _synthetic(8000.0, 5.9, mu, rot, times - 12345.0, retro)
    fr = sg.fit_hyperbola_frame(times, rel, mu)
    assert fr.rp_km == pytest.approx(8000.0, rel=1e-6)
    assert fr.t_ca_sim_s == pytest.approx(12345.0, abs=1.0)
    x_sim = fr.rotation @ sg.pericentre_frame(sg.R_CA_KM, sg.V_CA_KMS)[0]
    np.testing.assert_allclose(x_sim, rot @ [1, 0, 0], atol=1e-6)
    z_sim = fr.rotation @ sg.pericentre_frame(sg.R_CA_KM, sg.V_CA_KMS)[2]
    np.testing.assert_allclose(z_sim, rot @ [0, 0, -1 if retro else 1], atol=1e-6)

def test_fit_inbound_only():
    rel = _synthetic(8000.0, 5.9, 398600.4, np.eye(3), np.linspace(-40000, -5000, 6))
    fr = sg.fit_hyperbola_frame(np.linspace(-40000, -5000, 6), rel, 398600.4)
    assert fr.t_ca_sim_s == pytest.approx(0.0, abs=1.0)

def test_fit_needs_three_frames():
    assert sg.fit_hyperbola_frame([0.0, 1.0], np.ones((2, 3)), 398600.4) is None

def test_sun_scene_keeps_real_pericentre_components():
    rot = _rotation_about((0, 1, 1), 1.1)
    rel = _synthetic(8000.0, 5.9, 398600.4, rot, np.linspace(-3e4, 3e4, 9))
    fr = sg.fit_hyperbola_frame(np.linspace(-3e4, 3e4, 9), rel, 398600.4)
    s = sg.sun_direction_scene(fr, 0.0)
    np.testing.assert_allclose([s @ (rot @ [1, 0, 0]), s @ (rot @ [0, 1, 0]), s @ (rot @ [0, 0, 1])],
                               [-0.1188, 0.9655, 0.2318], atol=5e-4)

def test_earth_subsolar_point_at_closest_approach():
    # Real 2029-04-13 21:46 TDB subsolar point: lon -146.1 E, lat +9.4 (declination).
    fr = sg.HyperbolaFrame(rotation=_rotation_about((1, 0, 0), 0.4), t_ca_sim_s=500.0, rp_km=8000.0, ecc=1.7)
    q = sg.earth_quaternion_scene(fr, 500.0)
    s = sg.sun_direction_scene(fr, 500.0)
    body = eo.rotate((q[0], -q[1], -q[2], -q[3]), s)
    assert math.degrees(math.atan2(body[1], body[0])) == pytest.approx(-146.1, abs=0.5)
    assert math.degrees(math.asin(body[2])) == pytest.approx(9.4, abs=0.5)

@pytest.mark.skipif(not MIA_BODIES.is_dir(), reason="needs Max's DEMCSVs")
def test_fit_on_mia_run():
    # MIA_BODIES = DATA / 'DEMCSVs/mia_full_B_s1_cb_x0.7/run_0001_bodies_output'
    fr = sg.hyperbola_frame_from_bodies_dir(str(MIA_BODIES))
    assert fr.rp_km == pytest.approx(7637.0, abs=0.5)
    assert fr.t_ca_sim_s == pytest.approx(29822.2, abs=1.0)

def test_bodies_dir_rules(tmp_path):
    # write tiny CSVs (header: name,time_s,x_au,y_au,z_au,x_km,y_km,z_km,h,Reff,m)
    # Sun row present -> None; no Earth row -> None; 2 files -> None; 3 synthetic files -> HyperbolaFrame
    ...
```

(`test_bodies_dir_rules`: the implementer writes the four cases exactly as the comment lists; `MIA_BODIES` from `tests._paths.DATA`.)

- [ ] **Step 2: Run** `cd sobol && python3 -m pytest tests/test_sun_geometry.py -v`. Expected: new tests FAIL (`AttributeError: fit_hyperbola_frame`).

- [ ] **Step 3: Implement the Produces list in `sun_geometry.py`** with the pinned fit algorithm. `earth_quaternion_scene` composes with `earth_orientation._quat_mul` (same module family; keep quaternion maths in one place).

- [ ] **Step 4: Run** the tests. Expected: all PASS (Mia test runs, not skips, on the laptop).

- [ ] **Step 5: Checkpoint** (no commit).

---

### Task 3: Scene wiring in `DEMCinematicSetup.py` and the composite path

**Files:**
- Modify: `Code/BlenderConvert/DEMCinematicSetup.py` (`ensure_sun_lamp` ~698, `ensure_sun_visual` ~752, `keyframe_bodies_and_earth` ~310, constants ~47-90)
- Modify: `Code/BlenderConvert/DEMBulkVizBlender.py:332` (add `cinematic.ensure_sun_visual(has_solar, bodies_dir)` after `ensure_earth_placeholder`)
- Test: `sobol/tests/test_hyperbola_sun_scene.py` (Blender, pattern of `test_dem_headless_paths.py` `subprocess.run([BLENDER_EXE, "--background", "--factory-startup", "--python-exit-code", "1", "--python", script])` with a script written to `WIN_TEMP`)

**Interfaces:**
- Consumes: `sun_geometry.hyperbola_frame_from_bodies_dir`, `sun_direction_scene`, `earth_quaternion_scene`, `sun_visual_radius_bu`, `sun_visual_location`, `sun_disk_emission_strength`, `SUN_VISUAL_DISTANCE_BU`.
- Produces:
  - `ensure_sun_visual(has_solar, bodies_dir)`: creates `Sun_Visual` Empty + `Sun_Body` sphere when the bodies dir has a `Sun` row **or** `hyperbola_frame_from_bodies_dir` is not `None`; radius `sun_visual_radius_bu()`; emission strength `sun_disk_emission_strength(SUN_LAMP_ENERGY)`; `Sun_Body` ray visibility: `visible_camera=True`, `visible_diffuse=visible_glossy=visible_transmission=visible_volume_scatter=visible_shadow=False`. Remove `SUN_VIS_SCALE`, `SUN_BODY_RADIUS_BU`, `SUN_EMISSION_STRENGTH` constants.
  - `keyframe_bodies_and_earth(...)` unchanged signature. Per frame: ephemeris (`Sun` row) keeps the lamp code; `Sun_Visual` location = `sun_visual_location(sun_km - apo_km)`. Hyperbola (`frame = sun_geometry.hyperbola_frame_from_bodies_dir(bodies_dir)` computed once, before the loop, from the whole dir): lamp rotation from `-sun_direction_scene(frame, time_s)` (same `to_track_quat('-Z', 'Y')` call as the ephemeris branch), `Sun_Visual` location `sun_visual_location(sun_direction_scene(...))`, Earth Surface/Clouds quaternion `earth_quaternion_scene(frame, time_s)`; log line "Earth time-of-day: real 2029 (hyperbola mapped to closest approach)". `frame is None` on a hyperbola-like dir: keep today's fixed direction and print `"WARNING: hyperbola Sun fit unavailable (<3 frames); using fixed lamp direction, no Sun visual"`.
  - Earth-occlusion guard: once per run, if any keyed Earth distance (BU) > `SUN_VISUAL_DISTANCE_BU / 10`, print a warning naming the distance.
  - `ensure_sun_lamp`: the fixed `APOPHIS_ONLY_SUN_DIRECTION` stays as the initial pose; keys override it on hyperbola runs.

- [ ] **Step 1: Write the failing Blender tests** in `sobol/tests/test_hyperbola_sun_scene.py`. Each builds a scene by calling `ensure_sun_lamp`, `ensure_earth_placeholder`, `ensure_sun_visual`, `keyframe_bodies_and_earth` on a bodies dir, then prints JSON of the checked values; the WSL side asserts:

```python
def test_hyperbola_scene_has_real_sun():           # DATA/DEMCSVs/mia_full_B_s1_cb_x0.7/run_0001_bodies_output
    # at frame 1 and frame 40:
    #   lamp -Z axis (world) == -sun_geometry.sun_direction_scene(fr, time_s)  (atol 1e-5)
    #   Sun_Visual location == sun_visual_location(sun_direction_scene(...))  (rel 1e-6)
    #   Sun_Body dimensions.x / 2 == sun_visual_radius_bu()                   (rel 1e-4)
    #   Sun_Body.visible_camera and not (visible_diffuse or visible_glossy or visible_shadow)
    #   Earth Surface rotation_quaternion == earth_quaternion_scene(fr, time_s) (atol 1e-5, up to sign)

def test_start_index_window_uses_whole_dir():       # same dir, start_index=30
    # Sun_Visual location at frame 1 == value for CSV index 30 from the whole-dir fit

def test_ephemeris_scene_lamp_unchanged():          # FIXTURE_RUN bodies (has a Sun row)
    # lamp keys == direction from Sun row minus Apophis (as before)
    # Sun_Visual location == sun_visual_location(sun_km - apo_km); radius == sun_visual_radius_bu()

def test_apophis_only_has_no_sun_visual(tmp_path):  # bodies CSVs with Apophis rows only
    # bpy.data.objects.get('Sun_Visual') is None; lamp -Z == -APOPHIS_ONLY_SUN_DIRECTION

def test_two_frame_hyperbola_falls_back(tmp_path):  # copy 2 Mia CSVs
    # stdout contains "hyperbola Sun fit unavailable"; no Sun_Visual; render setup does not raise
```

- [ ] **Step 2: Run** `cd sobol && python3 -m pytest tests/test_hyperbola_sun_scene.py -v`. Expected: FAIL (no `Sun_Visual` on hyperbola; radius mismatch on ephemeris).

- [ ] **Step 3: Implement the Produces list** in `DEMCinematicSetup.py` and the one-line call in `DEMBulkVizBlender.py`.

- [ ] **Step 4: Run** the new tests plus the existing Blender and framing suites: `python3 -m pytest tests/test_hyperbola_sun_scene.py tests/test_dem_headless_paths.py tests/test_camera_framing.py tests/test_earth_orientation.py -v`. Expected: all PASS. (`test_instance_parity.py` is expected to fail until Task 4.)

- [ ] **Step 5: Checkpoint** (no commit).

---

### Task 4: Hero-script parity for the ephemeris Sun visual

**Files:**
- Modify: `Code/BlenderConvert/DEMGrainsBlenderEarthCam.py` (constants 112-115, Sun visual block ~773-806, keyframe block ~989-997; `import sun_geometry` beside `import earth_orientation` at line 38)
- Test: `sobol/tests/test_instance_parity.py` (unchanged)

**Interfaces:**
- Consumes: `sun_geometry.sun_visual_radius_bu()`, `sun_visual_location(rel_km)`, `sun_disk_emission_strength(SUN_LAMP_ENERGY)`.
- Produces: hero `Sun_Visual` / `Sun_Body` with the same `matrix_world` and radius as the instance path on ephemeris runs; same camera-only ray visibility. Remove the hero's `SUN_VIS_SCALE`, `SUN_BODY_RADIUS_BU`, `SUN_EMISSION_STRENGTH`. Hyperbola behaviour of the hero script unchanged (still no Sun visual there).

- [ ] **Step 1: Run** `cd sobol && python3 -m pytest tests/test_instance_parity.py -v`. Expected: `test_scene_objects_match` FAIL on `Sun_Visual`/`Sun_Body` (Task 3 changed the instance side).
- [ ] **Step 2: Implement** the swap in the hero script.
- [ ] **Step 3: Run** the parity test again. Expected: PASS, test file unchanged.
- [ ] **Step 4: Checkpoint** (no commit).

---

### Task 5: Disk brightness calibration and compositor glare

**Files:**
- Modify: `Code/BlenderConvert/sun_geometry.py` (`SUN_EMISSION_CALIBRATION`)
- Modify: `Code/BlenderConvert/DEMHeadlessRender.py` (constants near the top; new `apply_sun_glare`; call in `render_prepared_scene` before `bpy.ops.render.render`)
- Test: `sobol/tests/test_sun_render.py` (Blender)

**Interfaces:**
- Consumes: `sun_geometry.sun_disk_emission_strength`, `SUN_ANGULAR_RADIUS_RAD`; `Sun_Visual` object from Task 3.
- Produces:
  - `SUN_EMISSION_CALIBRATION` pinned to the measured value.
  - In `DEMHeadlessRender`: `SUN_GLARE_ENABLED = True`, `SUN_GLARE_TYPE`, `SUN_GLARE_THRESHOLD`, `SUN_GLARE_SIZE`, `SUN_GLARE_STRENGTH` (module constants, no CLI flag), and `apply_sun_glare(scene) -> bool`: returns `False` and touches nothing when `SUN_GLARE_ENABLED` is false or `bpy.data.objects.get('Sun_Visual')` is `None`; otherwise builds Render Layers → Glare → Composite in the Blender 5.2 compositor node group, enables compositing, returns `True`. Idempotent (a second call reuses the group).

- [ ] **Step 1: Probe the 5.2 API.** Headless one-off: print `bpy.types.CompositorNodeGlare` properties/inputs and how a scene gets a compositor node tree in 5.2 (`scene.compositing_node_group` vs `scene.node_tree`). Record the answer as a comment above `apply_sun_glare`.
- [ ] **Step 2: Write the failing Blender tests:**

```python
def test_sun_disk_matches_lamp_radiance():
    # Standard view transform, 32-bit EXR, 64x64, Cycles CPU 16 samples, black world.
    # Scene A: white diffuse plane (albedo 1) facing a Sun lamp of energy 10, camera looking at it -> mean pixel P_a.
    # Scene B: Sun_Body sphere (camera-only), strength sun_disk_emission_strength(10), filling the frame -> P_b.
    # assert P_b / P_a == approx(1 / SUN_ANGULAR_RADIUS_RAD**2, rel=0.05)

def test_glare_only_with_sun_visual():
    # empty scene -> apply_sun_glare(scene) is False and the scene has no compositor glare node
    # scene with a Sun_Visual -> True; exactly one Glare node; calling twice still one

def test_glare_spares_lit_surfaces():
    # Sun lamp energy 10 + the DEMCinematicSetup camera fill light on a white diffuse sphere, no Sun in frame:
    # max pixel with apply_sun_glare == max pixel without (abs 1e-3)  -> threshold is above lit surfaces
```

- [ ] **Step 3: Run** `cd sobol && python3 -m pytest tests/test_sun_render.py -v`. Expected: calibration test FAIL unless the constant happens to be right; glare tests FAIL (`apply_sun_glare` missing).
- [ ] **Step 4: Implement.** Set `SUN_EMISSION_CALIBRATION` from the measured `P_b/P_a` ratio; implement `apply_sun_glare` with the probed API; start glare at bloom (or fog glow if bloom is absent), threshold above the brightest lit surface in `test_glare_spares_lit_surfaces`.
- [ ] **Step 5: Run** the three tests plus `tests/test_dem_headless_paths.py` and `tests/test_instance_parity.py`. Expected: all PASS.
- [ ] **Step 6: Checkpoint** (no commit).

---

### Task 6: Render check on real data, legibility tuning, docs

**Files:**
- Modify (tuning only): glare constants in `DEMHeadlessRender.py`, `SUN_DISK_BRIGHTNESS_SCALE` in `sun_geometry.py`
- Modify: `Code/CLAUDE.md` (Pipelines, hyperbola line: replace "the lamp gets the fixed `APOPHIS_ONLY_SUN_DIRECTION`" with the real-2029 mapping, `sun_geometry.py`, true-size camera-only Sun visual, glare), `Code/docs/BLENDER_OPS.md` (new constants), `../docs/CONTEXT_CHANGELOG.md` (dated entry)

- [ ] **Step 1: Render stills** with `python3 sobol/render_dumps.py` (or `DEMHeadlessRender.py` directly) on `DEMCSVs/mia_full_B_s1_cb_x0.7/run_0001`, instance path, `--camera-mode auto`, default fov, into a new output dir (do not overwrite the existing renders): frames where `sun_direction_scene` is within the camera FOV (compute and list them from the keyed camera; expected on the inbound leg) plus 2 around pericentre (`t_ca_sim_s` = 29822 s). Expected: Sun visible near Earth in inbound frames, Earth crescent with the 2029 sunlit side (Pacific subsolar at closest approach), Apophis grains readable.
- [ ] **Step 2: Legibility rule.** If Apophis grains or Earth's limb are unreadable in any Step 1 frame, lower glare first, then `SUN_DISK_BRIGHTNESS_SCALE`; re-render those frames. Record the final values.
- [ ] **Step 3: `earth_chase` still** on the same run (one inbound frame) next to the matching frame of `run_0001_render_instance_grains_earth_chase_slowmo5`. Report any silhouette problem; do not fix.
- [ ] **Step 4: Ephemeris still** on `FIXTURE_RUN` (`DEMCSVs/_verify_lattice_fix/sobol_20260705_131545_baseline_58deg`) before/after: lighting unchanged; Sun smaller; no bloom on Earth sunglint or clouds.
- [ ] **Step 5: Full suite** `cd sobol && python3 -m pytest tests -q`. Expected: no new failures versus a run before Task 1 (record both counts).
- [ ] **Step 6: Docs** as listed in Files.
- [ ] **Step 7: Hand to Max** with the still paths, the final constants, and the Step 3 report. No commit.
