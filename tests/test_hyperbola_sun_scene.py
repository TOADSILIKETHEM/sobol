"""Real 2029 Sun and Earth orientation in the shared Blender scene (DEMCinematicSetup.py).

Each test builds a scene headless with ensure_sun_lamp / ensure_earth_placeholder /
ensure_sun_visual / keyframe_bodies_and_earth on one bodies dir, then checks the
sampled lamp, Sun_Visual and Earth keys against sun_geometry (pure Python) here.
"""
import csv
import json
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tests._paths import CODE, DATA
from tests.test_dem_headless_paths import (
    BLENDER_EXE, BLENDERCONVERT_WIN, FIXTURE_RUN, WIN_TEMP, WIN_TEMP_WIN,
)

sys.path.insert(0, str(CODE / 'BlenderConvert'))
import earth_orientation as eo  # noqa: E402
import sun_geometry as sg  # noqa: E402

MIA_BODIES = DATA / 'DEMCSVs/mia_full_B_s1_cb_x0.7/run_0001_bodies_output'
EPHEMERIS_BODIES = FIXTURE_RUN / 'run_0001_bodies_output'
APOPHIS_ONLY_SUN_DIRECTION = (1.0, 0.0, 0.0)

pytestmark = [
    pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed"),
    pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted"),
]

_SCENE = '''
import json, sys, traceback
sys.path.insert(0, {bc!r})
import bpy
from mathutils import Vector
import DEMCinematicSetup as cin
try:
    bodies = {bodies!r}
    has_solar = cin.detect_solar_system(bodies)
    cin.ensure_sun_lamp(has_solar, bodies)
    earth = cin.ensure_earth_placeholder(has_solar, bodies)
    cin.ensure_sun_visual(has_solar, bodies)
    earth = cin.keyframe_bodies_and_earth(bodies, {frame_count!r}, has_solar, earth,
                                          start_index={start_index!r})
    scene = bpy.context.scene
    lamp = bpy.data.objects['DEM_SunLight']
    sv = bpy.data.objects.get('Sun_Visual')
    sb = bpy.data.objects.get('Sun_Body')
    surf = None
    if earth is not None:
        surf = next((c for c in earth.children_recursive if c.name.startswith('Surface')), None)
    out = {{'frame_start': scene.frame_start, 'sun_visual': sv is not None, 'frames': {{}}}}
    if sb is not None:
        out['sun_body'] = {{
            'radius': sb.dimensions.x / 2.0,
            'visible_camera': sb.visible_camera,
            'visible_diffuse': sb.visible_diffuse,
            'visible_glossy': sb.visible_glossy,
            'visible_transmission': sb.visible_transmission,
            'visible_volume_scatter': sb.visible_volume_scatter,
            'visible_shadow': sb.visible_shadow,
        }}
    for f in {frames!r}:
        scene.frame_set(f)
        rec = {{'lamp_neg_z': list(lamp.rotation_euler.to_matrix() @ Vector((0.0, 0.0, -1.0)))}}
        if sv is not None:
            rec['sun_visual'] = list(sv.location)
        if surf is not None:
            rec['earth_q'] = list(surf.rotation_quaternion)
        out['frames'][str(f)] = rec
    print('SCENE_JSON:' + json.dumps(out))
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


def _win(path):
    """WSL path under WIN_TEMP, or a data path, as blender.exe sees it."""
    from tests._paths import _paths
    return _paths.to_windows_path(path)


def build_scene(name, bodies_dir, frames, frame_count, start_index=0):
    """Build the scene in blender.exe; returns (parsed JSON, combined stdout+stderr)."""
    script = WIN_TEMP / f"hsun_{name}.py"
    script.write_text(_SCENE.format(
        bc=BLENDERCONVERT_WIN, bodies=_win(bodies_dir), frames=list(frames),
        frame_count=frame_count, start_index=start_index,
    ), encoding="utf-8")
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", f"{WIN_TEMP_WIN}/hsun_{name}.py"],
        capture_output=True, text=True, check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    line = next(ln for ln in combined.splitlines() if ln.startswith('SCENE_JSON:'))
    return json.loads(line[len('SCENE_JSON:'):]), combined


def _rows(path):
    with open(path, newline='') as f:
        return {row['name']: row for row in csv.DictReader(f)}


def _km(row):
    return np.array([float(row[k]) for k in ('x_km', 'y_km', 'z_km')])


def _csvs(bodies_dir):
    return sorted(bodies_dir.glob('*.csv'))


def _assert_quat(got, want):
    got, want = np.array(got), np.array(want)
    if got @ want < 0:
        got = -got
    np.testing.assert_allclose(got, want, atol=1e-5)


def _fresh_dir(name):
    d = WIN_TEMP / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    return d


def _write_bodies(path, rows):
    """rows: (name, time_s, (x_km, y_km, z_km), m)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'time_s', 'x_au', 'y_au', 'z_au', 'x_km', 'y_km', 'z_km', 'h', 'Reff', 'm'])
        for name, t, (x, y, z), m in rows:
            au = 1.495978707e8
            w.writerow([name, t, x / au, y / au, z / au, x, y, z, 1.0, 1.0, m])


def test_hyperbola_scene_has_real_sun():
    csvs = _csvs(MIA_BODIES)
    frame = sg.hyperbola_frame_from_bodies_dir(MIA_BODIES)
    out, combined = build_scene("hyperbola", MIA_BODIES, [1, 40], len(csvs))
    assert out['sun_visual']
    sb = out['sun_body']
    assert sb['radius'] == pytest.approx(sg.sun_visual_radius_bu(), rel=1e-4)
    assert sb['visible_camera']
    assert not (sb['visible_diffuse'] or sb['visible_glossy'] or sb['visible_shadow']
                or sb['visible_transmission'] or sb['visible_volume_scatter'])
    for f in (1, 40):
        t = float(_rows(csvs[f - out['frame_start']])['Apophis']['time_s'])
        rec = out['frames'][str(f)]
        sun = sg.sun_direction_scene(frame, t)
        np.testing.assert_allclose(rec['lamp_neg_z'], -sun, atol=1e-5)
        np.testing.assert_allclose(rec['sun_visual'], sg.sun_visual_location(sun), rtol=1e-6)
        _assert_quat(rec['earth_q'], sg.earth_quaternion_scene(frame, t))
    assert "Earth time-of-day: real 2029 (hyperbola mapped to closest approach)" in combined
    assert "occlude" not in combined


@pytest.mark.parametrize("start_index", [30, 49])
def test_start_index_window_uses_whole_dir(start_index):
    # 49 leaves a 2-CSV window: a window-only fit would return None (no Sun_Visual).
    csvs = _csvs(MIA_BODIES)
    frame = sg.hyperbola_frame_from_bodies_dir(MIA_BODIES)
    out, _ = build_scene(f"window_{start_index}", MIA_BODIES, [1], len(csvs) - start_index,
                         start_index=start_index)
    assert out['sun_visual']
    t = float(_rows(csvs[start_index])['Apophis']['time_s'])
    sun = sg.sun_direction_scene(frame, t)
    np.testing.assert_allclose(out['frames']['1']['sun_visual'], sg.sun_visual_location(sun), rtol=1e-6)
    np.testing.assert_allclose(out['frames']['1']['lamp_neg_z'], -sun, atol=1e-5)


def test_ephemeris_scene_lamp_unchanged():
    csvs = _csvs(EPHEMERIS_BODIES)
    frames = [1, len(csvs)]
    out, combined = build_scene("ephemeris", EPHEMERIS_BODIES, frames, len(csvs))
    assert out['sun_visual']
    assert out['sun_body']['radius'] == pytest.approx(sg.sun_visual_radius_bu(), rel=1e-4)
    for f in frames:
        rows = _rows(csvs[f - out['frame_start']])
        rel = _km(rows['Sun']) - _km(rows['Apophis'])
        rec = out['frames'][str(f)]
        np.testing.assert_allclose(rec['lamp_neg_z'], -rel / np.linalg.norm(rel), atol=1e-5)
        np.testing.assert_allclose(rec['sun_visual'], sg.sun_visual_location(rel), rtol=1e-6)
        t = float(rows['Earth']['time_s'])
        _assert_quat(rec['earth_q'], eo.earth_quaternion(t, eo.DEFAULT_EPOCH_TDB))
    assert f"Earth time-of-day: epoch {eo.DEFAULT_EPOCH_TDB} TDB + time_s." in combined


def test_apophis_only_has_no_sun_visual():
    d = _fresh_dir("hsun_apophis_only_bodies")
    for i in range(3):
        _write_bodies(d / f"sobol_{i:05d}.csv",
                      [('Apophis', 100.0 * i, (10.0 * i, 0.0, 0.0), 2e-20)])
    out, _ = build_scene("apophis_only", d, [1, 3], 3)
    assert not out['sun_visual']
    for f in ('1', '3'):
        np.testing.assert_allclose(out['frames'][f]['lamp_neg_z'],
                                   -np.array(APOPHIS_ONLY_SUN_DIRECTION), atol=1e-6)


def test_two_frame_hyperbola_falls_back():
    d = _fresh_dir("hsun_two_frame_bodies")
    for src in _csvs(MIA_BODIES)[:2]:
        shutil.copy(src, d / src.name)
    out, combined = build_scene("two_frame", d, [1, 2], 2)
    assert "hyperbola Sun fit unavailable" in combined
    assert not out['sun_visual']
    for f in ('1', '2'):
        np.testing.assert_allclose(out['frames'][f]['lamp_neg_z'],
                                   -np.array(APOPHIS_ONLY_SUN_DIRECTION), atol=1e-6)


def test_far_earth_warns_about_sun_occlusion():
    # Earth 1e7 km from Apophis = 2e6 BU at EARTH_VIS_SCALE 0.2, beyond SUN_VISUAL_DISTANCE_BU / 10.
    d = _fresh_dir("hsun_far_earth_bodies")
    for i in range(3):
        _write_bodies(d / f"sobol_{i:05d}.csv", [
            ('Sun', 100.0 * i, (1.5e8, 0.0, 0.0), 1.0),
            ('Earth', 100.0 * i, (0.0, 0.0, 0.0), 3.0e-6),
            ('Apophis', 100.0 * i, (1.0e7, 0.0, 1000.0 * i), 2e-20),
        ])
    out, combined = build_scene("far_earth", d, [1], 3)
    assert out['sun_visual']
    warn = [ln for ln in combined.splitlines() if ln.startswith('WARNING') and 'occlude' in ln]
    assert len(warn) == 1, combined[-3000:]
    assert f"{math.hypot(1.0e7, 2000.0) * 0.2:.4g}" in warn[0]
