"""Blender F-curve handle port (Code/viz/fcurve_handles.py)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from tests._paths import CODE

sys.path.insert(0, str(CODE))
from viz import fcurve_handles as fh  # noqa: E402

from tests.test_dem_headless_paths import BLENDER_EXE, WIN_TEMP, WIN_TEMP_WIN  # noqa: E402


def _track(y):
    y = np.asarray(y, dtype=np.float64)
    return y, fh.right_handle_offsets(y)


def test_passes_through_keys_exactly():
    y, h = _track([0.0, 2.0, -1.0, 5.0, 5.5])
    for k in range(len(y)):
        assert fh.evaluate_track(y, h, k) == pytest.approx(y[k])


def test_constant_extrapolation():
    y, h = _track([1.0, 3.0, 2.0])
    assert fh.evaluate_track(y, h, -4.0) == pytest.approx(1.0)
    assert fh.evaluate_track(y, h, 9.0) == pytest.approx(2.0)


def test_end_and_extremum_handles_are_flat():
    y, h = _track([0.0, 1.0, 3.0, 1.0, 2.0])
    assert h[0] == 0.0 and h[-1] == 0.0
    assert h[2] == 0.0  # local maximum


def test_no_overshoot_between_keys():
    y, h = _track([0.0, 0.1, 10.0, 10.1, 0.0, 5.0])
    for k in range(len(y) - 1):
        lo, hi = sorted((y[k], y[k + 1]))
        for t in np.linspace(0.0, 1.0, 101):
            v = fh.evaluate(y[k], h[k], y[k + 1], h[k + 1], t)
            assert lo - 1e-12 <= v <= hi + 1e-12


def test_columns_are_independent():
    tracks = np.random.default_rng(1).normal(size=(9, 4, 3)).cumsum(axis=0)
    h = fh.right_handle_offsets(tracks)
    assert h.shape == tracks.shape
    for i in range(4):
        for c in range(3):
            assert np.allclose(h[:, i, c], fh.right_handle_offsets(tracks[:, i, c]))


_BLENDER_FCURVE_SCRIPT = '''
import json, sys, traceback
import bpy
import numpy as np
try:
    rng = np.random.default_rng({seed})
    # Random walks with mixed step sizes: extrema, clamped and free handles.
    steps = rng.normal(size=({n_keys}, 3)) * rng.choice([0.05, 1.0, 20.0], size=({n_keys}, 3))
    keys = np.cumsum(steps, axis=0).astype(np.float32)
    bpy.ops.object.empty_add(location=(0, 0, 0))
    obj = bpy.context.object
    scene = bpy.context.scene
    # Same calls DEMGrainsBlenderEarthCam.py makes per grain.
    for i, row in enumerate(keys, start=1):
        scene.frame_set(i)
        obj.location = tuple(float(v) for v in row)
        obj.keyframe_insert(data_path="location", frame=i)
    action = obj.animation_data.action
    fcurves = list(getattr(action, "fcurves", None) or [])
    if not fcurves:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    fcurves.extend(bag.fcurves)
    fcurves.sort(key=lambda fc: fc.array_index)
    for fc in fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"
    times = np.linspace(-1.0, {n_keys} + 2.0, {n_samples})
    vals = [[fc.evaluate(float(t)) for fc in fcurves] for t in times]
    out = {{"keys": keys.tolist(), "times": times.tolist(), "vals": vals,
           "smoothing": [fc.auto_smoothing for fc in fcurves]}}
    open({out!r}, "w").write(json.dumps(out))
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
@pytest.mark.parametrize("seed", [7, 8, 9])
def test_matches_blender_fcurves(seed):
    name = f"fcurve_handles_{seed}"
    out = WIN_TEMP / f"{name}.json"
    if out.exists():
        out.unlink()
    (WIN_TEMP / f"{name}.py").write_text(
        _BLENDER_FCURVE_SCRIPT.format(seed=seed, n_keys=60, n_samples=3001,
                                      out=f"{WIN_TEMP_WIN}/{name}.json"),
        encoding="utf-8",
    )
    # User preferences (no --factory-startup): per-sphere renders run with them.
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", f"{WIN_TEMP_WIN}/{name}.py"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, ((result.stdout or "") + (result.stderr or ""))[-3000:]
    data = json.loads(out.read_text())
    assert set(data["smoothing"]) == {fh.SMOOTHING}
    keys = np.array(data["keys"], dtype=np.float64)
    h = fh.right_handle_offsets(keys)
    expected = np.array(data["vals"])
    got = np.array([fh.evaluate_track(keys, h, t - 1.0) for t in data["times"]])  # frame 1 == key 0
    scale = np.abs(keys).max()
    err = np.abs(got - expected).max()
    assert err <= 1e-5 * scale, f"max |numpy - Blender| = {err:.3g} (key scale {scale:.3g})"
