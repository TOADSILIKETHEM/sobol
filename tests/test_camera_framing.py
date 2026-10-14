"""Earth-fixed camera framing (Code/viz/camera_framing.py)."""
import math
import sys

import numpy as np
import pytest

from tests._paths import CODE

sys.path.insert(0, str(CODE))
from viz import camera_framing as cf  # noqa: E402

R = 1274.0  # Earth radius in BU (6371 km x EARTH_VIS_SCALE 0.2)
FOV = math.radians(70.0)


def _hyperbola_earth_pos(rp=1600.0, n=60):
    """Earth positions in the Apophis-centred scene for a hyperbolic pass (Apophis = -earth_pos)."""
    t = np.linspace(-3.0, 3.0, n)
    apo = np.column_stack([rp * np.cosh(t) - 1.5 * rp, 2.0 * rp * np.sinh(t), 0.3 * rp * np.sinh(t)])
    return -apo


def _project(pts, offset, right, up):
    """Image-plane tangents (x / depth, y / depth) of world points seen from Earth + offset."""
    back = np.cross(right, up)
    rel = np.asarray(pts) - offset
    depth = -(rel @ back)
    return (rel @ right) / depth, (rel @ up) / depth, depth


def test_whole_path_and_earth_fit_the_frame():
    earth_pos = _hyperbola_earth_pos()
    offset, right, up = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9, window_ca_factor=None)
    path = -earth_pos
    limb = np.array([R * right, -R * right, R * up, -R * up])
    x, y, depth = _project(np.vstack([path, limb]), offset, right, up)
    assert (depth > 0).all()
    assert np.abs(x).max() <= math.tan(FOV / 2) + 1e-9
    assert np.abs(y).max() <= math.tan(FOV / 2) * 9 / 16 + 1e-9


def test_axes_are_orthonormal_and_right_handed():
    offset, right, up = cf.earth_fixed_pose(_hyperbola_earth_pos(), R, FOV, 16 / 9)
    assert np.dot(right, up) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    back = np.cross(right, up)
    # camera sits on the +back side and looks along -back, so the scene is in front of it
    assert np.dot(offset, back) > 0


def test_tilt_puts_the_camera_on_the_closest_approach_side():
    earth_pos = _hyperbola_earth_pos()
    path = -earth_pos
    ca = path[np.argmin(np.linalg.norm(path, axis=1))]
    ca_dir = ca / np.linalg.norm(ca)
    _, right, up = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9, tilt_deg=30.0)
    assert np.dot(np.cross(right, up), ca_dir) == pytest.approx(math.sin(math.radians(30.0)))
    _, right, up = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9, tilt_deg=0.0)
    assert np.dot(np.cross(right, up), ca_dir) == pytest.approx(0.0, abs=1e-9)


def test_window_frames_a_tighter_shot():
    earth_pos = _hyperbola_earth_pos()
    wide, _, _ = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9, window_ca_factor=None)
    tight, _, _ = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9, window_bu=4000.0)
    assert np.linalg.norm(tight) < 0.5 * np.linalg.norm(wide)


def test_default_window_is_three_times_closest_approach():
    earth_pos = _hyperbola_earth_pos()
    dist = np.linalg.norm(earth_pos, axis=1)
    path = cf.apophis_path(earth_pos, window_ca_factor=3.0)
    assert len(path) == (dist <= 3.0 * dist.min()).sum() < len(earth_pos)
    default = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9)
    explicit = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9, window_bu=3.0 * dist.min())
    assert np.allclose(default[0], explicit[0])


def test_window_always_keeps_closest_approach():
    path = cf.apophis_path(_hyperbola_earth_pos(), window_bu=1.0)
    assert len(path) == 1


def test_degenerate_paths_still_give_a_pose():
    for earth_pos in ([[5000.0, 0.0, 0.0]], [[5000.0, 0.0, 0.0], [4000.0, 0.0, 0.0]],
                      [[0.0, 0.0, 5000.0], [0.0, 0.0, 4000.0]]):
        offset, right, up = cf.earth_fixed_pose(earth_pos, R, FOV, 16 / 9)
        assert np.isfinite(offset).all() and np.isfinite(right).all() and np.isfinite(up).all()
        assert np.linalg.norm(offset) >= 2 * R


def test_no_earth_positions_is_an_error():
    with pytest.raises(ValueError, match="Earth"):
        cf.earth_fixed_pose(np.zeros((0, 3)), R, FOV, 16 / 9)


def _chase(fov_deg, earth_pos=None, cores=None):
    earth_pos = _hyperbola_earth_pos() if earth_pos is None else earth_pos
    cores = np.zeros_like(earth_pos) if cores is None else cores
    return cf.earth_chase_offsets(cores, earth_pos, math.radians(fov_deg), 16 / 9,
                                  ref_fov_rad=FOV, ref_dist_bu=150.0)


def test_chase_puts_apophis_on_the_lower_third_line_aimed_at_earth():
    earth_pos = _hyperbola_earth_pos()
    for fov in (70.0, 20.0):
        offsets, up = _chase(fov, earth_pos)
        tan_y = math.tan(math.radians(fov) / 2) * 9 / 16
        for off, e in zip(offsets, earth_pos):
            axis = (e - off) / np.linalg.norm(e - off)  # camera at core + off looks at Earth
            to_core = -off / np.linalg.norm(off)
            below = math.atan2(-np.dot(to_core, up - axis * np.dot(axis, up)) /
                               np.linalg.norm(up - axis * np.dot(axis, up)), np.dot(to_core, axis))
            assert math.tan(below) / tan_y == pytest.approx(1 / 3, abs=1e-6)


def test_chase_distance_keeps_apophis_size_across_lenses():
    wide, _ = _chase(70.0)
    tele, _ = _chase(20.0)
    ratio = np.linalg.norm(wide, axis=1) / np.linalg.norm(tele, axis=1)
    assert ratio == pytest.approx(math.tan(math.radians(10)) / math.tan(math.radians(35)))


def test_chase_up_is_the_orbit_normal_and_follows_the_core():
    earth_pos = _hyperbola_earth_pos()
    cores = np.tile([3.0, -2.0, 1.0], (len(earth_pos), 1))
    offsets, up = _chase(70.0, earth_pos)
    moved, up2 = _chase(70.0, earth_pos + cores, cores)
    assert np.allclose(up, up2) and np.allclose(offsets, moved)  # depends only on Earth - core
    rel = earth_pos
    assert np.allclose(rel @ up, 0.0, atol=1e-6 * np.linalg.norm(rel, axis=1).max())
