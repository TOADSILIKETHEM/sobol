"""Real 2029 Sun geometry (Code/BlenderConvert/sun_geometry.py, pure Python)."""

import math
import sys

import numpy as np
import pytest

from tests._paths import CODE, DATA

sys.path.insert(0, str(CODE / 'BlenderConvert'))
import earth_orientation as eo  # noqa: E402
import sun_geometry as sg  # noqa: E402

MIA_BODIES = DATA / 'DEMCSVs/mia_full_B_s1_cb_x0.7/run_0001_bodies_output'


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


def _rotation_about(axis, angle):
    """Rodrigues rotation matrix about ``axis`` (any length) by ``angle`` rad."""
    k = np.asarray(axis, dtype=float)
    k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


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


def test_matrix_to_quaternion_round_trip():
    for rot in (_rotation_about((1, 2, 3), 0.7), _rotation_about((0, 0, 1), math.pi),
                _rotation_about((1, 0, 0), 3.0), np.eye(3)):
        q = sg.matrix_to_quaternion(rot)
        assert q[0] >= 0
        v = (0.3, -0.5, 0.8)
        np.testing.assert_allclose(eo.rotate(q, v), rot @ v, atol=1e-12)


@pytest.mark.skipif(not MIA_BODIES.is_dir(), reason="needs Max's DEMCSVs")
def test_fit_on_mia_run():
    fr = sg.hyperbola_frame_from_bodies_dir(str(MIA_BODIES))
    assert fr.rp_km == pytest.approx(7637.0, abs=0.5)
    assert fr.t_ca_sim_s == pytest.approx(29822.2, abs=1.0)


def _write_bodies(directory, names, rel, times):
    """One CSV per time; ``rel`` (km) is Apophis relative to Earth, Earth at the origin."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, (t, r) in enumerate(zip(times, rel)):
        with open(directory / f'sobol_{i:05d}.csv', 'w') as f:
            f.write('name,time_s,x_au,y_au,z_au,x_km,y_km,z_km,h,Reff,m\n')
            for name in names:
                pos = r if name == 'Apophis' else (0.0, 0.0, 0.0)
                m = 3.0e-6 if name == 'Earth' else 1.0e-20
                f.write(f'{name},{t},0,0,0,{float(pos[0])!r},{float(pos[1])!r},{float(pos[2])!r},1,1,{m}\n')


def test_bodies_dir_rules(tmp_path):
    mu = 3.0e-6 * sg.GM_SUN_KM3_S2
    times = np.linspace(-30000, 20000, 5) + 1000.0
    rel = _synthetic(8000.0, 5.9, mu, _rotation_about((1, 2, 3), 0.7), times - 1000.0)
    _write_bodies(tmp_path / 'sun', ['Sun', 'Earth', 'Apophis'], rel, times)
    assert sg.hyperbola_frame_from_bodies_dir(str(tmp_path / 'sun')) is None
    _write_bodies(tmp_path / 'noearth', ['Apophis'], rel, times)
    assert sg.hyperbola_frame_from_bodies_dir(str(tmp_path / 'noearth')) is None
    _write_bodies(tmp_path / 'two', ['Earth', 'Apophis'], rel[:2], times[:2])
    assert sg.hyperbola_frame_from_bodies_dir(str(tmp_path / 'two')) is None
    _write_bodies(tmp_path / 'ok', ['Earth', 'Apophis'], rel, times)
    fr = sg.hyperbola_frame_from_bodies_dir(str(tmp_path / 'ok'))
    assert isinstance(fr, sg.HyperbolaFrame)
    assert fr.rp_km == pytest.approx(8000.0, rel=1e-6)
    assert fr.t_ca_sim_s == pytest.approx(1000.0, abs=1.0)


def test_fit_degenerate_orbits_return_none():
    t = np.linspace(0.0, 100.0, 6)
    stationary = np.tile([7000.0, 0.0, 0.0], (6, 1))
    assert sg.fit_hyperbola_frame(t, stationary, 398600.4) is None
    collinear = np.outer(np.linspace(1e4, 5e4, 6), [1.0, 0.0, 0.0])
    assert sg.fit_hyperbola_frame(t, collinear, 398600.4) is None


def test_fit_zero_mu_returns_none():
    th = np.linspace(-1.0, 1.0, 9)
    p, e = 20000.0, 1.8
    r = p / (1.0 + e * np.cos(th))
    rel = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(9)])
    assert sg.fit_hyperbola_frame(np.linspace(-3e4, 3e4, 9), rel, 0.0) is None


def test_fit_near_parabolic_returns_none():
    th = np.linspace(-1.0, 1.0, 9)
    p, e = 20000.0, 1.0003
    r = p / (1.0 + e * np.cos(th))
    rel = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(9)])
    fr = sg.fit_hyperbola_frame(np.linspace(-3e4, 3e4, 9), rel, 398600.4)
    assert fr is None or (math.isfinite(fr.t_ca_sim_s) and np.isfinite(fr.rotation).all())


def test_sun_visual_location_zero_direction_raises():
    with pytest.raises(ValueError):
        sg.sun_visual_location((0.0, 0.0, 0.0))
