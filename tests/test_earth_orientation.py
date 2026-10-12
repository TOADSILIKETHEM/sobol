"""Earth spin + time of day (Code/BlenderConvert/earth_orientation.py, pure Python)."""

import glob
import math
import os
import sys

import pytest

from tests._paths import CODE, DATA

sys.path.insert(0, str(CODE / 'BlenderConvert'))
import earth_orientation as eo  # noqa: E402

REAL_BODIES_DIR = str(DATA / 'DEMCSVs' / '1000np p1p6hr Aligned' / 'run_0001_bodies_output')


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def test_gmst_at_j2000():
    # Meeus, Astronomical Algorithms, eq. 12.4: GMST at JD 2451545.0 UT1.
    assert eo.gmst_deg(2451545.0) == pytest.approx(280.46061837, abs=1e-6)


def test_gmst_meeus_example_12a():
    # Meeus example 12.a: 1987 April 10, 0h UT -> 13h10m46.3668s.
    expected = (13 + 10 / 60 + 46.3668 / 3600) * 15.0
    assert eo.gmst_deg(2446895.5) == pytest.approx(expected, abs=1e-4)


def test_julian_date_of_epoch():
    # sobol/earth.txt: 2029-Apr-10 00:00 TDB = JD 2462236.5
    assert eo.julian_date('2029-04-10') == pytest.approx(2462236.5)
    assert eo.julian_date('2029-Apr-10 06:00:00') == pytest.approx(2462236.75)


def test_pole_points_to_ecliptic_lambda90_beta66():
    # Earth's north pole in J2000 ecliptic coords: lambda 90 deg, beta ~66.56 deg
    # -> (0, +sin eps, cos eps). Precession moves it by < 0.5 deg by 2029.
    pole = eo.rotate(eo.earth_quaternion(0.0, '2029-04-10', 0.0), (0.0, 0.0, 1.0))
    assert pole[0] == pytest.approx(0.0, abs=0.01)
    assert pole[1] == pytest.approx(0.3978, abs=0.01)
    assert pole[2] == pytest.approx(0.9175, abs=0.01)


def test_pole_in_sim_frame_rotated_90():
    # Sim frame = J2000 ecliptic rotated +90 deg about z (set_binary posang_ascnode).
    pole = eo.rotate(eo.earth_quaternion(0.0, '2029-04-10'), (0.0, 0.0, 1.0))
    assert pole[0] == pytest.approx(-0.3978, abs=0.01)
    assert pole[1] == pytest.approx(0.0, abs=0.01)
    assert pole[2] == pytest.approx(0.9175, abs=0.01)


def test_pole_fixed_over_a_day():
    q0 = eo.earth_quaternion(0.0, '2029-04-10')
    for t in (21600.0, 43200.0, 86400.0, 300000.0):
        p = eo.rotate(eo.earth_quaternion(t, '2029-04-10'), (0.0, 0.0, 1.0))
        p0 = eo.rotate(q0, (0.0, 0.0, 1.0))
        assert math.dist(p, p0) < 1e-4


def test_sidereal_day_returns_to_same_orientation():
    q0 = eo.earth_quaternion(0.0, '2029-04-10')
    q1 = eo.earth_quaternion(eo.EARTH_SIDEREAL_DAY_S, '2029-04-10')
    x0 = eo.rotate(q0, (1.0, 0.0, 0.0))
    x1 = eo.rotate(q1, (1.0, 0.0, 0.0))
    assert math.dist(x0, x1) < 1e-4


def test_quaternion_keys_continuous():
    # Blender interpolates quaternion F-curves per component: consecutive keys
    # must stay in the same hemisphere (no q -> -q flip at a GMST wrap).
    prev = eo.earth_quaternion(0.0)
    for k in range(1, 5 * 288):
        q = eo.earth_quaternion(k * 300.0)
        assert sum(a * b for a, b in zip(q, prev)) > 0.99
        prev = q


def test_geographic_roundtrip():
    q = eo.earth_quaternion(12345.0, '2029-04-10')
    for lon, lat in ((0.0, 0.0), (-40.0, 30.0), (120.0, -60.0)):
        v_body = (math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
                  math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
                  math.sin(math.radians(lat)))
        v_ecl = eo.rotate(q, v_body)
        lon2, lat2 = eo.ecliptic_to_geographic(v_ecl, 12345.0, '2029-04-10')
        assert _wrap180(lon2 - lon) == pytest.approx(0.0, abs=1e-6)
        assert lat2 == pytest.approx(lat, abs=1e-6)


def _low_precision_sun_ecliptic(jd):
    """Meeus ch. 25 low-precision apparent solar longitude (deg), latitude 0."""
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T
    M = math.radians(357.52911 + 35999.05029 * T)
    C = (1.914602 - 0.004817 * T) * math.sin(M) + 0.019993 * math.sin(2 * M) + 0.000289 * math.sin(3 * M)
    lam_date = L0 + C
    # back to J2000 ecliptic (undo precession in longitude)
    lam = math.radians(lam_date - 1.396971 * T)
    return (math.cos(lam), math.sin(lam), 0.0)


def test_subsolar_point_noon_utc_on_greenwich():
    # 2029-04-13 12:00 UT1: subsolar lon ~ -(equation of time) ~ 0 deg (EoT ~ +0.4 min,
    # i.e. ~0.1 deg), declination ~ +9 deg in mid-April.
    epoch = '2029-04-10'
    t = 3 * 86400.0 + 12 * 3600.0 + eo.DELTA_T_S  # TDB seconds since epoch
    jd = eo.julian_date(epoch) + t / 86400.0
    lon, lat = eo.ecliptic_to_geographic(_low_precision_sun_ecliptic(jd), t, epoch, 0.0)
    assert _wrap180(lon) == pytest.approx(0.0, abs=0.5)
    assert lat == pytest.approx(9.0, abs=0.7)


def _bodies_rows(path):
    import pandas as pd
    df = pd.read_csv(path).set_index('name')
    return df


@pytest.mark.skipif(not os.path.isdir(REAL_BODIES_DIR), reason='OneDrive DEMCSVs not mounted')
def test_real_run_subsolar_and_apophis_closest_approach():
    import numpy as np
    files = sorted(glob.glob(os.path.join(REAL_BODIES_DIR, '*.csv')))
    epoch = '2029-04-10'

    # Subsolar point from the sim's own Sun/Earth positions, every ~6 h.
    for f in files[::72]:
        df = _bodies_rows(f)
        t = float(df.loc['Earth', 'time_s'])
        s = df.loc['Sun', ['x_km', 'y_km', 'z_km']].to_numpy(float) - df.loc['Earth', ['x_km', 'y_km', 'z_km']].to_numpy(float)
        lon, lat = eo.ecliptic_to_geographic(tuple(s), t, epoch)
        ut_hours = ((t - eo.DELTA_T_S) / 3600.0) % 24.0
        assert _wrap180(lon - (12.0 - ut_hours) * 15.0) == pytest.approx(0.0, abs=0.5), f
        assert 7.5 < lat < 10.5, f

    # Apophis closest approach: 2029-04-13 ~21:46 UT, over the Atlantic.
    def closest(indices):
        best = None
        for i in indices:
            df = _bodies_rows(files[i])
            d = df.loc['Apophis', ['x_km', 'y_km', 'z_km']].to_numpy(float) - df.loc['Earth', ['x_km', 'y_km', 'z_km']].to_numpy(float)
            r = float(np.linalg.norm(d))
            if best is None or r < best[0]:
                best = (r, float(df.loc['Earth', 'time_s']), tuple(d), i)
        return best

    step = 24  # coarse pass, then every file around the coarse minimum (OneDrive reads are slow)
    i0 = closest(range(0, len(files), step))[3]
    r, t, d, _ = closest(range(max(0, i0 - step), min(len(files), i0 + step + 1)))
    lon, lat = eo.ecliptic_to_geographic(d, t, epoch)
    print(f'closest approach r={r:.0f} km, t={t:.0f} s, sub-Apophis lon={lon:.1f} lat={lat:.1f}')
    ut_hours = (t - eo.DELTA_T_S) / 3600.0 - 72.0
    assert 21.0 < ut_hours < 22.5
    assert -60.0 < _wrap180(lon) < 0.0


def _nlerp(q0, q1, t):
    m = [(1 - t) * a + t * b for a, b in zip(q0, q1)]
    n = math.sqrt(sum(c * c for c in m))
    return [c / n for c in m]


def _lerped_turn_deg(step_deg):
    """Spin the per-component-lerped keys actually show (Blender quaternion keys)."""
    dt = step_deg / 360.0 * eo.EARTH_SIDEREAL_DAY_S
    q0, q1 = eo.earth_quaternion(0.0), eo.earth_quaternion(dt)
    lon = lambda q: math.degrees(math.atan2(*reversed(eo.rotate(q, (1, 0, 0))[:2])))
    prev, total = lon(q0), 0.0
    for k in range(1, 41):
        cur = lon(_nlerp(q0, q1, k / 40))
        total += (cur - prev + 180) % 360 - 180
        prev = cur
    return total


def test_lerped_spin_keys_reverse_only_past_360_deg():
    # Unwrapped GMST keeps the quaternions continuous: keys up to 360 deg apart
    # still turn forward; past 360 they turn backwards.
    assert _lerped_turn_deg(270) > 0
    assert _lerped_turn_deg(350) > 0
    assert _lerped_turn_deg(370) < 0


def test_spin_step_check_quiet_for_small_steps():
    chk = eo.SpinStepCheck()
    day = eo.EARTH_SIDEREAL_DAY_S
    assert [chk.warning(t, f) for f, t in enumerate((0.0, 700.0, 0.4 * day, 0.8 * day))] == [None] * 4


def test_spin_step_check_flags_uneven_then_backwards_once_each_run():
    day = eo.EARTH_SIDEREAL_DAY_S
    chk = eo.SpinStepCheck()
    assert chk.warning(0.0, 1) is None
    msg = chk.warning(0.75 * day, 2)  # 270 deg: forward but uneven
    assert "uneven" in msg and "frame 2" in msg
    assert chk.warning(1.5 * day, 3) is None  # one warning per run, not per frame
    chk = eo.SpinStepCheck()
    chk.warning(0.0, 1)
    assert "backwards" in chk.warning(1.2 * day, 2)  # 432 deg
