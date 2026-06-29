"""Tests for Bug A (utime early-return) and Bug B (signed omega)."""
import math
import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Run from sobol/ so the runner module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner


# ── Bug B: signed omega ─────────────────────────────────────────────────────

def test_period_hr_from_omega_code_positive():
    """Prograde spin (positive omega) must return a finite positive period."""
    utime = 4.0  # PHANTOM code-time units (arbitrary)
    period = runner._period_hr_from_omega_code(1.0, utime)
    assert math.isfinite(period), f"Expected finite period for omega=1.0, got {period}"
    assert period > 0.0


def test_period_hr_from_omega_code_zero():
    """Zero omega must return NaN (cannot infer period)."""
    period = runner._period_hr_from_omega_code(0.0, 4.0)
    assert math.isnan(period)


def test_spin_period_hr_retrograde():
    """After fix: retrograde spin (L antiparallel to n) must yield a finite positive period.

    Before fix: omega_code = w < 0 → _period_hr_from_omega_code returns NaN.
    After fix:  omega_code = abs(w) > 0 → finite positive period.
    """
    # Construct a toy rubble pile with ~5 grains in retrograde rotation.
    # Spin axis n = +z. Angular velocity ω = -1.0 code units (retrograde).
    n_grains = 10
    # Positions on unit circle in x–y plane
    theta = np.linspace(0, 2 * np.pi, n_grains, endpoint=False)
    r = 0.5
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n_grains)
    pos = np.column_stack([x, y, z])

    # Retrograde rotation: v = omega × r with omega = -1 z-hat
    omega_true = np.array([0.0, 0.0, -1.0])
    vel = np.cross(omega_true, pos)

    mass = np.ones(n_grains)
    com_pos = pos.mean(0)
    com_vel = vel.mean(0)
    dr = pos - com_pos
    dv = vel - com_vel

    # Build a fake EV array: [time, x, y, z, mass, vx, vy, vz, ...]
    # Only mass, dr, dv matter for _spin_period_hr_bound_rubble internals.
    # We'll call the private helper directly with synthesised inputs.
    n = np.array([0.0, 0.0, 1.0])  # spin axis = +z

    I_mat = (
        np.einsum("k,k->", mass, (dr**2).sum(1)) * np.eye(3)
        - np.einsum("k,ki,kj->ij", mass, dr, dr)
    )
    L = (mass[:, None] * np.cross(dr, dv)).sum(0)
    In = I_mat @ n
    n_I_n = float(n @ In)
    L_n = float(L @ n)
    # L_n / n_I_n should be negative (retrograde)
    w = L_n / n_I_n
    assert w < 0.0, f"Expected retrograde spin (w<0), got w={w}"

    # After fix: omega_code = abs(w)
    omega_code_fixed = abs(w)
    utime = 4.0  # arbitrary code-time unit
    period = runner._period_hr_from_omega_code(omega_code_fixed, utime)
    assert math.isfinite(period), f"Expected finite period after fix, got {period}"
    assert period > 0.0


# ── Bug A: utime early-return ────────────────────────────────────────────────

def test_dump_in_any_spin_window_returns_false_when_utime_none():
    """_dump_in_any_spin_window must return False (not crash) when utime=None.

    This is the key precondition for Bug A fix — if this function returns False,
    do_spin=False in the loop and breakup metrics are still computed.
    """
    result = runner._dump_in_any_spin_window(
        t=1.0, apophis_only=False, t_ca=10.0, utime=None
    )
    assert result is False


def test_dump_in_any_spin_window_returns_false_apophis_only_utime_none():
    """apophis_only path also returns False when utime=None."""
    result = runner._dump_in_any_spin_window(
        t=1.0, apophis_only=True, t_ca=None, utime=None
    )
    assert result is False


def test_spin_period_hr_bound_rubble_retrograde():
    """_spin_period_hr_bound_rubble must return a finite positive period for retrograde spin.

    Without the abs(w) fix in the function body, this test fails because
    _period_hr_from_omega_code returns NaN for negative omega_code.
    """
    # 10 grains on unit circle, retrograde rotation (omega = -1 z-hat)
    n_grains = 10
    theta = np.linspace(0, 2 * np.pi, n_grains, endpoint=False)
    r = 0.5
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n_grains)])
    omega_true = np.array([0.0, 0.0, -1.0])
    vel = np.cross(omega_true, pos)
    mass = np.ones(n_grains)
    # _spin_period_hr_bound_rubble expects arr with shape (N, 7): columns [x, y, z, mass, vx, vy, vz]
    ev = np.column_stack([pos, mass, vel])
    assert ev.shape == (n_grains, 7), f"Expected shape ({n_grains}, 7), got {ev.shape}"
    spin_axis = np.array([0.0, 0.0, 1.0])  # +z axis
    utime = 4.0  # code-time units per hour (arbitrary)
    period = runner._spin_period_hr_bound_rubble(ev, utime, spin_axis=spin_axis)
    assert math.isfinite(period), (
        f"_spin_period_hr_bound_rubble returned {period} for retrograde spin — "
        "check that omega_code = abs(w) in the function body"
    )
    assert period > 0.0


# ── Task 2: verify_metric_extraction crash on non-DEM runs ──────────────────

import verify_metric_extraction as vme


def test_metrics_for_run_single_sink_returns_no_dispersion_key(tmp_path):
    """_metrics_for_run with n_sinks < 2 must not include dispersion_ratio key."""
    # Create a minimal fake run dir with a single-sink .ev file.
    # Content format: header + one row for sink ID 11 only.
    ev_content = (
        "# time x y z mass vx vy vz spinx spiny spinz\n"
        "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    run_dir = tmp_path / "run_0001"
    run_dir.mkdir()
    ev_file = run_dir / "sobol_00011.ev"
    ev_file.write_text(ev_content)

    result = vme._metrics_for_run(
        run_dir, "sobol", earth_sink_id=4, apophis_sink_id=11, apophis_only=True,
        legacy_substeps=False
    )
    assert "dispersion_ratio" not in result, (
        f"Expected no 'dispersion_ratio' key for single-sink run, got keys: {list(result)}"
    )


def test_main_does_not_crash_on_single_sink_run(tmp_path):
    """verify_metric_extraction main() must exit 0 (not KeyError) for non-DEM runs.

    Patches _metrics_for_run to return a non-DEM result (no dispersion_ratio key)
    so the test is independent of .ev file parsing format.
    """
    run_dir = tmp_path / "run_0001"
    run_dir.mkdir()

    with patch.object(vme, "_metrics_for_run", return_value={"n_dump_groups": 0, "n_sinks": 1}):
        orig_argv = sys.argv[:]
        sys.argv = ["verify_metric_extraction.py", "--run-dir", str(run_dir), "--apophis-only"]
        try:
            exit_code = vme.main()
        except SystemExit as e:
            exit_code = e.code
        finally:
            sys.argv = orig_argv
    assert exit_code == 0, f"Expected exit 0 for non-DEM run, got {exit_code}"
