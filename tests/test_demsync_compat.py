import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner
from run_mass_sobol_phantom import RunSample, apply_run_sample_to_in, apply_run_sample_to_setup


def test_spin_period_hr_to_setup_seconds():
    assert runner.spin_period_hr_to_setup_seconds(2.0) == 7200.0
    assert runner.spin_period_hr_to_setup_seconds(1.55) == 5580.0


def test_ecliptic_spin_axis_unit_reference():
    nx, ny, nz = runner.ecliptic_spin_axis_unit(17.4, 332.2)
    assert abs(nx - 0.276) < 0.02
    assert abs(ny - (-0.116)) < 0.03
    assert abs(nz - 0.954) < 0.02
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    assert abs(n - 1.0) < 1e-6


def test_coh_gap_max_cgs_from_dn_np500():
    gap = runner.coh_gap_max_cgs_from_dn(dn=0.1, np_apophis=500)
    assert 400.0 < gap < 470.0


def test_kt_from_kc_identity():
    assert runner.kt_from_kc(1e7) == 1e7


def test_read_kt_cgs_prefers_kt():
    assert runner.read_kt_cgs({"kt_cgs": "1e7", "kc_cgs": "0"}) == "1e7"
    assert runner.read_kt_cgs({"kc_cgs": "1e7"}) == "1e7"


def test_apply_run_sample_writes_spin_period_seconds(tmp_path):
    setup = " apophis_spin_period = 30.0\n"
    p = tmp_path / "sobol.setup"
    p.write_text(setup)
    sample = RunSample(apophis_spin_period=2.0)
    cols = apply_run_sample_to_setup(p, sample)
    text = p.read_text()
    assert "7200" in text.replace(" ", "")
    assert cols["apophis_spin_period"] == "2"


def test_apply_run_sample_writes_axis_from_obliquity_azimuth(tmp_path):
    p = tmp_path / "sobol.setup"
    p.write_text(
        "apophis_spin_axis_x = 0.0\n"
        "apophis_spin_axis_y = 0.0\n"
        "apophis_spin_axis_z = 1.0\n"
        "apophis_spin_torque_align_deg = -1.0\n"
    )
    sample = RunSample(
        apophis_spin_obliquity=17.4,
        apophis_spin_azimuth=332.2,
        apophis_spin_torque_align_deg=None,
    )
    apply_run_sample_to_setup(p, sample)
    text = p.read_text()
    assert "apophis_spin_obliquity" not in text
    assert "apophis_spin_axis_x" in text
    assert "apophis_spin_torque_align_deg" in text
    assert "-1" in text.split("apophis_spin_torque_align_deg")[1][:20]


def test_apply_run_sample_to_in_writes_kt_and_gap(tmp_path):
    in_path = tmp_path / "sobol.in"
    in_path.write_text(
        "isink_potential = 2\n"
        "kn_cgs = 1e7\n"
        "kt_cgs = 0\n"
        "coh_gap_max_cgs = 0\n"
    )
    sample = RunSample(kt_cgs=1e7, coh_gap_max_cgs=420.0, np_apophis=500)
    cols = apply_run_sample_to_in(in_path, sample)
    text = in_path.read_text()
    assert "kc_cgs" not in text
    assert "kt_cgs" in text
    assert cols["kt_cgs"] == "10000000"
    assert cols["coh_gap_max_cgs"] == "420"


def test_phantomsetup_needs_rerun_detects_message(tmp_path):
    log = tmp_path / "setup.log"
    log.write_text("STOP rerun phantomsetup after editing .setup file\n")
    assert runner._phantomsetup_needs_rerun(log) is True
    log.write_text("setup complete\n")
    assert runner._phantomsetup_needs_rerun(log) is False
