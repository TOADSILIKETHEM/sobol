import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner
from run_mass_sobol_phantom import RunSample, apply_run_sample_to_setup

SETUP = (
    " np_apophis = 1 ! n\n"
    " use_dem = F ! dem\n"
    " use_dem_as_sinks = F ! legacy\n"
    " scale_rho = 1.000 ! rho\n"
    " mass_apophis = 0.000 ! g\n"
)


def _setup(tmp_path, text=SETUP):
    p = tmp_path / "sobol.setup"
    p.write_text(text)
    return p


def _val(text, key):
    return re.search(rf"^\s*{key}\s*=\s*([^!\n]*)", text, re.M).group(1).strip()


def test_particle_model_sets_use_dem_only(tmp_path):
    p = _setup(tmp_path)
    cols = apply_run_sample_to_setup(p, RunSample(use_dem=True, dem_model="particle"), None, None)
    t = p.read_text()
    assert _val(t, "use_dem") == "T" and _val(t, "use_dem_as_sinks") == "F"
    assert cols["use_dem"] == "T" and cols["dem_model"] == "particle"


def test_sink_model_sets_use_dem_as_sinks_only(tmp_path):
    p = _setup(tmp_path)
    cols = apply_run_sample_to_setup(p, RunSample(use_dem=True, dem_model="sink"), None, None)
    t = p.read_text()
    assert _val(t, "use_dem") == "F" and _val(t, "use_dem_as_sinks") == "T"
    assert cols["dem_model"] == "sink"


def test_use_dem_false_clears_both(tmp_path):
    p = _setup(tmp_path, SETUP.replace("use_dem = F", "use_dem = T"))
    apply_run_sample_to_setup(p, RunSample(use_dem=False, dem_model="sink"), None, None)
    t = p.read_text()
    assert _val(t, "use_dem") == "F" and _val(t, "use_dem_as_sinks") == "F"


def test_setup_missing_use_dem_as_sinks_raises(tmp_path):
    p = _setup(tmp_path, " np_apophis = 1\n use_dem = F\n")
    with pytest.raises(RuntimeError, match="use_dem_as_sinks"):
        apply_run_sample_to_setup(p, RunSample(use_dem=True), None, None)


def test_mass_written_as_mass_apophis_grams(tmp_path):
    p = _setup(tmp_path)
    apply_run_sample_to_setup(p, RunSample(mass_kg=4.0e10), None, None)
    t = p.read_text()
    assert float(_val(t, "mass_apophis")) == pytest.approx(4.0e13)
    assert float(_val(t, "scale_rho")) == pytest.approx(1.0)


def test_shape_config_is_mia_scale():
    text = runner.DEFAULT_APOPHIS_SHAPE_CONFIG.read_text()
    assert "apophis_v233s7.obj 0.170" in text
    assert runner.LITERATURE_MESH_SCALE_KM == pytest.approx(0.170)


def test_literature_scale_override_removed():
    assert not hasattr(runner, "_maybe_apply_literature_scale_r")
    assert not hasattr(runner, "LITERATURE_SCALE_R_APOPHIS")


def test_parser_dem_model_default_particle():
    args = runner.build_parser().parse_args([])
    assert args.dem_model == "particle"


def test_template_setup_has_merged_keys():
    text = (Path(__file__).parent.parent / "sobol.setup").read_text()
    for key in ("use_dem_as_sinks", "mass_apophis", "pack_settle", "pack_expand", "pack_phi"):
        assert re.search(rf"^\s*{key}\s*=", text, re.M), key
    assert not re.search(r"^\s*m_apophis_in\s*=", text, re.M)


import shutil  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def test_in_patch_applies_without_isink_potential_2(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("isink_potential = 0\nkn_cgs = 1e7\nkt_cgs = 0\ncoh_gap_max_cgs = 0\n")
    cols = runner.apply_run_sample_to_in(p, RunSample(kt_cgs=2e7))
    assert "kt_cgs" in cols
    assert float(_val(p.read_text(), "kt_cgs")) == pytest.approx(2e7)


def test_in_patch_noop_when_dem_keys_absent(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("isink_potential = 0\n")
    assert runner.apply_run_sample_to_in(p, RunSample(kt_cgs=2e7)) == {}


def test_setup_log_torque_fallback_raises(tmp_path):
    log = tmp_path / "setup.log"
    log.write_text(" WARNING! setup_solarsystem: torque-align spin needs a sink index range;"
                   " using apophis_spin_axis instead\n")
    with pytest.raises(RuntimeError, match="torque-align"):
        runner.check_setup_log(log, RunSample(use_dem=True, apophis_spin_torque_align_deg=10.0))


def test_setup_log_spin_not_applied_raises(tmp_path):
    log = tmp_path / "setup.log"
    log.write_bytes(b"\x00 WARNING! setup_solarsystem: could not resolve Apophis spin axis; spin not applied\n")
    with pytest.raises(RuntimeError, match="spin not applied"):
        runner.check_setup_log(log, RunSample(use_dem=True, apophis_spin_period=2.0))


def test_setup_log_clean_passes(tmp_path):
    log = tmp_path / "setup.log"
    log.write_text(" Apophis spin axis  =  0.1 0.2 0.97\n")
    runner.check_setup_log(log, RunSample(use_dem=True, apophis_spin_period=2.0))


def test_coh_gap_from_dump_particle(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIX / "particle_np300", run)
    s = RunSample(use_dem=True, dem_model="particle", kt_cgs=1e7, dn_cohes_factor=0.1)
    runner.resolve_coh_gap_after_setup(run, "sobol", s)
    import particle_dem as pdem
    r = pdem.grain_radius_cm_from_dump(run / "sobol_00000", "particle")
    assert s.coh_gap_max_cgs == pytest.approx(0.1 * 2.0 * r)


def test_coh_gap_explicit_value_wins(tmp_path):
    s = RunSample(use_dem=True, kt_cgs=1e7, dn_cohes_factor=0.1, coh_gap_max_cgs=123.0)
    runner.resolve_coh_gap_after_setup(tmp_path, "sobol", s)
    assert s.coh_gap_max_cgs == 123.0


import math  # noqa: E402

SINK_FIX = FIX / "sink_np300"
needs_sink_fixture = pytest.mark.skipif(
    not (SINK_FIX / "sobol_00000").is_file(),
    reason="sink_np300 fixture absent; regenerate per tests/fixtures/README.md",
)


def test_load_groups_dispatches_particle():
    g, tok, n, model = runner._load_apophis_groups(FIX / "particle_np300", "sobol", runner.APOPHIS_SINK_ID_DEFAULT)
    assert model == "particle" and n >= 290 and len(g) >= 4


@needs_sink_fixture
def test_load_groups_dispatches_sink():
    g, tok, n, model = runner._load_apophis_groups(SINK_FIX, "sobol", runner.APOPHIS_SINK_ID_DEFAULT)
    assert model == "sink" and n >= 290


def test_compute_run_metrics_particle_fixture(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIX / "particle_np300", run)
    s = RunSample(use_dem=True, dem_model="particle", np_apophis=300, apophis_only=False)
    m = runner.compute_run_metrics(run, "sobol", s, runner.EARTH_SINK_ID_DEFAULT, runner.APOPHIS_SINK_ID_DEFAULT)
    assert math.isfinite(m.closest_km) and m.closest_km > 1e4   # ~1 hr of a flyby 3 days out
    # fixture spins at P=2 hr (omega/omega_crit ~1.01), so the pile spreads in both models
    assert 0.95 < m.dispersion_ratio < 3.0
    assert 0.0 <= m.unbound_fraction < 0.05


@needs_sink_fixture
def test_particle_and_sink_metrics_agree_on_fixture(tmp_path):
    out = {}
    for model in ("particle", "sink"):
        run = tmp_path / model
        shutil.copytree(FIX / f"{model}_np300", run)
        s = RunSample(use_dem=True, dem_model=model, np_apophis=300, apophis_only=False)
        out[model] = runner.compute_run_metrics(
            run, "sobol", s, runner.EARTH_SINK_ID_DEFAULT, runner.APOPHIS_SINK_ID_DEFAULT)
    assert out["particle"].closest_km == pytest.approx(out["sink"].closest_km, rel=1e-3)


def test_point_mass_run_uses_sink_ev_for_ca(tmp_path, monkeypatch):
    called = {}

    def fake_ca(run_dir, prefix, e, a):
        called["yes"] = True
        return 4.0e4, 4.0e4 / runner.AU_IN_KM, 1.0

    monkeypatch.setattr(runner, "_earth_apophis_closest_approach", fake_ca)
    (tmp_path / "sobol.setup").write_text(" np_apophis = 1\n use_dem = F\n use_dem_as_sinks = F\n")
    s = RunSample(np_apophis=1, apophis_only=False)
    m = runner.compute_run_metrics(tmp_path, "sobol", s, runner.EARTH_SINK_ID_DEFAULT, runner.APOPHIS_SINK_ID_DEFAULT)
    assert called and m.closest_km == 4.0e4
    assert math.isnan(m.dispersion_ratio)


def test_particle_loader_works_under_package_import():
    # reextract_spin_metrics / verify_metric_extraction import the runner as sobol.run_mass_sobol_phantom
    import subprocess
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from sobol.run_mass_sobol_phantom import _load_apophis_groups\n"
        "g, t, n, m = _load_apophis_groups(Path(%r), 'sobol', 11)\n"
        "assert m == 'particle' and n >= 290, (m, n)\n"
    ) % (str(repo), str(FIX / "particle_np300"))
    res = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-800:]


def test_wizard_argv_keeps_dem_model(monkeypatch):
    # full run_interactive_wizard is broken on this parser before this work (--spin-period-list
    # nargs='+'); exercise the per-action collector the wizard uses for dem_model
    import interactive_run_mass_sobol as wiz
    monkeypatch.setattr(wiz, "_read_line", lambda prompt: "")
    parser = runner.build_parser()
    state = parser.parse_args(["--dem-model", "sink"])
    action = next(a for a in parser._actions if a.dest == "dem_model")
    out = []
    wiz._collect_store(action, state, out)
    assert out == ["--dem-model", "sink"]
    assert wiz._section_for_dest("dem_model") == "Setup toggles"


def test_torque_reruns_parser_has_dem_model():
    import run_torque_align_blender_reruns as tr
    assert tr.build_parser().parse_args([]).dem_model == "particle"
    assert tr.build_parser().parse_args(["--dem-model", "sink"]).dem_model == "sink"
