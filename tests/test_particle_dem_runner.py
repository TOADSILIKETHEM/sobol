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
