import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd
import run_mass_sobol_phantom as runner
from tests.test_encounter import _body, _flyby, _sample, TEMPLATE, _val
import encounter as enc

TEMPLATE_IN = Path(__file__).parent.parent / "sobol.in"


def _sample_with_boulders(**kw):
    return runner.RunSample(use_dem=True, boulders=bd.BoulderSpec(bond_sigma_pa=10.0, **kw))


def test_apply_in_inserts_bond_keys(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   2.000E+07    ! DEM normal spring constant\n")
    cols = runner.apply_run_sample_to_in(p, _sample_with_boulders())
    t = p.read_text()
    assert float(_val(t, "kb_cgs")) == 2e7          # default = kn_cgs of this .in
    assert float(_val(t, "bond_sigma_pa")) == 10.0 and float(_val(t, "bond_tau_pa")) == 10.0
    assert float(_val(t, "bond_reach")) == 0.1
    assert cols["kb_cgs"] == "20000000"


def test_apply_in_replaces_present_bond_keys(tmp_path):
    p = tmp_path / "fly.in"
    p.write_text("              kn_cgs =   1.000E+07    ! x\n"
                 "              kb_cgs =       0.000    ! x\n"
                 "          bond_reach =       0.100    ! x\n"
                 "       bond_sigma_pa =       0.000    ! x\n"
                 "         bond_tau_pa =       0.000    ! x\n")
    runner.apply_run_sample_to_in(p, _sample_with_boulders(kb_cgs=3e7))
    t = p.read_text()
    assert float(_val(t, "kb_cgs")) == 3e7
    assert len(re.findall(r"^\s*kb_cgs\s*=", t, re.M)) == 1


def _json_n_boulders(run_dir, n):
    (run_dir / "boulders.json").write_text(json.dumps({"spec": {}, "result": {"n_boulders": n}}))


def test_apply_in_skips_bond_keys_when_no_boulders_made(tmp_path):
    p = tmp_path / "sobol.in"
    original = "              kn_cgs =   2.000E+07    ! DEM normal spring constant\n"
    p.write_text(original)
    _json_n_boulders(tmp_path, 0)
    assert runner.apply_run_sample_to_in(p, _sample_with_boulders()) == {}
    assert p.read_text() == original


def test_apply_in_writes_bond_keys_when_boulders_made(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   2.000E+07    ! DEM normal spring constant\n")
    _json_n_boulders(tmp_path, 4)
    cols = runner.apply_run_sample_to_in(p, _sample_with_boulders())
    assert float(_val(p.read_text(), "kb_cgs")) == 2e7 and "bond_sigma_pa" in cols


def test_apply_in_all_dust_keeps_other_in_axes(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   2.000E+07    ! x\n              mu_s =   0.000    ! x\n")
    _json_n_boulders(tmp_path, 0)
    s = _sample_with_boulders()
    s.mu_s = 0.4
    cols = runner.apply_run_sample_to_in(p, s)
    assert cols == {"mu_s": "0.4"} and "kb_cgs" not in p.read_text()


def test_apply_in_without_boulders_unchanged(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   1.000E+07    ! x\n")
    assert runner.apply_run_sample_to_in(p, runner.RunSample(use_dem=True)) == {}
    assert "kb_cgs" not in p.read_text()


def test_settled_setup_stages_given_dump(tmp_path):
    body = _body(tmp_path)
    object.__setattr__(body, "shape_file", str(Path(__file__).parent.parent.parent / "Shapes" / "apophis.shape"))
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "sobol.setup").write_text(TEMPLATE.read_text())
    (run / "clumped_00000").write_text("clumped dump")
    enc.apply_settled_body_to_setup(run, "sobol", body, dump=run / "clumped_00000")
    assert (run / enc.PACKING_FILE_NAME).read_text() == "clumped dump"


def test_hyperbola_flies_given_dump(tmp_path):
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "clumped_00000").write_text("clumped dump")
    enc.prepare_hyperbola_run(run, "sobol", _sample(), _body(tmp_path), _flyby(tmp_path), TEMPLATE,
                              dump=run / "clumped_00000")
    assert (run / "body_00000").read_text() == "clumped dump"


def test_cleanup_removes_clumped_dump(tmp_path):
    for name in ("clumped_00000", "clumped.in", "unclumped_00000", "boulders.json", "clumps.log", "sobol_00001"):
        (tmp_path / name).write_text("x")
    runner._cleanup_run_dir(tmp_path, "sobol")
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["boulders.json", "clumps.log"]
