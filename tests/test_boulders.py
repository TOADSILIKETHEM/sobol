import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd
import settled_body as sb

# Summary block of a real run (scratchpad spike 2026-10-06, body np300_srho1_712bb7806311,
# f_boulder=0.3, contact binary, all other prompts default).
CLUMPS_LOG_BINARY = """\
 --- gluing grains into boulders ---
 contact binary along the long axis
 lobe 1:       65 grains, radius  1.072E-01 km
 lobe 2:       64 grains, radius  8.105E-02 km
 grains               =       391
 placed boulders      =         2  holding       129 grains ( 0.330 of the body)
 random boulders      =         8  shape: round
 grains in random     =       142  ( 0.363 of the body, target  0.300)
 grains per random boulder =      6 to     46, mean     17.8
 dust (free grains)   =  0.307
 grains cut off by carving, back to dust =         2
 random boulders under the minimum size, skipped =         1
 now set kb_cgs > 0 (and the same bond_reach) in the .in file and settle again
"""

CLUMPS_LOG_DUST = """\
 --- gluing grains into boulders ---
 grains               =       391
 random boulders      =         0  shape: round
 grains in random     =         0  ( 0.000 of the body, target  0.000)
 dust (free grains)   =  1.000
 grains cut off by carving, back to dust =         0
 random boulders under the minimum size, skipped =         0
"""


def test_stdin_defaults_round_none():
    s = bd.BoulderSpec()
    assert bd.clumps_stdin(s).split("\n") == ["0.3", "1.5", "4", "3.5", "0", "0", "0.1", "", "", ""]


@pytest.mark.parametrize("shape,extra", [("round", []), ("ellipsoid", ["0.5"]), ("grown", ["0.7"])])
@pytest.mark.parametrize("placed,pextra", [("none", []), ("binary", ["0.6"]), ("file", ["boulders.txt"])])
def test_stdin_conditional_prompts(shape, extra, placed, pextra):
    s = bd.BoulderSpec(shape=shape, axis_min=0.5 if shape == "ellipsoid" else None,
                       jagged=0.7 if shape == "grown" else None, placed=placed,
                       lobe_ratio=0.6 if placed == "binary" else None,
                       placed_file="/x/b.txt" if placed == "file" else None, nmin_clump=3, seed=42)
    lines = bd.clumps_stdin(s).rstrip("\n").split("\n")
    expect = (["0.3", "1.5", "4", "3.5", str(bd.SHAPES.index(shape))] + extra
              + [str(bd.PLACED.index(placed))] + pextra + ["0.1", "3", "42"])
    assert lines == expect


def test_parse_binary_log():
    r = bd.parse_clumps_log(CLUMPS_LOG_BINARY)
    assert (r.n_grains, r.n_placed, r.n_placed_grains, r.n_random, r.n_random_grains) == (391, 2, 129, 8, 142)
    assert r.n_dropped == 2 and r.f_dust == pytest.approx(0.307) and not r.ran_out
    assert r.n_boulders == 10 and r.n_in_boulders == 271
    assert r.f_boulder_actual == pytest.approx(271 / 391)


def test_parse_dust_log_has_no_placed_line():
    r = bd.parse_clumps_log(CLUMPS_LOG_DUST)
    assert (r.n_placed, r.n_random, r.n_boulders, r.f_boulder_actual) == (0, 0, 0, 0.0)


def test_parse_flags_ran_out():
    text = CLUMPS_LOG_BINARY + " *** WARNING: ran out of free grains or attempts before the random target ***\n"
    assert bd.parse_clumps_log(text).ran_out


def test_parse_without_summary_raises():
    with pytest.raises(RuntimeError, match="summary"):
        bd.parse_clumps_log("Phantom moddump: crashed\n")


def test_bond_in_keys_defaults():
    k = bd.bond_in_keys(bd.BoulderSpec(bond_sigma_pa=10.0), kn_cgs=1e7)
    assert k == {"kb_cgs": 1e7, "bond_reach": 0.1, "bond_sigma_pa": 10.0, "bond_tau_pa": 10.0}


def test_bond_in_keys_overrides():
    k = bd.bond_in_keys(bd.BoulderSpec(bond_sigma_pa=10.0, bond_tau_ratio=0.5, kb_cgs=3e7, bond_reach=0.05), 1e7)
    assert k == {"kb_cgs": 3e7, "bond_reach": 0.05, "bond_sigma_pa": 10.0, "bond_tau_pa": 5.0}


@pytest.mark.parametrize("kw,match", [
    ({"f_boulder": 1.2}, "f_boulder"),
    ({"rb_min": 0.5}, "rb_min"),
    ({"rb_min": 3.0, "rb_max": 2.0}, "rb_max"),
    ({"q_slope": -1.0}, "q_slope"),
    ({"shape": "cube"}, "shape"),
    ({"axis_min": 0.5}, "ellipsoid"),
    ({"shape": "ellipsoid", "axis_min": 0.05}, "axis_min"),
    ({"jagged": 0.5}, "grown"),
    ({"lobe_ratio": 0.5}, "binary"),
    ({"placed": "binary", "lobe_ratio": 0.05}, "lobe_ratio"),
    ({"placed": "file"}, "file"),
    ({"placed_file": "/tmp/x"}, "file"),
    ({"nmin_clump": 1}, "nmin"),
    ({"bond_sigma_pa": -1.0}, "bond_sigma_pa"),
    ({"bond_tau_ratio": -1.0}, "bond_tau_ratio"),
    ({"kb_cgs": 0.0}, "kb_cgs"),
    ({"bond_reach": 0.0}, "bond_reach"),
])
def test_validate_spec_rejects(kw, match):
    with pytest.raises(ValueError, match=match):
        bd.validate_spec(bd.BoulderSpec(**kw))


def test_validate_spec_file_must_exist(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        bd.validate_spec(bd.BoulderSpec(placed="file", placed_file=str(tmp_path / "nope.txt")))
    f = tmp_path / "b.txt"
    f.write_text("0 0 0 0.05 0.05 0.05\n")
    bd.validate_spec(bd.BoulderSpec(placed="file", placed_file=str(f)))


def test_requested():
    assert bd.requested(bd.BoulderSpec(), 391)
    assert bd.requested(bd.BoulderSpec(f_boulder=0.0, placed="binary"), 391)
    assert not bd.requested(bd.BoulderSpec(f_boulder=0.0), 391)


def test_requested_uses_fortran_nint_target():
    # moddump_clumps.f90: ntarget = nint(f_boulder * npart), half away from zero
    assert not bd.requested(bd.BoulderSpec(f_boulder=0.001), 391)   # 0.391 -> 0
    assert bd.requested(bd.BoulderSpec(f_boulder=0.002), 391)       # 0.782 -> 1
    assert bd.requested(bd.BoulderSpec(f_boulder=0.5), 1)           # exactly 0.5 -> 1 (Python round gives 0)
    assert not bd.requested(bd.BoulderSpec(f_boulder=0.4), 1)
    assert bd.requested(bd.BoulderSpec(f_boulder=0.001, placed="binary"), 391)


def test_fortran_nint():
    assert [bd.fortran_nint(x) for x in (0.0, 0.49, 0.5, 1.5, 2.5, 2.51)] == [0, 0, 1, 2, 3, 3]


def test_resolve_takes_axis_values_from_sample():
    base = bd.BoulderSpec(bond_sigma_pa=10.0)
    s = SimpleNamespace(boulders=base, f_boulder=0.55, rb_max=None)
    r = bd.resolve(s)
    assert (r.f_boulder, r.rb_max, r.bond_sigma_pa) == (0.55, 4.0, 10.0)


def test_check_bonds_on():
    bd.check_bonds_on("...\n DEM clump bonds enabled\n   kb_cgs =   1.0000E+07\n")
    with pytest.raises(RuntimeError, match="unbonded"):
        bd.check_bonds_on("no bonds here\n")


# Task 2: run side tests

FAKE_CLUMPS = r"""#!/bin/sh
echo "clumps $*" >> calls.log
[ -e "$2.in" ] && { echo "stale $2.in present"; exit 5; }
cat > clumps_stdin.txt
cp "$1" "$2_00000"
cat "$FAKE_CLUMPS_LOG"
"""


def _fake_clumps(tmp_path, log_text):
    log = tmp_path / "fake_clumps_output.txt"
    log.write_text(log_text)
    p = tmp_path / "phantomclumps"
    p.write_text(FAKE_CLUMPS)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p, log


def _body(tmp_path):
    d = tmp_path / "body"
    d.mkdir()
    (d / "cropped_00002").write_text("dump")
    (d / "cropped.in").write_text("kn_cgs = 1.0E+07\n")
    (d / "body.json").write_text(json.dumps({
        "key": "np300_srho1_abc", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 391,
        "n_settled": 923, "packing_fraction": 0.83, "utime_s": 2.745e-6}))
    return sb.SettledBody.load(d)


def _run(tmp_path, monkeypatch, log_text, spec):
    clumps, log = _fake_clumps(tmp_path, log_text)
    monkeypatch.setenv("FAKE_CLUMPS_LOG", str(log))
    run = tmp_path / "run_0001"
    run.mkdir()
    s = SimpleNamespace(boulders=spec, f_boulder=None, rb_max=None, phantomclumps_bin=str(clumps))
    return run, bd.clump_sample(run, _body(tmp_path), s)


def test_clump_body_runs_tool_and_stages_body(tmp_path, monkeypatch):
    clumps, log = _fake_clumps(tmp_path, CLUMPS_LOG_BINARY)
    monkeypatch.setenv("FAKE_CLUMPS_LOG", str(log))
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "clumped.in").write_text("stale")  # moddump would read it as revised defaults
    spec = bd.BoulderSpec(placed="binary", bond_sigma_pa=10.0)
    r = bd.clump_body(run, _body(tmp_path), spec, clumps)
    assert r.n_boulders == 10
    assert (run / "unclumped_00000").read_text() == "dump"
    assert (run / "unclumped.in").is_file()
    assert (run / bd.CLUMPED_DUMP).is_file()
    assert (run / "clumps_stdin.txt").read_text() == bd.clumps_stdin(spec)
    assert "unclumped_00000 clumped 0 --maxp=2000" in (run / "calls.log").read_text()


def test_clump_body_failure_raises(tmp_path):
    bad = tmp_path / "phantomclumps"
    bad.write_text("#!/bin/sh\nexit 3\n")
    bad.chmod(bad.stat().st_mode | stat.S_IXUSR)
    run = tmp_path / "run_0001"
    run.mkdir()
    with pytest.raises(RuntimeError, match="phantomclumps failed"):
        bd.clump_body(run, _body(tmp_path), bd.BoulderSpec(), bad)


def test_clump_body_stages_placed_file(tmp_path, monkeypatch):
    clumps, log = _fake_clumps(tmp_path, CLUMPS_LOG_BINARY)
    monkeypatch.setenv("FAKE_CLUMPS_LOG", str(log))
    f = tmp_path / "my_boulders.txt"
    f.write_text("0 0 0 0.05 0.05 0.05\n")
    run = tmp_path / "run_0001"
    run.mkdir()
    bd.clump_body(run, _body(tmp_path), bd.BoulderSpec(placed="file", placed_file=str(f)), clumps)
    assert (run / bd.PLACED_FILE_NAME).read_text() == f.read_text()


def test_clump_sample_columns_and_json(tmp_path, monkeypatch):
    run, cols = _run(tmp_path, monkeypatch, CLUMPS_LOG_BINARY, bd.BoulderSpec(placed="binary", bond_sigma_pa=10.0))
    assert cols == {"f_boulder": "0.3", "rb_max": "4", "f_boulder_actual": f"{271 / 391:.6g}",
                    "n_boulders": "10", "boulder_placed": "binary", "bond_sigma_pa": "10"}
    info = json.loads((run / "boulders.json").read_text())
    assert info["spec"]["placed"] == "binary" and info["result"]["n_boulders"] == 10


def test_clump_sample_no_boulders_when_requested_raises(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no boulders"):
        _run(tmp_path, monkeypatch, CLUMPS_LOG_DUST, bd.BoulderSpec(f_boulder=0.3, bond_sigma_pa=10.0))


def test_clump_sample_tiny_f_boulder_all_dust_is_not_an_error(tmp_path, monkeypatch):
    # nint(0.001 * 391) = 0: phantomclumps makes nothing, which is what was asked for
    run, cols = _run(tmp_path, monkeypatch, CLUMPS_LOG_DUST, bd.BoulderSpec(f_boulder=0.001, bond_sigma_pa=10.0))
    assert cols["n_boulders"] == "0" and cols["f_boulder"] == "0.001"


def test_bonds_wanted(tmp_path):
    assert bd.bonds_wanted(tmp_path)                       # no boulders.json: bonds on
    (tmp_path / "boulders.json").write_text(json.dumps({"spec": {}, "result": {"n_boulders": 0}}))
    assert not bd.bonds_wanted(tmp_path)
    (tmp_path / "boulders.json").write_text(json.dumps({"spec": {}, "result": {"n_boulders": 3}}))
    assert bd.bonds_wanted(tmp_path)


def test_clump_sample_all_dust_is_not_an_error(tmp_path, monkeypatch):
    run, cols = _run(tmp_path, monkeypatch, CLUMPS_LOG_DUST, bd.BoulderSpec(f_boulder=0.0, bond_sigma_pa=10.0))
    assert cols["n_boulders"] == "0"


def test_clump_sample_needs_binary(tmp_path):
    s = SimpleNamespace(boulders=bd.BoulderSpec(), f_boulder=None, rb_max=None, phantomclumps_bin=None)
    with pytest.raises(RuntimeError, match="make demtools"):
        bd.clump_sample(tmp_path, _body(tmp_path), s)


def _write_info(run, n_boulders):
    (run / "boulders.json").write_text(json.dumps({"spec": {}, "result": {"n_boulders": n_boulders}}))


def test_check_run_logs(tmp_path):
    _write_info(tmp_path, 3)
    (tmp_path / "setup.log").write_text(" boulders in body     = 3 (40 grains)\n")
    (tmp_path / "phantom.log").write_text(" DEM clump bonds enabled\n")
    bd.check_run_logs(tmp_path, setup_log=True)
    (tmp_path / "setup.log").write_text("placed 391 pre-built grains\n")
    with pytest.raises(RuntimeError, match="boulders in body"):
        bd.check_run_logs(tmp_path, setup_log=True)
    bd.check_run_logs(tmp_path, setup_log=False)  # hyperbola: no phantomsetup
    (tmp_path / "phantom.log").write_text("nothing\n")
    with pytest.raises(RuntimeError, match="unbonded"):
        bd.check_run_logs(tmp_path, setup_log=False)


def test_check_run_logs_skips_all_dust(tmp_path):
    _write_info(tmp_path, 0)
    bd.check_run_logs(tmp_path, setup_log=True)  # no setup.log / phantom.log needed


def test_attach_tools(tmp_path):
    (tmp_path / "phantomclumps").write_text("x")
    phantom = tmp_path / "phantom"
    phantom.write_bytes(b"\x00" + bd.BOND_BINARY_MARKER + b"\x00")
    on = SimpleNamespace(boulders=bd.BoulderSpec(), phantomclumps_bin=None)
    off = SimpleNamespace(boulders=None, phantomclumps_bin=None)
    bd.attach_tools([on, off], tmp_path, phantom, dry_run=False)
    assert on.phantomclumps_bin == str(tmp_path / "phantomclumps") and off.phantomclumps_bin is None


def test_attach_tools_refuses_old_phantom(tmp_path):
    (tmp_path / "phantomclumps").write_text("x")
    phantom = tmp_path / "phantom"
    phantom.write_bytes(b"old build")
    with pytest.raises(RuntimeError, match="clump bonds"):
        bd.attach_tools([SimpleNamespace(boulders=bd.BoulderSpec(), phantomclumps_bin=None)],
                        tmp_path, phantom, dry_run=False)


def test_attach_tools_missing_clumps(tmp_path):
    with pytest.raises(FileNotFoundError, match="make demtools"):
        bd.attach_tools([SimpleNamespace(boulders=bd.BoulderSpec(), phantomclumps_bin=None)],
                        tmp_path, tmp_path / "phantom", dry_run=False)
