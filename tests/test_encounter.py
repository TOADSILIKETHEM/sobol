import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import encounter as enc
import settled_body as sb
from run_mass_sobol_phantom import RunSample, resolve_default_shape_file

TEMPLATE = Path(__file__).parent.parent / "sobol.setup"

FAKE_FLYBY = r"""#!/bin/sh
echo "flyby $*" >> calls.log
[ -e "$2.in" ] && { echo "stale $2.in present"; exit 5; }
cat > flyby_stdin.txt
: > "$2_00000"
cp body.in "$2.in"
"""

BODY_IN = """\
                tmax =   1.000E+09    ! end time
               dtmax =   1.000E+08    ! time between dumps
           nfulldump =           1    ! full dump every n dumps
              kn_cgs =   1.000E+07    ! DEM normal spring constant
               idamp =           2    ! artificial damping of velocities
"""


def _val(text, key):
    return re.search(rf"^\s*{key}\s*=\s*([^!\n]*)", text, re.M).group(1).strip()


def _body(tmp_path):
    d = tmp_path / "body"
    d.mkdir()
    (d / "cropped_00002").write_text("dump")
    (d / "cropped.in").write_text(BODY_IN)
    (d / "body.json").write_text(json.dumps({
        "key": "np300_srho1_abc", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 310,
        "n_settled": 812, "packing_fraction": 0.651, "utime_s": 2.745e-6}))
    return sb.SettledBody.load(d)


def _flyby(tmp_path):
    p = tmp_path / "phantomflyby"
    p.write_text(FAKE_FLYBY)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def _sample():
    return RunSample(use_dem=True, np_apophis=300, body_source="settled", encounter="hyperbola",
                     flyby_rp_km=38000.0, flyby_vinf_kms=5.9, flyby_start_sep_km=4.0e5,
                     flyby_perturber_earth_masses=1.0, tmax_hours=48.0, dtmax_hours=0.5)


def _prepare(tmp_path):
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "sobol.in").write_text("stale template\n")  # moddump would read it as revised defaults
    cols = enc.prepare_hyperbola_run(run, "sobol", _sample(), _body(tmp_path), _flyby(tmp_path), TEMPLATE)
    return run, cols


def test_flyby_stdin_prompt_order():
    # moddump_earthflyby.f90: pericentre km, v_inf km/s, initial separation km, Earth masses
    assert enc.flyby_stdin(38000.0, 5.9, 4.0e5, 1.0) == "38000\n5.9\n400000\n1\n"


def test_prepare_hyperbola_run_turns_damping_off(tmp_path):
    run, _ = _prepare(tmp_path)
    t = (run / "sobol.in").read_text()
    assert _val(t, "idamp") == "0"
    assert _val(t, "nfulldump") == "1"
    assert float(_val(t, "tmax")) == pytest.approx(48.0 * 3600.0 / 2.745e-6, rel=1e-9)
    assert float(_val(t, "dtmax")) == pytest.approx(0.5 * 3600.0 / 2.745e-6, rel=1e-9)


def test_prepare_hyperbola_run_files(tmp_path):
    run, cols = _prepare(tmp_path)
    assert (run / "flyby_stdin.txt").read_text() == "38000\n5.9\n400000\n1\n"
    assert (run / "sobol_00000").is_file()
    s = (run / "sobol.setup").read_text()
    assert (_val(s, "use_dem"), _val(s, "use_dem_as_sinks"), _val(s, "apophis_only")) == ("T", "F", "F")
    assert _val(s, "np_apophis") == "310"
    m = json.loads((run / "encounter.json").read_text())
    assert m["mode"] == "hyperbola" and m["n_kept"] == 310 and m["rp_km"] == 38000.0
    assert cols == {"body_source": "settled", "encounter": "hyperbola", "flyby_rp_km": "38000",
                    "flyby_vinf_kms": "5.9", "np_kept": "310"}


@pytest.mark.skipif(os.environ.get("SOBOL_SLOW") != "1", reason="real settle + flyby; set SOBOL_SLOW=1")
def test_real_hyperbola_run_hits_pericentre(tmp_path):
    root = Path(__file__).parent.parent
    cmd = [sys.executable, str(root / "run_mass_sobol_phantom.py"),
           "--base-dir", str(root), "--prefix", "sobol", "--phantom-dir", str(root),
           "--output-root", str(tmp_path), "--ephemeris-cache-dir", str(root),
           "--np-apophis-list", "300", "--use-dem-fixed", "true",
           "--body-source", "settled", "--encounter", "hyperbola",
           "--flyby-rp-km", "38000", "--flyby-vinf-kms", "5.9", "--flyby-start-sep-km", "1e5",
           "--tmax-hours", "8", "--dtmax-hours", "0.1", "--settle-tdyn", "5", "--relax-tdyn", "0.25",
           "--jobs", "1", "--no-cleanup", "--batch-label", "flyby_smoke"]
    subprocess.run(cmd, check=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    import csv
    out = next(tmp_path.glob("sobol_*_flyby_smoke")) / "sobol_mass_outputs.csv"
    row = next(csv.DictReader(out.open()))
    assert row["status"] == "ok", row["error"]
    assert float(row["closest_approach_km"]) == pytest.approx(38000.0, rel=0.02)
    assert row["unbound_fraction"] != ""

SHAPE = resolve_default_shape_file()


def _ephem_body(tmp_path):
    d = tmp_path / "ebody"
    d.mkdir()
    (d / "cropped_00002").write_text("dump")
    (d / "cropped.in").write_text(BODY_IN)
    (d / "body.json").write_text(json.dumps({
        "key": "np300_srho1_abc", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 310,
        "n_settled": 812, "packing_fraction": 0.651, "utime_s": 2.745e-6,
        "spec": {"np_apophis": 300, "scale_rho": 1.0, "shape_file": str(SHAPE)}}))
    return sb.SettledBody.load(d)


def _ephem_run(tmp_path):
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "sobol.setup").write_text(TEMPLATE.read_text())
    return run


def test_settled_body_load_reads_shape_file_from_spec(tmp_path):
    assert _ephem_body(tmp_path).shape_file == str(SHAPE)
    assert _body(tmp_path).shape_file == ""  # manifests without a spec block still load


def test_template_setup_has_blank_packing_file_key():
    # phantomsetup since 853b86818 counts a missing packing_file key as a read error and stops;
    # blank = build the body as before. Pre-merge binaries (bin_demsync_7a243de) ignore the extra key.
    s = TEMPLATE.read_text()
    assert len(re.findall(r"^\s*packing_file\s*=", s, re.M)) == 1
    assert _val(s, "packing_file") == ""


def test_apply_settled_body_to_setup(tmp_path):
    run = _ephem_run(tmp_path)
    cols = enc.apply_settled_body_to_setup(run, "sobol", _ephem_body(tmp_path))
    s = (run / "sobol.setup").read_text()
    assert _val(s, "packing_file") == "settled_body"
    assert len(re.findall(r"^\s*packing_file\s*=", s, re.M)) == 1
    assert _val(s, "pack_settle") == "F"
    assert _val(s, "np_apophis") == "310"
    assert _val(s, "apophis_shape_file") == SHAPE.name
    assert (run / "settled_body").read_text() == "dump"
    assert (run / SHAPE.name).is_file()
    assert cols == {"body_source": "settled", "np_kept": "310"}


def test_apply_settled_body_to_setup_twice_keeps_one_key(tmp_path):
    run = _ephem_run(tmp_path)
    body = _ephem_body(tmp_path)
    enc.apply_settled_body_to_setup(run, "sobol", body)
    enc.apply_settled_body_to_setup(run, "sobol", body)
    s = (run / "sobol.setup").read_text()
    assert len(re.findall(r"^\s*packing_file\s*=", s, re.M)) == 1


def test_apply_settled_body_refuses_body_without_shape(tmp_path):
    with pytest.raises(RuntimeError, match="shape_file"):
        enc.apply_settled_body_to_setup(_ephem_run(tmp_path), "sobol", _body(tmp_path))


def test_check_packing_setup_log_ok(tmp_path):
    log = tmp_path / "setup.log"
    log.write_bytes(b" loaded 310 grains from settled_body\n\x00 placed 310 pre-built grains on the ephemeris orbit\n")
    enc.check_packing_setup_log(log, 310)


def test_check_packing_setup_log_stale_binary(tmp_path):
    log = tmp_path / "setup.log"
    log.write_text(" particles kept: 300\n")
    with pytest.raises(RuntimeError, match="packing_file"):
        enc.check_packing_setup_log(log, 310)


def test_check_packing_setup_log_sphere_fallback(tmp_path):
    log = tmp_path / "setup.log"
    log.write_text(" placed 310 pre-built grains on the ephemeris orbit\n"
                   " WARNING! apophis: could not read shape for packing_file volume: using a sphere\n")
    with pytest.raises(RuntimeError, match="sphere"):
        enc.check_packing_setup_log(log, 310)


def test_check_packing_setup_log_wrong_count(tmp_path):
    log = tmp_path / "setup.log"
    log.write_text(" placed 300 pre-built grains on the ephemeris orbit\n")
    with pytest.raises(RuntimeError, match="310"):
        enc.check_packing_setup_log(log, 310)
