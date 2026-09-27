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
from run_mass_sobol_phantom import RunSample

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
