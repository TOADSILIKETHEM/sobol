import os
import re
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import settled_body as sb

TEMPLATE = Path(__file__).parent.parent / "sobol.setup"


def _val(text, key):
    return re.search(rf"^\s*{key}\s*=\s*([^!\n]*)", text, re.M).group(1).strip()


def _shape(tmp_path, obj_text="v 0 0 0\n"):
    d = tmp_path / "shape"
    d.mkdir(exist_ok=True)
    (d / "tiny.obj").write_text(obj_text)
    (d / "apophis.shape").write_text("mesh tiny.obj 0.170\n")
    return d / "apophis.shape"


def test_resolve_dem_tools_prefers_bin(tmp_path):
    (tmp_path / "bin").mkdir()
    for n in sb.DEM_TOOL_NAMES:
        (tmp_path / "bin" / n).write_text("")
        (tmp_path / n).write_text("")
    t = sb.resolve_dem_tools(tmp_path)
    assert t.moddump == tmp_path / "bin" / "phantommoddump"
    assert t.flyby == tmp_path / "bin" / "phantomflyby"
    assert t.analysis == tmp_path / "bin" / "phantomanalysis"


def test_resolve_dem_tools_missing_names_make_target(tmp_path):
    with pytest.raises(FileNotFoundError, match="make demtools"):
        sb.resolve_dem_tools(tmp_path)


def test_tdyn_matches_mia_relax_time():
    # Mia: 0.5 t_dyn = 1305 s at rho = 2.2 g/cm^3 (scale_rho 0.815 of rho_0 2.7)
    assert abs(sb.tdyn_seconds(0.815) - 2610.0) / 2610.0 < 0.005


def test_settle_key_changes_with_binary_and_mesh(tmp_path):
    shape = _shape(tmp_path)
    b1 = tmp_path / "phantom"
    b1.write_bytes(b"v1")
    spec = sb.SettleSpec(np_apophis=300, scale_rho=1.0, shape_file=str(shape))
    k1 = sb.settle_key(spec, [b1])
    assert k1 == sb.settle_key(spec, [b1])
    assert k1.startswith("np300_srho1_")
    b1.write_bytes(b"v2")
    k2 = sb.settle_key(spec, [b1])
    assert k2 != k1
    (tmp_path / "shape" / "tiny.obj").write_text("v 1 0 0\n")
    assert sb.settle_key(spec, [b1]) != k2
    assert sb.settle_key(sb.SettleSpec(301, 1.0, str(shape)), [b1]) != sb.settle_key(spec, [b1])


def test_write_settle_setup_sets_settle_keys(tmp_path):
    spec = sb.SettleSpec(np_apophis=300, scale_rho=0.815, shape_file="x", settle_tdyn=5.0)
    dst = tmp_path / "settle.setup"
    sb.write_settle_setup(TEMPLATE, dst, spec, "apophis.shape")
    t = dst.read_text()
    for key, want in (("pack_settle", "T"), ("apophis_only", "T"), ("use_dem", "T"),
                      ("use_dem_as_sinks", "F"), ("np_apophis", "300"),
                      ("apophis_shape_file", "apophis.shape")):
        assert _val(t, key) == want, key
    assert float(_val(t, "scale_rho")) == 0.815
    assert float(_val(t, "mass_apophis")) == 0.0
    assert float(_val(t, "apophis_spin_period")) == 0.0
    tmax_hr = float(_val(t, "tmax_in").split()[0])
    assert abs(tmax_hr - 5.0 * sb.tdyn_seconds(0.815) / 3600.0) < 1e-3


def test_parse_crop_log():
    text = (" --- cropping a settled packing to shape ---\n"
            " grains kept          =       310  of       812\n"
            " packing fraction     = 0.651     \n")
    assert sb.parse_crop_log(text) == (310, 812, 0.651)


def test_parse_crop_log_rejects_missing_lines():
    with pytest.raises(RuntimeError, match="grains kept"):
        sb.parse_crop_log("ERROR: could not read shape file\n")


FAKE_SETUP = r"""#!/bin/sh
echo "phantomsetup $*" >> calls.log
grep -Eq '^ *pack_settle *= *T' "$1.setup" || { echo "pack_settle not T"; exit 4; }
cat > "$1.in" <<'EOF'
                tmax =   5.000E+00    ! end time
               dtmax =   1.000E+00    ! time between dumps
           nfulldump =          10    ! full dump every n dumps
              kn_cgs =   1.000E+07    ! DEM normal spring constant
               idamp =           2    ! artificial damping of velocities
EOF
: > "$1_00000.tmp"
"""

FAKE_PHANTOM = r"""#!/bin/sh
echo "phantom $*" >> calls.log
grep -Eq '^ *nfulldump *= *1( |$)' "$1" || { echo "nfulldump must be 1 in $1"; exit 3; }
p=$(basename "$1" .in)
echo "     Mass:  1.989E+33 g       Length:  1.000E+05 cm    Time:  2.745E-06 s"
: > "${p}_00001"; : > "${p}_00002"
"""

FAKE_CROP = r"""#!/bin/sh
echo "moddump $*" >> calls.log
cat > crop_stdin.txt
echo " grains kept          =       310  of       812"
echo " packing fraction     = 0.651"
: > "$2_00000"
cp settle.in "$2.in"
"""


def _exe(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_bins(tmp_path):
    d = tmp_path / "bins"
    d.mkdir()
    setup = _exe(d / "phantomsetup", FAKE_SETUP)
    phantom = _exe(d / "phantom", FAKE_PHANTOM)
    crop = _exe(d / "phantommoddump", FAKE_CROP)
    tools = sb.DemTools(moddump=crop, flyby=d / "phantomflyby", analysis=d / "phantomanalysis")
    return setup, phantom, tools


def _build(tmp_path, spec, cache):
    if not (tmp_path / "bins").exists():
        _fake_bins(tmp_path)
    d = tmp_path / "bins"
    tools = sb.DemTools(d / "phantommoddump", d / "phantomflyby", d / "phantomanalysis")
    return sb.build_settled_body(spec, cache, TEMPLATE, d / "phantomsetup", d / "phantom", tools, None)


def test_build_settled_body_runs_all_stages(tmp_path):
    spec = sb.SettleSpec(np_apophis=300, scale_rho=1.0, shape_file=str(_shape(tmp_path)))
    body = _build(tmp_path, spec, tmp_path / "cache")
    assert (body.n_kept, body.n_settled, body.packing_fraction) == (310, 812, 0.651)
    assert body.utime_s == pytest.approx(2.745e-6)
    assert body.dump == "cropped_00002" and body.infile == "cropped.in"
    assert (body.dir / "body.json").is_file() and not body.dir.name.endswith(".partial")
    assert (body.dir / "crop_stdin.txt").read_text() == "apophis.shape\n2.7\n"
    relax_tmax = float(_val((body.dir / "cropped.in").read_text(), "tmax"))
    assert relax_tmax == pytest.approx(0.5 * sb.tdyn_seconds(1.0) / 2.745e-6, rel=1e-6)
    assert sb.SettledBody.load(body.dir) == body


def test_build_reuses_cache(tmp_path):
    spec = sb.SettleSpec(np_apophis=300, scale_rho=1.0, shape_file=str(_shape(tmp_path)))
    body = _build(tmp_path, spec, tmp_path / "cache")
    n_calls = len((body.dir / "calls.log").read_text().splitlines())
    again = _build(tmp_path, spec, tmp_path / "cache")
    assert again == body
    assert len((body.dir / "calls.log").read_text().splitlines()) == n_calls


def test_build_discards_partial_dir(tmp_path):
    spec = sb.SettleSpec(np_apophis=300, scale_rho=1.0, shape_file=str(_shape(tmp_path)))
    setup, phantom, tools = _fake_bins(tmp_path)
    key = sb.settle_key(spec, (setup, phantom, tools.moddump))
    junk = tmp_path / "cache" / f"{key}.partial" / "junk"
    junk.parent.mkdir(parents=True)
    junk.write_text("half-built")
    body = sb.build_settled_body(spec, tmp_path / "cache", TEMPLATE, setup, phantom, tools, None)
    assert not junk.parent.exists()
    assert not (body.dir / "junk").exists()


def test_relax_zero_skips_relax_stage(tmp_path):
    spec = sb.SettleSpec(np_apophis=300, scale_rho=1.0, shape_file=str(_shape(tmp_path)), relax_tdyn=0.0)
    body = _build(tmp_path, spec, tmp_path / "cache")
    assert body.dump == "cropped_00000"
    assert (body.dir / "calls.log").read_text().count("phantom cropped.in") == 0


@pytest.mark.skipif(os.environ.get("SOBOL_SLOW") != "1", reason="real settle; set SOBOL_SLOW=1")
def test_real_settle_np300(tmp_path):
    root = Path(__file__).parent.parent
    tools = sb.resolve_dem_tools(root)
    import run_mass_sobol_phantom as r
    spec = sb.SettleSpec(np_apophis=300, scale_rho=0.815,
                         shape_file=str(r.resolve_default_shape_file()), settle_tdyn=1.0, relax_tdyn=0.25)
    body = sb.build_settled_body(spec, tmp_path / "cache", root / "sobol.setup",
                                 r.resolve_phantom_executable(root, "phantomsetup", must_exist=True),
                                 r.resolve_phantom_executable(root, "phantom", must_exist=True),
                                 tools, root)
    assert 0.85 * 300 <= body.n_kept <= 1.15 * 300
    assert 0.5 < body.packing_fraction < 0.75
