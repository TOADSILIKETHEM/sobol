import re
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
