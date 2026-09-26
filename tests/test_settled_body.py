import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import settled_body as sb


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
