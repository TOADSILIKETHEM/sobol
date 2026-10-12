"""Code/make_asset_pack.py: shareable example pack, no materials/textures/HDRI."""
import sys
import zipfile

import pytest

from honours_code import CODE_DIR

sys.path.insert(0, str(CODE_DIR))
import make_asset_pack as mp  # noqa: E402


@pytest.mark.skipif(not mp.EXAMPLE_RUN.is_dir(), reason="example run not on this machine")
def test_pack_has_example_run_and_shape_but_no_blender_assets(tmp_path):
    assert mp.main(["--out", str(tmp_path)]) == 0
    names = zipfile.ZipFile(tmp_path / "honours_viz_pack.zip").namelist()
    run = [n for n in names if n.startswith("honours_viz_pack/example_run/run_0001/sobol_")]
    assert len(run) == len(list(mp.EXAMPLE_RUN.glob("sobol_[0-9]*")))
    assert "honours_viz_pack/example_run/run_0001/sobol.setup" in names
    assert "honours_viz_pack/example_run/run_0001/encounter.json" in names
    assert any(n.endswith("apophis_v233s7.obj") for n in names)
    assert "honours_viz_pack/PACK_README.txt" in names
    assert not [n for n in names if n.endswith((".blend", ".exr", ".hdr", ".png", ".jpg"))]
