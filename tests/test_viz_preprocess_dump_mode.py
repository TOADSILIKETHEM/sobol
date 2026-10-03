"""viz_preprocess.py (composite) --input-mode dump: same assets as npz mode, plus camera grains."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._paths import CODE

_spec = importlib.util.spec_from_file_location("viz_preprocess", CODE / "viz" / "viz_preprocess.py")
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)

sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
from run_demtocsv_batch import DEFAULT_DEM_DUMP_CONVERT, run_dem_dump_convert  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _both_routes(tmp_path):
    run = tmp_path / "phantom" / "batch" / "run_0001"
    shutil.copytree(FIX / "particle_np300", run)
    out = tmp_path / "npz_route" / "batch"
    run_dem_dump_convert(run, DEFAULT_DEM_DUMP_CONVERT, tmp_path / "npz_route", None)
    a = vp.run_preprocess(out / "run_0001_grains_output", out / "run_0001_viz",
                          bodies_dir=out / "run_0001_bodies_output")
    d = tmp_path / "dump_route" / "batch"
    b = vp.run_preprocess(None, d / "run_0001_viz", run_dir=run, bodies_dir=d / "run_0001_bodies_output")
    return out, a, d, b


def _tree(root: Path, sub: str):
    return sorted(p.relative_to(root) for p in (root / sub).rglob("*") if p.is_file())


def test_dump_mode_assets_match_npz_mode(tmp_path):
    out, a, d, b = _both_routes(tmp_path)
    for sub in ("envelopes", "ejecta", "metrics"):
        ta, tb = _tree(a.parent, sub), _tree(b.parent, sub)
        assert ta == tb, sub
        for rel in ta:
            fa, fb = a.parent / rel, b.parent / rel
            if fa.suffix == ".json":
                assert fa.read_text() == fb.read_text(), rel
            else:
                xa, xb = np.load(fa), np.load(fb)
                assert set(xa.files) == set(xb.files)
                for k in xa.files:
                    np.testing.assert_array_equal(xa[k], xb[k], err_msg=f"{rel}:{k}")
    ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
    assert ma.pop("input_mode") == "npz" and mb.pop("input_mode") == "dump"
    for k in ("grains_dir", "bodies_csv_dir"):
        ma.pop(k), mb.pop(k)
    assert ma == mb
    ca = sorted((out / "run_0001_bodies_output").glob("*.csv"))
    cb = sorted((d / "run_0001_bodies_output").glob("*.csv"))
    assert [c.name for c in ca] == [c.name for c in cb]
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(ca, cb))


def test_dump_mode_writes_camera_grains(tmp_path):
    out, a, d, b = _both_routes(tmp_path)
    m = json.loads(b.read_text())
    assert m["grains_dir"] == "grains"
    cam = sorted((b.parent / "grains").glob("*.npz"))
    assert [p.stem for p in cam] == m["frame_ids"]
    full = sorted((out / "run_0001_grains_output").glob("*.npz"))
    for c, f in zip(cam, full):
        dc, df = np.load(c), np.load(f)
        assert sorted(dc.files) == ["x_vis", "y_vis", "z_vis"]
        for k in dc.files:
            assert dc[k].dtype == np.float64
            np.testing.assert_array_equal(dc[k], df[k])
    # npz mode points Blender at the full grains npz, and writes no grains/ dir
    assert not (a.parent / "grains").exists()


def test_dump_mode_no_dumps_raises(tmp_path):
    empty = tmp_path / "run_0001"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No usable DEM dumps"):
        vp.run_preprocess(None, tmp_path / "viz", run_dir=empty, bodies_dir=tmp_path / "b")


@pytest.mark.parametrize("args,msg", [
    (["--input-mode", "dump"], "--input-mode dump needs --run-dir and --bodies-dir"),
    (["--input-mode", "npz"], "--input-mode npz needs --grains-dir"),
])
def test_cli_needs_the_dir_for_its_mode(tmp_path, capsys, args, msg):
    with pytest.raises(SystemExit) as exc:
        vp.main(args + ["--output-dir", str(tmp_path / "v"), "--bodies-dir", str(tmp_path / "b")])
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err
