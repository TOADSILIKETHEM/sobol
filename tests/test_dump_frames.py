"""viz/dump_frames.py: one PHANTOM DEM dump -> grain frame dict + bodies table.

Parity reference = the current CSVconvert/DEMDumpConvert.py output for the same dumps.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._paths import CODE

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from viz import dump_frames as dfm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
from run_demtocsv_batch import DEFAULT_DEM_DUMP_CONVERT, run_dem_dump_convert  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
needs_sink_fixture = pytest.mark.skipif(
    not (FIX / "sink_np300" / "sobol_00000").is_file(),
    reason="sink_np300 fixture absent; regenerate per tests/fixtures/README.md",
)


def _converted(tmp_path, name, min_grains=None, drop_setup=False):
    run = tmp_path / "batch" / "run_0001"
    shutil.copytree(FIX / name, run)
    if drop_setup:
        (run / "sobol.setup").unlink()
    out = tmp_path / "out"
    run_dem_dump_convert(run, DEFAULT_DEM_DUMP_CONVERT, out, min_grains)
    return run, out / "batch"


def _assert_frames_match_converter(run, out, min_grains=None):
    frames = list(dfm.iter_dump_frames(run, min_dem_grains=min_grains))
    npz = sorted((out / "run_0001_grains_output").glob("*.npz"))
    csv = sorted((out / "run_0001_bodies_output").glob("*.csv"))
    assert frames, "fixture produced no frames"
    assert [f.name for f in frames] == [p.stem for p in npz] == [p.stem for p in csv]
    for f, p, c in zip(frames, npz, csv):
        ref = np.load(p)
        assert set(f.grains) == set(ref.files)
        for k in ref.files:
            got = np.asarray(f.grains[k])
            assert got.dtype == ref[k].dtype, k
            np.testing.assert_array_equal(got, ref[k], err_msg=k)  # NaN == NaN here
        assert f.bodies.to_csv(index=False) == c.read_text()


def test_particle_frames_match_converter(tmp_path):
    run, out = _converted(tmp_path, "particle_np300")
    _assert_frames_match_converter(run, out)


def test_flyby_frames_match_converter(tmp_path):
    run, out = _converted(tmp_path, "particle_flyby_np300")
    _assert_frames_match_converter(run, out)
    for f in dfm.iter_dump_frames(run):
        assert list(f.bodies["name"]) == ["Earth", "Apophis"]


@needs_sink_fixture
def test_sink_frames_match_converter(tmp_path):
    run, out = _converted(tmp_path, "sink_np300")
    _assert_frames_match_converter(run, out)


@needs_sink_fixture
def test_sink_without_setup_frames_match_converter(tmp_path):
    run, out = _converted(tmp_path, "sink_np300", min_grains=250, drop_setup=True)
    _assert_frames_match_converter(run, out, min_grains=250)


def test_read_setup_flags():
    flags = dfm.read_setup_flags(FIX / "particle_np300")
    assert flags.dem_model == "particle"
    assert flags.np_apophis == 300
    assert flags.min_dem_grains_auto == 250
    assert flags.encounter == "ephemeris"
    assert flags.body_names[3] == "Earth"
    fly = dfm.read_setup_flags(FIX / "particle_flyby_np300")
    assert fly.encounter == "hyperbola" and fly.body_names == ["Earth"]


def test_missing_setup_defaults_to_sink_and_450(tmp_path):
    flags = dfm.read_setup_flags(tmp_path)
    assert flags.dem_model == "sink"
    assert flags.min_dem_grains_auto == 450


def test_mini_dump_returns_skip_reason():
    flags = dfm.read_setup_flags(FIX / "particle_np300")
    got = dfm.frame_from_dump(FIX / "particle_np300" / "sobol_00000", flags, min_dem_grains=10**6)
    assert isinstance(got, str) and "mini dump" in got


def test_no_dumps_yields_nothing(tmp_path):
    assert dfm.dump_paths(tmp_path) == []
    assert list(dfm.iter_dump_frames(tmp_path)) == []


def test_unreadable_dump_is_logged_not_raised(tmp_path, capsys):
    (tmp_path / "sobol_00000").write_bytes(b"not a phantom dump")
    assert list(dfm.iter_dump_frames(tmp_path, min_dem_grains=1)) == []
    assert "ERROR" in capsys.readouterr().out


def test_import_needs_no_sarracen_or_pandas():
    code = (
        "import sys; sys.modules['sarracen'] = None; sys.modules['pandas'] = None; "
        f"sys.path.insert(0, {str(CODE)!r}); import viz.dump_frames"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_converter_uses_dump_frames():
    src = Path("/home/mboyle/Honours/Code/CSVconvert/DEMDumpConvert.py").read_text()
    assert "from viz.dump_frames import" in src
    assert "sarracen.read_phantom" not in src  # dump reading lives in one place


def test_dump_mode_bodies_csv_uses_lf_on_windows(tmp_path, monkeypatch):
    # pandas' to_csv default line ending is os.linesep: CRLF under the Windows
    # .venv, while DEMDumpConvert.py (WSL) writes LF. Keep the bytes identical.
    import os
    monkeypatch.setattr(os, "linesep", "\r\n")
    run = tmp_path / "run_0001"
    shutil.copytree(FIX / "particle_np300", run)
    bodies = tmp_path / "bodies"
    frames = list(dfm.iter_grain_frames(run_dir=run, bodies_dir=bodies, max_frames=1))
    assert frames
    csv = sorted(bodies.glob("*.csv"))[0].read_bytes()
    assert b"\r" not in csv
