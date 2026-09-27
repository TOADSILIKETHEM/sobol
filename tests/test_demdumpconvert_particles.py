import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
from run_demtocsv_batch import run_dem_dump_convert

FIX = Path(__file__).parent / "fixtures"
CONVERT = Path("/home/mboyle/Honours/Code/CSVconvert/DEMDumpConvert.py")
needs_sink_fixture = pytest.mark.skipif(
    not (FIX / "sink_np300" / "sobol_00000").is_file(),
    reason="sink_np300 fixture absent; regenerate per tests/fixtures/README.md",
)


def _convert(tmp_path, name):
    run = tmp_path / "batch" / "run_0001"
    shutil.copytree(FIX / name, run)
    out = tmp_path / "out"
    run_dem_dump_convert(run, CONVERT, out, None)
    return out / "batch"


def test_particle_run_writes_grains_and_bodies(tmp_path):
    out = _convert(tmp_path, "particle_np300")
    npz = sorted((out / "run_0001_grains_output").glob("*.npz"))
    csv = sorted((out / "run_0001_bodies_output").glob("*.csv"))
    assert len(npz) >= 4 and len(csv) >= 4
    g = np.load(npz[0])
    assert len(g["x_vis"]) >= 290
    assert np.all(g["Reff_vis"] > 0)
    import pandas as pd
    bodies = pd.read_csv(csv[0])
    assert "Earth" in set(bodies["name"])
    # 10 solar-system sinks + synthetic Apophis CoM row (row 10); no grains among bodies
    assert len(bodies) == 11 and bodies["name"].iloc[-1] == "Apophis"


@needs_sink_fixture
def test_particle_and_sink_grain_layout_match(tmp_path):
    gp = np.load(sorted((_convert(tmp_path / "p", "particle_np300") / "run_0001_grains_output").glob("*.npz"))[0])
    gs = np.load(sorted((_convert(tmp_path / "s", "sink_np300") / "run_0001_grains_output").glob("*.npz"))[0])
    assert set(gp.files) == set(gs.files)
    assert len(gp["x_vis"]) == len(gs["x_vis"])


SOBOL = Path(__file__).parent.parent


@pytest.mark.skipif(not (SOBOL / "phantom").is_file(), reason="needs built sobol/phantom")
def test_apophis_only_particle_run_converts(tmp_path):
    import os
    import subprocess
    run = tmp_path / "batch" / "run_0001"
    env = dict(os.environ, APOPHIS_ONLY="T")
    subprocess.run([str(FIX / "dem_run_dir.sh"), "particle", str(run), "300", "20 min", "10 min", "-1", "run"],
                   check=True, env=env, capture_output=True)
    out = tmp_path / "out"
    run_dem_dump_convert(run, CONVERT, out, None)
    npz = sorted((out / "batch" / "run_0001_grains_output").glob("*.npz"))
    assert len(npz) >= 2
    assert len(np.load(npz[0])["x_vis"]) >= 290


@needs_sink_fixture
def test_sink_run_without_setup_file_still_converts(tmp_path):
    run = tmp_path / "batch" / "run_0001"
    shutil.copytree(FIX / "sink_np300", run)
    (run / "sobol.setup").unlink()
    out = tmp_path / "out"
    # no setup -> legacy MIN_DEM_GRAINS default 450 > 307 fixture grains; pass it explicitly
    run_dem_dump_convert(run, CONVERT, out, 250)
    assert len(sorted((out / "batch" / "run_0001_grains_output").glob("*.npz"))) >= 4


def test_flyby_run_bodies_are_earth_and_apophis(tmp_path):
    import pandas as pd
    out = _convert(tmp_path, "particle_flyby_np300")
    csvs = sorted((out / "run_0001_bodies_output").glob("*.csv"))
    assert csvs
    for c in csvs:
        assert list(pd.read_csv(c)["name"]) == ["Earth", "Apophis"]
    g = np.load(sorted((out / "run_0001_grains_output").glob("*.npz"))[0])
    assert len(g["x_vis"]) >= 250
