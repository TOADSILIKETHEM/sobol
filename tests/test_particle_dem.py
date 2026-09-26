import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import particle_dem as pdem

FIX = Path(__file__).parent / "fixtures"
SINK_FIX = FIX / "sink_np300"
needs_sink_fixture = pytest.mark.skipif(
    not (SINK_FIX / "sobol_00000").is_file(),
    reason="sink_np300 fixture absent; regenerate per tests/fixtures/README.md",
)


def _write(tmp_path, text):
    p = tmp_path / "sobol.setup"
    p.write_text(text)
    return p


def test_dem_model_particle(tmp_path):
    p = _write(tmp_path, " np_apophis = 300\n use_dem = T\n use_dem_as_sinks = F\n")
    assert pdem.dem_model_from_setup(p) == "particle"


def test_dem_model_sink(tmp_path):
    p = _write(tmp_path, " np_apophis = 300\n use_dem = F\n use_dem_as_sinks = T\n")
    assert pdem.dem_model_from_setup(p) == "sink"


def test_dem_model_legacy_setup_is_sink(tmp_path):
    # pre-merge DEMsync batches: use_dem=T meant sink DEM, no use_dem_as_sinks key
    p = _write(tmp_path, " np_apophis = 500 ! n\n use_dem = T ! dem\n")
    assert pdem.dem_model_from_setup(p) == "sink"


def test_dem_model_point_mass(tmp_path):
    p = _write(tmp_path, " np_apophis = 1\n use_dem = T\n use_dem_as_sinks = F\n")
    assert pdem.dem_model_from_setup(p) == "none"


def test_list_full_dumps_excludes_tmp(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIX / "particle_np300", run)
    (run / "sobol_00000.tmp").write_bytes(b"")
    names = [d.name for d in pdem.list_full_dumps(run, "sobol")]
    assert names == sorted(names)
    assert "sobol_00000.tmp" not in names
    assert names[0] == "sobol_00000" and len(names) >= 4


def test_groups_from_particle_dumps_shape_and_mass():
    groups, tok, n = pdem.apophis_time_groups_from_dumps(FIX / "particle_np300", "sobol")
    assert n >= 290
    assert len(groups) == len(pdem.list_full_dumps(FIX / "particle_np300", "sobol"))
    for key, arr in groups.items():
        assert arr.shape == (n, 7)
        assert np.all(arr[:, 3] > 0)
        assert np.allclose(arr[:, 3], arr[0, 3])  # equal-mass grains
        assert tok[key] == pytest.approx(float(key), rel=1e-8)
    assert min(tok.values()) == pytest.approx(0.0, abs=1e-12)


@needs_sink_fixture
def test_particle_t0_matches_sink_t0():
    # same setup lattice -> identical initial grain positions in both models.
    # Sink .ev files have no t=0 row, so the sink side is read from its t=0 dump.
    gp, tp, _ = pdem.apophis_time_groups_from_dumps(FIX / "particle_np300", "sobol")
    p0 = gp[min(tp, key=tp.get)]
    _, sinks = pdem._read_dump(SINK_FIX / "sobol_00000")
    grains = sinks[sinks["Reff"] > 0]
    assert p0.shape[0] == len(grains)
    assert np.allclose(np.sort(p0[:, 0]), np.sort(grains["x"].to_numpy()), rtol=1e-12, atol=0)
    assert p0[:, 3].sum() == pytest.approx(grains["m"].sum(), rel=1e-9)


def test_frame_without_velocities_is_skipped(tmp_path, monkeypatch):
    run = tmp_path / "run"
    shutil.copytree(FIX / "particle_np300", run)

    real = pdem._read_dump

    def fake(path):
        sdf, sinks = real(path)
        if Path(path).name == "sobol_00001":
            sdf = sdf.drop(columns=["vx", "vy", "vz"])
        return sdf, sinks

    monkeypatch.setattr(pdem, "_read_dump", fake)
    groups, _, _ = pdem.apophis_time_groups_from_dumps(run, "sobol")
    assert len(groups) == len(pdem.list_full_dumps(run, "sobol")) - 1
