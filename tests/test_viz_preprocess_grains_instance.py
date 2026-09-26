"""Real-grain instance writer (viz/viz_preprocess_grains_instance.py)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests._paths import CODE
_spec = importlib.util.spec_from_file_location(
    "viz_preprocess_grains_instance", CODE / "viz" / "viz_preprocess_grains_instance.py"
)
vgi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vgi)


def _write_grain_npz(path: Path, grain_ids, offset=0.0):
    gid = np.asarray(grain_ids, dtype=np.int64)
    np.savez_compressed(
        path,
        grain_id=gid,
        x_vis=gid * 1.0 + offset,
        y_vis=gid * 2.0,
        z_vis=gid * 3.0,
        Reff_vis=0.5 + 0.1 * gid,
        time_s=np.float64(0.0),
    )


def _run_dirs(tmp_path):
    batch = tmp_path / "sim_x"
    grains = batch / "run_0001_grains_output"
    bodies = batch / "run_0001_bodies_output"
    out = batch / "run_0001_viz_instance"
    grains.mkdir(parents=True)
    bodies.mkdir(parents=True)
    return grains, bodies, out


def test_writes_points_sorted_by_grain_id(tmp_path):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [2, 0, 1], offset=0.0)
    _write_grain_npz(grains / "sobol_00001.npz", [1, 2, 0], offset=10.0)

    manifest_path = vgi.build_grains_instance(grains, bodies, out)

    f0 = np.load(out / "points" / "00000.npz")
    f1 = np.load(out / "points" / "00001.npz")
    assert f0["pos"].dtype == np.float32 and f0["pos"].shape == (3, 3)
    assert f0["radius"].dtype == np.float32 and f0["radius"].shape == (3,)
    assert np.allclose(f0["pos"][:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(f1["pos"][:, 0], [10.0, 11.0, 12.0])
    assert np.allclose(f0["pos"][:, 2], [0.0, 3.0, 6.0])
    assert np.allclose(f0["radius"], [0.5, 0.6, 0.7])
    assert manifest_path == out / "manifest.json"


def test_manifest_contents(tmp_path):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [0, 1, 2])
    _write_grain_npz(grains / "sobol_00001.npz", [0, 1, 2])

    m = json.loads(vgi.build_grains_instance(grains, bodies, out).read_text())

    assert m["schema_version"] == 1
    assert m["viz_mode"] == "instance"
    assert m["static"] is False
    assert m["source"] == "grains"
    assert m["sim_name"] == "sim_x"
    assert m["run_name"] == "run_0001"
    assert m["n_frames"] == 2
    assert m["frame_ids"] == ["00000", "00001"]
    assert m["points_dir"] == "points"
    assert m["n_points"] == 3
    assert m["grain_radius_vis"] == pytest.approx(0.6)
    assert m["bodies_csv_dir"] == "../run_0001_bodies_output"
    assert m["bodies_start_index"] == 0
    for key in ("synthetic", "motion_model", "angles_rad"):
        assert key not in m


def test_max_frames_truncates(tmp_path):
    grains, bodies, out = _run_dirs(tmp_path)
    for i in range(4):
        _write_grain_npz(grains / f"sobol_{i:05d}.npz", [0, 1])

    m = json.loads(vgi.build_grains_instance(grains, bodies, out, max_frames=2).read_text())

    assert m["n_frames"] == 2
    assert sorted(p.name for p in (out / "points").glob("*.npz")) == ["00000.npz", "00001.npz"]


def test_grain_count_mismatch_exits_1_without_manifest(tmp_path, capsys):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [0, 1, 2])
    _write_grain_npz(grains / "sobol_00001.npz", [0, 1])

    rc = vgi.main(["--grains-dir", str(grains), "--bodies-dir", str(bodies), "--output-dir", str(out)])

    assert rc == 1
    assert "frame 00001 has 2 grains, frame 00000 has 3" in capsys.readouterr().err
    assert not (out / "manifest.json").exists()


def test_empty_grains_dir_exits_1(tmp_path, capsys):
    grains, bodies, out = _run_dirs(tmp_path)
    rc = vgi.main(["--grains-dir", str(grains), "--bodies-dir", str(bodies), "--output-dir", str(out)])
    assert rc == 1
    assert "No grain npz files" in capsys.readouterr().err


def test_main_success_returns_0(tmp_path):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [0, 1])
    rc = vgi.main(["--grains-dir", str(grains), "--bodies-dir", str(bodies), "--output-dir", str(out),
                   "--sim-name", "S", "--run-name", "R"])
    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert (m["sim_name"], m["run_name"]) == ("S", "R")
