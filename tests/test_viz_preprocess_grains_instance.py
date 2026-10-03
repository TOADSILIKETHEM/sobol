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


def test_rotations_match_per_sphere_draw(tmp_path):
    # DEMGrainsBlenderEarthCam.py draws one Euler triple per grain in frame 0's
    # stored order; the sorted points must carry each grain's own triple.
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [2, 0, 1])
    _write_grain_npz(grains / "sobol_00001.npz", [0, 1, 2])

    vgi.build_grains_instance(grains, bodies, out, rotation_seed=0)

    per_sphere = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, size=(3, 3))
    by_gid = {2: per_sphere[0], 0: per_sphere[1], 1: per_sphere[2]}
    for fid in ("00000", "00001"):
        f = np.load(out / "points" / f"{fid}.npz")
        assert f["rot"].dtype == np.float32
        assert list(f["grain_id"]) == [0, 1, 2]
        for i, gid in enumerate(f["grain_id"]):
            assert np.allclose(f["rot"][i], by_gid[int(gid)])


def test_handles_follow_fcurve_solve(tmp_path):
    from viz.fcurve_handles import right_handle_offsets

    grains, bodies, out = _run_dirs(tmp_path)
    for i, off in enumerate([0.0, 1.0, 5.0, 4.0]):
        _write_grain_npz(grains / f"sobol_{i:05d}.npz", [1, 0], offset=off)

    m = json.loads(vgi.build_grains_instance(grains, bodies, out).read_text())

    pos = np.stack([np.load(out / "points" / f"{fid}.npz")["pos"] for fid in m["frame_ids"]])
    h = np.stack([np.load(out / "points" / f"{fid}.npz")["handle"] for fid in m["frame_ids"]])
    assert np.allclose(h, right_handle_offsets(pos.astype(np.float64)), atol=1e-6)
    assert m["interpolation"] == {"mode": "bezier", "auto_smoothing": "CONT_ACCEL", "handle_key": "handle"}
    assert m["rotation_seed"] == 0


def test_fixed_set_output_unchanged(tmp_path):
    from viz.fcurve_handles import right_handle_offsets

    grains, bodies, out = _run_dirs(tmp_path)
    for i, off in enumerate([0.0, 1.0, 5.0, 4.0]):
        _write_grain_npz(grains / f"sobol_{i:05d}.npz", [1, 0, 2], offset=off)

    m = json.loads(vgi.build_grains_instance(grains, bodies, out).read_text())

    assert m["n_points"] == 3
    assert m["variable_grain_set"] is False
    f = [np.load(out / "points" / f"{fid}.npz") for fid in m["frame_ids"]]
    for x in f:
        assert x["present"].dtype == bool and x["present"].all()
        assert x["grain_id"].dtype == np.int64
        assert np.array_equal(x["grain_id"], [0, 1, 2])
    # Same values as before variable grain sets: sorted positions, one handle
    # solve over the whole run, frame-0 radius, rotations drawn in stored order.
    ids = np.array([0, 1, 2])
    want_pos = np.stack([np.column_stack([ids + off, ids * 2.0, ids * 3.0])
                         for off in [0.0, 1.0, 5.0, 4.0]]).astype(np.float32)
    assert np.array_equal(np.stack([x["pos"] for x in f]), want_pos)
    want_h = right_handle_offsets(want_pos.astype(np.float64)).astype(np.float32)
    assert np.array_equal(np.stack([x["handle"] for x in f]), want_h)
    assert np.array_equal(f[0]["radius"], np.float32(0.5 + 0.1 * ids))
    draw = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, size=(3, 3)).astype(np.float32)
    assert np.array_equal(f[0]["rot"], draw[[1, 0, 2]])  # stored order was [1, 0, 2]


def test_grain_leaving_is_hidden_and_held(tmp_path):
    from viz.fcurve_handles import evaluate, right_handle_offsets

    grains, bodies, out = _run_dirs(tmp_path)
    for i, off in enumerate([0.0, 1.0, 5.0, 4.0, 2.0]):
        ids = [0, 1, 2] if i < 3 else [0, 1]
        _write_grain_npz(grains / f"sobol_{i:05d}.npz", ids, offset=off)

    m = json.loads(vgi.build_grains_instance(grains, bodies, out).read_text())

    f = [np.load(out / "points" / f"{fid}.npz") for fid in m["frame_ids"]]
    assert m["n_points"] == 3 and m["variable_grain_set"] is True
    assert [bool(x["present"][2]) for x in f] == [True, True, True, False, False]
    assert all(x["present"][:2].all() for x in f)
    # Grain 2 rests at its last present position once it has left.
    assert np.array_equal(f[3]["pos"][2], f[2]["pos"][2])
    assert np.array_equal(f[4]["pos"][2], f[2]["pos"][2])
    # Its handles are the solve over its own three keys, zero after.
    own = right_handle_offsets(np.stack([x["pos"][2:3] for x in f[:3]]).astype(np.float64))
    assert np.allclose(np.stack([x["handle"][2:3] for x in f[:3]]), own, atol=1e-6)
    assert not f[3]["handle"][2].any() and not f[4]["handle"][2].any()
    # The curve from its last key to the next frame stays at the last key.
    mid = evaluate(f[2]["pos"][2], f[2]["handle"][2], f[3]["pos"][2], f[3]["handle"][2], 0.5)
    assert np.allclose(mid, f[2]["pos"][2])
    # Grains 0 and 1 (present throughout) keep the whole-run solve.
    whole = right_handle_offsets(np.stack([x["pos"][:2] for x in f]).astype(np.float64))
    assert np.allclose(np.stack([x["handle"][:2] for x in f]), whole, atol=1e-6)


def test_grain_joining_late(tmp_path):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [2, 0], offset=0.0)
    _write_grain_npz(grains / "sobol_00001.npz", [1, 2, 0], offset=1.0)
    _write_grain_npz(grains / "sobol_00002.npz", [0, 1, 2], offset=3.0)

    m = json.loads(vgi.build_grains_instance(grains, bodies, out, rotation_seed=0).read_text())

    f = [np.load(out / "points" / f"{fid}.npz") for fid in m["frame_ids"]]
    assert list(f[0]["grain_id"]) == [0, 1, 2]
    assert [bool(x["present"][1]) for x in f] == [False, True, True]
    # Before joining it rests where it first appears.
    assert np.array_equal(f[0]["pos"][1], f[1]["pos"][1])
    assert not f[0]["handle"][1].any()
    # Frame 0 draws for 2 then 0 (stored order); grain 1 draws next, in frame 1.
    r = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, size=(3, 3))
    by_gid = {2: r[0], 0: r[1], 1: r[2]}
    for i, gid in enumerate(f[0]["grain_id"]):
        assert np.allclose(f[0]["rot"][i], by_gid[int(gid)])
    assert np.allclose(f[0]["radius"], [0.5, 0.6, 0.7])


def test_grain_set_change_renders(tmp_path):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [0, 1, 2])
    _write_grain_npz(grains / "sobol_00001.npz", [0, 1, 3])

    rc = vgi.main(["--grains-dir", str(grains), "--bodies-dir", str(bodies), "--output-dir", str(out)])

    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_points"] == 4
    f1 = np.load(out / "points" / "00001.npz")
    assert list(f1["grain_id"]) == [0, 1, 2, 3]
    assert list(f1["present"]) == [True, True, False, True]


def test_grain_reappearing_exits_1_without_manifest(tmp_path, capsys):
    grains, bodies, out = _run_dirs(tmp_path)
    _write_grain_npz(grains / "sobol_00000.npz", [0, 1])
    _write_grain_npz(grains / "sobol_00001.npz", [0])
    _write_grain_npz(grains / "sobol_00002.npz", [0, 1])

    rc = vgi.main(["--grains-dir", str(grains), "--bodies-dir", str(bodies), "--output-dir", str(out)])

    assert rc == 1
    assert "grain 1 is absent at frame 00001 but present before and after it" in capsys.readouterr().err
    assert not (out / "manifest.json").exists()


# --- --input-mode dump (read PHANTOM dumps directly, no grains npz) ---------

import shutil  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
from run_demtocsv_batch import DEFAULT_DEM_DUMP_CONVERT, run_dem_dump_convert  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _fixture_run(tmp_path, name="particle_np300"):
    run = tmp_path / "phantom" / "batch" / "run_0001"
    shutil.copytree(FIX / name, run)
    return run


def _both_routes(tmp_path, **kw):
    run = _fixture_run(tmp_path)
    out = tmp_path / "npz_route"
    run_dem_dump_convert(run, DEFAULT_DEM_DUMP_CONVERT, out, None)
    a = vgi.build_grains_instance(
        out / "batch" / "run_0001_grains_output", out / "batch" / "run_0001_bodies_output",
        out / "batch" / "run_0001_viz_instance", **kw)
    d = tmp_path / "dump_route" / "batch"
    b = vgi.build_grains_instance(
        None, d / "run_0001_bodies_output", d / "run_0001_viz_instance", run_dir=run, **kw)
    return a, b


def test_dump_mode_matches_npz_mode(tmp_path):
    a, b = _both_routes(tmp_path)
    ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
    assert ma.pop("input_mode") == "npz" and mb.pop("input_mode") == "dump"
    assert ma == mb
    pa = sorted((a.parent / "points").glob("*.npz"))
    pb = sorted((b.parent / "points").glob("*.npz"))
    assert [p.name for p in pa] == [p.name for p in pb] and pa
    for x, y in zip(pa, pb):
        dx, dy = np.load(x), np.load(y)
        assert set(dx.files) == set(dy.files)
        for k in dx.files:
            np.testing.assert_array_equal(dx[k], dy[k], err_msg=f"{x.name}:{k}")
    ca = sorted((a.parent.parent / "run_0001_bodies_output").glob("*.csv"))
    cb = sorted((b.parent.parent / "run_0001_bodies_output").glob("*.csv"))
    assert [c.name for c in ca] == [c.name for c in cb]
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(ca, cb))
    assert not (b.parent.parent / "run_0001_grains_output").exists()


def test_dump_route_variable_set_matches_npz(tmp_path, monkeypatch):
    # Drop the highest grain id from every frame after the first, in both routes,
    # so the dump route's builder sees the same changing set as the npz route.
    real_add = vgi.InstanceBuilder.add

    def add(self, d):
        if self.frame_ids:
            keep = np.asarray(d["grain_id"]) != np.asarray(d["grain_id"]).max()
            d = {k: (np.asarray(v)[keep] if np.ndim(v) and len(v) == len(keep) else v)
                 for k, v in d.items()}
        real_add(self, d)
    monkeypatch.setattr(vgi.InstanceBuilder, "add", add)

    a, b = _both_routes(tmp_path)
    ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
    assert ma["variable_grain_set"] is True and mb["variable_grain_set"] is True
    for x, y in zip(sorted((a.parent / "points").glob("*.npz")), sorted((b.parent / "points").glob("*.npz"))):
        dx, dy = np.load(x), np.load(y)
        for k in dx.files:
            np.testing.assert_array_equal(dx[k], dy[k], err_msg=f"{x.name}:{k}")
    assert not np.load(sorted((a.parent / "points").glob("*.npz"))[-1])["present"].all()


def test_dump_mode_clears_stale_bodies_csv(tmp_path):
    run = _fixture_run(tmp_path)
    bodies = tmp_path / "b" / "run_0001_bodies_output"
    bodies.mkdir(parents=True)
    (bodies / "sobol_99999.csv").write_text("stale")
    vgi.build_grains_instance(None, bodies, tmp_path / "b" / "viz", run_dir=run)
    names = sorted(p.name for p in bodies.glob("*.csv"))
    assert "sobol_99999.csv" not in names and names


def test_dump_mode_max_frames_reads_only_needed_dumps(tmp_path, monkeypatch):
    import viz.dump_frames as dfm
    calls = []
    real = dfm.frame_from_dump

    def counting(*a, **kw):
        calls.append(a[0])
        return real(*a, **kw)
    monkeypatch.setattr(dfm, "frame_from_dump", counting)
    run = _fixture_run(tmp_path)
    m = vgi.build_grains_instance(None, tmp_path / "b", tmp_path / "viz", run_dir=run, max_frames=2)
    assert json.loads(m.read_text())["n_frames"] == 2
    assert len(calls) == 2


def test_dump_mode_no_dumps_exits_1(tmp_path, capsys):
    empty = tmp_path / "run_0001"
    empty.mkdir()
    rc = vgi.main(["--input-mode", "dump", "--run-dir", str(empty),
                   "--bodies-dir", str(tmp_path / "b"), "--output-dir", str(tmp_path / "viz")])
    assert rc == 1
    assert "No usable DEM dumps" in capsys.readouterr().err
    assert not (tmp_path / "viz" / "manifest.json").exists()


@pytest.mark.parametrize("args,msg", [
    (["--input-mode", "dump"], "--input-mode dump needs --run-dir"),
    (["--input-mode", "npz"], "--input-mode npz needs --grains-dir"),
])
def test_cli_needs_the_dir_for_its_mode(tmp_path, capsys, args, msg):
    with pytest.raises(SystemExit) as exc:
        vgi.main(args + ["--bodies-dir", str(tmp_path / "b"), "--output-dir", str(tmp_path / "v")])
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err
