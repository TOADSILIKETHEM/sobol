"""viz_preprocess_dump.py: one read of each dump feeds composite + instance_grains (spec §3b)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._paths import CODE

sys.path.insert(0, str(CODE))
from viz import dump_frames as dfm  # noqa: E402
from viz import viz_preprocess as vp  # noqa: E402
from viz import viz_preprocess_grains_instance as vgi  # noqa: E402
from viz import viz_preprocess_dump as vpd  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
N_DUMPS = len(list((FIX / "particle_np300").glob("sobol_[0-9]*")))  # all usable


def _run(tmp_path):
    run = tmp_path / "phantom" / "batch" / "run_0001"
    shutil.copytree(FIX / "particle_np300", run)
    return run


def _layout(root: Path):
    b = root / "batch"
    return b / "run_0001_bodies_output", b / "run_0001_viz", b / "run_0001_viz_instance"


def _builders(root: Path, run: Path):
    bodies, viz, inst = _layout(root)
    return bodies, {
        "composite": vp.CompositeBuilder(viz, run_dir=run, bodies_dir=bodies),
        "instance_grains": vgi.InstanceBuilder(inst, run_dir=run, bodies_dir=bodies),
    }


def _assert_same_tree(a: Path, b: Path):
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert fa == fb
    for rel in fa:
        if rel.suffix == ".npz":
            xa, xb = np.load(a / rel), np.load(b / rel)
            assert set(xa.files) == set(xb.files), rel
            for k in xa.files:
                np.testing.assert_array_equal(xa[k], xb[k], err_msg=f"{rel}:{k}")
        else:
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_single_pass_matches_single_path_scripts(tmp_path):
    run = _run(tmp_path)
    bodies, builders = _builders(tmp_path / "one_pass", run)
    res = vpd.run_dump_preprocess(run, bodies, builders)
    assert list(res) == ["composite", "instance_grains"]
    assert all(isinstance(p, Path) and p.is_file() for p in res.values())

    rb, rv, ri = _layout(tmp_path / "per_path")
    vp.run_preprocess(None, rv, run_dir=run, bodies_dir=rb)
    vgi.build_grains_instance(None, rb, ri, run_dir=run)
    _, v, i = _layout(tmp_path / "one_pass")
    _assert_same_tree(rv, v)
    _assert_same_tree(ri, i)
    _assert_same_tree(rb, bodies)


def test_each_dump_read_once(tmp_path, monkeypatch):
    run = _run(tmp_path)
    calls = []
    real = dfm.frame_from_dump

    def counting(dump, *a, **k):
        calls.append(dump)
        return real(dump, *a, **k)
    monkeypatch.setattr(dfm, "frame_from_dump", counting)
    bodies, builders = _builders(tmp_path / "o", run)
    vpd.run_dump_preprocess(run, bodies, builders)
    assert len(calls) == N_DUMPS
    assert len(set(calls)) == N_DUMPS


def _fail_instance_on_second_frame(monkeypatch):
    real_add = vgi.InstanceBuilder.add

    def add(self, d):
        if len(self.frame_ids) == 1:
            raise vgi.GrainReappeared("grain 7 is absent at frame 00001 but present before and after it")
        real_add(self, d)
    monkeypatch.setattr(vgi.InstanceBuilder, "add", add)


def test_failing_builder_does_not_stop_the_other(tmp_path, monkeypatch):
    run = _run(tmp_path)
    rb, rv, _ = _layout(tmp_path / "ref")
    vp.run_preprocess(None, rv, run_dir=run, bodies_dir=rb)
    _fail_instance_on_second_frame(monkeypatch)
    bodies, builders = _builders(tmp_path / "o", run)
    res = vpd.run_dump_preprocess(run, bodies, builders)
    assert isinstance(res["instance_grains"], vgi.GrainReappeared)
    assert isinstance(res["composite"], Path)
    _assert_same_tree(rv, _layout(tmp_path / "o")[1])


def test_no_usable_dumps_fails_every_builder(tmp_path):
    run = tmp_path / "phantom" / "batch" / "run_0001"
    run.mkdir(parents=True)
    shutil.copy(FIX / "particle_np300" / "sobol.setup", run)
    bodies, builders = _builders(tmp_path / "o", run)
    res = vpd.run_dump_preprocess(run, bodies, builders)
    assert all(isinstance(e, FileNotFoundError) for e in res.values())


def _argv(tmp_path, run, *targets):
    bodies, viz, inst = _layout(tmp_path / "cli")
    argv = ["--run-dir", str(run), "--bodies-dir", str(bodies)]
    if "composite" in targets:
        argv += ["--composite-dir", str(viz)]
    if "instance_grains" in targets:
        argv += ["--instance-dir", str(inst)]
    return argv, viz, inst


def test_cli_ok(tmp_path, capsys):
    run = _run(tmp_path)
    argv, viz, inst = _argv(tmp_path, run, "composite", "instance_grains")
    assert vpd.main(argv) == 0
    out = capsys.readouterr().out
    assert f"OK composite -> {viz / 'manifest.json'}" in out
    assert f"OK instance_grains -> {inst / 'manifest.json'}" in out
    assert json.loads((viz / "manifest.json").read_text())["input_mode"] == "dump"


def test_cli_reports_failure_and_removes_stale_manifest(tmp_path, capsys, monkeypatch):
    run = _run(tmp_path)
    argv, viz, inst = _argv(tmp_path, run, "composite", "instance_grains")
    inst.mkdir(parents=True)
    (inst / "manifest.json").write_text('{"stale": true}')
    _fail_instance_on_second_frame(monkeypatch)
    assert vpd.main(argv) == 1
    out = capsys.readouterr().out
    assert "FAILED instance_grains: GrainReappeared: grain 7" in out
    assert "OK composite" in out
    assert not (inst / "manifest.json").exists()


def test_cli_needs_a_target(tmp_path, capsys):
    run = _run(tmp_path)
    argv, _, _ = _argv(tmp_path, run)
    with pytest.raises(SystemExit) as e:
        vpd.main(argv)
    assert e.value.code == 2
    assert "need --composite-dir and/or --instance-dir" in capsys.readouterr().err


def test_cli_grain_leaving_mid_run_reaches_instance_output(tmp_path, capsys, monkeypatch):
    # dump_frames.particle_grains drops particles with h <= 0 (e.g. accreted onto a
    # sink); emulate one leaving from the second dump on, through the real single pass.
    real = dfm.particle_grains
    calls = []

    def particle_grains(sdf_parts):
        g = real(sdf_parts)
        calls.append(len(g))
        if len(calls) > 1:
            g = g[g["iorig"] != g["iorig"].max()].reset_index(drop=True)
        return g
    monkeypatch.setattr(dfm, "particle_grains", particle_grains)

    run = _run(tmp_path)
    argv, viz, inst = _argv(tmp_path, run, "composite", "instance_grains")
    assert vpd.main(argv) == 0
    assert len(calls) == N_DUMPS

    m = json.loads((inst / "manifest.json").read_text())
    assert m["variable_grain_set"] is True and m["n_frames"] == N_DUMPS
    f = [np.load(inst / "points" / f"{fid}.npz") for fid in m["frame_ids"]]
    j = int(np.argmax(f[0]["grain_id"]))
    assert [bool(x["present"][j]) for x in f] == [True] + [False] * (N_DUMPS - 1)
    assert [int(x["present"].sum()) for x in f] == [calls[0]] + [calls[0] - 1] * (N_DUMPS - 1)
    assert json.loads((viz / "manifest.json").read_text())["n_frames"] == N_DUMPS
