"""sobol/render_dumps.py: render existing PHANTOM dumps without the sim stage."""
import errno
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import render_dumps as rd  # noqa: E402


def _dumps(d: Path, prefix="sobol", n=3):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{prefix}_{i:05d}").write_bytes(b"x")
    return d


def test_ready_run_dir_is_used_in_place(tmp_path):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")
    assert rd.stage_run(run, tmp_path / "stage", "sobol", None, None, None) == run


def test_missing_setup_without_dem_model_refuses(tmp_path):
    run = _dumps(tmp_path / "run")
    with pytest.raises(SystemExit, match="--dem-model"):
        rd.stage_run(run, tmp_path / "stage", "sobol", None, None, None)


def test_dem_model_without_np_apophis_refuses(tmp_path):
    run = _dumps(tmp_path / "run")
    with pytest.raises(SystemExit, match="--np-apophis"):
        rd.stage_run(run, tmp_path / "stage", "sobol", "particle", True, None)


def test_other_prefix_is_renamed_and_setup_written(tmp_path):
    run = _dumps(tmp_path / "run", prefix="apophis", n=2)
    out = rd.stage_run(run, tmp_path / "stage", "apophis", "particle", True, 32530)
    assert sorted(p.name for p in out.glob("sobol_*")) == ["sobol_00000", "sobol_00001"]
    setup = (out / "sobol.setup").read_text()
    for line in ("apophis_only = T", "np_apophis = 32530", "use_dem = T", "use_dem_as_sinks = F"):
        assert line in setup


def test_sink_model_writes_sink_flag(tmp_path):
    out = rd.stage_run(_dumps(tmp_path / "run"), tmp_path / "stage", "sobol", "sink", False, 500)
    assert "use_dem_as_sinks = T" in (out / "sobol.setup").read_text()
    assert "apophis_only = F" in (out / "sobol.setup").read_text()


def test_restaging_drops_stale_dumps(tmp_path):
    stage = tmp_path / "stage"
    rd.stage_run(_dumps(tmp_path / "a", n=3), stage, "sobol", "particle", True, 10)
    out = rd.stage_run(_dumps(tmp_path / "b", n=1), stage, "sobol", "particle", True, 10)
    assert [p.name for p in out.glob("sobol_[0-9]*")] == ["sobol_00000"]


def test_encounter_json_is_carried(tmp_path):
    run = _dumps(tmp_path / "run")
    (run / "encounter.json").write_text('{"mode": "hyperbola"}')
    out = rd.stage_run(run, tmp_path / "stage", "sobol", "particle", False, 10)
    assert (out / "encounter.json").read_text() == '{"mode": "hyperbola"}'


def test_no_dumps_is_a_clear_error(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit, match=r"no apophis_\[0-9\]\* dumps"):
        rd.stage_run(tmp_path / "run", tmp_path / "stage", "apophis", "particle", True, 10)


def test_link_falls_back_across_filesystems(tmp_path, monkeypatch):
    src = tmp_path / "a"
    src.write_bytes(b"x")

    def exdev(*a, **k):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(rd.os, "link", exdev)
    assert rd.link_or_copy(src, tmp_path / "b") in ("symlink", "copy")
    assert (tmp_path / "b").read_bytes() == b"x"


def test_npz_paths_rejected(tmp_path):
    with pytest.raises(SystemExit, match="tui_sim_render"):
        rd.main([str(tmp_path), "--paths", "per_sphere"])


def test_main_wires_preprocess_and_render(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")
    seen = {}

    def fake_pre(cmd, targets, on_line=None):
        seen["pre"] = (cmd, targets)
        return dict(targets)

    class Result:
        ok = True

    def fake_render(spec, ctx, on_line=None):
        seen["ren"] = ctx
        return Result

    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)
    monkeypatch.setattr(rd.t, "run_dump_preprocess_stage", fake_pre)
    monkeypatch.setattr(rd.t, "run_render_path", fake_render)
    monkeypatch.setattr(rd.t, "format_path_result", lambda r: "ok")
    rc = rd.main([str(run), "--name", "demo", "--out-root", str(tmp_path / "out"),
                  "--samples", "8", "--max-frames", "2", "--fps", "8"])
    assert rc == 0
    ctx = seen["ren"]
    assert ctx.render_form.samples == 8 and ctx.render_form.max_frames == 2
    assert ctx.n_expected_frames == 3
    assert ctx.batch_dir == tmp_path / "out" / "demo"
    assert seen["pre"][1] == {"instance_grains": tmp_path / "out" / "demo" / "run_0001_viz_instance"}


def test_main_returns_1_when_a_render_fails(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")

    class Bad:
        ok = False

    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)
    monkeypatch.setattr(rd.t, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: dict(targets))
    monkeypatch.setattr(rd.t, "run_render_path", lambda spec, ctx, on_line=None: Bad)
    monkeypatch.setattr(rd.t, "format_path_result", lambda r: "bad")
    assert rd.main([str(run), "--out-root", str(tmp_path / "out")]) == 1


def test_no_symlink_when_windows_tool_reads_the_dumps(tmp_path, monkeypatch):
    # WSL symlinks are unreadable from Windows python.exe: copy instead.
    src = tmp_path / "a"
    src.write_bytes(b"x")

    def exdev(*a, **k):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(rd.os, "link", exdev)
    assert rd.link_or_copy(src, tmp_path / "b", allow_symlink=False) == "copy"
    assert not (tmp_path / "b").is_symlink()


def test_stage_dir_option_and_windows_python_disables_symlinks(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "run", prefix="body", n=2)
    seen = {}
    monkeypatch.setattr(rd, "stage_run", lambda src, stage_dir, *a, **k: seen.update(stage=stage_dir, **k) or run)
    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)
    monkeypatch.setattr(rd.t, "WIN_VENV_PYTHON", Path("/mnt/c/x/python.exe"))
    monkeypatch.setattr(rd.t, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: dict(targets))

    class Ok:
        ok = True

    monkeypatch.setattr(rd.t, "run_render_path", lambda spec, ctx, on_line=None: Ok)
    monkeypatch.setattr(rd.t, "format_path_result", lambda r: "ok")
    rd.main([str(run), "--out-root", str(tmp_path / "out"), "--stage-dir", str(tmp_path / "stg")])
    assert seen["stage"] == tmp_path / "stg"
    assert seen["allow_symlink"] is False


# -- final-review fixes (2026-10-10) --

def _fake_stages(monkeypatch, seen):
    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)

    def pre(cmd, targets, on_line=None):
        seen["pre_cmd"] = cmd
        return dict(targets)

    class Ok:
        ok = True

    def ren(spec, ctx, on_line=None):
        seen["ctx"] = ctx
        return Ok

    monkeypatch.setattr(rd.t, "run_dump_preprocess_stage", pre)
    monkeypatch.setattr(rd.t, "run_render_path", ren)
    monkeypatch.setattr(rd.t, "format_path_result", lambda r: "ok")


def test_relative_out_root_is_made_absolute(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")
    seen = {}
    _fake_stages(monkeypatch, seen)
    monkeypatch.chdir(tmp_path)
    rd.main([str(run), "--name", "demo", "--out-root", "out"])
    assert seen["ctx"].batch_dir == tmp_path / "out" / "demo"
    assert seen["ctx"].batch_dir.is_absolute()


def test_staging_never_touches_the_source_or_a_neighbouring_run(tmp_path):
    # --stage-dir = the dumps' own parent (as the README suggests) must not
    # overwrite them, nor a real run_0001 already sitting there.
    run = _dumps(tmp_path / "run_0001", prefix="body")
    other = _dumps(tmp_path / "batch" / "run_0001")
    out = rd.stage_run(run, tmp_path, "body", "particle", True, 10)
    assert out != run and out.parent.name == "_staged_run_0001"
    assert sorted(p.name for p in run.iterdir()) == ["body_00000", "body_00001", "body_00002"]
    rd.stage_run(run, tmp_path / "batch", "body", "particle", True, 10)
    assert sorted(p.name for p in other.iterdir()) == ["sobol_00000", "sobol_00001", "sobol_00002"]


def test_staging_refuses_a_folder_it_did_not_make(tmp_path):
    run = _dumps(tmp_path / "fb40", prefix="body")
    (tmp_path / "_staged_fb40").mkdir()
    (tmp_path / "_staged_fb40" / "keep.txt").write_text("user data")
    with pytest.raises(SystemExit, match="not made by render_dumps"):
        rd.stage_run(run, tmp_path, "body", "particle", True, 10)
    assert (tmp_path / "_staged_fb40" / "keep.txt").exists()


def test_default_stage_is_beside_the_dumps_and_removed_after_preprocess(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "mia" / "fb40", prefix="body")
    seen = {}
    _fake_stages(monkeypatch, seen)
    rd.main([str(run), "--prefix", "body", "--dem-model", "particle", "--np-apophis", "10",
             "--apophis-only", "--out-root", str(tmp_path / "out")])
    staged_in = [a for a in seen["pre_cmd"] if "_staged_fb40" in str(a)]
    assert staged_in, seen["pre_cmd"]
    assert not (tmp_path / "mia" / "_staged_fb40").exists()  # cleaned up
    assert sorted(p.name for p in run.iterdir()) == ["body_00000", "body_00001", "body_00002"]


def test_output_run_name_follows_source_run_dir(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "batch" / "run_0007")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")
    seen = {}
    _fake_stages(monkeypatch, seen)
    rd.main([str(run), "--out-root", str(tmp_path / "out")])
    assert seen["ctx"].run_name == "run_0007"
    assert (tmp_path / "out" / "batch" / "run_0007_viz_instance").is_dir()


@pytest.mark.parametrize("flag", [["--np-apophis", "40000"], ["--apophis-only"]])
def test_setup_flags_without_dem_model_are_refused(flag, capsys):
    # Without --dem-model the run's own sobol.setup is used, so these would do nothing.
    with pytest.raises(SystemExit):
        rd._parse(["run_0003", *flag])
    assert "--dem-model" in capsys.readouterr().err


def test_slowmo_and_resume_reach_the_render_form(tmp_path, monkeypatch):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")
    seen = {}

    class Result:
        ok = True

    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)
    monkeypatch.setattr(rd.t, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: dict(targets))
    monkeypatch.setattr(rd.t, "run_render_path", lambda spec, ctx, on_line=None: seen.setdefault("ctx", ctx) and Result)
    monkeypatch.setattr(rd.t, "format_path_result", lambda r: "ok")
    assert rd.main([str(run), "--out-root", str(tmp_path / "out"), "--slowmo", "4", "--resume"]) == 0
    assert seen["ctx"].render_form.slowmo == 4 and seen["ctx"].render_form.resume is True


def test_slowmo_without_an_instance_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)
    with pytest.raises(SystemExit, match="instance"):
        rd.main([str(_dumps(tmp_path / "run")), "--paths", "composite", "--slowmo", "4"])


def test_samples_default_is_124():
    assert rd._parse(["run"]).samples == 124
