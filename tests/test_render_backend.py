"""render_backend: lazy tui_sim_render import and render_run on an existing sobol run dir."""
import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import render_backend as rb

T = rb.load()
needs_t = pytest.mark.skipif(isinstance(T, str), reason=f"render stages unavailable here: {T}")


def test_load_turns_import_exit_into_reason(monkeypatch):
    def boom(name):
        raise SystemExit("Code/local_paths.json missing")

    monkeypatch.setattr(importlib, "import_module", boom)
    reason = rb.load()
    assert isinstance(reason, str) and "local_paths.json" in reason


def test_is_sobol_run(tmp_path):
    assert not rb.is_sobol_run(tmp_path)
    (tmp_path / "sobol.setup").write_text("")
    (tmp_path / "sobol_00000").write_text("")
    assert rb.is_sobol_run(tmp_path)


@needs_t
def test_form_dict_round_trip():
    form = T.RenderFormValues(paths=("composite", "instance_grains"), slowmo=1, compare=True)
    assert rb.form_from_dict(T, rb.form_to_dict(form)) == form


@needs_t
def test_render_run_dump_route(tmp_path, monkeypatch):
    run = tmp_path / "batch" / "run_0003"
    run.mkdir(parents=True)
    (run / "sobol.setup").write_text("")
    for i in range(3):
        (run / f"sobol_{i:05d}").write_text("")
    calls = []
    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", lambda spec, ctx, **k: calls.append((spec.name, ctx.n_expected_frames))
                        or T.PathResult(spec.name, True, "done", "ok", n_frames=3))
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: calls.append(("cleanup", d)))
    res = rb.render_run(T, run, T.RenderFormValues(paths=("instance_grains",)), base_output_dir=tmp_path / "out")
    assert res.ok and res.stage == "done"
    assert calls == [("instance_grains", 3), ("cleanup", run)]


@needs_t
def test_render_run_keeps_dumps_when_asked(tmp_path, monkeypatch):
    run = tmp_path / "batch" / "run_0001"
    run.mkdir(parents=True)
    (run / "sobol_00000").write_text("")
    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", lambda spec, ctx, **k: T.PathResult(spec.name, True, "done", "ok"))
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: pytest.fail("dumps deleted"))
    assert rb.render_run(T, run, T.RenderFormValues(), keep_dumps=True, base_output_dir=tmp_path).ok


@needs_t
def test_render_run_without_dumps_fails_cleanly(tmp_path):
    run = tmp_path / "batch" / "run_0001"
    run.mkdir(parents=True)
    res = rb.render_run(T, run, T.RenderFormValues(), base_output_dir=tmp_path)
    assert not res.ok and res.stage == "convert" and "no sobol_" in res.message


@needs_t
def test_render_dumps_argv():
    form = T.RenderFormValues(paths=("instance_grains",), samples=64, encode_video=True, slowmo=4, resume=True)
    argv = rb.render_dumps_argv(Path("/r/fly"), form, name="mia", prefix="fly", dem_model="particle",
                                np_apophis=40000, apophis_only=True, encounter="hyperbola")
    assert argv[:1] == ["/r/fly"]
    for pair in (["--prefix", "fly"], ["--name", "mia"], ["--np-apophis", "40000"], ["--samples", "64"],
                 ["--slowmo", "4"], ["--encounter", "hyperbola"]):
        i = argv.index(pair[0])
        assert argv[i:i + 2] == pair
    assert "--video" in argv and "--resume" in argv and "--apophis-only" in argv


@needs_t
def test_render_dumps_argv_passes_camera_envelope_and_video_fps():
    form = T.RenderFormValues(paths=("composite",), camera_modes=("grain_only",), envelope_method="sdf",
                              encode_video=True, video_fps=12)
    argv = rb.render_dumps_argv(Path("/r"), form)
    for pair in (["--camera-mode", "grain_only"], ["--envelope-method", "sdf"], ["--video-fps", "12"]):
        i = argv.index(pair[0])
        assert argv[i:i + 2] == pair
    defaults = rb.render_dumps_argv(Path("/r"), T.RenderFormValues(paths=("composite",)))
    assert not {"--camera-mode", "--envelope-method", "--video-fps"} & set(defaults)


@needs_t
def test_render_dumps_argv_refuses_npz_paths():
    with pytest.raises(ValueError, match="instance_grains / composite"):
        rb.render_dumps_argv(Path("/r"), T.RenderFormValues(paths=("instance_static",)))


@needs_t
def test_render_run_reports_progress(tmp_path, monkeypatch):
    run = tmp_path / "batch" / "run_0003"
    run.mkdir(parents=True)
    (run / "sobol.setup").write_text("")
    for i in range(3):
        (run / f"sobol_{i:05d}").write_text("")
    frames = tmp_path / "frames"

    def render_path(spec, ctx, on_stage=None, index=1, total=1, **k):
        on_stage("render", path=spec.name, index=index, total=total, n_expected=3, output_dir=frames)
        return T.PathResult(spec.name, True, "done", "ok", n_frames=3)

    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", render_path)
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: None)
    seen = []
    rb.render_run(T, run, T.RenderFormValues(paths=("instance_grains",)), base_output_dir=tmp_path / "out",
                  on_progress=seen.append)
    since = seen[1].pop("since")  # not resume: frames already in the folder are not this render's
    assert abs(since - time.time()) < 60
    assert seen == [{"phase": "preprocess"},
                    {"phase": "render", "path": "instance_grains", "index": 1, "total": 1, "n_expected": 3,
                     "dir": str(frames), "glob": "frame_*.png", "log": str(frames / "render.log")}]


@needs_t
def test_render_run_progress_counts_kept_frames_on_resume(tmp_path, monkeypatch):
    run = tmp_path / "batch" / "run_0003"
    run.mkdir(parents=True)
    (run / "sobol.setup").write_text("")
    (run / "sobol_00000").write_text("")

    def render_path(spec, ctx, on_stage=None, index=1, total=1, **k):
        on_stage("render", path=spec.name, index=index, total=total, n_expected=1, output_dir=tmp_path / "f")
        return T.PathResult(spec.name, True, "done", "ok", n_frames=1)

    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", render_path)
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: None)
    seen = []
    rb.render_run(T, run, T.RenderFormValues(paths=("instance_grains",), resume=True),
                  base_output_dir=tmp_path / "out", on_progress=seen.append)
    assert "since" not in seen[1]  # resume keeps earlier frames, so they count


@needs_t
def test_render_run_writes_pipeline_timing(tmp_path, monkeypatch):
    import json
    run = tmp_path / "batch" / "run_0002"
    run.mkdir(parents=True)
    (run / "sobol.setup").write_text("np_apophis = 300\n")
    for i in range(2):
        (run / f"sobol_{i:05d}").write_text("")
    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", lambda spec, ctx, **k: T.PathResult(
        spec.name, True, "done", "ok", n_frames=2, render_s=8.0, elapsed_s=8.0))
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: None)
    res = rb.render_run(T, run, T.RenderFormValues(paths=("instance_grains",)), base_output_dir=tmp_path / "out")
    rep = json.loads((run / "pipeline_timing.json").read_text())
    assert res.timing_path == run / "pipeline_timing.json"
    assert rep["counts"]["np_apophis"] == 300 and rep["counts"]["n_dumps"] == 2
    assert rep["paths"][0]["name"] == "instance_grains" and rep["paths"][0]["render_s_per_frame"] == 4.0
    assert rep["wall_clock_s"]["sim"] is None and rep["wall_clock_s"]["dump_preprocess"] >= 0
    assert rep["render"]["paths"] == ["instance_grains"] and rep["outcome"] == {"stage": "done", "ok": True}
    assert "sobol" in rep["git"]


@needs_t
def test_rerender_keeps_sim_time_from_an_earlier_timing_file(tmp_path, monkeypatch):
    import json
    run = tmp_path / "batch" / "run_0001"
    run.mkdir(parents=True)
    (run / "sobol.setup").write_text("")
    (run / "sobol_00000").write_text("")
    (run / "pipeline_timing.json").write_text(json.dumps({"wall_clock_s": {"sim": 812.5}, "sim": {"np_apophis": 9}}))
    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", lambda spec, ctx, **k: T.PathResult(spec.name, True, "done", "ok"))
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: None)
    rb.render_run(T, run, T.RenderFormValues(), base_output_dir=tmp_path / "out")
    rep = json.loads((run / "pipeline_timing.json").read_text())
    assert rep["wall_clock_s"]["sim"] == 812.5 and rep["sim"] == {"np_apophis": 9}


@needs_t
def test_render_dumps_argv_passes_every_camera_and_fov():
    form = T.RenderFormValues(paths=("instance_grains",), camera_modes=("auto", "earth_chase"), camera_fov=25.0)
    argv = rb.render_dumps_argv(Path("/r"), form)
    i = argv.index("--camera-mode")
    assert argv[i:i + 3] == ["--camera-mode", "auto", "earth_chase"]
    j = argv.index("--camera-fov")
    assert argv[j:j + 2] == ["--camera-fov", "25.0"]
    defaults = rb.render_dumps_argv(Path("/r"), T.RenderFormValues(paths=("instance_grains",)))
    assert "--camera-fov" not in defaults and "--camera-mode" not in defaults


@needs_t
def test_form_dict_round_trips_and_reads_old_single_camera_jobs():
    form = T.RenderFormValues(camera_modes=("auto", "earth_chase"), camera_fov=25.0)
    assert rb.form_from_dict(T, rb.form_to_dict(form)) == form
    old = {**rb.form_to_dict(T.RenderFormValues()), "camera_mode": "grain_only", "apophis_scale": 1.0}
    del old["camera_modes"], old["camera_fov"]
    back = rb.form_from_dict(T, old)
    assert back.camera_modes == ("grain_only",) and back.camera_fov == 70.0
