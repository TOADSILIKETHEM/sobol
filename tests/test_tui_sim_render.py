"""Tests for tui_sim_render.py (sim+render pipeline TUI)."""
import argparse
import asyncio
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Run from sobol/ so the module is importable, same convention as
# sobol/tests/test_metric_fixes.py.
sys.path.insert(0, str(Path(__file__).parent.parent))
import tui_sim_render as tsr


def test_to_windows_path_translates_mnt_c():
    result = tsr.to_windows_path(Path("/mnt/c/Users/22boy/DEMCSVs/run_0001"))
    assert result == "C:/Users/22boy/DEMCSVs/run_0001"


def test_to_windows_path_passthrough_for_windows_style():
    # Already Windows-style (e.g. passed in directly) -> returned unchanged
    # apart from Path's own normalization, which does not apply here since
    # the string never starts with "/mnt/".
    result = tsr.to_windows_path(Path("C:/Users/22boy/DEMCSVs/run_0001"))
    assert result == "C:/Users/22boy/DEMCSVs/run_0001"


def test_sim_params_defaults():
    p = tsr.SimParams()
    assert p.prefix == "sobol"
    assert p.dry_run is False
    assert p.earth_sink_id == 4
    assert p.apophis_sink_id == 11


def test_render_form_values_defaults():
    r = tsr.RenderFormValues()
    assert r.resolution == "1920x1080"
    assert r.samples == 500
    assert r.fps == 24
    assert r.max_frames is None
    assert r.camera_mode == "auto"


def test_run_sim_stage_forwards_params(monkeypatch):
    fake_paths = (Path("base.setup"), Path("base.in"), Path("phantomsetup"), Path("phantom"))
    fake_preflight = MagicMock(return_value=fake_paths)
    fake_record = tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/run_0001",
        status="ok", closest_approach_km=1234.5, closest_approach_au=0.01,
        error="",
    )
    fake_run_one_case = MagicMock(return_value=fake_record)
    monkeypatch.setattr(tsr, "preflight", fake_preflight)
    monkeypatch.setattr(tsr, "run_one_case", fake_run_one_case)

    params = tsr.SimParams(prefix="sobol", earth_sink_id=4, apophis_sink_id=11)
    result = tsr.run_sim_stage(params)

    assert result is fake_record
    fake_preflight.assert_called_once()
    preflight_args = fake_preflight.call_args[0][0]
    assert isinstance(preflight_args, argparse.Namespace)
    assert preflight_args.prefix == "sobol"
    assert preflight_args.dry_run is False

    fake_run_one_case.assert_called_once()
    _, kwargs = fake_run_one_case.call_args
    assert kwargs["run_id"] == 1
    assert kwargs["sample"] is params.sample
    assert kwargs["earth_sink_id"] == 4
    assert kwargs["apophis_sink_id"] == 11
    assert kwargs["dry_run"] is False


def test_grains_and_bodies_dirs():
    run_dir = Path("/mnt/c/.../sobol_mass_runs/sobol_20260914_120000_batch/run_0001")
    base_output_dir = Path("/mnt/c/.../Code/DEMCSVs")
    output_root = Path("/mnt/c/.../sobol_mass_runs/sobol_20260914_120000_batch")

    grains_dir, bodies_dir = tsr._grains_and_bodies_dirs(run_dir, base_output_dir, output_root)

    assert grains_dir == base_output_dir / "sobol_20260914_120000_batch" / "run_0001_grains_output"
    assert bodies_dir == base_output_dir / "sobol_20260914_120000_batch" / "run_0001_bodies_output"


def test_render_output_dir():
    run_dir = Path("/mnt/c/.../sobol_mass_runs/sobol_20260914_120000_batch/run_0001")
    base_output_dir = Path("/mnt/c/.../Code/DEMCSVs")
    output_root = Path("/mnt/c/.../sobol_mass_runs/sobol_20260914_120000_batch")

    out_dir = tsr._render_output_dir(run_dir, base_output_dir, output_root)

    assert out_dir == base_output_dir / "sobol_20260914_120000_batch" / "run_0001_render"


def test_run_convert_stage_calls_converter(monkeypatch):
    fake_convert = MagicMock()
    fake_min_grains = MagicMock(return_value=450)
    monkeypatch.setattr(tsr, "run_dem_dump_convert", fake_convert)
    monkeypatch.setattr(tsr, "_min_dem_grains_from_setup", fake_min_grains)

    record = tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/batch/run_0001",
        status="ok", closest_approach_km=1.0, closest_approach_au=0.01, error="",
    )
    tsr.run_convert_stage(record, base_output_dir=Path("/mnt/c/.../Code/DEMCSVs"))

    fake_min_grains.assert_called_once_with(Path("sobol_mass_runs/batch/run_0001"))
    fake_convert.assert_called_once_with(
        input_dir=Path("sobol_mass_runs/batch/run_0001"),
        dump_convert_path=tsr.DEFAULT_DEM_DUMP_CONVERT,
        base_output_dir=Path("/mnt/c/.../Code/DEMCSVs"),
        min_dem_grains=450,
    )


def test_run_convert_stage_wraps_converter_exception(monkeypatch):
    def boom(**kwargs):
        raise ValueError("bad dump")
    monkeypatch.setattr(tsr, "run_dem_dump_convert", boom)
    monkeypatch.setattr(tsr, "_min_dem_grains_from_setup", lambda run_dir: None)

    record = tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/batch/run_0001",
        status="ok", closest_approach_km=1.0, closest_approach_au=0.01, error="",
    )
    with pytest.raises(tsr.ConvertError, match="bad dump"):
        tsr.run_convert_stage(record, base_output_dir=Path("/mnt/c/.../Code/DEMCSVs"))


def test_run_convert_stage_skips_when_sim_not_ok(monkeypatch):
    fake_convert = MagicMock()
    monkeypatch.setattr(tsr, "run_dem_dump_convert", fake_convert)

    record = tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/batch/run_0001",
        status="prepared_only", closest_approach_km=float("nan"),
        closest_approach_au=float("nan"), error="",
    )
    with pytest.raises(tsr.ConvertError, match="prepared_only"):
        tsr.run_convert_stage(record, base_output_dir=Path("/mnt/c/.../Code/DEMCSVs"))
    fake_convert.assert_not_called()


def test_build_render_command_without_max_frames():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_grains_output"),
        bodies_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_bodies_output"),
        output_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_render"),
        resolution="1280x720", samples=64, fps=30, camera_mode="auto",
    )
    cmd = tsr.build_render_command(params)

    assert cmd[0] == tsr.BLENDER_EXE
    assert cmd[1:4] == ["--background", "--python", tsr.DEM_HEADLESS_RENDER]
    assert cmd[4] == "--"
    assert "--grains-dir" in cmd and "C:/DEMCSVs/batch/run_0001_grains_output" in cmd
    assert "--resolution" in cmd and "1280x720" in cmd
    assert "--samples" in cmd and "64" in cmd
    assert "--fps" in cmd and "30" in cmd
    assert "--camera-mode" in cmd and "auto" in cmd
    assert "--max-frames" not in cmd


def test_build_render_command_with_max_frames():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_grains_output"),
        bodies_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_bodies_output"),
        output_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_render"),
        max_frames=3,
    )
    cmd = tsr.build_render_command(params)
    idx = cmd.index("--max-frames")
    assert cmd[idx + 1] == "3"


def test_build_render_command_without_encode_video_omits_flags():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    cmd = tsr.build_render_command(params)
    assert "--encode-video" not in cmd
    assert "--video-fps" not in cmd


def test_build_render_command_with_encode_video_default_fps():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
        encode_video=True,
    )
    cmd = tsr.build_render_command(params)
    assert "--encode-video" in cmd
    assert "--video-fps" not in cmd  # None means DEMHeadlessRender.py falls back to --fps


def test_build_render_command_with_encode_video_and_explicit_fps():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
        encode_video=True, video_fps=60,
    )
    cmd = tsr.build_render_command(params)
    idx = cmd.index("--video-fps")
    assert cmd[idx + 1] == "60"


def test_run_render_stage_success(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=["blender"], returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(tsr.subprocess, "run", lambda *a, **kw: fake_result)

    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    result = tsr.run_render_stage(params)
    assert result is fake_result


def test_run_render_stage_failure_raises_with_combined_tail(monkeypatch):
    # Blender's own print() output goes to stdout; a Python traceback goes to
    # stderr -- a failure can show up in either, so both must be checked.
    stdout_text = "\n".join(f"stdout line {i}" for i in range(30))
    stderr_text = "\n".join(f"stderr line {i}" for i in range(30))
    fake_result = subprocess.CompletedProcess(
        args=["blender"], returncode=1, stdout=stdout_text, stderr=stderr_text,
    )
    monkeypatch.setattr(tsr.subprocess, "run", lambda *a, **kw: fake_result)

    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    with pytest.raises(tsr.RenderError) as excinfo:
        tsr.run_render_stage(params)
    msg = str(excinfo.value)
    assert "stderr line 29" in msg  # last lines of the combined text are kept
    assert "stdout line 0" not in msg  # only the last ~40 combined lines survive


def test_run_render_stage_failure_reads_stdout_only_traceback(monkeypatch):
    # Regression case: some failures print everything to stdout with an
    # empty stderr (e.g. exec()'d script prints its own error and returns
    # non-zero without raising) -- stderr-only capture would show nothing.
    fake_result = subprocess.CompletedProcess(
        args=["blender"], returncode=1, stdout="GRAINS_DIR pattern not found", stderr="",
    )
    monkeypatch.setattr(tsr.subprocess, "run", lambda *a, **kw: fake_result)

    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    with pytest.raises(tsr.RenderError, match="GRAINS_DIR pattern not found"):
        tsr.run_render_stage(params)


def _ok_record(run_dir="sobol_mass_runs/batch/run_0001"):
    return tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir=run_dir, status="ok",
        closest_approach_km=1.0, closest_approach_au=0.01, error="",
    )


def test_run_pipeline_stops_after_failed_sim(monkeypatch):
    bad_record = tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/batch/run_0001",
        status="phantomsetup_failed", closest_approach_km=float("nan"),
        closest_approach_au=float("nan"), error="phantomsetup exited 1",
    )
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: bad_record)
    convert_mock = MagicMock()
    render_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_convert_stage", convert_mock)
    monkeypatch.setattr(tsr, "run_render_stage", render_mock)

    result = tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(), Path("/mnt/c/.../DEMCSVs"))

    assert result.stage == "sim"
    assert result.ok is False
    assert "phantomsetup exited 1" in result.message
    convert_mock.assert_not_called()
    render_mock.assert_not_called()


def test_run_pipeline_stops_after_failed_convert(monkeypatch):
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record())

    def boom_convert(record, base_output_dir):
        raise tsr.ConvertError("conversion failed: bad dump")
    monkeypatch.setattr(tsr, "run_convert_stage", boom_convert)
    render_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_render_stage", render_mock)

    result = tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(), Path("/mnt/c/.../DEMCSVs"))

    assert result.stage == "convert"
    assert result.ok is False
    assert "bad dump" in result.message
    render_mock.assert_not_called()


def _write_fake_npz(grains_dir: Path, n: int = 1) -> None:
    grains_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (grains_dir / f"sobol_{i:05d}.npz").write_bytes(b"")


def test_run_pipeline_stops_after_failed_render(monkeypatch, tmp_path):
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record())

    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(record, base_output_dir):
        _write_fake_npz(grains_dir, 1)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)

    def boom_render(params):
        raise tsr.RenderError("blender exited 1:\nsome error")
    monkeypatch.setattr(tsr, "run_render_stage", boom_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(), base_output_dir)

    assert result.stage == "render"
    assert result.ok is False
    assert "some error" in result.message


def test_run_pipeline_convert_succeeds_but_no_npz_fails_before_render(monkeypatch, tmp_path):
    # DEMDumpConvert.py never raises -- with no dumps, or a dump that throws,
    # it prints and carries on, so run_convert_stage() "succeeding" with zero
    # npz files must itself be treated as a convert failure, and render must
    # never start.
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record())
    monkeypatch.setattr(tsr, "run_convert_stage", lambda record, base_output_dir: None)
    render_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_render_stage", render_mock)

    base_output_dir = tmp_path / "DEMCSVs"
    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(), base_output_dir)

    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"
    assert result.stage == "convert"
    assert result.ok is False
    assert str(grains_dir) in result.message
    assert "sobol" in result.message  # names the converter's hardcoded prefix
    render_mock.assert_not_called()


def test_run_pipeline_full_success(monkeypatch, tmp_path):
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)

    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(rec, base_output_dir):
        _write_fake_npz(grains_dir, 1)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)

    captured = {}
    def fake_render(params):
        captured["params"] = params
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    render_form = tsr.RenderFormValues(resolution="640x360", samples=8, camera_mode="grain_only")
    result = tsr.run_pipeline(sim_params, render_form, base_output_dir)

    assert result.stage == "done"
    assert result.ok is True
    rp = captured["params"]
    assert rp.grains_dir == base_output_dir / "batch" / "run_0001_grains_output"
    assert rp.bodies_dir == base_output_dir / "batch" / "run_0001_bodies_output"
    assert rp.output_dir == base_output_dir / "batch" / "run_0001_render"
    assert rp.resolution == "640x360"
    assert rp.samples == 8
    assert rp.camera_mode == "grain_only"


def test_run_pipeline_success_message_counts_rendered_frames(monkeypatch, tmp_path):
    # Ruling: run_pipeline's success message counts rendered PNGs in
    # output_dir rather than a fixed "rendered to <dir>" string, and now also
    # the converted npz count (finding 2).
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)

    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(rec, base_output_dir):
        _write_fake_npz(grains_dir, 3)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)

    def fake_render(params):
        params.output_dir.mkdir(parents=True, exist_ok=True)
        (params.output_dir / "frame_0001.png").write_bytes(b"")
        (params.output_dir / "frame_0002.png").write_bytes(b"")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(), base_output_dir)

    output_dir = base_output_dir / "batch" / "run_0001_render"
    assert result.message == f"converted 3 frame(s); rendered 2 frame(s) to {output_dir}"


def test_run_pipeline_message_reports_encoded_video(monkeypatch, tmp_path):
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)

    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(rec, base_output_dir):
        _write_fake_npz(grains_dir, 2)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)

    def fake_render(params):
        params.output_dir.mkdir(parents=True, exist_ok=True)
        (params.output_dir / "frame_0001.png").write_bytes(b"")
        (params.output_dir / "frame_0002.png").write_bytes(b"")
        (params.output_dir / "animation.mp4").write_bytes(b"")  # DEMHeadlessRender.py's output
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    render_form = tsr.RenderFormValues(encode_video=True)
    result = tsr.run_pipeline(sim_params, render_form, base_output_dir)

    output_dir = base_output_dir / "batch" / "run_0001_render"
    assert result.message == (
        f"converted 2 frame(s); rendered 2 frame(s) to {output_dir}"
        f"; encoded to {output_dir / 'animation.mp4'}"
    )


def test_run_pipeline_message_flags_missing_video_when_requested(monkeypatch, tmp_path):
    # If --encode-video was requested but DEMHeadlessRender.py somehow didn't
    # produce the mp4 (and didn't raise either), the success message should
    # say so rather than silently claiming a video that isn't there.
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)

    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(rec, base_output_dir):
        _write_fake_npz(grains_dir, 1)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)

    def fake_render(params):
        params.output_dir.mkdir(parents=True, exist_ok=True)
        (params.output_dir / "frame_0001.png").write_bytes(b"")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    render_form = tsr.RenderFormValues(encode_video=True)
    result = tsr.run_pipeline(sim_params, render_form, base_output_dir)

    assert "video encode requested but" in result.message
    assert "not found" in result.message


def test_run_pipeline_notifies_stage_callbacks(monkeypatch, tmp_path):
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)

    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(rec, base_output_dir):
        _write_fake_npz(grains_dir, 3)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)
    monkeypatch.setattr(
        tsr, "run_render_stage",
        lambda params: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )

    stages = []

    def on_stage(stage, **info):
        stages.append((stage, info))

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    tsr.run_pipeline(
        sim_params, tsr.RenderFormValues(), base_output_dir, on_stage=on_stage,
    )

    assert [s[0] for s in stages] == ["sim", "convert", "render"]
    assert stages[2][1]["n_expected"] == 3


def test_run_pipeline_render_n_expected_respects_max_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record())
    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"

    def fake_convert(rec, base_output_dir):
        _write_fake_npz(grains_dir, 5)
    monkeypatch.setattr(tsr, "run_convert_stage", fake_convert)
    monkeypatch.setattr(
        tsr, "run_render_stage",
        lambda params: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )

    stages = []
    tsr.run_pipeline(
        tsr.SimParams(output_root=Path("sobol_mass_runs/batch")),
        tsr.RenderFormValues(max_frames=2),
        base_output_dir,
        on_stage=lambda stage, **info: stages.append((stage, info)),
    )

    assert stages[-1][0] == "render"
    assert stages[-1][1]["n_expected"] == 2


def test_format_live_progress_sim():
    assert tsr.format_live_progress(44, "sim", n_dumps=14) == (
        "Running sim... 44s elapsed, 14 dump file(s)"
    )


def test_format_live_progress_convert_with_expected():
    assert tsr.format_live_progress(900, "convert", n_npz=80, n_expected=217) == (
        "Converting... 900s elapsed, 80/217 npz"
    )


def test_format_live_progress_render_with_expected():
    assert tsr.format_live_progress(1200, "render", n_png=20, n_expected=217) == (
        "Rendering... 1200s elapsed, 20/217 frame(s)"
    )


def test_format_live_progress_render_without_expected():
    assert tsr.format_live_progress(10, "render", n_png=0) == (
        "Rendering... 10s elapsed, 0 frame(s)"
    )


def test_format_live_progress_warning_prefix():
    msg = tsr.format_live_progress(1, "sim", n_dumps=0, warning="Warning: np_apophis=5000 — x. ")
    assert msg.startswith("Warning: np_apophis=5000")
    assert "Running sim..." in msg


def test_count_matching_missing_dir(tmp_path):
    assert tsr.count_matching(tmp_path / "nope", "*.png") == 0
    assert tsr.count_matching(None, "*.png") == 0


def test_count_matching_pngs(tmp_path):
    (tmp_path / "frame_0001.png").write_bytes(b"")
    (tmp_path / "frame_0002.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("skip")
    assert tsr.count_matching(tmp_path, "frame_*.png") == 2


def test_tick_progress_render_stage_counts_pngs(tmp_path):
    """Status bar must leave dump counts once the worker reports render."""

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            render_dir = tmp_path / "render"
            render_dir.mkdir()
            (render_dir / "frame_0001.png").write_bytes(b"")
            (render_dir / "frame_0002.png").write_bytes(b"")
            app._pipeline_start = time.monotonic()
            app._pipeline_stage = "render"
            app._pipeline_n_expected = 217
            app._pipeline_render_dir = render_dir
            app._pipeline_warning = ""
            app._pipeline_dry_run = False
            app._tick_progress()
            return str(app.query_one("#status", tsr.Static).content)

    text = asyncio.run(_scenario())
    assert "Rendering..." in text
    assert "2/217 frame(s)" in text
    assert "dump file" not in text


def test_check_blender_exe_raises_when_missing(tmp_path):
    missing = tmp_path / "does_not_exist" / "blender.exe"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        tsr.check_blender_exe(str(missing))


def test_check_blender_exe_passes_when_present(tmp_path):
    fake_exe = tmp_path / "blender.exe"
    fake_exe.write_text("not a real binary, just needs to exist")
    tsr.check_blender_exe(str(fake_exe))  # must not raise


def test_warn_if_high_grain_count_below_threshold():
    assert tsr.warn_if_high_grain_count(500) is None
    assert tsr.warn_if_high_grain_count(2000) is None  # boundary is inclusive


def test_warn_if_high_grain_count_above_threshold():
    msg = tsr.warn_if_high_grain_count(5000)
    assert msg is not None
    assert "5000" in msg
    assert "2000" in msg


# --- Textual pilot tests -----------------------------------------------
#
# No pytest-asyncio in this environment: each test below is a normal sync
# function that runs its own async scenario via asyncio.run(). The worker
# started by @work(thread=True) lands its result via call_from_thread from a
# background thread, so a single fixed pilot.pause() is flaky -- instead poll
# the status text in a bounded loop (up to ~5s) until it stops being one of
# the known *transient* strings _launch()/_tick_progress() write ("Dry
# run...", "Running..." with or without the live-progress suffix). The final
# report from _report_result() always starts with "[<n>s] ", which neither
# transient string does, so this reliably detects "worker has landed" without
# depending on timing.

async def _click_and_wait_for_settled_status(
    app, button_id: str, timeout_s: float = 5.0
) -> tuple[str, bool]:
    async with app.run_test() as pilot:
        await pilot.click(button_id)
        status = app.query_one("#status", tsr.Static)
        text = str(status.content)
        elapsed = 0.0
        step = 0.1
        while elapsed < timeout_s and (
            text.startswith("Dry run")
            or text.startswith("Running")
            or text.startswith("Converting")
            or text.startswith("Rendering")
        ):
            await pilot.pause(step)
            elapsed += step
            text = str(status.content)
        still_running = app.is_running
    return text, still_running


def test_dry_run_stops_before_convert_and_render(monkeypatch):
    dry_run_record = tsr.RunRecord(
        run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/batch/run_0001",
        status="prepared_only", closest_approach_km=float("nan"),
        closest_approach_au=float("nan"), error="",
    )
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: dry_run_record)
    convert_mock = MagicMock()
    render_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_convert_stage", convert_mock)
    monkeypatch.setattr(tsr, "run_render_stage", render_mock)
    monkeypatch.setattr(tsr, "check_blender_exe", lambda *a, **kw: None)

    app = tsr.SimRenderTUIApp()
    text, still_running = asyncio.run(_click_and_wait_for_settled_status(app, "#btn-dryrun"))

    assert "Dry run complete" in text
    assert "sobol_mass_runs/batch/run_0001" in text
    assert "nothing to convert/render" in text
    assert still_running is True
    convert_mock.assert_not_called()
    render_mock.assert_not_called()


def test_worker_exception_reported_without_crashing_app(monkeypatch):
    # Ruling 5: an uncaught exception inside run_pipeline() (e.g. preflight()
    # raising FileNotFoundError for a missing sobol.setup) must be caught by
    # the worker and reported, not crash the app.
    def boom(params):
        raise FileNotFoundError("sobol.setup not found")

    monkeypatch.setattr(tsr, "run_sim_stage", boom)
    monkeypatch.setattr(tsr, "check_blender_exe", lambda *a, **kw: None)

    app = tsr.SimRenderTUIApp()
    text, still_running = asyncio.run(_click_and_wait_for_settled_status(app, "#btn-dryrun"))

    assert "Failed at error" in text
    assert "FileNotFoundError" in text
    assert "sobol.setup not found" in text
    assert still_running is True


def test_second_launch_while_running_does_not_start_second_worker(monkeypatch):
    # Review fix round 1, Important #1: a second click while a pipeline is
    # in flight must not start a second @work(thread=True) worker (which
    # defaults to exclusive=False) -- that would stomp the first run's
    # _pipeline_* state. run_sim_stage blocks on a threading.Event so the
    # test controls exactly when the first (and only) worker completes.
    release_event = threading.Event()
    call_count = {"n": 0}

    def blocking_run_sim_stage(params):
        call_count["n"] += 1
        assert release_event.wait(timeout=5.0), "test event was never released"
        return tsr.RunRecord(
            run_id=1, mass_input_kg=float("nan"), run_dir="sobol_mass_runs/batch/run_0001",
            status="prepared_only", closest_approach_km=float("nan"),
            closest_approach_au=float("nan"), error="",
        )

    monkeypatch.setattr(tsr, "run_sim_stage", blocking_run_sim_stage)
    monkeypatch.setattr(tsr, "check_blender_exe", lambda *a, **kw: None)

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            await pilot.click("#btn-dryrun")
            await pilot.pause(0.2)  # let the worker thread start and block

            run_btn = app.query_one("#btn-run", tsr.Button)
            dryrun_btn = app.query_one("#btn-dryrun", tsr.Button)
            assert run_btn.disabled is True
            assert dryrun_btn.disabled is True

            # Second click while the first pipeline is still running. The
            # button is disabled, so Textual itself swallows this click (no
            # Pressed message) -- that is the primary defense. Status stays
            # whatever the first launch already set.
            await pilot.click("#btn-dryrun")
            await pilot.pause(0.2)
            status = app.query_one("#status", tsr.Static)
            text_after_disabled_click = str(status.content)

            # Belt-and-braces: call _launch() directly (bypassing button
            # state) the way a race between the click event queue and a
            # fast-landing worker could in principle re-enter it -- the
            # _pipeline_running guard inside _launch() must still refuse a
            # second run and say so in the status bar.
            app._launch(dry_run=True)
            text_from_direct_relaunch = str(status.content)

            release_event.set()  # let the (only) worker finish

            elapsed = 0.0
            step = 0.1
            while elapsed < 5.0 and run_btn.disabled:
                await pilot.pause(step)
                elapsed += step

            final_text = str(status.content)
            run_disabled_after = run_btn.disabled
            dryrun_disabled_after = dryrun_btn.disabled
        return (
            text_after_disabled_click, text_from_direct_relaunch, final_text,
            run_disabled_after, dryrun_disabled_after,
        )

    (
        text_after_disabled_click, text_from_direct_relaunch, final_text,
        run_disabled_after, dryrun_disabled_after,
    ) = asyncio.run(_scenario())

    assert text_after_disabled_click.startswith("Dry run")  # disabled click was a no-op
    assert "already running" in text_from_direct_relaunch.lower()
    assert call_count["n"] == 1  # neither second attempt started a second worker
    assert "Dry run complete" in final_text
    assert run_disabled_after is False
    assert dryrun_disabled_after is False


# --- _build_sim_params coverage (review fix round 1, Important #2) -----

def test_build_sim_params_output_root_shape():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            # prefix="sobol", batch_label="sim_render" are the form defaults.
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    # Was Path("sobol_mass_runs").resolve() (cwd-dependent) before finding 1's
    # fix -- the form's output-root Input now defaults to the absolute,
    # cwd-independent Honours/sobol_mass_runs path.
    assert params.output_root.parent == tsr._default_output_root()
    assert re.match(r"^sobol_\d{8}_\d{6}_sim_render$", params.output_root.name), (
        params.output_root.name
    )


def test_blank_batch_label_shows_error_and_never_starts_worker(monkeypatch):
    run_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_sim_stage", run_mock)
    monkeypatch.setattr(tsr, "check_blender_exe", lambda *a, **kw: None)

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            app.query_one("#batch-label", tsr.Input).value = ""
            await pilot.click("#btn-dryrun")
            await pilot.pause(0.2)
            status = app.query_one("#status", tsr.Static)
            return str(status.content)

    text = asyncio.run(_scenario())

    assert text.startswith("Error:")
    run_mock.assert_not_called()


def test_build_sim_params_shape_file_set():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#shape-file", tsr.Input).value = "/mnt/c/shapes/apophis.obj"
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.shape_file == "/mnt/c/shapes/apophis.obj"
    assert params.sample.use_shape_crop is True


def test_build_sim_params_shape_file_blank():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            # shape-file Input defaults to "" -- left untouched here.
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.shape_file is None
    assert params.sample.use_shape_crop is None


# --- Finding 1: path defaults must not depend on launch cwd ------------

def test_default_base_dir_is_module_directory():
    assert tsr._default_base_dir() == Path(tsr.__file__).resolve().parent


def test_default_base_dir_unaffected_by_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert tsr._default_base_dir().is_absolute()
    assert tsr._default_base_dir() == Path(tsr.__file__).resolve().parent


def test_default_phantom_dir_matches_run_mass_sobol_phantom_default(monkeypatch):
    monkeypatch.delenv("PHANTOM_DIR", raising=False)
    from run_mass_sobol_phantom import _DEFAULT_PHANTOM_DIR as rms_default
    assert tsr._default_phantom_dir() == str(rms_default)
    assert Path(tsr._default_phantom_dir()).is_absolute()


def test_default_phantom_dir_honors_env_var(monkeypatch):
    monkeypatch.setenv("PHANTOM_DIR", "/custom/phantom/install")
    assert tsr._default_phantom_dir() == "/custom/phantom/install"


def test_default_output_root_is_sibling_of_sobol_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = tsr._default_output_root()
    assert out.is_absolute()
    assert out.name == "sobol_mass_runs"
    assert out.parent == Path(tsr.__file__).resolve().parent.parent


def test_sim_params_path_defaults_are_absolute_and_match_helpers():
    p = tsr.SimParams()
    assert p.base_dir == tsr._default_base_dir()
    assert str(p.phantom_dir) == tsr._default_phantom_dir()
    assert p.output_root == tsr._default_output_root()


def test_form_path_inputs_default_to_absolute_paths():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            return (
                app.query_one("#base-dir", tsr.Input).value,
                app.query_one("#phantom-dir", tsr.Input).value,
                app.query_one("#output-root", tsr.Input).value,
            )

    base_dir_val, phantom_dir_val, output_root_val = asyncio.run(_scenario())

    assert Path(base_dir_val).is_absolute()
    assert Path(phantom_dir_val).is_absolute()
    assert Path(output_root_val).is_absolute()
    assert output_root_val == str(tsr._default_output_root())


# --- Finding 3: camera-mode Select must not allow a blank selection -----

def test_camera_mode_select_disallows_blank():
    # Textual 8.2.6's Select has no public allow_blank reader -- only the
    # constructor kwarg and the private _allow_blank it's stored in.
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            select = app.query_one("#camera-mode", tsr.Select)
            return select._allow_blank, select.value

    allow_blank, value = asyncio.run(_scenario())
    assert allow_blank is False
    assert value != tsr.Select.BLANK


# --- Finding 4: render form values are validated eagerly ----------------

def test_build_render_form_rejects_malformed_resolution():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#resolution", tsr.Input).value = "not-a-resolution"
            with pytest.raises(ValueError):
                app._build_render_form()

    asyncio.run(_scenario())


def test_build_render_form_rejects_zero_dimension_resolution():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#resolution", tsr.Input).value = "0x1080"
            with pytest.raises(ValueError):
                app._build_render_form()

    asyncio.run(_scenario())


def test_build_render_form_rejects_nonpositive_samples():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#samples", tsr.Input).value = "0"
            with pytest.raises(ValueError):
                app._build_render_form()

    asyncio.run(_scenario())


def test_build_render_form_rejects_nonpositive_fps():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#fps", tsr.Input).value = "-1"
            with pytest.raises(ValueError):
                app._build_render_form()

    asyncio.run(_scenario())


def test_build_render_form_rejects_nonpositive_max_frames():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#max-frames", tsr.Input).value = "0"
            with pytest.raises(ValueError):
                app._build_render_form()

    asyncio.run(_scenario())


def test_build_render_form_accepts_valid_values():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            return app._build_render_form()

    form = asyncio.run(_scenario())
    assert form.resolution == "1920x1080"


def test_bad_resolution_shows_error_and_never_starts_worker(monkeypatch):
    run_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_sim_stage", run_mock)
    monkeypatch.setattr(tsr, "check_blender_exe", lambda *a, **kw: None)

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            app.query_one("#resolution", tsr.Input).value = "bogus"
            await pilot.click("#btn-dryrun")
            await pilot.pause(0.2)
            status = app.query_one("#status", tsr.Static)
            return str(status.content)

    text = asyncio.run(_scenario())

    assert text.startswith("Error:")
    run_mock.assert_not_called()


# --- Finding 5: kt_cgs blank means "template default", not 0 ------------

def test_kt_cgs_input_defaults_to_blank_with_template_placeholder():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            inp = app.query_one("#kt-cgs", tsr.Input)
            return inp.value, inp.placeholder

    value, placeholder = asyncio.run(_scenario())

    assert value == ""
    assert placeholder == "(template default)"


def test_build_sim_params_kt_cgs_blank_maps_to_none():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.kt_cgs is None
    assert params.sample.coh_gap_max_cgs is None


def test_build_sim_params_kt_cgs_positive_computes_coh_gap_max():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#kt-cgs", tsr.Input).value = "1e7"
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.kt_cgs == 1e7
    assert params.sample.coh_gap_max_cgs is not None


def test_build_sim_params_kt_cgs_zero_does_not_compute_coh_gap_max():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#kt-cgs", tsr.Input).value = "0"
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.kt_cgs == 0.0
    assert params.sample.coh_gap_max_cgs is None
