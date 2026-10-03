"""Tests for tui_sim_render.py (sim+render pipeline TUI)."""
import argparse
import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_to_windows_path_home_to_unc(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    result = tsr.to_windows_path(Path("/home/mboyle/Honours/Code/BlenderConvert/DEMHeadlessRender.py"))
    assert result == r"\\wsl.localhost\Ubuntu\home\mboyle\Honours\Code\BlenderConvert\DEMHeadlessRender.py"


def test_to_windows_path_home_with_spaces(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    result = tsr.to_windows_path(Path("/home/mboyle/Honours/Code/Data analysis/x.py"))
    assert result == r"\\wsl.localhost\Ubuntu\home\mboyle\Honours\Code\Data analysis\x.py"


def test_code_and_data_dirs_split():
    assert tsr.CODE_DIR == Path("/home/mboyle/Honours/Code")
    assert tsr.DATA_DIR == Path("/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code")
    assert tsr.DEM_HEADLESS_RENDER.startswith("\\\\wsl.localhost\\")
    assert tsr.WIN_VENV_PYTHON == tsr.DATA_DIR / ".venv" / "Scripts" / "python.exe"
    assert tsr.DEFAULT_PLACEHOLDER_OBJ.startswith("C:/Users/22boy/OneDrive/")


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
    batch_dir = Path("/mnt/c/.../Code/DEMCSVs/sobol_20260914_120000_batch")
    out_dir = tsr._render_output_dir(batch_dir, "run_0001", "per_sphere")
    assert out_dir == batch_dir / "run_0001_render_per_sphere"


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

    result = tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(paths=("per_sphere",)), Path("/mnt/c/.../DEMCSVs"))

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

    result = tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(paths=("per_sphere",)), Path("/mnt/c/.../DEMCSVs"))

    assert result.stage == "convert"
    assert result.ok is False
    assert "bad dump" in result.message
    render_mock.assert_not_called()


def _write_fake_npz(grains_dir: Path, n: int = 1) -> None:
    grains_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (grains_dir / f"sobol_{i:05d}.npz").write_bytes(b"")


def _write_fake_pngs(output_dir: Path, n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (output_dir / f"frame_{i:04d}.png").write_bytes(b"")


def _ok_completed():
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


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
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("per_sphere",)), base_output_dir)

    assert result.stage == "partial"
    assert result.ok is False
    assert result.paths[0].name == "per_sphere"
    assert result.paths[0].stage == "render"
    assert "some error" in result.message
    assert "per_sphere: failed at render:" in result.message


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
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("per_sphere",)), base_output_dir)

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
        _write_fake_pngs(params.output_dir, 1)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    render_form = tsr.RenderFormValues(resolution="640x360", samples=8, camera_mode="grain_only", paths=("per_sphere",))
    result = tsr.run_pipeline(sim_params, render_form, base_output_dir)

    assert result.stage == "done"
    assert result.ok is True
    assert [r.name for r in result.paths] == ["per_sphere"]
    rp = captured["params"]
    assert rp.viz_path == "per_sphere"
    assert rp.manifest is None
    assert rp.grains_dir == base_output_dir / "batch" / "run_0001_grains_output"
    assert rp.bodies_dir == base_output_dir / "batch" / "run_0001_bodies_output"
    assert rp.output_dir == base_output_dir / "batch" / "run_0001_render_per_sphere"
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
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("per_sphere",)), base_output_dir)

    output_dir = base_output_dir / "batch" / "run_0001_render_per_sphere"
    assert result.message.startswith("converted 3 frame(s); per_sphere: ok, 2 frame(s) in ")
    assert result.message.endswith(f"s to {output_dir}")


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
    render_form = tsr.RenderFormValues(encode_video=True, paths=("per_sphere",))
    result = tsr.run_pipeline(sim_params, render_form, base_output_dir)

    output_dir = base_output_dir / "batch" / "run_0001_render_per_sphere"
    assert f"per_sphere: ok, 2 frame(s) in " in result.message
    assert result.message.endswith(f"s to {output_dir}, encoded to {output_dir / 'animation.mp4'}")


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
    render_form = tsr.RenderFormValues(encode_video=True, paths=("per_sphere",))
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
        sim_params, tsr.RenderFormValues(paths=("per_sphere",)), base_output_dir, on_stage=on_stage,
    )

    assert [s[0] for s in stages] == ["sim", "convert", "render"]
    assert stages[2][1]["n_expected"] == 3
    assert stages[2][1]["path"] == "per_sphere"
    assert (stages[2][1]["index"], stages[2][1]["total"]) == (1, 1)
    assert stages[2][1]["output_dir"] == base_output_dir / "batch" / "run_0001_render_per_sphere"


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
        tsr.RenderFormValues(max_frames=2, paths=("per_sphere",)),
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
            or text.startswith("Preprocessing")
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
    assert params.sample.dn_cohes_factor is None


def test_build_sim_params_kt_cgs_positive_carries_dn_for_coh_gap():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#kt-cgs", tsr.Input).value = "1e7"
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.kt_cgs == 1e7
    # gap is resolved after phantomsetup from the real grain radius
    assert params.sample.coh_gap_max_cgs is None
    assert params.sample.dn_cohes_factor == tsr.DEFAULT_DN_COHES_FACTOR


def test_build_sim_params_kt_cgs_zero_does_not_compute_coh_gap_max():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#kt-cgs", tsr.Input).value = "0"
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.kt_cgs == 0.0
    assert params.sample.coh_gap_max_cgs is None
    assert params.sample.dn_cohes_factor is None


# --- Bulk render paths: commands, preprocess stage, partial failure -------

def _mnt_ctx(**form_kw):
    batch = Path("/mnt/c/DEMCSVs/batch")
    return tsr.PathContext(
        grains_dir=batch / "run_0001_grains_output",
        bodies_dir=batch / "run_0001_bodies_output",
        batch_dir=batch,
        run_name="run_0001",
        render_form=tsr.RenderFormValues(**form_kw),
        n_expected_frames=3,
    )


def test_render_path_registry_order_and_wiring():
    assert tsr.RENDER_PATH_NAMES == ("per_sphere", "composite", "instance_grains", "instance_static")
    assert tsr.RENDER_PATHS["per_sphere"].build_preprocess_command is None
    assert tsr.RENDER_PATHS["composite"].viz_dir_suffix == "_viz"
    assert tsr.RENDER_PATHS["composite"].headless_viz_path == "composite"
    assert tsr.RENDER_PATHS["instance_grains"].viz_dir_suffix == "_viz_instance"
    assert tsr.RENDER_PATHS["instance_grains"].headless_viz_path == "instance"
    assert tsr.RENDER_PATHS["instance_static"].viz_dir_suffix == "_viz_instance_static"
    assert tsr.RENDER_PATHS["instance_static"].headless_viz_path == "instance"


def test_render_form_values_new_defaults():
    r = tsr.RenderFormValues()
    assert r.paths == ("instance_grains",)
    assert r.envelope_method == "hull"
    assert r.placeholder_obj == tsr.DEFAULT_PLACEHOLDER_OBJ
    assert r.placeholder_obj.endswith("/apophis_v233s7.obj")
    assert r.placeholder_obj.startswith("C:/")
    assert r.n_points == 1_000_000


def test_composite_preprocess_command():
    ctx = _mnt_ctx(envelope_method="sdf", max_frames=3)
    cmd = tsr.build_composite_preprocess_command(ctx, ctx.batch_dir / "run_0001_viz")
    assert cmd[0] == str(tsr.WIN_VENV_PYTHON)
    assert cmd[1] == tsr.to_windows_path(tsr.CODE_DIR / "viz" / "viz_preprocess.py")
    assert cmd[cmd.index("--grains-dir") + 1] == "C:/DEMCSVs/batch/run_0001_grains_output"
    assert cmd[cmd.index("--bodies-dir") + 1] == "C:/DEMCSVs/batch/run_0001_bodies_output"
    assert cmd[cmd.index("--output-dir") + 1] == "C:/DEMCSVs/batch/run_0001_viz"
    assert cmd[cmd.index("--envelope-method") + 1] == "sdf"
    assert cmd[cmd.index("--max-frames") + 1] == "3"


def test_composite_preprocess_command_without_max_frames():
    ctx = _mnt_ctx()
    cmd = tsr.build_composite_preprocess_command(ctx, ctx.batch_dir / "run_0001_viz")
    assert "--max-frames" not in cmd
    assert cmd[cmd.index("--envelope-method") + 1] == "hull"


def test_instance_grains_preprocess_command():
    ctx = _mnt_ctx(max_frames=2)
    cmd = tsr.build_instance_grains_preprocess_command(ctx, ctx.batch_dir / "run_0001_viz_instance")
    assert cmd[0] == str(tsr.WIN_VENV_PYTHON)
    assert cmd[1] == tsr.to_windows_path(
        tsr.CODE_DIR / "viz" / "viz_preprocess_grains_instance.py"
    )
    assert cmd[cmd.index("--output-dir") + 1] == "C:/DEMCSVs/batch/run_0001_viz_instance"
    assert cmd[cmd.index("--max-frames") + 1] == "2"
    assert "--envelope-method" not in cmd


def test_instance_static_preprocess_command():
    ctx = _mnt_ctx(placeholder_obj="C:/shapes/a.obj", n_points=5000, max_frames=2)
    cmd = tsr.build_instance_static_preprocess_command(
        ctx, ctx.batch_dir / "run_0001_viz_instance_static"
    )
    assert cmd[0] == tsr.BLENDER_EXE
    assert cmd[1:6] == [
        "--background", "--python-exit-code", "1", "--python",
        tsr.to_windows_path(tsr.CODE_DIR / "viz" / "viz_preprocess_lite.py"),
    ]
    assert cmd[6] == "--"
    assert cmd[cmd.index("--shape-obj") + 1] == "C:/shapes/a.obj"
    assert cmd[cmd.index("--n-points") + 1] == "5000"
    assert cmd[cmd.index("--output-dir") + 1] == "C:/DEMCSVs/batch/run_0001_viz_instance_static"
    assert "--max-frames" not in cmd


def test_instance_static_preprocess_command_converts_wsl_obj_path():
    ctx = _mnt_ctx(placeholder_obj="/mnt/c/shapes/a.obj")
    cmd = tsr.build_instance_static_preprocess_command(ctx, ctx.batch_dir / "x")
    assert cmd[cmd.index("--shape-obj") + 1] == "C:/shapes/a.obj"


def test_build_render_command_bulk_path_uses_manifest():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"),
        output_dir=Path("/mnt/c/DEMCSVs/batch/run_0001_render_composite"),
        viz_path="composite", manifest=Path("/mnt/c/DEMCSVs/batch/run_0001_viz/manifest.json"),
    )
    cmd = tsr.build_render_command(params)
    assert cmd[4] == "--"
    assert cmd[cmd.index("--viz-path") + 1] == "composite"
    assert cmd[cmd.index("--manifest") + 1] == "C:/DEMCSVs/batch/run_0001_viz/manifest.json"
    assert "--grains-dir" not in cmd and "--bodies-dir" not in cmd
    assert cmd[cmd.index("--output-dir") + 1] == "C:/DEMCSVs/batch/run_0001_render_composite"


def test_build_render_command_per_sphere_passes_viz_path():
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    cmd = tsr.build_render_command(params)
    assert cmd[cmd.index("--viz-path") + 1] == "per_sphere"
    assert "--manifest" not in cmd


def test_run_preprocess_stage_failure_raises_with_tail(monkeypatch, tmp_path):
    fake = subprocess.CompletedProcess(args=[], returncode=2, stdout="a\nqhull boom", stderr="")
    captured = {}
    def fake_run(cmd, **kw):
        captured.update(kw)
        return fake
    monkeypatch.setattr(tsr.subprocess, "run", fake_run)
    with pytest.raises(tsr.PreprocessError, match="qhull boom"):
        tsr.run_preprocess_stage(["python.exe"], tmp_path / "viz")
    assert captured["cwd"] == str(tsr.CODE_DIR)


def test_run_preprocess_stage_exit_0_without_manifest_raises(monkeypatch, tmp_path):
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Traceback ... boom", stderr="",
    )
    monkeypatch.setattr(tsr.subprocess, "run", lambda *a, **kw: fake)
    with pytest.raises(tsr.PreprocessError, match="wrote no manifest at") as excinfo:
        tsr.run_preprocess_stage(["python.exe"], tmp_path / "viz")
    assert "Traceback ... boom" in str(excinfo.value)


def test_run_preprocess_stage_returns_manifest(monkeypatch, tmp_path):
    out = tmp_path / "viz"
    def fake_run(cmd, **kw):
        out.mkdir()
        (out / "manifest.json").write_text("{}")
        return _ok_completed()
    monkeypatch.setattr(tsr.subprocess, "run", fake_run)
    assert tsr.run_preprocess_stage(["python.exe"], out) == out / "manifest.json"


def test_run_preprocess_stage_spawn_oserror_raises_preprocess_error(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        raise FileNotFoundError("no python.exe")
    monkeypatch.setattr(tsr.subprocess, "run", fake_run)
    with pytest.raises(tsr.PreprocessError, match="could not start"):
        tsr.run_preprocess_stage(["python.exe"], tmp_path / "viz")


def test_run_render_stage_spawn_oserror_raises_render_error(monkeypatch):
    def fake_run(cmd, **kw):
        raise FileNotFoundError("no blender.exe")
    monkeypatch.setattr(tsr.subprocess, "run", fake_run)
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    with pytest.raises(tsr.RenderError, match="could not start"):
        tsr.run_render_stage(params)


def test_run_preprocess_stage_passes_errors_replace(monkeypatch, tmp_path):
    out = tmp_path / "viz"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    captured = {}
    def fake_run(cmd, **kw):
        captured.update(kw)
        return _ok_completed()
    monkeypatch.setattr(tsr.subprocess, "run", fake_run)
    tsr.run_preprocess_stage(["python.exe"], out)
    assert captured["errors"] == "replace"


def test_run_render_stage_passes_errors_replace(monkeypatch):
    captured = {}
    def fake_run(cmd, **kw):
        captured.update(kw)
        return _ok_completed()
    monkeypatch.setattr(tsr.subprocess, "run", fake_run)
    params = tsr.RenderParams(
        grains_dir=Path("/mnt/c/g"), bodies_dir=Path("/mnt/c/b"), output_dir=Path("/mnt/c/o"),
    )
    tsr.run_render_stage(params)
    assert captured["errors"] == "replace"


def _force_npz_route(monkeypatch):
    # these tests pin per-path preprocess semantics (npz route); the dump
    # route's shared preprocess has its own tests (test_dump_route_*)
    monkeypatch.setattr(tsr, "needs_npz", lambda paths: True)


def _pipeline_setup(monkeypatch, tmp_path, n_npz=2):
    _force_npz_route(monkeypatch)
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record())
    base = tmp_path / "DEMCSVs"
    grains = base / "batch" / "run_0001_grains_output"
    monkeypatch.setattr(tsr, "run_convert_stage", lambda rec, b: _write_fake_npz(grains, n_npz))
    return base, tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))


def _fake_preprocess_ok(cmd, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text("{}")
    return output_dir / "manifest.json"


def test_run_pipeline_partial_failure_continues_other_paths(monkeypatch, tmp_path):
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path)

    def fake_pre(cmd, output_dir):
        if output_dir.name == "run_0001_viz":
            raise tsr.PreprocessError("exited 1:\nqhull boom")
        return _fake_preprocess_ok(cmd, output_dir)
    monkeypatch.setattr(tsr, "run_preprocess_stage", fake_pre)

    rendered = []
    def fake_render(params):
        rendered.append((params.viz_path, params.manifest))
        _write_fake_pngs(params.output_dir, 2)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    form = tsr.RenderFormValues(paths=("instance_grains", "per_sphere", "composite"))
    result = tsr.run_pipeline(sim_params, form, base)

    assert result.stage == "partial"
    assert result.ok is False
    assert [r.name for r in result.paths] == ["per_sphere", "composite", "instance_grains"]
    assert [r.ok for r in result.paths] == [True, False, True]
    assert result.paths[1].stage == "preprocess"
    assert rendered == [
        ("per_sphere", None),
        ("instance", base / "batch" / "run_0001_viz_instance" / "manifest.json"),
    ]
    assert "per_sphere: ok, 2 frame(s)" in result.message
    assert "composite: failed at preprocess:" in result.message
    assert "qhull boom" in result.message
    assert "instance_grains: ok, 2 frame(s)" in result.message


def test_run_pipeline_spawn_failure_on_one_path_keeps_other_paths(monkeypatch, tmp_path):
    # Fix round 1: a spawn-level OSError (e.g. WIN_VENV_PYTHON missing) from
    # subprocess.run itself -- not a non-zero returncode -- must still be
    # contained to that one path's PathResult, not escape run_pipeline
    # entirely and discard the paths that already succeeded.
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path)

    def fake_subprocess_run(cmd, **kw):
        raise FileNotFoundError("no python.exe")
    monkeypatch.setattr(tsr.subprocess, "run", fake_subprocess_run)

    def fake_render(params):
        _write_fake_pngs(params.output_dir, 1)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    form = tsr.RenderFormValues(paths=("per_sphere", "composite"))
    result = tsr.run_pipeline(sim_params, form, base)

    assert result.stage == "partial"
    assert [r.ok for r in result.paths] == [True, False]
    assert result.paths[1].stage == "preprocess"
    assert "could not start" in result.message


def test_run_pipeline_all_paths_ok_is_done(monkeypatch, tmp_path):
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(tsr, "run_preprocess_stage", _fake_preprocess_ok)
    out_dirs = []
    def fake_render(params):
        out_dirs.append(params.output_dir.name)
        _write_fake_pngs(params.output_dir, 1)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    form = tsr.RenderFormValues(paths=tsr.RENDER_PATH_NAMES)
    result = tsr.run_pipeline(sim_params, form, base)

    assert result.stage == "done" and result.ok is True
    assert out_dirs == [
        "run_0001_render_per_sphere", "run_0001_render_composite",
        "run_0001_render_instance_grains", "run_0001_render_instance_static",
    ]


def test_run_pipeline_zero_pngs_is_render_failure(monkeypatch, tmp_path):
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(tsr, "run_render_stage", lambda params: _ok_completed())
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("per_sphere",)), base)
    assert result.ok is False
    assert result.paths[0].stage == "render"
    assert "render wrote no frames to" in result.message


def test_run_pipeline_rejects_empty_or_unknown_paths_before_sim(monkeypatch):
    sim_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_sim_stage", sim_mock)
    with pytest.raises(ValueError, match="render path"):
        tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(paths=()), Path("/x"))
    with pytest.raises(ValueError, match="bogus"):
        tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(paths=("bogus",)), Path("/x"))
    sim_mock.assert_not_called()


def test_run_pipeline_stage_callbacks_per_path(monkeypatch, tmp_path):
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path, n_npz=5)
    monkeypatch.setattr(tsr, "run_preprocess_stage", _fake_preprocess_ok)
    monkeypatch.setattr(
        tsr, "run_render_stage",
        lambda params: (_write_fake_pngs(params.output_dir, 1), _ok_completed())[1],
    )
    stages = []
    tsr.run_pipeline(
        sim_params, tsr.RenderFormValues(paths=("per_sphere", "composite"), max_frames=2), base,
        on_stage=lambda stage, **info: stages.append((stage, info)),
    )
    assert [s[0] for s in stages] == ["sim", "convert", "render", "preprocess", "render"]
    pre = stages[3][1]
    assert (pre["path"], pre["index"], pre["total"]) == ("composite", 2, 2)
    assert pre["output_dir"] == base / "batch" / "run_0001_viz"
    assert stages[4][1]["n_expected"] == 2
    assert stages[4][1]["output_dir"] == base / "batch" / "run_0001_render_composite"


def test_run_pipeline_summary_is_single_line_with_multiline_failure(monkeypatch, tmp_path):
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path)

    def fake_pre(cmd, output_dir):
        raise tsr.PreprocessError("exited 1:\n" + "\n".join(f"l{i}" for i in range(40)))
    monkeypatch.setattr(tsr, "run_preprocess_stage", fake_pre)

    def fake_render(params):
        _write_fake_pngs(params.output_dir, 1)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    form = tsr.RenderFormValues(paths=("per_sphere", "composite"))
    result = tsr.run_pipeline(sim_params, form, base)

    assert "\n" not in result.message
    assert "composite: failed at preprocess: exited 1: … l39" in result.message
    assert "per_sphere: ok" in result.message
    assert "l0" in result.paths[1].message


def test_format_path_result():
    ok = tsr.PathResult(name="composite", ok=True, stage="done", message="3 frame(s) in 5s to X")
    bad = tsr.PathResult(name="instance_static", ok=False, stage="preprocess", message="boom")
    assert tsr.format_path_result(ok) == "composite: ok, 3 frame(s) in 5s to X"
    assert tsr.format_path_result(bad) == "instance_static: failed at preprocess: boom"


def test_format_path_result_shortens_multiline_failure():
    bad = tsr.PathResult(
        name="composite", ok=False, stage="preprocess",
        message="exited 1:\nline a\n\nline b\n",
    )
    assert tsr.format_path_result(bad) == "composite: failed at preprocess: exited 1: … line b"


def test_format_live_progress_preprocess_with_label():
    text = tsr.format_live_progress(12.0, "preprocess", path_label="composite 2/3")
    assert text == "Preprocessing [composite 2/3]... 12s elapsed"


def test_format_live_progress_render_with_label():
    text = tsr.format_live_progress(3.0, "render", n_png=1, n_expected=4, path_label="per_sphere 1/2")
    assert text == "Rendering [per_sphere 1/2]... 3s elapsed, 1/4 frame(s)"


def test_from_windows_path():
    assert tsr.from_windows_path("C:/Users/x/a.obj") == Path("/mnt/c/Users/x/a.obj")
    assert tsr.from_windows_path("D:\\data\\b.obj") == Path("/mnt/d/data/b.obj")
    assert tsr.from_windows_path("/mnt/c/y") == Path("/mnt/c/y")


def test_pipeline_warning_only_for_per_sphere():
    assert tsr.pipeline_warning(5000, ("composite",)) == ""
    assert "5000" in tsr.pipeline_warning(5000, ("per_sphere", "composite"))
    assert tsr.pipeline_warning(500, ("per_sphere",)) == ""


# --- Task 6: render path form ----------------------------------------------

def _render_form_or_error(setup, *, venv_ok=True):
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            setup(app)
            try:
                return app._build_render_form()
            except ValueError as exc:
                return exc
    if venv_ok and not Path(tsr.WIN_VENV_PYTHON).is_file():
        # instance_grains (default) needs the venv python; any existing file stands in.
        with patch.object(tsr, "WIN_VENV_PYTHON", Path(__file__)):
            return asyncio.run(_scenario())
    return asyncio.run(_scenario())


def _tick(app, name, value=True):
    app.query_one(f"#{tsr.PATH_CHECKBOX_IDS[name]}", tsr.Checkbox).value = value


def test_path_checkbox_defaults():
    form = _render_form_or_error(lambda app: None)
    assert form.paths == ("instance_grains",)
    assert form.envelope_method == "hull"
    assert form.placeholder_obj == tsr.DEFAULT_PLACEHOLDER_OBJ
    assert form.n_points == 1_000_000


def test_no_path_ticked_is_error():
    err = _render_form_or_error(lambda app: _tick(app, "instance_grains", False))
    assert isinstance(err, ValueError)
    assert "render path" in str(err)


def test_composite_requires_venv_python(monkeypatch, tmp_path):
    monkeypatch.setattr(tsr, "WIN_VENV_PYTHON", tmp_path / "missing" / "python.exe")
    def setup(app):
        _tick(app, "instance_grains", False)  # default path also needs the venv; isolate composite
        _tick(app, "composite")

    err = _render_form_or_error(setup, venv_ok=False)
    assert isinstance(err, ValueError)
    assert "python.exe" in str(err)


def test_instance_grains_requires_venv_python(monkeypatch, tmp_path):
    monkeypatch.setattr(tsr, "WIN_VENV_PYTHON", tmp_path / "missing" / "python.exe")
    err = _render_form_or_error(lambda app: None, venv_ok=False)  # instance_grains is the default
    assert isinstance(err, ValueError)


def test_composite_with_venv_and_sdf(monkeypatch, tmp_path):
    exe = tmp_path / "python.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(tsr, "WIN_VENV_PYTHON", exe)

    def setup(app):
        _tick(app, "composite")
        app.query_one("#envelope-method", tsr.Select).value = "sdf"

    form = _render_form_or_error(setup)
    assert form.paths == ("composite", "instance_grains")
    assert form.envelope_method == "sdf"


def test_instance_static_missing_obj_is_error(tmp_path):
    def setup(app):
        _tick(app, "instance_static")
        app.query_one("#placeholder-obj", tsr.Input).value = str(tmp_path / "nope.obj")

    err = _render_form_or_error(setup)
    assert isinstance(err, ValueError)
    assert "placeholder_obj" in str(err)


def test_instance_static_nonpositive_n_points_is_error(tmp_path):
    obj = tmp_path / "a.obj"
    obj.write_text("v 0 0 0\n")

    def setup(app):
        _tick(app, "instance_static")
        app.query_one("#placeholder-obj", tsr.Input).value = str(obj)
        app.query_one("#n-points", tsr.Input).value = "0"

    assert isinstance(_render_form_or_error(setup), ValueError)


def test_instance_static_valid(tmp_path):
    obj = tmp_path / "a.obj"
    obj.write_text("v 0 0 0\n")

    def setup(app):
        _tick(app, "instance_grains", False)
        _tick(app, "instance_static")
        app.query_one("#placeholder-obj", tsr.Input).value = str(obj)
        app.query_one("#n-points", tsr.Input).value = "5000"

    form = _render_form_or_error(setup)
    assert form.paths == ("instance_static",)
    assert form.placeholder_obj == str(obj)
    assert form.n_points == 5000


def test_instance_static_relative_obj_is_error():
    def setup(app):
        _tick(app, "instance_static")
        app.query_one("#placeholder-obj", tsr.Input).value = "shapes/a.obj"

    err = _render_form_or_error(setup)
    assert isinstance(err, ValueError)
    assert "absolute" in str(err)


def test_unticked_path_fields_are_not_validated():
    def setup(app):
        app.query_one("#n-points", tsr.Input).value = "garbage"
        app.query_one("#placeholder-obj", tsr.Input).value = "C:/nope.obj"

    form = _render_form_or_error(setup)
    assert form.paths == ("instance_grains",)


def test_envelope_method_select_disallows_blank():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            select = app.query_one("#envelope-method", tsr.Select)
            return select._allow_blank, select.value

    allow_blank, value = asyncio.run(_scenario())
    assert allow_blank is False
    assert value == "hull"


def test_set_pipeline_stage_preprocess_shows_path_label():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app._pipeline_running = True
            app._pipeline_start = time.monotonic()
            app._pipeline_dry_run = False
            app._pipeline_warning = ""
            app._set_pipeline_stage(
                "preprocess",
                {"path": "composite", "index": 2, "total": 3, "output_dir": Path("/tmp/x_viz")},
            )
            return str(app.query_one("#status", tsr.Static).content)

    text = asyncio.run(_scenario())
    assert text.startswith("Preprocessing [composite 2/3]...")


def test_set_pipeline_stage_render_tracks_path_output_dir(tmp_path):
    render_dir = tmp_path / "run_0001_render_composite"
    render_dir.mkdir()
    (render_dir / "frame_0001.png").write_bytes(b"")

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app._pipeline_running = True
            app._pipeline_start = time.monotonic()
            app._pipeline_dry_run = False
            app._pipeline_warning = ""
            app._set_pipeline_stage(
                "render",
                {"n_expected": 3, "path": "composite", "index": 2, "total": 3, "output_dir": render_dir},
            )
            return app._pipeline_render_dir, str(app.query_one("#status", tsr.Static).content)

    tracked, text = asyncio.run(_scenario())
    assert tracked == render_dir
    assert "Rendering [composite 2/3]..." in text
    assert "1/3 frame(s)" in text


def test_report_result_partial_is_error_status():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app._report_result(tsr.PipelineResult(
                stage="partial", ok=False,
                message="converted 3 frame(s); per_sphere: ok, 3 frame(s) in 1s to X; "
                        "composite: failed at preprocess: boom",
            ))
            s = app.query_one("#status", tsr.Static)
            return str(s.content), s.has_class("err")

    text, is_err = asyncio.run(_scenario())
    assert "Finished with failures:" in text
    assert "composite: failed at preprocess: boom" in text
    assert is_err is True


# --- raw dumps cleaned after a verified convert (default on) --------------------

def _pipeline_with_run_dir(monkeypatch, tmp_path, npz_frames=1):
    run_dir = tmp_path / "sobol_mass_runs" / "batch" / "run_0001"
    run_dir.mkdir(parents=True)
    for name in ("sobol_00000", "sobol_00001", "sobolSink0012N01.ev", "sobol.setup", "sobol.in"):
        (run_dir / name).write_text("x")
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record(str(run_dir)))
    base_output_dir = tmp_path / "DEMCSVs"
    grains_dir = base_output_dir / "batch" / "run_0001_grains_output"
    monkeypatch.setattr(tsr, "run_convert_stage",
                        lambda rec, out: _write_fake_npz(grains_dir, npz_frames) if npz_frames else None)

    def fake_render(params):
        _write_fake_pngs(params.output_dir, 1)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)
    return run_dir, base_output_dir


def test_sim_params_keep_dumps_default_off():
    assert tsr.SimParams().keep_dumps is False


@pytest.mark.parametrize("keep", [False, True])
def test_run_pipeline_cleans_raw_dumps_after_convert(monkeypatch, tmp_path, keep):
    run_dir, base_output_dir = _pipeline_with_run_dir(monkeypatch, tmp_path)
    sim_params = tsr.SimParams(output_root=run_dir.parent, keep_dumps=keep)
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("per_sphere",)), base_output_dir)
    assert result.ok is True
    assert (run_dir / "sobol_00000").exists() is keep
    assert (run_dir / "sobolSink0012N01.ev").exists() is keep
    assert (run_dir / "sobol.setup").exists()


def test_run_pipeline_keeps_dumps_when_convert_produced_nothing(monkeypatch, tmp_path):
    run_dir, base_output_dir = _pipeline_with_run_dir(monkeypatch, tmp_path, npz_frames=0)
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), tsr.RenderFormValues(paths=("per_sphere",)), base_output_dir)
    assert result.ok is False
    assert (run_dir / "sobol_00000").exists()


# --- settled body / hyperbola encounter (Task 10) --------------------------


def test_run_sim_stage_builds_settled_body_and_uses_earth_1(tmp_path, monkeypatch):
    import tui_sim_render as t
    from run_mass_sobol_phantom import RunSample
    seen = {}

    def fake_attach(samples, **kw):
        samples[0].settled_body_dir = str(tmp_path / "body")
        seen["attach"] = kw
        return 1

    def fake_run_one_case(**kw):
        seen["run"] = kw
        return "record"

    monkeypatch.setattr(t, "preflight", lambda a, b, o: (tmp_path / "s", tmp_path / "i",
                                                         tmp_path / "ps", tmp_path / "p"))
    monkeypatch.setattr(t, "attach_settled_bodies", fake_attach)
    monkeypatch.setattr(t, "run_one_case", fake_run_one_case)
    import settled_body as sb
    monkeypatch.setattr(sb, "resolve_dem_tools",
                        lambda d: sb.DemTools(tmp_path / "m", tmp_path / "f", tmp_path / "a"))
    sample = RunSample(use_dem=True, np_apophis=300, body_source="settled", encounter="hyperbola",
                       flyby_rp_km=38000.0, flyby_vinf_kms=5.9, flyby_start_sep_km=1e5,
                       flyby_perturber_earth_masses=1.0, tmax_hours=8.0, dtmax_hours=0.25)
    p = t.SimParams(output_root=tmp_path / "out", sample=sample, shape_file=None)
    assert t.run_sim_stage(p) == "record"
    assert seen["run"]["earth_sink_id"] == 1
    assert seen["run"]["phantomflyby_bin"] == tmp_path / "f"
    assert seen["run"]["sample"].settled_body_dir == str(tmp_path / "body")
    assert seen["attach"]["cache_root"] == (tmp_path / "out").parent / "settled_bodies"


# --- _build_sim_params hyperbola validation (Task 10 fix round 1) ----------


def test_flyby_and_time_inputs_are_prefilled():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            return {
                "tmax": app.query_one("#tmax-hours", tsr.Input).value,
                "dtmax": app.query_one("#dtmax-hours", tsr.Input).value,
                "rp": app.query_one("#flyby-rp", tsr.Input).value,
                "vinf": app.query_one("#flyby-vinf", tsr.Input).value,
            }

    values = asyncio.run(_scenario())

    assert values == {
        "tmax": tsr.DEFAULT_TMAX_HOURS,
        "dtmax": tsr.DEFAULT_DTMAX_HOURS,
        "rp": tsr.DEFAULT_FLYBY_RP_KM,
        "vinf": tsr.DEFAULT_FLYBY_VINF_KMS,
    }


def test_hyperbola_accepts_prefilled_flyby_and_time_defaults():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#body-source", tsr.Select).value = "settled"
            app.query_one("#encounter", tsr.Select).value = "hyperbola"
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.flyby_rp_km == 38000.0
    assert params.sample.flyby_vinf_kms == 5.9
    assert params.sample.tmax_hours == 108.0
    assert params.sample.dtmax_hours == 0.5


def _set_hyperbola_form(app, *, rp="38000", vinf="5.9", sep="1e5", tmax="8", dtmax="0.5"):
    app.query_one("#body-source", tsr.Select).value = "settled"
    app.query_one("#encounter", tsr.Select).value = "hyperbola"
    app.query_one("#flyby-rp", tsr.Input).value = rp
    app.query_one("#flyby-vinf", tsr.Input).value = vinf
    app.query_one("#flyby-sep", tsr.Input).value = sep
    app.query_one("#tmax-hours", tsr.Input).value = tmax
    app.query_one("#dtmax-hours", tsr.Input).value = dtmax


def test_build_sim_params_hyperbola_rejects_too_short_tmax():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            _set_hyperbola_form(app, rp="38000", vinf="5.9", sep="4e5", tmax="1")
            with pytest.raises(ValueError, match="tmax"):
                app._build_sim_params(False)

    asyncio.run(_scenario())


def test_build_sim_params_hyperbola_rejects_non_positive_rp():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            _set_hyperbola_form(app, rp="-1000")
            with pytest.raises(ValueError, match="flyby_rp_km must be > 0"):
                app._build_sim_params(False)

    asyncio.run(_scenario())


def test_build_sim_params_hyperbola_rejects_rp_inside_earth():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            _set_hyperbola_form(app, rp="5000")  # < R_EARTH_KM (6371)
            with pytest.raises(ValueError, match="R_EARTH_KM|Earth"):
                app._build_sim_params(False)

    asyncio.run(_scenario())


def test_build_sim_params_hyperbola_rejects_rp_past_start_sep():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            _set_hyperbola_form(app, rp="150000", sep="100000")
            with pytest.raises(ValueError, match="start_sep|start-sep"):
                app._build_sim_params(False)

    asyncio.run(_scenario())


def test_build_sim_params_hyperbola_step5_settings_pass():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            _set_hyperbola_form(app, rp="38000", vinf="5.9", sep="1e5", tmax="8", dtmax="0.5")
            return app._build_sim_params(False)

    params = asyncio.run(_scenario())

    assert params.sample.body_source == "settled"
    assert params.sample.encounter == "hyperbola"
    assert params.sample.flyby_rp_km == 38000.0
    assert params.sample.flyby_vinf_kms == 5.9
    assert params.sample.flyby_start_sep_km == 1e5
    assert params.sample.tmax_hours == 8.0


# --- settled body on the ephemeris (packing_file) ------------------------


def test_build_sim_params_settled_ephemeris_allowed_and_drops_flyby():
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app.query_one("#body-source", tsr.Select).value = "settled"
            app.query_one("#encounter", tsr.Select).value = "ephemeris"
            app.query_one("#shape-file", tsr.Input).value = str(tsr.resolve_default_shape_file())
            return app._build_sim_params(False)

    p = asyncio.run(_scenario())
    assert (p.sample.body_source, p.sample.encounter) == ("settled", "ephemeris")
    assert p.sample.flyby_rp_km is None and p.sample.flyby_vinf_kms is None
    assert p.sample.flyby_start_sep_km is None
    assert p.sample.use_shape_crop is None  # the body carries its own shape
    assert p.earth_sink_id == tsr.EARTH_SINK_ID_DEFAULT


def test_run_sim_stage_settled_ephemeris_keeps_earth_4_and_no_flyby(tmp_path, monkeypatch):
    import tui_sim_render as t
    from run_mass_sobol_phantom import RunSample
    seen = {}

    def fake_attach(samples, **kw):
        samples[0].settled_body_dir = str(tmp_path / "body")
        return 1

    monkeypatch.setattr(t, "preflight", lambda a, b, o: (tmp_path / "s", tmp_path / "i",
                                                         tmp_path / "ps", tmp_path / "p"))
    monkeypatch.setattr(t, "attach_settled_bodies", fake_attach)
    monkeypatch.setattr(t, "run_one_case", lambda **kw: seen.setdefault("run", kw) and "record")
    import settled_body as sb
    monkeypatch.setattr(sb, "resolve_dem_tools", lambda d: pytest.fail("phantomflyby not needed"))
    sample = RunSample(use_dem=True, np_apophis=300, body_source="settled", encounter="ephemeris")
    p = t.SimParams(output_root=tmp_path / "out", sample=sample, shape_file=None)
    t.run_sim_stage(p)
    assert seen["run"]["earth_sink_id"] == t.EARTH_SINK_ID_DEFAULT
    assert seen["run"]["phantomflyby_bin"] is None


# --- side-by-side compare stage ------------------------------------------

def test_render_form_compare_defaults_off():
    assert tsr.RenderFormValues().compare is False


def test_compare_output_dir():
    assert tsr._compare_output_dir(Path("/b/batch"), "run_0001") == Path("/b/batch/run_0001_compare")


def test_run_pipeline_compare_needs_two_paths_before_sim(monkeypatch):
    sim_mock = MagicMock()
    monkeypatch.setattr(tsr, "run_sim_stage", sim_mock)
    with pytest.raises(ValueError, match="compare"):
        tsr.run_pipeline(
            tsr.SimParams(), tsr.RenderFormValues(paths=("composite",), compare=True), Path("/x"),
        )
    sim_mock.assert_not_called()


def _compare_pipeline(monkeypatch, tmp_path, fail_paths=(), compare_exc=None):
    base, sim_params = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(tsr, "run_preprocess_stage", _fake_preprocess_ok)

    def fake_render(params):
        if params.output_dir.name.split("_render_", 1)[1] in fail_paths:
            raise tsr.RenderError("blender exited 1:\nboom")
        _write_fake_pngs(params.output_dir, 2)
        return _ok_completed()
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    calls = []
    def fake_compare(render_dirs, output_dir, fps=24, encode_video=False):
        calls.append({"render_dirs": list(render_dirs), "output_dir": output_dir,
                      "fps": fps, "encode_video": encode_video})
        if compare_exc is not None:
            raise compare_exc
        return tsr.CompareResult(output_dir, 2, [label for label, _ in render_dirs], [])
    monkeypatch.setattr(tsr, "run_compare_stage", fake_compare)
    return base, sim_params, calls


def test_run_pipeline_compare_composite_and_point_cloud(monkeypatch, tmp_path):
    base, sim_params, calls = _compare_pipeline(monkeypatch, tmp_path)
    form = tsr.RenderFormValues(paths=("instance_grains", "composite"), compare=True)
    result = tsr.run_pipeline(sim_params, form, base)

    assert result.stage == "done" and result.ok is True
    assert len(calls) == 1
    batch = base / "batch"
    assert calls[0]["render_dirs"] == [
        ("composite", batch / "run_0001_render_composite"),
        ("instance_grains", batch / "run_0001_render_instance_grains"),
    ]
    assert calls[0]["output_dir"] == batch / "run_0001_compare"
    assert calls[0]["fps"] == 24 and calls[0]["encode_video"] is False
    assert result.compare is not None and result.compare.n_frames == 2
    assert "compare: ok, 2 frame(s) (composite | instance_grains)" in result.message


def test_run_pipeline_compare_uses_video_fps_and_encode(monkeypatch, tmp_path):
    base, sim_params, calls = _compare_pipeline(monkeypatch, tmp_path)
    form = tsr.RenderFormValues(
        paths=("per_sphere", "composite"), compare=True, fps=24, video_fps=30, encode_video=True,
    )
    tsr.run_pipeline(sim_params, form, base)
    assert calls[0]["fps"] == 30 and calls[0]["encode_video"] is True


def test_run_pipeline_compare_off_never_calls_compare(monkeypatch, tmp_path):
    base, sim_params, calls = _compare_pipeline(monkeypatch, tmp_path)
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("per_sphere", "composite")), base)
    assert calls == []
    assert result.compare is None
    assert "compare:" not in result.message  # tmp_path itself contains "compare"


def test_run_pipeline_compare_runs_on_surviving_paths(monkeypatch, tmp_path):
    base, sim_params, calls = _compare_pipeline(monkeypatch, tmp_path, fail_paths=("composite",))
    form = tsr.RenderFormValues(paths=("per_sphere", "composite", "instance_grains"), compare=True)
    result = tsr.run_pipeline(sim_params, form, base)
    assert [label for label, _ in calls[0]["render_dirs"]] == ["per_sphere", "instance_grains"]
    assert result.stage == "partial" and result.ok is False
    assert "compare: ok" in result.message


def test_run_pipeline_compare_skipped_when_one_path_left(monkeypatch, tmp_path):
    base, sim_params, calls = _compare_pipeline(monkeypatch, tmp_path, fail_paths=("composite",))
    form = tsr.RenderFormValues(paths=("per_sphere", "composite"), compare=True)
    result = tsr.run_pipeline(sim_params, form, base)
    assert calls == []
    assert result.stage == "partial"
    assert "compare: skipped, only 1 path(s) rendered" in result.message


def test_run_pipeline_compare_failure_is_partial(monkeypatch, tmp_path):
    base, sim_params, calls = _compare_pipeline(
        monkeypatch, tmp_path, compare_exc=tsr.CompareError("ffmpeg not found on PATH"),
    )
    form = tsr.RenderFormValues(paths=("per_sphere", "composite"), compare=True)
    result = tsr.run_pipeline(sim_params, form, base)
    assert [r.ok for r in result.paths] == [True, True]
    assert result.stage == "partial" and result.ok is False
    assert result.compare is None
    assert "compare: failed: ffmpeg not found on PATH" in result.message


def test_run_pipeline_compare_stage_callback(monkeypatch, tmp_path):
    base, sim_params, _calls = _compare_pipeline(monkeypatch, tmp_path)
    stages = []
    tsr.run_pipeline(
        sim_params, tsr.RenderFormValues(paths=("per_sphere", "composite"), compare=True), base,
        on_stage=lambda stage, **info: stages.append((stage, info)),
    )
    assert stages[-1][0] == "compare"
    assert stages[-1][1]["output_dir"] == base / "batch" / "run_0001_compare"
    assert stages[-1][1]["n_expected"] == 2


def test_format_live_progress_compare():
    text = tsr.format_live_progress(12.0, "compare", n_png=3, n_expected=10)
    assert text == "Comparing... 12s elapsed, 3/10 frame(s)"


def test_tick_progress_compare_stage_counts_compare_pngs(tmp_path):
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            out = tmp_path / "run_0001_compare"
            out.mkdir()
            (out / "compare_0001.png").write_bytes(b"")
            (out / "frame_0001.png").write_bytes(b"")  # must not be counted
            app._pipeline_start = time.monotonic()
            app._pipeline_warning = ""
            app._pipeline_dry_run = False
            app._pipeline_running = False  # stop _set_pipeline_stage from ticking early
            app._set_pipeline_stage("compare", {"n_expected": 5, "output_dir": out})
            app._tick_progress()
            return str(app.query_one("#status", tsr.Static).content)

    text = asyncio.run(_scenario())
    assert "Comparing..." in text
    assert "1/5 frame(s)" in text


def test_form_compare_with_one_path_is_error():
    err = _render_form_or_error(
        lambda app: setattr(app.query_one("#compare", tsr.Checkbox), "value", True),
    )
    assert isinstance(err, ValueError)
    assert "compare" in str(err)


def test_form_compare_with_two_paths(monkeypatch, tmp_path):
    exe = tmp_path / "python.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(tsr, "WIN_VENV_PYTHON", exe)

    def setup(app):
        _tick(app, "per_sphere", False)
        _tick(app, "composite")
        _tick(app, "instance_grains")
        app.query_one("#compare", tsr.Checkbox).value = True

    form = _render_form_or_error(setup)
    assert form.paths == ("composite", "instance_grains")
    assert form.compare is True


def test_run_pipeline_compare_unexpected_error_is_partial(monkeypatch, tmp_path):
    # e.g. OneDrive lock on a stale compare_*.png, or a full disk on mkdir
    base, sim_params, _calls = _compare_pipeline(
        monkeypatch, tmp_path, compare_exc=PermissionError("compare_0003.png is locked"),
    )
    form = tsr.RenderFormValues(paths=("per_sphere", "composite"), compare=True)
    result = tsr.run_pipeline(sim_params, form, base)
    assert [r.ok for r in result.paths] == [True, True]
    assert result.stage == "partial" and result.ok is False
    assert "compare: failed: PermissionError: compare_0003.png is locked" in result.message


# --- dump-direct route: convert skipped, one shared dump preprocess -----------

@pytest.mark.parametrize("paths,expected", [
    (("per_sphere",), True),
    (("instance_static",), True),
    (("composite", "per_sphere"), True),
    (("composite",), False),
    (("composite", "instance_grains"), False),
])
def test_needs_npz(paths, expected):
    assert tsr.needs_npz(paths) is expected


def _ctx(tmp_path, **form):
    return tsr.PathContext(
        grains_dir=tmp_path / "g", bodies_dir=tmp_path / "b", batch_dir=tmp_path / "batch",
        run_name="run_0001", render_form=tsr.RenderFormValues(**form), n_expected_frames=2,
    )


def test_build_dump_preprocess_command_both_targets(tmp_path):
    ctx = _ctx(tmp_path, paths=("composite", "instance_grains"), envelope_method="sdf", max_frames=3)
    targets = {"composite": tmp_path / "v", "instance_grains": tmp_path / "vi"}
    rel_run = Path(os.path.relpath(tmp_path / "run_0001"))
    cmd = tsr.build_dump_preprocess_command(ctx, rel_run, targets)
    assert cmd[0] == str(tsr.WIN_VENV_PYTHON)
    assert cmd[1].endswith("viz_preprocess_dump.py")
    assert cmd[cmd.index("--run-dir") + 1] == tsr.to_windows_path((tmp_path / "run_0001").resolve())
    assert cmd[cmd.index("--composite-dir") + 1] == tsr.to_windows_path(tmp_path / "v")
    assert cmd[cmd.index("--instance-dir") + 1] == tsr.to_windows_path(tmp_path / "vi")
    assert cmd[cmd.index("--envelope-method") + 1] == "sdf"
    assert cmd[cmd.index("--max-frames") + 1] == "3"


def test_build_dump_preprocess_command_instance_only_has_no_envelope(tmp_path):
    ctx = _ctx(tmp_path, paths=("instance_grains",))
    cmd = tsr.build_dump_preprocess_command(ctx, tmp_path, {"instance_grains": tmp_path / "vi"})
    assert "--composite-dir" not in cmd and "--envelope-method" not in cmd


def _fake_run(monkeypatch, stdout, rc, write=()):
    def run(cmd, **kw):
        for d in write:
            d.mkdir(parents=True, exist_ok=True)
            (d / "manifest.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
    monkeypatch.setattr(tsr.subprocess, "run", run)


def test_run_dump_preprocess_stage_splits_results(tmp_path, monkeypatch):
    v, vi = tmp_path / "v", tmp_path / "vi"
    vi.mkdir()
    (vi / "manifest.json").write_text("stale")
    _fake_run(monkeypatch, "OK composite -> x\nFAILED instance_grains: GrainReappeared: boom\n", 1, write=(v,))
    res = tsr.run_dump_preprocess_stage(["py"], {"composite": v, "instance_grains": vi})
    assert res["composite"] == v / "manifest.json"
    assert isinstance(res["instance_grains"], tsr.PreprocessError)
    assert str(res["instance_grains"]) == "GrainReappeared: boom"


def test_run_dump_preprocess_stage_crash_gives_tail(tmp_path, monkeypatch):
    _fake_run(monkeypatch, "Traceback ...\nModuleNotFoundError: sarracen\n", 1)
    res = tsr.run_dump_preprocess_stage(["py"], {"composite": tmp_path / "v"})
    assert "exited 1" in str(res["composite"]) and "sarracen" in str(res["composite"])


def test_run_dump_preprocess_stage_spawn_error(tmp_path, monkeypatch):
    def boom(cmd, **kw):
        raise OSError("no python")
    monkeypatch.setattr(tsr.subprocess, "run", boom)
    res = tsr.run_dump_preprocess_stage(["py"], {"composite": tmp_path / "v", "instance_grains": tmp_path / "vi"})
    assert all("could not start py: no python" in str(e) for e in res.values())


def _dump_route(monkeypatch, tmp_path, fail=()):
    run_dir, base = _pipeline_with_run_dir(monkeypatch, tmp_path)
    convert = MagicMock()
    monkeypatch.setattr(tsr, "run_convert_stage", convert)
    per_path = MagicMock()
    monkeypatch.setattr(tsr, "run_preprocess_stage", per_path)
    cmds = []

    def fake_dump(cmd, targets):
        cmds.append(cmd)
        out = {}
        for name, d in targets.items():
            if name in fail:
                out[name] = tsr.PreprocessError(f"{name} boom")
            else:
                out[name] = _fake_preprocess_ok(cmd, d)
        return out
    monkeypatch.setattr(tsr, "run_dump_preprocess_stage", fake_dump)
    return run_dir, base, convert, per_path, cmds


def test_dump_route_one_shared_preprocess(monkeypatch, tmp_path):
    run_dir, base, convert, per_path, cmds = _dump_route(monkeypatch, tmp_path)
    form = tsr.RenderFormValues(paths=("composite", "instance_grains"))
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), form, base)
    assert result.ok is True and result.stage == "done"
    convert.assert_not_called()
    per_path.assert_not_called()
    assert len(cmds) == 1
    assert "--composite-dir" in cmds[0] and "--instance-dir" in cmds[0]
    assert "read 2 dump(s) directly (convert skipped)" in result.message


def test_dump_route_one_failed_target_other_renders(monkeypatch, tmp_path):
    run_dir, base, *_ = _dump_route(monkeypatch, tmp_path, fail=("instance_grains",))
    form = tsr.RenderFormValues(paths=("composite", "instance_grains"))
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), form, base)
    assert result.stage == "partial"
    by = {r.name: r for r in result.paths}
    assert by["composite"].ok is True
    assert by["instance_grains"].ok is False and by["instance_grains"].stage == "preprocess"
    assert "instance_grains boom" in by["instance_grains"].message


def test_npz_route_kept_when_per_sphere_ticked(monkeypatch, tmp_path):
    run_dir, base = _pipeline_with_run_dir(monkeypatch, tmp_path)
    dump = MagicMock()
    monkeypatch.setattr(tsr, "run_dump_preprocess_stage", dump)
    cmds = []
    monkeypatch.setattr(tsr, "run_preprocess_stage",
                        lambda cmd, out: cmds.append(cmd) or _fake_preprocess_ok(cmd, out))
    form = tsr.RenderFormValues(paths=("per_sphere", "composite"))
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), form, base)
    assert result.ok is True
    dump.assert_not_called()
    assert "--grains-dir" in cmds[0]
    assert "converted 1 frame(s)" in result.message


@pytest.mark.parametrize("keep", [False, True])
def test_dump_route_cleans_dumps_after_render(monkeypatch, tmp_path, keep):
    run_dir, base, *_ = _dump_route(monkeypatch, tmp_path)
    sim_params = tsr.SimParams(output_root=run_dir.parent, keep_dumps=keep)
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(paths=("composite",)), base)
    assert result.ok is True
    assert (run_dir / "sobol_00000").exists() is keep
    assert (run_dir / "sobol.setup").exists()


def test_dump_route_keeps_dumps_when_every_path_failed(monkeypatch, tmp_path):
    run_dir, base, *_ = _dump_route(monkeypatch, tmp_path, fail=("composite", "instance_grains"))
    form = tsr.RenderFormValues(paths=("composite", "instance_grains"))
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), form, base)
    assert result.ok is False and result.stage == "partial"
    assert (run_dir / "sobol_00000").exists()
    assert f"dumps kept in {run_dir} (every render path failed)" in result.message


def test_dump_route_no_dumps_fails_before_preprocess(monkeypatch, tmp_path):
    run_dir, base, _c, _p, cmds = _dump_route(monkeypatch, tmp_path)
    for f in run_dir.glob("sobol_[0-9]*"):
        f.unlink()
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent),
                              tsr.RenderFormValues(paths=("composite",)), base)
    assert result.ok is False and result.stage == "convert"
    assert "no sobol_[0-9]* dumps" in result.message
    assert cmds == []


def test_dump_route_stages(monkeypatch, tmp_path):
    run_dir, base, *_ = _dump_route(monkeypatch, tmp_path)
    stages = []
    form = tsr.RenderFormValues(paths=("composite", "instance_grains"), max_frames=1)
    tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), form, base,
                     on_stage=lambda s, **i: stages.append((s, i)))
    names = [s for s, _ in stages]
    assert "convert" not in names and "preprocess" not in names
    pre = dict(stages)["preprocess_dumps"]
    assert pre["n_expected"] == 1
    assert pre["output_dir"] == base / "batch" / "run_0001_viz" / "grains"
    assert ("render", 1) in [(s, i.get("n_expected")) for s, i in stages]


def test_dump_route_instance_only_progress_has_no_count(monkeypatch, tmp_path):
    run_dir, base, *_ = _dump_route(monkeypatch, tmp_path)
    stages = []
    tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent),
                     tsr.RenderFormValues(paths=("instance_grains",)), base,
                     on_stage=lambda s, **i: stages.append((s, i)))
    pre = dict(stages)["preprocess_dumps"]
    assert pre["n_expected"] is None and pre["output_dir"] is None


def test_format_live_progress_preprocess_dumps():
    assert tsr.format_live_progress(5, "preprocess_dumps", n_png=3, n_expected=10) == \
        "Preprocessing dumps... 5s elapsed, 3/10 dump(s)"
    assert tsr.format_live_progress(5, "preprocess_dumps", n_png=0, n_expected=None) == \
        "Preprocessing dumps... 5s elapsed"


def test_set_pipeline_stage_preprocess_dumps_counts_camera_files(tmp_path):
    cam = tmp_path / "run_0001_viz" / "grains"
    cam.mkdir(parents=True)
    for i in range(2):
        (cam / f"{i:05d}.npz").write_bytes(b"")

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            app._pipeline_running = True
            app._pipeline_start = time.monotonic()
            app._pipeline_dry_run = False
            app._pipeline_warning = ""
            app._pipeline_path_label = "composite 1/2"
            app._set_pipeline_stage("preprocess_dumps", {"n_expected": 5, "output_dir": cam})
            return str(app.query_one("#status", tsr.Static).content)

    text = asyncio.run(_scenario())
    assert "Preprocessing dumps..." in text and "2/5 dump(s)" in text


def test_dump_route_keeps_dumps_when_a_path_failed_at_preprocess(monkeypatch, tmp_path):
    # the dumps are the only grain source: a retry of the failed path needs them
    run_dir, base, *_ = _dump_route(monkeypatch, tmp_path, fail=("instance_grains",))
    form = tsr.RenderFormValues(paths=("composite", "instance_grains"))
    result = tsr.run_pipeline(tsr.SimParams(output_root=run_dir.parent), form, base)
    assert result.stage == "partial"
    assert (run_dir / "sobol_00000").exists()
    assert f"dumps kept in {run_dir} (instance_grains failed at preprocess)" in result.message


def test_run_dump_preprocess_stage_failed_line_beats_existing_manifest(tmp_path, monkeypatch):
    # finish() can die mid-write (disk full, OneDrive lock) leaving a truncated manifest
    v = tmp_path / "v"
    _fake_run(monkeypatch, "FAILED composite: OSError: disk full\n", 1, write=(v,))
    res = tsr.run_dump_preprocess_stage(["py"], {"composite": v})
    assert isinstance(res["composite"], tsr.PreprocessError)
    assert str(res["composite"]) == "OSError: disk full"
