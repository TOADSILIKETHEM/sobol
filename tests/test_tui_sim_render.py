"""Tests for tui_sim_render.py (sim+render pipeline TUI)."""
import argparse
import subprocess
import sys
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
    assert r.samples == 128
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


def test_run_pipeline_stops_after_failed_render(monkeypatch):
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: _ok_record())
    monkeypatch.setattr(tsr, "run_convert_stage", lambda record, base_output_dir: None)

    def boom_render(params):
        raise tsr.RenderError("blender exited 1:\nsome error")
    monkeypatch.setattr(tsr, "run_render_stage", boom_render)

    result = tsr.run_pipeline(tsr.SimParams(), tsr.RenderFormValues(), Path("/mnt/c/.../DEMCSVs"))

    assert result.stage == "render"
    assert result.ok is False
    assert "some error" in result.message


def test_run_pipeline_full_success(monkeypatch):
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)
    monkeypatch.setattr(tsr, "run_convert_stage", lambda rec, base_output_dir: None)

    captured = {}
    def fake_render(params):
        captured["params"] = params
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    render_form = tsr.RenderFormValues(resolution="640x360", samples=8, camera_mode="grain_only")
    result = tsr.run_pipeline(sim_params, render_form, Path("/mnt/c/.../DEMCSVs"))

    assert result.stage == "done"
    assert result.ok is True
    rp = captured["params"]
    assert rp.grains_dir == Path("/mnt/c/.../DEMCSVs") / "batch" / "run_0001_grains_output"
    assert rp.bodies_dir == Path("/mnt/c/.../DEMCSVs") / "batch" / "run_0001_bodies_output"
    assert rp.output_dir == Path("/mnt/c/.../DEMCSVs") / "batch" / "run_0001_render"
    assert rp.resolution == "640x360"
    assert rp.samples == 8
    assert rp.camera_mode == "grain_only"


def test_run_pipeline_success_message_counts_rendered_frames(monkeypatch, tmp_path):
    # Ruling: run_pipeline's success message counts rendered PNGs in
    # output_dir rather than a fixed "rendered to <dir>" string.
    record = _ok_record()
    monkeypatch.setattr(tsr, "run_sim_stage", lambda params: record)
    monkeypatch.setattr(tsr, "run_convert_stage", lambda rec, base_output_dir: None)

    def fake_render(params):
        params.output_dir.mkdir(parents=True, exist_ok=True)
        (params.output_dir / "frame_0001.png").write_bytes(b"")
        (params.output_dir / "frame_0002.png").write_bytes(b"")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tsr, "run_render_stage", fake_render)

    base_output_dir = tmp_path / "DEMCSVs"
    sim_params = tsr.SimParams(output_root=Path("sobol_mass_runs/batch"))
    result = tsr.run_pipeline(sim_params, tsr.RenderFormValues(), base_output_dir)

    output_dir = base_output_dir / "batch" / "run_0001_render"
    assert result.message == f"rendered 2 frame(s) to {output_dir}"
