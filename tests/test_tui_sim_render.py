"""Tests for tui_sim_render.py (sim+render pipeline TUI)."""
import argparse
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
