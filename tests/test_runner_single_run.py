"""Runner: one all-fixed run (num_samples 1) and SOBOL_REPORT_BATCH_DIR."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import params_schema as ps
import run_mass_sobol_phantom as runner

FIXED = ["--use-dem-fixed", "true", "--np-apophis", "300", "--spin-period-fixed", "30"]


def test_single_run_needs_no_dimension():
    runner.validate_args(runner.parse_args(FIXED + ["--num-samples", "1"]))


def test_zero_dims_still_rejected_for_sweeps():
    with pytest.raises(ValueError, match="No varying dimensions"):
        runner.validate_args(runner.parse_args(FIXED))  # num_samples defaults to 8


def test_zero_dims_still_rejected_with_saltelli():
    with pytest.raises(ValueError):
        runner.validate_args(runner.parse_args(FIXED + ["--num-samples", "1", "--saltelli-n", "4"]))


def test_single_mode_form_passes_check():
    fv = ps.from_argv(FIXED)
    assert ps.check(ps.to_argv(ps.apply_mode(fv, "single"))).error is None


def test_report_batch_dir(tmp_path, monkeypatch):
    report = tmp_path / "batch_dir.txt"
    monkeypatch.setenv(runner.REPORT_BATCH_DIR_ENV, str(report))
    monkeypatch.setattr(sys, "argv", ["run_mass_sobol_phantom.py", "--output-root", str(tmp_path / "out"),
                                      *FIXED, "--num-samples", "1"])

    def stop(*_a, **_k):
        raise RuntimeError("stop after report")

    monkeypatch.setattr(runner, "preflight", stop)
    assert runner.main() == 1
    batch = Path(report.read_text(encoding="utf-8").strip())
    assert batch.parent == (tmp_path / "out").resolve()
    assert batch.name.startswith("sobol_")


def test_no_report_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(runner.REPORT_BATCH_DIR_ENV, raising=False)
    runner._report_batch_dir(tmp_path / "x")  # no file named anywhere, no error
    assert list(tmp_path.iterdir()) == []
