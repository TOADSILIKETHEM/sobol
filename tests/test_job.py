"""job.py against a stub runner: done / partial / failed / cancelled, run selection, renders."""
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import job
import jobstore as js

STUB = textwrap.dedent("""
    import os, pathlib, sys, time
    batch = pathlib.Path(os.environ["STUB_BATCH"])
    batch.mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["SOBOL_REPORT_BATCH_DIR"]).write_text(str(batch) + "\\n")
    rows = [r.split(":") for r in os.environ.get("STUB_ROWS", "").split()]
    (batch / "sobol_mass_samples.csv").write_text("run_id\\n" + "".join(f"{i}\\n" for i, _ in rows))
    (batch / "sobol_mass_outputs.csv").write_text(
        "run_id,run_dir,status\\n" + "".join(f"{i},{batch}/run_{int(i):04d},{s}\\n" for i, s in rows))
    time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
    sys.exit(int(os.environ.get("STUB_RC", "0")))
""")


@pytest.fixture
def stub(tmp_path, monkeypatch):
    script = tmp_path / "stub_runner.py"
    script.write_text(STUB)
    monkeypatch.setenv("SOBOL_JOB_RUNNER", str(script))
    monkeypatch.setenv("STUB_BATCH", str(tmp_path / "batch"))
    return tmp_path


def _job(tmp_path, **spec):
    d = js.new_job_dir(tmp_path / "out", "t")
    js.write_spec(d, js.JobSpec(label="t", sim_argv=spec.pop("sim_argv", ["--jobs", "1"]), **spec))
    return d


def _fake_backend(monkeypatch, fail_ids=()):
    rendered = []
    t = SimpleNamespace(RenderFormValues=lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(job.render_backend, "load", lambda: t)
    monkeypatch.setattr(job.render_backend, "form_from_dict", lambda t, d: d)
    monkeypatch.setattr(job.render_backend, "is_sobol_run", lambda p: True)

    def render_run(t, run_dir, form, keep_dumps=False, **k):
        rendered.append(Path(run_dir).name)
        ok = Path(run_dir).name not in fail_ids
        return SimpleNamespace(ok=ok, stage="done" if ok else "partial", message="m",
                               paths=[SimpleNamespace(name="instance_grains", n_frames=3, elapsed_s=12.4)])

    monkeypatch.setattr(job.render_backend, "render_run", render_run)
    return rendered


def test_sim_only_done(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok")
    d = _job(stub)
    assert job.main([str(d)]) == 0
    st = js.read_status(d)
    assert st.state == "done" and st.batch_dir == str(stub / "batch")
    assert [s.name for s in st.stages] == ["sim"]


def test_sim_with_no_ok_rows_fails(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:failed")
    monkeypatch.setenv("STUB_RC", "1")
    d = _job(stub, render={"paths": ["instance_grains"]})
    assert job.main([str(d)]) == 1
    st = js.read_status(d)
    assert st.state == "failed" and [s.name for s in st.stages] == ["sim"]


def test_render_after_selected_runs(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok 3:ok")
    rendered = _fake_backend(monkeypatch)
    d = _job(stub, render={"paths": ["instance_grains"]}, run_selection={"first": 2})
    assert job.main([str(d)]) == 0
    assert rendered == ["run_0001", "run_0002"]
    st = js.read_status(d)
    assert st.state == "done"
    assert "instance_grains 3 frames 12 s" in st.stages[-1].message  # per-path timing shown in the Jobs tab


def test_selection_skips_failed_and_missing_ids(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok 3:failed")
    rendered = _fake_backend(monkeypatch)
    d = _job(stub, render={"paths": ["instance_grains"]}, run_selection={"ids": [2, 3, 99]})
    job.main([str(d)])
    st = js.read_status(d)
    assert rendered == ["run_0002"]
    assert st.state == "partial"
    select = next(s for s in st.stages if s.name == "select")
    assert "3" in select.message and "99" in select.message


def test_render_failure_is_partial(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok")
    _fake_backend(monkeypatch, fail_ids={"run_0002"})
    d = _job(stub, render={"paths": ["instance_grains"]})
    job.main([str(d)])
    assert js.read_status(d).state == "partial"


def test_render_unavailable_is_partial(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")
    monkeypatch.setattr(job.render_backend, "load", lambda: "render unavailable: no Blender")
    d = _job(stub, render={"paths": ["instance_grains"]})
    job.main([str(d)])
    st = js.read_status(d)
    assert st.state == "partial" and "no Blender" in st.stages[-1].message


def test_dry_run_is_done_without_render(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:prepared_only")
    d = _job(stub, sim_argv=["--dry-run"], render={"paths": ["instance_grains"]})
    assert job.main([str(d)]) == 0
    assert js.read_status(d).state == "done"


def test_render_only_paths(tmp_path, monkeypatch):
    rendered = _fake_backend(monkeypatch)
    d = js.new_job_dir(tmp_path, "r")
    js.write_spec(d, js.JobSpec(label="r", render={"paths": ["instance_grains"]},
                                run_selection={"paths": [str(tmp_path / "b" / "run_0004")]}))
    assert job.main([str(d)]) == 0
    assert rendered == ["run_0004"]


def test_cancel_marks_cancelled(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")
    monkeypatch.setenv("STUB_SLEEP", "60")
    d = _job(stub)
    pid = js.launch(d, js.read_spec(d))
    for _ in range(100):
        st = js.read_status(d)
        if st and st.stages and (d / "batch_dir.txt").is_file():  # runner has reported its batch
            break
        time.sleep(0.1)
    js.cancel(st)
    for _ in range(100):
        if js.read_status(d).state != "running":
            break
        time.sleep(0.1)
    st = js.read_status(d)
    assert st.state == "cancelled"
    assert st.batch_dir == str(stub / "batch")  # Resume needs it even though the runner never returned
    assert [s.state for s in st.stages] == ["failed"] and "cancelled" in st.stages[0].message


def test_cancel_during_render_warns_about_blender(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")
    _fake_backend(monkeypatch)

    def cancelled(*_a, **_k):
        raise job.Cancelled()

    monkeypatch.setattr(job.render_backend, "render_run", cancelled)
    d = _job(stub, render={"paths": ["instance_grains"]})
    job.main([str(d)])
    st = js.read_status(d)
    assert st.state == "cancelled" and "blender.exe" in st.stages[-1].message


def test_select_runs():
    ok = [(1, Path("r1")), (2, Path("r2")), (3, Path("r3"))]
    assert job.select_runs(ok, "all") == ([Path("r1"), Path("r2"), Path("r3")], [])
    assert job.select_runs(ok, {"first": 1}) == ([Path("r1")], [])
    assert job.select_runs(ok, {"ids": [3, 7]}) == ([Path("r3")], ["7"])


def test_render_progress_is_saved_while_running_and_cleared_after(tmp_path, monkeypatch):
    _fake_backend(monkeypatch)
    d = js.new_job_dir(tmp_path, "r")
    js.write_spec(d, js.JobSpec(label="r", render={"paths": ["instance_grains"]},
                                run_selection={"paths": [str(tmp_path / "b" / "run_0004")]}))
    during = []

    def render_run(t, run_dir, form, keep_dumps=False, on_progress=None, **k):
        on_progress({"phase": "render", "dir": "x", "glob": "frame_*.png", "n_expected": 5})
        during.append(js.read_status(d).stages[-1].progress)
        return SimpleNamespace(ok=True, stage="done", message="m", paths=[])

    monkeypatch.setattr(job.render_backend, "render_run", render_run)
    assert job.main([str(d)]) == 0
    assert during == [{"phase": "render", "dir": "x", "glob": "frame_*.png", "n_expected": 5}]
    assert js.read_status(d).stages[-1].progress is None


def test_render_dumps_job_points_its_stage_at_a_progress_file(tmp_path, monkeypatch):
    # the stub copies what job_status.json says while it runs, and the env var it was given
    stub = tmp_path / "stub_render_dumps.py"
    stub.write_text(textwrap.dedent("""
        import json, os, pathlib
        pf = pathlib.Path(os.environ["SOBOL_PROGRESS_FILE"])
        status = json.loads((pf.parent / "job_status.json").read_text())
        (pf.parent / "seen.json").write_text(json.dumps([str(pf), status["stages"][-1]["progress"]]))
    """))
    monkeypatch.setattr(job, "RENDER_DUMPS", stub)
    d = js.new_job_dir(tmp_path, "rd")
    js.write_spec(d, js.JobSpec(label="rd", render_dumps_argv=["/x/run"]))
    assert job.main([str(d)]) == 0
    import json
    pf, progress = json.loads((d / "seen.json").read_text())
    assert pf == str(d / "progress.json") and progress == {"file": pf}
    assert js.read_status(d).stages[-1].progress is None


def test_resume_does_not_render_runs_listed_in_skip_runs(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok 3:ok")
    rendered = _fake_backend(monkeypatch)
    d = _job(stub, render={"paths": ["instance_grains"]}, skip_runs=["run_0001", "run_0002"])
    assert job.main([str(d)]) == 0
    assert rendered == ["run_0003"]


def test_render_after_cleans_dumps_of_runs_it_did_not_render(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok 3:failed")
    _fake_backend(monkeypatch)
    batch = stub / "batch"
    for n in (1, 2, 3):
        (batch / f"run_{n:04d}").mkdir(parents=True)
        (batch / f"run_{n:04d}" / "sobol_00001").write_text("")
    d = _job(stub, render={"paths": ["instance_grains"]}, run_selection={"first": 1})
    job.main([str(d)])  # partial: run 3 failed in the sim
    assert (batch / "run_0001" / "sobol_00001").exists()  # rendered: render_run decides (stubbed here)
    assert not (batch / "run_0002" / "sobol_00001").exists()  # ok, not chosen: cleaned as the runner would
    assert (batch / "run_0003" / "sobol_00001").exists()  # failed: kept, as the runner keeps them
    assert js.read_status(d).stages[-1].name == "cleanup"


def test_render_after_keeps_unrendered_dumps_with_keep_dumps(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok")
    _fake_backend(monkeypatch)
    run2 = stub / "batch" / "run_0002"
    run2.mkdir(parents=True)
    (run2 / "sobol_00001").write_text("")
    d = _job(stub, render={"paths": ["instance_grains"]}, run_selection={"first": 1}, keep_dumps=True)
    job.main([str(d)])
    assert (run2 / "sobol_00001").exists()


def test_error_closes_running_stages(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")

    def boom(batch):
        raise ValueError("bad row")

    monkeypatch.setattr(job, "_ok_runs", boom)
    d = _job(stub)
    assert job.main([str(d)]) == 1
    st = js.read_status(d)
    assert st.state == "failed"
    assert [s.state for s in st.stages] == ["failed", "failed"]  # sim closed, then the error stage
    assert "bad row" in st.stages[-1].message
