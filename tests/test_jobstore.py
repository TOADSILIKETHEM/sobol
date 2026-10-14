"""jobstore: job files, detached launch, liveness, cancel."""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import jobstore as js


def test_new_job_dir_is_unique(tmp_path):
    now = datetime(2026, 10, 13, 12, 0, 0)
    a = js.new_job_dir(tmp_path, "np300 single", now=now)
    b = js.new_job_dir(tmp_path, "np300 single", now=now)
    assert a.name == "20261013_120000_np300_single"
    assert b.name == "20261013_120000_np300_single_2"
    assert a.parent == tmp_path / "_jobs"


def test_spec_and_status_round_trip(tmp_path):
    spec = js.JobSpec(label="x", sim_argv=["--jobs", "2"], run_selection={"ids": [1, 3]}, omp_num_threads=4)
    js.write_spec(tmp_path, spec)
    assert js.read_spec(tmp_path) == spec
    st = js.JobStatus(pid=5, pgid=5, host="h", stages=[js.Stage("sim", "done", "a", "b", "ok")])
    js.write_status(tmp_path, st)
    assert js.read_status(tmp_path) == st
    assert not list(tmp_path.glob("*.tmp"))


def test_read_status_tolerates_corrupt_file(tmp_path):
    assert js.read_status(tmp_path) is None
    (tmp_path / "job_status.json").write_text("{\"state\": \"runn")
    assert js.read_status(tmp_path) is None


def test_list_jobs_newest_first(tmp_path):
    a = js.new_job_dir(tmp_path, "a", now=datetime(2026, 1, 1))
    b = js.new_job_dir(tmp_path, "b", now=datetime(2026, 1, 2))
    js.new_job_dir(tmp_path, "half_made", now=datetime(2026, 1, 3))  # no job.json yet: not listed
    for d in (a, b):
        js.write_spec(d, js.JobSpec(label=d.name))
    assert js.list_jobs(tmp_path) == [b, a]


def test_liveness():
    me = js.JobStatus(pid=os.getpid(), host=js.socket.gethostname())
    assert js.liveness(me) == "running"
    p = subprocess.Popen(["true"])
    p.wait()
    assert js.liveness(js.JobStatus(pid=p.pid, host=me.host)) == "died"
    assert js.liveness(js.JobStatus(pid=1, host="elsewhere")) == "running on elsewhere"
    assert js.liveness(js.JobStatus(state="done", pid=p.pid, host=me.host)) == "done"


def test_launch_detaches_and_cancel_kills_group(tmp_path):
    script = tmp_path / "fake_job.py"
    script.write_text(
        "import os, sys, time, pathlib\n"
        "pathlib.Path(sys.argv[1], 'sid').write_text(str(os.getsid(0)))\n"
        "pathlib.Path(sys.argv[1], 'omp').write_text(os.environ['OMP_NUM_THREADS'])\n"
        "time.sleep(60)\n")
    pid = js.launch(tmp_path, js.JobSpec(label="t", omp_num_threads=3), script=script)
    for _ in range(50):
        if (tmp_path / "omp").exists():
            break
        time.sleep(0.1)
    assert int((tmp_path / "sid").read_text()) == pid != os.getsid(0)  # own session: survives SSH drop
    assert (tmp_path / "omp").read_text() == "3"
    js.cancel(js.JobStatus(pid=pid, pgid=pid))
    for _ in range(50):
        if not js.pid_alive(pid):
            break
        time.sleep(0.1)
    assert not js.pid_alive(pid)


def test_batch_progress(tmp_path):
    (tmp_path / "sobol_mass_samples.csv").write_text("run_id,x\n1,a\n2,b\n3,c\n")
    (tmp_path / "sobol_mass_outputs.csv").write_text("run_id,run_dir,status\n1,r1,ok\n2,r2,failed\n")
    assert js.batch_progress(tmp_path) == "2/3 runs finished: 1 ok, 1 failed"


def test_batch_progress_dry_run_is_not_failed(tmp_path):
    (tmp_path / "sobol_mass_samples.csv").write_text("run_id\n1\n2\n")
    (tmp_path / "sobol_mass_outputs.csv").write_text("run_id,run_dir,status\n1,r1,prepared_only\n2,r2,prepared_only\n")
    assert js.batch_progress(tmp_path) == "2/2 runs prepared (dry run)"


def test_stage_progress_counts_frames_and_shows_last_log_line(tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    for i in range(3):
        (out / f"frame_{i:04d}.png").write_text("")
    (out / "render.log").write_text("start\nFra:4 Mem:12M | Sample 64/128\n")
    st = js.Stage("render run_0001", progress={
        "phase": "render", "path": "instance_grains", "index": 1, "total": 2, "dir": str(out),
        "glob": "frame_*.png", "n_expected": 10, "log": str(out / "render.log")})
    assert js.stage_progress(st) == "rendering instance_grains (path 1/2): 3/10 frames | Fra:4 Mem:12M | Sample 64/128"


def test_stage_progress_other_phases(tmp_path):
    convert = js.Stage("render run_0001", progress={"phase": "convert", "dir": str(tmp_path / "missing"),
                                                    "glob": "*.npz", "n_expected": 5})
    assert js.stage_progress(convert) == "converting: 0/5 npz"
    assert js.stage_progress(js.Stage("x", progress={"phase": "preprocess"})) == "preprocessing"
    assert js.stage_progress(js.Stage("sim")) == ""


def test_status_round_trips_progress_and_reads_old_files(tmp_path):
    st = js.JobStatus(stages=[js.Stage("render run_0001", progress={"phase": "preprocess"})])
    js.write_status(tmp_path, st)
    assert js.read_status(tmp_path) == st
    (tmp_path / "job_status.json").write_text('{"state": "done", "stages": [{"name": "sim", "state": "done"}]}')
    assert js.read_status(tmp_path).stages[0].progress is None


def test_stage_progress_ignores_frames_older_than_the_stage(tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    old = out / "frame_0001.png"
    old.write_text("")
    os.utime(old, (1000, 1000))  # left over from an earlier render of the same run
    (out / "frame_0002.png").write_text("")
    st = js.Stage("render run_0001", progress={"phase": "render", "dir": str(out), "glob": "frame_*.png",
                                               "n_expected": 4, "since": time.time() - 60})
    assert js.stage_progress(st) == "rendering: 1/4 frames"


def test_last_line_shortens_blender_saved_line(tmp_path):
    log = tmp_path / "render.log"
    log.write_text("00:16.516  render           | Saved: 'C:\\Users\\x\\DEMCSVs\\run_0002_render\\frame_0009.png'\n")
    assert js.last_line(log) == "Saved: 'frame_0009.png'"


def test_stage_progress_reads_a_progress_file(tmp_path):
    # render_dumps.py runs in a child process: it writes its phase to a file the stage points at
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_0001.png").write_text("")
    pf = tmp_path / "progress.json"
    st = js.Stage("render_dumps", progress={"file": str(pf)})
    assert js.stage_progress(st) == ""  # nothing written yet
    js.write_progress_file(pf, js.progress_info("render", resume=True, path="composite", index=2, total=2,
                                                n_expected=3, output_dir=frames))
    assert js.stage_progress(st) == "rendering composite (path 2/2): 1/3 frames"


def test_progress_info_marks_start_time_unless_resuming(tmp_path):
    fresh = js.progress_info("render", resume=False, output_dir=tmp_path, n_expected=2)
    assert abs(fresh.pop("since") - time.time()) < 60
    assert fresh == {"phase": "render", "n_expected": 2, "dir": str(tmp_path), "glob": "frame_*.png",
                     "log": str(tmp_path / "render.log")}
    assert js.progress_info("compare", resume=False, output_dir=tmp_path, n_expected=2) == {
        "phase": "compare", "n_expected": 2, "dir": str(tmp_path), "glob": "compare_*.png"}
    assert js.progress_info("preprocess", resume=False) == {"phase": "preprocess"}


def test_reused_pid_is_not_the_job():
    me = js.proc_start_time(os.getpid())
    assert me is not None
    assert js.pid_alive(os.getpid(), me)
    assert not js.pid_alive(os.getpid(), str(int(me) + 1))  # same pid, other process start: reused
    other = js.JobStatus(pid=os.getpid(), pgid=os.getpgid(0), start_time=str(int(me) + 1),
                         host=js.socket.gethostname())
    assert js.liveness(other) == "died"
    with pytest.raises(ProcessLookupError):
        js.cancel(other)  # never signals the process group now holding that id


def test_status_without_start_time_reads_and_checks_pid_only(tmp_path):
    js.write_status(tmp_path, js.JobStatus(pid=os.getpid(), host=js.socket.gethostname()))
    st = js.read_status(tmp_path)
    assert st.start_time is None and js.liveness(st) == "running"
