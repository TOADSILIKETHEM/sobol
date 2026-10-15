"""Code/render_queue.py: render jobs queued by render_dumps.py --queue on another machine."""
import json
import sys

from honours_code import CODE_DIR

sys.path.insert(0, str(CODE_DIR))
import render_queue as rq  # noqa: E402


def _queue(root, n_frames=3, points=3, csvs=3):
    viz = root / "demo" / "run_0001_viz_instance"
    (viz / "points").mkdir(parents=True)
    bodies = root / "demo" / "run_0001_bodies_output"
    bodies.mkdir(parents=True)
    for i in range(points):
        (viz / "points" / f"{i:05d}.npz").write_bytes(b"x")
    for i in range(csvs):
        (bodies / f"sobol_{i:05d}.csv").write_text("x")
    (viz / "manifest.json").write_text(json.dumps(
        {"n_frames": n_frames, "points_dir": "points", "bodies_csv_dir": "../run_0001_bodies_output"}))
    q = root / rq.QUEUE_DIR_NAME
    q.mkdir()
    job = q / "demo__run_0001_render_instance_grains.json"
    job.write_text(json.dumps({
        "args": ["--viz-path", "instance", "--manifest", "demo/run_0001_viz_instance/manifest.json",
                 "--output-dir", "demo/run_0001_render_instance_grains", "--samples", "8",
                 "--resume", "--encode-video", "--video-fps", "24"],
        "output_dir": "demo/run_0001_render_instance_grains",
        "manifest": "demo/run_0001_viz_instance/manifest.json",
    }))
    return job


def _fake_blender(monkeypatch, frames=2, rc=0):
    seen = []

    class Done:
        returncode = rc

    def run(cmd, **kw):
        seen.append(cmd)
        out = cmd[cmd.index("--output-dir") + 1]
        for i in range(frames):
            (__import__("pathlib").Path(out) / f"frame_{i:04d}.png").write_bytes(b"x")
        return Done

    monkeypatch.setattr(rq.subprocess, "run", run)
    return seen


def test_job_renders_with_absolute_paths_and_moves_to_done(tmp_path, monkeypatch):
    job = _queue(tmp_path)
    seen = _fake_blender(monkeypatch)
    assert rq.one_pass(tmp_path, "blender") == 1
    cmd = seen[0]
    assert cmd[:5] == ["blender", "--background", "--python-exit-code", "1", "--python"]
    assert cmd[cmd.index("--manifest") + 1] == str(tmp_path / "demo/run_0001_viz_instance/manifest.json")
    assert not job.exists()
    assert (job.parent / "done" / job.name).is_file()
    assert "render_queue.py" in (tmp_path / "demo/run_0001_render_instance_grains/render.log").read_text()


def test_failed_render_moves_to_failed(tmp_path, monkeypatch):
    job = _queue(tmp_path)
    _fake_blender(monkeypatch, frames=0, rc=1)
    rq.one_pass(tmp_path, "blender")
    assert (job.parent / "failed" / job.name).is_file()


def test_unsynced_inputs_wait(tmp_path, monkeypatch):
    job = _queue(tmp_path, points=1)
    seen = _fake_blender(monkeypatch)
    assert rq.one_pass(tmp_path, "blender") == 0
    assert job.exists() and not seen


def test_max_frames_test_render_leaves_job_queued_and_skips_video(tmp_path, monkeypatch):
    job = _queue(tmp_path)
    seen = _fake_blender(monkeypatch)
    rq.one_pass(tmp_path, "blender", max_frames=2)
    assert job.exists()
    cmd = seen[0]
    assert cmd[cmd.index("--max-frames") + 1] == "2"
    assert "--encode-video" not in cmd and "--video-fps" not in cmd


def test_own_running_job_is_resumed_first(tmp_path, monkeypatch):
    job = _queue(tmp_path)
    running = job.with_name(f"{job.name}.{rq.HOST}.running")
    job.rename(running)
    _fake_blender(monkeypatch)
    assert rq.one_pass(tmp_path, "blender") == 1
    assert (job.parent / "done" / job.name).is_file() and not running.exists()
