"""Tests for render_compare.py (side-by-side comparison of render paths)."""
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import render_compare as rc


def _png(path: Path, size=(64, 36), color=(200, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color + (255,)).save(path)
    return path


def _render_dir(root: Path, name: str, n: int, size=(64, 36), color=(200, 0, 0), start=1) -> Path:
    d = root / f"run_0001_render_{name}"
    for i in range(start, start + n):
        _png(d / f"frame_{i:04d}.png", size, color)
    return d


# --- pairing --------------------------------------------------------------

def test_frame_numbers_parses_only_frame_pngs(tmp_path):
    _png(tmp_path / "frame_0001.png")
    _png(tmp_path / "frame_0010.png")
    (tmp_path / "animation.mp4").write_bytes(b"")
    (tmp_path / "frame_notes.txt").write_text("x")
    assert sorted(rc.frame_numbers(tmp_path)) == [1, 10]


def test_frame_numbers_missing_dir_is_empty(tmp_path):
    assert rc.frame_numbers(tmp_path / "nope") == {}


def test_label_for_strips_run_render_prefix():
    assert rc.label_for(Path("/x/run_0001_render_composite")) == "composite"
    assert rc.label_for(Path("/x/run_0001_render_instance_grains")) == "instance_grains"
    assert rc.label_for(Path("/x/my_renders")) == "my_renders"


def test_default_compare_dir_beside_first_render_dir():
    assert rc.default_compare_dir(Path("/b/run_0001_render_composite")) == Path("/b/run_0001_compare")


def test_plan_compare_needs_two_dirs(tmp_path):
    d = _render_dir(tmp_path, "composite", 2)
    with pytest.raises(rc.CompareError, match="at least 2"):
        rc.plan_compare([("composite", d)])


def test_plan_compare_pairs_by_frame_number_in_given_order(tmp_path):
    a = _render_dir(tmp_path, "composite", 3)
    b = _render_dir(tmp_path, "instance_grains", 3)
    plan = rc.plan_compare([("composite", a), ("instance_grains", b)])
    assert plan.labels == ["composite", "instance_grains"]
    assert [n for n, _ in plan.frames] == [1, 2, 3]
    assert plan.frames[1][1] == [a / "frame_0002.png", b / "frame_0002.png"]
    assert plan.warnings == []


def test_plan_compare_uses_shared_frames_and_warns(tmp_path):
    a = _render_dir(tmp_path, "composite", 5)
    b = _render_dir(tmp_path, "instance_grains", 3)
    plan = rc.plan_compare([("composite", a), ("instance_grains", b)])
    assert [n for n, _ in plan.frames] == [1, 2, 3]
    assert len(plan.warnings) == 1
    assert "composite 5" in plan.warnings[0]
    assert "instance_grains 3" in plan.warnings[0]


def test_plan_compare_empty_dir_is_error(tmp_path):
    a = _render_dir(tmp_path, "composite", 2)
    empty = tmp_path / "run_0001_render_instance_grains"
    empty.mkdir()
    with pytest.raises(rc.CompareError, match="instance_grains"):
        rc.plan_compare([("composite", a), ("instance_grains", empty)])


def test_plan_compare_no_shared_numbers_is_error(tmp_path):
    a = _render_dir(tmp_path, "composite", 2, start=1)
    b = _render_dir(tmp_path, "instance_grains", 2, start=10)
    with pytest.raises(rc.CompareError, match="no frame number shared"):
        rc.plan_compare([("composite", a), ("instance_grains", b)])


# --- stitching ------------------------------------------------------------

def test_stitch_frame_places_tiles_left_to_right(tmp_path):
    red = _png(tmp_path / "a.png", (64, 36), (255, 0, 0))
    blue = _png(tmp_path / "b.png", (64, 36), (0, 0, 255))
    out = tmp_path / "out" / "compare_0001.png"
    assert rc.stitch_frame([red, blue], ["a", "b"], out) == (128, 36)
    im = Image.open(out).convert("RGB")
    assert im.getpixel((32, 30)) == (255, 0, 0)   # below the label box
    assert im.getpixel((96, 30)) == (0, 0, 255)


def test_stitch_frame_mixed_heights_scaled_to_smallest(tmp_path):
    big = _png(tmp_path / "a.png", (128, 72))
    small = _png(tmp_path / "b.png", (64, 36))
    w, h = rc.stitch_frame([big, small], ["a", "b"], tmp_path / "o.png")
    assert (w, h) == (128, 36)   # 128x72 -> 64x36, plus 64x36


def test_stitch_frame_caps_width_and_makes_dims_even(tmp_path):
    tiles = [_png(tmp_path / f"{i}.png", (1921, 1081)) for i in range(4)]
    w, h = rc.stitch_frame(tiles, list("abcd"), tmp_path / "o.png")
    assert w <= rc.MAX_COMPARE_WIDTH
    assert w % 2 == 0 and h % 2 == 0


# --- stage + video ----------------------------------------------------------

def test_run_compare_stage_writes_contiguous_frames(tmp_path):
    a = _render_dir(tmp_path, "composite", 3)
    b = _render_dir(tmp_path, "instance_grains", 3)
    out = tmp_path / "run_0001_compare"
    r = rc.run_compare_stage([("composite", a), ("instance_grains", b)], out)
    assert r.n_frames == 3
    assert r.labels == ["composite", "instance_grains"]
    assert r.video is None
    assert sorted(p.name for p in out.glob("compare_*.png")) == [
        "compare_0001.png", "compare_0002.png", "compare_0003.png",
    ]
    # per-path renders untouched
    assert len(list(a.glob("frame_*.png"))) == 3


def test_run_compare_stage_removes_stale_frames(tmp_path):
    out = tmp_path / "run_0001_compare"
    for i in range(1, 6):
        _png(out / f"compare_{i:04d}.png")
    (out / "animation.mp4").write_bytes(b"old")
    a = _render_dir(tmp_path, "composite", 2)
    b = _render_dir(tmp_path, "instance_grains", 2)
    rc.run_compare_stage([("composite", a), ("instance_grains", b)], out)
    assert sorted(p.name for p in out.glob("compare_*.png")) == ["compare_0001.png", "compare_0002.png"]
    assert not (out / "animation.mp4").exists()


def test_run_compare_stage_corrupt_png_is_compare_error(tmp_path):
    a = _render_dir(tmp_path, "composite", 2)
    b = tmp_path / "run_0001_render_instance_grains"
    b.mkdir()
    (b / "frame_0001.png").write_bytes(b"")
    (b / "frame_0002.png").write_bytes(b"")
    with pytest.raises(rc.CompareError, match="frame 1"):
        rc.run_compare_stage([("composite", a), ("instance_grains", b)], tmp_path / "out")


def test_encode_compare_video_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    with pytest.raises(rc.CompareError, match="ffmpeg not found"):
        rc.encode_compare_video(tmp_path, 24)


def test_encode_compare_video_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        (tmp_path / "animation.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    video = rc.encode_compare_video(tmp_path, 30, ffmpeg="/usr/bin/ffmpeg")
    assert video == tmp_path / "animation.mp4"
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert cmd[cmd.index("-framerate") + 1] == "30"
    assert cmd[cmd.index("-i") + 1] == str(tmp_path / "compare_%04d.png")
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_encode_compare_video_failure_reports_last_stderr_line(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rc.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "noise\nreal problem\n"),
    )
    with pytest.raises(rc.CompareError, match="real problem"):
        rc.encode_compare_video(tmp_path, 24, ffmpeg="/usr/bin/ffmpeg")


@pytest.mark.skipif(rc.shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_run_compare_stage_encodes_real_video(tmp_path):
    a = _render_dir(tmp_path, "composite", 3)
    b = _render_dir(tmp_path, "instance_grains", 3)
    r = rc.run_compare_stage(
        [("composite", a), ("instance_grains", b)], tmp_path / "cmp", fps=24, encode_video=True,
    )
    assert r.video is not None and r.video.stat().st_size > 0


def test_format_compare_result():
    r = rc.CompareResult(Path("/b/run_0001_compare"), 24, ["composite", "instance_grains"],
                         ["frame counts differ (composite 30, instance_grains 24); compared the 24 shared"],
                         Path("/b/run_0001_compare/animation.mp4"))
    text = rc.format_compare_result(r)
    assert text.startswith("compare: ok, 24 frame(s) (composite | instance_grains) to /b/run_0001_compare")
    assert "encoded to /b/run_0001_compare/animation.mp4" in text
    assert "frame counts differ" in text
    assert "\n" not in text


# --- CLI ------------------------------------------------------------------

def test_main_default_output_dir(tmp_path, capsys):
    a = _render_dir(tmp_path, "composite", 2)
    b = _render_dir(tmp_path, "per_sphere", 2)
    assert rc.main([str(a), str(b)]) == 0
    assert len(list((tmp_path / "run_0001_compare").glob("compare_*.png"))) == 2
    assert "compare: ok, 2 frame(s) (composite | per_sphere)" in capsys.readouterr().out


def test_main_error_exit_code(tmp_path, capsys):
    a = _render_dir(tmp_path, "composite", 2)
    assert rc.main([str(a), str(tmp_path / "missing")]) == 1
    assert "compare failed" in capsys.readouterr().err
