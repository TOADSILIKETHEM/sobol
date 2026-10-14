"""Optional video-encode stage in DEMHeadlessRender.py (TUI headless path)."""
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import CODE, CODE_WIN

BLENDERCONVERT = CODE / "BlenderConvert"
BLENDERCONVERT_WIN = CODE_WIN + r"\BlenderConvert"
from tests._paths import BLENDER_EXE  # noqa: E402  (configured Windows blender.exe, else skip)
WIN_TEMP = Path("/mnt/c/Users/22boy/AppData/Local/Temp")

sys.path.insert(0, str(BLENDERCONVERT))
import DEMHeadlessRender as dhr  # noqa: E402


def _parse_with_argv(tmp_path, extra_argv):
    from tests.test_dem_headless_paths import _manifest
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "blender", "--background", "--python", "DEMHeadlessRender.py", "--",
            "--manifest", str(_manifest(tmp_path, "instance")), "--output-dir", "/o",
        ] + extra_argv
        return dhr._parse_args()
    finally:
        sys.argv = argv_backup


def test_parse_args_encode_video_defaults_off(tmp_path):
    args = _parse_with_argv(tmp_path, [])
    assert args.encode_video is False
    assert args.video_fps is None


def test_parse_args_encode_video_with_explicit_fps(tmp_path):
    args = _parse_with_argv(tmp_path, ["--encode-video", "--video-fps", "60"])
    assert args.encode_video is True
    assert args.video_fps == 60


def test_encode_video_requires_bpy():
    assert dhr.bpy is None
    with pytest.raises(RuntimeError, match="bpy"):
        dhr.encode_video(Path("/tmp"), n_frames=1, fps=24)


def test_encode_video_rejects_frame_count_mismatch(tmp_path, monkeypatch):
    # Guards against a silent partial encode if the render stage wrote
    # fewer frames than expected -- this must be checked before touching
    # bpy, so fake bpy in just enough to reach the count check.
    (tmp_path / "frame_0001.png").write_bytes(b"")
    monkeypatch.setattr(dhr, "bpy", object())
    with pytest.raises(RuntimeError, match="expected 2.*found 1"):
        dhr.encode_video(tmp_path, n_frames=2, fps=24)


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_encodes_frame_sequence_to_mp4():
    frames_dir_wsl = WIN_TEMP / "dhr_video_test_frames"
    frames_dir_wsl.mkdir(exist_ok=True)
    for f in frames_dir_wsl.glob("*"):
        f.unlink()
    frames_dir_win = "C:/Users/22boy/AppData/Local/Temp/dhr_video_test_frames"

    script = WIN_TEMP / "assert_encode_video.py"
    script.write_text(
        "\n".join(
            [
                "import sys, pathlib",
                f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
                "import bpy",
                "import DEMHeadlessRender as dhr",
                f"out_dir = pathlib.Path(r'{frames_dir_win}')",
                "for i in range(1, 3):",
                "    img = bpy.data.images.new(f'frame{i}', width=16, height=16)",
                "    img.filepath_raw = str(out_dir / f'frame_{i:04d}.png')",
                "    img.file_format = 'PNG'",
                "    img.save()",
                "video_path = dhr.encode_video(out_dir, n_frames=2, fps=24)",
                "assert video_path.is_file(), video_path",
                "assert video_path.stat().st_size > 0",
                "print('VIDEO_OK', video_path)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    win_script = "C:/Users/22boy/AppData/Local/Temp/assert_encode_video.py"
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", win_script],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "VIDEO_OK" in combined


def test_render_prepared_scene_encodes_when_asked(tmp_path, monkeypatch):
    # the encode flag must not hide the module's encode_video() (it was a bool called as a function)
    from types import SimpleNamespace as NS
    for name in ("require_camera", "setup_cycles_gpu", "apply_gpu_denoise", "apply_render_speedups",
                 "cap_image_textures", "apply_resume"):
        monkeypatch.setattr(dhr, name, lambda *a, **k: None)
    monkeypatch.setattr(dhr, "render_banner", lambda *a, **k: "")
    monkeypatch.setattr(dhr, "bpy", NS(ops=NS(render=NS(render=lambda **k: None)), data=NS(images=[])),
                        raising=False)
    calls = []
    monkeypatch.setattr(dhr, "encode_video", lambda out_dir, n, fps: calls.append((n, fps)))
    scene = NS(frame_start=1, frame_end=3, cycles=NS(), render=NS(image_settings=NS()))
    dhr.render_prepared_scene(scene, viz_path="instance", out_dir=tmp_path / "o", res_x=8, res_y=8,
                              samples=1, fps=24, device="gpu", encode=True)
    assert calls == [(3, 24)]
