"""World HDRI used by DEMHeadlessRender.py (TUI headless path)."""
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import CODE, CODE_WIN

BLENDERCONVERT = CODE / "BlenderConvert"
BLENDERCONVERT_WIN = CODE_WIN + r"\BlenderConvert"
BLENDER_EXE = Path("/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe")
WIN_TEMP = Path("/mnt/c/Users/22boy/AppData/Local/Temp")

sys.path.insert(0, str(BLENDERCONVERT))
import DEMHeadlessRender as dhr  # noqa: E402


def test_world_name_matches_apophisonly_blend():
    assert dhr.WORLD_NAME == "Deep Dark Space with Stars.001"


def test_hdri_path_is_repo_deep_dark_space_exr():
    hdri = dhr.deep_dark_space_hdri_path()
    assert hdri.name == (
        "deep-dark-space-with-stars_4K_56e0040f-4645-4b62-b685-637b0eb970ed.exr"
    )
    assert hdri.is_file(), f"missing repo HDRI: {hdri}"


def test_setup_returns_false_when_hdri_missing(tmp_path):
    missing = tmp_path / "missing.exr"
    assert dhr.setup_deep_dark_space_world(missing) is False


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_applies_deep_dark_space_world(tmp_path):
    script = WIN_TEMP / "assert_deep_dark_world.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                    f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
                "import bpy",
                "import DEMHeadlessRender as dhr",
                "ok = dhr.setup_deep_dark_space_world()",
                "assert ok, 'setup_deep_dark_space_world returned False'",
                "world = bpy.context.scene.world",
                "assert world is not None",
                "assert world.name == dhr.WORLD_NAME",
                "nt = world.node_tree",
                "env = next(n for n in nt.nodes if n.type == 'TEX_ENVIRONMENT')",
                "assert env.image is not None",
                "assert 'deep-dark-space-with-stars' in env.image.filepath.replace('\\\\', '/')",
                "bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')",
                "assert abs(bg.inputs['Strength'].default_value - 1.0) < 1e-6",
                "assert bpy.context.scene.view_settings.view_transform == 'AgX'",
                "print('WORLD_OK')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    win_script = "C:/Users/22boy/AppData/Local/Temp/assert_deep_dark_world.py"
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", win_script],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "WORLD_OK" in combined
