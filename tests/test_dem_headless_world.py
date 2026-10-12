"""World HDRI used by DEMHeadlessRender.py (TUI headless path)."""
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import CODE, CODE_WIN, max_laptop

BLENDERCONVERT = CODE / "BlenderConvert"
BLENDERCONVERT_WIN = CODE_WIN + r"\BlenderConvert"
from tests._paths import BLENDER_EXE  # noqa: E402  (configured Windows blender.exe, else skip)
WIN_TEMP = Path("/mnt/c/Users/22boy/AppData/Local/Temp")

sys.path.insert(0, str(BLENDERCONVERT))
import DEMHeadlessRender as dhr  # noqa: E402


def test_world_name_matches_apophisonly_blend():
    assert dhr.WORLD_NAME == "Deep Dark Space with Stars.001"


@max_laptop
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


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_removes_startup_cube_camera_light():
    script = WIN_TEMP / "assert_no_startup_defaults.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
                "import bpy",
                "import DEMHeadlessRender as dhr",
                "assert {'Cube', 'Camera', 'Light'} <= set(bpy.data.objects.keys())",
                "removed = dhr.remove_startup_defaults()",
                "assert sorted(removed) == ['Camera', 'Cube', 'Light'], removed",
                "left = set(bpy.data.objects.keys())",
                "assert not left & {'Cube', 'Camera', 'Light'}, left",
                "assert 'Cube' not in bpy.data.meshes and 'Light' not in bpy.data.lights",
                "assert 'Camera' not in bpy.data.cameras",
                "print('DEFAULTS_GONE')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    win_script = "C:/Users/22boy/AppData/Local/Temp/assert_no_startup_defaults.py"
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", win_script],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "DEFAULTS_GONE" in combined


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_missing_hdri_gives_black_sky():
    # Shared asset pack has no HDRI: space should be black, not Blender's grey default World.
    script = WIN_TEMP / "dhr_black_sky.py"
    script.write_text("\n".join([
        "import sys",
        f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
        "import bpy",
        "import DEMHeadlessRender as dhr",
        "assert dhr.setup_deep_dark_space_world('D:/no/such.exr') is False",
        "w = bpy.context.scene.world",
        "bg = w.node_tree.nodes['Background']",
        "assert tuple(bg.inputs['Color'].default_value)[:3] == (0.0, 0.0, 0.0), tuple(bg.inputs['Color'].default_value)",
        "print('BLACK_OK')",
    ]) + "\n", encoding="utf-8")
    result = subprocess.run([str(BLENDER_EXE), "--background", "--factory-startup", "--python-exit-code", "1",
                             "--python", "C:/Users/22boy/AppData/Local/Temp/dhr_black_sky.py"],
                            capture_output=True, text=True, check=False)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert "BLACK_OK" in combined


def test_require_camera_names_the_cause_when_the_script_set_none():
    # Startup Camera is removed before exec; a viz script that skips its tracking
    # camera must fail with a reason, not Blender's bare "No camera found in scene".
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match="tracking camera"):
        dhr.require_camera(SimpleNamespace(camera=None))
    dhr.require_camera(SimpleNamespace(camera=object()))  # a camera: no error
