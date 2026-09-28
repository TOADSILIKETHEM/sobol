"""Cycles GPU enable used by DEMHeadlessRender.py (TUI headless path)."""
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


def test_setup_cycles_gpu_requires_bpy():
    assert dhr.bpy is None
    with pytest.raises(RuntimeError, match="bpy"):
        dhr.setup_cycles_gpu()


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_enables_gpu_and_disables_cpu():
    script = WIN_TEMP / "assert_cycles_gpu.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
                "import bpy",
                "import DEMHeadlessRender as dhr",
                "backend = dhr.setup_cycles_gpu()",
                "assert backend in ('OPTIX', 'CUDA'), backend",
                "scene = bpy.context.scene",
                "assert scene.render.engine == 'CYCLES'",
                "assert scene.cycles.device == 'GPU'",
                "prefs = bpy.context.preferences.addons['cycles'].preferences",
                "assert prefs.compute_device_type == backend",
                "devices = list(prefs.devices)",
                "cpu = [d for d in devices if d.type == 'CPU']",
                "gpu = [d for d in devices if d.type == backend]",
                "assert gpu, f'no {backend} devices: {[(d.name, d.type) for d in devices]}'",
                "assert all(d.use for d in gpu)",
                "assert all(not d.use for d in cpu)",
                "print('GPU_OK', backend)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    win_script = "C:/Users/22boy/AppData/Local/Temp/assert_cycles_gpu.py"
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", win_script],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "GPU_OK" in combined
