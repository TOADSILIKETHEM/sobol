"""Cycles GPU enable used by DEMHeadlessRender.py (TUI headless path)."""
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


def test_device_flag_defaults_to_gpu():
    from tests.test_dem_headless_paths import _parse
    assert _parse(["--grains-dir", "/g", "--bodies-dir", "/b"]).device == "gpu"
    assert _parse(["--grains-dir", "/g", "--bodies-dir", "/b", "--device", "cpu"]).device == "cpu"


def test_device_flag_default_follows_configured_cycles_device(monkeypatch):
    # local_paths.json cycles_device / HONOURS_CYCLES_DEVICE, for direct headless runs too
    from tests.test_dem_headless_paths import _parse
    monkeypatch.setattr(dhr.honours_paths, "cycles_device", lambda: "cpu")
    assert _parse(["--grains-dir", "/g", "--bodies-dir", "/b"]).device == "cpu"
    assert _parse(["--grains-dir", "/g", "--bodies-dir", "/b", "--device", "gpu"]).device == "gpu"


def test_gpu_backend_order_keeps_optix_then_cuda_first():
    assert dhr.GPU_BACKENDS[:2] == ("OPTIX", "CUDA")
    assert set(dhr.GPU_BACKENDS) >= {"HIP", "ONEAPI", "METAL"}


def test_setup_cycles_cpu_requires_bpy():
    with pytest.raises(RuntimeError, match="bpy"):
        dhr.setup_cycles_cpu()


def test_tui_passes_device_only_when_cpu(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import tui_sim_render as tsr
    rp = tsr.RenderParams(grains_dir=Path("/g"), bodies_dir=Path("/b"), output_dir=Path("/o"))
    monkeypatch.setattr(tsr, "CYCLES_DEVICE", "gpu")
    assert "--device" not in tsr.build_render_command(rp)
    monkeypatch.setattr(tsr, "CYCLES_DEVICE", "cpu")
    cmd = tsr.build_render_command(rp)
    assert cmd[cmd.index("--device") + 1] == "cpu"


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_cpu_device():
    script = WIN_TEMP / "assert_cycles_cpu.py"
    script.write_text("\n".join([
        "import sys",
        f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
        "import bpy",
        "import DEMHeadlessRender as dhr",
        "assert dhr.setup_cycles_cpu() == 'CPU'",
        "assert bpy.context.scene.render.engine == 'CYCLES'",
        "assert bpy.context.scene.cycles.device == 'CPU'",
        "print('CPU_OK')",
    ]) + "\n", encoding="utf-8")
    result = subprocess.run([str(BLENDER_EXE), "--background", "--python-exit-code", "1", "--python",
                             "C:/Users/22boy/AppData/Local/Temp/assert_cycles_cpu.py"],
                            capture_output=True, text=True, check=False)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "CPU_OK" in combined
