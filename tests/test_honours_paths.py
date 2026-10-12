"""Code/honours_paths.py: one place for machine paths (2026-10-10)."""
import json
import sys

import pytest

from honours_code import CODE_DIR, paths as hp

MAX_DATA = "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Point honours_paths at a temp config; clear env overrides; reset the cache."""
    f = tmp_path / "local_paths.json"
    monkeypatch.setenv("HONOURS_PATHS_FILE", str(f))
    for k in ("HONOURS_DATA_DIR", "HONOURS_BLENDER_ASSETS_DIR", "HONOURS_BLENDER_EXE",
              "HONOURS_PREPROCESS_PYTHON", "HONOURS_CYCLES_DEVICE"):
        monkeypatch.delenv(k, raising=False)
    hp.load(reload=True)
    yield f
    monkeypatch.delenv("HONOURS_PATHS_FILE")
    hp.load(reload=True)


def _write(f, **kv):
    f.write_text(json.dumps(kv), encoding="utf-8")
    hp.load(reload=True)


def test_windows_form_paths_open_from_wsl(cfg, monkeypatch):
    # setup_paths stores "C:/..." as typed; WSL-side python must get /mnt/c/...
    monkeypatch.setattr(hp.os, "name", "posix")
    _write(cfg, data_dir="C:/viz/data", blender_exe="D:\\Blender\\blender.exe")
    assert hp.data_dir() == "/mnt/c/viz/data"
    assert hp.assets_dir() == "/mnt/c/viz/data/BlenderConvert"
    assert hp.blender_exe() == "/mnt/d/Blender/blender.exe"
    assert hp.to_windows_path(hp.data_dir()) == "C:/viz/data"


def test_code_dir_is_sibling_of_sobol():
    assert (CODE_DIR / "honours_paths.py").is_file()


def test_no_config_gives_todays_values(cfg):
    assert hp.data_dir() == MAX_DATA
    assert hp.assets_dir() == MAX_DATA + "/BlenderConvert"
    assert hp.blender_exe() == "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
    assert hp.preprocess_python() == MAX_DATA + "/.venv/Scripts/python.exe"
    assert hp.cycles_device() == "gpu"


def test_file_overrides_default_and_derives_children(cfg):
    _write(cfg, data_dir="/home/jane/viz_data")
    assert hp.data_dir() == "/home/jane/viz_data"
    assert hp.assets_dir() == "/home/jane/viz_data/BlenderConvert"
    assert hp.preprocess_python() == "/home/jane/viz_data/.venv/Scripts/python.exe"


def test_env_beats_file(cfg, monkeypatch):
    _write(cfg, blender_exe="/usr/bin/blender")
    monkeypatch.setenv("HONOURS_BLENDER_EXE", "/opt/blender/blender")
    hp.load(reload=True)
    assert hp.blender_exe() == "/opt/blender/blender"


def test_existing_assets_env_name_still_wins(cfg, monkeypatch):
    monkeypatch.setenv("HONOURS_BLENDER_ASSETS_DIR", "/elsewhere/assets")
    hp.load(reload=True)
    assert hp.assets_dir() == "/elsewhere/assets"


def test_bad_json_names_the_file(cfg):
    cfg.write_text("{data_dir: nope", encoding="utf-8")
    with pytest.raises(hp.ConfigError, match="local_paths.json"):
        hp.load(reload=True)


def test_unknown_key_lists_allowed_keys(cfg):
    cfg.write_text(json.dumps({"blender_path": "/x"}), encoding="utf-8")
    with pytest.raises(hp.ConfigError, match="blender_path.*allowed.*blender_exe"):
        hp.load(reload=True)


def test_bad_cycles_device_rejected(cfg):
    cfg.write_text(json.dumps({"cycles_device": "tpu"}), encoding="utf-8")
    with pytest.raises(hp.ConfigError, match="cycles_device"):
        hp.load(reload=True)


def test_to_windows_path_matches_tui_contract():
    assert hp.to_windows_path("/mnt/c/Users/x/a b") == "C:/Users/x/a b"
    assert hp.to_windows_path("C:/already") == "C:/already"
    assert hp.to_windows_path("/home/u/Code", distro="Debian") == r"\\wsl.localhost\Debian\home\u\Code"


def test_tool_path_only_translates_for_exe_on_posix(monkeypatch):
    monkeypatch.setattr(hp.os, "name", "posix")
    assert hp.tool_path("/mnt/c/d/x.py", "/mnt/c/B/blender.exe") == "C:/d/x.py"
    assert hp.tool_path("/mnt/c/d/x.py", "/usr/bin/blender") == "/mnt/c/d/x.py"


def test_tool_path_keeps_spaces(monkeypatch):
    monkeypatch.setattr(hp.os, "name", "posix")
    out = hp.tool_path("/mnt/c/Users/Jane Doe/Data analysis/r", "/mnt/c/Program Files/B/blender.exe")
    assert out == "C:/Users/Jane Doe/Data analysis/r"


def test_host_path_on_windows_maps_mnt_and_home(cfg, monkeypatch):
    _write(cfg, wsl_distro="Ubuntu-24.04")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    hp.load(reload=True)
    with monkeypatch.context() as m:  # undo before cfg teardown builds a Path
        m.setattr(hp.os, "name", "nt")
        assert hp.host_path("/mnt/d/assets") == "D:/assets"
        assert hp.host_path("/home/jane/a") == r"\\wsl.localhost\Ubuntu-24.04\home\jane\a"
        assert hp.host_path("C:/x") == "C:/x"


def test_host_path_identity_on_posix(monkeypatch):
    monkeypatch.setattr(hp.os, "name", "posix")
    assert hp.host_path("/home/jane/a") == "/home/jane/a"


# -- Blender side (Windows blender.exe reads the same resolver over \\wsl.localhost) --
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

from tests._paths import CODE_WIN  # noqa: E402

from tests._paths import BLENDER_EXE as _BLENDER  # noqa: E402
_WIN_TEMP = Path("/mnt/c/Users/22boy/AppData/Local/Temp")


def _run_probe(name, lines):
    probe = _WIN_TEMP / name
    probe.write_text("\n".join(["import sys, os", f"sys.path.insert(0, {CODE_WIN + chr(92) + 'BlenderConvert'!r})"]
                               + lines) + "\n", encoding="utf-8")
    r = subprocess.run([str(_BLENDER), "--background", "--factory-startup", "--python-exit-code", "1",
                        "--python", f"C:/Users/22boy/AppData/Local/Temp/{name}"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(chr(92), "/")
    assert r.returncode == 0, out
    return out


_needs_win_blender = pytest.mark.skipif(not _BLENDER.is_file() or not _WIN_TEMP.is_dir(),
                                        reason="Windows blender.exe not available")


@_needs_win_blender
def test_blender_default_assets_and_earth():
    out = _run_probe("hp_probe_default.py", [
        "import DEMHeadlessRender as d",
        "import DEMCinematicSetup as c",
        "print('ASSETS', d.ASSETS_DIR)",
        "print('EARTH', os.path.basename(c.EARTH_MODEL_PATH), os.path.isfile(c.EARTH_MODEL_PATH))",
    ])
    assert "ASSETS C:/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/BlenderConvert" in out
    assert "EARTH TheEarth1.blend True" in out


@_needs_win_blender
def test_blender_assets_override_reaches_earth_path():
    # Another machine's assets dir must move the Earth .blend too, not just the HDRI/rock.
    out = _run_probe("hp_probe_override.py", [
        "os.environ['HONOURS_BLENDER_ASSETS_DIR'] = 'D:/pack/BlenderConvert'",
        "import DEMHeadlessRender as d",
        "import DEMCinematicSetup as c",
        "print('ASSETS', d.ASSETS_DIR)",
        "print('EARTH', c.EARTH_MODEL_PATH)",
    ])
    assert "ASSETS D:/pack/BlenderConvert" in out
    assert "EARTH D:/pack/BlenderConvert/TheEarth1.blend" in out


def test_converter_defaults_follow_data_dir(tmp_path):
    # run_demtocsv_batch default output + DEMDumpConvert standalone default both track data_dir.
    import os as _os
    env = dict(_os.environ, HONOURS_DATA_DIR="/x/data")
    code = ("import sys; sys.path.insert(0, 'Analysis'); import run_demtocsv_batch as r; "
            "print(r.default_base_out())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=str(CODE_DIR.parent / "sobol"))
    assert out.stdout.strip() == "/x/data/DEMCSVs", out.stderr
    src = (CODE_DIR / "CSVconvert" / "DEMDumpConvert.py").read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if ln.startswith("BASE_OUTPUT_DIR"))
    assert "honours_paths.data_dir()" in line


def test_child_env_forwards_overrides_to_windows_exe(monkeypatch):
    # Windows blender.exe / python.exe only see WSL env vars listed in WSLENV.
    monkeypatch.setattr(hp.os, "name", "posix")
    env = {"HONOURS_PATHS_FILE": "/home/j/lp.json", "HONOURS_CYCLES_DEVICE": "cpu", "WSLENV": "FOO/u"}
    out = hp.child_env("/mnt/c/B/blender.exe", env)
    parts = out["WSLENV"].split(":")
    assert "FOO/u" in parts and "HONOURS_PATHS_FILE/p" in parts and "HONOURS_CYCLES_DEVICE" in parts
    assert out["HONOURS_PATHS_FILE"] == "/home/j/lp.json"


def test_child_env_is_none_when_nothing_to_forward(monkeypatch):
    monkeypatch.setattr(hp.os, "name", "posix")
    assert hp.child_env("/mnt/c/B/blender.exe", {"PATH": "/usr/bin"}) is None
    assert hp.child_env("/usr/bin/blender", {"HONOURS_CYCLES_DEVICE": "cpu"}) is None


def test_run_logged_passes_child_env(tmp_path, monkeypatch):
    sys.path.insert(0, str(CODE_DIR.parent / "sobol"))
    import tui_sim_render as tsr
    seen = {}

    class FakeProc:
        stdout = iter(["hi\n"])
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def wait(self): return 0

    def fake_popen(cmd, **kw):
        seen.update(kw)
        return FakeProc()

    monkeypatch.setattr(tsr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tsr._paths, "child_env", lambda exe, environ=None: {"MARK": exe})
    tsr._run_logged(["/mnt/c/x/blender.exe", "-b"], tmp_path / "log.txt")
    assert seen["env"] == {"MARK": "/mnt/c/x/blender.exe"}


def test_non_string_value_is_a_config_error(cfg):
    cfg.write_text(json.dumps({"data_dir": 5}), encoding="utf-8")
    with pytest.raises(hp.ConfigError, match="data_dir"):
        hp.load(reload=True)


def test_child_env_skips_p_flag_for_windows_form_values(monkeypatch):
    # WSLENV /p mangles an already-Windows path (D:/pack -> D;\\wsl.localhost\...).
    monkeypatch.setattr(hp.os, "name", "posix")
    out = hp.child_env("/mnt/c/B/blender.exe", {"HONOURS_BLENDER_ASSETS_DIR": "D:/pack",
                                                 "HONOURS_DATA_DIR": "/home/j/data"})
    parts = out["WSLENV"].split(":")
    assert "HONOURS_BLENDER_ASSETS_DIR" in parts
    assert "HONOURS_DATA_DIR/p" in parts


@pytest.mark.parametrize("script", ["tui_sim_render.py", "render_dumps.py"])
def test_broken_config_gives_one_clean_line_not_a_traceback(tmp_path, script):
    bad = tmp_path / "local_paths.json"
    bad.write_text('{"cycles_device": "GPU0"}', encoding="utf-8")
    import os as _os
    env = dict(_os.environ, HONOURS_PATHS_FILE=str(bad))
    out = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, env=env,
                         cwd=str(CODE_DIR.parent / "sobol"), timeout=60)
    assert out.returncode != 0
    assert "Traceback" not in out.stderr, out.stderr
    assert "setup_paths.py" in out.stderr


def test_tui_placeholder_shape_is_under_assets_dir():
    # make_asset_pack ships the shape under BlenderConvert/ = assets_dir, which may sit outside data_dir.
    import os as _os
    env = dict(_os.environ, HONOURS_DATA_DIR="/x/data", HONOURS_BLENDER_ASSETS_DIR="/y/pack",
               HONOURS_BLENDER_EXE="/usr/bin/blender")
    out = subprocess.run([sys.executable, "-c", "import tui_sim_render as t; print(t.DEFAULT_PLACEHOLDER_OBJ)"],
                         capture_output=True, text=True, env=env, cwd=str(CODE_DIR.parent / "sobol"), timeout=60)
    assert out.stdout.strip().startswith("/y/pack/Shapes/"), out.stdout + out.stderr
    assert out.stdout.strip().endswith("apophis_v233s7.obj")


@pytest.mark.parametrize("value", ["~/codecopy", "rel/codecopy"])
def test_code_dir_env_is_made_absolute_for_children(tmp_path, value):
    # Children run with cwd=CODE_DIR and get HONOURS_CODE_DIR via env: both need it absolute.
    import os as _os
    (tmp_path / "rel").mkdir()
    for d in (tmp_path / "codecopy", tmp_path / "rel" / "codecopy"):
        d.symlink_to(CODE_DIR)
    env = dict(_os.environ, HOME=str(tmp_path), HONOURS_CODE_DIR=value)
    code = (f"import sys; sys.path.insert(0, {str(CODE_DIR.parent / 'sobol')!r}); import os, honours_code as h; "
            "print(h.CODE_DIR); print(os.environ['HONOURS_CODE_DIR'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    lines = out.stdout.split()
    assert len(lines) == 2 and lines[0] == lines[1], out.stdout + out.stderr
    assert lines[0].startswith("/") and "~" not in lines[0]


def test_is_windows_tool(monkeypatch):
    monkeypatch.setattr(hp.os, "name", "posix")
    assert hp.is_windows_tool("/mnt/c/B/blender.EXE")
    assert not hp.is_windows_tool("/usr/bin/blender")
    monkeypatch.setattr(hp.os, "name", "nt")
    assert not hp.is_windows_tool("C:/B/blender.exe")  # Windows host: no path translation


def test_exe_available(tmp_path):
    f = tmp_path / "tool"
    f.write_text("")
    assert hp.exe_available(f) and hp.exe_available("sh")
    assert not hp.exe_available(tmp_path / "missing") and not hp.exe_available("no-such-tool-xyz")
