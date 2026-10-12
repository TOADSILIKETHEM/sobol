"""sobol/setup_paths.py: write Code/local_paths.json and check the install."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import setup_paths as sp  # noqa: E402
from honours_code import paths as hp  # noqa: E402


@pytest.fixture
def cfgfile(tmp_path, monkeypatch):
    f = tmp_path / "local_paths.json"
    monkeypatch.setenv("HONOURS_PATHS_FILE", str(f))
    for k in ("HONOURS_DATA_DIR", "HONOURS_BLENDER_ASSETS_DIR", "HONOURS_BLENDER_EXE",
              "HONOURS_PREPROCESS_PYTHON", "HONOURS_CYCLES_DEVICE"):
        monkeypatch.delenv(k, raising=False)
    hp.load(reload=True)
    yield f
    monkeypatch.delenv("HONOURS_PATHS_FILE")
    hp.load(reload=True)


def test_flags_write_config(cfgfile, tmp_path):
    rc = sp.main(["--data-dir", str(tmp_path), "--blender", "/usr/bin/blender",
                  "--python", sys.executable, "--device", "cpu"])
    assert rc == 0
    data = json.loads(cfgfile.read_text())
    assert data["data_dir"] == str(tmp_path) and data["cycles_device"] == "cpu"
    assert hp.blender_exe() == "/usr/bin/blender"


def test_flags_merge_into_existing_file(cfgfile, tmp_path):
    sp.main(["--data-dir", str(tmp_path)])
    sp.main(["--device", "cpu"])
    data = json.loads(cfgfile.read_text())
    assert data["data_dir"] == str(tmp_path) and data["cycles_device"] == "cpu"


def test_bad_device_rejected_before_writing(cfgfile):
    with pytest.raises(SystemExit):
        sp.main(["--device", "tpu"])
    assert not cfgfile.exists()


def test_check_reports_missing_things(cfgfile, tmp_path):
    sp.main(["--data-dir", str(tmp_path / "nope"), "--blender", str(tmp_path / "noblender"),
             "--python", sys.executable])
    rows = {item: ok for item, ok, _, _ in sp.check()}
    assert rows["data_dir"] is False
    assert rows["blender_exe"] is False
    assert rows["preprocess_python"] is True


def test_missing_assets_are_optional(cfgfile, tmp_path):
    # Shared pack has no .blend/HDRI (2026-10-10): renders fall back, so --check passes without them.
    sp.main(["--data-dir", str(tmp_path), "--python", sys.executable])
    rows = {item: (ok, required) for item, ok, required, _ in sp.check()}
    assert rows["asset:Blenders/11.blend"] == (False, False)
    assert rows["asset:TheEarth1.blend"] == (False, False)


def test_check_exit_code_ignores_optional_rows(cfgfile, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "check", lambda: [("data_dir", True, True, ""), ("asset:x", False, False, "")])
    assert sp.main(["--check"]) == 0
    monkeypatch.setattr(sp, "check", lambda: [("data_dir", False, True, "")])
    assert sp.main(["--check"]) == 1


def _fake_blender(tmp_path, has_pandas: bool):
    exe = tmp_path / "blender"
    exe.write_text("#!/bin/sh\n"
                   'case "$*" in *--python-expr*) exit ' + ("0" if has_pandas else "1") + ";; esac\n"
                   'echo "Blender 5.2.1 LTS"\n')
    exe.chmod(0o755)
    return exe


def test_check_flags_blender_without_pandas(cfgfile, tmp_path):
    # Stock Blender lacks pandas (DEMCinematicSetup imports it); Max's has it added by hand.
    sp.main(["--data-dir", str(tmp_path), "--blender", str(_fake_blender(tmp_path, False)),
             "--python", sys.executable])
    rows = {item: (ok, required, d) for item, ok, required, d in sp.check()}
    assert rows["blender_deps"][:2] == (False, True)
    assert "pip install" in rows["blender_deps"][2]


def test_check_passes_blender_with_pandas(cfgfile, tmp_path):
    sp.main(["--data-dir", str(tmp_path), "--blender", str(_fake_blender(tmp_path, True)),
             "--python", sys.executable])
    rows = {item: ok for item, ok, _, _ in sp.check()}
    assert rows["blender_deps"] is True


# -- final-review fixes (2026-10-10) --

def test_relative_and_home_paths_are_stored_absolute(cfgfile, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    sp.main(["--data-dir", "pack/data", "--assets-dir", "~/assets", "--python", "venv/bin/python",
             "--blender", "blender"])
    data = json.loads(cfgfile.read_text())
    assert data["data_dir"] == str(tmp_path / "pack" / "data")
    assert data["assets_dir"] == str(tmp_path / "home" / "assets")
    assert data["preprocess_python"] == str(tmp_path / "venv" / "bin" / "python")
    assert data["blender_exe"] == "blender"  # bare name on PATH stays a name


def test_broken_config_check_is_one_clean_line(cfgfile, capsys):
    cfgfile.write_text("{not json", encoding="utf-8")
    assert sp.main(["--check"]) == 1
    err = capsys.readouterr()
    assert "local_paths.json" in err.out + err.err
    assert "Traceback" not in err.out + err.err


def test_broken_config_can_be_rewritten(cfgfile, tmp_path):
    cfgfile.write_text("{not json", encoding="utf-8")
    assert sp.main(["--data-dir", str(tmp_path)]) == 0
    assert json.loads(cfgfile.read_text())["data_dir"] == str(tmp_path)


def _answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_prompt_enter_lets_derived_paths_follow_new_data_dir(cfgfile, tmp_path, monkeypatch):
    # data_dir, assets_dir, blender_exe, preprocess_python, cycles_device
    _answers(monkeypatch, str(tmp_path / "viz"), "", "/usr/bin/blender", "", "")
    assert sp.main([]) == 0
    data = json.loads(cfgfile.read_text())
    assert "assets_dir" not in data and "preprocess_python" not in data
    assert hp.assets_dir() == str(tmp_path / "viz") + "/BlenderConvert"


def test_prompt_enter_keeps_paths_the_file_already_set(cfgfile, tmp_path, monkeypatch):
    sp.main(["--assets-dir", str(tmp_path / "pack")])
    _answers(monkeypatch, str(tmp_path / "viz"), "", "", "", "")
    assert sp.main([]) == 0
    assert json.loads(cfgfile.read_text())["assets_dir"] == str(tmp_path / "pack")


def test_prompt_enter_on_broken_config_writes_no_blank_paths(cfgfile, tmp_path, monkeypatch):
    cfgfile.write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _answers(monkeypatch, str(tmp_path / "viz"), "", "", "", "")
    assert sp.main([]) == 0
    data = json.loads(cfgfile.read_text())
    assert "assets_dir" not in data and "preprocess_python" not in data
    assert hp.preprocess_python() == str(tmp_path / "viz") + "/.venv/Scripts/python.exe"
