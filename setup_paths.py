#!/usr/bin/env python3
"""Write Code/local_paths.json (where data, assets, Blender and the preprocess
python live on this machine) and check the install.

    python3 sobol/setup_paths.py                # prompt for each value
    python3 sobol/setup_paths.py --data-dir ~/viz_data --blender /usr/bin/blender \\
        --python "$(command -v python3)" --device cpu
    python3 sobol/setup_paths.py --check        # verify paths, assets, deps, Blender

HONOURS_PATHS_FILE points at a different config file (e.g. a throwaway test).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from honours_code import CODE_DIR, paths as hp  # noqa: E402



def _assets():
    """(row label, path relative to assets_dir, what it is for). All optional:
    without them the renders fall back to a bare-sphere Earth, a plain grain
    material and a black sky (the shared pack leaves them out)."""
    # imported here: DEMHeadlessRender reads the config at import, which may be
    # the broken file this tool is about to report or rewrite
    sys.path.insert(0, str(CODE_DIR / "BlenderConvert"))
    import DEMHeadlessRender as dhr  # bpy-free outside Blender
    return (
        ("Blenders/11.blend", "Blenders/11.blend", "rock material; plain grains without it"),
        ("TheEarth1.blend", "TheEarth1.blend", "textured Earth; bare sphere without it"),
        ("hdri", str(dhr._HDRI_RELATIVE).replace("\\", "/"), "starfield sky; black without it"),
    )

PREPROCESS_IMPORTS = ("numpy", "pandas", "scipy", "skimage", "sarracen")
_FLAG_KEY = {"data_dir": "data_dir", "assets_dir": "assets_dir", "blender": "blender_exe",
             "python": "preprocess_python", "device": "cycles_device"}
_PATH_KEYS = ("data_dir", "assets_dir")
_EXE_KEYS = ("blender_exe", "preprocess_python")
_MISSING_PROBE = ("import importlib.util,sys;"
                  "print(','.join(n for n in sys.argv[1:] if importlib.util.find_spec(n) is None))")


def _absolute(key: str, value: str) -> str:
    """Paths stored absolute: the tools run with other working directories
    (preprocess cwd = Code/, Blender its own). A bare exe name stays a PATH lookup."""
    if key in _EXE_KEYS and "/" not in value and "\\" not in value and not value.startswith("~"):
        return value
    if key in _PATH_KEYS + _EXE_KEYS and not value[1:3] in (":/", ":\\"):  # leave C:/... alone
        return str(Path(value).expanduser().absolute())
    return value


def write_config(values: dict, path: Path) -> Path:
    """Validate through honours_paths, then write; a bad value leaves the old file alone."""
    clean = {k: _absolute(k, v) if isinstance(v, str) else v
             for k, v in values.items() if k in hp.KEYS and v is not None}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean, indent=1) + "\n", encoding="utf-8")
    old = os.environ.get("HONOURS_PATHS_FILE")
    os.environ["HONOURS_PATHS_FILE"] = str(tmp)
    try:
        hp.load(reload=True)  # raises ConfigError on a bad value
    except hp.ConfigError:
        tmp.unlink()
        raise
    finally:
        if old is None:
            os.environ.pop("HONOURS_PATHS_FILE", None)
        else:
            os.environ["HONOURS_PATHS_FILE"] = old
    tmp.replace(path)
    hp.load(reload=True)
    return path


_BLENDER_PROBE = "import bpy; print('BLENDER_VERSION', bpy.app.version_string, flush=True); import pandas"


def _run(cmd, timeout=300):
    """(CompletedProcess, None), or (None, why) when the program can't start or hangs."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout), None
    except subprocess.TimeoutExpired:
        return None, f"{cmd[0]} timed out after {timeout} s"
    except OSError as exc:
        return None, f"could not run {cmd[0]}: {exc.strerror or exc}"


def check():
    """[(item, ok, required, detail)]; only required rows decide the exit code."""
    cfg = hp.load(reload=True)
    f = hp.config_file()
    rows = [("config", True, True, str(f) if f.is_file() else "none (built-in defaults)")]
    rows.append(("data_dir", Path(cfg["data_dir"]).is_dir(), True, cfg["data_dir"]))
    rows.append(("assets_dir", Path(cfg["assets_dir"]).is_dir(), False, cfg["assets_dir"]))
    for label, rel, why in _assets():
        rows.append((f"asset:{label}", (Path(cfg["assets_dir"]) / rel).is_file(), False, why))
    blender_ok = hp.exe_available(cfg["blender_exe"])
    rows.append(("blender_exe", blender_ok, True, cfg["blender_exe"]))
    if blender_ok:
        # one start: print the version first, so it shows even when pandas is missing
        # (DEMCinematicSetup.py / DEMGrainsBlenderEarthCam.py import pandas inside
        # Blender; stock Blender only bundles numpy)
        r, err = _run([cfg["blender_exe"], "--background", "--factory-startup", "--python-exit-code", "1",
                       "--python-expr", _BLENDER_PROBE])
        version = next((ln.split(None, 1)[1] for ln in (r.stdout if r else "").splitlines()
                        if ln.startswith("BLENDER_VERSION ")), None)
        if version:
            rows.append(("blender_version", version.startswith("5.2"), False, f"Blender {version} (tested: 5.2.x)"))
        ok = r is not None and r.returncode == 0
        rows.append(("blender_deps", ok, True,
                     "pandas ok" if ok else err or
                     "Blender's python lacks pandas: <blender dir>/5.2/python/bin/python3.13 -m pip "
                     "install pandas (Code/README.md)"))
    py = cfg["preprocess_python"]
    py_ok = hp.exe_available(py)
    rows.append(("preprocess_python", py_ok, True, py))
    if py_ok:
        r, err = _run([py, "-c", _MISSING_PROBE, *PREPROCESS_IMPORTS])
        missing = r.stdout.strip() if r else ""
        ok = r is not None and r.returncode == 0 and not missing
        detail = "ok" if ok else err or (f"missing: {missing} (pip install -r Code/requirements-preprocess.txt)"
                                         if missing else (r.stderr.strip().splitlines() or ["failed"])[-1])
        rows.append(("preprocess_deps", ok, True, detail))
    rows.append(("cycles_device", True, True, cfg["cycles_device"]))
    return rows


_DERIVED = {"assets_dir": "<data_dir>/BlenderConvert",
            "preprocess_python": "<data_dir>/.venv/Scripts/python.exe"}


def _prompt(cfg: dict, old: dict) -> dict:
    """Enter keeps what the file set, else leaves the key unset: assets_dir and
    preprocess_python then follow the (new) data_dir instead of being pinned to
    the old one."""
    out = {}
    for key in ("data_dir", "assets_dir", "blender_exe", "preprocess_python", "cycles_device"):
        shown = old.get(key) or (_DERIVED[key] if key in _DERIVED else cfg[key])
        v = input(f"{key} [{shown}]: ").strip()
        out[key] = v or old.get(key)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", help="data folder (DEMCSVs/ outputs go here)")
    p.add_argument("--assets-dir", help="Blender assets folder (default <data-dir>/BlenderConvert)")
    p.add_argument("--blender", help="Blender 5.2 executable (path or name on PATH)")
    p.add_argument("--python", help="python with Code/requirements-preprocess.txt")
    p.add_argument("--device", choices=("gpu", "cpu"), help="Cycles device")
    p.add_argument("--check", action="store_true", help="verify the install")
    a = p.parse_args(argv)
    path = hp.config_file()
    given = {_FLAG_KEY[k]: v for k, v in vars(a).items() if k in _FLAG_KEY and v is not None}
    if given or not a.check:
        try:
            old = hp._read_file(path)
        except hp.ConfigError as exc:
            print(f"{exc}\n-> replacing it", file=sys.stderr)
            old = {}
        if not given:
            try:
                current = hp.load(reload=True)
            except hp.ConfigError:
                current = dict(hp._DEFAULTS)
            given = _prompt(current, old)
        values = {**old, **given}
        if os.environ.get("WSL_DISTRO_NAME"):
            values.setdefault("wsl_distro", os.environ["WSL_DISTRO_NAME"])
        try:
            write_config(values, path)
        except hp.ConfigError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"wrote {path}")
    if a.check:
        try:
            rows = check()
        except hp.ConfigError as exc:
            print(f"{exc}\nRewrite it with: python3 sobol/setup_paths.py", file=sys.stderr)
            return 1
        for item, ok, required, detail in rows:
            mark = "OK  " if ok else ("BAD " if required else "warn")
            print(f"{mark}  {item:<24} {detail}")
        return 0 if all(ok for _, ok, required, _ in rows if required) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
