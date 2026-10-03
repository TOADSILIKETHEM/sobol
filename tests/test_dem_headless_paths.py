"""--viz-path contract for DEMHeadlessRender.py (TUI bulk render paths)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._paths import CODE, CODE_WIN, DATA, DATA_WIN

BLENDERCONVERT = CODE / "BlenderConvert"
BLENDERCONVERT_WIN = CODE_WIN + r"\BlenderConvert"
BLENDER_EXE = Path("/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe")
WIN_TEMP = Path("/mnt/c/Users/22boy/AppData/Local/Temp")
WIN_TEMP_WIN = "C:/Users/22boy/AppData/Local/Temp"
FIXTURE_RUN = DATA / "DEMCSVs/_verify_lattice_fix/sobol_20260705_131545_baseline_58deg"
FIXTURE_RUN_WIN = DATA_WIN + "/DEMCSVs/_verify_lattice_fix/sobol_20260705_131545_baseline_58deg"

sys.path.insert(0, str(BLENDERCONVERT))
import DEMHeadlessRender as dhr  # noqa: E402


def _parse(extra_argv):
    backup = sys.argv[:]
    try:
        sys.argv = ["blender", "--background", "--python", "DEMHeadlessRender.py", "--",
                    "--output-dir", "/o"] + extra_argv
        return dhr._parse_args()
    finally:
        sys.argv = backup


def _manifest(tmp_path, viz_mode):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"viz_mode": viz_mode}), encoding="utf-8")
    return p


def test_viz_path_defaults_to_per_sphere():
    args = _parse(["--grains-dir", "/g", "--bodies-dir", "/b"])
    assert args.viz_path == "per_sphere"
    assert args.manifest is None


def test_per_sphere_requires_grains_and_bodies():
    with pytest.raises(ValueError, match="--grains-dir"):
        _parse(["--bodies-dir", "/b"])


def test_composite_requires_manifest():
    with pytest.raises(ValueError, match="--manifest"):
        _parse(["--viz-path", "composite"])


def test_manifest_must_exist(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _parse(["--viz-path", "instance", "--manifest", str(tmp_path / "nope.json")])


def test_viz_mode_mismatch_rejected(tmp_path):
    m = _manifest(tmp_path, "instance")
    with pytest.raises(ValueError, match="viz_mode"):
        _parse(["--viz-path", "composite", "--manifest", str(m)])


def test_composite_with_matching_manifest_parses(tmp_path):
    m = _manifest(tmp_path, "composite")
    args = _parse(["--viz-path", "composite", "--manifest", str(m)])
    assert args.viz_path == "composite"
    assert args.grains_dir is None


def test_parse_resolution():
    assert dhr.parse_resolution("640x360") == (640, 360)
    with pytest.raises(ValueError):
        dhr.parse_resolution("640by360")


def _real_src(viz_path):
    return (BLENDERCONVERT / dhr.SCRIPT_FOR_VIZ_PATH[viz_path]).read_text(encoding="utf-8")


def test_patch_per_sphere_real_script():
    out = dhr.patch_script_source(
        "per_sphere", _real_src("per_sphere"),
        grains_dir="C:/g/", bodies_dir="C:/b/", camera_mode="grain_only",
    )
    assert "GRAINS_DIR = 'C:/g/'" in out
    assert "BODIES_CSV_DIR = 'C:/b/'" in out
    assert "CAMERA_MODE = 'grain_only'" in out


def test_patch_composite_real_script():
    out = dhr.patch_script_source(
        "composite", _real_src("composite"),
        manifest="C:/run_0001_viz/manifest.json", camera_mode="grain_only",
    )
    assert "VIZ_MANIFEST = 'C:/run_0001_viz/manifest.json'" in out
    assert "CAMERA_MODE = 'grain_only'" in out


def test_patch_instance_real_script():
    out = dhr.patch_script_source(
        "instance", _real_src("instance"),
        manifest="C:/v/manifest.json", camera_mode="auto", resolution=(640, 360),
    )
    assert "VIZ_MANIFEST = 'C:/v/manifest.json'" in out
    assert "\nSLOWMO_FACTOR = 1\n" in out
    assert "RENDER_RESOLUTION = (640, 360)" in out
    assert "CAMERA_MODE = None" in out  # auto leaves it untouched


def test_patch_per_sphere_max_frames_limits_scene_build():
    # The frame limit must reach the builder before it keyframes, not only
    # clamp scene.frame_end after every grain file has been keyframed.
    out = dhr.patch_script_source(
        "per_sphere", _real_src("per_sphere"),
        grains_dir="C:/g/", bodies_dir="C:/b/", max_frames=3,
    )
    assert "\nMAX_FRAMES = 3\n" in out


def test_patch_per_sphere_without_max_frames_leaves_constant():
    out = dhr.patch_script_source(
        "per_sphere", _real_src("per_sphere"), grains_dir="C:/g/", bodies_dir="C:/b/",
    )
    assert "\nMAX_FRAMES = None\n" in out


def test_patch_composite_ignores_max_frames():
    # Composite loads geometry per frame in a frame_change_pre handler, so it
    # has no up-front build to limit; its script has no MAX_FRAMES constant.
    out = dhr.patch_script_source(
        "composite", _real_src("composite"), manifest="C:/m.json", max_frames=3,
    )
    assert "MAX_FRAMES" not in out


def test_patch_windows_backslashes_are_literal():
    out = dhr.patch_script_source(
        "composite", "VIZ_MANIFEST = (\n    'x'\n)\nCAMERA_MODE = None\n",
        manifest="C:\\new\\manifest.json",
    )
    assert "VIZ_MANIFEST = 'C:\\\\new\\\\manifest.json'" in out


def test_patch_raises_when_pattern_missing():
    with pytest.raises(RuntimeError, match="VIZ_MANIFEST"):
        dhr.patch_script_source("composite", "CAMERA_MODE = None\n", manifest="C:/m.json")


def test_patch_raises_when_pattern_duplicated():
    src = "VIZ_MANIFEST = ('a')\nVIZ_MANIFEST = ('b')\nCAMERA_MODE = None\n"
    with pytest.raises(RuntimeError, match="2"):
        dhr.patch_script_source("composite", src, manifest="C:/m.json")


def test_ensure_procedural_rock_material_requires_bpy():
    assert dhr.bpy is None
    with pytest.raises(RuntimeError, match="bpy"):
        dhr.ensure_procedural_rock_material()


_RECORDER = '''
import json, sys, traceback
sys.path.insert(0, {bc!r})
import bpy
import numpy as np
import DEMHeadlessRender as dhr
records = []
def _record(scene, *_a):
    obj = bpy.data.objects.get({obj!r})
    if obj is None:
        return
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    rec = {{"frame": scene.frame_current, "n": n, "mean": co.reshape(-1, 3).mean(axis=0).tolist()}}
    if "radius" in mesh.attributes:
        r = np.empty(n, dtype=np.float32)
        mesh.attributes["radius"].data.foreach_get("value", r)
        rec["radius_sum"] = float(r.sum())
    records.append(rec)
bpy.app.handlers.render_post.append(_record)
try:
    dhr.main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
finally:
    open({out!r}, "w").write(json.dumps(records))
'''


def run_recorded_render(name, obj_name, dhr_args):
    """Run DEMHeadlessRender.main() under blender with a render_post recorder.
    Returns (CompletedProcess, records list, output dir WSL Path). Shared with Task 4."""
    out_dir = WIN_TEMP / f"dhr_{name}_out"
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()
    rec_path_win = f"{WIN_TEMP_WIN}/dhr_{name}_records.json"
    script = WIN_TEMP / f"dhr_{name}_recorder.py"
    script.write_text(
        _RECORDER.format(bc=BLENDERCONVERT_WIN, obj=obj_name, out=rec_path_win),
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", f"{WIN_TEMP_WIN}/dhr_{name}_recorder.py", "--",
         "--output-dir", f"{WIN_TEMP_WIN}/dhr_{name}_out",
         "--resolution", "160x90", "--samples", "1"] + dhr_args,
        capture_output=True, text=True, check=False,
    )
    rec_file = WIN_TEMP / f"dhr_{name}_records.json"
    records = json.loads(rec_file.read_text()) if rec_file.is_file() else []
    return result, records, out_dir


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_composite_renders_each_envelope_frame():
    viz = FIXTURE_RUN / "run_0001_viz"
    fids = json.loads((viz / "manifest.json").read_text())["frame_ids"][:4]
    expected = [np.load(viz / "envelopes" / f"{fid}_f0.npz")["verts"].astype(np.float32).mean(axis=0)
                for fid in fids]
    assert not np.allclose(expected[0], expected[3], atol=1e-4), "fixture frames identical"

    result, records, out_dir = run_recorded_render(
        "composite", "VizFragment_00",
        ["--viz-path", "composite", "--manifest", FIXTURE_RUN_WIN + "/run_0001_viz/manifest.json",
         "--max-frames", "4"],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert len(list(out_dir.glob("frame_*.png"))) == 4
    assert len(records) == 4
    for i, rec in enumerate(records):
        assert np.allclose(rec["mean"], expected[i], atol=1e-3), (i, rec, expected[i])


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_instance_grains_uses_per_grain_radius_and_moves():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "viz_preprocess_grains_instance", CODE / "viz" / "viz_preprocess_grains_instance.py"
    )
    vgi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgi)

    viz_dir = WIN_TEMP / "dhr_instance_grains_viz"
    if viz_dir.exists():
        import shutil
        shutil.rmtree(viz_dir)
    vgi.build_grains_instance(
        FIXTURE_RUN / "run_0001_grains_output",
        FIXTURE_RUN / "run_0001_bodies_output",
        viz_dir,
        max_frames=2,
    )
    p0 = np.load(viz_dir / "points" / "00000.npz")
    p1 = np.load(viz_dir / "points" / "00001.npz")

    result, records, out_dir = run_recorded_render(
        "instance_grains", "DEM_InstancePoints",
        ["--viz-path", "instance", "--manifest", f"{WIN_TEMP_WIN}/dhr_instance_grains_viz/manifest.json",
         "--max-frames", "2"],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert len(list(out_dir.glob("frame_*.png"))) == 2
    assert len(records) == 2
    assert records[0]["n"] == len(p0["pos"])
    assert records[0]["radius_sum"] == pytest.approx(float(p0["radius"].sum()), rel=1e-4)
    assert np.allclose(records[0]["mean"], p0["pos"].mean(axis=0), atol=1e-3)
    assert np.allclose(records[1]["mean"], p1["pos"].mean(axis=0), atol=1e-3)


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_build_point_cloud_writes_given_radii():
    script = WIN_TEMP / "dhr_radii_check.py"
    script.write_text(
        "\n".join([
            "import sys, traceback",
            f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
            "import bpy, numpy as np",
            "import DEMHeadlessRender as dhr",
            "try:",
            "    dhr.ensure_procedural_rock_material()",
            "    import DEMBulkInstanceBlender as dbi",
            "    pos = np.zeros((3, 3), dtype=np.float32)",
            "    radii = np.array([0.1, 0.2, 0.3], dtype=np.float32)",
            "    obj, _g = dbi._build_point_cloud(pos, 9.9, radii=radii)",
            "    r = np.empty(3, dtype=np.float32)",
            "    obj.data.attributes['radius'].data.foreach_get('value', r)",
            "    assert np.allclose(r, radii), r",
            "    obj2, _g2 = dbi._build_point_cloud(pos, 9.9)",
            "    r2 = np.empty(3, dtype=np.float32)",
            "    obj2.data.attributes['radius'].data.foreach_get('value', r2)",
            "    assert np.allclose(r2, 9.9), r2",
            "    print('RADII_OK')",
            "except Exception:",
            "    traceback.print_exc()",
            "    sys.exit(1)",
        ]) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", f"{WIN_TEMP_WIN}/dhr_radii_check.py"],
        capture_output=True, text=True, check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert "RADII_OK" in combined


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_per_sphere_max_frames_builds_only_those_frames():
    n_fixture = len(list((FIXTURE_RUN / "run_0001_grains_output").glob("*.npz")))
    assert n_fixture > 2, "fixture needs more than 2 frames to show the limit"

    result, _records, out_dir = run_recorded_render(
        "per_sphere_max_frames", "DEM_Grain_0000",
        ["--viz-path", "per_sphere",
         "--grains-dir", FIXTURE_RUN_WIN + "/run_0001_grains_output",
         "--bodies-dir", FIXTURE_RUN_WIN + "/run_0001_bodies_output",
         "--max-frames", "2"],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert len(list(out_dir.glob("frame_*.png"))) == 2
    assert "Keyframing 2 frames" in combined, combined[-3000:]
    assert f"Keyframing {n_fixture} frames" not in combined
