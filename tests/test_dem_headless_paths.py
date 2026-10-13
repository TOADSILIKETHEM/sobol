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
from tests._paths import BLENDER_EXE  # noqa: E402  (configured Windows blender.exe, else skip)
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


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_blender_rock_material_falls_back_without_11_blend():
    # Shared asset pack has no 11.blend (2026-10-10): the instance path must still
    # render, with a plain stand-in under the same name, and say so.
    script = WIN_TEMP / "dhr_rock_fallback.py"
    script.write_text("\n".join([
        "import sys",
        f"sys.path.insert(0, {BLENDERCONVERT_WIN!r})",
        "import bpy",
        "import DEMHeadlessRender as dhr",
        "dhr.ensure_procedural_rock_material('D:/no/such/11.blend')",
        "mat = bpy.data.materials.get(dhr.ROCK_MATERIAL_NAME)",
        "assert mat is not None and mat.use_nodes",
        "bsdf = mat.node_tree.nodes.get('Principled BSDF')",
        "assert bsdf is not None",
        "print('FALLBACK_OK')",
    ]) + "\n", encoding="utf-8")
    result = subprocess.run([str(BLENDER_EXE), "--background", "--factory-startup", "--python-exit-code", "1",
                             "--python", f"{WIN_TEMP_WIN}/dhr_rock_fallback.py"],
                            capture_output=True, text=True, check=False)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert "FALLBACK_OK" in combined
    assert "WARNING" in combined and "11.blend" in combined


_NO_EARTH_ASSET = '''
import os, sys, traceback
os.environ["HONOURS_BLENDER_ASSETS_DIR"] = {assets!r}  # empty: no TheEarth1.blend
sys.path.insert(0, {bc!r})
import bpy
from pathlib import Path
import DEMHeadlessRender as dhr
try:
    script = Path({bc!r}) / "DEMGrainsBlenderEarthCam.py"
    src = dhr.patch_script_source("per_sphere", script.read_text(encoding="utf-8"),
                                  grains_dir={grains!r}, bodies_dir={bodies!r}, max_frames=2)
    exec(compile(src, str(script), "exec"), {{"__name__": "__main__", "__file__": str(script)}})
    earth = bpy.data.objects.get("Earth")
    assert earth is not None, "no Earth empty"
    kids = [o.name for o in earth.children]
    assert any(n.startswith("Surface") for n in kids), kids
    print("BARE_EARTH_OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


@pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed")
@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
@pytest.mark.skipif(not FIXTURE_RUN.is_dir(), reason="fixture run not on disk")
def test_per_sphere_without_earth_model_gets_bare_sphere():
    # README promises a bare-sphere Earth when the asset pack has no TheEarth1.blend.
    (WIN_TEMP / "no_assets").mkdir(exist_ok=True)
    script = WIN_TEMP / "per_sphere_bare_earth.py"
    script.write_text(_NO_EARTH_ASSET.format(
        assets=WIN_TEMP_WIN + "/no_assets", bc=BLENDERCONVERT_WIN,
        grains=FIXTURE_RUN_WIN + "/run_0001_grains_output/",
        bodies=FIXTURE_RUN_WIN + "/run_0001_bodies_output/"), encoding="utf-8")
    r = subprocess.run([str(BLENDER_EXE), "--background", "--python", WIN_TEMP_WIN + "/per_sphere_bare_earth.py"],
                       capture_output=True, text=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0 and "BARE_EARTH_OK" in out, out[-3000:]


# -- --slowmo: more timeline frames over the same sim time (instance path) --

def test_patch_instance_slowmo_sets_factor():
    out = dhr.patch_script_source("instance", _real_src("instance"), manifest="C:/v/m.json", slowmo=4)
    assert "\nSLOWMO_FACTOR = 4\n" in out


def test_slowmo_flag_parses_and_defaults_to_1(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text('{"viz_mode": "instance"}')
    assert _parse(["--viz-path", "instance", "--manifest", str(m)]).slowmo == 1
    assert _parse(["--viz-path", "instance", "--manifest", str(m), "--slowmo", "4"]).slowmo == 4


@pytest.mark.parametrize("extra,match", [
    (["--viz-path", "per_sphere", "--grains-dir", "/g", "--bodies-dir", "/b", "--slowmo", "2"], "instance"),
    (["--viz-path", "instance", "--manifest", "/m", "--slowmo", "0"], ">= 1"),
])
def test_slowmo_rejected_off_instance_or_below_1(extra, match):
    with pytest.raises(ValueError, match=match):
        dhr.validate_args(_parse(extra))


def test_slowmo_range_starts_on_the_first_dump():
    # timeline frame S*k shows dump k (frame_map 100 -> 100*S); frames before S would
    # hold dump 1, so the render starts at S and ends on the last dump.
    from types import SimpleNamespace
    scene = SimpleNamespace(frame_start=1, frame_end=62 * 4)
    dhr.apply_slowmo_range(scene, 4)
    assert (scene.frame_start, scene.frame_end) == (4, 248)
    scene = SimpleNamespace(frame_start=1, frame_end=62)
    dhr.apply_slowmo_range(scene, 1)
    assert (scene.frame_start, scene.frame_end) == (1, 62)


# -- --resume: carry on an interrupted render, keeping the frames already on disk --

def test_resume_flag_parses_and_defaults_off(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text('{"viz_mode": "instance"}')
    assert _parse(["--viz-path", "instance", "--manifest", str(m)]).resume is False
    assert _parse(["--viz-path", "instance", "--manifest", str(m), "--resume"]).resume is True


def test_resume_skips_frames_already_written():
    from types import SimpleNamespace
    scene = SimpleNamespace(render=SimpleNamespace(use_overwrite=True, use_placeholder=True))
    dhr.apply_resume(scene, False)
    assert scene.render.use_overwrite is True
    dhr.apply_resume(scene, True)
    assert (scene.render.use_overwrite, scene.render.use_placeholder) == (False, False)


# -- render speed defaults: persistent data + 4K texture cap (measured 41 -> 16 s/frame, same pixels) --

def test_render_speedups_keep_scene_and_cap_textures():
    from types import SimpleNamespace
    scene = SimpleNamespace(objects=[], render=SimpleNamespace(use_persistent_data=False),
                            cycles=SimpleNamespace(texture_limit_render="OFF"))
    dhr.apply_render_speedups(scene)
    assert scene.render.use_persistent_data is True
    # per-image cap (cap_image_textures) instead of Cycles' global limit, so the clouds can stay full size
    assert scene.cycles.texture_limit_render == "OFF"


def test_image_textures_capped_at_4k_except_clouds():
    from types import SimpleNamespace as NS

    class Img(NS):
        def scale(self, w, h):
            self.size = (w, h)

    imgs = [Img(name="earth_nightlights_21K.tif", source="FILE", size=(21600, 10800)),
            Img(name="earth_clouds_8K.tif", source="FILE", size=(8192, 4096)),
            Img(name="small.png", source="FILE", size=(2048, 1024)),
            Img(name="Render Result", source="VIEWER", size=(0, 0))]
    dhr.cap_image_textures(imgs)
    assert [tuple(i.size) for i in imgs] == [(4096, 2048), (8192, 4096), (2048, 1024), (0, 0)]


def test_samples_default_is_124(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text('{"viz_mode": "instance"}')
    assert _parse(["--viz-path", "instance", "--manifest", str(m)]).samples == 124


# -- adaptive subdivision: the Earth asset dices at 1 px/edge, so memory grew with Earth's screen
#    size (40k flyby approach: 10 -> 20 GB, 15 -> 58 s/frame). Floor 4 px; off where nothing displaces.

def _subsurf_obj(name, pixel_size, displaced):
    from types import SimpleNamespace as NS
    mod = NS(type="SUBSURF", use_adaptive_subdivision=True, adaptive_space="PIXEL", adaptive_pixel_size=pixel_size)
    out = NS(type="OUTPUT_MATERIAL", inputs=[NS(name="Surface", is_linked=True),
                                             NS(name="Displacement", is_linked=displaced)])
    mat = NS(node_tree=NS(nodes=[out]))
    return NS(name=name, modifiers=[mod], material_slots=[NS(material=mat)]), mod


def test_adaptive_subdivision_gets_a_4px_floor_and_is_off_without_displacement():
    from types import SimpleNamespace as NS
    surface, s_mod = _subsurf_obj("Surface", 1.0, displaced=True)
    coarse, c_mod = _subsurf_obj("Coarse", 8.0, displaced=True)
    atmos, a_mod = _subsurf_obj("Atmosphere", 1.0, displaced=False)
    scene = NS(objects=[surface, coarse, atmos], render=NS(use_persistent_data=False),
               cycles=NS(texture_limit_render="OFF"))
    dhr.apply_render_speedups(scene)
    assert (s_mod.use_adaptive_subdivision, s_mod.adaptive_pixel_size) == (True, 4.0)
    assert c_mod.adaptive_pixel_size == 8.0  # already coarser: left alone
    assert a_mod.use_adaptive_subdivision is False


def test_gpu_render_denoises_on_the_gpu():
    # OIDN on the CPU was ~7 of 9.5 s per 1080p frame; on the GPU 2.6 s, same pixels (0.01/255)
    from types import SimpleNamespace
    scene = SimpleNamespace(cycles=SimpleNamespace(denoising_use_gpu=False))
    dhr.apply_gpu_denoise(scene)
    assert scene.cycles.denoising_use_gpu is True


# -- camera aim between keys: location keys are Bezier, rotation keys LINEAR, so with --slowmo
#    the body swung 2.5-4.5 deg sideways between keys (40k flyby approach). Track the core instead.

_CAMERA_AIM = r'''
import math, sys
sys.path.insert(0, r"{bc}")
sys.path.insert(0, r"{code}")
import bpy
from mathutils import Vector
import DEMCinematicSetup as cin
scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
n = 8
scene.frame_start, scene.frame_end = 1, n
earth = bpy.data.objects.new("Earth", None); scene.collection.objects.link(earth)
cores = []
for i in range(n):
    t = i - 4.0  # hyperbola-like pass: Earth sweeps round the body, fast near i = 4
    earth.location = (300.0 * t / (1 + abs(t)), 80.0, 0.0)
    earth.keyframe_insert("location", frame=1 + i)
    cores.append(Vector((0.3 * t * t, 2.0 * t, 0.1 * t)))
cam = cin.setup_camera(n, lambda i: cores[i], earth, True, rotation_interp="LINEAR")
# reference orientation at keys: what setup_camera has always keyed
ref = {{}}
for i in range(n):
    scene.frame_set(1 + i)
    look = (cores[i] - cam.matrix_world.translation).normalized()
    ref[1 + i] = look.to_track_quat("-Z", "Y")
S = 5
scene.frame_end = n * S
scene.render.frame_map_old, scene.render.frame_map_new = 100, 100 * S
aim = bpy.data.objects.get("DEM_CameraAim")
assert aim is not None, "no DEM_CameraAim target"
worst_between, worst_key, worst_roll = 0.0, 0.0, 0.0
for f in range(S, n * S + 1):
    scene.frame_set(f)
    m = cam.matrix_world
    look = -m.col[2].xyz.normalized()
    to_aim = (aim.matrix_world.translation - m.translation).normalized()
    worst_between = max(worst_between, math.degrees(look.angle(to_aim)))
    if f % S == 0:
        k = f // S
        to_core = (cores[k - 1] - m.translation).normalized()
        worst_key = max(worst_key, math.degrees(look.angle(to_core)))
        worst_roll = max(worst_roll, math.degrees(m.to_quaternion().rotation_difference(ref[k]).angle))
print(f"AIM between {{worst_between:.5f}} key {{worst_key:.5f}} roll {{worst_roll:.5f}}")
assert worst_between < 0.01 and worst_key < 0.01 and worst_roll < 0.01
print("CAMERA_AIM_OK")
'''


@pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted")
def test_camera_points_at_the_core_between_keys():
    script = WIN_TEMP / "camera_aim.py"
    script.write_text(_CAMERA_AIM.format(bc=BLENDERCONVERT_WIN, code=CODE_WIN), encoding="utf-8")
    r = subprocess.run([str(BLENDER_EXE), "--background", "--factory-startup", "--python",
                        WIN_TEMP_WIN + "/camera_aim.py"], capture_output=True, text=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0 and "CAMERA_AIM_OK" in out, out[-3000:]
