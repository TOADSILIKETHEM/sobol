"""instance_grains must reproduce per_sphere exactly (geometry, motion, scene).

Both Blender scripts run headless on the same run; at integer and sub-frame
times every grain's world-space mesh vertices, and every non-grain scene
object's transform, must agree. Grains are scaled x3 from the fixture so the
texture-scale and radius paths are exercised at a realistic grain size.
"""
import importlib.util
import json
import shutil
import subprocess

import numpy as np
import pytest

from tests._paths import CODE
from tests.test_dem_headless_paths import (
    BLENDER_EXE, BLENDERCONVERT_WIN, FIXTURE_RUN, WIN_TEMP, WIN_TEMP_WIN,
)

SCALE = 3.0
TIMES = [1.0, 1.25, 1.5, 2.0, 2.6, 3.3, 4.0]
WORK = WIN_TEMP / "parity"
WORK_WIN = WIN_TEMP_WIN + "/parity"

pytestmark = [
    pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed"),
    pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted"),
]

_SAMPLER = '''
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, {bc!r})
import bpy
import numpy as np
import DEMHeadlessRender as dhr

VIZ_PATH = {viz_path!r}
TIMES = {times!r}
SCENE_OBJECTS = ("DEM_TrackCam", "DEM_SunLight", "DEM_CamFill", "Earth", "Moon", "Moon_Body",
                 "Sun_Visual", "Sun_Body")


def surface_points(mesh):
    """Vertices then face centres (local space); covers geometry and topology."""
    co = np.array([v.co[:] for v in mesh.vertices])
    centres = np.array([co[list(f.vertices)].mean(axis=0) for f in mesh.polygons])
    return np.vstack([co, centres]), len(mesh.vertices)


def grain_world_verts():
    """(n_grains, n_points, 3) world-space surface points, grains in grain_id order."""
    if VIZ_PATH == "per_sphere":
        objs = sorted(bpy.data.collections["DEM_Grains"].objects, key=lambda o: o.name)
        out = []
        for o in objs:
            pts, _ = surface_points(o.data)
            m = np.array(o.matrix_world)
            out.append(pts @ m[:3, :3].T + m[:3, 3])
        return np.array(out)
    dg = bpy.context.evaluated_depsgraph_get()
    pts, _ = surface_points(bpy.data.objects["DEM_InstanceMesh"].data)
    out = []
    for inst in dg.object_instances:
        if inst.is_instance and inst.parent and inst.parent.name == "DEM_InstancePoints":
            m = np.array(inst.matrix_world)
            out.append(pts @ m[:3, :3].T + m[:3, 3])
    return np.array(out)


try:
    script = Path({bc!r}) / dhr.SCRIPT_FOR_VIZ_PATH[VIZ_PATH if VIZ_PATH == "per_sphere" else "instance"]
    if VIZ_PATH == "per_sphere":
        src = dhr.patch_script_source("per_sphere", script.read_text(encoding="utf-8"),
                                      grains_dir={grains!r}, bodies_dir={bodies!r})
    else:
        dhr.ensure_procedural_rock_material()
        src = dhr.patch_script_source("instance", script.read_text(encoding="utf-8"),
                                      manifest={manifest!r})
    exec(compile(src, str(script), "exec"), {{"__name__": "__main__", "__file__": str(script)}})
    scene = bpy.context.scene
    verts, objects = [], []
    for t in TIMES:
        f = int(t)
        scene.frame_set(f, subframe=t - f)
        verts.append(grain_world_verts())
        objects.append({{name: np.array(bpy.data.objects[name].matrix_world).tolist()
                        for name in SCENE_OBJECTS if name in bpy.data.objects}})
    lights = {{o.name: [o.data.type, o.data.energy] for o in bpy.data.objects if o.type == "LIGHT"}}
    cam = bpy.data.objects["DEM_TrackCam"].data
    np.save({out_npy!r}, np.array(verts))
    json.dump({{"objects": objects, "lights": lights,
               "camera": [cam.angle, cam.clip_start, cam.clip_end],
               "frame_range": [scene.frame_start, scene.frame_end]}},
              open({out_json!r}, "w"))
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


def _scaled_fixture():
    """Fixture run with grain positions and radii x SCALE (4 frames)."""
    grains = WORK / "run_0001_grains_output"
    bodies = WORK / "run_0001_bodies_output"
    if WORK.exists():
        shutil.rmtree(WORK)
    grains.mkdir(parents=True)
    shutil.copytree(FIXTURE_RUN / "run_0001_bodies_output", bodies)
    for src in sorted((FIXTURE_RUN / "run_0001_grains_output").glob("*.npz")):
        d = dict(np.load(src))
        for key in ("x_vis", "y_vis", "z_vis", "Reff_vis"):
            d[key] = d[key] * SCALE
        np.savez_compressed(grains / src.name, **d)
    return grains, bodies


def _sample(viz_path, **paths):
    name = f"parity_{viz_path}"
    script = WORK / f"{name}.py"
    script.write_text(_SAMPLER.format(
        bc=BLENDERCONVERT_WIN, viz_path=viz_path, times=TIMES,
        out_npy=f"{WORK_WIN}/{name}.npy", out_json=f"{WORK_WIN}/{name}.json",
        grains=paths.get("grains"), bodies=paths.get("bodies"), manifest=paths.get("manifest"),
    ), encoding="utf-8")
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", f"{WORK_WIN}/{name}.py"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, ((result.stdout or "") + (result.stderr or ""))[-4000:]
    return np.load(WORK / f"{name}.npy"), json.loads((WORK / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def sampled():
    grains, bodies = _scaled_fixture()
    spec = importlib.util.spec_from_file_location(
        "viz_preprocess_grains_instance", CODE / "viz" / "viz_preprocess_grains_instance.py"
    )
    vgi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgi)
    vgi.build_grains_instance(grains, bodies, WORK / "run_0001_viz_instance")

    per_sphere = _sample("per_sphere", grains=f"{WORK_WIN}/run_0001_grains_output/",
                         bodies=f"{WORK_WIN}/run_0001_bodies_output/")
    instance = _sample("instance", manifest=f"{WORK_WIN}/run_0001_viz_instance/manifest.json")
    return per_sphere, instance


def _unordered_max_gap(a, b):
    """Per time: max over grains and points of a's distance to the nearest point of b's same grain.

    bpy.ops.mesh.primitive_uv_sphere_add and bmesh.ops.create_uvsphere build the
    same sphere with a different vertex numbering, so match points as sets.
    """
    gaps = []
    for at, bt in zip(a, b):
        d = np.linalg.norm(at[:, :, None, :] - bt[:, None, :, :], axis=-1)
        gaps.append(max(d.min(axis=2).max(), d.min(axis=1).max()))
    return np.array(gaps)


def test_grain_surfaces_match_at_keys_and_between_them(sampled):
    (ps_verts, _), (in_verts, _) = sampled
    assert ps_verts.shape == in_verts.shape, (ps_verts.shape, in_verts.shape)
    assert ps_verts.shape[1] == 503
    # Grains actually move between samples (else this proves nothing).
    assert np.abs(ps_verts[-1] - ps_verts[0]).max() > 0.1
    gap = _unordered_max_gap(ps_verts, in_verts)
    assert (gap < 1e-3).all(), dict(zip([str(t) for t in TIMES], gap.tolist()))


def test_scene_objects_match(sampled):
    (_, ps), (_, ins) = sampled
    assert ps["frame_range"] == ins["frame_range"]
    assert ps["lights"].keys() == ins["lights"].keys()
    for name, (kind, energy) in ps["lights"].items():
        assert ins["lights"][name][0] == kind
        assert ins["lights"][name][1] == pytest.approx(energy)
    assert ins["camera"] == pytest.approx(ps["camera"])
    for t, ps_objs, in_objs in zip(TIMES, ps["objects"], ins["objects"]):
        assert ps_objs.keys() == in_objs.keys(), t
        for name in ps_objs:
            a, b = np.array(ps_objs[name]), np.array(in_objs[name])
            tol = 1e-4 * max(1.0, np.abs(a[:3, 3]).max())
            assert np.abs(a - b).max() <= tol, (t, name, a.tolist(), b.tolist())


def _render(viz_path, out_name, **paths):
    args = [str(BLENDER_EXE), "--background", "--python", BLENDERCONVERT_WIN + r"\DEMHeadlessRender.py", "--",
            "--viz-path", viz_path, "--output-dir", f"{WORK_WIN}/{out_name}",
            "--resolution", "480x270", "--samples", "512", "--max-frames", "1"]
    if viz_path == "per_sphere":
        args += ["--grains-dir", paths["grains"], "--bodies-dir", paths["bodies"]]
    else:
        args += ["--manifest", paths["manifest"]]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, ((result.stdout or "") + (result.stderr or ""))[-3000:]
    from PIL import Image
    return np.asarray(Image.open(WORK / out_name / "frame_0001.png").convert("RGB")).astype(float) / 255


def test_rendered_frames_match(sampled):
    # Same geometry, so the only difference left is Monte Carlo noise: the
    # instance BVH and the per-object meshes decorrelate grazing shadow rays
    # between touching grains. Measured 2026-09-27 (480x270, 512 spp): mean
    # |diff| 9.7e-5, no pixel > 0.05, body bias -4e-5 on brightness 0.116.
    # Without the radius-scaled texture coordinates the mean is ~2x larger at 32 spp.
    ps = _render("per_sphere", "render_parity_ps", grains=f"{WORK_WIN}/run_0001_grains_output",
                 bodies=f"{WORK_WIN}/run_0001_bodies_output")
    ins = _render("instance", "render_parity_in", manifest=f"{WORK_WIN}/run_0001_viz_instance/manifest.json")
    d = ins - ps
    body = ps.max(axis=2) > 0.05
    assert body.mean() > 0.01  # the body is in frame
    assert np.abs(d).mean() < 3e-4
    assert np.abs(d).max(axis=2).max() < 0.1
    assert abs(d[body].mean()) < 5e-4
