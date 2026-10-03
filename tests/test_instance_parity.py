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
VISIBLE_IDS = []
SCENE_OBJECTS = ("DEM_TrackCam", "DEM_SunLight", "DEM_CamFill", "Earth", "Moon", "Moon_Body",
                 "Sun_Visual", "Sun_Body")


def surface_points(mesh):
    """Vertices then face centres (local space); covers geometry and topology."""
    co = np.array([v.co[:] for v in mesh.vertices])
    centres = np.array([co[list(f.vertices)].mean(axis=0) for f in mesh.polygons])
    return np.vstack([co, centres]), len(mesh.vertices)


def grain_world_verts():
    """(n_visible, n_points, 3) world-space surface points, visible grains in grain_id order."""
    if VIZ_PATH == "per_sphere":
        objs = sorted((o for o in bpy.data.collections["DEM_Grains"].objects if not o.hide_render),
                      key=lambda o: int(o.name.rsplit("_", 1)[1]))
        VISIBLE_IDS.append([int(o.name.rsplit("_", 1)[1]) for o in objs])
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


def hide_key_counts():
    """Per-sphere grain name -> number of hide_render keyframes (grains with any)."""
    out = {{}}
    for o in bpy.data.collections["DEM_Grains"].objects:
        act = o.animation_data.action if o.animation_data else None
        if act is None:
            continue
        fcurves = list(getattr(act, "fcurves", None) or [])
        for layer in getattr(act, "layers", []):
            for strip in layer.strips:
                for bag in strip.channelbags:
                    fcurves.extend(bag.fcurves)
        n = sum(len(fc.keyframe_points) for fc in fcurves if fc.data_path == "hide_render")
        if n:
            out[o.name] = n
    return out


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
    np.savez({out_npz!r}, *verts)
    json.dump({{"objects": objects, "lights": lights, "visible_ids": VISIBLE_IDS,
               "hide_keys": hide_key_counts() if VIZ_PATH == "per_sphere" else {{}},
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


def _sample(viz_path, work=WORK, work_win=WORK_WIN, **paths):
    name = f"parity_{viz_path}"
    script = work / f"{name}.py"
    script.write_text(_SAMPLER.format(
        bc=BLENDERCONVERT_WIN, viz_path=viz_path, times=TIMES,
        out_npz=f"{work_win}/{name}.npz", out_json=f"{work_win}/{name}.json",
        grains=paths.get("grains"), bodies=paths.get("bodies"), manifest=paths.get("manifest"),
    ), encoding="utf-8")
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--python", f"{work_win}/{name}.py"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, ((result.stdout or "") + (result.stderr or ""))[-4000:]
    f = np.load(work / f"{name}.npz")
    return [f[f"arr_{i}"] for i in range(len(f.files))], json.loads((work / f"{name}.json").read_text())


def _instance_manifest(grains, bodies, out):
    spec = importlib.util.spec_from_file_location(
        "viz_preprocess_grains_instance", CODE / "viz" / "viz_preprocess_grains_instance.py"
    )
    vgi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgi)
    return vgi.build_grains_instance(grains, bodies, out)


@pytest.fixture(scope="module")
def sampled():
    grains, bodies = _scaled_fixture()
    _instance_manifest(grains, bodies, WORK / "run_0001_viz_instance")

    per_sphere = _sample("per_sphere", grains=f"{WORK_WIN}/run_0001_grains_output/",
                         bodies=f"{WORK_WIN}/run_0001_bodies_output/")
    instance = _sample("instance", manifest=f"{WORK_WIN}/run_0001_viz_instance/manifest.json")
    return (np.array(per_sphere[0]), per_sphere[1]), (np.array(instance[0]), instance[1])


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


def _assert_scene_objects_match(ps, ins):
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


def test_scene_objects_match(sampled):
    (_, ps), (_, ins) = sampled
    _assert_scene_objects_match(ps, ins)


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


WORK_VAR = WIN_TEMP / "parity_var"
WORK_VAR_WIN = WIN_TEMP_WIN + "/parity_var"


def _variable_fixture(work=WORK_VAR, empty_last=False):
    """Scaled fixture where one grain joins at frame 1 and five leave from frame 2.

    ``empty_last`` drops every grain from the last frame instead. Returns
    (grains_dir, bodies_dir, expected) with expected[i] = sorted visible grain
    ids at TIMES[i] (Blender frame f shows key f-1; visible iff present[floor]).
    """
    if work.exists():
        shutil.rmtree(work)
    grains = work / "run_0001_grains_output"
    bodies = work / "run_0001_bodies_output"
    grains.mkdir(parents=True)
    shutil.copytree(FIXTURE_RUN / "run_0001_bodies_output", bodies)
    srcs = sorted((FIXTURE_RUN / "run_0001_grains_output").glob("*.npz"))
    ids0 = np.load(srcs[0])["grain_id"]
    join, leave = int(ids0[0]), {int(g) for g in ids0[-5:]}
    present = []
    for k, src in enumerate(srcs):
        d = dict(np.load(src))
        n = len(d["grain_id"])
        drop = {join} if k == 0 else (leave if k >= 2 else set())
        if empty_last and k == len(srcs) - 1:
            drop = {int(g) for g in d["grain_id"]}
        keep = np.array([int(g) not in drop for g in d["grain_id"]])
        for key, v in d.items():
            if np.ndim(v) and len(v) == n:
                d[key] = v[keep]
        for key in ("x_vis", "y_vis", "z_vis", "Reff_vis"):
            d[key] = d[key] * SCALE
        np.savez_compressed(grains / src.name, **d)
        present.append(sorted(int(g) for g in d["grain_id"]))
    expected = [present[min(int(t) - 1, len(present) - 1)] for t in TIMES]
    return grains, bodies, expected


@pytest.fixture(scope="module")
def per_sphere_variable():
    grains, bodies, expected = _variable_fixture()
    ps = _sample("per_sphere", WORK_VAR, WORK_VAR_WIN,
                 grains=f"{WORK_VAR_WIN}/run_0001_grains_output/",
                 bodies=f"{WORK_VAR_WIN}/run_0001_bodies_output/")
    return ps, expected, grains, bodies


def test_per_sphere_hides_absent_grains(per_sphere_variable):
    (verts, meta), expected, _, _ = per_sphere_variable
    assert meta["visible_ids"] == expected
    assert [len(v) for v in verts] == [len(e) for e in expected]
    # The fixture really changes the set at both ends.
    assert len(expected[0]) == 502 and len(expected[-1]) == 498


def test_per_sphere_hide_keys_only_where_visibility_changes(per_sphere_variable):
    # One key at frame 1 and one where the grain joins or leaves, not one per frame.
    (_, meta), _, _, _ = per_sphere_variable
    assert len(meta["hide_keys"]) == 6
    assert set(meta["hide_keys"].values()) == {2}


@pytest.fixture(scope="module")
def sampled_variable(per_sphere_variable):
    ps, expected, grains, bodies = per_sphere_variable
    _instance_manifest(grains, bodies, WORK_VAR / "run_0001_viz_instance")
    ins = _sample("instance", WORK_VAR, WORK_VAR_WIN,
                  manifest=f"{WORK_VAR_WIN}/run_0001_viz_instance/manifest.json")
    return ps, ins, expected


def test_variable_set_surfaces_match(sampled_variable):
    (ps_verts, _), (in_verts, _), expected = sampled_variable
    assert [len(v) for v in in_verts] == [len(e) for e in expected]
    for t, a, b in zip(TIMES, ps_verts, in_verts):
        gap = _unordered_max_gap(a[None], b[None])[0]
        assert gap < 1e-3, (t, gap)


def test_variable_set_scene_objects_match(sampled_variable):
    (_, ps), (_, ins), _ = sampled_variable
    _assert_scene_objects_match(ps, ins)


def test_points_without_present_render_every_grain(sampled):
    # Output written before `present` existed: every point is drawn, no selection.
    (ps_verts, _), _ = sampled
    old = WORK / "run_0001_viz_instance_old"
    if old.exists():
        shutil.rmtree(old)
    shutil.copytree(WORK / "run_0001_viz_instance", old)
    for p in (old / "points").glob("*.npz"):
        d = dict(np.load(p))
        d.pop("present")
        np.savez_compressed(p, **d)
    verts, _ = _sample("instance", manifest=f"{WORK_WIN}/run_0001_viz_instance_old/manifest.json")
    gap = _unordered_max_gap(ps_verts, np.array(verts))
    assert (gap < 1e-3).all(), gap.tolist()


WORK_EMPTY = WIN_TEMP / "parity_empty"
WORK_EMPTY_WIN = WIN_TEMP_WIN + "/parity_empty"


@pytest.fixture(scope="module")
def sampled_empty_frame():
    grains, bodies, expected = _variable_fixture(WORK_EMPTY, empty_last=True)
    ps = _sample("per_sphere", WORK_EMPTY, WORK_EMPTY_WIN,
                 grains=f"{WORK_EMPTY_WIN}/run_0001_grains_output/",
                 bodies=f"{WORK_EMPTY_WIN}/run_0001_bodies_output/")
    _instance_manifest(grains, bodies, WORK_EMPTY / "run_0001_viz_instance")
    ins = _sample("instance", WORK_EMPTY, WORK_EMPTY_WIN,
                  manifest=f"{WORK_EMPTY_WIN}/run_0001_viz_instance/manifest.json")
    return ps, ins, expected


def test_frame_with_no_grains_keeps_camera_finite(sampled_empty_frame):
    # Every grain gone: both paths aim the camera at the Apophis CoM (origin).
    (ps_verts, ps), (in_verts, ins), expected = sampled_empty_frame
    assert expected[-1] == [] and ps["visible_ids"][-1] == []
    assert len(ps_verts[-1]) == 0 and len(in_verts[-1]) == 0
    for meta in (ps, ins):
        cams = np.array([o["DEM_TrackCam"] for o in meta["objects"]])
        assert np.isfinite(cams).all()
    _assert_scene_objects_match(ps, ins)
