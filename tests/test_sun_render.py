"""Sun disk brightness vs the sun lamp, and the headless driver's Sun glare (Blender).

Each test renders tiny Cycles CPU frames (64x64, 16 samples, Standard view, 32-bit
EXR, black world) in blender.exe and reads the linear pixels back.
"""
import json
import subprocess
import sys

import pytest

from tests._paths import CODE
from tests.test_dem_headless_paths import BLENDER_EXE, BLENDERCONVERT_WIN, WIN_TEMP, WIN_TEMP_WIN

sys.path.insert(0, str(CODE / 'BlenderConvert'))
import sun_geometry as sg  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not BLENDER_EXE.is_file(), reason="blender.exe not installed"),
    pytest.mark.skipif(not WIN_TEMP.is_dir(), reason="Windows Temp not mounted"),
]

# Shared by every script: tiny linear render of the current scene to EXR, read back.
_PRELUDE = '''
import json, math, sys, traceback
sys.path.insert(0, {bc!r})
import bpy
import numpy as np

TEMP = {temp!r}

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    return scene

def add_camera(location, angle_deg, clip_end=1000.0):
    cam = bpy.data.objects.new('TestCam', bpy.data.cameras.new('TestCam'))
    bpy.context.scene.collection.objects.link(cam)
    cam.location = location
    cam.data.angle = math.radians(angle_deg)
    cam.data.clip_end = clip_end
    bpy.context.scene.camera = cam
    return cam

def diffuse_material(name):
    mat = bpy.data.materials.new(name)
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.0
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def render_rgb(name):
    scene = bpy.context.scene
    path = TEMP + '/' + name + '.exr'
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path, check_existing=False)
    px = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(-1, 4)[:, :3]
'''

_CALIBRATION = '''
import DEMCinematicSetup as cin
import sun_geometry
try:
    # Scene A: white diffuse plane facing a Sun lamp of SUN_LAMP_ENERGY.
    scene = clear_scene()
    bpy.ops.mesh.primitive_plane_add(size=100.0)
    bpy.context.object.data.materials.append(diffuse_material('White'))
    lamp = bpy.data.objects.new('Lamp', bpy.data.lights.new('Lamp', 'SUN'))
    lamp.data.energy = cin.SUN_LAMP_ENERGY
    scene.collection.objects.link(lamp)
    add_camera((0.0, 0.0, 5.0), 40.0)
    p_a = float(render_rgb('sunt5_plane').mean())
    # Scene B: the Sun_Body sphere alone, filling the frame.
    scene = clear_scene()
    cin.ensure_sun_visual(True, {bodies!r})
    r = sun_geometry.sun_visual_radius_bu()
    add_camera((0.0, 0.0, 3.0 * r), 10.0, clip_end=10.0 * r)
    p_b = float(render_rgb('sunt5_disk').mean())
    print('SUN_JSON:' + json.dumps({{'p_a': p_a, 'p_b': p_b}}))
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

_GLARE_NODES = '''
import DEMHeadlessRender as dhr
def glare_nodes(scene):
    tree = scene.compositing_node_group
    return [] if tree is None else [n for n in tree.nodes if n.bl_idname == 'CompositorNodeGlare']
try:
    scene = clear_scene()
    out = {{'empty': dhr.apply_sun_glare(scene), 'empty_glare': len(glare_nodes(scene))}}
    bpy.context.scene.collection.objects.link(bpy.data.objects.new('Sun_Visual', None))
    out['first'] = dhr.apply_sun_glare(scene)
    out['second'] = dhr.apply_sun_glare(scene)
    tree = scene.compositing_node_group
    glare = glare_nodes(scene)
    out['glare'] = len(glare)
    out['groups'] = sum(1 for g in bpy.data.node_groups if g.bl_idname == 'CompositorNodeTree')
    out['use_compositing'] = scene.render.use_compositing
    out['links'] = sorted((l.from_node.bl_idname, l.to_node.bl_idname) for l in tree.links)
    out['types'] = sorted(g.inputs['Type'].default_value for g in glare)
    kernels = [i for i in bpy.data.images if i.name.startswith(('DEM_SunStarburst', 'DEM_SunGhost_'))]
    out['kernels'] = sorted([i.name, i.packed_file is not None] for i in kernels)
    out['starburst_size'] = list(bpy.data.images[dhr.SUN_GLARE_KERNEL_IMAGE].size)
    out['transforms'] = sorted(round(n.inputs['Scale'].default_value, 2)
                               for n in tree.nodes if n.bl_idname == 'CompositorNodeTransform')
    print('SUN_JSON:' + json.dumps(out))
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

_SPARES = '''
import DEMCinematicSetup as cin
import DEMHeadlessRender as dhr
try:
    # Lit grain stand-in: white diffuse sphere under the sun lamp and the camera fill
    # light, at the default camera distance and fov; no Sun in frame.
    scene = clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3 * cin.CAM_DIST_BU)
    bpy.context.object.data.materials.append(diffuse_material('White'))
    lamp = bpy.data.objects.new('Lamp', bpy.data.lights.new('Lamp', 'SUN'))
    lamp.data.energy = cin.SUN_LAMP_ENERGY
    lamp.rotation_euler = (math.radians(30.0), 0.0, math.radians(30.0))
    scene.collection.objects.link(lamp)
    cam = add_camera((0.0, -cin.CAM_DIST_BU, 0.0), cin.CAMERA_FOV_DEG)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    cin.add_camera_fill_light(cam)
    plain = render_rgb('sunt5_spares_plain')
    scene.collection.objects.link(bpy.data.objects.new('Sun_Visual', None))
    assert dhr.apply_sun_glare(scene)
    glared = render_rgb('sunt5_spares_glare')
    print('SUN_JSON:' + json.dumps({{'plain_max': float(plain.max()),
                                     'glare_max': float(glared.max())}}))
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


def _run(name, body, **fmt):
    script = WIN_TEMP / f"sunt5_{name}.py"
    script.write_text(_PRELUDE.format(bc=BLENDERCONVERT_WIN, temp=WIN_TEMP_WIN)
                      + body.format(**fmt), encoding="utf-8")
    result = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", f"{WIN_TEMP_WIN}/sunt5_{name}.py"],
        capture_output=True, text=True, check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    line = next(ln for ln in combined.splitlines() if ln.startswith('SUN_JSON:'))
    return json.loads(line[len('SUN_JSON:'):])


def test_sun_disk_matches_lamp_radiance():
    # A disk of radiance L and angular radius a gives irradiance E = L * pi * a^2; a white
    # Lambertian plane under E has radiance E / pi. So disk / plane = 1 / a^2.
    from tests.test_hyperbola_sun_scene import EPHEMERIS_BODIES, _win
    out = _run("calibration", _CALIBRATION, bodies=_win(EPHEMERIS_BODIES))
    print(f"P_a={out['p_a']:.6g} P_b={out['p_b']:.6g} ratio={out['p_b'] / out['p_a']:.6g}")
    assert out['p_b'] / out['p_a'] == pytest.approx(1.0 / sg.SUN_ANGULAR_RADIUS_RAD ** 2, rel=0.05)


def test_glare_only_with_sun_visual():
    out = _run("glare_nodes", _GLARE_NODES)
    assert out['empty'] is False
    assert out['empty_glare'] == 0
    assert out['first'] is True and out['second'] is True
    # Bloom, the starburst and 7 ghosts, each ghost placed by one Transform.
    assert out['glare'] == 9
    assert out['types'] == ['Bloom'] + ['Kernel'] * 8
    assert out['starburst_size'] == [1023, 1023]
    assert out['kernels'] == sorted([['DEM_SunStarburst', True]]
                                    + [[f'DEM_SunGhost_{i}', True] for i in range(7)])
    assert out['transforms'] == [0.33, 0.38, 0.52, 0.62, 0.75, 1.3, 1.45]
    assert out['groups'] == 1
    assert out['use_compositing'] is True
    # Each pass: RLayers -> Glare -> (Transform) -> its own Add; Adds chained -> output.
    links = [tuple(l) for l in out['links']]
    assert links.count(('CompositorNodeRLayers', 'CompositorNodeGlare')) == 9
    assert links.count(('CompositorNodeImage', 'CompositorNodeGlare')) == 8
    assert links.count(('CompositorNodeGlare', 'CompositorNodeTransform')) == 7
    assert links.count(('CompositorNodeTransform', 'ShaderNodeMix')) == 7
    assert links.count(('CompositorNodeGlare', 'ShaderNodeMix')) == 2
    assert links.count(('ShaderNodeMix', 'ShaderNodeMix')) == 8
    assert links.count(('ShaderNodeMix', 'NodeGroupOutput')) == 1

def test_glare_spares_lit_surfaces():
    out = _run("spares", _SPARES)
    assert out['plain_max'] > 1.0  # the sphere is lit, not black
    assert out['glare_max'] == pytest.approx(out['plain_max'], abs=1e-3)
