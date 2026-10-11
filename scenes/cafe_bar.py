"""The Jumper Barista cafe: a wooden bar with the accepted brew station on it.

Simulation only. The world is generated, not hand-written:

- the **brew station** (server, V60 and free gooseneck pitcher, with the hero and
  overhead cameras the M4 verifier looks up) is read from the hash-checked
  ``scenes/coffee_bar.scene`` package and moved by one rigid translation, so its
  masses, dimensions and collision geometry are the accepted ones by
  construction; the accepted counter slab, back lip, lights and standing-slot
  marker are dropped because the bar replaces them;
- everything else -- the bar, the room, the off-camera entrance door with its
  ``door_chime`` site, the kettle and cup dressing, the warm lights and the
  cameras -- is project-authored primitive geometry and one procedurally generated
  wood texture, dimensioned by ``barista/cafe/cafe_dimensions.json``.

The generated world is written as an ordinary ``kk-scene-package/1`` (manifest
with SHA-256 per file) and loaded through ``scenes.package.load_package``, so it
gets the same treatment as ``coffee_bar``: the accepted counter slab (moved with
the station, invisible under a wooden visual top) becomes the task's ``terrain``
geom, every static collidable geom -- including the bar-top extensions -- is
rebuilt into the terrain body, and the pitcher stays a free body.

Frame: origin on the bar top at Jumper's start slot, z up, bar top z = 0, the bar
runs along +X, the customer (camera, no 3D human) is on -Y looking +Y.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .registry import Headlight, Scene

REPO = Path(__file__).resolve().parents[1]
DIMENSIONS = REPO / "barista" / "cafe" / "cafe_dimensions.json"
COFFEE_PACKAGE = Path(__file__).with_name("coffee_bar.scene")
SCHEMA = "jumper-barista-cafe/1"
WOOD_TEXTURE = "cafe_wood.png"
#: The accepted package's own bodies and cameras, moved as one rigid body.
STATION_BODIES = ("server", "v60", "pitcher")
STATION_CAMERAS = ("hero", "overhead")
#: What of the accepted package the bar replaces.
#: The accepted counter slab is kept -- translated with the station, its collision
#: untouched, only made invisible -- because the M4 replay is sensitive to the
#: support it stands on: measured, a differently sized bar-top box under the same
#: robot changes box-vs-mesh contact enough to move the carried grasp by 1.3 mm.
STATION_GEOMS = ("counter_surface",)
DROPPED_GEOMS = ("counter_back",)
DROPPED_SITES = ("jumper_standing_slot", "jumper_forward")
DROPPED_LIGHTS = ("key", "fill")
GROUND_GEOM = "counter_surface"
#: The bar top beyond the accepted slab: three coplanar collidable extensions.
EXTENSIONS = ("bar_top_ext_left", "bar_top_ext_right", "bar_top_ext_back")
INVISIBLE = {"rgba": [0.0, 0.0, 0.0, 0.0], "group": 3}
VISUAL_GROUP = 1

#: The visual V60 shell, in the dripper's own frame: from just above the support
#: plate to just below the rim, following the accepted ribs' outer surface.
SHELL_Z = (0.0045, 0.0865)
SHELL_THICKNESS = 0.002

_BUILT: dict[str, Any] = {}


def shell_inner_radius(z: float) -> float:
    """Inner radius of the visual shell at height `z` in the dripper frame.

    The accepted ribs' centre lines run from 12 mm at z = 5 mm to 37 mm at z = 85 mm
    with a 2 mm radius; the shell sits 0.8 mm outside them, so it never hides a rib
    contact and the stream (inside 33 mm) never meets it.
    """
    return 0.012 + (0.037 - 0.012) / 0.080 * (z - 0.005) + 0.0028


def dimensions() -> dict[str, Any]:
    """The cafe's authored dimensions and route, schema-checked."""
    data = json.loads(DIMENSIONS.read_text("utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{DIMENSIONS} declares {data.get('schema')!r}, expected {SCHEMA!r}")
    return data


# -- small geometry helpers ---------------------------------------------------


def _look_at_quat(pos, target) -> list[float]:
    """Camera quaternion (wxyz) looking from `pos` at `target`, world z up."""
    import mujoco
    import numpy as np

    forward = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # MuJoCo cameras look along their local -z with +y up.
    mat = np.column_stack([right, up, -forward]).ravel()
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat.tolist()


def _frustum_shell(z0: float, z1: float, r_in, thickness: float, segments: int = 48):
    """A closed, watertight conical shell (vertex list, face list) about +z."""
    verts: list[list[float]] = []
    rings = (
        (z0, r_in(z0) + thickness),  # 0 outer bottom
        (z1, r_in(z1) + thickness),  # 1 outer top
        (z1, r_in(z1)),  # 2 inner top
        (z0, r_in(z0)),  # 3 inner bottom
    )
    for z, radius in rings:
        for k in range(segments):
            angle = 2.0 * math.pi * k / segments
            verts.append([radius * math.cos(angle), radius * math.sin(angle), z])
    faces: list[list[int]] = []
    for ring in range(4):
        nxt = (ring + 1) % 4
        for k in range(segments):
            a = ring * segments + k
            b = ring * segments + (k + 1) % segments
            c = nxt * segments + (k + 1) % segments
            d = nxt * segments + k
            faces += [[a, b, c], [a, c, d]]
    return verts, faces


def _wood_png() -> bytes:
    """A deterministic, project-generated warm wood-grain texture (PNG bytes)."""
    import io

    import numpy as np
    from PIL import Image

    width, height = 1024, 256
    rng = np.random.default_rng(20261011)
    y, x = np.mgrid[0:height, 0:width].astype(float)
    warp = np.zeros_like(x)
    for scale, amp in ((180.0, 9.0), (61.0, 3.0), (23.0, 1.2)):
        phase = rng.uniform(0, 2 * math.pi, 2)
        warp += amp * np.sin(x / scale + phase[0]) * np.cos(y / (scale * 0.37) + phase[1])
    grain = 0.5 + 0.5 * np.sin((y + warp) / 3.1)
    rings = 0.5 + 0.5 * np.sin((y + 2.5 * warp) / 17.0)
    fleck = rng.normal(0.0, 0.035, size=(height, width))
    plank = ((x // 256) % 2) * 0.05
    shade = np.clip(0.55 + 0.22 * grain * rings + 0.1 * rings + fleck - plank, 0.0, 1.0)
    shade[:, ::256] *= 0.55  # plank seams
    base = np.array([0.50, 0.30, 0.16])
    dark = np.array([0.26, 0.14, 0.07])
    rgb = dark + (base - dark) * shade[..., None]
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


# -- world construction -------------------------------------------------------


def build_world(dims: dict[str, Any] | None = None) -> tuple[str, dict[str, bytes]]:
    """Return the cafe world MJCF and the extra asset files it names.

    Two halves, joined as XML:

    - the **accepted station**: the server, V60 and pitcher ``<body>`` elements and
      the hero/overhead ``<camera>`` elements of the hash-checked
      ``coffee_bar.scene/scene.xml``, copied *verbatim* (attribute text and all)
      into one ``<frame pos=authored_offset_m>``, together with its compiler,
      defaults and assets. Nothing of it passes through ``MjSpec.to_xml``, which
      writes six significant digits and would move the pitcher's geometry by
      fractions of a micrometre -- enough, in a 57 s contact-rich replay, to be
      a different experiment;
    - the **cafe**: authored with ``MjSpec`` on an emptied copy of the same
      package (so its geoms inherit the accepted defaults) and serialised.
    """
    import xml.etree.ElementTree as ET

    import mujoco

    from .package import read_package

    dims = dims or dimensions()
    coffee = read_package(COFFEE_PACKAGE)  # SHA-256 verified
    world_file = coffee.manifest["world"]["file"]
    offset = [float(v) for v in dims["brew_station"]["authored_offset_m"]]

    # The cafe half: the accepted package's defaults and assets, nothing else.
    spec = coffee.spec(world_file)
    station_v60_pos = [spec.body("v60").pos[i] + offset[i] for i in range(3)]
    for body in list(spec.worldbody.bodies):
        spec.delete(body)
    for element in (*spec.worldbody.geoms, *spec.worldbody.sites, *spec.worldbody.cameras,
                    *spec.worldbody.lights):
        spec.delete(element)
    world = spec.worldbody
    visual = {"contype": 0, "conaffinity": 0, "group": VISUAL_GROUP}
    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    cap = mujoco.mjtGeom.mjGEOM_CAPSULE
    sph = mujoco.mjtGeom.mjGEOM_SPHERE

    # Materials: one generated wood texture, the rest flat colours.
    tex = spec.add_texture(name="cafe_wood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           file=WOOD_TEXTURE)
    del tex
    wood = spec.add_material(name="cafe_wood_top", texrepeat=[2.0, 1.0], texuniform=False,
                             reflectance=0.04, specular=0.3, shininess=0.35)
    wood.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cafe_wood"
    panel = spec.add_material(name="cafe_wood_panel", texrepeat=[3.0, 2.0],
                              rgba=[0.62, 0.55, 0.50, 1.0], specular=0.2)
    panel.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cafe_wood"
    spec.add_texture(name="cafe_floor_tiles", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                     rgb1=[0.16, 0.12, 0.10], rgb2=[0.22, 0.17, 0.13])
    floor_mat = spec.add_material(name="cafe_floor", texrepeat=[6.0, 6.0], texuniform=True,
                                  reflectance=0.05)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cafe_floor_tiles"
    colours = {
        "cafe_plaster": [0.42, 0.30, 0.22, 1.0],
        "cafe_brass": [0.78, 0.58, 0.28, 1.0],
        "cafe_dark": [0.08, 0.07, 0.065, 1.0],
        "cafe_kettle": [0.12, 0.12, 0.12, 1.0],
        "cafe_cup": [0.90, 0.88, 0.82, 1.0],
        "cafe_door": [0.30, 0.20, 0.13, 1.0],
        "cafe_glass": [0.70, 0.80, 0.82, 0.25],
        "cafe_jar": [0.55, 0.40, 0.25, 0.85],
        "cafe_coffee": [0.17, 0.09, 0.045, 1.0],
        "cafe_ceramic": [0.95, 0.94, 0.91, 1.0],
    }
    for name, rgba in colours.items():
        spec.add_material(name=name, rgba=rgba, specular=0.3 if name != "cafe_plaster" else 0.05,
                          reflectance=0.25 if name in ("cafe_brass", "cafe_kettle") else 0.0)
    lamp = spec.add_material(name="cafe_lamp", rgba=[1.0, 0.78, 0.45, 1.0])
    lamp.emission = 2.5

    # -- the bar. Collision: the accepted counter slab (moved with the station, see
    # STATION_GEOMS; it is the declared ground) plus coplanar extensions out to the
    # bar's edges, all invisible. Look: one wooden visual top over the whole bar,
    # 0.3 mm below the collision surface so it never shows through a foot.
    bar = dims["bar"]
    x0, x1 = bar["top_x_range_m"]
    y0, y1 = bar["top_y_range_m"]
    t = float(bar["top_thickness_m"])
    contact = bar["contact"]
    sx0, sx1 = (offset[0] + v for v in bar["station_slab_x_range_m"])
    sy0, sy1 = (offset[1] + v for v in bar["station_slab_y_range_m"])
    if not (x0 < sx0 < sx1 < x1 and abs(sy0 - y0) < 1e-9 and sy1 < y1):
        raise ValueError("the accepted counter slab must sit inside the bar, flush with its "
                         "front edge")
    exact: dict[str, dict[str, list[float]]] = {}
    for name, (ax, bx, ay, by) in zip(EXTENSIONS, (
        (x0, sx0, y0, y1), (sx1, x1, y0, y1), (sx0, sx1, sy1, y1),
    ), strict=True):
        # MjSpec.to_xml keeps six significant digits; the seams must close exactly,
        # so these two attributes are rewritten at full precision below.
        exact[name] = {"size": [(bx - ax) / 2, (by - ay) / 2, t / 2],
                       "pos": [(ax + bx) / 2, (ay + by) / 2, -t / 2]}
        world.add_geom(
            name=name, type=box, **exact[name], contype=1, conaffinity=1,
            condim=int(contact["condim"]), friction=list(contact["friction"]),
            solref=list(contact["solref"]), **INVISIBLE,
        )
    world.add_geom(name="cafe_bar_top_visual", type=box, material="cafe_wood_top",
                   size=[(x1 - x0) / 2, (y1 - y0) / 2, (t - 0.0003) / 2],
                   pos=[(x0 + x1) / 2, (y0 + y1) / 2, -0.0003 - (t - 0.0003) / 2], **visual)
    floor_z = float(bar["floor_z_m"])
    front_y = float(bar["front_panel_y_m"])
    body_h = -t - floor_z
    world.add_geom(name="cafe_bar_front", type=box, material="cafe_wood_panel",
                   size=[(x1 - x0) / 2 - 0.01, 0.02, body_h / 2],
                   pos=[(x0 + x1) / 2, front_y, -t - body_h / 2], **visual)
    world.add_geom(name="cafe_bar_body", type=box, material="cafe_dark",
                   size=[(x1 - x0) / 2 - 0.02, (y1 - front_y) / 2 - 0.03, body_h / 2],
                   pos=[(x0 + x1) / 2, (front_y + y1) / 2, -t - body_h / 2], **visual)
    world.add_geom(name="cafe_foot_rail", type=cap, size=[0.012, 0, 0], material="cafe_brass",
                   fromto=[x0 + 0.05, front_y - 0.14, floor_z + 0.25,
                           x1 - 0.05, front_y - 0.14, floor_z + 0.25], **visual)
    world.add_geom(name="cafe_edge_trim", type=cap, size=[0.006, 0, 0], material="cafe_brass",
                   fromto=[x0, y0 + 0.004, -0.008, x1, y0 + 0.004, -0.008], **visual)

    # -- the room
    room_x = (-2.2, 2.6)
    room_y = (-3.2, 1.25)
    ceiling_z = 1.75
    world.add_geom(name="cafe_floor", type=box, material="cafe_floor",
                   size=[(room_x[1] - room_x[0]) / 2, (room_y[1] - room_y[0]) / 2, 0.01],
                   pos=[sum(room_x) / 2, sum(room_y) / 2, floor_z - 0.01], **visual)
    wall_h = ceiling_z - floor_z
    world.add_geom(name="cafe_back_wall", type=box, material="cafe_plaster",
                   size=[(room_x[1] - room_x[0]) / 2, 0.02, wall_h / 2],
                   pos=[sum(room_x) / 2, room_y[1], floor_z + wall_h / 2], **visual)
    world.add_geom(name="cafe_right_wall", type=box, material="cafe_plaster",
                   size=[0.02, (room_y[1] - room_y[0]) / 2, wall_h / 2],
                   pos=[room_x[1], sum(room_y) / 2, floor_z + wall_h / 2], **visual)
    door = dims["door"]
    wall_x = float(door["wall_x_m"])
    dy0 = float(door["centre_y_m"]) - float(door["width_m"]) / 2
    dy1 = float(door["centre_y_m"]) + float(door["width_m"]) / 2
    door_top = floor_z + float(door["height_m"])
    # Side wall with a door opening: three panels around the doorway.
    for name, ya, yb, za, zb in (
        ("cafe_side_wall_a", room_y[0], dy0, floor_z, ceiling_z),
        ("cafe_side_wall_b", dy1, room_y[1], floor_z, ceiling_z),
        ("cafe_side_wall_lintel", dy0, dy1, door_top, ceiling_z),
    ):
        world.add_geom(name=name, type=box, material="cafe_plaster",
                       size=[0.02, (yb - ya) / 2, (zb - za) / 2],
                       pos=[wall_x, (ya + yb) / 2, (za + zb) / 2], **visual)
    for name, ya, yb, za, zb in (
        ("cafe_door_frame_a", dy0 - 0.05, dy0, floor_z, door_top + 0.05),
        ("cafe_door_frame_b", dy1, dy1 + 0.05, floor_z, door_top + 0.05),
        ("cafe_door_frame_top", dy0, dy1, door_top, door_top + 0.05),
    ):
        world.add_geom(name=name, type=box, material="cafe_door",
                       size=[0.035, (yb - ya) / 2, (zb - za) / 2],
                       pos=[wall_x + 0.01, (ya + yb) / 2, (za + zb) / 2], **visual)
    # The door stands slightly ajar into the room, hinged at its +Y jamb.
    ajar = math.radians(18.0)
    width = float(door["width_m"])
    hinge = (wall_x + 0.02, dy1)
    centre = (hinge[0] + math.sin(ajar) * width / 2, hinge[1] - math.cos(ajar) * width / 2)
    door_quat = [math.cos(-ajar / 2), 0.0, 0.0, math.sin(-ajar / 2)]
    world.add_geom(name="cafe_door_panel", type=box, material="cafe_door", quat=door_quat,
                   size=[0.022, width / 2 - 0.01, (door_top - floor_z) / 2 - 0.01],
                   pos=[centre[0], centre[1], (floor_z + door_top) / 2], **visual)
    world.add_geom(name="cafe_door_window", type=box, material="cafe_glass", quat=door_quat,
                   size=[0.024, width / 2 - 0.12, 0.45],
                   pos=[centre[0], centre[1], door_top - 0.6], **visual)
    knob = (hinge[0] + math.sin(ajar) * (width - 0.08) + 0.03,
            hinge[1] - math.cos(ajar) * (width - 0.08))
    world.add_geom(name="cafe_door_knob", type=sph, size=[0.03, 0, 0], material="cafe_brass",
                   pos=[knob[0], knob[1], floor_z + 1.0], **visual)
    # The chime: a small brass bell on a bracket above the door, inside the room.
    chime = [float(v) for v in door["door_chime"]["position_m"]]
    world.add_geom(name="cafe_chime_bracket", type=cap, size=[0.006, 0, 0],
                   material="cafe_brass",
                   fromto=[wall_x + 0.02, chime[1], chime[2] + 0.07,
                           chime[0], chime[1], chime[2] + 0.07], **visual)
    world.add_geom(name="cafe_chime_bell", type=cyl, size=[0.03, 0.035, 0],
                   material="cafe_brass", pos=[chime[0], chime[1], chime[2] + 0.01], **visual)
    world.add_geom(name="cafe_chime_clapper", type=sph, size=[0.012, 0, 0],
                   material="cafe_brass", pos=[chime[0], chime[1], chime[2] - 0.035], **visual)
    world.add_site(name=str(door["door_chime"]["site"]), pos=chime, size=[0.02, 0, 0],
                   rgba=[1.0, 0.8, 0.2, 1.0], group=4)

    # Back-bar shelving with jars (visual), and pendant lamps over the bar.
    for k, z in enumerate((0.25, 0.65, 1.05)):
        world.add_geom(name=f"cafe_shelf_{k}", type=box, material="cafe_wood_panel",
                       size=[1.5, 0.12, 0.015], pos=[0.45, room_y[1] - 0.14, z], **visual)
        for j in range(9):
            jx = -0.9 + j * 0.33 + (0.05 if k % 2 else 0.0)
            world.add_geom(name=f"cafe_jar_{k}_{j}", type=cyl, material="cafe_jar",
                           size=[0.045, 0.07 + 0.015 * ((j + k) % 3), 0],
                           pos=[jx, room_y[1] - 0.14, z + 0.085 + 0.015 * ((j + k) % 3)],
                           **visual)
    lamp_positions = ((-0.05, 0.12), (0.9, 0.18))
    for k, (lx, ly) in enumerate(lamp_positions):
        world.add_geom(name=f"cafe_pendant_cord_{k}", type=cap, size=[0.003, 0, 0],
                       material="cafe_dark", fromto=[lx, ly, ceiling_z, lx, ly, 1.12], **visual)
        world.add_geom(name=f"cafe_pendant_shade_{k}", type=cyl, size=[0.09, 0.05, 0],
                       material="cafe_brass", pos=[lx, ly, 1.09], **visual)
        world.add_geom(name=f"cafe_pendant_bulb_{k}", type=sph, size=[0.035, 0, 0],
                       material="cafe_lamp", pos=[lx, ly, 1.03], **visual)
        world.add_light(name=f"cafe_pendant_{k}", pos=[lx, ly, 1.0], dir=[0.0, 0.0, -1.0],
                        type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=55.0, exponent=8.0,
                        diffuse=[0.95, 0.70, 0.42], specular=[0.25, 0.2, 0.15],
                        attenuation=[1.0, 0.0, 0.0], castshadow=True)
    world.add_light(name="cafe_key", pos=[0.2, -1.4, 1.6], dir=[0.25, 0.8, -0.9],
                    type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=40.0, exponent=4.0,
                    diffuse=[0.85, 0.66, 0.45], specular=[0.2, 0.18, 0.15], castshadow=True)
    world.add_light(name="cafe_window_fill", pos=[-2.0, -1.95, 0.9], dir=[1.0, 0.6, -0.3],
                    type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=70.0, exponent=2.0,
                    diffuse=[0.16, 0.17, 0.20], specular=[0.0, 0.0, 0.0], castshadow=False)

    # -- dressing on the bar top (static, collidable: the robot may not walk through it)
    kettle = dims["dressing"]["kettle"]
    kx, ky, _ = kettle["position_m"]
    kr, kh = float(kettle["body_radius_m"]), float(kettle["body_height_m"])
    world.add_geom(name="cafe_kettle_body", type=cyl, size=[kr, kh / 2, 0],
                   pos=[kx, ky, kh / 2], material="cafe_kettle", contype=1, conaffinity=1)
    world.add_geom(name="cafe_kettle_lid", type=cyl, size=[kr * 0.7, 0.008, 0],
                   pos=[kx, ky, kh + 0.008], material="cafe_kettle", **visual)
    world.add_geom(name="cafe_kettle_knob", type=sph, size=[0.012, 0, 0],
                   pos=[kx, ky, kh + 0.024], material="cafe_brass", **visual)
    spout = [[kx - kr + 0.005, ky, 0.025], [kx - kr - 0.035, ky, 0.07],
             [kx - kr - 0.05, ky, 0.14], [kx - kr - 0.11, ky, 0.165]]
    for k in range(3):
        world.add_geom(name=f"cafe_kettle_spout_{k}", type=cap, size=[0.007 - 0.001 * k, 0, 0],
                       fromto=[*spout[k], *spout[k + 1]], material="cafe_kettle", **visual)
    world.add_geom(name="cafe_kettle_handle", type=cap, size=[0.009, 0, 0],
                   fromto=[kx + kr + 0.02, ky, 0.02, kx + kr + 0.02, ky, kh - 0.005],
                   material="cafe_dark", **visual)
    for k, z in enumerate((0.02, kh - 0.01)):
        world.add_geom(name=f"cafe_kettle_handle_arm_{k}", type=cap, size=[0.007, 0, 0],
                       fromto=[kx + kr - 0.005, ky, z, kx + kr + 0.02, ky, z],
                       material="cafe_dark", **visual)
    cups = dims["dressing"]["cup_stack"]
    cx, cy, _ = cups["position_m"]
    cr, chh = float(cups["radius_m"]), float(cups["height_m"])
    world.add_geom(name="cafe_cup_stack", type=cyl, size=[cr, chh / 2, 0],
                   pos=[cx, cy, chh / 2], material="cafe_cup", contype=1, conaffinity=1)
    for k in range(3):
        world.add_geom(name=f"cafe_cup_rim_{k}", type=cyl, size=[cr + 0.003, 0.004, 0],
                       pos=[cx, cy, 0.05 * (k + 1) - 0.004], material="cafe_brass", **visual)

    # -- V60 cues on the accepted dripper (attached to its body, so they move with it)
    # Attached to a static body at the dripper's own pose, so the accepted v60
    # body stays exactly as authored.
    v60 = world.add_body(name="v60_cues", pos=station_v60_pos)
    # Coffee grounds: physical, and brewer geometry for every M4 gate ("v60_" token).
    v60.add_geom(name="v60_coffee_bed", type=cyl, size=[0.032, 0.002, 0],
                 pos=[0.0, 0.0, 0.084 - 0.0015 - 0.002], material="cafe_coffee",
                 contype=1, conaffinity=1, mass=0.0)

    shell_verts, shell_faces = _frustum_shell(*SHELL_Z, shell_inner_radius, SHELL_THICKNESS)
    spec.add_mesh(name="dripper_shell_mesh", uservert=[c for v in shell_verts for c in v],
                  userface=[i for f in shell_faces for i in f])
    v60.add_geom(name="dripper_shell", type=mujoco.mjtGeom.mjGEOM_MESH,
                 meshname="dripper_shell_mesh", material="cafe_ceramic", mass=0.0, **visual)
    v60.add_geom(name="dripper_flange", type=cyl, size=[0.049, 0.0015, 0],
                 pos=[0.0, 0.0, 0.0015], material="cafe_ceramic", mass=0.0, **visual)
    # Back-left (+Y side), away from the robot (about -117 deg), the pitcher's
    # resting spout (0 deg) and its pour approach; measured clear by
    # run_m4_transfer.py's visual-cue check.
    handle_angle = math.radians(120.0)
    ca, sa = math.cos(handle_angle), math.sin(handle_angle)
    loop = [(0.040, 0.074), (0.066, 0.071), (0.066, 0.032), (0.026, 0.028)]
    for k in range(3):
        (ra, za), (rb, zb) = loop[k], loop[k + 1]
        v60.add_geom(name=f"dripper_handle_{k}", type=cap, size=[0.0045, 0, 0],
                     fromto=[ra * ca, ra * sa, za, rb * ca, rb * sa, zb],
                     material="cafe_ceramic", mass=0.0, **visual)

    # -- named route sites and cameras
    route = dims["route"]
    start = route["start_slot"]
    world.add_site(name="start_slot", pos=[start["x_m"], start["y_m"], 0.001],
                   type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.001, 0],
                   rgba=[0.3, 0.7, 0.4, 0.6], group=4)
    pour = route["pour_start_pose"]
    world.add_site(name="pour_start", pos=[pour["x_m"], pour["y_m"], 0.001],
                   type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.001, 0],
                   rgba=[0.9, 0.5, 0.2, 0.6], group=4)
    dance = dims["dance_area"]
    world.add_site(name="dance_area", pos=[*dance["centre_m"], 0.0005],
                   type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[float(dance["radius_m"]), 0.0005, 0],
                   rgba=[0.3, 0.5, 0.9, 0.15], group=4)
    for key in ("customer_camera", "wide_camera"):
        cam = dims[key]
        world.add_camera(name=str(cam["camera"]), pos=list(cam["position_m"]),
                         quat=_look_at_quat(cam["position_m"], cam["look_at_m"]),
                         fovy=float(cam["fovy_deg"]))
    world.add_camera(name="cafe_overhead", pos=[0.45, 0.0, 2.4], quat=[1.0, 0.0, 0.0, 0.0],
                     fovy=50.0)

    assets = {WOOD_TEXTURE: _wood_png()}
    for name, content in assets.items():
        spec.assets[name] = content  # to_xml compiles, so the texture must resolve
    cafe = ET.fromstring(spec.to_xml())

    # The station half, verbatim from the accepted world file.
    accepted = ET.fromstring((coffee.root / world_file).read_text("utf-8"))
    accepted.set("model", "jumper_cafe_bar")
    worldbody = accepted.find("worldbody")
    assert worldbody is not None
    dropped = set(DROPPED_GEOMS) | set(DROPPED_SITES) | set(DROPPED_LIGHTS)
    moved = set(STATION_BODIES) | set(STATION_CAMERAS) | set(STATION_GEOMS)
    frame = ET.Element("frame", {"name": "brew_station",
                                 "pos": " ".join(repr(v) for v in offset)})
    for child in list(worldbody):
        name = child.get("name")
        if name in dropped:
            worldbody.remove(child)
        elif name in moved:
            worldbody.remove(child)
            frame.append(child)
    for child in frame:
        if child.get("name") in STATION_GEOMS:
            # Collision untouched; only its look is replaced by the wooden top.
            child.attrib.pop("material", None)
            child.set("rgba", " ".join(str(v) for v in INVISIBLE["rgba"]))
            child.set("group", str(INVISIBLE["group"]))
    missing = moved - {child.get("name") for child in frame}
    if missing:
        raise RuntimeError(f"the accepted world no longer has {sorted(missing)}")
    worldbody.insert(0, frame)
    asset = accepted.find("asset")
    assert asset is not None
    have = {child.get("name") for child in asset}
    cafe_asset, cafe_world = cafe.find("asset"), cafe.find("worldbody")
    assert cafe_asset is not None and cafe_world is not None
    for child in list(cafe_asset):
        if child.get("name") not in have:
            asset.append(child)
    for child in list(cafe_world):
        if child.get("name") in exact:
            for key, values in exact[child.get("name")].items():
                child.set(key, " ".join(repr(float(v)) for v in values))
        worldbody.append(child)
    ET.indent(accepted, space="  ")
    xml = ET.tostring(accepted, encoding="unicode") + "\n"
    return xml, assets


def write_package(root: Path, dims: dict[str, Any] | None = None) -> Path:
    """Write the generated cafe as a ``kk-scene-package/1`` directory under `root`."""
    dims = dims or dimensions()
    xml, assets = build_world(dims)
    root.mkdir(parents=True, exist_ok=True)
    files = {"scene.xml": xml.encode("utf-8"), **assets}
    for name, content in files.items():
        (root / name).write_bytes(content)
    coffee = json.loads((COFFEE_PACKAGE / "scene-package.json").read_text("utf-8"))
    pitcher = next(p for p in coffee["props"] if p["name"] == "pitcher")
    offset = dims["brew_station"]["authored_offset_m"]
    manifest = {
        "schema": "kk-scene-package/1",
        "id": "jumper_cafe_bar",
        "description": "A warm cafe bar with the accepted V60 brew station, kettle and an "
                       "off-camera entrance door; simulation only.",
        "tags": ["barista", "coffee", "cafe", "presentation"],
        "use": "watching",
        "requiredCapabilities": ["rigid"],
        "world": {
            "file": "scene.xml",
            "assetDir": ".",
            "terrainType": "none",
            "ground": {"geom": GROUND_GEOM},
            "attach": {"prefix": "bar_"},
        },
        "props": [{
            **pitcher,
            "position": [round(pitcher["position"][i] + offset[i], 6) for i in range(3)],
        }],
        "spawn": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        "rules": {
            "simulationOnly": True,
            "waterModel": "rigid mass proxy in the pitcher; the stream is a visual approximation",
        },
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(files.items())
        },
    }
    (root / "scene-package.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    return root


def package_dir() -> Path:
    """The generated package for this process, built once into a temporary directory."""
    if "dir" not in _BUILT:
        holder = tempfile.TemporaryDirectory(prefix="cafe_bar-")
        _BUILT["holder"] = holder  # kept alive: the env reads the world at build time
        _BUILT["dir"] = write_package(Path(holder.name) / "cafe_bar.scene")
    return _BUILT["dir"]


def scene() -> Scene:
    """Build the cafe package and load it as the registered scene."""
    from mjlab.utils import spec_config as sc

    from .package import load_package

    loaded = load_package(package_dir())
    # Warm, dark cafe: a low ambient floor so the pendant and key lights carry the
    # picture, plus one ordinary dim light for the registry's non-black invariant.
    return replace(
        loaded,
        headlight=Headlight(ambient=(0.10, 0.08, 0.065), diffuse=(0.16, 0.13, 0.10)),
        lights=(
            sc.LightCfg(
                name="cafe_ambient_fill",
                type="directional",
                pos=(-0.6, -1.0, 1.6),
                dir=(0.3, 0.5, -1.0),
                diffuse=(0.10, 0.085, 0.07),
                castshadow=False,
            ),
        ),
    )
