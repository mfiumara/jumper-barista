#!/usr/bin/env python3
"""Render the Act 2 barista outfit CONCEPTS on the real Jumper model (simulation only).

This is the A2a design-gate tool. It does not build the outfit: every accessory here
is a rough primitive drawn straight into the renderer's scene (``mjvGeom`` decor), and
every garment is a render-time recolour of an existing visual geom. Nothing is added to
the compiled model, so no collision, mass or inertia exists for any of it, and the
images are concept mock-ups, not renders of a generated skin.

Poses come from the accepted M4 replay's ``render_states.npz`` (written by
``barista/verification/run_m4.py --save-render-states``), checked against its
``summary.json`` exactly as ``barista/film/make_film.py`` checks it. Each concept is
rendered at the pour pose (pitcher in the claw) from a front three-quarter and a left
side camera, labelled ``CONCEPT A/B/C``, and tiled into one contact sheet.

The script also measures, per concept, the clearance between the accessories (which
would ride on ``upper_shell_link``) and everything that moves relative to the shell:
the two front claws, the four legs and the pitcher. It sweeps the whole M4 replay
(approach, grasp, lift, carry, spiral pour, set-down), the five dance clips' reference
motions and the flat jump reference (crouch, take-off, tuck, landing). Clearances are
vertex-sampled (meshes voxelised at ``VOXEL_M``, primitives surface-sampled), so they are
an estimate at that resolution, not an exact distance.

    python barista/outfit/concepts.py --replay <run_m4 --out dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "barista" / "film"))

WIDTH, HEIGHT = 960, 720
POUR_TIME_S = 30.0  # mid-spiral of the accepted M4 replay (spiral runs 23.5-41.5 s)
SHELL_BODY = "robot/upper_shell_link"
# Bodies rigidly fixed to the shell: an accessory on the shell never moves relative to them.
SHELL_RIGID = ("base_link", "upper_shell_link", "display_module_link", "tof_sensor_link",
               "camera_link")
# The eye panel A3 owns (barista/character/eyes.py), in the shell frame: nothing may sit
# in front of it.
FACE_BOX = ((0.060, 0.140), (-0.058, 0.058), (-0.004, 0.038))
CLEARANCE_MIN_M = 0.010
VOXEL_M = 0.0025
SAMPLE_STEP_M = 0.0025
SEARCH_M = 0.020
DECAL_PROUD_M = 0.0008
DECAL_HALF_T = 0.0008
LAYERS = ("proud", "decal")
GROUPS = ("claws", "legs", "pitcher")
BARE_FACE = ("robot", "bare_face")
# Static poses of the accepted M4 replay: claw closed on the handle (end of "close"),
# and mid-spiral pour.
POSES = {"grasp": 4.0, "pour": POUR_TIME_S}
POSE_SEARCH_M = 0.100
SAMPLE_HZ = 25.0
DANCE_CLIPS = (
    "tasks/jumper/dance/media/demo.npz",
    "tasks/jumper/dance_brazilian/media/brazilian.npz",
    "tasks/jumper/dance_dream_wings/media/dream_wings.npz",
    "tasks/jumper/dance_maze/media/maze.npz",
    "tasks/jumper/dance_waist/media/waist.npz",
)
JUMP_CLIP = "tasks/jumper/jump/ref/high_jump_flat.npz"
FOOTER = ("concept mock-up: rough primitives on the real Jumper model, accepted M4 pour pose"
          "  |  MuJoCo simulation  |  not a final skin")


# ── Concept definitions (shell frame: +x forward/face, +y left, +z up, metres) ──────────


@dataclass(frozen=True)
class Part:
    """One primitive. ``kind`` is a MuJoCo geom type name, ``seg`` for a capsule
    between two points (``a``, ``b``) of radius ``size[0]``, or ``decal`` for a flush
    stroke on the front face: a flat ellipsoid from (y, z) of ``a`` to (y, z) of ``b``,
    half-width ``size[0]``, laid on the face surface (``a``/``b`` x is ignored) and
    standing at most ``DECAL_PROUD_M`` proud of it."""

    name: str
    kind: str
    rgba: tuple[float, float, float, float]
    size: tuple[float, ...] = (0.0, 0.0, 0.0)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    a: tuple[float, float, float] | None = None
    b: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Concept:
    key: str
    name: str
    tagline: str
    description: str
    palette: dict[str, tuple[float, float, float]]
    recolour: dict[str, tuple[float, float, float, float]]  # visual geom name -> rgba
    parts: tuple[Part, ...] = field(default_factory=tuple)


def _rgba(rgb, alpha=1.0):
    return (rgb[0], rgb[1], rgb[2], alpha)


def _mirror(parts):
    """Each part plus its mirror across the y=0 plane (for symmetric pieces)."""
    out = []
    for p in parts:
        out.append(p)
        flip = (lambda v: None if v is None else (v[0], -v[1], v[2]))
        out.append(Part(p.name + "_r", p.kind, p.rgba, p.size, flip(p.pos),
                        (-p.euler_deg[0], p.euler_deg[1], -p.euler_deg[2]),
                        flip(p.a), flip(p.b)))
    return out


def _ring(name, centre, radius, normal_pitch_deg, wire, rgba, n=16):
    """A wire ring (glasses rim) as ``n`` capsule segments; its plane faces +x, tipped
    up by ``normal_pitch_deg``."""
    c = np.asarray(centre)
    t = math.radians(normal_pitch_deg)
    up = np.array((math.sin(t), 0.0, math.cos(t)))  # in-plane "up", tilted back
    side = np.array((0.0, 1.0, 0.0))
    pts = [c + radius * (math.cos(2 * math.pi * k / n) * side
                         + math.sin(2 * math.pi * k / n) * up) for k in range(n + 1)]
    return [Part(f"{name}_{k}", "seg", rgba, (wire,), a=tuple(pts[k]), b=tuple(pts[k + 1]))
            for k in range(n)]


# Palettes
TWEED = (0.36, 0.27, 0.20)
CHARCOAL = (0.17, 0.17, 0.19)
CREAM = (0.93, 0.89, 0.80)
BRASS = (0.80, 0.64, 0.30)
ESPRESSO = (0.07, 0.05, 0.04)
BORDEAUX = (0.42, 0.07, 0.12)
FOREST = (0.12, 0.33, 0.22)
GOLD_WIRE = (0.78, 0.62, 0.25)
CHESTNUT = (0.30, 0.17, 0.09)
STRAW = (0.86, 0.74, 0.50)
VERDE = (0.0, 0.55, 0.27)
BIANCO = (0.96, 0.96, 0.94)
ROSSO = (0.80, 0.11, 0.17)
STOCK_RED = (0.85, 0.08, 0.05)

CONCEPTS = (
    Concept(
        key="A", name="Il Classico Milanese",
        tagline="tweed flat cap, handlebar moustache, charcoal waistcoat",
        description=(
            "The neighbourhood barista who has pulled shots since 1987. A low tweed "
            "coppola (flat cap) sits on the back of the dome with its short brim stopping "
            "well behind the eye panel; a black handlebar moustache with curled tips is "
            "a flush decal under the sensor 'nose'. The red upper shell becomes a charcoal waistcoat with "
            "two rows of brass buttons along its rim, and the lower body reads as a cream "
            "shirt. Legs and claws stay stock red, so it is still unmistakably Jumper."),
        palette={"tweed cap": TWEED, "waistcoat charcoal": CHARCOAL, "shirt cream": CREAM,
                 "brass buttons": BRASS, "moustache espresso-black": ESPRESSO},
        recolour={"robot/upper_shell_link_visual": _rgba(CHARCOAL),
                  "robot/base_link_visual": _rgba(CREAM)},
        parts=(
            Part("cap_crown", "ellipsoid", _rgba(TWEED), (0.052, 0.047, 0.015),
                 (-0.022, 0.0, 0.040), (0.0, 6.0, 0.0)),
            Part("cap_brim", "ellipsoid", _rgba(TWEED), (0.024, 0.036, 0.0036),
                 (0.032, 0.0, 0.042), (0.0, 12.0, 0.0)),
            Part("cap_button", "sphere", _rgba(TWEED), (0.005,), (-0.022, 0.0, 0.056)),
            Part("tache_bridge", "decal", _rgba(ESPRESSO), (0.0026,),
                 a=(0.0, -0.004, -0.0255), b=(0.0, 0.004, -0.0255)),
            *_mirror([
                Part("tache_l1", "decal", _rgba(ESPRESSO), (0.0026,),
                     a=(0.0, 0.004, -0.0255), b=(0.0, 0.020, -0.0268)),
                Part("tache_l2", "decal", _rgba(ESPRESSO), (0.0019,),
                     a=(0.0, 0.020, -0.0268), b=(0.0, 0.032, -0.0250)),
                Part("tache_l3", "decal", _rgba(ESPRESSO), (0.0013,),
                     a=(0.0, 0.032, -0.0250), b=(0.0, 0.036, -0.0205)),
                Part("button_l1", "sphere", _rgba(BRASS), (0.0032,), (0.040, 0.087, 0.018)),
                Part("button_l2", "sphere", _rgba(BRASS), (0.0032,), (0.010, 0.089, 0.018)),
                Part("button_l3", "sphere", _rgba(BRASS), (0.0032,), (-0.020, 0.089, 0.018)),
            ]),
        ),
    ),
    Concept(
        key="B", name="Boheme del Caffe",
        tagline="tilted bordeaux beret, round gold glasses, green waist apron",
        description=(
            "The third-wave hipster who talks about the farm while he pours. A soft "
            "bordeaux beret is pulled to one side of the dome, with round gold-wire "
            "glasses pushed up onto its front edge, above the eye panel rather than over "
            "it. A full chestnut chevron moustache is a flush decal under the nose. The lower body "
            "becomes a forest-green waist apron tied at the back with a cream bow; the "
            "red shell, legs and claws stay stock."),
        palette={"beret bordeaux": BORDEAUX, "glasses gold wire": GOLD_WIRE,
                 "moustache chestnut": CHESTNUT, "apron forest green": FOREST,
                 "apron ties cream": CREAM, "shell stock red": STOCK_RED},
        recolour={"robot/base_link_visual": _rgba(FOREST)},
        parts=(
            Part("beret", "ellipsoid", _rgba(BORDEAUX), (0.058, 0.055, 0.014),
                 (-0.026, 0.012, 0.039), (-8.0, 4.0, 0.0)),
            Part("beret_tip", "seg", _rgba(BORDEAUX), (0.0022,),
                 a=(-0.026, 0.014, 0.052), b=(-0.028, 0.015, 0.060)),
            *_ring("glasses_l", (0.040, 0.020, 0.052), 0.014, 55.0, 0.0011, _rgba(GOLD_WIRE)),
            *_ring("glasses_r", (0.040, -0.020, 0.052), 0.014, 55.0, 0.0011, _rgba(GOLD_WIRE)),
            Part("glasses_bridge", "seg", _rgba(GOLD_WIRE), (0.0011,),
                 a=(0.043, 0.006, 0.056), b=(0.043, -0.006, 0.056)),
            *_mirror([
                Part("glasses_temple_l", "seg", _rgba(GOLD_WIRE), (0.0011,),
                     a=(0.040, 0.034, 0.052), b=(0.008, 0.046, 0.048)),
                Part("tache_l", "decal", _rgba(CHESTNUT), (0.0032,),
                     a=(0.0, 0.0, -0.0232), b=(0.0, 0.026, -0.0252)),
                Part("apron_bow_loop_l", "ellipsoid", _rgba(CREAM), (0.0025, 0.009, 0.005),
                     (-0.104, 0.009, -0.012), (0.0, 0.0, 0.0)),
                Part("apron_tail_l", "seg", _rgba(CREAM), (0.0018,),
                     a=(-0.104, 0.003, -0.014), b=(-0.101, 0.010, -0.032)),
            ]),
            Part("apron_knot", "sphere", _rgba(CREAM), (0.0040,), (-0.105, 0.0, -0.012)),
        ),
    ),
    Concept(
        key="C", name="Dapper Tricolore",
        tagline="straw trilby with tricolore band, pencil moustache, bar towel",
        description=(
            "The dapper espresso-bar host from a 1960s Roman film. A narrow-brim straw "
            "trilby with a thin green-white-red band gives the tallest silhouette; a neat "
            "pencil moustache decal keeps the face light. A cream bar towel with a red stripe is "
            "folded over the back of the shell, and the lower body becomes a crisp white "
            "barista jacket. The red shell, legs and claws stay stock, so the tricolore "
            "comes from the robot itself."),
        palette={"straw": STRAW, "band verde": VERDE, "band bianco": BIANCO,
                 "band rosso": ROSSO, "towel cream": CREAM, "jacket white": BIANCO,
                 "moustache espresso-black": ESPRESSO, "shell stock red": STOCK_RED},
        recolour={"robot/base_link_visual": _rgba(BIANCO)},
        parts=(
            Part("hat_brim", "cylinder", _rgba(STRAW), (0.056, 0.0020),
                 (-0.030, 0.0, 0.041), (0.0, 8.0, 0.0)),
            Part("hat_crown", "cylinder", _rgba(STRAW), (0.034, 0.015),
                 (-0.032, 0.0, 0.060), (0.0, 8.0, 0.0)),
            Part("hat_crown_top", "ellipsoid", _rgba(STRAW), (0.034, 0.032, 0.006),
                 (-0.034, 0.0, 0.075), (0.0, 8.0, 0.0)),
            Part("band_verde", "cylinder", _rgba(VERDE), (0.0345, 0.0018),
                 (-0.0310, 0.0, 0.0480), (0.0, 8.0, 0.0)),
            Part("band_bianco", "cylinder", _rgba(BIANCO), (0.0345, 0.0018),
                 (-0.0315, 0.0, 0.0516), (0.0, 8.0, 0.0)),
            Part("band_rosso", "cylinder", _rgba(ROSSO), (0.0345, 0.0018),
                 (-0.0320, 0.0, 0.0552), (0.0, 8.0, 0.0)),
            *_mirror([
                Part("tache_l", "decal", _rgba(ESPRESSO), (0.0015,),
                     a=(0.0, 0.003, -0.0232), b=(0.0, 0.024, -0.0240)),
            ]),
            Part("towel_top", "box", _rgba(CREAM), (0.016, 0.026, 0.0025),
                 (-0.084, -0.040, 0.020), (0.0, -20.0, 0.0)),
            Part("towel_back", "box", _rgba(CREAM), (0.0025, 0.026, 0.014),
                 (-0.104, -0.040, 0.000), (0.0, -6.0, 0.0)),
            Part("towel_stripe", "box", _rgba(ROSSO), (0.0030, 0.026, 0.0022),
                 (-0.104, -0.040, -0.004), (0.0, -6.0, 0.0)),
        ),
    ),
)


# ── Geometry helpers ──────────────────────────────────────────────────────────


def _euler_mat(deg) -> np.ndarray:
    """Intrinsic x-y-z rotation (MuJoCo's default ``eulerseq="xyz"``)."""
    rx, ry, rz = (math.radians(v) for v in deg)
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry),
                              math.cos(rz), math.sin(rz))
    mx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    my = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    mz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return mx @ my @ mz


def _seg_frame(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    z = axis / length
    helper = np.array((1.0, 0.0, 0.0)) if abs(z[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return (a + b) / 2, np.stack((x, y, z), axis=1), length / 2


def _face_x(y: float) -> float:
    """The front face around the sensors' lower edge (z -0.029..-0.020 in the shell
    frame), measured from the base_link visual mesh: x = 0.1010 m on the centre line,
    curving back by about 2.2 mm at y = +/-40 mm, i.e. x = 0.1010 - 1.4 y^2."""
    return 0.1010 - 1.4 * y * y


def _part_pose(p: Part):
    """(type name, size[3], pos[3], mat[3x3]) in the shell frame."""
    if p.kind == "decal":
        a = np.array((_face_x(p.a[1]), p.a[1], p.a[2]))
        b = np.array((_face_x(p.b[1]), p.b[1], p.b[2]))
        along = b - a
        along[0] = 0.0
        half = float(np.linalg.norm(along)) / 2
        u = along / (2 * half)
        normal = np.array((1.0, 0.0, 0.0))
        mat = np.stack((normal, u, np.cross(normal, u)), axis=1)
        pos = (a + b) / 2 + normal * (DECAL_PROUD_M - DECAL_HALF_T)
        return "ellipsoid", np.array((DECAL_HALF_T, half + p.size[0], p.size[0])), pos, mat
    if p.kind == "seg":
        pos, mat, half = _seg_frame(p.a, p.b)
        return "capsule", np.array((p.size[0], half, 0.0)), pos, mat
    size = np.zeros(3)
    size[: len(p.size)] = p.size
    return p.kind, size, np.asarray(p.pos, float), _euler_mat(p.euler_deg)


def _fib_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = math.pi * (1 + 5 ** 0.5) * k
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                     np.cos(phi)), axis=1)


def _surface(kind: str, size: np.ndarray, step: float = SAMPLE_STEP_M) -> np.ndarray:
    """Deterministic surface samples of a primitive in its own frame."""
    if kind in ("sphere", "ellipsoid"):
        radii = np.full(3, size[0]) if kind == "sphere" else size[:3]
        area = 4 * math.pi * max(radii) ** 2
        return _fib_sphere(max(64, int(area / step**2))) * radii
    if kind in ("capsule", "cylinder"):
        r, h = size[0], size[1]
        nt = max(12, int(2 * math.pi * r / step))
        nz = max(2, int(2 * h / step) + 1)
        t = 2 * math.pi * np.arange(nt) / nt
        zs = np.linspace(-h, h, nz)
        side = np.stack([np.repeat(r * np.cos(t)[None], nz, 0).ravel(),
                         np.repeat(r * np.sin(t)[None], nz, 0).ravel(),
                         np.repeat(zs[:, None], nt, 1).ravel()], axis=1)
        if kind == "capsule":
            cap = _fib_sphere(max(32, int(4 * math.pi * r * r / step**2))) * r
            ends = np.concatenate((cap[cap[:, 2] >= 0] + (0, 0, h),
                                   cap[cap[:, 2] < 0] - (0, 0, h)))
        else:
            rr = np.linspace(0, r, max(2, int(r / step) + 1))
            disc = np.array([(ri * math.cos(a), ri * math.sin(a))
                             for ri in rr for a in t]).reshape(-1, 2)
            ends = np.concatenate((np.c_[disc, np.full(len(disc), h)],
                                   np.c_[disc, np.full(len(disc), -h)]))
        return np.concatenate((side, ends))
    if kind == "box":
        pts = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            g = [np.linspace(-size[i], size[i], max(2, int(2 * size[i] / step) + 1))
                 for i in others]
            u, v = np.meshgrid(*g, indexing="ij")
            for sign in (-1, 1):
                face = np.zeros((u.size, 3))
                face[:, others[0]] = u.ravel()
                face[:, others[1]] = v.ravel()
                face[:, axis] = sign * size[axis]
                pts.append(face)
        return np.concatenate(pts)
    raise ValueError(f"no surface sampler for {kind}")


def _concept_points(concept: Concept, decal: bool) -> tuple[np.ndarray, list[str]]:
    """Surface samples of the concept's proud parts (``decal=False``) or its flush
    decals (``decal=True``), with the owning part name per sample."""
    pts, owner = [], []
    for p in concept.parts:
        if (p.kind == "decal") != decal:
            continue
        kind, size, pos, mat = _part_pose(p)
        s = _surface(kind, size) @ mat.T + pos
        pts.append(s)
        owner += [p.name] * len(s)
    return np.concatenate(pts), owner


def _voxel(points: np.ndarray, step: float) -> np.ndarray:
    keys = np.floor(points / step).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


# ── Model, poses and clips ────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _targets(model, mujoco):
    """Moving geometry, grouped, as geom-local point clouds: claws, legs, pitcher."""
    groups: dict[str, list[tuple[int, np.ndarray, float]]] = {"claws": [], "legs": [], "pitcher": []}
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if body.startswith("robot/"):
            short = body.split("/", 1)[1]
            if short in SHELL_RIGID or short == "floating_base":
                continue
            group = "claws" if short.startswith(("LF_", "RF_")) else "legs"
        elif body == "bar_pitcher":
            group = "pitcher"
        else:
            continue
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            start = model.mesh_vertadr[m]
            pts = np.asarray(model.mesh_vert[start:start + model.mesh_vertnum[m]], float)
        else:
            kind = mujoco.mjtGeom(gtype).name.removeprefix("mjGEOM_").lower()
            if kind not in ("sphere", "ellipsoid", "capsule", "cylinder", "box"):
                continue
            pts = _surface(kind, np.asarray(model.geom_size[g], float))
        pts = _voxel(pts, VOXEL_M)
        groups[group].append((g, pts, float(np.linalg.norm(pts, axis=1).max())))
    return groups


def _robot_qadr(model, mujoco, names):
    return np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                         f"robot/{n}")] for n in names])


def _clips(model, mujoco, qpos_m4, render_fps):
    """Yield (clip name, kind, list of qpos rows) for every swept motion."""
    from tasks.jumper.common.constants import HOME
    from tasks.jumper.common.dance.motion import load_source

    stride = max(1, round(render_fps / SAMPLE_HZ))
    yield ("m4_replay (approach, grasp, lift, carry, pour, set-down)", "replay",
           qpos_m4[::stride])

    home = list(HOME)
    adr = _robot_qadr(model, mujoco, home)
    base = qpos_m4[0].copy()

    def rows(joint_pos, dt):
        stride = max(1, round(1.0 / (SAMPLE_HZ * dt)))
        out = np.repeat(base[None], len(joint_pos[::stride]), 0)
        out[:, adr] = joint_pos[::stride]
        return out

    for rel in DANCE_CLIPS:
        clip = load_source(REPO / rel)
        yield f"dance reference {Path(rel).stem}", "kinematic", rows(clip.joint_pos, clip.dt)
    with np.load(REPO / JUMP_CLIP) as z:
        meta = json.loads(str(z["meta"]))
        order = meta["joint_order"]
        q = np.asarray(z["q"], float)[:, [order.index(n) for n in home]]
        dt = float(np.median(np.diff(z["t"])))
    yield "flat jump reference high_jump_flat (crouch, take-off, tuck, landing)", \
        "kinematic", rows(q, dt)


def _bare_face(model, mujoco, qpos_row):
    """The robot's own front face where the moustache decals sit (base_link visual
    vertices in the shell frame): the reference the decal minima are read against."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_row
    mujoco.mj_kinematics(model, data)
    shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHELL_BODY)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot/base_link_visual")
    m = model.geom_dataid[g]
    verts = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    local = (world - data.xpos[shell]) @ data.xmat[shell].reshape(3, 3)
    zone = local[(local[:, 0] > 0.090) & (np.abs(local[:, 1]) < 0.040)
                 & (local[:, 2] > -0.030) & (local[:, 2] < -0.020)]
    return zone, ["bare_face"] * len(zone)


def _frame_minima(model, mujoco, data, row, groups, layers, trees, search, pitcher):
    """{(layer key, group): (distance, part)} for one pose, distances < ``search``."""
    shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHELL_BODY)
    data.qpos[:] = row
    mujoco.mj_kinematics(model, data)
    rot = data.xmat[shell].reshape(3, 3)
    org = data.xpos[shell]
    found = {}
    for group, items in groups.items():
        if group == "pitcher" and not pitcher:
            continue
        centres = (data.geom_xpos[[g for g, _, _ in items]] - org) @ rot
        for key, (pts, _owner) in layers.items():
            lo, hi = pts.min(0) - search, pts.max(0) + search
            # Skip whole geoms whose bounding sphere misses the search box.
            gap = np.linalg.norm(np.maximum(0.0, np.maximum(lo - centres, centres - hi)),
                                 axis=1)
            keep = [k for k, (_, _, radius) in enumerate(items) if gap[k] <= radius]
            if not keep:
                continue
            local = np.concatenate([
                (items[k][1] @ data.geom_xmat[items[k][0]].reshape(3, 3).T
                 + data.geom_xpos[items[k][0]] - org) @ rot for k in keep])
            near = local[np.all((local > lo) & (local < hi), axis=1)]
            if not len(near):
                continue
            dist, idx = trees[key].query(near, k=1, distance_upper_bound=search, workers=-1)
            j = int(np.argmin(dist))
            if not math.isinf(dist[j]):
                found[(key, group)] = (float(dist[j]), layers[key][1][int(idx[j])])
    return found


def _clearance(model, mujoco, qpos_m4, render_fps):
    """Per concept, per layer (proud parts, flush decals), per target group: the
    minimum distance over every swept clip, plus the static grasp and pour poses.

    Only distances below ``SEARCH_M`` are resolved; a group that never comes that close
    is reported as ``min_m: null`` (clear by more than ``SEARCH_M``).
    """
    from scipy.spatial import cKDTree

    layers = {(c.key, layer): _concept_points(c, layer == "decal")
              for c in CONCEPTS for layer in LAYERS}
    layers[BARE_FACE] = _bare_face(model, mujoco, qpos_m4[0])
    trees = {k: cKDTree(v[0]) for k, v in layers.items()}
    groups = _targets(model, mujoco)
    data = mujoco.MjData(model)
    args = (groups, layers, trees)

    poses = {}
    for pose, time_s in POSES.items():
        index = round(time_s * render_fps)
        found = _frame_minima(model, mujoco, data, qpos_m4[index], *args, POSE_SEARCH_M,
                              pitcher=True)
        poses[pose] = {"time_s": time_s, "render_index": index, "min_m": {
            f"{k[0]}/{k[1]}/{g}": (round(found[(k, g)][0], 4) if (k, g) in found else None)
            for k in layers for g in groups}}

    best = {(k, g): {"min_m": None} for k in layers for g in groups}
    clips = []
    for clip_name, kind, rows in _clips(model, mujoco, qpos_m4, render_fps):
        clip_min = {(k, g): math.inf for k in layers for g in groups}
        for i, row in enumerate(rows):
            found = _frame_minima(model, mujoco, data, row, *args, SEARCH_M,
                                  pitcher=kind == "replay")
            for key, (d, part) in found.items():
                if d >= clip_min[key]:
                    continue
                clip_min[key] = d
                prior = best[key]["min_m"]
                if prior is None or d < prior:
                    best[key] = {"min_m": d, "clip": clip_name, "sample": i,
                                 "time_s": round(i / SAMPLE_HZ, 3), "part": part}
        clips.append({"clip": clip_name, "kind": kind, "samples": len(rows),
                      "min_m": {f"{c}/{layer}/{g}": (None if math.isinf(v) else round(v, 4))
                                for ((c, layer), g), v in clip_min.items()}})
        print(f"[concepts] swept {clip_name}: {len(rows)} samples", flush=True)
    face = {}
    for c in CONCEPTS:
        pts, owner = layers[(c.key, "proud")]
        inside = np.all([(pts[:, i] > FACE_BOX[i][0]) & (pts[:, i] < FACE_BOX[i][1])
                         for i in range(3)], axis=0)
        face[c.key] = sorted({owner[i] for i in np.flatnonzero(inside)})
    return best, clips, face, poses


# ── Rendering ─────────────────────────────────────────────────────────────────


def _add_part(mujoco, scene, p: Part, rot, org):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("renderer scene is full")
    kind, size, pos, mat = _part_pose(p)
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(geom, getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}").value,
                        size, rot @ pos + org, (rot @ mat).reshape(-1),
                        np.asarray(p.rgba, np.float32))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def _render(mujoco, renderer, model, data, concept, camera):
    renderer.update_scene(data, camera=camera)
    scene = renderer.scene
    recolour = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): rgba
                for n, rgba in concept.recolour.items()}
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in recolour:
            g.rgba[:] = recolour[g.objid]
    shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHELL_BODY)
    rot, org = data.xmat[shell].reshape(3, 3), data.xpos[shell]
    for p in concept.parts:
        _add_part(mujoco, scene, p, rot, org)
    return renderer.render()


def _font(size):
    from PIL import ImageFont

    # Pillow's bundled Aileron Regular (CC0; see barista/SOURCES.md).
    return ImageFont.load_default(size=size)


def _label(frame, concept, view):
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    title = f"CONCEPT {concept.key}  -  {concept.name}"
    draw.rectangle((0, 0, WIDTH, 92), fill=(0, 0, 0, 170))
    draw.text((24, 14), title, font=_font(38), fill=(255, 236, 200, 255))
    draw.text((24, 60), f"{concept.tagline}  |  {view}", font=_font(20),
              fill=(255, 255, 255, 230))
    draw.rectangle((0, HEIGHT - 38, WIDTH, HEIGHT), fill=(0, 0, 0, 170))
    draw.text((WIDTH // 2, HEIGHT - 19), FOOTER, font=_font(16),
              fill=(255, 255, 255, 230), anchor="mm")
    return Image.alpha_composite(image, layer).convert("RGB")


def _contact_sheet(images, out):
    from PIL import Image, ImageDraw

    cell_w, cell_h = WIDTH // 2, HEIGHT // 2
    head = 70
    sheet = Image.new("RGB", (cell_w * len(CONCEPTS), head + cell_h * 2), (18, 14, 12))
    draw = ImageDraw.Draw(sheet)
    draw.text((sheet.width // 2, 24), "Jumper Barista: outfit CONCEPTS for Mattia's pick",
              font=_font(30), fill=(255, 236, 200), anchor="mm")
    draw.text((sheet.width // 2, 54), "rough primitive mock-ups on the real model at the "
              "M4 pour pose; top: front three-quarter, bottom: left side; simulation only",
              font=_font(16), fill=(220, 220, 220), anchor="mm")
    for col, c in enumerate(CONCEPTS):
        for row, view in enumerate(("threequarter", "side")):
            tile = images[(c.key, view)].resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(tile, (col * cell_w, head + row * cell_h))
    sheet.save(out)


def _rounded(entry: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=Path, required=True,
                        help="run_m4.py --out directory holding render_states.npz and "
                             "summary.json")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--skip-clearance", action="store_true",
                        help="render only (the clearance sweep takes a few minutes)")
    return parser


def main() -> int:
    args = _parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    import mujoco
    from make_film import Camera, _load, _model

    replay = args.replay.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    qpos, _streams, render_fps, summary = _load(replay)
    model = _model()
    if model.nq != qpos.shape[1]:
        raise SystemExit(f"[concepts] model nq={model.nq} but the replay logged {qpos.shape[1]}")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    model.vis.quality.shadowsize = 4096
    model.vis.headlight.ambient[:] = (0.35, 0.33, 0.30)
    model.vis.headlight.diffuse[:] = (0.55, 0.52, 0.48)

    data = mujoco.MjData(model)
    pour_index = round(POUR_TIME_S * render_fps)
    data.qpos[:] = qpos[pour_index]
    mujoco.mj_forward(model, data)
    shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHELL_BODY)
    centre = data.xpos[shell] + data.xmat[shell].reshape(3, 3) @ np.array((0.02, 0.0, 0.02))
    views = {
        "threequarter": ("front three-quarter, pour pose",
                         Camera(mujoco, centre, 0.46, 150.0, -14.0)),
        "side": ("left side, pour pose (pitcher in the claw)",
                 Camera(mujoco, centre, 0.50, 268.0, -6.0)),
    }
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH, max_geom=20000)
    images, files = {}, {}
    try:
        for c in CONCEPTS:
            for view, (caption, cam) in views.items():
                image = _label(_render(mujoco, renderer, model, data, c, cam.cam), c, caption)
                path = out / f"concept_{c.key}_{view}.png"
                image.save(path)
                images[(c.key, view)] = image
                files[f"{c.key}_{view}"] = str(path)
    finally:
        renderer.close()
    sheet = out / "concepts_contact_sheet.png"
    _contact_sheet(images, sheet)
    files["contact_sheet"] = str(sheet)

    manifest = {
        "simulation_only": True,
        "concept_mockups_not_final_geometry": True,
        "source_replay": str(replay),
        "source_commit_sha": summary["commit_sha"],
        "source_qpos_sha256": summary["determinism"]["qpos_sha256"],
        "pour_pose": {"time_s": pour_index / render_fps, "render_index": pour_index},
        "concepts": {c.key: {"name": c.name, "tagline": c.tagline,
                             "palette": {k: [round(v, 3) for v in rgb]
                                         for k, rgb in c.palette.items()},
                             "recolour": sorted(c.recolour),
                             "parts": [p.name for p in c.parts]} for c in CONCEPTS},
        "files": files,
        "png_sha256": {k: _sha256_file(Path(v)) for k, v in files.items()},
    }
    if not args.skip_clearance:
        best, clips, face, poses = _clearance(model, mujoco, qpos, render_fps)
        manifest["clearance"] = {
            "method": (f"vertex-sampled: target meshes voxelised at {VOXEL_M * 1000:.1f} mm, "
                       f"primitives surface-sampled at {SAMPLE_STEP_M * 1000:.1f} mm, motions "
                       f"sampled at {SAMPLE_HZ:g} Hz; distances in the upper_shell_link frame; "
                       f"null = clear by more than {SEARCH_M * 1000:.0f} mm"),
            "threshold_m": CLEARANCE_MIN_M,
            "search_m": SEARCH_M,
            "pose_search_m": POSE_SEARCH_M,
            "decal_proud_m": DECAL_PROUD_M,
            "rule": ("proud parts keep >= threshold_m from claws, legs and pitcher on every "
                     "swept sample and stay out of the eye-panel box; decals stand at most "
                     "decal_proud_m off the face, so their sweep minimum is reported, not "
                     "gated; bare_face_reference is the same sweep against the robot's own face "
                     "under the decals"),
            "bare_face_reference": {g: _rounded(best[(BARE_FACE, g)]) for g in GROUPS},
            "per_concept": {},
            "poses": poses,
            "clips": clips,
        }
        ref = manifest["clearance"]["bare_face_reference"]
        print("[concepts] bare face under the decals: " + ", ".join(
            f"{g} >{SEARCH_M * 1000:.0f} mm" if v["min_m"] is None
            else f"{g} {v['min_m'] * 1000:.1f} mm ({v['clip']})" for g, v in ref.items()))
        for c in CONCEPTS:
            row = {
                "face_box_intrusions": face[c.key],
                "proud": {g: _rounded(best[((c.key, "proud"), g)]) for g in GROUPS},
                "decal": {g: _rounded(best[((c.key, "decal"), g)]) for g in GROUPS},
            }
            row["passes"] = (not face[c.key]) and all(
                v["min_m"] is None or v["min_m"] >= CLEARANCE_MIN_M
                for v in row["proud"].values())
            manifest["clearance"]["per_concept"][c.key] = row
            for layer in LAYERS:
                mins = ", ".join(
                    f"{g} >{SEARCH_M * 1000:.0f} mm" if v["min_m"] is None
                    else f"{g} {v['min_m'] * 1000:.1f} mm ({v['part']}, {v['clip']})"
                    for g, v in row[layer].items())
                print(f"[concepts] {c.key} {layer}: {mins}")
            for pose, entry in poses.items():
                vals = ", ".join(
                    f"{layer} {g} " + (f">{POSE_SEARCH_M * 1000:.0f}" if (v := entry["min_m"][
                        f"{c.key}/{layer}/{g}"]) is None else f"{v * 1000:.1f}") + " mm"
                    for layer in LAYERS for g in GROUPS)
                print(f"[concepts] {c.key} at {pose} pose ({entry['time_s']:g} s): {vals}")
            print(f"[concepts] {c.key}: passes={row['passes']} "
                  f"face_box_intrusions={row['face_box_intrusions']}")
    (out / "concepts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    for k, v in files.items():
        print(f"[concepts] {k}: {v}")
    print(f"[concepts] manifest: {out / 'concepts_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
