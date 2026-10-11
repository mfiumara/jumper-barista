"""The Jumper Barista outfit: Concept C, "Dapper Tricolore" (A2b, simulation only).

Mattia picked Concept C as is (kanban gate t_bae58085: "Outfit = Concept C, Dapper
Tricolore (CONCEPTS.md @711e027), as is."). This module turns the A2a concept
primitives into real, **display-only** geoms in the compiled model:

- a straw trilby (brim, crown, flattened top) with a green-white-red band,
- a pencil moustache, a flush decal at most ``DECAL_PROUD_M`` (0.3 mm) proud of the face,
- a cream bar towel with a red stripe over the rear-right of the shell,
- the lower body (``base_link_visual``) recoloured to a white barista jacket.

Every part is a primitive geom on ``upper_shell_link`` with ``contype=0``,
``conaffinity=0``, ``density=0`` and visual group 2. ``upper_shell_link`` (like every
Jumper body) carries an explicit ``<inertial>``, so the geoms add no mass or inertia, and
they are outside the robot's ``CollisionCfg`` name list, so nothing re-enables
collision on them. The recolour touches ``rgba`` only. The physics is therefore the
stock robot's; ``tests/test_barista_outfit.py`` asserts this and ``outfit_check.py m4``
proves it against the accepted M4 replay hashes.

Geometry is the A2a concept geometry verbatim (``concepts.py`` ``CONCEPTS[C]``), with one
change CONCEPTS.md asked for: the moustache decal is 0.3 mm proud instead of 0.8 mm.

Use ``add_to_spec(spec)`` on the robot's own MjSpec (names unprefixed), ``apply(env_cfg)``
on an mjlab env config, or ``installed()`` to make every ``tasks.load_env_cfg`` call in a
block return a dressed robot (that is how unmodified scripts such as ``run_m4.py`` and the
dance/jump policy rollouts are run in costume).
"""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from concepts import CONCEPTS, Part, _euler_mat, _face_x, _seg_frame

KEY = "C"
CONCEPT = next(c for c in CONCEPTS if c.key == KEY)
NAME = CONCEPT.name  # "Dapper Tricolore"
SHELL_BODY = "upper_shell_link"
PREFIX = "outfit_"
#: Decal standoff: the flush moustache stands at most this far off the face.
DECAL_PROUD_M = 0.0003
DECAL_HALF_T = 0.00015
VISUAL_GROUP = 2
#: Garment recolours, by unprefixed visual geom name.
RECOLOUR = {k.split("/", 1)[1]: v for k, v in CONCEPT.recolour.items()}
PARTS: tuple[Part, ...] = CONCEPT.parts


@dataclass(frozen=True)
class GeomSpec:
    name: str
    part: str
    kind: str  # MuJoCo geom type name
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # wxyz
    rgba: tuple[float, float, float, float]
    decal: bool


def _mat2quat(mat: np.ndarray) -> tuple[float, float, float, float]:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat, dtype=np.float64).reshape(-1))
    return tuple(float(v) for v in q)


def _pose(p: Part):
    """(type, size[3], pos[3], mat) in the shell frame; concepts._part_pose with the
    thinner decal."""
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


def geoms() -> list[GeomSpec]:
    out = []
    for p in PARTS:
        kind, size, pos, mat = _pose(p)
        out.append(GeomSpec(PREFIX + p.name, p.name, kind,
                            tuple(round(float(v), 9) for v in size),
                            tuple(round(float(v), 9) for v in pos), _mat2quat(mat),
                            tuple(float(v) for v in p.rgba), p.kind == "decal"))
    return out


def add_to_spec(spec) -> None:
    """Dress a robot MjSpec in place (unprefixed names, as ``get_spec`` returns it)."""
    import mujoco

    body = spec.body(SHELL_BODY)
    if body is None:
        raise KeyError(f"robot spec has no body {SHELL_BODY!r}")
    if any(g.name.startswith(PREFIX) for g in spec.geoms):
        raise RuntimeError("the outfit is already applied to this spec")
    for g in geoms():
        body.add_geom(
            name=g.name,
            type=getattr(mujoco.mjtGeom, f"mjGEOM_{g.kind.upper()}"),
            size=list(g.size), pos=list(g.pos), quat=list(g.quat), rgba=list(g.rgba),
            contype=0, conaffinity=0, group=VISUAL_GROUP, density=0.0,
        )
    for name, rgba in RECOLOUR.items():
        geom = spec.geom(name)
        if geom is None:
            raise KeyError(f"robot spec has no geom {name!r} to recolour")
        geom.rgba = list(rgba)


def dressed_spec_fn(spec_fn):
    def build():
        spec = spec_fn()
        add_to_spec(spec)
        return spec

    build.barista_outfit = NAME
    return build


def apply(env_cfg, entity: str = "robot"):
    """Dress the robot entity of an mjlab env config; returns the config."""
    cfg = env_cfg.scene.entities[entity]
    if getattr(cfg.spec_fn, "barista_outfit", None) is None:
        cfg.spec_fn = dressed_spec_fn(cfg.spec_fn)
    return env_cfg


@contextlib.contextmanager
def installed():
    """Within the block, ``tasks.load_env_cfg`` returns configs with the outfit applied."""
    import tasks

    original = tasks.load_env_cfg

    def load_env_cfg(*args, **kwargs):
        return apply(original(*args, **kwargs))

    tasks.load_env_cfg = load_env_cfg
    try:
        yield
    finally:
        tasks.load_env_cfg = original


def geom_ids(model) -> list[int]:
    """Compiled-model ids of the outfit geoms (prefix-tolerant)."""
    import mujoco

    ids = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.rsplit("/", 1)[-1].startswith(PREFIX):
            ids.append(g)
    return ids


def visual_overrides() -> dict:
    """The outfit as a portable display-only description (``visual_overrides`` JSON)."""
    return {
        "format": "jumper-barista-visual-overrides/1",
        "outfit": NAME,
        "concept": KEY,
        "source": "barista/outfit/concepts.py CONCEPTS[C] @711e027; decal 0.3 mm proud",
        "display_only": True,
        "frame": SHELL_BODY + " (+x face, +y robot-left, +z up), metres, quat wxyz",
        "collision": {"contype": 0, "conaffinity": 0},
        "density": 0.0,
        "group": VISUAL_GROUP,
        "recolour": [{"geom": k, "rgba": list(v)} for k, v in RECOLOUR.items()],
        "geoms": [{"name": g.name, "body": SHELL_BODY, "type": g.kind, "size": list(g.size),
                   "pos": list(g.pos), "quat": [round(v, 9) for v in g.quat],
                   "rgba": list(g.rgba), "decal": g.decal} for g in geoms()],
    }


def write_visual_overrides(path: Path) -> Path:
    path.write_text(json.dumps(visual_overrides(), indent=2) + "\n", "utf-8")
    return path


def hex_rgb(rgba) -> str:
    return "#" + "".join(f"{round(v * 255):02X}" for v in rgba[:3])


