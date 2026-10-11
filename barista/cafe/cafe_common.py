"""Shared helpers for the cafe (Act 2 / A1) verification scripts.

Simulation only. Everything here runs on ``native:cpu`` and renders through
offscreen CGL (macOS) or EGL; nothing opens a window.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIMENSIONS = REPO / "barista" / "cafe" / "cafe_dimensions.json"
M4_TRAJECTORY = REPO / "barista" / "trajectory" / "m4_spiral_pour.json"
SCENE = "cafe_bar"
PEAK_LIMIT_NM = 1.7464
CONTINUOUS_LIMIT_NM = 1.2
PEAK_WINDOW_S = 0.3


def headless() -> None:
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


def dims() -> dict:
    return json.loads(DIMENSIONS.read_text("utf-8"))


def backend():
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)
    return resolution


def pin_reset(env_cfg) -> None:
    """Pin a task's randomised planar reset so the robot starts at the env origin."""
    reset = env_cfg.events.get("reset_base")
    if reset is None:
        return
    pose_range = dict(reset.params.get("pose_range", {}))
    for axis in ("x", "y", "roll", "pitch", "yaw"):
        pose_range[axis] = (0.0, 0.0)
    reset.params["pose_range"] = pose_range
    velocity_range = dict(reset.params.get("velocity_range", {}))
    for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
        velocity_range[axis] = (0.0, 0.0)
    reset.params["velocity_range"] = velocity_range


def name(model, kind, index: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def geom_name(model, index: int) -> str:
    import mujoco

    return name(model, mujoco.mjtObj.mjOBJ_GEOM, index)


def bar_geoms(model) -> set[int]:
    """The bar top's collision geoms: the accepted slab (now `terrain`) and the
    coplanar extensions. Standing on any of them is standing on the bar."""
    return {g for g in range(model.ngeom)
            if geom_name(model, g) == "terrain" or "bar_top_ext_" in geom_name(model, g)}


def robot_collision_geoms(model) -> list[int]:
    return [
        g for g in range(model.ngeom)
        if geom_name(model, g).startswith("robot/")
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def base_body(model) -> int:
    import mujoco

    for b in range(model.nbody):
        if name(model, mujoco.mjtObj.mjOBJ_BODY, b).endswith("base_link"):
            return b
    raise RuntimeError("no base_link body")


def geom_points(model, data, g: int, circle: int = 24) -> np.ndarray:
    """World-frame points whose XY convex hull bounds the geom's XY projection."""
    import mujoco

    kind = int(model.geom_type[g])
    pos = data.geom_xpos[g]
    rot = data.geom_xmat[g].reshape(3, 3)
    size = model.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[g])
        start = int(model.mesh_vertadr[mesh])
        verts = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])]
        return pos + verts @ rot.T
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return pos + (corners * size) @ rot.T
    angles = np.linspace(0.0, 2.0 * math.pi, circle, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos + np.concatenate([ring, ring[:, [2, 0, 1]], ring[:, [0, 2, 1]]]) * size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        local = np.concatenate([
            ring * size[0] + [0, 0, size[1]], ring * size[0] - [0, 0, size[1]],
        ])
        return pos + local @ rot.T
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ends = np.array([[0, 0, size[1]], [0, 0, -size[1]]]) @ rot.T + pos
        # A sphere of radius r at each end bounds the capsule in every projection.
        sphere = np.concatenate([ring, ring[:, [2, 0, 1]], ring[:, [0, 2, 1]]]) * size[0]
        return np.concatenate([ends[0] + sphere, ends[1] + sphere])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        sphere = np.concatenate([ring, ring[:, [2, 0, 1]], ring[:, [0, 2, 1]]])
        return pos + (sphere * size) @ rot.T
    raise ValueError(f"geom {geom_name(model, g)} has unsupported type {kind}")


def hull(points: np.ndarray) -> np.ndarray:
    """2D convex hull (counter-clockwise), Andrew's monotone chain."""
    pts = np.unique(np.round(np.asarray(points, dtype=float)[:, :2], 9), axis=0)
    if len(pts) <= 2:
        return pts

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons."""
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _points_to_polygon(points: np.ndarray, poly: np.ndarray) -> float:
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a  # (m, 2)
    ap = points[:, None, :] - a[None, :, :]  # (n, m, 2)
    denom = np.maximum((ab * ab).sum(-1), 1e-30)
    t = np.clip((ap * ab[None]).sum(-1) / denom, 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return float(np.linalg.norm(points[:, None, :] - closest, axis=-1).min())


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex polygons; 0 when they overlap."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) >= 3 and len(b) >= 3 and _overlap(a, b):
        return 0.0
    return min(_points_to_polygon(a, b), _points_to_polygon(b, a))


def transform(poly: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return poly @ rot.T + np.array([x, y])


def torque_summary(torque: np.ndarray, dt_s: float, names: list[str]) -> dict:
    """Peak vs the servo curve's 1.7464 N m and the longest run above 1.2 N m."""
    if not len(torque):
        return {}
    magnitude = np.abs(torque)
    peak_index = np.unravel_index(magnitude.argmax(), magnitude.shape)
    longest = 0
    for column in range(magnitude.shape[1]):
        run = 0
        for value in magnitude[:, column] > CONTINUOUS_LIMIT_NM:
            run = run + 1 if value else 0
            longest = max(longest, run)
    return {
        "peak_abs_nm": float(magnitude[peak_index]),
        "peak_actuator": names[peak_index[1]],
        "peak_within_curve": bool(magnitude[peak_index] <= PEAK_LIMIT_NM + 1e-6),
        "samples_over_continuous": int(np.count_nonzero(magnitude > CONTINUOUS_LIMIT_NM)),
        "longest_over_continuous_s": longest * dt_s,
        "peak_window_passes": bool(longest * dt_s <= PEAK_WINDOW_S),
        "continuous_limit_nm": CONTINUOUS_LIMIT_NM,
        "peak_limit_nm": PEAK_LIMIT_NM,
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", "utf-8")
