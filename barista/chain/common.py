"""Shared paths, loaders and small pose helpers for the A4 motion chain.

``barista/`` is not an installed package and repository code may not rewrite ``sys.path``
(``tests/test_layout.py``), so sibling modules are loaded by path with :func:`load_module`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CHAIN_DIR = REPO / "barista" / "chain"
M4_TRAJECTORY = REPO / "barista" / "trajectory" / "m4_spiral_pour.json"
CAFE_DIMENSIONS = REPO / "barista" / "cafe" / "cafe_dimensions.json"
RUN_M4 = REPO / "barista" / "verification" / "run_m4.py"
SCENE = "cafe_bar"
SEED = 41001
PHYSICS_HZ = 1000.0
CONTROL_HZ = 200.0
#: run_m4 keyframe-target decimation at 1 kHz physics: 50 Hz targets, as M4's own 4 x 5 ms.
M4_DECIMATION_1K = 20
BEATS_SCHEMA = "jumper-barista-beats/1"
BEAT_NAMES = (
    "dance", "chime", "stop", "excited_jump", "exclaim", "look", "listen", "confirm",
    "turn", "walk", "align", "pour", "set_down", "bow",
)
FEET = ("LF", "LM", "LR", "RF", "RM", "RR")


def headless() -> None:
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


def load_module(name: str, path: Path):
    """Import a repository file by path (``barista/`` is not a package on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def chain_module(stem: str):
    return load_module(f"barista_chain_{stem}", CHAIN_DIR / f"{stem}.py")


def load_run_m4():
    return load_module("barista_run_m4", RUN_M4)


def cafe_dims() -> dict:
    return json.loads(CAFE_DIMENSIONS.read_text("utf-8"))


def commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tree_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.strip())


def sha256_array(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cafe_m4_trajectory(physics_dt_s: float, decimation: int) -> dict:
    """The accepted M4 trajectory, rigidly moved into the cafe (A1's transfer) and re-rated.

    Only the three world-frame positions move (by the cafe's compiled station offset) and the
    physics/control/stream periods change; every keyframe, profile and threshold is the
    accepted one.
    """
    config = json.loads(M4_TRAJECTORY.read_text("utf-8"))
    offset = np.asarray(cafe_dims()["brew_station"]["compiled_offset_m"], dtype=float)

    def moved(values) -> list[float]:
        return [round(float(v), 9) for v in np.asarray(values, dtype=float) + offset]

    config["base_start"]["position_m"] = moved(config["base_start"]["position_m"])
    config["stream"]["bed_centre_m"] = moved(config["stream"]["bed_centre_m"])
    config["set_down_target_m"] = moved(config["set_down_target_m"])
    config["physics_dt_s"] = physics_dt_s
    config["control_dt_s"] = physics_dt_s * decimation
    config["stream_dt_s"] = physics_dt_s * decimation
    config["a4_chain"] = {
        "source_trajectory": "barista/trajectory/m4_spiral_pour.json",
        "source_sha256": sha256_file(M4_TRAJECTORY),
        "cafe_compiled_offset_m": offset.tolist(),
        "moved_fields": ["base_start.position_m", "stream.bed_centre_m", "set_down_target_m"],
        "rerated_fields": ["physics_dt_s", "control_dt_s", "stream_dt_s"],
        "decimation": decimation,
        "scene": SCENE,
    }
    return config


# ------------------------------------------------------------------------- pose helpers
def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_of(quat) -> tuple[float, float]:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def tilt_of(quat) -> float:
    _, x, y, _ = quat
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def rot_log(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-9:
        return 0.5 * vee
    return vee * (angle / (2.0 * math.sin(angle)))


def smooth(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)
