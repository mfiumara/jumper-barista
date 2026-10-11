"""Dance-area envelope from saved render-rate states (the spike's p8 tool, fixed).

The spike's ``p8_footprint.py`` always rebuilt the *plain* world (qpos width 29) and then
replayed states recorded in a bar scene with a free pitcher (width 36), which crashed. Here
the model is built for the scene the states were recorded in, and the width is checked
before any replay with an error that names both sides.

Radius = max horizontal distance from the centre to any robot collision geom, centre plus
bounding radius (conservative), as in A1's ``footprints.py`` and the spike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

common = sys.modules.get("barista_chain_common")
if common is None:  # loaded by path
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("barista_chain_common", Path(__file__).with_name("common.py"))
    common = _ilu.module_from_spec(_spec)
    sys.modules["barista_chain_common"] = common
    _spec.loader.exec_module(common)


class StateWidthError(ValueError):
    pass


def check_state_width(states: np.ndarray, nq: int, scene: str) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != nq:
        raise StateWidthError(
            f"states have qpos width {states.shape[-1] if states.ndim else 'scalar'} but the "
            f"{scene!r} model has nq={nq}: replay them in the scene they were recorded in")
    return states


def robot_collision_geoms(model) -> list[int]:
    import mujoco

    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def envelope_radius(model, states, centre_xy, scene: str = "?") -> tuple[float, int]:
    """Largest footprint radius over ``states`` about ``centre_xy``; and the frame index."""
    import mujoco

    states = check_state_width(states, model.nq, scene)
    data = mujoco.MjData(model)
    geoms = np.asarray(robot_collision_geoms(model))
    centre = np.asarray(centre_xy, dtype=float)
    best, where = 0.0, -1
    for i, q in enumerate(states):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        r = float(np.max(np.linalg.norm(data.geom_xpos[geoms][:, :2] - centre, axis=1)
                         + model.geom_rbound[geoms]))
        if r > best:
            best, where = r, i
    return best, where


def build_model(scene: str):
    """The jumper.barista world (robot + scene) for ``scene``, compiled once, no physics."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import scenes
    import tasks

    use_backend(resolve(backend="native", device="cpu", num_envs=1, training=False))
    cfg = tasks.load_env_cfg("jumper.barista", play=True)
    scenes.apply(cfg, scene)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="a run_chain output directory")
    parser.add_argument("--scene", default=None, help="defaults to the run's beats.json scene")
    args = parser.parse_args()
    common.headless()
    beats = json.loads((args.run / "beats.json").read_text())
    scene = args.scene or beats.get("scene", common.SCENE)
    rs = np.load(args.run / "render_states.npz")
    env = build_model(scene)
    try:
        model = env.sim.mj_model
        t = rs["time_s"]
        window = [b for b in beats["beats"] if b["beat"] in ("dance", "stop", "excited_jump")]
        t0, t1 = window[0]["t_start_s"], window[-1]["t_end_s"]
        sel = (t >= t0) & (t <= t1)
        area = common.cafe_dims()["dance_area"]
        r, i = envelope_radius(model, rs["qpos"][sel], area["centre_m"], scene)
        out = {"scene": scene, "window_s": [t0, t1], "frames": int(sel.sum()),
               "envelope_radius_m": r, "at_time_s": float(t[sel][i]),
               "dance_area_radius_m": area["radius_m"], "inside": r <= area["radius_m"]}
    finally:
        env.close()
    (args.run / "dance_area.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["inside"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
