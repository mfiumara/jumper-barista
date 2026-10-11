#!/usr/bin/env python3
"""Replay the accepted M4 pour in the cafe by one rigid translation, and gate it.

Simulation only (native:cpu). Criterion A1-3. Nothing of the accepted M4 is edited:

1. The accepted trajectory (``barista/trajectory/m4_spiral_pour.json``) is copied
   with its three world-frame quantities -- ``base_start.position_m``,
   ``stream.bed_centre_m`` and ``set_down_target_m`` -- moved by the cafe's
   ``brew_station.compiled_offset_m``. Yaw is unchanged (the copy is a pure
   translation), and every joint-space keyframe, profile and threshold is the
   accepted one byte for byte.
2. ``barista/verification/run_m4.py`` -- the accepted verifier, unmodified -- runs
   on that trajectory with its one scene lookup (``scenes.apply(env_cfg,
   "coffee_bar")``) redirected to ``cafe_bar``. Every M4 gate (turns, landings,
   stream obstruction ray-casts, set-down, brewer and free-leg collisions, torque
   audit) therefore runs unchanged against the cafe world.
3. From the saved render-rate states it additionally measures how close the robot
   and pitcher come to the cafe's visual-only V60 cues (``dripper_*``), which no
   M4 gate can see because they do not collide.

    python barista/cafe/run_m4_transfer.py --out out/cafe/m4_transfer
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cafe_common as cc
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true",
                        help="skip the M4 verifier's hero video (gates are unaffected)")
    return parser


def transferred_trajectory(offset) -> dict:
    """The accepted M4 trajectory with its world-frame positions translated."""
    config = json.loads(cc.M4_TRAJECTORY.read_text("utf-8"))
    offset = np.asarray(offset, dtype=float)

    def moved(values) -> list[float]:
        return [round(float(v), 9) for v in np.asarray(values, dtype=float) + offset]

    config["base_start"]["position_m"] = moved(config["base_start"]["position_m"])
    config["stream"]["bed_centre_m"] = moved(config["stream"]["bed_centre_m"])
    config["set_down_target_m"] = moved(config["set_down_target_m"])
    config["cafe_transfer"] = {
        "source_trajectory": "barista/trajectory/m4_spiral_pour.json",
        "source_sha256": hashlib.sha256(cc.M4_TRAJECTORY.read_bytes()).hexdigest(),
        "compiled_offset_m": offset.tolist(),
        "moved_fields": ["base_start.position_m", "stream.bed_centre_m", "set_down_target_m"],
        "scene": cc.SCENE,
    }
    return config


def _shell_distance(points_local: np.ndarray) -> float:
    """Distance from points (dripper frame) to the axisymmetric shell solid; 0 inside."""
    from scenes.cafe_bar import SHELL_THICKNESS, SHELL_Z, shell_inner_radius

    z0, z1 = SHELL_Z
    section = np.array([
        [shell_inner_radius(z0), z0], [shell_inner_radius(z0) + SHELL_THICKNESS, z0],
        [shell_inner_radius(z1) + SHELL_THICKNESS, z1], [shell_inner_radius(z1), z1],
    ])
    p = np.stack([np.hypot(points_local[:, 0], points_local[:, 1]), points_local[:, 2]], 1)
    distance = np.full(len(p), np.inf)
    inside = np.ones(len(p), dtype=bool)
    for i in range(4):
        a, b = section[i], section[(i + 1) % 4]
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
        distance = np.minimum(distance, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
        inside &= (ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) >= 0.0
    distance[inside] = 0.0
    return float(distance.min()) if len(distance) else float("inf")


def visual_clearance(out: Path) -> dict:
    """Minimum distance from robot and pitcher geometry to the visual V60 cues.

    The cues do not collide, so no M4 gate sees them. The flange and handle are
    convex primitives and use MuJoCo's own geom distance; the shell is an open
    cone whose mesh hull MuJoCo would fill in, so it is measured exactly as an
    axisymmetric solid against sampled surface points of every robot and pitcher
    geom (mesh vertices; primitive bounding samples).
    """
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv

    import scenes
    import tasks

    states = np.load(out / "render_states.npz")
    qpos = states["qpos"]
    fps = float(states["render_fps"])
    resolution = cc.backend()
    env_cfg = tasks.load_env_cfg("jumper.barista", play=True)
    scenes.apply(env_cfg, cc.SCENE)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    try:
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        cues = [g for g in range(model.ngeom) if "dripper_" in cc.geom_name(model, g)]
        shell = next(g for g in cues if cc.geom_name(model, g).endswith("dripper_shell"))
        convex = [g for g in cues if g != shell]
        shell_body = int(model.geom_bodyid[shell])
        pitcher_body = next(b for b in range(model.nbody)
                            if cc.name(model, mujoco.mjtObj.mjOBJ_BODY, b).endswith("_pitcher"))
        movers = [g for g in range(model.ngeom)
                  if model.geom_bodyid[g] == pitcher_body
                  or (cc.geom_name(model, g).startswith("robot/")
                      and (model.geom_contype[g] or model.geom_conaffinity[g]))]
        best = {"distance_m": np.inf}
        fromto = np.zeros(6)
        reach = 0.12  # anything further from the dripper axis than this is irrelevant
        for frame, state in enumerate(qpos):
            data.qpos[:] = state
            mujoco.mj_kinematics(model, data)
            # The dripper body's frame, not the mesh geom's: MuJoCo recentres and
            # may rotate a mesh geom's frame onto the mesh's inertial axes.
            origin = data.xpos[shell_body]
            rot = data.xmat[shell_body].reshape(3, 3)
            for mover in movers:
                gap = float(np.linalg.norm(data.geom_xpos[mover] - origin))  # body origin
                if gap - model.geom_rbound[mover] > reach:
                    continue
                local = (cc.geom_points(model, data, mover) - origin) @ rot
                distance = _shell_distance(local)
                cue = shell
                for g in convex:
                    value = float(mujoco.mj_geomDistance(model, data, g, mover, 0.1, fromto))
                    if value < distance:
                        distance, cue = value, g
                if distance < best["distance_m"]:
                    best = {"distance_m": distance, "frame": frame, "time_s": frame / fps,
                            "cue": cc.geom_name(model, cue), "geom": cc.geom_name(model, mover)}
        best["cue_geoms"] = [cc.geom_name(model, g) for g in cues]
        best["frames"] = len(qpos)
        best["no_visual_intersection"] = best["distance_m"] > 0.0
        return best
    finally:
        env.close()


def main() -> int:
    args = _parser().parse_args()
    cc.headless()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = cc.dims()
    offset = d["brew_station"]["compiled_offset_m"]
    config = transferred_trajectory(offset)
    trajectory = out / "m4_spiral_pour_cafe.json"
    trajectory.write_text(json.dumps(config, indent=2) + "\n", "utf-8")

    # Check the declared pour-start pose is the transferred M4 start, not a copy of it.
    pose = d["route"]["pour_start_pose"]
    start = config["base_start"]
    declared = np.array([pose["x_m"], pose["y_m"], pose["base_z_m"]])
    if np.abs(declared - np.asarray(start["position_m"])).max() > 1e-6 or \
            abs(float(start["yaw_rad"]) - np.radians(pose["yaw_deg"])) > 1e-9:
        raise SystemExit(f"pour_start_pose {declared} differs from the transferred M4 "
                         f"start {start}")

    import scenes

    original_apply = scenes.apply

    def apply(env_cfg, scene_id, **kwargs):
        return original_apply(env_cfg, cc.SCENE if scene_id == "coffee_bar" else scene_id,
                              **kwargs)

    scenes.apply = apply
    # Loaded by path: `barista/` is not an installed package (and nothing here
    # rewrites sys.path -- tests/test_layout.py forbids it).
    path = cc.REPO / "barista" / "verification" / "run_m4.py"
    module_spec = importlib.util.spec_from_file_location("barista_run_m4", path)
    run_m4 = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(run_m4)
    argv = ["run_m4.py", "--out", str(out), "--trajectory", str(trajectory),
            "--save-render-states"]
    if args.no_render:
        argv.append("--no-render")
    saved_argv = sys.argv
    sys.argv = argv
    try:
        code = run_m4.main()
    finally:
        sys.argv = saved_argv
        scenes.apply = original_apply

    summary = json.loads((out / "summary.json").read_text("utf-8"))
    stream = summary["stream"]
    motion = summary["motion"]
    clearance = visual_clearance(out)
    transfer = {
        "simulation_only": True,
        "scene": cc.SCENE,
        "note": "run_m4.py's summary.json says scene 'coffee_bar' because its label is "
                "hard-coded; the environment it built is cafe_bar (see this file)",
        "compiled_offset_m": offset,
        "trajectory": str(trajectory),
        "run_m4_exit_code": code,
        "m4_passes": summary["passes"],
        "turns": stream["turn_count"],
        "landing_success": [stream["landing_success_count"], stream["sample_count"]],
        "landing_success_rate": stream["landing_success_rate"],
        "maximum_landing_radius_m": stream["maximum_landing_radius_m"],
        "obstructed_paths": stream["obstructed_path_count"],
        "obstruction_ray_control_hit_geom": stream["obstruction_ray_control_hit_geom"],
        "stream_checks": stream["checks"],
        "set_down_error_m": motion["maximum_final_position_error_m"],
        "minimum_pitcher_brewer_clearance_m": motion["minimum_pitcher_brewer_clearance_all_phases_m"],
        "minimum_robot_brewer_clearance_m": motion["minimum_robot_brewer_clearance_all_phases_m"],
        "robot_scene_contact_rows": motion["robot_scene_contact_rows"],
        "free_leg_contact_rows": motion["free_leg_contact_rows"],
        "motion_checks": motion["checks"],
        "torque_passes": summary["torque"]["passes"],
        "payload_kg": summary["payload"]["design_total_kg"],
        "visual_cue_clearance": clearance,
        "passes": bool(summary["passes"] and clearance["no_visual_intersection"] and code == 0),
    }
    cc.write_json(out / "transfer_summary.json", transfer)
    print(f"[cafe-m4] turns={transfer['turns']:.4f}; landings "
          f"{transfer['landing_success'][0]}/{transfer['landing_success'][1]} "
          f"(rate {transfer['landing_success_rate']:.3f}); set-down error "
          f"{transfer['set_down_error_m'] * 1000:.2f} mm; torque pass={transfer['torque_passes']}; "
          f"M4 gates pass={transfer['m4_passes']}")
    print(f"[cafe-m4] nearest visual V60 cue: {clearance['distance_m'] * 1000:.2f} mm "
          f"({clearance.get('geom')} vs {clearance.get('cue')})")
    print(f"[cafe-m4] artifact: {out / 'transfer_summary.json'}")
    return 0 if transfer["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
