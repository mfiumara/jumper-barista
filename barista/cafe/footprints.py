#!/usr/bin/env python3
"""Measure the bar-top footprint of the existing dance, jump and walking policies.

Simulation only (native:cpu, no window). Each policy is replayed **in the
cafe_bar scene** from its own example checkpoint, the way ``play.py`` would, and
at every control step the XY convex hull of all of Jumper's collision geometry is
recorded. That gives, per policy:

- the swept footprint and its largest distance from the start-slot centre, which
  is what sizes ``dance_area`` (dances and the flat jump are performed in place);
- any robot contact with something other than the bar top, a fall
  (termination) and the actuator torque against the servo budget;
- for the walking policy (``jumper.posture``, the bundle's default walk) two
  route segments driven by constant commands: the +90 deg turn in place at the
  start slot, and the walk along +X towards the brew station, stopped at the
  route's ``pre_align`` waypoint.

    python barista/cafe/footprints.py --out out/cafe/footprints
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import cafe_common as cc
import numpy as np

POLICIES = {
    "dance_crab": ("jumper.dance", "tasks/jumper/dance/out/example/model_2300.pt"),
    "dance_brazilian": ("jumper.dance_brazilian",
                        "tasks/jumper/dance_brazilian/out/example/model_1999.pt"),
    "dance_dream_wings": ("jumper.dance_dream_wings",
                          "tasks/jumper/dance_dream_wings/out/example/model_1999.pt"),
    "dance_maze": ("jumper.dance_maze", "tasks/jumper/dance_maze/out/example/model_1999.pt"),
    "dance_waist": ("jumper.dance_waist", "tasks/jumper/dance_waist/out/example/model_1999.pt"),
    "jump": ("jumper.jump", "tasks/jumper/jump/out/example/model_3100.pt"),
    "walk_turn": ("jumper.posture", "tasks/jumper/posture/out/example/model_74800.pt"),
    "walk_along_bar": ("jumper.posture", "tasks/jumper/posture/out/example/model_74800.pt"),
}
TURN_RATE_RAD_S = 0.6
WALK_SPEED_M_S = 0.15
MAX_WALK_S = 12.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(POLICIES))
    return parser


def _neutral_posture(env_cfg) -> None:
    posture = env_cfg.commands["posture"]
    posture.rel_neutral_envs = 1.0
    posture.rel_neutral_standing_envs = 1.0
    posture.resampling_time_range = (1.0e6, 1.0e6)


def _hold_twist(env, vx: float, wz: float) -> None:
    """Pin the velocity command to (vx, 0, wz) after every compute.

    The ranges cannot simply be collapsed onto the value: the operator layer
    refuses a range that excludes the stick's rest. So the term samples as usual
    and this writes the constant last, the same way the operator itself does.
    """
    import torch

    term = env.command_manager.get_term("twist")
    original = term.compute
    value = torch.tensor([vx, 0.0, wz], dtype=torch.float, device=env.device)

    def compute(dt, env_ids=None):
        original(dt, env_ids)
        term.vel_command_b[:] = value

    term.compute = compute
    term.vel_command_b[:] = value


def rollout(label: str, out: Path) -> dict:
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import scenes
    import tasks

    task_id, checkpoint = POLICIES[label]
    d = cc.dims()
    resolution = cc.backend()
    env_cfg = tasks.load_env_cfg(task_id, play=True)
    env_cfg.seed = 0
    env_cfg.scene.num_envs = 1
    cc.pin_reset(env_cfg)
    start_yaw = 0.0
    twist = {"walk_turn": (0.0, TURN_RATE_RAD_S), "walk_along_bar": (WALK_SPEED_M_S, 0.0)}
    if label in twist:
        _neutral_posture(env_cfg)
    if label == "walk_turn":
        start_yaw = math.radians(d["route"]["start_slot"]["yaw_deg"])
    scenes.apply(env_cfg, cc.SCENE)
    agent_cfg = tasks.load_agent_cfg(task_id)
    control_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=resolution.device)
        runner.load(str(cc.REPO / checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=resolution.device)
        policy = runner.get_inference_policy(device=resolution.device)
        robot = env.scene["robot"]
        if label in twist:
            _hold_twist(env, *twist[label])
        obs, _ = wrapped.reset()
        if start_yaw:
            pose = robot.data.root_link_pose_w.clone()
            pose[0, 3:7] = torch.tensor([math.cos(start_yaw / 2), 0.0, 0.0,
                                         math.sin(start_yaw / 2)])
            robot.write_root_link_pose_to_sim(pose)
            env.sim.forward()
            env.scene.update(dt=0.0)
            obs = wrapped.get_observations()
        model = env.sim.mj_model
        data = env.sim._datas[0]
        geoms = cc.robot_collision_geoms(model)
        bar = cc.bar_geoms(model)
        base = cc.base_body(model)
        if "motion" in env.command_manager.active_terms:
            frames = int(env.command_manager.get_term("motion").motion.time_step_total) - 1
        elif label == "jump":
            frames = round(env_cfg.episode_length_s / control_dt) - 1
        elif label == "walk_turn":
            frames = round((abs(start_yaw) / TURN_RATE_RAD_S + 4.0) / control_dt)
        else:
            frames = round(MAX_WALK_S / control_dt)
        pre_align_x = next(w["x_m"] for w in d["route"]["walk"] if w["name"] == "pre_align")
        centre = np.zeros(2)
        hulls: list[np.ndarray] = []
        base_xy: list[list[float]] = []
        yaws: list[float] = []
        radius = []
        torque = []
        others: set[tuple[str, str]] = set()
        stopped = None
        mujoco.mj_kinematics(model, data)
        start_xy = data.xpos[base][:2].copy()
        with torch.inference_mode():
            for frame in range(frames):
                obs, _, dones, _ = wrapped.step(policy(obs))
                if bool(dones[0]):
                    stopped = frame
                    break
                points = np.concatenate([cc.geom_points(model, data, g) for g in geoms])
                poly = cc.hull(points)
                hulls.append(poly)
                radius.append(float(np.linalg.norm(poly - centre, axis=1).max()))
                rot = data.xmat[base].reshape(3, 3)
                yaw = math.atan2(rot[1, 0], rot[0, 0])
                yaws.append(yaw)
                base_xy.append(data.xpos[base][:2].tolist())
                torque.append(robot.data.actuator_force[0].detach().cpu().numpy().copy())
                for k in range(data.ncon):
                    c = data.contact[k]
                    n1, n2 = cc.geom_name(model, int(c.geom1)), cc.geom_name(model, int(c.geom2))
                    for rn, og, on in ((n1, int(c.geom2), n2), (n2, int(c.geom1), n1)):
                        if rn.startswith("robot/") and not on.startswith("robot/") \
                                and og not in bar:
                            others.add((rn, on))
                if label == "walk_turn" and yaw >= -math.radians(1.0) and frame > 10:
                    break
                if label == "walk_along_bar" and data.xpos[base][0] >= pre_align_x:
                    break
        names = [cc.name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        base_arr = np.asarray(base_xy)
        result = {
            "label": label, "task": task_id, "checkpoint": checkpoint, "scene": cc.SCENE,
            "simulation_only": True, "control_dt_s": control_dt,
            "frames_planned": frames, "frames_recorded": len(hulls),
            "duration_s": len(hulls) * control_dt,
            "terminated_early_at_frame": stopped,
            "max_footprint_radius_from_start_m": max(radius) if radius else None,
            "max_base_excursion_m": float(np.linalg.norm(base_arr - start_xy, axis=1).max())
            if len(base_arr) else None,
            "base_start_xy_m": start_xy.tolist(),
            "base_end_xy_m": base_arr[-1].tolist() if len(base_arr) else None,
            "yaw_start_deg": math.degrees(yaws[0]) if yaws else None,
            "yaw_end_deg": math.degrees(yaws[-1]) if yaws else None,
            "max_abs_lateral_drift_m": float(np.abs(base_arr[:, 1] - start_xy[1]).max())
            if len(base_arr) else None,
            "non_bar_contacts": sorted(others),
            "torque": cc.torque_summary(np.asarray(torque), control_dt, names),
        }
        if label.startswith("walk"):
            result["command"] = {"vx_m_s": WALK_SPEED_M_S if label == "walk_along_bar" else 0.0,
                                 "wz_rad_s": TURN_RATE_RAD_S if label == "walk_turn" else 0.0}
        swept = cc.hull(np.concatenate(hulls)) if hulls else np.zeros((0, 2))
        result["swept_hull_xy_m"] = swept.tolist()
        np.savez_compressed(out / f"{label}_hulls.npz",
                            hull_points=np.concatenate(hulls) if hulls else np.zeros((0, 2)),
                            hull_sizes=np.asarray([len(h) for h in hulls], dtype=np.int64),
                            base_xy=base_arr, yaw=np.asarray(yaws))
        cc.write_json(out / f"{label}.json", result)
        return result
    finally:
        env.close()


def main() -> int:
    args = _parser().parse_args()
    cc.headless()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = args.only or list(POLICIES)
    failures = 0
    for label in labels:
        result = rollout(label, out)
        ok = result["terminated_early_at_frame"] is None and not result["non_bar_contacts"]
        failures += not ok
        print(f"[cafe-fp] {label}: {result['duration_s']:.2f} s, footprint radius "
              f"{result['max_footprint_radius_from_start_m']:.4f} m, base excursion "
              f"{result['max_base_excursion_m']:.4f} m, terminated={result['terminated_early_at_frame']}, "
              f"non-bar contacts={len(result['non_bar_contacts'])}, peak torque "
              f"{result['torque']['peak_abs_nm']:.3f} N m -> {'OK' if ok else 'CHECK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
