#!/usr/bin/env python3
"""Spawn Jumper on the cafe bar, settle it 3 s, and render the cafe offscreen.

Simulation only (native:cpu, offscreen CGL/EGL). Criterion A1-1 and A1-5:

- Jumper is spawned standing at ``start_slot`` (facing -Y, the customer) with
  ``jumper.flat``'s home action and settles for 3 s. Standing means at the height
  it rests at on this bar, measured by a first drop-in pass from the task's own
  spawn height. (At ``pour_start_pose`` the robot is not in its home stance but in
  the accepted M4 start configuration; ``run_m4_transfer.py`` covers that pose.) Logged per control step: contact penetration,
  base slide from the spawn, roll/pitch, any robot contact with something other
  than the bar top, and actuator torque against the servo budget.
- From the settled start-slot state it renders the wide establishing shot, the
  customer point of view and the M4 hero angle as PNGs.

    python barista/cafe/settle_and_render.py --out out/cafe/settle
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cafe_common as cc
import numpy as np

SETTLE_S = 3.0
SLIDE_LIMIT_M = 0.005
TILT_LIMIT_DEG = 5.0
#: The soft-contact equilibrium of the accepted contact model (solref 0.008 s)
#: sits a fraction of a millimetre inside the surface; more than this is a
#: robot sinking into the bar, not resting on it.
PENETRATION_LIMIT_M = 0.002
WIDTH, HEIGHT = 1280, 720


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    return parser


def _roll_pitch_deg(rot: np.ndarray) -> tuple[float, float]:
    roll = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2, 0]))))
    return roll, pitch


def settle(slot: str, x: float, y: float, yaw_deg: float, out: Path):
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import scenes
    import tasks

    resolution = cc.backend()
    env_cfg = tasks.load_env_cfg("jumper.flat", play=True)
    env_cfg.seed = 0
    cc.pin_reset(env_cfg)
    scenes.apply(env_cfg, cc.SCENE)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    try:
        env.reset()
        robot = env.scene["robot"]
        model = env.sim.mj_model
        data = env.sim._datas[0]
        yaw = math.radians(yaw_deg)
        default_joints = robot.data.default_joint_pos.clone()
        actions = torch.zeros(env.action_space.shape, device=env.device)
        control_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation

        def place(z: float, joints) -> None:
            pose = torch.tensor([[x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                                dtype=torch.float, device=resolution.device)
            robot.write_root_link_pose_to_sim(pose)
            robot.write_root_link_velocity_to_sim(torch.zeros((1, 6), device=resolution.device))
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            env.sim.forward()
            env.scene.update(dt=0.0)

        # Pass 1 (not evidence): drop in from the task's own spawn height to find the
        # height at which Jumper stands on this bar in its home stance.
        drop_z = float(robot.data.root_link_pose_w[0, 2])
        place(drop_z, default_joints)
        for _ in range(round(1.0 / control_dt)):
            env.step(actions)
        rest_z = float(robot.data.root_link_pose_w[0, 2])
        rest_joints = robot.data.joint_pos.clone()
        # Pass 2 (evidence): spawn standing -- the stance and height it rests at under
        # its own weight, 0.5 mm above the bar -- at the slot, at rest.
        z = rest_z + 0.0005
        place(z, rest_joints)
        bar = cc.bar_geoms(model)
        base = cc.base_body(model)
        pitcher = next(b for b in range(model.nbody)
                       if cc.name(model, mujoco.mjtObj.mjOBJ_BODY, b).endswith("_pitcher"))
        steps = round(SETTLE_S / control_dt)
        spawn_xy = np.array([x, y])
        pitcher0 = data.xpos[pitcher].copy()
        rows = []
        torque = []
        other_contacts: set[tuple[str, str]] = set()
        for step in range(steps):
            env.step(actions)
            rot = data.xmat[base].reshape(3, 3)
            roll, pitch = _roll_pitch_deg(rot)
            penetration = 0.0
            for k in range(data.ncon):
                contact = data.contact[k]
                penetration = max(penetration, -float(contact.dist))
                g1, g2 = int(contact.geom1), int(contact.geom2)
                n1, n2 = cc.geom_name(model, g1), cc.geom_name(model, g2)
                for robot_name, other, other_name in ((n1, g2, n2), (n2, g1, n1)):
                    if robot_name.startswith("robot/") and not other_name.startswith("robot/") \
                            and other not in bar:
                        other_contacts.add((robot_name, other_name))
            yaw_now = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
            rows.append({
                "t_s": (step + 1) * control_dt,
                "base_x_m": float(data.xpos[base][0]), "base_y_m": float(data.xpos[base][1]),
                "base_z_m": float(data.xpos[base][2]),
                "slide_m": float(np.linalg.norm(data.xpos[base][:2] - spawn_xy)),
                "roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw_now,
                "penetration_m": penetration,
            })
            torque.append(robot.data.actuator_force[0].detach().cpu().numpy().copy())
        actuator_names = [cc.name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        torque_info = cc.torque_summary(np.asarray(torque), control_dt, actuator_names)
        maximum = {
            "max_slide_m": max(r["slide_m"] for r in rows),
            "max_abs_roll_deg": max(abs(r["roll_deg"]) for r in rows),
            "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in rows),
            "max_penetration_m": max(r["penetration_m"] for r in rows),
            "final_penetration_m": rows[-1]["penetration_m"],
            "yaw_drift_deg": abs(((rows[-1]["yaw_deg"] - yaw_deg + 180) % 360) - 180),
            "pitcher_displacement_m": float(np.linalg.norm(data.xpos[pitcher] - pitcher0)),
        }
        checks = {
            "no_slide_over_5mm": maximum["max_slide_m"] <= SLIDE_LIMIT_M,
            "no_tip_over_5deg": max(maximum["max_abs_roll_deg"],
                                    maximum["max_abs_pitch_deg"]) <= TILT_LIMIT_DEG,
            "no_penetration_over_2mm": maximum["max_penetration_m"] <= PENETRATION_LIMIT_M,
            "only_bar_top_contact": not other_contacts,
            "torque_within_budget": bool(torque_info["peak_within_curve"]
                                         and torque_info["peak_window_passes"]),
            "pitcher_undisturbed": maximum["pitcher_displacement_m"] <= 0.001,
        }
        summary = {
            "simulation_only": True, "backend": "native:cpu", "task": "jumper.flat (home action)",
            "scene": cc.SCENE, "slot": slot, "spawn": {"x_m": x, "y_m": y, "yaw_deg": yaw_deg,
                                                       "z_m": z, "task_drop_in_z_m": drop_z,
                                                       "measured_rest_z_m": rest_z},
            "settle_s": SETTLE_S, "control_dt_s": control_dt, "steps": steps,
            "metrics": maximum, "torque": torque_info,
            "non_bar_contacts": sorted(other_contacts), "checks": checks,
            "passes": all(checks.values()),
        }
        cc.write_json(out / f"settle_{slot}.json", {**summary, "rows": rows})
        qpos = data.qpos.copy()
        return summary, model, qpos
    finally:
        env.close()


def render(model, qpos: np.ndarray, out: Path) -> list[Path]:
    import mujoco
    from PIL import Image

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    paths = []
    try:
        for camera, label in (("cafe_wide", "wide_establishing"),
                              ("customer_pov", "customer_pov"), ("hero", "brew_station_hero")):
            cam = next(c for c in range(model.ncam)
                       if cc.name(model, mujoco.mjtObj.mjOBJ_CAMERA, c).endswith(camera))
            renderer.update_scene(data, camera=cam)
            path = out / f"cafe_{label}.png"
            Image.fromarray(renderer.render()).save(path)
            paths.append(path)
    finally:
        renderer.close()
    return paths


def main() -> int:
    args = _parser().parse_args()
    cc.headless()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = cc.dims()
    start = d["route"]["start_slot"]
    results = {}
    start_summary, model, qpos = settle("start_slot", start["x_m"], start["y_m"],
                                        start["yaw_deg"], out)
    results["start_slot"] = start_summary
    renders = [] if args.no_render else render(model, qpos, out)
    cc.write_json(out / "settle_summary.json", {
        "results": results, "renders": [str(p) for p in renders],
        "passes": all(r["passes"] for r in results.values()),
    })
    for slot, summary in results.items():
        m = summary["metrics"]
        print(f"[cafe] {slot}: slide {m['max_slide_m'] * 1000:.2f} mm, "
              f"roll {m['max_abs_roll_deg']:.2f} deg, pitch {m['max_abs_pitch_deg']:.2f} deg, "
              f"penetration {m['max_penetration_m'] * 1000:.3f} mm, "
              f"peak torque {summary['torque']['peak_abs_nm']:.3f} N m, "
              f"non-bar contacts {len(summary['non_bar_contacts'])} -> "
              f"{'PASS' if summary['passes'] else 'FAIL'}")
    for path in renders:
        print(f"[cafe] render: {path}")
    return 0 if all(r["passes"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
