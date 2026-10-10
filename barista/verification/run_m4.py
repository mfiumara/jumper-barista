#!/usr/bin/env python3
"""Replay and verify the simulation-only M4 spiral pour and stable set-down."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = REPO / "barista" / "trajectory" / "m4_spiral_pour.json"
DIMENSIONS = REPO / "barista" / "coffee_scene_dimensions.json"
WIDTH, HEIGHT = 960, 540
CONTROL_DECIMATION = 4
FAST_CHECK_END_PHASE = "pour_out"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=TRAJECTORY,
        help="trajectory JSON (defaults to the canonical M4 trajectory)",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--fast-check",
        action="store_true",
        help=(
            "run a non-rendered diagnostic bounded at the end of pour_out; "
            "this is not release evidence and does not replace the default verifier"
        ),
    )
    return parser


def _ending(model, mujoco, kind, count_name: str, suffix: str) -> int:
    matches = []
    for index in range(getattr(model, count_name)):
        name = mujoco.mj_id2name(model, kind, index) or ""
        if name == suffix or name.endswith((f"/{suffix}", f"_{suffix}")):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r}; found {matches}")
    return matches[0]


def _name(model, mujoco, kind, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def _smooth(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _target(config: dict, time_s: float) -> tuple[str, np.ndarray, float]:
    keyframes = config["keyframes"]
    previous_time = 0.0
    previous_arm = np.asarray(keyframes[0]["arm_rad"], dtype=np.float64)
    previous_finger = float(keyframes[0]["finger_rad"])
    for keyframe in keyframes:
        end = float(keyframe["end_s"])
        arm = np.asarray(keyframe["arm_rad"], dtype=np.float64)
        finger = float(keyframe["finger_rad"])
        if time_s <= end:
            fraction = _smooth((time_s - previous_time) / max(end - previous_time, 1e-12))
            return (
                keyframe["phase"],
                previous_arm + (arm - previous_arm) * fraction,
                previous_finger + (finger - previous_finger) * fraction,
            )
        previous_time, previous_arm, previous_finger = end, arm, finger
    return keyframes[-1]["phase"], previous_arm, previous_finger


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _roll_pitch_deg(relative: np.ndarray) -> tuple[float, float]:
    roll = math.atan2(float(relative[2, 1]), float(relative[2, 2]))
    pitch = math.asin(float(np.clip(-relative[2, 0], -1.0, 1.0)))
    return math.degrees(roll), math.degrees(pitch)


def _longest_true(values: list[bool], dt_s: float) -> float:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * dt_s


def _render(
    model,
    mujoco,
    states: np.ndarray,
    stream_landings: list[np.ndarray | None],
    spout_site: int,
    bed_z: float,
    camera: int,
    fps: float,
    path: Path,
) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), size=(WIDTH, HEIGHT), fps=fps, macro_block_size=1,
        codec="libx264", pix_fmt_out="yuv420p", quality=7,
    )
    writer.send(None)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    data = mujoco.MjData(model)
    try:
        for state, landing in zip(states, stream_landings, strict=True):
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if landing is not None:
                start = data.site_xpos[spout_site].copy()
                flight_s = math.sqrt(max(2.0 * (start[2] - bed_z) / 9.81, 1e-9))
                velocity = (landing - start) / flight_s
                velocity[2] = 0.0
                points = []
                for fraction in np.linspace(0.0, 1.0, 13):
                    time_s = fraction * flight_s
                    point = start + velocity * time_s
                    point[2] -= 0.5 * 9.81 * time_s * time_s
                    points.append(point)
                for first, second in itertools.pairwise(points):
                    renderer.scene.ngeom += 1
                    geom = renderer.scene.geoms[renderer.scene.ngeom - 1]
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    mujoco.mjv_initGeom(
                        geom=geom,
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
                        size=np.zeros(3),
                        pos=np.zeros(3),
                        mat=np.zeros(9),
                        rgba=np.asarray((0.18, 0.58, 0.82, 0.82), dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom=geom,
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
                        width=0.0017,
                        from_=first,
                        to=second,
                    )
            writer.send(renderer.render().tobytes())
    finally:
        renderer.close()
        writer.close()


def _payload(model, mujoco, pitcher_body: int) -> dict:
    source = json.loads(DIMENSIONS.read_text("utf-8"))
    dry = float(source["objects"]["pitcher"]["dry_mass_kg"])
    addon = source["objects"]["pitcher_top_grip_addon"]
    water = float(source["objects"]["nominal_water"]["mass_kg"])
    total = dry + water
    compiled = float(model.body_subtreemass[pitcher_body])
    limit = float(source["payload_check"]["design_limit_kg"])
    return {
        "pitcher_dry_kg": dry,
        "grip_addon_kg": float(addon["mass_kg"]),
        "represented_water_kg": water,
        "design_total_kg": total,
        "compiled_total_kg": compiled,
        "design_limit_kg": limit,
        "margin_kg": limit - total,
        "compiled_loaded_com_in_pitcher_m": model.body_ipos[pitcher_body].tolist(),
        "grip_bar_centre_in_pitcher_m": addon["bar_centre_m"],
        "grip_bar_centre_xy_offset_from_com_m": float(
            np.linalg.norm(
                np.asarray(addon["bar_centre_m"][:2]) - model.body_ipos[pitcher_body, :2]
            )
        ),
        "passes": total <= limit and abs(total - compiled) <= 1e-9,
    }


def _summarize_torque(rows: list[dict], dt_s: float) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["joint"], "whole_run")].append(row)
        grouped[(row["joint"], row["phase"])].append(row)

    out: dict[str, dict] = {}
    for (joint, phase), samples in grouped.items():
        key = f"{joint}:{phase}"
        above = [sample["above_continuous"] for sample in samples]
        violations = [sample for sample in samples if sample["curve_violation"]]
        thermal_error = max(sample["thermal_recompute_error"] for sample in samples)
        out[key] = {
            "joint": joint,
            "phase": phase,
            "sample_count": len(samples),
            "peak_speed_rad_s": max(abs(sample["qd_pre_rad_s"]) for sample in samples),
            "peak_raw_abs_nm": max(abs(sample["tau_raw_nm"]) for sample in samples),
            "peak_applied_abs_nm": max(abs(sample["tau_applied_nm"]) for sample in samples),
            "minimum_curve_margin_nm": min(sample["curve_margin_nm"] for sample in samples),
            "minimum_effective_margin_nm": min(sample["effective_margin_nm"] for sample in samples),
            "maximum_thermal_theta": max(
                max(sample["thermal_theta_pre"], sample["thermal_theta_post"])
                for sample in samples
            ),
            "above_continuous_count": sum(above),
            "above_continuous_total_s": sum(above) * dt_s,
            "above_continuous_max_contiguous_s": _longest_true(above, dt_s),
            "derated_substep_count": sum(sample["derating_active"] for sample in samples),
            "first_violation_time_s": violations[0]["time_s"] if violations else None,
            "maximum_thermal_recompute_error": thermal_error,
        }
    return out


def _json_default(value):
    """Turn NumPy scalar reductions into ordinary JSON scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> int:
    args = _parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

    import imageio_ffmpeg
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import scenes
    import tasks
    from tasks.jumper.barista.audited_servo import AuditedServoCurveActuator
    from tasks.jumper.common.actuator import CURVE, ServoCurveActuator
    from tasks.jumper.common.constants import HOME
    from tasks.jumper.five_foot.claw import GRIPPER_OPEN, LEAD_PER_SPEED, SQUEEZE_LEAD
    from tasks.jumper.five_foot.objects import PROP_CONE, PROP_IMPRATIO

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    trajectory = args.trajectory.resolve()
    config = json.loads(trajectory.read_text("utf-8"))
    seed = int(config["seed"])
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)
    print(resolution.banner())

    env_cfg = tasks.load_env_cfg("jumper.barista", play=True)
    env_cfg.seed = seed
    scenes.apply(env_cfg, "coffee_bar")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    env.reset()
    model = env.sim.mj_model
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = PROP_IMPRATIO
    if PROP_CONE != "elliptic":
        raise RuntimeError(f"expected the measured grasp cone to be elliptic, got {PROP_CONE}")
    robot = env.scene["robot"]
    if len(robot.actuators) != 1 or not isinstance(
        robot.actuators[0], AuditedServoCurveActuator
    ):
        raise RuntimeError("jumper.barista did not construct its audited runtime actuator")
    servo = robot.actuators[0]

    joint_ids = {
        name: _ending(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, "njnt", name)
        for name in HOME
    }
    base_start = config["base_start"]
    yaw = float(base_start["yaw_rad"])
    root_pose = torch.tensor(
        [[*base_start["position_m"], math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
        dtype=torch.float,
        device=resolution.device,
    )
    runtime_joint_col = {name: index for index, name in enumerate(robot.joint_names)}
    if set(runtime_joint_col) != set(HOME):
        raise RuntimeError("barista runtime joint set differs from the 22-joint hardware order")
    initial = np.asarray([HOME[name] for name in robot.joint_names], dtype=np.float64)
    for name, value in config["support_base_rad"].items():
        initial[runtime_joint_col[name]] = float(value)
    arm_names = config["arm_joint_order"]
    for name, value in zip(arm_names, config["keyframes"][0]["arm_rad"], strict=True):
        initial[runtime_joint_col[name]] = value
    initial[runtime_joint_col["LF_J4_joint"]] = GRIPPER_OPEN
    initial_t = torch.tensor(initial[None, :], dtype=torch.float, device=resolution.device)
    robot.write_root_link_pose_to_sim(root_pose)
    robot.write_joint_state_to_sim(initial_t, torch.zeros_like(initial_t))
    robot.set_joint_position_target(initial_t)
    env.sim.forward()
    env.scene.update(dt=0.0)
    # Native refresh restores environment 0 into this MjData after every step. It
    # is the registered runtime's state, not a separate probe data object.
    data = env.sim._datas[0]

    joint_order = list(HOME)
    joint_col = {name: index for index, name in enumerate(joint_order)}
    target = np.asarray([HOME[name] for name in joint_order], dtype=np.float64)
    finger_col = joint_col["LF_J4_joint"]
    audit_col = {name: index for index, name in enumerate(servo.target_names)}
    if set(audit_col) != set(joint_order):
        raise RuntimeError("audited actuator does not cover all 22 hardware joints")
    servo.enable_audit(f"m4-seed-{seed}")

    pitcher_body = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "nbody", "pitcher")
    base_body = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "nbody", "base_link")
    palm_body = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "nbody", "LF_palm_link")
    spout_site = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "nsite", "pitcher_spout_tip")
    bed_site = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "nsite", "coffee_bed_target")
    pitcher_joint = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, "njnt", "pitcher_free")
    pitcher_dof = int(model.jnt_dofadr[pitcher_joint])
    camera = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_CAMERA, "ncam", "hero")
    terrain = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, "ngeom", "terrain")
    brewer_geoms = {
        index for index in range(model.ngeom)
        if any(token in _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index)
               for token in ("server_", "v60_"))
    }
    pitcher_geoms = {
        index for index in range(model.ngeom) if model.geom_bodyid[index] == pitcher_body
    }
    support_suffixes = (
        "RF_palm_pad_b_link_meshcol", "LM_foot_tip_link_meshcol",
        "RM_foot_tip_link_meshcol", "LR_foot_tip_link_meshcol", "RR_foot_tip_link_meshcol",
    )
    support_geoms = {
        _ending(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, "ngeom", suffix)
        for suffix in support_suffixes
    }

    nominal_dt_s = float(config["physics_dt_s"])
    dt_s = float(model.opt.timestep)
    if abs(dt_s - nominal_dt_s) > 1e-12:
        raise RuntimeError(f"compiled dt {dt_s} differs from trajectory dt")
    control_decimation = CONTROL_DECIMATION
    control_dt_s = dt_s * control_decimation
    if abs(control_dt_s - float(config["control_dt_s"])) > 1e-12:
        raise RuntimeError(f"control dt {control_dt_s} differs from trajectory dt")
    full_duration_s = float(config["keyframes"][-1]["end_s"])
    duration_s = (
        next(
            float(frame["end_s"])
            for frame in config["keyframes"]
            if frame["phase"] == FAST_CHECK_END_PHASE
        )
        if args.fast_check else full_duration_s
    )
    steps = round(duration_s / dt_s)
    render_stride = round(1.0 / (float(config["render_fps"]) * dt_s))
    if render_stride * dt_s * float(config["render_fps"]) != 1.0:
        raise RuntimeError("render rate must divide the physics rate exactly")

    torque_rows: list[dict] = []
    motion_rows: list[dict] = []
    stream_rows: list[dict] = []
    render_states: list[np.ndarray] = []
    render_stream_landings: list[np.ndarray | None] = []
    above_window = np.zeros(len(joint_order), dtype=np.int64)
    above_elapsed = np.zeros(len(joint_order), dtype=np.float64)
    next_window = np.zeros(len(joint_order), dtype=np.int64)
    contact_pairs_seen: set[tuple[str, str]] = set()
    settled_pitcher_z = None
    settled_pitcher_rot = None
    settled_base_xy = None
    settled_base_rot = None
    grasp_reference = None
    carried_reference_set = False
    config_sha256 = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    stream_cfg = config["stream"]
    bed_centre = np.asarray(stream_cfg["bed_centre_m"], dtype=np.float64)
    compiled_bed = data.site_xpos[bed_site].copy()
    if np.linalg.norm(compiled_bed - bed_centre) > 1e-9:
        raise RuntimeError(f"compiled bed centre {compiled_bed} differs from trajectory")
    set_down_contact_seen = False
    set_down_contact_time_s = None
    set_down_arm = None

    try:
        for step in range(steps):
            time_s = step * dt_s
            phase, _, _ = _target(config, time_s)
            if step % control_decimation == 0:
                _, arm, finger_request = _target(config, time_s)
                if set_down_contact_seen and phase in ("lower", "set_down", "release"):
                    assert set_down_arm is not None
                    arm = set_down_arm
                    finger_request = -1.4
                target[:] = [HOME[name] for name in joint_order]
                base_fraction = _smooth((time_s - 4.0) / 2.0)
                support_base = dict(HOME)
                for name, value in config["support_base_rad"].items():
                    support_base[name] = (
                        HOME[name] + (float(value) - HOME[name]) * base_fraction
                    )
                profile = config["support_pour_profile_s"]
                if time_s < float(profile["raise_start"]):
                    support_fraction = 0.0
                elif time_s < float(profile["raise_end"]):
                    support_fraction = _smooth(
                        (time_s - float(profile["raise_start"]))
                        / (float(profile["raise_end"]) - float(profile["raise_start"]))
                    )
                elif time_s < float(profile["lower_start"]):
                    support_fraction = 1.0
                elif time_s < float(profile["lower_end"]):
                    support_fraction = 1.0 - _smooth(
                        (time_s - float(profile["lower_start"]))
                        / (float(profile["lower_end"]) - float(profile["lower_start"]))
                    )
                else:
                    support_fraction = 0.0
                support_fraction *= float(profile["scale"])
                for name, lifted in config["support_pour_rad"].items():
                    target[joint_col[name]] = (
                        support_base[name]
                        + (float(lifted) - support_base[name]) * support_fraction
                    )
                for name, value in zip(arm_names, arm, strict=True):
                    target[joint_col[name]] = value
                finger_runtime_col = runtime_joint_col["LF_J4_joint"]
                finger_q = float(robot.data.joint_pos[0, finger_runtime_col])
                finger_qd = float(robot.data.joint_vel[0, finger_runtime_col])
                if finger_request > finger_q:
                    target[finger_col] = min(
                        finger_request,
                        finger_q + SQUEEZE_LEAD + LEAD_PER_SPEED * max(finger_qd, 0.0),
                    )
                else:
                    target[finger_col] = finger_request

            runtime_target = torch.tensor(
                [[target[joint_col[name]] for name in robot.joint_names]],
                dtype=torch.float,
                device=resolution.device,
            )
            robot.set_joint_position_target(runtime_target)
            servo.set_audit_context(
                phase=phase,
                substep=step,
                control_step=step // control_decimation,
                time_s=time_s,
            )
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=dt_s)
            batches = servo.drain_audit()
            if len(batches) != 1:
                raise RuntimeError(f"expected one actuator audit batch, got {len(batches)}")
            batch = batches[0]
            arrays = {
                name: getattr(batch, name).detach().cpu().numpy()[0]
                for name in (
                    "q_pre_rad",
                    "qd_pre_rad_s",
                    "tau_raw_nm",
                    "tau_applied_nm",
                    "curve_limit_nm",
                    "effective_limit_nm",
                    "thermal_theta_pre",
                    "thermal_theta_post",
                )
            }
            for index, joint in enumerate(joint_order):
                column = audit_col[joint]
                tau_applied = float(arrays["tau_applied_nm"][column])
                thermal_pre = float(arrays["thermal_theta_pre"][column])
                thermal_post = float(arrays["thermal_theta_post"][column])
                curve_limit = float(arrays["curve_limit_nm"][column])
                effective_limit = float(arrays["effective_limit_nm"][column])
                magnitude = abs(tau_applied)
                above = magnitude > CURVE.continuous_torque
                if above and above_elapsed[index] == 0.0:
                    next_window[index] += 1
                    above_window[index] = next_window[index]
                if above:
                    above_elapsed[index] += dt_s
                else:
                    above_window[index] = 0
                    above_elapsed[index] = 0.0
                thermal_recomputed = thermal_pre + (
                    (tau_applied / CURVE.continuous_torque) ** 2
                    - thermal_pre
                ) * (dt_s / CURVE.thermal_time_constant)
                curve_margin = curve_limit - magnitude
                effective_margin = effective_limit - magnitude
                torque_rows.append({
                    "run_id": batch.run_id, "phase": batch.phase, "joint": joint,
                    "substep": batch.substep, "control_step": batch.control_step,
                    "time_s": batch.time_s, "dt_s": batch.dt_s,
                    "q_pre_rad": float(arrays["q_pre_rad"][column]),
                    "qd_pre_rad_s": float(arrays["qd_pre_rad_s"][column]),
                    "tau_raw_nm": float(arrays["tau_raw_nm"][column]),
                    "tau_applied_nm": tau_applied,
                    "curve_limit_nm": curve_limit,
                    "continuous_limit_nm": CURVE.continuous_torque,
                    "effective_limit_nm": effective_limit,
                    "thermal_theta_pre": thermal_pre,
                    "thermal_theta_post": thermal_post,
                    "curve_margin_nm": curve_margin,
                    "effective_margin_nm": effective_margin,
                    "above_continuous": above,
                    "derating_active": thermal_pre >= 1.0,
                    "curve_violation": curve_margin < -1e-4 or effective_margin < -1e-4,
                    "above_continuous_window_id": int(above_window[index]),
                    "above_continuous_elapsed_s": float(above_elapsed[index]),
                    "thermal_recompute_error": abs(thermal_recomputed - thermal_post),
                })

            pitcher_pos = data.xpos[pitcher_body].copy()
            pitcher_rot = data.xmat[pitcher_body].reshape(3, 3).copy()
            spout_pos = data.site_xpos[spout_site].copy()
            spout_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_SITE, spout_site, spout_velocity, 0
            )
            spout_linear_velocity = spout_velocity[3:].copy()
            pitcher_linear_speed = float(np.linalg.norm(data.qvel[pitcher_dof : pitcher_dof + 3]))
            pitcher_angular_speed = float(
                np.linalg.norm(data.qvel[pitcher_dof + 3 : pitcher_dof + 6])
            )
            base_pos = data.xpos[base_body].copy()
            base_rot = data.xmat[base_body].reshape(3, 3).copy()
            palm_pos = data.xpos[palm_body].copy()
            palm_rot = data.xmat[palm_body].reshape(3, 3).copy()
            if settled_pitcher_z is None and time_s + dt_s >= 1.0:
                settled_pitcher_z = float(pitcher_pos[2])
                settled_pitcher_rot = pitcher_rot.copy()
                settled_base_xy = base_pos[:2].copy()
                settled_base_rot = base_rot.copy()

            fixed_contact = moving_contact = counter_contact = brewer_contact = False
            robot_brewer_contact = False
            non_claw_robot_contact = self_contact = False
            fixed_contact_positions: list[np.ndarray] = []
            moving_contact_positions: list[np.ndarray] = []
            support_touching: set[int] = set()
            contact_names: list[str] = []
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                name1 = _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                name2 = _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                pair = tuple(sorted((name1, name2)))
                contact_pairs_seen.add(pair)
                if geom1 in brewer_geoms and name2.startswith("robot/"):
                    robot_brewer_contact = True
                if geom2 in brewer_geoms and name1.startswith("robot/"):
                    robot_brewer_contact = True
                if name1.startswith("robot/") and name2.startswith("robot/"):
                    self_contact = True
                if terrain in (geom1, geom2):
                    support_touching.update({geom for geom in (geom1, geom2) if geom in support_geoms})
                if not pitcher_geoms.intersection((geom1, geom2)):
                    continue
                other = geom2 if geom1 in pitcher_geoms else geom1
                other_name = _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, other)
                contact_names.append(other_name)
                fixed_contact |= "robot/LF_palm" in other_name
                moving_contact |= "robot/LF_finger" in other_name
                contact_in_pitcher = pitcher_rot.T @ (
                    np.asarray(contact.pos, dtype=np.float64) - pitcher_pos
                )
                if "robot/LF_palm" in other_name:
                    fixed_contact_positions.append(contact_in_pitcher)
                if "robot/LF_finger" in other_name:
                    moving_contact_positions.append(contact_in_pitcher)
                counter_contact |= other == terrain
                brewer_contact |= other in brewer_geoms
                if other_name.startswith("robot/") and not any(
                    token in other_name for token in ("robot/LF_palm", "robot/LF_finger")
                ):
                    non_claw_robot_contact = True

            if (
                counter_contact and phase in ("lower", "set_down")
                and not set_down_contact_seen
            ):
                set_down_contact_seen = True
                set_down_contact_time_s = time_s + dt_s
                set_down_arm = np.asarray([
                    float(robot.data.joint_pos[0, runtime_joint_col[name]])
                    for name in arm_names
                ])

            clearance = math.inf
            fromto = np.zeros(6, dtype=np.float64)
            for pitcher_geom in pitcher_geoms:
                for brewer_geom in brewer_geoms:
                    clearance = min(
                        clearance,
                        float(mujoco.mj_geomDistance(
                            model, data, pitcher_geom, brewer_geom, 1.0, fromto
                        )),
                    )

            if settled_pitcher_rot is None:
                tilt = base_roll = base_pitch = base_drift = lift = 0.0
            else:
                upright_cosine = float(np.clip(
                    np.dot(settled_pitcher_rot[:, 2], pitcher_rot[:, 2]), -1.0, 1.0
                ))
                tilt = math.degrees(math.acos(upright_cosine))
                base_roll, base_pitch = _roll_pitch_deg(settled_base_rot.T @ base_rot)
                base_drift = float(np.linalg.norm(base_pos[:2] - settled_base_xy))
                lift = float(pitcher_pos[2] - settled_pitcher_z)
            local_pos = palm_rot.T @ (pitcher_pos - palm_pos)
            local_rot = palm_rot.T @ pitcher_rot
            if phase == "lift_hold" and grasp_reference is None:
                grasp_reference = (local_pos.copy(), local_rot.copy())
            if phase == "transit" and not carried_reference_set:
                grasp_reference = (local_pos.copy(), local_rot.copy())
                carried_reference_set = True
            if grasp_reference is None:
                grasp_translation = grasp_rotation = 0.0
            else:
                grasp_translation = float(np.linalg.norm(local_pos - grasp_reference[0]))
                grasp_rotation = _rotation_angle_deg(grasp_reference[1].T @ local_rot)

            joint_limit_violation = any(
                data.qpos[model.jnt_qposadr[joint_ids[name]]] < model.jnt_range[joint_ids[name], 0] - 1e-6
                or data.qpos[model.jnt_qposadr[joint_ids[name]]] > model.jnt_range[joint_ids[name], 1] + 1e-6
                for name in joint_order
            )
            motion_rows.append({
                "substep": step, "time_s": time_s + dt_s, "phase": phase,
                "pitcher_x_m": float(pitcher_pos[0]), "pitcher_y_m": float(pitcher_pos[1]),
                "pitcher_z_m": float(pitcher_pos[2]), "lift_m": lift,
                "spout_x_m": float(spout_pos[0]), "spout_y_m": float(spout_pos[1]),
                "spout_z_m": float(spout_pos[2]),
                "spout_vx_m_s": float(spout_linear_velocity[0]),
                "spout_vy_m_s": float(spout_linear_velocity[1]),
                "spout_vz_m_s": float(spout_linear_velocity[2]),
                "spout_height_above_bed_m": float(spout_pos[2] - bed_centre[2]),
                "pitcher_xx": float(pitcher_rot[0, 0]), "pitcher_xy": float(pitcher_rot[0, 1]),
                "pitcher_xz": float(pitcher_rot[0, 2]), "pitcher_yx": float(pitcher_rot[1, 0]),
                "pitcher_yy": float(pitcher_rot[1, 1]), "pitcher_yz": float(pitcher_rot[1, 2]),
                "pitcher_zx": float(pitcher_rot[2, 0]), "pitcher_zy": float(pitcher_rot[2, 1]),
                "pitcher_zz": float(pitcher_rot[2, 2]),
                "pitcher_linear_speed_m_s": pitcher_linear_speed,
                "pitcher_angular_speed_rad_s": pitcher_angular_speed,
                "palm_x_m": float(palm_pos[0]), "palm_y_m": float(palm_pos[1]),
                "palm_z_m": float(palm_pos[2]),
                "palm_xx": float(palm_rot[0, 0]), "palm_xy": float(palm_rot[0, 1]),
                "palm_xz": float(palm_rot[0, 2]), "palm_yx": float(palm_rot[1, 0]),
                "palm_yy": float(palm_rot[1, 1]), "palm_yz": float(palm_rot[1, 2]),
                "palm_zx": float(palm_rot[2, 0]), "palm_zy": float(palm_rot[2, 1]),
                "palm_zz": float(palm_rot[2, 2]),
                "pitcher_tilt_deg": tilt,
                "grasp_translation_m": grasp_translation,
                "grasp_rotation_deg": grasp_rotation,
                "base_x_m": float(base_pos[0]), "base_y_m": float(base_pos[1]),
                "base_z_m": float(base_pos[2]), "base_xy_drift_m": base_drift,
                "base_roll_deg": base_roll, "base_pitch_deg": base_pitch,
                "support_foot_contacts": len(support_touching),
                "brewer_clearance_m": clearance,
                "fixed_jaw_contact": fixed_contact,
                "moving_jaw_contact": moving_contact,
                "fixed_jaw_contact_centre_pitcher_m": (
                    ";".join(f"{value:.9f}" for value in np.mean(fixed_contact_positions, axis=0))
                    if fixed_contact_positions else ""
                ),
                "moving_jaw_contact_centre_pitcher_m": (
                    ";".join(f"{value:.9f}" for value in np.mean(moving_contact_positions, axis=0))
                    if moving_contact_positions else ""
                ),
                "counter_contact": counter_contact, "brewer_contact": brewer_contact,
                "robot_brewer_contact": robot_brewer_contact,
                "non_claw_robot_contact": non_claw_robot_contact,
                "self_contact": self_contact, "joint_limit_violation": joint_limit_violation,
                "pitcher_contact_geoms": ";".join(sorted(set(contact_names))),
            })
            landing = None
            if phase == "spiral" and step % control_decimation == 0:
                # The massless visual stream inherits the measured outlet velocity and then
                # follows a gravity-only ballistic path to the bed. Both its origin and landing
                # are therefore driven by the actual spout, never a prescribed target spiral.
                height = float(spout_pos[2] - bed_centre[2])
                intersects = height > 0.0
                flight_s = (
                    (
                        float(spout_linear_velocity[2])
                        + math.sqrt(
                            float(spout_linear_velocity[2]) ** 2
                            + 2.0 * float(stream_cfg["gravity_m_s2"]) * height
                        )
                    ) / float(stream_cfg["gravity_m_s2"])
                    if intersects else math.nan
                )
                landing = bed_centre.copy()
                landing[:2] = spout_pos[:2] + spout_linear_velocity[:2] * flight_s
                velocity_xy = spout_linear_velocity[:2] if intersects else np.full(2, math.nan)
                spout_radius = float(np.linalg.norm(spout_pos[:2] - bed_centre[:2]))
                landing_radius = float(np.linalg.norm(landing[:2] - bed_centre[:2]))
                horizontal_span = float(np.linalg.norm(landing[:2] - spout_pos[:2]))
                horizontal_launch_speed = float(np.linalg.norm(velocity_xy))
                angle = math.atan2(
                    spout_pos[1] - bed_centre[1], spout_pos[0] - bed_centre[0]
                )
                accepted = (
                    intersects
                    and landing_radius <= float(stream_cfg["bed_acceptance_radius_m"])
                )
                stream_rows.append({
                    "sample": len(stream_rows), "time_s": time_s + dt_s,
                    "stream_dt_s": float(config["stream_dt_s"]), "phase": phase,
                    "emitted": True, "intersection_valid": intersects,
                    "spout_x_m": float(spout_pos[0]), "spout_y_m": float(spout_pos[1]),
                    "spout_z_m": float(spout_pos[2]), "bed_z_m": float(bed_centre[2]),
                    "spout_height_m": height, "flight_time_s": flight_s,
                    "spout_radius_m": spout_radius,
                    "horizontal_span_m": horizontal_span,
                    "launch_vx_m_s": float(velocity_xy[0]),
                    "launch_vy_m_s": float(velocity_xy[1]),
                    "launch_vz_m_s": float(spout_linear_velocity[2]),
                    "horizontal_launch_speed_m_s": horizontal_launch_speed,
                    "landing_x_m": float(landing[0]), "landing_y_m": float(landing[1]),
                    "landing_z_m": float(landing[2]), "landing_radius_m": landing_radius,
                    "unwrapped_angle_rad": angle, "angle_rad": angle,
                    "accepted": accepted,
                    "miss_reason": "" if accepted else "outside acceptance disk",
                })
            if step % render_stride == 0:
                render_states.append(data.qpos.copy())
                render_stream_landings.append(None if landing is None else landing.copy())
    finally:
        env.close()

    if settled_pitcher_z is None:
        raise RuntimeError("trajectory ended before the settled reference sample")

    torque_csv = out / "torque_substeps.csv"
    motion_csv = out / "motion_substeps.csv"
    stream_csv = out / "stream_samples.csv"
    for path, rows in (
        (torque_csv, torque_rows), (motion_csv, motion_rows), (stream_csv, stream_rows)
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    torque_summary = _summarize_torque(torque_rows, dt_s)
    whole_joint = [value for value in torque_summary.values() if value["phase"] == "whole_run"]
    torque_pass = all(
        value["minimum_curve_margin_nm"] >= -1e-4
        and value["minimum_effective_margin_nm"] >= -1e-4
        and value["above_continuous_max_contiguous_s"] <= CURVE.peak_duration + 1e-12
        and value["maximum_thermal_theta"] < 1.0
        and value["maximum_thermal_recompute_error"] <= 1e-5
        for value in whole_joint
    )
    threshold = config["thresholds"]
    lift_flags = [row["lift_m"] >= float(threshold["lift_m"]) for row in motion_rows]
    continuous_lift_duration = _longest_true(lift_flags, dt_s)
    hold_rows = [row for row in motion_rows if row["phase"] == "lift_hold"]
    pour_rows = [row for row in motion_rows if row["phase"] == "spiral"]
    carried_phases = {"transit", "pour_in", "spiral", "pour_out", "return"}
    carried_rows = [row for row in motion_rows if row["phase"] in carried_phases]
    final_rows = [row for row in motion_rows if row["phase"] == "final_settle"]
    retreat_rows = [
        row for row in motion_rows if row["phase"] in ("clear_retreat", "retreat")
    ]

    hold_duration = len(hold_rows) * dt_s
    valid_hold_duration = _longest_true(
        [row["lift_m"] >= float(threshold["lift_m"]) for row in hold_rows], dt_s
    )
    pre_lift_rows = [row for row in motion_rows if row["phase"] in ("approach", "close")]
    fixed_before_lift = any(row["fixed_jaw_contact"] for row in pre_lift_rows)
    moving_before_lift = any(row["moving_jaw_contact"] for row in pre_lift_rows)

    raw_angles = np.asarray([row["angle_rad"] for row in stream_rows])
    wrapped_delta = (np.diff(raw_angles) + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped = np.r_[0.0, np.cumsum(wrapped_delta)]
    radii = np.asarray([row["spout_radius_m"] for row in stream_rows])
    landing_radii = np.asarray([row["landing_radius_m"] for row in stream_rows])
    if float(np.ptp(unwrapped)) <= 1e-12:
        # A failed grasp can leave the actual spout stationary. That is a diagnostic
        # gate failure, not a reason for the diagnostic verifier itself to crash.
        slope, intercept = 0.0, float(np.mean(radii))
    else:
        slope, intercept = np.polyfit(unwrapped, radii, 1)
    fit = intercept + slope * unwrapped
    radial_rmse = float(np.sqrt(np.mean((radii - fit) ** 2)))
    radial_progress = float(slope * (unwrapped[-1] - unwrapped[0]))
    inward_excursion = float(np.max(np.maximum.accumulate(radii) - radii))
    angle_advance = float(unwrapped[-1] - unwrapped[0])
    turns = angle_advance / (2.0 * math.pi)
    misses = [
        {key: row[key] for key in ("sample", "time_s", "miss_reason")}
        for row in stream_rows if not row["accepted"]
    ]
    landing_success_rate = (len(stream_rows) - len(misses)) / len(stream_rows)
    edge_margin = float(stream_cfg["bed_acceptance_radius_m"]) - float(np.max(radii))
    spout_min = min(row["spout_height_above_bed_m"] for row in pour_rows)
    spout_max = max(row["spout_height_above_bed_m"] for row in pour_rows)
    pour_in_rows = [row for row in motion_rows if row["phase"] == "pour_in"]
    pour_out_rows = [row for row in motion_rows if row["phase"] == "pour_out"]
    maximum_pour_tilt = max(row["pitcher_tilt_deg"] for row in pour_rows)
    final_pour_out_tilt = pour_out_rows[-1]["pitcher_tilt_deg"]
    maximum_horizontal_span = max(row["horizontal_span_m"] for row in stream_rows)
    maximum_horizontal_launch_speed = max(
        row["horizontal_launch_speed_m_s"] for row in stream_rows
    )
    maximum_landing_radius = float(np.max(landing_radii))

    final_target = np.asarray(config["set_down_target_m"], dtype=np.float64)
    final_positions = np.asarray([
        [row["pitcher_x_m"], row["pitcher_y_m"], row["pitcher_z_m"]] for row in final_rows
    ])
    final_position_error = (
        float(np.max(np.linalg.norm(final_positions - final_target, axis=1)))
        if final_rows else math.nan
    )
    if retreat_rows:
        retreat_start = np.asarray([
            retreat_rows[0]["pitcher_x_m"], retreat_rows[0]["pitcher_y_m"],
            retreat_rows[0]["pitcher_z_m"],
        ])
        retreat_motion = max(
            float(np.linalg.norm(np.asarray([
                row["pitcher_x_m"], row["pitcher_y_m"], row["pitcher_z_m"],
            ]) - retreat_start))
            for row in retreat_rows
        )
    else:
        retreat_motion = math.nan

    motion_passes = {
        "m3_lift_and_hold": (
            hold_duration >= float(threshold["m3_hold_s"])
            and valid_hold_duration >= float(threshold["m3_hold_s"])
            and max(row["grasp_translation_m"] for row in hold_rows)
            <= threshold["grasp_translation_m"]
            and max(row["grasp_rotation_deg"] for row in hold_rows)
            <= threshold["grasp_rotation_deg"]
        ),
        "opposing_contacts_before_lift": fixed_before_lift and moving_before_lift,
        "spout_height": (
            spout_min >= float(threshold["spout_height_min_m"])
            and spout_max <= float(threshold["spout_height_max_m"])
        ),
        "pour_duration": 15.0 <= len(pour_rows) * dt_s <= 25.0,
        "visible_pour_in_rotation": (
            max(row["pitcher_tilt_deg"] for row in pour_in_rows)
            >= threshold["minimum_pour_tilt_deg"]
        ),
        "visible_pour_out_rotation": final_pour_out_tilt <= 15.0,
        "carried_grasp_translation": max(row["grasp_translation_m"] for row in carried_rows)
        <= threshold["grasp_translation_m"],
        "carried_grasp_rotation": max(row["grasp_rotation_deg"] for row in carried_rows)
        <= threshold["carried_grasp_rotation_deg"],
        "carried_base_drift": max(row["base_xy_drift_m"] for row in carried_rows)
        <= threshold["base_xy_drift_m"],
        "carried_base_roll_pitch": max(
            max(abs(row["base_roll_deg"]), abs(row["base_pitch_deg"]))
            for row in carried_rows
        ) <= threshold["base_roll_pitch_deg"],
        "stable_support_polygon": min(
            row["support_foot_contacts"] for row in carried_rows
        ) >= 4,
        "no_brewer_or_carried_forbidden_contact": not any(
            row["counter_contact"] or row["brewer_contact"]
            or row["robot_brewer_contact"] or row["non_claw_robot_contact"]
            for row in carried_rows
        ),
        "no_brewer_contact": not any(
            row["brewer_contact"] or row["robot_brewer_contact"] for row in motion_rows
        ),
        "carried_brewer_clearance": min(
            row["brewer_clearance_m"] for row in carried_rows
        ) >= threshold["minimum_brewer_clearance_m"],
        "no_joint_limit_or_self_contact": not any(
            row["joint_limit_violation"] or row["self_contact"] for row in motion_rows
        ),
        "support_before_release": set_down_contact_seen,
        "stable_final_set_down": (
            bool(final_rows)
            and len(final_rows) * dt_s >= float(threshold["final_settle_s"])
            and final_position_error <= float(threshold["set_down_position_error_m"])
            and max(row["pitcher_linear_speed_m_s"] for row in final_rows)
            <= float(threshold["set_down_linear_speed_m_s"])
            and max(row["pitcher_angular_speed_rad_s"] for row in final_rows)
            <= float(threshold["set_down_angular_speed_rad_s"])
            and all(row["counter_contact"] for row in final_rows)
        ),
        "retreat_does_not_move_pitcher": retreat_motion
        <= float(threshold["retreat_pitcher_motion_m"]) if retreat_rows else False,
    }
    stream_passes = {
        "sample_period": all(
            abs(stream_rows[index]["time_s"] - stream_rows[index - 1]["time_s"]
                - float(config["stream_dt_s"])) <= 1e-9
            for index in range(1, len(stream_rows))
        ),
        "no_undersampled_angle_step": bool(np.all(np.abs(wrapped_delta) <= math.pi / 2.0)),
        "turn_count": threshold["minimum_turns"] <= turns <= threshold["maximum_turns"],
        "starts_near_centre": radii[0] <= 0.010,
        "outward_progress": radial_progress >= threshold["minimum_radial_progress_m"],
        "radial_fit": radial_rmse <= threshold["maximum_radial_fit_rmse_m"],
        "inward_excursion": inward_excursion <= threshold["maximum_inward_excursion_m"],
        "edge_margin": edge_margin >= threshold["paper_edge_margin_m"],
        "landing_success": landing_success_rate >= threshold["landing_success_rate"],
        "all_intersections_valid": all(row["intersection_valid"] for row in stream_rows),
        "plausible_horizontal_span": (
            maximum_horizontal_span <= threshold["maximum_horizontal_span_m"]
        ),
        "plausible_horizontal_launch_speed": (
            maximum_horizontal_launch_speed
            <= threshold["maximum_horizontal_launch_speed_m_s"]
        ),
        "no_emission_after_pour_out": True,
    }

    payload = _payload(model, mujoco, pitcher_body)
    qpos_array = np.asarray(render_states)
    state_sha256 = hashlib.sha256(qpos_array.astype("<f8").tobytes()).hexdigest()
    video = out / "m4_spiral_pour.mp4"
    spiral_frame = out / "m4_outer_spiral.png"
    set_down_frame = out / "m4_set_down.png"
    render_enabled = not args.no_render and not args.fast_check
    if render_enabled:
        _render(
            model, mujoco, qpos_array, render_stream_landings, spout_site,
            float(bed_centre[2]), camera, float(config["render_fps"]), video,
        )
        for timestamp, frame in ((35.0, spiral_frame), (55.0, set_down_frame)):
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", str(frame),
            ], check=True)

    summary = {
        "simulation_only": True,
        "evidence_class": (
            "diagnostic_non_release_fast_check" if args.fast_check else "full_m4_verification"
        ),
        "release_evidence": not args.fast_check,
        "fast_check": args.fast_check,
        "fast_check_notice": (
            "NON-RELEASE DIAGNOSTIC: nominal 5 ms physics bounded at pour_out; "
            "run the default verifier for M4 release evidence"
            if args.fast_check else None
        ),
        "water_model": (
            "particle/visual ballistic approximation only; represented water is a rigid mass "
            "proxy; no fluid simulation and no real-robot run"
        ),
        "commit_sha": commit_sha,
        "backend": "native:cpu", "seed": seed,
        "task_geometry": "jumper.barista", "scene": "coffee_bar",
        "runtime": {
            "task_id": "jumper.barista", "actuator_class": type(servo).__name__,
            "actuator_base_class": ServoCurveActuator.__name__,
            "common_compute_called_by_subclass": True,
        },
        "trajectory_source": str(trajectory), "trajectory_sha256": config_sha256,
        "trajectory_method": config["source"], "input_shas": config["provenance"],
        "physics_dt_s": dt_s, "physics_sample_rate_hz": 1.0 / dt_s,
        "nominal_physics_dt_s": nominal_dt_s,
        "control_decimation": control_decimation,
        "control_rate_hz": 1.0 / control_dt_s,
        "stream_dt_s": float(config["stream_dt_s"]),
        "render_fps": float(config["render_fps"]), "duration_s": duration_s,
        "full_duration_s": full_duration_s,
        "bounded_end_phase": FAST_CHECK_END_PHASE if args.fast_check else None,
        "friction_cone": PROP_CONE, "impratio": PROP_IMPRATIO,
        "payload": payload,
        "motion": {
            "maximum_lift_m": max(row["lift_m"] for row in motion_rows),
            "continuous_lift_at_least_0_05_m_s": continuous_lift_duration,
            "valid_m3_hold_s": valid_hold_duration,
            "pour_duration_s": len(pour_rows) * dt_s,
            "spout_height_min_m": spout_min, "spout_height_max_m": spout_max,
            "maximum_pour_tilt_deg": maximum_pour_tilt,
            "final_pour_out_tilt_deg": final_pour_out_tilt,
            "maximum_carried_pitcher_tilt_deg": max(
                row["pitcher_tilt_deg"] for row in carried_rows
            ),
            "maximum_carried_grasp_translation_m": max(
                row["grasp_translation_m"] for row in carried_rows
            ),
            "maximum_carried_grasp_rotation_deg": max(
                row["grasp_rotation_deg"] for row in carried_rows
            ),
            "maximum_carried_base_xy_drift_m": max(
                row["base_xy_drift_m"] for row in carried_rows
            ),
            "maximum_carried_base_abs_roll_deg": max(
                abs(row["base_roll_deg"]) for row in carried_rows
            ),
            "maximum_carried_base_abs_pitch_deg": max(
                abs(row["base_pitch_deg"]) for row in carried_rows
            ),
            "minimum_carried_support_contacts": min(
                row["support_foot_contacts"] for row in carried_rows
            ),
            "minimum_carried_brewer_clearance_m": min(
                row["brewer_clearance_m"] for row in carried_rows
            ),
            "final_set_down_target_m": final_target.tolist(),
            "maximum_final_position_error_m": final_position_error,
            "maximum_final_linear_speed_m_s": max(
                row["pitcher_linear_speed_m_s"] for row in final_rows
            ) if final_rows else math.nan,
            "maximum_final_angular_speed_rad_s": max(
                row["pitcher_angular_speed_rad_s"] for row in final_rows
            ) if final_rows else math.nan,
            "maximum_retreat_pitcher_motion_m": retreat_motion,
            "set_down_support_contact_time_s": set_down_contact_time_s,
            "fixed_jaw_contact_before_lift": fixed_before_lift,
            "moving_jaw_contact_before_lift": moving_before_lift,
            "contact_pairs_seen": [list(pair) for pair in sorted(contact_pairs_seen)],
            "thresholds": threshold, "checks": motion_passes,
        },
        "stream": {
            "schema_columns": list(stream_rows[0]), "sample_count": len(stream_rows),
            "direction": stream_cfg["direction"], "unwrapped_angle_rad": angle_advance,
            "actual_spout_turn_count": turns, "turn_count": turns,
            "first_radius_m": float(radii[0]),
            "last_radius_m": float(radii[-1]), "maximum_radius_m": float(np.max(radii)),
            "radial_fit_slope_m_per_rad": float(slope),
            "radial_fit_intercept_m": float(intercept),
            "radial_fit_rmse_m": radial_rmse, "radial_progress_m": radial_progress,
            "maximum_inward_excursion_m": inward_excursion,
            "paper_edge_radius_m": float(stream_cfg["bed_acceptance_radius_m"]),
            "minimum_edge_margin_m": edge_margin,
            "maximum_landing_radius_m": maximum_landing_radius,
            "maximum_horizontal_span_m": maximum_horizontal_span,
            "maximum_horizontal_launch_speed_m_s": maximum_horizontal_launch_speed,
            "landing_success_count": len(stream_rows) - len(misses),
            "landing_miss_count": len(misses), "landing_success_rate": landing_success_rate,
            "misses": misses, "checks": stream_passes,
        },
        "torque": {
            "units": {"q": "rad", "qd": "rad/s", "tau": "N m", "time": "s"},
            "row_count": len(torque_rows), "rows_per_substep": len(joint_order),
            "joint_count": len(joint_order), "curve_tolerance_nm": 1e-4,
            "thermal_tolerance": 1e-5, "passes": torque_pass,
            "per_joint_and_phase": torque_summary,
        },
        "determinism": {
            "render_state_count": len(render_states), "qpos_sha256": state_sha256,
        },
        "artifacts": {
            "torque_csv": str(torque_csv), "motion_csv": str(motion_csv),
            "stream_csv": str(stream_csv), "video": str(video) if render_enabled else None,
            "spiral_frame": str(spiral_frame) if render_enabled else None,
            "set_down_frame": str(set_down_frame) if render_enabled else None,
            "dimensions_px": [WIDTH, HEIGHT],
        },
    }
    summary["passes"] = (
        payload["passes"] and torque_pass and all(motion_passes.values())
        and all(stream_passes.values())
    )
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", "utf-8")
    if args.fast_check:
        print("[m4] NON-RELEASE DIAGNOSTIC FAST CHECK (bounded at pour_out; no render)")
    print(f"[m4] actual-spout turns={turns:.6f}; radial progress={radial_progress:.6f} m; "
          f"landing={landing_success_rate:.3%} ({len(misses)} misses)")
    print(f"[m4] spout height={spout_min:.6f}..{spout_max:.6f} m; "
          f"edge margin={edge_margin:.6f} m")
    print(f"[m4] set-down error={final_position_error:.6f} m; "
          f"retreat motion={retreat_motion:.6f} m")
    print(f"[m4] payload={payload['design_total_kg']:.3f} kg; "
          f"torque pass={torque_pass}; overall pass={summary['passes']}")
    print(f"[m4] qpos sha256={state_sha256}")
    for path in (torque_csv, motion_csv, stream_csv, summary_path):
        print(f"[m4] artifact: {path}")
    if render_enabled:
        for path in (video, spiral_frame, set_down_frame):
            print(f"[m4] artifact: {path}")
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
