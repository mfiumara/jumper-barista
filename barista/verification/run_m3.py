#!/usr/bin/env python3
"""Replay and verify the simulation-only M3 pitcher grasp, lift, and hold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = REPO / "barista" / "trajectory" / "m3_grasp_lift.json"
DIMENSIONS = REPO / "barista" / "coffee_scene_dimensions.json"
WIDTH, HEIGHT = 960, 540
CONTROL_DECIMATION = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
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


def _render(model, mujoco, states: np.ndarray, camera: int, fps: float, path: Path) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), size=(WIDTH, HEIGHT), fps=fps, macro_block_size=1,
        codec="libx264", pix_fmt_out="yuv420p", quality=7,
    )
    writer.send(None)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    data = mujoco.MjData(model)
    try:
        for state in states:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
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
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import scenes
    import tasks
    from tasks.jumper.barista.audited_servo import AuditedServo
    from tasks.jumper.common.actuator import CURVE
    from tasks.jumper.common.constants import HOME
    from tasks.jumper.five_foot.claw import GRIPPER_OPEN, LEAD_PER_SPEED, SQUEEZE_LEAD
    from tasks.jumper.five_foot.objects import PROP_CONE, PROP_IMPRATIO

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(TRAJECTORY.read_text("utf-8"))
    seed = int(config["seed"])
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)
    print(resolution.banner())

    env_cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
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
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

    joint_ids = {
        name: _ending(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, "njnt", name)
        for name in HOME
    }
    compiled_joint_names = [_name(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, joint_ids[name]) for name in HOME]
    base_joint = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, "njnt", "floating_base")
    base_qadr = int(model.jnt_qposadr[base_joint])
    base_start = config["base_start"]
    yaw = float(base_start["yaw_rad"])
    data.qpos[base_qadr:base_qadr + 3] = base_start["position_m"]
    data.qpos[base_qadr + 3:base_qadr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    for name, value in HOME.items():
        data.qpos[model.jnt_qposadr[joint_ids[name]]] = value
    arm_names = config["arm_joint_order"]
    for name, value in zip(arm_names, config["keyframes"][0]["arm_rad"], strict=True):
        data.qpos[model.jnt_qposadr[joint_ids[name]]] = value
    data.qpos[model.jnt_qposadr[joint_ids["LF_J4_joint"]]] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    servo = AuditedServo(model, mujoco, compiled_joint_names)
    joint_order = list(HOME)
    joint_col = {name: index for index, name in enumerate(joint_order)}
    target = np.asarray([HOME[name] for name in joint_order], dtype=np.float64)
    finger_col = joint_col["LF_J4_joint"]

    pitcher_body = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "nbody", "pitcher")
    base_body = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "nbody", "base_link")
    palm_body = _ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "nbody", "LF_palm_link")
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

    dt_s = float(model.opt.timestep)
    if abs(dt_s - float(config["physics_dt_s"])) > 1e-12:
        raise RuntimeError(f"compiled dt {dt_s} differs from trajectory dt")
    control_dt_s = dt_s * CONTROL_DECIMATION
    if abs(control_dt_s - float(config["control_dt_s"])) > 1e-12:
        raise RuntimeError(f"control dt {control_dt_s} differs from trajectory dt")
    duration_s = float(config["keyframes"][-1]["end_s"])
    steps = round(duration_s / dt_s)
    render_stride = round(1.0 / (float(config["render_fps"]) * dt_s))
    if render_stride * dt_s * float(config["render_fps"]) != 1.0:
        raise RuntimeError("render rate must divide the physics rate exactly")

    torque_rows: list[dict] = []
    motion_rows: list[dict] = []
    render_states: list[np.ndarray] = []
    above_window = np.zeros(len(joint_order), dtype=np.int64)
    above_elapsed = np.zeros(len(joint_order), dtype=np.float64)
    next_window = np.zeros(len(joint_order), dtype=np.int64)
    contact_pairs_seen: set[tuple[str, str]] = set()
    settled_pitcher_z = None
    settled_pitcher_rot = None
    settled_base_xy = None
    settled_base_rot = None
    grasp_reference = None
    config_sha256 = hashlib.sha256(TRAJECTORY.read_bytes()).hexdigest()

    try:
        for step in range(steps):
            time_s = step * dt_s
            phase, _, _ = _target(config, time_s)
            if step % CONTROL_DECIMATION == 0:
                _, arm, finger_request = _target(config, time_s)
                target[:] = [HOME[name] for name in joint_order]
                if time_s > 4.0:
                    support_fraction = _smooth((time_s - 4.0) / 2.0)
                    for name, lifted in config["support_lift_rad"].items():
                        target[joint_col[name]] = (
                            HOME[name] + (float(lifted) - HOME[name]) * support_fraction
                        )
                for name, value in zip(arm_names, arm, strict=True):
                    target[joint_col[name]] = value
                finger_q = float(data.qpos[servo.qadr[finger_col]])
                finger_qd = float(data.qvel[servo.dadr[finger_col]])
                if finger_request > finger_q:
                    target[finger_col] = min(
                        finger_request,
                        finger_q + SQUEEZE_LEAD + LEAD_PER_SPEED * max(finger_qd, 0.0),
                    )
                else:
                    target[finger_col] = finger_request

            samples = servo.apply(data, target)
            for index, (joint, sample) in enumerate(zip(joint_order, samples, strict=True)):
                magnitude = abs(sample.tau_applied_nm)
                above = magnitude > CURVE.continuous_torque
                if above and above_elapsed[index] == 0.0:
                    next_window[index] += 1
                    above_window[index] = next_window[index]
                if above:
                    above_elapsed[index] += dt_s
                else:
                    above_window[index] = 0
                    above_elapsed[index] = 0.0
                thermal_recomputed = sample.thermal_theta_pre + (
                    (sample.tau_applied_nm / CURVE.continuous_torque) ** 2
                    - sample.thermal_theta_pre
                ) * (dt_s / CURVE.thermal_time_constant)
                curve_margin = sample.curve_limit_nm - magnitude
                effective_margin = sample.effective_limit_nm - magnitude
                torque_rows.append({
                    "run_id": f"m3-seed-{seed}", "phase": phase, "joint": joint,
                    "substep": step, "control_step": step // CONTROL_DECIMATION,
                    "time_s": time_s, "dt_s": dt_s,
                    "q_pre_rad": sample.q_pre_rad, "qd_pre_rad_s": sample.qd_pre_rad_s,
                    "tau_raw_nm": sample.tau_raw_nm, "tau_applied_nm": sample.tau_applied_nm,
                    "curve_limit_nm": sample.curve_limit_nm,
                    "continuous_limit_nm": CURVE.continuous_torque,
                    "effective_limit_nm": sample.effective_limit_nm,
                    "thermal_theta_pre": sample.thermal_theta_pre,
                    "thermal_theta_post": sample.thermal_theta_post,
                    "curve_margin_nm": curve_margin,
                    "effective_margin_nm": effective_margin,
                    "above_continuous": above,
                    "derating_active": sample.thermal_theta_pre >= 1.0,
                    "curve_violation": curve_margin < -1e-4 or effective_margin < -1e-4,
                    "above_continuous_window_id": int(above_window[index]),
                    "above_continuous_elapsed_s": float(above_elapsed[index]),
                    "thermal_recompute_error": abs(thermal_recomputed - sample.thermal_theta_post),
                })

            mujoco.mj_step(model, data)
            pitcher_pos = data.xpos[pitcher_body].copy()
            pitcher_rot = data.xmat[pitcher_body].reshape(3, 3).copy()
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
            non_claw_robot_contact = self_contact = False
            support_touching: set[int] = set()
            contact_names: list[str] = []
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                name1 = _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                name2 = _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                pair = tuple(sorted((name1, name2)))
                contact_pairs_seen.add(pair)
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
                counter_contact |= other == terrain
                brewer_contact |= other in brewer_geoms
                if other_name.startswith("robot/") and not any(
                    token in other_name for token in ("robot/LF_palm", "robot/LF_finger")
                ):
                    non_claw_robot_contact = True

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
            if phase == "hold" and grasp_reference is None:
                grasp_reference = (local_pos.copy(), local_rot.copy())
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
                "counter_contact": counter_contact, "brewer_contact": brewer_contact,
                "non_claw_robot_contact": non_claw_robot_contact,
                "self_contact": self_contact, "joint_limit_violation": joint_limit_violation,
                "pitcher_contact_geoms": ";".join(sorted(set(contact_names))),
            })
            if step % render_stride == 0:
                render_states.append(data.qpos.copy())
    finally:
        env.close()

    if settled_pitcher_z is None:
        raise RuntimeError("trajectory ended before the settled reference sample")

    torque_csv = out / "torque_substeps.csv"
    motion_csv = out / "motion_substeps.csv"
    torque_fields = list(torque_rows[0])
    motion_fields = list(motion_rows[0])
    with torque_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=torque_fields)
        writer.writeheader()
        writer.writerows(torque_rows)
    with motion_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=motion_fields)
        writer.writeheader()
        writer.writerows(motion_rows)

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
    hold_rows = [row for row in motion_rows if row["phase"] == "hold"]
    hold_duration = len(hold_rows) * dt_s
    valid_hold_duration = _longest_true(
        [row["lift_m"] >= float(threshold["lift_m"]) for row in hold_rows], dt_s
    )
    pre_lift_rows = [row for row in motion_rows if row["phase"] in ("approach", "close")]
    fixed_before_lift = any(row["fixed_jaw_contact"] for row in pre_lift_rows)
    moving_before_lift = any(row["moving_jaw_contact"] for row in pre_lift_rows)
    motion_passes = {
        "lift_and_hold": (
            hold_duration >= float(threshold["hold_s"])
            and valid_hold_duration >= float(threshold["hold_s"])
        ),
        "opposing_contacts_before_lift": fixed_before_lift and moving_before_lift,
        "tilt": max(row["pitcher_tilt_deg"] for row in hold_rows) <= threshold["pitcher_tilt_deg"],
        "grasp_translation": max(row["grasp_translation_m"] for row in hold_rows)
        <= threshold["grasp_translation_m"],
        "grasp_rotation": max(row["grasp_rotation_deg"] for row in hold_rows)
        <= threshold["grasp_rotation_deg"],
        "base_drift": max(row["base_xy_drift_m"] for row in hold_rows)
        <= threshold["base_xy_drift_m"],
        "base_roll_pitch": max(
            max(abs(row["base_roll_deg"]), abs(row["base_pitch_deg"])) for row in hold_rows
        ) <= threshold["base_roll_pitch_deg"],
        "all_support_feet": min(row["support_foot_contacts"] for row in hold_rows) == 5,
        "no_forbidden_contacts": not any(
            row["counter_contact"] or row["brewer_contact"] or row["non_claw_robot_contact"]
            for row in hold_rows
        ),
        "no_joint_limit_or_self_contact": not any(
            row["joint_limit_violation"] or row["self_contact"] for row in motion_rows
        ),
    }
    payload = _payload(model, mujoco, pitcher_body)
    qpos_array = np.asarray(render_states)
    state_sha256 = hashlib.sha256(qpos_array.astype("<f8").tobytes()).hexdigest()

    video = out / "m3_grasp_lift.mp4"
    frame = out / "m3_hold.png"
    if not args.no_render:
        _render(model, mujoco, qpos_array, camera, float(config["render_fps"]), video)
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-ss", "13.5", "-i", str(video), "-frames:v", "1", str(frame),
        ], check=True)

    summary = {
        "simulation_only": True,
        "water_model": "rigid represented-water mass; no fluid simulation",
        "commit_sha": commit_sha,
        "backend": "native:cpu", "seed": seed,
        "task_geometry": "jumper.five_foot", "scene": "coffee_bar",
        "trajectory_source": str(TRAJECTORY),
        "trajectory_sha256": config_sha256,
        "trajectory_method": config["source"],
        "input_shas": config["provenance"],
        "physics_dt_s": dt_s, "physics_sample_rate_hz": 1.0 / dt_s,
        "control_decimation": CONTROL_DECIMATION,
        "control_rate_hz": 1.0 / (dt_s * CONTROL_DECIMATION),
        "render_fps": float(config["render_fps"]), "duration_s": duration_s,
        "friction_cone": PROP_CONE, "impratio": PROP_IMPRATIO,
        "payload": payload,
        "motion": {
            "settled_pitcher_origin_z_m": settled_pitcher_z,
            "maximum_pitcher_origin_z_m": max(row["pitcher_z_m"] for row in motion_rows),
            "maximum_lift_m": max(row["lift_m"] for row in motion_rows),
            "continuous_lift_at_least_0_05_m_s": continuous_lift_duration,
            "declared_hold_phase_duration_s": hold_duration,
            "valid_hold_phase_duration_s": valid_hold_duration,
            "maximum_hold_tilt_deg": max(row["pitcher_tilt_deg"] for row in hold_rows),
            "maximum_hold_grasp_translation_m": max(
                row["grasp_translation_m"] for row in hold_rows
            ),
            "maximum_hold_grasp_rotation_deg": max(row["grasp_rotation_deg"] for row in hold_rows),
            "maximum_hold_base_xy_drift_m": max(row["base_xy_drift_m"] for row in hold_rows),
            "maximum_hold_base_abs_roll_deg": max(abs(row["base_roll_deg"]) for row in hold_rows),
            "maximum_hold_base_abs_pitch_deg": max(abs(row["base_pitch_deg"]) for row in hold_rows),
            "minimum_hold_support_foot_contacts": min(
                row["support_foot_contacts"] for row in hold_rows
            ),
            "minimum_hold_brewer_clearance_m": min(row["brewer_clearance_m"] for row in hold_rows),
            "fixed_jaw_contact_before_lift": fixed_before_lift,
            "moving_jaw_contact_before_lift": moving_before_lift,
            "contact_pairs_seen": [list(pair) for pair in sorted(contact_pairs_seen)],
            "thresholds": threshold, "checks": motion_passes,
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
            "video": str(video) if not args.no_render else None,
            "frame": str(frame) if not args.no_render else None,
            "dimensions_px": [WIDTH, HEIGHT],
        },
    }
    summary["passes"] = payload["passes"] and torque_pass and all(motion_passes.values())
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", "utf-8"
    )
    print(f"[m3] lift={summary['motion']['maximum_lift_m']:.6f} m; "
          f"continuous >=50 mm={continuous_lift_duration:.3f} s; "
          f"valid hold={valid_hold_duration:.3f} s")
    print(f"[m3] max hold tilt={summary['motion']['maximum_hold_tilt_deg']:.3f} deg; "
          f"base drift={summary['motion']['maximum_hold_base_xy_drift_m']:.6f} m")
    print(f"[m3] payload={payload['design_total_kg']:.3f} kg; torque pass={torque_pass}")
    print(f"[m3] qpos sha256={state_sha256}; overall pass={summary['passes']}")
    for path in (torque_csv, motion_csv, summary_path):
        print(f"[m3] artifact: {path}")
    if not args.no_render:
        print(f"[m3] artifact: {video}")
        print(f"[m3] artifact: {frame}")
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
