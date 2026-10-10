#!/usr/bin/env python3
"""Focused simulation-only checks for the M4 grip and open-V60 prerequisite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HOLD_TRAJECTORY = REPO / "barista" / "trajectory" / "m4_prerequisite_hold.json"
DIMENSIONS = REPO / "barista" / "coffee_scene_dimensions.json"
SCENE = REPO / "scenes" / "coffee_bar.scene" / "scene.xml"
RUN_M4 = REPO / "barista" / "verification" / "run_m4.py"
HOLD_PHASE = "spiral"
HOLD_REQUIRED_S = 25.0
TRANSLATION_LIMIT_M = 0.020
ROTATION_LIMIT_DEG = 12.0
BASE_DRIFT_LIMIT_M = 0.020
BASE_TIP_LIMIT_DEG = 5.0
SPOUT_HEIGHT_M = 0.030
MINIMUM_CLEARANCE_M = 0.005


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("hold", "clearance"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    return parser


def _longest_true(values: list[bool], dt_s: float) -> float:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * dt_s


def _rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _event_diagnostics(motion: list[dict[str, str]], column: str) -> dict:
    rows = [row for row in motion if row[column] == "True"]
    phases = {}
    for phase in dict.fromkeys(row["phase"] for row in rows):
        phase_rows = [row for row in rows if row["phase"] == phase]
        phases[phase] = {
            "sample_count": len(phase_rows),
            "first_time_s": float(phase_rows[0]["time_s"]),
            "last_time_s": float(phase_rows[-1]["time_s"]),
        }
    return {
        "sample_count": len(rows),
        "first_time_s": float(rows[0]["time_s"]) if rows else None,
        "last_time_s": float(rows[-1]["time_s"]) if rows else None,
        "phases": phases,
    }


def _hold(out: Path) -> int:
    raw = out / "raw_hold"
    raw.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUN_M4),
        "--fast-check",
        "--trajectory",
        str(HOLD_TRAJECTORY),
        "--out",
        str(raw),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    elapsed_s = time.perf_counter() - started
    (out / "raw_hold.log").write_text(completed.stdout + completed.stderr, "utf-8")
    raw_summary_path = raw / "summary.json"
    if not raw_summary_path.is_file():
        raise RuntimeError(f"hold replay did not produce {raw_summary_path}")

    raw_summary = json.loads(raw_summary_path.read_text("utf-8"))
    with (raw / "motion_substeps.csv").open(newline="", encoding="utf-8") as handle:
        motion = list(csv.DictReader(handle))
    hold = [row for row in motion if row["phase"] == HOLD_PHASE]
    if not hold:
        raise RuntimeError("hold replay contains no static hold rows")
    dt_s = float(raw_summary["physics_dt_s"])
    both_contacts = [
        row["fixed_jaw_contact"] == "True" and row["moving_jaw_contact"] == "True"
        for row in hold
    ]
    # The 25 s gate is a retention gate, not a repeat of M3's 50 mm lift gate:
    # the loaded pitcher's origin remains more than 30 mm above the counter and
    # both ordinary jaw contacts must remain present throughout.
    no_drop = [
        float(row["pitcher_z_m"]) >= 0.03
        and row["counter_contact"] == "False"
        and both
        for row, both in zip(hold, both_contacts, strict=True)
    ]
    first_position = np.asarray(
        [float(hold[0][f"pitcher_{axis}_m"]) for axis in "xyz"], dtype=np.float64
    )
    last_position = np.asarray(
        [float(hold[-1][f"pitcher_{axis}_m"]) for axis in "xyz"], dtype=np.float64
    )
    first_rotation = np.asarray(
        [[float(hold[0][f"pitcher_{axis}{basis}"]) for basis in "xyz"] for axis in "xyz"]
    )
    last_rotation = np.asarray(
        [[float(hold[-1][f"pitcher_{axis}{basis}"]) for basis in "xyz"] for axis in "xyz"]
    )
    both_contact_duration_s = _longest_true(both_contacts, dt_s)
    no_drop_duration_s = _longest_true(no_drop, dt_s)
    translation_m = float(np.linalg.norm(last_position - first_position))
    rotation_deg = _rotation_angle_deg(first_rotation, last_rotation)
    event_columns = {
        "robot_brewer": "robot_brewer_contact",
        "pitcher_brewer": "brewer_contact",
        "pitcher_non_claw_robot": "non_claw_robot_contact",
        "robot_self": "self_contact",
        "joint_limit": "joint_limit_violation",
    }
    forbidden_events = {
        name: _event_diagnostics(motion, column)
        for name, column in event_columns.items()
    }
    forbidden = any(event["sample_count"] for event in forbidden_events.values())
    whole_joint = {
        value["joint"]: value
        for value in raw_summary["torque"]["per_joint_and_phase"].values()
        if value["phase"] == "whole_run"
    }
    checks = {
        "both_jaws_contact_before_lift": (
            raw_summary["motion"]["fixed_jaw_contact_before_lift"]
            and raw_summary["motion"]["moving_jaw_contact_before_lift"]
        ),
        "both_jaws_contact_for_25_s": both_contact_duration_s >= HOLD_REQUIRED_S,
        "no_drop_for_25_s": no_drop_duration_s >= HOLD_REQUIRED_S,
        "bounded_hold_translation": translation_m <= TRANSLATION_LIMIT_M,
        "bounded_hold_rotation": rotation_deg <= ROTATION_LIMIT_DEG,
        "no_robot_brewer_contact_whole_run": not forbidden_events["robot_brewer"]["sample_count"],
        "no_pitcher_brewer_contact_whole_run": not forbidden_events["pitcher_brewer"]["sample_count"],
        "no_pitcher_non_claw_robot_contact_whole_run": not forbidden_events[
            "pitcher_non_claw_robot"
        ]["sample_count"],
        "no_robot_self_contact_whole_run": not forbidden_events["robot_self"]["sample_count"],
        "no_forbidden_or_self_contact_whole_run": not forbidden,
        "support_stable": min(int(row["support_foot_contacts"]) for row in hold) >= 5,
        "body_drift_bounded": max(float(row["base_xy_drift_m"]) for row in hold)
        <= BASE_DRIFT_LIMIT_M,
        "body_tip_bounded": max(
            max(abs(float(row["base_roll_deg"])), abs(float(row["base_pitch_deg"])))
            for row in hold
        ) <= BASE_TIP_LIMIT_DEG,
        "joint_limits_whole_run": not forbidden_events["joint_limit"]["sample_count"],
        "payload": raw_summary["payload"]["passes"],
        "servo_curve_peak_dwell_and_thermal": raw_summary["torque"]["passes"],
    }
    summary = {
        "simulation_only": True,
        "check": "ordinary_contact_static_hold",
        "backend": raw_summary["backend"],
        "inner_command": command,
        "inner_exit_code": completed.returncode,
        "inner_exit_note": "run_m4 also evaluates intentionally out-of-scope spiral/set-down gates",
        "elapsed_s": elapsed_s,
        "physics_dt_s": dt_s,
        "hold_phase_duration_s": len(hold) * dt_s,
        "both_jaw_contact_duration_s": both_contact_duration_s,
        "no_drop_duration_s": no_drop_duration_s,
        "hold_translation_m": translation_m,
        "hold_rotation_deg": rotation_deg,
        "minimum_support_contacts": min(int(row["support_foot_contacts"]) for row in hold),
        "maximum_base_xy_drift_m": max(float(row["base_xy_drift_m"]) for row in hold),
        "maximum_base_tip_deg": max(
            max(abs(float(row["base_roll_deg"])), abs(float(row["base_pitch_deg"])))
            for row in hold
        ),
        "minimum_physical_brewer_clearance_m": min(
            float(row["brewer_clearance_m"]) for row in hold
        ),
        "forbidden_contact": forbidden,
        "forbidden_contact_events_whole_run": forbidden_events,
        "joint_limit_violation": bool(forbidden_events["joint_limit"]["sample_count"]),
        "drop_state": not all(no_drop),
        "payload": raw_summary["payload"],
        "torque_joint_count": len(whole_joint),
        "torque_per_joint": whole_joint,
        "checks": checks,
        "passes": all(checks.values()),
        "artifacts": {
            "raw_summary": str(raw_summary_path),
            "motion_csv": str(raw / "motion_substeps.csv"),
            "torque_csv": str(raw / "torque_substeps.csv"),
            "raw_log": str(out / "raw_hold.log"),
        },
    }
    path = out / "hold_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"[m4-prerequisite] hold contacts={both_contact_duration_s:.3f} s; "
          f"translation={translation_m:.6f} m; rotation={rotation_deg:.3f} deg")
    print(f"[m4-prerequisite] payload={raw_summary['payload']['design_total_kg']:.3f} kg; "
          f"torque={raw_summary['torque']['passes']}; pass={summary['passes']}")
    print(f"[m4-prerequisite] artifact: {path}")
    return 0 if summary["passes"] else 1


def _point_segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    delta = second - first
    fraction = float(np.clip(np.dot(point - first, delta) / np.dot(delta, delta), 0.0, 1.0))
    return float(np.linalg.norm(point - (first + fraction * delta)))


def _clearance(out: Path, render: bool) -> int:
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    import mujoco

    started = time.perf_counter()
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    pitcher_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pitcher")
    pitcher_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pitcher_free")
    spout_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pitcher_spout_tip")
    bed_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "coffee_bed_target")
    mujoco.mj_forward(model, data)
    bed = data.site_xpos[bed_site].copy()
    local_spout = model.site_pos[spout_site].copy()
    desired_spout = bed + np.asarray([0.0, 0.0, SPOUT_HEIGHT_M])
    qpos = int(model.jnt_qposadr[pitcher_joint])
    data.qpos[qpos : qpos + 3] = desired_spout - local_spout
    data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    pitcher_geoms = {
        index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == pitcher_body
    }
    brewer_geoms = {
        index
        for index in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(
            ("server_", "v60_")
        )
    }
    fromto = np.zeros(6, dtype=np.float64)
    pair_clearances = []
    for pitcher_geom in pitcher_geoms:
        for brewer_geom in brewer_geoms:
            pair_clearances.append(
                float(mujoco.mj_geomDistance(model, data, pitcher_geom, brewer_geom, 1.0, fromto))
            )
    minimum_clearance = min(pair_clearances)
    forbidden_contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} & pitcher_geoms and {
            int(contact.geom1), int(contact.geom2)
        } & brewer_geoms:
            forbidden_contacts.append(index)

    v60_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        for index in brewer_geoms
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("v60_")
    ]
    ribs = [name for name in v60_names if name.startswith("v60_rib_")]
    rims = [name for name in v60_names if name.startswith("v60_rim_")]
    centre = np.asarray([0.0, 0.0, 0.084])
    rib_axis_clearances = []
    for name in ribs:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body = int(model.geom_bodyid[geom])
        rotation = data.xmat[body].reshape(3, 3)
        midpoint = data.geom_xpos[geom]
        half_length = float(model.geom_size[geom, 1])
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        first, second = midpoint - axis * half_length, midpoint + axis * half_length
        rib_axis_clearances.append(_point_segment_distance(data.xpos[body] + rotation @ centre, first, second)
                                   - float(model.geom_size[geom, 0]))
    dimensions = json.loads(DIMENSIONS.read_text("utf-8"))
    checks = {
        "open_rib_count": len(ribs) == 12,
        "collidable_rim_count": len(rims) == 12,
        "physical_support_present": "v60_support" in v60_names,
        "no_solid_stacked_cylinders": not any(
            name.startswith("v60_") and name[4:].isdigit() for name in v60_names
        ),
        "centred_spout": float(np.linalg.norm(data.site_xpos[spout_site, :2] - bed[:2])) <= 1e-9,
        "spout_height_2_to_5_cm": 0.02 <= float(data.site_xpos[spout_site, 2] - bed[2]) <= 0.05,
        "pitcher_brewer_clearance": minimum_clearance >= MINIMUM_CLEARANCE_M,
        "no_pitcher_brewer_contact": not forbidden_contacts,
        "open_interior_axis": min(rib_axis_clearances) >= 0.009,
        "payload_under_450_g": dimensions["payload_check"]["pitcher_plus_water_mass_kg"] <= 0.450,
    }
    frame = out / "open_v60_centered.png"
    if render:
        import imageio.v3 as iio

        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "hero")
        renderer = mujoco.Renderer(model, height=360, width=640)
        try:
            renderer.update_scene(data, camera=camera)
            iio.imwrite(frame, renderer.render())
        finally:
            renderer.close()
    elapsed_s = time.perf_counter() - started
    summary = {
        "simulation_only": True,
        "check": "open_v60_centered_spout_clearance",
        "backend": "native:cpu",
        "commit_sha": commit_sha,
        "elapsed_s": elapsed_s,
        "spout_position_m": data.site_xpos[spout_site].tolist(),
        "bed_position_m": bed.tolist(),
        "spout_height_above_bed_m": float(data.site_xpos[spout_site, 2] - bed[2]),
        "minimum_physical_brewer_clearance_m": minimum_clearance,
        "minimum_open_axis_clearance_m": min(rib_axis_clearances),
        "v60_colliding_geoms": sorted(v60_names),
        "forbidden_contact_count": len(forbidden_contacts),
        "payload_mass_kg": dimensions["payload_check"]["pitcher_plus_water_mass_kg"],
        "checks": checks,
        "passes": all(checks.values()),
        "artifacts": {"frame": str(frame) if render else None},
    }
    path = out / "clearance_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"[m4-prerequisite] centred spout height={summary['spout_height_above_bed_m']:.6f} m; "
          f"physical clearance={minimum_clearance:.6f} m")
    print(f"[m4-prerequisite] open-axis clearance={min(rib_axis_clearances):.6f} m; "
          f"pass={summary['passes']}")
    print(f"[m4-prerequisite] artifact: {path}")
    if render:
        print(f"[m4-prerequisite] artifact: {frame}")
    return 0 if summary["passes"] else 1


def main() -> int:
    args = _parser().parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.check == "hold":
        return _hold(out)
    return _clearance(out, not args.no_render)


if __name__ == "__main__":
    raise SystemExit(main())
