#!/usr/bin/env python3
"""A4: one continuous, deterministic MuJoCo run in ``cafe_bar`` (simulation only).

dance idle -> chime -> stop -> excited flat jump in place -> settle (exclaim) -> look/listen
-> confirm -> turn ~90 deg -> walk along the bar -> align (+ the scripted pre_align ->
pour_start approach) -> the accepted M4 spiral pour, re-targeted closed-loop -> set-down.

One world, built once and never reset or written to after the boot: every beat continues
from the previous beat's state. Physics 1 kHz, app/policy tick 200 Hz, M4 targets 50 Hz.
Writes beats.json (jumper-barista-beats/1), summary.json (per-beat metrics, torque audit,
determinism hashes), trace.csv, torque_applied.npz, render_states.npz (+ the pour's own
run_m4 evidence under pour/).

    python barista/chain/run_chain.py --bundle <bundle>/jumper --out out/chain/run1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _load(stem: str):
    name = f"barista_chain_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{stem}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("common")
world = _load("world")
navigation = _load("navigation")
approach = _load("approach")
pour = _load("pour")
beats_mod = _load("beats")
footprint = _load("footprint")

DANCE_DPAD = {"dance_crab": (0, -1), "dance_brazilian": (0, 1), "dance_maze": (-1, 0),
              "dance_dream_wings": (1, 0)}
POUR_PHASES = ("settle", "approach", "close", "lift", "lift_settle", "lift_hold", "transit",
               "pour_in", "spiral", "pour_out")
SET_DOWN_PHASES = ("return", "lower", "set_down", "release", "clear_retreat", "retreat",
                   "final_settle")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle", type=Path, required=True, help="the built jumper bundle dir")
    p.add_argument("--reference", type=Path, required=True,
                   help="pour_reference.npz (pour.py --mode reference, then augment_reference.py)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dance", default="dance_brazilian", choices=sorted(DANCE_DPAD))
    p.add_argument("--dance-s", type=float, default=10.0)
    p.add_argument("--chime-s", type=float, default=1.0, help="chime cue length before the stop")
    p.add_argument("--exclaim-s", type=float, default=beats_mod.DEFAULT_HOLDS_S["exclaim"])
    p.add_argument("--listen-s", type=float, default=beats_mod.DEFAULT_HOLDS_S["listen"])
    p.add_argument("--confirm-s", type=float, default=beats_mod.DEFAULT_HOLDS_S["confirm"])
    p.add_argument("--seed", type=int, default=common.SEED)
    p.add_argument("--stop-after", default=None, choices=("align",),
                   help="debug: stop after this beat and save the arrival state")
    return p.parse_args()


class Recorder:
    def __init__(self, chain) -> None:
        self.c = chain
        self.beats: list[dict] = []
        self.boundaries: list[dict] = []

    def mark(self, beat: str, t0: float, t1: float) -> None:
        self.beats.append({"beat": beat, "t_start_s": t0, "t_end_s": t1})

    def boundary(self, label: str) -> None:
        q = self.c.data.qpos
        self.boundaries.append({"at": label, "t_s": self.c.t_substep(),
                                "qpos_sha256": common.sha256_array(q),
                                "base_xyz": [float(v) for v in q[:3]]})


def guard_state_writes(chain) -> dict:
    """After the boot nothing may write the simulated state: count any attempt."""
    counter = {"root": 0, "joint": 0}
    robot = chain.robot
    real_root = robot.write_root_link_pose_to_sim
    real_joint = robot.write_joint_state_to_sim

    def root(*a, **k):
        counter["root"] += 1
        return real_root(*a, **k)

    def joint(*a, **k):
        counter["joint"] += 1
        return real_joint(*a, **k)

    robot.write_root_link_pose_to_sim = root
    robot.write_joint_state_to_sim = joint
    return counter


def window_metrics(log: list[dict], t0: float, t1: float) -> dict:
    rows = [r for r in log if t0 < r["t"] <= t1 + 1e-9]
    if not rows:
        return {}
    x = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    return {
        "max_abs_roll_deg": math.degrees(max(abs(r["roll"]) for r in rows)),
        "max_abs_pitch_deg": math.degrees(max(abs(r["pitch"]) for r in rows)),
        "max_tilt_deg": math.degrees(max(r["tilt"] for r in rows)),
        "foot_slip_max_per_tick_mm": 1e3 * max(r["slip_step_m"] for r in rows),
        "foot_slip_sum_mm": 1e3 * sum(r["slip_sum_m"] for r in rows),
        "min_feet_in_contact": min(r["contacts"] for r in rows),
        "torso_on_bar_ticks": sum(r["torso_on_bar"] for r in rows),
        "nonfoot_bar_contact_ticks": sum(r["body_contact"] for r in rows),
        "brewer_contact_ticks": sum(r["brewer_contact"] for r in rows),
        "pitcher_contact_ticks": sum(r["pitcher_contact"] for r in rows),
        "off_bar_contact_ticks": sum(r["off_bar_contact"] for r in rows),
        "base_xy_start": [float(x[0]), float(y[0])], "base_xy_end": [float(x[-1]), float(y[-1])],
        "base_xy_travel_m": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
        "base_z_min_m": float(min(r["z"] for r in rows)),
        "base_z_max_m": float(max(r["z"] for r in rows)),
        "yaw_end_deg": math.degrees(rows[-1]["yaw"]),
    }


def main() -> int:
    args = parse()
    wall = time.time()
    common.headless()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dims = common.cafe_dims()
    route = {w["name"]: w for w in dims["route"]["walk"]}
    start = dims["route"]["start_slot"]
    chain = world.Chain(args.bundle, seed=args.seed,
                        root_xyzyaw=(start["x_m"], start["y_m"], 0.11, math.radians(start["yaw_deg"])))
    writes = guard_state_writes(chain)
    rec = Recorder(chain)
    events: dict[str, float] = {}
    from tasks.jumper.common.constants import HOME
    from tasks.jumper.five_foot.claw import GRIPPER_OPEN

    # ------------------------------------------------------------ dance (idle)
    rec.boundary("start")
    t_dance0 = 0.0
    chain.run(2.0, beat="dance")  # warm start into locomotion standing
    chain.press(["menu"], beat="dance")
    chain.press(["menu"], DANCE_DPAD[args.dance], beat="dance")
    chain.release(beat="dance")
    events["dance_press_s"] = chain.t_substep()
    chain.run(args.dance_s - args.chime_s, beat="dance")
    events["chime_s"] = chain.t_substep()
    chain.run(args.chime_s, beat="dance")
    t_stop0 = chain.t_substep()
    rec.mark("dance", t_dance0, t_stop0)
    rec.mark("chime", events["chime_s"], t_stop0)
    rec.boundary("dance->stop")
    # ------------------------------------------------------------ stop
    chain.press(["menu"], beat="stop", hold_s=0.1)
    chain.release(beat="stop")
    quiet_since = None
    loco_since = None
    while chain.t_substep() - t_stop0 < 6.0:
        chain.run(chain.step_dt, beat="stop")
        r = chain.log[-1]
        if r["mode"] == "locomotion" and r["running"]:
            loco_since = loco_since if loco_since is not None else r["t"]
        else:
            loco_since = None
        prev = chain.log[-2]
        v = math.hypot(r["x"] - prev["x"], r["y"] - prev["y"]) / chain.step_dt
        wz = abs(common.wrap(r["yaw"] - prev["yaw"])) / chain.step_dt
        quiet = math.degrees(r["tilt"]) < 2.0 and v < 0.05 and math.degrees(wz) < 10.0
        quiet_since = (quiet_since if quiet_since is not None else r["t"]) if quiet else None
        if (loco_since is not None and r["t"] - loco_since >= 1.0 and quiet_since is not None
                and r["t"] - quiet_since >= 0.25 and r["t"] - t_stop0 >= 2.5):
            break
    t_jump0 = chain.t_substep()
    rec.mark("stop", t_stop0, t_jump0)
    rec.boundary("stop->excited_jump")
    pre = [r for r in chain.log if t_jump0 - 0.5 <= r["t"] <= t_jump0]
    jump_ref = {k: float(np.mean([r[k] for r in pre])) for k in ("x", "y", "z", "yaw")}
    # ------------------------------------------------------------ excited jump (+ exclaim)
    chain.press(["A"], beat="excited_jump", hold_s=0.06)
    chain.release(beat="excited_jump")
    airborne = False
    touchdown = None
    settled = None
    while chain.t_substep() - t_jump0 < 8.0:
        chain.run(chain.step_dt, beat="excited_jump")
        r = chain.log[-1]
        if r["contacts"] == 0:
            airborne = True
        elif airborne and touchdown is None:
            touchdown = r["t"]
        if touchdown is not None and settled is None:
            win = [w for w in chain.log if w["t"] > r["t"] - 0.5]
            if (r["t"] - touchdown >= 0.5 and r["mode"] == "locomotion" and r["running"]
                    and all(math.degrees(w["tilt"]) < 2.0 for w in win)
                    and max(w["z"] for w in win) - min(w["z"] for w in win) < 0.003):
                settled = r["t"]
        if settled is not None and r["t"] >= max(settled, touchdown + args.exclaim_s):
            break
    if touchdown is None or settled is None:
        raise SystemExit(f"jump did not land and settle (touchdown={touchdown}, settled={settled})")
    t_look0 = chain.t_substep()
    rec.mark("excited_jump", t_jump0, t_look0)
    rec.mark("exclaim", touchdown, touchdown + args.exclaim_s)
    events.update(jump_touchdown_s=touchdown, jump_settled_s=settled)
    rec.boundary("excited_jump->look")
    # ------------------------------------------------------------ look / listen / confirm
    chain.run(args.listen_s, beat="look")
    t_conf = chain.t_substep()
    chain.run(args.confirm_s, beat="look")
    t_turn0 = chain.t_substep()
    rec.mark("look", t_look0, t_turn0)
    rec.mark("listen", t_look0, t_conf)
    rec.mark("confirm", t_conf, t_turn0)
    rec.boundary("look->turn")
    # ------------------------------------------------------------ turn ~90 deg in place
    x, y, _ = chain.pose2d()
    navigation.turn_to(chain, (x, y, math.radians(route["start"]["yaw_deg"])), beat="turn")
    t_walk0 = chain.t_substep()
    rec.mark("turn", t_turn0, t_walk0)
    rec.boundary("turn->walk")
    # ------------------------------------------------------------ walk along the bar
    pre_align = (route["pre_align"]["x_m"], route["pre_align"]["y_m"],
                 math.radians(route["pre_align"]["yaw_deg"]))
    navigation.walk_to(chain, pre_align, beat="walk")
    t_align0 = chain.t_substep()
    rec.mark("walk", t_walk0, t_align0)
    walk_end = navigation.nav_errors(chain, pre_align)
    rec.boundary("walk->align")
    # ------------------------------------------------------------ align (+ approach)
    align_hist = navigation.align_to(chain, pre_align, beat="align")
    pre_align_err = navigation.nav_errors(chain, pre_align)
    cfg = common.cafe_m4_trajectory(1.0 / common.PHYSICS_HZ, common.M4_DECIMATION_1K)
    m4_start = approach.m4_start_pose(dict(HOME), cfg, GRIPPER_OPEN)
    bs = cfg["base_start"]
    goal = (bs["position_m"][0], bs["position_m"][1], 0.0, float(bs["yaw_rad"]))
    ref = np.load(args.reference)
    arm_ik = pour.ArmIK(chain.model)

    def lf_final(qpos):
        arm, _ = arm_ik.solve(qpos, np.asarray([m4_start[n] for n in pour.ARM]),
                              palm_pos=ref["palm_cmd"][0], palm_rot=ref["palm_cmd_rot"][0].reshape(3, 3))
        return dict(zip(pour.ARM, (float(v) for v in arm), strict=True))

    approach_report = approach.run_approach(chain, dict(HOME), goal, m4_start, lf_final_fn=lf_final)
    t_pour0 = chain.t_substep()
    rec.mark("align", t_align0, t_pour0)
    rec.boundary("align->pour")
    arrival = chain.state_dict()
    np.savez_compressed(out / "arrival_state.npz", **arrival)
    if args.stop_after == "align":
        print(json.dumps({"approach": approach_report["final"], "align": align_hist}, indent=1))
        return 0
    # ------------------------------------------------------------ pour + set-down (run_m4)
    chain.external_audit = True
    chain.beat = "pour"
    chain.phase_beat = {**{p: "pour" for p in POUR_PHASES}, **{p: "set_down" for p in SET_DOWN_PHASES}}
    pour_sub0 = chain.substep

    def on_substep():
        chain.substep += 1
        if (chain.substep - pour_sub0) % chain.decimation == 0:
            chain.observe()

    result = pour.run_pour(out / "pour", mode="retarget", reference_path=args.reference,
                           attach_env=chain.env, on_substep=on_substep)
    chain.external_audit = False
    pour_end_phase = max(float(k["end_s"]) for k in cfg["keyframes"] if k["phase"] == "pour_out")
    t_setdown0 = t_pour0 + pour_end_phase
    t_end = chain.t_substep()
    rec.mark("pour", t_pour0, t_setdown0)
    rec.mark("set_down", t_setdown0, t_end)
    rec.boundary("end")
    # beat labels for the trace rows logged during the pour window
    for r in chain.log:
        if t_pour0 < r["t"] <= t_end + 1e-9:
            r["beat"] = "pour" if r["t"] <= t_setdown0 + 1e-9 else "set_down"
            r["mode"] = "direct"
    # splice the pour's own render-rate states (with the logged stream paths) into the run
    prs = np.load(out / "pour" / "render_states.npz")
    pour_times = t_pour0 + (np.arange(len(prs["qpos"])) * world.RENDER_STRIDE + 1) * chain.dt
    qpos_all = np.vstack([np.asarray(chain.render_qpos), prs["qpos"]])
    time_all = np.concatenate([np.asarray(chain.render_time), pour_times])
    stream_all = np.full((len(qpos_all), prs["stream_paths"].shape[1], 3), np.nan)
    stream_all[len(chain.render_qpos):] = prs["stream_paths"]
    qpos_sha = common.sha256_array(qpos_all)
    torque = np.asarray(chain.torque_applied, dtype=np.float32)
    torque_sha = hashlib.sha256(np.ascontiguousarray(torque).tobytes()).hexdigest()
    np.savez_compressed(out / "torque_applied.npz", tau_applied_nm=torque,
                        joint_names=np.asarray(chain.joint_names), physics_dt_s=np.float64(chain.dt))
    # ------------------------------------------------------------ beats + metrics
    rec.beats.sort(key=lambda b: (b["t_start_s"], b["beat"] not in beats_mod.MOTION_BEATS))
    motion = [b for b in rec.beats if b["beat"] in beats_mod.MOTION_BEATS]
    others = [b for b in rec.beats if b["beat"] not in beats_mod.MOTION_BEATS]
    ordered = []
    for b in motion:
        ordered.append(b)
        ordered += [o for o in others if beats_mod.OVERLAY_HOST[o["beat"]] == b["beat"]
                    and b["t_start_s"] - 1e-9 <= o["t_start_s"] <= b["t_end_s"] + 1e-9]
    problems = beats_mod.validate(ordered)
    beats_doc = beats_mod.document(
        ordered, commit=common.commit_sha(), physics_dt_s=chain.dt, control_dt_s=chain.step_dt,
        extra={"scene": common.SCENE, "seed": args.seed, "dance": args.dance,
               "holds_s": {"exclaim": args.exclaim_s, "listen": args.listen_s,
                           "confirm": args.confirm_s},
               "simulation_only": True})
    beats_mod.write(out / "beats.json", beats_doc)
    beats_sha = common.sha256_file(out / "beats.json")
    per_beat = {}
    for b in motion:
        per_beat[b["beat"]] = {"t_start_s": b["t_start_s"], "t_end_s": b["t_end_s"],
                               **window_metrics(chain.log, b["t_start_s"], b["t_end_s"])}
    # dance area: the footprint envelope over dance + stop + jump, about the dance spot
    area = dims["dance_area"]
    sel = (time_all >= t_dance0) & (time_all <= t_look0)
    env_r, env_i = footprint.envelope_radius(chain.model, qpos_all[sel], area["centre_m"], common.SCENE)
    land = [r for r in chain.log if t_look0 - 0.3 <= r["t"] <= t_look0]
    jump = {
        "z_stand_m": jump_ref["z"],
        "apex_rise_m": max(r["z"] for r in chain.log if t_jump0 <= r["t"] <= t_look0) - jump_ref["z"],
        "touchdown_s": touchdown, "settled_s": settled,
        "touchdown_to_settled_s": settled - touchdown,
        "landing_drift_xy_m": math.hypot(np.mean([r["x"] for r in land]) - jump_ref["x"],
                                         np.mean([r["y"] for r in land]) - jump_ref["y"]),
        "landing_drift_yaw_deg": math.degrees(common.wrap(np.mean([r["yaw"] for r in land]) - jump_ref["yaw"])),
        "landing_base_radius_from_dance_spot_m": math.hypot(
            np.mean([r["x"] for r in land]) - area["centre_m"][0],
            np.mean([r["y"] for r in land]) - area["centre_m"][1]),
        "dance_stop_jump_footprint_envelope_m": env_r,
        "envelope_at_s": float(time_all[sel][env_i]),
        "dance_area_radius_m": area["radius_m"],
        "inside_dance_area": env_r <= area["radius_m"],
    }
    torque_summary = chain.torque_summary()
    whole = torque_summary["whole_run"]
    fell = sum(r["torso_on_bar"] for r in chain.log)
    pour_summary = result["summary"]
    summary = {
        "simulation_only": True,
        "water_model": "visual ballistic stream from the measured spout (run_m4); no fluid",
        "commit_sha": common.commit_sha(), "tree_dirty": common.tree_dirty(),
        "scene": common.SCENE, "seed": args.seed, "backend": "native:cpu",
        "physics_dt_s": chain.dt, "control_dt_s": chain.step_dt,
        "m4_target_dt_s": chain.dt * common.M4_DECIMATION_1K,
        "duration_s": t_end, "substeps": chain.substep,
        "events": events,
        "continuity": {
            "state_writes_after_boot": writes, "env_resets_after_boot": 0,
            "boundaries": rec.boundaries,
            "max_base_step_per_tick_m": float(max(
                math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                for a, b in zip(chain.log, chain.log[1:], strict=False))),
            "passes": writes["root"] == 0 and writes["joint"] == 0,
        },
        "beats_valid": not problems, "beat_problems": problems,
        "per_beat": per_beat,
        "jump": jump,
        "walk_end_pre_align": {"along_mm": 1e3 * walk_end[0], "lat_mm": 1e3 * walk_end[1],
                               "yaw_deg": math.degrees(walk_end[2])},
        "align_passes": align_hist,
        "pre_align_error": {"along_mm": 1e3 * pre_align_err[0], "lat_mm": 1e3 * pre_align_err[1],
                            "yaw_deg": math.degrees(pre_align_err[2])},
        "approach": approach_report,
        "falls_torso_on_bar_ticks": fell,
        "pour": {"run_m4_exit_code": result["exit_code"], "m4_passes": result["passes"],
                 "failing_gates": pour.failing_gates(pour_summary),
                 "turns": pour_summary["stream"]["turn_count"],
                 "landing_success": [pour_summary["stream"]["landing_success_count"],
                                     pour_summary["stream"]["sample_count"]],
                 "spout_height_m": [pour_summary["motion"]["spout_height_min_m"],
                                    pour_summary["motion"]["spout_height_max_m"]],
                 "set_down_error_m": pour_summary["motion"]["maximum_final_position_error_m"],
                 "free_leg_contact_rows": pour_summary["motion"]["free_leg_contact_rows"],
                 "robot_scene_contact_rows": pour_summary["motion"]["robot_scene_contact_rows"],
                 "min_pitcher_brewer_clearance_m":
                     pour_summary["motion"]["minimum_pitcher_brewer_clearance_all_phases_m"],
                 "min_robot_brewer_clearance_m":
                     pour_summary["motion"]["minimum_robot_brewer_clearance_all_phases_m"],
                 "payload_kg": pour_summary["payload"]["design_total_kg"],
                 "payload_passes": pour_summary["payload"]["passes"],
                 "torque_passes": pour_summary["torque"]["passes"],
                 "retarget": result["retarget"]},
        "torque": {"joints_audited": whole["joints_covered"], "substeps_audited": whole["substeps"],
                   "whole_run": whole, "per_beat": {k: v for k, v in torque_summary.items()
                                                    if k != "whole_run"},
                   "violations": whole["curve_violations"], "passes": whole["pass"]},
        "determinism": {"render_state_count": len(qpos_all), "qpos_sha256": qpos_sha,
                        "beats_sha256": beats_sha, "torque_sha256": torque_sha,
                        "torque_substeps": len(torque)},
        "artifacts": {"beats": str(out / "beats.json"), "trace": str(out / "trace.csv"),
                      "render_states": str(out / "render_states.npz"),
                      "torque_applied": str(out / "torque_applied.npz"),
                      "pour_summary": str(out / "pour" / "summary.json"),
                      "arrival_state": str(out / "arrival_state.npz")},
        "wall_s": time.time() - wall,
    }
    summary["passes"] = bool(
        summary["continuity"]["passes"] and summary["beats_valid"] and fell == 0
        and jump["inside_dance_area"] and result["passes"] and whole["pass"])
    np.savez_compressed(out / "render_states.npz", qpos=qpos_all.astype("<f8"),
                        time_s=time_all, stream_paths=stream_all, render_fps=np.float64(50.0),
                        qpos_sha256=np.str_(qpos_sha), commit_sha=np.str_(summary["commit_sha"]),
                        beats_sha256=np.str_(beats_sha))
    with open(out / "trace.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chain.log[0]))
        w.writeheader()
        w.writerows(chain.log)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"[chain] duration {t_end:.2f} s sim, {summary['wall_s']:.0f} s wall; beats valid "
          f"{summary['beats_valid']}; falls {fell}; state writes {writes}")
    print(f"[chain] jump envelope {env_r:.3f} m (dance_area {area['radius_m']} m); "
          f"pour M4 pass {result['passes']} {summary['pour']['failing_gates']}")
    print(f"[chain] torque whole run pass {whole['pass']}: peak {whole['peak_applied_nm']:.3f} N m, "
          f"theta {whole['max_thermal_theta']:.3f}, longest >1.2 N m {whole['max_contiguous_above_1p2_s']:.3f} s, "
          f"violations {whole['curve_violations']}")
    print(f"[chain] qpos sha256 {qpos_sha}\n[chain] beats sha256 {beats_sha}\n"
          f"[chain] torque sha256 {torque_sha}\n[chain] overall pass {summary['passes']}")
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
