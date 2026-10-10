#!/usr/bin/env python3
"""Independent M4 metrics computed only from a replay's raw substep logs.

This deliberately does not reuse `run_m4.py`'s stream arithmetic: it reads the
logged ACTUAL spout trajectory (`motion_substeps.csv`, every 5 ms physics
substep) and the audited actuator log (`torque_substeps.csv`) and recomputes
spiral, pour-rotation, contact/grasp, body, clearance, set-down and per-joint
torque tables.  Simulation-only evidence; no real-robot run is implied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

CONTINUOUS_NM = 1.2
PEAK_DURATION_S = 0.3
PAPER_EDGE_M = 0.038
EDGE_MARGIN_M = 0.005
GRAVITY = 9.81
CARRIED_PHASES = {"transit", "pour_in", "spiral", "pour_out", "return"}
REPO = Path(__file__).resolve().parents[2]


def _bool(value: str) -> bool:
    return value == "True"


def _longest(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def spiral_metrics(rows: list[dict], bed: np.ndarray) -> dict:
    pour = [row for row in rows if row["phase"] == "spiral"]
    xy = np.array([[float(r["spout_x_m"]), float(r["spout_y_m"])] for r in pour]) - bed[:2]
    z = np.array([float(r["spout_z_m"]) for r in pour]) - bed[2]
    radius = np.hypot(xy[:, 0], xy[:, 1])
    angle = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    angle -= angle[0]
    slope, intercept = np.polyfit(angle, radius, 1)
    fit = intercept + slope * angle
    times = [float(r["time_s"]) for r in pour]
    dt = times[1] - times[0]
    return {
        "samples": len(pour),
        "pour_duration_s": len(pour) * dt,
        "xy_radius_min_m": float(radius.min()),
        "xy_radius_max_m": float(radius.max()),
        "first_radius_m": float(radius[0]),
        "last_radius_m": float(radius[-1]),
        "unwrapped_turns": float(angle[-1] / (2 * math.pi)),
        "direction": "counter-clockwise" if angle[-1] > 0 else "clockwise",
        "radial_fit_slope_m_per_rad": float(slope),
        "radial_progression_m": float(slope * angle[-1]),
        "radial_fit_rmse_m": float(np.sqrt(np.mean((radius - fit) ** 2))),
        "max_inward_excursion_m": float(np.max(np.maximum.accumulate(radius) - radius)),
        "edge_margin_m": float(PAPER_EDGE_M - radius.max()),
        "height_above_bed_min_m": float(z.min()),
        "height_above_bed_max_m": float(z.max()),
    }


def rotation_metrics(rows: list[dict]) -> dict:
    def tilt(phase):
        return [float(r["pitcher_tilt_deg"]) for r in rows if r["phase"] == phase]

    carry = tilt("transit")
    pour_in, spiral, pour_out = tilt("pour_in"), tilt("spiral"), tilt("pour_out")
    return {
        "carry_tilt_deg": [min(carry), max(carry)] if carry else None,
        "pour_in_max_tilt_deg": max(pour_in),
        "spiral_tilt_deg": [min(spiral), max(spiral)],
        "max_pour_tilt_deg": max(pour_in + spiral),
        "final_pour_out_tilt_deg": pour_out[-1],
        "pour_out_tilt_monotone_fraction": float(
            np.mean(np.diff(pour_out) <= 1e-3)
        ) if len(pour_out) > 1 else None,
    }


def phase_tables(rows: list[dict]) -> dict:
    phases: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        phases.setdefault(row["phase"], []).append(row)
    out = OrderedDict()
    for phase, rs in phases.items():
        out[phase] = {
            "start_s": float(rs[0]["time_s"]),
            "end_s": float(rs[-1]["time_s"]),
            "both_jaw_fraction": float(np.mean(
                [_bool(r["fixed_jaw_contact"]) and _bool(r["moving_jaw_contact"]) for r in rs]
            )),
            "max_grasp_translation_m": max(float(r["grasp_translation_m"]) for r in rs),
            "max_grasp_rotation_deg": max(float(r["grasp_rotation_deg"]) for r in rs),
            "forbidden_rows": {
                key: sum(_bool(r[key]) for r in rs)
                for key in ("brewer_contact", "robot_brewer_contact", "non_claw_robot_contact",
                            "self_contact", "joint_limit_violation", "counter_contact")
            },
            "min_support_contacts": min(int(r["support_foot_contacts"]) for r in rs),
            "max_base_xy_drift_m": max(float(r["base_xy_drift_m"]) for r in rs),
            "max_base_abs_roll_deg": max(abs(float(r["base_roll_deg"])) for r in rs),
            "max_base_abs_pitch_deg": max(abs(float(r["base_pitch_deg"])) for r in rs),
            "min_brewer_clearance_m": min(float(r["brewer_clearance_m"]) for r in rs),
            "tilt_deg": [min(float(r["pitcher_tilt_deg"]) for r in rs),
                         max(float(r["pitcher_tilt_deg"]) for r in rs)],
            "pitcher_z_m_end": float(rs[-1]["pitcher_z_m"]),
            "max_pitcher_linear_speed_m_s": max(float(r["pitcher_linear_speed_m_s"]) for r in rs),
        }
    return out


def torque_table(path: Path) -> dict:
    per: dict[str, dict] = defaultdict(lambda: {
        "peak": 0.0, "sq": 0.0, "n": 0, "theta": 0.0, "violations": 0, "flags": [],
        "min_margin": math.inf, "dt": None,
    })
    with path.open() as handle:
        for row in csv.DictReader(handle):
            entry = per[row["joint"]]
            tau = abs(float(row["tau_applied_nm"]))
            entry["peak"] = max(entry["peak"], tau)
            entry["sq"] += tau * tau
            entry["n"] += 1
            entry["theta"] = max(entry["theta"], float(row["thermal_theta_post"]),
                                 float(row["thermal_theta_pre"]))
            margin = min(float(row["curve_limit_nm"]), float(row["effective_limit_nm"])) - tau
            entry["min_margin"] = min(entry["min_margin"], margin)
            entry["violations"] += margin < -1e-4
            entry["flags"].append(tau > CONTINUOUS_NM)
            entry["dt"] = float(row["dt_s"])
    out = OrderedDict()
    for joint, entry in per.items():
        dwell = _longest(entry["flags"]) * entry["dt"]
        out[joint] = {
            "peak_applied_nm": entry["peak"],
            "rms_applied_nm": math.sqrt(entry["sq"] / entry["n"]),
            "max_thermal_theta": entry["theta"],
            "max_dwell_above_1_2_nm_s": dwell,
            "min_curve_margin_nm": entry["min_margin"],
            "curve_violations": entry["violations"],
            "passes": entry["violations"] == 0 and dwell <= PEAK_DURATION_S + 1e-12
            and entry["theta"] < 1.0,
        }
    return out


def stream_metrics(rows: list[dict], stream_path: Path, bed: np.ndarray) -> dict:
    """Recompute the visual stream from the logged actual spout, every physics substep.

    The logged stream is also checked against the motion log: each emitted stream sample
    must start at exactly the spout position and velocity recorded at that substep.
    """
    pour = [row for row in rows if row["phase"] == "spiral"]
    pos = np.array([[float(r[k]) for k in ("spout_x_m", "spout_y_m", "spout_z_m")] for r in pour])
    vel = np.array([[float(r[k]) for k in ("spout_vx_m_s", "spout_vy_m_s", "spout_vz_m_s")]
                    for r in pour])
    height = pos[:, 2] - bed[2]
    valid = height > 0.0
    flight = (vel[:, 2] + np.sqrt(vel[:, 2] ** 2 + 2.0 * GRAVITY * np.maximum(height, 0.0)))
    flight /= GRAVITY
    landing = pos[:, :2] + vel[:, :2] * flight[:, None]
    span = np.linalg.norm(landing - pos[:, :2], axis=1)
    speed = np.linalg.norm(vel[:, :2], axis=1)
    landing_radius = np.linalg.norm(landing - bed[:2], axis=1)
    inside = valid & (landing_radius <= PAPER_EDGE_M - EDGE_MARGIN_M)
    by_time = {round(float(r["time_s"]), 9): r for r in pour}
    with stream_path.open() as handle:
        logged = list(csv.DictReader(handle))
    max_origin_error = max_velocity_error = 0.0
    unmatched = 0
    for sample in logged:
        source = by_time.get(round(float(sample["time_s"]), 9))
        if source is None:
            unmatched += 1
            continue
        max_origin_error = max(max_origin_error, max(
            abs(float(sample[k]) - float(source[k])) for k in ("spout_x_m", "spout_y_m", "spout_z_m")
        ))
        max_velocity_error = max(max_velocity_error, max(
            abs(float(sample[a]) - float(source[b]))
            for a, b in (("launch_vx_m_s", "spout_vx_m_s"), ("launch_vy_m_s", "spout_vy_m_s"),
                         ("launch_vz_m_s", "spout_vz_m_s"))
        ))
    return {
        "substep_samples": len(pour),
        "logged_stream_samples": len(logged),
        "logged_samples_unmatched_to_motion": unmatched,
        "logged_origin_max_abs_error_m": max_origin_error,
        "logged_launch_velocity_max_abs_error_m_s": max_velocity_error,
        "logged_unobstructed_paths": sum(_bool(s["path_unobstructed"]) for s in logged),
        "valid_intersections": int(valid.sum()),
        "landings_inside_limit": int(inside.sum()),
        "landing_rate": float(inside.mean()),
        "landing_limit_m": PAPER_EDGE_M - EDGE_MARGIN_M,
        "max_landing_radius_m": float(landing_radius.max()),
        "min_landing_edge_margin_m": float(PAPER_EDGE_M - landing_radius.max()),
        "max_horizontal_span_m": float(span.max()),
        "max_horizontal_launch_speed_m_s": float(speed.max()),
        "flight_time_s": [float(flight.min()), float(flight.max())],
    }


def clearance_and_contact_metrics(rows: list[dict]) -> dict:
    carried = [r for r in rows if r["phase"] in CARRIED_PHASES]
    final = [r for r in rows if r["phase"] == "final_settle"]
    flags = ("brewer_contact", "robot_brewer_contact", "robot_scene_contact", "free_leg_contact",
             "non_claw_robot_contact", "self_contact", "joint_limit_violation")
    return {
        "all_phase_rows_flagged": {key: sum(_bool(r[key]) for r in rows) for key in flags},
        "carried_counter_contact_rows": sum(_bool(r["counter_contact"]) for r in carried),
        "min_pitcher_brewer_clearance_all_m": min(float(r["brewer_clearance_m"]) for r in rows),
        "min_robot_brewer_clearance_all_m": min(
            float(r["robot_brewer_clearance_m"]) for r in rows
        ),
        "min_carried_pitcher_brewer_clearance_m": min(
            float(r["brewer_clearance_m"]) for r in carried
        ),
        "min_carried_robot_brewer_clearance_m": min(
            float(r["robot_brewer_clearance_m"]) for r in carried
        ),
        "min_carried_support_contacts": min(int(r["support_foot_contacts"]) for r in carried),
        "carried_both_jaw_fraction": float(np.mean([
            _bool(r["fixed_jaw_contact"]) and _bool(r["moving_jaw_contact"]) for r in carried
        ])),
        "final_settle_s": len(final) * 0.005,
        "final_max_tilt_deg": max(float(r["pitcher_tilt_deg"]) for r in final),
        "final_max_linear_speed_m_s": max(float(r["pitcher_linear_speed_m_s"]) for r in final),
        "final_max_angular_speed_rad_s": max(
            float(r["pitcher_angular_speed_rad_s"]) for r in final
        ),
        "final_all_counter_contact": all(_bool(r["counter_contact"]) for r in final),
        "final_no_robot_contact": all(
            "robot/" not in r["pitcher_contact_geoms"] for r in final
        ),
    }


def payload_metrics(path: Path) -> dict:
    source = json.loads(path.read_text("utf-8"))["objects"]
    dry = float(source["pitcher"]["dry_mass_kg"])
    water = float(source["nominal_water"]["mass_kg"])
    return {"pitcher_dry_kg": dry, "water_kg": water, "total_kg": dry + water,
            "limit_kg": 0.450}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bed", type=float, nargs=3, default=(0.0934, 0.32, 0.155))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with (args.run_dir / "motion_substeps.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    bed = np.asarray(args.bed, dtype=np.float64)
    result = {
        "source": str(args.run_dir),
        "simulation_only": True,
        "spiral": spiral_metrics(rows, bed),
        "rotation": rotation_metrics(rows),
        "phases": phase_tables(rows),
        "torque": torque_table(args.run_dir / "torque_substeps.csv"),
        "stream": stream_metrics(rows, args.run_dir / "stream_samples.csv", bed),
        "clearance_contact_set_down": clearance_and_contact_metrics(rows),
        "payload": payload_metrics(REPO / "barista" / "coffee_scene_dimensions.json"),
    }
    stream, contact = result["stream"], result["clearance_contact_set_down"]
    targets = {
        "pour_duration_15_25_s": 15.0 <= result["spiral"]["pour_duration_s"] <= 25.0,
        "height_0_020_0_050_m": result["spiral"]["height_above_bed_min_m"] >= 0.020
        and result["spiral"]["height_above_bed_max_m"] <= 0.050,
        "turns_3_4": 3.0 <= result["spiral"]["unwrapped_turns"] <= 4.0,
        "max_radius_le_0_033_m": result["spiral"]["xy_radius_max_m"] <= 0.033,
        "starts_le_0_010_m": result["spiral"]["first_radius_m"] <= 0.010,
        "progression_ge_0_015_m": result["spiral"]["radial_progression_m"] >= 0.015,
        "rmse_le_0_005_m": result["spiral"]["radial_fit_rmse_m"] <= 0.005,
        "inward_le_0_010_m": result["spiral"]["max_inward_excursion_m"] <= 0.010,
        "max_tilt_ge_12_deg": result["rotation"]["max_pour_tilt_deg"] >= 12.0,
        "pour_out_le_15_deg": result["rotation"]["final_pour_out_tilt_deg"] <= 15.0,
        "torque_all_joints": all(v["passes"] for v in result["torque"].values()),
        "torque_22_joints": len(result["torque"]) == 22,
        "stream_tied_to_measured_spout": stream["logged_samples_unmatched_to_motion"] == 0
        and stream["logged_origin_max_abs_error_m"] == 0.0
        and stream["logged_launch_velocity_max_abs_error_m_s"] == 0.0,
        "stream_paths_unobstructed": (
            stream["logged_unobstructed_paths"] == stream["logged_stream_samples"]
        ),
        "stream_intersections_valid": stream["valid_intersections"] == stream["substep_samples"],
        "stream_landing_rate_ge_0_95": stream["landing_rate"] >= 0.95,
        "stream_span_le_0_006_m": stream["max_horizontal_span_m"] <= 0.006,
        "stream_launch_speed_le_0_2_m_s": stream["max_horizontal_launch_speed_m_s"] <= 0.2,
        "no_forbidden_contact_rows": not any(contact["all_phase_rows_flagged"].values())
        and contact["carried_counter_contact_rows"] == 0,
        "nonnegative_clearances": contact["min_pitcher_brewer_clearance_all_m"] >= 0.0
        and contact["min_robot_brewer_clearance_all_m"] >= 0.0,
        "carried_stable_grasp_and_support": contact["carried_both_jaw_fraction"] == 1.0
        and contact["min_carried_support_contacts"] >= 4,
        "upright_stable_set_down": contact["final_settle_s"] >= 2.0
        and contact["final_max_tilt_deg"] <= 5.0
        and contact["final_max_linear_speed_m_s"] <= 0.01
        and contact["final_max_angular_speed_rad_s"] <= 0.1
        and contact["final_all_counter_contact"] and contact["final_no_robot_contact"],
        "payload_le_0_450_kg": result["payload"]["total_kg"] <= 0.450,
    }
    result["targets"] = targets
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", "utf-8")
    s, r = result["spiral"], result["rotation"]
    print(f"spiral: turns={s['unwrapped_turns']:.3f} r={s['xy_radius_min_m']:.4f}..{s['xy_radius_max_m']:.4f}"
          f" first={s['first_radius_m']:.4f} last={s['last_radius_m']:.4f}"
          f" progression={s['radial_progression_m']:.4f} rmse={s['radial_fit_rmse_m']:.4f}"
          f" inward={s['max_inward_excursion_m']:.4f} edge={s['edge_margin_m']:.4f}"
          f" h={s['height_above_bed_min_m']:.4f}..{s['height_above_bed_max_m']:.4f}"
          f" duration={s['pour_duration_s']:.3f}s")
    print(f"rotation: max={r['max_pour_tilt_deg']:.2f} pour_out_final={r['final_pour_out_tilt_deg']:.2f}"
          f" spiral={r['spiral_tilt_deg'][0]:.1f}..{r['spiral_tilt_deg'][1]:.1f}")
    print(f"stream: substeps={stream['substep_samples']} landing={stream['landing_rate']:.4f}"
          f" max_landing_r={stream['max_landing_radius_m']:.4f}"
          f" span<={stream['max_horizontal_span_m']:.5f} m"
          f" launch<={stream['max_horizontal_launch_speed_m_s']:.4f} m/s"
          f" origin_err={stream['logged_origin_max_abs_error_m']:.1e}")
    print(f"clearance: pitcher>={contact['min_pitcher_brewer_clearance_all_m']:.4f}"
          f" robot>={contact['min_robot_brewer_clearance_all_m']:.4f}"
          f" flagged={contact['all_phase_rows_flagged']}"
          f" final_tilt={contact['final_max_tilt_deg']:.2f} deg")
    print("targets:", {k: v for k, v in targets.items()})
    return 0 if all(targets.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
