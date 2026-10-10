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
    }
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
    print("targets:", {k: v for k, v in targets.items()})
    return 0 if all(targets.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
