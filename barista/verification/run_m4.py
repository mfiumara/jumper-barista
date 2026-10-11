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
CONTROL_DECIMATION = 20  # 1 kHz physics, 50 Hz targets; derived from the trajectory at run time
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
        "--save-render-states",
        action="store_true",
        help=(
            "also write render_states.npz (render-rate qpos and the logged stream paths) "
            "so barista/film/make_film.py can re-render the replay from other angles"
        ),
    )
    parser.add_argument(
        "--fast-check",
        action="store_true",
        help=(
            "run a non-rendered diagnostic bounded at the end of pour_out; "
            "this is not release evidence and does not replace the default verifier"
        ),
    )
    parser.add_argument(
        "--reachability-check",
        action="store_true",
        help=(
            "with --fast-check, gate the carried approach, representative spiral and "
            "pour-out retreat; this prerequisite does not accept final M4 stream or set-down"
        ),
    )
    parser.add_argument(
        "--spawn-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX_M", "DY_M", "DYAW_DEG"),
        help=(
            "robustness sweep: spawn the robot rigidly offset from the trajectory's "
            "base_start (world x/y in metres, yaw in degrees); the scene and every gate "
            "stay unchanged"
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


# mj_geomDistance's convex solver occasionally returns a spurious 0 (or bogus
# negative) for a pair that is centimetres apart, for one substep only. A
# non-positive reading is therefore confirmed at several distmax values and the
# largest reading kept: a genuine contact or penetration reads <= 0 at every
# distmax, so this never hides one.
CLEARANCE_CONFIRM_DISTMAX_M = (0.003, 0.01, 0.05, 0.1, 0.5, 1.0)


# The pitcher-to-robot clearance (reported, not gated) confirms on a 0.5 mm ladder instead:
# a reading capped at distmax d means "at least d", so a coarse ladder would report a
# spurious solver zero near the head as exactly 3 or 10 mm. The fine ladder keeps the
# reading a lower bound while resolving it to 0.5 mm.
PITCHER_ROBOT_CONFIRM_DISTMAX_M = tuple(
    round(0.0005 * step, 4) for step in range(1, 101)
) + (0.1, 0.5, 1.0)


def _normal_contact_force(mujoco, model, data, index: int) -> float:
    """Read the normal force of an actual contact, never a visual distance alias."""
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, wrench)
    return float(wrench[0])


def _geom_clearance(
    mujoco, model, data, geom1: int, geom2: int, fromto,
    confirm: tuple[float, ...] = CLEARANCE_CONFIRM_DISTMAX_M,
) -> tuple[float, float]:
    """Return (confirmed, raw) signed distance between two geoms."""
    raw = float(mujoco.mj_geomDistance(model, data, geom1, geom2, 1.0, fromto))
    if raw > 0.0:
        return raw, raw
    confirmed = max(
        float(mujoco.mj_geomDistance(model, data, geom1, geom2, distmax, fromto))
        for distmax in confirm
    )
    return confirmed, raw


# Pairs whose bounding spheres are further apart than this are not passed to the exact
# distance solver; their bounding-sphere gap (a lower bound on the true clearance) is used
# instead, so a reported minimum is never larger than the true minimum.
CLEARANCE_PREFILTER_M = 0.05
# The pitcher's spout, the water surface and the rigid water-mass proxy are visual or
# mass-only by scene design (since the M2 scene, eb664d8). They are listed so that any
# further physical geometry made non-collidable fails the integrity gate. Clearance to
# the brewer is still measured from every pitcher geom, these included.
EXPECTED_PITCHER_NONCOLLIDING = (
    "nominal_water_mass", "spout_lower", "spout_riser", "spout_tip", "water_surface",
)
STREAM_PATH_POINTS = 13
# The front-left limb carries the claw. It is the one limb that is never a support, so it
# is the "free leg" of the charter: it may touch only the pitcher, never the counter, the
# brewer or any other scene geometry.
FREE_LEG_PREFIX = "robot/LF_"
# Upright set-down: the released pitcher must stand within this tilt of its settled pose.
UPRIGHT_SET_DOWN_TILT_DEG = 5.0
# Pitcher-to-robot clearance (reported): the claw that holds the pitcher is excluded, and
# so are the pitcher's internal water proxies, which sit inside its body.
CLAW_PREFIXES = ("robot/LF_palm", "robot/LF_finger")
PITCHER_INTERNAL_GEOMS = ("nominal_water_mass", "water_surface")
CARRIED_PHASES = ("transit", "pour_in", "spiral", "pour_out", "return")


def _filter_compatible(model, geom1: int, geom2: int) -> bool:
    return bool(
        (model.geom_contype[geom1] & model.geom_conaffinity[geom2])
        or (model.geom_contype[geom2] & model.geom_conaffinity[geom1])
    )


def _collision_integrity(model, mujoco, robot_geoms, brewer_geoms, pitcher_geoms) -> dict:
    """Prove collisions are physically enabled rather than evaded by disabling geometry."""

    def short(index: int) -> str:
        return _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index).split("/")[-1]

    def collidable(index: int) -> bool:
        return bool(model.geom_contype[index] or model.geom_conaffinity[index])

    brewer_noncollidable = [short(index) for index in brewer_geoms if not collidable(index)]
    pitcher_noncollidable = sorted(
        short(index) for index in pitcher_geoms if not collidable(index)
    )
    pitcher_physical = [index for index in pitcher_geoms if collidable(index)]
    robot_brewer_incompatible = sum(
        not _filter_compatible(model, robot, brewer)
        for robot in robot_geoms for brewer in brewer_geoms
    )
    pitcher_brewer_incompatible = sum(
        not _filter_compatible(model, pitcher, brewer)
        for pitcher in pitcher_physical for brewer in brewer_geoms
    )
    guarded_bodies = {int(model.geom_bodyid[index]) for index in brewer_geoms}
    excluded = []
    for signature in np.asarray(model.exclude_signature).ravel():
        body1, body2 = int(signature) >> 16, int(signature) & 0xFFFF
        if body1 in guarded_bodies or body2 in guarded_bodies:
            excluded.append([
                _name(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, body1),
                _name(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, body2),
            ])
    contact_disabled = bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
    checks = {
        "contact_enabled": not contact_disabled,
        "every_brewer_geom_collidable": not brewer_noncollidable and bool(brewer_geoms),
        "pitcher_noncollidable_set_unchanged": sorted(
            next((name for name in EXPECTED_PITCHER_NONCOLLIDING if geom.endswith(name)), geom)
            for geom in pitcher_noncollidable
        ) == sorted(EXPECTED_PITCHER_NONCOLLIDING),
        "robot_brewer_filters_compatible": robot_brewer_incompatible == 0,
        "pitcher_brewer_filters_compatible": pitcher_brewer_incompatible == 0,
        "no_brewer_contact_exclusions": not excluded,
    }
    return {
        "robot_collision_geom_count": len(robot_geoms),
        "brewer_geom_count": len(brewer_geoms),
        "pitcher_collidable_geoms": sorted(short(index) for index in pitcher_physical),
        "pitcher_noncollidable_geoms": pitcher_noncollidable,
        "brewer_noncollidable_geoms": brewer_noncollidable,
        "robot_brewer_incompatible_pairs": robot_brewer_incompatible,
        "pitcher_brewer_incompatible_pairs": pitcher_brewer_incompatible,
        "brewer_contact_exclusions": excluded,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _set_clearance(
    mujoco, model, data, set_a: np.ndarray, set_b: np.ndarray, fromto,
    confirm: tuple[float, ...] = CLEARANCE_CONFIRM_DISTMAX_M,
) -> tuple[float, int, int]:
    """Lower-bound-safe minimum signed distance between two geom sets.

    Near pairs (bounding-sphere gap <= CLEARANCE_PREFILTER_M) are measured exactly with
    the confirmed solver; far pairs contribute their bounding-sphere gap, which is never
    larger than the true distance.
    """
    gap = (
        np.linalg.norm(data.geom_xpos[set_a][:, None, :] - data.geom_xpos[set_b][None], axis=2)
        - model.geom_rbound[set_a][:, None] - model.geom_rbound[set_b][None]
    )
    best, best_a, best_b = math.inf, -1, -1
    for row, col in zip(*np.nonzero(gap <= CLEARANCE_PREFILTER_M), strict=True):
        confirmed, _ = _geom_clearance(
            mujoco, model, data, int(set_a[row]), int(set_b[col]), fromto, confirm
        )
        if confirmed < best:
            best, best_a, best_b = confirmed, int(set_a[row]), int(set_b[col])
    far = gap > CLEARANCE_PREFILTER_M
    if far.any():
        far_gap = np.where(far, gap, np.inf)
        row, col = np.unravel_index(int(np.argmin(far_gap)), far_gap.shape)
        if far_gap[row, col] < best:
            best, best_a, best_b = float(far_gap[row, col]), int(set_a[row]), int(set_b[col])
    return best, best_a, best_b


def _stream_path(
    start: np.ndarray, velocity: np.ndarray, flight_s: float, gravity: float
) -> list[np.ndarray]:
    """Gravity-only ballistic path from the measured spout pose and velocity."""
    points = []
    for fraction in np.linspace(0.0, 1.0, STREAM_PATH_POINTS):
        time_s = fraction * flight_s
        point = start + velocity * time_s
        point[2] -= 0.5 * gravity * time_s * time_s
        points.append(point)
    return points


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


def _blend_from(config: dict, time_s: float, start_arm: np.ndarray) -> np.ndarray:
    """Interpolate the current keyframe segment from a measured start pose.

    Used after a contact-triggered set-down, where the arm is held at its measured
    contact pose rather than the nominal keyframe: starting the next segment from the
    nominal pose would step the joint targets and shove the released pitcher.
    """
    previous_time = 0.0
    for keyframe in config["keyframes"]:
        end = float(keyframe["end_s"])
        if time_s <= end:
            fraction = _smooth((time_s - previous_time) / max(end - previous_time, 1e-12))
            arm = np.asarray(keyframe["arm_rad"], dtype=np.float64)
            return start_arm + (arm - start_arm) * fraction
        previous_time = end
    return np.asarray(config["keyframes"][-1]["arm_rad"], dtype=np.float64)


def _segment_fraction(config: dict, time_s: float) -> float:
    """Smoothed progress through the keyframe segment containing ``time_s``."""
    previous_time = 0.0
    for keyframe in config["keyframes"]:
        end = float(keyframe["end_s"])
        if time_s <= end:
            return _smooth((time_s - previous_time) / max(end - previous_time, 1e-12))
        previous_time = end
    return 1.0


def _key_frame_times(config: dict, frames: tuple[Path, Path, Path, Path]) -> list:
    """Still times from the keyframe phases: start, mid-spiral, outer spiral, set-down."""
    def span(phase: str) -> tuple[float, float]:
        previous, start, end = 0.0, None, None
        for keyframe in config["keyframes"]:
            if keyframe["phase"] == phase:
                start = previous if start is None else start
                end = float(keyframe["end_s"])
            previous = float(keyframe["end_s"])
        if start is None or end is None:
            raise RuntimeError(f"trajectory has no {phase!r} phase")
        return start, end

    spiral_start, spiral_end = span("spiral")
    settle_start, settle_end = span("final_settle")
    times = (
        0.5, 0.5 * (spiral_start + spiral_end), spiral_end - 3.0,
        min(settle_start + 1.0, settle_end - 0.1),
    )
    return list(zip(times, frames, strict=True))


def _load_sibling(stem: str):
    """Import a module next to this file by path (``barista/`` is not on ``sys.path``)."""
    import importlib.util

    name = f"barista_verification_{stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{stem}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _apply_grip_contact(models: list, mujoco, spec: dict | None) -> dict | None:
    """Optional grip-contact impedance from the trajectory (see its ``grip_contact`` note).

    Only the named pitcher grip geoms change, and only ``solimp`` (how hard the contact
    is) and ``solmix`` (whose impedance a contact pair uses): no friction coefficient, no
    gate. Returns what was applied, for the summary.
    """
    if not spec:
        return None
    solimp = [float(value) for value in spec["solimp"]]
    solmix = float(spec.get("solmix", 1.0))
    applied = {}
    for suffix in spec["geoms"]:
        index = _ending(models[0], mujoco, mujoco.mjtObj.mjOBJ_GEOM, "ngeom", suffix)
        applied[suffix] = {
            "solimp_before": models[0].geom_solimp[index].tolist(),
            "solmix_before": float(models[0].geom_solmix[index]),
        }
        for model in models:
            # solimp is (dmin, dmax, width, midpoint, power); unspecified tail values stay.
            model.geom_solimp[index, : len(solimp)] = solimp
            model.geom_solmix[index] = solmix
    return {
        "solimp": solimp, "solmix": solmix, "geoms": applied, "model_copies": len(models),
        "note": spec.get("note", ""),
    }


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
    stream_paths: list[list[np.ndarray] | None],
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
        for state, points in zip(states, stream_paths, strict=True):
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            # The drawn stream is exactly the logged ballistic path launched from the
            # measured spout pose and velocity at this substep (simulation-only visual).
            if points is not None:
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
            "rms_applied_nm": math.sqrt(
                sum(sample["tau_applied_nm"] ** 2 for sample in samples) / len(samples)
            ),
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


def _pitcher_robot_clearance_summary(rows: list[dict]) -> dict:
    """Minimum carried pitcher-to-robot (non-claw) clearance, overall and per phase.

    Reported, not gated: the gate for actual contact is
    ``no_brewer_or_carried_forbidden_contact``. This is the margin before that gate trips.
    """
    out: dict[str, dict] = {}
    for scope in ("carried", *CARRIED_PHASES):
        scoped = [
            row for row in rows
            if row["phase"] in CARRIED_PHASES and scope in ("carried", row["phase"])
        ]
        if not scoped:
            continue
        nearest = min(scoped, key=lambda row: row["pitcher_robot_clearance_m"])
        out[scope] = {
            "minimum_m": nearest["pitcher_robot_clearance_m"],
            "nearest_pair": nearest["pitcher_robot_nearest_pair"],
            "time_s": nearest["time_s"],
        }
    return {
        "note": (
            "carried pitcher geoms incl. the visual spout (not the internal water proxies) "
            "vs every robot collision geom except the LF claw (LF_palm*, LF_finger*)"
        ),
        "by_scope": out,
    }


class _TorqueLog:
    """Streams the per-substep torque audit to CSV and keeps only its running summary.

    At 1 kHz physics the audit is 22 rows per substep (about 1.25 M rows for the full M4
    replay); held as dicts that is several GB per process, which made parallel sweeps
    thrash. The CSV bytes and the summary are exactly those of writing every row at the end
    and calling :func:`_summarize_torque` on them (same order, same arithmetic).
    """

    def __init__(self, path: Path, dt_s: float) -> None:
        self.path = path
        self.dt_s = dt_s
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter | None = None
        self.count = 0
        self.groups: dict[tuple[str, str], dict] = {}

    def add(self, row: dict) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1
        for key in ((row["joint"], "whole_run"), (row["joint"], row["phase"])):
            group = self.groups.get(key)
            if group is None:
                group = self.groups[key] = {
                    "n": 0, "speed": -math.inf, "raw": -math.inf, "applied": -math.inf,
                    "squares": [], "curve": math.inf, "effective": math.inf,
                    "theta": -math.inf, "above": 0, "streak": 0, "longest": 0, "derated": 0,
                    "first_violation": None, "thermal_error": -math.inf,
                }
            group["n"] += 1
            group["speed"] = max(group["speed"], abs(row["qd_pre_rad_s"]))
            group["raw"] = max(group["raw"], abs(row["tau_raw_nm"]))
            group["applied"] = max(group["applied"], abs(row["tau_applied_nm"]))
            # Kept as floats and summed once with sum(), whose (compensated) result is the
            # one _summarize_torque reports.
            group["squares"].append(row["tau_applied_nm"] ** 2)
            group["curve"] = min(group["curve"], row["curve_margin_nm"])
            group["effective"] = min(group["effective"], row["effective_margin_nm"])
            group["theta"] = max(
                group["theta"], max(row["thermal_theta_pre"], row["thermal_theta_post"])
            )
            if row["above_continuous"]:
                group["above"] += 1
                group["streak"] += 1
                group["longest"] = max(group["longest"], group["streak"])
            else:
                group["streak"] = 0
            group["derated"] += row["derating_active"]
            if row["curve_violation"] and group["first_violation"] is None:
                group["first_violation"] = row["time_s"]
            group["thermal_error"] = max(group["thermal_error"], row["thermal_recompute_error"])

    def close(self) -> None:
        self.handle.close()

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for (joint, phase), group in self.groups.items():
            out[f"{joint}:{phase}"] = {
                "joint": joint,
                "phase": phase,
                "sample_count": group["n"],
                "peak_speed_rad_s": group["speed"],
                "peak_raw_abs_nm": group["raw"],
                "peak_applied_abs_nm": group["applied"],
                "rms_applied_nm": math.sqrt(sum(group["squares"]) / group["n"]),
                "minimum_curve_margin_nm": group["curve"],
                "minimum_effective_margin_nm": group["effective"],
                "maximum_thermal_theta": group["theta"],
                "above_continuous_count": group["above"],
                "above_continuous_total_s": group["above"] * self.dt_s,
                "above_continuous_max_contiguous_s": group["longest"] * self.dt_s,
                "derated_substep_count": group["derated"],
                "first_violation_time_s": group["first_violation"],
                "maximum_thermal_recompute_error": group["thermal_error"],
            }
        return out


def _json_default(value):
    """Turn NumPy scalar reductions into ordinary JSON scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> int:
    args = _parser().parse_args()
    if args.reachability_check and not args.fast_check:
        raise SystemExit("--reachability-check requires --fast-check")
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
    # The physics rate is the trajectory's: M4 runs at 1 kHz physics with 50 Hz keyframe
    # targets so it chains with the 1 kHz locomotion policies (MOTION_CHAIN.md R3).
    env_cfg.sim.mujoco.timestep = float(config["physics_dt_s"])
    scenes.apply(env_cfg, "coffee_bar")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    env.reset()
    model = env.sim.mj_model
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = PROP_IMPRATIO
    if PROP_CONE != "elliptic":
        raise RuntimeError(f"expected the measured grasp cone to be elliptic, got {PROP_CONE}")
    # The native backend steps per-environment model copies when domain randomisation is
    # on, so a model edit has to reach those copies too.
    grip_contact = _apply_grip_contact(
        [model, *(getattr(env.sim, "_models", None) or [])], mujoco, config.get("grip_contact")
    )
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
    spawn_dx, spawn_dy, spawn_dyaw_deg = (float(value) for value in args.spawn_offset)
    yaw = float(base_start["yaw_rad"]) + math.radians(spawn_dyaw_deg)
    spawn_position = [
        float(base_start["position_m"][0]) + spawn_dx,
        float(base_start["position_m"][1]) + spawn_dy,
        float(base_start["position_m"][2]),
    ]
    root_pose = torch.tensor(
        [[*spawn_position, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
        dtype=torch.float,
        device=resolution.device,
    )
    runtime_joint_col = {name: index for index, name in enumerate(robot.joint_names)}
    if set(runtime_joint_col) != set(HOME):
        raise RuntimeError("barista runtime joint set differs from the 22-joint hardware order")
    # support_base_rad is blended in from HOME during lift (4-6 s), so the initial state stays
    # at HOME; starting at the blended end value made RF_J4 snap back at t=0.
    initial = np.asarray([HOME[name] for name in robot.joint_names], dtype=np.float64)
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
    # A robot spawned (or arrived) off base_start has its keyframes re-targeted by IK so the
    # pour stays in the brew station's frame; exactly at base_start nothing changes.
    retarget = _load_sibling("m4_retarget").StanceRetarget(
        model, mujoco, base_start, config.get("set_down_servo"),
        config.get("spiral_radius_scale"),
    )
    retarget.measure(data)
    if retarget.active:
        corrected = retarget.correct(
            data, config["keyframes"][0]["phase"], config["keyframes"][0]["arm_rad"]
        )
        for name, value in zip(arm_names, corrected, strict=True):
            initial[runtime_joint_col[name]] = value
        initial_t = torch.tensor(initial[None, :], dtype=torch.float, device=resolution.device)
        robot.write_joint_state_to_sim(initial_t, torch.zeros_like(initial_t))
        robot.set_joint_position_target(initial_t)
        env.sim.forward()
        env.scene.update(dt=0.0)

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
    robot_collision_geoms = sorted(
        index for index in range(model.ngeom)
        if _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index).startswith("robot/")
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    )
    collision_integrity = _collision_integrity(
        model, mujoco, robot_collision_geoms, sorted(brewer_geoms), sorted(pitcher_geoms)
    )
    robot_geom_array = np.asarray(robot_collision_geoms, dtype=np.int64)
    brewer_geom_array = np.asarray(sorted(brewer_geoms), dtype=np.int64)
    # Reported (not gated) clearance between the carried pitcher, including its visual
    # spout, and every robot collision geom that is not the claw holding it: the
    # display_module/head, the trunk, the support legs and the LF arm links.
    pitcher_shape_array = np.asarray(sorted(
        index for index in pitcher_geoms
        if not _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index).endswith(
            PITCHER_INTERNAL_GEOMS
        )
    ), dtype=np.int64)
    robot_nonclaw_array = np.asarray([
        index for index in robot_collision_geoms
        if not _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index).startswith(CLAW_PREFIXES)
    ], dtype=np.int64)
    pair_clearance = _load_sibling("m4_pair_clearance").PairClearance(
        [index for index in range(model.ngeom)
         if _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index).startswith("robot/")],
        pitcher_shape_array,
        {index: _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, index)
         for index in range(model.ngeom)},
    )

    nominal_dt_s = float(config["physics_dt_s"])
    dt_s = float(model.opt.timestep)
    if abs(dt_s - nominal_dt_s) > 1e-12:
        raise RuntimeError(f"compiled dt {dt_s} differs from trajectory dt")
    control_decimation = round(float(config["control_dt_s"]) / dt_s)
    control_dt_s = dt_s * control_decimation
    if abs(control_dt_s - float(config["control_dt_s"])) > 1e-12:
        raise RuntimeError(f"control dt {control_dt_s} differs from trajectory dt")
    full_duration_s = float(config["keyframes"][-1]["end_s"])
    duration_s = (
        max(
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

    torque_log = _TorqueLog(out / "torque_substeps.csv", dt_s)
    motion_rows: list[dict] = []
    stream_rows: list[dict] = []
    render_states: list[np.ndarray] = []
    corrected_clearance_readings: list[dict] = []
    render_stream_paths: list[list[np.ndarray] | None] = []
    ray_sanity_geom: str | None = None
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
    # Grip preload: the claw's squeeze lead (the servo pushes with stiffness x lead once the
    # finger is stopped by the pitcher). The trajectory may raise it within the servo's
    # continuous rating; the torque audit below still gates every substep.
    squeeze_lead = float(config.get("grip_squeeze_lead_rad", SQUEEZE_LEAD))
    stream_cfg = config["stream"]
    bed_centre = np.asarray(stream_cfg["bed_centre_m"], dtype=np.float64)
    # A landing is accepted only inside the acceptance radius less the required edge margin.
    landing_limit_m = float(stream_cfg["bed_acceptance_radius_m"]) - float(
        stream_cfg["required_edge_margin_m"]
    )
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
            if step % round(5.0 / dt_s) == 0:
                print(f"[m4] t={time_s:.0f} s ({phase})", flush=True)
            if step % control_decimation == 0:
                _, arm, finger_request = _target(config, time_s)
                keyframe_arm = arm
                arm = retarget.correct(data, phase, keyframe_arm, time_s)
                if set_down_contact_seen and phase in ("lower", "set_down", "release"):
                    assert set_down_arm is not None
                    arm = set_down_arm
                    finger_request = -1.4
                elif set_down_contact_seen and phase == "clear_retreat":
                    assert set_down_arm is not None
                    # Blend from the held contact arm toward the re-targeted keyframe.
                    arm = _blend_from(config, time_s, set_down_arm) + (
                        arm - keyframe_arm
                    ) * _segment_fraction(config, time_s)
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
                        finger_q + squeeze_lead + LEAD_PER_SPEED * max(finger_qd, 0.0),
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
                torque_log.add({
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
            robot_brewer_contact = robot_scene_contact = free_leg_contact = False
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
                for robot_name, other_geom, other_name in (
                    (name1, geom2, name2), (name2, geom1, name1)
                ):
                    if not robot_name.startswith("robot/") or other_name.startswith("robot/"):
                        continue
                    if other_geom not in pitcher_geoms and other_geom != terrain:
                        robot_scene_contact = True
                    if robot_name.startswith(FREE_LEG_PREFIX) and other_geom not in pitcher_geoms:
                        free_leg_contact = True
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
                    confirmed, raw = _geom_clearance(
                        mujoco, model, data, pitcher_geom, brewer_geom, fromto
                    )
                    if confirmed != raw:
                        corrected_clearance_readings.append({
                            "substep": step, "phase": phase,
                            "pitcher_geom": _name(
                                model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, pitcher_geom
                            ),
                            "brewer_geom": _name(
                                model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, brewer_geom
                            ),
                            "raw_m": raw, "confirmed_m": confirmed,
                        })
                    clearance = min(clearance, confirmed)
            robot_clearance, robot_near, brewer_near = _set_clearance(
                mujoco, model, data, robot_geom_array, brewer_geom_array, fromto
            )
            if phase in CARRIED_PHASES:
                pitcher_robot_clearance, robot_body_near, pitcher_near = _set_clearance(
                    mujoco, model, data, robot_nonclaw_array, pitcher_shape_array, fromto,
                    PITCHER_ROBOT_CONFIRM_DISTMAX_M,
                )
                pitcher_robot_pair = (
                    f"{_name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, robot_body_near)};"
                    f"{_name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, pitcher_near)}"
                )
            else:
                pitcher_robot_clearance, pitcher_robot_pair = math.inf, ""

            if phase == "spiral":
                pair_clearance.observe(
                    model, data, time_s + dt_s,
                    lambda robot_geom, pitcher_geom, fromto=fromto: _geom_clearance(
                        mujoco, model, data, robot_geom, pitcher_geom, fromto,
                    )[0],
                    CLEARANCE_PREFILTER_M,
                    contact_force=lambda index: _normal_contact_force(mujoco, model, data, index),
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
                "pitcher_robot_clearance_m": pitcher_robot_clearance,
                "pitcher_robot_nearest_pair": pitcher_robot_pair,
                "robot_brewer_clearance_m": robot_clearance,
                "robot_brewer_nearest_pair": (
                    f"{_name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, robot_near)};"
                    f"{_name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, brewer_near)}"
                ),
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
                "robot_scene_contact": robot_scene_contact,
                "free_leg_contact": free_leg_contact,
                "non_claw_robot_contact": non_claw_robot_contact,
                "self_contact": self_contact, "joint_limit_violation": joint_limit_violation,
                "pitcher_contact_geoms": ";".join(sorted(set(contact_names))),
            })
            landing = None
            stream_points = None
            if phase == "spiral" and step % control_decimation == 0:
                # The massless visual stream inherits the measured outlet velocity and then
                # follows a gravity-only ballistic path to the bed. Both its origin and landing
                # are therefore driven by the actual spout, never a prescribed target spiral.
                gravity = float(stream_cfg["gravity_m_s2"])
                height = float(spout_pos[2] - bed_centre[2])
                intersects = height > 0.0
                flight_s = (
                    (
                        float(spout_linear_velocity[2])
                        + math.sqrt(float(spout_linear_velocity[2]) ** 2 + 2.0 * gravity * height)
                    ) / gravity
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
                # The drawn path must reach the bed through open air: every segment is ray
                # cast against all scene and robot geometry except the pitcher it leaves.
                obstruction = ""
                if intersects:
                    stream_points = _stream_path(
                        spout_pos.copy(), spout_linear_velocity.copy(), flight_s, gravity
                    )
                    geom_hit = np.zeros(1, dtype=np.int32)
                    for first, second in itertools.pairwise(stream_points):
                        fraction = float(mujoco.mj_ray(
                            model, data, first, second - first, None, 1, pitcher_body, geom_hit
                        ))
                        if 0.0 <= fraction < 1.0 - 1e-6:
                            obstruction = _name(
                                model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, int(geom_hit[0])
                            )
                            break
                path_clear = intersects and not obstruction
                if ray_sanity_geom is None:
                    # Control: the same ray cast straight down through the bed must hit real
                    # scene geometry, proving the obstruction check can see the brewer.
                    geom_hit = np.zeros(1, dtype=np.int32)
                    mujoco.mj_ray(
                        model, data, spout_pos.copy(), np.asarray([0.0, 0.0, -1.0]),
                        None, 1, pitcher_body, geom_hit,
                    )
                    ray_sanity_geom = (
                        _name(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, int(geom_hit[0]))
                        if geom_hit[0] >= 0 else ""
                    )
                accepted = path_clear and landing_radius <= landing_limit_m
                if accepted:
                    miss_reason = ""
                elif not intersects:
                    miss_reason = "spout not above bed"
                elif obstruction:
                    miss_reason = f"path obstructed by {obstruction}"
                else:
                    miss_reason = "outside acceptance disk less edge margin"
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
                    "landing_edge_margin_m": (
                        float(stream_cfg["bed_acceptance_radius_m"]) - landing_radius
                    ),
                    "path_unobstructed": path_clear, "path_obstruction_geom": obstruction,
                    "unwrapped_angle_rad": angle, "angle_rad": angle,
                    "accepted": accepted,
                    "miss_reason": miss_reason,
                })
            if step % render_stride == 0:
                render_states.append(data.qpos.copy())
                render_stream_paths.append(
                    None if stream_points is None else [p.copy() for p in stream_points]
                )
    finally:
        torque_log.close()
        env.close()

    if settled_pitcher_z is None:
        raise RuntimeError("trajectory ended before the settled reference sample")

    torque_csv = out / "torque_substeps.csv"
    motion_csv = out / "motion_substeps.csv"
    stream_csv = out / "stream_samples.csv"
    for path, rows in ((motion_csv, motion_rows), (stream_csv, stream_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    torque_summary = torque_log.summary()
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
    reachability_phases = {"transit", "pour_in", "spiral", "pour_out"}
    reachability_rows = [
        row for row in motion_rows if row["phase"] in reachability_phases
    ]
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
    # The paper-edge margin is taken from whichever is further out: the actual spout or
    # where its stream lands.
    edge_margin = float(stream_cfg["bed_acceptance_radius_m"]) - max(
        float(np.max(radii)), maximum_landing_radius
    )
    final_maximum_tilt = (
        max(row["pitcher_tilt_deg"] for row in final_rows) if final_rows else math.nan
    )
    minimum_robot_brewer_clearance = min(
        row["robot_brewer_clearance_m"] for row in motion_rows
    )
    nearest_robot_row = min(motion_rows, key=lambda row: row["robot_brewer_clearance_m"])

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
        "upright_final_set_down": (
            bool(final_rows) and final_maximum_tilt <= UPRIGHT_SET_DOWN_TILT_DEG
        ),
        "no_free_leg_or_robot_scene_contact": not any(
            row["free_leg_contact"] or row["robot_scene_contact"] for row in motion_rows
        ),
        "nonnegative_robot_brewer_clearance": (
            minimum_robot_brewer_clearance >= threshold["minimum_brewer_clearance_m"]
        ),
        "carried_robot_and_pitcher_brewer_clearance": min(
            min(row["brewer_clearance_m"], row["robot_brewer_clearance_m"])
            for row in carried_rows
        ) >= threshold["minimum_brewer_clearance_m"],
        "physical_collision_geometry_intact": collision_integrity["passes"],
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
        "all_paths_unobstructed": all(row["path_unobstructed"] for row in stream_rows),
        "obstruction_ray_sees_brewer": any(
            token in (ray_sanity_geom or "") for token in ("server_", "v60_")
        ),
        "plausible_horizontal_span": (
            maximum_horizontal_span <= threshold["maximum_horizontal_span_m"]
        ),
        "plausible_horizontal_launch_speed": (
            maximum_horizontal_launch_speed
            <= threshold["maximum_horizontal_launch_speed_m_s"]
        ),
        "no_emission_after_pour_out": all(row["phase"] == "spiral" for row in stream_rows),
    }
    reachability_checks = {
        "physical_pickup_and_hold": (
            valid_hold_duration >= float(threshold["m3_hold_s"])
            and fixed_before_lift
            and moving_before_lift
        ),
        "actual_spout_duration": 15.0 <= len(pour_rows) * dt_s <= 25.0,
        "actual_spout_turn_count": 3.0 <= turns <= 4.0,
        "actual_spout_radius": float(np.max(radii)) <= 0.033,
        "actual_spout_height": 0.020 <= spout_min and spout_max <= 0.050,
        "opposing_jaw_contacts": all(
            row["fixed_jaw_contact"] and row["moving_jaw_contact"]
            for row in reachability_rows
        ),
        "carried_grasp_rotation": max(
            row["grasp_rotation_deg"] for row in reachability_rows
        ) <= threshold["carried_grasp_rotation_deg"],
        "no_forbidden_contacts": not any(
            row["counter_contact"]
            or row["brewer_contact"]
            or row["robot_brewer_contact"]
            or row["non_claw_robot_contact"]
            or row["self_contact"]
            or row["joint_limit_violation"]
            for row in reachability_rows
        ),
        "nonnegative_brewer_clearance": min(
            min(row["brewer_clearance_m"], row["robot_brewer_clearance_m"])
            for row in reachability_rows
        ) >= 0.0,
        "no_free_leg_or_robot_scene_contact": not any(
            row["free_leg_contact"] or row["robot_scene_contact"]
            for row in reachability_rows
        ),
        "physical_collision_geometry_intact": collision_integrity["passes"],
        "support_stable": min(
            row["support_foot_contacts"] for row in reachability_rows
        ) >= 4,
        "base_stable": (
            max(row["base_xy_drift_m"] for row in reachability_rows)
            <= threshold["base_xy_drift_m"]
            and max(
                max(abs(row["base_roll_deg"]), abs(row["base_pitch_deg"]))
                for row in reachability_rows
            ) <= threshold["base_roll_pitch_deg"]
        ),
        "torque_speed_thermal": torque_pass,
    }
    # Reported, not gated here: the full M4 verifier gates this, and it is not one of the
    # reachability acceptance criteria. Keeping it visible stops it being lost before M4.
    reachability_carried_translation = max(
        row["grasp_translation_m"] for row in reachability_rows
    )
    reachability_full_m4_risks = {
        "carried_grasp_translation_m": reachability_carried_translation,
        "full_m4_threshold_m": threshold["grasp_translation_m"],
        "within_full_m4_threshold": (
            reachability_carried_translation <= threshold["grasp_translation_m"]
        ),
    }

    payload = _payload(model, mujoco, pitcher_body)
    qpos_array = np.asarray(render_states)
    state_sha256 = hashlib.sha256(qpos_array.astype("<f8").tobytes()).hexdigest()
    video = out / "m4_spiral_pour.mp4"
    start_frame = out / "m4_start.png"
    mid_spiral_frame = out / "m4_mid_spiral.png"
    spiral_frame = out / "m4_outer_spiral.png"
    set_down_frame = out / "m4_set_down.png"
    render_enabled = not args.no_render and not args.fast_check
    render_states_path = out / "render_states.npz"
    if args.save_render_states:
        # Pure export of the already-logged render-rate state; it changes no gate,
        # log or hash above. NaN rows mean "no stream drawn at this frame".
        stream_array = np.full(
            (len(render_stream_paths), STREAM_PATH_POINTS, 3), np.nan, dtype=np.float64
        )
        for index, points in enumerate(render_stream_paths):
            if points is not None:
                stream_array[index] = np.asarray(points, dtype=np.float64)
        np.savez_compressed(
            render_states_path,
            qpos=qpos_array.astype("<f8"),
            stream_paths=stream_array,
            render_fps=np.float64(config["render_fps"]),
            qpos_sha256=np.str_(state_sha256),
            commit_sha=np.str_(commit_sha),
            trajectory_sha256=np.str_(config_sha256),
        )
    if render_enabled:
        _render(
            model, mujoco, qpos_array, render_stream_paths, camera,
            float(config["render_fps"]), video,
        )
        for timestamp, frame in _key_frame_times(config, (
            start_frame, mid_spiral_frame, spiral_frame, set_down_frame,
        )):
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", str(frame),
            ], check=True)

    summary = {
        "simulation_only": True,
        "evidence_class": (
            "m4_reachability_prerequisite_fast_check"
            if args.reachability_check
            else "diagnostic_non_release_fast_check"
            if args.fast_check
            else "full_m4_verification"
        ),
        "release_evidence": not args.fast_check,
        "fast_check": args.fast_check,
        "reachability_check": args.reachability_check,
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
        "stance_retarget": retarget.report(),
        "grip_contact": grip_contact,
        "grip_squeeze_lead_rad": squeeze_lead,
        "spawn_offset": {
            "dx_m": spawn_dx, "dy_m": spawn_dy, "dyaw_deg": spawn_dyaw_deg,
            "nominal": not (spawn_dx or spawn_dy or spawn_dyaw_deg),
        },
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
            "minimum_carried_robot_brewer_clearance_m": min(
                row["robot_brewer_clearance_m"] for row in carried_rows
            ),
            "minimum_pitcher_brewer_clearance_all_phases_m": min(
                row["brewer_clearance_m"] for row in motion_rows
            ),
            "pitcher_robot_clearance": _pitcher_robot_clearance_summary(motion_rows),
            "pitcher_robot_pair_clearance": pair_clearance.report(),
            "minimum_robot_brewer_clearance_all_phases_m": minimum_robot_brewer_clearance,
            "nearest_robot_brewer_pair": nearest_robot_row["robot_brewer_nearest_pair"],
            "nearest_robot_brewer_time_s": nearest_robot_row["time_s"],
            "free_leg_contact_rows": sum(row["free_leg_contact"] for row in motion_rows),
            "robot_scene_contact_rows": sum(row["robot_scene_contact"] for row in motion_rows),
            "maximum_final_tilt_deg": final_maximum_tilt,
            "upright_set_down_tilt_limit_deg": UPRIGHT_SET_DOWN_TILT_DEG,
            "collision_integrity": collision_integrity,
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
            "landing_acceptance_limit_m": landing_limit_m,
            "obstructed_path_count": sum(
                not row["path_unobstructed"] for row in stream_rows
            ),
            "obstruction_ray_control_hit_geom": ray_sanity_geom,
            "maximum_horizontal_span_m": maximum_horizontal_span,
            "maximum_horizontal_launch_speed_m_s": maximum_horizontal_launch_speed,
            "landing_success_count": len(stream_rows) - len(misses),
            "landing_miss_count": len(misses), "landing_success_rate": landing_success_rate,
            "misses": misses, "checks": stream_passes,
        },
        "reachability": {
            "scope_phases": sorted(reachability_phases),
            "minimum_brewer_clearance_m": min(
                row["brewer_clearance_m"] for row in reachability_rows
            ),
            "minimum_support_contacts": min(
                row["support_foot_contacts"] for row in reachability_rows
            ),
            "maximum_base_xy_drift_m": max(
                row["base_xy_drift_m"] for row in reachability_rows
            ),
            "maximum_base_abs_roll_pitch_deg": max(
                max(abs(row["base_roll_deg"]), abs(row["base_pitch_deg"]))
                for row in reachability_rows
            ),
            "checks": reachability_checks,
            "full_m4_open_risks": reachability_full_m4_risks,
            "passes": payload["passes"] and all(reachability_checks.values()),
        },
        "torque": {
            "units": {"q": "rad", "qd": "rad/s", "tau": "N m", "time": "s"},
            "row_count": torque_log.count, "rows_per_substep": len(joint_order),
            "joint_count": len(joint_order), "curve_tolerance_nm": 1e-4,
            "thermal_tolerance": 1e-5, "passes": torque_pass,
            "per_joint_and_phase": torque_summary,
        },
        "brewer_clearance_solver_corrections": {
            "note": (
                "non-positive mj_geomDistance readings re-measured at several distmax "
                "values; the largest reading is kept (genuine contacts stay <= 0)"
            ),
            "count": len(corrected_clearance_readings),
            "readings": corrected_clearance_readings[:50],
        },
        "determinism": {
            "render_state_count": len(render_states), "qpos_sha256": state_sha256,
        },
        "artifacts": {
            "torque_csv": str(torque_csv), "motion_csv": str(motion_csv),
            "stream_csv": str(stream_csv), "video": str(video) if render_enabled else None,
            "start_frame": str(start_frame) if render_enabled else None,
            "mid_spiral_frame": str(mid_spiral_frame) if render_enabled else None,
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
    if args.reachability_check:
        print(
            "[m4] reachability prerequisite pass="
            f"{summary['reachability']['passes']} (not final M4 acceptance)"
        )
        if not reachability_full_m4_risks["within_full_m4_threshold"]:
            print(
                "[m4] WARNING full-M4 risk: carried grasp translation="
                f"{reachability_carried_translation:.6f} m exceeds "
                f"{threshold['grasp_translation_m']:.3f} m (reported, not gated here)"
            )
    print(f"[m4] actual-spout turns={turns:.6f}; radial progress={radial_progress:.6f} m; "
          f"landing={landing_success_rate:.3%} ({len(misses)} misses)")
    print(f"[m4] spout height={spout_min:.6f}..{spout_max:.6f} m; "
          f"edge margin={edge_margin:.6f} m")
    print(f"[m4] set-down error={final_position_error:.6f} m; "
          f"retreat motion={retreat_motion:.6f} m")
    print(f"[m4] stream span<={maximum_horizontal_span:.6f} m; "
          f"launch speed<={maximum_horizontal_launch_speed:.6f} m/s; "
          f"max landing radius={maximum_landing_radius:.6f} m; "
          f"obstructed paths={summary['stream']['obstructed_path_count']}")
    print(f"[m4] min clearance to brewer: pitcher="
          f"{summary['motion']['minimum_pitcher_brewer_clearance_all_phases_m']:.6f} m, "
          f"robot={minimum_robot_brewer_clearance:.6f} m "
          f"({nearest_robot_row['robot_brewer_nearest_pair']}); "
          f"collision geometry intact={collision_integrity['passes']}")
    print(f"[m4] payload={payload['design_total_kg']:.3f} kg; "
          f"torque pass={torque_pass}; overall pass={summary['passes']}")
    carried_clearance = summary["motion"]["pitcher_robot_clearance"]["by_scope"]["carried"]
    print(f"[m4] min carried pitcher-robot clearance="
          f"{carried_clearance['minimum_m'] * 1000:.2f} mm "
          f"({carried_clearance['nearest_pair']} at {carried_clearance['time_s']:.3f} s); "
          f"carried grasp translation="
          f"{summary['motion']['maximum_carried_grasp_translation_m'] * 1000:.3f} mm")
    print(f"[m4] qpos sha256={state_sha256}")
    for path in (torque_csv, motion_csv, stream_csv, summary_path):
        print(f"[m4] artifact: {path}")
    if args.save_render_states:
        print(f"[m4] artifact: {render_states_path}")
    if render_enabled:
        for path in (video, start_frame, mid_spiral_frame, spiral_frame, set_down_frame):
            print(f"[m4] artifact: {path}")
    checked_pass = (
        summary["reachability"]["passes"]
        if args.reachability_check
        else summary["passes"]
    )
    return 0 if checked_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
